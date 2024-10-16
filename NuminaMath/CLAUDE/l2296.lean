import Mathlib

namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2296_229610

theorem integer_solutions_of_inequalities :
  let S := { x : ℤ | (4 * (1 + x) : ℚ) / 3 - 1 ≤ (5 + x : ℚ) / 2 ∧
                     (x : ℚ) - 5 ≤ (3 / 2) * ((3 * x : ℚ) - 2) }
  S = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2296_229610


namespace NUMINAMATH_CALUDE_median_to_hypotenuse_length_l2296_229638

theorem median_to_hypotenuse_length (a b c m : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → m = c / 2 → m = 2.5 := by
sorry

end NUMINAMATH_CALUDE_median_to_hypotenuse_length_l2296_229638


namespace NUMINAMATH_CALUDE_double_sides_same_perimeter_l2296_229682

/-- A regular polygon with n sides and side length s -/
structure RegularPolygon where
  n : ℕ
  s : ℝ
  h_n : n ≥ 3

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ := p.n * p.s

theorem double_sides_same_perimeter (p : RegularPolygon) :
  ∃ (q : RegularPolygon), q.n = 2 * p.n ∧ perimeter q = perimeter p ∧ q.s = p.s / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_sides_same_perimeter_l2296_229682


namespace NUMINAMATH_CALUDE_speeding_fine_lawyer_hours_mark_speeding_fine_l2296_229681

theorem speeding_fine_lawyer_hours 
  (base_fine : ℕ) 
  (fine_increase_per_mph : ℕ) 
  (actual_speed : ℕ) 
  (speed_limit : ℕ) 
  (court_costs : ℕ) 
  (lawyer_hourly_rate : ℕ) 
  (total_owed : ℕ) : ℕ :=
  let speed_over_limit := actual_speed - speed_limit
  let speed_penalty := speed_over_limit * fine_increase_per_mph
  let initial_fine := base_fine + speed_penalty
  let doubled_fine := initial_fine * 2
  let fine_with_court_costs := doubled_fine + court_costs
  let lawyer_fees := total_owed - fine_with_court_costs
  lawyer_fees / lawyer_hourly_rate

theorem mark_speeding_fine 
  (h1 : speeding_fine_lawyer_hours 50 2 75 30 300 80 820 = 3) : 
  speeding_fine_lawyer_hours 50 2 75 30 300 80 820 = 3 := by
  sorry

end NUMINAMATH_CALUDE_speeding_fine_lawyer_hours_mark_speeding_fine_l2296_229681


namespace NUMINAMATH_CALUDE_base7_to_base10_65432_l2296_229627

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number --/
def base7Number : List Nat := [2, 3, 4, 5, 6]

/-- Theorem stating that the base-10 equivalent of 65432 in base-7 is 16340 --/
theorem base7_to_base10_65432 :
  base7ToBase10 base7Number = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_65432_l2296_229627


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2296_229617

/-- Given a quadratic equation 6x^2 + 5x + 7, prove that the sum of the reciprocals of its roots is -5/7 -/
theorem sum_of_reciprocals_of_roots (x : ℝ) (γ δ : ℝ) :
  (6 * x^2 + 5 * x + 7 = 0) →
  (∃ p q : ℝ, 6 * p^2 + 5 * p + 7 = 0 ∧ 6 * q^2 + 5 * q + 7 = 0 ∧ γ = 1/p ∧ δ = 1/q) →
  γ + δ = -5/7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2296_229617


namespace NUMINAMATH_CALUDE_equilateral_triangle_roots_l2296_229636

/-- Given complex roots z₁ and z₂ of z² + az + b = 0 where a and b are complex,
    and z₂ = ω z₁ with ω = e^(2πi/3), prove that a²/b = 1 -/
theorem equilateral_triangle_roots (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 →
  z₂^2 + a*z₂ + b = 0 →
  z₂ = (Complex.exp (2 * Complex.I * Real.pi / 3)) * z₁ →
  a^2 / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_roots_l2296_229636


namespace NUMINAMATH_CALUDE_city_wall_length_l2296_229640

/-- Represents a city layout with 5 congruent squares in an isosceles cross shape -/
structure CityLayout where
  square_side : ℝ
  num_squares : Nat
  num_squares_eq : num_squares = 5

/-- Calculates the perimeter of the city layout -/
def perimeter (city : CityLayout) : ℝ :=
  12 * city.square_side

/-- Calculates the area of the city layout -/
def area (city : CityLayout) : ℝ :=
  city.num_squares * city.square_side^2

/-- Theorem stating that if the perimeter equals the area, then the perimeter is 28.8 km -/
theorem city_wall_length (city : CityLayout) :
  perimeter city = area city → perimeter city = 28.8 := by
  sorry


end NUMINAMATH_CALUDE_city_wall_length_l2296_229640


namespace NUMINAMATH_CALUDE_sum_of_roots_satisfies_equation_l2296_229649

-- Define the polynomial
def polynomial (a b c x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

-- Define the equation for the sum of two roots
def sum_of_roots_equation (a b c u : ℝ) : ℝ := u^6 + 2*a*u^4 + (a^2 - 4*c)*u^2 - b^2

-- Theorem statement
theorem sum_of_roots_satisfies_equation (a b c : ℝ) :
  ∃ (x₁ x₂ : ℝ), polynomial a b c x₁ = 0 ∧ polynomial a b c x₂ = 0 ∧
  (∃ (u : ℝ), u = x₁ + x₂ ∧ sum_of_roots_equation a b c u = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_satisfies_equation_l2296_229649


namespace NUMINAMATH_CALUDE_number_difference_l2296_229693

theorem number_difference (a b : ℕ) : 
  a + b = 34800 → 
  b % 25 = 0 → 
  b = 25 * a → 
  b - a = 32112 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2296_229693


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_28_l2296_229651

theorem consecutive_integers_around_sqrt_28 (a b : ℤ) : 
  (b = a + 1) → (↑a < Real.sqrt 28 ∧ Real.sqrt 28 < ↑b) → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_28_l2296_229651


namespace NUMINAMATH_CALUDE_negation_of_implication_l2296_229607

theorem negation_of_implication (a b : ℝ) : 
  ¬(ab = 2 → a^2 + b^2 ≥ 4) ↔ (ab ≠ 2 → a^2 + b^2 < 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2296_229607


namespace NUMINAMATH_CALUDE_number_of_trailing_zeros_l2296_229665

theorem number_of_trailing_zeros : ∃ n : ℕ, (10^100 * 100^10 : ℕ) = n * 10^120 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_number_of_trailing_zeros_l2296_229665


namespace NUMINAMATH_CALUDE_equation_solution_l2296_229648

theorem equation_solution :
  ∃ x : ℚ, x ≠ 2 ∧ (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2296_229648


namespace NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l2296_229645

def base_expansion (n : ℕ) (b : ℕ) : List ℕ :=
  sorry

def sum_digits (digits : List ℕ) : ℕ :=
  sorry

def is_largest_base (b : ℕ) : Prop :=
  (∀ k > b, sum_digits (base_expansion ((k + 2)^4) k) = 32) ∧
  sum_digits (base_expansion ((b + 2)^4) b) ≠ 32

theorem largest_base_for_12_4th_power : is_largest_base 7 :=
  sorry

end NUMINAMATH_CALUDE_largest_base_for_12_4th_power_l2296_229645


namespace NUMINAMATH_CALUDE_abs_sum_gt_abs_prod_plus_one_implies_prod_zero_l2296_229660

theorem abs_sum_gt_abs_prod_plus_one_implies_prod_zero (a b : ℤ) : 
  |a + b| > |1 + a * b| → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_gt_abs_prod_plus_one_implies_prod_zero_l2296_229660


namespace NUMINAMATH_CALUDE_white_most_likely_probabilities_game_is_fair_l2296_229694

/-- Represents the colors of ping-pong balls in the box -/
inductive Color
  | White
  | Yellow
  | Red

/-- The total number of balls in the box -/
def totalBalls : ℕ := 6

/-- The number of balls of each color -/
def numBalls (c : Color) : ℕ :=
  match c with
  | Color.White => 3
  | Color.Yellow => 2
  | Color.Red => 1

/-- The probability of picking a ball of a given color -/
def prob (c : Color) : ℚ :=
  (numBalls c : ℚ) / totalBalls

/-- Theorem stating that white is the most likely color to be picked -/
theorem white_most_likely :
  ∀ c : Color, c ≠ Color.White → prob Color.White > prob c := by sorry

/-- Theorem stating the probabilities for each color -/
theorem probabilities :
  prob Color.White = 1/2 ∧ prob Color.Yellow = 1/3 ∧ prob Color.Red = 1/6 := by sorry

/-- Theorem stating that the game is fair -/
theorem game_is_fair :
  prob Color.White = 1 - prob Color.White := by sorry

end NUMINAMATH_CALUDE_white_most_likely_probabilities_game_is_fair_l2296_229694


namespace NUMINAMATH_CALUDE_water_gun_game_theorem_l2296_229691

/-- Represents a student with a position -/
structure Student where
  position : ℝ × ℝ

/-- The environment of the water gun game -/
structure WaterGunGame where
  n : ℕ
  students : Fin (2*n+1) → Student
  distinct_distances : ∀ i j k l, i ≠ j → k ≠ l → 
    (students i).position ≠ (students j).position → 
    (students k).position ≠ (students l).position →
    dist (students i).position (students j).position ≠ 
    dist (students k).position (students l).position

/-- A student squirts another student -/
def squirts (game : WaterGunGame) (i j : Fin (2*game.n+1)) : Prop :=
  ∀ k, k ≠ j → 
    dist (game.students i).position (game.students j).position < 
    dist (game.students i).position (game.students k).position

theorem water_gun_game_theorem (game : WaterGunGame) : 
  (∃ i j, i ≠ j ∧ squirts game i j ∧ squirts game j i) ∧ 
  (∃ i, ∀ j, ¬squirts game j i) :=
sorry

end NUMINAMATH_CALUDE_water_gun_game_theorem_l2296_229691


namespace NUMINAMATH_CALUDE_edward_summer_earnings_l2296_229689

/-- Edward's lawn mowing business earnings --/
def lawn_mowing_problem (spring_earnings summer_earnings supplies_cost final_amount : ℕ) : Prop :=
  spring_earnings + summer_earnings = supplies_cost + final_amount

theorem edward_summer_earnings :
  ∃ (summer_earnings : ℕ),
    lawn_mowing_problem 2 summer_earnings 5 24 ∧ summer_earnings = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_edward_summer_earnings_l2296_229689


namespace NUMINAMATH_CALUDE_triangle_property_l2296_229699

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.c / (t.b - t.a) = (sin t.A + sin t.B) / (sin t.A + sin t.C))
  (h2 : t.b = 2 * sqrt 2)
  (h3 : t.a + t.c = 3) :
  t.B = 2 * π / 3 ∧ 
  (1/2) * t.a * t.c * sin t.B = sqrt 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l2296_229699


namespace NUMINAMATH_CALUDE_triangle_theorem_l2296_229604

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : (2 * Real.sin t.C - Real.sin t.B) / Real.sin t.B = (t.a * Real.cos t.B) / (t.b * Real.cos t.A))
  (h2 : t.a = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.B) :
  t.A = π/3 ∧ t.b = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2296_229604


namespace NUMINAMATH_CALUDE_plaza_design_properties_l2296_229602

/-- Represents the plaza design and cost structure -/
structure PlazaDesign where
  sideLength : ℝ
  lightTileCost : ℝ
  darkTileCost : ℝ
  borderWidth : ℝ

/-- Calculates the total cost of materials for the plaza design -/
def totalCost (design : PlazaDesign) : ℝ :=
  sorry

/-- Calculates the side length of the central light square -/
def centralSquareSideLength (design : PlazaDesign) : ℝ :=
  sorry

/-- Theorem stating the properties of the plaza design -/
theorem plaza_design_properties (design : PlazaDesign) 
  (h1 : design.sideLength = 20)
  (h2 : design.lightTileCost = 100000)
  (h3 : design.darkTileCost = 300000)
  (h4 : design.borderWidth = 2)
  (h5 : totalCost design = 2 * (design.darkTileCost / 4)) :
  totalCost design = 150000 ∧ centralSquareSideLength design = 10.5 :=
sorry

end NUMINAMATH_CALUDE_plaza_design_properties_l2296_229602


namespace NUMINAMATH_CALUDE_three_times_relation_l2296_229690

/-- Given four numbers M₁, M₂, M₃, and M₄, prove that M₄ = 3M₂ -/
theorem three_times_relation (M₁ M₂ M₃ M₄ : ℝ) 
  (hM₁ : M₁ = 2.02e-6)
  (hM₂ : M₂ = 0.0000202)
  (hM₃ : M₃ = 0.00000202)
  (hM₄ : M₄ = 6.06e-5) :
  M₄ = 3 * M₂ := by
  sorry

end NUMINAMATH_CALUDE_three_times_relation_l2296_229690


namespace NUMINAMATH_CALUDE_not_equivalent_fraction_l2296_229622

theorem not_equivalent_fraction : (1 : ℚ) / 20000000 ≠ (48 : ℚ) / 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_not_equivalent_fraction_l2296_229622


namespace NUMINAMATH_CALUDE_store_discount_proof_l2296_229686

/-- Calculates the actual discount percentage given the initial discount and VIP discount -/
def actual_discount (initial_discount : ℝ) (vip_discount : ℝ) : ℝ :=
  1 - (1 - initial_discount) * (1 - vip_discount)

/-- Proves that the actual discount is 28% given a 20% initial discount and 10% VIP discount -/
theorem store_discount_proof :
  actual_discount 0.2 0.1 = 0.28 := by
  sorry

#eval actual_discount 0.2 0.1

end NUMINAMATH_CALUDE_store_discount_proof_l2296_229686


namespace NUMINAMATH_CALUDE_root_sum_equality_l2296_229623

-- Define the polynomial f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x

-- Define the theorem
theorem root_sum_equality 
  (a b c : ℝ) 
  (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ) : 
  (f a b c x₁ = 1) ∧ (f a b c x₂ = 1) ∧ (f a b c x₃ = 1) ∧ (f a b c x₄ = 1) →
  (f a b c y₁ = 2) ∧ (f a b c y₂ = 2) ∧ (f a b c y₃ = 2) ∧ (f a b c y₄ = 2) →
  (x₁ + x₂ = x₃ + x₄) →
  (y₁ + y₂ = y₃ + y₄) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_equality_l2296_229623


namespace NUMINAMATH_CALUDE_empty_seats_calculation_l2296_229620

/-- Calculates the number of empty seats in a theater -/
def empty_seats (total_seats people_watching : ℕ) : ℕ :=
  total_seats - people_watching

/-- Theorem: The number of empty seats is the difference between total seats and people watching -/
theorem empty_seats_calculation (total_seats people_watching : ℕ) 
  (h1 : total_seats ≥ people_watching) :
  empty_seats total_seats people_watching = total_seats - people_watching :=
by sorry

end NUMINAMATH_CALUDE_empty_seats_calculation_l2296_229620


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l2296_229673

/-- An arithmetic sequence of 8 terms with positive values and non-zero common difference -/
structure ArithmeticSequence8 where
  a : Fin 8 → ℝ
  positive : ∀ i, a i > 0
  is_arithmetic : ∃ d ≠ 0, ∀ i j, a j - a i = (j - i : ℝ) * d

theorem arithmetic_sequence_product_inequality (seq : ArithmeticSequence8) :
  seq.a 0 * seq.a 7 < seq.a 3 * seq.a 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l2296_229673


namespace NUMINAMATH_CALUDE_sum_a_equals_1649_l2296_229653

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 20 = 0 then 15
  else if n % 20 = 0 ∧ n % 18 = 0 then 20
  else if n % 18 = 0 ∧ n % 15 = 0 then 18
  else 0

theorem sum_a_equals_1649 :
  (Finset.range 2999).sum a = 1649 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_equals_1649_l2296_229653


namespace NUMINAMATH_CALUDE_parallel_vectors_m_values_l2296_229626

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The theorem statement -/
theorem parallel_vectors_m_values (m : ℝ) :
  parallel (2, m) (m, 2) → m = -2 ∨ m = 2 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_m_values_l2296_229626


namespace NUMINAMATH_CALUDE_coffee_cost_per_ounce_l2296_229675

/-- The cost of coffee per ounce, given the household consumption and weekly spending -/
theorem coffee_cost_per_ounce 
  (people : ℕ)
  (cups_per_person : ℕ)
  (ounces_per_cup : ℚ)
  (weekly_spending : ℚ) :
  people = 4 →
  cups_per_person = 2 →
  ounces_per_cup = 1/2 →
  weekly_spending = 35 →
  (weekly_spending / (people * cups_per_person * 7 * ounces_per_cup) : ℚ) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cost_per_ounce_l2296_229675


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2296_229615

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, r₁ + r₂ = -p ∧ r₁ * r₂ = m ∧
    r₁ / 2 + r₂ / 2 = -m ∧ (r₁ / 2) * (r₂ / 2) = n) →
  n / p = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2296_229615


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l2296_229647

/-- Given distinct non-zero real numbers x, y, and z, and a real number r,
    if x^2(y-z), y^2(z-x), and z^2(x-y) form a geometric progression with common ratio r,
    then r satisfies the equation r^2 + r + 1 = 0 -/
theorem geometric_progression_ratio_equation (x y z r : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (hprogression : ∃ (a : ℝ), a ≠ 0 ∧ 
    x^2 * (y - z) = a ∧ 
    y^2 * (z - x) = a * r ∧ 
    z^2 * (x - y) = a * r^2) :
  r^2 + r + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l2296_229647


namespace NUMINAMATH_CALUDE_black_beads_fraction_l2296_229632

/-- Proves that the fraction of black beads pulled out is 1/6 given the initial conditions -/
theorem black_beads_fraction (total_white : ℕ) (total_black : ℕ) (total_pulled : ℕ) :
  total_white = 51 →
  total_black = 90 →
  total_pulled = 32 →
  (total_pulled - (total_white / 3)) / total_black = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_black_beads_fraction_l2296_229632


namespace NUMINAMATH_CALUDE_thabo_hardcover_count_l2296_229611

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabos_books : BookCollection where
  hardcover_nonfiction := 25
  paperback_nonfiction := 45
  paperback_fiction := 90

theorem thabo_hardcover_count :
  ∀ (books : BookCollection),
    books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction = 160 →
    books.paperback_nonfiction = books.hardcover_nonfiction + 20 →
    books.paperback_fiction = 2 * books.paperback_nonfiction →
    books.hardcover_nonfiction = 25 := by
  sorry

#eval thabos_books.hardcover_nonfiction

end NUMINAMATH_CALUDE_thabo_hardcover_count_l2296_229611


namespace NUMINAMATH_CALUDE_tan_150_degrees_l2296_229683

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l2296_229683


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l2296_229630

theorem bobby_candy_consumption (initial : ℕ) (additional : ℕ) : 
  initial = 26 → additional = 17 → initial + additional = 43 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l2296_229630


namespace NUMINAMATH_CALUDE_count_ak_divisible_by_9_l2296_229659

/-- The number obtained by writing the integers 1 to n from left to right -/
def a (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The count of a_k divisible by 9 for 1 ≤ k ≤ 100 -/
def countDivisibleBy9 : ℕ := sorry

theorem count_ak_divisible_by_9 : countDivisibleBy9 = 22 := by sorry

end NUMINAMATH_CALUDE_count_ak_divisible_by_9_l2296_229659


namespace NUMINAMATH_CALUDE_unique_positive_integers_sum_l2296_229606

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 73) / 2 + 5 / 2)

theorem unique_positive_integers_sum (a b c : ℕ+) :
  x^80 = 3*x^78 + 18*x^74 + 15*x^72 - x^40 + (a : ℝ)*x^36 + (b : ℝ)*x^34 + (c : ℝ)*x^30 →
  a + b + c = 265 := by sorry

end NUMINAMATH_CALUDE_unique_positive_integers_sum_l2296_229606


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2296_229601

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- Number of participants excluding the 12 lowest-scoring players
  total_points : ℕ → ℕ → ℚ  -- Function to calculate total points between two groups of players
  lowest_twelve_points : ℚ  -- Points earned by the 12 lowest-scoring players among themselves

/-- The theorem stating the total number of participants in the tournament -/
theorem chess_tournament_participants (t : ChessTournament) : 
  (t.n + 12 = 24) ∧ 
  (t.total_points t.n 12 = t.total_points t.n t.n / 2) ∧
  (t.lowest_twelve_points = 66) ∧
  (t.total_points (t.n + 12) (t.n + 12) / 2 = t.total_points t.n t.n + 2 * t.lowest_twelve_points) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2296_229601


namespace NUMINAMATH_CALUDE_trig_identity_l2296_229641

theorem trig_identity (α : ℝ) : 
  Real.sin α ^ 2 + Real.cos (π/6 - α) ^ 2 - Real.sin α * Real.cos (π/6 - α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2296_229641


namespace NUMINAMATH_CALUDE_joan_balloons_l2296_229668

theorem joan_balloons (initial : ℕ) : initial + 2 = 10 → initial = 8 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l2296_229668


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2296_229600

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  -10 * x * y^2 + y^3 + 25 * x^2 * y = y * (5 * x - y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (a b : ℝ) :
  a^3 + a^2 * b - a * b^2 - b^3 = (a + b)^2 * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l2296_229600


namespace NUMINAMATH_CALUDE_product_equals_zero_l2296_229613

theorem product_equals_zero (a : ℤ) (h : a = -1) : (a - 3) * (a - 2) * (a - 1) * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2296_229613


namespace NUMINAMATH_CALUDE_system_solution_l2296_229603

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3*x + Real.sqrt (3*x - y) + y = 6
def equation2 (x y : ℝ) : Prop := 9*x^2 + 3*x - y - y^2 = 36

-- Define the solution set
def solutions : Set (ℝ × ℝ) := {(2, -3), (6, -18)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2296_229603


namespace NUMINAMATH_CALUDE_zoo_animals_l2296_229684

/-- The number of sea horses at the zoo -/
def num_sea_horses : ℕ := 70

/-- The number of penguins at the zoo -/
def num_penguins : ℕ := num_sea_horses + 85

/-- The ratio of sea horses to penguins is 5:11 -/
axiom ratio_constraint : (num_sea_horses : ℚ) / num_penguins = 5 / 11

theorem zoo_animals : num_sea_horses = 70 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_l2296_229684


namespace NUMINAMATH_CALUDE_savings_amount_l2296_229635

/-- Represents the price of a single book -/
def book_price : ℝ := 45

/-- Represents the discount percentage for Promotion A -/
def promotion_a_discount : ℝ := 0.4

/-- Represents the fixed discount amount for Promotion B -/
def promotion_b_discount : ℝ := 15

/-- Represents the local tax rate -/
def tax_rate : ℝ := 0.08

/-- Calculates the total cost for Promotion A including tax -/
def total_cost_a : ℝ :=
  (book_price + book_price * (1 - promotion_a_discount)) * (1 + tax_rate)

/-- Calculates the total cost for Promotion B including tax -/
def total_cost_b : ℝ :=
  (book_price + (book_price - promotion_b_discount)) * (1 + tax_rate)

/-- Theorem stating the savings amount by choosing Promotion A over Promotion B -/
theorem savings_amount : 
  total_cost_b - total_cost_a = 3.24 := by sorry

end NUMINAMATH_CALUDE_savings_amount_l2296_229635


namespace NUMINAMATH_CALUDE_transformation_interval_l2296_229678

theorem transformation_interval (x : ℝ) :
  x ∈ Set.Icc 0 1 → (8 * x - 2) ∈ Set.Icc (-2) 6 := by
  sorry

end NUMINAMATH_CALUDE_transformation_interval_l2296_229678


namespace NUMINAMATH_CALUDE_first_square_length_is_correct_l2296_229695

/-- The length of the first square of fabric -/
def first_square_length : ℝ := 8

/-- The height of the first square of fabric -/
def first_square_height : ℝ := 5

/-- The length of the second square of fabric -/
def second_square_length : ℝ := 10

/-- The height of the second square of fabric -/
def second_square_height : ℝ := 7

/-- The length of the third square of fabric -/
def third_square_length : ℝ := 5

/-- The height of the third square of fabric -/
def third_square_height : ℝ := 5

/-- The desired length of the flag -/
def flag_length : ℝ := 15

/-- The desired height of the flag -/
def flag_height : ℝ := 9

theorem first_square_length_is_correct : 
  first_square_length * first_square_height + 
  second_square_length * second_square_height + 
  third_square_length * third_square_height = 
  flag_length * flag_height := by
  sorry

end NUMINAMATH_CALUDE_first_square_length_is_correct_l2296_229695


namespace NUMINAMATH_CALUDE_transform_OAB_l2296_229670

/-- Transformation from xy-plane to uv-plane -/
def transform (x y : ℝ) : ℝ × ℝ := (x^2 - y^2, x * y)

/-- Triangle OAB in xy-plane -/
def triangle_OAB : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ x ∧ p = (x, y)}

/-- Image of triangle OAB in uv-plane -/
def image_OAB : Set (ℝ × ℝ) :=
  {q | ∃ p ∈ triangle_OAB, q = transform p.1 p.2}

theorem transform_OAB :
  (0, 0) ∈ image_OAB ∧ (1, 0) ∈ image_OAB ∧ (0, 1) ∈ image_OAB :=
sorry

end NUMINAMATH_CALUDE_transform_OAB_l2296_229670


namespace NUMINAMATH_CALUDE_binomial_2024_1_l2296_229629

theorem binomial_2024_1 : Nat.choose 2024 1 = 2024 := by
  sorry

end NUMINAMATH_CALUDE_binomial_2024_1_l2296_229629


namespace NUMINAMATH_CALUDE_spring_spending_is_1_7_l2296_229644

/-- The spending of Rivertown government in millions of dollars -/
structure RivertownSpending where
  /-- Total accumulated spending by the end of February -/
  february_end : ℝ
  /-- Total accumulated spending by the end of May -/
  may_end : ℝ

/-- The spending during March, April, and May -/
def spring_spending (s : RivertownSpending) : ℝ :=
  s.may_end - s.february_end

theorem spring_spending_is_1_7 (s : RivertownSpending) 
  (h_feb : s.february_end = 0.8)
  (h_may : s.may_end = 2.5) : 
  spring_spending s = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_spring_spending_is_1_7_l2296_229644


namespace NUMINAMATH_CALUDE_angle_in_square_l2296_229605

/-- In a square ABCD with a segment CE, if CE forms angles of 7α and 8α with the sides of the square, then α = 9°. -/
theorem angle_in_square (α : ℝ) : 
  (7 * α + 8 * α + 45 = 180) → α = 9 := by sorry

end NUMINAMATH_CALUDE_angle_in_square_l2296_229605


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l2296_229674

def initial_money : ℕ := 56
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def notebook_cost : ℕ := 4
def book_cost : ℕ := 7

theorem money_left_after_purchase : 
  initial_money - (notebooks_bought * notebook_cost + books_bought * book_cost) = 14 :=
by sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l2296_229674


namespace NUMINAMATH_CALUDE_function_inequality_condition_l2296_229677

theorem function_inequality_condition (f : ℝ → ℝ) (a c : ℝ) :
  (∀ x, f x = 2 * x + 3) →
  a > 0 →
  c > 0 →
  (∀ x, |x + 5| < c → |f x + 5| < a) ↔
  c > a / 2 := by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l2296_229677


namespace NUMINAMATH_CALUDE_no_prime_sum_53_less_than_30_l2296_229656

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primeSum53LessThan30 : Prop :=
  ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 ∧ (p < 30 ∨ q < 30)

theorem no_prime_sum_53_less_than_30 : primeSum53LessThan30 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_less_than_30_l2296_229656


namespace NUMINAMATH_CALUDE_average_monthly_sales_l2296_229639

def may_sales : ℝ := 150
def june_sales : ℝ := 75
def july_sales : ℝ := 50
def august_sales : ℝ := 175

def total_months : ℕ := 4

def total_sales : ℝ := may_sales + june_sales + july_sales + august_sales

theorem average_monthly_sales : 
  total_sales / total_months = 112.5 := by sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l2296_229639


namespace NUMINAMATH_CALUDE_cubic_expression_zero_l2296_229669

theorem cubic_expression_zero (x : ℝ) (h : x^2 + 3*x - 3 = 0) : 
  x^3 + 2*x^2 - 6*x + 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_expression_zero_l2296_229669


namespace NUMINAMATH_CALUDE_kim_shoe_pairs_l2296_229608

/-- The number of shoes Kim has -/
def total_shoes : ℕ := 18

/-- The probability of selecting two shoes of the same color -/
def probability : ℚ := 58823529411764705 / 1000000000000000000

/-- The number of pairs of shoes Kim has -/
def num_pairs : ℕ := total_shoes / 2

theorem kim_shoe_pairs :
  (probability = 1 / (total_shoes - 1)) → num_pairs = 9 := by
  sorry

end NUMINAMATH_CALUDE_kim_shoe_pairs_l2296_229608


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l2296_229650

/-- The value of m for which the circle x^2 + y^2 = 4m is tangent to the line x + y = 2√m -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4*m ∧ x + y = 2*Real.sqrt m → 
    (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 4*m ∧ p.1 + p.2 = 2*Real.sqrt m)) → 
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l2296_229650


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2296_229628

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  (∀ k < m, ¬ ∃ (i : ℕ) (s : ℝ), k = (i + s : ℝ)^3 ∧ 0 < s ∧ s < 1/2000) →
  m = (n + r : ℝ)^3 →
  n > 0 →
  0 < r →
  r < 1/2000 →
  n = 26 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l2296_229628


namespace NUMINAMATH_CALUDE_characterize_f_l2296_229666

def is_valid_f (f : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, f (n + 1) ≥ f n) ∧
  (∀ m n : ℕ+, Nat.gcd m.val n.val = 1 → f (m * n) = f m * f n)

theorem characterize_f (f : ℕ+ → ℝ) (hf : is_valid_f f) :
  (∃ a : ℝ, a ≥ 0 ∧ ∀ n : ℕ+, f n = (n : ℝ) ^ a) ∨ (∀ n : ℕ+, f n = 0) :=
sorry

end NUMINAMATH_CALUDE_characterize_f_l2296_229666


namespace NUMINAMATH_CALUDE_frequency_table_purpose_l2296_229679

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable where
  /-- The table analyzes sample data -/
  analyzes_sample_data : Bool
  /-- The table groups data into categories -/
  groups_data : Bool

/-- The purpose of creating a frequency distribution table -/
def purpose_of_frequency_table (table : FrequencyDistributionTable) : Prop :=
  table.analyzes_sample_data ∧ 
  table.groups_data → 
  (∃ (proportion_understanding : Prop) (population_estimation : Prop),
    proportion_understanding ∧ population_estimation)

/-- Theorem stating the purpose of creating a frequency distribution table -/
theorem frequency_table_purpose (table : FrequencyDistributionTable) : 
  purpose_of_frequency_table table :=
sorry

end NUMINAMATH_CALUDE_frequency_table_purpose_l2296_229679


namespace NUMINAMATH_CALUDE_problem_solution_l2296_229618

theorem problem_solution (p q r : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_equation : p / (q - r) + q / (r - p) + r / (p - q) = 1) : 
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2296_229618


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_3d_l2296_229657

theorem cauchy_schwarz_inequality_3d (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  (x₁ * x₂ + y₁ * y₂ + z₁ * z₂)^2 ≤ (x₁^2 + y₁^2 + z₁^2) * (x₂^2 + y₂^2 + z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_3d_l2296_229657


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2296_229696

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + a*x₁ - 2 = 0) → 
  (x₂^2 + a*x₂ - 2 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^3 + 22/x₂ = x₂^3 + 22/x₁) →
  (a = 3 ∨ a = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2296_229696


namespace NUMINAMATH_CALUDE_not_right_triangle_1_5_2_3_l2296_229631

/-- A function that checks if three numbers can form the sides of a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- Theorem stating that 1.5, 2, and 3 cannot form the sides of a right triangle -/
theorem not_right_triangle_1_5_2_3 : ¬ is_right_triangle 1.5 2 3 := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_1_5_2_3_l2296_229631


namespace NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l2296_229692

theorem solution_set_absolute_value_inequality :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_absolute_value_inequality_l2296_229692


namespace NUMINAMATH_CALUDE_hiker_distance_l2296_229655

theorem hiker_distance (hours_day1 : ℝ) : 
  hours_day1 > 0 →
  3 * hours_day1 + 4 * (hours_day1 - 1) + 4 * hours_day1 = 62 →
  3 * hours_day1 = 18 :=
by sorry

end NUMINAMATH_CALUDE_hiker_distance_l2296_229655


namespace NUMINAMATH_CALUDE_total_toes_on_bus_l2296_229654

/-- Represents a race of beings on Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toes_per_hand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of students of each race on the bus -/
def students (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes for a single being of a given race -/
def toes_per_being (r : Race) : ℕ :=
  hands r * toes_per_hand r

/-- Total number of toes for all students of a given race on the bus -/
def total_toes_per_race (r : Race) : ℕ :=
  students r * toes_per_being r

/-- Theorem: The total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus :
  total_toes_per_race Race.Hoopit + total_toes_per_race Race.Neglart = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_toes_on_bus_l2296_229654


namespace NUMINAMATH_CALUDE_bug_probability_7_l2296_229614

/-- Probability of a bug being at the starting vertex of a regular tetrahedron after n steps -/
def bug_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | m + 1 => (1 / 3) * (1 - bug_probability m)

/-- The probability of the bug being at the starting vertex after 7 steps is 182/729 -/
theorem bug_probability_7 : bug_probability 7 = 182 / 729 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_7_l2296_229614


namespace NUMINAMATH_CALUDE_square_area_is_26_l2296_229667

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square of the distance between two points -/
def squaredDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The area of a square given its four vertices -/
def squareArea (p q r s : Point) : ℝ :=
  squaredDistance p q

theorem square_area_is_26 : 
  let p : Point := ⟨1, 2⟩
  let q : Point := ⟨-4, 3⟩
  let r : Point := ⟨-3, -2⟩
  let s : Point := ⟨2, -3⟩
  squareArea p q r s = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_26_l2296_229667


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2296_229652

theorem rectangular_to_polar_conversion :
  ∀ (x y r θ : ℝ),
  x = 8 ∧ y = -8 * Real.sqrt 3 ∧
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = Real.sqrt (x^2 + y^2) ∧
  θ = 2 * Real.pi - Real.pi / 3 →
  r = 16 ∧ θ = 5 * Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2296_229652


namespace NUMINAMATH_CALUDE_water_consumption_in_five_hours_l2296_229664

/-- The number of glasses of water consumed in a given time period. -/
def glasses_consumed (rate : ℚ) (time : ℚ) : ℚ :=
  time / rate

/-- Theorem stating that drinking a glass of water every 20 minutes for 5 hours results in 15 glasses. -/
theorem water_consumption_in_five_hours :
  glasses_consumed (20 : ℚ) (5 * 60 : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_in_five_hours_l2296_229664


namespace NUMINAMATH_CALUDE_apple_price_theorem_l2296_229662

/-- The price of apples with a two-tier pricing system -/
theorem apple_price_theorem 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 360) 
  (h2 : 30 * l + 6 * q = 420) : 
  25 * l = 250 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_theorem_l2296_229662


namespace NUMINAMATH_CALUDE_seating_arrangements_l2296_229687

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Given a bench with 9 seats and 3 people to be seated with at least 2 empty seats between any two people,
    the number of different seating arrangements is 60. -/
theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 9) (h2 : people = 3) :
  A people people * (C 4 2 + C 4 1) = 60 := by
  sorry


end NUMINAMATH_CALUDE_seating_arrangements_l2296_229687


namespace NUMINAMATH_CALUDE_square_of_binomial_l2296_229676

/-- If ax^2 + 18x + 16 is the square of a binomial, then a = 81/16 -/
theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2296_229676


namespace NUMINAMATH_CALUDE_selection_methods_count_l2296_229658

/-- The number of different ways to select one teacher and one student -/
def selection_methods (num_teachers : ℕ) (num_male_students : ℕ) (num_female_students : ℕ) : ℕ :=
  num_teachers * (num_male_students + num_female_students)

/-- Theorem stating that the number of selection methods for the given problem is 39 -/
theorem selection_methods_count :
  selection_methods 3 8 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l2296_229658


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2296_229616

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  MonicQuarticPolynomial p →
  p 2 = 7 →
  p 3 = 12 →
  p 4 = 19 →
  p 5 = 28 →
  p 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2296_229616


namespace NUMINAMATH_CALUDE_inequality_solution_l2296_229609

theorem inequality_solution : 
  {x : ℝ | 5*x > 4*x + 2} = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2296_229609


namespace NUMINAMATH_CALUDE_bert_grocery_fraction_l2296_229663

def bert_spending (initial_amount : ℚ) (hardware_fraction : ℚ) (dry_cleaner_amount : ℚ) (final_amount : ℚ) : Prop :=
  let hardware_spent := initial_amount * hardware_fraction
  let after_hardware := initial_amount - hardware_spent
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_spent := after_dry_cleaner - final_amount
  grocery_spent / after_dry_cleaner = 1/2

theorem bert_grocery_fraction :
  bert_spending 44 (1/4) 9 12 :=
by
  sorry

end NUMINAMATH_CALUDE_bert_grocery_fraction_l2296_229663


namespace NUMINAMATH_CALUDE_power_multiplication_l2296_229625

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2296_229625


namespace NUMINAMATH_CALUDE_zoo_guide_problem_l2296_229633

/-- Given two zoo guides speaking to groups of children, where one guide spoke to 25 children
    and the total number of children spoken to is 44, prove that the number of children
    the other guide spoke to is 19. -/
theorem zoo_guide_problem (total_children : ℕ) (second_guide_children : ℕ) :
  total_children = 44 →
  second_guide_children = 25 →
  total_children - second_guide_children = 19 :=
by sorry

end NUMINAMATH_CALUDE_zoo_guide_problem_l2296_229633


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l2296_229624

theorem fraction_ratio_equality : 
  (15 / 8) / (2 / 5) = (3 / 8) / (1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l2296_229624


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2296_229643

theorem equation_solutions_count :
  let f : ℝ → ℝ := fun x => (x^2 - 7)^2 + 2*x^2 - 33
  ∃! (s : Finset ℝ), (∀ x ∈ s, f x = 0) ∧ Finset.card s = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2296_229643


namespace NUMINAMATH_CALUDE_shortest_distance_to_E_l2296_229698

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Calculate the distance between two points on the grid -/
def gridDistance (p1 p2 : GridPoint) : Nat :=
  (p2.x - p1.x) + (p2.y - p1.y)

theorem shortest_distance_to_E :
  let P : GridPoint := ⟨0, 0⟩
  let A : GridPoint := ⟨5, 4⟩
  let B : GridPoint := ⟨6, 2⟩
  let C : GridPoint := ⟨3, 3⟩
  let D : GridPoint := ⟨5, 1⟩
  let E : GridPoint := ⟨1, 4⟩
  (gridDistance P E ≤ gridDistance P A) ∧
  (gridDistance P E ≤ gridDistance P B) ∧
  (gridDistance P E ≤ gridDistance P C) ∧
  (gridDistance P E ≤ gridDistance P D) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_E_l2296_229698


namespace NUMINAMATH_CALUDE_true_masses_l2296_229697

/-- Represents the uneven lever scale with a linear relationship between left and right sides -/
structure UnevenLeverScale where
  k : ℝ
  b : ℝ
  left_to_right : ℝ → ℝ
  right_to_left : ℝ → ℝ
  hk_pos : k > 0
  hleft_to_right : left_to_right = fun x => k * x + b
  hright_to_left : right_to_left = fun y => (y - b) / k

/-- The equilibrium conditions observed on the uneven lever scale -/
structure EquilibriumConditions (scale : UnevenLeverScale) where
  melon_right : scale.left_to_right 3 = scale.right_to_left 5.5
  melon_left : scale.right_to_left 5.5 = scale.left_to_right 3
  watermelon_right : scale.left_to_right 5 = scale.right_to_left 10
  watermelon_left : scale.right_to_left 10 = scale.left_to_right 5

/-- The theorem stating the true masses of the melon and watermelon -/
theorem true_masses (scale : UnevenLeverScale) (conditions : EquilibriumConditions scale) :
  ∃ (melon_mass watermelon_mass : ℝ),
    melon_mass = 5.5 ∧
    watermelon_mass = 10 ∧
    scale.left_to_right 3 = melon_mass ∧
    scale.right_to_left 5.5 = melon_mass ∧
    scale.left_to_right 5 = watermelon_mass ∧
    scale.right_to_left 10 = watermelon_mass := by
  sorry

end NUMINAMATH_CALUDE_true_masses_l2296_229697


namespace NUMINAMATH_CALUDE_intersection_theorem_union_theorem_complement_union_theorem_l2296_229671

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 3 < x ∧ x < 10}

-- State the theorems to be proved
theorem intersection_theorem : A ∩ B = {x | 3 < x ∧ x < 7} := by sorry

theorem union_theorem : A ∪ B = {x | 2 ≤ x ∧ x < 10} := by sorry

theorem complement_union_theorem : (Set.univ \ A) ∪ (Set.univ \ B) = {x | x ≤ 3 ∨ x ≥ 7} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_union_theorem_complement_union_theorem_l2296_229671


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l2296_229637

theorem inequality_system_solutions (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), x₁ > 4 ∧ x₁ ≤ a ∧ 
                      x₂ > 4 ∧ x₂ ≤ a ∧ 
                      x₃ > 4 ∧ x₃ ≤ a ∧ 
                      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                      (∀ (y : ℤ), y > 4 ∧ y ≤ a → y = x₁ ∨ y = x₂ ∨ y = x₃)) →
  7 ≤ a ∧ a < 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l2296_229637


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2296_229661

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  ∀ (r h : ℝ) (lateral_area : ℝ),
    r = 3 →
    h = 4 →
    lateral_area = π * r * (Real.sqrt (r^2 + h^2)) →
    lateral_area = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2296_229661


namespace NUMINAMATH_CALUDE_vector_linear_combination_l2296_229619

/-- Given vectors a, b, and c in ℝ², prove that c is a linear combination of a and b. -/
theorem vector_linear_combination (a b c : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → c = (-1, 2) → c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l2296_229619


namespace NUMINAMATH_CALUDE_like_terms_exponents_l2296_229672

theorem like_terms_exponents (m n : ℕ) : 
  (∀ x y : ℝ, ∃ k : ℝ, 2 * x^(n+2) * y^3 = k * (-3 * x^3 * y^(2*m-1))) → 
  (m = 2 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l2296_229672


namespace NUMINAMATH_CALUDE_even_function_properties_l2296_229680

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_incr : increasing_on f 0 7)
  (h_f7 : f 7 = 6) :
  decreasing_on f (-7) 0 ∧ ∀ x, -7 ≤ x → x ≤ 7 → f x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_even_function_properties_l2296_229680


namespace NUMINAMATH_CALUDE_xiaojie_purchase_solution_l2296_229646

/-- Represents the stationery purchase problem --/
structure StationeryPurchase where
  red_black_pen_price : ℕ
  black_refill_price : ℕ
  red_refill_price : ℕ
  black_discount : ℚ
  red_discount : ℚ
  red_black_pens_bought : ℕ
  total_refills_bought : ℕ
  total_spent : ℕ

/-- The specific purchase made by Xiaojie --/
def xiaojie_purchase : StationeryPurchase :=
  { red_black_pen_price := 10
  , black_refill_price := 6
  , red_refill_price := 8
  , black_discount := 1/2
  , red_discount := 3/4
  , red_black_pens_bought := 2
  , total_refills_bought := 10
  , total_spent := 74
  }

/-- Theorem stating the correct number of refills bought and amount saved --/
theorem xiaojie_purchase_solution (p : StationeryPurchase) (h : p = xiaojie_purchase) :
  ∃ (black_refills red_refills : ℕ) (savings : ℕ),
    black_refills + red_refills = p.total_refills_bought ∧
    black_refills = 2 ∧
    red_refills = 8 ∧
    savings = 22 ∧
    p.red_black_pen_price * p.red_black_pens_bought +
    (p.black_refill_price * black_refills + p.red_refill_price * red_refills) -
    p.total_spent = savings :=
  sorry

end NUMINAMATH_CALUDE_xiaojie_purchase_solution_l2296_229646


namespace NUMINAMATH_CALUDE_triangle_max_area_l2296_229621

theorem triangle_max_area (a b c : ℝ) (h1 : a + b = 10) (h2 : c = 6) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  S ≤ 12 ∧ ∃ a b, a + b = 10 ∧ S = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2296_229621


namespace NUMINAMATH_CALUDE_w_squared_values_l2296_229634

theorem w_squared_values (w : ℝ) :
  (2 * w + 17)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = 19.69140625 ∨ w^2 = 43.06640625 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_values_l2296_229634


namespace NUMINAMATH_CALUDE_exists_multiple_of_ones_l2296_229685

theorem exists_multiple_of_ones (n : ℕ) (h_pos : 0 < n) (h_coprime : Nat.Coprime n 10) :
  ∃ k : ℕ, (10^k - 1) % (9 * n) = 0 := by
sorry

end NUMINAMATH_CALUDE_exists_multiple_of_ones_l2296_229685


namespace NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l2296_229642

theorem distance_to_point : ℝ × ℝ → ℝ
  | (x, y) => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point :
  distance_to_point (12, -5) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l2296_229642


namespace NUMINAMATH_CALUDE_sum_squares_of_roots_l2296_229688

theorem sum_squares_of_roots (x₁ x₂ : ℝ) : 
  6 * x₁^2 + 11 * x₁ - 35 = 0 →
  6 * x₂^2 + 11 * x₂ - 35 = 0 →
  x₁ > 2 →
  x₂ > 2 →
  x₁^2 + x₂^2 = 541 / 36 := by
sorry

end NUMINAMATH_CALUDE_sum_squares_of_roots_l2296_229688


namespace NUMINAMATH_CALUDE_expression_evaluation_l2296_229612

theorem expression_evaluation : -20 + 7 * (8 - 2 / 2) = 29 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2296_229612
