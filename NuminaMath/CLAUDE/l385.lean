import Mathlib

namespace NUMINAMATH_CALUDE_jim_age_l385_38510

theorem jim_age (jim fred sam : ℕ) 
  (h1 : jim = 2 * fred)
  (h2 : fred = sam + 9)
  (h3 : jim - 6 = 5 * (sam - 6)) :
  jim = 46 := by sorry

end NUMINAMATH_CALUDE_jim_age_l385_38510


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l385_38569

/-- The function f(x) = x³ + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a

/-- f is monotonically increasing on ℝ -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f a x < f a y

/-- The statement "if a > 1, then f(x) = x³ + a is monotonically increasing on ℝ" 
    is a sufficient but not necessary condition -/
theorem sufficient_not_necessary : 
  (∀ a : ℝ, a > 1 → is_monotone_increasing a) ∧ 
  (∃ a : ℝ, a ≤ 1 ∧ is_monotone_increasing a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l385_38569


namespace NUMINAMATH_CALUDE_smallest_sum_a_b_l385_38577

theorem smallest_sum_a_b (a b : ℕ+) 
  (h : (1 : ℚ) / a + (1 : ℚ) / (2 * a) + (1 : ℚ) / (3 * a) = (1 : ℚ) / (b^2 - 2*b)) : 
  ∀ (x y : ℕ+), 
    ((1 : ℚ) / x + (1 : ℚ) / (2 * x) + (1 : ℚ) / (3 * x) = (1 : ℚ) / (y^2 - 2*y)) → 
    (x + y : ℕ) ≥ (a + b : ℕ) ∧ (a + b : ℕ) = 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_a_b_l385_38577


namespace NUMINAMATH_CALUDE_sum_of_cubic_roots_l385_38568

theorem sum_of_cubic_roots (a b c d : ℝ) (h : ∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24 ↔ a*x^3 + b*x^2 + c*x + d = 0) :
  a ≠ 0 → (sum_of_roots : ℝ) = -b / a ∧ sum_of_roots = -1 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_cubic_roots_l385_38568


namespace NUMINAMATH_CALUDE_monotonic_functional_equation_implies_f_zero_eq_one_l385_38507

/-- A function f: ℝ → ℝ is monotonic if for all x, y ∈ ℝ, x ≤ y implies f(x) ≤ f(y) -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- A function f: ℝ → ℝ satisfies the functional equation f(x+y) = f(x)f(y) for all x, y ∈ ℝ -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

theorem monotonic_functional_equation_implies_f_zero_eq_one
  (f : ℝ → ℝ) (h_mono : Monotonic f) (h_eq : SatisfiesFunctionalEquation f) :
  f 0 = 1 :=
sorry

end NUMINAMATH_CALUDE_monotonic_functional_equation_implies_f_zero_eq_one_l385_38507


namespace NUMINAMATH_CALUDE_tylenol_interval_l385_38508

/-- Represents the duration of Jeremy's Tylenol regimen in weeks -/
def duration : ℕ := 2

/-- Represents the total number of pills Jeremy takes -/
def total_pills : ℕ := 112

/-- Represents the amount of Tylenol in each pill in milligrams -/
def mg_per_pill : ℕ := 500

/-- Represents the amount of Tylenol Jeremy takes per dose in milligrams -/
def mg_per_dose : ℕ := 1000

/-- Theorem stating that the time interval between doses is 6 hours -/
theorem tylenol_interval : 
  (duration * 7 * 24) / ((total_pills * mg_per_pill) / mg_per_dose) = 6 := by
  sorry


end NUMINAMATH_CALUDE_tylenol_interval_l385_38508


namespace NUMINAMATH_CALUDE_joan_balloons_l385_38500

theorem joan_balloons (total : ℕ) (melanie : ℕ) (joan : ℕ) : 
  total = 81 → melanie = 41 → total = joan + melanie → joan = 40 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l385_38500


namespace NUMINAMATH_CALUDE_dinner_bill_proof_l385_38537

/-- The number of friends who went to dinner -/
def total_friends : ℕ := 10

/-- The number of friends who paid -/
def paying_friends : ℕ := 9

/-- The extra amount each paying friend contributed -/
def extra_payment : ℚ := 3

/-- The discount rate applied to the bill -/
def discount_rate : ℚ := 1/10

/-- The original bill before discount -/
def original_bill : ℚ := 300

theorem dinner_bill_proof :
  let discounted_bill := original_bill * (1 - discount_rate)
  let individual_share := discounted_bill / total_friends
  paying_friends * (individual_share + extra_payment) = discounted_bill :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_proof_l385_38537


namespace NUMINAMATH_CALUDE_elena_frog_count_l385_38595

/-- Given a total number of frog eyes and the number of eyes per frog,
    calculate the number of frogs. -/
def count_frogs (total_eyes : ℕ) (eyes_per_frog : ℕ) : ℕ :=
  total_eyes / eyes_per_frog

/-- The problem statement -/
theorem elena_frog_count :
  let total_eyes : ℕ := 20
  let eyes_per_frog : ℕ := 2
  count_frogs total_eyes eyes_per_frog = 10 := by
  sorry

end NUMINAMATH_CALUDE_elena_frog_count_l385_38595


namespace NUMINAMATH_CALUDE_max_product_constraint_max_product_achievable_l385_38527

theorem max_product_constraint (x y : ℕ+) (h : 7 * x + 4 * y = 150) : x * y ≤ 200 := by
  sorry

theorem max_product_achievable : ∃ (x y : ℕ+), 7 * x + 4 * y = 150 ∧ x * y = 200 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_max_product_achievable_l385_38527


namespace NUMINAMATH_CALUDE_problem_solution_l385_38511

theorem problem_solution (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) :
  (b > 1) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a + b ≤ x + y) ∧
  (a * b = 16 ∧ ∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → 16 ≤ x * y) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l385_38511


namespace NUMINAMATH_CALUDE_quadratic_factorization_l385_38580

theorem quadratic_factorization (y : ℝ) : 9*y^2 - 30*y + 25 = (3*y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l385_38580


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l385_38536

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l385_38536


namespace NUMINAMATH_CALUDE_star_five_three_l385_38538

/-- Define the binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a^2 - 2*a*b + b^2

/-- Theorem: When a = 5 and b = 3, a ⋆ b = 4 -/
theorem star_five_three : star 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l385_38538


namespace NUMINAMATH_CALUDE_expression_simplification_l385_38567

theorem expression_simplification (y : ℝ) : 
  3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l385_38567


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l385_38584

theorem cubic_polynomial_root (a b : ℚ) :
  (∃ (x : ℝ), x^3 + a*x + b = 0 ∧ x = 3 - Real.sqrt 5) →
  (∃ (r : ℤ), r^3 + a*r + b = 0) →
  (∃ (r : ℤ), r^3 + a*r + b = 0 ∧ r = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l385_38584


namespace NUMINAMATH_CALUDE_tiles_per_row_l385_38530

/-- Proves that a square room with an area of 144 square feet,
    when covered with 8-inch by 8-inch tiles, will have 18 tiles in each row. -/
theorem tiles_per_row (room_area : ℝ) (tile_size : ℝ) :
  room_area = 144 →
  tile_size = 8 →
  (Real.sqrt room_area * 12) / tile_size = 18 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l385_38530


namespace NUMINAMATH_CALUDE_problem_solution_l385_38564

theorem problem_solution : ∀ A B : ℕ, 
  A = 55 * 100 + 19 * 10 → 
  B = 173 + 5 * 224 → 
  A - B = 4397 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l385_38564


namespace NUMINAMATH_CALUDE_josh_candy_purchase_l385_38516

/-- Given an initial amount of money and the cost of a purchase, 
    calculate the remaining change. -/
def calculate_change (initial_amount cost : ℚ) : ℚ :=
  initial_amount - cost

/-- Prove that given an initial amount of $1.80 and a purchase of $0.45, 
    the remaining change is $1.35. -/
theorem josh_candy_purchase : 
  calculate_change (180/100) (45/100) = 135/100 := by
  sorry

end NUMINAMATH_CALUDE_josh_candy_purchase_l385_38516


namespace NUMINAMATH_CALUDE_circumradius_is_five_l385_38506

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/12 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define a point P on the hyperbola
def P : ℝ × ℝ := sorry

-- Assert that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the centroid G of triangle F₁PF₂
def G : ℝ × ℝ := sorry

-- Define the incenter I of triangle F₁PF₂
def I : ℝ × ℝ := sorry

-- Assert that G and I are parallel to the x-axis
axiom G_I_parallel_x : G.2 = I.2

-- Define the circumradius of triangle F₁PF₂
def circumradius (F₁ F₂ P : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumradius of triangle F₁PF₂ is 5
theorem circumradius_is_five : circumradius F₁ F₂ P = 5 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_is_five_l385_38506


namespace NUMINAMATH_CALUDE_line_plane_parallelism_l385_38532

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem line_plane_parallelism 
  (a b : Line) (α : Plane) 
  (h1 : parallel_plane a α) 
  (h2 : parallel_line a b) 
  (h3 : ¬ contained_in b α) : 
  parallel_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_plane_parallelism_l385_38532


namespace NUMINAMATH_CALUDE_problem_proof_l385_38523

theorem problem_proof : 
  (14^2 * 5^3) / 568 = 43.13380281690141 := by sorry

end NUMINAMATH_CALUDE_problem_proof_l385_38523


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l385_38596

/-- Two planar vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → x = 8 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l385_38596


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l385_38520

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Plane → Prop)
variable (perpLines : Line → Line → Prop)

-- Define the property of being different planes
variable (different : Plane → Plane → Prop)

-- Define the property of being non-coincident lines
variable (nonCoincident : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : different α β)
  (h2 : nonCoincident m n)
  (h3 : perpLine m α)
  (h4 : perpLine n β)
  (h5 : perp α β) :
  perpLines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l385_38520


namespace NUMINAMATH_CALUDE_yoongi_age_l385_38593

theorem yoongi_age (hoseok_age yoongi_age : ℕ) 
  (h1 : yoongi_age = hoseok_age - 2)
  (h2 : yoongi_age + hoseok_age = 18) :
  yoongi_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_age_l385_38593


namespace NUMINAMATH_CALUDE_least_common_multiple_5_to_15_l385_38553

theorem least_common_multiple_5_to_15 : ∃ n : ℕ, 
  (∀ k : ℕ, 5 ≤ k → k ≤ 15 → k ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, 5 ≤ k → k ≤ 15 → k ∣ m) → n ≤ m) ∧
  n = 360360 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_5_to_15_l385_38553


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l385_38581

theorem arithmetic_sequence_first_term 
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : (30 : ℚ) / 2 * (2 * a + 29 * d) = 450) -- Sum of first 30 terms
  (h2 : (30 : ℚ) / 2 * (2 * (a + 30 * d) + 29 * d) = 1950) -- Sum of next 30 terms
  : a = -55 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l385_38581


namespace NUMINAMATH_CALUDE_gerbils_sold_l385_38513

theorem gerbils_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 85 → remaining = 16 → sold = initial - remaining → sold = 69 := by
  sorry

end NUMINAMATH_CALUDE_gerbils_sold_l385_38513


namespace NUMINAMATH_CALUDE_center_number_is_five_l385_38562

-- Define the 2x3 array type
def Array2x3 := Fin 2 → Fin 3 → Nat

-- Define a predicate for consecutive numbers
def Consecutive (a b : Nat) : Prop := a + 1 = b ∨ b + 1 = a

-- Define diagonal adjacency
def DiagonallyAdjacent (i1 j1 i2 j2 : Nat) : Prop :=
  (i1 + 1 = i2 ∧ j1 + 1 = j2) ∨ (i1 + 1 = i2 ∧ j1 = j2 + 1) ∨
  (i1 = i2 + 1 ∧ j1 + 1 = j2) ∨ (i1 = i2 + 1 ∧ j1 = j2 + 1)

-- Define the property of consecutive numbers being diagonally adjacent
def ConsecutiveAreDiagonallyAdjacent (arr : Array2x3) : Prop :=
  ∀ i1 j1 i2 j2, Consecutive (arr i1 j1) (arr i2 j2) → DiagonallyAdjacent i1 j1 i2 j2

-- Define the property that all numbers from 1 to 5 are present
def ContainsAllNumbers (arr : Array2x3) : Prop :=
  ∀ n, n ≥ 1 ∧ n ≤ 5 → ∃ i j, arr i j = n

-- Define the property that corner numbers on one long side sum to 6
def CornersSum6 (arr : Array2x3) : Prop :=
  (arr 0 0 + arr 0 2 = 6) ∨ (arr 1 0 + arr 1 2 = 6)

-- The main theorem
theorem center_number_is_five (arr : Array2x3) 
  (h1 : ConsecutiveAreDiagonallyAdjacent arr)
  (h2 : ContainsAllNumbers arr)
  (h3 : CornersSum6 arr) :
  (arr 0 1 = 5) ∨ (arr 1 1 = 5) :=
sorry

end NUMINAMATH_CALUDE_center_number_is_five_l385_38562


namespace NUMINAMATH_CALUDE_postcards_cost_l385_38517

/-- Represents a country --/
inductive Country
| Italy
| Germany
| Canada
| Japan

/-- Represents a decade --/
inductive Decade
| Fifties
| Sixties
| Seventies
| Eighties
| Nineties

/-- Price of a postcard in cents for a given country --/
def price (c : Country) : ℕ :=
  match c with
  | Country.Italy => 8
  | Country.Germany => 8
  | Country.Canada => 5
  | Country.Japan => 7

/-- Number of postcards for a given country and decade --/
def quantity (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Italy, Decade.Fifties => 5
  | Country.Italy, Decade.Sixties => 12
  | Country.Italy, Decade.Seventies => 11
  | Country.Italy, Decade.Eighties => 10
  | Country.Italy, Decade.Nineties => 6
  | Country.Germany, Decade.Fifties => 9
  | Country.Germany, Decade.Sixties => 5
  | Country.Germany, Decade.Seventies => 13
  | Country.Germany, Decade.Eighties => 15
  | Country.Germany, Decade.Nineties => 7
  | Country.Canada, Decade.Fifties => 3
  | Country.Canada, Decade.Sixties => 7
  | Country.Canada, Decade.Seventies => 6
  | Country.Canada, Decade.Eighties => 10
  | Country.Canada, Decade.Nineties => 11
  | Country.Japan, Decade.Fifties => 6
  | Country.Japan, Decade.Sixties => 8
  | Country.Japan, Decade.Seventies => 9
  | Country.Japan, Decade.Eighties => 5
  | Country.Japan, Decade.Nineties => 9

/-- Total cost of postcards for a given country and set of decades --/
def totalCost (c : Country) (decades : List Decade) : ℕ :=
  (decades.map (quantity c)).sum * price c

/-- Theorem: The total cost of postcards from Canada and Japan issued in the '60s, '70s, and '80s is 269 cents --/
theorem postcards_cost :
  totalCost Country.Canada [Decade.Sixties, Decade.Seventies, Decade.Eighties] +
  totalCost Country.Japan [Decade.Sixties, Decade.Seventies, Decade.Eighties] = 269 := by
  sorry

end NUMINAMATH_CALUDE_postcards_cost_l385_38517


namespace NUMINAMATH_CALUDE_range_of_G_l385_38556

-- Define the function G
def G (x : ℝ) : ℝ := |x + 2| - 2 * |x - 2|

-- State the theorem about the range of G
theorem range_of_G :
  Set.range G = Set.Icc (-8 : ℝ) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_G_l385_38556


namespace NUMINAMATH_CALUDE_six_x_value_l385_38586

theorem six_x_value (x : ℝ) (h : 3 * x - 9 = 12) : 6 * x = 42 := by
  sorry

end NUMINAMATH_CALUDE_six_x_value_l385_38586


namespace NUMINAMATH_CALUDE_exactly_two_subsets_implies_a_values_l385_38519

def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + a = 0}

theorem exactly_two_subsets_implies_a_values (a : ℝ) :
  (∀ S : Set ℝ, S ⊆ A a → (S = ∅ ∨ S = A a)) →
  a = -1 ∨ a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_subsets_implies_a_values_l385_38519


namespace NUMINAMATH_CALUDE_function_inequality_l385_38590

theorem function_inequality (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → a + 2 * 2^x + 4^x < 0) → a < -8 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l385_38590


namespace NUMINAMATH_CALUDE_unique_positive_solution_l385_38550

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l385_38550


namespace NUMINAMATH_CALUDE_bob_net_increase_theorem_l385_38558

/-- Calculates the net increase in weekly earnings given a raise, work hours, and benefit reduction --/
def netIncreaseInWeeklyEarnings (hourlyRaise : ℚ) (weeklyHours : ℕ) (monthlyBenefitReduction : ℚ) : ℚ :=
  let weeklyRaise := hourlyRaise * weeklyHours
  let weeklyBenefitReduction := monthlyBenefitReduction / 4
  weeklyRaise - weeklyBenefitReduction

/-- Theorem stating that given the specified conditions, the net increase in weekly earnings is $5 --/
theorem bob_net_increase_theorem :
  netIncreaseInWeeklyEarnings (1/2) 40 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bob_net_increase_theorem_l385_38558


namespace NUMINAMATH_CALUDE_linear_functions_property_l385_38597

/-- Given two linear functions f and g with specific properties, prove that A + B + 2C equals itself. -/
theorem linear_functions_property (A B C : ℝ) (h1 : A ≠ B) (h2 : A + B ≠ 0) (h3 : C ≠ 0)
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = A * x + B + C)
  (hg : ∀ x, g x = B * x + A - C)
  (h4 : ∀ x, f (g x) - g (f x) = 2 * C) :
  A + B + 2 * C = A + B + 2 * C := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_property_l385_38597


namespace NUMINAMATH_CALUDE_total_non_basalt_rocks_l385_38554

/-- Given two boxes of rocks with some being basalt, calculate the total number of non-basalt rocks --/
theorem total_non_basalt_rocks (total_A total_B basalt_A basalt_B : ℕ) 
  (h1 : total_A = 57)
  (h2 : basalt_A = 25)
  (h3 : total_B = 49)
  (h4 : basalt_B = 19) :
  (total_A - basalt_A) + (total_B - basalt_B) = 62 := by
  sorry

#check total_non_basalt_rocks

end NUMINAMATH_CALUDE_total_non_basalt_rocks_l385_38554


namespace NUMINAMATH_CALUDE_total_candles_l385_38543

theorem total_candles (bedroom : ℕ) (living_room : ℕ) (donovan : ℕ) : 
  bedroom = 20 →
  bedroom = 2 * living_room →
  donovan = 20 →
  bedroom + living_room + donovan = 50 := by
sorry

end NUMINAMATH_CALUDE_total_candles_l385_38543


namespace NUMINAMATH_CALUDE_cistern_length_l385_38571

theorem cistern_length (width depth area : ℝ) (h1 : width = 4)
    (h2 : depth = 1.25) (h3 : area = 55.5) :
  ∃ length : ℝ, length = 7 ∧ 
    area = (length * width) + 2 * (length * depth) + 2 * (width * depth) :=
by sorry

end NUMINAMATH_CALUDE_cistern_length_l385_38571


namespace NUMINAMATH_CALUDE_initial_pups_proof_l385_38515

/-- The number of initial mice -/
def initial_mice : ℕ := 8

/-- The number of additional pups each mouse has in the second round -/
def second_round_pups : ℕ := 6

/-- The number of pups eaten by each adult mouse -/
def eaten_pups : ℕ := 2

/-- The total number of mice at the end -/
def total_mice : ℕ := 280

/-- The initial number of pups per mouse -/
def initial_pups_per_mouse : ℕ := 6

theorem initial_pups_proof :
  initial_mice +
  initial_mice * initial_pups_per_mouse +
  (initial_mice + initial_mice * initial_pups_per_mouse) * second_round_pups -
  (initial_mice + initial_mice * initial_pups_per_mouse) * eaten_pups =
  total_mice :=
by sorry

end NUMINAMATH_CALUDE_initial_pups_proof_l385_38515


namespace NUMINAMATH_CALUDE_simultaneous_hit_probability_l385_38565

theorem simultaneous_hit_probability 
  (prob_A_hit : ℝ) 
  (prob_B_miss : ℝ) 
  (h1 : prob_A_hit = 0.8) 
  (h2 : prob_B_miss = 0.3) 
  (h3 : 0 ≤ prob_A_hit ∧ prob_A_hit ≤ 1) 
  (h4 : 0 ≤ prob_B_miss ∧ prob_B_miss ≤ 1) :
  prob_A_hit * (1 - prob_B_miss) = 14/25 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_hit_probability_l385_38565


namespace NUMINAMATH_CALUDE_proposition_truth_l385_38545

-- Define the propositions P and q
def P : Prop := ∀ x y : ℝ, x > y → -x > -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Define the compound propositions
def prop1 : Prop := P ∧ q
def prop2 : Prop := ¬P ∨ ¬q
def prop3 : Prop := P ∧ ¬q
def prop4 : Prop := ¬P ∨ q

-- Theorem statement
theorem proposition_truth : 
  ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
sorry

end NUMINAMATH_CALUDE_proposition_truth_l385_38545


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l385_38588

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((2 ∣ m) ∧ (3 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m))) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l385_38588


namespace NUMINAMATH_CALUDE_pythagorean_diagonal_l385_38591

theorem pythagorean_diagonal (m : ℕ) (h_m : m ≥ 3) : 
  let width : ℕ := 2 * m
  let diagonal : ℕ := m^2 + 1
  let height : ℕ := diagonal - 2
  (width : ℤ)^2 + height^2 = diagonal^2 := by sorry

end NUMINAMATH_CALUDE_pythagorean_diagonal_l385_38591


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l385_38528

theorem cylinder_cone_volume_relation :
  ∀ (d : ℝ) (h : ℝ),
    d > 0 →
    h = 2 * d →
    π * (d / 2)^2 * h = 81 * π →
    (1 / 3) * π * (d / 2)^2 * h = 27 * π * (6 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_relation_l385_38528


namespace NUMINAMATH_CALUDE_orchestra_members_count_l385_38599

theorem orchestra_members_count : ∃! n : ℕ,
  150 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧
  n % 8 = 3 ∧
  n % 9 = 5 ∧
  n = 211 := by sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l385_38599


namespace NUMINAMATH_CALUDE_complex_sum_problem_l385_38529

theorem complex_sum_problem (a b c d e f : ℂ) : 
  b = 1 →
  e = -a - 2*c →
  a + b * Complex.I + c + d * Complex.I + e + f * Complex.I = 3 + 2 * Complex.I →
  d + f = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l385_38529


namespace NUMINAMATH_CALUDE_implication_contrapositive_equivalence_l385_38572

theorem implication_contrapositive_equivalence (R S : Prop) :
  (R → S) ↔ (¬S → ¬R) := by sorry

end NUMINAMATH_CALUDE_implication_contrapositive_equivalence_l385_38572


namespace NUMINAMATH_CALUDE_f_composition_value_l385_38573

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_value : f (f (f (-1))) = Real.pi + 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l385_38573


namespace NUMINAMATH_CALUDE_team_cautions_l385_38574

theorem team_cautions (total_players : ℕ) (red_cards : ℕ) (yellow_per_red : ℕ) :
  total_players = 11 →
  red_cards = 3 →
  yellow_per_red = 2 →
  ∃ (no_caution players_with_yellow : ℕ),
    no_caution + players_with_yellow = total_players ∧
    players_with_yellow = red_cards * yellow_per_red ∧
    no_caution = 5 :=
by sorry

end NUMINAMATH_CALUDE_team_cautions_l385_38574


namespace NUMINAMATH_CALUDE_fathers_age_l385_38502

theorem fathers_age (n m f : ℕ) (h1 : n * m = f / 7) (h2 : (n + 3) * (m + 3) = f + 3) : f = 21 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l385_38502


namespace NUMINAMATH_CALUDE_square_difference_equality_l385_38552

theorem square_difference_equality : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l385_38552


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l385_38598

/-- Given a quadratic inequality ax^2 + b > 0 with solution set (-∞, -1/2) ∪ (1/3, ∞), prove ab = 24 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, a * x^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3) → a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l385_38598


namespace NUMINAMATH_CALUDE_sum_difference_3010_l385_38525

/-- The sum of the first n odd counting numbers -/
def sum_odd (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The sum of the first n even counting numbers -/
def sum_even (n : ℕ) : ℕ := n * (2 * n + 2)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sum_difference (n : ℕ) : ℕ := sum_even n - sum_odd n

theorem sum_difference_3010 :
  sum_difference 3010 = 3010 := by sorry

end NUMINAMATH_CALUDE_sum_difference_3010_l385_38525


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l385_38566

theorem real_part_of_complex_number (z : ℂ) (a : ℝ) :
  z = (1 : ℂ) / (1 + I) + a * I → z.im = 0 → a = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l385_38566


namespace NUMINAMATH_CALUDE_saree_price_calculation_l385_38578

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.15) = 306 → P = 450 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l385_38578


namespace NUMINAMATH_CALUDE_sum_division_l385_38587

/-- The problem of dividing a sum among x, y, and z -/
theorem sum_division (x y z : ℝ) : 
  (∀ (r : ℝ), y = 0.45 * r → z = 0.5 * r → x = r) →  -- For each rupee x gets, y gets 0.45 and z gets 0.5
  y = 63 →  -- y's share is 63 rupees
  x + y + z = 273 := by  -- The total amount is 273 rupees
sorry


end NUMINAMATH_CALUDE_sum_division_l385_38587


namespace NUMINAMATH_CALUDE_cos_2alpha_2beta_l385_38548

theorem cos_2alpha_2beta (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) : 
  Real.cos (2*α + 2*β) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_2beta_l385_38548


namespace NUMINAMATH_CALUDE_steve_wood_needed_l385_38589

/-- The amount of wood Steve needs to buy for his bench project -/
def total_wood_needed (long_pieces : ℕ) (long_length : ℕ) (short_pieces : ℕ) (short_length : ℕ) : ℕ :=
  long_pieces * long_length + short_pieces * short_length

/-- Proof that Steve needs to buy 28 feet of wood -/
theorem steve_wood_needed : total_wood_needed 6 4 2 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_steve_wood_needed_l385_38589


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l385_38549

/-- Given a triangle with angles 45°, 3x°, and x°, prove that x = 33.75° -/
theorem triangle_angle_proof (x : ℝ) : 
  45 + 3*x + x = 180 → x = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l385_38549


namespace NUMINAMATH_CALUDE_min_value_M_min_value_expression_equality_condition_l385_38505

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem 1: Minimum value of M
theorem min_value_M : 
  (∃ (M : ℝ), ∀ (m : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ m) → M ≤ m) ∧ 
  (∃ (x₀ : ℝ), f x₀ ≤ 2) := by sorry

-- Theorem 2: Minimum value of 1/(2a) + 1/(a+b)
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3*a + b = 2) :
  1/(2*a) + 1/(a+b) ≥ 2 := by sorry

-- Theorem 3: Equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3*a + b = 2) :
  1/(2*a) + 1/(a+b) = 2 ↔ a = 1/2 ∧ b = 1/2 := by sorry

end NUMINAMATH_CALUDE_min_value_M_min_value_expression_equality_condition_l385_38505


namespace NUMINAMATH_CALUDE_complex_multiplication_l385_38592

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l385_38592


namespace NUMINAMATH_CALUDE_solution_set_and_range_l385_38579

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≥ 1 ↔ x ≤ -5/2 ∨ x ≥ 3/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ -1 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l385_38579


namespace NUMINAMATH_CALUDE_billy_age_is_47_25_l385_38531

-- Define Billy's and Joe's ages
def billy_age : ℝ := sorry
def joe_age : ℝ := sorry

-- State the theorem
theorem billy_age_is_47_25 :
  (billy_age = 3 * joe_age) →  -- Billy's age is three times Joe's age
  (billy_age + joe_age = 63) → -- The sum of their ages is 63 years
  (billy_age = 47.25) :=       -- Billy's age is 47.25 years
by
  sorry

end NUMINAMATH_CALUDE_billy_age_is_47_25_l385_38531


namespace NUMINAMATH_CALUDE_quadratic_roots_l385_38583

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 2 ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l385_38583


namespace NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l385_38555

theorem modular_inverse_15_mod_17 : ∃ x : ℕ, x ≤ 16 ∧ (15 * x) % 17 = 1 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_modular_inverse_15_mod_17_l385_38555


namespace NUMINAMATH_CALUDE_find_a_value_l385_38582

theorem find_a_value (x y : ℝ) (a : ℝ) 
  (h1 : x = 2)
  (h2 : y = 1)
  (h3 : a * x - y = 3) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l385_38582


namespace NUMINAMATH_CALUDE_x_equation_implies_y_values_l385_38544

theorem x_equation_implies_y_values (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 54 →
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 4)
  y = 11.25 ∨ y = 10.125 := by
sorry

end NUMINAMATH_CALUDE_x_equation_implies_y_values_l385_38544


namespace NUMINAMATH_CALUDE_unique_quadratic_pair_l385_38504

theorem unique_quadratic_pair : 
  ∃! (b c : ℕ+), 
    (∀ x : ℝ, (x^2 + 2*b*x + c ≤ 0 → x^2 + 2*b*x + c = 0)) ∧ 
    (∀ x : ℝ, (x^2 + 2*c*x + b ≤ 0 → x^2 + 2*c*x + b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_pair_l385_38504


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l385_38585

/-- Given a cubic function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem cubic_function_derivative (a : ℝ) :
  let f := λ x : ℝ => a * x^3 + 3 * x^2 + 2
  let f' := λ x : ℝ => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l385_38585


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l385_38542

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 = 3*x + 4) → (x = Real.sqrt (3*x + 4)) ∧
  ¬(∀ x : ℝ, (x = Real.sqrt (3*x + 4)) → (x^2 = 3*x + 4)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l385_38542


namespace NUMINAMATH_CALUDE_tournament_prize_orders_l385_38514

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of elimination rounds in the tournament -/
def num_rounds : ℕ := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Theorem stating the number of possible prize orders in the tournament -/
theorem tournament_prize_orders :
  (outcomes_per_match ^ num_rounds : ℕ) = 32 := by sorry

end NUMINAMATH_CALUDE_tournament_prize_orders_l385_38514


namespace NUMINAMATH_CALUDE_interest_calculation_years_l385_38546

theorem interest_calculation_years (P r : ℝ) (h1 : P = 625) (h2 : r = 0.04) : 
  ∃ n : ℕ, n = 2 ∧ P * ((1 + r)^n - 1) - P * r * n = 1 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_years_l385_38546


namespace NUMINAMATH_CALUDE_recurrence_relation_initial_conditions_sequence_satisfies_conditions_l385_38501

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 3 * 2^n + 3^n + (1/2) * n + 11/4

/-- Recurrence relation -/
theorem recurrence_relation (n : ℕ) (h : n ≥ 2) :
  a n = 5 * a (n-1) - 6 * a (n-2) + n + 2 := by sorry

/-- Initial conditions -/
theorem initial_conditions :
  a 0 = 27/4 ∧ a 1 = 49/4 := by sorry

/-- Main theorem: The sequence satisfies the recurrence relation and initial conditions -/
theorem sequence_satisfies_conditions :
  (∀ n : ℕ, n ≥ 2 → a n = 5 * a (n-1) - 6 * a (n-2) + n + 2) ∧
  a 0 = 27/4 ∧ a 1 = 49/4 := by sorry

end NUMINAMATH_CALUDE_recurrence_relation_initial_conditions_sequence_satisfies_conditions_l385_38501


namespace NUMINAMATH_CALUDE_profit_maximizing_prices_l385_38563

/-- Represents the selling price in yuan -/
def selling_price : ℝ → ℝ := id

/-- Represents the daily sales quantity as a function of selling price -/
def daily_sales (x : ℝ) : ℝ := 200 - (x - 20) * 20

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 12) * (daily_sales x)

/-- The theorem states that 19 and 23 are the only selling prices that achieve a daily profit of 1540 yuan -/
theorem profit_maximizing_prices :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  daily_profit x₁ = 1540 ∧ 
  daily_profit x₂ = 1540 ∧
  (∀ x : ℝ, daily_profit x = 1540 → (x = x₁ ∨ x = x₂)) :=
sorry

end NUMINAMATH_CALUDE_profit_maximizing_prices_l385_38563


namespace NUMINAMATH_CALUDE_function_value_problem_l385_38570

theorem function_value_problem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (2 * x + 1) = 3 * x - 2) →
  f a = 4 →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_function_value_problem_l385_38570


namespace NUMINAMATH_CALUDE_count_non_multiples_is_675_l385_38540

/-- The count of three-digit numbers that are not multiples of 6 or 8 -/
def count_non_multiples : ℕ :=
  let total_three_digit_numbers := 999 - 100 + 1
  let multiples_of_6 := (999 / 6) - (99 / 6)
  let multiples_of_8 := (999 / 8) - (99 / 8)
  let multiples_of_24 := (999 / 24) - (99 / 24)
  total_three_digit_numbers - (multiples_of_6 + multiples_of_8 - multiples_of_24)

theorem count_non_multiples_is_675 : count_non_multiples = 675 := by
  sorry

end NUMINAMATH_CALUDE_count_non_multiples_is_675_l385_38540


namespace NUMINAMATH_CALUDE_nancy_apples_l385_38575

def mike_apples : ℕ := 7
def keith_apples : ℕ := 6
def total_apples : ℕ := 16

theorem nancy_apples :
  total_apples - (mike_apples + keith_apples) = 3 :=
by sorry

end NUMINAMATH_CALUDE_nancy_apples_l385_38575


namespace NUMINAMATH_CALUDE_move_right_two_units_l385_38521

/-- Moving a point 2 units to the right in a Cartesian coordinate system -/
theorem move_right_two_units (initial_x initial_y : ℝ) :
  let initial_point := (initial_x, initial_y)
  let final_point := (initial_x + 2, initial_y)
  initial_point = (1, 1) → final_point = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_move_right_two_units_l385_38521


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l385_38524

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x < 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l385_38524


namespace NUMINAMATH_CALUDE_simplify_expression_l385_38541

theorem simplify_expression (x : ℝ) (h : x < 0) :
  (2 * abs x + (x^6)^(1/6) + (x^5)^(1/5)) / x = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l385_38541


namespace NUMINAMATH_CALUDE_probability_two_white_is_three_tenths_l385_38533

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def drawn_balls : ℕ := 2

def probability_two_white : ℚ := (white_balls.choose drawn_balls : ℚ) / (total_balls.choose drawn_balls)

theorem probability_two_white_is_three_tenths :
  probability_two_white = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_two_white_is_three_tenths_l385_38533


namespace NUMINAMATH_CALUDE_student_marks_l385_38503

theorem student_marks (total_marks : ℕ) (passing_percentage : ℚ) (failed_by : ℕ) (obtained_marks : ℕ) : 
  total_marks = 400 →
  passing_percentage = 33 / 100 →
  failed_by = 40 →
  obtained_marks = (total_marks * passing_percentage).floor - failed_by →
  obtained_marks = 92 := by
sorry

end NUMINAMATH_CALUDE_student_marks_l385_38503


namespace NUMINAMATH_CALUDE_uniform_random_transformation_l385_38594

/-- A uniform random variable on an interval -/
def UniformRandom (a b : ℝ) (X : ℝ → Prop) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → X x

theorem uniform_random_transformation (b₁ : ℝ → Prop) (b : ℝ → Prop) :
  UniformRandom 0 1 b₁ →
  (∀ x, b x ↔ ∃ y, b₁ y ∧ x = 3 * (y - 2)) →
  UniformRandom (-6) (-3) b :=
sorry

end NUMINAMATH_CALUDE_uniform_random_transformation_l385_38594


namespace NUMINAMATH_CALUDE_simplify_fraction_l385_38561

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l385_38561


namespace NUMINAMATH_CALUDE_f_properties_l385_38559

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2

theorem f_properties (a : ℝ) :
  (∀ x > 1, Monotone (f (-2)))
  ∧ (∀ x ∈ Set.Icc 1 (exp 1), f a x ≥ 
      (if a ≥ -2 then 1
       else if a > -2 * (exp 1)^2 then a/2 * log (-a/2) - a/2
       else a + (exp 1)^2))
  ∧ (∃ x ∈ Set.Icc 1 (exp 1), f a x = 
      (if a ≥ -2 then 1
       else if a > -2 * (exp 1)^2 then a/2 * log (-a/2) - a/2
       else a + (exp 1)^2))
  ∧ (∃ x ∈ Set.Icc 1 (exp 1), f a x = 
      (if a ≥ -2 then f a 1
       else if a > -2 * (exp 1)^2 then f a (sqrt (-a/2))
       else f a (exp 1))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l385_38559


namespace NUMINAMATH_CALUDE_impossible_to_equalize_l385_38560

/-- Represents the numbers in the six sectors of the circle -/
structure CircleNumbers where
  sectors : Fin 6 → ℤ

/-- Represents an operation of increasing two adjacent numbers by 1 -/
inductive Operation
  | increase_adjacent : Fin 6 → Operation

/-- Applies an operation to the circle numbers -/
def apply_operation (nums : CircleNumbers) (op : Operation) : CircleNumbers :=
  match op with
  | Operation.increase_adjacent i =>
      let j := (i + 1) % 6
      { sectors := fun k =>
          if k = i || k = j then nums.sectors k + 1 else nums.sectors k }

/-- Checks if all numbers in the circle are equal -/
def all_equal (nums : CircleNumbers) : Prop :=
  ∀ i j : Fin 6, nums.sectors i = nums.sectors j

/-- The main theorem stating that it's impossible to make all numbers equal -/
theorem impossible_to_equalize (initial : CircleNumbers) :
  ¬∃ (ops : List Operation), all_equal (ops.foldl apply_operation initial) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_equalize_l385_38560


namespace NUMINAMATH_CALUDE_triangle_with_specific_circumcircle_l385_38534

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def circumscribed_circle_diameter (a b c : ℕ) : ℚ :=
  (a * b * c : ℚ) / ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) : ℚ) * 4

theorem triangle_with_specific_circumcircle :
  ∀ a b c : ℕ,
    is_triangle a b c →
    circumscribed_circle_diameter a b c = 25/4 →
    (a = 5 ∧ b = 5 ∧ c = 6) ∨ (a = 5 ∧ b = 6 ∧ c = 5) ∨ (a = 6 ∧ b = 5 ∧ c = 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_specific_circumcircle_l385_38534


namespace NUMINAMATH_CALUDE_team_formation_with_girls_l385_38518

-- Define the total number of people
def total_people : Nat := 10

-- Define the number of boys
def num_boys : Nat := 5

-- Define the number of girls
def num_girls : Nat := 5

-- Define the team size
def team_size : Nat := 3

-- Theorem statement
theorem team_formation_with_girls (total_people num_boys num_girls team_size : Nat) 
  (h1 : total_people = num_boys + num_girls)
  (h2 : num_boys = 5)
  (h3 : num_girls = 5)
  (h4 : team_size = 3) :
  (Nat.choose total_people team_size) - (Nat.choose num_boys team_size) = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_with_girls_l385_38518


namespace NUMINAMATH_CALUDE_ivan_travel_theorem_l385_38522

/-- Represents the travel scenario of Ivan Semenovich -/
structure TravelScenario where
  usual_travel_time : ℝ
  usual_arrival_time : ℝ
  late_departure : ℝ
  speed_increase : ℝ
  new_arrival_time : ℝ

/-- The theorem to be proved -/
theorem ivan_travel_theorem (scenario : TravelScenario) 
  (h1 : scenario.usual_arrival_time = 9 * 60)  -- 9:00 AM in minutes
  (h2 : scenario.late_departure = 40)
  (h3 : scenario.speed_increase = 0.6)
  (h4 : scenario.new_arrival_time = 8 * 60 + 35)  -- 8:35 AM in minutes
  : ∃ (optimal_increase : ℝ),
    optimal_increase = 0.3 ∧
    scenario.usual_arrival_time = 
      scenario.usual_travel_time * (1 - scenario.late_departure / scenario.usual_travel_time) / (1 + optimal_increase) + 
      scenario.late_departure :=
by sorry

end NUMINAMATH_CALUDE_ivan_travel_theorem_l385_38522


namespace NUMINAMATH_CALUDE_set_operations_and_range_l385_38576

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 1}

-- State the theorem
theorem set_operations_and_range :
  (A ∩ B = {x : ℝ | 3 < x ∧ x < 6}) ∧
  ((Set.univ \ A) ∪ (Set.univ \ B) = {x : ℝ | x ≤ 3 ∨ x ≥ 6}) ∧
  (∀ a : ℝ, (B ∪ C a = B) ↔ (a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5))) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l385_38576


namespace NUMINAMATH_CALUDE_min_value_of_function_l385_38557

theorem min_value_of_function (x : ℝ) (h : x ≥ 0) :
  x + 1 / (x + 1) ≥ 1 ∧ ∃ y : ℝ, y ≥ 0 ∧ y + 1 / (y + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l385_38557


namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_implies_s_positive_l385_38509

theorem quadratic_roots_greater_than_one_implies_s_positive
  (b c : ℝ)
  (h1 : ∃ x y : ℝ, x > 1 ∧ y > 1 ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0)
  : b + c + 1 > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_one_implies_s_positive_l385_38509


namespace NUMINAMATH_CALUDE_equation_c_is_quadratic_l385_38551

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing (x-1)(x+2)=1 --/
def f (x : ℝ) : ℝ := (x - 1) * (x + 2) - 1

theorem equation_c_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_c_is_quadratic_l385_38551


namespace NUMINAMATH_CALUDE_g_zeros_l385_38547

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - 2 * x + 1

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x * Real.log x

theorem g_zeros (a : ℝ) (h : a > 0) :
  (∃! x, g a x = 0 ∧ x > 0) ∧ a = Real.exp (-1) ∨
  (∀ x > 0, g a x ≠ 0) ∧ a > Real.exp (-1) ∨
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧
    ∀ x, x > 0 → g a x = 0 → x = x₁ ∨ x = x₂) ∧ 0 < a ∧ a < Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_g_zeros_l385_38547


namespace NUMINAMATH_CALUDE_pond_to_field_ratio_l385_38512

theorem pond_to_field_ratio : 
  let field_length : ℝ := 48
  let field_width : ℝ := 24
  let pond_side : ℝ := 8
  let field_area : ℝ := field_length * field_width
  let pond_area : ℝ := pond_side * pond_side
  pond_area / field_area = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_ratio_l385_38512


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l385_38526

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, is_prime n ∧ p = 2^n - 1 ∧ is_prime p

theorem largest_mersenne_prime_under_1000 :
  (∀ p : ℕ, p < 1000 → is_mersenne_prime p → p ≤ 127) ∧
  is_mersenne_prime 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l385_38526


namespace NUMINAMATH_CALUDE_matching_color_probability_l385_38535

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Calculates the total number of jelly beans a person has -/
def total_jelly_beans (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.blue

/-- Abe's jelly beans -/
def abe_jb : JellyBeans := { green := 2, red := 3, blue := 0 }

/-- Bob's jelly beans -/
def bob_jb : JellyBeans := { green := 2, red := 3, blue := 2 }

/-- Calculates the probability of picking a specific color -/
def prob_color (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / (total_jelly_beans jb)

/-- Theorem: The probability of Abe and Bob showing the same color is 13/35 -/
theorem matching_color_probability : 
  (prob_color abe_jb abe_jb.green * prob_color bob_jb bob_jb.green) +
  (prob_color abe_jb abe_jb.red * prob_color bob_jb bob_jb.red) = 13/35 := by
  sorry

end NUMINAMATH_CALUDE_matching_color_probability_l385_38535


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l385_38539

/-- A quadratic radical is considered simpler if it cannot be further simplified to a non-radical form or a simpler radical form. -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop :=
  ∀ y : ℝ, y ≠ x → (∃ z : ℝ, y = z ^ 2) → ¬(∃ w : ℝ, x = w ^ 2)

/-- The given options for quadratic radicals -/
def QuadraticRadicals (a b : ℝ) : Set ℝ :=
  {Real.sqrt 9, Real.sqrt (a^2 + b^2), Real.sqrt 0.7, Real.sqrt (a^3)}

/-- Theorem stating that √(a² + b²) is the simplest quadratic radical among the given options -/
theorem simplest_quadratic_radical (a b : ℝ) :
  ∀ x ∈ QuadraticRadicals a b, x = Real.sqrt (a^2 + b^2) ∨ ¬(IsSimplestQuadraticRadical x) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l385_38539
