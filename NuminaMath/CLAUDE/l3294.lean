import Mathlib

namespace NUMINAMATH_CALUDE_common_chord_length_l3294_329421

/-- The length of the common chord of two overlapping circles -/
theorem common_chord_length (r : ℝ) (h : r = 15) : 
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 15 * Real.sqrt 3 := by
  sorry

#check common_chord_length

end NUMINAMATH_CALUDE_common_chord_length_l3294_329421


namespace NUMINAMATH_CALUDE_license_plate_increase_l3294_329435

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters in a license plate -/
def num_letters_in_plate : ℕ := 2

/-- The number of digits in the old license plate format -/
def num_digits_old : ℕ := 3

/-- The number of digits in the new license plate format -/
def num_digits_new : ℕ := 4

/-- The number of possible license plates in the old format -/
def old_plates : ℕ := num_letters ^ num_letters_in_plate * num_digits ^ num_digits_old

/-- The number of possible license plates in the new format -/
def new_plates : ℕ := num_letters ^ num_letters_in_plate * num_digits ^ num_digits_new

/-- The theorem stating that the ratio of new plates to old plates is 10 -/
theorem license_plate_increase : new_plates / old_plates = 10 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3294_329435


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3294_329402

theorem geometric_sequence_middle_term (b : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 30 * r = b ∧ b * r = 9/4) → b = 3 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l3294_329402


namespace NUMINAMATH_CALUDE_number_of_green_balls_l3294_329449

/-- Given a bag with blue and green balls, prove the number of green balls. -/
theorem number_of_green_balls
  (blue_balls : ℕ)
  (total_balls : ℕ)
  (h1 : blue_balls = 10)
  (h2 : (blue_balls : ℚ) / total_balls = 1 / 5)
  : total_balls - blue_balls = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_green_balls_l3294_329449


namespace NUMINAMATH_CALUDE_germination_probability_approx_095_l3294_329481

/-- Represents a batch of seeds with its germination data -/
structure SeedBatch where
  seeds : ℕ
  germinations : ℕ

/-- Calculates the germination rate for a batch of seeds -/
def germinationRate (batch : SeedBatch) : ℚ :=
  batch.germinations / batch.seeds

/-- The data from the experiment -/
def seedData : List SeedBatch := [
  ⟨100, 96⟩,
  ⟨300, 284⟩,
  ⟨400, 380⟩,
  ⟨600, 571⟩,
  ⟨1000, 948⟩,
  ⟨2000, 1902⟩,
  ⟨3000, 2848⟩
]

/-- The average germination rate from the experiment -/
def averageGerminationRate : ℚ :=
  (seedData.map germinationRate).sum / seedData.length

/-- Theorem stating that the probability of germination is approximately 0.95 -/
theorem germination_probability_approx_095 :
  ∃ ε > 0, abs (averageGerminationRate - 95/100) < ε ∧ ε < 1/100 :=
sorry

end NUMINAMATH_CALUDE_germination_probability_approx_095_l3294_329481


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3294_329498

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : r - p = 28) :
  (q + r) / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3294_329498


namespace NUMINAMATH_CALUDE_tournament_committee_count_l3294_329443

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team -/
def host_selection : ℕ := 3

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- Theorem stating the total number of possible tournament committees -/
theorem tournament_committee_count :
  (num_teams : ℕ) * (Nat.choose team_size host_selection) * 
  (Nat.choose team_size non_host_selection)^(num_teams - 1) = 1296540 := by
  sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l3294_329443


namespace NUMINAMATH_CALUDE_f_increasing_l3294_329474

-- Define the function f(x) = x^3 - 1
def f (x : ℝ) : ℝ := x^3 - 1

-- Theorem stating that f is increasing over its domain
theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_increasing_l3294_329474


namespace NUMINAMATH_CALUDE_not_strictly_decreasing_cubic_function_l3294_329445

theorem not_strictly_decreasing_cubic_function (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (-x₁^3 + b*x₁^2 - (2*b + 3)*x₁ + 2 - b) ≤ (-x₂^3 + b*x₂^2 - (2*b + 3)*x₂ + 2 - b)) ↔ 
  (b < -1 ∨ b > 3) :=
by sorry

end NUMINAMATH_CALUDE_not_strictly_decreasing_cubic_function_l3294_329445


namespace NUMINAMATH_CALUDE_revenue_calculation_l3294_329466

def calculateDailyRevenue (booksSold : ℕ) (price : ℚ) (discount : ℚ) : ℚ :=
  booksSold * (price * (1 - discount))

def totalRevenue (initialStock : ℕ) (price : ℚ) 
  (mondaySales : ℕ) (mondayDiscount : ℚ)
  (tuesdaySales : ℕ) (tuesdayDiscount : ℚ)
  (wednesdaySales : ℕ) (wednesdayDiscount : ℚ)
  (thursdaySales : ℕ) (thursdayDiscount : ℚ)
  (fridaySales : ℕ) (fridayDiscount : ℚ) : ℚ :=
  calculateDailyRevenue mondaySales price mondayDiscount +
  calculateDailyRevenue tuesdaySales price tuesdayDiscount +
  calculateDailyRevenue wednesdaySales price wednesdayDiscount +
  calculateDailyRevenue thursdaySales price thursdayDiscount +
  calculateDailyRevenue fridaySales price fridayDiscount

theorem revenue_calculation :
  totalRevenue 800 25 
    60 (10 / 100)
    10 0
    20 (5 / 100)
    44 (15 / 100)
    66 (20 / 100) = 4330 := by
  sorry

end NUMINAMATH_CALUDE_revenue_calculation_l3294_329466


namespace NUMINAMATH_CALUDE_smallest_m_no_real_roots_l3294_329486

theorem smallest_m_no_real_roots : 
  ∀ m : ℤ, (∀ x : ℝ, 3*x*(m*x-5) - 2*x^2 + 7 ≠ 0) → m ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_roots_l3294_329486


namespace NUMINAMATH_CALUDE_min_product_of_primes_l3294_329426

theorem min_product_of_primes (x y z : Nat) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
  Odd x ∧ Odd y ∧ Odd z ∧
  (y^5 + 1) % x = 0 ∧
  (z^5 + 1) % y = 0 ∧
  (x^5 + 1) % z = 0 →
  ∀ a b c : Nat, 
    Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
    Odd a ∧ Odd b ∧ Odd c ∧
    (b^5 + 1) % a = 0 ∧
    (c^5 + 1) % b = 0 ∧
    (a^5 + 1) % c = 0 →
    x * y * z ≤ a * b * c ∧
    x * y * z = 2013 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_primes_l3294_329426


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3294_329479

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → x + y^2 + z^4 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3294_329479


namespace NUMINAMATH_CALUDE_inverse_function_root_uniqueness_l3294_329473

theorem inverse_function_root_uniqueness 
  (f : ℝ → ℝ) (h_inv : Function.Injective f) :
  ∀ m : ℝ, ∃! x : ℝ, f x = m ∨ ∀ y : ℝ, f y ≠ m :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_root_uniqueness_l3294_329473


namespace NUMINAMATH_CALUDE_castle_provisions_duration_l3294_329431

/-- 
Proves that given the conditions of the castle's food provisions,
the initial food supply was meant to last 120 days.
-/
theorem castle_provisions_duration 
  (initial_people : ℕ) 
  (people_left : ℕ) 
  (days_before_leaving : ℕ) 
  (days_after_leaving : ℕ) 
  (h1 : initial_people = 300)
  (h2 : people_left = 100)
  (h3 : days_before_leaving = 30)
  (h4 : days_after_leaving = 90)
  : ℕ := by
  sorry

#check castle_provisions_duration

end NUMINAMATH_CALUDE_castle_provisions_duration_l3294_329431


namespace NUMINAMATH_CALUDE_arithmetic_operations_l3294_329401

theorem arithmetic_operations :
  ((-12) + (-6) - (-28) = 10) ∧
  ((-8/5) * (15/4) / (-9) = 2/3) ∧
  ((-3/16 - 7/24 + 5/6) * (-48) = -17) ∧
  (-(3^2) + (7/8 - 1) * ((-2)^2) = -19/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l3294_329401


namespace NUMINAMATH_CALUDE_parabolas_sum_l3294_329446

/-- Given two parabolas that intersect the coordinate axes at four points forming a rhombus -/
structure Parabolas where
  a : ℝ
  b : ℝ
  parabola1 : ℝ → ℝ
  parabola2 : ℝ → ℝ
  h_parabola1 : ∀ x, parabola1 x = a * x^2 - 2
  h_parabola2 : ∀ x, parabola2 x = 6 - b * x^2
  h_rhombus : ∃ x1 x2 y1 y2, 
    parabola1 x1 = 0 ∧ parabola1 x2 = 0 ∧
    parabola2 0 = y1 ∧ parabola2 0 = y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2
  h_area : (x2 - x1) * (y2 - y1) = 24
  h_b_eq_2a : b = 2 * a

/-- The sum of a and b is 6 -/
theorem parabolas_sum (p : Parabolas) : p.a + p.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_sum_l3294_329446


namespace NUMINAMATH_CALUDE_octagon_angle_property_l3294_329464

theorem octagon_angle_property (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 ↔ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_octagon_angle_property_l3294_329464


namespace NUMINAMATH_CALUDE_sin_cos_difference_65_35_l3294_329407

theorem sin_cos_difference_65_35 :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) -
  Real.cos (65 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_65_35_l3294_329407


namespace NUMINAMATH_CALUDE_remainder_after_adding_2947_l3294_329430

theorem remainder_after_adding_2947 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2947_l3294_329430


namespace NUMINAMATH_CALUDE_cow_count_is_twelve_l3294_329452

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 12 under the given conditions -/
theorem cow_count_is_twelve (count : AnimalCount) 
    (h : totalLegs count = 2 * totalHeads count + 24) : 
    count.cows = 12 := by
  sorry


end NUMINAMATH_CALUDE_cow_count_is_twelve_l3294_329452


namespace NUMINAMATH_CALUDE_odd_integer_minus_twenty_l3294_329461

theorem odd_integer_minus_twenty : 
  (2 * 53 - 1) - 20 = 85 := by sorry

end NUMINAMATH_CALUDE_odd_integer_minus_twenty_l3294_329461


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_8_l3294_329434

theorem coefficient_x3y5_in_x_plus_y_8 :
  Finset.sum (Finset.range 9) (λ k => Nat.choose 8 k * (if k = 3 then 1 else 0)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_x_plus_y_8_l3294_329434


namespace NUMINAMATH_CALUDE_brady_hours_june_l3294_329410

def hours_per_day_april : ℝ := 6
def hours_per_day_september : ℝ := 8
def average_hours_per_month : ℝ := 190
def days_per_month : ℕ := 30

theorem brady_hours_june :
  ∃ (hours_per_day_june : ℝ),
    hours_per_day_june * days_per_month +
    hours_per_day_april * days_per_month +
    hours_per_day_september * days_per_month =
    average_hours_per_month * 3 ∧
    hours_per_day_june = 5 := by
  sorry

end NUMINAMATH_CALUDE_brady_hours_june_l3294_329410


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3294_329470

theorem concentric_circles_radii_difference (r : ℝ) (h : r > 0) :
  let R := (4 * r ^ 2) ^ (1 / 2 : ℝ)
  R - r = r :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3294_329470


namespace NUMINAMATH_CALUDE_hexagonal_tiles_count_l3294_329484

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- triangular
| 1 => 4  -- square
| 2 => 6  -- hexagonal

/-- The total number of tiles in the box -/
def total_tiles : ℕ := 35

/-- The total number of edges from all tiles -/
def total_edges : ℕ := 128

theorem hexagonal_tiles_count :
  ∃ (a b c : ℕ),
    a + b + c = total_tiles ∧
    3 * a + 4 * b + 6 * c = total_edges ∧
    c = 6 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_tiles_count_l3294_329484


namespace NUMINAMATH_CALUDE_shortening_powers_l3294_329412

def is_power (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m^k

def shorten (n : ℕ) : ℕ :=
  n / 10

theorem shortening_powers (n : ℕ) :
  n > 1000000 →
  is_power (shorten n) 2 →
  is_power (shorten (shorten n)) 3 →
  is_power (shorten (shorten (shorten n))) 4 →
  is_power (shorten (shorten (shorten (shorten n)))) 5 →
  is_power (shorten (shorten (shorten (shorten (shorten n))))) 6 :=
by sorry

end NUMINAMATH_CALUDE_shortening_powers_l3294_329412


namespace NUMINAMATH_CALUDE_exists_non_isosceles_equidistant_inscribed_center_l3294_329457

/-- A triangle with side lengths a, b, and c. -/
structure Triangle :=
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

/-- The center of the inscribed circle of a triangle. -/
def InscribedCenter (t : Triangle) : ℝ × ℝ := sorry

/-- The midpoint of a line segment. -/
def Midpoint (a b : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The distance between two points. -/
def Distance (a b : ℝ × ℝ) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles. -/
def IsIsosceles (t : Triangle) : Prop := 
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Theorem: There exists a non-isosceles triangle where the center of its inscribed circle
    is equidistant from the midpoints of two sides. -/
theorem exists_non_isosceles_equidistant_inscribed_center :
  ∃ (t : Triangle), 
    ¬IsIsosceles t ∧
    ∃ (s₁ s₂ : ℝ × ℝ), 
      Distance (InscribedCenter t) (Midpoint s₁ s₂) = 
      Distance (InscribedCenter t) (Midpoint s₂ (s₁.1 + t.a - s₁.1, s₁.2 + t.b - s₁.2)) :=
sorry

end NUMINAMATH_CALUDE_exists_non_isosceles_equidistant_inscribed_center_l3294_329457


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_arithmetic_relation_l3294_329453

/-- A geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => q * geometricSequence a q n

theorem geometric_sequence_ratio (a q : ℝ) (h : q ≠ 0) (h₁ : a ≠ 0) :
  ∀ n : ℕ, geometricSequence a q (n + 1) / geometricSequence a q n = q := by sorry

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmeticSequence a d n + d

theorem geometric_arithmetic_relation (a q : ℝ) (h : q ≠ 0) (h₁ : a ≠ 0) :
  (∃ d : ℝ, arithmeticSequence 1 d 0 = 1 ∧
            arithmeticSequence 1 d 1 = geometricSequence a q 1 ∧
            arithmeticSequence 1 d 2 = geometricSequence a q 2 - 1) →
  (geometricSequence a q 2 + geometricSequence a q 3) / (geometricSequence a q 4 + geometricSequence a q 5) = 1/4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_arithmetic_relation_l3294_329453


namespace NUMINAMATH_CALUDE_parabola_properties_l3294_329467

/-- A parabola with vertex at the origin, symmetric about the x-axis, passing through (-3, -6) -/
def parabola (x y : ℝ) : Prop := y^2 = -12*x

theorem parabola_properties :
  (parabola 0 0) ∧ 
  (∀ x y : ℝ, parabola x y → parabola x (-y)) ∧
  (parabola (-3) (-6)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3294_329467


namespace NUMINAMATH_CALUDE_treasure_gold_amount_l3294_329475

theorem treasure_gold_amount (total_mass : ℝ) (num_brothers : ℕ) 
  (eldest_gold : ℝ) (eldest_silver_fraction : ℝ) :
  total_mass = num_brothers * 100 →
  eldest_gold = 25 →
  eldest_silver_fraction = 1 / 8 →
  ∃ (total_gold total_silver : ℝ),
    total_gold + total_silver = total_mass ∧
    total_gold = 100 ∧
    eldest_gold + eldest_silver_fraction * total_silver = 100 :=
by sorry

end NUMINAMATH_CALUDE_treasure_gold_amount_l3294_329475


namespace NUMINAMATH_CALUDE_two_year_increase_l3294_329494

/-- Calculates the final amount after a given number of years with a fixed annual increase rate -/
def finalAmount (initialAmount : ℝ) (annualRate : ℝ) (years : ℕ) : ℝ :=
  initialAmount * (1 + annualRate) ^ years

/-- Theorem: An initial amount of 57600, increasing by 1/8 annually, becomes 72900 after 2 years -/
theorem two_year_increase : 
  finalAmount 57600 (1/8) 2 = 72900 := by sorry

end NUMINAMATH_CALUDE_two_year_increase_l3294_329494


namespace NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l3294_329497

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y + 100 = 0

/-- The center of a hyperbola -/
def hyperbola_center (h k : ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    hyperbola_equation x y ↔ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (3, 4) -/
theorem hyperbola_center_is_3_4 : hyperbola_center 3 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_3_4_l3294_329497


namespace NUMINAMATH_CALUDE_remaining_candies_josh_candy_distribution_l3294_329436

/-- Given Josh's initial candy distribution and his plans, calculate the remaining candies --/
theorem remaining_candies (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) 
  (candies_to_eat : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let remaining_after_friend := remaining_after_siblings / 2
  remaining_after_friend - candies_to_eat

/-- Prove that given the initial conditions, 19 candies are left to be shared --/
theorem josh_candy_distribution : 
  remaining_candies 100 3 10 16 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remaining_candies_josh_candy_distribution_l3294_329436


namespace NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_area_l3294_329488

theorem triangle_with_consecutive_sides_and_area :
  ∃ (a b c S : ℕ), 
    (a + 1 = b) ∧ 
    (b + 1 = c) ∧ 
    (c + 1 = S) ∧
    (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (S = 6) ∧
    (2 * S = a * b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_area_l3294_329488


namespace NUMINAMATH_CALUDE_jerry_feathers_left_l3294_329495

def hawk_feathers : ℕ := 6
def eagle_multiplier : ℕ := 17
def given_away : ℕ := 10

def total_feathers : ℕ := hawk_feathers + eagle_multiplier * hawk_feathers
def remaining_after_gift : ℕ := total_feathers - given_away
def sold : ℕ := remaining_after_gift / 2

theorem jerry_feathers_left : total_feathers - given_away - sold = 49 := by
  sorry

end NUMINAMATH_CALUDE_jerry_feathers_left_l3294_329495


namespace NUMINAMATH_CALUDE_circle_triangle_area_l3294_329432

/-- Given a circle C with center (a, 2/a) that passes through the origin (0, 0)
    and intersects the x-axis at (2a, 0) and the y-axis at (0, 4/a),
    prove that the area of the triangle formed by these three points is 4. -/
theorem circle_triangle_area (a : ℝ) (ha : a ≠ 0) : 
  let center : ℝ × ℝ := (a, 2/a)
  let origin : ℝ × ℝ := (0, 0)
  let point_A : ℝ × ℝ := (2*a, 0)
  let point_B : ℝ × ℝ := (0, 4/a)
  let triangle_area := abs ((point_A.1 - origin.1) * (point_B.2 - origin.2)) / 2
  triangle_area = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_l3294_329432


namespace NUMINAMATH_CALUDE_min_RS_value_l3294_329400

/-- Represents a rhombus ABCD with given diagonals -/
structure Rhombus where
  AC : ℝ
  BD : ℝ

/-- Represents a point M on side AB of the rhombus -/
structure PointM where
  BM : ℝ

/-- The minimum value of RS given the rhombus and point M -/
noncomputable def min_RS (r : Rhombus) (m : PointM) : ℝ :=
  Real.sqrt (8 * m.BM^2 - 40 * m.BM + 400)

/-- Theorem stating the minimum value of RS -/
theorem min_RS_value (r : Rhombus) : 
  r.AC = 24 → r.BD = 40 → ∃ (m : PointM), min_RS r m = 5 * Real.sqrt 14 := by
  sorry

#check min_RS_value

end NUMINAMATH_CALUDE_min_RS_value_l3294_329400


namespace NUMINAMATH_CALUDE_uphill_distance_l3294_329420

/-- Proves that the uphill distance is 45 km given the conditions of the problem -/
theorem uphill_distance (flat_speed : ℝ) (uphill_speed : ℝ) (extra_flat_distance : ℝ) :
  flat_speed = 20 →
  uphill_speed = 12 →
  extra_flat_distance = 30 →
  ∃ (uphill_distance : ℝ),
    uphill_distance / uphill_speed = (uphill_distance + extra_flat_distance) / flat_speed ∧
    uphill_distance = 45 :=
by sorry

end NUMINAMATH_CALUDE_uphill_distance_l3294_329420


namespace NUMINAMATH_CALUDE_price_of_added_toy_l3294_329485

/-- Given 5 toys with an average price of $10, adding one toy to make the new average $11 for 6 toys, prove the price of the added toy is $16. -/
theorem price_of_added_toy (num_toys : ℕ) (avg_price : ℚ) (new_num_toys : ℕ) (new_avg_price : ℚ) :
  num_toys = 5 →
  avg_price = 10 →
  new_num_toys = num_toys + 1 →
  new_avg_price = 11 →
  (new_num_toys : ℚ) * new_avg_price - (num_toys : ℚ) * avg_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_price_of_added_toy_l3294_329485


namespace NUMINAMATH_CALUDE_train_distance_problem_l3294_329451

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 60) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 540 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3294_329451


namespace NUMINAMATH_CALUDE_derivative_greater_than_average_rate_of_change_l3294_329448

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (1 - 2*a) * x - Real.log x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2*a*x + (1 - 2*a) - 1/x

-- Theorem statement
theorem derivative_greater_than_average_rate_of_change 
  (a : ℝ) (x0 x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x0 ≠ (x1 + x2) / 2) :
  f' a x0 > (f a x1 - f a x2) / (x1 - x2) := by
  sorry

end

end NUMINAMATH_CALUDE_derivative_greater_than_average_rate_of_change_l3294_329448


namespace NUMINAMATH_CALUDE_gift_budget_calculation_l3294_329437

theorem gift_budget_calculation (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) 
  (num_family : ℕ) : 
  total_budget = 200 → 
  num_friends = 12 → 
  friend_gift_cost = 15 → 
  num_family = 4 → 
  (total_budget - num_friends * friend_gift_cost) / num_family = 5 := by
sorry

end NUMINAMATH_CALUDE_gift_budget_calculation_l3294_329437


namespace NUMINAMATH_CALUDE_min_value_inequality_l3294_329490

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/(2*b) + 1/(3*c) = 1) : a + 2*b + 3*c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3294_329490


namespace NUMINAMATH_CALUDE_intersection_condition_for_singleton_zero_l3294_329469

theorem intersection_condition_for_singleton_zero (A : Set ℕ) :
  (A = {0} → A ∩ {0, 1} = {0}) ∧
  ∃ A : Set ℕ, A ∩ {0, 1} = {0} ∧ A ≠ {0} :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_for_singleton_zero_l3294_329469


namespace NUMINAMATH_CALUDE_range_of_a_l3294_329463

/-- Proposition p: For all x in ℝ, ax^2 + ax + 1 > 0 always holds -/
def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The function f(x) = 4x^2 - ax is monotonically increasing on [1, +∞) -/
def proposition_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → (4 * x^2 - a * x) < (4 * y^2 - a * y)

/-- The main theorem -/
theorem range_of_a :
  (∃ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ ¬proposition_p a) →
  (∃ a : ℝ, a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3294_329463


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_11_l3294_329438

theorem three_digit_number_divisible_by_11 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 6 ∧ (n / 100) % 10 = 3 ∧ n % 11 = 0 ∧ n = 396 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_11_l3294_329438


namespace NUMINAMATH_CALUDE_distribute_items_eq_36_l3294_329487

/-- The number of ways to distribute 4 distinct items into 3 non-empty groups -/
def distribute_items : ℕ :=
  (Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1) * Nat.factorial 3 / Nat.factorial 2

/-- Theorem stating that the number of ways to distribute 4 distinct items
    into 3 non-empty groups is 36 -/
theorem distribute_items_eq_36 : distribute_items = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_items_eq_36_l3294_329487


namespace NUMINAMATH_CALUDE_initial_forks_count_l3294_329460

theorem initial_forks_count (F : ℚ) : 
  (F + 2) + (F + 11) + (2 * F + 20) + (F / 2 + 2) = 62 ↔ F = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_forks_count_l3294_329460


namespace NUMINAMATH_CALUDE_max_m_value_l3294_329492

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ m : ℝ, m / (3 * a + b) - 3 / a - 1 / b ≤ 0) →
  (∃ m : ℝ, m = 16 ∧ ∀ n : ℝ, (n / (3 * a + b) - 3 / a - 1 / b ≤ 0) → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3294_329492


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3294_329422

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l3294_329422


namespace NUMINAMATH_CALUDE_window_cost_is_700_l3294_329417

/-- The cost of damages caused by Jack -/
def total_damage : ℕ := 1450

/-- The number of tires damaged -/
def num_tires : ℕ := 3

/-- The cost of each tire -/
def tire_cost : ℕ := 250

/-- The cost of the window -/
def window_cost : ℕ := total_damage - (num_tires * tire_cost)

theorem window_cost_is_700 : window_cost = 700 := by
  sorry

end NUMINAMATH_CALUDE_window_cost_is_700_l3294_329417


namespace NUMINAMATH_CALUDE_f_sum_negative_l3294_329493

/-- A function f satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y)

/-- The main theorem -/
theorem f_sum_negative
  (f : ℝ → ℝ)
  (h_f : f_properties f)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ < 0)
  (h_prod : x₁ * x₂ < 0) :
  f x₁ + f x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_f_sum_negative_l3294_329493


namespace NUMINAMATH_CALUDE_ball_purchase_theorem_l3294_329468

/-- Represents the cost and quantity of balls in two purchases -/
structure BallPurchase where
  soccer_price : ℝ
  volleyball_price : ℝ
  soccer_quantity1 : ℕ
  volleyball_quantity1 : ℕ
  total_cost1 : ℝ
  total_quantity2 : ℕ
  soccer_price_increase : ℝ
  volleyball_price_decrease : ℝ
  total_cost2_ratio : ℝ

/-- Theorem stating the prices of balls and the quantity of volleyballs in the second purchase -/
theorem ball_purchase_theorem (bp : BallPurchase)
  (h1 : bp.soccer_quantity1 * bp.soccer_price + bp.volleyball_quantity1 * bp.volleyball_price = bp.total_cost1)
  (h2 : bp.soccer_price = bp.volleyball_price + 30)
  (h3 : bp.soccer_quantity1 = 40)
  (h4 : bp.volleyball_quantity1 = 30)
  (h5 : bp.total_cost1 = 4000)
  (h6 : bp.total_quantity2 = 50)
  (h7 : bp.soccer_price_increase = 0.1)
  (h8 : bp.volleyball_price_decrease = 0.1)
  (h9 : bp.total_cost2_ratio = 0.86) :
  bp.soccer_price = 70 ∧ bp.volleyball_price = 40 ∧
  ∃ m : ℕ, m = 10 ∧ 
    (bp.total_quantity2 - m) * (bp.soccer_price * (1 + bp.soccer_price_increase)) +
    m * (bp.volleyball_price * (1 - bp.volleyball_price_decrease)) =
    bp.total_cost1 * bp.total_cost2_ratio :=
by sorry

end NUMINAMATH_CALUDE_ball_purchase_theorem_l3294_329468


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_property_three_digit_numbers_with_property_exists_infinite_sequence_l3294_329477

-- Define a function to check if a number has the desired property
def has_property (n : ℕ) (base : ℕ) : Prop :=
  n^2 % base = n

-- Theorem for two-digit numbers
theorem two_digit_numbers_with_property :
  ∃ (A B : ℕ), A ≠ B ∧ 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧
  has_property A 100 ∧ has_property B 100 ∧
  ∀ (C : ℕ), 10 ≤ C ∧ C < 100 ∧ has_property C 100 → (C = A ∨ C = B) :=
sorry

-- Theorem for three-digit numbers
theorem three_digit_numbers_with_property :
  ∃ (A B : ℕ), A ≠ B ∧ 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
  has_property A 1000 ∧ has_property B 1000 ∧
  ∀ (C : ℕ), 100 ≤ C ∧ C < 1000 ∧ has_property C 1000 → (C = A ∨ C = B) :=
sorry

-- Define a function to represent a number from a sequence of digits
def number_from_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * 10 + a (n - 1 - i)) 0

-- Theorem for the existence of an infinite sequence
theorem exists_infinite_sequence :
  ∃ (a : ℕ → ℕ), ∀ (n : ℕ), has_property (number_from_sequence a n) (10^n) ∧
  ¬(a 0 = 1 ∧ ∀ (k : ℕ), k > 0 → a k = 0) :=
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_property_three_digit_numbers_with_property_exists_infinite_sequence_l3294_329477


namespace NUMINAMATH_CALUDE_battle_station_staffing_l3294_329450

/-- The number of job openings in the battle station -/
def num_jobs : ℕ := 5

/-- The total number of applicants -/
def total_applicants : ℕ := 30

/-- The number of suitable applicants -/
def suitable_applicants : ℕ := total_applicants - (total_applicants / 3)

/-- The number of candidates qualified for the Radio Specialist role -/
def radio_qualified : ℕ := 5

/-- The number of ways to staff the battle station -/
def staffing_ways : ℕ := radio_qualified * (suitable_applicants - 1) * (suitable_applicants - 2) * (suitable_applicants - 3) * (suitable_applicants - 4)

theorem battle_station_staffing :
  staffing_ways = 292320 :=
sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l3294_329450


namespace NUMINAMATH_CALUDE_last_two_digits_of_sequence_sum_l3294_329418

/-- The sum of the sequence 8, 88, 888, ..., up to 2008 digits -/
def sequence_sum : ℕ := 8 + 88 * 2007

/-- The last two digits of a number -/
def last_two_digits (n : ℕ) : ℕ := n % 100

/-- Theorem: The last two digits of the sequence sum are 24 -/
theorem last_two_digits_of_sequence_sum :
  last_two_digits sequence_sum = 24 := by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sequence_sum_l3294_329418


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l3294_329462

/-- A circle inscribed in a quadrilateral EFGH -/
structure InscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The point where the circle is tangent to EF -/
  R : Point
  /-- The point where the circle is tangent to GH -/
  S : Point
  /-- The length of ER -/
  ER : ℝ
  /-- The length of RF -/
  RF : ℝ
  /-- The length of GS -/
  GS : ℝ
  /-- The length of SH -/
  SH : ℝ

/-- Theorem: The square of the radius of the inscribed circle is 868 -/
theorem inscribed_circle_radius_squared (c : InscribedCircle)
    (h1 : c.ER = 21)
    (h2 : c.RF = 28)
    (h3 : c.GS = 40)
    (h4 : c.SH = 32) :
    c.r^2 = 868 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_squared_l3294_329462


namespace NUMINAMATH_CALUDE_product_of_roots_l3294_329409

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → ∃ (x₁ x₂ : ℝ), x₁ * x₂ = -30 ∧ (x₁ + 3) * (x₁ - 4) = 18 ∧ (x₂ + 3) * (x₂ - 4) = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3294_329409


namespace NUMINAMATH_CALUDE_sin_75_degrees_l3294_329472

theorem sin_75_degrees : 
  Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  -- Define known values
  have sin_45 : Real.sin (45 * π / 180) = Real.sqrt 2 / 2 := sorry
  have cos_45 : Real.cos (45 * π / 180) = Real.sqrt 2 / 2 := sorry
  have sin_30 : Real.sin (30 * π / 180) = 1 / 2 := sorry
  have cos_30 : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := sorry

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l3294_329472


namespace NUMINAMATH_CALUDE_old_machine_rate_proof_l3294_329480

/-- The rate at which the new machine makes bolts (in bolts per hour) -/
def new_machine_rate : ℝ := 150

/-- The time both machines work together (in hours) -/
def work_time : ℝ := 2

/-- The total number of bolts produced by both machines -/
def total_bolts : ℝ := 500

/-- The rate at which the old machine makes bolts (in bolts per hour) -/
def old_machine_rate : ℝ := 100

theorem old_machine_rate_proof :
  old_machine_rate * work_time + new_machine_rate * work_time = total_bolts :=
by sorry

end NUMINAMATH_CALUDE_old_machine_rate_proof_l3294_329480


namespace NUMINAMATH_CALUDE_sum_of_squares_l3294_329423

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_seven : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = -6/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3294_329423


namespace NUMINAMATH_CALUDE_george_second_half_correct_l3294_329403

def trivia_game (first_half_correct : ℕ) (points_per_question : ℕ) (final_score : ℕ) : ℕ :=
  (final_score - first_half_correct * points_per_question) / points_per_question

theorem george_second_half_correct :
  trivia_game 6 3 30 = 4 :=
sorry

end NUMINAMATH_CALUDE_george_second_half_correct_l3294_329403


namespace NUMINAMATH_CALUDE_train_length_l3294_329458

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), 
  abs (length - (speed * (5/18) * time)) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3294_329458


namespace NUMINAMATH_CALUDE_repeating_decimal_35_eq_fraction_l3294_329411

/-- Represents a repeating decimal where the digits 35 repeat infinitely after the decimal point. -/
def repeating_decimal_35 : ℚ :=
  35 / 99

/-- The theorem states that the repeating decimal 0.353535... is equal to the fraction 35/99. -/
theorem repeating_decimal_35_eq_fraction :
  repeating_decimal_35 = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_35_eq_fraction_l3294_329411


namespace NUMINAMATH_CALUDE_point_on_line_l3294_329499

/-- A point (x, y) lies on the line y = 2x + 1 if y equals 2x + 1 -/
def lies_on_line (x y : ℚ) : Prop := y = 2 * x + 1

/-- The point (-2, -3) lies on the line y = 2x + 1 -/
theorem point_on_line : lies_on_line (-2) (-3) := by sorry

end NUMINAMATH_CALUDE_point_on_line_l3294_329499


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3294_329454

theorem sphere_surface_area (R : ℝ) : 
  R > 0 → 
  (∃ (x : ℝ), x > 0 ∧ x < R ∧ 
    (∀ (y : ℝ), y > 0 → y < R → 
      2 * π * x^2 * (2 * Real.sqrt (R^2 - x^2)) ≥ 2 * π * y^2 * (2 * Real.sqrt (R^2 - y^2)))) →
  2 * π * R * (2 * Real.sqrt (R^2 - (R * Real.sqrt 6 / 3)^2)) = 16 * Real.sqrt 2 * π →
  4 * π * R^2 = 48 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3294_329454


namespace NUMINAMATH_CALUDE_equation_solution_l3294_329496

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((2 * x - 5) / (x - 2) = (3 * x - 3) / (x - 2) - 3) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3294_329496


namespace NUMINAMATH_CALUDE_steves_day_assignments_l3294_329459

/-- Given that Steve's day is divided into sleeping, school, family time, and assignments,
    prove that the fraction of the day spent on assignments is 1/12. -/
theorem steves_day_assignments (total_hours : ℝ) (sleep_fraction : ℝ) (school_fraction : ℝ) 
    (family_hours : ℝ) (assignment_fraction : ℝ) : 
    total_hours = 24 ∧ 
    sleep_fraction = 1/3 ∧ 
    school_fraction = 1/6 ∧ 
    family_hours = 10 ∧ 
    sleep_fraction + school_fraction + (family_hours / total_hours) + assignment_fraction = 1 →
    assignment_fraction = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_steves_day_assignments_l3294_329459


namespace NUMINAMATH_CALUDE_apples_distribution_l3294_329429

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 5

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 2 * benny_apples

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of apples each friend received -/
def apples_per_friend : ℕ := total_apples / num_friends

theorem apples_distribution :
  apples_per_friend = 5 :=
sorry

end NUMINAMATH_CALUDE_apples_distribution_l3294_329429


namespace NUMINAMATH_CALUDE_selene_and_tanya_spend_16_l3294_329491

/-- Represents the prices of items in the school canteen -/
structure CanteenPrices where
  sandwich : ℕ
  hamburger : ℕ
  hotdog : ℕ
  fruitJuice : ℕ

/-- Represents an order in the canteen -/
structure Order where
  sandwiches : ℕ
  hamburgers : ℕ
  hotdogs : ℕ
  fruitJuices : ℕ

/-- Calculates the total cost of an order given the prices -/
def orderCost (prices : CanteenPrices) (order : Order) : ℕ :=
  prices.sandwich * order.sandwiches +
  prices.hamburger * order.hamburgers +
  prices.hotdog * order.hotdogs +
  prices.fruitJuice * order.fruitJuices

/-- The main theorem stating that Selene and Tanya spend $16 together -/
theorem selene_and_tanya_spend_16 (prices : CanteenPrices) 
    (seleneOrder : Order) (tanyaOrder : Order) : 
    prices.sandwich = 2 → 
    prices.hamburger = 2 → 
    prices.hotdog = 1 → 
    prices.fruitJuice = 2 → 
    seleneOrder.sandwiches = 3 → 
    seleneOrder.fruitJuices = 1 → 
    seleneOrder.hamburgers = 0 → 
    seleneOrder.hotdogs = 0 → 
    tanyaOrder.hamburgers = 2 → 
    tanyaOrder.fruitJuices = 2 → 
    tanyaOrder.sandwiches = 0 → 
    tanyaOrder.hotdogs = 0 → 
    orderCost prices seleneOrder + orderCost prices tanyaOrder = 16 := by
  sorry

end NUMINAMATH_CALUDE_selene_and_tanya_spend_16_l3294_329491


namespace NUMINAMATH_CALUDE_bus_distance_theorem_l3294_329424

/-- Calculates the total distance traveled by a bus with increasing speed over a given number of hours -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initialSpeed + (hours - 1) * speedIncrease) / 2

/-- Theorem stating that a bus with given initial speed and speed increase travels a specific distance in 12 hours -/
theorem bus_distance_theorem (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) :
  initialSpeed = 35 →
  speedIncrease = 2 →
  hours = 12 →
  totalDistance initialSpeed speedIncrease hours = 552 := by
  sorry

#eval totalDistance 35 2 12  -- This should evaluate to 552

end NUMINAMATH_CALUDE_bus_distance_theorem_l3294_329424


namespace NUMINAMATH_CALUDE_largest_number_from_digits_l3294_329482

def digits : List Nat := [6, 3]

theorem largest_number_from_digits :
  (63 : Nat) = digits.foldl (fun acc d => acc * 10 + d) 0 ∧
  ∀ (n : Nat), n ≠ 63 → n < 63 ∨ ¬(∃ (perm : List Nat), perm.Perm digits ∧ n = perm.foldl (fun acc d => acc * 10 + d) 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_from_digits_l3294_329482


namespace NUMINAMATH_CALUDE_circle_intersection_condition_l3294_329428

-- Define the circles B and C
def circle_B (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + b = 0}

def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 16 = 0}

-- Define the condition for no common points
def no_common_points (b : ℝ) : Prop :=
  circle_B b ∩ circle_C = ∅

-- State the theorem
theorem circle_intersection_condition :
  ∀ b : ℝ, no_common_points b ↔ (-4 < b ∧ b < 0) ∨ b < -64 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_l3294_329428


namespace NUMINAMATH_CALUDE_percentage_of_500_l3294_329476

/-- Prove that 25% of Rs. 500 is equal to Rs. 125 -/
theorem percentage_of_500 : (500 : ℝ) * 0.25 = 125 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_500_l3294_329476


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l3294_329406

def total_potatoes : ℕ := 13
def cooked_potatoes : ℕ := 5
def cooking_time_per_potato : ℕ := 6

theorem remaining_cooking_time : (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l3294_329406


namespace NUMINAMATH_CALUDE_marble_draw_theorem_l3294_329483

/-- Represents the number of marbles of each color in the bucket -/
structure MarbleCounts where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  orange : Nat
  purple : Nat

/-- The actual counts of marbles in the bucket -/
def initialCounts : MarbleCounts :=
  { red := 35, green := 25, blue := 24, yellow := 18, orange := 15, purple := 12 }

/-- The minimum number of marbles to guarantee at least 20 of a single color -/
def minMarblesToDraw : Nat := 103

theorem marble_draw_theorem (counts : MarbleCounts := initialCounts) :
  (∀ n : Nat, n < minMarblesToDraw →
    ∃ c : MarbleCounts, c.red < 20 ∧ c.green < 20 ∧ c.blue < 20 ∧
      c.yellow < 20 ∧ c.orange < 20 ∧ c.purple < 20 ∧
      c.red + c.green + c.blue + c.yellow + c.orange + c.purple = n) ∧
  (∀ c : MarbleCounts,
    c.red + c.green + c.blue + c.yellow + c.orange + c.purple = minMarblesToDraw →
    c.red ≥ 20 ∨ c.green ≥ 20 ∨ c.blue ≥ 20 ∨ c.yellow ≥ 20 ∨ c.orange ≥ 20 ∨ c.purple ≥ 20) :=
by sorry

end NUMINAMATH_CALUDE_marble_draw_theorem_l3294_329483


namespace NUMINAMATH_CALUDE_grid_toothpick_count_l3294_329439

/-- Calculates the number of toothpicks in a grid with missing pieces -/
def toothpick_count (length width missing_vertical missing_horizontal : ℕ) : ℕ :=
  let vertical_lines := length + 1 - missing_vertical
  let horizontal_lines := width + 1 - missing_horizontal
  (vertical_lines * width) + (horizontal_lines * length)

/-- Theorem stating the correct number of toothpicks in the specific grid -/
theorem grid_toothpick_count :
  toothpick_count 45 25 8 5 = 1895 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpick_count_l3294_329439


namespace NUMINAMATH_CALUDE_distinguishable_triangles_count_l3294_329433

/-- The number of available colors for the triangles -/
def num_colors : ℕ := 8

/-- The number of corner triangles in the large triangle -/
def num_corners : ℕ := 3

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := num_corners + 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  (num_colors + 
   num_colors * (num_colors - 1) + 
   choose num_colors num_corners) * num_colors

theorem distinguishable_triangles_count :
  num_distinguishable_triangles = 960 :=
sorry

end NUMINAMATH_CALUDE_distinguishable_triangles_count_l3294_329433


namespace NUMINAMATH_CALUDE_minimum_interval_for_f_l3294_329416

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem minimum_interval_for_f (t s : ℝ) (h : f t = g s) :
  ∃ (a : ℝ), (a > (1/2) ∧ a < Real.log 2) ∧
  (∀ (t' s' : ℝ), f t' = g s' → s' - t' ≥ s - t → f t = a) :=
sorry

end NUMINAMATH_CALUDE_minimum_interval_for_f_l3294_329416


namespace NUMINAMATH_CALUDE_root_product_expression_l3294_329471

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α - 1 = 0) → 
  (β^2 + p*β - 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = p^2 - q^2 := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l3294_329471


namespace NUMINAMATH_CALUDE_second_grade_volunteers_l3294_329413

/-- Given a total population and a subgroup, calculate the proportion of volunteers
    to be selected from the subgroup in a stratified random sampling. -/
def stratified_sampling_proportion (total_population : ℕ) (subgroup : ℕ) (total_volunteers : ℕ) : ℕ :=
  (subgroup * total_volunteers) / total_population

/-- Prove that in a stratified random sampling of 30 volunteers from a population of 3000 students,
    where 1000 students are in the second grade, the number of volunteers to be selected from
    the second grade is 10. -/
theorem second_grade_volunteers :
  stratified_sampling_proportion 3000 1000 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_volunteers_l3294_329413


namespace NUMINAMATH_CALUDE_closest_point_on_line_l3294_329489

def v (t : ℝ) : ℝ × ℝ × ℝ := (3 + 8*t, -2 + 6*t, -4 - 2*t)

def a : ℝ × ℝ × ℝ := (5, 7, 3)

def direction : ℝ × ℝ × ℝ := (8, 6, -2)

theorem closest_point_on_line (t : ℝ) : 
  (t = 7/13) ↔ 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) :=
sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l3294_329489


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3294_329404

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3294_329404


namespace NUMINAMATH_CALUDE_simplify_fraction_l3294_329478

/-- Given x = 3 and y = 4, prove that (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 -/
theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3294_329478


namespace NUMINAMATH_CALUDE_mrs_brown_shoe_price_l3294_329442

/-- Calculates the final price for a Mother's Day purchase with additional discount for multiple children -/
def mothersDayPrice (originalPrice : ℝ) (numChildren : ℕ) : ℝ :=
  let mothersDayDiscount := 0.1
  let additionalDiscount := 0.04
  let discountedPrice := originalPrice * (1 - mothersDayDiscount)
  if numChildren ≥ 3 then
    discountedPrice * (1 - additionalDiscount)
  else
    discountedPrice

theorem mrs_brown_shoe_price :
  mothersDayPrice 125 4 = 108 := by
  sorry

end NUMINAMATH_CALUDE_mrs_brown_shoe_price_l3294_329442


namespace NUMINAMATH_CALUDE_exponential_inequality_l3294_329441

theorem exponential_inequality (a b : ℝ) (h : a > b) : (2 : ℝ) ^ a > (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3294_329441


namespace NUMINAMATH_CALUDE_twelve_million_plus_twelve_thousand_l3294_329415

theorem twelve_million_plus_twelve_thousand : 
  12000000 + 12000 = 12012000 := by
  sorry

end NUMINAMATH_CALUDE_twelve_million_plus_twelve_thousand_l3294_329415


namespace NUMINAMATH_CALUDE_red_shirt_pairs_red_shirt_pairs_correct_l3294_329465

theorem red_shirt_pairs (green_students : ℕ) (red_students : ℕ) (total_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) : ℕ :=
  by
  have h1 : green_students = 67 := by sorry
  have h2 : red_students = 89 := by sorry
  have h3 : total_students = 156 := by sorry
  have h4 : total_pairs = 78 := by sorry
  have h5 : green_green_pairs = 25 := by sorry
  have h6 : total_students = green_students + red_students := by sorry
  have h7 : green_students + red_students = total_pairs * 2 := by sorry

  -- The number of pairs where both students are wearing red shirts
  sorry

theorem red_shirt_pairs_correct : red_shirt_pairs 67 89 156 78 25 = 36 := by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_red_shirt_pairs_correct_l3294_329465


namespace NUMINAMATH_CALUDE_extreme_values_imply_b_l3294_329427

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_imply_b (a b : ℝ) :
  (f' a b 1 = 0) → (f' a b 2 = 0) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_imply_b_l3294_329427


namespace NUMINAMATH_CALUDE_discount_difference_l3294_329405

def initial_amount : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_amount 0.25) 0.1) 0.05

def option2_price : ℝ :=
  apply_discount (apply_discount (apply_discount initial_amount 0.3) 0.1) 0.1

theorem discount_difference :
  option1_price - option2_price = 1113.75 := by sorry

end NUMINAMATH_CALUDE_discount_difference_l3294_329405


namespace NUMINAMATH_CALUDE_commute_problem_l3294_329425

theorem commute_problem (y : ℕ) 
  (morning_bike : ℕ) (afternoon_bike : ℕ) (tram_commutes : ℕ)
  (h1 : morning_bike = 10)
  (h2 : afternoon_bike = 13)
  (h3 : tram_commutes = 11)
  (h4 : y = morning_bike + afternoon_bike - tram_commutes / 2) :
  y = 17 := by
  sorry

end NUMINAMATH_CALUDE_commute_problem_l3294_329425


namespace NUMINAMATH_CALUDE_function_inequality_equivalence_l3294_329444

theorem function_inequality_equivalence 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = 3 * (x + 2)^2 - 1) 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  (∀ x, |x + 2| < b → |f x - 7| < a) ↔ b^2 = (8 + a) / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_equivalence_l3294_329444


namespace NUMINAMATH_CALUDE_area_of_section_ABD_l3294_329408

theorem area_of_section_ABD (a : ℝ) (S : ℝ) (V : ℝ) : 
  a > 0 → 0 < S → S < π / 2 → V > 0 →
  let area_ABD := (Real.sqrt 3 / Real.sin S) * (V ^ (2 / 3) * Real.tan S) ^ (1 / 3)
  ∃ (h : ℝ), h > 0 ∧ 
    V = (a ^ 3 / 8) * Real.tan S ∧
    area_ABD = (a ^ 2 * Real.sqrt 3) / (4 * Real.cos S) :=
by sorry

#check area_of_section_ABD

end NUMINAMATH_CALUDE_area_of_section_ABD_l3294_329408


namespace NUMINAMATH_CALUDE_rational_sqrt_n_minus_3_over_n_plus_1_l3294_329414

theorem rational_sqrt_n_minus_3_over_n_plus_1 
  (r q n : ℚ) 
  (h : 1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q)) :
  ∃ (a b : ℚ), b ≠ 0 ∧ (n - 3) / (n + 1) = (a / b) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_rational_sqrt_n_minus_3_over_n_plus_1_l3294_329414


namespace NUMINAMATH_CALUDE_percentage_problem_l3294_329419

theorem percentage_problem (x y : ℝ) : 
  x = 0.18 * 4750 →
  y = 1.3 * x →
  y / 8950 * 100 = 12.42 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3294_329419


namespace NUMINAMATH_CALUDE_parallel_vectors_proportion_l3294_329440

-- Define the Cartesian coordinate system
def Point := ℝ × ℝ

-- Define points A and B
def A : Point := (-1, -2)
def B : Point := (2, 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector a
def a (k : ℝ) : ℝ × ℝ := (1, k)

-- Theorem statement
theorem parallel_vectors_proportion :
  ∃ k : ℝ, a k = (1, k) ∧ ∃ c : ℝ, c • (AB.1, AB.2) = a k :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_proportion_l3294_329440


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3294_329455

/-- Calculates the annual rent per square foot for a shop given its dimensions and monthly rent -/
theorem annual_rent_per_square_foot
  (length : ℝ) (width : ℝ) (monthly_rent : ℝ)
  (h1 : length = 18)
  (h2 : width = 20)
  (h3 : monthly_rent = 3600) :
  monthly_rent * 12 / (length * width) = 120 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l3294_329455


namespace NUMINAMATH_CALUDE_triangle_side_length_l3294_329456

/-- Given a triangle ABC where ∠B = 45°, AB = 100, and AC = 100√2, prove that BC = 100√(5 + √2(√6 - √2)). -/
theorem triangle_side_length (A B C : ℝ×ℝ) : 
  let angleB := Real.arccos ((B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))
  let sideAB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let sideAC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  angleB = π/4 ∧ sideAB = 100 ∧ sideAC = 100 * Real.sqrt 2 →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 100 * Real.sqrt (5 + Real.sqrt 2 * (Real.sqrt 6 - Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3294_329456


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l3294_329447

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  q > 0 ∧ r > 0 ∧ 
  1013 = 23 * q + r ∧
  ∀ (q' r' : ℕ), q' > 0 → r' > 0 → 1013 = 23 * q' + r' → q' - r' ≤ q - r ∧
  q - r = 39 := by
sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l3294_329447
