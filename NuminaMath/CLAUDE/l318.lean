import Mathlib

namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l318_31813

theorem fourth_number_in_sequence (a b c d : ℝ) : 
  a / b = 5 / 3 ∧ 
  b / c = 3 / 4 ∧ 
  a + b + c = 108 ∧ 
  d - c = c - b ∧ 
  c - b = b - a 
  → d = 45 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l318_31813


namespace NUMINAMATH_CALUDE_factorization_equality_l318_31806

theorem factorization_equality (x y : ℝ) :
  (5 * x - 4 * y) * (x + 2 * y) = 5 * x^2 + 6 * x * y - 8 * y^2 := by sorry

end NUMINAMATH_CALUDE_factorization_equality_l318_31806


namespace NUMINAMATH_CALUDE_sin_cos_45_degrees_l318_31828

theorem sin_cos_45_degrees : 
  Real.sin (π / 4) = 1 / Real.sqrt 2 ∧ Real.cos (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_45_degrees_l318_31828


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l318_31864

/-- A right-angled triangle with one angle of 30° and base length of 6 units has a base length of 6 units. -/
theorem right_triangle_base_length (a b c : ℝ) (θ : ℝ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  θ = π/6 →  -- 30° angle in radians
  a = 6 →  -- base length
  a = 6 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_base_length_l318_31864


namespace NUMINAMATH_CALUDE_min_age_difference_proof_l318_31863

/-- The number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- The number of days in a common year -/
def daysInCommonYear : ℕ := 365

/-- The year Adil was born -/
def adilBirthYear : ℕ := 2015

/-- The year Bav was born -/
def bavBirthYear : ℕ := 2018

/-- The minimum age difference in days between Adil and Bav -/
def minAgeDifference : ℕ := daysInLeapYear + daysInCommonYear + 1

theorem min_age_difference_proof :
  minAgeDifference = 732 :=
sorry

end NUMINAMATH_CALUDE_min_age_difference_proof_l318_31863


namespace NUMINAMATH_CALUDE_jakes_desired_rate_l318_31879

/-- Jake's hourly rate for planting flowers -/
def jakes_hourly_rate (total_charge : ℚ) (hours_worked : ℚ) : ℚ :=
  total_charge / hours_worked

/-- Theorem: Jake's hourly rate for planting flowers is $22.50 -/
theorem jakes_desired_rate :
  jakes_hourly_rate 45 2 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_jakes_desired_rate_l318_31879


namespace NUMINAMATH_CALUDE_equations_equivalence_l318_31854

-- Define the function types
variable {X : Type} [Nonempty X]
variable (f₁ f₂ f₃ f₄ : X → ℝ)

-- Define the equations
def eq1 (x : X) := f₁ x / f₂ x = f₃ x / f₄ x
def eq2 (x : X) := f₁ x / f₂ x = (f₁ x + f₃ x) / (f₂ x + f₄ x)

-- Define the conditions
def cond1 (x : X) := eq1 f₁ f₂ f₃ f₄ x → f₂ x + f₄ x ≠ 0
def cond2 (x : X) := eq2 f₁ f₂ f₃ f₄ x → f₄ x ≠ 0

-- State the theorem
theorem equations_equivalence :
  (∀ x, eq1 f₁ f₂ f₃ f₄ x ↔ eq2 f₁ f₂ f₃ f₄ x) ↔
  (∀ x, cond1 f₁ f₂ f₃ f₄ x ∧ cond2 f₁ f₂ f₃ f₄ x) :=
sorry

end NUMINAMATH_CALUDE_equations_equivalence_l318_31854


namespace NUMINAMATH_CALUDE_odd_divisibility_l318_31803

theorem odd_divisibility (n : ℕ) (h : Odd (94 * n)) :
  ∃ k : ℕ, n * (n - 1) ^ ((n - 1) ^ n + 1) + n = k * ((n - 1) ^ n + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisibility_l318_31803


namespace NUMINAMATH_CALUDE_complex_equality_sum_l318_31808

theorem complex_equality_sum (a b : ℝ) : 
  (Complex.mk a b = (2 * Complex.I) / (1 + Complex.I)) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l318_31808


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_progression_l318_31835

/-- Given an arithmetic progression with the 25th term equal to 173 and a common difference of 7,
    prove that the first term is 5. -/
theorem first_term_of_arithmetic_progression :
  ∀ (a : ℕ → ℤ),
    (∀ n : ℕ, a (n + 1) = a n + 7) →  -- Common difference is 7
    a 25 = 173 →                      -- 25th term is 173
    a 1 = 5 :=                        -- First term is 5
by
  sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_progression_l318_31835


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l318_31861

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 4 → x ≥ 4) ∧ (∃ x, x ≥ 4 ∧ ¬(x > 4)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l318_31861


namespace NUMINAMATH_CALUDE_fish_caught_l318_31874

theorem fish_caught (initial_fish : ℕ) (initial_tadpoles : ℕ) (fish_caught : ℕ) : 
  initial_fish = 50 →
  initial_tadpoles = 3 * initial_fish →
  initial_tadpoles / 2 = (initial_fish - fish_caught) + 32 →
  fish_caught = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fish_caught_l318_31874


namespace NUMINAMATH_CALUDE_special_triangle_third_side_l318_31883

/-- Triangle sides satisfy the given conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  side_condition : Real.sqrt (a - 9) + (b - 2)^2 = 0
  c_odd : ∃ (k : ℤ), c = 2 * k + 1

/-- The third side of the special triangle is 9 -/
theorem special_triangle_third_side (t : SpecialTriangle) : t.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_third_side_l318_31883


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l318_31878

theorem complex_square_one_plus_i : (1 + Complex.I) ^ 2 = 2 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l318_31878


namespace NUMINAMATH_CALUDE_total_packs_bought_l318_31810

/-- The number of index card packs John buys for each student -/
def packs_per_student : ℕ := 3

/-- The number of students in the first class -/
def class1_students : ℕ := 20

/-- The number of students in the second class -/
def class2_students : ℕ := 25

/-- The number of students in the third class -/
def class3_students : ℕ := 18

/-- The number of students in the fourth class -/
def class4_students : ℕ := 22

/-- The number of students in the fifth class -/
def class5_students : ℕ := 15

/-- The total number of students across all classes -/
def total_students : ℕ := class1_students + class2_students + class3_students + class4_students + class5_students

/-- Theorem: The total number of index card packs bought by John is 300 -/
theorem total_packs_bought : packs_per_student * total_students = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_bought_l318_31810


namespace NUMINAMATH_CALUDE_janet_dresses_l318_31894

/-- The number of dresses Janet has -/
def total_dresses : ℕ := 24

/-- The number of pockets in Janet's dresses -/
def total_pockets : ℕ := 32

theorem janet_dresses :
  (total_dresses / 2 / 3 * 2 + total_dresses / 2 * 2 / 3 * 3 = total_pockets) ∧
  (total_dresses > 0) := by
  sorry

end NUMINAMATH_CALUDE_janet_dresses_l318_31894


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l318_31862

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m}

theorem subset_implies_m_values (m : ℝ) :
  B m ⊆ A m → m = 1 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l318_31862


namespace NUMINAMATH_CALUDE_original_price_from_profit_and_selling_price_l318_31836

/-- Given an article sold at a 10% profit with a selling price of 550, 
    the original price of the article is 500. -/
theorem original_price_from_profit_and_selling_price :
  ∀ (original_price selling_price : ℝ),
    selling_price = 550 →
    selling_price = original_price * 1.1 →
    original_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_original_price_from_profit_and_selling_price_l318_31836


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l318_31848

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))
  (vertex_count : vertices.card = 20)
  (edge_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 5)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (I : Icosahedron) : ℚ :=
  (I.edges.card : ℚ) / (I.vertices.card.choose 2 : ℚ)

/-- The main theorem stating the probability is 10/19 -/
theorem icosahedron_edge_probability (I : Icosahedron) :
  edge_probability I = 10 / 19 := by
  sorry

end NUMINAMATH_CALUDE_icosahedron_edge_probability_l318_31848


namespace NUMINAMATH_CALUDE_three_color_theorem_l318_31873

theorem three_color_theorem (a b : ℕ) : 
  ∃ (f : ℤ → Fin 3), ∀ x : ℤ, f x ≠ f (x + a) ∧ f x ≠ f (x + b) := by
  sorry

end NUMINAMATH_CALUDE_three_color_theorem_l318_31873


namespace NUMINAMATH_CALUDE_fixed_point_theorem_dot_product_range_l318_31805

-- Define the curves and line
def curve_C (x y : ℝ) : Prop := y^2 = 4*x
def curve_M (x y : ℝ) : Prop := (x-1)^2 + y^2 = 4 ∧ x ≥ 1
def line_l (m n x y : ℝ) : Prop := x = m*y + n

-- Define the dot product
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

-- Part I
theorem fixed_point_theorem (m n x1 y1 x2 y2 : ℝ) :
  curve_C x1 y1 ∧ curve_C x2 y2 ∧
  line_l m n x1 y1 ∧ line_l m n x2 y2 ∧
  dot_product x1 y1 x2 y2 = -4 →
  n = 2 :=
sorry

-- Part II
theorem dot_product_range (m n x1 y1 x2 y2 : ℝ) :
  curve_C x1 y1 ∧ curve_C x2 y2 ∧
  line_l m n x1 y1 ∧ line_l m n x2 y2 ∧
  curve_M 1 0 ∧
  (∀ x y, curve_M x y → ¬(line_l m n x y ∧ (x, y) ≠ (1, 0))) →
  dot_product (x1-1) y1 (x2-1) y2 ≤ -8 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_dot_product_range_l318_31805


namespace NUMINAMATH_CALUDE_polynomial_value_l318_31877

theorem polynomial_value (x : ℝ) (h : x = (1 + Real.sqrt 1994) / 2) :
  (4 * x^3 - 1997 * x - 1994)^20001 = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l318_31877


namespace NUMINAMATH_CALUDE_max_gcd_triangular_number_l318_31825

def triangular_number (n : ℕ+) : ℕ := (n * (n + 1)) / 2

theorem max_gcd_triangular_number :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (6 * triangular_number n) (n - 2) ≤ 12 ∧
  Nat.gcd (6 * triangular_number k) (k - 2) = 12 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_triangular_number_l318_31825


namespace NUMINAMATH_CALUDE_condition_for_reciprocal_less_than_one_l318_31802

theorem condition_for_reciprocal_less_than_one (a : ℝ) :
  (a > 1 → (1 / a) < 1) ∧ (∃ b : ℝ, (1 / b) < 1 ∧ b ≤ 1) := by sorry

end NUMINAMATH_CALUDE_condition_for_reciprocal_less_than_one_l318_31802


namespace NUMINAMATH_CALUDE_f_positive_at_one_l318_31809

def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem f_positive_at_one (a : ℝ) :
  f a 1 > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_f_positive_at_one_l318_31809


namespace NUMINAMATH_CALUDE_manuscript_cost_l318_31884

/-- Represents the cost structure for typing and revising pages -/
structure TypingRates :=
  (initial : ℕ)
  (first_revision : ℕ)
  (second_revision : ℕ)
  (subsequent_revisions : ℕ)

/-- Represents the manuscript details -/
structure Manuscript :=
  (total_pages : ℕ)
  (revised_once : ℕ)
  (revised_twice : ℕ)
  (revised_thrice : ℕ)

/-- Calculates the total cost of typing and revising a manuscript -/
def total_cost (rates : TypingRates) (manuscript : Manuscript) : ℕ :=
  rates.initial * manuscript.total_pages +
  rates.first_revision * manuscript.revised_once +
  rates.second_revision * manuscript.revised_twice +
  rates.subsequent_revisions * manuscript.revised_thrice

/-- The typing service rates -/
def service_rates : TypingRates :=
  { initial := 10
  , first_revision := 5
  , second_revision := 7
  , subsequent_revisions := 10 }

/-- The manuscript details -/
def manuscript : Manuscript :=
  { total_pages := 150
  , revised_once := 20
  , revised_twice := 30
  , revised_thrice := 10 }

/-- Theorem stating that the total cost for the given manuscript is 1910 -/
theorem manuscript_cost : total_cost service_rates manuscript = 1910 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_l318_31884


namespace NUMINAMATH_CALUDE_grid_selection_count_l318_31852

theorem grid_selection_count : ℕ := by
  -- Define the size of the grid
  let n : ℕ := 6
  
  -- Define the number of blocks to select
  let k : ℕ := 4
  
  -- Define the function to calculate combinations
  let choose (n m : ℕ) := Nat.choose n m
  
  -- Define the total number of combinations
  let total_combinations := choose n k * choose n k * Nat.factorial k
  
  -- Prove that the total number of combinations is 5400
  sorry

end NUMINAMATH_CALUDE_grid_selection_count_l318_31852


namespace NUMINAMATH_CALUDE_diana_total_earnings_l318_31858

def july_earnings : ℝ := 150

def august_earnings : ℝ := 3 * july_earnings

def september_earnings : ℝ := 2 * august_earnings

def october_earnings : ℝ := september_earnings * 1.1

def november_earnings : ℝ := october_earnings * 0.95

def total_earnings : ℝ := july_earnings + august_earnings + september_earnings + october_earnings + november_earnings

theorem diana_total_earnings : total_earnings = 3430.50 := by
  sorry

end NUMINAMATH_CALUDE_diana_total_earnings_l318_31858


namespace NUMINAMATH_CALUDE_coin_value_equality_l318_31851

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of quarters in the first group -/
def quarters_1 : ℕ := 15

/-- The number of dimes in the first group -/
def dimes_1 : ℕ := 10

/-- The number of quarters in the second group -/
def quarters_2 : ℕ := 25

theorem coin_value_equality (n : ℕ) : 
  quarters_1 * quarter_value + dimes_1 * dime_value = 
  quarters_2 * quarter_value + n * dime_value → n = 35 := by
sorry

end NUMINAMATH_CALUDE_coin_value_equality_l318_31851


namespace NUMINAMATH_CALUDE_app_total_cost_l318_31872

/-- Calculates the total cost of an app with online access -/
def total_cost (initial_price : ℕ) (monthly_fee : ℕ) (months : ℕ) : ℕ :=
  initial_price + monthly_fee * months

/-- Proves that the total cost for the given conditions is $21 -/
theorem app_total_cost : total_cost 5 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_app_total_cost_l318_31872


namespace NUMINAMATH_CALUDE_cultural_shirt_production_theorem_l318_31826

/-- Represents the production and pricing of cultural shirts --/
structure CulturalShirtProduction where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  second_batch_quantity_multiplier : ℝ
  second_batch_cost_increase : ℝ
  discount_rate : ℝ
  discount_quantity : ℕ
  target_profit_margin : ℝ

/-- Calculates the cost per shirt in the first batch and the price per shirt for a given profit margin --/
def calculate_shirt_costs_and_price (prod : CulturalShirtProduction) :
  (ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct cost and price for the given conditions --/
theorem cultural_shirt_production_theorem (prod : CulturalShirtProduction)
  (h1 : prod.first_batch_cost = 3000)
  (h2 : prod.second_batch_cost = 6600)
  (h3 : prod.second_batch_quantity_multiplier = 2)
  (h4 : prod.second_batch_cost_increase = 3)
  (h5 : prod.discount_rate = 0.6)
  (h6 : prod.discount_quantity = 30)
  (h7 : prod.target_profit_margin = 0.5) :
  calculate_shirt_costs_and_price prod = (30, 50) :=
  sorry

end NUMINAMATH_CALUDE_cultural_shirt_production_theorem_l318_31826


namespace NUMINAMATH_CALUDE_sqrt_four_minus_one_l318_31844

theorem sqrt_four_minus_one : Real.sqrt 4 - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_minus_one_l318_31844


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l318_31881

theorem cubic_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 94 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l318_31881


namespace NUMINAMATH_CALUDE_oldest_child_age_l318_31857

-- Define the problem parameters
def num_children : ℕ := 4
def average_age : ℝ := 8
def younger_ages : List ℝ := [5, 7, 9]

-- State the theorem
theorem oldest_child_age :
  ∀ (oldest_age : ℝ),
  (List.sum younger_ages + oldest_age) / num_children = average_age →
  oldest_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l318_31857


namespace NUMINAMATH_CALUDE_doris_erasers_taken_l318_31807

def initial_erasers : ℕ := 69
def remaining_erasers : ℕ := 15

def erasers_taken_out : ℕ := initial_erasers - remaining_erasers

theorem doris_erasers_taken : erasers_taken_out = 54 := by
  sorry

end NUMINAMATH_CALUDE_doris_erasers_taken_l318_31807


namespace NUMINAMATH_CALUDE_bianca_recycling_points_l318_31867

theorem bianca_recycling_points 
  (points_per_bag : ℕ) 
  (total_bags : ℕ) 
  (bags_not_recycled : ℕ) 
  (h1 : points_per_bag = 5)
  (h2 : total_bags = 17)
  (h3 : bags_not_recycled = 8) :
  (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by sorry

end NUMINAMATH_CALUDE_bianca_recycling_points_l318_31867


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l318_31832

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on the highway
  city : ℝ     -- Miles per gallon in the city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- The conditions given in the problem -/
def problem_conditions (car : CarFuelEfficiency) : Prop :=
  car.highway * car.tank_size = 560 ∧
  car.city * car.tank_size = 336 ∧
  car.city = car.highway - 6

/-- The theorem to be proved -/
theorem city_fuel_efficiency (car : CarFuelEfficiency) :
  problem_conditions car → car.city = 9 := by
  sorry


end NUMINAMATH_CALUDE_city_fuel_efficiency_l318_31832


namespace NUMINAMATH_CALUDE_percentage_of_adult_men_l318_31899

theorem percentage_of_adult_men (total : ℕ) (children : ℕ) 
  (h1 : total = 2000) 
  (h2 : children = 200) 
  (h3 : ∃ (men women : ℕ), men + women + children = total ∧ women = 2 * men) :
  ∃ (men : ℕ), men * 100 / total = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_adult_men_l318_31899


namespace NUMINAMATH_CALUDE_matrix_power_sum_l318_31821

/-- Given a matrix B and its mth power, prove that b + m = 381 -/
theorem matrix_power_sum (b m : ℕ) : 
  let B : Matrix (Fin 3) (Fin 3) ℕ := !![1, 3, b; 0, 1, 5; 0, 0, 1]
  let B_pow_m : Matrix (Fin 3) (Fin 3) ℕ := !![1, 33, 4054; 0, 1, 55; 0, 0, 1]
  B^m = B_pow_m → b + m = 381 := by
sorry

end NUMINAMATH_CALUDE_matrix_power_sum_l318_31821


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l318_31876

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l318_31876


namespace NUMINAMATH_CALUDE_slope_of_line_l318_31859

theorem slope_of_line (x y : ℝ) :
  4 * x + 6 * y = 24 → (y - 4) / x = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l318_31859


namespace NUMINAMATH_CALUDE_maria_has_nineteen_towels_l318_31898

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def marias_remaining_towels (green_towels white_towels given_to_mother : ℕ) : ℕ :=
  green_towels + white_towels - given_to_mother

/-- Theorem stating that Maria ended up with 19 towels. -/
theorem maria_has_nineteen_towels :
  marias_remaining_towels 40 44 65 = 19 := by
  sorry

end NUMINAMATH_CALUDE_maria_has_nineteen_towels_l318_31898


namespace NUMINAMATH_CALUDE_triangle_special_area_implies_angle_l318_31847

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S of the triangle is (√3/4)(a² + b² - c²), then angle C measures π/3 --/
theorem triangle_special_area_implies_angle (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_area : (a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))) / 2 = 
            (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_special_area_implies_angle_l318_31847


namespace NUMINAMATH_CALUDE_tile_problem_l318_31831

theorem tile_problem (total_tiles : ℕ) : 
  (∃ n : ℕ, total_tiles = n^2 + 36 ∧ total_tiles = (n + 1)^2 + 3) → 
  total_tiles = 292 := by
sorry

end NUMINAMATH_CALUDE_tile_problem_l318_31831


namespace NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l318_31870

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = (x+1)(x-A) -/
def f (A : ℝ) (x : ℝ) : ℝ :=
  (x + 1) * (x - A)

/-- If f(x) = (x+1)(x-A) is an even function, then A = 1 -/
theorem even_function_implies_A_equals_one :
  ∀ A : ℝ, IsEven (f A) → A = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l318_31870


namespace NUMINAMATH_CALUDE_typing_time_proof_l318_31866

/-- Calculates the time in hours required to type a research paper given the typing speed and number of words. -/
def time_to_type (typing_speed : ℕ) (total_words : ℕ) : ℚ :=
  (total_words : ℚ) / (typing_speed : ℚ) / 60

/-- Proves that given a typing speed of 38 words per minute and a research paper with 4560 words, the time required to type the paper is 2 hours. -/
theorem typing_time_proof :
  time_to_type 38 4560 = 2 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_proof_l318_31866


namespace NUMINAMATH_CALUDE_lara_baking_cookies_l318_31834

/-- The number of baking trays Lara is using. -/
def num_trays : ℕ := 4

/-- The number of rows of cookies on each tray. -/
def rows_per_tray : ℕ := 5

/-- The number of cookies in each row. -/
def cookies_per_row : ℕ := 6

/-- The total number of cookies Lara is baking. -/
def total_cookies : ℕ := num_trays * rows_per_tray * cookies_per_row

theorem lara_baking_cookies : total_cookies = 120 := by
  sorry

end NUMINAMATH_CALUDE_lara_baking_cookies_l318_31834


namespace NUMINAMATH_CALUDE_adjacent_angles_theorem_l318_31829

/-- Given two adjacent angles forming a straight line, where one angle is 4x and the other is x, 
    prove that x = 18°. -/
theorem adjacent_angles_theorem (x : ℝ) : 
  (4 * x + x = 180) → x = 18 := by sorry

end NUMINAMATH_CALUDE_adjacent_angles_theorem_l318_31829


namespace NUMINAMATH_CALUDE_ab_value_l318_31801

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 33) : a * b = 18 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l318_31801


namespace NUMINAMATH_CALUDE_balcony_price_is_eight_l318_31891

/-- Represents the theater ticket sales scenario --/
structure TheaterSales where
  totalTickets : ℕ
  totalCost : ℕ
  orchestraPrice : ℕ
  balconyExcess : ℕ

/-- Calculates the price of a balcony seat given the theater sales data --/
def balconyPrice (sales : TheaterSales) : ℕ :=
  let orchestraTickets := (sales.totalTickets - sales.balconyExcess) / 2
  let balconyTickets := sales.totalTickets - orchestraTickets
  (sales.totalCost - orchestraTickets * sales.orchestraPrice) / balconyTickets

/-- Theorem stating that the balcony price is $8 given the specific sales data --/
theorem balcony_price_is_eight :
  balconyPrice ⟨370, 3320, 12, 190⟩ = 8 := by
  sorry

#eval balconyPrice ⟨370, 3320, 12, 190⟩

end NUMINAMATH_CALUDE_balcony_price_is_eight_l318_31891


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l318_31820

theorem rectangle_dimension_change (x : ℝ) : 
  (1 + x / 100) * (1 - 5 / 100) = 1 + 14.000000000000002 / 100 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l318_31820


namespace NUMINAMATH_CALUDE_outfit_combinations_l318_31818

def number_of_shirts : ℕ := 5
def number_of_pants : ℕ := 4
def number_of_hats : ℕ := 2

theorem outfit_combinations : 
  number_of_shirts * number_of_pants * number_of_hats = 40 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l318_31818


namespace NUMINAMATH_CALUDE_miles_driven_proof_l318_31886

def miles_per_gallon : ℝ := 40
def cost_per_gallon : ℝ := 5
def budget : ℝ := 25

theorem miles_driven_proof : 
  (budget / cost_per_gallon) * miles_per_gallon = 200 := by sorry

end NUMINAMATH_CALUDE_miles_driven_proof_l318_31886


namespace NUMINAMATH_CALUDE_cookies_in_box_l318_31855

/-- The number of cookies Jackson's oldest son gets after school -/
def oldest_son_cookies : ℕ := 4

/-- The number of cookies Jackson's youngest son gets after school -/
def youngest_son_cookies : ℕ := 2

/-- The number of days a box of cookies lasts -/
def box_duration : ℕ := 9

/-- The total number of cookies in the box -/
def total_cookies : ℕ := oldest_son_cookies + youngest_son_cookies * box_duration

theorem cookies_in_box : total_cookies = 54 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_box_l318_31855


namespace NUMINAMATH_CALUDE_x_plus_y_equals_1003_l318_31837

theorem x_plus_y_equals_1003 
  (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 1003 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_1003_l318_31837


namespace NUMINAMATH_CALUDE_total_pills_taken_l318_31893

-- Define the given conditions
def dose_mg : ℕ := 1000
def dose_interval_hours : ℕ := 6
def treatment_weeks : ℕ := 2
def mg_per_pill : ℕ := 500
def hours_per_day : ℕ := 24
def days_per_week : ℕ := 7

-- Define the theorem
theorem total_pills_taken : 
  (dose_mg / mg_per_pill) * 
  (hours_per_day / dose_interval_hours) * 
  (treatment_weeks * days_per_week) = 112 := by
sorry

end NUMINAMATH_CALUDE_total_pills_taken_l318_31893


namespace NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_to_line_parallel_l318_31888

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the basic relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (parallel_to_line : Plane → Line → Prop)
variable (perpendicular_to_line : Plane → Line → Prop)
variable (coincident : Plane → Plane → Prop)

-- Theorem 1
theorem parallel_planes_transitive (α β γ : Plane) :
  parallel α γ → parallel β γ → ¬coincident α β → parallel α β := by sorry

-- Theorem 2
theorem perpendicular_to_line_parallel (α β : Plane) (l : Line) :
  perpendicular_to_line α l → perpendicular_to_line β l → ¬coincident α β → parallel α β := by sorry

end NUMINAMATH_CALUDE_parallel_planes_transitive_perpendicular_to_line_parallel_l318_31888


namespace NUMINAMATH_CALUDE_coin_toss_probability_l318_31817

/-- The probability of a coin with diameter 1/2 not touching any lattice lines when tossed onto a 1x1 square -/
def coin_probability : ℚ := 1 / 4

/-- The diameter of the coin -/
def coin_diameter : ℚ := 1 / 2

/-- The side length of the square -/
def square_side : ℚ := 1

theorem coin_toss_probability :
  coin_probability = (square_side - coin_diameter)^2 / square_side^2 :=
by sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l318_31817


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l318_31824

theorem arctan_sum_equation (x : ℝ) : 
  3 * Real.arctan (1/4) + Real.arctan (1/20) + Real.arctan (1/x) = π/4 → x = 1985 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l318_31824


namespace NUMINAMATH_CALUDE_cosine_sum_theorem_l318_31892

theorem cosine_sum_theorem (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_theorem_l318_31892


namespace NUMINAMATH_CALUDE_car_speed_comparison_l318_31875

/-- Proves that given a car traveling at 80 km/hour takes 5 seconds longer to travel 1 km than at another speed, the other speed is 90 km/hour. -/
theorem car_speed_comparison (v : ℝ) : 
  v > 0 →  -- Ensure speed is positive
  (1 / (80 / 3600)) - (1 / (v / 3600)) = 5 → 
  v = 90 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l318_31875


namespace NUMINAMATH_CALUDE_club_leadership_selection_l318_31887

theorem club_leadership_selection (total_members : Nat) (boys : Nat) (girls : Nat)
  (senior_members : Nat) (senior_boys : Nat) (senior_girls : Nat)
  (h1 : total_members = 24)
  (h2 : boys = 14)
  (h3 : girls = 10)
  (h4 : senior_members = 6)
  (h5 : senior_boys = 4)
  (h6 : senior_girls = 2)
  (h7 : total_members = boys + girls)
  (h8 : senior_members = senior_boys + senior_girls) :
  (senior_boys * girls + senior_girls * boys) = 68 := by
  sorry


end NUMINAMATH_CALUDE_club_leadership_selection_l318_31887


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l318_31885

theorem min_four_dollar_frisbees (total_frisbees : ℕ) (total_receipts : ℕ) : 
  total_frisbees = 64 →
  total_receipts = 200 →
  ∃ (three_dollar : ℕ) (four_dollar : ℕ),
    three_dollar + four_dollar = total_frisbees ∧
    3 * three_dollar + 4 * four_dollar = total_receipts ∧
    ∀ (other_four_dollar : ℕ),
      other_four_dollar + (total_frisbees - other_four_dollar) = total_frisbees ∧
      3 * (total_frisbees - other_four_dollar) + 4 * other_four_dollar = total_receipts →
      four_dollar ≤ other_four_dollar ∧
      four_dollar = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l318_31885


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l318_31819

/-- Parabola C defined by y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Line l defined by x = ty + m where m > 0 -/
structure Line where
  t : ℝ
  m : ℝ
  h_m_pos : m > 0

/-- Point on the parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Intersection points of line and parabola -/
structure Intersection (C : Parabola) (l : Line) where
  A : ParabolaPoint C
  B : ParabolaPoint C
  h_A_on_line : A.x = l.t * A.y + l.m
  h_B_on_line : B.x = l.t * B.y + l.m

/-- Main theorem -/
theorem parabola_and_line_properties
  (C : Parabola)
  (P : ParabolaPoint C)
  (h_P_x : P.x = 2)
  (h_P_dist : (P.x - C.p/2)^2 + P.y^2 = 4^2)
  (l : Line)
  (i : Intersection C l)
  (h_circle : i.A.x * i.B.x + i.A.y * i.B.y = 0) :
  (C.p = 4 ∧ ∀ (y : ℝ), y^2 = 8 * (l.t * y + l.m)) ∧
  (l.m = 8 ∧ ∀ (y : ℝ), l.t * y + l.m = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l318_31819


namespace NUMINAMATH_CALUDE_rosa_initial_flowers_l318_31853

theorem rosa_initial_flowers (flowers_from_andre : ℕ) (total_flowers : ℕ) 
  (h1 : flowers_from_andre = 23)
  (h2 : total_flowers = 90) :
  total_flowers - flowers_from_andre = 67 := by
  sorry

end NUMINAMATH_CALUDE_rosa_initial_flowers_l318_31853


namespace NUMINAMATH_CALUDE_square_to_rectangle_l318_31896

theorem square_to_rectangle (s : ℝ) (h1 : s > 0) 
  (h2 : s * (s + 3) - s * s = 18) : 
  s * s = 36 ∧ s * (s + 3) = 54 := by
  sorry

#check square_to_rectangle

end NUMINAMATH_CALUDE_square_to_rectangle_l318_31896


namespace NUMINAMATH_CALUDE_opposite_corners_not_tileable_different_color_cells_tileable_l318_31860

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (removed : List (Nat × Nat))

/-- Represents a domino -/
inductive Domino
  | horizontal : Nat → Nat → Domino
  | vertical : Nat → Nat → Domino

/-- A tiling of a chessboard with dominoes -/
def Tiling := List Domino

/-- Returns true if the given coordinates represent a black square on the chessboard -/
def isBlack (x y : Nat) : Bool :=
  (x + y) % 2 = 0

/-- Returns true if the two given cells have different colors -/
def differentColors (x1 y1 x2 y2 : Nat) : Bool :=
  isBlack x1 y1 ≠ isBlack x2 y2

/-- Returns true if the given tiling is valid for the given chessboard -/
def isValidTiling (board : Chessboard) (tiling : Tiling) : Bool :=
  sorry

theorem opposite_corners_not_tileable :
  ∀ (board : Chessboard),
    board.size = 8 →
    board.removed = [(0, 0), (7, 7)] →
    ¬∃ (tiling : Tiling), isValidTiling board tiling :=
  sorry

theorem different_color_cells_tileable :
  ∀ (board : Chessboard) (x1 y1 x2 y2 : Nat),
    board.size = 8 →
    x1 < 8 ∧ y1 < 8 ∧ x2 < 8 ∧ y2 < 8 →
    differentColors x1 y1 x2 y2 →
    board.removed = [(x1, y1), (x2, y2)] →
    ∃ (tiling : Tiling), isValidTiling board tiling :=
  sorry

end NUMINAMATH_CALUDE_opposite_corners_not_tileable_different_color_cells_tileable_l318_31860


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l318_31814

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (1 - 3 * x) = 7 → x = -16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l318_31814


namespace NUMINAMATH_CALUDE_water_fountain_build_time_l318_31800

/-- Represents the work rate for building water fountains -/
def work_rate (men : ℕ) (length : ℕ) (days : ℕ) : ℚ :=
  length / (men * days)

/-- Theorem stating the relationship between different teams building water fountains -/
theorem water_fountain_build_time 
  (men1 : ℕ) (length1 : ℕ) (days1 : ℕ)
  (men2 : ℕ) (length2 : ℕ) (days2 : ℕ)
  (h1 : men1 = 20) (h2 : length1 = 56) (h3 : days1 = 7)
  (h4 : men2 = 35) (h5 : length2 = 42) (h6 : days2 = 3) :
  work_rate men1 length1 days1 = work_rate men2 length2 days2 :=
by sorry

#check water_fountain_build_time

end NUMINAMATH_CALUDE_water_fountain_build_time_l318_31800


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l318_31811

/-- The surface area of a cuboid given its dimensions -/
def cuboid_surface_area (width length height : ℝ) : ℝ :=
  2 * (width * length + width * height + length * height)

/-- Theorem: The surface area of a cuboid with width 3, length 4, and height 5 is 94 -/
theorem cuboid_surface_area_example : cuboid_surface_area 3 4 5 = 94 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l318_31811


namespace NUMINAMATH_CALUDE_weekly_savings_l318_31812

def hourly_rate_1 : ℚ := 20
def hourly_rate_2 : ℚ := 22
def subsidy : ℚ := 6
def hours_per_week : ℚ := 40

def weekly_cost_1 : ℚ := hourly_rate_1 * hours_per_week
def effective_hourly_rate_2 : ℚ := hourly_rate_2 - subsidy
def weekly_cost_2 : ℚ := effective_hourly_rate_2 * hours_per_week

theorem weekly_savings : weekly_cost_1 - weekly_cost_2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_weekly_savings_l318_31812


namespace NUMINAMATH_CALUDE_smaller_number_problem_l318_31845

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x ∧ x + y = 124 → x = 31 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l318_31845


namespace NUMINAMATH_CALUDE_garden_area_l318_31838

theorem garden_area (total_distance : ℝ) (length_walks : ℕ) (perimeter_walks : ℕ) :
  total_distance = 1500 →
  length_walks = 30 →
  perimeter_walks = 12 →
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length * length_walks = total_distance ∧
    2 * (length + width) * perimeter_walks = total_distance ∧
    length * width = 625 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_l318_31838


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l318_31871

def total_beads : ℕ := 6
def red_beads : ℕ := 3
def white_beads : ℕ := 2
def blue_beads : ℕ := 1

def total_arrangements : ℕ := (Nat.factorial total_beads) / ((Nat.factorial red_beads) * (Nat.factorial white_beads) * (Nat.factorial blue_beads))

def valid_arrangements : ℕ := 10

theorem bead_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l318_31871


namespace NUMINAMATH_CALUDE_gift_cost_l318_31856

/-- Proves that the cost of the gift is $250 given the specified conditions --/
theorem gift_cost (erika_savings : ℕ) (cake_cost : ℕ) (leftover : ℕ) :
  erika_savings = 155 →
  cake_cost = 25 →
  leftover = 5 →
  ∃ (gift_cost : ℕ), 
    gift_cost = 250 ∧
    erika_savings + gift_cost / 2 = gift_cost + cake_cost + leftover :=
by
  sorry

end NUMINAMATH_CALUDE_gift_cost_l318_31856


namespace NUMINAMATH_CALUDE_number_relations_l318_31850

theorem number_relations (A B C : ℝ) : 
  A - B = 1860 ∧ 
  0.075 * A = 0.125 * B ∧ 
  0.15 * B = 0.05 * C → 
  A = 4650 ∧ B = 2790 ∧ C = 8370 := by
sorry

end NUMINAMATH_CALUDE_number_relations_l318_31850


namespace NUMINAMATH_CALUDE_fraction_equality_l318_31816

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 2023) :
  (w + z)/(w - z) = -1012 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l318_31816


namespace NUMINAMATH_CALUDE_most_colored_pencils_l318_31842

theorem most_colored_pencils (total : ℕ) (red blue yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow :=
by sorry

end NUMINAMATH_CALUDE_most_colored_pencils_l318_31842


namespace NUMINAMATH_CALUDE_cos_fifteen_squared_formula_l318_31846

theorem cos_fifteen_squared_formula : 2 * (Real.cos (15 * π / 180))^2 - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_fifteen_squared_formula_l318_31846


namespace NUMINAMATH_CALUDE_jill_and_emily_total_l318_31823

/-- The number of peaches each person has -/
structure Peaches where
  steven : ℕ
  jake : ℕ
  jill : ℕ
  maria : ℕ
  emily : ℕ

/-- The conditions of the peach distribution problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.steven = 14 ∧
  p.jake = p.steven - 6 ∧
  p.jake = p.jill + 3 ∧
  p.maria = 2 * p.jake ∧
  p.emily = p.maria - 9

/-- The theorem stating that Jill and Emily have 12 peaches in total -/
theorem jill_and_emily_total (p : Peaches) (h : peach_conditions p) : 
  p.jill + p.emily = 12 := by
  sorry

end NUMINAMATH_CALUDE_jill_and_emily_total_l318_31823


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l318_31827

/-- A quadratic function f(x) = ax^2 - bx satisfying certain conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (a b : ℝ) :
  (f a b 2 = 0) →
  (∃ x : ℝ, (f a b x = x) ∧ (∀ y : ℝ, f a b y = y → y = x)) →
  ((∀ x : ℝ, f a b x = -1/2 * x^2 + x) ∧
   (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a b x ≤ 1/2) ∧
   (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f a b x = 1/2)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l318_31827


namespace NUMINAMATH_CALUDE_opposite_of_2023_l318_31841

theorem opposite_of_2023 : 
  (2023 : ℤ) + (-2023) = 0 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l318_31841


namespace NUMINAMATH_CALUDE_cuboid_gluing_theorem_l318_31889

/-- A cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+
  different_dimensions : length ≠ width ∧ width ≠ height ∧ height ≠ length

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℕ := c.length * c.width * c.height

/-- Two cuboids can be glued if they share a face -/
def can_be_glued (c1 c2 : Cuboid) : Prop :=
  (c1.length = c2.length ∧ c1.width = c2.width) ∨
  (c1.length = c2.length ∧ c1.height = c2.height) ∨
  (c1.width = c2.width ∧ c1.height = c2.height)

/-- The resulting cuboid after gluing two cuboids -/
def glued_cuboid (c1 c2 : Cuboid) : Cuboid :=
  if c1.length = c2.length ∧ c1.width = c2.width then
    ⟨c1.length, c1.width, c1.height + c2.height, sorry⟩
  else if c1.length = c2.length ∧ c1.height = c2.height then
    ⟨c1.length, c1.width + c2.width, c1.height, sorry⟩
  else
    ⟨c1.length + c2.length, c1.width, c1.height, sorry⟩

theorem cuboid_gluing_theorem (c1 c2 : Cuboid) :
  volume c1 = 12 →
  volume c2 = 30 →
  can_be_glued c1 c2 →
  let c := glued_cuboid c1 c2
  (c.length = 1 ∧ c.width = 2 ∧ c.height = 21) ∨
  (c.length = 1 ∧ c.width = 3 ∧ c.height = 14) ∨
  (c.length = 1 ∧ c.width = 6 ∧ c.height = 7) :=
by sorry

end NUMINAMATH_CALUDE_cuboid_gluing_theorem_l318_31889


namespace NUMINAMATH_CALUDE_range_of_a_l318_31804

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2 ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l318_31804


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l318_31839

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 4*a - 12 = 0) : 
  -2*a^2 + 8*a + 40 = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l318_31839


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l318_31865

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l318_31865


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l318_31869

/-- The area of the shaded region formed by semicircles in a pattern -/
theorem semicircle_pattern_area (pattern_length : ℝ) (semicircle_diameter : ℝ) :
  pattern_length = 18 →
  semicircle_diameter = 3 →
  let num_semicircles : ℝ := pattern_length / semicircle_diameter
  let num_full_circles : ℝ := num_semicircles / 2
  let circle_radius : ℝ := semicircle_diameter / 2
  pattern_length > 0 →
  semicircle_diameter > 0 →
  (num_full_circles * π * circle_radius^2) = (27 / 4) * π :=
by sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l318_31869


namespace NUMINAMATH_CALUDE_bianca_coloring_books_l318_31890

/-- Represents the number of coloring books Bianca gave away -/
def books_given_away : ℕ := 6

/-- Represents Bianca's initial number of coloring books -/
def initial_books : ℕ := 45

/-- Represents the number of coloring books Bianca bought -/
def books_bought : ℕ := 20

/-- Represents Bianca's final number of coloring books -/
def final_books : ℕ := 59

theorem bianca_coloring_books : 
  initial_books - books_given_away + books_bought = final_books :=
by sorry

end NUMINAMATH_CALUDE_bianca_coloring_books_l318_31890


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l318_31840

/-- 
Given two algebraic terms are like terms, prove that the product of their exponents is 6.
-/
theorem like_terms_exponent_product (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), 5 * a^3 * b^n = k * (-3 * a^m * b^2)) → m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l318_31840


namespace NUMINAMATH_CALUDE_ladder_problem_l318_31833

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 15)
  (h2 : height = 9) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 12 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l318_31833


namespace NUMINAMATH_CALUDE_complex_fraction_product_l318_31830

theorem complex_fraction_product (a b : ℝ) : 
  (1 + 7 * Complex.I) / (2 - Complex.I) = Complex.mk a b → a * b = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_product_l318_31830


namespace NUMINAMATH_CALUDE_total_cost_theorem_l318_31882

-- Define the given conditions
def cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cards_per_pack : ℕ := 50
def cost_per_pack : ℚ := 3

-- Define the total number of index cards needed
def total_cards_needed : ℕ := cards_per_student * students_per_class * periods_per_day

-- Define the number of packs needed
def packs_needed : ℕ := (total_cards_needed + cards_per_pack - 1) / cards_per_pack

-- State the theorem
theorem total_cost_theorem : 
  cost_per_pack * packs_needed = 108 := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l318_31882


namespace NUMINAMATH_CALUDE_vertex_at_max_min_l318_31822

/-- The quadratic function f parameterized by k -/
def f (x k : ℝ) : ℝ := x^2 - 2*(2*k - 1)*x + 3*k^2 - 2*k + 6

/-- The x-coordinate of the vertex of f for a given k -/
def vertex_x (k : ℝ) : ℝ := 2*k - 1

/-- The minimum value of f for a given k -/
def min_value (k : ℝ) : ℝ := f (vertex_x k) k

/-- The theorem stating that the x-coordinate of the vertex when the minimum value is maximized is 1 -/
theorem vertex_at_max_min : 
  ∃ (k : ℝ), ∀ (k' : ℝ), min_value k ≥ min_value k' ∧ vertex_x k = 1 := by sorry

end NUMINAMATH_CALUDE_vertex_at_max_min_l318_31822


namespace NUMINAMATH_CALUDE_union_complement_eq_specific_set_l318_31815

open Set

def U : Finset ℕ := {0, 1, 2, 4, 6, 8}
def M : Finset ℕ := {0, 4, 6}
def N : Finset ℕ := {0, 1, 6}

theorem union_complement_eq_specific_set :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end NUMINAMATH_CALUDE_union_complement_eq_specific_set_l318_31815


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l318_31868

theorem multiple_with_binary_digits (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, m ≠ 0 ∧ n ∣ m ∧ (Nat.digits 10 m).length ≤ n ∧ ∀ d ∈ Nat.digits 10 m, d = 0 ∨ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l318_31868


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_l318_31895

/-- The minimum perimeter of a rectangle with area 100 is 40 -/
theorem min_perimeter_rectangle (x y : ℝ) (h : x * y = 100) :
  2 * (x + y) ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_rectangle_l318_31895


namespace NUMINAMATH_CALUDE_product_expansion_l318_31897

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l318_31897


namespace NUMINAMATH_CALUDE_special_ellipse_d_value_l318_31880

/-- An ellipse in the first quadrant tangent to both axes with foci at (5,10) and (d,10) --/
structure Ellipse where
  d : ℝ
  tangent_x : Bool
  tangent_y : Bool
  first_quadrant : Bool
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The d value for the special ellipse described in the problem --/
def special_ellipse_d : ℝ := 20

/-- Theorem stating that the d value for the special ellipse is 20 --/
theorem special_ellipse_d_value (e : Ellipse) 
  (h_tangent_x : e.tangent_x = true)
  (h_tangent_y : e.tangent_y = true)
  (h_first_quadrant : e.first_quadrant = true)
  (h_focus1 : e.focus1 = (5, 10))
  (h_focus2 : e.focus2 = (e.d, 10)) :
  e.d = special_ellipse_d := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_d_value_l318_31880


namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_achieves_minimum_l318_31843

theorem quadratic_minimum (x : ℝ) (h : x > 0) : x^2 - 2*x + 3 ≥ 2 := by
  sorry

theorem quadratic_achieves_minimum : ∃ (x : ℝ), x > 0 ∧ x^2 - 2*x + 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_achieves_minimum_l318_31843


namespace NUMINAMATH_CALUDE_apples_per_basket_l318_31849

theorem apples_per_basket (baskets_per_tree : ℕ) (trees : ℕ) (total_apples : ℕ) :
  baskets_per_tree = 20 →
  trees = 10 →
  total_apples = 3000 →
  total_apples / (trees * baskets_per_tree) = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_basket_l318_31849
