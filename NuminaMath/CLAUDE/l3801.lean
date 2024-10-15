import Mathlib

namespace NUMINAMATH_CALUDE_cathys_initial_amount_l3801_380103

/-- The amount of money Cathy had before her parents sent her money -/
def initial_amount : ℕ := sorry

/-- The amount of money Cathy's dad sent her -/
def dad_amount : ℕ := 25

/-- The amount of money Cathy's mom sent her -/
def mom_amount : ℕ := 2 * dad_amount

/-- The total amount Cathy has now -/
def total_amount : ℕ := 87

theorem cathys_initial_amount : 
  initial_amount = total_amount - (dad_amount + mom_amount) ∧ 
  initial_amount = 12 := by sorry

end NUMINAMATH_CALUDE_cathys_initial_amount_l3801_380103


namespace NUMINAMATH_CALUDE_sum_of_squares_bound_l3801_380121

theorem sum_of_squares_bound (x y z t : ℝ) 
  (h1 : |x + y + z - t| ≤ 1)
  (h2 : |y + z + t - x| ≤ 1)
  (h3 : |z + t + x - y| ≤ 1)
  (h4 : |t + x + y - z| ≤ 1) :
  x^2 + y^2 + z^2 + t^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_bound_l3801_380121


namespace NUMINAMATH_CALUDE_equation_solution_l3801_380195

theorem equation_solution : ∃ x : ℝ, (x + 2) / (2 * x - 1) = 1 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3801_380195


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3801_380190

theorem sum_of_two_numbers (A B : ℝ) : 
  A - B = 8 → (A + B) / 4 = 6 → A = 16 → A + B = 24 := by sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3801_380190


namespace NUMINAMATH_CALUDE_train_length_l3801_380174

/-- Given a train that crosses a post in 15 seconds and a platform 100 m long in 25 seconds, its length is 150 meters. -/
theorem train_length (post_crossing_time platform_crossing_time platform_length : ℝ)
  (h1 : post_crossing_time = 15)
  (h2 : platform_crossing_time = 25)
  (h3 : platform_length = 100) :
  ∃ (train_length train_speed : ℝ),
    train_length = train_speed * post_crossing_time ∧
    train_length + platform_length = train_speed * platform_crossing_time ∧
    train_length = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3801_380174


namespace NUMINAMATH_CALUDE_pirate_treasure_l3801_380170

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l3801_380170


namespace NUMINAMATH_CALUDE_parabola_vertex_l3801_380189

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -3 * x^2 + 6 * x + 1

/-- The x-coordinate of the vertex -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex -/
def vertex_y : ℝ := 4

/-- Theorem: The vertex of the parabola y = -3x^2 + 6x + 1 is (1, 4) -/
theorem parabola_vertex :
  (∀ x : ℝ, parabola x ≤ vertex_y) ∧
  parabola vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3801_380189


namespace NUMINAMATH_CALUDE_sues_necklace_beads_l3801_380107

/-- The number of beads in Sue's necklace -/
def total_beads (purple blue green : ℕ) : ℕ := purple + blue + green

/-- Theorem stating the total number of beads in Sue's necklace -/
theorem sues_necklace_beads : 
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  total_beads purple blue green = 46 := by
sorry

end NUMINAMATH_CALUDE_sues_necklace_beads_l3801_380107


namespace NUMINAMATH_CALUDE_one_and_half_of_number_l3801_380142

theorem one_and_half_of_number (x : ℚ) : (3 / 2) * x = 30 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_one_and_half_of_number_l3801_380142


namespace NUMINAMATH_CALUDE_expected_trait_count_is_forty_l3801_380123

/-- The probability of an individual having the genetic trait -/
def trait_probability : ℚ := 1 / 8

/-- The total number of people in the sample -/
def sample_size : ℕ := 320

/-- The expected number of people with the genetic trait in the sample -/
def expected_trait_count : ℚ := trait_probability * sample_size

theorem expected_trait_count_is_forty : expected_trait_count = 40 := by
  sorry

end NUMINAMATH_CALUDE_expected_trait_count_is_forty_l3801_380123


namespace NUMINAMATH_CALUDE_seventh_term_largest_coefficient_l3801_380118

def binomial_expansion (x : ℝ) (n : ℕ) : ℕ → ℝ
  | r => (-1)^r * (Nat.choose n r) * x^(2*n - 3*r)

theorem seventh_term_largest_coefficient :
  ∃ (x : ℝ), ∀ (r : ℕ), r ≠ 6 →
    |binomial_expansion x 11 6| ≥ |binomial_expansion x 11 r| :=
sorry

end NUMINAMATH_CALUDE_seventh_term_largest_coefficient_l3801_380118


namespace NUMINAMATH_CALUDE_min_value_theorem_l3801_380108

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - 2 * x - y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b - 2 * a - b = 0 → x + y / 2 ≤ a + b / 2 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y - 2 * x - y = 0 ∧ x + y / 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3801_380108


namespace NUMINAMATH_CALUDE_max_parts_correct_max_parts_2004_l3801_380160

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating that max_parts gives the correct maximum number of parts -/
theorem max_parts_correct (n : ℕ) : 
  max_parts n = 1 + n * (n + 1) / 2 := by sorry

/-- The specific case for 2004 lines -/
theorem max_parts_2004 : max_parts 2004 = 2009011 := by sorry

end NUMINAMATH_CALUDE_max_parts_correct_max_parts_2004_l3801_380160


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l3801_380150

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (3 ∣ n) ∧ (4 ∣ n) ∧ (7 ∣ n) ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((3 ∣ m) ∧ (4 ∣ m) ∧ (7 ∣ m))) ∧
  n = 168 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l3801_380150


namespace NUMINAMATH_CALUDE_complex_product_equals_24_plus_18i_l3801_380168

/-- Complex number multiplication -/
def complex_mult (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

/-- The imaginary unit i -/
def i : ℤ × ℤ := (0, 1)

theorem complex_product_equals_24_plus_18i : 
  complex_mult 3 (-4) 0 6 = (24, 18) := by sorry

end NUMINAMATH_CALUDE_complex_product_equals_24_plus_18i_l3801_380168


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_unique_pair_l3801_380165

theorem sqrt_equality_implies_unique_pair :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (49 + Real.sqrt (153 + 24 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 49 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_unique_pair_l3801_380165


namespace NUMINAMATH_CALUDE_solution_set_equals_interval_l3801_380153

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := {x : ℝ | 2 * x^2 + 8 * x ≤ -6}

-- State the theorem
theorem solution_set_equals_interval : 
  solution_set = Set.Icc (-3) (-1) := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_interval_l3801_380153


namespace NUMINAMATH_CALUDE_increasing_shift_l3801_380185

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Theorem statement
theorem increasing_shift (h : IncreasingOn f (-2) 3) :
  IncreasingOn (fun x => f (x + 5)) (-7) (-2) :=
sorry

end NUMINAMATH_CALUDE_increasing_shift_l3801_380185


namespace NUMINAMATH_CALUDE_one_circle_exists_l3801_380188

def circle_equation (a x y : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0

def is_circle (a : ℝ) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    radius > 0 ∧
    ∀ (x y : ℝ), circle_equation a x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

def a_set : Set ℝ := {-2, 0, 1, 3/4}

theorem one_circle_exists :
  ∃! (a : ℝ), a ∈ a_set ∧ is_circle a :=
sorry

end NUMINAMATH_CALUDE_one_circle_exists_l3801_380188


namespace NUMINAMATH_CALUDE_ratio_345_iff_arithmetic_sequence_l3801_380175

/-- Represents a right-angled triangle with side lengths a, b, c where a < b < c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_lt_b : a < b
  b_lt_c : b < c
  right_angle : a^2 + b^2 = c^2

/-- The ratio of sides is 3:4:5 -/
def has_ratio_345 (t : RightTriangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k

/-- The sides form an arithmetic sequence -/
def is_arithmetic_sequence (t : RightTriangle) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ t.b = t.a + d ∧ t.c = t.b + d

/-- The main theorem stating the equivalence of the two conditions -/
theorem ratio_345_iff_arithmetic_sequence (t : RightTriangle) :
  has_ratio_345 t ↔ is_arithmetic_sequence t :=
sorry

end NUMINAMATH_CALUDE_ratio_345_iff_arithmetic_sequence_l3801_380175


namespace NUMINAMATH_CALUDE_ratios_neither_necessary_nor_sufficient_l3801_380161

-- Define the coefficients for the two quadratic inequalities
variable (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ)

-- Define the solution sets for the two inequalities
def SolutionSet1 (x : ℝ) := a₁ * x^2 + b₁ * x + c₁ > 0
def SolutionSet2 (x : ℝ) := a₂ * x^2 + b₂ * x + c₂ > 0

-- Define the equality of ratios condition
def RatiosEqual := (a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)

-- Define the property of having the same solution set
def SameSolutionSet := ∀ x, SolutionSet1 a₁ b₁ c₁ x ↔ SolutionSet2 a₂ b₂ c₂ x

-- Theorem stating that the equality of ratios is neither necessary nor sufficient
theorem ratios_neither_necessary_nor_sufficient :
  ¬(RatiosEqual a₁ b₁ c₁ a₂ b₂ c₂ → SameSolutionSet a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(SameSolutionSet a₁ b₁ c₁ a₂ b₂ c₂ → RatiosEqual a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_ratios_neither_necessary_nor_sufficient_l3801_380161


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_b_values_min_a_value_min_a_value_achieved_l3801_380166

noncomputable def f (a b x : ℝ) : ℝ := b * x / Real.log x - a * x

theorem tangent_line_implies_a_b_values (a b : ℝ) :
  (∀ x y : ℝ, y = f a b x → 3 * x + 4 * y - Real.exp 2 = 0) →
  a = 1 ∧ b = 1 := by sorry

theorem min_a_value (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2) →
    x₂ ∈ Set.Icc (Real.exp 1) (Real.exp 2) →
    f 1 1 x₁ ≤ (deriv (f 1 1)) x₂ + a) →
  a ≥ 1/2 - 1/(4 * Real.exp 2) := by sorry

theorem min_a_value_achieved (a : ℝ) :
  a = 1/2 - 1/(4 * Real.exp 2) →
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧
    x₂ ∈ Set.Icc (Real.exp 1) (Real.exp 2) ∧
    f 1 1 x₁ ≤ (deriv (f 1 1)) x₂ + a := by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_b_values_min_a_value_min_a_value_achieved_l3801_380166


namespace NUMINAMATH_CALUDE_bird_nest_area_scientific_notation_l3801_380125

/-- The construction area of the National Stadium "Bird's Nest" in square meters -/
def bird_nest_area : ℝ := 258000

/-- The scientific notation representation of the bird_nest_area -/
def bird_nest_scientific : ℝ := 2.58 * (10 ^ 5)

theorem bird_nest_area_scientific_notation :
  bird_nest_area = bird_nest_scientific := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_area_scientific_notation_l3801_380125


namespace NUMINAMATH_CALUDE_minimum_time_to_fill_buckets_l3801_380181

def bucket_times : List Nat := [2, 4, 5, 7, 9]

def total_time (times : List Nat) : Nat :=
  (times.enum.map (fun (i, t) => t * (times.length - i))).sum

theorem minimum_time_to_fill_buckets :
  total_time bucket_times = 55 := by
  sorry

end NUMINAMATH_CALUDE_minimum_time_to_fill_buckets_l3801_380181


namespace NUMINAMATH_CALUDE_union_condition_intersection_condition_l3801_380131

-- Define sets A and B
def A : Set ℝ := {x | 2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < 3 * a}

-- Theorem 1
theorem union_condition (a : ℝ) : A ∪ B a = {x | 2 < x ∧ x < 6} → a = 2 := by
  sorry

-- Theorem 2
theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → 2/3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_union_condition_intersection_condition_l3801_380131


namespace NUMINAMATH_CALUDE_triangle_third_altitude_l3801_380155

theorem triangle_third_altitude (h₁ h₂ h₃ : ℝ) :
  h₁ = 8 → h₂ = 12 → h₃ > 0 →
  (1 / h₁ + 1 / h₂ > 1 / h₃) →
  h₃ > 4.8 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_altitude_l3801_380155


namespace NUMINAMATH_CALUDE_student_survey_l3801_380173

theorem student_survey (french_and_english : ℕ) (french_not_english : ℕ) 
  (percent_not_french : ℚ) :
  french_and_english = 25 →
  french_not_english = 65 →
  percent_not_french = 55/100 →
  french_and_english + french_not_english = (100 : ℚ) / (100 - percent_not_french) * 100 :=
by sorry

end NUMINAMATH_CALUDE_student_survey_l3801_380173


namespace NUMINAMATH_CALUDE_petes_walking_distance_l3801_380137

/-- Represents a pedometer with a maximum count --/
structure Pedometer where
  max_count : ℕ
  reset_count : ℕ
  final_reading : ℕ

/-- Calculates the total steps based on pedometer data --/
def total_steps (p : Pedometer) : ℕ :=
  p.reset_count * (p.max_count + 1) + p.final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem petes_walking_distance (p : Pedometer) (steps_per_mile : ℕ) :
  p.max_count = 99999 ∧
  p.reset_count = 38 ∧
  p.final_reading = 75000 ∧
  steps_per_mile = 1800 →
  steps_to_miles (total_steps p) steps_per_mile = 2150 := by
  sorry

#eval steps_to_miles (total_steps { max_count := 99999, reset_count := 38, final_reading := 75000 }) 1800

end NUMINAMATH_CALUDE_petes_walking_distance_l3801_380137


namespace NUMINAMATH_CALUDE_whitewash_fence_l3801_380144

theorem whitewash_fence (k : ℕ) : 
  ∀ (x y : Fin (2^(k+1))), 
    (∃ (z : Fin (2^(k+1))), z ≠ x ∧ z ≠ y ∧ 
      (2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (y.val^2 + 3*y.val - 2)) ↔ 
       2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (z.val^2 + 3*z.val - 2)))) ∧
    (∀ (w : Fin (2^(k+1))), 
      2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (w.val^2 + 3*w.val - 2)) → 
      w = x ∨ w = y) :=
by sorry

#check whitewash_fence

end NUMINAMATH_CALUDE_whitewash_fence_l3801_380144


namespace NUMINAMATH_CALUDE_percentage_problem_l3801_380143

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 700 → 
  0.3 * N = (P / 100) * 150 + 120 → 
  P = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3801_380143


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3801_380199

theorem sum_of_solutions_is_zero (x : ℝ) :
  ((-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  (∃ y : ℝ, (-12 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 9 / (y - 1) ∧ y ≠ x) →
  x + y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3801_380199


namespace NUMINAMATH_CALUDE_birds_taken_out_l3801_380129

theorem birds_taken_out (initial_birds remaining_birds : ℕ) 
  (h1 : initial_birds = 19)
  (h2 : remaining_birds = 9) :
  initial_birds - remaining_birds = 10 := by
  sorry

end NUMINAMATH_CALUDE_birds_taken_out_l3801_380129


namespace NUMINAMATH_CALUDE_dad_steps_l3801_380114

theorem dad_steps (dad_masha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l3801_380114


namespace NUMINAMATH_CALUDE_sqrt_equation_l3801_380124

theorem sqrt_equation (x y : ℝ) (h : Real.sqrt (x - 2) + (y - 3)^2 = 0) : 
  Real.sqrt (2*x + 3*y + 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l3801_380124


namespace NUMINAMATH_CALUDE_probability_one_of_each_l3801_380140

/-- The number of t-shirts in the wardrobe -/
def num_tshirts : ℕ := 3

/-- The number of pairs of jeans in the wardrobe -/
def num_jeans : ℕ := 7

/-- The number of hats in the wardrobe -/
def num_hats : ℕ := 4

/-- The total number of clothing items in the wardrobe -/
def total_items : ℕ := num_tshirts + num_jeans + num_hats

/-- The probability of selecting one t-shirt, one pair of jeans, and one hat -/
theorem probability_one_of_each : 
  (num_tshirts * num_jeans * num_hats : ℚ) / (total_items.choose 3) = 21 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l3801_380140


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_99_l3801_380138

theorem last_three_digits_of_7_to_99 : 7^99 ≡ 573 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_99_l3801_380138


namespace NUMINAMATH_CALUDE_consecutive_sum_property_l3801_380116

theorem consecutive_sum_property : ∃ (a : Fin 10 → ℝ),
  (∀ i : Fin 6, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) > 0) ∧
  (∀ j : Fin 4, (a j) + (a (j+1)) + (a (j+2)) + (a (j+3)) + (a (j+4)) + (a (j+5)) + (a (j+6)) < 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_property_l3801_380116


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3801_380172

theorem sufficient_not_necessary (a b : ℝ) :
  (a > b ∧ b > 0) → (1 / a^2 < 1 / b^2) ∧
  ∃ (x y : ℝ), (1 / x^2 < 1 / y^2) ∧ ¬(x > y ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3801_380172


namespace NUMINAMATH_CALUDE_marcy_water_amount_l3801_380117

/-- The amount of water Marcy keeps by her desk -/
def water_amount (sip_interval : ℕ) (sip_volume : ℕ) (total_time : ℕ) : ℚ :=
  (total_time / sip_interval * sip_volume : ℚ) / 1000

/-- Theorem stating that Marcy keeps 2 liters of water by her desk -/
theorem marcy_water_amount :
  water_amount 5 40 250 = 2 := by
  sorry

#eval water_amount 5 40 250

end NUMINAMATH_CALUDE_marcy_water_amount_l3801_380117


namespace NUMINAMATH_CALUDE_solution_characterization_l3801_380163

def system_equations (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

theorem solution_characterization :
  ∀ y : ℝ,
  (∀ x₁ x₂ x₃ x₄ x₅ : ℝ, system_equations x₁ x₂ x₃ x₄ x₅ y →
    ((y ≠ 2 ∧ y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
     (y = 2 → ∃ u : ℝ, x₁ = u ∧ x₂ = u ∧ x₃ = u ∧ x₄ = u ∧ x₅ = u) ∧
     (y^2 + y - 1 = 0 →
       ∃ u v : ℝ, x₁ = u ∧ x₂ = v ∧ x₃ = -u + y*v ∧ x₄ = -y*(u + v) ∧ x₅ = y*u - v))) :=
by sorry


end NUMINAMATH_CALUDE_solution_characterization_l3801_380163


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3801_380132

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem statement
theorem sixth_term_of_geometric_sequence 
  (a₁ a₂ : ℝ) 
  (h₁ : a₁ = 5) 
  (h₂ : a₂ = 15) : 
  geometric_sequence a₁ (a₂ / a₁) 6 = 1215 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3801_380132


namespace NUMINAMATH_CALUDE_regular_octagon_area_equals_diagonal_product_l3801_380146

/-- A regular octagon -/
structure RegularOctagon where
  -- We don't need to specify all properties of a regular octagon,
  -- just the existence of such a shape
  dummy : Unit

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The length of the longest diagonal of a regular octagon -/
def longest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The length of the shortest diagonal of a regular octagon -/
def shortest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of a regular octagon is equal to the product of 
    the lengths of its longest and shortest diagonals -/
theorem regular_octagon_area_equals_diagonal_product (o : RegularOctagon) :
  area o = longest_diagonal o * shortest_diagonal o :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_area_equals_diagonal_product_l3801_380146


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3801_380156

/-- Given a geometric sequence where the third term is 81 and the sixth term is 1,
    prove that the sum of the fourth and fifth terms is 36. -/
theorem geometric_sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  (∀ n, a (n + 1) = a n * r) →  -- Each term is r times the previous term
  a 3 = 81 →                    -- The third term is 81
  a 6 = 1 →                     -- The sixth term is 1
  a 4 + a 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3801_380156


namespace NUMINAMATH_CALUDE_f_value_at_2013_l3801_380198

theorem f_value_at_2013 (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^3 + b * Real.sin x + 9) →
  f (-2013) = 7 →
  f 2013 = 11 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_2013_l3801_380198


namespace NUMINAMATH_CALUDE_thirteen_factorial_divisible_by_eleven_l3801_380141

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: 13! is divisible by 11 -/
theorem thirteen_factorial_divisible_by_eleven :
  13 % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_factorial_divisible_by_eleven_l3801_380141


namespace NUMINAMATH_CALUDE_x_squared_less_than_x_l3801_380139

theorem x_squared_less_than_x (x : ℝ) : x^2 < x ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_less_than_x_l3801_380139


namespace NUMINAMATH_CALUDE_expected_stones_approx_l3801_380112

/-- The width of the river (scaled to 1) -/
def river_width : ℝ := 1

/-- The maximum jump distance (scaled to 0.01) -/
def jump_distance : ℝ := 0.01

/-- The probability that we cannot cross the river after n throws -/
noncomputable def P (n : ℕ) : ℝ :=
  ∑' i, (-1)^(i-1) * (n+1).choose i * (max (1 - i * jump_distance) 0)^n

/-- The expected number of stones needed to cross the river -/
noncomputable def expected_stones : ℝ :=
  ∑' n, P n

/-- Theorem stating the approximation of the expected number of stones -/
theorem expected_stones_approx :
  ∃ ε > 0, |expected_stones - 712.811| < ε :=
sorry

end NUMINAMATH_CALUDE_expected_stones_approx_l3801_380112


namespace NUMINAMATH_CALUDE_cassy_jars_proof_l3801_380100

def initial_jars (boxes_type1 boxes_type2 jars_per_box1 jars_per_box2 leftover_jars : ℕ) : ℕ :=
  boxes_type1 * jars_per_box1 + boxes_type2 * jars_per_box2 + leftover_jars

theorem cassy_jars_proof :
  initial_jars 10 30 12 10 80 = 500 := by
  sorry

end NUMINAMATH_CALUDE_cassy_jars_proof_l3801_380100


namespace NUMINAMATH_CALUDE_max_children_in_candy_game_l3801_380177

/-- Represents the candy distribution game. -/
structure CandyGame where
  n : ℕ  -- number of children
  k : ℕ  -- number of complete circles each child passes candies
  a : ℕ  -- number of candies each child has when the game is interrupted

/-- Checks if the game satisfies the conditions. -/
def is_valid_game (game : CandyGame) : Prop :=
  ∃ (i j : ℕ), i < game.n ∧ j < game.n ∧ i ≠ j ∧
  (game.a + 2 * game.n * game.k - 2 * i) / (game.a + 2 * game.n * game.k - 2 * j) = 13

/-- The theorem stating the maximum number of children in the game. -/
theorem max_children_in_candy_game :
  ∃ (game : CandyGame), is_valid_game game ∧
    (∀ (other_game : CandyGame), is_valid_game other_game → other_game.n ≤ game.n) ∧
    game.n = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_children_in_candy_game_l3801_380177


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_l3801_380182

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_2 :
  f' 2 = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_l3801_380182


namespace NUMINAMATH_CALUDE_polygon_with_150_degree_angles_polygon_with_14_diagonals_l3801_380157

-- Define a polygon
structure Polygon where
  sides : ℕ
  interiorAngle : ℝ
  diagonals : ℕ

-- Theorem 1: A polygon with interior angles of 150° has 12 sides
theorem polygon_with_150_degree_angles (p : Polygon) : 
  p.interiorAngle = 150 → p.sides = 12 := by sorry

-- Theorem 2: A polygon with 14 diagonals has interior angles that sum to 900°
theorem polygon_with_14_diagonals (p : Polygon) :
  p.diagonals = 14 → (p.sides - 2) * 180 = 900 := by sorry

end NUMINAMATH_CALUDE_polygon_with_150_degree_angles_polygon_with_14_diagonals_l3801_380157


namespace NUMINAMATH_CALUDE_prob_two_white_balls_prob_one_white_one_black_l3801_380134

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- Calculates the probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- Theorem stating the probability of drawing two white balls -/
theorem prob_two_white_balls : 
  probability (white_balls.choose 2) (total_balls.choose 2) = 2 / 5 := by sorry

/-- Theorem stating the probability of drawing one white ball and one black ball -/
theorem prob_one_white_one_black : 
  probability (white_balls * black_balls) (total_balls.choose 2) = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_two_white_balls_prob_one_white_one_black_l3801_380134


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_nine_l3801_380187

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_neg_two_eq_neg_nine
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x ∈ Set.Icc 1 5, f x = x^3 + 1) :
  f (-2) = -9 := by sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_nine_l3801_380187


namespace NUMINAMATH_CALUDE_triangle_concurrency_l3801_380145

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b c : ℝ)

-- Define the triangle
def Triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define perpendicular
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define reflection
def Reflection (P Q : Point) (l : Line) : Prop := sorry

-- Define intersection
def Intersect (l1 l2 : Line) : Point := sorry

-- Define concurrency
def Concurrent (l1 l2 l3 : Line) : Prop := sorry

-- Theorem statement
theorem triangle_concurrency 
  (A B C D E F H E' F' X Y : Point) 
  (ABC : Triangle A B C)
  (not_right : ¬ Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0)) -- Assuming right angle is between x and y axes
  (D_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (E_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (F_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (H_ortho : H = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0))
  (E'_refl : Reflection E E' (Line.mk 1 0 0))
  (F'_refl : Reflection F F' (Line.mk 1 0 0))
  (X_def : X = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0))
  (Y_def : Y = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0)) :
  Concurrent (Line.mk 1 0 0) (Line.mk 0 1 0) (Line.mk 1 1 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_concurrency_l3801_380145


namespace NUMINAMATH_CALUDE_find_b_l3801_380147

/-- Given two functions f and g, and a condition on their composition, prove the value of b. -/
theorem find_b (f g : ℝ → ℝ) (b : ℝ) 
  (hf : ∀ x, f x = (3 * x) / 7 + 4)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h_comp : f (g b) = 10) :
  b = -4.5 := by sorry

end NUMINAMATH_CALUDE_find_b_l3801_380147


namespace NUMINAMATH_CALUDE_max_correct_answers_l3801_380110

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 5 →
  incorrect_score = -2 →
  total_score = 60 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 14 ∧
    ∀ c, c > 14 →
      ¬∃ (i u : ℕ), c + i + u = total_questions ∧
                    correct_score * c + incorrect_score * i = total_score :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l3801_380110


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l3801_380105

theorem complex_subtraction_simplification :
  (5 - 3 * Complex.I) - (-2 + 7 * Complex.I) = 7 - 10 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l3801_380105


namespace NUMINAMATH_CALUDE_reduce_tiles_to_less_than_five_l3801_380186

/-- Represents the operation of removing prime-numbered tiles and renumbering --/
def remove_primes_and_renumber (n : ℕ) : ℕ := sorry

/-- Counts the number of operations needed to reduce the set to fewer than 5 tiles --/
def count_operations (initial_count : ℕ) : ℕ := sorry

/-- Theorem stating that 5 operations are needed to reduce 50 tiles to fewer than 5 --/
theorem reduce_tiles_to_less_than_five :
  count_operations 50 = 5 ∧ remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber (remove_primes_and_renumber 50)))) < 5 := by
  sorry

end NUMINAMATH_CALUDE_reduce_tiles_to_less_than_five_l3801_380186


namespace NUMINAMATH_CALUDE_expression_never_equals_negative_one_l3801_380192

theorem expression_never_equals_negative_one (a y : ℝ) (ha : a ≠ 0) (hy1 : y ≠ -a) (hy2 : y ≠ 2*a) :
  (2*a^2 + y^2) / (a*y - y^2 - a^2) ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_never_equals_negative_one_l3801_380192


namespace NUMINAMATH_CALUDE_special_equation_solution_l3801_380126

theorem special_equation_solution :
  ∃ x : ℝ, 9 - 8 / 7 * x + 10 = 13.285714285714286 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_equation_solution_l3801_380126


namespace NUMINAMATH_CALUDE_specific_quadrilateral_area_l3801_380167

/-- A point in the 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a quadrilateral given its four vertices. -/
def quadrilateralArea (p q r s : Point) : ℝ := sorry

/-- Theorem: The area of the quadrilateral with vertices at (7,6), (-5,1), (-2,-3), and (10,2) is 63 square units. -/
theorem specific_quadrilateral_area :
  let p : Point := ⟨7, 6⟩
  let q : Point := ⟨-5, 1⟩
  let r : Point := ⟨-2, -3⟩
  let s : Point := ⟨10, 2⟩
  quadrilateralArea p q r s = 63 := by sorry

end NUMINAMATH_CALUDE_specific_quadrilateral_area_l3801_380167


namespace NUMINAMATH_CALUDE_circle_op_inequality_solution_set_l3801_380106

-- Define the ⊙ operation
def circle_op (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem circle_op_inequality_solution_set :
  ∀ x : ℝ, circle_op x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
sorry

end NUMINAMATH_CALUDE_circle_op_inequality_solution_set_l3801_380106


namespace NUMINAMATH_CALUDE_complex_argument_one_minus_i_sqrt_three_l3801_380128

/-- The argument of the complex number 1 - i√3 is 5π/3 -/
theorem complex_argument_one_minus_i_sqrt_three (z : ℂ) : 
  z = 1 - Complex.I * Real.sqrt 3 → Complex.arg z = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_argument_one_minus_i_sqrt_three_l3801_380128


namespace NUMINAMATH_CALUDE_rotate90_matches_optionC_l3801_380133

-- Define the plane
def Plane : Type := ℝ × ℝ

-- Define the X-like shape
def XLikeShape : Type := Set Plane

-- Define rotation function
def rotate90Clockwise (shape : XLikeShape) : XLikeShape := sorry

-- Define the original shape
def originalShape : XLikeShape := sorry

-- Define the shape in option C
def optionCShape : XLikeShape := sorry

-- Theorem statement
theorem rotate90_matches_optionC : 
  rotate90Clockwise originalShape = optionCShape := by sorry

end NUMINAMATH_CALUDE_rotate90_matches_optionC_l3801_380133


namespace NUMINAMATH_CALUDE_original_number_proof_l3801_380171

theorem original_number_proof (n : ℕ) : 
  n - 7 = 62575 ∧ (62575 % 99 = 92) → n = 62582 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l3801_380171


namespace NUMINAMATH_CALUDE_prize_order_count_is_32_l3801_380148

/-- Represents a bowling tournament with 6 players and a specific playoff system. -/
structure BowlingTournament where
  players : Fin 6
  /-- The number of matches in the tournament -/
  num_matches : Nat
  /-- Each match has two possible outcomes -/
  match_outcomes : Nat → Bool

/-- The number of different possible prize orders in the tournament -/
def prizeOrderCount (t : BowlingTournament) : Nat :=
  2^t.num_matches

/-- Theorem stating that the number of different prize orders is 32 -/
theorem prize_order_count_is_32 (t : BowlingTournament) :
  prizeOrderCount t = 32 :=
by sorry

end NUMINAMATH_CALUDE_prize_order_count_is_32_l3801_380148


namespace NUMINAMATH_CALUDE_chemistry_mixture_volume_l3801_380136

theorem chemistry_mixture_volume (V : ℝ) :
  (0.6 * V + 100) / (V + 100) = 0.7 →
  V = 300 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_mixture_volume_l3801_380136


namespace NUMINAMATH_CALUDE_evaluate_logarithmic_expression_l3801_380159

theorem evaluate_logarithmic_expression :
  Real.sqrt (Real.log 8 / Real.log 3 - Real.log 8 / Real.log 2 + Real.log 8 / Real.log 4) =
  Real.sqrt (3 * (2 * Real.log 2 - Real.log 3)) / Real.sqrt (2 * Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_evaluate_logarithmic_expression_l3801_380159


namespace NUMINAMATH_CALUDE_fruit_bowl_cherries_l3801_380164

theorem fruit_bowl_cherries :
  ∀ (b s r c : ℕ),
    b + s + r + c = 360 →
    s = 2 * b →
    r = 4 * s →
    c = 2 * r →
    c = 640 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_bowl_cherries_l3801_380164


namespace NUMINAMATH_CALUDE_ellipse_parabola_shared_focus_eccentricity_l3801_380120

/-- The eccentricity of an ellipse sharing a focus with a parabola -/
theorem ellipse_parabola_shared_focus_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (hab : a > b) 
  (hb : b > 0) : 
  ∃ (x y : ℝ), 
    x^2 = 2*p*y ∧ 
    y^2/a^2 + x^2/b^2 = 1 ∧ 
    (∃ (t : ℝ), x = 2*p*t ∧ y = p*t^2) → 
    Real.sqrt 2 - 1 = Real.sqrt (1 - b^2/a^2) := by
  sorry

#check ellipse_parabola_shared_focus_eccentricity

end NUMINAMATH_CALUDE_ellipse_parabola_shared_focus_eccentricity_l3801_380120


namespace NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l3801_380178

theorem inverse_true_implies_negation_true (P : Prop) :
  (¬P → ¬P) → (¬P) :=
by sorry

end NUMINAMATH_CALUDE_inverse_true_implies_negation_true_l3801_380178


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3801_380102

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔ (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3801_380102


namespace NUMINAMATH_CALUDE_buddy_cards_thursday_l3801_380191

def baseball_cards_problem (initial_cards : ℕ) (bought_wednesday : ℕ) : ℕ :=
  let tuesday_cards := initial_cards / 2
  let wednesday_cards := tuesday_cards + bought_wednesday
  let thursday_bought := tuesday_cards / 3
  wednesday_cards + thursday_bought

theorem buddy_cards_thursday (initial_cards : ℕ) (bought_wednesday : ℕ) 
  (h1 : initial_cards = 30) (h2 : bought_wednesday = 12) : 
  baseball_cards_problem initial_cards bought_wednesday = 32 := by
  sorry

end NUMINAMATH_CALUDE_buddy_cards_thursday_l3801_380191


namespace NUMINAMATH_CALUDE_remainder_proof_l3801_380113

theorem remainder_proof (x y : ℤ) 
  (hx : x % 52 = 19) 
  (hy : (3 * y) % 7 = 5) : 
  ((x + 2*y)^2) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3801_380113


namespace NUMINAMATH_CALUDE_trigonometric_equation_l3801_380122

theorem trigonometric_equation (x : ℝ) (h : |Real.cos (2 * x)| ≠ 1) :
  8.451 * ((1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))) = (1/3) * Real.tan x ^ 4 ↔
  ∃ k : ℤ, x = π/3 * (3 * k + 1) ∨ x = π/3 * (3 * k - 1) :=
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l3801_380122


namespace NUMINAMATH_CALUDE_complex_linear_combination_l3801_380179

theorem complex_linear_combination :
  let x : ℂ := 3 + 2*I
  let y : ℂ := 2 - 3*I
  3*x + 4*y = 17 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_linear_combination_l3801_380179


namespace NUMINAMATH_CALUDE_figure_area_theorem_l3801_380197

theorem figure_area_theorem (x : ℝ) :
  let square1_area := (3 * x)^2
  let square2_area := (7 * x)^2
  let triangle_area := (1 / 2) * (3 * x) * (7 * x)
  square1_area + square2_area + triangle_area = 1300 →
  x = Real.sqrt (2600 / 137) := by
sorry

end NUMINAMATH_CALUDE_figure_area_theorem_l3801_380197


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l3801_380196

/-- If the terminal side of angle α passes through point (-2, 1), then 1/(sin 2α) = -5/4 -/
theorem angle_terminal_side_point (α : ℝ) : 
  (Real.cos α = -2 / Real.sqrt 5 ∧ Real.sin α = 1 / Real.sqrt 5) → 
  1 / Real.sin (2 * α) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l3801_380196


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l3801_380184

theorem two_digit_number_difference (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 → 
  (10 * x + y) - (10 * y + x) = 72 →
  x - y = 8 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l3801_380184


namespace NUMINAMATH_CALUDE_square_roots_problem_l3801_380176

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (2*x - 3)^2 = a) (h3 : (5 - x)^2 = a) : a = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3801_380176


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l3801_380193

theorem quadratic_completing_square : ∀ x : ℝ, x^2 - 4*x + 5 = (x - 2)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l3801_380193


namespace NUMINAMATH_CALUDE_triangle_properties_l3801_380162

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  b * Real.sin A = 2 * Real.sqrt 3 * a * (Real.cos (B / 2))^2 - Real.sqrt 3 * a ∧
  b = 4 * Real.sqrt 3 ∧
  Real.sin A * Real.cos B + Real.cos A * Real.sin B = 2 * Real.sin A →
  B = π / 3 ∧ 
  (1/2) * a * c * Real.sin B = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3801_380162


namespace NUMINAMATH_CALUDE_holly_initial_amount_l3801_380119

/-- The amount of chocolate milk Holly drinks at breakfast, in ounces. -/
def breakfast_consumption : ℕ := 8

/-- The amount of chocolate milk Holly drinks at lunch, in ounces. -/
def lunch_consumption : ℕ := 8

/-- The amount of chocolate milk Holly drinks at dinner, in ounces. -/
def dinner_consumption : ℕ := 8

/-- The amount of chocolate milk Holly ends the day with, in ounces. -/
def end_of_day_amount : ℕ := 56

/-- The size of the new container Holly buys during lunch, in ounces. -/
def new_container_size : ℕ := 64

/-- Theorem stating that Holly began the day with 80 ounces of chocolate milk. -/
theorem holly_initial_amount :
  breakfast_consumption + lunch_consumption + dinner_consumption + end_of_day_amount = 80 :=
by sorry

end NUMINAMATH_CALUDE_holly_initial_amount_l3801_380119


namespace NUMINAMATH_CALUDE_fruit_store_total_weight_l3801_380180

/-- Given a store with apples and pears, where the weight of pears is three times
    that of apples, calculate the total weight of apples and pears. -/
theorem fruit_store_total_weight (apple_weight : ℕ) (pear_weight : ℕ) : 
  apple_weight = 3200 →
  pear_weight = 3 * apple_weight →
  apple_weight + pear_weight = 12800 := by
sorry

end NUMINAMATH_CALUDE_fruit_store_total_weight_l3801_380180


namespace NUMINAMATH_CALUDE_first_valid_row_count_l3801_380127

def is_valid_arrangement (total_trees : ℕ) (num_rows : ℕ) : Prop :=
  num_rows > 0 ∧ total_trees % num_rows = 0

theorem first_valid_row_count : 
  let total_trees := 84
  ∀ (n : ℕ), n > 0 → is_valid_arrangement total_trees n →
    (is_valid_arrangement total_trees 6 ∧
     is_valid_arrangement total_trees 4) →
    2 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_first_valid_row_count_l3801_380127


namespace NUMINAMATH_CALUDE_sum_congruence_modulo_9_l3801_380151

theorem sum_congruence_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_modulo_9_l3801_380151


namespace NUMINAMATH_CALUDE_bus_stop_walk_time_l3801_380115

/-- The time taken to walk to the bus stop at the usual speed, in minutes -/
def usual_time : ℝ := 30

/-- The time taken to walk to the bus stop at 4/5 of the usual speed, in minutes -/
def slower_time : ℝ := usual_time + 6

theorem bus_stop_walk_time : usual_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_walk_time_l3801_380115


namespace NUMINAMATH_CALUDE_prob_bus_251_theorem_l3801_380183

/-- Represents the bus schedule system with two routes -/
structure BusSchedule where
  interval_152 : ℕ
  interval_251 : ℕ

/-- The probability of getting on bus No. 251 given a bus schedule -/
def prob_bus_251 (schedule : BusSchedule) : ℚ :=
  5 / 14

/-- Theorem stating the probability of getting on bus No. 251 -/
theorem prob_bus_251_theorem (schedule : BusSchedule) 
  (h1 : schedule.interval_152 = 5)
  (h2 : schedule.interval_251 = 7) :
  prob_bus_251 schedule = 5 / 14 := by
  sorry

#eval prob_bus_251 ⟨5, 7⟩

end NUMINAMATH_CALUDE_prob_bus_251_theorem_l3801_380183


namespace NUMINAMATH_CALUDE_max_ant_path_theorem_l3801_380104

/-- Represents a cube with edge length 12 cm -/
structure Cube where
  edge_length : ℝ
  edge_length_eq : edge_length = 12

/-- Represents a path on the cube's edges -/
structure CubePath where
  length : ℝ
  no_repeat : Bool

/-- The maximum distance an ant can walk on the cube's edges without repetition -/
def max_ant_path (c : Cube) : ℝ := 108

/-- Theorem stating the maximum distance an ant can walk on the cube -/
theorem max_ant_path_theorem (c : Cube) :
  ∀ (path : CubePath), path.no_repeat → path.length ≤ max_ant_path c :=
by sorry

end NUMINAMATH_CALUDE_max_ant_path_theorem_l3801_380104


namespace NUMINAMATH_CALUDE_dream_car_cost_proof_l3801_380169

/-- Calculates the cost of a dream car given monthly earnings, savings, and total earnings before purchase. -/
def dream_car_cost (monthly_earnings : ℕ) (monthly_savings : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / monthly_earnings) * monthly_savings

/-- Proves that the cost of the dream car is £45,000 given the specified conditions. -/
theorem dream_car_cost_proof :
  dream_car_cost 4000 500 360000 = 45000 := by
  sorry

end NUMINAMATH_CALUDE_dream_car_cost_proof_l3801_380169


namespace NUMINAMATH_CALUDE_james_total_toys_l3801_380101

/-- The number of toy cars James buys to maximize his discount -/
def num_cars : ℕ := 26

/-- The number of toy soldiers James buys -/
def num_soldiers : ℕ := 2 * num_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := num_cars + num_soldiers

theorem james_total_toys :
  (num_soldiers = 2 * num_cars) ∧ 
  (num_cars > 25) ∧
  (∀ n : ℕ, n > num_cars → n > 25) →
  total_toys = 78 := by
  sorry

end NUMINAMATH_CALUDE_james_total_toys_l3801_380101


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l3801_380149

theorem polar_coordinates_of_point (x y : ℝ) (h : (x, y) = (-1, Real.sqrt 3)) :
  ∃ (ρ θ : ℝ), ρ = 2 ∧ θ = 2 * Real.pi / 3 ∧ 
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l3801_380149


namespace NUMINAMATH_CALUDE_starting_number_proof_l3801_380135

theorem starting_number_proof : ∃ x : ℝ, ((x - 2 + 4) / 1) / 2 * 8 = 77 ∧ x = 17.25 := by
  sorry

end NUMINAMATH_CALUDE_starting_number_proof_l3801_380135


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l3801_380109

theorem no_positive_integer_solution :
  ¬∃ (x y : ℕ+), x^2017 - 1 = (x - 1) * (y^2015 - 1) := by
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l3801_380109


namespace NUMINAMATH_CALUDE_academy_skills_l3801_380158

theorem academy_skills (total : ℕ) (dancers : ℕ) (calligraphers : ℕ) (both : ℕ) : 
  total = 120 → 
  dancers = 88 → 
  calligraphers = 32 → 
  both = 18 → 
  total - (dancers + calligraphers - both) = 18 := by
sorry

end NUMINAMATH_CALUDE_academy_skills_l3801_380158


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l3801_380194

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (α β : Plane) (l : Line) 
  (h1 : perpendicular l α) 
  (h2 : parallel α β) : 
  perpendicular l β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l3801_380194


namespace NUMINAMATH_CALUDE_cracker_sales_percentage_increase_l3801_380152

theorem cracker_sales_percentage_increase
  (total_boxes : ℕ)
  (saturday_boxes : ℕ)
  (h1 : total_boxes = 150)
  (h2 : saturday_boxes = 60) :
  let sunday_boxes := total_boxes - saturday_boxes
  ((sunday_boxes - saturday_boxes) : ℚ) / saturday_boxes * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cracker_sales_percentage_increase_l3801_380152


namespace NUMINAMATH_CALUDE_craft_fair_ring_cost_l3801_380130

/-- Given the sales data from a craft fair, prove the cost of each ring --/
theorem craft_fair_ring_cost :
  let total_sales : ℚ := 320
  let num_necklaces : ℕ := 4
  let num_rings : ℕ := 8
  let num_earrings : ℕ := 5
  let num_bracelets : ℕ := 6
  let cost_necklace : ℚ := 20
  let cost_earrings : ℚ := 15
  let cost_ring : ℚ := 8.25
  let cost_bracelet : ℚ := 2 * cost_ring
  total_sales = num_necklaces * cost_necklace + num_rings * cost_ring +
                num_earrings * cost_earrings + num_bracelets * cost_bracelet
  → cost_ring = 8.25 := by
  sorry


end NUMINAMATH_CALUDE_craft_fair_ring_cost_l3801_380130


namespace NUMINAMATH_CALUDE_point_coordinates_l3801_380111

theorem point_coordinates (x y : ℝ) : 
  (x < 0 ∧ y > 0) →  -- Point P is in the second quadrant
  (|x| = 2) →        -- |x| = 2
  (y^2 = 1) →        -- y is the square root of 1
  (x = -2 ∧ y = 1)   -- Coordinates of P are (-2, 1)
  := by sorry

end NUMINAMATH_CALUDE_point_coordinates_l3801_380111


namespace NUMINAMATH_CALUDE_wendis_chickens_l3801_380154

theorem wendis_chickens (initial : ℕ) 
  (h1 : 2 * initial - 1 + 6 = 13) : initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendis_chickens_l3801_380154
