import Mathlib

namespace NUMINAMATH_CALUDE_average_age_of_first_and_fifth_dog_l2892_289288

def dog_ages (age1 age2 age3 age4 age5 : ℕ) : Prop :=
  age1 = 10 ∧
  age2 = age1 - 2 ∧
  age3 = age2 + 4 ∧
  age4 * 2 = age3 ∧
  age5 = age4 + 20

theorem average_age_of_first_and_fifth_dog (age1 age2 age3 age4 age5 : ℕ) :
  dog_ages age1 age2 age3 age4 age5 →
  (age1 + age5) / 2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_first_and_fifth_dog_l2892_289288


namespace NUMINAMATH_CALUDE_factorization_expr1_l2892_289226

theorem factorization_expr1 (a b : ℝ) :
  -3 * a^2 * b + 12 * a * b - 12 * b = -3 * b * (a - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_expr1_l2892_289226


namespace NUMINAMATH_CALUDE_power_comparison_l2892_289265

theorem power_comparison : 1.6^0.3 > 0.9^3.1 := by
  sorry

end NUMINAMATH_CALUDE_power_comparison_l2892_289265


namespace NUMINAMATH_CALUDE_min_fixed_amount_l2892_289285

def fixed_amount (F : ℝ) : Prop :=
  ∀ (S : ℝ), S ≥ 7750 → F + 0.04 * S ≥ 500

theorem min_fixed_amount :
  ∃ (F : ℝ), F ≥ 190 ∧ fixed_amount F :=
sorry

end NUMINAMATH_CALUDE_min_fixed_amount_l2892_289285


namespace NUMINAMATH_CALUDE_rectangle_area_and_ratio_l2892_289205

/-- Given a rectangle with original length a and width b -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The new rectangle after increasing dimensions -/
def new_rectangle (r : Rectangle) : Rectangle :=
  { length := 1.12 * r.length,
    width := 1.15 * r.width }

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: The area increase and length-to-width ratio of the rectangle -/
theorem rectangle_area_and_ratio (r : Rectangle) :
  (area (new_rectangle r) = 1.288 * area r) ∧
  (perimeter (new_rectangle r) = 1.13 * perimeter r → r.length = 2 * r.width) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_and_ratio_l2892_289205


namespace NUMINAMATH_CALUDE_karen_ham_sandwich_days_l2892_289271

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of days Karen packs peanut butter sandwiches -/
def peanut_butter_days : ℕ := 2

/-- The number of days Karen packs cake -/
def cake_days : ℕ := 1

/-- The probability of packing a ham sandwich and cake on the same day -/
def prob_ham_and_cake : ℚ := 12 / 100

/-- The number of days Karen packs ham sandwiches -/
def ham_days : ℕ := school_days - peanut_butter_days

theorem karen_ham_sandwich_days :
  ham_days = 3 ∧
  (ham_days : ℚ) / school_days * (cake_days : ℚ) / school_days = prob_ham_and_cake :=
sorry

end NUMINAMATH_CALUDE_karen_ham_sandwich_days_l2892_289271


namespace NUMINAMATH_CALUDE_exams_left_to_grade_l2892_289242

theorem exams_left_to_grade (total_exams : ℕ) (monday_percent : ℚ) (tuesday_percent : ℚ)
  (h1 : total_exams = 120)
  (h2 : monday_percent = 60 / 100)
  (h3 : tuesday_percent = 75 / 100) :
  total_exams - (monday_percent * total_exams).floor - (tuesday_percent * (total_exams - (monday_percent * total_exams).floor)).floor = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_exams_left_to_grade_l2892_289242


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2892_289214

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) := a * x^2 + 5 * x - 2

-- Define the solution set condition
def solution_set_condition (a : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (1/2 < x ∧ x < 2)

-- Theorem statement
theorem quadratic_inequality_problem (a : ℝ) (h : solution_set_condition a) :
  a = -2 ∧ 
  (∀ x, a * x^2 + 5 * x + a^2 - 1 > 0 ↔ (-1/2 < x ∧ x < 3)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2892_289214


namespace NUMINAMATH_CALUDE_function_value_at_3000_l2892_289279

/-- Given a function f: ℕ → ℕ satisfying the following properties:
  1) f(0) = 1
  2) For all x, f(x + 3) = f(x) + 2x + 3
  Prove that f(3000) = 3000001 -/
theorem function_value_at_3000 (f : ℕ → ℕ) 
  (h1 : f 0 = 1) 
  (h2 : ∀ x, f (x + 3) = f x + 2 * x + 3) : 
  f 3000 = 3000001 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_3000_l2892_289279


namespace NUMINAMATH_CALUDE_inequality_proof_l2892_289268

theorem inequality_proof :
  (∀ x y : ℝ, x^2 + y^2 + 1 > x * (y + 1)) ∧
  (∀ k : ℝ, (∀ x y : ℝ, x^2 + y^2 + 1 ≥ k * x * (y + 1)) → k ≤ Real.sqrt 2) ∧
  (∀ k : ℝ, (∀ m n : ℤ, (m : ℝ)^2 + (n : ℝ)^2 + 1 ≥ k * (m : ℝ) * ((n : ℝ) + 1)) → k ≤ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2892_289268


namespace NUMINAMATH_CALUDE_senior_mean_score_l2892_289246

-- Define the total number of students
def total_students : ℕ := 120

-- Define the overall mean score
def overall_mean : ℝ := 110

-- Define the relationship between number of seniors and juniors
def junior_senior_ratio : ℝ := 0.75

-- Define the relationship between senior and junior mean scores
def senior_junior_mean_ratio : ℝ := 1.4

-- Theorem statement
theorem senior_mean_score :
  ∃ (seniors juniors : ℕ) (senior_mean junior_mean : ℝ),
    seniors + juniors = total_students ∧
    juniors = Int.floor (junior_senior_ratio * seniors) ∧
    senior_mean = senior_junior_mean_ratio * junior_mean ∧
    (seniors * senior_mean + juniors * junior_mean) / total_students = overall_mean ∧
    Int.floor senior_mean = 124 := by
  sorry

end NUMINAMATH_CALUDE_senior_mean_score_l2892_289246


namespace NUMINAMATH_CALUDE_pascal_triangle_51_row_5th_number_l2892_289216

theorem pascal_triangle_51_row_5th_number : 
  let n : ℕ := 51  -- number of elements in the row
  let k : ℕ := 4   -- index of the number we're looking for (0-based)
  Nat.choose (n - 1) k = 220500 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_51_row_5th_number_l2892_289216


namespace NUMINAMATH_CALUDE_inscribed_sphere_theorem_l2892_289256

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphere where
  cone_base_radius : ℝ
  cone_height : ℝ
  sphere_radius : ℝ

/-- The condition that the sphere is inscribed in the cone -/
def is_inscribed (s : InscribedSphere) : Prop :=
  s.sphere_radius * (s.cone_base_radius^2 + s.cone_height^2).sqrt =
    s.cone_base_radius * (s.cone_height - s.sphere_radius)

/-- The theorem to be proved -/
theorem inscribed_sphere_theorem (b d : ℝ) :
  let s := InscribedSphere.mk 15 20 (b * d.sqrt - b)
  is_inscribed s → b + d = 12 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_sphere_theorem_l2892_289256


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l2892_289241

/-- The ratio of senior class size to junior class size -/
def class_ratio (senior_size junior_size : ℚ) : ℚ := senior_size / junior_size

theorem senior_junior_ratio 
  (senior_size junior_size : ℚ)
  (h1 : senior_size > 0)
  (h2 : junior_size > 0)
  (h3 : ∃ k : ℚ, k > 0 ∧ senior_size = k * junior_size)
  (h4 : (3/8) * senior_size + (1/4) * junior_size = (1/3) * (senior_size + junior_size)) :
  class_ratio senior_size junior_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l2892_289241


namespace NUMINAMATH_CALUDE_prob_more_surgeons_than_internists_mean_surgeons_selected_variance_surgeons_selected_l2892_289292

/-- Represents the selection of doctors for a medical outreach program. -/
structure DoctorSelection where
  total : Nat
  surgeons : Nat
  internists : Nat
  ophthalmologists : Nat
  selected : Nat

/-- The specific scenario of selecting 3 out of 6 doctors. -/
def scenario : DoctorSelection :=
  { total := 6
  , surgeons := 2
  , internists := 2
  , ophthalmologists := 2
  , selected := 3 }

/-- The probability of selecting more surgeons than internists. -/
def probMoreSurgeonsThanInternists (s : DoctorSelection) : ℚ :=
  3 / 10

/-- The mean number of surgeons selected. -/
def meanSurgeonsSelected (s : DoctorSelection) : ℚ :=
  1

/-- The variance of the number of surgeons selected. -/
def varianceSurgeonsSelected (s : DoctorSelection) : ℚ :=
  2 / 5

/-- Theorem stating the probability of selecting more surgeons than internists. -/
theorem prob_more_surgeons_than_internists :
  probMoreSurgeonsThanInternists scenario = 3 / 10 := by
  sorry

/-- Theorem stating the mean number of surgeons selected. -/
theorem mean_surgeons_selected :
  meanSurgeonsSelected scenario = 1 := by
  sorry

/-- Theorem stating the variance of the number of surgeons selected. -/
theorem variance_surgeons_selected :
  varianceSurgeonsSelected scenario = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_more_surgeons_than_internists_mean_surgeons_selected_variance_surgeons_selected_l2892_289292


namespace NUMINAMATH_CALUDE_min_sum_squares_l2892_289278

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 10) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 10 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2892_289278


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2892_289276

/-- 
Given a regular triangular pyramid with angle α between a lateral edge and a side of the base,
and a cross-section of area S made through the midpoint of a lateral edge parallel to the lateral face,
the volume V of the pyramid is (8√3 S cos²α) / (3 sin(2α)), where π/6 < α < π/2.
-/
theorem regular_triangular_pyramid_volume 
  (α : Real) 
  (S : Real) 
  (h1 : π/6 < α) 
  (h2 : α < π/2) 
  (h3 : S > 0) : 
  ∃ V : Real, V = (8 * Real.sqrt 3 * S * (Real.cos α)^2) / (3 * Real.sin (2 * α)) := by
  sorry

#check regular_triangular_pyramid_volume

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2892_289276


namespace NUMINAMATH_CALUDE_clown_balloons_l2892_289236

theorem clown_balloons (initial_balloons : ℕ) : 
  initial_balloons + 13 = 60 → initial_balloons = 47 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l2892_289236


namespace NUMINAMATH_CALUDE_triangle_area_and_perimeter_l2892_289257

theorem triangle_area_and_perimeter 
  (DE FD : ℝ) 
  (h_DE : DE = 12) 
  (h_FD : FD = 20) 
  (h_right_angle : DE * FD = 2 * (1/2 * DE * FD)) : 
  let EF := Real.sqrt (DE^2 + FD^2)
  (1/2 * DE * FD = 120) ∧ (DE + FD + EF = 32 + 2 * Real.sqrt 136) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_perimeter_l2892_289257


namespace NUMINAMATH_CALUDE_range_of_a_l2892_289233

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → a < x + 1/x) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2892_289233


namespace NUMINAMATH_CALUDE_lee_initial_savings_l2892_289203

/-- Calculates Lee's initial savings before selling action figures -/
def initial_savings (sneaker_cost : ℕ) (action_figures_sold : ℕ) (price_per_figure : ℕ) (money_left : ℕ) : ℕ :=
  sneaker_cost + money_left - (action_figures_sold * price_per_figure)

theorem lee_initial_savings :
  initial_savings 90 10 10 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_lee_initial_savings_l2892_289203


namespace NUMINAMATH_CALUDE_k_range_given_one_integer_solution_l2892_289243

/-- The inequality system has only one integer solution -/
def has_one_integer_solution (k : ℝ) : Prop :=
  ∃! (x : ℤ), (x^2 - 2*x - 8 > 0) ∧ (2*x^2 + (2*k+7)*x + 7*k < 0)

/-- The range of k -/
def k_range (k : ℝ) : Prop :=
  (k ≥ -5 ∧ k < 3) ∨ (k > 4 ∧ k ≤ 5)

/-- Theorem stating the range of k given the conditions -/
theorem k_range_given_one_integer_solution :
  ∀ k : ℝ, has_one_integer_solution k ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_k_range_given_one_integer_solution_l2892_289243


namespace NUMINAMATH_CALUDE_stream_speed_l2892_289251

/-- Prove that the speed of a stream is 3.75 km/h given the boat's travel times and distances -/
theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) 
  (h1 : downstream_distance = 100)
  (h2 : downstream_time = 8)
  (h3 : upstream_distance = 75)
  (h4 : upstream_time = 15) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧
    stream_speed = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2892_289251


namespace NUMINAMATH_CALUDE_greatest_n_condition_l2892_289262

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

def condition (n : ℕ) : Prop :=
  is_perfect_square (sum_of_squares n * (sum_of_squares (2 * n) - sum_of_squares n))

theorem greatest_n_condition :
  (1921 ≤ 2023) ∧ 
  condition 1921 ∧
  ∀ m : ℕ, (m > 1921 ∧ m ≤ 2023) → ¬(condition m) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_condition_l2892_289262


namespace NUMINAMATH_CALUDE_distance_swum_back_l2892_289210

/-- The distance a person swims back against the current -/
def swim_distance (still_water_speed : ℝ) (water_speed : ℝ) (time : ℝ) : ℝ :=
  (still_water_speed - water_speed) * time

/-- Theorem: The distance swum back against the current is 8 km -/
theorem distance_swum_back (still_water_speed : ℝ) (water_speed : ℝ) (time : ℝ)
    (h1 : still_water_speed = 8)
    (h2 : water_speed = 4)
    (h3 : time = 2) :
    swim_distance still_water_speed water_speed time = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_swum_back_l2892_289210


namespace NUMINAMATH_CALUDE_rational_expression_proof_l2892_289247

theorem rational_expression_proof (a b c : ℚ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) : 
  ∃ (q : ℚ), q = |1 / (a - b) + 1 / (b - c) + 1 / (c - a)| := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_proof_l2892_289247


namespace NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l2892_289234

/-- Given a triangle PQR with inradius r, circumradius R, and angles P, Q, R,
    prove that if r = 8, R = 25, and 2 * cos Q = cos P + cos R, then the area of the triangle is 96. -/
theorem triangle_area_with_given_conditions (P Q R : Real) (r R : ℝ) : 
  r = 8 → R = 25 → 2 * Real.cos Q = Real.cos P + Real.cos R → 
  ∃ (area : ℝ), area = 96 ∧ area = r * (R * Real.sin Q) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_conditions_l2892_289234


namespace NUMINAMATH_CALUDE_smallest_angle_satisfying_condition_l2892_289291

theorem smallest_angle_satisfying_condition : 
  ∃ (x : ℝ), x > 0 ∧ x < (π / 180) * 360 ∧ 
  Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) ∧
  (∀ (y : ℝ), 0 < y ∧ y < x → 
    Real.sin (4 * y) * Real.sin (5 * y) ≠ Real.cos (4 * y) * Real.cos (5 * y)) ∧
  x = (π / 180) * 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_satisfying_condition_l2892_289291


namespace NUMINAMATH_CALUDE_sequence_equality_l2892_289224

/-- Sequence definition -/
def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

/-- Main theorem -/
theorem sequence_equality (x : ℝ) :
  (a x 2)^2 = (a x 1) * (a x 3) →
  ∀ n ≥ 3, (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_equality_l2892_289224


namespace NUMINAMATH_CALUDE_max_ages_for_given_params_l2892_289286

/-- Calculates the maximum number of different integer ages within one standard deviation of the average age. -/
def max_different_ages (average_age : ℤ) (std_dev : ℤ) : ℕ :=
  let lower_bound := average_age - std_dev
  let upper_bound := average_age + std_dev
  (upper_bound - lower_bound + 1).toNat

/-- Theorem stating that for an average age of 10 and standard deviation of 8,
    the maximum number of different integer ages within one standard deviation is 17. -/
theorem max_ages_for_given_params :
  max_different_ages 10 8 = 17 := by
  sorry

#eval max_different_ages 10 8

end NUMINAMATH_CALUDE_max_ages_for_given_params_l2892_289286


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2892_289277

theorem sin_cos_identity (x : ℝ) (h : Real.sin (x + π/3) = 1/3) :
  Real.sin (5*π/3 - x) - Real.cos (2*x - π/3) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2892_289277


namespace NUMINAMATH_CALUDE_earliest_100_degrees_l2892_289274

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 15*t + 40

-- State the theorem
theorem earliest_100_degrees :
  ∃ t : ℝ, t ≥ 0 ∧ temperature t = 100 ∧ ∀ s, s ≥ 0 ∧ temperature s = 100 → s ≥ t :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_earliest_100_degrees_l2892_289274


namespace NUMINAMATH_CALUDE_sum_of_data_l2892_289259

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) :
  a + b + c = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_data_l2892_289259


namespace NUMINAMATH_CALUDE_hair_color_cost_l2892_289221

theorem hair_color_cost (makeup_palettes : ℕ) (palette_cost : ℚ)
  (lipsticks : ℕ) (lipstick_cost : ℚ)
  (hair_color_boxes : ℕ) (total_payment : ℚ) :
  makeup_palettes = 3 →
  palette_cost = 15 →
  lipsticks = 4 →
  lipstick_cost = 5/2 →
  hair_color_boxes = 3 →
  total_payment = 67 →
  (total_payment - (makeup_palettes * palette_cost + lipsticks * lipstick_cost)) / hair_color_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_hair_color_cost_l2892_289221


namespace NUMINAMATH_CALUDE_school_students_problem_l2892_289266

theorem school_students_problem (total : ℕ) (x : ℕ) : 
  total = 1150 →
  (total - x : ℚ) = (x : ℚ) * (total : ℚ) / 100 →
  x = 92 := by
sorry

end NUMINAMATH_CALUDE_school_students_problem_l2892_289266


namespace NUMINAMATH_CALUDE_complex_number_equation_l2892_289250

theorem complex_number_equation (z : ℂ) : (z * Complex.I = Complex.I + z) → z = (1/2 : ℂ) - (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equation_l2892_289250


namespace NUMINAMATH_CALUDE_dayan_20th_term_dayan_even_term_formula_l2892_289252

def dayan_sequence : ℕ → ℕ
| 0 => 0
| 1 => 2
| 2 => 4
| 3 => 8
| 4 => 12
| 5 => 18
| 6 => 24
| 7 => 32
| 8 => 40
| 9 => 50
| n + 10 => dayan_sequence n  -- placeholder for terms beyond 10th

theorem dayan_20th_term : dayan_sequence 19 = 200 := by
  sorry

theorem dayan_even_term_formula (n : ℕ) : dayan_sequence (2 * n - 1) = 2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_dayan_20th_term_dayan_even_term_formula_l2892_289252


namespace NUMINAMATH_CALUDE_tan_monotonic_interval_l2892_289283

/-- The monotonic increasing interval of tan(x + π/4) -/
theorem tan_monotonic_interval (k : ℤ) :
  ∀ x : ℝ, (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) →
    Monotone (fun x => Real.tan (x + π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_tan_monotonic_interval_l2892_289283


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l2892_289295

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l2892_289295


namespace NUMINAMATH_CALUDE_women_count_is_twenty_l2892_289239

/-- Represents a social event with dancing participants -/
structure DancingEvent where
  num_men : ℕ
  num_women : ℕ
  dances_per_man : ℕ
  dances_per_woman : ℕ

/-- The number of women at the event given the conditions -/
def women_count (event : DancingEvent) : ℕ :=
  (event.num_men * event.dances_per_man) / event.dances_per_woman

/-- Theorem stating that the number of women at the event is 20 -/
theorem women_count_is_twenty (event : DancingEvent) 
  (h1 : event.num_men = 15)
  (h2 : event.dances_per_man = 4)
  (h3 : event.dances_per_woman = 3) :
  women_count event = 20 := by
  sorry

end NUMINAMATH_CALUDE_women_count_is_twenty_l2892_289239


namespace NUMINAMATH_CALUDE_misread_weight_calculation_l2892_289229

/-- Proves that the misread weight in a class of 20 boys is 56 kg given the initial and correct average weights --/
theorem misread_weight_calculation (n : ℕ) (initial_avg : ℝ) (correct_avg : ℝ) (correct_weight : ℝ) :
  n = 20 →
  initial_avg = 58.4 →
  correct_avg = 58.65 →
  correct_weight = 61 →
  ∃ (misread_weight : ℝ),
    misread_weight = 56 ∧
    n * initial_avg + (correct_weight - misread_weight) = n * correct_avg :=
by
  sorry

end NUMINAMATH_CALUDE_misread_weight_calculation_l2892_289229


namespace NUMINAMATH_CALUDE_complex_number_i_properties_l2892_289209

/-- Given a complex number i such that i^2 = -1, prove the properties of i raised to different powers -/
theorem complex_number_i_properties (i : ℂ) (n : ℕ) (h : i^2 = -1) :
  i^(4*n + 1) = i ∧ i^(4*n + 2) = -1 ∧ i^(4*n + 3) = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_i_properties_l2892_289209


namespace NUMINAMATH_CALUDE_tree_planting_solution_l2892_289253

/-- Represents the tree planting problem during Arbor Day -/
structure TreePlanting where
  students : ℕ
  typeA : ℕ
  typeB : ℕ

/-- The conditions of the tree planting problem -/
def valid_tree_planting (tp : TreePlanting) : Prop :=
  3 * tp.students + 20 = tp.typeA + tp.typeB ∧
  4 * tp.students = tp.typeA + tp.typeB + 25 ∧
  30 * tp.typeA + 40 * tp.typeB ≤ 5400

/-- The theorem stating the solution to the tree planting problem -/
theorem tree_planting_solution :
  ∃ (tp : TreePlanting), valid_tree_planting tp ∧ tp.students = 45 ∧ tp.typeA ≥ 80 :=
sorry

end NUMINAMATH_CALUDE_tree_planting_solution_l2892_289253


namespace NUMINAMATH_CALUDE_problem_statement_l2892_289228

theorem problem_statement (ℓ : ℝ) (h : (1 + ℓ)^2 / (1 + ℓ^2) = 13/37) :
  (1 + ℓ)^3 / (1 + ℓ^3) = 156/1369 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2892_289228


namespace NUMINAMATH_CALUDE_fraction_difference_l2892_289294

theorem fraction_difference (m n : ℝ) (h1 : m^2 - n^2 = m*n) (h2 : m*n ≠ 0) :
  n/m - m/n = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l2892_289294


namespace NUMINAMATH_CALUDE_taxi_service_comparison_l2892_289290

-- Define the taxi services
structure TaxiService where
  initialFee : ℚ
  chargePerUnit : ℚ
  unitDistance : ℚ

def jimTaxi : TaxiService := { initialFee := 2.25, chargePerUnit := 0.35, unitDistance := 2/5 }
def susanTaxi : TaxiService := { initialFee := 3.00, chargePerUnit := 0.40, unitDistance := 1/3 }
def johnTaxi : TaxiService := { initialFee := 1.75, chargePerUnit := 0.30, unitDistance := 1/4 }

-- Function to calculate total charge
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  service.initialFee + (distance / service.unitDistance).ceil * service.chargePerUnit

-- Theorem to prove
theorem taxi_service_comparison :
  let tripDistance : ℚ := 3.6
  let jimCharge := totalCharge jimTaxi tripDistance
  let susanCharge := totalCharge susanTaxi tripDistance
  let johnCharge := totalCharge johnTaxi tripDistance
  (jimCharge < johnCharge) ∧ (johnCharge < susanCharge) := by sorry

end NUMINAMATH_CALUDE_taxi_service_comparison_l2892_289290


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2892_289245

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 770 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 17)
  (eq2 : a * b + c + d = 98)
  (eq3 : a * d + b * c = 176)
  (eq4 : c * d = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 ∧ ∃ a b c d, a^2 + b^2 + c^2 + d^2 = 770 := by
  sorry

#check max_sum_of_squares

end NUMINAMATH_CALUDE_max_sum_of_squares_l2892_289245


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l2892_289208

/-- Given a paint mixture with a ratio of red:yellow:white as 5:3:7,
    if 21 quarts of white paint are used, then 9 quarts of yellow paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) (total : ℚ) : 
  red / total = 5 / 15 →
  yellow / total = 3 / 15 →
  white / total = 7 / 15 →
  white = 21 →
  yellow = 9 := by
sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l2892_289208


namespace NUMINAMATH_CALUDE_floor_square_minus_square_floor_l2892_289280

theorem floor_square_minus_square_floor (x : ℝ) : x = 13.7 → ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 18 := by
  sorry

end NUMINAMATH_CALUDE_floor_square_minus_square_floor_l2892_289280


namespace NUMINAMATH_CALUDE_top_pyramid_volume_calculation_l2892_289237

/-- A right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ

/-- The volume of the top portion of a right square pyramid cut by a plane parallel to its base -/
def top_pyramid_volume (p : RightSquarePyramid) (cut_ratio : ℝ) : ℝ :=
  sorry

/-- The main theorem stating the volume of the top portion of the cut pyramid -/
theorem top_pyramid_volume_calculation (p : RightSquarePyramid) 
  (h_base : p.base_edge = 10 * Real.sqrt 2)
  (h_slant : p.slant_edge = 12)
  (h_cut_ratio : cut_ratio = 1/4) :
  top_pyramid_volume p cut_ratio = 84.375 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_top_pyramid_volume_calculation_l2892_289237


namespace NUMINAMATH_CALUDE_unicorn_rope_problem_l2892_289244

theorem unicorn_rope_problem (rope_length : ℝ) (tower_radius : ℝ) (rope_height : ℝ) (rope_distance : ℝ) 
  (h1 : rope_length = 30)
  (h2 : tower_radius = 10)
  (h3 : rope_height = 6)
  (h4 : rope_distance = 6) :
  ∃ (p q r : ℕ), 
    (p > 0 ∧ q > 0 ∧ r > 0) ∧ 
    Nat.Prime r ∧
    (p - Real.sqrt q) / r = 
      (rope_length * Real.sqrt ((tower_radius + rope_distance)^2 + rope_height^2)) / 
      (tower_radius + Real.sqrt ((tower_radius + rope_distance)^2 + rope_height^2)) ∧
    p + q + r = 1290 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_rope_problem_l2892_289244


namespace NUMINAMATH_CALUDE_system_solution_l2892_289269

theorem system_solution (a b c : ℝ) :
  ∃ x y z : ℝ,
  (a * x^3 + b * y = c * z^5 ∧
   a * z^3 + b * x = c * y^5 ∧
   a * y^3 + b * z = c * x^5) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   ∃ s t : ℝ, s^2 = (a + t * Real.sqrt (a^2 + 4*b*c)) / (2*c) ∧
             (x = s ∧ y = s ∧ z = s) ∧
             (t = 1 ∨ t = -1)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2892_289269


namespace NUMINAMATH_CALUDE_condition_equivalence_l2892_289273

-- Define the sets A, B, and C
def A : Set ℝ := {x | x - 2 > 0}
def B : Set ℝ := {x | x < 0}
def C : Set ℝ := {x | x * (x - 2) > 0}

-- State the theorem
theorem condition_equivalence : ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C := by
  sorry

end NUMINAMATH_CALUDE_condition_equivalence_l2892_289273


namespace NUMINAMATH_CALUDE_prob_three_blue_value_l2892_289287

/-- The number of red balls in the urn -/
def num_red : ℕ := 8

/-- The number of blue balls in the urn -/
def num_blue : ℕ := 6

/-- The total number of balls in the urn -/
def total_balls : ℕ := num_red + num_blue

/-- The number of balls drawn -/
def num_drawn : ℕ := 3

/-- The probability of drawing 3 blue balls consecutively without replacement -/
def prob_three_blue : ℚ := (num_blue.choose num_drawn) / (total_balls.choose num_drawn)

theorem prob_three_blue_value : prob_three_blue = 5/91 := by sorry

end NUMINAMATH_CALUDE_prob_three_blue_value_l2892_289287


namespace NUMINAMATH_CALUDE_average_weight_of_students_l2892_289284

theorem average_weight_of_students (girls_count boys_count : ℕ) 
  (girls_avg_weight boys_avg_weight : ℝ) :
  girls_count = 5 →
  boys_count = 5 →
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  let total_count := girls_count + boys_count
  let total_weight := girls_count * girls_avg_weight + boys_count * boys_avg_weight
  (total_weight / total_count : ℝ) = 50 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_students_l2892_289284


namespace NUMINAMATH_CALUDE_bart_mixtape_problem_l2892_289227

/-- A mixtape with two sides -/
structure Mixtape where
  first_side_songs : ℕ
  second_side_songs : ℕ
  song_length : ℕ
  total_length : ℕ

/-- The problem statement -/
theorem bart_mixtape_problem (m : Mixtape) 
  (h1 : m.second_side_songs = 4)
  (h2 : m.song_length = 4)
  (h3 : m.total_length = 40) :
  m.first_side_songs = 6 := by
  sorry


end NUMINAMATH_CALUDE_bart_mixtape_problem_l2892_289227


namespace NUMINAMATH_CALUDE_floor_e_equals_two_l2892_289296

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_e_equals_two_l2892_289296


namespace NUMINAMATH_CALUDE_soda_price_ratio_l2892_289201

theorem soda_price_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let brand_y_volume := v
  let brand_y_price := p
  let brand_z_volume := 1.3 * v
  let brand_z_price := 0.85 * p
  (brand_z_price / brand_z_volume) / (brand_y_price / brand_y_volume) = 17 / 26 := by
sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l2892_289201


namespace NUMINAMATH_CALUDE_alicia_score_l2892_289206

theorem alicia_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) (alicia_score : ℕ) : 
  total_score = 75 →
  other_players = 8 →
  avg_score = 6 →
  total_score = other_players * avg_score + alicia_score →
  alicia_score = 27 := by
sorry

end NUMINAMATH_CALUDE_alicia_score_l2892_289206


namespace NUMINAMATH_CALUDE_survey_results_l2892_289217

theorem survey_results (total : ℕ) (believe_percent : ℚ) (not_believe_percent : ℚ) 
  (h_total : total = 1240)
  (h_believe : believe_percent = 46/100)
  (h_not_believe : not_believe_percent = 31/100)
  (h_rounding : ∀ x : ℚ, 0 ≤ x → x < 1 → ⌊x * total⌋ + 1 = ⌈x * total⌉) :
  let min_believers := ⌈(believe_percent - 1/200) * total⌉
  let min_non_believers := ⌈(not_believe_percent - 1/200) * total⌉
  let max_refusals := total - min_believers - min_non_believers
  min_believers = 565 ∧ max_refusals = 296 := by
  sorry

#check survey_results

end NUMINAMATH_CALUDE_survey_results_l2892_289217


namespace NUMINAMATH_CALUDE_swim_meet_car_occupancy_l2892_289289

theorem swim_meet_car_occupancy :
  let num_cars : ℕ := 2
  let num_vans : ℕ := 3
  let people_per_van : ℕ := 3
  let max_car_capacity : ℕ := 6
  let max_van_capacity : ℕ := 8
  let additional_capacity : ℕ := 17
  
  let total_van_occupancy : ℕ := num_vans * people_per_van
  let total_max_capacity : ℕ := num_cars * max_car_capacity + num_vans * max_van_capacity
  let actual_total_occupancy : ℕ := total_max_capacity - additional_capacity
  let car_occupancy : ℕ := actual_total_occupancy - total_van_occupancy
  
  car_occupancy / num_cars = 5 :=
by sorry

end NUMINAMATH_CALUDE_swim_meet_car_occupancy_l2892_289289


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2892_289211

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  (n * (n - 1) / 2) * games_per_pair

/-- Theorem: In a chess tournament with 30 players, where each player plays 
    5 times against every other player, the total number of games is 2175 -/
theorem chess_tournament_games : num_games 30 5 = 2175 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_games_l2892_289211


namespace NUMINAMATH_CALUDE_barbi_monthly_loss_is_one_point_five_l2892_289282

/-- Represents the weight loss scenario of Barbi and Luca -/
structure WeightLossScenario where
  barbi_monthly_loss : ℝ
  months_in_year : ℕ
  luca_yearly_loss : ℝ
  luca_years : ℕ
  difference : ℝ

/-- The weight loss scenario satisfies the given conditions -/
def satisfies_conditions (scenario : WeightLossScenario) : Prop :=
  scenario.months_in_year = 12 ∧
  scenario.luca_yearly_loss = 9 ∧
  scenario.luca_years = 11 ∧
  scenario.difference = 81 ∧
  scenario.luca_yearly_loss * scenario.luca_years = 
    scenario.barbi_monthly_loss * scenario.months_in_year + scenario.difference

/-- Theorem stating that under the given conditions, Barbi's monthly weight loss is 1.5 kg -/
theorem barbi_monthly_loss_is_one_point_five 
  (scenario : WeightLossScenario) 
  (h : satisfies_conditions scenario) : 
  scenario.barbi_monthly_loss = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_barbi_monthly_loss_is_one_point_five_l2892_289282


namespace NUMINAMATH_CALUDE_hospital_staff_count_l2892_289231

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h1 : total = 250)
  (h2 : doctor_ratio = 2)
  (h3 : nurse_ratio = 3) :
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 150 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l2892_289231


namespace NUMINAMATH_CALUDE_z_extrema_l2892_289249

-- Define the triangle G
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 4}

-- Define the function z
def z (p : ℝ × ℝ) : ℝ :=
  p.1^2 + p.2^2 - 2*p.1*p.2 - p.1 - 2*p.2

theorem z_extrema :
  (∃ p ∈ G, ∀ q ∈ G, z q ≤ z p) ∧
  (∃ p ∈ G, ∀ q ∈ G, z q ≥ z p) ∧
  (∃ p ∈ G, z p = 12) ∧
  (∃ p ∈ G, z p = -1/4) :=
sorry

end NUMINAMATH_CALUDE_z_extrema_l2892_289249


namespace NUMINAMATH_CALUDE_total_cost_is_598_l2892_289200

/-- The cost of 1 kg of flour in dollars -/
def flour_cost : ℝ := 23

/-- The cost relationship between mangos and rice -/
def mango_rice_relation (mango_cost rice_cost : ℝ) : Prop :=
  10 * mango_cost = rice_cost * 10

/-- The cost relationship between flour and rice -/
def flour_rice_relation (rice_cost : ℝ) : Prop :=
  6 * flour_cost = 2 * rice_cost

/-- The total cost of the given quantities of mangos, rice, and flour -/
def total_cost (mango_cost rice_cost : ℝ) : ℝ :=
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost

/-- Theorem stating the total cost is $598 given the conditions -/
theorem total_cost_is_598 (mango_cost rice_cost : ℝ) 
  (h1 : mango_rice_relation mango_cost rice_cost)
  (h2 : flour_rice_relation rice_cost) : 
  total_cost mango_cost rice_cost = 598 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_598_l2892_289200


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2892_289219

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- Get the nth term of an arithmetic sequence. -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1 : ℝ) * seq.common_difference

/-- Our specific arithmetic sequence with given conditions. -/
def our_sequence : ArithmeticSequence :=
  { first_term := 0,  -- We don't know the first term yet, so we use a placeholder
    common_difference := 0 }  -- We don't know the common difference yet, so we use a placeholder

theorem arithmetic_sequence_problem :
  our_sequence.nthTerm 3 = 10 ∧
  our_sequence.nthTerm 20 = 65 →
  our_sequence.nthTerm 32 = 103.8235294118 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2892_289219


namespace NUMINAMATH_CALUDE_white_dandelions_on_saturday_l2892_289264

/-- Represents the state of dandelions in a meadow on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- The dandelion blooming cycle -/
def dandelionCycle : ℕ := 5

/-- The number of days a dandelion remains yellow -/
def yellowDays : ℕ := 3

/-- The state of dandelions on Monday -/
def mondayState : DandelionState :=
  { yellow := 20, white := 14 }

/-- The state of dandelions on Wednesday -/
def wednesdayState : DandelionState :=
  { yellow := 15, white := 11 }

/-- The number of days between Monday and Saturday -/
def daysToSaturday : ℕ := 5

theorem white_dandelions_on_saturday :
  (wednesdayState.yellow + wednesdayState.white) - mondayState.yellow =
  (mondayState.yellow + mondayState.white + daysToSaturday - dandelionCycle) :=
by sorry

end NUMINAMATH_CALUDE_white_dandelions_on_saturday_l2892_289264


namespace NUMINAMATH_CALUDE_pink_crayons_count_l2892_289207

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ

/-- Theorem stating the number of pink crayons in the given crayon box. -/
theorem pink_crayons_count (box : CrayonBox) : box.pink = 6 :=
  by
  have h1 : box.total = 24 := by sorry
  have h2 : box.red = 8 := by sorry
  have h3 : box.blue = 6 := by sorry
  have h4 : box.green = 4 := by sorry
  have h5 : box.green = (2 * box.blue) / 3 := by sorry
  have h6 : box.total = box.red + box.blue + box.green + box.pink := by sorry
  sorry


end NUMINAMATH_CALUDE_pink_crayons_count_l2892_289207


namespace NUMINAMATH_CALUDE_circle_M_fixed_point_l2892_289260

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Define the curve on which the center of M lies
def center_curve (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / x

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = -Real.sqrt 3 / 3 * x + 4

-- Define the line y = √3
def line_sqrt3 (x y : ℝ) : Prop :=
  y = Real.sqrt 3

-- Define the line x = 5
def line_x5 (x : ℝ) : Prop :=
  x = 5

-- Theorem statement
theorem circle_M_fixed_point :
  ∀ (O C D E F G H P : ℝ × ℝ),
    (O = (0, 0)) →
    (circle_M O.1 O.2) →
    (∃ (cx cy : ℝ), center_curve cx cy ∧ circle_M cx cy) →
    (line_l C.1 C.2) ∧ (line_l D.1 D.2) →
    (circle_M C.1 C.2) ∧ (circle_M D.1 D.2) →
    (Real.sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2) = Real.sqrt ((D.1 - O.1)^2 + (D.2 - O.2)^2)) →
    (line_sqrt3 E.1 E.2) ∧ (line_sqrt3 F.1 F.2) →
    (circle_M E.1 E.2) ∧ (circle_M F.1 F.2) →
    (line_x5 P.1) →
    (∃ (k b : ℝ), G.2 = k * G.1 + b ∧ H.2 = k * H.1 + b) →
    (circle_M G.1 G.2) ∧ (circle_M H.1 H.2) →
    (∃ (m : ℝ), G.2 - E.2 = m * (G.1 - E.1) ∧ G.2 - P.2 = m * (G.1 - P.1)) →
    (∃ (n : ℝ), H.2 - F.2 = n * (H.1 - F.1) ∧ H.2 - P.2 = n * (H.1 - P.1)) →
    (((E.1 < G.1 ∧ G.1 < F.1) ∧ (F.1 < H.1 ∨ H.1 < E.1)) ∨
     ((E.1 < H.1 ∧ H.1 < F.1) ∧ (F.1 < G.1 ∨ G.1 < E.1))) →
    ∃ (k b : ℝ), G.2 = k * G.1 + b ∧ H.2 = k * H.1 + b ∧ 2 = k * 2 + b ∧ Real.sqrt 3 = k * 2 + b :=
by sorry

end NUMINAMATH_CALUDE_circle_M_fixed_point_l2892_289260


namespace NUMINAMATH_CALUDE_integral_polynomial_l2892_289263

theorem integral_polynomial (x : ℝ) :
  deriv (fun x => x^3 - x^2 + 5*x) x = 3*x^2 - 2*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_integral_polynomial_l2892_289263


namespace NUMINAMATH_CALUDE_min_value_sum_ratios_l2892_289202

theorem min_value_sum_ratios (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hcommon_root : ∃ x : ℝ, a*x^3 + b*x + c = 0 ∧ b*x^3 + c*x + a = 0 ∧ c*x^3 + a*x + b = 0)
  (hreal_roots : ∃ (eq1 eq2 : ℕ → Prop), 
    (eq1 0 ∧ eq1 1 ∧ eq1 2) ∧ 
    (eq2 0 ∧ eq2 1 ∧ eq2 2) ∧
    (∀ x : ℂ, eq1 0 → (a*x^3 + b*x + c = 0 → x.im = 0)) ∧
    (∀ x : ℂ, eq1 1 → (b*x^3 + c*x + a = 0 → x.im = 0)) ∧
    (∀ x : ℂ, eq1 2 → (c*x^3 + a*x + b = 0 → x.im = 0)) ∧
    (∀ x : ℂ, eq2 0 → (a*x^3 + b*x + c = 0 → x.im = 0)) ∧
    (∀ x : ℂ, eq2 1 → (b*x^3 + c*x + a = 0 → x.im = 0)) ∧
    (∀ x : ℂ, eq2 2 → (c*x^3 + a*x + b = 0 → x.im = 0)) ∧
    eq1 ≠ eq2) :
  a/b + b/c + c/a = 3.833 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_ratios_l2892_289202


namespace NUMINAMATH_CALUDE_egg_distribution_l2892_289213

theorem egg_distribution (total_eggs : ℕ) (num_adults num_boys num_girls : ℕ) 
  (eggs_per_adult eggs_per_girl : ℕ) :
  total_eggs = 36 →
  num_adults = 3 →
  num_boys = 10 →
  num_girls = 7 →
  eggs_per_adult = 3 →
  eggs_per_girl = 1 →
  ∃ (eggs_per_boy : ℕ),
    total_eggs = num_adults * eggs_per_adult + num_boys * eggs_per_boy + num_girls * eggs_per_girl ∧
    eggs_per_boy = eggs_per_girl + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l2892_289213


namespace NUMINAMATH_CALUDE_rent_increase_problem_l2892_289281

/-- Proves that given the conditions of the rent increase scenario, 
    the original rent of the friend whose rent was increased was $1250 -/
theorem rent_increase_problem (num_friends : ℕ) (initial_avg : ℝ) 
  (increase_percent : ℝ) (new_avg : ℝ) : 
  num_friends = 4 →
  initial_avg = 800 →
  increase_percent = 0.16 →
  new_avg = 850 →
  ∃ (original_rent : ℝ), 
    original_rent * (1 + increase_percent) + 
    (num_friends - 1 : ℝ) * initial_avg = 
    num_friends * new_avg ∧ 
    original_rent = 1250 := by
  sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l2892_289281


namespace NUMINAMATH_CALUDE_apple_cost_is_twelve_l2892_289258

/-- The cost of an apple given the total money, number of apples, and number of kids -/
def apple_cost (total_money : ℕ) (num_apples : ℕ) (num_kids : ℕ) : ℚ :=
  (total_money : ℚ) / (num_apples : ℚ)

/-- Theorem stating that the cost of each apple is 12 dollars -/
theorem apple_cost_is_twelve :
  apple_cost 360 30 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_is_twelve_l2892_289258


namespace NUMINAMATH_CALUDE_four_cube_painted_subcubes_l2892_289298

/-- Represents a cube with some faces painted -/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ
  unpainted_faces : ℕ

/-- Calculates the number of subcubes with at least one painted face -/
def subcubes_with_paint (c : PaintedCube) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 cube with 4 painted faces has 48 subcubes with paint -/
theorem four_cube_painted_subcubes :
  let c : PaintedCube := { size := 4, painted_faces := 4, unpainted_faces := 2 }
  subcubes_with_paint c = 48 := by
  sorry

end NUMINAMATH_CALUDE_four_cube_painted_subcubes_l2892_289298


namespace NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l2892_289230

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- The distance between foci -/
def focal_distance : ℝ := 10

/-- Point P is on the hyperbola -/
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_equation P.1 P.2

/-- The distance from P to the right focus F₂ -/
def distance_PF₂ : ℝ := 7

/-- The perimeter of triangle F₁PF₂ -/
def triangle_perimeter (d_PF₁ : ℝ) : ℝ :=
  d_PF₁ + distance_PF₂ + focal_distance

theorem hyperbola_triangle_perimeter :
  ∀ P : ℝ × ℝ, point_on_hyperbola P →
  ∃ d_PF₁ : ℝ, triangle_perimeter d_PF₁ = 30 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_perimeter_l2892_289230


namespace NUMINAMATH_CALUDE_v₃_value_l2892_289220

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

/-- The value of x -/
def x : ℝ := 5

/-- The definition of v₃ -/
def v₃ : ℝ := (((5*x + 2)*x + 3.5)*x - 2.6)

/-- Theorem stating that v₃ equals 689.9 -/
theorem v₃_value : v₃ = 689.9 := by
  sorry

end NUMINAMATH_CALUDE_v₃_value_l2892_289220


namespace NUMINAMATH_CALUDE_min_max_abs_polynomial_l2892_289240

open Real

theorem min_max_abs_polynomial :
  ∃ y : ℝ, ∀ z : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 - x^3 * z| ≤ |x^2 - x^3 * y|) ∧
    |x^2 - x^3 * y| ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_polynomial_l2892_289240


namespace NUMINAMATH_CALUDE_mrs_brown_utility_bill_l2892_289297

/-- Calculates the actual utility bill amount given the initial payment and returned amount -/
def actualUtilityBill (initialPayment returnedAmount : ℕ) : ℕ :=
  initialPayment - returnedAmount

/-- Theorem stating that Mrs. Brown's actual utility bill is $710 -/
theorem mrs_brown_utility_bill :
  let initialPayment := 4 * 100 + 5 * 50 + 7 * 20
  let returnedAmount := 3 * 20 + 2 * 10
  actualUtilityBill initialPayment returnedAmount = 710 := by
  sorry

#eval actualUtilityBill (4 * 100 + 5 * 50 + 7 * 20) (3 * 20 + 2 * 10)

end NUMINAMATH_CALUDE_mrs_brown_utility_bill_l2892_289297


namespace NUMINAMATH_CALUDE_sandy_marks_l2892_289267

/-- Sandy's marks calculation -/
theorem sandy_marks :
  ∀ (total_sums correct_sums : ℕ)
    (marks_per_correct marks_lost_per_incorrect : ℕ),
  total_sums = 30 →
  correct_sums = 23 →
  marks_per_correct = 3 →
  marks_lost_per_incorrect = 2 →
  (marks_per_correct * correct_sums) -
  (marks_lost_per_incorrect * (total_sums - correct_sums)) = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_l2892_289267


namespace NUMINAMATH_CALUDE_base3_of_256_l2892_289248

/-- Converts a base-10 number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

theorem base3_of_256 :
  toBase3 256 = [1, 0, 1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base3_of_256_l2892_289248


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2892_289238

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 450 → percentage = 75 → result = initial * (1 + percentage / 100) → result = 787.5 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2892_289238


namespace NUMINAMATH_CALUDE_negation_existential_square_positive_l2892_289270

theorem negation_existential_square_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_existential_square_positive_l2892_289270


namespace NUMINAMATH_CALUDE_prob_green_ball_is_13_30_l2892_289223

/-- Represents a container of balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def probGreenFromContainer (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The probability of selecting a specific container -/
def probSelectContainer (numContainers : ℕ) : ℚ :=
  1 / numContainers

theorem prob_green_ball_is_13_30 
  (containerX containerY containerZ : Container)
  (h1 : containerX = { red := 3, green := 7 })
  (h2 : containerY = { red := 7, green := 3 })
  (h3 : containerZ = { red := 7, green := 3 })
  (h4 : probSelectContainer 3 = 1/3) :
  probSelectContainer 3 * probGreenFromContainer containerX +
  probSelectContainer 3 * probGreenFromContainer containerY +
  probSelectContainer 3 * probGreenFromContainer containerZ = 13/30 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_ball_is_13_30_l2892_289223


namespace NUMINAMATH_CALUDE_complement_of_B_in_U_l2892_289293

open Set

theorem complement_of_B_in_U (U A B : Set ℕ) : 
  U = A ∪ B → 
  A = {1, 2, 3} → 
  A ∩ B = {1} → 
  U \ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_U_l2892_289293


namespace NUMINAMATH_CALUDE_m_range_l2892_289275

-- Define the conditions
def P (x : ℝ) : Prop := x^2 - 3*x + 2 > 0
def q (x m : ℝ) : Prop := x < m

-- Define the theorem
theorem m_range (m : ℝ) : 
  (∀ x, ¬(P x) → q x m) ∧ (∃ x, q x m ∧ P x) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2892_289275


namespace NUMINAMATH_CALUDE_lcm_of_8_and_12_l2892_289225

theorem lcm_of_8_and_12 :
  let a : ℕ := 8
  let b : ℕ := 12
  let hcf : ℕ := 4
  (Nat.gcd a b = hcf) → (Nat.lcm a b = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_and_12_l2892_289225


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2892_289212

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 7 * Real.pi / 4
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 2 * Real.sqrt 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2892_289212


namespace NUMINAMATH_CALUDE_infinitely_many_primes_l2892_289204

theorem infinitely_many_primes : ∀ S : Finset Nat, (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_l2892_289204


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2892_289272

def digits : List Nat := [0, 2, 4, 6, 8, 9]

def is_valid_number (n : Nat) : Prop :=
  let digits_used := n.digits 10
  (digits_used.toFinset = digits.toFinset) ∧ 
  (digits_used.length = digits.length) ∧
  (n ≥ 100000)

theorem smallest_number_proof :
  (is_valid_number 204689) ∧ 
  (∀ m : Nat, is_valid_number m → m ≥ 204689) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2892_289272


namespace NUMINAMATH_CALUDE_distribution_proportion_l2892_289261

theorem distribution_proportion (total : ℚ) (p q r s : ℚ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  s - p = 250 →
  p + q + r + s = total →
  q / r = 1 := by
  sorry

end NUMINAMATH_CALUDE_distribution_proportion_l2892_289261


namespace NUMINAMATH_CALUDE_cars_sold_last_three_days_l2892_289218

/-- Represents the number of cars sold by a salesman over 6 days -/
structure CarSales where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  day5 : ℕ
  day6 : ℕ

/-- Calculates the mean of car sales over 6 days -/
def meanSales (sales : CarSales) : ℚ :=
  (sales.day1 + sales.day2 + sales.day3 + sales.day4 + sales.day5 + sales.day6 : ℚ) / 6

/-- Theorem stating the number of cars sold in the last three days -/
theorem cars_sold_last_three_days (sales : CarSales) 
  (h1 : sales.day1 = 8)
  (h2 : sales.day2 = 3)
  (h3 : sales.day3 = 10)
  (h_mean : meanSales sales = 5.5) :
  sales.day4 + sales.day5 + sales.day6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cars_sold_last_three_days_l2892_289218


namespace NUMINAMATH_CALUDE_binomial_20_19_equals_20_l2892_289255

theorem binomial_20_19_equals_20 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_equals_20_l2892_289255


namespace NUMINAMATH_CALUDE_min_value_expression_l2892_289232

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2892_289232


namespace NUMINAMATH_CALUDE_probability_above_curve_l2892_289299

-- Define the set of single-digit positive integers
def SingleDigitPos : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

-- Define the condition for (a,c) to be above the curve
def AboveCurve (a c : ℕ) : Prop := ∀ x : ℝ, c > a * x^3 - c * x^2

-- Define the count of valid points
def ValidPointsCount : ℕ := 16

-- Define the total number of possible points
def TotalPointsCount : ℕ := 81

-- State the theorem
theorem probability_above_curve :
  (↑ValidPointsCount / ↑TotalPointsCount : ℚ) = 16/81 :=
sorry

end NUMINAMATH_CALUDE_probability_above_curve_l2892_289299


namespace NUMINAMATH_CALUDE_tan_sin_identity_l2892_289222

theorem tan_sin_identity : 2 * Real.tan (10 * π / 180) + 3 * Real.sin (10 * π / 180) = 5 * Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_identity_l2892_289222


namespace NUMINAMATH_CALUDE_power_multiplication_division_equality_l2892_289254

theorem power_multiplication_division_equality : (12 : ℕ)^1 * 6^4 / 432 = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_division_equality_l2892_289254


namespace NUMINAMATH_CALUDE_magazine_selling_price_l2892_289215

/-- Given the cost price, number of magazines, and total gain, 
    calculate the selling price per magazine. -/
theorem magazine_selling_price 
  (cost_price : ℝ) 
  (num_magazines : ℕ) 
  (total_gain : ℝ) 
  (h1 : cost_price = 3)
  (h2 : num_magazines = 10)
  (h3 : total_gain = 5) :
  (cost_price * num_magazines + total_gain) / num_magazines = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_magazine_selling_price_l2892_289215


namespace NUMINAMATH_CALUDE_sophie_donuts_to_sister_l2892_289235

/-- The number of donuts Sophie gave to her sister --/
def donuts_to_sister (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_to_mom : ℕ) (donuts_for_self : ℕ) : ℕ :=
  total_boxes * donuts_per_box - boxes_to_mom * donuts_per_box - donuts_for_self

theorem sophie_donuts_to_sister :
  donuts_to_sister 4 12 1 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sophie_donuts_to_sister_l2892_289235
