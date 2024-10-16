import Mathlib

namespace NUMINAMATH_CALUDE_max_students_distribution_max_students_is_184_l3112_311288

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem max_students_distribution (pens pencils markers : ℕ) : Prop :=
  pens = 1080 →
  pencils = 920 →
  markers = 680 →
  ∃ (students : ℕ) (pens_per_student pencils_per_student markers_per_student : ℕ),
    students > 0 ∧
    students * pens_per_student = pens ∧
    students * pencils_per_student = pencils ∧
    students * markers_per_student = markers ∧
    pens_per_student > 0 ∧
    pencils_per_student > 0 ∧
    markers_per_student > 0 ∧
    is_prime pencils_per_student ∧
    ∀ (n : ℕ), n > students →
      ¬(∃ (p q r : ℕ),
        p > 0 ∧ q > 0 ∧ r > 0 ∧
        is_prime q ∧
        n * p = pens ∧
        n * q = pencils ∧
        n * r = markers)

theorem max_students_is_184 : max_students_distribution 1080 920 680 → 
  ∃ (pens_per_student pencils_per_student markers_per_student : ℕ),
    184 * pens_per_student = 1080 ∧
    184 * pencils_per_student = 920 ∧
    184 * markers_per_student = 680 ∧
    pens_per_student > 0 ∧
    pencils_per_student > 0 ∧
    markers_per_student > 0 ∧
    is_prime pencils_per_student :=
by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_max_students_is_184_l3112_311288


namespace NUMINAMATH_CALUDE_trig_identity_l3112_311225

theorem trig_identity (α : ℝ) :
  3.410 * (Real.sin (2 * α))^3 * Real.cos (6 * α) + 
  (Real.cos (2 * α))^3 * Real.sin (6 * α) = 
  3/4 * Real.sin (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3112_311225


namespace NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_l3112_311267

theorem sqrt_3_times_612_times_3_and_half : Real.sqrt 3 * 612 * (3 + 3/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_l3112_311267


namespace NUMINAMATH_CALUDE_tan_product_zero_l3112_311273

theorem tan_product_zero (a b : ℝ) 
  (h : 6 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) : 
  Real.tan (a / 2) * Real.tan (b / 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_product_zero_l3112_311273


namespace NUMINAMATH_CALUDE_probability_matching_shoes_l3112_311269

theorem probability_matching_shoes (n : ℕ) (h : n = 9) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_matching_shoes_l3112_311269


namespace NUMINAMATH_CALUDE_smallest_disjoint_r_l3112_311296

def A : Set ℤ := {n | ∃ k : ℕ, (n = 3 + 10 * k) ∨ (n = 6 + 26 * k) ∨ (n = 5 + 29 * k)}

def is_disjoint (r b : ℤ) : Prop :=
  ∀ k l : ℕ, (b + r * k) ∉ A

theorem smallest_disjoint_r : 
  (∃ b : ℤ, is_disjoint 290 b) ∧ 
  (∀ r : ℕ, r < 290 → ¬∃ b : ℤ, is_disjoint r b) :=
sorry

end NUMINAMATH_CALUDE_smallest_disjoint_r_l3112_311296


namespace NUMINAMATH_CALUDE_expression_evaluation_l3112_311242

theorem expression_evaluation : 
  |Real.sqrt 3 - 2| + (Real.pi - Real.sqrt 10)^0 - Real.sqrt 12 = 3 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3112_311242


namespace NUMINAMATH_CALUDE_cube_difference_l3112_311232

theorem cube_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l3112_311232


namespace NUMINAMATH_CALUDE_calculation_difference_l3112_311283

theorem calculation_difference : ∀ x : ℝ, (x - 3) + 49 = 66 → (3 * x + 49) - 66 = 43 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l3112_311283


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3112_311297

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3112_311297


namespace NUMINAMATH_CALUDE_coin_probability_l3112_311206

/-- The probability of a specific sequence of coin flips -/
def sequence_probability (p : ℝ) : ℝ := p^2 * (1 - p)^3

/-- Theorem: If the probability of getting heads on the first 2 flips
    and tails on the last 3 flips is 1/32, then the probability of
    getting heads on a single flip is 1/2 -/
theorem coin_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : sequence_probability p = 1/32) : 
  p = 1/2 := by
  sorry

#check coin_probability

end NUMINAMATH_CALUDE_coin_probability_l3112_311206


namespace NUMINAMATH_CALUDE_proportion_problem_l3112_311240

theorem proportion_problem (x : ℝ) : x / 12 = 9 / 360 → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l3112_311240


namespace NUMINAMATH_CALUDE_least_value_theorem_l3112_311215

theorem least_value_theorem (x y z : ℕ+) (h : 2 * x.val = 5 * y.val ∧ 5 * y.val = 6 * z.val) :
  ∃ n : ℤ, x.val + y.val + n = 26 ∧ ∀ m : ℤ, x.val + y.val + m = 26 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_value_theorem_l3112_311215


namespace NUMINAMATH_CALUDE_initial_trees_count_l3112_311272

theorem initial_trees_count (died cut left : ℕ) 
  (h1 : died = 15)
  (h2 : cut = 23)
  (h3 : left = 48) :
  died + cut + left = 86 := by
  sorry

end NUMINAMATH_CALUDE_initial_trees_count_l3112_311272


namespace NUMINAMATH_CALUDE_complex_power_result_l3112_311245

theorem complex_power_result : (((Complex.I * Real.sqrt 2) / (1 + Complex.I)) ^ 100 : ℂ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l3112_311245


namespace NUMINAMATH_CALUDE_min_value_theorem_l3112_311226

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (x - 1/y)^2 = 16*y/x) :
  (∀ x' y', x' > 0 → y' > 0 → (x' - 1/y')^2 = 16*y'/x' → x + 1/y ≤ x' + 1/y') →
  x^2 + 1/y^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3112_311226


namespace NUMINAMATH_CALUDE_molecular_weight_NaOCl_approx_l3112_311210

/-- The atomic weight of Sodium in g/mol -/
def atomic_weight_Na : ℝ := 22.99

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The molecular weight of NaOCl in g/mol -/
def molecular_weight_NaOCl : ℝ := atomic_weight_Na + atomic_weight_O + atomic_weight_Cl

/-- Theorem stating that the molecular weight of NaOCl is approximately 74.44 g/mol -/
theorem molecular_weight_NaOCl_approx :
  ∀ ε > 0, |molecular_weight_NaOCl - 74.44| < ε :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_NaOCl_approx_l3112_311210


namespace NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l3112_311223

/-- The maximum distance from a point on the ellipse x^2/4 + y^2 = 1 to the line x + 2y = 0 -/
theorem max_distance_ellipse_to_line :
  let ellipse := {P : ℝ × ℝ | P.1^2/4 + P.2^2 = 1}
  let line := {P : ℝ × ℝ | P.1 + 2*P.2 = 0}
  ∃ (d : ℝ), d = 2*Real.sqrt 10/5 ∧
    ∀ P ∈ ellipse, ∀ Q ∈ line,
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ d ∧
      ∃ P' ∈ ellipse, ∃ Q' ∈ line,
        Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) = d :=
by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_to_line_l3112_311223


namespace NUMINAMATH_CALUDE_at_least_one_good_part_l3112_311216

theorem at_least_one_good_part (total : ℕ) (good : ℕ) (defective : ℕ) (pick : ℕ) :
  total = 20 →
  good = 16 →
  defective = 4 →
  pick = 3 →
  total = good + defective →
  (Nat.choose total pick) - (Nat.choose defective pick) = 1136 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_good_part_l3112_311216


namespace NUMINAMATH_CALUDE_students_guinea_pigs_difference_l3112_311278

theorem students_guinea_pigs_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 22 →
  guinea_pigs_per_class = 3 →
  num_classes = 5 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 95 :=
by sorry

end NUMINAMATH_CALUDE_students_guinea_pigs_difference_l3112_311278


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3112_311284

theorem min_value_quadratic (x : ℝ) : x^2 - 3*x + 2023 ≥ 2020 + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3112_311284


namespace NUMINAMATH_CALUDE_project_time_remaining_l3112_311274

/-- Given a student's project time allocation, prove the remaining time for report writing. -/
theorem project_time_remaining (total hours_research hours_proposal hours_visual_aids hours_editing : ℕ) 
  (h_total : total = 25)
  (h_research : hours_research = 10)
  (h_proposal : hours_proposal = 2)
  (h_visual_aids : hours_visual_aids = 5)
  (h_editing : hours_editing = 3) :
  total - (hours_research + hours_proposal + hours_visual_aids + hours_editing) = 5 := by
  sorry

end NUMINAMATH_CALUDE_project_time_remaining_l3112_311274


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3112_311261

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A (domain of f(x))
def A : Set ℝ := {x | x ≥ Real.exp 1}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = Set.Ioo 0 (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3112_311261


namespace NUMINAMATH_CALUDE_line_AB_equation_min_area_and_B_coords_l3112_311201

-- Define the line l: y = 4x
def line_l (x y : ℝ) : Prop := y = 4 * x

-- Define point P
def point_P : ℝ × ℝ := (6, 4)

-- Define that point A is in the first quadrant and lies on line l
def point_A_condition (A : ℝ × ℝ) : Prop :=
  A.1 > 0 ∧ A.2 > 0 ∧ line_l A.1 A.2

-- Define that line PA intersects the positive half of the x-axis at point B
def point_B_condition (A B : ℝ × ℝ) : Prop :=
  B.2 = 0 ∧ B.1 > 0 ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  B.1 = t * A.1 + (1 - t) * point_P.1 ∧
  B.2 = t * A.2 + (1 - t) * point_P.2

-- Theorem for part (1)
theorem line_AB_equation (A B : ℝ × ℝ) 
  (h1 : point_A_condition A) 
  (h2 : point_B_condition A B) 
  (h3 : (A.2 - point_P.2) * (B.1 - point_P.1) = -(A.1 - point_P.1) * (B.2 - point_P.2)) :
  ∃ k c : ℝ, k = -3/2 ∧ c = 13 ∧ ∀ x y : ℝ, y = k * x + c ↔ 3 * x + 2 * y - 26 = 0 :=
sorry

-- Theorem for part (2)
theorem min_area_and_B_coords (A B : ℝ × ℝ) 
  (h1 : point_A_condition A) 
  (h2 : point_B_condition A B) :
  ∃ S_min : ℝ, S_min = 40 ∧
  (∀ A' B' : ℝ × ℝ, point_A_condition A' → point_B_condition A' B' →
    1/2 * A'.1 * B'.2 - 1/2 * A'.2 * B'.1 ≥ S_min) ∧
  (∃ A_min B_min : ℝ × ℝ, 
    point_A_condition A_min ∧ 
    point_B_condition A_min B_min ∧
    1/2 * A_min.1 * B_min.2 - 1/2 * A_min.2 * B_min.1 = S_min ∧
    B_min = (10, 0)) :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_min_area_and_B_coords_l3112_311201


namespace NUMINAMATH_CALUDE_red_balls_count_l3112_311231

theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 30 →
  yellow = 8 →
  purple = 3 →
  prob = 88/100 →
  prob = (white + green + yellow : ℚ) / total →
  ∃ red : ℕ, red = 9 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l3112_311231


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l3112_311271

/-- A cubic polynomial p(x) satisfying specific conditions -/
def p (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 3

/-- Theorem stating that p(x) satisfies the given conditions -/
theorem p_satisfies_conditions :
  p 1 = -7 ∧ p 2 = -9 ∧ p 3 = -15 ∧ p 4 = -31 := by
  sorry

#eval p 1
#eval p 2
#eval p 3
#eval p 4

end NUMINAMATH_CALUDE_p_satisfies_conditions_l3112_311271


namespace NUMINAMATH_CALUDE_percent_calculation_l3112_311202

theorem percent_calculation :
  (0.02 / 100) * 12356 = 2.4712 := by sorry

end NUMINAMATH_CALUDE_percent_calculation_l3112_311202


namespace NUMINAMATH_CALUDE_skittles_shared_l3112_311253

theorem skittles_shared (starting_amount ending_amount : ℕ) 
  (h1 : starting_amount = 76)
  (h2 : ending_amount = 4) :
  starting_amount - ending_amount = 72 := by
  sorry

end NUMINAMATH_CALUDE_skittles_shared_l3112_311253


namespace NUMINAMATH_CALUDE_abc_ratio_theorem_l3112_311277

theorem abc_ratio_theorem (a b c : ℚ) 
  (h : (|a|/a) + (|b|/b) + (|c|/c) = 1) : 
  a * b * c / |a * b * c| = -1 := by
sorry

end NUMINAMATH_CALUDE_abc_ratio_theorem_l3112_311277


namespace NUMINAMATH_CALUDE_race_difference_l3112_311259

/-- Represents a racer in the competition -/
structure Racer where
  time : ℝ  -- Time taken to complete the race in seconds
  speed : ℝ  -- Speed of the racer in meters per second

/-- Calculates the distance covered by a racer in a given time -/
def distance_covered (r : Racer) (t : ℝ) : ℝ := r.speed * t

theorem race_difference (race_distance : ℝ) (a b : Racer) 
  (h1 : race_distance = 80)
  (h2 : a.time = 20)
  (h3 : b.time = 25)
  (h4 : a.speed = race_distance / a.time)
  (h5 : b.speed = race_distance / b.time) :
  race_distance - distance_covered b a.time = 16 := by
  sorry

end NUMINAMATH_CALUDE_race_difference_l3112_311259


namespace NUMINAMATH_CALUDE_mark_current_age_l3112_311230

/-- Mark's current age -/
def mark_age : ℕ := 28

/-- Aaron's current age -/
def aaron_age : ℕ := 11

/-- Theorem stating that Mark's current age is 28, given the conditions about their ages -/
theorem mark_current_age :
  (mark_age - 3 = 3 * (aaron_age - 3) + 1) ∧
  (mark_age + 4 = 2 * (aaron_age + 4) + 2) →
  mark_age = 28 := by
  sorry


end NUMINAMATH_CALUDE_mark_current_age_l3112_311230


namespace NUMINAMATH_CALUDE_button_collection_value_l3112_311276

theorem button_collection_value (total_buttons : ℕ) (sample_buttons : ℕ) (sample_value : ℚ) :
  total_buttons = 10 →
  sample_buttons = 2 →
  sample_value = 8 →
  (sample_value / sample_buttons) * total_buttons = 40 := by
sorry

end NUMINAMATH_CALUDE_button_collection_value_l3112_311276


namespace NUMINAMATH_CALUDE_triangle_side_and_angle_l3112_311235

open Real

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem triangle_side_and_angle (t : Triangle) :
  t.perimeter = Real.sqrt 3 + 1 →
  sin t.B + sin t.C = Real.sqrt 3 * sin t.A →
  t.c = 1 ∧
  (t.perimeter = Real.sqrt 3 + 1 →
   sin t.B + sin t.C = Real.sqrt 3 * sin t.A →
   (1/2) * t.a * t.b * sin t.A = (1/3) * sin t.A →
   t.A = π/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_and_angle_l3112_311235


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l3112_311282

/-- The perimeter of a semi-circle with radius 21.977625925131363 cm is approximately 113.024 cm. -/
theorem semicircle_perimeter_approx : 
  let r : ℝ := 21.977625925131363
  let π : ℝ := Real.pi
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 113.024) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l3112_311282


namespace NUMINAMATH_CALUDE_incorrect_value_at_three_l3112_311281

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- Calculates the y-value for a given x-value using the linear function -/
def LinearFunction.eval (f : LinearFunction) (x : ℝ) : ℝ :=
  f.k * x + f.b

/-- Theorem: The value -2 for x = 3 is incorrect for the linear function passing through (-1, 3) and (0, 2) -/
theorem incorrect_value_at_three (f : LinearFunction) 
  (h1 : f.eval (-1) = 3)
  (h2 : f.eval 0 = 2) : 
  f.eval 3 ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_value_at_three_l3112_311281


namespace NUMINAMATH_CALUDE_lcm_of_8_9_10_21_l3112_311218

theorem lcm_of_8_9_10_21 : Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 21)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_9_10_21_l3112_311218


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l3112_311238

/-- The number of ways to distribute students among attractions -/
def distribute_students (n m k : ℕ) : ℕ :=
  Nat.choose n k * (m - 1)^(n - k)

/-- Theorem: The number of ways to distribute 6 students among 6 attractions,
    where exactly 2 students visit a specific attraction, is C₆² × 5⁴ -/
theorem student_distribution_theorem :
  distribute_students 6 6 2 = Nat.choose 6 2 * 5^4 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l3112_311238


namespace NUMINAMATH_CALUDE_distribute_7_4_l3112_311270

/-- The number of ways to distribute n identical objects into k identical containers,
    with each container containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 identical apples into 4 identical packages,
    with each package containing at least one apple. -/
theorem distribute_7_4 : distribute 7 4 = 350 := by sorry

end NUMINAMATH_CALUDE_distribute_7_4_l3112_311270


namespace NUMINAMATH_CALUDE_binomial_variance_problem_l3112_311279

/-- A function representing the expectation of a binomial distribution -/
def expectation_binomial (n : ℕ) (p : ℝ) : ℝ := n * p

/-- A function representing the variance of a binomial distribution -/
def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_problem (p : ℝ) 
  (hX : expectation_binomial 3 p = 1) 
  (hY : expectation_binomial 4 p = 4 * p) :
  variance_binomial 4 p = 8/9 := by
sorry

end NUMINAMATH_CALUDE_binomial_variance_problem_l3112_311279


namespace NUMINAMATH_CALUDE_special_polygon_area_l3112_311205

/-- A polygon with 24 congruent sides, where each side is perpendicular to its adjacent sides -/
structure SpecialPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  sides_eq : sides = 24
  perimeter_eq : perimeter = 48
  perimeter_formula : perimeter = sides * side_length

/-- The area of the special polygon is 64 -/
theorem special_polygon_area (p : SpecialPolygon) : 16 * p.side_length ^ 2 = 64 := by
  sorry

#check special_polygon_area

end NUMINAMATH_CALUDE_special_polygon_area_l3112_311205


namespace NUMINAMATH_CALUDE_bridge_length_l3112_311292

/-- The length of a bridge given train parameters --/
theorem bridge_length
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (h1 : train_length = 100)
  (h2 : crossing_time = 26.997840172786177)
  (h3 : train_speed_kmph = 36) :
  ∃ (bridge_length : ℝ),
    bridge_length = 169.97840172786177 :=
by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l3112_311292


namespace NUMINAMATH_CALUDE_shifted_sine_symmetry_l3112_311255

open Real

theorem shifted_sine_symmetry (φ : Real) (h1 : 0 < φ) (h2 : φ < π) :
  let f : Real → Real := λ x ↦ sin (3 * x + φ)
  let g : Real → Real := λ x ↦ f (x - π / 12)
  (∀ x, g x = g (-x)) → φ = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_shifted_sine_symmetry_l3112_311255


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l3112_311233

/-- Represents the number of chocolate bars in the colossal box -/
def chocolate_bars_in_colossal_box : ℕ :=
  let sizable_boxes : ℕ := 350
  let small_boxes_per_sizable : ℕ := 49
  let chocolate_bars_per_small : ℕ := 75
  sizable_boxes * small_boxes_per_sizable * chocolate_bars_per_small

/-- Proves that the number of chocolate bars in the colossal box is 1,287,750 -/
theorem chocolate_bars_count : chocolate_bars_in_colossal_box = 1287750 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l3112_311233


namespace NUMINAMATH_CALUDE_base6_multiplication_l3112_311214

/-- Converts a base-6 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base-10 number to base-6 --/
def toBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

theorem base6_multiplication :
  toBase6 (toBase10 [6] * toBase10 [1, 2]) = [2, 1, 0] := by sorry

end NUMINAMATH_CALUDE_base6_multiplication_l3112_311214


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l3112_311265

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2

theorem y1_less_than_y2 (y₁ y₂ : ℝ) :
  quadratic_function (-1) = y₁ →
  quadratic_function 4 = y₂ →
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l3112_311265


namespace NUMINAMATH_CALUDE_movie_ticket_ratio_l3112_311275

def horror_tickets : ℕ := 93
def romance_tickets : ℕ := 25
def ticket_difference : ℕ := 18

theorem movie_ticket_ratio :
  (horror_tickets : ℚ) / romance_tickets = 93 / 25 ∧
  horror_tickets = romance_tickets + ticket_difference :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_ratio_l3112_311275


namespace NUMINAMATH_CALUDE_unique_number_theorem_l3112_311258

/-- A function that checks if a number n can be expressed as 2a + xb 
    where a and b are positive integers --/
def isExpressible (n x : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ n = 2 * a + x * b

/-- The main theorem stating that 5 is the unique number satisfying the condition --/
theorem unique_number_theorem :
  ∃! (x : ℕ), x > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = 8 ∧ 
    (∀ n ∈ S, n < 15 ∧ isExpressible n x) ∧
    (∀ n < 15, isExpressible n x → n ∈ S)) ∧
  x = 5 := by
  sorry


end NUMINAMATH_CALUDE_unique_number_theorem_l3112_311258


namespace NUMINAMATH_CALUDE_scale_drawing_conversion_l3112_311234

/-- Given a scale where 1 inch represents 1000 feet, 
    a line segment of 3.6 inches represents 3600 feet. -/
theorem scale_drawing_conversion (scale : ℝ) (drawing_length : ℝ) :
  scale = 1000 →
  drawing_length = 3.6 →
  drawing_length * scale = 3600 := by
  sorry

end NUMINAMATH_CALUDE_scale_drawing_conversion_l3112_311234


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3112_311224

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3112_311224


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3112_311293

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h1 : a > 0) (h2 : d > 0) (h3 : k > 0) :
  a = 3 ∧ d = 1 ∧ k = 2 →
  (a + k * d) ^ 2 = a ^ 2 + (a + d) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3112_311293


namespace NUMINAMATH_CALUDE_min_distance_sum_to_y_axis_l3112_311289

/-- The problem of finding the minimum distance sum from two fixed points to a point on the y-axis. -/
theorem min_distance_sum_to_y_axis (A B C : ℝ × ℝ) (k : ℝ) : 
  A = (7, 3) →
  B = (3, 0) →
  C = (0, k) →
  (∀ t : ℝ, (7 - 0)^2 + (3 - k)^2 + (3 - 0)^2 + (0 - k)^2 
           ≤ (7 - 0)^2 + (3 - t)^2 + (3 - 0)^2 + (0 - t)^2) →
  k = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_to_y_axis_l3112_311289


namespace NUMINAMATH_CALUDE_divisor_35_power_l3112_311241

theorem divisor_35_power (k : ℕ) : 35^k ∣ 1575320897 → 7^k - k^7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_35_power_l3112_311241


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3112_311236

theorem algebraic_expression_value :
  let a : ℝ := Real.sqrt 2 + 1
  let b : ℝ := Real.sqrt 2 - 1
  (a^2 - 2*a*b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3112_311236


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3112_311260

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 = 1
  let c := Real.sqrt (a^2 + 1)
  let e := c / a
  let asymptote_slope := 1 / a
  let perpendicular_slope := -a
  let y_coordinate_P := a * c / (1 + a^2)
  y_coordinate_P = 2 * Real.sqrt 5 / 5 →
  e = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3112_311260


namespace NUMINAMATH_CALUDE_tree_initial_height_l3112_311207

/-- Represents the height of a tree over time -/
def TreeHeight (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (current_age : ℝ) : ℝ :=
  initial_height + growth_rate * (current_age - initial_age)

theorem tree_initial_height :
  ∀ (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (current_age : ℝ) (current_height : ℝ),
  growth_rate = 3 →
  initial_age = 1 →
  current_age = 7 →
  current_height = 23 →
  TreeHeight initial_height growth_rate initial_age current_age = current_height →
  initial_height = 5 := by
sorry

end NUMINAMATH_CALUDE_tree_initial_height_l3112_311207


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_pow6_l3112_311285

theorem nearest_integer_to_3_plus_sqrt5_pow6 :
  ∃ n : ℤ, n = 22608 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_pow6_l3112_311285


namespace NUMINAMATH_CALUDE_proportion_equality_l3112_311262

theorem proportion_equality (x y : ℝ) (h1 : 3 * x = 2 * y) (h2 : y ≠ 0) : x / 2 = y / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l3112_311262


namespace NUMINAMATH_CALUDE_unique_valid_list_l3112_311217

def isValidList (l : List Nat) : Prop :=
  l.length = 10 ∧
  (∀ n ∈ l, n % 2 = 0 ∧ n > 0) ∧
  (∀ i ∈ l.enum.tail, 
    let (idx, n) := i
    if n > 2 then 
      l[idx-1]? = some (n - 2)
    else 
      true) ∧
  (∀ i ∈ l.enum.tail,
    let (idx, n) := i
    if n % 4 = 0 then
      l[idx-1]? = some (n - 1)
    else
      true)

theorem unique_valid_list : 
  ∃! l : List Nat, isValidList l :=
sorry

end NUMINAMATH_CALUDE_unique_valid_list_l3112_311217


namespace NUMINAMATH_CALUDE_specific_ellipse_equation_l3112_311254

/-- Represents an ellipse with center at the origin and foci on the x-axis -/
structure Ellipse where
  a : ℝ  -- Semi-major axis length
  c : ℝ  -- Distance from center to focus

/-- The equation of an ellipse given its parameters -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / (e.a^2 - e.c^2) = 1

/-- Theorem: The equation of an ellipse with specific properties -/
theorem specific_ellipse_equation :
  ∀ (e : Ellipse),
    e.a = 9 →  -- Half of the major axis length (18/2)
    e.c = 3 →  -- One-third of the semi-major axis (trisecting condition)
    ∀ (x y : ℝ),
      ellipse_equation e x y ↔ x^2 / 81 + y^2 / 72 = 1 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_equation_l3112_311254


namespace NUMINAMATH_CALUDE_not_divisible_by_59_l3112_311212

theorem not_divisible_by_59 (x y : ℕ) 
  (hx : ¬ 59 ∣ x) 
  (hy : ¬ 59 ∣ y) 
  (h_sum : 59 ∣ (3 * x + 28 * y)) : 
  ¬ 59 ∣ (5 * x + 16 * y) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_59_l3112_311212


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l3112_311208

theorem inverse_proportion_order (k : ℝ) :
  let f (x : ℝ) := (k^2 + 1) / x
  let y₁ := f (-1)
  let y₂ := f 1
  let y₃ := f 2
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l3112_311208


namespace NUMINAMATH_CALUDE_legs_more_than_twice_heads_l3112_311286

-- Define the group of animals
structure AnimalGroup where
  donkeys : ℕ
  pigs : ℕ

-- Define the properties of the group
def AnimalGroup.heads (g : AnimalGroup) : ℕ := g.donkeys + g.pigs
def AnimalGroup.legs (g : AnimalGroup) : ℕ := 4 * g.donkeys + 4 * g.pigs

-- Theorem statement
theorem legs_more_than_twice_heads (g : AnimalGroup) (h : g.donkeys = 8) :
  g.legs ≥ 2 * g.heads + 16 := by
  sorry

end NUMINAMATH_CALUDE_legs_more_than_twice_heads_l3112_311286


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l3112_311222

/-- Two natural numbers are consecutive odd numbers -/
def ConsecutiveOddNumbers (p q : ℕ) : Prop :=
  ∃ k : ℕ, p = 2*k + 1 ∧ q = 2*k + 3

/-- A number a is divisible by a number b -/
def IsDivisibleBy (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = b * k

theorem consecutive_odd_numbers_divisibility (p q : ℕ) :
  ConsecutiveOddNumbers p q → IsDivisibleBy (p^q + q^p) (p + q) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_divisibility_l3112_311222


namespace NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l3112_311249

theorem square_area_ratio_when_tripled (s : ℝ) (h : s > 0) :
  (3 * s)^2 / s^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l3112_311249


namespace NUMINAMATH_CALUDE_certain_number_equation_l3112_311291

theorem certain_number_equation (x : ℝ) : 112 * x^4 = 70000 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3112_311291


namespace NUMINAMATH_CALUDE_vote_combinations_l3112_311266

/-- The number of ways to select k items from n distinct items with replacement,
    where order doesn't matter (combinations with repetition) -/
def combinations_with_repetition (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

/-- Theorem: There are 6 ways to select 2 items from 3 items with replacement,
    where order doesn't matter -/
theorem vote_combinations : combinations_with_repetition 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_vote_combinations_l3112_311266


namespace NUMINAMATH_CALUDE_circle_radius_relation_l3112_311247

theorem circle_radius_relation (square_area : ℝ) (small_circle_circumference : ℝ) :
  square_area = 784 →
  small_circle_circumference = 8 →
  ∃ (x : ℝ) (r_s r_l : ℝ),
    r_s = 4 / π ∧
    r_l = 14 ∧
    r_l = x - (1/3) * r_s ∧
    x = 14 + 4 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_relation_l3112_311247


namespace NUMINAMATH_CALUDE_midpoint_polar_specific_points_l3112_311257

/-- The midpoint of a line segment in polar coordinates --/
def midpoint_polar (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem midpoint_polar_specific_points :
  let A : ℝ × ℝ := (9, π/3)
  let B : ℝ × ℝ := (9, 2*π/3)
  let M := midpoint_polar A.1 A.2 B.1 B.2
  (0 < A.1 ∧ 0 ≤ A.2 ∧ A.2 < 2*π) ∧
  (0 < B.1 ∧ 0 ≤ B.2 ∧ B.2 < 2*π) →
  M = (9 * Real.sqrt 3 / 2, π/2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_polar_specific_points_l3112_311257


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3112_311294

theorem solve_exponential_equation :
  ∃ n : ℕ, 4^n * 4^n * 4^n = 16^3 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3112_311294


namespace NUMINAMATH_CALUDE_complement_of_M_l3112_311244

def M : Set ℝ := {x | (1 + x) / (1 - x) > 0}

theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = {x | x ≤ -1 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3112_311244


namespace NUMINAMATH_CALUDE_tangent_point_min_value_on_interval_l3112_311219

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem tangent_point (a : ℝ) :
  (∃ x > 0, f a x = 0 ∧ (deriv (f a)) x = 0) → a = 1 / Real.exp 1 :=
sorry

theorem min_value_on_interval (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ (if a < Real.log 2 then -a else Real.log 2 - 2 * a)) ∧
  (∃ x ∈ Set.Icc 1 2, f a x = (if a < Real.log 2 then -a else Real.log 2 - 2 * a)) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_min_value_on_interval_l3112_311219


namespace NUMINAMATH_CALUDE_star_property_l3112_311263

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation *
def star : Element → Element → Element
  | Element.one, Element.one => Element.one
  | Element.one, Element.two => Element.two
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.four
  | Element.two, Element.one => Element.two
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.one
  | Element.two, Element.four => Element.three
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.two
  | Element.four, Element.one => Element.four
  | Element.four, Element.two => Element.three
  | Element.four, Element.three => Element.two
  | Element.four, Element.four => Element.one

theorem star_property : 
  star (star Element.two Element.four) (star Element.one Element.three) = Element.four := by
  sorry

end NUMINAMATH_CALUDE_star_property_l3112_311263


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l3112_311268

/-- Represents the age ratio of a man to his son after two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  (son_age + age_difference + 2) / (son_age + 2)

theorem man_son_age_ratio :
  let son_age : ℕ := 20
  let age_difference : ℕ := 22
  age_ratio son_age age_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l3112_311268


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l3112_311280

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1 / 500) →
  (m : ℝ)^(1/3) = n + r →
  (∀ k < n, ¬∃ s, (0 < s) ∧ (s < 1 / 500) ∧ (∃ l : ℕ, (l : ℝ)^(1/3) = k + s)) →
  n = 13 := by
  sorry

#check smallest_cube_root_with_small_fraction

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l3112_311280


namespace NUMINAMATH_CALUDE_ratio_equivalence_l3112_311204

theorem ratio_equivalence (x : ℚ) : 
  (12 : ℚ) / 8 = 6 / (x * 60) → x = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l3112_311204


namespace NUMINAMATH_CALUDE_sandwich_cost_l3112_311229

/-- The cost of a sandwich given the total cost of sandwiches and sodas -/
theorem sandwich_cost (total_cost : ℚ) (num_sandwiches : ℕ) (num_sodas : ℕ) (soda_cost : ℚ) : 
  total_cost = 8.36 ∧ 
  num_sandwiches = 2 ∧ 
  num_sodas = 4 ∧ 
  soda_cost = 0.87 → 
  (total_cost - num_sodas * soda_cost) / num_sandwiches = 2.44 := by
sorry

end NUMINAMATH_CALUDE_sandwich_cost_l3112_311229


namespace NUMINAMATH_CALUDE_periodic_function_property_l3112_311243

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, β are non-zero real numbers,
    if f(2011) = 5, then f(2012) = 3 -/
theorem periodic_function_property (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  (f 2011 = 5) → (f 2012 = 3) := by sorry

end NUMINAMATH_CALUDE_periodic_function_property_l3112_311243


namespace NUMINAMATH_CALUDE_sin_cos_difference_65_35_l3112_311237

theorem sin_cos_difference_65_35 :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) - 
  Real.cos (65 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_65_35_l3112_311237


namespace NUMINAMATH_CALUDE_single_elimination_tournament_games_l3112_311203

/-- Calculates the number of games in a single-elimination tournament -/
def tournament_games (n : ℕ) : ℕ :=
  n - 1

/-- The number of teams in the tournament -/
def num_teams : ℕ := 24

theorem single_elimination_tournament_games :
  tournament_games num_teams = 23 := by
  sorry

end NUMINAMATH_CALUDE_single_elimination_tournament_games_l3112_311203


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3112_311200

def license_plate_combinations : ℕ :=
  let alphabet_size : ℕ := 26
  let digit_size : ℕ := 10
  let letter_positions : ℕ := 4
  let digit_positions : ℕ := 2

  let choose_repeated_letter := alphabet_size
  let choose_distinct_letters := Nat.choose (alphabet_size - 1) 2
  let place_repeated_letter := Nat.choose letter_positions 2
  let arrange_nonrepeated_letters := 2

  let letter_combinations := 
    choose_repeated_letter * choose_distinct_letters * place_repeated_letter * arrange_nonrepeated_letters

  let digit_combinations := digit_size ^ digit_positions

  letter_combinations * digit_combinations

theorem license_plate_theorem : license_plate_combinations = 936000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3112_311200


namespace NUMINAMATH_CALUDE_grape_juice_amount_l3112_311298

/-- Represents a fruit drink composition -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem: The amount of grape juice in the drink is 105 ounces -/
theorem grape_juice_amount (drink : FruitDrink) 
  (h1 : drink.total = 300)
  (h2 : drink.orange_percent = 0.25)
  (h3 : drink.watermelon_percent = 0.40)
  (h4 : drink.grape_ounces = drink.total - (drink.orange_percent * drink.total + drink.watermelon_percent * drink.total)) :
  drink.grape_ounces = 105 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_amount_l3112_311298


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3112_311248

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 2 * x) = 8 → x = -30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3112_311248


namespace NUMINAMATH_CALUDE_unique_modulus_for_xy_plus_one_implies_x_plus_y_l3112_311287

theorem unique_modulus_for_xy_plus_one_implies_x_plus_y (n : ℕ+) : 
  (∀ x y : ℤ, (x * y + 1) % n = 0 → (x + y) % n = 0) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_modulus_for_xy_plus_one_implies_x_plus_y_l3112_311287


namespace NUMINAMATH_CALUDE_cross_in_square_l3112_311251

theorem cross_in_square (a : ℝ) (h : a > 0) : 
  (2 * (a/2)^2 + 2 * (a/4)^2 = 810) → a = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l3112_311251


namespace NUMINAMATH_CALUDE_family_size_theorem_l3112_311213

def family_size_problem (fathers_side : ℕ) (total : ℕ) : Prop :=
  let mothers_side := total - fathers_side
  let difference := mothers_side - fathers_side
  let percentage := (difference : ℚ) / fathers_side * 100
  fathers_side = 10 ∧ total = 23 → percentage = 30

theorem family_size_theorem :
  family_size_problem 10 23 := by
  sorry

end NUMINAMATH_CALUDE_family_size_theorem_l3112_311213


namespace NUMINAMATH_CALUDE_bakers_cakes_l3112_311211

theorem bakers_cakes (initial_cakes : ℕ) : 
  initial_cakes - 105 + 170 = 186 → initial_cakes = 121 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l3112_311211


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_product_l3112_311290

theorem two_numbers_sum_and_product : 
  ∃ (x y : ℝ), x + y = 10 ∧ x * y = 24 ∧ ((x = 4 ∧ y = 6) ∨ (x = 6 ∧ y = 4)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_product_l3112_311290


namespace NUMINAMATH_CALUDE_factory_output_increase_l3112_311239

theorem factory_output_increase (P : ℝ) : 
  (1 + P / 100) * (1 + 20 / 100) * (1 - 24.242424242424242 / 100) = 1 → P = 10 := by
sorry

end NUMINAMATH_CALUDE_factory_output_increase_l3112_311239


namespace NUMINAMATH_CALUDE_triangle_side_length_l3112_311228

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a = 8 → b = 7 → B = Real.pi / 3 → (c = 3 ∨ c = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3112_311228


namespace NUMINAMATH_CALUDE_wallet_total_l3112_311221

theorem wallet_total (nada ali john : ℕ) : 
  ali = nada - 5 →
  john = 4 * nada →
  john = 48 →
  ali + nada + john = 67 := by
sorry

end NUMINAMATH_CALUDE_wallet_total_l3112_311221


namespace NUMINAMATH_CALUDE_solve_sleep_problem_l3112_311220

def sleep_problem (connor_sleep : ℕ) : Prop :=
  let luke_sleep := connor_sleep + 2
  let emma_sleep := connor_sleep - 1
  let puppy_sleep := luke_sleep * 2
  connor_sleep = 6 →
  connor_sleep + luke_sleep + emma_sleep + puppy_sleep = 35

theorem solve_sleep_problem :
  sleep_problem 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_sleep_problem_l3112_311220


namespace NUMINAMATH_CALUDE_austin_robot_purchase_l3112_311299

theorem austin_robot_purchase (num_robots : ℕ) (robot_cost tax change : ℚ) : 
  num_robots = 7 → 
  robot_cost = 8.75 → 
  tax = 7.22 → 
  change = 11.53 → 
  (num_robots : ℚ) * robot_cost + tax + change = 80 :=
by sorry

end NUMINAMATH_CALUDE_austin_robot_purchase_l3112_311299


namespace NUMINAMATH_CALUDE_bridge_length_l3112_311227

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3112_311227


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3112_311295

theorem sqrt_equation_solution : 
  ∀ x : ℝ, (Real.sqrt (5 * x - 4) + 12 / Real.sqrt (5 * x - 4) = 9) ↔ (x = 13/5 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3112_311295


namespace NUMINAMATH_CALUDE_championship_outcomes_l3112_311256

theorem championship_outcomes (n : ℕ) (m : ℕ) : 
  n = 5 → m = 3 → n ^ m = 125 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_l3112_311256


namespace NUMINAMATH_CALUDE_all_four_digit_numbers_generated_l3112_311250

/-- Represents the operations that can be performed on a number -/
inductive Operation
  | mul2sub2 : Operation
  | mul3add4 : Operation
  | add7 : Operation

/-- Applies an operation to a number -/
def applyOperation (op : Operation) (x : ℕ) : ℕ :=
  match op with
  | Operation.mul2sub2 => 2 * x - 2
  | Operation.mul3add4 => 3 * x + 4
  | Operation.add7 => x + 7

/-- Returns true if the number is four digits -/
def isFourDigits (n : ℕ) : Bool :=
  1000 ≤ n ∧ n ≤ 9999

/-- The set of all four-digit numbers -/
def fourDigitNumbers : Set ℕ :=
  {n : ℕ | isFourDigits n}

/-- The set of numbers that can be generated from 1 using the given operations -/
def generatedNumbers : Set ℕ :=
  {n : ℕ | ∃ (ops : List Operation), n = ops.foldl (fun acc op => applyOperation op acc) 1}

/-- Theorem stating that all four-digit numbers can be generated -/
theorem all_four_digit_numbers_generated :
  fourDigitNumbers ⊆ generatedNumbers :=
sorry

end NUMINAMATH_CALUDE_all_four_digit_numbers_generated_l3112_311250


namespace NUMINAMATH_CALUDE_a_and_b_know_own_results_a_and_b_dont_know_each_others_results_l3112_311264

-- Define the possible results
inductive Result
| Excellent
| Good

-- Define the students
inductive Student
| A
| B
| C
| D

-- Define the function that assigns results to students
def result : Student → Result := sorry

-- Define the knowledge state of each student
structure Knowledge where
  knows_b : Bool
  knows_c : Bool
  knows_d : Bool

-- Define the initial knowledge state
def initial_knowledge : Student → Knowledge
| Student.A => { knows_b := false, knows_c := false, knows_d := true }
| Student.B => { knows_b := false, knows_c := true,  knows_d := false }
| Student.C => { knows_b := false, knows_c := false, knows_d := false }
| Student.D => { knows_b := true,  knows_c := true,  knows_d := false }

-- Theorem stating that A and B can know their own results
theorem a_and_b_know_own_results :
  (∃ (s₁ s₂ : Student), result s₁ = Result.Excellent ∧ result s₂ = Result.Excellent) ∧
  (∃ (s₃ s₄ : Student), result s₃ = Result.Good ∧ result s₄ = Result.Good) ∧
  (result Student.B ≠ result Student.C) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) →
  (∃ (f : Student → Result),
    (f Student.A = result Student.A) ∧
    (f Student.B = result Student.B) ∧
    (f Student.C ≠ result Student.C ∨ f Student.D ≠ result Student.D)) :=
sorry

-- Theorem stating that A and B cannot know each other's results
theorem a_and_b_dont_know_each_others_results :
  (∃ (s₁ s₂ : Student), result s₁ = Result.Excellent ∧ result s₂ = Result.Excellent) ∧
  (∃ (s₃ s₄ : Student), result s₃ = Result.Good ∧ result s₄ = Result.Good) ∧
  (result Student.B ≠ result Student.C) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) →
  ¬(∃ (f : Student → Result),
    (f Student.A = result Student.A) ∧
    (f Student.B = result Student.B) ∧
    (f Student.C = result Student.C) ∧
    (f Student.D = result Student.D)) :=
sorry

end NUMINAMATH_CALUDE_a_and_b_know_own_results_a_and_b_dont_know_each_others_results_l3112_311264


namespace NUMINAMATH_CALUDE_johnny_guitar_practice_l3112_311252

/-- Represents the number of days Johnny has been practicing guitar -/
def current_practice : ℕ := 40

/-- Represents the daily practice amount -/
def daily_practice : ℕ := 2

theorem johnny_guitar_practice :
  let days_to_triple := (3 * current_practice - current_practice) / daily_practice
  (2 * (current_practice - 20 * daily_practice) = current_practice) →
  days_to_triple = 80 := by
  sorry

end NUMINAMATH_CALUDE_johnny_guitar_practice_l3112_311252


namespace NUMINAMATH_CALUDE_mean_proportional_81_100_l3112_311246

theorem mean_proportional_81_100 : 
  Real.sqrt (81 * 100) = 90 := by sorry

end NUMINAMATH_CALUDE_mean_proportional_81_100_l3112_311246


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l3112_311209

/-- Given three square regions A, B, and C, where the perimeter of A is 16 units,
    the perimeter of B is 32 units, and the side length of each subsequent region doubles,
    prove that the ratio of the area of region B to the area of region C is 1/4. -/
theorem area_ratio_of_squares (side_A side_B side_C : ℝ) : 
  side_A * 4 = 16 →
  side_B * 4 = 32 →
  side_C = 2 * side_B →
  (side_B ^ 2) / (side_C ^ 2) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l3112_311209
