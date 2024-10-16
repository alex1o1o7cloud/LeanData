import Mathlib

namespace NUMINAMATH_CALUDE_potato_cost_theorem_l2025_202506

-- Define the given conditions
def people_count : ℕ := 40
def potatoes_per_person : ℚ := 3/2
def bag_weight : ℕ := 20
def bag_cost : ℕ := 5

-- Define the theorem
theorem potato_cost_theorem : 
  (people_count : ℚ) * potatoes_per_person / bag_weight * bag_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_potato_cost_theorem_l2025_202506


namespace NUMINAMATH_CALUDE_lassis_production_l2025_202558

/-- Given a ratio of lassis to fruit units, calculate the number of lassis that can be made from a given number of fruit units -/
def calculate_lassis (ratio_lassis ratio_fruits fruits : ℕ) : ℕ :=
  (ratio_lassis * fruits) / ratio_fruits

/-- Proof that 25 fruit units produce 75 lassis given the initial ratio -/
theorem lassis_production : calculate_lassis 15 5 25 = 75 := by
  sorry

end NUMINAMATH_CALUDE_lassis_production_l2025_202558


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2025_202503

/-- The cost of a single candy bar given Carl's earnings and purchasing power -/
theorem candy_bar_cost (weekly_earnings : ℚ) (weeks : ℕ) (bars_bought : ℕ) : 
  weekly_earnings = 3/4 ∧ weeks = 4 ∧ bars_bought = 6 → 
  (weekly_earnings * weeks) / bars_bought = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2025_202503


namespace NUMINAMATH_CALUDE_investment_value_l2025_202537

theorem investment_value (x : ℝ) : 
  x > 0 ∧ 
  0.07 * x + 0.15 * 1500 = 0.13 * (x + 1500) →
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_investment_value_l2025_202537


namespace NUMINAMATH_CALUDE_major_axis_length_is_eight_l2025_202583

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- The length of the major axis of an ellipse with given properties -/
def major_axis_length (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating that the length of the major axis is 8 for the given ellipse -/
theorem major_axis_length_is_eight :
  let e : Ellipse := {
    tangent_to_axes := true,
    focus1 := (5, -4 + 2 * Real.sqrt 3),
    focus2 := (5, -4 - 2 * Real.sqrt 3)
  }
  major_axis_length e = 8 :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_is_eight_l2025_202583


namespace NUMINAMATH_CALUDE_sector_angle_l2025_202560

theorem sector_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  r = 1 →
  l = 2 →
  l = α * r →
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_angle_l2025_202560


namespace NUMINAMATH_CALUDE_max_distance_ratio_l2025_202511

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the parabola
def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 2 * P.1

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) (m : ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = m^2 * ((P.1 - 1)^2 + P.2^2)

-- Theorem statement
theorem max_distance_ratio :
  ∃ (m_max : ℝ), m_max = Real.sqrt 3 ∧
  (∀ (P : ℝ × ℝ) (m : ℝ), on_parabola P → distance_ratio P m → m ≤ m_max) ∧
  (∃ (P : ℝ × ℝ), on_parabola P ∧ distance_ratio P m_max) :=
sorry

end NUMINAMATH_CALUDE_max_distance_ratio_l2025_202511


namespace NUMINAMATH_CALUDE_max_area_rectangle_max_area_achieved_l2025_202555

/-- The maximum area of a rectangle with integer side lengths and perimeter 160 -/
theorem max_area_rectangle (x y : ℕ) (h : x + y = 80) : x * y ≤ 1600 :=
sorry

/-- The maximum area is achieved when both sides are 40 -/
theorem max_area_achieved (x y : ℕ) (h : x + y = 80) : x * y = 1600 ↔ x = 40 ∧ y = 40 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_max_area_achieved_l2025_202555


namespace NUMINAMATH_CALUDE_students_in_band_or_sports_l2025_202520

theorem students_in_band_or_sports 
  (total : ℕ) 
  (band : ℕ) 
  (sports : ℕ) 
  (both : ℕ) 
  (h_total : total = 320)
  (h_band : band = 85)
  (h_sports : sports = 200)
  (h_both : both = 60) :
  band + sports - both = 225 := by
  sorry

end NUMINAMATH_CALUDE_students_in_band_or_sports_l2025_202520


namespace NUMINAMATH_CALUDE_greatest_among_five_l2025_202534

theorem greatest_among_five : ∀ (a b c d e : ℕ), 
  a = 5 → b = 8 → c = 4 → d = 3 → e = 2 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) := by
  sorry

end NUMINAMATH_CALUDE_greatest_among_five_l2025_202534


namespace NUMINAMATH_CALUDE_M_intersect_N_l2025_202572

def M : Set ℤ := {-1, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = x^2}

theorem M_intersect_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2025_202572


namespace NUMINAMATH_CALUDE_no_bounded_integral_exists_l2025_202524

/-- Base 2 representation of x in [0, 1) -/
def base2Rep (x : ℝ) : ℕ → Fin 2 :=
  sorry

/-- Function f_n as defined in the problem -/
def f_n (n : ℕ) (x : ℝ) : ℤ :=
  sorry

/-- The main theorem -/
theorem no_bounded_integral_exists :
  ∀ (φ : ℝ → ℝ),
    (∀ y, 0 ≤ φ y) →
    (∀ M, ∃ N, ∀ x, N ≤ x → M ≤ φ x) →
    (∀ B, ∃ n : ℕ, B < ∫ x in (0 : ℝ)..1, φ (|f_n n x|)) :=
  sorry

end NUMINAMATH_CALUDE_no_bounded_integral_exists_l2025_202524


namespace NUMINAMATH_CALUDE_length_MN_is_eleven_thirds_l2025_202566

/-- Triangle ABC with given side lengths and points M and N -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Point M on AB such that CM is the angle bisector of ∠ACB
  M : ℝ
  -- Point N on AB such that CN is the altitude to AB
  N : ℝ
  -- Conditions
  h_AB : AB = 50
  h_BC : BC = 20
  h_AC : AC = 40
  h_M_angle_bisector : M = AB / 3
  h_N_altitude : N = BC * (AB^2 + BC^2 - AC^2) / (2 * AB * BC)

/-- The length of MN in the given triangle -/
def length_MN (t : TriangleABC) : ℝ := t.M - t.N

/-- Theorem stating that the length of MN is 11/3 -/
theorem length_MN_is_eleven_thirds (t : TriangleABC) :
  length_MN t = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_length_MN_is_eleven_thirds_l2025_202566


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l2025_202599

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percentage = 15 / 100)
  (h4 : good_fruits_percentage = 878 / 1000)
  : (total_bananas - (total_oranges + total_bananas - 
     (good_fruits_percentage * (total_oranges + total_bananas)).floor - 
     (rotten_oranges_percentage * total_oranges).floor)) / total_bananas = 8 / 100 := by
  sorry


end NUMINAMATH_CALUDE_rotten_bananas_percentage_l2025_202599


namespace NUMINAMATH_CALUDE_simplify_fourth_roots_l2025_202580

theorem simplify_fourth_roots : Real.sqrt (Real.sqrt 81) - Real.sqrt (Real.sqrt 256) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fourth_roots_l2025_202580


namespace NUMINAMATH_CALUDE_rectangle_area_l2025_202527

/-- The area of a rectangle with vertices at (-7, 1), (1, 1), (1, -6), and (-7, -6) in a rectangular coordinate system is 56 square units. -/
theorem rectangle_area : ℝ := by
  -- Define the vertices of the rectangle
  let v1 : ℝ × ℝ := (-7, 1)
  let v2 : ℝ × ℝ := (1, 1)
  let v3 : ℝ × ℝ := (1, -6)
  let v4 : ℝ × ℝ := (-7, -6)

  -- Calculate the length and width of the rectangle
  let length : ℝ := v2.1 - v1.1
  let width : ℝ := v1.2 - v4.2

  -- Calculate the area of the rectangle
  let area : ℝ := length * width

  -- Prove that the area is equal to 56
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2025_202527


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2025_202522

theorem trigonometric_problem (α : ℝ) 
  (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1/4) :
  (Real.tan α = -2) ∧ 
  ((Real.sin (2*α) + 1) / (1 + Real.sin (2*α) + Real.cos (2*α)) = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2025_202522


namespace NUMINAMATH_CALUDE_marble_sharing_l2025_202589

theorem marble_sharing (initial_marbles : ℝ) (initial_marbles_pos : initial_marbles > 0) :
  let remaining_after_lara := initial_marbles * (1 - 0.3)
  let remaining_after_max := remaining_after_lara * (1 - 0.15)
  let remaining_after_ben := remaining_after_max * (1 - 0.2)
  remaining_after_ben / initial_marbles = 0.476 := by
sorry

end NUMINAMATH_CALUDE_marble_sharing_l2025_202589


namespace NUMINAMATH_CALUDE_real_part_of_z_l2025_202535

theorem real_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2) : 
  Complex.re z = 1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2025_202535


namespace NUMINAMATH_CALUDE_square_perimeter_l2025_202568

theorem square_perimeter (s : ℝ) (h1 : s > 0) : 
  (2 * s + 2 * (s / 5) = 36) → (4 * s = 60) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2025_202568


namespace NUMINAMATH_CALUDE_cookie_eating_contest_l2025_202505

theorem cookie_eating_contest (first_friend second_friend : ℚ) 
  (h1 : first_friend = 5/6)
  (h2 : second_friend = 2/3) :
  first_friend - second_friend = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_eating_contest_l2025_202505


namespace NUMINAMATH_CALUDE_jerry_shower_limit_l2025_202591

/-- Calculates the number of full showers Jerry can take in July --/
def showers_in_july (total_water : ℕ) (drinking_cooking : ℕ) (shower_water : ℕ)
  (pool_length : ℕ) (pool_width : ℕ) (pool_depth : ℕ)
  (odd_day_leakage : ℕ) (even_day_leakage : ℕ) (evaporation_rate : ℕ)
  (odd_days : ℕ) (even_days : ℕ) : ℕ :=
  let pool_volume := pool_length * pool_width * pool_depth
  let total_leakage := odd_day_leakage * odd_days + even_day_leakage * even_days
  let total_evaporation := evaporation_rate * (odd_days + even_days)
  let pool_water_usage := pool_volume + total_leakage + total_evaporation
  let remaining_water := total_water - drinking_cooking - pool_water_usage
  remaining_water / shower_water

/-- Theorem stating that Jerry can take at most 1 full shower in July --/
theorem jerry_shower_limit :
  showers_in_july 1000 100 20 10 10 6 5 8 2 16 15 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_jerry_shower_limit_l2025_202591


namespace NUMINAMATH_CALUDE_subset_union_equality_l2025_202502

theorem subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
  (h_nonempty : ∀ i, (A i).Nonempty) : 
  ∃ (I J : Set (Fin (n + 1))), I.Nonempty ∧ J.Nonempty ∧ I ∩ J = ∅ ∧ 
  (⋃ (i ∈ I), A i) = (⋃ (j ∈ J), A j) := by
sorry

end NUMINAMATH_CALUDE_subset_union_equality_l2025_202502


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l2025_202562

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line ax + 2y = 0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := 2, c := 0 }

/-- The second line x + y = 1 -/
def line2 : Line :=
  { a := 1, b := 1, c := -1 }

/-- Theorem: a = 2 is necessary and sufficient for the lines to be parallel -/
theorem parallel_iff_a_eq_two :
  ∀ a : ℝ, parallel (line1 a) line2 ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l2025_202562


namespace NUMINAMATH_CALUDE_original_typing_speed_l2025_202597

theorem original_typing_speed 
  (original_speed : ℕ) 
  (speed_decrease : ℕ) 
  (words_typed : ℕ) 
  (time_taken : ℕ) :
  speed_decrease = 40 →
  words_typed = 3440 →
  time_taken = 20 →
  (original_speed - speed_decrease) * time_taken = words_typed →
  original_speed = 212 := by
sorry

end NUMINAMATH_CALUDE_original_typing_speed_l2025_202597


namespace NUMINAMATH_CALUDE_nine_sevenths_to_fourth_l2025_202510

theorem nine_sevenths_to_fourth (x : ℚ) : x = 9 * (1 / 7)^4 → x = 9 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_nine_sevenths_to_fourth_l2025_202510


namespace NUMINAMATH_CALUDE_max_value_when_a_is_one_a_values_when_max_is_two_l2025_202594

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- Part 1
theorem max_value_when_a_is_one :
  ∃ (max : ℝ), (∀ x, f 1 x ≤ max) ∧ (∃ x, f 1 x = max) ∧ max = 1 := by sorry

-- Part 2
theorem a_values_when_max_is_two :
  (∃ (max : ℝ), (∀ x ∈ Set.Icc 0 1, f a x ≤ max) ∧ 
   (∃ x ∈ Set.Icc 0 1, f a x = max) ∧ max = 2) → (a = -1 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_max_value_when_a_is_one_a_values_when_max_is_two_l2025_202594


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2025_202584

-- Define a sequence of real numbers
def Sequence := ℕ → ℝ

-- Define what it means for a sequence to be geometric
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the condition given in the problem
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2

-- Theorem statement
theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2025_202584


namespace NUMINAMATH_CALUDE_jenna_reading_goal_l2025_202542

/-- Calculates the number of pages Jenna needs to read per day to meet her reading goal --/
theorem jenna_reading_goal (total_pages : ℕ) (total_days : ℕ) (busy_days : ℕ) (special_day_pages : ℕ) :
  total_pages = 600 →
  total_days = 30 →
  busy_days = 4 →
  special_day_pages = 100 →
  (total_pages - special_day_pages) / (total_days - busy_days - 1) = 20 := by
  sorry

#check jenna_reading_goal

end NUMINAMATH_CALUDE_jenna_reading_goal_l2025_202542


namespace NUMINAMATH_CALUDE_inequality_proof_l2025_202561

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2025_202561


namespace NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l2025_202550

theorem consecutive_integer_product_divisibility (k : ℤ) : 
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 5 * m) →
  (∃ m : ℤ, n = 10 * m) ∧
  (∃ m : ℤ, n = 15 * m) ∧
  (∃ m : ℤ, n = 30 * m) ∧
  (∃ m : ℤ, n = 60 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 20 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_product_divisibility_l2025_202550


namespace NUMINAMATH_CALUDE_rearrange_segments_l2025_202557

theorem rearrange_segments (a b : ℕ) : 
  ∃ (f g : Fin 1961 → Fin 1961), 
    ∀ i : Fin 1961, ∃ k : ℕ, 
      (a + f i) + (b + g i) = k + i.val ∧ 
      k + 1960 = (a + f ⟨1960, by norm_num⟩) + (b + g ⟨1960, by norm_num⟩) := by
  sorry

end NUMINAMATH_CALUDE_rearrange_segments_l2025_202557


namespace NUMINAMATH_CALUDE_multiplication_mistake_l2025_202590

theorem multiplication_mistake (number : ℕ) (correct_multiplier : ℕ) (mistaken_multiplier : ℕ) :
  number = 138 →
  correct_multiplier = 43 →
  mistaken_multiplier = 34 →
  (number * correct_multiplier) - (number * mistaken_multiplier) = 1242 := by
sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l2025_202590


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2025_202548

-- Define the square PQRS
def square_side : ℝ := 7

-- Define the shaded areas
def shaded_area_1 : ℝ := 2^2
def shaded_area_2 : ℝ := 5^2 - 3^2
def shaded_area_3 : ℝ := square_side^2 - 6^2

-- Total shaded area
def total_shaded_area : ℝ := shaded_area_1 + shaded_area_2 + shaded_area_3

-- Total area of square PQRS
def total_area : ℝ := square_side^2

-- Theorem statement
theorem shaded_area_percentage :
  total_shaded_area = 33 ∧ (total_shaded_area / total_area) = 33 / 49 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2025_202548


namespace NUMINAMATH_CALUDE_chuck_puppy_shot_cost_l2025_202598

/-- The total cost of shots for puppies --/
def total_shot_cost (num_dogs : ℕ) (puppies_per_dog : ℕ) (shots_per_puppy : ℕ) (cost_per_shot : ℕ) : ℕ :=
  num_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot

/-- Theorem stating the total cost of shots for Chuck's puppies --/
theorem chuck_puppy_shot_cost :
  total_shot_cost 3 4 2 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chuck_puppy_shot_cost_l2025_202598


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2025_202579

/-- Given a line L1 with equation 2x - 3y + 4 = 0, prove that the line L2 with equation 3x + 2y - 1 = 0
    passes through the point (-1, 2) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2 * x - 3 * y + 4 = 0
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 3 * x + 2 * y - 1 = 0
  let point : ℝ × ℝ := (-1, 2)
  (L2 point.1 point.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → L2 x1 y1 → L2 x2 y2 → 
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 → 
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) =
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2025_202579


namespace NUMINAMATH_CALUDE_range_of_a_l2025_202586

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3*a else a^x

theorem range_of_a (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x y : ℝ, x < y → f a x > f a y) →
  (1/3 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2025_202586


namespace NUMINAMATH_CALUDE_general_equation_l2025_202551

theorem general_equation (n : ℝ) : n ≠ 4 ∧ 8 - n ≠ 4 → 
  (n / (n - 4)) + ((8 - n) / ((8 - n) - 4)) = 2 := by sorry

end NUMINAMATH_CALUDE_general_equation_l2025_202551


namespace NUMINAMATH_CALUDE_set_equality_l2025_202519

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {1, 3, 6}

theorem set_equality : (U \ M) ∩ (U \ N) = {2, 7} := by sorry

end NUMINAMATH_CALUDE_set_equality_l2025_202519


namespace NUMINAMATH_CALUDE_john_yasmin_children_ratio_l2025_202545

/-- The number of children John has -/
def john_children : ℕ := sorry

/-- The number of children Yasmin has -/
def yasmin_children : ℕ := 2

/-- The total number of grandchildren Gabriel has -/
def gabriel_grandchildren : ℕ := 6

/-- The ratio of John's children to Yasmin's children -/
def children_ratio : ℚ := john_children / yasmin_children

theorem john_yasmin_children_ratio :
  (john_children + yasmin_children = gabriel_grandchildren) →
  children_ratio = 2 := by
sorry

end NUMINAMATH_CALUDE_john_yasmin_children_ratio_l2025_202545


namespace NUMINAMATH_CALUDE_between_a_and_b_l2025_202569

theorem between_a_and_b (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a < b) :
  a < (|3*a + 2*b| / 5) ∧ (|3*a + 2*b| / 5) < b := by
  sorry

end NUMINAMATH_CALUDE_between_a_and_b_l2025_202569


namespace NUMINAMATH_CALUDE_swim_club_percentage_passed_l2025_202556

/-- The percentage of swim club members who have passed the lifesaving test -/
def percentage_passed (total_members : ℕ) (not_passed_with_course : ℕ) (not_passed_without_course : ℕ) : ℚ :=
  1 - (not_passed_with_course + not_passed_without_course : ℚ) / total_members

theorem swim_club_percentage_passed :
  percentage_passed 100 40 30 = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_swim_club_percentage_passed_l2025_202556


namespace NUMINAMATH_CALUDE_blake_change_l2025_202587

/-- The amount Blake spent on oranges -/
def orange_cost : ℕ := 40

/-- The amount Blake spent on apples -/
def apple_cost : ℕ := 50

/-- The amount Blake spent on mangoes -/
def mango_cost : ℕ := 60

/-- The initial amount Blake had -/
def initial_amount : ℕ := 300

/-- The change Blake received after shopping -/
def change : ℕ := initial_amount - (orange_cost + apple_cost + mango_cost)

theorem blake_change : change = 150 := by sorry

end NUMINAMATH_CALUDE_blake_change_l2025_202587


namespace NUMINAMATH_CALUDE_overhead_cost_calculation_l2025_202559

/-- The overhead cost for Steve's circus production -/
def overhead_cost : ℕ := sorry

/-- The production cost per performance -/
def production_cost_per_performance : ℕ := 7000

/-- The revenue from a sold-out performance -/
def revenue_per_performance : ℕ := 16000

/-- The number of sold-out performances needed to break even -/
def break_even_performances : ℕ := 9

/-- Theorem stating that the overhead cost is $81,000 -/
theorem overhead_cost_calculation :
  overhead_cost = 81000 :=
by
  sorry

end NUMINAMATH_CALUDE_overhead_cost_calculation_l2025_202559


namespace NUMINAMATH_CALUDE_tyrone_total_money_l2025_202536

-- Define the currency values
def one_dollar : ℚ := 1
def ten_dollar : ℚ := 10
def five_dollar : ℚ := 5
def quarter : ℚ := 0.25
def half_dollar : ℚ := 0.5
def dime : ℚ := 0.1
def nickel : ℚ := 0.05
def penny : ℚ := 0.01
def two_dollar : ℚ := 2
def fifty_cent : ℚ := 0.5

-- Define Tyrone's currency counts
def one_dollar_bills : ℕ := 3
def ten_dollar_bills : ℕ := 1
def five_dollar_bills : ℕ := 2
def quarters : ℕ := 26
def half_dollar_coins : ℕ := 5
def dimes : ℕ := 45
def nickels : ℕ := 8
def one_dollar_coins : ℕ := 3
def pennies : ℕ := 56
def two_dollar_bills : ℕ := 2
def fifty_cent_coins : ℕ := 4

-- Define the total amount function
def total_amount : ℚ :=
  (one_dollar_bills : ℚ) * one_dollar +
  (ten_dollar_bills : ℚ) * ten_dollar +
  (five_dollar_bills : ℚ) * five_dollar +
  (quarters : ℚ) * quarter +
  (half_dollar_coins : ℚ) * half_dollar +
  (dimes : ℚ) * dime +
  (nickels : ℚ) * nickel +
  (one_dollar_coins : ℚ) * one_dollar +
  (pennies : ℚ) * penny +
  (two_dollar_bills : ℚ) * two_dollar +
  (fifty_cent_coins : ℚ) * fifty_cent

-- Theorem stating that the total amount is $46.46
theorem tyrone_total_money : total_amount = 46.46 := by
  sorry

end NUMINAMATH_CALUDE_tyrone_total_money_l2025_202536


namespace NUMINAMATH_CALUDE_investment_ratio_l2025_202526

theorem investment_ratio (a b c : ℝ) (total_profit b_share : ℝ) :
  b = (2/3) * c →
  a = n * b →
  total_profit = 3300 →
  b_share = 600 →
  b_share / total_profit = b / (a + b + c) →
  a / b = 3 :=
sorry

end NUMINAMATH_CALUDE_investment_ratio_l2025_202526


namespace NUMINAMATH_CALUDE_max_cookies_juan_l2025_202588

/-- Represents the ingredients required for baking cookies -/
structure Ingredients where
  milk : ℚ
  sugar : ℚ
  flour : ℚ

/-- Represents the storage capacity for ingredients -/
structure StorageCapacity where
  milk : ℚ
  sugar : ℚ
  flour : ℚ

/-- Calculate the maximum number of cookies that can be baked given the ingredients per cookie and storage capacity -/
def max_cookies (ingredients_per_cookie : Ingredients) (storage : StorageCapacity) : ℚ :=
  min (storage.milk / ingredients_per_cookie.milk)
      (min (storage.sugar / ingredients_per_cookie.sugar)
           (storage.flour / ingredients_per_cookie.flour))

/-- Theorem: The maximum number of cookies Juan can bake within storage constraints is 320 -/
theorem max_cookies_juan :
  let ingredients_per_40_cookies : Ingredients := { milk := 10, sugar := 5, flour := 15 }
  let ingredients_per_cookie : Ingredients := {
    milk := ingredients_per_40_cookies.milk / 40,
    sugar := ingredients_per_40_cookies.sugar / 40,
    flour := ingredients_per_40_cookies.flour / 40
  }
  let storage : StorageCapacity := { milk := 80, sugar := 200, flour := 220 }
  max_cookies ingredients_per_cookie storage = 320 := by sorry

end NUMINAMATH_CALUDE_max_cookies_juan_l2025_202588


namespace NUMINAMATH_CALUDE_events_B_C_mutually_exclusive_not_complementary_l2025_202509

-- Define the sample space
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {n ∈ Ω | n % 2 = 1}
def B : Set Nat := {n ∈ Ω | n ≤ 2}
def C : Set Nat := {n ∈ Ω | n ≥ 4}

-- Theorem statement
theorem events_B_C_mutually_exclusive_not_complementary :
  (B ∩ C = ∅) ∧ (B ∪ C ≠ Ω) :=
sorry

end NUMINAMATH_CALUDE_events_B_C_mutually_exclusive_not_complementary_l2025_202509


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2025_202592

theorem sin_cos_identity : 
  Real.sin (73 * π / 180) * Real.cos (13 * π / 180) - 
  Real.sin (167 * π / 180) * Real.cos (73 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2025_202592


namespace NUMINAMATH_CALUDE_complex_simplification_l2025_202538

theorem complex_simplification : (1 - Complex.I)^2 + 4 * Complex.I = 2 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2025_202538


namespace NUMINAMATH_CALUDE_visitors_scientific_notation_l2025_202575

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_scientific_notation :
  toScientificNotation 564200 = ScientificNotation.mk 5.642 5 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_visitors_scientific_notation_l2025_202575


namespace NUMINAMATH_CALUDE_concatenation_product_sum_l2025_202529

theorem concatenation_product_sum : ∃! (n m : ℕ), 
  (10 ≤ n ∧ n < 100) ∧ 
  (100 ≤ m ∧ m < 1000) ∧ 
  (1000 * n + m = 9 * n * m) ∧ 
  (n + m = 126) := by
  sorry

end NUMINAMATH_CALUDE_concatenation_product_sum_l2025_202529


namespace NUMINAMATH_CALUDE_alfred_ranking_bounds_l2025_202539

/-- Represents a participant in the Generic Math Tournament -/
structure Participant where
  algebra_rank : Nat
  combinatorics_rank : Nat
  geometry_rank : Nat

/-- The total number of participants in the tournament -/
def total_participants : Nat := 99

/-- Alfred's rankings in each subject -/
def alfred : Participant :=
  { algebra_rank := 16
  , combinatorics_rank := 30
  , geometry_rank := 23 }

/-- Calculate the total score of a participant -/
def total_score (p : Participant) : Nat :=
  p.algebra_rank + p.combinatorics_rank + p.geometry_rank

/-- The best possible ranking Alfred could achieve -/
def best_ranking : Nat := 1

/-- The worst possible ranking Alfred could achieve -/
def worst_ranking : Nat := 67

theorem alfred_ranking_bounds :
  (∀ p : Participant, p ≠ alfred → total_score p ≠ total_score alfred) →
  (best_ranking = 1 ∧ worst_ranking = 67) :=
by sorry

end NUMINAMATH_CALUDE_alfred_ranking_bounds_l2025_202539


namespace NUMINAMATH_CALUDE_power_of_product_l2025_202582

theorem power_of_product (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2025_202582


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l2025_202565

theorem solution_implies_m_value (m : ℝ) : 
  (2 * 2 + m - 1 = 0) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l2025_202565


namespace NUMINAMATH_CALUDE_not_coplanar_implies_not_intersect_exists_not_intersect_but_coplanar_l2025_202595

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Check if two lines intersect -/
def linesIntersect (l1 l2 : Line3D) : Prop := sorry

theorem not_coplanar_implies_not_intersect 
  (E F G H : Point3D) 
  (EF : Line3D) 
  (GH : Line3D) 
  (h1 : EF = Line3D.mk E F) 
  (h2 : GH = Line3D.mk G H) : 
  ¬(areCoplanar E F G H) → ¬(linesIntersect EF GH) := 
sorry

theorem exists_not_intersect_but_coplanar :
  ∃ (E F G H : Point3D) (EF GH : Line3D),
    EF = Line3D.mk E F ∧ 
    GH = Line3D.mk G H ∧ 
    ¬(linesIntersect EF GH) ∧ 
    areCoplanar E F G H :=
sorry

end NUMINAMATH_CALUDE_not_coplanar_implies_not_intersect_exists_not_intersect_but_coplanar_l2025_202595


namespace NUMINAMATH_CALUDE_f_passes_through_origin_l2025_202515

def f (x : ℝ) : ℝ := -2 * x

theorem f_passes_through_origin : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_passes_through_origin_l2025_202515


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2025_202577

theorem cube_root_simplification : 
  (80^3 + 100^3 + 120^3 : ℝ)^(1/3) = 20 * 405^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2025_202577


namespace NUMINAMATH_CALUDE_min_value_of_f_l2025_202563

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2025_202563


namespace NUMINAMATH_CALUDE_max_value_inequality_l2025_202554

theorem max_value_inequality (x y : ℝ) (hx : x > 1/2) (hy : y > 1) :
  (∃ m : ℝ, ∀ x y : ℝ, x > 1/2 → y > 1 → (4 * x^2) / (y - 1) + y^2 / (2 * x - 1) ≥ m) ∧
  (∀ m : ℝ, (∀ x y : ℝ, x > 1/2 → y > 1 → (4 * x^2) / (y - 1) + y^2 / (2 * x - 1) ≥ m) → m ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2025_202554


namespace NUMINAMATH_CALUDE_range_of_a_l2025_202564

-- Define the functions f and g
def f (a x : ℝ) := a - x^2
def g (x : ℝ) := x + 1

-- Define the symmetry condition
def symmetric_about_x_axis (f g : ℝ → ℝ) (a : ℝ) :=
  ∃ x, 1 ≤ x ∧ x ≤ 2 ∧ f x = -g x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (symmetric_about_x_axis (f a) g a) → -1 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2025_202564


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l2025_202541

/-- The number of pairs of shoes Helga tried on at the first store -/
def first_store : ℕ := 7

/-- The number of pairs of shoes Helga tried on at the second store -/
def second_store : ℕ := first_store + 2

/-- The number of pairs of shoes Helga tried on at the third store -/
def third_store : ℕ := 0

/-- The total number of pairs of shoes Helga tried on at the first three stores -/
def first_three_stores : ℕ := first_store + second_store + third_store

/-- The number of pairs of shoes Helga tried on at the fourth store -/
def fourth_store : ℕ := 2 * first_three_stores

/-- The total number of pairs of shoes Helga tried on -/
def total_shoes : ℕ := first_three_stores + fourth_store

theorem helga_shoe_shopping : total_shoes = 48 := by
  sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l2025_202541


namespace NUMINAMATH_CALUDE_solve_for_y_l2025_202516

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2025_202516


namespace NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l2025_202501

theorem twenty_four_is_eighty_percent_of_thirty : 
  ∀ x : ℝ, (24 : ℝ) / x = (80 : ℝ) / 100 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_twenty_four_is_eighty_percent_of_thirty_l2025_202501


namespace NUMINAMATH_CALUDE_volleyball_tickets_l2025_202521

def initial_tickets (jude_tickets andrea_tickets sandra_tickets tickets_left : ℕ) : Prop :=
  andrea_tickets = 2 * jude_tickets ∧
  sandra_tickets = jude_tickets / 2 + 4 ∧
  jude_tickets = 16 ∧
  tickets_left = 40 ∧
  jude_tickets + andrea_tickets + sandra_tickets + tickets_left = 100

theorem volleyball_tickets :
  ∃ (jude_tickets andrea_tickets sandra_tickets tickets_left : ℕ),
    initial_tickets jude_tickets andrea_tickets sandra_tickets tickets_left :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_tickets_l2025_202521


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l2025_202504

/-- Systematic sampling problem -/
theorem systematic_sampling_problem 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (group_size : ℕ) 
  (sample_size : ℕ) 
  (sixteenth_group_num : ℕ) :
  total_students = 160 →
  num_groups = 20 →
  group_size = 8 →
  sample_size = 20 →
  sixteenth_group_num = 126 →
  ∃ (first_group_num : ℕ), 
    first_group_num + (15 * group_size) = sixteenth_group_num ∧
    first_group_num = 6 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_problem_l2025_202504


namespace NUMINAMATH_CALUDE_right_triangle_a_value_l2025_202578

/-- Proves that for a right triangle with given properties, the value of a is 14 -/
theorem right_triangle_a_value (a b : ℝ) : 
  a > 0 → -- a is positive
  b = 4 → -- b equals 4
  (1/2) * a * b = 28 → -- area of the triangle is 28
  a = 14 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_a_value_l2025_202578


namespace NUMINAMATH_CALUDE_complete_square_result_l2025_202571

theorem complete_square_result (x : ℝ) : 
  x^2 + 6*x - 4 = 0 ↔ (x + 3)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_result_l2025_202571


namespace NUMINAMATH_CALUDE_simplify_fraction_l2025_202533

theorem simplify_fraction : (144 : ℚ) / 216 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2025_202533


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2025_202525

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - x₁ - 2 = 0 ∧ x₂^2 - x₂ - 2 = 0 ∧ x₁ = 2 ∧ x₂ = -1) ∧
  (∃ y₁ y₂ : ℝ, 2*y₁^2 + 2*y₁ - 1 = 0 ∧ 2*y₂^2 + 2*y₂ - 1 = 0 ∧ 
    y₁ = (-1 + Real.sqrt 3) / 2 ∧ y₂ = (-1 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2025_202525


namespace NUMINAMATH_CALUDE_product_of_four_integers_l2025_202549

theorem product_of_four_integers (A B C D : ℕ) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0)
  (h_sum : A + B + C + D = 64)
  (h_relation : A + 3 = B - 3 ∧ A + 3 = C * 3 ∧ A + 3 = D / 3) :
  A * B * C * D = 19440 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_integers_l2025_202549


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2025_202581

theorem complex_equation_sum (a b : ℝ) :
  (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2025_202581


namespace NUMINAMATH_CALUDE_min_socks_for_eight_pairs_l2025_202593

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (yellow : ℕ)
  (green : ℕ)
  (purple : ℕ)

/-- The minimum number of socks needed to guarantee at least n pairs -/
def minSocksForPairs (drawer : SockDrawer) (n : ℕ) : ℕ :=
  sorry

/-- The specific drawer configuration in the problem -/
def problemDrawer : SockDrawer :=
  { red := 50, yellow := 100, green := 70, purple := 30 }

theorem min_socks_for_eight_pairs :
  minSocksForPairs problemDrawer 8 = 28 :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_eight_pairs_l2025_202593


namespace NUMINAMATH_CALUDE_quadratic_and_inequality_system_l2025_202547

theorem quadratic_and_inequality_system :
  -- Part 1: Quadratic equation
  (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  -- Part 2: Inequality system
  (∀ x : ℝ, x - 2*(x-1) ≤ 1 ∧ (1+x)/3 > x-1 ↔ -1 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_inequality_system_l2025_202547


namespace NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l2025_202518

-- Define the set T
def T : Set (ℝ × ℝ) := {p : ℝ × ℝ | 
  (p.1 - 3 = 5 ∧ p.2 + 1 ≥ 5) ∨
  (p.1 - 3 = p.2 + 1 ∧ 5 ≥ p.1 - 3) ∨
  (5 = p.2 + 1 ∧ p.1 - 3 ≥ 5)}

-- Define what it means for three lines to intersect at a single point
def three_lines_intersect_at_point (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∃ (l₁ l₂ l₃ : Set (ℝ × ℝ)),
    (∀ q ∈ S, q ∈ l₁ ∨ q ∈ l₂ ∨ q ∈ l₃) ∧
    (l₁ ∩ l₂ = {p}) ∧ (l₂ ∩ l₃ = {p}) ∧ (l₃ ∩ l₁ = {p}) ∧
    (∀ q ∈ l₁, ∃ r ∈ l₁, q ≠ r) ∧
    (∀ q ∈ l₂, ∃ r ∈ l₂, q ≠ r) ∧
    (∀ q ∈ l₃, ∃ r ∈ l₃, q ≠ r)

-- Theorem statement
theorem T_is_three_intersecting_lines :
  ∃ p : ℝ × ℝ, three_lines_intersect_at_point T p :=
sorry

end NUMINAMATH_CALUDE_T_is_three_intersecting_lines_l2025_202518


namespace NUMINAMATH_CALUDE_gino_bears_count_l2025_202546

theorem gino_bears_count (total : ℕ) (brown : ℕ) (white : ℕ) (black : ℕ) : 
  total = 66 → brown = 15 → white = 24 → total = brown + white + black → black = 27 := by
sorry

end NUMINAMATH_CALUDE_gino_bears_count_l2025_202546


namespace NUMINAMATH_CALUDE_tens_digit_of_square_even_for_odd_numbers_up_to_99_l2025_202517

/-- The tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem tens_digit_of_square_even_for_odd_numbers_up_to_99 :
  ∀ n : ℕ, n ≤ 99 → isOdd n → Even (tensDigit (n^2)) := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_square_even_for_odd_numbers_up_to_99_l2025_202517


namespace NUMINAMATH_CALUDE_no_integer_points_between_l2025_202523

def point_C : ℤ × ℤ := (2, 3)
def point_D : ℤ × ℤ := (101, 200)

def is_between (a b c : ℤ) : Prop := a < b ∧ b < c

theorem no_integer_points_between :
  ¬ ∃ (x y : ℤ), 
    (is_between point_C.1 x point_D.1) ∧ 
    (is_between point_C.2 y point_D.2) ∧ 
    (y - point_C.2) * (point_D.1 - point_C.1) = (point_D.2 - point_C.2) * (x - point_C.1) :=
sorry

end NUMINAMATH_CALUDE_no_integer_points_between_l2025_202523


namespace NUMINAMATH_CALUDE_toy_cost_price_l2025_202528

/-- The cost price of a toy, given the selling conditions -/
def cost_price (selling_price : ℕ) (num_sold : ℕ) (gain_equiv : ℕ) : ℚ :=
  selling_price / (num_sold + gain_equiv)

theorem toy_cost_price :
  let selling_price : ℕ := 25200
  let num_sold : ℕ := 18
  let gain_equiv : ℕ := 3
  cost_price selling_price num_sold gain_equiv = 1200 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2025_202528


namespace NUMINAMATH_CALUDE_extreme_point_value_bound_l2025_202576

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * Real.exp x - x^2

-- Define the derivative of f
def f_deriv (k : ℝ) (x : ℝ) : ℝ := k * Real.exp x - 2 * x

-- Theorem statement
theorem extreme_point_value_bound 
  (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : f_deriv k x₁ = 0) 
  (h3 : f_deriv k x₂ = 0) 
  (h4 : ∀ x, x₁ < x → x < x₂ → f_deriv k x ≠ 0) : 
  0 < f k x₁ ∧ f k x₁ < 1 := by
sorry

end

end NUMINAMATH_CALUDE_extreme_point_value_bound_l2025_202576


namespace NUMINAMATH_CALUDE_emma_money_l2025_202531

theorem emma_money (emma daya jeff brenda : ℝ) : 
  daya = 1.25 * emma →
  jeff = 0.4 * daya →
  brenda = jeff + 4 →
  brenda = 8 →
  emma = 8 := by
sorry

end NUMINAMATH_CALUDE_emma_money_l2025_202531


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2025_202573

theorem simplify_sqrt_expression (m : ℝ) (h : m < 1) : 
  Real.sqrt (m^2 - 2*m + 1) = 1 - m := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2025_202573


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2025_202544

theorem binomial_expansion_coefficient (a b : ℝ) : 
  (∃ x, (1 + a*x)^5 = 1 + 10*x + b*x^2 + a^5*x^5) → b = 40 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l2025_202544


namespace NUMINAMATH_CALUDE_fuel_a_amount_proof_l2025_202508

-- Define the tank capacity
def tank_capacity : ℝ := 200

-- Define the ethanol content percentages
def ethanol_content_a : ℝ := 0.12
def ethanol_content_b : ℝ := 0.16

-- Define the total ethanol in the full tank
def total_ethanol : ℝ := 28

-- Define the amount of fuel A added (to be proved)
def fuel_a_added : ℝ := 100

-- Theorem statement
theorem fuel_a_amount_proof :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ 
    x ≤ tank_capacity ∧
    ethanol_content_a * x + ethanol_content_b * (tank_capacity - x) = total_ethanol ∧
    x = fuel_a_added :=
by
  sorry


end NUMINAMATH_CALUDE_fuel_a_amount_proof_l2025_202508


namespace NUMINAMATH_CALUDE_unique_solution_l2025_202567

/-- A single digit is a natural number from 0 to 9. -/
def SingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- The equation that Θ must satisfy. -/
def SatisfiesEquation (Θ : ℕ) : Prop := 504 * Θ = 40 + Θ + Θ^2

theorem unique_solution :
  ∃! Θ : ℕ, SingleDigit Θ ∧ SatisfiesEquation Θ ∧ Θ = 9 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2025_202567


namespace NUMINAMATH_CALUDE_max_distance_on_circle_l2025_202513

open Complex

theorem max_distance_on_circle (z : ℂ) :
  abs (z - (1 + I)) = 1 →
  (∃ (w : ℂ), abs (w - (1 + I)) = 1 ∧ abs (w - (4 + 5*I)) ≥ abs (z - (4 + 5*I))) ∧
  (∀ (w : ℂ), abs (w - (1 + I)) = 1 → abs (w - (4 + 5*I)) ≤ 6) ∧
  (∃ (w : ℂ), abs (w - (1 + I)) = 1 ∧ abs (w - (4 + 5*I)) = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_l2025_202513


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2025_202512

theorem rectangular_solid_diagonal (a b c : ℝ) : 
  (2 * (a * b + b * c + a * c) = 26) →
  (4 * (a + b + c) = 28) →
  (a^2 + b^2 + c^2 = 23) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2025_202512


namespace NUMINAMATH_CALUDE_isabel_morning_runs_l2025_202585

/-- Represents the number of times Isabel runs the circuit in the morning -/
def morning_runs : ℕ := 7

/-- Represents the length of the circuit in meters -/
def circuit_length : ℕ := 365

/-- Represents the number of times Isabel runs the circuit in the afternoon -/
def afternoon_runs : ℕ := 3

/-- Represents the total distance Isabel runs in a week in meters -/
def weekly_distance : ℕ := 25550

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

theorem isabel_morning_runs :
  morning_runs * circuit_length * days_in_week +
  afternoon_runs * circuit_length * days_in_week = weekly_distance :=
sorry

end NUMINAMATH_CALUDE_isabel_morning_runs_l2025_202585


namespace NUMINAMATH_CALUDE_shares_ratio_l2025_202507

/-- Represents the shares of money for three individuals -/
structure Shares where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem setup -/
def problem_setup (s : Shares) : Prop :=
  s.a + s.b + s.c = 700 ∧  -- Total amount
  s.a = 280 ∧              -- A's share
  ∃ x, s.a = x * (s.b + s.c) ∧  -- A's share as a fraction of B and C
  s.b = (6/9) * (s.a + s.c)     -- B's share as 6/9 of A and C

/-- The theorem to prove -/
theorem shares_ratio (s : Shares) (h : problem_setup s) : 
  s.a / (s.b + s.c) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_shares_ratio_l2025_202507


namespace NUMINAMATH_CALUDE_bathroom_extension_l2025_202574

/-- Given a rectangular bathroom with area and width, calculate the new area after extension --/
theorem bathroom_extension (area : ℝ) (width : ℝ) (extension : ℝ) :
  area = 96 →
  width = 8 →
  extension = 2 →
  (area / width + extension) * (width + extension) = 140 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_extension_l2025_202574


namespace NUMINAMATH_CALUDE_cotton_planting_solution_l2025_202543

/-- Represents the cotton planting problem with given parameters -/
structure CottonPlanting where
  total_area : ℕ
  total_days : ℕ
  first_crew_tractors : ℕ
  first_crew_days : ℕ
  second_crew_tractors : ℕ
  second_crew_days : ℕ

/-- Calculates the required acres per tractor per day -/
def acres_per_tractor_per_day (cp : CottonPlanting) : ℚ :=
  cp.total_area / (cp.first_crew_tractors * cp.first_crew_days + cp.second_crew_tractors * cp.second_crew_days)

/-- Theorem stating that for the given parameters, each tractor needs to plant 68 acres per day -/
theorem cotton_planting_solution (cp : CottonPlanting) 
  (h1 : cp.total_area = 1700)
  (h2 : cp.total_days = 5)
  (h3 : cp.first_crew_tractors = 2)
  (h4 : cp.first_crew_days = 2)
  (h5 : cp.second_crew_tractors = 7)
  (h6 : cp.second_crew_days = 3) :
  acres_per_tractor_per_day cp = 68 := by
  sorry

end NUMINAMATH_CALUDE_cotton_planting_solution_l2025_202543


namespace NUMINAMATH_CALUDE_functional_inequality_l2025_202552

theorem functional_inequality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ -1) :
  let f : ℝ → ℝ := λ y => y^2 - y + 1
  2 * f x + x^2 * f (1/x) ≥ (3*x^3 - x^2 + 4*x + 3) / (x + 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_inequality_l2025_202552


namespace NUMINAMATH_CALUDE_correct_calculation_l2025_202596

theorem correct_calculation (x : ℝ) (h : x - 21 = 52) : 40 * x = 2920 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2025_202596


namespace NUMINAMATH_CALUDE_unique_solution_l2025_202500

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  (∃ (a b c d e : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)

theorem unique_solution :
  ∃! (n : ℕ), is_valid_number n ∧ n * 3 = 100000 * n + n + 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2025_202500


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l2025_202553

theorem inverse_proportion_order : ∀ y₁ y₂ y₃ : ℝ,
  y₁ = -6 / (-3) →
  y₂ = -6 / (-1) →
  y₃ = -6 / 2 →
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l2025_202553


namespace NUMINAMATH_CALUDE_integral_absolute_value_l2025_202532

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

theorem integral_absolute_value : ∫ x in (0)..(4), f x = 10 := by sorry

end NUMINAMATH_CALUDE_integral_absolute_value_l2025_202532


namespace NUMINAMATH_CALUDE_equation_solutions_l2025_202570

/-- The set of real solutions to the equation ∛(3 - x) + √(x - 2) = 1 -/
def solution_set : Set ℝ := {2, 3, 11}

/-- The equation ∛(3 - x) + √(x - 2) = 1 -/
def equation (x : ℝ) : Prop := Real.rpow (3 - x) (1/3) + Real.sqrt (x - 2) = 1

theorem equation_solutions :
  ∀ x : ℝ, x ∈ solution_set ↔ equation x ∧ x ≥ 2 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2025_202570


namespace NUMINAMATH_CALUDE_jo_stair_climbing_l2025_202514

/-- Number of ways to climb n stairs with 1, 2, or 3 steps at a time -/
def f : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| n + 3 => f (n + 2) + f (n + 1) + f n

/-- Number of ways to climb n stairs, finishing with a 3-step -/
def g (n : ℕ) : ℕ := if n < 3 then 0 else f (n - 3)

theorem jo_stair_climbing :
  g 8 = 13 := by sorry

end NUMINAMATH_CALUDE_jo_stair_climbing_l2025_202514


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l2025_202530

theorem cubic_polynomial_integer_root 
  (a b : ℚ) 
  (h1 : ∃ x : ℝ, x^3 + a*x + b = 0 ∧ x = 4 - 2*Real.sqrt 5) 
  (h2 : ∃ y : ℤ, (y : ℝ)^3 + a*(y : ℝ) + b = 0) :
  ∃ y : ℤ, (y : ℝ)^3 + a*(y : ℝ) + b = 0 ∧ y = -8 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l2025_202530


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2025_202540

-- Define the function f(x) = x² + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem statement
theorem f_increasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l2025_202540
