import Mathlib

namespace NUMINAMATH_CALUDE_lucky_lucy_calculation_l189_18968

theorem lucky_lucy_calculation (p q r s t : ℤ) 
  (hp : p = 2) (hq : q = 3) (hr : r = 5) (hs : s = 8) : 
  (p - q - r - s + t = p - (q - (r - (s + t)))) ↔ t = 5 := by
  sorry

end NUMINAMATH_CALUDE_lucky_lucy_calculation_l189_18968


namespace NUMINAMATH_CALUDE_mandy_bike_time_l189_18956

/-- Represents Mandy's exercise routine --/
structure ExerciseRoutine where
  yoga_time : ℝ
  gym_time : ℝ
  bike_time : ℝ

/-- Theorem: Given Mandy's exercise routine conditions, she spends 18 minutes riding her bike --/
theorem mandy_bike_time (routine : ExerciseRoutine) : 
  routine.yoga_time = 20 →
  routine.gym_time + routine.bike_time = 3/2 * routine.yoga_time →
  routine.gym_time = 2/3 * routine.bike_time →
  routine.bike_time = 18 := by
  sorry


end NUMINAMATH_CALUDE_mandy_bike_time_l189_18956


namespace NUMINAMATH_CALUDE_markers_multiple_of_four_l189_18960

-- Define the types of items
structure Items where
  coloring_books : ℕ
  markers : ℕ
  crayons : ℕ

-- Define the function to calculate the maximum number of baskets
def max_baskets (items : Items) : ℕ :=
  min (min (items.coloring_books) (items.markers)) (items.crayons)

-- Theorem statement
theorem markers_multiple_of_four (items : Items) 
  (h1 : items.coloring_books = 12)
  (h2 : items.crayons = 36)
  (h3 : max_baskets items = 4) :
  ∃ k : ℕ, items.markers = 4 * k :=
sorry

end NUMINAMATH_CALUDE_markers_multiple_of_four_l189_18960


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l189_18944

theorem gcd_lcm_product_24_54 : Nat.gcd 24 54 * Nat.lcm 24 54 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l189_18944


namespace NUMINAMATH_CALUDE_right_triangle_iff_sum_squares_eq_eight_R_squared_l189_18962

/-- A triangle with side lengths a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  circumradius_positive : 0 < R

/-- Definition of a right triangle -/
def IsRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

/-- Theorem: A triangle satisfies a² + b² + c² = 8R² if and only if it is a right triangle -/
theorem right_triangle_iff_sum_squares_eq_eight_R_squared (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 8 * t.R^2 ↔ IsRightTriangle t := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_iff_sum_squares_eq_eight_R_squared_l189_18962


namespace NUMINAMATH_CALUDE_exists_same_acquaintance_count_exists_no_three_same_acquaintance_count_l189_18917

/-- Represents a meeting with participants and their acquaintances -/
structure Meeting where
  participants : Finset ℕ
  acquaintances : ℕ → Finset ℕ
  valid : ∀ i ∈ participants, acquaintances i ⊆ participants ∧ i ∉ acquaintances i

/-- There exist at least two participants with the same number of acquaintances -/
theorem exists_same_acquaintance_count (m : Meeting) (h : 1 < m.participants.card) :
  ∃ i j, i ∈ m.participants ∧ j ∈ m.participants ∧ i ≠ j ∧
    (m.acquaintances i).card = (m.acquaintances j).card :=
  sorry

/-- There exists an arrangement of acquaintances such that no three participants have the same number of acquaintances -/
theorem exists_no_three_same_acquaintance_count (n : ℕ) (h : 1 < n) :
  ∃ m : Meeting, m.participants.card = n ∧
    ∀ i j k, i ∈ m.participants → j ∈ m.participants → k ∈ m.participants →
      i ≠ j → j ≠ k → i ≠ k →
        (m.acquaintances i).card ≠ (m.acquaintances j).card ∨
        (m.acquaintances j).card ≠ (m.acquaintances k).card ∨
        (m.acquaintances i).card ≠ (m.acquaintances k).card :=
  sorry

end NUMINAMATH_CALUDE_exists_same_acquaintance_count_exists_no_three_same_acquaintance_count_l189_18917


namespace NUMINAMATH_CALUDE_segment_construction_l189_18903

/-- A list of 99 natural numbers from 1 to 99 -/
def segments : List ℕ := List.range 99

/-- The sum of all segments -/
def total_length : ℕ := List.sum segments

/-- Predicate to check if a square can be formed -/
def can_form_square (segs : List ℕ) : Prop :=
  ∃ (side : ℕ), 4 * side = List.sum segs

/-- Predicate to check if a rectangle can be formed -/
def can_form_rectangle (segs : List ℕ) : Prop :=
  ∃ (length width : ℕ), length * width = List.sum segs ∧ length ≠ width

/-- Predicate to check if an equilateral triangle can be formed -/
def can_form_equilateral_triangle (segs : List ℕ) : Prop :=
  ∃ (side : ℕ), 3 * side = List.sum segs

theorem segment_construction :
  ¬ can_form_square segments ∧
  can_form_rectangle segments ∧
  can_form_equilateral_triangle segments :=
sorry

end NUMINAMATH_CALUDE_segment_construction_l189_18903


namespace NUMINAMATH_CALUDE_base_4_7_digit_difference_l189_18949

def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

theorem base_4_7_digit_difference : 
  num_digits 4563 4 - num_digits 4563 7 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_base_4_7_digit_difference_l189_18949


namespace NUMINAMATH_CALUDE_percent_of_self_equal_sixteen_l189_18905

theorem percent_of_self_equal_sixteen (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 16) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_self_equal_sixteen_l189_18905


namespace NUMINAMATH_CALUDE_exists_year_with_special_form_l189_18983

def is_21st_century (y : ℕ) : Prop := 2001 ≤ y ∧ y ≤ 2100

def are_distinct_digits (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem exists_year_with_special_form :
  ∃ (y : ℕ) (a b c d e f g h i j : ℕ),
    is_21st_century y ∧
    are_distinct_digits a b c d e f g h i j ∧
    is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
    is_digit f ∧ is_digit g ∧ is_digit h ∧ is_digit i ∧ is_digit j ∧
    y = (a + b * c * d * e) / (f + g * h * i * j) :=
sorry

end NUMINAMATH_CALUDE_exists_year_with_special_form_l189_18983


namespace NUMINAMATH_CALUDE_power_six_sum_l189_18979

theorem power_six_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end NUMINAMATH_CALUDE_power_six_sum_l189_18979


namespace NUMINAMATH_CALUDE_root_sum_theorem_l189_18902

theorem root_sum_theorem (a b : ℝ) : 
  (Complex.I * Real.sqrt 7 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 7 + 2) + b = 0 → 
  a + b = 39 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l189_18902


namespace NUMINAMATH_CALUDE_janice_initial_sentences_janice_started_with_258_l189_18916

/-- Calculates the number of sentences Janice started with today -/
theorem janice_initial_sentences 
  (typing_speed : ℕ) 
  (total_typing_time : ℕ) 
  (erased_sentences : ℕ) 
  (final_sentence_count : ℕ) : ℕ :=
let typed_sentences := typing_speed * total_typing_time
let added_sentences := typed_sentences - erased_sentences
final_sentence_count - added_sentences

/-- Proves that Janice started with 258 sentences today -/
theorem janice_started_with_258 : 
  janice_initial_sentences 6 53 40 536 = 258 := by
sorry

end NUMINAMATH_CALUDE_janice_initial_sentences_janice_started_with_258_l189_18916


namespace NUMINAMATH_CALUDE_equal_selection_probability_l189_18964

/-- Given a population size and sample size, prove that the probability of selection
    is equal for simple random sampling, systematic sampling, and stratified sampling. -/
theorem equal_selection_probability
  (N n : ℕ) -- Population size and sample size
  (h_N_pos : N > 0) -- Assumption: Population size is positive
  (h_n_le_N : n ≤ N) -- Assumption: Sample size is not greater than population size
  (P₁ P₂ P₃ : ℚ) -- Probabilities for each sampling method
  (h_P₁ : P₁ = n / N) -- Definition of P₁ for simple random sampling
  (h_P₂ : P₂ = n / N) -- Definition of P₂ for systematic sampling
  (h_P₃ : P₃ = n / N) -- Definition of P₃ for stratified sampling
  : P₁ = P₂ ∧ P₂ = P₃ := by
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l189_18964


namespace NUMINAMATH_CALUDE_flooring_rate_calculation_l189_18959

/-- Given a rectangular room with specified dimensions and total flooring cost,
    calculate the rate per square meter for flooring. -/
theorem flooring_rate_calculation
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500)
  : (total_cost / (length * width)) = 800 := by
  sorry

end NUMINAMATH_CALUDE_flooring_rate_calculation_l189_18959


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l189_18988

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l189_18988


namespace NUMINAMATH_CALUDE_double_reflection_of_H_l189_18953

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ := (p.2 + 1, p.1 + 1)

def H : ℝ × ℝ := (5, 1)

theorem double_reflection_of_H :
  reflect_line (reflect_x H) = (0, 4) := by sorry

end NUMINAMATH_CALUDE_double_reflection_of_H_l189_18953


namespace NUMINAMATH_CALUDE_four_people_seven_steps_l189_18978

/-- The number of ways to arrange n people on m steps with at most k people per step -/
def arrangements (n m k : ℕ) : ℕ := sorry

/-- The number of ways 4 people can stand on 7 steps with at most 3 people per step -/
theorem four_people_seven_steps : arrangements 4 7 3 = 2394 := by sorry

end NUMINAMATH_CALUDE_four_people_seven_steps_l189_18978


namespace NUMINAMATH_CALUDE_orange_count_difference_l189_18981

/-- Proves that the difference between Marcie's and Brian's orange counts is 0 -/
theorem orange_count_difference (marcie_oranges brian_oranges : ℕ) 
  (h1 : marcie_oranges = 12) (h2 : brian_oranges = 12) : 
  marcie_oranges - brian_oranges = 0 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_difference_l189_18981


namespace NUMINAMATH_CALUDE_albrecht_equation_solutions_l189_18934

theorem albrecht_equation_solutions :
  ∀ a b : ℕ+, 
    (a + 2*b - 3)^2 = a^2 + 4*b^2 - 9 ↔ 
    ((a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_albrecht_equation_solutions_l189_18934


namespace NUMINAMATH_CALUDE_lily_shopping_theorem_l189_18939

/-- Calculates the remaining amount for coffee after Lily's shopping trip --/
def remaining_for_coffee (initial_amount : ℚ) (celery_cost : ℚ) (cereal_cost : ℚ) (cereal_discount : ℚ)
  (bread_cost : ℚ) (milk_cost : ℚ) (milk_discount : ℚ) (potato_cost : ℚ) (potato_quantity : ℕ) : ℚ :=
  initial_amount - (celery_cost + cereal_cost * (1 - cereal_discount) + bread_cost + 
  milk_cost * (1 - milk_discount) + potato_cost * potato_quantity)

theorem lily_shopping_theorem (initial_amount : ℚ) (celery_cost : ℚ) (cereal_cost : ℚ) (cereal_discount : ℚ)
  (bread_cost : ℚ) (milk_cost : ℚ) (milk_discount : ℚ) (potato_cost : ℚ) (potato_quantity : ℕ) :
  initial_amount = 60 ∧ 
  celery_cost = 5 ∧ 
  cereal_cost = 12 ∧ 
  cereal_discount = 0.5 ∧ 
  bread_cost = 8 ∧ 
  milk_cost = 10 ∧ 
  milk_discount = 0.1 ∧ 
  potato_cost = 1 ∧ 
  potato_quantity = 6 →
  remaining_for_coffee initial_amount celery_cost cereal_cost cereal_discount bread_cost milk_cost milk_discount potato_cost potato_quantity = 26 := by
  sorry

#eval remaining_for_coffee 60 5 12 0.5 8 10 0.1 1 6

end NUMINAMATH_CALUDE_lily_shopping_theorem_l189_18939


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l189_18932

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 1/x + 1/y + 1/z ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l189_18932


namespace NUMINAMATH_CALUDE_parabola_equation_l189_18935

/-- A parabola with vertex at the origin, opening upward -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ
  pointOnParabola : ℝ × ℝ

/-- The parabola satisfies the given conditions -/
def satisfiesConditions (p : Parabola) : Prop :=
  let (xF, yF) := p.focus
  let (xA, yA) := p.pointOnParabola
  let yM := p.directrix 0
  yF > 0 ∧ 
  Real.sqrt ((xA - 0)^2 + (yA - yM)^2) = Real.sqrt 17 ∧
  Real.sqrt ((xA - xF)^2 + (yA - yF)^2) = 3

/-- The equation of the parabola is x² = 12y -/
def hasEquation (p : Parabola) : Prop :=
  let (x, y) := p.pointOnParabola
  x^2 = 12 * y

theorem parabola_equation (p : Parabola) 
  (h : satisfiesConditions p) : hasEquation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l189_18935


namespace NUMINAMATH_CALUDE_right_triangle_max_area_l189_18974

/-- Given a right triangle with perimeter 2, its maximum area is 3 - 2√2 -/
theorem right_triangle_max_area :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  a + b + Real.sqrt (a^2 + b^2) = 2 →
  (1/2) * a * b ≤ 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_area_l189_18974


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l189_18927

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13
  right_angle : a^2 + b^2 = c^2

/-- Square inscribed in the first triangle with vertex at right angle -/
def square_at_vertex (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the second triangle with side on hypotenuse -/
def square_on_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ t.a * y / t.c = y

/-- The main theorem -/
theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ) 
    (hx : square_at_vertex t1 x) (hy : square_on_hypotenuse t2 y) : 
    x / y = 144 / 221 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l189_18927


namespace NUMINAMATH_CALUDE_DE_length_l189_18992

-- Define the fixed points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 2)^2 + P.2^2 = 4 * ((P.1 - 1)^2 + P.2^2)

-- Define the line l
def l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 - 3}

-- Define the intersection points D and E
def intersection_points (k : ℝ) : Prop :=
  ∃ (D E : ℝ × ℝ), D ∈ C ∩ l k ∧ E ∈ C ∩ l k ∧ D ≠ E

-- Define the condition x₁x₂ + y₁y₂ = 3
def point_product_condition (D E : ℝ × ℝ) : Prop :=
  D.1 * E.1 + D.2 * E.2 = 3

-- Theorem statement
theorem DE_length :
  ∀ (k : ℝ) (D E : ℝ × ℝ),
  k > 5/12 →
  intersection_points k →
  point_product_condition D E →
  (D.1 - E.1)^2 + (D.2 - E.2)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_DE_length_l189_18992


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l189_18950

/-- 
Given a person swimming against a current, prove their swimming speed in still water.
-/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 2) 
  (h2 : distance = 6) 
  (h3 : time = 3) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 4 ∧ 
    distance = (still_water_speed - current_speed) * time := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l189_18950


namespace NUMINAMATH_CALUDE_complex_equation_solution_l189_18966

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l189_18966


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l189_18951

theorem geometric_sequence_second_term 
  (a₁ : ℝ) 
  (a₃ : ℝ) 
  (b : ℝ) 
  (h₁ : a₁ = 120) 
  (h₂ : a₃ = 64 / 30) 
  (h₃ : b > 0) 
  (h₄ : ∃ r : ℝ, a₁ * r = b ∧ b * r = a₃) : 
  b = 16 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_second_term_l189_18951


namespace NUMINAMATH_CALUDE_simultaneous_presence_probability_l189_18955

/-- The probability of two people being at a location simultaneously -/
theorem simultaneous_presence_probability :
  let arrival_window : ℝ := 2  -- 2-hour window
  let stay_duration : ℝ := 1/3  -- 20 minutes in hours
  let total_area : ℝ := arrival_window * arrival_window
  let meeting_area : ℝ := total_area - 2 * (1/2 * stay_duration * (arrival_window - stay_duration))
  meeting_area / total_area = 4/9 := by
sorry

end NUMINAMATH_CALUDE_simultaneous_presence_probability_l189_18955


namespace NUMINAMATH_CALUDE_bob_spending_theorem_l189_18910

def spending_problem (initial_amount : ℚ) : ℚ :=
  let after_monday := initial_amount / 2
  let after_tuesday := after_monday - (after_monday / 5)
  let after_wednesday := after_tuesday - (after_tuesday * 3 / 8)
  after_wednesday

theorem bob_spending_theorem :
  spending_problem 80 = 20 := by sorry

end NUMINAMATH_CALUDE_bob_spending_theorem_l189_18910


namespace NUMINAMATH_CALUDE_congruent_face_tetrahedron_volume_l189_18924

/-- A tetrahedron with congruent triangular faces -/
structure CongruentFaceTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

/-- The volume of a tetrahedron with congruent triangular faces -/
noncomputable def volume (t : CongruentFaceTetrahedron) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((-t.a^2 + t.b^2 + t.c^2) * (t.a^2 - t.b^2 + t.c^2) * (t.a^2 + t.b^2 - t.c^2))

/-- Theorem: The volume of a tetrahedron with congruent triangular faces is given by the formula -/
theorem congruent_face_tetrahedron_volume (t : CongruentFaceTetrahedron) :
  ∃ V, V = volume t ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_congruent_face_tetrahedron_volume_l189_18924


namespace NUMINAMATH_CALUDE_vector_not_parallel_l189_18952

def a : ℝ × ℝ := (1, -2)

theorem vector_not_parallel (k : ℝ) : 
  ¬ ∃ (t : ℝ), (k^2 + 1, k^2 + 1) = t • a := by sorry

end NUMINAMATH_CALUDE_vector_not_parallel_l189_18952


namespace NUMINAMATH_CALUDE_exponential_inequality_l189_18989

theorem exponential_inequality (m : ℝ) (h : 0 < m ∧ m < 1) :
  (1 - m) ^ (1/3 : ℝ) > (1 - m) ^ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l189_18989


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l189_18957

/-- The fractional equation -/
def fractional_equation (x a : ℝ) : Prop :=
  (x + a) / (x - 2) - 5 / x = 1

theorem solution_part1 :
  ∀ a : ℝ, fractional_equation 5 a → a = 1 := by sorry

theorem solution_part2 :
  fractional_equation (-5) 5 := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l189_18957


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l189_18930

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) 
  (is_right_triangle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0) 
  (tan_R : (R.2 - P.2) / (R.1 - P.1) = 4/3) 
  (PQ_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 3) : 
  Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l189_18930


namespace NUMINAMATH_CALUDE_ratio_simplification_l189_18937

theorem ratio_simplification (a b c d : ℚ) (m n : ℕ) :
  (a : ℚ) / (b : ℚ) = (c : ℚ) / (d : ℚ) →
  (m : ℚ) / (n : ℚ) = ((250 : ℚ) * 1000) / ((2 : ℚ) / 5 * 1000000) →
  (1.25 : ℚ) / (5 / 8 : ℚ) = (2 : ℚ) / (1 : ℚ) ∧
  (m : ℚ) / (n : ℚ) = (5 : ℚ) / (8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ratio_simplification_l189_18937


namespace NUMINAMATH_CALUDE_filtration_theorem_l189_18967

/-- The reduction rate of impurities per filtration -/
def reduction_rate : ℝ := 0.2

/-- The target percentage of impurities relative to the original amount -/
def target_percentage : ℝ := 0.05

/-- The logarithm of 2 -/
def log_2 : ℝ := 0.301

/-- The minimum number of filtrations required -/
def min_filtrations : ℕ := 14

theorem filtration_theorem : 
  ∀ n : ℕ, (1 - reduction_rate) ^ n < target_percentage ↔ n ≥ min_filtrations := by
  sorry

end NUMINAMATH_CALUDE_filtration_theorem_l189_18967


namespace NUMINAMATH_CALUDE_quadratic_passes_through_point_l189_18907

/-- A quadratic function passing through (-1, 0) given a - b + c = 0 -/
theorem quadratic_passes_through_point
  (a b c : ℝ) -- Coefficients of the quadratic function
  (h : a - b + c = 0) -- Given condition
  : let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c -- Definition of the quadratic function
    f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_passes_through_point_l189_18907


namespace NUMINAMATH_CALUDE_sequence_general_formula_l189_18975

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 3^n - 2) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) →
  (∀ n : ℕ, n = 1 → a n = 1) ∧ 
  (∀ n : ℕ, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l189_18975


namespace NUMINAMATH_CALUDE_calculate_expression_l189_18945

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l189_18945


namespace NUMINAMATH_CALUDE_euler_conjecture_counterexample_l189_18922

theorem euler_conjecture_counterexample : 133^5 + 110^5 + 84^5 + 27^5 = 144^5 := by
  sorry

end NUMINAMATH_CALUDE_euler_conjecture_counterexample_l189_18922


namespace NUMINAMATH_CALUDE_physics_players_l189_18904

def total_players : ℕ := 30
def math_players : ℕ := 15
def both_subjects : ℕ := 6

theorem physics_players :
  ∃ (physics_players : ℕ),
    physics_players = total_players - (math_players - both_subjects) ∧
    physics_players = 21 :=
by sorry

end NUMINAMATH_CALUDE_physics_players_l189_18904


namespace NUMINAMATH_CALUDE_grace_september_earnings_775_l189_18948

/-- Represents Grace's landscaping business earnings for September --/
def grace_september_earnings : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun small_lawn_rate large_lawn_rate small_garden_rate large_garden_rate small_mulch_rate large_mulch_rate
      small_lawn_hours large_lawn_hours small_garden_hours large_garden_hours small_mulch_hours large_mulch_hours =>
    small_lawn_rate * small_lawn_hours +
    large_lawn_rate * large_lawn_hours +
    small_garden_rate * small_garden_hours +
    large_garden_rate * large_garden_hours +
    small_mulch_rate * small_mulch_hours +
    large_mulch_rate * large_mulch_hours

/-- Theorem stating that Grace's September earnings were $775 --/
theorem grace_september_earnings_775 :
  grace_september_earnings 6 10 11 15 9 13 20 43 4 5 6 4 = 775 := by
  sorry

end NUMINAMATH_CALUDE_grace_september_earnings_775_l189_18948


namespace NUMINAMATH_CALUDE_simplify_expression_l189_18985

theorem simplify_expression (x : ℝ) : ((3 * x + 8) - 5 * x) / 2 = -x + 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l189_18985


namespace NUMINAMATH_CALUDE_matrix_product_l189_18925

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 4; 3, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, -7; 2, 3]

theorem matrix_product :
  A * B = !![8, 5; -4, -27] := by sorry

end NUMINAMATH_CALUDE_matrix_product_l189_18925


namespace NUMINAMATH_CALUDE_x0_range_l189_18998

/-- Circle C with equation x^2 + y^2 = 1 -/
def Circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Line l with equation 3x + 2y - 4 = 0 -/
def Line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 2 * p.2 - 4 = 0}

/-- Condition that there always exist two different points A, B on circle C such that OA + OB = OP -/
def ExistPoints (P : ℝ × ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, A ∈ Circle_C → B ∈ Circle_C → A ≠ B → 
    (A.1, A.2) + (B.1, B.2) = P

theorem x0_range (x0 y0 : ℝ) (hP : (x0, y0) ∈ Line_l) 
    (hExist : ExistPoints (x0, y0)) : 
  0 < x0 ∧ x0 < 24/13 := by
  sorry

end NUMINAMATH_CALUDE_x0_range_l189_18998


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l189_18931

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (initial_distance : ℝ)
  (train_length : ℝ)
  (h1 : jogger_speed = 9 * 1000 / 3600) -- 9 km/hr in m/s
  (h2 : train_speed = 45 * 1000 / 3600) -- 45 km/hr in m/s
  (h3 : initial_distance = 240)
  (h4 : train_length = 110) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 := by
sorry


end NUMINAMATH_CALUDE_train_passing_jogger_time_l189_18931


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l189_18999

/-- An isosceles triangle with one side of length 7 and perimeter 17 has other sides of lengths (5, 5) or (7, 3) -/
theorem isosceles_triangle_side_lengths :
  ∀ (a b c : ℝ),
  a = 7 ∧ 
  a + b + c = 17 ∧
  ((b = c) ∨ (a = b) ∨ (a = c)) →
  ((b = 5 ∧ c = 5) ∨ (b = 7 ∧ c = 3) ∨ (b = 3 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l189_18999


namespace NUMINAMATH_CALUDE_inequality_proof_l189_18973

theorem inequality_proof (a b c d : ℝ) :
  (a + b + c + d) * (a * b * (c + d) + (a + b) * c * d) - a * b * c * d ≤ 
  (1 / 2) * (a * (b + d) + b * (c + d) + c * (d + a))^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l189_18973


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l189_18977

/-- The volume of a rectangular parallelepiped with diagonal d, which forms angles of 60° and 45° with two of its edges, is equal to d³√2 / 8 -/
theorem parallelepiped_volume (d : ℝ) (h_d_pos : d > 0) : ∃ (V : ℝ),
  V = d^3 * Real.sqrt 2 / 8 ∧
  ∃ (a b h : ℝ),
    a > 0 ∧ b > 0 ∧ h > 0 ∧
    V = a * b * h ∧
    d^2 = a^2 + b^2 + h^2 ∧
    a / d = Real.cos (π / 4) ∧
    b / d = Real.cos (π / 3) :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l189_18977


namespace NUMINAMATH_CALUDE_unique_three_prime_product_l189_18919

def isPrime (n : ℕ) : Prop := Nat.Prime n

def primeFactors (n : ℕ) : List ℕ := sorry

theorem unique_three_prime_product : 
  ∃! n : ℕ, 
    ∃ p1 p2 p3 : ℕ, 
      isPrime p1 ∧ isPrime p2 ∧ isPrime p3 ∧
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
      n = p1 * p2 * p3 ∧
      p1 + p2 + p3 = (primeFactors 9271).sum := by sorry

end NUMINAMATH_CALUDE_unique_three_prime_product_l189_18919


namespace NUMINAMATH_CALUDE_max_value_tangent_l189_18996

theorem max_value_tangent (x₀ : ℝ) : 
  (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x ≤ 3 * Real.sin x₀ - 4 * Real.cos x₀) → 
  Real.tan x₀ = -3/4 := by
sorry

end NUMINAMATH_CALUDE_max_value_tangent_l189_18996


namespace NUMINAMATH_CALUDE_inequality_proof_l189_18918

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l189_18918


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l189_18908

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : x + y = 4) : 
  x^2 * y + x * y^2 = -8 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l189_18908


namespace NUMINAMATH_CALUDE_percentage_increase_in_earnings_l189_18954

theorem percentage_increase_in_earnings (initial_earnings new_earnings : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : new_earnings = 84) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_in_earnings_l189_18954


namespace NUMINAMATH_CALUDE_container_evaporation_l189_18946

theorem container_evaporation (initial_content : ℝ) : 
  initial_content = 1 →
  let remaining_after_day1 := initial_content - (2/3 * initial_content)
  let remaining_after_day2 := remaining_after_day1 - (1/4 * remaining_after_day1)
  remaining_after_day2 = 1/4 * initial_content := by sorry

end NUMINAMATH_CALUDE_container_evaporation_l189_18946


namespace NUMINAMATH_CALUDE_bus_trip_speed_l189_18941

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 360 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (v : ℝ), v > 0 ∧ distance / v - time_decrease = distance / (v + speed_increase) ∧ v = 40 := by
sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l189_18941


namespace NUMINAMATH_CALUDE_find_a_l189_18942

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, a^2 + 9*a + 3, 6}

-- Define set A
def A (a : ℝ) : Set ℝ := {2, |a + 3|}

-- Define the complement of A relative to U
def complement_A (a : ℝ) : Set ℝ := {3}

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = {2, a^2 + 9*a + 3, 6}) ∧ 
  (A a = {2, |a + 3|}) ∧ 
  (complement_A a = {3}) ∧ 
  (a = -9) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l189_18942


namespace NUMINAMATH_CALUDE_yellow_peaches_count_red_yellow_relation_l189_18990

/-- The number of yellow peaches in the basket -/
def yellow_peaches : ℕ := 11

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 19

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 12

/-- The difference between red and yellow peaches -/
def red_yellow_difference : ℕ := 8

theorem yellow_peaches_count : yellow_peaches = 11 := by
  sorry

theorem red_yellow_relation : red_peaches = yellow_peaches + red_yellow_difference := by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_red_yellow_relation_l189_18990


namespace NUMINAMATH_CALUDE_n_power_37_minus_n_divisibility_l189_18921

theorem n_power_37_minus_n_divisibility (n : ℤ) : 
  (∃ k : ℤ, n^37 - n = 91 * k) ∧ 
  (∃ m : ℤ, n^37 - n = 3276 * m) ∧
  (∀ l : ℤ, l > 3276 → ∃ p : ℤ, ¬ (∃ q : ℤ, p^37 - p = l * q)) :=
by sorry

end NUMINAMATH_CALUDE_n_power_37_minus_n_divisibility_l189_18921


namespace NUMINAMATH_CALUDE_car_value_correct_l189_18982

/-- The value of the car Lil Jon bought for DJ Snake's engagement -/
def car_value : ℕ := 30000

/-- The cost of the hotel stay per night -/
def hotel_cost_per_night : ℕ := 4000

/-- The number of nights stayed at the hotel -/
def nights_stayed : ℕ := 2

/-- The total value of all treats received -/
def total_value : ℕ := 158000

/-- Theorem stating that the car value is correct given the conditions -/
theorem car_value_correct :
  car_value = 30000 ∧
  hotel_cost_per_night = 4000 ∧
  nights_stayed = 2 ∧
  total_value = 158000 ∧
  (hotel_cost_per_night * nights_stayed + car_value + 4 * car_value = total_value) :=
by sorry

end NUMINAMATH_CALUDE_car_value_correct_l189_18982


namespace NUMINAMATH_CALUDE_cube_sum_problem_l189_18938

theorem cube_sum_problem (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : 
  x^9 + y^9 = 343 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l189_18938


namespace NUMINAMATH_CALUDE_halloween_candy_l189_18947

/-- The number of candy pieces Debby's sister had -/
def sister_candy : ℕ := 42

/-- The number of candy pieces eaten on the first night -/
def eaten_candy : ℕ := 35

/-- The number of candy pieces left after eating -/
def remaining_candy : ℕ := 39

/-- Debby's candy pieces -/
def debby_candy : ℕ := 32

theorem halloween_candy :
  debby_candy + sister_candy - eaten_candy = remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_l189_18947


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l189_18923

/-- Given a quadratic equation x^2 + px + q = 0 with roots p and q, 
    the product pq is either 0 or -2 -/
theorem quadratic_roots_product (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) → 
  pq = 0 ∨ pq = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l189_18923


namespace NUMINAMATH_CALUDE_puzzle_palace_spending_l189_18987

theorem puzzle_palace_spending (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 90)
  (h2 : remaining_amount = 12) :
  initial_amount - remaining_amount = 78 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_palace_spending_l189_18987


namespace NUMINAMATH_CALUDE_perpendicular_bisector_intersection_l189_18970

/-- The perpendicular bisector of two points A and B intersects the line AB at a point C.
    This theorem proves that for specific points A and B, the coordinates of C satisfy a linear equation. -/
theorem perpendicular_bisector_intersection (A B C : ℝ × ℝ) :
  A = (30, 10) →
  B = (6, 3) →
  C.1 = (A.1 + B.1) / 2 →
  C.2 = (A.2 + B.2) / 2 →
  2 * C.1 - 4 * C.2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_intersection_l189_18970


namespace NUMINAMATH_CALUDE_rectangle_width_l189_18920

/-- Given a rectangle with length 5.4 cm and area 48.6 cm², prove its width is 9 cm -/
theorem rectangle_width (length : ℝ) (area : ℝ) (h1 : length = 5.4) (h2 : area = 48.6) :
  area / length = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l189_18920


namespace NUMINAMATH_CALUDE_exp_inequality_equivalence_l189_18912

theorem exp_inequality_equivalence (x : ℝ) : 1 < Real.exp x ∧ Real.exp x < 2 ↔ 0 < x ∧ x < Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_exp_inequality_equivalence_l189_18912


namespace NUMINAMATH_CALUDE_only_nice_number_l189_18972

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def is_valid_sequence (s : ℕ → ℕ × ℕ) : Prop :=
  s 1 = (1, 3) ∧ 
  ∀ k, (s (k + 1) = (P (s k).1, Q (s k).2) ∨ s (k + 1) = (Q (s k).1, P (s k).2))

def is_nice (n : ℕ) : Prop :=
  ∃ s, is_valid_sequence s ∧ (s n).1 = (s n).2

theorem only_nice_number : ∀ n : ℕ, is_nice n ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_only_nice_number_l189_18972


namespace NUMINAMATH_CALUDE_fraction_multiplication_l189_18901

theorem fraction_multiplication : (1 : ℚ) / 3 * (3 : ℚ) / 5 * (5 : ℚ) / 6 = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l189_18901


namespace NUMINAMATH_CALUDE_angle_measure_angle_measure_proof_l189_18958

theorem angle_measure : ℝ → Prop :=
  fun x =>
    (180 - x = 4 * (90 - x)) →
    x = 60

-- The proof is omitted
theorem angle_measure_proof : ∃ x, angle_measure x :=
  sorry

end NUMINAMATH_CALUDE_angle_measure_angle_measure_proof_l189_18958


namespace NUMINAMATH_CALUDE_k_value_for_given_factors_l189_18969

/-- The length of an integer is the number of positive prime factors, not necessarily distinct, whose product is equal to the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- The prime factors of an integer as a multiset. -/
def primeFactors (n : ℕ) : Multiset ℕ := sorry

theorem k_value_for_given_factors :
  ∀ k : ℕ,
    k > 1 →
    length k = 4 →
    primeFactors k = {2, 2, 2, 3} →
    k = 24 := by
  sorry

end NUMINAMATH_CALUDE_k_value_for_given_factors_l189_18969


namespace NUMINAMATH_CALUDE_equality_equivalence_l189_18909

theorem equality_equivalence (a b c : ℝ) : 
  (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔ 
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equality_equivalence_l189_18909


namespace NUMINAMATH_CALUDE_linear_approximation_of_f_l189_18984

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 5)

theorem linear_approximation_of_f :
  let a : ℝ := 2
  let x : ℝ := 1.97
  let f_a : ℝ := f a
  let f'_a : ℝ := a / Real.sqrt (a^2 + 5)
  let Δx : ℝ := x - a
  let approximation : ℝ := f_a + f'_a * Δx
  ∃ ε > 0, |approximation - 2.98| < ε :=
by
  sorry

#check linear_approximation_of_f

end NUMINAMATH_CALUDE_linear_approximation_of_f_l189_18984


namespace NUMINAMATH_CALUDE_max_b_minus_a_l189_18900

/-- Given a function f and a constant a, finds the maximum value of b-a -/
theorem max_b_minus_a (a : ℝ) (f : ℝ → ℝ) (h1 : a > -1) 
  (h2 : ∀ x, f x = Real.exp x - a * x + (1/2) * x^2) 
  (h3 : ∀ x b, f x ≥ (1/2) * x^2 + x + b) :
  ∃ (b : ℝ), b - a ≤ 1 + Real.exp (-1) ∧ 
  (∀ c, (∀ x, f x ≥ (1/2) * x^2 + x + c) → c - a ≤ 1 + Real.exp (-1)) :=
sorry

end NUMINAMATH_CALUDE_max_b_minus_a_l189_18900


namespace NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_1_l189_18940

/-- The quadratic function f(x) = x² - 2mx + 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 5

/-- f(x) is decreasing for all x < 1 -/
def is_decreasing_before_1 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ < 1 → f m x₁ > f m x₂

theorem quadratic_decreasing_implies_m_geq_1 (m : ℝ) :
  is_decreasing_before_1 m → m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_implies_m_geq_1_l189_18940


namespace NUMINAMATH_CALUDE_evaluate_expression_l189_18913

theorem evaluate_expression : 1234562 - (12 * 3 * (2 + 7)) = 1234238 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l189_18913


namespace NUMINAMATH_CALUDE_circle_center_distance_l189_18928

/-- The distance between the center of the circle x^2 + y^2 = 4x + 6y + 3 and the point (5, -2) is √34 -/
theorem circle_center_distance :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 = 4*x + 6*y + 3
  let center : ℝ × ℝ := (2, 3)
  let point : ℝ × ℝ := (5, -2)
  (∃ x y, circle_eq x y) →
  Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_circle_center_distance_l189_18928


namespace NUMINAMATH_CALUDE_number_of_pupils_in_class_number_of_pupils_in_class_is_correct_l189_18936

/-- The number of pupils in a class, given an error in mark entry and its effect on the class average. -/
theorem number_of_pupils_in_class : ℕ :=
  let incorrect_mark : ℕ := 73
  let correct_mark : ℕ := 63
  let average_increase : ℚ := 1/2
  20

/-- Proof that the number of pupils in the class is correct. -/
theorem number_of_pupils_in_class_is_correct (n : ℕ) 
  (h1 : n = number_of_pupils_in_class)
  (h2 : (incorrect_mark - correct_mark : ℚ) / n = average_increase) : 
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_in_class_number_of_pupils_in_class_is_correct_l189_18936


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l189_18976

theorem smallest_integer_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  x % 6 = 5 ∧
  (∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 6 = 5 → x ≤ y) ∧
  x = 59 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l189_18976


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l189_18961

theorem divisibility_implies_equality (a b : ℕ) 
  (h : (a^2 + a*b + 1) % (b^2 + b*a + 1) = 0) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l189_18961


namespace NUMINAMATH_CALUDE_rosy_age_l189_18929

/-- Proves that Rosy's current age is 12 years, given the conditions about David's age -/
theorem rosy_age (rosy_age david_age : ℕ) 
  (h1 : david_age = rosy_age + 18)
  (h2 : david_age + 6 = 2 * (rosy_age + 6)) : 
  rosy_age = 12 := by
  sorry

#check rosy_age

end NUMINAMATH_CALUDE_rosy_age_l189_18929


namespace NUMINAMATH_CALUDE_evaluate_expression_l189_18995

theorem evaluate_expression : 
  Real.sqrt (9/4) - Real.sqrt (4/9) + (Real.sqrt (9/4) + Real.sqrt (4/9))^2 = 199/36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l189_18995


namespace NUMINAMATH_CALUDE_inscribed_rhombus_triangle_sides_l189_18943

/-- A triangle with an inscribed rhombus -/
structure InscribedRhombusTriangle where
  -- Side lengths of the triangle
  BC : ℝ
  AB : ℝ
  AC : ℝ
  -- Length of rhombus side
  m : ℝ
  -- Segments of BC
  p : ℝ
  q : ℝ
  -- Conditions
  rhombus_inscribed : m > 0
  positive_segments : p > 0 ∧ q > 0
  k_on_bc : BC = p + q

/-- Theorem: The sides of the triangle with an inscribed rhombus -/
theorem inscribed_rhombus_triangle_sides 
  (t : InscribedRhombusTriangle) : 
  t.BC = t.p + t.q ∧ 
  t.AB = t.m * (t.p + t.q) / t.q ∧ 
  t.AC = t.m * (t.p + t.q) / t.p :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rhombus_triangle_sides_l189_18943


namespace NUMINAMATH_CALUDE_helen_oranges_l189_18965

/-- The number of oranges Helen started with -/
def initial_oranges : ℕ := sorry

/-- Helen gets 29 more oranges from Ann -/
def oranges_from_ann : ℕ := 29

/-- Helen ends up with 38 oranges -/
def final_oranges : ℕ := 38

/-- Theorem stating that the initial number of oranges plus the oranges from Ann equals the final number of oranges -/
theorem helen_oranges : initial_oranges + oranges_from_ann = final_oranges := by sorry

end NUMINAMATH_CALUDE_helen_oranges_l189_18965


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l189_18993

/-- Calculates the cost of tax-free items given total spend, tax percentage, and tax rate -/
def cost_of_tax_free_items (total_spend : ℚ) (tax_percentage : ℚ) (tax_rate : ℚ) : ℚ :=
  let taxable_cost := total_spend * (1 - tax_percentage / 100)
  let rounded_tax := (taxable_cost * tax_rate / 100).ceil
  total_spend - (taxable_cost + rounded_tax)

theorem tax_free_items_cost :
  cost_of_tax_free_items 40 30 6 = 10 :=
by sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l189_18993


namespace NUMINAMATH_CALUDE_binomial_10_2_l189_18911

theorem binomial_10_2 : (10 : ℕ).choose 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_2_l189_18911


namespace NUMINAMATH_CALUDE_measure_of_inequality_is_zero_l189_18963

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable (μ : Measure Ω)
variable (ξ η : Ω → ℝ)

theorem measure_of_inequality_is_zero 
  (hξ_integrable : IntegrableOn (|ξ|) Set.univ μ)
  (hη_integrable : IntegrableOn (|η|) Set.univ μ)
  (h_inequality : ∀ (A : Set Ω), MeasurableSet A → ∫ x in A, ξ x ∂μ ≤ ∫ x in A, η x ∂μ) :
  μ {x | ξ x > η x} = 0 := by
  sorry

end NUMINAMATH_CALUDE_measure_of_inequality_is_zero_l189_18963


namespace NUMINAMATH_CALUDE_rectangular_prism_problem_l189_18994

theorem rectangular_prism_problem (m n r : ℕ) : 
  m > 0 → n > 0 → r > 0 → m ≤ n → n ≤ r →
  (m - 2) * (n - 2) * (r - 2) - 
  2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) + 
  4 * ((m - 2) + (n - 2) + (r - 2)) = 1985 →
  ((m = 1 ∧ n = 3 ∧ r = 1987) ∨
   (m = 1 ∧ n = 7 ∧ r = 399) ∨
   (m = 3 ∧ n = 3 ∧ r = 1981) ∨
   (m = 5 ∧ n = 5 ∧ r = 1981) ∨
   (m = 5 ∧ n = 7 ∧ r = 663)) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_problem_l189_18994


namespace NUMINAMATH_CALUDE_smallest_multiple_of_3_4_5_l189_18980

theorem smallest_multiple_of_3_4_5 : 
  ∀ n : ℕ, (3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n) → n ≥ 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_3_4_5_l189_18980


namespace NUMINAMATH_CALUDE_probability_is_seven_ninety_sixths_l189_18906

/-- Triangle PQR with given side lengths -/
structure Triangle :=
  (PQ : ℝ)
  (QR : ℝ)
  (PR : ℝ)

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { PQ := 7,
    QR := 24,
    PR := 25 }

/-- A point randomly selected inside the triangle -/
def S : Type := Unit

/-- The midpoint of side QR -/
def M (t : Triangle) : ℝ × ℝ := sorry

/-- Function to determine if a point is closer to M than to P or R -/
def closerToM (t : Triangle) (s : S) : Prop := sorry

/-- The probability of the event -/
def probability (t : Triangle) : ℝ := sorry

/-- The main theorem -/
theorem probability_is_seven_ninety_sixths :
  probability problemTriangle = 7 / 96 := by sorry

end NUMINAMATH_CALUDE_probability_is_seven_ninety_sixths_l189_18906


namespace NUMINAMATH_CALUDE_rachels_weight_l189_18997

theorem rachels_weight (rachel jimmy adam : ℝ) 
  (h1 : jimmy = rachel + 6)
  (h2 : rachel = adam + 15)
  (h3 : (rachel + jimmy + adam) / 3 = 72) :
  rachel = 75 := by
  sorry

end NUMINAMATH_CALUDE_rachels_weight_l189_18997


namespace NUMINAMATH_CALUDE_gcd_7163_209_l189_18915

theorem gcd_7163_209 : Nat.gcd 7163 209 = 19 := by
  have h1 : 7163 = 209 * 34 + 57 := by sorry
  have h2 : 209 = 57 * 3 + 38 := by sorry
  have h3 : 57 = 38 * 1 + 19 := by sorry
  have h4 : 38 = 19 * 2 := by sorry
  sorry

#check gcd_7163_209

end NUMINAMATH_CALUDE_gcd_7163_209_l189_18915


namespace NUMINAMATH_CALUDE_lap_time_improvement_l189_18914

/-- Represents the performance data for a runner -/
structure Performance where
  laps : ℕ
  time : ℕ  -- time in minutes

/-- Calculates the lap time in seconds given a Performance -/
def lapTimeInSeconds (p : Performance) : ℚ :=
  (p.time * 60) / p.laps

theorem lap_time_improvement (initial : Performance) (current : Performance) 
  (h1 : initial.laps = 8) (h2 : initial.time = 36)
  (h3 : current.laps = 10) (h4 : current.time = 35) :
  lapTimeInSeconds initial - lapTimeInSeconds current = 60 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_improvement_l189_18914


namespace NUMINAMATH_CALUDE_tim_income_percentage_tim_income_less_than_juan_l189_18971

theorem tim_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.5 * tim) 
  (h2 : mary = 0.8999999999999999 * juan) : 
  tim = 0.6 * juan := by
  sorry

theorem tim_income_less_than_juan (tim juan : ℝ) 
  (h : tim = 0.6 * juan) : 
  (juan - tim) / juan = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_tim_income_percentage_tim_income_less_than_juan_l189_18971


namespace NUMINAMATH_CALUDE_num_valid_schedules_l189_18991

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 8

/-- Represents the number of courses to be scheduled -/
def num_courses : ℕ := 4

/-- 
Calculates the number of ways to schedule courses with exactly one consecutive pair
num_periods: The total number of periods in a day
num_courses: The number of courses to be scheduled
-/
def schedule_with_one_consecutive_pair (num_periods : ℕ) (num_courses : ℕ) : ℕ := sorry

/-- The main theorem stating the number of valid schedules -/
theorem num_valid_schedules : 
  schedule_with_one_consecutive_pair num_periods num_courses = 1680 := by sorry

end NUMINAMATH_CALUDE_num_valid_schedules_l189_18991


namespace NUMINAMATH_CALUDE_part_one_part_two_l189_18986

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Define set B
def B : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

-- Part I
theorem part_one : 
  {x : ℝ | f 5 x > 9} = {x : ℝ | x < -6 ∨ x > 3} := by sorry

-- Part II
-- Define set A
def A (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ |x - 4|}

theorem part_two :
  {a : ℝ | A a ∪ B = A a} = {a : ℝ | -1 ≤ a ∧ a ≤ 0} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l189_18986


namespace NUMINAMATH_CALUDE_common_tangent_existence_l189_18926

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 169/100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 49/4

/-- Common tangent line L -/
def L (a b c : ℕ) (x y : ℝ) : Prop := a * x + b * y = c

theorem common_tangent_existence :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧
      L a b c x₁ y₁ ∧ L a b c x₂ y₂) ∧
    a + b + c = 52 :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_existence_l189_18926


namespace NUMINAMATH_CALUDE_circle_area_difference_l189_18933

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l189_18933
