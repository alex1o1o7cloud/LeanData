import Mathlib

namespace NUMINAMATH_CALUDE_special_sequence_first_term_l366_36600

/-- An arithmetic sequence with common difference 2 where a₁, a₂, and a₄ form a geometric sequence -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧  -- arithmetic sequence with difference 2
  ∃ r, a 2 = a 1 * r ∧ a 4 = a 2 * r  -- a₁, a₂, a₄ form geometric sequence

/-- The first term of the special sequence is 2 -/
theorem special_sequence_first_term (a : ℕ → ℝ) (h : special_sequence a) : a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_special_sequence_first_term_l366_36600


namespace NUMINAMATH_CALUDE_min_distance_point_l366_36680

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2 * Real.log x + a

def g (x : ℝ) : ℝ := -x^2 + 3*x - 4

def h (x : ℝ) : ℝ := f 0 x - g x

theorem min_distance_point (t : ℝ) :
  t > 0 →
  (∀ x > 0, |h x| ≥ |h t|) →
  t = (3 + Real.sqrt 33) / 6 :=
sorry

end

end NUMINAMATH_CALUDE_min_distance_point_l366_36680


namespace NUMINAMATH_CALUDE_triangle_properties_l366_36649

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) :
  (t.a / t.b = (1 + Real.cos t.A) / Real.cos t.C) →
  (t.A = π / 2) ∧
  (t.a = 1 → ∃ S : ℝ, S ≤ 1/4 ∧ 
    ∀ S' : ℝ, (∃ t' : Triangle, t'.a = 1 ∧ t'.A = π/2 ∧ S' = 1/2 * t'.b * t'.c) → 
      S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l366_36649


namespace NUMINAMATH_CALUDE_kiwi_apple_equivalence_l366_36637

/-- The value of kiwis in terms of apples -/
def kiwi_value (k : ℚ) : ℚ := k * 2

theorem kiwi_apple_equivalence :
  kiwi_value (1/4 * 20) = 10 →
  kiwi_value (3/4 * 12) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_kiwi_apple_equivalence_l366_36637


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l366_36682

theorem arctan_equation_solution (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^2) = π / 3 →
  x = (1 + Real.sqrt (13 + 4 * Real.sqrt 3)) / (2 * Real.sqrt 3) ∨
  x = (1 - Real.sqrt (13 + 4 * Real.sqrt 3)) / (2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l366_36682


namespace NUMINAMATH_CALUDE_flowers_used_for_bouquets_l366_36612

theorem flowers_used_for_bouquets (tulips roses extra_flowers : ℕ) :
  tulips = 4 → roses = 11 → extra_flowers = 4 →
  tulips + roses - extra_flowers = 11 := by
  sorry

end NUMINAMATH_CALUDE_flowers_used_for_bouquets_l366_36612


namespace NUMINAMATH_CALUDE_min_value_at_neg_one_l366_36699

/-- The quadratic function y = x^2 + 2x - 5 -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

/-- Theorem: The minimum value of f occurs at x = -1 -/
theorem min_value_at_neg_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_at_neg_one_l366_36699


namespace NUMINAMATH_CALUDE_divisibility_problem_l366_36672

theorem divisibility_problem (a : ℝ) : 
  (∃ k : ℤ, 2 * 10^10 + a = 11 * k) → 
  0 ≤ a → 
  a < 11 → 
  a = 9 := by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l366_36672


namespace NUMINAMATH_CALUDE_plane_air_time_l366_36624

/-- Proves that the time the plane spent in the air is 10/3 hours given the problem conditions. -/
theorem plane_air_time (total_distance : ℝ) (icebreaker_speed : ℝ) (plane_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 840)
  (h2 : icebreaker_speed = 20)
  (h3 : plane_speed = 120)
  (h4 : total_time = 22) :
  (total_distance - icebreaker_speed * total_time) / plane_speed = 10 / 3 := by
  sorry

#check plane_air_time

end NUMINAMATH_CALUDE_plane_air_time_l366_36624


namespace NUMINAMATH_CALUDE_sam_balloons_l366_36690

theorem sam_balloons (fred_balloons : ℝ) (dan_destroyed : ℝ) (total_after : ℝ) 
  (h1 : fred_balloons = 10.0)
  (h2 : dan_destroyed = 16.0)
  (h3 : total_after = 40.0) :
  fred_balloons + (total_after + dan_destroyed - fred_balloons) - dan_destroyed = total_after :=
by sorry

end NUMINAMATH_CALUDE_sam_balloons_l366_36690


namespace NUMINAMATH_CALUDE_half_difference_donations_l366_36669

theorem half_difference_donations (margo_donation julie_donation : ℕ) 
  (h1 : margo_donation = 4300) 
  (h2 : julie_donation = 4700) : 
  (julie_donation - margo_donation) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_half_difference_donations_l366_36669


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l366_36607

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute_balls 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l366_36607


namespace NUMINAMATH_CALUDE_fathers_age_is_45_l366_36692

/-- Proves that the father's age is 45 given the problem conditions -/
theorem fathers_age_is_45 (F C : ℕ) : 
  F = 3 * C →  -- Father's age is three times the sum of the ages of his two children
  F + 5 = 2 * (C + 10) →  -- After 5 years, father's age will be twice the sum of age of two children
  F = 45 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_is_45_l366_36692


namespace NUMINAMATH_CALUDE_jacque_suitcase_weight_l366_36686

/-- Calculates the final weight of Jacque's suitcase after his trip to France -/
theorem jacque_suitcase_weight (initial_weight : ℝ) 
  (perfume_bottles : ℕ) (perfume_weight : ℝ)
  (chocolate_weight : ℝ)
  (soap_bars : ℕ) (soap_weight : ℝ)
  (jam_jars : ℕ) (jam_weight : ℝ)
  (ounces_per_pound : ℝ) :
  initial_weight = 5 →
  perfume_bottles = 5 →
  perfume_weight = 1.2 →
  chocolate_weight = 4 →
  soap_bars = 2 →
  soap_weight = 5 →
  jam_jars = 2 →
  jam_weight = 8 →
  ounces_per_pound = 16 →
  initial_weight + 
  (perfume_bottles * perfume_weight + 
   soap_bars * soap_weight + 
   jam_jars * jam_weight) / ounces_per_pound +
  chocolate_weight = 11 := by
  sorry

end NUMINAMATH_CALUDE_jacque_suitcase_weight_l366_36686


namespace NUMINAMATH_CALUDE_bridge_length_l366_36664

/-- The length of a bridge given specific train and crossing conditions -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 125 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 250 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l366_36664


namespace NUMINAMATH_CALUDE_range_of_a_l366_36665

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (P a ∨ Q a) ∧ ¬(P a ∧ Q a) →
  ((-1 < a ∧ a ≤ 1) ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l366_36665


namespace NUMINAMATH_CALUDE_negation_equivalence_l366_36645

theorem negation_equivalence (x : ℝ) : 
  ¬(x ≥ 1 → x^2 - 4*x + 2 ≥ -1) ↔ (x < 1 ∧ x^2 - 4*x + 2 < -1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l366_36645


namespace NUMINAMATH_CALUDE_haji_mother_sales_l366_36679

theorem haji_mother_sales (tough_week_sales : ℕ) (good_weeks : ℕ) (tough_weeks : ℕ)
  (h1 : tough_week_sales = 800)
  (h2 : tough_week_sales * 2 = tough_week_sales + tough_week_sales)
  (h3 : good_weeks = 5)
  (h4 : tough_weeks = 3) :
  tough_week_sales * tough_weeks + (tough_week_sales * 2) * good_weeks = 10400 := by
  sorry

end NUMINAMATH_CALUDE_haji_mother_sales_l366_36679


namespace NUMINAMATH_CALUDE_overlap_area_is_1_2_l366_36608

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculates the area of overlap between two triangles -/
def areaOfOverlap (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The 3x3 grid of points -/
def grid : List Point :=
  [ {x := 0, y := 2}, {x := 1.5, y := 2}, {x := 3, y := 2},
    {x := 0, y := 1}, {x := 1.5, y := 1}, {x := 3, y := 1},
    {x := 0, y := 0}, {x := 1.5, y := 0}, {x := 3, y := 0} ]

/-- Triangle 1: top-left corner, middle of right edge, bottom-center point -/
def triangle1 : Triangle :=
  { p1 := {x := 0, y := 2},
    p2 := {x := 3, y := 1},
    p3 := {x := 1.5, y := 0} }

/-- Triangle 2: bottom-left corner, middle of top edge, right-center point -/
def triangle2 : Triangle :=
  { p1 := {x := 0, y := 0},
    p2 := {x := 1.5, y := 2},
    p3 := {x := 3, y := 1} }

/-- Theorem stating that the area of overlap between triangle1 and triangle2 is 1.2 square units -/
theorem overlap_area_is_1_2 : areaOfOverlap triangle1 triangle2 = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_is_1_2_l366_36608


namespace NUMINAMATH_CALUDE_factor_polynomial_l366_36617

theorem factor_polynomial (x : ℝ) : 72 * x^7 - 250 * x^13 = 2 * x^7 * (2^2 * 3^2 - 5^3 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l366_36617


namespace NUMINAMATH_CALUDE_classroom_gpa_problem_l366_36630

theorem classroom_gpa_problem (class_size : ℝ) (h_class_size_pos : class_size > 0) :
  let third_size := class_size / 3
  let rest_size := class_size - third_size
  let third_gpa := 60
  let overall_gpa := 64
  let rest_gpa := (overall_gpa * class_size - third_gpa * third_size) / rest_size
  rest_gpa = 66 := by sorry

end NUMINAMATH_CALUDE_classroom_gpa_problem_l366_36630


namespace NUMINAMATH_CALUDE_system_demonstrates_transformational_thinking_l366_36671

/-- A system of two linear equations in two variables -/
structure LinearSystem :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)

/-- The process of substituting one equation into another -/
def substitute (sys : LinearSystem) : ℝ → ℝ :=
  λ y => sys.eq1 (sys.eq2 y y) y

/-- Transformational thinking in the context of solving linear systems -/
def transformational_thinking (sys : LinearSystem) : Prop :=
  ∃ (simplified_eq : ℝ → ℝ), substitute sys = simplified_eq

/-- The given system of linear equations -/
def given_system : LinearSystem :=
  { eq1 := λ x y => 2*x + y
  , eq2 := λ x y => x - 2*y }

/-- Theorem stating that the given system demonstrates transformational thinking -/
theorem system_demonstrates_transformational_thinking :
  transformational_thinking given_system :=
sorry


end NUMINAMATH_CALUDE_system_demonstrates_transformational_thinking_l366_36671


namespace NUMINAMATH_CALUDE_probability_second_black_given_first_black_l366_36668

/-- A bag of balls with white and black colors -/
structure BallBag where
  white : ℕ
  black : ℕ

/-- The probability of drawing a specific color ball given the current state of the bag -/
def drawProbability (bag : BallBag) (isBlack : Bool) : ℚ :=
  if isBlack then
    bag.black / (bag.white + bag.black)
  else
    bag.white / (bag.white + bag.black)

/-- The probability of drawing a black ball in the second draw given a black ball was drawn first -/
def secondBlackGivenFirstBlack (initialBag : BallBag) : ℚ :=
  let bagAfterFirstDraw := BallBag.mk initialBag.white (initialBag.black - 1)
  drawProbability bagAfterFirstDraw true

theorem probability_second_black_given_first_black :
  let initialBag := BallBag.mk 3 2
  secondBlackGivenFirstBlack initialBag = 1/4 := by
  sorry

#eval secondBlackGivenFirstBlack (BallBag.mk 3 2)

end NUMINAMATH_CALUDE_probability_second_black_given_first_black_l366_36668


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l366_36622

theorem divisibility_by_nine : ∃ k : ℤ, 8 * 10^18 + 1^18 = 9 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l366_36622


namespace NUMINAMATH_CALUDE_problem_statement_l366_36643

theorem problem_statement (m : ℝ) (h : |m| = m + 1) : (4 * m + 1)^2013 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l366_36643


namespace NUMINAMATH_CALUDE_cos_beta_eq_four_fifths_l366_36678

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_E_eq_angle_G (q : Quadrilateral) (β : ℝ) : Prop := sorry

def side_EF_eq_side_GH (q : Quadrilateral) : Prop := sorry

def side_EH_ne_side_FG (q : Quadrilateral) : Prop := sorry

def perimeter (q : Quadrilateral) : ℝ := sorry

-- Main theorem
theorem cos_beta_eq_four_fifths (q : Quadrilateral) (β : ℝ) :
  is_convex q →
  angle_E_eq_angle_G q β →
  side_EF_eq_side_GH q →
  side_EH_ne_side_FG q →
  perimeter q = 720 →
  Real.cos β = 4/5 := by sorry

end NUMINAMATH_CALUDE_cos_beta_eq_four_fifths_l366_36678


namespace NUMINAMATH_CALUDE_gravel_cost_proof_l366_36628

/-- Calculates the cost of graveling two intersecting roads on a rectangular lawn. -/
def gravel_cost (lawn_length lawn_width road_width gravel_cost_per_sqm : ℕ) : ℕ :=
  let road_length_area := lawn_length * road_width
  let road_width_area := (lawn_width - road_width) * road_width
  let total_area := road_length_area + road_width_area
  total_area * gravel_cost_per_sqm

/-- Proves that the cost of graveling two intersecting roads on a rectangular lawn
    with given dimensions and costs is equal to 3900. -/
theorem gravel_cost_proof :
  gravel_cost 80 60 10 3 = 3900 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_proof_l366_36628


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l366_36623

/-- A geometric sequence where the sum of every two adjacent terms forms a geometric sequence --/
def GeometricSequenceWithAdjacentSums (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 2) + a (n + 3) = r * (a n + a (n + 1))

/-- The theorem stating the sum of specific terms in the geometric sequence --/
theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequenceWithAdjacentSums a)
  (h_sum1 : a 1 + a 2 = 1/2)
  (h_sum2 : a 3 + a 4 = 1) :
  a 7 + a 8 + a 9 + a 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l366_36623


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_l366_36654

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six :
  let a : ℚ := 1/2
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/6144 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_l366_36654


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l366_36684

theorem quadratic_roots_theorem (b c : ℝ) : 
  ({1, 2} : Set ℝ) = {x | x^2 + b*x + c = 0} → b = -3 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l366_36684


namespace NUMINAMATH_CALUDE_boys_in_class_l366_36648

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 32) (h2 : ratio_girls = 3) (h3 : ratio_boys = 5) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 20 := by
  sorry

end NUMINAMATH_CALUDE_boys_in_class_l366_36648


namespace NUMINAMATH_CALUDE_log_inequality_solution_set_complex_expression_evaluation_l366_36626

-- Part 1
theorem log_inequality_solution_set (x : ℝ) :
  (Real.log (x + 2) / Real.log (1/2) > -3) ↔ (-2 < x ∧ x < 6) :=
sorry

-- Part 2
theorem complex_expression_evaluation :
  (1/8)^(1/3) * (-7/6)^0 + 8^0.25 * 2^(1/4) + (2^(1/3) * 3^(1/2))^6 = 221/2 :=
sorry

end NUMINAMATH_CALUDE_log_inequality_solution_set_complex_expression_evaluation_l366_36626


namespace NUMINAMATH_CALUDE_suzy_books_wednesday_morning_l366_36629

/-- The number of books Suzy had at the end of Friday -/
def friday_end : ℕ := 80

/-- The number of books returned on Friday -/
def friday_returned : ℕ := 7

/-- The number of books checked out on Thursday -/
def thursday_checked_out : ℕ := 5

/-- The number of books returned on Thursday -/
def thursday_returned : ℕ := 23

/-- The number of books checked out on Wednesday -/
def wednesday_checked_out : ℕ := 43

/-- The number of books Suzy had on Wednesday morning -/
def wednesday_morning : ℕ := friday_end + friday_returned + thursday_checked_out - thursday_returned + wednesday_checked_out

theorem suzy_books_wednesday_morning : wednesday_morning = 98 := by
  sorry

end NUMINAMATH_CALUDE_suzy_books_wednesday_morning_l366_36629


namespace NUMINAMATH_CALUDE_parabola_equation_from_distances_l366_36663

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 2 * C.p * y

/-- Theorem: If a point on the parabola is 8 units from the focus and 6 units from the x-axis,
    then the parabola's equation is x^2 = 8y -/
theorem parabola_equation_from_distances (C : Parabola) (P : PointOnParabola C)
    (h_focus : Real.sqrt ((P.x)^2 + (P.y - C.p/2)^2) = 8)
    (h_xaxis : P.y = 6) :
    C.p = 4 ∧ ∀ (x y : ℝ), x^2 = 2 * C.p * y ↔ x^2 = 8 * y := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_from_distances_l366_36663


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l366_36656

/-- An increasing geometric sequence with specific conditions has a common ratio of 2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) > a n) →  -- increasing sequence
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence
  (a 1 + a 5 = 17) →  -- first condition
  (a 2 * a 4 = 16) →  -- second condition
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l366_36656


namespace NUMINAMATH_CALUDE_problem_solution_l366_36627

/-- The function f(x) defined in the problem -/
noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x) / (x + 1) + b / x

/-- The theorem statement -/
theorem problem_solution :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x > 0 → x + 2 * f a b x - 3 = 0 → x = 1) ∧
    (a = 1 ∧ b = 1) ∧
    (∀ k x : ℝ, k ≤ 0 → x > 0 → x ≠ 1 → f a b x > Real.log x / (x - 1) + k / x) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l366_36627


namespace NUMINAMATH_CALUDE_product_mod_600_l366_36647

theorem product_mod_600 : (1853 * 2101) % 600 = 553 := by sorry

end NUMINAMATH_CALUDE_product_mod_600_l366_36647


namespace NUMINAMATH_CALUDE_game_result_l366_36611

theorem game_result (x : ℝ) : ((x + 90) - 27 - x) * 11 / 3 = 231 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l366_36611


namespace NUMINAMATH_CALUDE_restaurant_theorem_l366_36676

def restaurant_problem (expenditures : List ℝ) : Prop :=
  let n := 6
  let avg := (List.sum (List.take n expenditures)) / n
  let g_spent := avg - 5
  let h_spent := 2 * (avg - g_spent)
  let total_spent := (List.sum expenditures) + g_spent + h_spent
  expenditures.length = 8 ∧
  List.take n expenditures = [13, 17, 9, 15, 11, 20] ∧
  total_spent = 104.17

theorem restaurant_theorem (expenditures : List ℝ) :
  restaurant_problem expenditures :=
sorry

end NUMINAMATH_CALUDE_restaurant_theorem_l366_36676


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l366_36618

theorem sqrt_sum_fractions : Real.sqrt (1/9 + 1/16) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l366_36618


namespace NUMINAMATH_CALUDE_hotel_stay_cost_l366_36685

/-- Calculates the total cost for a group staying at a hotel. -/
def total_hotel_cost (cost_per_night : ℕ) (num_nights : ℕ) (num_people : ℕ) : ℕ :=
  cost_per_night * num_nights * num_people

/-- Proves that the total cost for 3 people staying 3 nights at $40 per night is $360. -/
theorem hotel_stay_cost :
  total_hotel_cost 40 3 3 = 360 := by
sorry

end NUMINAMATH_CALUDE_hotel_stay_cost_l366_36685


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l366_36639

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 9 = 0 ∧ x^3 - 3*x^2 - 9*x + 27 = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l366_36639


namespace NUMINAMATH_CALUDE_grapefruit_juice_percentage_l366_36689

def total_volume : ℝ := 50
def orange_juice : ℝ := 20
def lemon_juice_percentage : ℝ := 35

theorem grapefruit_juice_percentage :
  (total_volume - orange_juice - (lemon_juice_percentage / 100) * total_volume) / total_volume * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_grapefruit_juice_percentage_l366_36689


namespace NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degree_l366_36602

def is_valid_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 180

def is_scalene (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem scalene_triangle_with_double_angle_and_36_degree :
  ∀ a b c : ℝ,
  is_valid_triangle a b c →
  is_scalene a b c →
  ((a = 2 * b ∨ b = 2 * a ∨ a = 2 * c ∨ c = 2 * a ∨ b = 2 * c ∨ c = 2 * b) ∧
   (a = 36 ∨ b = 36 ∨ c = 36)) →
  ((a = 36 ∧ b = 48 ∧ c = 96) ∨ (a = 18 ∧ b = 36 ∧ c = 126) ∨
   (a = 48 ∧ b = 96 ∧ c = 36) ∨ (a = 36 ∧ b = 126 ∧ c = 18) ∨
   (a = 96 ∧ b = 36 ∧ c = 48) ∨ (a = 126 ∧ b = 18 ∧ c = 36)) :=
by sorry

end NUMINAMATH_CALUDE_scalene_triangle_with_double_angle_and_36_degree_l366_36602


namespace NUMINAMATH_CALUDE_product_mod_seventeen_l366_36653

theorem product_mod_seventeen :
  (5007 * 5008 * 5009 * 5010 * 5011) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_l366_36653


namespace NUMINAMATH_CALUDE_total_voters_l366_36696

/-- The number of voters in each district --/
structure VoterCount where
  district1 : ℕ
  district2 : ℕ
  district3 : ℕ
  district4 : ℕ
  district5 : ℕ
  district6 : ℕ
  district7 : ℕ

/-- The conditions for voter counts in each district --/
def validVoterCount (v : VoterCount) : Prop :=
  v.district1 = 322 ∧
  v.district2 = v.district1 / 2 - 19 ∧
  v.district3 = 2 * v.district1 ∧
  v.district4 = v.district2 + 45 ∧
  v.district5 = 3 * v.district3 - 150 ∧
  v.district6 = (v.district1 + v.district4) + (v.district1 + v.district4) / 5 ∧
  v.district7 = v.district2 + (v.district5 - v.district2) / 2

/-- The theorem stating that the sum of voters in all districts is 4650 --/
theorem total_voters (v : VoterCount) (h : validVoterCount v) :
  v.district1 + v.district2 + v.district3 + v.district4 + v.district5 + v.district6 + v.district7 = 4650 := by
  sorry

end NUMINAMATH_CALUDE_total_voters_l366_36696


namespace NUMINAMATH_CALUDE_custom_op_example_l366_36615

/-- Custom binary operation ※ -/
def custom_op (A B : ℤ) : ℤ := (A + 3) * (B - 2)

/-- Theorem stating that 12 ※ 17 = 225 -/
theorem custom_op_example : custom_op 12 17 = 225 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l366_36615


namespace NUMINAMATH_CALUDE_smallest_base_for_256_is_correct_l366_36619

/-- The smallest base in which 256 (decimal) has exactly 4 digits -/
def smallest_base_for_256 : ℕ := 5

/-- Predicate to check if a number has exactly 4 digits in a given base -/
def has_exactly_four_digits (n : ℕ) (base : ℕ) : Prop :=
  base ^ 3 ≤ n ∧ n < base ^ 4

theorem smallest_base_for_256_is_correct :
  (has_exactly_four_digits 256 smallest_base_for_256) ∧
  (∀ b : ℕ, 0 < b → b < smallest_base_for_256 → ¬(has_exactly_four_digits 256 b)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_256_is_correct_l366_36619


namespace NUMINAMATH_CALUDE_point_p_transformation_l366_36688

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the rotation function
def rotate90ClockwiseAboutOrigin (p : Point2D) : Point2D :=
  { x := p.y, y := -p.x }

-- Define the reflection function
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define the composition of rotation and reflection
def rotateAndReflect (p : Point2D) : Point2D :=
  reflectAcrossXAxis (rotate90ClockwiseAboutOrigin p)

theorem point_p_transformation :
  let p : Point2D := { x := 3, y := -5 }
  rotateAndReflect p = { x := -5, y := 3 } := by sorry

end NUMINAMATH_CALUDE_point_p_transformation_l366_36688


namespace NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l366_36604

/-- Calculates the cost of paving an L-shaped floor with two types of slabs -/
theorem l_shaped_floor_paving_cost
  (length1 width1 length2 width2 : ℝ)
  (cost_a cost_b : ℝ)
  (percent_a : ℝ)
  (h_length1 : length1 = 5.5)
  (h_width1 : width1 = 3.75)
  (h_length2 : length2 = 4.25)
  (h_width2 : width2 = 2.5)
  (h_cost_a : cost_a = 1000)
  (h_cost_b : cost_b = 1200)
  (h_percent_a : percent_a = 0.6)
  (h_nonneg : length1 ≥ 0 ∧ width1 ≥ 0 ∧ length2 ≥ 0 ∧ width2 ≥ 0 ∧ cost_a ≥ 0 ∧ cost_b ≥ 0 ∧ percent_a ≥ 0 ∧ percent_a ≤ 1) :
  let area1 := length1 * width1
  let area2 := length2 * width2
  let total_area := area1 + area2
  let area_a := total_area * percent_a
  let area_b := total_area * (1 - percent_a)
  let cost := area_a * cost_a + area_b * cost_b
  cost = 33750 :=
by sorry

end NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l366_36604


namespace NUMINAMATH_CALUDE_ali_sold_ten_books_tuesday_l366_36670

/-- The number of books Ali sold on Tuesday -/
def books_sold_tuesday (initial_stock : ℕ) (sold_monday : ℕ) (sold_wednesday : ℕ)
  (sold_thursday : ℕ) (sold_friday : ℕ) (not_sold : ℕ) : ℕ :=
  initial_stock - not_sold - (sold_monday + sold_wednesday + sold_thursday + sold_friday)

/-- Theorem stating that Ali sold 10 books on Tuesday -/
theorem ali_sold_ten_books_tuesday :
  books_sold_tuesday 800 60 20 44 66 600 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ali_sold_ten_books_tuesday_l366_36670


namespace NUMINAMATH_CALUDE_sequence_a_int_l366_36613

def sequence_a (c : ℕ) : ℕ → ℤ
  | 0 => 2
  | n + 1 => c * sequence_a c n + Int.sqrt ((c^2 - 1) * (sequence_a c n^2 - 4))

theorem sequence_a_int (c : ℕ) (hc : c ≥ 1) :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a c n = k :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_int_l366_36613


namespace NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l366_36625

theorem sum_of_factorization_coefficients (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x : ℝ, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factorization_coefficients_l366_36625


namespace NUMINAMATH_CALUDE_affine_preserves_ratio_l366_36675

/-- An affine transformation in a vector space -/
noncomputable def AffineTransformation (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  V → V

/-- The ratio in which a point divides a line segment -/
def divides_segment_ratio {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (A B C : V) (p q : ℝ) : Prop :=
  q • (C - A) = p • (B - C)

/-- Theorem: Affine transformations preserve segment division ratios -/
theorem affine_preserves_ratio {V : Type*} [AddCommGroup V] [Module ℝ V]
  (L : AffineTransformation V) (A B C A' B' C' : V) (p q : ℝ) :
  L A = A' → L B = B' → L C = C' →
  divides_segment_ratio A B C p q →
  divides_segment_ratio A' B' C' p q :=
by sorry

end NUMINAMATH_CALUDE_affine_preserves_ratio_l366_36675


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l366_36610

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_2_l366_36610


namespace NUMINAMATH_CALUDE_dartboard_section_angle_l366_36691

theorem dartboard_section_angle (p : ℝ) (θ : ℝ) : 
  p = 1 / 4 →  -- probability of dart landing in a section
  p = θ / 360 →  -- probability equals ratio of central angle to full circle
  θ = 90 :=  -- central angle is 90 degrees
by sorry

end NUMINAMATH_CALUDE_dartboard_section_angle_l366_36691


namespace NUMINAMATH_CALUDE_brenda_age_is_three_l366_36632

/-- Represents the ages of family members -/
structure FamilyAges where
  addison : ℕ
  brenda : ℕ
  janet : ℕ

/-- The conditions given in the problem -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.addison = 4 * ages.brenda ∧
  ages.janet = ages.brenda + 9 ∧
  ages.addison = ages.janet

/-- Theorem stating that if the family ages are valid, Brenda's age is 3 -/
theorem brenda_age_is_three (ages : FamilyAges) 
  (h : validFamilyAges ages) : ages.brenda = 3 := by
  sorry

#check brenda_age_is_three

end NUMINAMATH_CALUDE_brenda_age_is_three_l366_36632


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l366_36644

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l366_36644


namespace NUMINAMATH_CALUDE_germination_rate_proof_l366_36681

/-- The relative frequency of germinating seeds -/
def relative_frequency_germinating_seeds (total_seeds : ℕ) (non_germinating_seeds : ℕ) : ℚ :=
  (total_seeds - non_germinating_seeds : ℚ) / total_seeds

/-- Theorem: The relative frequency of germinating seeds in a sample of 1000 seeds, 
    where 90 seeds did not germinate, is equal to 0.91 -/
theorem germination_rate_proof :
  relative_frequency_germinating_seeds 1000 90 = 91 / 100 := by
  sorry

end NUMINAMATH_CALUDE_germination_rate_proof_l366_36681


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l366_36641

/-- Given the cost of 5 dozen oranges, calculate the cost of 8 dozen oranges at the same rate -/
theorem orange_cost_calculation (cost_five_dozen : ℝ) : cost_five_dozen = 42 →
  (8 : ℝ) * (cost_five_dozen / 5) = 67.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_calculation_l366_36641


namespace NUMINAMATH_CALUDE_inequality_proof_l366_36698

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l366_36698


namespace NUMINAMATH_CALUDE_intersection_of_lines_l366_36661

/-- The x-coordinate of the intersection point of two lines -/
def intersection_x (m₁ b₁ a₂ b₂ c₂ : ℚ) : ℚ :=
  (c₂ + 2 * b₁) / (2 * m₁ + a₂)

theorem intersection_of_lines :
  let line1 : ℚ → ℚ := λ x => 3 * x - 24
  let line2 : ℚ → ℚ → Prop := λ x y => 5 * x + 2 * y = 102
  ∃ x y : ℚ, line2 x y ∧ y = line1 x ∧ x = 150 / 11 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l366_36661


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l366_36658

/-- A quadratic function is always positive if and only if its coefficient of x^2 is positive and its discriminant is negative -/
theorem quadratic_always_positive (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (a > 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l366_36658


namespace NUMINAMATH_CALUDE_problem_solution_l366_36687

-- Define the equation from the problem
def equation (n : ℕ) : Prop := 2^(2*n) = 2^n + 992

-- Define the constant term function
def constant_term (n : ℕ) : ℕ := Nat.choose (2*n) 2

-- Theorem statement
theorem problem_solution :
  (∃ n : ℕ, equation n ∧ n = 5) ∧
  constant_term 5 = 45 := by
sorry


end NUMINAMATH_CALUDE_problem_solution_l366_36687


namespace NUMINAMATH_CALUDE_convention_handshakes_l366_36673

theorem convention_handshakes (twin_sets triplet_sets : ℕ) 
  (h1 : twin_sets = 10)
  (h2 : triplet_sets = 7)
  (h3 : ∀ t : ℕ, t ≤ twin_sets → (t * 2 - 2) * 2 = t * 2 * (t * 2 - 2))
  (h4 : ∀ t : ℕ, t ≤ triplet_sets → (t * 3 - 3) * 3 = t * 3 * (t * 3 - 3))
  (h5 : ∀ t : ℕ, t ≤ twin_sets → (t * 2) * (2 * triplet_sets) = 3 * (t * 2) * triplet_sets)
  (h6 : ∀ t : ℕ, t ≤ triplet_sets → (t * 3) * (2 * twin_sets) = 3 * (t * 3) * twin_sets) :
  ((twin_sets * 2) * ((twin_sets * 2) - 2)) / 2 +
  ((triplet_sets * 3) * ((triplet_sets * 3) - 3)) / 2 +
  (twin_sets * 2) * (2 * triplet_sets) / 3 +
  (triplet_sets * 3) * (2 * twin_sets) / 3 = 922 := by
sorry

end NUMINAMATH_CALUDE_convention_handshakes_l366_36673


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l366_36642

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b : ℝ, a ≥ 1 ∧ b ≥ 1 → a + b ≥ 2) ∧ 
  (∃ a b : ℝ, a + b ≥ 2 ∧ ¬(a ≥ 1 ∧ b ≥ 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l366_36642


namespace NUMINAMATH_CALUDE_factor_polynomial_l366_36621

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l366_36621


namespace NUMINAMATH_CALUDE_two_minus_i_in_fourth_quadrant_l366_36635

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

/-- The complex number 2 - i is in the fourth quadrant. -/
theorem two_minus_i_in_fourth_quadrant :
  in_fourth_quadrant (2 - I) := by
  sorry

end NUMINAMATH_CALUDE_two_minus_i_in_fourth_quadrant_l366_36635


namespace NUMINAMATH_CALUDE_bridget_apples_l366_36601

theorem bridget_apples (x : ℕ) : 
  (x : ℚ) / 3 + 5 + 6 = x → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l366_36601


namespace NUMINAMATH_CALUDE_three_digit_special_property_l366_36609

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_three_digit (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem three_digit_special_property : 
  {n : Nat | is_three_digit n ∧ 
             is_three_digit (6 * n) ∧ 
             digit_sum n = digit_sum (6 * n)} = {117, 135} := by
  sorry

end NUMINAMATH_CALUDE_three_digit_special_property_l366_36609


namespace NUMINAMATH_CALUDE_smartphone_loss_percentage_l366_36636

/-- Calculates the percentage loss when selling an item -/
def percentageLoss (initialCost sellPrice : ℚ) : ℚ :=
  (initialCost - sellPrice) / initialCost * 100

/-- Proves that selling a $300 item for $255 results in a 15% loss -/
theorem smartphone_loss_percentage :
  percentageLoss 300 255 = 15 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_loss_percentage_l366_36636


namespace NUMINAMATH_CALUDE_line_passes_through_point_l366_36640

/-- Given that the midpoint of (k, 0) and (b, 0) is (-1, 0),
    prove that the line y = kx + b passes through (1, -2) -/
theorem line_passes_through_point
  (k b : ℝ) -- k and b are real numbers
  (h : (k + b) / 2 = -1) -- midpoint condition
  : k * 1 + b = -2 := by -- line passes through (1, -2)
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l366_36640


namespace NUMINAMATH_CALUDE_sin_cos_identity_l366_36631

theorem sin_cos_identity (x : ℝ) : 
  Real.sin x ^ 6 + Real.cos x ^ 6 + Real.sin x ^ 2 = 2 * Real.sin x ^ 4 + Real.cos x ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l366_36631


namespace NUMINAMATH_CALUDE_negation_of_existence_proposition_l366_36605

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ 3^x < x^3) ↔ (∀ x : ℝ, x > 0 → 3^x ≥ x^3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_proposition_l366_36605


namespace NUMINAMATH_CALUDE_max_abs_quadratic_function_l366_36693

theorem max_abs_quadratic_function (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (|f 0| ≤ 2) → (|f 2| ≤ 2) → (|f (-2)| ≤ 2) →
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, |f x| ≤ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_quadratic_function_l366_36693


namespace NUMINAMATH_CALUDE_sqrt_product_plus_ten_l366_36616

theorem sqrt_product_plus_ten : Real.sqrt 18 * Real.sqrt 32 + 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_ten_l366_36616


namespace NUMINAMATH_CALUDE_square_remainder_l366_36677

theorem square_remainder (n : ℤ) : n % 5 = 3 → (n^2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_l366_36677


namespace NUMINAMATH_CALUDE_meaningful_fraction_l366_36646

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l366_36646


namespace NUMINAMATH_CALUDE_complex_modulus_l366_36667

theorem complex_modulus (Z : ℂ) (h : Z * Complex.I = 2 + Complex.I) : Complex.abs Z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l366_36667


namespace NUMINAMATH_CALUDE_negative_three_squared_times_negative_one_third_cubed_l366_36651

theorem negative_three_squared_times_negative_one_third_cubed :
  -3^2 * (-1/3)^3 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_times_negative_one_third_cubed_l366_36651


namespace NUMINAMATH_CALUDE_sara_quarters_count_l366_36657

theorem sara_quarters_count (initial : Nat) (from_dad : Nat) (total : Nat) : 
  initial = 21 → from_dad = 49 → total = initial + from_dad → total = 70 :=
by sorry

end NUMINAMATH_CALUDE_sara_quarters_count_l366_36657


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l366_36695

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = Complex.abs (4 + 3*I)) :
  z.im = 4/5 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l366_36695


namespace NUMINAMATH_CALUDE_variance_mean_preserved_l366_36652

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def mean (xs : List Int) : ℚ := (xs.sum : ℚ) / xs.length

def variance (xs : List Int) : ℚ :=
  let m := mean xs
  ((xs.map (λ x => ((x : ℚ) - m) ^ 2)).sum) / xs.length

def replacement_set1 : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, -1, 5]
def replacement_set2 : List Int := [-5, 1, -3, -2, -1, 0, 1, 2, 3, 4, 5, -5]

theorem variance_mean_preserved :
  (mean initial_set = mean replacement_set1 ∧
   variance initial_set = variance replacement_set1) ∨
  (mean initial_set = mean replacement_set2 ∧
   variance initial_set = variance replacement_set2) :=
by sorry

end NUMINAMATH_CALUDE_variance_mean_preserved_l366_36652


namespace NUMINAMATH_CALUDE_least_valid_n_three_is_valid_three_is_least_valid_l366_36614

def is_valid (n : ℕ) : Prop :=
  n > 0 ∧ 
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ (n - 1)^2 % k = 0) ∧
  (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ (n - 1)^2 % k ≠ 0)

theorem least_valid_n : ∀ n : ℕ, is_valid n → n ≥ 3 :=
sorry

theorem three_is_valid : is_valid 3 :=
sorry

theorem three_is_least_valid : ∀ n : ℕ, is_valid n → n = 3 :=
sorry

end NUMINAMATH_CALUDE_least_valid_n_three_is_valid_three_is_least_valid_l366_36614


namespace NUMINAMATH_CALUDE_equation_positive_root_l366_36683

theorem equation_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_positive_root_l366_36683


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l366_36634

theorem point_on_unit_circle (s : ℝ) : 
  let x := (3 - s^2) / (3 + s^2)
  let y := 4*s / (3 + s^2)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l366_36634


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l366_36659

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), 3 * |y| - 2 * y + 8 < 23 → x ≤ y) ∧ (3 * |x| - 2 * x + 8 < 23) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l366_36659


namespace NUMINAMATH_CALUDE_dawsons_friends_l366_36633

def total_cost : ℕ := 13500
def cost_per_person : ℕ := 900

theorem dawsons_friends :
  (total_cost / cost_per_person) - 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dawsons_friends_l366_36633


namespace NUMINAMATH_CALUDE_pyramid_base_side_length_l366_36660

/-- Given a right pyramid with a square base, if the area of one lateral face
    is 120 square meters and the slant height is 40 meters, then the length
    of the side of its base is 6 meters. -/
theorem pyramid_base_side_length
  (area : ℝ) (slant_height : ℝ) (base_side : ℝ) :
  area = 120 →
  slant_height = 40 →
  area = (1/2) * base_side * slant_height →
  base_side = 6 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_base_side_length_l366_36660


namespace NUMINAMATH_CALUDE_P_in_third_quadrant_l366_36638

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The given point P -/
def P : Point :=
  { x := -2, y := -3 }

/-- Theorem: Point P lies in the third quadrant -/
theorem P_in_third_quadrant : ThirdQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_third_quadrant_l366_36638


namespace NUMINAMATH_CALUDE_second_day_sales_l366_36674

/-- Represents the ticket sales for a choral performance --/
structure TicketSales where
  senior_price : ℝ
  student_price : ℝ
  day1_senior : ℕ
  day1_student : ℕ
  day1_total : ℝ
  day2_senior : ℕ
  day2_student : ℕ

/-- The theorem to prove --/
theorem second_day_sales (ts : TicketSales)
  (h1 : ts.student_price = 9)
  (h2 : ts.day1_senior * ts.senior_price + ts.day1_student * ts.student_price = ts.day1_total)
  (h3 : ts.day1_senior = 4)
  (h4 : ts.day1_student = 3)
  (h5 : ts.day1_total = 79)
  (h6 : ts.day2_senior = 12)
  (h7 : ts.day2_student = 10) :
  ts.day2_senior * ts.senior_price + ts.day2_student * ts.student_price = 246 := by
  sorry


end NUMINAMATH_CALUDE_second_day_sales_l366_36674


namespace NUMINAMATH_CALUDE_g_composition_of_three_l366_36666

-- Define the function g
def g (x : ℝ) : ℝ := 7 * x - 3

-- State the theorem
theorem g_composition_of_three : g (g (g 3)) = 858 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l366_36666


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l366_36650

-- Define the sample space
def Ω : Type := Unit

-- Define the events
def both_miss : Set Ω := sorry
def hit_at_least_once : Set Ω := sorry

-- Define the theorem
theorem mutually_exclusive_events : 
  both_miss ∩ hit_at_least_once = ∅ := by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l366_36650


namespace NUMINAMATH_CALUDE_system_solution_l366_36606

theorem system_solution (x y z : ℝ) : 
  (x * y + z = 40 ∧ x * z + y = 51 ∧ x + y + z = 19) ↔ 
  ((x = 12 ∧ y = 3 ∧ z = 4) ∨ (x = 6 ∧ y = 5.4 ∧ z = 7.6)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l366_36606


namespace NUMINAMATH_CALUDE_decagon_diagonals_l366_36620

/-- The number of distinct diagonals in a convex decagon -/
def num_diagonals_decagon : ℕ := 35

/-- A convex decagon is a 10-sided polygon -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals_decagon = (decagon_sides * (decagon_sides - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l366_36620


namespace NUMINAMATH_CALUDE_min_xy_value_l366_36662

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z : ℝ, x * y ≥ z → z ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_min_xy_value_l366_36662


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l366_36694

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked (initial current : ℕ) : ℕ :=
  current - initial

/-- Proof that Sally picked 42 peaches from the orchard -/
theorem sally_picked_42_peaches (initial current : ℕ) 
  (h1 : initial = 13) 
  (h2 : current = 55) : 
  peaches_picked initial current = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l366_36694


namespace NUMINAMATH_CALUDE_pollution_filtration_time_l366_36603

/-- Given a pollution filtration process where:
    1. The relationship between pollutants (P mg/L) and time (t h) is given by P = P₀e^(-kt)
    2. 10% of pollutants were removed in the first 5 hours
    
    This theorem proves that the time required to remove 27.1% of pollutants is 15 hours. -/
theorem pollution_filtration_time (P₀ k : ℝ) (h1 : P₀ > 0) (h2 : k > 0) : 
  (∃ t : ℝ, t > 0 ∧ P₀ * Real.exp (-k * 5) = 0.9 * P₀) → 
  (∃ t : ℝ, t > 0 ∧ P₀ * Real.exp (-k * t) = 0.271 * P₀ ∧ t = 15) :=
by sorry


end NUMINAMATH_CALUDE_pollution_filtration_time_l366_36603


namespace NUMINAMATH_CALUDE_possible_zero_point_l366_36697

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem possible_zero_point (hf : Continuous f) 
  (h2007 : f 2007 < 0) (h2008 : f 2008 < 0) (h2009 : f 2009 > 0) :
  ∃ x ∈ Set.Ioo 2007 2008, f x = 0 ∨ ∃ y ∈ Set.Ioo 2008 2009, f y = 0 :=
by sorry


end NUMINAMATH_CALUDE_possible_zero_point_l366_36697


namespace NUMINAMATH_CALUDE_scientific_notation_of_170000_l366_36655

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10 ∧ ∃ (x : ℝ), x = a * (10 : ℝ) ^ n

/-- The problem statement -/
theorem scientific_notation_of_170000 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ 170000 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_170000_l366_36655
