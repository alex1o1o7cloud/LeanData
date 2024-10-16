import Mathlib

namespace NUMINAMATH_CALUDE_max_sleep_duration_l2177_217704

/-- A time represented by hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Checks if a given time is a happy moment -/
def is_happy_moment (t : Time) : Prop :=
  (t.hours = 4 * t.minutes) ∨ (t.minutes = 4 * t.hours)

/-- List of all happy moments in a day -/
def happy_moments : List Time :=
  sorry

/-- Calculates the time difference between two times in minutes -/
def time_difference (t1 t2 : Time) : ℕ :=
  sorry

/-- Theorem stating the maximum sleep duration -/
theorem max_sleep_duration :
  ∃ (t1 t2 : Time),
    t1 ∈ happy_moments ∧
    t2 ∈ happy_moments ∧
    time_difference t1 t2 = 239 ∧
    ∀ (t3 t4 : Time),
      t3 ∈ happy_moments →
      t4 ∈ happy_moments →
      time_difference t3 t4 ≤ 239 :=
  sorry

end NUMINAMATH_CALUDE_max_sleep_duration_l2177_217704


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2177_217765

theorem possible_values_of_a (a : ℝ) : 
  2 ∈ ({1, a^2 - 3*a + 2, a + 1} : Set ℝ) → a = 3 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2177_217765


namespace NUMINAMATH_CALUDE_weight_replacement_l2177_217705

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℝ) (new_person_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 6 →
  new_person_weight = 88 →
  ∃ (replaced_weight : ℝ),
    replaced_weight = 40 ∧
    (initial_count : ℝ) * weight_increase = new_person_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l2177_217705


namespace NUMINAMATH_CALUDE_jose_wrong_questions_l2177_217703

theorem jose_wrong_questions (total_questions : ℕ) (marks_per_question : ℕ) 
  (meghan_score jose_score alisson_score : ℕ) : 
  total_questions = 50 →
  marks_per_question = 2 →
  meghan_score = jose_score - 20 →
  jose_score = alisson_score + 40 →
  meghan_score + jose_score + alisson_score = 210 →
  total_questions * marks_per_question - jose_score = 5 * marks_per_question :=
by sorry

end NUMINAMATH_CALUDE_jose_wrong_questions_l2177_217703


namespace NUMINAMATH_CALUDE_like_terms_sum_of_exponents_l2177_217749

/-- Given two terms 5a^m * b^4 and -4a^3 * b^(n+2) are like terms, prove that m + n = 5 -/
theorem like_terms_sum_of_exponents (m n : ℕ) : 
  (∃ (a b : ℝ), 5 * a^m * b^4 = -4 * a^3 * b^(n+2)) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_of_exponents_l2177_217749


namespace NUMINAMATH_CALUDE_mike_score_l2177_217747

theorem mike_score (max_score : ℕ) (passing_percentage : ℚ) (shortfall : ℕ) (actual_score : ℕ) : 
  max_score = 780 → 
  passing_percentage = 30 / 100 → 
  shortfall = 22 → 
  actual_score = (max_score * passing_percentage).floor - shortfall → 
  actual_score = 212 := by
sorry

end NUMINAMATH_CALUDE_mike_score_l2177_217747


namespace NUMINAMATH_CALUDE_ratio_sum_squares_l2177_217783

theorem ratio_sum_squares (a b c : ℝ) : 
  b = 2 * a ∧ c = 3 * a ∧ a^2 + b^2 + c^2 = 2016 → a + b + c = 72 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_l2177_217783


namespace NUMINAMATH_CALUDE_untouchedShapesAfterGame_l2177_217724

/-- Represents a shape made of matches -/
inductive Shape
| Triangle
| Square
| Pentagon

/-- Represents the game state -/
structure GameState where
  triangles : Nat
  squares : Nat
  pentagons : Nat
  untouchedShapes : Nat
  currentPlayer : Bool  -- true for Petya, false for Vasya

/-- Represents a player's move -/
structure Move where
  shapeType : Shape
  isNewShape : Bool

/-- Optimal strategy for a player -/
def optimalMove (state : GameState) : Move :=
  sorry

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Play the game for a given number of turns -/
def playGame (initialState : GameState) (turns : Nat) : GameState :=
  sorry

/-- The main theorem to prove -/
theorem untouchedShapesAfterGame :
  let initialState : GameState := {
    triangles := 3,
    squares := 4,
    pentagons := 5,
    untouchedShapes := 12,
    currentPlayer := true
  }
  let finalState := playGame initialState 10
  finalState.untouchedShapes = 6 := by
  sorry

end NUMINAMATH_CALUDE_untouchedShapesAfterGame_l2177_217724


namespace NUMINAMATH_CALUDE_complex_real_implies_m_equals_five_l2177_217733

theorem complex_real_implies_m_equals_five (m : ℝ) (z : ℂ) :
  z = Complex.I * (m^2 - 2*m - 15) → z.im = 0 → m = 5 := by sorry

end NUMINAMATH_CALUDE_complex_real_implies_m_equals_five_l2177_217733


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l2177_217781

theorem smallest_sum_of_factors (p q r s : ℕ+) 
  (h : p.val * q.val * r.val * s.val = Nat.factorial 12) :
  p.val + q.val + r.val + s.val ≥ 1402 ∧ 
  ∃ (a b c d : ℕ+), a.val * b.val * c.val * d.val = Nat.factorial 12 ∧ 
                    a.val + b.val + c.val + d.val = 1402 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l2177_217781


namespace NUMINAMATH_CALUDE_jessica_cut_nineteen_orchids_l2177_217716

/-- The number of orchids Jessica cut from her garden -/
def orchids_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) : ℕ :=
  final_orchids - initial_orchids

/-- Theorem stating that Jessica cut 19 orchids -/
theorem jessica_cut_nineteen_orchids :
  orchids_cut 12 2 10 21 = 19 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_nineteen_orchids_l2177_217716


namespace NUMINAMATH_CALUDE_card_value_decrease_l2177_217731

theorem card_value_decrease (v : ℝ) (h : v > 0) : 
  let value_after_first_year := v * (1 - 0.5)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := (v - value_after_second_year) / v
  total_decrease = 0.55
:= by sorry

end NUMINAMATH_CALUDE_card_value_decrease_l2177_217731


namespace NUMINAMATH_CALUDE_water_addition_changes_ratio_l2177_217711

/-- Proves that adding 3 litres of water to a 45-litre mixture with initial milk to water ratio of 4:1 results in a new mixture with milk to water ratio of 3:1 -/
theorem water_addition_changes_ratio :
  let initial_volume : ℝ := 45
  let initial_milk_ratio : ℝ := 4
  let initial_water_ratio : ℝ := 1
  let added_water : ℝ := 3
  let final_milk_ratio : ℝ := 3
  let final_water_ratio : ℝ := 1

  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water

  initial_milk / final_water = final_milk_ratio / final_water_ratio :=
by
  sorry

#check water_addition_changes_ratio

end NUMINAMATH_CALUDE_water_addition_changes_ratio_l2177_217711


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2177_217758

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 < x ∧ x < 2*a + 3}

-- Theorem 1: A ⊆ B iff a ∈ [-1/2, 0]
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ∈ Set.Icc (-1/2) 0 := by sorry

-- Theorem 2: A ∩ B = ∅ iff a ∈ (-∞, -2] ∪ [3/2, +∞)
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ∈ Set.Iic (-2) ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2177_217758


namespace NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l2177_217755

theorem percentage_error_division_vs_multiplication (x : ℝ) : 
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_division_vs_multiplication_l2177_217755


namespace NUMINAMATH_CALUDE_problem_statement_l2177_217799

theorem problem_statement (x y : ℝ) : 
  y = (3/4) * x →
  x^y = y^x →
  x + y = 448/81 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2177_217799


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2177_217742

-- Define the property of being a quadratic equation
def is_quadratic (m : ℤ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, x^(m+1) - (m+1)*x - 2 = a*x^2 + b*x + c

-- State the theorem
theorem quadratic_equation_m_value :
  is_quadratic m → m = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2177_217742


namespace NUMINAMATH_CALUDE_hall_dimension_difference_l2177_217734

/-- Represents the dimensions and volume of a rectangular hall -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- The width is half the length, the height is one-third of the width, 
    and the volume is 600 cubic meters -/
def hall_constraints (hall : RectangularHall) : Prop :=
  hall.width = hall.length / 2 ∧
  hall.height = hall.width / 3 ∧
  hall.volume = 600

/-- The theorem stating the difference between length, width, and height -/
theorem hall_dimension_difference (hall : RectangularHall) 
  (h : hall_constraints hall) : 
  ∃ ε > 0, |hall.length - hall.width - hall.height - 6.43| < ε :=
sorry

end NUMINAMATH_CALUDE_hall_dimension_difference_l2177_217734


namespace NUMINAMATH_CALUDE_union_A_complement_B_l2177_217726

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1, 2}
def B : Set Int := {-1, 0, 3}

theorem union_A_complement_B : A ∪ (U \ B) = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l2177_217726


namespace NUMINAMATH_CALUDE_jerry_showers_l2177_217746

/-- Represents the water usage scenario for Jerry in July --/
structure WaterUsage where
  totalAllowance : ℕ
  drinkingCooking : ℕ
  showerUsage : ℕ
  poolLength : ℕ
  poolWidth : ℕ
  poolHeight : ℕ
  gallonToCubicFoot : ℕ

/-- Calculates the number of showers Jerry can take in July --/
def calculateShowers (w : WaterUsage) : ℕ :=
  let poolVolume := w.poolLength * w.poolWidth * w.poolHeight
  let remainingWater := w.totalAllowance - w.drinkingCooking - poolVolume
  remainingWater / w.showerUsage

/-- Theorem stating that Jerry can take 15 showers in July --/
theorem jerry_showers (w : WaterUsage) 
  (h1 : w.totalAllowance = 1000)
  (h2 : w.drinkingCooking = 100)
  (h3 : w.showerUsage = 20)
  (h4 : w.poolLength = 10)
  (h5 : w.poolWidth = 10)
  (h6 : w.poolHeight = 6)
  (h7 : w.gallonToCubicFoot = 1) :
  calculateShowers w = 15 := by
  sorry

end NUMINAMATH_CALUDE_jerry_showers_l2177_217746


namespace NUMINAMATH_CALUDE_negative_seven_x_is_product_l2177_217795

theorem negative_seven_x_is_product : ∀ x : ℝ, -7 * x = -7 * x := by sorry

end NUMINAMATH_CALUDE_negative_seven_x_is_product_l2177_217795


namespace NUMINAMATH_CALUDE_pairball_playing_time_l2177_217730

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) (h1 : total_time = 120) (h2 : num_children = 6) : 
  (2 * total_time) / num_children = 40 :=
by sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l2177_217730


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2177_217743

/-- Given a circle with equation x^2 + y^2 - 2x + 6y = 0, 
    prove that its center is at (1, -3) and its radius is √10 -/
theorem circle_center_and_radius :
  ∃ (x y : ℝ), x^2 + y^2 - 2*x + 6*y = 0 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -3) ∧ radius = Real.sqrt 10 ∧
    ∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ↔ 
                   p.1^2 + p.2^2 - 2*p.1 + 6*p.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2177_217743


namespace NUMINAMATH_CALUDE_car_speed_time_relationship_car_q_graph_representation_l2177_217741

/-- Represents a car's travel characteristics -/
structure CarTravel where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The theorem stating the relationship between Car P and Car Q's travel characteristics -/
theorem car_speed_time_relationship 
  (p q : CarTravel) 
  (h1 : p.distance = q.distance) 
  (h2 : q.speed = 3 * p.speed) : 
  q.time = p.time / 3 := by
sorry

/-- The theorem proving the graphical representation of Car Q's travel -/
theorem car_q_graph_representation 
  (p q : CarTravel) 
  (h1 : p.distance = q.distance) 
  (h2 : q.speed = 3 * p.speed) : 
  q.speed = 3 * p.speed ∧ q.time = p.time / 3 := by
sorry

end NUMINAMATH_CALUDE_car_speed_time_relationship_car_q_graph_representation_l2177_217741


namespace NUMINAMATH_CALUDE_h_piecewise_l2177_217707

/-- Piecewise function g(x) -/
noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then 3 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (9 - (x - 1.5)^2) - 3
  else if 2 ≤ x ∧ x ≤ 4 then 3 * (x - 2)
  else 0

/-- Function h(x) = g(x) + g(-x) -/
noncomputable def h (x : ℝ) : ℝ := g x + g (-x)

theorem h_piecewise :
  ∀ x : ℝ,
    ((-4 ≤ x ∧ x < -3) → h x = -3 * (x + 2)) ∧
    ((-3 ≤ x ∧ x < 0) → h x = 6) ∧
    ((0 ≤ x ∧ x < 2) → h x = 2 * Real.sqrt (9 - (x - 1.5)^2) - 6) ∧
    ((2 ≤ x ∧ x ≤ 4) → h x = 3 * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_h_piecewise_l2177_217707


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_30_60_one_third_l2177_217706

def arithmeticSeriesSum (a1 : ℚ) (an : ℚ) (d : ℚ) : ℚ :=
  let n : ℚ := (an - a1) / d + 1
  n * (a1 + an) / 2

theorem arithmetic_series_sum_30_60_one_third :
  arithmeticSeriesSum 30 60 (1/3) = 4095 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_30_60_one_third_l2177_217706


namespace NUMINAMATH_CALUDE_remainder_problem_l2177_217768

theorem remainder_problem (n : ℤ) : 
  ∃ (r : ℕ), r < 25 ∧ 
  n % 25 = r ∧ 
  (n + 15) % 5 = r % 5 → 
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2177_217768


namespace NUMINAMATH_CALUDE_girls_in_college_l2177_217798

theorem girls_in_college (total : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) (girls : ℕ) :
  total = 440 →
  ratio_boys = 6 →
  ratio_girls = 5 →
  ratio_boys * girls = ratio_girls * (total - girls) →
  girls = 200 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l2177_217798


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_means_l2177_217720

theorem arithmetic_and_geometric_means : 
  (let a := (5 + 17) / 2
   a = 11) ∧
  (let b := Real.sqrt (4 * 9)
   b = 6 ∨ b = -6) := by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_means_l2177_217720


namespace NUMINAMATH_CALUDE_consecutive_points_distance_l2177_217754

/-- Given 5 consecutive points on a straight line, prove that ae = 22 -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b = 2 * (d - c)) →   -- bc = 2 cd
  (e - d = 8) →             -- de = 8
  (b - a = 5) →             -- ab = 5
  (c - a = 11) →            -- ac = 11
  (e - a = 22) :=           -- ae = 22
by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l2177_217754


namespace NUMINAMATH_CALUDE_correct_division_result_l2177_217792

theorem correct_division_result (incorrect_divisor correct_divisor incorrect_quotient : ℕ)
  (h1 : incorrect_divisor = 63)
  (h2 : correct_divisor = 36)
  (h3 : incorrect_quotient = 24) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 42 := by
sorry

end NUMINAMATH_CALUDE_correct_division_result_l2177_217792


namespace NUMINAMATH_CALUDE_cone_height_l2177_217790

/-- The height of a cone with volume 8192π cubic inches and a vertical cross-section vertex angle of 45 degrees is equal to the cube root of 24576 inches. -/
theorem cone_height (V : ℝ) (θ : ℝ) (h : V = 8192 * Real.pi) (angle : θ = 45) :
  ∃ (H : ℝ), H = (24576 : ℝ) ^ (1/3) ∧ V = (1/3) * Real.pi * H^3 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_l2177_217790


namespace NUMINAMATH_CALUDE_cube_of_product_l2177_217794

theorem cube_of_product (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_product_l2177_217794


namespace NUMINAMATH_CALUDE_no_integer_square_diff_222_l2177_217718

theorem no_integer_square_diff_222 : ¬ ∃ (a b : ℤ), a^2 - b^2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_diff_222_l2177_217718


namespace NUMINAMATH_CALUDE_pencil_cost_is_0_602_l2177_217760

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a ruler in dollars -/
def ruler_cost : ℝ := sorry

/-- The total cost of six notebooks and four pencils is $7.44 -/
axiom six_notebooks_four_pencils : 6 * notebook_cost + 4 * pencil_cost = 7.44

/-- The total cost of three notebooks and seven pencils is $6.73 -/
axiom three_notebooks_seven_pencils : 3 * notebook_cost + 7 * pencil_cost = 6.73

/-- The total cost of one notebook, two pencils, and a ruler is $3.36 -/
axiom one_notebook_two_pencils_ruler : notebook_cost + 2 * pencil_cost + ruler_cost = 3.36

/-- The cost of each pencil is $0.602 -/
theorem pencil_cost_is_0_602 : pencil_cost = 0.602 := by sorry

end NUMINAMATH_CALUDE_pencil_cost_is_0_602_l2177_217760


namespace NUMINAMATH_CALUDE_average_price_per_book_l2177_217789

theorem average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 40 →
  books2 = 20 →
  price1 = 600 →
  price2 = 240 →
  (price1 + price2) / (books1 + books2 : ℚ) = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_price_per_book_l2177_217789


namespace NUMINAMATH_CALUDE_sum_of_magnitudes_of_roots_l2177_217719

theorem sum_of_magnitudes_of_roots (z₁ z₂ z₃ z₄ : ℂ) : 
  (z₁^4 + 3*z₁^3 + 3*z₁^2 + 3*z₁ + 1 = 0) →
  (z₂^4 + 3*z₂^3 + 3*z₂^2 + 3*z₂ + 1 = 0) →
  (z₃^4 + 3*z₃^3 + 3*z₃^2 + 3*z₃ + 1 = 0) →
  (z₄^4 + 3*z₄^3 + 3*z₄^2 + 3*z₄ + 1 = 0) →
  Complex.abs z₁ + Complex.abs z₂ + Complex.abs z₃ + Complex.abs z₄ = (7 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_magnitudes_of_roots_l2177_217719


namespace NUMINAMATH_CALUDE_field_trip_students_l2177_217778

theorem field_trip_students (teachers : ℕ) (student_ticket_cost adult_ticket_cost total_cost : ℚ) :
  teachers = 4 →
  student_ticket_cost = 1 →
  adult_ticket_cost = 3 →
  total_cost = 24 →
  ∃ (students : ℕ), students * student_ticket_cost + teachers * adult_ticket_cost = total_cost ∧ students = 12 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_students_l2177_217778


namespace NUMINAMATH_CALUDE_farm_cows_l2177_217728

theorem farm_cows (milk_per_6_cows : ℝ) (total_milk : ℝ) (weeks : ℕ) :
  milk_per_6_cows = 108 →
  total_milk = 2160 →
  weeks = 5 →
  (total_milk / (milk_per_6_cows / 6) / weeks : ℝ) = 24 :=
by sorry

end NUMINAMATH_CALUDE_farm_cows_l2177_217728


namespace NUMINAMATH_CALUDE_total_sugar_calculation_l2177_217738

def chocolate_bars : ℕ := 14
def sugar_per_bar : ℕ := 10
def lollipop_sugar : ℕ := 37

theorem total_sugar_calculation :
  chocolate_bars * sugar_per_bar + lollipop_sugar = 177 := by
  sorry

end NUMINAMATH_CALUDE_total_sugar_calculation_l2177_217738


namespace NUMINAMATH_CALUDE_fraction_to_fourth_power_l2177_217722

theorem fraction_to_fourth_power (a b : ℝ) (hb : b ≠ 0) :
  (2 * a / b) ^ 4 = 16 * a ^ 4 / b ^ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_to_fourth_power_l2177_217722


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l2177_217727

theorem average_of_four_numbers (r s t u : ℝ) :
  (5 / 2) * (r + s + t + u) = 25 → (r + s + t + u) / 4 = 2.5 := by
sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l2177_217727


namespace NUMINAMATH_CALUDE_solve_linear_system_l2177_217774

/-- Given a system of linear equations with parameters m and n,
    prove that m + n = -2 when x = 2 and y = 1 is a solution. -/
theorem solve_linear_system (m n : ℚ) : 
  (2 * m + 1 = -3) → (2 - 2 * 1 = 2 * n) → m + n = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l2177_217774


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l2177_217786

/-- A rectangle divided into four smaller rectangles -/
structure DividedRectangle where
  /-- Width of the left rectangles -/
  a : ℝ
  /-- Height of the top rectangles -/
  b : ℝ
  /-- Width of the right rectangles -/
  c : ℝ
  /-- Height of the bottom rectangles -/
  d : ℝ
  /-- Area of the top left rectangle is 6 -/
  top_left_area : a * b = 6
  /-- Area of the top right rectangle is 15 -/
  top_right_area : b * c = 15
  /-- Area of the bottom right rectangle is 25 -/
  bottom_right_area : c * d = 25

/-- The area of the fourth (shaded) rectangle in a DividedRectangle is 10 -/
theorem fourth_rectangle_area (r : DividedRectangle) : r.a * r.d = 10 := by
  sorry


end NUMINAMATH_CALUDE_fourth_rectangle_area_l2177_217786


namespace NUMINAMATH_CALUDE_gcd_chain_theorem_l2177_217773

/-- Represents the operation of finding the greatest common divisor -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- The main theorem to be proved -/
theorem gcd_chain_theorem : gcd_op (gcd_op (gcd_op 20 16) (gcd_op 18 24)) 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_chain_theorem_l2177_217773


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2177_217723

theorem smallest_n_congruence (n : ℕ) : 
  (0 ≤ n ∧ n < 53 ∧ 50 * n % 53 = 47 % 53) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2177_217723


namespace NUMINAMATH_CALUDE_process_time_600_parts_l2177_217752

/-- Linear regression equation for processing time -/
def process_time (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem: The time required to process 600 parts is 6.5 hours -/
theorem process_time_600_parts : process_time 600 = 6.5 := by
  sorry

#check process_time_600_parts

end NUMINAMATH_CALUDE_process_time_600_parts_l2177_217752


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_eight_l2177_217785

theorem sum_of_A_and_C_is_eight :
  ∀ (A B C D : ℕ),
    A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A : ℚ) / B - (C : ℚ) / D = 2 →
    A + C = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_A_and_C_is_eight_l2177_217785


namespace NUMINAMATH_CALUDE_apartment_households_l2177_217757

/-- Represents the position and structure of an apartment building --/
structure ApartmentBuilding where
  houses_per_row : ℕ
  floors : ℕ
  households_per_house : ℕ

/-- Represents the position of Mijoo's house in the apartment building --/
structure MijooHousePosition where
  from_left : ℕ
  from_right : ℕ
  from_top : ℕ
  from_bottom : ℕ

/-- Calculates the total number of households in the apartment building --/
def total_households (building : ApartmentBuilding) : ℕ :=
  building.houses_per_row * building.floors * building.households_per_house

/-- Theorem stating the total number of households in the apartment building --/
theorem apartment_households 
  (building : ApartmentBuilding)
  (mijoo_position : MijooHousePosition)
  (h1 : mijoo_position.from_left = 1)
  (h2 : mijoo_position.from_right = 7)
  (h3 : mijoo_position.from_top = 2)
  (h4 : mijoo_position.from_bottom = 4)
  (h5 : building.houses_per_row = mijoo_position.from_left + mijoo_position.from_right - 1)
  (h6 : building.floors = mijoo_position.from_top + mijoo_position.from_bottom - 1)
  (h7 : building.households_per_house = 3) :
  total_households building = 105 := by
  sorry

#eval total_households { houses_per_row := 7, floors := 5, households_per_house := 3 }

end NUMINAMATH_CALUDE_apartment_households_l2177_217757


namespace NUMINAMATH_CALUDE_prime_sum_product_l2177_217753

theorem prime_sum_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p ≠ q → p + q = 10 → p * q = 21 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l2177_217753


namespace NUMINAMATH_CALUDE_checkerboard_swap_iff_div_three_l2177_217793

/-- Represents the color of a cell -/
inductive Color
  | White
  | Black
  | Green

/-- Represents a grid of size n × n -/
def Grid (n : ℕ) := Fin n → Fin n → Color

/-- Initial checkerboard coloring with at least one corner black -/
def initialGrid (n : ℕ) : Grid n := 
  λ i j => if (i.val + j.val) % 2 = 0 then Color.Black else Color.White

/-- Recoloring rule for a 2×2 subgrid -/
def recolorSubgrid (g : Grid n) (i j : Fin n) : Grid n :=
  λ x y => if (x.val ≥ i.val && x.val < i.val + 2 && y.val ≥ j.val && y.val < j.val + 2)
    then match g x y with
      | Color.White => Color.Black
      | Color.Black => Color.Green
      | Color.Green => Color.White
    else g x y

/-- Check if the grid is in a swapped checkerboard pattern -/
def isSwappedCheckerboard (g : Grid n) : Prop :=
  ∀ i j, g i j = if (i.val + j.val) % 2 = 0 then Color.White else Color.Black

/-- Main theorem: The checkerboard color swap is possible iff n is divisible by 3 -/
theorem checkerboard_swap_iff_div_three (n : ℕ) :
  (∃ (moves : List (Fin n × Fin n)), 
    isSwappedCheckerboard (moves.foldl (λ g (i, j) => recolorSubgrid g i j) (initialGrid n))) 
  ↔ 
  3 ∣ n := by sorry

end NUMINAMATH_CALUDE_checkerboard_swap_iff_div_three_l2177_217793


namespace NUMINAMATH_CALUDE_ellipse_condition_l2177_217721

/-- Represents an ellipse with equation ax^2 + by^2 = 1 -/
structure Ellipse (a b : ℝ) where
  equation : ∀ x y : ℝ, a * x^2 + b * y^2 = 1
  is_ellipse : True  -- We assume it's an ellipse
  foci_on_x_axis : True  -- We assume foci are on x-axis

/-- 
If ax^2 + by^2 = 1 represents an ellipse with foci on the x-axis,
where a and b are real numbers, then b > a > 0.
-/
theorem ellipse_condition (a b : ℝ) (e : Ellipse a b) : b > a ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2177_217721


namespace NUMINAMATH_CALUDE_power_calculation_l2177_217769

theorem power_calculation : (-8 : ℝ)^2023 * (1/8 : ℝ)^2024 = -1/8 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l2177_217769


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2177_217784

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2177_217784


namespace NUMINAMATH_CALUDE_perfect_square_count_l2177_217759

theorem perfect_square_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n > 0 ∧ n ≤ 2000 ∧ ∃ k : Nat, 21 * n = k * k) ∧ 
  S.card = 9 ∧
  (∀ n : Nat, n > 0 ∧ n ≤ 2000 ∧ (∃ k : Nat, 21 * n = k * k) → n ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_count_l2177_217759


namespace NUMINAMATH_CALUDE_carly_grill_capacity_l2177_217748

/-- The number of burgers Carly can fit on the grill at once -/
def burgers_on_grill (guests : ℕ) (cooking_time_per_burger : ℕ) (total_cooking_time : ℕ) : ℕ :=
  let total_burgers := guests / 2 * 2 + guests / 2 * 1
  total_burgers * cooking_time_per_burger / total_cooking_time

theorem carly_grill_capacity :
  burgers_on_grill 30 8 72 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carly_grill_capacity_l2177_217748


namespace NUMINAMATH_CALUDE_mel_weight_is_70_l2177_217782

-- Define Mel's weight
def mel_weight : ℝ := 70

-- Define Brenda's weight in terms of Mel's weight
def brenda_weight (m : ℝ) : ℝ := 3 * m + 10

-- Theorem statement
theorem mel_weight_is_70 : 
  brenda_weight mel_weight = 220 ∧ mel_weight = 70 :=
by sorry

end NUMINAMATH_CALUDE_mel_weight_is_70_l2177_217782


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2177_217739

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 4) :
  (((2 * x + 2) / (x^2 - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1))) = 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2177_217739


namespace NUMINAMATH_CALUDE_triangle_side_length_l2177_217766

noncomputable def triangleConfiguration (OA OC OD OB BD : ℝ) : ℝ → Prop :=
  λ y => OA = 5 ∧ OC = 12 ∧ OD = 5 ∧ OB = 3 ∧ BD = 6 ∧ 
    y^2 = OA^2 + OC^2 - 2 * OA * OC * ((OD^2 + OB^2 - BD^2) / (2 * OD * OB))

theorem triangle_side_length : 
  ∃ (OA OC OD OB BD : ℝ), triangleConfiguration OA OC OD OB BD (3 * Real.sqrt 67) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2177_217766


namespace NUMINAMATH_CALUDE_solution_for_y_l2177_217788

theorem solution_for_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 1 + 1/y) (eq2 : y = 2 + 1/x) :
  y = 1 + Real.sqrt 3 ∨ y = 1 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_for_y_l2177_217788


namespace NUMINAMATH_CALUDE_solution_is_one_third_l2177_217714

-- Define the logarithm function with base √3
noncomputable def log_sqrt3 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 3)

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ log_sqrt3 x * Real.sqrt (log_sqrt3 3 - (Real.log 9 / Real.log x)) + 4 = 0

-- Theorem statement
theorem solution_is_one_third :
  ∃ (x : ℝ), equation x ∧ x = 1/3 :=
sorry

end NUMINAMATH_CALUDE_solution_is_one_third_l2177_217714


namespace NUMINAMATH_CALUDE_baguette_cost_is_two_l2177_217767

/-- The cost of a single baguette given the initial amount, number of items bought,
    cost of water, and remaining amount after purchase. -/
def baguette_cost (initial_amount : ℚ) (num_baguettes : ℕ) (num_water : ℕ) 
                  (water_cost : ℚ) (remaining_amount : ℚ) : ℚ :=
  (initial_amount - remaining_amount - num_water * water_cost) / num_baguettes

/-- Theorem stating that the cost of each baguette is $2 given the problem conditions. -/
theorem baguette_cost_is_two :
  baguette_cost 50 2 2 1 44 = 2 := by
  sorry

end NUMINAMATH_CALUDE_baguette_cost_is_two_l2177_217767


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l2177_217763

theorem complex_roots_quadratic (b c : ℝ) : 
  (Complex.I + 1) ^ 2 + b * (Complex.I + 1) + c = 0 →
  (b = -2 ∧ c = 2) ∧ 
  ((Complex.I - 1) ^ 2 + b * (Complex.I - 1) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l2177_217763


namespace NUMINAMATH_CALUDE_overlapping_rectangles_area_l2177_217710

/-- Given two overlapping rectangles, prove the area of the non-overlapping part of one rectangle -/
theorem overlapping_rectangles_area (a b c d overlap : ℕ) : 
  a * b = 80 → 
  c * d = 108 → 
  overlap = 37 → 
  c * d - (a * b - overlap) = 65 := by
sorry

end NUMINAMATH_CALUDE_overlapping_rectangles_area_l2177_217710


namespace NUMINAMATH_CALUDE_pi_fourth_in_range_of_f_l2177_217713

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem pi_fourth_in_range_of_f : ∃ (x : ℝ), f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_pi_fourth_in_range_of_f_l2177_217713


namespace NUMINAMATH_CALUDE_coupon_usage_day_l2177_217764

theorem coupon_usage_day (coupon_count : Nat) (interval : Nat) : 
  coupon_count = 6 →
  interval = 10 →
  (∀ i : Fin coupon_count, ((i.val * interval) % 7 ≠ 0)) →
  (0 * interval) % 7 = 3 :=
by sorry

end NUMINAMATH_CALUDE_coupon_usage_day_l2177_217764


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2177_217715

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 210 → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2177_217715


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2177_217777

/-- Given a rectangle with one side of length 18 and the sum of its area and perimeter being 2016,
    the perimeter of the rectangle is 234. -/
theorem rectangle_perimeter : ∀ w : ℝ,
  let l : ℝ := 18
  let area : ℝ := l * w
  let perimeter : ℝ := 2 * (l + w)
  area + perimeter = 2016 →
  perimeter = 234 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2177_217777


namespace NUMINAMATH_CALUDE_evaluate_expression_l2177_217787

theorem evaluate_expression (x : ℕ) (h : x = 3) : x + x^2 * (x^(x^2)) = 177150 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2177_217787


namespace NUMINAMATH_CALUDE_symmetric_function_value_l2177_217702

/-- A function symmetric about x=1 -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (2 - x)

/-- The main theorem -/
theorem symmetric_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOne f)
  (h_def : ∀ x ≥ 1, f x = x * (1 - x)) : 
  f (-2) = -12 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_value_l2177_217702


namespace NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_joker_l2177_217744

theorem probability_at_least_one_diamond_or_joker :
  let total_cards : ℕ := 60
  let diamond_cards : ℕ := 15
  let joker_cards : ℕ := 6
  let favorable_cards : ℕ := diamond_cards + joker_cards
  let prob_not_favorable : ℚ := (total_cards - favorable_cards) / total_cards
  let prob_neither_favorable : ℚ := prob_not_favorable * prob_not_favorable
  prob_neither_favorable = 169 / 400 →
  1 - prob_neither_favorable = 231 / 400 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_diamond_or_joker_l2177_217744


namespace NUMINAMATH_CALUDE_octal_subtraction_correct_l2177_217772

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def octalToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToOctal (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else go (m / 8) ((m % 8) :: acc)
  go n []

/-- The main theorem stating the correctness of the octal subtraction. -/
theorem octal_subtraction_correct :
  let a := octalToNat [2, 1, 0, 1]
  let b := octalToNat [1, 2, 4, 5]
  natToOctal (a - b) = [0, 6, 3, 4] := by sorry

end NUMINAMATH_CALUDE_octal_subtraction_correct_l2177_217772


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l2177_217791

/-- Represents the number of branches in Professor Fernando's tree after n weeks -/
def tree_branches : ℕ → ℕ
  | 0 => 0  -- No branches before the tree starts growing
  | 1 => 1  -- One branch in the first week
  | 2 => 1  -- Still one branch in the second week
  | n + 3 => tree_branches (n + 1) + tree_branches (n + 2)  -- Fibonacci recurrence for subsequent weeks

theorem tree_growth_theorem :
  (tree_branches 6 = 8) ∧ 
  (tree_branches 7 = 13) ∧ 
  (tree_branches 13 = 233) := by
sorry

#eval tree_branches 6  -- Expected: 8
#eval tree_branches 7  -- Expected: 13
#eval tree_branches 13  -- Expected: 233

end NUMINAMATH_CALUDE_tree_growth_theorem_l2177_217791


namespace NUMINAMATH_CALUDE_triangle_configuration_theorem_l2177_217775

/-- A configuration of wire triangles in space. -/
structure TriangleConfiguration where
  /-- The number of wire triangles. -/
  k : ℕ
  /-- The number of triangles converging at each vertex. -/
  p : ℕ
  /-- Each pair of triangles has exactly one common vertex. -/
  one_common_vertex : True
  /-- At each vertex, the same number p of triangles converge. -/
  p_triangles_at_vertex : True

/-- The theorem stating the possible configurations of wire triangles. -/
theorem triangle_configuration_theorem (config : TriangleConfiguration) :
  (config.k = 1 ∧ config.p = 1) ∨ (config.k = 4 ∧ config.p = 2) ∨ (config.k = 7 ∧ config.p = 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_configuration_theorem_l2177_217775


namespace NUMINAMATH_CALUDE_conic_is_parabola_l2177_217770

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + y^2)

-- Define what it means for an equation to describe a parabola
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ) (h : a ≠ 0), 
    ∀ x y, f x y ↔ y = a * x^2 + b * x + c ∨ x = a * y^2 + b * y + d

-- Theorem statement
theorem conic_is_parabola : is_parabola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_parabola_l2177_217770


namespace NUMINAMATH_CALUDE_system_solution_l2177_217779

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 3 ∧ x - y = 1

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {pair | system pair.1 pair.2}

-- Theorem statement
theorem system_solution :
  solution_set = {(2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2177_217779


namespace NUMINAMATH_CALUDE_least_multiple_of_next_three_primes_after_5_l2177_217732

def next_three_primes_after_5 : List Nat := [7, 11, 13]

theorem least_multiple_of_next_three_primes_after_5 :
  (∀ p ∈ next_three_primes_after_5, Nat.Prime p) →
  (∀ p ∈ next_three_primes_after_5, p > 5) →
  (∀ n < 1001, ∃ p ∈ next_three_primes_after_5, ¬(p ∣ n)) →
  (∀ p ∈ next_three_primes_after_5, p ∣ 1001) :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_of_next_three_primes_after_5_l2177_217732


namespace NUMINAMATH_CALUDE_harry_apples_l2177_217756

/-- The number of apples Harry ends up with after buying more -/
def final_apples (initial : ℕ) (bought : ℕ) : ℕ := initial + bought

/-- Theorem: Harry ends up with 84 apples -/
theorem harry_apples : final_apples 79 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l2177_217756


namespace NUMINAMATH_CALUDE_trapezoid_area_is_147_l2177_217717

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD with intersection point E of diagonals -/
structure Trapezoid :=
  (A B C D E : Point)

/-- The area of a triangle -/
def triangle_area (p1 p2 p3 : Point) : ℝ := sorry

/-- The area of a trapezoid -/
def trapezoid_area (t : Trapezoid) : ℝ := sorry

/-- Theorem: Area of trapezoid ABCD is 147 square units -/
theorem trapezoid_area_is_147 (ABCD : Trapezoid) :
  (ABCD.A.x - ABCD.B.x) * (ABCD.C.y - ABCD.D.y) = (ABCD.C.x - ABCD.D.x) * (ABCD.A.y - ABCD.B.y) →
  triangle_area ABCD.A ABCD.B ABCD.E = 75 →
  triangle_area ABCD.A ABCD.D ABCD.E = 30 →
  trapezoid_area ABCD = 147 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_147_l2177_217717


namespace NUMINAMATH_CALUDE_swimmer_time_proof_l2177_217776

/-- Proves that a swimmer takes 3 hours for both downstream and upstream swims given specific conditions -/
theorem swimmer_time_proof (downstream_distance upstream_distance still_water_speed : ℝ) 
  (h1 : downstream_distance = 18)
  (h2 : upstream_distance = 12)
  (h3 : still_water_speed = 5)
  (h4 : downstream_distance / (still_water_speed + (downstream_distance - upstream_distance) / 6) = 
        upstream_distance / (still_water_speed - (downstream_distance - upstream_distance) / 6)) :
  downstream_distance / (still_water_speed + (downstream_distance - upstream_distance) / 6) = 3 ∧
  upstream_distance / (still_water_speed - (downstream_distance - upstream_distance) / 6) = 3 := by
  sorry

#check swimmer_time_proof

end NUMINAMATH_CALUDE_swimmer_time_proof_l2177_217776


namespace NUMINAMATH_CALUDE_weight_of_B_l2177_217712

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l2177_217712


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2177_217740

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y * f (2 * x) - x * f (2 * y) = 8 * x * y * (x^2 - y^2)

/-- The theorem stating that any function satisfying the functional equation
    has the form f(x) = x³ + cx for some constant c -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2177_217740


namespace NUMINAMATH_CALUDE_reroll_one_die_probability_l2177_217735

def dice_sum (d1 d2 d3 : Nat) : Nat := d1 + d2 + d3

def is_valid_die (d : Nat) : Prop := 1 ≤ d ∧ d ≤ 6

def reroll_one_probability : ℚ :=
  let total_outcomes : Nat := 6^3
  let favorable_outcomes : Nat := 19 * 6
  favorable_outcomes / total_outcomes

theorem reroll_one_die_probability :
  ∀ (d1 d2 d3 : Nat),
    is_valid_die d1 → is_valid_die d2 → is_valid_die d3 →
    (∃ (r : Nat), is_valid_die r ∧ dice_sum d1 d2 r = 9 ∨
                  dice_sum d1 r d3 = 9 ∨
                  dice_sum r d2 d3 = 9) →
    reroll_one_probability = 19/216 :=
by sorry

end NUMINAMATH_CALUDE_reroll_one_die_probability_l2177_217735


namespace NUMINAMATH_CALUDE_horner_method_v3_l2177_217762

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v1 (x : ℝ) : ℝ := 3*x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 6

def horner_v3 (x : ℝ) : ℝ := horner_v2 x * x + 79

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l2177_217762


namespace NUMINAMATH_CALUDE_x_squared_eq_two_is_quadratic_l2177_217700

/-- Definition of a quadratic equation in x -/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 2 -/
def f (x : ℝ) : ℝ := x^2 - 2

/-- Theorem: x^2 = 2 is a quadratic equation in x -/
theorem x_squared_eq_two_is_quadratic : is_quadratic_in_x f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_two_is_quadratic_l2177_217700


namespace NUMINAMATH_CALUDE_box_volume_example_l2177_217797

/-- The volume of a rectangular box -/
def box_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a box with width 9 cm, length 4 cm, and height 7 cm is 252 cm³ -/
theorem box_volume_example : box_volume 9 4 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_example_l2177_217797


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l2177_217780

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + x - 2 = 0) ∧ (a * 1^2 + 1 - 2 = 0) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l2177_217780


namespace NUMINAMATH_CALUDE_train_length_l2177_217725

/-- The length of a train given its crossing times over a post and a platform -/
theorem train_length (post_time : ℝ) (platform_length platform_time : ℝ) : 
  post_time = 10 →
  platform_length = 150 →
  platform_time = 20 →
  ∃ (train_length : ℝ), 
    train_length / post_time = (train_length + platform_length) / platform_time ∧
    train_length = 150 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2177_217725


namespace NUMINAMATH_CALUDE_sine_cosine_2009_sum_l2177_217751

theorem sine_cosine_2009_sum (α : Real) : 
  let A : Set Real := {Real.sin α, Real.cos α, 1}
  let B : Set Real := {(Real.sin α)^2, Real.sin α + Real.cos α, 0}
  A = B → (Real.sin α)^2009 + (Real.cos α)^2009 = -1 := by
sorry

end NUMINAMATH_CALUDE_sine_cosine_2009_sum_l2177_217751


namespace NUMINAMATH_CALUDE_unique_solution_trig_equation_l2177_217729

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  (Real.tan ((150 : ℝ) - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) ∧
  x = 120 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_trig_equation_l2177_217729


namespace NUMINAMATH_CALUDE_range_of_3a_plus_2b_l2177_217737

theorem range_of_3a_plus_2b (a b : ℝ) (h : a^2 + b^2 = 4) :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) ∧ 
  x = 3*a + 2*b ∧ 
  ∀ (y : ℝ), y = 3*a + 2*b → y ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_plus_2b_l2177_217737


namespace NUMINAMATH_CALUDE_fraction_equality_l2177_217796

theorem fraction_equality : (45 : ℚ) / (7 - 3 / 4) = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2177_217796


namespace NUMINAMATH_CALUDE_special_function_characterization_l2177_217771

/-- A function f: ℝ² → ℝ satisfying specific conditions -/
def SpecialFunction (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f x y = f y x) ∧
  (∀ x y z : ℝ, (f x y - f y z) * (f y z - f z x) * (f z x - f x y) = 0) ∧
  (∀ x y a : ℝ, f (x + a) (y + a) = f x y + a) ∧
  (∀ x y : ℝ, x ≤ y → f 0 x ≤ f 0 y)

/-- The theorem stating that any SpecialFunction must be of a specific form -/
theorem special_function_characterization (f : ℝ → ℝ → ℝ) (hf : SpecialFunction f) :
  ∃ a : ℝ, (∀ x y : ℝ, f x y = a + min x y) ∨ (∀ x y : ℝ, f x y = a + max x y) := by
  sorry

end NUMINAMATH_CALUDE_special_function_characterization_l2177_217771


namespace NUMINAMATH_CALUDE_sqrt_two_not_periodic_l2177_217701

-- Define a property for numbers with periodic decimal expansions
def has_periodic_decimal_expansion (x : ℝ) : Prop := sorry

-- State that numbers with periodic decimal expansions are rational
axiom periodic_is_rational : ∀ x : ℝ, has_periodic_decimal_expansion x → ∃ q : ℚ, x = q

-- State that √2 is irrational
axiom sqrt_two_irrational : ∀ q : ℚ, q * q ≠ 2

-- Theorem: √2 does not have a periodic decimal expansion
theorem sqrt_two_not_periodic : ¬ has_periodic_decimal_expansion (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_not_periodic_l2177_217701


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2177_217709

/-- Triangle inequality theorems -/
theorem triangle_inequalities (a b c : ℝ) (S : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0) : 
  (a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S) ∧ 
  (a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 ≥ 4 * Real.sqrt 3 * S) ∧
  ((a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S ∨ 
    a^2 + b^2 + c^2 - (a-b)^2 - (b-c)^2 - (c-a)^2 = 4 * Real.sqrt 3 * S) ↔ 
   a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2177_217709


namespace NUMINAMATH_CALUDE_average_weight_a_b_l2177_217745

-- Define the weights of a, b, and c
variable (a b c : ℝ)

-- Define the conditions
variable (h1 : (a + b + c) / 3 = 45)
variable (h2 : (b + c) / 2 = 45)
variable (h3 : b = 35)

-- Theorem statement
theorem average_weight_a_b : (a + b) / 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_a_b_l2177_217745


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2177_217761

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- Theorem: If (1,1), (-1,0), and (2,k) are collinear, then k = 3/2 -/
theorem collinear_points_k_value :
  collinear 1 1 (-1) 0 2 k → k = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2177_217761


namespace NUMINAMATH_CALUDE_duck_percentage_among_non_herons_l2177_217750

theorem duck_percentage_among_non_herons (total : ℝ) (geese swan heron duck : ℝ) :
  geese = 0.28 * total →
  swan = 0.20 * total →
  heron = 0.15 * total →
  duck = 0.32 * total →
  (duck / (total - heron)) * 100 = 37.6 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_percentage_among_non_herons_l2177_217750


namespace NUMINAMATH_CALUDE_container_dimensions_l2177_217708

theorem container_dimensions (a b c : ℝ) :
  a * b * 16 = 2400 →
  a * c * 10 = 2400 →
  b * c * 9.6 = 2400 →
  a = 12 ∧ b = 12.5 ∧ c = 20 := by
sorry

end NUMINAMATH_CALUDE_container_dimensions_l2177_217708


namespace NUMINAMATH_CALUDE_problem_one_l2177_217736

theorem problem_one : (1 / 3)⁻¹ + Real.sqrt 12 - |Real.sqrt 3 - 2| - (π - 2023)^0 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_l2177_217736
