import Mathlib

namespace NUMINAMATH_CALUDE_set_operation_result_l2469_246969

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 2, 3}
def N : Finset Nat := {3, 4, 5}

theorem set_operation_result : 
  (U \ M) ∩ N = {4, 5} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l2469_246969


namespace NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l2469_246929

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A line lies on a plane -/
def lineOnPlane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Two lines determine a unique plane -/
theorem two_intersecting_lines_determine_plane (l1 l2 : Line3D) :
  intersect l1 l2 → ∃! p : Plane3D, lineOnPlane l1 p ∧ lineOnPlane l2 p :=
sorry

end NUMINAMATH_CALUDE_two_intersecting_lines_determine_plane_l2469_246929


namespace NUMINAMATH_CALUDE_b_can_complete_in_27_days_l2469_246993

/-- The number of days A needs to complete the entire work -/
def a_total_days : ℕ := 15

/-- The number of days A actually works -/
def a_worked_days : ℕ := 5

/-- The number of days B needs to complete the remaining work after A leaves -/
def b_remaining_days : ℕ := 18

/-- The fraction of work completed by A -/
def a_work_fraction : ℚ := a_worked_days / a_total_days

/-- The fraction of work completed by B -/
def b_work_fraction : ℚ := 1 - a_work_fraction

/-- The number of days B needs to complete the entire work alone -/
def b_total_days : ℚ := b_remaining_days / b_work_fraction

theorem b_can_complete_in_27_days : b_total_days = 27 := by
  sorry

end NUMINAMATH_CALUDE_b_can_complete_in_27_days_l2469_246993


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l2469_246904

theorem sqrt_equality_implies_specific_integers :
  ∀ a b : ℕ+,
  a < b →
  Real.sqrt (1 + Real.sqrt (33 + 16 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 17 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l2469_246904


namespace NUMINAMATH_CALUDE_camp_III_selected_count_l2469_246971

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  totalStudents : Nat
  sampleSize : Nat
  startNumber : Nat
  campIIIStart : Nat
  campIIIEnd : Nat

/-- Calculates the number of students selected from Camp III in a systematic sample -/
def countCampIIISelected (s : SystematicSample) : Nat :=
  let interval := s.totalStudents / s.sampleSize
  let firstCampIII := s.startNumber + interval * ((s.campIIIStart - s.startNumber + interval - 1) / interval)
  let lastSelected := s.startNumber + interval * (s.sampleSize - 1)
  if firstCampIII > s.campIIIEnd then 0
  else ((min lastSelected s.campIIIEnd) - firstCampIII) / interval + 1

theorem camp_III_selected_count (s : SystematicSample) 
  (h1 : s.totalStudents = 600) 
  (h2 : s.sampleSize = 50) 
  (h3 : s.startNumber = 3) 
  (h4 : s.campIIIStart = 496) 
  (h5 : s.campIIIEnd = 600) : 
  countCampIIISelected s = 8 := by
  sorry

end NUMINAMATH_CALUDE_camp_III_selected_count_l2469_246971


namespace NUMINAMATH_CALUDE_job_completion_time_l2469_246956

/-- If m men can do a job in d days, and n men can do a different job in k days,
    then m+n men can do both jobs in (m * d + n * k) / (m + n) days. -/
theorem job_completion_time
  (m n d k : ℕ) (hm : m > 0) (hn : n > 0) (hd : d > 0) (hk : k > 0) :
  let total_time := (m * d + n * k) / (m + n)
  ∃ (time : ℚ), time = total_time ∧ time > 0 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2469_246956


namespace NUMINAMATH_CALUDE_equation_solution_unique_l2469_246945

theorem equation_solution_unique :
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 3*x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_equation_solution_unique_l2469_246945


namespace NUMINAMATH_CALUDE_children_group_size_l2469_246992

theorem children_group_size (adults_per_group : ℕ) (total_adults : ℕ) (total_children : ℕ) :
  adults_per_group = 17 →
  total_adults = 255 →
  total_children = total_adults →
  total_adults % adults_per_group = 0 →
  ∃ (children_per_group : ℕ),
    children_per_group > 0 ∧
    total_children % children_per_group = 0 ∧
    total_children / children_per_group = total_adults / adults_per_group ∧
    children_per_group = 17 := by
  sorry

end NUMINAMATH_CALUDE_children_group_size_l2469_246992


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l2469_246946

theorem sarahs_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 40 →
  (sarah_score + greg_score) / 2 = 102 →
  sarah_score = 122 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l2469_246946


namespace NUMINAMATH_CALUDE_problem_solution_l2469_246951

theorem problem_solution (x y : ℝ) (h1 : x / y = 15 / 5) (h2 : y = 25) : x = 75 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2469_246951


namespace NUMINAMATH_CALUDE_katies_old_friends_games_l2469_246948

theorem katies_old_friends_games 
  (total_friends_games : ℕ) 
  (new_friends_games : ℕ) 
  (h1 : total_friends_games = 141) 
  (h2 : new_friends_games = 88) : 
  total_friends_games - new_friends_games = 53 := by
sorry

end NUMINAMATH_CALUDE_katies_old_friends_games_l2469_246948


namespace NUMINAMATH_CALUDE_probability_red_card_equal_suits_l2469_246949

structure Deck :=
  (total_cards : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = (red_suits + black_suits) * cards_per_suit)
  (h_equal_suits : red_suits = black_suits)

def probability_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit : ℚ) / d.total_cards

theorem probability_red_card_equal_suits (d : Deck) :
  probability_red_card d = 1 :=
sorry

end NUMINAMATH_CALUDE_probability_red_card_equal_suits_l2469_246949


namespace NUMINAMATH_CALUDE_ln_plus_x_eq_three_solution_exists_in_two_three_l2469_246976

open Real

theorem ln_plus_x_eq_three_solution_exists_in_two_three :
  ∃! x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ln_plus_x_eq_three_solution_exists_in_two_three_l2469_246976


namespace NUMINAMATH_CALUDE_play_recording_distribution_l2469_246958

theorem play_recording_distribution (play_duration : ℕ) (disc_capacity : ℕ) 
  (h1 : play_duration = 385)
  (h2 : disc_capacity = 75) : 
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * disc_capacity ≥ play_duration ∧
    (num_discs - 1) * disc_capacity < play_duration ∧
    play_duration / num_discs = 64 := by
  sorry

end NUMINAMATH_CALUDE_play_recording_distribution_l2469_246958


namespace NUMINAMATH_CALUDE_min_saltwater_animals_is_1136_l2469_246970

/-- The minimum number of saltwater animals Tyler has -/
def min_saltwater_animals : ℕ :=
  let freshwater_aquariums : ℕ := 52
  let full_freshwater_aquariums : ℕ := 38
  let animals_per_full_freshwater : ℕ := 64
  let total_freshwater_animals : ℕ := 6310
  let saltwater_aquariums : ℕ := 28
  let full_saltwater_aquariums : ℕ := 18
  let animals_per_full_saltwater : ℕ := 52
  let min_animals_per_saltwater : ℕ := 20
  
  let full_saltwater_animals : ℕ := full_saltwater_aquariums * animals_per_full_saltwater
  let min_remaining_saltwater_animals : ℕ := (saltwater_aquariums - full_saltwater_aquariums) * min_animals_per_saltwater
  
  full_saltwater_animals + min_remaining_saltwater_animals

theorem min_saltwater_animals_is_1136 : min_saltwater_animals = 1136 := by
  sorry

end NUMINAMATH_CALUDE_min_saltwater_animals_is_1136_l2469_246970


namespace NUMINAMATH_CALUDE_distance_between_trees_l2469_246922

/-- Proves that in a yard of given length with a given number of trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is as calculated. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 250)
  (h2 : num_trees = 51)
  (h3 : num_trees ≥ 2) :
  yard_length / (num_trees - 1) = 5 :=
sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2469_246922


namespace NUMINAMATH_CALUDE_only_translation_preserves_pattern_l2469_246966

-- Define the pattern
structure Pattern where
  square_side : ℝ
  triangle_hypotenuse : ℝ
  line_segments_length : ℝ
  total_length : ℝ
  triangle_faces_away : Bool

-- Define the line and the repeating pattern
def infinite_line_with_pattern : Pattern :=
  { square_side := 1
  , triangle_hypotenuse := 1
  , line_segments_length := 2
  , total_length := 4
  , triangle_faces_away := true
  }

-- Define the rigid motion transformations
inductive RigidMotion
  | Rotation (center : ℝ × ℝ) (angle : ℝ)
  | Translation (distance : ℝ)
  | ReflectionAcross
  | ReflectionPerpendicular (point : ℝ)

-- Theorem statement
theorem only_translation_preserves_pattern :
  ∀ (motion : RigidMotion),
    (∃ (k : ℤ), motion = RigidMotion.Translation (↑k * infinite_line_with_pattern.total_length)) ↔
    (motion ≠ RigidMotion.ReflectionAcross ∧
     (∀ (center : ℝ × ℝ) (angle : ℝ), motion ≠ RigidMotion.Rotation center angle) ∧
     (∀ (point : ℝ), motion ≠ RigidMotion.ReflectionPerpendicular point) ∧
     (∃ (distance : ℝ), motion = RigidMotion.Translation distance ∧
        distance = ↑k * infinite_line_with_pattern.total_length)) :=
by sorry

end NUMINAMATH_CALUDE_only_translation_preserves_pattern_l2469_246966


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2469_246926

variable (w z : ℂ)

theorem complex_magnitude_problem (h1 : w * z = 24 - 10 * I) (h2 : Complex.abs w = Real.sqrt 34) :
  Complex.abs z = (13 * Real.sqrt 34) / 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2469_246926


namespace NUMINAMATH_CALUDE_carl_first_six_probability_l2469_246991

/-- The probability of rolling a 6 on a single die roll -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a single die roll -/
def prob_not_six : ℚ := 1 - prob_six

/-- The sequence of probabilities for Carl rolling the first 6 on his nth turn -/
def carl_first_six (n : ℕ) : ℚ := (prob_not_six ^ (3 * n - 1)) * prob_six

/-- The sum of the geometric series representing the probability of Carl rolling the first 6 -/
def probability_carl_first_six : ℚ := (carl_first_six 1) / (1 - (prob_not_six ^ 3))

theorem carl_first_six_probability :
  probability_carl_first_six = 25 / 91 :=
sorry

end NUMINAMATH_CALUDE_carl_first_six_probability_l2469_246991


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l2469_246936

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a vertical shift to a quadratic function -/
def verticalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b, c := f.c + shift }

/-- Applies a horizontal shift to a quadratic function -/
def horizontalShift (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := 2 * f.a * shift + f.b, c := f.a * shift^2 + f.b * shift + f.c }

/-- The main theorem stating that shifting y = -2x^2 down 3 units and left 1 unit 
    results in y = -2(x + 1)^2 - 3 -/
theorem quadratic_shift_theorem :
  let f : QuadraticFunction := { a := -2, b := 0, c := 0 }
  let shifted := horizontalShift (verticalShift f (-3)) (-1)
  shifted.a = -2 ∧ shifted.b = 4 ∧ shifted.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l2469_246936


namespace NUMINAMATH_CALUDE_q_prime_div_p_prime_eq_550_l2469_246975

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def distinct_numbers : ℕ := 12

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing all five slips with the same number -/
def p' : ℚ := 12 / (total_slips.choose drawn_slips)

/-- The probability of drawing three slips with one number and two with another -/
def q' : ℚ := (6600 : ℚ) / (total_slips.choose drawn_slips)

/-- The main theorem stating the ratio of q' to p' -/
theorem q_prime_div_p_prime_eq_550 : q' / p' = 550 := by sorry

end NUMINAMATH_CALUDE_q_prime_div_p_prime_eq_550_l2469_246975


namespace NUMINAMATH_CALUDE_no_true_propositions_l2469_246981

theorem no_true_propositions : 
  let prop1 := ∀ x : ℝ, x^2 - 3*x + 2 = 0
  let prop2 := ∃ x : ℚ, x^2 = 2
  let prop3 := ∃ x : ℝ, x^2 + 1 = 0
  let prop4 := ∀ x : ℝ, 4*x^2 > 2*x - 1 + 3*x^2
  ¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4 :=
by
  sorry

#check no_true_propositions

end NUMINAMATH_CALUDE_no_true_propositions_l2469_246981


namespace NUMINAMATH_CALUDE_triangle_construction_l2469_246937

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (α β γ : ℝ)
  (hγ : γ = 2 * π / 3)  -- 120° in radians
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : α + β + γ = π)

-- Define the new triangles
structure NewTriangle :=
  (x y z : ℝ)
  (θ φ ψ : ℝ)
  (h_triangle : x + y > z ∧ y + z > x ∧ z + x > y)
  (h_angles : θ + φ + ψ = π)

-- Statement of the theorem
theorem triangle_construction (abc : Triangle) :
  ∃ (t1 t2 : NewTriangle),
    -- First new triangle
    (t1.x = abc.a ∧ t1.y = abc.c ∧ t1.z = abc.a + abc.b) ∧
    (t1.θ = π / 3 ∧ t1.φ = abc.α ∧ t1.ψ = π / 3 + abc.β) ∧
    -- Second new triangle
    (t2.x = abc.b ∧ t2.y = abc.c ∧ t2.z = abc.a + abc.b) ∧
    (t2.θ = π / 3 ∧ t2.φ = abc.β ∧ t2.ψ = π / 3 + abc.α) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_l2469_246937


namespace NUMINAMATH_CALUDE_oliver_quarters_problem_l2469_246905

theorem oliver_quarters_problem (initial_cash : ℝ) (quarters_given : ℕ) (final_amount : ℝ) :
  initial_cash = 40 →
  quarters_given = 120 →
  final_amount = 55 →
  ∃ (Q : ℕ), 
    (initial_cash + 0.25 * Q) - (5 + 0.25 * quarters_given) = final_amount ∧
    Q = 200 :=
by sorry

end NUMINAMATH_CALUDE_oliver_quarters_problem_l2469_246905


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l2469_246998

/-- The sum of the infinite series ∑(1 / (n(n+3))) for n from 1 to infinity is equal to 7/9. -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 3)) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l2469_246998


namespace NUMINAMATH_CALUDE_salt_from_seawater_l2469_246943

/-- Calculates the amount of salt obtained from seawater after evaporation -/
def salt_after_evaporation (volume : ℝ) (salt_concentration : ℝ) : ℝ :=
  volume * 1000 * salt_concentration

/-- Theorem: 2 liters of seawater with 20% salt concentration yields 400 ml of salt after evaporation -/
theorem salt_from_seawater :
  salt_after_evaporation 2 0.2 = 400 := by
  sorry

#eval salt_after_evaporation 2 0.2

end NUMINAMATH_CALUDE_salt_from_seawater_l2469_246943


namespace NUMINAMATH_CALUDE_revenue_division_l2469_246923

theorem revenue_division (total_revenue : ℝ) (ratio_sum : ℕ) (salary_ratio rent_ratio marketing_ratio : ℕ) :
  total_revenue = 10000 →
  ratio_sum = 3 + 5 + 2 + 7 →
  salary_ratio = 3 →
  rent_ratio = 2 →
  marketing_ratio = 7 →
  (salary_ratio + rent_ratio + marketing_ratio) * (total_revenue / ratio_sum) = 7058.88 := by
  sorry

end NUMINAMATH_CALUDE_revenue_division_l2469_246923


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2469_246989

theorem quadratic_root_relation (p q : ℝ) :
  (∃ α : ℝ, (α^2 + p*α + q = 0) ∧ ((2*α)^2 + p*(2*α) + q = 0)) →
  2*p^2 = 9*q :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2469_246989


namespace NUMINAMATH_CALUDE_birdhouse_wood_pieces_l2469_246924

/-- The number of pieces of wood used for each birdhouse -/
def wood_pieces : ℕ := sorry

/-- The cost of wood per piece in dollars -/
def cost_per_piece : ℚ := 3/2

/-- The profit made on each birdhouse in dollars -/
def profit_per_birdhouse : ℚ := 11/2

/-- The total price for two birdhouses in dollars -/
def price_for_two : ℚ := 32

theorem birdhouse_wood_pieces :
  (2 * ((wood_pieces : ℚ) * cost_per_piece + profit_per_birdhouse) = price_for_two) →
  wood_pieces = 7 := by sorry

end NUMINAMATH_CALUDE_birdhouse_wood_pieces_l2469_246924


namespace NUMINAMATH_CALUDE_coffee_cream_ratio_l2469_246942

/-- Represents the amount of coffee and cream in a cup -/
structure Coffee :=
  (coffee : ℚ)
  (cream : ℚ)

/-- Calculates the ratio of cream in two coffees -/
def creamRatio (c1 c2 : Coffee) : ℚ :=
  c1.cream / c2.cream

theorem coffee_cream_ratio :
  let max_initial := Coffee.mk 14 0
  let maxine_initial := Coffee.mk 16 0
  let max_after_drinking := Coffee.mk (max_initial.coffee - 4) 0
  let max_final := Coffee.mk max_after_drinking.coffee 3
  let maxine_with_cream := Coffee.mk maxine_initial.coffee 3
  let maxine_final := Coffee.mk (maxine_with_cream.coffee * 14 / 19) (maxine_with_cream.cream * 14 / 19)
  creamRatio max_final maxine_final = 19 / 14 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cream_ratio_l2469_246942


namespace NUMINAMATH_CALUDE_divisibility_of_p_and_q_l2469_246933

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def p (n : ℕ) : ℕ := (ones n) * (10^(3*n) + 9*10^(2*n) + 8*10^n + 7)

def q (n : ℕ) : ℕ := (ones (n+1)) * (10^(3*(n+1)) + 9*10^(2*(n+1)) + 8*10^(n+1) + 7)

theorem divisibility_of_p_and_q (n : ℕ) (h : (1987 : ℕ) ∣ ones n) :
  (1987 : ℕ) ∣ p n ∧ (1987 : ℕ) ∣ q n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_p_and_q_l2469_246933


namespace NUMINAMATH_CALUDE_school_sections_l2469_246963

/-- The number of sections formed when dividing boys and girls into equal groups -/
def total_sections (num_boys num_girls : ℕ) : ℕ :=
  (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls))

/-- Theorem stating that for 408 boys and 192 girls, the total number of sections is 25 -/
theorem school_sections : total_sections 408 192 = 25 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_l2469_246963


namespace NUMINAMATH_CALUDE_pool_width_is_40_l2469_246925

/-- Represents a rectangular pool with given length and width -/
structure Pool where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangular pool -/
def Pool.perimeter (p : Pool) : ℝ := 2 * (p.length + p.width)

/-- Represents the speeds of Ruth and Sarah -/
structure Speeds where
  ruth : ℝ
  sarah : ℝ

theorem pool_width_is_40 (p : Pool) (s : Speeds) : p.width = 40 :=
  by
  have h1 : p.length = 50 := by sorry
  have h2 : s.ruth = 3 * s.sarah := by sorry
  have h3 : 6 * p.length = 5 * p.perimeter := by sorry
  sorry

end NUMINAMATH_CALUDE_pool_width_is_40_l2469_246925


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l2469_246916

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l2469_246916


namespace NUMINAMATH_CALUDE_evaluate_expression_l2469_246902

theorem evaluate_expression : (49^2 - 35^2) + (15^2 - 9^2) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2469_246902


namespace NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l2469_246950

theorem arithmetic_geometric_harmonic_mean_sum_of_squares
  (x y z : ℝ)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_harmonic_mean_sum_of_squares_l2469_246950


namespace NUMINAMATH_CALUDE_weed_ratio_l2469_246940

/-- Represents the number of weeds pulled on each day -/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℚ
  friday : ℚ

/-- The problem of Sarah's weed pulling -/
def weed_problem (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.wednesday = 3 * w.tuesday ∧
  w.thursday = (1 : ℚ) / 5 * w.wednesday ∧
  w.friday = w.thursday - 10 ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- The theorem stating the ratio of weeds pulled on Thursday to Wednesday -/
theorem weed_ratio (w : WeedCount) (h : weed_problem w) : 
  w.thursday / w.wednesday = (1 : ℚ) / 5 := by
  sorry


end NUMINAMATH_CALUDE_weed_ratio_l2469_246940


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2469_246995

def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_of_M_and_N :
  M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2469_246995


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2469_246959

/-- An ellipse passing through (1,0) with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  eq : 1 / a^2 + 0 / b^2 = 1

/-- A parabola with focus at (1,0) and vertex at (m,0) -/
structure Parabola where
  m : ℝ

/-- The theorem statement -/
theorem ellipse_parabola_intersection (ε : Ellipse) (ρ : Parabola) :
  let e := Real.sqrt (1 - ε.b^2)
  (Real.sqrt (2/3) < e ∧ e < 1) →
  (1 < ρ.m ∧ ρ.m < (3 + Real.sqrt 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l2469_246959


namespace NUMINAMATH_CALUDE_mark_households_visited_mark_collection_proof_l2469_246932

theorem mark_households_visited (days : ℕ) (total_collected : ℕ) (donation : ℕ) : ℕ :=
  let households_per_day := 20
  have days_collecting := 5
  have half_households_donate := households_per_day / 2
  have donation_amount := 2 * 20
  have total_collected_calculated := days_collecting * half_households_donate * donation_amount
  households_per_day

theorem mark_collection_proof 
  (days : ℕ) 
  (total_collected : ℕ) 
  (donation : ℕ) 
  (h1 : days = 5) 
  (h2 : donation = 2 * 20) 
  (h3 : total_collected = 2000) :
  mark_households_visited days total_collected donation = 20 := by
  sorry

end NUMINAMATH_CALUDE_mark_households_visited_mark_collection_proof_l2469_246932


namespace NUMINAMATH_CALUDE_round_trip_distance_solve_specific_problem_l2469_246987

/-- Calculates the one-way distance of a round trip given the speeds and total time -/
theorem round_trip_distance 
  (speed_to : ℝ) 
  (speed_from : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_to > 0)
  (h2 : speed_from > 0)
  (h3 : total_time > 0) :
  ∃ (distance : ℝ), 
    distance > 0 ∧ 
    (distance / speed_to + distance / speed_from = total_time) := by
  sorry

/-- Solves the specific problem with given values -/
theorem solve_specific_problem :
  ∃ (distance : ℝ), 
    distance > 0 ∧ 
    (distance / 50 + distance / 75 = 10) ∧
    distance = 300 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_solve_specific_problem_l2469_246987


namespace NUMINAMATH_CALUDE_distinct_roots_imply_distinct_roots_l2469_246973

theorem distinct_roots_imply_distinct_roots (p q : ℝ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (h1 : (p^2 - 4*q) > 0) 
  (h2 : (q^2 - 4*p) > 0) : 
  ((p + q)^2 - 8*(p + q)) > 0 := by
sorry


end NUMINAMATH_CALUDE_distinct_roots_imply_distinct_roots_l2469_246973


namespace NUMINAMATH_CALUDE_sign_sum_theorem_l2469_246947

theorem sign_sum_theorem (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) :
  (a / |a| + b / |b| + c / |c| - (a * b * c) / |a * b * c|) = 2 ∨
  (a / |a| + b / |b| + c / |c| - (a * b * c) / |a * b * c|) = -2 :=
by sorry

end NUMINAMATH_CALUDE_sign_sum_theorem_l2469_246947


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2469_246952

theorem expand_and_simplify (x y : ℝ) : (2*x + 3*y)^2 - (2*x - 3*y)^2 = 24*x*y := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2469_246952


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2469_246996

theorem min_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 2*a + 3*b + 4*c = 120) :
  a^2 + b^2 + c^2 ≥ 14400/29 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    2*a₀ + 3*b₀ + 4*c₀ = 120 ∧ a₀^2 + b₀^2 + c₀^2 = 14400/29 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2469_246996


namespace NUMINAMATH_CALUDE_impossibleEvent_l2469_246985

/-- A fair dice with faces numbered 1 to 6 -/
def Dice : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The event of getting a number divisible by 10 when rolling the dice -/
def DivisibleBy10 (n : ℕ) : Prop := n % 10 = 0

/-- Theorem: The event of rolling a number divisible by 10 is impossible -/
theorem impossibleEvent : ∀ n ∈ Dice, ¬ DivisibleBy10 n := by
  sorry

end NUMINAMATH_CALUDE_impossibleEvent_l2469_246985


namespace NUMINAMATH_CALUDE_emilys_final_score_l2469_246912

/-- A trivia game with 5 rounds and specific scoring rules -/
def triviaGame (round1 round2 round3 round4Base round5Base lastRoundLoss : ℕ) 
               (round4Multiplier round5Multiplier : ℕ) : ℕ :=
  round1 + round2 + round3 + 
  (round4Base * round4Multiplier) + 
  (round5Base * round5Multiplier) - 
  lastRoundLoss

/-- The final score of Emily's trivia game -/
theorem emilys_final_score : 
  triviaGame 16 33 21 10 4 48 2 3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_emilys_final_score_l2469_246912


namespace NUMINAMATH_CALUDE_find_number_l2469_246901

theorem find_number : ∃ x : ℝ, 1.35 + 0.321 + x = 1.794 ∧ x = 0.123 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2469_246901


namespace NUMINAMATH_CALUDE_chocolate_division_l2469_246984

theorem chocolate_division (total : ℝ) (total_positive : 0 < total) : 
  let al_share := (4 / 10) * total
  let bert_share := (3 / 10) * total
  let carl_share := (2 / 10) * total
  let dana_share := (1 / 10) * total
  al_share + bert_share + carl_share + dana_share = total :=
by sorry

end NUMINAMATH_CALUDE_chocolate_division_l2469_246984


namespace NUMINAMATH_CALUDE_remainder_of_2743_base12_div_9_l2469_246903

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (n : ℕ) : ℕ :=
  let d0 := n % 12
  let d1 := (n / 12) % 12
  let d2 := (n / 144) % 12
  let d3 := n / 1728
  d3 * 1728 + d2 * 144 + d1 * 12 + d0

/-- The base-12 number 2743 --/
def n : ℕ := 2743

theorem remainder_of_2743_base12_div_9 :
  (base12ToBase10 n) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2743_base12_div_9_l2469_246903


namespace NUMINAMATH_CALUDE_divisible_by_six_count_and_percentage_l2469_246921

theorem divisible_by_six_count_and_percentage :
  let n : ℕ := 120
  let divisible_count : ℕ := (n / 6 : ℕ)
  divisible_count = 20 ∧ 
  (divisible_count : ℚ) / n * 100 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_count_and_percentage_l2469_246921


namespace NUMINAMATH_CALUDE_brick_width_calculation_l2469_246953

theorem brick_width_calculation (wall_length wall_width wall_height : ℝ)
                                (brick_length brick_height : ℝ)
                                (num_bricks : ℕ) :
  wall_length = 800 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 80 →
  brick_height = 6 →
  num_bricks = 2000 →
  ∃ brick_width : ℝ,
    num_bricks * (brick_length * brick_width * brick_height) = wall_length * wall_width * wall_height ∧
    brick_width = 5.625 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l2469_246953


namespace NUMINAMATH_CALUDE_eighteen_digit_divisible_by_99_l2469_246999

def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

def is_single_digit (d : ℕ) : Prop := d ≤ 9

def construct_number (x y : ℕ) : ℕ :=
  x * 10^17 + 3640548981270644 + y

theorem eighteen_digit_divisible_by_99 (x y : ℕ) :
  is_single_digit x ∧ is_single_digit y →
  (is_divisible_by_99 (construct_number x y) ↔ x = 9 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_digit_divisible_by_99_l2469_246999


namespace NUMINAMATH_CALUDE_all_functions_are_zero_l2469_246938

def is_valid_function (f : ℕ → ℕ) : Prop :=
  (∀ x y : ℕ, f (x * y) = f x + f y) ∧
  (f 30 = 0) ∧
  (∀ x : ℕ, x % 10 = 7 → f x = 0)

theorem all_functions_are_zero (f : ℕ → ℕ) (h : is_valid_function f) :
  ∀ n : ℕ, f n = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_functions_are_zero_l2469_246938


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l2469_246911

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l2469_246911


namespace NUMINAMATH_CALUDE_multiple_in_denominator_l2469_246939

theorem multiple_in_denominator (a b k : ℚ) : 
  a / b = 4 / 1 →
  (a - 3 * b) / (k * (a - b)) = 1 / 7 →
  k = 7 / 3 := by sorry

end NUMINAMATH_CALUDE_multiple_in_denominator_l2469_246939


namespace NUMINAMATH_CALUDE_hydrolysis_weight_change_l2469_246907

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Sodium in g/mol -/
def Na_weight : ℝ := 22.99

/-- Molecular weight of acetylsalicylic acid (C9H8O4) in g/mol -/
def acetylsalicylic_acid_weight : ℝ := 9 * C_weight + 8 * H_weight + 4 * O_weight

/-- Molecular weight of sodium hydroxide (NaOH) in g/mol -/
def sodium_hydroxide_weight : ℝ := Na_weight + O_weight + H_weight

/-- Molecular weight of salicylic acid (C7H6O3) in g/mol -/
def salicylic_acid_weight : ℝ := 7 * C_weight + 6 * H_weight + 3 * O_weight

/-- Molecular weight of sodium acetate (CH3COONa) in g/mol -/
def sodium_acetate_weight : ℝ := 2 * C_weight + 3 * H_weight + 2 * O_weight + Na_weight

/-- Theorem stating that the overall molecular weight change during the hydrolysis reaction is 0 g/mol -/
theorem hydrolysis_weight_change :
  acetylsalicylic_acid_weight + sodium_hydroxide_weight = salicylic_acid_weight + sodium_acetate_weight :=
by sorry

end NUMINAMATH_CALUDE_hydrolysis_weight_change_l2469_246907


namespace NUMINAMATH_CALUDE_junhyun_travel_distance_l2469_246935

/-- The distance Junhyun traveled by bus in kilometers -/
def bus_distance : ℝ := 2.6

/-- The distance Junhyun traveled by subway in kilometers -/
def subway_distance : ℝ := 5.98

/-- The total distance Junhyun traveled using public transportation -/
def total_distance : ℝ := bus_distance + subway_distance

/-- Theorem stating that the total distance Junhyun traveled is 8.58 km -/
theorem junhyun_travel_distance : total_distance = 8.58 := by sorry

end NUMINAMATH_CALUDE_junhyun_travel_distance_l2469_246935


namespace NUMINAMATH_CALUDE_marias_age_half_anns_l2469_246915

/-- Proves that Maria's age was half of Ann's age 4 years ago -/
theorem marias_age_half_anns (maria_current_age ann_current_age years_ago : ℕ) : 
  maria_current_age = 7 →
  ann_current_age = maria_current_age + 3 →
  maria_current_age - years_ago = (ann_current_age - years_ago) / 2 →
  years_ago = 4 := by
sorry

end NUMINAMATH_CALUDE_marias_age_half_anns_l2469_246915


namespace NUMINAMATH_CALUDE_association_confidence_level_l2469_246955

-- Define the χ² value
def chi_squared : ℝ := 6.825

-- Define the degrees of freedom for a 2x2 contingency table
def degrees_of_freedom : ℕ := 1

-- Define the critical value for 99% confidence level with 1 degree of freedom
def critical_value : ℝ := 6.635

-- Define the confidence level we want to prove
def target_confidence_level : ℝ := 99

-- Theorem statement
theorem association_confidence_level :
  chi_squared > critical_value →
  (∃ (confidence_level : ℝ), confidence_level ≥ target_confidence_level) :=
sorry

end NUMINAMATH_CALUDE_association_confidence_level_l2469_246955


namespace NUMINAMATH_CALUDE_sample_size_is_number_of_individuals_l2469_246900

/-- Definition of a sample in statistics -/
structure Sample (α : Type) where
  elements : List α

/-- Definition of sample size -/
def sampleSize {α : Type} (s : Sample α) : ℕ :=
  s.elements.length

/-- Theorem: The sample size is the number of individuals in the sample -/
theorem sample_size_is_number_of_individuals {α : Type} (s : Sample α) :
  sampleSize s = s.elements.length := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_number_of_individuals_l2469_246900


namespace NUMINAMATH_CALUDE_impossible_to_change_all_signs_l2469_246930

/-- Represents a point in the decagon configuration -/
structure Point where
  value : Int
  mk_point : value = 1 ∨ value = -1

/-- Represents the decagon configuration -/
structure DecagonConfig where
  points : Finset Point
  mk_config : points.card = 220

/-- Represents an operation on the decagon -/
inductive Operation
  | side : Operation
  | diagonal : Operation

/-- Applies an operation to the decagon configuration -/
def apply_operation (config : DecagonConfig) (op : Operation) : DecagonConfig :=
  sorry

/-- Checks if all points in the configuration are -1 -/
def all_negative (config : DecagonConfig) : Prop :=
  ∀ p ∈ config.points, p.value = -1

/-- Main theorem: It's impossible to change all signs to their opposites -/
theorem impossible_to_change_all_signs (initial_config : DecagonConfig) :
  ¬∃ (ops : List Operation), all_negative (ops.foldl apply_operation initial_config) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_change_all_signs_l2469_246930


namespace NUMINAMATH_CALUDE_all_lines_pass_through_fixed_point_l2469_246968

/-- The line equation passing through a fixed point for all real a -/
def line_equation (a x y : ℝ) : Prop :=
  (a + 1) * x + y - 2 - a = 0

/-- The fixed point that all lines pass through -/
def fixed_point : ℝ × ℝ := (1, 1)

/-- Theorem: All lines in the family pass through the fixed point (1, 1) -/
theorem all_lines_pass_through_fixed_point :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) :=
by
  sorry

end NUMINAMATH_CALUDE_all_lines_pass_through_fixed_point_l2469_246968


namespace NUMINAMATH_CALUDE_tyler_initial_money_l2469_246917

def scissors_cost : ℕ := 8 * 5
def erasers_cost : ℕ := 10 * 4
def remaining_money : ℕ := 20

theorem tyler_initial_money :
  scissors_cost + erasers_cost + remaining_money = 100 :=
by sorry

end NUMINAMATH_CALUDE_tyler_initial_money_l2469_246917


namespace NUMINAMATH_CALUDE_first_number_remainder_l2469_246919

/-- A permutation of numbers from 1 to 2023 -/
def Arrangement := Fin 2023 → Fin 2023

/-- Property that any three numbers with one in between have different remainders when divided by 3 -/
def ValidArrangement (arr : Arrangement) : Prop :=
  ∀ i : Fin 2020, (arr i % 3) ≠ (arr (i + 2) % 3) ∧ (arr i % 3) ≠ (arr (i + 4) % 3) ∧ (arr (i + 2) % 3) ≠ (arr (i + 4) % 3)

/-- Theorem stating that the first number in a valid arrangement must have remainder 1 when divided by 3 -/
theorem first_number_remainder (arr : Arrangement) (h : ValidArrangement arr) : arr 0 % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_number_remainder_l2469_246919


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l2469_246983

/-- Given a function f(x) = ax³ + 4x² + 3x, prove that if f'(1) = 2, then a = -3 -/
theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + 4 * x^2 + 3 * x
  let f' : ℝ → ℝ := λ x => 3 * a * x^2 + 8 * x + 3
  f' 1 = 2 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l2469_246983


namespace NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l2469_246974

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 - x) / (x - 2) + 1 / (2 - x) = 1
def equation2 (x : ℝ) : Prop := 3 / (x^2 - 4) + 2 / (x + 2) = 1 / (x - 2)

-- Theorem for equation 1
theorem no_solution_equation1 : ¬∃ x : ℝ, equation1 x := by sorry

-- Theorem for equation 2
theorem unique_solution_equation2 : ∃! x : ℝ, equation2 x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_no_solution_equation1_unique_solution_equation2_l2469_246974


namespace NUMINAMATH_CALUDE_machine_job_time_l2469_246908

theorem machine_job_time (y : ℝ) : 
  (1 / (y + 8) + 1 / (y + 3) + 1 / (1.5 * y) = 1 / y) →
  y = (-25 + Real.sqrt 421) / 6 :=
by sorry

end NUMINAMATH_CALUDE_machine_job_time_l2469_246908


namespace NUMINAMATH_CALUDE_dodgeball_tournament_l2469_246944

theorem dodgeball_tournament (N : ℕ) : 
  (∃ W D : ℕ, 
    W + D = N * (N - 1) / 2 ∧ 
    15 * W + 22 * D = 1151) → 
  N = 12 := by
sorry

end NUMINAMATH_CALUDE_dodgeball_tournament_l2469_246944


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l2469_246957

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l2469_246957


namespace NUMINAMATH_CALUDE_trajectory_equation_l2469_246977

/-- The trajectory of point M(x, y) such that the ratio of its distance from the line x = 25/4
    to its distance from the point (4, 0) is 5/4 -/
theorem trajectory_equation (x y : ℝ) :
  (|x - 25/4| / Real.sqrt ((x - 4)^2 + y^2) = 5/4) →
  (x^2 / 25 + y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2469_246977


namespace NUMINAMATH_CALUDE_average_income_Q_R_l2469_246941

/-- The average monthly income of P and Q is Rs. 5050, 
    the average monthly income of P and R is Rs. 5200, 
    and the monthly income of P is Rs. 4000. 
    Prove that the average monthly income of Q and R is Rs. 6250. -/
theorem average_income_Q_R (P Q R : ℕ) : 
  (P + Q) / 2 = 5050 → 
  (P + R) / 2 = 5200 → 
  P = 4000 → 
  (Q + R) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_Q_R_l2469_246941


namespace NUMINAMATH_CALUDE_m_range_l2469_246980

theorem m_range : ∃ m : ℝ, m = Real.sqrt 5 - 1 ∧ 1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2469_246980


namespace NUMINAMATH_CALUDE_binary_101101_conversion_l2469_246910

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, x) => acc + if x then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_101101_conversion :
  let binary := [true, false, true, true, false, true]
  (binary_to_decimal binary = 45) ∧
  (decimal_to_base7 (binary_to_decimal binary) = [6, 3]) := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_conversion_l2469_246910


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_is_eight_l2469_246961

/-- Given two distinct positive integers that are factors of 60, 
    this function returns the smallest product of these integers 
    that is not a factor of 60. -/
def smallest_non_factor_product : ℕ → ℕ → ℕ :=
  fun x y =>
    if x ≠ y ∧ x > 0 ∧ y > 0 ∧ 60 % x = 0 ∧ 60 % y = 0 ∧ 60 % (x * y) ≠ 0 then
      x * y
    else
      0

theorem smallest_non_factor_product_is_eight :
  ∀ x y : ℕ, x ≠ y → x > 0 → y > 0 → 60 % x = 0 → 60 % y = 0 → 60 % (x * y) ≠ 0 →
  smallest_non_factor_product x y ≥ 8 ∧
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ 60 % a = 0 ∧ 60 % b = 0 ∧ 60 % (a * b) ≠ 0 ∧ a * b = 8 :=
by sorry

#check smallest_non_factor_product_is_eight

end NUMINAMATH_CALUDE_smallest_non_factor_product_is_eight_l2469_246961


namespace NUMINAMATH_CALUDE_sector_area_l2469_246954

theorem sector_area (centralAngle : Real) (radius : Real) : 
  centralAngle = π / 6 → radius = 2 → (1 / 2) * centralAngle * radius^2 = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2469_246954


namespace NUMINAMATH_CALUDE_calculator_presses_to_exceed_250_l2469_246964

def calculator_function (x : ℕ) : ℕ := x^2 + 3

def iterate_calculator (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => calculator_function (iterate_calculator n m)

theorem calculator_presses_to_exceed_250 :
  ∃ n : ℕ, n > 0 ∧ iterate_calculator n 3 > 250 ∧ ∀ m : ℕ, m < n → iterate_calculator n m ≤ 250 :=
by sorry

end NUMINAMATH_CALUDE_calculator_presses_to_exceed_250_l2469_246964


namespace NUMINAMATH_CALUDE_non_egg_laying_hens_l2469_246960

/-- Proves that the number of non-egg-laying hens is 20 given the total number of chickens,
    number of roosters, and number of egg-laying hens. -/
theorem non_egg_laying_hens (total_chickens roosters egg_laying_hens : ℕ) : 
  total_chickens = 325 →
  roosters = 28 →
  egg_laying_hens = 277 →
  total_chickens - roosters - egg_laying_hens = 20 := by
sorry

end NUMINAMATH_CALUDE_non_egg_laying_hens_l2469_246960


namespace NUMINAMATH_CALUDE_system_solution_l2469_246914

theorem system_solution : ∃! (x y : ℝ), 
  (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
  (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
  x = (1 : ℝ) / 2 ∧ y = -(3 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2469_246914


namespace NUMINAMATH_CALUDE_multiply_72519_9999_l2469_246931

theorem multiply_72519_9999 : 72519 * 9999 = 724817481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_9999_l2469_246931


namespace NUMINAMATH_CALUDE_parabola_equation_l2469_246972

/-- A parabola is defined by its directrix and focus. -/
structure Parabola where
  directrix : ℝ  -- y-coordinate of the directrix
  focus : ℝ      -- y-coordinate of the focus

/-- The standard equation of a parabola. -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = -4 * p.directrix * y) ↔ (y = p.directrix ∨ (x^2 + (y - p.focus)^2 = (y - p.directrix)^2))

/-- Theorem: For a parabola with directrix y = 4, its standard equation is x² = -16y. -/
theorem parabola_equation (p : Parabola) (h : p.directrix = 4) : 
  standardEquation p ↔ ∀ x y : ℝ, x^2 = -16*y ↔ (y = 4 ∨ (x^2 + (y - p.focus)^2 = (y - 4)^2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2469_246972


namespace NUMINAMATH_CALUDE_holly_pill_ratio_l2469_246979

/-- Represents the daily pill intake for Holly --/
structure DailyPillIntake where
  insulin : ℕ
  blood_pressure : ℕ
  anticonvulsant : ℕ

/-- Calculates the total number of pills taken in a week --/
def weekly_total (d : DailyPillIntake) : ℕ :=
  7 * (d.insulin + d.blood_pressure + d.anticonvulsant)

/-- Represents the ratio of two numbers --/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem holly_pill_ratio :
  ∀ (d : DailyPillIntake),
    d.insulin = 2 →
    d.blood_pressure = 3 →
    weekly_total d = 77 →
    ∃ (r : Ratio), r.numerator = 2 ∧ r.denominator = 1 ∧
      r.numerator * d.blood_pressure = r.denominator * d.anticonvulsant :=
by sorry

end NUMINAMATH_CALUDE_holly_pill_ratio_l2469_246979


namespace NUMINAMATH_CALUDE_calculate_fraction_l2469_246965

theorem calculate_fraction : (2015^2) / (2014^2 + 2016^2 - 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_fraction_l2469_246965


namespace NUMINAMATH_CALUDE_seating_arrangements_l2469_246928

/-- The number of ways to arrange n people in a row. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with k specific people consecutive. -/
def consecutiveArrangements (n k : ℕ) : ℕ := 
  (Nat.factorial (n - k + 1)) * (Nat.factorial k)

/-- The number of people to be seated. -/
def totalPeople : ℕ := 10

/-- The number of specific individuals who refuse to sit consecutively. -/
def specificIndividuals : ℕ := 4

/-- The number of ways to arrange people with restrictions. -/
def arrangementsWithRestrictions : ℕ := 
  totalArrangements totalPeople - consecutiveArrangements totalPeople specificIndividuals

theorem seating_arrangements :
  arrangementsWithRestrictions = 3507840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2469_246928


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l2469_246920

/-- Proves that the cost price of a computer table is 6625 when the selling price is 8215 with a 24% markup -/
theorem computer_table_cost_price (selling_price : ℕ) (markup_percentage : ℕ) (cost_price : ℕ) :
  selling_price = 8215 →
  markup_percentage = 24 →
  selling_price = cost_price + (cost_price * markup_percentage) / 100 →
  cost_price = 6625 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l2469_246920


namespace NUMINAMATH_CALUDE_participation_plans_l2469_246986

/-- The number of students -/
def total_students : ℕ := 4

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The number of subjects -/
def subjects : ℕ := 3

/-- The number of students that can be freely selected -/
def free_selection : ℕ := total_students - 1

theorem participation_plans :
  (Nat.choose free_selection (selected_students - 1)) * (Nat.factorial subjects) = 18 := by
  sorry

end NUMINAMATH_CALUDE_participation_plans_l2469_246986


namespace NUMINAMATH_CALUDE_shelves_used_l2469_246982

def initial_stock : ℕ := 5
def new_shipment : ℕ := 7
def bears_per_shelf : ℕ := 6

theorem shelves_used (initial_stock new_shipment bears_per_shelf : ℕ) :
  initial_stock = 5 →
  new_shipment = 7 →
  bears_per_shelf = 6 →
  (initial_stock + new_shipment) / bears_per_shelf = 2 := by
sorry

end NUMINAMATH_CALUDE_shelves_used_l2469_246982


namespace NUMINAMATH_CALUDE_employee_gross_pay_l2469_246906

/-- Calculate the gross pay for an employee given regular and overtime hours and rates -/
def calculate_gross_pay (regular_rate : ℚ) (regular_hours : ℚ) (overtime_rate : ℚ) (overtime_hours : ℚ) : ℚ :=
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that the employee's gross pay for the week is $622 -/
theorem employee_gross_pay :
  let regular_rate : ℚ := 11.25
  let regular_hours : ℚ := 40
  let overtime_rate : ℚ := 16
  let overtime_hours : ℚ := 10.75
  calculate_gross_pay regular_rate regular_hours overtime_rate overtime_hours = 622 := by
  sorry

#eval calculate_gross_pay (11.25 : ℚ) (40 : ℚ) (16 : ℚ) (10.75 : ℚ)

end NUMINAMATH_CALUDE_employee_gross_pay_l2469_246906


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l2469_246967

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l2469_246967


namespace NUMINAMATH_CALUDE_fraction_difference_equals_two_l2469_246988

theorem fraction_difference_equals_two 
  (a b : ℝ) 
  (h1 : 2 * b = 1 + a * b) 
  (h2 : a ≠ 1) 
  (h3 : b ≠ 1) : 
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_two_l2469_246988


namespace NUMINAMATH_CALUDE_tan_alpha_half_implies_expression_equals_two_l2469_246994

theorem tan_alpha_half_implies_expression_equals_two (α : Real) 
  (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_half_implies_expression_equals_two_l2469_246994


namespace NUMINAMATH_CALUDE_equation_solution_l2469_246934

theorem equation_solution : ∃ x : ℚ, (2 * x + 1) / 5 - x / 10 = 2 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2469_246934


namespace NUMINAMATH_CALUDE_remainder_theorem_l2469_246990

/-- A polynomial of the form Ax^6 + Bx^4 + Cx^2 + 5 -/
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 5

/-- The remainder when p(x) is divided by x-2 is 13 -/
def remainder_condition (A B C : ℝ) : Prop := p A B C 2 = 13

theorem remainder_theorem (A B C : ℝ) (h : remainder_condition A B C) :
  p A B C (-2) = 13 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2469_246990


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2469_246978

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 6| + |x + 7| ≥ 10 ∧ ∃ y : ℝ, |y + 3| + |y + 6| + |y + 7| = 10 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l2469_246978


namespace NUMINAMATH_CALUDE_parallel_line_k_value_l2469_246927

/-- Given a line passing through points (5, -3) and (k, 20) that is parallel to the line 3x - 2y = 12, 
    prove that k = 61/3 -/
theorem parallel_line_k_value (k : ℚ) : 
  (∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + b → (x = 5 ∧ y = -3) ∨ (x = k ∧ y = 20)) ∧
                (∀ x y : ℚ, 3 * x - 2 * y = 12 → y = m * x - 6)) → 
  k = 61 / 3 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_k_value_l2469_246927


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_l2469_246997

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific coordinates for this problem

-- Define the inscribed circle with center O in triangle ABC
def InscribedCircle (O : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  true  -- We don't need to define the specific properties of the inscribed circle

-- Define points M and N where the circle touches sides AB and AC
def TouchPoints (M N : ℝ × ℝ) (ABC : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  true  -- We don't need to define the specific properties of these points

-- Define the inscribed circle with center Q in triangle AMN
def InscribedCircleAMN (Q : ℝ × ℝ) (A M N : ℝ × ℝ) : Prop :=
  true  -- We don't need to define the specific properties of this inscribed circle

-- Define the distances between points
def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry  -- We don't need to implement this function for the statement

theorem inscribed_circle_distance
  (A B C O Q M N : ℝ × ℝ)
  (h1 : Triangle A B C)
  (h2 : InscribedCircle O (A, B, C))
  (h3 : TouchPoints M N (A, B, C))
  (h4 : InscribedCircleAMN Q A M N)
  (h5 : Distance A B = 13)
  (h6 : Distance B C = 15)
  (h7 : Distance A C = 14) :
  Distance O Q = 4 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_l2469_246997


namespace NUMINAMATH_CALUDE_chalk_pieces_count_l2469_246962

/-- Given a box capacity and number of full boxes, calculates the total number of chalk pieces -/
def total_chalk_pieces (box_capacity : ℕ) (full_boxes : ℕ) : ℕ :=
  box_capacity * full_boxes

/-- Proves that the total number of chalk pieces is 3492 -/
theorem chalk_pieces_count :
  let box_capacity := 18
  let full_boxes := 194
  total_chalk_pieces box_capacity full_boxes = 3492 := by
  sorry

end NUMINAMATH_CALUDE_chalk_pieces_count_l2469_246962


namespace NUMINAMATH_CALUDE_max_citizens_for_minister_l2469_246909

theorem max_citizens_for_minister (n : ℕ) : 
  (∀ m : ℕ, m > n → Nat.choose m 4 ≥ Nat.choose m 2) ↔ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_citizens_for_minister_l2469_246909


namespace NUMINAMATH_CALUDE_base_13_conversion_l2469_246918

-- Define a function to convert a base 10 number to base 13
def toBase13 (n : ℕ) : String :=
  sorry

-- Define a function to convert a base 13 string to base 10
def fromBase13 (s : String) : ℕ :=
  sorry

-- Theorem statement
theorem base_13_conversion :
  toBase13 136 = "A6" ∧ fromBase13 "A6" = 136 :=
sorry

end NUMINAMATH_CALUDE_base_13_conversion_l2469_246918


namespace NUMINAMATH_CALUDE_parity_of_expression_l2469_246913

theorem parity_of_expression (o₁ o₂ n : ℤ) 
  (h₁ : ∃ k : ℤ, o₁ = 2*k + 1) 
  (h₂ : ∃ m : ℤ, o₂ = 2*m + 1) :
  (o₁^2 + n*o₁*o₂) % 2 = 1 ↔ n % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parity_of_expression_l2469_246913
