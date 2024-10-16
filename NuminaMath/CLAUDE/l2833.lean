import Mathlib

namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l2833_283331

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 12*x - 64 = 0 ∧ 
  (∀ y : ℝ, y^2 + 12*y - 64 = 0 → x ≤ y) → 
  x = -16 := by
sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l2833_283331


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2833_283321

/-- Represents a collection of sheets with page numbers -/
structure Sheets :=
  (total_sheets : ℕ)
  (total_pages : ℕ)
  (borrowed_sheets : ℕ)

/-- Calculates the average page number of remaining sheets -/
def average_remaining_pages (s : Sheets) : ℚ :=
  let remaining_pages := s.total_pages - 2 * s.borrowed_sheets
  let sum_remaining := (s.total_pages * (s.total_pages + 1) / 2) -
                       (2 * s.borrowed_sheets * (2 * s.borrowed_sheets + 1) / 2)
  sum_remaining / remaining_pages

/-- The main theorem to prove -/
theorem borrowed_sheets_theorem (s : Sheets) 
  (h1 : s.total_sheets = 30)
  (h2 : s.total_pages = 60)
  (h3 : s.borrowed_sheets = 10) :
  average_remaining_pages s = 25 := by
  sorry

#eval average_remaining_pages ⟨30, 60, 10⟩

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l2833_283321


namespace NUMINAMATH_CALUDE_b_value_l2833_283348

theorem b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l2833_283348


namespace NUMINAMATH_CALUDE_football_players_count_l2833_283356

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 38)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 9)
  (h5 : total = (football - both) + (tennis - both) + both + neither) :
  football = 26 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l2833_283356


namespace NUMINAMATH_CALUDE_ice_volume_problem_l2833_283386

theorem ice_volume_problem (V : ℝ) : 
  (V * (1/4) * (1/4) = 0.4) → V = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_ice_volume_problem_l2833_283386


namespace NUMINAMATH_CALUDE_vehicles_with_only_cd_player_l2833_283354

/-- Represents the percentage of vehicles with specific features -/
structure VehicleFeatures where
  power_windows : ℝ
  anti_lock_brakes : ℝ
  cd_player : ℝ
  power_windows_and_anti_lock : ℝ
  anti_lock_and_cd : ℝ
  power_windows_and_cd : ℝ

/-- The theorem stating the percentage of vehicles with only a CD player -/
theorem vehicles_with_only_cd_player (v : VehicleFeatures)
  (h1 : v.power_windows = 60)
  (h2 : v.anti_lock_brakes = 25)
  (h3 : v.cd_player = 75)
  (h4 : v.power_windows_and_anti_lock = 10)
  (h5 : v.anti_lock_and_cd = 15)
  (h6 : v.power_windows_and_cd = 22)
  (h7 : v.power_windows_and_anti_lock + v.anti_lock_and_cd + v.power_windows_and_cd ≤ v.cd_player) :
  v.cd_player - (v.power_windows_and_cd + v.anti_lock_and_cd) = 38 := by
  sorry

end NUMINAMATH_CALUDE_vehicles_with_only_cd_player_l2833_283354


namespace NUMINAMATH_CALUDE_square_perimeter_7m_l2833_283374

/-- The perimeter of a square with side length 7 meters is 28 meters. -/
theorem square_perimeter_7m : 
  ∀ (s : ℝ), s = 7 → 4 * s = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_7m_l2833_283374


namespace NUMINAMATH_CALUDE_routes_3x2_grid_l2833_283390

/-- The number of routes in a grid from top-left to bottom-right -/
def numRoutes (width height : ℕ) : ℕ :=
  Nat.choose (width + height) width

theorem routes_3x2_grid : numRoutes 3 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_routes_3x2_grid_l2833_283390


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1503_odd_integers_l2833_283366

def first_n_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def sum_of_squares (list : List ℕ) : ℕ :=
  list.map (fun x => x * x) |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_squares_of_first_1503_odd_integers :
  units_digit (sum_of_squares (first_n_odd_integers 1503)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_squares_of_first_1503_odd_integers_l2833_283366


namespace NUMINAMATH_CALUDE_quiz_winning_probability_l2833_283385

/-- The number of questions in the quiz -/
def num_questions : ℕ := 4

/-- The number of choices for each question -/
def num_choices : ℕ := 4

/-- The minimum number of correct answers needed to win -/
def min_correct : ℕ := 3

/-- The probability of answering a single question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The probability of answering a single question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of winning the quiz -/
def prob_winning : ℚ := (num_questions.choose min_correct) * (prob_correct ^ min_correct * prob_incorrect ^ (num_questions - min_correct)) +
                        (num_questions.choose num_questions) * (prob_correct ^ num_questions)

theorem quiz_winning_probability :
  prob_winning = 13 / 256 := by
  sorry

end NUMINAMATH_CALUDE_quiz_winning_probability_l2833_283385


namespace NUMINAMATH_CALUDE_max_distance_complex_l2833_283368

/-- Given a complex number z₁ = i(1-i)³ and any complex number z such that |z| = 1,
    the maximum value of |z - z₁| is 1 + 2√2. -/
theorem max_distance_complex (z : ℂ) : 
  let z₁ : ℂ := Complex.I * (1 - Complex.I)^3
  Complex.abs z = 1 →
  (⨆ (z : ℂ), Complex.abs (z - z₁)) = 1 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2833_283368


namespace NUMINAMATH_CALUDE_impossible_single_piece_on_center_l2833_283360

/-- Represents a square on the solitaire board -/
inductive Square
| One
| Two
| Three

/-- Represents the state of the solitaire board -/
structure BoardState where
  occupied_ones : Nat
  occupied_twos : Nat

/-- Represents a valid move in the solitaire game -/
inductive Move
| HorizontalMove
| VerticalMove

/-- Defines K as the sum of occupied 1-squares and 2-squares -/
def K (state : BoardState) : Nat :=
  state.occupied_ones + state.occupied_twos

/-- The initial state of the board -/
def initial_state : BoardState :=
  { occupied_ones := 15, occupied_twos := 15 }

/-- Applies a move to the board state -/
def apply_move (state : BoardState) (move : Move) : BoardState :=
  sorry

/-- Theorem stating that it's impossible to end with a single piece on the central square -/
theorem impossible_single_piece_on_center :
  ∀ (moves : List Move),
    let final_state := moves.foldl apply_move initial_state
    ¬(K final_state = 1 ∧ final_state.occupied_ones + final_state.occupied_twos = 1) :=
  sorry

end NUMINAMATH_CALUDE_impossible_single_piece_on_center_l2833_283360


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2833_283335

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ r < 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_ratio
  (a : ℕ → ℝ)
  (h_geom : is_positive_geometric_sequence a)
  (h_decreasing : ∀ n : ℕ, a (n + 1) < a n)
  (h_product : a 2 * a 8 = 6)
  (h_sum : a 4 + a 6 = 5) :
  a 5 / a 7 = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2833_283335


namespace NUMINAMATH_CALUDE_number_ordering_l2833_283337

def A : ℕ := 9^(9^9)
def B : ℕ := 99^9
def C : ℕ := (9^9)^9
def D : ℕ := (Nat.factorial 9)^(Nat.factorial 9)

theorem number_ordering : B < C ∧ C < A ∧ A < D := by sorry

end NUMINAMATH_CALUDE_number_ordering_l2833_283337


namespace NUMINAMATH_CALUDE_three_balls_selected_l2833_283318

def num_balls : ℕ := 100
def prob_odd_first : ℚ := 2/3

theorem three_balls_selected 
  (h1 : num_balls = 100)
  (h2 : prob_odd_first = 2/3)
  (h3 : ∃ (odd_count even_count : ℕ), 
    odd_count = 2 ∧ even_count = 1 ∧ 
    odd_count + even_count = num_selected) :
  num_selected = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_balls_selected_l2833_283318


namespace NUMINAMATH_CALUDE_triangle_area_in_square_pyramid_l2833_283363

/-- Square pyramid with given dimensions and points -/
structure SquarePyramid where
  -- Base side length
  base_side : ℝ
  -- Altitude
  altitude : ℝ
  -- Points P, Q, R are located 1/4 of the way from B, D, C to E respectively
  point_ratio : ℝ

/-- The area of triangle PQR in the square pyramid -/
def triangle_area (pyramid : SquarePyramid) : ℝ := sorry

/-- Theorem statement -/
theorem triangle_area_in_square_pyramid :
  ∀ (pyramid : SquarePyramid),
  pyramid.base_side = 4 ∧ 
  pyramid.altitude = 8 ∧ 
  pyramid.point_ratio = 1/4 →
  triangle_area pyramid = (45 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_square_pyramid_l2833_283363


namespace NUMINAMATH_CALUDE_snowboard_final_price_l2833_283399

/-- 
Given a snowboard with an original price and two successive discounts,
calculate the final price after both discounts are applied.
-/
theorem snowboard_final_price 
  (original_price : ℝ)
  (friday_discount : ℝ)
  (monday_discount : ℝ)
  (h1 : original_price = 200)
  (h2 : friday_discount = 0.4)
  (h3 : monday_discount = 0.25) :
  original_price * (1 - friday_discount) * (1 - monday_discount) = 90 :=
by sorry

#check snowboard_final_price

end NUMINAMATH_CALUDE_snowboard_final_price_l2833_283399


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2833_283384

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2833_283384


namespace NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2833_283319

/-- Represents a 2006 × 2006 table filled with numbers from 1 to 2006² -/
def Table := Fin 2006 → Fin 2006 → Fin (2006^2)

/-- Checks if two positions in the table are adjacent -/
def adjacent (p q : Fin 2006 × Fin 2006) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ q.2 = p.2 + 1)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ q.1 = p.1 + 1)) ∨
  (p.1 = q.1 + 1 ∧ p.2 = q.2 + 1) ∨
  (q.1 = p.1 + 1 ∧ q.2 = p.2 + 1) ∨
  (p.1 = q.1 + 1 ∧ q.2 = p.2 + 1) ∨
  (q.1 = p.1 + 1 ∧ p.2 = q.2 + 1)

/-- The main theorem to be proved -/
theorem adjacent_sum_divisible_by_four (t : Table) :
  ∃ (p q : Fin 2006 × Fin 2006),
    adjacent p q ∧ (((t p.1 p.2).val + (t q.1 q.2).val + 2) % 4 = 0) := by
  sorry


end NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l2833_283319


namespace NUMINAMATH_CALUDE_students_after_yoongi_l2833_283358

theorem students_after_yoongi (total_students : ℕ) (students_before : ℕ) : 
  total_students = 20 → students_before = 11 → total_students - (students_before + 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_after_yoongi_l2833_283358


namespace NUMINAMATH_CALUDE_seat_39_is_51_l2833_283341

/-- Calculates the seat number for the nth person in a circular seating arrangement --/
def seatNumber (n : ℕ) (totalSeats : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let binaryRep := (n - 1).digits 2
    let seatCalc := binaryRep.foldl (fun acc (b : ℕ) => (2 * acc + b) % totalSeats) 1
    if seatCalc = 0 then totalSeats else seatCalc

/-- The theorem stating that the 39th person sits on seat 51 in a 128-seat arrangement --/
theorem seat_39_is_51 : seatNumber 39 128 = 51 := by
  sorry

/-- Verifies the seating arrangement for the first few people --/
example : List.map (fun n => seatNumber n 128) [1, 2, 3, 4, 5] = [1, 65, 33, 97, 17] := by
  sorry

end NUMINAMATH_CALUDE_seat_39_is_51_l2833_283341


namespace NUMINAMATH_CALUDE_pot_contribution_proof_l2833_283342

theorem pot_contribution_proof (total_people : Nat) (first_place_percent : Real) 
  (third_place_amount : Real) : 
  total_people = 8 → 
  first_place_percent = 0.8 → 
  third_place_amount = 4 → 
  ∃ (individual_contribution : Real),
    individual_contribution = 5 ∧ 
    individual_contribution * total_people = third_place_amount / ((1 - first_place_percent) / 2) :=
by sorry

end NUMINAMATH_CALUDE_pot_contribution_proof_l2833_283342


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l2833_283397

/-- An isosceles triangle with base 16 and height 15 -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  isBase16 : base = 16
  isHeight15 : height = 15

/-- A semicircle inscribed in the isosceles triangle -/
structure InscribedSemicircle (t : IsoscelesTriangle) where
  radius : ℝ
  diameterOnBase : radius * 2 ≤ t.base

/-- The radius of the inscribed semicircle is 120/17 -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangle) 
  (s : InscribedSemicircle t) : s.radius = 120 / 17 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l2833_283397


namespace NUMINAMATH_CALUDE_tangent_intersection_points_l2833_283369

/-- Given a function f(x) = x^3 - x^2 + ax + 1, prove that the tangent line passing through
    the origin intersects the curve y = f(x) at the points (1, a + 1) and (-1, -a - 1). -/
theorem tangent_intersection_points (a : ℝ) :
  let f := λ x : ℝ => x^3 - x^2 + a*x + 1
  let tangent_line := λ x : ℝ => (a + 1) * x
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -1 ∧
    f x₁ = tangent_line x₁ ∧
    f x₂ = tangent_line x₂ ∧
    (∀ x : ℝ, f x = tangent_line x → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_tangent_intersection_points_l2833_283369


namespace NUMINAMATH_CALUDE_part_1_part_2_l2833_283328

-- Define the sets M, N, and H
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | |x - a| ≤ 2}

-- Define the custom set operation ∆
def triangleOp (A B : Set ℝ) : Set ℝ := A ∩ (Set.univ \ B)

-- Theorem for part (1)
theorem part_1 :
  triangleOp M N = {x | 1 < x ∧ x < 2} ∧
  triangleOp N M = {x | 3 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for part (2)
theorem part_2 (a : ℝ) :
  triangleOp (triangleOp N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then
      {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then
      {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then
      {x | a + 2 < x ∧ x < 2}
    else
      ∅ := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l2833_283328


namespace NUMINAMATH_CALUDE_hidden_dots_on_three_dice_l2833_283373

def total_dots_on_die : ℕ := 21

def total_dots_on_three_dice : ℕ := 3 * total_dots_on_die

def visible_faces : List ℕ := [1, 2, 2, 3, 5, 4, 5, 6]

def sum_visible_faces : ℕ := visible_faces.sum

theorem hidden_dots_on_three_dice : 
  total_dots_on_three_dice - sum_visible_faces = 35 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_on_three_dice_l2833_283373


namespace NUMINAMATH_CALUDE_gis_may_lead_to_overfishing_l2833_283349

/-- Represents the use of GIS technology in fishery production -/
structure GISTechnology where
  locateSchools : Bool
  widelyIntroduced : Bool

/-- Represents the state of fishery resources -/
structure FisheryResources where
  overfishing : Bool
  exhausted : Bool

/-- The impact of GIS technology on fishery resources -/
def gisImpact (tech : GISTechnology) : FisheryResources :=
  { overfishing := tech.locateSchools ∧ tech.widelyIntroduced,
    exhausted := tech.locateSchools ∧ tech.widelyIntroduced }

theorem gis_may_lead_to_overfishing (tech : GISTechnology) 
  (h1 : tech.locateSchools = true) 
  (h2 : tech.widelyIntroduced = true) : 
  (gisImpact tech).overfishing = true ∧ (gisImpact tech).exhausted = true :=
by sorry

end NUMINAMATH_CALUDE_gis_may_lead_to_overfishing_l2833_283349


namespace NUMINAMATH_CALUDE_circular_track_circumference_l2833_283387

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (meeting_time : ℝ) 
  (h1 : speed1 = 7) 
  (h2 : speed2 = 8) 
  (h3 : meeting_time = 45) : 
  speed1 * meeting_time + speed2 * meeting_time = 675 := by
  sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l2833_283387


namespace NUMINAMATH_CALUDE_simplify_power_l2833_283344

theorem simplify_power (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_l2833_283344


namespace NUMINAMATH_CALUDE_subset_condition_iff_m_geq_three_l2833_283326

theorem subset_condition_iff_m_geq_three (m : ℝ) : 
  (∀ x : ℝ, x^2 - x ≤ 0 → x^2 - 4*x + m ≥ 0) ↔ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_iff_m_geq_three_l2833_283326


namespace NUMINAMATH_CALUDE_smallest_class_size_l2833_283396

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∀ i, scores i ≤ 120) →  -- Each student took a 120-point test
  (∃ s : Finset (Fin n), s.card = 8 ∧ ∀ i ∈ s, scores i = 120) →  -- Eight students scored 120
  (∀ i, scores i ≥ 72) →  -- Each student scored at least 72
  (Finset.sum Finset.univ scores / n = 84) →  -- The mean score was 84
  (n ≥ 32 ∧ ∀ m : ℕ, m < 32 → ¬ (∃ scores' : Fin m → ℕ, 
    (∀ i, scores' i ≤ 120) ∧ 
    (∃ s : Finset (Fin m), s.card = 8 ∧ ∀ i ∈ s, scores' i = 120) ∧ 
    (∀ i, scores' i ≥ 72) ∧ 
    (Finset.sum Finset.univ scores' / m = 84))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2833_283396


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l2833_283314

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + 2 * c = 20) →                  -- condition on a and c
  (a < c) →                           -- additional condition
  (a = 10 - 5 * Real.sqrt 2 ∧ c = 5 + (5 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l2833_283314


namespace NUMINAMATH_CALUDE_inequality_proof_l2833_283364

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / a + 1 / b + 1 / c ≥ 9 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2833_283364


namespace NUMINAMATH_CALUDE_no_solution_iff_a_geq_bound_l2833_283375

theorem no_solution_iff_a_geq_bound (a : ℝ) :
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) ↔ a ≥ (Real.sqrt 3 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_geq_bound_l2833_283375


namespace NUMINAMATH_CALUDE_only_sunrise_certain_l2833_283313

-- Define the type for events
inductive Event
  | MovieTicket
  | TVAdvertisement
  | Rain
  | Sunrise

-- Define what it means for an event to be certain
def is_certain (e : Event) : Prop :=
  match e with
  | Event.Sunrise => true
  | _ => false

-- Theorem stating that only the sunrise event is certain
theorem only_sunrise_certain :
  ∀ (e : Event), is_certain e ↔ e = Event.Sunrise :=
by
  sorry

end NUMINAMATH_CALUDE_only_sunrise_certain_l2833_283313


namespace NUMINAMATH_CALUDE_jill_study_time_difference_l2833_283395

/-- Represents the study time in minutes for each day -/
def StudyTime := Fin 3 → ℕ

theorem jill_study_time_difference (study : StudyTime) : 
  (study 0 = 120) →  -- First day study time in minutes
  (study 1 = 2 * study 0) →  -- Second day is double the first day
  (study 0 + study 1 + study 2 = 540) →  -- Total study time over 3 days
  (study 1 - study 2 = 60) :=  -- Difference between second and third day
by
  sorry

end NUMINAMATH_CALUDE_jill_study_time_difference_l2833_283395


namespace NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l2833_283301

theorem largest_multiple_six_negation_greater_than_neg_150 :
  (∀ n : ℤ, n % 6 = 0 ∧ -n > -150 → n ≤ 144) ∧
  144 % 6 = 0 ∧ -144 > -150 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l2833_283301


namespace NUMINAMATH_CALUDE_jill_marathon_time_l2833_283333

/-- The length of a marathon in kilometers -/
def marathon_length : ℝ := 40

/-- Jack's marathon time in hours -/
def jack_time : ℝ := 4.5

/-- The ratio of Jack's speed to Jill's speed -/
def speed_ratio : ℝ := 0.888888888888889

/-- Jill's marathon time in hours -/
def jill_time : ℝ := 4

theorem jill_marathon_time :
  marathon_length / (marathon_length / jack_time * (1 / speed_ratio)) = jill_time := by
  sorry

end NUMINAMATH_CALUDE_jill_marathon_time_l2833_283333


namespace NUMINAMATH_CALUDE_first_day_of_month_l2833_283376

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (day_after d n)

theorem first_day_of_month (d : DayOfWeek) :
  day_after d 29 = DayOfWeek.Wednesday → d = DayOfWeek.Tuesday :=
by
  sorry


end NUMINAMATH_CALUDE_first_day_of_month_l2833_283376


namespace NUMINAMATH_CALUDE_no_natural_n_for_sum_of_squares_l2833_283336

theorem no_natural_n_for_sum_of_squares : 
  ¬ ∃ (n : ℕ), ∃ (x y : ℕ+), 
    2 * n * (n + 1) * (n + 2) * (n + 3) + 12 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_n_for_sum_of_squares_l2833_283336


namespace NUMINAMATH_CALUDE_rain_probability_in_tel_aviv_l2833_283393

/-- The probability of exactly k successes in n independent trials,
    where the probability of success in each trial is p. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of rain on any given day in Tel Aviv -/
def probabilityOfRain : ℝ := 0.5

/-- The number of randomly chosen days -/
def totalDays : ℕ := 6

/-- The number of rainy days we're interested in -/
def rainyDays : ℕ := 4

theorem rain_probability_in_tel_aviv :
  binomialProbability totalDays rainyDays probabilityOfRain = 0.234375 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_in_tel_aviv_l2833_283393


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_to_100_l2833_283350

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic series from 10 to 100 with common difference 1 is 5005 -/
theorem arithmetic_sum_10_to_100 :
  arithmetic_sum 10 100 1 = 5005 := by
  sorry

#eval arithmetic_sum 10 100 1

end NUMINAMATH_CALUDE_arithmetic_sum_10_to_100_l2833_283350


namespace NUMINAMATH_CALUDE_sphere_radius_is_4_l2833_283338

/-- Represents a cylindrical container with spheres -/
structure Container where
  initialHeight : ℝ
  sphereRadius : ℝ
  numSpheres : ℕ

/-- Calculates the final height of water in the container after adding spheres -/
def finalHeight (c : Container) : ℝ :=
  c.initialHeight + c.sphereRadius * 2

/-- The problem statement -/
theorem sphere_radius_is_4 (c : Container) :
  c.initialHeight = 8 ∧
  c.numSpheres = 3 ∧
  finalHeight c = c.initialHeight + c.sphereRadius * 2 →
  c.sphereRadius = 4 := by
  sorry


end NUMINAMATH_CALUDE_sphere_radius_is_4_l2833_283338


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_l2833_283320

theorem least_x_for_even_prime (x : ℕ+) (p : ℕ) : 
  Nat.Prime p → (x.val : ℚ) / (11 * p) = 2 → x.val ≥ 44 :=
sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_l2833_283320


namespace NUMINAMATH_CALUDE_bakery_pie_production_l2833_283398

/-- The number of pies a bakery can make in one hour, given specific pricing and profit conditions. -/
theorem bakery_pie_production (piece_price : ℚ) (pieces_per_pie : ℕ) (pie_cost : ℚ) (total_profit : ℚ) 
  (h1 : piece_price = 4)
  (h2 : pieces_per_pie = 3)
  (h3 : pie_cost = 1/2)
  (h4 : total_profit = 138) :
  (total_profit / (piece_price * ↑pieces_per_pie - pie_cost) : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_production_l2833_283398


namespace NUMINAMATH_CALUDE_max_value_a_l2833_283312

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 80) :
  a ≤ 4724 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4724 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 80 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l2833_283312


namespace NUMINAMATH_CALUDE_cats_left_after_sale_l2833_283317

theorem cats_left_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : siamese = 38 → house = 25 → sold = 45 → siamese + house - sold = 18 := by
  sorry

end NUMINAMATH_CALUDE_cats_left_after_sale_l2833_283317


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l2833_283311

theorem tan_alpha_minus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (β + Real.pi / 4) = 3) :
  Real.tan (α - Real.pi / 4) = -1 / 7 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_fourth_l2833_283311


namespace NUMINAMATH_CALUDE_negative_two_and_sqrt_four_are_opposite_l2833_283307

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- State the theorem
theorem negative_two_and_sqrt_four_are_opposite : 
  are_opposite (-2 : ℝ) (Real.sqrt ((-2 : ℝ)^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_negative_two_and_sqrt_four_are_opposite_l2833_283307


namespace NUMINAMATH_CALUDE_cruise_ship_tourists_l2833_283310

theorem cruise_ship_tourists : ∃ (x : ℕ) (tourists : ℕ), 
  x > 1 ∧ 
  tourists = 12 * x + 1 ∧
  ∃ (y : ℕ), y ≤ 15 ∧ tourists = y * (x - 1) ∧
  tourists = 169 := by
  sorry

end NUMINAMATH_CALUDE_cruise_ship_tourists_l2833_283310


namespace NUMINAMATH_CALUDE_unfinished_courses_l2833_283353

/-- Given the conditions of a construction project, calculate the number of unfinished courses in the last wall. -/
theorem unfinished_courses
  (courses_per_wall : ℕ)
  (bricks_per_course : ℕ)
  (total_walls : ℕ)
  (bricks_used : ℕ)
  (h1 : courses_per_wall = 6)
  (h2 : bricks_per_course = 10)
  (h3 : total_walls = 4)
  (h4 : bricks_used = 220) :
  (courses_per_wall * bricks_per_course * total_walls - bricks_used) / bricks_per_course = 2 :=
by sorry

end NUMINAMATH_CALUDE_unfinished_courses_l2833_283353


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2833_283330

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x - b

-- Define the solution set of the quadratic inequality
def solution_set (a b : ℝ) := {x : ℝ | 1 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h : ∀ x, x ∈ solution_set a b ↔ f a b x < 0) :
  a = 4 ∧ b = -3 ∧
  (∀ x, (2*x + a) / (x + b) > 1 ↔ x > -7 ∨ x > 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2833_283330


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2833_283389

/-- Given that when x = 1, the value of (1/2)ax³ - 3bx + 4 is 9,
    prove that when x = -1, the value of the expression is -1 -/
theorem algebraic_expression_value (a b : ℝ) :
  (1/2 * a * 1^3 - 3 * b * 1 + 4 = 9) →
  (1/2 * a * (-1)^3 - 3 * b * (-1) + 4 = -1) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2833_283389


namespace NUMINAMATH_CALUDE_unripe_orange_harvest_l2833_283378

/-- The number of sacks of unripe oranges harvested per day -/
def daily_unripe_harvest : ℕ := 65

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges harvested over the harvest period -/
def total_unripe_harvest : ℕ := daily_unripe_harvest * harvest_days

theorem unripe_orange_harvest : total_unripe_harvest = 390 := by
  sorry

end NUMINAMATH_CALUDE_unripe_orange_harvest_l2833_283378


namespace NUMINAMATH_CALUDE_total_selling_price_l2833_283334

/-- Calculate the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem total_selling_price
  (quantity : ℕ)
  (profit_per_meter : ℚ)
  (cost_price_per_meter : ℚ)
  (h1 : quantity = 92)
  (h2 : profit_per_meter = 24)
  (h3 : cost_price_per_meter = 83.5)
  : (quantity : ℚ) * (cost_price_per_meter + profit_per_meter) = 9890 := by
  sorry

end NUMINAMATH_CALUDE_total_selling_price_l2833_283334


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2833_283322

/-- Calculates the sample size for a stratified sampling based on gender -/
def calculateSampleSize (totalEmployees : ℕ) (maleEmployees : ℕ) (maleSampleSize : ℕ) : ℕ :=
  (totalEmployees * maleSampleSize) / maleEmployees

/-- Proves that the sample size is 24 given the conditions -/
theorem stratified_sample_size :
  let totalEmployees : ℕ := 120
  let maleEmployees : ℕ := 90
  let maleSampleSize : ℕ := 18
  calculateSampleSize totalEmployees maleEmployees maleSampleSize = 24 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2833_283322


namespace NUMINAMATH_CALUDE_proportional_relation_l2833_283306

/-- Given that x is directly proportional to y^2 and y is inversely proportional to z,
    prove that if x = 5 when z = 20, then x = 40/81 when z = 45. -/
theorem proportional_relation (x y z : ℝ) (c d : ℝ) (h1 : x = c * y^2) (h2 : y * z = d)
  (h3 : z = 20 → x = 5) : z = 45 → x = 40 / 81 := by
  sorry

end NUMINAMATH_CALUDE_proportional_relation_l2833_283306


namespace NUMINAMATH_CALUDE_existence_of_odd_powers_sum_l2833_283327

theorem existence_of_odd_powers_sum (m : ℤ) :
  ∃ (a b k : ℤ), 
    Odd a ∧ 
    Odd b ∧ 
    k > 0 ∧ 
    2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_odd_powers_sum_l2833_283327


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2833_283362

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (2 * x^2 - x * (x - 4) = 5) ↔ (x^2 + 4*x - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2833_283362


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l2833_283380

/-- A parabola with equation y = x^2 + x -/
def parabola (x : ℝ) : ℝ := x^2 + x

/-- A circle inside the parabola, tangent at two points -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ
  tangent_to_parabola1 : parabola tangentPoint1.1 = tangentPoint1.2
  tangent_to_parabola2 : parabola tangentPoint2.1 = tangentPoint2.2
  on_circle1 : (tangentPoint1.1 - center.1)^2 + (tangentPoint1.2 - center.2)^2 = radius^2
  on_circle2 : (tangentPoint2.1 - center.1)^2 + (tangentPoint2.2 - center.2)^2 = radius^2

/-- The theorem stating the height difference -/
theorem tangent_circle_height_difference (c : TangentCircle) :
  c.center.2 - c.tangentPoint1.2 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l2833_283380


namespace NUMINAMATH_CALUDE_sum_243_62_base5_l2833_283329

/-- Converts a natural number to its base 5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry  -- Implementation details omitted

/-- Theorem: The sum of 243 and 62 in base 5 is 2170₅ --/
theorem sum_243_62_base5 :
  addBase5 (toBase5 243) (toBase5 62) = [0, 7, 1, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_243_62_base5_l2833_283329


namespace NUMINAMATH_CALUDE_first_quartile_of_numbers_l2833_283381

def numbers : List ℝ := [42, 24, 30, 22, 26, 27, 33, 35]

def median (l : List ℝ) : ℝ := sorry

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  let less_than_m := l.filter (λ x => x < m)
  median less_than_m

theorem first_quartile_of_numbers :
  first_quartile numbers = 25 := by sorry

end NUMINAMATH_CALUDE_first_quartile_of_numbers_l2833_283381


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2833_283332

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 = 2 →
  a 2 + a 4 = 6 →
  a 1 + a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2833_283332


namespace NUMINAMATH_CALUDE_train_crossing_time_l2833_283361

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (pole_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 300)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 200) :
  (train_length + platform_length) / (train_length / pole_crossing_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2833_283361


namespace NUMINAMATH_CALUDE_polynomial_real_root_condition_l2833_283372

theorem polynomial_real_root_condition (a : ℝ) :
  (∃ x : ℝ, x^4 + a*x^3 - x^2 + a^2*x + 1 = 0) ↔ (a ≤ -1 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_real_root_condition_l2833_283372


namespace NUMINAMATH_CALUDE_f_shifted_l2833_283347

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem f_shifted (x : ℝ) :
  (1 ≤ x ∧ x ≤ 3) → (2 ≤ x ∧ x ≤ 4) → f (x - 1) = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_l2833_283347


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2833_283394

/-- Given a polynomial P(z) = 4z^3 - 5z^2 - 19z + 4, when divided by 4z + 6
    with quotient z^2 - 4z + 1, prove that the remainder is 5z^2 + z - 2. -/
theorem polynomial_division_remainder
  (z : ℂ)
  (P : ℂ → ℂ)
  (D : ℂ → ℂ)
  (Q : ℂ → ℂ)
  (h1 : P z = 4 * z^3 - 5 * z^2 - 19 * z + 4)
  (h2 : D z = 4 * z + 6)
  (h3 : Q z = z^2 - 4 * z + 1)
  : ∃ R : ℂ → ℂ, P z = D z * Q z + R z ∧ R z = 5 * z^2 + z - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2833_283394


namespace NUMINAMATH_CALUDE_incorrect_calculation_l2833_283315

theorem incorrect_calculation (m n : ℕ) (h1 : n ≤ 100) : 
  ¬ (∃ k : ℕ, ∃ B : ℕ, 
    (m : ℚ) / n = (k : ℚ) + B / (1000 * n) ∧ 
    167 ≤ B ∧ B < 168) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_calculation_l2833_283315


namespace NUMINAMATH_CALUDE_sum_of_valid_numbers_l2833_283305

def digits : List Nat := [1, 3, 5]

def isValidNumber (n : Nat) : Bool :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100) ∈ digits ∧
  ((n / 10) % 10) ∈ digits ∧
  (n % 10) ∈ digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧
  (n / 100) ≠ (n % 10) ∧
  ((n / 10) % 10) ≠ (n % 10)

def validNumbers : List Nat :=
  (List.range 1000).filter isValidNumber

theorem sum_of_valid_numbers :
  validNumbers.sum = 1998 := by sorry

end NUMINAMATH_CALUDE_sum_of_valid_numbers_l2833_283305


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l2833_283300

theorem charity_ticket_revenue 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (full_price : ℕ) 
  (full_price_tickets : ℕ) 
  (half_price_tickets : ℕ) :
  total_tickets = 140 →
  total_revenue = 2001 →
  total_tickets = full_price_tickets + half_price_tickets →
  total_revenue = full_price * full_price_tickets + (full_price / 2) * half_price_tickets →
  full_price > 0 →
  full_price_tickets * full_price = 782 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l2833_283300


namespace NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l2833_283388

/-- The product of digits of a two-digit number -/
def P (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- The sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- A two-digit number M satisfying M = P(M) + S(M) + 6 has a tens digit of either 1 or 2 -/
theorem tens_digit_of_special_two_digit_number :
  ∀ M : ℕ, 
    (10 ≤ M ∧ M < 100) →  -- M is a two-digit number
    (M = P M + S M + 6) →  -- M satisfies the special condition
    (M / 10 = 1 ∨ M / 10 = 2) :=  -- The tens digit is either 1 or 2
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_special_two_digit_number_l2833_283388


namespace NUMINAMATH_CALUDE_rabbit_walk_distance_l2833_283343

/-- The perimeter of a square park -/
def park_perimeter (side_length : ℝ) : ℝ := 4 * side_length

/-- The theorem stating that a rabbit walking along the perimeter of a square park
    with a side length of 13 meters walks 52 meters -/
theorem rabbit_walk_distance : park_perimeter 13 = 52 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_walk_distance_l2833_283343


namespace NUMINAMATH_CALUDE_chewing_gum_cost_l2833_283346

/-- Proves that the cost of each pack of chewing gum is $1, given the initial amount,
    purchases, and remaining amount. -/
theorem chewing_gum_cost
  (initial_amount : ℝ)
  (num_gum_packs : ℕ)
  (num_chocolate_bars : ℕ)
  (chocolate_bar_price : ℝ)
  (num_candy_canes : ℕ)
  (candy_cane_price : ℝ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 10)
  (h2 : num_gum_packs = 3)
  (h3 : num_chocolate_bars = 5)
  (h4 : chocolate_bar_price = 1)
  (h5 : num_candy_canes = 2)
  (h6 : candy_cane_price = 0.5)
  (h7 : remaining_amount = 1) :
  (initial_amount - remaining_amount
    - (num_chocolate_bars * chocolate_bar_price + num_candy_canes * candy_cane_price))
  / num_gum_packs = 1 := by
sorry


end NUMINAMATH_CALUDE_chewing_gum_cost_l2833_283346


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_68_l2833_283340

theorem consecutive_even_integers_sum_68 :
  ∃ (x y z w : ℕ+), 
    (x : ℤ) + y + z + w = 68 ∧
    y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧
    Even x ∧ Even y ∧ Even z ∧ Even w :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_68_l2833_283340


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2833_283309

/-- Given a line with equation y - 6 = -2(x - 3), 
    the sum of its x-intercept and y-intercept is 18 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 6 = -2 * (x - 3) → 
  ∃ (x_int y_int : ℝ), 
    (y_int - 6 = -2 * (x_int - 3) ∧ y_int = 0) ∧
    (0 - 6 = -2 * (0 - 3) ∧ y_int = 0) ∧
    x_int + y_int = 18 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2833_283309


namespace NUMINAMATH_CALUDE_exactly_two_late_probability_l2833_283357

/-- The probability of a worker being late on any given day -/
def p_late : ℚ := 1 / 40

/-- The probability of a worker being on time on any given day -/
def p_on_time : ℚ := 1 - p_late

/-- The number of workers considered -/
def n_workers : ℕ := 3

/-- The number of workers that need to be late -/
def n_late : ℕ := 2

theorem exactly_two_late_probability :
  (n_workers.choose n_late : ℚ) * p_late ^ n_late * p_on_time ^ (n_workers - n_late) = 117 / 64000 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_late_probability_l2833_283357


namespace NUMINAMATH_CALUDE_piggy_bank_coins_l2833_283379

theorem piggy_bank_coins (nickels : ℕ) (dimes : ℕ) (quarters : ℕ) : 
  dimes = 2 * nickels →
  quarters = dimes / 2 →
  5 * nickels + 10 * dimes + 25 * quarters = 1950 →
  nickels = 39 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_coins_l2833_283379


namespace NUMINAMATH_CALUDE_symmetric_normal_distribution_l2833_283302

/-- Represents a normally distributed population with a given mean -/
structure NormalPopulation where
  mean : ℝ
  size : ℕ
  above_threshold : ℕ
  threshold : ℝ

/-- 
Given a normally distributed population with mean 75,
if 960 out of 1200 individuals score at least 60,
then 240 individuals score above 90.
-/
theorem symmetric_normal_distribution 
  (pop : NormalPopulation)
  (h_mean : pop.mean = 75)
  (h_size : pop.size = 1200)
  (h_threshold : pop.threshold = 60)
  (h_above_threshold : pop.above_threshold = 960) :
  pop.size - pop.above_threshold = 240 :=
sorry

end NUMINAMATH_CALUDE_symmetric_normal_distribution_l2833_283302


namespace NUMINAMATH_CALUDE_max_candies_theorem_l2833_283392

/-- Represents the distribution of candies among students -/
structure CandyDistribution where
  num_students : ℕ
  total_candies : ℕ
  min_candies : ℕ
  max_candies : ℕ

/-- The greatest number of candies one student could have taken -/
def max_student_candies (d : CandyDistribution) : ℕ :=
  min d.max_candies (d.total_candies - (d.num_students - 1) * d.min_candies)

/-- Theorem stating the maximum number of candies one student could have taken -/
theorem max_candies_theorem (d : CandyDistribution) 
    (h1 : d.num_students = 50)
    (h2 : d.total_candies = 50 * 7)
    (h3 : d.min_candies = 1)
    (h4 : d.max_candies = 20) :
    max_student_candies d = 20 := by
  sorry

#eval max_student_candies { num_students := 50, total_candies := 350, min_candies := 1, max_candies := 20 }

end NUMINAMATH_CALUDE_max_candies_theorem_l2833_283392


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2833_283324

theorem sum_of_reciprocals (a b : ℝ) (ha : a^2 + a = 4) (hb : b^2 + b = 4) (hab : a ≠ b) :
  b / a + a / b = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2833_283324


namespace NUMINAMATH_CALUDE_simplified_fraction_numerator_problem_solution_l2833_283371

theorem simplified_fraction_numerator (a : ℕ) (h : a > 0) : 
  ((a + 1 : ℚ) / a - a / (a + 1)) * (a * (a + 1)) = 2 * a + 1 :=
by sorry

theorem problem_solution : 
  ((2024 : ℚ) / 2023 - 2023 / 2024) * (2023 * 2024) = 4047 :=
by sorry

end NUMINAMATH_CALUDE_simplified_fraction_numerator_problem_solution_l2833_283371


namespace NUMINAMATH_CALUDE_market_value_calculation_l2833_283339

/-- Calculates the market value of a share given its nominal value, dividend rate, and desired interest rate. -/
def marketValue (nominalValue : ℚ) (dividendRate : ℚ) (desiredInterestRate : ℚ) : ℚ :=
  (nominalValue * dividendRate) / desiredInterestRate

/-- Theorem stating that for a share with nominal value of 48, 9% dividend rate, and 12% desired interest rate, the market value is 36. -/
theorem market_value_calculation :
  marketValue 48 (9/100) (12/100) = 36 := by
  sorry

end NUMINAMATH_CALUDE_market_value_calculation_l2833_283339


namespace NUMINAMATH_CALUDE_jacket_cost_ratio_l2833_283351

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 4/5
  let cost : ℝ := selling_price * cost_rate
  cost / marked_price = 3/5 := by
sorry

end NUMINAMATH_CALUDE_jacket_cost_ratio_l2833_283351


namespace NUMINAMATH_CALUDE_dice_probabilities_l2833_283370

/-- Represents the probabilities of an unfair 6-sided dice -/
structure DiceProbabilities where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  sum_one : a + b + c + d + e + f = 1
  all_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f

/-- The probability of rolling the same number twice -/
def P (probs : DiceProbabilities) : ℝ :=
  probs.a^2 + probs.b^2 + probs.c^2 + probs.d^2 + probs.e^2 + probs.f^2

/-- The probability of rolling an odd number first and an even number second -/
def Q (probs : DiceProbabilities) : ℝ :=
  (probs.a + probs.c + probs.e) * (probs.b + probs.d + probs.f)

theorem dice_probabilities (probs : DiceProbabilities) :
  P probs ≥ 1/6 ∧ Q probs ≤ 1/4 ∧ Q probs ≥ 1/2 - 3/2 * P probs := by
  sorry

end NUMINAMATH_CALUDE_dice_probabilities_l2833_283370


namespace NUMINAMATH_CALUDE_distinct_triangles_in_cube_l2833_283377

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  /-- The number of vertices in a cube. -/
  num_vertices : ℕ
  /-- The number of edges in a cube. -/
  num_edges : ℕ
  /-- The number of edges meeting at each vertex of a cube. -/
  edges_per_vertex : ℕ
  /-- Assertion that a cube has 8 vertices. -/
  vertices_axiom : num_vertices = 8
  /-- Assertion that a cube has 12 edges. -/
  edges_axiom : num_edges = 12
  /-- Assertion that 3 edges meet at each vertex of a cube. -/
  edges_per_vertex_axiom : edges_per_vertex = 3

/-- A function that calculates the number of distinct triangles in a cube. -/
def count_distinct_triangles (c : Cube) : ℕ :=
  c.num_vertices * (c.edges_per_vertex.choose 2) / 2

/-- Theorem stating that the number of distinct triangles formed by connecting three different edges of a cube, 
    where each set of edges shares a common vertex, is equal to 12. -/
theorem distinct_triangles_in_cube (c : Cube) : 
  count_distinct_triangles c = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_cube_l2833_283377


namespace NUMINAMATH_CALUDE_min_value_theorem_l2833_283316

theorem min_value_theorem (a : ℝ) (h : a > 0) : 
  (∃ (x : ℝ), x = 3 / (2 * a) + 4 * a ∧ ∀ (y : ℝ), y = 3 / (2 * a) + 4 * a → x ≤ y) ∧ 
  (∃ (z : ℝ), z = 3 / (2 * a) + 4 * a ∧ z = 2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2833_283316


namespace NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l2833_283325

theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l2833_283325


namespace NUMINAMATH_CALUDE_snowfall_sum_l2833_283365

/-- The total snowfall recorded during a three-day snowstorm -/
def total_snowfall (wednesday thursday friday : ℝ) : ℝ :=
  wednesday + thursday + friday

/-- Proof that the total snowfall is 0.88 cm given the daily measurements -/
theorem snowfall_sum :
  total_snowfall 0.33 0.33 0.22 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_sum_l2833_283365


namespace NUMINAMATH_CALUDE_local_road_speed_l2833_283308

/-- Proves that the speed on local roads is 30 mph given the specified conditions --/
theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) 
  (highway_speed : ℝ) (average_speed : ℝ) (v : ℝ) : 
  local_distance = 90 →
  highway_distance = 75 →
  highway_speed = 60 →
  average_speed = 38.82 →
  (local_distance + highway_distance) / average_speed = local_distance / v + highway_distance / highway_speed →
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_local_road_speed_l2833_283308


namespace NUMINAMATH_CALUDE_function_symmetry_l2833_283352

/-- Given a polynomial function f(x) = ax^7 + bx^5 + cx^3 + dx + 5 where a, b, c, d are constants,
    if f(-7) = -7, then f(7) = 17 -/
theorem function_symmetry (a b c d : ℝ) :
  let f := fun x : ℝ => a * x^7 + b * x^5 + c * x^3 + d * x + 5
  (f (-7) = -7) → (f 7 = 17) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2833_283352


namespace NUMINAMATH_CALUDE_birds_on_fence_l2833_283323

theorem birds_on_fence (initial_birds additional_birds : ℕ) :
  initial_birds = 12 → additional_birds = 8 →
  initial_birds + additional_birds = 20 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2833_283323


namespace NUMINAMATH_CALUDE_browns_children_divisibility_l2833_283359

theorem browns_children_divisibility : 
  ∃! n : Nat, n ∈ Finset.range 10 ∧ ¬(7773 % (n + 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_browns_children_divisibility_l2833_283359


namespace NUMINAMATH_CALUDE_jim_age_in_two_years_l2833_283304

theorem jim_age_in_two_years :
  let tom_age_five_years_ago : ℕ := 32
  let years_since_tom_age : ℕ := 5
  let years_to_past_reference : ℕ := 7
  let jim_age_difference : ℕ := 5
  let years_to_future : ℕ := 2

  let tom_current_age : ℕ := tom_age_five_years_ago + years_since_tom_age
  let tom_age_at_reference : ℕ := tom_current_age - years_to_past_reference
  let jim_age_at_reference : ℕ := (tom_age_at_reference / 2) + jim_age_difference
  let jim_current_age : ℕ := jim_age_at_reference + years_to_past_reference
  let jim_future_age : ℕ := jim_current_age + years_to_future

  jim_future_age = 29 := by sorry

end NUMINAMATH_CALUDE_jim_age_in_two_years_l2833_283304


namespace NUMINAMATH_CALUDE_hall_ratio_l2833_283355

theorem hall_ratio (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width * length = 288 →
  length - width = 12 →
  width / length = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_hall_ratio_l2833_283355


namespace NUMINAMATH_CALUDE_expression_simplification_l2833_283391

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) (h3 : x ≠ -1) :
  (x + 1) / (x^2 - 2*x) / (1 + 1/x) = 1 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2833_283391


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l2833_283382

def collagen_protein_per_2_scoops : ℕ := 18
def protein_powder_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def arnold_consumption (collagen_scoops protein_scoops : ℕ) : ℕ :=
  (collagen_scoops * collagen_protein_per_2_scoops / 2) + 
  (protein_scoops * protein_powder_per_scoop) + 
  steak_protein

theorem arnold_protein_consumption : 
  arnold_consumption 1 1 = 86 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l2833_283382


namespace NUMINAMATH_CALUDE_standard_deviation_of_applicant_ages_l2833_283345

def average_age : ℕ := 10
def num_different_ages : ℕ := 17

theorem standard_deviation_of_applicant_ages :
  ∃ (s : ℕ),
    s > 0 ∧
    (average_age + s) - (average_age - s) + 1 = num_different_ages ∧
    s = 8 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_of_applicant_ages_l2833_283345


namespace NUMINAMATH_CALUDE_scientific_notation_141260_l2833_283383

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_141260 :
  toScientificNotation 141260 = ScientificNotation.mk 1.4126 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_141260_l2833_283383


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2833_283303

theorem sqrt_meaningful_range (m : ℝ) : 
  (∃ (x : ℝ), x^2 = m + 4) ↔ m ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2833_283303


namespace NUMINAMATH_CALUDE_tangent_from_cosine_central_angle_from_perimeter_and_area_l2833_283367

-- Part 1
theorem tangent_from_cosine (m : ℝ) (α : ℝ) :
  (m : ℝ) = -Real.sqrt 2 / 4 →
  Real.cos α = -1/3 →
  Real.tan α = -2 * Real.sqrt 2 :=
by sorry

-- Part 2
theorem central_angle_from_perimeter_and_area (r l : ℝ) :
  2 * r + l = 8 →
  1/2 * l * r = 3 →
  (l / r = 2/3 ∨ l / r = 6) :=
by sorry

end NUMINAMATH_CALUDE_tangent_from_cosine_central_angle_from_perimeter_and_area_l2833_283367
