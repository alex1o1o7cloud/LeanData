import Mathlib

namespace pictures_per_album_l2114_211408

theorem pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) 
  (h1 : total_pictures = 480) (h2 : num_albums = 24) :
  total_pictures / num_albums = 20 := by
  sorry

end pictures_per_album_l2114_211408


namespace smallest_divisible_by_20_and_36_l2114_211475

theorem smallest_divisible_by_20_and_36 : ∃ n : ℕ, n > 0 ∧ 20 ∣ n ∧ 36 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 20 ∣ m ∧ 36 ∣ m) → n ≤ m :=
by sorry

end smallest_divisible_by_20_and_36_l2114_211475


namespace ab_bc_ratio_l2114_211497

/-- A rectangle divided into five congruent rectangles -/
structure DividedRectangle where
  -- The width of each congruent rectangle
  x : ℝ
  -- Assumption that x is positive
  x_pos : x > 0

/-- The length of side AB in the divided rectangle -/
def length_AB (r : DividedRectangle) : ℝ := 5 * r.x

/-- The length of side BC in the divided rectangle -/
def length_BC (r : DividedRectangle) : ℝ := 3 * r.x

/-- Theorem stating that the ratio of AB to BC is 5:3 -/
theorem ab_bc_ratio (r : DividedRectangle) :
  length_AB r / length_BC r = 5 / 3 := by
  sorry

end ab_bc_ratio_l2114_211497


namespace f_is_quadratic_l2114_211499

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + x - 2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l2114_211499


namespace birds_in_tree_l2114_211424

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) : 
  initial_birds = initial_birds := by sorry

end birds_in_tree_l2114_211424


namespace subtraction_from_percentage_l2114_211415

theorem subtraction_from_percentage (n : ℝ) : n = 100 → (0.8 * n - 20 = 60) := by
  sorry

end subtraction_from_percentage_l2114_211415


namespace craigs_apples_l2114_211402

/-- 
Given:
- Craig's initial number of apples
- The number of apples Craig shares with Eugene
Prove that Craig's final number of apples is equal to the initial number minus the shared number.
-/
theorem craigs_apples (initial_apples shared_apples : ℕ) :
  initial_apples - shared_apples = initial_apples - shared_apples :=
by sorry

end craigs_apples_l2114_211402


namespace john_pill_payment_john_pays_54_dollars_l2114_211400

/-- The amount John pays for pills in a 30-day month, given the specified conditions. -/
theorem john_pill_payment (pills_per_day : ℕ) (cost_per_pill : ℚ) 
  (insurance_coverage_percent : ℚ) (days_in_month : ℕ) : ℚ :=
  let total_cost := (pills_per_day : ℚ) * cost_per_pill * days_in_month
  let insurance_coverage := total_cost * (insurance_coverage_percent / 100)
  total_cost - insurance_coverage

/-- Proof that John pays $54 for his pills in a 30-day month. -/
theorem john_pays_54_dollars : 
  john_pill_payment 2 (3/2) 40 30 = 54 := by
  sorry

end john_pill_payment_john_pays_54_dollars_l2114_211400


namespace circle_intersection_range_l2114_211442

theorem circle_intersection_range (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 ∧ x^2 + y^2 = 2) ↔ 
  (a < -1 ∧ a > -3) ∨ (a > 1 ∧ a < 3) :=
sorry

end circle_intersection_range_l2114_211442


namespace fourth_root_simplification_l2114_211494

theorem fourth_root_simplification (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (2^7 * 3^3 : ℚ)^(1/4) = a * b^(1/4) → a + b = 218 := by
  sorry

end fourth_root_simplification_l2114_211494


namespace triangle_properties_l2114_211419

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.c * Real.sin t.C = (2 * t.b + t.a) * Real.sin t.B + (2 * t.a - 3 * t.b) * Real.sin t.A)
  (h2 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)
  (h3 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h4 : t.A + t.B + t.C = π) : 
  (t.C = π / 3) ∧ 
  (t.c = 4 → 4 < t.a + t.b ∧ t.a + t.b ≤ 8) :=
sorry


end triangle_properties_l2114_211419


namespace jimmy_cards_ratio_l2114_211458

def jimmy_cards_problem (initial_cards : ℕ) (cards_to_bob : ℕ) (cards_left : ℕ) : Prop :=
  let cards_to_mary := initial_cards - cards_left - cards_to_bob
  (cards_to_mary : ℚ) / cards_to_bob = 2 / 1

theorem jimmy_cards_ratio : jimmy_cards_problem 18 3 9 := by
  sorry

end jimmy_cards_ratio_l2114_211458


namespace different_subject_book_choices_l2114_211483

def chinese_books : ℕ := 8
def math_books : ℕ := 6
def english_books : ℕ := 5

theorem different_subject_book_choices :
  chinese_books * math_books + 
  chinese_books * english_books + 
  math_books * english_books = 118 := by
  sorry

end different_subject_book_choices_l2114_211483


namespace travel_distance_l2114_211428

theorem travel_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 75 → time = 4 → distance = speed * time → distance = 300 := by
  sorry

end travel_distance_l2114_211428


namespace shaded_area_of_grid_square_l2114_211438

theorem shaded_area_of_grid_square (d : ℝ) (h1 : d = 10) : 
  let s := d / Real.sqrt 2
  let small_square_side := s / 5
  let small_square_area := small_square_side ^ 2
  let total_area := 25 * small_square_area
  total_area = 50 := by sorry

end shaded_area_of_grid_square_l2114_211438


namespace systematic_sample_theorem_l2114_211427

/-- Represents a systematic sample of bottles -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  start : Nat
  step : Nat

/-- Generates the sample numbers for a systematic sample -/
def generate_sample (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.step)

/-- Theorem: The systematic sample for 60 bottles with 6 selections starts at 3 with step 10 -/
theorem systematic_sample_theorem :
  let s : SystematicSample := ⟨60, 6, 3, 10⟩
  generate_sample s = [3, 13, 23, 33, 43, 53] := by sorry

end systematic_sample_theorem_l2114_211427


namespace mary_remaining_sheep_l2114_211405

def initial_sheep : ℕ := 400

def sheep_after_sister (initial : ℕ) : ℕ :=
  initial - (initial / 4)

def sheep_after_brother (after_sister : ℕ) : ℕ :=
  after_sister - (after_sister / 2)

theorem mary_remaining_sheep :
  sheep_after_brother (sheep_after_sister initial_sheep) = 150 := by
  sorry

end mary_remaining_sheep_l2114_211405


namespace quotient_problem_l2114_211476

theorem quotient_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ n : ℤ, a / b = n) (h4 : a / b = a / 2 ∨ a / b = 6 * b) : 
  a / b = 12 := by
sorry

end quotient_problem_l2114_211476


namespace dining_bill_calculation_l2114_211451

theorem dining_bill_calculation (number_of_people : ℕ) (individual_payment : ℚ) (tip_percentage : ℚ) 
  (h1 : number_of_people = 6)
  (h2 : individual_payment = 25.48)
  (h3 : tip_percentage = 0.10) :
  (number_of_people : ℚ) * individual_payment / (1 + tip_percentage) = 139.89 := by
  sorry

end dining_bill_calculation_l2114_211451


namespace ellipse_k_range_l2114_211421

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse (k : ℝ) where
  eq : ∀ (x y : ℝ), x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the foci condition

/-- The range of k for an ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
theorem ellipse_k_range (k : ℝ) (e : Ellipse k) : 0 < k ∧ k < 1 := by
  sorry

end ellipse_k_range_l2114_211421


namespace masha_result_non_negative_l2114_211420

theorem masha_result_non_negative (a b c d : ℝ) 
  (sum_eq_prod : a + b = c * d) 
  (prod_eq_sum : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := by
sorry

end masha_result_non_negative_l2114_211420


namespace min_draws_for_all_colors_l2114_211454

theorem min_draws_for_all_colors (white black yellow : ℕ) 
  (hw : white = 8) (hb : black = 9) (hy : yellow = 7) :
  (white + black + yellow - (white + black - 1)) = 18 := by
  sorry

end min_draws_for_all_colors_l2114_211454


namespace interior_triangle_perimeter_is_715_l2114_211480

/-- Triangle ABC with parallel lines forming interior triangle XYZ -/
structure ParallelLineTriangle where
  /-- Side length of AB -/
  ab : ℝ
  /-- Side length of BC -/
  bc : ℝ
  /-- Side length of AC -/
  ac : ℝ
  /-- Length of intersection of ℓA with interior of triangle ABC -/
  ℓa_intersection : ℝ
  /-- Length of intersection of ℓB with interior of triangle ABC -/
  ℓb_intersection : ℝ
  /-- Length of intersection of ℓC with interior of triangle ABC -/
  ℓc_intersection : ℝ

/-- Perimeter of the interior triangle XYZ formed by lines ℓA, ℓB, and ℓC -/
def interior_triangle_perimeter (t : ParallelLineTriangle) : ℝ := sorry

/-- Theorem stating that the perimeter of the interior triangle is 715 for the given conditions -/
theorem interior_triangle_perimeter_is_715 (t : ParallelLineTriangle) 
  (h1 : t.ab = 120)
  (h2 : t.bc = 220)
  (h3 : t.ac = 180)
  (h4 : t.ℓa_intersection = 55)
  (h5 : t.ℓb_intersection = 45)
  (h6 : t.ℓc_intersection = 15) :
  interior_triangle_perimeter t = 715 := by sorry

end interior_triangle_perimeter_is_715_l2114_211480


namespace abc_value_l2114_211492

def is_valid_abc (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > b ∧ b > c ∧
  (10 * a + b) + (10 * b + a) = 55 ∧
  1300 < 222 * (a + b + c) ∧ 222 * (a + b + c) < 1400

theorem abc_value :
  ∀ a b c : ℕ, is_valid_abc a b c → a = 3 ∧ b = 2 ∧ c = 1 :=
sorry

end abc_value_l2114_211492


namespace horner_method_v3_l2114_211401

def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := a*x + v

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 5
  let v1 := horner_step 2 x v0
  let v2 := horner_step 3.5 x v1
  horner_step (-2.6) x v2

theorem horner_method_v3 :
  horner_v3 1 = 7.9 :=
sorry

end horner_method_v3_l2114_211401


namespace thirtieth_term_is_59_l2114_211486

/-- A sequence where each term is 2 more than the previous term, starting with 1 -/
def counting_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => counting_sequence n + 2

/-- The 30th term of the counting sequence is 59 -/
theorem thirtieth_term_is_59 : counting_sequence 29 = 59 := by
  sorry

end thirtieth_term_is_59_l2114_211486


namespace conference_trip_distance_l2114_211450

/-- Conference Trip Problem -/
theorem conference_trip_distance :
  ∀ (d : ℝ) (t : ℝ),
    -- Initial speed
    let v₁ : ℝ := 40
    -- Speed increase
    let v₂ : ℝ := 20
    -- Time late if continued at initial speed
    let t_late : ℝ := 0.75
    -- Time early with speed increase
    let t_early : ℝ := 0.25
    -- Distance equation at initial speed
    d = v₁ * (t + t_late) →
    -- Distance equation with speed increase
    d - v₁ = (v₁ + v₂) * (t - 1 - t_early) →
    -- Conclusion: distance is 160 miles
    d = 160 := by
  sorry

end conference_trip_distance_l2114_211450


namespace student_number_problem_l2114_211489

theorem student_number_problem (x : ℝ) : (7 * x - 150 = 130) → x = 40 := by
  sorry

end student_number_problem_l2114_211489


namespace jons_laundry_loads_l2114_211441

/-- Represents the laundry machine and Jon's clothes -/
structure LaundryProblem where
  machine_capacity : ℝ
  shirt_weight : ℝ
  pants_weight : ℝ
  sock_weight : ℝ
  jacket_weight : ℝ
  shirt_count : ℕ
  pants_count : ℕ
  sock_count : ℕ
  jacket_count : ℕ

/-- Calculates the minimum number of loads required -/
def minimum_loads (problem : LaundryProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of loads for Jon's laundry is 5 -/
theorem jons_laundry_loads :
  let problem : LaundryProblem :=
    { machine_capacity := 8
    , shirt_weight := 1/4
    , pants_weight := 1/2
    , sock_weight := 1/6
    , jacket_weight := 2
    , shirt_count := 20
    , pants_count := 20
    , sock_count := 18
    , jacket_count := 6
    }
  minimum_loads problem = 5 := by
  sorry

end jons_laundry_loads_l2114_211441


namespace terminal_side_in_third_quadrant_l2114_211434

/-- Given a point P with coordinates (cosθ, tanθ) in the second quadrant,
    prove that the terminal side of angle θ is in the third quadrant. -/
theorem terminal_side_in_third_quadrant (θ : Real) :
  (cosθ < 0 ∧ tanθ > 0) →  -- Point P is in the second quadrant
  (cosθ < 0 ∧ sinθ < 0)    -- Terminal side of θ is in the third quadrant
:= by sorry

end terminal_side_in_third_quadrant_l2114_211434


namespace vector_parallel_and_dot_product_l2114_211435

/-- Given two vectors a and b, and an angle α, prove the following statements -/
theorem vector_parallel_and_dot_product (α : Real) 
    (h1 : α ∈ Set.Ioo 0 (π/4)) 
    (a : Fin 2 → Real) (b : Fin 2 → Real)
    (h2 : a = λ i => if i = 0 then 2 * Real.sin α else 1)
    (h3 : b = λ i => if i = 0 then Real.cos α else 1) :
  (∃ (k : Real), a = k • b → Real.tan α = 1/2) ∧
  (a • b = 9/5 → Real.sin (2*α + π/4) = 7*Real.sqrt 2/10) := by
  sorry

end vector_parallel_and_dot_product_l2114_211435


namespace prob_sum_eight_two_dice_l2114_211413

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 5

theorem prob_sum_eight_two_dice : 
  (favorable_outcomes : ℚ) / dice_outcomes = 5 / 36 := by sorry

end prob_sum_eight_two_dice_l2114_211413


namespace norbs_age_l2114_211456

def guesses : List Nat := [24, 28, 30, 32, 36, 38, 41, 44, 47, 49]

def is_prime (n : Nat) : Prop := Nat.Prime n

def at_least_half_too_low (age : Nat) : Prop :=
  (guesses.filter (· < age)).length ≥ guesses.length / 2

def two_off_by_one (age : Nat) : Prop :=
  (guesses.filter (λ g => g = age - 1 ∨ g = age + 1)).length = 2

theorem norbs_age :
  ∃! age : Nat,
    age ∈ guesses ∧
    is_prime age ∧
    at_least_half_too_low age ∧
    two_off_by_one age ∧
    age = 37 :=
sorry

end norbs_age_l2114_211456


namespace quadratic_equation_solution_l2114_211457

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = (2 + Real.sqrt 14) / 2 ∧ 
  x₂ = (2 - Real.sqrt 14) / 2 ∧ 
  2 * x₁^2 - 4 * x₁ - 5 = 0 ∧ 
  2 * x₂^2 - 4 * x₂ - 5 = 0 := by
  sorry

end quadratic_equation_solution_l2114_211457


namespace basketball_handshakes_l2114_211490

theorem basketball_handshakes :
  let team_size : ℕ := 5
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let inter_team_handshakes := team_size * team_size
  let player_referee_handshakes := (team_size * num_teams) * num_referees
  inter_team_handshakes + player_referee_handshakes = 55 :=
by sorry

end basketball_handshakes_l2114_211490


namespace sum_pqr_values_l2114_211436

theorem sum_pqr_values (p q r : ℝ) (distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (eq1 : q = p * (4 - p)) (eq2 : r = q * (4 - q)) (eq3 : p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end sum_pqr_values_l2114_211436


namespace grants_room_count_l2114_211462

def danielles_rooms : ℕ := 6

def heidis_rooms (danielles_rooms : ℕ) : ℕ := 3 * danielles_rooms

def grants_rooms (heidis_rooms : ℕ) : ℚ := (1 : ℚ) / 9 * heidis_rooms

theorem grants_room_count :
  grants_rooms (heidis_rooms danielles_rooms) = 2 := by
  sorry

end grants_room_count_l2114_211462


namespace sum_of_squares_of_roots_l2114_211467

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 16*x + 15 = 0 → ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 226 ∧ (x = s₁ ∨ x = s₂) :=
by sorry

end sum_of_squares_of_roots_l2114_211467


namespace tan_70_cos_10_sqrt_3_tan_20_minus_1_l2114_211432

theorem tan_70_cos_10_sqrt_3_tan_20_minus_1 : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_70_cos_10_sqrt_3_tan_20_minus_1_l2114_211432


namespace train_speed_l2114_211491

-- Define the length of the train in meters
def train_length : ℝ := 130

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 3.249740020798336

-- Define the conversion factor from m/s to km/hr
def ms_to_kmhr : ℝ := 3.6

-- Theorem to prove the train's speed
theorem train_speed : 
  (train_length / crossing_time) * ms_to_kmhr = 144 := by
  sorry

end train_speed_l2114_211491


namespace line_passes_through_point_l2114_211418

/-- The line equation y = 2x - 1 passes through the point (0, -1) -/
theorem line_passes_through_point :
  let f : ℝ → ℝ := λ x => 2 * x - 1
  f 0 = -1 := by sorry

end line_passes_through_point_l2114_211418


namespace function_zero_points_theorem_l2114_211484

open Real

theorem function_zero_points_theorem (f : ℝ → ℝ) (a : ℝ) (x₁ x₂ : ℝ) 
  (h_f : ∀ x, f x = log x - a * x)
  (h_zero : f x₁ = 0 ∧ f x₂ = 0)
  (h_distinct : x₁ < x₂) :
  (0 < a ∧ a < 1 / Real.exp 1) ∧ 
  (2 / (x₁ + x₂) < a) := by
  sorry

end function_zero_points_theorem_l2114_211484


namespace bookstore_shipment_size_l2114_211453

theorem bookstore_shipment_size (displayed_percentage : ℚ) (stored_amount : ℕ) : 
  displayed_percentage = 1/4 →
  stored_amount = 225 →
  ∃ total : ℕ, total = 300 ∧ (1 - displayed_percentage) * total = stored_amount :=
by
  sorry

end bookstore_shipment_size_l2114_211453


namespace proportional_increase_l2114_211407

/-- Given the equation 3x - 2y = 7, this theorem proves that y increases proportionally to x
    and determines the proportionality coefficient. -/
theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) :
  ∃ (k b : ℝ), y = k * x + b ∧ k = 3 / 2 := by
  sorry

end proportional_increase_l2114_211407


namespace nested_root_equality_l2114_211461

theorem nested_root_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 7) ^ (1 / 4) :=
by sorry

end nested_root_equality_l2114_211461


namespace third_element_in_tenth_bracket_l2114_211449

/-- The number of elements in the nth bracket -/
def bracket_size (n : ℕ) : ℕ := n

/-- The sum of elements in the first n brackets -/
def sum_bracket_sizes (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last element in the nth bracket -/
def last_element_in_bracket (n : ℕ) : ℕ := sum_bracket_sizes n

theorem third_element_in_tenth_bracket :
  ∃ (k : ℕ), k = last_element_in_bracket 9 + 3 ∧ k = 48 :=
sorry

end third_element_in_tenth_bracket_l2114_211449


namespace grid_solution_l2114_211403

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two cells are adjacent in the grid -/
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ l.val + 1 = j.val)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ k.val + 1 = i.val))

/-- The sum of any two numbers in adjacent cells is less than 12 -/
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → (g i j).val + (g k l).val < 12

/-- The given positions of known numbers in the grid -/
def known_positions (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7 ∧ g 0 2 = 9

/-- The theorem to be proved -/
theorem grid_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : known_positions g) : 
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end grid_solution_l2114_211403


namespace system_of_equations_solution_l2114_211439

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) :=
by
  -- Proof goes here
  sorry

end system_of_equations_solution_l2114_211439


namespace weighted_average_score_l2114_211410

-- Define the scores and weights
def interview_score : ℕ := 90
def computer_score : ℕ := 85
def design_score : ℕ := 80

def interview_weight : ℕ := 5
def computer_weight : ℕ := 2
def design_weight : ℕ := 3

-- Define the total weighted score
def total_weighted_score : ℕ := 
  interview_score * interview_weight + 
  computer_score * computer_weight + 
  design_score * design_weight

-- Define the sum of weights
def sum_of_weights : ℕ := 
  interview_weight + computer_weight + design_weight

-- Theorem to prove
theorem weighted_average_score : 
  total_weighted_score / sum_of_weights = 86 := by
  sorry

end weighted_average_score_l2114_211410


namespace f_decreasing_when_a_negative_l2114_211445

-- Define the function f(x) = ax^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- Theorem statement
theorem f_decreasing_when_a_negative (a : ℝ) (h1 : a ≠ 0) (h2 : a < 0) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
by
  sorry

end f_decreasing_when_a_negative_l2114_211445


namespace rationalize_denominator_l2114_211448

theorem rationalize_denominator : 
  (35 - Real.sqrt 35) / Real.sqrt 35 = Real.sqrt 35 - 1 := by
sorry

end rationalize_denominator_l2114_211448


namespace murtha_pebble_collection_l2114_211481

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebble_collection : arithmetic_sum 1 2 12 = 144 := by
  sorry

end murtha_pebble_collection_l2114_211481


namespace sin_330_degrees_l2114_211425

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end sin_330_degrees_l2114_211425


namespace quadratic_equation_solution_l2114_211465

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 6 * x
  (f 0 = 0 ∧ f (3/2) = 0) ∧
  ∀ x : ℝ, f x = 0 → (x = 0 ∨ x = 3/2) :=
by sorry

end quadratic_equation_solution_l2114_211465


namespace line_translation_proof_l2114_211423

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The vertical translation distance between two lines with the same slope -/
def verticalTranslation (l1 l2 : Line) : ℝ :=
  l2.yIntercept - l1.yIntercept

theorem line_translation_proof (l1 l2 : Line) 
  (h1 : l1.slope = 3 ∧ l1.yIntercept = -1)
  (h2 : l2.slope = 3 ∧ l2.yIntercept = 6)
  : verticalTranslation l1 l2 = 7 := by
  sorry

end line_translation_proof_l2114_211423


namespace assignments_for_twenty_points_l2114_211468

/-- Calculates the number of assignments required for a given number of points -/
def assignments_required (points : ℕ) : ℕ :=
  let segments := (points + 3) / 4
  (segments * (segments + 1) * 2) 

/-- The theorem stating that 60 assignments are required for 20 points -/
theorem assignments_for_twenty_points :
  assignments_required 20 = 60 := by
  sorry

end assignments_for_twenty_points_l2114_211468


namespace race_length_for_simultaneous_finish_l2114_211455

theorem race_length_for_simultaneous_finish 
  (speed_ratio : ℝ) 
  (head_start : ℝ) 
  (race_length : ℝ) : 
  speed_ratio = 4 →
  head_start = 63 →
  race_length / speed_ratio = (race_length - head_start) / 1 →
  race_length = 84 := by
sorry

end race_length_for_simultaneous_finish_l2114_211455


namespace problems_left_to_grade_l2114_211487

theorem problems_left_to_grade (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) :
  total_worksheets = 14 →
  problems_per_worksheet = 2 →
  graded_worksheets = 7 →
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by sorry

end problems_left_to_grade_l2114_211487


namespace angle_B_is_pi_over_four_l2114_211431

/-- Given a triangle ABC with circumradius R, if 2R(sin²A - sin²B) = (√2a - c)sinC, 
    then the measure of angle B is π/4. -/
theorem angle_B_is_pi_over_four 
  (A B C : ℝ) 
  (a b c R : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) 
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h6 : 0 < R) 
  (h7 : a = 2 * R * Real.sin A) 
  (h8 : b = 2 * R * Real.sin B) 
  (h9 : c = 2 * R * Real.sin C) 
  (h10 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = π / 4 := by
sorry

end angle_B_is_pi_over_four_l2114_211431


namespace max_reverse_sum_theorem_l2114_211477

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem max_reverse_sum_theorem (a b : ℕ) 
  (h1 : is_three_digit a) 
  (h2 : is_three_digit b) 
  (h3 : a % 10 ≠ 0) 
  (h4 : b % 10 ≠ 0) 
  (h5 : a + b = 1372) : 
  ∃ (max : ℕ), reverse_number a + reverse_number b ≤ max ∧ max = 1372 := by
  sorry

end max_reverse_sum_theorem_l2114_211477


namespace exists_valid_set_l2114_211430

def is_valid_set (S : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (n ∈ S ↔ 
    (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = n) ∨
    (∃ a b : ℕ, a ∉ S ∧ b ∉ S ∧ a ≠ b ∧ a > 0 ∧ b > 0 ∧ a + b = n))

theorem exists_valid_set : ∃ S : Set ℕ, is_valid_set S := by
  sorry

end exists_valid_set_l2114_211430


namespace percent_problem_l2114_211493

theorem percent_problem : ∃ x : ℝ, (1 / 100) * x = 123.56 ∧ x = 12356 := by
  sorry

end percent_problem_l2114_211493


namespace class_size_possibilities_l2114_211485

theorem class_size_possibilities (N : ℕ) : 
  (∃ k : ℕ, N = 8 + k) →  -- Total students is 8 bullies plus some honor students
  (7 : ℚ) / (N - 1 : ℚ) < (1 : ℚ) / 3 →  -- Bullies' condition
  (8 : ℚ) / (N - 1 : ℚ) ≥ (1 : ℚ) / 3 →  -- Honor students' condition
  N ∈ ({23, 24, 25} : Set ℕ) :=
by sorry

end class_size_possibilities_l2114_211485


namespace fraction_simplification_l2114_211406

theorem fraction_simplification (a : ℝ) (h : a^2 ≠ 9) :
  3 / (a^2 - 9) - a / (9 - a^2) = 1 / (a - 3) := by
  sorry

end fraction_simplification_l2114_211406


namespace line_through_point_parallel_to_line_l2114_211473

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (l : Line) 
  (p : Point) 
  (given_line : Line) :
  l.a = 1 ∧ l.b = 3 ∧ l.c = -2 →
  p.x = -1 ∧ p.y = 1 →
  given_line.a = 1 ∧ given_line.b = 3 ∧ given_line.c = 4 →
  p.liesOn l ∧ l.isParallelTo given_line :=
by sorry

end line_through_point_parallel_to_line_l2114_211473


namespace find_divisor_l2114_211433

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 23 + remainder) :
  dividend = 997 → quotient = 43 → remainder = 8 → 23 = dividend / quotient :=
by sorry

end find_divisor_l2114_211433


namespace bill_calculation_l2114_211411

def original_bill : ℝ := 500

def first_late_charge_rate : ℝ := 0.02

def second_late_charge_rate : ℝ := 0.03

def final_bill_amount : ℝ := original_bill * (1 + first_late_charge_rate) * (1 + second_late_charge_rate)

theorem bill_calculation :
  final_bill_amount = 525.30 := by
  sorry

end bill_calculation_l2114_211411


namespace bridge_length_proof_l2114_211417

/-- The length of a bridge given train parameters -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 275 :=
by
  sorry

end bridge_length_proof_l2114_211417


namespace second_hose_spray_rate_l2114_211452

/-- Calculates the spray rate of the second hose needed to fill a pool --/
theorem second_hose_spray_rate 
  (pool_capacity : ℝ) 
  (first_hose_rate : ℝ) 
  (total_time : ℝ) 
  (second_hose_time : ℝ) 
  (h1 : pool_capacity = 390)
  (h2 : first_hose_rate = 50)
  (h3 : total_time = 5)
  (h4 : second_hose_time = 2)
  : ∃ (second_hose_rate : ℝ), 
    second_hose_rate * second_hose_time + first_hose_rate * total_time = pool_capacity ∧ 
    second_hose_rate = 20 := by
  sorry

end second_hose_spray_rate_l2114_211452


namespace min_distance_theorem_l2114_211469

theorem min_distance_theorem (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ - a)^2 + (Real.log (x₀^2) - 2*a)^2 ≤ 4/5) →
  a = 1/5 := by
  sorry

end min_distance_theorem_l2114_211469


namespace fruit_garden_ratio_l2114_211479

/-- Given a garden with the specified conditions, prove the ratio of fruit section to whole garden --/
theorem fruit_garden_ratio 
  (total_area : ℝ) 
  (fruit_quarter : ℝ) 
  (h1 : total_area = 64) 
  (h2 : fruit_quarter = 8) : 
  (4 * fruit_quarter) / total_area = 1 / 2 := by
  sorry

end fruit_garden_ratio_l2114_211479


namespace john_earnings_calculation_l2114_211464

/-- Calculates John's weekly earnings after fees and taxes --/
def johnWeeklyEarnings : ℝ :=
  let streamingHours : ℕ := 4
  let mondayRate : ℝ := 10
  let wednesdayRate : ℝ := 12
  let fridayRate : ℝ := 15
  let saturdayRate : ℝ := 20
  let platformFeeRate : ℝ := 0.20
  let taxRate : ℝ := 0.25

  let grossEarnings : ℝ := streamingHours * (mondayRate + wednesdayRate + fridayRate + saturdayRate)
  let platformFee : ℝ := grossEarnings * platformFeeRate
  let netEarningsBeforeTax : ℝ := grossEarnings - platformFee
  let tax : ℝ := netEarningsBeforeTax * taxRate
  netEarningsBeforeTax - tax

theorem john_earnings_calculation :
  johnWeeklyEarnings = 136.80 := by sorry

end john_earnings_calculation_l2114_211464


namespace solution_satisfies_system_l2114_211478

theorem solution_satisfies_system :
  let f (x y : ℝ) := x + y + 2 - 4*x*y
  ∀ (x y z : ℝ), 
    (f x y = 0 ∧ f y z = 0 ∧ f z x = 0) →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by sorry

end solution_satisfies_system_l2114_211478


namespace power_multiplication_l2114_211459

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end power_multiplication_l2114_211459


namespace harriet_return_speed_harriet_return_speed_approx_145_l2114_211404

/-- Calculates the return speed given the conditions of Harriet's trip -/
theorem harriet_return_speed (outbound_speed : ℝ) (total_time : ℝ) (outbound_time_minutes : ℝ) : ℝ :=
  let outbound_time : ℝ := outbound_time_minutes / 60
  let distance : ℝ := outbound_speed * outbound_time
  let return_time : ℝ := total_time - outbound_time
  distance / return_time

/-- Proves that Harriet's return speed is approximately 145 km/h -/
theorem harriet_return_speed_approx_145 :
  ∃ ε > 0, abs (harriet_return_speed 105 5 174 - 145) < ε :=
sorry

end harriet_return_speed_harriet_return_speed_approx_145_l2114_211404


namespace swap_counts_correct_l2114_211440

/-- Represents a circular sequence of letters -/
def CircularSequence := List Char

/-- Counts the minimum number of adjacent swaps needed to transform one sequence into another -/
def minAdjacentSwaps (seq1 seq2 : CircularSequence) : Nat :=
  sorry

/-- Counts the minimum number of arbitrary swaps needed to transform one sequence into another -/
def minArbitrarySwaps (seq1 seq2 : CircularSequence) : Nat :=
  sorry

/-- The two given sequences -/
def sequence1 : CircularSequence := ['A', 'Z', 'O', 'R', 'S', 'Z', 'Á', 'G', 'H', 'Á', 'Z', 'A']
def sequence2 : CircularSequence := ['S', 'Á', 'R', 'G', 'A', 'A', 'Z', 'H', 'O', 'Z', 'Z', 'Ā']

theorem swap_counts_correct :
  minAdjacentSwaps sequence1 sequence2 = 14 ∧
  minArbitrarySwaps sequence1 sequence2 = 4 := by
  sorry

end swap_counts_correct_l2114_211440


namespace eighth_fibonacci_is_21_l2114_211474

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem eighth_fibonacci_is_21 : fibonacci 7 = 21 := by
  sorry

end eighth_fibonacci_is_21_l2114_211474


namespace smallest_common_multiple_of_10_and_6_l2114_211495

theorem smallest_common_multiple_of_10_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, (10 ∣ m) ∧ (6 ∣ m) → n ≤ m) ∧ (10 ∣ n) ∧ (6 ∣ n) := by
  sorry

end smallest_common_multiple_of_10_and_6_l2114_211495


namespace car_rental_cost_per_mile_l2114_211446

/-- Represents a car rental plan with an initial fee and a per-mile cost. -/
structure RentalPlan where
  initialFee : ℝ
  costPerMile : ℝ

/-- The total cost of a rental plan for a given number of miles. -/
def totalCost (plan : RentalPlan) (miles : ℝ) : ℝ :=
  plan.initialFee + plan.costPerMile * miles

theorem car_rental_cost_per_mile :
  let plan1 : RentalPlan := { initialFee := 65, costPerMile := x }
  let plan2 : RentalPlan := { initialFee := 0, costPerMile := 0.60 }
  let miles : ℝ := 325
  totalCost plan1 miles = totalCost plan2 miles →
  x = 0.40 := by
sorry

end car_rental_cost_per_mile_l2114_211446


namespace sum_of_specific_repeating_decimals_l2114_211422

/-- Represents a repeating decimal with a repeating part and a period -/
def RepeatingDecimal (repeating_part : ℕ) (period : ℕ) : ℚ :=
  repeating_part / (10^period - 1)

/-- The sum of three specific repeating decimals -/
theorem sum_of_specific_repeating_decimals :
  RepeatingDecimal 12 2 + RepeatingDecimal 34 3 + RepeatingDecimal 567 5 = 16133 / 99999 := by
  sorry

#eval RepeatingDecimal 12 2 + RepeatingDecimal 34 3 + RepeatingDecimal 567 5

end sum_of_specific_repeating_decimals_l2114_211422


namespace problem_solution_l2114_211463

theorem problem_solution (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -1) : 
  (3 * a + a * b + 3 * b = 5) ∧ (a^2 + b^2 = 6) := by
  sorry

end problem_solution_l2114_211463


namespace photos_to_cover_poster_l2114_211443

def poster_length : ℕ := 3
def poster_width : ℕ := 5
def photo_length : ℕ := 3
def photo_width : ℕ := 5
def inches_per_foot : ℕ := 12

theorem photos_to_cover_poster :
  (poster_length * inches_per_foot * poster_width * inches_per_foot) / (photo_length * photo_width) = 144 := by
  sorry

end photos_to_cover_poster_l2114_211443


namespace arctan_equation_solution_l2114_211429

theorem arctan_equation_solution :
  ∃ y : ℝ, 2 * Real.arctan (1/5) + 2 * Real.arctan (1/25) + Real.arctan (1/y) = π/4 ∧ y = 1 := by
  sorry

end arctan_equation_solution_l2114_211429


namespace round_trip_speed_l2114_211426

/-- Proves that given specific conditions for a round trip, the return speed must be 48 mph -/
theorem round_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) (speed_ba : ℝ) : 
  distance = 120 →
  speed_ab = 80 →
  avg_speed = 60 →
  (2 * distance) / (distance / speed_ab + distance / speed_ba) = avg_speed →
  speed_ba = 48 := by
sorry

end round_trip_speed_l2114_211426


namespace binomial_coefficient_fifth_power_fourth_term_l2114_211466

theorem binomial_coefficient_fifth_power_fourth_term : 
  Nat.choose 5 3 = 10 := by sorry

end binomial_coefficient_fifth_power_fourth_term_l2114_211466


namespace system_solution_l2114_211460

theorem system_solution (x y : ℝ) : 
  x^3 + y^3 = 7 ∧ x*y*(x + y) = -2 ↔ (x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) :=
by sorry

end system_solution_l2114_211460


namespace equal_power_implies_equal_l2114_211482

theorem equal_power_implies_equal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 0 < a) (h4 : a < 1) (h5 : a^b = b^a) : a = b := by
  sorry

end equal_power_implies_equal_l2114_211482


namespace mom_t_shirt_purchase_l2114_211416

/-- The number of packages of white t-shirts Mom bought -/
def num_packages : ℕ := 14

/-- The number of white t-shirts in each package -/
def t_shirts_per_package : ℕ := 5

/-- The total number of white t-shirts Mom bought -/
def total_t_shirts : ℕ := num_packages * t_shirts_per_package

theorem mom_t_shirt_purchase : total_t_shirts = 70 := by
  sorry

end mom_t_shirt_purchase_l2114_211416


namespace hash_four_neg_three_l2114_211470

-- Define the # operation
def hash (x y : Int) : Int := x * (y + 2) + 2 * x * y

-- Theorem statement
theorem hash_four_neg_three : hash 4 (-3) = -28 := by
  sorry

end hash_four_neg_three_l2114_211470


namespace symmetric_point_y_axis_l2114_211444

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point with respect to the y-axis --/
def symmetricYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The original point (2,5) --/
def originalPoint : Point :=
  { x := 2, y := 5 }

/-- The expected symmetric point (-2,5) --/
def expectedSymmetricPoint : Point :=
  { x := -2, y := 5 }

theorem symmetric_point_y_axis :
  symmetricYAxis originalPoint = expectedSymmetricPoint := by
  sorry

end symmetric_point_y_axis_l2114_211444


namespace valid_arrangement_iff_odd_l2114_211471

/-- A permutation of numbers from 1 to n -/
def OuterRingPermutation (n : ℕ) := Fin n → Fin n

/-- Checks if a permutation satisfies the rotation property -/
def SatisfiesRotationProperty (n : ℕ) (p : OuterRingPermutation n) : Prop :=
  ∀ k : Fin n, ∃! j : Fin n, (p j - j : ℤ) ≡ k [ZMOD n]

/-- The main theorem: a valid arrangement exists if and only if n is odd -/
theorem valid_arrangement_iff_odd (n : ℕ) (h : n ≥ 3) :
  (∃ p : OuterRingPermutation n, SatisfiesRotationProperty n p) ↔ Odd n :=
sorry

end valid_arrangement_iff_odd_l2114_211471


namespace least_three_digit_with_digit_product_12_l2114_211437

/-- A function that returns the product of the digits of a three-digit number -/
def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

/-- A predicate that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
by sorry

end least_three_digit_with_digit_product_12_l2114_211437


namespace museum_exhibit_count_l2114_211412

def base5ToBase10 (n : ℕ) : ℕ := sorry

theorem museum_exhibit_count : 
  let clay_tablets := base5ToBase10 1432
  let bronze_sculptures := base5ToBase10 2041
  let stone_carvings := base5ToBase10 232
  clay_tablets + bronze_sculptures + stone_carvings = 580 := by sorry

end museum_exhibit_count_l2114_211412


namespace number_of_students_l2114_211498

theorem number_of_students (initial_avg : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (correct_avg : ℝ) :
  initial_avg = 100 →
  wrong_mark = 60 →
  correct_mark = 10 →
  correct_avg = 95 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * correct_avg = n * initial_avg - (wrong_mark - correct_mark) ∧ n = 10 :=
by sorry

end number_of_students_l2114_211498


namespace expression_simplification_l2114_211447

theorem expression_simplification (a : ℝ) (h : a^2 - a - 2 = 0) :
  (1 + 1/a) / ((a^2 - 1)/a) - (2*a - 2)/(a^2 - 2*a + 1) = -1 :=
by sorry

end expression_simplification_l2114_211447


namespace kevin_ran_17_miles_l2114_211472

/-- Calculates the total distance Kevin ran given his running segments -/
def kevin_total_distance (speed1 speed2 speed3 : ℝ) (time1 time2 time3 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3

/-- Theorem stating that Kevin's total distance is 17 miles -/
theorem kevin_ran_17_miles :
  kevin_total_distance 10 20 8 0.5 0.5 0.25 = 17 := by
  sorry

#eval kevin_total_distance 10 20 8 0.5 0.5 0.25

end kevin_ran_17_miles_l2114_211472


namespace fraction_product_exponents_l2114_211409

theorem fraction_product_exponents : (3 / 4 : ℚ)^5 * (4 / 3 : ℚ)^2 = 8 / 19 := by
  sorry

end fraction_product_exponents_l2114_211409


namespace litter_count_sum_l2114_211414

theorem litter_count_sum : 
  let glass_bottles : ℕ := 25
  let aluminum_cans : ℕ := 18
  let plastic_bags : ℕ := 12
  let paper_cups : ℕ := 7
  let cigarette_packs : ℕ := 5
  let face_masks : ℕ := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + face_masks = 70 := by
  sorry

end litter_count_sum_l2114_211414


namespace reflection_theorem_l2114_211488

/-- Original function -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Reflection line -/
def reflection_line : ℝ := -2

/-- Resulting function after reflection -/
def g (x : ℝ) : ℝ := 2 * x + 9

/-- Theorem stating that g is the reflection of f across x = -2 -/
theorem reflection_theorem :
  ∀ x : ℝ, g (2 * reflection_line - x) = f x :=
sorry

end reflection_theorem_l2114_211488


namespace cube_root_scaling_l2114_211496

theorem cube_root_scaling (a b c d : ℝ) (ha : a > 0) (hc : c > 0) :
  (a^(1/3) = b) → (c^(1/3) = d) →
  ((1000 * a)^(1/3) = 10 * b) ∧ ((-0.001 * c)^(1/3) = -0.1 * d) := by
  sorry

/- The theorem above captures the essence of the problem without directly using the specific numbers.
   It shows the scaling properties of cube roots that are used to solve the original problem. -/

end cube_root_scaling_l2114_211496
