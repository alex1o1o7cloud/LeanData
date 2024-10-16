import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_property_l243_24366

theorem intersection_point_property (x : Real) : 
  x ∈ Set.Ioo 0 (π / 2) → 
  6 * Real.cos x = 9 * Real.tan x → 
  Real.sin x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_property_l243_24366


namespace NUMINAMATH_CALUDE_euler_line_l243_24311

/-- The centroid of a triangle -/
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The orthocenter of a triangle -/
def orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

theorem euler_line (A B C : ℝ × ℝ) : 
  collinear (centroid A B C) (orthocenter A B C) (circumcenter A B C) := by
  sorry

end NUMINAMATH_CALUDE_euler_line_l243_24311


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l243_24310

theorem roots_of_polynomial (x : ℝ) : 
  x^3 - 3*x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l243_24310


namespace NUMINAMATH_CALUDE_set_union_equality_l243_24391

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem set_union_equality : (Set.univ \ S) ∪ T = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_set_union_equality_l243_24391


namespace NUMINAMATH_CALUDE_count_common_divisors_60_108_l243_24337

/-- The number of positive integers that are divisors of both 60 and 108 -/
def commonDivisorCount : ℕ := 
  (Finset.filter (fun n => n ∣ 60 ∧ n ∣ 108) (Finset.range 109)).card

theorem count_common_divisors_60_108 : commonDivisorCount = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_60_108_l243_24337


namespace NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l243_24309

/-- The length of a rectangle with width 3 cm and area equal to a 3 cm square -/
theorem rectangle_length_equal_square_side : 
  ∀ (length : ℝ), 
  (3 : ℝ) * length = (3 : ℝ) * (3 : ℝ) → 
  length = (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l243_24309


namespace NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l243_24353

theorem polygon_with_equal_angle_sums (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 360 → n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_equal_angle_sums_l243_24353


namespace NUMINAMATH_CALUDE_hash_sum_plus_six_l243_24370

def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_sum_plus_six (a b : ℕ) (h : hash a b = 100) : (a + b) + 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_hash_sum_plus_six_l243_24370


namespace NUMINAMATH_CALUDE_fourth_power_sum_l243_24376

theorem fourth_power_sum (x y z : ℝ) 
  (h1 : x + y + z = 2) 
  (h2 : x^2 + y^2 + z^2 = 6) 
  (h3 : x^3 + y^3 + z^3 = 8) : 
  x^4 + y^4 + z^4 = 26 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l243_24376


namespace NUMINAMATH_CALUDE_sum_product_over_sum_squares_l243_24377

theorem sum_product_over_sum_squares (x y z : ℝ) (h1 : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h2 : x + y + z = 3) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = 9 / (2*(x^2 + y^2 + z^2)) - 1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_product_over_sum_squares_l243_24377


namespace NUMINAMATH_CALUDE_incoming_class_size_l243_24372

theorem incoming_class_size : ∃! n : ℕ, 
  0 < n ∧ n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 ∧ n = 418 := by
  sorry

end NUMINAMATH_CALUDE_incoming_class_size_l243_24372


namespace NUMINAMATH_CALUDE_sets_problem_l243_24369

-- Define the universe set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

theorem sets_problem (a : ℝ) :
  (((Set.compl A) ∪ (B a)) = U ↔ a ≤ 0) ∧
  ((A ∩ (B a)) = (B a) ↔ a ≥ (1/2)) := by
  sorry


end NUMINAMATH_CALUDE_sets_problem_l243_24369


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l243_24352

/-- Number of diagonals in a convex polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l243_24352


namespace NUMINAMATH_CALUDE_tim_remaining_seashells_l243_24389

def initial_seashells : ℕ := 679
def seashells_given_away : ℕ := 172

theorem tim_remaining_seashells : 
  initial_seashells - seashells_given_away = 507 := by sorry

end NUMINAMATH_CALUDE_tim_remaining_seashells_l243_24389


namespace NUMINAMATH_CALUDE_only_set_A_forms_triangle_l243_24320

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem only_set_A_forms_triangle :
  can_form_triangle 3 4 5 ∧
  ¬can_form_triangle 4 4 8 ∧
  ¬can_form_triangle 3 10 4 ∧
  ¬can_form_triangle 4 5 10 :=
sorry

end NUMINAMATH_CALUDE_only_set_A_forms_triangle_l243_24320


namespace NUMINAMATH_CALUDE_right_triangle_tan_b_l243_24378

theorem right_triangle_tan_b (A B C : ℝ) (h1 : C = π / 2) (h2 : Real.cos A = 3 / 5) : 
  Real.tan B = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_b_l243_24378


namespace NUMINAMATH_CALUDE_na2so4_formation_l243_24364

-- Define the chemical species
structure Chemical where
  name : String
  moles : ℚ

-- Define the reaction conditions
structure ReactionConditions where
  temperature : ℚ
  pressure : ℚ

-- Define the reaction
def reaction (h2so4 : Chemical) (naoh : Chemical) (hcl : Chemical) (koh : Chemical) (conditions : ReactionConditions) : Chemical :=
  { name := "Na2SO4", moles := 1 }

-- Theorem statement
theorem na2so4_formation
  (h2so4 : Chemical)
  (naoh : Chemical)
  (hcl : Chemical)
  (koh : Chemical)
  (conditions : ReactionConditions)
  (h_h2so4_moles : h2so4.moles = 1)
  (h_naoh_moles : naoh.moles = 2)
  (h_hcl_moles : hcl.moles = 1/2)
  (h_koh_moles : koh.moles = 1/2)
  (h_temperature : conditions.temperature = 25)
  (h_pressure : conditions.pressure = 1) :
  (reaction h2so4 naoh hcl koh conditions).moles = 1 := by
  sorry

end NUMINAMATH_CALUDE_na2so4_formation_l243_24364


namespace NUMINAMATH_CALUDE_timi_ears_count_l243_24350

-- Define the inhabitants
structure Inhabitant where
  name : String
  ears_seen : Nat

-- Define the problem setup
def zog_problem : List Inhabitant :=
  [{ name := "Imi", ears_seen := 8 },
   { name := "Dimi", ears_seen := 7 },
   { name := "Timi", ears_seen := 5 }]

-- Theorem: Timi has 5 ears
theorem timi_ears_count (problem : List Inhabitant) : 
  problem = zog_problem → 
  (problem.find? (fun i => i.name = "Timi")).map (fun i => 
    List.sum (problem.map (fun j => j.ears_seen)) / 2 - i.ears_seen) = some 5 := by
  sorry

end NUMINAMATH_CALUDE_timi_ears_count_l243_24350


namespace NUMINAMATH_CALUDE_backpack_cost_l243_24300

def wallet_cost : ℕ := 50
def sneakers_cost : ℕ := 100
def jeans_cost : ℕ := 50
def total_spent : ℕ := 450

def leonard_spent : ℕ := wallet_cost + 2 * sneakers_cost
def michael_jeans_cost : ℕ := 2 * jeans_cost

theorem backpack_cost (backpack_price : ℕ) :
  backpack_price = total_spent - (leonard_spent + michael_jeans_cost) →
  backpack_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_backpack_cost_l243_24300


namespace NUMINAMATH_CALUDE_johns_initial_money_l243_24304

theorem johns_initial_money (M : ℝ) : 
  (M > 0) →
  ((1 - 1/5) * M * (1 - 3/4) = 4) →
  (M = 20) := by
sorry

end NUMINAMATH_CALUDE_johns_initial_money_l243_24304


namespace NUMINAMATH_CALUDE_min_value_of_sum_l243_24313

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / (3 * a + 2) + 1 / (3 * b + 2) + 1 / (3 * c + 2)) ≥ 1 ∧
  (1 / (3 * a + 2) + 1 / (3 * b + 2) + 1 / (3 * c + 2) = 1 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l243_24313


namespace NUMINAMATH_CALUDE_amare_initial_fabric_l243_24398

/-- The amount of fabric Amare initially has -/
def initial_fabric (yards_per_dress : ℝ) (num_dresses : ℕ) (feet_still_needed : ℝ) : ℝ :=
  num_dresses * yards_per_dress * 3 - feet_still_needed

/-- Theorem stating that Amare initially has 7 feet of fabric -/
theorem amare_initial_fabric :
  initial_fabric 5.5 4 59 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amare_initial_fabric_l243_24398


namespace NUMINAMATH_CALUDE_problem_solution_l243_24345

open Real

noncomputable def α : ℝ := sorry

-- Given conditions
axiom cond1 : (sin (π/2 - α) + sin (-π - α)) / (3 * cos (2*π + α) + cos (3*π/2 - α)) = 3
axiom cond2 : ∃ (a : ℝ), ∀ (x y : ℝ), (x - a)^2 + y^2 = 7 → y = 0
axiom cond3 : ∃ (a : ℝ), abs (2*a) / sqrt 5 = sqrt 5
axiom cond4 : ∃ (a r : ℝ), r > 0 ∧ (2*sqrt 2)^2 + (sqrt 5)^2 = (2*r)^2 ∧ ∀ (x y : ℝ), (x - a)^2 + y^2 = r^2

-- Theorem to prove
theorem problem_solution :
  (sin α - 3*cos α) / (sin α + cos α) = -1/3 ∧
  ∃ (a : ℝ), (∀ (x y : ℝ), (x - a)^2 + y^2 = 7 ∨ (x + a)^2 + y^2 = 7) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l243_24345


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l243_24393

theorem triangle_third_side_length 
  (a b : ℝ) 
  (angle : ℝ) 
  (ha : a = 9) 
  (hb : b = 8) 
  (hangle : angle = 150 * π / 180) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*(Real.cos angle) ∧ 
            c = Real.sqrt (145 + 72 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l243_24393


namespace NUMINAMATH_CALUDE_intersection_with_complement_l243_24349

def A : Set ℝ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {1, 2, 5, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l243_24349


namespace NUMINAMATH_CALUDE_fishermans_red_snappers_l243_24301

/-- The number of Red snappers caught daily -/
def red_snappers : ℕ := sorry

/-- The number of Tunas caught daily -/
def tunas : ℕ := 14

/-- The price of a Red snapper in dollars -/
def red_snapper_price : ℕ := 3

/-- The price of a Tuna in dollars -/
def tuna_price : ℕ := 2

/-- The total daily earnings in dollars -/
def total_earnings : ℕ := 52

theorem fishermans_red_snappers :
  red_snappers * red_snapper_price + tunas * tuna_price = total_earnings ∧
  red_snappers = 8 := by sorry

end NUMINAMATH_CALUDE_fishermans_red_snappers_l243_24301


namespace NUMINAMATH_CALUDE_y_percentage_more_than_z_l243_24328

/-- Given that x gets 25% more than y, the total amount is 370, and z's share is 100,
    prove that y gets 20% more than z. -/
theorem y_percentage_more_than_z (x y z : ℝ) : 
  x = 1.25 * y →  -- x gets 25% more than y
  x + y + z = 370 →  -- total amount is 370
  z = 100 →  -- z's share is 100
  y = 1.2 * z  -- y gets 20% more than z
  := by sorry

end NUMINAMATH_CALUDE_y_percentage_more_than_z_l243_24328


namespace NUMINAMATH_CALUDE_three_rug_overlap_l243_24329

theorem three_rug_overlap (total_rug_area floor_area double_layer_area : ℝ) 
  (h1 : total_rug_area = 90)
  (h2 : floor_area = 60)
  (h3 : double_layer_area = 12) : 
  ∃ (triple_layer_area : ℝ),
    triple_layer_area = 9 ∧
    ∃ (single_layer_area : ℝ),
      single_layer_area + double_layer_area + triple_layer_area = floor_area ∧
      single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = total_rug_area :=
by sorry

end NUMINAMATH_CALUDE_three_rug_overlap_l243_24329


namespace NUMINAMATH_CALUDE_sum_of_transformed_numbers_l243_24392

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
  3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_transformed_numbers_l243_24392


namespace NUMINAMATH_CALUDE_sequence_property_l243_24322

def sequence_a : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_a n ^ 2 + n * sequence_a n

def S : Set ℕ := {p : ℕ | Nat.Prime p ∧ ∃ i, p ∣ sequence_a i}

theorem sequence_property :
  (Set.Infinite S) ∧ (S ≠ {p : ℕ | Nat.Prime p}) := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l243_24322


namespace NUMINAMATH_CALUDE_class_size_l243_24362

theorem class_size (football : ℕ) (long_tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : football = 26)
  (h2 : long_tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 9) :
  football + long_tennis - both + neither = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l243_24362


namespace NUMINAMATH_CALUDE_sum_of_xy_l243_24341

theorem sum_of_xy (x y : ℕ) : 
  0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧ x + y + x * y = 143 → 
  x + y = 22 ∨ x + y = 23 ∨ x + y = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l243_24341


namespace NUMINAMATH_CALUDE_runner_position_l243_24390

/-- Represents a quarter of a circular track -/
inductive Quarter : Type
| E : Quarter
| F : Quarter
| G : Quarter
| H : Quarter

/-- Represents a point on the circular track -/
structure TrackPoint where
  distance : ℝ  -- Distance from the start point
  quarter : Quarter

/-- The circular track -/
structure Track where
  circumference : ℝ
  start_point : TrackPoint

theorem runner_position (track : Track) 
  (h1 : track.circumference = 100)
  (h2 : track.start_point.quarter = Quarter.E)
  (h3 : track.start_point.distance = 0)
  (run_distance : ℝ)
  (h4 : run_distance = 3000) :
  (Track.start_point track).quarter = 
    (TrackPoint.mk (run_distance % track.circumference) Quarter.E).quarter :=
sorry

end NUMINAMATH_CALUDE_runner_position_l243_24390


namespace NUMINAMATH_CALUDE_team_a_championship_probability_l243_24361

-- Define the game state
structure GameState where
  team_a_wins_needed : ℕ
  team_b_wins_needed : ℕ

-- Define the probability of Team A winning
def prob_team_a_wins (state : GameState) : ℚ :=
  if state.team_a_wins_needed = 0 then 1
  else if state.team_b_wins_needed = 0 then 0
  else sorry

-- Theorem statement
theorem team_a_championship_probability :
  let initial_state : GameState := ⟨1, 2⟩
  prob_team_a_wins initial_state = 3/4 :=
sorry

end NUMINAMATH_CALUDE_team_a_championship_probability_l243_24361


namespace NUMINAMATH_CALUDE_added_value_theorem_l243_24347

theorem added_value_theorem (x : ℝ) (y : ℝ) (h1 : x > 0) (h2 : x = 8) :
  x + y = 128 * (1/x) → y = 8 := by
sorry

end NUMINAMATH_CALUDE_added_value_theorem_l243_24347


namespace NUMINAMATH_CALUDE_equation_solutions_l243_24387

theorem equation_solutions :
  (∃ x : ℚ, x / (3/4) = 2 / (9/10) ∧ x = 5/3) ∧
  (∃ x : ℚ, 0.5 / x = 0.75 / 6 ∧ x = 4) ∧
  (∃ x : ℚ, x / 20 = 2/5 ∧ x = 8) ∧
  (∃ x : ℚ, (3/4 * x) / 15 = 2/3 ∧ x = 40/3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l243_24387


namespace NUMINAMATH_CALUDE_inequality_equivalence_l243_24331

theorem inequality_equivalence (x : ℝ) : x / 3 - 2 < 0 ↔ x < 6 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l243_24331


namespace NUMINAMATH_CALUDE_only_first_proposition_correct_l243_24317

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem only_first_proposition_correct 
  (m l : Line) (α β : Plane) 
  (h_diff_lines : m ≠ l) 
  (h_diff_planes : α ≠ β) :
  ((perpendicular_plane_line α l ∧ parallel_plane_line α m → perpendicular l m) ∧
   ¬(parallel m l ∧ line_in_plane m α → parallel_plane_line α l) ∧
   ¬(perpendicular_planes α β ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular m l) ∧
   ¬(perpendicular m l ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular_planes α β)) :=
by sorry

end NUMINAMATH_CALUDE_only_first_proposition_correct_l243_24317


namespace NUMINAMATH_CALUDE_exist_three_distinct_naturals_sum_product_squares_l243_24380

theorem exist_three_distinct_naturals_sum_product_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ (m : ℕ), a + b + c = m^2) ∧
  (∃ (n : ℕ), a * b * c = n^2) := by
  sorry

end NUMINAMATH_CALUDE_exist_three_distinct_naturals_sum_product_squares_l243_24380


namespace NUMINAMATH_CALUDE_floor_of_5_7_l243_24305

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l243_24305


namespace NUMINAMATH_CALUDE_infinite_sum_equals_one_fourth_l243_24357

open Real
open BigOperators

theorem infinite_sum_equals_one_fourth :
  ∑' n : ℕ, (3^n : ℝ) / (1 + 3^n + 3^(n+1) + 3^(2*n+1)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_one_fourth_l243_24357


namespace NUMINAMATH_CALUDE_triangle_tangent_relation_l243_24343

theorem triangle_tangent_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A) ∧ (A < π / 2) ∧
  (0 < B) ∧ (B < π / 2) ∧
  (0 < C) ∧ (C < π / 2) ∧
  (A + B + C = π) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) ∧
  (c^2 = a^2 + b^2 - 2 * a * b * Real.cos C) ∧
  (Real.tan A * Real.tan B = Real.tan A * Real.tan C + Real.tan C * Real.tan B) →
  (a^2 + b^2) / c^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_relation_l243_24343


namespace NUMINAMATH_CALUDE_larger_number_problem_l243_24323

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l243_24323


namespace NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l243_24373

theorem quadratic_roots_pure_imaginary (m : ℝ) (hm : m < 0) :
  ∀ (z : ℂ), 8 * z^2 + 4 * Complex.I * z - m = 0 →
  ∃ (y : ℝ), z = Complex.I * y :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_pure_imaginary_l243_24373


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l243_24308

/-- Pascal's triangle contains every positive integer -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → ∃ (row k : ℕ), Nat.choose row k = n

/-- The binomial coefficient function -/
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

/-- The smallest four-digit number -/
def smallest_four_digit : ℕ := 1000

/-- Theorem: 1000 is the smallest four-digit number in Pascal's triangle -/
theorem smallest_four_digit_in_pascal : 
  ∃ (row k : ℕ), binomial_coeff row k = smallest_four_digit ∧ 
  (∀ (r s : ℕ), binomial_coeff r s < smallest_four_digit → binomial_coeff r s < 1000) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l243_24308


namespace NUMINAMATH_CALUDE_triangle_equilateral_conditions_l243_24388

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (h_a h_b h_c : ℝ)
  (ha_pos : h_a > 0)
  (hb_pos : h_b > 0)
  (hc_pos : h_c > 0)

-- Define the property of having equal sums of side and height
def equal_side_height_sums (t : Triangle) : Prop :=
  t.a + t.h_a = t.b + t.h_b ∧ t.b + t.h_b = t.c + t.h_c

-- Define the property of having equal inscribed squares
def equal_inscribed_squares (t : Triangle) : Prop :=
  (2 * t.a * t.h_a) / (t.a + t.h_a) = (2 * t.b * t.h_b) / (t.b + t.h_b) ∧
  (2 * t.b * t.h_b) / (t.b + t.h_b) = (2 * t.c * t.h_c) / (t.c + t.h_c)

-- Define an equilateral triangle
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- State the theorem
theorem triangle_equilateral_conditions (t : Triangle) :
  (equal_side_height_sums t ∨ equal_inscribed_squares t) → is_equilateral t :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_conditions_l243_24388


namespace NUMINAMATH_CALUDE_tax_discount_commute_l243_24356

theorem tax_discount_commute (p t d : ℝ) (h1 : 0 ≤ p) (h2 : 0 ≤ t) (h3 : 0 ≤ d) (h4 : d ≤ 1) :
  p * (1 + t) * (1 - d) = p * (1 - d) * (1 + t) :=
by sorry

#check tax_discount_commute

end NUMINAMATH_CALUDE_tax_discount_commute_l243_24356


namespace NUMINAMATH_CALUDE_product_repeating_decimal_one_third_and_eight_l243_24330

def repeating_decimal_one_third : ℚ := 1/3

theorem product_repeating_decimal_one_third_and_eight :
  repeating_decimal_one_third * 8 = 8/3 := by sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_one_third_and_eight_l243_24330


namespace NUMINAMATH_CALUDE_same_school_probability_same_school_probability_proof_l243_24302

/-- The probability of selecting two teachers from the same school when randomly choosing
    two teachers out of three from School A and three from School B. -/
theorem same_school_probability : ℚ :=
  let total_teachers : ℕ := 6
  let teachers_per_school : ℕ := 3
  let selected_teachers : ℕ := 2

  2 / 5

/-- Proof that the probability of selecting two teachers from the same school is 2/5. -/
theorem same_school_probability_proof :
  same_school_probability = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_school_probability_same_school_probability_proof_l243_24302


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l243_24340

/-- Given an arithmetic sequence where the 5th term is 23 and the 7th term is 37, 
    prove that the 9th term is 51. -/
theorem arithmetic_sequence_ninth_term 
  (a : ℤ) -- First term of the sequence
  (d : ℤ) -- Common difference
  (h1 : a + 4 * d = 23) -- 5th term is 23
  (h2 : a + 6 * d = 37) -- 7th term is 37
  : a + 8 * d = 51 := by -- 9th term is 51
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l243_24340


namespace NUMINAMATH_CALUDE_descending_order_XYZ_l243_24351

theorem descending_order_XYZ : ∀ (X Y Z : ℝ),
  X = 0.6 * 0.5 + 0.4 →
  Y = 0.6 * 0.5 / 0.4 →
  Z = 0.6 * 0.5 * 0.4 →
  Y > X ∧ X > Z :=
by
  sorry

end NUMINAMATH_CALUDE_descending_order_XYZ_l243_24351


namespace NUMINAMATH_CALUDE_souvenir_sales_profit_l243_24385

/-- Represents the profit function for souvenir sales -/
def profit_function (x : ℝ) : ℝ :=
  (x - 5) * (32 - 4 * (x - 9))

/-- Theorem stating the properties of the profit function -/
theorem souvenir_sales_profit :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit_function x₁ = 140 ∧ profit_function x₂ = 140 ∧ 
    (x₁ = 10 ∨ x₁ = 12) ∧ (x₂ = 10 ∨ x₂ = 12)) ∧ 
  (∃ x_max : ℝ, x_max = 11 ∧ 
    ∀ x : ℝ, profit_function x ≤ profit_function x_max) ∧
  profit_function 11 = 144 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_sales_profit_l243_24385


namespace NUMINAMATH_CALUDE_election_expectation_l243_24374

/-- The number of voters and candidates in the election -/
def n : ℕ := 5

/-- The probability of a candidate receiving no votes -/
def p_no_votes : ℚ := (4/5)^n

/-- The probability of a candidate receiving at least one vote -/
def p_at_least_one_vote : ℚ := 1 - p_no_votes

/-- The expected number of candidates receiving at least one vote -/
def expected_candidates_with_votes : ℚ := n * p_at_least_one_vote

theorem election_expectation :
  expected_candidates_with_votes = 2101/625 :=
by sorry

end NUMINAMATH_CALUDE_election_expectation_l243_24374


namespace NUMINAMATH_CALUDE_inequality_solution_set_l243_24307

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo 0 2 : Set ℝ) = {x | |2*x - 1| < |x| + 1} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l243_24307


namespace NUMINAMATH_CALUDE_parabola_vertex_l243_24335

/-- The parabola defined by y = x^2 - 2 -/
def parabola (x : ℝ) : ℝ := x^2 - 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (0, -2)

/-- Theorem: The vertex of the parabola y = x^2 - 2 is at the point (0, -2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l243_24335


namespace NUMINAMATH_CALUDE_earnings_ratio_l243_24394

/-- Proves that given Mork's tax rate of 30%, Mindy's tax rate of 20%, and their combined tax rate of 22.5%, the ratio of Mindy's earnings to Mork's earnings is 3:1. -/
theorem earnings_ratio (mork_earnings mindy_earnings : ℝ) 
  (mork_tax_rate : ℝ) (mindy_tax_rate : ℝ) (combined_tax_rate : ℝ)
  (h1 : mork_tax_rate = 0.3)
  (h2 : mindy_tax_rate = 0.2)
  (h3 : combined_tax_rate = 0.225)
  (h4 : mork_earnings > 0)
  (h5 : mindy_earnings > 0)
  (h6 : mindy_tax_rate * mindy_earnings + mork_tax_rate * mork_earnings = 
        combined_tax_rate * (mindy_earnings + mork_earnings)) :
  mindy_earnings / mork_earnings = 3 := by
  sorry


end NUMINAMATH_CALUDE_earnings_ratio_l243_24394


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_cone_l243_24344

theorem lateral_surface_area_of_cone (slant_height base_radius : ℝ) 
  (h1 : slant_height = 4)
  (h2 : base_radius = 2) :
  (1/2) * slant_height * (2 * Real.pi * base_radius) = 8 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_cone_l243_24344


namespace NUMINAMATH_CALUDE_wrong_observation_value_l243_24360

theorem wrong_observation_value (n : ℕ) (original_mean new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 30)
  (h3 : new_mean = 30.5) :
  ∃ (wrong_value correct_value : ℝ),
    (n : ℝ) * original_mean = (n - 1 : ℝ) * original_mean + wrong_value ∧
    (n : ℝ) * new_mean = (n - 1 : ℝ) * original_mean + correct_value ∧
    wrong_value = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l243_24360


namespace NUMINAMATH_CALUDE_complement_A_intersect_integers_l243_24336

def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}

theorem complement_A_intersect_integers :
  (Set.univ \ A) ∩ Set.range (Int.cast : ℤ → ℝ) = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_integers_l243_24336


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l243_24318

def triangle_area (r R : ℝ) (cosA cosB cosC : ℝ) (a b c : ℝ) : Prop :=
  r = 7 ∧
  R = 20 ∧
  3 * cosB = 2 * cosA + cosC ∧
  cosA + cosB + cosC = 1 + r / R ∧
  b = 2 * R * Real.sqrt (1 - cosB^2) ∧
  a^2 + c^2 - a * c * cosB = b^2 ∧
  cosA = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  cosC = (a^2 + b^2 - c^2) / (2 * a * b) ∧
  (7 * (a + c + 2 * Real.sqrt 319)) / 2 = 7 * ((a + b + c) / 2)

theorem triangle_area_theorem :
  ∀ (r R : ℝ) (cosA cosB cosC : ℝ) (a b c : ℝ),
    triangle_area r R cosA cosB cosC a b c →
    (7 * (a + c + 2 * Real.sqrt 319)) / 2 = 7 * ((a + b + c) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l243_24318


namespace NUMINAMATH_CALUDE_rosie_pies_l243_24327

/-- Given that Rosie can make 3 pies out of 12 apples, 
    prove that she can make 9 pies out of 36 apples. -/
theorem rosie_pies (apples_per_batch : ℕ) (pies_per_batch : ℕ) 
  (h1 : apples_per_batch = 12) 
  (h2 : pies_per_batch = 3) 
  (h3 : 36 = 3 * apples_per_batch) : 
  (36 / (apples_per_batch / pies_per_batch) : ℕ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l243_24327


namespace NUMINAMATH_CALUDE_factorization_sum_l243_24375

theorem factorization_sum (x y : ℝ) : ∃ (a b c d e f g h j k : ℤ),
  (27 * x^9 - 512 * y^9 = (a * x + b * y) * (c * x^3 + d * x * y^2 + e * y^3) * 
                          (f * x + g * y) * (h * x^3 + j * x * y^2 + k * y^3)) ∧
  (a + b + c + d + e + f + g + h + j + k = 12) := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l243_24375


namespace NUMINAMATH_CALUDE_tree_planting_probability_l243_24306

def num_cedar : ℕ := 4
def num_pine : ℕ := 3
def num_alder : ℕ := 6

def total_trees : ℕ := num_cedar + num_pine + num_alder

def probability_no_adjacent_alders : ℚ := 2 / 4290

theorem tree_planting_probability :
  let total_arrangements : ℕ := (Nat.factorial total_trees) / 
    (Nat.factorial num_cedar * Nat.factorial num_pine * Nat.factorial num_alder)
  let valid_arrangements : ℕ := Nat.choose (num_cedar + num_pine + 1) num_alder * 
    (Nat.factorial (num_cedar + num_pine) / (Nat.factorial num_cedar * Nat.factorial num_pine))
  (valid_arrangements : ℚ) / total_arrangements = probability_no_adjacent_alders :=
sorry

end NUMINAMATH_CALUDE_tree_planting_probability_l243_24306


namespace NUMINAMATH_CALUDE_circle_radii_equation_l243_24358

theorem circle_radii_equation (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (ha_def : a = 1 / a) (hb_def : b = 1 / b) (hc_def : c = 1 / c) (hd_def : d = 1 / d) :
  2 * (a^2 + b^2 + c^2 + d^2) = (a + b + c + d)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_equation_l243_24358


namespace NUMINAMATH_CALUDE_product_of_differences_l243_24325

theorem product_of_differences (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2008) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2008) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2008) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 4015/2008 := by
  sorry

end NUMINAMATH_CALUDE_product_of_differences_l243_24325


namespace NUMINAMATH_CALUDE_triangle_area_l243_24326

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ S = (2 * Real.sqrt 3) / 3) :=
by sorry

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l243_24326


namespace NUMINAMATH_CALUDE_banquet_guests_l243_24383

theorem banquet_guests (x : ℚ) : 
  (1/3 * x + 3/5 * (2/3 * x) + 4 = x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_banquet_guests_l243_24383


namespace NUMINAMATH_CALUDE_calum_disco_ball_spending_l243_24303

/-- Represents the problem of calculating the maximum amount Calum can spend on each disco ball. -/
theorem calum_disco_ball_spending (
  disco_ball_count : ℕ)
  (food_box_count : ℕ)
  (decoration_set_count : ℕ)
  (food_box_cost : ℚ)
  (decoration_set_cost : ℚ)
  (total_budget : ℚ)
  (disco_ball_budget_percentage : ℚ)
  (h1 : disco_ball_count = 4)
  (h2 : food_box_count = 10)
  (h3 : decoration_set_count = 20)
  (h4 : food_box_cost = 25)
  (h5 : decoration_set_cost = 10)
  (h6 : total_budget = 600)
  (h7 : disco_ball_budget_percentage = 0.3)
  : (total_budget * disco_ball_budget_percentage) / disco_ball_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_calum_disco_ball_spending_l243_24303


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l243_24346

theorem contrapositive_equivalence (a b c d : ℝ) :
  ((a = b ∧ c = d) → a + c = b + d) ↔ (a + c ≠ b + d → a ≠ b ∨ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l243_24346


namespace NUMINAMATH_CALUDE_fourth_power_inequality_l243_24319

theorem fourth_power_inequality (a b c : ℝ) :
  a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_inequality_l243_24319


namespace NUMINAMATH_CALUDE_integer_chord_lines_count_l243_24315

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point : Point
  direction : Point  -- Direction vector

/-- Define the circle from the problem -/
def problemCircle : Circle :=
  { center := { x := 2, y := -2 },
    radius := 5 }

/-- Define the point M -/
def pointM : Point :=
  { x := 2, y := 2 }

/-- Function to check if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

/-- Function to count lines passing through M that cut off integer-length chords -/
def countIntegerChordLines (c : Circle) (m : Point) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem -/
theorem integer_chord_lines_count :
  isInside pointM problemCircle →
  countIntegerChordLines problemCircle pointM = 8 := by
  sorry

end NUMINAMATH_CALUDE_integer_chord_lines_count_l243_24315


namespace NUMINAMATH_CALUDE_motorcyclist_problem_l243_24316

/-- The time taken by the first motorcyclist to travel the distance AB -/
def time_first : ℝ := 80

/-- The time taken by the second motorcyclist to travel the distance AB -/
def time_second : ℝ := 60

/-- The time taken by the third motorcyclist to travel the distance AB -/
def time_third : ℝ := 3240

/-- The head start of the first motorcyclist -/
def head_start : ℝ := 5

/-- The time difference between the third and second motorcyclist overtaking the first -/
def overtake_diff : ℝ := 10

/-- The distance between points A and B -/
def distance : ℝ := 1  -- We can set this to any positive real number

theorem motorcyclist_problem :
  ∃ (speed_first speed_second speed_third : ℝ),
    speed_first > 0 ∧ speed_second > 0 ∧ speed_third > 0 ∧
    speed_first ≠ speed_second ∧ speed_first ≠ speed_third ∧ speed_second ≠ speed_third ∧
    speed_first = distance / time_first ∧
    speed_second = distance / time_second ∧
    speed_third = distance / time_third ∧
    (time_third - head_start) * speed_third = time_first * speed_first ∧
    (time_second - head_start) * speed_second = (time_first + overtake_diff) * speed_first :=
by sorry

end NUMINAMATH_CALUDE_motorcyclist_problem_l243_24316


namespace NUMINAMATH_CALUDE_sandy_second_shop_amount_l243_24368

/-- The amount Sandy paid for books from the second shop -/
def second_shop_amount (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (first_shop_amount : ℚ) (average_price : ℚ) : ℚ :=
  (first_shop_books + second_shop_books) * average_price - first_shop_amount

/-- Proof that Sandy paid $900 for books from the second shop -/
theorem sandy_second_shop_amount :
  second_shop_amount 65 55 1380 19 = 900 := by
  sorry

end NUMINAMATH_CALUDE_sandy_second_shop_amount_l243_24368


namespace NUMINAMATH_CALUDE_function_properties_l243_24397

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x * Real.exp 1

-- State the theorem
theorem function_properties :
  -- Part 1: The constant a is 2
  (∃ a : ℝ, (deriv (f a)) 0 = -1 ∧ a = 2) ∧
  -- Part 2: For x > 0, x^2 < e^x
  (∀ x : ℝ, x > 0 → x^2 < Real.exp x) ∧
  -- Part 3: For any positive c, there exists x₀ such that for x ∈ (x₀, +∞), x^2 < ce^x
  (∀ c : ℝ, c > 0 → ∃ x₀ : ℝ, ∀ x : ℝ, x > x₀ → x^2 < c * Real.exp x) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l243_24397


namespace NUMINAMATH_CALUDE_unit_digit_G_1000_l243_24334

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The unit digit of a natural number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The unit digit of G_{1000} is 2 -/
theorem unit_digit_G_1000 : unitDigit (G 1000) = 2 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_G_1000_l243_24334


namespace NUMINAMATH_CALUDE_simplification_and_exponent_sum_l243_24348

-- Define the expression
def original_expression (x y z : ℝ) : ℝ := (40 * x^5 * y^7 * z^9) ^ (1/3)

-- Define the simplified expression
def simplified_expression (x y z : ℝ) : ℝ := 2 * x * y * z^3 * (5 * x^2 * y) ^ (1/3)

-- Define the sum of exponents outside the radical
def sum_of_exponents : ℕ := 1 + 1 + 3

-- Theorem statement
theorem simplification_and_exponent_sum :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  original_expression x y z = simplified_expression x y z ∧
  sum_of_exponents = 5 := by sorry

end NUMINAMATH_CALUDE_simplification_and_exponent_sum_l243_24348


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l243_24386

theorem geometric_sequence_first_term (a b c d : ℚ) :
  (∃ r : ℚ, r ≠ 0 ∧ 
    a * r^4 = 48 ∧ 
    a * r^5 = 192) →
  a = 3/16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l243_24386


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_18_l243_24371

/-- The sum of n consecutive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A predicate that checks if a set of consecutive integers sums to 18 -/
def isValidSet (a n : ℕ) : Prop :=
  n ≥ 3 ∧ consecutiveSum a n = 18

theorem unique_consecutive_sum_18 :
  ∃! p : ℕ × ℕ, isValidSet p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_18_l243_24371


namespace NUMINAMATH_CALUDE_line_equation_through_points_l243_24324

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = x₁ + t * (x₂ - x₁) ∧ p.2 = y₁ + t * (y₂ - y₁)}

-- Define the general form of a line equation
def general_line_equation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem line_equation_through_points :
  line_through_points 2 5 0 3 = general_line_equation 1 (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l243_24324


namespace NUMINAMATH_CALUDE_weight_of_B_l243_24367

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) :
  B = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l243_24367


namespace NUMINAMATH_CALUDE_money_difference_l243_24342

/-- The amount of money Gwen received from her dad -/
def money_from_dad : ℕ := 5

/-- The amount of money Gwen received from her mom -/
def money_from_mom : ℕ := 7

/-- The difference between the amount Gwen received from her mom and her dad -/
def difference : ℕ := money_from_mom - money_from_dad

theorem money_difference : difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l243_24342


namespace NUMINAMATH_CALUDE_max_intersections_proof_l243_24399

/-- The maximum number of intersection points between two circles -/
def max_circle_intersections : ℕ := 2

/-- The maximum number of intersection points between a line and a circle -/
def max_line_circle_intersections : ℕ := 2

/-- The maximum number of intersection points between two lines -/
def max_line_line_intersections : ℕ := 1

/-- The number of circles -/
def num_circles : ℕ := 2

/-- The number of lines -/
def num_lines : ℕ := 3

/-- The maximum number of intersection points between all figures -/
def max_total_intersections : ℕ := 17

theorem max_intersections_proof :
  max_total_intersections = 
    (num_circles.choose 2) * max_circle_intersections +
    num_lines * num_circles * max_line_circle_intersections +
    (num_lines.choose 2) * max_line_line_intersections :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersections_proof_l243_24399


namespace NUMINAMATH_CALUDE_sandbox_area_l243_24354

-- Define the sandbox dimensions
def sandbox_length : ℝ := 312
def sandbox_width : ℝ := 146

-- State the theorem
theorem sandbox_area : sandbox_length * sandbox_width = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_l243_24354


namespace NUMINAMATH_CALUDE_fifth_rest_day_is_monday_l243_24338

def day_of_week (n : ℕ) : ℕ := n % 7 + 1

def rest_day (n : ℕ) : ℕ := 4 * n - 2

theorem fifth_rest_day_is_monday :
  day_of_week (rest_day 5) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fifth_rest_day_is_monday_l243_24338


namespace NUMINAMATH_CALUDE_square_exterior_points_distance_l243_24379

/-- Given a square ABCD with side length 13 and exterior points E and F,
    prove that EF² = 578 when BE = DF = 5 and AE = CF = 12 -/
theorem square_exterior_points_distance (A B C D E F : ℝ × ℝ) : 
  let side_length : ℝ := 13
  -- Square ABCD
  A = (0, side_length) ∧ 
  B = (side_length, side_length) ∧ 
  C = (side_length, 0) ∧ 
  D = (0, 0) ∧
  -- Exterior points E and F
  dist B E = 5 ∧
  dist D F = 5 ∧
  dist A E = 12 ∧
  dist C F = 12
  →
  dist E F ^ 2 = 578 := by
sorry


end NUMINAMATH_CALUDE_square_exterior_points_distance_l243_24379


namespace NUMINAMATH_CALUDE_negative_seven_to_fourth_power_l243_24384

theorem negative_seven_to_fourth_power : (-7 : ℤ) ^ 4 = (-7) * (-7) * (-7) * (-7) := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_to_fourth_power_l243_24384


namespace NUMINAMATH_CALUDE_intersection_A_B_l243_24312

-- Define the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l243_24312


namespace NUMINAMATH_CALUDE_outfits_count_l243_24382

/-- Represents the number of shirts available -/
def num_shirts : Nat := 7

/-- Represents the number of pants available -/
def num_pants : Nat := 5

/-- Represents the number of ties available -/
def num_ties : Nat := 4

/-- Represents the number of jackets available -/
def num_jackets : Nat := 2

/-- Represents the number of tie options (wearing a tie or not) -/
def tie_options : Nat := num_ties + 1

/-- Represents the number of jacket options (wearing a jacket or not) -/
def jacket_options : Nat := num_jackets + 1

/-- Calculates the total number of possible outfits -/
def total_outfits : Nat := num_shirts * num_pants * tie_options * jacket_options

/-- Proves that the total number of possible outfits is 525 -/
theorem outfits_count : total_outfits = 525 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l243_24382


namespace NUMINAMATH_CALUDE_quadratic_vertex_in_first_quadrant_l243_24332

/-- Given a quadratic function y = ax² + bx + c where a, b, and c satisfy certain conditions,
    prove that its vertex lies in the first quadrant. -/
theorem quadratic_vertex_in_first_quadrant
  (a b c : ℝ)
  (eq1 : a - b + c = 0)
  (eq2 : 9*a + 3*b + c = 0)
  (b_pos : b > 0) :
  let x := -b / (2*a)
  let y := a * x^2 + b * x + c
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_in_first_quadrant_l243_24332


namespace NUMINAMATH_CALUDE_grass_withering_is_certain_event_l243_24355

/-- An event that occurs regularly and predictably every year -/
structure AnnualEvent where
  occurs_yearly : Bool
  predictable : Bool

/-- Definition of a certain event in probability theory -/
def CertainEvent (e : AnnualEvent) : Prop :=
  e.occurs_yearly ∧ e.predictable

/-- The withering of grass on a plain as described in the poem -/
def grass_withering : AnnualEvent :=
  { occurs_yearly := true
  , predictable := true }

/-- Theorem stating that the grass withering is a certain event -/
theorem grass_withering_is_certain_event : CertainEvent grass_withering := by
  sorry

end NUMINAMATH_CALUDE_grass_withering_is_certain_event_l243_24355


namespace NUMINAMATH_CALUDE_circle_equation_radius_7_l243_24359

/-- The equation x^2 + 8x + y^2 + 4y - k = 0 represents a circle of radius 7 if and only if k = 29 -/
theorem circle_equation_radius_7 (x y k : ℝ) : 
  (x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 7^2) ↔ k = 29 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_radius_7_l243_24359


namespace NUMINAMATH_CALUDE_real_part_of_one_plus_i_over_i_l243_24333

/-- The real part of (1+i)/i is 1 -/
theorem real_part_of_one_plus_i_over_i : 
  Complex.re ((1 + Complex.I) / Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_one_plus_i_over_i_l243_24333


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l243_24395

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  ∃ k : ℕ, n + 1 = 4 * k ∧
  ∃ l : ℕ, n + 1 = 5 * l ∧
  ∃ m : ℕ, n + 1 = 6 * m ∧
  ∃ p : ℕ, n + 1 = 8 * p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l243_24395


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l243_24381

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → t = 3.7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l243_24381


namespace NUMINAMATH_CALUDE_time_after_2021_hours_l243_24314

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a time of day -/
structure TimeOfDay where
  hour : Nat
  minute : Nat
  h_valid : hour < 24
  m_valid : minute < 60

/-- Represents a moment in time -/
structure Moment where
  day : DayOfWeek
  time : TimeOfDay

/-- Adds hours to a given moment and returns the new moment -/
def addHours (start : Moment) (hours : Nat) : Moment :=
  sorry

theorem time_after_2021_hours :
  let start : Moment := ⟨DayOfWeek.Monday, ⟨20, 21, sorry, sorry⟩⟩
  let end_moment : Moment := addHours start 2021
  end_moment = ⟨DayOfWeek.Tuesday, ⟨1, 21, sorry, sorry⟩⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_2021_hours_l243_24314


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l243_24363

theorem quadratic_root_implies_m_value :
  ∀ m : ℝ, (2^2 + m*2 + 2 = 0) → m = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l243_24363


namespace NUMINAMATH_CALUDE_b_income_percentage_over_c_l243_24396

/-- Given the monthly incomes of A, B, and C, prove that B's monthly income is 12% more than C's. -/
theorem b_income_percentage_over_c (a_annual : ℕ) (c_monthly : ℕ) (h1 : a_annual = 571200) (h2 : c_monthly = 17000) :
  let a_monthly : ℕ := a_annual / 12
  let b_monthly : ℕ := (2 * a_monthly) / 5
  (b_monthly : ℚ) / c_monthly - 1 = 12 / 100 := by sorry

end NUMINAMATH_CALUDE_b_income_percentage_over_c_l243_24396


namespace NUMINAMATH_CALUDE_terminal_side_angle_expression_l243_24339

theorem terminal_side_angle_expression (α : Real) :
  let P : Real × Real := (1, 3)
  let r : Real := Real.sqrt (P.1^2 + P.2^2)
  (P.1 / r = Real.cos α) ∧ (P.2 / r = Real.sin α) →
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_angle_expression_l243_24339


namespace NUMINAMATH_CALUDE_international_shipping_charge_l243_24365

theorem international_shipping_charge 
  (total_letters : ℕ) 
  (standard_postage : ℚ) 
  (international_letters : ℕ) 
  (total_cost : ℚ) : 
  total_letters = 4 → 
  standard_postage = 108/100 → 
  international_letters = 2 → 
  total_cost = 460/100 → 
  (total_cost - total_letters * standard_postage) / international_letters * 100 = 14 :=
by sorry

end NUMINAMATH_CALUDE_international_shipping_charge_l243_24365


namespace NUMINAMATH_CALUDE_machine_value_theorem_l243_24321

/-- Calculates the machine's value after 2 years given the initial conditions -/
def machine_value_after_two_years (initial_value : ℝ) (depreciation_rate_year1 : ℝ) 
  (depreciation_rate_subsequent : ℝ) (inflation_rate_year1 : ℝ) (inflation_rate_year2 : ℝ) 
  (maintenance_cost_year1 : ℝ) (maintenance_cost_increase_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the machine's value after 2 years is $754.58 -/
theorem machine_value_theorem : 
  machine_value_after_two_years 1000 0.12 0.08 0.02 0.035 50 0.05 = 754.58 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_theorem_l243_24321
