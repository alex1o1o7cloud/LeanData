import Mathlib

namespace complex_not_purely_imaginary_l924_92425

theorem complex_not_purely_imaginary (a : ℝ) : 
  (Complex.mk (a^2 - a - 2) (|a - 1| - 1) ≠ Complex.I * (Complex.mk 0 (|a - 1| - 1))) ↔ 
  (a ≠ -1) := by
  sorry

end complex_not_purely_imaginary_l924_92425


namespace f_minimum_value_l924_92487

noncomputable def f (x : ℝ) : ℝ := (1 + 4 * x) / Real.sqrt x

theorem f_minimum_value (x : ℝ) (hx : x > 0) : 
  f x ≥ 4 ∧ ∃ x₀ > 0, f x₀ = 4 :=
sorry

end f_minimum_value_l924_92487


namespace length_to_width_ratio_l924_92405

def field_perimeter : ℝ := 384
def field_width : ℝ := 80

theorem length_to_width_ratio :
  let field_length := (field_perimeter - 2 * field_width) / 2
  field_length / field_width = 7 / 5 := by
  sorry

end length_to_width_ratio_l924_92405


namespace min_trig_expression_l924_92472

open Real

theorem min_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  (3 * cos θ + 1 / sin θ + 4 * tan θ) ≥ 3 * (6 ^ (1 / 3)) ∧
  ∃ θ₀, 0 < θ₀ ∧ θ₀ < π / 2 ∧ 3 * cos θ₀ + 1 / sin θ₀ + 4 * tan θ₀ = 3 * (6 ^ (1 / 3)) :=
sorry

end min_trig_expression_l924_92472


namespace workshop_workers_count_l924_92451

theorem workshop_workers_count :
  let total_average : ℝ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℝ := 14000
  let non_technician_average : ℝ := 6000
  ∃ (total_workers : ℕ) (non_technician_workers : ℕ),
    total_workers = technician_count + non_technician_workers ∧
    total_average * (technician_count + non_technician_workers : ℝ) =
      technician_average * technician_count + non_technician_average * non_technician_workers ∧
    total_workers = 28 :=
by
  sorry

#check workshop_workers_count

end workshop_workers_count_l924_92451


namespace smallest_positive_integer_l924_92468

theorem smallest_positive_integer : ∀ n : ℕ, n > 0 → n ≥ 1 :=
by
  sorry

end smallest_positive_integer_l924_92468


namespace right_triangle_third_side_l924_92440

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 3 ∧ b = 5) ∨ (a = 3 ∧ c = 5) ∨ (b = 3 ∧ c = 5) →
  (a^2 + b^2 = c^2) →
  c = 4 ∨ c = Real.sqrt 34 := by
  sorry

end right_triangle_third_side_l924_92440


namespace sqrt_meaningful_implies_a_geq_neg_one_l924_92465

theorem sqrt_meaningful_implies_a_geq_neg_one (a : ℝ) : 
  (∃ (x : ℝ), x^2 = a + 1) → a ≥ -1 := by
  sorry

end sqrt_meaningful_implies_a_geq_neg_one_l924_92465


namespace fraction_meaningful_iff_not_three_l924_92447

theorem fraction_meaningful_iff_not_three (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x - 3)) ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_iff_not_three_l924_92447


namespace range_of_a_l924_92459

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → a ≤ x^2 - 4*x) → a ≤ -3 := by
  sorry

end range_of_a_l924_92459


namespace six_eight_x_ten_y_l924_92404

theorem six_eight_x_ten_y (x y Q : ℝ) (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q := by
  sorry

end six_eight_x_ten_y_l924_92404


namespace point_distance_on_number_line_l924_92492

theorem point_distance_on_number_line :
  ∀ x : ℝ, |x - (-3)| = 4 ↔ x = -7 ∨ x = 1 := by sorry

end point_distance_on_number_line_l924_92492


namespace paige_remaining_stickers_l924_92479

/-- The number of space stickers Paige has -/
def space_stickers : ℕ := 100

/-- The number of cat stickers Paige has -/
def cat_stickers : ℕ := 50

/-- The number of friends Paige is sharing with -/
def num_friends : ℕ := 3

/-- The function to calculate the number of remaining stickers -/
def remaining_stickers (space : ℕ) (cat : ℕ) (friends : ℕ) : ℕ :=
  (space % friends) + (cat % friends)

/-- Theorem stating that Paige will have 3 stickers left -/
theorem paige_remaining_stickers :
  remaining_stickers space_stickers cat_stickers num_friends = 3 := by
  sorry

end paige_remaining_stickers_l924_92479


namespace diary_pieces_not_complete_l924_92442

theorem diary_pieces_not_complete : ¬∃ (n : ℕ), 4^n = 50 := by sorry

end diary_pieces_not_complete_l924_92442


namespace fuel_purchase_l924_92482

/-- Fuel purchase problem -/
theorem fuel_purchase (total_spent : ℝ) (initial_cost final_cost : ℝ) :
  total_spent = 90 ∧
  initial_cost = 3 ∧
  final_cost = 4 ∧
  ∃ (mid_cost : ℝ), initial_cost < mid_cost ∧ mid_cost < final_cost →
  ∃ (quantity : ℝ),
    quantity > 0 ∧
    total_spent = initial_cost * quantity + ((initial_cost + final_cost) / 2) * quantity + final_cost * quantity ∧
    initial_cost * quantity + final_cost * quantity = 60 :=
by sorry

end fuel_purchase_l924_92482


namespace probability_divisible_by_three_l924_92403

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 15}

def is_divisible_by_three (x y z : ℕ) : Prop :=
  (x * y * z - x * y - y * z - z * x + x + y + z) % 3 = 0

def favorable_outcomes : ℕ := 60

def total_outcomes : ℕ := Nat.choose 15 3

theorem probability_divisible_by_three :
  (favorable_outcomes : ℚ) / total_outcomes = 12 / 91 := by sorry

end probability_divisible_by_three_l924_92403


namespace k_value_l924_92485

theorem k_value (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := by
sorry

end k_value_l924_92485


namespace f_properties_l924_92452

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - (1 / 2) * Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ), 
    (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -1/2) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1/2) :=
by sorry

end f_properties_l924_92452


namespace cloth_sale_worth_l924_92458

/-- Represents the worth of cloth sold given a commission rate and amount -/
def worthOfClothSold (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem stating that given a 4% commission rate and Rs. 12.50 commission,
    the worth of cloth sold is Rs. 312.50 -/
theorem cloth_sale_worth :
  worthOfClothSold (4 : ℚ) (25 / 2) = 625 / 2 := by
  sorry

end cloth_sale_worth_l924_92458


namespace P_intersect_Q_l924_92462

def P : Set ℝ := {x | x^2 - 16 < 0}
def Q : Set ℝ := {x | ∃ n : ℤ, x = 2 * ↑n}

theorem P_intersect_Q : P ∩ Q = {-2, 0, 2} := by sorry

end P_intersect_Q_l924_92462


namespace x_positive_sufficient_not_necessary_for_x_nonzero_l924_92414

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by sorry

end x_positive_sufficient_not_necessary_for_x_nonzero_l924_92414


namespace solve_for_x_l924_92428

-- Define the variables
variable (x y : ℝ)

-- State the theorem
theorem solve_for_x (eq1 : x + 2 * y = 12) (eq2 : y = 3) : x = 6 := by
  sorry

end solve_for_x_l924_92428


namespace exam_pass_count_l924_92434

theorem exam_pass_count (total_boys : ℕ) (overall_avg : ℚ) (pass_avg : ℚ) (fail_avg : ℚ) :
  total_boys = 120 →
  overall_avg = 38 →
  pass_avg = 39 →
  fail_avg = 15 →
  ∃ (pass_count : ℕ), pass_count = 115 ∧ 
    pass_count * pass_avg + (total_boys - pass_count) * fail_avg = total_boys * overall_avg :=
by sorry

end exam_pass_count_l924_92434


namespace contrapositive_equivalence_l924_92477

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 ≤ 1 → -1 ≤ x ∧ x ≤ 1) ↔
  (∀ x : ℝ, (x < -1 ∨ x > 1) → x^2 > 1) :=
by sorry

end contrapositive_equivalence_l924_92477


namespace coefficient_x2y4_is_30_l924_92401

/-- The coefficient of x^2y^4 in the expansion of (1+x+y^2)^5 -/
def coefficient_x2y4 : ℕ :=
  (Nat.choose 5 2) * (Nat.choose 3 2)

/-- Theorem stating that the coefficient of x^2y^4 in (1+x+y^2)^5 is 30 -/
theorem coefficient_x2y4_is_30 : coefficient_x2y4 = 30 := by
  sorry

end coefficient_x2y4_is_30_l924_92401


namespace soccer_substitutions_remainder_l924_92450

/-- Represents the number of ways to make exactly n substitutions -/
def b (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => 12 * (12 - n) * b n

/-- The total number of possible substitution ways -/
def total_ways : Nat :=
  b 0 + b 1 + b 2 + b 3 + b 4 + b 5

theorem soccer_substitutions_remainder :
  total_ways % 100 = 93 := by
  sorry

end soccer_substitutions_remainder_l924_92450


namespace sum_and_product_calculation_l924_92481

theorem sum_and_product_calculation :
  (199 + 298 + 397 + 496 + 595 + 20 = 2005) ∧
  (39 * 25 = 975) := by
sorry

end sum_and_product_calculation_l924_92481


namespace hyperbola_proof_l924_92424

-- Define the original hyperbola
def original_hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 = 1

-- Define the new hyperbola
def new_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

-- Define a function to check if two hyperbolas have the same asymptotes
def same_asymptotes (h1 h2 : (ℝ → ℝ → Prop)) : Prop := sorry

-- Theorem statement
theorem hyperbola_proof :
  same_asymptotes original_hyperbola new_hyperbola ∧
  new_hyperbola 2 0 := by sorry

end hyperbola_proof_l924_92424


namespace remainder_369963_div_6_l924_92454

theorem remainder_369963_div_6 : 369963 % 6 = 3 := by
  sorry

end remainder_369963_div_6_l924_92454


namespace quadratic_roots_sum_of_squares_l924_92476

theorem quadratic_roots_sum_of_squares (p q : ℝ) (r s : ℝ) : 
  (2 * r^2 - p * r + q = 0) → 
  (2 * s^2 - p * s + q = 0) → 
  (r^2 + s^2 = p^2 / 4 - q) := by
sorry

end quadratic_roots_sum_of_squares_l924_92476


namespace parabola_focus_l924_92490

/-- The parabola equation: x = -1/4 * (y - 2)^2 -/
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * (y - 2)^2

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Theorem: The focus of the parabola x = -1/4 * (y - 2)^2 is at (-1, 2) -/
theorem parabola_focus :
  ∃ (f : Focus), f.x = -1 ∧ f.y = 2 ∧
  ∀ (x y : ℝ), parabola_equation x y →
    (x - f.x)^2 + (y - f.y)^2 = (x + 1)^2 :=
sorry

end parabola_focus_l924_92490


namespace food_drive_ratio_l924_92489

theorem food_drive_ratio (total_students : ℕ) (no_cans_students : ℕ) (four_cans_students : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  no_cans_students = 2 →
  four_cans_students = 13 →
  total_cans = 232 →
  let twelve_cans_students := total_students - no_cans_students - four_cans_students
  (twelve_cans_students : ℚ) / total_students = 1 / 2 :=
by
  sorry

end food_drive_ratio_l924_92489


namespace total_bread_making_time_l924_92499

/-- The time it takes to make bread, given the time for rising, kneading, and baking. -/
def bread_making_time (rise_time : ℕ) (kneading_time : ℕ) (baking_time : ℕ) : ℕ :=
  2 * rise_time + kneading_time + baking_time

/-- Theorem stating that the total time to make bread is 280 minutes. -/
theorem total_bread_making_time :
  bread_making_time 120 10 30 = 280 := by
  sorry

end total_bread_making_time_l924_92499


namespace perimeter_gt_three_times_diameter_l924_92423

/-- A convex polyhedron. -/
class ConvexPolyhedron (M : Type*) where
  -- Add necessary axioms for convex polyhedron

/-- The perimeter of a convex polyhedron. -/
def perimeter (M : Type*) [ConvexPolyhedron M] : ℝ := sorry

/-- The diameter of a convex polyhedron. -/
def diameter (M : Type*) [ConvexPolyhedron M] : ℝ := sorry

/-- Theorem: The perimeter of a convex polyhedron is greater than three times its diameter. -/
theorem perimeter_gt_three_times_diameter (M : Type*) [ConvexPolyhedron M] :
  perimeter M > 3 * diameter M := by sorry

end perimeter_gt_three_times_diameter_l924_92423


namespace first_player_wins_l924_92420

/-- A proper divisor of n is a positive integer that divides n and is less than n. -/
def ProperDivisor (d n : ℕ) : Prop :=
  d > 0 ∧ d < n ∧ n % d = 0

/-- The game state, representing the number of tokens in the bowl. -/
structure GameState where
  tokens : ℕ

/-- A valid move in the game. -/
def ValidMove (s : GameState) (m : ℕ) : Prop :=
  ProperDivisor m s.tokens

/-- The game ends when the number of tokens exceeds 2024. -/
def GameOver (s : GameState) : Prop :=
  s.tokens > 2024

/-- The theorem stating that the first player has a winning strategy. -/
theorem first_player_wins :
  ∃ (strategy : GameState → ℕ),
    (∀ s : GameState, ¬GameOver s → ValidMove s (strategy s)) ∧
    (∀ (play : ℕ → GameState),
      play 0 = ⟨2⟩ →
      (∀ n : ℕ, ¬GameOver (play n) →
        play (n + 1) = ⟨(play n).tokens + strategy (play n)⟩ ∨
        (∃ m : ℕ, ValidMove (play n) m ∧
          play (n + 1) = ⟨(play n).tokens + m⟩)) →
      ∃ k : ℕ, GameOver (play k) ∧ k % 2 = 0) :=
sorry

end first_player_wins_l924_92420


namespace triangle_equilateral_condition_l924_92443

open Real

theorem triangle_equilateral_condition (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  a / cos A = b / cos B →
  a / cos A = c / cos C →
  A = B ∧ B = C :=
by sorry

end triangle_equilateral_condition_l924_92443


namespace A_suff_not_nec_D_l924_92498

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_iff_C : B ↔ C
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem A_suff_not_nec_D : (A → D) ∧ ¬(D → A) :=
by sorry

end A_suff_not_nec_D_l924_92498


namespace geometric_sequence_common_ratio_l924_92435

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 3 :=
sorry

end geometric_sequence_common_ratio_l924_92435


namespace unique_solution_quadratic_l924_92457

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) + (x - 3)^2 = 3 + k * x) ↔ 
  (k = -3 + 2 * Real.sqrt 10 ∨ k = -3 - 2 * Real.sqrt 10) :=
sorry

end unique_solution_quadratic_l924_92457


namespace correct_large_slices_per_pepper_l924_92480

/-- The number of bell peppers Tamia uses. -/
def num_peppers : ℕ := 5

/-- The total number of slices and pieces Tamia wants to add to her meal. -/
def total_slices : ℕ := 200

/-- Calculates the total number of slices and pieces based on the number of large slices per pepper. -/
def total_slices_func (x : ℕ) : ℕ :=
  num_peppers * x + num_peppers * (x / 2) * 3

/-- The number of large slices Tamia cuts each bell pepper into. -/
def large_slices_per_pepper : ℕ := 16

/-- Theorem stating that the number of large slices per pepper is correct. -/
theorem correct_large_slices_per_pepper : 
  total_slices_func large_slices_per_pepper = total_slices :=
by sorry

end correct_large_slices_per_pepper_l924_92480


namespace dealer_purchase_problem_l924_92491

theorem dealer_purchase_problem (total_cost : ℚ) (selling_price : ℚ) (num_sold : ℕ) (profit_percentage : ℚ) :
  total_cost = 25 →
  selling_price = 32 →
  num_sold = 12 →
  profit_percentage = 60 →
  (∃ (num_purchased : ℕ), 
    num_purchased * (selling_price / num_sold) = total_cost * (1 + profit_percentage / 100) ∧
    num_purchased = 15) :=
by sorry

end dealer_purchase_problem_l924_92491


namespace parabola_tangent_and_intersecting_line_l924_92449

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent line passing through (-1, 0)
def tangent_line (x y : ℝ) : Prop := ∃ t : ℝ, x = t*y - 1

-- Define the point P in the first quadrant
def point_P (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ parabola x y ∧ tangent_line x y

-- Define the line l passing through (2, 0)
def line_l (x y : ℝ) : Prop := ∃ m : ℝ, x = m*y + 2

-- Define the circle M with AB as diameter passing through P
def circle_M (xa ya xb yb : ℝ) : Prop :=
  ∃ xc yc : ℝ, (xc - 1)^2 + (yc - 2)^2 = ((xa - xb)^2 + (ya - yb)^2) / 4

theorem parabola_tangent_and_intersecting_line :
  -- Part 1: Point of tangency P
  (∃! x y : ℝ, point_P x y ∧ x = 1 ∧ y = 2) ∧
  -- Part 2: Equation of line l
  (∀ xa ya xb yb : ℝ,
    parabola xa ya ∧ parabola xb yb ∧
    line_l xa ya ∧ line_l xb yb ∧
    circle_M xa ya xb yb →
    ∃ m : ℝ, m = -2/3 ∧ ∀ x y : ℝ, line_l x y ↔ y = m*x + 4/3) :=
by sorry

end parabola_tangent_and_intersecting_line_l924_92449


namespace arc_length_for_36_degree_angle_l924_92470

theorem arc_length_for_36_degree_angle (d : ℝ) (θ : ℝ) (L : ℝ) : 
  d = 4 → θ = 36 * π / 180 → L = d * π * θ / 360 → L = 2 * π / 5 := by
  sorry

end arc_length_for_36_degree_angle_l924_92470


namespace tennis_tournament_result_l924_92497

/-- Represents the number of participants with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * (m.choose k)

/-- The number of participants in the tournament -/
def num_participants : ℕ := 254

/-- The number of rounds in the tournament -/
def num_rounds : ℕ := 8

/-- The number of points we're interested in -/
def target_points : ℕ := 5

theorem tennis_tournament_result :
  f 8 num_rounds target_points = 56 :=
sorry

#eval f 8 num_rounds target_points

end tennis_tournament_result_l924_92497


namespace function_property_l924_92412

theorem function_property (k : ℝ) (h_k : k > 0) :
  ∀ (f : ℝ → ℝ), 
  (∀ (x : ℝ), x > 0 → (f (x^2 + 1))^(Real.sqrt x) = k) →
  ∀ (y : ℝ), y > 0 → (f ((9 + y^2) / y^2))^(Real.sqrt (12 / y)) = k^2 := by
  sorry

end function_property_l924_92412


namespace product_is_2008th_power_l924_92483

theorem product_is_2008th_power : ∃ (a b c : ℕ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  ((a = (b + c) / 2) ∨ (b = (a + c) / 2) ∨ (c = (a + b) / 2)) ∧
  (∃ (n : ℕ), a * b * c = n ^ 2008) :=
by sorry

end product_is_2008th_power_l924_92483


namespace one_third_minus_decimal_l924_92433

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / 3000 := by
  sorry

end one_third_minus_decimal_l924_92433


namespace function_non_negative_l924_92494

theorem function_non_negative 
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * (deriv f x) > 0) : 
  ∀ x, f x ≥ 0 := by
  sorry

end function_non_negative_l924_92494


namespace shelleys_weight_l924_92418

/-- Given the weights of three people on a scale in pairs, find one person's weight -/
theorem shelleys_weight (p s r : ℕ) : 
  p + s = 151 → s + r = 132 → p + r = 115 → s = 84 := by
  sorry

end shelleys_weight_l924_92418


namespace binomial_expansion_properties_l924_92421

theorem binomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (2*x - 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  (a₁ + a₂ + a₃ + a₄ = -80) ∧ 
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 625) := by
sorry

end binomial_expansion_properties_l924_92421


namespace function_and_tangent_line_l924_92408

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- State the theorem
theorem function_and_tangent_line 
  (a b c : ℝ) 
  (h1 : ∃ (x : ℝ), x = 2 ∧ (3 * a * x^2 + 2 * b * x + c = 0)) -- extremum at x = 2
  (h2 : f a b c 2 = -6) -- f(2) = -6
  (h3 : c = -4) -- f'(0) = -4
  : 
  (∀ x, f a b c x = x^3 - 2*x^2 - 4*x + 2) ∧ -- f(x) = x³ - 2x² - 4x + 2
  (∃ (m n : ℝ), m = 3 ∧ n = 6 ∧ ∀ x y, y = (f a b c (-1)) + (3 * a * (-1)^2 + 2 * b * (-1) + c) * (x - (-1)) ↔ m * x - y + n = 0) -- Tangent line equation at x = -1
  := by sorry

end function_and_tangent_line_l924_92408


namespace diamond_olivine_difference_l924_92456

theorem diamond_olivine_difference (agate olivine diamond : ℕ) : 
  agate = 30 →
  olivine = agate + 5 →
  diamond > olivine →
  agate + olivine + diamond = 111 →
  diamond - olivine = 11 :=
by sorry

end diamond_olivine_difference_l924_92456


namespace max_product_sum_2020_l924_92402

theorem max_product_sum_2020 :
  (∃ (x y : ℤ), x + y = 2020 ∧ x * y = 1020100) ∧
  (∀ (a b : ℤ), a + b = 2020 → a * b ≤ 1020100) := by
  sorry

end max_product_sum_2020_l924_92402


namespace order_amount_for_88200_l924_92444

/-- Calculates the discount rate based on the order quantity -/
def discount_rate (x : ℕ) : ℚ :=
  if x < 250 then 0
  else if x < 500 then 1/20
  else if x < 1000 then 1/10
  else 3/20

/-- Calculates the payable amount given the order quantity and unit price -/
def payable_amount (x : ℕ) (A : ℚ) : ℚ :=
  A * x * (1 - discount_rate x)

/-- The unit price determined from the given condition -/
def unit_price : ℚ := 100

theorem order_amount_for_88200 :
  payable_amount 980 unit_price = 88200 :=
sorry

end order_amount_for_88200_l924_92444


namespace john_nada_money_multiple_l924_92436

/-- Given the money distribution among Ali, Nada, and John, prove that John has 4 times Nada's amount. -/
theorem john_nada_money_multiple (total : ℕ) (john_money : ℕ) (nada_money : ℕ) :
  total = 67 →
  john_money = 48 →
  nada_money + (nada_money - 5) + john_money = total →
  john_money = 4 * nada_money :=
by sorry

end john_nada_money_multiple_l924_92436


namespace yellow_to_red_ratio_l924_92426

/-- Represents the number of chairs of each color in Rodrigo's classroom --/
structure ChairCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- Represents the state of chairs in Rodrigo's classroom --/
def classroom_state : ChairCounts → Prop
  | ⟨red, yellow, blue⟩ => 
    red = 4 ∧ 
    blue = yellow - 2 ∧ 
    red + yellow + blue = 18 ∧ 
    red + yellow + blue - 3 = 15

/-- The theorem stating the ratio of yellow to red chairs --/
theorem yellow_to_red_ratio (chairs : ChairCounts) :
  classroom_state chairs → chairs.yellow / chairs.red = 2 := by
  sorry

#check yellow_to_red_ratio

end yellow_to_red_ratio_l924_92426


namespace min_time_two_students_l924_92439

-- Define the processes
inductive Process : Type
| A | B | C | D | E | F | G

-- Define the time required for each process
def processTime (p : Process) : Nat :=
  match p with
  | Process.A => 9
  | Process.B => 9
  | Process.C => 7
  | Process.D => 9
  | Process.E => 7
  | Process.F => 10
  | Process.G => 2

-- Define the dependency relation between processes
def dependsOn : Process → Process → Prop
| Process.C, Process.A => True
| Process.D, Process.A => True
| Process.E, Process.B => True
| Process.E, Process.D => True
| Process.F, Process.C => True
| Process.F, Process.D => True
| _, _ => False

-- Define a valid schedule as a list of processes
def ValidSchedule (schedule : List Process) : Prop := sorry

-- Define the time taken by a schedule
def scheduleTime (schedule : List Process) : Nat := sorry

-- Theorem: The minimum time for two students to complete the project is 28 minutes
theorem min_time_two_students :
  ∃ (schedule1 schedule2 : List Process),
    ValidSchedule schedule1 ∧
    ValidSchedule schedule2 ∧
    (∀ p, p ∈ schedule1 ∨ p ∈ schedule2) ∧
    max (scheduleTime schedule1) (scheduleTime schedule2) = 28 ∧
    (∀ s1 s2, ValidSchedule s1 → ValidSchedule s2 →
      (∀ p, p ∈ s1 ∨ p ∈ s2) →
      max (scheduleTime s1) (scheduleTime s2) ≥ 28) := by sorry

end min_time_two_students_l924_92439


namespace tangent_parallel_to_x_axis_l924_92495

/-- The function f(x) = x^4 - 4x -/
def f (x : ℝ) : ℝ := x^4 - 4*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 4*x^3 - 4

theorem tangent_parallel_to_x_axis :
  ∃ (x y : ℝ), f x = y ∧ f_derivative x = 0 → x = 1 ∧ y = -3 := by
  sorry

end tangent_parallel_to_x_axis_l924_92495


namespace reciprocal_of_complex_l924_92406

/-- Given a complex number z = -1 + √3i, prove that its reciprocal is -1/4 - (√3/4)i -/
theorem reciprocal_of_complex (z : ℂ) : 
  z = -1 + Complex.I * Real.sqrt 3 → 
  z⁻¹ = -(1/4 : ℂ) - Complex.I * ((Real.sqrt 3)/4) := by
  sorry

end reciprocal_of_complex_l924_92406


namespace R_value_at_S_5_l924_92427

/-- Given R = gS^2 - 4S, and R = 11 when S = 3, prove that R = 395/9 when S = 5 -/
theorem R_value_at_S_5 (g : ℚ) :
  (∀ S : ℚ, g * S^2 - 4 * S = 11 → S = 3) →
  g * 5^2 - 4 * 5 = 395 / 9 := by
sorry

end R_value_at_S_5_l924_92427


namespace race_result_l924_92437

/-- Represents the state of the race between Alex and Max -/
structure RaceState where
  alex_lead : Int
  distance_covered : Int

/-- Calculates the remaining distance for Max to catch up to Alex -/
def remaining_distance (total_length : Int) (final_state : RaceState) : Int :=
  total_length - final_state.distance_covered - final_state.alex_lead

/-- Updates the race state after a change in lead -/
def update_state (state : RaceState) (lead_change : Int) : RaceState :=
  { alex_lead := state.alex_lead + lead_change,
    distance_covered := state.distance_covered }

theorem race_result (total_length : Int) (initial_even : Int) (alex_lead1 : Int) 
                     (max_lead : Int) (alex_lead2 : Int) : 
  total_length = 5000 →
  initial_even = 200 →
  alex_lead1 = 300 →
  max_lead = 170 →
  alex_lead2 = 440 →
  let initial_state : RaceState := { alex_lead := 0, distance_covered := initial_even }
  let state1 := update_state initial_state alex_lead1
  let state2 := update_state state1 (-max_lead)
  let final_state := update_state state2 alex_lead2
  remaining_distance total_length final_state = 4430 := by
  sorry

#check race_result

end race_result_l924_92437


namespace trig_identity_l924_92466

theorem trig_identity (α : Real) (h : Real.tan α = 3) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = 6 / 5 := by
  sorry

end trig_identity_l924_92466


namespace average_of_numbers_l924_92407

def numbers : List ℝ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140]

theorem average_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 125397.5 ∧
  (numbers.sum / numbers.length : ℝ) ≠ 858.5454545454545 := by
  sorry

end average_of_numbers_l924_92407


namespace last_group_count_l924_92467

theorem last_group_count (total : Nat) (total_avg : ℚ) (first_group : Nat) (first_avg : ℚ) (middle : ℚ) (last_avg : ℚ) 
  (h_total : total = 13)
  (h_total_avg : total_avg = 60)
  (h_first_group : first_group = 6)
  (h_first_avg : first_avg = 57)
  (h_middle : middle = 50)
  (h_last_avg : last_avg = 61) :
  ∃ (last_group : Nat), last_group = total - first_group - 1 ∧ last_group = 6 := by
  sorry

#check last_group_count

end last_group_count_l924_92467


namespace crazy_silly_school_movies_l924_92461

theorem crazy_silly_school_movies :
  let number_of_books : ℕ := 8
  let movies_more_than_books : ℕ := 2
  let number_of_movies : ℕ := number_of_books + movies_more_than_books
  number_of_movies = 10 := by
  sorry

end crazy_silly_school_movies_l924_92461


namespace simplify_fraction_l924_92438

theorem simplify_fraction : 5 * (18 / 7) * (21 / -45) = -6 / 5 := by
  sorry

end simplify_fraction_l924_92438


namespace complex_magnitude_squared_l924_92430

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = 6 + 10 * Complex.I) : 
  Complex.abs z ^ 2 = 1156 / 9 := by
  sorry

end complex_magnitude_squared_l924_92430


namespace investor_share_price_l924_92448

theorem investor_share_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.125)
  (h2 : face_value = 40)
  (h3 : roi = 0.25) :
  let dividend_per_share := dividend_rate * face_value
  let price := dividend_per_share / roi
  price = 20 := by sorry

end investor_share_price_l924_92448


namespace odd_function_zero_condition_l924_92431

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_zero_condition :
  (∀ f : RealFunction, IsOdd f → f 0 = 0) ∧
  (∃ f : RealFunction, f 0 = 0 ∧ ¬IsOdd f) :=
sorry

end odd_function_zero_condition_l924_92431


namespace relay_race_arrangements_l924_92486

/-- The number of students in the class --/
def total_students : ℕ := 6

/-- The number of students needed for the relay race --/
def relay_team_size : ℕ := 4

/-- The possible positions for student A --/
inductive PositionA
| first
| second

/-- The possible positions for student B --/
inductive PositionB
| second
| fourth

/-- A function to calculate the number of arrangements --/
def count_arrangements (total : ℕ) (team_size : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the number of arrangements is 36 --/
theorem relay_race_arrangements :
  count_arrangements total_students relay_team_size = 36 :=
sorry

end relay_race_arrangements_l924_92486


namespace calculation_proof_l924_92473

theorem calculation_proof : 2.5 * 8 * (5.2 + 4.8)^2 = 2000 := by
  sorry

end calculation_proof_l924_92473


namespace police_emergency_number_prime_divisor_l924_92496

theorem police_emergency_number_prime_divisor (k : ℕ) :
  ∃ (p : ℕ), Prime p ∧ p > 7 ∧ p ∣ (1000 * k + 133) := by
  sorry

end police_emergency_number_prime_divisor_l924_92496


namespace boys_from_beethoven_l924_92484

/-- Given the following conditions about a music camp:
  * There are 120 total students
  * There are 65 boys and 55 girls
  * 50 students are from Mozart Middle School
  * 70 students are from Beethoven Middle School
  * 17 girls are from Mozart Middle School
  This theorem proves that there are 32 boys from Beethoven Middle School -/
theorem boys_from_beethoven (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (mozart_students : ℕ) (beethoven_students : ℕ) (mozart_girls : ℕ) :
  total_students = 120 →
  total_boys = 65 →
  total_girls = 55 →
  mozart_students = 50 →
  beethoven_students = 70 →
  mozart_girls = 17 →
  beethoven_students - (beethoven_students - total_boys + mozart_students - mozart_girls) = 32 :=
by sorry

end boys_from_beethoven_l924_92484


namespace profit_maximized_at_optimal_reduction_optimal_reduction_is_five_profit_function_correct_l924_92429

/-- Profit function for a product with given initial conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 10*x + 600

/-- The price reduction that maximizes profit -/
def optimal_reduction : ℝ := 5

theorem profit_maximized_at_optimal_reduction :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_reduction :=
sorry

theorem optimal_reduction_is_five :
  optimal_reduction = 5 :=
sorry

theorem profit_function_correct (x : ℝ) :
  profit_function x = (100 - 70 - x) * (20 + x) :=
sorry

end profit_maximized_at_optimal_reduction_optimal_reduction_is_five_profit_function_correct_l924_92429


namespace square_area_10m_l924_92441

/-- The area of a square with side length 10 meters is 100 square meters. -/
theorem square_area_10m : 
  let side_length : ℝ := 10
  let square_area := side_length ^ 2
  square_area = 100 := by sorry

end square_area_10m_l924_92441


namespace shift_increasing_interval_l924_92413

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem shift_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 5)) (-7) (-2) :=
by
  sorry

end shift_increasing_interval_l924_92413


namespace mean_of_middle_numbers_l924_92460

theorem mean_of_middle_numbers (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 90 →
  max a (max b (max c d)) = 105 →
  min a (min b (min c d)) = 75 →
  (a + b + c + d - 105 - 75) / 2 = 90 := by
sorry

end mean_of_middle_numbers_l924_92460


namespace triangle_angle_problem_l924_92419

theorem triangle_angle_problem (A B C : ℕ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle
  A < B →
  B < C →
  4 * C = 7 * A →
  B = 59 := by
sorry

end triangle_angle_problem_l924_92419


namespace jar_water_problem_l924_92400

theorem jar_water_problem (s l : ℝ) (hs : s > 0) (hl : l > 0) : 
  (1/8 : ℝ) * s = (1/6 : ℝ) * l → (1/6 : ℝ) * l + (1/8 : ℝ) * s = (1/3 : ℝ) * l :=
by sorry

end jar_water_problem_l924_92400


namespace original_men_count_prove_original_men_count_l924_92411

/-- Represents the amount of work to be done -/
def work : ℝ := 1

/-- The number of days taken by the original group to complete the work -/
def original_days : ℕ := 60

/-- The number of days taken by the augmented group to complete the work -/
def augmented_days : ℕ := 50

/-- The number of additional men in the augmented group -/
def additional_men : ℕ := 8

/-- Theorem stating that the original number of men is 48 -/
theorem original_men_count : ℕ :=
  48

/-- Proof that the original number of men is 48 -/
theorem prove_original_men_count : 
  ∃ (m : ℕ), 
    (m * (work / original_days) = (m + additional_men) * (work / augmented_days)) ∧ 
    (m = original_men_count) := by
  sorry

end original_men_count_prove_original_men_count_l924_92411


namespace smallest_x_with_remainders_l924_92469

theorem smallest_x_with_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧ 
  ∀ y : ℕ, y > 0 → y % 3 = 2 → y % 4 = 3 → y % 5 = 4 → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_x_with_remainders_l924_92469


namespace vampire_conversion_theorem_l924_92488

/-- The number of people each vampire turns into vampires per night. -/
def vampire_conversion_rate : ℕ → Prop := λ x =>
  let initial_population : ℕ := 300
  let initial_vampires : ℕ := 2
  let nights : ℕ := 2
  let final_vampires : ℕ := 72
  
  -- After first night: initial_vampires + (initial_vampires * x)
  -- After second night: (initial_vampires + (initial_vampires * x)) + 
  --                     (initial_vampires + (initial_vampires * x)) * x
  
  (initial_vampires + (initial_vampires * x)) + 
  (initial_vampires + (initial_vampires * x)) * x = final_vampires

theorem vampire_conversion_theorem : vampire_conversion_rate 5 := by
  sorry

end vampire_conversion_theorem_l924_92488


namespace complement_A_in_U_l924_92409

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≥ 1}

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -1 ∨ x = 1 ∨ x > 2} := by sorry

end complement_A_in_U_l924_92409


namespace article_sale_loss_l924_92474

theorem article_sale_loss (cost : ℝ) (profit_rate : ℝ) (discount_rate : ℝ) : 
  profit_rate = 0.425 → 
  discount_rate = 2/3 →
  let original_price := cost * (1 + profit_rate)
  let discounted_price := original_price * discount_rate
  let loss := cost - discounted_price
  let loss_rate := loss / cost
  loss_rate = 0.05 := by
  sorry

end article_sale_loss_l924_92474


namespace percentage_of_burpees_l924_92445

/-- Calculate the percentage of burpees in Emmett's workout routine -/
theorem percentage_of_burpees (jumping_jacks pushups situps burpees lunges : ℕ) :
  jumping_jacks = 25 →
  pushups = 15 →
  situps = 30 →
  burpees = 10 →
  lunges = 20 →
  (burpees : ℚ) / (jumping_jacks + pushups + situps + burpees + lunges) * 100 = 10 := by
  sorry

end percentage_of_burpees_l924_92445


namespace number_of_operations_is_important_indicator_l924_92410

-- Define the concept of an algorithm
structure Algorithm where
  operations : ℕ → ℕ  -- Number of operations as a function of input size

-- Define the concept of computer characteristics
structure ComputerCharacteristics where
  speed_importance : Prop  -- Speed is an important characteristic

-- Define the concept of algorithm quality indicators
structure QualityIndicator where
  is_important : Prop  -- Whether the indicator is important for algorithm quality

-- Define the specific indicator for number of operations
def number_of_operations : QualityIndicator where
  is_important := sorry  -- We'll prove this

-- State the theorem
theorem number_of_operations_is_important_indicator 
  (computer : ComputerCharacteristics) 
  (algo_quality_multifactor : Prop) : 
  computer.speed_importance → 
  algo_quality_multifactor → 
  number_of_operations.is_important :=
by sorry


end number_of_operations_is_important_indicator_l924_92410


namespace airplane_luggage_problem_l924_92471

/-- Calculates the number of bags per person given the problem conditions -/
def bagsPerPerson (numPeople : ℕ) (bagWeight : ℕ) (totalCapacity : ℕ) (additionalBags : ℕ) : ℕ :=
  let totalBags := totalCapacity / bagWeight
  let currentBags := totalBags - additionalBags
  currentBags / numPeople

/-- Theorem stating that under the given conditions, each person has 5 bags -/
theorem airplane_luggage_problem :
  bagsPerPerson 6 50 6000 90 = 5 := by
  sorry

end airplane_luggage_problem_l924_92471


namespace cos_power_five_identity_l924_92416

/-- For all real angles θ, cos^5 θ = (1/64) cos 5θ + (65/64) cos θ -/
theorem cos_power_five_identity (θ : ℝ) : 
  Real.cos θ ^ 5 = (1/64) * Real.cos (5 * θ) + (65/64) * Real.cos θ := by
  sorry

end cos_power_five_identity_l924_92416


namespace stacy_heather_walk_stacy_heather_initial_distance_l924_92475

/-- The problem of Stacy and Heather walking towards each other -/
theorem stacy_heather_walk (stacy_speed heather_speed : ℝ) 
  (heather_start_delay : ℝ) (heather_distance : ℝ) : ℝ :=
  let initial_distance : ℝ := 
    by {
      -- Define the conditions
      have h1 : stacy_speed = heather_speed + 1 := by sorry
      have h2 : heather_speed = 5 := by sorry
      have h3 : heather_start_delay = 24 / 60 := by sorry
      have h4 : heather_distance = 5.7272727272727275 := by sorry

      -- Calculate the initial distance
      sorry
    }
  initial_distance

/-- The theorem stating that Stacy and Heather were initially 15 miles apart -/
theorem stacy_heather_initial_distance : 
  stacy_heather_walk 6 5 (24/60) 5.7272727272727275 = 15 := by sorry

end stacy_heather_walk_stacy_heather_initial_distance_l924_92475


namespace systematic_sampling_probability_l924_92415

/-- The probability of an individual being selected in systematic sampling -/
theorem systematic_sampling_probability 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 1003)
  (h2 : sample_size = 50) :
  (sample_size : ℚ) / population_size = 50 / 1003 := by
  sorry

end systematic_sampling_probability_l924_92415


namespace base_five_of_232_l924_92446

def base_five_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_five_of_232 :
  base_five_repr 232 = [1, 4, 1, 2] := by
sorry

end base_five_of_232_l924_92446


namespace profit_percent_calculation_l924_92478

/-- Given an article with a certain selling price, prove that the profit percent is 42.5%
    when selling at 2/3 of that price would result in a loss of 5%. -/
theorem profit_percent_calculation (P : ℝ) (C : ℝ) (h : (2/3) * P = 0.95 * C) :
  (P - C) / C * 100 = 42.5 := by
  sorry

end profit_percent_calculation_l924_92478


namespace fraction_relation_l924_92417

theorem fraction_relation (m n p s : ℝ) 
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / s = 1 / 9) :
  m / s = 1 / 2 := by
sorry

end fraction_relation_l924_92417


namespace right_triangle_third_side_product_l924_92455

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end right_triangle_third_side_product_l924_92455


namespace triangle_excircle_radii_relation_l924_92453

/-- For a triangle ABC with side lengths a, b, c and excircle radii r_a, r_b, r_c opposite to vertices A, B, C respectively -/
theorem triangle_excircle_radii_relation 
  (a b c r_a r_b r_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_excircle : r_a = (a + b + c) * (b + c - a) / (4 * (b + c)) ∧
                r_b = (a + b + c) * (c + a - b) / (4 * (c + a)) ∧
                r_c = (a + b + c) * (a + b - c) / (4 * (a + b))) :
  a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b)) = 2 := by
sorry

end triangle_excircle_radii_relation_l924_92453


namespace product_of_fractions_equals_81_l924_92463

theorem product_of_fractions_equals_81 : 
  (1 / 3) * (9 / 1) * (1 / 27) * (81 / 1) * (1 / 243) * (729 / 1) * (1 / 2187) * (6561 / 1) = 81 := by
  sorry

end product_of_fractions_equals_81_l924_92463


namespace A_empty_iff_a_in_range_l924_92493

/-- The set A for a given real number a -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - a * x + 1 ≤ 0}

/-- Theorem stating the equivalence between A being empty and the range of a -/
theorem A_empty_iff_a_in_range : 
  ∀ a : ℝ, A a = ∅ ↔ 0 ≤ a ∧ a < 4 := by sorry

end A_empty_iff_a_in_range_l924_92493


namespace find_B_value_l924_92422

theorem find_B_value (A B : ℕ) (h1 : A < 10) (h2 : B < 10) 
  (h3 : 600 + 10 * A + 5 + 100 * B + 3 = 748) : B = 1 := by
  sorry

end find_B_value_l924_92422


namespace exp_sum_greater_than_two_l924_92464

theorem exp_sum_greater_than_two (a b : ℝ) (h1 : a ≠ b) (h2 : a * Real.exp b - b * Real.exp a = Real.exp a - Real.exp b) : 
  Real.exp a + Real.exp b > 2 := by
  sorry

end exp_sum_greater_than_two_l924_92464


namespace solution_set_inequalities_l924_92432

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end solution_set_inequalities_l924_92432
