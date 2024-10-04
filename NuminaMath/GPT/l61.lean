import Mathlib

namespace problems_per_page_l61_61189

def total_problems : ℕ := 60
def finished_problems : ℕ := 20
def remaining_pages : ℕ := 5

theorem problems_per_page :
  (total_problems - finished_problems) / remaining_pages = 8 :=
by
  sorry

end problems_per_page_l61_61189


namespace negation_of_forall_statement_l61_61162

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l61_61162


namespace proof_problem_l61_61230

variable {a b m n x : ℝ}

theorem proof_problem (h1 : a = -b) (h2 : m * n = 1) (h3 : m ≠ n) (h4 : |x| = 2) :
    (-2 * m * n + (b + a) / (m - n) - x = -4 ∧ x = 2) ∨
    (-2 * m * n + (b + a) / (m - n) - x = 0 ∧ x = -2) :=
by
  sorry

end proof_problem_l61_61230


namespace identity_x_squared_minus_y_squared_l61_61844

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61844


namespace age_of_B_present_l61_61353

theorem age_of_B_present (A B C : ℕ) (h1 : A + B + C = 90)
  (h2 : (A - 10) * 2 = (B - 10))
  (h3 : (B - 10) * 3 = (C - 10) * 2) :
  B = 30 := 
sorry

end age_of_B_present_l61_61353


namespace quadratic_inequality_l61_61041

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - 3*x - m = 0 ∧ (∃ y : ℝ, y^2 - 3*y - m = 0 ∧ x ≠ y)) ↔ m > - 9 / 4 := 
by
  sorry

end quadratic_inequality_l61_61041


namespace rectangle_width_l61_61731

theorem rectangle_width (L W : ℕ)
  (h1 : W = L + 3)
  (h2 : 2 * L + 2 * W = 54) :
  W = 15 :=
by
  sorry

end rectangle_width_l61_61731


namespace equation_of_circle_ABC_l61_61613

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61613


namespace triangle_areas_l61_61339

theorem triangle_areas (S₁ S₂ : ℝ) :
  ∃ (ABC : ℝ), ABC = Real.sqrt (S₁ * S₂) :=
sorry

end triangle_areas_l61_61339


namespace total_payment_correct_l61_61524

def cost (n : ℕ) : ℕ :=
  if n <= 10 then n * 25
  else 10 * 25 + (n - 10) * (4 * 25 / 5)

def final_cost_with_discount (n : ℕ) : ℕ :=
  let initial_cost := cost n
  if n > 20 then initial_cost - initial_cost / 10
  else initial_cost

def orders_X := 60 * 20 / 100
def orders_Y := 60 * 25 / 100
def orders_Z := 60 * 55 / 100

def cost_X := final_cost_with_discount orders_X
def cost_Y := final_cost_with_discount orders_Y
def cost_Z := final_cost_with_discount orders_Z

theorem total_payment_correct : cost_X + cost_Y + cost_Z = 1279 := by
  sorry

end total_payment_correct_l61_61524


namespace find_polygon_sides_l61_61999

theorem find_polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by
  sorry

end find_polygon_sides_l61_61999


namespace correct_expression_l61_61497

theorem correct_expression (a b c : ℝ) : 3 * a - (2 * b - c) = 3 * a - 2 * b + c :=
sorry

end correct_expression_l61_61497


namespace distance_between_centers_l61_61069

theorem distance_between_centers (r1 r2 d x : ℝ) (h1 : r1 = 10) (h2 : r2 = 6) (h3 : d = 30) :
  x = 2 * Real.sqrt 229 := 
sorry

end distance_between_centers_l61_61069


namespace circle_through_points_l61_61671

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61671


namespace Kates_hair_length_l61_61302

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l61_61302


namespace division_addition_problem_l61_61082

theorem division_addition_problem :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end division_addition_problem_l61_61082


namespace ferry_speed_difference_l61_61222

variable (v_P v_Q d_P d_Q t_P t_Q x : ℝ)

-- Defining the constants and conditions provided in the problem
axiom h1 : v_P = 8 
axiom h2 : t_P = 2 
axiom h3 : d_P = t_P * v_P 
axiom h4 : d_Q = 3 * d_P 
axiom h5 : t_Q = t_P + 2
axiom h6 : d_Q = v_Q * t_Q 
axiom h7 : x = v_Q - v_P 

-- The theorem that corresponds to the solution
theorem ferry_speed_difference : x = 4 := by
  sorry

end ferry_speed_difference_l61_61222


namespace tangent_line_k_value_l61_61096

theorem tangent_line_k_value : ∃ k : ℝ, (∀ (x y : ℝ), 4 * x + 7 * y + k = 0 → y^2 = 16 * x → k = 49) :=
begin
  sorry
end

end tangent_line_k_value_l61_61096


namespace equation_of_circle_passing_through_points_l61_61685

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61685


namespace find_circle_equation_l61_61708

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61708


namespace total_bricks_used_l61_61314

def numberOfCoursesPerWall := 6
def bricksPerCourse := 10
def numberOfWalls := 4
def incompleteCourses := 2

theorem total_bricks_used :
  (numberOfCoursesPerWall * bricksPerCourse * (numberOfWalls - 1)) + ((numberOfCoursesPerWall - incompleteCourses) * bricksPerCourse) = 220 :=
by
  -- Proof goes here
  sorry

end total_bricks_used_l61_61314


namespace election_winner_votes_l61_61909

-- Define the conditions and question in Lean 4
theorem election_winner_votes (V : ℝ) (h1 : V > 0) 
  (h2 : 0.54 * V - 0.46 * V = 288) : 0.54 * V = 1944 :=
by
  sorry

end election_winner_votes_l61_61909


namespace integer_values_of_x_l61_61248

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l61_61248


namespace drink_price_half_promotion_l61_61358

theorem drink_price_half_promotion (P : ℝ) (h : P + (1/2) * P = 13.5) : P = 9 := 
by
  sorry

end drink_price_half_promotion_l61_61358


namespace inequality_solution_l61_61013

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l61_61013


namespace face_value_of_share_l61_61753

theorem face_value_of_share (FV : ℝ) (market_value : ℝ) (dividend_rate : ℝ) (desired_return_rate : ℝ) 
  (H1 : market_value = 15) 
  (H2 : dividend_rate = 0.09) 
  (H3 : desired_return_rate = 0.12) 
  (H4 : dividend_rate * FV = desired_return_rate * market_value) :
  FV = 20 := 
by
  sorry

end face_value_of_share_l61_61753


namespace negation_of_proposition_l61_61159

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l61_61159


namespace hare_probability_l61_61547

-- Definitions for events
def P (x : Type) [ProbabilityMeasure x] := measure x
def event_A : Prop := -- The creature is a rabbit
def event_B : Prop := -- The creature declared it is not a rabbit
def event_C : Prop := -- The creature declared it is not a hare

-- Given conditions
variable [ProbabilityMeasure (event_A : ℙ)] (h1 : P[event_A] = 1/2)
variable (h2 : P[event_B | event_A] = 3/4) -- Probability declared not rabbit given is rabbit (75% correct)
variable (h3 : P[event_C | event_A] = 1/4) -- Probability declared not hare given is rabbit (25% wrong)
variable (h4 : P[¬event_A | event_B] = 2/3) -- Probability declared not rabbit given is hare (67% wrong)
variable (h5 : P[¬event_A | event_C] = 1/3) -- Probability declared not hare given is hare (33% wrong)

-- Goal: Prove the conditional probability
theorem hare_probability :
  P[event_A | event_B ∧ event_C] = 27 / 59 := by
  sorry

end hare_probability_l61_61547


namespace distance_between_foci_of_given_ellipse_l61_61065

noncomputable def distance_between_foci_of_ellipse : ℝ :=
  let h := 6
  let k := 3
  let a := h
  let b := k
  real.sqrt ((a : ℝ)^2 - (b : ℝ)^2)

theorem distance_between_foci_of_given_ellipse :
  distance_between_foci_of_ellipse = 6 * real.sqrt 3 :=
by
  let h := 6
  let k := 3
  let a := h
  let b := k
  calc
    distance_between_foci_of_ellipse
        = real.sqrt (a^2 - b^2) : rfl
    ... = real.sqrt (6^2 - 3^2) : by norm_num
    ... = real.sqrt 27 : by norm_num
    ... = 3 * real.sqrt 3 : by norm_num
  done

end distance_between_foci_of_given_ellipse_l61_61065


namespace scientific_notation_500_billion_l61_61206

theorem scientific_notation_500_billion :
  ∃ (a : ℝ), 500000000000 = a * 10 ^ 10 ∧ 1 ≤ a ∧ a < 10 :=
by
  sorry

end scientific_notation_500_billion_l61_61206


namespace circle_through_points_l61_61643

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61643


namespace price_decrease_required_to_initial_l61_61170

theorem price_decrease_required_to_initial :
  let P0 := 100.0
  let P1 := P0 * 1.15
  let P2 := P1 * 0.90
  let P3 := P2 * 1.20
  let P4 := P3 * 0.70
  let P5 := P4 * 1.10
  let P6 := P5 * (1.0 - d / 100.0)
  P6 = P0 -> d = 5.0 :=
by
  sorry

end price_decrease_required_to_initial_l61_61170


namespace third_factorial_is_7_l61_61038

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Problem conditions
def b : ℕ := 9
def factorial_b_minus_2 : ℕ := factorial (b - 2)
def factorial_b_plus_1 : ℕ := factorial (b + 1)
def GCD_value : ℕ := Nat.gcd (Nat.gcd factorial_b_minus_2 factorial_b_plus_1) (factorial 7)

-- Theorem statement
theorem third_factorial_is_7 :
  Nat.gcd (Nat.gcd (factorial (b - 2)) (factorial (b + 1))) (factorial 7) = 5040 →
  ∃ k : ℕ, factorial k = 5040 ∧ k = 7 :=
by
  sorry

end third_factorial_is_7_l61_61038


namespace prove_y_identity_l61_61320

theorem prove_y_identity (y : ℤ) (h1 : y^2 = 2209) : (y + 2) * (y - 2) = 2205 :=
by
  sorry

end prove_y_identity_l61_61320


namespace roses_left_unsold_l61_61766

def price_per_rose : ℕ := 4
def initial_roses : ℕ := 13
def total_earned : ℕ := 36

theorem roses_left_unsold : (initial_roses - (total_earned / price_per_rose) = 4) :=
by
  sorry

end roses_left_unsold_l61_61766


namespace shaded_rectangle_area_l61_61767

def area_polygon : ℝ := 2016
def sides_polygon : ℝ := 18
def segments_persh : ℝ := 4

theorem shaded_rectangle_area :
  (area_polygon / sides_polygon) * segments_persh = 448 := 
sorry

end shaded_rectangle_area_l61_61767


namespace divisors_remainder_5_l61_61496

theorem divisors_remainder_5 (d : ℕ) : d ∣ 2002 ∧ d > 5 ↔ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 14 ∨ 
                                      d = 22 ∨ d = 26 ∨ d = 77 ∨ d = 91 ∨ 
                                      d = 143 ∨ d = 154 ∨ d = 182 ∨ d = 286 ∨ 
                                      d = 1001 ∨ d = 2002 :=
by sorry

end divisors_remainder_5_l61_61496


namespace correct_expression_l61_61061

-- Definitions for the problem options.
def optionA (m n : ℕ) : ℕ := 2 * m + n
def optionB (m n : ℕ) : ℕ := m + 2 * n
def optionC (m n : ℕ) : ℕ := 2 * (m + n)
def optionD (m n : ℕ) : ℕ := (m + n) ^ 2

-- Statement for the proof problem.
theorem correct_expression (m n : ℕ) : optionB m n = m + 2 * n :=
by sorry

end correct_expression_l61_61061


namespace naomi_total_time_l61_61311

-- Definitions
def time_to_parlor : ℕ := 60
def speed_ratio : ℕ := 2 -- because her returning speed is half of the going speed
def first_trip_delay : ℕ := 15
def coffee_break : ℕ := 10
def second_trip_delay : ℕ := 20
def detour_time : ℕ := 30

-- Calculate total round trip times
def first_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + first_trip_delay + coffee_break
def second_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + second_trip_delay + detour_time

-- Hypothesis
def total_round_trip_time : ℕ := first_round_trip_time + second_round_trip_time

-- Main theorem statement
theorem naomi_total_time : total_round_trip_time = 435 := by
  sorry

end naomi_total_time_l61_61311


namespace cube_surface_divisible_into_12_squares_l61_61057

theorem cube_surface_divisible_into_12_squares (a : ℝ) :
  (∃ b : ℝ, b = a / Real.sqrt 2 ∧
  ∀ cube_surface_area: ℝ, cube_surface_area = 6 * a^2 →
  ∀ smaller_square_area: ℝ, smaller_square_area = b^2 →
  12 * smaller_square_area = cube_surface_area) :=
sorry

end cube_surface_divisible_into_12_squares_l61_61057


namespace find_water_and_bucket_weight_l61_61745

-- Define the original amount of water (x) and the weight of the bucket (y)
variables (x y : ℝ)

-- Given conditions described as hypotheses
def conditions (x y : ℝ) : Prop :=
  4 * x + y = 16 ∧ 6 * x + y = 22

-- The goal is to prove the values of x and y
theorem find_water_and_bucket_weight (h : conditions x y) : x = 3 ∧ y = 4 :=
by
  sorry

end find_water_and_bucket_weight_l61_61745


namespace find_circle_equation_l61_61715

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61715


namespace area_of_polygon_l61_61382

-- Define the structure of the equilateral triangular grid
structure TriangularGrid :=
  (points_on_side : ℕ)
  (points : finset (ℕ × ℕ))
  (total_points : points.card = (points_on_side * (points_on_side + 1)) / 2)

-- Define the polygon with specific properties
structure Polygon (G : TriangularGrid) :=
  (vertices : finset (ℕ × ℕ))
  (closed : (∃ v, v ∈ vertices) → (∀ v, v ∈ vertices ↔ v ∈ G.points . subst G))
  (non_selfintersecting : true) -- Placeholder, needs formal definition
  (uses_all_points : vertices = G.points)

-- Given grid G and polygon S, prove the area of S
def polygon_area (G : TriangularGrid) (S : Polygon G) : ℝ :=
  52 * Real.sqrt 3

theorem area_of_polygon (G : TriangularGrid) (S : Polygon G) : 
  polygon_area G S = 52 * Real.sqrt 3 := sorry

end area_of_polygon_l61_61382


namespace find_ordered_pair_l61_61330

-- We need to define the variables and conditions first.
variables (a c : ℝ)

-- Now we state the conditions.
def quadratic_has_one_solution : Prop :=
  a * c = 25 ∧ a + c = 12 ∧ a < c

-- Finally, we state the main goal to prove.
theorem find_ordered_pair (ha : quadratic_has_one_solution a c) :
  a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11 :=
by sorry

end find_ordered_pair_l61_61330


namespace remainder_sum_59_l61_61737

theorem remainder_sum_59 (x y z : ℕ) (h1 : x % 59 = 30) (h2 : y % 59 = 27) (h3 : z % 59 = 4) :
  (x + y + z) % 59 = 2 := 
sorry

end remainder_sum_59_l61_61737


namespace perpendicular_lines_l61_61543

theorem perpendicular_lines (a : ℝ) 
  (h1 : (3 : ℝ) * y + (2 : ℝ) * x - 6 = 0) 
  (h2 : (4 : ℝ) * y + a * x - 5 = 0) : 
  a = -6 :=
sorry

end perpendicular_lines_l61_61543


namespace university_minimum_spend_l61_61493

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def box_cost : ℝ := 1.20
def total_volume : ℝ := 3.06 * (10^6)

def box_volume : ℕ := box_length * box_width * box_height

noncomputable def number_of_boxes : ℕ := Nat.ceil (total_volume / box_volume)
noncomputable def total_cost : ℝ := number_of_boxes * box_cost

theorem university_minimum_spend : total_cost = 612 := by
  sorry

end university_minimum_spend_l61_61493


namespace circle_through_points_l61_61637

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61637


namespace negate_prop_l61_61164

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l61_61164


namespace circle_through_points_l61_61639

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61639


namespace time_for_one_mile_l61_61934

theorem time_for_one_mile (d v : ℝ) (mile_in_feet : ℝ) (num_circles : ℕ) 
  (circle_circumference : ℝ) (distance_in_miles : ℝ) (time : ℝ) :
  d = 50 ∧ v = 10 ∧ mile_in_feet = 5280 ∧ num_circles = 106 ∧ 
  circle_circumference = 50 * Real.pi ∧ 
  distance_in_miles = (106 * 50 * Real.pi) / 5280 ∧ 
  time = distance_in_miles / v →
  time = Real.pi / 10 :=
by {
  sorry
}

end time_for_one_mile_l61_61934


namespace probability_both_selected_l61_61188

-- Given conditions
def jamie_probability : ℚ := 2 / 3
def tom_probability : ℚ := 5 / 7

-- Statement to prove
theorem probability_both_selected :
  jamie_probability * tom_probability = 10 / 21 :=
by
  sorry

end probability_both_selected_l61_61188


namespace sqrt_floor_8_integer_count_l61_61251

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l61_61251


namespace intersection_x_value_l61_61728

theorem intersection_x_value :
  ∀ x y: ℝ,
    (y = 3 * x - 15) ∧ (3 * x + y = 120) → x = 22.5 := by
  sorry

end intersection_x_value_l61_61728


namespace probability_of_continuous_stripe_loop_l61_61775

-- Definitions corresponding to identified conditions:
def cube_faces : ℕ := 6

def diagonal_orientations_per_face : ℕ := 2

def total_stripe_combinations (faces : ℕ) (orientations : ℕ) : ℕ :=
  orientations ^ faces

def satisfying_stripe_combinations : ℕ := 2

-- Proof statement:
theorem probability_of_continuous_stripe_loop :
  (satisfying_stripe_combinations : ℚ) / (total_stripe_combinations cube_faces diagonal_orientations_per_face : ℚ) = 1 / 32 :=
by
  -- Proof goes here
  sorry

end probability_of_continuous_stripe_loop_l61_61775


namespace max_volume_is_16_l61_61914

noncomputable def max_volume (width : ℝ) (material : ℝ) : ℝ :=
  let l := (material - 2 * width) / (2 + 2 * width)
  let h := (material - 2 * l) / (2 * width + 2 * l)
  l * width * h

theorem max_volume_is_16 :
  max_volume 2 32 = 16 :=
by
  sorry

end max_volume_is_16_l61_61914


namespace sum_of_rel_prime_ints_l61_61903

theorem sum_of_rel_prime_ints (a b : ℕ) (h1 : a < 15) (h2 : b < 15) (h3 : a * b + a + b = 71)
    (h4 : Nat.gcd a b = 1) : a + b = 16 := by
  sorry

end sum_of_rel_prime_ints_l61_61903


namespace height_of_tree_l61_61361

-- Definitions based on conditions
def net_gain (hop: ℕ) (slip: ℕ) : ℕ := hop - slip

def total_distance (hours: ℕ) (net_gain: ℕ) (final_hop: ℕ) : ℕ :=
  hours * net_gain + final_hop

-- Conditions
def hop : ℕ := 3
def slip : ℕ := 2
def time : ℕ := 20

-- Deriving the net gain per hour
#eval net_gain hop slip  -- Evaluates to 1

-- Final height proof problem
theorem height_of_tree : total_distance 19 (net_gain hop slip) hop = 22 := by
  sorry  -- Proof to be filled in

end height_of_tree_l61_61361


namespace num_valid_k_values_l61_61786

theorem num_valid_k_values :
  ∃ (s : Finset ℕ), s = { 1, 2, 3, 6, 9, 18 } ∧ s.card = 6 :=
by
  sorry

end num_valid_k_values_l61_61786


namespace solution_set_I_range_of_m_l61_61825

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem solution_set_I (x : ℝ) : f x < 8 ↔ -5 / 2 < x ∧ x < 3 / 2 :=
sorry

theorem range_of_m (m : ℝ) (h : ∃ x, f x ≤ |3 * m + 1|) : m ≤ -5 / 3 ∨ m ≥ 1 :=
sorry

end solution_set_I_range_of_m_l61_61825


namespace ratio_of_segments_l61_61576

theorem ratio_of_segments (a b c r s : ℝ) (h : a / b = 1 / 4)
  (h₁ : c ^ 2 = a ^ 2 + b ^ 2)
  (h₂ : r = a ^ 2 / c)
  (h₃ : s = b ^ 2 / c) :
  r / s = 1 / 16 :=
by
  sorry

end ratio_of_segments_l61_61576


namespace melissa_total_repair_time_l61_61881

def time_flat_shoes := 3 + 8 + 9
def time_sandals :=  4 + 5
def time_high_heels := 6 + 12 + 10

def first_session_flat_shoes := 6 * time_flat_shoes
def first_session_sandals := 4 * time_sandals
def first_session_high_heels := 3 * time_high_heels

def second_session_flat_shoes := 4 * time_flat_shoes
def second_session_sandals := 7 * time_sandals
def second_session_high_heels := 5 * time_high_heels

def total_first_session := first_session_flat_shoes + first_session_sandals + first_session_high_heels
def total_second_session := second_session_flat_shoes + second_session_sandals + second_session_high_heels

def break_time := 15

def total_repair_time := total_first_session + total_second_session
def total_time_including_break := total_repair_time + break_time

theorem melissa_total_repair_time : total_time_including_break = 538 := by
  sorry

end melissa_total_repair_time_l61_61881


namespace sum_of_new_dimensions_l61_61199

theorem sum_of_new_dimensions (s : ℕ) (h₁ : s^2 = 36) (h₂ : s' = s - 1) : s' + s' + s' = 15 :=
sorry

end sum_of_new_dimensions_l61_61199


namespace find_t_l61_61540

theorem find_t (t a b : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60) →
  t = -9 :=
by
  intros h
  -- We'll skip the proof part
  sorry

end find_t_l61_61540


namespace length_of_each_stone_l61_61508

-- Define the dimensions of the hall in decimeters
def hall_length_dm : ℕ := 36 * 10
def hall_breadth_dm : ℕ := 15 * 10

-- Define the width of each stone in decimeters
def stone_width_dm : ℕ := 5

-- Define the number of stones
def number_of_stones : ℕ := 1350

-- Define the total area of the hall
def hall_area : ℕ := hall_length_dm * hall_breadth_dm

-- Define the area of one stone
def stone_area : ℕ := hall_area / number_of_stones

-- Define the length of each stone and state the theorem
theorem length_of_each_stone : (stone_area / stone_width_dm) = 8 :=
by
  sorry

end length_of_each_stone_l61_61508


namespace smallest_positive_period_l61_61412

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem smallest_positive_period (ω : ℝ) (hω : ω > 0)
  (H : ∀ x1 x2 : ℝ, abs (f ω x1 - f ω x2) = 2 → abs (x1 - x2) = Real.pi / 2) :
  ∃ T > 0, T = Real.pi ∧ (∀ x : ℝ, f ω (x + T) = f ω x) := 
sorry

end smallest_positive_period_l61_61412


namespace shrimp_cost_per_pound_l61_61180

theorem shrimp_cost_per_pound 
    (shrimp_per_guest : ℕ) 
    (num_guests : ℕ) 
    (shrimp_per_pound : ℕ) 
    (total_cost : ℝ)
    (H1 : shrimp_per_guest = 5)
    (H2 : num_guests = 40)
    (H3 : shrimp_per_pound = 20)
    (H4 : total_cost = 170) : 
    (total_cost / ((num_guests * shrimp_per_guest) / shrimp_per_pound) = 17) :=
by
    sorry

end shrimp_cost_per_pound_l61_61180


namespace find_b_l61_61462

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (317212435 * 101 - b) % 25 = 0 ∧ b = 13 := by
  sorry

end find_b_l61_61462


namespace candy_bar_cost_correct_l61_61371

noncomputable def candy_bar_cost : ℕ := 25 -- Correct answer from the solution

theorem candy_bar_cost_correct (C : ℤ) (H1 : 3 * C + 150 + 50 = 11 * 25)
  (H2 : ∃ C, C ≥ 0) : C = candy_bar_cost :=
by
  sorry

end candy_bar_cost_correct_l61_61371


namespace evaluate_expression_l61_61090

variable (a b c d : ℝ)

theorem evaluate_expression :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c :=
sorry

end evaluate_expression_l61_61090


namespace RandomEvent_Proof_l61_61739

-- Define the events and conditions
def EventA : Prop := ∀ (θ₁ θ₂ θ₃ : ℝ), θ₁ + θ₂ + θ₃ = 360 → False
def EventB : Prop := ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 6) → n < 7
def EventC : Prop := ∃ (factors : ℕ → ℝ), (∃ (uncertainty : ℕ → ℝ), True)
def EventD : Prop := ∀ (balls : ℕ), (balls = 0 ∨ balls ≠ 0) → False

-- The theorem represents the proof problem
theorem RandomEvent_Proof : EventC :=
by
  sorry

end RandomEvent_Proof_l61_61739


namespace initial_students_count_l61_61151

-- Definitions based on conditions
def initial_average_age (T : ℕ) (n : ℕ) : Prop := T = 14 * n
def new_average_age_after_adding (T : ℕ) (n : ℕ) : Prop := (T + 5 * 17) / (n + 5) = 15

-- Main proposition stating the problem
theorem initial_students_count (n : ℕ) (T : ℕ) 
  (h1 : initial_average_age T n)
  (h2 : new_average_age_after_adding T n) :
  n = 10 :=
by
  sorry

end initial_students_count_l61_61151


namespace area_of_blackboard_l61_61919

def side_length : ℝ := 6
def area (side : ℝ) : ℝ := side * side

theorem area_of_blackboard : area side_length = 36 := by
  -- proof
  sorry

end area_of_blackboard_l61_61919


namespace max_smoothie_servings_l61_61509

def servings (bananas yogurt strawberries : ℕ) : ℕ :=
  min (bananas * 4 / 3) (min (yogurt * 4 / 2) (strawberries * 4 / 1))

theorem max_smoothie_servings :
  servings 9 10 3 = 12 :=
by
  -- Proof steps would be inserted here
  sorry

end max_smoothie_servings_l61_61509


namespace exists_six_numbers_multiple_2002_l61_61099

theorem exists_six_numbers_multiple_2002 (a : Fin 41 → ℕ) (h : Function.Injective a) :
  ∃ (i j k l m n : Fin 41),
    i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    (a i - a j) * (a k - a l) * (a m - a n) % 2002 = 0 := sorry

end exists_six_numbers_multiple_2002_l61_61099


namespace Phillip_correct_total_l61_61449

def number_questions_math : ℕ := 40
def number_questions_english : ℕ := 50
def percentage_correct_math : ℚ := 0.75
def percentage_correct_english : ℚ := 0.98

noncomputable def total_correct_answers : ℚ :=
  (number_questions_math * percentage_correct_math) + (number_questions_english * percentage_correct_english)

theorem Phillip_correct_total : total_correct_answers = 79 := by
  sorry

end Phillip_correct_total_l61_61449


namespace trigonometric_identity_l61_61972

theorem trigonometric_identity (theta : ℝ) (h : Real.cos ((5 * Real.pi)/12 - theta) = 1/3) :
  Real.sin ((Real.pi)/12 + theta) = 1/3 :=
by
  sorry

end trigonometric_identity_l61_61972


namespace circle_equation_through_points_l61_61698

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61698


namespace problem_power_function_l61_61896

-- Defining the conditions
variable {f : ℝ → ℝ}
variable (a : ℝ)
variable (h₁ : ∀ x, f x = x^a)
variable (h₂ : f 2 = Real.sqrt 2)

-- Stating what we need to prove
theorem problem_power_function : f 4 = 2 :=
by sorry

end problem_power_function_l61_61896


namespace find_y_l61_61275

theorem find_y (x y: ℝ) (h1: x = 680) (h2: 0.25 * x = 0.20 * y - 30) : y = 1000 :=
by 
  sorry

end find_y_l61_61275


namespace equation_of_circle_through_three_points_l61_61676

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61676


namespace total_cost_is_correct_l61_61885

-- Define the price of pizzas
def pizza_price : ℕ := 5

-- Define the count of triple cheese and meat lovers pizzas
def triple_cheese_pizzas : ℕ := 10
def meat_lovers_pizzas : ℕ := 9

-- Define the special offers
def buy1get1free (count : ℕ) : ℕ := count / 2 + count % 2
def buy2get1free (count : ℕ) : ℕ := (count / 3) * 2 + count % 3

-- Define the cost calculations using the special offers
def cost_triple_cheese : ℕ := buy1get1free triple_cheese_pizzas * pizza_price
def cost_meat_lovers : ℕ := buy2get1free meat_lovers_pizzas * pizza_price

-- Define the total cost calculation
def total_cost : ℕ := cost_triple_cheese + cost_meat_lovers

-- The theorem we need to prove
theorem total_cost_is_correct :
  total_cost = 55 := by
  sorry

end total_cost_is_correct_l61_61885


namespace equation_of_circle_ABC_l61_61619

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61619


namespace convert_binary_to_decimal_l61_61782

-- Define the binary number and its decimal conversion function
def bin_to_dec (bin : List ℕ) : ℕ :=
  list.foldl (λ acc b, 2 * acc + b) 0 bin

-- Define the specific problem conditions
def binary_oneonezeronethreeone : List ℕ := [1, 1, 0, 1, 1]

theorem convert_binary_to_decimal :
  bin_to_dec binary_oneonezeronethreeone = 27 := by
  sorry

end convert_binary_to_decimal_l61_61782


namespace xy_square_diff_l61_61848

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61848


namespace cubic_boxes_properties_l61_61046

-- Define the lengths of the edges of the cubic boxes
def edge_length_1 : ℝ := 3
def edge_length_2 : ℝ := 5
def edge_length_3 : ℝ := 6

-- Define the volumes of the respective cubic boxes
def volume (edge_length : ℝ) : ℝ := edge_length ^ 3
def volume_1 := volume edge_length_1
def volume_2 := volume edge_length_2
def volume_3 := volume edge_length_3

-- Define the surface areas of the respective cubic boxes
def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)
def surface_area_1 := surface_area edge_length_1
def surface_area_2 := surface_area edge_length_2
def surface_area_3 := surface_area edge_length_3

-- Total volume and surface area calculations
def total_volume := volume_1 + volume_2 + volume_3
def total_surface_area := surface_area_1 + surface_area_2 + surface_area_3

-- Theorem statement to be proven
theorem cubic_boxes_properties :
  total_volume = 368 ∧ total_surface_area = 420 := by
  sorry

end cubic_boxes_properties_l61_61046


namespace circle_passes_through_points_l61_61624

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61624


namespace sequence_eventually_congruent_mod_l61_61550

theorem sequence_eventually_congruent_mod (n : ℕ) (hn : n ≥ 1) : 
  ∃ N, ∀ m ≥ N, ∃ k, m = k * n + N ∧ (2^N.succ - 2^k) % n = 0 :=
by
  sorry

end sequence_eventually_congruent_mod_l61_61550


namespace equation_of_circle_passing_through_points_l61_61687

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61687


namespace rostov_survey_min_players_l61_61514

theorem rostov_survey_min_players :
  ∃ m : ℕ, (∀ n : ℕ, n < m → (95 + n * 1) % 100 ≠ 0) ∧ m = 11 :=
sorry

end rostov_survey_min_players_l61_61514


namespace increase_in_sold_items_l61_61506

variable (P N M : ℝ)
variable (discounted_price := 0.9 * P)
variable (increased_total_income := 1.17 * P * N)

theorem increase_in_sold_items (h: 0.9 * P * M = increased_total_income):
  M = 1.3 * N :=
  by sorry

end increase_in_sold_items_l61_61506


namespace anna_correct_percentage_l61_61940

theorem anna_correct_percentage :
  let test1_problems := 30
  let test1_score := 0.75
  let test2_problems := 50
  let test2_score := 0.85
  let test3_problems := 20
  let test3_score := 0.65
  let correct_test1 := test1_score * test1_problems
  let correct_test2 := test2_score * test2_problems
  let correct_test3 := test3_score * test3_problems
  let total_problems := test1_problems + test2_problems + test3_problems
  let total_correct := correct_test1 + correct_test2 + correct_test3
  (total_correct / total_problems) * 100 = 78 :=
by
  sorry

end anna_correct_percentage_l61_61940


namespace shadow_area_l61_61060

theorem shadow_area (y : ℝ) (cube_side : ℝ) (shadow_excl_area : ℝ) 
  (h₁ : cube_side = 2) 
  (h₂ : shadow_excl_area = 200)
  (h₃ : ((14.28 - 2) / 2 = y)) :
  ⌊1000 * y⌋ = 6140 :=
by
  sorry

end shadow_area_l61_61060


namespace max_total_toads_l61_61908

variable (x y : Nat)
variable (frogs total_frogs : Nat)
variable (total_toads : Nat)

def pond1_frogs := 3 * x
def pond1_toads := 4 * x
def pond2_frogs := 5 * y
def pond2_toads := 6 * y

def all_frogs := pond1_frogs x + pond2_frogs y
def all_toads := pond1_toads x + pond2_toads y

theorem max_total_toads (h_frogs : all_frogs x y = 36) : all_toads x y = 46 := 
sorry

end max_total_toads_l61_61908


namespace exam_scores_l61_61551

theorem exam_scores (A B C D : ℤ) 
  (h1 : A + B = C + D + 17) 
  (h2 : A = B - 4) 
  (h3 : C = D + 5) :
  ∃ highest lowest, (highest - lowest = 13) ∧ 
                   (highest = A ∨ highest = B ∨ highest = C ∨ highest = D) ∧ 
                   (lowest = A ∨ lowest = B ∨ lowest = C ∨ lowest = D) :=
by
  sorry

end exam_scores_l61_61551


namespace angles_same_terminal_side_l61_61375

def angle_equiv (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angles_same_terminal_side : angle_equiv (-390 : ℝ) (330 : ℝ) :=
sorry

end angles_same_terminal_side_l61_61375


namespace solution_set_of_inequality_l61_61959

theorem solution_set_of_inequality:
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_inequality_l61_61959


namespace circle_equation_correct_l61_61608

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61608


namespace circle_equation_correct_l61_61604

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61604


namespace number_of_right_handed_players_l61_61884

/-- 
Given:
(1) There are 70 players on a football team.
(2) 34 players are throwers.
(3) One third of the non-throwers are left-handed.
(4) All throwers are right-handed.
Prove:
The total number of right-handed players is 58.
-/
theorem number_of_right_handed_players 
  (total_players : ℕ) (throwers : ℕ) (non_throwers : ℕ) (left_handed_non_throwers : ℕ) (right_handed_non_throwers : ℕ) : 
  total_players = 70 ∧ throwers = 34 ∧ non_throwers = total_players - throwers ∧ left_handed_non_throwers = non_throwers / 3 ∧ right_handed_non_throwers = non_throwers - left_handed_non_throwers ∧ right_handed_non_throwers + throwers = 58 :=
by
  sorry

end number_of_right_handed_players_l61_61884


namespace min_value2k2_minus_4n_l61_61983

-- We state the problem and set up the conditions
variable (k n : ℝ)
variable (nonneg_k : k ≥ 0)
variable (nonneg_n : n ≥ 0)
variable (eq1 : 2 * k + n = 2)

-- Main statement to prove
theorem min_value2k2_minus_4n : ∃ k n : ℝ, k ≥ 0 ∧ n ≥ 0 ∧ 2 * k + n = 2 ∧ (∀ k' n' : ℝ, k' ≥ 0 ∧ n' ≥ 0 ∧ 2 * k' + n' = 2 → 2 * k'^2 - 4 * n' ≥ -8) := 
sorry

end min_value2k2_minus_4n_l61_61983


namespace contrapositive_false_l61_61763

theorem contrapositive_false : ¬ (∀ x : ℝ, x^2 = 1 → x = 1) → ∀ x : ℝ, x^2 = 1 → x ≠ 1 :=
by
  sorry

end contrapositive_false_l61_61763


namespace find_a_l61_61990

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 := by
  sorry

end find_a_l61_61990


namespace valid_marble_arrangements_eq_48_l61_61499

def ZaraMarbleArrangements (n : ℕ) : ℕ := sorry

theorem valid_marble_arrangements_eq_48 : ZaraMarbleArrangements 5 = 48 := sorry

end valid_marble_arrangements_eq_48_l61_61499


namespace lost_marble_count_l61_61294

def initial_marble_count : ℕ := 16
def remaining_marble_count : ℕ := 9

theorem lost_marble_count : initial_marble_count - remaining_marble_count = 7 := by
  -- Proof goes here
  sorry

end lost_marble_count_l61_61294


namespace right_triangle_area_l61_61289

variable (AB AC : ℝ) (angle_A : ℝ)

def is_right_triangle (AB AC : ℝ) (angle_A : ℝ) : Prop :=
  angle_A = 90

def area_of_triangle (AB AC : ℝ) : ℝ :=
  0.5 * AB * AC

theorem right_triangle_area :
  is_right_triangle AB AC angle_A →
  AB = 35 →
  AC = 15 →
  area_of_triangle AB AC = 262.5 :=
by
  intros
  simp [is_right_triangle, area_of_triangle]
  sorry

end right_triangle_area_l61_61289


namespace f_div_36_l61_61580

open Nat

def f (n : ℕ) : ℕ :=
  (2 * n + 7) * 3^n + 9

theorem f_div_36 (n : ℕ) : (f n) % 36 = 0 := 
  sorry

end f_div_36_l61_61580


namespace value_of_a7_l61_61432

theorem value_of_a7 (a : ℕ → ℤ) (h1 : a 1 = 0) (h2 : ∀ n, a (n + 2) - a n = 2) : a 7 = 6 :=
by {
  sorry -- Proof goes here
}

end value_of_a7_l61_61432


namespace circle_passing_through_points_l61_61629

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61629


namespace circle_through_points_l61_61640

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61640


namespace value_of_fraction_l61_61270

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l61_61270


namespace xy_square_diff_l61_61849

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61849


namespace question_1_question_2_question_3_l61_61986
-- Importing the Mathlib library for necessary functions

-- Definitions and assumptions based on the problem conditions
def z0 (m : ℝ) : ℂ := 1 - m * Complex.I
def z (x y : ℝ) : ℂ := x + y * Complex.I
def w (x' y' : ℝ) : ℂ := x' + y' * Complex.I

/-- The proof problem in Lean 4 to find necessary values and relationships -/
theorem question_1 (m : ℝ) (hm : m > 0) :
  (Complex.abs (z0 m) = 2 → m = Real.sqrt 3) ∧
  (∀ (x y : ℝ), ∃ (x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y) :=
by
  sorry

theorem question_2 (x y : ℝ) (hx : y = x + 1) :
  ∃ x' y', x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ 
  y' = (2 - Real.sqrt 3) * x' - 2 * Real.sqrt 3 + 2 :=
by
  sorry

theorem question_3 (x y : ℝ) :
  (∃ (k b : ℝ), y = k * x + b ∧ 
  (∀ (x y x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ y' = k * x' + b → 
  y = Real.sqrt 3 / 3 * x ∨ y = - Real.sqrt 3 * x)) :=
by
  sorry

end question_1_question_2_question_3_l61_61986


namespace binary_to_decimal_11011_is_27_l61_61780

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l61_61780


namespace divisible_by_12_l61_61088

theorem divisible_by_12 (n : ℤ) : 12 ∣ (n^4 - n^2) := sorry

end divisible_by_12_l61_61088


namespace female_employees_count_l61_61906

-- Define constants
def E : ℕ  -- Total number of employees
def M : ℕ := (2 / 5) * E  -- Total number of managers
def Male_E : ℕ  -- Total number of male employees
def Female_E : ℕ := E - Male_E  -- Total number of female employees
def Male_M : ℕ := (2 / 5) * Male_E  -- Total number of male managers
def Female_M : ℕ := 200  -- Total number of female managers

-- Given equation relating managers and employees
theorem female_employees_count : Female_E = 500 :=
by
  -- Required proof goes here
  sorry

end female_employees_count_l61_61906


namespace circle_passing_through_points_l61_61600

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61600


namespace cost_per_top_l61_61071
   
   theorem cost_per_top 
     (total_spent : ℕ) 
     (short_pairs : ℕ) 
     (short_cost_per_pair : ℕ) 
     (shoe_pairs : ℕ) 
     (shoe_cost_per_pair : ℕ) 
     (top_count : ℕ)
     (remaining_cost : ℕ)
     (total_short_cost : ℕ) 
     (total_shoe_cost : ℕ) 
     (total_short_shoe_cost : ℕ)
     (total_top_cost : ℕ) :
     total_spent = 75 →
     short_pairs = 5 →
     short_cost_per_pair = 7 →
     shoe_pairs = 2 →
     shoe_cost_per_pair = 10 →
     top_count = 4 →
     total_short_cost = short_pairs * short_cost_per_pair →
     total_shoe_cost = shoe_pairs * shoe_cost_per_pair →
     total_short_shoe_cost = total_short_cost + total_shoe_cost →
     total_top_cost = total_spent - total_short_shoe_cost →
     remaining_cost = total_top_cost / top_count →
     remaining_cost = 5 :=
   by
     intros
     sorry
   
end cost_per_top_l61_61071


namespace total_weight_of_10_moles_CaH2_is_420_96_l61_61917

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008
def molecular_weight_CaH2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_H
def moles_CaH2 : ℝ := 10
def total_weight_CaH2 : ℝ := molecular_weight_CaH2 * moles_CaH2

theorem total_weight_of_10_moles_CaH2_is_420_96 :
  total_weight_CaH2 = 420.96 :=
by
  sorry

end total_weight_of_10_moles_CaH2_is_420_96_l61_61917


namespace integer_values_of_x_l61_61247

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l61_61247


namespace other_acute_angle_of_right_triangle_l61_61114

theorem other_acute_angle_of_right_triangle (a : ℝ) (h₀ : 0 < a ∧ a < 90) (h₁ : a = 20) :
  ∃ b, b = 90 - a ∧ b = 70 := by
    sorry

end other_acute_angle_of_right_triangle_l61_61114


namespace hash_difference_l61_61113

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference :
  (hash 8 5) - (hash 5 8) = -12 :=
by
  sorry

end hash_difference_l61_61113


namespace probability_interval_l61_61724

-- Define the probability distribution and conditions
def P (xi : ℕ) (c : ℚ) : ℚ := c / (xi * (xi + 1))

-- Given conditions
variables (c : ℚ)
axiom condition : P 1 c + P 2 c + P 3 c + P 4 c = 1

-- Define the interval probability
def interval_prob (c : ℚ) : ℚ := P 1 c + P 2 c

-- Prove that the computed probability matches the expected value
theorem probability_interval : interval_prob (5 / 4) = 5 / 6 :=
by
  -- skip proof
  sorry

end probability_interval_l61_61724


namespace xy_square_diff_l61_61847

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61847


namespace last_digit_of_exponents_l61_61327

theorem last_digit_of_exponents : 
  (∃k, 2011 = 4 * k + 3 ∧ 
         (2^2011 % 10 = 8) ∧ 
         (3^2011 % 10 = 7)) → 
  ((2^2011 + 3^2011) % 10 = 5) := 
by 
  sorry

end last_digit_of_exponents_l61_61327


namespace largest_integer_modulo_l61_61799

theorem largest_integer_modulo (a : ℤ) : a < 93 ∧ a % 7 = 4 ∧ (∀ b : ℤ, b < 93 ∧ b % 7 = 4 → b ≤ a) ↔ a = 88 :=
by
    sorry

end largest_integer_modulo_l61_61799


namespace max_at_pi_six_l61_61717

theorem max_at_pi_six : ∃ (x : ℝ), (0 ≤ x ∧ x ≤ π / 2) ∧ (∀ y, (0 ≤ y ∧ y ≤ π / 2) → (x + 2 * Real.cos x) ≥ (y + 2 * Real.cos y)) ∧ x = π / 6 := sorry

end max_at_pi_six_l61_61717


namespace least_perimeter_of_triangle_l61_61869

theorem least_perimeter_of_triangle (cosA cosB cosC : ℝ)
  (h₁ : cosA = 13 / 16)
  (h₂ : cosB = 4 / 5)
  (h₃ : cosC = -3 / 5) :
  ∃ a b c : ℕ, a + b + c = 28 ∧ 
  a^2 + b^2 - c^2 = 2 * a * b * cosC ∧ 
  b^2 + c^2 - a^2 = 2 * b * c * cosA ∧ 
  c^2 + a^2 - b^2 = 2 * c * a * cosB :=
sorry

end least_perimeter_of_triangle_l61_61869


namespace abs_diff_of_roots_eq_one_l61_61956

theorem abs_diff_of_roots_eq_one {p q : ℝ} (h₁ : p + q = 7) (h₂ : p * q = 12) : |p - q| = 1 := 
by 
  sorry

end abs_diff_of_roots_eq_one_l61_61956


namespace sum_b_a1_a2_a3_a4_eq_60_l61_61561

def a_n (n : ℕ) : ℕ := n + 2
def b_n (n : ℕ) : ℕ := 2^(n-1)

theorem sum_b_a1_a2_a3_a4_eq_60 :
  b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) = 60 :=
by
  sorry

end sum_b_a1_a2_a3_a4_eq_60_l61_61561


namespace min_stool_height_l61_61937

/-
Alice needs to reach a ceiling fan switch located 15 centimeters below a 3-meter-tall ceiling.
Alice is 160 centimeters tall and can reach 50 centimeters above her head. She uses a stack of books
12 centimeters tall to assist her reach. We aim to show that the minimum height of the stool she needs is 63 centimeters.
-/

def ceiling_height_cm : ℕ := 300
def alice_height_cm : ℕ := 160
def reach_above_head_cm : ℕ := 50
def books_height_cm : ℕ := 12
def switch_below_ceiling_cm : ℕ := 15

def total_reach_with_books := alice_height_cm + reach_above_head_cm + books_height_cm
def switch_height_from_floor := ceiling_height_cm - switch_below_ceiling_cm

theorem min_stool_height : total_reach_with_books + 63 = switch_height_from_floor := by
  unfold total_reach_with_books switch_height_from_floor
  sorry

end min_stool_height_l61_61937


namespace election_invalid_votes_percentage_l61_61862

theorem election_invalid_votes_percentage (x : ℝ) :
  (∀ (total_votes valid_votes_in_favor_of_A : ℝ),
    total_votes = 560000 →
    valid_votes_in_favor_of_A = 357000 →
    0.75 * ((1 - x / 100) * total_votes) = valid_votes_in_favor_of_A) →
  x = 15 :=
by
  intro h
  specialize h 560000 357000 (rfl : 560000 = 560000) (rfl : 357000 = 357000)
  sorry

end election_invalid_votes_percentage_l61_61862


namespace smallest_n_in_T_and_largest_N_not_in_T_l61_61438

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3 * x + 4) / (x + 3)}

theorem smallest_n_in_T_and_largest_N_not_in_T :
  (∀ n, n = 4 / 3 → n ∈ T) ∧ (∀ N, N = 3 → N ∉ T) :=
by
  sorry

end smallest_n_in_T_and_largest_N_not_in_T_l61_61438


namespace inequality_system_solution_l61_61022

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l61_61022


namespace trajectory_of_center_of_P_l61_61811

-- Define circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the conditions for the moving circle P
def externally_tangent (x y r : ℝ) : Prop := (x + 1)^2 + y^2 = (1 + r)^2
def internally_tangent (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = (5 - r)^2

-- The statement we need to prove
theorem trajectory_of_center_of_P : ∃ (x y : ℝ), 
  (externally_tangent x y r) ∧ (internally_tangent x y r) →
  (x^2 / 9 + y^2 / 8 = 1) :=
by
  -- Proof will go here
  sorry

end trajectory_of_center_of_P_l61_61811


namespace find_x_plus_y_l61_61234

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3000)
  (h2 : x + 3000 * Real.sin y = 2999) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
  sorry

end find_x_plus_y_l61_61234


namespace xy_square_diff_l61_61850

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61850


namespace range_of_a_l61_61283

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end range_of_a_l61_61283


namespace total_admission_methods_l61_61387

noncomputable def number_of_admission_methods : ℕ :=
  (Combinatorics.choose 4 1) * (Combinatorics.perm 5 3)

theorem total_admission_methods : number_of_admission_methods = 240 := by
  sorry

end total_admission_methods_l61_61387


namespace time_between_ticks_at_6_l61_61527

def intervals_12 := 11
def ticks_12 := 12
def seconds_12 := 77
def intervals_6 := 5
def ticks_6 := 6

theorem time_between_ticks_at_6 :
  let interval_time := seconds_12 / intervals_12
  let total_time_6 := intervals_6 * interval_time
  total_time_6 = 35 := sorry

end time_between_ticks_at_6_l61_61527


namespace constructible_angles_l61_61789

def is_constructible (θ : ℝ) : Prop :=
  -- Define that θ is constructible if it can be constructed using compass and straightedge.
  sorry

theorem constructible_angles (α : ℝ) (β : ℝ) (k n : ℤ) (hβ : is_constructible β) :
  is_constructible (k * α / 2^n + β) :=
sorry

end constructible_angles_l61_61789


namespace M_values_l61_61408

theorem M_values (m n p M : ℝ) (h1 : M = m / (n + p)) (h2 : M = n / (p + m)) (h3 : M = p / (m + n)) :
  M = 1 / 2 ∨ M = -1 :=
by
  sorry

end M_values_l61_61408


namespace kate_hair_length_l61_61298

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l61_61298


namespace octagon_area_l61_61733

noncomputable def area_of_octagon_concentric_squares : ℚ :=
  let m := 1
  let n := 8
  (m + n)

theorem octagon_area (O : ℝ × ℝ) (side_small side_large : ℚ) (AB : ℚ) 
  (h1 : side_small = 2) (h2 : side_large = 3) (h3 : AB = 1/4) : 
  area_of_octagon_concentric_squares = 9 := 
  by
  have h_area : 1/8 = 1/8 := rfl
  sorry

end octagon_area_l61_61733


namespace circle_passing_through_points_l61_61599

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61599


namespace distinct_factorizations_72_l61_61380

-- Define the function D that calculates the number of distinct factorizations.
noncomputable def D (n : Nat) : Nat := 
  -- Placeholder function to represent D, the actual implementation is skipped.
  sorry

-- Theorem stating the number of distinct factorizations of 72 considering the order of factors.
theorem distinct_factorizations_72 : D 72 = 119 :=
  sorry

end distinct_factorizations_72_l61_61380


namespace average_speed_l61_61523

def total_distance : ℝ := 200
def total_time : ℝ := 40

theorem average_speed (d t : ℝ) (h₁: d = total_distance) (h₂: t = total_time) : d / t = 5 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end average_speed_l61_61523


namespace fraction_addition_l61_61793

theorem fraction_addition :
  (2 / 5 : ℚ) + (3 / 8) = 31 / 40 :=
sorry

end fraction_addition_l61_61793


namespace company_bought_oil_l61_61466

-- Define the conditions
def tank_capacity : ℕ := 32
def oil_in_tank : ℕ := 24

-- Formulate the proof problem
theorem company_bought_oil : oil_in_tank = 24 := by
  sorry

end company_bought_oil_l61_61466


namespace students_drawn_from_grade10_l61_61195

-- Define the initial conditions
def total_students_grade12 : ℕ := 750
def total_students_grade11 : ℕ := 850
def total_students_grade10 : ℕ := 900
def sample_size : ℕ := 50

-- Prove the number of students drawn from grade 10 is 18
theorem students_drawn_from_grade10 : 
  total_students_grade12 = 750 ∧
  total_students_grade11 = 850 ∧
  total_students_grade10 = 900 ∧
  sample_size = 50 →
  (sample_size * total_students_grade10 / 
  (total_students_grade12 + total_students_grade11 + total_students_grade10) = 18) :=
by
  sorry

end students_drawn_from_grade10_l61_61195


namespace problem4_l61_61144

theorem problem4 (a : ℝ) : (a-1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := 
by sorry

end problem4_l61_61144


namespace proof_problem_l61_61559

noncomputable def p : Prop := ∃ x : ℝ, Real.sin x > 1
noncomputable def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

theorem proof_problem : ¬ (p ∨ q) :=
by sorry

end proof_problem_l61_61559


namespace max_tan_B_l61_61809

theorem max_tan_B (A B : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) (h : Real.tan (A + B) = 2 * Real.tan A) :
  ∃ B_max, B_max = Real.tan B ∧ B_max ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l61_61809


namespace Force_Inversely_Proportional_l61_61470

theorem Force_Inversely_Proportional
  (L₁ F₁ L₂ F₂ : ℝ)
  (h₁ : L₁ = 12)
  (h₂ : F₁ = 480)
  (h₃ : L₂ = 18)
  (h_inv : F₁ * L₁ = F₂ * L₂) :
  F₂ = 320 :=
by
  sorry

end Force_Inversely_Proportional_l61_61470


namespace min_value_of_x_plus_y_l61_61974

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l61_61974


namespace bus_ride_cost_l61_61048

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.85) (h2 : T + B = 9.65) : B = 1.40 :=
sorry

end bus_ride_cost_l61_61048


namespace joker_then_spade_probability_correct_l61_61200

-- Defining the conditions of the deck
def deck_size : ℕ := 60
def joker_count : ℕ := 4
def suit_count : ℕ := 4
def cards_per_suit : ℕ := 15

-- The probability of drawing a Joker first and then a spade
def prob_joker_then_spade : ℚ :=
  (joker_count * (cards_per_suit - 1) + (deck_size - joker_count) * cards_per_suit) /
  (deck_size * (deck_size - 1))

-- The expected probability according to the solution
def expected_prob : ℚ := 224 / 885

theorem joker_then_spade_probability_correct :
  prob_joker_then_spade = expected_prob :=
by
  -- Skipping the actual proof steps
  sorry

end joker_then_spade_probability_correct_l61_61200


namespace first_number_in_proportion_l61_61116

variable (x y : ℝ)

theorem first_number_in_proportion
  (h1 : x = 0.9)
  (h2 : y / x = 5 / 6) : 
  y = 0.75 := 
  by 
    sorry

end first_number_in_proportion_l61_61116


namespace math_problem_l61_61530

-- Define the main variables a and b
def a : ℕ := 312
def b : ℕ := 288

-- State the main theorem to be proved
theorem math_problem : (a^2 - b^2) / 24 + 50 = 650 := 
by 
  sorry

end math_problem_l61_61530


namespace pump_leak_drain_time_l61_61365

theorem pump_leak_drain_time {P L : ℝ} (hP : P = 0.25) (hPL : P - L = 0.05) : (1 / L) = 5 :=
by sorry

end pump_leak_drain_time_l61_61365


namespace sum_y_coeffs_eq_38_l61_61573

noncomputable def sum_of_y_coefficients (p q : Polynomial ℤ) : ℤ :=
  (p * q).support.sum (λ n, if n ≥ 1 then (p * q).coeff n else 0)

theorem sum_y_coeffs_eq_38 :
  let p := Polynomial.C 2 + Polynomial.C 3 * Polynomial.Y + Polynomial.X in
  let q := Polynomial.C 3 + Polynomial.C 2 * Polynomial.Y + Polynomial.X in
  sum_of_y_coefficients p q = 38 :=
by
  -- Setup polynomial p and q
  let p := Polynomial.C 2 + Polynomial.C 3 * Polynomial.Y + Polynomial.X
  let q := Polynomial.C 3 + Polynomial.C 2 * Polynomial.Y + Polynomial.X
  -- Sorry to skip the proof for now
  sorry

end sum_y_coeffs_eq_38_l61_61573


namespace taxi_ride_cost_l61_61074

-- Definitions
def initial_fee : Real := 2
def distance_in_miles : Real := 4
def cost_per_mile : Real := 2.5

-- Theorem statement
theorem taxi_ride_cost (initial_fee : Real) (distance_in_miles : Real) (cost_per_mile : Real) : 
  initial_fee + distance_in_miles * cost_per_mile = 12 := 
by
  sorry

end taxi_ride_cost_l61_61074


namespace fraction_arithmetic_proof_l61_61182

theorem fraction_arithmetic_proof :
  (7 / 6) + (5 / 4) - (3 / 2) = 11 / 12 :=
by sorry

end fraction_arithmetic_proof_l61_61182


namespace triangle_area_l61_61518

noncomputable def line_eq := λ x : ℝ, 9 - 3 * x

theorem triangle_area :
    let x_intercept := (3 : ℝ)
    let y_intercept := (9 : ℝ)
    let triangle_base := x_intercept
    let triangle_height := y_intercept
    let area := (1 / 2) * triangle_base * triangle_height
    area = 13.5 :=
by
    sorry

end triangle_area_l61_61518


namespace correct_fraction_order_l61_61920

noncomputable def fraction_ordering : Prop := 
  (16 / 12 < 18 / 13) ∧ (18 / 13 < 21 / 14) ∧ (21 / 14 < 20 / 15)

theorem correct_fraction_order : fraction_ordering := 
by {
  repeat { sorry }
}

end correct_fraction_order_l61_61920


namespace problem1_problem2_l61_61458

theorem problem1 (x : ℝ) : (5 - 2 * x) ^ 2 - 16 = 0 ↔ x = 1 / 2 ∨ x = 9 / 2 := 
by 
  sorry

theorem problem2 (x : ℝ) : 2 * (x - 3) = x^2 - 9 ↔ x = 3 ∨ x = -1 := 
by 
  sorry

end problem1_problem2_l61_61458


namespace circle_passing_three_points_l61_61706

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61706


namespace symmetric_points_on_parabola_l61_61407

theorem symmetric_points_on_parabola {a b m n : ℝ}
  (hA : m = a^2 - 2*a - 2)
  (hB : m = b^2 - 2*b - 2)
  (hP : n = (a + b)^2 - 2*(a + b) - 2)
  (h_symmetry : (a + b) / 2 = 1) :
  n = -2 :=
by {
  -- Proof omitted
  sorry
}

end symmetric_points_on_parabola_l61_61407


namespace arithmetic_sequence_and_sum_conditions_l61_61985

open Nat

noncomputable theory

def sequence_a (n : ℕ) : ℕ := 2n - 1
def sequence_b (n : ℕ) : ℕ := (3 * n - 2) / (2 * n - 1)

theorem arithmetic_sequence_and_sum_conditions (a d : ℕ) (h_d_pos : d > 0)
  (h_2_3 : (a + d) * (a + 2 * d) = 15) (h_sum_4 : 4 * a + 6 * d = 16) 
  (b_1 : ℕ) (hb_1_eq_a1 : b_1 = a) 
  (hb_recurrence : ∀ n, b n + 1 - b n = 1 / (a_n * (a_n + 1)) →
  (∀ n, a n = 2n - 1) ∧ ∀ n, b_n = (3n - 2) / (2n - 1)) :=
begin
  sorry,
end

end arithmetic_sequence_and_sum_conditions_l61_61985


namespace point_A_lies_on_plane_l61_61977

-- Define the plane equation
def plane (x y z : ℝ) : Prop := 2 * x - y + 2 * z = 7

-- Define the specific point
def point_A : Prop := plane 2 3 3

-- The theorem stating that point A lies on the plane
theorem point_A_lies_on_plane : point_A :=
by
  -- Proof skipped
  sorry

end point_A_lies_on_plane_l61_61977


namespace intersection_points_of_curve_with_axes_l61_61473

theorem intersection_points_of_curve_with_axes :
  (∃ t : ℝ, (-2 + 5 * t = 0) ∧ (1 - 2 * t = 1/5)) ∧
  (∃ t : ℝ, (1 - 2 * t = 0) ∧ (-2 + 5 * t = 1/2)) :=
by {
  -- Proving the intersection points with the coordinate axes
  sorry
}

end intersection_points_of_curve_with_axes_l61_61473


namespace job_completion_days_l61_61835

variable (m r h d : ℕ)

theorem job_completion_days :
  (m + 2 * r) * (h + 1) * (m * h * d / ((m + 2 * r) * (h + 1))) = m * h * d :=
by
  sorry

end job_completion_days_l61_61835


namespace identity_x_squared_minus_y_squared_l61_61843

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61843


namespace no_solution_exists_l61_61392

open Int

theorem no_solution_exists (x y z : ℕ) (hx : x > 0) (hy : y > 0)
  (hz : z = Nat.gcd x y) : x + y^2 + z^3 ≠ x * y * z := 
sorry

end no_solution_exists_l61_61392


namespace gcd_factorials_l61_61966

open Nat

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials (h : ∀ n, 0 < n → factorial n = n * factorial (n - 1)) :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 :=
sorry

end gcd_factorials_l61_61966


namespace factorization_of_polynomial_l61_61536

theorem factorization_of_polynomial (x : ℂ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^2 - 3) * (x^4 + 3 * x^2 + 9) :=
by sorry

end factorization_of_polynomial_l61_61536


namespace simplify_sqrt_l61_61890

-- Define the domain and main trigonometric properties
open Real

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  sqrt (1 - 2 * sin x * cos x)

-- Define the main theorem with given conditions
theorem simplify_sqrt {x : ℝ} (h1 : (5 / 4) * π < x) (h2 : x < (3 / 2) * π) (h3 : cos x > sin x) :
  simplify_expression x = cos x - sin x :=
  sorry

end simplify_sqrt_l61_61890


namespace integer_solutions_l61_61953

theorem integer_solutions (n : ℤ) : (n^2 + 1) ∣ (n^5 + 3) ↔ n = -3 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2 := 
sorry

end integer_solutions_l61_61953


namespace positive_integer_solution_l61_61932

theorem positive_integer_solution (x : Int) (h_pos : x > 0) (h_cond : x + 1000 > 1000 * x) : x = 2 :=
sorry

end positive_integer_solution_l61_61932


namespace smallest_k_for_g_l61_61396

theorem smallest_k_for_g (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x + k = -3) ↔ k ≤ -3/4 := sorry

end smallest_k_for_g_l61_61396


namespace circle_passing_through_points_l61_61632

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61632


namespace price_of_each_orange_l61_61377

theorem price_of_each_orange 
  (x : ℕ)
  (a o : ℕ)
  (h1 : a + o = 20)
  (h2 : 40 * a + x * o = 1120)
  (h3 : (a + o - 10) * 52 = 1120 - 10 * x) :
  x = 60 :=
sorry

end price_of_each_orange_l61_61377


namespace total_profit_correct_l61_61056

-- We define the conditions
variables (a m : ℝ)

-- The item's cost per piece
def cost_per_piece : ℝ := a
-- The markup percentage
def markup_percentage : ℝ := 0.20
-- The discount percentage
def discount_percentage : ℝ := 0.10
-- The number of pieces sold
def pieces_sold : ℝ := m

-- Definitions derived from conditions
def selling_price_markup : ℝ := cost_per_piece a * (1 + markup_percentage)
def selling_price_discount : ℝ := selling_price_markup a * (1 - discount_percentage)
def profit_per_piece : ℝ := selling_price_discount a - cost_per_piece a
def total_profit : ℝ := profit_per_piece a * pieces_sold m

theorem total_profit_correct (a m : ℝ) : total_profit a m = 0.08 * a * m :=
by sorry

end total_profit_correct_l61_61056


namespace inequality_proof_l61_61100

-- Defining the conditions
variable (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (cond : 1 / a + 1 / b = 1)

-- Defining the theorem to be proved
theorem inequality_proof (n : ℕ) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by
  sorry

end inequality_proof_l61_61100


namespace circle_through_points_l61_61642

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61642


namespace remainder_43_pow_43_plus_43_mod_44_l61_61924

theorem remainder_43_pow_43_plus_43_mod_44 : (43^43 + 43) % 44 = 42 :=
by 
    sorry

end remainder_43_pow_43_plus_43_mod_44_l61_61924


namespace value_of_first_equation_l61_61989

variables (x y z w : ℝ)

theorem value_of_first_equation (h1 : xw + yz = 8) (h2 : (2 * x + y) * (2 * z + w) = 20) : xz + yw = 1 := by
  sorry

end value_of_first_equation_l61_61989


namespace face_value_of_share_l61_61755

theorem face_value_of_share 
  (F : ℝ)
  (dividend_rate : ℝ := 0.09)
  (desired_return_rate : ℝ := 0.12)
  (market_value : ℝ := 15) 
  (h_eq : (dividend_rate * F) / market_value = desired_return_rate) : 
  F = 20 := 
by 
  sorry

end face_value_of_share_l61_61755


namespace seed_mixture_x_percentage_l61_61351

theorem seed_mixture_x_percentage (x y : ℝ) (h : 0.40 * x + 0.25 * y = 0.30 * (x + y)) : 
  (x / (x + y)) * 100 = 33.33 := sorry

end seed_mixture_x_percentage_l61_61351


namespace number_of_two_bedroom_units_l61_61192

-- Definitions based on the conditions
def is_solution (x y : ℕ) : Prop :=
  (x + y = 12) ∧ (360 * x + 450 * y = 4950)

theorem number_of_two_bedroom_units : ∃ y : ℕ, is_solution (12 - y) y ∧ y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l61_61192


namespace leo_score_l61_61859

-- Definitions for the conditions
def caroline_score : ℕ := 13
def anthony_score : ℕ := 19
def winning_score : ℕ := 21

-- Lean statement for the proof problem
theorem leo_score : ∃ (leo_score : ℕ), leo_score = winning_score := by
  have h_caroline := caroline_score
  have h_anthony := anthony_score
  have h_winning := winning_score
  use 21
  sorry

end leo_score_l61_61859


namespace linear_function_points_relation_l61_61831

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ), 
  (y1 = -3 * 2 + 1) ∧ (y2 = -3 * 3 + 1) → y1 > y2 :=
by
  intro y1 y2
  intro h
  cases h
  sorry

end linear_function_points_relation_l61_61831


namespace find_A_and_area_l61_61980

open Real

variable (A B C a b c : ℝ)
variable (h1 : 2 * sin A * cos B = 2 * sin C - sin B)
variable (h2 : a = 4 * sqrt 3)
variable (h3 : b + c = 8)
variable (h4 : a^2 = b^2 + c^2 - 2*b*c* cos A)

theorem find_A_and_area :
  A = π / 3 ∧ (1/2 * b * c * sin A = 4 * sqrt 3 / 3) :=
by
  sorry

end find_A_and_area_l61_61980


namespace solve_inequality_system_l61_61026

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61026


namespace liam_balloons_remainder_l61_61877

def balloons : Nat := 24 + 45 + 78 + 96
def friends : Nat := 10
def remainder := balloons % friends

theorem liam_balloons_remainder : remainder = 3 := by
  sorry

end liam_balloons_remainder_l61_61877


namespace tara_marbles_modulo_l61_61149

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem tara_marbles_modulo : 
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  N % 1000 = 564 :=
by
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  have : N % 1000 = 564 := sorry
  exact this

end tara_marbles_modulo_l61_61149


namespace tree_F_height_l61_61043

variable (A B C D E F : ℝ)

def height_conditions : Prop :=
  A = 150 ∧ -- Tree A's height is 150 feet
  B = (2 / 3) * A ∧ -- Tree B's height is 2/3 of Tree A's height
  C = (1 / 2) * B ∧ -- Tree C's height is 1/2 of Tree B's height
  D = C + 25 ∧ -- Tree D's height is 25 feet more than Tree C's height
  E = 0.40 * A ∧ -- Tree E's height is 40% of Tree A's height
  F = (B + D) / 2 -- Tree F's height is the average of Tree B's height and Tree D's height

theorem tree_F_height : height_conditions A B C D E F → F = 87.5 :=
by
  intros
  sorry

end tree_F_height_l61_61043


namespace driver_days_off_l61_61176

theorem driver_days_off 
  (drivers : ℕ) 
  (cars : ℕ) 
  (maintenance_rate : ℚ) 
  (days_in_month : ℕ)
  (needed_driver_days : ℕ)
  (x : ℚ) :
  drivers = 54 →
  cars = 60 →
  maintenance_rate = 0.25 →
  days_in_month = 30 →
  needed_driver_days = 45 * days_in_month →
  54 * (30 - x) = needed_driver_days →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end driver_days_off_l61_61176


namespace Kates_hair_length_l61_61301

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l61_61301


namespace money_left_after_shopping_l61_61124

-- Conditions
def cost_mustard_oil : ℤ := 2 * 13
def cost_pasta : ℤ := 3 * 4
def cost_sauce : ℤ := 1 * 5
def total_cost : ℤ := cost_mustard_oil + cost_pasta + cost_sauce
def total_money : ℤ := 50

-- Theorem to prove
theorem money_left_after_shopping : total_money - total_cost = 7 := by
  sorry

end money_left_after_shopping_l61_61124


namespace circle_passing_through_points_eqn_l61_61653

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61653


namespace fraction_comparison_l61_61814

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) :=
by sorry

end fraction_comparison_l61_61814


namespace circle_equation_and_range_of_a_l61_61101

theorem circle_equation_and_range_of_a :
  (∃ m : ℤ, (x - m)^2 + y^2 = 25 ∧ (abs (4 * m - 29)) = 25) ∧
  (∀ a : ℝ, (a > 0 → (4 * (5 * a - 1)^2 - 4 * (a^2 + 1) > 0 → a > 5 / 12 ∨ a < 0))) :=
by
  sorry

end circle_equation_and_range_of_a_l61_61101


namespace algebraic_expression_value_l61_61834

theorem algebraic_expression_value (m: ℝ) (h: m^2 + m - 1 = 0) : 2023 - m^2 - m = 2022 := 
by 
  sorry

end algebraic_expression_value_l61_61834


namespace solve_for_y_l61_61457

theorem solve_for_y :
  ∃ y : ℚ, 2 * y + 3 * y = 200 - (4 * y + (10 * y / 2)) ∧ y = 100 / 7 :=
by {
  -- Assertion only, proof is not required as per instructions.
  sorry
}

end solve_for_y_l61_61457


namespace find_polynomial_h_l61_61592

theorem find_polynomial_h (f h : ℝ → ℝ) (hf : ∀ x, f x = x^2) (hh : ∀ x, f (h x) = 9 * x^2 + 6 * x + 1) : 
  (∀ x, h x = 3 * x + 1) ∨ (∀ x, h x = -3 * x - 1) :=
by
  sorry

end find_polynomial_h_l61_61592


namespace circle_through_points_l61_61636

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61636


namespace election_invalid_votes_percentage_l61_61861

theorem election_invalid_votes_percentage (x : ℝ) :
  (∀ (total_votes valid_votes_in_favor_of_A : ℝ),
    total_votes = 560000 →
    valid_votes_in_favor_of_A = 357000 →
    0.75 * ((1 - x / 100) * total_votes) = valid_votes_in_favor_of_A) →
  x = 15 :=
by
  intro h
  specialize h 560000 357000 (rfl : 560000 = 560000) (rfl : 357000 = 357000)
  sorry

end election_invalid_votes_percentage_l61_61861


namespace circle_passing_through_points_l61_61634

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61634


namespace relationship_between_abc_l61_61224

theorem relationship_between_abc 
  (a b c : ℝ) 
  (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) 
  (ha : Real.exp a = 9 * a * Real.log 11)
  (hb : Real.exp b = 10 * b * Real.log 10)
  (hc : Real.exp c = 11 * c * Real.log 9) : 
  a < b ∧ b < c :=
sorry

end relationship_between_abc_l61_61224


namespace rabbit_travel_time_l61_61759

theorem rabbit_travel_time :
  let distance := 2
  let speed := 5
  let hours_to_minutes := 60
  (distance / speed) * hours_to_minutes = 24 := by
sorry

end rabbit_travel_time_l61_61759


namespace num_possible_integer_values_x_l61_61249

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l61_61249


namespace total_bricks_used_l61_61313

def numCoursesPerWall : Nat := 6
def bricksPerCourse : Nat := 10
def numWalls : Nat := 4
def unfinishedCoursesLastWall : Nat := 2

theorem total_bricks_used : 
  let totalCourses := numWalls * numCoursesPerWall
  let bricksRequired := totalCourses * bricksPerCourse
  let bricksMissing := unfinishedCoursesLastWall * bricksPerCourse
  let bricksUsed := bricksRequired - bricksMissing
  bricksUsed = 220 := 
by
  sorry

end total_bricks_used_l61_61313


namespace polynomial_exists_l61_61213

open Polynomial

noncomputable def exists_polynomial_2013 : Prop :=
  ∃ (f : Polynomial ℤ), (∀ (n : ℕ), n ≤ f.natDegree → (coeff f n = 1 ∨ coeff f n = -1))
                         ∧ ((X - 1) ^ 2013 ∣ f)

theorem polynomial_exists : exists_polynomial_2013 :=
  sorry

end polynomial_exists_l61_61213


namespace moles_of_CaCO3_formed_l61_61093

theorem moles_of_CaCO3_formed (m n : ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : ∀ m n : ℕ, (m = n) → (m = 3) → (n = 3) → moles_of_CaCO3 = m) : 
  moles_of_CaCO3 = 3 := by
  sorry

end moles_of_CaCO3_formed_l61_61093


namespace time_train_passes_jogger_l61_61360

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

noncomputable def initial_lead_m : ℝ := 150
noncomputable def train_length_m : ℝ := 100

noncomputable def total_distance_to_cover_m : ℝ := initial_lead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_to_cover_m / relative_speed_mps

theorem time_train_passes_jogger : time_to_pass_jogger_s = 25 := by
  sorry

end time_train_passes_jogger_l61_61360


namespace complete_the_square_example_l61_61212

theorem complete_the_square_example : ∀ x m n : ℝ, (x^2 - 12 * x + 33 = 0) → 
  (x + m)^2 = n → m = -6 ∧ n = 3 :=
by
  sorry

end complete_the_square_example_l61_61212


namespace f_of_x_l61_61833

variable (f : ℝ → ℝ)

theorem f_of_x (x : ℝ) (h : f (x - 1 / x) = x^2 + 1 / x^2) : f x = x^2 + 2 :=
sorry

end f_of_x_l61_61833


namespace find_c_squared_ab_l61_61328

theorem find_c_squared_ab (a b c : ℝ) (h1 : a^2 * (b + c) = 2008) (h2 : b^2 * (a + c) = 2008) (h3 : a ≠ b) : 
  c^2 * (a + b) = 2008 :=
sorry

end find_c_squared_ab_l61_61328


namespace equation_of_circle_through_three_points_l61_61679

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61679


namespace max_f_l61_61776

noncomputable def f (x : ℝ) : ℝ :=
  1 / (|x + 3| + |x + 1| + |x - 2| + |x - 5|)

theorem max_f : ∃ x : ℝ, f x = 1 / 11 :=
by
  sorry

end max_f_l61_61776


namespace circle_equation_l61_61648

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61648


namespace inequality_system_solution_l61_61017

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l61_61017


namespace find_x_l61_61281

def operation (x y : ℕ) : ℕ := 2 * x * y

theorem find_x : 
  (operation 4 5 = 40) ∧ (operation x 40 = 480) → x = 6 :=
by
  sorry

end find_x_l61_61281


namespace Josh_lost_marbles_l61_61873

theorem Josh_lost_marbles :
  let original_marbles := 9.5
  let current_marbles := 4.25
  original_marbles - current_marbles = 5.25 :=
by
  sorry

end Josh_lost_marbles_l61_61873


namespace common_root_value_l61_61221

theorem common_root_value (p : ℝ) (hp : p > 0) : 
  (∃ x : ℝ, 3 * x ^ 2 - 4 * p * x + 9 = 0 ∧ x ^ 2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by {
  sorry
}

end common_root_value_l61_61221


namespace total_problems_l61_61742

theorem total_problems (math_pages reading_pages problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end total_problems_l61_61742


namespace sum_polynomial_coefficients_l61_61418

theorem sum_polynomial_coefficients :
  let a := 1
  let a_sum := -2
  (2009 * a + a_sum) = 2007 :=
by
  sorry

end sum_polynomial_coefficients_l61_61418


namespace missing_fraction_of_coins_l61_61446

-- Defining the initial conditions
def total_coins (x : ℕ) := x
def lost_coins (x : ℕ) := (1 / 2) * x
def found_coins (x : ℕ) := (3 / 8) * x

-- Theorem statement
theorem missing_fraction_of_coins (x : ℕ) : 
  (total_coins x - lost_coins x + found_coins x) = (7 / 8) * x :=
by
  sorry  -- proof is omitted as per the instructions

end missing_fraction_of_coins_l61_61446


namespace negation_of_proposition_l61_61158

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l61_61158


namespace ways_to_select_books_l61_61481

theorem ways_to_select_books (nChinese nMath nEnglish : ℕ) (h1 : nChinese = 9) (h2 : nMath = 7) (h3 : nEnglish = 5) :
  (nChinese * nMath + nChinese * nEnglish + nMath * nEnglish) = 143 :=
by
  sorry

end ways_to_select_books_l61_61481


namespace trajectory_point_M_l61_61556

theorem trajectory_point_M (x y : ℝ) : 
  (∃ (m n : ℝ), x^2 + y^2 = 9 ∧ (m = x) ∧ (n = 3 * y)) → 
  (x^2 / 9 + y^2 = 1) :=
by
  sorry

end trajectory_point_M_l61_61556


namespace equation_of_circle_ABC_l61_61618

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61618


namespace target_heart_rate_of_30_year_old_l61_61376

variable (age : ℕ) (T M : ℕ)

def maximum_heart_rate (age : ℕ) : ℕ :=
  210 - age

def target_heart_rate (M : ℕ) : ℕ :=
  (75 * M) / 100

theorem target_heart_rate_of_30_year_old :
  maximum_heart_rate 30 = 180 →
  target_heart_rate (maximum_heart_rate 30) = 135 :=
by
  intros h1
  sorry

end target_heart_rate_of_30_year_old_l61_61376


namespace simplest_square_root_l61_61498

theorem simplest_square_root (a b c d : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 3) (h3 : c = (Real.sqrt 2) / 2) (h4 : d = Real.sqrt 10) :
  d = Real.sqrt 10 ∧ (a ≠ Real.sqrt 10) ∧ (b ≠ Real.sqrt 10) ∧ (c ≠ Real.sqrt 10) := 
by 
  sorry

end simplest_square_root_l61_61498


namespace number_pairs_sum_diff_prod_quotient_l61_61491

theorem number_pairs_sum_diff_prod_quotient (x y : ℤ) (h : x ≥ y) :
  (x + y) + (x - y) + x * y + x / y = 800 ∨ (x + y) + (x - y) + x * y + x / y = 400 :=
sorry

-- Correct answers for A = 800
example : (38 + 19) + (38 - 19) + 38 * 19 + 38 / 19 = 800 := by norm_num
example : (-42 + -21) + (-42 - -21) + (-42 * -21) + (-42 / -21) = 800 := by norm_num
example : (72 + 9) + (72 - 9) + 72 * 9 + 72 / 9 = 800 := by norm_num
example : (-88 + -11) + (-88 - -11) + -(88 * -11) + (-88 / -11) = 800 := by norm_num
example : (128 + 4) + (128 - 4) + 128 * 4 + 128 / 4 = 800 := by norm_num
example : (-192 + -6) + (-192 - -6) + -192 * -6 + ( -192 / -6 ) = 800 := by norm_num
example : (150 + 3) + (150 - 3) + 150 * 3 + 150 / 3 = 800 := by norm_num
example : (-250 + -5) + (-250 - -5) + (-250 * -5) + (-250 / -5) = 800 := by norm_num
example : (200 + 1) + (200 - 1) + 200 * 1 + 200 / 1 = 800 := by norm_num
example : (-600 + -3) + (-600 - -3) + -600 * -3 + -600 / -3 = 800 := by norm_num

-- Correct answers for A = 400
example : (19 + 19) + (19 - 19) + 19 * 19 + 19 / 19 = 400 := by norm_num
example : (-21 + -21) + (-21 - -21) + (-21 * -21) + (-21 / -21) = 400 := by norm_num
example : (36 + 9) + (36 - 9) + 36 * 9 + 36 / 9 = 400 := by norm_num
example : (-44 + -11) + (-44 - -11) + (-44 * -11) + (-44 / -11) = 400 := by norm_num
example : (64 + 4) + (64 - 4) + 64 * 4 + 64 / 4 = 400 := by norm_num
example : (-96 + -6) + (-96 - -6) + (-96 * -6) + (-96 / -6) = 400 := by norm_num
example : (75 + 3) + (75 - 3) + 75 * 3 + 75 / 3 = 400 := by norm_num
example : (-125 + -5) + (-125 - -5) + (-125 * -5) + (-125 / -5) = 400 := by norm_num
example : (100 + 1) + (100 - 1) + 100 * 1 + 100 / 1 = 400 := by norm_num
example : (-300 + -3) + (-300 - -3) + (-300 * -3) + (-300 / -3) = 400 := by norm_num

end number_pairs_sum_diff_prod_quotient_l61_61491


namespace total_people_museum_l61_61172

def bus1 := 12
def bus2 := 2 * bus1
def bus3 := bus2 - 6
def bus4 := bus1 + 9
def total := bus1 + bus2 + bus3 + bus4

theorem total_people_museum : total = 75 := by
  sorry

end total_people_museum_l61_61172


namespace determine_a_value_l61_61089

theorem determine_a_value :
  ∀ (a b c d : ℕ), 
  (a = b + 3) →
  (b = c + 6) →
  (c = d + 15) →
  (d = 50) →
  a = 74 :=
by
  intros a b c d h1 h2 h3 h4
  sorry

end determine_a_value_l61_61089


namespace cos_alpha_minus_270_l61_61973

open Real

theorem cos_alpha_minus_270 (α : ℝ) : 
  sin (540 * (π / 180) + α) = -4 / 5 → cos (α - 270 * (π / 180)) = -4 / 5 :=
by
  sorry

end cos_alpha_minus_270_l61_61973


namespace problem_a_b_n_geq_1_l61_61437

theorem problem_a_b_n_geq_1 (a b n : ℕ) (h1 : a > b) (h2 : b > 1) (h3 : Odd b) (h4 : n > 0)
  (h5 : b^n ∣ a^n - 1) : a^b > 3^n / n := 
by 
  sorry

end problem_a_b_n_geq_1_l61_61437


namespace number_of_ways_to_put_cousins_in_rooms_l61_61135

/-- Given 5 cousins and 4 identical rooms, the number of distinct ways to assign the cousins to the rooms is 52. -/
theorem number_of_ways_to_put_cousins_in_rooms : 
  let num_cousins := 5
  let num_rooms := 4
  number_of_ways_to_put_cousins_in_rooms num_cousins num_rooms := 52 :=
sorry

end number_of_ways_to_put_cousins_in_rooms_l61_61135


namespace num_possible_integer_values_l61_61264

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l61_61264


namespace circle_passing_three_points_l61_61701

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61701


namespace find_angles_l61_61549

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  2 * y = x + z

theorem find_angles (a : ℝ) (h1 : 0 < a) (h2 : a < 360)
  (h3 : is_arithmetic_sequence (Real.sin a) (Real.sin (2 * a)) (Real.sin (3 * a))) :
  a = 90 ∨ a = 270 := by
  sorry

end find_angles_l61_61549


namespace triangle_obtuse_at_most_one_l61_61494

open Real -- Work within the Real number system

-- Definitions and main proposition
def is_obtuse (angle : ℝ) : Prop := angle > 90

def triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem triangle_obtuse_at_most_one (a b c : ℝ) (h : triangle a b c) :
  is_obtuse a ∧ is_obtuse b → false :=
by
  sorry

end triangle_obtuse_at_most_one_l61_61494


namespace gcd_factorial_8_10_l61_61969

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l61_61969


namespace determine_range_of_m_l61_61807

variable {m : ℝ}

-- Condition (p) for all x in ℝ, x^2 - mx + 3/2 > 0
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + (3 / 2) > 0

-- Condition (q) the foci of the ellipse lie on the x-axis, implying 2 < m < 3
def condition_q (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ ((3 - m) > 0) ∧ ((m - 1) > (3 - m))

theorem determine_range_of_m (h1 : condition_p m) (h2 : condition_q m) : 2 < m ∧ m < Real.sqrt 6 :=
  sorry

end determine_range_of_m_l61_61807


namespace circle_equation_through_points_l61_61696

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61696


namespace blue_lights_count_l61_61480

def num_colored_lights := 350
def num_red_lights := 85
def num_yellow_lights := 112
def num_green_lights := 65
def num_blue_lights := num_colored_lights - (num_red_lights + num_yellow_lights + num_green_lights)

theorem blue_lights_count : num_blue_lights = 88 := by
  sorry

end blue_lights_count_l61_61480


namespace circle_passes_through_points_l61_61625

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61625


namespace faye_money_left_is_30_l61_61091

-- Definitions and conditions
def initial_money : ℝ := 20
def mother_gave (initial : ℝ) : ℝ := 2 * initial
def cost_of_cupcakes : ℝ := 10 * 1.5
def cost_of_cookies : ℝ := 5 * 3

-- Calculate the total money Faye has left
def total_money_left (initial : ℝ) (mother_gave_ : ℝ) (cost_cupcakes : ℝ) (cost_cookies : ℝ) : ℝ :=
  initial + mother_gave_ - (cost_cupcakes + cost_cookies)

-- Theorem stating the money left
theorem faye_money_left_is_30 :
  total_money_left initial_money (mother_gave initial_money) cost_of_cupcakes cost_of_cookies = 30 :=
by sorry

end faye_money_left_is_30_l61_61091


namespace solve_inequality_system_l61_61029

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61029


namespace negate_prop_l61_61165

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l61_61165


namespace solve_inequality_system_l61_61027

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61027


namespace circle_equation_correct_l61_61611

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61611


namespace circle_equation_value_l61_61855

theorem circle_equation_value (a : ℝ) :
  (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0 → False) → a = -1 :=
by
  intros h
  sorry

end circle_equation_value_l61_61855


namespace sequence_sum_l61_61868

-- Define the arithmetic sequence and conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values used in the problem
def specific_condition (a : ℕ → ℝ) : Prop :=
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450)

-- The proof goal that needs to be established
theorem sequence_sum (a : ℕ → ℝ) (h1 : arithmetic_seq a) (h2 : specific_condition a) : a 2 + a 8 = 180 :=
by
  sorry

end sequence_sum_l61_61868


namespace neg_ln_gt_zero_l61_61167

theorem neg_ln_gt_zero {x : ℝ} : (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ ∃ x : ℝ, Real.log (x^2 + 1) ≤ 0 := by
  sorry

end neg_ln_gt_zero_l61_61167


namespace fractional_addition_l61_61792

theorem fractional_addition : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by
  sorry

end fractional_addition_l61_61792


namespace cannot_reach_eighth_vertex_l61_61430

def Point := ℕ × ℕ × ℕ

def symmetry (p1 p2 : Point) : Point :=
  let (a, b, c) := p1
  let (a', b', c') := p2
  (2 * a' - a, 2 * b' - b, 2 * c' - c)

def vertices : List Point :=
  [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

theorem cannot_reach_eighth_vertex : ∀ (p : Point), p ∈ vertices → ∀ (q : Point), q ∈ vertices → 
  ¬(symmetry p q = (1, 1, 1)) :=
by
  sorry

end cannot_reach_eighth_vertex_l61_61430


namespace inequality_system_solution_l61_61020

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l61_61020


namespace triangle_perimeter_l61_61563

-- Define the conditions of the problem
def a := 4
def b := 8
def quadratic_eq (x : ℝ) : Prop := x^2 - 14 * x + 40 = 0

-- Define the perimeter calculation, ensuring triangle inequality and correct side length
def valid_triangle (x : ℝ) : Prop :=
  x ≠ a ∧ x ≠ b ∧ quadratic_eq x ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)

-- Define the problem statement as a theorem
theorem triangle_perimeter : ∃ x : ℝ, valid_triangle x ∧ (a + b + x = 22) :=
by {
  -- Placeholder for the proof
  sorry
}

end triangle_perimeter_l61_61563


namespace sum_of_possible_values_d_l61_61355

theorem sum_of_possible_values_d :
  let range_8 := (512, 4095)
  let digits_in_base_16 := 3
  (∀ n, n ∈ Set.Icc range_8.1 range_8.2 → (Nat.digits 16 n).length = digits_in_base_16)
  → digits_in_base_16 = 3 :=
by
  sorry

end sum_of_possible_values_d_l61_61355


namespace not_possible_to_fill_grid_l61_61434

theorem not_possible_to_fill_grid :
  ¬ ∃ (f : Fin 7 → Fin 7 → ℝ), ∀ i j : Fin 7,
    ((if j > 0 then f i (j - 1) else 0) +
     (if j < 6 then f i (j + 1) else 0) +
     (if i > 0 then f (i - 1) j else 0) +
     (if i < 6 then f (i + 1) j else 0)) = 1 :=
by
  sorry

end not_possible_to_fill_grid_l61_61434


namespace circle_equation_l61_61647

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61647


namespace negation_of_forall_statement_l61_61163

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l61_61163


namespace ellipse_focus_value_l61_61856

theorem ellipse_focus_value (m : ℝ) (h1 : m > 0) :
  (∃ (x y : ℝ), (x, y) = (-4, 0) ∧ (25 - m^2 = 16)) → m = 3 :=
by
  sorry

end ellipse_focus_value_l61_61856


namespace original_number_is_14_l61_61426

def two_digit_number_increased_by_2_or_4_results_fourfold (x : ℕ) : Prop :=
  (x >= 10) ∧ (x < 100) ∧ 
  (∃ (a b : ℕ), a + 2 = ((x / 10 + 2) % 10) ∧ b + 2 = (x % 10)) ∧
  (4 * x = ((x / 10 + 2) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 2) * 10 + (x % 10 + 4)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 4)))

theorem original_number_is_14 : ∃ x : ℕ, two_digit_number_increased_by_2_or_4_results_fourfold x ∧ x = 14 :=
by
  sorry

end original_number_is_14_l61_61426


namespace prime_square_plus_eight_is_prime_l61_61796

theorem prime_square_plus_eight_is_prime (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 8)) : p = 3 :=
sorry

end prime_square_plus_eight_is_prime_l61_61796


namespace sum_of_remainders_mod_30_l61_61485

theorem sum_of_remainders_mod_30 (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 11) (h3 : c % 30 = 19) :
  (a + b + c) % 30 = 14 :=
by
  sorry

end sum_of_remainders_mod_30_l61_61485


namespace acute_triangle_inequality_l61_61431

theorem acute_triangle_inequality
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (h1 : 0 < A ∧ A < π/2)
  (h2 : 0 < B ∧ B < π/2)
  (h3 : 0 < C ∧ C < π/2)
  (h4 : A + B + C = π)
  (h5 : R = 1)
  (h6 : a = 2 * R * Real.sin A)
  (h7 : b = 2 * R * Real.sin B)
  (h8 : c = 2 * R * Real.sin C) :
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 :=
by
  sorry

end acute_triangle_inequality_l61_61431


namespace functional_equation_option_A_option_B_option_C_l61_61816

-- Given conditions
def domain_of_f (f : ℝ → ℝ) : Set ℝ := Set.univ

theorem functional_equation (f : ℝ → ℝ) (x y : ℝ) :
  f (x * y) = y^2 * f x + x^2 * f y := sorry

-- Statements to be proved
theorem option_A (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 0 = 0 := sorry

theorem option_B (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : f 1 = 0 := sorry

theorem option_C (f : ℝ → ℝ) (h : ∀ x y, f(x * y) = y^2 * f(x) + x^2 * f(y)) : ∀ x, f(-x) = f x := sorry

end functional_equation_option_A_option_B_option_C_l61_61816


namespace arithmetic_sequence_third_eighth_term_sum_l61_61102

variable {α : Type*} [AddCommGroup α] [Module ℚ α]

def arith_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_third_eighth_term_sum {a : ℕ → ℚ} {S : ℕ → ℚ} 
  (h_seq: ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum: arith_sequence_sum a S) 
  (h_S10 : S 10 = 4) : 
  a 3 + a 8 = 4 / 5 :=
by
  sorry

end arithmetic_sequence_third_eighth_term_sum_l61_61102


namespace circle_equation_l61_61645

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61645


namespace equation_of_circle_passing_through_points_l61_61689

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61689


namespace trader_gain_percentage_l61_61207

-- Definition of the given conditions
def cost_per_pen (C : ℝ) := C
def num_pens_sold := 90
def gain_from_sale (C : ℝ) := 15 * C
def total_cost (C : ℝ) := 90 * C

-- Statement of the problem
theorem trader_gain_percentage (C : ℝ) : 
  (((gain_from_sale C) / (total_cost C)) * 100) = 16.67 :=
by
  -- This part will contain the step-by-step proof, omitted here
  sorry

end trader_gain_percentage_l61_61207


namespace RectangleAreaDiagonalk_l61_61760

theorem RectangleAreaDiagonalk {length width : ℝ} {d : ℝ}
  (h_ratio : length / width = 5 / 2)
  (h_perimeter : 2 * (length + width) = 42)
  (h_diagonal : d = Real.sqrt (length^2 + width^2))
  : (∃ k, k = 10 / 29 ∧ ∀ A, A = k * d^2) :=
by {
  sorry
}

end RectangleAreaDiagonalk_l61_61760


namespace functional_eq_properties_l61_61818

theorem functional_eq_properties (f : ℝ → ℝ) (h_domain : ∀ x, x ∈ ℝ) (h_func_eq : ∀ x y, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x, f x = f (-x) :=
by {
  sorry
}

end functional_eq_properties_l61_61818


namespace solve_inequality_system_l61_61007

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61007


namespace smallest_k_satisfies_l61_61946

noncomputable def sqrt (x : ℝ) : ℝ := x ^ (1 / 2 : ℝ)

theorem smallest_k_satisfies (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (sqrt (x * y)) + (1 / 2) * (sqrt (abs (x - y))) ≥ (x + y) / 2 :=
by
  sorry

end smallest_k_satisfies_l61_61946


namespace pizza_cost_per_slice_correct_l61_61870

noncomputable def pizza_cost_per_slice : ℝ :=
  let base_pizza_cost := 10.00
  let first_topping_cost := 2.00
  let next_two_toppings_cost := 2.00
  let remaining_toppings_cost := 2.00
  let total_cost := base_pizza_cost + first_topping_cost + next_two_toppings_cost + remaining_toppings_cost
  total_cost / 8

theorem pizza_cost_per_slice_correct :
  pizza_cost_per_slice = 2.00 :=
by
  unfold pizza_cost_per_slice
  sorry

end pizza_cost_per_slice_correct_l61_61870


namespace circle_passing_through_points_eqn_l61_61652

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61652


namespace digits_base8_sum_l61_61832

open Nat

theorem digits_base8_sum (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z) 
  (h_distinct : X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) (h_base8 : X < 8 ∧ Y < 8 ∧ Z < 8) 
  (h_eq : (8^2 * X + 8 * Y + Z) + (8^2 * Y + 8 * Z + X) + (8^2 * Z + 8 * X + Y) = 8^3 * X + 8^2 * X + 8 * X) : 
  Y + Z = 7 :=
by
  sorry

end digits_base8_sum_l61_61832


namespace lcm_gcd_eq_product_l61_61318

theorem lcm_gcd_eq_product {a b : ℕ} (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 :=
  sorry

end lcm_gcd_eq_product_l61_61318


namespace problem_l61_61819

open Function

variable {R : Type*} [Ring R]

def f (x : R) : R := -- Assume the function type

theorem problem (h : ∀ x y : R, f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end problem_l61_61819


namespace two_fifths_in_fraction_l61_61416

theorem two_fifths_in_fraction : 
  (∃ (k : ℚ), k = (9/3) / (2/5) ∧ k = 15/2) :=
by 
  sorry

end two_fifths_in_fraction_l61_61416


namespace total_ladders_climbed_in_inches_l61_61435

-- Define the conditions as hypotheses
def keaton_ladder_length := 30
def keaton_climbs := 20
def reece_ladder_difference := 4
def reece_climbs := 15
def feet_to_inches := 12

-- Define the lengths climbed by Keaton and Reece
def keaton_total_feet := keaton_ladder_length * keaton_climbs
def reece_ladder_length := keaton_ladder_length - reece_ladder_difference
def reece_total_feet := reece_ladder_length * reece_climbs

-- Calculate the total feet climbed and convert to inches
def total_feet := keaton_total_feet + reece_total_feet
def total_inches := total_feet * feet_to_inches

-- Prove the final result
theorem total_ladders_climbed_in_inches : total_inches = 11880 := by
  sorry

end total_ladders_climbed_in_inches_l61_61435


namespace minimum_value_of_angle_l61_61236

theorem minimum_value_of_angle
  (α : ℝ)
  (h : ∃ x y : ℝ, (x, y) = (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))) :
  α = 11 * Real.pi / 6 :=
sorry

end minimum_value_of_angle_l61_61236


namespace calc_hash_80_l61_61087

def hash (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem calc_hash_80 : hash (hash (hash 80)) = 17.28 :=
by 
  sorry

end calc_hash_80_l61_61087


namespace initial_courses_of_bricks_l61_61872

theorem initial_courses_of_bricks (x : ℕ) : 
    400 * x + 2 * 400 - 400 / 2 = 1800 → x = 3 :=
by
  sorry

end initial_courses_of_bricks_l61_61872


namespace value_of_a5_l61_61867

variable (a_n : ℕ → ℝ)
variable (a1 a9 a5 : ℝ)

-- Given conditions
axiom a1_plus_a9_eq_10 : a1 + a9 = 10
axiom arithmetic_sequence : ∀ n, a_n n = a1 + (n - 1) * (a_n 2 - a1)

-- Prove that a5 = 5
theorem value_of_a5 : a5 = 5 :=
by
  sorry

end value_of_a5_l61_61867


namespace mass_percentage_H_in_chlorous_acid_l61_61801

noncomputable def mass_percentage_H_in_HClO2 : ℚ :=
  let molar_mass_H : ℚ := 1.01
  let molar_mass_Cl : ℚ := 35.45
  let molar_mass_O : ℚ := 16.00
  let molar_mass_HClO2 : ℚ := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  (molar_mass_H / molar_mass_HClO2) * 100

theorem mass_percentage_H_in_chlorous_acid :
  mass_percentage_H_in_HClO2 = 1.475 := by
  sorry

end mass_percentage_H_in_chlorous_acid_l61_61801


namespace ethanol_in_full_tank_l61_61204

theorem ethanol_in_full_tank:
  ∀ (capacity : ℕ) (vol_A : ℕ) (vol_B : ℕ) (eth_A_perc : ℝ) (eth_B_perc : ℝ) (eth_A : ℝ) (eth_B : ℝ),
  capacity = 208 →
  vol_A = 82 →
  vol_B = (capacity - vol_A) →
  eth_A_perc = 0.12 →
  eth_B_perc = 0.16 →
  eth_A = vol_A * eth_A_perc →
  eth_B = vol_B * eth_B_perc →
  eth_A + eth_B = 30 :=
by
  intros capacity vol_A vol_B eth_A_perc eth_B_perc eth_A eth_B h1 h2 h3 h4 h5 h6 h7
  sorry

end ethanol_in_full_tank_l61_61204


namespace x_in_M_sufficient_condition_for_x_in_N_l61_61587

def M := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0}
def N := {y : ℝ | ∃ x : ℝ, y = Real.sqrt ((1 - x) / x)}

theorem x_in_M_sufficient_condition_for_x_in_N :
  (∀ x, x ∈ M → x ∈ N) ∧ ¬ (∀ x, x ∈ N → x ∈ M) :=
by sorry

end x_in_M_sufficient_condition_for_x_in_N_l61_61587


namespace circle_equation_correct_l61_61610

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61610


namespace circle_through_points_l61_61667

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61667


namespace similar_triangle_shortest_side_l61_61155

theorem similar_triangle_shortest_side (a b c : ℕ) (p : ℕ) (h : a = 8 ∧ b = 10 ∧ c = 12 ∧ p = 150) :
  ∃ x : ℕ, (x = p / (a + b + c) ∧ 8 * x = 40) :=
by
  sorry

end similar_triangle_shortest_side_l61_61155


namespace intersection_points_with_x_axis_l61_61718

theorem intersection_points_with_x_axis (a : ℝ) :
    (∃ x : ℝ, a * x^2 - a * x + 3 * x + 1 = 0 ∧ 
              ∀ x' : ℝ, (x' ≠ x → a * x'^2 - a * x' + 3 * x' + 1 ≠ 0)) ↔ 
    (a = 0 ∨ a = 1 ∨ a = 9) := by 
  sorry

end intersection_points_with_x_axis_l61_61718


namespace radical_axis_eq_l61_61797

-- Definitions of the given circles
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- The theorem proving that the equation of the radical axis is 3x - y - 9 = 0
theorem radical_axis_eq (x y : ℝ) :
  (circle1_eq x y) ∧ (circle2_eq x y) → 3 * x - y - 9 = 0 :=
sorry

end radical_axis_eq_l61_61797


namespace equation_of_circle_ABC_l61_61616

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61616


namespace circle_passing_through_points_l61_61603

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61603


namespace ellipse_foci_distance_l61_61067

noncomputable def distance_between_foci (h k a b : ℝ) : ℝ :=
  2 * real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (h k a b : ℝ), 
  (h = 6) → (k = 3) → (a = 6) → (b = 3) → 
  distance_between_foci h k a b = 6 * real.sqrt 3 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l61_61067


namespace roots_of_equation_l61_61085

theorem roots_of_equation :
  ∀ x : ℚ, (3 * x^2 / (x - 2) - (5 * x + 10) / 4 + (9 - 9 * x) / (x - 2) + 2 = 0) ↔ 
           (x = 6 ∨ x = 17/3) := 
sorry

end roots_of_equation_l61_61085


namespace circle_equation_l61_61651

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61651


namespace cubic_expression_equals_two_l61_61995

theorem cubic_expression_equals_two (x : ℝ) (h : 2 * x ^ 2 - 3 * x - 2022 = 0) :
  2 * x ^ 3 - x ^ 2 - 2025 * x - 2020 = 2 :=
sorry

end cubic_expression_equals_two_l61_61995


namespace good_permutations_count_l61_61557

-- Define the main problem and the conditions
theorem good_permutations_count (n : ℕ) (hn : n > 0) : 
  ∃ P : ℕ → ℕ, 
  (P n = (1 / Real.sqrt 5) * (((1 + Real.sqrt 5) / 2) ^ (n + 1) - ((1 - Real.sqrt 5) / 2) ^ (n + 1))) := 
sorry

end good_permutations_count_l61_61557


namespace equation_of_circle_ABC_l61_61614

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61614


namespace circle_through_points_l61_61669

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61669


namespace topics_assignment_l61_61052

theorem topics_assignment (students groups arrangements : ℕ) (h1 : students = 6) (h2 : groups = 3) (h3 : arrangements = 90) :
  let T := arrangements / (students * (students - 1) / 2 * (4 * 3 / 2 * 1))
  T = 1 :=
by
  sorry

end topics_assignment_l61_61052


namespace reciprocal_neg_2023_l61_61726

theorem reciprocal_neg_2023 : (1 / (-2023: ℤ)) = - (1 / 2023) :=
by
  -- proof goes here
  sorry

end reciprocal_neg_2023_l61_61726


namespace sum_of_consecutive_numbers_mod_13_l61_61217

theorem sum_of_consecutive_numbers_mod_13 :
  ((8930 + 8931 + 8932 + 8933 + 8934) % 13) = 5 :=
by
  sorry

end sum_of_consecutive_numbers_mod_13_l61_61217


namespace brownies_total_l61_61308

theorem brownies_total :
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  after_mooney_ate + additional_brownies = 36 :=
by
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  show after_mooney_ate + additional_brownies = 36
  sorry

end brownies_total_l61_61308


namespace find_number_l61_61319

theorem find_number (x : ℤ) (h : x - 254 + 329 = 695) : x = 620 :=
sorry

end find_number_l61_61319


namespace evaluate_expression_l61_61545

theorem evaluate_expression : (Real.sqrt (Real.sqrt 5 ^ 4))^3 = 125 := by
  sorry

end evaluate_expression_l61_61545


namespace total_people_museum_l61_61174

-- Conditions
def first_bus_people : ℕ := 12
def second_bus_people := 2 * first_bus_people
def third_bus_people := second_bus_people - 6
def fourth_bus_people := first_bus_people + 9

-- Question to prove
theorem total_people_museum : first_bus_people + second_bus_people + third_bus_people + fourth_bus_people = 75 :=
by
  -- The proof is skipped but required to complete the theorem
  sorry

end total_people_museum_l61_61174


namespace identity_x_squared_minus_y_squared_l61_61840

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61840


namespace nominal_rate_of_interest_l61_61468

noncomputable def nominal_rate (EAR : ℝ) (n : ℕ) : ℝ :=
  2 * (Real.sqrt (1 + EAR) - 1)

theorem nominal_rate_of_interest :
  nominal_rate 0.1025 2 = 0.100476 :=
by sorry

end nominal_rate_of_interest_l61_61468


namespace max_value_64_l61_61424

-- Define the types and values of gemstones
structure Gemstone where
  weight : ℕ
  value : ℕ

-- Introduction of the three types of gemstones
def gem1 : Gemstone := ⟨3, 9⟩
def gem2 : Gemstone := ⟨5, 16⟩
def gem3 : Gemstone := ⟨2, 5⟩

-- Maximum weight Janet can carry
def max_weight := 20

-- Problem statement: Proving the maximum value Janet can carry is $64
theorem max_value_64 (n1 n2 n3 : ℕ) (h1 : n1 ≥ 15) (h2 : n2 ≥ 15) (h3 : n3 ≥ 15) 
  (weight_limit : n1 * gem1.weight + n2 * gem2.weight + n3 * gem3.weight ≤ max_weight) : 
  n1 * gem1.value + n2 * gem2.value + n3 * gem3.value ≤ 64 :=
sorry

end max_value_64_l61_61424


namespace fraction_simplification_l61_61316

theorem fraction_simplification (x : ℚ) : 
  (3 / 4) * 60 - x * 60 + 63 = 12 → 
  x = (8 / 5) :=
by
  sorry

end fraction_simplification_l61_61316


namespace four_digit_integer_l61_61152

theorem four_digit_integer (a b c d : ℕ) 
(h1: a + b + c + d = 14) (h2: b + c = 9) (h3: a - d = 1)
(h4: (a - b + c - d) % 11 = 0) : 1000 * a + 100 * b + 10 * c + d = 3542 :=
by
  sorry

end four_digit_integer_l61_61152


namespace geometric_series_sum_l61_61348

noncomputable def geometric_sum : ℚ :=
  let a := (2^3 : ℚ) / (3^3)
  let r := (2 : ℚ) / 3
  let n := 12 - 3 + 1
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum = 1440600 / 59049 :=
by
  sorry

end geometric_series_sum_l61_61348


namespace first_tap_fill_time_l61_61928

theorem first_tap_fill_time (T : ℝ) (h1 : T > 0) (h2 : 12 > 0) 
  (h3 : 1/T - 1/12 = 1/12) : T = 6 :=
sorry

end first_tap_fill_time_l61_61928


namespace number_of_ways_to_put_cousins_in_rooms_l61_61136

/-- Given 5 cousins and 4 identical rooms, the number of distinct ways to assign the cousins to the rooms is 52. -/
theorem number_of_ways_to_put_cousins_in_rooms : 
  let num_cousins := 5
  let num_rooms := 4
  number_of_ways_to_put_cousins_in_rooms num_cousins num_rooms := 52 :=
sorry

end number_of_ways_to_put_cousins_in_rooms_l61_61136


namespace b_distance_behind_proof_l61_61860

-- Given conditions
def race_distance : ℕ := 1000
def a_time : ℕ := 40
def b_delay : ℕ := 10

def a_speed : ℕ := race_distance / a_time
def b_distance_behind : ℕ := a_speed * b_delay

theorem b_distance_behind_proof : b_distance_behind = 250 := by
  -- Prove that b_distance_behind = 250
  sorry

end b_distance_behind_proof_l61_61860


namespace pure_imaginary_complex_l61_61279

theorem pure_imaginary_complex (a : ℝ) (i : ℂ) (h : i * i = -1) (p : (1 + a * i) / (1 - i) = (0 : ℂ) + b * i) :
  a = 1 := 
sorry

end pure_imaginary_complex_l61_61279


namespace cistern_problem_l61_61198

theorem cistern_problem (fill_rate empty_rate net_rate : ℝ) (T : ℝ) : 
  fill_rate = 1 / 3 →
  net_rate = 7 / 30 →
  empty_rate = 1 / T →
  net_rate = fill_rate - empty_rate →
  T = 10 :=
by
  intros
  sorry

end cistern_problem_l61_61198


namespace larger_cylinder_candies_l61_61748

theorem larger_cylinder_candies (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) (h₁ : v₁ = 72) (h₂ : c₁ = 30) (h₃ : v₂ = 216) (h₄ : (c₁ : ℝ)/v₁ = (c₂ : ℝ)/v₂) : c₂ = 90 := by
  -- v1 h1 h2 v2 c2 h4 are directly appearing in the conditions
  -- ratio h4 states the condition for densities to be the same 
  sorry

end larger_cylinder_candies_l61_61748


namespace sin_1200_eq_sqrt3_div_2_l61_61542

theorem sin_1200_eq_sqrt3_div_2 : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_1200_eq_sqrt3_div_2_l61_61542


namespace passenger_speed_relative_forward_correct_l61_61761

-- Define the conditions
def train_speed : ℝ := 60     -- Train's speed in km/h
def passenger_speed_inside_train : ℝ := 3  -- Passenger's speed inside the train in km/h

-- Define the effective speed of the passenger relative to the railway track when moving forward
def passenger_speed_relative_forward (train_speed passenger_speed_inside_train : ℝ) : ℝ :=
  train_speed + passenger_speed_inside_train

-- Prove that the passenger's speed relative to the railway track is 63 km/h when moving forward
theorem passenger_speed_relative_forward_correct :
  passenger_speed_relative_forward train_speed passenger_speed_inside_train = 63 := by
  sorry

end passenger_speed_relative_forward_correct_l61_61761


namespace taxi_ride_cost_l61_61075

-- Definitions
def initial_fee : Real := 2
def distance_in_miles : Real := 4
def cost_per_mile : Real := 2.5

-- Theorem statement
theorem taxi_ride_cost (initial_fee : Real) (distance_in_miles : Real) (cost_per_mile : Real) : 
  initial_fee + distance_in_miles * cost_per_mile = 12 := 
by
  sorry

end taxi_ride_cost_l61_61075


namespace average_marks_all_students_proof_l61_61923

-- Definitions based on the given conditions
def class1_student_count : ℕ := 35
def class2_student_count : ℕ := 45
def class1_average_marks : ℕ := 40
def class2_average_marks : ℕ := 60

-- Total marks calculations
def class1_total_marks : ℕ := class1_student_count * class1_average_marks
def class2_total_marks : ℕ := class2_student_count * class2_average_marks
def total_marks : ℕ := class1_total_marks + class2_total_marks

-- Total student count
def total_student_count : ℕ := class1_student_count + class2_student_count

-- Average marks of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_student_count

-- Lean statement to prove
theorem average_marks_all_students_proof
  (h1 : class1_student_count = 35)
  (h2 : class2_student_count = 45)
  (h3 : class1_average_marks = 40)
  (h4 : class2_average_marks = 60) :
  average_marks_all_students = 51.25 := by
  sorry

end average_marks_all_students_proof_l61_61923


namespace total_ice_cream_volume_l61_61040

def cone_height : ℝ := 10
def cone_radius : ℝ := 1.5
def cylinder_height : ℝ := 2
def cylinder_radius : ℝ := 1.5
def hemisphere_radius : ℝ := 1.5

theorem total_ice_cream_volume : 
  (1 / 3 * π * cone_radius ^ 2 * cone_height) +
  (π * cylinder_radius ^ 2 * cylinder_height) +
  (2 / 3 * π * hemisphere_radius ^ 3) = 14.25 * π :=
by sorry

end total_ice_cream_volume_l61_61040


namespace total_bricks_used_l61_61315

def numberOfCoursesPerWall := 6
def bricksPerCourse := 10
def numberOfWalls := 4
def incompleteCourses := 2

theorem total_bricks_used :
  (numberOfCoursesPerWall * bricksPerCourse * (numberOfWalls - 1)) + ((numberOfCoursesPerWall - incompleteCourses) * bricksPerCourse) = 220 :=
by
  -- Proof goes here
  sorry

end total_bricks_used_l61_61315


namespace inequality_system_solution_l61_61021

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l61_61021


namespace compute_expression_l61_61350

theorem compute_expression : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end compute_expression_l61_61350


namespace number_of_integer_values_l61_61255

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l61_61255


namespace equation_of_circle_passing_through_points_l61_61686

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61686


namespace circle_passing_through_points_l61_61628

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61628


namespace sum_of_remainders_l61_61938

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9) = 3 :=
by
  sorry

end sum_of_remainders_l61_61938


namespace find_f_2016_l61_61305

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_0_eq_2016 : f 0 = 2016

axiom f_x_plus_2_minus_f_x_leq : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2 ^ x

axiom f_x_plus_6_minus_f_x_geq : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2 ^ x

theorem find_f_2016 : f 2016 = 2015 + 2 ^ 2020 :=
sorry

end find_f_2016_l61_61305


namespace gcd_factorial_l61_61965

-- Definitions and conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Theorem statement
theorem gcd_factorial : gcd (factorial 8) (factorial 10) = factorial 8 := by
  -- The proof is omitted
  sorry

end gcd_factorial_l61_61965


namespace grandma_vasya_cheapest_option_l61_61244

/-- Constants and definitions for the cost calculations --/
def train_ticket_cost : ℕ := 200
def collected_berries_kg : ℕ := 5
def market_berries_cost_per_kg : ℕ := 150
def sugar_cost_per_kg : ℕ := 54
def jam_made_per_kg_combination : ℕ := 15 / 10  -- representing 1.5 kg (as ratio 15/10)
def ready_made_jam_cost_per_kg : ℕ := 220

/-- Compute the cost per kg of jam for different methods --/
def cost_per_kg_jam_option1 : ℕ := (train_ticket_cost / collected_berries_kg + sugar_cost_per_kg)
def cost_per_kg_jam_option2 : ℕ := market_berries_cost_per_kg + sugar_cost_per_kg
def cost_per_kg_jam_option3 : ℕ := ready_made_jam_cost_per_kg

/-- Numbers converted to per 1.5 kg --/
def cost_for_1_5_kg (cost_per_kg: ℕ) : ℕ := cost_per_kg * (15 / 10)

/-- Theorem stating option 1 is the cheapest --/
theorem grandma_vasya_cheapest_option :
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option2 ∧
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option3 :=
by sorry

end grandma_vasya_cheapest_option_l61_61244


namespace coupon_discount_l61_61051

theorem coupon_discount (total_before_coupon : ℝ) (amount_paid_per_friend : ℝ) (number_of_friends : ℕ) :
  total_before_coupon = 100 ∧ amount_paid_per_friend = 18.8 ∧ number_of_friends = 5 →
  ∃ discount_percentage : ℝ, discount_percentage = 6 :=
by
  sorry

end coupon_discount_l61_61051


namespace range_of_f_minimal_lambda_l61_61826

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ∈ set.Ioo (-1 : ℝ) 1 :=
sorry

theorem minimal_lambda :
  ∃ (λ : ℝ) (h : λ = 1 / 2), ∀ x₁ x₂ ∈ set.Icc (Real.log (1 / 2)) (Real.log 2),
  abs ((f x₁ + f x₂) / (x₁ + x₂)) < λ :=
sorry

end range_of_f_minimal_lambda_l61_61826


namespace second_set_number_l61_61177

theorem second_set_number (x : ℕ) (sum1 : ℕ) (avg2 : ℕ) (total_avg : ℕ)
  (h1 : sum1 = 98) (h2 : avg2 = 11) (h3 : total_avg = 8)
  (h4 : 16 + x ≠ 0) :
  (98 + avg2 * x = total_avg * (x + 16)) → x = 10 :=
by
  sorry

end second_set_number_l61_61177


namespace inequality_system_solution_l61_61018

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l61_61018


namespace honey_harvest_this_year_l61_61947

def last_year_harvest : ℕ := 2479
def increase_this_year : ℕ := 6085

theorem honey_harvest_this_year : last_year_harvest + increase_this_year = 8564 :=
by {
  sorry
}

end honey_harvest_this_year_l61_61947


namespace joan_picked_apples_l61_61871

theorem joan_picked_apples (a b c : ℕ) (h1 : b = 27) (h2 : c = 70) (h3 : c = a + b) : a = 43 :=
by
  sorry

end joan_picked_apples_l61_61871


namespace solve_inequality_system_l61_61003

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61003


namespace percentage_exceeds_self_l61_61362

theorem percentage_exceeds_self (N : ℝ) (P : ℝ) (hN : N = 75) (h_condition : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end percentage_exceeds_self_l61_61362


namespace plane_equation_and_gcd_l61_61469

variable (x y z : ℝ)

theorem plane_equation_and_gcd (A B C D : ℤ) (h1 : A = 8) (h2 : B = -6) (h3 : C = 5) (h4 : D = -125) :
    (A * x + B * y + C * z + D = 0 ↔ x = 8 ∧ y = -6 ∧ z = 5) ∧
    Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by sorry

end plane_equation_and_gcd_l61_61469


namespace cousin_distribution_count_l61_61141

-- Definition of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Definition to count the number of distributions
noncomputable def count_cousin_distributions : ℕ :=
  let case1 := 1 in -- (5,0,0,0)
  let case2 := choose 5 1 in -- (4,1,0,0)
  let case3 := choose 5 3 in -- (3,2,0,0)
  let case4 := choose 5 3 in -- (3,1,1,0)
  let case5 := choose 5 2 * choose 3 2 in -- (2,2,1,0)
  let case6 := choose 5 2 in -- (2,1,1,1)
  case1 + case2 + case3 + case4 + case5 + case6

-- Theorem to prove
theorem cousin_distribution_count : count_cousin_distributions = 66 := by
  sorry

end cousin_distribution_count_l61_61141


namespace obtuse_triangle_l61_61433

variable (A B C : ℝ)
variable (angle_sum : A + B + C = 180)
variable (cond1 : A + B = 141)
variable (cond2 : B + C = 165)

theorem obtuse_triangle : B > 90 :=
by
  sorry

end obtuse_triangle_l61_61433


namespace number_of_zeros_f_l61_61901

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x

theorem number_of_zeros_f : ∃! n : ℕ, n = 2 ∧ ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end number_of_zeros_f_l61_61901


namespace circle_through_points_l61_61662

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61662


namespace quadrilateral_area_l61_61044

theorem quadrilateral_area 
  (AB BC DC : ℝ)
  (hAB_perp_BC : true)
  (hDC_perp_BC : true)
  (hAB_eq : AB = 8)
  (hDC_eq : DC = 3)
  (hBC_eq : BC = 10) : 
  (1 / 2 * (AB + DC) * BC = 55) :=
by 
  sorry

end quadrilateral_area_l61_61044


namespace circle_through_points_l61_61641

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61641


namespace Ken_bought_2_pounds_of_steak_l61_61528

theorem Ken_bought_2_pounds_of_steak (pound_cost total_paid change: ℝ) 
    (h1 : pound_cost = 7) 
    (h2 : total_paid = 20) 
    (h3 : change = 6) : 
    (total_paid - change) / pound_cost = 2 :=
by
  sorry

end Ken_bought_2_pounds_of_steak_l61_61528


namespace typeA_selling_price_maximize_profit_l61_61077

theorem typeA_selling_price (sales_last_year : ℝ) (sales_increase_rate : ℝ) (price_increase : ℝ) 
                            (cars_sold_last_year : ℝ) : 
                            (sales_last_year = 32000) ∧ (sales_increase_rate = 1.25) ∧ 
                            (price_increase = 400) ∧ 
                            (sales_last_year / cars_sold_last_year = (sales_last_year * sales_increase_rate) / (cars_sold_last_year + price_increase)) → 
                            (cars_sold_last_year = 1600) :=
by
  sorry

theorem maximize_profit (typeA_price : ℝ) (typeB_price : ℝ) (typeA_cost : ℝ) (typeB_cost : ℝ) 
                        (total_cars : ℕ) :
                        (typeA_price = 2000) ∧ (typeB_price = 2400) ∧ 
                        (typeA_cost = 1100) ∧ (typeB_cost = 1400) ∧ 
                        (total_cars = 50) ∧ 
                        (∀ m : ℕ, m ≤ 50 / 3) → 
                        ∃ m : ℕ, (m = 17) ∧ (50 - m * 2 ≤ 33) :=
by
  sorry

end typeA_selling_price_maximize_profit_l61_61077


namespace percentage_increase_in_overtime_rate_l61_61504

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def total_compensation : ℝ := 976
def total_hours_worked : ℝ := 52

theorem percentage_increase_in_overtime_rate :
  ((total_compensation - (regular_rate * regular_hours)) / (total_hours_worked - regular_hours) - regular_rate) / regular_rate * 100 = 75 :=
by
  sorry

end percentage_increase_in_overtime_rate_l61_61504


namespace ufo_convention_attendees_l61_61205

theorem ufo_convention_attendees (f m total : ℕ) 
  (h1 : m = 62) 
  (h2 : m = f + 4) : 
  total = 120 :=
by
  sorry

end ufo_convention_attendees_l61_61205


namespace circle_passing_three_points_l61_61705

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61705


namespace polygon_sides_l61_61369

theorem polygon_sides (n : ℕ) (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 156 = (180 * (n - 2)) / n) : n = 15 := sorry

end polygon_sides_l61_61369


namespace jenny_spent_180_minutes_on_bus_l61_61123

noncomputable def jennyBusTime : ℕ :=
  let timeAwayFromHome := 9 * 60  -- in minutes
  let classTime := 5 * 45  -- 5 classes each lasting 45 minutes
  let lunchTime := 45  -- in minutes
  let extracurricularTime := 90  -- 1 hour and 30 minutes
  timeAwayFromHome - (classTime + lunchTime + extracurricularTime)

theorem jenny_spent_180_minutes_on_bus : jennyBusTime = 180 :=
  by
  -- We need to prove that the total time Jenny was away from home minus time spent in school activities is 180 minutes.
  sorry  -- Proof to be completed.

end jenny_spent_180_minutes_on_bus_l61_61123


namespace probability_of_drawing_2_red_1_white_l61_61423

def total_balls : ℕ := 7
def red_balls : ℕ := 4
def white_balls : ℕ := 3
def draws : ℕ := 3

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_drawing_2_red_1_white :
  (combinations red_balls 2) * (combinations white_balls 1) / (combinations total_balls draws) = 18 / 35 := by
  sorry

end probability_of_drawing_2_red_1_white_l61_61423


namespace value_of_fraction_l61_61269

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l61_61269


namespace gcd_217_155_l61_61720

theorem gcd_217_155 : Nat.gcd 217 155 = 1 := by
  sorry

end gcd_217_155_l61_61720


namespace kite_diagonal_ratio_l61_61130

theorem kite_diagonal_ratio (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx1 : 0 ≤ x) (hx2 : x < a) (hy1 : 0 ≤ y) (hy2 : y < b)
  (orthogonal_diagonals : a^2 + y^2 = b^2 + x^2) :
  (a / b)^2 = 4 / 3 := 
sorry

end kite_diagonal_ratio_l61_61130


namespace circle_passing_three_points_l61_61707

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61707


namespace variable_is_eleven_l61_61571

theorem variable_is_eleven (x : ℕ) (h : (1/2)^22 * (1/81)^x = 1/(18^22)) : x = 11 :=
by
  sorry

end variable_is_eleven_l61_61571


namespace quadratic_monotonic_range_l61_61238

theorem quadratic_monotonic_range {a : ℝ} :
  (∀ x1 x2 : ℝ, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → (x1^2 - 2*a*x1 + 1) ≤ (x2^2 - 2*a*x2 + 1) ∨ (x1^2 - 2*a*x1 + 1) ≥ (x2^2 - 2*a*x2 + 1)) → (a ≤ 2 ∨ a ≥ 3) := 
sorry

end quadratic_monotonic_range_l61_61238


namespace mower_next_tangent_point_l61_61773

theorem mower_next_tangent_point (r_garden r_mower : ℝ) (h_garden : r_garden = 15) (h_mower : r_mower = 5) :
    ∃ θ : ℝ, θ = (2 * π * r_mower / (2 * π * r_garden)) * 360 ∧ θ = 120 :=
sorry

end mower_next_tangent_point_l61_61773


namespace face_value_of_share_l61_61752

theorem face_value_of_share (FV : ℝ) (market_value : ℝ) (dividend_rate : ℝ) (desired_return_rate : ℝ) 
  (H1 : market_value = 15) 
  (H2 : dividend_rate = 0.09) 
  (H3 : desired_return_rate = 0.12) 
  (H4 : dividend_rate * FV = desired_return_rate * market_value) :
  FV = 20 := 
by
  sorry

end face_value_of_share_l61_61752


namespace percentage_increase_in_yield_after_every_harvest_is_20_l61_61575

theorem percentage_increase_in_yield_after_every_harvest_is_20
  (P : ℝ)
  (h1 : ∀ n : ℕ, n = 1 → 20 * n = 20)
  (h2 : 20 + 20 * (1 + P / 100) = 44) :
  P = 20 := 
sorry

end percentage_increase_in_yield_after_every_harvest_is_20_l61_61575


namespace fred_red_marbles_l61_61769

theorem fred_red_marbles (total_marbles : ℕ) (dark_blue_marbles : ℕ) (green_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = 63) 
  (h2 : dark_blue_marbles ≥ total_marbles / 3)
  (h3 : green_marbles = 4)
  (h4 : red_marbles = total_marbles - dark_blue_marbles - green_marbles) : 
  red_marbles = 38 := 
sorry

end fred_red_marbles_l61_61769


namespace complex_number_z_l61_61823

theorem complex_number_z (z : ℂ) (h : (3 + 1 * I) * z = 4 - 2 * I) : z = 1 - I :=
by
  sorry

end complex_number_z_l61_61823


namespace real_roots_in_intervals_l61_61185

theorem real_roots_in_intervals (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x1 x2 : ℝ, (x1 = a / 3 ∨ x1 = -2 * b / 3) ∧ (x2 = a / 3 ∨ x2 = -2 * b / 3) ∧ x1 ≠ x2 ∧
  (a / 3 ≤ x1 ∧ x1 ≤ 2 * a / 3) ∧ (-2 * b / 3 ≤ x2 ∧ x2 ≤ -b / 3) ∧
  (x1 > 0 ∧ x2 < 0) ∧ (1 / x1 + 1 / (x1 - a) + 1 / (x1 + b) = 0) ∧
  (1 / x2 + 1 / (x2 - a) + 1 / (x2 + b) = 0) :=
sorry

end real_roots_in_intervals_l61_61185


namespace root_in_interval_l61_61721

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval : ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
  sorry

end root_in_interval_l61_61721


namespace find_circle_equation_l61_61709

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61709


namespace interesting_seven_digit_numbers_l61_61933

theorem interesting_seven_digit_numbers :
  ∃ n : Fin 2 → ℕ, (∀ i : Fin 2, n i = 128) :=
by sorry

end interesting_seven_digit_numbers_l61_61933


namespace compute_expr_l61_61084

theorem compute_expr : 6^2 - 4 * 5 + 2^2 = 20 := by
  sorry

end compute_expr_l61_61084


namespace number_of_x_values_l61_61260

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l61_61260


namespace exists_100_distinct_sums_l61_61214

theorem exists_100_distinct_sums : ∃ (a : Fin 100 → ℕ), (∀ i j k l : Fin 100, i ≠ j → k ≠ l → (i, j) ≠ (k, l) → a i + a j ≠ a k + a l) ∧ (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 25000) :=
by
  sorry

end exists_100_distinct_sums_l61_61214


namespace circle_passing_three_points_l61_61700

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61700


namespace product_of_c_values_l61_61338

theorem product_of_c_values :
  ∃ (c1 c2 : ℕ), (c1 > 0 ∧ c2 > 0) ∧
  (∃ (x1 x2 : ℚ), (7 * x1^2 + 15 * x1 + c1 = 0) ∧ (7 * x2^2 + 15 * x2 + c2 = 0)) ∧
  (c1 * c2 = 16) :=
sorry

end product_of_c_values_l61_61338


namespace inequality_am_gm_l61_61233

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (y * z) + y / (z * x) + z / (x * y)) ≥ (1 / x + 1 / y + 1 / z) := 
by
  sorry

end inequality_am_gm_l61_61233


namespace prob_two_sunny_days_l61_61463

-- Define the probability of rain and sunny
def probRain : ℚ := 3 / 4
def probSunny : ℚ := 1 / 4

-- Define the problem statement
theorem prob_two_sunny_days : (10 * (probSunny^2) * (probRain^3)) = 135 / 512 := 
by
  sorry

end prob_two_sunny_days_l61_61463


namespace total_people_museum_l61_61171

def bus1 := 12
def bus2 := 2 * bus1
def bus3 := bus2 - 6
def bus4 := bus1 + 9
def total := bus1 + bus2 + bus3 + bus4

theorem total_people_museum : total = 75 := by
  sorry

end total_people_museum_l61_61171


namespace function_intersects_all_lines_l61_61788

theorem function_intersects_all_lines :
  (∃ f : ℝ → ℝ, (∀ a : ℝ, ∃ y : ℝ, y = f a) ∧ (∀ k b : ℝ, ∃ x : ℝ, f x = k * x + b)) :=
sorry

end function_intersects_all_lines_l61_61788


namespace count_integer_values_l61_61253

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l61_61253


namespace find_const_functions_l61_61952

theorem find_const_functions
  (f g : ℝ → ℝ)
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → f (x^2 + y^2) = g (x * y)) :
  ∃ c : ℝ, (∀ x, 0 < x → f x = c) ∧ (∀ x, 0 < x → g x = c) :=
sorry

end find_const_functions_l61_61952


namespace danny_total_bottle_caps_l61_61383

def danny_initial_bottle_caps : ℕ := 37
def danny_found_bottle_caps : ℕ := 18

theorem danny_total_bottle_caps : danny_initial_bottle_caps + danny_found_bottle_caps = 55 := by
  sorry

end danny_total_bottle_caps_l61_61383


namespace bullet_train_speed_l61_61053

theorem bullet_train_speed 
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (speed_train2 : ℝ)
  (time_cross : ℝ)
  (combined_length : ℝ)
  (time_cross_hours : ℝ)
  (relative_speed : ℝ)
  (speed_train1 : ℝ) :
  length_train1 = 270 → 
  length_train2 = 230.04 →
  speed_train2 = 80 →
  time_cross = 9 →
  combined_length = (length_train1 + length_train2) / 1000 →
  time_cross_hours = time_cross / 3600 →
  relative_speed = combined_length / time_cross_hours →
  relative_speed = speed_train1 + speed_train2 →
  speed_train1 = 120.016 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bullet_train_speed_l61_61053


namespace cross_prod_correct_l61_61394

open Matrix

def vec1 : ℝ × ℝ × ℝ := (3, -1, 4)
def vec2 : ℝ × ℝ × ℝ := (-4, 6, 2)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
  a.2.2 * b.1 - a.1 * b.2.2,
  a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_prod_correct :
  cross_product vec1 vec2 = (-26, -22, 14) := by
  -- sorry is used to simplify the proof.
  sorry

end cross_prod_correct_l61_61394


namespace complex_quadrant_l61_61975

theorem complex_quadrant 
  (z : ℂ) 
  (h : (2 + 3 * Complex.I) * z = 1 + Complex.I) : 
  z.re > 0 ∧ z.im < 0 := 
sorry

end complex_quadrant_l61_61975


namespace mark_jump_rope_hours_l61_61878

theorem mark_jump_rope_hours 
    (record : ℕ := 54000)
    (jump_per_second : ℕ := 3)
    (seconds_per_hour : ℕ := 3600)
    (total_jumps_to_break_record : ℕ := 54001)
    (jumps_per_hour : ℕ := jump_per_second * seconds_per_hour) 
    (hours_needed : ℕ := total_jumps_to_break_record / jumps_per_hour) 
    (round_up : ℕ := if total_jumps_to_break_record % jumps_per_hour = 0 then hours_needed else hours_needed + 1) :
    round_up = 5 :=
sorry

end mark_jump_rope_hours_l61_61878


namespace integer_solutions_l61_61541

-- Define the polynomial equation as a predicate
def polynomial (n : ℤ) : Prop := n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0

-- The theorem statement
theorem integer_solutions :
  {n : ℤ | polynomial n} = {-1, 3} :=
by 
  sorry

end integer_solutions_l61_61541


namespace compound_interest_principal_l61_61916

theorem compound_interest_principal 
  (CI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hCI : CI = 315)
  (hR : R = 10)
  (hT : T = 2) :
  CI = P * ((1 + R / 100)^T - 1) → P = 1500 := by
  sorry

end compound_interest_principal_l61_61916


namespace correct_option_is_D_l61_61345

noncomputable def option_A := 230
noncomputable def option_B := [251, 260]
noncomputable def option_B_average := 256
noncomputable def option_C := [21, 212, 256]
noncomputable def option_C_average := 163
noncomputable def option_D := [210, 240, 250]
noncomputable def option_D_average := 233

theorem correct_option_is_D :
  ∃ (correct_option : String), correct_option = "D" :=
  sorry

end correct_option_is_D_l61_61345


namespace find_equation_of_line_l61_61237

open Real

noncomputable def equation_of_line : Prop :=
  ∃ c : ℝ, (∀ (x y : ℝ), (3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 → 2 * x + 3 * y + c = 0)) ∧
  ∃ x y : ℝ, 3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 ∧
              (2 * x + 3 * y + c = 0 → 6 * x + 9 * y - 7 = 0)

theorem find_equation_of_line : equation_of_line :=
sorry

end find_equation_of_line_l61_61237


namespace false_propositions_l61_61993

theorem false_propositions (p q : Prop) (hnp : ¬ p) (hq : q) :
  (¬ p) ∧ (¬ (p ∧ q)) ∧ (¬ ¬ q) :=
by {
  exact ⟨hnp, not_and_of_not_left q hnp, not_not_intro hq⟩
}

end false_propositions_l61_61993


namespace factorization_x12_minus_729_l61_61534

theorem factorization_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^3 - 3) * (x^3 + 3) :=
by sorry

end factorization_x12_minus_729_l61_61534


namespace maxwell_distance_when_meeting_l61_61922

theorem maxwell_distance_when_meeting 
  (distance_between_homes : ℕ)
  (maxwell_speed : ℕ) 
  (brad_speed : ℕ) 
  (total_distance : ℕ) 
  (h : distance_between_homes = 36) 
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 4) 
  (h3 : 6 * (total_distance / 6) = distance_between_homes) :
  total_distance = 12 :=
sorry

end maxwell_distance_when_meeting_l61_61922


namespace circle_passing_through_points_l61_61601

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61601


namespace percentage_invalid_votes_l61_61866

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end percentage_invalid_votes_l61_61866


namespace find_x_values_l61_61400

theorem find_x_values (x : ℝ) :
  x^3 - 9 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
by
  sorry

end find_x_values_l61_61400


namespace part1_part2_l61_61565

open Set Real

def f (x : ℝ) := sqrt (6 / (x + 1) - 1)

def A := { x : ℝ | -1 < x ∧ x ≤ 5 }

def B (m : ℝ) := { x : ℝ | -x^2 + m * x + 4 > 0 }

variable (m : ℝ)

theorem part1 (hm : m = 3) : A ∩ (B m)ᶜ = Icc 4 5 :=
by {
  rw hm,
  sorry
}

theorem part2 : ¬ ∃ m, B m ⊆ A :=
by {
  sorry
}

end part1_part2_l61_61565


namespace convert_binary_to_decimal_l61_61781

-- Define the binary number and its decimal conversion function
def bin_to_dec (bin : List ℕ) : ℕ :=
  list.foldl (λ acc b, 2 * acc + b) 0 bin

-- Define the specific problem conditions
def binary_oneonezeronethreeone : List ℕ := [1, 1, 0, 1, 1]

theorem convert_binary_to_decimal :
  bin_to_dec binary_oneonezeronethreeone = 27 := by
  sorry

end convert_binary_to_decimal_l61_61781


namespace ellipse_equation_hyperbola_equation_l61_61927

/-- Ellipse problem -/
def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_equation (e a c b : ℝ) (h_c : c = 3) (h_e : e = 0.5) (h_a : a = 6) (h_b : b^2 = 27) :
  ellipse_eq x y a b := 
sorry

/-- Hyperbola problem -/
def hyperbola_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_equation (a b c : ℝ) 
  (h_c : c = 6) 
  (h_A : ∀ (x y : ℝ), (x, y) = (-5, 2) → hyperbola_eq x y a b) 
  (h_eq1 : a^2 + b^2 = 36) 
  (h_eq2 : 25 / (a^2) - 4 / (b^2) = 1) :
  hyperbola_eq x y a b :=
sorry

end ellipse_equation_hyperbola_equation_l61_61927


namespace compute_expr_l61_61083

theorem compute_expr : 6^2 - 4 * 5 + 2^2 = 20 := by
  sorry

end compute_expr_l61_61083


namespace caleb_counted_right_angles_l61_61943

-- Definitions for conditions
def rectangular_park_angles : ℕ := 4
def square_field_angles : ℕ := 4
def total_angles (x y : ℕ) : ℕ := x + y

-- Theorem stating the problem
theorem caleb_counted_right_angles (h : total_angles rectangular_park_angles square_field_angles = 8) : 
   "type of anges Caleb counted" = "right angles" :=
sorry

end caleb_counted_right_angles_l61_61943


namespace two_vectors_less_than_45_deg_angle_l61_61287

theorem two_vectors_less_than_45_deg_angle (n : ℕ) (h : n = 30) (v : Fin n → ℝ → ℝ → ℝ) :
  ∃ i j : Fin n, i ≠ j ∧ ∃ θ : ℝ, θ < (45 * Real.pi / 180) :=
  sorry

end two_vectors_less_than_45_deg_angle_l61_61287


namespace equation_of_circle_through_three_points_l61_61681

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61681


namespace percentage_invalid_votes_l61_61865

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end percentage_invalid_votes_l61_61865


namespace exists_sum_coprime_seventeen_not_sum_coprime_l61_61453

/-- 
 For any integer \( n \) where \( n > 17 \), there exist integers \( a \) and \( b \) 
 such that \( n = a + b \), \( a > 1 \), \( b > 1 \), and \( \gcd(a, b) = 1 \).
 Additionally, the integer 17 does not have this property.
-/
theorem exists_sum_coprime (n : ℤ) (h : n > 17) : 
  ∃ (a b : ℤ), n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

/-- 
 The integer 17 cannot be expressed as the sum of two integers greater than 1 
 that are coprime.
-/
theorem seventeen_not_sum_coprime : 
  ¬ ∃ (a b : ℤ), 17 = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

end exists_sum_coprime_seventeen_not_sum_coprime_l61_61453


namespace divisible_by_6_and_sum_15_l61_61853

theorem divisible_by_6_and_sum_15 (A B : ℕ) (h1 : A + B = 15) (h2 : (10 * A + B) % 6 = 0) :
  (A * B = 56) ∨ (A * B = 54) :=
by sorry

end divisible_by_6_and_sum_15_l61_61853


namespace problem_statement_l61_61583

theorem problem_statement (a b m : ℕ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_m : m > 0) :
  (∃ n : ℕ, n > 0 ∧ m ∣ (a^n - 1) * b) ↔ (Nat.gcd (a * b) m = Nat.gcd b m) :=
by
  sorry

end problem_statement_l61_61583


namespace forest_area_relationship_l61_61323

variable (a b c x : ℝ)

theorem forest_area_relationship
    (hb : b = a * (1 + x))
    (hc : c = a * (1 + x) ^ 2) :
    a * c = b ^ 2 := by
  sorry

end forest_area_relationship_l61_61323


namespace complex_pure_imaginary_is_x_eq_2_l61_61854

theorem complex_pure_imaginary_is_x_eq_2
  (x : ℝ)
  (z : ℂ)
  (h : z = ⟨x^2 - 3 * x + 2, x - 1⟩)
  (pure_imaginary : z.re = 0) :
  x = 2 :=
by
  sorry

end complex_pure_imaginary_is_x_eq_2_l61_61854


namespace updated_mean_of_decrement_l61_61186

theorem updated_mean_of_decrement 
  (mean_initial : ℝ)
  (num_observations : ℕ)
  (decrement_per_observation : ℝ)
  (h1 : mean_initial = 200)
  (h2 : num_observations = 50)
  (h3 : decrement_per_observation = 6) : 
  (mean_initial * num_observations - decrement_per_observation * num_observations) / num_observations = 194 :=
by
  sorry

end updated_mean_of_decrement_l61_61186


namespace inequalities_hold_l61_61404

variables {a b c : ℝ}

theorem inequalities_hold (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := 
  by
    sorry

end inequalities_hold_l61_61404


namespace find_circle_equation_l61_61712

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61712


namespace gcd_372_684_is_12_l61_61039

theorem gcd_372_684_is_12 : gcd 372 684 = 12 := by
  sorry

end gcd_372_684_is_12_l61_61039


namespace gov_addresses_l61_61427

theorem gov_addresses (S H K : ℕ) 
  (H1 : S = 2 * H) 
  (H2 : K = S + 10) 
  (H3 : S + H + K = 40) : 
  S = 12 := 
sorry 

end gov_addresses_l61_61427


namespace tanya_total_sticks_l61_61148

theorem tanya_total_sticks (n : ℕ) (h : n = 11) : 3 * (n * (n + 1) / 2) = 198 :=
by
  have H : n = 11 := h
  sorry

end tanya_total_sticks_l61_61148


namespace flashes_in_fraction_of_hour_l61_61501

-- Definitions for the conditions
def flash_interval : ℕ := 6       -- The light flashes every 6 seconds
def hour_in_seconds : ℕ := 3600 -- There are 3600 seconds in an hour
def fraction_of_hour : ℚ := 3/4 -- ¾ of an hour

-- The translated proof problem statement in Lean
theorem flashes_in_fraction_of_hour (interval : ℕ) (sec_in_hour : ℕ) (fraction : ℚ) :
  interval = flash_interval →
  sec_in_hour = hour_in_seconds →
  fraction = fraction_of_hour →
  (fraction * sec_in_hour) / interval = 450 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end flashes_in_fraction_of_hour_l61_61501


namespace abs_eq_sqrt_five_l61_61836

theorem abs_eq_sqrt_five (x : ℝ) (h : |x| = Real.sqrt 5) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := 
sorry

end abs_eq_sqrt_five_l61_61836


namespace number_of_x_values_l61_61259

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l61_61259


namespace function_properties_l61_61821

-- Given conditions
def domain (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ ℝ

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = y^2 * f(x) + x^2 * f(y)

-- Lean 4 theorem statement
theorem function_properties (f : ℝ → ℝ)
  (Hdom : domain f)
  (Heqn : functional_eqn f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry


end function_properties_l61_61821


namespace sufficient_but_not_necessary_condition_l61_61560

noncomputable def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → (a - 1) * (a ^ x) < (a - 1) * (a ^ y) → a > 1) ∧
  (¬ (∀ c : ℝ, is_increasing_function (λ x => (c - 1) * (c ^ x)) → c > 1)) :=
sorry

end sufficient_but_not_necessary_condition_l61_61560


namespace absolute_value_of_difference_of_quadratic_roots_l61_61957
noncomputable theory

open Real

def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
let discr := b^2 - 4 * a * c in
((−b + sqrt discr) / (2 * a), (−b - sqrt discr) / (2 * a))

theorem absolute_value_of_difference_of_quadratic_roots :
  ∀ r1 r2 : ℝ, 
  r1^2 - 7 * r1 + 12 = 0 → r2^2 - 7 * r2 + 12 = 0 →
  abs (r1 - r2) = 5 :=
by
  sorry

end absolute_value_of_difference_of_quadratic_roots_l61_61957


namespace closest_fraction_l61_61395

theorem closest_fraction (n : ℤ) : 
  let frac1 := 37 / 57 
  let closest := 15 / 23
  n = 15 ∧ abs (851 - 57 * n) = min (abs (851 - 57 * 14)) (abs (851 - 57 * 15)) :=
by
  let frac1 := (37 : ℚ) / 57
  let closest := (15 : ℚ) / 23
  have h : 37 * 23 = 851 := by norm_num
  have denom : 57 * 23 = 1311 := by norm_num
  let num := 851
  sorry

end closest_fraction_l61_61395


namespace solve_problem_l61_61183

def num : ℕ := 1 * 3 * 5 * 7
def den : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7

theorem solve_problem : (num : ℚ) / den = 3.75 := 
by
  sorry

end solve_problem_l61_61183


namespace dacid_physics_marks_l61_61538

theorem dacid_physics_marks 
  (english : ℕ := 73)
  (math : ℕ := 69)
  (chem : ℕ := 64)
  (bio : ℕ := 82)
  (avg_marks : ℕ := 76)
  (num_subjects : ℕ := 5)
  : ∃ physics : ℕ, physics = 92 :=
by
  let total_marks := avg_marks * num_subjects
  let known_marks := english + math + chem + bio
  have physics := total_marks - known_marks
  use physics
  sorry

end dacid_physics_marks_l61_61538


namespace price_reduction_achieves_profit_l61_61505

theorem price_reduction_achieves_profit :
  ∃ x : ℝ, (40 - x) * (20 + 2 * (x / 4) * 8) = 1200 ∧ x = 20 :=
by
  sorry

end price_reduction_achieves_profit_l61_61505


namespace max_S_n_value_arithmetic_sequence_l61_61978

-- Definitions and conditions
def S_n (n : ℕ) : ℤ := 3 * n - n^2

def a_n (n : ℕ) : ℤ := 
if n = 0 then 0 else S_n n - S_n (n - 1)

-- Statement of the first part of the proof problem
theorem max_S_n_value (n : ℕ) (h : n = 1 ∨ n = 2) : S_n n = 2 :=
sorry

-- Statement of the second part of the proof problem
theorem arithmetic_sequence :
  ∀ n : ℕ, n ≥ 1 → a_n (n + 1) - a_n n = -2 :=
sorry

end max_S_n_value_arithmetic_sequence_l61_61978


namespace kayla_scores_on_sixth_level_l61_61303

-- Define the sequence of points scored in each level
def points (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 3
  | 2 => 5
  | 3 => 8
  | 4 => 12
  | n + 5 => points (n + 4) + (n + 1) + 1

-- Statement to prove that Kayla scores 17 points on the sixth level
theorem kayla_scores_on_sixth_level : points 5 = 17 :=
by
  sorry

end kayla_scores_on_sixth_level_l61_61303


namespace problem_l61_61976

variable (a : ℕ → ℝ) (n m : ℕ)

-- Condition: non-negative sequence and a_{n+m} ≤ a_n + a_m
axiom condition (n m : ℕ) : a n ≥ 0 ∧ a (n + m) ≤ a n + a m

-- Theorem: for any n ≥ m
theorem problem (h : n ≥ m) : a n ≤ m * a 1 + ((n / m) - 1) * a m :=
sorry

end problem_l61_61976


namespace no_such_polynomial_exists_l61_61386

theorem no_such_polynomial_exists :
  ∀ (P : ℤ → ℤ), (∃ a b c d : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
                  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4) → false :=
by
  sorry

end no_such_polynomial_exists_l61_61386


namespace kate_hair_length_l61_61297

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l61_61297


namespace vegan_non_soy_fraction_l61_61417

theorem vegan_non_soy_fraction (total_menu : ℕ) (vegan_dishes soy_free_vegan_dish : ℕ) 
  (h1 : vegan_dishes = 6) (h2 : vegan_dishes = total_menu / 3) (h3 : soy_free_vegan_dish = vegan_dishes - 5) :
  (soy_free_vegan_dish / total_menu = 1 / 18) :=
by
  sorry

end vegan_non_soy_fraction_l61_61417


namespace polygon_sides_l61_61370

theorem polygon_sides (n : ℕ) (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 156 = (180 * (n - 2)) / n) : n = 15 := sorry

end polygon_sides_l61_61370


namespace cube_iff_diagonal_perpendicular_l61_61813

-- Let's define the rectangular parallelepiped as a type
structure RectParallelepiped :=
-- Define the property of being a cube
(isCube : Prop)

-- Define the property q: any diagonal of the parallelepiped is perpendicular to the diagonal of its non-intersecting face
def diagonal_perpendicular (S : RectParallelepiped) : Prop := 
 sorry -- This depends on how you define diagonals and perpendicularity within the structure

-- Prove the biconditional relationship
theorem cube_iff_diagonal_perpendicular (S : RectParallelepiped) :
 S.isCube ↔ diagonal_perpendicular S :=
sorry

end cube_iff_diagonal_perpendicular_l61_61813


namespace sum_f_x₁_f_x₂_lt_0_l61_61384

variable (f : ℝ → ℝ)
variable (x₁ x₂ : ℝ)

-- Condition: y = f(x + 2) is an odd function
def odd_function_on_shifted_domain : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

-- Condition: f(x) is monotonically increasing for x > 2
def monotonically_increasing_for_x_gt_2 : Prop :=
  ∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Condition: x₁ + x₂ < 4
def sum_lt_4 : Prop :=
  x₁ + x₂ < 4

-- Condition: (x₁-2)(x₂-2) < 0
def product_shift_lt_0 : Prop :=
  (x₁ - 2) * (x₂ - 2) < 0

-- Main theorem to prove f(x₁) + f(x₂) < 0
theorem sum_f_x₁_f_x₂_lt_0
  (h1 : odd_function_on_shifted_domain f)
  (h2 : monotonically_increasing_for_x_gt_2 f)
  (h3 : sum_lt_4 x₁ x₂)
  (h4 : product_shift_lt_0 x₁ x₂) :
  f x₁ + f x₂ < 0 := sorry

end sum_f_x₁_f_x₂_lt_0_l61_61384


namespace solve_equation_l61_61333

theorem solve_equation (x : ℝ) : x^2 = 5 * x → x = 0 ∨ x = 5 := 
by
  sorry

end solve_equation_l61_61333


namespace solve_inequality_system_l61_61004

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61004


namespace min_value_fraction_l61_61104

theorem min_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 3) : 
  ∃ v, v = (x + y) / x ∧ v = -2 := 
by 
  sorry

end min_value_fraction_l61_61104


namespace alex_money_left_l61_61762

noncomputable def alex_main_income : ℝ := 900
noncomputable def alex_side_income : ℝ := 300
noncomputable def main_job_tax_rate : ℝ := 0.15
noncomputable def side_job_tax_rate : ℝ := 0.20
noncomputable def water_bill : ℝ := 75
noncomputable def main_job_tithe_rate : ℝ := 0.10
noncomputable def side_job_tithe_rate : ℝ := 0.15
noncomputable def grocery_expense : ℝ := 150
noncomputable def transportation_expense : ℝ := 50

theorem alex_money_left :
  let main_income_after_tax := alex_main_income * (1 - main_job_tax_rate)
  let side_income_after_tax := alex_side_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_income_after_tax + side_income_after_tax
  let main_tithe := alex_main_income * main_job_tithe_rate
  let side_tithe := alex_side_income * side_job_tithe_rate
  let total_tithe := main_tithe + side_tithe
  let total_deductions := water_bill + grocery_expense + transportation_expense + total_tithe
  let money_left := total_income_after_tax - total_deductions
  money_left = 595 :=
by
  -- Proof goes here
  sorry

end alex_money_left_l61_61762


namespace simplify_and_evaluate_expression_l61_61888

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m ^ 2 - 2 * m + 1) / m) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l61_61888


namespace lorie_total_bills_l61_61132

-- Definitions for the conditions
def initial_hundred_bills := 2
def hundred_to_fifty (bills : Nat) : Nat := bills * 2 / 100
def hundred_to_ten (bills : Nat) : Nat := (bills / 2) / 10
def hundred_to_five (bills : Nat) : Nat := (bills / 2) / 5

-- Statement of the problem
theorem lorie_total_bills : 
  let fifty_bills := hundred_to_fifty 100
  let ten_bills := hundred_to_ten 100
  let five_bills := hundred_to_five 100
  fifty_bills + ten_bills + five_bills = 2 + 5 + 10 :=
sorry

end lorie_total_bills_l61_61132


namespace nate_age_when_ember_14_l61_61949

theorem nate_age_when_ember_14 (nate_age : ℕ) (ember_age : ℕ) 
    (h1 : nate_age = 14) (h2 : ember_age = nate_age / 2) 
    (h3 : ember_age = 7) (h4 : ember_14_years_later : ℕ)
    (h : ember_14_years_later = ember_age + (14 - ember_age)) :
  ember_14_years_later = 14 → (nate_age + (14 - ember_age)) = 21 := by
  intros h_ember_14
  sorry

end nate_age_when_ember_14_l61_61949


namespace solve_inequality_system_l61_61001

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61001


namespace value_of_a_ab_b_l61_61719

-- Define conditions
variables {a b : ℝ} (h1 : a * b = 1) (h2 : b = a + 2)

-- The proof problem
theorem value_of_a_ab_b : a - a * b - b = -3 :=
by
  sorry

end value_of_a_ab_b_l61_61719


namespace simplify_and_evaluate_expression_l61_61889

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m ^ 2 - 2 * m + 1) / m) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l61_61889


namespace abs_diff_of_roots_l61_61955

theorem abs_diff_of_roots : 
  ∀ r1 r2 : ℝ, 
  (r1 + r2 = 7) ∧ (r1 * r2 = 12) → abs (r1 - r2) = 1 :=
by
  -- Assume the roots are r1 and r2
  intros r1 r2 H,
  -- Decompose the assumption H into its components
  cases H with Hsum Hprod,
  -- Calculate the square of the difference using the given identities
  have H_squared_diff : (r1 - r2)^2 = (r1 + r2)^2 - 4 * (r1 * r2),
  { sorry },
  -- Substitute the known values to find the square of the difference
  have H_squared_vals : (r1 - r2)^2 = 49 - 4 * 12,
  { sorry },
  -- Simplify to get (r1 - r2)^2 = 1
  have H1 : (r1 - r2)^2 = 1,
  { sorry },
  -- The absolute value of the difference is the square root of this result
  have abs_diff : abs (r1 - r2) = 1,
  { sorry },
  -- Conclude the proof by showing the final result matches the expected answer
  exact abs_diff

end abs_diff_of_roots_l61_61955


namespace gcd_factorial_8_10_l61_61968

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l61_61968


namespace geometric_sequence_third_term_l61_61751

theorem geometric_sequence_third_term (a₁ a₄ : ℕ) (r : ℕ) (h₁ : a₁ = 4) (h₂ : a₄ = 256) (h₃ : a₄ = a₁ * r^3) : a₁ * r^2 = 64 := 
by
  sorry

end geometric_sequence_third_term_l61_61751


namespace circle_through_points_l61_61661

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61661


namespace machine_tool_comparison_l61_61344

def defective_A : List ℕ := [1, 0, 2, 0, 2]
def defective_B : List ℕ := [1, 0, 1, 0, 3]

-- Calculate the total pairs for machine tool A
def total_pairs_A : List (ℕ × ℕ) :=
  [(1, 0), (1, 2), (1, 0), (1, 2), (0, 2), (0, 0), (0, 2), (2, 0), (2, 2), (0, 2)]

-- Calculate the number of pairs where the defective count is ≤ 1
def favorable_pairs_A : List (ℕ × ℕ) :=
  [(1, 0), (1, 0), (0, 0)]

-- Calculate mean of defective components
def mean (l : List ℕ) : ℝ :=
  (l.sum.toReal) / (l.length.toReal)

-- Calculate variance of defective components
def variance (l : List ℕ) : ℝ :=
  let m := mean l in
  (l.map (λ x => (x - m)^2)).sum.toReal / (l.length.toReal)

-- Prove the probability and variance comparison
theorem machine_tool_comparison :
  (favorable_pairs_A.length = 3 ∧ total_pairs_A.length = 10 ∧
   (favorable_pairs_A.length.toReal / total_pairs_A.length.toReal) = 3 / 10) ∧
  variance defective_A < variance defective_B :=
by
  sorry

end machine_tool_comparison_l61_61344


namespace gcd_lcm_of_45_and_150_l61_61325

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem gcd_lcm_of_45_and_150 :
  GCD 45 150 = 15 ∧ LCM 45 150 = 450 :=
by
  sorry

end gcd_lcm_of_45_and_150_l61_61325


namespace find_pairs_l61_61482

def regions_divided (h s : ℕ) : ℕ :=
  1 + s * (s + 1) / 2 + h * (s + 1)

theorem find_pairs (h s : ℕ) :
  regions_divided h s = 1992 →
  (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by
  sorry

end find_pairs_l61_61482


namespace julia_paid_for_puppy_l61_61582

theorem julia_paid_for_puppy :
  let dog_food := 20
  let treat := 2.5
  let treats := 2 * treat
  let toys := 15
  let crate := 20
  let bed := 20
  let collar_leash := 15
  let discount_rate := 0.20
  let total_before_discount := dog_food + treats + toys + crate + bed + collar_leash
  let discount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount
  let total_spent := 96
  total_spent - total_after_discount = 20 := 
by 
  sorry

end julia_paid_for_puppy_l61_61582


namespace factorization_x12_minus_729_l61_61533

theorem factorization_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^3 - 3) * (x^3 + 3) :=
by sorry

end factorization_x12_minus_729_l61_61533


namespace G_greater_F_l61_61098

theorem G_greater_F (x : ℝ) : 
  let F := 2*x^2 - 3*x - 2
  let G := 3*x^2 - 7*x + 5
  G > F := 
sorry

end G_greater_F_l61_61098


namespace general_term_arithmetic_sequence_l61_61822

theorem general_term_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (a1 : a 1 = -1) 
  (d : ℤ) 
  (h : d = 4) : 
  ∀ n : ℕ, a n = 4 * n - 5 :=
by
  sorry

end general_term_arithmetic_sequence_l61_61822


namespace max_n_for_neg_sum_correct_l61_61558

noncomputable def max_n_for_neg_sum (S : ℕ → ℤ) : ℕ :=
  if h₁ : S 19 > 0 then
    if h₂ : S 20 < 0 then
      11
    else 0  -- default value
  else 0  -- default value

theorem max_n_for_neg_sum_correct (S : ℕ → ℤ) (h₁ : S 19 > 0) (h₂ : S 20 < 0) : max_n_for_neg_sum S = 11 :=
by
  sorry

end max_n_for_neg_sum_correct_l61_61558


namespace value_of_fraction_l61_61271

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l61_61271


namespace solve_inequality_system_l61_61000

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61000


namespace mean_score_juniors_is_103_l61_61478

noncomputable def mean_score_juniors : Prop :=
  ∃ (students juniors non_juniors m_j m_nj : ℝ),
  students = 160 ∧
  (students * 82) = (juniors * m_j + non_juniors * m_nj) ∧
  juniors = 0.4 * non_juniors ∧
  m_j = 1.4 * m_nj ∧
  m_j = 103

theorem mean_score_juniors_is_103 : mean_score_juniors :=
by
  sorry

end mean_score_juniors_is_103_l61_61478


namespace flower_shop_options_l61_61309

theorem flower_shop_options:
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, 2 * p.1 + 3 * p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) ∧ S.card = 4 :=
sorry

end flower_shop_options_l61_61309


namespace probability_of_red_light_l61_61529

-- Definitions based on the conditions
def red_duration : ℕ := 30
def yellow_duration : ℕ := 5
def green_duration : ℕ := 40

def total_cycle_time : ℕ := red_duration + yellow_duration + green_duration

-- Statement of the problem to prove the probability of seeing red light
theorem probability_of_red_light : (red_duration : ℚ) / total_cycle_time = 2 / 5 := 
by sorry

end probability_of_red_light_l61_61529


namespace symmetric_line_equation_l61_61036

theorem symmetric_line_equation (x y : ℝ) :
  (∀ x y : ℝ, x - 3 * y + 5 = 0 ↔ 3 * x - y - 5 = 0) :=
by 
  sorry

end symmetric_line_equation_l61_61036


namespace max_dot_product_l61_61284

theorem max_dot_product (a b : ℝ) (h : a^2 + b^2 - a * b = 3) :
  (∀ CA CB : ℝ, (CA = a * b / 2) → CA * CB ≤ 3 / 2) :=
begin
  sorry
end

end max_dot_product_l61_61284


namespace radius_of_circle_l61_61891

theorem radius_of_circle:
  (∃ (r: ℝ), 
    (∀ (x: ℝ), (x^2 + r - x) = 0 → 1 - 4 * r = 0)
  ) → r = 1 / 4 := 
sorry

end radius_of_circle_l61_61891


namespace correct_statement_D_l61_61911

-- Conditions expressed as definitions
def candidates_selected_for_analysis : ℕ := 500

def statement_A : Prop := candidates_selected_for_analysis = 500
def statement_B : Prop := "The mathematics scores of the 500 candidates selected are the sample size."
def statement_C : Prop := "The 500 candidates selected are individuals."
def statement_D : Prop := "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

-- Problem statement in Lean
theorem correct_statement_D :
  statement_D := sorry

end correct_statement_D_l61_61911


namespace circle_through_points_l61_61673

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61673


namespace negation_of_p_is_neg_p_l61_61229

def p (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

def neg_p (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0

theorem negation_of_p_is_neg_p (f : ℝ → ℝ) : ¬ p f ↔ neg_p f :=
by
  sorry -- Proof of this theorem

end negation_of_p_is_neg_p_l61_61229


namespace circle_passing_through_points_l61_61596

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61596


namespace marble_distribution_correct_l61_61526

def num_ways_to_distribute_marbles : ℕ :=
  -- Given:
  -- Evan divides 100 marbles among three volunteers with each getting at least one marble
  -- Lewis selects a positive integer n > 1 and for each volunteer, steals exactly 1/n of marbles if possible.
  -- Prove that the number of ways to distribute the marbles such that Lewis cannot steal from all volunteers
  3540

theorem marble_distribution_correct :
  num_ways_to_distribute_marbles = 3540 :=
sorry

end marble_distribution_correct_l61_61526


namespace xy_square_diff_l61_61846

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61846


namespace intersection_M_N_l61_61107

open Set

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := sorry

end intersection_M_N_l61_61107


namespace evaluate_9_x_minus_1_l61_61419

theorem evaluate_9_x_minus_1 (x : ℝ) (h : (3 : ℝ)^(2 * x) = 16) : (9 : ℝ)^(x - 1) = 16 / 9 := by
  sorry

end evaluate_9_x_minus_1_l61_61419


namespace nate_age_when_ember_is_14_l61_61948

theorem nate_age_when_ember_is_14
  (nate_age : ℕ)
  (ember_age : ℕ)
  (h_half_age : ember_age = nate_age / 2)
  (h_nate_current_age : nate_age = 14) :
  nate_age + (14 - ember_age) = 21 :=
by
  sorry

end nate_age_when_ember_is_14_l61_61948


namespace tautology_a_tautology_b_tautology_c_tautology_d_l61_61525

variable (p q : Prop)

theorem tautology_a : p ∨ ¬ p := by
  sorry

theorem tautology_b : ¬ ¬ p ↔ p := by
  sorry

theorem tautology_c : ((p → q) → p) → p := by
  sorry

theorem tautology_d : ¬ (p ∧ ¬ p) := by
  sorry

end tautology_a_tautology_b_tautology_c_tautology_d_l61_61525


namespace neg_neg_one_eq_one_l61_61764

theorem neg_neg_one_eq_one : -(-1) = 1 :=
by
  sorry

end neg_neg_one_eq_one_l61_61764


namespace inequality_solution_l61_61014

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l61_61014


namespace ratio_netbooks_is_one_third_l61_61310

open Nat

def total_computers (total : ℕ) : Prop := total = 72
def laptops_sold (laptops : ℕ) (total : ℕ) : Prop := laptops = total / 2
def desktops_sold (desktops : ℕ) : Prop := desktops = 12
def netbooks_sold (netbooks : ℕ) (total laptops desktops : ℕ) : Prop :=
  netbooks = total - (laptops + desktops)
def ratio_netbooks_total (netbooks total : ℕ) : Prop :=
  netbooks * 3 = total

theorem ratio_netbooks_is_one_third
  (total laptops desktops netbooks : ℕ)
  (h_total : total_computers total)
  (h_laptops : laptops_sold laptops total)
  (h_desktops : desktops_sold desktops)
  (h_netbooks : netbooks_sold netbooks total laptops desktops) :
  ratio_netbooks_total netbooks total :=
by
  sorry

end ratio_netbooks_is_one_third_l61_61310


namespace count_integer_values_l61_61254

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l61_61254


namespace evaluate_expression_at_3_l61_61492

theorem evaluate_expression_at_3 :
  (∀ x ≠ 2, (x = 3) → (x^2 - 5 * x + 6) / (x - 2) = 0) :=
by
  sorry

end evaluate_expression_at_3_l61_61492


namespace ab_square_value_l61_61443

noncomputable def cyclic_quadrilateral (AX AY BX BY CX CY AB2 : ℝ) : Prop :=
  AX * AY = 6 ∧
  BX * BY = 5 ∧
  CX * CY = 4 ∧
  AB2 = 122 / 15

theorem ab_square_value :
  ∃ (AX AY BX BY CX CY : ℝ), cyclic_quadrilateral AX AY BX BY CX CY (122 / 15) :=
by
  sorry

end ab_square_value_l61_61443


namespace total_fish_correct_l61_61732

-- Define the number of pufferfish
def num_pufferfish : ℕ := 15

-- Define the number of swordfish as 5 times the number of pufferfish
def num_swordfish : ℕ := 5 * num_pufferfish

-- Define the total number of fish as the sum of pufferfish and swordfish
def total_num_fish : ℕ := num_pufferfish + num_swordfish

-- Theorem stating the total number of fish
theorem total_fish_correct : total_num_fish = 90 := by
  -- Proof is omitted
  sorry

end total_fish_correct_l61_61732


namespace negation_is_correct_l61_61168

-- Define the condition: we have two integers a and b
variables (a b : ℤ)

-- Original proposition: If the sum of two integers is even, then both integers are even.
def original_proposition := (a + b) % 2 = 0 → (a % 2 = 0) ∧ (b % 2 = 0)

-- Negation of the proposition: There exist two integers such that their sum is even and not both are even.
def negation_of_proposition := (a + b) % 2 = 0 ∧ ¬((a % 2 = 0) ∧ (b % 2 = 0))

theorem negation_is_correct :
  ¬ original_proposition a b = negation_of_proposition a b :=
by
  sorry

end negation_is_correct_l61_61168


namespace distinct_paths_to_B_and_C_l61_61055

def paths_to_red_arrows : ℕ × ℕ := (1, 2)
def paths_from_first_red_to_blue : ℕ := 3 * 2
def paths_from_second_red_to_blue : ℕ := 4 * 2
def total_paths_to_blue_arrows : ℕ := paths_from_first_red_to_blue + paths_from_second_red_to_blue

def paths_from_first_two_blue_to_green : ℕ := 5 * 4
def paths_from_third_and_fourth_blue_to_green : ℕ := 6 * 4
def total_paths_to_green_arrows : ℕ := paths_from_first_two_blue_to_green + paths_from_third_and_fourth_blue_to_green

def paths_to_B : ℕ := total_paths_to_green_arrows * 3
def paths_to_C : ℕ := total_paths_to_green_arrows * 4
def total_paths : ℕ := paths_to_B + paths_to_C

theorem distinct_paths_to_B_and_C :
  total_paths = 4312 := 
by
  -- all conditions can be used within this proof
  sorry

end distinct_paths_to_B_and_C_l61_61055


namespace correct_operation_is_B_l61_61921

-- Definitions of the operations as conditions
def operation_A (x : ℝ) : Prop := 3 * x - x = 3
def operation_B (x : ℝ) : Prop := x^2 * x^3 = x^5
def operation_C (x : ℝ) : Prop := x^6 / x^2 = x^3
def operation_D (x : ℝ) : Prop := (x^2)^3 = x^5

-- Prove that the correct operation is B
theorem correct_operation_is_B (x : ℝ) : operation_B x :=
by
  show x^2 * x^3 = x^5
  sorry

end correct_operation_is_B_l61_61921


namespace max_consecutive_sum_terms_l61_61900

theorem max_consecutive_sum_terms (S : ℤ) (n : ℕ) (H1 : S = 2015) (H2 : 0 < n) :
  (∃ a : ℤ, S = (a * n + (n * (n - 1)) / 2)) → n = 4030 :=
sorry

end max_consecutive_sum_terms_l61_61900


namespace johns_profit_l61_61293

noncomputable def profit_made 
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ) : ℕ :=
(trees_chopped * planks_per_tree / planks_per_table) * price_per_table - labor_cost

theorem johns_profit : profit_made 30 25 15 300 3000 = 12000 :=
by sorry

end johns_profit_l61_61293


namespace six_digit_number_count_correct_l61_61487

-- Defining the 6-digit number formation problem
def count_six_digit_numbers_with_conditions : Nat := 1560

-- Problem statement
theorem six_digit_number_count_correct :
  count_six_digit_numbers_with_conditions = 1560 :=
sorry

end six_digit_number_count_correct_l61_61487


namespace inequality_solution_l61_61010

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l61_61010


namespace rad_to_deg_eq_l61_61211

theorem rad_to_deg_eq : (4 / 3) * 180 = 240 := by
  sorry

end rad_to_deg_eq_l61_61211


namespace mary_money_after_purchase_l61_61201

def mary_initial_money : ℕ := 58
def pie_cost : ℕ := 6
def mary_friend_money : ℕ := 43  -- This is an extraneous condition, included for completeness.

theorem mary_money_after_purchase : mary_initial_money - pie_cost = 52 := by
  sorry

end mary_money_after_purchase_l61_61201


namespace abs_inequality_range_l61_61420

theorem abs_inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 6| > a) ↔ a < 5 :=
by
  sorry

end abs_inequality_range_l61_61420


namespace rounding_problem_l61_61454

def given_number : ℝ := 3967149.487234

theorem rounding_problem : (3967149.487234).round = 3967149 := sorry

end rounding_problem_l61_61454


namespace circle_through_points_l61_61664

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61664


namespace complement_of_A_l61_61992

def U : Set ℤ := {-1, 2, 4}
def A : Set ℤ := {-1, 4}

theorem complement_of_A : U \ A = {2} := by
  sorry

end complement_of_A_l61_61992


namespace gf_3_eq_495_l61_61127

def f (x : ℝ) : ℝ := x^2 + 4
def g (x : ℝ) : ℝ := 3 * x^2 - x + 1

theorem gf_3_eq_495 : g (f 3) = 495 := by
  sorry

end gf_3_eq_495_l61_61127


namespace circle_equation_l61_61644

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61644


namespace base_conversion_b_eq_3_l61_61894

theorem base_conversion_b_eq_3 (b : ℕ) (hb : b > 0) :
  (3 * 6^1 + 5 * 6^0 = 23) →
  (1 * b^2 + 3 * b + 2 = 23) →
  b = 3 :=
by {
  sorry
}

end base_conversion_b_eq_3_l61_61894


namespace number_square_25_l61_61902

theorem number_square_25 (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end number_square_25_l61_61902


namespace count_possible_integer_values_l61_61257

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l61_61257


namespace hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l61_61735

noncomputable def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

noncomputable def sum_first_n_odd_positive_integers (n : ℕ) : ℕ :=
  n * n

theorem hundredth_odd_integer_is_199 : nth_odd_positive_integer 100 = 199 :=
  by
  sorry

theorem sum_of_first_100_odd_integers_is_10000 : sum_first_n_odd_positive_integers 100 = 10000 :=
  by
  sorry

end hundredth_odd_integer_is_199_sum_of_first_100_odd_integers_is_10000_l61_61735


namespace james_prom_cost_l61_61122

def total_cost (ticket_cost dinner_cost tip_percent limo_cost_per_hour limo_hours tuxedo_cost persons : ℕ) : ℕ :=
  (ticket_cost * persons) +
  ((dinner_cost * persons) + (tip_percent * dinner_cost * persons) / 100) +
  (limo_cost_per_hour * limo_hours) + tuxedo_cost

theorem james_prom_cost :
  total_cost 100 120 30 80 8 150 4 = 1814 :=
by
  sorry

end james_prom_cost_l61_61122


namespace train_length_is_correct_l61_61935

noncomputable def train_length (speed_kmph : ℝ) (crossing_time_s : ℝ) (platform_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * crossing_time_s
  total_distance - platform_length_m

theorem train_length_is_correct :
  train_length 60 14.998800095992321 150 = 100 := by
  sorry

end train_length_is_correct_l61_61935


namespace cannon_hit_probability_l61_61196

theorem cannon_hit_probability {P2 P3 : ℝ} (hP1 : 0.5 <= P2) (hP2 : P2 = 0.2) (hP3 : P3 = 0.3) (h_none_hit : (1 - 0.5) * (1 - P2) * (1 - P3) = 0.28) :
  0.5 = 0.5 :=
by sorry

end cannon_hit_probability_l61_61196


namespace quotient_is_33_minus_G_l61_61225

variable (a b c d : ℕ)
variable (h_a : a < 10) (h_b : b < 10) (h_c : c < 10) (h_d : d < 10)
variable (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

def S : ℕ := a + b + c + d
def G : ℕ := Nat.gcd (Nat.gcd a b) (Nat.gcd c d)

theorem quotient_is_33_minus_G :
  (33 * S - G * S) / S = 33 - G :=
by
  dsimp [S, G]
  sorry

end quotient_is_33_minus_G_l61_61225


namespace intersection_eq_l61_61971

def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log x / Real.log 2 < 1}

theorem intersection_eq : {x : ℝ | x ∈ M ∧ x ∈ N} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l61_61971


namespace fish_to_corn_value_l61_61288

/-- In an island kingdom, five fish can be traded for three jars of honey, 
    and a jar of honey can be traded for six cobs of corn. 
    Prove that one fish is worth 3.6 cobs of corn. -/

theorem fish_to_corn_value (f h c : ℕ) (h1 : 5 * f = 3 * h) (h2 : h = 6 * c) : f = 18 * c / 5 := by
  sorry

end fish_to_corn_value_l61_61288


namespace inequality_solution_l61_61015

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l61_61015


namespace invalid_votes_percentage_l61_61864

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end invalid_votes_percentage_l61_61864


namespace solve_inequality_system_l61_61030

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61030


namespace cousins_rooms_distribution_l61_61139

theorem cousins_rooms_distribution : 
  (∑ n in ({ (5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1) } : finset (ℕ × ℕ × ℕ × ℕ)), 
    match n with 
    | (5,0,0,0) => 1
    | (4,1,0,0) => 5
    | (3,2,0,0) => 10 
    | (3,1,1,0) => 20 
    | (2,2,1,0) => 30 
    | (2,1,1,1) => 10 
    | _ => 0 
    end) = 76 := 
by 
  sorry

end cousins_rooms_distribution_l61_61139


namespace gcd_8251_6105_l61_61326

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 :=
by
  sorry

end gcd_8251_6105_l61_61326


namespace chickens_problem_l61_61032

theorem chickens_problem 
    (john_took_more_mary : ∀ (john mary : ℕ), john = mary + 5)
    (ray_took : ℕ := 10)
    (john_took_more_ray : ∀ (john ray : ℕ), john = ray + 11) :
    ∃ mary : ℕ, ray = mary - 6 :=
by
    sorry

end chickens_problem_l61_61032


namespace peanuts_in_box_l61_61574

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (h1 : initial_peanuts = 4) (h2 : added_peanuts = 2) : initial_peanuts + added_peanuts = 6 := by
  sorry

end peanuts_in_box_l61_61574


namespace value_of_fraction_l61_61268

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l61_61268


namespace solve_abs_eq_l61_61954

theorem solve_abs_eq : ∀ x : ℚ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) := by
  intros x
  sorry

end solve_abs_eq_l61_61954


namespace square_octagon_can_cover_ground_l61_61495

def square_interior_angle := 90
def octagon_interior_angle := 135

theorem square_octagon_can_cover_ground :
  square_interior_angle + 2 * octagon_interior_angle = 360 :=
by
  -- Proof skipped with sorry
  sorry

end square_octagon_can_cover_ground_l61_61495


namespace mary_avg_speed_round_trip_l61_61879

theorem mary_avg_speed_round_trip :
  let distance_to_school := 1.5 -- in km
  let time_to_school := 45 / 60 -- in hours (converted from minutes)
  let time_back_home := 15 / 60 -- in hours (converted from minutes)
  let total_distance := 2 * distance_to_school
  let total_time := time_to_school + time_back_home
  let avg_speed := total_distance / total_time
  avg_speed = 3 := by
  -- Definitions used directly appear in the conditions.
  -- Each condition used:
  -- Mary lives 1.5 km -> distance_to_school = 1.5
  -- Time to school 45 minutes -> time_to_school = 45 / 60
  -- Time back home 15 minutes -> time_back_home = 15 / 60
  -- Route is same -> total_distance = 2 * distance_to_school, total_time = time_to_school + time_back_home
  -- Proof to show avg_speed = 3
  sorry

end mary_avg_speed_round_trip_l61_61879


namespace perimeter_of_specific_figure_l61_61059

-- Define the grid size and additional column properties as given in the problem
structure Figure :=
  (rows : ℕ)
  (cols : ℕ)
  (additionalCols : ℕ)
  (additionalRows : ℕ)

-- The specific figure properties from the problem statement
def specificFigure : Figure := {
  rows := 3,
  cols := 4,
  additionalCols := 1,
  additionalRows := 2
}

-- Define the perimeter computation
def computePerimeter (fig : Figure) : ℕ :=
  2 * (fig.rows + fig.cols + fig.additionalCols) + fig.additionalRows

theorem perimeter_of_specific_figure : computePerimeter specificFigure = 13 :=
by
  sorry

end perimeter_of_specific_figure_l61_61059


namespace geometric_sequence_product_l61_61121

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = a 1 * (a 2 / a 1) ^ n)
  (h1 : a 1 * a 4 = -3) : a 2 * a 3 = -3 :=
by
  -- sorry is placed here to indicate the proof is not provided.
  sorry

end geometric_sequence_product_l61_61121


namespace markup_percentage_l61_61931

theorem markup_percentage (S M : ℝ) (h1 : S = 56 + M * S) (h2 : 0.80 * S - 56 = 8) : M = 0.30 :=
sorry

end markup_percentage_l61_61931


namespace binary_to_decimal_l61_61784

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l61_61784


namespace number_of_beetles_in_sixth_jar_l61_61354

theorem number_of_beetles_in_sixth_jar :
  ∃ (x : ℕ), 
      (x + (x+1) + (x+2) + (x+3) + (x+4) + (x+5) + (x+6) + (x+7) + (x+8) + (x+9) = 150) ∧
      (2 * x ≥ x + 9) ∧
      (x + 5 = 16) :=
by {
  -- This is just the statement, the proof steps are ommited.
  -- You can fill in the proof here using Lean tactics as necessary.
  sorry
}

end number_of_beetles_in_sixth_jar_l61_61354


namespace negation_proposition_l61_61475

theorem negation_proposition :
  (¬ (x ≠ 3 ∧ x ≠ 2) → ¬ (x ^ 2 - 5 * x + 6 ≠ 0)) =
  ((x = 3 ∨ x = 2) → (x ^ 2 - 5 * x + 6 = 0)) :=
by
  sorry

end negation_proposition_l61_61475


namespace sum_of_series_equals_negative_682_l61_61210

noncomputable def geometric_sum : ℤ :=
  let a := 2
  let r := -2
  let n := 10
  (a * (r ^ n - 1)) / (r - 1)

theorem sum_of_series_equals_negative_682 : geometric_sum = -682 := 
by sorry

end sum_of_series_equals_negative_682_l61_61210


namespace Murtha_pebbles_l61_61589

-- Definition of the geometric series sum formula
noncomputable def sum_geometric_series (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Constants for the problem
def a : ℕ := 1
def r : ℕ := 2
def n : ℕ := 10

-- The theorem to be proven
theorem Murtha_pebbles : sum_geometric_series a r n = 1023 :=
by
  -- Our condition setup implies the formula
  sorry

end Murtha_pebbles_l61_61589


namespace equation_of_circle_through_three_points_l61_61678

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61678


namespace find_a2_l61_61998

theorem find_a2 (f : ℤ → ℤ) (a : ℕ → ℤ) (x : ℤ) :
  (x^2 + (x + 1)^7 = a[0] + a[1] * (x + 2) + a[2] * (x + 2)^2 + a[3] * (x + 2)^3 + a[4] * (x + 2)^4 + a[5] * (x + 2)^5 + a[6] * (x + 2)^6 + a[7] * (x + 2)^7) →
  (a[2] = -20) :=
by sorry

end find_a2_l61_61998


namespace solve_inequality_system_l61_61005

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61005


namespace purchase_price_of_article_l61_61725

theorem purchase_price_of_article (P : ℝ) (h : 45 = 0.20 * P + 12) : P = 165 :=
by
  sorry

end purchase_price_of_article_l61_61725


namespace davonte_ran_further_than_mercedes_l61_61126

-- Conditions
variable (jonathan_distance : ℝ) (mercedes_distance : ℝ) (davonte_distance : ℝ)

-- Given conditions
def jonathan_ran := jonathan_distance = 7.5
def mercedes_ran_twice_jonathan := mercedes_distance = 2 * jonathan_distance
def mercedes_and_davonte_total := mercedes_distance + davonte_distance = 32

-- Prove the distance Davonte ran farther than Mercedes is 2 kilometers
theorem davonte_ran_further_than_mercedes :
  jonathan_ran jonathan_distance ∧
  mercedes_ran_twice_jonathan jonathan_distance mercedes_distance ∧
  mercedes_and_davonte_total mercedes_distance davonte_distance →
  davonte_distance - mercedes_distance = 2 :=
by
  sorry

end davonte_ran_further_than_mercedes_l61_61126


namespace greatest_A_satisfies_condition_l61_61216

theorem greatest_A_satisfies_condition :
  ∃ (A : ℝ), A = 64 ∧ ∀ (s : Fin₇ → ℝ), (∀ i, 1 ≤ s i ∧ s i ≤ A) →
  ∃ (i j : Fin₇), i ≠ j ∧ (1 / 2 ≤ s i / s j ∧ s i / s j ≤ 2) :=
by 
  sorry

end greatest_A_satisfies_condition_l61_61216


namespace students_called_back_l61_61341

theorem students_called_back (g b d t c : ℕ) (h1 : g = 9) (h2 : b = 14) (h3 : d = 21) (h4 : t = g + b) (h5 : c = t - d) : c = 2 := by 
  sorry

end students_called_back_l61_61341


namespace division_addition_problem_l61_61081

theorem division_addition_problem :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end division_addition_problem_l61_61081


namespace infinite_geometric_series_common_ratio_l61_61765

theorem infinite_geometric_series_common_ratio
  (a S : ℝ)
  (h₁ : a = 500)
  (h₂ : S = 4000)
  (h₃ : S = a / (1 - (r : ℝ))) :
  r = 7 / 8 :=
by
  sorry

end infinite_geometric_series_common_ratio_l61_61765


namespace parallelogram_fourth_vertex_distance_l61_61033

theorem parallelogram_fourth_vertex_distance (d1 d2 d3 d4 : ℝ) (h1 : d1 = 1) (h2 : d2 = 3) (h3 : d3 = 5) :
    d4 = 7 :=
sorry

end parallelogram_fourth_vertex_distance_l61_61033


namespace binomial_expansion_l61_61994

theorem binomial_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (1 + 2 * 1)^5 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 ∧
  (1 + 2 * -1)^5 = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 → 
  a_0 + a_2 + a_4 = 121 :=
by
  intro h
  let h₁ := h.1
  let h₂ := h.2
  sorry

end binomial_expansion_l61_61994


namespace circle_equation_correct_l61_61606

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61606


namespace circle_through_points_l61_61638

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l61_61638


namespace cone_volume_l61_61332

theorem cone_volume :
  ∀ (l h : ℝ) (r : ℝ), l = 15 ∧ h = 9 ∧ h = 3 * r → 
  (1 / 3) * Real.pi * r^2 * h = 27 * Real.pi :=
by
  intros l h r
  intro h_eqns
  sorry

end cone_volume_l61_61332


namespace identity_x_squared_minus_y_squared_l61_61837

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61837


namespace natural_eq_rational_exists_diff_l61_61184

-- Part (a)
theorem natural_eq (x y : ℕ) (h : x^3 + y = y^3 + x) : x = y := 
by sorry

-- Part (b)
theorem rational_exists_diff (x y : ℚ) (h : x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) : ∃ (x y : ℚ), x ≠ y ∧ x^3 + y = y^3 + x := 
by sorry

end natural_eq_rational_exists_diff_l61_61184


namespace arithmetic_sequence_first_term_l61_61304

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (k : ℕ) (hk : k ≥ 2)
  (hS : S k = 5)
  (ha_k2_p1 : a (k^2 + 1) = -45)
  (ha_sum : (Finset.range (2 * k + 1) \ Finset.range (k + 1)).sum a = -45) :
  a 1 = 5 := 
sorry

end arithmetic_sequence_first_term_l61_61304


namespace circle_passing_through_points_l61_61635

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61635


namespace circle_passes_through_points_l61_61622

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61622


namespace cousins_in_rooms_l61_61133

theorem cousins_in_rooms : 
  (number_of_ways : ℕ) (cousins : ℕ) (rooms : ℕ)
  (ways : ℕ) (is_valid_distribution : (ℕ → ℕ))
  (h_cousins : cousins = 5)
  (h_rooms : rooms = 4)
  (h_number_of_ways : ways = 67)
  :
  ∃ (distribute : ℕ → ℕ → ℕ), distribute cousins rooms = ways :=
sorry

end cousins_in_rooms_l61_61133


namespace hyperbola_condition_l61_61988

theorem hyperbola_condition (m : ℝ) :
  (∃ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) → m < 0 ∨ m > 2 :=
sorry

end hyperbola_condition_l61_61988


namespace circle_equation_through_points_l61_61692

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61692


namespace gcd_factorial_l61_61964

-- Definitions and conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Theorem statement
theorem gcd_factorial : gcd (factorial 8) (factorial 10) = factorial 8 := by
  -- The proof is omitted
  sorry

end gcd_factorial_l61_61964


namespace circle_equation_l61_61650

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61650


namespace max_value_of_s_l61_61585

-- Define the conditions
variables (p q r s : ℝ)

-- Add assumptions
axiom h1 : p + q + r + s = 10
axiom h2 : p * q + p * r + p * s + q * r + q * s + r * s = 20

-- State the theorem
theorem max_value_of_s : s ≤ (5 + real.sqrt 105) / 2 :=
sorry

end max_value_of_s_l61_61585


namespace determine_z_l61_61898

theorem determine_z (z : ℕ) (h1: z.factors.count = 18) (h2: 16 ∣ z) (h3: 18 ∣ z) : z = 288 := 
  by 
  sorry

end determine_z_l61_61898


namespace delta_evaluation_l61_61785

def delta (a b : ℕ) : ℕ := a^3 - b

theorem delta_evaluation :
  delta (2^(delta 3 8)) (5^(delta 4 9)) = 2^19 - 5^55 := 
sorry

end delta_evaluation_l61_61785


namespace repeating_decimals_sum_is_fraction_l61_61744

-- Define the repeating decimals as fractions
def x : ℚ := 1 / 3
def y : ℚ := 2 / 99

-- Define the sum of the repeating decimals
def sum := x + y

-- State the theorem
theorem repeating_decimals_sum_is_fraction :
  sum = 35 / 99 := sorry

end repeating_decimals_sum_is_fraction_l61_61744


namespace final_answer_l61_61820

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l61_61820


namespace neighbors_have_even_total_bells_not_always_divisible_by_3_l61_61479

def num_bushes : ℕ := 19

def is_neighbor (circ : ℕ → ℕ) (i j : ℕ) : Prop := 
  if i = num_bushes - 1 then j = 0
  else j = i + 1

-- Part (a)
theorem neighbors_have_even_total_bells (bells : Fin num_bushes → ℕ) :
  ∃ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 2 = 0 := sorry

-- Part (b)
theorem not_always_divisible_by_3 (bells : Fin num_bushes → ℕ) :
  ¬ (∀ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 3 = 0) := sorry

end neighbors_have_even_total_bells_not_always_divisible_by_3_l61_61479


namespace find_number_l61_61572

theorem find_number :
  ∃ x : Int, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 :=
by
  sorry

end find_number_l61_61572


namespace gain_percentage_l61_61455

theorem gain_percentage (MP CP : ℝ) (h1 : 0.90 * MP = 1.17 * CP) :
  (((MP - CP) / CP) * 100) = 30 := 
by
  sorry

end gain_percentage_l61_61455


namespace total_people_museum_l61_61173

-- Conditions
def first_bus_people : ℕ := 12
def second_bus_people := 2 * first_bus_people
def third_bus_people := second_bus_people - 6
def fourth_bus_people := first_bus_people + 9

-- Question to prove
theorem total_people_museum : first_bus_people + second_bus_people + third_bus_people + fourth_bus_people = 75 :=
by
  -- The proof is skipped but required to complete the theorem
  sorry

end total_people_museum_l61_61173


namespace geometric_sequence_min_value_l61_61875

theorem geometric_sequence_min_value (r : ℝ) (a1 a2 a3 : ℝ) 
  (h1 : a1 = 1) 
  (h2 : a2 = a1 * r) 
  (h3 : a3 = a2 * r) :
  4 * a2 + 5 * a3 ≥ -(4 / 5) :=
by
  sorry

end geometric_sequence_min_value_l61_61875


namespace equation_of_circle_ABC_l61_61617

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61617


namespace encryption_of_hope_is_correct_l61_61215

def shift_letter (c : Char) : Char :=
  if 'a' ≤ c ∧ c ≤ 'z' then
    Char.ofNat ((c.toNat - 'a'.toNat + 4) % 26 + 'a'.toNat)
  else 
    c

def encrypt (s : String) : String :=
  s.map shift_letter

theorem encryption_of_hope_is_correct : encrypt "hope" = "lsti" :=
by
  sorry

end encryption_of_hope_is_correct_l61_61215


namespace pencils_total_l61_61042

/-- The students in class 5A had a total of 2015 pencils. One of them lost a box containing five pencils and replaced it with a box containing 50 pencils. Prove the final number of pencils is 2060. -/
theorem pencils_total {initial_pencils lost_pencils gained_pencils final_pencils : ℕ} 
  (h1 : initial_pencils = 2015) 
  (h2 : lost_pencils = 5) 
  (h3 : gained_pencils = 50) 
  (h4 : final_pencils = (initial_pencils - lost_pencils + gained_pencils)) 
  : final_pencils = 2060 :=
sorry

end pencils_total_l61_61042


namespace equation_of_circle_through_three_points_l61_61682

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61682


namespace circle_equation_correct_l61_61609

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61609


namespace range_of_a_l61_61830

theorem range_of_a (a : ℝ) 
  (h : ∀ x y, (a * x^2 - 3 * x + 2 = 0) ∧ (a * y^2 - 3 * y + 2 = 0) → x = y) :
  a = 0 ∨ a ≥ 9/8 :=
sorry

end range_of_a_l61_61830


namespace original_prices_l61_61145

theorem original_prices 
  (S P J : ℝ)
  (hS : 0.80 * S = 780)
  (hP : 0.70 * P = 2100)
  (hJ : 0.90 * J = 2700) :
  S = 975 ∧ P = 3000 ∧ J = 3000 :=
by
  sorry

end original_prices_l61_61145


namespace circle_passing_through_points_l61_61631

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61631


namespace circle_through_points_l61_61674

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61674


namespace cost_equivalence_l61_61378

theorem cost_equivalence (b a p : ℕ) (h1 : 4 * b = 3 * a) (h2 : 9 * a = 6 * p) : 24 * b = 12 * p :=
  sorry

end cost_equivalence_l61_61378


namespace number_of_two_bedroom_units_l61_61191

-- Definitions based on the conditions
def is_solution (x y : ℕ) : Prop :=
  (x + y = 12) ∧ (360 * x + 450 * y = 4950)

theorem number_of_two_bedroom_units : ∃ y : ℕ, is_solution (12 - y) y ∧ y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l61_61191


namespace parabola_intercepts_l61_61385

def parabola_eqn : ℝ → ℝ := λ y, -3 * y^2 + 2 * y + 3

theorem parabola_intercepts :
  ((∀ y, parabola_eqn 0 = 3) ∧
   ∃ y1 y2 : ℝ, parabola_eqn y1 = 0 ∧ parabola_eqn y2 = 0 ∧ y1 ≠ y2) :=
by
  use [(-(1 / 3)) * (1 + Real.sqrt 10), (-(1 / 3)) * (1 - Real.sqrt 10)]
  simp [parabola_eqn]
  sorry

end parabola_intercepts_l61_61385


namespace circle_equation_through_points_l61_61693

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61693


namespace circle_passing_through_points_eqn_l61_61656

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61656


namespace line_tangent_to_parabola_l61_61095

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 7 * y + k = 0 ↔ y^2 = 16 * x) → k = 49 :=
by
  sorry

end line_tangent_to_parabola_l61_61095


namespace cost_of_each_top_l61_61073

theorem cost_of_each_top
  (total_spent : ℝ)
  (num_shorts : ℕ)
  (price_per_short : ℝ)
  (num_shoes : ℕ)
  (price_per_shoe : ℝ)
  (num_tops : ℕ)
  (total_cost_shorts : ℝ)
  (total_cost_shoes : ℝ)
  (amount_spent_on_tops : ℝ)
  (cost_per_top : ℝ) :
  total_spent = 75 →
  num_shorts = 5 →
  price_per_short = 7 →
  num_shoes = 2 →
  price_per_shoe = 10 →
  num_tops = 4 →
  total_cost_shorts = num_shorts * price_per_short →
  total_cost_shoes = num_shoes * price_per_shoe →
  amount_spent_on_tops = total_spent - (total_cost_shorts + total_cost_shoes) →
  cost_per_top = amount_spent_on_tops / num_tops →
  cost_per_top = 5 :=
by
  sorry

end cost_of_each_top_l61_61073


namespace prove_a2_l61_61406

def arithmetic_seq (a d : ℕ → ℝ) : Prop :=
  ∀ n m, a n + d (n - m) = a m

theorem prove_a2 (a : ℕ → ℝ) (d : ℕ → ℝ) :
  (∀ n, a n = a 0 + (n - 1) * 2) → 
  (a 1 + 4) / a 1 = (a 1 + 6) / (a 1 + 4) →
  (d 1 = 2) →
  a 2 = -6 :=
by
  intros h_seq h_geo h_common_diff
  sorry

end prove_a2_l61_61406


namespace number_of_ways_to_fold_cube_with_one_face_missing_l61_61774

-- Definitions:
-- The polygon is initially in the shape of a cross with 5 congruent squares.
-- One additional square can be attached to any of the 12 possible edge positions around this polygon.
-- Define what it means for the resulting figure to fold into a cube with one face missing.

-- Statement:
theorem number_of_ways_to_fold_cube_with_one_face_missing 
  (initial_squares : ℕ)
  (additional_positions : ℕ)
  (valid_folding_positions : ℕ) : 
  initial_squares = 5 ∧ additional_positions = 12 → valid_folding_positions = 8 :=
by
  sorry

end number_of_ways_to_fold_cube_with_one_face_missing_l61_61774


namespace probability_pair_tile_l61_61390

def letters_in_word : List Char := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
def target_letters : List Char := ['P', 'A', 'I', 'R']

def num_favorable_outcomes : Nat :=
  -- Count occurrences of letters in target_letters within letters_in_word
  List.count 'P' letters_in_word +
  List.count 'A' letters_in_word +
  List.count 'I' letters_in_word + 
  List.count 'R' letters_in_word

def total_outcomes : Nat := letters_in_word.length

theorem probability_pair_tile :
  (num_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

end probability_pair_tile_l61_61390


namespace range_of_m_l61_61812

def proposition_p (m : ℝ) : Prop :=
  ∀ x > 0, m^2 + 2 * m - 1 ≤ x + 1 / x

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (5 - m^2) ^ x > (5 - m^2) ^ (x - 1)

theorem range_of_m (m : ℝ) : (proposition_p m ∨ proposition_q m) ∧ ¬ (proposition_p m ∧ proposition_q m) ↔ (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l61_61812


namespace inverse_ratio_l61_61722

theorem inverse_ratio (a b c d : ℝ) :
  (∀ x, x ≠ -6 → (3 * x - 2) / (x + 6) = (a * x + b) / (c * x + d)) →
  a/c = -6 :=
by
  sorry

end inverse_ratio_l61_61722


namespace circle_passes_through_points_l61_61621

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61621


namespace equation_of_circle_passing_through_points_l61_61690

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61690


namespace equation_of_circle_through_three_points_l61_61683

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61683


namespace ratio_boys_girls_l61_61120

variable (S G : ℕ)

theorem ratio_boys_girls (h : (2 / 3 : ℚ) * G = (1 / 5 : ℚ) * S) :
  (S - G) * 3 = 7 * G := by
  -- Proof goes here
  sorry

end ratio_boys_girls_l61_61120


namespace area_of_triangle_l61_61520

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the points of intercepts with axes
def intercepts (x0 y0 x1 y1 : ℝ) : Prop :=
  line_eq 0 y0 ∧ y0 = 9 ∧
  line_eq x1 0 ∧ x1 = 3

-- Define the area of the triangle
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem area_of_triangle : ∃ (x0 y0 x1 y1 : ℝ), 
  intercepts x0 y0 x1 y1 ∧ triangle_area x1 y0 = 13.5 :=
by
  sorry

end area_of_triangle_l61_61520


namespace expand_expression_l61_61950

variable {x y z : ℝ}

theorem expand_expression :
  (2 * x + 5) * (3 * y + 15 + 4 * z) = 6 * x * y + 30 * x + 8 * x * z + 15 * y + 20 * z + 75 :=
by
  sorry

end expand_expression_l61_61950


namespace charlie_more_apples_than_bella_l61_61531

variable (D : ℝ) 

theorem charlie_more_apples_than_bella 
    (hC : C = 1.75 * D)
    (hB : B = 1.50 * D) :
    (C - B) / B = 0.1667 := 
by
  sorry

end charlie_more_apples_than_bella_l61_61531


namespace find_triples_l61_61393

theorem find_triples (a b c : ℝ) 
  (h1 : a = (b + c) ^ 2) 
  (h2 : b = (a + c) ^ 2) 
  (h3 : c = (a + b) ^ 2) : 
  (a = 0 ∧ b = 0 ∧ c = 0) 
  ∨ 
  (a = 1/4 ∧ b = 1/4 ∧ c = 1/4) :=
  sorry

end find_triples_l61_61393


namespace july14_2030_is_sunday_l61_61296

-- Define the given condition that July 3, 2030 is a Wednesday. 
def july3_2030_is_wednesday : Prop := true -- Assume the existence and correctness of this statement.

-- Define the proof problem that July 14, 2030 is a Sunday given the above condition.
theorem july14_2030_is_sunday : july3_2030_is_wednesday → (14 % 7 = 0) := 
sorry

end july14_2030_is_sunday_l61_61296


namespace number_of_plain_lemonade_sold_l61_61939

theorem number_of_plain_lemonade_sold
  (price_per_plain_lemonade : ℝ)
  (earnings_strawberry_lemonade : ℝ)
  (earnings_more_plain_than_strawberry : ℝ)
  (P : ℝ)
  (H1 : price_per_plain_lemonade = 0.75)
  (H2 : earnings_strawberry_lemonade = 16)
  (H3 : earnings_more_plain_than_strawberry = 11)
  (H4 : price_per_plain_lemonade * P = earnings_strawberry_lemonade + earnings_more_plain_than_strawberry) :
  P = 36 :=
by
  sorry

end number_of_plain_lemonade_sold_l61_61939


namespace f_even_l61_61037

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_const : ¬ (∀ x y : ℝ, f x = f y)
axiom f_equiv1 : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_equiv2 : ∀ x : ℝ, f (1 + x) = -f x

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  sorry

end f_even_l61_61037


namespace solution_set_l61_61808

noncomputable def f : ℝ → ℝ := sorry
def dom := {x : ℝ | x < 0 ∨ x > 0 } -- Definition of the function domain

-- Assumptions and conditions as definitions in Lean
axiom f_odd : ∀ x ∈ dom, f (-x) = -f x
axiom f_at_1 : f 1 = 1
axiom symmetric_f : ∀ x ∈ dom, (f (x + 1)) = -f (-x + 1)
axiom inequality_condition : ∀ (x1 x2 : ℝ), x1 ∈ dom → x2 ∈ dom → x1 ≠ x2 → (x1^3 * f x1 - x2^3 * f x2) / (x1 - x2) > 0

-- The main statement to be proved
theorem solution_set :
  {x ∈ dom | f x ≤ 1 / x^3} = {x ∈ dom | x ≤ -1} ∪ {x ∈ dom | 0 < x ∧ x ≤ 1} :=
sorry

end solution_set_l61_61808


namespace correct_option_l61_61738

-- Definitions based on the conditions of the problem
def exprA (a : ℝ) : Prop := 7 * a + a = 7 * a^2
def exprB (x y : ℝ) : Prop := 3 * x^2 * y - 2 * x^2 * y = x^2 * y
def exprC (y : ℝ) : Prop := 5 * y - 3 * y = 2
def exprD (a b : ℝ) : Prop := 3 * a + 2 * b = 5 * a * b

-- Proof problem statement verifying the correctness of the given expressions
theorem correct_option (x y : ℝ) : exprB x y :=
by
  -- (No proof is required, the statement is sufficient)
  sorry

end correct_option_l61_61738


namespace circle_passing_through_points_eqn_l61_61659

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61659


namespace hyperbola_asymptote_m_value_l61_61984

theorem hyperbola_asymptote_m_value
  (m : ℝ)
  (h1 : m > 0)
  (h2 : ∀ x y : ℝ, (5 * x - 2 * y = 0) → ((x^2 / 4) - (y^2 / m^2) = 1)) :
  m = 5 :=
sorry

end hyperbola_asymptote_m_value_l61_61984


namespace equation_of_circle_ABC_l61_61615

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61615


namespace distance_between_foci_of_ellipse_l61_61803

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 21 :=
by
  sorry

end distance_between_foci_of_ellipse_l61_61803


namespace circle_equation_through_points_l61_61694

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61694


namespace find_a_plus_b_l61_61231

-- Given conditions
variable (a b : ℝ)

-- The imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Condition equation
def equation := (a + i) * i = b - 2 * i

-- Define the lean statement
theorem find_a_plus_b (h : equation a b) : a + b = -3 :=
by sorry

end find_a_plus_b_l61_61231


namespace face_value_of_share_l61_61754

theorem face_value_of_share 
  (F : ℝ)
  (dividend_rate : ℝ := 0.09)
  (desired_return_rate : ℝ := 0.12)
  (market_value : ℝ := 15) 
  (h_eq : (dividend_rate * F) / market_value = desired_return_rate) : 
  F = 20 := 
by 
  sorry

end face_value_of_share_l61_61754


namespace line_ellipse_tangent_l61_61802

theorem line_ellipse_tangent (m : ℝ) : 
  (∀ x y : ℝ, (y = m * x + 2) → (x^2 + (y^2 / 4) = 1)) → m^2 = 0 :=
sorry

end line_ellipse_tangent_l61_61802


namespace geometric_seq_condition_l61_61415

variable (n : ℕ) (a : ℕ → ℝ)

-- The definition of a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n + 1) = a n * a (n + 2)

-- The main theorem statement
theorem geometric_seq_condition :
  (is_geometric_seq a n → ∀ n, |a n| ≥ 0) →
  ∃ (a : ℕ → ℝ), (∀ n, a n * a (n + 2) = a (n + 1) * a (n + 1)) →
  (∀ m, a m = 0 → ¬(is_geometric_seq a n)) :=
sorry

end geometric_seq_condition_l61_61415


namespace polygon_sides_l61_61367

theorem polygon_sides (n : ℕ) (h₁ : ∀ (m : ℕ), m = n → n > 2) (h₂ : 180 * (n - 2) = 156 * n) : n = 15 :=
by
  sorry

end polygon_sides_l61_61367


namespace arithmetic_sequence_general_formula_l61_61203

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h₁ : a 1 = 39) (h₂ : a 1 + a 3 = 74) : 
  ∀ n, a n = 41 - 2 * n :=
sorry

end arithmetic_sequence_general_formula_l61_61203


namespace circle_equation_correct_l61_61607

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61607


namespace circle_through_points_l61_61668

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61668


namespace totalCorrectQuestions_l61_61451

-- Definitions for the conditions
def mathQuestions : ℕ := 40
def mathCorrectPercentage : ℕ := 75
def englishQuestions : ℕ := 50
def englishCorrectPercentage : ℕ := 98

-- Function to calculate the number of correctly answered questions
def correctQuestions (totalQuestions : ℕ) (percentage : ℕ) : ℕ :=
  (percentage * totalQuestions) / 100

-- Main theorem to prove the total number of correct questions
theorem totalCorrectQuestions : 
  correctQuestions mathQuestions mathCorrectPercentage +
  correctQuestions englishQuestions englishCorrectPercentage = 79 :=
by
  sorry

end totalCorrectQuestions_l61_61451


namespace total_number_of_slices_l61_61490

def number_of_pizzas : ℕ := 7
def slices_per_pizza : ℕ := 2

theorem total_number_of_slices :
  number_of_pizzas * slices_per_pizza = 14 :=
by
  sorry

end total_number_of_slices_l61_61490


namespace number_of_friends_l61_61460

def total_gold := 100
def lost_gold := 20
def gold_per_friend := 20

theorem number_of_friends :
  (total_gold - lost_gold) / gold_per_friend = 4 := by
  sorry

end number_of_friends_l61_61460


namespace cousin_distribution_count_l61_61142

-- Definition of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Definition to count the number of distributions
noncomputable def count_cousin_distributions : ℕ :=
  let case1 := 1 in -- (5,0,0,0)
  let case2 := choose 5 1 in -- (4,1,0,0)
  let case3 := choose 5 3 in -- (3,2,0,0)
  let case4 := choose 5 3 in -- (3,1,1,0)
  let case5 := choose 5 2 * choose 3 2 in -- (2,2,1,0)
  let case6 := choose 5 2 in -- (2,1,1,1)
  case1 + case2 + case3 + case4 + case5 + case6

-- Theorem to prove
theorem cousin_distribution_count : count_cousin_distributions = 66 := by
  sorry

end cousin_distribution_count_l61_61142


namespace arithmetic_mean_after_removal_l61_61068

theorem arithmetic_mean_after_removal 
  (mean_original : ℝ) (num_original : ℕ) 
  (nums_removed : List ℝ) (mean_new : ℝ)
  (h1 : mean_original = 50) 
  (h2 : num_original = 60) 
  (h3 : nums_removed = [60, 65, 70, 40]) 
  (h4 : mean_new = 49.38) :
  let sum_original := mean_original * num_original
  let num_remaining := num_original - nums_removed.length
  let sum_removed := List.sum nums_removed
  let sum_new := sum_original - sum_removed
  
  mean_new = sum_new / num_remaining :=
sorry

end arithmetic_mean_after_removal_l61_61068


namespace power_sum_inequality_l61_61500

theorem power_sum_inequality (k l m : ℕ) : 
  2 ^ (k + l) + 2 ^ (k + m) + 2 ^ (l + m) ≤ 2 ^ (k + l + m + 1) + 1 := 
by 
  sorry

end power_sum_inequality_l61_61500


namespace income_to_expenditure_ratio_l61_61897

variable (I E S : ℕ)

def Ratio (a b : ℕ) : ℚ := a / (b : ℚ)

theorem income_to_expenditure_ratio (h1 : I = 14000) (h2 : S = 2000) (h3 : S = I - E) : 
  Ratio I E = 7 / 6 :=
by
  sorry

end income_to_expenditure_ratio_l61_61897


namespace current_population_correct_l61_61578

def initial_population : ℕ := 4079
def percentage_died : ℕ := 5
def percentage_left : ℕ := 15

def calculate_current_population (initial_population : ℕ) (percentage_died : ℕ) (percentage_left : ℕ) : ℕ :=
  let died := (initial_population * percentage_died) / 100
  let remaining_after_bombardment := initial_population - died
  let left := (remaining_after_bombardment * percentage_left) / 100
  remaining_after_bombardment - left

theorem current_population_correct : calculate_current_population initial_population percentage_died percentage_left = 3295 :=
  by
  unfold calculate_current_population
  sorry

end current_population_correct_l61_61578


namespace vertical_angles_equal_l61_61936

-- Given: Definition for pairs of adjacent angles summing up to 180 degrees
def adjacent_add_to_straight_angle (α β : ℝ) : Prop := 
  α + β = 180

-- Given: Two intersecting lines forming angles
variables (α β γ δ : ℝ)

-- Given: Relationship of adjacent angles being supplementary
axiom adj1 : adjacent_add_to_straight_angle α β
axiom adj2 : adjacent_add_to_straight_angle β γ
axiom adj3 : adjacent_add_to_straight_angle γ δ
axiom adj4 : adjacent_add_to_straight_angle δ α

-- Question: Prove that vertical angles are equal
theorem vertical_angles_equal : α = γ :=
by sorry

end vertical_angles_equal_l61_61936


namespace count_possible_integer_values_l61_61258

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l61_61258


namespace P1_coordinates_l61_61452

-- Define initial point coordinates
def P : (ℝ × ℝ) := (0, 3)

-- Define the transformation functions
def move_left (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1 - units, p.2)
def move_up (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1, p.2 + units)

-- Calculate the coordinates of point P1
def P1 : (ℝ × ℝ) := move_up (move_left P 2) 1

-- Statement to prove
theorem P1_coordinates : P1 = (-2, 4) := by
  sorry

end P1_coordinates_l61_61452


namespace projection_matrix_ratio_l61_61157

theorem projection_matrix_ratio
  (x y : ℚ)
  (h1 : (4/29) * x - (10/29) * y = x)
  (h2 : -(10/29) * x + (25/29) * y = y) :
  y / x = -5/2 :=
by
  sorry

end projection_matrix_ratio_l61_61157


namespace sum_of_first_10_terms_is_350_l61_61892

-- Define the terms and conditions for the arithmetic sequence
variables (a d : ℤ)

-- Define the 4th and 8th terms of the sequence
def fourth_term := a + 3*d
def eighth_term := a + 7*d

-- Given conditions
axiom h1 : fourth_term a d = 23
axiom h2 : eighth_term a d = 55

-- Sum of the first 10 terms of the sequence
def sum_first_10_terms := 10 / 2 * (2*a + (10 - 1)*d)

-- Theorem to prove
theorem sum_of_first_10_terms_is_350 : sum_first_10_terms a d = 350 :=
by sorry

end sum_of_first_10_terms_is_350_l61_61892


namespace price_of_72_cans_is_18_36_l61_61727

def regular_price_per_can : ℝ := 0.30
def discount_percent : ℝ := 0.15
def number_of_cans : ℝ := 72

def discounted_price_per_can : ℝ := regular_price_per_can - (discount_percent * regular_price_per_can)
def total_price (num_cans : ℝ) : ℝ := num_cans * discounted_price_per_can

theorem price_of_72_cans_is_18_36 :
  total_price number_of_cans = 18.36 :=
by
  /- Proof details omitted -/
  sorry

end price_of_72_cans_is_18_36_l61_61727


namespace sum_of_perimeters_l61_61484

theorem sum_of_perimeters (x y z : ℝ) 
    (h_large_triangle_perimeter : 3 * 20 = 60)
    (h_hexagon_perimeter : 60 - (x + y + z) = 40) :
    3 * (x + y + z) = 60 := by
  sorry

end sum_of_perimeters_l61_61484


namespace compute_fg_l61_61240

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem compute_fg : f (g (-3)) = 3 := by
  sorry

end compute_fg_l61_61240


namespace prob_13_know_news_prob_14_know_news_expected_know_news_l61_61544

variables (num_scientists knowers : ℕ) (know_news : ℕ → Prop)

-- conditions
def conditions : Prop := num_scientists = 18 ∧ knowers = 10 ∧ (∀ i < num_scientists, know_news i) ∧ num_scientists % 2 = 0

-- Question 1: Probability of exactly 13 scientists knowing the news after the coffee break is 0
theorem prob_13_know_news (h : conditions num_scientists knowers know_news) : 
  probability (λ s, s.countp know_news = 13) = 0 := 
sorry

-- Question 2: Probability of exactly 14 scientists knowing the news after the coffee break is 1120/2431
theorem prob_14_know_news (h : conditions num_scientists knowers know_news) : 
  probability (λ s, s.countp know_news = 14) = 1120 / 2431 := 
sorry

-- Question 3: Expected number of scientists who know the news after the coffee break is approximately 14.7
theorem expected_know_news (h : conditions num_scientists knowers know_news) : 
  E (λ s, s.countp know_news) ≈ 14.7 :=
sorry

end prob_13_know_news_prob_14_know_news_expected_know_news_l61_61544


namespace circle_equation_through_points_l61_61695

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61695


namespace percentage_of_second_solution_is_16point67_l61_61570

open Real

def percentage_second_solution (x : ℝ) : Prop :=
  let v₁ := 50
  let c₁ := 0.10
  let c₂ := x / 100
  let v₂ := 200 - v₁
  let c_final := 0.15
  let v_final := 200
  (c₁ * v₁) + (c₂ * v₂) = (c_final * v_final)

theorem percentage_of_second_solution_is_16point67 :
  ∃ x, percentage_second_solution x ∧ x = (50/3) :=
sorry

end percentage_of_second_solution_is_16point67_l61_61570


namespace range_of_a_for_empty_solution_set_l61_61282

theorem range_of_a_for_empty_solution_set :
  {a : ℝ | ∀ x : ℝ, (a^2 - 9) * x^2 + (a + 3) * x - 1 < 0} = 
  {a : ℝ | -3 ≤ a ∧ a < 9 / 5} :=
sorry

end range_of_a_for_empty_solution_set_l61_61282


namespace negation_of_proposition_l61_61160

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l61_61160


namespace problem_divisibility_l61_61806

theorem problem_divisibility 
  (m n : ℕ) 
  (a : Fin (mn + 1) → ℕ)
  (h_pos : ∀ i, 0 < a i)
  (h_order : ∀ i j, i < j → a i < a j) :
  (∃ (b : Fin (m + 1) → Fin (mn + 1)), ∀ i j, i ≠ j → ¬(a (b i) ∣ a (b j))) ∨
  (∃ (c : Fin (n + 1) → Fin (mn + 1)), ∀ i, i < n → a (c i) ∣ a (c i.succ)) :=
sorry

end problem_divisibility_l61_61806


namespace distance_between_ellipse_foci_l61_61064

-- Define the conditions of the problem
def center_of_ellipse (x1 y1 x2 y2 : ℝ) : Prop :=
  (2 * x1 = x2) ∧ (2 * y1 = y2)

def semi_axes (a b : ℝ) : Prop :=
  (a = 6) ∧ (b = 3)

-- Define the distance between the foci of the ellipse
def distance_between_foci (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

open Real

-- Statement of the theorem with the given conditions and expected result
theorem distance_between_ellipse_foci : 
  ∀ (x1 y1 x2 y2 a b : ℝ), 
  center_of_ellipse x1 y1 x2 y2 →
  semi_axes a b →
  distance_between_foci a b = 6 * sqrt 3 :=
by
  intros x1 y1 x2 y2 a b h_center h_axes,
  rw [center_of_ellipse, semi_axes] at h_axes,
  cases h_axes with h_a h_b,
  rw [distance_between_foci, h_a, h_b],
  sorry -- proof omitted

end distance_between_ellipse_foci_l61_61064


namespace find_rectangle_length_l61_61352

theorem find_rectangle_length (L W : ℕ) (h_area : L * W = 300) (h_perimeter : 2 * L + 2 * W = 70) : L = 20 :=
by
  sorry

end find_rectangle_length_l61_61352


namespace range_of_x_plus_y_l61_61439

theorem range_of_x_plus_y (x y : ℝ) (hx1 : y = 3 * ⌊x⌋ + 4) (hx2 : y = 4 * ⌊x - 3⌋ + 7) (hxnint : ¬ ∃ z : ℤ, x = z): 
  40 < x + y ∧ x + y < 41 :=
by
  sorry

end range_of_x_plus_y_l61_61439


namespace maximum_marks_l61_61447

theorem maximum_marks (M : ℝ) (h : 0.5 * M = 50 + 10) : M = 120 :=
by
  sorry

end maximum_marks_l61_61447


namespace denom_asymptotes_sum_l61_61324

theorem denom_asymptotes_sum (A B C : ℤ)
  (h_denom : ∀ x, (x = -1 ∨ x = 3 ∨ x = 4) → x^3 + A * x^2 + B * x + C = 0) :
  A + B + C = 11 := 
sorry

end denom_asymptotes_sum_l61_61324


namespace range_of_m_l61_61987

theorem range_of_m (m : ℝ) : (2 + m > 0) ∧ (1 - m > 0) ∧ (2 + m > 1 - m) → -1/2 < m ∧ m < 1 :=
by
  intros h
  sorry

end range_of_m_l61_61987


namespace monica_total_savings_l61_61588

noncomputable def weekly_savings (week: ℕ) : ℕ :=
  if week < 6 then 15 + 5 * week
  else if week < 11 then 40 - 5 * (week - 5)
  else weekly_savings (week % 10)

theorem monica_total_savings : 
  let cycle_savings := (15 + 20 + 25 + 30 + 35 + 40) + (40 + 35 + 30 + 25 + 20 + 15) - 40 
  let total_savings := 5 * cycle_savings
  total_savings = 1450 := by
  sorry

end monica_total_savings_l61_61588


namespace building_height_l61_61359

theorem building_height (h : ℕ) (flagpole_height : ℕ) (flagpole_shadow : ℕ) (building_shadow : ℕ) :
  flagpole_height = 18 ∧ flagpole_shadow = 45 ∧ building_shadow = 60 → h = 24 :=
by
  intros
  sorry

end building_height_l61_61359


namespace math_problem_l61_61441

theorem math_problem (p q : ℕ) (hp : p % 13 = 7) (hq : q % 13 = 7) (hp_lower : 1000 ≤ p) (hp_upper : p < 10000) (hq_lower : 10000 ≤ q) (min_p : ∀ n, n % 13 = 7 → 1000 ≤ n → n < 10000 → p ≤ n) (min_q : ∀ n, n % 13 = 7 → 10000 ≤ n → q ≤ n) : 
  q - p = 8996 := 
sorry

end math_problem_l61_61441


namespace find_number_l61_61586

def sum : ℕ := 2468 + 1375
def diff : ℕ := 2468 - 1375
def first_quotient : ℕ := 3 * diff
def second_quotient : ℕ := 5 * diff
def remainder : ℕ := 150

theorem find_number (N : ℕ) (h1 : sum = 3843) (h2 : diff = 1093) 
                    (h3 : first_quotient = 3279) (h4 : second_quotient = 5465)
                    (h5 : remainder = 150) (h6 : N = sum * first_quotient + remainder)
                    (h7 : N = sum * second_quotient + remainder) :
  N = 12609027 := 
by 
  sorry

end find_number_l61_61586


namespace percent_time_in_meetings_l61_61307

-- Define the conditions
def work_day_minutes : ℕ := 10 * 60  -- Total minutes in a 10-hour work day is 600 minutes
def first_meeting_minutes : ℕ := 60  -- The first meeting took 60 minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes  -- The second meeting took three times as long as the first meeting

-- Total time spent in meetings
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes  -- 60 + 180 = 240 minutes

-- The task is to prove that Makarla spent 40% of her work day in meetings.
theorem percent_time_in_meetings : (total_meeting_minutes / work_day_minutes : ℚ) * 100 = 40 := by
  sorry

end percent_time_in_meetings_l61_61307


namespace value_of_frac_l61_61267

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l61_61267


namespace circle_passing_through_points_l61_61633

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61633


namespace part1_part2_l61_61226

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

/-- Given sequence properties -/
axiom h1 : a 1 = 5
axiom h2 : ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1) + 2^n - 1

/-- Part (I): Proving the sequence is arithmetic -/
theorem part1 (n : ℕ) : ∃ d, (∀ m ≥ 1, (a (m + 1) - 1) / 2^(m + 1) - (a m - 1) / 2^m = d)
∧ ((a 1 - 1) / 2 = 2) := sorry

/-- Part (II): Sum of the first n terms -/
theorem part2 (n : ℕ) : S n = n * 2^(n+1) := sorry

end part1_part2_l61_61226


namespace circle_passing_three_points_l61_61702

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61702


namespace heartsuit_fraction_l61_61828

def heartsuit (n m : ℕ) : ℕ := n ^ 4 * m ^ 3

theorem heartsuit_fraction :
  (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 :=
by
  sorry

end heartsuit_fraction_l61_61828


namespace sum_of_asymptotes_l61_61154

theorem sum_of_asymptotes :
  let c := -3/2
  let d := -1
  c + d = -5/2 :=
by
  -- Definitions corresponding to the problem conditions
  let c := -3/2
  let d := -1
  -- Statement of the theorem
  show c + d = -5/2
  sorry

end sum_of_asymptotes_l61_61154


namespace circle_equation_through_points_l61_61699

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61699


namespace expected_number_of_adjacent_black_pairs_l61_61321

theorem expected_number_of_adjacent_black_pairs :
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_per_black_card := black_cards * adjacent_probability / total_cards
  let expected_total := black_cards * adjacent_probability
  expected_total = 650 / 51 := 
by
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_total := black_cards * adjacent_probability
  sorry

end expected_number_of_adjacent_black_pairs_l61_61321


namespace value_of_fraction_l61_61273

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l61_61273


namespace trajectory_of_point_l61_61280

theorem trajectory_of_point (x y : ℝ)
  (h1 : (x - 1)^2 + (y - 1)^2 = ((3 * x + y - 4)^2) / 10) :
  x - 3 * y + 2 = 0 :=
sorry

end trajectory_of_point_l61_61280


namespace equation_of_circle_passing_through_points_l61_61684

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61684


namespace circle_passes_through_points_l61_61626

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61626


namespace range_of_a_l61_61829

-- Given conditions
variable (a : ℝ)

def A (a : ℝ) : set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

-- Statement of the proof problem
theorem range_of_a (h : ∀ x1 x2 ∈ A a, x1 = x2) : a ≥ 9 / 8 ∨ a = 0 := 
sorry

end range_of_a_l61_61829


namespace evaluate_x_squared_plus_y_squared_l61_61232

theorem evaluate_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 20) :
  x^2 + y^2 = 80 := by
  sorry

end evaluate_x_squared_plus_y_squared_l61_61232


namespace parallel_lines_slope_l61_61156

theorem parallel_lines_slope (a : ℝ) :
  (∀ (x y : ℝ), x + a * y + 6 = 0 ∧ (a - 2) * x + 3 * y + 2 * a = 0 → (1 / (a - 2) = a / 3)) →
  a = -1 :=
by {
  sorry
}

end parallel_lines_slope_l61_61156


namespace total_area_correct_l61_61758

noncomputable def total_area (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : ℝ :=
  let rect_area := 588 -- Area of the rectangle
  let semi_circle_area := 24.5 * Real.pi -- Area of the semi-circle based on given diameter
  rect_area + semi_circle_area

theorem total_area_correct (b l: ℝ) (h1: l = 3 * b) (h2: l * b = 588) : 
  total_area b l h1 h2 = 588 + 24.5 * Real.pi :=
by
  sorry

end total_area_correct_l61_61758


namespace parabola_distance_l61_61981

theorem parabola_distance (p : ℝ) (hp : 0 < p) (hf : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  dist P (0, p / 2) = 16) (hx : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  P.2 = 10) : p = 12 :=
sorry

end parabola_distance_l61_61981


namespace two_bedroom_units_l61_61194

theorem two_bedroom_units (x y : ℕ) (h1 : x + y = 12) (h2 : 360 * x + 450 * y = 4950) : y = 7 :=
by
  sorry

end two_bedroom_units_l61_61194


namespace sum_of_ages_is_42_l61_61335

-- Define the variables for present ages of the son (S) and the father (F)
variables (S F : ℕ)

-- Define the conditions:
-- 1. 6 years ago, the father's age was 4 times the son's age.
-- 2. After 6 years, the son's age will be 18 years.

def son_age_condition := S + 6 = 18
def father_age_6_years_ago_condition := F - 6 = 4 * (S - 6)

-- Theorem statement to prove:
theorem sum_of_ages_is_42 (S F : ℕ)
  (h1 : son_age_condition S)
  (h2 : father_age_6_years_ago_condition F S) :
  S + F = 42 :=
sorry

end sum_of_ages_is_42_l61_61335


namespace expression_1_expression_2_expression_3_expression_4_l61_61951

section problem1

variable {x : ℝ}

theorem expression_1:
  (x^2 - 1 + x)*(x^2 - 1 + 3*x) + x^2  = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 :=
sorry

end problem1

section problem2

variable {x a : ℝ}

theorem expression_2:
  (x - a)^4 + 4*a^4 = (x^2 + a^2)*(x^2 - 4*a*x + 5*a^2) :=
sorry

end problem2

section problem3

variable {a : ℝ}

theorem expression_3:
  (a + 1)^4 + 2*(a + 1)^3 + a*(a + 2) = (a + 1)^4 + 2*(a + 1)^3 + 1 :=
sorry

end problem3

section problem4

variable {p : ℝ}

theorem expression_4:
  (p + 2)^4 + 2*(p^2 - 4)^2 + (p - 2)^4 = 4*p^4 :=
sorry

end problem4

end expression_1_expression_2_expression_3_expression_4_l61_61951


namespace sufficient_but_not_necessary_l61_61743

theorem sufficient_but_not_necessary (x : ℝ) : (x - 1 > 0) → (x^2 - 1 > 0) ∧ ¬((x^2 - 1 > 0) → (x - 1 > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l61_61743


namespace inequality_solution_l61_61306

theorem inequality_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b + c = 1) : (1 / (b * c + a + 1 / a) + 1 / (c * a + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31) :=
by sorry

end inequality_solution_l61_61306


namespace relationship_xyz_l61_61223

theorem relationship_xyz (a b : ℝ) (x y z : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a > b) (hab_sum : a + b = 1) 
  (hx : x = Real.log b / Real.log a)
  (hy : y = Real.log (1 / b) / Real.log a)
  (hz : z = Real.log 3 / Real.log ((1 / a) + (1 / b))) : 
  y < z ∧ z < x := 
sorry

end relationship_xyz_l61_61223


namespace circle_equation_l61_61646

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61646


namespace inequality_solution_l61_61016

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l61_61016


namespace frog_probability_vertical_side_l61_61507

-- Definition of initial frog position and grid dimensions
def frog_initial_position := (2, 3)
def grid_bottom_left := (0, 0)
def grid_top_left := (0, 5)
def grid_top_right := (6, 5)
def grid_bottom_right := (6, 0)

-- Definition of grid boundaries (vertical sides)
def is_on_vertical_side (x y : ℕ) : Prop :=
  x = 0 ∨ x = 6

-- Probability that frog ends on vertical side given initial position and grid restrictions
def P (x y : ℕ) : ℚ := sorry

theorem frog_probability_vertical_side :
  P 2 3 = 2 / 3 := sorry

end frog_probability_vertical_side_l61_61507


namespace john_change_received_is_7_l61_61291

def cost_per_orange : ℝ := 0.75
def num_oranges : ℝ := 4
def amount_paid : ℝ := 10.0
def total_cost : ℝ := num_oranges * cost_per_orange
def change_received : ℝ := amount_paid - total_cost

theorem john_change_received_is_7 : change_received = 7 :=
by
  sorry

end john_change_received_is_7_l61_61291


namespace pizza_share_l61_61770

theorem pizza_share :
  forall (friends : ℕ) (leftover_pizza : ℚ), friends = 4 -> leftover_pizza = 5/6 -> (leftover_pizza / friends) = (5 / 24) :=
by
  intros friends leftover_pizza h_friends h_leftover_pizza
  sorry

end pizza_share_l61_61770


namespace solve_inequality_system_l61_61024

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61024


namespace factorization_correct_l61_61349

theorem factorization_correct : 
  ¬(∃ x : ℝ, -x^2 + 4 * x = -x * (x + 4)) ∧
  ¬(∃ x y: ℝ, x^2 + x * y + x = x * (x + y)) ∧
  (∀ x y: ℝ, x * (x - y) + y * (y - x) = (x - y)^2) ∧
  ¬(∃ x : ℝ, x^2 - 4 * x + 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correct_l61_61349


namespace alternating_draws_probability_l61_61054

noncomputable def probability_alternating_draws : ℚ :=
  let total_draws := 11
  let white_balls := 5
  let black_balls := 6
  let successful_sequences := 1
  let total_sequences := @Nat.choose total_draws black_balls
  successful_sequences / total_sequences

theorem alternating_draws_probability :
  probability_alternating_draws = 1 / 462 := by
  sorry

end alternating_draws_probability_l61_61054


namespace cubes_not_arithmetic_progression_l61_61178

theorem cubes_not_arithmetic_progression (x y z : ℤ) (h1 : y = (x + z) / 2) (h2 : x ≠ y) (h3 : y ≠ z) : x^3 + z^3 ≠ 2 * y^3 :=
by
  sorry

end cubes_not_arithmetic_progression_l61_61178


namespace problem_l61_61790

open Complex

noncomputable def zeta := exp (2 * π * I / 14)

theorem problem : (3 - zeta) * (3 - zeta^2) * (3 - zeta^3) * (3 - zeta^4) * (3 - zeta^5) * (3 - zeta^6) * (3 - zeta^7) * (3 - zeta^8) * (3 - zeta^9) * (3 - zeta^10) * (3 - zeta^11) * (3 - zeta^12) * (3 - zeta^13) = 2143588 :=
by sorry

end problem_l61_61790


namespace find_cd_product_l61_61220

open Complex

theorem find_cd_product :
  let u : ℂ := -3 + 4 * I
  let v : ℂ := 2 - I
  let c : ℂ := -5 + 5 * I
  let d : ℂ := -5 - 5 * I
  c * d = 50 :=
by
  sorry

end find_cd_product_l61_61220


namespace fraction_equality_l61_61111

-- Defining the hypotheses and the goal
theorem fraction_equality (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 10 :=
by
  sorry

end fraction_equality_l61_61111


namespace circle_through_points_l61_61675

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61675


namespace tim_biking_time_l61_61910

theorem tim_biking_time
  (work_days : ℕ := 5) 
  (distance_to_work : ℕ := 20) 
  (weekend_ride : ℕ := 200) 
  (speed : ℕ := 25) 
  (weekly_work_distance := 2 * distance_to_work * work_days)
  (total_distance := weekly_work_distance + weekend_ride) : 
  (total_distance / speed = 16) := 
by
  sorry

end tim_biking_time_l61_61910


namespace negation_of_forall_statement_l61_61161

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l61_61161


namespace equation_of_line_l61_61483

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 4

-- Define the line equation with parameters m and b
def line (m b x : ℝ) : ℝ := m * x + b

-- Define the point of intersection with the parabola on the line x = k
def intersection_point_parabola (k : ℝ) : ℝ := parabola k

-- Define the point of intersection with the line on the line x = k
def intersection_point_line (m b k : ℝ) : ℝ := line m b k

-- Define the vertical distance between the points on x = k
def vertical_distance (k m b : ℝ) : ℝ :=
  abs ((parabola k) - (line m b k))

-- Define the condition that vertical distance is exactly 4 units
def intersection_distance_condition (k m b : ℝ) : Prop :=
  vertical_distance k m b = 4

-- The line passes through point (2, 8)
def passes_through_point (m b : ℝ) : Prop :=
  line m b 2 = 8

-- Non-zero y-intercept condition
def non_zero_intercept (b : ℝ) : Prop := 
  b ≠ 0

-- The final theorem stating the required equation of the line
theorem equation_of_line (m b : ℝ) (h1 : ∃ k, intersection_distance_condition k m b)
  (h2 : passes_through_point m b) (h3 : non_zero_intercept b) : 
  (m = 12 ∧ b = -16) :=
by
  sorry

end equation_of_line_l61_61483


namespace problem_statement_l61_61274

variables {a b x : ℝ}

theorem problem_statement (h1 : x = a / b + 2) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + 2 * b) / (a - 2 * b) = x / (x - 4) := 
sorry

end problem_statement_l61_61274


namespace group_total_cost_l61_61539

noncomputable def total_cost
  (num_people : Nat) 
  (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem group_total_cost (num_people := 15) (cost_per_person := 900) :
  total_cost num_people cost_per_person = 13500 :=
by
  sorry

end group_total_cost_l61_61539


namespace gcd_condition_l61_61477

def seq (a : ℕ → ℕ) := a 0 = 3 ∧ ∀ n, a (n + 1) - a n = n * (a n - 1)

theorem gcd_condition (a : ℕ → ℕ) (m : ℕ) (h : seq a) :
  m ≥ 2 → (∀ n, Nat.gcd m (a n) = 1) ↔ ∃ k : ℕ, m = 2^k ∧ k ≥ 1 := 
sorry

end gcd_condition_l61_61477


namespace find_t_l61_61805

-- Define the elements and the conditions
def vector_a : ℝ × ℝ := (1, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 1)

def add_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def sub_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Lean statement of the problem
theorem find_t (t : ℝ) : 
  parallel (add_vectors vector_a (vector_b t)) (sub_vectors vector_a (vector_b t)) → t = -1 :=
by
  sorry

end find_t_l61_61805


namespace find_circle_equation_l61_61713

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61713


namespace cousins_room_distributions_l61_61138

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l61_61138


namespace binary_to_decimal_11011_is_27_l61_61779

def binary_to_decimal : ℕ :=
  1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem binary_to_decimal_11011_is_27 : binary_to_decimal = 27 := by
  sorry

end binary_to_decimal_11011_is_27_l61_61779


namespace depth_of_melted_sauce_l61_61513

theorem depth_of_melted_sauce
  (r_sphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) (volume_conserved : Bool) :
  r_sphere = 3 ∧ r_cylinder = 10 ∧ volume_conserved → h_cylinder = 9/25 :=
by
  -- Explanation of the condition: 
  -- r_sphere is the radius of the original spherical globe (3 inches)
  -- r_cylinder is the radius of the cylindrical puddle (10 inches)
  -- h_cylinder is the depth we need to prove is 9/25 inches
  -- volume_conserved indicates that the volume is conserved
  sorry

end depth_of_melted_sauce_l61_61513


namespace sum_first_13_terms_l61_61399

variable {a_n : ℕ → ℝ} (S : ℕ → ℝ)
variable (a_1 d : ℝ)

-- Arithmetic sequence properties
axiom arithmetic_sequence (n : ℕ) : a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms
axiom sum_of_terms (n : ℕ) : S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_specific_terms : a_n 2 + a_n 7 + a_n 12 = 30

-- Theorem to prove
theorem sum_first_13_terms : S 13 = 130 := sorry

end sum_first_13_terms_l61_61399


namespace calculate_polygon_sides_l61_61512

-- Let n be the number of sides of the regular polygon with each exterior angle of 18 degrees
def regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 18 ∧ n * exterior_angle = 360

theorem calculate_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  regular_polygon_sides n exterior_angle → n = 20 :=
by
  intro h
  sorry

end calculate_polygon_sides_l61_61512


namespace ball_count_l61_61286

theorem ball_count (white red blue : ℕ)
  (h_ratio : white = 4 ∧ red = 3 ∧ blue = 2)
  (h_white : white = 16) :
  red = 12 ∧ blue = 8 :=
by
  -- Proof skipped
  sorry

end ball_count_l61_61286


namespace base_number_equals_2_l61_61421

theorem base_number_equals_2 (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^26) (h2 : n = 25) : x = 2 :=
by
  sorry

end base_number_equals_2_l61_61421


namespace no_three_distinct_nat_numbers_sum_prime_l61_61787

theorem no_three_distinct_nat_numbers_sum_prime:
  ¬∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 
  Nat.Prime (a + b) ∧ Nat.Prime (a + c) ∧ Nat.Prime (b + c) := 
sorry

end no_three_distinct_nat_numbers_sum_prime_l61_61787


namespace circle_passing_three_points_l61_61704

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61704


namespace mode_is_3_5_of_salaries_l61_61278

def salaries : List ℚ := [30, 14, 9, 6, 4, 3.5, 3]
def frequencies : List ℕ := [1, 2, 3, 4, 5, 6, 4]

noncomputable def mode_of_salaries (salaries : List ℚ) (frequencies : List ℕ) : ℚ :=
by
  sorry

theorem mode_is_3_5_of_salaries :
  mode_of_salaries salaries frequencies = 3.5 :=
by
  sorry

end mode_is_3_5_of_salaries_l61_61278


namespace part_A_part_B_part_C_l61_61817

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f(x * y) = y^2 * f(x) + x^2 * f(y)

theorem part_A : f(0) = 0 := sorry
theorem part_B : f(1) = 0 := sorry
theorem part_C : ∀ x : ℝ, f(x) = f(-x) := sorry

end part_A_part_B_part_C_l61_61817


namespace cos_pi_minus_2alpha_l61_61276

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (Real.pi - 2 * α) = -5 / 9 := by
  sorry

end cos_pi_minus_2alpha_l61_61276


namespace product_pattern_l61_61926

theorem product_pattern (a b : ℕ) (h1 : b < 10) (h2 : 10 - b < 10) :
    (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) :=
by
  sorry

end product_pattern_l61_61926


namespace intersection_of_A_and_B_l61_61569

open Set

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
  sorry

end intersection_of_A_and_B_l61_61569


namespace remaining_area_is_344_l61_61150

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def shed_side : ℕ := 4

def area_rectangle : ℕ := garden_length * garden_width
def area_shed : ℕ := shed_side * shed_side

def remaining_garden_area : ℕ := area_rectangle - area_shed

theorem remaining_area_is_344 : remaining_garden_area = 344 := by
  sorry

end remaining_area_is_344_l61_61150


namespace min_editors_at_conference_l61_61768

variable (x E : ℕ)

theorem min_editors_at_conference (h1 : x ≤ 26) 
    (h2 : 100 = 35 + E + x) 
    (h3 : 2 * x ≤ 100 - 35 - E + x) : 
    E ≥ 39 :=
by
  sorry

end min_editors_at_conference_l61_61768


namespace equation_of_circle_passing_through_points_l61_61691

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61691


namespace find_circle_equation_l61_61714

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61714


namespace compare_neg_third_and_neg_point_three_l61_61532

/-- Compare two numbers -1/3 and -0.3 -/
theorem compare_neg_third_and_neg_point_three : (-1 / 3 : ℝ) < -0.3 :=
sorry

end compare_neg_third_and_neg_point_three_l61_61532


namespace find_sum_l61_61467

def f (x : ℝ) : ℝ := sorry

axiom f_non_decreasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 ≤ 1 → 0 ≤ x2 → x2 ≤ 1 → x1 < x2 → f x1 ≤ f x2
axiom f_at_0 : f 0 = 0
axiom f_scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (x / 3) = (1 / 2) * f x
axiom f_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (1 - x) = 1 - f x

theorem find_sum :
  f (1 / 3) + f (2 / 3) + f (1 / 9) + f (1 / 6) + f (1 / 8) = 7 / 4 :=
by
  sorry

end find_sum_l61_61467


namespace total_ladders_climbed_in_inches_l61_61436

-- Define the conditions as hypotheses
def keaton_ladder_length := 30
def keaton_climbs := 20
def reece_ladder_difference := 4
def reece_climbs := 15
def feet_to_inches := 12

-- Define the lengths climbed by Keaton and Reece
def keaton_total_feet := keaton_ladder_length * keaton_climbs
def reece_ladder_length := keaton_ladder_length - reece_ladder_difference
def reece_total_feet := reece_ladder_length * reece_climbs

-- Calculate the total feet climbed and convert to inches
def total_feet := keaton_total_feet + reece_total_feet
def total_inches := total_feet * feet_to_inches

-- Prove the final result
theorem total_ladders_climbed_in_inches : total_inches = 11880 := by
  sorry

end total_ladders_climbed_in_inches_l61_61436


namespace expression_evaluation_l61_61112

theorem expression_evaluation (m : ℝ) (h : m = Real.sqrt 2023 + 2) : m^2 - 4 * m + 5 = 2024 :=
by sorry

end expression_evaluation_l61_61112


namespace librarians_all_work_together_l61_61590

/-- Peter works every 5 days -/
def Peter_days := 5

/-- Quinn works every 8 days -/
def Quinn_days := 8

/-- Rachel works every 10 days -/
def Rachel_days := 10

/-- Sam works every 14 days -/
def Sam_days := 14

/-- Least common multiple of the intervals at which Peter, Quinn, Rachel, and Sam work -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem librarians_all_work_together : LCM (LCM (LCM Peter_days Quinn_days) Rachel_days) Sam_days = 280 :=
  by
  sorry

end librarians_all_work_together_l61_61590


namespace problem1_problem2_l61_61103

-- Definition of sets A and B
def A : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def B (p : ℝ) : Set ℝ := { x | abs (x - p) > 1 }

-- Statement for the first problem
theorem problem1 : B 0 ∩ A = { x | 1 < x ∧ x < 3 } := 
by
  sorry

-- Statement for the second problem
theorem problem2 (p : ℝ) (h : A ∪ B p = B p) : p ≤ -2 ∨ p ≥ 4 := 
by
  sorry

end problem1_problem2_l61_61103


namespace average_speed_of_train_l61_61373

theorem average_speed_of_train
  (distance1 : ℝ) (time1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (time2 : ℝ)
  (h1 : distance1 = 240) (h2 : time1 = 3) (h3 : stop_time = 0.5)
  (h4 : distance2 = 450) (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = 81.18 := 
sorry

end average_speed_of_train_l61_61373


namespace find_xy_such_that_product_is_fifth_power_of_prime_l61_61795

theorem find_xy_such_that_product_is_fifth_power_of_prime
  (x y : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (x^2 + y) * (y^2 + x) = p^5) :
  (x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2) :=
sorry

end find_xy_such_that_product_is_fifth_power_of_prime_l61_61795


namespace xy_square_diff_l61_61852

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61852


namespace chord_length_perpendicular_bisector_of_radius_l61_61197

theorem chord_length_perpendicular_bisector_of_radius (r : ℝ) (h : r = 15) :
  ∃ (CD : ℝ), CD = 15 * Real.sqrt 3 :=
by
  sorry

end chord_length_perpendicular_bisector_of_radius_l61_61197


namespace area_of_isosceles_triangle_l61_61343

open Real

theorem area_of_isosceles_triangle 
  (PQ PR QR : ℝ) (PQ_eq_PR : PQ = PR) (PQ_val : PQ = 13) (QR_val : QR = 10) : 
  1 / 2 * QR * sqrt (PQ^2 - (QR / 2)^2) = 60 := 
by 
sorry

end area_of_isosceles_triangle_l61_61343


namespace millie_bracelets_left_l61_61882

def millie_bracelets_initial : ℕ := 9
def millie_bracelets_lost : ℕ := 2

theorem millie_bracelets_left : millie_bracelets_initial - millie_bracelets_lost = 7 := 
by
  sorry

end millie_bracelets_left_l61_61882


namespace evaluate_expression_l61_61086

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 5 * x + 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end evaluate_expression_l61_61086


namespace gcd_fact_8_10_l61_61963

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l61_61963


namespace range_of_f_neg2_l61_61414

def quadratic_fn (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_neg2 (a b : ℝ) (h1 : 1 ≤ quadratic_fn a b (-1) ∧ quadratic_fn a b (-1) ≤ 2)
    (h2 : 2 ≤ quadratic_fn a b 1 ∧ quadratic_fn a b 1 ≤ 4) :
    3 ≤ quadratic_fn a b (-2) ∧ quadratic_fn a b (-2) ≤ 12 :=
sorry

end range_of_f_neg2_l61_61414


namespace find_sum_of_numbers_l61_61904

variables (a b c : ℕ) (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300)

theorem find_sum_of_numbers (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300) :
  a + b + c = 14700 :=
sorry

end find_sum_of_numbers_l61_61904


namespace no_square_ends_with_four_identical_digits_except_0_l61_61049

theorem no_square_ends_with_four_identical_digits_except_0 (n : ℤ) :
  ¬ (∃ k : ℕ, (1 ≤ k ∧ k < 10) ∧ (n^2 % 10000 = k * 1111)) :=
by {
  sorry
}

end no_square_ends_with_four_identical_digits_except_0_l61_61049


namespace daily_harvest_sacks_l61_61716

theorem daily_harvest_sacks (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 65 → num_sections = 12 → total_sacks = sacks_per_section * num_sections → total_sacks = 780 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end daily_harvest_sacks_l61_61716


namespace circle_equation_correct_l61_61605

theorem circle_equation_correct :
  ∃ (D E F : ℝ), (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), 
    ((x, y) = (0,0) ∨ (x, y) = (4,0) ∨ (x, y) = (-1,1)) →
    (x^2 + y^2 + D * x + E * y + F = 0)) :=
begin
  existsi [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  intros x y h,
  cases h,
  { rw [h, mul_zero, mul_zero, add_zero, add_zero, add_zero] },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  { rw [h, pow_smul, pow_smul, mul_add, mul_add, add_smul, add_smul,
        pow_smul, add], ring },
  sorry, -- complete the proof or provide specific details if necessary
end

end circle_equation_correct_l61_61605


namespace sqrt_product_simplification_l61_61078

variable (q : ℝ)

theorem sqrt_product_simplification (hq : q ≥ 0) : 
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by
  sorry

end sqrt_product_simplification_l61_61078


namespace arithmetic_geo_sequences_l61_61810

theorem arithmetic_geo_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) (d : ℕ) (q : ℕ) (n : ℕ) :
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, b n = 3 ^ n) →
  (∀ k, ∑ i in Finset.range k, b (2 * i + 1) = (3 ^ k - 1) / 2) :=
  sorry

end arithmetic_geo_sequences_l61_61810


namespace solve_recursive_fn_eq_l61_61398

-- Define the recursive function
def recursive_fn (x : ℝ) : ℝ :=
  2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1

-- State the theorem we need to prove
theorem solve_recursive_fn_eq (x : ℝ) : recursive_fn x = x → x = 1 :=
by
  sorry

end solve_recursive_fn_eq_l61_61398


namespace smallest_x_value_l61_61218

open Real

theorem smallest_x_value (x : ℝ) 
  (h : x * abs x = 3 * x + 2) : 
  x = -2 ∨ (∀ y, y * abs y = 3 * y + 2 → y ≥ -2) := sorry

end smallest_x_value_l61_61218


namespace solve_for_x_l61_61461

theorem solve_for_x (x y : ℚ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 :=
by
  intro h
  sorry

end solve_for_x_l61_61461


namespace remainder_of_N_mod_103_l61_61442

noncomputable def N : ℕ :=
  sorry -- This will capture the mathematical calculation of N using the conditions stated.

theorem remainder_of_N_mod_103 : (N % 103) = 43 :=
  sorry

end remainder_of_N_mod_103_l61_61442


namespace problem_statement_l61_61961

noncomputable def f (n : ℕ) : ℝ := Real.log (n^2) / Real.log 3003

theorem problem_statement : f 33 + f 13 + f 7 = 2 := 
by
  sorry

end problem_statement_l61_61961


namespace least_time_for_4_horses_sum_of_digits_S_is_6_l61_61336

-- Definition of horse run intervals
def horse_intervals : List Nat := List.range' 1 9 |>.map (λ k => 2 * k)

-- Function to compute LCM of a set of numbers
def lcm_set (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

-- Proving that 4 of the horse intervals have an LCM of 24
theorem least_time_for_4_horses : 
  ∃ S > 0, (S = 24 ∧ (lcm_set [2, 4, 6, 8] = S)) ∧
  (List.length (horse_intervals.filter (λ t => S % t = 0)) ≥ 4) := 
by
  sorry

-- Proving the sum of the digits of S (24) is 6
theorem sum_of_digits_S_is_6 : 
  let S := 24
  (S / 10 + S % 10 = 6) :=
by
  sorry

end least_time_for_4_horses_sum_of_digits_S_is_6_l61_61336


namespace number_of_ways_to_choose_committee_l61_61428

theorem number_of_ways_to_choose_committee {P V M F : ℕ} (hP : P = 10) (hV : V = 1) (hM : M = 6) (hF : F = 4) :
  let ways_choose_president_and_vp := (P - V) * (P - V - 1),
      ways_choose_committee :=
        (choose (M - V) 1 * choose (F - V) 2) +
        (choose (M - V) 2 * choose (F - V) 1)
  in ways_choose_president_and_vp * ways_choose_committee = 8640 := by
  sorry

end number_of_ways_to_choose_committee_l61_61428


namespace fraction_addition_l61_61794

theorem fraction_addition :
  (2 / 5 : ℚ) + (3 / 8) = 31 / 40 :=
sorry

end fraction_addition_l61_61794


namespace circle_passing_three_points_l61_61703

def point := (ℝ × ℝ)

def circle_eq (D E F : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D*x + E*y + F = 0

def circle_passes_through (D E F : ℝ) (p : point) : Prop :=
  circle_eq D E F p.1 p.2

theorem circle_passing_three_points :
  ∃ D E F : ℝ, circle_passes_through D E F (0,0) ∧
               circle_passes_through D E F (4,0) ∧
               circle_passes_through D E F (-1,1) ∧ 
               (circle_eq D E F = λ x y, x^2 + y^2 - 4*x - 6*y) :=
by
  sorry

end circle_passing_three_points_l61_61703


namespace circle_passing_through_points_eqn_l61_61658

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61658


namespace cost_equiv_banana_pear_l61_61379

-- Definitions based on conditions
def Banana : Type := ℝ
def Apple : Type := ℝ
def Pear : Type := ℝ

-- Given conditions
axiom cost_equiv_1 : 4 * (Banana : ℝ) = 3 * (Apple : ℝ)
axiom cost_equiv_2 : 9 * (Apple : ℝ) = 6 * (Pear : ℝ)

-- Theorem to prove
theorem cost_equiv_banana_pear : 24 * (Banana : ℝ) = 12 * (Pear : ℝ) :=
by
  sorry

end cost_equiv_banana_pear_l61_61379


namespace find_circle_equation_l61_61711

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61711


namespace park_area_l61_61329

theorem park_area (P : ℝ) (w l : ℝ) (hP : P = 120) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 675 :=
by
  sorry

end park_area_l61_61329


namespace solveForX_l61_61918

theorem solveForX : ∃ (x : ℚ), x + 5/8 = 7/24 + 1/4 ∧ x = -1/12 := by
  sorry

end solveForX_l61_61918


namespace math_problem_l61_61246

variable (x Q : ℝ)

theorem math_problem (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := 
by
  sorry

end math_problem_l61_61246


namespace inequality_system_solution_l61_61019

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l61_61019


namespace inequalities_hold_l61_61402

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ (b - a) / c > 0 ∧ (a - c) / (a * c) < 0 :=
by 
  sorry

end inequalities_hold_l61_61402


namespace x_axis_line_l61_61422

variable (A B C : ℝ)

theorem x_axis_line (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : B ≠ 0 ∧ A = 0 ∧ C = 0 := by
  sorry

end x_axis_line_l61_61422


namespace determine_h_l61_61593

open Polynomial

noncomputable def f (x : ℚ) : ℚ := x^2

theorem determine_h (h : ℚ → ℚ) : 
  (∀ x, f (h x) = 9 * x^2 + 6 * x + 1) ↔ 
  (∀ x, h x = 3 * x + 1 ∨ h x = - (3 * x + 1)) :=
by
  sorry

end determine_h_l61_61593


namespace find_circle_equation_l61_61710

theorem find_circle_equation :
  ∃ (D E F : ℝ), 
  (D = -4) ∧ (E = -6) ∧ (F = 0) ∧
  (∀ (x y: ℝ), (x = 0 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = 4 ∧ y = 0) → (x^2 + y^2 + D*x + E*y + F = 0)) ∧
  (∀ (x y: ℝ), (x = -1 ∧ y = 1) → (x^2 + y^2 + D*x + E*y + F = 0)) :=
begin
  use [-4, -6, 0],
  split, refl,
  split, refl,
  split, refl,
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg] },
  split,
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] },
  { rintros x y ⟨hx, hy⟩,
    subst hx, subst hy,
    simp only [zero_add, add_zero, mul_zero, add_eq_zero_iff_eq_zero_of_nonneg, neg_mul_eq_neg_mul, neg_add_eq_sub] }
end

end find_circle_equation_l61_61710


namespace intersection_of_complements_l61_61131

theorem intersection_of_complements {U S T : Set ℕ}
  (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
  (hS : S = {1, 3, 5})
  (hT : T = {3, 6}) :
  (U \ S) ∩ (U \ T) = {2, 4, 7, 8} :=
by
  sorry

end intersection_of_complements_l61_61131


namespace bus_ride_difference_l61_61488

theorem bus_ride_difference (vince_bus_length zachary_bus_length : Real)
    (h_vince : vince_bus_length = 0.62)
    (h_zachary : zachary_bus_length = 0.5) :
    vince_bus_length - zachary_bus_length = 0.12 :=
by
  sorry

end bus_ride_difference_l61_61488


namespace num_possible_integer_values_x_l61_61250

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l61_61250


namespace identity_x_squared_minus_y_squared_l61_61841

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61841


namespace equation_of_circle_passing_through_points_l61_61688

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end equation_of_circle_passing_through_points_l61_61688


namespace shaded_area_inequality_l61_61389

theorem shaded_area_inequality 
    (A : ℝ) -- All three triangles have the same total area, A.
    {a1 a2 a3 : ℝ} -- a1, a2, a3 are the shaded areas of Triangle I, II, and III respectively.
    (h1 : a1 = A / 6) 
    (h2 : a2 = A / 2) 
    (h3 : a3 = (2 * A) / 3) : 
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 :=
by
  -- Proof steps would go here, but they are not required as per the instructions
  sorry

end shaded_area_inequality_l61_61389


namespace find_fx_plus_1_l61_61105

theorem find_fx_plus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x - 1) = x^2 + 4 * x - 5) : 
  ∀ x : ℤ, f (x + 1) = x^2 + 8 * x + 7 :=
sorry

end find_fx_plus_1_l61_61105


namespace min_product_of_positive_numbers_l61_61045

theorem min_product_of_positive_numbers {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b) : a * b = 4 :=
sorry

end min_product_of_positive_numbers_l61_61045


namespace circle_equation_through_points_l61_61697

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l61_61697


namespace sum_difference_l61_61181

def even_sum (n : ℕ) : ℕ :=
  n * (n + 1)

def odd_sum (n : ℕ) : ℕ :=
  n^2

theorem sum_difference : even_sum 100 - odd_sum 100 = 100 := by
  sorry

end sum_difference_l61_61181


namespace Kates_hair_length_l61_61300

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l61_61300


namespace water_in_pool_after_35_days_l61_61357

theorem water_in_pool_after_35_days :
  ∀ (initial_amount : ℕ) (evap_rate : ℕ) (cycle_days : ℕ) (add_amount : ℕ) (total_days : ℕ),
  initial_amount = 300 → evap_rate = 1 → cycle_days = 5 → add_amount = 5 → total_days = 35 →
  initial_amount - evap_rate * total_days + (total_days / cycle_days) * add_amount = 300 :=
by
  intros initial_amount evap_rate cycle_days add_amount total_days h₁ h₂ h₃ h₄ h₅
  sorry

end water_in_pool_after_35_days_l61_61357


namespace original_rectangle_perimeter_l61_61510

theorem original_rectangle_perimeter (l w : ℝ) (h1 : w = l / 2)
  (h2 : 2 * (w + l / 3) = 40) : 2 * l + 2 * w = 72 :=
by
  sorry

end original_rectangle_perimeter_l61_61510


namespace hyperbola_asymptote_slope_l61_61397

theorem hyperbola_asymptote_slope :
  ∀ {x y : ℝ}, (x^2 / 144 - y^2 / 81 = 1) → (∃ m : ℝ, ∀ x, y = m * x ∨ y = -m * x ∧ m = 3 / 4) :=
by
  sorry

end hyperbola_asymptote_slope_l61_61397


namespace circumference_of_tire_l61_61277

theorem circumference_of_tire (rotations_per_minute : ℕ) (speed_kmh : ℕ) 
  (h1 : rotations_per_minute = 400) (h2 : speed_kmh = 72) :
  let speed_mpm := speed_kmh * 1000 / 60
  let circumference := speed_mpm / rotations_per_minute
  circumference = 3 :=
by
  sorry

end circumference_of_tire_l61_61277


namespace y_value_l61_61106

theorem y_value (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + 1 / y = 8) (h4 : y + 1 / x = 7 / 12) (h5 : x + y = 7) : y = 49 / 103 :=
by
  sorry

end y_value_l61_61106


namespace gcd_fact_8_10_l61_61962

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l61_61962


namespace original_price_is_975_l61_61741

variable (x : ℝ)
variable (discounted_price : ℝ := 780)
variable (discount : ℝ := 0.20)

-- The condition that Smith bought the shirt for Rs. 780 after a 20% discount
def original_price_calculation (x : ℝ) (discounted_price : ℝ) (discount : ℝ) : Prop :=
  (1 - discount) * x = discounted_price

theorem original_price_is_975 : ∃ x : ℝ, original_price_calculation x 780 0.20 ∧ x = 975 := 
by
  -- Proof will be provided here
  sorry

end original_price_is_975_l61_61741


namespace students_liked_strawberries_l61_61887

theorem students_liked_strawberries : 
  let total_students := 450 
  let students_oranges := 70 
  let students_pears := 120 
  let students_apples := 147 
  let students_strawberries := total_students - (students_oranges + students_pears + students_apples)
  students_strawberries = 113 :=
by
  sorry

end students_liked_strawberries_l61_61887


namespace part_b_part_c_l61_61740

-- Definitions for the problem conditions
def board4x4 := Fin 4 × Fin 4
def board8x8 := Fin 8 × Fin 8

-- 4x4 board problem (part b)
theorem part_b:
  ∃ (choices: Finset board4x4), 
    (∀ i : Fin 4, (choices.filter (λ pos, pos.1 = i)).card = 3) ∧ 
    (∀ j : Fin 4, (choices.filter (λ pos, pos.2 = j)).card = 3) →
    (Finset.univ.card * 3 = 24) :=
sorry

-- 8x8 board problem (part c)
theorem part_c:
  ∃ (black_squares white_squares: Finset board8x8),
    (black_squares.filter (λ pos, (pos.1 + pos.2) % 2 = 0) ∧ 
    pos ∉ white_squares = Finset.univ.filter (λ pos, (pos.1 + pos.2) % 2 = 0)).card = 32 ∧
    (white_squares.filter (λ pos, (pos.1 + pos.2) % 2 ≠ 0)).card = 24 ∧
    (∀ i : Fin 8, (white_squares.filter (λ pos, pos.1 = i)).card = 3) ∧
    (∀ j : Fin 8, (white_squares.filter (λ pos, pos.2 = j)).card = 3) →
    ((Finset.univ.card // 2)^2 = 576) :=
sorry

end part_b_part_c_l61_61740


namespace sequence_general_term_l61_61555

noncomputable def a_n (n : ℕ) : ℝ :=
  sorry

-- The main statement
theorem sequence_general_term (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ m n : ℕ, |a (m + n) - a m - a n| ≤ 1 / (p * m + q * n)) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l61_61555


namespace distribute_diamonds_among_two_safes_l61_61388

theorem distribute_diamonds_among_two_safes (N : ℕ) :
  ∀ banker : ℕ, banker < 777 → ∃ s1 s2 : ℕ, s1 ≠ s2 ∧ s1 + s2 = N := sorry

end distribute_diamonds_among_two_safes_l61_61388


namespace value_of_frac_l61_61265

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l61_61265


namespace litter_patrol_total_l61_61464

theorem litter_patrol_total (glass_bottles : Nat) (aluminum_cans : Nat) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 :=
by
  sorry

end litter_patrol_total_l61_61464


namespace jason_initial_cards_l61_61581

/-- Jason initially had some Pokemon cards, Alyssa bought him 224 more, 
and now Jason has 900 Pokemon cards in total.
Prove that initially Jason had 676 Pokemon cards. -/
theorem jason_initial_cards (a b c : ℕ) (h_a : a = 224) (h_b : b = 900) (h_cond : b = a + 676) : 676 = c :=
by 
  sorry

end jason_initial_cards_l61_61581


namespace solve_for_x_l61_61146

theorem solve_for_x (x : ℤ) : 27 - 5 = 4 + x → x = 18 :=
by
  intro h
  sorry

end solve_for_x_l61_61146


namespace mady_balls_sum_of_digits_2010_l61_61445

def senary_sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 6 n).sum

theorem mady_balls_sum_of_digits_2010 :
  senary_sum_of_digits 2010 = 11 :=
by
  -- The proof is omitted as requested.
  sorry

end mady_balls_sum_of_digits_2010_l61_61445


namespace circle_passing_through_points_l61_61597

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61597


namespace circle_through_points_l61_61670

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61670


namespace measurement_units_correct_l61_61548

structure Measurement (A : Type) where
  value : A
  unit : String

def height_of_desk : Measurement ℕ := ⟨70, "centimeters"⟩
def weight_of_apple : Measurement ℕ := ⟨240, "grams"⟩
def duration_of_soccer_game : Measurement ℕ := ⟨90, "minutes"⟩
def dad_daily_work_duration : Measurement ℕ := ⟨8, "hours"⟩

theorem measurement_units_correct :
  height_of_desk.unit = "centimeters" ∧
  weight_of_apple.unit = "grams" ∧
  duration_of_soccer_game.unit = "minutes" ∧
  dad_daily_work_duration.unit = "hours" :=
by
  sorry

end measurement_units_correct_l61_61548


namespace unique_solution_to_functional_eq_l61_61092

theorem unique_solution_to_functional_eq :
  (∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2) :=
by
  sorry

end unique_solution_to_functional_eq_l61_61092


namespace triangle_area_bound_by_line_l61_61517

noncomputable def area_of_triangle : ℝ :=
let x_intercept := 3 in
let y_intercept := 9 in
1 / 2 * x_intercept * y_intercept

theorem triangle_area_bound_by_line (x y : ℝ) (h : 3 * x + y = 9) :
  area_of_triangle = 13.5 :=
sorry

end triangle_area_bound_by_line_l61_61517


namespace cannot_reach_eighth_vertex_l61_61429

def vertices : set (ℕ × ℕ × ℕ) := { (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0) }

def symmetric_point (a b : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2, 2 * b.3 - a.3)

theorem cannot_reach_eighth_vertex : ∀ (a b : ℕ × ℕ × ℕ), 
  a ∈ vertices → b ∈ vertices → 
  ¬ (symmetric_point a b = (1, 1, 1)) :=
by
  intros a b ha hb
  sorry

end cannot_reach_eighth_vertex_l61_61429


namespace group_age_analysis_l61_61381

theorem group_age_analysis (total_members : ℕ) (average_age : ℝ) (zero_age_members : ℕ) 
  (h1 : total_members = 50) (h2 : average_age = 5) (h3 : zero_age_members = 10) :
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  non_zero_members = 40 ∧ non_zero_average_age = 6.25 :=
by
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  have h_non_zero_members : non_zero_members = 40 := by sorry
  have h_non_zero_average_age : non_zero_average_age = 6.25 := by sorry
  exact ⟨h_non_zero_members, h_non_zero_average_age⟩

end group_age_analysis_l61_61381


namespace solve_inequality_system_l61_61002

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61002


namespace smallest_five_digit_divisible_by_15_32_54_l61_61347

theorem smallest_five_digit_divisible_by_15_32_54 : 
  ∃ n : ℤ, n >= 10000 ∧ n < 100000 ∧ (15 ∣ n) ∧ (32 ∣ n) ∧ (54 ∣ n) ∧ n = 17280 :=
  sorry

end smallest_five_digit_divisible_by_15_32_54_l61_61347


namespace sequence_2007th_number_l61_61175

-- Defining the sequence according to the given rule
def a (n : ℕ) : ℕ := 2 ^ n

theorem sequence_2007th_number : a 2007 = 2 ^ 2007 :=
by
  -- Proof is omitted
  sorry

end sequence_2007th_number_l61_61175


namespace solve_system_of_equations_l61_61031

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + x * y = 15) (h2 : x^2 + x * y = 10) :
  (x = 2 ∧ y = 3) ∨ (x = -2 ∧ y = -3) :=
sorry

end solve_system_of_equations_l61_61031


namespace range_of_m_l61_61554

open Real Set

variable (x m : ℝ)

def p (x : ℝ) := (x + 1) * (x - 1) ≤ 0
def q (x m : ℝ) := (x + 1) * (x - (3 * m - 1)) ≤ 0 ∧ m > 0

theorem range_of_m (hpsuffq : ∀ x, p x → q x m) (hqnotsuffp : ∃ x, q x m ∧ ¬ p x) : m > 2 / 3 := by
  sorry

end range_of_m_l61_61554


namespace max_n_m_sum_l61_61729

-- Definition of the function f
def f (x : ℝ) : ℝ := -x^2 + 4 * x

-- Statement of the problem
theorem max_n_m_sum {m n : ℝ} (h : n > m) (h_range : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4) : n + m = 7 :=
sorry

end max_n_m_sum_l61_61729


namespace range_of_t_l61_61228

open Real

noncomputable def f (x : ℝ) : ℝ := x / log x
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - exp 1 * x + exp 1 ^ 2)

theorem range_of_t :
  (∀ x > 1, ∀ t > 0, (t + 1) * g x ≤ t * f x)
  ↔ (∀ t > 0, t ≥ 1 / (exp 1 ^ 2 - 1)) :=
by
  sorry

end range_of_t_l61_61228


namespace range_of_a_l61_61413

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + 4

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = 0 → (f a x = 0 → x > 0)) ↔ a > 3 := sorry

end range_of_a_l61_61413


namespace sector_angle_l61_61564

theorem sector_angle (r l θ : ℝ) (h : 2 * r + l = π * r) : θ = π - 2 :=
sorry

end sector_angle_l61_61564


namespace inequalities_hold_l61_61405

variables {a b c : ℝ}

theorem inequalities_hold (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := 
  by
    sorry

end inequalities_hold_l61_61405


namespace solution_proof_l61_61097

noncomputable def f (n : ℕ) : ℝ := Real.logb 143 (n^2)

theorem solution_proof : f 7 + f 11 + f 13 = 2 + 2 * Real.logb 143 7 := by
  sorry

end solution_proof_l61_61097


namespace value_of_fraction_l61_61272

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l61_61272


namespace pair_d_same_function_l61_61062

theorem pair_d_same_function : ∀ x : ℝ, x = (x ^ 5) ^ (1 / 5) := 
by
  intro x
  sorry

end pair_d_same_function_l61_61062


namespace probability_f_has_zero_point_l61_61410

-- Defining the binomial distribution of X with parameters n=5 and p=1/2
noncomputable def X : ℕ → ℝ := λ k, ℙ (binomial 5 (1/2)) k

-- Function f(x) = x² + 4x + X
def f (x : ℝ) (X : ℝ) : ℝ := x^2 + 4*x + X

-- Event that f(x) has a zero point
def has_zero_point (X : ℕ) : Prop :=
  ∃ x : ℝ, f x (X : ℝ) = 0

-- Theorem statement: the probability that f(x) has a zero point given X ~ Binomial(5, 1/2) is 31/32
theorem probability_f_has_zero_point : ℙ (has_zero_point (X)) = 31/32 :=
sorry

end probability_f_has_zero_point_l61_61410


namespace functional_equations_l61_61798

noncomputable def f (x y z : ℝ) : ℝ := 
  (y + Real.sqrt(y ^ 2 + 4 * x * z)) / (2 * x)

theorem functional_equations (x y z t : ℝ) (k : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) (hk : k > 0) :
  (x * f x y z = z * f z y x) ∧
  (f x (t * y) (t^2 * z) = t * f x y z) ∧
  (f 1 k (k + 1) = k + 1) :=
by
  sorry

end functional_equations_l61_61798


namespace largest_rectangle_area_l61_61169

theorem largest_rectangle_area (x y : ℝ) (h1 : 2*x + 2*y = 60) (h2 : x ≥ 2*y) : ∃ A, A = x*y ∧ A ≤ 200 := by
  sorry

end largest_rectangle_area_l61_61169


namespace circle_passes_through_points_l61_61627

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61627


namespace inequality_system_solution_l61_61023

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_system_solution_l61_61023


namespace cousins_in_rooms_l61_61134

theorem cousins_in_rooms : 
  (number_of_ways : ℕ) (cousins : ℕ) (rooms : ℕ)
  (ways : ℕ) (is_valid_distribution : (ℕ → ℕ))
  (h_cousins : cousins = 5)
  (h_rooms : rooms = 4)
  (h_number_of_ways : ways = 67)
  :
  ∃ (distribute : ℕ → ℕ → ℕ), distribute cousins rooms = ways :=
sorry

end cousins_in_rooms_l61_61134


namespace no_valid_rook_placement_l61_61143

theorem no_valid_rook_placement :
  ∀ (r b g : ℕ), r + b + g = 50 →
  (2 * r ≤ b) →
  (2 * b ≤ g) →
  (2 * g ≤ r) →
  False :=
by
  -- Proof goes here
  sorry

end no_valid_rook_placement_l61_61143


namespace johns_change_l61_61292

theorem johns_change
  (num_oranges : ℤ) 
  (cost_per_orange : ℝ) 
  (amount_paid : ℝ)
  (h_oranges : num_oranges = 4)
  (h_cost : cost_per_orange = 0.75)
  (h_paid : amount_paid = 10.00) :
  amount_paid - num_oranges * cost_per_orange = 7.00 :=
by 
  rw [h_oranges, h_cost, h_paid]
  norm_num
  sorry

end johns_change_l61_61292


namespace factorization_of_polynomial_l61_61535

theorem factorization_of_polynomial (x : ℂ) :
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^2 - 3) * (x^4 + 3 * x^2 + 9) :=
by sorry

end factorization_of_polynomial_l61_61535


namespace solve_for_y_l61_61317

def solution (y : ℝ) : Prop :=
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = Real.pi / 4

theorem solve_for_y (y : ℝ) : solution y → y = 31 / 9 :=
by
  intro h
  sorry

end solve_for_y_l61_61317


namespace cafeteria_pies_l61_61925

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h_initial : initial_apples = 50)
  (h_handed_out : handed_out_apples = 5)
  (h_apples_per_pie : apples_per_pie = 5) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := 
by
  sorry

end cafeteria_pies_l61_61925


namespace pupils_correct_l61_61340

def totalPeople : ℕ := 676
def numberOfParents : ℕ := 22
def numberOfPupils : ℕ := totalPeople - numberOfParents

theorem pupils_correct :
  numberOfPupils = 654 := 
by
  sorry

end pupils_correct_l61_61340


namespace faster_current_takes_more_time_l61_61486

theorem faster_current_takes_more_time (v v1 v2 S : ℝ) (h_v1_gt_v2 : v1 > v2) :
  let t1 := (2 * S * v) / (v^2 - v1^2)
  let t2 := (2 * S * v) / (v^2 - v2^2)
  t1 > t2 :=
by
  sorry

end faster_current_takes_more_time_l61_61486


namespace range_of_m_min_value_of_7a_4b_l61_61562

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| - m ≥ 0) → m ≤ 2 :=
sorry

theorem min_value_of_7a_4b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_eq : 2 / (3 * a + b) + 1 / (a + 2 * b) = 2) : 7 * a + 4 * b ≥ 9 / 2 :=
sorry

end range_of_m_min_value_of_7a_4b_l61_61562


namespace solve_inequality_system_l61_61028

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61028


namespace equation_of_circle_through_three_points_l61_61677

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61677


namespace circle_passing_through_points_l61_61602

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61602


namespace added_amount_l61_61756

theorem added_amount (x y : ℕ) (h1 : x = 17) (h2 : 3 * (2 * x + y) = 117) : y = 5 :=
by
  sorry

end added_amount_l61_61756


namespace breadth_of_rectangular_plot_l61_61899

theorem breadth_of_rectangular_plot
  (b l : ℕ)
  (h1 : l = 3 * b)
  (h2 : l * b = 2028) :
  b = 26 :=
sorry

end breadth_of_rectangular_plot_l61_61899


namespace problem_l61_61982

theorem problem (a : ℕ) (b : ℚ) (c : ℤ) 
  (h1 : a = 1) 
  (h2 : b = 0) 
  (h3 : abs (c) = 6) :
  (a - b + c = (7 : ℤ)) ∨ (a - b + c = (-5 : ℤ)) := by
  sorry

end problem_l61_61982


namespace roger_earned_54_dollars_l61_61502

-- Definitions based on problem conditions
def lawns_had : ℕ := 14
def lawns_forgot : ℕ := 8
def earn_per_lawn : ℕ := 9

-- The number of lawns actually mowed
def lawns_mowed : ℕ := lawns_had - lawns_forgot

-- The amount of money earned
def money_earned : ℕ := lawns_mowed * earn_per_lawn

-- Proof statement: Roger actually earned 54 dollars
theorem roger_earned_54_dollars : money_earned = 54 := sorry

end roger_earned_54_dollars_l61_61502


namespace closest_ratio_of_adults_to_children_l61_61893

def total_fees (a c : ℕ) : ℕ := 20 * a + 10 * c
def adults_children_equation (a c : ℕ) : Prop := 2 * a + c = 160

theorem closest_ratio_of_adults_to_children :
  ∃ a c : ℕ, 
    total_fees a c = 1600 ∧
    a ≥ 1 ∧ c ≥ 1 ∧
    adults_children_equation a c ∧
    (∀ a' c' : ℕ, total_fees a' c' = 1600 ∧ 
        a' ≥ 1 ∧ c' ≥ 1 ∧ 
        adults_children_equation a' c' → 
        abs ((a : ℝ) / c - 1) ≤ abs ((a' : ℝ) / c' - 1)) :=
  sorry

end closest_ratio_of_adults_to_children_l61_61893


namespace polynomial_roots_identity_l61_61440

variables {c d : ℂ}

theorem polynomial_roots_identity (hc : c + d = 5) (hd : c * d = 6) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 :=
by {
  sorry
}

end polynomial_roots_identity_l61_61440


namespace problem1_problem2_problem3_l61_61804

def is_real (m : ℝ) : Prop := (m^2 - 3 * m) = 0
def is_complex (m : ℝ) : Prop := (m^2 - 3 * m) ≠ 0
def is_pure_imaginary (m : ℝ) : Prop := (m^2 - 5 * m + 6) = 0 ∧ (m^2 - 3 * m) ≠ 0

theorem problem1 (m : ℝ) : is_real m ↔ (m = 0 ∨ m = 3) :=
sorry

theorem problem2 (m : ℝ) : is_complex m ↔ (m ≠ 0 ∧ m ≠ 3) :=
sorry

theorem problem3 (m : ℝ) : is_pure_imaginary m ↔ (m = 2) :=
sorry

end problem1_problem2_problem3_l61_61804


namespace product_of_slopes_l61_61241

theorem product_of_slopes (m n : ℝ) (φ₁ φ₂ : ℝ) 
  (h1 : ∀ x, y = m * x)
  (h2 : ∀ x, y = n * x)
  (h3 : φ₁ = 2 * φ₂) 
  (h4 : m = 3 * n)
  (h5 : m ≠ 0 ∧ n ≠ 0)
  : m * n = 3 / 5 :=
sorry

end product_of_slopes_l61_61241


namespace man_age_difference_l61_61930

theorem man_age_difference (S M : ℕ) (h1 : S = 24) (h2 : M + 2 = 2 * (S + 2)) : M - S = 26 := by
  sorry

end man_age_difference_l61_61930


namespace two_bedroom_units_l61_61193

theorem two_bedroom_units (x y : ℕ) (h1 : x + y = 12) (h2 : 360 * x + 450 * y = 4950) : y = 7 :=
by
  sorry

end two_bedroom_units_l61_61193


namespace find_polynomial_h_l61_61591

theorem find_polynomial_h (f h : ℝ → ℝ) (hf : ∀ x, f x = x^2) (hh : ∀ x, f (h x) = 9 * x^2 + 6 * x + 1) : 
  (∀ x, h x = 3 * x + 1) ∨ (∀ x, h x = -3 * x - 1) :=
by
  sorry

end find_polynomial_h_l61_61591


namespace odd_function_has_specific_a_l61_61118

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
x / ((2 * x + 1) * (x - a))

theorem odd_function_has_specific_a :
  ∀ a, is_odd (f a) → a = 1 / 2 :=
by sorry

end odd_function_has_specific_a_l61_61118


namespace sequence_general_formula_l61_61227

theorem sequence_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+2) = 2 * a (n+1) / (2 + a (n+1))) :
  (a 1 = 1) → ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end sequence_general_formula_l61_61227


namespace circle_passes_through_points_l61_61623

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61623


namespace domain_of_y_eq_one_div_log_two_x_plus_one_l61_61034

open Set

noncomputable def domain_of_function := {x : ℝ | x > -1/2 ∧ x ≠ 0}

theorem domain_of_y_eq_one_div_log_two_x_plus_one :
  {x : ℝ | (2 * x + 1 > 0) ∧ (Real.log (2 * x + 1) ≠ 0)} = domain_of_function :=
begin
  sorry
end

end domain_of_y_eq_one_div_log_two_x_plus_one_l61_61034


namespace find_x9_y9_l61_61242

theorem find_x9_y9 (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : x^9 + y^9 = 343 :=
by
  sorry

end find_x9_y9_l61_61242


namespace fractional_addition_l61_61791

theorem fractional_addition : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by
  sorry

end fractional_addition_l61_61791


namespace eq_30_apples_n_7_babies_min_3_max_6_l61_61337

theorem eq_30_apples_n_7_babies_min_3_max_6 (x : ℕ) 
    (h1 : 30 = x + 7 * 4)
    (h2 : 21 ≤ 30) 
    (h3 : 30 ≤ 42) 
    (h4 : x = 2) :
  x = 2 :=
by
  sorry

end eq_30_apples_n_7_babies_min_3_max_6_l61_61337


namespace problem1_problem2_l61_61772

-- Problem 1: Prove that 3 * sqrt(20) - sqrt(45) + sqrt(1 / 5) = (16 * sqrt(5)) / 5
theorem problem1 : 3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1 / 5) = (16 * Real.sqrt 5) / 5 := 
sorry

-- Problem 2: Prove that (sqrt(6) - 2 * sqrt(3))^2 - (2 * sqrt(5) + sqrt(2)) * (2 * sqrt(5) - sqrt(2)) = -12 * sqrt(2)
theorem problem2 : (Real.sqrt 6 - 2 * Real.sqrt 3) ^ 2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2 := 
sorry

end problem1_problem2_l61_61772


namespace harry_weekly_earnings_l61_61108

def dogs_walked_per_day : Nat → Nat
| 1 => 7  -- Monday
| 2 => 12 -- Tuesday
| 3 => 7  -- Wednesday
| 4 => 9  -- Thursday
| 5 => 7  -- Friday
| _ => 0  -- Other days (not relevant for this problem)

def payment_per_dog : Nat := 5

def daily_earnings (day : Nat) : Nat :=
  dogs_walked_per_day day * payment_per_dog

def total_weekly_earnings : Nat :=
  (daily_earnings 1) + (daily_earnings 2) + (daily_earnings 3) +
  (daily_earnings 4) + (daily_earnings 5)

theorem harry_weekly_earnings : total_weekly_earnings = 210 :=
by
  sorry

end harry_weekly_earnings_l61_61108


namespace solve_inequality_system_l61_61025

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61025


namespace base_satisfying_eq_l61_61094

theorem base_satisfying_eq : ∃ a : ℕ, (11 < a) ∧ (293 * a^2 + 9 * a + 3 + (4 * a^2 + 6 * a + 8) = 7 * a^2 + 3 * a + 11) ∧ (a = 12) :=
by
  sorry

end base_satisfying_eq_l61_61094


namespace max_k_range_minus_five_l61_61800

theorem max_k_range_minus_five :
  ∃ k : ℝ, (∀ x : ℝ, x^2 + 5 * x + k = -5) → k = 5 / 4 :=
by
  sorry

end max_k_range_minus_five_l61_61800


namespace interest_difference_l61_61757

-- Conditions
def principal : ℕ := 350
def rate : ℕ := 4
def time : ℕ := 8

-- Question rewritten as a statement to prove
theorem interest_difference :
  let SI := (principal * rate * time) / 100 
  let difference := principal - SI
  difference = 238 := by
  sorry

end interest_difference_l61_61757


namespace same_sign_abc_l61_61553
open Classical

theorem same_sign_abc (a b c : ℝ) (h1 : (b / a) * (c / a) > 1) (h2 : (b / a) + (c / a) ≥ -2) : 
  (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0) :=
sorry

end same_sign_abc_l61_61553


namespace joey_route_length_l61_61125

-- Definitions
def time_one_way : ℝ := 1
def avg_speed : ℝ := 8
def return_speed : ℝ := 12

-- Theorem to prove
theorem joey_route_length : (∃ D : ℝ, D = 6 ∧ (D / avg_speed = time_one_way + D / return_speed)) :=
sorry

end joey_route_length_l61_61125


namespace cheapest_option_is_1_l61_61243

-- Definitions of the costs and amounts
def cost_train_ticket : ℝ := 200
def berries_collected : ℝ := 5
def cost_per_kg_berries_market : ℝ := 150
def cost_per_kg_sugar : ℝ := 54
def jam_production_rate : ℝ := 1.5
def cost_per_kg_jam_market : ℝ := 220

-- Calculations for cost per kg of jam for each option
def cost_per_kg_berries_collect := cost_train_ticket / berries_collected
def cost_per_kg_jam_collect := cost_per_kg_berries_collect + cost_per_kg_sugar
def cost_for_1_5_kg_jam_collect := cost_per_kg_jam_collect
def cost_for_1_5_kg_jam_market := cost_per_kg_berries_market + cost_per_kg_sugar
def cost_for_1_5_kg_jam_ready := cost_per_kg_jam_market * jam_production_rate

-- Proof that Option 1 is the cheapest
theorem cheapest_option_is_1 : (cost_for_1_5_kg_jam_collect < cost_for_1_5_kg_jam_market ∧ cost_for_1_5_kg_jam_collect < cost_for_1_5_kg_jam_ready) :=
by
  sorry

end cheapest_option_is_1_l61_61243


namespace smallest_non_factor_product_of_factors_of_72_l61_61734

theorem smallest_non_factor_product_of_factors_of_72 : 
  ∃ x y : ℕ, x ≠ y ∧ x * y ∣ 72 ∧ ¬ (x * y ∣ 72) ∧ x * y = 32 := 
by
  sorry

end smallest_non_factor_product_of_factors_of_72_l61_61734


namespace number_of_integer_values_l61_61256

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l61_61256


namespace probability_tile_in_PAIR_l61_61391

theorem probability_tile_in_PAIR :
  let tiles := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
  let pair_letters := ['P', 'A', 'I', 'R']
  let matching_counts := (sum ([1, 1, 2, 1] : List ℕ))
  matching_counts.fst = 5
  let total_tiles := 12
  (matching_counts.toRational / total_tiles.toRational) = (5 / 12) :=
by sorry

end probability_tile_in_PAIR_l61_61391


namespace greatest_possible_value_of_y_l61_61147

theorem greatest_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 :=
sorry

end greatest_possible_value_of_y_l61_61147


namespace inequality_solution_l61_61012

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l61_61012


namespace segment_length_is_13_l61_61110

def point := (ℝ × ℝ)

def p1 : point := (2, 3)
def p2 : point := (7, 15)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem segment_length_is_13 : distance p1 p2 = 13 := by
  sorry

end segment_length_is_13_l61_61110


namespace cousins_room_distributions_l61_61137

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l61_61137


namespace xy_square_diff_l61_61851

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61851


namespace solve_inequality_system_l61_61006

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61006


namespace fraction_calculation_l61_61942

noncomputable def improper_frac_1 : ℚ := 21 / 8
noncomputable def improper_frac_2 : ℚ := 33 / 14
noncomputable def improper_frac_3 : ℚ := 37 / 12
noncomputable def improper_frac_4 : ℚ := 35 / 8
noncomputable def improper_frac_5 : ℚ := 179 / 9

theorem fraction_calculation :
  (improper_frac_1 - (2 / 3) * improper_frac_2) / ((improper_frac_3 + improper_frac_4) / improper_frac_5) = 59 / 21 :=
by
  sorry

end fraction_calculation_l61_61942


namespace harmony_implication_at_least_N_plus_1_zero_l61_61979

noncomputable def is_harmony (A B : ℕ → ℕ) (i : ℕ) : Prop :=
  A i = (1 / (2 * B i + 1)) * (Finset.range (2 * B i + 1)).sum (fun s => A (i + s - B i))

theorem harmony_implication_at_least_N_plus_1_zero {N : ℕ} (A B : ℕ → ℕ)
  (hN : N ≥ 2) 
  (h_nonneg_A : ∀ i, 0 ≤ A i)
  (h_nonneg_B : ∀ i, 0 ≤ B i)
  (h_periodic_A : ∀ i, A i = A ((i % N) + 1))
  (h_periodic_B : ∀ i, B i = B ((i % N) + 1))
  (h_harmony_AB : ∀ i, is_harmony A B i)
  (h_harmony_BA : ∀ i, is_harmony B A i)
  (h_not_constant_A : ¬ ∀ i j, A i = A j)
  (h_not_constant_B : ¬ ∀ i j, B i = B j) :
  Finset.card (Finset.filter (fun i => A i = 0 ∨ B i = 0) (Finset.range (N * 2))) ≥ N + 1 := by
  sorry

end harmony_implication_at_least_N_plus_1_zero_l61_61979


namespace find_prime_b_l61_61815

-- Define the polynomial function f
def f (n a : ℕ) : ℕ := n^3 - 4 * a * n^2 - 12 * n + 144

-- Define b as a prime number
def b (n : ℕ) (a : ℕ) : ℕ := f n a

-- Theorem statement
theorem find_prime_b (n : ℕ) (a : ℕ) (h : n = 7) (ha : a = 2) (hb : ∃ p : ℕ, Nat.Prime p ∧ p = b n a) :
  b n a = 11 :=
by
  sorry

end find_prime_b_l61_61815


namespace circle_through_points_l61_61665

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61665


namespace volumes_of_rotated_solids_l61_61991

theorem volumes_of_rotated_solids
  (π : ℝ)
  (b c a : ℝ)
  (h₁ : a^2 = b^2 + c^2)
  (v v₁ v₂ : ℝ)
  (hv : v = (1/3) * π * (b^2 * c^2) / a)
  (hv₁ : v₁ = (1/3) * π * c^2 * b)
  (hv₂ : v₂ = (1/3) * π * b^2 * c) :
  (1 / v^2) = (1 / v₁^2) + (1 / v₂^2) := 
by sorry

end volumes_of_rotated_solids_l61_61991


namespace rectangle_pentagon_ratio_l61_61511

theorem rectangle_pentagon_ratio
  (l w p : ℝ)
  (h1 : l = 2 * w)
  (h2 : 2 * (l + w) = 30)
  (h3 : 5 * p = 30) :
  l / p = 5 / 3 :=
by
  sorry

end rectangle_pentagon_ratio_l61_61511


namespace triangle_area_l61_61519

-- Define the line equation 3x + y = 9
def line_eq (x y : ℝ) : Prop := 3 * x + y = 9

-- Define the triangle bounded by the coordinate axes and the line
def bounded_triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ 0 ≤ y ∧ y ≤ 9) ∨  -- y-axis segment
  (y = 0 ∧ 0 ≤ x ∧ x ≤ 3) ∨  -- x-axis segment
  (3 * x + y = 9 ∧ 0 ≤ x ∧ 0 ≤ y)  -- line segment

-- The area of the triangle formed by the coordinate axes and the line 3x + y = 9
theorem triangle_area : 
  let area := 1/2 * 3 * 9 in -- The calculation of the area
  area = 13.5 := 
by
  sorry

end triangle_area_l61_61519


namespace find_f_2010_l61_61331

noncomputable def f : ℕ → ℤ := sorry

theorem find_f_2010 (f_prop : ∀ {a b n : ℕ}, a + b = 3 * 2^n → f a + f b = 2 * n^2) :
  f 2010 = 193 :=
sorry

end find_f_2010_l61_61331


namespace exponent_equivalence_l61_61552

theorem exponent_equivalence (a b : ℕ) (m n : ℕ) (m_pos : 0 < m) (n_pos : 0 < n) (h1 : 9 ^ m = a) (h2 : 3 ^ n = b) : 
  3 ^ (2 * m + 4 * n) = a * b ^ 4 := 
by 
  sorry

end exponent_equivalence_l61_61552


namespace smallest_divisor_subtracted_l61_61958

theorem smallest_divisor_subtracted (a b d : ℕ) (h1: a = 899830) (h2: b = 6) (h3: a - b = 899824) (h4 : 6 < d) 
(h5 : d ∣ (a - b)) : d = 8 :=
by
  sorry

end smallest_divisor_subtracted_l61_61958


namespace negate_prop_l61_61166

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l61_61166


namespace max_board_size_l61_61295

theorem max_board_size : ∀ (n : ℕ), 
  (∃ (board : Fin n → Fin n → Prop),
    ∀ i j k l : Fin n,
      (i ≠ k ∧ j ≠ l) → board i j ≠ board k l) ↔ n ≤ 4 :=
by sorry

end max_board_size_l61_61295


namespace sqrt_floor_8_integer_count_l61_61252

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l61_61252


namespace identity_x_squared_minus_y_squared_l61_61842

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61842


namespace distance_between_foci_of_ellipse_l61_61066

theorem distance_between_foci_of_ellipse :
  let h := 6
  let k := 3
  let a := 6
  let b := 3
  let c := Real.sqrt (a^2 - b^2)
  in 2 * c = 6 * Real.sqrt 3 := by
  sorry

end distance_between_foci_of_ellipse_l61_61066


namespace circle_passing_through_points_l61_61630

open Real

theorem circle_passing_through_points :
  ∃ D E F : ℝ, (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0) ∧
  x^2 + y^2 - 4 * x - 6 * y = 0 :=
sorry

end circle_passing_through_points_l61_61630


namespace find_scalars_l61_61730

open Real

-- Definitions for the given vectors
def a : ℝ × ℝ × ℝ := (2, 2, 2)
def b : ℝ × ℝ × ℝ := (3, -4, 1)
def c : ℝ × ℝ × ℝ := (5, 1, -6)
def x : ℝ × ℝ × ℝ := (-8, 14, 6)

-- Dot product definition
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Norm squared definition
def norm_squared (v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v v

-- Orthogonality condition (a ⊥ b, a ⊥ c, b ⊥ c)
def orthogonal (v w : ℝ × ℝ × ℝ) : Prop :=
  dot_product v w = 0

-- Main statement
theorem find_scalars :
  orthogonal a b ∧ orthogonal a c ∧ orthogonal b c
  ∧ (∃ (p q r : ℝ), x = (p * a.1 + q * b.1 + r * c.1,
                         p * a.2 + q * b.2 + r * c.2,
                         p * a.3 + q * b.3 + r * c.3))
  ∧ ∃ (p q r : ℝ), p = 2 ∧ q = -37 / 13 ∧ r = -1 :=
by
  sorry

end find_scalars_l61_61730


namespace gcd_factorials_l61_61967

open Nat

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials (h : ∀ n, 0 < n → factorial n = n * factorial (n - 1)) :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 :=
sorry

end gcd_factorials_l61_61967


namespace necessary_but_not_sufficient_l61_61996

theorem necessary_but_not_sufficient (a : ℝ) : (a ≠ 1) → (a^2 ≠ 1) → (a ≠ 1) ∧ ¬((a ≠ 1) → (a^2 ≠ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l61_61996


namespace sum_of_coefficients_l61_61471

theorem sum_of_coefficients (A B C : ℤ)
  (h : ∀ x, x^3 + A * x^2 + B * x + C = (x + 3) * x * (x - 3))
  : A + B + C = -9 :=
sorry

end sum_of_coefficients_l61_61471


namespace calculation_is_correct_l61_61079

theorem calculation_is_correct :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := 
by
  sorry

end calculation_is_correct_l61_61079


namespace oven_clock_actual_time_l61_61285

theorem oven_clock_actual_time :
  ∀ (h : ℕ), (oven_time : h = 10) →
  (oven_gains : ℕ) = 8 →
  (initial_time : ℕ) = 18 →          
  (initial_wall_time : ℕ) = 18 →
  (wall_time_after_one_hour : ℕ) = 19 →
  (oven_time_after_one_hour : ℕ) = 19 + 8/60 →
  ℕ := sorry

end oven_clock_actual_time_l61_61285


namespace circle_passing_through_points_eqn_l61_61655

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61655


namespace circle_passes_through_points_l61_61620

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end circle_passes_through_points_l61_61620


namespace ellipse_focus_eccentricity_l61_61409

theorem ellipse_focus_eccentricity (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 2) + (y^2 / m) = 1 → y = 0 ∨ x = 0) ∧
  (∀ e : ℝ, e = 1 / 2) →
  m = 3 / 2 :=
sorry

end ellipse_focus_eccentricity_l61_61409


namespace income_expenditure_ratio_l61_61472

theorem income_expenditure_ratio (I E S : ℝ) (h1 : I = 20000) (h2 : S = 4000) (h3 : S = I - E) :
    I / E = 5 / 4 :=
sorry

end income_expenditure_ratio_l61_61472


namespace calculate_discount_percentage_l61_61771

theorem calculate_discount_percentage :
  ∃ (x : ℝ), (∀ (P S : ℝ),
    (S = 439.99999999999966) →
    (S = 1.10 * P) →
    (1.30 * (1 - x / 100) * P = S + 28) →
    x = 10) :=
sorry

end calculate_discount_percentage_l61_61771


namespace percent_of_y_l61_61050

theorem percent_of_y (y : ℝ) (hy : y > 0) : (8 * y) / 20 + (3 * y) / 10 = 0.7 * y :=
by
  sorry

end percent_of_y_l61_61050


namespace cost_of_each_top_l61_61072

theorem cost_of_each_top
  (total_spent : ℝ)
  (num_shorts : ℕ)
  (price_per_short : ℝ)
  (num_shoes : ℕ)
  (price_per_shoe : ℝ)
  (num_tops : ℕ)
  (total_cost_shorts : ℝ)
  (total_cost_shoes : ℝ)
  (amount_spent_on_tops : ℝ)
  (cost_per_top : ℝ) :
  total_spent = 75 →
  num_shorts = 5 →
  price_per_short = 7 →
  num_shoes = 2 →
  price_per_shoe = 10 →
  num_tops = 4 →
  total_cost_shorts = num_shorts * price_per_short →
  total_cost_shoes = num_shoes * price_per_shoe →
  amount_spent_on_tops = total_spent - (total_cost_shorts + total_cost_shoes) →
  cost_per_top = amount_spent_on_tops / num_tops →
  cost_per_top = 5 :=
by
  sorry

end cost_of_each_top_l61_61072


namespace problem_solution_l61_61997

theorem problem_solution :
  ∀ x y : ℝ, 9 * y^2 + 6 * x * y + x + 12 = 0 → (x ≤ -3 ∨ x ≥ 4) :=
  sorry

end problem_solution_l61_61997


namespace sum_of_possible_values_l61_61245

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 5) = 20) : x = -2 ∨ x = 7 :=
sorry

end sum_of_possible_values_l61_61245


namespace sum_of_terms_l61_61749

def geometric_sequence (a b c d : ℝ) :=
  ∃ q : ℝ, a = b / q ∧ c = b * q ∧ d = c * q

def symmetric_sequence_of_length_7 (s : Fin 8 → ℝ) :=
  ∀ i : Fin 8, s i = s (Fin.mk (7 - i) sorry)

def sequence_conditions (s : Fin 8 → ℝ) :=
  symmetric_sequence_of_length_7 s ∧
  geometric_sequence (s ⟨1,sorry⟩) (s ⟨2,sorry⟩) (s ⟨3,sorry⟩) (s ⟨4,sorry⟩) ∧
  s ⟨1,sorry⟩ = 2 ∧
  s ⟨3,sorry⟩ = 8

theorem sum_of_terms (s : Fin 8 → ℝ) (h : sequence_conditions s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = 44 ∨
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = -4 :=
sorry

end sum_of_terms_l61_61749


namespace Mitch_saved_amount_l61_61883

theorem Mitch_saved_amount :
  let boat_cost_per_foot := 1500
  let license_and_registration := 500
  let docking_fees := 3 * 500
  let longest_boat_length := 12
  let total_license_and_fees := license_and_registration + docking_fees
  let total_boat_cost := boat_cost_per_foot * longest_boat_length
  let total_saved := total_boat_cost + total_license_and_fees
  total_saved = 20000 :=
by
  sorry

end Mitch_saved_amount_l61_61883


namespace inequalities_hold_l61_61403

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ (b - a) / c > 0 ∧ (a - c) / (a * c) < 0 :=
by 
  sorry

end inequalities_hold_l61_61403


namespace find_a_l61_61857

theorem find_a (a x : ℝ) (h1 : 2 * (x - 1) - 6 = 0) (h2 : 1 - (3 * a - x) / 3 = 0) (h3 : x = 4) : a = -1 / 3 :=
by
  sorry

end find_a_l61_61857


namespace binary_to_decimal_l61_61777

theorem binary_to_decimal : 
  let binary_number := [1, 1, 0, 1, 1] in
  let weights := [2^4, 2^3, 2^2, 2^1, 2^0] in
  (List.zipWith (λ digit weight => digit * weight) binary_number weights).sum = 27 := by
  sorry

end binary_to_decimal_l61_61777


namespace independence_necessary_and_sufficient_l61_61401

variables {Ω : Type*} {P : ProbabilityTheory ℕ}
variables (A B : Set Ω)
variable [MeasurableSpace Ω]
variable [ProbabilityMeasure P]

theorem independence_necessary_and_sufficient (hA : 0 < P(A)) (hB : 0 < P(B)) :
  (P(A ∩ B) = P(A) * P(B)) ↔ (A ∩ B = P(A) * P(B)) := 
sorry

end independence_necessary_and_sufficient_l61_61401


namespace binary_to_decimal_l61_61783

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4) = 27 :=
by
  sorry

end binary_to_decimal_l61_61783


namespace ratio_of_speeds_l61_61342

variables (v_A v_B v_C : ℝ)

-- Conditions definitions
def condition1 : Prop := v_A - v_B = 5
def condition2 : Prop := v_A + v_C = 15

-- Theorem statement (the mathematically equivalent proof problem)
theorem ratio_of_speeds (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_C) : (v_A / v_B) = 3 :=
sorry

end ratio_of_speeds_l61_61342


namespace experimental_fertilizer_height_is_correct_l61_61290

/-- Define the static heights and percentages for each plant's growth conditions. -/
def control_plant_height : ℝ := 36
def bone_meal_multiplier : ℝ := 1.25
def cow_manure_multiplier : ℝ := 2
def experimental_fertilizer_multiplier : ℝ := 1.5

/-- Define each plant's height based on the given multipliers and conditions. -/
def bone_meal_plant_height : ℝ := bone_meal_multiplier * control_plant_height
def cow_manure_plant_height : ℝ := cow_manure_multiplier * bone_meal_plant_height
def experimental_fertilizer_plant_height : ℝ := experimental_fertilizer_multiplier * cow_manure_plant_height

/-- Proof that the height of the experimental fertilizer plant is 135 inches. -/
theorem experimental_fertilizer_height_is_correct :
  experimental_fertilizer_plant_height = 135 := by
    sorry

end experimental_fertilizer_height_is_correct_l61_61290


namespace border_area_is_198_l61_61366

-- We define the dimensions of the picture and the border width
def picture_height : ℝ := 12
def picture_width : ℝ := 15
def border_width : ℝ := 3

-- We compute the entire framed height and width
def framed_height : ℝ := picture_height + 2 * border_width
def framed_width : ℝ := picture_width + 2 * border_width

-- We compute the area of the picture and framed area
def picture_area : ℝ := picture_height * picture_width
def framed_area : ℝ := framed_height * framed_width

-- We compute the area of the border
def border_area : ℝ := framed_area - picture_area

-- Now we pose the theorem to prove the area of the border is 198 square inches
theorem border_area_is_198 : border_area = 198 := by
  sorry

end border_area_is_198_l61_61366


namespace distinct_students_27_l61_61076

variable (students_euler : ℕ) (students_fibonacci : ℕ) (students_gauss : ℕ) (overlap_euler_fibonacci : ℕ)

-- Conditions
def conditions : Prop := 
  students_euler = 12 ∧ 
  students_fibonacci = 10 ∧ 
  students_gauss = 11 ∧ 
  overlap_euler_fibonacci = 3

-- Question and correct answer
def distinct_students (students_euler students_fibonacci students_gauss overlap_euler_fibonacci : ℕ) : ℕ :=
  (students_euler + students_fibonacci + students_gauss) - overlap_euler_fibonacci

theorem distinct_students_27 : conditions students_euler students_fibonacci students_gauss overlap_euler_fibonacci →
  distinct_students students_euler students_fibonacci students_gauss overlap_euler_fibonacci = 27 :=
by
  sorry

end distinct_students_27_l61_61076


namespace expression_evaluation_l61_61537

theorem expression_evaluation :
  2^3 + 4 * 5 - Real.sqrt 9 + (3^2 * 2) / 3 = 31 := sorry

end expression_evaluation_l61_61537


namespace circle_passing_through_points_eqn_l61_61657

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61657


namespace mutually_exclusive_pairs_l61_61970

-- Define the events based on the conditions
def event_two_red_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 2 ∧ drawn.count "white" = 1)

def event_one_red_two_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 1 ∧ drawn.count "white" = 2)

def event_three_red (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "red" = 3

def event_at_least_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ 1 ≤ drawn.count "white"

def event_three_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "white" = 3

-- Define mutually exclusive property
def mutually_exclusive (A B : List String → List String → Prop) (bag : List String) : Prop :=
  ∀ drawn, A bag drawn → ¬ B bag drawn

-- Define the main theorem statement
theorem mutually_exclusive_pairs (bag : List String) (condition : bag = ["red", "red", "red", "red", "red", "white", "white", "white", "white", "white"]) :
  mutually_exclusive event_three_red event_at_least_one_white bag ∧
  mutually_exclusive event_three_red event_three_white bag :=
by
  sorry

end mutually_exclusive_pairs_l61_61970


namespace equation_of_circle_ABC_l61_61612

def point := (ℝ × ℝ)

noncomputable def circle_eq_pass_through_points (A B C : point) (D E F : ℝ) : Prop :=
  (A.1^2 + A.2^2 + D * A.1 + E * A.2 + F = 0) ∧
  (B.1^2 + B.2^2 + D * B.1 + E * B.2 + F = 0) ∧
  (C.1^2 + C.2^2 + D * C.1 + E * C.2 + F = 0) 

theorem equation_of_circle_ABC 
  (A B C : point) 
  (D E F : ℝ)
  (h : circle_eq_pass_through_points A B C D E F) 
  : A = (0, 0) ∨ A = (4, 0) ∨ A = (-1, 1) ∨ A = (4, 2) →
    B = (0, 0) ∨ B = (4, 0) ∨ B = (-1, 1) ∨ B = (4, 2) →
    C = (0, 0) ∨ C = (4, 0) ∨ C = (-1, 1) ∨ C = (4, 2) →
    D = -4 ∨ D = -4 ∨ D = -(8 / 3) ∨ D = -(16 / 5) →
    E = -6 ∨ E = -2 ∨ E = -(14 / 3) ∨ E = -2 →
    F = 0 ∨ F = 0 ∨ F = 0 ∨ F = -(16 / 5) :=
by 
  intros A_condition B_condition C_condition D_condition E_condition F_condition
  sorry

end equation_of_circle_ABC_l61_61612


namespace totalCorrectQuestions_l61_61450

-- Definitions for the conditions
def mathQuestions : ℕ := 40
def mathCorrectPercentage : ℕ := 75
def englishQuestions : ℕ := 50
def englishCorrectPercentage : ℕ := 98

-- Function to calculate the number of correctly answered questions
def correctQuestions (totalQuestions : ℕ) (percentage : ℕ) : ℕ :=
  (percentage * totalQuestions) / 100

-- Main theorem to prove the total number of correct questions
theorem totalCorrectQuestions : 
  correctQuestions mathQuestions mathCorrectPercentage +
  correctQuestions englishQuestions englishCorrectPercentage = 79 :=
by
  sorry

end totalCorrectQuestions_l61_61450


namespace remainder_17_plus_x_mod_31_l61_61129

theorem remainder_17_plus_x_mod_31 {x : ℕ} (h : 13 * x ≡ 3 [MOD 31]) : (17 + x) % 31 = 22 := 
sorry

end remainder_17_plus_x_mod_31_l61_61129


namespace find_number_l61_61503

theorem find_number (x : ℕ) (h : 695 - 329 = x - 254) : x = 620 :=
sorry

end find_number_l61_61503


namespace find_c_for_degree_3_l61_61945

theorem find_c_for_degree_3 (c : ℚ) :
  let f : ℚ[X] := 1 - 12 * X + 3 * X^2 - 4 * X^3 + 5 * X^4
  let g : ℚ[X] := 3 - 2 * X + X^2 - 6 * X^3 + 11 * X^4
  (degree (f + c * g) = 3) ↔ c = (-5/11 : ℚ) :=
by sorry

end find_c_for_degree_3_l61_61945


namespace total_cats_l61_61363

def num_white_cats : Nat := 2
def num_black_cats : Nat := 10
def num_gray_cats : Nat := 3

theorem total_cats : (num_white_cats + num_black_cats + num_gray_cats) = 15 :=
by
  sorry

end total_cats_l61_61363


namespace max_police_officers_needed_l61_61880

theorem max_police_officers_needed : 
  let streets := 10
  let non_parallel := true
  let curved_streets := 2
  let additional_intersections_per_curved := 3 
  streets = 10 ∧ 
  non_parallel = true ∧ 
  curved_streets = 2 ∧ 
  additional_intersections_per_curved = 3 → 
  ( (streets * (streets - 1) / 2) + (curved_streets * additional_intersections_per_curved) ) = 51 :=
by
  intros
  sorry

end max_police_officers_needed_l61_61880


namespace second_train_length_l61_61913

noncomputable def length_of_second_train (speed1_kmph speed2_kmph : ℝ) (time_seconds : ℝ) (length1_meters : ℝ) : ℝ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed_mps := speed1_mps + speed2_mps
  let distance := relative_speed_mps * time_seconds
  distance - length1_meters

theorem second_train_length :
  length_of_second_train 72 18 17.998560115190784 200 = 250 :=
by
  sorry

end second_train_length_l61_61913


namespace average_age_with_teacher_l61_61465

theorem average_age_with_teacher (A : ℕ) (h : 21 * 16 = 20 * A + 36) : A = 15 := by
  sorry

end average_age_with_teacher_l61_61465


namespace circle_passing_through_points_l61_61598

theorem circle_passing_through_points :
  ∃ D E F : ℝ, 
    (∀ x y : ℝ, 
      (x = 0 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = 4 ∧ y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
      (x = -1 ∧ y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 → x^2 + y^2 - 4 * x - 6 * y = 0) :=
sorry

end circle_passing_through_points_l61_61598


namespace display_total_cans_l61_61119

def row_num_cans (row : ℕ) : ℕ :=
  if row < 7 then 19 - 3 * (7 - row)
  else 19 + 3 * (row - 7)

def total_cans : ℕ :=
  List.sum (List.map row_num_cans (List.range 10))

theorem display_total_cans : total_cans = 145 := 
  sorry

end display_total_cans_l61_61119


namespace xy_square_diff_l61_61845

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end xy_square_diff_l61_61845


namespace evaluate_expression_l61_61546

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end evaluate_expression_l61_61546


namespace solve_inequality_system_l61_61008

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61008


namespace expression_value_l61_61208

theorem expression_value : 
  (2 ^ 1501 + 5 ^ 1502) ^ 2 - (2 ^ 1501 - 5 ^ 1502) ^ 2 = 20 * 10 ^ 1501 := 
by
  sorry

end expression_value_l61_61208


namespace range_of_a_l61_61153

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l61_61153


namespace kate_hair_length_l61_61299

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l61_61299


namespace correct_population_statement_l61_61912

def correct_statement :=
  "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

def sample_size : ℕ := 500

def is_correct (statement : String) : Prop :=
  statement = correct_statement

theorem correct_population_statement (scores : Fin 500 → ℝ) :
  is_correct "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population." :=
by
  sorry

end correct_population_statement_l61_61912


namespace paper_clips_in_two_cases_l61_61747

-- Define the conditions
variables (c b : ℕ)

-- Define the theorem statement
theorem paper_clips_in_two_cases (c b : ℕ) : 
    2 * c * b * 400 = 2 * c * b * 400 :=
by
  sorry

end paper_clips_in_two_cases_l61_61747


namespace cost_per_top_l61_61070
   
   theorem cost_per_top 
     (total_spent : ℕ) 
     (short_pairs : ℕ) 
     (short_cost_per_pair : ℕ) 
     (shoe_pairs : ℕ) 
     (shoe_cost_per_pair : ℕ) 
     (top_count : ℕ)
     (remaining_cost : ℕ)
     (total_short_cost : ℕ) 
     (total_shoe_cost : ℕ) 
     (total_short_shoe_cost : ℕ)
     (total_top_cost : ℕ) :
     total_spent = 75 →
     short_pairs = 5 →
     short_cost_per_pair = 7 →
     shoe_pairs = 2 →
     shoe_cost_per_pair = 10 →
     top_count = 4 →
     total_short_cost = short_pairs * short_cost_per_pair →
     total_shoe_cost = shoe_pairs * shoe_cost_per_pair →
     total_short_shoe_cost = total_short_cost + total_shoe_cost →
     total_top_cost = total_spent - total_short_shoe_cost →
     remaining_cost = total_top_cost / top_count →
     remaining_cost = 5 :=
   by
     intros
     sorry
   
end cost_per_top_l61_61070


namespace range_of_a_l61_61567

variable (a : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 4

theorem range_of_a :
  (∀ x : ℝ, f a x < 0) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l61_61567


namespace empty_bidon_weight_l61_61750

theorem empty_bidon_weight (B M : ℝ) 
  (h1 : B + M = 34) 
  (h2 : B + M / 2 = 17.5) : 
  B = 1 := 
by {
  -- The proof steps would go here, but we just add sorry
  sorry
}

end empty_bidon_weight_l61_61750


namespace sequence_general_formula_l61_61235

-- Define the sequence S_n and the initial conditions
def S (n : ℕ) : ℕ := 3^(n + 1) - 1

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 8 else 2 * 3^n

-- Theorem statement proving the general formula
theorem sequence_general_formula (n : ℕ) : 
  a n = if n = 1 then 8 else 2 * 3^n := by
  -- This is where the proof would go
  sorry

end sequence_general_formula_l61_61235


namespace time_at_2010_minutes_after_3pm_is_930pm_l61_61322

def time_after_2010_minutes (current_time : Nat) (minutes_passed : Nat) : Nat :=
  sorry

theorem time_at_2010_minutes_after_3pm_is_930pm :
  time_after_2010_minutes 900 2010 = 1290 :=
by
  sorry

end time_at_2010_minutes_after_3pm_is_930pm_l61_61322


namespace max_a_plus_b_l61_61874

/-- Given real numbers a and b such that 5a + 3b <= 11 and 3a + 6b <= 12,
    the largest possible value of a + b is 23/9. -/
theorem max_a_plus_b (a b : ℝ) (h1 : 5 * a + 3 * b ≤ 11) (h2 : 3 * a + 6 * b ≤ 12) :
  a + b ≤ 23 / 9 :=
sorry

end max_a_plus_b_l61_61874


namespace correct_inequality_l61_61876

variable (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b)

theorem correct_inequality : (1 / (a * b^2)) < (1 / (a^2 * b)) :=
by
  sorry

end correct_inequality_l61_61876


namespace original_time_taken_by_bullet_train_is_50_minutes_l61_61746

-- Define conditions as assumptions
variables (T D : ℝ) (h0 : D = 48 * T) (h1 : D = 60 * (40 / 60))

-- Define the theorem we want to prove
theorem original_time_taken_by_bullet_train_is_50_minutes :
  T = 50 / 60 :=
by
  sorry

end original_time_taken_by_bullet_train_is_50_minutes_l61_61746


namespace smallest_natural_greater_than_12_l61_61736

def smallest_greater_than (n : ℕ) : ℕ := n + 1

theorem smallest_natural_greater_than_12 : smallest_greater_than 12 = 13 :=
by
  sorry

end smallest_natural_greater_than_12_l61_61736


namespace average_runs_next_10_matches_l61_61595

theorem average_runs_next_10_matches (avg_first_10 : ℕ) (avg_all_20 : ℕ) (n_matches : ℕ) (avg_next_10 : ℕ) :
  avg_first_10 = 40 ∧ avg_all_20 = 35 ∧ n_matches = 10 → avg_next_10 = 30 :=
by
  intros h
  sorry

end average_runs_next_10_matches_l61_61595


namespace intersection_and_area_l61_61827

theorem intersection_and_area (A B : ℝ × ℝ) (x y : ℝ):
  (x - 2 * y - 5 = 0) → (x ^ 2 + y ^ 2 = 50) →
  (A = (-5, -5) ∨ A = (7, 1)) → (B = (-5, -5) ∨ B = (7, 1)) →
  (A ≠ B) →
  ∃ (area : ℝ), area = 15 :=
by
  sorry

end intersection_and_area_l61_61827


namespace range_of_m_l61_61239

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), -2 ≤ x ∧ x ≤ 3 ∧ m * x + 6 = 0) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l61_61239


namespace intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l61_61411

noncomputable def f (x : ℝ) := x * Real.log (-x)
noncomputable def g (x a : ℝ) := x * f (a * x) - Real.exp (x - 2)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < -1 / Real.exp 1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 / Real.exp 1 < x ∧ x < 0 → deriv f x < 0) ∧
  f (-1 / Real.exp 1) = 1 / Real.exp 1 :=
sorry

theorem number_of_zeros_of_g (a : ℝ) :
  (a > 0 ∨ a = -1 / Real.exp 1 → ∃! x : ℝ, g x a = 0) ∧
  (a < 0 ∧ a ≠ -1 / Real.exp 1 → ∀ x : ℝ, g x a ≠ 0) :=
sorry

end intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l61_61411


namespace triangle_area_l61_61515

theorem triangle_area (x y : ℝ) (h : 3 * x + y = 9) : 
  (1 / 2) * 3 * 9 = 13.5 :=
sorry

end triangle_area_l61_61515


namespace area_of_wall_photo_l61_61521

theorem area_of_wall_photo (width_frame : ℕ) (width_paper : ℕ) (length_paper : ℕ) 
  (h_width_frame : width_frame = 2) (h_width_paper : width_paper = 8) (h_length_paper : length_paper = 12) :
  (width_paper + 2 * width_frame) * (length_paper + 2 * width_frame) = 192 :=
by
  sorry

end area_of_wall_photo_l61_61521


namespace train_crosses_pole_in_9_seconds_l61_61372

theorem train_crosses_pole_in_9_seconds
  (speed_kmh : ℝ) (train_length_m : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 58) 
  (h2 : train_length_m = 145) 
  (h3 : time_s = train_length_m / (speed_kmh * 1000 / 3600)) :
  time_s = 9 :=
by
  sorry

end train_crosses_pole_in_9_seconds_l61_61372


namespace percentage_difference_l61_61346

theorem percentage_difference (x : ℝ) (h1 : 0.38 * 80 = 30.4) (h2 : 30.4 - (x / 100) * 160 = 11.2) :
    x = 12 :=
by
  sorry

end percentage_difference_l61_61346


namespace find_angle_A_l61_61858

theorem find_angle_A (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : a > 0)
  (h5 : b > 0)
  (h6 : c > 0)
  (sin_eq : Real.sin (C + π / 6) = b / (2 * a)) :
  A = π / 6 :=
sorry

end find_angle_A_l61_61858


namespace total_students_multiple_of_8_l61_61723

theorem total_students_multiple_of_8 (B G T : ℕ) (h : G = 7 * B) (ht : T = B + G) : T % 8 = 0 :=
by
  sorry

end total_students_multiple_of_8_l61_61723


namespace identity_x_squared_minus_y_squared_l61_61838

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61838


namespace geom_prog_235_l61_61209

theorem geom_prog_235 (q : ℝ) (k n : ℕ) (hk : 1 < k) (hn : k < n) : 
  ¬ (q > 0 ∧ q ≠ 1 ∧ 3 = 2 * q^(k - 1) ∧ 5 = 2 * q^(n - 1)) := 
by 
  sorry

end geom_prog_235_l61_61209


namespace binary_to_decimal_l61_61778

theorem binary_to_decimal : 
  let binary_number := [1, 1, 0, 1, 1] in
  let weights := [2^4, 2^3, 2^2, 2^1, 2^0] in
  (List.zipWith (λ digit weight => digit * weight) binary_number weights).sum = 27 := by
  sorry

end binary_to_decimal_l61_61778


namespace amoeba_count_after_two_weeks_l61_61202

theorem amoeba_count_after_two_weeks :
  let initial_day_count := 1
  let days_double_split := 7
  let days_triple_split := 7
  let end_of_first_phase := initial_day_count * 2 ^ days_double_split
  let final_amoeba_count := end_of_first_phase * 3 ^ days_triple_split
  final_amoeba_count = 279936 :=
by
  sorry

end amoeba_count_after_two_weeks_l61_61202


namespace triangle_area_l61_61516

theorem triangle_area (x y : ℝ) (h1 : 3 * x + y = 9) (h2 : x = 0 ∨ y = 0) : 
  let base := 3
  let height := 9
  base * height / 2 = 13.5 :=
by
  let base := 3
  let height := 9
  have h_base : base = 3 := rfl
  have h_height : height = 9 := rfl
  calc
    base * height / 2 = 3 * 9 / 2 : by rw [h_base, h_height]
                    ... = 27 / 2   : by norm_num
                    ... = 13.5     : by norm_num

end triangle_area_l61_61516


namespace percentage_of_employees_in_manufacturing_l61_61187

theorem percentage_of_employees_in_manufacturing (d total_degrees : ℝ) (h1 : d = 144) (h2 : total_degrees = 360) :
    (d / total_degrees) * 100 = 40 :=
by
  sorry

end percentage_of_employees_in_manufacturing_l61_61187


namespace ordered_pairs_count_l61_61109

noncomputable def number_of_ordered_pairs : ℤ :=
  let S := {p : ℝ × ℕ | 
              (0 < p.fst) ∧ 
              (5 ≤ p.snd) ∧ (p.snd ≤ 25) ∧ 
              ((Real.log p.fst / Real.log p.snd) ^ 4 = Real.log (p.fst ^ 4) / Real.log p.snd) ∧ 
              (p.fst = p.snd ^ (Real.log p.fst / Real.log p.snd))} in
  Finset.card (Finset.filter (λ x, true) (Finset.image (prod.mk) (Finset.range (5, 26) × Finset.range (1, 1000000)))) -- placeholder for actual count

theorem ordered_pairs_count : number_of_ordered_pairs = 42 :=
sorry

end ordered_pairs_count_l61_61109


namespace diophantine_infinite_solutions_l61_61190

theorem diophantine_infinite_solutions :
  ∃ (a b c x y : ℤ), (a + b + c = x + y) ∧ (a^3 + b^3 + c^3 = x^3 + y^3) ∧ 
  ∃ (d : ℤ), (a = b - d) ∧ (c = b + d) :=
sorry

end diophantine_infinite_solutions_l61_61190


namespace circle_through_points_l61_61666

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61666


namespace max_s_value_l61_61584

theorem max_s_value (p q r s : ℝ) (h1 : p + q + r + s = 10) (h2 : (p * q) + (p * r) + (p * s) + (q * r) + (q * s) + (r * s) = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end max_s_value_l61_61584


namespace equation_of_circle_through_three_points_l61_61680

theorem equation_of_circle_through_three_points:
  ∃ (D E F : ℝ), 
    F = 0 ∧ 
    (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = 4 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ (x y : ℝ), (x = -1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = x^2 + y^2 - 4 * x - 6 * y) :=
begin
  sorry
end

end equation_of_circle_through_three_points_l61_61680


namespace problem1_problem2_l61_61459

theorem problem1 (x : ℝ) : (5 - 2 * x) ^ 2 - 16 = 0 ↔ x = 1 / 2 ∨ x = 9 / 2 := 
by 
  sorry

theorem problem2 (x : ℝ) : 2 * (x - 3) = x^2 - 9 ↔ x = 3 ∨ x = -1 := 
by 
  sorry

end problem1_problem2_l61_61459


namespace minimum_area_of_cyclic_quadrilateral_l61_61179

theorem minimum_area_of_cyclic_quadrilateral :
  ∀ (r1 r2 : ℝ), (r1 = 1) ∧ (r2 = 2) →
    ∃ (A : ℝ), A = 3 * Real.sqrt 3 ∧ 
    (∀ (q : ℝ) (circumscribed : q ≤ A),
      ∀ (p : Prop), (p = (∃ x y z w, 
        ∀ (cx : ℝ) (cy : ℝ) (cr : ℝ), 
          cr = r2 ∧ 
          (Real.sqrt ((x - cx)^2 + (y - cy)^2) = r2) ∧ 
          (Real.sqrt ((z - cx)^2 + (w - cy)^2) = r2) ∧ 
          (Real.sqrt ((x - cx)^2 + (w - cy)^2) = r1) ∧ 
          (Real.sqrt ((z - cx)^2 + (y - cy)^2) = r1)
      )) → q = A) :=
sorry

end minimum_area_of_cyclic_quadrilateral_l61_61179


namespace value_of_frac_l61_61266

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l61_61266


namespace circle_through_points_l61_61663

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61663


namespace arithmetic_sequence_general_term_l61_61568

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n - 1) = 2) : ∀ n, a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_general_term_l61_61568


namespace number_of_integer_values_l61_61261

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l61_61261


namespace solve_inequality_system_l61_61009

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_system_l61_61009


namespace female_employees_count_l61_61907

theorem female_employees_count (E Male_E Female_E M : ℕ)
  (h1: M = (2 / 5) * E)
  (h2: 200 = (E - Male_E) * (2 / 5))
  (h3: M = (2 / 5) * Male_E + 200) :
  Female_E = 500 := by
{
  sorry
}

end female_employees_count_l61_61907


namespace therapy_sessions_l61_61356

theorem therapy_sessions (F A n : ℕ) 
  (h1 : F = A + 25)
  (h2 : F + A = 115)
  (h3 : F + (n - 1) * A = 250) : 
  n = 5 := 
by sorry

end therapy_sessions_l61_61356


namespace ajay_saves_each_month_l61_61374

def monthly_income : ℝ := 90000
def spend_household : ℝ := 0.50 * monthly_income
def spend_clothes : ℝ := 0.25 * monthly_income
def spend_medicines : ℝ := 0.15 * monthly_income
def total_spent : ℝ := spend_household + spend_clothes + spend_medicines
def amount_saved : ℝ := monthly_income - total_spent

theorem ajay_saves_each_month : amount_saved = 9000 :=
by sorry

end ajay_saves_each_month_l61_61374


namespace bad_iff_prime_l61_61489

def a_n (n : ℕ) : ℕ := (2 * n)^2 + 1

def is_bad (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a_n n = a^2 + b^2

theorem bad_iff_prime (n : ℕ) : is_bad n ↔ Nat.Prime (a_n n) :=
by
  sorry

end bad_iff_prime_l61_61489


namespace contingency_table_confidence_l61_61117

theorem contingency_table_confidence (k_squared : ℝ) (h1 : k_squared = 4.013) : 
  confidence_99 :=
  sorry

end contingency_table_confidence_l61_61117


namespace circle_passing_through_points_eqn_l61_61654

theorem circle_passing_through_points_eqn :
  ∃ (D E F : ℝ), (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ 
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) :=
by
  exists -4, -6, 0
  intros x y
  split
  {
    intro h
    have h' := h x y
    sorry -- Proof steps go here
  }
  {
    intro hy
    cases hy
    {
      rewrite h.1
      rw [hy, hy] -- Fill-in exact transformations steps
      sorry -- Proof steps go here
    }
    {
      cases hy
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
      {
        rewrite h.1
        rw [hy, hy] -- Fill-in exact transformations steps
        sorry -- Proof steps go here
      }
    }
  }

end circle_passing_through_points_eqn_l61_61654


namespace part_1_part_2_part_3_l61_61566

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (2 * Real.exp x) / (Real.exp x + 1) + k

theorem part_1 (k : ℝ) :
  (∀ x, f x k = -f (-x) k) → k = -1 :=
sorry

theorem part_2 (m : ℝ) :
  (∀ x > 0, (2 * Real.exp x - 1) / (Real.exp x + 1) ≤ m * (Real.exp x - 1) / (Real.exp x + 1)) → 2 ≤ m :=
sorry

noncomputable def g (x : ℝ) : ℝ := (f x (-1) + 1) / (1 - f x (-1))

theorem part_3 (n : ℝ) :
  (∀ a b c : ℝ, 0 < a ∧ a ≤ n → 0 < b ∧ b ≤ n → 0 < c ∧ c ≤ n → (a + b > c ∧ b + c > a ∧ c + a > b) →
   (g a + g b > g c ∧ g b + g c > g a ∧ g c + g a > g b)) → n = 2 * Real.log 2 :=
sorry

end part_1_part_2_part_3_l61_61566


namespace polynomial_has_three_real_roots_l61_61364

theorem polynomial_has_three_real_roots (a b c : ℝ) (h1 : b < 0) (h2 : a * b = 9 * c) :
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1^3 + a * x1^2 + b * x1 + c = 0) ∧ 
    (x2^3 + a * x2^2 + b * x2 + c = 0) ∧ 
    (x3^3 + a * x3^2 + b * x3 + c = 0) := sorry

end polynomial_has_three_real_roots_l61_61364


namespace find_marks_in_physics_l61_61047

theorem find_marks_in_physics (P C M : ℕ) (h1 : P + C + M = 225) (h2 : P + M = 180) (h3 : P + C = 140) : 
    P = 95 :=
sorry

end find_marks_in_physics_l61_61047


namespace determine_h_l61_61594

open Polynomial

noncomputable def f (x : ℚ) : ℚ := x^2

theorem determine_h (h : ℚ → ℚ) : 
  (∀ x, f (h x) = 9 * x^2 + 6 * x + 1) ↔ 
  (∀ x, h x = 3 * x + 1 ∨ h x = - (3 * x + 1)) :=
by
  sorry

end determine_h_l61_61594


namespace minimum_value_l61_61128

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + (1 / y)) * (x + (1 / y) - 1024) +
  (y + (1 / x)) * (y + (1 / x) - 1024) ≥ -524288 :=
by sorry

end minimum_value_l61_61128


namespace num_possible_integer_values_l61_61263

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l61_61263


namespace shoes_remaining_l61_61444

theorem shoes_remaining (monthly_goal : ℕ) (sold_last_week : ℕ) (sold_this_week : ℕ) (remaining_shoes : ℕ) :
  monthly_goal = 80 →
  sold_last_week = 27 →
  sold_this_week = 12 →
  remaining_shoes = monthly_goal - sold_last_week - sold_this_week →
  remaining_shoes = 41 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end shoes_remaining_l61_61444


namespace wanda_blocks_l61_61915

theorem wanda_blocks (initial_blocks: ℕ) (additional_blocks: ℕ) (total_blocks: ℕ) : 
  initial_blocks = 4 → additional_blocks = 79 → total_blocks = initial_blocks + additional_blocks → total_blocks = 83 :=
by
  intros hi ha ht
  rw [hi, ha] at ht
  exact ht

end wanda_blocks_l61_61915


namespace minimize_total_time_l61_61577

def exercise_time (s : ℕ → ℕ) : Prop :=
  ∀ i, s i < 45

def total_exercises (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 25

def minimize_time (a : ℕ → ℕ) (s : ℕ → ℕ) : Prop :=
  ∃ (j : ℕ), (1 ≤ j ∧ j ≤ 7 ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 7 → if i = j then a i = 25 else a i = 0) ∧
  ∀ i, 1 ≤ i ∧ i ≤ 7 → s i ≥ s j)

theorem minimize_total_time
  (a : ℕ → ℕ) (s : ℕ → ℕ) 
  (h_exercise_time : exercise_time s)
  (h_total_exercises : total_exercises a) :
  minimize_time a s := by
  sorry

end minimize_total_time_l61_61577


namespace Phillip_correct_total_l61_61448

def number_questions_math : ℕ := 40
def number_questions_english : ℕ := 50
def percentage_correct_math : ℚ := 0.75
def percentage_correct_english : ℚ := 0.98

noncomputable def total_correct_answers : ℚ :=
  (number_questions_math * percentage_correct_math) + (number_questions_english * percentage_correct_english)

theorem Phillip_correct_total : total_correct_answers = 79 := by
  sorry

end Phillip_correct_total_l61_61448


namespace Alyssa_total_spent_l61_61522

-- Declare the costs of grapes and cherries.
def costOfGrapes : ℝ := 12.08
def costOfCherries : ℝ := 9.85

-- Total amount spent by Alyssa.
def totalSpent : ℝ := 21.93

-- Statement to prove that the sum of the costs is equal to the total spent.
theorem Alyssa_total_spent (g : ℝ) (c : ℝ) (t : ℝ) 
  (hg : g = costOfGrapes) 
  (hc : c = costOfCherries) 
  (ht : t = totalSpent) :
  g + c = t := by
  sorry

end Alyssa_total_spent_l61_61522


namespace n_product_expression_l61_61886

theorem n_product_expression (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 :=
sorry

end n_product_expression_l61_61886


namespace invalid_votes_percentage_l61_61863

def total_votes : ℕ := 560000
def valid_votes_A : ℕ := 357000
def percentage_A : ℝ := 0.75
def invalid_percentage (x : ℝ) : Prop := (percentage_A * (1 - x / 100) * total_votes = valid_votes_A)

theorem invalid_votes_percentage : ∃ x : ℝ, invalid_percentage x ∧ x = 15 :=
by 
  use 15
  unfold invalid_percentage
  sorry

end invalid_votes_percentage_l61_61863


namespace sqrt_six_greater_two_l61_61944

theorem sqrt_six_greater_two : Real.sqrt 6 > 2 :=
by
  sorry

end sqrt_six_greater_two_l61_61944


namespace expected_winnings_is_correct_l61_61929

variable (prob_1 prob_23 prob_456 : ℚ)
variable (win_1 win_23 loss_456 : ℚ)

theorem expected_winnings_is_correct :
  prob_1 = 1/4 → 
  prob_23 = 1/2 → 
  prob_456 = 1/4 → 
  win_1 = 2 → 
  win_23 = 4 → 
  loss_456 = -3 → 
  (prob_1 * win_1 + prob_23 * win_23 + prob_456 * loss_456 = 1.75) :=
by
  intros
  sorry

end expected_winnings_is_correct_l61_61929


namespace cousins_rooms_distribution_l61_61140

theorem cousins_rooms_distribution : 
  (∑ n in ({ (5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1) } : finset (ℕ × ℕ × ℕ × ℕ)), 
    match n with 
    | (5,0,0,0) => 1
    | (4,1,0,0) => 5
    | (3,2,0,0) => 10 
    | (3,1,1,0) => 20 
    | (2,2,1,0) => 30 
    | (2,1,1,1) => 10 
    | _ => 0 
    end) = 76 := 
by 
  sorry

end cousins_rooms_distribution_l61_61140


namespace circle_through_points_l61_61660

variable {D E F : ℝ}
def circle_eq (x y : ℝ) := x^2 + y^2 + D*x + E*y + F = 0

theorem circle_through_points :
  (circle_eq 0 0) ∧ (circle_eq 4 0) ∧ (circle_eq (-1) 1) → ∃ (D E : ℝ), D = -4 ∧ E = -6 ∧ (F = 0 ∧ (∀ x y : ℝ, circle_eq x y = x^2 + y^2 - 4*x - 6*y)) :=
sorry

end circle_through_points_l61_61660


namespace simplify_and_evaluate_expression_l61_61456

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 2) : 
  (1 / (x - 3) / (1 / (x^2 - 9)) - x / (x + 1) * ((x^2 + x) / x^2)) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l61_61456


namespace sum_sublist_eq_100_l61_61334

theorem sum_sublist_eq_100 {l : List ℕ}
  (h_len : l.length = 2 * 31100)
  (h_max : ∀ x ∈ l, x ≤ 100)
  (h_sum : l.sum = 200) :
  ∃ (s : List ℕ), s ⊆ l ∧ s.sum = 100 := 
sorry

end sum_sublist_eq_100_l61_61334


namespace merchant_profit_percentage_l61_61058

theorem merchant_profit_percentage 
    (cost_price : ℝ) 
    (markup_percentage : ℝ) 
    (discount_percentage : ℝ) 
    (h1 : cost_price = 100) 
    (h2 : markup_percentage = 0.20) 
    (h3 : discount_percentage = 0.05) 
    : ((cost_price * (1 + markup_percentage) * (1 - discount_percentage) - cost_price) / cost_price * 100) = 14 := 
by 
    sorry

end merchant_profit_percentage_l61_61058


namespace remaining_unit_area_l61_61895

theorem remaining_unit_area
    (total_units : ℕ)
    (total_area : ℕ)
    (num_12x6_units : ℕ)
    (length_12x6_unit : ℕ)
    (width_12x6_unit : ℕ)
    (remaining_units_area : ℕ)
    (num_remaining_units : ℕ)
    (remaining_unit_area : ℕ) :
  total_units = 72 →
  total_area = 8640 →
  num_12x6_units = 30 →
  length_12x6_unit = 12 →
  width_12x6_unit = 6 →
  remaining_units_area = total_area - (num_12x6_units * length_12x6_unit * width_12x6_unit) →
  num_remaining_units = total_units - num_12x6_units →
  remaining_unit_area = remaining_units_area / num_remaining_units →
  remaining_unit_area = 154 :=
by
  intros h_total_units h_total_area h_num_12x6_units h_length_12x6_unit h_width_12x6_unit h_remaining_units_area h_num_remaining_units h_remaining_unit_area
  sorry

end remaining_unit_area_l61_61895


namespace transformed_curve_l61_61824

def curve_C (x y : ℝ) := (x - y)^2 + y^2 = 1

theorem transformed_curve (x y : ℝ) (A : Matrix (Fin 2) (Fin 2) ℝ) :
    A = ![![2, -2], ![0, 1]] →
    (∃ (x0 y0 : ℝ), curve_C x0 y0 ∧ x = 2 * x0 - 2 * y0 ∧ y = y0) →
    (∃ (x y : ℝ), (x^2 / 4 + y^2 = 1)) :=
by
  -- Proof to be completed
  sorry

end transformed_curve_l61_61824


namespace area_of_sector_radius_2_angle_90_l61_61476

-- Given conditions
def radius := 2
def central_angle := 90

-- Required proof: the area of the sector with given conditions equals π.
theorem area_of_sector_radius_2_angle_90 : (90 * Real.pi * (2^2) / 360) = Real.pi := 
by
  sorry

end area_of_sector_radius_2_angle_90_l61_61476


namespace total_bricks_used_l61_61312

def numCoursesPerWall : Nat := 6
def bricksPerCourse : Nat := 10
def numWalls : Nat := 4
def unfinishedCoursesLastWall : Nat := 2

theorem total_bricks_used : 
  let totalCourses := numWalls * numCoursesPerWall
  let bricksRequired := totalCourses * bricksPerCourse
  let bricksMissing := unfinishedCoursesLastWall * bricksPerCourse
  let bricksUsed := bricksRequired - bricksMissing
  bricksUsed = 220 := 
by
  sorry

end total_bricks_used_l61_61312


namespace calculation_is_correct_l61_61080

theorem calculation_is_correct :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := 
by
  sorry

end calculation_is_correct_l61_61080


namespace circle_through_points_l61_61672

noncomputable def circle_eq (D E F : ℝ) : (ℝ × ℝ) → ℝ := 
λ p, p.1 ^ 2 + p.2 ^ 2 + D * p.1 + E * p.2 + F

theorem circle_through_points :
  ∃ D E F : ℝ, 
    (circle_eq D E F (0, 0) = 0) ∧ 
    (circle_eq D E F (4, 0) = 0) ∧ 
    (circle_eq D E F (-1, 1) = 0) ∧ 
    (circle_eq D E F = λ ⟨x, y⟩, x^2 + y^2 - 4*x - 6*y) :=
begin
  use [-4, -6, 0],
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  split,
  { simp [circle_eq] },
  { ext ⟨x, y⟩,
    simp [circle_eq] }
end

#eval circle_eq -4 -6 0 (4, 0) -- should output 0
#eval circle_eq -4 -6 0 (-1, 1) -- should output 0
#eval circle_eq -4 -6 0 (0, 0) -- should output 0

end circle_through_points_l61_61672


namespace average_study_diff_l61_61579

theorem average_study_diff (diff : List ℤ) (h_diff : diff = [15, -5, 25, -10, 5, 20, -15]) :
  (List.sum diff) / (List.length diff) = 5 := by
  sorry

end average_study_diff_l61_61579


namespace union_A_B_eq_A_union_B_l61_61115

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def B : Set ℝ := { x | x > 3 / 2 }

theorem union_A_B_eq_A_union_B :
  (A ∪ B) = { x | -1 ≤ x } :=
by
  sorry

end union_A_B_eq_A_union_B_l61_61115


namespace dara_jane_age_ratio_l61_61474

theorem dara_jane_age_ratio :
  ∀ (min_age : ℕ) (jane_current_age : ℕ) (dara_years_til_min_age : ℕ) (d : ℕ) (j : ℕ),
  min_age = 25 →
  jane_current_age = 28 →
  dara_years_til_min_age = 14 →
  d = 17 →
  j = 34 →
  d = dara_years_til_min_age - 14 + 6 →
  j = jane_current_age + 6 →
  (d:ℚ) / j = 1 / 2 := 
by
  intros
  sorry

end dara_jane_age_ratio_l61_61474


namespace polygon_sides_l61_61368

theorem polygon_sides (n : ℕ) (h₁ : ∀ (m : ℕ), m = n → n > 2) (h₂ : 180 * (n - 2) = 156 * n) : n = 15 :=
by
  sorry

end polygon_sides_l61_61368


namespace sum_first_seven_terms_geometric_seq_l61_61960

theorem sum_first_seven_terms_geometric_seq :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  S_7 = 127 / 192 := 
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  have h : S_7 = 127 / 192 := sorry
  exact h

end sum_first_seven_terms_geometric_seq_l61_61960


namespace number_of_integer_values_l61_61262

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l61_61262


namespace car_speed_first_hour_l61_61905

theorem car_speed_first_hour (x : ℕ) :
  (x + 60) / 2 = 75 → x = 90 :=
by
  -- To complete the proof in Lean, we would need to solve the equation,
  -- reversing the steps provided in the solution. 
  -- But as per instructions, we don't need the proof, hence we put sorry.
  sorry

end car_speed_first_hour_l61_61905


namespace original_number_from_sum_l61_61425

variable (a b c : ℕ) (m S : ℕ)

/-- Given a three-digit number, the magician asks the participant to add all permutations -/
def three_digit_number_permutations_sum (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c + (100 * a + 10 * c + b) + (100 * b + 10 * c + a) +
  (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)

/-- Given the sum of all permutations of the three-digit number is 4239, determine the original number -/
theorem original_number_from_sum (S : ℕ) (hS : S = 4239) (Sum_conditions : three_digit_number_permutations_sum a b c = S) :
  (100 * a + 10 * b + c) = 429 := by
  sorry

end original_number_from_sum_l61_61425


namespace find_length_of_brick_l61_61219

-- Definitions given in the problem
def w : ℕ := 4
def h : ℕ := 2
def SA : ℕ := 112
def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

-- Lean 4 statement for the proof problem
theorem find_length_of_brick (l : ℕ) (h w SA : ℕ) (h_w : w = 4) (h_h : h = 2) (h_SA : SA = 112) :
  surface_area l w h = SA → l = 8 := by
  intros H
  simp [surface_area, h_w, h_h, h_SA] at H
  sorry

end find_length_of_brick_l61_61219


namespace ellipse_foci_distance_l61_61063

noncomputable def center : ℝ×ℝ := (6, 3)
noncomputable def semi_major_axis_length : ℝ := 6
noncomputable def semi_minor_axis_length : ℝ := 3
noncomputable def distance_between_foci : ℝ :=
  let a := semi_major_axis_length
  let b := semi_minor_axis_length
  let c := Real.sqrt (a^2 - b^2)
  2 * c

theorem ellipse_foci_distance :
  distance_between_foci = 6 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l61_61063


namespace parabola_axis_l61_61035

theorem parabola_axis (p : ℝ) (h_parabola : ∀ x : ℝ, y = x^2 → x^2 = y) : (y = - p / 2) :=
by
  sorry

end parabola_axis_l61_61035


namespace brian_total_video_length_l61_61941

theorem brian_total_video_length :
  let cat_length := 4
  let dog_length := 2 * cat_length
  let gorilla_length := cat_length ^ 2
  let elephant_length := cat_length + dog_length + gorilla_length
  let cat_dog_gorilla_elephant_sum := cat_length + dog_length + gorilla_length + elephant_length
  let penguin_length := cat_dog_gorilla_elephant_sum ^ 3
  let dolphin_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length
  let total_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length + dolphin_length
  total_length = 351344 := by
    sorry

end brian_total_video_length_l61_61941


namespace identity_x_squared_minus_y_squared_l61_61839

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l61_61839


namespace inequality_solution_l61_61011

theorem inequality_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) :=
by
  sorry

end inequality_solution_l61_61011


namespace circle_equation_l61_61649

/-- The equation of a circle passing through the points (0,0), (4,0), and (-1,1) is x^2 + y^2 - 4x - 6y = 0. -/
theorem circle_equation :
  ∃ (D E F : ℝ), D = -4 ∧ E = -6 ∧ F = 0 ∧ ∀ (x y : ℝ), 
    (x - 4)^2 + (y - 0)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x + 1)^2 + (y - 1)^2 =
    (x - 0)^2 + (y - 0)^2 ∧
    (x^2 + y^2 + D * x + E * y + F = 0) :=
begin
  sorry
end

end circle_equation_l61_61649
