import Mathlib

namespace joggers_meet_l215_215492

theorem joggers_meet (t_cathy t_david t_elena : ℕ) (h_cathy : t_cathy = 5) (h_david : t_david = 9) (h_elena : t_elena = 8) :
  ∃ t : ℕ, t = Nat.lcm (Nat.lcm t_cathy t_david) t_elena ∧ t = 360 ∧ t / t_cathy = 72 :=
begin
  use 360,
  split,
  { rw [h_cathy, h_david, h_elena],
    exact Nat.lcm_assoc 5 9 8 },
  split,
  { rw [Nat.lcm_assoc, h_cathy, h_david, h_elena],
    calc
      Nat.lcm (Nat.lcm 5 9) 8 = Nat.lcm 45 8      : by rw Nat.lcm_comm
                          ... = 360               : by sorry  },
  { rw h_cathy,
    exact Nat.div_eq_of_eq_mul Nat.zero_le Nat.zero_le sorry }
end

end joggers_meet_l215_215492


namespace min_real_roots_property_l215_215646

noncomputable def min_real_roots {α : Type*} [linear_ordered_field α]
  (p : polynomial α) (roots : fin 2010 → α)
  (h_deg : p.degree = 2010)
  (h_real_coeff : ∀ coeff, coeff ∈ p.coeff → coeff ∈ ℝ)
  (h_distinct_abs : (finset.image (abs ∘ roots) finset.univ).card = 1009)
  : nat :=
8

-- The statement to be proven
theorem min_real_roots_property {α : Type*} [linear_ordered_field α]
  (p : polynomial α) 
  (roots : fin 2010 → α)
  (h_deg : p.degree = 2010)
  (h_real_coeff : ∀ coeff, coeff ∈ p.coeff → coeff ∈ ℝ)
  (h_distinct_abs : (finset.image (abs ∘ roots) finset.univ).card = 1009)
  : min_real_roots p roots h_deg h_real_coeff h_distinct_abs = 8 := 
sorry

end min_real_roots_property_l215_215646


namespace find_x_l215_215583

theorem find_x 
  (λ : ℝ) (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) :
  a = (1, 1) →
  b = (-2, x) →
  a = λ • b →
  x = -2 :=
by
  -- the proof would go here
  sorry

end find_x_l215_215583


namespace sufficient_not_necessary_l215_215580

variable (x a : ℝ)

def p := 1 / (x - 2) ≥ 1
def q := abs (x - a) < 1

theorem sufficient_not_necessary (h : ∀ x, p x → q x ∧ ∃ x, ¬p x ∧ q x) :
  2 < a ∧ a ≤ 3 := 
begin
  sorry
end

end sufficient_not_necessary_l215_215580


namespace intervals_of_increase_max_perimeter_triangle_l215_215538

-- Define the function f(x)
def f (x : ℝ) : ℝ := sqrt 3 * (sin x) ^ 2 + sin x * cos x - sqrt 3 / 2

-- Define the intervals of increase for f(x)
theorem intervals_of_increase (k : ℤ) :
  ∀ x : ℝ, (k * π - π / 12 <= x ∧ x <= k * π + 5 * π / 12) ↔ 
           (deriv f x > 0) :=
sorry

-- Define the properties of the triangle and function at A
variables {A : ℝ} (A_acute : A > 0 ∧ A < π / 2)
variables {a b c : ℝ} (a_eq_2 : a = 2) (f_eq_sqrt3_div_2 : f A = sqrt 3 / 2)

-- The maximum perimeter of triangle ABC
theorem max_perimeter_triangle : 
  (2 * cos A * sqrt ((sin A) ^ 2 + 1 - 2 * sin A * cos A * (sin A - c / b) ^ 2)) ≤ 6 :=
sorry

end intervals_of_increase_max_perimeter_triangle_l215_215538


namespace Jeanine_more_pencils_than_Clare_l215_215303

variables (Jeanine_pencils : ℕ) (Clare_pencils : ℕ)

def Jeanine_initial_pencils := 18
def Clare_initial_pencils := Jeanine_initial_pencils / 2
def Jeanine_pencils_given_to_Abby := Jeanine_initial_pencils / 3
def Jeanine_remaining_pencils := Jeanine_initial_pencils - Jeanine_pencils_given_to_Abby

theorem Jeanine_more_pencils_than_Clare :
  Jeanine_remaining_pencils - Clare_initial_pencils = 3 :=
by
  -- This is just the statement, the proof is not provided as instructed.
  sorry

end Jeanine_more_pencils_than_Clare_l215_215303


namespace chess_piece_position_l215_215066

theorem chess_piece_position :
  ∀ (col row : ℕ), col = 3 ∧ row = 7 → (col, row) = (3, 7) :=
by
  intros col row h
  cases h with h_col h_row
  rw [h_col, h_row]
  sorry

end chess_piece_position_l215_215066


namespace abs_neg2023_eq_2023_l215_215374

-- Define a function to represent the absolute value
def abs (x : ℤ) : ℤ :=
  if x < 0 then -x else x

-- Prove that abs (-2023) = 2023
theorem abs_neg2023_eq_2023 : abs (-2023) = 2023 :=
by
  -- In this theorem, all necessary definitions are already included
  sorry

end abs_neg2023_eq_2023_l215_215374


namespace rectangle_perimeter_not_necessarily_integer_l215_215452

theorem rectangle_perimeter_not_necessarily_integer (r : ℝ × ℝ) (rectangles : list (ℝ × ℝ)) 
  (h : r ∈ rectangles) 
  (h_int_perimeters : ∀ rect ∈ rectangles, ∃ m : ℕ, 2 * (rect.1 + rect.2) = m) 
  : ¬ ∃ n : ℕ, 2 * (r.1 + r.2) = n := 
sorry

end rectangle_perimeter_not_necessarily_integer_l215_215452


namespace factory_fills_boxes_per_hour_l215_215068

theorem factory_fills_boxes_per_hour
  (colors_per_box : ℕ)
  (crayons_per_color : ℕ)
  (total_crayons : ℕ)
  (hours : ℕ)
  (crayons_per_hour := total_crayons / hours)
  (crayons_per_box := colors_per_box * crayons_per_color)
  (boxes_per_hour := crayons_per_hour / crayons_per_box) :
  colors_per_box = 4 →
  crayons_per_color = 2 →
  total_crayons = 160 →
  hours = 4 →
  boxes_per_hour = 5 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end factory_fills_boxes_per_hour_l215_215068


namespace functional_equation_solution_l215_215144

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x * f(y) + y) = f(x * y) + f(y)) →
  (f = (fun x => 0) ∨ f = (fun x => x)) :=
by
  sorry

end functional_equation_solution_l215_215144


namespace find_expression_l215_215647

variables (x y z : ℝ) (ω : ℂ)

theorem find_expression
  (h1 : x ≠ -1) (h2 : y ≠ -1) (h3 : z ≠ -1)
  (h4 : ω^3 = 1) (h5 : ω ≠ 1)
  (h6 : (1 / (x + ω) + 1 / (y + ω) + 1 / (z + ω) = ω)) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) = -1 / 3 :=
sorry

end find_expression_l215_215647


namespace point_P_distance_l215_215184

variable (a b c d x : ℝ)

-- Define the points on the line
def O := 0
def A := a
def B := b
def C := c
def D := d

-- Define the conditions for point P
def AP_PDRatio := (|a - x| / |x - d| = 2 * |b - x| / |x - c|)

theorem point_P_distance : AP_PDRatio a b c d x → b + c - a = x :=
by
  sorry

end point_P_distance_l215_215184


namespace max_min_vector_magnitude_tan_of_sum_angled_l215_215946

-- Given vectors and function f(x)
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 2)
def f (x : ℝ) : ℝ := a x.1 * b x.1 + a x.2 * b x.2

-- Statement for (1)
theorem max_min_vector_magnitude :
  let magnitude := λ x : ℝ, Real.sqrt ((Real.sin x + Real.cos x)^2 + 9)
  ∃ (xmin xmax : ℝ),
    xmin = -Real.pi / 12 ∧ xmax = Real.pi / 3 ∧
    (∀ x ∈ Icc (-Real.pi / 12) (Real.pi / 3), xmin ≤ magnitude x ∧ magnitude x ≤ xmax) ∧
    magnitude xmin = Real.sqrt 10 - Real.sqrt 1/2 ∧
    magnitude xmax = Real.sqrt 11 := sorry

-- Statement for (2)
theorem tan_of_sum_angled :
  ∃ α : ℝ,
    (π / 4 < α ∧ α < π / 2) ∧
    f α = 12 / 5 ∧
    Real.tan (2 * α + 3 * π / 4) = 7 := sorry

end max_min_vector_magnitude_tan_of_sum_angled_l215_215946


namespace expression_simplifies_to_zero_l215_215420

noncomputable def logarithmic_expression (a : ℝ) :=
  11 ^ real.log 20 / real.log a * (12 ^ real.log 21 / real.log a - 13 ^ real.log 22 / real.log a) - 
  20 ^ real.log 11 / real.log a * (21 ^ real.log 12 / real.log a - 22 ^ real.log 13 / real.log a)

theorem expression_simplifies_to_zero (a : ℝ) (h_a_pos : 0 < a) :
  logarithmic_expression a = 0 :=
by
  sorry

end expression_simplifies_to_zero_l215_215420


namespace problem_statement_l215_215219

variable {R : Type*} [LinearOrderedField R]

def is_even_function (f : R → R) : Prop := ∀ x : R, f x = f (-x)

theorem problem_statement (f : R → R)
  (h1 : is_even_function f)
  (h2 : ∀ x1 x2 : R, x1 ≤ -1 → x2 ≤ -1 → (x2 - x1) * (f x2 - f x1) < 0) :
  f (-1) < f (-3 / 2) ∧ f (-3 / 2) < f 2 :=
sorry

end problem_statement_l215_215219


namespace log_base_change_l215_215640

variable (a b : Real)
hypothesis h1 : Real.log 2 = a
hypothesis h2 : Real.log 3 = b

theorem log_base_change :
  Real.logb 5 12 = (b + 2 * a) / (1 - a) :=
sorry

end log_base_change_l215_215640


namespace anna_coins_value_l215_215471

theorem anna_coins_value : 
  ∀ (p n : ℕ),
  p + n = 15 ∧ p = 2 * (n + 1) + 1 → 5 * n + p = 31 :=
by
  intros p n h,
  cases h with h1 h2,
  sorry

end anna_coins_value_l215_215471


namespace math_problem_l215_215206

variable {ℕ : Type*}

def sequence_sum (S : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), S n = 2^n - 1

def sequence_a (a S : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), n > 0 → a n = S n - S (n - 1)

def sequence_b (b a : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), n > 0 → b n = (Real.log (a n) / Real.log 4) + 1

def sum_bn (T b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), T n = (n^2 + 3*n) / 4

theorem math_problem (S a b T : ℕ → ℕ) :
  sequence_sum S →
  sequence_a a S →
  sequence_b b a →
  sum_bn T b →
  (∀ (n : ℕ), n > 0 → a n = 2^(n-1)) ∧
  (∀ (n : ℕ), T n = (n^2 + 3 * n) / 4) :=
by
  intros hS hA hB hT
  split
  { sorry }
  { sorry }

end math_problem_l215_215206


namespace equiangular_hexagon_l215_215903

-- Define a structure representing a convex hexagon
structure ConvexHexagon (α : Type) [OrderedAddCommMonoid α] :=
(A B C D E F : α)

-- Define the condition function
/-- Distance between midpoints of opposite sides equals sqrt(3)/2 times the sum of their lengths -/
def midpoint_distance_condition {α : Type} [LinearOrderedField α] (h : ConvexHexagon α) : Prop :=
  let dist := λ x y : α, abs (x - y)
  let midpoint := λ x y : α, (x + y) / 2
  (dist (midpoint h.A h.B) (midpoint h.D h.E) = (sqrt 3 / 2) * (abs (h.A - h.B) + abs (h.D - h.E))) ∧
  (dist (midpoint h.B h.C) (midpoint h.E h.F) = (sqrt 3 / 2) * (abs (h.B - h.C) + abs (h.E - h.F))) ∧
  (dist (midpoint h.C h.D) (midpoint h.F h.A) = (sqrt 3 / 2) * (abs (h.C - h.D) + abs (h.F - h.A)))

-- Define the theorem using the condition
theorem equiangular_hexagon {α : Type} [LinearOrderedField α] (h : ConvexHexagon α) 
  (H : midpoint_distance_condition h) : 
  (∃ θ : α, ∀ v, v ∈ [h.A, h.B, h.C, h.D, h.E, h.F] → angle v = θ) :=
sorry

end equiangular_hexagon_l215_215903


namespace points_in_circle_l215_215350

theorem points_in_circle 
  (S : set (ℝ × ℝ))  -- S is the set of points on a unit square
  (hS : ∀ p, p ∈ S → p.1 ≥ 0 ∧ p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ 1)  -- Condition for being inside the unit square
  (h_card : S.card = 51)  -- There are exactly 51 points
  (r : ℝ)
  (hr : r = 1/7)  -- Radius of the circle
  : ∃ P : set (ℝ × ℝ), P.card ≥ 3 ∧ ∃ c : (ℝ × ℝ), ∀ p ∈ P, dist p c ≤ r :=
sorry

end points_in_circle_l215_215350


namespace odd_divisors_iff_perfect_square_l215_215043

-- Definition stating a number has an odd number of divisors if and only if it is a perfect square
theorem odd_divisors_iff_perfect_square (n : ℕ) : 
  (∃ k : ℕ, n = k * k) ↔ (∃ m : ℕ, (∃ i : ℕ, (∃ (h : i * i = n), (1 ≤ i ≤ m))) → ¬ (∃ j : ℕ, 2 * j + 1 = (factors n).length)) :=
by sorry

end odd_divisors_iff_perfect_square_l215_215043


namespace problem_proof_l215_215258

-- Define the conditions
def a (n : ℕ) : Real := sorry  -- a is some real number, so it's non-deterministic here

def a_squared (n : ℕ) : Real := a n ^ (2 * n)  -- a^(2n)

-- Main theorem to prove
theorem problem_proof (n : ℕ) (h : a_squared n = 3) : 2 * (a n ^ (6 * n)) - 1 = 53 :=
by
  sorry  -- Proof to be completed

end problem_proof_l215_215258


namespace probability_same_spot_l215_215734

theorem probability_same_spot :
  let students := ["A", "B"]
  let spots := ["Spot 1", "Spot 2"]
  let total_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 1"), ("B", "Spot 2")),
                         (("A", "Spot 2"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]
  let favorable_outcomes := [(("A", "Spot 1"), ("B", "Spot 1")),
                             (("A", "Spot 2"), ("B", "Spot 2"))]
  ∀ (students : List String) (spots : List String)
    (total_outcomes favorable_outcomes : List ((String × String) × (String × String))),
  (students = ["A", "B"]) →
  (spots = ["Spot 1", "Spot 2"]) →
  (total_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                     (("A", "Spot 1"), ("B", "Spot 2")),
                     (("A", "Spot 2"), ("B", "Spot 1")),
                     (("A", "Spot 2"), ("B", "Spot 2"))]) →
  (favorable_outcomes = [(("A", "Spot 1"), ("B", "Spot 1")),
                         (("A", "Spot 2"), ("B", "Spot 2"))]) →
  favorable_outcomes.length / total_outcomes.length = 1 / 2 := 
by
  intros
  sorry

end probability_same_spot_l215_215734


namespace exponential_log_transform_l215_215888

variable (a b x y u c : ℝ)

-- Define the given exponential curve and logarithmic transformations
def exponential_curve (a b x y : ℝ) := y = a * real.exp(b * x)
def logarithmic_transform (y : ℝ) := real.log y
def log_a (a : ℝ) := real.log a

-- State the proof problem
theorem exponential_log_transform (a b x y u c : ℝ)
  (h1 : exponential_curve a b x y)
  (h2 : u = logarithmic_transform y)
  (h3 : c = log_a a) :
  u = b + c :=
by
  sorry

end exponential_log_transform_l215_215888


namespace train_length_is_300_l215_215464

-- Definitions based on the conditions
def trainCrossesPlatform (L V : ℝ) : Prop :=
  L + 400 = V * 42

def trainCrossesSignalPole (L V : ℝ) : Prop :=
  L = V * 18

-- The main theorem statement
theorem train_length_is_300 (L V : ℝ)
  (h1 : trainCrossesPlatform L V)
  (h2 : trainCrossesSignalPole L V) :
  L = 300 :=
by
  sorry

end train_length_is_300_l215_215464


namespace cars_in_north_america_correct_l215_215061

def total_cars_produced : ℕ := 6755
def cars_produced_in_europe : ℕ := 2871

def cars_produced_in_north_america : ℕ := total_cars_produced - cars_produced_in_europe

theorem cars_in_north_america_correct : cars_produced_in_north_america = 3884 :=
by sorry

end cars_in_north_america_correct_l215_215061


namespace friends_traveled_21_kilometers_l215_215688

def birgit_pace : ℝ := 48 / 8
def average_pace : ℝ := birgit_pace + 4
def hiking_time_minutes : ℝ := 3.5 * 60
def distance_traveled : ℝ := hiking_time_minutes / average_pace

theorem friends_traveled_21_kilometers :
  distance_traveled = 21 :=
by
  sorry

end friends_traveled_21_kilometers_l215_215688


namespace number_of_monic_polynomials_l215_215651

-- Define the conditions
variables (p : ℕ) 
  (hp_prime : Nat.Prime p)
  (hp_gt_2 : p > 2)

-- State the theorem
theorem number_of_monic_polynomials (φ_p_minus_1 : ℕ) :
  φ (p - 1) = φ_p_minus_1 → 
  (∃ n, number_of_monic_polynomials_in_Zmod_p (p - 2) = n) → n = φ_p_minus_1 := 
sorry

end number_of_monic_polynomials_l215_215651


namespace ADE_equilateral_l215_215634

-- Definitions
variables {A B C D E O H : Type} [euclidean_geometry A B C] 

-- Conditions
def angle_BAC_eq_sixty (A B C : Type) [euclidean_geometry A B C] : Prop :=
angle A B C = 60

def Euler_line_intersects (A B C D E O H : Type) [euclidean_geometry A B C D E O H] : Prop :=
is_circumcenter O A B C ∧ is_orthocenter H A B C ∧
is_euler_line O H A B C ∧ 
intersects_lines Euler_line A B = some D ∧ intersects_lines Euler_line A C = some E

-- Theorem
theorem ADE_equilateral (A B C D E O H : Type) [euclidean_geometry A B C D E O H]
    (h1 : acute_triangle A B C)
    (h2 : angle_BAC_eq_sixty A B C)
    (h3 : Euler_line_intersects A B C D E O H) :
    is_equilateral_triangle A D E :=
begin
    sorry
end

end ADE_equilateral_l215_215634


namespace avg_velocity_correct_l215_215451

-- Given the equation of motion s = 5 - 3 * t^2
def s (t : ℝ) : ℝ := 5 - 3 * t^2

-- To find the average velocity during the time interval [1, 1 + Δt],
-- defined as Δs / Δt
def avg_velocity (Δt : ℝ) : ℝ :=
  let Δs := s (1 + Δt) - s 1
  in Δs / Δt

-- Prove that the average velocity during the interval [1, 1 + Δt] is -3 Δt - 6
theorem avg_velocity_correct (Δt : ℝ) : avg_velocity Δt = -3 * Δt - 6 := sorry

end avg_velocity_correct_l215_215451


namespace trigonometric_identity_l215_215430

theorem trigonometric_identity :
  (Real.sqrt 3 / Real.cos (10 * Real.pi / 180) - 1 / Real.sin (170 * Real.pi / 180) = -4) :=
by
  -- Proof goes here
  sorry

end trigonometric_identity_l215_215430


namespace parabola_line_intersection_l215_215797

noncomputable def parabola {x y : ℝ} : Prop := y^2 = 4 * x

def focus := (1 : ℝ, 0)

def line_through_focus (l : ℝ × ℝ → Prop) : Prop :=
∀ P : ℝ × ℝ, l P → P = focus ∨ P.fst > 1 ∧ P.snd ≠ 0

def intersects_parabola (l : ℝ × ℝ → Prop) : Prop :=
∃ A B : ℝ × ℝ, parabola A /\ parabola B ∧ l A ∧ l B

def midpoint_x (A B : ℝ × ℝ) : ℝ := (A.fst + B.fst) / 2

def is_midpoint_x_3 (A B : ℝ × ℝ) : Prop :=
midpoint_x A B = 3

def length_AB (A B : ℝ × ℝ) : ℝ :=
abs (A.fst - B.fst)

theorem parabola_line_intersection (l : ℝ × ℝ → Prop) (A B : ℝ × ℝ) :
  line_through_focus l →
  intersects_parabola l →
  is_midpoint_x_3 A B →
  length_AB A B = 8 :=
by
  sorry

end parabola_line_intersection_l215_215797


namespace graduation_photo_arrangements_l215_215675

theorem graduation_photo_arrangements : 
  let students := ["A", "B", "C", "D", "E", "F", "G"]
  ((let num_arrangements : ℕ :=
     2  -- B and C can be on the left or right.
     * (2 * 1)  -- Arrange B and C.
     * (4.choose 1) -- Choose one out of 4 students to stand on the left.
     * (2 * 1)  -- Arrange the chosen person.
     * 1.factorial  -- Factorial for placing students on the left.
     * 3.factorial) -- Factorial for remaining three students on the right.
     in num_arrangements = 192) := 
begin
  sorry
end

end graduation_photo_arrangements_l215_215675


namespace clock_angle_at_3_15_l215_215027

def hour_hand_position (h m : ℕ) : ℝ := (h % 12) * 30 + (m / 60) * 30
def minute_hand_position (m : ℕ) : ℝ := (m % 60) * 6
def acute_angle (a b : ℝ) : ℝ := if abs (a - b) <= 180 then abs (a - b) else 360 - abs (a - b)

theorem clock_angle_at_3_15 : acute_angle (hour_hand_position 3 15) (minute_hand_position 15) = 7.5 :=
by
  sorry

end clock_angle_at_3_15_l215_215027


namespace fib_solution_l215_215633

def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib n + fib (n+1)

theorem fib_solution : ∀ x y : ℕ, (5 * fib x - 3 * fib y = 1) ↔ (x = 2 ∧ y = 3) ∨ (x = 5 ∧ y = 8) ∨ (x = 8 ∧ y = 13) := 
by sorry

end fib_solution_l215_215633


namespace november_2017_consecutive_days_no_classes_l215_215973

noncomputable def problem : Prop :=
  ∃ (n : ℕ) (saturday sunday : ℕ → Bool),
    (saturday 4 = true) ∧ (saturday 11 = true) ∧ (saturday 18 = true) ∧ (saturday 25 = true) ∧ 
    (sunday 5 = true) ∧ (sunday 12 = true) ∧ (sunday 19 = true) ∧ (sunday 26 = true) ∧ 
    ((11 ≤ ∑ i in finset.range(30), if saturday i || sunday i then 0 else 1) → 
    ∃ (d : ℕ), (d < 30) ∧ (¬ saturday d) ∧ (¬ sunday d) ∧ (¬ saturday (d + 1)) ∧ (¬ sunday (d + 1)) ∧ (¬ saturday (d + 2)) ∧ (¬ sunday (d + 2)))

theorem november_2017_consecutive_days_no_classes : problem := 
  sorry

end november_2017_consecutive_days_no_classes_l215_215973


namespace number_of_books_per_continent_l215_215483

theorem number_of_books_per_continent (total_books : ℕ) (total_continents : ℕ) 
  (h1 : total_books = 488) (h2 : total_continents = 4) :
  (total_books / total_continents) = 122 :=
begin
  -- Lean part does not need the proof steps.
  sorry
end

end number_of_books_per_continent_l215_215483


namespace exists_pair_distinct_integers_l215_215858

theorem exists_pair_distinct_integers :
  ∃ (a b : ℤ), a ≠ b ∧ (a / 2015 + b / 2016 = (2015 + 2016) / (2015 * 2016)) :=
by
  -- Constructing the proof or using sorry to skip it if not needed here
  sorry

end exists_pair_distinct_integers_l215_215858


namespace largest_clowns_number_l215_215605

theorem largest_clowns_number {C : Type} (n : ℕ) (h1 : ∃ (colors : finset ℕ) (clowns_colors : C → finset ℕ), 
  colors.card = 12 ∧ ∀ c : C, (clowns_colors c).card ≥ 5 ∧ ∀ c₁ c₂ : C, c₁ ≠ c₂ → clowns_colors c₁ ≠ clowns_colors c₂ ∧ 
  ∀ i : ℕ, i ∈ colors → (finset.filter (λ c, i ∈ clowns_colors c) finset.univ).card ≤ 20) : n ≤ 48 ∧ ∃ (colors : finset ℕ) (clowns_colors : C → finset ℕ),
  colors.card = 12 ∧ ∀ c : C, (clowns_colors c).card ≥ 5 ∧ ∀ c₁ c₂ : C, c₁ ≠ c₂ → clowns_colors c₁ ≠ clowns_colors c₂ ∧ 
  ∀ i : ℕ, i ∈ colors → (finset.filter (λ c, i ∈ clowns_colors c) finset.univ).card ≤ 20 ∧ finset.card (finset.univ : finset C) = 48 :=
sorry

end largest_clowns_number_l215_215605


namespace find_a_common_tangent_l215_215272

-- Define the functions and conditions
def f (x : ℝ) (a : ℝ) := a * x^2
def f' (x : ℝ) (a : ℝ) := 2 * a * x

def g (x : ℝ) := Real.log x
def g' (x : ℝ) := 1 / x

-- The common tangent point
variables (s t a : ℝ)

-- Stating the theorem
theorem find_a_common_tangent (h1 : t = f s a)
                              (h2 : t = g s)
                              (h3 : f' s a = g' s)
                              (h4 : 0 < a) :
                              a = 1 / (2 * Real.exp 1) :=
by
  sorry

end find_a_common_tangent_l215_215272


namespace range_of_a_cond1_range_of_a_cond2_l215_215939

-- Define sets A and B as given in the conditions
def setA : Set ℝ := { y : ℝ | ∃ (x : ℝ), y = 2 * x - 1 ∧ 0 < x ∧ x ≤ 1 }
def setB (a : ℝ) : Set ℝ := { x : ℝ | (x - a) * (x - (a + 3)) < 0 }

-- Define the two main conditions as hypotheses
def cond1 (a : ℝ) : Prop := setA ⊆ setB a
def cond2 (a : ℝ) : Prop := ∃ y, y ∈ setA ∧ y ∈ setB a

-- Define the two main theorems to be proved
theorem range_of_a_cond1 : ∀ a : ℝ, cond1 a → a ∈ Icc (-2 : ℝ) (-1 : ℝ) :=
by
  sorry

theorem range_of_a_cond2 : ∀ a : ℝ, cond2 a → a ∈ Ioo (-4 : ℝ) (1 : ℝ) :=
by
  sorry

end range_of_a_cond1_range_of_a_cond2_l215_215939


namespace factorial_simplification_l215_215745

theorem factorial_simplification :
  (10.factorial * 6.factorial * 3.factorial) / (9.factorial * 7.factorial) = 60 / 7 :=
by
-- The proof details would go here
sorry

end factorial_simplification_l215_215745


namespace locus_of_midpoints_common_circumcircle_exists_l215_215992

-- Definitions for the problem statement
structure Triangle :=
(A B C : ℝ × ℝ)
(is_acute : ∀ (α β γ : ℝ), α < 90 ∧ β < 90 ∧ γ < 90)

structure Rectangle (T : Triangle) :=
(P1 P2 P3 P4 : ℝ × ℝ)
(one_side_on : T.B = P3.1 ∧ T.C = P4.1 ∧ P3.2 = P4.2)

noncomputable def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

noncomputable def altitude_midpoint_to_bc (T : Triangle) : ℝ × ℝ :=
let AM := midpoint T.B T.C in
midpoint (T.A.1, AM.2) AM

-- Problem statement for part (a)
theorem locus_of_midpoints (T : Triangle) (R : Rectangle T) :
  ∃ L : ℝ × ℝ → Prop,
    (L (midpoint R.P1 R.P3)) ∧
    (L = λ p, ∃ k : ℝ, p = (altitude_midpoint_to_bc T)) := sorry

-- Problem statement for part (b)
theorem common_circumcircle_exists (T : Triangle) :
  ∃ (R1 R2 R3 : Rectangle T), ∃ N : ℝ × ℝ,
    (N = midpoint R1.P1 R1.P4) ∧
    (N = midpoint R2.P1 R2.P4) ∧
    (N = midpoint R3.P1 R3.P4) := sorry

end locus_of_midpoints_common_circumcircle_exists_l215_215992


namespace triangles_exist_l215_215152

def exists_triangles : Prop :=
  ∃ (T : Fin 100 → Type) 
    (h : (i : Fin 100) → ℝ) 
    (A : (i : Fin 100) → ℝ)
    (is_isosceles : (i : Fin 100) → Prop),
    (∀ i : Fin 100, is_isosceles i) ∧
    (∀ i : Fin 99, h (i + 1) = 200 * h i) ∧
    (∀ i : Fin 99, A (i + 1) = A i / 20000) ∧
    (∀ i : Fin 100, 
      ¬(∃ (cover : (Fin 99) → Type),
        (∀ j : Fin 99, cover j = T j) ∧
        (∀ j : Fin 99, ∀ k : Fin 100, k ≠ i → ¬(cover j = T k))))

theorem triangles_exist : exists_triangles :=
sorry

end triangles_exist_l215_215152


namespace arithmetic_sequence_value_l215_215977

-- Define an arithmetic sequence
variable {α : Type} [Add α] [Mul α] [HasSmul ℕ α]

def a (n : ℕ) : α := sorry  -- a general term of the arithmetic sequence

-- Define the conditions
axiom h1 : a 2 + 4 * a 7 + a 12 = (96 : α)

-- Proof statement
theorem arithmetic_sequence_value :
  2 * a 3 + a 15 = (48 : α) :=
by
  sorry

end arithmetic_sequence_value_l215_215977


namespace pos_rat_unique_appearance_l215_215498

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (2 * n) = a n + 1) ∧ (∀ n, a (2 * n + 1) = 1 / a (2 * n))

theorem pos_rat_unique_appearance (a : ℕ → ℚ) (h : sequence a) (r : ℚ) (hr : 0 < r) :
  ∃! n, a n = r :=
sorry

end pos_rat_unique_appearance_l215_215498


namespace alyssa_puppies_left_l215_215090

def initial_puppies : Nat := 7
def puppies_per_puppy : Nat := 4
def given_away : Nat := 15

theorem alyssa_puppies_left :
  (initial_puppies + initial_puppies * puppies_per_puppy) - given_away = 20 := 
  by
    sorry

end alyssa_puppies_left_l215_215090


namespace Shelby_gold_stars_l215_215360

theorem Shelby_gold_stars (yesterday today total : ℕ) (h1 : yesterday = 4) (h2 : today = 3) : total = yesterday + today → total = 7 :=
by
  intros h
  rw [h1, h2, h]
  exact add_assoc 4 3 0

# Testing the theorem using Lean's interactive mode (you can ignore this part for translation, but it's beneficial for confirming correctness):
example : Shelby_gold_stars 4 3 7 4 rfl := sorry

end Shelby_gold_stars_l215_215360


namespace prove_gx_plus_2_minus_gx_l215_215261

theorem prove_gx_plus_2_minus_gx (x : ℝ) (g : ℝ → ℝ) (h : ∀ x, g x = 3^x) :
  g(x+2) - g(x) = 8 * g(x) :=
by
  have h1 : g(x + 2) = 3^(x + 2) := h (x + 2)
  have h2 : g(x) = 3^x := h x
  sorry

end prove_gx_plus_2_minus_gx_l215_215261


namespace assign_students_to_classes_l215_215477

def student : Type := {A, B, C, D}
def classes : Type := finset student

theorem assign_students_to_classes (students : classes) (C1 C2 : classes) :
  C1 ∪ C2 = students ∧ C1 ∩ C2 = ∅ ∧ C1 ≠ ∅ ∧ C2 ≠ ∅ ∧ A ∈ C1 ∧ B ∈ C2 
  ∨ C1 ∪ C2 = students ∧ C1 ∩ C2 = ∅ ∧ C1 ≠ ∅ ∧ C2 ≠ ∅ ∧ A ∈ C2 ∧ B ∈ C1
  → (∃ C1 C2, 
          C1 ∪ C2 = students ∧ 
          C1 ∩ C2 = ∅ ∧ 
          C1 ≠ ∅ ∧ 
          C2 ≠ ∅ ∧ 
          A ∈ C1 ∧ B ∈ C2 
          ∨ A ∈ C2 ∧ B ∈ C1) := 
by
  -- Placeholder for the proof
  sorry

end assign_students_to_classes_l215_215477


namespace scientific_notation_of_700_3_l215_215872

theorem scientific_notation_of_700_3 : 
  ∃ (a : ℝ) (n : ℤ), 700.3 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ n = 2 ∧ a = 7.003 :=
by
  use [7.003, 2]
  simp
  sorry

end scientific_notation_of_700_3_l215_215872


namespace probability_divisible_by_three_l215_215267

noncomputable def prob_divisible_by_three : ℚ :=
  1 - (4/6)^6

theorem probability_divisible_by_three :
  prob_divisible_by_three = 665 / 729 :=
by
  sorry

end probability_divisible_by_three_l215_215267


namespace man_takes_nine_days_l215_215040

noncomputable def man_completion_time : ℕ :=
  let W := (1 : ℚ) / 6 -- Women's rate
  let B := (1 : ℚ) / 18 -- Boy's rate
  let combined_rate := (1 : ℚ) / 3 -- Combined rate of man, woman, and boy
  let M := combined_rate - W - B -- Man's rate
  (1 / M).nat_abs -- Time it takes for the man to complete the work, converted to natural number

theorem man_takes_nine_days : man_completion_time = 9 := by
  sorry

end man_takes_nine_days_l215_215040


namespace num_subsets_set_ex_l215_215002

def set_ex : Set Int := {-1, 0, 1}

theorem num_subsets_set_ex : Fintype.card (Set.setOf set_ex) = 8 := 
sorry

end num_subsets_set_ex_l215_215002


namespace complex_quadrant_l215_215901

noncomputable def z : ℂ := (4 + 3 * complex.I) / (1 + 2 * complex.I) / complex.I

noncomputable def z_conjugate : ℂ := complex.conj z

theorem complex_quadrant :
  let iz : ℂ := complex.I * z
  let z := (4 + 3 * complex.I) / (1 + 2 * complex.I) / complex.I in
  z_conjugate.re < 0 ∧ z_conjugate.im > 0 :=
begin
  sorry
end

end complex_quadrant_l215_215901


namespace arc_length_EF_l215_215377

-- Definitions based on the conditions
def angle_DEF_degrees : ℝ := 45
def circumference_D : ℝ := 80
def total_circle_degrees : ℝ := 360

-- Theorems/lemmata needed to prove the required statement
theorem arc_length_EF :
  let proportion := angle_DEF_degrees / total_circle_degrees
  let arc_length := proportion * circumference_D
  arc_length = 10 :=
by
  -- Placeholder for the proof
  sorry

end arc_length_EF_l215_215377


namespace archer_fish_l215_215826

variable (F A : ℕ)

def fish_caught_in_first_round := 8

def fish_caught_in_second_round := F + A

def fish_caught_in_third_round := 1.6 * (F + A)

def total_fish_caught := 60

theorem archer_fish (F A : ℕ) 
  (h1 : F = 8)
  (h2 : 0 < A)
  (h3 : 2 * F + A + 1.6 * F + 1.6 * A = total_fish_caught) :
  A = 12 :=
  sorry

end archer_fish_l215_215826


namespace new_rate_of_commission_l215_215714

theorem new_rate_of_commission 
  (R1 : ℝ) (R1_eq : R1 = 0.04) 
  (slump_percentage : ℝ) (slump_percentage_eq : slump_percentage = 0.20000000000000007)
  (income_unchanged : ∀ (B B_new : ℝ) (R2 : ℝ),
    B_new = B * (1 - slump_percentage) →
    B * R1 = B_new * R2 → 
    R2 = 0.05) : 
  true := 
by 
  sorry

end new_rate_of_commission_l215_215714


namespace mode_and_median_correct_l215_215606

def scores : List ℕ := [97, 88, 85, 93, 85]

def mode_and_median (l : List ℕ) : ℕ × ℕ :=
  let sorted_l := l.qsort (≤)
  let mode := sorted_l.foldl (λ (r : ℕ × ℕ) x =>
    if x = r.2 then (r.1 + 1, x) else
    if x = r.2 + 1 then r else (1, x)) (0, 0)
  let median := sorted_l.get! (sorted_l.length / 2)
  (mode.2, median)

theorem mode_and_median_correct : mode_and_median scores = (85, 88) := by
  sorry

end mode_and_median_correct_l215_215606


namespace abs_neg_value_l215_215371

-- Definition of absolute value using the conditions given.
def abs (x : Int) : Int :=
  if x < 0 then -x else x

-- Theorem statement that |-2023| = 2023
theorem abs_neg_value : abs (-2023) = 2023 :=
  sorry

end abs_neg_value_l215_215371


namespace binom_sum_is_pow_l215_215182

noncomputable def binomial_sum : ℂ :=
  (range 51).sum (λ k, if even k then binom 101 k * (complex.I^k) else 0)

theorem binom_sum_is_pow : binomial_sum = 2^50 := by
  sorry

end binom_sum_is_pow_l215_215182


namespace distance_proof_l215_215763

-- Definitions from the conditions
def avg_speed_to_retreat := 50
def avg_speed_back_home := 75
def total_round_trip_time := 10
def distance_between_home_and_retreat := 300

-- Theorem stating the problem
theorem distance_proof 
  (D : ℝ)
  (h1 : D / avg_speed_to_retreat + D / avg_speed_back_home = total_round_trip_time) :
  D = distance_between_home_and_retreat :=
sorry

end distance_proof_l215_215763


namespace reachable_iff_even_sum_l215_215635

variables (a b : ℕ)
variables (x y : ℤ)

/- Positive integers a and b such that gcd(a, b) = 1 -/
axiom gcd_ab : Int.gcd a b = 1
axiom positive_a : 0 < a
axiom positive_b : 0 < b

/-
Define the moves of type A and type B
-/
inductive stepA : ℤ × ℤ → ℤ × ℤ → Prop
| mk : ∀ (x y : ℤ), stepA (x, y) (x + a, y + a)
| mk_neg_y : ∀ (x y : ℤ), stepA (x, y) (x + a, y - a)
| mk_neg_x : ∀ (x y : ℤ), stepA (x, y) (x - a, y + a)
| mk_neg_xy : ∀ (x y : ℤ), stepA (x, y) (x - a, y - a)

inductive stepB : ℤ × ℤ → ℤ × ℤ → Prop
| mk : ∀ (x y : ℤ), stepB (x, y) (x + b, y + b)
| mk_neg_y : ∀ (x y : ℤ), stepB (x, y) (x + b, y - b)
| mk_neg_x : ∀ (x y : ℤ), stepB (x, y) (x - b, y + b)
| mk_neg_xy : ∀ (x y : ℤ), stepB (x, y) (x - b, y - b)

/- Define a sequence of alternating stepA and stepB starting from (0, 0) -/
inductive reachable : ℤ × ℤ → Prop
| base : reachable (0, 0)
| step_a : ∀ {p q : ℤ × ℤ}, reachable p → stepA a b p q → reachable q
| step_b : ∀ {p q : ℤ × ℤ}, reachable p → stepB a b p q → reachable q

/-
Define the property that needs to be satisfied: (x + y) is even
-/
noncomputable def is_even (z : ℤ) : Prop := ∃ k : ℤ, z = 2 * k

theorem reachable_iff_even_sum (a b : ℕ) (x y : ℤ) (hx : gcd a b = 1) (ha : 0 < a) (hb : 0 < b) :
  reachable a b (x, y) → is_even (x + y) :=
by
  sorry

end reachable_iff_even_sum_l215_215635


namespace points_in_circle_l215_215351

theorem points_in_circle 
  (S : set (ℝ × ℝ))  -- S is the set of points on a unit square
  (hS : ∀ p, p ∈ S → p.1 ≥ 0 ∧ p.1 ≤ 1 ∧ p.2 ≥ 0 ∧ p.2 ≤ 1)  -- Condition for being inside the unit square
  (h_card : S.card = 51)  -- There are exactly 51 points
  (r : ℝ)
  (hr : r = 1/7)  -- Radius of the circle
  : ∃ P : set (ℝ × ℝ), P.card ≥ 3 ∧ ∃ c : (ℝ × ℝ), ∀ p ∈ P, dist p c ≤ r :=
sorry

end points_in_circle_l215_215351


namespace area_of_the_region_l215_215837

noncomputable def region_area (C D : ℝ×ℝ) (rC rD : ℝ) (y : ℝ) : ℝ :=
  let rect_area := (D.1 - C.1) * y
  let sector_areaC := (1 / 2) * Real.pi * rC^2
  let sector_areaD := (1 / 2) * Real.pi * rD^2
  rect_area - (sector_areaC + sector_areaD)

theorem area_of_the_region :
  region_area (3, 5) (10, 5) 3 5 5 = 35 - 17 * Real.pi := by
  sorry

end area_of_the_region_l215_215837


namespace number_of_subsets_of_M_l215_215579

def M : Set ℕ := {1, 2, 3}

theorem number_of_subsets_of_M : (M.powerset.card = 8) :=
by sorry

end number_of_subsets_of_M_l215_215579


namespace simplify_expr_eval_log_expr_l215_215771

-- Problem 1: Simplify the given expression
theorem simplify_expr (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (2 * a^(2/3) * b^(1/2) * (-6 * a^(1/2) * 3 * b)) / (3 * a^(1/6) * b^(5/6)) = -4 * a :=
sorry

-- Problem 2: Evaluate the logarithmic expression
theorem eval_log_expr : 
  (log 5 35 + 2 * log 0.5 (sqrt 2) - log 5 (1/50) - log 5 14 + 10^log 3) = 5 :=
sorry

end simplify_expr_eval_log_expr_l215_215771


namespace correct_option_l215_215215

-- Condition: Definitions for the geometric sequence and sum.
variables (a : ℕ → ℝ)
variable (q : ℝ)
hypothesis h1 : q ≠ 1
hypothesis geo_seq : ∀ n : ℕ, a (n + 1) = a n * q

-- Condition: The specific equation for terms of the sequence.
hypothesis h2 : 2 * a 2022 = a 2023 + a 2024

-- Sum function for the first n terms of the sequence.
noncomputable def S (n : ℕ) := (finset.range n).sum (λ k, a k)

-- Proof the statement
theorem correct_option : S 2024 + S 2023 = 2 * S 2022 :=
sorry

end correct_option_l215_215215


namespace least_possible_number_l215_215998

theorem least_possible_number {x : ℕ} (h1 : x % 6 = 2) (h2 : x % 4 = 3) : x = 50 :=
sorry

end least_possible_number_l215_215998


namespace triangle_area_l215_215293

theorem triangle_area {ABC : Type} [MetricSpace ABC] {A B C : ABC}
  (hAB : dist A B = 7) (hAC : dist A C = 15) 
  (M : ABC) (hM : is_midpoint M B C)
  (hAM : dist A M = 10) : 
  area_of_triangle A B C = 42 :=
sorry

end triangle_area_l215_215293


namespace three_digit_arithmetic_sequences_count_l215_215950

theorem three_digit_arithmetic_sequences_count :
  (∃ (S : finset (ℕ × ℕ × ℕ)), 
    (∀ s ∈ S, ∃ a b c : ℕ, 
      s = (a, b, c) ∧ 
      1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 9 ∧
      2 * b = a + c) ∧ 
      S.card = 132) := sorry

end three_digit_arithmetic_sequences_count_l215_215950


namespace minimum_value_k_l215_215202

theorem minimum_value_k (k : ℝ) :
  (∃ P : ℝ × ℝ, 
      P.2 = sqrt k * P.1 + 2 ∧
      ∃ A B : ℝ × ℝ, 
        A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧
        (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧
        (A.1 * B.1 + A.2 * B.2 = 0)) →
  k ≥ 1 :=
begin
  sorry
end

end minimum_value_k_l215_215202


namespace proof_problem_l215_215929

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

def g (x : ℝ) (a b : ℝ) := (1/2) * x - (f a b x)

def tangent_condition (a b : ℝ) : Prop :=
  let pi_over_3 := Real.pi / 3
  (f a b pi_over_3 = 0) ∧ ((a * Real.cos pi_over_3 - b * Real.sin pi_over_3) = 1)

def monotonic_decreasing_interval (a b : ℝ) : Set ℝ :=
  { x : ℝ | 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 }

theorem proof_problem :
  (tangent_condition (1/2) (-Real.sqrt 3 / 2)) ∧ ( ∀ x ∈ (monotonic_decreasing_interval 0 2), (g x (1/2) (-Real.sqrt 3 / 2)) < (g (x + Real.pi/6) (1/2) (-Real.sqrt 3 / 2)) ) := 
sorry

end proof_problem_l215_215929


namespace continuous_random_variable_iff_l215_215676

noncomputable def isCDF (F : ℝ → ℝ) (ξ : ℝ → Prop) : Prop :=
∀ x, F x = ℙ (ξ ≤ x)

noncomputable def is_continuous (F : ℝ → ℝ) : Prop :=
∀ x, F x = (λ t, F t) x⁻

theorem continuous_random_variable_iff (ξ : ℝ → Prop) (F : ℝ → ℝ) :
  isCDF F ξ → (∀ x, ℙ (ξ = x) = 0) ↔ is_continuous F :=
by
  intros hF hP
  sorry

end continuous_random_variable_iff_l215_215676


namespace find_year_brother_twice_sister_l215_215972

def brother_age_2010 := 16
def sister_age_2010 := 10

-- We need to prove that there exists an integer year y,
-- such that the brother's age in that year is twice the sister's age.
theorem find_year_brother_twice_sister : ∃ y, (y < 2010) ∧ (brother_age_2010 + (y - 2010) = 2 * (sister_age_2010 + (y - 2010))) :=
by {
  use 2006,
  split,
  { -- Proof that y < 2010
    exact dec_trivial,
  },
  { -- Proof that the brother's age in 2006 is twice the sister's age in 2006
    have h1 : brother_age_2010 + (2006 - 2010) = 12 := by linarith,
    have h2 : 2 * (sister_age_2010 + (2006 - 2010)) = 12 := by linarith,
    rw [h1, h2],
  }
}

end find_year_brother_twice_sister_l215_215972


namespace perpendicular_lines_condition_l215_215943

variables {Line Plane : Type} [has_perp Line Plane] [has_parallel Line Plane] [has_parallel Plane Plane] [has_perp Line Line]

def non_coincident_lines (m n : Line) : Prop := m ≠ n
def non_coincident_planes (α β : Plane) : Prop := α ≠ β

theorem perpendicular_lines_condition 
  {m n : Line} {α β : Plane} 
  (h_non_coincident_lines: non_coincident_lines m n) 
  (h_non_coincident_planes: non_coincident_planes α β) 
  (h1 : m ⟂ α) 
  (h2 : n ∥ α) : m ⟂ n :=
sorry

end perpendicular_lines_condition_l215_215943


namespace problem_1_problem_2_problem_3_l215_215980

def Point := (ℝ × ℝ)

def LineEquation (a b c : ℝ) (p : Point) : Prop := a * p.1 + b * p.2 = c

def collinear (A B C : Point) : Prop := 
  ∃ (a b c : ℝ), LineEquation a b c A ∧ LineEquation a b c B ∧ LineEquation a b c C

def midpoint (A B : Point) : Point := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

def perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * a₂ + b₁ * b₂ = 0

def area_of_triangle (A B : Point) : ℝ := 
  1 / 2 * abs (A.1 * B.2)

theorem problem_1 (A B C : Point) (hA : A = (-4, 0)) (hB : B = (0, 6)) (hC : C = (1, 2)) : ¬ collinear A B C := 
  sorry

theorem problem_2 (A B : Point) (hA : A = (-4, 0)) (hB : B = (0, 6)) (m : Point) (hm : m = midpoint A B) : 
  LineEquation 1 1 1 m :=
  sorry

theorem problem_3 (A B C : Point) (hA : A = (-4, 0)) (hB : B = (0, 6)) (hC : C = (1, 2)) :
  area_of_triangle (4, 0) (0, 8/3) = 16/3 :=
  sorry

end problem_1_problem_2_problem_3_l215_215980


namespace regions_formed_by_three_planes_l215_215592

-- Define the type for spatial relationships among three planes
inductive PlaneRelationship
| AllParallel
| AllIntersectSingleLine
| PairwiseIntersectThreeLines
| TwoIntersectThirdCutsBoth

open PlaneRelationship

-- Define a function to calculate the number of regions based on the relationship
def calculateRegions (rel: PlaneRelationship) : ℕ :=
  match rel with
  | AllParallel => 4
  | AllIntersectSingleLine => 6
  | PairwiseIntersectThreeLines => 7
  | TwoIntersectThirdCutsBoth => 8

-- Example property: Prove that the number of regions is one of the expected values
theorem regions_formed_by_three_planes (rel : PlaneRelationship) : 
  calculateRegions rel = 4 ∨ calculateRegions rel = 6 ∨ calculateRegions rel = 7 ∨ calculateRegions rel = 8 :=
by
  cases rel
  all_goals
    simp
    try { exact Or.inl rfl <|> exact Or.inr (Or.inl rfl) <|> exact Or.inr (Or.inr (Or.inl rfl)) <|> exact Or.inr (Or.inr (Or.inr rfl)) }

sorry -- Complete the proof

end regions_formed_by_three_planes_l215_215592


namespace obtuse_triangles_condition_l215_215904

-- Let A, B, C, D be the points of the quadrilateral ABCD
variables (A B C D P : Type*)
variables [point A] [point B] [point C] [point D] [point P]

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Type*) := convex_quadrilateral A B C D

-- Define the obtuse angle at D
def obtuse_angle_D (A B C D : Type*) := obtuse_angle D

-- Define the property of the no new vertices condition
def no_new_vertices (A B C D : Type*) (P : Set Point) := P ⊆ {A, B, C, D}

-- Define the number of obtuse triangles in the segmentation
def num_obtuse_triangles (A B C D : Type*) (n : ℕ) := 
  ∃ P : Set Point, no_new_vertices A B C D P ∧ segmentation_into_obtuse_triangles A B C D P = n

-- The main statement to prove
theorem obtuse_triangles_condition (A B C D : Type*) (n : ℕ)
  (h1 : is_quadrilateral A B C D)
  (h2 : obtuse_angle_D A B C D)
  (h3 : num_obtuse_triangles A B C D n):
  n ≥ 4 :=
sorry

end obtuse_triangles_condition_l215_215904


namespace chord_length_l215_215716

theorem chord_length (x y : ℝ) : 
  ((x - 2) ^ 2 + (y + 2) ^ 2 = 2) → 
  (x - y - 5 = 0) → 
  ∃ l : ℝ, l = sqrt 6 := 
by 
  intros h1 h2
  sorry

end chord_length_l215_215716


namespace max_area_of_garden_l215_215627

theorem max_area_of_garden (l w : ℝ) (h : l + 2*w = 270) : l * w ≤ 9112.5 :=
sorry

end max_area_of_garden_l215_215627


namespace smallest_sector_angle_l215_215658

theorem smallest_sector_angle : 
  ∃ (a1 d : ℕ), 
  (∀ i : ℕ, i < 15 → ∃ ai : ℕ, ai = a1 + i * d) ∧ 
  (∑ i in Finset.range 15, (a1 + i * d) = 360) ∧ 
  (∀ ai, ∃ i : ℕ, i < 15 → ai = a1 + i * d → ai ≥ 3 ∧ a1 = 3) :=
by sorry

end smallest_sector_angle_l215_215658


namespace odd_numbers_not_dividing_each_other_l215_215194

theorem odd_numbers_not_dividing_each_other (n : ℕ) (hn : n ≥ 4) :
  ∃ (a b : ℕ), a ≠ b ∧ (2 ^ (2 * n) < a ∧ a < 2 ^ (3 * n)) ∧ 
  (2 ^ (2 * n) < b ∧ b < 2 ^ (3 * n)) ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ 
  ¬ (a ∣ b * b) ∧ ¬ (b ∣ a * a) := by
sorry

end odd_numbers_not_dividing_each_other_l215_215194


namespace triangle_condition_l215_215277

variable {A B C a b c : ℝ}

def is_triangle (A B C : ℝ) : Prop := A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0

def is_isosceles_or_right_triangle (a b c : ℝ) : Prop := a = b ∨ a^2 + b^2 = c^2

theorem triangle_condition (h : is_triangle A B C)
  (h1 : a - b = c * (Real.cos B - Real.cos A)) :
  is_isosceles_or_right_triangle a b c :=
sorry

end triangle_condition_l215_215277


namespace evaluate_trig_expressions_l215_215117

-- Define the necessary trigonometric properties and values
def sin_pi_div_6 : ℝ := Real.sin (π / 6)
def sin_7pi_div_6 : ℝ := Real.sin (7 * π / 6)
def cos_neg_pi_div_4 : ℝ := Real.cos (-π / 4)
def cos_pi_div_4 : ℝ := Real.cos (π / 4)
def tan_pi_div_8 : ℝ := Real.tan (π / 8)
def tan_3pi_div_8 : ℝ := Real.tan (3 * π / 8)
def sin_3pi_div_5 : ℝ := Real.sin (3 * π / 5)
def sin_4pi_div_5 : ℝ := Real.sin (4 * π / 5)

-- State the theorem
theorem evaluate_trig_expressions :
  sin_pi_div_6 ≠ sin_7pi_div_6 ∧
  cos_neg_pi_div_4 = cos_pi_div_4 ∧
  tan_pi_div_8 < tan_3pi_div_8 ∧
  sin_3pi_div_5 < sin_4pi_div_5 :=
  by sorry

end evaluate_trig_expressions_l215_215117


namespace θ_eq_neg_pi_div_4_max_value_of_a_add_b_l215_215250

variable (θ : ℝ)
def a := (Real.sin θ, 1)
def b := (1, Real.cos θ)
def a_dot_b := (Prod.fst a) * (Prod.fst b) + (Prod.snd a) * (Prod.snd b)
def a_add_b := (Prod.fst a + Prod.fst b, Prod.snd a + Prod.snd b)
def abs_squared (x : ℝ × ℝ) : ℝ := (Prod.fst x) * (Prod.fst x) + (Prod.snd x) * (Prod.snd x)

theorem θ_eq_neg_pi_div_4 (h1 : a_dot_b θ = 0) (h2 : -Real.pi / 2 < θ ∧ θ < Real.pi / 2) : 
  θ = -Real.pi / 4 :=
sorry

theorem max_value_of_a_add_b (h2 : -Real.pi / 2 < θ ∧ θ < Real.pi / 2) :
  ∃ θ_max, abs (a_add_b θ_max) = 1 + Real.sqrt 2 :=
sorry

end θ_eq_neg_pi_div_4_max_value_of_a_add_b_l215_215250


namespace tan_double_angle_l215_215535

theorem tan_double_angle {x : ℝ} (h : Real.tan (π - x) = 3 / 4) : Real.tan (2 * x) = -24 / 7 :=
by 
  sorry

end tan_double_angle_l215_215535


namespace milu_deer_population_2016_l215_215349

theorem milu_deer_population_2016 (a x y : ℝ) (h1 : ∀ x, y = a * log (x + 1) / log 2) (h2 : y = 100) (hx : x = 31):
    a = 100 → y = 500 := 
begin
  intro ha,
  simp only [ha] at h1,
  have hy : y = 100 * log (x + 1) / log 2, from h1 x,
  rw hx at hy,
  have h3 : 31 + 1 = 32,
  from rfl,
  rw h3 at hy,
  have h4 : log 32 / log 2 = log (2 ^ 5) / log 2,
  {
      congr, 
      rw [log_pow, h3],
      field_simp,
      norm_num,
  },
  rw h4 at hy,
  have log2 : log 2 / log 2 = 1,
  from by { field_simp [(ne_of_lt (real.log_pos (by norm_num : 2 > 0)))] },
  rw [mul_comm, ← mul_assoc, log2, one_mul] at hy,
  exact hy,
end

end milu_deer_population_2016_l215_215349


namespace proof_problem_l215_215840

noncomputable def problem_statement : Prop :=
  (∑ k in finset.range 98 + 3, log 3 (1 + 2 / k) * log k 3 * log (k + 2) 3) = 1 / 2

theorem proof_problem : problem_statement :=
sorry

end proof_problem_l215_215840


namespace cross_sectional_area_l215_215724

variables {Point : Type} [MetricSpace Point] [MeasurableSpace Point]

noncomputable def volume_pyramid (A B C D : Point) : ℝ := 5

theorem cross_sectional_area (A B C D P Q M: Point) [MetricSpace.PointAD: Midpoint.Point AD = P ]
  [MetricSpace.PointBC: Midpoint.Point BC = Q] (h_M: Dist DM MC 2 3) 
  (dist_plane_A: Real.Equal Distance (Plane PQM) A 1) : 
  area (CrossSection PQM A) = 3 :=
    sorry

end cross_sectional_area_l215_215724


namespace unique_property_of_rectangles_l215_215094

-- Definitions of the conditions
structure Quadrilateral (Q : Type*) :=
(sum_of_interior_angles : ∀ (q : Q), angle (interior q) = 360)

structure Rectangle (R : Type*) extends Quadrilateral R :=
(diagonals_bisect_each_other : ∀ (r : R), bisects (diagonals r))
(diagonals_equal_length : ∀ (r : R), equal_length (diagonals r))
(diagonals_not_necessarily_perpendicular : ∀ (r : R), not (necessarily_perpendicular (diagonals r)))

structure Rhombus (H : Type*) extends Quadrilateral H :=
(diagonals_bisect_each_other : ∀ (h : H), bisects (diagonals h))
(diagonals_perpendicular : ∀ (h : H), perpendicular (diagonals h))
(diagonals_not_necessarily_equal_length : ∀ (h : H), not (necessarily_equal_length (diagonals h)))

-- The proof statement
theorem unique_property_of_rectangles (R : Type*) [Rectangle R] (H : Type*) [Rhombus H] :
  ∀ (r : R), ∃ (h : H), equal_length (diagonals r) ∧ not (equal_length (diagonals h)) :=
sorry

end unique_property_of_rectangles_l215_215094


namespace sum_as_fraction_l215_215515

theorem sum_as_fraction :
  (0.1 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) = (13467 / 100000 : ℝ) :=
by
  sorry

end sum_as_fraction_l215_215515


namespace Emily_sixth_quiz_score_l215_215160

theorem Emily_sixth_quiz_score :
  let scores := [92, 95, 87, 89, 100]
  ∃ s : ℕ, (s + scores.sum : ℚ) / 6 = 93 :=
  by
    sorry

end Emily_sixth_quiz_score_l215_215160


namespace find_segment_length_l215_215612

-- Assume an acute triangle ABC
variables {A B C D E : Type}

-- Assume segments AD, DC, BE, EC in the triangle
variables (AD DC BE EC : ℝ)

-- Given conditions
def triangle_conditions (AD DC BE EC : ℝ) : Prop :=
AD = 4 ∧ DC = 6 ∧ BE = 3 ∧ (EC = y) 

-- Prove that EC (or y) = 4.5
theorem find_segment_length (AD DC BE EC : ℝ) (y : ℝ) 
(h : triangle_conditions AD DC BE EC) : 
EC = 4.5 := 
sorry

end find_segment_length_l215_215612


namespace ratio_of_initial_and_doubled_l215_215447

theorem ratio_of_initial_and_doubled (x : ℕ) (h : 3 * (2 * x + 5) = 117) : x : 2 * x = 1 : 2 :=
by
  sorry

end ratio_of_initial_and_doubled_l215_215447


namespace math_problem_l215_215551

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
    (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1)

def eccentricity (c a : ℝ) : Prop :=
    c / a = (Real.sqrt 2) / 2

def distance_center_to_line (d : ℝ) : Prop :=
    d = 2 / (Real.sqrt 3)

def point_P_conditions (λ μ x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
    x = λ * x₁ + 2 * μ * x₂ ∧ y = λ * y₁ + 2 * μ * y₂

def slopes_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
    (y₁ * y₂) / (x₁ * x₂) = -1 / 2
    
def point_Q_conditions (λ μ : ℝ) : Prop :=
    (λ^2 + 4 * μ^2 = 1)

noncomputable def derived_ellipse_eq (x y : ℝ) : Prop :=
    (x^2 / 4) + (y^2 / 2) = 1

noncomputable def sum_distances_to_foci (λ μ : ℝ) : ℝ :=
    2

theorem math_problem (a b : ℝ) (c : ℝ) (d : ℝ) (x y : ℝ)
(M N P : ℝ) (λ μ : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
(h1 : eccentricity c a) 
(h2 : distance_center_to_line d) 
(h3 : ellipse a b x y) 
(h4 : point_P_conditions λ μ x₁ y₁ x₂ y₂ x y)
(h5: slopes_condition x₁ y₁ x₂ y₂)
(h6 : point_Q_conditions λ μ) :
    derived_ellipse_eq x y ∧ sum_distances_to_foci λ μ = 2 := 
sorry

end math_problem_l215_215551


namespace trapezium_parallel_side_length_l215_215874

theorem trapezium_parallel_side_length (x : ℝ) (h1 : x + 14 = y) (h_area : 342 = 9 * (x + 14)) : x = 24 :=
by
  have h1 : x + 14 = 38 := by sorry
  have h2 : x = 38 - 14 := by sorry
  exact h2

end trapezium_parallel_side_length_l215_215874


namespace penguin_permutations_correct_l215_215588

def num_permutations_of_multiset (total : ℕ) (freqs : List ℕ) : ℕ :=
  Nat.factorial total / (freqs.foldl (λ acc x => acc * Nat.factorial x) 1)

def penguin_permutations : ℕ := num_permutations_of_multiset 7 [2, 1, 1, 1, 1, 1]

theorem penguin_permutations_correct : penguin_permutations = 2520 := by
  sorry

end penguin_permutations_correct_l215_215588


namespace total_amount_spent_is_40_l215_215342

-- Definitions based on conditions
def tomatoes_pounds : ℕ := 2
def tomatoes_price_per_pound : ℕ := 5
def apples_pounds : ℕ := 5
def apples_price_per_pound : ℕ := 6

-- Total amount spent computed
def total_spent : ℕ :=
  (tomatoes_pounds * tomatoes_price_per_pound) +
  (apples_pounds * apples_price_per_pound)

-- The Lean theorem statement
theorem total_amount_spent_is_40 : total_spent = 40 := by
  unfold total_spent
  unfold tomatoes_pounds tomatoes_price_per_pound apples_pounds apples_price_per_pound
  calc
    2 * 5 + 5 * 6 = 10 + 30 : by rfl
    ... = 40 : by rfl

end total_amount_spent_is_40_l215_215342


namespace find_m_l215_215944

open Real

namespace VectorPerpendicular

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def perpendicular (v₁ v₂ : ℝ × ℝ) : Prop := (v₁.1 * v₂.1 + v₁.2 * v₂.2) = 0

theorem find_m (m : ℝ) (h : perpendicular a (b m)) : m = 1 / 2 :=
by
  sorry -- Proof is omitted

end VectorPerpendicular

end find_m_l215_215944


namespace r_earns_per_day_l215_215046

variables (P Q R S : ℝ)

theorem r_earns_per_day
  (h1 : P + Q + R + S = 240)
  (h2 : P + R + S = 160)
  (h3 : Q + R = 150)
  (h4 : Q + R + S = 650 / 3) :
  R = 70 :=
by
  sorry

end r_earns_per_day_l215_215046


namespace brother_catch_up_in_3_minutes_l215_215376

variables (v_s v_b : ℝ) (t t_new : ℝ)

-- Conditions
def brother_speed_later_leaves_catch (v_b : ℝ) (v_s : ℝ) : Prop :=
18 * v_s = 12 * v_b

def new_speed_of_brother (v_b v_s : ℝ) : ℝ :=
2 * v_b

def time_to_catch_up (v_s : ℝ) (t_new : ℝ) : Prop :=
6 + t_new = 3 * t_new

-- Goal: prove that t_new = 3
theorem brother_catch_up_in_3_minutes (v_s v_b : ℝ) (t_new : ℝ) :
  (brother_speed_later_leaves_catch v_b v_s) → 
  (new_speed_of_brother v_b v_s) = 3 * v_s → 
  time_to_catch_up v_s t_new → 
  t_new = 3 :=
by sorry

end brother_catch_up_in_3_minutes_l215_215376


namespace minimum_distance_ellipse_proof_l215_215317

noncomputable def minimum_distance_ellipse : ℝ :=
  let P (x y : ℝ) := (x^2 / 16 + y^2 / 4 = 1)
  let Q := (2 : ℝ, 0 : ℝ)
  let distance (x y : ℝ) := real.sqrt ((x - 2)^2 + y^2)
  let dist_x := distance 8/3 (real.sqrt (4 - (8/3)^2 / 4))
  dist_x

theorem minimum_distance_ellipse_proof :
  minimum_distance_ellipse = 2 * real.sqrt 6 / 3 :=
sorry

end minimum_distance_ellipse_proof_l215_215317


namespace probability_product_divisible_by_3_l215_215264

theorem probability_product_divisible_by_3 :
  let p := (1 : ℚ) - (2 / 3) ^ 6 in
  p = 665 / 729 :=
by
  sorry

end probability_product_divisible_by_3_l215_215264


namespace find_number_l215_215777

theorem find_number (x : ℝ) (h1 : 0.90 * 40 = 36) (h2 : 0.80 * x = 36 - 12) : x = 30 :=
by
  sorry

end find_number_l215_215777


namespace count_polynomials_l215_215467

def is_polynomial (exp : String) : Prop :=
  exp = "a / 2" ∨ exp = "-2 * x^2 * y" ∨ exp = " -2 * x + y^2" ∨ exp = "π"

theorem count_polynomials : 
  (is_polynomial "a / 2") ∧ 
  ¬ (is_polynomial "2 / a") ∧ 
  (is_polynomial "-2 * x^2 * y") ∧ 
  (is_polynomial " -2 * x + y^2") ∧
  ¬ (is_polynomial "cbrt(a)") ∧ 
  (is_polynomial "π") → 
  4 := 
by
  sorry

end count_polynomials_l215_215467


namespace find_CD_l215_215394

variables (A B C D : Type*)
variables (AC BC : Real) 
variables (AD BC : Real) (θ : Real)
variables (volume : Real) 

def tetrahedron_conditions 
  (V : Real)
  (angle_ACB : Real)
  (sum_AD_BC_AC : Real) : Prop :=
  ∃ (AD BC AC : Real),
    (volume = V) ∧ 
    (θ = angle_ACB) ∧ 
    (AD + BC + AC / sqrt 2 = sum_AD_BC_AC)

theorem find_CD 
  (V : Real := 1/6)
  (angle_ACB : Real := π/4)
  (sum_AD_BC_AC : Real := 3) 
  (AD BC AC : Real)
  (H : tetrahedron_conditions V angle_ACB sum_AD_BC_AC) : 
  ∃ (CD : Real), CD = sqrt 3 :=
by 
  sorry

end find_CD_l215_215394


namespace total_cost_of_items_l215_215469

variables (E P M : ℝ)

-- Conditions
def condition1 : Prop := E + 3 * P + 2 * M = 240
def condition2 : Prop := 2 * E + 5 * P + 4 * M = 440

-- Question to prove
def question (E P M : ℝ) : ℝ := 3 * E + 4 * P + 6 * M

theorem total_cost_of_items (E P M : ℝ) :
  condition1 E P M →
  condition2 E P M →
  question E P M = 520 := 
by 
  intros h1 h2
  sorry

end total_cost_of_items_l215_215469


namespace overlap_area_of_rotated_triangles_l215_215014

def hypotenuse_length (triangle : ℝ) : Prop := triangle = 10

def area_of_overlap (common_area : ℝ) : Prop :=
  common_area = 25 * Real.sqrt 3 / 4

theorem overlap_area_of_rotated_triangles (hypotenuse_length_10 : hypotenuse_length 10) : 
  ∃ (common_area : ℝ), area_of_overlap common_area :=
begin
  sorry
end

end overlap_area_of_rotated_triangles_l215_215014


namespace ladder_base_distance_l215_215824

noncomputable def length_of_ladder : ℝ := 8.5
noncomputable def height_on_wall : ℝ := 7.5

theorem ladder_base_distance (x : ℝ) (h : x ^ 2 + height_on_wall ^ 2 = length_of_ladder ^ 2) :
  x = 4 :=
by sorry

end ladder_base_distance_l215_215824


namespace A_can_do_work_in_5_days_l215_215060

noncomputable def work_days_for_A : ℕ := 5

theorem A_can_do_work_in_5_days (d B_days: ℕ) 
  (B_works: B_days = 10) 
  (sum_of_ABC_work: ∃ (C_share : ℕ), C_share = 200 ∧ (d = 5 ∧ 2 * (1/d + 1/B_days + 1/(5)) = 1)): 
  d = 5 := 
by 
  rcases sum_of_ABC_work with ⟨C_share, hC_share, hd⟩ 
  exact hd.1
  sorry

end A_can_do_work_in_5_days_l215_215060


namespace volleyball_team_geography_l215_215113

theorem volleyball_team_geography (total_players history_players both_subjects : ℕ) 
  (H1 : total_players = 15) 
  (H2 : history_players = 9) 
  (H3 : both_subjects = 4) : 
  ∃ (geography_players : ℕ), geography_players = 10 :=
by
  -- Definitions / Calculations
  -- Using conditions to derive the number of geography players
  let only_geography_players : ℕ := total_players - history_players
  let geography_players : ℕ := only_geography_players + both_subjects

  -- Prove the statement
  use geography_players
  sorry

end volleyball_team_geography_l215_215113


namespace find_a4_inverse_a4_l215_215556

theorem find_a4_inverse_a4 (a : ℝ) (h : (a + 1/a)^3 = 3) :
  a^4 + 1/a^4 = 9^(1/3::ℝ) - 4 * 3^(1/3::ℝ) + 2 :=
by
  sorry

end find_a4_inverse_a4_l215_215556


namespace problem_1_problem_2_problem_3_l215_215000

-- Definition of sequences and conditions in Lean
def Arith_Seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n - a (n + 1) = d

variables {a b c : ℕ → ℝ}
variables {d d' : ℝ}

-- Problem 1
theorem problem_1 (h : Arith_Seq a d) (hb : ∀ n, b n = a n - 2 * a (n + 1)) :
  Arith_Seq b (-d) :=
sorry

-- Problem 2
theorem problem_2 (hb : Arith_Seq b d') (hc : Arith_Seq c d'') 
  (hb_def : ∀ n, b n = a n - 2 * a (n + 1)) (hc_def : ∀ n, c n = a (n + 1) + 2 * a (n + 2) - 2) :
  Arith_Seq (λ n, a (n + 1)) (d / 2) :=
sorry

-- Problem 3
theorem problem_3 (hb : Arith_Seq b d') (hb_def : ∀ n, b n = a n - 2 * a (n + 1)) 
  (h_cond : b 1 + a 3 = 0) :
  Arith_Seq a (-d') :=
sorry

end problem_1_problem_2_problem_3_l215_215000


namespace length_of_the_bridge_l215_215715

-- Conditions
def train_length : ℝ := 80
def train_speed_kmh : ℝ := 45
def crossing_time_seconds : ℝ := 30

-- Conversion factor
def km_to_m : ℝ := 1000
def hr_to_s : ℝ := 3600

-- Calculation
noncomputable def train_speed_ms : ℝ := train_speed_kmh * km_to_m / hr_to_s
noncomputable def total_distance : ℝ := train_speed_ms * crossing_time_seconds
noncomputable def bridge_length : ℝ := total_distance - train_length

-- Proof statement
theorem length_of_the_bridge : bridge_length = 295 :=
by
  sorry

end length_of_the_bridge_l215_215715


namespace Jeanine_has_more_pencils_than_Clare_l215_215305

def number_pencils_Jeanine_bought : Nat := 18
def number_pencils_Clare_bought := number_pencils_Jeanine_bought / 2
def number_pencils_given_to_Abby := number_pencils_Jeanine_bought / 3
def number_pencils_Jeanine_now := number_pencils_Jeanine_bought - number_pencils_given_to_Abby 

theorem Jeanine_has_more_pencils_than_Clare :
  number_pencils_Jeanine_now - number_pencils_Clare_bought = 3 := by
  sorry

end Jeanine_has_more_pencils_than_Clare_l215_215305


namespace remainder_of_fractions_l215_215414

theorem remainder_of_fractions : 
  ∀ (x y : ℚ), x = 5/7 → y = 3/4 → (x - y * ⌊x / y⌋) = 5/7 :=
by
  intros x y hx hy
  rw [hx, hy]
  -- Additional steps can be filled in here, if continuing with the proof.
  sorry

end remainder_of_fractions_l215_215414


namespace group_sum_180_in_range_1_to_60_l215_215987

def sum_of_arithmetic_series (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem group_sum_180_in_range_1_to_60 :
  ∃ (a n : ℕ), 1 ≤ a ∧ a + n - 1 ≤ 60 ∧ sum_of_arithmetic_series a 1 n = 180 :=
by
  sorry

end group_sum_180_in_range_1_to_60_l215_215987


namespace find_100_positive_integers_with_distinct_sums_l215_215625

theorem find_100_positive_integers_with_distinct_sums :
  ∃ S : set ℕ, S.card = 100 ∧ (∀ x ∈ S, x ≤ 25000) ∧ (∀ a b ∈ S, ∀ c d ∈ S, a ≠ b → c ≠ d → a + b ≠ c + d) :=
begin
  sorry
end

end find_100_positive_integers_with_distinct_sums_l215_215625


namespace geometric_series_sum_eq_4_over_3_l215_215121

theorem geometric_series_sum_eq_4_over_3 : 
  let a := 1
  let r := 1/4
  inf_geometric_sum a r = 4/3 := by
begin
  intros,
  sorry
end

end geometric_series_sum_eq_4_over_3_l215_215121


namespace find_p_plus_q_l215_215142

def step_area (n : ℕ) : ℚ :=
  if n = 0 then 16
  else if n = 1 then 4
  else 4 * (3 / 16)^(n - 1)

noncomputable def total_area : ℚ :=
  16 + 4 + ∑' n : ℕ, if n > 1 then step_area n else 0

theorem find_p_plus_q : total_area = 272 / 13 ∧ nat.gcd 272 13 = 1 → 272 + 13 = 285 :=
by
  sorry

end find_p_plus_q_l215_215142


namespace emily_sixth_quiz_score_l215_215161

theorem emily_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (target_average : ℕ) : 
  s1 = 92 → s2 = 95 → s3 = 87 → s4 = 89 → s5 = 100 → target_average = 93 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = target_average :=
begin
  intros h1 h2 h3 h4 h5 h6,
  use 95,
  have : (92 + 95 + 87 + 89 + 100 + 95) = 558,
  { trivial },
  simp [h1, h2, h3, h4, h5, h6, this],
  norm_num,
  exact rfl,
end

end emily_sixth_quiz_score_l215_215161


namespace problem1_problem2_l215_215834

-- Problem 1 Statement
theorem problem1 : (3 * Real.sqrt 48 - 2 * Real.sqrt 27) / Real.sqrt 3 = 6 :=
by sorry

-- Problem 2 Statement
theorem problem2 : 
  (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + 1 / (2 - Real.sqrt 5) = -3 - Real.sqrt 5 :=
by sorry

end problem1_problem2_l215_215834


namespace circle_eq_of_given_center_and_radius_l215_215900

theorem circle_eq_of_given_center_and_radius :
  (∀ (x y : ℝ),
    let C := (-1, 2)
    let r := 4
    (x + 1) ^ 2 + (y - 2) ^ 2 = 16) :=
by
  sorry

end circle_eq_of_given_center_and_radius_l215_215900


namespace sum_of_numbers_in_ratio_with_lcm_l215_215694

theorem sum_of_numbers_in_ratio_with_lcm
  (x : ℕ)
  (h1 : Nat.lcm (2 * x) (Nat.lcm (3 * x) (5 * x)) = 120) :
  (2 * x) + (3 * x) + (5 * x) = 40 := 
sorry

end sum_of_numbers_in_ratio_with_lcm_l215_215694


namespace books_per_continent_l215_215481

-- Definition of the given conditions
def total_books := 488
def continents_visited := 4

-- The theorem we need to prove
theorem books_per_continent : total_books / continents_visited = 122 :=
sorry

end books_per_continent_l215_215481


namespace max_brownies_l215_215947

theorem max_brownies (m n : ℕ) (h : (m - 2) * (n - 2) = 2 * m + 2 * n - 4) : m * n ≤ 60 :=
sorry

end max_brownies_l215_215947


namespace hyperbola_equation_l215_215876

-- Define the conditions given in the problem
def shares_same_asymptotes (h: ℝ → ℝ → ℝ) : Prop :=
  ∃ λ: ℝ, ∀ x y: ℝ, h x y = x^2 - 4 * y^2 - λ

def passes_through_point (h: ℝ → ℝ → ℝ) (p: ℝ × ℝ) : Prop :=
  ∀ x y: ℝ, p = (x, y) → h x y = 0

-- Define the point M(2, sqrt(5))
def M := (2 : ℝ, Real.sqrt 5)

-- Define the equation of the hyperbola that we want to prove
def desired_hyperbola (x y : ℝ) : ℝ := y^2 / 4 - x^2 / 16 - 1

-- The main theorem to prove
theorem hyperbola_equation :
  ∃ h: ℝ → ℝ → ℝ, shares_same_asymptotes h ∧ passes_through_point h M ∧ ∀ x y, desired_hyperbola x y = 0 :=
by
  -- Existential variable serving as a placeholder for the actual function
  use λ x y, x^2 - 4 * y^2 + 16

  -- Prove it shares the same asymptotes
  split
  · use -16
    intros x y
    rfl

  -- Prove it passes through the given point M
  split
  · intros x y hM
    cases hM
    simp

  -- Finally, prove the desired form is the same
  intros x y
  rfl

-- Placeholder proof
sorry

end hyperbola_equation_l215_215876


namespace greatest_power_of_2_divides_l215_215023

-- Definitions
def f := λ (n m : ℕ), 15^n - 6^m

-- Theorem statement
theorem greatest_power_of_2_divides : 
  greatest_power_of_2 (f 504 502) = 502 :=
sorry

end greatest_power_of_2_divides_l215_215023


namespace mark_total_spending_l215_215344

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end mark_total_spending_l215_215344


namespace complex_number_conjugate_solution_l215_215701

theorem complex_number_conjugate_solution (z : ℂ) (hz : (3 - 4 * Complex.i) * Complex.conj z = 1 + 2 * Complex.i) :
  z = - (1 / 5) - (2 / 5) * Complex.i :=
by
  sorry

end complex_number_conjugate_solution_l215_215701


namespace area_bounded_by_given_line_l215_215486

noncomputable def area_bounded_polar := (3 * Real.pi) / 4

theorem area_bounded_by_given_line {α β : ℝ} (r : ℝ → ℝ) (A : ℝ)
  (h : ∀ φ : ℝ, r φ = (1 / 2) + Real.cos φ ∧ α = 0 ∧ β = Real.pi ∧ ∫ φ in α..β, (r φ) ^ 2 = (3/4) * Real.pi) :
  A = 2 * (1 / 2) * ∫ φ in α..β, (r φ) ^ 2 := by
  sorry

example : area_bounded_by_given_line (λ φ, (0.5 + Real.cos φ)) ((3 * Real.pi) / 4) := by
  sorry

end area_bounded_by_given_line_l215_215486


namespace min_edges_for_3_clique_min_edges_for_4_clique_l215_215057

-- Part (i) proof statement
theorem min_edges_for_3_clique (n : ℕ) : 
  ∃ G : SimpleGraph (Fin n), 
    G.edge_count = n - 1 ∧ 
    (∀ v w, v ≠ w → ¬G.adj v w) ∧ 
    (∀ e : Sym2 (Fin n), ¬G.adj e.1 e.2 → (
      ∃ u v w, u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ G.adj u v ∧ G.adj v w ∧ G.adj u w)) := sorry

-- Part (ii) proof statement
theorem min_edges_for_4_clique (n : ℕ) :
  ∃ G : SimpleGraph (Fin n), 
    G.edge_count = 2 * n - 3 ∧ 
    (∀ u v w, u ≠ v → v ≠ w → w ≠ u → ¬(G.adj u v ∧ G.adj v w ∧ G.adj u w)) ∧ 
    (∀ e : Sym2 (Fin n), ¬G.adj e.1 e.2 → (
      ∃ x y u v, x ≠ y ∧ y ≠ u ∧ u ≠ v ∧ v ≠ x ∧ 
      G.adj x y ∧ G.adj y u ∧ G.adj u v ∧ G.adj x v)) := sorry

end min_edges_for_3_clique_min_edges_for_4_clique_l215_215057


namespace depth_of_melted_ice_cream_l215_215079

theorem depth_of_melted_ice_cream
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ)
  (h : ℝ)
  (sphere_volume_eq : V_sphere = (4 / 3) * Real.pi * r_sphere^3)
  (cylinder_volume_eq : V_sphere = Real.pi * r_cylinder^2 * h)
  (r_sphere_eq : r_sphere = 3)
  (r_cylinder_eq : r_cylinder = 9)
  : h = 4 / 9 :=
by
  -- Proof is omitted
  sorry

end depth_of_melted_ice_cream_l215_215079


namespace points_in_circle_l215_215352

theorem points_in_circle (points : Set (ℝ × ℝ)) (h_points_card : points.card = 51) : 
  ∃ (c : ℝ × ℝ), ∃ (r : ℝ), r = 1 / 7 ∧ (∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (dist p1 c ≤ r ∧ dist p2 c ≤ r ∧ dist p3 c ≤ r)) := by
    sorry

end points_in_circle_l215_215352


namespace rectangular_prism_diagonals_l215_215455

theorem rectangular_prism_diagonals :
  let l := 3
  let w := 4
  let h := 5
  let face_diagonals := 6 * 2
  let space_diagonals := 4
  face_diagonals + space_diagonals = 16 := 
by
  sorry

end rectangular_prism_diagonals_l215_215455


namespace FGH_supermarkets_US_l215_215397

/-- There are 60 supermarkets in the FGH chain,
all of them are either in the US or Canada,
there are 14 more FGH supermarkets in the US than in Canada.
Prove that there are 37 FGH supermarkets in the US. -/
theorem FGH_supermarkets_US (C U : ℕ) (h1 : C + U = 60) (h2 : U = C + 14) : U = 37 := by
  sorry

end FGH_supermarkets_US_l215_215397


namespace abs_neg_value_l215_215369

-- Definition of absolute value using the conditions given.
def abs (x : Int) : Int :=
  if x < 0 then -x else x

-- Theorem statement that |-2023| = 2023
theorem abs_neg_value : abs (-2023) = 2023 :=
  sorry

end abs_neg_value_l215_215369


namespace orthogonal_planes_k_value_l215_215387

theorem orthogonal_planes_k_value
  (k : ℝ)
  (h : 3 * (-1) + 1 * 1 + (-2) * k = 0) : 
  k = -1 :=
sorry

end orthogonal_planes_k_value_l215_215387


namespace determine_n_from_series_l215_215148

theorem determine_n_from_series :
  (∀ n : ℕ, n ∈ {30, 31, ..., 119} → ∑ k in finset.Ico 30 119, (1 / (Real.sin (k : ℝ) * Real.sin (k + 1 : ℝ))) = 1 / Real.sin n) → n = 30 :=
sorry

end determine_n_from_series_l215_215148


namespace businessmen_neither_coffee_nor_tea_l215_215114

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) (coffee_drinkers : ℕ) (tea_drinkers : ℕ) (both_drinkers : ℕ) : 
  coffee_drinkers = 15 → tea_drinkers = 12 → both_drinkers = 7 → total = 30 →
  (total - (coffee_drinkers + tea_drinkers - both_drinkers)) = 10 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end businessmen_neither_coffee_nor_tea_l215_215114


namespace triangle_angle_sum_l215_215990

theorem triangle_angle_sum (BAC ACB ABC : Real) (H1 : BAC = 50) (H2 : ACB = 40) : ABC = 90 :=
by
  -- Using the angle sum property of a triangle
  have angle_sum : BAC + ACB + ABC = 180 := sorry
  -- Substituting the given angles
  rw [H1, H2] at angle_sum
  -- Performing the calculation to obtain the result
  linarith

end triangle_angle_sum_l215_215990


namespace on_time_departure_rate_over_60_l215_215047

theorem on_time_departure_rate_over_60 (flights_late : ℕ)
  (next_flights_on_time : ℕ)
  (total_flights : ℕ)
  (on_time_flights : ℕ)
  (additional_on_time_flights : ℕ) :
  flights_late = 1 →
  next_flights_on_time = 3 →
  total_flights = 4 →
  on_time_flights = 3 →
  additional_on_time_flights = 1 →
  (on_time_flights + additional_on_time_flights) / (total_flights + additional_on_time_flights) > 0.60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end on_time_departure_rate_over_60_l215_215047


namespace find_m_l215_215224

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Given points M, N, and P defined
def M : Point := { x := 2, y := -1 }
def N : Point := { x := 4, y := 5 }
def P (m : ℝ) : Point := { x := 3, y := m }

-- Definition of a line passing through two points
def slope (p1 p2 : Point) : ℝ := (p2.y - p1.y) / (p2.x - p1.x)

-- The theorem to be proved
theorem find_m (m : ℝ) (h : slope M N = slope M (P m)) : m = 2 := by
  sorry

end find_m_l215_215224


namespace cone_surface_area_l215_215807

theorem cone_surface_area (R : ℝ) (h r : ℝ) (h_cone : h^2 + r^2 = R^2 / 4) :
  2 * r * Mathlib.pi + r * R * Mathlib.pi = (3 / 4) * R^2 * Mathlib.pi :=
by 
  sorry

end cone_surface_area_l215_215807


namespace installments_count_is_correct_l215_215062

noncomputable def find_num_installments : ℕ :=
  let first_payment := 410
  let additional_payment := 65
  let num_first_payments := 8
  let total_sum (n : ℕ) := num_first_payments * first_payment + (n - num_first_payments) * (first_payment + additional_payment)
  let average_payment (n : ℕ) := total_sum n / n
  nat.find (λ n, average_payment n = 465)

theorem installments_count_is_correct :
  find_num_installments = 52 := 
sorry

end installments_count_is_correct_l215_215062


namespace min_value_4_range_of_x_l215_215216

-- Define the conditions
variables (a b : ℝ)
hypothesis non_zero_a : a ≠ 0
hypothesis non_zero_b : b ≠ 0

-- Proof of minimum value
theorem min_value_4 : ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → (|2 * a + b| + |2 * a - b|) / |a| = 4 :=
by
  intro a b
  intros non_zero_a non_zero_b
  sorry

-- Proof of the range of x
theorem range_of_x : ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → ∀ x : ℝ, (|2 * a + b| + |2 * a - b|) / |a| ≥ |a| * (|2 + x| + |2 - x|) ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  intro a b
  intros non_zero_a non_zero_b
  intro x
  sorry

end min_value_4_range_of_x_l215_215216


namespace permutation_sum_mod_p_l215_215636

theorem permutation_sum_mod_p (p : ℕ) (hp : p ≥ 3) (prime_p : nat.prime p) :
  ∃ (σ : fin (p - 1) → fin (p - 1)), ((∑ i in finset.range (p - 2), ((σ i) * (σ i.succ))) : ℤ) % p = 2 :=
sorry

end permutation_sum_mod_p_l215_215636


namespace complex_modulus_l215_215541

open Complex

noncomputable def z : ℂ := 3 + 4 * I

def satisfy_condition (z : ℂ) : Prop := 2 * z - conj z = 3 + 12 * I

theorem complex_modulus :
  satisfy_condition z →
  complex.abs z = 5 :=
by
  intro h
  sorry

end complex_modulus_l215_215541


namespace cosine_sine_power_eight_l215_215139

theorem cosine_sine_power_eight :
  (3 * Real.cos (Float.pi / 4) - 3 * Complex.I * Real.sin (Float.pi / 4)) ^ 8 = 6552 :=
by
  sorry

end cosine_sine_power_eight_l215_215139


namespace product_value_l215_215035

noncomputable def product_of_sequence : ℝ :=
  (1/3) * 9 * (1/27) * 81 * (1/243) * 729 * (1/2187) * 6561

theorem product_value : product_of_sequence = 729 := by
  sorry

end product_value_l215_215035


namespace factorization_correctness_l215_215756

theorem factorization_correctness :
  (∀ x, x^2 + 2 * x + 1 = (x + 1)^2) ∧
  ¬ (∀ x, x * (x + 1) = x^2 + x) ∧
  ¬ (∀ x y, x^2 + x * y - 3 = x * (x + y) - 3) ∧
  ¬ (∀ x, x^2 + 6 * x + 4 = (x + 3)^2 - 5) :=
by
  sorry

end factorization_correctness_l215_215756


namespace solution_l215_215846

noncomputable def polynomial_real_root_exists (k : ℝ) := 
  ∀ k > 0, ∃ z : ℂ, 5 * z^3 - 4 * complex.I * z^2 + z - k = 0 ∧ z.im = 0

theorem solution : polynomial_real_root_exists k := sorry

end solution_l215_215846


namespace translate_even_function_l215_215731

noncomputable def f (x : ℝ) : ℝ := sin (3 * x) + cos (3 * x)

theorem translate_even_function (varnothing : ℝ) (h : varnothing = π / 12) :
  ∀ x : ℝ, f (x - varnothing) = f (-(x - varnothing)) := by
  sorry

end translate_even_function_l215_215731


namespace Zhang_Bin_tape_longer_l215_215759

theorem Zhang_Bin_tape_longer (a : ℝ) : ∃ (L : ℝ), ((1 + 0.40) * a - L * a) / (L * a) = 0.39 ∧ L = 140 / 139 ∧ (L - 1) = 1 / 139 :=
by {
  use (140 / 139),
  split,
  { sorry },
  split,
  { refl },
  { linarith }
}

end Zhang_Bin_tape_longer_l215_215759


namespace smallest_multiple_of_seven_gt_neg50_l215_215030

theorem smallest_multiple_of_seven_gt_neg50 : ∃ (n : ℤ), n % 7 = 0 ∧ n > -50 ∧ ∀ (m : ℤ), m % 7 = 0 → m > -50 → n ≤ m :=
sorry

end smallest_multiple_of_seven_gt_neg50_l215_215030


namespace math_problem_l215_215091

theorem math_problem:
  (∀ α : ℝ, 0 < α ∧ α < π/2 → sin α < tan α) ∧
  (∀ α : ℝ, 
      (∃ k : ℤ, π/2 + 2 * k * π < α ∧ α < π + 2 * k * π) → 
      (∃ k : ℤ, π/4 + k * π < α/2 ∧ α/2 < π/2 + k * π)) ∧
  (∀ k : ℝ, k ≠ 0 → ∀ α : ℝ, 
      let P := (3 * k, 4 * k) in 
      sin α ≠ 4 / 5 → P) ∧
  (∀ l r : ℝ, l = 6 - 2 * r ∧ r = 2 → 1 = 1) :=
by
  sorry

end math_problem_l215_215091


namespace problem1_problem2_problem3_problem4_l215_215770

-- Problem 1
def f1 (x : ℝ) := log (x) / log (8) + 2 * log (x) / log (2)
theorem problem1 : f1 8 = 7 := sorry

-- Problem 2
variables {x a : ℝ}
def binom_coeff (n k : ℕ) := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))
theorem problem2 (c : ℕ) (h_c : c = binom_coeff 9 3) : 84 = c * (a^3) → a = 1 := sorry

-- Problem 3
variables {y : ℝ}
def parabola := y^2 = 4
def line := λ (m: ℝ) (y: ℝ), y^2 - 4*m*y - 4 = 0
theorem problem3 : ∀ l : ℝ, (1 - 4 * m = (y1 + y2)) ∧ y = (4 * m)  → (m = (± (2 * sqrt 5 / 5))) := sorry

-- Problem 4
sequence : ℕ → ℝ
| 0 := 12
| (n + 1) := (3 * sequence n) / (3 * n + 4)

lemma seq_S_n (n : ℕ) : real := 
(Sum_basis (λ (i : ℕ), (sequence n) / (3 * n + 1)))

lemma seq_T_n (n : ℕ) : real := 
(Sum_basis (λ (i : ℕ) => (sequence n) / (3^n)))

theorem problem4 : 
(geometric_seq sequence (1, 4) 3) ∧ 
(∀ n, (12 ≤ sequence n → sequence n % 11 ≠ 0)) ∧
(∀ S10 > T243) ∧ 
(T21 % 51 == 0) := sorry

end problem1_problem2_problem3_problem4_l215_215770


namespace upward_shift_of_parabola_l215_215380

variable (k : ℝ) -- Define k as a real number representing the vertical shift

def original_function (x : ℝ) : ℝ := -x^2 -- Define the original function

def shifted_function (x : ℝ) : ℝ := original_function x + 2 -- Define the shifted function by 2 units upwards

theorem upward_shift_of_parabola (x : ℝ) : shifted_function x = -x^2 + k :=
by
  sorry

end upward_shift_of_parabola_l215_215380


namespace monotonicity_of_f_range_of_a_minimum_m_l215_215924

noncomputable def f (x a : ℝ) (hx : x > 0) : ℝ := ((3 - x) * Real.exp x + a) / x

noncomputable def f_prime (x a : ℝ) (hx : x > 0) : ℝ := ((-x^2 + 3*x - 3) * Real.exp x - a) / x^2

theorem monotonicity_of_f (a : ℝ) (hx : ∀ x, x > 0) (h_a : a > - 3 / 4) :
  ∀ x, f_prime x a (hx x) < 0 := 
sorry

theorem range_of_a (two_extreme_points : ∃ x₁ x₂, x₁ < x₂ ∧ f_prime x₁ a (hx x₁) = 0 ∧ f_prime x₂ a (hx x₂) = 0) : 
  ∃ a : ℝ, -3 < a ∧ a < -Real.exp 1 := 
sorry

theorem minimum_m (a : ℝ) (ha : -3 < a ∧ a < -Real.exp 1) 
  (two_extreme_points : ∃ x₁ x₂, 1 < x₂ ∧ x₂ < 3 / 2 ∧ f x₂ a (λ x₂, by simp; linarith) > 2) : 
  ∃ m : ℕ, ∀ x, f x a (λ x, by simp; linarith) < m → m = 3 := 
sorry

end monotonicity_of_f_range_of_a_minimum_m_l215_215924


namespace train_crossing_time_l215_215624

noncomputable def km_per_hr_to_m_per_s (v : ℝ) : ℝ := (v * 1000) / 3600

noncomputable def time_to_cross_pole (length : ℝ) (speed_km_per_hr : ℝ) : ℝ :=
  let speed_m_per_s := km_per_hr_to_m_per_s speed_km_per_hr
  length / speed_m_per_s

theorem train_crossing_time :
  time_to_cross_pole 135 140 ≈ 3.47 :=
by
  sorry

end train_crossing_time_l215_215624


namespace evaluate_powers_of_i_l215_215863

-- Definitions based on the given conditions
noncomputable def i_power (n : ℤ) := 
  if n % 4 = 0 then (1 : ℂ)
  else if n % 4 = 1 then complex.I
  else if n % 4 = 2 then -1
  else -complex.I

-- Statement of the problem
theorem evaluate_powers_of_i :
  i_power 14760 + i_power 14761 + i_power 14762 + i_power 14763 = 0 :=
by
  sorry

end evaluate_powers_of_i_l215_215863


namespace tangent_slope_at_pi_over_4_l215_215528

def curve (x : ℝ) : ℝ := (sin x / (sin x + cos x)) - 1/2

theorem tangent_slope_at_pi_over_4 : (deriv curve (π / 4)) = 1/2 :=
by { sorry }

end tangent_slope_at_pi_over_4_l215_215528


namespace book_price_l215_215533

theorem book_price (n p : ℕ) (h : n * p = 104) (hn : 10 < n ∧ n < 60) : p = 2 ∨ p = 4 ∨ p = 8 :=
sorry

end book_price_l215_215533


namespace max_distance_from_origin_l215_215664

/-- Define the Cartesian coordinates of the pole and the origin -/
def pole : (ℝ × ℝ) := (5, -2)
def origin : (ℝ × ℝ) := (0, 0)

/-- Define the rope length as 15 meters -/
def rope_length : ℝ := 15

/-- Define the distance formula between two points in a Cartesian plane -/
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Define maximum distance from the origin given the constraints -/
theorem max_distance_from_origin : distance origin pole + rope_length = 15 + real.sqrt 29 :=
by
  sorry

end max_distance_from_origin_l215_215664


namespace even_a_injective_xor_l215_215188

def xor (a b : ℕ) : ℕ := (a lor b) - (a land b) * 2

theorem even_a_injective_xor (a : ℕ) (h_pos : a > 0) :
  (∀ (x y : ℕ), x > y → x ≥ 0 → y ≥ 0 → xor x (a * x) ≠ xor y (a * y)) ↔ a % 2 = 0
:= by
sorry

end even_a_injective_xor_l215_215188


namespace min_value_of_quadratic_l215_215708

theorem min_value_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ x : ℝ, (x = -p / 2) ∧ ∀ y : ℝ, (y^2 + p * y + q) ≥ ((-p/2)^2 + p * (-p/2) + q) :=
sorry

end min_value_of_quadratic_l215_215708


namespace rectangles_have_unique_property_of_equal_diagonals_l215_215105

theorem rectangles_have_unique_property_of_equal_diagonals (rectangle rhombus : Type)
  (is_rectangle : rectangle → Prop)
  (is_rhombus : rhombus → Prop)
  (sum_of_angles_360 : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∑ ang : ℝ in {q}, ang = 360))
  (diagonals_bisect : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∃ midp : Prop, ∀ l r : ℝ, l ≠ r → l + r = 0))
  (diagonals_equal_length : ∀ (r : rectangle), Prop := ∀ x y : ℝ, x = y)
  (diagonals_perpendicular : ∀ (r : rhombus), Prop := ⊥)
  : ∀ r : rectangle, is_rectangle r → diagonals_equal_length r :=
  begin
    intros r h,
    sorry
  end

end rectangles_have_unique_property_of_equal_diagonals_l215_215105


namespace block_of_flats_l215_215981

theorem block_of_flats :
  let total_floors := 12
  let half_floors := total_floors / 2
  let apartments_per_half_floor := 6
  let max_residents_per_apartment := 4
  let total_max_residents := 264
  let apartments_on_half_floors := half_floors * apartments_per_half_floor
  ∃ (x : ℝ), 
    4 * (apartments_on_half_floors + half_floors * x) = total_max_residents ->
    x = 5 :=
sorry

end block_of_flats_l215_215981


namespace ordered_pair_exists_l215_215642

theorem ordered_pair_exists (a b : ℝ) :
  (∃ a b : ℝ, 
    (\<boolean>) (by sorry) = (false) (\<boolean>)
  ) :∃ (a, b) $statement_primitive(\(\L \L)).\t\N \begin{pmatrix} 3 \\ 2 \end{pmatrix}
    + a \begin{pmatrix} 6 \\ -4 \end{pmatrix} = \begin{pmatrix} -1 
 \\ 1 \end{pmatrix}+ b \begin{pmatrix} -3 \t 5 \end{pmatrix}
∧ a = -\frac{23}{18} ∧ b = \test 
ans
  result(insert Answer)
Proof Proof_ = in_previous_window 
$\boxed negative Numbers$, (\<function>\(math));end
definition), 
\begin, translation, {
\begin  
 problem, {
√true,(Applied_Math) })\haltrecurse\(false)
\N 
\iteration (Math_Solver_Visualized) 
∧ resulted\infinal

end ordered_pair_exists_l215_215642


namespace combination_eq_solutions_l215_215196

theorem combination_eq_solutions (x : ℕ) 
  (h : nat.choose 20 (3 * x) = nat.choose 20 (x + 4)) : 
  x = 2 ∨ x = 4 :=
sorry

end combination_eq_solutions_l215_215196


namespace four_digit_increasing_digits_ends_with_even_l215_215590

theorem four_digit_increasing_digits_ends_with_even :
  ∃ n : ℕ, n = 46 ∧ 
  ∃ a b c d : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d.even ∧ 1000 * a + 100 * b + 10 * c + d  = n := 
sorry

end four_digit_increasing_digits_ends_with_even_l215_215590


namespace arithmetic_mean_neg3_to_7_l215_215407

theorem arithmetic_mean_neg3_to_7 : 
  (let S := (-3 : ℤ) :: (-2 : ℤ) :: (-1 : ℤ) :: 0 :: 1 :: 2 :: 3 :: 4 :: 5 :: 6 :: (7 : ℤ) :: []) in
  (S.sum : ℚ) / S.length = 2.0 := 
by
  -- Placeholder for proof
  sorry

end arithmetic_mean_neg3_to_7_l215_215407


namespace cos_is_periodic_l215_215110

theorem cos_is_periodic :
  (∀ f : ℝ → ℝ, (f = cos) → (∀ x : ℝ, f x = cos x) → (∀ T : ℝ, T ≠ 0 → f (x + T) = f x)) → 
  (∀ f : ℝ → ℝ, (cos x = f x) → Periodic f) := 
sorry

end cos_is_periodic_l215_215110


namespace find_a_b_l215_215151

theorem find_a_b : 
  ∃ (a b : ℝ), 
  (∀ x : ℝ, a * x^4 + b * x^3 + 1 = 0 → x = 1) ∧ 
  (∀ x : ℝ, a * 4 * x^3 + b * 3 * x^2 = 0 → x = 1) → 
  a = 3 ∧ b = -4 :=
by
  -- This is where the proof would go, adding sorry for placeholder
  sorry

end find_a_b_l215_215151


namespace statement_A_statement_B_statement_C_statement_D_l215_215755

-- Statement A: Definition of the function and conditions
def f (ω : ℝ) (x : ℝ) := sin (ω * x) + 2 * cos (ω * x + π / 3)

-- Statement A: Prove that if the smallest positive period of f is π, then ω = 2
theorem statement_A (ω : ℝ) (hω : ω > 0) (h_smallest_period : ∃ T > 0, T = π ∧ (∀ x, f ω (x + T) = f ω x)) : ω = 2 :=
sorry

-- Statement B: Given a triangle ABC with sides opposite to the angles
-- Prove: A > B iff a > b
variables (a b c : ℝ) (A B C : ℝ)
theorem statement_B (triangle_ABC : Triangle a b c A B C) (h_ineq : A > B) : a > b :=
sorry

-- Statement C: Three non-equal real numbers form an arithmetic progression
-- Prove: 2^a, 2^b, 2^c do not form an arithmetic progression
variables (a b c : ℝ)
theorem statement_C (h_neq : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_arith_prog : b - a = c - b) : ¬(2^a + 2^c = 2 * 2^b) :=
sorry

-- Statement D: The oblique projection is an equilateral triangle with side length 2
-- Prove: The area of triangle ABC is 2√6
variables (Projection : TriangleEquilateralProjection)
theorem statement_D (h_equilateral : EquilateralProjection Projection) : area Projection = 2 * sqrt 6 :=
sorry

end statement_A_statement_B_statement_C_statement_D_l215_215755


namespace right_triangle_altitude_l215_215979

theorem right_triangle_altitude {DE DF EF altitude : ℝ} (h_right_triangle : DE^2 = DF^2 + EF^2)
  (h_DE : DE = 15) (h_DF : DF = 9) (h_EF : EF = 12) (h_area : (DF * EF) / 2 = 54) :
  altitude = 7.2 := 
  sorry

end right_triangle_altitude_l215_215979


namespace find_a_l215_215925

noncomputable def f (a x : ℝ) : ℝ := Real.log (x - (1/2) * a * x^2 - x)

theorem find_a (a : ℝ) (hf : ∀ x : ℝ, deriv (λ x, f a x) x = (1/x) - a * x - 1)
  (hext : (deriv (λ x, f a x) 1 = 0)) : a = 0 :=
sorry

end find_a_l215_215925


namespace task1_task2_l215_215869

/-- Given conditions -/
def cost_A : Nat := 30
def cost_B : Nat := 40
def sell_A : Nat := 35
def sell_B : Nat := 50
def max_cost : Nat := 1550
def min_profit : Nat := 365
def total_cars : Nat := 40

/-- Task 1: Prove maximum B-type cars produced if 10 A-type cars are produced -/
theorem task1 (A: Nat) (B: Nat) (hA: A = 10) (hC: cost_A * A + cost_B * B ≤ max_cost) : B ≤ 31 :=
by sorry

/-- Task 2: Prove the possible production plans producing 40 cars meeting profit and cost constraints -/
theorem task2 (A: Nat) (B: Nat) (hTotal: A + B = total_cars)
(hCost: cost_A * A + cost_B * B ≤ max_cost) 
(hProfit: (sell_A - cost_A) * A + (sell_B - cost_B) * B ≥ min_profit) : 
  (A = 5 ∧ B = 35) ∨ (A = 6 ∧ B = 34) ∨ (A = 7 ∧ B = 33) 
∧ (375 ≤ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35 ∧ 375 ≥ (sell_A - cost_A) * 5 + (sell_B - cost_B) * 35) :=
by sorry

end task1_task2_l215_215869


namespace sin_cos_equal_tangent_identity_l215_215256

theorem sin_cos_equal_tangent_identity (x : ℝ) (hx : sin (4 * x) * sin (5 * x) = cos (4 * x) * cos (5 * x)) : 
  x = 10 * (π / 180) :=
by sorry

end sin_cos_equal_tangent_identity_l215_215256


namespace sphere_surface_area_increase_l215_215601

theorem sphere_surface_area_increase (V A : ℝ) (r : ℝ)
  (hV : V = (4/3) * π * r^3)
  (hA : A = 4 * π * r^2)
  : (∃ r', (V = 8 * ((4/3) * π * r'^3)) ∧ (∃ A', A' = 4 * A)) :=
by
  sorry

end sphere_surface_area_increase_l215_215601


namespace reciprocal_roots_condition_l215_215671

theorem reciprocal_roots_condition (a b c : ℝ) (h : a ≠ 0) (roots_reciprocal : ∃ r s : ℝ, r * s = 1 ∧ r + s = -b/a ∧ r * s = c/a) : c = a :=
by
  sorry

end reciprocal_roots_condition_l215_215671


namespace find_real_x_l215_215169

noncomputable def solution_set (x : ℝ) := (5 ≤ x) ∧ (x < 5.25)

theorem find_real_x (x : ℝ) :
  (⌊x * ⌊x⌋⌋ = 20) ↔ solution_set x :=
by
  sorry

end find_real_x_l215_215169


namespace selling_price_is_correct_l215_215078

noncomputable def costPrice : ℝ := 925
noncomputable def profitPercentage : ℝ := 20 / 100

def profit := costPrice * profitPercentage
def sellingPrice := costPrice + profit

theorem selling_price_is_correct : sellingPrice = 1110 := by
  sorry

end selling_price_is_correct_l215_215078


namespace integral_sin_div_one_plus_sin_sq_l215_215487

theorem integral_sin_div_one_plus_sin_sq :
  ∫ x in 0..(Real.pi / 2), ((Real.sin x) / (1 + (Real.sin x))^2) = 1 / 3 :=
begin
  sorry
end

end integral_sin_div_one_plus_sin_sq_l215_215487


namespace horner_rule_v3_is_36_l215_215894

def f (x : ℤ) : ℤ := x^5 + 2*x^3 + 3*x^2 + x + 1

theorem horner_rule_v3_is_36 :
  let v0 := 1;
  let v1 := v0 * 3 + 0;
  let v2 := v1 * 3 + 2;
  let v3 := v2 * 3 + 3;
  v3 = 36 := 
by
  sorry

end horner_rule_v3_is_36_l215_215894


namespace inequality_holds_l215_215760

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  (1 / (x^2 - 4) + 4 / (2 * x^2 + 7 * x + 6) ≤ 1 / (2 * x + 3) + 4 / (2 * x^3 + 3 * x^2 - 8 * x - 12))

-- The main theorem statement
theorem inequality_holds :
  { x : ℝ // inequality x } ⊆
    { x : ℝ // (x > -2 ∧ x < -3 / 2) ∨ (x ≥ 1 ∧ x < 2) ∨ (x ≥ 5) } :=
sorry

end inequality_holds_l215_215760


namespace fahrenheit_to_celsius_conversion_l215_215497

theorem fahrenheit_to_celsius_conversion :
  let C (F : ℤ) := (5 * (F - 30)) / 9
  let F_to_C_round (F : ℤ) := (5 * (F - 30) + 4) / 9
  let C_to_F_round (C : ℤ) := (9 * C + 12) / 5 + 30
  let valid_F (F : ℤ) := C_to_F_round (F_to_C_round F) = F
  Int.range (30, 800) |>.count valid_F = 443
:= sorry

end fahrenheit_to_celsius_conversion_l215_215497


namespace discount_correct_l215_215629

-- Define the prices of items and the total amount paid
def t_shirt_price : ℕ := 30
def backpack_price : ℕ := 10
def cap_price : ℕ := 5
def total_paid : ℕ := 43

-- Define the total cost before discount
def total_cost := t_shirt_price + backpack_price + cap_price

-- Define the discount
def discount := total_cost - total_paid

-- Prove that the discount is 2 dollars
theorem discount_correct : discount = 2 :=
by
  -- We need to prove that (30 + 10 + 5) - 43 = 2
  sorry

end discount_correct_l215_215629


namespace sequence_contains_infinite_powers_of_two_l215_215907

theorem sequence_contains_infinite_powers_of_two
  (a : ℕ → ℕ)
  (h₁ : ¬ (5 ∣ a 1))
  (h₂ : ∀ n, let b_n := a n % 10 in a (n + 1) = a n + b_n) :
  ∃ infinitely_many k, ∃ n, a n = 2^k :=
sorry

end sequence_contains_infinite_powers_of_two_l215_215907


namespace sum_of_ages_is_55_l215_215393

def sum_of_ages (Y : ℕ) (interval : ℕ) (number_of_children : ℕ) : ℕ :=
  let ages := List.range number_of_children |>.map (λ i => Y + i * interval)
  ages.sum

theorem sum_of_ages_is_55 :
  sum_of_ages 7 2 5 = 55 :=
by
  sorry

end sum_of_ages_is_55_l215_215393


namespace triangle_b_c_triangle_shape_l215_215970

theorem triangle_b_c (a A S : ℝ) (h_a : a = 2 * real.sqrt 3) (h_A : A = real.pi / 3) (h_S : S = 2 * real.sqrt 3) :
  ∃ b c : ℝ, (1/2) * b * c * real.sin A = S ∧ a^2 = b^2 + c^2 - 2 * b * c * real.cos A ∧ b + c = 6 :=
begin
  sorry
end

theorem triangle_shape (A B C : ℝ) (h_sin_eq : real.sin (C - B) = real.sin (2 * B) - real.sin A) :
  (B = real.pi / 2) ∨ (C = B) :=
begin
  sorry
end

end triangle_b_c_triangle_shape_l215_215970


namespace seal_earnings_l215_215359

theorem seal_earnings :
  let months := 3 * 12 in
  let songs_per_month := 3 in
  let earnings_per_song := 2000 in
  (months * songs_per_month) * earnings_per_song = 216000 := by
  sorry

end seal_earnings_l215_215359


namespace smallest_number_with_digit_sum_10_and_valid_digits_l215_215031

def valid_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (n.digits 10) → d = 1 ∨ d = 2

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_number_with_digit_sum_10_and_valid_digits :
  ∃ n : ℕ, valid_digits n ∧ digit_sum n = 10 ∧ (∀ m : ℕ, valid_digits m ∧ digit_sum m = 10 → n ≤ m) :=
begin
  use 111111112,
  split,
  { unfold valid_digits,
    intros d hd,
    rw list.mem_digits 10 111111112 at hd,
    simpa using hd, },
  split,
  { rw digit_sum,
    norm_num, },
  intros m H,
  cases H with H1 H2,
  -- Proof that 111111112 is the smallest number will go here.
  sorry,
end

end smallest_number_with_digit_sum_10_and_valid_digits_l215_215031


namespace four_letter_arrangements_count_l215_215253

-- Define the problem conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def is_valid_arrangement (arr : List Char) : Prop :=
  arr.length = 4 ∧
  arr.head = 'C' ∧
  'B' ∈ arr.tail ∧
  'D' ∈ arr.tail ∧
  arr = arr.dedup

noncomputable def count_valid_arrangements : Nat :=
  (List.permutations (letters.erase 'C')).count (λ l, is_valid_arrangement (List.cons 'C' l))

theorem four_letter_arrangements_count : count_valid_arrangements = 24 :=
by
  sorry

end four_letter_arrangements_count_l215_215253


namespace area_of_circumcircle_proof_l215_215296

noncomputable def area_of_circumcircle 
  (A : ℝ) (b : ℝ) (area : ℝ) : ℝ :=
  π * (b^2 / (2 * sin (A / 2)))^2

theorem area_of_circumcircle_proof : 
  area_of_circumcircle (45 * π / 180) (2 * sqrt 2) 1 = 5 / 2 * π := 
sorry

end area_of_circumcircle_proof_l215_215296


namespace evaluate_expression_l215_215257

def operation_star (A B : ℕ) : ℕ := (A + B) / 2
def operation_ominus (A B : ℕ) : ℕ := A - B

theorem evaluate_expression :
  operation_ominus (operation_star 6 10) (operation_star 2 4) = 5 := 
by 
  sorry

end evaluate_expression_l215_215257


namespace minimum_toothpicks_removal_l215_215192

theorem minimum_toothpicks_removal
  (total_toothpicks : ℕ)
  (grid_size : ℕ)
  (toothpicks_per_square : ℕ)
  (shared_sides : ℕ)
  (interior_toothpicks : ℕ) 
  (diagonal_toothpicks : ℕ)
  (min_removal : ℕ) 
  (no_squares_or_triangles : Bool)
  (h1 : total_toothpicks = 40)
  (h2 : grid_size = 3)
  (h3 : toothpicks_per_square = 4)
  (h4 : shared_sides = 16)
  (h5 : interior_toothpicks = 16) 
  (h6 : diagonal_toothpicks = 12)
  (h7 : min_removal = 16)
: no_squares_or_triangles := 
sorry

end minimum_toothpicks_removal_l215_215192


namespace min_distance_l215_215913

-- Define the curve equation
def curve (x y : ℝ) : Prop := x^2 - y - ln x = 0

-- Define the line equation
def line (x y : ℝ) : Prop := y = x - 3

-- Define the distance formula from a point to a line y = x - 3
noncomputable def distance_to_line (x y : ℝ) : ℝ := abs (x - y - 3) / sqrt 2

-- Prove that the minimum distance from a point on the curve to the line is (3 * sqrt 2) / 2
theorem min_distance (x y : ℝ) (h_curve: curve x y) (h_line: line x y) : distance_to_line x y = (3 * sqrt 2) / 2 := by
  sorry

end min_distance_l215_215913


namespace width_of_second_square_is_seven_l215_215479

-- The conditions translated into Lean definitions
def first_square : ℕ × ℕ := (8, 5)
def third_square : ℕ × ℕ := (5, 5)
def flag_dimensions : ℕ × ℕ := (15, 9)

-- The area calculation functions
def area (dim : ℕ × ℕ) : ℕ := dim.fst * dim.snd

-- Given areas for the first and third square
def area_first_square : ℕ := area first_square
def area_third_square : ℕ := area third_square

-- Desired flag area
def flag_area : ℕ := area flag_dimensions

-- Total area of first and third squares
def total_area_first_and_third : ℕ := area_first_square + area_third_square

-- Required area for the second square
def area_needed_second_square : ℕ := flag_area - total_area_first_and_third

-- Given length of the second square
def second_square_length : ℕ := 10

-- Solve for the width of the second square
def second_square_width : ℕ := area_needed_second_square / second_square_length

-- The proof goal
theorem width_of_second_square_is_seven : second_square_width = 7 := by
  sorry

end width_of_second_square_is_seven_l215_215479


namespace max_p_l215_215618

theorem max_p (p q r s t u v w : ℕ)
  (h1 : p + q + r + s = 35)
  (h2 : q + r + s + t = 35)
  (h3 : r + s + t + u = 35)
  (h4 : s + t + u + v = 35)
  (h5 : t + u + v + w = 35)
  (h6 : q + v = 14) :
  p ≤ 20 :=
sorry

end max_p_l215_215618


namespace line_contains_point_l215_215189

theorem line_contains_point (k : ℝ) (x : ℝ) (y : ℝ) (H : 2 - 2 * k * x = -4 * y) : k = -1 ↔ (x = 3 ∧ y = -2) :=
by
  sorry

end line_contains_point_l215_215189


namespace solve_trig_eq_l215_215684

noncomputable theory

open Real

theorem solve_trig_eq (x : ℝ) : (12 * sin x - 5 * cos x = 13) →
  ∃ (k : ℤ), x = (π / 2) + arctan (5 / 12) + 2 * k * π :=
by
s∞rry

end solve_trig_eq_l215_215684


namespace range_of_a_l215_215230

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 :=
by
  sorry

end range_of_a_l215_215230


namespace ellipse_equation_from_hyperbola_l215_215575

theorem ellipse_equation_from_hyperbola :
  (∃ (a b : ℝ), ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) →
  (x^2 / 16 + y^2 / 12 = 1)) :=
by
  sorry

end ellipse_equation_from_hyperbola_l215_215575


namespace lcm_9_16_21_eq_1008_l215_215024

theorem lcm_9_16_21_eq_1008 : Nat.lcm (Nat.lcm 9 16) 21 = 1008 := by
  sorry

end lcm_9_16_21_eq_1008_l215_215024


namespace jim_skips_per_rock_l215_215478

def bob_skips := 12
def total_skips := 270
def num_rocks := 10

theorem jim_skips_per_rock (bob_skips : ℕ) (total_skips : ℕ) (num_rocks : ℕ) 
  (h_bob : bob_skips = 12) (h_total : total_skips = 270) (h_rocks : num_rocks = 10) : 
  (total_skips - bob_skips * num_rocks) / num_rocks = 15 :=
by
  rw [h_bob, h_total, h_rocks]
  sorry

end jim_skips_per_rock_l215_215478


namespace perfect_square_trinomial_m_equals_4_l215_215220

-- Define the polynomial as a perfect square trinomial
def is_perfect_square_trinomial (a b m : ℝ) := (a * x + b) ^ 2 = a ^ 2 * x ^ 2 + 2 * a * b * x + b ^ 2

-- Given conditions
def polynomial (x m: ℝ) := x ^ 2 - 4 * x + m

-- Prove that the given polynomial is a perfect square trinomial if and only if m = 4
theorem perfect_square_trinomial_m_equals_4 (x m : ℝ) : 
  (∃ a b : ℝ, is_perfect_square_trinomial a b m ∧ a = 1 ∧ b = -2) ↔ m = 4 := 
by
  sorry

end perfect_square_trinomial_m_equals_4_l215_215220


namespace volume_TABC_eq_216_l215_215354

noncomputable def volume_pyramid (TA TB TC : ℝ) (h1 : TA = 12) (h2 : TB = 12) (h3 : TC = 9) : ℝ :=
  let base_area := (1 / 2) * TA * TC in
  (1 / 3) * base_area * TB

theorem volume_TABC_eq_216 (TA TB TC : ℝ) (h1 : TA = 12) (h2 : TB = 12) (h3 : TC = 9) :
  volume_pyramid TA TB TC h1 h2 h3 = 216 :=
by {
  -- sorry here to skip the proof
  sorry,
}

end volume_TABC_eq_216_l215_215354


namespace sin_cos_transformation_l215_215914

theorem sin_cos_transformation (x : ℝ) (h : cos x - 7 * sin x = 5) :
  ∃ k, (k = -25 / 7 ∨ k = 15 / 7) ∧ (sin x - 3 * cos x = k) :=
sorry

end sin_cos_transformation_l215_215914


namespace relation_abc_l215_215537

noncomputable def a : ℝ := 3 ^ 1.2
noncomputable def b : ℝ := 3 ^ 0
noncomputable def c : ℝ := (1 / 3) ^ (-0.9)

theorem relation_abc : b < c ∧ c < a := by 
  sorry

end relation_abc_l215_215537


namespace astroid_volume_l215_215494

noncomputable def volume_of_solid (a : ℝ) (ha : a > 0) : ℝ :=
  (3/4) * π^2 * a^3

theorem astroid_volume (a : ℝ) (ha : a > 0) :
  ∃ (V : ℝ), V = volume_of_solid a ha :=
by
  use volume_of_solid a ha
  exact sorry

end astroid_volume_l215_215494


namespace max_boxes_fit_l215_215766

theorem max_boxes_fit 
  (L_large W_large H_large : ℕ) 
  (L_small W_small H_small : ℕ) 
  (h1 : L_large = 12) 
  (h2 : W_large = 14) 
  (h3 : H_large = 16) 
  (h4 : L_small = 3) 
  (h5 : W_small = 7) 
  (h6 : H_small = 2) 
  : ((L_large * W_large * H_large) / (L_small * W_small * H_small) = 64) :=
by
  sorry

end max_boxes_fit_l215_215766


namespace find_length_of_second_train_l215_215017

noncomputable def length_of_second_train
    (length_first_train : ℕ) 
    (speed_first_train : ℕ) 
    (speed_second_train : ℕ)
    (time_to_clear : ℝ) : ℝ := 
  let rel_speed_m_s := (speed_first_train + speed_second_train) * 1000 / 3600
  let total_distance := rel_speed_m_s * time_to_clear
  total_distance - length_first_train

theorem find_length_of_second_train :
  length_of_second_train 120 80 65 7.0752960452818945 ≈ 165.1222222222222 := 
sorry

end find_length_of_second_train_l215_215017


namespace compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l215_215138

theorem compare_pi_314 : Real.pi > 3.14 :=
by sorry

theorem compare_neg_sqrt3_neg_sqrt2 : -Real.sqrt 3 < -Real.sqrt 2 :=
by sorry

theorem compare_2_sqrt5 : 2 < Real.sqrt 5 :=
by sorry

end compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l215_215138


namespace distance_between_vertices_hyperbola_l215_215171

theorem distance_between_vertices_hyperbola :
  let hyperb_eq := 9 * x^2 + 54 * x - y^2 + 10 * y + 55 = 0
  in distance(hyperb_eq) = 2 / 3 :=
by
  sorry

end distance_between_vertices_hyperbola_l215_215171


namespace minimum_city_pairs_connected_l215_215982

theorem minimum_city_pairs_connected : 
  ∃ (n : ℕ), n = 125 ∧ 
  (∀ (cities : Fin n → Fin n → Prop), 
  (∀ c1 c2 c3 c4 : Fin n, (cities c1 c2 ∧ cities c2 c3 ∧ cities c3 c4 ∧ cities c4 c1) ∨
   (cities c1 c3 ∧ cities c3 c2 ∧ cities c2 c4 ∧ cities c4 c1)) → 
  ∃ (p : ℕ), p = 7688 ∧ ∃ (connected_pairs : Fin n → Fin n → Prop), 
  (∀ city : Fin n, ∃ at_least_123_connected : finset.univ.card (λ x, connected_pairs city x) ≥ 123) ∧
  (((finset.card (connected_pairs)) = p))) :=
sorry

end minimum_city_pairs_connected_l215_215982


namespace hyperbola_equation_l215_215905

-- Define the hyperbola passing through a point with same asymptotes

def hyperbola_passes_through {α : Type*} [field α] (C : α → α → Prop) (p : α × α) :=
  C p.1 p.2

def same_asymptotes_hyperbola {α : Type*} [field α] (C : α → α → Prop) :=
  ∃ m ≠ 0, ∀ x y, C x y ↔ (y^2 / 4 - x^2 = m)

theorem hyperbola_equation {α : Type*} [field α] :
  (hyperbola_passes_through (λ x y, x^2 / 3 - y^2 / 12 = 1) (2, 2)) ∧ same_asymptotes_hyperbola (λ x y, x^2 / 3 - y^2 / 12 = 1) :=
by
  sorry

end hyperbola_equation_l215_215905


namespace binom_sum_is_pow_l215_215181

noncomputable def binomial_sum : ℂ :=
  (range 51).sum (λ k, if even k then binom 101 k * (complex.I^k) else 0)

theorem binom_sum_is_pow : binomial_sum = 2^50 := by
  sorry

end binom_sum_is_pow_l215_215181


namespace distinguishing_property_of_rectangles_l215_215101

theorem distinguishing_property_of_rectangles (rect rhomb : Type)
  [quadrilateral rect] [quadrilateral rhomb]
  (sum_of_interior_angles_rect : interior_angle_sum rect = 360)
  (sum_of_interior_angles_rhomb : interior_angle_sum rhomb = 360)
  (diagonals_bisect_each_other_rect : diagonals_bisect_each_other rect)
  (diagonals_bisect_each_other_rhomb : diagonals_bisect_each_other rhomb)
  (diagonals_equal_length_rect : diagonals_equal_length rect)
  (diagonals_perpendicular_rhomb : diagonals_perpendicular rhomb) :
  distinguish_property rect rhomb := by
  sorry

end distinguishing_property_of_rectangles_l215_215101


namespace power_of_two_digit_rearrangement_l215_215577

open Nat

theorem power_of_two_digit_rearrangement (k n : ℕ) (h1 : k > 3) (h2 : n > k) :
    (∀ m : ℕ, rearrange_digits (2^k) m → m ≠ 2^n) :=
by
  sorry

end power_of_two_digit_rearrangement_l215_215577


namespace triangle_inequality_m_l215_215887

noncomputable def m (P Q R : ℝ × ℝ) : ℝ :=
if collinear P Q R then 0 else
let a := dist Q R,
    b := dist R P,
    c := dist P Q in
min (2 * area P Q R / a) 
    (min (2 * area P Q R / b) 
         (2 * area P Q R / c))

theorem triangle_inequality_m {A B C X : ℝ × ℝ} :
  m A B C ≤ m A B X + m A X C + m X B C :=
sorry

end triangle_inequality_m_l215_215887


namespace perpendicular_bisector_of_projections_meets_midpoints_l215_215841

theorem perpendicular_bisector_of_projections_meets_midpoints
  {A B C D S E F H I : Type*}
  [cyclic_quadrilateral A B C D]
  (S_def : S = intersection (line_through_pts A C) (line_through_pts B D))
  (E_projection : orthogonal_projection S A B = E)
  (F_projection : orthogonal_projection S C D = F)
  (H_midpoint : midpoint A D = H)
  (I_midpoint : midpoint B C = I):
  intersects (perpendicular_bisector E F) H ∧ intersects (perpendicular_bisector E F) I := sorry

end perpendicular_bisector_of_projections_meets_midpoints_l215_215841


namespace cos_angle_a_c_l215_215198

variable (a : ℝ^3) (c : ℝ^3)
variable (norm_a : ∥a∥ = 3)
variable (c_val : c = (1, 2, 0))
variable (dot_eq : (a - c) ∙ a = 4)

theorem cos_angle_a_c : real.cos (angle a c) = (real.sqrt 5) / 3 :=
by
  sorry

end cos_angle_a_c_l215_215198


namespace ones_digit_of_highest_power_of_three_dividing_18_factorial_l215_215525

theorem ones_digit_of_highest_power_of_three_dividing_18_factorial :
  ∃ k : ℕ, (3 ^ k ∣ nat.factorial 18) ∧ (3 ^ (k + 1) ∣ nat.factorial 18 = false) ∧ 
  ∃ d : ℕ, d < 10 ∧ (3 ^ k % 10 = d) ∧ (d = 1) :=
sorry

end ones_digit_of_highest_power_of_three_dividing_18_factorial_l215_215525


namespace geometric_sequence_sum_l215_215280

theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : a + a * r = 8)
  (h2 : a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 120) :
  a * (1 + r + r^2 + r^3) = 30 := 
by
  sorry

end geometric_sequence_sum_l215_215280


namespace find_M_l215_215849

variables (r h : ℝ) (π : ℝ)

-- Conditions
def height_of_C_eq_diameter_of_D := 2 * r = 2 * r
def diameter_of_C_eq_height_of_D := h = h
def volume_D_three_times_volume_C :=
  π * r^2 * h = 3 * (π * (h / 2)^2 * (2 * r))

-- Theorem
theorem find_M : 
  height_of_C_eq_diameter_of_D r h ∧ 
  diameter_of_C_eq_height_of_D h ∧ 
  volume_D_three_times_volume_C π r h → 
  3 * (π * (h / 2)^2 * (2 * r)) = (9 * π * h^3) / 4 :=
by
  sorry

end find_M_l215_215849


namespace probability_of_slope_condition_l215_215316

noncomputable theory
open Real

def point_in_unit_square (P : ℝ × ℝ) : Prop :=
  P.1 ∈ Icc 0 1 ∧ P.2 ∈ Icc 0 1

def slope_condition (P : ℝ × ℝ) : Prop :=
  let Q := (5 / 8, 3 / 8)
  in (P.2 - Q.2) / (P.1 - Q.1) ≥ 1 / 2

def area_satisfying_slope_condition (P : ℝ × ℝ) : Prop :=
  point_in_unit_square P ∧ slope_condition P

theorem probability_of_slope_condition :
  ∃ (m n : ℕ), m + n = 171 ∧ gcd m n = 1 :=
sorry

end probability_of_slope_condition_l215_215316


namespace prove_shift_results_in_even_function_l215_215361

noncomputable def shifted_function (x : ℝ) : ℝ :=
  sin (2 * (x + 5 / 12 * π) + 2 * π / 3)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem prove_shift_results_in_even_function :
  is_even_function (λ x, sin (2 * (x + 5 / 12 * π) + 2 * π / 3)) :=
sorry

end prove_shift_results_in_even_function_l215_215361


namespace small_square_perimeter_l215_215827

-- Condition Definitions
def perimeter_difference := 17
def side_length_of_square (x : ℝ) := 2 * x = perimeter_difference

-- Theorem Statement
theorem small_square_perimeter (x : ℝ) (h : side_length_of_square x) : 4 * x = 34 :=
by
  sorry

end small_square_perimeter_l215_215827


namespace perfect_cubes_in_range_l215_215150

theorem perfect_cubes_in_range (K : ℤ) (hK_pos : K > 1) (Z : ℤ) 
  (hZ_eq : Z = K ^ 3) (hZ_range: 600 < Z ∧ Z < 2000) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 :=
by
  sorry

end perfect_cubes_in_range_l215_215150


namespace area_shared_square_circle_l215_215080

theorem area_shared_square_circle (side_length : ℝ) (radius : ℝ) (h1 : side_length = 4) (h2 : radius = 2 * Real.sqrt 2) (center_shared : Prop := True) :
  let circle_area := π * radius^2
  in circle_area = 8 * π :=
by
  sorry

end area_shared_square_circle_l215_215080


namespace train_length_is_300_l215_215815

theorem train_length_is_300 (L V : ℝ)
    (h1 : L = V * 20)
    (h2 : L + 285 = V * 39) :
    L = 300 := by
  sorry

end train_length_is_300_l215_215815


namespace necessary_not_sufficient_l215_215895

theorem necessary_not_sufficient (m a : ℝ) (h : a ≠ 0) :
  (|m| = a → m = -a ∨ m = a) ∧ ¬ (m = -a ∨ m = a → |m| = a) :=
by
  sorry

end necessary_not_sufficient_l215_215895


namespace largest_subset_T_largest_subset_T_specific_l215_215318

noncomputable def max_elements_subset {T : Set ℕ} (hT : T ⊆ (Finset.range 1500).val) 
  (h_diff : ∀ {x y : ℕ}, x ∈ T → y ∈ T → x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 8) : ℕ :=
  -- Placeholder function definition
  580

theorem largest_subset_T :
  ∃ T : Set ℕ, T ⊆ (Finset.range 1500).val ∧ 
              (∀ {x y : ℕ}, x ∈ T → y ∈ T → x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 8) ∧
              @max_elements_subset T (by simp) (by simp) = 580 :=
sorry

-- If using a specific subset
theorem largest_subset_T_specific (T : Set ℕ) 
  (hT : T ⊆ (Finset.range 1500).val) 
  (h_diff : ∀ {x y : ℕ}, x ∈ T → y ∈ T → x ≠ y → |x - y| ≠ 5 ∧ |x - y| ≠ 8) :
  max_elements_subset hT h_diff = 580 :=
sorry

end largest_subset_T_largest_subset_T_specific_l215_215318


namespace find_daily_wage_c_l215_215423

noncomputable def daily_wage_c (total_earning : ℕ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (days_d : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) (ratio_d : ℕ) : ℝ :=
  let total_ratio := days_a * ratio_a + days_b * ratio_b + days_c * ratio_c + days_d * ratio_d
  let x := total_earning / total_ratio
  ratio_c * x

theorem find_daily_wage_c :
  daily_wage_c 3780 6 9 4 12 3 4 5 7 = 119.60 :=
by
  sorry

end find_daily_wage_c_l215_215423


namespace part1_solution_part2_solution_l215_215692

noncomputable def find_prices (price_peanuts price_tea : ℝ) : Prop :=
price_peanuts + 40 = price_tea ∧
50 * price_peanuts = 10 * price_tea

theorem part1_solution :
  ∃ (price_peanuts price_tea : ℝ), find_prices price_peanuts price_tea :=
by
  sorry

def cost_function (m : ℝ) : ℝ :=
6 * m + 36 * (60 - m)

def profit_function (m : ℝ) : ℝ :=
(10 - 6) * m + (50 - 36) * (60 - m)

noncomputable def max_profit := 540

theorem part2_solution :
  ∃ (m t : ℝ), 30 ≤ m ∧ m ≤ 40 ∧ cost_function m ≤ 1260 ∧ profit_function m = max_profit :=
by
  sorry

end part1_solution_part2_solution_l215_215692


namespace floor_x_floor_x_eq_20_l215_215167

theorem floor_x_floor_x_eq_20 (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := 
sorry

end floor_x_floor_x_eq_20_l215_215167


namespace total_interest_is_8500011_l215_215674

noncomputable def total_annual_interest (total_amount : ℝ) (part_y : ℝ) (interest_rate_x : ℝ) (interest_rate_y : ℝ) : ℝ :=
  let part_x := total_amount - part_y in
  (part_x * interest_rate_x) + (part_y * interest_rate_y)

theorem total_interest_is_8500011 :
  total_annual_interest 1600 1100 0.06 0.0500001 = 85.00011 := 
by 
  sorry

end total_interest_is_8500011_l215_215674


namespace abs_neg2023_eq_2023_l215_215373

-- Define a function to represent the absolute value
def abs (x : ℤ) : ℤ :=
  if x < 0 then -x else x

-- Prove that abs (-2023) = 2023
theorem abs_neg2023_eq_2023 : abs (-2023) = 2023 :=
by
  -- In this theorem, all necessary definitions are already included
  sorry

end abs_neg2023_eq_2023_l215_215373


namespace probability_product_divisible_by_3_l215_215265

theorem probability_product_divisible_by_3 :
  let p := (1 : ℚ) - (2 / 3) ^ 6 in
  p = 665 / 729 :=
by
  sorry

end probability_product_divisible_by_3_l215_215265


namespace germs_per_petri_dish_l215_215045

theorem germs_per_petri_dish 
  (total_germs : ℝ) 
  (total_dishes : ℝ) 
  (h_germs : total_germs = 0.037 * 10^5) 
  (h_dishes : total_dishes = 74000 * 10^(-3)) : 
  total_germs / total_dishes = 50 :=
by
  sorry

end germs_per_petri_dish_l215_215045


namespace find_x_l215_215190

theorem find_x :
  let a := 5^3
  let b := 6^2
  a - 7 = b + 82 := 
by
  sorry

end find_x_l215_215190


namespace find_a8_l215_215290

def seq (a : Nat → Int) := a 1 = -1 ∧ ∀ n, a (n + 1) = a n - 3

theorem find_a8 (a : Nat → Int) (h : seq a) : a 8 = -22 :=
by {
  sorry
}

end find_a8_l215_215290


namespace solve_problem_l215_215839

def problem_statement (x y : ℕ) : Prop :=
  (x = 3) ∧ (y = 2) → (x^8 + 2 * x^4 * y^2 + y^4) / (x^4 + y^2) = 85

theorem solve_problem : problem_statement 3 2 :=
  by sorry

end solve_problem_l215_215839


namespace percentage_loss_is_10_l215_215073

-- Define the cost price (CP) and selling price (SP)
def CP : ℝ := 1200
def SP : ℝ := 1080

-- Define the theorem stating the percentage loss
theorem percentage_loss_is_10 :
  let Loss := CP - SP in
  let PercentageLoss := (Loss / CP) * 100 in
  PercentageLoss = 10 :=
by
  sorry

end percentage_loss_is_10_l215_215073


namespace quadratic_solutions_l215_215941

theorem quadratic_solutions (a b : ℝ) :
  (-2 = -a * (-1)^2 + b * (-1)) →
  (-2 = -a * 2^2 + b * 2) →
  (∃ x1 x2, (x1 = -1 ∧ x2 = 2) ∧
  ∀ x, (-a * x^2 + b * x + 2 = 0 → (x = x1 ∨ x = x2))) :=
by
  intros h1 h2
  use [-1, 2]
  split
  { exact ⟨rfl, rfl⟩ }
  intros x hx
  sorry

end quadratic_solutions_l215_215941


namespace velocity_zero_at_2_and_4_l215_215802

-- Defining the displacement function
def displacement (t : ℝ) : ℝ := (1 / 3) * t^3 - 3 * t^2 + 8 * t

-- Define the velocity as the derivative of displacement
def velocity (t : ℝ) : ℝ := derivative (displacement t)

theorem velocity_zero_at_2_and_4 :
  (velocity 2 = 0) ∧ (velocity 4 = 0) :=
by
  -- Placeholder for the proof
  sorry

end velocity_zero_at_2_and_4_l215_215802


namespace find_a_l215_215847

def F (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := a * b^3 + c

theorem find_a (a : ℚ) : F a 2 3 = F a 3 8 → a = -5 / 19 :=
by
  sorry

end find_a_l215_215847


namespace ratio_of_rises_l215_215405

noncomputable def radius_narrower_cone : ℝ := 4
noncomputable def radius_wider_cone : ℝ := 8
noncomputable def sphere_radius : ℝ := 2

noncomputable def height_ratio (h1 h2 : ℝ) : Prop := h1 = 4 * h2

noncomputable def volume_displacement := (4 / 3) * Real.pi * (sphere_radius^3)

noncomputable def new_height_narrower (h1 : ℝ) : ℝ := h1 + (volume_displacement / ((Real.pi * (radius_narrower_cone^2))))

noncomputable def new_height_wider (h2 : ℝ) : ℝ := h2 + (volume_displacement / ((Real.pi * (radius_wider_cone^2))))

theorem ratio_of_rises (h1 h2 : ℝ) (hr : height_ratio h1 h2) :
  (new_height_narrower h1 - h1) / (new_height_wider h2 - h2) = 4 :=
sorry

end ratio_of_rises_l215_215405


namespace inverse_function_exp_l215_215772

theorem inverse_function_exp (x : ℝ) (h : x > 1) : (∃ y : ℝ, y ≥ 1 ∧ y = e^f y) ↔ (f⁻¹ x = log (x - 1)) :=
by
  sorry

end inverse_function_exp_l215_215772


namespace max_value_dist_PQ_evolute_C3_polar_l215_215880

open Real

noncomputable def curve_param (theta : ℝ) : ℝ × ℝ := (cos theta, sin theta)

noncomputable def curve_C2_param (alpha : ℝ) : ℝ × ℝ := (2 * cos alpha, sin alpha)

noncomputable def max_dist_PQ (theta : ℝ) (alpha : ℝ) : ℝ :=
  let P := curve_C2_param alpha
  let Q := (2 * cos theta, sin theta)
  (2 * cos theta - 2 * P.1)^2 + (sin theta - P.2)^2

theorem max_value_dist_PQ :
  ∃ theta : ℝ, ∃ alpha : ℝ, max_dist_PQ theta alpha = (4 * sqrt 3 / 3)^2 := sorry

noncomputable def evolute_C3_eq (alpha : ℝ) : ℝ × ℝ :=
  let x_c := (2 / 3) * cos alpha
  let y_c := (7 / 3) * sin alpha
  (x_c, y_c)

theorem evolute_C3_polar :
  ∀ x y : ℝ, 
  let r := sqrt (x^2 + y^2) in
  r^2 = (1/9) * (4 * x^2 + 49 * y^2) := sorry

end max_value_dist_PQ_evolute_C3_polar_l215_215880


namespace opposite_of_neg_three_fifths_l215_215389

theorem opposite_of_neg_three_fifths :
  -(-3 / 5) = 3 / 5 :=
by
  sorry

end opposite_of_neg_three_fifths_l215_215389


namespace hall_length_l215_215072

theorem hall_length
  (breadth : ℝ) (stone_length_dm stone_width_dm : ℝ) (num_stones : ℕ) (L : ℝ)
  (h_breadth : breadth = 15)
  (h_stone_length : stone_length_dm = 6)
  (h_stone_width : stone_width_dm = 5)
  (h_num_stones : num_stones = 1800)
  (h_length : L = 36) :
  let stone_length := stone_length_dm / 10
  let stone_width := stone_width_dm / 10
  let stone_area := stone_length * stone_width
  let total_area := num_stones * stone_area
  total_area / breadth = L :=
by {
  sorry
}

end hall_length_l215_215072


namespace rebate_difference_not_equal_product_l215_215156

-- Let's formalize the conditions and assertions based on the problem statement.

theorem rebate_difference_not_equal_product (a b : ℕ) (h_permutation : (perm a b)) :
  ∃ a b, ∃ h_perm : (perm a b), 
  let difference := a - b in 
  let product := 2012 * 2013 in 
  ¬ (difference = product) :=
begin
  sorry
end

end rebate_difference_not_equal_product_l215_215156


namespace largest_possible_value_b_l215_215325

theorem largest_possible_value_b : 
  ∃ b : ℚ, (3 * b + 7) * (b - 2) = 4 * b ∧ b = 40 / 15 := 
by 
  sorry

end largest_possible_value_b_l215_215325


namespace smallest_positive_period_maximum_value_monotonically_increasing_interval_l215_215572

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 3) + 2

-- Problem 1: Smallest positive period
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
by sorry

-- Problem 2: Maximum value
theorem maximum_value : ∃ x, f x = 3 :=
by sorry

-- Problem 3: Monotonically increasing interval
theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π →
  f x ≤ f (x + ε) →
  x + ε ≤ 5 * π / 12 + k * π :=
by sorry

end smallest_positive_period_maximum_value_monotonically_increasing_interval_l215_215572


namespace upward_shift_of_parabola_l215_215379

variable (k : ℝ) -- Define k as a real number representing the vertical shift

def original_function (x : ℝ) : ℝ := -x^2 -- Define the original function

def shifted_function (x : ℝ) : ℝ := original_function x + 2 -- Define the shifted function by 2 units upwards

theorem upward_shift_of_parabola (x : ℝ) : shifted_function x = -x^2 + k :=
by
  sorry

end upward_shift_of_parabola_l215_215379


namespace calculate_expression_l215_215485

theorem calculate_expression (x : ℝ) (h₁ : x ≠ 5) (h₂ : x = 4) : (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  sorry

end calculate_expression_l215_215485


namespace sum_bcd_l215_215619

def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n ≤ 4 then 3 else if n ≤ 9 then 5 else 7 -- and so on

theorem sum_bcd :
  let b := 2
      c := -1
      d := 1
  in ∀ n : ℕ, (0 < n) → seq n = b * (Int.floor (Real.sqrt (↑n + c))) + d → b + c + d = 2 :=
  by
    let b := 2
    let c := -1
    let d := 1
    intro n hn seq_n
    sorry

end sum_bcd_l215_215619


namespace unique_property_of_rectangles_l215_215095

-- Definitions of the conditions
structure Quadrilateral (Q : Type*) :=
(sum_of_interior_angles : ∀ (q : Q), angle (interior q) = 360)

structure Rectangle (R : Type*) extends Quadrilateral R :=
(diagonals_bisect_each_other : ∀ (r : R), bisects (diagonals r))
(diagonals_equal_length : ∀ (r : R), equal_length (diagonals r))
(diagonals_not_necessarily_perpendicular : ∀ (r : R), not (necessarily_perpendicular (diagonals r)))

structure Rhombus (H : Type*) extends Quadrilateral H :=
(diagonals_bisect_each_other : ∀ (h : H), bisects (diagonals h))
(diagonals_perpendicular : ∀ (h : H), perpendicular (diagonals h))
(diagonals_not_necessarily_equal_length : ∀ (h : H), not (necessarily_equal_length (diagonals h)))

-- The proof statement
theorem unique_property_of_rectangles (R : Type*) [Rectangle R] (H : Type*) [Rhombus H] :
  ∀ (r : R), ∃ (h : H), equal_length (diagonals r) ∧ not (equal_length (diagonals h)) :=
sorry

end unique_property_of_rectangles_l215_215095


namespace insurance_covers_80_percent_l215_215300

def total_cost : ℝ := 300
def out_of_pocket_cost : ℝ := 60
def insurance_coverage : ℝ := 0.8  -- Representing 80%

theorem insurance_covers_80_percent :
  (total_cost - out_of_pocket_cost) / total_cost = insurance_coverage := by
  sorry

end insurance_covers_80_percent_l215_215300


namespace number_of_triangles_from_8_points_on_circle_l215_215512

-- Definitions based on the problem conditions
def points_on_circle : ℕ := 8

-- Problem statement without the proof
theorem number_of_triangles_from_8_points_on_circle :
  ∃ n : ℕ, n = (points_on_circle.choose 3) ∧ n = 56 := 
by
  sorry

end number_of_triangles_from_8_points_on_circle_l215_215512


namespace albert_brother_younger_l215_215819

variables (A B Y F M : ℕ)
variables (h1 : F = 48)
variables (h2 : M = 46)
variables (h3 : F - M = 4)
variables (h4 : Y = A - B)

theorem albert_brother_younger (h_cond : (F - M = 4) ∧ (F = 48) ∧ (M = 46) ∧ (Y = A - B)) : Y = 2 :=
by
  rcases h_cond with ⟨h_diff, h_father, h_mother, h_ages⟩
  -- Assuming that each step provided has correct assertive logic.
  sorry

end albert_brother_younger_l215_215819


namespace vasya_result_correct_l215_215986

def num : ℕ := 10^1990 + (10^1989 * 6 - 1)
def denom : ℕ := 10 * (10^1989 * 6 - 1) + 4

theorem vasya_result_correct : (num / denom) = (1 / 4) := 
  sorry

end vasya_result_correct_l215_215986


namespace eccentricity_of_ellipse_l215_215385

theorem eccentricity_of_ellipse (e : ℝ) : 
  (∀ P on ellipse, (min_distance P directrix = semi_latus_rectum) → e = (1 / (2^0.5))) := sorry

end eccentricity_of_ellipse_l215_215385


namespace range_of_k_for_two_solutions_l215_215754

theorem range_of_k_for_two_solutions (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ sqrt (9 - x^2) = k * (x - 3) + 4 ∧ sqrt (9 - y^2) = k * (y - 3) + 4) ↔ (7 / 24 < k ∧ k ≤ 2 / 3) :=
by
  sorry

end range_of_k_for_two_solutions_l215_215754


namespace factorial_problem_l215_215147

theorem factorial_problem (N : ℕ) (h : 7! * 11! = 18 * N!) : N = 13 := 
sorry

end factorial_problem_l215_215147


namespace geometric_series_sum_eq_4_div_3_l215_215126

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l215_215126


namespace rectangles_diagonals_equal_not_rhombuses_l215_215099

/--
A proof that the property of having diagonals of equal length is a characteristic of rectangles
and not necessarily of rhombuses.
-/
theorem rectangles_diagonals_equal_not_rhombuses (R : Type) [rectangle R] (H : Type) [rhombus H] :
  (∀ r : R, diagonals_equal r) ∧ ¬(∀ h : H, diagonals_equal h) :=
sorry

end rectangles_diagonals_equal_not_rhombuses_l215_215099


namespace rectangles_have_unique_property_of_equal_diagonals_l215_215104

theorem rectangles_have_unique_property_of_equal_diagonals (rectangle rhombus : Type)
  (is_rectangle : rectangle → Prop)
  (is_rhombus : rhombus → Prop)
  (sum_of_angles_360 : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∑ ang : ℝ in {q}, ang = 360))
  (diagonals_bisect : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∃ midp : Prop, ∀ l r : ℝ, l ≠ r → l + r = 0))
  (diagonals_equal_length : ∀ (r : rectangle), Prop := ∀ x y : ℝ, x = y)
  (diagonals_perpendicular : ∀ (r : rhombus), Prop := ⊥)
  : ∀ r : rectangle, is_rectangle r → diagonals_equal_length r :=
  begin
    intros r h,
    sorry
  end

end rectangles_have_unique_property_of_equal_diagonals_l215_215104


namespace prime_divisor_congruent_one_mod_p_l215_215652

theorem prime_divisor_congruent_one_mod_p (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ q ∣ p^p - 1 ∧ q % p = 1 :=
sorry

end prime_divisor_congruent_one_mod_p_l215_215652


namespace geometric_series_sum_l215_215129

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l215_215129


namespace linear_inequality_solution_set_l215_215038

variable (x : ℝ)

theorem linear_inequality_solution_set :
  ∀ x : ℝ, (2 * x - 4 > 0) → (x > 2) := 
by
  sorry

end linear_inequality_solution_set_l215_215038


namespace problem_statement_l215_215743

def mean (l : List ℚ) : ℚ :=
  (l.sum) / (l.length)

def median (l : List ℚ) : ℚ :=
  if (l.length % 2 = 1)
  then l.nthLe ((l.length) / 2) (by linarith)
  else (l.nthLe ((l.length) / 2 - 1) (by linarith) + l.nthLe ((l.length) / 2) (by linarith)) / 2

def mode (l : List ℚ) : ℚ :=
  l.maxBy (λ x, l.count x)

noncomputable def sum_mean_median_mode (l : List ℚ) : ℚ :=
  (mean l) + (median l) + (mode l)

theorem problem_statement :
  sum_mean_median_mode [2, 3, 4, 3, 1, 6, 3, 7] = 9.625 :=
by
  -- Proof goes here
  sorry

end problem_statement_l215_215743


namespace lambda_mu_result_l215_215539

theorem lambda_mu_result (λ μ : ℝ) 
  (h1 : ∃ k : ℝ, k * (λ + 1) = 6) 
  (h2 : ∃ k : ℝ, k * 0 = 2 * μ - 1) 
  (h3 : ∃ k : ℝ, k * (2 * λ) = 2) : 
  λ * μ = 1 / 10 := by
  sorry

end lambda_mu_result_l215_215539


namespace parabola_focus_property_l215_215800

noncomputable def parabola_focus_condition (p : ℝ) (p_pos : p > 0) 
  (A B : ℝ × ℝ) (F : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  let x1 := A.1 in
  let x2 := B.1 in
  let y1 := A.2 in
  let y2 := B.2 in
  let C := ∀ a : ℝ × ℝ, ∃ (x : ℝ) (y : ℝ), y^2 = 2 * p * x in
  (F = (p / 2, 0)) ∧
  (O = (0, 0)) ∧ 
  (x1 + p / 2 + x2 + p / 2 = 10) ∧ 
  ((x1 + x2 + 0) / 3 = p / 2) 

theorem parabola_focus_property (p : ℝ) (h : parabola_focus_condition p (by linarith) (1,2) (3,4) 0) :
  p = 4 :=
sorry

end parabola_focus_property_l215_215800


namespace range_of_m_l215_215705

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4 * cos x + sin x ^ 2 + m - 4 = 0) ↔ (0 ≤ m ∧ m ≤ 8) :=
by
  sorry

end range_of_m_l215_215705


namespace triangle_inequality_l215_215324

variable (a b c : ℝ) (n : ℕ)
noncomputable def s := (a + b + c) / 2

theorem triangle_inequality (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) (hn : n ≥ 1):
  (a^n/(b+c) + b^n/(c+a) + c^n/(a+b)) ≥ ((2/3)^(n-2) * s^(n-1)) :=
sorry

end triangle_inequality_l215_215324


namespace spadesuit_evaluation_l215_215508

def spadesuit (a b : ℤ) : ℤ := Int.natAbs (a - b)

theorem spadesuit_evaluation :
  spadesuit 5 (spadesuit 3 9) = 1 := 
by 
  sorry

end spadesuit_evaluation_l215_215508


namespace no_scores_of_four_or_five_l215_215975

variable {n x : ℕ}
variable (students : Finset ℕ) -- assuming students are represented by a finite set of ids

-- Given conditions
def nine_people_solved_first_problem : ∀ s ∈ students, s = 9 → s = 1 := sorry
def seven_people_solved_second_problem : ∀ s ∈ students, s = 7 → s = 1 := sorry
def five_people_solved_third_problem : ∀ s ∈ students, s = 5 → s = 1 := sorry
def three_people_solved_fourth_problem : ∀ s ∈ students, s = 3 → s = 1 := sorry
def one_person_solved_fifth_problem : ∀ s ∈ students, s = 1 → s = 1 := sorry
def petya_solved_one_more : ∀ s ∈ students, s ≠ 9 ∧ s ≠ 7 ∧ s ≠ 5 ∧ s ≠ 3 ∧ s ≠ 1 → s + 1 := sorry

-- Prove that no scores of four or five were received
theorem no_scores_of_four_or_five : ∀ s ∈ students, s < 4 ∨ s > 5 := sorry

end no_scores_of_four_or_five_l215_215975


namespace material_needed_l215_215995

-- Define the required conditions
def feet_per_tee_shirt : ℕ := 4
def number_of_tee_shirts : ℕ := 15

-- State the theorem and the proof obligation
theorem material_needed : feet_per_tee_shirt * number_of_tee_shirts = 60 := 
by 
  sorry

end material_needed_l215_215995


namespace a_in_A_sufficient_not_necessary_l215_215555

def A : Set ℝ := {1, 2, 3}
def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem a_in_A_sufficient_not_necessary (a : ℝ) :
  a ∈ A → a ∈ B ∧ (a ∈ B → a ∈ A) = false :=
by
  sorry

end a_in_A_sufficient_not_necessary_l215_215555


namespace factorial_expression_value_l215_215752

theorem factorial_expression_value :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 :=
by
  sorry

end factorial_expression_value_l215_215752


namespace complement_union_l215_215246

def U := { x : ℕ | 0 < x ∧ x < 6 }
def A := {1, 3, 4}
def B := {3, 5}
def complement (S U : Set ℕ) : Set ℕ := U \ S

theorem complement_union :
  complement (A ∪ B) U = {2} :=
by
  sorry

end complement_union_l215_215246


namespace sequence_general_term_l215_215208

theorem sequence_general_term (a : ℕ → ℤ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, 1 < n → a n = 2 * (n + a (n - 1))) :
  ∀ n, 1 ≤ n → a n = 2 ^ (n + 2) - 2 * n - 4 :=
by
  sorry

end sequence_general_term_l215_215208


namespace problem_700_3_in_scientific_notation_l215_215870

-- Definitions
def scientific_notation (a : ℝ) (n : ℤ) : ℝ :=
  a * (10 : ℝ)^n

def valid_scientific_form (a : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10

-- Theorem statement based on the given problem and conditions
theorem problem_700_3_in_scientific_notation :
  ∃ a n, valid_scientific_form a ∧ scientific_notation a n = 700.3 ∧ a = 7.003 ∧ n = 2 :=
by {
  existsi (7.003 : ℝ), 
  existsi (2 : ℤ),
  sorry
}

end problem_700_3_in_scientific_notation_l215_215870


namespace Cara_cookie_price_l215_215890

theorem Cara_cookie_price 
  (side_length_amy : ℕ) (num_cookies_amy : ℕ) (price_per_cookie_amy : ℕ)
  (length_cara : ℕ) (width_cara : ℕ)
  (total_dough : ℕ := num_cookies_amy * (side_length_amy ^ 2)) :
  (price_per_cookie_cara : ℕ) :=
  let area_cara_cookie : ℕ := length_cara * width_cara
  let num_cookies_cara := total_dough / area_cara_cookie
  let total_earnings_amy := num_cookies_amy * price_per_cookie_amy
  let price_per_cookie_cara := total_earnings_amy / num_cookies_cara
  price_per_cookie_cara = 44 := by
  sorry

end Cara_cookie_price_l215_215890


namespace probability_at_least_6_heads_in_8_flips_l215_215790

/-- Let n be the number of flips, and k be the number of consecutive heads required. 
  The probability of getting at least k consecutive heads in n flips of a fair coin is computed. 
  For n = 8 and k = 6, the probability is 3/128. -/
theorem probability_at_least_6_heads_in_8_flips : 
  let n := 8; let k := 6;
  let total_outcomes := (2 ^ n) in
  let successful_outcomes := 6 in
  successful_outcomes / total_outcomes = 3 / 128 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l215_215790


namespace exists_cube_number_divisible_by_six_in_range_l215_215516

theorem exists_cube_number_divisible_by_six_in_range :
  ∃ (y : ℕ), y > 50 ∧ y < 350 ∧ (∃ (n : ℕ), y = n^3) ∧ y % 6 = 0 :=
by 
  use 216
  sorry

end exists_cube_number_divisible_by_six_in_range_l215_215516


namespace correct_statements_about_lines_l215_215417

-- Definitions based on conditions
def is_line_segment (A B : Point) : Prop := 
  ∃ l : Line, A ∈ l ∧ B ∈ l ∧ (∀ x, x ∈ l → dist A x + dist x B = dist A B)

def is_ray (A B : Point) : Prop := 
  exists l : Line, A ∈ l ∧ B ∈ l ∧ (∀ x, x ∈ l → dist A x ≤ dist A B)

def is_straight_line (A B : Point) : Prop := 
  ∃ l : Line, A ∈ l ∧ B ∈ l

-- Problem statement encoded in Lean
theorem correct_statements_about_lines : 
  let A B C : Point in 
  ¬ is_straight_line A B ∧ -- Statement 1
  is_line_segment A B ∧ -- Statement 2
  ¬ is_ray A B ∧ -- Statement 3
  ¬ (is_straight_line A B ∧ dist A B = 5) ∧ -- Statement 4
  ∃ C, is_ray A B ∧ dist A C = 5 -- Statement 5
:=
sorry

end correct_statements_about_lines_l215_215417


namespace eighty_fifth_digit_is_one_l215_215963

theorem eighty_fifth_digit_is_one :
  let seq := ([60, 59, ..., 1] : List ℕ).join
  seq.nth 84 = '1' :=
by
  let digits := seq.toString.data.map (λ c, c - '0')
  exact digits.nth 84
sorry

end eighty_fifth_digit_is_one_l215_215963


namespace shift_upwards_by_2_l215_215381

-- Define the original function
def f (x : ℝ) : ℝ := - x^2

-- Define the shift transformation
def shift_upwards (g : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x, g x + k

-- Define the expected result after shifting f by 2 units upwards
def shifted_f (x : ℝ) : ℝ := - x^2 + 2

-- The proof statement itself
theorem shift_upwards_by_2 :
  shift_upwards f 2 = shifted_f :=
by sorry

end shift_upwards_by_2_l215_215381


namespace positive_difference_of_coordinates_l215_215403

theorem positive_difference_of_coordinates (A B C : ℝ × ℝ)
  (xR yR : ℝ)
  (area_RSC : ℝ) :
  A = (0, 10) →
  B = (3, -1) →
  C = (9, -1) →
  (xR, yR) ∈ line_AC xR →
  (xR, yR) ∈ vertical_line xR →
  area_RSC = 20 →
  abs (xR - yR) = 50 / 9 :=
sorry

end positive_difference_of_coordinates_l215_215403


namespace disjoint_circles_l215_215362

theorem disjoint_circles:
  ∃ (S: ℕ → circle) (rational_points: ℕ → ℚ),
  (∀ i j, i ≠ j → disjoint (S i) (S j)) ∧
  (∀ i, touches_x_axis_at (S i) (rational_points i)) ∧
  (∀ q, q ∈ ℚ → ∃ i, rational_points i = q) ∧
  (¬ ∃ (S: ℝ → circle) (irrational_points: ℝ → ℝ),
  (∀ i j, i ≠ j → disjoint (S i) (S j)) ∧
  (∀ i, touches_x_axis_at (S i) (irrational_points i)) ∧
  (∀ q, q ∈ ℝ \ ℚ → ∃ i, irrational_points i = q)) := sorry

end disjoint_circles_l215_215362


namespace variance_scaled_data_l215_215581

variable {α : Type*} [CommRing α] {x : Fin 5 → α}
variable (s2 : α) (h : s2 = 3)

theorem variance_scaled_data : variance (λ i, 2 * x i) = 12 :=
by
  sorry

end variance_scaled_data_l215_215581


namespace winning_candidate_percentage_l215_215978

/-- 
In an election, a candidate won by a majority of 1040 votes out of a total of 5200 votes.
Prove that the winning candidate received 60% of the votes.
-/
theorem winning_candidate_percentage {P : ℝ} (h_majority : (P * 5200) - ((1 - P) * 5200) = 1040) : P = 0.60 := 
by
  sorry

end winning_candidate_percentage_l215_215978


namespace factorial_simplification_l215_215746

theorem factorial_simplification :
  (10.factorial * 6.factorial * 3.factorial) / (9.factorial * 7.factorial) = 60 / 7 :=
by
-- The proof details would go here
sorry

end factorial_simplification_l215_215746


namespace diagonal_crosses_768_unit_cubes_l215_215456

-- Defining the dimensions of the rectangular prism
def a : ℕ := 150
def b : ℕ := 324
def c : ℕ := 375

-- Computing the gcd values
def gcd_ab : ℕ := Nat.gcd a b
def gcd_ac : ℕ := Nat.gcd a c
def gcd_bc : ℕ := Nat.gcd b c
def gcd_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- Using the formula to compute the number of unit cubes the diagonal intersects
def num_unit_cubes : ℕ := a + b + c - gcd_ab - gcd_ac - gcd_bc + gcd_abc

-- Stating the theorem to prove
theorem diagonal_crosses_768_unit_cubes : num_unit_cubes = 768 := by
  sorry

end diagonal_crosses_768_unit_cubes_l215_215456


namespace angle_between_vectors_l215_215641

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem angle_between_vectors (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ‖(2:ℝ) • a + (3:ℝ) • b‖ = ‖(2:ℝ) • a - (3:ℝ) • b‖) :
  ⟪a, b⟫ = 0 := 
begin
  sorry
end

end angle_between_vectors_l215_215641


namespace arcsin_one_half_eq_pi_six_l215_215493

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry -- Proof omitted

end arcsin_one_half_eq_pi_six_l215_215493


namespace investment_rate_l215_215816

/--
Given:
- initial amount: 12000 USD
- investment 1: 5000 USD at 3%
- investment 2: 4000 USD at 3.5%
- desired yearly income: 430 USD

Prove that the remainder amount should be invested at a rate of 4.67% to achieve the desired yearly income.
-/
theorem investment_rate
  (initial_amount : ℕ)
  (investment_1 : ℕ)
  (rate_1 : ℚ)
  (investment_2 : ℕ)
  (rate_2 : ℚ)
  (desired_income : ℚ)
  (remaining_amount : ℕ)
  (remaining_rate : ℚ) :
  initial_amount = 12000 →
  investment_1 = 5000 →
  rate_1 = 0.03 →
  investment_2 = 4000 →
  rate_2 = 0.035 →
  desired_income = 430 →
  remaining_amount = initial_amount - (investment_1 + investment_2) →
  remaining_rate = 4.67 →
  (investment_1 * rate_1 + investment_2 * rate_2 + remaining_amount * remaining_rate / 100) = desired_income := 
begin
  intros,
  sorry
end

end investment_rate_l215_215816


namespace probability_divisible_by_5_l215_215881

theorem probability_divisible_by_5 : 
  let num_digits := 3 in
  let ones_digit := 3 in
  let tens_digit := 4 in
  let valid_range := Set.Icc 1 9 in
  let valid_numbers := {x | x ∈ valid_range} in
  let valid_x_count := Set.card valid_numbers in
  let valid_x_divisible_by_5 := {x | x ∈ valid_range ∧ ((100 * x + 43) % 5 = 0)} in
  let valid_x_divisible_by_5_count := Set.card valid_x_divisible_by_5 in
  let probability := valid_x_divisible_by_5_count / valid_x_count in
  probability = 1/9 :=
by
  repeat sorry

end probability_divisible_by_5_l215_215881


namespace prove_MN_eq_one_l215_215959

noncomputable def log (a b : ℝ) := real.log b / real.log a

theorem prove_MN_eq_one (M N : ℝ) (h1 : (log M N)^2 = (log N M)^2)
  (h2 : M ≠ N) (h3 : M * N > 0) (h4 : M ≠ 1) (h5 : N ≠ 1) : M * N = 1 := 
by 
  sorry

end prove_MN_eq_one_l215_215959


namespace inequality_solution_set_l215_215228

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : (a - 1) * x > 2) : x < 2 / (a - 1) ↔ a < 1 :=
by
  sorry

end inequality_solution_set_l215_215228


namespace ratio_of_areas_l215_215689

noncomputable def side_length_WXYZ : ℝ := 16

noncomputable def WJ : ℝ := (3/4) * side_length_WXYZ
noncomputable def JX : ℝ := (1/4) * side_length_WXYZ

noncomputable def side_length_JKLM := 4 * Real.sqrt 2

noncomputable def area_JKLM := (side_length_JKLM)^2
noncomputable def area_WXYZ := (side_length_WXYZ)^2

theorem ratio_of_areas : area_JKLM / area_WXYZ = 1 / 8 :=
by
  sorry

end ratio_of_areas_l215_215689


namespace exterior_angle_of_square_and_pentagon_l215_215462

theorem exterior_angle_of_square_and_pentagon 
  (square_interior_angle : ℝ := 90)
  (pentagon_interior_angle : ℝ := 108) :
  let exterior_angle : ℝ := 360 - pentagon_interior_angle - square_interior_angle
  in exterior_angle = 162 :=
by
  sorry

end exterior_angle_of_square_and_pentagon_l215_215462


namespace ring_area_of_equilateral_triangle_l215_215067

theorem ring_area_of_equilateral_triangle (a : ℝ) : 
  let side_length := a,
      semi_perimeter := (3 * a) / 2,
      area_triangle := (real.sqrt 3 / 4) * a^2,
      radius_inscribed_circle := area_triangle / semi_perimeter,
      radius_circumscribed_circle := a / (real.sqrt 3),
      area_ring := real.pi * radius_circumscribed_circle ^ 2 - real.pi * radius_inscribed_circle ^ 2
  in area_ring = real.pi * a^2 / 4 :=
by
  sorry

end ring_area_of_equilateral_triangle_l215_215067


namespace each_album_contains_correct_pictures_l215_215018

def pictures_in_each_album (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat) :=
  (pictures_per_album_phone + pictures_per_album_camera)

theorem each_album_contains_correct_pictures (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat)
  (h1 : pictures_phone = 80)
  (h2 : pictures_camera = 40)
  (h3 : albums = 10)
  (h4 : pictures_per_album_phone = 8)
  (h5 : pictures_per_album_camera = 4)
  : pictures_in_each_album pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera = 12 := by
  sorry

end each_album_contains_correct_pictures_l215_215018


namespace find_nm_2023_l215_215554

theorem find_nm_2023 (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : (n + m) ^ 2023 = -1 := by
  sorry

end find_nm_2023_l215_215554


namespace alternating_even_binomial_sum_l215_215180

theorem alternating_even_binomial_sum :
  (∑ k in finset.Ico 0 51, (-1)^k * nat.choose 101 (2*k)) = -2^50 :=
by
  sorry

end alternating_even_binomial_sum_l215_215180


namespace problem_i_problem_ii_l215_215773

-- Problem (I)
theorem problem_i (x : ℝ) (h : x^(1 / 2) + x^(-1 / 2) = 3) :
  (x^2 + x^(-2) - 7) / (x + x^(-1) + 3) = 4 :=
sorry

-- Problem (II)
theorem problem_ii :
  (2^(1/4))^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (3 / 2)^(-2) = 1 / 2 :=
sorry

end problem_i_problem_ii_l215_215773


namespace hidden_cards_correct_l215_215662

-- Given conditions
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def visible_cards : List ℕ := [1, 3, 4, 6, 7, 8]

-- Ensure uniqueness and constraints
def no_three_consecutive (xs : List ℕ) : Prop := 
  ∀ (a b c : ℕ), (a::b::c::xs).subperm cards → (¬ (a < b ∧ b < c)) ∧ ¬ (c < b ∧ b < a)

-- Correct answers to the problem
def card_A : ℕ := 5
def card_B : ℕ := 2
def card_C : ℕ := 9

-- Proof statement
theorem hidden_cards_correct :
  ∃ (A B C : ℕ),
    (A ∈ cards ∧ B ∈ cards ∧ C ∈ cards) ∧
    (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
    ([1,3,4,6,7,8,A,B,C].perm cards) ∧
    no_three_consecutive [1, 3, 4, 6, 7, 8, A, B, C] ∧
    A = card_A ∧ B = card_B ∧ C = card_C :=
by {
  -- This skips the proof itself as requested
  sorry
}

end hidden_cards_correct_l215_215662


namespace find_a_l215_215242

noncomputable def set_A (a : ℝ) : set (ℝ × ℝ) :=
  {p | |p.fst| + |p.snd| = a ∧ a > 0}

noncomputable def set_B : set (ℝ × ℝ) :=
  {p | |p.fst * p.snd| + 1 = |p.fst| + |p.snd|}

theorem find_a (a : ℝ) :
  (∃ A B, A = set_A a ∧ B = set_B ∧
    (A ∩ B = let octagon_vertices := ...) -- Intersection forming vertices of a regular octagon
  ) → a = 2 + Real.sqrt 2 ∨ a = Real.sqrt 2 :=
sorry

end find_a_l215_215242


namespace number_of_girls_in_club_l215_215728

theorem number_of_girls_in_club (total : ℕ) (C1 : total = 36) 
    (C2 : ∀ (S : Finset ℕ), S.card = 33 → ∃ g b : ℕ, g + b = 33 ∧ g > b) 
    (C3 : ∃ (S : Finset ℕ), S.card = 31 ∧ ∃ g b : ℕ, g + b = 31 ∧ b > g) : 
    ∃ G : ℕ, G = 20 :=
by
  sorry

end number_of_girls_in_club_l215_215728


namespace minimize_path_condition_l215_215526

-- Define the isosceles triangle ABC.
variables (A B C N M : Point)
variable (ABC_isosceles : triangle A B C ∧ (AC = BC))

-- Define the conditions for N and M.
variable (N_on_AC : line AC ∋ N)
variable (M_on_BC : line BC ∋ M)

-- Define the parallel condition.
variable (MN_parallel_AB : parallel (line MN) (line AB))

-- Lean statement to prove the condition for minimizing the path.
theorem minimize_path_condition :
  (∃ N M, N_on_AC N ∧ M_on_BC M ∧ MN_parallel_AB N M ∧ (∀ P Q, 
  P ∈ line AC → Q ∈ line BC → parallel (line PQ) (line AB) → 
  AM + length MN + NB ≤ AP + length PQ + QB)) ↔
  ∠ ACB < 60 :=
sorry

end minimize_path_condition_l215_215526


namespace james_carrot_sticks_left_l215_215626

variable (original_carrot_sticks : ℕ)
variable (eaten_before_dinner : ℕ)
variable (eaten_after_dinner : ℕ)
variable (given_away_during_dinner : ℕ)

theorem james_carrot_sticks_left 
  (h1 : original_carrot_sticks = 50)
  (h2 : eaten_before_dinner = 22)
  (h3 : eaten_after_dinner = 15)
  (h4 : given_away_during_dinner = 8) :
  original_carrot_sticks - eaten_before_dinner - eaten_after_dinner - given_away_during_dinner = 5 := 
sorry

end james_carrot_sticks_left_l215_215626


namespace find_m_plus_n_l215_215441

-- Definitions reflecting the conditions in the problem statement
def num_cards : ℕ := 60
def cards_per_number : ℕ := 5
def removed_pairs : ℕ := 2
def remaining_cards : ℕ := num_cards - 4

-- The combinatorial result after removing pairs from the deck
def total_pairs : ℕ := choose remaining_cards 2
def pairing_ways : ℕ := 11 * choose cards_per_number 2

-- The probability in its lowest terms
def m : ℕ := 11
def n : ℕ := 154
def probability := m / n

-- Proof goal
theorem find_m_plus_n : m + n = 165 := by
  have : m = 11 := rfl
  have : n = 154 := rfl
  show 11 + 154 = 165 from rfl
  sorry

end find_m_plus_n_l215_215441


namespace units_digit_of_n_l215_215764

noncomputable theory
open BigOperators

def units_digit(n : ℕ) : ℕ := n % 10

theorem units_digit_of_n :
  let n := 33 ^ 43 + 43 ^ 32 in
  units_digit n = 8 :=
by
  let n := 33 ^ 43 + 43 ^ 32
  have units_digit_n_eq : units_digit n = 8 := sorry
  exact units_digit_n_eq

end units_digit_of_n_l215_215764


namespace smallest_number_l215_215700

-- Definitions of conditions for H, P, and S
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5
def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def satisfies_conditions_H (H : ℕ) : Prop :=
  is_cube (H / 2) ∧ is_fifth_power (H / 3) ∧ is_square (H / 5)

def satisfies_conditions_P (P A B C : ℕ) : Prop :=
  P / 2 = A^2 ∧ P / 3 = B^3 ∧ P / 5 = C^5

def satisfies_conditions_S (S D E F : ℕ) : Prop :=
  S / 2 = D^5 ∧ S / 3 = E^2 ∧ S / 5 = F^3

-- Main statement: Prove that P is the smallest number satisfying the conditions
theorem smallest_number (H P S A B C D E F : ℕ)
  (hH : satisfies_conditions_H H)
  (hP : satisfies_conditions_P P A B C)
  (hS : satisfies_conditions_S S D E F) :
  P ≤ H ∧ P ≤ S :=
  sorry

end smallest_number_l215_215700


namespace binom_sum_is_pow_l215_215183

noncomputable def binomial_sum : ℂ :=
  (range 51).sum (λ k, if even k then binom 101 k * (complex.I^k) else 0)

theorem binom_sum_is_pow : binomial_sum = 2^50 := by
  sorry

end binom_sum_is_pow_l215_215183


namespace circle_equation_l215_215836

theorem circle_equation
  (a b r : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : r = 2 * Real.sqrt 2)
  (h4 : b = 2)
  (h5 : (2 * Real.sqrt 2)^2 = a^2 + 1) :
  ∃ (x y : ℝ), (x - Real.sqrt 7)^2 + (y - 2)^2 = 8 :=
by {
  use a,
  use b,
  sorry,
}

end circle_equation_l215_215836


namespace sequence_relation_l215_215851

noncomputable def a (n : ℕ) : ℂ :=
if n = 0 then complex.cpow complex.I (2 / 3)
else if n = 1 then complex.cpow complex.I (1 / 3)
else sorry

theorem sequence_relation 
  (a : ℕ → ℂ) 
  (h1 : ∀ (n : ℕ), a (n+1) * a (n-1) - a n ^ 2 + complex.I * (a (n+1) + a (n-1) - 2 * a n) = 0)
  (h2 : a 1 ^ 2 + complex.I * a 1 - 1 = 0) 
  (h3 : a 2 ^ 2 + complex.I * a 2 - 1 = 0)
  : ∀ n : ℕ, a n ^ 2 + a (n+1) ^ 2 + a (n+2) ^ 2 = a n * a (n+1) + a (n+1) * a (n+2) + a (n+2) * a n :=
begin
  sorry
end

end sequence_relation_l215_215851


namespace problem_solution_l215_215006

noncomputable def exists_red_connected_subgraph (points : Finset ℕ) (edges : Finset (Fin (9) × Fin (9))) (is_red : (Fin (9) × Fin (9)) → Prop) : Prop :=
  ∃ (subset : Finset ℕ), subset.card = 4 ∧ ∀ (a b ∈ subset), a ≠ b → is_red (a, b)

theorem problem_solution :
  ∃ points : Finset ℕ, points.card = 9 →
  ∃ edges : Finset (Fin (9) × Fin (9)), edges.card = 36 →
  (∀ (triangle : Finset (Fin (9))), triangle.card = 3 → ∃ (e ∈ triangle × triangle), is_red e) →
  exists_red_connected_subgraph points edges is_red :=
sorry

end problem_solution_l215_215006


namespace triangle_perimeter_is_12_l215_215383

noncomputable def perimeter_of_triangle : Nat :=
  let side1 := 3
  let side2 := 4
  let third_side_roots := [5, 7]
  let valid_side := third_side_roots.filter (λ x => 3 + 4 > x ∧ x > |3 - 4|.natAbs) -- third side must be within valid range
  let third_side := if valid_side.head = 5 then 5 else 7
  side1 + side2 + third_side

theorem triangle_perimeter_is_12 :
  perimeter_of_triangle = 12 :=
by
  sorry

end triangle_perimeter_is_12_l215_215383


namespace max_handshakes_in_group_of_20_l215_215408

theorem max_handshakes_in_group_of_20 
  (G : Fintype (Fin 20)) 
  (handshakes : G → G → Prop) 
  (no_shake_subset : ∀ (a b c : G), 
    ¬(handshakes a b ∧ handshakes b c ∧ handshakes a c)) : 
  ∃ m ≤ 100, ∀ n, n ≤ m :=
by
  sorry

end max_handshakes_in_group_of_20_l215_215408


namespace find_numbers_l215_215519

theorem find_numbers (x y z : ℕ) (h_x_digit : x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_y_digit : y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_z_digit : z ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_positive: x ≠ 0) :
  100*x + 10*y + z = x + y + z + x*y + y*z + z*x + x*y*z ↔
  ∃ n : ℕ, n ∈ {199, 299, 399, 499, 599, 699, 799, 899, 999} := sorry

end find_numbers_l215_215519


namespace find_m_and_k_l215_215238

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m-1)^2 * x^(m^2 - 4*m + 2)
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

theorem find_m_and_k :
  (∀ x > 0, Monotone (f 0 x)) ∧
  (∀ x ∈ Icc (1:ℝ) (2:ℝ), ∃ k ∈ Icc (0:ℝ) (1:ℝ), range (λ x, x^2) ∪ range (λ x, 2^x - k) = range (λ x, x^2))
:= by
  sorry

end find_m_and_k_l215_215238


namespace pure_imaginary_solution_l215_215271

theorem pure_imaginary_solution (m : ℝ) 
  (h : (m^2 + 2*m - 3) + (m-1)*complex.I = (0 : ℂ) + (m-1)*complex.I) : 
  m = -3 :=
sorry

end pure_imaginary_solution_l215_215271


namespace right_angle_clerts_l215_215346

theorem right_angle_clerts (full_circle_clerts : ℕ) (right_angle_degrees : ℕ) : 
  right_angle_degrees = 90 ∧ full_circle_clerts = 500 → 
  (full_circle_clerts / 4) = 125 :=
by
  intros h,
  cases h with h1 h2,
  rw h2,
  exact (Nat.div_eq_of_eq_mul _ _ _ rfl).symm.trans rfl

-- Assume there are 500 clerts in a full circle, and a right angle is 90 degrees
-- Then the number of clerts in a right angle is 125

end right_angle_clerts_l215_215346


namespace determine_angle_A_l215_215209

-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
def sin_rule_condition (a b c A B C : ℝ) : Prop :=
  (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C

-- The proof statement
theorem determine_angle_A (a b c A B C : ℝ) (h : sin_rule_condition a b c A B C) : A = π / 3 :=
  sorry

end determine_angle_A_l215_215209


namespace factorial_expression_value_l215_215751

theorem factorial_expression_value :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 :=
by
  sorry

end factorial_expression_value_l215_215751


namespace range_of_f_find_cos2θ_l215_215571

open Real

-- Definitions for f(x)
def f (x : ℝ) : ℝ := cos x * cos (x + π / 3)

-- Proof problem 1: Prove the range of f(x) in [0, π/2] is [-1/4, 1/2]
theorem range_of_f : set.image f (set.Icc 0 (π / 2)) = set.Icc (-1 / 4) (1 / 2) := 
sorry

-- Proof problem 2: If f(θ) = 13/20 and -π/6 < θ < π/6, find cos 2θ
theorem find_cos2θ (θ : ℝ) (h1 : f θ = 13 / 20) (h2 : -π / 6 < θ) (h3 : θ < π / 6) : cos (2 * θ) = (4 - 3 * sqrt 3) / 10 := 
sorry

end range_of_f_find_cos2θ_l215_215571


namespace triangles_exist_l215_215153

def exists_triangles : Prop :=
  ∃ (T : Fin 100 → Type) 
    (h : (i : Fin 100) → ℝ) 
    (A : (i : Fin 100) → ℝ)
    (is_isosceles : (i : Fin 100) → Prop),
    (∀ i : Fin 100, is_isosceles i) ∧
    (∀ i : Fin 99, h (i + 1) = 200 * h i) ∧
    (∀ i : Fin 99, A (i + 1) = A i / 20000) ∧
    (∀ i : Fin 100, 
      ¬(∃ (cover : (Fin 99) → Type),
        (∀ j : Fin 99, cover j = T j) ∧
        (∀ j : Fin 99, ∀ k : Fin 100, k ≠ i → ¬(cover j = T k))))

theorem triangles_exist : exists_triangles :=
sorry

end triangles_exist_l215_215153


namespace largest_number_l215_215723

theorem largest_number (a b c : ℤ) 
  (h_sum : a + b + c = 67)
  (h_diff1 : c - b = 7)
  (h_diff2 : b - a = 3)
  : c = 28 :=
sorry

end largest_number_l215_215723


namespace inv_of_log_base2_l215_215917

noncomputable def f (x : ℝ) : ℝ := log (x + 4) / log 2
noncomputable def f_inv (y : ℝ) : ℝ := 2 ^ y - 4

theorem inv_of_log_base2 :
  f_inv 2 = 0 :=
by
  -- Using the condition that f_inv is the inverse of f
  have h : f (f_inv 2) = 2 := by
    rw [f_inv]
    simp [f, id]
    sorry -- detailed steps to verify

  -- If f_inv 2 is the inverse function of f at value 2, then f_inv 2 should be 0
  exact id sorry

end inv_of_log_base2_l215_215917


namespace Alice_can_win_4_turns_l215_215820

theorem Alice_can_win_4_turns (A : ℕ) (B : ℕ) 
  (hA : A = 6) 
  (hB : B = 0) 
  (O1 : ∀ A B, A > 0 → B < 6 → (A - 1, B + 1))
  (O2 : ∀ A B, B > 0 → (A - B, B)) :
  ∃ (A' B' : ℕ), (A', B') = (4, 0) :=
  sorry

end Alice_can_win_4_turns_l215_215820


namespace multiple_of_3_iff_has_odd_cycle_l215_215328

-- Define the undirected simple graph G
variable {V : Type} (G : SimpleGraph V)

-- Define the function f(G) which counts the number of acyclic orientations
def f (G : SimpleGraph V) : ℕ := sorry

-- Define what it means for a graph to have an odd-length cycle
def has_odd_cycle (G : SimpleGraph V) : Prop := sorry

-- The theorem statement
theorem multiple_of_3_iff_has_odd_cycle (G : SimpleGraph V) : 
  (f G) % 3 = 0 ↔ has_odd_cycle G := 
sorry

end multiple_of_3_iff_has_odd_cycle_l215_215328


namespace limit_s_zero_as_m_zero_l215_215313

-- Define the intersection point function J
def J (m : ℝ) : ℝ := min (- (6 : ℝ)^(1/3)) m

-- Define the function s
def s (m : ℝ) : ℝ := (J (-m) - J m) / m^2

-- Formalize the proof statement
theorem limit_s_zero_as_m_zero (m : ℝ) (hm : m ≠ 0) : 
  ∀ ε > 0, ∃ δ > 0, ∀ m (h : abs m < δ), abs (s m) < ε :=
by
  sorry

end limit_s_zero_as_m_zero_l215_215313


namespace problem_provable_l215_215364

noncomputable def given_expression (a : ℝ) : ℝ :=
  (1 / (a + 2)) / ((a^2 - 4 * a + 4) / (a^2 - 4)) - (2 / (a - 2))

theorem problem_provable : given_expression (Real.sqrt 5 + 2) = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_provable_l215_215364


namespace sum_series_a_sum_series_b_sum_series_c_l215_215690

-- Part (a)
theorem sum_series_a : (∑' n : ℕ, (1 / 2) ^ (n + 1)) = 1 := by
  --skip proof
  sorry

-- Part (b)
theorem sum_series_b : (∑' n : ℕ, (1 / 3) ^ (n + 1)) = 1/2 := by
  --skip proof
  sorry

-- Part (c)
theorem sum_series_c : (∑' n : ℕ, (1 / 4) ^ (n + 1)) = 1/3 := by
  --skip proof
  sorry

end sum_series_a_sum_series_b_sum_series_c_l215_215690


namespace trajectory_eq_l215_215233

-- Define the conditions provided in the problem
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m + 3) * x + 2 * (1 - 4 * m^2) + 16 * m^4 + 9 = 0

-- Define the required range for m based on the derivation
def m_valid (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

-- Prove that the equation of the trajectory of the circle's center is y = 4(x-3)^2 -1 
-- and it's valid in the required range for x
theorem trajectory_eq (x y : ℝ) :
  (∃ m : ℝ, m_valid m ∧ y = 4 * (x - 3)^2 - 1 ∧ (x = m + 3) ∧ (y = 4 * m^2 - 1)) →
  y = 4 * (x - 3)^2 - 1 ∧ (20/7 < x) ∧ (x < 4) :=
by
  intro h
  cases' h with m hm
  sorry

end trajectory_eq_l215_215233


namespace least_x_value_l215_215739

theorem least_x_value : ∀ x : ℝ, (4 * x^2 + 7 * x + 3 = 5) → x = -2 ∨ x >= -2 := by 
    intro x
    intro h
    sorry

end least_x_value_l215_215739


namespace discount_amount_l215_215632

def tshirt_cost : ℕ := 30
def backpack_cost : ℕ := 10
def blue_cap_cost : ℕ := 5
def total_spent : ℕ := 43

theorem discount_amount : (tshirt_cost + backpack_cost + blue_cap_cost) - total_spent = 2 := by
  sorry

end discount_amount_l215_215632


namespace numbers_not_perfect_square_cube_or_product_l215_215254

theorem numbers_not_perfect_square_cube_or_product :
  let n := 200 in
  let perfect_squares := { x : ℕ | x^2 ≤ n } in
  let perfect_cubes := { x : ℕ | x^3 ≤ n } in
  let sixth_powers := { x : ℕ | x^6 ≤ n } in
  let products := { x : ℕ | ∃ a b : ℕ, a^2 * b^3 = x ∧ a^2 * b^3 ≤ n } in
  let special_numbers := perfect_squares ∪ perfect_cubes ∪ products in
  finset.card (finset.Ico 1 (n+1) \ special_numbers) = 178 :=
sorry

end numbers_not_perfect_square_cube_or_product_l215_215254


namespace line_contains_at_most_one_point_M_l215_215400

theorem line_contains_at_most_one_point_M 
  (A B C M : Point) 
  (h1 : A ≠ B) 
  (h2 : A ≠ C) 
  (h3 : A ≠ M)
  (h4 : angle A B M = angle A C M) :
  ∀ (line : Line), 
    (line_through A line ∧ 
    (line ≠ line_through A B ∧ 
    line ≠ line_through A C ∧ 
    line ≠ angle_bisector A B C ∧ 
    line ≠ tangent A circumcircle ∧ 
    line ≠ tangent A excircle)) → 
    (¬ contains line M) :=
sorry

end line_contains_at_most_one_point_M_l215_215400


namespace find_functions_l215_215517

noncomputable def monotonic_positive_function (f : ℝ → ℝ) : Prop :=
∀ x y > 0, (f (x * y) * f (f y / x)) = 1

theorem find_functions (f : ℝ → ℝ) :
  (monotonic_positive_function f) →
  (∀ x > 0, f(x) > 0) ∧ (∀ x y > 0, f(x) ≤ f(y) ∨ f(x) ≥ f(y)) →
  (∀ x > 0, f(x) = 1 / x) ∨ (∀ x > 0, f(x) = 1)
:=
by 
  intros h1 h2
  sorry

end find_functions_l215_215517


namespace line_equation_midpoint_l215_215384

variable {x y a b : ℝ}

-- Define points A, B, and P
def P : ℝ × ℝ := (1, 3)
def A : ℝ × ℝ := (a, 0)
def B : ℝ × ℝ := (0, b)

-- Define the midpoint condition
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((fst A + fst B) / 2, (snd A + snd B) / 2)

-- Define the equation of the line L based on intercept form
def equation_of_line (x y : ℝ) (a b : ℝ) : Prop :=
  (b ≠ 0) → (a ≠ 0) → (b * x + a * y = a * b)

-- Main theorem to be proved
theorem line_equation_midpoint
  (hP : P = (1, 3))
  (hA : A = (2, 0))
  (hB : B = (0, 6))
  (midpoint_cond : is_midpoint P A B)
  (intercept_form : equation_of_line x y a b) :
  3 * x + y - 6 = 0 :=
sorry

end line_equation_midpoint_l215_215384


namespace expected_value_xiaoqiang_l215_215531

/-- In a relay game with 5 players, each choosing between two paths (left or right) randomly, 
 if more than 2 people choose the same path, each of them gets 1 point, 
 if fewer than 3 people choose the same path, each of them gets 0 points.
 Let ξ denote the score of a specific player named Xiaoqiang.
 Prove that the expected value of ξ is 11/16. -/
theorem expected_value_xiaoqiang :
  ∃ ξ : ℝ, Eξ ξ = 11 / 16 :=
sorry

end expected_value_xiaoqiang_l215_215531


namespace sequence_max_length_l215_215908

def L (M : ℕ) : ℕ :=
  if M = 1 then 2
  else Nat.ceil (3 * M / 2) + 1

theorem sequence_max_length (M : ℕ) (X : ℕ → ℕ) (h1 : ∀ n, X n > 0)
  (h2 : ∀ n, X n ≤ M) (h3 : ∀ k > 2, X k = abs (X (k - 1) - X (k - 2))) :
  ∃ n, L M = n :=
sorry

end sequence_max_length_l215_215908


namespace sum_two_digit_integers_ending_in_06_l215_215412

theorem sum_two_digit_integers_ending_in_06 :
  let ns := [n | n ∈ finset.range (100) ∧ (n ≥ 10) ∧ (n^2 % 100 =  6)] in
  ns.sum = 166 :=
by
  let ns := ([n | n ∈ finset.range (100) ∧ (n ≥ 10) ∧ (n^2 % 100 = 6)] : list ℕ)
  have ns_val : ns = [14, 66, 86] := by sorry
  have sum_val : ns.sum = 166 := by sorry
  exact sum_val

end sum_two_digit_integers_ending_in_06_l215_215412


namespace part_I_part_II_l215_215933

noncomputable def f (x: ℝ) (a: ℝ) : ℝ := x + a * log x
noncomputable def g (x: ℝ) : ℝ := (exp (x - 1)) / x - 1

theorem part_I (a: ℝ) (h: ∃ x0 > 0, f x0 a = 0 ∧ deriv (λ x, f x a) x0 = 0) : a = -exp 1 :=
sorry

theorem part_II (a: ℝ) (h1: a > 0)
(h2: ∀ x1 x2: ℝ, 3 ≤ x1 → 3 ≤ x2 → x1 ≠ x2 → |f x1 a - f x2 a| < |g x1 - g x2|):
  0 < a ∧ a ≤ (2 * exp (2 * 1) / 3) - 3 :=
sorry

end part_I_part_II_l215_215933


namespace collinear_midpoints_circumscribed_quadrilateral_AC_BD_pass_through_P_l215_215111

-- Part 1: Prove that M, N, O are collinear
variables {A B C D O P M N E F G H: Type}
variables [CircumscribedQuadrilateral A B C D O] [Midpoints M N B D A C]

theorem collinear_midpoints_circumscribed_quadrilateral 
  (circumscribed: CircumscribedQuadrilateral A B C D O) 
  (midpoints: Midpoints M N B D A C) : 
  Collinear M N O :=
sorry

-- Part 2: Prove that lines AC and BD also pass through P
variables {circumscribed: CircumscribedQuadrilateral A B C D O}
variables [MeetAt P E G F H]

theorem AC_BD_pass_through_P 
  (circumscribed: CircumscribedQuadrilateral A B C D O )
  (meet: MeetAt P E G F H) : 
  (Collinear A P C) ∧ (Collinear B P D) :=
sorry

end collinear_midpoints_circumscribed_quadrilateral_AC_BD_pass_through_P_l215_215111


namespace slips_with_3_l215_215691

theorem slips_with_3 (x : ℤ) 
    (h1 : 15 > 0) 
    (h2 : 3 > 0 ∧ 9 > 0) 
    (h3 : (3 * x + 9 * (15 - x)) / 15 = 5) : 
    x = 10 := 
sorry

end slips_with_3_l215_215691


namespace simplify_sqrt_l215_215201

theorem simplify_sqrt (x : ℝ) (h : x < 2) : Real.sqrt (x^2 - 4*x + 4) = 2 - x :=
by
  sorry

end simplify_sqrt_l215_215201


namespace num_factors_1728_l215_215589

open Nat

noncomputable def num_factors (n : ℕ) : ℕ :=
  (6 + 1) * (3 + 1)

theorem num_factors_1728 : 
  num_factors 1728 = 28 := by
  sorry

end num_factors_1728_l215_215589


namespace probability_of_quitters_from_10_member_tribe_is_correct_l215_215391

noncomputable def probability_quitters_from_10_member_tribe : ℚ :=
  let total_contestants := 18
  let ten_member_tribe := 10
  let total_quitters := 2
  let comb (n k : ℕ) : ℕ := Nat.choose n k
  
  let total_combinations := comb total_contestants total_quitters
  let ten_tribe_combinations := comb ten_member_tribe total_quitters
  
  ten_tribe_combinations / total_combinations

theorem probability_of_quitters_from_10_member_tribe_is_correct :
  probability_quitters_from_10_member_tribe = 5 / 17 :=
  by
    sorry

end probability_of_quitters_from_10_member_tribe_is_correct_l215_215391


namespace average_number_of_glasses_per_box_l215_215828

theorem average_number_of_glasses_per_box (x : ℕ) :
  (12 * x + 16 * (x + 16) = 480) → (480 / (x + (x + 16)) = 15) :=
begin
  intro h,
  sorry
end

end average_number_of_glasses_per_box_l215_215828


namespace abs_neg_value_l215_215370

-- Definition of absolute value using the conditions given.
def abs (x : Int) : Int :=
  if x < 0 then -x else x

-- Theorem statement that |-2023| = 2023
theorem abs_neg_value : abs (-2023) = 2023 :=
  sorry

end abs_neg_value_l215_215370


namespace trigonometric_identity_l215_215856

theorem trigonometric_identity :
  Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) +
  Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l215_215856


namespace polyhedron_edges_vertices_l215_215805

-- Define the polyhedron with 12 pentagonal faces
def polyhedron_faces : ℕ := 12
def faces_is_pentagon (F : ℕ) : Prop := F = polyhedron_faces ∧ ∀ f, f ∈ {1, 2, 3, ... , F} → ∃ E, edges_of_face f = 5

-- Euler's formula for polyhedra
def euler_formula (V E F : ℕ) : Prop := V - E + F = 2

-- The theorem to prove
theorem polyhedron_edges_vertices (F : ℕ) (EF : faces_is_pentagon F) :
  ∃ (E V : ℕ), E = 30 ∧ V = 20 ∧ euler_formula V E F :=
by
  let E := 30
  let V := 20
  use [E, V]
  split
  exact rfl
  split
  exact rfl
  sorry

end polyhedron_edges_vertices_l215_215805


namespace mike_baseball_cards_l215_215348

theorem mike_baseball_cards :
  let InitialCards : ℕ := 87
  let BoughtCards : ℕ := 13
  (InitialCards - BoughtCards = 74)
:= by
  sorry

end mike_baseball_cards_l215_215348


namespace hockey_games_this_year_l215_215010

-- Define the conditions
def missed_games_this_year : ℕ := 7
def games_last_year : ℕ := 9
def total_games : ℕ := 13

-- Define the statement to be proved
theorem hockey_games_this_year : 
  ∃ x : ℕ, x = total_games - games_last_year ∧ x = 4 := 
by
  exists 4
  simp [total_games, games_last_year]
  sorry

end hockey_games_this_year_l215_215010


namespace polynomial_A_and_value_of_A_plus_2B_l215_215812

def B (x: ℝ) : ℝ := 4 * x^2 - 5 * x - 7

theorem polynomial_A_and_value_of_A_plus_2B :
  (∃ (A : ℝ → ℝ), (∀ x : ℝ, A x - 2 * B x = -2 * x^2 + 10 * x + 14) ∧ A = (λ x, 6 * x^2))
  ∧ (let A := (λ x: ℝ, 6 * x^2) in A (-1) + 2 * B (-1) = 10) :=
by
  sorry

end polynomial_A_and_value_of_A_plus_2B_l215_215812


namespace orthocenter_of_ABC_l215_215553

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨-1, 3, 2⟩
def B : Point3D := ⟨4, -2, 2⟩
def C : Point3D := ⟨2, -1, 6⟩

def orthocenter (A B C : Point3D) : Point3D :=
  -- formula to calculate the orthocenter
  sorry

theorem orthocenter_of_ABC :
  orthocenter A B C = ⟨101 / 150, 192 / 150, 232 / 150⟩ :=
by 
  -- proof steps
  sorry

end orthocenter_of_ABC_l215_215553


namespace arithmetic_sequence_100th_term_l215_215964

theorem arithmetic_sequence_100th_term (a b : ℤ)
  (h1 : 2 * a - a = a) -- definition of common difference d where d = a
  (h2 : b - 2 * a = a) -- b = 3a
  (h3 : a - 6 - b = -2 * a - 6) -- consistency of fourth term
  (h4 : 6 * a = -6) -- equation to solve for a
  : (a + 99 * (2 * a - a)) = -100 := 
sorry

end arithmetic_sequence_100th_term_l215_215964


namespace move_inequality_l215_215332

theorem move_inequality (n m h : ℕ) (h_moves: ∀ k < h, ∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b)^2 ≥ 4 * a * b) 
  (final_state: ∀ i < n, ∀ k ≤ h, ∃ m : ℝ, m^n ≥ 2^(2 * h)) : 
  h ≤ (n / 2) * Real.log 2 m :=
begin
  sorry
end

end move_inequality_l215_215332


namespace mark_total_spending_l215_215343

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end mark_total_spending_l215_215343


namespace range_a_for_monotonic_l215_215570

noncomputable def f : ℝ → ℝ → ℝ
| a, x => if x ≤ 1 then -x^2 + a * x - 2 else log a x

def is_monotonic (f : ℝ → ℝ) := 
  ∀ ⦃x y : ℝ⦄, x ≤ y → f x ≤ f y

theorem range_a_for_monotonic {a : ℝ} :
  (∀ x : ℝ, is_monotonic (f a)) ↔ 2 ≤ a ∧ a ≤ 3 :=
sorry

end range_a_for_monotonic_l215_215570


namespace sum_of_fractions_and_decimal_l215_215727

theorem sum_of_fractions_and_decimal : 
    (3 / 25 : ℝ) + (1 / 5) + 55.21 = 55.53 :=
by 
  sorry

end sum_of_fractions_and_decimal_l215_215727


namespace find_A_and_value_at_minus_one_l215_215811

section polynomial_proof

variable (x : ℝ)

-- Define polynomial B
def B := 4 * x^2 - 5 * x - 7

-- Mistaken calculation result from Xiao Li
def mistaken_calc := -2 * x^2 + 10 * x + 14

-- Define the expected polynomial A
def A := 6 * x^2

-- Define correct calculation of A + 2B
def correct_A_plus_2B := A + 2 * B

-- The main theorem to prove
theorem find_A_and_value_at_minus_one :
  (A - (2 * B) = mistaken_calc) → (A = 6 * x^2) ∧ (correct_A_plus_2B.eval (-1) = 10) :=
by
  sorry

end polynomial_proof

end find_A_and_value_at_minus_one_l215_215811


namespace number_of_factors_N_l215_215831

noncomputable def N : ℕ := 2^5 * 3^3 * 5^2 * 7^2 * 11^1

theorem number_of_factors_N :
  ∀ n, n = N → (∃ k: ℕ, k = 432 ∧ ∀ d: ℕ, d ∣ n → d ∈ (1..k))
:= by
  sorry

end number_of_factors_N_l215_215831


namespace log_one_half_sixteen_l215_215164

theorem log_one_half_sixteen : log (1/2) 16 = -4 := by
  sorry

end log_one_half_sixteen_l215_215164


namespace minimum_posts_required_l215_215453

-- Given conditions:
def field_length : ℕ := 150
def field_width : ℕ := 50
def post_interval_long_side : ℕ := 15
def post_interval_short_side : ℕ := 10

-- Define the total number of posts needed
theorem minimum_posts_required :
  let long_side_posts := field_length / post_interval_long_side + 1,
      short_side_posts := field_width / post_interval_short_side + 1
  in long_side_posts + 2 * (short_side_posts - 1) = 21 := 
sorry

end minimum_posts_required_l215_215453


namespace num_elements_eq_one_l215_215388

theorem num_elements_eq_one :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧
  log (x^3 + 1/3 * y^3 + 1/9) = log x + log y :=
sorry

end num_elements_eq_one_l215_215388


namespace common_element_exists_l215_215053

theorem common_element_exists {S : Fin 2011 → Set ℤ}
  (h_nonempty : ∀ (i : Fin 2011), (S i).Nonempty)
  (h_consecutive : ∀ (i : Fin 2011), ∃ a b : ℤ, S i = Set.Icc a b)
  (h_common : ∀ (i j : Fin 2011), (S i ∩ S j).Nonempty) :
  ∃ a : ℤ, 0 < a ∧ ∀ (i : Fin 2011), a ∈ S i := sorry

end common_element_exists_l215_215053


namespace largest_of_seven_consecutive_integers_l215_215722

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2821) : 
  n + 6 = 406 := 
by
  -- Proof steps can be added here
  sorry

end largest_of_seven_consecutive_integers_l215_215722


namespace real_solution_sum_eq_l215_215953

theorem real_solution_sum_eq (a : ℝ) (h : a > 2) :
  (∃ x : ℝ, (sqrt (a - sqrt (a - x)) = x) ∧ (x = (sqrt (4 * a - 3) - 1) / 2)) :=
sorry

end real_solution_sum_eq_l215_215953


namespace find_length_of_BC_l215_215279

noncomputable def circle_consts :=
  let O := (0, 0)  -- Center of the circle
  let A := (1, 0)  -- Point on the circle, assume AD is on x-axis
  let D := (-1, 0) -- Opposite point on the circle (diameter)
  let B := (cos (pi/3), sin (pi/3))  -- 60-degree from A, on the circle
  let C := (cos (2*pi/3), sin (2*pi/3))  -- 120-degree from A, opposite to B, on the circle
  (O, A, D, B, C)

noncomputable def circle_radius := (6 : ℝ) -- Given BO = 6

noncomputable def chord_length := sqrt(6) - sqrt(2)

theorem find_length_of_BC (O A D B C : ℝ × ℝ)
  (radius : ℝ)
  (h_cent_O : O = (0, 0))
  (h_diam_AD : A = (1, 0) ∧ D = (-1, 0))
  (h_chord_ABC : B = (cos (pi/3), sin (pi/3)) ∧ C = (cos (2*pi/3), sin (2*pi/3)))
  (h_BO_eq_6 : sqrt((B.1 - O.1)^2 + (B.2 - O.2)^2) = 6)
  (h_angle_ABO : ∠A B O = 60)
  : (sqrt((B.1 - C.1)^2 + (B.2 - C.2)^2)) = (3 * (sqrt(6) - sqrt(2))) / 2 := 
by
  sorry -- Proof omitted

end find_length_of_BC_l215_215279


namespace fraction_absent_l215_215284

-- Definitions according to the conditions
def work_per_person (p : ℕ) : ℝ := 1 / p
def increased_work_per_person (p : ℕ) (x : ℝ) : ℝ := 1 / ((1 - x) * p)
def fraction_increase : ℝ := 1 / 7

-- Lean statement for the problem
theorem fraction_absent (p : ℕ) (x : ℝ) (hp: p > 0) (hx : x > 0) :
  increased_work_per_person p x - work_per_person p = fraction_increase * work_per_person p →
  x = 1 / 7 :=
sorry

end fraction_absent_l215_215284


namespace largest_possible_perimeter_l215_215717

theorem largest_possible_perimeter :
  ∃ (l w : ℕ), 8 * l + 8 * w = l * w - 1 ∧ 2 * l + 2 * w = 164 :=
sorry

end largest_possible_perimeter_l215_215717


namespace part_a_7_pieces_l215_215434

theorem part_a_7_pieces (grid : Fin 4 × Fin 4 → Prop) (h : ∀ i j, ∃ n, grid (i, j) → n < 7)
  (hnoTwoInSameCell : ∀ (i₁ i₂ : Fin 4) (j₁ j₂ : Fin 4), (i₁, j₁) ≠ (i₂, j₂) → grid (i₁, j₁) ≠ grid (i₂, j₂))
  : ∀ (rowsRemoved colsRemoved : Finset (Fin 4)), rowsRemoved.card = 2 → colsRemoved.card = 2
    → ∃ i j, ¬ grid (i, j) := by sorry

end part_a_7_pieces_l215_215434


namespace length_of_segment_MN_l215_215698

-- Define the given lengths in the problem
def AB : ℝ := 12
def BC : ℝ := 12
def AC : ℝ := 8

-- Define the angle bisectors' intersection with the legs
def M (x : ℝ) : Prop := line_intersect_angle_bisector_BCA BC AB
def N (x : ℝ) : Prop := line_intersect_angle_bisector_BAC AB BC

-- Given angle bisector properties and parallelism
axiom angle_bisector_property : ∀ (bn nc : ℝ), (bn / nc = AB / AC)
axiom MN_parallel_AC : ∀ (mn : ℝ), parallel mn AC

-- Objective: Prove the length of segment MN is 4.8 cm
theorem length_of_segment_MN : (exists MN, MN = 4.8 ∧ 
  MN_parallel_AC MN ∧ 
  angle_bisector_property (12 - MN) MN) :=
sorry

end length_of_segment_MN_l215_215698


namespace part_I_part_II_l215_215971

-- First part: Prove the magnitude of angle A is π/3
theorem part_I (a b c : ℝ) (h : (2 * c - b) / a = (Real.cos B) / (Real.cos A)) : 
  cos A = 1 / 2 → A = Real.pi / 3 := 
  sorry

-- Second part: Prove the area of triangle ABC is maximized at 5√3 when a = 2√5
theorem part_II (a b c : ℝ) (h1 : a = 2 * Real.sqrt 5) (h2 : (2 * c - b) / a = (Real.cos B) / (Real.cos A)) : 
  ∃ (m : ℝ), m = 5 * Real.sqrt 3 : 
  sorry

end part_I_part_II_l215_215971


namespace exist_100_noncoverable_triangles_l215_215155

theorem exist_100_noncoverable_triangles :
  ∃ (T : Fin 100 → Triangle), (∀ i j : Fin 100, i ≠ j → ¬ (T i ⊆ T j)) ∧
  (∀ i : Fin 99, height (T (i + 1)) = 200 * diameter (T i) ∧ area (T (i + 1)) = area (T i) / 20000) :=
sorry

end exist_100_noncoverable_triangles_l215_215155


namespace savings_percentage_l215_215425

theorem savings_percentage
  (S : ℝ)
  (last_year_saved : ℝ := 0.06 * S)
  (this_year_salary : ℝ := 1.10 * S)
  (this_year_saved : ℝ := 0.10 * this_year_salary)
  (ratio := this_year_saved / last_year_saved * 100):
  ratio = 183.33 := 
sorry

end savings_percentage_l215_215425


namespace expression_evaluation_l215_215867

open Rat

theorem expression_evaluation :
  ∀ (a b c : ℚ),
  c = b - 4 →
  b = a + 4 →
  a = 3 →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 :=
by
  intros a b c hc hb ha h1 h2 h3
  simp [hc, hb, ha]
  have h1 : 3 + 1 ≠ 0 := by norm_num
  have h2 : 7 - 3 ≠ 0 := by norm_num
  have h3 : 3 + 7 ≠ 0 := by norm_num
  -- Placeholder for the simplified expression computation
  sorry

end expression_evaluation_l215_215867


namespace new_rate_of_commission_l215_215713

theorem new_rate_of_commission 
  (R1 : ℝ) (R1_eq : R1 = 0.04) 
  (slump_percentage : ℝ) (slump_percentage_eq : slump_percentage = 0.20000000000000007)
  (income_unchanged : ∀ (B B_new : ℝ) (R2 : ℝ),
    B_new = B * (1 - slump_percentage) →
    B * R1 = B_new * R2 → 
    R2 = 0.05) : 
  true := 
by 
  sorry

end new_rate_of_commission_l215_215713


namespace subset_sum_multiple_of_2n_l215_215200

theorem subset_sum_multiple_of_2n (n : ℕ) (h : n ≥ 4) (S : Finset ℕ) (hS : S.card = n) 
  (hS_sub : S ⊆ Finset.range(2 * n)) : 
  ∃ T ⊆ S, T.sum id % (2 * n) = 0 :=
sorry

end subset_sum_multiple_of_2n_l215_215200


namespace amusement_park_ticket_price_l215_215792

-- Conditions as definitions in Lean
def weekday_adult_ticket_cost : ℕ := 22
def weekday_children_ticket_cost : ℕ := 7
def weekend_adult_ticket_cost : ℕ := 25
def weekend_children_ticket_cost : ℕ := 10
def adult_discount_rate : ℕ := 20
def sales_tax_rate : ℕ := 10
def num_of_adults : ℕ := 2
def num_of_children : ℕ := 2

-- Correct Answer to be proved equivalent:
def expected_total_price := 66

-- Statement translating the problem to Lean proof obligation
theorem amusement_park_ticket_price :
  let cost_before_discount := (num_of_adults * weekend_adult_ticket_cost) + (num_of_children * weekend_children_ticket_cost)
  let discount := (num_of_adults * weekend_adult_ticket_cost) * adult_discount_rate / 100
  let subtotal := cost_before_discount - discount
  let sales_tax := subtotal * sales_tax_rate / 100
  let total_cost := subtotal + sales_tax
  total_cost = expected_total_price :=
by
  sorry

end amusement_park_ticket_price_l215_215792


namespace minimum_theta_l215_215928

noncomputable def f (x : ℝ) : ℝ := sin x + sqrt 3 * cos x

theorem minimum_theta (θ : ℝ) (hθ : θ > 0) :
  (∀ x : ℝ, 2 * sin (2 * x + π / 3 - 2 * θ) = 2 * sin (2 * ( (3 * π / 4 - x) ) + π / 3)) → θ = π / 6 :=
by
  sorry

end minimum_theta_l215_215928


namespace number_of_zeros_of_f_on_interval_0_to_2013_l215_215707

noncomputable def f (x : ℝ) : ℝ := if -1 < x ∧ x ≤ 4 then x^2 - 2^x else f (x - 5)

theorem number_of_zeros_of_f_on_interval_0_to_2013 :
  ∀ (x : ℝ), f(x) - f(x - 5) = 0 → (∀ x ∈ Icc (-1 : ℝ) 4, f x = x^2 - 2^x) →
             (∀ x ∈ Icc (0 : ℝ) 2013, f x = 0 → (x ∈ {2} ∪ ⋃ (n : ℕ) (h : n < 402), (Icc (5 * n + -1) (5 * n + 4)) ∧ ∃ y ∈ Icc (5 * n + -1) (5 * n + 4), f y = 0)) →
             1207 := 
sorry

end number_of_zeros_of_f_on_interval_0_to_2013_l215_215707


namespace coord_of_point_B_l215_215287
-- Necessary import for mathematical definitions and structures

-- Define the initial point A and the translation conditions
def point_A : ℝ × ℝ := (1, -2)
def translation_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1, p.2 + units)

-- The target point B after translation
def point_B := translation_up point_A 1

-- The theorem to prove that the coordinates of B are (1, -1)
theorem coord_of_point_B : point_B = (1, -1) :=
by
  -- Placeholder for proof
  sorry

end coord_of_point_B_l215_215287


namespace triangle_side_length_CF_l215_215294

-- Defining sides of triangle ABC
def AB : ℝ := 13
def BC : ℝ := 30
def CA : ℝ := 23

-- Defining the length of CF to be proven
def CF : ℝ := 24

-- Declaring the conditions for the problem
theorem triangle_side_length_CF
  (h1 : AB = 13)
  (h2 : BC = 30)
  (h3 : CA = 23)
  (h4 : ∃ D : Point, ∀ (D ∈ bisectorAB), D ∈ segmentBC)
  (h5 : ∃ E : Point, on_circumcircle_of_ABC ∧ E ≠ A)
  (h6 : ∃ F : Point, F ∈ circumcircle_of_BED ∧ F ∈ line_of_AB ∧ F ≠ B)
  : CF = 24 :=
sorry

end triangle_side_length_CF_l215_215294


namespace find_monic_polynomials_l215_215166

open Polynomial

noncomputable def P_Q_solution (P Q : Polynomial ℚ) : Prop :=
  P = 1 ∧ Q = X^4 ∨ P = X^4 ∧ Q = 1

theorem find_monic_polynomials (P Q : Polynomial ℚ) :
  Monic P ∧ Monic Q ∧ P^3 + Q^3 = X^12 + 1 → P_Q_solution P Q := 
begin
  sorry
end

end find_monic_polynomials_l215_215166


namespace no_play_students_count_l215_215666

theorem no_play_students_count :
  let total_students := 420
  let football_players := 325
  let cricket_players := 175
  let both_players := 130
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end no_play_students_count_l215_215666


namespace least_three_digit_9_heavy_l215_215466

def is_9_heavy (n : ℕ) : Prop :=
  n % 9 > 5

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem least_three_digit_9_heavy : ∃ n, is_9_heavy n ∧ is_three_digit n ∧ (∀ m, is_9_heavy m ∧ is_three_digit m → n ≤ m) :=
  exists.intro 105 (by
    -- proof steps goes here
    sorry)

end least_three_digit_9_heavy_l215_215466


namespace floor_x_floor_x_eq_20_l215_215168

theorem floor_x_floor_x_eq_20 (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := 
sorry

end floor_x_floor_x_eq_20_l215_215168


namespace num_valid_labelings_l215_215158

-- Definitions of labelings and cube edges
def is_valid_labeling (labels : ℕ → ℕ) : Prop :=
  (∀ i j, labels i ∈ {0, 1}) ∧  -- All labels are either 0 or 1
  (∀ f : Fin 6, (labels (4 * f) + labels (4 * f + 1) + labels (4 * f + 2) + labels (4 * f + 3) = 3))  -- Each face sums to exactly 3

-- Statement to be proved
theorem num_valid_labelings : ∃ (count : ℕ), count = 8 ∧ (∃ f : ℕ → ℕ, is_valid_labeling f) :=
  sorry

end num_valid_labelings_l215_215158


namespace integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l215_215514

-- Proof problem 1
theorem integers_abs_no_greater_than_2 :
    {n : ℤ | |n| ≤ 2} = {-2, -1, 0, 1, 2} :=
by {
  sorry
}

-- Proof problem 2
theorem pos_div_by_3_less_than_10 :
    {n : ℕ | n > 0 ∧ n % 3 = 0 ∧ n < 10} = {3, 6, 9} :=
by {
  sorry
}

-- Proof problem 3
theorem non_neg_int_less_than_5 :
    {n : ℤ | n = |n| ∧ n < 5} = {0, 1, 2, 3, 4} :=
by {
  sorry
}

-- Proof problem 4
theorem sum_eq_6_in_nat :
    {p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0} = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)} :=
by {
  sorry
}

-- Proof problem 5
theorem expressing_sequence:
    {-3, -1, 1, 3, 5} = {x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3} :=
by {
  sorry
}

end integers_abs_no_greater_than_2_pos_div_by_3_less_than_10_non_neg_int_less_than_5_sum_eq_6_in_nat_expressing_sequence_l215_215514


namespace total_toys_l215_215475

theorem total_toys (m a t : ℕ) (h1 : a = m + 3 * m) (h2 : t = a + 2) (h3 : m = 6) : m + a + t = 56 := by
  sorry

end total_toys_l215_215475


namespace minimum_semi_focal_distance_l215_215545

noncomputable def min_semi_focal_distance (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : ℝ :=
  let c := sqrt (a^2 + b^2) in
  if h : (1 / 4 : ℝ) * c + 1 = (a * b) / c 
  then min c 4
  else 0

theorem minimum_semi_focal_distance (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (min_semi_focal_distance a b a_pos b_pos) = 4 := 
  sorry

end minimum_semi_focal_distance_l215_215545


namespace geometric_series_sum_eq_4_over_3_l215_215123

theorem geometric_series_sum_eq_4_over_3 : 
  let a := 1
  let r := 1/4
  inf_geometric_sum a r = 4/3 := by
begin
  intros,
  sorry
end

end geometric_series_sum_eq_4_over_3_l215_215123


namespace rational_solution_l215_215323

theorem rational_solution (a b c : ℚ) 
  (h : (3 * a - 2 * b + c - 4)^2 + (a + 2 * b - 3 * c + 6)^2 + (2 * a - b + 2 * c - 2)^2 ≤ 0) : 
  2 * a + b - 4 * c = -4 := 
by
  sorry

end rational_solution_l215_215323


namespace solve_inequality_l215_215542

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (m n : ℝ) : f (m + n) = f m * f n
axiom f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1

theorem solve_inequality (x : ℝ) : f (x^2) * f (2 * x - 3) > 1 ↔ -3 < x ∧ x < 1 := sorry

end solve_inequality_l215_215542


namespace minimum_value_of_expression_l215_215915

noncomputable def min_value (t : ℝ) (a b : E) [inner_product_space ℝ E] 
  (ha : a ≠ 0) (hb : b ≠ 0) : ℝ :=
  ∥2 • a + t • b∥ / ∥b∥

theorem minimum_value_of_expression {E : Type*} [inner_product_space ℝ E]
  (a b : E) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_angle : real.angle a b = 5 * real.pi / 6)
  (h_eq_norms : ∥a∥ = ∥a + b∥) :
  ∃ t : ℝ, min_value t a b ha hb = (real.sqrt 3) / 3 :=
begin
  sorry
end

end minimum_value_of_expression_l215_215915


namespace median_age_team_l215_215085

def ages : List ℕ := [18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 21, 22]

theorem median_age_team : 
  List.median ages = 19 :=
by
  sorry

end median_age_team_l215_215085


namespace domain_of_h_l215_215738

noncomputable def h (x : ℝ) : ℝ := (2 * x - 3) / (x - 5)

theorem domain_of_h : {x : ℝ | h x ≠ (2 * x - 3) / (x - 5) } = Ioo -∞ 5 ∪ Ioo 5 ∞ :=
sorry

end domain_of_h_l215_215738


namespace volumes_equal_l215_215052

open Real

-- Definitions of the regions and volumes:
def region_R1 (x y : ℝ) := (x^2 = 4 * y ∨ x^2 = -4 * y) ∧ (-4 ≤ y ∧ y ≤ 4) ∧ (-4 ≤ x ∧ x ≤ 4)
def region_R2 (x y : ℝ) := (x^2 - y^2 ≤ 16) ∧ (x^2 + (y - 2)^2 ≥ 4) ∧ (x^2 + (y + 2)^2 ≥ 4)

noncomputable def V1 : ℝ :=
   2 * π * (∫ y in 0..4, 16 - y^2)

noncomputable def V2 : ℝ := V1

theorem volumes_equal :
  V1 = V2 :=
by {
  -- skipping the proof
  sorry
}

end volumes_equal_l215_215052


namespace collinear_Pa_Pb_Pc_l215_215329

variables {A B C O A' Q_a P_a P_b P_c : Type} [Nonempty A] [Nonempty B] [Nonempty C] 
          [Nonempty O] [Nonempty A'] [Nonempty Q_a] [Nonempty P_a] [Nonempty P_b] [Nonempty P_c]

-- Conditions
def is_midpoint (A' : Type) (B C : Type) : Prop :=
  -- Define the midpoint property here.
  sorry

def is_circumcenter (O : Type) (A B C : Type) : Prop :=
  -- Define the circumcenter property here.
  sorry

def intersect_circumcircle (A'' : Type) (AA' : Type) (circumcircle : Type) : Prop :=
  -- Define the intersection with circumcircle property here.
  sorry

def perpendicular (A'Q_a : Type) (AO : Type) : Prop :=
  -- Define the perpendicular property here.
  sorry

def tangent_at_intersection (P_a : Type) (circumcircle : Type) (A'Q_a : Type) (A'') : Prop :=
  -- Define the tangent at intersection property here.
  sorry

def symmetric_construction (P_b : Type) (P_c : Type) : Prop :=
  -- Define the symmetric construction for points P_b and P_c here.
  sorry


theorem collinear_Pa_Pb_Pc 
  (h_midA' : is_midpoint A' B C)
  (h_circumcenter : is_circumcenter O A B C)
  (h_intersection : intersect_circumcircle A'' AA' O)
  (h_perpendicular : perpendicular A'Q_a AO)
  (h_tangent : tangent_at_intersection P_a O A'Q_a A'')
  (h_symmetric : symmetric_construction P_b P_c) :
  collinear P_a P_b P_c := 
sorry

end collinear_Pa_Pb_Pc_l215_215329


namespace number_of_terms_before_five_l215_215593

def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)

theorem number_of_terms_before_five :
  ∃ n : ℕ, arithmetic_sequence 95 (-5) n = 5 ∧ n - 1 = 18 :=
by
  use 19
  split
  · simp [arithmetic_sequence]
    ring
  · rfl

end number_of_terms_before_five_l215_215593


namespace jane_payment_per_bulb_l215_215301

theorem jane_payment_per_bulb :
  let tulip_bulbs := 20
  let iris_bulbs := tulip_bulbs / 2
  let daffodil_bulbs := 30
  let crocus_bulbs := 3 * daffodil_bulbs
  let total_bulbs := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let total_earned := 75
  let payment_per_bulb := total_earned / total_bulbs
  payment_per_bulb = 0.50 := 
by
  sorry

end jane_payment_per_bulb_l215_215301


namespace inequality_subtraction_l215_215954

theorem inequality_subtraction (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end inequality_subtraction_l215_215954


namespace solve_for_m_l215_215957

theorem solve_for_m (m x : ℤ) (h : 4 * x + 2 * m - 14 = 0) (hx : x = 2) : m = 3 :=
by
  -- Proof steps will go here.
  sorry

end solve_for_m_l215_215957


namespace series_sum_correct_l215_215868

/-- Define the initial term of the series -/
def a : ℚ := 2 / 5

/-- Define the common ratio of the series -/
def r : ℚ := 1 / 2

/-- Define the infinite geometric series sum formula for |r| < 1 -/
def series_sum (a : ℚ) (r : ℚ) (hr : |r| < 1) : ℚ :=
  a / (1 - r)

/-- Prove that the sum of the given infinite geometric series is 4/5 -/
theorem series_sum_correct : series_sum a r (by norm_num) = (4 / 5) := 
  sorry

end series_sum_correct_l215_215868


namespace average_rate_of_change_l215_215697

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 6 := 
by 
  calc
    (f 3 - f 1) / (3 - 1) = (18 - 6) / 2 : by rw [f]; norm_num
                        ... = 12 / 2   : by norm_num
                        ... = 6    : by norm_num

end average_rate_of_change_l215_215697


namespace eq_of_line_through_center_eq_of_line_bisects_chord_length_of_chord_slant_angle_l215_215898

theorem eq_of_line_through_center (x y : ℝ) :
  ∀ (P : ℝ × ℝ), P = (2, 2) → 
  ∀ (C : ℝ × ℝ), C = (1, 0) →
  ∀ (hx : (x - 1)^2 + y^2 = 9),
  (2x - y - 2 = 0) :=
sorry

theorem eq_of_line_bisects_chord (x y : ℝ) :
  ∀ (P : ℝ × ℝ), P = (2, 2) →
  ∀ (C : ℝ × ℝ), C = (1, 0) →
  ∀ (hx : (x - 1)^2 + y^2 = 9),
  (x + 2y - 6 = 0) :=
sorry

theorem length_of_chord_slant_angle (x y : ℝ) :
  ∀ (P : ℝ × ℝ), P = (2, 2) →
  ∀ (C : ℝ × ℝ), C = (1, 0) →
  ∀ (hx : (x - 1)^2 + y^2 = 9),
  (x - y = 0) →
  (|2 * sqrt(9 - ((1 : ℝ) / sqrt(2))^2)| = sqrt(34)) :=
sorry

end eq_of_line_through_center_eq_of_line_bisects_chord_length_of_chord_slant_angle_l215_215898


namespace fraction_subtraction_l215_215020

theorem fraction_subtraction : 
  (3 + 6 + 9) = 18 ∧ (2 + 5 + 8) = 15 ∧ (2 + 5 + 8) = 15 ∧ (3 + 6 + 9) = 18 →
  (18 / 15 - 15 / 18) = 11 / 30 :=
by
  intro h
  sorry

end fraction_subtraction_l215_215020


namespace angle_B_area_of_triangle_l215_215610

-- Definitions and conditions for Problem 1
def acute_triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0

def condition1 (a b : ℝ) (A : ℝ) : Prop :=
  2 * b * sin A = sqrt 3 * a

-- Problem 1
theorem angle_B (A B C a b c : ℝ) (h1 : acute_triangle A B C a b c) (h2 : condition1 a b A)
  : B = π / 3 := sorry

-- Definitions and conditions for Problem 2
def condition2 (a c : ℝ) : Prop :=
  a + c = 8

def condition3 (a c : ℝ) : Prop :=
  a * c = 28 / 3

def triangle_area (a c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * sin B

-- Problem 2
theorem area_of_triangle (B A C a b c S : ℝ) (h1 : acute_triangle A B C a b c) (h2 : b = 6) (h3 : B = π / 3)
  (h4 : condition2 a c) (h5 : condition3 a c) : S = 7 * sqrt 3 / 3 :=
  sorry

end angle_B_area_of_triangle_l215_215610


namespace find_naturals_by_digits_l215_215706

theorem find_naturals_by_digits (n : ℕ) (h : List.perm (Nat.digits 10 (n ^ 5)) [1, 2, 3, 3, 7, 9]) : n = 13 :=
sorry

end find_naturals_by_digits_l215_215706


namespace num_real_solutions_to_eqn_l215_215175

theorem num_real_solutions_to_eqn : 
  (∀ x : ℝ, 9 * x^2 - 63 * (⌊x⌋) + 72 = 0 → (x = xpos ∨ x = -xpos → (∃ (n : ℕ), n = 6))) :=
begin
  sorry
end

end num_real_solutions_to_eqn_l215_215175


namespace round_155_628_l215_215735

theorem round_155_628 :
  round (155.628 : Real) = 156 := by
  sorry

end round_155_628_l215_215735


namespace min_value_expression_l215_215334

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (9 / a) + (16 / b) + (25 / c)

theorem min_value_expression :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 6 →
  min_expression a b c ≥ 18 :=
by
  intro a b c ha hb hc habc
  sorry

end min_value_expression_l215_215334


namespace ratio_largest_middle_l215_215774

-- Definitions based on given conditions
def A : ℕ := 24  -- smallest number
def B : ℕ := 40  -- middle number
def C : ℕ := 56  -- largest number

theorem ratio_largest_middle (h1 : C = 56) (h2 : A = C - 32) (h3 : A = 24) (h4 : B = 40) :
  C / B = 7 / 5 := by
  sorry

end ratio_largest_middle_l215_215774


namespace polynomial_root_k_value_l215_215855

theorem polynomial_root_k_value 
  (a b k : ℝ)
  (h_roots : (3 * a^3 - 9 * a^2 - 81 * a + k = 0) ∧ (3 * (2*a)^3 - 9 * (2*a)^2 - 81 * (2*a) + k = 0 ∧ a + 2*a + b = 3))
  (h_k_positive : k > 0) :
  k = -6 * ( (9 + real.sqrt 837) / 14 )^2 * ( 3 - 3 * ( (9 + real.sqrt 837) / 14 ) ) :=
begin
  sorry
end

end polynomial_root_k_value_l215_215855


namespace equation_of_line_l215_215274

theorem equation_of_line (θ : ℝ) (P : ℝ × ℝ) (l : ℝ → ℝ) (hθ : θ = 135) (hP : P = (1, 1)) (hl : ∀ x, l x = -x + 2) :
  ∃ m b, (∀ x, l x = m * x + b) ∧ m = tan θ ∧ (P.snd = m * P.fst + b) :=
by
  sorry

end equation_of_line_l215_215274


namespace geometric_sequence_proof_l215_215672

theorem geometric_sequence_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : b^2 = a * c) :
  ∃ r : ℝ, r > 0 ∧ a + b + c = r * (sqrt (3 * (a * b + b * c + c * a))) ∧ (sqrt (3 * (a * b + b * c + c * a))) = r * (cbrt (27 * a * b * c)) :=
sorry

end geometric_sequence_proof_l215_215672


namespace acute_angle_at_3_15_l215_215028

/-- Each minute on a clock represents 6 degrees. -/
def degrees_per_minute := 6

/-- At 3:00, the hour hand is at 90 degrees from the 12 o'clock position. -/
def degrees_at_3 := 3 * 30

/-- The hour hand moves 7.5 degrees further from the 3 o'clock position by 3:15. -/
def degrees_hour_hand_at_3_15 := degrees_at_3 + (15 / 60) * 30

/-- The minute hand is exactly 90 degrees from the 12 o'clock position at 3:15. -/
def degrees_minute_hand_at_3_15 := 15 * degrees_per_minute

/-- The acute angle formed between the hour and minute hands of a clock at 3:15 is 7.5 degrees. -/
theorem acute_angle_at_3_15 : abs (degrees_hour_hand_at_3_15 - degrees_minute_hand_at_3_15) = 7.5 := by
  sorry

end acute_angle_at_3_15_l215_215028


namespace part1_part2_l215_215536

theorem part1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) :=
sorry

theorem part2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = a * b ∧ (|2 * a - 1| + |3 * b - 1| = 2 * Real.sqrt 6 + 3)) :=
sorry

end part1_part2_l215_215536


namespace percentage_of_earrings_l215_215118

theorem percentage_of_earrings (B M R : ℕ) (hB : B = 10) (hM : M = 2 * R) (hTotal : B + M + R = 70) : 
  (B * 100) / M = 25 := 
by
  sorry

end percentage_of_earrings_l215_215118


namespace geometric_series_sum_eq_4_div_3_l215_215127

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l215_215127


namespace coeff_x3_in_x_mul_1_add_x_pow_6_l215_215145

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem coeff_x3_in_x_mul_1_add_x_pow_6 :
  ∀ x : ℕ, (∃ c : ℕ, c * x^3 = x * (1 + x)^6 ∧ c = 15) :=
by
  sorry

end coeff_x3_in_x_mul_1_add_x_pow_6_l215_215145


namespace slant_asymptote_sum_l215_215177

noncomputable def rational_function : ℚ[X] → ℚ[X] → (ℚ[X] → ℚ[X]) := 
  λ num den, num / den

theorem slant_asymptote_sum (x : ℝ) :
  let y := (3 * x^2 - 2 * x + 5) / (x^2 - 4 * x + 3),
      slant_asymptote := ∀ x (h : x → real), y → mx + b
in (0 + 3) = 3 := 
sorry

end slant_asymptote_sum_l215_215177


namespace train_length_l215_215814

-- Define the given conditions
def train_cross_time : ℕ := 40 -- time in seconds
def train_speed_kmh : ℕ := 144 -- speed in km/h

-- Convert the speed from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ :=
  (speed_kmh * 5) / 18 

def train_speed_ms : ℕ := kmh_to_ms train_speed_kmh

-- Theorem statement
theorem train_length :
  train_speed_ms * train_cross_time = 1600 :=
by
  sorry

end train_length_l215_215814


namespace existence_of_zero_point_l215_215877

def f (x : ℝ) : ℝ := log x / log 2 + x - 2

theorem existence_of_zero_point :
  (∀ x > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs (f y - f x) < ε) → -- Continuity condition for Intermediate Value Theorem
  f 1 < 0 → 
  f 2 > 0 →
  ∃ c, 1 < c ∧ c < 2 ∧ f c = 0 :=
by
  sorry

end existence_of_zero_point_l215_215877


namespace angle_C_magnitude_area_triangle_l215_215969

variable {a b c A B C : ℝ}

namespace triangle

-- Conditions and variable declarations
axiom condition1 : 2 * b * Real.cos C = a * Real.cos C + c * Real.cos A
axiom triangle_sides : a = 3 ∧ b = 2 ∧ c = Real.sqrt 7

-- Prove the magnitude of angle C is π/3
theorem angle_C_magnitude : C = Real.pi / 3 :=
by sorry

-- Prove that given b = 2 and c = sqrt(7), a = 3 and the area of triangle ABC is 3*sqrt(3)/2
theorem area_triangle :
  (b = 2 ∧ c = Real.sqrt 7 ∧ C = Real.pi / 3) → 
  (a = 3 ∧ (1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2)) :=
by sorry

end triangle

end angle_C_magnitude_area_triangle_l215_215969


namespace Emily_sixth_quiz_score_l215_215159

theorem Emily_sixth_quiz_score :
  let scores := [92, 95, 87, 89, 100]
  ∃ s : ℕ, (s + scores.sum : ℚ) / 6 = 93 :=
  by
    sorry

end Emily_sixth_quiz_score_l215_215159


namespace problem_value_of_reciprocal_sum_l215_215598

theorem problem_value_of_reciprocal_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_log_eq : 2 + log 2 a = 3 + log 3 b ∧ 3 + log 3 b = log 6 (a + b)) : 
  1 / a + 1 / b = 108 := 
by
  sorry

end problem_value_of_reciprocal_sum_l215_215598


namespace ratio_of_areas_l215_215703

theorem ratio_of_areas (A B C D : Point) (AC BD : Line) 
  (h_diameter : AC.is_diameter_of_circumcircle ABCD)
  (h_ratio : BD.divides AC 2 5)
  (h_angle : angle BAC = 45) :
  area ABC / area ACD = 29 / 20 := 
sorry

end ratio_of_areas_l215_215703


namespace triangle_area_tripled_sides_l215_215465

variable (a b : ℝ) (θ : ℝ)

theorem triangle_area_tripled_sides (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ θ ∧ θ ≤ π) :
  let original_area := (1 / 2) * a * b * Real.sin θ in
  let new_area := (1 / 2) * (3 * a) * (3 * b) * Real.sin θ in
  new_area = 9 * original_area := 
by 
  let original_area := (1 / 2) * a * b * Real.sin θ
  let new_area := (1 / 2) * (3 * a) * (3 * b) * Real.sin θ
  sorry

end triangle_area_tripled_sides_l215_215465


namespace ordered_pairs_count_l215_215333

theorem ordered_pairs_count (p q : ℕ) (hp : p > 13)
  (p_eq : p = 2 * q + 1) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) :
  ∃ k, k = q - 1 ∧ 
       (∀ m n, 0 ≤ m → m < n → n < p - 1 →
        (3^m + (-12)^m ≡ 3^n + (-12)^n [MOD p]) ↔ 
        (∃ i, i ≤ q - 1 ∧ (m = i ∨ n = i))) := sorry

end ordered_pairs_count_l215_215333


namespace arithmetic_sequence_properties_l215_215502

theorem arithmetic_sequence_properties
    (a_1 : ℕ)
    (d : ℕ)
    (sequence : Fin 240 → ℕ)
    (h1 : ∀ n, sequence n = a_1 + n * d)
    (h2 : sequence 0 = a_1)
    (h3 : 1 ≤ a_1 ∧ a_1 ≤ 9)
    (h4 : ∃ n₁, sequence n₁ = 100)
    (h5 : ∃ n₂, sequence n₂ = 3103) :
  (a_1 = 9 ∧ d = 13) ∨ (a_1 = 1 ∧ d = 33) ∨ (a_1 = 9 ∧ d = 91) :=
sorry

end arithmetic_sequence_properties_l215_215502


namespace factorial_division_l215_215747

theorem factorial_division :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 := by
  sorry

end factorial_division_l215_215747


namespace distinguishing_property_of_rectangles_l215_215100

theorem distinguishing_property_of_rectangles (rect rhomb : Type)
  [quadrilateral rect] [quadrilateral rhomb]
  (sum_of_interior_angles_rect : interior_angle_sum rect = 360)
  (sum_of_interior_angles_rhomb : interior_angle_sum rhomb = 360)
  (diagonals_bisect_each_other_rect : diagonals_bisect_each_other rect)
  (diagonals_bisect_each_other_rhomb : diagonals_bisect_each_other rhomb)
  (diagonals_equal_length_rect : diagonals_equal_length rect)
  (diagonals_perpendicular_rhomb : diagonals_perpendicular rhomb) :
  distinguish_property rect rhomb := by
  sorry

end distinguishing_property_of_rectangles_l215_215100


namespace triangle_ratio_l215_215432

theorem triangle_ratio :
  ∀ (A B C K L : Type)
    [is_right_triangle A B C]
    (h1 : angle A = 50)
    (h2 : is_point_on_cathethus K BC)
    (h3 : is_point_on_cathethus L BC)
    (h4 : angle KAC = 10)
    (h5 : angle LAB = 10),
  CK / LB = 2 := 
sorry

end triangle_ratio_l215_215432


namespace total_days_off_l215_215088

-- Definitions for the problem conditions
def days_off_personal (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_professional (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_teambuilding (quarters_in_year : ℕ) (days_per_quarter : ℕ) : ℕ :=
  days_per_quarter * quarters_in_year

-- Main theorem to prove
theorem total_days_off
  (months_in_year : ℕ) (quarters_in_year : ℕ)
  (days_per_month_personal : ℕ) (days_per_month_professional : ℕ) (days_per_quarter_teambuilding: ℕ)
  (h_months : months_in_year = 12) (h_quarters : quarters_in_year = 4) 
  (h_days_personal : days_per_month_personal = 4) (h_days_professional : days_per_month_professional = 2) (h_days_teambuilding : days_per_quarter_teambuilding = 1) :
  days_off_personal months_in_year days_per_month_personal
  + days_off_professional months_in_year days_per_month_professional
  + days_off_teambuilding quarters_in_year days_per_quarter_teambuilding
  = 76 := 
by {
  -- Calculation
  sorry
}

end total_days_off_l215_215088


namespace arithmetic_sequence_difference_property_l215_215185

theorem arithmetic_sequence_difference_property
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (d : ℤ)
  (h_a_sequence : ∀ n, a (n + 1) - a n = d)
  (h_S_def : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h_d_value : d = 3) :
  (S 20 - S 10, S 30 - S 20, S 40 - S 30) forms_arithmetic_sequence_with_common_difference 300 :=
sorry

end arithmetic_sequence_difference_property_l215_215185


namespace smallest_integer_x_l215_215742

theorem smallest_integer_x (x : ℤ) (h : x < 3 * x - 12) : x ≥ 7 :=
sorry

end smallest_integer_x_l215_215742


namespace parallelogram_area_proof_l215_215832

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]

noncomputable theory

def parallelogram_area (p q : V) (a b : V) (hp : ∥p∥ = 7)
                      (hq : ∥q∥ = 2) (theta : real.angle) 
                      (htheta : theta = real.pi / 4)
                      (ha : a = 4 • p + q)
                      (hb : b = p - q) : ℝ :=
  (∥a ×ₗ b∥ : ℝ)

theorem parallelogram_area_proof (p q : V)
                      (a b : V) (hp : ∥p∥ = 7)
                      (hq : ∥q∥ = 2) 
                      (theta : real.angle) 
                      (htheta : theta = real.pi / 4)
                      (ha : a = 4 • p + q)
                      (hb : b = p - q) :
                      parallelogram_area p q a b hp hq theta htheta ha hb = 35 * real.sqrt 2 :=
by sorry

end parallelogram_area_proof_l215_215832


namespace geom_series_sum_l215_215134

theorem geom_series_sum : 
  let a := 1 : ℝ
  let r := 1 / 4 : ℝ
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geom_series_sum_l215_215134


namespace evaluate_expression_l215_215866

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : (2 * x - b + 5) = (b + 23) :=
by
  sorry

end evaluate_expression_l215_215866


namespace problem_1_problem_2_problem_3_l215_215442

noncomputable def S (n : ℕ) (a : ℕ) : ℕ := 2^(n+6) - a
noncomputable def a_n (n : ℕ) : ℕ := 2^(n+5)
noncomputable def b_n (n : ℕ) (a_seq : ℕ → ℕ) : ℕ := (1 / n) * (Finset.sum (Finset.range n) (λ i, log 2 (a_seq i)))

theorem problem_1 (n : ℕ) : 
  a_n 1 = 64 ∧ (∀ k, a_n (k+1) - a_n k = 2^(k+5)) := 
by
  sorry

theorem problem_2 (n : ℕ) : 
  (Finset.sum (Finset.range n) (λ i, 1 / (b_n i a_n * b_n (i + 1) a_n))) = 
  4 * (1 / 12 - 1 / (n + 12)) := 
by
  sorry

theorem problem_3 (n : ℕ) : 
  (∀ k, (a_n k / b_n k a_n) > (a_n (k + 1) / b_n (k + 1) a_n)) ≥ 32 / 3 :=
by
  sorry

end problem_1_problem_2_problem_3_l215_215442


namespace pastries_cannot_be_determined_l215_215829

-- Define the conditions given in the problem.
def cakes_initial : ℕ := 149
def cakes_sold : ℕ := 10
def pastries_sold : ℕ := 90
def cakes_left : ℕ := 139

-- State the proof problem.
theorem pastries_cannot_be_determined : 
  (cakes_initial - cakes_sold = cakes_left) →
  ∃ p : ℕ, ∀ pastries_initial : ℕ, pastries_initial = p → pastries_initial ≠ ?m_1
:= by 
  sorry

end pastries_cannot_be_determined_l215_215829


namespace evaluate_statement_b_l215_215050

noncomputable def quadratic_roots (p : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + p * x1 + 4 = 0 ∧ x2^2 + p * x2 + 4 = 0

theorem evaluate_statement_b (p x1 x2 : ℝ) (h : quadratic_roots p) (hp : |x1 + x2| = |p|) : |x1 + x2| > 4 :=
  by {
    cases h with x1 hx,
    cases hx with x2 hxy,
    exact sorry
  }

end evaluate_statement_b_l215_215050


namespace number_of_clients_l215_215460

theorem number_of_clients (cars_clients_selects : ℕ)
                          (cars_selected_per_client : ℕ)
                          (each_car_selected_times : ℕ)
                          (total_cars : ℕ)
                          (h1 : total_cars = 18)
                          (h2 : cars_clients_selects = total_cars * each_car_selected_times)
                          (h3 : each_car_selected_times = 3)
                          (h4 : cars_selected_per_client = 3)
                          : total_cars * each_car_selected_times / cars_selected_per_client = 18 :=
by {
  sorry
}

end number_of_clients_l215_215460


namespace fixed_point_of_exponential_function_l215_215932

-- The function definition and conditions are given as hypotheses
theorem fixed_point_of_exponential_function
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ P : ℝ × ℝ, (∀ x : ℝ, (x = 1) → P = (x, a^(x-1) - 2)) → P = (1, -1) :=
by
  sorry

end fixed_point_of_exponential_function_l215_215932


namespace cyclists_speed_l215_215733

theorem cyclists_speed 
  (same_place : Prop) 
  (opposite_directions : Prop) 
  (same_speed : Prop) 
  (time : ℝ) 
  (distance_apart : ℝ) 
  (h_time : time = 2.5) 
  (h_distance : distance_apart = 50) 
  (h_equation : distance_apart = 2 * v * time) 
  : v = 10 := 
by {
  have equation := calc
    distance_apart 
    = 2 * v * time : h_equation
  ... = 2.5 * 2 * v : by rw [h_time]
  ... = 5 * v : by rw [← mul_assoc]
  ... = 50 : h_distance,
  exact eq_of_mul_eq_mul_right zero_ne_five _ equation,
  sorry
}

end cyclists_speed_l215_215733


namespace ratio_of_areas_l215_215732

theorem ratio_of_areas (O P X : Point) (r : ℝ)
  (h1 : ∃ κ, κ = 3 ∧ X = O + (P - O) / (1 + κ))    -- X divides OP in the ratio 1:3
  (h2 : ∥P - O∥ = r)                                -- P is on the larger circle, OP=r
  (h3 : ∥X - O∥ = (1/4) * r) :                      -- OX is 1/4 of OP
  (π * (∥X - O∥^2)) / (π * (∥P - O∥^2)) = 1 / 16 := /- Proof -/ sorry

end ratio_of_areas_l215_215732


namespace value_of_m_l215_215237

theorem value_of_m (m : ℝ) (h_linear : ∃ a b : ℝ, (m+2)*x^(|m|-1) - 1 = a * x + b) :
  (|m| - 1 = 1) ∧ (m + 2 ≠ 0) → m = 2 :=
by
  intros h_cond
  cases h_cond with h1 h2
  have h_abs : |m| = 2 := 
    by
      exact Eq.trans h1 (by norm_num)
  obtain (h_eq1 | h_eq2) := abs_eq 2
  { linarith }
  { contradiction }

end value_of_m_l215_215237


namespace two_digit_numbers_satisfying_R_n_eq_R_n_plus_1_l215_215884

open Nat

def R (n : ℕ) : ℕ :=
  (List.range' 2 9).map (λ k => n % (k + 2)).sum

theorem two_digit_numbers_satisfying_R_n_eq_R_n_plus_1 :
  (List.range' 10 90).countp (λ n => R n = R (n + 1)) = 2 :=
  sorry

end two_digit_numbers_satisfying_R_n_eq_R_n_plus_1_l215_215884


namespace solve_for_x_l215_215003

theorem solve_for_x (x : ℝ) (h : 2 * x + 10 = (1 / 2) * (5 * x + 30)) : x = -10 :=
sorry

end solve_for_x_l215_215003


namespace equation_many_solutions_l215_215889

noncomputable def f (x : ℝ) : ℝ :=
  (-x^3 - 13 * x + 6) / (x^2)

theorem equation_many_solutions (a : ℝ) :
  (a ∈ Set.Icc (-8) (-20/3) ∨ a ∈ Set.Ici (61/8))
  → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^3 + a * x₁^2 + 13 * x₁ - 6 = 0) ∧ (x₂^3 + a * x₂^2 + 13 * x₂ - 6 = 0) :=
begin
  sorry
end

end equation_many_solutions_l215_215889


namespace min_value_expression_l215_215653

theorem min_value_expression (θ φ : ℝ) :
  ∃ (θ φ : ℝ), (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
sorry

end min_value_expression_l215_215653


namespace scientific_notation_conversion_l215_215702

theorem scientific_notation_conversion : (0.000000045 : ℝ) = 4.5 * 10^(-8) := 
by 
  sorry

end scientific_notation_conversion_l215_215702


namespace sum_of_first_n_terms_l215_215599

noncomputable def geometric_sequence_sum (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 2) - (n^2 + n - 4)

theorem sum_of_first_n_terms
  (hn1 : ∀ n, (a n : ℕ) → {a n} : ℕ → (a n / n + 1))
  (hn2 : a 1 = 1) :
  S n = (n - 1) * 2^(n + 2) - (n^2 + n - 4) / 2 :=
sorry

end sum_of_first_n_terms_l215_215599


namespace find_k_l215_215321

variables (e₁ e₂ : ℝ^3)
variables (AB CD CB BD : ℝ^3)
variables (k : ℝ)

-- Conditions
def condition1 : Prop := ∀ a b : ℝ, a • e₁ + b • e₂ = 0 → a = 0 ∧ b = 0
def vec_AB : AB = 2 • e₁ + k • e₂ := sorry
def vec_CD : CD = 2 • e₁ - e₂ := sorry
def vec_CB : CB = e₁ + 3 • e₂ := sorry
def collinear : ∃ k' : ℝ, AB = k' • BD := sorry
def vec_BD : BD = CD - CB := sorry

-- Theorem to prove
theorem find_k (cond1 : condition1)
               (h1 : vec_AB)
               (h2 : vec_CD)
               (h3 : vec_CB)
               (h4 : vec_BD)
               (h5 : collinear) :
  k = -8 := sorry

end find_k_l215_215321


namespace clock_angle_at_3_15_l215_215026

def hour_hand_position (h m : ℕ) : ℝ := (h % 12) * 30 + (m / 60) * 30
def minute_hand_position (m : ℕ) : ℝ := (m % 60) * 6
def acute_angle (a b : ℝ) : ℝ := if abs (a - b) <= 180 then abs (a - b) else 360 - abs (a - b)

theorem clock_angle_at_3_15 : acute_angle (hour_hand_position 3 15) (minute_hand_position 15) = 7.5 :=
by
  sorry

end clock_angle_at_3_15_l215_215026


namespace unique_ordered_triple_lcm_l215_215320

theorem unique_ordered_triple_lcm:
  ∃! (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c), 
    Nat.lcm a b = 2100 ∧ Nat.lcm b c = 3150 ∧ Nat.lcm c a = 4200 :=
by
  sorry

end unique_ordered_triple_lcm_l215_215320


namespace smallest_N_proof_l215_215281

theorem smallest_N_proof (N c1 c2 c3 c4 : ℕ)
  (h1 : N + c1 = 4 * c3 - 2)
  (h2 : N + c2 = 4 * c1 - 3)
  (h3 : 2 * N + c3 = 4 * c4 - 1)
  (h4 : 3 * N + c4 = 4 * c2) :
  N = 12 :=
sorry

end smallest_N_proof_l215_215281


namespace inequality_problem_l215_215199

theorem inequality_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := 
sorry

end inequality_problem_l215_215199


namespace distance_from_midpoint_to_y_axis_l215_215547
open Real

-- Definitions of points A and B on the parabola
def is_on_parabola (A : ℝ × ℝ) : Prop :=
  A.snd ^ 2 = 4 * A.fst

def vec_eq (AF FB : ℝ) : Prop :=
  AF = 3 * FB

-- Midpoint calculation
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)

-- The proof problem statement
theorem distance_from_midpoint_to_y_axis {A B : ℝ × ℝ}
  (hA : is_on_parabola A)
  (hB : is_on_parabola B)
  (hvec : vec_eq (A.fst + 1) ((3 * (B.fst + 1)))) :
  abs ((midpoint A B).fst - 0) = 5 / 3 :=
sorry

end distance_from_midpoint_to_y_axis_l215_215547


namespace geometric_sequence_log_sum_l215_215974

theorem geometric_sequence_log_sum (b : ℕ → ℝ) (h : ∀ n, 0 < b n) (h_geometric : ∀ n, b (n) * b (15 - n) = 3) :
  ∑ i in finset.range 14, real.log_base 3 (b (i + 1)) = 7 :=
sorry

end geometric_sequence_log_sum_l215_215974


namespace grid_columns_l215_215794

theorem grid_columns (n : ℕ) : 
  (∀ grid : ℕ → ℕ → (ℕ × ℕ), 
    (∀ (r c: ℕ), grid r c = (10, n)) →
    (∀ domino_positions : ℕ, domino_positions = 2004) →
    (∃ n, 9 * n + 10 * (n - 1) = 2004) →
    n = 106) := 
by {
  intros,
  sorry
}

end grid_columns_l215_215794


namespace quadrilateral_angle_identity_l215_215428

theorem quadrilateral_angle_identity
  (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB AC AD BC BD CD : ℝ)
  (angle_DAB angle_BCD : ℝ)
  (h_angle : angle_DAB + angle_BCD = 90) :
  AB^2 * CD^2 + AD^2 * BC^2 = AC^2 * BD^2 :=
by
  sorry

end quadrilateral_angle_identity_l215_215428


namespace probability_sum_8_twice_l215_215409

-- Define a structure for the scenario: a 7-sided die.
structure Die7 :=
(sides : Fin 7)

-- Define a function to check if the sum of two dice equals 8.
def is_sum_8 (d1 d2 : Die7) : Prop :=
  (d1.sides.val + 1) + (d2.sides.val + 1) = 8

-- Define the probability of the event given the conditions.
def probability_event_twice (successes total_outcomes : ℕ) : ℚ :=
  (successes / total_outcomes) * (successes / total_outcomes)

-- The total number of outcomes when rolling two 7-sided dice.
def total_outcomes : ℕ := 7 * 7

-- The number of successful outcomes that yield a sum of 8 with two rolls.
def successful_outcomes : ℕ := 7

-- Main theorem statement to be proved.
theorem probability_sum_8_twice :
  probability_event_twice successful_outcomes total_outcomes = 1 / 49 :=
by
  -- Sorry to indicate that the proof is omitted.
  sorry

end probability_sum_8_twice_l215_215409


namespace lamps_cycle_even_lamps_odd_lamps_l215_215729

/-- The mathematical problem of toggling lamps in a circle. -/
theorem lamps_cycle (n : ℕ) (h : n > 1) : ∃ M : ℕ, ∀ (s : ℕ → bool) (L : ℕ → bool),
  (∀ (i : ℕ), L i = tt) ∧  -- Initial state: all lamps are on
  (∀ (i : ℕ) (t : ℕ), s t = tt → L (i + 1) = bnot (L i)) →  -- Step function
  (∀ (t : ℕ), t > M → ∀ (i : ℕ), L i = tt) := 
sorry

/-- For even number of lamps n = 2k, it takes n^2 - 1 steps to return all lamps to on state. -/
theorem even_lamps (k : ℕ) : ∃ M : ℕ, M = (2 * k) ^ 2 - 1 :=
sorry

/-- For odd number of lamps n = 2k + 1, it takes n^2 - n + 1 steps to return all lamps to on state. -/
theorem odd_lamps (k : ℕ) : ∃ M : ℕ, M = (2 * k + 1) ^ 2 - (2 * k + 1) + 1 :=
sorry

end lamps_cycle_even_lamps_odd_lamps_l215_215729


namespace hyperbola_property_l215_215225

noncomputable def F1 := (-sqrt 6, 0)
noncomputable def F2 := (sqrt 6, 0)
noncomputable def hyperbola (p : ℝ × ℝ) := (p.1^2 - p.2^2 = 3)
noncomputable def on_hyperbola (P : ℝ × ℝ) := hyperbola P
noncomputable def angle_F1PF2 := 120 * (real.pi / 180)

theorem hyperbola_property (P : ℝ × ℝ) (hP : on_hyperbola P) (h_angle : angle P F1 F2 = angle_F1PF2) :
  dist P F1 ^ 2 + dist P F2 ^ 2 = 20 :=
sorry

end hyperbola_property_l215_215225


namespace eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l215_215948

theorem eight_digit_numbers_with_012 :
  let total_sequences := 3^8 
  let invalid_sequences := 3^7 
  total_sequences - invalid_sequences = 4374 :=
by sorry

theorem eight_digit_numbers_with_00012222 :
  let total_sequences := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 4)
  let invalid_sequences := Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 4)
  total_sequences - invalid_sequences = 175 :=
by sorry

theorem eight_digit_numbers_starting_with_1_0002222 :
  let number_starting_with_1 := Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 4)
  number_starting_with_1 = 35 :=
by sorry

end eight_digit_numbers_with_012_eight_digit_numbers_with_00012222_eight_digit_numbers_starting_with_1_0002222_l215_215948


namespace fill_sacks_times_l215_215137

-- Define the capacities of the sacks
def father_sack_capacity : ℕ := 20
def senior_ranger_sack_capacity : ℕ := 30
def volunteer_sack_capacity : ℕ := 25
def number_of_volunteers : ℕ := 2

-- Total wood gathered
def total_wood_gathered : ℕ := 200

-- Statement of the proof problem
theorem fill_sacks_times : (total_wood_gathered / (father_sack_capacity + senior_ranger_sack_capacity + (number_of_volunteers * volunteer_sack_capacity))) = 2 := by
  sorry

end fill_sacks_times_l215_215137


namespace tangent_line_y_intercept_l215_215783

-- Define the circles
def circle1_center : (ℝ × ℝ) := (3, 0)
def circle1_radius : ℝ := 3

def circle2_center : (ℝ × ℝ) := (7, 0)
def circle2_radius : ℝ := 2

-- Define the problem as a theorem
theorem tangent_line_y_intercept :
  ∃ (line : ℝ → ℝ), -- representing the tangent line as a function of x (going through first quadrant)
    (∀ x : ℝ, 
      -- line is tangent to circle1
      ((x - fst circle1_center)^2 + (line x - snd circle1_center)^2 = circle1_radius^2) ∨
      -- line is tangent to circle2
      ((x - fst circle2_center)^2 + (line x - snd circle2_center)^2 = circle2_radius^2)
    ) ∧ -- and the y-intercept of this line is 8
    line 0 = 8 :=
sorry

end tangent_line_y_intercept_l215_215783


namespace geom_series_sum_l215_215136

theorem geom_series_sum : 
  let a := 1 : ℝ
  let r := 1 / 4 : ℝ
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geom_series_sum_l215_215136


namespace problem_proof_l215_215886

noncomputable def f : ℕ → ℝ
| 0 => 2019
| (n + 1) => f n + (20 / ((n + 1) * (n + 2))) - 1

theorem problem_proof : 
  (for all x : ℕ, f (x + 1) + 1 = f x + (20 / (real.of_nat (x + 1) * real.of_nat (x + 2)))) ∧ 
  (f 0 = 2019) → 
  (2019 / f 2019 = 101) :=
sorry

end problem_proof_l215_215886


namespace john_age_is_24_l215_215307

noncomputable def john_age_condition (j d b : ℕ) : Prop :=
  j = d - 28 ∧
  j + d = 76 ∧
  j + 5 = 2 * (b + 5)

theorem john_age_is_24 (d b : ℕ) : ∃ j, john_age_condition j d b ∧ j = 24 :=
by
  use 24
  unfold john_age_condition
  sorry

end john_age_is_24_l215_215307


namespace sqrt_37_between_6_and_7_l215_215163

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := 
by 
  have h₁ : Real.sqrt 36 = 6 := by sorry
  have h₂ : Real.sqrt 49 = 7 := by sorry
  sorry

end sqrt_37_between_6_and_7_l215_215163


namespace value_x_plus_y_l215_215019

noncomputable def vector_a : ℝ × ℝ × ℝ := (2, 4, sorry)
noncomputable def vector_b : ℝ × ℝ × ℝ := (2, sorry, 2)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

axiom a_magnitude : magnitude vector_a = 6
axiom a_dot_product : dot_product vector_a vector_b = 0

theorem value_x_plus_y : ∃ x y : ℝ, x + y = -3 ∨ x + y = 1 :=
sorry

end value_x_plus_y_l215_215019


namespace first_employee_hourly_wage_l215_215491

theorem first_employee_hourly_wage (x : ℝ) 
  (h1 : ∀ x, ∃ total_cost1 weekly_savings . 
    (total_cost1 = 40 * x) ∧ 
    (weekly_savings = 40 * 16 - total_cost1) ∧ 
    (weekly_savings = 160)) : 
  x = 12 :=
by sorry

end first_employee_hourly_wage_l215_215491


namespace ratio_of_coefficients_l215_215600

theorem ratio_of_coefficients (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (H1 : 8 * x - 6 * y = c) (H2 : 12 * y - 18 * x = d) :
  c / d = -4 / 9 := 
by {
  sorry
}

end ratio_of_coefficients_l215_215600


namespace cost_of_airplane_l215_215086

theorem cost_of_airplane (amount : ℝ) (change : ℝ) (h_amount : amount = 5) (h_change : change = 0.72) : 
  amount - change = 4.28 := 
by
  sorry

end cost_of_airplane_l215_215086


namespace coprime_b_sigma_b_l215_215312

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (fun d => n % d = 0).sum

theorem coprime_b_sigma_b (r b : ℕ) (hr : 0 < r) (hb_odd : is_odd b)
  (hN : ∃ N : ℕ, N = 2^r * b ∧ sum_of_divisors N = 2 * N - 1) : Nat.coprime b (sum_of_divisors b) :=
by
  sorry

end coprime_b_sigma_b_l215_215312


namespace two_point_question_count_l215_215758

/-- Define the number of questions and points on the test,
    and prove that the number of 2-point questions is 30. -/
theorem two_point_question_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 := by
  sorry

end two_point_question_count_l215_215758


namespace part1_monotonic_intervals_part1_tangent_line_part2_range_of_a_l215_215573

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := exp x - a * x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2 * f x a - x^2 - a^2

theorem part1_monotonic_intervals (x : ℝ) : 
  (f x 1).deriv < 0 ↔ x < 0 ∧ (f x 1).deriv > 0 ↔ x > 0 := 
sorry

theorem part1_tangent_line (x : ℝ) : 
  let tangent_eq (x : ℝ) := (exp 1 - 1) * x
  tangent_eq x = (exp (1 : ℝ) - 1) * x := 
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x ≥ 0, g x a ≥ 0) ↔ a ∈ set.Icc (-real.sqrt 2) (2 - real.log 2) :=
sorry

end part1_monotonic_intervals_part1_tangent_line_part2_range_of_a_l215_215573


namespace sqrt_3_irrational_l215_215416

theorem sqrt_3_irrational
  (h0_3 : ∃ (q : ℚ), (q : ℝ) = 0.3)
  (h3_14 : ∃ (q : ℚ), (q : ℝ) = 3.14)
  (h_sqrt9 : ∃ (q : ℚ), (q : ℝ) = real.sqrt 9) :
  ¬ ∃ (q : ℚ), (q : ℝ) = real.sqrt 3 :=
sorry

end sqrt_3_irrational_l215_215416


namespace min_balls_draw_l215_215012

def box1_red := 40
def box1_green := 30
def box1_yellow := 25
def box1_blue := 15

def box2_red := 35
def box2_green := 25
def box2_yellow := 20

def min_balls_to_draw_to_get_20_balls_of_single_color (totalRed totalGreen totalYellow totalBlue : ℕ) : ℕ :=
  let maxNoColor :=
    (min totalRed 19) + (min totalGreen 19) + (min totalYellow 19) + (min totalBlue 15)
  maxNoColor + 1

theorem  min_balls_draw {r1 r2 g1 g2 y1 y2 b1 : ℕ} :
  r1 = box1_red -> g1 = box1_green -> y1 = box1_yellow -> b1 = box1_blue ->
  r2 = box2_red -> g2 = box2_green -> y2 = box2_yellow ->
  min_balls_to_draw_to_get_20_balls_of_single_color (r1 + r2) (g1 + g2) (y1 + y2) b1 = 73 :=
by
  intros
  unfold min_balls_to_draw_to_get_20_balls_of_single_color
  sorry

end min_balls_draw_l215_215012


namespace linear_function_condition_l215_215552

def is_linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f(x) = m * x + b

theorem linear_function_condition (f : ℝ → ℝ) (h : is_linear_function f) :
  (∀ x : ℝ, (x = 3 → 2 * f x - 10 = f (x - 2))) →
  (∃ m b : ℝ, (5 * m + b = 10)) :=
by
  sorry

end linear_function_condition_l215_215552


namespace log_ratio_pi_l215_215782

theorem log_ratio_pi (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  r = log10 (x ^ 3) ∧ C = log10 (y ^ 6) → log x y = π := 
by 
  sorry

end log_ratio_pi_l215_215782


namespace num_new_terms_in_sequence_l215_215406

theorem num_new_terms_in_sequence (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end num_new_terms_in_sequence_l215_215406


namespace find_lambda_value_l215_215213

variables {V : Type*} [inner_product_space ℚ V]

noncomputable def lambda_value 
    (a b : V) (hab : a ≠ 0 ∧ b ≠ 0) 
    (h1 : ∥a + b∥ = ∥b∥) 
    (h2 : a ⊥ (a + λ • b)) 
    : ℚ :=
  2

theorem find_lambda_value 
    (a b : V) (hab : a ≠ 0 ∧ b ≠ 0) 
    (h1 : ∥a + b∥ = ∥b∥) 
    (h2 : a ⊥ (a + (2 : ℚ) • b)) 
    : lambda_value a b hab h1 h2 = 2 := 
by 
  sorry

end find_lambda_value_l215_215213


namespace min_pos_period_f_max_min_f_on_interval_cos_2alpha_given_f_alpha_l215_215569

noncomputable def f (x : ℝ) := 2 * sqrt 3 * (sin x) * (cos x) + 2 * (cos x) ^ 2 - 1

theorem min_pos_period_f : ∀ x : ℝ, f(x) = f(x + π) := sorry

theorem max_min_f_on_interval :
  let interval := set.Icc (0 : ℝ) (π / 2)
  ∃ x₁ x₂ ∈ interval, 
    (∀ x ∈ interval, f x ≤ f x₁) ∧ f x₁ = 2 ∧
    (∀ x ∈ interval, f x₂ ≤ f x) ∧ f x₂ = -1 := sorry

theorem cos_2alpha_given_f_alpha :
  ∀ α : ℝ, α ∈ set.Icc (π / 4) (π / 2) → f α = 6 / 5 → cos (2 * α) = (3 - 4 * sqrt 3) / 10 := sorry

end min_pos_period_f_max_min_f_on_interval_cos_2alpha_given_f_alpha_l215_215569


namespace BM_squared_eq_X_cot_B_div_2_l215_215450

noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem BM_squared_eq_X_cot_B_div_2 
  {a b c BM : ℝ} {A B C X : ℝ} {M : ℝ}
  (hM_on_AC : 0 < M ∧ M < AC)
  (h_radii_eq : insradii (A B M) = insradii (B M C))
  (hX : X = area a b c) :
  BM^2 = X * Real.cot (B / 2) := sorry

end BM_squared_eq_X_cot_B_div_2_l215_215450


namespace factorial_simplification_l215_215744

theorem factorial_simplification :
  (10.factorial * 6.factorial * 3.factorial) / (9.factorial * 7.factorial) = 60 / 7 :=
by
-- The proof details would go here
sorry

end factorial_simplification_l215_215744


namespace Manny_lasagna_pieces_l215_215657

-- Define variables and conditions
variable (M : ℕ) -- Manny's desired number of pieces
variable (A : ℕ := 0) -- Aaron's pieces
variable (K : ℕ := 2 * M) -- Kai's pieces
variable (R : ℕ := M / 2) -- Raphael's pieces
variable (L : ℕ := 2 + R) -- Lisa's pieces

-- Prove that Manny wants 1 piece of lasagna
theorem Manny_lasagna_pieces (M : ℕ) (A : ℕ := 0) (K : ℕ := 2 * M) (R : ℕ := M / 2) (L : ℕ := 2 + R) 
  (h : M + A + K + R + L = 6) : M = 1 :=
by
  sorry

end Manny_lasagna_pieces_l215_215657


namespace number_of_balanced_colorings_l215_215282

-- Define the grid dimensions and the set of colors
def grid_width : ℕ := 8
def grid_height : ℕ := 6
def colors := { "red", "blue", "yellow", "green" }

-- Define what it means for a 2x2 subgrid to be "balanced"
def balanced_2x2 (subgrid : Fin 2 × Fin 2 → String): Prop :=
  (∀ color ∈ colors, ∃ (i j : Fin 2), subgrid (i, j) = color)

-- Define the "balanced coloring" condition for an entire grid
def balanced_coloring (grid : Fin 8 × Fin 6 → String): Prop :=
  ∀ i j, i + 1 < 8 → j + 1 < 6 → balanced_2x2 (λ (a : Fin 2 × Fin 2), grid (i + a.1, j + a.2))

-- The main theorem stating the number of balanced colorings
theorem number_of_balanced_colorings :
  ∃ (count : ℕ), count = 1896 ∧
    ∃ (f : Fin 8 × Fin 6 → String), balanced_coloring f := by
  sorry

end number_of_balanced_colorings_l215_215282


namespace intersection_P_Q_is_2_3_l215_215243

def P : Set ℝ := { x | 1 ≤ Real.logBase 2 x ∧ Real.logBase 2 x < 2 }
def Q : Set ℕ := { 1, 2, 3 }

theorem intersection_P_Q_is_2_3 :
  P ∩ (Q.map (λ x, (x : ℝ))) = {2, 3} :=
by
  sorry

end intersection_P_Q_is_2_3_l215_215243


namespace vertices_divisible_by_three_l215_215292

namespace PolygonDivisibility

theorem vertices_divisible_by_three (v : Fin 2018 → ℤ) 
  (h_initial : (Finset.univ.sum v) = 1) 
  (h_move : ∀ i : Fin 2018, ∃ j : Fin 2018, abs (v i - v j) = 1) :
  ¬ ∃ (k : Fin 2018 → ℤ), (∀ n : Fin 2018, k n % 3 = 0) :=
by {
  sorry
}

end PolygonDivisibility

end vertices_divisible_by_three_l215_215292


namespace Jeanine_has_more_pencils_than_Clare_l215_215304

def number_pencils_Jeanine_bought : Nat := 18
def number_pencils_Clare_bought := number_pencils_Jeanine_bought / 2
def number_pencils_given_to_Abby := number_pencils_Jeanine_bought / 3
def number_pencils_Jeanine_now := number_pencils_Jeanine_bought - number_pencils_given_to_Abby 

theorem Jeanine_has_more_pencils_than_Clare :
  number_pencils_Jeanine_now - number_pencils_Clare_bought = 3 := by
  sorry

end Jeanine_has_more_pencils_than_Clare_l215_215304


namespace line_intersects_circle_l215_215775

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, y = k * (x - 1) ∧ x^2 + y^2 = 1 :=
by
  sorry

end line_intersects_circle_l215_215775


namespace simple_interest_principal_is_correct_l215_215001

noncomputable def compound_interest (P r t : ℝ) : ℝ :=
  P * (1 + r) ^ t

noncomputable def simple_interest (P r t : ℝ) : ℝ :=
  P * r * t / 100

noncomputable def find_principal (si : ℝ) (r t : ℝ) : ℝ :=
  si * 100 / (r * t)

theorem simple_interest_principal_is_correct :
  let CI := compound_interest 4000 0.10 2 - 4000 in
  let SI := CI / 2 in
  find_principal SI 8 2 = 2625 :=
by
  let CI := compound_interest 4000 0.10 2 - 4000
  let SI := CI / 2
  let P := find_principal SI 8 2
  have h : CI = 4000 * (1 + 0.10) ^ 2 - 4000 := rfl
  rw [h] at CI
  have h1 : 1 + 0.10 = 1.1 := rfl
  rw [h1] at CI
  have h2 : 1.1 ^ 2 = 1.21 := rfl
  rw [h2] at CI
  have h3 : 4000 * 1.21 = 4840 := rfl
  rw [h3] at CI
  have h4 : CI = 4840 - 4000 := rfl
  rw [h4] at CI
  have h5 : CI = 840 := rfl
  rw [h5] at SI
  have h6 : SI = 420 := rfl
  rw [h6] at P
  have h7 : P = find_principal 420 8 2 := rfl
  rw [h7]
  have h8 : find_principal 420 8 2 = 2625 := rfl
  exact h8

end simple_interest_principal_is_correct_l215_215001


namespace unique_property_of_rectangles_l215_215093

-- Definitions of the conditions
structure Quadrilateral (Q : Type*) :=
(sum_of_interior_angles : ∀ (q : Q), angle (interior q) = 360)

structure Rectangle (R : Type*) extends Quadrilateral R :=
(diagonals_bisect_each_other : ∀ (r : R), bisects (diagonals r))
(diagonals_equal_length : ∀ (r : R), equal_length (diagonals r))
(diagonals_not_necessarily_perpendicular : ∀ (r : R), not (necessarily_perpendicular (diagonals r)))

structure Rhombus (H : Type*) extends Quadrilateral H :=
(diagonals_bisect_each_other : ∀ (h : H), bisects (diagonals h))
(diagonals_perpendicular : ∀ (h : H), perpendicular (diagonals h))
(diagonals_not_necessarily_equal_length : ∀ (h : H), not (necessarily_equal_length (diagonals h)))

-- The proof statement
theorem unique_property_of_rectangles (R : Type*) [Rectangle R] (H : Type*) [Rhombus H] :
  ∀ (r : R), ∃ (h : H), equal_length (diagonals r) ∧ not (equal_length (diagonals h)) :=
sorry

end unique_property_of_rectangles_l215_215093


namespace train_cross_bridge_time_l215_215252

theorem train_cross_bridge_time
  (length_train : ℕ) (speed_train_kmph : ℕ) (length_bridge : ℕ) 
  (km_to_m : ℕ) (hour_to_s : ℕ)
  (h1 : length_train = 165) 
  (h2 : speed_train_kmph = 54) 
  (h3 : length_bridge = 720) 
  (h4 : km_to_m = 1000) 
  (h5 : hour_to_s = 3600) 
  : (length_train + length_bridge) / ((speed_train_kmph * km_to_m) / hour_to_s) = 59 := 
sorry

end train_cross_bridge_time_l215_215252


namespace geometric_series_sum_eq_4_div_3_l215_215128

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l215_215128


namespace total_cost_l215_215308

def num_professionals := 2
def hours_per_professional_per_day := 6
def days_worked := 7
def hourly_rate := 15

theorem total_cost : 
  (num_professionals * hours_per_professional_per_day * days_worked * hourly_rate) = 1260 := by
  sorry

end total_cost_l215_215308


namespace prove_beta_value_l215_215893

theorem prove_beta_value (α β : ℝ) (h1 : cos α = 3 / 5) (h2 : cos (α - β) = 7 * sqrt 2 / 10) (h3 : 0 < β ∧ β < α ∧ α < π / 2) : 
  β = π / 4 :=
sorry

end prove_beta_value_l215_215893


namespace eigenvector_of_g_h_monotonic_intervals_cos_x1_x2_l215_215543

noncomputable def f (a b x : ℝ) : ℝ := a * sin x + b * cos x

def eigenvector_of_f (a b : ℝ) : ℝ × ℝ := (a, b)

noncomputable def g (x : ℝ) : ℝ := sin (x + π / 3) + cos (x - π / 6)

theorem eigenvector_of_g :
  eigenvector_of_f 1 (sqrt 3) = eigenvector_of_f (fst (eigenvector_of_f 1 (sqrt 3))) (snd (eigenvector_of_f 1 (sqrt 3))) :=
sorry

noncomputable def p (x : ℝ) : ℝ := sqrt 3 * sin x - cos x

noncomputable def q (x : ℝ) : ℝ := sin x + sqrt 3 * cos x

noncomputable def h (x : ℝ) : ℝ := p x * q x

theorem h_monotonic_intervals (k : ℤ) :
  ∀ x, -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π →
  ∀ u v, (u < v ∧ h u ≤ h v) ∨ (u > v ∧ h u ≥ h v) :=
sorry

theorem cos_x1_x2 (x1 x2 : ℝ) (hx : 0 < x1 ∧ x1 < π ∧ 0 < x2 ∧ x2 < π)
  (h_eq : h x1 = 2 / 3 ∧ h x2 = 2 / 3) : 
  cos (x1 - x2) = 1 / 3 :=
sorry

end eigenvector_of_g_h_monotonic_intervals_cos_x1_x2_l215_215543


namespace solution_to_cubed_root_eq_l215_215520

theorem solution_to_cubed_root_eq (x : ℝ) : (cbrt (5 * x - 2) = 2) ↔ (x = 2) :=
by
  sorry

end solution_to_cubed_root_eq_l215_215520


namespace percent_increase_equilateral_triangles_l215_215468

theorem percent_increase_equilateral_triangles :
  let s₁ := 3
  let s₂ := 2 * s₁
  let s₃ := 2 * s₂
  let s₄ := 2 * s₃
  let P₁ := 3 * s₁
  let P₄ := 3 * s₄
  (P₄ - P₁) / P₁ * 100 = 700 :=
by
  sorry

end percent_increase_equilateral_triangles_l215_215468


namespace line_plane_relationship_l215_215269

/--
If a line is perpendicular to countless lines within a plane, 
then the positional relationship between the line and the plane 
is either parallel, perpendicular, intersecting, or within the plane.
-/
theorem line_plane_relationship (l : Line) (P : Plane) 
  (h : ∀ p ∈ P, ∃ l' ∈ P, l ⟂ l') :
  l ∥ P ∨ l ⟂ P ∨ (∃ p ∈ P, l ∩ P = p) ∨ (l ∈ P) :=
by
  sorry

end line_plane_relationship_l215_215269


namespace proof_problem_l215_215931

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := Real.exp x * Real.log (1 + x)
def g (x : ℝ) : ℝ := (deriv f) x

-- The statement for the problem to be proved
theorem proof_problem 
  (tangent_line_at_origin : ∀ x, f(0) = 0 → deriv f 0 = 1 → (f x = x))
  (monotonicity_of_g : ∀ x, 0 ≤ x → g' x > 0 → Monotone g)
  (inequality : ∀ s t : ℝ, 0 < s → 0 < t → f (s + t) > f s + f t) 
: Prop := sorry

end proof_problem_l215_215931


namespace volumes_rotation_equal_distance_for_given_ratio_max_iq_position_l215_215089

open Real

noncomputable def volumes_equal (r : ℝ) (x : ℝ) : Prop :=
  let AH := (2 * r * x^2) / (r^2 + x^2)
  let HB := (2 * r^2 * x) / (r^2 + x^2)
  let v1 := (2 * π * r * x^4) / (3 * (r^2 + x^2))
  let v2 := v1 -- Derived from the solution step, showing v1 equals v2.
  v1 = v2

noncomputable def distance_ratio (r : ℝ) (n : ℝ) : ℝ :=
  r * sqrt(n - 1 + sqrt(n^2 + 4 * n))

noncomputable def iq_maximized_at (r : ℝ) : ℝ :=
  r * sqrt(3)

theorem volumes_rotation_equal (r : ℝ) (x : ℝ) : volumes_equal r x := by {
  sorry
}

theorem distance_for_given_ratio (r : ℝ) (n : ℝ) : 
  distance_ratio r n := by {
  sorry
}

theorem max_iq_position (r : ℝ) : 
  iq_maximized_at r := by {
  sorry
}

end volumes_rotation_equal_distance_for_given_ratio_max_iq_position_l215_215089


namespace number_of_convex_quadrilaterals_l215_215587

def is_convex_quadrilateral (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≤ b + c + d ∧ b ≤ a + c + d ∧ c ≤ a + b + d ∧ d ≤ a + b + c

def has_parallel_sides (a b c d : ℕ) : Prop :=
  (a = c ∧ b = d) ∨ (a = b ∧ c = d)

theorem number_of_convex_quadrilaterals : 
  (count_quadrilaterals (36 : ℕ) (1498 : ℕ)) :=
begin
  sorry
end

end number_of_convex_quadrilaterals_l215_215587


namespace trig_identity_proof_l215_215489

theorem trig_identity_proof :
  Real.sin (30 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) + 
  Real.sin (60 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) =
  Real.sqrt 2 / 2 := 
by
  sorry

end trig_identity_proof_l215_215489


namespace non_negative_combined_quadratic_l215_215390

theorem non_negative_combined_quadratic (a b c A B C : ℝ) (h1 : a ≥ 0) (h2 : b^2 ≤ a * c) (h3 : A ≥ 0) (h4 : B^2 ≤ A * C) :
  ∀ x : ℝ, a * A * x^2 + 2 * b * B * x + c * C ≥ 0 :=
by
  sorry

end non_negative_combined_quadratic_l215_215390


namespace variance_transform_l215_215923

variable {α : Type*} [LinearOrder α] [DivisionRing α]

-- Given the variance function for a set of data
def variance (X : list ℝ) : ℝ := sorry

-- Given transformation function on the data
def transform (X : list ℝ) : list ℝ := X.map (λ x, 2 * x + 1)

theorem variance_transform (X : list ℝ) (h : variance X = 4) : variance (transform X) = 16 := sorry

end variance_transform_l215_215923


namespace sum_of_two_digit_integers_with_squares_ending_in_06_l215_215411

theorem sum_of_two_digit_integers_with_squares_ending_in_06 :
  let nums := [n | n ∈ ([10 * a + b | a ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9], b ∈ [4, 6]], 
                             let sqr := n * n in (sqr % 100) = 6)]
  let unique_nums := nums.erase_dup
  (unique_nums.sum = 176) :=
by {
  sorry
}

end sum_of_two_digit_integers_with_squares_ending_in_06_l215_215411


namespace center_of_circumscribed_sphere_ratio_l215_215696

theorem center_of_circumscribed_sphere_ratio {α : ℝ} (hα : α < π / 4) :
  ∃ r : ℝ, r = cos (2 * α) :=
sorry

end center_of_circumscribed_sphere_ratio_l215_215696


namespace ellipse_and_fixed_points_l215_215578

noncomputable def parabola_focus_and_directrix (a b c : ℝ) : Prop :=
  a = 2 ∧ b^2 = a^2 - c^2 ∧ c = sqrt 3

theorem ellipse_and_fixed_points :
  (∃ a b : ℝ, parabola_focus_and_directrix a b (sqrt 3) ∧ a > b ∧
   (∃ x y : ℝ, (x,y) ≠ (0,0) ∧ (x^2)/4 + y^2 = 1 
    ∧ (∀ x y : ℝ, x ≠ 0 ∨ y ≠ 0 →
      ∃ m n : ℝ, 
        m = sqrt 2 ∧ n = -sqrt 2 ∧
        ∃ t : ℝ, t = -1/4 ∧ 
          parabola_focus_and_directrix a b (sqrt 3)))) :=
begin
  sorry
end

end ellipse_and_fixed_points_l215_215578


namespace waiter_earned_in_tips_l215_215116

theorem waiter_earned_in_tips (total_customers : ℕ) (no_tip : ℕ) (tip_amount : ℕ) 
  (h1 : total_customers = 10) (h2 : no_tip = 5) (h3 : tip_amount = 3) : 
  ((total_customers - no_tip) * tip_amount) = 15 :=
by
  -- Using the conditions
  rw [h1, h2, h3]
  -- Simplifying the expression
  calc (10 - 5) * 3 = 5 * 3 := by rfl
    ... = 15 := by rfl

end waiter_earned_in_tips_l215_215116


namespace geometric_series_sum_l215_215132

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l215_215132


namespace find_xy_sum_l215_215882

theorem find_xy_sum :
  ∃ (x y : ℝ), (15 * x = x + 196) ∧ (50 * y = y + 842) ∧ (x + y = 31.2) :=
by
  -- Definitions of x and y
  let x := 14
  let y := 17.2

  -- Verification of conditions
  have hx : 15 * x = x + 196 := by
    calc 15 * 14 = 14 + 196 := by norm_num

  have hy : 50 * y = y + 842 := by
    calc 50 * 17.2 = 17.2 + 842 := by norm_num

  -- Sum of x and y
  have hsum : x + y = 31.2 := by
    calc 14 + 17.2 = 31.2 := by norm_num

  -- Conclusion
  exact ⟨x, y, hx, hy, hsum⟩

end find_xy_sum_l215_215882


namespace isosceles_triangle_perimeter_l215_215911

def is_isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 10) :
∃ c : ℝ, is_isosceles_triangle a b c ∧ perimeter a b c = 25 :=
by {
  sorry
}

end isosceles_triangle_perimeter_l215_215911


namespace mode_and_median_correct_l215_215607

def scores : List ℕ := [97, 88, 85, 93, 85]

def mode_and_median (l : List ℕ) : ℕ × ℕ :=
  let sorted_l := l.qsort (≤)
  let mode := sorted_l.foldl (λ (r : ℕ × ℕ) x =>
    if x = r.2 then (r.1 + 1, x) else
    if x = r.2 + 1 then r else (1, x)) (0, 0)
  let median := sorted_l.get! (sorted_l.length / 2)
  (mode.2, median)

theorem mode_and_median_correct : mode_and_median scores = (85, 88) := by
  sorry

end mode_and_median_correct_l215_215607


namespace geom_series_sum_l215_215133

theorem geom_series_sum : 
  let a := 1 : ℝ
  let r := 1 / 4 : ℝ
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geom_series_sum_l215_215133


namespace probability_at_least_6_heads_in_8_flips_l215_215791

/-- Let n be the number of flips, and k be the number of consecutive heads required. 
  The probability of getting at least k consecutive heads in n flips of a fair coin is computed. 
  For n = 8 and k = 6, the probability is 3/128. -/
theorem probability_at_least_6_heads_in_8_flips : 
  let n := 8; let k := 6;
  let total_outcomes := (2 ^ n) in
  let successful_outcomes := 6 in
  successful_outcomes / total_outcomes = 3 / 128 :=
by
  sorry

end probability_at_least_6_heads_in_8_flips_l215_215791


namespace jo_thinking_number_l215_215997

theorem jo_thinking_number 
  (n : ℕ) 
  (h1 : n < 100) 
  (h2 : n % 8 = 7) 
  (h3 : n % 7 = 4) 
  : n = 95 :=
sorry

end jo_thinking_number_l215_215997


namespace volume_of_inscribed_sphere_l215_215076

noncomputable theory

/-- Conditions on the right circular cone and inscribed sphere --/
def cone_conditions (diameter : ℝ) (vertex_angle_deg : ℝ) (sphere_touches_sides_and_rests_on_table : Prop) : Prop :=
diameter = 24 ∧ vertex_angle_deg = 90 ∧ sphere_touches_sides_and_rests_on_table

/-- Define the radius of the sphere based on cone properties --/
def sphere_radius (diameter : ℝ) : ℝ :=
12 -- as derived from the given solution steps

/-- Define the volume of a sphere given its radius --/
def sphere_volume (radius : ℝ) : ℝ :=
(4/3) * Real.pi * radius^3

/-- Main theorem: The volume of the sphere inscribed in the given cone conditions --/
theorem volume_of_inscribed_sphere (d : ℝ) (angle : ℝ) (rest_table : Prop)
  (h_cond : cone_conditions d angle rest_table) :
  sphere_volume (sphere_radius d) = 2304 * Real.pi :=
by
  simp [cone_conditions, sphere_radius, sphere_volume] at h_cond
  cases h_cond
  sorry

end volume_of_inscribed_sphere_l215_215076


namespace five_goats_choir_l215_215693

theorem five_goats_choir 
  (total_members : ℕ)
  (num_rows : ℕ)
  (total_members_eq : total_members = 51)
  (num_rows_eq : num_rows = 4) :
  ∃ row_people : ℕ, row_people ≥ 13 :=
by 
  sorry

end five_goats_choir_l215_215693


namespace cannot_be_calculated_using_square_difference_formula_l215_215823

variables (x y : ℝ)

def exprA := (-4*x + 3*y) * (4*x + 3*y)
def exprB := (4*x - 3*y) * (3*y - 4*x)
def exprC := (-4*x + 3*y) * (-4*x - 3*y)
def exprD := (4*x + 3*y) * (4*x - 3*y)

theorem cannot_be_calculated_using_square_difference_formula : 
  ∀ (a b: ℝ), exprA = a^2 - b^2 ∧ exprC = a^2 - b^2 ∧ exprD = a^2 - b^2 → exprB ≠ a^2 - b^2 := 
by
  sorry

end cannot_be_calculated_using_square_difference_formula_l215_215823


namespace machines_together_work_time_l215_215336

theorem machines_together_work_time :
  let rate_A := 1 / 4
  let rate_B := 1 / 12
  let rate_C := 1 / 6
  let rate_D := 1 / 8
  let rate_E := 1 / 18
  let total_rate := rate_A + rate_B + rate_C + rate_D + rate_E
  total_rate ≠ 0 → 
  let total_time := 1 / total_rate
  total_time = 72 / 49 :=
by
  sorry

end machines_together_work_time_l215_215336


namespace find_cost_price_of_clock_l215_215422

namespace ClockCost

variable (C : ℝ)

def cost_price_each_clock (n : ℝ) (gain1 : ℝ) (gain2 : ℝ) (uniform_gain : ℝ) (price_difference : ℝ) :=
  let selling_price1 := 40 * C * (1 + gain1)
  let selling_price2 := 50 * C * (1 + gain2)
  let uniform_selling_price := n * C * (1 + uniform_gain)
  selling_price1 + selling_price2 - uniform_selling_price = price_difference

theorem find_cost_price_of_clock (C : ℝ) (h : cost_price_each_clock C 90 0.10 0.20 0.15 40) : C = 80 :=
  sorry

end ClockCost

end find_cost_price_of_clock_l215_215422


namespace sqrt_15_decimal_part_sqrt_3a_b_c_square_root_l215_215584

theorem sqrt_15_decimal_part : ∀ (sqrt_15 : ℝ), 3 < sqrt_15 ∧ sqrt_15 < 4 → sqrt_15 - ⌊sqrt_15⌋ = sqrt_15 - 3 :=
by
  intro sqrt_15 h
  have int_part : ⌊sqrt_15⌋ = 3, from (by linarith : 3 ≤ sqrt_15 ∧ sqrt_15 < 4)
  rw int_part
  sorry

theorem sqrt_3a_b_c_square_root : ∀ (a b c : ℝ), (3 <= c ∧ c < 4) ∧ sqrt_15 = c →  (5a + 2 = 27) ∧ (3a + b - 1 = 16) → a = 5 ∧ b = 2 ∧ c = 3 → ∃ r : ℝ, sqrt (3a - b + c) = r ∧ r = 4 ∨ r = -4 :=
by
  intros a b c h1 h2 h3
  have h4 : a = 5, by linarith
  have h5 : b = 2, by linarith
  have h6 : c = 3, by linarith
  use 4
  intro r
  rw h4
  rw h5
  rw h6
  sorry

end sqrt_15_decimal_part_sqrt_3a_b_c_square_root_l215_215584


namespace solve_linear_system_l215_215245

theorem solve_linear_system :
  ∀ (x y : ℝ), (2 * x + 3 * y = 6) ∧ (3 * x + 2 * y = 4) → (x + y = 2) :=
by
  intro x y
  intro h
  cases h with h1 h2
  sorry

end solve_linear_system_l215_215245


namespace more_roses_than_orchids_l215_215730

-- Definitions
def roses_now : Nat := 12
def orchids_now : Nat := 2

-- Theorem statement
theorem more_roses_than_orchids : (roses_now - orchids_now) = 10 := by
  sorry

end more_roses_than_orchids_l215_215730


namespace solve_for_x_l215_215268

theorem solve_for_x (x y : ℝ) (h₁ : y = 1 / (4 * x + 2)) (h₂ : y = 1 / 2) : x = 0 :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l215_215268


namespace max_value_expr_l215_215961

theorem max_value_expr (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
sorry

end max_value_expr_l215_215961


namespace ellipse_geometry_l215_215984

noncomputable def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : set (ℝ × ℝ) :=
  {p | p.1 ^ 2 / a ^ 2 + p.2 ^ 2 / b ^ 2 = 1}

def ellipse_points (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : 
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let A1 := (-a, 0) in
  let A2 := (a, 0) in
  let c := (a ^ 2 - b ^ 2).sqrt in
  (A1, A2, (c, 0))

def directrix (a b : ℝ) (h : a > b) : ℝ :=
  let c := (a ^ 2 - b ^ 2).sqrt in
  a ^ 2 / c

def any_point_on_ellipse (a b : ℝ) (theta : ℝ) :
  (ℝ × ℝ) :=
  let x := a * Real.cos theta in
  let y := b * Real.sin theta in
  (x, y)

theorem ellipse_geometry (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) 
  (P : ℝ × ℝ) (hP : P ∈ ellipse a b ha hb h) :
  let (A1, A2, F2) := ellipse_points a b ha hb h in
  let l := directrix a b h in
  -- Conditions and intermediate definitions for M and N
  let M := infer_M a b P in -- intersection of A1P with l
  let N := infer_N a b P in -- intersection of A2P with l
  -- Prove the following:
  ((line_through F2 M).slope = -(line_through F2 N).slope)
  ∧ (is_tangent (circle_with_diameter M N) (line_through P F2))
  ∧ (angle_bisector F2 (line_through P F2) (line_through F2 A2) = line_through F2 M) := 
  sorry

end ellipse_geometry_l215_215984


namespace length_segment_PQ_l215_215921

-- Define the Cartesian equation of the circle
def cartesian_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 2 * y = 0

-- Define the parametric equation of the line
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-1 + t, t)

-- Define the polar equation of the ray OM
def polar_ray (θ : ℝ) : Prop :=
  θ = 3 * Real.pi / 4

-- The main theorem statement
theorem length_segment_PQ :
  ∀ (O P Q : ℝ × ℝ),
    (∃ (x y : ℝ), cartesian_circle x y ∧ 
    (O = (0, 0) ∧ P = (2*sqrt 2 * sin (3 * Real.pi / 4 - Real.pi / 4), 3 * Real.pi / 4)) ∧ 
    (∃ t : ℝ, parametric_line t = Q ∧ 
    (Q = (sqrt 2 / 2, 3 * Real.pi / 4)))) →
    dist O P - dist O Q = 3 * sqrt 2 / 2 :=
by sorry

end length_segment_PQ_l215_215921


namespace probability_defective_probability_not_defective_given_B3_l215_215726

variable (P_A_B1 P_A_B2 P_A_B3 : ℝ)
variable (P_A P_A_not_given_B3 P_B1 P_B2 P_B3 : ℝ)

-- Given conditions
def P_A_given_B1 : ℝ := 0.06
def P_B1 : ℝ := 0.1

def P_A_given_B2 : ℝ := 0.05
def P_B2 : ℝ := 0.4

def P_A_given_B3 : ℝ := 0.02
def P_B3 : ℝ := 0.5

-- Defining the probabilities
def P_A_B1 : ℝ := P_A_given_B1 * P_B1
def P_A_B2 : ℝ := P_A_given_B2 * P_B2
def P_A_B3 : ℝ := P_A_given_B3 * P_B3

-- Proofs required

theorem probability_defective : P_A = P_A_B1 + P_A_B2 + P_A_B3 := by sorry

theorem probability_not_defective_given_B3 : 
  P_A_not_given_B3 = 1 - P_A_given_B3 := by sorry

end probability_defective_probability_not_defective_given_B3_l215_215726


namespace solve_inequality_l215_215506

theorem solve_inequality (x : ℝ) : 
  (3 * x - 6 > 12 - 2 * x + x^2) ↔ (-1 < x ∧ x < 6) :=
sorry

end solve_inequality_l215_215506


namespace max_x_plus_2y_l215_215222

theorem max_x_plus_2y {x y : ℝ} (h : x^2 - x * y + y^2 = 1) :
  x + 2 * y ≤ (2 * Real.sqrt 21) / 3 :=
sorry

end max_x_plus_2y_l215_215222


namespace triangle_angle_C_triangle_side_sum_square_l215_215622

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ)
  (h1 : sqrt 3 * tan A * tan B - tan A - tan B = sqrt 3)
  (h2 : c = 2)
  (h3 : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ A + B < π)
  : C = π / 3 := 
sorry

theorem triangle_side_sum_square (A B C : ℝ) (a b c : ℝ)
  (h1 : sqrt 3 * tan A * tan B - tan A - tan B = sqrt 3)
  (h2 : c = 2)
  (h3 : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ A + B < π)
  : 20 / 3 < a^2 + b^2 ∧ a^2 + b^2 ≤ 8 := 
sorry

end triangle_angle_C_triangle_side_sum_square_l215_215622


namespace mark_total_spending_l215_215337

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end mark_total_spending_l215_215337


namespace tshirt_profit_l215_215695

-- Let T be the amount the shop makes off each t-shirt in dollars
variables {T : ℝ}

-- Condition 1: The Razorback shop makes $115 off each jersey
def jersey_amount := 115

-- Condition 2: During the Arkansas and Texas Tech game, the shop sold 113 t-shirts and 78 jerseys
def tshirts_sold := 113
def jerseys_sold := 78

-- Condition 3: A jersey costs $90 more than a t-shirt
def jersey_price_increase := 90

-- Condition 4: The amount the shop makes off each jersey is T + 90
def jersey_sale (T : ℝ) := T + jersey_price_increase

-- Given the conditions, prove that T == 25
theorem tshirt_profit : T + jersey_price_increase = jersey_amount → T = 25 :=
by
  intro h
  have h1 : 78 * (T + jersey_price_increase) = 78 * jersey_amount := calc
    78 * (T + jersey_price_increase) = 78 * T + 78 * 90 : by ring
    ... = 78T + 7020 : by unfold jersey_price_increase
    ... = 8970 : by unfold jersey_amount
  sorry

end tshirt_profit_l215_215695


namespace constant_sequence_from_conditions_l215_215560

variable (k b : ℝ) [Nontrivial ℝ]
variable (a_n : ℕ → ℝ)

-- Define the conditions function
def cond1 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond2 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (d : ℝ), ∀ n, a_n (n + 1) = a_n n + d) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond3 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b - (k * a_n n + b) = m)

-- Lean statement to prove the problem
theorem constant_sequence_from_conditions (k b : ℝ) [Nontrivial ℝ] (a_n : ℕ → ℝ) :
  (cond1 k b a_n ∨ cond2 k b a_n ∨ cond3 k b a_n) → 
  ∃ c : ℝ, ∀ n, a_n n = c :=
by
  -- To be proven
  intros
  sorry

end constant_sequence_from_conditions_l215_215560


namespace solve_trig_eq_l215_215679

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end solve_trig_eq_l215_215679


namespace tangency_condition_l215_215567

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 3)^2 = 4

theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m → x^2 = 9 - 9 * y^2 ∧ x^2 = 4 + m * (y + 3)^2 → ((m - 9) * y^2 + 6 * m * y + (9 * m - 5) = 0 → 36 * m^2 - 4 * (m - 9) * (9 * m - 5) = 0 ) ) → 
  m = 5 / 54 :=
by
  sorry

end tangency_condition_l215_215567


namespace value_of_B_l215_215821

noncomputable def find_B (r : Fin 6 → ℕ) : ℤ :=
  -∑ i1 in Finset.univ, ∑ i2 in Finset.univ, ∑ i3 in Finset.univ,
    ite (i1 = i2 ∨ i1 = i3 ∨ i2 = i3) 0 (r i1 * r i2 * r i3)

theorem value_of_B (r : Fin 6 → ℕ)
  (hr_sum : ∑ i in Finset.univ, r i = 11)
  (hr_prod : ∏ i in Finset.univ, r i = 24)
  (hr_pos : ∀ i, 0 < r i) :
  find_B r = -148 := sorry

end value_of_B_l215_215821


namespace scientific_notation_347000_l215_215378

theorem scientific_notation_347000 :
  347000 = 3.47 * 10^5 :=
by 
  -- Proof will go here
  sorry

end scientific_notation_347000_l215_215378


namespace t_100_mod_7_l215_215852

noncomputable def T : ℕ → ℕ 
| 1       := 11
| (n + 1) := 11 ^ (T n)

theorem t_100_mod_7 : (T 100) % 7 = 2 := 
by sorry

end t_100_mod_7_l215_215852


namespace BK_eq_BC_l215_215297

variable (ABC : Triangle)
variable (BL : LineSegment)
variable (K L : Point)
variable (B A C : Point)

-- Given conditions:
-- 1. In triangle ABC, the angle bisector BL is drawn.
-- Assume BL is the angle bisector of ∠ABC.
axiom angle_bisector_BL : angle_bisector BL ∠A B ∣ C

-- 2. BL = AB
axiom BL_eq_AB : length BL = length A B

-- 3. A point K is selected on the extension of BL beyond point L.
axiom K_on_extension_BL : is_on_extension K BL L

-- 4. ∠BAK + ∠BAL = 180°
axiom angle_sum_BAK_BAL : ∠BAK + ∠BAL = 180

-- Question: prove that BK = BC
theorem BK_eq_BC (ABC : Triangle) : length B K = length B C := by
  sorry

end BK_eq_BC_l215_215297


namespace concurrent_segments_l215_215860

noncomputable section

open_locale classical

variables {k : Type*} [euclidean_space k]
variables {E F : k} (c : ℝ)
variables (circle K : set k)
variables (tangent e : set k)
variables {A B A₁ B₁ : k}

-- Define the circle with diameter EF
def is_circle_diameter (K : set k) (E F : k) : Prop :=
  ∃ (O : k) (r : ℝ), K = metric.sphere O r ∧ dist O E = r ∧ dist O F = r ∧ dist E F = 2 * r

-- Definition of the tangent e at E
def is_tangent (e : set k) (K : set k) (E : k) : Prop :=
  K ∩ e = {E} ∧ (∀ P ∈ e, ∀ Q ∈ K, E ≠ P ∧ E ≠ Q → ⟪E - P, E - Q⟫ = 0)

-- Definition of pairs A, B on tangent e such that E in AB and AE * EB = c
def valid_pairs (A B : k) (E : k) (c : ℝ) : Prop :=
  A ≠ B ∧ (∀ (P : k), (P = A ∨ P = B → E ∈ segment k A B) ∧ norm (E - A) * norm (E - B) = c)

-- Intersection points (A₁, B₁) for lines AF and BF with circle k
def intersection_points (A B F : k) (K : set k) (A₁ B₁ : k) : Prop :=
  ∃ (P Q : k), P ∈ line F A ∧ Q ∈ line F B ∧ P ∈ K ∧ Q ∈ K ∧ (A₁ = P ∧ B₁ = Q)

-- The final theorem statement: segments A₁B₁ all concur in one point
theorem concurrent_segments
  (h1 : is_circle_diameter K E F)
  (h2 : is_tangent e K E)
  (h3 : ∃ A B, valid_pairs A B E c)
  (h4 : ∃ A₁ B₁, intersection_points A B F K A₁ B₁)
: ∃ X : k, ∀ A₁ B₁, intersection_points A B F K A₁ B₁ → collinear k {A₁, B₁, X} :=
sorry

end concurrent_segments_l215_215860


namespace Jeanine_more_pencils_than_Clare_l215_215302

variables (Jeanine_pencils : ℕ) (Clare_pencils : ℕ)

def Jeanine_initial_pencils := 18
def Clare_initial_pencils := Jeanine_initial_pencils / 2
def Jeanine_pencils_given_to_Abby := Jeanine_initial_pencils / 3
def Jeanine_remaining_pencils := Jeanine_initial_pencils - Jeanine_pencils_given_to_Abby

theorem Jeanine_more_pencils_than_Clare :
  Jeanine_remaining_pencils - Clare_initial_pencils = 3 :=
by
  -- This is just the statement, the proof is not provided as instructed.
  sorry

end Jeanine_more_pencils_than_Clare_l215_215302


namespace ABCD_concyclic_l215_215056

theorem ABCD_concyclic 
  (A B C D I1 I2 E F P : Point) 
  (h_convex : ConvexQuadrilateral A B C D)
  (h_I1 : Incenter A B C I1)
  (h_I2 : Incenter D B C I2)
  (h_intersect_E : Line I1 I2 ∩ Line A B = {E})
  (h_intersect_F : Line I1 I2 ∩ Line D C = {F})
  (h_extend_intersect : Line A B ∩ Line D C = {P})
  (h_PE_PF : dist P E = dist P F) : 
  Concyclic A B C D := 
sorry

end ABCD_concyclic_l215_215056


namespace infinite_triples_sum_of_squares_l215_215363

theorem infinite_triples_sum_of_squares :
  ∀ (n : ℤ), ∃ (N : ℤ) (m : ℤ),
    (N = 2 * n^2 * (n + 1)^2) ∧
    (∃ a b : ℤ, N = a^2 + b^2) ∧
    (∃ c d : ℤ, (N + 1) = c^2 + d^2) ∧
    (∃ e f : ℤ, (N + 2) = e^2 + f^2) :=
by
  intro n
  let N := 2 * n^2 * (n + 1)^2
  use N
  let m := n * (n + 1)
  use m
  split
  { refl }
  split
  { use m, m
    rw [← bit0]
    refl }
  split
  { use (n^2 - 1), (n^2 + 2 * n)
    sorry }
  { use (m + 1), (m - 1)
    sorry }

end infinite_triples_sum_of_squares_l215_215363


namespace total_spent_is_49_l215_215796

-- Define the prices of items
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define Paula's purchases
def paula_bracelets := 3
def paula_keychains := 2
def paula_coloring_book := 1
def paula_stickers := 4

-- Define Olive's purchases
def olive_bracelets := 2
def olive_coloring_book := 1
def olive_toy_car := 1
def olive_stickers := 3

-- Calculate total expenses
def paula_total := paula_bracelets * price_bracelet + paula_keychains * price_keychain + paula_coloring_book * price_coloring_book + paula_stickers * price_sticker
def olive_total := olive_coloring_book * price_coloring_book + olive_bracelets * price_bracelet + olive_toy_car * price_toy_car + olive_stickers * price_sticker
def total_expense := paula_total + olive_total

-- Prove the total expenses amount to $49
theorem total_spent_is_49 : total_expense = 49 :=
by
  have : paula_total = (3 * 4) + (2 * 5) + (1 * 3) + (4 * 1) := rfl
  have : olive_total = (1 * 3) + (2 * 4) + (1 *6) + (3 * 1) := rfl
  have : paula_total = 29 := rfl
  have : olive_total = 20 := rfl
  have : total_expense = 29 + 20 := rfl
  exact rfl

end total_spent_is_49_l215_215796


namespace distinct_ways_to_place_digits_l215_215419

theorem distinct_ways_to_place_digits :
  let boxes : Finset ℕ := {0, 1, 2, 3, 4, 5}
  (boxes.card = 6) →
  let placed_digits : Finset ℕ := {1, 2, 3, 4, 5}
  (placed_digits.card = 5) →
  ∃ ways : ℕ, ways = 6! ∧ ways = 720 :=
by {
  intros,
  sorry
}

end distinct_ways_to_place_digits_l215_215419


namespace books_per_continent_l215_215480

-- Definition of the given conditions
def total_books := 488
def continents_visited := 4

-- The theorem we need to prove
theorem books_per_continent : total_books / continents_visited = 122 :=
sorry

end books_per_continent_l215_215480


namespace max_non_overlapping_diagonals_l215_215399

-- Define the structure of the grid
structure Grid6x6 :=
  (cells : list (fin 6 × fin 6))

-- Define the properties of diagonal lines
def no_common_points (lines : list (Grid6x6 × Grid6x6)) : Prop :=
  ∀ (l1 l2 : Grid6x6 × Grid6x6), l1 ∈ lines → l2 ∈ lines → 
  l1 ≠ l2 → l1.1 ∩ l1.2 = ∅ ∧ l2.1 ∩ l2.2 = ∅

-- The maximum number of non-overlapping diagonals
theorem max_non_overlapping_diagonals :
  ∃ (lines : list (Grid6x6 × Grid6x6)), 
  no_common_points lines ∧ lines.length = 21 :=
sorry

end max_non_overlapping_diagonals_l215_215399


namespace find_f8_l215_215956

theorem find_f8 (f : ℕ → ℕ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : f 8 = 26 :=
by
  sorry

end find_f8_l215_215956


namespace speed_in_still_water_l215_215421

def upstream_speed : ℝ := 15
def downstream_speed : ℝ := 35
def still_water_speed : ℝ := (upstream_speed + downstream_speed) / 2

theorem speed_in_still_water : still_water_speed = 25 :=
by
  sorry

end speed_in_still_water_l215_215421


namespace range_of_real_number_l215_215962

theorem range_of_real_number (a : ℝ) : (a > 0) ∧ (a - 1 > 0) → a > 1 :=
by
  sorry

end range_of_real_number_l215_215962


namespace formula_for_an_l215_215433

def sequence (x : ℕ → ℚ) : Prop :=
  x 1 = 0 ∧ x 2 = 1 / 2 ∧ ∀ n, n > 2 → x n = (x (n - 1) + x (n - 2)) / 2

theorem formula_for_an (x : ℕ → ℚ) (h_seq : sequence x) (n : ℕ) :
  n ≥ 1 → 
  (x (n + 1) - x n) = (1 : ℚ) / (bit0 (bit0 1) ^ n) * (-1) ^ (n - 1) := 
by {
  sorry
}

end formula_for_an_l215_215433


namespace value_range_f_l215_215004

def f (x : ℝ) : ℝ := 3 * sin (2 * x - π / 6)

theorem value_range_f :
  ∀ x, 0 ≤ x ∧ x ≤ π / 2 →
  -3 / 2 ≤ f x ∧ f x ≤ 3 := by
  sorry

end value_range_f_l215_215004


namespace magician_decks_l215_215798

theorem magician_decks :
  ∀ (initial_decks price_per_deck earnings decks_sold decks_left_unsold : ℕ),
  initial_decks = 5 →
  price_per_deck = 2 →
  earnings = 4 →
  decks_sold = earnings / price_per_deck →
  decks_left_unsold = initial_decks - decks_sold →
  decks_left_unsold = 3 :=
by
  intros initial_decks price_per_deck earnings decks_sold decks_left_unsold
  intros h_initial h_price h_earnings h_sold h_left
  rw [h_initial, h_price, h_earnings] at *
  sorry

end magician_decks_l215_215798


namespace fraction_not_covered_l215_215039

theorem fraction_not_covered {d_small d_big : ℝ} (h1 : d_small = 10) (h2 : d_big = 12) :
  let r_small := d_small / 2,
      r_big := d_big / 2,
      A_small := Real.pi * r_small^2,
      A_big := Real.pi * r_big^2,
      diff := A_big - A_small,
      fraction := diff / A_big
  in fraction = 11 / 36 :=
by
  sorry

end fraction_not_covered_l215_215039


namespace points_in_circle_l215_215353

theorem points_in_circle (points : Set (ℝ × ℝ)) (h_points_card : points.card = 51) : 
  ∃ (c : ℝ × ℝ), ∃ (r : ℝ), r = 1 / 7 ∧ (∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (dist p1 c ≤ r ∧ dist p2 c ≤ r ∧ dist p3 c ≤ r)) := by
    sorry

end points_in_circle_l215_215353


namespace money_problem_l215_215779

-- Define the conditions and the required proof
theorem money_problem (B S : ℕ) 
  (h1 : B = 2 * S) -- Condition 1: Brother brought twice as much money as the sister
  (h2 : B - 180 = S - 30) -- Condition 3: Remaining money of brother and sister are equal
  : B = 300 ∧ S = 150 := -- Correct answer to prove
  
sorry -- Placeholder for proof

end money_problem_l215_215779


namespace area_below_line_l215_215457

variables (a b c d : ℝ)
variables (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d)

theorem area_below_line :
  let rect_area := (2 * a + b) * (d + 2 * c),
      tri_area := (1 / 2) * (d + 2 * c + b + a) * (a + b + 2 * c)
  in rect_area - tri_area = (2 * a + b) * (d + 2 * c) - (1 / 2) * (d + 2 * c + b + a) * (a + b + 2 * c) := 
by
  sorry

end area_below_line_l215_215457


namespace division_of_decimals_l215_215737

theorem division_of_decimals : (0.05 / 0.002) = 25 :=
by
  -- Proof will be filled here
  sorry

end division_of_decimals_l215_215737


namespace shift_upwards_by_2_l215_215382

-- Define the original function
def f (x : ℝ) : ℝ := - x^2

-- Define the shift transformation
def shift_upwards (g : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x, g x + k

-- Define the expected result after shifting f by 2 units upwards
def shifted_f (x : ℝ) : ℝ := - x^2 + 2

-- The proof statement itself
theorem shift_upwards_by_2 :
  shift_upwards f 2 = shifted_f :=
by sorry

end shift_upwards_by_2_l215_215382


namespace exponential_range_l215_215558

theorem exponential_range (k : ℝ) (h : k > 0) :
  (set.range (λ x : ℝ, if x < 1 then e^k else e^(k * x))) = set.Ici (e^k) :=
by
  sorry

end exponential_range_l215_215558


namespace triangle_area_division_l215_215621

theorem triangle_area_division
  (A B C D E : Point)
  (angle_A : ∠ A = 60)
  (angle_B : ∠ B = 45)
  (angle_ADE : ∠ ADE = 45)
  (DE_divides_area : triangleArea A B C / 2 = triangleArea A D E) :
  (length AD / length AB) = 2 / (2 + sqrt 2) :=
sorry

end triangle_area_division_l215_215621


namespace g_at_1_l215_215262

theorem g_at_1 : ∀ (g : ℕ → ℤ), (∀ x, g (x + 1) = 2 * x - 3) → g 1 = -3 :=
by
  intro g h
  have h1 : g 1 = g (0 + 1) := by rfl
  rw [h 0] at h1
  exact h1

end g_at_1_l215_215262


namespace acute_angle_at_3_15_l215_215029

/-- Each minute on a clock represents 6 degrees. -/
def degrees_per_minute := 6

/-- At 3:00, the hour hand is at 90 degrees from the 12 o'clock position. -/
def degrees_at_3 := 3 * 30

/-- The hour hand moves 7.5 degrees further from the 3 o'clock position by 3:15. -/
def degrees_hour_hand_at_3_15 := degrees_at_3 + (15 / 60) * 30

/-- The minute hand is exactly 90 degrees from the 12 o'clock position at 3:15. -/
def degrees_minute_hand_at_3_15 := 15 * degrees_per_minute

/-- The acute angle formed between the hour and minute hands of a clock at 3:15 is 7.5 degrees. -/
theorem acute_angle_at_3_15 : abs (degrees_hour_hand_at_3_15 - degrees_minute_hand_at_3_15) = 7.5 := by
  sorry

end acute_angle_at_3_15_l215_215029


namespace quadrant_classification_l215_215576

theorem quadrant_classification :
  ∀ (x y : ℝ), (4 * x - 3 * y = 24) → (|x| = |y|) → 
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  intros x y h_line h_eqdist
  sorry

end quadrant_classification_l215_215576


namespace min_questionnaires_mailed_l215_215424

theorem min_questionnaires_mailed (responses_needed : ℕ) (response_rate : ℝ) (min_questionnaires : ℕ) : 
  responses_needed = 300 → 
  response_rate = 0.62 → 
  min_questionnaires = 484 := 
by 
  intros h1 h2 h3
  sorry

end min_questionnaires_mailed_l215_215424


namespace discount_comparison_l215_215458

theorem discount_comparison :
  ∃ n : ℕ, n > max (0.28 * 100) (max (1 - (0.92)^3 * 100) (0.25 * 100)) ∧ n = 29 :=
by
  sorry

end discount_comparison_l215_215458


namespace angle_mod_equiv_theta_third_quadrant_beta_in_interval_and_equiv_l215_215912

def alpha := (2010 : ℝ) * (Real.pi / 180)
def theta := Real.arctan 1 7 / 6
def k (n : ℤ) := theta + 2 * n * Real.pi
def beta (n : ℤ) := - ((7 * Real.pi / 6) + 2 * n * Real.pi)

theorem angle_mod_equiv : ∃ (n : ℤ), alpha = theta + 2 * n * Real.pi := sorry

theorem theta_third_quadrant : Real.pi < theta ∧ theta < 3 * Real.pi / 2 := sorry

theorem beta_in_interval_and_equiv {β : ℝ} :
  β ∈ set.Ico (- 5 * Real.pi) 0 ∧ (∃ (n : ℤ), β = beta n) →
  β = - (29 * Real.pi / 6) ∨ β = - (17 * Real.pi / 6) ∨ β = - (5 * Real.pi / 6) := sorry

end angle_mod_equiv_theta_third_quadrant_beta_in_interval_and_equiv_l215_215912


namespace power_mod_congruence_l215_215418

theorem power_mod_congruence (h : 3^400 ≡ 1 [MOD 500]) : 3^800 ≡ 1 [MOD 500] :=
by {
  sorry
}

end power_mod_congruence_l215_215418


namespace find_slip_3_5_in_correct_cup_l215_215009

noncomputable def cupSumProblem (slips : List ℝ) (cups : List (List ℝ)) (correctCup : ℝ) : Prop :=
  cups.length = 5 ∧
  slips.sum = 35 ∧
  cups[0].sum = 5 ∧
  cups[1].sum = 6 ∧
  cups[2].sum = 7 ∧
  cups[3].sum = 8 ∧
  cups[4].sum = 9 ∧
  cups[4].contains 2 ∧
  cups[1].contains 3 ∧
  cups[3].contains correctCup

theorem find_slip_3_5_in_correct_cup :
  ∃ (cups : List (List ℝ)), cupSumProblem [2, 2, 2, 2.5, 2.5, 3, 3, 3, 3, 3.5, 4, 4.5] cups 3.5 :=
begin
  sorry
end

end find_slip_3_5_in_correct_cup_l215_215009


namespace f_prime_at_zero_l215_215326

def f (x : ℝ) (n : ℕ) : ℝ := (List.range (n + 1)).map (λ k => x + k).prod

theorem f_prime_at_zero (n : ℕ) : (deriv (λ x => f x n) 0) = n! :=
by
  sorry

end f_prime_at_zero_l215_215326


namespace correct_statements_about_h_l215_215918

def g (x : ℝ) : ℝ := 2^x

def symmetric_wrt_line (f g : ℝ → ℝ) (L : ℝ → ℝ) : Prop :=
  ∀ x y, L (f x) = g y

def f : ℝ → ℝ := sorry  -- assume f is given such that it's symmetric to g wrt y=x

def h (x : ℝ) : ℝ := f (1 - |x|)

theorem correct_statements_about_h :
  (∀ x, h x = h (-x)) ∧
  ∀ x y : ℝ, -1 < x → x < 0 → -1 < y → y < 0 → x < y → h x < h y :=
sorry

end correct_statements_about_h_l215_215918


namespace solve_polynomial_equation_l215_215677

theorem solve_polynomial_equation (x : ℝ) :
  (x - 4)^4 + (x - 6)^4 = 240 ↔ x = 5 + (Real.sqrt (5 * Real.sqrt 2 - 3)) ∨ x = 5 - (Real.sqrt (5 * Real.sqrt 2 - 3)) :=
by
  sorry

end solve_polynomial_equation_l215_215677


namespace smallest_a_exists_l215_215322

theorem smallest_a_exists (P : ℤ[X]) (a : ℤ) (h₀ : a > 0)
  (h₁ : P.eval 2 = a) (h₂ : P.eval 4 = a) (h₃ : P.eval 6 = a) (h₄ : P.eval 8 = a)
  (h₅ : P.eval 1 = -a) (h₆ : P.eval 3 = -a) (h₇ : P.eval 5 = -a) (h₈ : P.eval 7 = -a) :
  a = 315 := 
sorry

end smallest_a_exists_l215_215322


namespace dan_speed_must_exceed_48_l215_215704

theorem dan_speed_must_exceed_48 (d : ℕ) (s_cara : ℕ) (time_delay : ℕ) : 
  d = 120 → s_cara = 30 → time_delay = 3 / 2 → ∃ v : ℕ, v > 48 :=
by
  intro h1 h2 h3
  use 49
  sorry

end dan_speed_must_exceed_48_l215_215704


namespace collinear_m_value_right_angle_triangle_m_value_l215_215249

theorem collinear_m_value
    (m : ℝ)
    (OA OB OC : ℝ × ℝ)
    (H1 : OA = (3, -4))
    (H2 : OB = (6, -3))
    (H3 : OC = (5 - m, -3 - m))
    (collinear : ∃ λ : ℝ, λ ≠ 0 ∧ (OB.1 - OA.1, OB.2 - OA.2) = λ * (OC.1 - OA.1, OC.2 - OA.2)) :
    m = 1 / 2 := 
begin
  sorry
end

theorem right_angle_triangle_m_value
    (m : ℝ)
    (OA OB OC : ℝ × ℝ)
    (H1 : OA = (3, -4))
    (H2 : OB = (6, -3))
    (H3 : OC = (5 - m, -3 - m))
    (right_angle : (OC.1 - OA.1, OC.2 - OA.2) ⬝ (OC.1 - OB.1, OC.2 - OB.2) = 0) :
    m = 1 + real.sqrt 3 ∨ m = 1 - real.sqrt 3 := 
begin
  sorry
end

end collinear_m_value_right_angle_triangle_m_value_l215_215249


namespace solve_equation_l215_215521

theorem solve_equation : ∃ x : ℝ, (real.sqrt (real.sqrt (4 - x))) = -2 / 3 ↔ x = 308 / 81 :=
by
  sorry

end solve_equation_l215_215521


namespace sum_of_squares_of_distances_const_l215_215699

theorem sum_of_squares_of_distances_const (A B C P O : ℝ) (a : ℝ) (h_tri_equil : -- details of equilateral triangle)
  (h_circle : -- conditions of circle dividing each side into three equal parts)
  (h_P_on_circle : -- condition stating P lies on the circle):
  ( (dist P A) ^ 2 + (dist P B) ^ 2 + (dist P C) ^ 2 ) = 2 * a ^ 2 := 
sorry

end sum_of_squares_of_distances_const_l215_215699


namespace find_f_zero_l215_215563

variable (f : ℝ → ℝ)
variable (hf : ∀ x y : ℝ, f (x + y) = f x + f y + 1 / 2)

theorem find_f_zero : f 0 = -1 / 2 :=
by
  sorry

end find_f_zero_l215_215563


namespace egg_roll_ratio_l215_215659

-- Define the conditions as hypotheses 
variables (Matthew_eats Patrick_eats Alvin_eats : ℕ)

-- Define the specific conditions
def conditions : Prop :=
  (Matthew_eats = 6) ∧
  (Patrick_eats = Alvin_eats / 2) ∧
  (Alvin_eats = 4)

-- Define the ratio of Matthew's egg rolls to Patrick's egg rolls
def ratio (a b : ℕ) := a / b

-- State the theorem with the corresponding proof problem
theorem egg_roll_ratio : conditions Matthew_eats Patrick_eats Alvin_eats → ratio Matthew_eats Patrick_eats = 3 :=
by
  -- Proof is not required as mentioned. Adding sorry to skip the proof.
  sorry

end egg_roll_ratio_l215_215659


namespace supermarket_arrangements_l215_215463

-- Definitions based on problem conditions
def lanes := {1, 2, 3, 4, 5, 6}
def non_adjacent_sets := {{1, 3, 5}, {1, 3, 6}, {1, 4, 6}, {2, 4, 6}}
def checkout_states := {1, 2, 3} -- 1: checkout point 1, 2: checkout point 2, 3: both checkout points

-- The total number of different arrangements is given by:
theorem supermarket_arrangements : 4 * (3 ^ 3) = 108 :=
by
  -- In the actual proof we would show that there are 4 ways to choose non-adjacent lanes
  -- and each choice of lane has 3 options for the checkout points.
  sorry

end supermarket_arrangements_l215_215463


namespace geometric_series_sum_eq_4_div_3_l215_215125

theorem geometric_series_sum_eq_4_div_3 (a : ℝ) (r : ℝ) (h₀ : a = 1) (h₁ : r = 1 / 4) :
  ∑' n : ℕ, a * r^n = 4 / 3 := by
  sorry

end geometric_series_sum_eq_4_div_3_l215_215125


namespace solve_trig_eq_l215_215681

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end solve_trig_eq_l215_215681


namespace laura_survey_total_people_l215_215309

theorem laura_survey_total_people (total_people misinformed_people : ℕ)
  (H1 : misinformed_people = 31)
  (H2 : 0.523 * (0.754 * total_people) = misinformed_people) :
  total_people = 78 :=
by
  sorry

end laura_survey_total_people_l215_215309


namespace find_y_l215_215327

noncomputable def y_value (c d y : ℝ) : Prop :=
  let s := (3*c)^(3*d)
  ∧ d ≠ 0
  ∧ s = c^d * y^(3*d)
  ∧ 3*c = y

theorem find_y (c d : ℝ) (h : d ≠ 0) : ∃ y : ℝ, y_value c d y :=
by
  exists 3*c
  simp [y_value]
  sorry

end find_y_l215_215327


namespace correlation_height_weight_l215_215285

def is_functional_relationship (pair: String) : Prop :=
  pair = "The area of a square and its side length" ∨
  pair = "The distance traveled by a vehicle moving at a constant speed and time"

def has_no_correlation (pair: String) : Prop :=
  pair = "A person's height and eyesight"

def is_correlation (pair: String) : Prop :=
  ¬ is_functional_relationship pair ∧ ¬ has_no_correlation pair

theorem correlation_height_weight :
  is_correlation "A person's height and weight" :=
by sorry

end correlation_height_weight_l215_215285


namespace large_block_volume_l215_215761

theorem large_block_volume (W D L : ℝ) (h1 : W * D * L = 3) : 
  (2 * W) * (2 * D) * (3 * L) = 36 := 
by 
  sorry

end large_block_volume_l215_215761


namespace opponents_team_points_l215_215988

theorem opponents_team_points (M D V O : ℕ) (hM : M = 5) (hD : D = 3) 
    (hV : V = 2 * (M + D)) (hO : O = (M + D + V) + 16) : O = 40 := by
  sorry

end opponents_team_points_l215_215988


namespace viewing_spot_coordinate_correct_l215_215398

-- Define the coordinates of the landmarks
def first_landmark := 150
def second_landmark := 450

-- The expected coordinate of the viewing spot
def expected_viewing_spot := 350

-- The theorem that formalizes the problem
theorem viewing_spot_coordinate_correct :
  let distance := second_landmark - first_landmark
  let fractional_distance := (2 / 3) * distance
  let viewing_spot := first_landmark + fractional_distance
  viewing_spot = expected_viewing_spot := 
by
  -- This is where the proof would go
  sorry

end viewing_spot_coordinate_correct_l215_215398


namespace m_parallel_β_l215_215644

-- Define the parallel and perpendicular relations
def parallel (α β : Type*) [plane α] [plane β] : Prop := ∀ p : α, p ∈ β
def perpendicular (m α : Type*) [line m] [plane α] : Prop := ∀ p : m, p ∉ α

variables {α β : Type*} [plane α] [plane β]
variables {m n : Type*} [line m] [line n]

-- Given conditions
axiom α_parallel_β : parallel α β
axiom m_perpendicular_α : perpendicular m α
axiom n_perpendicular_β : perpendicular n β

-- Statement to prove: m is parallel to β
theorem m_parallel_β : parallel m β :=
sorry

end m_parallel_β_l215_215644


namespace geometric_sequence_a3_l215_215885

theorem geometric_sequence_a3 (a : ℕ → ℝ)
  (h : ∀ n m : ℕ, a (n + m) = a n * a m)
  (pos : ∀ n, 0 < a n)
  (a1 : a 1 = 1)
  (a5 : a 5 = 9) :
  a 3 = 3 := by
  sorry

end geometric_sequence_a3_l215_215885


namespace evaluate_exponent_sum_l215_215862

theorem evaluate_exponent_sum : 
  let i : ℂ := Complex.I in 
  i^14760 + i^14761 + i^14762 + i^14763 = 0 := by
  sorry

end evaluate_exponent_sum_l215_215862


namespace solve_for_d_l215_215356

theorem solve_for_d (r s t d c : ℝ)
  (h1 : (t = -r - s))
  (h2 : (c = rs + rt + st))
  (h3 : (t - 1 = -(r + 5) - (s - 4)))
  (h4 : (c = (r + 5) * (s - 4) + (r + 5) * (t - 1) + (s - 4) * (t - 1)))
  (h5 : (d = -r * s * t))
  (h6 : (d + 210 = -(r + 5) * (s - 4) * (t - 1))) :
  d = 240 ∨ d = 420 :=
by
  sorry

end solve_for_d_l215_215356


namespace exists_tangent_circle_l215_215899

variable (S : Circle) (A : Point) (l : Line)

-- Lean statement for the problem above
theorem exists_tangent_circle (hA : A ∈ S) :
  ∃ S' : Circle, Circle.tangent_at S S' A ∧ ∃ B : Point, B ∈ S' ∧ Line.tangent_at l S' B :=
sorry

end exists_tangent_circle_l215_215899


namespace value_of_f_f_3_l215_215331

def f (x : ℝ) := 3 * x^2 + 3 * x - 2

theorem value_of_f_f_3 : f (f 3) = 3568 :=
by {
  -- Definition of f is already given in the conditions
  sorry
}

end value_of_f_f_3_l215_215331


namespace grocer_coffee_stock_l215_215070

theorem grocer_coffee_stock 
  (coffee_stock_1 : ℝ)
  (decaf_percent_1 : ℝ)
  (coffee_stock_2 : ℝ)
  (decaf_percent_2 : ℝ)
  (total_decaf_percent : ℝ)
  (total_weight : ℝ)
  (total_stock : coffee_stock_1 + coffee_stock_2 = total_weight)
  (decaf_1 : coffee_stock_1 * decaf_percent_1)
  (decaf_2 : coffee_stock_2 * decaf_percent_2)
  (combined_decaf : (coffee_stock_1 * decaf_percent_1) + (coffee_stock_2 * decaf_percent_2) = total_weight * total_decaf_percent)
  : coffee_stock_2 = 100 := 
sorry

end grocer_coffee_stock_l215_215070


namespace unique_property_of_rectangles_l215_215092

-- Definitions of the conditions
structure Quadrilateral (Q : Type*) :=
(sum_of_interior_angles : ∀ (q : Q), angle (interior q) = 360)

structure Rectangle (R : Type*) extends Quadrilateral R :=
(diagonals_bisect_each_other : ∀ (r : R), bisects (diagonals r))
(diagonals_equal_length : ∀ (r : R), equal_length (diagonals r))
(diagonals_not_necessarily_perpendicular : ∀ (r : R), not (necessarily_perpendicular (diagonals r)))

structure Rhombus (H : Type*) extends Quadrilateral H :=
(diagonals_bisect_each_other : ∀ (h : H), bisects (diagonals h))
(diagonals_perpendicular : ∀ (h : H), perpendicular (diagonals h))
(diagonals_not_necessarily_equal_length : ∀ (h : H), not (necessarily_equal_length (diagonals h)))

-- The proof statement
theorem unique_property_of_rectangles (R : Type*) [Rectangle R] (H : Type*) [Rhombus H] :
  ∀ (r : R), ∃ (h : H), equal_length (diagonals r) ∧ not (equal_length (diagonals h)) :=
sorry

end unique_property_of_rectangles_l215_215092


namespace range_of_a_l215_215927

noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x ^ 2 + 3)
noncomputable def g (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 - Real.log x - a
noncomputable def mu (x : ℝ) : ℝ := (1 / 2) * x ^ 2 - Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Icc (1 : ℝ) (2 : ℝ), g x a = 0) ∧ 
  (∀ x1 ∈ Ioo (0 : ℝ) (2 : ℝ), ∃ x2 ∈ Icc (1 : ℝ) (2 : ℝ), f x1 = g x2 a) ↔
  a ∈ Icc (1/2 : ℝ) (4 / 3 - Real.log 2) :=
sorry

end range_of_a_l215_215927


namespace ryan_funding_l215_215358

theorem ryan_funding (total_cost avg_funding recruit_people already_have : ℕ)
  (h1 : total_cost = 1000)
  (h2 : avg_funding = 10)
  (h3 : recruit_people = 80)
  (h4 : already_have = total_cost - recruit_people * avg_funding) :
  already_have = 200 :=
by
  rw [h1, h2, h3]
  simp
  exact h4

end ryan_funding_l215_215358


namespace five_digit_even_divisible_by_4_count_l215_215586

theorem five_digit_even_divisible_by_4_count :
  let even_digits := {0, 2, 4, 6, 8}
  let is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
  let is_valid_number (n : ℕ) : Prop :=
    n >= 10000 ∧ n <= 99999 ∧
    (∀ d, d ∈ (n.digits 10) → d ∈ even_digits) ∧
    is_divisible_by_4 n
  (finset.filter is_valid_number (finset.range 100000)).card = 2875 :=
by 
  sorry

end five_digit_even_divisible_by_4_count_l215_215586


namespace construct_triangle_correct_l215_215499

variable (Triangle : Type)
variable (Point : Triangle → Type)

variable (M K_a M_b : Point)

structure TriangleABC where
  A B C : Point
  isOrthocenter : (M : Point) (triangle : TriangleABC) → Prop

structure FootOfAltitude where
  foot : Point
  correspondingVertex : Point
  isFootOfAltitude : (M_b : FootOfAltitude) (triangle : TriangleABC) → Prop

structure Midpoint where
  midpoint : Point
  ofSegment : Point × Point
  isMidpoint : (K_a : Midpoint) (triangle : TriangleABC) → Prop

noncomputable def construct_trianlge (M K_a M_b : Point) : Triangle :=
  let T : TriangleABC := sorry
  let footAltitude : FootOfAltitude := sorry
  let midPoint : Midpoint := sorry
  T

theorem construct_triangle_correct :
  ∀ (T : TriangleABC) (M_b : FootOfAltitude) (K_a : Midpoint),
  T.isOrthocenter M →
  M_b.isFootOfAltitude M_b T →
  K_a.isMidpoint K_a T →
  M ∈ { T | T.isOrthocenter M } ∧
  M_b ∈ { F | F.isFootOfAltitude M_b T } ∧
  K_a ∈ { M | M.isMidpoint K_a T } :=
by
  intros
  sorry

end construct_triangle_correct_l215_215499


namespace cos_angle_second_quadrant_l215_215595

theorem cos_angle_second_quadrant (A : ℝ) :
  (π / 2 < A ∧ A < π) ∧ sin A = 3 / 4 → cos A = - (real.sqrt 7) / 4 :=
by
  sorry

end cos_angle_second_quadrant_l215_215595


namespace f_decreasing_f_odd_l215_215544

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b

axiom negativity (x : ℝ) (h_pos : 0 < x) : f x < 0

theorem f_decreasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intros x
  sorry

end f_decreasing_f_odd_l215_215544


namespace find_a_n_l215_215204

theorem find_a_n (S : ℕ → ℚ) (a : ℕ → ℚ) (h1 : S 1 = 1) (h2 : S 2 = -3/2)
  (h3 : ∀ n, n ≥ 3 → S n - S (n-2) = 3 * (-1/2)^(n-1)) :
  ∀ n, a n = 
  if odd n then 4 - 3 * (1/2)^(n-1)
  else -4 + 3 * (1/2)^(n-1) := 
sorry

end find_a_n_l215_215204


namespace exists_m_sum_l215_215429

noncomputable def seq_a (n : ℕ) : ℂ :=
  (1 + complex.i) * (1 + complex.i / complex.sqrt 2) * 
  ∏ i in finset.range n \.succ, 1 + complex.i / complex.sqrt (i + 1)

theorem exists_m_sum (m : ℕ) (h : m = 1990) : 
  ∃ (m : ℕ), (finset.range m).sum (λ n, |seq_a n - seq_a (n + 1)|) = 1990 :=
begin
  use 1990,
  sorry
end

end exists_m_sum_l215_215429


namespace valid_M_check_l215_215431

theorem valid_M_check : 
  (∀ x ∈ {0, 1, e}, \ln (real.abs x) ∈ {0, 1}) → false :=
by
  sorry

end valid_M_check_l215_215431


namespace sum_of_possible_values_l215_215218

-- Define the conditions
def is_integer (a : ℤ) : Prop := True
def is_prime_number (n : ℕ) : Prop := n.prime

-- Define the main problem statement
theorem sum_of_possible_values (a : ℤ) (h : is_integer a) (h1 : is_prime_number (Int.natAbs (4 * a^2 - 12 * a - 27))) : 
  ∃ (s : ℤ), s = -1 + -2 + 4 + 5 ∧ s = 6 :=
by
  sorry

end sum_of_possible_values_l215_215218


namespace power_function_inverse_l215_215273

theorem power_function_inverse (f : ℝ → ℝ) (h₁ : f 2 = (Real.sqrt 2) / 2) : f⁻¹ 2 = 1 / 4 :=
by
  -- Lean proof will be filled here
  sorry

end power_function_inverse_l215_215273


namespace majka_numbers_product_l215_215656

/-- Majka created a three-digit funny and a three-digit cheerful number.
    - The funny number starts with an odd digit and alternates between odd and even.
    - The cheerful number starts with an even digit and alternates between even and odd.
    - All digits are distinct and nonzero.
    - The sum of these two numbers is 1617.
    - The product of these two numbers ends in 40.
    Prove that the product of these numbers is 635040.
-/
theorem majka_numbers_product :
  ∃ (a b c : ℕ) (D E F : ℕ),
    -- Define 3-digit funny number as (100 * a + 10 * b + c)
    -- with a and c odd, b even, and all distinct and nonzero
    (a % 2 = 1) ∧ (c % 2 = 1) ∧ (b % 2 = 0) ∧
    -- Define 3-digit cheerful number as (100 * D + 10 * E + F)
    -- with D and F even, E odd, and all distinct and nonzero
    (D % 2 = 0) ∧ (F % 2 = 0) ∧ (E % 2 = 1) ∧
    -- All digits are distinct and nonzero
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ D ∧ a ≠ E ∧ a ≠ F ∧
     b ≠ c ∧ b ≠ D ∧ b ≠ E ∧ b ≠ F ∧
     c ≠ D ∧ c ≠ E ∧ c ≠ F ∧
     D ≠ E ∧ D ≠ F ∧ E ≠ F) ∧
    (100 * a + 10 * b + c + 100 * D + 10 * E + F = 1617) ∧
    ((100 * a + 10 * b + c) * (100 * D + 10 * E + F) = 635040) := sorry

end majka_numbers_product_l215_215656


namespace no_such_sequence_exists_l215_215859

open Real

theorem no_such_sequence_exists :
  ¬(∃ (a : Fin 7 → ℝ), a 0 = 0 ∧ a 6 = 0 ∧ (∀ i : Fin 5, 2 ≤ i.val + 2 ∧ i.val + 2 ≤ 6 → a (i + 2) + a (i) > sqrt(3) * a (i + 1) ∧ (∀ j, 0 ≤ a j))) :=
by
  sorry

end no_such_sequence_exists_l215_215859


namespace number_of_female_athletes_l215_215082

theorem number_of_female_athletes (male_athletes female_athletes male_selected female_selected : ℕ)
  (h1 : male_athletes = 56)
  (h2 : female_athletes = 42)
  (h3 : male_selected = 8)
  (ratio : male_athletes / female_athletes = 4 / 3)
  (stratified_sampling : female_selected = (3 / 4) * male_selected)
  : female_selected = 6 := by
  sorry

end number_of_female_athletes_l215_215082


namespace students_only_one_language_l215_215286

-- Define the sets of students for each language class
def French := {1, 2, 3, ..., 30} -- Assume we have numbers representing students
def Spanish := {31, 32, 33, ..., 55} -- Continued numbering for Spanish class
def German := {56, 57, 58, ..., 75} -- Continued numbering for German class

-- Define the intersection sizes for given overlaps
def French_Spanish := 10
def French_German := 7
def Spanish_German := 5
def All_three := 4

-- Given total students in language classes considering overlaps
def total_students := 75

-- Given total number of students actually enrolled in exactly one language
def only_one_language_students := 45

-- Lean 4 statement to express the proof
theorem students_only_one_language :
  ∀ (French Spanish German : Finset ℕ)
    (French_Spanish French_German Spanish_German All_three total_students only_one_language_students : ℕ),
  French.card + Spanish.card + German.card - 
  (French.card ∩ Spanish.card + French.card ∩ German.card + Spanish.card ∩ German.card) +
  All_three = total_students →
  total_students - (French_Spanish + French_German + Spanish_German - 2 * All_three) - All_three = only_one_language_students :=
begin
  sorry
end

end students_only_one_language_l215_215286


namespace minimum_value_f_on_interval_l215_215878

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 6)

theorem minimum_value_f_on_interval :
  ∃ c ∈ Icc (0 : ℝ) (Real.pi / 12), ∀ x ∈ Icc (0 : ℝ) (Real.pi / 12), f c ≤ f x ∧ f c = 1 :=
by
  use 0
  split
  { norm_num }
  { intros x hx
    sorry }

end minimum_value_f_on_interval_l215_215878


namespace find_curves_and_intersection_angle_l215_215069

noncomputable def ellipse_equation := "The equation of the ellipse is x^2/49 + y^2/36 = 1"
noncomputable def hyperbola_equation := "The equation of the hyperbola is x^2/9 - y^2/4 = 1"
noncomputable def cos_angle := "cos∠F_1PF_2 = 4/5"

theorem find_curves_and_intersection_angle (
    h_foci_distance : |((2:ℝ) * (Real.sqrt 13))| = 2 * Real.sqrt 13,
    h_axis_diff : ∃ a:ℝ, a - (a - 4) = 4,
    h_eccentricities_ratio : ∃ a:ℝ, Real.sqrt 13 / a / (Real.sqrt 13 / (a-4)) = 3/7
  ) :
    (ellipse_equation : "∃ a:ℝ, equation is x^2/a^2 + y^2/(a^2 - (Real.sqrt 49)/(a - 4)^2)=1 → x^2/49 + y^2/36=1") ∧
    (hyperbola_equation : "∃ a:ℝ, equation is x^2/((a-4)^2)-(y^2/((a-4)^2 - (Real.sqrt 49 - a^2)/a)= 1) → x^2/9 - y^2/4=1") ∧
    (cos_angle: "∃ f1 f2 p : ℝ, angle calculation, equation derived (apply cosine rule) → cos∠F_1PF_2 = 4/5") := by
      sorry

end find_curves_and_intersection_angle_l215_215069


namespace color_table_equal_cells_l215_215510

theorem color_table_equal_cells (m n : ℕ) (color : Fin m → Fin n → Bool)
  (h1 : ∀ i j : Fin m, ∀ k : Fin n, color i k = color j k → ∀ l : Fin n, l ≠ k → color i l ≠ color j l) :
  ∀ i : Fin m, (∀ j : Fin m, ∑ k : Fin n, ite (color i k) 1 0 = ∑ k : Fin n, ite (¬ color j k) 1 0 ) :=
sorry

end color_table_equal_cells_l215_215510


namespace kevin_correct_answer_l215_215999

theorem kevin_correct_answer (k : ℝ) (h : (20 + 1) * (6 + k) = 126 + 21 * k) :
  (20 + 1 * 6 + k) = 21 := by
sorry

end kevin_correct_answer_l215_215999


namespace total_toys_l215_215473

theorem total_toys (A M T : ℕ) (h1 : A = 3 * M + M) (h2 : T = A + 2) (h3 : M = 6) : A + M + T = 56 :=
by
  sorry

end total_toys_l215_215473


namespace perpendicular_projection_interior_point_l215_215736

structure ConvexPolygon (V : Type) [InnerProductSpace ℝ V] := 
  (vertices : List V)
  (is_convex : Convex ℝ (convexHull ℝ (set.of vertices)))

variables {V : Type} [InnerProductSpace ℝ V]

theorem perpendicular_projection_interior_point {P : V} {polygon : ConvexPolygon V} :
  P ∈ interior (convexHull ℝ (set.of polygon.vertices)) →
  ∃ (A B : V), 
    (A, B ∈ polygon.vertices) ∧ 
    let T := foot P (affineSpan ℝ A B) in 
    T ∈ openSegment ℝ A B :=
by
  sorry

end perpendicular_projection_interior_point_l215_215736


namespace line_point_coordinates_l215_215582

theorem line_point_coordinates (t : ℝ) (x y z : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) →
  t = 1/2 →
  (x, y, z) = (5, 3/2, 3) :=
by
  intros h1 h2
  sorry

end line_point_coordinates_l215_215582


namespace hyperbola_equation_l215_215906

-- Definitions and conditions
def is_parabola_focus (C : ℝ → ℝ) (point : ℝ × ℝ) : Prop :=
  C point.1 = 4 * point.2^2

def is_hyperbola_center (G : ℝ → ℝ → ℝ) (center : ℝ × ℝ) : Prop :=
  center = (0, 0)

def hyperbola_point_condition (G : ℝ → ℝ → ℝ) (point : ℝ × ℝ) : Prop :=
  G (point.1) (point.2) = 1

-- Problem
theorem hyperbola_equation :
  ∃ (G : ℝ → ℝ → ℝ), 
    is_hyperbola_center G (0, 0) ∧
    hyperbola_point_condition G (sqrt 5, 4) ∧
    is_parabola_focus (λ x, x) (1, 0) ∧
    (∀ x y, G x y = x^2 - (y^2 / 4)) :=
sorry

end hyperbola_equation_l215_215906


namespace minimum_value_l215_215295

-- Define the problem setup

variables {A B C D E F : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [AddCommGroup D] [AddCommGroup E] [AddCommGroup F]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C]
variables [Module ℝ D] [Module ℝ E] [Module ℝ F]

-- Conditions of the problem
variables (AB AC AE AF : ℝ → A)
variables (lambda mu : ℝ) (h_lambda_pos : 0 < lambda) (h_mu_pos : 0 < mu)

-- Definition of midpoint and collinearity condition
def midpoint_eq (D : A) : Prop :=
  D = (1/2 : ℝ) • (AB lambda + AC mu)

def collinear (D E F : A) : Prop :=
  ∀ (k : ℝ), D = k • E + (1 - k) • F

-- Statement of the problem to prove
theorem minimum_value (h1 : midpoint_eq D)
  (h2 : collinear D AE AF)
  (h3 : AE = lambda • (AB lambda))
  (h4 : AF = mu • (AC mu)) :
  lambda * mu = 1 :=
sorry

end minimum_value_l215_215295


namespace admission_price_for_adults_l215_215087

theorem admission_price_for_adults (A : ℕ) (ticket_price_children : ℕ) (total_children_tickets : ℕ) 
    (total_amount : ℕ) (total_tickets : ℕ) (children_ticket_costs : ℕ) 
    (adult_tickets : ℕ) (adult_ticket_costs : ℕ) :
    ticket_price_children = 5 → 
    total_children_tickets = 21 → 
    total_amount = 201 → 
    total_tickets = 33 → 
    children_ticket_costs = 21 * 5 → 
    adult_tickets = 33 - 21 → 
    adult_ticket_costs = 201 - 21 * 5 → 
    A = (201 - 21 * 5) / (33 - 21) → 
    A = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end admission_price_for_adults_l215_215087


namespace billy_hike_distance_l215_215119

noncomputable def billy_distance : ℝ :=
  real.sqrt (113 + 56 * real.sqrt 2)

theorem billy_hike_distance :
  ∃ d : ℝ, d = billy_distance :=
begin
  use real.sqrt (113 + 56 * real.sqrt 2),
  refl,
end

end billy_hike_distance_l215_215119


namespace factorial_expression_value_l215_215750

theorem factorial_expression_value :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 :=
by
  sorry

end factorial_expression_value_l215_215750


namespace num_selected_from_each_teacher_probability_at_least_one_from_wang_l215_215063

-- Given conditions
def wu_questions : ℕ := 350
def wang_questions : ℕ := 700
def zhang_questions : ℕ := 1050
def total_questions : ℕ := wu_questions + wang_questions + zhang_questions
def sample_size : ℕ := 6
def sampling_ratio : ℚ := sample_size / total_questions
def wu_sample : ℚ := wu_questions * sampling_ratio
def wang_sample : ℚ := wang_questions * sampling_ratio
def zhang_sample : ℚ := zhang_questions * sampling_ratio

-- Prove that the number of selected test questions from Wu, Wang, and Zhang is 1, 2, and 3 respectively.
theorem num_selected_from_each_teacher : 
  wu_sample = 1 ∧ wang_sample = 2 ∧ zhang_sample = 3 := 
sorry

-- Possible combinations selected
def total_combinations : ℕ := 15
def favorable_combinations : ℕ := 9

-- Prove the probability that at least one of the 2 selected questions is from Wang is 3/5.
theorem probability_at_least_one_from_wang : 
  (favorable_combinations / total_combinations : ℚ) = 3/5 := 
sorry

end num_selected_from_each_teacher_probability_at_least_one_from_wang_l215_215063


namespace ratio_of_m_div_x_l215_215427

theorem ratio_of_m_div_x (a b : ℝ) (h1 : a / b = 4 / 5) (h2 : a > 0) (h3 : b > 0) :
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  (m / x) = 2 / 5 :=
by
  -- Define x and m
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  -- Include the steps or assumptions here if necessary
  sorry

end ratio_of_m_div_x_l215_215427


namespace triangle_area_pqr_l215_215445

def point := (ℝ × ℝ)

def line (slope : ℝ) (pt : point) : point := (pt.1 + pt.2 / slope, 0)

def area (p q r : point) : ℝ :=
  (1 / 2) * abs (q.1 - r.1) * p.2

theorem triangle_area_pqr :
  let P := (2, 5)
  let Q := line (-1) P
  let R := line (-2) P
  area P Q R = 6.25 := 
  by
    calc
      area P Q R = 6.25 : sorry

end triangle_area_pqr_l215_215445


namespace horses_meet_at_least_6_simultaneously_l215_215725

theorem horses_meet_at_least_6_simultaneously:
  ∃ T > 0, 
    (∃ S ⊆ {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, 
     S.card = 6 ∧ (∀ k ∈ S, T % k = 0)) ∧ 
    (T.digits.sum = 6) :=
by
  sorry

end horses_meet_at_least_6_simultaneously_l215_215725


namespace ellipse_eccentricity_l215_215910

-- Define the conditions of the problem in terms of the ellipse and its properties
variable (a b c : ℝ)
variable (h1 : a > b) (h2 : b > 0)
variable (h3 : c = real.sqrt(a^2 - b^2))

-- Define points and their relationships
variable (M F1 F2 A : ℝ × ℝ)
variable (h4 : M.1^2 / a^2 + M.2^2 / b^2 = 1)
variable (h5 : F1 = (-c, 0))
variable (h6 : F2 = (c, 0))
variable (h7 : dist M F1 = 2 * c)
variable (h8 : line_through F1 M ∩ (0, A.2) = A)
variable (h9 : angle_bisector (F2, A) (M, F2) F1)

-- Proof goal: The eccentricity of the ellipse is (sqrt(5) - 1) / 2
theorem ellipse_eccentricity : (c / a) = (real.sqrt 5 - 1) / 2 :=
by
  sorry -- Proof skipped

end ellipse_eccentricity_l215_215910


namespace sum_consecutive_odd_numbers_l215_215392

theorem sum_consecutive_odd_numbers :
  ∃ a b c d e : ℤ, 
  (a + b + c + d + e = 275) ∧ 
  (b = a + 2) ∧ 
  (c = a + 4) ∧ 
  (d = a + 6) ∧ 
  (e = a + 8) ∧ 
  a = 51 ∧ b = 53 ∧ c = 55 ∧ d = 57 ∧ e = 59 :=
begin
  sorry
end

end sum_consecutive_odd_numbers_l215_215392


namespace distinct_elements_count_l215_215639

def greatest_integer_leq (x : ℝ) : ℤ :=
  intFloor x

theorem distinct_elements_count :
  let s : Finset ℤ := (Finset.range 2004).image (λ n, greatest_integer_leq ((n : ℝ)^2 / 2003))
  s.card = 1503 := by
  sorry

end distinct_elements_count_l215_215639


namespace lines_through_circumcenter_l215_215210

variables {α : Type*}
variables (A B C M1 M2 : α)
variables [metric_space α] [euclidean_space α]

-- Definitions of ratios
def ratio_AM_BM (M : α) : ℝ := (dist A M) / (dist B M)
def ratio_BM_CM (M : α) : ℝ := (dist B M) / (dist C M)
def ratio_CM_AM (M : α) : ℝ := (dist C M) / (dist A M)

-- Condition: The ratios are the same for both M1 and M2
def same_ratios (M1 M2 : α) : Prop :=
  ratio_AM_BM A B M1 = ratio_AM_BM A B M2 ∧
  ratio_BM_CM B C M1 = ratio_BM_CM B C M2 ∧
  ratio_CM_AM C A M1 = ratio_CM_AM C A M2

-- Theorem: All lines M1 M2 pass through the circumcenter O
theorem lines_through_circumcenter 
  (h : same_ratios A B C M1 M2):
  ∃ O : α, is_circumcenter A B C O ∧ ∀ M1 M2 : α, same_ratios A B C M1 M2 -> lies_on_line O M1 M2 :=
sorry

end lines_through_circumcenter_l215_215210


namespace geom_seq_arith_seq_l215_215203

def is_geom_seq (a : ℕ → ℝ) : Prop :=
∀ n, ∃ r, a (n+1) = r * a n

def sum_geom_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
a 0 * (1 - (a 1 / a 0)^n) / (1 - a 1 / a 0)

def is_arith_seq (b : list ℝ) : Prop :=
∃ d, ∀ (i : ℕ) (h1 : i < b.length - 1), b.nth_le (i+1) h1 = b.nth_le i _ + d

theorem geom_seq_arith_seq
  (a : ℕ → ℝ)
  (S : ℕ → ℝ 
  (h1 : is_geom_seq a)
  (h2 : is_arith_seq [a 2, 2 * a 5, 3 * a 8]) :
  (3 * S 3 / S 6 = 3 / 2 ∨ 3 * S 3 / S 6 = 9 / 4) :=
sorry

end geom_seq_arith_seq_l215_215203


namespace line_perp_through_origin_eqn_l215_215960

variables {a b c : ℝ}

def is_perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y = 0 ∧ l2 x y = 0 → x = 0 ∧ y = 0

def line_through_origin (l : ℝ → ℝ → Prop) : Prop :=
  l 0 0 = 0

theorem line_perp_through_origin_eqn (h : line_through_origin (λ x y, a * x + b * y + c)) :
  ∃ k m, k ≠ 0 ∧ m ≠ 0 ∧ line_through_origin (λ x y, k * y - m * x) ∧ is_perpendicular (λ x y, a * x + b * y + c) (λ x y, k * y - m * x) :=
sorry

end line_perp_through_origin_eqn_l215_215960


namespace increasing_on_0_2pi_l215_215857

def f (x : ℝ) : ℝ := 1 + x - Real.sin x

theorem increasing_on_0_2pi : ∀ x ∈ Set.Ioo 0 (2 * Real.pi), 0 ≤ deriv f x :=
by
  sorry

end increasing_on_0_2pi_l215_215857


namespace problem_statement_l215_215920

-- Define the polar to rectangular coordinate transformation
def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- Define the given polar equation of curve C1
def C1_polar_eq (ρ θ : ℝ) : Prop :=
  ρ ^ 2 = 8 * ρ * sin θ - 15

-- Define the given rectangular equation after transformation
def C1_rectangular_eq (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 - 8 * y + 15 = 0

-- Define the parameterized equation of curve C2
def C2_eq (α : ℝ) : ℝ × ℝ :=
  (2 * sqrt 2 * cos α, sqrt 2 * sin α)

-- The point Q on C2 when α = 3π/4
def Q : ℝ × ℝ := C2_eq (3 * Real.pi / 4)

-- Define the point Q at (3π/4)
def Q_coords : ℝ × ℝ := (-2, 1)

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the center and radius of the circle from C1's rectangular equation
def circle_center : ℝ × ℝ := (0, 4)
def circle_radius : ℝ := 1

-- Define the minimum distance PQ
def PQ_min (P Q : ℝ × ℝ) (radius : ℝ) : ℝ :=
  distance P Q - radius

-- Problem statement: prove the equivalence
theorem problem_statement (ρ θ : ℝ) :
  (C1_polar_eq ρ θ → C1_rectangular_eq (ρ * cos θ) (ρ * sin θ)) ∧
  PQ_min (0, 4) Q_coords circle_radius = sqrt 13 - 1 :=
by
  sorry

end problem_statement_l215_215920


namespace ratio_of_surface_areas_is_sqrt3_l215_215785

noncomputable def cube_side_length : ℝ := 2
noncomputable def tetrahedron_vertices : List (ℝ × ℝ × ℝ) := [(0, 0, 0), (2, 2, 0), (2, 0, 2), (0, 2, 2)]

theorem ratio_of_surface_areas_is_sqrt3 :
  let s := cube_side_length
  let V := tetrahedron_vertices
  let distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)
  let tetrahedron_side_length := distance (V.head!) (V.nth! 1)
  tetrahedron_side_length = 2 * Real.sqrt 2 →
  let tetrahedron_surface_area := Real.sqrt 3 * (tetrahedron_side_length ^ 2)
  let cube_surface_area := 6 * (s ^ 2)
  (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3 := sorry

end ratio_of_surface_areas_is_sqrt3_l215_215785


namespace intersection_A_B_complement_U_A_union_A_complement_U_B_l215_215940

universe u

variable {α : Type u} (x : α)

def U := { x : ℝ | x^2 ≤ 25 }
def A := { x : ℝ | 0 < x ∧ x ≤ 3 }
def B := { x : ℝ | -2 ≤ x ∧ x ≤ 1 }

def complement (S : set ℝ) := { x : ℝ | x ∈ U ∧ x ∉ S }

theorem intersection_A_B :
  { x : ℝ | x ∈ A ∧ x ∈ B } = { x | 0 < x ∧ x ≤ 1 } :=
sorry

theorem complement_U_A :
  complement A = { x | (-5 ≤ x ∧ x ≤ 0) ∨ (3 < x ∧ x ≤ 5) } :=
sorry

theorem union_A_complement_U_B :
  { x : ℝ | x ∈ A ∨ x ∈ complement B } = { x | (-5 ≤ x ∧ x < -2) ∨ (0 < x ∧ x ≤ 5) } :=
sorry

end intersection_A_B_complement_U_A_union_A_complement_U_B_l215_215940


namespace part1_part2_part3_part3_solution_l215_215710

noncomputable def f (x : ℝ) : ℝ := x + 1/x
noncomputable def g (x : ℝ) : ℝ := x - 2 + 1/(x - 4)

theorem part1 :
  g x = 2 - (x + 1/x) ∧
  g (4 - x) = x - 2 + 1/(x - 4) := sorry

theorem part2 (b : ℝ) :
  (b = 0 ∧ (∀ x, g x = 0 → x = 3)) ∨
  (b = 4 ∧ (∀ x, g x = 4 → x = 5)) := sorry

theorem part3 :
  ∀ x, log 3 (g x) < log 3 (9/2) ↔ g x < 9/2 := sorry

theorem part3_solution :
  ∀ x, g x < 9/2 ↔ x ∈ Ioo 4 6 := sorry

end part1_part2_part3_part3_solution_l215_215710


namespace number_of_subsets_of_intersection_l215_215214

open Set

theorem number_of_subsets_of_intersection :
  let M := {0, 1, 2, 3, 4}
  let N := {1, 3, 5}
  let P := M ∩ N
  P = {1, 3} → P.subsets.card = 4 :=
by
  intros h
  rw h
  exact sorry

end number_of_subsets_of_intersection_l215_215214


namespace sum_g_diverges_l215_215186

def g (n : ℕ) : ℝ :=
  ∑' k : ℕ, if k ≥ 3 then (1 : ℝ) / (k : ℝ) ^ n else 0

theorem sum_g_diverges : 
  ∑' n, g n = ∞ := by
  sorry

end sum_g_diverges_l215_215186


namespace least_possible_number_of_straight_lines_l215_215778

theorem least_possible_number_of_straight_lines :
  ∀ (segments : Fin 31 → (Fin 2 → ℝ)), 
  (∀ i j, i ≠ j → (segments i 0 = segments j 0) ∧ (segments i 1 = segments j 1) → false) →
  ∃ (lines_count : ℕ), lines_count = 16 :=
by
  sorry

end least_possible_number_of_straight_lines_l215_215778


namespace frogs_never_larger_square_l215_215665

-- Define the conditions
noncomputable def infinite_grid (l : ℝ) := { p : ℝ × ℝ | ∃ m n : ℤ, p = (m * l, n * l)}

-- Hypothetical initial positions of the frogs
def initial_positions (l : ℝ) : set (ℝ × ℝ) :=
  {(0, 0), (l, 0), (l, l), (0, l)}

-- Move function to simulate one jump
def frog_jump (p q : ℝ × ℝ) : ℝ × ℝ :=
  (2 * q.1 - p.1, 2 * q.2 - p.2)

-- Define a condition where frogs are at vertices of a larger square
def vertices_of_larger_square (l h : ℝ) (p₁ p₂ p₃ p₄ : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 1 ∧
    p₁ = (0, 0) ∧
    p₂ = (h * k, 0) ∧
    p₃ = (h * k, h * k) ∧
    p₄ = (0, h * k)

-- The main theorem statement
theorem frogs_never_larger_square (l : ℝ) :
  ¬ ∃ p₁ p₂ p₃ p₄ : ℝ × ℝ, vertices_of_larger_square l l p₁ p₂ p₃ p₄ :=
by
  sorry

end frogs_never_larger_square_l215_215665


namespace max_sequence_length_l215_215025

-- Define the conditions of the sequence
def sequence_condition (x : ℕ → ℕ) : Prop :=
  (∀ i, x i ≤ 1998) ∧ 
  (∀ i, i ≥ 3 → x i = | x (i - 1) - x (i - 2) |)

-- Statement of the problem
theorem max_sequence_length :
  ∀ (x : ℕ → ℕ), sequence_condition x → ∃ n, n = 2998 :=
by
  sorry

end max_sequence_length_l215_215025


namespace find_pairs_l215_215518

theorem find_pairs (x y : Nat) (h : 1 + x + x^2 + x^3 + x^4 = y^2) : (x, y) = (0, 1) ∨ (x, y) = (3, 11) := by
  sorry

end find_pairs_l215_215518


namespace geometric_series_sum_l215_215130

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l215_215130


namespace geometric_series_sum_l215_215131

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℚ) / 4
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geometric_series_sum_l215_215131


namespace xyz_stock_final_price_l215_215157

theorem xyz_stock_final_price :
  let s0 := 120
  let s1 := s0 + s0 * 1.5
  let s2 := s1 - s1 * 0.3
  let s3 := s2 + s2 * 0.2
  s3 = 252 := by
  sorry

end xyz_stock_final_price_l215_215157


namespace perimeter_of_ABFCDE_is_80_l215_215848

-- Definitions
variables (AB BF FC CD DE EA : ℝ)
variables (square_perimeter BF_length FC_length : ℝ)

-- Conditions
def square_side : ℝ := square_perimeter / 4
def BFC_is_right_triangle : Prop := BF_length = 10 ∧ FC_length = 10

-- Theorem statement
theorem perimeter_of_ABFCDE_is_80
  (h1 : square_perimeter = 60)
  (h2 : BFC_is_right_triangle true)
  (h3 : AB = square_side)
  (h4 : CD = square_side)
  (h5 : DE = square_side)
  (h6 : EA = square_side) :
  AB + BF + FC + CD + DE + EA = 80 :=
by sorry

end perimeter_of_ABFCDE_is_80_l215_215848


namespace mats_weaved_by_mat_weavers_l215_215776

variable (M : ℕ)

theorem mats_weaved_by_mat_weavers :
  -- 10 mat-weavers can weave 25 mats in 10 days
  (10 * 10) * M / (4 * 4) = 25 / (10 / 4)  →
  -- number of mats woven by 4 mat-weavers in 4 days
  M = 4 :=
sorry

end mats_weaved_by_mat_weavers_l215_215776


namespace number_of_1989_periodic_points_l215_215330

noncomputable def number_of_periodic_points 
  (m : ℕ) (hm : m > 1) : ℕ :=
  m ^ 1989 - m ^ 663 - m ^ 153 - m ^ 117 + 
  m ^ 51 + m ^ 39 + m ^ 9 - m ^ 3

theorem number_of_1989_periodic_points (m : ℕ) (hm : m > 1) :
  ∃ (T : ℕ), T = number_of_periodic_points m hm :=
begin
  use (m ^ 1989 - m ^ 663 - m ^ 153 - m ^ 117 + m ^ 51 + m ^ 39 + m ^ 9 - m ^ 3),
  refl,
end

end number_of_1989_periodic_points_l215_215330


namespace find_M_at_x_eq_3_l215_215335

noncomputable def M (a b c d x : ℝ) := a * x^5 + b * x^3 + c * x + d

theorem find_M_at_x_eq_3
  (a b c d M : ℝ)
  (h₀ : d = -5)
  (h₁ : 243 * a + 27 * b + 3 * c = -12) :
  M = -17 :=
by
  sorry

end find_M_at_x_eq_3_l215_215335


namespace probability_purple_between_green_and_twice_l215_215804

noncomputable def prob_purple_greater_green_but_less_than_twice (x_dist : measure ℝ) (y_dist : measure ℝ) : ℝ := 
  ∫⁻ x in Icc (0 : ℝ) 2, ∫⁻ y in Icc (0 : ℝ) 1, 
  if (x < y ∧ y < 2 * x) then 1 else 0 ∂y_dist ∂x_dist

theorem probability_purple_between_green_and_twice :
  let x_dist := measure.Uniform (Icc (0 : ℝ) 2),
      y_dist := measure.Uniform (Icc (0 : ℝ) 1) in
  prob_purple_greater_green_but_less_than_twice x_dist y_dist = 1 / 8 :=
by
  sorry

end probability_purple_between_green_and_twice_l215_215804


namespace min_k_valid_l215_215240

def S : Set ℕ := {1, 2, 3, 4}

def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ b : Fin 4 → ℕ,
    (∀ i : Fin 4, b i ∈ S) ∧ b 3 ≠ 1 →
    ∃ i1 i2 i3 i4 : Fin (k + 1), i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧
      (a i1 = b 0 ∧ a i2 = b 1 ∧ a i3 = b 2 ∧ a i4 = b 3)

def min_k := 11

theorem min_k_valid : ∀ a : ℕ → ℕ,
  valid_sequence a min_k → 
  min_k = 11 :=
sorry

end min_k_valid_l215_215240


namespace centers_of_rectangles_lie_on_segment_l215_215909

-- Definition of the triangle and its sides
def Triangle (α β γ : Type) := α × β × γ

-- Definition of points on a side of the triangle
def on_side (p q r : Type) := ℝ → p × q × r

-- Definitions for midpoints, center of rectangles
noncomputable def midpoint (A B : Type) := sorry -- precise formalization of midpoints
noncomputable def center_of_rectangle (PQRS : Type) := sorry -- precise formalization of center

-- Problem statement
theorem centers_of_rectangles_lie_on_segment 
  (A B C P Q R S O M : Type) 
  (h1 : on_side Q P AC)
  (h2 : on_side R AB)
  (h3 : on_side S BC)
  (h4 : O = midpoint B (foot_of_perpendicular B AC))
  (h5 : M = midpoint A C)
  (h6 : ∀PQRS, P Q R S center_of_rectangle PQRS): 
  set_of (center_of_rectangle PQRS) = segment_excluding_ends O M := 
sorry

end centers_of_rectangles_lie_on_segment_l215_215909


namespace mode_of_data_set_l215_215549

theorem mode_of_data_set :
  ∃ x y : ℕ,
    let data := [2, 5, x, y, 2 * x, 11] in
    data.sorted ∧
    (data.sum / 6 = 7) ∧
    (data.drop 2).take 2.avg = 7 ∧
    data.mode = 5 :=
begin
  sorry
end

end mode_of_data_set_l215_215549


namespace expressions_opposite_sign_l215_215036

theorem expressions_opposite_sign : 
  let x := (7 / 5 : ℚ) in 
  (x - 1) / 2 * (x - 2) / 3 < 0 := 
by
  let x := (7 / 5 : ℚ)
  have h₀ : (x - 1) / 2 * (x - 2) / 3 = (3 / 5) / 2 * (2 / 5) / 3 := by
    field_simp
  exact sorry

end expressions_opposite_sign_l215_215036


namespace sum_of_factors_of_30_multiplied_by_2_equals_144_l215_215034

-- We define the factors of 30
def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- We define the function to multiply each factor by 2 and sum them
def sum_factors_multiplied_by_2 (factors : List ℕ) : ℕ :=
  factors.foldl (λ acc x => acc + 2 * x) 0

-- The final statement to be proven
theorem sum_of_factors_of_30_multiplied_by_2_equals_144 :
  sum_factors_multiplied_by_2 factors_of_30 = 144 :=
by sorry

end sum_of_factors_of_30_multiplied_by_2_equals_144_l215_215034


namespace _l215_215968

noncomputable def a : ℝ := 1 -- placeholder definition
noncomputable def b : ℝ := 1 -- placeholder definition
noncomputable def c : ℝ := 1 -- placeholder definition

noncomputable theorem sum_of_a_b_c_is_correct
  (h₁ : a^2 + b^2 + c^2 = 48)
  (h₂ : a * b + b * c + c * a = 24)
  (h₃ : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) :
  a + b + c = 4 * real.sqrt 6 := 
sorry

end _l215_215968


namespace lena_and_ira_actual_weight_l215_215310

-- Define the variables and conditions
variables (u x y : ℝ) 

-- Conditions given in the problem
def lena_weighs : Prop := x + y = 2 + u
def ira_weighs : Prop := 2 * x + 2 * y = 3 + u
def combined_weighs : Prop := x + y = 4.5

-- Main statement we want to prove
theorem lena_and_ira_actual_weight (hc : combined_weighs ∧ lena_weighs ∧ ira_weighs) :
  (x = 1.5 ∧ y = 2.5) := 
begin
  sorry
end

end lena_and_ira_actual_weight_l215_215310


namespace scallops_per_person_l215_215059

theorem scallops_per_person 
    (scallops_per_pound : ℕ)
    (cost_per_pound : ℝ)
    (total_cost : ℝ)
    (people : ℕ)
    (total_pounds : ℝ)
    (total_scallops : ℕ)
    (scallops_per_person : ℕ)
    (h1 : scallops_per_pound = 8)
    (h2 : cost_per_pound = 24)
    (h3 : total_cost = 48)
    (h4 : people = 8)
    (h5 : total_pounds = total_cost / cost_per_pound)
    (h6 : total_scallops = scallops_per_pound * total_pounds)
    (h7 : scallops_per_person = total_scallops / people) : 
    scallops_per_person = 2 := 
by {
    sorry
}

end scallops_per_person_l215_215059


namespace present_age_of_son_l215_215041

theorem present_age_of_son (S F : ℕ) (h1 : F = S + 25) (h2 : F + 2 = 2 * (S + 2)) : S = 23 :=
by
  sorry

end present_age_of_son_l215_215041


namespace cross_product_zero_l215_215523

open Matrix

def a : ℝ ^ 3 :=
  ![3, -1, 4]

def b : ℝ ^ 3 :=
  ![6, -2, 8]

def cross_product (u v : ℝ ^ 3) : ℝ ^ 3 :=
  ![
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]
  ]

theorem cross_product_zero : cross_product a b = ![0, 0, 0] :=
by
  -- Proof goes here
  sorry

end cross_product_zero_l215_215523


namespace locus_of_orthocenter_is_ellipse_l215_215083

noncomputable def locus_of_orthocenter (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) :=
  -- ellipses conditions
  ∀ (C : ℝ × ℝ), ( (fst(C))^2 / a^2 + (snd(C))^2 / b^2 = 1 ) → 
  ∃ (M  : ℝ × ℝ), 
    (fst(M))^2 / a^2 + (snd(M))^2 / (a^2/b)^2 = 1

theorem locus_of_orthocenter_is_ellipse (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) :
  locus_of_orthocenter a b h_pos :=
by sorry

end locus_of_orthocenter_is_ellipse_l215_215083


namespace value_of_expression_l215_215958

theorem value_of_expression (a b : ℤ) (h : 2 * a - b = 10) : 2023 - 2 * a + b = 2013 :=
by
  sorry

end value_of_expression_l215_215958


namespace jessica_shoveling_hours_l215_215996

noncomputable def snow_shoveling_hours (initial_rate : ℕ) (decrement : ℕ) (total_volume : ℕ) : ℕ :=
nat.find (λ n, (list.range n).scanl (λ acc i, acc - (initial_rate - decrement * i)) total_volume |>.last' = some 0)

theorem jessica_shoveling_hours : snow_shoveling_hours 25 2 150 = 9 := 
by {
  sorry
}

end jessica_shoveling_hours_l215_215996


namespace sum_of_two_digit_integers_with_squares_ending_in_06_l215_215410

theorem sum_of_two_digit_integers_with_squares_ending_in_06 :
  let nums := [n | n ∈ ([10 * a + b | a ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9], b ∈ [4, 6]], 
                             let sqr := n * n in (sqr % 100) = 6)]
  let unique_nums := nums.erase_dup
  (unique_nums.sum = 176) :=
by {
  sorry
}

end sum_of_two_digit_integers_with_squares_ending_in_06_l215_215410


namespace highway_length_l215_215013

theorem highway_length 
  (speed_car1 speed_car2 : ℕ) (time : ℕ)
  (h_speed_car1 : speed_car1 = 54)
  (h_speed_car2 : speed_car2 = 57)
  (h_time : time = 3) : 
  speed_car1 * time + speed_car2 * time = 333 := by
  sorry

end highway_length_l215_215013


namespace gail_fish_difference_l215_215892

structure TankInfo where
  size_first_tank : ℕ -- gallons
  size_second_tank : ℕ -- gallons
  size_first_fish : ℕ -- inches
  size_second_fish : ℕ -- inches

def gail_calculate (info : TankInfo) : ℕ :=
  let fish_first_tank := info.size_first_tank / info.size_first_fish
  let fish_second_tank := info.size_second_tank / info.size_second_fish
  fish_first_tank - 1 - fish_second_tank

theorem gail_fish_difference : 
  ∀ (info : TankInfo),
  info.size_first_tank = 48 →
  info.size_first_fish = 3 →
  info.size_second_fish = 2 →
  info.size_second_tank = info.size_first_tank / 2 →
  gail_calculate info = 3 :=
by
  intros info h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  dsimp [gail_calculate]
  norm_num
  sorry

end gail_fish_difference_l215_215892


namespace sum_and_product_of_roots_l215_215032

-- Define the polynomial equation and the conditions on the roots
def cubic_eqn (x : ℝ) : Prop := 3 * x ^ 3 - 18 * x ^ 2 + 27 * x - 6 = 0

-- The Lean statement for the given problem
theorem sum_and_product_of_roots (p q r : ℝ) :
  cubic_eqn p ∧ cubic_eqn q ∧ cubic_eqn r →
  (p + q + r = 6) ∧ (p * q * r = 2) :=
by
  sorry

end sum_and_product_of_roots_l215_215032


namespace evaluate_exponent_sum_l215_215861

theorem evaluate_exponent_sum : 
  let i : ℂ := Complex.I in 
  i^14760 + i^14761 + i^14762 + i^14763 = 0 := by
  sorry

end evaluate_exponent_sum_l215_215861


namespace maximum_students_per_dentist_l215_215395

theorem maximum_students_per_dentist (dentists students : ℕ) (min_students : ℕ) (attended_students : ℕ)
  (h_dentists : dentists = 12)
  (h_students : students = 29)
  (h_min_students : min_students = 2)
  (h_total_students : attended_students = students) :
  ∃ max_students, 
    (∀ d, d < dentists → min_students ≤ attended_students / dentists) ∧
    (∀ d, d < dentists → attended_students = students - (dentists * min_students) + min_students) ∧
    max_students = 7 :=
by
  sorry

end maximum_students_per_dentist_l215_215395


namespace complement_correct_l215_215319

-- Given conditions
def U := {1, 2, 3, 4, 5}
def M := {1, 3, 5}

-- Define the complement in terms of set difference
def complement_U_M := U \ M

-- The theorem to prove: the complement of M with respect to U is equal to {2, 4}
theorem complement_correct : complement_U_M = {2, 4} :=
by 
  sorry

end complement_correct_l215_215319


namespace angle_y_80_degree_l215_215617

def E : Prop := sorry  -- Placeholder for edge E
def F : Prop := sorry  -- Placeholder for edge F
def G : Prop := sorry  -- Placeholder for edge G
def H : Prop := sorry  -- Placeholder for edge H

def EF_straight : Prop := sorry  -- EF is a straight line
def GH_straight : Prop := sorry  -- GH is a straight line

def angle_EPF : Prop := (∃ EPF_angle: ℝ, EPF_angle = 180)  -- \angle EPF = 180°
def angle_QPR : Prop := (70 + 40 + ∃ QPR_angle: ℝ,  QPR_angle = 70)  -- \angle QPR = 70°
def angle_QRP : Prop := (∃ QRP_angle: ℝ, QRP_angle = 30)  -- \angle QRP = 30°
def triangle_sum_QPR : Prop := (∃ a b c QPR_sum: ℝ, QPR_sum = 180 ∧ a + b + c = 180) -- angles in ∆QPR add to 180°

theorem angle_y_80_degree (EF_straight GH_straight angle_EPF angle_QPR angle_QRP triangle_sum_QPR) :
  ∃ y: ℝ, y = 80 :=
sorry

end angle_y_80_degree_l215_215617


namespace evaluate_powers_of_i_l215_215864

-- Definitions based on the given conditions
noncomputable def i_power (n : ℤ) := 
  if n % 4 = 0 then (1 : ℂ)
  else if n % 4 = 1 then complex.I
  else if n % 4 = 2 then -1
  else -complex.I

-- Statement of the problem
theorem evaluate_powers_of_i :
  i_power 14760 + i_power 14761 + i_power 14762 + i_power 14763 = 0 :=
by
  sorry

end evaluate_powers_of_i_l215_215864


namespace triangle_problems_l215_215623

-- Define the pieces we need
variables {A B C : ℝ} {a b c : ℝ}

-- Condition 1: Definition of sides opposite angles in triangle ΔABC
-- Condition 2: Given equation involving sides and angles
def condition2 (A B C a b c : ℝ) : Prop := (2 * b - Real.sqrt 3 * c) * Real.cos A = Real.sqrt 3 * a * Real.cos C

-- Condition 3: Given |AB - AC| = 2√2
def condition3 (AB AC : ℝ) : Prop := Real.abs (AB - AC) = 2 * Real.sqrt 2

-- We need to show:
def problem1 (A : ℝ) : Prop := A = Real.pi / 6

def problem2 (a b c : ℝ) : Prop := 
  let bc := b * c in 
  let area := (bc * (1 / 2)) / 2 in 
  area ≤ 4 + 2 * Real.sqrt 3

-- Main theorem statement combining conditions and required proofs
theorem triangle_problems
  (A B C a b c AB AC : ℝ)
  (h1 : condition2 A B C a b c)
  (h2 : condition3 AB AC) : problem1 A ∧ problem2 a b c :=
by sorry

end triangle_problems_l215_215623


namespace mark_total_spending_l215_215345

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end mark_total_spending_l215_215345


namespace sin_721_eq_sin_1_l215_215140

theorem sin_721_eq_sin_1 : Real.sin (721 * Real.pi / 180) = Real.sin (1 * Real.pi / 180) := 
by
  sorry

end sin_721_eq_sin_1_l215_215140


namespace roots_of_g_l215_215223

theorem roots_of_g
  (a b : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_f : f = λ x, a * x + b)
  (h_g : g = λ x, b * x^2 - a * x)
  (root_f : f 2 = 0) :
  (g 0 = 0) ∧ (g (-1/2) = 0) :=
by
  sorry

end roots_of_g_l215_215223


namespace intersection_product_l215_215232

noncomputable def point := (ℝ × ℝ)

noncomputable def curve (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

noncomputable def line (x y t : ℝ) : Prop :=
  x = 1 + 1/2 * t ∧ y = 2 + sqrt 3 / 2 * t

noncomputable def point_on_curve (p : point) : Prop :=
  curve p.1 p.2

noncomputable def point_on_line (p : point) (t : ℝ) : Prop :=
  line p.1 p.2 t

noncomputable def intersection_points (A B : point) (M : point) : Prop :=
  ∃ t₁ t₂ : ℝ, point_on_line A t₁ ∧ point_on_curve A ∧ point_on_line B t₂ ∧ point_on_curve B ∧
  (A.1 - M.1)^2 + (A.2 - M.2)^2 = t₁^2 ∧
  (B.1 - M.1)^2 + (B.2 - M.2)^2 = t₂^2

theorem intersection_product
    (A B M : point)
    (hM : M = (1, 2))
    (hIntersection : intersection_points A B M) :
    (A.1 - M.1)^2 + (A.2 - M.2)^2 * (B.1 - M.1)^2 + (B.2 - M.2)^2 = 28 / 15 :=
sorry

end intersection_product_l215_215232


namespace tree_count_l215_215401

theorem tree_count (m N : ℕ) 
  (h1 : 12 ≡ (33 - m) [MOD N])
  (h2 : (105 - m) ≡ 8 [MOD N]) :
  N = 76 := 
sorry

end tree_count_l215_215401


namespace part1_part2_part3_l215_215902

open Complex

noncomputable def z := (-1 - 2*I)

theorem part1 : (1 + I) * z = 1 - 3 * I → 
  z.im = -2 := by
  intros h
  unfold z
  simp

theorem part2 (a : ℝ) : (1 + a * I) * z = (0 : ℂ) → 
  a = 1/2 := by
  intros h
  unfold z at h
  simp at h
  sorry

theorem part3 :  abs ((conj z) / (z + 1)) = (sqrt 5) / 2 := by
  unfold z
  simp
  sorry

end part1_part2_part3_l215_215902


namespace bread_cooling_time_l215_215994

theorem bread_cooling_time 
  (dough_room_temp : ℕ := 60)   -- 1 hour in minutes
  (shape_dough : ℕ := 15)       -- 15 minutes
  (proof_dough : ℕ := 120)      -- 2 hours in minutes
  (bake_bread : ℕ := 30)        -- 30 minutes
  (start_time : ℕ := 2 * 60)    -- 2:00 am in minutes
  (end_time : ℕ := 6 * 60)      -- 6:00 am in minutes
  : (end_time - start_time) - (dough_room_temp + shape_dough + proof_dough + bake_bread) = 15 := 
  by
  sorry

end bread_cooling_time_l215_215994


namespace sin_expression_equals_zero_l215_215120

theorem sin_expression_equals_zero :
  (sin (75 * Real.pi / 180) * sin (165 * Real.pi / 180) - sin (15 * Real.pi / 180) * sin (105 * Real.pi / 180)) = 0 :=
by
  have h1 : sin (165 * Real.pi / 180) = sin (15 * Real.pi / 180), by sorry
  have h2 : sin (105 * Real.pi / 180) = sin (75 * Real.pi / 180), by sorry
  sorry

end sin_expression_equals_zero_l215_215120


namespace total_toys_l215_215474

theorem total_toys (m a t : ℕ) (h1 : a = m + 3 * m) (h2 : t = a + 2) (h3 : m = 6) : m + a + t = 56 := by
  sorry

end total_toys_l215_215474


namespace solve_trig_eq_l215_215686

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end solve_trig_eq_l215_215686


namespace cube_volume_l215_215786

theorem cube_volume (s : ℝ) (h : s ^ 2 = 64) : s ^ 3 = 512 :=
sorry

end cube_volume_l215_215786


namespace range_of_k_l215_215546

theorem range_of_k (k : ℝ) : (-3 ≤ k ∧ k ≤ 2 ∧ k ≠ 0) ↔ (∀ x : ℝ, x > -1 → (kx - 2 < 2x + 3)) :=
by {
  sorry
}

end range_of_k_l215_215546


namespace binary_to_base6_l215_215500

theorem binary_to_base6 
  (n_binary : ℕ) 
  (h : n_binary = 0b1011001)
  : n_binary.to_nat.to_base 6 = "225" := 
  sorry

end binary_to_base6_l215_215500


namespace abs_neg2023_eq_2023_l215_215372

-- Define a function to represent the absolute value
def abs (x : ℤ) : ℤ :=
  if x < 0 then -x else x

-- Prove that abs (-2023) = 2023
theorem abs_neg2023_eq_2023 : abs (-2023) = 2023 :=
by
  -- In this theorem, all necessary definitions are already included
  sorry

end abs_neg2023_eq_2023_l215_215372


namespace union_sets_bounds_l215_215650

open Set

theorem union_sets_bounds (n : ℕ) (hn : n ≥ 2)
  (A : Fin n → Set α)
  (hA : ∀ i, (A i).card = n)
  (h_inter : ∀ (k : ℕ) (Hk : 2 ≤ k ∧ k ≤ n - 1)
             (s : Finset (Fin n)), s.card = k → (s.val.to_finset).inter_size ≥ n + 1 - k):
  let S := (A 0 ∪ A 1 ∪ ⋯ ∪ A (n - 1)).card in
  (n + 1 ≤ S) ∧ (S ≤ 2 * n - 1) :=
sorry

end union_sets_bounds_l215_215650


namespace paint_ratio_l215_215195

theorem paint_ratio
  (blue yellow white : ℕ)
  (ratio_b : ℕ := 4)
  (ratio_y : ℕ := 3)
  (ratio_w : ℕ := 5)
  (total_white : ℕ := 15)
  : yellow = 9 := by
  have ratio := ratio_b + ratio_y + ratio_w
  have white_parts := total_white * ratio_w / ratio_w
  have yellow_parts := white_parts * ratio_y / ratio_w
  exact sorry

end paint_ratio_l215_215195


namespace sum_two_digit_integers_ending_in_06_l215_215413

theorem sum_two_digit_integers_ending_in_06 :
  let ns := [n | n ∈ finset.range (100) ∧ (n ≥ 10) ∧ (n^2 % 100 =  6)] in
  ns.sum = 166 :=
by
  let ns := ([n | n ∈ finset.range (100) ∧ (n ≥ 10) ∧ (n^2 % 100 = 6)] : list ℕ)
  have ns_val : ns = [14, 66, 86] := by sorry
  have sum_val : ns.sum = 166 := by sorry
  exact sum_val

end sum_two_digit_integers_ending_in_06_l215_215413


namespace find_triangle_with_properties_l215_215522

-- Define the angles forming an arithmetic progression
def angles_arithmetic_progression (α β γ : ℝ) : Prop :=
  β - α = γ - β

-- Define the sides forming an arithmetic progression
def sides_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

-- Define the sides forming a geometric progression
def sides_geometric_progression (a b c : ℝ) : Prop :=
  b^2 = a * c

-- Define the sum of angles in a triangle
def sum_of_angles (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- The problem statement:
theorem find_triangle_with_properties 
    (α β γ a b c : ℝ)
    (h1 : angles_arithmetic_progression α β γ)
    (h2 : sum_of_angles α β γ)
    (h3 : sides_arithmetic_progression a b c ∨ sides_geometric_progression a b c) :
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by 
  sorry

end find_triangle_with_properties_l215_215522


namespace problem_statement_l215_215784

noncomputable theory
open_locale classical

def is_knight (p : Prop) : Prop := p
def is_liar (p : Prop) : Prop := ¬p

variables (A B C : Prop)
variables (A_unclear : Prop) (B_claims_A_knight : Prop) (C_claims_A_liar : Prop)

theorem problem_statement
  (h1 : is_knight B_claims_A_knight = B)
  (h2 : is_liar C_claims_A_liar = C)
  (h3 : is_knight A ∧ ¬is_liar A)
  (h4 : B_claims_A_knight = A)
  (h5 : C_claims_A_liar = ¬A)
  : is_knight B ∧ is_liar C :=
sorry

end problem_statement_l215_215784


namespace line_intersects_ellipse_two_points_l215_215226

theorem line_intersects_ellipse_two_points {m n : ℝ} (h1 : ¬∃ x y : ℝ, m*x + n*y = 4 ∧ x^2 + y^2 = 4)
  (h2 : m^2 + n^2 < 4) : 
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ (m * p1.1 + n * p1.2 = 4) ∧ (m * p2.1 + n * p2.2 = 4) ∧ 
  (p1.1^2 / 9 + p1.2^2 / 4 = 1) ∧ (p2.1^2 / 9 + p2.2^2 / 4 = 1) :=
sorry

end line_intersects_ellipse_two_points_l215_215226


namespace quadratic_roots_identity_l215_215221

-- Define constants for the quadratic equation
def a : ℝ := 1
def b : ℝ := -1
def c : ℝ := -2022

-- Vieta's formulas applied to the quadratic equation
def sum_roots : ℝ := -b / a
def product_roots : ℝ := c / a

theorem quadratic_roots_identity :
  let x₁ := sum_roots
  let x₂ := product_roots in
  x₁ + x₂ - x₁ * x₂ = 2023 :=
by
  sorry

end quadratic_roots_identity_l215_215221


namespace ceil_floor_eq_zero_implies_sum_l215_215952

theorem ceil_floor_eq_zero_implies_sum (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ + ⌊x⌋ = 2 * x :=
by
  sorry

end ceil_floor_eq_zero_implies_sum_l215_215952


namespace solve_trig_eq_l215_215687

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end solve_trig_eq_l215_215687


namespace factorial_division_l215_215748

theorem factorial_division :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 := by
  sorry

end factorial_division_l215_215748


namespace parabola_with_vertex_origin_and_directrix_neg_one_has_eq_l215_215270

noncomputable def parabola_directrix_vertex_equation : Prop :=
  let vertex : (ℝ × ℝ) := (0, 0) in
  let directrix : ℝ → Prop := λ x, x = -1 in
  ∀ p : ℝ, 
    1 = p / 2 →
    ∃ y x : ℝ, 
      y^2 = 4 * x

theorem parabola_with_vertex_origin_and_directrix_neg_one_has_eq :
  parabola_directrix_vertex_equation :=
by
  sorry

end parabola_with_vertex_origin_and_directrix_neg_one_has_eq_l215_215270


namespace range_of_a_l215_215967

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3) ∧ (x - a > 0)) ↔ (a ≤ -1) :=
sorry

end range_of_a_l215_215967


namespace jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l215_215891

theorem jia_can_formulate_quadratic :
  ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem yi_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem bing_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 2 ∧ x₂ % 3 = 2 ∧ p % 3 = 1 ∧ q % 3 = 1 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

theorem ding_cannot_formulate_quadratic :
  ¬ ∃ (p q x₁ x₂ : ℤ), 
    x₁ % 3 = 1 ∧ x₂ % 3 = 1 ∧ p % 3 = 2 ∧ q % 3 = 2 ∧ 
    (p = -(x₁ + x₂)) ∧ (q = x₁ * x₂) :=
  sorry

end jia_can_formulate_quadratic_yi_cannot_formulate_quadratic_bing_cannot_formulate_quadratic_ding_cannot_formulate_quadratic_l215_215891


namespace snow_clearance_volume_l215_215787

-- Define conditions
def driveway_length : ℝ := 30 -- in feet
def driveway_width : ℝ := 3 -- in feet
def snow_depth : ℝ := 0.5 -- in feet (converted from 6 inches)
def melt_factor : ℝ := 0.9 -- 10% melted, so 90% remains
def cubic_feet_to_cubic_yards : ℝ := 1 / 27 -- conversion factor

-- Define total volume of snow before melting in cubic feet
def initial_snow_volume : ℝ := driveway_length * driveway_width * snow_depth

-- Define the volume of snow after melting in cubic feet
def effective_snow_volume_cubic_feet : ℝ := melt_factor * initial_snow_volume

-- Convert the volume from cubic feet to cubic yards
def effective_snow_volume_cubic_yards : ℝ := effective_snow_volume_cubic_feet * cubic_feet_to_cubic_yards

-- Main theorem stating the above conditions lead to the required proof
theorem snow_clearance_volume : effective_snow_volume_cubic_yards = 1.5 :=
by
  unfold initial_snow_volume effective_snow_volume_cubic_feet effective_snow_volume_cubic_yards
  sorry

end snow_clearance_volume_l215_215787


namespace mass_percentage_S_in_Al2S3_l215_215174

-- Variables
def molar_mass_Al := 26.98 -- g/mol
def molar_mass_S := 32.06 -- g/mol
def Al2S3_Al_atoms := 2
def Al2S3_S_atoms := 3
def molar_mass_Al2S3 := (Al2S3_Al_atoms * molar_mass_Al) + (Al2S3_S_atoms * molar_mass_S) -- g/mol

-- Theorem for mass percentage of S in aluminum sulfide
theorem mass_percentage_S_in_Al2S3 : 
  (Al2S3_S_atoms * molar_mass_S / molar_mass_Al2S3) * 100 = 64.07 :=
by
  sorry

end mass_percentage_S_in_Al2S3_l215_215174


namespace sin_sq_seq_not_converge_to_zero_l215_215355

theorem sin_sq_seq_not_converge_to_zero :
  ¬(∀ (ε : ℝ), ε > 0 → ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → abs (sin (n ^ 2)) < ε) :=
by
  sorry

end sin_sq_seq_not_converge_to_zero_l215_215355


namespace conjugate_in_second_quadrant_l215_215566

-- Define the given complex number z
noncomputable def z : ℂ := -2 * complex.I + (3 - complex.I) / complex.I

-- Define the conjugate of z
noncomputable def z_conjugate : ℂ := complex.conj z

-- Determine the quadrant for the conjugate complex number
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The theorem statement
theorem conjugate_in_second_quadrant : in_second_quadrant z_conjugate :=
  by sorry

end conjugate_in_second_quadrant_l215_215566


namespace probability_of_at_least_6_consecutive_heads_l215_215789

-- Represent the coin flip outcomes and the event of interest
inductive CoinFlip
| H
| T

def all_flips : List (List CoinFlip) :=
  let base := [CoinFlip.H, CoinFlip.T]
  base.product (base.product (base.product (base.product (base.product (base.product (base.product base))))))

def consecutive_heads (n : Nat) (flips : List CoinFlip) : Bool :=
  List.any (List.tails flips) (λ l => l.take n = List.repeat CoinFlip.H n)

def atLeast6ConsecutiveHeads (flips : List CoinFlip) : Prop :=
  consecutive_heads 6 flips || consecutive_heads 7 flips || consecutive_heads 8 flips

def probabilityAtLeast6ConsecutiveHeads (flipsList : List (List CoinFlip)) : ℚ :=
  (flipsList.filter atLeast6ConsecutiveHeads).length / flipsList.length

-- The proof statement
theorem probability_of_at_least_6_consecutive_heads :
  probabilityAtLeast6ConsecutiveHeads all_flips = 13 / 256 :=
by 
  sorry

end probability_of_at_least_6_consecutive_heads_l215_215789


namespace hyperbola_eccentricity_correct_l215_215557

-- Definitions based on provided conditions
def hyperbola (a b : ℝ) := ∀ (x y : ℝ), (a > 0) ∧ (b > 0) → (x^2 / a^2) - (y^2 / b^2) = 1

def midPoint_on_hyperbola (F1 F2 : ℝ × ℝ) (M : ℝ × ℝ) (a b : ℝ) (c : ℝ) : Prop :=
  let midpoint := (-(c / 2), c / 2) in
  hyperbola a b (midpoint.1) (midpoint.2)

noncomputable def eccentricity_is (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2) in
  (Real.sqrt (10) + Real.sqrt (2)) / 2

theorem hyperbola_eccentricity_correct (a b : ℝ) (e : ℝ) (F1 F2 : ℝ × ℝ) (M : ℝ × ℝ) (c : ℝ) :
  hyperbola a b F1.1 F1.2 →
  hyperbola a b F2.1 F2.2 →
  midPoint_on_hyperbola F1 F2 M a b c →
  e = eccentricity_is a b :=
by
  sorry

end hyperbola_eccentricity_correct_l215_215557


namespace possible_values_of_a_plus_b_l215_215217

/-- Given that a and b are constants and that the sum of the three monomials
    4xy², axy^b, -5xy is still a monomial -/
theorem possible_values_of_a_plus_b (a b : ℤ) (h : ∃ (F : ℝ → ℝ → ℝ), 
    ∀ (x y : ℝ), F (4 * x * y^2) (a * x * y^b + -5 * x * y) = 
    f (4 * x * y^2 + a * x * y^b + -5 * x * y)) :
    a + b = -2 ∨ a + b = 6 := sorry

end possible_values_of_a_plus_b_l215_215217


namespace number_of_H_atoms_in_compound_l215_215440

theorem number_of_H_atoms_in_compound :
  ∀ (Ca O H : ℝ),
  Ca = 40.08 → O = 16.00 → H = 1.008 →
  ∀ (num_Ca num_O : ℕ),
  num_Ca = 1 → num_O = 2 →
  ∀ (molecular_weight : ℝ),
  molecular_weight = 74 →
  ∃ (num_H_atoms : ℕ),
  num_H_atoms = 2 :=
begin
  intros Ca O H,
  intros hCa hO hH,
  intros num_Ca num_O,
  intros hnum_Ca hnum_O,
  intros molecular_weight,
  intros h_molecular_weight,
  let weightCa := num_Ca * Ca,
  let weightO := num_O * O,
  let known_weight := weightCa + weightO,
  let weightH := molecular_weight - known_weight,
  have h_weightCa : weightCa = 40.08 := by simp [hCa, hnum_Ca],
  have h_weightO : weightO = 2 * 16 := by simp [hO, hnum_O],
  have h_known_weight : known_weight = 72.08 := by simp [hCa, hO, hnum_Ca, hnum_O, h_weightCa, h_weightO],
  have h_weightH : weightH = 1.92 := by simp [h_molecular_weight, h_known_weight],
  let num_H_atoms := weightH / H,
  have h_num_H_atoms : num_H_atoms ≈ 1.92 / 1.008 := by simp [hH, h_weightH],
  have h_approx_H_atoms : num_H_atoms ≈ 1.90476 := by norm_num,
  existsi 2,
  have : num_H_atoms ≈ 2,
  sorry
end

end number_of_H_atoms_in_compound_l215_215440


namespace valid_lineups_count_l215_215368

-- Define the conditions
def total_players : ℕ := 15
def max : ℕ := 1
def rex : ℕ := 1
def tex : ℕ := 1

-- Proving the number of valid lineups
theorem valid_lineups_count :
  ∃ n, n = 5 ∧ total_players = 15 ∧ max + rex + tex ≤ 1 → n = 2277 :=
sorry

end valid_lineups_count_l215_215368


namespace minimum_small_pipes_needed_l215_215449

-- Definitions based on conditions
def volume_of_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

def diameter_large_pipe : ℝ := 12
def diameter_small_pipe : ℝ := 3
def radius_large_pipe : ℝ := diameter_large_pipe / 2
def radius_small_pipe : ℝ := diameter_small_pipe / 2

-- Proof statement
theorem minimum_small_pipes_needed (h : ℝ) : 16 * volume_of_cylinder radius_small_pipe h = volume_of_cylinder radius_large_pipe h :=
by
  sorry

end minimum_small_pipes_needed_l215_215449


namespace music_collections_l215_215470

open Finset

variables {A J M : Finset ℕ}

theorem music_collections :
  (|A ∩ J ∩ M| = 12) ∧
  (|A| = 25) ∧
  (|J \ (A ∪ M)| = 8) ∧
  (|M \ (A ∪ J)| = 5) →
  (|A \ (J ∪ M)| + |J \ (A ∪ M)| + |M \ (A ∪ J)| = 26) :=
begin
  sorry,
end

end music_collections_l215_215470


namespace semerka_connected_l215_215983

open Combinatorics

def semerka_graph : SimpleGraph (Fin 15) := sorry

theorem semerka_connected (G : SimpleGraph (Fin 15)) 
  (H : ∀ v : (Fin 15), 7 ≤ (G.degree v)) : G.IsConnected :=
begin
  -- proof steps here
  sorry
end

end semerka_connected_l215_215983


namespace alternating_even_binomial_sum_l215_215178

theorem alternating_even_binomial_sum :
  (∑ k in finset.Ico 0 51, (-1)^k * nat.choose 101 (2*k)) = -2^50 :=
by
  sorry

end alternating_even_binomial_sum_l215_215178


namespace monotonic_intervals_l215_215236

noncomputable def f (a x : ℝ) : ℝ := x^2 - (2*a + 1)*x + a*real.log x

theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (0 < a ∧ a < 1 → 
    (∀ x, (0 < x ∧ x < a) → f a x > 0) ∧ 
    (∀ x, (1 < x) → f a x > 0) ∧ 
    (∀ x, (a < x ∧ x < 1) → f a x < 0)) ∧ 
  (a = 1 → ∀ x, (0 < x) → f a x > 0) ∧ 
  (a > 1 → 
    (∀ x, (0 < x ∧ x < 1) → f a x > 0) ∧ 
    (∀ x, (a < x) → f a x > 0) ∧
    (∀ x, (1 < x ∧ x < a) → f a x < 0)) :=
by sorry

end monotonic_intervals_l215_215236


namespace dog_walk_area_l215_215768

-- Define the problem
def hexagonSideLength : ℝ := 1
def leashLength : ℝ := 3

-- Prove that the area the dog can walk on is equal to \(\frac{23}{3} \pi\)
theorem dog_walk_area (hex_s : ℝ) (leash_l : ℝ) (h1 : hex_s = 1) (h2 : leash_l = 3) :
  (∃ area : ℝ, area = (23 / 3) * Real.pi) :=
by
  use (23 / 3) * Real.pi
  sorry

end dog_walk_area_l215_215768


namespace abs_diff_roots_quadratic_eq_l215_215936

theorem abs_diff_roots_quadratic_eq : 
  let r := (8: ℝ) in
  let a := 1 in
  let b := -6 in
  let c := 8 in
  (c = r) →
  let Δ := b * b - 4 * a * c in
  let sqrtΔ := real.sqrt Δ in
  let r1 := (-b + sqrtΔ)/(2*a) in
  let r2 := (-b - sqrtΔ)/(2*a) in
  |r1 - r2| = 2 :=
by
  sorry

end abs_diff_roots_quadratic_eq_l215_215936


namespace chessboard_partition_l215_215108

theorem chessboard_partition :
  ∃ (p : ℕ) (sizes : set ℕ), p = 7 ∧ sizes = {2, 4, 6, 10, 12, 14, 16} ∧
  (∀ (R : ℕ), R ∈ sizes → R % 2 = 0) ∧
  (sizes.to_finset.sum = 64) ∧
  sizes.to_finset.card = p :=
sorry

end chessboard_partition_l215_215108


namespace mark_total_spending_l215_215339

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end mark_total_spending_l215_215339


namespace volume_ratio_is_1_over_9_l215_215806

noncomputable def ratio_of_volumes (s : ℝ) : ℝ :=
let V_tet := (s^3 * real.sqrt 2) / 12 in
let h := real.sqrt (2/3) * s in
let a := (h / 2) in
let V_oct := (a^3 * real.sqrt 2) / 3 in
V_oct / V_tet

theorem volume_ratio_is_1_over_9 (s : ℝ) : ratio_of_volumes s = 1 / 9 :=
by sorry

end volume_ratio_is_1_over_9_l215_215806


namespace evaluate_expression_l215_215865

theorem evaluate_expression (b x : ℝ) (h : x = b + 9) : (2 * x - b + 5) = (b + 23) :=
by
  sorry

end evaluate_expression_l215_215865


namespace emily_sixth_quiz_score_l215_215162

theorem emily_sixth_quiz_score (s1 s2 s3 s4 s5 : ℕ) (target_average : ℕ) : 
  s1 = 92 → s2 = 95 → s3 = 87 → s4 = 89 → s5 = 100 → target_average = 93 →
  ∃ s6 : ℕ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = target_average :=
begin
  intros h1 h2 h3 h4 h5 h6,
  use 95,
  have : (92 + 95 + 87 + 89 + 100 + 95) = 558,
  { trivial },
  simp [h1, h2, h3, h4, h5, h6, this],
  norm_num,
  exact rfl,
end

end emily_sixth_quiz_score_l215_215162


namespace rectangles_have_unique_property_of_equal_diagonals_l215_215106

theorem rectangles_have_unique_property_of_equal_diagonals (rectangle rhombus : Type)
  (is_rectangle : rectangle → Prop)
  (is_rhombus : rhombus → Prop)
  (sum_of_angles_360 : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∑ ang : ℝ in {q}, ang = 360))
  (diagonals_bisect : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∃ midp : Prop, ∀ l r : ℝ, l ≠ r → l + r = 0))
  (diagonals_equal_length : ∀ (r : rectangle), Prop := ∀ x y : ℝ, x = y)
  (diagonals_perpendicular : ∀ (r : rhombus), Prop := ⊥)
  : ∀ r : rectangle, is_rectangle r → diagonals_equal_length r :=
  begin
    intros r h,
    sorry
  end

end rectangles_have_unique_property_of_equal_diagonals_l215_215106


namespace curve_not_parabola_l215_215532

theorem curve_not_parabola (k : ℝ) : ¬(∃ (a b c d e f : ℝ),
  x² + k*y² = a*x² + b*x*y + c*y² + d*x + e*y + f  
  ∧ a ≠ 0 ∧ c ≠ 0 ∧ b² - 4*a*c = 0) :=
sorry

end curve_not_parabola_l215_215532


namespace f_extremum_m_and_monotonicity_l215_215234

noncomputable def f (x m : ℝ) : ℝ := log x - exp (x + m)

theorem f_extremum_m_and_monotonicity :
  (∃ m : ℝ, (∃ x : ℝ, f x m = ln x - exp (x + m) ∧ x = 1 ∧ (derivative (λ x, f x m)) 1 = 0) ∧ m = -1) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → (derivative (λ x, f x (-1))) x > 0) ∧
  (∀ x : ℝ, 1 < x → (derivative (λ x, f x (-1))) x < 0) :=
begin
  sorry,
end

end f_extremum_m_and_monotonicity_l215_215234


namespace ratio_second_to_third_l215_215850

-- Variables to represent the volumes of the containers
variables (C D E : ℝ)

-- Conditions derived from the problem
axiom initial_pour : (3 / 5) * C = (2 / 3) * D
axiom transfer_pour : (1 / 2) * E = (2 / 3) * D - x
  where x : ℝ  -- remaining volume in D after transferring into E

-- The theorem we need to prove
theorem ratio_second_to_third : D / E = 2 / 3 :=
sorry

end ratio_second_to_third_l215_215850


namespace fraction_transformed_l215_215596

variables (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab_pos : a * b > 0)

noncomputable def frac_orig := (a + 2 * b) / (2 * a * b)
noncomputable def frac_new := (3 * a + 2 * 3 * b) / (2 * 3 * a * 3 * b)

theorem fraction_transformed :
  frac_new a b = (1 / 3) * frac_orig a b :=
sorry

end fraction_transformed_l215_215596


namespace length_PQ_l215_215276

-- Definitions
noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (-√3, 0)
noncomputable def C : ℝ × ℝ := (√3, 0)
noncomputable def D : ℝ × ℝ := (0, 0)
noncomputable def P : ℝ × ℝ := (-√3 / 2, 1 / 2)
noncomputable def Q : ℝ × ℝ := (√6 / 2, 1 - √2 / 2)

-- 3D positions after folding
noncomputable def P' : ℝ × ℝ × ℝ := (3 / 4, -1, 0)

-- Distance function
noncomputable def distance (X Y : ℝ × ℝ × ℝ) : ℝ :=
  ((X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (X.3 - Y.3)^2)^(1 / 2)

-- Define PQ length
noncomputable def PQ_length : ℝ :=
  distance P' (√6 / 2, 1 - √2 / 2, 0)

-- Assertion
theorem length_PQ : PQ_length = sqrt((3 / 4 - √6 / 2)^2 + (-2 + √2 / 2)^2) :=
sorry

end length_PQ_l215_215276


namespace problem_statement_l215_215366

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)

theorem problem_statement : ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by 
  sorry

end problem_statement_l215_215366


namespace remainder_of_3_pow_100_plus_5_mod_8_l215_215741

theorem remainder_of_3_pow_100_plus_5_mod_8 :
  (3^100 + 5) % 8 = 6 := by
sorry

end remainder_of_3_pow_100_plus_5_mod_8_l215_215741


namespace incorrect_probability_statement_D_l215_215007

-- Definitions of the problem conditions
def students : ℕ := 3
def topics : ℕ := 2
def male_student : ℕ := 1
def female_students : ℕ := 2

-- The mathematical problem to prove (skipping the actual proof by using sorry)
theorem incorrect_probability_statement_D :
  ∃ (p: ℚ), p = 1 ∧ p ≠ 3/4 := 
by { use (1:ℚ), split, sorry, sorry }

end incorrect_probability_statement_D_l215_215007


namespace length_of_side_AC_of_triangle_median_extended_circumcircle_l215_215298

theorem length_of_side_AC_of_triangle_median_extended_circumcircle
  (a b m n : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (m_pos : 0 < m) (n_pos : 0 < n)
  (BD : ℝ) (DE : ℝ) (h_ratio : BD / DE = m / n) :
  AC = sqrt ((2 * n) / (m + n) * (a^2 + b^2)) := by
  sorry

end length_of_side_AC_of_triangle_median_extended_circumcircle_l215_215298


namespace Georgia_students_l215_215476

-- Define conditions
def muffins_per_batch : ℕ := 6
def total_batches : ℕ := 36
def months : ℕ := 9

-- Define total muffins and times muffins brought
def total_muffins : ℕ := total_batches * muffins_per_batch := by rfl
def times_muffins_brought : ℕ := months := by rfl

-- Statement of the problem
theorem Georgia_students : (total_muffins / times_muffins_brought = 24) := by
    -- Use the condition definitions
    have h1 : total_muffins = total_batches * muffins_per_batch := by rfl
    have h2 : times_muffins_brought = months := by rfl
    sorry

end Georgia_students_l215_215476


namespace solve_trig_eq_l215_215682

noncomputable theory

open Real

theorem solve_trig_eq (x : ℝ) : (12 * sin x - 5 * cos x = 13) →
  ∃ (k : ℤ), x = (π / 2) + arctan (5 / 12) + 2 * k * π :=
by
s∞rry

end solve_trig_eq_l215_215682


namespace range_of_a_l215_215275

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a*x^2 - a*x - 2 ≤ 0) → (-8 ≤ a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l215_215275


namespace max_sum_yk_difference_l215_215357

theorem max_sum_yk_difference :
  ∀ (x : ℕ → ℝ),
    (∑ k in finset.range 2005, |x k - x (k + 1)|) = 2007 →
    let y k := (∑ i in finset.range k, x i) / k in
    (∑ k in finset.range 2006, |y k - y (k + 1)|) ≤ 2006 :=
begin
  sorry
end

end max_sum_yk_difference_l215_215357


namespace solve_custom_eq_l215_215501

namespace CustomProof

def custom_mul (a b : ℕ) : ℕ := a * b + a + b

theorem solve_custom_eq (x : ℕ) (h : custom_mul 3 x = 31) : x = 7 := 
by
  sorry

end CustomProof

end solve_custom_eq_l215_215501


namespace max_value_sum_faces_edges_vertices_l215_215454

def rectangular_prism_faces : ℕ := 6
def rectangular_prism_edges : ℕ := 12
def rectangular_prism_vertices : ℕ := 8

def pyramid_faces_added : ℕ := 4
def pyramid_base_faces_covered : ℕ := 1
def pyramid_edges_added : ℕ := 4
def pyramid_vertices_added : ℕ := 1

def resulting_faces : ℕ := rectangular_prism_faces - pyramid_base_faces_covered + pyramid_faces_added
def resulting_edges : ℕ := rectangular_prism_edges + pyramid_edges_added
def resulting_vertices : ℕ := rectangular_prism_vertices + pyramid_vertices_added

def sum_resulting_faces_edges_vertices : ℕ := resulting_faces + resulting_edges + resulting_vertices

theorem max_value_sum_faces_edges_vertices : sum_resulting_faces_edges_vertices = 34 :=
by
  sorry

end max_value_sum_faces_edges_vertices_l215_215454


namespace locus_of_intersection_l215_215146

-- Define the conditions
def line_e (m_e x y : ℝ) : Prop := y = m_e * (x - 1) + 1
def line_f (m_f x y : ℝ) : Prop := y = m_f * (x + 1) + 1
def slope_diff_cond (m_e m_f : ℝ) : Prop := (m_e - m_f = 2 ∨ m_f - m_e = 2)
def not_at_points (x y : ℝ) : Prop := (x, y) ≠ (1, 1) ∧ (x, y) ≠ (-1, 1)

-- Define the proof problem
theorem locus_of_intersection (x y m_e m_f : ℝ) :
  line_e m_e x y → line_f m_f x y → slope_diff_cond m_e m_f → not_at_points x y →
  (y = x^2 ∨ y = 2 - x^2) :=
by
  intros he hf h_diff h_not_at
  sorry

end locus_of_intersection_l215_215146


namespace prob_exactly_3_out_of_5_correct_max_num_correct_with_prob_not_less_than_1_6_l215_215883

-- Definition for the total number of ways for 5 individuals
def total_ways := 5!

-- Definition for the number of ways exactly 3 out of 5 individuals sit in their designated seats
def num_ways_exactly_3_correct := Nat.choose 5 3

-- Definition for the probability of exactly 3 out of 5 individuals sitting in their assigned seats
def prob_exactly_3_correct := num_ways_exactly_3_correct / total_ways

-- Definition for the number of ways exactly 2 out of 5 individuals sit in their designated seats
def num_ways_exactly_2_correct := (Nat.choose 5 2) * 2

-- Definition for the probability threshold
def prob_threshold := 1 / 6

-- Lean 4 statement for Part 1
theorem prob_exactly_3_out_of_5_correct : prob_exactly_3_correct = 1 / 12 :=
by
  sorry

-- Lean 4 statement for Part 2
theorem max_num_correct_with_prob_not_less_than_1_6 : 
  ∃ n, n ≤ 5 ∧ prob_threshold <= num_ways_exactly_2_correct / total_ways ∧ n = 2 :=
by
  sorry

end prob_exactly_3_out_of_5_correct_max_num_correct_with_prob_not_less_than_1_6_l215_215883


namespace product_of_all_possible_values_l215_215594

theorem product_of_all_possible_values (x : ℝ) :
  (|16 / x + 4| = 3) → ((x = -16 ∨ x = -16 / 7) →
  (x_1 = -16 ∧ x_2 = -16 / 7) →
  (x_1 * x_2 = 256 / 7)) :=
sorry

end product_of_all_possible_values_l215_215594


namespace circle_tangent_to_yaxis_and_line_l215_215561

theorem circle_tangent_to_yaxis_and_line :
  (∃ C : ℝ → ℝ → Prop, 
    (∀ x y r : ℝ, C x y ↔ (x - 3) ^ 2 + (y - 2) ^ 2 = 9 ∨ (x + 1 / 3) ^ 2 + (y - 2) ^ 2 = 1 / 9) ∧ 
    (∀ y : ℝ, C 0 y → y = 2) ∧ 
    (∀ x y: ℝ, C x y → (∃ x1 : ℝ, 4 * x - 3 * y + 9 = 0 → 4 * x1 + 3 = 0))) :=
sorry

end circle_tangent_to_yaxis_and_line_l215_215561


namespace factorize_polynomial_l215_215165

theorem factorize_polynomial (m : ℤ) : 4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end factorize_polynomial_l215_215165


namespace rectangles_have_unique_property_of_equal_diagonals_l215_215107

theorem rectangles_have_unique_property_of_equal_diagonals (rectangle rhombus : Type)
  (is_rectangle : rectangle → Prop)
  (is_rhombus : rhombus → Prop)
  (sum_of_angles_360 : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∑ ang : ℝ in {q}, ang = 360))
  (diagonals_bisect : ∀ (q : Type), q → Prop → Prop → (∀ p : q, Prop) → p → Prop → (∃ midp : Prop, ∀ l r : ℝ, l ≠ r → l + r = 0))
  (diagonals_equal_length : ∀ (r : rectangle), Prop := ∀ x y : ℝ, x = y)
  (diagonals_perpendicular : ∀ (r : rhombus), Prop := ⊥)
  : ∀ r : rectangle, is_rectangle r → diagonals_equal_length r :=
  begin
    intros r h,
    sorry
  end

end rectangles_have_unique_property_of_equal_diagonals_l215_215107


namespace necessary_not_sufficient_l215_215896

theorem necessary_not_sufficient (m a : ℝ) (h : a ≠ 0) :
  (|m| = a → m = -a ∨ m = a) ∧ ¬ (m = -a ∨ m = a → |m| = a) :=
by
  sorry

end necessary_not_sufficient_l215_215896


namespace second_order_derivative_proof_l215_215176

noncomputable def second_order_derivative (t : ℝ) : ℝ :=
  (t * (1 - t^2)) / (1 + t^2)^2

theorem second_order_derivative_proof (t : ℝ) :
  ∀ (x y : ℝ), 
  x = Real.log t →
  y = Real.arctan t →
  (∂[t] ∂[t] y / (∂[t] x)^2 = second_order_derivative t) :=
by
  assume x y
  assume h₁ : x = Real.log t
  assume h₂ : y = Real.arctan t
  sorry

end second_order_derivative_proof_l215_215176


namespace find_omega_and_fval_l215_215597

variables {ω : ℝ} {f : ℝ → ℝ}

def given_and_conditions (f : ℝ → ℝ): Prop :=
  (∀ x, f x = Real.tan (ω * x + Real.pi / 4)) ∧
  (ω > 0) ∧
  (∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = 2 * Real.pi)

theorem find_omega_and_fval 
  (h : given_and_conditions f) :
  ω = 1 / 2 ∧ f (Real.pi / 6) = Real.sqrt 3 := 
sorry

end find_omega_and_fval_l215_215597


namespace max_value_on_interval_l215_215211

variables {a : ℝ} {f : ℝ → ℝ}

-- Define the properties of the function f
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def continuous_and_differentiable (f : ℝ → ℝ) : Prop := continuous f ∧ differentiable ℝ f
def func_property1 (f : ℝ → ℝ) : Prop := ∀ x, f(1 + x) = f(1 - x)
def func_property2 (f : ℝ → ℝ) : f 1 = a
def func_property3 (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 1 → deriv f x < f x

-- Statement to prove the maximum value of f on [2015, 2016] is -a
theorem max_value_on_interval 
  (h_odd : odd_function f)
  (h_cont_diff : continuous_and_differentiable f)
  (h_prop1 : func_property1 f)
  (h_prop2 : func_property2 f)
  (h_prop3 : func_property3 f) :
  ∀ x ∈ set.Icc 2015 2016, f x ≤ -a :=
sorry

end max_value_on_interval_l215_215211


namespace basketball_weight_l215_215663

variable (b c : ℝ)

theorem basketball_weight (h1 : 9 * b = 5 * c) (h2 : 3 * c = 75) : b = 125 / 9 :=
by
  sorry

end basketball_weight_l215_215663


namespace mode_of_scores_is_85_median_of_scores_is_88_l215_215609

-- Define the scores list
def scores : List ℕ := [97, 88, 85, 93, 85]

-- Prove that the mode of scores is 85
theorem mode_of_scores_is_85 : mode scores = 85 := 
by sorry

-- Prove that the median of scores is 88
theorem median_of_scores_is_88 : median scores = 88 := 
by sorry

end mode_of_scores_is_85_median_of_scores_is_88_l215_215609


namespace cyclic_quadrilateral_max_incircle_radius_l215_215843

theorem cyclic_quadrilateral_max_incircle_radius :
  ∀ (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
    (a b c d : ℝ),
  a = 15 →
  b = 10 →
  c = 8 →
  d = 13 →
  ∃ r : ℝ, r = 5.7 :=
begin
  intros A B C D _ _ _ _ a b c d ha hb hc hd,
  sorry -- Proof goes here
end

end cyclic_quadrilateral_max_incircle_radius_l215_215843


namespace diana_higher_than_apollo_probability_l215_215509

def diana_die : ℕ := 8
def apollo_die : ℕ := 6

def total_outcomes : ℕ := diana_die * apollo_die
def successful_outcomes : ℕ := 27

theorem diana_higher_than_apollo_probability :
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = 9 / 16 := by
  sorry

end diana_higher_than_apollo_probability_l215_215509


namespace average_speed_of_driver_B_l215_215015

theorem average_speed_of_driver_B :
  ∃ (v_B : ℝ), (∀ (t : ℝ),
    let d_A := 90 + 90 * t in
    let d_B := v_B * t in
    (d_A - d_B = 145) ∧ (d_A + d_B = 1025) ∧ v_B = 485 / 6) :=
begin
  use 485 / 6,
  intro t,
  split,
  { intro,
    rw [←add_sub_assoc, add_comm 145, ←add_assoc, add_sub_cancel'_right, ←mul_add],
    sorry },
  { intro,
    sorry }
end

end average_speed_of_driver_B_l215_215015


namespace solve_trig_eq_l215_215680

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end solve_trig_eq_l215_215680


namespace find_angle_between_a_and_b_l215_215919

variables (a b : Vector ℝ) -- Define vectors a and b
variables (abs_a abs_b : ℝ) -- Magnitudes of vectors a and b
variables (α : ℝ) -- Angle between vectors a and b

-- Conditions
axiom magnitudes_are_integers : abs_a ∈ ℕ ∧ abs_b ∈ ℕ
axiom equation1 : (abs_a + abs_b) * (abs_a + 3 * abs_b) = 105
axiom equation2 : (a + b) • (a + 3 * b) = 33    -- Using dot product for vectors

-- Statement to prove
theorem find_angle_between_a_and_b : α = 120 :=
by
  sorry

end find_angle_between_a_and_b_l215_215919


namespace average_score_correct_l215_215278

-- Define the parameters and the given conditions
variables (total_students assigned_day_students makeup_day_students : ℕ)
variables (assigned_day_avg makeup_day_avg : ℚ)

-- Specify the conditions based on the problem statement
def conditions (total_students = 100) (assigned_day_students = 70)
  (makeup_day_students = 30) 
  (assigned_day_avg = 0.6) (makeup_day_avg = 0.9) :=
  True

-- Definition of the average score for the entire class
def average_score (total_students : ℕ) (assigned_day_students : ℕ) 
  (makeup_day_students : ℕ) (assigned_day_avg : ℚ) (makeup_day_avg: ℚ) : ℚ :=
((assigned_day_avg * assigned_day_students) + (makeup_day_avg * makeup_day_students)) / total_students

-- The theorem stating that the average score for the entire class is 0.69
theorem average_score_correct : 
  average_score 100 70 30 0.6 0.9 = 0.69 := sorry

end average_score_correct_l215_215278


namespace geom_series_sum_l215_215135

theorem geom_series_sum : 
  let a := 1 : ℝ
  let r := 1 / 4 : ℝ
  let S := a / (1 - r)
  S = 4 / 3 :=
by
  sorry

end geom_series_sum_l215_215135


namespace total_birds_in_pet_store_l215_215054

theorem total_birds_in_pet_store
  (number_of_cages : ℕ)
  (parrots_per_cage : ℕ)
  (parakeets_per_cage : ℕ)
  (total_birds_in_cage : ℕ)
  (total_birds : ℕ) :
  number_of_cages = 8 →
  parrots_per_cage = 2 →
  parakeets_per_cage = 7 →
  total_birds_in_cage = parrots_per_cage + parakeets_per_cage →
  total_birds = number_of_cages * total_birds_in_cage →
  total_birds = 72 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_birds_in_pet_store_l215_215054


namespace range_of_a_l215_215965

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - a

theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ (-9 < a ∧ a < 5/3) :=
by
  sorry

end range_of_a_l215_215965


namespace flower_team_participation_l215_215402

-- Definitions based on the conditions in the problem
def num_rows : ℕ := 60
def first_row_people : ℕ := 40
def people_increment : ℕ := 1

-- Statement to be proved in Lean
theorem flower_team_participation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ num_rows) : 
  ∃ y : ℕ, y = first_row_people - people_increment + x :=
by
  -- Placeholder for the proof
  sorry

end flower_team_participation_l215_215402


namespace distinct_lcm_values_count_l215_215513

theorem distinct_lcm_values_count {a b : ℕ} (ha : a % 2 = 0) (hb : b % 2 = 0) 
  (h : Nat.gcd a b + Nat.lcm a b = 2^23) : 
  ∃ S : Finset ℕ, S.card = 22 ∧ ∀ d ∈ S, ∃ a b : ℕ, ha, hb, h ∧ Nat.lcm a b = d :=
sorry

end distinct_lcm_values_count_l215_215513


namespace total_amount_spent_is_40_l215_215341

-- Definitions based on conditions
def tomatoes_pounds : ℕ := 2
def tomatoes_price_per_pound : ℕ := 5
def apples_pounds : ℕ := 5
def apples_price_per_pound : ℕ := 6

-- Total amount spent computed
def total_spent : ℕ :=
  (tomatoes_pounds * tomatoes_price_per_pound) +
  (apples_pounds * apples_price_per_pound)

-- The Lean theorem statement
theorem total_amount_spent_is_40 : total_spent = 40 := by
  unfold total_spent
  unfold tomatoes_pounds tomatoes_price_per_pound apples_pounds apples_price_per_pound
  calc
    2 * 5 + 5 * 6 = 10 + 30 : by rfl
    ... = 40 : by rfl

end total_amount_spent_is_40_l215_215341


namespace smallest_group_b_largest_group_b_l215_215058

-- Definition of the set containing numbers from 1 to 20
def numbers_set : List ℕ := List.range' 1 20

-- Sum of elements of Group A is equal to product of elements of Group B
def valid_partition (A B : List ℕ) : Prop :=
  (List.sum A = List.prod B) ∧ (A ++ B = numbers_set) ∧ (A.intersect B = [])

theorem smallest_group_b :
  ∃ (B : List ℕ), valid_partition (numbers_set.filter (λ x, B.mem x.not)) B ∧ B.length = 3 :=
begin
  sorry
end

theorem largest_group_b :
  ∃ (B : List ℕ), valid_partition (numbers_set.filter (λ x, B.mem x.not)) B ∧ B.length = 5 :=
begin
  sorry
end

end smallest_group_b_largest_group_b_l215_215058


namespace proof_trig_identity_l215_215055

noncomputable def trig_identity : Prop :=
  sin (π / 12) * cos (π / 12) = 1 / 4

theorem proof_trig_identity : trig_identity :=
by sorry

end proof_trig_identity_l215_215055


namespace min_S_n_at_7_or_8_l215_215289

noncomputable def a_n (n : ℕ) := -3 + (n - 1) * d

def S_n (n : ℕ) := (n / 2) * (2 * a_n 1 + (n - 1) * d)

theorem min_S_n_at_7_or_8 (d : ℝ) (h : S_n 5 = S_n 10) : S_n (7 : ℕ) = S_n (8 : ℕ) :=
by
  sorry

end min_S_n_at_7_or_8_l215_215289


namespace math_problem_l215_215207

-- Conditions from the problem
def sum_seq (a_n : ℕ → ℕ) (S_n : ℕ → ℕ): Prop :=
  ∀ n : ℕ, 0 < n → S_n n = a_n n + n^2 - 1

-- The general formula to prove
def general_formula (a_n : ℕ → ℕ): Prop := 
  ∀ n : ℕ, 0 < n → a_n n = 2 * n + 1

-- The inequality to prove
def sum_inequality (S_n : ℕ → ℕ): Prop :=
  ∀ n : ℕ, 0 < n → (finset.range n).sum (λ k, 1 / (S_n (k + 1))) < 3 / 4

theorem math_problem (a_n S_n : ℕ → ℕ):
  sum_seq a_n S_n →
  general_formula a_n ∧ sum_inequality S_n :=
by
  intro h
  split
  sorry
  sorry

end math_problem_l215_215207


namespace problem_700_3_in_scientific_notation_l215_215871

-- Definitions
def scientific_notation (a : ℝ) (n : ℤ) : ℝ :=
  a * (10 : ℝ)^n

def valid_scientific_form (a : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10

-- Theorem statement based on the given problem and conditions
theorem problem_700_3_in_scientific_notation :
  ∃ a n, valid_scientific_form a ∧ scientific_notation a n = 700.3 ∧ a = 7.003 ∧ n = 2 :=
by {
  existsi (7.003 : ℝ), 
  existsi (2 : ℤ),
  sorry
}

end problem_700_3_in_scientific_notation_l215_215871


namespace moles_NaHCO3_combined_l215_215524

-- Define conditions as given in the problem
def moles_HNO3_combined := 1
def moles_NaNO3_result := 1

-- The chemical equation as a definition
def balanced_reaction (moles_NaHCO3 moles_HNO3 moles_NaNO3 : ℕ) : Prop :=
  moles_HNO3 = moles_NaNO3 ∧ moles_NaHCO3 = moles_HNO3

-- The proof problem statement
theorem moles_NaHCO3_combined :
  balanced_reaction 1 moles_HNO3_combined moles_NaNO3_result → 1 = 1 :=
by 
  sorry

end moles_NaHCO3_combined_l215_215524


namespace ram_work_rate_l215_215426

-- Definitions as given in the problem
variable (W : ℕ) -- Total work can be represented by some natural number W
variable (R M : ℕ) -- Raja's work rate and Ram's work rate, respectively

-- Given conditions
variable (combined_work_rate : R + M = W / 4)
variable (raja_work_rate : R = W / 12)

-- Theorem to be proven
theorem ram_work_rate (combined_work_rate : R + M = W / 4) (raja_work_rate : R = W / 12) : M = W / 6 := 
  sorry

end ram_work_rate_l215_215426


namespace geometric_series_sum_eq_4_over_3_l215_215124

theorem geometric_series_sum_eq_4_over_3 : 
  let a := 1
  let r := 1/4
  inf_geometric_sum a r = 4/3 := by
begin
  intros,
  sorry
end

end geometric_series_sum_eq_4_over_3_l215_215124


namespace find_angle_BAC_l215_215311

-- Define the incenter I
def incenter (A B C : Point) : Point := sorry -- Placeholder definition

-- Define conditions
variables {A B C I : Point}
variables {b : ℝ} (h : ℝ)
variables (a : ℝ)
axiom isosceles_triangle (ABC : Triangle) : ABC.is_isosceles AB AC
axiom incenter_position : incenter A B C = I
axiom BC_eq_AB_plus_AI (ABC : Triangle) : length (BC) = length (AB) + length (segment A I)

-- Define length function placeholder
def length (s : Segment) : ℝ := sorry -- Placeholder definition

-- Define angle function placeholder
def angle (A B C : Point) : ℝ := sorry -- Placeholder definition

-- The theorem we want to prove
theorem find_angle_BAC : angle B A C = 45 := sorry

end find_angle_BAC_l215_215311


namespace discount_amount_l215_215631

def tshirt_cost : ℕ := 30
def backpack_cost : ℕ := 10
def blue_cap_cost : ℕ := 5
def total_spent : ℕ := 43

theorem discount_amount : (tshirt_cost + backpack_cost + blue_cap_cost) - total_spent = 2 := by
  sorry

end discount_amount_l215_215631


namespace quadratic_func_properties_l215_215239

theorem quadratic_func_properties :
  let y := λ x : ℝ => 2 * x^2 - 8 * x + 6
  in
  (∀ x, y x = 2 * (x - 2)^2 - 2)
  ∧ ((2, -2) is_vertex_of y)
  ∧ (axis_of_symmetry y = 2)
  ∧ (intersection_with_x_axis y = [(1, 0), (3, 0)])
  ∧ (intersection_with_y_axis y = [(0, 6)]) := by
  sorry

def is_vertex_of (f : ℝ → ℝ) (v : ℝ × ℝ) : Prop := v.1 = (f v.1) ∧ v.2 = (f v.2)
def axis_of_symmetry (f : ℝ → ℝ) : ℝ := -(f 1) / (2 * f (2))
def intersection_with_x_axis (f : ℝ → ℝ) : List (ℝ × ℝ) := 
  if h : ∃ x, f x = 0 then
    let (a, b) := Classical.some (h : ∃ p : ℝ × ℝ, p.1 = 1 ∧ p.2 = 3)
    [⟨a, 0⟩, ⟨b, 0⟩]
  else []
def intersection_with_y_axis (f : ℝ → ℝ) : List (ℝ × ℝ) := [⟨0, f 0⟩]


end quadratic_func_properties_l215_215239


namespace log_eq_0_implies_x_inv_cubed_l215_215255

-- Define the condition as a Lean proposition
theorem log_eq_0_implies_x_inv_cubed (x : ℝ) (h : log 7 (log 5 (log 2 x)) = 0) : 
  x^(-1/3) = 1 / (2 * real.cbrt 4) :=
sorry  -- proof to be provided

end log_eq_0_implies_x_inv_cubed_l215_215255


namespace length_of_bridge_l215_215016

theorem length_of_bridge 
  (lenA : ℝ) (speedA : ℝ) (lenB : ℝ) (speedB : ℝ) (timeA : ℝ) (timeB : ℝ) (startAtSameTime : Prop)
  (h1 : lenA = 120) (h2 : speedA = 12.5) (h3 : lenB = 150) (h4 : speedB = 15.28) 
  (h5 : timeA = 30) (h6 : timeB = 25) : 
  (∃ X : ℝ, X = 757) :=
by
  sorry

end length_of_bridge_l215_215016


namespace units_digit_G1000_is_3_l215_215507

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 1

theorem units_digit_G1000_is_3 : (G 1000) % 10 = 3 := sorry

end units_digit_G1000_is_3_l215_215507


namespace total_amount_spent_is_40_l215_215340

-- Definitions based on conditions
def tomatoes_pounds : ℕ := 2
def tomatoes_price_per_pound : ℕ := 5
def apples_pounds : ℕ := 5
def apples_price_per_pound : ℕ := 6

-- Total amount spent computed
def total_spent : ℕ :=
  (tomatoes_pounds * tomatoes_price_per_pound) +
  (apples_pounds * apples_price_per_pound)

-- The Lean theorem statement
theorem total_amount_spent_is_40 : total_spent = 40 := by
  unfold total_spent
  unfold tomatoes_pounds tomatoes_price_per_pound apples_pounds apples_price_per_pound
  calc
    2 * 5 + 5 * 6 = 10 + 30 : by rfl
    ... = 40 : by rfl

end total_amount_spent_is_40_l215_215340


namespace quadrilateral_area_perimeter_l215_215614

theorem quadrilateral_area_perimeter 
  (EF FG GH HE : ℝ) (angleF angleG : ℝ) 
  (hf : EF = 5) (hg : FG = 6) (hh : GH = 7) (he : HE = 4)
  (haF : angleF = 135) (haG : angleG = 135) :
  (5 + 6 + 7 + 4 = 22 ∧ 
  (1 / 2 * 5 * 6 * (Real.sin (135 * Real.pi / 180) + 1 / 2 * 6 * 7 * (Real.sin (135 * Real.pi / 180)) = 18 * Real.sqrt 2)) := 
by
  sorry

end quadrilateral_area_perimeter_l215_215614


namespace probability_x_y_le_5_l215_215803

noncomputable section

open MeasureTheory

namespace probability

def region (x y : ℝ) : Prop := (0 ≤ x ∧ x ≤ 4) ∧ (0 ≤ y ∧ y ≤ 4)

noncomputable def probability_of_event : ℝ :=
  (volume {p : ℝ × ℝ | region p.1 p.2 ∧ p.1 + p.2 ≤ 5}) / (volume {p : ℝ × ℝ | region p.1 p.2})

theorem probability_x_y_le_5 : probability_of_event = 17 / 32 := 
 by 
  sorry

end probability

end probability_x_y_le_5_l215_215803


namespace find_real_x_l215_215170

noncomputable def solution_set (x : ℝ) := (5 ≤ x) ∧ (x < 5.25)

theorem find_real_x (x : ℝ) :
  (⌊x * ⌊x⌋⌋ = 20) ↔ solution_set x :=
by
  sorry

end find_real_x_l215_215170


namespace triangle_side_length_l215_215603

variable {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define points A, B, and C
variables (a b c : A)

-- Define midpoints M and N
def midpoint (x y : A) : A := sorry

-- Assume the given conditions:
-- Median from A to M is perpendicular to median from B to N
def median_perpendicular (M N : A) : Prop := sorry

-- BC = 8 and AC = 10
def length (x y : A) : ℝ := sorry

variable (BC_length : length b c = 8)
variable (AC_length : length a c = 10)
variable (M : A := midpoint b c)
variable (N : A := midpoint a c)
variable (medians_perpendicular : median_perpendicular M N)

-- Prove length AB = sqrt(41)
theorem triangle_side_length : length a b = Real.sqrt 41 :=
by sorry

end triangle_side_length_l215_215603


namespace mode_of_scores_is_85_median_of_scores_is_88_l215_215608

-- Define the scores list
def scores : List ℕ := [97, 88, 85, 93, 85]

-- Prove that the mode of scores is 85
theorem mode_of_scores_is_85 : mode scores = 85 := 
by sorry

-- Prove that the median of scores is 88
theorem median_of_scores_is_88 : median scores = 88 := 
by sorry

end mode_of_scores_is_85_median_of_scores_is_88_l215_215608


namespace part1_min_triangles_part2_colorable_l215_215649

noncomputable def min_triangles (P : Finset (Fin 1994)) (groups : Fin 83 → Finset (Fin 1994)) : ℕ :=
  ∑ i, Nat.choose (groups i).card 3

theorem part1_min_triangles (P : Finset (Fin 1994)) (groups : Fin 83 → Finset (Fin 1994))
  (hP : ∀ (i j : Fin 83), i ≠ j → Disjoint (groups i) (groups j))
  (h_card : (P.card = 1994) ∧ (∀ i, 3 ≤ (groups i).card)) :
  min_triangles P groups = 168544 := sorry

noncomputable def colorable (P : Finset (Fin 1994)) (groups : Fin 83 → Finset (Fin 1994)) : Prop :=
  ∃ (coloring : Sym2 (Fin 1994) → Fin 4),
    ∀ {a b c : Fin 1994}, a ∈ P → b ∈ P → c ∈ P →
    a ≠ b → b ≠ c → c ≠ a →
    ({a, b, c} ⊆ P) →
    coloring ⟨a, b⟩ ≠ coloring ⟨b, c⟩ ∨ coloring ⟨b, c⟩ ≠ coloring ⟨c, a⟩ ∨ coloring ⟨c, a⟩ ≠ coloring ⟨a, b⟩

theorem part2_colorable (P : Finset (Fin 1994)) (groups : Fin 83 → Finset (Fin 1994))
  (hP : ∀ (i j : Fin 83), i ≠ j → Disjoint (groups i) (groups j))
  (h_card : (P.card = 1994) ∧ (∀ i, 3 ≤ (groups i).card))
  (h_triang : min_triangles P groups = 168544) :
  colorable P groups := sorry

end part1_min_triangles_part2_colorable_l215_215649


namespace t_minus_s_calc_l215_215459

-- Definitions based on conditions
def students : ℕ := 120
def teachers : ℕ := 6
def enrollments : List ℕ := [40, 40, 20, 10, 5, 5]

-- Average number of students per teacher (t)
def t : ℝ := (enrollments.sum : ℝ) / (teachers : ℝ)

-- Average number of students per student (s)
def s : ℝ :=
  (enrollments.map (λ n => n * (n : ℝ) / (students : ℝ))).sum / (students : ℝ)

-- Proof objective
theorem t_minus_s_calc : t - s = -11.25 := by
  sorry

end t_minus_s_calc_l215_215459


namespace range_of_b_l215_215241

-- Define the set M as the points (x, y) satisfying the ellipse equation
def M : Set (ℝ × ℝ) := { p | p.1^2 + 2 * p.2^2 = 3 }

-- Define the set N as the points (x, y) satisfying the line equation y = mx + b
def N (m b : ℝ) : Set (ℝ × ℝ) := { p | p.2 = m * p.1 + b }

-- The main theorem to be proved
theorem range_of_b (b : ℝ) : (∀ m : ℝ, ∃ x y : ℝ, (x, y) ∈ M ∧ (x, y) ∈ N m b) ↔ (- real.sqrt 6 / 2 ≤ b ∧ b ≤ real.sqrt 6 / 2) :=
by
  sorry

end range_of_b_l215_215241


namespace find_five_digit_number_l215_215793

theorem find_five_digit_number (a b c d e : ℕ) 
  (h : [ (10 * a + a), (10 * a + b), (10 * a + b), (10 * a + b), (10 * a + c), 
         (10 * b + c), (10 * b + b), (10 * b + c), (10 * c + b), (10 * c + b)] = 
         [33, 37, 37, 37, 38, 73, 77, 78, 83, 87]) :
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 37837 :=
sorry

end find_five_digit_number_l215_215793


namespace vectors_perpendicular_if_length_equal_l215_215248

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem vectors_perpendicular_if_length_equal 
  (h₁ : a ≠ 0 ∧ b ≠ 0) 
  (h₂ : ∥a + b∥ = ∥a - b∥) : 
  a ⬝ b = 0 :=
by sorry

end vectors_perpendicular_if_length_equal_l215_215248


namespace car_speed_l215_215437

theorem car_speed (distance time speed : ℝ)
  (h_const_speed : ∀ t : ℝ, t = time → speed = distance / t)
  (h_distance : distance = 48)
  (h_time : time = 8) :
  speed = 6 :=
by
  sorry

end car_speed_l215_215437


namespace digit_counts_in_range_l215_215808

theorem digit_counts_in_range (a b : Nat) :
  ∑ k in Finset.Icc a b, (if k % 10 = 3 then 1 else 0 + if k / 10 % 10 = 3 then 1 else 0) = 20 ∧
  ∑ k in Finset.Icc a b, (if k % 10 = 7 then 1 else 0 + if k / 10 % 10 = 7 then 1 else 0) = 20 ∧
  ∑ k in Finset.Icc a b, (if k % 10 = 3 then 1 else 0 + if k / 10 % 10 = 3 then 1 else 0 +
                         if k % 10 = 7 then 1 else 0 + if k / 10 % 10 = 7 then 1 else 0) = 40 :=
by
  sorry

end digit_counts_in_range_l215_215808


namespace expected_value_of_winnings_is_52_06_l215_215404

noncomputable def expected_value_of_winnings : ℝ :=
  let p : ℕ → ℝ := λ s, match s with
    | 2 | 12   => 1/36
    | 3 | 11   => 2/36
    | 4 | 10   => 3/36
    | 5 | 9    => 4/36
    | 6 | 8    => 5/36
    | 7        => 6/36
    | _        => 0
  in
  ∑ s in finset.range 13, p s * (s ^ 2)

theorem expected_value_of_winnings_is_52_06 : expected_value_of_winnings ≈ 52.06 := by
  sorry

end expected_value_of_winnings_is_52_06_l215_215404


namespace determine_b16_l215_215075

noncomputable def g (z : ℂ) (b : ℕ → ℕ) : ℂ :=
(1 - z)^b 1 * (1 - z^2)^b 2 * (1 - z^3)^b 3 * (1 - z^16)^b 16

theorem determine_b16 (b : ℕ → ℕ) :
  (g z b) ≡ (1 - 3 * z) [SOL mod z^17] → b 16 = 3 * 2^11 - 3 := 
sorry

end determine_b16_l215_215075


namespace ellipse_standard_equation_and_triangle_area_l215_215288

theorem ellipse_standard_equation_and_triangle_area (
    (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
    (B1 : ℝ × ℝ) (hB1 : B1 = (0, -sqrt 3))
    (F : ℝ × ℝ) (hF : angle 0 F B1 = π/3)
    (M : ℝ × ℝ) (hM : on_ellipse M a b) -- condition that M is on the ellipse
    (N : ℝ × ℝ) (hN : N = (M.1 / a, M.2 / b)) -- N is the "ellipse point" of M
    (k m : ℝ) 
    (A B : ℝ × ℝ) (line_l_hits_ellipse : on_line_l k m A ∧ on_line_l k m B)
    (P : ℝ × ℝ) (hP : P = (A.1 / a, A.2 / b))
    (Q : ℝ × ℝ) (hQ : Q = (B.1 / a, B.2 / b))
    (circle_PQ_passes_origin : circle_through_origin P Q)) :
  (∃ a b : ℝ, a = 2 ∧ b = sqrt 3 ∧ (C : ℝ × ℝ)) [hC : standard_ellipse_equation C a b] ∧
  area_triangle (0,0) A B = sqrt 3 := 
sorry

end ellipse_standard_equation_and_triangle_area_l215_215288


namespace new_commission_rate_l215_215711

theorem new_commission_rate (C1 : ℝ) (slump : ℝ) : C2 : ℝ :=
  assume (h1 : C1 = 0.04)
  (h2 : slump = 0.20000000000000007),
  have h3: C2 = (C1 / (1 - slump)), from
  sorry,
  show C2 = 0.05, by sorry

end new_commission_rate_l215_215711


namespace sum_of_roots_l215_215033

theorem sum_of_roots (a b c : ℝ) (h_eq : a = 1 ∧ b = -4 ∧ c = 3) :
  let roots := sum_of_roots_quadratic (x^2 + b * x + c) in
  roots = 4 := 
begin 
  sorry
end

end sum_of_roots_l215_215033


namespace angle_equality_l215_215193

/-
Given an angle ∠PAQ with vertex A and a point M inside this angle.
Let MP and MQ be perpendiculars from point M to AP and AQ respectively.
Let AK be a perpendicular from point A to segment PQ.
Prove that ∠PAK = ∠MAQ.
-/

theorem angle_equality
  (A P Q M K : Point)
  (h1 : IsInsideAngle A P Q M)
  (h2 : Perpendicular MP A P M)
  (h3 : Perpendicular MQ A Q M)
  (h4 : Perpendicular AK A P Q K) :
  ∠PAK = ∠MAQ :=
sorry

end angle_equality_l215_215193


namespace mr_lee_harvested_apples_l215_215661

theorem mr_lee_harvested_apples :
  let number_of_baskets := 19
  let apples_per_basket := 25
  (number_of_baskets * apples_per_basket = 475) :=
by
  sorry

end mr_lee_harvested_apples_l215_215661


namespace S_10_value_l215_215637

noncomputable def a_sequence (n : ℕ) : ℕ → ℝ
| 0 := 0
| (n + 1) :=
  if (n + 1) % 2 = 0 then 2 ^ (n + 2) / 3
  else 0

def S (n : ℕ) : ℝ := (Finset.range n).sum (λ i, a_sequence (i + 1))

theorem S_10_value : S 10 = 2728 / 3 := 
sorry

end S_10_value_l215_215637


namespace quadratic_passing_origin_l215_215718

theorem quadratic_passing_origin (a b c : ℝ) (h : a ≠ 0) :
  ((∀ x y : ℝ, x = 0 → y = 0 → y = a * x^2 + b * x + c) ↔ c = 0) := 
by
  sorry

end quadratic_passing_origin_l215_215718


namespace alternating_even_binomial_sum_l215_215179

theorem alternating_even_binomial_sum :
  (∑ k in finset.Ico 0 51, (-1)^k * nat.choose 101 (2*k)) = -2^50 :=
by
  sorry

end alternating_even_binomial_sum_l215_215179


namespace solve_equation_l215_215365

theorem solve_equation (x : ℝ) (h_nonzero : Real.sin x ≠ 0) :
  2 * Real.cot x ^ 2 * Real.cos x ^ 2 + 4 * Real.cos x ^ 2 - Real.cot x ^ 2 - 2 = 0 →
  ∃ k : ℤ, x = (Int.ofNat k + 1) * (π / 4) := sorry

end solve_equation_l215_215365


namespace hyperbola_asymptote_foci_distance_l215_215505

theorem hyperbola_asymptote_foci_distance {a b : ℝ} (h_a : a > 0) (h_b : b > 0)
  (P : ℝ × ℝ) (h_P : ∃ x y, x / a = y / b ∧ P = (x, y)) :
  let F1 := (c, 0) in
  let F2 := (-c, 0) in
  let c := real.sqrt (a^2 + b^2) in
  abs ((dist P F1) - (dist P F2)) < 2 * a :=
sorry

end hyperbola_asymptote_foci_distance_l215_215505


namespace max_possible_salary_l215_215461

theorem max_possible_salary (n : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ) (team_size : ℕ) 
  (h_team_size : team_size = 25) 
  (h_min_salary : min_salary = 15000) 
  (h_total_salary_cap : total_salary_cap = 800000) :
  ∃ max_salary : ℕ, max_salary = 440000 := 
by
  let total_min_salary_24 := 24 * min_salary
  have h_total_min_salary_24 : total_min_salary_24 = 24 * min_salary := rfl
  let remaining_budget := total_salary_cap - total_min_salary_24
  have h_remaining_budget : remaining_budget = total_salary_cap - total_min_salary_24 := rfl
  have h_max_salary : ∃ max_salary : ℕ, max_salary = remaining_budget := 
    by use remaining_budget; exact h_remaining_budget
  exact h_max_salary

end max_possible_salary_l215_215461


namespace point_distance_l215_215842

theorem point_distance (x y n : ℝ) 
    (h1 : abs x = 8) 
    (h2 : (x - 3)^2 + (y - 10)^2 = 225) 
    (h3 : y > 10) 
    (hn : n = Real.sqrt (x^2 + y^2)) : 
    n = Real.sqrt (364 + 200 * Real.sqrt 2) := 
sorry

end point_distance_l215_215842


namespace value_of_business_l215_215446

variable (business_value : ℝ) -- We are looking for the value of the business
variable (man_ownership_fraction : ℝ := 2/3) -- The fraction of the business the man owns
variable (sale_fraction : ℝ := 3/4) -- The fraction of the man's shares that were sold
variable (sale_amount : ℝ := 6500) -- The amount for which the fraction of the shares were sold

-- The main theorem we are trying to prove
theorem value_of_business (h1 : man_ownership_fraction = 2/3) (h2 : sale_fraction = 3/4) (h3 : sale_amount = 6500) :
    business_value = 39000 := 
sorry

end value_of_business_l215_215446


namespace probability_divisible_by_three_l215_215266

noncomputable def prob_divisible_by_three : ℚ :=
  1 - (4/6)^6

theorem probability_divisible_by_three :
  prob_divisible_by_three = 665 / 729 :=
by
  sorry

end probability_divisible_by_three_l215_215266


namespace perpendicular_lines_solve_a_l215_215942

theorem perpendicular_lines_solve_a (a : ℝ) :
  (3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0 → a = 0 ∨ a = 12 / 11 :=
by 
  sorry

end perpendicular_lines_solve_a_l215_215942


namespace area_of_triangle_90deg_area_of_triangle_60deg_area_of_triangle_120deg_l215_215935

def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2) - (y^2 / b^2) = 1

-- Definitions for the coordinates of foci based on hyperbola equation x^2 / a^2 - y^2 / b^2 = 1
def foci_coordinates (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((a, 0), (-a, 0))

-- Definition of the area of triangle F1 M F2 based on the angle and sides
def area_of_triangle (b θ : ℝ) : ℝ := 
  b^2 * Real.cot(θ / 2)

-- Proof statement for the specific angles given in the problem
theorem area_of_triangle_90deg (a b : ℝ) (h : hyperbola_equation a b) :
  area_of_triangle b (Real.pi / 2) = 9 :=
sorry

theorem area_of_triangle_60deg (a b : ℝ) (h : hyperbola_equation a b) :
  area_of_triangle b (Real.pi / 3) = 9 * Real.sqrt 3 :=
sorry

theorem area_of_triangle_120deg (a b : ℝ) (h : hyperbola_equation a b) :
  area_of_triangle b (2 * Real.pi / 3) = 3 * Real.sqrt 3 :=
sorry

end area_of_triangle_90deg_area_of_triangle_60deg_area_of_triangle_120deg_l215_215935


namespace smallest_number_is_a_l215_215822

noncomputable def a : ℝ := -2
noncomputable def b : ℝ := 2
noncomputable def c : ℝ := -1 / 2
noncomputable def d : ℝ := 1 / 2

theorem smallest_number_is_a :
  ∀ x ∈ {a, b, c, d}, a ≤ x :=
by {
  intro x,
  intro hx,
  finish,
  sorry
}

end smallest_number_is_a_l215_215822


namespace no_hamiltonian_cycle_in_any_convex_polyhedron_l215_215993

theorem no_hamiltonian_cycle_in_any_convex_polyhedron (P : ConvexPolyhedron) : 
  ¬ (∃ (cycle : List P.Vertex), ∀ v ∈ cycle, v ∈ P.Vertices ∧ (∀ (e ∈ cycle.adjacentEdges), e ∈ P.Edges) ∧ cycle.head = cycle.tail ∧ cycle.length = P.Vertices.length) :=
sorry

end no_hamiltonian_cycle_in_any_convex_polyhedron_l215_215993


namespace amoeba_after_ten_days_l215_215809

def amoeba_count (n : ℕ) : ℕ := 
  3^n

theorem amoeba_after_ten_days : amoeba_count 10 = 59049 := 
by
  -- proof omitted
  sorry

end amoeba_after_ten_days_l215_215809


namespace cup_order_l215_215306

noncomputable def volume_of_cylinder (r : ℝ) (h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1/3) * π * r^2 * h

noncomputable def volume_of_hemisphere (r : ℝ) : ℝ :=
  (2/3) * π * r^3

theorem cup_order {r : ℝ} (r_pos : r = 1) (V : ℝ) :
  (volume_of_cylinder r (2/3) = V) →
  (volume_of_cone r 2 = V) →
  (volume_of_hemisphere r = V) →
  (∀ ha hb hc, (ha = (2/3)) → (hb = 2) → (hc = 1) → [ha, hc, hb] = [2/3, 1, 2]) :=
by 
  intros
  sorry

end cup_order_l215_215306


namespace largest_y_coordinate_l215_215496

theorem largest_y_coordinate : 
  ∀ (x y : ℝ), (x^2 / 49 + (y - 3)^2 / 25 + y = 0) → 
  y ≤ ( -19 + real.sqrt 325 ) / 2 :=
begin
  sorry
end

end largest_y_coordinate_l215_215496


namespace largest_prime_factor_of_expression_l215_215037

theorem largest_prime_factor_of_expression :
  ∃ p : ℕ, prime p ∧ p = 71 ∧ (∀ q : ℕ, prime q ∧ q ∣ (16^4 + 2 * 16^2 + 1 - 13^4) → q ≤ p) := by
  sorry

end largest_prime_factor_of_expression_l215_215037


namespace yellow_block_heavier_than_green_l215_215628

theorem yellow_block_heavier_than_green :
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  yellow_block_weight - green_block_weight = 0.2 := by
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  show yellow_block_weight - green_block_weight = 0.2
  sorry

end yellow_block_heavier_than_green_l215_215628


namespace slope_of_min_chord_length_line_l215_215444

theorem slope_of_min_chord_length_line
  {l : ℝ → ℝ}
  (hl : ∀ x y, y = l x ↔ y = 1 * x + 1)
  (h_line_through_point : l 0 = 1)
  (h_intersects_circle : ∃ x1 x2, (l x1, x1) ≠ (l x2, x2) ∧ (x1 - 1)^2 + (l x1)^2 = 4 ∧ (x2 - 1)^2 + (l x2)^2 = 4) :
  (∀ m : ℝ, l = λ x, m * x + b → m = 1) :=
by
  sorry

end slope_of_min_chord_length_line_l215_215444


namespace part_a_l215_215048

theorem part_a (n : ℕ) (A : Fin n → ℝ × ℝ) : ∃ O : ℝ × ℝ, ∀ (l : ℝ × ℝ) (H : O = l), 
  ∃ (left_half_points right_half_points : set (ℝ × ℝ)), 
    |left_half_points| ≥ n / 3 ∧ |right_half_points| ≥ n / 3 :=
sorry

end part_a_l215_215048


namespace exists_polyhedron_volume_with_no_points_l215_215299

theorem exists_polyhedron_volume_with_no_points {V : ℝ} (n : ℕ) (points : set (euclidean_space ℝ)) 
  (hV : V > 0) (hpoints_card : points.card = 3 * (2^n - 1)) (convex_polyhedron : euclidean_space ℝ → Prop) 
  (hconvex : ∀ (x y : euclidean_space ℝ), convex_polyhedron x → convex_polyhedron y → 
    ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → convex_polyhedron ((1 - t) • x + t • y)) : 
  ∃ (sub_poly : euclidean_space ℝ → Prop), 
      (∃ (subV : ℝ), subV = V / 2^n ∧ 
      ∀ (p : euclidean_space ℝ), sub_poly p → convex_polyhedron p ∧ ¬points p) := 
sorry

end exists_polyhedron_volume_with_no_points_l215_215299


namespace inverse_undefined_at_one_l215_215260

def f (x : ℝ) : ℝ := (x - 3) / (x - 4)

theorem inverse_undefined_at_one : ¬ ∃ y : ℝ, (f y) = 1 :=
by sorry

end inverse_undefined_at_one_l215_215260


namespace A_O_H_collinear_l215_215616

open EuclideanGeometry

variables {A B C E F O H : Point} (triangle_ABC : Triangle A B C) 

def is_acute_angled (T : Triangle A B C) : Prop := 
  ∀ θ ∈ T.angles, θ < π / 2

def circle_diameter (P Q : Point) : Circle := sorry -- Assume we have a function creating a circle with P and Q as diameter points

structure is_circumcenter (O : Point) (T : Triangle) : Prop :=
  (circumcenter_property : ∀ P ∈ T.vertices, dist O P = dist O (circumcenter T))

noncomputable def orthocenter (T : Triangle) : Point := sorry -- Assume we have a way to get orthocenter

noncomputable def collinear (A B C : Point) : Prop := sorry -- Assume we have a way to check collinearity

theorem A_O_H_collinear (h_acute : is_acute_angled triangle_ABC) 
    (h_Γ1 : ∃ Γ_1, circle_diameter A B = Γ_1 ∧ E ∈ Γ_1)
    (h_Γ2 : ∃ Γ_2, circle_diameter A C = Γ_2 ∧ F ∈ Γ_2)
    (h_H : H = orthocenter triangle_ABC)
    (h_O : is_circumcenter O (Triangle A E F)) :
  collinear A O H :=
sorry

end A_O_H_collinear_l215_215616


namespace count_true_statements_l215_215244

def are_conjugate (z1 z2 : ℂ) : Prop :=
  z1 = complex.conj z2

def original_statement (z1 z2 : ℂ) : Prop :=
  are_conjugate z1 z2 → z1 * z2 = complex.abs z1 ^ 2

def converse_statement (z1 z2 : ℂ) : Prop :=
  z1 * z2 = complex.abs z1 ^ 2 → are_conjugate z1 z2

def inverse_statement (z1 z2 : ℂ) : Prop :=
  ¬(are_conjugate z1 z2) → z1 * z2 ≠ complex.abs z1 ^ 2

def contrapositive_statement (z1 z2 : ℂ) : Prop :=
  z1 * z2 ≠ complex.abs z1 ^ 2 → ¬(are_conjugate z1 z2)

theorem count_true_statements (z1 z2 : ℂ) :
  original_statement z1 z2 →
  (cond : (converse_statement z1 z2 = false ∧
           inverse_statement z1 z2 = false ∧
           contrapositive_statement z1 z2 = true)) →
  cond ∧ ((converse_statement z1 z2 ∧ inverse_statement z1 z2) → false) :=
begin
  sorry
end

end count_true_statements_l215_215244


namespace range_of_q_l215_215574

theorem range_of_q (q : ℝ) (hq : q < 0) :
  (∀ m n : ℕ, m > 0 → n > 0 → (2 * q^m + q) / (2 * q^n + q) ∈ Ioo (1 / 6 : ℝ) 6) →
  q ∈ Ioo (-1 / 4 : ℝ) 0 :=
by
  sorry

end range_of_q_l215_215574


namespace factorial_division_l215_215749

theorem factorial_division :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 := by
  sorry

end factorial_division_l215_215749


namespace eleven_step_paths_l215_215844

def H : (ℕ × ℕ) := (0, 0)
def K : (ℕ × ℕ) := (4, 3)
def J : (ℕ × ℕ) := (6, 5)

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem eleven_step_paths (H K J : (ℕ × ℕ)) (H_coords : H = (0, 0)) (K_coords : K = (4, 3)) (J_coords : J = (6, 5)) : 
  (binomial 7 4) * (binomial 4 2) = 210 := by 
  sorry

end eleven_step_paths_l215_215844


namespace rectangles_diagonals_equal_not_rhombuses_l215_215097

/--
A proof that the property of having diagonals of equal length is a characteristic of rectangles
and not necessarily of rhombuses.
-/
theorem rectangles_diagonals_equal_not_rhombuses (R : Type) [rectangle R] (H : Type) [rhombus H] :
  (∀ r : R, diagonals_equal r) ∧ ¬(∀ h : H, diagonals_equal h) :=
sorry

end rectangles_diagonals_equal_not_rhombuses_l215_215097


namespace min_rungs_l215_215109

theorem min_rungs (a b : ℕ) (h_a : a > 0) (h_b : b > 0) :
  ∃ (n : ℕ), (∀ x y : ℤ, a*x - b*y = n) ∧ n = a + b - Nat.gcd a b := 
sorry

end min_rungs_l215_215109


namespace exist_100_noncoverable_triangles_l215_215154

theorem exist_100_noncoverable_triangles :
  ∃ (T : Fin 100 → Triangle), (∀ i j : Fin 100, i ≠ j → ¬ (T i ⊆ T j)) ∧
  (∀ i : Fin 99, height (T (i + 1)) = 200 * diameter (T i) ∧ area (T (i + 1)) = area (T i) / 20000) :=
sorry

end exist_100_noncoverable_triangles_l215_215154


namespace spent_on_board_game_l215_215511

theorem spent_on_board_game (b : ℕ)
  (h1 : 4 * 7 = 28)
  (h2 : b + 28 = 30) : 
  b = 2 := 
sorry

end spent_on_board_game_l215_215511


namespace tank_filling_time_l215_215669

theorem tank_filling_time (p q r s : ℝ) (leakage : ℝ) :
  (p = 1 / 6) →
  (q = 1 / 12) →
  (r = 1 / 24) →
  (s = 1 / 18) →
  (leakage = -1 / 48) →
  (1 / (p + q + r + s + leakage) = 48 / 15.67) :=
by
  intros hp hq hr hs hleak
  rw [hp, hq, hr, hs, hleak]
  norm_num
  sorry

end tank_filling_time_l215_215669


namespace sqrt_square_l215_215833

theorem sqrt_square (n : ℝ) : (Real.sqrt 2023) ^ 2 = 2023 :=
by
  sorry

end sqrt_square_l215_215833


namespace sum_eq_17411_l215_215227

def a (n : ℕ) : ℕ := 2 * n - 1
def b (n : ℕ) : ℕ := 2 ^ (n - 1)

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k, a (k + 1) * b (k + 1))

theorem sum_eq_17411 : S 10 = 17411 := 
by
  sorry

end sum_eq_17411_l215_215227


namespace four_digit_numbers_count_l215_215602

theorem four_digit_numbers_count :
  ∀ (a b c d : ℕ), (a ∈ {1, 2, 3, 4}) → (b ∈ {1, 2, 3, 4}) → (c ∈ {1, 2, 3, 4}) → (d ∈ {1, 2, 3, 4}) →
  a ≠ b → b ≠ c → c ≠ d → d ≠ a → a < b ∧ a < c ∧ a < d →
  (∃ s : ℕ, s = 24) :=
by
  sorry

end four_digit_numbers_count_l215_215602


namespace mark_total_spending_l215_215338

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end mark_total_spending_l215_215338


namespace constant_term_in_binomial_expansion_is_correct_l215_215022

theorem constant_term_in_binomial_expansion_is_correct :
  (∃ (n m : ℕ), n + m = 10 ∧ 
    (∃ (k : ℕ), k = binomial 10 m ∧ 
      ∃ (a : ℕ), a = 3^m ∧
        k * a = 17010
    ) ∧ 
    (n / 2 = m)
  ) :=
by
  sorry

end constant_term_in_binomial_expansion_is_correct_l215_215022


namespace cost_price_is_1500_total_profit_is_12000_l215_215065

-- Define the original price, discount rate, and profit margin as constants
def original_price : ℝ := 1800
def discount_rate : ℝ := 0.1
def profit_margin : ℝ := 0.08
def units_sold : ℕ := 100

-- Calculate the discounted selling price
def selling_price : ℝ := original_price * (1 - discount_rate)

-- Theorem for part 1: cost price of the item
theorem cost_price_is_1500 : 
  ∃ x : ℝ, (1 + profit_margin) * x = selling_price ∧ x = 1500 := 
sorry

-- Assumption based on part 1's result
axiom cost_price : ℝ := 1500

-- Theorem for part 2: total profit for 100 units
theorem total_profit_is_12000 : 
  units_sold * (selling_price - cost_price) = 12000 :=
sorry

end cost_price_is_1500_total_profit_is_12000_l215_215065


namespace smallest_n_for_sum_or_difference_divisible_l215_215530

theorem smallest_n_for_sum_or_difference_divisible (n : ℕ) :
  (∃ n : ℕ, ∀ (S : Finset ℤ), S.card = n → (∃ (x y : ℤ) (h₁ : x ≠ y), ((x + y) % 1991 = 0) ∨ ((x - y) % 1991 = 0))) ↔ n = 997 :=
sorry

end smallest_n_for_sum_or_difference_divisible_l215_215530


namespace students_in_both_clubs_l215_215112

theorem students_in_both_clubs
  (T R B total_club_students : ℕ)
  (hT : T = 85) (hR : R = 120)
  (hTotal : T + R - B = total_club_students)
  (hTotalVal : total_club_students = 180) :
  B = 25 :=
by
  -- Placeholder for proof
  sorry

end students_in_both_clubs_l215_215112


namespace dark_squares_exceed_light_by_eight_l215_215757

theorem dark_squares_exceed_light_by_eight : 
  ∀ (m n : ℕ) (h_m : m = 8) (h_n : n = 9), 
  ((m * (n / 2 + 1 / 2 - 1)) + ((m - 1) * (n / 2 + 1 / 2))) - 
  ((m * (n / 2) + ((m - 1) * (n / 2)) = 8
𝑚) :=
by
  intro m n h_m h_n
  sorry

end dark_squares_exceed_light_by_eight_l215_215757


namespace probability_of_both_selected_l215_215767

variable (P_X P_Y P_X_and_Y : ℝ)

-- Conditions
axiom prob_X : P_X = 1/3
axiom prob_Y : P_Y = 2/5

-- The statement of the problem
theorem probability_of_both_selected : P_X_and_Y = 2/15 := by
  -- Definitions from the conditions
  have hPX : P_X = 1/3 := prob_X
  have hPY : P_Y = 2/5 := prob_Y
  -- The result follows from the given conditions
  sorry

end probability_of_both_selected_l215_215767


namespace solve_trig_eq_l215_215685

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end solve_trig_eq_l215_215685


namespace distinguishing_property_of_rectangles_l215_215102

theorem distinguishing_property_of_rectangles (rect rhomb : Type)
  [quadrilateral rect] [quadrilateral rhomb]
  (sum_of_interior_angles_rect : interior_angle_sum rect = 360)
  (sum_of_interior_angles_rhomb : interior_angle_sum rhomb = 360)
  (diagonals_bisect_each_other_rect : diagonals_bisect_each_other rect)
  (diagonals_bisect_each_other_rhomb : diagonals_bisect_each_other rhomb)
  (diagonals_equal_length_rect : diagonals_equal_length rect)
  (diagonals_perpendicular_rhomb : diagonals_perpendicular rhomb) :
  distinguish_property rect rhomb := by
  sorry

end distinguishing_property_of_rectangles_l215_215102


namespace smallest_points_2016_l215_215529

theorem smallest_points_2016 (n : ℕ) :
  n = 28225 →
  ∀ (points : Fin n → (ℤ × ℤ)),
  ∃ i j : Fin n, i ≠ j ∧
    let dist_sq := (points i).fst - (points j).fst ^ 2 + (points i).snd - (points j).snd ^ 2 
    ∃ k : ℤ, dist_sq = 2016 * k :=
by
  intro h points
  sorry

end smallest_points_2016_l215_215529


namespace solve_inequality_l215_215197

theorem solve_inequality (a b x : ℝ) (h : a ≠ b) :
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

end solve_inequality_l215_215197


namespace solution_set_l215_215527

theorem solution_set {x : ℝ} :
  abs ((7 - x) / 4) < 3 ∧ 0 ≤ x ↔ 0 ≤ x ∧ x < 19 :=
by
  sorry

end solution_set_l215_215527


namespace hyperbola_equation_l215_215615

theorem hyperbola_equation
  (a b : ℝ)
  (hyp_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (asymptote1 : ∀ x : ℝ, y = 2 * x)
  (asymptote2 : ∀ x : ℝ, y = -2 * x)
  (parabola_focus : ∀ x y : ℝ, y^2 = 4 * x → x = 1 ∧ y = 0)
  (λ : ℝ)
  (through_focus : hyp_eq 1 0) : 
  ∀ x y : ℝ, x^2 - (y^2 / 4) = 1 :=
by
  sorry

end hyperbola_equation_l215_215615


namespace exponent_m_n_add_l215_215259

variable (a : ℝ) (m n : ℕ)

theorem exponent_m_n_add (h1 : a ^ m = 2) (h2 : a ^ n = 3) : a ^ (m + n) = 6 := by
  sorry

end exponent_m_n_add_l215_215259


namespace find_tan_R_l215_215991

variable (P Q R : ℝ)
variable (cot P cot Q cot R : ℝ)
variable (tan P tan Q tan R : ℝ)

-- Assuming given conditions
axiom cotP_cotR: cot P * cot R = 1
axiom cotQ_cotR: cot Q * cot R = 1/8
axiom sum_angles: P + Q + R = 180

-- Main statement (Goal)
theorem find_tan_R (tanR : ℝ) : tanR = 4 + sqrt 7 := 
by
  have h1 := cotP_cotR
  have h2 := cotQ_cotR
  have h3 := sum_angles
  sorry

end find_tan_R_l215_215991


namespace unique_real_solution_l215_215504

def equation (x : ℝ) : Prop :=
  (x ^ 2010 + 2) * (x ^ 2008 + x ^ 2006 + x ^ 2004 + ... + x ^ 4 + x ^ 2 + 1) = 2010 * x ^ 2009

theorem unique_real_solution : ∃! x : ℝ, equation x := 
sorry

end unique_real_solution_l215_215504


namespace differential_solution_particular_solution_l215_215172

-- Definitions of the initial conditions
def y : ℝ → ℝ := λ x, e^(2 * x)
def y' : ℝ → ℝ := λ x, 2 * e^(2 * x)
def y'' : ℝ → ℝ := λ x, 4 * e^(2 * x)

-- Initial conditions
lemma initial_conditions : y 0 = 1 ∧ y' 0 = 2 := 
by {
  split,
  show y 0 = 1, from rfl,
  show y' 0 = 2, from rfl,
}

-- Theorem stating the solution satisfies the differential equation
theorem differential_solution (x : ℝ) : y'' x - 5 * y' x + 6 * y x = 0 :=
by {
  rw [y, y', y''],
  simp,
  ring,
}

-- Proof that the given y is a solution given the initial conditions
theorem particular_solution (y : ℝ → ℝ) (y' : ℝ → ℝ) (y'' : ℝ → ℝ) 
  (h₀ : y 0 = 1) (h₁ : y' 0 = 2) : 
  ∀ x, y x = e^(2 * x) := sorry

end differential_solution_particular_solution_l215_215172


namespace find_t_square_l215_215795

def isHyperbola (x y a b : ℝ) : Prop :=
  (y^2 / b^2) - (x^2 / a^2) = 1

def passesThrough (x y a b : ℝ) (p q : ℝ) : Prop := 
  isHyperbola p q a b

theorem find_t_square :
  ∃ t : ℝ, 
    let b := 2 in
    let a := sqrt 3 in
    isHyperbola 0 2 a b ∧
    isHyperbola 3 (-4) a b ∧
    isHyperbola t (-2) a b ∧
    t^2 = 0 :=
begin
  sorry
end

end find_t_square_l215_215795


namespace car_truck_meet_at_tunnel_end_l215_215008

theorem car_truck_meet_at_tunnel_end 
  (leaveBforA : ℕ → ℕ → ℕ → Prop)
  (leaveAforB : ℕ → ℕ → ℕ → Prop)
  (arriveAat : ℕ → ℕ → ℕ → Prop)
  (arriveBat : ℕ → ℕ → ℕ → Prop)
  (leave_B_for_A : leaveBforA 8 16 0)
  (leave_A_for_B : leaveAforB 9 0 0)
  (arrive_A_at : arriveAat 10 56 0)
  (arrive_B_at : arriveBat 12 20 0)
  (truck_leaves_2_minutes : leaveAforB 10 2 (0:ℕ))
  : ∃ t: nat, t = 10 :=
  sorry

end car_truck_meet_at_tunnel_end_l215_215008


namespace failed_by_35_l215_215780

variables (M S P : ℝ)
variables (hM : M = 153.84615384615384)
variables (hS : S = 45)
variables (hP : P = 0.52 * M)

theorem failed_by_35 (hM : M = 153.84615384615384) (hS : S = 45) (hP : P = 0.52 * M) : P - S = 35 :=
by
  sorry

end failed_by_35_l215_215780


namespace angle_difference_invariant_l215_215143

-- Define the midpoint M of side BC in an isosceles triangle ABC with AC = AB
variable (A B C M X T : Point)
variable [IsoscelesTriangle ABC (eq.refl _)]

-- Assume M is the midpoint of side BC
variable (mbc : midpoint M B C)

-- Assume X is on the smaller arc over arc MA of the circumcircle of triangle ABM
variable (circ_ABM : Circle A B M)
variable (on_arc_MA : onSmallerArc X M A circ_ABM)

-- Assume T is such that ∠TMX = 90° and TX = BX
variable (tmx_90 : angle T M X = 90)
variable (tx_bx : TX = BX)

-- Proof problem statement
theorem angle_difference_invariant :
  angle T M B - angle C T M = angle M A B :=
sorry

end angle_difference_invariant_l215_215143


namespace negation_of_statement_l215_215386

theorem negation_of_statement :
  ¬ (∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 := by
  sorry

end negation_of_statement_l215_215386


namespace bugs_meet_at_S_with_QS_12_l215_215011

-- Define the points and distances in the triangle
variables (P Q R S : Type) [MetricSpace P]
variables (PQ QR RP QS : ℝ)
variables (v : ℝ) (v_pos : v > 0)

-- Conditions
def PQ_len := PQ = 8
def QR_len := QR = 10
def RP_len := RP = 12

-- Prove that the bugs meet at S such that QS = 12
theorem bugs_meet_at_S_with_QS_12
  (triangle_PQR : Triangle P Q R)
  (start_P : True)
  (speed_v : Real)
  (speed_2v : Real)
  (first_meet : (distance_at_first_meet : ℝ) (distance travelled : ℝ) (distance_at_first_meet_sum := distance travelled + distance travelled)
  (meet_at_S := yes at point true first_meet meet_at_S)
  (QS_meet : distance first_meet first meet)
  (sum_of_speeds : distance travelled speed_v.speed_2v)))

  (total_distance := RP + RP + QP = 30)
  (QP_distance_ratio := speed_faster := == travelling_total distance)

  (meeting_pt_S : RP meeting_pt_S := QS := distance : distance == RP)
  (speed_v travelled_first distance))
  (total_distance distance := travel := speed travelled_true_total := QR)
  (speed_distance := meeting at meeting := RP distance.)

(sorry) =:= 12

(End_theory)

end bugs_meet_at_S_with_QS_12_l215_215011


namespace surface_area_of_circumscribing_sphere_l215_215559

theorem surface_area_of_circumscribing_sphere :
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  A = 17 * Real.pi :=
by
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  show A = 17 * Real.pi
  sorry

end surface_area_of_circumscribing_sphere_l215_215559


namespace no_real_solution_l215_215949

theorem no_real_solution (x : ℝ) : ¬ ∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -|x| := 
sorry

end no_real_solution_l215_215949


namespace count_three_digit_numbers_l215_215951

theorem count_three_digit_numbers : 
  ∃ n : ℕ, 
    (∀ (hundreds tens units : ℕ), 
      hundreds ∈ {1, 2, 3, 4} ∧ tens ∈ {1, 2, 3, 4} ∧ units ∈ {1, 2, 3, 4} ∧ 
      hundreds ≠ tens ∧ tens ≠ units ∧ hundreds ≠ units 
    → n = 24) :=
sorry

end count_three_digit_numbers_l215_215951


namespace find_n_times_s_l215_215654

theorem find_n_times_s :
  (∃ (f : ℕ → ℕ), (∀ a b : ℕ, 3 * f (a^2 + b^2) = f a ^ 2 + 2 * f b ^ 2) ∧
    let vals := {x // ∃ a b : ℕ, f (a^2 + b^2) = x ∧ a^2 + b^2 = 16},
    let n := vals.to_finset.card,
    let s := vals.to_finset.sum (λ x, x.val)
in n * s = 2) :=
sorry

end find_n_times_s_l215_215654


namespace inequality_solution_set_l215_215229

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : (a - 1) * x > 2) : x < 2 / (a - 1) ↔ a < 1 :=
by
  sorry

end inequality_solution_set_l215_215229


namespace initial_avg_weight_l215_215396

theorem initial_avg_weight (A : ℝ) (h : 6 * A + 121 = 7 * 151) : A = 156 :=
by
sorry

end initial_avg_weight_l215_215396


namespace find_f_of_three_halves_l215_215643

noncomputable def g (x : ℝ) : ℝ := 1 - x^2 + real.sqrt x

noncomputable def f (x : ℝ) : ℝ := (1 - x^2 + real.sqrt x) / x^2

theorem find_f_of_three_halves : f (3/2) = ? := 
by sorry

end find_f_of_three_halves_l215_215643


namespace five_digit_positive_integers_count_l215_215585

theorem five_digit_positive_integers_count : 
  ∃ n : ℕ, n = 500 ∧ (∀ x : ℕ, 10000 ≤ x ∧ x ≤ 99999 → 
  ∀ d : ℕ, (d ∈ [1, 3, 5, 7, 9]) → 
  (x % 5 = 0) ∧ (x % 3 = 0) →
  count_odd_digit_div3_and_5 x = n) :=
sorry

end five_digit_positive_integers_count_l215_215585


namespace maximum_profit_l215_215781

noncomputable def fixed_cost : ℕ := 20

def additional_cost (x : ℝ) : ℝ :=
  if (0 < x) ∧ (x < 100) then (1/2) * x^2 + 10 * x + 1100
  else if x ≥ 100 then 120 * x + 4500 / (x - 90) - 5400
  else 0

def selling_price : ℝ := 100

def profit (x : ℝ) : ℝ :=
  selling_price * x - (fixed_cost + additional_cost(x))

theorem maximum_profit : ∃ (x : ℝ), x = 105 ∧ profit x = 1000 :=
by {
  -- skipping the proof with sorry
  sorry
}

end maximum_profit_l215_215781


namespace nim_zero_sum_nonzero_result_l215_215709

theorem nim_zero_sum_nonzero_result (m : List ℕ) (h : List.nimSum m = 0) (i : ℕ) (hi : i < m.length) (k : ℕ) (hk : 0 < k ∧ k ≤ m[i]) :
    List.nimSum (m.set i (m[i] - k)) ≠ 0 := 
sorry

end nim_zero_sum_nonzero_result_l215_215709


namespace cost_price_per_meter_l215_215762

-- Conditions
variable (total_meters : ℕ) (total_selling_price loss_per_meter : ℝ) (C : ℝ)
hypothesis h1 : total_meters = 450
hypothesis h2 : total_selling_price = 18000
hypothesis h3 : loss_per_meter = 5

-- Calculation of selling price per meter
def selling_price_per_meter : ℝ := total_selling_price / total_meters

-- The main hypothesis about loss per meter and cost price
hypothesis h4 : C - loss_per_meter = selling_price_per_meter

-- The proposition to prove
theorem cost_price_per_meter : C = 45 :=
by
  sorry

end cost_price_per_meter_l215_215762


namespace BC_eq_2BP_l215_215976

noncomputable theory

-- Define the problem condition
variables {A B C P M D E : Type} [triangle ABC : triangle Type] [point_in_triangle P : point_in_triangle ABC]
variables [M_midpoint_of_BC : midpoint_of M B C] [AP_bisects_BAC : angle_bisector AP BAC]
variables [MP_intersect_circumcircles : intersect_circumcircle MP (circumcircle ABP) (circumcircle ACP) P D E]

-- Define the theorem statement
theorem BC_eq_2BP (h : DE = MP) : BC = 2 * BP :=
begin
  sorry -- proof not required
end

end BC_eq_2BP_l215_215976


namespace rectangle_area_l215_215955

variable (a b : ℝ)

-- Given conditions
axiom h1 : (a + b)^2 = 16 
axiom h2 : (a - b)^2 = 4

-- Objective: Prove that the area of the rectangle ab equals 3
theorem rectangle_area : a * b = 3 := by
  sorry

end rectangle_area_l215_215955


namespace percent_decrease_internet_cost_l215_215375

theorem percent_decrease_internet_cost :
  ∀ (initial_cost final_cost : ℝ), initial_cost = 120 → final_cost = 45 → 
  ((initial_cost - final_cost) / initial_cost) * 100 = 62.5 :=
by
  intros initial_cost final_cost h_initial h_final
  sorry

end percent_decrease_internet_cost_l215_215375


namespace speed_of_womans_train_l215_215817

noncomputable def speed_goods_train_kmh : ℝ := 51.99424046076314
noncomputable def length_goods_train_m : ℝ := 300
noncomputable def time_to_pass_s : ℝ := 15

noncomputable def speed_goods_train_ms : ℝ := speed_goods_train_kmh * 1000 / 3600
noncomputable def relative_speed : ℝ := length_goods_train_m / time_to_pass_s
noncomputable def speed_woman_train_ms : ℝ := relative_speed - speed_goods_train_ms
noncomputable def speed_woman_train_kmh : ℝ := speed_woman_train_ms * 3600 / 1000

theorem speed_of_womans_train :
    speed_woman_train_kmh ≈ 20.038 := 
by
  unfold speed_goods_train_kmh length_goods_train_m time_to_pass_s
  unfold speed_goods_train_ms relative_speed speed_woman_train_ms speed_woman_train_kmh
  sorry

end speed_of_womans_train_l215_215817


namespace acute_triangle_inequality_l215_215611

variable {R : ℝ} (A B C : ℝ) (a b c : ℝ)
variable [Real.lt (0) (R)]
variable [Real.lt (a) (2 * R * Real.sin A)]
variable [Real.lt (b) (2 * R * Real.sin B)]
variable [Real.lt (c) (2 * R * Real.sin C)]

theorem acute_triangle_inequality 
  (h_acute_triangle : Real.lt A (π / 2) ∧ Real.lt B (π / 2) ∧ Real.lt C (π / 2))
  (h_circumradius : R = 1) 
  (h_sides : a = 2 * R * Real.sin A ∧ b = 2 * R * Real.sin B ∧ c = 2 * R * Real.sin C) :
  (a / (1 - Real.sin A) + b / (1 - Real.sin B) + c / (1 - Real.sin C) 
    ≥ 18 + 12 * Real.sqrt 3) :=
  sorry

end acute_triangle_inequality_l215_215611


namespace problem_solution_l215_215548

noncomputable def sequence_a : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 1) := -sequence_a n + 1

noncomputable def sum_S : ℕ → ℕ
| 0       := 0
| (n + 1) := sum_S n + sequence_a (n + 1)

theorem problem_solution : sum_S 2019 = 1010 :=
by {
  sorry
}

end problem_solution_l215_215548


namespace angle_between_vectors_l215_215945

def vector (α : Type) [Field α] := (α × α)

theorem angle_between_vectors
    (a : vector ℝ)
    (b : vector ℝ)
    (ha : a = (4, 0))
    (hb : b = (-1, Real.sqrt 3)) :
  let dot_product (v w : vector ℝ) : ℝ := (v.1 * w.1 + v.2 * w.2)
  let norm (v : vector ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  let cos_theta := dot_product a b / (norm a * norm b)
  ∀ theta, Real.cos theta = cos_theta → theta = 2 * Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l215_215945


namespace sum_of_angles_l215_215845

-- Define the degrees as real numbers
noncomputable def deg := ℝ

-- Define the measures of arcs XY, YZ, and ZX
constant arc_XY : deg := 50
constant arc_YZ : deg := 45
constant arc_ZX : deg := 90

-- Calculate the measures of angle alpha and beta
def angle_alpha : deg := (arc_XY + arc_YZ - arc_YZ) / 2
def angle_beta : deg := arc_YZ / 2

-- The total sum of angles alpha and beta should be 47.5 degrees
theorem sum_of_angles : angle_alpha + angle_beta = 47.5 := by
  sorry

end sum_of_angles_l215_215845


namespace train_pass_time_l215_215042

-- Definitions and conditions
def train_length : ℝ := 250  -- meters
def train_speed : ℝ := 58    -- km/h
def man_speed : ℝ := 8       -- km/h
def relative_speed_kmh : ℝ := train_speed - man_speed
def relative_speed_ms : ℝ := (relative_speed_kmh * 1000) / 3600

-- Statement to prove
theorem train_pass_time : (train_length / relative_speed_ms) = 18 :=
by sorry

end train_pass_time_l215_215042


namespace remainder_when_M_divided_by_45_l215_215315

-- Conditions
def M := 123456789101112…5960  -- This representation means the number formed by writing integers from 1 to 60 consecutively.

-- Theorem statement
theorem remainder_when_M_divided_by_45 : M % 45 = 2 :=
by
  sorry

end remainder_when_M_divided_by_45_l215_215315


namespace rectangles_diagonals_equal_not_rhombuses_l215_215098

/--
A proof that the property of having diagonals of equal length is a characteristic of rectangles
and not necessarily of rhombuses.
-/
theorem rectangles_diagonals_equal_not_rhombuses (R : Type) [rectangle R] (H : Type) [rhombus H] :
  (∀ r : R, diagonals_equal r) ∧ ¬(∀ h : H, diagonals_equal h) :=
sorry

end rectangles_diagonals_equal_not_rhombuses_l215_215098


namespace geometric_sequence_log_sum_l215_215985

theorem geometric_sequence_log_sum (a₁ a₇ : ℕ) 
  (h₁ : a₁ > 0) 
  (h₂ : a₇ > 0) 
  (h₃ : ∃ a₀ r, a₀ > 0 ∧ r > 0 ∧ (∀ n, a_n = a₀ * r ^ n) ∧ (a₁ = a₀ * r) ∧ (a₇ = a₀ * r ^ 6)) 
  (h₄ : ∀ x, x^2 - 32*x + 64 = 0 → (x = a₁ ∨ x = a₇)) :
  (\lg (2 * a₁) + \lg (2 * a₂) + \lg (2 * a₃) + \lg (2 * a₄) + \lg (2 * a₅) + \lg (2 * a₆) + \lg (2 * a₇) + \lg (2 * a₈) + \lg (2 * a₉)) = 36 := by
  sorry

end geometric_sequence_log_sum_l215_215985


namespace maximum_profit_l215_215439

noncomputable def L1 (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def L2 (x : ℝ) : ℝ := 2 * x

theorem maximum_profit :
  (∀ (x1 x2 : ℝ), x1 + x2 = 15 → L1 x1 + L2 x2 ≤ 45.6) := sorry

end maximum_profit_l215_215439


namespace derivative_at_neg_one_l215_215926

noncomputable def f (x : ℝ) : ℝ := x^2 * (x + 1)

theorem derivative_at_neg_one : (derivative f (-1) = 1) :=
by
  have h1 : f x = x^3 + x^2 := sorry
  have h2 : (derivative f) = λ x, 3 * x^2 + 2 * x := sorry
  have h3 : (derivative f) (-1) = 1 := sorry
  exact h3

end derivative_at_neg_one_l215_215926


namespace six_valid_arrangements_wxyz_l215_215591

def is_valid_arrangement (s : List Char) : Prop :=
  (s.length = 4) ∧
  (List.nodup s) ∧
  ¬ ((('w', 'x') ∈ s.zip s.tail) ∨
     (('x', 'y') ∈ s.zip s.tail) ∨
     (('y', 'z') ∈ s.zip s.tail))

theorem six_valid_arrangements_wxyz :
  (List.permutations ['w', 'x', 'y', 'z']).filter is_valid_arrangement).length = 6 :=
by
  sorry

end six_valid_arrangements_wxyz_l215_215591


namespace scientific_notation_of_700_3_l215_215873

theorem scientific_notation_of_700_3 : 
  ∃ (a : ℝ) (n : ℤ), 700.3 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ n = 2 ∧ a = 7.003 :=
by
  use [7.003, 2]
  simp
  sorry

end scientific_notation_of_700_3_l215_215873


namespace price_of_adult_ticket_l215_215071

theorem price_of_adult_ticket (total_payment : ℕ) (child_price : ℕ) (difference : ℕ) (children : ℕ) (adults : ℕ) (A : ℕ)
  (h1 : total_payment = 720) 
  (h2 : child_price = 8) 
  (h3 : difference = 25) 
  (h4 : children = 15)
  (h5 : adults = children + difference)
  (h6 : total_payment = children * child_price + adults * A) :
  A = 15 :=
by
  sorry

end price_of_adult_ticket_l215_215071


namespace find_y_l215_215753

theorem find_y (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_rem : x % y = 3) (h_div : (x:ℝ) / y = 96.15) : y = 20 :=
by
  sorry

end find_y_l215_215753


namespace complement_intersection_l215_215937

open Set

variable (U A B : Set ℕ)

-- Definitions of the sets
def U := {2, 3, 6, 8}
def A := {2, 3}
def B := {2, 6, 8}

theorem complement_intersection :
  (U \ A) ∩ B = {6, 8} :=
sorry

end complement_intersection_l215_215937


namespace discount_equation_l215_215064

variable (P₀ P_f x : ℝ)
variable (h₀ : P₀ = 200)
variable (h₁ : P_f = 164)

theorem discount_equation :
  P₀ * (1 - x)^2 = P_f := by
  sorry

end discount_equation_l215_215064


namespace problem_solution_l215_215565

theorem problem_solution 
  (a b : ℝ)
  (h_eqn : ∀ x : ℝ, 1 < x ∧ x < b → ax^2 - 3*x + 2 < 0)
  : a = 1 ∧ b = 2 ∧ (∀ x : ℝ, 1 < x ∧ x < 2 → (2*a + b)*x - 9/((a - b)*x) ≥ 12) :=
begin
  sorry
end

end problem_solution_l215_215565


namespace problem1_min_alliances_problem2_min_alliances_l215_215367

open Classical

-- Define the problem using Lean definitions
structure AllianceProblem1 where
  countries : Set ℕ
  alliances : Finset (Finset ℕ)
  h1 : countries = (Finset.range 100).to_set
  h2 : ∀ c ∈ countries, ∃ A ∈ alliances, c ∈ A
  h3 : ∀ A ∈ alliances, A.card ≤ 50
  h4 : ∀ c1 c2 ∈ countries, c1 ≠ c2 → ∃ A ∈ alliances, c1 ∈ A ∧ c2 ∈ A

-- Prove the minimum number of alliances required
noncomputable def min_alliances_required : Nat :=
  6

-- Equivalent to Problem 1
theorem problem1_min_alliances (P : AllianceProblem1) :
  P.alliances.card = min_alliances_required :=
sorry

-- Define the problem for Part 2 using Lean definitions
structure AllianceProblem2 extends AllianceProblem1 where
  h5 : ∀ A1 A2 ∈ alliances, (A1 ∪ A2).card ≤ 80

-- Prove the minimum number of alliances with additional requirement
noncomputable def min_alliances_required_with_additional : Nat :=
  6

-- Equivalent to Problem 2
theorem problem2_min_alliances (P : AllianceProblem2) :
  P.alliances.card = min_alliances_required_with_additional :=
sorry

end problem1_min_alliances_problem2_min_alliances_l215_215367


namespace problem_statement_l215_215720

-- Definitions of the arithmetic and geometric sequences
def a (n : ℕ) : ℕ := 2 * n + 1
def b (n : ℕ) : ℕ := 2 ^ (n - 1)
def c (n : ℕ) : ℕ := a n * b n

-- Sum of the first n terms
def S (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i
def T (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), c i

-- Problem statement
theorem problem_statement (n : ℕ) (a₁ : ℕ := 3) (b₁ : ℕ := 1) (b₂ : ℕ := 2) : 
  (b 2 + S 2 = 10) ∧ (a 5 - 2 * b 2 = a 1) →
  (a n = 2 * n + 1) ∧ (b n = 2 ^ (n - 1)) ∧ (T n = (2 * n - 1) * 2^n + 1) :=
  by 
    sorry

end problem_statement_l215_215720


namespace part_I_part_II_l215_215205

def sequence_sn (n : ℕ) : ℚ := (3 / 2 : ℚ) * n^2 + (1 / 2 : ℚ) * n

def sequence_a (n : ℕ) : ℕ := 3 * n - 1

def sequence_b (n : ℕ) : ℚ := (1 / 2 : ℚ)^n

def sequence_C (n : ℕ) : ℚ := sequence_a (sequence_a n) + sequence_b (sequence_a n)

def sum_of_first_n_terms (f : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum f

theorem part_I (n : ℕ) : sequence_a n = 3 * n - 1 ∧ sequence_b n = (1 / 2)^n :=
by {
  sorry
}

theorem part_II (n : ℕ) : sum_of_first_n_terms sequence_C n =
  (n * (9 * n + 1) / 2) - (2 / 7) * (1 / 8)^n + (2 / 7) :=
by {
  sorry
}

end part_I_part_II_l215_215205


namespace instantaneous_velocity_at_1_is_6_l215_215801

noncomputable def s (t : ℝ) : ℝ := 2 * t^3

theorem instantaneous_velocity_at_1_is_6 :
  (deriv s 1 = 6) :=
by
  -- Transforming the expression for derivative
  have h : deriv s t = 6 * t^2, from sorry,
  -- Substituting t = 1 into the derivative expression
  calc
    deriv s 1 = 6 * 1^2 : by rw h
            ... = 6 : by norm_num

end instantaneous_velocity_at_1_is_6_l215_215801


namespace common_root_value_l215_215191

theorem common_root_value (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + 8 = 0 ∧ x^2 + x + a = 0) ↔ a = -6 :=
sorry

end common_root_value_l215_215191


namespace cosine_angle_sum_diff_l215_215562

noncomputable theory
open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions
def angle_ab : ℝ := π / 3
def norm_a : ℝ := 1
def norm_b : ℝ := 2

-- Question and proof (assertion)
theorem cosine_angle_sum_diff :
  real.cos (angle (a + b) (a - b)) = -real.sqrt 21 / 7 :=
by sorry

end cosine_angle_sum_diff_l215_215562


namespace candle_height_correct_l215_215436

def candle_height_after_half_time : ℕ :=
  let T : ℕ := 5 * (100 * 101 * 201) / 6
  let half_T : ℕ := T / 2
  let burn_time (k : ℕ) : ℕ := 5 * k^2
  let total_burn_time (m : ℕ) : ℕ := ∑ i in Finset.range (m + 1), burn_time (i + 1)
  let m : ℕ := Nat.sqrt (3 * half_T / 5)  -- Estimate m using the cube root approximation
  let total_burned_length : ℕ := 100 - m
  total_burned_length - (if total_burn_time m ≤ half_T ∧ half_T < total_burn_time (m + 1) then 0 else 1)

theorem candle_height_correct : candle_height_after_half_time = 20 :=
by
  -- The main part of the proof logic will go here
  sorry

end candle_height_correct_l215_215436


namespace ratio_man_to_son_in_two_years_l215_215799

-- Define the conditions
def son_current_age : ℕ := 32
def man_current_age : ℕ := son_current_age + 34

-- Define the ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- The theorem to prove the ratio in two years
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / son_age_in_two_years = 2 :=
by
  -- Skip the proof
  sorry

end ratio_man_to_son_in_two_years_l215_215799


namespace water_left_l215_215660

-- Conditions
def initial_water : ℚ := 3
def water_used : ℚ := 11 / 8

-- Proposition to be proven
theorem water_left :
  initial_water - water_used = 13 / 8 := by
  sorry

end water_left_l215_215660


namespace new_commission_rate_l215_215712

theorem new_commission_rate (C1 : ℝ) (slump : ℝ) : C2 : ℝ :=
  assume (h1 : C1 = 0.04)
  (h2 : slump = 0.20000000000000007),
  have h3: C2 = (C1 / (1 - slump)), from
  sorry,
  show C2 = 0.05, by sorry

end new_commission_rate_l215_215712


namespace sec_pi_over_18_minus_2_sin_pi_over_9_eq_zero_l215_215838

open Real

noncomputable def sec (x : ℝ) : ℝ := 1 / cos x

theorem sec_pi_over_18_minus_2_sin_pi_over_9_eq_zero :
  sec (π / 18) - 2 * sin (π / 9) = 0 :=
by
  have h1 : sin (π / 9) = 1 / 2 := sorry
  have h2 : sec (π / 18) = 1 / cos (π / 18) := rfl
  have h3 : sin (π / 9) = 2 * sin (π / 18) * cos (π / 18) := by rw [← sin_two_mul, mul_div_cancel' _ (ne_of_gt (cos_pos_of_mem_Ioo (half_lt_self pi_pos) (Ioo_subset_Ioo_left (by linarith))) .ne')]
  calc
    sec (π / 18) - 2 * sin (π / 9)
        = 1 / cos (π / 18) - 2 * (sin (π / 9))             : by rw [h2]
    ... = 1 / cos (π / 18) - 2 * (1 / 2)                  : by rw [h1]
    ... = 1 / cos (π / 18) - 1                            : by norm_num
    ... = (1 - cos (π / 18)) / cos (π / 18) - 1           : by field_simp [div_sub_div]
    ... = (1 - 1) / cos (π / 18)                          : by rw [h3, ← sin_two_mul, ctx := by norm_num; field_simp [ne_of_gt]]
    ... = 0                                              : by simp

end sec_pi_over_18_minus_2_sin_pi_over_9_eq_zero_l215_215838


namespace range_of_m_l215_215897

variables (m : ℝ)

def p := (m^2 - 4 > 0 ∧ -m < 0)
def q := (4 * (m - 2)^2 - 1 < 0)
def prop_or := p ∨ q
def prop_and := p ∧ q

theorem range_of_m :
  prop_or ∧ ¬prop_and ↔ m ∈ set.Ioo 1 2 ∪ set.Ici 3 :=
sorry

end range_of_m_l215_215897


namespace determine_value_of_m_l215_215149

theorem determine_value_of_m (m : ℤ) :
  2^2002 - 2^2000 - 2^1999 + 2^1998 = m * 2^1998 ↔ m = 11 := 
sorry

end determine_value_of_m_l215_215149


namespace probability_real_roots_quadratic_eq_die_roll_l215_215448

theorem probability_real_roots_quadratic_eq_die_roll:
  let outcomes := { (m, n) | m ∈ {1, 2, 3, 4, 5, 6} ∧ n ∈ {1, 2, 3, 4, 5, 6} }
  let favorable := { (m, n) ∈ outcomes | m^2 - 4 * n ≥ 0 }
  outcomes.nonempty → (favorable.card / outcomes.card = 19 / 36) := 
by
  -- Definitions of the sets and their cardinalities
  sorry

end probability_real_roots_quadratic_eq_die_roll_l215_215448


namespace find_b_l215_215853

noncomputable def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

theorem find_b (d b e : ℝ) (h1 : -d / 3 = -e) (h2 : -e = 1 + d + b + e) (h3 : e = 6) : b = -31 :=
by sorry

end find_b_l215_215853


namespace prob1_prob2_prob3_prob4_l215_215490

theorem prob1 : (3^3)^2 = 3^6 := by
  sorry

theorem prob2 : (-4 * x * y^3) * (-2 * x^2) = 8 * x^3 * y^3 := by
  sorry

theorem prob3 : 2 * x * (3 * y - x^2) + 2 * x * x^2 = 6 * x * y := by
  sorry

theorem prob4 : (20 * x^3 * y^5 - 10 * x^4 * y^4 - 20 * x^3 * y^2) / (-5 * x^3 * y^2) = -4 * y^3 + 2 * x * y^2 + 4 := by
  sorry

end prob1_prob2_prob3_prob4_l215_215490


namespace range_of_m_l215_215930

theorem range_of_m (m : ℝ) (h : m ≠ 0) : 
  (∀ x, x ∈ set.Icc 1 3 → (m * x^2 - m * x - 1) < -m + 5) ↔ (0 < m ∧ m < 6 / 7) :=
by
  sorry

end range_of_m_l215_215930


namespace solve_for_y_l215_215678

theorem solve_for_y (x y : ℝ) (h : 3 * x - 5 * y = 7) : y = (3 * x - 7) / 5 :=
sorry

end solve_for_y_l215_215678


namespace solve_trig_eq_l215_215683

noncomputable theory

open Real

theorem solve_trig_eq (x : ℝ) : (12 * sin x - 5 * cos x = 13) →
  ∃ (k : ℤ), x = (π / 2) + arctan (5 / 12) + 2 * k * π :=
by
s∞rry

end solve_trig_eq_l215_215683


namespace birds_in_sanctuary_l215_215115

theorem birds_in_sanctuary (x y : ℕ) 
    (h1 : x + y = 200)
    (h2 : 2 * x + 4 * y = 590) : 
    x = 105 :=
by
  sorry

end birds_in_sanctuary_l215_215115


namespace describe_set_T_l215_215638

theorem describe_set_T :
  let T := {p : ℝ × ℝ | (∃ (a b : ℝ), (a = 5 ∧ b = p.1 - 1 ∨ a = 5 ∧ b = p.2 + 3 ∨ a = p.1 - 1 ∧ b = p.2 + 3 ∧ a = b) ∧ (∃ c, c = 5 ∧ (c ≥ p.1 - 1 ∧ c ≥ p.2 + 3))}
  in T = {p : ℝ × ℝ | (p.1 = 6 ∧ p.2 ≤ 2) ∨ (p.2 = 2 ∧ p.1 ≤ 6) ∨ (p.2 = p.1 - 4 ∧ p.1 ≤ 6 ∧ p.2 ≤ 2)} := sorry

end describe_set_T_l215_215638


namespace pos_int_sum_div_8n_l215_215187

theorem pos_int_sum_div_8n :
  {n : ℕ | n > 0 ∧ (n * (n + 1) / 2) ∣ (8 * n)}.finite.count = 4 :=
begin
  sorry
end

end pos_int_sum_div_8n_l215_215187


namespace probability_of_at_least_6_consecutive_heads_l215_215788

-- Represent the coin flip outcomes and the event of interest
inductive CoinFlip
| H
| T

def all_flips : List (List CoinFlip) :=
  let base := [CoinFlip.H, CoinFlip.T]
  base.product (base.product (base.product (base.product (base.product (base.product (base.product base))))))

def consecutive_heads (n : Nat) (flips : List CoinFlip) : Bool :=
  List.any (List.tails flips) (λ l => l.take n = List.repeat CoinFlip.H n)

def atLeast6ConsecutiveHeads (flips : List CoinFlip) : Prop :=
  consecutive_heads 6 flips || consecutive_heads 7 flips || consecutive_heads 8 flips

def probabilityAtLeast6ConsecutiveHeads (flipsList : List (List CoinFlip)) : ℚ :=
  (flipsList.filter atLeast6ConsecutiveHeads).length / flipsList.length

-- The proof statement
theorem probability_of_at_least_6_consecutive_heads :
  probabilityAtLeast6ConsecutiveHeads all_flips = 13 / 256 :=
by 
  sorry

end probability_of_at_least_6_consecutive_heads_l215_215788


namespace trigonometric_identity_l215_215769

theorem trigonometric_identity :
  8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 :=
by
  sorry

end trigonometric_identity_l215_215769


namespace cannot_determine_total_movies_l215_215005

def number_of_books : ℕ := 22
def books_read : ℕ := 12
def books_to_read : ℕ := 10
def movies_watched : ℕ := 56

theorem cannot_determine_total_movies (n : ℕ) (h1 : books_read + books_to_read = number_of_books) : n ≠ movies_watched → n = 56 → False := 
by 
  intro h2 h3
  sorry

end cannot_determine_total_movies_l215_215005


namespace range_of_a_l215_215231

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 :=
by
  sorry

end range_of_a_l215_215231


namespace cyclic_pentagon_regular_l215_215667

theorem cyclic_pentagon_regular
  (A B C D E P Q : Type)
  [has_area A B P]
  [has_area A E Q]
  [has_area C D P]
  [has_area C D Q]
  [has_area A P Q]
  (h1 : cyclic A B C D E)
  (h2 : meet AC BD P)
  (h3 : meet AD CE Q)
  (h4 : equal_area (triangle A B P) (triangle A E Q))
  (h5 : equal_area (triangle A E Q) (triangle C D P))
  (h6 : equal_area (triangle C D P) (triangle C D Q))
  (h7 : equal_area (triangle C D Q) (triangle A P Q)) :
  regular_pentagon A B C D E :=
sorry

end cyclic_pentagon_regular_l215_215667


namespace first_reduction_percentage_l215_215719

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.70 = P * 0.525 ↔ x = 25 := by
  sorry

end first_reduction_percentage_l215_215719


namespace coefficient_x3_is_29_l215_215875

-- Define the expression
def expression := 2 * (x^3 - 2 * x^2 + x) + 4 * (x^4 + 3 * x^3 - x^2 + x) - 3 * (x - 5 * x^3 + 2 * x^5)

-- State the theorem
theorem coefficient_x3_is_29 : 
  (2 * 1 + 4 * 3 + 3 * 5 = 29) := 
by
  sorry  -- This will be replaced by the proof

end coefficient_x3_is_29_l215_215875


namespace instantaneous_velocity_at_t2_l215_215916

noncomputable def displacement (t : ℝ) : ℝ := t^2 * Real.exp (t - 2)

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2 = 8) :=
by
  sorry

end instantaneous_velocity_at_t2_l215_215916


namespace rectangles_diagonals_equal_not_rhombuses_l215_215096

/--
A proof that the property of having diagonals of equal length is a characteristic of rectangles
and not necessarily of rhombuses.
-/
theorem rectangles_diagonals_equal_not_rhombuses (R : Type) [rectangle R] (H : Type) [rhombus H] :
  (∀ r : R, diagonals_equal r) ∧ ¬(∀ h : H, diagonals_equal h) :=
sorry

end rectangles_diagonals_equal_not_rhombuses_l215_215096


namespace triangle_area_and_fraction_of_square_l215_215550

theorem triangle_area_and_fraction_of_square 
  (a b c s : ℕ) 
  (h_triangle : a = 9 ∧ b = 40 ∧ c = 41)
  (h_square : s = 41)
  (h_right_angle : a^2 + b^2 = c^2) :
  let area_triangle := (a * b) / 2
  let area_square := s^2
  let fraction := (a * b) / (2 * s^2)
  area_triangle = 180 ∧ fraction = 180 / 1681 := 
by
  sorry

end triangle_area_and_fraction_of_square_l215_215550


namespace sum_of_leading_digits_l215_215314

def leading_digit (n : ℕ) (x : ℝ) : ℕ := sorry

def M := 10^500 - 1

def g (r : ℕ) : ℕ := leading_digit r (M^(1 / r))

theorem sum_of_leading_digits :
  g 3 + g 4 + g 5 + g 7 + g 8 = 10 := sorry

end sum_of_leading_digits_l215_215314


namespace find_a_l215_215934

-- Define the constants b and the asymptote equation
def asymptote_eq (x y : ℝ) := 3 * x + 2 * y = 0

-- Define the hyperbola equation and the condition
def hyperbola_eq (x y a : ℝ) := x^2 / a^2 - y^2 / 9 = 1
def hyperbola_condition (a : ℝ) := a > 0

-- Theorem stating the value of a given the conditions
theorem find_a (a : ℝ) (hcond : hyperbola_condition a) 
  (h_asymp : ∀ x y : ℝ, asymptote_eq x y → y = -(3/2) * x) :
  a = 2 := 
sorry

end find_a_l215_215934


namespace length_XY_l215_215291

/-- In a 30-60-90 triangle, given the side opposite the 30 degree angle is 12, 
    show that the hypotenuse is 24. -/
theorem length_XY {X Y Z : Type*} [metric_space X] (P Q R : X) (h1 : angle P R Q = π / 2)
  (h2 : angle P Q R = π / 6) (d : dist P Q = 12) : dist P R = 24 :=
sorry

end length_XY_l215_215291


namespace equation_of_l1_range_for_b_l215_215212

theorem equation_of_l1 
  (x y : ℝ) 
  (C : x^2 + y^2 - 6*x - 4*y + 4 = 0) 
  (P : (5, 3)) 
  (is_midpoint : ∃l₁, is_chord_midpoint l₁ C P) : 
  ∃ (l₁ : ℝ → ℝ → Prop), ∀ x y, l₁ x y ↔ 2*x + y - 13 = 0 :=
sorry

theorem range_for_b 
  (x y b : ℝ) 
  (C : x^2 + y^2 - 6*x - 4*y + 4 = 0) 
  (l₂ : ∀ x y, x + y + b = 0) : 
  -3*Real.sqrt 2 - 5 < b ∧ b < 3*Real.sqrt 2 - 5 :=
sorry

end equation_of_l1_range_for_b_l215_215212


namespace a_2002_eq_3_l215_215049

-- Define the sequence using inductive construction
def a : ℕ → ℕ 
| 0       := 0            -- To avoid off-by-one issues, setting a_0 = 0.
| 1       := 1
| (n+2) := gcd (a (n + 1)) (n + 2) + 1

-- State the theorem
theorem a_2002_eq_3 : a 2002 = 3 := 
by sorry

end a_2002_eq_3_l215_215049


namespace problem_1_problem_2_problem_3_l215_215835

open Real

theorem problem_1 : (1 * (-12)) - (-20) + (-8) - 15 = -15 := by
  sorry

theorem problem_2 : -3^2 + ((2/3) - (1/2) + (5/8)) * (-24) = -28 := by
  sorry

theorem problem_3 : -1^(2023) + 3 * (-2)^2 - (-6) / ((-1/3)^2) = 65 := by
  sorry

end problem_1_problem_2_problem_3_l215_215835


namespace value_of_b_plus_c_l215_215564

theorem value_of_b_plus_c (b c : ℝ) (h1 : ∀ x : ℝ, 1 < x ∧ x < 3 → x^2 + b*x + c < 0) :
  b + c = -1 := 
begin
  /- Conditions handling and the proof to be completed -/
  sorry
end

end value_of_b_plus_c_l215_215564


namespace avg_production_last_5_days_l215_215765

theorem avg_production_last_5_days
  (avg_first_25_days : ℕ)
  (total_days : ℕ)
  (avg_entire_month : ℕ)
  (h1 : avg_first_25_days = 60)
  (h2 : total_days = 30)
  (h3 : avg_entire_month = 58) : 
  (total_days * avg_entire_month - 25 * avg_first_25_days) / 5 = 48 := 
by
  sorry

end avg_production_last_5_days_l215_215765


namespace amplitude_of_cosine_wave_l215_215830

theorem amplitude_of_cosine_wave 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_max_min : ∀ x : ℝ, d + a = 5 ∧ d - a = 1) 
  : a = 2 :=
by
  sorry

end amplitude_of_cosine_wave_l215_215830


namespace water_tank_capacity_l215_215084

theorem water_tank_capacity (x : ℝ)
  (h1 : (2 / 3) * x - (1 / 3) * x = 20) : x = 60 := 
  sorry

end water_tank_capacity_l215_215084


namespace transformed_variance_l215_215922

-- Definitions
def variance (xs : List ℝ) : ℝ :=
  let n := xs.length
  let mean := (List.sum xs) / n
  (List.sum (List.map (λ x => (x - mean) ^ 2) xs)) / n

-- Given data
variable (x1 x2 x3 x4 x5 : ℝ)
variable (h_var : variance [x1, x2, x3, x4, x5] = 3)

-- Definition of transformed data
def transformed_data := [2*x1 + 1, 2*x2 + 1, 2*x3 + 1, 2*x4 + 1, 2*x5 + 1]

-- Statement to prove
theorem transformed_variance : variance transformed_data = 12 :=
by
  sorry

end transformed_variance_l215_215922


namespace start_days_before_vacation_end_l215_215818

variable (x t V h : ℕ)

def days_condition := (∀ x t : ℕ, 3 * (x - t) = t + 6)

def distance_condition_one : (246 = (x - t) * V)
def distance_condition_two : (276 = t * V)

def travel_options_condition :=
(∀ x h : ℕ, 276 = (t * (V + h)) ∨ 276 = ((t - 1) * V + (V + 2 * h)))

theorem start_days_before_vacation_end : 
  (∃ x t V h : ℕ, days_condition x t ∧ distance_condition_one x t V 
                  ∧ distance_condition_two x t V 
                  ∧ travel_options_condition x h ∧ x = 4) := sorry

end start_days_before_vacation_end_l215_215818


namespace digit_divisibility_l215_215648

theorem digit_divisibility : 
  (∃ (A : ℕ), A < 10 ∧ 
   (4573198080 + A) % 2 = 0 ∧ 
   (4573198080 + A) % 5 = 0 ∧ 
   (4573198080 + A) % 8 = 0 ∧ 
   (4573198080 + A) % 10 = 0 ∧ 
   (4573198080 + A) % 16 = 0 ∧ A = 0) := 
by { use 0; sorry }

end digit_divisibility_l215_215648


namespace inter_of_A_and_B_union_of_A_and_B_range_of_a_if_inter_A_C_nonempty_l215_215247

open Set

variable {U : Type} [PartialOrder U] [LinearContinuousOrder U]

/-- Given sets A and B as defined, prove that A ∩ B = { x | 2 < x ∧ x ≤ 8 } -/
theorem inter_of_A_and_B :
  (λ x : ℝ, 1 < x ∧ x ≤ 8) ∩ (λ x : ℝ, 2 < x ∧ x < 9) = (λ x : ℝ, 2 < x ∧ x ≤ 8) :=
by sorry

/-- Given sets A and B as defined, prove that A ∪ B = { x | 1 < x ∧ x < 9 } -/
theorem union_of_A_and_B :
  (λ x : ℝ, 1 < x ∧ x ≤ 8) ∪ (λ x : ℝ, 2 < x ∧ x < 9) = (λ x : ℝ, 1 < x ∧ x < 9) :=
by sorry

/-- Given sets A and C as defined, prove that if A ∩ C ≠ ∅, then the range of a is (-∞, 8] -/
theorem range_of_a_if_inter_A_C_nonempty (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x ≤ 8 ∧ x ≥ a) ↔ a ≤ 8 :=
by sorry

end inter_of_A_and_B_union_of_A_and_B_range_of_a_if_inter_A_C_nonempty_l215_215247


namespace pass_in_both_subjects_l215_215283

variable (F_H F_E F_HE : ℝ)

theorem pass_in_both_subjects (h1 : F_H = 20) (h2 : F_E = 70) (h3 : F_HE = 10) :
  100 - ((F_H + F_E) - F_HE) = 20 :=
by
  sorry

end pass_in_both_subjects_l215_215283


namespace sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l215_215488

theorem sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022:
  ( (Real.sqrt 10 + 3) ^ 2023 * (Real.sqrt 10 - 3) ^ 2022 = Real.sqrt 10 + 3 ) :=
by {
  sorry
}

end sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l215_215488


namespace distinguishing_property_of_rectangles_l215_215103

theorem distinguishing_property_of_rectangles (rect rhomb : Type)
  [quadrilateral rect] [quadrilateral rhomb]
  (sum_of_interior_angles_rect : interior_angle_sum rect = 360)
  (sum_of_interior_angles_rhomb : interior_angle_sum rhomb = 360)
  (diagonals_bisect_each_other_rect : diagonals_bisect_each_other rect)
  (diagonals_bisect_each_other_rhomb : diagonals_bisect_each_other rhomb)
  (diagonals_equal_length_rect : diagonals_equal_length rect)
  (diagonals_perpendicular_rhomb : diagonals_perpendicular rhomb) :
  distinguish_property rect rhomb := by
  sorry

end distinguishing_property_of_rectangles_l215_215103


namespace number_of_valid_n_l215_215503

theorem number_of_valid_n (n : ℕ) : 
  ∀ (n : ℤ), 4800 * (3 / 4)^n ∈ ℤ ↔ n ≤ 3 := sorry

end number_of_valid_n_l215_215503


namespace smallest_sum_of_20_consecutive_integers_is_490_l215_215721

theorem smallest_sum_of_20_consecutive_integers_is_490 :
  ∃ n : ℕ, (∑ i in finset.range 20, (n + i)) = 490 ∧ nat.is_square (∑ i in finset.range 20, (n + i)) :=
sorry

end smallest_sum_of_20_consecutive_integers_is_490_l215_215721


namespace greatest_integer_less_than_or_equal_to_l215_215173

theorem greatest_integer_less_than_or_equal_to (x : ℝ) (h : x = 2 + Real.sqrt 3) : 
  ⌊x^3⌋ = 51 :=
by
  have h' : x ^ 3 = (2 + Real.sqrt 3) ^ 3 := by rw [h]
  sorry

end greatest_integer_less_than_or_equal_to_l215_215173


namespace total_stones_is_60_l215_215051

variable (x : ℕ) -- Number of stones in the third pile

-- The number of stones in each pile based on the given conditions
def fifth_pile := 6 * x
def second_pile := 2 * (x + fifth_pile)
def first_pile := fifth_pile / 3
def fourth_pile := first_pile + 10

-- Sum of stones in all piles
def total_stones := x + fifth_pile + second_pile + first_pile + (second_pile / 2)

-- The Lean statement to prove the total number of stones is 60
theorem total_stones_is_60 (h1: fifth_pile = 6 * x)
                           (h2: second_pile = 2 * (x + fifth_pile))
                           (h3: first_pile = fifth_pile / 3)
                           (h4: fourth_pile = first_pile + 10)
                           (h5: fourth_pile = second_pile / 2) :
                           total_stones = 60 := by
  sorry

end total_stones_is_60_l215_215051


namespace leftover_value_correct_l215_215077

-- Definitions based on the given conditions.
def roll_quarters := 40
def roll_nickels := 40
def mia_quarters := 92
def mia_nickels := 184
def thomas_quarters := 138
def thomas_nickels := 212
def quarter_value := 0.25
def nickel_value := 0.05

-- Statement of the problem.
theorem leftover_value_correct :
  let total_quarters := mia_quarters + thomas_quarters in
  let total_nickels := mia_nickels + thomas_nickels in
  let leftover_quarters := total_quarters % roll_quarters in
  let leftover_nickels := total_nickels % roll_nickels in
  (leftover_quarters * quarter_value + leftover_nickels * nickel_value) = 9.30 := by
  sorry

end leftover_value_correct_l215_215077


namespace no_more_than_eight_squares_l215_215673

   -- Definitions:
   -- We assume that the larger square and smaller squares are represented geometrically.
   
   -- Given that there is a larger square L, and it is not possible to attach more than 8 
   -- smaller squares such that each smaller square touches the perimeter of L and 
   -- none of them overlap with L or each other.
   def max_squares := 8

   noncomputable def attach_squares (L : Type) (squares : List Type) : Prop :=
     ∀ (s : Type), s ∈ squares → touches_perimeter s L ∧ ¬ overlaps s L ∧ ∀ s', s' ∈ squares → s ≠ s' → ¬ overlaps s s'

   theorem no_more_than_eight_squares (L : Type) (smaller_squares : List Type) :
     attach_squares L smaller_squares → length smaller_squares ≤ max_squares :=
   by sorry
   
end no_more_than_eight_squares_l215_215673


namespace seq_ratio_l215_215620

noncomputable def a : ℕ → ℚ
| 1       := 1
| (n + 2) := if (n + 2) % 2 = 0 then (a (n + 1) + 1) / a (n + 1) else (a (n + 1) + (-1)^1 : ℚ) / a (n + 1)

theorem seq_ratio : a 3 / a 5 = 3 / 4 := by
  sorry

end seq_ratio_l215_215620


namespace polynomial_A_and_value_of_A_plus_2B_l215_215813

def B (x: ℝ) : ℝ := 4 * x^2 - 5 * x - 7

theorem polynomial_A_and_value_of_A_plus_2B :
  (∃ (A : ℝ → ℝ), (∀ x : ℝ, A x - 2 * B x = -2 * x^2 + 10 * x + 14) ∧ A = (λ x, 6 * x^2))
  ∧ (let A := (λ x: ℝ, 6 * x^2) in A (-1) + 2 * B (-1) = 10) :=
by
  sorry

end polynomial_A_and_value_of_A_plus_2B_l215_215813


namespace inequality_solution_set_l215_215966

variable {f : ℝ → ℝ}

-- Conditions
def neg_domain : Set ℝ := {x | x < 0}
def pos_domain : Set ℝ := {x | x > 0}
def f_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def f_property_P (f : ℝ → ℝ) := ∀ (x1 x2 : ℝ), (0 < x1) → (0 < x2) → (x1 ≠ x2) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)

-- Translate question and correct answer into a proposition in Lean
theorem inequality_solution_set (h1 : ∀ x, f (-x) = -f x)
                                (h2 : ∀ x1 x2, (0 < x1) → (0 < x2) → (x1 ≠ x1) → ((x2 * f x1 - x1 * f x2) / (x1 - x2) < 0)) :
  {x | f (x - 2) < f (x^2 - 4) / (x + 2)} = {x | x < -3} ∪ {x | -1 < x ∧ x < 2} := 
sorry

end inequality_solution_set_l215_215966


namespace marys_mother_bought_3_pounds_of_beef_l215_215347

-- Define the variables and constants
def total_paid : ℝ := 16
def cost_of_chicken : ℝ := 2 * 1  -- 2 pounds of chicken
def cost_per_pound_beef : ℝ := 4
def cost_of_oil : ℝ := 1
def shares : ℝ := 3  -- Mary and her two friends

theorem marys_mother_bought_3_pounds_of_beef:
  total_paid - (cost_of_chicken / shares) - cost_of_oil = 3 * cost_per_pound_beef :=
by
  -- the proof goes here
  sorry

end marys_mother_bought_3_pounds_of_beef_l215_215347


namespace total_lotus_flowers_l215_215534

theorem total_lotus_flowers (x : ℕ) (h1 : x > 0) 
  (c1 : 3 ∣ x)
  (c2 : 5 ∣ x)
  (c3 : 6 ∣ x)
  (c4 : 4 ∣ x)
  (h_total : x = x / 3 + x / 5 + x / 6 + x / 4 + 6) : 
  x = 120 :=
by
  sorry

end total_lotus_flowers_l215_215534


namespace problem_unique_function_l215_215879

noncomputable def number_of_functions : ℕ :=
  if (∀ f : ℝ → ℝ, (∀ x y : ℝ, f(x + f(y) + 1) = x + y + 1) → (∃! g : ℝ → ℝ, g = f)) then 1 else 0

theorem problem_unique_function :
  number_of_functions = 1 := by
  sorry

end problem_unique_function_l215_215879


namespace min_dose_max_k_l215_215074

def f (x : ℝ) : ℝ :=
  if x > 0 ∧ x ≤ 4 then -0.5 * x^2 + 2 * x + 8
  else if x > 4 ∧ x ≤ 16 then -0.5 * x - Real.log 2 x + 12
  else 0

noncomputable def y (m x : ℝ) : ℝ := m * f x

theorem min_dose (m : ℝ) : (∀ x, 0 < x ∧ x ≤ 8 → y m x ≥ 12) → m ≥ 12 / 5 := sorry

theorem max_k : ∃ k : ℕ, (∀ x, 0 < x → x ≤ k → y 2 x ≥ 12) ∧ (∀ x, x > k → y 2 x < 12) ∧ k = 6 := sorry

end min_dose_max_k_l215_215074


namespace find_A_and_value_at_minus_one_l215_215810

section polynomial_proof

variable (x : ℝ)

-- Define polynomial B
def B := 4 * x^2 - 5 * x - 7

-- Mistaken calculation result from Xiao Li
def mistaken_calc := -2 * x^2 + 10 * x + 14

-- Define the expected polynomial A
def A := 6 * x^2

-- Define correct calculation of A + 2B
def correct_A_plus_2B := A + 2 * B

-- The main theorem to prove
theorem find_A_and_value_at_minus_one :
  (A - (2 * B) = mistaken_calc) → (A = 6 * x^2) ∧ (correct_A_plus_2B.eval (-1) = 10) :=
by
  sorry

end polynomial_proof

end find_A_and_value_at_minus_one_l215_215810


namespace proof_problem_l215_215938

def U : Set ℤ := {x | x^2 - x - 12 ≤ 0}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {0, 1, 3, 4}

theorem proof_problem : (U \ A) ∩ B = {0, 1, 4} := 
by sorry

end proof_problem_l215_215938


namespace range_of_x_l215_215655

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem range_of_x (x : ℝ) : f(x) + f(x - 1/2) > 1 ↔ x > -1/4 :=
begin
  sorry
end

end range_of_x_l215_215655


namespace equation_of_line_AB_l215_215438

theorem equation_of_line_AB 
  (x y : ℝ)
  (passes_through_P : (4 - 1)^2 + (1 - 0)^2 = 1)     
  (circle_eq : (x - 1)^2 + y^2 = 1) :
  3 * x + y - 4 = 0 :=
sorry

end equation_of_line_AB_l215_215438


namespace probability_two_boys_three_girls_l215_215740

theorem probability_two_boys_three_girls (n k : ℕ) (p : ℝ) (h_n : n = 5) (h_k : k = 2) (h_p : p = 0.5) :
  let binom := (Nat.factorial n) / (Nat.factorial k * (Nat.factorial (n - k)))
  let probability := binom * p^k * (1-p)^(n-k)
  probability = 0.3125 :=
by
  have h₁ : binom = 10, sorry
  have h₂ : probability = 10 * (0.5)^2 * (0.5)^3, sorry
  exact h₁.symm.trans h₂.symm.trans (by norm_num),

end probability_two_boys_three_girls_l215_215740


namespace shaded_triangles_area_approx_34_l215_215825

noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (sqrt 3 / 4) * side^2
noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ := a * ((1 - r^n) / (1 - r))

theorem shaded_triangles_area_approx_34 :
  let side_length : ℝ := 10 in
  let initial_area : ℝ := equilateral_triangle_area side_length in
  let a : ℝ := initial_area / 4 in
  let r : ℝ := 1 / 4 in
  let n : ℕ := 50 in
  let sum : ℝ := geometric_series_sum a r n in
  abs (sum - 25) < 1 ->
  34 ∈ [32, 34, 36, 38, 40] :=
by
  sorry

end shaded_triangles_area_approx_34_l215_215825


namespace total_toys_l215_215472

theorem total_toys (A M T : ℕ) (h1 : A = 3 * M + M) (h2 : T = A + 2) (h3 : M = 6) : A + M + T = 56 :=
by
  sorry

end total_toys_l215_215472


namespace absent_present_probability_l215_215604

theorem absent_present_probability : 
  ∀ (p_absent_normal p_absent_workshop p_present_workshop : ℚ), 
    p_absent_normal = 1 / 20 →
    p_absent_workshop = 2 * p_absent_normal →
    p_present_workshop = 1 - p_absent_workshop →
    p_absent_workshop = 1 / 10 →
    (p_present_workshop * p_absent_workshop + p_absent_workshop * p_present_workshop) * 100 = 18 :=
by
  intros
  sorry

end absent_present_probability_l215_215604


namespace random_event_l215_215415

-- Conditions
def triangle_inequality (a b c : ℝ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

def sun_rises_from_east : Prop := True  -- A predictable phenomenon

def vehicle_encounters_green_light : Prop := ∃ (random_event : ℕ → bool), random_event 0 = true

def abs_is_negative (x : ℚ) : Prop := x.abs < 0  -- An impossibility with rational numbers

-- Theorem to prove
theorem random_event : vehicle_encounters_green_light :=
by
  sorry

end random_event_l215_215415


namespace correctly_solved_statement_l215_215251

-- Conditions provided in the problem
def statement_1 (a b : ℝ) : Prop := 2 * a + 3 * b = 5 * a * b
def statement_2 (a : ℝ) : Prop := (3 * a^3)^2 = 6 * a^6
def statement_3 (a : ℝ) : Prop := a^6 / a^2 = a^3
def statement_4 (a : ℝ) : Prop := a^2 * a^3 = a^5

-- The question to prove that only statement 4 is correct
theorem correctly_solved_statement : ∀ (a b : ℝ), ¬(statement_1 a b) ∧ ¬(statement_2 a) ∧ ¬(statement_3 a) ∧ statement_4 a :=
by
  intros a b
  split
  { -- Proof that statement_1 is incorrect
    sorry },
  split
  { -- Proof that statement_2 is incorrect
    sorry },
  split
  { -- Proof that statement_3 is incorrect
    sorry },
  { -- Proof that statement_4 is correct
    sorry }

end correctly_solved_statement_l215_215251


namespace smallest_possible_six_digit_number_l215_215081

noncomputable def smallest_six_digit_number : ℕ :=
  let N := 122 in
  let k := 2 in
  60 * (1000 * k + N)

theorem smallest_possible_six_digit_number: smallest_six_digit_number = 122040 := by
  sorry

end smallest_possible_six_digit_number_l215_215081


namespace find_n_value_l215_215645

def n (x y : ℤ) : ℤ := x - (y^(x - y) + x * y)

theorem find_n_value : n 3 (-3) = -717 := by
  sorry

end find_n_value_l215_215645


namespace find_minimum_in_given_domain_l215_215854

-- Define the function
def fn (x y : ℝ) : ℝ := (x * y) / (2 * x^2 + 3 * y^2)

-- Define the domain constraints
def domain_x (x : ℝ) : Prop := 0.5 ≤ x ∧ x ≤ 0.7
def domain_y (y : ℝ) : Prop := 0.3 ≤ y ∧ y ≤ 0.6

-- Theorem statement
theorem find_minimum_in_given_domain :
  ∃ x y, domain_x x ∧ domain_y y ∧
  (∀ (u : ℝ), domain_x u → ∀ (v : ℝ), domain_y v → fn x y ≤ fn u v) ∧
  fn x y = 1 / (4 * Real.sqrt (3 / 2)) :=
sorry

end find_minimum_in_given_domain_l215_215854


namespace letters_posting_ways_l215_215613

theorem letters_posting_ways :
  let mailboxes := 4
  let letters := 3
  (mailboxes ^ letters) = 64 :=
by
  let mailboxes := 4
  let letters := 3
  show (mailboxes ^ letters) = 64
  sorry

end letters_posting_ways_l215_215613


namespace number_of_books_per_continent_l215_215482

theorem number_of_books_per_continent (total_books : ℕ) (total_continents : ℕ) 
  (h1 : total_books = 488) (h2 : total_continents = 4) :
  (total_books / total_continents) = 122 :=
begin
  -- Lean part does not need the proof steps.
  sorry
end

end number_of_books_per_continent_l215_215482


namespace discount_correct_l215_215630

-- Define the prices of items and the total amount paid
def t_shirt_price : ℕ := 30
def backpack_price : ℕ := 10
def cap_price : ℕ := 5
def total_paid : ℕ := 43

-- Define the total cost before discount
def total_cost := t_shirt_price + backpack_price + cap_price

-- Define the discount
def discount := total_cost - total_paid

-- Prove that the discount is 2 dollars
theorem discount_correct : discount = 2 :=
by
  -- We need to prove that (30 + 10 + 5) - 43 = 2
  sorry

end discount_correct_l215_215630


namespace markers_blue_l215_215495

theorem markers_blue {total_markers red_markers blue_markers : ℝ} 
  (h_total : total_markers = 64.0) 
  (h_red : red_markers = 41.0) 
  (h_blue : blue_markers = total_markers - red_markers) : 
  blue_markers = 23.0 := 
by 
  sorry

end markers_blue_l215_215495


namespace total_people_is_72_l215_215668

namespace KnightsAndLiars

def num_knights : Nat := 48

def evident_group_total_people (num_knights : Nat) : Nat :=
  let groups := num_knights / 2
  num_knights + groups

theorem total_people_is_72 (h : num_knights = 48) : evident_group_total_people num_knights = 72 := by
  rw [h]
  unfold evident_group_total_people
  norm_num
  sorry
end KnightsAndLiars

end total_people_is_72_l215_215668


namespace total_turns_to_fill_drum_l215_215484

variable (Q : ℝ) -- Capacity of bucket Q
variable (turnsP : ℝ) (P_capacity : ℝ) (R_capacity : ℝ) (drum_capacity : ℝ)

-- Condition: It takes 60 turns for bucket P to fill the empty drum
def bucketP_fills_drum_in_60_turns : Prop := turnsP = 60 ∧ P_capacity = 3 * Q ∧ drum_capacity = 60 * P_capacity

-- Condition: Bucket P has thrice the capacity as bucket Q
def bucketP_capacity : Prop := P_capacity = 3 * Q

-- Condition: Bucket R has half the capacity as bucket Q
def bucketR_capacity : Prop := R_capacity = Q / 2

-- Computation: Using all three buckets together, find the combined capacity filled in one turn
def combined_capacity_per_turn : ℝ := P_capacity + Q + R_capacity

-- Main Theorem: It takes 40 turns to fill the drum using all three buckets together
theorem total_turns_to_fill_drum
  (h1 : bucketP_fills_drum_in_60_turns Q turnsP P_capacity drum_capacity)
  (h2 : bucketP_capacity Q P_capacity)
  (h3 : bucketR_capacity Q R_capacity) :
  drum_capacity / combined_capacity_per_turn Q P_capacity (Q / 2) = 40 :=
by
  sorry

end total_turns_to_fill_drum_l215_215484


namespace inequalities_always_hold_l215_215263

theorem inequalities_always_hold (x y a b : ℝ) (hxy : x > y) (hab : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) :=
by
  sorry

end inequalities_always_hold_l215_215263


namespace tangent_line_at_one_f_gt_one_l215_215235

noncomputable def f (x : ℝ) : ℝ :=
  Real.exp x * Real.log x + (2 * Real.exp (x - 1)) / x

theorem tangent_line_at_one : 
  let y := f 1 + (Real.exp 1) * (x - 1)
  y = Real.exp (1 : ℝ) * (x - 1) + 2 := 
sorry

theorem f_gt_one (x : ℝ) (hx : 0 < x) : f x > 1 := 
sorry

end tangent_line_at_one_f_gt_one_l215_215235


namespace train_length_l215_215443

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_head_start_m : ℝ := 240
noncomputable def train_passing_time_s : ℝ := 35.99712023038157

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def distance_covered_by_train : ℝ := relative_speed_mps * train_passing_time_s

theorem train_length :
  distance_covered_by_train - jogger_head_start_m = 119.9712023038157 :=
by
  sorry

end train_length_l215_215443


namespace bus_is_there_probability_l215_215435

noncomputable def probability_bus_present : ℚ :=
  let total_area := 90 * 90
  let triangle_area := (75 * 75) / 2
  let parallelogram_area := 75 * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem bus_is_there_probability :
  probability_bus_present = 7/16 :=
by
  sorry

end bus_is_there_probability_l215_215435


namespace calculate_total_fruits_l215_215989

def W (t : ℝ) : ℝ := t^3 - 2*t^2 + 3*t - 1
def P (t : ℝ) : ℝ := -2*t^3 + 3*t^2 + 4*t + 2
def Pl (t : ℝ) : ℝ := t^3 - t^2 + 6*t - 3

def W_p (t : ℝ) : ℝ := (W t) / 60
def P_p (t : ℝ) : ℝ := (P t) / 40
def Pl_p (t : ℝ) : ℝ := (Pl t) / 20

def Fa_W (t : ℝ) : ℝ := (W t) * (W_p t)
def Fa_P (t : ℝ) : ℝ := (P t) * (P_p t) - (Fa_W t) - 12
def Fa_Pl (t : ℝ) : ℝ := (Pl t) * (Pl_p t) - 3 * (Fa_P t)

def total_fruits (t : ℝ) : ℝ := Fa_W t + Fa_P t + Fa_Pl t

theorem calculate_total_fruits : ∃ t : ℝ, P_p t > 0 ∧ total_fruits t > 0 := by
  sorry

end calculate_total_fruits_l215_215989


namespace solve_y_from_expression_l215_215141

-- Define the conditions
def given_conditions := (784 = 28^2) ∧ (49 = 7^2)

-- Define the equivalency to prove based on the given conditions
theorem solve_y_from_expression (h : given_conditions) : 784 + 2 * 28 * 7 + 49 = 1225 := by
  sorry

end solve_y_from_expression_l215_215141


namespace probability_of_exactly_one_instrument_l215_215044

-- Definitions
def total_people : ℕ := 800
def fraction_play_at_least_one_instrument : ℚ := 2 / 5
def num_play_two_or_more_instruments : ℕ := 96

-- Calculation
def num_play_at_least_one_instrument := fraction_play_at_least_one_instrument * total_people
def num_play_exactly_one_instrument := num_play_at_least_one_instrument - num_play_two_or_more_instruments

-- Probability calculation
def probability_play_exactly_one_instrument := num_play_exactly_one_instrument / total_people

-- Proof statement
theorem probability_of_exactly_one_instrument :
  probability_play_exactly_one_instrument = 0.28 := by
  sorry

end probability_of_exactly_one_instrument_l215_215044


namespace cannot_turn_all_white_2004_can_turn_all_white_2003_l215_215670

-- This is the main definition with placeholder Lean types 
-- and conditions based on the problem description.

def flipThree (cards : List Bool) (index : Nat) : List Bool :=
  cards -- This should be the real flipping function, used here as a placeholder.

def canFlipAllWhite (cards : List Bool) : Bool :=
  sorry -- This function must determine if all cards can be flipped white.

theorem cannot_turn_all_white_2004 (start_cards : List Bool) 
  (h_len : start_cards.length = 2004) 
  (h_one_black : start_cards.count (λ b => b = ff) = 1) : 
  canFlipAllWhite start_cards = false :=
sorry

theorem can_turn_all_white_2003 (start_cards : List Bool) 
  (h_len : start_cards.length = 2003) 
  (h_one_black : start_cards.count (λ b => b = ff) = 1) : 
  canFlipAllWhite start_cards = true :=
sorry

end cannot_turn_all_white_2004_can_turn_all_white_2003_l215_215670


namespace tetrahedron_same_color_exists_l215_215540

-- Define the problem conditions and statement
def exists_tetrahedron_same_color (points : Finset (ℝ × ℝ × ℝ)) (color : (ℝ × ℝ × ℝ) → Prop) : Prop :=
  (points.card = 20) ∧ -- Condition: There are 20 points
  (∀ p1 p2 p3 p4 : ℝ × ℝ × ℝ, -- Condition: No more than 3 points of the same color lie on any plane
    {p1, p2, p3, p4} ⊆ points →
    color p1 = color p2 ∧ color p2 = color p3 ∧ color p3 = color p1 →
    ∃ p5 : ℝ × ℝ × ℝ, p5 ∈ points ∧ color p5 ≠ color p1) →
  ∃ A B C D : ℝ × ℝ × ℝ, -- Exist a tetrahedron with the required properties
  A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ D ∈ points ∧
  color A = color B ∧ color B = color C ∧ color C = color D ∧
  ∃ (color' : (ℝ × ℝ × ℝ) → Prop), color' ≠ color ∧
  (¬((A, B) ∩ color') ∧ ¬((A, C) ∩ color') ∧ ¬((A, D) ∩ color') ∧ ¬((B, C) ∩ color') ∧ ¬((B, D) ∩ color') ∧ ¬((C, D) ∩ color'))

-- Sorry, I skip the proof as directed
theorem tetrahedron_same_color_exists (points : Finset (ℝ × ℝ × ℝ)) (color : (ℝ × ℝ × ℝ) → Prop) :
  exists_tetrahedron_same_color points color :=
sorry

end tetrahedron_same_color_exists_l215_215540


namespace angle_at_5_50_is_125_degrees_l215_215021

theorem angle_at_5_50_is_125_degrees :
  let hour_degree_per_hour := 30
  let hour_degree_per_minute := 0.5
  let minute_degree_per_minute := 6
  let hour_position_at_5 :=
    5 * hour_degree_per_hour
  let hour_position_at_5_50 :=
    hour_position_at_5 + (50 * hour_degree_per_minute)
  let minute_position_at_50 :=
    50 * minute_degree_per_minute
  |minute_position_at_50 - hour_position_at_5_50| = 125 := by
  sorry

end angle_at_5_50_is_125_degrees_l215_215021


namespace correct_number_of_judgments_is_one_l215_215568

theorem correct_number_of_judgments_is_one :
  let judgment1 := ∃ x₀ ∈ ℝ, Real.exp x₀ ≤ 0
  let judgment2 := ∀ x : ℝ, x > 0 → 2^x > x^2
  let judgment3 := ∀ a b : ℝ, (a > 1 ∧ b > 1) ↔ (a * b > 1)
  let judgment4 := ∀ p q : Prop, (¬q → ¬p) ↔ (p → q)
  num_correct_judgments = 1
:=
by
  sorry

end correct_number_of_judgments_is_one_l215_215568


namespace geometric_series_sum_eq_4_over_3_l215_215122

theorem geometric_series_sum_eq_4_over_3 : 
  let a := 1
  let r := 1/4
  inf_geometric_sum a r = 4/3 := by
begin
  intros,
  sorry
end

end geometric_series_sum_eq_4_over_3_l215_215122
