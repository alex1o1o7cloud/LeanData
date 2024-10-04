import Mathlib

namespace odd_function_g_l185_185180

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185180


namespace rolling_circle_arc_constant_length_l185_185767

-- Definitions
def IsIsosceles (ABC : Type) [Triangle ABC] (AB BC AC : ℝ) : Prop :=
  AB = BC

def IsAltitude (ABC : Type) [Triangle ABC] (B D : Point) : Prop :=
  ∃ A C : Point, Altitude B D A C

def CircleRollsAlongLine (c : Circle) (AC : ℝ) : Prop :=
  ∃ radius : ℝ, c.radius = radius ∧ ∀ P ∈ c, P ∈ Line AC

-- Problem Statement
theorem rolling_circle_arc_constant_length
  (ABC : Type) [Triangle ABC]
  (AB BC AC : ℝ)
  (B D : Point)
  (c : Circle)
  (cond1 : IsIsosceles ABC AB BC)
  (cond2 : IsAltitude ABC B D)
  (cond3 : CircleRollsAlongLine c AC)
  (P : Point) (hP : P ∈ c) (hB_in_c : B ∈ c) :
  ∃ KL : Arc, ∀ B_in_circle : B ∈ c, arc_length KL = constant :=
sorry

end rolling_circle_arc_constant_length_l185_185767


namespace odd_function_proof_l185_185433

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185433


namespace g_is_odd_l185_185236

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185236


namespace odd_function_g_l185_185162

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185162


namespace range_equals_average_equals_median_not_equals_stddev_not_equals_l185_185821

variables {α : Type*} [linear_order α] [add_group α] [has_scalar ℝ α] 
variables (x : ℕ → α) (x₀ : α)

-- Conditions
def average_is_x₀ (x₀ : α) := (x 0 + x 1 + x 2 + x 3 + x 4 + x 5) / 6 = x₀
def ordered_xs := x 0 < x 1 ∧ x 1 < x 2 ∧ x 2 < x 3 ∧ x 3 < x 4 ∧ x 4 < x 5

-- Questions rewritten as proofs
theorem range_equals (havg : average_is_x₀ x₀) (hord : ordered_xs x) :
  (max (x 0) (max (x 1) (max (x 2) (max (x 3) (max (x 4) (x 5))))) - 
   min (x 0) (min (x 1) (min (x 2) (min (x 3) (min (x 4) (x 5))))) = 
   max (x 0) (max (x 1) (max (x 2) (max (x 3) (max (x 4) (max (x 5) x₀))))) - 
   min (x 0) (min (x 1) (min (x 2) (min (x 3) (min (x 4) (min (x 5) x₀)))))
   :=
sorry

theorem average_equals (havg : average_is_x₀ x₀) (hord : ordered_xs x) :
  (x 0 + x 1 + x 2 + x 3 + x 4 + x 5) / 6 = 
   (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x₀) / 7
   :=
sorry

theorem median_not_equals (havg : average_is_x₀ x₀) (hord : ordered_xs x) :
  ¬(x 3 + x 4) / 2 = if x₀ ≤ x 2 then (x 2 + x 3) / 2
                     else if x₀ ≤ x 3 then (x₀ + x 3) / 2
                     else (x 3 + x 4 + x₀) / 2
  :=
sorry

theorem stddev_not_equals (havg : average_is_x₀ x₀) (hord : ordered_xs x) :
  ¬((1/6) * ((x 0 - x₀)^2 + (x 1 - x₀)^2 + (x 2 - x₀)^2 + (x 3 - x₀)^2 + (x 4 - x₀)^2 + (x 5 - x₀)^2)) =
   ((1/7) * ((x 0 - x₀)^2 + (x 1 - x₀)^2 + (x 2 - x₀)^2 + (x 3 - x₀)^2 + (x 4 - x₀)^2 + (x 5 - x₀)^2 + 0))
  :=
sorry

end range_equals_average_equals_median_not_equals_stddev_not_equals_l185_185821


namespace g_is_odd_l185_185225

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185225


namespace stratified_sampling_total_students_l185_185929

theorem stratified_sampling_total_students (n : ℕ) (h1 : 600 = some (t : Nat))
  (h2 : 21 = some (s : Nat)) (h3 : 1800 = some (T : Nat)) 
  (h4 : 21 * T = n * t) : n = 63 :=
sorry

end stratified_sampling_total_students_l185_185929


namespace area_points_nearer_to_vertex_l185_185732

noncomputable def area_percentage_of_points_closer_to_vertex_than_centroid
  (T : Type)
  [equilateral_triangle T]
  (A B C : T)
  (G : centroid T A B C) : ℝ := 33.33

theorem area_points_nearer_to_vertex (T : Type)
  [equilateral_triangle T]
  (A B C : T)
  (G : centroid T A B C)
  (percentage_area : ℝ)
  (h : percentage_area = area_percentage_of_points_closer_to_vertex_than_centroid T A B C G) :
  percentage_area = 33.33 := 
by 
  exact h

end area_points_nearer_to_vertex_l185_185732


namespace g_is_odd_l185_185625

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185625


namespace odd_function_l185_185138

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185138


namespace g_is_odd_l185_185212

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185212


namespace g_B_is_odd_l185_185662

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185662


namespace option_b_is_odd_l185_185119

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185119


namespace g_is_odd_l185_185322

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185322


namespace option_B_odd_l185_185655

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185655


namespace g_is_odd_l185_185089

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185089


namespace is_odd_g_l185_185463

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185463


namespace geometric_sequence_constants_l185_185727

variable {a : ℕ → ℝ}
variable {T : ℕ → ℝ}
variable {q : ℝ}
variable [geom_seq : ∀ n, a (n + 1) = a n * q]

theorem geometric_sequence_constants
  (h1 : a 3 * a 6 * a 18 = 1)
  (h2 : a n = a 0 * q ^ n)
  : (T 13 = (a 7)^13) ∧ (T 17 = (a 9)^17) := by
  sorry

end geometric_sequence_constants_l185_185727


namespace option_b_is_odd_l185_185111

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185111


namespace weight_of_172_is_around_60_316_l185_185865

noncomputable def weight_prediction (x : ℝ) : ℝ := 0.849 * x - 85.712

theorem weight_of_172_is_around_60_316 :
  ∀ (x : ℝ), x = 172 → abs (weight_prediction x - 60.316) < 1 :=
by
  sorry

end weight_of_172_is_around_60_316_l185_185865


namespace sum_of_coeffs_eq_zero_l185_185758

theorem sum_of_coeffs_eq_zero :
  let t := ∫ x in 0..(Real.pi / 4), Real.cos (2 * x)
  ∃ (a : ℕ → ℝ), (1 - x / t) ^ 2018 = ∑ i in Finset.range 2019, a i * x ^ i ∧ 
  (Finset.range 2019).sum (λ i, if i > 0 then a i else 0) = 0 :=
by
  let t := ∫ x in 0..(Real.pi / 4), Real.cos (2 * x)
  have : t = 1 / 2 := sorry -- given in the solution
  let a : ℕ → ℝ := λ i, 
    if i = 0 then 1 
    else (Finset.range 2019).sum (λ j, if j = i then (-2) ^ 2018 else 0)
  use a
  split
  · sorry -- Proof that (1 - x / t) ^ 2018 = ∑ i in Finset.range 2019, a i * x ^ i
  · sorry -- Proof that (Finset.range 2019).sum (λ i, if i > 0 then a i else 0) = 0

end sum_of_coeffs_eq_zero_l185_185758


namespace g_is_odd_l185_185224

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185224


namespace problem_statement_l185_185049

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185049


namespace optionB_is_odd_l185_185597

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185597


namespace odd_function_option_B_l185_185403

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185403


namespace g_is_odd_l185_185074

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185074


namespace total_books_in_series_l185_185902

-- Definitions for the conditions
def books_read : ℕ := 8
def books_to_read : ℕ := 6

-- Statement to be proved
theorem total_books_in_series : books_read + books_to_read = 14 := by
  sorry

end total_books_in_series_l185_185902


namespace problem_is_odd_function_proof_l185_185454

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185454


namespace odd_function_g_l185_185283

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185283


namespace odd_function_g_l185_185520

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185520


namespace problem_statement_l185_185064

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185064


namespace tracy_initial_candies_l185_185870

noncomputable def initial_candies : Nat := 80

theorem tracy_initial_candies
  (x : Nat)
  (hx1 : ∃ y : Nat, (1 ≤ y ∧ y ≤ 6) ∧ x = (5 * (44 + y)) / 3)
  (hx2 : x % 20 = 0) : x = initial_candies := by
  sorry

end tracy_initial_candies_l185_185870


namespace optionB_is_odd_l185_185589

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185589


namespace g_is_odd_l185_185223

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185223


namespace work_done_correct_l185_185708

-- Given conditions
def force : ℝ := 10 -- Force in Newtons
def compression : ℝ := 0.1 -- Compression in meters
def displacement : ℝ := 0.06 -- Displacement in meters

-- Definition of Hooke's Law spring constant
def spring_constant (F : ℝ) (x : ℝ) : ℝ := F / x

-- Hooke's Law spring constant for our problem
def k : ℝ := spring_constant force compression

-- Work done to stretch the spring within elastic limit
def work_done (k : ℝ) (x : ℝ) : ℝ := 0.5 * k * x^2

-- Goal: Prove the work done is 0.18J
theorem work_done_correct : work_done k displacement = 0.18 := by
  sorry

end work_done_correct_l185_185708


namespace problem_statement_l185_185043

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185043


namespace odd_function_g_l185_185529

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185529


namespace max_slope_parabola_l185_185749

/-- Definition of a point on a parabola -/
structure ParabolaPoint (p y0 : ℝ) : Prop :=
  (on_curve : y0^2 = 2 * p * (y0^2 / (2 * p)))

/-- Definition of the focus of the parabola -/
def ParabolaFocus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

/-- Definition of the midpoint of segment PF -/
def MidPoint (P F : ℝ × ℝ) : ℝ × ℝ := ((P.1 + F.1) / 2, (P.2 + F.2) / 2)

/-- Definition of the slope of line OM -/
def SlopeOM (O M : ℝ × ℝ) : ℝ := M.2 / M.1

/-- Given conditions, max slope of the line OM -/
theorem max_slope_parabola (p y0 : ℝ) (hp : 0 < p)
  (P : ℝ × ℝ) (hP : ParabolaPoint p y0 := (hP.on_curve : y0^2 = 2 * p * (y0^2 / (2 * p))))
  (F := ParabolaFocus p)
  (M := MidPoint P F)
  (O := (0, 0)) :
  SlopeOM O M ≤ 1 :=
sorry

end max_slope_parabola_l185_185749


namespace option_B_odd_l185_185653

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185653


namespace multiple_of_cans_of_corn_l185_185965

theorem multiple_of_cans_of_corn (peas corn : ℕ) (h1 : peas = 35) (h2 : corn = 10) (h3 : peas = 10 * x + 15) : x = 2 := 
by
  sorry

end multiple_of_cans_of_corn_l185_185965


namespace odd_function_option_B_l185_185404

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185404


namespace students_answered_both_correctly_l185_185789

theorem students_answered_both_correctly 
(total_students : ℕ) 
(did_not_answer_A_correctly : ℕ) 
(answered_A_correctly_but_not_B : ℕ) 
(h1 : total_students = 50) 
(h2 : did_not_answer_A_correctly = 12) 
(h3 : answered_A_correctly_but_not_B = 30) : 
    (total_students - did_not_answer_A_correctly - answered_A_correctly_but_not_B) = 8 :=
by
    sorry

end students_answered_both_correctly_l185_185789


namespace length_PQ_sum_squares_is_constant_l185_185688

noncomputable def point := {x : ℝ // 0 ≤ x}

noncomputable def circle_O (P : point) : Prop :=
  let ρ := sqrt(P.x^2) in ρ = 2

noncomputable def circle_C (P : point) : Prop :=
  let θ := atan2 P.x P.y
  let ρ := 4 * sin(θ) in true

-- Problem (1)
theorem length_PQ (P : point) (Q : point) 
  (hO : circle_O P) (hC : circle_C Q) (h : Q = (2 * sqrt(3), Q.y) ∧ P = (2, P.y)) 
  (hθ : θ = π / 3) :
  dist P Q = 2 * sqrt(3) - 2 := sorry

-- Problem (2)
theorem sum_squares_is_constant 
  (A B D P : point) 
  (hO : circle_O A) 
  (hO : circle_O B) 
  (hC : circle_C A) 
  (hC : circle_C B) 
  (hC : circle_C D) 
  : (dist P A)^2 + (dist P B)^2 + (dist P D)^2 = 24 := sorry

end length_PQ_sum_squares_is_constant_l185_185688


namespace odd_function_proof_l185_185419

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185419


namespace triple_composition_even_l185_185754

-- Define a function g is even if for all x, g(-x) = g(x)
def is_even (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g x

-- Define the given problem
theorem triple_composition_even (g : ℝ → ℝ) (h : is_even g) : is_even (λ x, g (g (g x))) :=
by
  intro x,
  have hx : g (g (g (-x))) = g (g (g x)), from
    calc
      g (g (g (-x))) = g (g (g x)) : by rw [h (-x)]
  exact hx

end triple_composition_even_l185_185754


namespace find_A_l185_185946

variable {A B C D E F G H I J : ℕ}

/-- A telephone number has the form ABC-DEF-GHIJ, where each letter represents a different digit.
The digits in each part of the number are in decreasing order; that is, A > B > C, D > E > F, and G > H > I > J.
Furthermore, D, E, F are three consecutive digits (not necessarily even); 
G, H, I, J are four consecutive digits starting from an even number; and 
A + B + C = 10.
Prove that A = 9. -/
theorem find_A (h1 : A > B > C) 
              (h2 : D > E > F) 
              (h3 : G > H > I > J) 
              (h4 : (E = D - 1) ∧ (F = D - 2))
              (h5 : ∃ k, G = 2*k ∧ H = G - 1 ∧ I = G - 2 ∧ J = G - 3)
              (h6 : A + B + C = 10) :
              A = 9 := 
    sorry

end find_A_l185_185946


namespace option_B_odd_l185_185637

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185637


namespace odd_function_option_B_l185_185566

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185566


namespace projectile_height_49_at_t_l185_185824

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
let discriminant := b^2 - 4 * a * c in
(((-b + Real.sqrt discriminant) / (2 * a)), ((-b - Real.sqrt discriminant) / (2 * a)))

theorem projectile_height_49_at_t (y t : ℝ) (h : y = -20 * t^2 + 100 * t) : t = 0.55 :=
begin
  have h_eq : 49 = -20 * t^2 + 100 * t, from h,
  have h_eq_quad: 0 = 20 * t^2 - 100 * t + 49,
  { ring at h_eq ⊢,
    linarith, },
  let (t1, t2) := quadratic_formula 20 (-100) 49,
  have : t1 = 0.55 ∨ t2 = 0.55 := sorry,
  cases this,
  { rwa this, },
  { sorry, },
end

end projectile_height_49_at_t_l185_185824


namespace g_B_is_odd_l185_185681

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185681


namespace odd_function_g_l185_185184

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185184


namespace fraction_of_fractions_l185_185878

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l185_185878


namespace fraction_of_third_is_eighth_l185_185883

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l185_185883


namespace g_is_odd_l185_185028

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185028


namespace unique_H_value_l185_185738

theorem unique_H_value :
  ∀ (T H R E F I V S : ℕ),
    T = 8 →
    E % 2 = 1 →
    E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ F ∧ E ≠ I ∧ E ≠ V ∧ E ≠ S ∧ 
    H ≠ T ∧ H ≠ R ∧ H ≠ F ∧ H ≠ I ∧ H ≠ V ∧ H ≠ S ∧
    F ≠ T ∧ F ≠ I ∧ F ≠ V ∧ F ≠ S ∧
    I ≠ T ∧ I ≠ V ∧ I ≠ S ∧
    V ≠ T ∧ V ≠ S ∧
    S ≠ T ∧
    (8 + 8) = 10 + F ∧
    (E + E) % 10 = 6 →
    H + H = 10 + 4 →
    H = 7 := 
sorry

end unique_H_value_l185_185738


namespace Ferris_wheel_ticket_cost_l185_185871

theorem Ferris_wheel_ticket_cost
  (cost_rc : ℕ) (rides_rc : ℕ) (cost_c : ℕ) (rides_c : ℕ) (total_tickets : ℕ) (rides_fw : ℕ)
  (H1 : cost_rc = 4) (H2 : rides_rc = 3) (H3 : cost_c = 4) (H4 : rides_c = 2) (H5 : total_tickets = 21) (H6 : rides_fw = 1) :
  21 - (3 * 4 + 2 * 4) = 1 :=
by
  sorry

end Ferris_wheel_ticket_cost_l185_185871


namespace smallest_factor_of_36_l185_185802

theorem smallest_factor_of_36 :
  ∃ a b c : ℤ, a * b * c = 36 ∧ a + b + c = 4 ∧ min (min a b) c = -4 :=
by
  sorry

end smallest_factor_of_36_l185_185802


namespace odd_function_shifted_f_l185_185350

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185350


namespace g_B_is_odd_l185_185683

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185683


namespace g_is_odd_l185_185331

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185331


namespace odd_function_shifted_f_l185_185369

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185369


namespace is_odd_g_l185_185478

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185478


namespace odd_function_option_B_l185_185378

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185378


namespace b_10_eq_64_l185_185692

noncomputable def a : ℕ → ℝ
| 1       := 1
| (n + 1) := 2^n / a n

noncomputable def b (n : ℕ) : ℝ := a n + a (n + 1)

theorem b_10_eq_64 : b 10 = 64 := by
  sorry

end b_10_eq_64_l185_185692


namespace is_odd_g_l185_185486

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185486


namespace option_B_is_odd_function_l185_185316

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185316


namespace g_is_odd_l185_185016

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185016


namespace problem_statement_l185_185048

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185048


namespace trig_identity_one_trig_identity_two_l185_185970

theorem trig_identity_one :
  2 * (Real.cos (45 * Real.pi / 180)) - (3 / 2) * (Real.tan (30 * Real.pi / 180)) * (Real.cos (30 * Real.pi / 180)) + (Real.sin (60 * Real.pi / 180))^2 = Real.sqrt 2 :=
sorry

theorem trig_identity_two :
  (Real.sin (30 * Real.pi / 180))⁻¹ * (Real.sin (60 * Real.pi / 180) - Real.cos (45 * Real.pi / 180)) - Real.sqrt ((1 - Real.tan (60 * Real.pi / 180))^2) = 1 - Real.sqrt 2 :=
sorry

end trig_identity_one_trig_identity_two_l185_185970


namespace trigonometric_simplification_l185_185808

theorem trigonometric_simplification (α : ℝ) :
  (cos (α - π) / sin (π - α)) * sin (α - π / 2) * cos (3 * π / 2 - α) = - (cos α) ^ 2 :=
by 
  sorry

end trigonometric_simplification_l185_185808


namespace sequence_1_correct_l185_185831

-- Define the sequence
def sequence_1 (n : ℕ) : ℚ := 2 * (n - 1) / (2 * n - 1)

-- Prove that this is indeed the sequence: 0, 2/3, 4/5, 6/7, ...
theorem sequence_1_correct : ∀ (n : ℕ), sequence_1 (n + 1) = 
  match n with 
  | 0 => 0
  | 1 => 2 / 3
  | 2 => 4 / 5
  | 3 => 6 / 7
  | _ => sequence_1 (n + 1) :=
begin
  sorry -- proof goes here
end

end sequence_1_correct_l185_185831


namespace odd_function_g_l185_185278

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185278


namespace g_is_odd_l185_185079

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185079


namespace age_difference_l185_185917

variable (Patrick_age Michael_age Monica_age : ℕ)

theorem age_difference 
  (h1 : ∃ x : ℕ, Patrick_age = 3 * x ∧ Michael_age = 5 * x)
  (h2 : ∃ y : ℕ, Michael_age = 3 * y ∧ Monica_age = 5 * y)
  (h3 : Patrick_age + Michael_age + Monica_age = 245) :
  Monica_age - Patrick_age = 80 := by 
sorry

end age_difference_l185_185917


namespace g_is_odd_l185_185033

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185033


namespace tiffany_initial_lives_l185_185863

theorem tiffany_initial_lives (x : ℕ) 
    (H1 : x - 14 + 27 = 56) : x = 43 :=
sorry

end tiffany_initial_lives_l185_185863


namespace subgroup_generation_eq_l185_185748

variables {G : Type} [Group G] {S : Set G} {x s : G}
variable (hS : closure (Set.insert x S) = closure (Set.insert (x * s) S) ∧ closure (Set.insert x S) = closure (Set.insert (s * x) S))

theorem subgroup_generation_eq (h : s ∈ closure S) : closure (Set.insert x S) = closure (Set.insert (x * s) S) ∧ closure (Set.insert x S) = closure (Set.insert (s * x) S) :=
begin
  exact hS,
end

end subgroup_generation_eq_l185_185748


namespace length_OD1_l185_185740

-- Define the hypothesis of the problem
noncomputable def sphere_center : Point := sorry -- center O of the sphere
noncomputable def radius_sphere : ℝ := 10 -- radius of the sphere

-- Define face intersection properties
noncomputable def face_AA1D1D_radius : ℝ := 1
noncomputable def face_A1B1C1D1_radius : ℝ := 1
noncomputable def face_CDD1C1_radius : ℝ := 3

-- Define the coordinates of D1 (or in abstract form, we'll assume it is a known point)
noncomputable def segment_OD1 : ℝ := sorry -- Length of OD1 segment to be calculated

-- The main theorem to prove
theorem length_OD1 : 
  -- Given conditions
  (face_AA1D1D_radius = 1) ∧ 
  (face_A1B1C1D1_radius = 1) ∧ 
  (face_CDD1C1_radius = 3) ∧ 
  (radius_sphere = 10) →
  -- Prove the length of segment OD1 is 17
  segment_OD1 = 17 :=
by
  sorry

end length_OD1_l185_185740


namespace boys_without_calculators_l185_185720

-- Definitions based on the conditions
def total_boys : Nat := 20
def students_with_calculators : Nat := 26
def girls_with_calculators : Nat := 15

-- We need to prove the number of boys who did not bring their calculators.
theorem boys_without_calculators : (total_boys - (students_with_calculators - girls_with_calculators)) = 9 :=
by {
    -- Proof goes here
    sorry
}

end boys_without_calculators_l185_185720


namespace morgan_time_in_fog_l185_185782

-- Define constants and variables
def speed_clear := (40 : ℝ) / 60 -- miles per minute
def speed_fog := (15 : ℝ) / 60 -- miles per minute
def total_time := (50 : ℝ) -- minutes
def total_distance := (25 : ℝ) -- miles

-- Define the equation based on the given condition
theorem morgan_time_in_fog : ∃ y : ℝ, speed_clear * (total_time - y) + speed_fog * y = total_distance ∧ y = 20 :=
by
  use 20
  have h1 : speed_clear * (total_time - 20) + speed_fog * 20 = total_distance,
  calc
    speed_clear * (total_time - 20) + speed_fog * 20
      = (2 / 3) * (50 - 20) + (1 / 4) * 20 : by norm_num [speed_clear, speed_fog, total_time]
  ... = (2 / 3) * 30 + 5 : by norm_num
  ... = 20 + 5 : by norm_num
  ... = 25 : by norm_num,
  exact ⟨h1, by norm_num [total_distance]⟩

end morgan_time_in_fog_l185_185782


namespace odd_function_result_l185_185251

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185251


namespace triple_composition_even_l185_185755

-- Define a function g is even if for all x, g(-x) = g(x)
def is_even (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = g x

-- Define the given problem
theorem triple_composition_even (g : ℝ → ℝ) (h : is_even g) : is_even (λ x, g (g (g x))) :=
by
  intro x,
  have hx : g (g (g (-x))) = g (g (g x)), from
    calc
      g (g (g (-x))) = g (g (g x)) : by rw [h (-x)]
  exact hx

end triple_composition_even_l185_185755


namespace inequality_abc_l185_185747

variable {a b c : ℝ}

theorem inequality_abc (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 8) :
  (a - 2) / (a + 1) + (b - 2) / (b + 1) + (c - 2) / (c + 1) ≤ 0 := by
  sorry

end inequality_abc_l185_185747


namespace g_is_odd_l185_185512

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185512


namespace odd_function_g_l185_185013

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185013


namespace odd_function_proof_l185_185412

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185412


namespace option_B_odd_l185_185639

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185639


namespace odd_function_g_l185_185193

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185193


namespace odd_function_option_B_l185_185396

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185396


namespace odd_function_g_l185_185268

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185268


namespace problem_statement_l185_185056

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185056


namespace option_B_odd_l185_185649

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185649


namespace minimum_value_of_f_l185_185751

noncomputable def f (a b : ℝ) : ℝ :=
  a^2 + b^2 + 16 / a^2 + 4 * b / a

theorem minimum_value_of_f :
  ∀ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 → (∃ x : ℝ, x = 4 * Real.sqrt 3 ∧ f a b ≥ x) :=
by
  intros a b h,
  use 4 * Real.sqrt 3,
  split,
  {
    -- x = 4 * sqrt 3
    refl,
  },
  {
    -- f a b ≥ 4 * sqrt 3
    -- a formal proof is required here
    sorry
  }

end minimum_value_of_f_l185_185751


namespace odd_function_g_l185_185288

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185288


namespace g_is_odd_l185_185334

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185334


namespace satisfies_equation_l185_185998

theorem satisfies_equation : 
  { (x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y } = 
  { (0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2) } :=
by
  sorry

end satisfies_equation_l185_185998


namespace is_odd_g_l185_185473

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185473


namespace odd_function_shifted_f_l185_185361

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185361


namespace odd_function_l185_185153

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185153


namespace g_is_odd_l185_185210

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185210


namespace odd_function_g_l185_185286

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185286


namespace range_of_a_l185_185686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - 1

def A (a : ℝ) : set ℝ := {x | f a x = x}

def B (a : ℝ) : set ℝ := {x | f a (f a x) = x}

theorem range_of_a (a : ℝ) (h : A a = B a ∧ A a ≠ ∅) : -1/4 ≤ a ∧ a ≤ 3/4 :=
sorry

end range_of_a_l185_185686


namespace odd_function_g_l185_185000

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185000


namespace log_base_10_of_2_bounds_l185_185877

theorem log_base_10_of_2_bounds :
  (10^3 = 1000) ∧ (10^4 = 10000) ∧ (2^11 = 2048) ∧ (2^14 = 16384) →
  (3 / 11 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (2 / 7 : ℝ) :=
by
  sorry

end log_base_10_of_2_bounds_l185_185877


namespace prove_80_integers_satisfy_condition_l185_185911

theorem prove_80_integers_satisfy_condition :
  let count_valid_k := count (λ k, 100 ≤ k ∧ k < 1000 ∧
    let h := k / 100 in
    let t := (k / 10) % 10 in
    let u := k % 10 in
    (100 * u + 10 * t + h) = k + 99) {k | 100 ≤ k ∧ k < 1000} in
  count_valid_k = 80 :=
by sorry

end prove_80_integers_satisfy_condition_l185_185911


namespace total_beads_in_necklace_l185_185936

noncomputable def amethyst_beads : ℕ := 7
noncomputable def amber_beads : ℕ := 2 * amethyst_beads
noncomputable def turquoise_beads : ℕ := 19
noncomputable def total_beads : ℕ := amethyst_beads + amber_beads + turquoise_beads

theorem total_beads_in_necklace : total_beads = 40 := by
  sorry

end total_beads_in_necklace_l185_185936


namespace odd_function_l185_185128

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185128


namespace g_is_odd_l185_185097

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185097


namespace true_propositions_count_l185_185953

theorem true_propositions_count :
  let cond1 := (∀ n : ℕ, n > 0 → (1 / (n * (n + 2))) = 1 / 120 → (n = 10) ∧ (n = 1))
  let cond2 := (∀ n : ℕ, n > 0 → (2 + 3 * (n - 1) = 3 * n - 1) → (n = \sqrt{3n - 1}))
  let cond3 := (∃ k : ℤ, k = 2 ∧ (∀ n : ℤ, kn - 5 → (a_8 = 11) ∧ (a_17 = 29)))
  let cond4 := (∀ n : ℕ, n > 0 → (a_n = a_{n + 1} + 5) → (a_{n + 1} - a_n = -5))
in cond1 ∧ cond2 ∧ cond3 ∧ cond4 → (number_of_true_propositions = 4) :=
by sorry

end true_propositions_count_l185_185953


namespace problem_statement_l185_185065

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185065


namespace odd_function_option_B_l185_185560

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185560


namespace number_of_people_who_purchased_only_book_A_l185_185918

-- Define the conditions and the problem
theorem number_of_people_who_purchased_only_book_A 
    (total_A : ℕ) (total_B : ℕ) (both_AB : ℕ) (only_B : ℕ) :
    (total_A = 2 * total_B) → 
    (both_AB = 500) → 
    (both_AB = 2 * only_B) → 
    (total_B = only_B + both_AB) → 
    (total_A - both_AB = 1000) :=
by
  sorry

end number_of_people_who_purchased_only_book_A_l185_185918


namespace odd_function_g_l185_185008

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185008


namespace odd_function_l185_185135

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185135


namespace odd_function_g_l185_185287

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185287


namespace odd_function_option_B_l185_185386

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185386


namespace odd_function_option_B_l185_185565

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185565


namespace odd_function_shifted_f_l185_185357

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185357


namespace opposite_of_neg_2022_eq_2022_l185_185839

-- Define what it means to find the opposite of a number
def opposite (n : Int) : Int := -n

-- State the theorem that needs to be proved
theorem opposite_of_neg_2022_eq_2022 : opposite (-2022) = 2022 :=
by
  -- Proof would go here but we skip it with sorry
  sorry

end opposite_of_neg_2022_eq_2022_l185_185839


namespace bianca_next_day_run_l185_185966

-- Define the conditions
variable (miles_first_day : ℕ) (total_miles : ℕ)

-- Set the conditions for Bianca's run
def conditions := miles_first_day = 8 ∧ total_miles = 12

-- State the proposition we need to prove
def miles_next_day (miles_first_day total_miles : ℕ) : ℕ := total_miles - miles_first_day

-- The theorem stating the problem to prove
theorem bianca_next_day_run (h : conditions 8 12) : miles_next_day 8 12 = 4 := by
  unfold conditions at h
  simp [miles_next_day] at h
  sorry

end bianca_next_day_run_l185_185966


namespace g_B_is_odd_l185_185663

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185663


namespace symmetry_origin_l185_185791

def point : Type := ℝ × ℝ

def A : point := (3, 4)
def B : point := (-3, -4)
def origin : point := (0, 0)

theorem symmetry_origin (A B origin : point) (hA : A = (3, 4)) (hB : B = (-3, -4)) : 
  (A = (3, 4) ∧ B = (-3, -4)) → symmetric_over_origin A B :=
  sorry

end symmetry_origin_l185_185791


namespace odd_function_g_l185_185166

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185166


namespace odd_function_result_l185_185257

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185257


namespace option_B_is_odd_function_l185_185318

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185318


namespace probability_non_defective_pens_l185_185912

/-- Probability that neither pen selected from a box of 8 pens, where 2 are defective, will be defective is 15/28. -/
theorem probability_non_defective_pens : (∀ (n_total n_defective n_selected : ℕ), 
  n_total = 8 → n_defective = 2 → n_selected = 2 →
  ∃ (p : ℚ), p = 15 / 28) :=
by {
  intros n_total n_defective n_selected h_total h_defective h_selected,
  use 15 / 28,
  exact sorry
}

end probability_non_defective_pens_l185_185912


namespace g_is_odd_l185_185018

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185018


namespace odd_function_g_l185_185525

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185525


namespace g_is_odd_l185_185509

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185509


namespace union_sets_l185_185776

-- Define the sets A and B as conditions
def A : Set ℝ := {0, 1}  -- Since lg 1 = 0
def B : Set ℝ := {-1, 0}

-- Define that A union B equals {-1, 0, 1}
theorem union_sets : A ∪ B = {-1, 0, 1} := by
  sorry

end union_sets_l185_185776


namespace option_b_is_odd_l185_185117

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185117


namespace odd_function_option_B_l185_185548

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185548


namespace odd_function_g_l185_185178

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185178


namespace carrie_third_day_miles_l185_185787

theorem carrie_third_day_miles :
  let day1 := 135
      day2 := 124
      day4 := 189
      per_charge := 106
      charges := 7
      total_miles := charges * per_charge
      driven_first_two_and_fourth := day1 + (day1 + day2) + day4
  in total_miles - driven_first_two_and_fourth = 159 :=
by
  let day1 := 135
  let day2 := 124
  let day4 := 189
  let per_charge := 106
  let charges := 7
  let total_miles := charges * per_charge
  let driven_first_two_and_fourth := day1 + (day1 + day2) + day4
  show total_miles - driven_first_two_and_fourth = 159
  sorry

end carrie_third_day_miles_l185_185787


namespace age_ratio_l185_185803

/-- Given that Sandy's age after 6 years will be 30 years,
    and Molly's current age is 18 years, 
    prove that the current ratio of Sandy's age to Molly's age is 4:3. -/
theorem age_ratio (M S : ℕ) 
  (h1 : M = 18) 
  (h2 : S + 6 = 30) : 
  S / gcd S M = 4 ∧ M / gcd S M = 3 :=
by
  sorry

end age_ratio_l185_185803


namespace odd_function_l185_185131

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185131


namespace odd_function_g_l185_185177

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185177


namespace batting_average_excluding_highest_and_lowest_l185_185822

theorem batting_average_excluding_highest_and_lowest
  (batting_average : ℤ) (number_of_innings : ℤ)
  (highest_score : ℤ) (score_difference : ℤ)
  (batting_average_eq : batting_average = 60)
  (innings_eq : number_of_innings = 46)
  (highest_score_eq : highest_score = 174)
  (difference_eq : score_difference = 140) :

  (batting_average * number_of_innings - highest_score - (highest_score - score_difference)) / (number_of_innings - 2) = 58 := 
by
  rw [batting_average_eq, innings_eq, highest_score_eq, difference_eq]
  dsimp
  norm_num
  sorry

end batting_average_excluding_highest_and_lowest_l185_185822


namespace number_of_draw_matches_eq_points_difference_l185_185726

-- Definitions based on the conditions provided
def teams : ℕ := 16
def matches_per_round : ℕ := 8
def rounds : ℕ := 16
def total_points : ℕ := 222
def total_matches : ℕ := matches_per_round * rounds
def hypothetical_points : ℕ := total_matches * 2
def points_difference : ℕ := hypothetical_points - total_points

-- Theorem stating the equivalence to be proved
theorem number_of_draw_matches_eq_points_difference : 
  points_difference = 34 := 
by
  sorry

end number_of_draw_matches_eq_points_difference_l185_185726


namespace option_b_is_odd_l185_185099

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185099


namespace optionB_is_odd_l185_185575

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185575


namespace odd_function_l185_185148

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185148


namespace theater_ticket_difference_l185_185947

theorem theater_ticket_difference
  (O B V : ℕ) 
  (h₁ : O + B + V = 550) 
  (h₂ : 15 * O + 10 * B + 20 * V = 8000) : 
  B - (O + V) = 370 := 
sorry

end theater_ticket_difference_l185_185947


namespace problem_is_odd_function_proof_l185_185441

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185441


namespace option_B_is_odd_function_l185_185321

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185321


namespace g_even_l185_185742

-- Define the function g
def g (x : ℝ) : ℝ := 5 / (3 * x ^ 4 - 7)

-- State that g is an even function
theorem g_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  rw [neg_pow, pow_four, neg_mul, myId]
  rw [mul_four, mul_zero, powe_four]
  unfold g_twice
  congr
  sorry

end g_even_l185_185742


namespace option_b_is_odd_l185_185114

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185114


namespace odd_function_g_l185_185293

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185293


namespace sum_repeating_decimals_l185_185969

noncomputable def repeating_decimals_to_fractions : Prop :=
  (0.\overline{6} : ℚ) = 2/3 ∧ (0.\overline{3} : ℚ) = 1/3

theorem sum_repeating_decimals :
  repeating_decimals_to_fractions →
  (0.\overline{6} + 0.\overline{3} = (1 : ℚ)) := 
by
    intro h
    cases h with h1 h2
    rw [h1, h2]
    norm_num

end sum_repeating_decimals_l185_185969


namespace min_buses_needed_l185_185928

-- Given definitions from conditions
def students_per_bus : ℕ := 45
def total_students : ℕ := 495

-- The proposition to prove
theorem min_buses_needed : ∃ n : ℕ, 45 * n ≥ 495 ∧ (∀ m : ℕ, 45 * m ≥ 495 → n ≤ m) :=
by
  -- Preliminary calculations that lead to the solution
  let n := total_students / students_per_bus
  have h : total_students % students_per_bus = 0 := by sorry
  
  -- Conclude that the minimum n so that 45 * n ≥ 495 is indeed 11
  exact ⟨n, by sorry, by sorry⟩

end min_buses_needed_l185_185928


namespace integer_solutions_l185_185996

-- Define the problem statement in Lean
theorem integer_solutions :
  {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y) ∧ x^2 + x = y^4 + y^3 + y^2 + y} =
  {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)} :=
by
  sorry

end integer_solutions_l185_185996


namespace fraction_saved_l185_185907

variable {P : ℝ} (hP : P > 0)

theorem fraction_saved (f : ℝ) (hf0 : 0 ≤ f) (hf1 : f ≤ 1) (condition : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  sorry

end fraction_saved_l185_185907


namespace odd_function_shifted_f_l185_185377

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185377


namespace odd_function_g_l185_185279

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185279


namespace g_is_odd_l185_185095

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185095


namespace find_x_l185_185852

theorem find_x (n : ℕ) (h1 : n = 125) (h2 : ∃ k : ℕ, log x n + log 5 n = k) : x = 5 :=
sorry

end find_x_l185_185852


namespace projectile_height_49_at_t_l185_185823

noncomputable def quadratic_formula (a b c : ℝ) : (ℝ × ℝ) :=
let discriminant := b^2 - 4 * a * c in
(((-b + Real.sqrt discriminant) / (2 * a)), ((-b - Real.sqrt discriminant) / (2 * a)))

theorem projectile_height_49_at_t (y t : ℝ) (h : y = -20 * t^2 + 100 * t) : t = 0.55 :=
begin
  have h_eq : 49 = -20 * t^2 + 100 * t, from h,
  have h_eq_quad: 0 = 20 * t^2 - 100 * t + 49,
  { ring at h_eq ⊢,
    linarith, },
  let (t1, t2) := quadratic_formula 20 (-100) 49,
  have : t1 = 0.55 ∨ t2 = 0.55 := sorry,
  cases this,
  { rwa this, },
  { sorry, },
end

end projectile_height_49_at_t_l185_185823


namespace max_colored_cells_100x100_l185_185721

theorem max_colored_cells_100x100
  (n : ℕ) (H_n : n = 100)
  (cells : fin n → fin n → bool)
  (H_unique_colored :
    ∀ (i j : fin n), cells i j = tt →
      (∀ k : fin n, k ≠ i → cells k j = ff) ∧
      (∀ k : fin n, k ≠ j → cells i k = ff)) :
  ∃ m : ℕ, m = 198 ∧
    (∀ c : ℕ, (∀ (i j : fin n), cells i j = tt →
      (∀ k : fin n, k ≠ i → cells k j = ff) ∧
      (∀ k : fin n, k ≠ j → cells i k = ff)) →
      c ≤ m) :=
sorry

end max_colored_cells_100x100_l185_185721


namespace problem_is_odd_function_proof_l185_185436

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185436


namespace option_B_is_odd_function_l185_185306

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185306


namespace incorrect_calculation_l185_185897

theorem incorrect_calculation :
  ¬ (sqrt 7 + sqrt 3 = sqrt 10) ∧ 
  (sqrt 3 * sqrt 5 = sqrt 15) ∧ 
  (sqrt 6 / sqrt 3 = sqrt 2) ∧ 
  ((-sqrt 3) ^ 2 = 3) := by
  sorry

end incorrect_calculation_l185_185897


namespace option_B_odd_l185_185636

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185636


namespace odd_function_shifted_f_l185_185363

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185363


namespace odd_function_g_l185_185275

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185275


namespace g_B_is_odd_l185_185665

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185665


namespace odd_function_option_B_l185_185564

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185564


namespace odd_function_result_l185_185254

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185254


namespace problem_statement_l185_185062

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185062


namespace smallest_factor_of_36_sum_4_l185_185800

theorem smallest_factor_of_36_sum_4 : ∃ a b c : ℤ, (a * b * c = 36) ∧ (a + b + c = 4) ∧ (a = -4 ∨ b = -4 ∨ c = -4) :=
by
  sorry

end smallest_factor_of_36_sum_4_l185_185800


namespace option_b_is_odd_l185_185109

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185109


namespace fraction_of_area_shaded_is_half_l185_185733

-- Define the key elements: b represents the common side length of triangles and h the height of the rectangle
variables (b h : ℝ)

-- Define the areas based on the conditions provided
def area_rectangle : ℝ := 3 * b * h
def area_triangle : ℝ := (1 / 2) * b * h
def total_shaded_area : ℝ := 3 * area_triangle b h

-- The statement to prove
theorem fraction_of_area_shaded_is_half (hb : b > 0) (hh : h > 0) :
  total_shaded_area b h / area_rectangle b h = 1 / 2 :=
by
  sorry

end fraction_of_area_shaded_is_half_l185_185733


namespace g_is_odd_l185_185333

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185333


namespace delaney_bus_miss_theorem_l185_185975

def delaneyMissesBus : Prop :=
  let busDeparture := 8 * 60               -- bus departure time in minutes (8:00 a.m.)
  let travelTime := 30                     -- travel time in minutes
  let departureTime := 7 * 60 + 50         -- departure time from home in minutes (7:50 a.m.)
  let arrivalTime := departureTime + travelTime -- arrival time at the pick-up point
  arrivalTime - busDeparture = 20 -- he misses the bus by 20 minutes

theorem delaney_bus_miss_theorem : delaneyMissesBus := sorry

end delaney_bus_miss_theorem_l185_185975


namespace odd_function_l185_185150

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185150


namespace price_per_ticket_is_10_l185_185867

-- Define the initial cost constant
def initial_cost : ℝ := 100000

-- Define the daily operating cost as 1% of the initial cost
def daily_operating_cost : ℝ := 0.01 * initial_cost

-- Define the number of tickets sold per day
def tickets_per_day : ℕ := 150

-- Define the number of days to break even
def days_to_break_even : ℕ := 200

-- Define the total cost after 200 days
def total_cost : ℝ := initial_cost + (daily_operating_cost * days_to_break_even)

-- Define the daily revenue needed to break even in 200 days
def daily_revenue_needed : ℝ := total_cost / days_to_break_even

-- Given the daily revenue needed and the number of tickets sold per day,
-- define the price per ticket
def price_per_ticket : ℝ := daily_revenue_needed / tickets_per_day

-- The theorem stating that the price per ticket is $10
theorem price_per_ticket_is_10 : price_per_ticket = 10 := by
  sorry

end price_per_ticket_is_10_l185_185867


namespace g_is_odd_l185_185023

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185023


namespace odd_function_g_l185_185176

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185176


namespace odd_function_g_l185_185292

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185292


namespace vector_subtraction_l185_185699

theorem vector_subtraction (m n : ℝ) (λ : ℝ) 
  (h1 : abs (λ) = 2 * sqrt 5)
  (h2 : (m, n) = (λ, -2 * λ)) 
  (h3 : λ < 0) : 
  m - n = -6 := 
by
  -- need to prove m - n = -6
  sorry

end vector_subtraction_l185_185699


namespace solve_system_l185_185704

noncomputable def solutions (a b c : ℝ) : Prop :=
  a^4 - b^4 = c ∧ b^4 - c^4 = a ∧ c^4 - a^4 = b

theorem solve_system :
  { (a, b, c) | solutions a b c } =
  { (0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0) } :=
by
  sorry

end solve_system_l185_185704


namespace projectile_reaches_49_first_time_at_1_point_4_l185_185826

-- Define the equation for the height of the projectile
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t

-- State the theorem to prove
theorem projectile_reaches_49_first_time_at_1_point_4 :
  ∃ t : ℝ, height t = 49 ∧ (∀ t' : ℝ, height t' = 49 → t ≤ t') :=
sorry

end projectile_reaches_49_first_time_at_1_point_4_l185_185826


namespace bicycle_trip_l185_185777

theorem bicycle_trip
  (v_Af : ℝ) (v_Bf : ℝ) (v_Cf : ℝ)
  (v_Ab : ℝ) (v_Bb : ℝ) (v_Cb : ℝ)
  (A B C : ℝ)
  (total_distance : ℝ) :
  v_Af = 4 ∧ v_Bf = 5 ∧ v_Cf = 3 ∧
  v_Ab = 10 ∧ v_Bb = 8 ∧ v_Cb = 12 ∧
  total_distance = 20 ∧
  A + B + C = total_distance →
  A = 7 + (11/27 : ℝ) ∧ B = 1 + (13/27 : ℝ) ∧ C = 11 + (3/27 : ℝ) :=
begin
  sorry,
end

end bicycle_trip_l185_185777


namespace odd_function_result_l185_185239

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185239


namespace angle_difference_l185_185725

theorem angle_difference (A B : ℝ) 
  (h1 : A = 85) 
  (h2 : A + B = 180) : B - A = 10 := 
by sorry

end angle_difference_l185_185725


namespace g_is_odd_l185_185511

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185511


namespace option_B_odd_l185_185652

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185652


namespace scientific_notation_l185_185989

theorem scientific_notation (num : ℝ) (significant_figures : ℝ) (exponent : ℤ) :
  num = 0.0000036 → significant_figures = 3.6 → exponent = -6 → num = significant_figures * (10 ^ exponent) :=
by
  intros hnum hfig hexp
  rw [hnum, hfig, hexp]
  norm_num
  exact rfl

end scientific_notation_l185_185989


namespace average_of_15_25_x_is_23_l185_185820

theorem average_of_15_25_x_is_23 : ∃ x : ℝ, (15 + 25 + x) / 3 = 23 ∧ x = 29 := by
  use 29
  split
  {
    have h : (15 + 25 + 29) / 3 = 23 := by norm_num
    exact h
  }
  {
    norm_num
  }

end average_of_15_25_x_is_23_l185_185820


namespace g_is_odd_l185_185620

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185620


namespace g_B_is_odd_l185_185679

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185679


namespace g_is_odd_l185_185495

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185495


namespace smallest_lucky_integer_l185_185957

def is_lucky (B : ℤ) : Prop :=
  ∃ (k : ℕ), ∃ (m : ℤ), (m ≥ B) ∧ (list.sum (list.map (λ i, m + i) (list.range (k + 1))) = 2023)

theorem smallest_lucky_integer : ∃ (B : ℤ), is_lucky B ∧ ∀ B', is_lucky B' → B ≤ B'  :=
by
  existsi (-2022 : ℤ)
  split
  {
    sorry
  }
  {
    introv HB'
    cases HB' with k₁ HB''
    cases HB'' with m₁ Hm
    sorry
  }

end smallest_lucky_integer_l185_185957


namespace g_B_is_odd_l185_185684

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185684


namespace g_is_odd_l185_185222

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185222


namespace option_B_odd_l185_185647

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185647


namespace option_B_is_odd_function_l185_185301

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185301


namespace eccentricity_is_sqrt_10_div_2_l185_185771

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  let P := λ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ x^2 + y^2 = a^2 + b^2 in
  let F1 := (-c, 0) -- left focus of hyperbola
  let F2 := (c, 0)  -- right focus of hyperbola
  let pf1_pf2 : ∀ (P : ℝ × ℝ), |P.1 - F1.1| = 3 * |P.1 - F2.1| in
  let r := sqrt (a^2 + b^2) in
  sqrt (b^2 + c^2) / a -- eccentricity e = sqrt (1 + b^2) 

-- Main theorem statement
theorem eccentricity_is_sqrt_10_div_2
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (P : ℝ × ℝ) (hP : P.1^2 / a^2 - P.2^2 / b^2 = 1 ∧ P.1^2 + P.2^2 = a^2 + b^2)
  (F1 F2 : ℝ × ℝ) (h_F1 : F1 = (-sqrt(a^2 + b^2), 0)) (h_F2 : F2 = (sqrt(a^2 + b^2), 0))
  (h_relation : |P.1 - F1.1| = 3 * |P.1 - F2.1|) :
  hyperbola_eccentricity a b h_a h_b = sqrt(10) / 2 :=
sorry

end eccentricity_is_sqrt_10_div_2_l185_185771


namespace is_odd_g_l185_185480

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185480


namespace largest_digit_divisible_by_9_l185_185737

theorem largest_digit_divisible_by_9 : ∀ (B : ℕ), B < 10 → (∃ n : ℕ, 9 * n = 5 + B + 4 + 8 + 6 + 1) → B = 9 := by
  sorry

end largest_digit_divisible_by_9_l185_185737


namespace optionB_is_odd_l185_185585

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185585


namespace team_A_more_uniform_l185_185817

noncomputable def average_height : ℝ := 2.07

variables (S_A S_B : ℝ) (h_variance : S_A^2 < S_B^2)

theorem team_A_more_uniform : true ∧ false :=
by
  sorry

end team_A_more_uniform_l185_185817


namespace g_is_odd_l185_185083

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185083


namespace largest_angle_in_triangle_l185_185886

theorem largest_angle_in_triangle (x : ℝ) :
  (3 * x + x + 6 * x = 180) → (∃ x, 6 * x = 108) :=
by
  intro h
  use 18
  sorry

end largest_angle_in_triangle_l185_185886


namespace even_triple_composition_l185_185757

theorem even_triple_composition {g : ℝ → ℝ} (h_even : ∀ x : ℝ, g(-x) = g(x)) :
  ∀ x : ℝ, g(g(g(x))) = g(g(g(-x))) :=
by
  intros
  sorry

end even_triple_composition_l185_185757


namespace odd_function_result_l185_185258

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185258


namespace odd_function_g_l185_185201

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185201


namespace problem1_problem2_l185_185968

noncomputable def expression1 : Real := (2 + 1/4) ^ (1/2) - (9.6 ^ 0) - (-(3 + 3/8)) ^ (-2 / 3) + (1.5) ^ (-2)
noncomputable def expression2 : Real := 2 * log 2 25 * log 3 (2 * sqrt 2) * log 5 9

theorem problem1 : expression1 = 1 / 2 := sorry
theorem problem2 : expression2 = 6 := sorry

end problem1_problem2_l185_185968


namespace problem_statement_l185_185053

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185053


namespace mean_age_of_oldest_three_l185_185818

theorem mean_age_of_oldest_three (x : ℕ) (h : (x + (x + 1) + (x + 2)) / 3 = 6) : 
  (((x + 4) + (x + 5) + (x + 6)) / 3 = 10) := 
by
  sorry

end mean_age_of_oldest_three_l185_185818


namespace odd_function_g_l185_185154

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185154


namespace g_is_odd_l185_185040

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185040


namespace odd_function_option_B_l185_185405

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185405


namespace sum_of_ages_l185_185962

/-- Define the necessary variables -/
def Beckett_age := 12
def Olaf_age := Beckett_age ^ 2
def Shannen_age := (1/2 : ℝ) * Olaf_age - 2
def Jack_age := 5 + 2 * (Shannen_age + Olaf_age^(1/3 : ℝ))
def Emma_age := (Beckett_age^(1/2 : ℝ) + Shannen_age^(1/2 : ℝ)) * (Jack_age - Olaf_age)

/-- The sum of the ages of all 5 people is 615 -/
theorem sum_of_ages : Beckett_age + Olaf_age + Shannen_age + (Jack_age : ℝ) + (Emma_age : ℝ) = 615 := by
  sorry

end sum_of_ages_l185_185962


namespace odd_function_option_B_l185_185384

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185384


namespace g_is_odd_l185_185341

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185341


namespace odd_function_g_l185_185207

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185207


namespace optionB_is_odd_l185_185592

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185592


namespace odd_function_shifted_f_l185_185374

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185374


namespace correct_calculation_l185_185896

theorem correct_calculation (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 :=
by sorry

end correct_calculation_l185_185896


namespace option_B_odd_l185_185632

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185632


namespace sum_of_minimal_positive_solutions_l185_185833

noncomputable def greatest_integer_function (x : ℝ) : ℝ := ⌊x⌋

theorem sum_of_minimal_positive_solutions :
  (let x1 := 1 + 1/2 in
   let x2 := 2 + 1/4 in
   let x3 := 3 + 1/6 in
   x1 + x2 + x3 = 6 + 11/12) := by
  sorry

end sum_of_minimal_positive_solutions_l185_185833


namespace odd_function_g_l185_185280

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185280


namespace problem_is_odd_function_proof_l185_185461

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185461


namespace option_B_odd_l185_185641

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185641


namespace product_of_m_and_M_l185_185759

theorem product_of_m_and_M : 
  ∀ (x y : ℝ), 3 * x^2 + 6 * x * y + 4 * y^2 = 1 
    → let m := (1 - Real.sqrt 3) / 2
    let M := (1 + Real.sqrt 3) / 2
    in m * M = -1 / 2 :=
by
  sorry

end product_of_m_and_M_l185_185759


namespace g_is_odd_l185_185514

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185514


namespace problem_statement_l185_185051

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185051


namespace odd_function_proof_l185_185429

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185429


namespace is_odd_g_l185_185481

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185481


namespace fraction_of_fractions_l185_185880

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l185_185880


namespace g_is_odd_l185_185070

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185070


namespace odd_function_l185_185145

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185145


namespace g_is_odd_l185_185606

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185606


namespace g_is_odd_l185_185628

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185628


namespace solution_l185_185977

theorem solution :
  ∀ (x : ℝ), x ≠ 0 → (9 * x) ^ 18 = (27 * x) ^ 9 → x = 1 / 3 :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solution_l185_185977


namespace g_is_odd_l185_185339

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185339


namespace odd_function_g_l185_185285

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185285


namespace odd_function_shifted_f_l185_185364

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185364


namespace odd_function_proof_l185_185408

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185408


namespace odd_function_g_l185_185192

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185192


namespace number_of_subsets_of_intersection_l185_185709

def M : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 1 }
def N : Set (ℝ × ℝ) := { p | p.1 - p.2 = 0 }

theorem number_of_subsets_of_intersection : 
  ∃ (M N : Set (ℝ × ℝ)), (M = { p | p.1^2 + p.2^2 = 1 }) ∧ (N = { p | p.1 - p.2 = 0 }) ∧ 
  (M ∩ N).to_finset.card = 2^2 :=
by
  use M
  use N
  split
  · exact rfl
  split
  · exact rfl
  -- Real proof skipped
  · sorry


end number_of_subsets_of_intersection_l185_185709


namespace odd_function_g_l185_185532

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185532


namespace initial_maple_trees_l185_185861

theorem initial_maple_trees
  (initial_maple_trees : ℕ)
  (to_be_planted : ℕ)
  (final_maple_trees : ℕ)
  (h1 : to_be_planted = 9)
  (h2 : final_maple_trees = 11) :
  initial_maple_trees + to_be_planted = final_maple_trees → initial_maple_trees = 2 := 
by 
  sorry

end initial_maple_trees_l185_185861


namespace odd_function_option_B_l185_185563

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185563


namespace max_min_adj_diff_l185_185762

noncomputable def min_adj_diff (S : List ℕ) : ℕ :=
List.minimum ((List.zipWith (λ a b => abs (a - b)) S.tail S.init))

theorem max_min_adj_diff (n : ℕ) (S : List ℕ) (hS : S ~ List.range (n + 1)) :
    min_adj_diff S ≤ n / 2 :=
sorry

end max_min_adj_diff_l185_185762


namespace g_is_odd_l185_185605

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185605


namespace g_is_odd_l185_185232

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185232


namespace g_is_odd_l185_185344

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185344


namespace g_is_odd_l185_185499

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185499


namespace g_is_odd_l185_185325

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185325


namespace g_is_odd_l185_185492

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185492


namespace odd_function_shifted_f_l185_185362

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185362


namespace g_is_odd_l185_185336

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185336


namespace odd_function_g_l185_185282

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185282


namespace odd_function_shifted_f_l185_185365

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185365


namespace odd_function_option_B_l185_185382

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185382


namespace option_B_odd_l185_185635

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185635


namespace is_odd_g_l185_185464

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185464


namespace satisfies_equation_l185_185999

theorem satisfies_equation : 
  { (x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y } = 
  { (0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2) } :=
by
  sorry

end satisfies_equation_l185_185999


namespace factory_output_in_2009_l185_185840

theorem factory_output_in_2009 (a : ℝ) (annual_growth_rate : ℝ) (h_growth_rate : annual_growth_rate = 0.08) :
  let output_2009 := a * (1 + annual_growth_rate)^5
  (output_2009 = a * (1 + 0.08)^5) :=
by
  simp [h_growth_rate]
  sorry

end factory_output_in_2009_l185_185840


namespace age_of_Rahim_l185_185719

theorem age_of_Rahim (R : ℕ) (h1 : ∀ (a : ℕ), a = (R + 1) → (a + 5) = (2 * R)) (h2 : ∀ (a : ℕ), a = (R + 1) → a = R + 1) :
  R = 6 := by
  sorry

end age_of_Rahim_l185_185719


namespace g_is_odd_l185_185071

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185071


namespace is_odd_g_l185_185488

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185488


namespace even_triple_composition_l185_185756

theorem even_triple_composition {g : ℝ → ℝ} (h_even : ∀ x : ℝ, g(-x) = g(x)) :
  ∀ x : ℝ, g(g(g(x))) = g(g(g(-x))) :=
by
  intros
  sorry

end even_triple_composition_l185_185756


namespace trajectory_of_M_l185_185697

-- Conditions
def C1 : set (ℝ × ℝ) := {p | (p.1 - 4)^2 + p.2^2 = 169}
def C2 : set (ℝ × ℝ) := {p | (p.1 + 4)^2 + p.2^2 = 9}
def inside_c1 (x y : ℝ) : Prop := ∃ r, (x - 4)^2 + y^2 < 169 ∧ r ≥ 0
def tangent_inside_c1 (x y : ℝ) (r : ℝ) : Prop := (x - 4)^2 + y^2 = (13 - r)^2
def tangent_outside_c2 (x y : ℝ) (r : ℝ) : Prop := (x + 4)^2 + y^2 = (r + 3)^2

-- Proof Statement
theorem trajectory_of_M (M : ℝ × ℝ) :
  (∃ x y r, inside_c1 x y ∧ tangent_inside_c1 x y r ∧ tangent_outside_c2 x y r) →
  (M.1^2 / 64 + M.2^2 / 48 = 1) := sorry

end trajectory_of_M_l185_185697


namespace line_intersects_y_axis_at_origin_l185_185932

theorem line_intersects_y_axis_at_origin 
  (x₁ y₁ x₂ y₂ : ℤ) 
  (h₁ : (x₁, y₁) = (3, 9)) 
  (h₂ : (x₂, y₂) = (-7, -21)) 
  : 
  ∃ y : ℤ, (0, y) = (0, 0) := by
  sorry

end line_intersects_y_axis_at_origin_l185_185932


namespace move_right_number_line_l185_185783

theorem move_right_number_line (x : ℤ) (h : x = 3) : x + 4 = 7 :=
by
  rw h
  rfl

end move_right_number_line_l185_185783


namespace option_b_is_odd_l185_185123

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185123


namespace solve_m_l185_185711

theorem solve_m (m : ℝ) : 
  (m - 3) * x^2 - 3 * x + m^2 = 9 → m^2 - 9 = 0 → m = -3 :=
by
  sorry

end solve_m_l185_185711


namespace g_is_odd_l185_185088

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185088


namespace problem_is_odd_function_proof_l185_185450

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185450


namespace g_is_odd_l185_185217

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185217


namespace count_identical_pairs_l185_185952

def y1a (x : ℝ) := if x ≠ -3 then (x + 3)*(x - 5) / (x + 3) else 0
def y2a (x : ℝ) := x - 5

def y1b (x : ℝ) := sqrt (x + 1) * sqrt (x - 1)
def y2b (x : ℝ) := sqrt ((x + 1) * (x - 1))

def f3 (x : ℝ) := x
def g3 (x : ℝ) := abs x  -- sqrt(x^2) is equivalent to |x| in ℝ

def f4 (x : ℝ) := (x^4 - x^3)^(1/3)
def F4 (x : ℝ) := x * (x - 1)^(1/3)

theorem count_identical_pairs :
  (∀ x : ℝ, (x ≠ -3 → y1a x = y2a x)) ∧
  (∀ x : ℝ, (x ≥ 1 → y1b x = y2b x)) ∧
  (∀ x : ℝ, f3 x ≠ g3 x) ∧
  (∀ x : ℝ, f4 x = F4 x) →
  (count (λ (p : ℕ × ℕ), ∀ x : ℝ, (match p.1, p.2 with
    | 1, 1 => y1a x = y2a x
    | 2, 2 => y1b x = y2b x
    | 3, 3 => f3 x = g3 x
    | 4, 4 => f4 x = F4 x
    | _, _ => false
  end)) = 1) :=
sorry

end count_identical_pairs_l185_185952


namespace series_zero_distance_l185_185766

def series_phi (m : ℕ) : Set ℝ := sorry -- Define the series here

variables {m : ℕ} {x y : ℝ}
hypothesis h1 : x ∈ series_phi m
hypothesis h2 : y ∈ series_phi m
hypothesis h3 : ∃ z ∈ series_phi m, z = 0 ∧ (abs (x - z) = abs (y - z))

theorem series_zero_distance :
  x + y = 0 ∨ x - y = 0 :=
sorry

end series_zero_distance_l185_185766


namespace largest_number_l185_185899

theorem largest_number :
  let A := (1:ℝ) / 2
      B := 37.5 / 100
      C := (7:ℝ) / 22
      D := (Real.pi) / 10
  in A > B ∧ A > C ∧ A > D := 
by
  let A : ℝ := (1:ℝ) / 2
  let B : ℝ := 37.5 / 100
  let C : ℝ := (7:ℝ) / 22
  let D : ℝ := (Real.pi) / 10
  sorry

end largest_number_l185_185899


namespace g_is_odd_l185_185216

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185216


namespace subset_difference_union_triangle_subset_union_triangle_union_intersection_union_triangle_difference_union_l185_185797

variable {α : Type*} (A B C D : Set α)

-- 1. A - D ⊆ (A - B) ∪ (B - C) ∪ (C - D)
theorem subset_difference_union :
  A \ D ⊆ (A \ B) ∪ (B \ C) ∪ (C \ D) := sorry

-- 2. A △ C ⊆ (A △ B) ∪ (B △ C)
theorem triangle_subset_union_triangle :
  A ∆ C ⊆ (A ∆ B) ∪ (B ∆ C) := sorry

-- 3. (A ∪ B) ∩ (B ∪ C) ∩ (C ∪ A) = (A ∩ B) ∪ (B ∩ C) ∪ (C ∩ A)
theorem union_intersection_union :
  (A ∪ B) ∩ (B ∪ C) ∩ (C ∪ A) = (A ∩ B) ∪ (B ∩ C) ∪ (C ∩ A) := sorry

-- 4. (A - B) △ B = A ∪ B
theorem triangle_difference_union :
  (A \ B) ∆ B = A ∪ B := sorry

end subset_difference_union_triangle_subset_union_triangle_union_intersection_union_triangle_difference_union_l185_185797


namespace odd_function_l185_185133

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185133


namespace odd_function_option_B_l185_185401

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185401


namespace odd_function_result_l185_185252

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185252


namespace geometric_series_first_term_is_35_l185_185956

noncomputable def infinite_geometric_series_first_term (a r S : ℝ) : ℝ :=
  if |r| < 1 then a / (1 - r) else 0

theorem geometric_series_first_term_is_35 :
  ∀ (a : ℝ),
  let r := 1 / 6
  let S := 42
  ∃ (a : ℝ), infinite_geometric_series_first_term a r S = S ∧ a = 35 :=
by 
  intro a
  let r := (1 : ℝ) / 6
  let S := 42
  use 35
  dsimp [infinite_geometric_series_first_term]
  have hr : |r| < 1 := by norm_num
  rw if_pos hr
  have h1 : (1 : ℝ) - r = 5 / 6 := by norm_num
  rw [h1, div_mul_eq_mul_div]
  norm_num
  split
  · refl
  · refl
  sorry

end geometric_series_first_term_is_35_l185_185956


namespace g_B_is_odd_l185_185677

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185677


namespace g_is_odd_l185_185092

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185092


namespace sum_of_undefined_domain_values_l185_185980

theorem sum_of_undefined_domain_values :
  ∀ (x : ℝ), (x = 0 ∨ (1 + 1/x) = 0 ∨ (1 + 1/(1 + 1/x)) = 0 ∨ (1 + 1/(1 + 1/(1 + 1/x))) = 0) →
  x = 0 ∧ x = -1 ∧ x = -1/2 ∧ x = -1/3 →
  (0 + (-1) + (-1/2) + (-1/3) = -11/6) := sorry

end sum_of_undefined_domain_values_l185_185980


namespace find_a_l185_185722

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (t, sqrt 3 * t + 2)

noncomputable def circle_C (a θ : ℝ) : ℝ × ℝ :=
  (a * cos θ, a * sin θ)

theorem find_a (a : ℝ) (h : a > 0) :
  (∀ (θ t : ℝ),
    sqrt ((circle_C a θ).fst^2 + (circle_C a θ).snd^2) = a ∧
    abs ((line_l t).fst * (1/2) + (line_l t).snd * (sqrt 3 / 2) - 2) / sqrt ((1/2)^2 + (sqrt 3 / 2)^2) = 3) →
  a = 1 :=
by sorry

end find_a_l185_185722


namespace problem_is_odd_function_proof_l185_185434

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185434


namespace option_b_is_odd_l185_185115

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185115


namespace odd_function_g_l185_185518

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185518


namespace problem_statement_l185_185044

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185044


namespace option_b_is_odd_l185_185122

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185122


namespace odd_function_l185_185142

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185142


namespace option_B_is_odd_function_l185_185320

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185320


namespace g_is_odd_l185_185041

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185041


namespace g_is_odd_l185_185037

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185037


namespace find_k_l185_185872

noncomputable def point_P := (9, 12) : ℝ × ℝ
def larger_circle_radius : ℝ := Real.sqrt (point_P.1^2 + point_P.2^2)
noncomputable def QR_distance : ℝ := 5
def smaller_circle_radius : ℝ := larger_circle_radius - QR_distance

theorem find_k (k : ℝ) (S := (0, k) : ℝ × ℝ) :
  (S.1 = 0 ∧ Real.sqrt (S.1 ^ 2 + S.2 ^ 2) = smaller_circle_radius) →
  (k = 10 ∨ k = -10) :=
by
  sorry

end find_k_l185_185872


namespace radius_of_larger_ball_l185_185846

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem radius_of_larger_ball :
  (six_ball_volume : volume_of_sphere 2 * 6 = volume_of_sphere R) →
  R = 2 * Real.cbrt 3 := by
  sorry

end radius_of_larger_ball_l185_185846


namespace g_B_is_odd_l185_185671

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185671


namespace odd_function_result_l185_185248

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185248


namespace odd_function_g_l185_185186

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185186


namespace problem_statement_l185_185063

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185063


namespace odd_function_g_l185_185540

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185540


namespace cookies_to_milk_l185_185864

theorem cookies_to_milk (milk_quarts : ℕ) (cookies : ℕ) (cups_in_quart : ℕ) 
  (H : milk_quarts = 3) (C : cookies = 24) (Q : cups_in_quart = 4) : 
  ∃ x : ℕ, x = 3 ∧ ∀ y : ℕ, y = 6 → x = (milk_quarts * cups_in_quart * y) / cookies := 
by {
  sorry
}

end cookies_to_milk_l185_185864


namespace g_is_odd_l185_185024

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185024


namespace min_strikes_to_hit_ship_l185_185785

/-- On a 10 x 10 game board for "Battleship", a four-cell "ship" is placed.
    Prove that the minimum number of "strikes" needed to hit the ship is 24.
-/
theorem min_strikes_to_hit_ship : 
  (∀ (board : fin 10 × fin 10 → Prop),
    (∃ (ship : fin 10 × fin 10 → Prop), 
      (∀ coord, ship coord → board coord ∧ ship) 
      ∧ 
      ((∀ s, (count ship board s = 4 → board coord = true)) ∨ 
       (count strikes ship < 24))) 
      → (24 ≤ strikes) := sorry

end min_strikes_to_hit_ship_l185_185785


namespace odd_function_option_B_l185_185380

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185380


namespace is_odd_g_l185_185485

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185485


namespace odd_function_g_l185_185542

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185542


namespace option_b_is_odd_l185_185100

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185100


namespace odd_function_g_l185_185007

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185007


namespace odd_function_proof_l185_185432

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185432


namespace optionB_is_odd_l185_185595

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185595


namespace optionB_is_odd_l185_185580

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185580


namespace odd_for_fourth_team_l185_185731

open Real

noncomputable def first_team_odd : ℝ := 1.28
noncomputable def second_team_odd : ℝ := 5.23
noncomputable def third_team_odd : ℝ := 3.25
noncomputable def bet_amount : ℝ := 5.0
noncomputable def expected_winnings : ℝ := 223.0072

theorem odd_for_fourth_team : 
  let combined_odds_three_teams := first_team_odd * second_team_odd * third_team_odd in
  let total_odds := expected_winnings / bet_amount in
  let odd_fourth_team := total_odds / combined_odds_three_teams in
  odd_fourth_team = 2.061 :=
by 
  let combined_odds_three_teams := first_team_odd * second_team_odd * third_team_odd
  let total_odds := expected_winnings / bet_amount
  let odd_fourth_team := total_odds / combined_odds_three_teams
  show odd_fourth_team = 2.061
  sorry

end odd_for_fourth_team_l185_185731


namespace toll_for_18_wheel_truck_l185_185855

-- Define the number of wheels on the front axle and the other axles
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def total_wheels : ℕ := 18

-- Define the toll formula
def toll (x : ℕ) : ℝ := 3.50 + 0.50 * (x - 2)

-- Calculate the number of axles for the 18-wheel truck
def num_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the expected toll for the given number of axles
def expected_toll : ℝ := 5.00

-- State the theorem
theorem toll_for_18_wheel_truck : toll num_axles = expected_toll := by
    sorry

end toll_for_18_wheel_truck_l185_185855


namespace odd_function_g_l185_185274

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185274


namespace payback_duration_l185_185746

-- Define constants for the problem conditions
def C : ℝ := 25000
def R : ℝ := 4000
def E : ℝ := 1500

-- Formal statement to be proven
theorem payback_duration : C / (R - E) = 10 := 
by
  sorry

end payback_duration_l185_185746


namespace odd_function_g_l185_185185

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185185


namespace odd_function_g_l185_185172

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185172


namespace notebook_cost_l185_185938

theorem notebook_cost (n p : ℝ) (h1 : n + p = 2.40) (h2 : n = 2 + p) : n = 2.20 := by
  sorry

end notebook_cost_l185_185938


namespace option_B_odd_l185_185648

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185648


namespace g_is_odd_l185_185626

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185626


namespace g_is_odd_l185_185508

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185508


namespace sum_of_distances_l185_185687

noncomputable def line (t : ℝ) : ℝ × ℝ := (sqrt 3 * t, 2 - t)
def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def point_A : ℝ × ℝ := (0, 2)

theorem sum_of_distances
  (P1 P2 : ℝ × ℝ)
  (hl1 : ∃ t1, P1 = line t1)
  (hl2 : ∃ t2, P2 = line t2)
  (hc1 : parabola (P1.1) (P1.2))
  (hc2 : parabola (P2.1) (P2.2)) :
  dist point_A P1 + dist point_A P2 = 4 * (2 + sqrt 3) :=
sorry

end sum_of_distances_l185_185687


namespace divisible_12_or_36_l185_185796

theorem divisible_12_or_36 (x : ℕ) (n : ℕ) (h1 : Nat.Prime x) (h2 : 3 < x) (h3 : x = 3 * n + 1 ∨ x = 3 * n - 1) :
  12 ∣ (x^6 - x^3 - x^2 + x) ∨ 36 ∣ (x^6 - x^3 - x^2 + x) := 
by
  sorry

end divisible_12_or_36_l185_185796


namespace range_of_g_l185_185978

def range_exclude := Set.Univ \ Set.singleton (-21)

theorem range_of_g :
  (Set.Ioi (-5)).fun_image (λ x : ℝ, (3 * (x + 5) * (x - 2)) / (x + 5)) = range_exclude := 
sorry

end range_of_g_l185_185978


namespace g_is_odd_l185_185219

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185219


namespace g_is_odd_l185_185211

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185211


namespace g_is_odd_l185_185501

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185501


namespace solution_set_of_inequality_l185_185851

theorem solution_set_of_inequality (x : ℝ) : (5 - x^2 > 4x) ↔ (x > -5 ∧ x < 1) :=
sorry

end solution_set_of_inequality_l185_185851


namespace odd_function_g_l185_185163

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185163


namespace odd_function_option_B_l185_185383

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185383


namespace sum_of_subsets_l185_185752

theorem sum_of_subsets (a1 a2 a3 : ℝ) (h : (a1 + a2 + a3) + (a1 + a2 + a1 + a3 + a2 + a3) = 12) : 
  a1 + a2 + a3 = 4 := 
by 
  sorry

end sum_of_subsets_l185_185752


namespace g_is_odd_l185_185329

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185329


namespace option_B_odd_l185_185630

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185630


namespace letter_at_position_in_pattern_l185_185976

/-- Determine the 150th letter in the repeating pattern XYZ is "Z"  -/
theorem letter_at_position_in_pattern :
  ∀ (pattern : List Char) (position : ℕ), pattern = ['X', 'Y', 'Z'] → position = 150 → pattern.get! ((position - 1) % pattern.length) = 'Z' :=
by
  intros pattern position
  intro hPattern hPosition
  rw [hPattern, hPosition]
  -- pattern = ['X', 'Y', 'Z'] and position = 150
  sorry

end letter_at_position_in_pattern_l185_185976


namespace odd_function_g_l185_185157

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185157


namespace odd_function_g_l185_185531

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185531


namespace odd_function_shifted_f_l185_185355

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185355


namespace problem_is_odd_function_proof_l185_185448

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185448


namespace problem_statement_l185_185046

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185046


namespace length_of_CD_l185_185792

theorem length_of_CD (C D R S : ℝ) 
  (h1 : R = C + 3/8 * (D - C))
  (h2 : S = C + 4/11 * (D - C))
  (h3 : |S - R| = 3) :
  D - C = 264 := 
sorry

end length_of_CD_l185_185792


namespace sum_of_cubes_is_nine_l185_185900

def sum_of_cubes_of_consecutive_integers (n : ℤ) : ℤ :=
  n^3 + (n + 1)^3

theorem sum_of_cubes_is_nine :
  ∃ n : ℤ, sum_of_cubes_of_consecutive_integers n = 9 :=
by
  sorry

end sum_of_cubes_is_nine_l185_185900


namespace option_B_odd_l185_185650

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185650


namespace sequence_squared_l185_185691

theorem sequence_squared (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = a (n - 1) + 2 * (n - 1)) 
  : ∀ n, a n = n^2 := 
by
  sorry

end sequence_squared_l185_185691


namespace odd_function_result_l185_185240

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185240


namespace odd_function_g_l185_185528

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185528


namespace fraction_of_third_is_eighth_l185_185881

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l185_185881


namespace option_B_is_odd_function_l185_185314

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185314


namespace option_B_odd_l185_185646

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185646


namespace odd_function_proof_l185_185417

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185417


namespace odd_function_g_l185_185523

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185523


namespace g_B_is_odd_l185_185685

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185685


namespace closest_fraction_to_sqrt2_l185_185895

theorem closest_fraction_to_sqrt2 : 
  ∃ (p q : ℕ), p < 100 ∧ q < 100 ∧ p / q = 99 / 70 ∧ abs (sqrt 2 - p / q) = abs (sqrt 2 - 99 / 70) :=
sorry

end closest_fraction_to_sqrt2_l185_185895


namespace odd_function_result_l185_185244

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185244


namespace is_odd_g_l185_185465

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185465


namespace is_odd_g_l185_185471

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185471


namespace sara_has_green_marbles_l185_185805

-- Definition of the total number of green marbles and Tom's green marbles
def total_green_marbles : ℕ := 7
def tom_green_marbles : ℕ := 4

-- Definition of Sara's green marbles
def sara_green_marbles : ℕ := total_green_marbles - tom_green_marbles

-- The proof statement
theorem sara_has_green_marbles : sara_green_marbles = 3 :=
by
  -- The proof will be filled in here
  sorry

end sara_has_green_marbles_l185_185805


namespace horse_rent_problem_l185_185909

theorem horse_rent_problem (total_rent : ℝ) (b_payment : ℝ) (a_horses b_horses c_horses : ℝ) 
  (a_months b_months c_months : ℝ) (h_total_rent : total_rent = 870) (h_b_payment : b_payment = 360)
  (h_a_horses : a_horses = 12) (h_b_horses : b_horses = 16) (h_c_horses : c_horses = 18) 
  (h_b_months : b_months = 9) (h_c_months : c_months = 6) : 
  ∃ (a_months : ℝ), (a_horses * a_months * 2.5 + b_payment + c_horses * c_months * 2.5 = total_rent) :=
by
  use 8
  sorry

end horse_rent_problem_l185_185909


namespace g_is_odd_l185_185507

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185507


namespace optionB_is_odd_l185_185591

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185591


namespace compute_FG_length_l185_185958

-- Definition for the problem
def length_FG (A B C D F G : ℝ) (AC : ℝ) (E : ℝ) : ℝ :=
  let side_length := 1 in
  let diagonal := Real.sqrt 2 in
  2 * (Real.sqrt 2 - 1)

-- Statement of the mathematical proof problem
theorem compute_FG_length (A B C D F G E : ℝ) (h_square: (A - B) * (A - B) + (B - C) * (B - C) = 1)
    (h_fold_AC_E: E ∈ AC) (h_F_AB: F ∈ AB) (h_G_AD: G ∈ AD) : 
    length_FG A B C D F G AC E = 2 * (Real.sqrt 2 - 1) :=
begin
  sorry
end

end compute_FG_length_l185_185958


namespace odd_function_option_B_l185_185399

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185399


namespace g_B_is_odd_l185_185680

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185680


namespace option_B_odd_l185_185657

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185657


namespace odd_function_g_l185_185530

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185530


namespace g_is_odd_l185_185022

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185022


namespace base7_addition_l185_185994

theorem base7_addition (Y X : Nat) (k m : Int) :
    (Y + 2 = X + 7 * k) ∧ (X + 5 = 4 + 7 * m) ∧ (5 = 6 + 7 * -1) → X + Y = 10 :=
by
  sorry

end base7_addition_l185_185994


namespace triangle_side_ratios_l185_185718

theorem triangle_side_ratios (A B C a b c : ℝ) (hA : A = 45) (hB : B = 60) (hC : C = 75)
  (hSinA : a = real.sin (real.pi * A / 180))
  (hSinB : b = real.sin (real.pi * B / 180))
  (hSinC : c = real.sin (real.pi * C / 180)) :
  a : b : c = Real.sqrt 2 : Real.sqrt 3 : (Real.sqrt 6 + Real.sqrt 2) / 2 :=
by
  sorry

end triangle_side_ratios_l185_185718


namespace g_B_is_odd_l185_185675

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185675


namespace option_B_odd_l185_185642

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185642


namespace g_is_odd_l185_185623

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185623


namespace g_is_odd_l185_185607

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185607


namespace g_is_odd_l185_185229

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185229


namespace odd_function_g_l185_185545

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185545


namespace odd_function_g_l185_185197

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185197


namespace g_B_is_odd_l185_185664

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185664


namespace g_is_odd_l185_185019

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185019


namespace smallest_sum_of_three_l185_185979

/-- Define the set of numbers to choose from --/
def number_set : set ℤ := {-7, 0, 15, 5, -2}

/-- Prove the smallest sum possible by adding three different numbers from the set is -9. --/
theorem smallest_sum_of_three : 
  ∃ a b c ∈ number_set, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
sorry

end smallest_sum_of_three_l185_185979


namespace option_B_is_odd_function_l185_185312

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185312


namespace g_is_odd_l185_185610

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185610


namespace Rachel_books_total_l185_185798

-- Define the conditions
def mystery_shelves := 6
def picture_shelves := 2
def scifi_shelves := 3
def bio_shelves := 4
def books_per_shelf := 9

-- Define the total number of books
def total_books := 
  mystery_shelves * books_per_shelf + 
  picture_shelves * books_per_shelf + 
  scifi_shelves * books_per_shelf + 
  bio_shelves * books_per_shelf

-- Statement of the problem
theorem Rachel_books_total : total_books = 135 := 
by
  -- Proof can be added here
  sorry

end Rachel_books_total_l185_185798


namespace scale_model_feet_to_yards_l185_185729

theorem scale_model_feet_to_yards:
  (original_height : ℝ) (model_height : ℝ) (feet_per_yard : ℝ) 
  (h_original : original_height = 305) 
  (h_model : model_height = 10)
  (h_yard : feet_per_yard = 3) :
  (1 / model_height) * original_height = 30.5 ∧ (30.5 / feet_per_yard) ≈ 10.1667 :=
by
  have h1 : (305 / 10) = 30.5 := by norm_num
  have h2 : (30.5 / 3) = 10.1667 := by norm_num
  exact ⟨h1, h2⟩
  sorry

end scale_model_feet_to_yards_l185_185729


namespace time_after_3456_minutes_l185_185701

theorem time_after_3456_minutes :
  ∀ (start_date : String) (start_time : Nat × Nat) (minutes_added : Nat),
  start_date = "July 4, 2023" →
  start_time = (12, 0) →
  minutes_added = 3456 →
  let (days, hours, mins) := (57, 9, 36) in -- breakdown of 3456 minutes calculation
  let end_date := "July 6, 2023" in
  let end_time := (21, 36) in -- 9:00 PM + 36 minutes
  end_date = "July 6, 2023" ∧ end_time = (21, 36) :=
by
  sorry

end time_after_3456_minutes_l185_185701


namespace odd_function_result_l185_185265

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185265


namespace series_convergence_l185_185741

noncomputable def series_converges (x : ℝ) : Prop :=
  -- Define the series
  has_sum (λ n : ℕ, (x^(2 * n + 1)) / ((n^2 + 1) * 3^n)) 0

theorem series_convergence (x : ℝ) :
  series_converges x ↔ - real.sqrt 3 ≤ x ∧ x ≤ real.sqrt 3 :=
sorry

end series_convergence_l185_185741


namespace decreasing_at_half_implies_a_leq_2_l185_185828

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := -2 * x^2 + a * x + 1

-- Define what it means for the function to be decreasing at a point
def is_decreasing_at (f : ℝ → ℝ) (x : ℝ) : Prop := deriv f x ≤ 0

-- State the problem as a theorem
theorem decreasing_at_half_implies_a_leq_2 (a : ℝ) :
  is_decreasing_at (λ x, f x a) (1/2) → a ≤ 2 :=
by
  sorry

end decreasing_at_half_implies_a_leq_2_l185_185828


namespace is_odd_g_l185_185468

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185468


namespace odd_function_proof_l185_185422

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185422


namespace drink_total_amount_l185_185931

theorem drink_total_amount (total_amount: ℝ) (grape_juice: ℝ) (grape_proportion: ℝ) 
  (h1: grape_proportion = 0.20) (h2: grape_juice = 40) : total_amount = 200 :=
by
  -- Definitions and assumptions
  let calculation := grape_juice / grape_proportion
  -- Placeholder for the proof
  sorry

end drink_total_amount_l185_185931


namespace optionB_is_odd_l185_185588

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185588


namespace g_is_odd_l185_185094

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185094


namespace odd_function_g_l185_185543

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185543


namespace problem_is_odd_function_proof_l185_185456

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185456


namespace g_is_odd_l185_185602

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185602


namespace odd_function_proof_l185_185413

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185413


namespace odd_function_option_B_l185_185547

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185547


namespace odd_function_option_B_l185_185381

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185381


namespace odd_function_proof_l185_185426

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185426


namespace odd_function_l185_185139

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185139


namespace g_is_odd_l185_185505

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185505


namespace odd_function_g_l185_185188

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185188


namespace option_B_is_odd_function_l185_185305

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185305


namespace g_is_odd_l185_185032

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185032


namespace parallel_if_otimes_zero_parallelogram_area_l185_185974

variables {V : Type*} [inner_product_space ℝ V] (a b : V)
noncomputable def otimes (a b : V) : ℝ := ∥a∥ * ∥b∥ * Real.sin (inner_product_space.angle a b)

-- Proof that if a ⊗ b = 0, then a is parallel to b
theorem parallel_if_otimes_zero (h : otimes a b = 0) : a ∥ b :=
sorry

variables {P : Type*} [inner_product_space ℝ P] {A B C D : P}

-- Proof that area of parallelogram ABCD is equal to AB ⊗ AD
theorem parallelogram_area (hABCD : is_parallelogram A B C D) :
  parallelogram_area A B C D = otimes (A - B) (A - D) :=
sorry

end parallel_if_otimes_zero_parallelogram_area_l185_185974


namespace option_B_is_odd_function_l185_185308

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185308


namespace avg_winning_amount_for_other_tickets_is_ten_l185_185744

def total_tickets : ℕ := 200
def cost_per_ticket : ℕ := 2
def total_spent : ℕ := total_tickets * cost_per_ticket

def win_percentage : ℚ := 0.20
def num_winning_tickets : ℕ := total_tickets * win_percentage

def percent_winning_five : ℚ := 0.80
def num_five_dollar_winners : ℕ := num_winning_tickets * percent_winning_five
def amount_won_five_dollar_tickets : ℕ := num_five_dollar_winners * 5

def grand_prize_tickets : ℕ := 1
def grand_prize_amount : ℕ := 5000

def num_other_winners : ℕ := num_winning_tickets - num_five_dollar_winners - grand_prize_tickets

def profit : ℕ := 4830
def total_winnings : ℕ := total_spent + profit
def total_won_other_tickets : ℕ := total_winnings - (amount_won_five_dollar_tickets + grand_prize_amount)

def avg_other_winning_amount : ℚ := total_won_other_tickets / num_other_winners

theorem avg_winning_amount_for_other_tickets_is_ten : avg_other_winning_amount = 10 := sorry

end avg_winning_amount_for_other_tickets_is_ten_l185_185744


namespace odd_function_option_B_l185_185395

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185395


namespace find_B_l185_185693

variable {U : Set ℕ}

def A : Set ℕ := {1, 3, 5, 7}
def complement_A : Set ℕ := {2, 4, 6}
def complement_B : Set ℕ := {1, 4, 6}
def B : Set ℕ := {2, 3, 5, 7}

theorem find_B
  (hU : U = A ∪ complement_A)
  (A_comp : ∀ x, x ∈ complement_A ↔ x ∉ A)
  (B_comp : ∀ x, x ∈ complement_B ↔ x ∉ B) :
  B = {2, 3, 5, 7} :=
sorry

end find_B_l185_185693


namespace len_seg_AB_l185_185933

theorem len_seg_AB
  (a : ℝ) (h₀ : a = 1/2) -- distance from vertex to focus
  (M_dist : ℝ) (h₁ : M_dist = 5) -- distance from M to the axis
  (p : point) (A B : point) -- points of intersection
  (parabola_eq : ∀ x y, y^2 = 2*x ↔ on_parabola (x, y)) -- equation of parabola
  (focus_on_line : on_line (focus, A) ∧ on_line (focus, B)) -- line through focus and points A, B
  (midpoint_M: M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) -- midpoint M
  (axis_distance: ∀ y, axis_distance y = |y|) -- definition of axis distance
  (h2 : axis_distance M.y = 5) -- given distance from M to the axis
  :
  dist A B = 10 :=
by
  sorry

end len_seg_AB_l185_185933


namespace odd_function_option_B_l185_185571

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185571


namespace podium_height_l185_185876

theorem podium_height (l w h : ℝ) (r s : ℝ) (H1 : r = l + h - w) (H2 : s = w + h - l) 
  (Hr : r = 40) (Hs : s = 34) : h = 37 :=
by
  sorry

end podium_height_l185_185876


namespace odd_function_g_l185_185281

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185281


namespace odd_function_g_l185_185155

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185155


namespace minimize_time_l185_185943

variable (problems_per_week : Nat)
variable (days : Fin 7 → Fin 45 → Nat)

def weekly_problems_solved : Nat := 25

theorem minimize_time {T : Fin 7 → ℕ → ℕ} (h_constant_time : ∀ d, ∃ t < 45, ∀ p, T d p = t)
  (h_weekly_problems : ∀ d, (days d).sum = weekly_problems_solved) :
  ∃ k, (∀ j, k ≠ j → days j = 0) ∧ (days k = weekly_problems_solved) :=
by
  sorry

end minimize_time_l185_185943


namespace g_is_odd_l185_185503

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185503


namespace two_digit_numbers_with_reverse_sum_multiple_of_13_l185_185705

theorem two_digit_numbers_with_reverse_sum_multiple_of_13 : 
  {N : ℕ // 10 ≤ N ∧ N < 100 ∧ ∃ t u, N = 10 * t + u ∧ t + u = 13}.to_finset.card = 6 := 
by
  sorry

end two_digit_numbers_with_reverse_sum_multiple_of_13_l185_185705


namespace odd_function_result_l185_185261

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185261


namespace g_B_is_odd_l185_185669

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185669


namespace find_f_five_l185_185830

-- Define the function f and the conditions as given in the problem.
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (h₁ : ∀ x y : ℝ, f (x - y) = f x * g y)
variable (h₂ : ∀ y : ℝ, g y = Real.exp (-y))
variable (h₃ : ∀ x : ℝ, f x ≠ 0)

-- Goal: Prove that f(5) = e^{2.5}.
theorem find_f_five : f 5 = Real.exp 2.5 :=
by
  -- Proof is omitted as per the instructions.
  sorry

end find_f_five_l185_185830


namespace sixteenth_root_of_unity_l185_185982

noncomputable def tan_pi_over_8 : ℂ := real.tan (real.pi / 8)
noncomputable def z : ℂ := (tan_pi_over_8 + complex.i) / (tan_pi_over_8 - complex.i)
noncomputable def root_of_unity_12 : ℂ := complex.exp (complex.I * real.pi * 12 / 16)

theorem sixteenth_root_of_unity : z = root_of_unity_12 :=
sorry

end sixteenth_root_of_unity_l185_185982


namespace probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l185_185948

namespace Sprinters

def P_A : ℚ := 2 / 5
def P_B : ℚ := 3 / 4
def P_C : ℚ := 1 / 3

def P_all_qualified := P_A * P_B * P_C
def P_two_qualified := P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C
def P_at_least_one_qualified := 1 - (1 - P_A) * (1 - P_B) * (1 - P_C)

theorem probability_all_qualified : P_all_qualified = 1 / 10 :=
by 
  -- proof here
  sorry

theorem probability_two_qualified : P_two_qualified = 23 / 60 :=
by 
  -- proof here
  sorry

theorem probability_at_least_one_qualified : P_at_least_one_qualified = 9 / 10 :=
by 
  -- proof here
  sorry

end Sprinters

end probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l185_185948


namespace smallest_factor_of_36_sum_4_l185_185799

theorem smallest_factor_of_36_sum_4 : ∃ a b c : ℤ, (a * b * c = 36) ∧ (a + b + c = 4) ∧ (a = -4 ∨ b = -4 ∨ c = -4) :=
by
  sorry

end smallest_factor_of_36_sum_4_l185_185799


namespace optionB_is_odd_l185_185596

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185596


namespace odd_function_g_l185_185160

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185160


namespace sum_row_10_pascal_triangle_l185_185893

theorem sum_row_10_pascal_triangle :
  (∑ k in Finset.range (11), Nat.choose 10 k) = 1024 :=
by
  sorry

end sum_row_10_pascal_triangle_l185_185893


namespace odd_function_shifted_f_l185_185354

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185354


namespace scientific_notation_of_218000000_l185_185813

theorem scientific_notation_of_218000000 :
  218000000 = 2.18 * 10^8 :=
sorry

end scientific_notation_of_218000000_l185_185813


namespace g_is_odd_l185_185220

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185220


namespace option_B_odd_l185_185643

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185643


namespace odd_function_result_l185_185256

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185256


namespace g_is_odd_l185_185612

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185612


namespace odd_function_g_l185_185171

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185171


namespace odd_function_option_B_l185_185402

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185402


namespace is_odd_g_l185_185483

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185483


namespace odd_function_g_l185_185005

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185005


namespace optionB_is_odd_l185_185590

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185590


namespace option_b_is_odd_l185_185120

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185120


namespace quadrilateral_opposite_sides_equal_l185_185795

theorem quadrilateral_opposite_sides_equal
  (A B C D O : Type) [LinearOrderoid A] 
  (on_segment_AO : LineSegment A O)
  (on_segment_OC : LineSegment O C)
  (segment_bisect : seg_eq A O O C)
  (perimeter_eq : seg_eq (seg_sum (seg_length A B) (seg_length B C))
                         (seg_sum (seg_length A D) (seg_length D C))) :
  seg_eq (seg_length A B) (seg_length C D) ∧
  seg_eq (seg_length A D) (seg_length B C) := 
sorry

end quadrilateral_opposite_sides_equal_l185_185795


namespace is_odd_g_l185_185467

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185467


namespace solve_triangle_problem_l185_185696

noncomputable def triangle_problem : Prop :=
  ∃ (a b c : ℝ) (A B C : ℝ) (S : ℝ),
    A = 3 * Real.pi / 4 ∧
    b = Real.sqrt 2 * c ∧
    S = 2 ∧
    (a = Real.sqrt 5 * c) ∧
    (cos B = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
    (cos C = (a^2 + b^2 - c^2) / (2 * a * b)) ∧
    (cos B * cos C = 3 * Real.sqrt 2 / 5) ∧
    (a = 2 * Real.sqrt 5) ∧
    (b = 2 * Real.sqrt 2) ∧
    (c = 2)

theorem solve_triangle_problem : triangle_problem := 
begin 
    sorry 
end

end solve_triangle_problem_l185_185696


namespace g_is_odd_l185_185516

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185516


namespace odd_function_shifted_f_l185_185358

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185358


namespace ratio_of_spots_to_wrinkles_l185_185745

-- Definitions
def E : ℕ := 3
def W : ℕ := 3 * E
def S : ℕ := E + W - 69

-- Theorem
theorem ratio_of_spots_to_wrinkles : S / W = 7 :=
by
  sorry

end ratio_of_spots_to_wrinkles_l185_185745


namespace odd_function_shifted_f_l185_185351

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185351


namespace problem_is_odd_function_proof_l185_185455

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185455


namespace odd_function_proof_l185_185414

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185414


namespace g_is_odd_l185_185020

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185020


namespace odd_function_option_B_l185_185567

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185567


namespace odd_function_g_l185_185290

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185290


namespace minimum_tangent_length_l185_185944

theorem minimum_tangent_length :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle_center := (3, 0)
  let circle_radius := 1
  ∃ p ∈ line, ∀ q ∈ line, distance circle_center p ≤ distance circle_center q → 
    distance (p : ℝ × ℝ) (circle_center : ℝ × ℝ) = 2 * Real.sqrt 2 ∧
    tangent_length p circle_center circle_radius = Real.sqrt 7 := sorry

end minimum_tangent_length_l185_185944


namespace triangle_angle_contradiction_l185_185925

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180)
(h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l185_185925


namespace max_sum_l185_185856

open Function

theorem max_sum (f g h j : ℕ) 
  (hf : f ∈ {3, 4, 5, 6})
  (hg : g ∈ {3, 4, 5, 6})
  (hh : h ∈ {3, 4, 5, 6})
  (hj : j ∈ {3, 4, 5, 6})
  (distinct : (f ≠ g ∧ f ≠ h ∧ f ≠ j ∧ g ≠ h ∧ g ≠ j ∧ h ≠ j)) :
  f * g + g * h + h * j + j * f ≤ 80 :=
sorry

end max_sum_l185_185856


namespace problem_1_problem_2_l185_185923

-- Define the conditions and required questions for the first problem.
noncomputable def cos2_minus_3sin_cos (m : ℝ) (hm : m ≠ 0) : ℝ :=
  let α := real.atan 3 in
  (real.cos α)^2 - 3 * (real.sin α) * (real.cos α)

theorem problem_1 (m : ℝ) (hm : m ≠ 0) : cos2_minus_3sin_cos m hm = -4 / 5 :=
  sorry

-- Define the conditions and required questions for the second problem.
noncomputable def find_a (a : ℝ) (h1 : real.sin θ = (1 - a) / (1 + a)) (h2 : real.cos θ = (3 * a - 1) / (1 + a)) (h3 : π / 2 < θ ∧ θ < π) : ℝ :=
  a

theorem problem_2 (θ : ℝ) (a : ℝ)
  (h1 : real.sin θ = (1 - a) / (1 + a))
  (h2 : real.cos θ = (3 * a - 1) / (1 + a))
  (h3 : π / 2 < θ ∧ θ < π) : find_a a h1 h2 h3 = 1 / 9 :=
  sorry

end problem_1_problem_2_l185_185923


namespace fill_tank_total_time_l185_185945

theorem fill_tank_total_time (h : 1 / 16) :
  let t1 := 8
  let t2 := 2 in
  t1 + t2 = 10 :=
by
  -- Definitions from the conditions
  let t_full := 16
  let rate := 1 / t_full
  let half_time := t_full / 2
  have h1 : half_time = 8 := by linarith
  let four_taps_time := t_full / 4
  let half_four_taps_time := four_taps_time / 2
  have h2 : half_four_taps_time = 2 := by linarith

  -- Final step: adding the times
  exact add_eq_of_eq_sub' h1 h2

end fill_tank_total_time_l185_185945


namespace problem_statement_l185_185058

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185058


namespace beads_per_bracelet_l185_185963

def beads_bella_has : Nat := 36
def beads_bella_needs : Nat := 12
def total_bracelets : Nat := 6

theorem beads_per_bracelet : (beads_bella_has + beads_bella_needs) / total_bracelets = 8 :=
by
  sorry

end beads_per_bracelet_l185_185963


namespace odd_function_option_B_l185_185546

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185546


namespace factor_x4_minus_81_l185_185990

variable (x : ℝ)

theorem factor_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
  by { -- proof steps would go here 
    sorry 
}

end factor_x4_minus_81_l185_185990


namespace pascal_row_10_sum_l185_185891

-- Definition: sum of the numbers in Row n of Pascal's Triangle is 2^n
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- Theorem: sum of the numbers in Row 10 of Pascal's Triangle is 1024
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  sorry

end pascal_row_10_sum_l185_185891


namespace odd_function_g_l185_185196

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185196


namespace odd_function_shifted_f_l185_185366

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185366


namespace optionB_is_odd_l185_185593

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185593


namespace g_is_odd_l185_185504

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185504


namespace odd_function_g_l185_185191

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185191


namespace optionB_is_odd_l185_185587

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185587


namespace is_odd_g_l185_185474

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185474


namespace solve_for_x_l185_185809

theorem solve_for_x (x : ℝ) : 16^(2*x - 4) = 4^(3 - x) → x = 11/5 := by
  sorry

end solve_for_x_l185_185809


namespace odd_function_proof_l185_185421

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185421


namespace odd_function_option_B_l185_185572

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185572


namespace odd_function_g_l185_185537

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185537


namespace odd_function_g_l185_185535

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185535


namespace option_B_odd_l185_185633

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185633


namespace g_is_odd_l185_185036

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185036


namespace odd_function_l185_185143

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185143


namespace is_odd_g_l185_185472

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185472


namespace g_is_odd_l185_185085

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185085


namespace g_is_odd_l185_185603

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185603


namespace g_is_odd_l185_185627

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185627


namespace problem_is_odd_function_proof_l185_185457

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185457


namespace odd_function_option_B_l185_185392

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185392


namespace sum_of_consecutive_integers_l185_185859

theorem sum_of_consecutive_integers (x y z : ℤ) (h1 : y = x + 1) (h2 : z = y + 1) (h3 : z = 12) :
  x + y + z = 33 :=
sorry

end sum_of_consecutive_integers_l185_185859


namespace option_B_odd_l185_185631

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185631


namespace sqrt_identity_1_sqrt_identity_2_l185_185794

variable {a b : ℝ}

-- Conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_sq_gt_b : a^2 > b

theorem sqrt_identity_1 : 
  ∀ (a b : ℝ), (a > 0) → (b > 0) → (a^2 > b) → 
    sqrt (a + sqrt b) = sqrt ((a + sqrt (a^2 - b)) / 2) + sqrt ((a - sqrt (a^2 - b)) / 2) :=
by 
  intro a b
  intro a_pos b_pos a_sq_gt_b
  sorry

theorem sqrt_identity_2 :
  ∀ (a b : ℝ), (a > 0) → (b > 0) → (a^2 > b) → 
    sqrt (a - sqrt b) = sqrt ((a + sqrt (a^2 - b)) / 2) - sqrt ((a - sqrt (a^2 - b)) / 2) :=
by 
  intro a b
  intro a_pos b_pos a_sq_gt_b
  sorry

end sqrt_identity_1_sqrt_identity_2_l185_185794


namespace option_B_is_odd_function_l185_185294

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185294


namespace g_is_odd_l185_185093

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185093


namespace g_is_odd_l185_185075

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185075


namespace problem_is_odd_function_proof_l185_185439

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185439


namespace odd_function_g_l185_185170

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185170


namespace odd_function_g_l185_185200

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185200


namespace split_costs_evenly_l185_185868

theorem split_costs_evenly (t d : ℕ) 
  (Tom_paid : 150) (Dorothy_paid : 180) (Sammy_paid : 220) (Nick_paid : 250) 
  (total_paid : 150 + 180 + 220 + 250 = 800)
  (even_split : 800 / 4 = 200)
  (Tom_owes : Tom_paid < even_split)
  (Dorothy_owes : Dorothy_paid < even_split)
  (t_def : t = even_split - Tom_paid)
  (d_def : d = even_split - Dorothy_paid) :
  t - d = 30 := 
by
  sorry

end split_costs_evenly_l185_185868


namespace arithmetic_sequence_sum_l185_185853

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 3 = 6)
  (h2 : S 9 = 27) :
  S 6 = 15 :=
sorry

end arithmetic_sequence_sum_l185_185853


namespace g_is_odd_l185_185622

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185622


namespace find_XY_base10_l185_185992

theorem find_XY_base10 (X Y : ℕ) (h₁ : Y + 2 = X) (h₂ : X + 5 = 11) : X + Y = 10 := 
by 
  sorry

end find_XY_base10_l185_185992


namespace base_256_6_digits_l185_185922

theorem base_256_6_digits (b : ℕ) (h1 : b ^ 5 ≤ 256) (h2 : 256 < b ^ 6) : b = 3 := 
sorry

end base_256_6_digits_l185_185922


namespace is_odd_g_l185_185482

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185482


namespace g_is_odd_l185_185621

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185621


namespace margo_paired_with_irma_probability_l185_185873

theorem margo_paired_with_irma_probability :
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  probability = (1 / 15) :=
by
  let n := 15
  let total_outcomes := n
  let favorable_outcomes := 1
  let probability := favorable_outcomes / total_outcomes
  have h : probability = 1 / 15 := by
    -- skipping the proof details as per instructions
    sorry
  exact h

end margo_paired_with_irma_probability_l185_185873


namespace option_B_is_odd_function_l185_185315

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185315


namespace odd_function_proof_l185_185427

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185427


namespace option_B_is_odd_function_l185_185303

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185303


namespace option_b_is_odd_l185_185106

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185106


namespace odd_function_g_l185_185270

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185270


namespace problem_statement_l185_185052

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185052


namespace problem_solved_by_3_girls_3_boys_l185_185858

-- Assumptions
variables (Girls Boys Problems : Type) 
variables [Fintype Girls] [Fintype Boys] [Fintype Problems]
variable [DecidableEq Problems]

-- Number of participants
variable (num_girls : ℕ) (num_boys : ℕ)
variable (h_num_girls : Fintype.card Girls = 21) (h_num_boys : Fintype.card Boys = 20)

-- Each participant solves at most 6 problems
variable (solves : Π (g : Girls) (b : Boys), Finset Problems)
variable (max_solved_girls : ∀ g : Girls, (solves g).card ≤ 6)
variable (max_solved_boys : ∀ b : Boys, (solves b).card ≤ 6)

-- Each pair of one girl and one boy has at least one common solved problem
variable (common_solved : ∀ g : Girls, ∀ b : Boys, (solves g ∩ solves b).nonempty)

-- Theorem to prove
theorem problem_solved_by_3_girls_3_boys : 
  ∃ p : Problems, 3 ≤ (solves g).count p ∧ 3 ≤ (solves b).count p :=
sorry

end problem_solved_by_3_girls_3_boys_l185_185858


namespace is_odd_g_l185_185484

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185484


namespace is_odd_g_l185_185487

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185487


namespace g_is_odd_l185_185609

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185609


namespace odd_function_l185_185134

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185134


namespace optionB_is_odd_l185_185599

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185599


namespace odd_function_g_l185_185206

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185206


namespace period_and_theta_of_sine_l185_185713

noncomputable def function_properties (f : ℝ → ℝ) (T θ : ℝ) : Prop :=
  (∀ x, f(x + T) = f(x)) ∧ 
  (∀ x, f(x) ≤ 1) ∧ 
  (f 2 = 1) ∧ 
  (0 < θ < 2 * π)

theorem period_and_theta_of_sine : 
  ∃ T θ, 
  function_properties (λ x, Real.sin (π * x + θ)) T θ ∧ T = 2 ∧ θ = π / 2 :=
sorry

end period_and_theta_of_sine_l185_185713


namespace find_g_four_l185_185829

theorem find_g_four (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 11 / 2 := 
by
  sorry

end find_g_four_l185_185829


namespace problem_statement_l185_185050

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185050


namespace next_palindromic_year_product_l185_185857

theorem next_palindromic_year_product (starting_year : ℕ) (h : starting_year = 2200) :
  ∃ next_year, next_year > starting_year ∧
               (palindrome next_year) ∧ (digit_product next_year = 16) := sorry

-- Definitions required for Lean 4 statement
def palindrome (n : ℕ) : Prop :=
  let s := n.repr in s = s.reverse

def digit_product (n : ℕ) : ℕ :=
  let digits := n.digits 10 in digits.foldl (*) 1

end next_palindromic_year_product_l185_185857


namespace problem_statement_l185_185045

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185045


namespace odd_function_result_l185_185242

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185242


namespace odd_function_proof_l185_185416

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185416


namespace odd_function_g_l185_185527

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185527


namespace g_is_odd_l185_185345

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185345


namespace g_is_odd_l185_185494

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185494


namespace option_B_odd_l185_185640

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185640


namespace g_is_odd_l185_185493

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185493


namespace g_is_odd_l185_185347

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185347


namespace g_B_is_odd_l185_185660

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185660


namespace is_odd_g_l185_185469

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185469


namespace negation_equiv_l185_185689

theorem negation_equiv (a b : ℝ) :
  ¬ (a > b → 2^a > 2^b) ↔ (2^a ≤ 2^b → a ≤ b) :=
by
  sorry

end negation_equiv_l185_185689


namespace motorcyclist_cross_time_l185_185935

/-- Definitions and conditions -/
def speed_X := 2 -- Rounds per hour
def speed_Y := 4 -- Rounds per hour

/-- Proof statement -/
theorem motorcyclist_cross_time : (1 / (speed_X + speed_Y) * 60 = 10) :=
by
  sorry

end motorcyclist_cross_time_l185_185935


namespace g_is_odd_l185_185513

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185513


namespace problem_statement_l185_185060

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185060


namespace cheryl_material_usage_l185_185835

theorem cheryl_material_usage:
  let bought := (3 / 8) + (1 / 3)
  let left := (15 / 40)
  let used := bought - left
  used = (1 / 3) := 
by
  sorry

end cheryl_material_usage_l185_185835


namespace toothpicks_20th_stage_l185_185862

-- Define the initial conditions
def first_stage_toothpicks : Nat := 5
def added_toothpicks : Nat := 3

-- Define the function to calculate the number of toothpicks at the n-th stage
def toothpicks (n : Nat) : Nat :=
  first_stage_toothpicks + added_toothpicks * (n - 1)

-- Prove that the number of toothpicks at the 20th stage is 62
theorem toothpicks_20th_stage : toothpicks 20 = 62 := by
  unfold toothpicks
  rw [first_stage_toothpicks, added_toothpicks]
  norm_num
  sorry

end toothpicks_20th_stage_l185_185862


namespace option_B_odd_l185_185656

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185656


namespace g_is_odd_l185_185081

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185081


namespace odd_function_g_l185_185534

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185534


namespace g_B_is_odd_l185_185682

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185682


namespace radius_of_circular_garden_l185_185930

theorem radius_of_circular_garden
  (r : ℝ) 
  (h₁ : 2 * real.pi * r = (1/8) * real.pi * r^2) : r = 16 :=
by 
  sorry

end radius_of_circular_garden_l185_185930


namespace coordinates_of_third_vertex_l185_185875

theorem coordinates_of_third_vertex
  (A B : ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (6, 4))
  (hB : B = (0, 0))
  (hC_neg_x_axis : ∃ x : ℝ, x < 0 ∧ C = (x, 0))
  (h_area : 30 = 1 / 2 * real.abs (fst C - fst B) * (snd A - snd B)) :
  C = (-15, 0) :=
sorry

end coordinates_of_third_vertex_l185_185875


namespace g_is_odd_l185_185348

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185348


namespace g_is_odd_l185_185078

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185078


namespace g_is_odd_l185_185017

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185017


namespace odd_function_option_B_l185_185550

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185550


namespace odd_function_option_B_l185_185568

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185568


namespace g_is_odd_l185_185327

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185327


namespace odd_function_l185_185129

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185129


namespace g_is_odd_l185_185226

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185226


namespace problem_is_odd_function_proof_l185_185438

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185438


namespace g_is_odd_l185_185082

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185082


namespace problem_is_odd_function_proof_l185_185445

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185445


namespace odd_function_g_l185_185202

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185202


namespace g_is_odd_l185_185230

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185230


namespace odd_function_option_B_l185_185390

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185390


namespace g_is_odd_l185_185491

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185491


namespace people_left_first_hour_l185_185937

theorem people_left_first_hour 
  (X : ℕ)
  (h1 : X ≥ 0)
  (h2 : 94 - X + 18 - 9 = 76) :
  X = 27 := 
sorry

end people_left_first_hour_l185_185937


namespace odd_function_g_l185_185194

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185194


namespace odd_function_shifted_f_l185_185375

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185375


namespace g_is_odd_l185_185611

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185611


namespace equate_operations_l185_185986

theorem equate_operations :
  (15 * 5) / (10 + 2) = 3 → 8 / 4 = 2 → ((18 * 6) / (14 + 4) = 6) :=
by
sorry

end equate_operations_l185_185986


namespace probability_red_or_blue_marbles_l185_185887

theorem probability_red_or_blue_marbles (red blue green total : ℕ) (h_red : red = 4) (h_blue : blue = 3) (h_green : green = 6) (h_total : total = red + blue + green) :
  (red + blue) / total = 7 / 13 :=
by
  sorry

end probability_red_or_blue_marbles_l185_185887


namespace thirteen_pow_2045_mod_19_l185_185889

theorem thirteen_pow_2045_mod_19 : (13 ^ 2045) % 19 = 9 := 
  by
    have h1: 13 % 19 = 13 := by norm_num
    have h2: (13 ^ 2) % 19 = 12 := by norm_num
    have h3: (13 ^ 4) % 19 = 17 := by norm_num
    have h4: (13 ^ 8) % 19 = 11 := by norm_num
    have h5: (13 ^ 16) % 19 = 4 := by norm_num
    sorry

end thirteen_pow_2045_mod_19_l185_185889


namespace g_is_odd_l185_185328

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185328


namespace average_speed_entire_journey_l185_185950

theorem average_speed_entire_journey :
  let kmph_to_mps (v : ℕ) := v * 1000 / 3600
  ∧ let hours_to_seconds (h : ℕ) := h * 3600
  ∧ let minutes_to_seconds (m : ℕ) := m * 60
  ∧ let acceleration_distance := (0 + kmph_to_mps 252) / 2 * hours_to_seconds 1.5
  ∧ let deceleration_distance := (kmph_to_mps 252 + 0) / 2 * minutes_to_seconds 45
  ∧ let total_distance := acceleration_distance + deceleration_distance
  ∧ let total_time := hours_to_seconds 1.5 + minutes_to_seconds 45
  in total_distance / total_time = 35 :=
by
  sorry

end average_speed_entire_journey_l185_185950


namespace option_b_is_odd_l185_185105

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185105


namespace radius_larger_ball_l185_185842

-- Define the volume formula for a sphere.
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Parameters for the problem
def radius_small_ball : ℝ := 2
def total_volume_small_balls : ℝ := 6 * volume_of_sphere radius_small_ball

-- Prove that the radius of the larger ball is 4 * 2^(1 / 3) (which is 4 * cube root of 2).
theorem radius_larger_ball : ∃ r : ℝ, volume_of_sphere r = total_volume_small_balls ∧ r = 4 * Real.cbrt 2 := by
  sorry

end radius_larger_ball_l185_185842


namespace sequence_general_formula_l185_185736

theorem sequence_general_formula (n : ℕ) (h : n > 0) :
  let a : ℕ → ℝ := λ n, if n = 1 then 1 else (a (n - 1)) / (3 * (a (n - 1)) + 1)
  in a n = 1 / (3 * n - 2) :=
by
  sorry

end sequence_general_formula_l185_185736


namespace num_terms_arithmetic_sequence_is_41_l185_185703

-- Definitions and conditions
def first_term : ℤ := 200
def common_difference : ℤ := -5
def last_term : ℤ := 0

-- Definition of the n-th term of arithmetic sequence
def nth_term (a : ℤ) (d : ℤ) (n : ℤ) : ℤ :=
  a + (n - 1) * d

-- Statement to prove
theorem num_terms_arithmetic_sequence_is_41 : 
  ∃ n : ℕ, nth_term first_term common_difference n = 0 ∧ n = 41 :=
by 
  sorry

end num_terms_arithmetic_sequence_is_41_l185_185703


namespace g_is_odd_l185_185337

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185337


namespace odd_function_g_l185_185011

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185011


namespace odd_function_g_l185_185522

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185522


namespace a_income_increase_l185_185908

/-- Given the conditions: 
1. a, b, and c are partners.
2. a receives 2/3 of the profits, b and c divide the remainder equally.
3. The rate of profit rises from 5% to 7%.
4. The capital of a is Rs. 10000.
We need to prove that a's income is increased by approximately Rs. 133.34.
--/
theorem a_income_increase : 
  let initial_rate := 5 / 100
  let final_rate := 7 / 100
  let capital_a := 10000
  let profit_initial := initial_rate * capital_a
  let profit_final := final_rate * capital_a
  let a_share_initial := (2 / 3) * profit_initial
  let a_share_final := (2 / 3) * profit_final
  a_share_final - a_share_initial ≈ 133.34 :=
by 
  sorry

end a_income_increase_l185_185908


namespace problem_statement_l185_185067

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185067


namespace find_a_l185_185995

theorem find_a (r s a : ℚ) (h1 : s^2 = 16) (h2 : 2 * r * s = 15) (h3 : a = r^2) : a = 225/64 := by
  sorry

end find_a_l185_185995


namespace odd_function_result_l185_185249

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185249


namespace projectile_reaches_49_first_time_at_1_point_4_l185_185825

-- Define the equation for the height of the projectile
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t

-- State the theorem to prove
theorem projectile_reaches_49_first_time_at_1_point_4 :
  ∃ t : ℝ, height t = 49 ∧ (∀ t' : ℝ, height t' = 49 → t ≤ t') :=
sorry

end projectile_reaches_49_first_time_at_1_point_4_l185_185825


namespace option_B_odd_l185_185645

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185645


namespace minimum_value_absolute_difference_perpendicular_tangents_l185_185714

noncomputable def curve (x : ℝ) : ℝ :=
  (1/4) * sin (2 * x) + (sqrt 3 / 2) * (cos x) ^ 2

theorem minimum_value_absolute_difference_perpendicular_tangents (x₁ x₂ : ℝ) :
  (deriv curve x₁) = 1 →
  (deriv curve x₂) = -1 →
  min_value (abs (x₁ - x₂)) = π / 2 :=
sorry

end minimum_value_absolute_difference_perpendicular_tangents_l185_185714


namespace g_is_odd_l185_185237

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185237


namespace odd_function_l185_185146

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185146


namespace odd_function_result_l185_185264

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185264


namespace odd_function_option_B_l185_185393

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185393


namespace option_b_is_odd_l185_185124

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185124


namespace odd_function_g_l185_185189

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185189


namespace g_is_odd_l185_185349

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185349


namespace complex_modulus_l185_185773

theorem complex_modulus (z : ℂ) (h : z = i / (1 - i)) : complex.abs z = real.sqrt 2 / 2 :=
sorry

end complex_modulus_l185_185773


namespace g_is_odd_l185_185231

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185231


namespace g_is_odd_l185_185323

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185323


namespace find_gamma_l185_185815

variable (γ δ : ℝ)

def directly_proportional (γ δ : ℝ) : Prop := ∃ c : ℝ, γ = c * δ

theorem find_gamma (h1 : directly_proportional γ δ) (h2 : γ = 5) (h3 : δ = -10) : δ = 25 → γ = -25 / 2 := by
  sorry

end find_gamma_l185_185815


namespace odd_function_g_l185_185006

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185006


namespace option_B_is_odd_function_l185_185302

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185302


namespace odd_function_option_B_l185_185389

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185389


namespace all_boxcars_combined_capacity_l185_185804

theorem all_boxcars_combined_capacity :
  let black_capacity := 4000
  let blue_capacity := 2 * black_capacity
  let red_capacity := 3 * blue_capacity
  let green_capacity := 1.5 * black_capacity
  let yellow_capacity := green_capacity + 2000
  let total_red := 3 * red_capacity
  let total_blue := 4 * blue_capacity
  let total_black := 7 * black_capacity
  let total_green := 2 * green_capacity
  let total_yellow := 5 * yellow_capacity
  total_red + total_blue + total_black + total_green + total_yellow = 184000 :=
by 
  -- Proof omitted
  sorry

end all_boxcars_combined_capacity_l185_185804


namespace total_handshakes_l185_185960

-- Define the groups and their properties
def GroupA := 30
def GroupB := 15
def GroupC := 5
def KnowEachOtherA := true -- All 30 people in Group A know each other
def KnowFromB := 10 -- Each person in Group B knows 10 people from Group A
def KnowNoOneC := true -- Each person in Group C knows no one

-- Define the number of handshakes based on the conditions
def handshakes_between_A_and_B : Nat := GroupB * (GroupA - KnowFromB)
def handshakes_between_B_and_C : Nat := GroupB * GroupC
def handshakes_within_C : Nat := (GroupC * (GroupC - 1)) / 2
def handshakes_between_A_and_C : Nat := GroupA * GroupC

-- Prove the total number of handshakes
theorem total_handshakes : 
  handshakes_between_A_and_B +
  handshakes_between_B_and_C +
  handshakes_within_C +
  handshakes_between_A_and_C = 535 :=
by sorry

end total_handshakes_l185_185960


namespace odd_function_g_l185_185174

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185174


namespace jimmy_more_sheets_than_tommy_l185_185869

-- Definitions for the conditions
def initial_jimmy_sheets : ℕ := 58
def initial_tommy_sheets : ℕ := initial_jimmy_sheets + 25
def ashton_gives_jimmy : ℕ := 85
def jessica_gives_jimmy : ℕ := 47
def cousin_gives_tommy : ℕ := 30
def aunt_gives_tommy : ℕ := 19

-- Lean 4 statement for the proof problem
theorem jimmy_more_sheets_than_tommy :
  let final_jimmy_sheets := initial_jimmy_sheets + ashton_gives_jimmy + jessica_gives_jimmy;
  let final_tommy_sheets := initial_tommy_sheets + cousin_gives_tommy + aunt_gives_tommy;
  final_jimmy_sheets - final_tommy_sheets = 58 :=
by sorry

end jimmy_more_sheets_than_tommy_l185_185869


namespace prism_surface_area_l185_185920

theorem prism_surface_area (a b : ℝ) :
  let S := 2 * a * (a + Real.sqrt (4 * b^2 + a^2)) in
  S = 2 * a * (a + Real.sqrt (4 * b^2 + a^2)) :=
by
  sorry

end prism_surface_area_l185_185920


namespace odd_function_g_l185_185521

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185521


namespace odd_function_g_l185_185001

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185001


namespace odd_function_g_l185_185267

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185267


namespace g_is_odd_l185_185510

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185510


namespace odd_function_g_l185_185179

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185179


namespace problem_statement_l185_185069

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185069


namespace pyramid_sphere_volume_l185_185715

theorem pyramid_sphere_volume :
  (∀ (a b c : ℝ), (a = 1) ∧ (b = sqrt 3) ∧ (c = 2) ∧
    (∀ u v w : ℝ, u = sqrt (a^2 + b^2 + c^2) → v = 2) →
    (4 / 3 * Real.pi * ((u / 2)^3) = 8 * sqrt 2 / 3 * Real.pi)) := 
begin
  -- Given conditions
  assume (a b c : ℝ),
  assume (h₁ : a = 1) (h₂ : b = sqrt 3) (h₃ : c = 2),

  -- Pyramid conditions with mutually perpendicular edges
  assume (u v w : ℝ),
  assume (h₄ : u = sqrt (a^2 + b^2 + c^2)),
  assume (h₅ : v = 2),

  -- Show the volume of the sphere
  show 4 / 3 * Real.pi * ((u / 2)^3) = 8 * sqrt 2 / 3 * Real.pi,
  sorry
end

end pyramid_sphere_volume_l185_185715


namespace volume_of_inscribed_sphere_l185_185983

theorem volume_of_inscribed_sphere (h : ℝ) (hd: 60°) : 
  ∃ V : ℝ, V = (4 / 81) * π * h^3 :=
by
  let r := h / 3
  let V := (4 / 3) * π * r^3
  have h_eq : r = h / 3 := sorry
  have V_eq : V = (4 / 81) * π * h^3 := sorry
  existsi V
  exact V_eq

end volume_of_inscribed_sphere_l185_185983


namespace odd_function_result_l185_185259

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185259


namespace g_is_odd_l185_185617

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185617


namespace arithmetic_sequence_150th_term_l185_185884

theorem arithmetic_sequence_150th_term :
  let a₁ := 3
  let d := 4
  a₁ + (150 - 1) * d = 599 :=
by
  sorry

end arithmetic_sequence_150th_term_l185_185884


namespace g_is_odd_l185_185615

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185615


namespace option_b_is_odd_l185_185102

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185102


namespace g_is_odd_l185_185500

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185500


namespace g_is_odd_l185_185330

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185330


namespace is_odd_g_l185_185476

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185476


namespace odd_function_option_B_l185_185569

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185569


namespace kelly_single_shot_decrease_l185_185914

def kelly_salary_decrease (s : ℝ) : ℝ :=
  let first_cut := s * 0.92
  let second_cut := first_cut * 0.86
  let third_cut := second_cut * 0.82
  third_cut

theorem kelly_single_shot_decrease :
  let original_salary := 1.0 -- Assume original salary is 1 for percentage calculation
  let final_salary := kelly_salary_decrease original_salary
  (100 : ℝ) - (final_salary * 100) = 34.8056 :=
by
  sorry

end kelly_single_shot_decrease_l185_185914


namespace floor_function_solution_l185_185810

def floor_eq_x_solutions : Prop :=
  ∀ x : ℤ, (⌊(x : ℝ) / 2⌋ + ⌊(x : ℝ) / 4⌋ = x) ↔ x = 0 ∨ x = -3 ∨ x = -2 ∨ x = -5

theorem floor_function_solution: floor_eq_x_solutions :=
by
  intro x
  sorry

end floor_function_solution_l185_185810


namespace exists_multiple_no_zero_digits_l185_185806

theorem exists_multiple_no_zero_digits (n : ℕ) (h : n % 10 ≠ 0) : 
  ∃ m : ℕ, m % n = 0 ∧ ∀ d : ℕ, d ∈ (Int.digits 10 m).toList → d ≠ 0 :=
by
  sorry

end exists_multiple_no_zero_digits_l185_185806


namespace odd_function_option_B_l185_185394

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185394


namespace circles_separate_and_tangent_line_l185_185698

theorem circles_separate_and_tangent_line:
  let O := (0, 0)
  let C := (6, 0)
  let r1 := 2
  let r2 := 3
  ∀ (x y : ℝ),
  (x^2 + y^2 = 4) ∧ ((x - 6)^2 + y^2 = 9) →
  dist O C > r1 + r2 ∧
  (∃ k : ℝ, k = 2 * real.sqrt 2 ∨ k = -2 * real.sqrt 2 ∧ (∀ x, y = k * x + 6)) :=
by
  sorry

end circles_separate_and_tangent_line_l185_185698


namespace no_14_consecutive_divisible_by_2_to_11_l185_185921

theorem no_14_consecutive_divisible_by_2_to_11 :
  ¬ ∃ (a : ℕ), ∀ i, i < 14 → ∃ p, Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 11 ∧ (a + i) % p = 0 :=
by sorry

end no_14_consecutive_divisible_by_2_to_11_l185_185921


namespace odd_function_g_l185_185009

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185009


namespace odd_function_option_B_l185_185559

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185559


namespace price_each_clock_is_correct_l185_185812

-- Definitions based on the conditions
def numberOfDolls := 3
def numberOfClocks := 2
def numberOfGlasses := 5
def pricePerDoll := 5
def pricePerGlass := 4
def totalCost := 40
def profit := 25

-- The total revenue from selling dolls and glasses
def revenueFromDolls := numberOfDolls * pricePerDoll
def revenueFromGlasses := numberOfGlasses * pricePerGlass
def totalRevenueNeeded := totalCost + profit
def revenueFromDollsAndGlasses := revenueFromDolls + revenueFromGlasses

-- The required revenue from clocks
def revenueFromClocks := totalRevenueNeeded - revenueFromDollsAndGlasses

-- The price per clock
def pricePerClock := revenueFromClocks / numberOfClocks

-- Statement to prove
theorem price_each_clock_is_correct : pricePerClock = 15 := sorry

end price_each_clock_is_correct_l185_185812


namespace option_b_is_odd_l185_185110

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185110


namespace solve_fraction_l185_185716

theorem solve_fraction (x : ℝ) (h : 2 / (x - 3) = 2) : x = 4 :=
by
  sorry

end solve_fraction_l185_185716


namespace g_B_is_odd_l185_185670

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185670


namespace optionB_is_odd_l185_185578

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185578


namespace odd_function_option_B_l185_185387

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185387


namespace min_value_is_26_l185_185765

-- Define the set of distinct elements
def elements : Set ℤ := {-9, -6, -4, -1, 3, 5, 7, 12}

-- Define the function that computes the minimum value of the given expression
def compute_min_value (p q r s t u v w : ℤ) : ℤ :=
  (p + q + r + s)^2 + (t + u + v + w)^2

-- Define the conditions that p, q, r, s, t, u, v, w are distinct elements of the set
def are_distinct_elements (p q r s t u v w : ℤ) : Prop :=
  {p, q, r, s, t, u, v, w}.card = 8 ∧ p ∈ elements ∧ q ∈ elements ∧ r ∈ elements ∧ s ∈ elements ∧ t ∈ elements ∧ u ∈ elements ∧ v ∈ elements ∧ w ∈ elements

-- The theorem to prove that the minimum value is 26.5
theorem min_value_is_26.5 : ∃ (p q r s t u v w : ℤ), 
  are_distinct_elements p q r s t u v w ∧ compute_min_value p q r s t u v w = 26.5 :=
sorry

end min_value_is_26_l185_185765


namespace odd_function_result_l185_185241

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185241


namespace option_b_is_odd_l185_185107

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185107


namespace odd_function_shifted_f_l185_185367

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185367


namespace odd_function_g_l185_185187

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185187


namespace option_B_odd_l185_185634

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185634


namespace problem_is_odd_function_proof_l185_185451

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185451


namespace g_is_odd_l185_185227

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185227


namespace g_is_odd_l185_185233

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185233


namespace range_of_a_l185_185772

def f (x a : ℝ) : ℝ := x^2 - 2*a*x

def P (a : ℝ) : Prop := ∀ x > 1, f x a ≥ f (x + 1) a

def Q (a : ℝ) : Prop := ∀ x : ℝ, ax^2 - x + a > 0

theorem range_of_a (a : ℝ) : (P a ∧ ¬Q a) ∨ (Q a ∧ ¬P a) → a ≤ 1/2 ∨ a > 1 := sorry

end range_of_a_l185_185772


namespace part1_l185_185753

def f (x : ℝ) := sin (2 * x - π / 6) - 2 * (sin x) ^ 2 + 1

theorem part1 (a : ℝ) (h1 : f a = 1 / 2) (h2 : 0 ≤ a ∧ a ≤ π / 2) : a = 0 ∨ a = π / 3 :=
sorry

end part1_l185_185753


namespace c_share_is_160_l185_185916

theorem c_share_is_160 (a b c : ℕ) (total : ℕ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 10 * c) (h_total : a + b + c = 880) : c = 160 :=
by
  sorry

end c_share_is_160_l185_185916


namespace odd_function_option_B_l185_185561

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185561


namespace g_is_odd_l185_185027

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185027


namespace find_max_m_l185_185764

noncomputable theory

-- Define the recursive function sequence using Euler's totient function
def a (m : ℕ) : ℕ → ℕ
| 0       := m
| (n + 1) := Nat.totient (a n)

-- Define the main theorem where m <= 2016 and check for the desired sequence property
theorem find_max_m (m : ℕ) (h1 : 1 < m) (h2 : m ≤ 2016) :
  (∀ k : ℕ, a m (k + 1) ∣ a m k) → m = 1944 :=
begin
  sorry
end

end find_max_m_l185_185764


namespace smallest_perimeter_of_square_sides_l185_185890

/-
  Define a predicate for the triangle inequality condition for squares of integers.
-/
def triangle_ineq_squares (a b c : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2)

/-
  Statement that proves the smallest possible perimeter given the conditions.
-/
theorem smallest_perimeter_of_square_sides : 
  ∃ a b c : ℕ, a < b ∧ b < c ∧ triangle_ineq_squares a b c ∧ a^2 + b^2 + c^2 = 77 :=
sorry

end smallest_perimeter_of_square_sides_l185_185890


namespace option_B_is_odd_function_l185_185296

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185296


namespace g_is_odd_l185_185496

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185496


namespace problem_statement_l185_185042

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185042


namespace g_is_odd_l185_185335

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185335


namespace log_identity_l185_185904

variable {a N : ℝ} {m : ℝ}

-- Define the conditions for the problem
def N_positive (N : ℝ) : Prop := N > 0
def m_nonzero (m : ℝ) : Prop := m ≠ 0

-- State the theorem
theorem log_identity (hN : N_positive N) (hm : m_nonzero m) : log (N^m) / log (a^m) = log N / log a := by
  sorry

end log_identity_l185_185904


namespace proof_third_length_gcd_l185_185885

/-- Statement: The greatest possible length that can be used to measure the given lengths exactly is 1 cm, 
and the third length is an unspecified number of centimeters that is relatively prime to both 1234 cm and 898 cm. -/
def third_length_gcd (x : ℕ) : Prop := 
  Int.gcd 1234 898 = 1 ∧ Int.gcd (Int.gcd 1234 898) x = 1

noncomputable def greatest_possible_length : ℕ := 1

theorem proof_third_length_gcd (x : ℕ) (h : third_length_gcd x) : greatest_possible_length = 1 := by
  sorry

end proof_third_length_gcd_l185_185885


namespace problem_is_odd_function_proof_l185_185459

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185459


namespace odd_function_option_B_l185_185549

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185549


namespace g_is_odd_l185_185031

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185031


namespace odd_function_g_l185_185271

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185271


namespace minimize_triangle_perimeter_l185_185954

-- Define points and basic geometric relations
variables {O A B P P1 P2 A' B' : Type}
variables [DecidableEq A] [DecidableEq B]
variables [AffineSpace S ℝ] [InnerProductSpace ℝ E] {P Q : affine_subspace ℝ E}

-- Define the conditions
axiom O_vertex : O
axiom acute_angle : ∀ {P Q R : affine_subspace ℝ E}, ∡ P Q R < π / 2
axiom P_inside : ∀ {A B O P A' B' : Type}, P ∈ convex_hull (insert A (insert B ∅))

-- Define symmetry points and intersections
axiom P1_symmetric : is_symmetric P1 O
axiom P2_symmetric : is_symmetric P2 O
axiom A'_intersection : A' = line_through P1 P2 ∩ line_through O A
axiom B'_intersection : B' = line_through P1 P2 ∩ line_through O B

-- Define the problem statement as a theorem to be proved
theorem minimize_triangle_perimeter 
  (A B : Type)
  [LinearOrder A] [LinearOrder B]
  (is_minimal_perimeter : ∀ {P A B} (O : O) (P_inside O P) 
    [
      A = A',
      B = B'  
    ]) 
  : ∀ {O A B P P1 P2 A' B' : A B}, 
    -- the positions of A and B uniquely minimize the perimeter of ∆PAB
    perimeter P A B = distance P1 P2 := 
sorry

end minimize_triangle_perimeter_l185_185954


namespace odd_function_g_l185_185208

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185208


namespace g_is_odd_l185_185332

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185332


namespace g_is_odd_l185_185086

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185086


namespace odd_function_g_l185_185175

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185175


namespace is_odd_g_l185_185489

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185489


namespace odd_function_g_l185_185277

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185277


namespace f_2012_8_eq_5_l185_185706

def sumOfDigits (n : Nat) : Nat :=
  n.digits.foldr (· + ·) 0

def f (n : Nat) : Nat :=
  sumOfDigits (n^2 + 1)

def f_1 (n : Nat) : Nat := f(n)
def f_2 (n : Nat) : Nat := f(f_1(n))
def f_k (k : Nat) (n : Nat) : Nat :=
  match k with
  | 1 => f_1(n)
  | 2 => f_2(n)
  | k+1 => f(f_k k n)

theorem f_2012_8_eq_5 : f_k 2012 8 = 5 := by
  sorry

end f_2012_8_eq_5_l185_185706


namespace g_is_odd_l185_185235

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185235


namespace odd_function_g_l185_185284

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185284


namespace is_odd_g_l185_185479

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185479


namespace is_odd_g_l185_185462

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185462


namespace odd_function_option_B_l185_185400

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185400


namespace odd_function_shifted_f_l185_185352

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185352


namespace g_is_odd_l185_185228

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185228


namespace optionB_is_odd_l185_185579

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185579


namespace odd_function_result_l185_185246

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185246


namespace odd_function_g_l185_185198

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185198


namespace odd_function_proof_l185_185425

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185425


namespace g_B_is_odd_l185_185672

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185672


namespace g_is_odd_l185_185340

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185340


namespace odd_function_g_l185_185183

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185183


namespace base7_addition_l185_185993

theorem base7_addition (Y X : Nat) (k m : Int) :
    (Y + 2 = X + 7 * k) ∧ (X + 5 = 4 + 7 * m) ∧ (5 = 6 + 7 * -1) → X + Y = 10 :=
by
  sorry

end base7_addition_l185_185993


namespace odd_function_proof_l185_185420

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185420


namespace option_b_is_odd_l185_185113

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185113


namespace g_is_odd_l185_185076

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185076


namespace sum_of_exponents_sqrt_l185_185807

theorem sum_of_exponents_sqrt (a b c : ℕ) : 2 + 4 + 6 = 12 := by
  sorry

end sum_of_exponents_sqrt_l185_185807


namespace odd_function_g_l185_185536

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185536


namespace measure_AOC_l185_185734

variables (A O B C D E Y : Point) -- Points involved.
variables (r s : Line) -- Lines r and s.

-- Conditions given in the problem:
hypothesis (h1 : ∠ A O D = 90)
hypothesis (h2 : ∠ B O Y = 90)
hypothesis (h3 : 40 ≤ ∠ D O Y ∧ ∠ D O Y ≤ 50)
hypothesis (h4 : lies_on C r)
hypothesis (h5 : lies_on Y r)
hypothesis (h6 : lies_on D s)
hypothesis (h7 : lies_on E s)

-- Statement to prove:
theorem measure_AOC : 40 ≤ ∠ A O C ∧ ∠ A O C ≤ 50 :=
sorry

end measure_AOC_l185_185734


namespace odd_function_proof_l185_185428

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185428


namespace g_is_odd_l185_185613

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185613


namespace odd_function_option_B_l185_185573

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185573


namespace odd_function_g_l185_185159

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185159


namespace g_is_odd_l185_185498

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185498


namespace g_is_odd_l185_185614

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185614


namespace g_is_odd_l185_185618

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185618


namespace radius_of_larger_ball_l185_185850

theorem radius_of_larger_ball :
  let V_small : ℝ := (4/3) * Real.pi * (2^3)
  let V_total : ℝ := 6 * V_small
  let R : ℝ := (48:ℝ)^(1/3)
  V_total = (4/3) * Real.pi * (R^3) := 
  by
  sorry

end radius_of_larger_ball_l185_185850


namespace odd_function_g_l185_185524

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185524


namespace function_does_not_have_property_P_l185_185763

-- Definition of property P
def hasPropertyP (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f ((x1 + x2) / 2) = (f x1 + f x2) / 2

-- Function in question
def f (x : ℝ) : ℝ :=
  x^2

-- Statement that function f does not have property P
theorem function_does_not_have_property_P : ¬hasPropertyP f :=
  sorry

end function_does_not_have_property_P_l185_185763


namespace option_b_is_odd_l185_185116

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185116


namespace g_B_is_odd_l185_185659

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185659


namespace g_is_odd_l185_185218

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185218


namespace g_is_odd_l185_185515

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185515


namespace g_is_odd_l185_185608

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185608


namespace cake_pieces_in_pan_l185_185723

theorem cake_pieces_in_pan :
  (24 * 30) / (3 * 2) = 120 := by
  sorry

end cake_pieces_in_pan_l185_185723


namespace is_odd_g_l185_185475

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185475


namespace odd_function_g_l185_185273

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185273


namespace odd_function_l185_185132

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185132


namespace odd_function_proof_l185_185430

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185430


namespace odd_function_option_B_l185_185562

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185562


namespace odd_function_g_l185_185182

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185182


namespace g_is_odd_l185_185629

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185629


namespace problem_statement_l185_185059

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185059


namespace odd_function_result_l185_185255

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185255


namespace odd_function_l185_185136

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185136


namespace book_distribution_l185_185860

/-- There are 6 different books. -/
def books : ℕ := 6

/-- There are three individuals: A, B, and C. -/
def individuals : ℕ := 3

/-- Each individual receives exactly 2 books. -/
def books_per_individual : ℕ := 2

/-- The number of distinct ways to distribute the books is 90. -/
theorem book_distribution :
  (Nat.choose books books_per_individual) * 
  (Nat.choose (books - books_per_individual) books_per_individual) = 90 :=
by
  sorry

end book_distribution_l185_185860


namespace option_B_is_odd_function_l185_185317

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185317


namespace sin_neg_31pi_over_6_l185_185981

theorem sin_neg_31pi_over_6 : sin (-31 * real.pi / 6) = 1 / 2 := by
  -- Conditions
  have h1 : ∀ θ, sin (-θ) = -sin (θ), from real.sin_neg
  have h2 : ∀ (θ k : ℤ), sin (θ + 2 * k * real.pi) = sin θ, from real.sin_periodic
  have h3 : 31 * real.pi / 6 - 6 * real.pi = 5 * real.pi / 6, by
    linarith
  have h4 : sin (5 * real.pi / 6) = 1 / 2, from real.sin_rat_pi (5 / 6)

  -- Proof
  sorry

end sin_neg_31pi_over_6_l185_185981


namespace odd_function_g_l185_185199

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185199


namespace odd_function_g_l185_185168

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185168


namespace problem_is_odd_function_proof_l185_185435

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185435


namespace optionB_is_odd_l185_185581

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185581


namespace odd_function_result_l185_185253

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185253


namespace externally_tangent_circles_l185_185710

/-- Given the equations of two circles and the condition that they are externally tangent, 
    prove that the value of \( m \) satisfies |m| = 3. --/
theorem externally_tangent_circles (m : ℝ) :
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4},
      C2 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * m * p.1 + m^2 - 1 = 0} in
  (∃ p₁ p₂ : ℝ × ℝ, p₁ ∈ C1 ∧ p₂ ∈ C2 ∧ dist p₁ p₂ = 3) ↔ (|m| = 3) :=
sorry

end externally_tangent_circles_l185_185710


namespace odd_function_g_l185_185003

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185003


namespace find_b_from_root_and_constant_l185_185955

theorem find_b_from_root_and_constant
  (b k : ℝ)
  (h₁ : k = 44)
  (h₂ : ∃ (x : ℝ), x = 4 ∧ 2*x^2 + b*x - k = 0) :
  b = 3 :=
by
  sorry

end find_b_from_root_and_constant_l185_185955


namespace g_B_is_odd_l185_185674

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185674


namespace positive_value_of_A_l185_185750

def my_relation (A B k : ℝ) : ℝ := A^2 + k * B^2

theorem positive_value_of_A (A : ℝ) (h1 : ∀ A B, my_relation A B 3 = A^2 + 3 * B^2) (h2 : my_relation A 7 3 = 196) :
  A = 7 := by
  sorry

end positive_value_of_A_l185_185750


namespace option_b_is_odd_l185_185125

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185125


namespace odd_function_result_l185_185243

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185243


namespace odd_function_g_l185_185165

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185165


namespace option_b_is_odd_l185_185121

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185121


namespace option_B_is_odd_function_l185_185295

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185295


namespace problem_is_odd_function_proof_l185_185449

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185449


namespace odd_function_option_B_l185_185552

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185552


namespace g_B_is_odd_l185_185676

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185676


namespace g_is_odd_l185_185506

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185506


namespace g_is_odd_l185_185087

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185087


namespace g_is_odd_l185_185014

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185014


namespace sum_row_10_pascal_triangle_l185_185894

theorem sum_row_10_pascal_triangle :
  (∑ k in Finset.range (11), Nat.choose 10 k) = 1024 :=
by
  sorry

end sum_row_10_pascal_triangle_l185_185894


namespace problem_is_odd_function_proof_l185_185446

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185446


namespace option_B_is_odd_function_l185_185300

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185300


namespace problem_is_odd_function_proof_l185_185452

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185452


namespace odd_function_shifted_f_l185_185370

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185370


namespace exists_x_such_that_f_x_eq_0_l185_185769

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then
  3 * x - 4
else
  -x^2 + 3 * x - 5

theorem exists_x_such_that_f_x_eq_0 :
  ∃ x : ℝ, f x = 0 ∧ x = 1.192 :=
sorry

end exists_x_such_that_f_x_eq_0_l185_185769


namespace g_is_odd_l185_185090

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185090


namespace g_is_odd_l185_185080

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185080


namespace curve_C2_parametric_eq_min_distance_C_C2_max_value_m_abc_inequality_l185_185924

-- Part 1: Curves and distance
theorem curve_C2_parametric_eq :
  ∀ θ : Real, ∃ x' y' : Real, x' = 3 * Real.cos θ ∧ y' = 2 * Real.sin θ :=
by
  intros θ
  use 3 * Real.cos θ, 2 * Real.sin θ
  exact ⟨rfl, rfl⟩

theorem min_distance_C_C2 :
  ∃ m : Real, m = Real.sqrt 5 :=
by
  use Real.sqrt 5
  exact rfl

-- Part 2: Inequalities
theorem max_value_m :
  ∀ x : Real, |x - 1| + |x - 2| ≥ 1 :=
by
  intro x
  calc
    |x - 1| + |x - 2| ≥ |(x - 1) - (x - 2)| : abs_add_abs_ge_abs_sub_abs (x - 1) (x - 2)
                   ... = 1 : abs_eq_self.mpr (sub_nonneg.mpr (le_rfl))

theorem abc_inequality
  (a b c : Real) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (1/a) + (1/(2*b)) + (1/(3*c)) = 1) : a + 2*b + 3*c ≥ 9 :=
by
  let k := 1 in
  have h1 : 1 / a + 1 / (2 * b) + 1 / (3 * c) = k := h
  calc
    a + 2 * b + 3 * c
        = (a + 2 * b + 3 * c) * (1 / a + 1 / (2 * b) + 1 / (3 * c)) : by rw [h1, mul_one]
    ... ≥ 9 : sorry -- Proof using AM-GM inequality

-- Placeholder for the proof involving AM-GM inequalities and detailed calculations

end curve_C2_parametric_eq_min_distance_C_C2_max_value_m_abc_inequality_l185_185924


namespace option_B_is_odd_function_l185_185304

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185304


namespace option_B_is_odd_function_l185_185298

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185298


namespace target_destroyed_probability_l185_185985

noncomputable def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  let miss1 := 1 - p1
  let miss2 := 1 - p2
  let miss3 := 1 - p3
  let prob_all_miss := miss1 * miss2 * miss3
  let prob_one_hit := (p1 * miss2 * miss3) + (miss1 * p2 * miss3) + (miss1 * miss2 * p3)
  let prob_destroyed := 1 - (prob_all_miss + prob_one_hit)
  prob_destroyed

theorem target_destroyed_probability :
  probability_hit 0.9 0.9 0.8 = 0.954 :=
sorry

end target_destroyed_probability_l185_185985


namespace option_b_is_odd_l185_185108

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185108


namespace megan_removed_albums_l185_185780

theorem megan_removed_albums :
  ∀ (albums_in_cart : ℕ) (songs_per_album : ℕ) (total_songs_bought : ℕ),
    albums_in_cart = 8 →
    songs_per_album = 7 →
    total_songs_bought = 42 →
    albums_in_cart - (total_songs_bought / songs_per_album) = 2 :=
by
  intros albums_in_cart songs_per_album total_songs_bought h1 h2 h3
  sorry

end megan_removed_albums_l185_185780


namespace max_value_when_a_equals_4_min_max_value_of_quadratic_func_l185_185700

-- (1) Prove that when \( a = 4 \), the maximum value of the function \( y = x^2 - 4x + 2 \) in the interval \( [0, 1] \) is 2.
theorem max_value_when_a_equals_4 :
  ∃ (x : ℝ), x ∈ Icc 0 1 ∧ (∀ (y : ℝ), y = x^2 - 4 * x + 2 → y ≤ 2) ∧ (∃ (x' : ℝ), x' ∈ Icc 0 1 ∧ x'^2 - 4 * x' + 2 = 2) :=
sorry

-- (2) Prove that the minimum value of the maximum value \( t \) of the function \( y = x^2 - ax + \frac{a}{2} \) in the interval \( [0, 1] \) is \( \frac{1}{2} \).
theorem min_max_value_of_quadratic_func :
  ∃ (a t : ℝ), (∀ (x : ℝ), x ∈ Icc 0 1 → (∀ (y : ℝ), y = x^2 - a * x + a / 2 → y ≤ t)) ∧ 
  (∀ (a : ℝ), (a = 1 → t = 1 / 2) ∧ (a < 1 → t = 1 - a / 2) ∧ (a > 1 → t = a / 2)) ∧ (t = 1 / 2) :=
sorry

end max_value_when_a_equals_4_min_max_value_of_quadratic_func_l185_185700


namespace bicycle_cost_after_tax_l185_185927

theorem bicycle_cost_after_tax :
  let original_price := 300
  let first_discount := original_price * 0.40
  let price_after_first_discount := original_price - first_discount
  let second_discount := price_after_first_discount * 0.20
  let price_after_second_discount := price_after_first_discount - second_discount
  let tax := price_after_second_discount * 0.05
  price_after_second_discount + tax = 151.20 :=
by
  sorry

end bicycle_cost_after_tax_l185_185927


namespace odd_function_shifted_f_l185_185371

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185371


namespace g_is_odd_l185_185624

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185624


namespace option_B_is_odd_function_l185_185299

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185299


namespace odd_function_g_l185_185158

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185158


namespace odd_function_option_B_l185_185558

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185558


namespace option_b_is_odd_l185_185112

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185112


namespace odd_function_g_l185_185002

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185002


namespace find_n_lines_l185_185730

theorem find_n_lines (n : ℕ) 
    (h1 : ∀ (i j : ℕ), i ≠ j → intersects i j)
    (h2 : ∀ (A B C D : Point), ¬ collinear  A B C D)
    (h3 : ∃ (P : Point), num_intersections P = 16)
    (h4 : ∃ (Q : Point), num_lines Q = 3 ∧ num_points Q = 6) :
    n = 8 := 
sorry

end find_n_lines_l185_185730


namespace g_is_odd_l185_185502

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185502


namespace is_odd_g_l185_185470

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185470


namespace odd_function_g_l185_185161

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185161


namespace possible_x_values_are_10_or_52_l185_185926

noncomputable def find_possible_x_values : set ℕ :=
{x | ∃ k : ℕ, k > 0 ∧ (k < 20) ∧ k * (x - 3) = 49 }

theorem possible_x_values_are_10_or_52 :
  find_possible_x_values = {10, 52} :=
by {
  -- Proof would go here
  sorry
}

end possible_x_values_are_10_or_52_l185_185926


namespace odd_function_option_B_l185_185391

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185391


namespace option_b_is_odd_l185_185104

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185104


namespace find_XY_base10_l185_185991

theorem find_XY_base10 (X Y : ℕ) (h₁ : Y + 2 = X) (h₂ : X + 5 = 11) : X + Y = 10 := 
by 
  sorry

end find_XY_base10_l185_185991


namespace odd_function_g_l185_185167

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185167


namespace max_area_of_right_triangle_with_hypotenuse_8_l185_185836

theorem max_area_of_right_triangle_with_hypotenuse_8 
  (h : ∀ a b c : ℝ, a^2 + b^2 = c^2 ∧ c = 8 → (1/2) * a * b ≤ 16) :
  ∃ a b : ℝ, a^2 + b^2 = 8^2 ∧ (1/2) * a * b = 16 :=
begin
  sorry
end

end max_area_of_right_triangle_with_hypotenuse_8_l185_185836


namespace option_B_is_odd_function_l185_185313

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185313


namespace radius_of_larger_ball_l185_185848

theorem radius_of_larger_ball :
  let V_small : ℝ := (4/3) * Real.pi * (2^3)
  let V_total : ℝ := 6 * V_small
  let R : ℝ := (48:ℝ)^(1/3)
  V_total = (4/3) * Real.pi * (R^3) := 
  by
  sorry

end radius_of_larger_ball_l185_185848


namespace odd_function_l185_185141

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185141


namespace odd_function_g_l185_185010

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185010


namespace sum_first_100_even_minus_sum_first_100_odd_l185_185854

theorem sum_first_100_even_minus_sum_first_100_odd : 
  let evens := list.range 100 |>.map (λ n, 2 * (n + 1))
  let odds := list.range 100 |>.map (λ n, 2 * (n + 1) - 1)
  list.sum evens - list.sum odds = 100 := by 
  sorry

end sum_first_100_even_minus_sum_first_100_odd_l185_185854


namespace odd_function_option_B_l185_185385

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185385


namespace find_GQ_in_triangle_l185_185739

theorem find_GQ_in_triangle 
  (X Y Z G Q : Type*) 
  [MetricSpace (X Y Z G Q)] 
  (dXY : dist X Y = 12) 
  (dXZ : dist X Z = 15) 
  (dYZ : dist Y Z = 25) 
  (is_centroid : is_centroid X Y Z G) 
  (is_foot_of_altitude : is_foot_of_altitude G Y Z Q) : 
  dist G Q = 2 * (real.sqrt 4004) / 75 :=
sorry

end find_GQ_in_triangle_l185_185739


namespace odd_function_result_l185_185245

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185245


namespace g_is_odd_l185_185221

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185221


namespace radius_larger_ball_l185_185844

-- Define the volume formula for a sphere.
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Parameters for the problem
def radius_small_ball : ℝ := 2
def total_volume_small_balls : ℝ := 6 * volume_of_sphere radius_small_ball

-- Prove that the radius of the larger ball is 4 * 2^(1 / 3) (which is 4 * cube root of 2).
theorem radius_larger_ball : ∃ r : ℝ, volume_of_sphere r = total_volume_small_balls ∧ r = 4 * Real.cbrt 2 := by
  sorry

end radius_larger_ball_l185_185844


namespace g_is_odd_l185_185035

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185035


namespace cost_price_per_meter_is_48_l185_185913

-- Definitions of the given values
def total_cost : ℝ := 444
def total_length_of_cloth : ℝ := 9.25

-- The cost price per meter
def cost_price_per_meter (total_cost : ℝ) (total_length_of_cloth : ℝ) : ℝ :=
  total_cost / total_length_of_cloth

-- Statement to prove
theorem cost_price_per_meter_is_48 :
  cost_price_per_meter total_cost total_length_of_cloth = 48 :=
by 
  unfold cost_price_per_meter 
  norm_num
  sorry

end cost_price_per_meter_is_48_l185_185913


namespace odd_function_l185_185140

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185140


namespace odd_function_proof_l185_185415

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185415


namespace problem_statement_l185_185055

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185055


namespace option_B_is_odd_function_l185_185311

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185311


namespace odd_function_g_l185_185289

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185289


namespace g_B_is_odd_l185_185678

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185678


namespace odd_function_g_l185_185269

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185269


namespace odd_function_option_B_l185_185551

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185551


namespace g_is_odd_l185_185616

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185616


namespace g_is_odd_l185_185021

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185021


namespace find_average_percent_l185_185707

variable {x : ℝ}

theorem find_average_percent
  (hx_pos : 0 ≤ x)
  (h_total_15 := 15 * x : ℝ)
  (h_total_10 : ℝ := 10 * 90)
  (h_avg_25 : (h_total_15 + h_total_10) / 25 = 78) :
  x = 70 :=
by
  let h1 := h_total_15 + h_total_10
  have h2 : h1 / 25 = 78 := h_avg_25
  have h3 : 15 * x + 900 = 78 * 25
  calc 
    15 * x + 900 = 1950 : by sorry
    15 * x = 1050 - 900 : by sorry
    x = 1050 / 15 : by sorry
    x = 70 : by sorry

end find_average_percent_l185_185707


namespace volume_of_cut_pyramid_l185_185940

theorem volume_of_cut_pyramid
  (base_length : ℝ)
  (slant_length : ℝ)
  (cut_height : ℝ)
  (original_base_area : ℝ)
  (original_height : ℝ)
  (new_base_area : ℝ)
  (volume : ℝ)
  (h_base_length : base_length = 8 * Real.sqrt 2)
  (h_slant_length : slant_length = 10)
  (h_cut_height : cut_height = 3)
  (h_original_base_area : original_base_area = (base_length ^ 2) / 2)
  (h_original_height : original_height = Real.sqrt (slant_length ^ 2 - (base_length / Real.sqrt 2) ^ 2))
  (h_new_base_area : new_base_area = original_base_area / 4)
  (h_volume : volume = (1 / 3) * new_base_area * cut_height) :
  volume = 32 :=
by
  sorry

end volume_of_cut_pyramid_l185_185940


namespace optionB_is_odd_l185_185576

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185576


namespace problem_is_odd_function_proof_l185_185440

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185440


namespace sam_bought_new_books_l185_185961

   def books_question (a m u : ℕ) : ℕ := (a + m) - u

   theorem sam_bought_new_books (a m u : ℕ) (h1 : a = 13) (h2 : m = 17) (h3 : u = 15) :
     books_question a m u = 15 :=
   by sorry
   
end sam_bought_new_books_l185_185961


namespace odd_function_option_B_l185_185557

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185557


namespace catch_up_distance_l185_185827

-- Define the setup and parameters
def speed_bike : ℝ := 16
def speed_motor : ℝ := 56
def start_delay : ℝ := 4
def first_travel_time : ℝ := 1.5
def break_time : ℝ := 1.5
def total_initial_time : ℝ := first_travel_time + break_time

-- Main theorem statement
theorem catch_up_distance : ∃ t : ℝ, (total_initial_time + t) * speed_bike = t * speed_motor ∧ (total_initial_time + t) * speed_bike = 56 := 
by
  use 1
  split
  sorry
  sorry

end catch_up_distance_l185_185827


namespace distinct_4_digit_numbers_l185_185702

theorem distinct_4_digit_numbers : 
  ∃ n, n = 7 ∧ ∀ x, (digits x = [2, 0, 3, 3] → x > 1000 → distinct_digits x) :=
sorry

end distinct_4_digit_numbers_l185_185702


namespace odd_function_l185_185151

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185151


namespace problem_is_odd_function_proof_l185_185442

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185442


namespace problem_statement_equality_case_l185_185760

variable {α : Type}
variables {A : set (set α)}
variables [fintype α] [∀ S : set α, fintype S]

noncomputable def union_all (A : set (set α)) : set α :=
  ⋃₀ A

noncomputable def union_k (A : set (set α)) (k : ℕ) (h : fintype A) : set (set α) :=
  {S | S ⊆ A ∧ cardinal.mk S = k}

noncomputable def union_k_minus_1 (A : set (set α)) (k : ℕ) (h : fintype A) : set (set α) :=
  {S | S ⊆ A ∧ cardinal.mk S = k-1}

theorem problem_statement {n k : ℕ} {X : set α} (r : ℕ)
  (h1 : ∀ (A_i : set α), A_i ∈ A → cardinal.mk A_i = r)
  (h2 : union_all A = X)
  (h3 : ∀ (k_set : set (set α)), k_set ∈ union_k A k (by apply_instance) → union_all k_set = X)
  (h4 : ∀ (k_minus_1_set : set (set α)), k_minus_1_set ∈ union_k_minus_1 A k (by apply_instance) → union_all k_minus_1_set ⊂ X) :
  ∃ x : ℤ, x ≥ 0 :=
by sorry

theorem equality_case {n k : ℕ} {X : set α} (r : ℕ)
  (h1 : ∀ (A_i : set α), A_i ∈ A → cardinal.mk A_i = r)
  (h2 : union_all A = X)
  (h3 : ∀ (k_set : set (set α)), k_set ∈ union_k A k (by apply_instance) → union_all k_set = X)
  (h4 : ∀ (k_minus_1_set : set (set α)), k_minus_1_set ∈ union_k_minus_1 A k (by apply_instance) → union_all k_minus_1_set ⊂ X)
  (h5 : cardinal.mk X = nat.choose n (k-1)) :
  r = nat.choose (n-1) (k-1) :=
by sorry

end problem_statement_equality_case_l185_185760


namespace g_is_odd_l185_185324

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185324


namespace odd_function_g_l185_185181

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185181


namespace option_B_is_odd_function_l185_185319

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185319


namespace odd_function_proof_l185_185407

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185407


namespace g_is_odd_l185_185077

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185077


namespace g_B_is_odd_l185_185666

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185666


namespace odd_function_g_l185_185272

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185272


namespace g_is_odd_l185_185029

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185029


namespace is_odd_g_l185_185477

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185477


namespace odd_function_option_B_l185_185555

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185555


namespace option_B_is_odd_function_l185_185297

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185297


namespace radius_of_larger_ball_l185_185849

theorem radius_of_larger_ball :
  let V_small : ℝ := (4/3) * Real.pi * (2^3)
  let V_total : ℝ := 6 * V_small
  let R : ℝ := (48:ℝ)^(1/3)
  V_total = (4/3) * Real.pi * (R^3) := 
  by
  sorry

end radius_of_larger_ball_l185_185849


namespace odd_function_g_l185_185205

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185205


namespace g_is_odd_l185_185026

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185026


namespace option_B_odd_l185_185638

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185638


namespace hyperbola_range_of_k_l185_185712

theorem hyperbola_range_of_k (k : ℝ) (h : 1 < real.sqrt(4 - k) / 2 ∧ real.sqrt(4 - k) / 2 < 2) : -12 < k ∧ k < 0 :=
by
  sorry

end hyperbola_range_of_k_l185_185712


namespace odd_function_option_B_l185_185379

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185379


namespace sum_p_q_l185_185832

-- Define the cubic polynomial q(x)
def cubic_q (q : ℚ) (x : ℚ) := q * x * (x - 1) * (x + 1)

-- Define the linear polynomial p(x)
def linear_p (p : ℚ) (x : ℚ) := p * x

-- Prove the result for p(x) + q(x)
theorem sum_p_q : 
  (∀ p q : ℚ, linear_p p 4 = 4 → cubic_q q 3 = 3 → (∀ x : ℚ, linear_p p x + cubic_q q x = (1 / 24) * x^3 + (23 / 24) * x)) :=
by
  intros p q hp hq x
  sorry

end sum_p_q_l185_185832


namespace total_pounds_of_peppers_l185_185972

def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335
def total_peppers : ℝ := 5.666666666666667

theorem total_pounds_of_peppers :
  green_peppers + red_peppers = total_peppers :=
by
  -- sorry: Proof is omitted
  sorry

end total_pounds_of_peppers_l185_185972


namespace odd_function_g_l185_185266

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185266


namespace proper_polygons_m_lines_l185_185788

noncomputable def smallest_m := 2

theorem proper_polygons_m_lines (P : Finset (Set (ℝ × ℝ)))
  (properly_placed : ∀ (p1 p2 : Set (ℝ × ℝ)), p1 ∈ P → p2 ∈ P → ∃ l : Set (ℝ × ℝ), (0, 0) ∈ l ∧ ∀ (p : Set (ℝ × ℝ)), p ∈ P → ¬Disjoint l p) :
  ∃ (m : ℕ), m = smallest_m ∧ ∀ (lines : Finset (Set (ℝ × ℝ))), 
    (∀ l ∈ lines, (0, 0) ∈ l) → lines.card = m → ∀ p ∈ P, ∃ l ∈ lines, ¬Disjoint l p := sorry

end proper_polygons_m_lines_l185_185788


namespace option_b_is_odd_l185_185103

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185103


namespace g_is_odd_l185_185214

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185214


namespace g_B_is_odd_l185_185661

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185661


namespace minimize_K_l185_185973

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 6 * (1 - p) * (1 - q)

noncomputable def K (p : ℝ) : ℝ :=
  max (H p 0) (H p 1)

theorem minimize_K : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → K p ≤ K p') ∧ p = 10 / 11 :=
begin
  sorry
end

end minimize_K_l185_185973


namespace odd_function_g_l185_185173

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185173


namespace optionB_is_odd_l185_185583

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185583


namespace optionB_is_odd_l185_185598

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185598


namespace odd_function_shifted_f_l185_185356

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185356


namespace problem_statement_l185_185047

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185047


namespace odd_function_l185_185144

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185144


namespace fraction_of_third_is_eighth_l185_185882

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_third_is_eighth_l185_185882


namespace g_is_odd_l185_185015

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185015


namespace odd_function_l185_185152

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185152


namespace mechanic_earns_on_fourth_day_l185_185819

theorem mechanic_earns_on_fourth_day 
  (E1 E2 E3 E4 E5 E6 E7 : ℝ)
  (h1 : (E1 + E2 + E3 + E4) / 4 = 18)
  (h2 : (E4 + E5 + E6 + E7) / 4 = 22)
  (h3 : (E1 + E2 + E3 + E4 + E5 + E6 + E7) / 7 = 21) 
  : E4 = 13 := 
by 
  sorry

end mechanic_earns_on_fourth_day_l185_185819


namespace g_is_odd_l185_185213

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185213


namespace odd_function_result_l185_185260

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185260


namespace optionB_is_odd_l185_185586

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185586


namespace odd_function_l185_185149

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185149


namespace gomoku_black_pieces_l185_185874

/--
Two students, A and B, are preparing to play a game of Gomoku but find that 
the box only contains a certain number of black and white pieces, each of the
same quantity, and the total does not exceed 10. Then, they find 20 more pieces 
(only black and white) and add them to the box. At this point, the ratio of 
the total number of white to black pieces is 7:8. We want to prove that the total number
of black pieces in the box after adding is 16.
-/
theorem gomoku_black_pieces (x y : ℕ) (hx : x = 15 * y - 160) (h_total : x + y ≤ 5)
  (h_ratio : 7 * (x + y) = 8 * (x + (20 - y))) : (x + y = 16) :=
by
  sorry

end gomoku_black_pieces_l185_185874


namespace remainder_when_divided_by_19_l185_185906

theorem remainder_when_divided_by_19 {N : ℤ} (h : N % 342 = 47) : N % 19 = 9 :=
sorry

end remainder_when_divided_by_19_l185_185906


namespace income_ratio_l185_185841

theorem income_ratio (I1 I2 E1 E2 : ℕ)
  (hI1 : I1 = 3500)
  (hE_ratio : (E1:ℚ) / E2 = 3 / 2)
  (hSavings : ∀ (x y : ℕ), x - E1 = 1400 ∧ y - E2 = 1400 → x = I1 ∧ y = I2) :
  I1 / I2 = 5 / 4 :=
by
  -- The proof steps would go here
  sorry

end income_ratio_l185_185841


namespace petya_run_time_l185_185790

-- Definitions
def time_petya_4_to_1 : ℕ := 12

-- Conditions
axiom time_mom_condition : ∃ (time_mom : ℕ), time_petya_4_to_1 = time_mom - 2
axiom time_mom_5_to_1_condition : ∃ (time_petya_5_to_1 : ℕ), ∀ time_mom : ℕ, time_mom = time_petya_5_to_1 - 2

-- Proof statement
theorem petya_run_time :
  ∃ (time_petya_4_to_1 : ℕ), time_petya_4_to_1 = 12 :=
sorry

end petya_run_time_l185_185790


namespace optionB_is_odd_l185_185601

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185601


namespace problem_is_odd_function_proof_l185_185447

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185447


namespace number_of_subsets_set_A_l185_185838

theorem number_of_subsets_set_A :
  ∃ (A : Finset ℕ), (A = {1, 2}) ∧ (A.powerset.card = 4) :=
begin
  let A : Finset ℕ := {1, 2},
  use A,
  split,
  { refl },
  { simp [Finset.powerset] }
end

end number_of_subsets_set_A_l185_185838


namespace odd_function_proof_l185_185418

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185418


namespace polynomial_division_l185_185984

variables {R : Type*} [CommRing R]   
open Polynomial

def f : R[X] := 4 * X^5 - 5 * X^4 + 3 * X^3 - 7 * X^2 + 9 * X - 1
def g : R[X] := X^2 + 2 * X + 3
def q : R[X] := 4 * X^3 - 13 * X^2 + 42 * X - (163 / 3 : R)
def r : R[X] := 0  -- Since the problem states it divides exactly, thus remainder is zero

theorem polynomial_division :
  f = g * q + r :=
by sorry

end polynomial_division_l185_185984


namespace problem_statement_l185_185066

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185066


namespace odd_function_g_l185_185164

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185164


namespace odd_function_option_B_l185_185397

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185397


namespace g_is_odd_l185_185072

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185072


namespace odd_function_g_l185_185012

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185012


namespace odd_function_option_B_l185_185554

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185554


namespace odd_function_shifted_f_l185_185353

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185353


namespace integer_solutions_l185_185997

-- Define the problem statement in Lean
theorem integer_solutions :
  {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y) ∧ x^2 + x = y^4 + y^3 + y^2 + y} =
  {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)} :=
by
  sorry

end integer_solutions_l185_185997


namespace massager_vibration_increase_l185_185779

theorem massager_vibration_increase :
  ∀ (v_low v_high : ℕ) (time_sec : ℕ) (total_vibrations : ℕ),
    v_low = 1600 →
    time_sec = 5 * 60 →
    total_vibrations = 768000 →
    v_high = total_vibrations / time_sec →
    (((v_high - v_low) / v_low) * 100) = 60 :=
by
  -- ∀ denotes "for all"
  -- ℕ denotes natural numbers (non-negative integers)
  intros v_low v_high time_sec total_vibrations
  assume h1 h2 h3 h4
  -- Here, we define assumptions based on the conditions
  sorry

end massager_vibration_increase_l185_185779


namespace orthocenter_DEH_on_CP_l185_185903

open EuclideanGeometry

variables (A B C H D E P : Point)
variables [AcuteAngledTriangle ABC]
variables (A' B' : Point)
variables (AA' BB' : Line)
variables [AltitudeIntersection AA' BB' ABC H]
variables (line_perpendicular_to_AB : Line)
variables [PerpendicularLineIntersection line_perpendicular_to_AB AB D E P]

theorem orthocenter_DEH_on_CP :
    Orthocenter DEH ∈ Segment C P :=
sorry

end orthocenter_DEH_on_CP_l185_185903


namespace option_b_is_odd_l185_185101

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185101


namespace projective_plane_verification_l185_185778

-- Define the type for the problem: points and lines in a projective plane
structure ProjectivePlane (P L : Type) :=
(points : P → Prop)
(lines : L → Prop)
(loc : P → L → Prop) -- a predicate indicating point lies on a line
(line_through_3_points : ∀ l : L, { p : P | loc p l }.card = 3)
(point_on_3_lines : ∀ p : P, { l : L | loc p l }.card = 3)
(num_points : ∃ n, finite { p : P | points p } ∧ { p : P | points p }.card = n)

noncomputable def projective_plane_of_order_2 (P L : Type) [fintype P] [fintype L]
  [ProjectivePlane P L] : Prop :=
(ProjectivePlane.num_points = 7) ∧
∀ l (h : @finite _ _ ProjectivePlane.lines l), set.card { p : P | ProjectivePlane.loc p l } = 3 ∧
∀ p (h : @finite _ _ ProjectivePlane.points p), set.card { l : L | ProjectivePlane.loc p l } = 3

-- The theorem stating the proof problem given the conditions
theorem projective_plane_verification (P L : Type) [fintype P] [fintype L]
  (proj_plane : ProjectivePlane P L) :
  projective_plane_of_order_2 P L :=
sorry

end projective_plane_verification_l185_185778


namespace total_amount_paid_is_correct_l185_185941

-- Definitions for the conditions
def original_price : ℝ := 150
def sale_discount : ℝ := 0.30
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

-- Calculation
def final_amount : ℝ :=
  let discounted_price := original_price * (1 - sale_discount)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price_after_tax := price_after_coupon * (1 + sales_tax)
  final_price_after_tax

-- Statement to prove
theorem total_amount_paid_is_correct : final_amount = 104.50 := by
  sorry

end total_amount_paid_is_correct_l185_185941


namespace problem_is_odd_function_proof_l185_185453

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185453


namespace g_is_odd_l185_185490

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185490


namespace pages_per_booklet_l185_185743

theorem pages_per_booklet (total_pages : ℕ) (total_booklets : ℕ) (h1 : total_pages = 441) (h2 : total_booklets = 49) : total_pages / total_booklets = 9 :=
by
  rw [h1, h2]
  exact Nat.div_self 441 49 sorry

end pages_per_booklet_l185_185743


namespace remainder_4063_div_97_l185_185888

theorem remainder_4063_div_97 : 4063 % 97 = 86 := 
by sorry

end remainder_4063_div_97_l185_185888


namespace odd_function_option_B_l185_185570

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185570


namespace g_is_odd_l185_185073

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185073


namespace optionB_is_odd_l185_185600

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185600


namespace option_B_odd_l185_185644

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185644


namespace g_is_odd_l185_185338

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185338


namespace g_is_odd_l185_185215

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185215


namespace triangle_area_l185_185949

theorem triangle_area :
  let A := (5, -2)
  let B := (13, 5)
  let C := (5, 5)
  let base := Real.sqrt ((13 - 5) ^ 2 + (5 - 5) ^ 2)
  let height := Real.sqrt ((5 - 5) ^ 2 + (5 - (-2)) ^ 2)
  let area := 0.5 * base * height
  base = 8 ∧ height = 7 → area = 28.0 := 
by
  intros A B C base height area h
  have h1 : base = 8 := by 
    -- proof of the base length
    sorry
  have h2 : height = 7 := by 
    -- proof of the height length
    sorry
  have h3 : area = 0.5 * 8 * 7 := by 
    -- proof of the area calculation
    sorry
  show area = 28.0 from
    calc
      area = 0.5 * 8 * 7 : by rw [h1, h2]
      ... = 28.0 : by norm_num

end triangle_area_l185_185949


namespace g_is_odd_l185_185034

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185034


namespace part1_part2_l185_185775

noncomputable def f (x a : ℝ) : ℝ :=
  |x + 1 / a| + |x - a|

theorem part1 (a : ℝ) (h : a > 0) (x : ℝ) : 
  f x a ≥ 2 :=
sorry

theorem part2 (a : ℝ) (h : f 3 a < 5) : 
  a ∈ set.Ioo ((1 + real.sqrt 5) / 2) ((5 + real.sqrt 21) / 2) :=
sorry

end part1_part2_l185_185775


namespace odd_function_g_l185_185276

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185276


namespace odd_function_proof_l185_185411

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185411


namespace stable_polynomials_l185_185761

def K := { n : ℕ | ∀ d ∈ n.digits 10, d ≠ 7 }

theorem stable_polynomials (f : ℕ → ℕ) (h_nonneg_coeffs : ∀ (n : ℕ), 0 ≤ n) (h_stable : ∀ (x : ℕ), x ∈ K → f x ∈ K) : ∃ e k : ℕ, (f = λ x, 10^e * x + k ∨ f = λ x, 10^e * x ∨ f = λ x, k) ∧ k ∈ K := 
sorry

end stable_polynomials_l185_185761


namespace odd_function_l185_185147

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185147


namespace option_b_is_odd_l185_185098

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185098


namespace g_B_is_odd_l185_185668

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185668


namespace cartesian_equation_of_curve_distance_sum_pa_pb_l185_185816

noncomputable def curve_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

theorem cartesian_equation_of_curve :
  ∀ (ρ θ : ℝ), ρ * sin (θ)^2 = 4 * cos (θ) → (curve_to_cartesian ρ θ).2^2 = 4 * (curve_to_cartesian ρ θ).1 :=
  by
  intros ρ θ h
  dsimp [curve_to_cartesian]
  sorry

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + (2 / sqrt 5) * t, 1 + (1 / sqrt 5) * t)

theorem distance_sum_pa_pb :
  (P A B : ℝ × ℝ) (P = (1,1)) (A ∈ curve_c) (B ∈ curve_c) (A B : ℝ × ℝ) → |A + (B : ℝ) - P| = 4 * sqrt 15 :=
  by
  intros P A B hP hA hB
  sorry

end cartesian_equation_of_curve_distance_sum_pa_pb_l185_185816


namespace rectangular_pyramid_surface_area_l185_185939

theorem rectangular_pyramid_surface_area :
  ∀ (d : ℝ) (l : ℝ) (a : ℝ),
  d = 2 → l = √2 → a = √2 →
  let s := 2*l/√2 + 4*l/√2*a/√2 in
  s = 2 + 4 * √2 :=
begin
  intros d l a hd hl ha,
  rw [hd, hl, ha],
  -- additional steps for simplification would be included here if proving,
  sorry -- since proof is not required
end

end rectangular_pyramid_surface_area_l185_185939


namespace g_is_odd_l185_185030

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185030


namespace odd_function_option_B_l185_185388

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185388


namespace odd_function_g_l185_185209

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185209


namespace trumpet_cost_l185_185781

def cost_of_song_book : Real := 5.84
def total_spent : Real := 151
def cost_of_trumpet : Real := total_spent - cost_of_song_book

theorem trumpet_cost : cost_of_trumpet = 145.16 :=
by
  sorry

end trumpet_cost_l185_185781


namespace find_triangle_l185_185814

theorem find_triangle : ∀ (triangle : ℕ), (∀ (d : ℕ), 0 ≤ d ∧ d ≤ 9) → (5 * 3 + triangle = 12 * triangle + 4) → triangle = 1 :=
by
  sorry

end find_triangle_l185_185814


namespace largest_equal_cost_integer_l185_185866

-- Define the cost function for Option 1: sum of the digits in base 10.
def cost_option_1 (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the cost function for Option 2: sum of the binary digits.
def cost_option_2 (n : ℕ) : ℕ :=
  (n.digits 2).sum

-- Define the main theorem we want to prove.
theorem largest_equal_cost_integer : ∃ n : ℕ, n < 500 ∧ cost_option_1 n = cost_option_2 n ∧ ∀ m < 500, cost_option_1 m = cost_option_2 m → m ≤ n :=
by
  have : ∃ n, n = 170 ∧ n < 500 ∧ cost_option_1 n = cost_option_2 n ∧ ∀ m < 500, cost_option_1 m = cost_option_2 m → m ≤ n := sorry
  let n := 170
  exact ⟨n, this.2.2.1, this.2.2.2, sorry⟩

end largest_equal_cost_integer_l185_185866


namespace problem_statement_l185_185068

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185068


namespace odd_function_result_l185_185238

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185238


namespace odd_function_g_l185_185169

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185169


namespace odd_function_proof_l185_185423

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185423


namespace payal_book_length_l185_185915

theorem payal_book_length (P : ℕ) 
  (h1 : (2/3 : ℚ) * P = (1/3 : ℚ) * P + 20) : P = 60 :=
sorry

end payal_book_length_l185_185915


namespace odd_function_shifted_f_l185_185368

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185368


namespace largest_4_digit_number_divisible_by_12_l185_185919

theorem largest_4_digit_number_divisible_by_12 : ∃ (n : ℕ), 1000 ≤ n ∧ n ≤ 9999 ∧ n % 12 = 0 ∧ ∀ m, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 12 = 0 → m ≤ n := 
sorry

end largest_4_digit_number_divisible_by_12_l185_185919


namespace g_is_odd_l185_185326

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185326


namespace radius_of_larger_ball_l185_185847

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem radius_of_larger_ball :
  (six_ball_volume : volume_of_sphere 2 * 6 = volume_of_sphere R) →
  R = 2 * Real.cbrt 3 := by
  sorry

end radius_of_larger_ball_l185_185847


namespace odd_function_proof_l185_185410

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185410


namespace radius_of_larger_ball_l185_185845

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem radius_of_larger_ball :
  (six_ball_volume : volume_of_sphere 2 * 6 = volume_of_sphere R) →
  R = 2 * Real.cbrt 3 := by
  sorry

end radius_of_larger_ball_l185_185845


namespace problem_statement_l185_185057

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185057


namespace g_is_odd_l185_185096

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185096


namespace geometric_sequence_condition_l185_185735

-- Definitions based on conditions
def S (n : ℕ) (m : ℤ) : ℤ := 3^(n + 1) + m
def a1 (m : ℤ) : ℤ := S 1 m
def a_n (n : ℕ) : ℤ := if n = 1 then a1 (-3) else 2 * 3^n

-- The proof statement
theorem geometric_sequence_condition (m : ℤ) (h1 : a1 m = 3^2 + m) (h2 : ∀ n, n ≥ 2 → a_n n = 2 * 3^n) :
  m = -3 :=
sorry

end geometric_sequence_condition_l185_185735


namespace g_is_odd_l185_185084

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185084


namespace odd_function_g_l185_185291

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end odd_function_g_l185_185291


namespace problem_statement_l185_185054

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185054


namespace unfair_selection_l185_185724

theorem unfair_selection (P : ℕ → ℚ) (classes : Finset ℕ) :
  (∀ n ∈ classes, P n = 
    match n with
    | 2 => 1 / 36
    | 3 => 1 / 18
    | 4 => 1 / 12
    | 5 => 1 / 9
    | 6 => 5 / 36
    | 7 => 1 / 6
    | 8 => 5 / 36
    | 9 => 1 / 9
    | 10 => 1 / 12
    | 11 => 1 / 18
    | 12 => 1 / 36
    | _ => 0
  ) → (∃ n ∈ classes, P n > P 7) :=
by
  intro P_classes
  have h : P 7 = 1 / 6 := P_classes 7 (Finset.mem_insert.mpr (Or.inl rfl))
  have h' : ∃ n ∈ classes, P n = 1 / 6 := ⟨7, Finset.mem_insert.mpr (Or.inl rfl), h⟩
  sorry

end unfair_selection_l185_185724


namespace odd_function_shifted_f_l185_185360

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185360


namespace fraction_of_fractions_l185_185879

theorem fraction_of_fractions : (1 / 8) / (1 / 3) = 3 / 8 := by
  sorry

end fraction_of_fractions_l185_185879


namespace pascal_row_10_sum_l185_185892

-- Definition: sum of the numbers in Row n of Pascal's Triangle is 2^n
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- Theorem: sum of the numbers in Row 10 of Pascal's Triangle is 1024
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  sorry

end pascal_row_10_sum_l185_185892


namespace g_is_odd_l185_185604

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185604


namespace odd_function_g_l185_185538

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185538


namespace g_is_odd_l185_185343

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185343


namespace problem_is_odd_function_proof_l185_185443

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185443


namespace knight_captures_pawns_in_minimum_moves_l185_185786

open Function

-- Define the initial positions as coordinates: B1 as (1, 1), B8 as (1, 8), G8 as (7, 8)
-- Knight's movement in "L" shape can be represented as offsets: (±2, ±1) or (±1, ±2)

noncomputable def knight_moves (start : ℕ × ℕ) (end : ℕ × ℕ) : ℕ := sorry

theorem knight_captures_pawns_in_minimum_moves :
  knight_moves (1, 1) (7, 8) + knight_moves (7, 8) (1, 8) = 7 :=
sorry

end knight_captures_pawns_in_minimum_moves_l185_185786


namespace optionB_is_odd_l185_185582

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185582


namespace g_B_is_odd_l185_185673

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185673


namespace assisted_work_time_l185_185910

theorem assisted_work_time (a b c : ℝ) (ha : a = 1 / 11) (hb : b = 1 / 20) (hc : c = 1 / 55) :
  (1 / ((a + b) + (a + c) / 2)) = 8 :=
by
  sorry

end assisted_work_time_l185_185910


namespace odd_function_l185_185137

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185137


namespace odd_function_l185_185130

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185130


namespace praveen_investment_calculation_l185_185793

-- Define the initial conditions and the problem statement
def praveen_initial_investment (H P : ℕ) : Prop :=
  let praveen_time := 12 in
  let hari_time := 7 in
  let profit_ratio := (2, 3) in
  (P * praveen_time) / (H * hari_time) = profit_ratio.1 / profit_ratio.2

-- The theorem stating the equivalent problem in Lean
theorem praveen_investment_calculation : praveen_initial_investment 8640 3360 :=
by
  -- Definitions and assumptions
  let H := 8640
  let P := 3360
  have h1 : (P * 12) = (2 * (H * 7)) / 3 := sorry
  have h2 : P * 12 = 60480 := sorry
  have h3 : P = 3360 := sorry
  exact sorry

end praveen_investment_calculation_l185_185793


namespace odd_function_g_l185_185544

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185544


namespace option_B_is_odd_function_l185_185309

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185309


namespace odd_function_g_l185_185533

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185533


namespace problem_is_odd_function_proof_l185_185444

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185444


namespace odd_function_result_l185_185247

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185247


namespace odd_function_proof_l185_185406

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185406


namespace odd_function_result_l185_185250

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185250


namespace odd_function_g_l185_185004

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  unfold g
  unfold f
  sorry

end odd_function_g_l185_185004


namespace g_is_odd_l185_185619

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = - g x := 
by
  -- Proof goes here
  sorry

end g_is_odd_l185_185619


namespace exists_valid_arrangement_l185_185959

-- Define the circle data type and connection
inductive circles : Type
| A | B | C | D

open circles

def connected : circles → circles → Prop
| A B | A C | B A | B D | C A | D B := true
| _ _ := false

def f : circles → ℕ
| A := 1
| B := 3
| C := 9
| D := 27

theorem exists_valid_arrangement :
  (∀ (a b : circles), connected a b → (f b / f a = 3 ∨ f b / f a = 9)) ∧
  (∀ (a b : circles), ¬ connected a b → (f b / f a ≠ 3 ∧ f b / f a ≠ 9)) :=
by
  sorry

end exists_valid_arrangement_l185_185959


namespace modulus_of_complex_l185_185988

open Complex

theorem modulus_of_complex :
  ∀ (re im : ℚ),
  re = 7/8 → im = 3/2 →
  complex.abs (re + im * Complex.I) = real.sqrt 193 / 8 :=
by
  intros re im h_re h_im
  rw [h_re, h_im]
  sorry

end modulus_of_complex_l185_185988


namespace smallest_factor_of_36_l185_185801

theorem smallest_factor_of_36 :
  ∃ a b c : ℤ, a * b * c = 36 ∧ a + b + c = 4 ∧ min (min a b) c = -4 :=
by
  sorry

end smallest_factor_of_36_l185_185801


namespace odd_function_g_l185_185156

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g(-x) = -g(x) := 
by
  sorry

end odd_function_g_l185_185156


namespace volume_ratio_condition_l185_185834

variables {m x S k : ℝ}

/-- Given the conditions on the volumes of a triangular prism and the pyramid formed by extending an edge -/
def triangular_prism_condition (m x S k : ℝ) : Prop :=
  let V := m * S in
  let V' := (m + x) * S / 3 in
  let S' := S * (x / (m + x))^2 in
  let V'' := (m + x) * S / 3 - x * S' / 3 in
  let V''' := V - V'' in
  V''' = k * V

/-- Prove that the ratio k must satisfy the given inequality -/
theorem volume_ratio_condition (m x S k : ℝ) (h : triangular_prism_condition m x S k) : k ≤ 3 / 4 :=
sorry

end volume_ratio_condition_l185_185834


namespace problem_is_odd_function_proof_l185_185458

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185458


namespace odd_function_option_B_l185_185553

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185553


namespace value_of_n_l185_185905

theorem value_of_n :
  let a := (4 + 8 + 12 + 16 + 20 + 24 + 28) / 7 in
  ∀ (n : ℕ), 
  let b := 2 * n in
  a^2 - b^2 = 0 → n = 8 :=
by
  intros
  let a := (4 + 8 + 12 + 16 + 20 + 24 + 28) / 7
  intros n
  let b := 2 * n
  intro h
  sorry -- proof goes here

end value_of_n_l185_185905


namespace g_B_is_odd_l185_185667

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185667


namespace sqrt_26_is_7th_term_l185_185690

open Real

def sequence (n : ℕ) : ℝ := sqrt (4 * n - 2)

theorem sqrt_26_is_7th_term : sequence 7 = sqrt 26 :=
by
  sorry

end sqrt_26_is_7th_term_l185_185690


namespace max_4tuples_signs_l185_185768

theorem max_4tuples_signs {a1 a2 a3 a4 b1 b2 b3 b4 : ℝ} 
(h1: a1 * b2 ≠ a2 * b1)
(h2: ∀ (x1 x2 x3 x4 : ℝ), 
    a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 = 0 ∧ 
    b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 = 0 →
    (x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ x4 ≠ 0)) :
    ∃ (s : set (ℤ × ℤ × ℤ × ℤ)), 
    s = {t | ∃ (x1 x2 x3 x4 : ℝ), 
    a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 = 0 ∧ 
    b1 * x1 + b2 * x2 + b3 * x3 + b4 * x4 = 0 ∧ 
    t = (sign x1, sign x2, sign x3, sign x4)} ∧ 
    s.card = 8 :=
sorry

end max_4tuples_signs_l185_185768


namespace lauren_total_earnings_l185_185784

-- Define earnings conditions
def mondayCommercialEarnings (views : ℕ) : ℝ := views * 0.40
def mondaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 0.80

def tuesdayCommercialEarnings (views : ℕ) : ℝ := views * 0.50
def tuesdaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 1.00

def weekendMerchandiseEarnings (sales : ℝ) : ℝ := 0.10 * sales

-- Specific conditions for each day
def mondayTotalEarnings : ℝ := mondayCommercialEarnings 80 + mondaySubscriptionEarnings 20
def tuesdayTotalEarnings : ℝ := tuesdayCommercialEarnings 100 + tuesdaySubscriptionEarnings 27
def weekendTotalEarnings : ℝ := weekendMerchandiseEarnings 150

-- Total earnings for the period
def totalEarnings : ℝ := mondayTotalEarnings + tuesdayTotalEarnings + weekendTotalEarnings

-- Examining the final value
theorem lauren_total_earnings : totalEarnings = 140.00 := by
  sorry

end lauren_total_earnings_l185_185784


namespace odd_function_proof_l185_185424

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185424


namespace odd_function_l185_185127

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185127


namespace odd_function_l185_185126

-- Define the original function
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the function which needs to be proven as an odd function
def g (x : ℝ) : ℝ := f (x - 1) + 1

-- The theorem statement to prove that 'g' is an odd function
theorem odd_function : ∀ (x : ℝ), g (-x) = -g (x) := by
  sorry

end odd_function_l185_185126


namespace odd_function_option_B_l185_185556

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g(x)

theorem odd_function_option_B :
  is_odd_function (λ x, f (x - 1) + 1) :=
  sorry

end odd_function_option_B_l185_185556


namespace min_positive_period_sin_square_l185_185837

theorem min_positive_period_sin_square (x : ℝ) : (∀y, sin^2 (2 * y - (π / 4)) = sin^2 (2 * (y + π/2) - (π / 4))) :=
by {
  sorry
}

end min_positive_period_sin_square_l185_185837


namespace problem_is_odd_function_proof_l185_185437

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185437


namespace g_is_odd_l185_185517

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185517


namespace option_B_odd_l185_185651

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185651


namespace odd_function_shifted_f_l185_185372

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185372


namespace optionB_is_odd_l185_185594

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185594


namespace odd_function_g_l185_185526

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185526


namespace odd_function_g_l185_185195

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185195


namespace odd_function_shifted_f_l185_185359

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185359


namespace parallel_lines_l185_185987

noncomputable def tangent_points (S S1 S2 : Circle) (A1 A2 : Point) : Prop :=
  S1.radius = S2.radius ∧
  S1.touches S A1 ∧
  S2.touches S A2

noncomputable def intersect_points (S S1 S2 : Circle) (C : Point) (A1 A2 B1 B2 : Point) : Prop :=
  (C ∈ S) ∧
  (line_through A1 C ∩ S1 = {B1}) ∧
  (line_through A2 C ∩ S2 = {B2})

theorem parallel_lines (S S1 S2 : Circle) (A1 A2 C B1 B2 : Point)
  (tangents : tangent_points S S1 S2 A1 A2)
  (intersections : intersect_points S S1 S2 C A1 A2 B1 B2) :
  parallel (line_through B1 B2) (line_through A1 A2) := sorry

end parallel_lines_l185_185987


namespace odd_function_g_l185_185204

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185204


namespace odd_function_g_l185_185539

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185539


namespace radius_larger_ball_l185_185843

-- Define the volume formula for a sphere.
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Parameters for the problem
def radius_small_ball : ℝ := 2
def total_volume_small_balls : ℝ := 6 * volume_of_sphere radius_small_ball

-- Prove that the radius of the larger ball is 4 * 2^(1 / 3) (which is 4 * cube root of 2).
theorem radius_larger_ball : ∃ r : ℝ, volume_of_sphere r = total_volume_small_balls ∧ r = 4 * Real.cbrt 2 := by
  sorry

end radius_larger_ball_l185_185843


namespace option_B_odd_l185_185654

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem option_B_odd : ∀ x : ℝ, f(x-1) + 1 = - (f(-(x-1)) + 1) :=
by
  sorry

end option_B_odd_l185_185654


namespace g_is_odd_l185_185025

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185025


namespace odd_function_proof_l185_185431

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185431


namespace raffle_ticket_cost_l185_185934

theorem raffle_ticket_cost (x : ℝ) 
  (h1 : 25 * x + 50 = 100) : x = 2 :=
begin
  sorry
end

end raffle_ticket_cost_l185_185934


namespace problem_statement_l185_185061

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem problem_statement : is_odd_function (λ x, f (x - 1) + 1) := sorry

end problem_statement_l185_185061


namespace odd_function_proof_l185_185409

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def transformed_function (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_proof : 
  ∀ x : ℝ, transformed_function (-x) = -transformed_function x :=
by
  sorry

end odd_function_proof_l185_185409


namespace solve_integral_eq_l185_185811

variable {α : Type*}
variable (λ : ℝ) (x : ℝ)
variable (φ : ℝ → ℝ)

def integral_term : ℝ → ℝ := 
  λ t, x * real.cos t + t^2 * real.sin x + real.cos x * real.sin t

def integral_eq : Prop := 
  ∫ t in -real.pi..real.pi, integral_term λ x t * φ t = 
    (x - φ x) / λ

noncomputable def φ_solution : ℝ → ℝ :=
  λ x, (2 * λ * real.pi / (1 + 2 * λ^2 * real.pi^2)) * (λ * real.pi * x - 4 * λ * real.pi * real.sin x + real.cos x) + x

theorem solve_integral_eq :
  (φ x - λ * ∫ t in -real.pi..real.pi, integral_term λ x t * φ t)  = x ↔ φ = φ_solution λ :=
  sorry

end solve_integral_eq_l185_185811


namespace g_is_odd_l185_185342

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185342


namespace optionB_is_odd_l185_185584

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185584


namespace optionB_is_odd_l185_185577

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185577


namespace smallest_n_for_geometric_sequence_divisibility_l185_185774

theorem smallest_n_for_geometric_sequence_divisibility :
  ∃ n : ℕ, (∀ m : ℕ, m < n → ¬ (2 * 10 ^ 6 ∣ (30 ^ (m - 1) * (5 / 6)))) ∧ (2 * 10 ^ 6 ∣ (30 ^ (n - 1) * (5 / 6))) ∧ n = 8 :=
by
  sorry

end smallest_n_for_geometric_sequence_divisibility_l185_185774


namespace g_is_odd_l185_185039

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185039


namespace odd_function_shifted_f_l185_185373

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185373


namespace g_is_odd_l185_185497

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f(x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by 
  sorry

end g_is_odd_l185_185497


namespace fractions_order_and_non_equality_l185_185898

theorem fractions_order_and_non_equality:
  (37 / 29 < 41 / 31) ∧ (41 / 31 < 31 / 23) ∧ 
  ((37 / 29 ≠ 4 / 3) ∧ (41 / 31 ≠ 4 / 3) ∧ (31 / 23 ≠ 4 / 3)) := by
  sorry

end fractions_order_and_non_equality_l185_185898


namespace optionB_is_odd_l185_185574

def f(x : ℝ) : ℝ := (1 - x) / (1 + x)

def optionB (x : ℝ) : ℝ := f(x - 1) + 1

theorem optionB_is_odd : ∀ x : ℝ, optionB (-x) = - optionB x :=
by
  sorry

end optionB_is_odd_l185_185574


namespace total_legs_is_26_l185_185967

-- Define the number of puppies and chicks
def number_of_puppies : Nat := 3
def number_of_chicks : Nat := 7

-- Define the number of legs per puppy and per chick
def legs_per_puppy : Nat := 4
def legs_per_chick : Nat := 2

-- Calculate the total number of legs
def total_legs := (number_of_puppies * legs_per_puppy) + (number_of_chicks * legs_per_chick)

-- Prove that the total number of legs is 26
theorem total_legs_is_26 : total_legs = 26 := by
  sorry

end total_legs_is_26_l185_185967


namespace white_red_balls_l185_185728

theorem white_red_balls (w r : ℕ) 
  (h1 : 3 * w = 5 * r)
  (h2 : w + 15 + r = 50) : 
  r = 12 :=
by
  sorry

end white_red_balls_l185_185728


namespace shaded_fraction_l185_185942

theorem shaded_fraction {S : ℝ} (h : 0 < S) :
  let frac_area := ∑' n : ℕ, (1/(4:ℝ)^1) * (1/(4:ℝ)^n)
  1/3 = frac_area :=
by
  sorry

end shaded_fraction_l185_185942


namespace g_is_odd_l185_185091

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define what it means for a function to be odd
def is_odd (h : ℝ → ℝ) : Prop :=
  ∀ x, h (-x) = -h (x)

-- Define the function g as per the problem
def g (x : ℝ) : ℝ := (1 - (x - 1)) / (1 + (x - 1)) + 1

-- The statement we need to prove
theorem g_is_odd : is_odd g := 
  sorry

end g_is_odd_l185_185091


namespace odd_function_option_B_l185_185398

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

theorem odd_function_option_B : is_odd_function (λ x => f (x - 1) + 1) :=
sorry

end odd_function_option_B_l185_185398


namespace problem_is_odd_function_proof_l185_185460

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_inverse_is_odd : Prop :=
  ∀ x : ℝ, f (x - 1) + 1 = -(f (-(x - 1)) + 1)

-- Statement that the function f(x-1) + 1 is odd
theorem problem_is_odd_function_proof : shifted_inverse_is_odd :=
sorry

end problem_is_odd_function_proof_l185_185460


namespace odd_function_g_l185_185519

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185519


namespace no_marriage_for_deceased_l185_185971

theorem no_marriage_for_deceased (C : Type) (is_widow : ∀ (widow : C), ∃ (husband : C), deceased(husband)) :
  ∀ (c : C), deceased(c) → ¬ ∃ (sister : C), can_marry(c, sister) :=
by
  sorry

end no_marriage_for_deceased_l185_185971


namespace is_odd_g_l185_185466

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the candidate function that should be proved to be odd
def g (x : ℝ) : ℝ := f(x - 1) + 1

-- Prove that g(x) is an odd function, i.e., g(-x) = -g(x)
theorem is_odd_g : ∀ x : ℝ, g (-x) = -g x := by
  -- Proof is not provided, and we use sorry to indicate this
  sorry

end is_odd_g_l185_185466


namespace option_B_is_odd_function_l185_185310

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185310


namespace odd_function_result_l185_185262

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185262


namespace g_is_odd_l185_185234

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := 
by
  sorry

end g_is_odd_l185_185234


namespace option_B_is_odd_function_l185_185307

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g (x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_B_is_odd_function : is_odd_function g :=
by
  sorry

end option_B_is_odd_function_l185_185307


namespace al_candies_on_sixth_day_l185_185951

theorem al_candies_on_sixth_day (a : ℕ) (h1 : ∑ i in Finset.range 7, (a + i * 4) = 140) :
  (a + 5 * 4) = 28 :=
by
  sorry

end al_candies_on_sixth_day_l185_185951


namespace odd_function_shifted_f_l185_185376

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def shifted_f (x : ℝ) : ℝ := f(x-1) + 1

theorem odd_function_shifted_f : ∀ x : ℝ, shifted_f(x) = -shifted_f(-x) :=
by sorry

end odd_function_shifted_f_l185_185376


namespace odd_function_g_l185_185541

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)
def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x, g(-x) = -g(x) :=
by
  sorry

end odd_function_g_l185_185541


namespace odd_function_result_l185_185263

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem odd_function_result (x : ℝ) : 
  (λ x, f (x - 1) + 1) (-x) = -((λ x, f (x - 1) + 1) x) :=
by
  sorry

end odd_function_result_l185_185263


namespace bert_ernie_ratio_l185_185964

theorem bert_ernie_ratio (berts_stamps ernies_stamps peggys_stamps : ℕ) 
  (h1 : peggys_stamps = 75) 
  (h2 : ernies_stamps = 3 * peggys_stamps) 
  (h3 : berts_stamps = peggys_stamps + 825) : 
  berts_stamps / ernies_stamps = 4 := 
by sorry

end bert_ernie_ratio_l185_185964


namespace perpendicular_vectors_l185_185694

noncomputable section

open Real

def a (θ : ℝ) : ℝ × ℝ := (1, cos θ)
def b (θ : ℝ) : ℝ × ℝ := (-1, 2 * cos θ)

theorem perpendicular_vectors (θ : ℝ) (hθ : θ ∈ Ioo 0 (π / 2)) (h : a θ.1 * b θ.1 + a θ.2 * b θ.2 = 0) :
  θ = π / 4 :=
by
  sorry

end perpendicular_vectors_l185_185694


namespace max_distance_boating_l185_185901

theorem max_distance_boating (v_b v_c : ℝ) (total_time minutes_in_hour : ℝ) 
(cycles_time paddling_time drifting_time : ℝ) (time_ratio : ℝ) 
(d_total d_paddled d_drifted : ℝ) :
  v_b = 3 → 
  v_c = 1.5 → 
  total_time = 120 → 
  minutes_in_hour = 60 → 
  cycles_time = 40 →
  paddling_time = 90 →
  drifting_time = 30 →
  time_ratio = 0.75 →
  d_paddled = 4.5 * (45 / 60) →
  d_drifted = 1.5 * (10 / 60) →
  d_total = d_paddled + d_drifted →
  d_total = 1.375 :=
begin
  sorry
end

end max_distance_boating_l185_185901


namespace g_is_odd_l185_185346

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185346


namespace number_of_positive_integers_l185_185717

noncomputable def condition (x : ℕ) : Prop :=
  let frac := (6 * x + 12) / (x^2 - x - 6)
  frac = frac.floor

theorem number_of_positive_integers (n : ℕ) : n = 6 ↔ 
  ∃ (l : list ℕ), (∀ x ∈ l, condition x) ∧ l.length = n ∧ (∀ x ∈ l, x > 0) :=
sorry

end number_of_positive_integers_l185_185717


namespace g_B_is_odd_l185_185658

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g_B (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_B_is_odd : ∀ x : ℝ, g_B (-x) = -g_B (x) := by
  sorry

end g_B_is_odd_l185_185658


namespace odd_function_g_l185_185203

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185203


namespace compare_values_l185_185770

theorem compare_values :
  let a := 3 ^ 0.4
  let b := Real.log 0.3 / Real.log 4 -- log4(0.3) can be written as ln(0.3)/ln(4)
  let c := Real.log 0.4 / Real.log 0.3 -- log0.3(0.4) can be written as ln(0.4)/ln(0.3)
  a > c ∧ c > b :=
by {
  let a := 3 ^ 0.4,
  let b := Real.log 0.3 / Real.log 4,
  let c := Real.log 0.4 / Real.log 0.3,
  sorry
}

end compare_values_l185_185770


namespace option_b_is_odd_l185_185118

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def option_b (x : ℝ) : ℝ := f (x - 1) + 1

theorem option_b_is_odd : ∀ x : ℝ, option_b (-x) = - option_b x :=
by
  sorry

end option_b_is_odd_l185_185118


namespace g_is_odd_l185_185038

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g (x) := by
  sorry

end g_is_odd_l185_185038


namespace quadratic_function_equation_l185_185695

theorem quadratic_function_equation
  (h k : ℝ)
  (vertex : h = 2 ∧ k = -1)
  (pass_point : ∀ a, 8 = (a * ((-1 : ℝ) - h) ^ 2 + k)) :
  ∃ a : ℝ, ∀ x, (h = 2) ∧ (k = -1) → 
            pass_point a → 
            (a = 1) ∧ (a * (x - 2)^2 - 1 = x^2 - 4 * x + 3) :=
by
  sorry

end quadratic_function_equation_l185_185695


namespace odd_function_g_l185_185190

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem odd_function_g : ∀ x : ℝ, g (-x) = -g (x) :=
by
  sorry

end odd_function_g_l185_185190
