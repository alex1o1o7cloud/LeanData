import Mathlib

namespace value_for_real_value_for_pure_imaginary_l98_98258

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def value_conditions (k : ℝ) : ℂ := ⟨k^2 - 3*k - 4, k^2 - 5*k - 6⟩

theorem value_for_real (k : ℝ) : is_real (value_conditions k) ↔ (k = 6 ∨ k = -1) :=
by
  sorry

theorem value_for_pure_imaginary (k : ℝ) : is_pure_imaginary (value_conditions k) ↔ (k = 4) :=
by
  sorry

end value_for_real_value_for_pure_imaginary_l98_98258


namespace sin_ratio_in_triangle_l98_98414

theorem sin_ratio_in_triangle
  (A B C D : Type)
  (BC : ℝ)
  (angle_B : Real.Angle := Real.Angle.pi / 4)
  (angle_C : Real.Angle := Real.Angle.pi / 3)
  (BD CD : ℝ)
  (h1 : BD = (2 / 5) * BC)
  (h2 : CD = (3 / 5) * BC)
  (BAD CAD : Real.Angle)
  (h3 : Real.sin angle_B = Real.sin (angle_B.to_real))
  (h4 : Real.sin BAD = BD / ((Real.sin 45°.to_radians) * D)) -- sine of BAD
  (h5 : Real.sin CAD = CD / ((Real.sin 60°.to_radians) * D) -- sine of CAD
  :
  (Real.sin BAD / Real.sin CAD = Real.sqrt 6 / 3) :=
sorry

end sin_ratio_in_triangle_l98_98414


namespace functional_equation_solution_l98_98667

-- Define the function type f and its properties
def f (x : ℝ) : ℝ := sorry 

-- The main theorem to prove 
theorem functional_equation_solution (f : ℝ → ℝ) : 
  (∀ x y, f (x * y) + f (x + y) = f x * f y + f x + f y) →
  (f = (λ x, 0) ∨ f = id) :=
by
  -- Insert proof here
  sorry

end functional_equation_solution_l98_98667


namespace distance_B_squared_center_l98_98642

-- Definitions based on conditions
def radius := Real.sqrt 52
def AB := 8
def BC := 4
def angle_ABC := Real.pi / 2  -- right angle in radians

-- Proven fact about the distance squared from point B to the center of the notched circle
theorem distance_B_squared_center (a b : ℝ)
  (h₁ : a^2 + (b + 8)^2 = 52)
  (h₂ : (a + 4)^2 + b^2 = 52) :
  a^2 + b^2 = 20 :=
by
  sorry

end distance_B_squared_center_l98_98642


namespace tourist_food_preparation_l98_98922

def periodic_tourist_count (x : ℕ) : ℝ := 200 * Real.sin (Real.pi / 6 * x - 5 * Real.pi / 6) + 300

theorem tourist_food_preparation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 12) : 
  periodic_tourist_count x ≥ 400 ↔ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10 :=
by sorry

end tourist_food_preparation_l98_98922


namespace distance_between_stations_l98_98579

/-- Two trains start at the same time from two stations and proceed towards each other.
    Train 1 travels at 20 km/hr.
    Train 2 travels at 25 km/hr.
    When they meet, Train 2 has traveled 55 km more than Train 1.
    Prove that the distance between the two stations is 495 km. -/
theorem distance_between_stations : ∃ x t : ℕ, 20 * t = x ∧ 25 * t = x + 55 ∧ 2 * x + 55 = 495 :=
by {
  sorry
}

end distance_between_stations_l98_98579


namespace gain_correct_l98_98609

-- Define the given conditions
def borrowed_amount : Float := 9000
def borrowed_rate : Float := 0.04
def borrowed_time : Float := 2
def borrow_compounds_per_year : Float := 4

def lent_amount : Float := 9000
def lent_rate : Float := 0.06
def lent_time : Float := 2
def lend_compounds_per_year : Float := 2

-- Define the formula for compound interest
noncomputable def compound_interest (P r t n : Float) : Float :=
  P * (1 + r / n)^(n * t)

-- Calculate the borrowed and lent amounts
noncomputable def total_borrowed_amount : Float :=
  compound_interest borrowed_amount borrowed_rate borrowed_time borrow_compounds_per_year

noncomputable def total_lent_amount : Float :=
  compound_interest lent_amount lent_rate lent_time lend_compounds_per_year

-- Calculate the gain
def total_gain : Float :=
  total_lent_amount - total_borrowed_amount

def annual_gain : Float :=
  total_gain / 2

-- Define the target gain
def target_gain : Float := 192.49

-- Prove the gain is equal to the target gain
theorem gain_correct :
  annual_gain = target_gain :=
by
  sorry

end gain_correct_l98_98609


namespace find_parallel_line_l98_98224

def line1 : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y + 2 = 0
def line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y + 2 = 0
def parallelLine : ℝ → ℝ → Prop := λ x y => 4 * x + y - 4 = 0

theorem find_parallel_line (x y : ℝ) (hx : line1 x y) (hy : line2 x y) : 
  ∃ c : ℝ, (λ x y => 4 * x + y + c = 0) (2:ℝ) (2:ℝ) ∧ 
          ∀ x' y', (λ x' y' => 4 * x' + y' + c = 0) x' y' ↔ 4 * x' + y' - 10 = 0 := 
sorry

end find_parallel_line_l98_98224


namespace two_points_on_AD_l98_98324

theorem two_points_on_AD (A B C D P : Point) (h_rect : is_rectangle A B C D)
  (h_short : segment_length A B < segment_length B C)
  (h_on_line : collinear A D P)
  (h_bisector : bisects (angle B P D) (line.from_points P C)) :
  ∃! Q R : Point, collinear A D Q ∧ collinear A D R ∧
  bisects (angle B Q D) (line.from_points Q C) ∧
  bisects (angle B R D) (line.from_points R C) := sorry

end two_points_on_AD_l98_98324


namespace even_heads_probability_fair_even_heads_probability_biased_l98_98239

variable  {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1)

/-- The probability of getting an even number of heads -/

/-- Case 1: Fair coin (p = 1/2) -/
theorem even_heads_probability_fair (n : ℕ) : 
  let p : ℝ := 1/2 in 
  let q : ℝ := 1 - p in 
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 0.5 := sorry

/-- Case 2: Biased coin -/
theorem even_heads_probability_biased {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1) : 
  let q : ℝ := 1 - p in
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 
  (1 + (1 - 2 * p)^n) / 2 := sorry

end even_heads_probability_fair_even_heads_probability_biased_l98_98239


namespace bus_stops_for_minutes_per_hour_l98_98665

theorem bus_stops_for_minutes_per_hour (speed_no_stops speed_with_stops : ℕ)
  (h1 : speed_no_stops = 60) (h2 : speed_with_stops = 45) : 
  (60 * (speed_no_stops - speed_with_stops) / speed_no_stops) = 15 :=
by
  sorry

end bus_stops_for_minutes_per_hour_l98_98665


namespace perfect_square_factors_count_l98_98752

def perfectSquares := [4, 9, 16, 25, 36, 49, 64, 81]

def countNumbersWithPerfectSquareFactors : Nat :=
  List.length (List.filter (fun n => perfectSquares.any (fun p => n % p = 0)) [1..100])

theorem perfect_square_factors_count :
  countNumbersWithPerfectSquareFactors = 41 := sorry

end perfect_square_factors_count_l98_98752


namespace perimeter_of_triangle_DEF_l98_98898

variable (D E F P : Type)
variable [line : DE ∧ DF ∧ EF]
variable [tangency : ∀ {P : Type}, tangent_circle P D E]
variables {radius tangent_point1 tangent_point2 : ℝ}
variable [radius_def : radius = 14]
variable [DP_def : tangent_point1 = 19]
variable [PE_def : tangent_point2 = 31]

theorem perimeter_of_triangle_DEF : Perimeter DEF = 300 := by
  sorry

end perimeter_of_triangle_DEF_l98_98898


namespace find_a7_l98_98268

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98268


namespace total_stock_worth_is_15000_l98_98171

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the conditions
def stock_condition_1 := 0.20 * X -- Worth of 20% of the stock
def stock_condition_2 := 0.10 * (0.20 * X) -- Profit from 20% of the stock
def stock_condition_3 := 0.80 * X -- Worth of 80% of the stock
def stock_condition_4 := 0.05 * (0.80 * X) -- Loss from 80% of the stock
def overall_loss := 0.04 * X - 0.02 * X

-- The question rewritten as a theorem statement
theorem total_stock_worth_is_15000 (h1 : overall_loss X = 300) : X = 15000 :=
by sorry

end total_stock_worth_is_15000_l98_98171


namespace katya_notebooks_l98_98173

theorem katya_notebooks (rubles : ℕ) (cost_per_notebook : ℕ) (stickers_per_notebook : ℕ) (sticker_rate : ℕ):
  rubles = 150 -> cost_per_notebook = 4 -> stickers_per_notebook = 1 -> sticker_rate = 5 ->
  let initial_notebooks := rubles / cost_per_notebook in
  let remaining_rubles := rubles % cost_per_notebook in
  let initial_stickers := initial_notebooks * stickers_per_notebook in
  ∃ final_notebooks : ℕ, 
    (initial_notebooks = 37 ∧ remaining_rubles = 2 ∧ initial_stickers = 37 ∧ 
    (final_notebooks = initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate) + (((initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate)) * stickers_per_notebook) / sticker_rate)) = 46) :=
begin
  intros,
  sorry
end

end katya_notebooks_l98_98173


namespace count_numbers_with_perfect_square_factors_l98_98773

open Set

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ m * m ∣ n

theorem count_numbers_with_perfect_square_factors (s : Finset ℕ) (hs : s = Finset.range 101) :
  (Finset.filter has_perfect_square_factor_other_than_one s).card = 41 :=
by {
  sorry
}

end count_numbers_with_perfect_square_factors_l98_98773


namespace ellipse_and_chord_theorem_l98_98701

-- Define variables a, b, c with conditions
variable (a b c : ℝ)

-- Define the conditions
def conditions (a b c : ℝ) (ecc : ℝ) (perimeter : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ ecc = √3 / 3 ∧ perimeter = 4 * √3 ∧ a^2 = b^2 + c^2 ∧ c / a = ecc

-- Define the ellipse equation
def ellipse_equation : Prop :=
  (∀ x y : ℝ, (x^2 / 3) + (y^2 / 2) = 1)

-- Define the chord’s line equation bisected at (1, 1)
def chord_bisect_line : Prop :=
  (∀ x y : ℝ, 2 * x + 3 * y - 5 = 0)

-- Main theorem given the conditions prove ellipse equation and chord bisect line
theorem ellipse_and_chord_theorem (a b c : ℝ) :
  ∀ (ecc : ℝ) (perimeter : ℝ),
  conditions a b c ecc perimeter →
  ellipse_equation ∧ chord_bisect_line :=
begin
  intros,
  sorry
end

end ellipse_and_chord_theorem_l98_98701


namespace group_D_forms_a_set_l98_98134

-- Definitions for the problem's conditions
def definiteness (S : Set α) : Prop := ∀ x ∈ S, ∃ y, x = y
def distinctness (S : Set α) : Prop := ∀ x y ∈ S, x = y → false
def unorderedness (S : Set α) : Prop := true  -- For simplicity, we assume unorderedness always holds

-- Given subsets according to the problem's conditions
def A : Set ℝ := { x | real.abs (x - real.pi) < ε }
def B : Set Person := { p | is_kind p }
def C : Set Student := { s | is_smart s ∧ s ∈ Class1_Grade1_SchoolA }
def D : Set Person := { p | p ∈ UnitB ∧ height p > 1.75 }

-- Main statement
theorem group_D_forms_a_set :
  (definiteness D ∧ distinctness D ∧ unorderedness D) ∧ 
  ¬(definiteness A ∧ distinctness A ∧ unorderedness A) ∧ 
  ¬(definiteness B ∧ distinctness B ∧ unorderedness B) ∧ 
  ¬(definiteness C ∧ distinctness C ∧ unorderedness C) := 
by
  sorry

end group_D_forms_a_set_l98_98134


namespace sum_complex_identity_l98_98436

noncomputable def z : ℂ := (1 / 2) * ((complex.sqrt 2) + complex.I * (complex.sqrt 2))

theorem sum_complex_identity :
  (∑ k in finset.range 14, (1 / (1 - z * complex.exp (k * complex.I * real.pi / 7)))) = (7 - 7 * complex.I) →
  let a := 7,
      b := -7 in 
  a + b = 0 :=
begin
  intro h,
  unfold a b,
  exact add_neg_self 7,
end

end sum_complex_identity_l98_98436


namespace paint_rate_5_l98_98082
noncomputable def rate_per_sq_meter (L : ℝ) (total_cost : ℝ) (B : ℝ) : ℝ :=
  let Area := L * B
  total_cost / Area

theorem paint_rate_5 : 
  ∀ (L B total_cost rate : ℝ),
    L = 19.595917942265423 →
    total_cost = 640 →
    L = 3 * B →
    rate = rate_per_sq_meter L total_cost B →
    rate = 5 :=
by
  intros L B total_cost rate hL hC hR hRate
  -- Proof goes here
  sorry

end paint_rate_5_l98_98082


namespace find_a7_l98_98292

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98292


namespace probability_each_number_appears_once_l98_98933

-- Define the event that each number from 1 to 6 appears at least once when 10 fair dice are rolled
def each_number_appears_once : ℕ → Prop := λ n, n = 10

-- Define the probability calculation
noncomputable def prob_each_number_appears_once :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

-- Prove that the probability is 0.272
theorem probability_each_number_appears_once :
  each_number_appears_once 10 → prob_each_number_appears_once = 0.272 :=
by
  intro h
  sorry

end probability_each_number_appears_once_l98_98933


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98234

-- Define the probability of getting heads as p for the biased coin
variable {p : ℝ} (hp : 0 < p ∧ p < 1)

-- Define the number of tosses
variable {n : ℕ}

-- Fair coin probability of getting an even number of heads
theorem fair_coin_even_heads : 
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * (1 / 2)^n else 0) = 0.5 :=
sorry

-- Biased coin probability of getting an even number of heads
theorem biased_coin_even_heads :
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) else 0) 
  = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98234


namespace find_percentage_l98_98595

theorem find_percentage (P : ℝ) (N : ℝ) (h1 : N = 140) (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end find_percentage_l98_98595


namespace B_finish_work_alone_in_12_days_l98_98594

theorem B_finish_work_alone_in_12_days (A_days B_days both_days : ℕ) :
  A_days = 6 →
  both_days = 4 →
  (1 / A_days + 1 / B_days = 1 / both_days) →
  B_days = 12 :=
by
  intros hA hBoth hRate
  sorry

end B_finish_work_alone_in_12_days_l98_98594


namespace ratio_RN_NS_l98_98423

open Real EuclideanGeometry

-- Define the setup of the problem
def Square : Type :=
{ s : ℝ // s > 0 }

noncomputable def square_WXYZ (s : ℝ) (h : s > 0) : Square := ⟨s, h⟩

noncomputable def point_W (s : Square) : ℝ × ℝ := (0, s.val)
noncomputable def point_X (s : Square) : ℝ × ℝ := (s.val, s.val)
noncomputable def point_Y (s : Square) : ℝ × ℝ := (s.val, 0)
noncomputable def point_Z (s : Square) : ℝ × ℝ := (0, 0)

noncomputable def point_T (s : Square) : ℝ × ℝ := (9, 0)

-- Define the conditions
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

def perpendicular_slope (m : ℝ) : ℝ := -1 / m

def line (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- Define the theorem statement
theorem ratio_RN_NS {s : ℝ} (h : s = 15) :
  let W := (0, 15),
      X := (15, 15),
      Y := (15, 0),
      Z := (0, 0),
      T := (9, 0),
      N := midpoint W T,
      m_WT := slope W T,
      m_perp_bisector := perpendicular_slope m_WT,
      y_intercept := N.2 - m_perp_bisector * N.1 in
  let R := (x : ℝ, (R_witness : (15 - y_intercept) / m_perp_bisector = x)),
      S := (x : ℝ, (S_witness : (0 - y_intercept) / m_perp_bisector = x)),
      RN := 15 - N.2,
      NS := N.2 in
  RN / NS = 1 :=
sorry

end ratio_RN_NS_l98_98423


namespace T_unreachable_from_S_l98_98978

def Position := ℕ × ℕ
def Start : Position := (0, 0)
def P : Position := (-3, -2)
def Q : Position := (3, -2)
def R : Position := (3, 2)
def T : Position := (3, 3)
def W : Position := (-3, 2)

-- Define a move type
inductive Move
| up : Move
| down : Move
| left : Move
| right : Move

open Move

-- Define the valid moves (three places in one direction and two places in a perpendicular direction)
def valid_moves (start : Position) : List Position :=
  [
    (start.1 + 3, start.2 + 2), (start.1 + 3, start.2 - 2),
    (start.1 - 3, start.2 + 2), (start.1 - 3, start.2 - 2),
    (start.1 + 2, start.2 + 3), (start.1 + 2, start.2 - 3),
    (start.1 - 2, start.2 + 3), (start.1 - 2, start.2 - 3)
  ]

-- The proof statement
theorem T_unreachable_from_S : ∀ (S : Position), ¬ (T ∈ (valid_moves S)) :=
by {
  intros,
  cases S with x y,
  sorry
}

end T_unreachable_from_S_l98_98978


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98231

variables (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1)

-- Definition for the fair coin case
def fair_coin_even_heads_probability (n : ℕ) : ℝ :=
  0.5

-- Definition for the biased coin case
def biased_coin_even_heads_probability (n : ℕ) (p : ℝ) : ℝ :=
  (1 + (1 - 2 * p)^n) / 2

-- Theorems/statements to prove
theorem fair_coin_even_heads (n : ℕ) :
  fair_coin_even_heads_probability n = 0.5 :=
sorry

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  biased_coin_even_heads_probability n p = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98231


namespace cos_plus_sin_of_tangent_slope_angle_l98_98090

theorem cos_plus_sin_of_tangent_slope_angle:
  let f : ℝ → ℝ := λ x, log x - 2 / x in
  let f' : ℝ → ℝ := λ x, 1 / x + 2 / (x ^ 2) in
  let alpha := Real.arctan 3 in
  let cos_alpha := Real.cos alpha in
  let sin_alpha := Real.sin alpha in
  (0 < alpha ∧ alpha < Real.pi / 2) →
  cos_alpha ^ 2 + sin_alpha ^ 2 = 1 →
  cos_alpha + sin_alpha = (2 * Real.sqrt 10) / 5 :=
by
  intro h_alpha_range h_cos_sin_squared
  sorry

end cos_plus_sin_of_tangent_slope_angle_l98_98090


namespace proof_AP_eq_AQ_l98_98329

-- Definitions based on the given conditions
def triangle (A B C : Type*) := 
  ∃ (vertices : set Type*), vertices = {A, B, C}

def acute_triangle {A B C : Type*} (t : triangle A B C) : Prop := sorry

def perpendicular_from (P : Type*) (line : set Type*) : Type* := sorry

def foot_of_perpendicular (P : Type*) (line : set Type*) : Type* := sorry

def is_circumcircle_of {A B C : Type*} 
  (circum : set Type*) (t : triangle A B C) : Prop := sorry

def is_intersection_of 
  (line1 line2 : set Type*) : Type* := sorry

def is_cyclic (points : set Type*) : Prop := sorry

theorem proof_AP_eq_AQ
  {A B C D E F P Q : Type*}
  (t : triangle A B C)
  (h_acute : acute_triangle t)
  (hD : D = foot_of_perpendicular B (line AC))
  (hE : E = foot_of_perpendicular C (line AB))
  (hF : F = foot_of_perpendicular A (line BC))
  (hP : P ∈ intersection (line EF) (circumcircle t))
  (hQ : Q ∈ intersection (line BP) (line DF)) :
  AP = AQ :=
sorry

end proof_AP_eq_AQ_l98_98329


namespace solve_arrangement_equation_l98_98958

def arrangement_numeral (x : ℕ) : ℕ :=
  x * (x - 1) * (x - 2)

theorem solve_arrangement_equation (x : ℕ) (h : 3 * (arrangement_numeral x)^3 = 2 * (arrangement_numeral (x + 1))^2 + 6 * (arrangement_numeral x)^2) : x = 5 := 
sorry

end solve_arrangement_equation_l98_98958


namespace arctan_sum_in_right_triangle_l98_98418

noncomputable def triangle_arctan_sum (a b c : ℝ) (h : b^2 + c^2 = a^2) : Prop :=
  arctan (b / (a + c)) + arctan (c / (a + b)) = π / 4

theorem arctan_sum_in_right_triangle
  (a b c : ℝ)
  (h : b^2 + c^2 = a^2) :
  triangle_arctan_sum a b c h :=
sorry

end arctan_sum_in_right_triangle_l98_98418


namespace Wendy_sweaters_l98_98548

theorem Wendy_sweaters (machine_capacity : ℕ) (num_shirts : ℕ) (num_loads : ℕ) : 
    machine_capacity = 8 → 
    num_shirts = 39 → 
    num_loads = 9 → 
    ∃ num_sweaters : ℕ, num_sweaters = 33 ∧ num_loads * machine_capacity = num_shirts + num_sweaters :=
by
  intros h1 h2 h3
  existsi 33
  rw [h1, h2, h3]
  split
  . refl
  . linarith
  -- sorry -- Uncomment this line to blockproof's solution if preferred

end Wendy_sweaters_l98_98548


namespace triangle_perimeter_l98_98676

-- Definitions based on the conditions
variables (a b : ℝ) (h : b < a)

-- Problem statement
theorem triangle_perimeter (h : b < a) : 
  let p := (a^2) / (a - b) in
  (2 * p) = (2 * a^2) / (a - b) := 
by
  sorry

end triangle_perimeter_l98_98676


namespace inequality_for_real_numbers_l98_98257

theorem inequality_for_real_numbers (x y z : ℝ) : 
  - (3 / 2) * (x^2 + y^2 + 2 * z^2) ≤ 3 * x * y + y * z + z * x ∧ 
  3 * x * y + y * z + z * x ≤ (3 + Real.sqrt 13) / 4 * (x^2 + y^2 + 2 * z^2) :=
by
  sorry

end inequality_for_real_numbers_l98_98257


namespace alex_vs_jane_pens_l98_98625

theorem alex_vs_jane_pens:
  let a : ℕ := 4 in
  let j : ℕ := 50 in
  let alex_pens := a * 3 * 3 * 3 in
  alex_pens - j = 58 :=
by
  sorry

end alex_vs_jane_pens_l98_98625


namespace probability_xi_in_A_eq_half_l98_98871

noncomputable def normal_dist (μ : ℝ) (σ : ℝ) := sorry -- Definition of normal distribution here.

lemma no_real_roots_iff (a : ℝ) : (∀ x : ℝ, 2 * x^2 - 4 * x + a ≠ 0) ↔ 16 - 8 * a < 0 :=
sorry

def set_A : set ℝ := {a | 16 - 8 * a < 0}

theorem probability_xi_in_A_eq_half (σ : ℝ) : 
  let xi := normal_dist 2 σ in
  P(xi ∈ set_A) = 1 / 2 :=
sorry

end probability_xi_in_A_eq_half_l98_98871


namespace polynomial_value_l98_98777

theorem polynomial_value (x : ℝ) (h : 3 * x^2 - x = 1) : 6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 :=
by
  sorry

end polynomial_value_l98_98777


namespace value_of_b_l98_98721

theorem value_of_b (a b c y1 y2 y3 : ℝ)
( h1 : y1 = a + b + c )
( h2 : y2 = a - b + c )
( h3 : y3 = 4 * a + 2 * b + c )
( h4 : y1 - y2 = 8 )
( h5 : y3 = y1 + 2 )
: b = 4 :=
sorry

end value_of_b_l98_98721


namespace unpainted_faces_area_is_correct_l98_98045

/-- The edge length of the cube in feet -/
def edge_length : ℝ := 12

/-- The total area of paint available in square feet -/
def paint_area : ℝ := 600

/-- The condition that Marla leaves two adjacent faces untouched -/
def unpainted_faces_count : ℕ := 2

/-- The correct answer for the problem, which is the area of the unpainted faces in square feet -/
def total_unpainted_area : ℝ := 288

/-- Definition of the total surface area of the cube. -/
def total_surface_area (edge: ℝ) : ℝ := 6 * (edge * edge)

/-- Definition of the area of one face of the cube. -/
def face_area (edge: ℝ) : ℝ := edge * edge

/-- Proof problem statement:
    Given the edge length of the cube and the amount of paint available, 
    prove the area of the unpainted faces equals the correct answer. -/
theorem unpainted_faces_area_is_correct :
  total_surface_area edge_length = 864 →
  (paint_area / face_area edge_length).toNat = 4 →
  (unpainted_faces_count * face_area edge_length) = total_unpainted_area :=
by
  sorry

end unpainted_faces_area_is_correct_l98_98045


namespace count_perfect_square_factors_l98_98759

open Finset

noncomputable def count_divisible_by (n : ℕ) (s : Finset ℕ) : ℕ :=
s.filter (λ x, x % n = 0).card

theorem count_perfect_square_factors :
  let S := (range 100).map (λ n, n + 1)
  let perfect_squares := [4, 9, 16, 25, 36, 49, 64, 81, 100] 
  let total := S.card 
  let count_4 := count_divisible_by 4 S
  let count_9 := count_divisible_by 9 S
  let count_16 := count_divisible_by 16 S
  let count_25 := count_divisible_by 25 S
  let count_36 := count_divisible_by 36 S
  let count_49 := count_divisible_by 49 S
  let count_64 := count_divisible_by 64 S
  let count_81 := count_divisible_by 81 S
  let count_100 := count_divisible_by 100 S
  count_4 + (count_9 - count_divisible_by (Nat.lcm 4 9) S) +
    (count_16 - count_divisible_by 4 S) +
    (count_25 - count_divisible_by (Nat.lcm 4 25) S) +
    (count_36 - count_divisible_by 4 S) +
    count_49 + (count_64 - count_divisible_by 4 S) +
    (count_81 - count_divisible_by (Nat.lcm 9 81) S) +
    (count_100 - count_divisible_by 4 S)
    = 40 := 
by
  sorry

end count_perfect_square_factors_l98_98759


namespace part_a_equiv_l98_98946

-- Definitions:
def polynomial (a b c : ℝ) : polynomial ℝ := X^4 - (polynomial.C a) * X^3 - (polynomial.C b) * X + polynomial.C c

-- Given conditions where a, b, c are roots:
variables {a b c : ℝ}
hypothesis (Ha : polynomial a b c).is_root a
hypothesis (Hb : polynomial a b c).is_root b
hypothesis (Hc : polynomial a b c).is_root c

-- The statement to prove:
theorem part_a_equiv (a b c : ℝ) (Ha : polynomial a b c).is_root a
  (Hb : polynomial a b c).is_root b (Hc : polynomial a b c).is_root c :
  polynomial a b c = X^4 - (polynomial.C a) * X^3 :=
sorry

end part_a_equiv_l98_98946


namespace profit_starts_from_third_year_most_beneficial_option_l98_98596

-- Define the conditions of the problem
def investment_cost := 144
def maintenance_cost (n : ℕ) := 4 * n^2 + 20 * n
def revenue_per_year := 1

-- Define the net profit function
def net_profit (n : ℕ) : ℤ :=
(revenue_per_year * n : ℤ) - (maintenance_cost n) - investment_cost

-- Question 1: Prove the project starts to make a profit from the 3rd year
theorem profit_starts_from_third_year (n : ℕ) (h : 2 < n ∧ n < 18) : 
net_profit n > 0 ↔ 3 ≤ n := sorry

-- Question 2: Prove the most beneficial option for company's development
theorem most_beneficial_option : (∃ o, o = 1) ∧ (∃ t1 t2, t1 = 264 ∧ t2 = 264 ∧ t1 < t2) := sorry

end profit_starts_from_third_year_most_beneficial_option_l98_98596


namespace find_a7_l98_98281

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98281


namespace area_of_roof_l98_98907

-- Definitions and conditions
def length (w : ℝ) := 4 * w
def difference_eq (l w : ℝ) := l - w = 39
def area (l w : ℝ) := l * w

-- Theorem statement
theorem area_of_roof (w l : ℝ) (h_length : l = length w) (h_diff : difference_eq l w) : area l w = 676 :=
by
  sorry

end area_of_roof_l98_98907


namespace john_rejection_percentage_l98_98027

variables (P : ℝ) (J : ℝ)
-- Jane inspected 50% of the products and rejected 0.8% of them.
def jane_rejected := 0.5 * P * 0.008
-- A total of 0.75% of the products produced last month were rejected.
def total_rejected := 0.0075 * P
-- John rejected J * 0.5 * P of the products.
def john_rejected := J * 0.5 * P

theorem john_rejection_percentage :
  jane_rejected + john_rejected = total_rejected → J = 0.007 :=
by
  sorry

end john_rejection_percentage_l98_98027


namespace number_of_integer_pairs_l98_98444

-- Let ω be a nonreal root of z^4 = 1
def ω : ℂ := Complex.I -- Complex.I represents the imaginary unit i in Lean

-- Define the condition that |a * ω + b| = 1
def magnitude_condition (a b : ℤ) : Prop := 
  abs (a * ω + b) = 1

-- Define the statement that we want to prove
theorem number_of_integer_pairs :
  {p : ℤ × ℤ // magnitude_condition p.1 p.2}.to_finset.card = 4 :=
by
  sorry

end number_of_integer_pairs_l98_98444


namespace probability_in_interval_l98_98983

theorem probability_in_interval (a b c d : ℝ) (h1 : a = 2) (h2 : b = 10) (h3 : c = 5) (h4 : d = 7) :
  (d - c) / (b - a) = 1 / 4 :=
by
  sorry

end probability_in_interval_l98_98983


namespace vector_combination_lambda_mu_is_one_l98_98357

-- Define the complex numbers
def z1 : ℂ := -1 + 2 * complex.I
def z2 : ℂ := 1 - complex.I
def z3 : ℂ := 3 - 4 * complex.I

-- Define the points corresponding to the complex numbers
def A : ℂ := z1
def B : ℂ := z2
def C : ℂ := z3

-- Define the parameters l and μ
variables (λ μ : ℝ)

-- Define the condition for the vector equality
def vector_eq : Prop := C = λ * A + μ * B

-- Introduce the condition that leads to the system of equations
def system_of_equations : Prop :=
  (-λ + μ = 3) ∧ (2 * λ - μ = -4)

-- The final statement to be proved
theorem vector_combination_lambda_mu_is_one (h1 : vector_eq) (h2 : system_of_equations) : 
  λ + μ = 1 := by 
  sorry

end vector_combination_lambda_mu_is_one_l98_98357


namespace part1_part2_l98_98044

open Real

variable (θ : ℝ)

def a : ℝ × ℝ := (1, cos (2 * θ))
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (4 * sin θ, 1)
def d : ℝ × ℝ := ((1 / 2) * sin θ, 1)
def f (x : ℝ) : ℝ := abs (x - 1)

theorem part1 (hθ : 0 < θ ∧ θ < π / 4) :
  0 < (a θ).1 * (b.1) + (a θ).2 * (b.2) - ((c θ).1 * (d θ).1 + (c θ).2 * (d θ).2) ∧
  (a θ).1 * (b.1) + (a θ).2 * (b.2) - ((c θ).1 * (d θ).1 + (c θ).2 * (d θ).2) < 2 :=
sorry

theorem part2 (hθ : 0 < θ ∧ θ < π / 4) :
  f ((a θ).1 * (b.1) + (a θ).2 * (b.2)) > f (((c θ).1 * (d θ).1 + (c θ).2 * (d θ).2)) :=
sorry

end part1_part2_l98_98044


namespace part_a_solution_part_b_solution_l98_98575

-- Definitions for Part (a)
def container (t : Type) := ℕ -- Type to represent container
def init_state (A B C : container ℕ) : Prop :=
  A = 3 ∧ B = 20 ∧ C = 0

-- Proposition for Part (a) to prove
theorem part_a_solution (A B C : container ℕ) (syrup water : ℕ) :
  init_state A B C →
  ∃ (X : container ℕ), X = 10 ∧ syrup = 3 ∧ water = 7 :=
sorry

-- Definitions for Part (b)
def init_state_n (A B C : container ℕ) (N : ℕ) : Prop :=
  A = 3 ∧ B = N ∧ C = 0

-- Proposition for Part (b) to prove
theorem part_b_solution (A B C : container ℕ) (N syrup water : ℕ) :
  (∀ k : ℕ, N = 3 * k + 1) →
  init_state A B C →
  ∃ (X : container ℕ), X = 10 ∧ syrup = 3 ∧ water = 7 :=
sorry

end part_a_solution_part_b_solution_l98_98575


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98235

-- Define the probability of getting heads as p for the biased coin
variable {p : ℝ} (hp : 0 < p ∧ p < 1)

-- Define the number of tosses
variable {n : ℕ}

-- Fair coin probability of getting an even number of heads
theorem fair_coin_even_heads : 
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * (1 / 2)^n else 0) = 0.5 :=
sorry

-- Biased coin probability of getting an even number of heads
theorem biased_coin_even_heads :
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) else 0) 
  = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98235


namespace inequality_not_always_true_l98_98715

variables {a b c d : ℝ}

theorem inequality_not_always_true
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0) (h4 : d ≠ 0) :
  ¬ ∀ (a b d : ℝ), (a > b) → (d ≠ 0) → (a + d)^2 > (b + d)^2 :=
by
  intro H
  specialize H a b d h1 h4
  sorry

end inequality_not_always_true_l98_98715


namespace sin_ratio_in_triangle_l98_98415

theorem sin_ratio_in_triangle
  (A B C D : Type)
  (BC : ℝ)
  (angle_B : Real.Angle := Real.Angle.pi / 4)
  (angle_C : Real.Angle := Real.Angle.pi / 3)
  (BD CD : ℝ)
  (h1 : BD = (2 / 5) * BC)
  (h2 : CD = (3 / 5) * BC)
  (BAD CAD : Real.Angle)
  (h3 : Real.sin angle_B = Real.sin (angle_B.to_real))
  (h4 : Real.sin BAD = BD / ((Real.sin 45°.to_radians) * D)) -- sine of BAD
  (h5 : Real.sin CAD = CD / ((Real.sin 60°.to_radians) * D) -- sine of CAD
  :
  (Real.sin BAD / Real.sin CAD = Real.sqrt 6 / 3) :=
sorry

end sin_ratio_in_triangle_l98_98415


namespace count_perfect_square_factors_l98_98760

open Finset

noncomputable def count_divisible_by (n : ℕ) (s : Finset ℕ) : ℕ :=
s.filter (λ x, x % n = 0).card

theorem count_perfect_square_factors :
  let S := (range 100).map (λ n, n + 1)
  let perfect_squares := [4, 9, 16, 25, 36, 49, 64, 81, 100] 
  let total := S.card 
  let count_4 := count_divisible_by 4 S
  let count_9 := count_divisible_by 9 S
  let count_16 := count_divisible_by 16 S
  let count_25 := count_divisible_by 25 S
  let count_36 := count_divisible_by 36 S
  let count_49 := count_divisible_by 49 S
  let count_64 := count_divisible_by 64 S
  let count_81 := count_divisible_by 81 S
  let count_100 := count_divisible_by 100 S
  count_4 + (count_9 - count_divisible_by (Nat.lcm 4 9) S) +
    (count_16 - count_divisible_by 4 S) +
    (count_25 - count_divisible_by (Nat.lcm 4 25) S) +
    (count_36 - count_divisible_by 4 S) +
    count_49 + (count_64 - count_divisible_by 4 S) +
    (count_81 - count_divisible_by (Nat.lcm 9 81) S) +
    (count_100 - count_divisible_by 4 S)
    = 40 := 
by
  sorry

end count_perfect_square_factors_l98_98760


namespace inequality_holds_l98_98683

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x * y + y * z) := 
by 
  sorry

end inequality_holds_l98_98683


namespace problem_equivalent_proof_l98_98313

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98313


namespace count_numbers_with_square_factors_l98_98767

theorem count_numbers_with_square_factors :
  let squares := [4, 9, 16, 25, 36, 49, 64]
  let multiples (n : ℕ) := ∀ k ∈ squares, n % k = 0
  let count_multiples (n : ℕ) := (1..100).count multiples
  count_multiples squares = 48 :=
  sorry

end count_numbers_with_square_factors_l98_98767


namespace range_of_m_l98_98334

variable (m : ℝ)

def p := (0 < m ∧ m < 1 / 3)
def q := (0 < m ∧ m < 15)
def p_or_q := p ∨ q
def p_and_q := p ∧ q

theorem range_of_m : (p_or_q m ∧ ¬p_and_q m) → (1 / 3 ≤ m ∧ m < 15) :=
by
  intros h
  sorry

end range_of_m_l98_98334


namespace base_length_of_isosceles_triangle_l98_98878

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l98_98878


namespace solution1_solution2_l98_98255

noncomputable def condition1 (x y : ℝ) : Prop :=
  (x + y) + (y - 1) * complex.I = (2 * x + 3 * y) + (2 * y + 1) * complex.I

noncomputable def condition2 (x y : ℝ) : Prop :=
  (x + y - 3) + (x - 2) * complex.I = 0

theorem solution1 : ∃ x y : ℝ, condition1 x y ∧ x = 4 ∧ y = -2 :=
by 
  sorry

theorem solution2 : ∃ x y : ℝ, condition2 x y ∧ x = 2 ∧ y = 1 :=
by 
  sorry

end solution1_solution2_l98_98255


namespace count_perfect_square_factors_l98_98764

open Nat

def has_larger_square_factor (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

theorem count_perfect_square_factors :
  (Finset.filter has_larger_square_factor (Finset.range 101)).card = 42 := by
sorry

end count_perfect_square_factors_l98_98764


namespace initial_pennies_in_each_compartment_l98_98867

theorem initial_pennies_in_each_compartment (x : ℕ) (h : 12 * (x + 6) = 96) : x = 2 :=
by sorry

end initial_pennies_in_each_compartment_l98_98867


namespace max_k_four_l98_98507

def is_sum_sequence {α : Type} [Add α] (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n, S n = ∑ i in finset.range n, a i

theorem max_k_four (a : ℕ → ℤ) (S : ℕ → ℤ) (k : ℕ) :
  (∀ n : ℕ, n > 0 → S n ∈ ({2, 3} : set ℤ)) →
  ∃ A : set ℤ, A.finite ∧ A.card = k ∧ (∀ n, a n ∈ A) → k ≤ 4 :=
by
  sorry

end max_k_four_l98_98507


namespace standard_equation_of_ellipse_intersection_points_of_line_and_ellipse_l98_98350

-- Definitions for ellipse parameters and line intersection
def ellipse_through_points (m n : ℝ) (h : m > 0 ∧ n > 0 ∧ m ≠ n) (A B : ℝ × ℝ) : Prop :=
  m * (A.1)^2 + n * (A.2)^2 = 1 ∧ m * (B.1)^2 + n * (B.2)^2 = 1

def line_intersects_ellipse (line : ℝ → ℝ → Prop) (ell : ℝ → ℝ → Prop) (p1 p2 : ℝ × ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, line x₁ y₁ ∧ ell x₁ y₁ ∧ line x₂ y₂ ∧ ell x₂ y₂ ∧ (x₁, y₁) = p1 ∧ (x₂, y₂) = p2

-- Problem statements 

-- Prove the standard equation of ellipse
theorem standard_equation_of_ellipse :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ ellipse_through_points m n (by solve_by_elim) (0, 2) (1/2, sqrt 3) :=
sorry

-- Prove the line intersects the ellipse at given points
theorem intersection_points_of_line_and_ellipse :
  let line := (fun x y => sqrt 3 * y - 2 * x - 2 = 0) in
  let ellipse := (fun x y => x^2 + (y^2) / 4 = 1) in
  line_intersects_ellipse line ellipse (-1, 0) (1/2, sqrt 3) :=
sorry

end standard_equation_of_ellipse_intersection_points_of_line_and_ellipse_l98_98350


namespace correct_division_result_l98_98431

theorem correct_division_result (x : ℝ) 
  (h : (x - 14) / 5 = 11) : (x - 5) / 7 = 64 / 7 :=
by
  sorry

end correct_division_result_l98_98431


namespace find_a7_l98_98265

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98265


namespace range_of_a_l98_98344

theorem range_of_a (a : ℝ) (h_a : a < 0)
  (h_p : ∀ x : ℝ, (x - a) * (x - 3 * a) < 0 → 2^(3 * x + 1) > 2^(-x - 7)) :
  -2/3 ≤ a ∧ a < 0 :=
by
  sorry

end range_of_a_l98_98344


namespace initial_apples_l98_98864

theorem initial_apples (picked: ℕ) (newly_grown: ℕ) (still_on_tree: ℕ) (initial: ℕ):
  (picked = 7) →
  (newly_grown = 2) →
  (still_on_tree = 6) →
  (still_on_tree + picked - newly_grown = initial) →
  initial = 11 :=
by
  intros hpicked hnewly_grown hstill_on_tree hcalculation
  sorry

end initial_apples_l98_98864


namespace distances_identical_l98_98204

-- Define the problem setting
variable {n : ℕ}
variable (red_vertices : Finset (Fin (2 * n))) (blue_vertices : Finset (Fin (2 * n)))

-- Assume that red and blue vertices partition the polygon
axiom red_blue_partition (h_partition : red_vertices ∪ blue_vertices = Finset.univ)
  (h_disjoint : disjoint red_vertices blue_vertices) : red_vertices.card = n ∧ blue_vertices.card = n

-- Define the function to calculate the distances between pairs of vertices
def distances (vertices : Finset (Fin (2 * n))) : Multiset ℕ :=
  vertices.sup (λ x, vertices.filter (λ y, y > x) .map_with (λ y, @fin_sub _ _ y | Fin.dist x y))

-- Define the sequences of distances
def red_distances : Multiset ℕ := distances red_vertices
def blue_distances : Multiset ℕ := distances blue_vertices

theorem distances_identical (h_partition : red_vertices ∪ blue_vertices = Finset.univ)
  (h_disjoint : disjoint red_vertices blue_vertices)
  (h_counts : red_vertices.card = n ∧ blue_vertices.card = n) :
  (red_distances = blue_distances) :=
sorry

end distances_identical_l98_98204


namespace distance_between_circle_center_and_point_is_sqrt50_l98_98928

def circle := (1 : ℝ) -- Placeholder to ensure lean can compile and we can create the structure

def calculateDistance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

-- The circle equation is given
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 3 = 0

-- Point given
def point : ℝ × ℝ := (2, 5)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  calculateDistance p1.1 p1.2 p2.1 p2.2

theorem distance_between_circle_center_and_point_is_sqrt50 :
  ∃ c : ℝ × ℝ, circleEq c.1 c.2 ∧ calculateDistance c.1 c.2 2 5 = real.sqrt 50 :=
by
  sorry

end distance_between_circle_center_and_point_is_sqrt50_l98_98928


namespace radius_area_tripled_l98_98875

theorem radius_area_tripled (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = (n * (Real.sqrt 3 - 1)) / 2 :=
by {
  sorry
}

end radius_area_tripled_l98_98875


namespace platform_length_150_l98_98183

def speed_kmph : ℕ := 54  -- Speed in km/hr

def speed_mps : ℚ := speed_kmph * 1000 / 3600  -- Speed in m/s

def time_pass_man : ℕ := 20  -- Time to pass a man in seconds
def time_pass_platform : ℕ := 30  -- Time to pass a platform in seconds

def length_train : ℚ := speed_mps * time_pass_man  -- Length of the train in meters

def length_platform (P : ℚ) : Prop :=
  length_train + P = speed_mps * time_pass_platform  -- The condition involving platform length

theorem platform_length_150 :
  length_platform 150 := by
  -- We would provide a proof here.
  sorry

end platform_length_150_l98_98183


namespace ratio_uncommon_cards_l98_98826

/-- John buys 10 packs of magic cards. Each pack has 20 cards. He got 50 uncommon cards.
    Prove that the ratio of uncommon cards to the total number of cards in each pack is 5:2. --/
theorem ratio_uncommon_cards (packs : ℕ) (cards_per_pack : ℕ) (uncommon_cards : ℕ) 
  (h_packs : packs = 10) (h_cards_per_pack : cards_per_pack = 20) (h_uncommon_cards : uncommon_cards = 50) :
  (uncommon_cards / 10) : (cards_per_pack / 10) = 5 : 2 :=
by 
  subst h_packs 
  subst h_cards_per_pack 
  subst h_uncommon_cards 
  simp
  exact congr_arg2 (/) rfl rfl 
  sorry

end ratio_uncommon_cards_l98_98826


namespace initial_total_toys_l98_98528

-- Definitions based on the conditions
def initial_red_toys (R : ℕ) : Prop := R - 2 = 88
def twice_as_many_red_toys (R W : ℕ) : Prop := R - 2 = 2 * W

-- The proof statement: show that initially there were 134 toys in the box
theorem initial_total_toys (R W : ℕ) (hR : initial_red_toys R) (hW : twice_as_many_red_toys R W) : R + W = 134 := 
by sorry

end initial_total_toys_l98_98528


namespace geometric_seq_a7_l98_98305

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98305


namespace determinant_of_matrix_l98_98216

variable (α γ : ℝ)

def mat : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [Real.cos α * Real.cos γ, Real.sin α, Real.cos α * Real.sin γ],
    [Real.sin γ, Real.cos γ, 0],
    [-Real.sin α * Real.cos γ, Real.cos α, -Real.sin α * Real.sin γ]
  ]

theorem determinant_of_matrix :
  (Matrix.det (mat α γ)) = Real.sin γ ^ 2 := by
  sorry

end determinant_of_matrix_l98_98216


namespace part_I_part_II_l98_98732

-- Define the function f(x) as per the problem's conditions
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

theorem part_I (x : ℝ) (h₁ : 1 ≠ 0) : 
  (f x 1 > 2) ↔ (x < 1 / 2 ∨ x > 5 / 2) :=
by
  sorry

theorem part_II (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f b a ≥ f a a ∧ (f b a = f a a ↔ ((2 * a - b ≥ 0 ∧ b - a ≥ 0) ∨ (2 * a - b ≤ 0 ∧ b - a ≤ 0) ∨ (2 * a - b = 0) ∨ (b - a = 0))) :=
by
  sorry

end part_I_part_II_l98_98732


namespace diagonal_AC_length_l98_98698

def length_of_diagonal_AC (AD DC AB : ℝ) (h₁ : AD = 6) 
  (h₂ : DC = 26) (h₃ : AB = 15)
  (h₄ : AD ⊥ AB) (h₅ : DC ⊥ AB) : ℝ :=
  sqrt (1090)
  
theorem diagonal_AC_length (AD DC AB : ℝ) (h₁ : AD = 6) 
  (h₂ : DC = 26) (h₃ : AB = 15)
  (h₄ : AD ⊥ AB) (h₅ : DC ⊥ AB) : 
  length_of_diagonal_AC AD DC AB h₁ h₂ h₃ h₄ h₅ = sqrt 1090 :=
sorry

end diagonal_AC_length_l98_98698


namespace expected_digits_fair_icosahedral_die_l98_98849

theorem expected_digits_fair_icosahedral_die :
  let probability_one_digit := 9 / 20
  let probability_two_digits := 10 / 20
  let probability_three_digits := 1 / 20
  let expected_value := (probability_one_digit * 1) + (probability_two_digits * 2) + (probability_three_digits * 3)
  expected_value = 1.6 :=
by
  -- definition of terms
  let probability_one_digit := 9 / 20
  let probability_two_digits := 10 / 20
  let probability_three_digits := 1 / 20
  let expected_value := (probability_one_digit * 1) + (probability_two_digits * 2) + (probability_three_digits * 3)
  -- expected value calculation
  have h : expected_value = (9 / 20 * 1) + (10 / 20 * 2) + (1 / 20 * 3) := rfl
  rw [h]
  have h1 : (9 / 20 * 1) = 9 / 20 := rfl
  have h2 : (10 / 20 * 2) = 1 := by
    calc
      10 / 20 * 2 = (1/2) * 2 := by norm_num
      ... = 1 := by norm_num
  have h3 : (1 / 20 * 3) = 3 / 20 := rfl
  rw [h1, h2, h3]
  have h_sum : (9 / 20) + 1 + (3 / 20) =  1.6 := by
    -- rewrite fraction addition
    calc
      (9 / 20) + 1 + (3 / 20) = (9 / 20) + (20 / 20) + (3 / 20) := by norm_num
      ... = (9 + 20 + 3) / 20 := by ring
      ... = 32 / 20 := rfl
      ... = 1.6 := by norm_num
  rw [h_sum]
  exact rfl

end expected_digits_fair_icosahedral_die_l98_98849


namespace find_a7_l98_98298

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98298


namespace maximum_price_for_360_skewers_price_for_1920_profit_l98_98138

-- Define the number of skewers sold as a function of the price
def skewers_sold (price : ℝ) : ℝ := 300 + 60 * (10 - price)

-- Define the profit as a function of the price
def profit (price : ℝ) : ℝ := (skewers_sold price) * (price - 3)

-- Maximum price for selling at least 360 skewers per day
theorem maximum_price_for_360_skewers (price : ℝ) (h : skewers_sold price ≥ 360) : price ≤ 9 :=
by {
    sorry
}

-- Price to achieve a profit of 1920 yuan per day with price constraint
theorem price_for_1920_profit (price : ℝ) (h₁ : profit price = 1920) (h₂ : price ≤ 8) : price = 7 :=
by {
    sorry
}

end maximum_price_for_360_skewers_price_for_1920_profit_l98_98138


namespace workers_planted_33_walnut_trees_l98_98102

def initial_walnut_trees : ℕ := 22
def total_walnut_trees_after_planting : ℕ := 55
def walnut_trees_planted (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem workers_planted_33_walnut_trees :
  walnut_trees_planted initial_walnut_trees total_walnut_trees_after_planting = 33 :=
by
  unfold walnut_trees_planted
  rfl

end workers_planted_33_walnut_trees_l98_98102


namespace total_cookies_calculation_l98_98943

def cookies_per_bag : ℕ := 11
def number_of_bags : ℕ := 3
def total_cookies : ℕ := 33

theorem total_cookies_calculation : 
  cookies_per_bag * number_of_bags = total_cookies := 
by 
  simp [cookies_per_bag, number_of_bags, total_cookies]
  sorry

end total_cookies_calculation_l98_98943


namespace convert_point_to_polar_l98_98646

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2),
      θ := if y ≠ 0 then real.atan (y / x) else if x > 0 then 0 else real.pi in
  (r, if θ < 0 then θ + 2 * real.pi else θ)

theorem convert_point_to_polar :
  rectangular_to_polar 3 (-3) = (3 * real.sqrt 2, 7 * real.pi / 4) :=
by sorry

end convert_point_to_polar_l98_98646


namespace base_length_of_isosceles_triangle_l98_98882

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l98_98882


namespace median_of_triangle_l98_98863

variable (a b c : ℝ)

noncomputable def AM : ℝ :=
  (Real.sqrt (2 * b * b + 2 * c * c - a * a)) / 2

theorem median_of_triangle :
  abs (((b + c) / 2) - (a / 2)) < AM a b c ∧ 
  AM a b c < (b + c) / 2 := 
by
  sorry

end median_of_triangle_l98_98863


namespace math_problem_solution_l98_98014

noncomputable def problem_statement : Prop :=
  let AB := 4
  let AC := 6
  let BC := 5
  let area_ABC := 9.9216 -- Using the approximated area directly for simplicity
  let K_div3 := area_ABC / 3
  let GP := (2 * K_div3) / BC
  let GQ := (2 * K_div3) / AC
  let GR := (2 * K_div3) / AB
  GP + GQ + GR = 4.08432

theorem math_problem_solution : problem_statement :=
by
  sorry

end math_problem_solution_l98_98014


namespace jackson_saving_l98_98428

theorem jackson_saving (total_amount : ℝ) (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ) :
  total_amount = 3000 → months = 15 → paychecks_per_month = 2 →
  savings_per_paycheck = total_amount / months / paychecks_per_month :=
by sorry

end jackson_saving_l98_98428


namespace find_f3_l98_98155

open Real

noncomputable def is_increasing (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x < y → f(x) < f(y)

theorem find_f3 (f : ℝ → ℝ) (h_increasing : is_increasing f) (h_eq : ∀ x : ℝ, f(f(x) - 3 * x) = 4) :
  f 3 = 10 :=
sorry

end find_f3_l98_98155


namespace reporters_not_covering_politics_l98_98664

theorem reporters_not_covering_politics
  (total_reporters : ℕ)
  (local_politics_reporters : ℕ)
  (non_local_politics_percentage : ℝ) 
  (h1 : local_politics_reporters = 0.3 * total_reporters)
  (h2 : 0.25 * (local_politics_reporters / 0.75) = (local_politics_reporters / 0.75) - local_politics_reporters) :
  ((total_reporters - (local_politics_reporters / 0.75)) / total_reporters) = 0.6 :=
by
  sorry

end reporters_not_covering_politics_l98_98664


namespace distance_between_foci_correct_l98_98256

/-- Define the given conditions for the ellipse -/
def ellipse_center : ℝ × ℝ := (3, -2)
def semi_major_axis : ℝ := 7
def semi_minor_axis : ℝ := 3

/-- Define the distance between the foci of the ellipse -/
noncomputable def distance_between_foci : ℝ :=
  2 * Real.sqrt (semi_major_axis ^ 2 - semi_minor_axis ^ 2)

theorem distance_between_foci_correct :
  distance_between_foci = 4 * Real.sqrt 10 := by
  sorry

end distance_between_foci_correct_l98_98256


namespace bobby_payment_l98_98630

theorem bobby_payment :
  let mold_cost := 250
  let labor_cost_per_hour := 75
  let hours := 8
  let discount := 0.80
  let total_labor_cost := labor_cost_per_hour * hours
  let discounted_labor_cost := discount * total_labor_cost
  let total_payment := mold_cost + discounted_labor_cost
  total_payment = 730 :=
by
  let mold_cost := 250
  let labor_cost_per_hour := 75
  let hours := 8
  let discount := 0.80
  let total_labor_cost := labor_cost_per_hour * hours
  let discounted_labor_cost := discount * total_labor_cost
  let total_payment := mold_cost + discounted_labor_cost
  sorry

end bobby_payment_l98_98630


namespace max_ab_l98_98361

-- Define the function f
def f (x : ℝ) := log (2 - x) + 1

-- Assumptions
axiom graph_passes_point (P : ℝ × ℝ) : P = (1, 1) ∧ P ∈ { (x, f x) | x : ℝ }

-- Line equation definition
def line (a b : ℝ) (P : ℝ × ℝ) : Prop := a * P.1 + b * P.2 = 1

-- Main theorem
theorem max_ab {a b : ℝ} (hline : line a b (1, 1)) : ab = 1 / 4 :=
by
  -- Proof goes here
  sorry

end max_ab_l98_98361


namespace base5_minus_base8_to_base10_l98_98217

def base5_to_base10 (n : Nat) : Nat :=
  5 * 5^5 + 4 * 5^4 + 3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 0 * 5^0

def base8_to_base10 (n : Nat) : Nat :=
  4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0

theorem base5_minus_base8_to_base10 :
  (base5_to_base10 543210 - base8_to_base10 43210) = 499 :=
by
  sorry

end base5_minus_base8_to_base10_l98_98217


namespace find_a7_l98_98277

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98277


namespace total_alligators_seen_l98_98482

theorem total_alligators_seen (samara : ℕ) (friend_avg : ℕ) (friends : ℕ) 
  (h_samara : samara = 20) (h_friend_avg : friend_avg = 10) (h_friends : friends = 3) : 
  samara + friends * friend_avg = 50 := 
by
  -- Initial conditions
  rw [h_samara, h_friend_avg, h_friends]
  -- Evaluate the expression
  simp
  -- Prove the final result
  sorry

end total_alligators_seen_l98_98482


namespace prove_inequality_l98_98514

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end prove_inequality_l98_98514


namespace probability_point_in_spherical_region_l98_98615

theorem probability_point_in_spherical_region :
  let cube_region := {p : ℝ × ℝ × ℝ | (-2 : ℝ) ≤ p.1 ∧ p.1 ≤ 2 ∧ (-2 : ℝ) ≤ p.2 ∧ p.2 ≤ 2 ∧ (-2 : ℝ) ≤ p.3 ∧ p.3 ≤ 2}
  let sphere_region := {p : ℝ × ℝ × ℝ | p.1^2 + p.2^2 + p.3^2 ≤ 4}
  P := ((measure_theory.volume sphere_region) / (measure_theory.volume cube_region))
  in P = (Real.pi / 6) :=
sorry

end probability_point_in_spherical_region_l98_98615


namespace cyclic_quadrilateral_l98_98191

-- Definitions of the points and segments
variables {A B C D E F P Q X Y : Type}
variables (AB AD CB CD AE EP AF FQ : A → ℝ)
variables (angle_ABC : A → ℝ)

-- Conditions of the problem
def quadrilateral_ABCD (A B C D : Type) (AB AD CB CD : A → ℝ) (angle_ABC : A → ℝ) :=
  AB = AD ∧ CB = CD ∧ angle_ABC A B C = 90

def points_on_segments (A B C D E F P Q X Y : Type) (AB AD : A → ℝ) :=
  E ∈ AB ∧ F ∈ AD ∧ P ∈ EF ∧ Q ∈ EF ∧ P ≠ Q ∧ X ∈ CP ∧ Y ∈ CQ

def ratio_condition (AE EP AF FQ : A → ℝ) :=
  AE / EP = AF / FQ

def perpendicular_condition (B C P X D Q Y : Type) :=
  BX ⊥ CP ∧ DY ⊥ CQ

-- Lean statement for the proof problem
theorem cyclic_quadrilateral
  (A B C D E F P Q X Y : Type)
  [H_ABC : quadrilateral_ABCD A B C D AB AD CB CD angle_ABC]
  [H_segments : points_on_segments A B C D E F P Q X Y AB AD]
  [H_ratio : ratio_condition AE EP AF FQ]
  [H_perpendicular : perpendicular_condition B C P X D Q Y] :
  cyclic_quad X P Q Y :=
sorry

end cyclic_quadrilateral_l98_98191


namespace series_sum_1997_l98_98200

def series_sum (n : ℕ) : ℤ :=
  let t (i : ℕ) : ℤ := if i % 5 = 1 then 1 else if i % 5 = 2 then -2 else if i % 5 = 3 then -3 else if i % 5 = 4 then 4 else 5
  in (Finset.range n).sum t

theorem series_sum_1997 : series_sum 1997 = 399002 := sorry

end series_sum_1997_l98_98200


namespace selling_price_of_cycle_l98_98142

theorem selling_price_of_cycle (cost_price : ℕ) (loss_percent : ℕ) (selling_price : ℕ) :
  cost_price = 1400 → loss_percent = 25 → selling_price = 1050 := by
  sorry

end selling_price_of_cycle_l98_98142


namespace fair_coin_even_heads_unfair_coin_even_heads_l98_98248

-- Define the probability function for an even number of heads for a fair coin
theorem fair_coin_even_heads (n : ℕ) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * 0.5^k * 0.5^(n-k) else 0) = 0.5 :=
sorry

-- Define the probability function for an even number of heads for an unfair coin
theorem unfair_coin_even_heads (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * p^k * (1-p)^(n-k) else 0) = 
    (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_unfair_coin_even_heads_l98_98248


namespace katya_total_notebooks_l98_98175

-- Definitions based on the conditions provided
def cost_per_notebook : ℕ := 4
def total_rubles : ℕ := 150
def stickers_for_free_notebook : ℕ := 5
def initial_stickers : ℕ := total_rubles / cost_per_notebook

-- Hypothesis stating the total notebooks Katya can obtain
theorem katya_total_notebooks : initial_stickers + (initial_stickers / stickers_for_free_notebook) + 
    ((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) / stickers_for_free_notebook) +
    (((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) % stickers_for_free_notebook + 1) / stickers_for_free_notebook) = 46 :=
by
  sorry

end katya_total_notebooks_l98_98175


namespace part_I_monotonic_interval_part_I_center_of_symmetry_part_II_min_value_of_m_l98_98367

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - 2 * (sin x)^2 + 2

theorem part_I_monotonic_interval (k : ℤ) :
  ∃ a b : ℝ, (a = -π / 3 + k * π) ∧ (b = π / 6 + k * π) ∧
    ∀ x, a ≤ x → x ≤ b → ∀ y, a ≤ y → y ≤ b → x < y → f x ≤ f y := sorry

theorem part_I_center_of_symmetry (k : ℤ) :
  ∃ x_center : ℝ, (x_center = k * π / 2 - π / 12) ∧
    ∀ x, f (x_center + x) = 2 * sin (2 * x + π / 6) + 1 - f (x_center - x) := sorry

noncomputable def g (x m : ℝ) : ℝ := 2 * sin (2 * (x + m) + π / 6) + 1

theorem part_II_min_value_of_m (m : ℝ) :
  (∃ k : ℤ, (m = k * π / 2 + π / 6) ∧ k * π / 2 + π / 6 > 0) ∧
  ∀ n : ℤ, (n * π / 2 + π / 6 > 0) → (n * π / 2 + π / 6 ≥ m) :=
  sorry

end part_I_monotonic_interval_part_I_center_of_symmetry_part_II_min_value_of_m_l98_98367


namespace votes_cast_l98_98944

theorem votes_cast (candidate_percentage : ℝ) (vote_difference : ℝ) (total_votes : ℝ) 
  (h1 : candidate_percentage = 0.30) 
  (h2 : vote_difference = 1760) 
  (h3 : total_votes = vote_difference / (1 - 2 * candidate_percentage)) 
  : total_votes = 4400 := by
  sorry

end votes_cast_l98_98944


namespace tied_part_length_l98_98530

theorem tied_part_length (length_of_each_string : ℕ) (num_strings : ℕ) (total_tied_length : ℕ) 
  (H1 : length_of_each_string = 217) (H2 : num_strings = 3) (H3 : total_tied_length = 627) : 
  (length_of_each_string * num_strings - total_tied_length) / (num_strings - 1) = 12 :=
by
  sorry

end tied_part_length_l98_98530


namespace find_angle_BPC_l98_98019

-- Define the problem conditions
variables (A B C D P : Type) [IsoscelesTriangle ABC A C B]
variables (on_side_D : OnSegment BC D) (BD_eq_DC : BD = DC)
variables (on_AD_P : OnSegment AD P) (AP_eq_PD : AP = PD)
variables (angle_APB : HasAngle APB 100)

-- Define the statement that we need to prove
theorem find_angle_BPC : angle BPC = 160 := 
sorry

end find_angle_BPC_l98_98019


namespace laura_park_time_l98_98435

theorem laura_park_time
  (T : ℝ) -- Time spent at the park each trip in hours
  (walk_time : ℝ := 0.5) -- Time spent walking to and from the park each trip in hours
  (trips : ℕ := 6) -- Total number of trips
  (park_time_percentage : ℝ := 0.80) -- Percentage of total time spent at the park
  (total_park_time_eq : trips * T = park_time_percentage * (trips * (T + walk_time))) :
  T = 2 :=
by
  sorry

end laura_park_time_l98_98435


namespace katya_notebooks_l98_98180

theorem katya_notebooks (rubles: ℕ) (cost_per_notebook: ℕ) (stickers_per_exchange: ℕ) 
  (initial_rubles: ℕ) (initial_notebooks: ℕ) :
  (initial_notebooks = initial_rubles / cost_per_notebook) →
  (rubles = initial_notebooks * cost_per_notebook) →
  (initial_notebooks = 37) →
  (initial_rubles = 150) →
  (cost_per_notebook = 4) →
  (stickers_per_exchange = 5) →
  (rubles = 148) →
  let rec total_notebooks (notebooks stickers : ℕ) : ℕ :=
      if stickers < stickers_per_exchange then notebooks
      else let new_notebooks := stickers / stickers_per_exchange in
           total_notebooks (notebooks + new_notebooks) 
                           (stickers % stickers_per_exchange + new_notebooks) in
  total_notebooks initial_notebooks initial_notebooks = 46 :=
begin
  sorry
end

end katya_notebooks_l98_98180


namespace find_a7_l98_98286

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98286


namespace geo_seq_bn_plus_2_general_formula_an_l98_98066

variable {a : ℕ → ℕ}
variable {b : ℕ → ℕ}

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 2 = 4
axiom h3 : ∀ n, b n = a (n + 1) - a n
axiom h4 : ∀ n, b (n + 1) = 2 * b n + 2

-- Proof goals
theorem geo_seq_bn_plus_2 : (∀ n, ∃ r : ℕ, b n + 2 = 4 * 2^n) :=
  sorry

theorem general_formula_an : (∀ n, a n = 2^(n + 1) - 2 * n) :=
  sorry

end geo_seq_bn_plus_2_general_formula_an_l98_98066


namespace quadratic_roots_k_relation_l98_98645

theorem quadratic_roots_k_relation (k a b k1 k2 : ℝ) 
    (h_eq : k * (a^2 - a) + 2 * a + 7 = 0)
    (h_eq_b : k * (b^2 - b) + 2 * b + 7 = 0)
    (h_ratio : a / b + b / a = 3)
    (h_k : k = k1 ∨ k = k2)
    (h_vieta_sum : k1 + k2 = 39)
    (h_vieta_product : k1 * k2 = 4) :
    k1 / k2 + k2 / k1 = 1513 / 4 := 
    sorry

end quadratic_roots_k_relation_l98_98645


namespace average_age_when_youngest_born_l98_98916

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_youngest_age total_age_when_youngest_born : ℝ) 
  (h1 : n = 7) (h2 : avg_age = 30) (h3 : current_youngest_age = 8) (h4 : total_age_when_youngest_born = (n * avg_age - n * current_youngest_age)) : 
  total_age_when_youngest_born / n = 22 :=
by
  sorry

end average_age_when_youngest_born_l98_98916


namespace isabela_total_spent_l98_98821

def initial_price := 20
def pencil_discount := 0.20
def notebook_discount := 0.30
def pencil_tax := 0.05
def cucumber_tax := 0.10

def num_cucumbers := 100
def num_notebooks := 25

def final_pencil_price := initial_price * (1 - pencil_discount) * (1 + pencil_tax)
def final_cucumber_price := initial_price * (1 + cucumber_tax)
def final_notebook_price := initial_price * (1 - notebook_discount)

def num_pencils := num_cucumbers / 2

def total_spent :=
  (num_pencils * final_pencil_price) +
  (num_cucumbers * final_cucumber_price) +
  (num_notebooks * final_notebook_price)

theorem isabela_total_spent : total_spent = 3390 := by
  sorry

end isabela_total_spent_l98_98821


namespace max_value_f_when_a_minus_one_value_of_a_given_max_value_no_real_solutions_for_f_eq_g_when_a_minus_one_l98_98737

-- Question 1
theorem max_value_f_when_a_minus_one :
  ∀ x : ℝ, 0 < x → (∀ y : ℝ, 0 < y → f (-1) y ≤ f (-1) x) ↔ f (-1) 1 = -1 := sorry

-- Question 2
theorem value_of_a_given_max_value :
  (∀ x : ℝ, 0 < x ∧ x ≤ Real.exp 1 → (∀ y : ℝ, 0 < y ∧ y ≤ Real.exp 1 → f a y ≤ f a x) ↔ f a (Real.exp (-2)) = -3) →
  a = -Real.exp 2 := sorry

-- Question 3
theorem no_real_solutions_for_f_eq_g_when_a_minus_one :
  ¬ ∃ x : ℝ, 0 < x ∧ |f (-1) x| = (ln x) / x + 1 / 2 := sorry

end max_value_f_when_a_minus_one_value_of_a_given_max_value_no_real_solutions_for_f_eq_g_when_a_minus_one_l98_98737


namespace rectangular_to_polar_l98_98650

theorem rectangular_to_polar : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := 
by
  sorry

end rectangular_to_polar_l98_98650


namespace babysitter_worked_52_hours_l98_98159

noncomputable def babysitter_hours (total_earnings adjusted_earnings : ℕ)
    (regular_rate : ℕ)
    (bonus per_bonus late penalty_rate : ℕ)
    (overtime1 over_time_rate1 : ℕ) (overtime2 over_time_rate2 : ℕ) (overtime3 over_time_rate3 : ℕ)
    (total_hours : ℕ) : Prop :=
  ∃ (x : ℕ), total_hours = x ∧
    total_earnings - (penalty_rate * late) + (per_bonus * bonus) = adjusted_earnings ∧
    ((x ≤ 30 ∧ adjusted_earnings = regular_rate * x) ∨
    (30 < x ∧ x ≤ 40 ∧ adjusted_earnings = regular_rate * 30 + over_time_rate1 * (x - 30)) ∨
    (40 < x ∧ x ≤ 50 ∧ adjusted_earnings = regular_rate * 30 + over_time_rate1 * 10 + over_time_rate2 * (x - 40)) ∨
    (x > 50 ∧ adjusted_earnings = regular_rate * 30 + over_time_rate1 * 10 + over_time_rate2 * 10 + over_time_rate3 * (x - 50)))

theorem babysitter_worked_52_hours : babysitter_hours
  1150 -- total earnings
  1160 -- adjusted earnings due to penalties and bonuses
  16 -- regular_rate
  8 -- number of bonus tasks
  5 -- per_bonus amount for each extra task
  3 -- number of late arrivals
  10 -- penalty_rate for each late arrival
  28 -- overtime1 (31-40 hours) rate (calculated as 75% higher than regular_rate)
  32 -- overtime2 (41-50 hours) rate (calculated as 100% higher than regular_rate)
  40 -- overtime3 (over 50 hours) rate (calculated as 150% higher than regular_rate)
  52 -- total hours worked that we need to prove
:= sorry

end babysitter_worked_52_hours_l98_98159


namespace distance_is_3_l98_98464

-- define the distance between Masha's and Misha's homes
def distance_between_homes (d : ℝ) : Prop :=
  -- Masha and Misha meet 1 kilometer from Masha's home in the first occasion
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / v_m = (d - 1) / v_i) ∧

  -- On the second occasion, Masha walked at twice her original speed,
  -- and Misha walked at half his original speed, and they met 1 kilometer away from Misha's home.
  (∃ v_m v_i : ℝ, v_m > 0 ∧ v_i > 0 ∧
  1 / (2 * v_m) = 2 * (d - 1) / (0.5 * v_i))

-- The theorem to prove the distance is 3
theorem distance_is_3 : distance_between_homes 3 :=
  sorry

end distance_is_3_l98_98464


namespace y_divides_x_squared_l98_98439

-- Define the conditions and proof problem in Lean 4
theorem y_divides_x_squared (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : ∃ (n : ℕ), n = (x^2 / y) + (y^2 / x)) : y ∣ x^2 :=
by {
  -- Proof steps are skipped
  sorry
}

end y_divides_x_squared_l98_98439


namespace yellow_ball_percentage_l98_98546

theorem yellow_ball_percentage
  (yellow_balls : ℕ)
  (brown_balls : ℕ)
  (blue_balls : ℕ)
  (green_balls : ℕ)
  (total_balls : ℕ := yellow_balls + brown_balls + blue_balls + green_balls)
  (h_yellow : yellow_balls = 75)
  (h_brown : brown_balls = 120)
  (h_blue : blue_balls = 45)
  (h_green : green_balls = 60) :
  (yellow_balls * 100) / total_balls = 25 := 
by
  sorry

end yellow_ball_percentage_l98_98546


namespace tetrahedron_projection_area_ratio_l98_98479

noncomputable def projection_area (tetra : Tetrahedron) (plane : Plane) : ℝ := sorry

theorem tetrahedron_projection_area_ratio :
  ∀ (tetra : Tetrahedron), 
  ∃ (plane1 plane2 : Plane), 
  (projection_area tetra plane1 / projection_area tetra plane2) ≥ sqrt 2 :=
sorry

end tetrahedron_projection_area_ratio_l98_98479


namespace integral_of_sqrt_equals_pi_over_4_l98_98904

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, real.sqrt(1 - (x - 1)^2)

theorem integral_of_sqrt_equals_pi_over_4 :
  integral_value = real.pi / 4 :=
by
  sorry

end integral_of_sqrt_equals_pi_over_4_l98_98904


namespace find_a_find_distance_l98_98960

-- Problem 1: Given conditions to find 'a'
theorem find_a (a : ℝ) :
  (∃ θ ρ, ρ = 2 * Real.cos θ ∧ 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0) →
  (a = 2 ∨ a = -8) :=
sorry

-- Problem 2: Given point and line, find the distance
theorem find_distance : 
  ∃ (d : ℝ), d = Real.sqrt 3 + 5/2 ∧
  (∃ θ ρ, θ = 11 * Real.pi / 6 ∧ ρ = 2 ∧ 
   (ρ = Real.sqrt (3 * (Real.sin θ - Real.pi / 6)^2 + (ρ * Real.cos (θ - Real.pi / 6))^2) 
   → ρ * Real.sin (θ - Real.pi / 6) = 1)) :=
sorry

end find_a_find_distance_l98_98960


namespace rectangular_to_polar_l98_98649

theorem rectangular_to_polar : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := 
by
  sorry

end rectangular_to_polar_l98_98649


namespace angle_A_value_sin_sum_range_l98_98328

variables {A B C : ℝ} {a b c : ℝ}

-- (1) Given the conditions, prove that A = π / 3
theorem angle_A_value 
    (h1 : 0 < A ∧ A < π / 2)
    (h2 : 0 < B ∧ B < π / 2)
    (h3 : 0 < C ∧ C < π / 2)
    (h4 : a = b)
    (h5 : c ≠ 0)
    (h_sin : sin(A) * (a^2 + b^2 - c^2) = a * b * (2 * sin(B) - sin(C)))
    : A = π / 3 :=
sorry

-- (2) Given the conditions, find the range of values for sin B + sin C
theorem sin_sum_range 
    (h1 : 0 < A ∧ A < π / 2)
    (h2 : 0 < B ∧ B < π / 2)
    (h3 : 0 < C ∧ C < π / 2)
    (h_A : A = π / 3)
    : ∃ x, sin(B) + sin(C) = x ∧ x ∈ (3 / 2 : ℝ, sqrt 3 ] :=
sorry

end angle_A_value_sin_sum_range_l98_98328


namespace factorial_product_squared_mult_two_l98_98558

theorem factorial_product_squared_mult_two :
  ( (real.sqrt ((nat.factorial 6) * (nat.factorial 4)))^2 * 2 ) = 34560 := by
  sorry

end factorial_product_squared_mult_two_l98_98558


namespace rectangular_to_polar_l98_98651

theorem rectangular_to_polar : 
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (3 * Real.sqrt 2, 7 * Real.pi / 4) := 
by
  sorry

end rectangular_to_polar_l98_98651


namespace conjugate_in_first_quadrant_l98_98356

noncomputable def complex_conjugate_quadrant (z : ℂ) : Set ℕ := 
  if (z.im > 0) && (z.re > 0) then {1}
  else if (z.im > 0) && (z.re < 0) then {2}
  else if (z.im < 0) && (z.re < 0) then {3}
  else if (z.im < 0) && (z.re > 0) then {4}
  else ∅

theorem conjugate_in_first_quadrant : 
  let z := (2 + (complex.i ^ 2016)) / (1 + complex.i)
  let z_conj := conj z
  complex_conjugate_quadrant z_conj = {1} :=
sorry

end conjugate_in_first_quadrant_l98_98356


namespace patrica_earns_more_l98_98025

noncomputable def compounding_annual (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r)^n

noncomputable def compounding_monthly (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r/12)^(n * 12)

theorem patrica_earns_more :
  let P := 30000
  let r := 0.05
  let n := 3
  compounding_monthly P r n - compounding_annual P r n ≈ 121.56 := 
by
  sorry

end patrica_earns_more_l98_98025


namespace dice_surface_dots_l98_98487

def total_dots_on_die := 1 + 2 + 3 + 4 + 5 + 6

def total_dots_on_seven_dice := 7 * total_dots_on_die

def hidden_dots_on_central_die := total_dots_on_die

def visible_dots_on_surface := total_dots_on_seven_dice - hidden_dots_on_central_die

theorem dice_surface_dots : visible_dots_on_surface = 105 := by
  sorry

end dice_surface_dots_l98_98487


namespace sqrt_abc_sum_l98_98841

variable (a b c : ℝ)

theorem sqrt_abc_sum (h1 : b + c = 17) (h2 : c + a = 20) (h3 : a + b = 23) :
  Real.sqrt (a * b * c * (a + b + c)) = 10 * Real.sqrt 273 := by
  sorry

end sqrt_abc_sum_l98_98841


namespace find_omega_l98_98365

noncomputable def f (ω x : ℝ) : ℝ := 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem find_omega (ω : ℝ) (h₁ : ∀ x₁ x₂, (-ω < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * ω) → f ω x₁ < f ω x₂)
  (h₂ : ∀ x, f ω x = f ω (-2 * ω - x)) :
  ω = Real.sqrt (3 * Real.pi) / 3 :=
by
  sorry

end find_omega_l98_98365


namespace isosceles_triangle_base_length_l98_98888

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l98_98888


namespace developed_surface_condition_l98_98028

noncomputable def is_developable (F : Surface) : Prop :=
  ∀ (p : Point F), GaussianCurvature F p = 0

theorem developed_surface_condition (F : Surface)
  (simple_cover_geodesics : ∀ (p : Point F), ∃ (G₁ G₂ : GeodesicSystem F), 
     (G₁.lines ∩ G₂.lines ≠ ∅) ∧ ∀ (g₁ ∈ G₁.lines) (g₂ ∈ G₂.lines), 
       angle_between g₁ g₂ = constant_angle) :
  is_developable F :=
begin
  -- proof is not required
  sorry
end

end developed_surface_condition_l98_98028


namespace biquadratic_equation_from_root_l98_98051

theorem biquadratic_equation_from_root (x : ℂ) (h : x = 2 + Complex.sqrt 3) :
  x^4 - 14 * x^2 + 1 = 0 :=
sorry

end biquadratic_equation_from_root_l98_98051


namespace b_12_is_156_l98_98831

def b : ℕ → ℕ
| 1     := 2
| (m + n) := b m + b n + 2 * m * n

theorem b_12_is_156 : b 12 = 156 :=
by
  sorry

end b_12_is_156_l98_98831


namespace power_function_properties_l98_98353

def power_function (f : ℝ → ℝ) (x : ℝ) (a : ℝ) : Prop :=
  f x = x ^ a

theorem power_function_properties :
  ∃ (f : ℝ → ℝ) (a : ℝ), power_function f 2 a ∧ f 2 = 1/2 ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → 0 < x2 → 
    (f x1 + f x2) / 2 > f ((x1 + x2) / 2)) :=
sorry

end power_function_properties_l98_98353


namespace domain_log_function_l98_98890

theorem domain_log_function : 
  {x : ℝ | -x^2 + 2x + 3 > 0} = set.Ioo (-1 : ℝ) (3 : ℝ) :=
sorry

end domain_log_function_l98_98890


namespace range_of_m_l98_98739

variable (x m : ℝ)

theorem range_of_m (h : (∀ x < 1,  x^2 - 2*m*x + 5) < (x + 1)^2 - 2*m*(x + 1) + 5):
  m ≥ 1 := sorry

end range_of_m_l98_98739


namespace monogram_count_is_correct_l98_98467

def count_possible_monograms : ℕ :=
  Nat.choose 23 2

theorem monogram_count_is_correct : 
  count_possible_monograms = 253 := 
by 
  -- The proof will show this matches the combination formula calculation
  -- The final proof is left incomplete as per the instructions
  sorry

end monogram_count_is_correct_l98_98467


namespace bobby_shoes_cost_l98_98632

theorem bobby_shoes_cost :
  let mold_cost := 250
  let hourly_rate := 75
  let hours_worked := 8
  let discount_rate := 0.20
  let labor_cost := hourly_rate * hours_worked
  let discounted_labor_cost := labor_cost * (1 - discount_rate)
  let total_cost := mold_cost + discounted_labor_cost
  mold_cost = 250 ∧ hourly_rate = 75 ∧ hours_worked = 8 ∧ discount_rate = 0.20 →
  total_cost = 730 := 
by
  sorry

end bobby_shoes_cost_l98_98632


namespace solution_set_inequality_l98_98091

theorem solution_set_inequality (x : ℝ) :
  ((x^2 - 4) * (x - 6)^2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 2 ∨ x = 6) :=
  sorry

end solution_set_inequality_l98_98091


namespace count_numbers_with_square_factors_l98_98769

theorem count_numbers_with_square_factors :
  let squares := [4, 9, 16, 25, 36, 49, 64]
  let multiples (n : ℕ) := ∀ k ∈ squares, n % k = 0
  let count_multiples (n : ℕ) := (1..100).count multiples
  count_multiples squares = 48 :=
  sorry

end count_numbers_with_square_factors_l98_98769


namespace base_length_of_isosceles_triangle_l98_98880

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l98_98880


namespace polynomial_roots_product_l98_98671

theorem polynomial_roots_product (a b : ℤ)
  (h1 : ∀ (r : ℝ), r^2 - r - 2 = 0 → r^3 - a * r - b = 0) : a * b = 6 := sorry

end polynomial_roots_product_l98_98671


namespace digit_sum_properties_l98_98919

theorem digit_sum_properties :
  ∀ (n: ℕ), (∀ x, x ∈ digits n → 1 ≤ x ∧ x ≤ 9) → 
  (∃ digits_A digits_B digits_C : ℕ,
    n.digits ∈ [digits_A.digits.permutations,
                digits_B.digits.permutations,
                digits_C.digits.permutations]) →
  (∀ y, y ∈ digits (n + digits_A + digits_B + digits_C) → y = 1) →
  (∃ x, x ∈ digits n ∧ x ≥ 5) :=
by sorry

end digit_sum_properties_l98_98919


namespace minimum_seats_l98_98100

-- Condition: 150 seats in a row.
def seats : ℕ := 150

-- Assertion: The fewest number of seats that must be occupied so that any additional person seated must sit next to someone.
def minOccupiedSeats : ℕ := 50

theorem minimum_seats (s : ℕ) (m : ℕ) (h_seats : s = 150) (h_min : m = 50) :
  (∀ x, x = 150 → ∀ n, n ≥ 0 ∧ n ≤ m → 
    ∃ y, y ≥ 0 ∧ y ≤ x ∧ ∀ z, z = n + 1 → ∃ w, w ≥ 0 ∧ w ≤ x ∧ w = n ∨ w = n + 1) := 
sorry

end minimum_seats_l98_98100


namespace polygon_has_odd_doubling_point_l98_98196

noncomputable theory
open_locale classical

-- Define a lattice point as a structure with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the condition that checks for a 1993-gon polygon of lattice points.
def is_1993_gon (C : list LatticePoint) : Prop :=
  C.length = 1993

-- Define the condition that no points lie on the sides of the 1993-gon except the vertices.
def sides_no_lattice_points (C : list LatticePoint) : Prop :=
  ∀ (i : ℕ) (H : i < C.length),
    let p := C.nth_le i H, q := C.nth_le ((i + 1) % C.length) sorry in
    ∀ (t : ℝ) (H' : 0 < t ∧ t < 1),
      ¬(∃ (pt : LatticePoint), pt.x = floor (p.x + t * (q.x - p.x)) ∧ pt.y = floor (p.y + t * (q.y - p.y)))

-- Define the point we are looking for
def has_point_with_odd_doubling (C : list LatticePoint) : Prop :=
  ∃ (p q : LatticePoint) (H : p ∈ C ∧ q ∈ C),
    let m := LatticePoint.mk ((p.x + q.x) / 2) ((p.y + q.y) / 2) in
    (p ≠ q) ∧ (2 * m.x % 2 = 1) ∧ (2 * m.y % 2 = 1)

-- The main theorem statement
theorem polygon_has_odd_doubling_point (C : list LatticePoint)
  (h1 : is_1993_gon C)
  (h2 : sides_no_lattice_points C) :
  has_point_with_odd_doubling C :=
sorry

end polygon_has_odd_doubling_point_l98_98196


namespace tan_pi_over_4_minus_alpha_l98_98714

theorem tan_pi_over_4_minus_alpha
  (α : ℝ)
  (h0 : cos α = -4/5)
  (h1 : α ∈ Ioo (π/2) π) :
  tan (π/4 - α) = 7 :=
sorry

end tan_pi_over_4_minus_alpha_l98_98714


namespace coles_avg_speed_work_l98_98640

-- Define the conditions
variables (speed_return : ℝ) (trip_time_hours : ℝ) (to_work_minutes : ℝ)
-- Assume known values
variables (h1 : speed_return = 105) (h2 : trip_time_hours = 6) (h3 : to_work_minutes = 210)

-- Define the required average speed to be proved
theorem coles_avg_speed_work : 
  let to_work_hours := to_work_minutes / 60 in
  let from_work_hours := trip_time_hours - to_work_hours in
  let distance := speed_return * from_work_hours in
  let to_work_speed := distance / to_work_hours in
  to_work_speed = 75 :=
sorry

end coles_avg_speed_work_l98_98640


namespace select_people_with_boys_and_girls_l98_98259

theorem select_people_with_boys_and_girls :
  let boys := 5
  let girls := 4
  ∃ (ways : ℕ), ways = (Nat.choose (boys + girls) 4 - Nat.choose boys 4 - Nat.choose girls 4) ∧ ways = 120 :=
by
  let boys := 5
  let girls := 4
  use (Nat.choose (boys + girls) 4 - Nat.choose boys 4 - Nat.choose girls 4)
  sorry

end select_people_with_boys_and_girls_l98_98259


namespace combine_square_roots_l98_98565

def can_be_combined (x y: ℝ) : Prop :=
  ∃ k: ℝ, y = k * x

theorem combine_square_roots :
  let sqrt12 := 2 * Real.sqrt 3
  let sqrt1_3 := Real.sqrt 1 / Real.sqrt 3
  let sqrt18 := 3 * Real.sqrt 2
  let sqrt27 := 6 * Real.sqrt 3
  can_be_combined (Real.sqrt 3) sqrt12 ∧
  can_be_combined (Real.sqrt 3) sqrt1_3 ∧
  ¬ can_be_combined (Real.sqrt 3) sqrt18 ∧
  can_be_combined (Real.sqrt 3) sqrt27 :=
by
  sorry

end combine_square_roots_l98_98565


namespace harmonic_not_integer_l98_98055

open BigOperators

-- Define harmonic series sum H_n
noncomputable def H_n (n : ℕ) : ℚ :=
  ∑ i in Finset.range n + 1, 1 / (i : ℚ)

-- Theorem statement
theorem harmonic_not_integer (n : ℕ) : ¬(∃ z : ℤ, H_n n = z) :=
by
  sorry

end harmonic_not_integer_l98_98055


namespace sum_of_reciprocals_correct_difference_of_squares_correct_l98_98094

noncomputable def sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : ℝ :=
  (1 / x) + (1 / y)

noncomputable def difference_of_squares (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : ℝ :=
  x^2 - y^2

theorem sum_of_reciprocals_correct : ∀ (x y : ℝ), x + y = 12 → x * y = 32 → sum_of_reciprocals x y 12 32 = 3 / 8 :=
by
  intros x y h1 h2
  sorry

theorem difference_of_squares_correct : ∀ (x y : ℝ), x + y = 12 → x * y = 32 → difference_of_squares x y 12 32 = 48 * Real.sqrt 5 :=
by
  intros x y h1 h2
  sorry

end sum_of_reciprocals_correct_difference_of_squares_correct_l98_98094


namespace full_boxes_prepared_l98_98602

theorem full_boxes_prepared (total_food_weight : ℝ) (max_shipping_weight : ℝ) 
  (total_boxes : ℕ) : total_food_weight = 777.5 ∧ max_shipping_weight = 2 ∧
  total_boxes = int.floor (total_food_weight / max_shipping_weight) → 
  total_boxes = 388 :=
by
  intros h
  obtain ⟨htw, hmw, htb⟩ := h
  rw [htw, hmw, htb]
  norm_num
  sorry

end full_boxes_prepared_l98_98602


namespace integer_pairs_l98_98211

def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem integer_pairs (a b : ℤ) :
  (is_perfect_square (a^2 + 4 * b) ∧ is_perfect_square (b^2 + 4 * a)) ↔ 
  (a = 0 ∧ b = 0) ∨ (a = -4 ∧ b = -4) ∨ (a = 4 ∧ b = -4) ∨
  (∃ (k : ℕ), a = k^2 ∧ b = 0) ∨ (∃ (k : ℕ), a = 0 ∧ b = k^2) ∨
  (a = -6 ∧ b = -5) ∨ (a = -5 ∧ b = -6) ∨
  (∃ (t : ℕ), a = t ∧ b = 1 - t) ∨ (∃ (t : ℕ), a = 1 - t ∧ b = t) :=
sorry

end integer_pairs_l98_98211


namespace pentagon_area_l98_98456

noncomputable def area_pentagon_AFDCB : ℝ :=
let AF := 10 in
let FD := 15 in
let AD := real.sqrt (AF^2 + FD^2) in
let area_square_ABCD := AD^2 in
let area_tri_AFD := 0.5 * AF * FD in
let area_pentagon := area_square_ABCD - area_tri_AFD in
area_pentagon

theorem pentagon_area : area_pentagon_AFDCB = 250 := by
  sorry

end pentagon_area_l98_98456


namespace male_students_count_l98_98860

theorem male_students_count :
  ∃ (N M : ℕ), 
  (N % 4 = 2) ∧ 
  (N % 5 = 1) ∧ 
  (N = M + 15) ∧ 
  (15 > M) ∧ 
  (M = 11) :=
sorry

end male_students_count_l98_98860


namespace sum_of_coefficients_l98_98339

-- Define a namespace to encapsulate the problem
namespace PolynomialCoefficients

-- Problem statement as a Lean theorem
theorem sum_of_coefficients (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  α^2005 + β^2005 = 1 :=
sorry -- Placeholder for the proof

end PolynomialCoefficients

end sum_of_coefficients_l98_98339


namespace fair_coin_even_heads_unfair_coin_even_heads_l98_98246

-- Define the probability function for an even number of heads for a fair coin
theorem fair_coin_even_heads (n : ℕ) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * 0.5^k * 0.5^(n-k) else 0) = 0.5 :=
sorry

-- Define the probability function for an even number of heads for an unfair coin
theorem unfair_coin_even_heads (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * p^k * (1-p)^(n-k) else 0) = 
    (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_unfair_coin_even_heads_l98_98246


namespace combined_set_average_and_variance_l98_98782

theorem combined_set_average_and_variance {n : ℕ} (x : ℕ → ℝ)
    (avg_x : (∑ i in finset.range n, x i) / n = 10)
    (var_modified_x : (∑ i in finset.range n, (2 * x i + 4 - 24) ^ 2) / n = 8) :
    let combined_x := (finset.range n).sum (λ i, x i) + (finset.range n).sum (λ i, 2 * x i + 4)
    in (combined_x / (2 * n) = 17) ∧
       (let sum_x_squared := (finset.range n).sum (λ i, x i ^ 2) in
        let var_combined_x := ((1 / (2 * n)) * 
                               ((finset.range n).sum (λ i, (x i - 17) ^ 2) + 
                                (finset.range n).sum (λ i, ((2 * x i + 4 - 17) ^ 2))))
        in (var_combined_x = 54)) :=
by sorry

end combined_set_average_and_variance_l98_98782


namespace find_smallest_k_l98_98089

variable (a : ℕ → ℝ)
variable (b : ℕ → ℤ)
variable (k : ℕ)

noncomputable theory

-- Conditions
def a_0 : ℝ := 1
def a_1 : ℝ := real.root 17 3 -- This represents ∛3
def a_recurrence (n : ℕ) (h1 : n ≥ 2) : a n = a (n - 1) * (a (n - 2))^3 := sorry

def b_0 : ℤ := 0
def b_1 : ℤ := 1
def b_recurrence (n : ℕ) (h1 : n ≥ 2) : b n = b (n - 1) + 3 * b (n - 2) := sorry

-- The main proof statement
theorem find_smallest_k : ∃ k : ℕ, k > 0 ∧ (∑ i in finset.range (k + 1), b i) % 17 = 0 := sorry

end find_smallest_k_l98_98089


namespace line_intersects_y_axis_at_2_l98_98624

noncomputable def moved_line (k b : ℝ) : Prop :=
  let original_line := λ x : ℝ, 2 * x - 1
  let new_line := λ x : ℝ, original_line x + 3
  (k = 2 ∧ b = 2)

theorem line_intersects_y_axis_at_2 (k b : ℝ) :
  moved_line k b → (0, 2) ∈ { p : ℝ × ℝ | p.snd = k * p.fst + b } :=
by
  intro h
  rw [moved_line] at h
  cases h with hk hb
  use 0
  rw [hk, hb]
  norm_num
  sorry

end line_intersects_y_axis_at_2_l98_98624


namespace omega_range_l98_98366

noncomputable def f (ω x : ℝ) : ℝ := 
  sin (ω * x + π / 6) * sin (ω * x + 2 * π / 3)

theorem omega_range {ω : ℝ} (hω : ω > 0) :
  (∀ x : ℝ, (∃ y : ℝ, y ∈ Ioo (π / 2) π ∧ f ω y = 0) → False) ↔
  ω ∈ Icc 0 (1 / 3) ∪ Icc (2 / 3) (5 / 6) :=
sorry

end omega_range_l98_98366


namespace find_a7_l98_98266

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98266


namespace problem_equivalent_proof_l98_98317

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98317


namespace express_in_scientific_notation_l98_98503

theorem express_in_scientific_notation : (0.0000028 = 2.8 * 10^(-6)) :=
sorry

end express_in_scientific_notation_l98_98503


namespace ordered_15_tuples_problem_l98_98227

noncomputable def count_ordered_15_tuples : ℕ :=
  12870

theorem ordered_15_tuples_problem :
  ∃ (a : Fin 15 → ℤ), (∀ i : Fin 15, a i ^ 2 = ∑ j, if j = i then 0 else a j) ∧ count_ordered_15_tuples = 12870 :=
sorry

end ordered_15_tuples_problem_l98_98227


namespace sum_first_11_terms_arithmetic_sequence_l98_98724

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℝ)
variable (a6_value : a 6 = 2)
variable (S11_value : ∑ i in Finset.range 11, a (i + 1) = 22)

theorem sum_first_11_terms_arithmetic_sequence :
  (∃ d : ℝ, arithmetic_sequence a ∧ a6_value) → S11_value :=
begin
  sorry
end

end sum_first_11_terms_arithmetic_sequence_l98_98724


namespace solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l98_98521

theorem solution_of_inequality (a b x : ℝ) :
    (b - a * x > 0) ↔
    (a > 0 ∧ x < b / a ∨ 
     a < 0 ∧ x > b / a ∨ 
     a = 0 ∧ false) :=
by sorry

-- Additional theorems to rule out incorrect answers
theorem answer_A_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a|) → false :=
by sorry

theorem answer_B_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x < |b| / |a|) → false :=
by sorry

theorem answer_C_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > -|b| / |a|) → false :=
by sorry

theorem D_is_correct (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a| ∨ x < |b| / |a| ∨ x > -|b| / |a|) → false :=
by sorry

end solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l98_98521


namespace intersection_polar_coordinates_l98_98589

-- Define polar coordinates and relationships
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Condition definitions
def curve1 (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sin θ

def curve2 (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = -1

-- Cartesian conversion
def cartesian_curve1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * y

def cartesian_curve2 (x : ℝ) : Prop :=
  x = -1

-- Define the final conversion from cartesian back to polar
def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  (Real.sqrt (x^2 + y^2), Real.atan2 y x)

theorem intersection_polar_coordinates :
  ∃ ρ θ, curve1 ρ θ ∧ curve2 ρ θ ∧ (ρ = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4) :=
by
  sorry

end intersection_polar_coordinates_l98_98589


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98237

-- Define the probability of getting heads as p for the biased coin
variable {p : ℝ} (hp : 0 < p ∧ p < 1)

-- Define the number of tosses
variable {n : ℕ}

-- Fair coin probability of getting an even number of heads
theorem fair_coin_even_heads : 
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * (1 / 2)^n else 0) = 0.5 :=
sorry

-- Biased coin probability of getting an even number of heads
theorem biased_coin_even_heads :
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) else 0) 
  = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98237


namespace probability_sum_eq_3_conditional_probability_sum_eq_3_l98_98106

noncomputable def fair_die : Type :=
  { d : ℕ // d ≥ 1 ∧ d ≤ 6 }

-- Indicator function to map faces to sequence values
def a (d : fair_die) : ℤ :=
  if d.val % 2 = 1 then 1 else -1

noncomputable def sum_a (seq : list fair_die) : ℤ :=
  seq.map a |>.sum

-- Lean 4 statement to prove the first part
theorem probability_sum_eq_3 :
  let prob := (21 : ℚ) / 128 in
  ∃ seq : list (fair_die × ℚ), 
    seq.prod (λ t, if sum_a (seq.map prod.fst) = 3 then t.snd else 0) = prob :=
sorry

-- Lean 4 statement to prove the second part
theorem conditional_probability_sum_eq_3 :
  let prob := (11 : ℚ) / 128 in
  ∃ seq : list (fair_die × ℚ), 
    (sum_a (seq.take 2.map prod.fst) ≠ 0) → 
    seq.prod (λ t, if sum_a (seq.map prod.fst) = 3 then t.snd else 0) = prob :=
sorry

end probability_sum_eq_3_conditional_probability_sum_eq_3_l98_98106


namespace union_of_A_and_B_l98_98337

open Set Real

noncomputable def A : Set ℝ := {x | log 2 x > 1}
noncomputable def B : Set ℝ := {x | x ≥ 1}

theorem union_of_A_and_B : A ∪ B = Ici 1 := by
  sorry

end union_of_A_and_B_l98_98337


namespace integral_value_eq_e_minus_1_l98_98097

noncomputable def definite_integral_value : ℝ :=
  ∫ x in 0..1, (x^2 + Real.exp x - 1/3)

theorem integral_value_eq_e_minus_1 : definite_integral_value = Real.exp 1 - 1 := 
  sorry

end integral_value_eq_e_minus_1_l98_98097


namespace find_a7_l98_98283

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98283


namespace parallelogram_area_increase_l98_98073
open Real

/-- The area of the parallelogram increases by 600 square meters when the base is increased by 20 meters. -/
theorem parallelogram_area_increase :
  ∀ (base height new_base : ℝ), 
    base = 65 → height = 30 → new_base = base + 20 → 
    (new_base * height - base * height) = 600 := 
by
  sorry

end parallelogram_area_increase_l98_98073


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98249

-- Problem 1: Fair coin, probability of even heads
def fair_coin_even_heads_prob (n : ℕ) : Prop :=
  0.5 = 0.5

-- Problem 2: Biased coin, probability of even heads
def biased_coin_even_heads_prob (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : Prop :=
  let q := 1 - p in
  (0.5 * (1 + (1 - 2 * p)^n) = (1 + (1 - 2*p)^n) / 2)

-- Mock proof to ensure Lean accepts the definitions
theorem fair_coin_even_heads (n : ℕ) : fair_coin_even_heads_prob n :=
begin
  -- Proof intentionally omitted
  sorry
end

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : biased_coin_even_heads_prob n p hp :=
begin
  -- Proof intentionally omitted
  sorry
end

end fair_coin_even_heads_biased_coin_even_heads_l98_98249


namespace find_m_if_f_monotonic_l98_98042

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  4 * x^3 + m * x^2 + (m - 3) * x + n

def is_monotonically_increasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2

theorem find_m_if_f_monotonic (m n : ℝ)
  (h : is_monotonically_increasing_on_ℝ (f m n)) :
  m = 6 :=
sorry

end find_m_if_f_monotonic_l98_98042


namespace symmetric_points_a_minus_b_l98_98709

theorem symmetric_points_a_minus_b (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = -1) :
  a - b = -4 := 
sorry

end symmetric_points_a_minus_b_l98_98709


namespace ratio_S9_S5_l98_98829

variable {a_n : ℕ → ℝ} -- Arithmetic sequence
variable {S : ℕ → ℝ} -- Sum of the first n terms
variable {a1 a5 a3 a9 : ℝ}

-- Conditions: Arithmetic sequence relationships and given ratio
axiom arithmetic_sequence (n : ℕ) : a_n n = a1 + (n - 1) * a_n 1
axiom sum_first_n_terms (n : ℕ) : S(n) = n * (a1 + a_n n) / 2
axiom given_ratio : a5 = 5 * (a3 / 9)

-- Prove \(\frac{S_9}{S_5} = 1\)
theorem ratio_S9_S5 : S 9 / S 5 = 1 :=
  sorry

end ratio_S9_S5_l98_98829


namespace part1_part2_part3_l98_98597

noncomputable def total_students : ℕ := 40
noncomputable def qualified_students : ℕ := 35
noncomputable def prob_qualified : ℚ := qualified_students / total_students

theorem part1 :
  prob_qualified = 7 / 8 :=
sorry

noncomputable def prob_X (x : ℕ) : ℚ :=
  if x = 0 then 1 / 30
  else if x = 1 then 3 / 10
  else if x = 2 then 1 / 2
  else if x = 3 then 1 / 6
  else 0

noncomputable def expected_X : ℚ :=
  ∑ i in finset.range 4, i * prob_X i

theorem part2 :
  expected_X = 9 / 5 :=
sorry

noncomputable def prob_male_excellent : ℚ := 1 / 5
noncomputable def prob_female_excellent : ℚ := 3 / 10
noncomputable def prob_two_excellent : ℚ :=
  prob_male_excellent ^ 2 * (1 - prob_female_excellent) +
  prob_male_excellent * (1 - prob_male_excellent) * prob_female_excellent +
  (1 - prob_male_excellent) * prob_male_excellent * prob_female_excellent

theorem part3 :
  prob_two_excellent = 31 / 250 :=
sorry

end part1_part2_part3_l98_98597


namespace transaction_loss_l98_98979

theorem transaction_loss 
  (sell_price_house sell_price_store : ℝ)
  (cost_price_house cost_price_store : ℝ)
  (house_loss_percent store_gain_percent : ℝ)
  (house_loss_eq : sell_price_house = (4/5) * cost_price_house)
  (store_gain_eq : sell_price_store = (6/5) * cost_price_store)
  (sell_prices_eq : sell_price_house = 12000 ∧ sell_price_store = 12000)
  (house_loss_percent_eq : house_loss_percent = 0.20)
  (store_gain_percent_eq : store_gain_percent = 0.20) :
  cost_price_house + cost_price_store - (sell_price_house + sell_price_store) = 1000 :=
by
  sorry

end transaction_loss_l98_98979


namespace ranges_of_a_and_m_l98_98372

open Set Real

def A : Set Real := {x | x^2 - 3*x + 2 = 0}
def B (a : Real) : Set Real := {x | x^2 - a*x + a - 1 = 0}
def C (m : Real) : Set Real := {x | x^2 - m*x + 2 = 0}

theorem ranges_of_a_and_m (a m : Real) :
  A ∪ B a = A → A ∩ C m = C m → (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2*sqrt 2 < m ∧ m < 2*sqrt 2)) :=
by
  have hA : A = {1, 2} := sorry
  sorry

end ranges_of_a_and_m_l98_98372


namespace cost_of_book_sold_at_loss_l98_98576

theorem cost_of_book_sold_at_loss:
  ∃ (C1 C2 : ℝ), 
    C1 + C2 = 490 ∧ 
    C1 * 0.85 = C2 * 1.19 ∧ 
    C1 = 285.93 :=
by
  sorry

end cost_of_book_sold_at_loss_l98_98576


namespace perimeter_triangle_angle_difference_cos_l98_98036

variable (a b c : ℝ)
variable (cosC : ℝ)
variable (triangleABC : Triangle)
variable (A B C : Angle)

-- Given conditions
def side_a := 1
def side_b := 2
def side_c := 2
def cos_C := cosC

-- Proof that the perimeter is 5
theorem perimeter_triangle : side_a + side_b + side_c = 5 := 
by
  -- Calculation steps from the solution

  sorry

-- Proof that cos(A - C) is a specific value
theorem angle_difference_cos (cosA : ℝ) (sinA : ℝ) (sinC : ℝ) : cos (A - C) = cosA * cosC + sinA * sinC := 
by
  -- Calculation steps from the solution

  sorry

end perimeter_triangle_angle_difference_cos_l98_98036


namespace coefficient_x4_expansion_l98_98074

theorem coefficient_x4_expansion :
  (polynomial.C 120) = polynomial.coeff ((polynomial.C 1 + polynomial.C 3 * polynomial.X^3) * 
  (polynomial.C 1 * polynomial.X^2 + polynomial.C (2:ℚ) * polynomial.X^(-1))^5) 4 :=
sorry

end coefficient_x4_expansion_l98_98074


namespace melted_mixture_weight_l98_98139

-- Definitions based on the given conditions
def ratio (z c : ℝ) : Prop := z / c = 9 / 11

def zinc_weight : ℝ := 35.1

def copper_weight_from_ratio (z_kg : ℝ) : ℝ := (z_kg * 11) / 9

-- The statement of the proposition to prove
theorem melted_mixture_weight : 
  let z := zinc_weight in
  let c := copper_weight_from_ratio z in
  (z + c) ≈ 77.889 :=
sorry

end melted_mixture_weight_l98_98139


namespace area_ratio_proof_l98_98950

noncomputable def area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) : ℝ := 
  (a * b) / (c * d)

theorem area_ratio_proof (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  area_ratio a b c d h1 h2 = 4 / 9 := by
  sorry

end area_ratio_proof_l98_98950


namespace min_product_of_distances_l98_98330

-- Definitions
def equilateral_triangle (A B C : Point) := 
  ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a

def distance (P Q : Point) : ℝ := sorry -- Assume this definition exists

def distance_from_point_to_side (P : Point) (side : Line) : ℝ := sorry -- Assume this definition exists

def product_of_distances (P : Point) (triangle : Triangle) :=
  let u := distance_from_point_to_side P triangle.side1
  let v := distance_from_point_to_side P triangle.side2
  let w := distance_from_point_to_side P triangle.side3
  u * v * w

-- Given the conditions of the problem
theorem min_product_of_distances :
  ∀ {A B C D E F R Q S P : Point}
    (ABC_equi : equilateral_triangle A B C)
    (side_lengths : distance A B = 4 ∧ distance B C = 4 ∧ distance C A = 4)
    (points_on_sides : distance A E = 1 ∧ distance B F = 1 ∧ distance C D = 1)
    (RQS_tri : ∃ (AD BE CF : Line), AD ∩ BE = R ∧ BE ∩ CF = Q ∧ CF ∩ AD = S),
  let triangle_RQS := {side1 := AD, side2 := BE, side3 := CF} in
  (∀ (P : Point),
    ∃ (R Q S : Point),
    product_of_distances P triangle_RQS = (product_of_distances R triangle_RQS) ∨ 
    product_of_distances P triangle_RQS = (product_of_distances Q triangle_RQS) ∨ 
    product_of_distances P triangle_RQS = (product_of_distances S triangle_RQS)) ∧
  min_value (product_of_distances P triangle_RQS) = (8 * real.sqrt 3 / 27) :=
sorry

end min_product_of_distances_l98_98330


namespace trapezoid_concyclic_points_l98_98523

noncomputable def is_trapezoid (A B C D : ℝ) : Prop :=
∃ (O : ℝ), is_cyclic_quadrilateral A B C D O

noncomputable def is_concyclic (A B C D : ℝ) : Prop :=
∃ (O : ℝ), on_circle A O ∧ on_circle B O ∧ on_circle C O ∧ on_circle D O

noncomputable def is_symmetric (A1 A2 P Q : ℝ) : Prop :=
let M := midpoint P Q in symmetric_about A1 A2 M

theorem trapezoid_concyclic_points
  (A B C D A1 B1 A2 B2 : ℝ)
  (H1 : is_trapezoid A B C D)
  (H2 : is_concyclic_points C D A1 B1 )
  (H3 : is_symmetric A1 A2 C A)
  (H4 : is_symmetric B1 B2 C B) :
  is_concyclic A B A2 B2 :=
sorry

end trapezoid_concyclic_points_l98_98523


namespace coin_problem_l98_98160

theorem coin_problem :
  ∃ n : ℕ, (n % 8 = 5) ∧ (n % 7 = 2) ∧ (n % 9 = 1) := 
sorry

end coin_problem_l98_98160


namespace find_x_pow_y_l98_98694

-- Define the given condition as a lemma.
lemma problem_condition (x y : ℝ) :
  |x - 2y| + (5x - 7y - 3)^2 = 0 → x = 2 ∧ y = 1 :=
by {
  intro h,
  have h1 : |x - 2y| = 0,
  {
    from (add_eq_zero_iff_eq_zero_of_nonneg (abs_nonneg _) (pow_two_nonneg _)).1 h,
  },
  have h2 : (5x - 7y - 3)^2 = 0,
  {
    from (add_eq_zero_iff_eq_zero_of_nonneg (abs_nonneg _) (pow_two_nonneg _)).2 h,
  },
  have h3 : x - 2y = 0,
  {
    from abs_eq_zero.1 h1,
  },
  have h4 : 5x - 7y - 3 = 0,
  {
    from pow_eq_zero_iff_eq_zero.1 h2,
  },
  
  split,
  {
    rw [← h3, ← h4],
    solve_by_elim,
  }
}

-- The final goal to be proven.
theorem find_x_pow_y :
  ∃ x y : ℝ, |x - 2y| + (5x - 7y - 3)^2 = 0 ∧ x^y = 2 :=
by {
  existsi (2:ℝ),
  existsi (1:ℝ),
  split,
  {
    calc |2 - 2 * 1| + (5 * 2 - 7 * 1 - 3)^2 = |0| + 0^2 : by norm_num
    ... = 0 : by norm_num,
  },
  {
    calc (2:ℝ)^(1:ℝ) = 2 : by norm_num,
  }
}

end find_x_pow_y_l98_98694


namespace jackson_savings_per_paycheck_l98_98425

-- Jackson wants to save $3,000
def total_savings : ℝ := 3000

-- The vacation is 15 months away
def months_to_save : ℝ := 15

-- Jackson is paid twice a month
def pay_times_per_month : ℝ := 2

-- Jackson's required savings per paycheck to have $3,000 saved in 15 months
theorem jackson_savings_per_paycheck :
  (total_savings / (months_to_save * pay_times_per_month)) = 100 :=
by simp [total_savings, months_to_save, pay_times_per_month]; norm_num; sorry

end jackson_savings_per_paycheck_l98_98425


namespace disjoint_range_probability_l98_98440

def A : Finset ℕ := {1, 2, 3, 4}

theorem disjoint_range_probability :
  let funcs := Finset.pi_finset (λ _ : A, A)
  let total_pairs := finset.card funcs * finset.card funcs
  let disjoint_pairs := 1740
  let gcd := nat.gcd 435 16384
  m = 435 := by
  let prob := (disjoint_pairs, total_pairs)
  let simplest_form := (prob.1 / gcd, prob.2 / gcd)
  have : simplest_form.1 = m := rfl
  have : simplest_form.1 = 435 := sorry
  sorry

end disjoint_range_probability_l98_98440


namespace cubic_poly_cd_ratio_l98_98095

noncomputable theory

def cubic_poly_roots (a b c d : ℤ) (r1 r2 r3 : ℚ) : Prop :=
r1 = 1 ∧ r2 = 1/2 ∧ r3 = 4 ∧
a ≠ 0 ∧ r1 * r2 * r3 = -d/a ∧ (r1 * r2 + r1 * r3 + r2 * r3) = c/a

theorem cubic_poly_cd_ratio (a b c d : ℤ) (h : cubic_poly_roots a b c d 1 (1/2) 4) :
  (c : ℚ)/(d : ℚ) = -13/4 := 
sorry

end cubic_poly_cd_ratio_l98_98095


namespace find_a7_l98_98279

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98279


namespace max_min_distance_from_point_and_slope_l98_98704

def circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

def max_min_distance_from_point_on_circle (x y : ℝ) (Qx Qy : ℕ) : Prop :=
  Qx = -2 ∧ Qy = 3 ∧ circle x y →
  ( ∀ (M : ℝ × ℝ), 
    circle M.1 M.2 →
    let MQ := ((M.1 - Qx)^2 + (M.2 - Qy)^2) in
    MQ ≤ (6 * Real.sqrt 2)^2 ∧ MQ ≥ (2 * Real.sqrt 2)^2)

def max_min_slope (m n : ℝ) : Prop :=
  circle m n →
  let k := (n - 3) / (m + 2) in
  k ≤ 2 + Real.sqrt 3 ∧ k ≥ 2 - Real.sqrt 3

-- To prove the propositions:
theorem max_min_distance_from_point_and_slope : max_min_distance_from_point_on_circle ∧ max_min_slope :=
by sorry

end max_min_distance_from_point_and_slope_l98_98704


namespace product_of_solutions_of_abs_eq_l98_98404

theorem product_of_solutions_of_abs_eq (x : ℝ) (h : |x - 5| - 4 = 3) : x * (if x = 12 then -2 else if x = -2 then 12 else 1) = -24 :=
by
  sorry

end product_of_solutions_of_abs_eq_l98_98404


namespace count_numbers_with_perfect_square_factors_l98_98775

open Set

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ m * m ∣ n

theorem count_numbers_with_perfect_square_factors (s : Finset ℕ) (hs : s = Finset.range 101) :
  (Finset.filter has_perfect_square_factor_other_than_one s).card = 41 :=
by {
  sorry
}

end count_numbers_with_perfect_square_factors_l98_98775


namespace sufficient_conditions_for_positive_product_l98_98093

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1) → a * b > 0 :=
by sorry

end sufficient_conditions_for_positive_product_l98_98093


namespace problem_arithmetic_sequence_l98_98810

-- Definitions based on given conditions
def a1 : ℕ := 2
def d := (13 - 2 * a1) / 3

-- Definition of the nth term in the arithmetic sequence
def a (n : ℕ) : ℕ := a1 + (n - 1) * d

-- The required proof problem statement
theorem problem_arithmetic_sequence : a 4 + a 5 + a 6 = 42 := 
by
  -- placeholders for the actual proof
  sorry

end problem_arithmetic_sequence_l98_98810


namespace max_cells_cut_diagonals_l98_98932

theorem max_cells_cut_diagonals (board_size : ℕ) (k : ℕ) (internal_cells : ℕ) :
  board_size = 9 →
  internal_cells = (board_size - 2) ^ 2 →
  64 = internal_cells →
  V = internal_cells + k →
  E = 4 * k →
  k ≤ 21 :=
by
  sorry

end max_cells_cut_diagonals_l98_98932


namespace six_digit_even_numbers_l98_98379

theorem six_digit_even_numbers {d : Finset ℕ} (h : d = {0, 1, 2, 3, 4, 5}) :
  (count_six_digit_even_numbers d = 336) := 
by sorry

noncomputable def count_six_digit_even_numbers (d : Finset ℕ) : ℕ :=
  if d = {0, 1, 2, 3, 4, 5} then
    let possibilities_last_digit_0 := 5 * nat.factorial 4,
        possibilities_last_digit_2_or_4 := 2 * 5 * nat.factorial 4 in
    possibilities_last_digit_0 + possibilities_last_digit_2_or_4
  else 
    0

end six_digit_even_numbers_l98_98379


namespace median_of_occupied_rooms_is_13_5_l98_98627

theorem median_of_occupied_rooms_is_13_5:
  let rooms : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25]
  let median := (rooms[10] + rooms[11].toReal) / 2
  median = 13.5 :=
by
  sorry

end median_of_occupied_rooms_is_13_5_l98_98627


namespace find_number_l98_98391

theorem find_number (x y a : ℝ) (h₁ : x * y = 1) (h₂ : (a ^ ((x + y) ^ 2)) / (a ^ ((x - y) ^ 2)) = 1296) : a = 6 :=
sorry

end find_number_l98_98391


namespace fruit_basket_count_l98_98522

theorem fruit_basket_count :
  ∃ (f : ℕ×ℕ×ℕ×ℕ → Prop), -- f is a predicate on tuples of four natural numbers (the counts of apples, bananas, oranges, and pears).
  (f = λ (t : ℕ × ℕ × ℕ × ℕ), let (a, b, c, d) = t in 2 * a + 5 * b + c + d = 62 ∧ c ≤ 4 ∧ d ≤ 1) ∧
  ( (∃ x, f x) ∧ 
    (∃ n, n + 1 = ∑ x in finset.filter f (finset.range ((62 / 2) + 1) ×ˢ finset.range ((62 / 5) + 1) ×ˢ finset.range 5 ×ˢ finset.range 2) ∧ n = 62) ) :=
by {
  sorry
}

end fruit_basket_count_l98_98522


namespace area_relation_l98_98413

noncomputable def triangle_area {α : Type*} [linear_ordered_field α] (A B C : euclidean_geometry.Point α) : α := 
  euclidean_geometry.triangle.area A B C

noncomputable def midpoint {α : Type*} [linear_ordered_field α] (A B : euclidean_geometry.Point α) : euclidean_geometry.Point α := 
  ⟨(A.1 + B.1) / 2, (A.2 + B.2) / 2⟩

noncomputable def centroid {α : Type*} [linear_ordered_field α] (A B C : euclidean_geometry.Point α) : euclidean_geometry.Point α := 
  ⟨(A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3⟩

noncomputable def trisect_point {α : Type*} [linear_ordered_field α] (A C : euclidean_geometry.Point α) : euclidean_geometry.Point α := 
  ⟨(3 * A.1 + C.1) / 4, (3 * A.2 + C.2) / 4⟩

noncomputable def triangle_intersection {α : Type*} [linear_ordered_field α] (M P C N : euclidean_geometry.Point α) : euclidean_geometry.Point α := 
  sorry -- Intersection implementation skipped. For simplicity, assuming the existence of such a point Q.

theorem area_relation (A B C M N O P Q : euclidean_geometry.Point ℝ)
  (h₁ : M = midpoint B C)
  (h₂ : N = midpoint A B)
  (h₃ : O = centroid A B C)
  (h₄ : P = trisect_point A C)
  (h₅ : ∃ Q, Q = triangle_intersection M P C N)
  (h₆ : triangle_area O M Q = n) : 
  triangle_area A B C = 48 * n :=
sorry -- Proof required for area calculation

end area_relation_l98_98413


namespace binary_to_decimal_l98_98207

theorem binary_to_decimal : let b := [1, 0, 1, 1, 1] in
                             b[0] * 2^0 + b[1] * 2^1 + b[2] * 2^2 + b[3] * 2^3 + b[4] * 2^4 = 23 := 
sorry

end binary_to_decimal_l98_98207


namespace log_arithmetic_l98_98587

theorem log_arithmetic :
  let log := Real.log10 in
  log 25 + log 2 * log 50 + (log 2) ^ 2 = 2 :=
by
  let log := Real.log10
  have h1 : log 25 = 2 * log 5 := sorry
  have h2 : log 50 + log 2 = log 100 := sorry
  have h3 : log 5 + log 2 = 1 := sorry
  sorry

end log_arithmetic_l98_98587


namespace coin_flip_probability_l98_98593

noncomputable def probability_of_heads (p : ℝ) : Prop :=
  p < 1/2 ∧ (C(6, 3) * p^3 * (1 - p)^3 = 1/20) 

theorem coin_flip_probability :
  ∃ p : ℝ, probability_of_heads p ∧ (p ≈ 0.125) :=
by
  sorry

end coin_flip_probability_l98_98593


namespace tom_final_payment_l98_98108

noncomputable def cost_of_fruit (kg: ℝ) (rate_per_kg: ℝ) := kg * rate_per_kg

noncomputable def total_bill := 
  cost_of_fruit 15.3 1.85 + cost_of_fruit 12.7 2.45 + cost_of_fruit 10.5 3.20 + cost_of_fruit 6.2 4.50

noncomputable def discount (bill: ℝ) := 0.10 * bill

noncomputable def discounted_total (bill: ℝ) := bill - discount bill

noncomputable def sales_tax (amount: ℝ) := 0.06 * amount

noncomputable def final_amount (bill: ℝ) := discounted_total bill + sales_tax (discounted_total bill)

theorem tom_final_payment : final_amount total_bill = 115.36 :=
  sorry

end tom_final_payment_l98_98108


namespace sum_three_greater_than_100_l98_98858

open Real

theorem sum_three_greater_than_100 
  (x : Fin 100 → ℝ)
  (h1 : ∑ i, x i < 300)
  (h2 : ∑ i, (x i)^2 > 10000)
  (h_pos : ∀ i, 0 < x i)
  : ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (x i + x j + x k > 100) :=
sorry

end sum_three_greater_than_100_l98_98858


namespace negate_proposition_l98_98084

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^2 + 2 > 6)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) :=
by sorry

end negate_proposition_l98_98084


namespace complement_union_l98_98961

open Set Real

noncomputable def S : Set ℝ := {x | x > -2}
noncomputable def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_union (x : ℝ): x ∈ (univ \ S) ∪ T ↔ x ≤ 1 :=
by
  sorry

end complement_union_l98_98961


namespace count_perfect_square_factors_l98_98757

open Finset

noncomputable def count_divisible_by (n : ℕ) (s : Finset ℕ) : ℕ :=
s.filter (λ x, x % n = 0).card

theorem count_perfect_square_factors :
  let S := (range 100).map (λ n, n + 1)
  let perfect_squares := [4, 9, 16, 25, 36, 49, 64, 81, 100] 
  let total := S.card 
  let count_4 := count_divisible_by 4 S
  let count_9 := count_divisible_by 9 S
  let count_16 := count_divisible_by 16 S
  let count_25 := count_divisible_by 25 S
  let count_36 := count_divisible_by 36 S
  let count_49 := count_divisible_by 49 S
  let count_64 := count_divisible_by 64 S
  let count_81 := count_divisible_by 81 S
  let count_100 := count_divisible_by 100 S
  count_4 + (count_9 - count_divisible_by (Nat.lcm 4 9) S) +
    (count_16 - count_divisible_by 4 S) +
    (count_25 - count_divisible_by (Nat.lcm 4 25) S) +
    (count_36 - count_divisible_by 4 S) +
    count_49 + (count_64 - count_divisible_by 4 S) +
    (count_81 - count_divisible_by (Nat.lcm 9 81) S) +
    (count_100 - count_divisible_by 4 S)
    = 40 := 
by
  sorry

end count_perfect_square_factors_l98_98757


namespace invertible_elements_mod8_invertible_elements_mod9_l98_98225

noncomputable def invertible_mod8 : {x : ℕ // x < 8} → option {y : ℕ // y < 8} :=
  λ x, if gcd x.1 8 = 1 then
          some ⟨x.1, by linarith [(x.2 : x.1 < 8)] ⟩
       else
          none

noncomputable def inverse_mod8 (x : {x : ℕ // x < 8}) : option {y : ℕ // y < 8} :=
  invertible_mod8 x

theorem invertible_elements_mod8 :
  ∀ x, x ∈ ({1, 3, 5, 7} : set {x : ℕ // x < 8}) ↔ ∃ y, inverse_mod8 x = some y :=
sorry

noncomputable def invertible_mod9 : {x : ℕ // x < 9} → option {y : ℕ // y < 9} :=
  λ x, if gcd x.1 9 = 1 then
          some $ classical.some $ nat.modeq.exists_modeq_of_coprime (gcd x.1 9).eq_one $
            by linarith [(x.2 : x.1 < 9)]
       else
          none

noncomputable def inverse_mod9 (x : {x : ℕ // x < 9}) : option {y : ℕ // y < 9} :=
  invertible_mod9 x

theorem invertible_elements_mod9 :
  ∀ x, x ∈ ({1, 2, 4, 5, 7, 8} : set {x : ℕ // x < 9}) ↔ ∃ y, inverse_mod9 x = some y :=
sorry

end invertible_elements_mod8_invertible_elements_mod9_l98_98225


namespace solution_y_volume_is_450_l98_98490

-- Define the variables and known conditions
variables (x_volume y_volume : ℝ)
variables (x_concentration y_concentration target_concentration : ℝ)
variables (alcohol_x total_volume_target alcohol_target : ℝ)

-- Given conditions
def conditions : Prop :=
  x_concentration = 0.10 ∧
  y_concentration = 0.30 ∧
  x_volume = 300 ∧
  target_concentration = 0.22

-- The specific problem to solve
def proof_problem : Prop :=
  x_volume * x_concentration + y_volume * y_concentration = target_concentration * (x_volume + y_volume) ∧ y_volume = 450

-- The theorem statement combining the conditions with the proof problem
theorem solution_y_volume_is_450 : conditions → proof_problem :=
by
  intros,
  sorry

end solution_y_volume_is_450_l98_98490


namespace cross_section_area_l98_98081

theorem cross_section_area (H α : ℝ) (h_pos : 0 < H) (α_pos : 0 < α) (α_lt_pi_div_2 : α < π / 2) :
  ∃ A : ℝ, A = H ^ 2 * (Real.sqrt 3) * Real.cot α / Real.sin α :=
  sorry

end cross_section_area_l98_98081


namespace linear_function_value_l98_98697

theorem linear_function_value
  (a b c : ℝ)
  (h1 : 3 * a + b = 8)
  (h2 : -2 * a + b = 3)
  (h3 : -3 * a + b = c) :
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 13 :=
by
  sorry

end linear_function_value_l98_98697


namespace unique_ordered_triples_l98_98674

theorem unique_ordered_triples (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 3) (h2 : (x + y + z) * (x^2 + y^2 + z^2) = 9) : 
  { (x, y, z) | x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x^2 + y^2 + z^2 = 3 ∧ (x + y + z) * (x^2 + y^2 + z^2) = 9 }.card = 1 :=
sorry

end unique_ordered_triples_l98_98674


namespace problem_equivalent_proof_l98_98312

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98312


namespace no_nat_numbers_satisfy_lcm_eq_l98_98494

theorem no_nat_numbers_satisfy_lcm_eq (n m : ℕ) :
  ¬ (Nat.lcm (n^2) m + Nat.lcm n (m^2) = 2019) :=
sorry

end no_nat_numbers_satisfy_lcm_eq_l98_98494


namespace train_speed_in_kmh_l98_98996

def length_of_train : ℝ := 600
def length_of_overbridge : ℝ := 100
def time_to_cross_overbridge : ℝ := 70

theorem train_speed_in_kmh :
  (length_of_train + length_of_overbridge) / time_to_cross_overbridge * 3.6 = 36 := 
by 
  sorry

end train_speed_in_kmh_l98_98996


namespace sqrt_sum_ineq_l98_98957

theorem sqrt_sum_ineq (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab_eq : a * b = c * d) (hab_sum_gt : a + b > c + d) : 
  sqrt a + sqrt b > sqrt c + sqrt d := by
  sorry

end sqrt_sum_ineq_l98_98957


namespace symmetric_points_a_minus_b_l98_98710

theorem symmetric_points_a_minus_b (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = -1) :
  a - b = -4 := 
sorry

end symmetric_points_a_minus_b_l98_98710


namespace geometric_seq_a7_l98_98309

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98309


namespace probability_of_king_then_ace_is_4_over_663_l98_98112

noncomputable def probability_king_and_ace {α : Type*} [ProbabilityTheory α] 
  (deck : Finset α) (is_king : α → Prop) (is_ace : α → Prop) 
  (h1 : deck.card = 52) (h2 : (deck.filter is_king).card = 4) 
  (h3 : (deck.filter is_ace).card = 4) : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_king_then_ace_is_4_over_663 
  {α : Type*} [ProbabilityTheory α] 
  (deck : Finset α) (is_king : α → Prop) (is_ace : α → Prop) 
  (h1 : deck.card = 52) (h2 : (deck.filter is_king).card = 4) 
  (h3 : (deck.filter is_ace).card = 4) :
  probability_king_and_ace deck is_king is_ace h1 h2 h3 = 4 / 663 := 
sorry

end probability_of_king_then_ace_is_4_over_663_l98_98112


namespace base_length_of_isosceles_triangle_l98_98881

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l98_98881


namespace math_problem_solution_l98_98015

noncomputable def problem_statement : Prop :=
  let AB := 4
  let AC := 6
  let BC := 5
  let area_ABC := 9.9216 -- Using the approximated area directly for simplicity
  let K_div3 := area_ABC / 3
  let GP := (2 * K_div3) / BC
  let GQ := (2 * K_div3) / AC
  let GR := (2 * K_div3) / AB
  GP + GQ + GR = 4.08432

theorem math_problem_solution : problem_statement :=
by
  sorry

end math_problem_solution_l98_98015


namespace average_weight_increase_l98_98876

theorem average_weight_increase (A : ℝ) (weights : Fin 12 → ℝ) (h : ∑ i, weights i = 12 * A)
  (k : Fin 12) (new_weight_person : ℝ) (h_old : weights k = 58) (h_new : new_weight_person = 106) :
  ∑ i, (if i = k then new_weight_person else weights i) = 12 * A + 48 ∧
  (∑ i, (if i = k then new_weight_person else weights i)) / 12 - A = 4 :=
by
  sorry

end average_weight_increase_l98_98876


namespace zoe_distance_more_than_leo_l98_98567

theorem zoe_distance_more_than_leo (d t s : ℝ)
  (maria_driving_time : ℝ := t + 2)
  (maria_speed : ℝ := s + 15)
  (zoe_driving_time : ℝ := t + 3)
  (zoe_speed : ℝ := s + 20)
  (leo_distance : ℝ := s * t)
  (maria_distance : ℝ := (s + 15) * (t + 2))
  (zoe_distance : ℝ := (s + 20) * (t + 3))
  (maria_leo_distance_diff : ℝ := 110)
  (h1 : maria_distance = leo_distance + maria_leo_distance_diff)
  : zoe_distance - leo_distance = 180 :=
by
  sorry

end zoe_distance_more_than_leo_l98_98567


namespace vector_subtraction_l98_98370

variables (a b : ℝ × ℝ)

-- Definitions based on conditions
def vector_a : ℝ × ℝ := (1, -2)
def m : ℝ := 2
def vector_b : ℝ × ℝ := (4, m)

-- Prove given question equals answer
theorem vector_subtraction :
  vector_a = (1, -2) →
  vector_b = (4, m) →
  (1 * 4 + (-2) * m = 0) →
  5 • vector_a - vector_b = (1, -12) := by
  intros h1 h2 h3
  sorry

end vector_subtraction_l98_98370


namespace focus_of_parabola_y_squared_eq_ax_given_directrix_l98_98785

noncomputable def parabola_focus (a : ℝ) : ℝ × ℝ :=
  let p := 1 * 2 in (p, 0)

theorem focus_of_parabola_y_squared_eq_ax_given_directrix :
  ∀ a, (∃ d : ℝ, (λ x : ℝ, y = sqrt (a * x)) = directrix) →
    (parabola_focus a) = (1, 0) := 
by
  intros a h,
  -- sorry to skip the proof
  sorry

end focus_of_parabola_y_squared_eq_ax_given_directrix_l98_98785


namespace car_speed_l98_98141

theorem car_speed (d t : ℝ) (h_d : d = 624) (h_t : t = 3) : d / t = 208 := by
  sorry

end car_speed_l98_98141


namespace malcolm_route_ratio_l98_98458

-- Define the conditions of the problem
def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def time_final_stage_first_route : ℕ := (time_uphill + time_path) / 3
def total_time_first_route : ℕ := time_uphill + time_path + time_final_stage_first_route

def time_flat_path : ℕ := 14
def R : ℚ := 2  -- Consider R as a rational number to allow division
def total_time_second_route : ℕ := time_flat_path + (R * time_flat_path).toNat

-- Add the main theorem, stating that the second route is 18 minutes longer than the first
theorem malcolm_route_ratio : total_time_second_route = total_time_first_route + 18 := by
  sorry

end malcolm_route_ratio_l98_98458


namespace jules_family_members_l98_98026

def vacation_cost := 1000
def start_walk_fee := 2
def block_fee := 1.25
def num_dogs := 20
def num_blocks := 128

theorem jules_family_members : 
  vacation_cost / (num_dogs * start_walk_fee + num_blocks * block_fee) = 5 := by
  sorry

end jules_family_members_l98_98026


namespace completing_square_l98_98128

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l98_98128


namespace probability_A_is_half_events_A_and_C_are_mutually_exclusive_l98_98805

structure Can :=
  (balls : List ℕ)

def CanA := Can.mk [1, 2, 3]
def CanB := Can.mk [1, 2]

def event_A (a b : ℕ) : Prop := a + b < 4
def event_B (a b : ℕ) : Prop := (a * b) % 2 = 0
def event_C (a b : ℕ) : Prop := a * b > 3

-- Using this theorem structure to denote the conclusive results to be proved
theorem probability_A_is_half :
  let outcomes := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
  let A_occurrences := List.filter (λ p, event_A p.fst p.snd) outcomes
  P(event_A) = 1 / 2 :=
by
  sorry

theorem events_A_and_C_are_mutually_exclusive :
  ∀ (a b : ℕ), event_A a b → ¬ event_C a b :=
by
  sorry

end probability_A_is_half_events_A_and_C_are_mutually_exclusive_l98_98805


namespace sum_of_first_8_terms_l98_98405

-- Define the geometric sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first n terms of a sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Given conditions
def c1 (a : ℕ → ℝ) : Prop := geometric_sequence a 2
def c2 (a : ℕ → ℝ) : Prop := sum_of_first_n_terms a 4 = 1

-- The statement to prove
theorem sum_of_first_8_terms (a : ℕ → ℝ) (h1 : c1 a) (h2 : c2 a) : sum_of_first_n_terms a 8 = 17 :=
by
  sorry

end sum_of_first_8_terms_l98_98405


namespace num_ways_528_as_sum_of_consecutive_integers_l98_98005

theorem num_ways_528_as_sum_of_consecutive_integers :
  (∃ (n : ℕ), ∑ i in finset.range n, (i + 1)) = 528 → 
  (f : ℕ → ℕ) (2 ≤ n ∧ ∃ k, 528 = n * k ∧ (k >= n ∧ k - (n * (n - 1) / 2) = 0)) → 
  (count n = 13) :=
by 
  sorry

end num_ways_528_as_sum_of_consecutive_integers_l98_98005


namespace length_of_floor_X_l98_98059

-- Definitions of the conditions
variable (L : ℝ)
variable (widthX : ℝ) := 10
variable (widthY : ℝ) := 9
variable (lengthY : ℝ) := 20
variable (areaX : ℝ) := widthX * L
variable (areaY : ℝ) := widthY * lengthY
variable (equal_area : areaX = areaY)

-- Statement to prove
theorem length_of_floor_X : L = 18 := by
  have areaY_calculation : areaY = 9 * 20 := rfl
  have areaX_calculation : areaX = 10 * L := rfl
  have equal_areas : 10 * L = 180 := by
    rw [areaX_calculation, areaY_calculation] at equal_area
    assumption
  have solve_for_L : L = 180 / 10 := by
    rw ←equal_areas
    norm_num
  exact solve_for_L

end length_of_floor_X_l98_98059


namespace quadrilateral_centroid_area_l98_98496

theorem quadrilateral_centroid_area :
  (∃ (W X Y Z Q : Type)
    (side_length : ℝ) (WQ XQ : ℝ)
    (is_square : WXYZ_square W X Y Z side_length)
    (point_inside_square : Q_inside_square Q W X Y Z),
    WQ = 15 ∧ XQ = 35 ∧ side_length = 40 → 
    area_centroids_quadrilateral W X Y Z Q = 355.56)
:= sorry

end quadrilateral_centroid_area_l98_98496


namespace fraction_geq_81_l98_98488

theorem fraction_geq_81 {p q r s : ℝ} (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) :
  ((p^2 + p + 1) * (q^2 + q + 1) * (r^2 + r + 1) * (s^2 + s + 1)) / (p * q * r * s) ≥ 81 :=
by
  sorry

end fraction_geq_81_l98_98488


namespace AH_over_CH_eq_1_over_sqrt_2_l98_98030

noncomputable def AH_div_CH {A B C H : Type} [IsRightAngledTriangle A B C AC] 
                              (H_foot : IsFootOfAltitude B H AC)
                              (right_triangle : IsRightAngledTriangle AB BC BH) : ℝ :=
  if H : ∃ (AH CH : ℝ), True then
    Classical.choose (H)
  else 0

theorem AH_over_CH_eq_1_over_sqrt_2 {A B C H : Type} [IsRightAngledTriangle A B C AC] 
                                   (H_foot : IsFootOfAltitude B H AC)
                                   (right_triangle : IsRightAngledTriangle AB BC BH) :
  AH_div_CH H_foot right_triangle = 1 / √2 := by
  sorry

end AH_over_CH_eq_1_over_sqrt_2_l98_98030


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98233

variables (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1)

-- Definition for the fair coin case
def fair_coin_even_heads_probability (n : ℕ) : ℝ :=
  0.5

-- Definition for the biased coin case
def biased_coin_even_heads_probability (n : ℕ) (p : ℝ) : ℝ :=
  (1 + (1 - 2 * p)^n) / 2

-- Theorems/statements to prove
theorem fair_coin_even_heads (n : ℕ) :
  fair_coin_even_heads_probability n = 0.5 :=
sorry

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  biased_coin_even_heads_probability n p = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98233


namespace length_of_PC_l98_98795

theorem length_of_PC
  (A B C P : Type)
  (AB BC CA PA PC : ℕ)
  (h1 : AB = 10)
  (h2 : BC = 9)
  (h3 : CA = 8)
  (h4 : similar (triangle P A B) (triangle P C A)) : PC = 16 :=
by
  sorry

end length_of_PC_l98_98795


namespace probability_inside_sphere_l98_98611

noncomputable def volume_cube : ℝ := 64

noncomputable def volume_sphere : ℝ := (4 * π * 8) / 3

theorem probability_inside_sphere : 
  volume_sphere / volume_cube = π / 6 :=
by 
  sorry

end probability_inside_sphere_l98_98611


namespace problem_equivalent_proof_l98_98315

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98315


namespace find_a7_l98_98284

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98284


namespace puppies_sold_in_one_day_l98_98985

theorem puppies_sold_in_one_day : 
  (total_puppies initial_left caged_cages puppies_per_cage : ℕ) :
  total_puppies = 78 →
  caged_cages = 6 →
  puppies_per_cage = 8 →
  initial_left = total_puppies - (caged_cages * puppies_per_cage) →
  initial_left + (caged_cages * puppies_per_cage) = total_puppies →
  initial_left = 30 :=
by
  intros;
  sorry

end puppies_sold_in_one_day_l98_98985


namespace part1_part2_max_area_l98_98421

noncomputable def triangle := {a b c : ℝ} -- sides of the triangle
noncomputable def angles := {A B C : ℝ} -- angles of the triangle

noncomputable def condition1 (a b c A B: ℝ) : Prop :=
  2 * c - b = a * (cos B / cos A)

noncomputable def condition2 (a : ℝ) : Prop :=
  a = 2 * real.sqrt 5

theorem part1 (a b c A B C: ℝ) (h1 : condition1 a b c A B) : A = π / 3 :=
sorry

theorem part2_max_area (a b c A B C : ℝ) (h1 : condition1 a b c A B) (h2 : condition2 a) : 
  1/2 * b * c * real.sin A <= 5 * real.sqrt 3 :=
sorry

end part1_part2_max_area_l98_98421


namespace frog_jumps_further_l98_98080

-- Definitions according to conditions
def grasshopper_jump : ℕ := 36
def frog_jump : ℕ := 53

-- Theorem: The frog jumped 17 inches farther than the grasshopper
theorem frog_jumps_further (g_jump f_jump : ℕ) (h1 : g_jump = grasshopper_jump) (h2 : f_jump = frog_jump) :
  f_jump - g_jump = 17 :=
by
  -- Proof is skipped in this statement
  sorry

end frog_jumps_further_l98_98080


namespace prob_three_cards_sequence_l98_98103

def fraction (num : ℕ) (den : ℕ) : ℚ := num / den

theorem prob_three_cards_sequence :
  let p1 := fraction 13 52 in
  let p2 := fraction 13 51 in
  let p3 := fraction 13 50 in
  p1 * p2 * p3 = fraction 2197 132600 := by
  sorry

end prob_three_cards_sequence_l98_98103


namespace ratio_of_u_to_v_l98_98540

theorem ratio_of_u_to_v {b u v : ℝ} 
  (h1 : b ≠ 0)
  (h2 : 0 = 12 * u + b)
  (h3 : 0 = 8 * v + b) : 
  u / v = 2 / 3 := 
by
  sorry

end ratio_of_u_to_v_l98_98540


namespace prove_inequality_l98_98515

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end prove_inequality_l98_98515


namespace range_of_a_l98_98682

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, (x > 0) → (y > 0) → (y / 4 - cos x ^ 2 ≥ a * sin x - 9 / y)) ↔ (-3 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l98_98682


namespace find_principal_sum_l98_98574

theorem find_principal_sum (R P : ℝ) 
  (h1 : (3 * P * (R + 1) / 100 - 3 * P * R / 100) = 72) : 
  P = 2400 := 
by 
  sorry

end find_principal_sum_l98_98574


namespace quadrilateral_offset_l98_98672

-- Define the problem statement
theorem quadrilateral_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h₀ : d = 10) 
  (h₁ : y = 3) 
  (h₂ : A = 50) :
  x = 7 :=
by
  -- Assuming the given conditions
  have h₃ : A = 1/2 * d * x + 1/2 * d * y :=
  by
    -- specific formula for area of the quadrilateral
    sorry
  
  -- Given A = 50, d = 10, y = 3, solve for x to show x = 7
  sorry

end quadrilateral_offset_l98_98672


namespace triangle_properties_l98_98794

theorem triangle_properties (A B C a b c : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π)
  (h7 : a = sin A) (h8 : b = sin B) (h9 : c = sin C) 
  (h10 : cos A ⋅ cos B = a * (2 * c - b)) : (A = π / 3) ∧ ((sin B + sin C = sqrt 3) ∧ (A = B ∧ B = C)) := by
  sorry

end triangle_properties_l98_98794


namespace nearest_sum_of_g_equals_2023_l98_98977

noncomputable def g (x : ℝ) : ℝ := (18 * x + 6 - 6 / x) / 8

theorem nearest_sum_of_g_equals_2023 :
  let T := 16178 / 18 in
  Int.nearest T = 900 := by
  sorry

end nearest_sum_of_g_equals_2023_l98_98977


namespace interior_diagonal_length_l98_98096

theorem interior_diagonal_length (x y z : ℝ) (h1 : 2 * (x * y + y * z + z * x) = 62)
  (h2 : 4 * (x + y + z) = 48) : sqrt (x^2 + y^2 + z^2) = sqrt 82 :=
by sorry

end interior_diagonal_length_l98_98096


namespace circle_line_distance_l98_98389

theorem circle_line_distance (r : ℝ) (h : r > 0) :
  (∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    (p1.1^2 + p1.2^2 = r^2) ∧ (p2.1^2 + p2.2^2 = r^2) ∧
    (distance_from_line (4, -3, 25) p1 = 1) ∧
    (distance_from_line (4, -3, 25) p2 = 1)) ↔ 4 < r ∧ r < 6 :=
by
  sorry

noncomputable def distance_from_line (A B C : ℝ) (p : ℝ × ℝ) : ℝ :=
  abs (A * p.1 + B * p.2 + C) / sqrt (A^2 + B^2)

end circle_line_distance_l98_98389


namespace cannot_achieve_arithmetic_means_cannot_achieve_geometric_means_l98_98652

theorem cannot_achieve_arithmetic_means :
  ∀ (l : List ℕ), (l = [1, 3, 6, 8, 11]) →
    (¬ ∃ (steps : ℕ) (f : ℕ → List ℕ → List ℕ),
      (∀ n x, (x = [2, 4, 5, 7, 9]) →
        (l = [2, 4, 5, 7, 9]))) := by
sorry

theorem cannot_achieve_geometric_means :
  ∀ (l : List ℕ), (l = [1, 3, 6, 8, 11]) →
    (¬ ∃ (steps : ℕ) (f : ℕ → List ℕ → List ℕ),
      (∀ n x, (x = [2, 4, 5, 7, 9]) →
        (l = [2, 4, 5, 7, 9]))) := by
sorry

end cannot_achieve_arithmetic_means_cannot_achieve_geometric_means_l98_98652


namespace find_value_in_geometric_sequence_l98_98817

noncomputable def geometric_sequence : Prop :=
  let a : ℕ → ℝ := sorry in  -- Assume a is the geometric sequence
  a 1 + a 3 = 8 ∧
  a 5 + a 7 = 4 ∧
  a 9 + a 11 + a 13 + a 15 = 3

theorem find_value_in_geometric_sequence : geometric_sequence := by
  sorry

end find_value_in_geometric_sequence_l98_98817


namespace Maria_coffee_order_l98_98460

variable (visits_per_day : ℕ) (cups_per_visit : ℕ)

theorem Maria_coffee_order (h1 : visits_per_day = 2) (h2 : cups_per_visit = 3) :
  (visits_per_day * cups_per_visit) = 6 := by
  rw [h1, h2]
  exact rfl

end Maria_coffee_order_l98_98460


namespace common_ratio_of_b_sequence_is_3_2_l98_98449

variable {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n, a (n + 1) = r * a n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (1 - q ^ (n + 1)) / (1 - q)

def b_sequence (S : ℕ → ℝ) (b : ℕ → ℝ) : Prop := ∀ n, b n = S n + 2

theorem common_ratio_of_b_sequence_is_3_2
  (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : a 0 = 1)
  (h2 : geometric_sequence a q)
  (h3 : sum_of_first_n_terms a S)
  (h4 : b_sequence S b)
  (h5 : geometric_sequence b q) :
  q = 3 / 2 :=
sorry

end common_ratio_of_b_sequence_is_3_2_l98_98449


namespace find_B_value_l98_98940

theorem find_B_value (A B : ℕ) : (A * 100 + B * 10 + 2) - 41 = 591 → B = 3 :=
by
  sorry

end find_B_value_l98_98940


namespace perfect_square_factors_count_l98_98755

def perfectSquares := [4, 9, 16, 25, 36, 49, 64, 81]

def countNumbersWithPerfectSquareFactors : Nat :=
  List.length (List.filter (fun n => perfectSquares.any (fun p => n % p = 0)) [1..100])

theorem perfect_square_factors_count :
  countNumbersWithPerfectSquareFactors = 41 := sorry

end perfect_square_factors_count_l98_98755


namespace unique_monic_polynomial_condition_l98_98210

def poly_condition (f : ℤ[X]) (N : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → 0 < f.eval p → p ∣ (2 * (nat.factorial (f.eval p)) + 1)

theorem unique_monic_polynomial_condition :
  ∃! (f : ℤ[X]), (f.leadingCoeff = 1) ∧ (∃ N : ℕ, poly_condition f N) ∧ (f = X - 3) :=
begin
  sorry
end

end unique_monic_polynomial_condition_l98_98210


namespace find_speed_of_car_y_l98_98203

noncomputable def average_speed_of_car_y (sₓ : ℝ) (delay : ℝ) (d_afterₓ_started : ℝ) : ℝ :=
  let tₓ_before := delay
  let dₓ_before := sₓ * tₓ_before
  let total_dₓ := dₓ_before + d_afterₓ_started
  let tₓ_after := d_afterₓ_started / sₓ
  let total_time_y := tₓ_after
  d_afterₓ_started / total_time_y

theorem find_speed_of_car_y (h₁ : ∀ t, t = 1.2) (h₂ : ∀ sₓ, sₓ = 35) (h₃ : ∀ d_afterₓ_started, d_afterₓ_started = 42) : 
  average_speed_of_car_y 35 1.2 42 = 35 := by
  unfold average_speed_of_car_y
  simp
  sorry

end find_speed_of_car_y_l98_98203


namespace cost_per_ball_after_discounts_l98_98851

theorem cost_per_ball_after_discounts (cost_per_pack : ℝ) (balls_per_pack : ℕ) (packs_bought : ℕ) 
  (bulk_discount_rate : ℝ) (sales_tax_rate : ℝ) (currency_conversion_rate : ℝ) :
  cost_per_pack = 6 → 
  balls_per_pack = 3 →
  packs_bought = 4 →
  bulk_discount_rate = 0.1 →
  sales_tax_rate = 0.08 →
  currency_conversion_rate = 0.05 →
  (let total_cost_before_discount := packs_bought * cost_per_pack in
   let discount := bulk_discount_rate * total_cost_before_discount in
   let discounted_total := total_cost_before_discount - discount in
   let sales_tax := sales_tax_rate * discounted_total in
   let total_with_tax := discounted_total + sales_tax in
   let currency_conversion_fee := currency_conversion_rate * total_with_tax in
   let final_total := total_with_tax + currency_conversion_fee in
   let total_balls := packs_bought * balls_per_pack in
   final_total / total_balls = 2.0412) :=
sorry

end cost_per_ball_after_discounts_l98_98851


namespace plane_divides_AC_l98_98610

open Set

structure Pyramid :=
(A B C D M N K : Point)

variable (P: Pyramid)

-- Definitions of midpoints and ratio conditions
def midpoint_AB (p : Pyramid) : Prop := p.M = midpoint p.A p.B
def midpoint_CD (p : Pyramid) : Prop := p.N = midpoint p.C p.D
def ratio_BD (p : Pyramid) : Prop := ratio p.B p.K p.D = 1 / 3

def divides_AC_in_ratio (p : Pyramid) : Prop :=
  ∃ (L : Point), L ∈ line p.A p.C ∧ ratio p.A L p.C = 1 / 3

-- The theorem to prove
theorem plane_divides_AC (p : Pyramid) 
  (h1 : midpoint_AB p)
  (h2 : midpoint_CD p)
  (h3 : ratio_BD p):
  divides_AC_in_ratio p := 
sorry

end plane_divides_AC_l98_98610


namespace find_a7_l98_98274

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98274


namespace problem_equivalent_proof_l98_98311

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98311


namespace find_lesser_fraction_l98_98912

-- Define the conditions of the problem
def sum_of_fractions (x y : ℚ) : Prop := x + y = 5 / 6
def product_of_fractions (x y : ℚ) : Prop := x * y = 1 / 8

-- Define what it means to be the lesser fraction
def lesser_fraction (x y : ℚ) (l : ℚ) : Prop := (x < y ∧ l = x) ∨ (y < x ∧ l = y)

-- Define the goal of the proof
theorem find_lesser_fraction : 
  ∃ x y : ℚ, sum_of_fractions x y ∧ product_of_fractions x y ∧ lesser_fraction x y (5 - real.sqrt 7 / 12) := sorry

end find_lesser_fraction_l98_98912


namespace domain_of_h_l98_98212

open Real

def h (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 3) / (|x^2 - 9| + |x - 2|^2)

theorem domain_of_h : ∀ x : ℝ, |x^2 - 9| + |x - 2|^2 ≠ 0 :=
by {
  intro x,
  have h1 : 0 ≤ |x^2 - 9|,
  { exact abs_nonneg _ },
  have h2 : 0 ≤ |x - 2|^2,
  { exact pow_two_nonneg _ },
  by_contradiction H,
  have : |x^2 - 9| = 0 ∧ |x - 2|^2 = 0,
  { rw← add_eq_zero_iff at H,
    exact H, },
  cases this with h3 h4,
  {
    rw abs_eq_zero at h3,
    rw pow_eq_zero_iff at h4,
    cases h4 with h4_1 h4_2,
    rw h3 at h4_1,
    rw h3 at h4_2,
    interval_cases x;
      norm_num at h1 h2 h3 h4_1 h4_2 *,
  }
}
sorry

end domain_of_h_l98_98212


namespace find_a7_l98_98276

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98276


namespace pos_operations_closed_l98_98566

theorem pos_operations_closed (a b : ℝ) (n : ℝ) (h : a > 0) (k : b > 0) (m : n > 0) : 
  (a + b > 0) ∧ 
  (a * b > 0) ∧ 
  (a / b > 0) ∧ 
  (a^n > 0) ∧ 
  (n.root a > 0) :=
by
  sorry

end pos_operations_closed_l98_98566


namespace sequence_sum_2023_l98_98532

theorem sequence_sum_2023 (initial_sequence : List ℤ) (n : ℕ) :
  initial_sequence = [7, 3, 5] →
  n = 2023 →
  (∀ s : List ℤ, ∀ k : ℕ, 
       k > 0 →
       (∀ i : ℕ, i < s.length - 1 → s.get i - s.get (i + 1) = (s.operation k).get (2 * i + 1))) →
  (initial_sequence.sum + n * 2 = 4061) :=
sorry

end sequence_sum_2023_l98_98532


namespace column_of_2008_l98_98359

theorem column_of_2008:
  (∃ k, 2008 = 2 * k) ∧
  ((2 % 8) = 2) ∧ ((4 % 8) = 4) ∧ ((6 % 8) = 6) ∧ ((8 % 8) = 0) ∧
  ((16 % 8) = 0) ∧ ((14 % 8) = 6) ∧ ((12 % 8) = 4) ∧ ((10 % 8) = 2) →
  (2008 % 8 = 4) :=
by
  sorry

end column_of_2008_l98_98359


namespace rate_of_current_l98_98608

theorem rate_of_current (c : ℝ) (h1 : ∀ d : ℝ, d / (3.9 - c) = 2 * (d / (3.9 + c))) : c = 1.3 :=
sorry

end rate_of_current_l98_98608


namespace pencils_difference_l98_98023

theorem pencils_difference (p : ℝ) (hp : p > 0.01) 
  (hj : ∃ n, ((2.32 : ℝ) / p = n ^ 2 ∧ n : ℕ ^ 2))
  (hm : 3.24 / p = 81) :
  ((3.24 - 2.32) / p = 23) :=
by
  sorry

end pencils_difference_l98_98023


namespace product_major_minor_axis_l98_98052

theorem product_major_minor_axis (O A B C D F : Point)
  (h1 : center O (ellipse (segment A B) (segment C D)))
  (h2 : focus F (ellipse (segment A B) (segment C D)))
  (h3 : distance O F = 5)
  (h4 : incircle_diameter (triangle O C F) = 3) :
  (segment_length A B) * (segment_length C D) = 152.25 :=
sorry

end product_major_minor_axis_l98_98052


namespace fourth_square_area_l98_98116

theorem fourth_square_area (A B C D : Type) 
  (ab ac ad bc cd : ℝ) 
  (h_squares : ab^2 = 36 ∧ bc^2 = 9 ∧ cd^2 = 16 ∧ ab * bc * cd ≠ 0)
  : ad^2 = 61 := 
by
  -- Define the right triangles and their properties
  have h1 : ac^2 = ab^2 + bc^2, from sorry,
  have h2 : ad^2 = ac^2 + cd^2, from sorry,
  -- Calculate the areas and prove the square of the desired side.
  calc
    ad^2 = (ab^2 + bc^2) + cd^2 : by rw [h1, h2]
    ...   = 36 + 9 + 16       : by simp [h_squares]
    ...   = 61 : by norm_num

end fourth_square_area_l98_98116


namespace weight_of_new_man_l98_98394

/-
  We have the following initial conditions:
  1. The boat initially has 25 oarsmen.
  2. Initial average weight of oarsmen is 75 kg.
  3. One crew member weighs 55 kg.
  4. Replacing a crew member increases the average weight by 3 kg.
  
  The goal is to prove that the weight of the new man is 130 kg.
-/

theorem weight_of_new_man :
  ∀ (initial_average_weight : ℕ) (increase_in_average : ℕ) (replaced_member_weight : ℕ) 
    (number_of_oarsmen : ℕ), 
  initial_average_weight = 75 → increase_in_average = 3 → replaced_member_weight = 55 → number_of_oarsmen = 25 → 
  ∃ (new_man_weight : ℕ), 
  new_man_weight = 130 := 
by
  intros initial_average_weight increase_in_average replaced_member_weight number_of_oarsmen
  intro h1 h2 h3 h4
  use 130
  sorry

end weight_of_new_man_l98_98394


namespace min_coins_for_1_50_l98_98544

def CoinType : Type := 
| penny 
| dime 
| quarter 
| half_dollar

structure Coins where
  pennies : Nat
  dimes : Nat
  quarters : Nat
  half_dollars : Nat

noncomputable def total_coins (c : Coins) : Nat :=
  c.pennies + c.dimes + c.quarters + c.half_dollars

noncomputable def total_value (c : Coins) : Real :=
  c.pennies * 0.01 + c.dimes * 0.10 + c.quarters * 0.25 + c.half_dollars * 0.50

theorem min_coins_for_1_50 (c : Coins) (h : total_value c < 1.50) : total_coins c = 14 :=
  sorry

end min_coins_for_1_50_l98_98544


namespace second_container_mass_l98_98969

-- Given conditions
def height1 := 4 -- height of first container in cm
def width1 := 2 -- width of first container in cm
def length1 := 8 -- length of first container in cm
def mass1 := 64 -- mass of material the first container can hold in grams

def height2 := 3 * height1 -- height of second container in cm
def width2 := 2 * width1 -- width of second container in cm
def length2 := length1 -- length of second container in cm

def volume (height width length : ℤ) : ℤ := height * width * length

-- The proof statement
theorem second_container_mass : volume height2 width2 length2 = 6 * volume height1 width1 length1 → 6 * mass1 = 384 :=
by
  sorry

end second_container_mass_l98_98969


namespace inverse_proportion_l98_98524

theorem inverse_proportion (a : ℝ) (b : ℝ) (k : ℝ) : 
  (a = k / b^2) → 
  (40 = k / 12^2) → 
  (a = 10) → 
  b = 24 := 
by
  sorry

end inverse_proportion_l98_98524


namespace sum_of_x_coordinates_l98_98476

-- Definitions based on the conditions provided in the problem
variables {a b : ℕ}
def x1 : ℚ := -2 / a
def x2 : ℚ := -b / 2

-- The main theorem stating the equivalence of the conditions and the resulting sum of x-coordinates
theorem sum_of_x_coordinates (h_ab : a * b = 4) (ha_pos : a > 0) (hb_pos : b > 0) :
  x1 + x2 = -7 / 2 := by
sorry

end sum_of_x_coordinates_l98_98476


namespace count_perfect_square_factors_l98_98758

open Finset

noncomputable def count_divisible_by (n : ℕ) (s : Finset ℕ) : ℕ :=
s.filter (λ x, x % n = 0).card

theorem count_perfect_square_factors :
  let S := (range 100).map (λ n, n + 1)
  let perfect_squares := [4, 9, 16, 25, 36, 49, 64, 81, 100] 
  let total := S.card 
  let count_4 := count_divisible_by 4 S
  let count_9 := count_divisible_by 9 S
  let count_16 := count_divisible_by 16 S
  let count_25 := count_divisible_by 25 S
  let count_36 := count_divisible_by 36 S
  let count_49 := count_divisible_by 49 S
  let count_64 := count_divisible_by 64 S
  let count_81 := count_divisible_by 81 S
  let count_100 := count_divisible_by 100 S
  count_4 + (count_9 - count_divisible_by (Nat.lcm 4 9) S) +
    (count_16 - count_divisible_by 4 S) +
    (count_25 - count_divisible_by (Nat.lcm 4 25) S) +
    (count_36 - count_divisible_by 4 S) +
    count_49 + (count_64 - count_divisible_by 4 S) +
    (count_81 - count_divisible_by (Nat.lcm 9 81) S) +
    (count_100 - count_divisible_by 4 S)
    = 40 := 
by
  sorry

end count_perfect_square_factors_l98_98758


namespace number_of_chickens_free_ranging_l98_98526

-- Defining the conditions
def chickens_in_coop : ℕ := 14
def chickens_in_run (coop_chickens : ℕ) : ℕ := 2 * coop_chickens
def chickens_free_ranging (run_chickens : ℕ) : ℕ := 2 * run_chickens - 4

-- Proving the number of chickens free ranging
theorem number_of_chickens_free_ranging : chickens_free_ranging (chickens_in_run chickens_in_coop) = 52 := by
  -- Lean will be able to infer
  sorry  -- proof is not required

end number_of_chickens_free_ranging_l98_98526


namespace increase_350_by_50_percent_is_525_l98_98963

theorem increase_350_by_50_percent_is_525 (original_number : ℕ) (percentage_increase : ℝ) (h1 : original_number = 350) (h2 : percentage_increase = 0.5) : 
  original_number + (original_number * percentage_increase) = 525 :=
by
  rw [h1, h2]
  norm_num
  sorry

end increase_350_by_50_percent_is_525_l98_98963


namespace log_base_two_implication_l98_98383

theorem log_base_two_implication {a : ℕ} (h : log 2 (a + 2) = 2) : 3 ^ a = 9 :=
by {
    sorry
}

end log_base_two_implication_l98_98383


namespace symmetric_points_origin_l98_98712

theorem symmetric_points_origin (a b : ℤ) (h1 : a = -5) (h2 : b = -1) : a - b = -4 :=
by
  sorry

end symmetric_points_origin_l98_98712


namespace smallest_multiple_l98_98555

theorem smallest_multiple (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 45 = 0 ∧ m % 60 = 0 ∧ m % 25 ≠ 0 ∧ m = n) → n = 180 :=
by
  sorry

end smallest_multiple_l98_98555


namespace max_product_two_integers_l98_98930

theorem max_product_two_integers (x : ℝ) (h : 0 ≤ x ∧ x ≤ 320) : 
  (x * (320 - x) ≤ 25600) :=
begin
  calc
    x * (320 - x) ≤ 25600 : sorry
end

end max_product_two_integers_l98_98930


namespace laptop_final_price_l98_98605

theorem laptop_final_price :
  let original_price := 1000.00
  let discount1 := 0.10
  let discount2 := 0.20
  let discount3 := 0.15
  let final_price := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  in final_price = 612.00 :=
by
  let original_price := 1000.00
  let discount1 := 0.10
  let discount2 := 0.20
  let discount3 := 0.15
  let final_price := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  sorry

end laptop_final_price_l98_98605


namespace minimum_area_ratio_l98_98935

/-- Define geometric properties for isosceles right triangles --/
structure IsoscelesRightTriangle (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C] :=
  (a b : ℝ)  -- lengths of the legs
  (right_angle: 45° + 45° + 90° = 180°)

def area (T : IsoscelesRightTriangle) : ℝ := T.a * T.b / 2

/-- Prove the smallest ratio of areas of two isosceles right triangles,
where the three vertices of the smaller lie on different sides of the larger,
is 1/5. --/
theorem minimum_area_ratio (T1 T2 : IsoscelesRightTriangle) 
  (h_condition : ∀ (v1 v2 v3 : ℝ), {v1, v2, v3} ⊆ {T1.a, T1.b, T1.a*sqrt(2), T1.b*sqrt(2)} → 
   v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3) :
area T1 / area T2 ≥ 1/5 :=
sorry

end minimum_area_ratio_l98_98935


namespace prism_surface_area_volume_eq_l98_98988

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem prism_surface_area_volume_eq (x : ℝ) (hx : 0 < x) (hx1 : x ≠ 1) :
  let a := log_base 3 x,
      b := log_base 5 x,
      c := log_base 6 x,
      SA := 2 * (a * b + b * c + c * a),
      V := a * b * c in
    SA = V → x = 8100 :=
by
  intros a b c SA V h
  sorry

end prism_surface_area_volume_eq_l98_98988


namespace b_share_l98_98143

-- Definitions based on the conditions
def salary (a b c d : ℕ) : Prop :=
  ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x ∧ d = 6 * x

def condition (d c : ℕ) : Prop :=
  d = c + 700

-- Proof problem based on the correct answer
theorem b_share (a b c d : ℕ) (x : ℕ) (salary_cond : salary a b c d) (cond : condition d c) :
  b = 1050 := by
  sorry

end b_share_l98_98143


namespace B_is_Brownian_bridge_l98_98840

noncomputable def B (X : ℝ → ℝ) (t : ℝ) : ℝ :=
if (0 < t ∧ t < 1) then (Real.sqrt (t * (1 - t)) * X (1 / 2) * Real.log (t / (1 - t))) else 0

theorem B_is_Brownian_bridge (X : ℝ → ℝ) (hOU : is_Ornstein_Uhlenbeck X) :
  is_Brownian_bridge (B X) := sorry

end B_is_Brownian_bridge_l98_98840


namespace count_numbers_with_square_factors_l98_98768

theorem count_numbers_with_square_factors :
  let squares := [4, 9, 16, 25, 36, 49, 64]
  let multiples (n : ℕ) := ∀ k ∈ squares, n % k = 0
  let count_multiples (n : ℕ) := (1..100).count multiples
  count_multiples squares = 48 :=
  sorry

end count_numbers_with_square_factors_l98_98768


namespace smallest_positive_integer_l98_98554

theorem smallest_positive_integer (n : ℕ) :
  (n % 45 = 0 ∧ n % 60 = 0 ∧ n % 25 ≠ 0 ↔ n = 180) :=
sorry

end smallest_positive_integer_l98_98554


namespace sufficient_but_not_necessary_condition_l98_98341

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h : |b| + a < 0) : b^2 < a^2 :=
  sorry

end sufficient_but_not_necessary_condition_l98_98341


namespace telepathic_connection_probability_l98_98115

theorem telepathic_connection_probability :
  let S := Finset.range 10,
      T := Finset.product S S,
      favorable := Finset.filter (λ (ab : ℕ × ℕ), abs (ab.1 - ab.2) = 1) T in
  (favorable.card : ℚ) / T.card = 9 / 50 :=
by
  let S := Finset.range 10
  let T := Finset.product S S
  let favorable := Finset.filter (λ (ab : ℕ × ℕ), abs (ab.1 - ab.2) = 1) T
  have favorable_card : favorable.card = 18 := sorry  -- Correctly identify 18 favorable outcomes
  have T_card : T.card = 100 := sorry  -- Total outcomes for pairs (a, b) where a, b ∈ {0, ..., 9}
  rw [favorable_card, T_card]
  norm_cast
  norm_num

end telepathic_connection_probability_l98_98115


namespace base_length_of_isosceles_triangle_l98_98879

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l98_98879


namespace have_inverses_l98_98560

noncomputable def a := λ x : ℝ, Real.sqrt (3 - x)
def a_domain := set.Iic 3

def b := λ x : ℝ, x^3 + x
def b_domain := set.univ

noncomputable def c := λ x : ℝ, x - (2 / x)
def c_domain := set.Ioi 0

def d := λ x : ℝ, 3 * x^2 + 6 * x + 11
def d_domain := set.Ici 0

noncomputable def e := λ x : ℝ, abs (x - 3) + abs (x + 4)
def e_domain := set.univ

noncomputable def f := λ x : ℝ, (2 : ℝ)^x + (8 : ℝ)^x
def f_domain := set.univ

noncomputable def g := λ x : ℝ, x + 1 / x
def g_domain := set.Ioi 0

def h := λ x : ℝ, x / 3
def h_domain := set.Ico (-3) 9

theorem have_inverses : 
  (∀ x1 x2 ∈ a_domain, a x1 = a x2 → x1 = x2) ∧ 
  (∀ x1 x2 ∈ d_domain, d x1 = d x2 → x1 = x2) ∧ 
  (∀ x1 x2 ∈ f_domain, f x1 = f x2 → x1 = x2) ∧ 
  (∀ x1 x2 ∈ g_domain, g x1 = g x2 → x1 = x2) ∧ 
  (∀ x1 x2 ∈ h_domain, h x1 = h x2 → x1 = x2) ∧ 
  ¬(∀ x1 x2 ∈ b_domain, b x1 = b x2 → x1 = x2) ∧ 
  ¬(∀ x1 x2 ∈ c_domain, c x1 = c x2 → x1 = x2) ∧ 
  ¬(∀ x1 x2 ∈ e_domain, e x1 = e x2 → x1 = x2) := 
sorry

end have_inverses_l98_98560


namespace find_a7_l98_98273

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98273


namespace derivative_at_0_5_l98_98320

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := -2

-- State the theorem
theorem derivative_at_0_5 : f' 0.5 = -2 :=
by {
  -- Proof placeholder
  sorry
}

end derivative_at_0_5_l98_98320


namespace sum_of_valid_n_l98_98678

theorem sum_of_valid_n :
  let valid_n (n : ℕ) := 2 * n - 3 ≠ 0 ∧ ∃ k : ℤ, 15 * (n.factorial ^ 2) + 1 = k * (2 * n - 3)
  in ∑ n in (Finset.filter valid_n (Finset.range 78)), n = 90 :=
by
  sorry

end sum_of_valid_n_l98_98678


namespace tens_digit_property_l98_98099

theorem tens_digit_property :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (∃ d1 d2 d3 : ℕ, d1 = n / 100 ∧ d2 = (n / 10) % 10 ∧ d3 = n % 10 ∧ 
  (d1 < d2 ∧ d3 < d2) ∧ count_three_digit_numbers_with_properties 120) :=
sorry

end tens_digit_property_l98_98099


namespace even_heads_probability_fair_even_heads_probability_biased_l98_98240

variable  {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1)

/-- The probability of getting an even number of heads -/

/-- Case 1: Fair coin (p = 1/2) -/
theorem even_heads_probability_fair (n : ℕ) : 
  let p : ℝ := 1/2 in 
  let q : ℝ := 1 - p in 
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 0.5 := sorry

/-- Case 2: Biased coin -/
theorem even_heads_probability_biased {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1) : 
  let q : ℝ := 1 - p in
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 
  (1 + (1 - 2 * p)^n) / 2 := sorry

end even_heads_probability_fair_even_heads_probability_biased_l98_98240


namespace f_passes_through_P_l98_98893

-- Declare the function f
def f (x : ℝ) : ℝ := 3^(x - 2)

-- State the theorem that we need to prove
theorem f_passes_through_P : f 2 = 1 :=
by
sorry

end f_passes_through_P_l98_98893


namespace inequality_holds_l98_98684

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x * y + y * z) := 
by 
  sorry

end inequality_holds_l98_98684


namespace daisy_lunch_vs_breakfast_spending_l98_98686

noncomputable def breakfast_cost : ℝ := 2 + 3 + 4 + 3.5 + 1.5
noncomputable def lunch_base_cost : ℝ := 3.5 + 4 + 5.25 + 6 + 1 + 3
noncomputable def service_charge : ℝ := 0.10 * lunch_base_cost
noncomputable def lunch_cost_with_service_charge : ℝ := lunch_base_cost + service_charge
noncomputable def food_tax : ℝ := 0.05 * lunch_cost_with_service_charge
noncomputable def total_lunch_cost : ℝ := lunch_cost_with_service_charge + food_tax
noncomputable def difference : ℝ := total_lunch_cost - breakfast_cost

theorem daisy_lunch_vs_breakfast_spending :
  difference = 12.28 :=
by 
  sorry

end daisy_lunch_vs_breakfast_spending_l98_98686


namespace expression_evaluation_l98_98638

theorem expression_evaluation : |(-7: ℤ)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end expression_evaluation_l98_98638


namespace no_factorization_of_f_l98_98843

noncomputable def f (x : ℤ) (n : ℤ) : ℤ := x^n + 5 * x^(n-1) + 3

theorem no_factorization_of_f (n : ℤ) (h : n > 1) : 
  ¬∃ (g h : polynomial ℤ), (∃ (deg_g deg_h : ℕ), deg_g ≥ 1 ∧ deg_h ≥ 1 ∧ g.degree = deg_g ∧ h.degree = deg_h) ∧ 
    (polynomial.eval₂ polynomial.C polynomial.X (f polynomial.X n) = g * h) :=
sorry

end no_factorization_of_f_l98_98843


namespace probability_commitee_boy_and_girl_l98_98874

theorem probability_commitee_boy_and_girl 
  (total_members : ℕ) 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (committee_size : ℕ)
  (h_total : total_members = 24)
  (h_boys : num_boys = 14)
  (h_girls : num_girls = 10)
  (h_comm_size : committee_size = 5) :
  let total_ways := Nat.choose total_members committee_size,
      all_boys_ways := Nat.choose num_boys committee_size,
      all_girls_ways := Nat.choose num_girls committee_size,
      invalid_ways := all_boys_ways + all_girls_ways,
      probability := 1 - invalid_ways / total_ways in
  probability = (4025 : ℚ) / 42504 :=
  by
  -- Proof is skipped.
  sorry

end probability_commitee_boy_and_girl_l98_98874


namespace smallest_T_value_l98_98125

theorem smallest_T_value : 
  ∃ (m : ℕ) (T : ℕ), (T = 9 * m - 2400) ∧ (2400 <= 8 * m) ∧ (T = 300) :=
by {
  use 300,
  use 300,
  split,
  sorry,
  split,
  sorry,
  sorry
}

end smallest_T_value_l98_98125


namespace number_of_distinct_values_l98_98939

theorem number_of_distinct_values : 
  ∃ (S : Set ℂ), (∀ (n : ℕ+), i^n + i^(-n) ∈ S) ∧ S = {0, 2, -2} ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_values_l98_98939


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98251

-- Problem 1: Fair coin, probability of even heads
def fair_coin_even_heads_prob (n : ℕ) : Prop :=
  0.5 = 0.5

-- Problem 2: Biased coin, probability of even heads
def biased_coin_even_heads_prob (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : Prop :=
  let q := 1 - p in
  (0.5 * (1 + (1 - 2 * p)^n) = (1 + (1 - 2*p)^n) / 2)

-- Mock proof to ensure Lean accepts the definitions
theorem fair_coin_even_heads (n : ℕ) : fair_coin_even_heads_prob n :=
begin
  -- Proof intentionally omitted
  sorry
end

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : biased_coin_even_heads_prob n p hp :=
begin
  -- Proof intentionally omitted
  sorry
end

end fair_coin_even_heads_biased_coin_even_heads_l98_98251


namespace servant_service_duration_l98_98980

variables (x : ℕ) (total_compensation full_months received_compensation : ℕ)
variables (price_uniform compensation_cash : ℕ)

theorem servant_service_duration :
  total_compensation = 1000 →
  full_months = 12 →
  received_compensation = (compensation_cash + price_uniform) →
  received_compensation = 750 →
  total_compensation = (compensation_cash + price_uniform) →
  x / full_months = 750 / total_compensation →
  x = 9 :=
by sorry

end servant_service_duration_l98_98980


namespace isosceles_triangle_base_length_l98_98887

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l98_98887


namespace ordered_pair_count_l98_98446

theorem ordered_pair_count :
  let ω := by { have h := z.pow(4) - 1; exact (h.roots.nonreal) } in
  (∃! (a b : ℤ), a^2 + b^2 = 1).card = 4 :=
sorry

end ordered_pair_count_l98_98446


namespace clubs_sum_l98_98656

def clubs (x : ℝ) : ℝ := (x + x^2 + x^3) / 3

theorem clubs_sum : clubs 1 + clubs 2 + clubs 3 + clubs 4 = 46 + 2 / 3 := 
by sorry

end clubs_sum_l98_98656


namespace sequence_part1_sequence_part2_l98_98699

theorem sequence_part1 :
  ∃ a1 a2 a3 : ℝ, a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧
  (a1 + a2 + a3)^2 = a1^3 + a2^3 + a3^3 ∧
  ((a1 = 1 ∧ a2 = 2 ∧ a3 = 3) ∨ 
   (a1 = 1 ∧ a2 = 2 ∧ a3 = -2) ∨ 
   (a1 = 1 ∧ a2 = -1 ∧ a3 = 1)) :=
by sorry

theorem sequence_part2 :
  ∃ (a : ℕ → ℝ), (∀ n, a n ≠ 0) ∧ 
  (∀ n, (finset.range n).sum (λ k, a k) ^ 2 = (finset.range n).sum (λ k, a k ^ 3)) ∧
  a 2013 = -2012 ∧ 
  (∀ n, n ≤ 2012 → a n = n) ∧ 
  (∀ n, n > 2012 → a n = 2012 * (-1)^(n+1)) :=
by sorry

end sequence_part1_sequence_part2_l98_98699


namespace length_of_brick_proof_l98_98973

noncomputable def length_of_brick (courtyard_length courtyard_width : ℕ) (brick_width : ℕ) (total_bricks : ℕ) : ℕ :=
  let total_area_cm := courtyard_length * courtyard_width * 10000
  total_area_cm / (brick_width * total_bricks)

theorem length_of_brick_proof :
  length_of_brick 25 16 10 20000 = 20 :=
by
  unfold length_of_brick
  sorry

end length_of_brick_proof_l98_98973


namespace fair_coin_even_heads_unfair_coin_even_heads_l98_98245

-- Define the probability function for an even number of heads for a fair coin
theorem fair_coin_even_heads (n : ℕ) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * 0.5^k * 0.5^(n-k) else 0) = 0.5 :=
sorry

-- Define the probability function for an even number of heads for an unfair coin
theorem unfair_coin_even_heads (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * p^k * (1-p)^(n-k) else 0) = 
    (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_unfair_coin_even_heads_l98_98245


namespace find_a7_l98_98293

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98293


namespace sufficient_but_not_necessary_condition_l98_98371

variables (a : ℝ)

def p := a > 2
def q := ¬(∀ x : ℝ, x^2 + a * x + 1 ≥ 0)

theorem sufficient_but_not_necessary_condition :
  (p → q) ∧ (¬(q → p)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l98_98371


namespace none_of_these_is_true_l98_98402

variable (O A B C D E P : Point)
variable (r : ℝ) -- radius of the circle
variable [Noncomputable] (AP pb AB AD DO DE OB AO r√2 : ℝ)

-- Conditions as given in the problem
variable (O_center : Center circle O)
variable (perp_AB_BC : Perpendicular AB BC)
variable (line_ADOE : Collinear [A, D, O, E])
variable (AP_eq_2AD : AP = 2 * AD)
variable (AB_eq_3r : AB = 3 * r)

-- Define additional variables as necessary
variable (AO_eq_2r√2 : AO = 2 * r * Real.sqrt 2)
variable (AD_eq_4r√2_3 : AD = 4 * r * Real.sqrt 2 / 3)
variable (PB_eq_3r_minus_8r√2_3 : PB = 3 * r - 8 * r * Real.sqrt 2 / 3)

-- Proving the equivalent proof problem:
theorem none_of_these_is_true
    (h1 : ¬ (AP^2 = PB * AB))
    (h2 : ¬ (AP * DO = PB * AD))
    (h3 : ¬ (AB^2 = AD * DE))
    (h4 : ¬ (AB * AD = OB * AO)) : 
    ¬ (AP^2 = PB * AB ∨ AP * DO = PB * AD ∨ AB^2 = AD * DE ∨ AB * AD = OB * AO) :=
sorry

end none_of_these_is_true_l98_98402


namespace volume_of_pyramid_l98_98617

-- Define the given conditions
def square_area : ℝ := 225
def triangle_abe_area : ℝ := 112.5
def triangle_cde_area : ℝ := 99

-- Define the side length of the base
def side_length (A_eq : square_area = 15 * 15) : ℝ := 15

-- Define the heights of the triangles
def height_abe (H_ABE_eq : triangle_abe_area = 1/2 * 15 * 15) : ℝ := 15
def height_cde (H_CDE_eq : triangle_cde_area = 1/2 * 15 * 13.2) : ℝ := 13.2

-- Define the height 'h' of the pyramid based on the given conditions
def pyramid_height (H_abe_eq : height_abe = 15) (H_cde_eq : height_cde = 13.2) : ℝ := 14.89

-- Main statement to be proven
theorem volume_of_pyramid (A_eq : square_area = 225) 
                           (H_ABE_eq : triangle_abe_area = 112.5) 
                           (H_CDE_eq : triangle_cde_area = 99) 
                           (H_eq : 15 * 15 = square_area) : 
    (1/3) * square_area * pyramid_height H_ABE_eq H_CDE_eq = 1117.375 := 
sorry

end volume_of_pyramid_l98_98617


namespace ac_length_l98_98144

theorem ac_length (a b c d e : ℝ)
  (h1 : b - a = 5)
  (h2 : c - b = 2 * (d - c))
  (h3 : e - d = 4)
  (h4 : e - a = 18) :
  d - a = 11 :=
by
  sorry

end ac_length_l98_98144


namespace average_score_l98_98111

variable (u v A : ℝ)
variable (h1 : v / u = 1/3)
variable (h2 : A = (u + v) / 2)

theorem average_score : A = (2/3) * u := by
  sorry

end average_score_l98_98111


namespace yan_distance_ratio_l98_98136

theorem yan_distance_ratio (w x y : ℝ) (h1 : w > 0) (h2 : x > 0) (h3 : y > 0)
(h4 : y / w = x / w + (x + y) / (5 * w)) : x / y = 2 / 3 :=
by
  sorry

end yan_distance_ratio_l98_98136


namespace find_a5_l98_98325

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2 + 1) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h3 : S 1 = 2) :
  a 5 = 9 :=
sorry

end find_a5_l98_98325


namespace trevor_quarters_counted_l98_98923

-- Define the conditions from the problem
variable (Q D : ℕ) 
variable (total_coins : ℕ := 77)
variable (excess : ℕ := 48)

-- Use the conditions to assert the existence of quarters and dimes such that the totals align with the given constraints
theorem trevor_quarters_counted : (Q + D = total_coins) ∧ (D = Q + excess) → Q = 29 :=
by
  -- Add sorry to skip the actual proof, as we are only writing the statement
  sorry

end trevor_quarters_counted_l98_98923


namespace inequality_proof_l98_98517

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l98_98517


namespace katya_notebooks_l98_98179

theorem katya_notebooks (rubles: ℕ) (cost_per_notebook: ℕ) (stickers_per_exchange: ℕ) 
  (initial_rubles: ℕ) (initial_notebooks: ℕ) :
  (initial_notebooks = initial_rubles / cost_per_notebook) →
  (rubles = initial_notebooks * cost_per_notebook) →
  (initial_notebooks = 37) →
  (initial_rubles = 150) →
  (cost_per_notebook = 4) →
  (stickers_per_exchange = 5) →
  (rubles = 148) →
  let rec total_notebooks (notebooks stickers : ℕ) : ℕ :=
      if stickers < stickers_per_exchange then notebooks
      else let new_notebooks := stickers / stickers_per_exchange in
           total_notebooks (notebooks + new_notebooks) 
                           (stickers % stickers_per_exchange + new_notebooks) in
  total_notebooks initial_notebooks initial_notebooks = 46 :=
begin
  sorry
end

end katya_notebooks_l98_98179


namespace zero_count_f_l98_98588

theorem zero_count_f (f : ℝ → ℝ) (c : ∃! x ∈ set.Ioo 0 1, f x = 0) : 
  ∃ x, x ∈ set.Ioo 0 1 ∧ f x = 0 :=
by
  let f : ℝ → ℝ := λ x, 4 * x - 3
  have h_continuous : continuous f := sorry
  have h_increasing : ∀ x y, x < y → f x < f y := sorry
  have h_f0 : f 0 = -3 := sorry
  have h_f1 : f 1 = 1 := sorry
  have zero_in_interval : ∃! x, x ∈ set.Ioo 0 1 ∧ f x = 0 := sorry
  exact zero_in_interval

end zero_count_f_l98_98588


namespace selling_price_of_cycle_l98_98570

def cost_price : ℝ := 1400
def loss_percentage : ℝ := 18

theorem selling_price_of_cycle : 
    (cost_price - (loss_percentage / 100) * cost_price) = 1148 := 
by
  sorry

end selling_price_of_cycle_l98_98570


namespace karlson_max_eat_chocolates_l98_98857

noncomputable def maximum_chocolates_eaten : ℕ :=
  34 * (34 - 1) / 2

theorem karlson_max_eat_chocolates : maximum_chocolates_eaten = 561 := by
  sorry

end karlson_max_eat_chocolates_l98_98857


namespace count_perfect_square_factors_l98_98762

open Nat

def has_larger_square_factor (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

theorem count_perfect_square_factors :
  (Finset.filter has_larger_square_factor (Finset.range 101)).card = 42 := by
sorry

end count_perfect_square_factors_l98_98762


namespace descending_order_l98_98319

theorem descending_order (a b c : ℝ) 
  (ha : a = (3 / 5 : ℝ) ^ (-1 / 3)) 
  (hb : b = (4 / 3 : ℝ) ^ (-1 / 2)) 
  (hc : c = Real.log (3 / 5)) : 
  a > b ∧ b > c := 
  by 
    sorry

end descending_order_l98_98319


namespace area_of_triangle_ABC_l98_98695

noncomputable def complex_area_of_triangle : ℂ → ℂ → ℂ → ℝ
| z1 z2 z3 := abs ((z1.re * z2.im + z2.re * z3.im + z3.re * z1.im) - (z1.im * z2.re + z2.im * z3.re + z3.im * z1.re)) / 2

theorem area_of_triangle_ABC (z : ℂ) 
  (h1 : abs z = real.sqrt 2)
  (h2 : z^2.im = 2) :
  (z = 1 + 1 * complex.I ∨ z = -1 - 1 * complex.I) ∧ 
  (complex_area_of_triangle z (z^2) (z - z^2) = 1) :=
by sorry

end area_of_triangle_ABC_l98_98695


namespace find_a7_l98_98264

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98264


namespace probability_jack_king_queen_l98_98531

theorem probability_jack_king_queen :
  let P := (4 / 52) * (4 / 51) * (4 / 50)
  in P = (8 / 16575) :=
by
  sorry

end probability_jack_king_queen_l98_98531


namespace max_elevator_distance_l98_98547

theorem max_elevator_distance :
  ∃ (floors : List ℕ), 
  (floors.length = 11) ∧ 
  (List.nodup floors) ∧ 
  (floors.head = 0) ∧ 
  (∀ i, i < 10 →
     (((floors.nth i).get % 2 = 0 → (floors.nth (i + 1)).get % 2 ≠ 0) ∧
      ((floors.nth i).get % 2 ≠ 0 → (floors.nth (i + 1)).get % 2 = 0))) ∧
  (let total_floors := Σ i in Finset.range 10, (floors.nth (i + 1)).get - (floors.nth i).get in
   total_floors * 4 = 216) :=
sorry

end max_elevator_distance_l98_98547


namespace exists_constant_C_l98_98454

theorem exists_constant_C (d : ℕ) (h_squarefree: ∀ m : ℕ, m^2 ∣ d → m = 1) : ∃ C : ℝ, ∀ N : ℕ, ∃ x y : ℤ, y > N ∧ |x^2 - d * y^2| < C :=
by 
  sorry

end exists_constant_C_l98_98454


namespace days_in_month_find_days_in_month_l98_98163

noncomputable def computers_per_thirty_minutes : ℕ := 225 / 100 -- representing 2.25
def monthly_computers : ℕ := 3024
def hours_per_day : ℕ := 24

theorem days_in_month (computers_per_hour : ℕ) (daily_production : ℕ) : ℕ :=
  let computers_per_hour := (2 * computers_per_thirty_minutes)
  let daily_production := (computers_per_hour * hours_per_day)
  (monthly_computers / daily_production)

theorem find_days_in_month :
  days_in_month (2 * computers_per_thirty_minutes) ((2 * computers_per_thirty_minutes) * hours_per_day) = 28 :=
by
  sorry

end days_in_month_find_days_in_month_l98_98163


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98229

variables (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1)

-- Definition for the fair coin case
def fair_coin_even_heads_probability (n : ℕ) : ℝ :=
  0.5

-- Definition for the biased coin case
def biased_coin_even_heads_probability (n : ℕ) (p : ℝ) : ℝ :=
  (1 + (1 - 2 * p)^n) / 2

-- Theorems/statements to prove
theorem fair_coin_even_heads (n : ℕ) :
  fair_coin_even_heads_probability n = 0.5 :=
sorry

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  biased_coin_even_heads_probability n p = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98229


namespace temperature_difference_l98_98897

def highest_temperature : ℤ := 8
def lowest_temperature : ℤ := -2

theorem temperature_difference :
  highest_temperature - lowest_temperature = 10 := by
  sorry

end temperature_difference_l98_98897


namespace cistern_fill_time_l98_98569

-- Let F be the rate at which the first tap fills the cistern (cisterns per hour)
def F : ℚ := 1 / 4

-- Let E be the rate at which the second tap empties the cistern (cisterns per hour)
def E : ℚ := 1 / 5

-- Prove that the time it takes to fill the cistern is 20 hours given the rates F and E
theorem cistern_fill_time : (1 / (F - E)) = 20 := 
by
  -- Insert necessary proofs here
  sorry

end cistern_fill_time_l98_98569


namespace geometric_seq_a7_l98_98307

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98307


namespace brick_length_is_20_cm_l98_98972

-- Define the conditions given in the problem
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def num_bricks : ℕ := 20000
def brick_width_cm : ℝ := 10
def total_area_cm2 : ℝ := 4000000

-- Define the goal to prove that the length of each brick is 20 cm
theorem brick_length_is_20_cm :
  (total_area_cm2 = num_bricks * (brick_width_cm * length)) → (length = 20) :=
by
  -- Assume the given conditions
  sorry

end brick_length_is_20_cm_l98_98972


namespace function_neither_even_nor_odd_l98_98820

def g (x : ℝ) : ℝ := log (x + sqrt (1 + x^3))

theorem function_neither_even_nor_odd :
  ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g (-x) = -g x) :=
by
  sorry

end function_neither_even_nor_odd_l98_98820


namespace greatest_additional_cars_l98_98209

variable (cars : Nat)
variable (rows : Nat)
variable (cars_per_row : Nat)

def additional_cars_needed (current_cars desired_total_cars : Nat) : Nat :=
  desired_total_cars - current_cars

theorem greatest_additional_cars (h : cars = 23) (h1 : cars_per_row = 6) :
  additional_cars_needed cars 24 = 1 :=
by
  rw [h, h1]
  have desired_total_cars := 24
  exact Nat.sub_eq_of_eq_add rfl
  sorry

end greatest_additional_cars_l98_98209


namespace part_one_p_q_part_two_a_b_part_three_l98_98865

theorem part_one_p_q (x1 x2 : ℝ) (h1 : x1 = -2) (h2 : x2 = 3) (h : ∀ x, x ≠ 0 → x + (p / x) = q → x = x1 ∨ x = x2) :
  p = -6 ∧ q = 1 := by
  sorry

theorem part_two_a_b (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -2) (h : ∀ x, x ≠ 0 → x + (-2 / x) = 3 → x = a ∨ x = b) :
  a^2 + b^2 = 13 := by
  sorry

theorem part_three (n : ℝ) (x1 x2 : ℝ) (h : ∀ x, 2x + (2n^2 + n) / (2x + 1) = 3n → x = x1 ∨ x = x2) (h_order : x1 < x2):
  (2 * x1 + 1) / (2 * x2 - 2) = n / (2 * n - 2) := by
  sorry

end part_one_p_q_part_two_a_b_part_three_l98_98865


namespace geometric_seq_a7_l98_98308

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98308


namespace rectangle_area_constant_k_l98_98088

theorem rectangle_area_constant_k (x : ℝ) (d : ℝ) (k : ℝ) 
  (h1 : d = 13) 
  (h2 : 25 * x * x + 4 * x * x = d * d)
  (h3 : k = 10 / 29) : 
  ∃ x : ℝ, x * (5 : ℝ) * x * (2 : ℝ) = k * d * d :=
begin
  sorry
end  

end rectangle_area_constant_k_l98_98088


namespace find_x_l98_98453

def h (x : ℝ) : ℝ := (2 * x + 5)^(1/3) / 5^(1/3)

theorem find_x : ∃ x : ℝ, h (3 * x) = 3 * h x ∧ x = -65 / 24 :=
by
  sorry

end find_x_l98_98453


namespace coffee_shop_visits_l98_98461

theorem coffee_shop_visits
  (visits_per_day : ℕ)
  (cups_per_visit : ℕ)
  (h1 : visits_per_day = 2)
  (h2 : cups_per_visit = 3) :
  visits_per_day * cups_per_visit = 6 :=
by
  rw [h1, h2]
  norm_num
  sorry

end coffee_shop_visits_l98_98461


namespace induction_step_even_l98_98343

theorem induction_step_even (P : ℕ → Prop) (h_base : P 2)
  (h_inductive : ∀ k : ℕ, k ≥ 2 → k % 2 = 0 → P k → P (k + 2)) :
  ∀ n : ℕ, n ≥ 2 → n % 2 = 0 → P n :=
begin
  assume n h_n_ge_2 h_n_even,
  sorry
end

end induction_step_even_l98_98343


namespace inequality_proof_l98_98519

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l98_98519


namespace find_a7_l98_98302

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98302


namespace geometric_sequence_sum_l98_98816

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, (r > 0) ∧ (∀ n : ℕ, a (n + 1) = a n * r)

theorem geometric_sequence_sum
  (a_seq_geometric : is_geometric_sequence a)
  (a_pos : ∀ n : ℕ, a n > 0)
  (eqn : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 :=
by
  sorry

end geometric_sequence_sum_l98_98816


namespace sum_and_product_of_primes_l98_98920

noncomputable def primes : List ℕ := [11, 13, 17, 19, 23, 29]

theorem sum_and_product_of_primes :
  ∀ (x y z : ℕ), x ∈ primes ∧ y ∈ primes ∧ z ∈ primes ∧ x < y ∧ y < z →
  let product_minus_sum := x * y * z - (x + y)
  in product_minus_sum ≠ 545 ∧ product_minus_sum ≠ 1862 ∧ product_minus_sum ≠ 3290 ∧ 
     product_minus_sum ≠ 5022 ∧ product_minus_sum ≠ 6890 :=
by
  intros x y z hx hy hz hxy hyz
  let product_minus_sum := x * y * z - (x + y)
  split_ifs
  · exactly sorry
  · exactly sorry
  · exactly sorry
  · exactly sorry
  · exactly sorry

end sum_and_product_of_primes_l98_98920


namespace exists_sum_diff_not_in_set_l98_98054

theorem exists_sum_diff_not_in_set :
  ∀ (a : Fin 25 → ℕ), 
    (∀ i j : Fin 25, i ≠ j → a i ≠ a j) →
    ∃ i j : Fin 25, i ≠ j ∧ 
      (∀ k : Fin 25, k ≠ i ∧ k ≠ j → (a k ≠ a i + a j ∧ a k ≠ (a i - a j).abs)) :=
by
  intro a h
  sorry

end exists_sum_diff_not_in_set_l98_98054


namespace problem_part_1_problem_part_2_problem_part_3_l98_98848

open Set

-- Definitions for the given problem conditions
def U : Set ℕ := { x | x > 0 ∧ x < 10 }
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6}
def D : Set ℕ := B ∩ C

-- Prove each part of the problem
theorem problem_part_1 :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

theorem problem_part_2 :
  D = {3, 4} ∧
  (∀ (s : Set ℕ), s ⊆ D ↔ s = ∅ ∨ s = {3} ∨ s = {4} ∨ s = {3, 4}) := by
  sorry

theorem problem_part_3 :
  (U \ D) = {1, 2, 5, 6, 7, 8, 9} := by
  sorry

end problem_part_1_problem_part_2_problem_part_3_l98_98848


namespace cost_per_remaining_ticket_is_seven_l98_98499

def total_tickets : ℕ := 29
def nine_dollar_tickets : ℕ := 11
def total_cost : ℕ := 225
def nine_dollar_ticket_cost : ℕ := 9
def remaining_tickets : ℕ := total_tickets - nine_dollar_tickets

theorem cost_per_remaining_ticket_is_seven :
  (total_cost - nine_dollar_tickets * nine_dollar_ticket_cost) / remaining_tickets = 7 :=
  sorry

end cost_per_remaining_ticket_is_seven_l98_98499


namespace train_speed_clicks_l98_98906

theorem train_speed_clicks (x : ℕ) (L : ℕ) (H : L = 40) :
  let feet_per_mile := 5280 
  let minutes_per_hour := 60
  let clicks_per_mile := (feet_per_mile / L)
  let clicks_per_hour := x * clicks_per_mile
  let clicks_per_minute := clicks_per_hour / minutes_per_hour
  let clicks_per_second := clicks_per_minute / 60
  let clicks_in_30_seconds := clicks_per_second * 30
  clicks_in_30_seconds ≈ x :=
by sorry

end train_speed_clicks_l98_98906


namespace find_a_l98_98194

theorem find_a (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_asymptote : ∀ x : ℝ, x = π/2 ∨ x = 3*π/2 ∨ x = -π/2 ∨ x = -3*π/2 → b*x = π/2 ∨ b*x = 3*π/2 ∨ b*x = -π/2 ∨ b*x = -3*π/2)
  (h_amplitude : ∀ x : ℝ, |a * (1 / Real.cos (b * x))| ≤ 3): 
  a = 3 := 
sorry

end find_a_l98_98194


namespace arithmetic_seq_term_l98_98351

noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k
noncomputable def A (n k : ℕ) : ℕ := Nat.perm n k

theorem arithmetic_seq_term :
  ∃ (a d n : ℤ), 
  a = C 10 7 - A 5 2 ∧ 
  d = -4 ∧ 
  n = (77 ^ 77 - 15) % 19 ∧ 
  a + (n - 1) * d = 104 - 4 * n :=
by
  sorry

end arithmetic_seq_term_l98_98351


namespace Sabrina_pencils_l98_98827

variable (S : ℕ) (J : ℕ)

theorem Sabrina_pencils (h1 : S + J = 50) (h2 : J = 2 * S + 8) :
  S = 14 :=
by
  sorry

end Sabrina_pencils_l98_98827


namespace tip_percentage_approx_l98_98913

theorem tip_percentage_approx (bill_total : ℝ) (num_people : ℕ) (person_share : ℝ) (T : ℝ)
  (h1 : bill_total = 211.0)
  (h2 : num_people = 5)
  (h3 : person_share = 48.53)
  (h4 : (num_people : ℝ) * person_share = bill_total * (1 + T)) :
  T ≈ 0.15 :=
by
  sorry

end tip_percentage_approx_l98_98913


namespace question_2_question_3_l98_98360

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then 1/x - 1 else 1 - 1/x

theorem question_2 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  1/a + 1/b = 2 :=
sorry

theorem question_3 (a b m : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : ∀ x, a ≤ x ∧ x ≤ b → f x ∈ Set.Icc (m * a) (m * b)) (h4 : m ≠ 0) :
  0 < m ∧ m < 1/4 :=
sorry

end question_2_question_3_l98_98360


namespace probability_point_in_spherical_region_l98_98616

theorem probability_point_in_spherical_region :
  let cube_region := {p : ℝ × ℝ × ℝ | (-2 : ℝ) ≤ p.1 ∧ p.1 ≤ 2 ∧ (-2 : ℝ) ≤ p.2 ∧ p.2 ≤ 2 ∧ (-2 : ℝ) ≤ p.3 ∧ p.3 ≤ 2}
  let sphere_region := {p : ℝ × ℝ × ℝ | p.1^2 + p.2^2 + p.3^2 ≤ 4}
  P := ((measure_theory.volume sphere_region) / (measure_theory.volume cube_region))
  in P = (Real.pi / 6) :=
sorry

end probability_point_in_spherical_region_l98_98616


namespace angle_measure_l98_98457

-- Define angles as real numbers representing their measures in degrees
variables (α₁ α₂ α₅ : ℝ)

-- Given conditions
axiom parallel_lines : m ∥ n
axiom angle_relation : α₁ = (1 / 8) * α₂

-- Prove that α₅ = 20
theorem angle_measure (h_parallel : parallel_lines m n) (h_angle : angle_relation α₁ α₂) : α₅ = 20 := by
  sorry

end angle_measure_l98_98457


namespace ellipse_equation_PQ_line_equation_l98_98348

-- Definitions of the given conditions
def center_of_ellipse := (0, 0)

def minor_axis_length := 2 * Real.sqrt 2

def focus_point (c : ℝ) (hc : c > 0) := (c, 0)

def fixed_point (c : ℝ) (hc : c > 0) := (10 / c - c, 0)

def vector_relation (c : ℝ) (hc : c > 0) : Prop := 
  (c, 0) = (2 * (10 / c - 2 * c), 0)

-- The proofs of the required equations
theorem ellipse_equation (c : ℝ) (hc : c > 0) (h_relation : vector_relation c hc) :
  let a := Real.sqrt (6)
  let b := Real.sqrt (2)
  (2 * c = 4) →
  (eqn_of_ellipse : (x y : ℝ) → x^2 / 6 + y^2 / 2 = 1) ∧
  (eccentricity := 2 / Real.sqrt(6)) :=
sorry

theorem PQ_line_equation (k : ℝ) :
  let A := (3, 0)
  k = ±(Real.sqrt(5) / 5) →
  ((x y : ℝ) → x - Real.sqrt(5) * y - 3 = 0) ∨ ((x y : ℝ) → x + Real.sqrt(5) * y - 3 = 0) :=
sorry

end ellipse_equation_PQ_line_equation_l98_98348


namespace problem_equivalent_proof_l98_98318

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98318


namespace num_valid_arrangements_equals_4_l98_98689

def num_valid_arrangements : ℕ :=
  let! (A, B, C, D) : Type := list (A :: B :: C :: D :: nil)
  let constraints (seating: List (A × B × C × D)) : Prop :=
    seating.filter (λ s, s.1 ≠ s.3 && s.2 ≠ s.4).length = 1
  constraints = 4
  sorry

theorem num_valid_arrangements_equals_4 : num_valid_arrangements = 4 :=
  by sorry

end num_valid_arrangements_equals_4_l98_98689


namespace profit_in_february_l98_98573

variable (C : ℝ) (H1 : C > 0)

-- Definitions based on conditions
def first_markup_price : ℝ := 1.2 * C
def second_markup_price : ℝ := 1.5 * C
def final_selling_price : ℝ := 1.23 * C

-- Theorem to prove profit is 23% of the cost price
theorem profit_in_february : (final_selling_price - C) = 0.23 * C := by
  sorry

end profit_in_february_l98_98573


namespace pie_cost_correct_l98_98463

-- Define the initial and final amounts of money Mary had.
def initial_amount : ℕ := 58
def final_amount : ℕ := 52

-- Define the cost of the pie as the difference between initial and final amounts.
def pie_cost : ℕ := initial_amount - final_amount

-- State the theorem that given the initial and final amounts, the cost of the pie is 6.
theorem pie_cost_correct : pie_cost = 6 := by 
  sorry

end pie_cost_correct_l98_98463


namespace greatest_integer_not_exceeding_1000x_l98_98997

-- Given the conditions of the problem
variables (x : ℝ)
-- Cond 1: Edge length of the cube
def edge_length := 2
-- Cond 2: Point light source is x centimeters above a vertex
-- Cond 3: Shadow area excluding the area beneath the cube is 98 square centimeters
def shadow_area_excluding_cube := 98
-- This is the condition total area of the shadow
def total_shadow_area := shadow_area_excluding_cube + edge_length ^ 2

-- Statement: Prove that the greatest integer not exceeding 1000x is 8100:
theorem greatest_integer_not_exceeding_1000x (h1 : total_shadow_area = 102) : x ≤ 8.1 :=
by
  sorry

end greatest_integer_not_exceeding_1000x_l98_98997


namespace ratio_of_equilateral_incircles_l98_98120

noncomputable def ratio_of_areas (r : ℝ) : ℝ :=
  let t1 := r * (real.sqrt 3)
  let area1 := (real.sqrt 3 / 4) * t1^2
  let t2 := r * (real.sqrt 3)
  let area2 := (real.sqrt 3 / 4) * t2^2
  area1 / area2

theorem ratio_of_equilateral_incircles (r : ℝ) : ratio_of_areas r = 1 := by
  sorry

end ratio_of_equilateral_incircles_l98_98120


namespace infinite_powers_of_2_l98_98438

def sequence_contains_infinitely_many_powers_of_2 (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ), ∀ n : ℕ, (∃ (i : ℕ), a i = 2^k)

theorem infinite_powers_of_2 (a : ℕ → ℕ) (h₀ : ∀ (n : ℕ), a (n+1) = a n + (a n % 10))
  (h₁ : ¬ (5 ∣ a 1)) : sequence_contains_infinitely_many_powers_of_2 a (λ n, a n % 10) :=
sorry

end infinite_powers_of_2_l98_98438


namespace find_angle_MAK_l98_98475

-- Definitions of points and square properties
variables (a : ℝ) -- side length of the square
variables (A B C D M K : ℝ × ℝ) -- points in the plane

def is_square (A B C D : ℝ × ℝ) (a : ℝ) : Prop :=
  A = (0, 0) ∧ B = (a, 0) ∧ C = (a, a) ∧ D = (0, a)

-- Points M and K on sides BC and CD respectively
def on_sides (M K : ℝ × ℝ) (B C D : ℝ × ℝ) : Prop :=
  ∃ (x : ℝ), M = (x, a) ∧ x >= 0 ∧ x <= a ∧
  ∃ (y : ℝ), K = (a, y) ∧ y >= 0 ∧ y <= a

-- Perimeter condition
def perimeter_condition (M K C : ℝ × ℝ) (a : ℝ) : Prop :=
  dist (C, M) + dist (M, K) + dist (K, C) = 2 * a

-- The angle we want to prove is 45 degrees
def is_angle_45 (A M K : ℝ × ℝ) : Prop :=
  angle A M K = 45

-- Lean theorem statement
theorem find_angle_MAK (a : ℝ) (A B C D M K : ℝ × ℝ) 
  (h1 : is_square A B C D a) 
  (h2 : on_sides M K B C D)
  (h3 : perimeter_condition M K C a) : 
  is_angle_45 A M K :=
sorry

end find_angle_MAK_l98_98475


namespace clerical_percentage_after_reduction_l98_98468

theorem clerical_percentage_after_reduction
  (total_employees : ℕ)
  (clerical_fraction : ℚ)
  (reduction_fraction : ℚ)
  (h1 : total_employees = 3600)
  (h2 : clerical_fraction = 1/4)
  (h3 : reduction_fraction = 1/4) : 
  let initial_clerical := clerical_fraction * total_employees
  let reduced_clerical := (1 - reduction_fraction) * initial_clerical
  let let_go := initial_clerical - reduced_clerical
  let new_total := total_employees - let_go
  let clerical_percentage := (reduced_clerical / new_total) * 100
  clerical_percentage = 20 :=
by sorry

end clerical_percentage_after_reduction_l98_98468


namespace train_speed_in_kilometers_per_hour_l98_98182

-- Define the conditions
def train_length : ℝ := 160 -- length of the train in meters
def time_to_pass_pole : ℝ := 8 -- time to pass the pole in seconds
def conversion_factor : ℝ := 3.6 -- factor to convert from m/s to km/hr

-- Define the speed of the train in m/s
def speed_in_meters_per_second : ℝ := train_length / time_to_pass_pole

-- Define the speed of the train in km/hr
def speed_in_kilometers_per_hour : ℝ := speed_in_meters_per_second * conversion_factor

-- The proof statement 
theorem train_speed_in_kilometers_per_hour :
  speed_in_kilometers_per_hour = 72 := by 
  sorry

end train_speed_in_kilometers_per_hour_l98_98182


namespace right_triangle_area_l98_98008

variable {X Y Z : Type*}
variable [metric_space X] [metric_space Y] [metric_space Z]
variable (xy xz yz : ℝ)

-- Conditions
def right_triangle (XY XZ YZ : X) : Prop := 
  (distance XY XZ) = 15 ∧
  (distance XZ YZ) = 10 ∧
  (angle X Y Z = 90)

-- Theorem
theorem right_triangle_area (h: right_triangle XY XZ YZ) : 
  calc_area XY XZ = 75 := sorry

end right_triangle_area_l98_98008


namespace sequence_limit_l98_98740

noncomputable def a (n : ℕ) : ℚ :=
  (3^(-n : ℤ) + 2^(-n : ℤ) + (-1 : ℚ)^[n] * (3^(-n : ℤ) - 2^(-n : ℤ))) / 2

theorem sequence_limit :
  (∃ L : ℚ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(∑ k in (finset.range n), a (k + 1)) - L| < ε) ∧
  (L : ℚ) = 19 / 24 :=
sorry

end sequence_limit_l98_98740


namespace scientific_notation_correct_l98_98126

-- Definitions for conditions
def number_to_convert : ℕ := 380000
def scientific_notation (a n : ℝ) := a * (10 ^ n)

-- The proof problem to be stated
theorem scientific_notation_correct :
  ∃ a n : ℝ, 1 ≤ abs a ∧ abs a < 10 ∧ ∃ (n_int : ℤ), n = n_int ∧ scientific_notation a n = number_to_convert :=
sorry

end scientific_notation_correct_l98_98126


namespace count_numbers_with_square_factors_l98_98766

theorem count_numbers_with_square_factors :
  let squares := [4, 9, 16, 25, 36, 49, 64]
  let multiples (n : ℕ) := ∀ k ∈ squares, n % k = 0
  let count_multiples (n : ℕ) := (1..100).count multiples
  count_multiples squares = 48 :=
  sorry

end count_numbers_with_square_factors_l98_98766


namespace sum_numbers_on_papers_l98_98917

-- Define the pieces of paper
def piece (n : ℕ) := {x : Finset ℕ // x.card = 3 ∧ ∀ i ∈ x, i ≤ n}

-- Define the condition of sharing exactly one common integer
def share_one_common (p1 p2 : Finset ℕ) : Prop := (p1 ∩ p2).card = 1

-- Define the main theorem
theorem sum_numbers_on_papers (n : ℕ) 
  (pieces : Fin n → piece n) 
  (H : ∀ i j, i ≠ j → share_one_common (pieces i).val (pieces j).val) : 
  ∑ i in Finset.range n, ∑ k in (pieces i).val, k = 3 * (n * (n + 1)) / 2 := 
sorry

end sum_numbers_on_papers_l98_98917


namespace fraction_lies_between_three_and_four_l98_98561

theorem fraction_lies_between_three_and_four :
  ∃ x : ℚ, (x = 13 / 4) ∧ (3 < x ∧ x < 4) :=
by {
  let x := 13 / 4,
  use x,
  split;
  sorry
}

end fraction_lies_between_three_and_four_l98_98561


namespace speed_faster_train_correct_l98_98117

noncomputable def speed_faster_train_proof
  (time_seconds : ℝ) 
  (speed_slower_train : ℝ)
  (train_length_meters : ℝ) :
  Prop :=
  let time_hours := time_seconds / 3600
  let train_length_km := train_length_meters / 1000
  let total_distance_km := train_length_km + train_length_km
  let relative_speed_km_hr := total_distance_km / time_hours
  let speed_faster_train := relative_speed_km_hr + speed_slower_train
  speed_faster_train = 46

theorem speed_faster_train_correct :
  speed_faster_train_proof 36.00001 36 50.000013888888894 :=
by 
  -- proof steps would go here
  sorry

end speed_faster_train_correct_l98_98117


namespace mushroom_picking_l98_98953

theorem mushroom_picking (n T : ℕ) (hn_min : n ≥ 5) (hn_max : n ≤ 7)
  (hmax : ∀ (M_max M_min : ℕ), M_max = T / 5 → M_min = T / 7 → 
    T ≠ 0 → M_max ≤ T / n ∧ M_min ≥ T / n) : n = 6 :=
by
  sorry

end mushroom_picking_l98_98953


namespace prime_sequence_divisibility_l98_98038

-- Definitions based on the given conditions
def is_prime (n : ℕ) : Prop := nat.prime n

def a_seq (p : ℕ) (n : ℕ) : ℕ :=
nat.rec_on n 2 (λ (n : ℕ) (a_n_minus_1 : ℕ), 
  a_n_minus_1 + (nat.ceil ((p * a_n_minus_1) / (n + 1))))

-- Theorem statement based on the problem and conditions
theorem prime_sequence_divisibility (p : ℕ) (n : ℕ) (h1 : is_prime p) (h2 : is_prime (p + 2)) (h3 : p > 3) (h4 : n ≥ 3) (h5 : n < p) :
  n ∣ (p * a_seq p (n - 1) + 1) :=
sorry

end prime_sequence_divisibility_l98_98038


namespace borrowing_schemes_l98_98070

theorem borrowing_schemes :
  ∃ borrowing_schemes : ℕ,
  borrowing_schemes = 60 ∧
  ∀ (students : Fin 5 → String) (novels : Fin 4 → String),
  students 0 = "A" → novels.contains "Romance of the Three Kingdoms" → 
  (∃ (f : Fin 5 → Fin 4), 
    (∀ i : Fin 5, novels (f i) ∈ ["Dream of the Red Chamber", "Romance of the Three Kingdoms", "Water Margin", "Journey to the West"]) ∧ 
    f 0 = 1 ∧ 
    Function.Injective f
  ) :=
begin
  existsi 60,
  split,
  { refl, },
  { intros students novels A_borrows_ROM determineROM,
    use sorry
  }
end

end borrowing_schemes_l98_98070


namespace sin_cos_halves_l98_98691

variable {α : Real} 

theorem sin_cos_halves (h : sin α = 1 / 3) : sin (α / 2) + cos (α / 2) = 2 * sqrt 3 / 3 ∨ sin (α / 2) + cos (α / 2) = -2 * sqrt 3 / 3 :=
by
  sorry

end sin_cos_halves_l98_98691


namespace tangent_line_derivative_l98_98388

-- Define the condition for the tangent line
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the derivative at the point of tangency
def derivative_at_point (f : ℝ → ℝ) (x₀ : ℝ) : ℝ := -f x₀ / 2

theorem tangent_line_derivative (f : ℝ → ℝ) (x₀ : ℝ) (hx : tangent_line x₀ (f x₀)) :
  derivative_at_point f x₀ = -1 / 2 :=
by sorry

end tangent_line_derivative_l98_98388


namespace count_perfect_square_factors_l98_98765

open Nat

def has_larger_square_factor (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

theorem count_perfect_square_factors :
  (Finset.filter has_larger_square_factor (Finset.range 101)).card = 42 := by
sorry

end count_perfect_square_factors_l98_98765


namespace pancakes_needed_l98_98193

def short_stack_pancakes : ℕ := 3
def big_stack_pancakes : ℕ := 5
def short_stack_customers : ℕ := 9
def big_stack_customers : ℕ := 6

theorem pancakes_needed : (short_stack_customers * short_stack_pancakes + big_stack_customers * big_stack_pancakes) = 57 :=
by
  sorry

end pancakes_needed_l98_98193


namespace transistors_2004_l98_98187

-- Definition of Moore's law specifying the initial amount and the doubling period
def moores_law (initial : ℕ) (years : ℕ) (doubling_period : ℕ) : ℕ :=
  initial * 2 ^ (years / doubling_period)

-- Condition: The number of transistors in 1992
def initial_1992 : ℕ := 2000000

-- Condition: The number of years between 1992 and 2004
def years_between : ℕ := 2004 - 1992

-- Condition: Doubling period every 2 years
def doubling_period : ℕ := 2

-- Goal: Prove the number of transistors in 2004 using the conditions above
theorem transistors_2004 : moores_law initial_1992 years_between doubling_period = 128000000 :=
by
  sorry

end transistors_2004_l98_98187


namespace total_alligators_seen_l98_98484

theorem total_alligators_seen (a_samara : ℕ) (n_friends : ℕ) (avg_friends : ℕ) (h_samara : a_samara = 20) (h_friends : avg_friends = 10) (h_n_friends : n_friends = 3) : a_samara + n_friends * avg_friends = 50 := by
    rw [h_samara, h_friends, h_n_friends]
    norm_num
    exact eq.refl 50

end total_alligators_seen_l98_98484


namespace range_of_a_l98_98733

noncomputable def f (a x : ℝ) : ℝ := if x >= a then x else x^3 - 3 * x

noncomputable def g (a x : ℝ) : ℝ := 2 * f a x - a * x

theorem range_of_a {a : ℝ} (h : ∃! x : ℝ, g a x = 0) : a ∈ (-3/2 : ℝ, 2) :=
sorry

end range_of_a_l98_98733


namespace probability_inside_sphere_l98_98612

noncomputable def volume_cube : ℝ := 64

noncomputable def volume_sphere : ℝ := (4 * π * 8) / 3

theorem probability_inside_sphere : 
  volume_sphere / volume_cube = π / 6 :=
by 
  sorry

end probability_inside_sphere_l98_98612


namespace completing_square_l98_98130

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l98_98130


namespace solve_for_z_l98_98062

theorem solve_for_z (z : ℂ) (h : 3 + 2 * complex.I * z = 7 - 4 * complex.I * z) : 
  z = -2 * complex.I / 3 :=
by
  sorry

end solve_for_z_l98_98062


namespace problems_per_page_l98_98868

variable (total_pages : ℕ) (total_problems : ℕ) (p : ℕ)

-- Conditions
def condition1 : total_pages = 4 + 6 := by
  rfl
def condition2 : total_problems = 40 := by
  rfl
def condition3 : total_pages * p = total_problems := by
  sorry  -- This will be integrated later in the proof

-- Goal Statement
theorem problems_per_page : p = 4 := by
  rw [condition1, condition2]
  sorry  -- This will include the actual proof steps

end problems_per_page_l98_98868


namespace hyperbola_equation_and_distance_range_l98_98508

noncomputable def hyperbola : Type :=
  {a b : ℝ // a > 0 ∧ b > 0 ∧ 4 * a * a + 3 * b * b = 12}

noncomputable def distance_range (a b : ℝ) (k1 k2 : ℝ) (h : k1 * k2 = -2) : set ℝ :=
  {d : ℝ | 3 * real.sqrt 3 < d ∧ d ≤ 6}

theorem hyperbola_equation_and_distance_range :
  ∃ (a b : ℝ) (hab : a > 0 ∧ b > 0), 
  (∀ x y : ℝ, x^2 - y^2 / 3 = 1) ∧ 
  (∀ k1 k2 : ℝ, k1 * k2 = -2 → 
    (∀ d : ℝ, d ∈ distance_range a b k1 k2 {d : ℝ | 3 * real.sqrt 3 < d ∧ d ≤ 6})) :=
sorry

end hyperbola_equation_and_distance_range_l98_98508


namespace count_perfect_square_factors_l98_98763

open Nat

def has_larger_square_factor (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

theorem count_perfect_square_factors :
  (Finset.filter has_larger_square_factor (Finset.range 101)).card = 42 := by
sorry

end count_perfect_square_factors_l98_98763


namespace total_cost_of_ads_l98_98660

-- Define the conditions
def cost_ad1 := 3500
def minutes_ad1 := 2
def cost_ad2 := 4500
def minutes_ad2 := 3
def cost_ad3 := 3000
def minutes_ad3 := 3
def cost_ad4 := 4000
def minutes_ad4 := 2
def cost_ad5 := 5500
def minutes_ad5 := 5

-- Define the function to calculate the total cost
def total_cost :=
  (cost_ad1 * minutes_ad1) +
  (cost_ad2 * minutes_ad2) +
  (cost_ad3 * minutes_ad3) +
  (cost_ad4 * minutes_ad4) +
  (cost_ad5 * minutes_ad5)

-- The statement to prove
theorem total_cost_of_ads : total_cost = 66000 := by
  sorry

end total_cost_of_ads_l98_98660


namespace solution_set_f_pos_l98_98342

noncomputable theory
open_locale classical

variables {R : Type*} [ordered_ring R] [topological_space R] [order_topology R]
variables {f : R → R}

-- Define the conditions as hypotheses
def is_odd (f : R → R) : Prop := ∀ x, f (-x) = -f (x)
def f_derivative_condition (f f' : R → R) : Prop := ∀ x, x > 0 → x*f'(x) < f(x)

-- State the theorem
theorem solution_set_f_pos 
  (h1 : is_odd f)
  (h2 : ∀ x, differentiable_at R f x)
  (h3 : f (1) = 0)
  (h4 : ∃ f' : R → R, ∀ x, deriv f x = f'(x) ∧ f_derivative_condition f f') :
  { x | f x > 0 } = (set.Ioo 0 1) ∪ (set.Iio (-1)) :=
sorry

end solution_set_f_pos_l98_98342


namespace find_vector_d_l98_98509

theorem find_vector_d :
  (∀ t : ℝ, ∃ (d v : ℝ × ℝ), 
    (∀ x : ℝ, x ≥ 4 → 
    (v.fst + t * d.fst = x ∧ v.snd + t * d.snd = (5 * x - 7) / 3) ∧
    dist (v.fst + t * d.fst, v.snd + t * d.snd) (4, 2) = t)) →
  ∃ d : ℝ × ℝ, d = (3 / real.sqrt 34, 5 / real.sqrt 34) :=
begin
  intro h,
  sorry
end

end find_vector_d_l98_98509


namespace dog_roaming_area_l98_98967

theorem dog_roaming_area :
  let shed_radius := 20
  let rope_length := 10
  let distance_from_edge := 10
  let radius_from_center := shed_radius - distance_from_edge
  radius_from_center = rope_length →
  (π * rope_length^2 = 100 * π) :=
by
  intros shed_radius rope_length distance_from_edge radius_from_center h
  sorry

end dog_roaming_area_l98_98967


namespace x_intercept_of_perpendicular_line_l98_98549

noncomputable def original_line (x y : ℝ) : Prop := 4 * x + 5 * y = 10

noncomputable def slope (m : ℝ) (x y : ℝ) : Prop := y = m * x

noncomputable def perpendicular_slope (m1 m2 : ℝ) : Prop := m1 * m2 = -1

noncomputable def new_line (m x y : ℝ) : Prop := y = m * x - 3

-- The math proof statement
theorem x_intercept_of_perpendicular_line : 
  (∃ x : ℝ, (∃ y : ℝ, original_line x y) ∧ (∀ m1 m2 : ℝ, original_line x y → slope m1 x y → perpendicular_slope m1 m2) ∧ (∃ y, new_line (5/4) x y) ∧ (new_line (5/4) x 0)) → 
  (∃ x : ℝ, new_line (5/4) x 0 ∧ x = 12 / 5):=
by
  sorry

end x_intercept_of_perpendicular_line_l98_98549


namespace games_lost_l98_98486

theorem games_lost (total_games won_games : ℕ) (h_total : total_games = 12) (h_won : won_games = 8) :
  (total_games - won_games) = 4 :=
by
  -- Placeholder for the proof
  sorry

end games_lost_l98_98486


namespace sum_of_digits_625_base5_l98_98123

def sum_of_digits_base_5 (n : ℕ) : ℕ :=
  let rec sum_digits n :=
    if n = 0 then 0
    else (n % 5) + sum_digits (n / 5)
  sum_digits n

theorem sum_of_digits_625_base5 : sum_of_digits_base_5 625 = 5 := by
  sorry

end sum_of_digits_625_base5_l98_98123


namespace length_of_AB_l98_98696

theorem length_of_AB 
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 + A.2 ^ 2 = 8)
  (hB : B.1 ^ 2 + B.2 ^ 2 = 8)
  (lA : A.1 - 2 * A.2 + 5 = 0)
  (lB : B.1 - 2 * B.2 + 5 = 0) :
  dist A B = 2 * Real.sqrt 3 := by
  sorry

end length_of_AB_l98_98696


namespace log_inequality_implies_exp_inequality_l98_98262

variable {a x y : ℝ}

theorem log_inequality_implies_exp_inequality 
  (hlog : log a x > log a y) 
  (ha : 0 < a) 
  (ha1 : a < 1) : 
  3^(x - y) < 1 := 
sorry

end log_inequality_implies_exp_inequality_l98_98262


namespace num_trips_from_A_to_C_l98_98048

-- Definitions for vertices A and C
def A : Type := sorry -- Define type for vertex A
def C : Type := sorry -- Define type for vertex C

-- Definition for the cube and the condition that C is opposite of A and not sharing any face
def is_opposite (a b : Type) : Prop := sorry -- Define the oppositeness property
axiom ac_opposite : is_opposite A C -- A and C are opposite on the cube

-- Number of different 4-edge trips
axiom trips : ℕ := 12

-- Prove the number of 4-edge trips from A to C is 12
theorem num_trips_from_A_to_C : trips = 12 :=
by sorry

end num_trips_from_A_to_C_l98_98048


namespace quadratic_roots_range_quadratic_root_condition_l98_98903

-- Problem 1: Prove that the range of real number \(k\) for which the quadratic 
-- equation \(x^{2} + (2k + 1)x + k^{2} + 1 = 0\) has two distinct real roots is \(k > \frac{3}{4}\). 
theorem quadratic_roots_range (k : ℝ) : 
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x^2 + (2*k+1)*x + k^2 + 1 = 0) ↔ (k > 3/4) := 
sorry

-- Problem 2: Given \(k > \frac{3}{4}\), prove that if the roots \(x₁\) and \(x₂\) of 
-- the equation satisfy \( |x₁| + |x₂| = x₁ \cdot x₂ \), then \( k = 2 \).
theorem quadratic_root_condition (k : ℝ) 
    (hk : k > 3 / 4)
    (x₁ x₂ : ℝ)
    (h₁ : x₁^2 + (2*k+1)*x₁ + k^2 + 1 = 0)
    (h₂ : x₂^2 + (2*k+1)*x₂ + k^2 + 1 = 0)
    (h3 : |x₁| + |x₂| = x₁ * x₂) : 
    k = 2 := 
sorry

end quadratic_roots_range_quadratic_root_condition_l98_98903


namespace games_that_didnt_work_l98_98047

-- Definitions based on conditions
def games_from_friend : Nat := 50
def games_from_garage_sale : Nat := 27
def total_games : Nat := games_from_friend + games_from_garage_sale
def good_games : Nat := 3
def bad_games : Nat := total_games - good_games

-- Theorem statement
theorem games_that_didnt_work : bad_games = 74 :=
by simp [total_games, bad_games, games_from_friend, games_from_garage_sale, good_games]; norm_num

end games_that_didnt_work_l98_98047


namespace equiv_conditions_l98_98828

variables {G : Type*} [group G] (H : subgroup G) (x y : G)

/-- Equivalence of the three given conditions for finite group G and subgroup H -/
theorem equiv_conditions :
  (x * H = y * H) ↔ ((x * H) ∩ (y * H) ≠ ∅) ∧ (x⁻¹ * y ∈ H) :=
sorry

end equiv_conditions_l98_98828


namespace time_for_tom_to_complete_wall_l98_98022

-- Define the conditions
def avery_work_rate := 1 / 3 -- Avery's work rate in walls per hour
def tom_work_rate := 1 / 3 -- Tom's work rate in walls per hour

-- Define the problem statement
theorem time_for_tom_to_complete_wall :
  let combined_work_rate := avery_work_rate + tom_work_rate in
  let work_done_in_first_hour := combined_work_rate in
  let remaining_work := 1 - work_done_in_first_hour in
  let time_for_tom := remaining_work / tom_work_rate in
  time_for_tom = 1 :=
sorry

end time_for_tom_to_complete_wall_l98_98022


namespace anna_always_wins_l98_98470

def is_proper_divisor (d n : ℕ) : Prop := d > 0 ∧ d < n ∧ n % d = 0

def game_move (n : ℕ) (d : ℕ) : ℕ := n + d

theorem anna_always_wins (initial_bowl : ℕ) (max_bowl : ℕ) :
  initial_bowl = 2 ∧ max_bowl = 2024 →
  (∀ (current_bowl : ℕ),  current_bowl <= max_bowl → 
   ∃ (d : ℕ), is_proper_divisor d current_bowl ∧
   (∀ (next_bowl : ℕ), next_bowl = game_move current_bowl d → 
    ∃ (d2 : ℕ), is_proper_divisor d2 next_bowl ∧
    (next_bowl + d2 > max_bowl → false))) → 
  (∃ (final_bowl : ℕ), final_bowl > max_bowl) :=
begin
  intros h1 h2,
  sorry
end

end anna_always_wins_l98_98470


namespace problem_equivalent_proof_l98_98314

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98314


namespace general_term_and_sum_formula_exists_minimum_M_l98_98700

section ArithmeticSequence

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Given conditions
def arithmetic_seq (d : ℤ) : Prop :=
  ∀ n : ℕ, a(n + 1) - a n = d

def sum_of_first_n_terms : Prop :=
  ∀ n : ℕ, S n = n • (a 1 + a n) / 2

def geometric_sequence (a_2 a_3 a_5 : ℤ) : Prop :=
  a_3^2 = a_2 * a_5

def S6_is_45 : Prop :=
  S 6 = 45

-- (1) Proving the general term and the sum formula
theorem general_term_and_sum_formula (h_seq : arithmetic_seq a) (h_sum : sum_of_first_n_terms S) 
  (h_geom : geometric_sequence (a 2) (a 3) (a 5)) (h_S6 : S6_is_45 S) :
  (∃ d, a = λ n, 3 * n - 3 ∧ S = λ n, 3 * n * (n - 1) / 2) := sorry

-- Define p_n
def p (n : ℕ) : ℤ :=
  S (n + 2) / S (n + 1) + (n - 1) / (n + 1)

-- (2) Existence and minimum value of M
theorem exists_minimum_M (h_seq : arithmetic_seq a) (h_sum : sum_of_first_n_terms S) 
  (h_geom : geometric_sequence (a 2) (a 3) (a 5)) (h_S6 : S6_is_45 S) :
  ∃ M : ℤ, (∀ n : ℕ, ∑ i in finset.range n, p i - 2 * n ≤ M) ∧ M = 2 := sorry

end ArithmeticSequence

end general_term_and_sum_formula_exists_minimum_M_l98_98700


namespace only_lines_and_circles_have_constant_curvature_l98_98619

-- Define the property of a curve that can slide along its own type while maintaining constant curvature
def can_slide_along_path (C : Type) := 
  ∃ (curve_path : Type), (∀ (p1 p2 : curve_path), constant_curvature p1 = constant_curvature p2)

-- Prove that only lines and circles have this property.
theorem only_lines_and_circles_have_constant_curvature : 
  ¬ (∃ (C : Type), C ≠ ℝ ∧ C ≠ set.univ → can_slide_along_path C) := 
sorry

end only_lines_and_circles_have_constant_curvature_l98_98619


namespace hexagonal_prism_volume_l98_98899

theorem hexagonal_prism_volume (d : ℝ) (h : d > 0) (angle : real.angle) (h_angle : angle = real.angle.of_degrees 30) :
  let H := d * real.cos (real.angle.to_radians angle) in
  let S := 3 * (d/4)^2 * real.sqrt 3 / 2 in
  let V := S * H in
  V = 9 * d^3 / 64 := by
  sorry

end hexagonal_prism_volume_l98_98899


namespace find_triangle_angles_l98_98891

noncomputable def triangle_medians_angles (A B C : Point) (M K E F : Point) : Prop :=
∀ (AM BK : Line),
  is_median AM A B C M →
  is_median BK B A C K →
  circle_centered_at A B C contains E →
  circle_centered_at A B C contains F →
  ratio AE AM = 2 →
  ratio BF BK = 3 / 2 →
  angles_of_triangle A B C = (90, arctan 2, 90 - arctan 2)

-- Additional necessary definitions such as Point, Line, is_median, circle_centered_at, ratio, and angles_of_triangle would need to be defined in the Lean library context.

theorem find_triangle_angles (A B C M K E F : Point) :
  triangle_medians_angles A B C M K E F :=
sorry

end find_triangle_angles_l98_98891


namespace factor_72x3_minus_252x7_l98_98218

theorem factor_72x3_minus_252x7 (x : ℝ) : (72 * x^3 - 252 * x^7) = (36 * x^3 * (2 - 7 * x^4)) :=
by
  sorry

end factor_72x3_minus_252x7_l98_98218


namespace no_anti_pascal_triangle_with_2018_rows_l98_98149

noncomputable def sum_first_n : ℕ → ℕ
| 0 := 0
| n := n + sum_first_n (n - 1)

theorem no_anti_pascal_triangle_with_2018_rows :
  ∀ (anti_pascal_triangle : Π (n : ℕ), fin n → ℕ),
    (∀ (n : ℕ) (i : fin (n-1)), abs (anti_pascal_triangle (n-1) i.1 - anti_pascal_triangle (n-1) (i.1 + 1)) = anti_pascal_triangle n i) →
    (∀ (n : ℕ), ∑ i, anti_pascal_triangle n i = nat.choose (n+1) 2 ) →
    (¬ ∃ (anti_pascal_triangle : Π (n : fin 2018), ℕ),
      is_permutation (finset.range 2037171) (finset.range' 1 2037171)
        (finset.univ.image (λ ⟨n, i⟩, anti_pascal_triangle n i))) :=
begin
  intros anti_pascal_triangle h_diff h_sum h_perm,
  sorry
end

end no_anti_pascal_triangle_with_2018_rows_l98_98149


namespace system_inequalities_solution_l98_98746

theorem system_inequalities_solution (a x : ℝ) (h₁ : x - 1 ≥ a^2) (h₂ : x - 4 < 2a) : 
  -1 < a ∧ a < 3 :=
sorry

end system_inequalities_solution_l98_98746


namespace smallest_n_fig2_valid_fig4_impossible_49_fig4_impossible_33_smallest_n_fig4_valid_l98_98151

noncomputable def smallest_n_fig2 : ℕ :=
  4

theorem smallest_n_fig2_valid : ∃ n : ℕ, n = smallest_n_fig2 ∧
  ∀ a b : ℕ, a ≠ b ∧ a, b ≤ n =
  (connected a b → gcd (a - b) n = 1) ∧
  (¬connected a b → gcd (a - b) n > 1) :=
sorry

theorem fig4_impossible_49 : ¬∃ (f : fin 5 → ℕ),
  (∀ (i j : fin 5), i ≠ j → gcd (f i - f j) 49 = 1 = connected i j ∧
    gcd (f i - f j) 49 > 1 = ¬connected i j) :=
sorry

theorem fig4_impossible_33 : ¬∃ (f : fin 5 → ℕ),
  (∀ (i j : fin 5), i ≠ j → gcd (f i - f j) 33 = 1 = connected i j ∧
    gcd (f i - f j) 33 > 1 = ¬connected i j) :=
sorry

noncomputable def smallest_n_fig4 : ℕ :=
  105

theorem smallest_n_fig4_valid : ∃ n : ℕ, n = smallest_n_fig4 ∧
  ∀ a b : ℕ, a ≠ b ∧ a, b ≤ n =
  (connected a b → gcd (a - b) n = 1) ∧
  (¬connected a b → gcd (a - b) n > 1) :=
sorry

end smallest_n_fig2_valid_fig4_impossible_49_fig4_impossible_33_smallest_n_fig4_valid_l98_98151


namespace room_area_ratio_l98_98430

-- Defining the conditions and problem in Lean 4
theorem room_area_ratio (init_length init_width : ℕ) (incr_length incr_width : ℕ)
  (num_rooms_same : ℕ) (total_area : ℕ) :
  init_length = 13 →
  init_width = 18 →
  incr_length = 2 →
  incr_width = 2 →
  num_rooms_same = 3 →
  total_area = 1800 →
  let new_length := init_length + incr_length,
      new_width := init_width + incr_width,
      room_area := new_length * new_width,
      all_equal_rooms_area := (1 + num_rooms_same) * room_area,
      additional_room_area := total_area - all_equal_rooms_area
  in additional_room_area / room_area = 2 :=
by {
  intros,
  let new_length := init_length + incr_length,
  let new_width := init_width + incr_width,
  let room_area := new_length * new_width,
  let all_equal_rooms_area := (1 + num_rooms_same) * room_area,
  let additional_room_area := total_area - all_equal_rooms_area,
  have ratio := additional_room_area / room_area,
  show additional_room_area / room_area = 2, from sorry
}

end room_area_ratio_l98_98430


namespace minimize_fraction_sum_l98_98041

theorem minimize_fraction_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 :=
sorry

end minimize_fraction_sum_l98_98041


namespace ratio_of_u_to_v_l98_98539

theorem ratio_of_u_to_v {b u v : ℝ} 
  (h1 : b ≠ 0)
  (h2 : 0 = 12 * u + b)
  (h3 : 0 = 8 * v + b) : 
  u / v = 2 / 3 := 
by
  sorry

end ratio_of_u_to_v_l98_98539


namespace relationship_between_a_and_b_l98_98718

variable (a b : ℝ)

-- Conditions: Points lie on the line y = 2x + 1
def point_M (a : ℝ) : Prop := a = 2 * 2 + 1
def point_N (b : ℝ) : Prop := b = 2 * 3 + 1

-- Prove that a < b given the conditions
theorem relationship_between_a_and_b (hM : point_M a) (hN : point_N b) : a < b := 
sorry

end relationship_between_a_and_b_l98_98718


namespace boys_not_studying_science_l98_98797

theorem boys_not_studying_science (total_boys : ℕ) (percentage_school_A : ℝ) (percentage_science : ℝ) (school_A_boys : ℕ) (science_boys : ℕ) : 
  total_boys = 200 → 
  percentage_school_A = 0.20 → 
  percentage_science = 0.30 → 
  school_A_boys = (percentage_school_A * total_boys).toNat → 
  science_boys = (percentage_science * school_A_boys).toNat → 
  school_A_boys - science_boys = 28 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end boys_not_studying_science_l98_98797


namespace completing_square_l98_98129

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l98_98129


namespace abs_condition_implies_l98_98951

theorem abs_condition_implies (x : ℝ) 
  (h : |x - 1| < 2) : x < 3 := by
  sorry

end abs_condition_implies_l98_98951


namespace sum_of_inscribed_angles_in_pentagon_l98_98166

theorem sum_of_inscribed_angles_in_pentagon (pentagon : Type) (circle : Type) 
  (arc_measure : ∀ (arc : Type), ℝ) 
  (inscribed_angle : ∀ (arc : Type), ℝ) 
  (inscribed_in_circle : pentagon → circle)
  (subtended_arc_by_side : pentagon → Type)
  (measure_of_total_arcs : ∀ (circle : Type), ℝ)
  (inscribed_angle_theorem : ∀ (arc : Type), inscribed_angle arc = arc_measure arc / 2) 
  (measure_of_arc_eq : ∀ (p : pentagon), arc_measure (subtended_arc_by_side p) = 72) :
  measure_of_total_arcs circle = 360 →
  (∃ (angles : fin 5 → ℝ), 
  (∑ i, angles i) = 180)
:=
begin
  sorry
end

end sum_of_inscribed_angles_in_pentagon_l98_98166


namespace find_a7_l98_98294

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98294


namespace ratio_of_areas_l98_98164

theorem ratio_of_areas (T A B : ℝ) (hT : T = 900) (hB : B = 405) (hSum : A + B = T) :
  (A - B) / ((A + B) / 2) = 1 / 5 :=
by
  sorry

end ratio_of_areas_l98_98164


namespace sin_angle_ratio_l98_98417

theorem sin_angle_ratio (A B C D : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  (triangle_ABC: is_triangle A B C)
  (angle_B_eq_45 : ∠ B = 45)
  (angle_C_eq_60 : ∠ C = 60)
  (D_divides_BC : divides D B C 2 3) :
  (sin (∠ BAD) / sin (∠ CAD)) = (4 * sqrt 6 / 9) :=
by 
  sorry

end sin_angle_ratio_l98_98417


namespace P_at_7_l98_98845

noncomputable def P (x : ℝ) : ℝ :=
  let Q := 3 * x ^ 3 - 39 * x ^ 2 + (a : ℝ) * x + (b : ℝ)
  let R := 4 * x ^ 3 - 40 * x ^ 2 + (c : ℝ) * x + (d : ℝ)
  Q * R

theorem P_at_7 (a b c d : ℝ)
    (h : ∀ x : ℝ, x ∈ {2, 3, 4, 5, 6} → (P x = 0)) :
  P 7 = 5760 := by
  sorry

end P_at_7_l98_98845


namespace geometric_seq_a7_l98_98304

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98304


namespace max_value_frac_ineq_l98_98778

theorem max_value_frac_ineq (a b : ℝ) (h1 : a^2 = 1 - 4 * b^2) (h2 : abs (a * b) ≤ 1 / 4) :
  ∃ c, c = sqrt 2 / 4 ∧ ∀ x, x = 2 * a * b / (abs a + 2 * abs b) → x ≤ c :=
sorry

end max_value_frac_ineq_l98_98778


namespace reconstruct_triangle_l98_98892

variables {V : Type*} [inner_product_space ℝ V]

def is_perpendicular (u v : V) : Prop := inner_product u v = 0

def is_altitude (A B C : V) (A' B' C' : V) : Prop :=
  is_perpendicular (A' - A) (B' - C') ∧
  is_perpendicular (B' - B) (A' - C') ∧
  is_perpendicular (C' - C) (A' - B')

theorem reconstruct_triangle
(A B C : V) (A' B' C' : V)
(h1 : is_altitude A B C A' B' C')
(h2 : ∃ (ABC_ext_bisectors : Triangle), triangle.is_defined_by_external_angle_bisectors ABC_ext_bisectors A' B' C') :
  ∃ (ABC : Triangle), triangle.vertices ABC = (A, B, C) :=
by
  sorry

end reconstruct_triangle_l98_98892


namespace large_square_pattern_l98_98855

theorem large_square_pattern :
  999999^2 = 1000000 * 999998 + 1 :=
by sorry

end large_square_pattern_l98_98855


namespace point_inside_circle_l98_98730

theorem point_inside_circle :
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_squared < radius^2 :=
by
  -- Definitions
  let center := (2, 3)
  let radius := 2
  let P := (1, 2)
  let distance_squared := (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2

  -- Goal
  show distance_squared < radius ^ 2
  
  -- Skip Proof
  sorry

end point_inside_circle_l98_98730


namespace minimum_perimeter_l98_98926

-- Define the area condition
def area_condition (l w : ℝ) : Prop := l * w = 64

-- Define the perimeter function
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

-- The theorem statement based on the conditions and the correct answer
theorem minimum_perimeter (l w : ℝ) (h : area_condition l w) : 
  perimeter l w ≥ 32 := by
sorry

end minimum_perimeter_l98_98926


namespace crank_slider_motion_l98_98581

def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 60
def t : ℝ := sorry -- t is a variable, no specific value required

theorem crank_slider_motion :
  (∀ t : ℝ, ((90 * Real.cos (10 * t)), (90 * Real.sin (10 * t) + 60)) = (x, y)) ∧
  (∀ t : ℝ, ((-900 * Real.sin (10 * t)), (900 * Real.cos (10 * t))) = (vx, vy)) :=
sorry

end crank_slider_motion_l98_98581


namespace odd_function_value_at_neg_2_l98_98789

def f (x : ℝ) : ℝ := 
if x ≥ 0 then x^2 + x else -((x^2) + -x)

theorem odd_function_value_at_neg_2 : f (-2) = -6 :=
by
  sorry

end odd_function_value_at_neg_2_l98_98789


namespace greatest_x_prime_condition_l98_98929

def is_prime (n : ℤ) : Prop := 
  n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem greatest_x_prime_condition :
  ∃ (x : ℤ), │5 * x^2 - 52 * x + 21│ ∈ {p : ℤ | is_prime p} ∧ ∀ (y : ℤ), │5 * y^2 - 52 * y + 21│ ∈ {p : ℤ | is_prime p} → y ≤ x :=
  sorry

end greatest_x_prime_condition_l98_98929


namespace range_of_a_if_p_and_q_l98_98333

variable (a : ℝ)

def proposition_p := ∀ x ∈ Icc (0:ℝ) 1, a ≥ 2^x
def proposition_q := ∃ x : ℝ, x^2 + 4 * x + a = 0

theorem range_of_a_if_p_and_q (hp : proposition_p a) (hq : proposition_q a) : 2 ≤ a ∧ a ≤ 4 :=
begin
  sorry
end

end range_of_a_if_p_and_q_l98_98333


namespace simplify_trig_expression_l98_98061

theorem simplify_trig_expression (α : ℝ) : 
  (sin (2 * real.pi - α) * cos (real.pi + α) * tan (real.pi / 2 + α)) /
  (cos (real.pi - α) * sin (3 * real.pi - α) * cot (-α)) = -1 := 
by 
  sorry

end simplify_trig_expression_l98_98061


namespace exists_int_x_l98_98373

theorem exists_int_x (K M N : ℤ) (h1 : K ≠ 0) (h2 : M ≠ 0) (h3 : N ≠ 0) (h_coprime : Int.gcd K M = 1) :
  ∃ x : ℤ, K ∣ (M * x + N) :=
by
  sorry

end exists_int_x_l98_98373


namespace min_value_of_a_l98_98736

def f (a x : ℝ) : ℝ := a * Real.log x - x + 2

theorem min_value_of_a (a : ℝ) (h : a ≠ 0)
    (h_forall_exists : ∀ x₁ ∈ Set.Icc (1 : ℝ) Real.exp, ∃ x₂ ∈ Set.Icc (1 : ℝ) Real.exp, f a x₁ + f a x₂ ≥ 3) :
    a ≥ Real.exp :=
sorry

end min_value_of_a_l98_98736


namespace find_a7_l98_98278

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98278


namespace largest_divisor_of_product_of_consecutive_evens_l98_98551

theorem largest_divisor_of_product_of_consecutive_evens (n : ℤ) : 
  ∃ d, d = 8 ∧ ∀ n, d ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end largest_divisor_of_product_of_consecutive_evens_l98_98551


namespace harmonic_conjugates_extension_l98_98068

noncomputable def harmonic_conjugates (A B C D : Point) : Prop :=
  ∃ (λ μ : ℝ), 
    (overvec A C = λ • overvec A B) ∧ 
    (overvec A D = μ • overvec A B) ∧ 
    (1 / λ + 1 / μ = 2)

theorem harmonic_conjugates_extension (A B C D : Point):
  harmonic_conjugates A B C D → 
  ¬(λ > 1 ∧ μ > 1 → λ * μ / (λ + μ) = 0) :=
by sorry

end harmonic_conjugates_extension_l98_98068


namespace min_p_for_3_order_Γ_l98_98326

variable {p : ℕ} (hp : p > 2)
variable {x : ℕ → ℕ}

-- Definition: A sequence is a 3-order Γ sequence if there exists two sets of consecutive t terms that are equal for t=3
def is_γ_sequence_3_order(p: ℕ) (x: ℕ → ℕ): Prop :=
  ∃ t: ℕ, (2 ≤ t ∧ t ≤ p - 1) ∧ ∃ i j, (i + t ≤ p ∧ j + t ≤ p ∧ (i ≠ j) ∧ list_pair_eq x i t (j + t))

-- Definition: list_pair_eq checks if two sets of consecutive terms are equal
def list_pair_eq {p : ℕ} (x : ℕ → ℕ) (i t : ℕ) : bool :=
  list_eq (list_of_fn (λ n, x (i + n)) t) (list_of_fn (λ n, x (j + n)) t)

-- Theorem: Minimum p for which a sequence is definitely a 3-order Γ sequence is 11
theorem min_p_for_3_order_Γ (hp: p > 2) :
  (∀ (x : ℕ → ℕ), is_γ_sequence_3_order p x) ↔ p = 11 :=
begin
  sorry
end

end min_p_for_3_order_Γ_l98_98326


namespace point_closer_to_B_l98_98412

def triangle (A B C : Type*) := ∃ a b c : Real, a + b + c = π
def inside_triangle (P A B C : Real) := True  -- Simplified condition for randomly selected point P 

def prob_point_closer_to_B (P A B C : Real) : Real := 
  if inside_triangle P A B C then
    if A = 6 ∧ B = 8 ∧ C = 10 then
      1 / 2
    else 
      0
  else 
    0

theorem point_closer_to_B : ∀ (P A B C : Real), A = 6 → B = 8 → C = 10 → 
  inside_triangle P A B C → prob_point_closer_to_B P A B C = 1 / 2 :=
by 
  intros P A B C h1 h2 h3 h4
  sorry

end point_closer_to_B_l98_98412


namespace solve_sqrt_equation_l98_98520

noncomputable def number_sqrt_equation (x : ℕ) : Prop :=
  real.sqrt 289 - real.sqrt x / real.sqrt 25 = 12

theorem solve_sqrt_equation (x : ℕ) (h : number_sqrt_equation x) : x = 625 :=
sorry

end solve_sqrt_equation_l98_98520


namespace average_speed_of_participant_l98_98572

noncomputable def average_speed (d : ℝ) : ℝ :=
  let total_distance := 4 * d
  let total_time := (d / 6) + (d / 12) + (d / 18) + (d / 24)
  total_distance / total_time

theorem average_speed_of_participant :
  ∀ (d : ℝ), d > 0 → average_speed d = 11.52 :=
by
  intros d hd
  unfold average_speed
  sorry

end average_speed_of_participant_l98_98572


namespace placement_of_numbers_is_periodic_l98_98021

theorem placement_of_numbers_is_periodic :
  ∃ f : ℤ × ℤ → ℕ, (∀ n : ℕ, ∃ p : ℤ × ℤ, f p = n) ∧
  (∀ a b c : ℤ, (a ≠ 0 ∨ b ≠ 0) ∧ c ≠ 0 →
  ∀ (x₁ y₁ x₂ y₂ : ℤ), (a * x₁ + b * y₁ = c) ∧ (a * x₂ + b * y₂ = c) →
  ∃ k : ℤ, f(x₁ + k * b, y₁ - k * a) = f(x₁, y₁) ∧
           f(x₂ + k * b, y₂ - k * a) = f(x₂, y₂)
  )
  :=
sorry

end placement_of_numbers_is_periodic_l98_98021


namespace LCM_5_711_1033_l98_98122

def is_prime (n : ℕ) : Prop := Nat.Prime n

def factorization (n : ℕ) : List (ℕ × ℕ) :=
  match n with
  | 5     => [(5, 1)]
  | 711   => [(3, 2), (79, 1)]
  | 1033  => [(1033, 1)]
  | _     => []

def LCM (a b c : ℕ) : ℕ := 
  (a :: b :: c :: []).foldr Nat.lcm 1

theorem LCM_5_711_1033 :
  is_prime 5 →
  (∃ l1 l2, factorization 711 = [(3, l1), (79, l2)]) →
  is_prime 1033 →
  LCM 5 711 1033 = 3683445 :=
by
  intros
  sorry

end LCM_5_711_1033_l98_98122


namespace tan_of_alpha_intersects_unit_circle_l98_98792

theorem tan_of_alpha_intersects_unit_circle (α : ℝ) (hα : ∃ P : ℝ × ℝ, P = (12 / 13, -5 / 13) ∧ ∀ x y : ℝ, P = (x, y) → x^2 + y^2 = 1) : 
  Real.tan α = -5 / 12 :=
by
  -- proof to be completed
  sorry

end tan_of_alpha_intersects_unit_circle_l98_98792


namespace more_than_1000_triplets_l98_98862

theorem more_than_1000_triplets :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 1000 < S.card ∧ 
  ∀ (a b c : ℕ), (a, b, c) ∈ S → a^15 + b^15 = c^16 :=
by sorry

end more_than_1000_triplets_l98_98862


namespace smallest_positive_integer_l98_98553

theorem smallest_positive_integer (n : ℕ) :
  (n % 45 = 0 ∧ n % 60 = 0 ∧ n % 25 ≠ 0 ↔ n = 180) :=
sorry

end smallest_positive_integer_l98_98553


namespace largest_mersenne_prime_lt_500_l98_98964

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ is_prime (2^p - 1)

theorem largest_mersenne_prime_lt_500 : 
  ∀ n, is_mersenne_prime n → 2^n - 1 < 500 → 2^n - 1 ≤ 127 :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_500_l98_98964


namespace gcd_pow_sub_one_l98_98477

open Nat

theorem gcd_pow_sub_one (a m n : ℕ) : 
  let d := gcd m n in gcd (a^m - 1) (a^n - 1) = a^d - 1 :=
by
  trivial

end gcd_pow_sub_one_l98_98477


namespace find_a_b_l98_98035

noncomputable def omega : ℂ := sorry -- Assume omega is some complex number meeting the conditions

def alpha (ω : ℂ) : ℂ := ω + ω^3 + ω^5
def beta (ω : ℂ) : ℂ := ω^2 + ω^4 + ω^6 + ω^7

theorem find_a_b
  (h_ω8 : ω^8 = 1)
  (h_ω_ne_1 : ω ≠ 1) :
  let α := alpha ω,
      β := beta ω in
  (α + β = -1) ∧ (α * β = 2) :=
sorry

end find_a_b_l98_98035


namespace no_solution_fraction_eq_l98_98786

theorem no_solution_fraction_eq (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → (a * x / (x - 1) + 3 / (1 - x) = 2) → false) ↔ a = 2 :=
by
  sorry

end no_solution_fraction_eq_l98_98786


namespace minimum_swaps_needed_to_alternate_l98_98190

-- Define the initial condition of the balls
def initialBalls : List (Sum Nat Nat) := 
  (List.range' 1 10).map Sum.inl ++  -- Black balls from 1 to 10
  (List.range' 11 10).map Sum.inr    -- White balls from 11 to 20

-- Define the target alternating pattern for black and white balls
def targetPattern : List (Sum Nat Nat) := 
  (List.range 20).map (λ i =>
    if i % 2 = 0 then Sum.inl (i / 2 + 1) else Sum.inr (i / 2 + 11))

-- The proof problem:
theorem minimum_swaps_needed_to_alternate : 
  ∃ (swaps : Nat), swaps = 5 ∧ (
    ∀ (arr : List (Sum Nat Nat)), 
    arr = initialBalls → 
    (∃ (arr' : List (Sum Nat Nat)), 
      arr' = List.swapN arr swaps ∧ arr' = targetPattern)
  ) :=
sorry

end minimum_swaps_needed_to_alternate_l98_98190


namespace completing_the_square_x_squared_minus_4x_plus_1_eq_0_l98_98869

theorem completing_the_square_x_squared_minus_4x_plus_1_eq_0 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x
  intros h
  sorry

end completing_the_square_x_squared_minus_4x_plus_1_eq_0_l98_98869


namespace katya_notebooks_l98_98172

theorem katya_notebooks (rubles : ℕ) (cost_per_notebook : ℕ) (stickers_per_notebook : ℕ) (sticker_rate : ℕ):
  rubles = 150 -> cost_per_notebook = 4 -> stickers_per_notebook = 1 -> sticker_rate = 5 ->
  let initial_notebooks := rubles / cost_per_notebook in
  let remaining_rubles := rubles % cost_per_notebook in
  let initial_stickers := initial_notebooks * stickers_per_notebook in
  ∃ final_notebooks : ℕ, 
    (initial_notebooks = 37 ∧ remaining_rubles = 2 ∧ initial_stickers = 37 ∧ 
    (final_notebooks = initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate) + (((initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate)) * stickers_per_notebook) / sticker_rate)) = 46) :=
begin
  intros,
  sorry
end

end katya_notebooks_l98_98172


namespace perimeter_of_ABCDE_is_28_diagonal_length_AC_is_sqrt_89_l98_98403

-- Define necessary points and properties
noncomputable def A : Point := ⟨0, 8⟩
noncomputable def B : Point := ⟨5, 8⟩
noncomputable def C : Point := ⟨5, 4⟩
noncomputable def D : Point := ⟨0, 0⟩
noncomputable def E : Point := ⟨10, 0⟩

-- Define the distances between the points according to the problem statement
def AB : ℝ := dist A B
def BC : ℝ := dist B C
def DE : ℝ := dist D E
def AD : ℝ := dist A D

-- Define the perimeter and diagonal length 
def perimeter_ABCDE : ℝ := AB + BC + (dist C E) + DE + AD
def diagonal_AC : ℝ := dist A C

-- Proof statements
theorem perimeter_of_ABCDE_is_28 : perimeter_ABCDE = 28 := by sorry
theorem diagonal_length_AC_is_sqrt_89 : diagonal_AC = Real.sqrt 89 := by sorry

end perimeter_of_ABCDE_is_28_diagonal_length_AC_is_sqrt_89_l98_98403


namespace parallel_vectors_l98_98374

noncomputable def vector_a : (ℤ × ℤ) := (1, 3)
noncomputable def vector_b (m : ℤ) : (ℤ × ℤ) := (-2, m)

theorem parallel_vectors (m : ℤ) (h : vector_a = (1, 3) ∧ vector_b m = (-2, m))
  (hp: ∃ k : ℤ, ∀ (a1 a2 b1 b2 : ℤ), (a1, a2) = vector_a ∧ (b1, b2) = (1 + k * (-2), 3 + k * m)):
  m = -6 :=
by
  sorry

end parallel_vectors_l98_98374


namespace mike_total_money_l98_98853

theorem mike_total_money (num_bills : ℕ) (value_per_bill : ℕ)
  (h1 : num_bills = 9)
  (h2 : value_per_bill = 5) :
  num_bills * value_per_bill = 45 :=
by
  rw [h1, h2]
  rfl

end mike_total_money_l98_98853


namespace katya_notebooks_l98_98174

theorem katya_notebooks (rubles : ℕ) (cost_per_notebook : ℕ) (stickers_per_notebook : ℕ) (sticker_rate : ℕ):
  rubles = 150 -> cost_per_notebook = 4 -> stickers_per_notebook = 1 -> sticker_rate = 5 ->
  let initial_notebooks := rubles / cost_per_notebook in
  let remaining_rubles := rubles % cost_per_notebook in
  let initial_stickers := initial_notebooks * stickers_per_notebook in
  ∃ final_notebooks : ℕ, 
    (initial_notebooks = 37 ∧ remaining_rubles = 2 ∧ initial_stickers = 37 ∧ 
    (final_notebooks = initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate) + (((initial_notebooks + (initial_stickers / sticker_rate) + ((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) / sticker_rate) + (((initial_stickers % sticker_rate + (initial_stickers / sticker_rate) * stickers_per_notebook) % sticker_rate + 1) / sticker_rate)) * stickers_per_notebook) / sticker_rate)) = 46) :=
begin
  intros,
  sorry
end

end katya_notebooks_l98_98174


namespace hexagon_ring_50th_l98_98206

theorem hexagon_ring_50th (n : ℕ) : n = 50 → 6 * n = 300 :=
by
  intro h
  rw [h]
  show 6 * 50 = 300
  sorry

end hexagon_ring_50th_l98_98206


namespace flower_bed_width_l98_98987

theorem flower_bed_width (length area : ℝ) (h_length : length = 4) (h_area : area = 143.2) :
  area / length = 35.8 :=
by
  sorry

end flower_bed_width_l98_98987


namespace find_k_l98_98510

theorem find_k (x y k : ℝ) (h₁ : 3 * x + y = k) (h₂ : -1.2 * x + y = -20) (hx : x = 7) : k = 9.4 :=
by
  sorry

end find_k_l98_98510


namespace min_distance_proof_l98_98728

-- Define the curve C parametric equations
def curve_C (theta : ℝ) : ℝ × ℝ :=
  (3 * Real.cos theta, 2 * Real.sin theta)

-- Define the line l polar equation
def polar_eq_line (rho theta : ℝ) : Prop :=
  rho * (Real.cos theta - 2 * Real.sin theta) = 12

-- Convert the polar equation to Cartesian form
def line_l_cartesian : Prop :=
  ∀ (x y : ℝ), (∃ theta rho, x = rho * Real.cos theta ∧ y = rho * Real.sin theta ∧ polar_eq_line rho theta) →
  x - 2 * y - 12 = 0

-- Define the minimum distance from a point P on curve C to line l
def min_distance (d : ℝ) : Prop :=
  ∀ theta, 
    let P := curve_C theta in
    d = Real.abs (3 * Real.cos theta - 4 * Real.sin theta - 12) / Real.sqrt 5 →
    d = 7 * Real.sqrt 5 / 5

-- Lean statement for the proof exercise
theorem min_distance_proof :
  line_l_cartesian ∧ min_distance (7 * Real.sqrt 5 / 5) :=
by
  sorry

end min_distance_proof_l98_98728


namespace stream_speed_l98_98146

theorem stream_speed (u v : ℝ) (h1 : 27 = 9 * (u - v)) (h2 : 81 = 9 * (u + v)) : v = 3 :=
by
  sorry

end stream_speed_l98_98146


namespace no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l98_98377

theorem no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 10000 ∧ (n % 10 = 0) ∧ (Prime n) → False :=
by
  sorry

end no_positive_integers_between_100_and_10000_are_multiples_of_10_and_prime_l98_98377


namespace area_less_than_4_l98_98839

noncomputable def convex_polygon (K : Type*) : Prop := sorry
noncomputable def no_nonzero_lattice_points (K : Type*) : Prop := sorry
noncomputable def area (K : Type*) : ℝ := sorry
noncomputable def is_positioned (K : Type*) : Prop := 
  ∀ (Q : ℕ) (hQ : Q ∈ {1, 2, 3, 4}), 
  area (K ∩ (quadrant Q)) = 1/4 * area K

theorem area_less_than_4 (K : Type*) 
  (h1 : convex_polygon K) 
  (h2 : is_positioned K)
  (h3 : no_nonzero_lattice_points K) : 
  area K < 4 := 
sorry

end area_less_than_4_l98_98839


namespace part1_part2_l98_98720

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then -3 * x + (1/2)^x - 1 else sorry -- Placeholder: function definition incomplete for x ≤ 0

def odd (f : ℝ → ℝ) :=
∀ x, f (-x) = - f x

def monotonic_decreasing (f : ℝ → ℝ) :=
∀ x y, x < y → f x > f y

axiom f_conditions :
  monotonic_decreasing f ∧
  odd f ∧
  (∀ x, x > 0 → f x = -3 * x + (1/2)^x - 1)

theorem part1 : f (-1) = 3.5 :=
by
  sorry

theorem part2 (t : ℝ) (k : ℝ) :
  (∀ t, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1/3 :=
by
  sorry

end part1_part2_l98_98720


namespace goldbach_10000_l98_98004

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem goldbach_10000 :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p q : ℕ), (p, q) ∈ S → is_prime p ∧ is_prime q ∧ p + q = 10000) ∧ S.card > 3 :=
sorry

end goldbach_10000_l98_98004


namespace AB_equals_A2B2_l98_98152

noncomputable def geometry_problem (P A B : Point) (Ox Oy : Line) (A1 B1 A2 B2 : Point) (OP AB : Line) : Prop :=
  let O : Point := Ox ∩ Oy in
  let A := Ox ∩ P and B := Oy ∩ P in
  let A1 := (P ⟂ AB) ∩ Ox and B1 := (P ⟂ AB) ∩ Oy in
  let A2 := (A1 ⟂ OP) and B2 := (B1 ⟂ OP) in
  A1 ∈ Ox ∧ B1 ∈ Oy ∧ A2 ∈ AB ∧ B2 ∈ AB

theorem AB_equals_A2B2
  (P A B : Point) (Ox Oy : Line) (A1 B1 A2 B2 : Point) 
  (OP AB : Line) 
  (hGeom : geometry_problem P A B Ox Oy A1 B1 A2 B2 OP AB) : 
  dist A B = dist A2 B2 := 
sorry

end AB_equals_A2B2_l98_98152


namespace value_of_box_l98_98723

theorem value_of_box (a b c : ℕ) (h1 : a + b = c) (h2 : a + b + c = 100) : c = 50 :=
sorry

end value_of_box_l98_98723


namespace convex_polyhedron_at_least_three_equal_edges_l98_98661

theorem convex_polyhedron_at_least_three_equal_edges
  (P : Type) [polyhedron P]
  (h1 : ∀ v ∈ vertices P, ∃ e1 e2 e3 ∈ edges P, incident v e1 ∧ incident v e2 ∧ incident v e3 ∧ e1 = e2)
  : ∃ e4 e5 e6 ∈ edges P, e4 = e5 ∧ e5 = e6 := 
sorry

end convex_polyhedron_at_least_three_equal_edges_l98_98661


namespace probability_of_king_then_ace_is_4_over_663_l98_98113

noncomputable def probability_king_and_ace {α : Type*} [ProbabilityTheory α] 
  (deck : Finset α) (is_king : α → Prop) (is_ace : α → Prop) 
  (h1 : deck.card = 52) (h2 : (deck.filter is_king).card = 4) 
  (h3 : (deck.filter is_ace).card = 4) : ℚ :=
  (4 / 52) * (4 / 51)

theorem probability_of_king_then_ace_is_4_over_663 
  {α : Type*} [ProbabilityTheory α] 
  (deck : Finset α) (is_king : α → Prop) (is_ace : α → Prop) 
  (h1 : deck.card = 52) (h2 : (deck.filter is_king).card = 4) 
  (h3 : (deck.filter is_ace).card = 4) :
  probability_king_and_ace deck is_king is_ace h1 h2 h3 = 4 / 663 := 
sorry

end probability_of_king_then_ace_is_4_over_663_l98_98113


namespace find_fraction_l98_98577

variable (n : ℚ) (x : ℚ)

theorem find_fraction (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 0.25 := by
  sorry

end find_fraction_l98_98577


namespace find_total_rhinestones_l98_98846

-- Definitions of the conditions
def total_rhinestones (n : ℕ) : ℕ := n
def bought_percentage_of_n (n : ℕ) : ℝ := 0.35 * n
def already_has_percentage_of_n (n : ℕ) : ℝ := 0.20 * n
def still_needs_rhinestones (n : ℕ) : ℕ := 51

-- The main statement to be proven
theorem find_total_rhinestones (n : ℕ) 
  (h1 : 0.35 * n + 0.20 * n = 0.55 * n)
  (h2 : 0.45 * n = 51) : n = 114 := 
sorry

end find_total_rhinestones_l98_98846


namespace distinct_remainders_partition_l98_98137

theorem distinct_remainders_partition {S : Finset ℤ} {m n k : ℕ}
  (hm : S.card = m)
  (hrem : ∀ (x ∈ S) (y ∈ S), x ≠ y → (x % n) ≠ (y % n))
  (hk : k ≤ m) :
  ∃ (P : Finset (Finset ℤ)), P.card = k ∧ (∀ A ∈ P, A.nonempty) ∧ (∀ A B ∈ P, A ≠ B → (∑ x in A, x % n) ≠ (∑ x in B, x % n)) :=
by
  sorry

end distinct_remainders_partition_l98_98137


namespace parabola_curve_intersection_l98_98644

def curve (x y : ℝ) := y^2 - 2 * y - x^2 - 2 * y - 1

theorem parabola_curve_intersection (B : ℝ) (hB : 0 < B) :
  ∃ u v x y : ℝ, (y = B * x^2 ∧ curve x y = 0) ∧
                 (∃ x1 x2 x3 x4 : ℝ,
                   ∀ y, y = B * x1^2 ∨ y = B * x2^2 ∨ y = B * x3^2 ∨ y = B * x4^2) :=
begin
  sorry,
end

end parabola_curve_intersection_l98_98644


namespace brick_length_is_20_cm_l98_98971

-- Define the conditions given in the problem
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def num_bricks : ℕ := 20000
def brick_width_cm : ℝ := 10
def total_area_cm2 : ℝ := 4000000

-- Define the goal to prove that the length of each brick is 20 cm
theorem brick_length_is_20_cm :
  (total_area_cm2 = num_bricks * (brick_width_cm * length)) → (length = 20) :=
by
  -- Assume the given conditions
  sorry

end brick_length_is_20_cm_l98_98971


namespace probability_point_in_spherical_region_l98_98614

theorem probability_point_in_spherical_region :
  let cube_region := {p : ℝ × ℝ × ℝ | (-2 : ℝ) ≤ p.1 ∧ p.1 ≤ 2 ∧ (-2 : ℝ) ≤ p.2 ∧ p.2 ≤ 2 ∧ (-2 : ℝ) ≤ p.3 ∧ p.3 ≤ 2}
  let sphere_region := {p : ℝ × ℝ × ℝ | p.1^2 + p.2^2 + p.3^2 ≤ 4}
  P := ((measure_theory.volume sphere_region) / (measure_theory.volume cube_region))
  in P = (Real.pi / 6) :=
sorry

end probability_point_in_spherical_region_l98_98614


namespace shovel_driveway_time_l98_98432

theorem shovel_driveway_time (j_time m_time : ℝ) (j_rate m_rate : ℝ) (combined_time: ℝ) : 
  j_time = 50 ∧ m_time = 20 ∧ j_rate = 1 / j_time ∧ m_rate = 1 / m_time ∧ combined_time = 1 / (j_rate + m_rate) → 
  Int.round (combined_time) = 14 :=
by
  intros
  sorry

end shovel_driveway_time_l98_98432


namespace janet_saving_l98_98824

def tile_cost_difference_saving : ℕ :=
  let turquoise_cost_per_tile := 13
  let purple_cost_per_tile := 11
  let area_wall1 := 5 * 8
  let area_wall2 := 7 * 8
  let total_area := area_wall1 + area_wall2
  let tiles_per_square_foot := 4
  let number_of_tiles := total_area * tiles_per_square_foot
  let cost_difference_per_tile := turquoise_cost_per_tile - purple_cost_per_tile
  number_of_tiles * cost_difference_per_tile

theorem janet_saving : tile_cost_difference_saving = 768 := by
  sorry

end janet_saving_l98_98824


namespace optimal_messenger_strategy_l98_98057

theorem optimal_messenger_strategy (p : ℝ) (hp : 0 < p ∧ p < 1) :
  (p < 1/3 → ∃ n : ℕ, n = 4 ∧ ∀ (k : ℕ), k = 10) ∧ 
  (1/3 ≤ p → ∃ n : ℕ, n = 2 ∧ ∀ (m : ℕ), m = 20) :=
by
  sorry

end optimal_messenger_strategy_l98_98057


namespace arcsin_eq_pi_div_two_solve_l98_98492

theorem arcsin_eq_pi_div_two_solve :
  ∀ (x : ℝ), (Real.arcsin x + Real.arcsin (3 * x) = Real.pi / 2) → x = Real.sqrt 10 / 10 :=
by
  intro x h
  sorry -- Proof is omitted as per instructions

end arcsin_eq_pi_div_two_solve_l98_98492


namespace non_overlapping_area_greater_than_one_ninth_l98_98185

theorem non_overlapping_area_greater_than_one_ninth (circles : Set (Set ℝ)) :
  (∀ c ∈ circles, is_circle c) →
  (Union setOf circs = 1) →
  (∃ non_overlapping ⊆ circles, 
    (∀ c₁ c₂ ∈ non_overlapping, c₁ ≠ c₂ → disjoint c₁ c₂) ∧ 
    (measure (Union setOf non_overlapping) > 1 / 9)) :=
sorry

end non_overlapping_area_greater_than_one_ninth_l98_98185


namespace roy_consumes_tablets_in_225_minutes_l98_98481

variables 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ)

def total_time_to_consume_all_tablets 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ) : ℕ :=
  (total_tablets - 1) * time_per_tablet

theorem roy_consumes_tablets_in_225_minutes 
  (h1 : total_tablets = 10) 
  (h2 : time_per_tablet = 25) : 
  total_time_to_consume_all_tablets total_tablets time_per_tablet = 225 :=
by
  -- Proof goes here
  sorry

end roy_consumes_tablets_in_225_minutes_l98_98481


namespace max_moves_21x21_max_moves_20x21_l98_98150

-- Definitions related to the initial conditions
def grid {m n : Nat} := 
  List (List (Bool))

-- Function that checks if all bulbs in the grid are off
def is_all_off {m n : Nat} (g : grid m n) : Bool := 
  g.all (fun row => row.all (fun bulb => bulb = false))

-- Function to represent drawing a line and turning on bulbs on one side
def make_move (g : grid m n) (line : (Nat × Nat) → Bool) : grid m n := sorry

-- Predicate that states a move turns on at least one bulb
def move_turns_on_at_least_one {m n : Nat} (g : grid m n) (new_g : grid m n) : Prop := sorry

-- Predicate that states all bulbs are turned on
def all_bulbs_on {m n : Nat} (g : grid m n) : Prop := sorry

-- Proof statement for the 21 x 21 grid
theorem max_moves_21x21 : 
  ∃ (moves : List ((Nat × Nat) → Bool)), 
  (∀ move ∈ moves, ∃ g' : grid 21 21, 
    is_all_off g → 
    (move_turns_on_at_least_one g g') ∧ 
    g = make_move g move) ∧ 
    length moves = 3 ∧ 
    all_bulbs_on (foldl make_move initial_grid moves) := sorry

-- Proof statement for the 20 x 21 grid
theorem max_moves_20x21 : 
  ∃ (moves : List ((Nat × Nat) → Bool)), 
  (∀ move ∈ moves, ∃ g' : grid 20 21, 
    is_all_off g → 
    (move_turns_on_at_least_one g g') ∧ 
    g = make_move g move) ∧ 
    length moves = 4 ∧ 
    all_bulbs_on (foldl make_move initial_grid moves) := sorry

end max_moves_21x21_max_moves_20x21_l98_98150


namespace symmetric_points_origin_l98_98711

theorem symmetric_points_origin (a b : ℤ) (h1 : a = -5) (h2 : b = -1) : a - b = -4 :=
by
  sorry

end symmetric_points_origin_l98_98711


namespace arcsin_eq_pi_div_two_solve_l98_98491

theorem arcsin_eq_pi_div_two_solve :
  ∀ (x : ℝ), (Real.arcsin x + Real.arcsin (3 * x) = Real.pi / 2) → x = Real.sqrt 10 / 10 :=
by
  intro x h
  sorry -- Proof is omitted as per instructions

end arcsin_eq_pi_div_two_solve_l98_98491


namespace shoe_probability_l98_98110

theorem shoe_probability : ∃ (m n : ℕ), Nat.coprime m n ∧ (m + n = 19) ∧
    let configurations := 20.factorial
    let valid_configurations := 19.factorial + (20.choose 10 / 2) * (9.factorial * 9.factorial)
    (valid_configurations / configurations = (1 / 18)) := 
by
  sorry

end shoe_probability_l98_98110


namespace infinite_quadruples_inequality_quadruple_l98_98087

theorem infinite_quadruples 
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  ∃ (a p q r : ℕ), 
    1 < p ∧ 1 < q ∧ 1 < r ∧
    p ∣ (a * q * r + 1) ∧
    q ∣ (a * p * r + 1) ∧
    r ∣ (a * p * q + 1) :=
sorry

theorem inequality_quadruple
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end infinite_quadruples_inequality_quadruple_l98_98087


namespace remainder_of_sequence_sum_mod_16_l98_98934

theorem remainder_of_sequence_sum_mod_16 : 
  (let seq := list.range' 2101 (15*2) 
   in (seq.filter (λ x, x % 2 = 1)).sum) % 16 = 0 := 
by
  sorry

end remainder_of_sequence_sum_mod_16_l98_98934


namespace irrational_number_problem_l98_98564

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number_problem :
  ∀ x ∈ ({(0.4 : ℝ), (2 / 3 : ℝ), (2 : ℝ), - (Real.sqrt 5)} : Set ℝ), 
  is_irrational x ↔ x = - (Real.sqrt 5) :=
by
  intros x hx
  -- Other proof steps can go here
  sorry

end irrational_number_problem_l98_98564


namespace positive_difference_of_perimeters_eq_zero_l98_98114

-- Definition of the first rectangle
def first_rectangle_perimeter (rows : ℕ) (cols : ℕ) : ℕ :=
  2 * (rows + cols)

-- Definition of the second rectangle
def second_rectangle_perimeter (rows : ℕ) (cols : ℕ) (missing_squares : ℕ) : ℕ :=
  2 * (rows + cols) - missing_squares + missing_squares

theorem positive_difference_of_perimeters_eq_zero
  (rows1 cols1 : ℕ) (rows2 cols2 : ℕ) (missing_squares : ℕ)
  (h1 : rows1 = 2) (h2 : cols1 = 6)
  (h3 : rows2 = 3) (h4 : cols2 = 5) (h5 : missing_squares = 2) :
  |first_rectangle_perimeter rows1 cols1 - second_rectangle_perimeter rows2 cols2 missing_squares| = 0 := 
sorry

end positive_difference_of_perimeters_eq_zero_l98_98114


namespace geometric_figure_l98_98104

noncomputable def Sphere (center : ℝ^3) (radius : ℝ) : Type := 
{c : ℝ^3 // (∥c - center∥ = radius)}

variables (S1 S2 S3 : Sphere ℝ^3 ℝ)
variables (center1 center2 center3 : ℝ^3)
variables (r1 r2 r3 : ℝ)

-- Conditions
axiom diff_center1_center2 : center1 ≠ center2
axiom diff_center2_center3 : center2 ≠ center3
axiom diff_center1_center3 : center1 ≠ center3
axiom not_collinear : ¬ collinear ℝ [center1, center2, center3]

-- Question to prove
theorem geometric_figure (
  S1 ≃ Sphere center1 r1 ∧ 
  S2 ≃ Sphere center2 r2 ∧ 
  S3 ≃ Sphere center3 r3 ∧ 
  ThreeSphereCondition S1 S2 S3 not_collinear) : 
  ∃ (L : Set (ℝ^3)), 
    L.card = 4 ∧ 
    ∃ (P : Set (ℝ^3)), 
      P.card = 6 ∧ 
      ∀ l ∈ L, ∃! p ∈ P, p ∈ l ∧ 
      ∀(p ∈ P), ∃ (l ∈ L), p ∈ l := 
sorry

end geometric_figure_l98_98104


namespace arrangement_count_proof_exactly_2_2_1_proof_at_least_1_math_proof_l98_98527

-- Definitions of the problem
def math_books := 3
def physics_books := 4
def chemistry_books := 2

-- The total number of books
def total_books := math_books + physics_books + chemistry_books

-- Ways to arrange the books on the shelf
def arrangement_count := (Mathlib.factorial math_books) * (Mathlib.factorial physics_books) * (Mathlib.factorial chemistry_books) * (Mathlib.factorial 3)

-- Number of ways to choose books
def choose_math_books := Mathlib.choose math_books 2
def choose_physics_books := Mathlib.choose physics_books 2
def choose_chemistry_books := Mathlib.choose chemistry_books 1
def choose_with_exactly_2_2_1 := choose_math_books * choose_physics_books * choose_chemistry_books

-- Total ways to choose 5 books out of 9
def choose_total := Mathlib.choose total_books 5
-- Ways to choose 5 books without any math books
def choose_without_math := Mathlib.choose (physics_books + chemistry_books) 5
-- At least one math book
def choose_with_at_least_1_math := choose_total - choose_without_math

-- The statements to prove
theorem arrangement_count_proof : arrangement_count = 1728 := by
  sorry

theorem exactly_2_2_1_proof : choose_with_exactly_2_2_1 = 36 := by
  sorry

theorem at_least_1_math_proof : choose_with_at_least_1_math = 120 := by
  sorry

end arrangement_count_proof_exactly_2_2_1_proof_at_least_1_math_proof_l98_98527


namespace remainder_T_mod_1000_l98_98442

noncomputable def sum_of_distinct_digits (n : ℕ) : ℕ :=
  if ∃ thousand hundred ten unit, thousand ≠ hundred ∧ thousand ≠ ten ∧ thousand ≠ unit ∧ hundred ≠ ten ∧ hundred ≠ unit ∧ ten ≠ unit
  then n
  else 0

def T : ℕ := ∑ n in finset.range (10000), sum_of_distinct_digits n

theorem remainder_T_mod_1000 : T % 1000 = 465 := sorry

end remainder_T_mod_1000_l98_98442


namespace participation_plans_count_l98_98260

theorem participation_plans_count :
  ∃ (A B C D : Type), 
  fintype A ∧ fintype B ∧ fintype C ∧ fintype D ∧
  (∀ (set : finset (A ⊕ B ⊕ C ⊕ D)), set.card = 4) ∧
  (∀ (select : finset (A ⊕ B ⊕ C ⊕ D)), 
    select.card = 3 ∧
    (A ∈ select) → 
    (finset.filter (λ x, x ≠ A) select).card = 2 ∧
    (finset.permutations select).card = 6) →
    3 * 6 = 18 := 
by
  sorry

end participation_plans_count_l98_98260


namespace find_constant_l98_98894

theorem find_constant 
  (f : ℝ → ℝ) 
  (c : ℝ) 
  (h₁ : ∀ x, f x = x + 4) 
  (h₂ : ∀ x, (3 * f(x - 2)) / f c + 4 = f(2 * x + 1)) 
  (x_solution : x = 0.4) : c = 0 := 
by 
  sorry

end find_constant_l98_98894


namespace total_amount_is_correct_l98_98994

noncomputable def total_amount (y_share : ℝ) (u v w x y z : ℝ) : ℝ :=
  let ru := y_share / y
  ru * (1 + v + w + x + y + z)

theorem total_amount_is_correct :
  let y_share := 178.34
  let u := 1
  let v := 0.31625
  let w := 0.34517
  let x := 0.45329
  let y := 0.52761
  let z := 0.61295
  total_amount y_share u v w x y z ≈ 1100.57 :=
by sorry

end total_amount_is_correct_l98_98994


namespace sum_of_m_and_b_l98_98658

-- Define the points and the line
def point1 := (2 : ℝ, -1 : ℝ)
def point2 := (5 : ℝ, 3 : ℝ)

-- Define the slope-intercept form of the line
def y_equals_mx_plus_b (m b x : ℝ) := m * x + b

-- Define the slope of the line
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the y-intercept using one of the points
def find_y_intercept (m : ℝ) (p : ℝ × ℝ) : ℝ := p.2 - m * p.1

-- Statement of the problem to be proved
theorem sum_of_m_and_b : 
  let m := slope point1 point2 in 
  let b := find_y_intercept m point1 in 
  m + b = -7 / 3 :=
by 
  sorry

end sum_of_m_and_b_l98_98658


namespace distance_library_to_post_office_spencer_walked_l98_98434

variable (d_total d_house_library d_post_office_home : ℝ)

theorem distance_library_to_post_office (h1 : d_total = 0.8) 
                                         (h2 : d_house_library = 0.3)
                                         (h3 : d_post_office_home = 0.4) :
  d_total - d_house_library - d_post_office_home = 0.1 :=
by
  simp [h1, h2, h3]
  norm_num

variable (d_library_post_office : ℝ)
theorem spencer_walked (h4 : d_library_post_office = d_total - d_house_library - d_post_office_home) (h5 : h4 = 0.1) : 
  d_library_post_office = 0.1 :=
by
  simp [h4, h5]
  norm_num


end distance_library_to_post_office_spencer_walked_l98_98434


namespace find_a7_l98_98297

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98297


namespace sum_q_s_eq_2048_l98_98830

open Polynomial

noncomputable def T : finset (fin 12 → ℕ) := by {
  have H : finset (fin 12 → ℕ) := (finset.pi_finset (λ _, {0, 1})) with finset (fin 12 → ℕ),
  exact H
}

noncomputable def q_s (s : fin 12 → ℕ) : Polynomial ℝ := lagrange.interpolate (λ n, s (n : fin 12))

noncomputable def q : Polynomial ℝ := ∑ s in T, q_s s

theorem sum_q_s_eq_2048 : (∑ s in T, (q_s s).eval 12) = 2048 := by
  sorry

end sum_q_s_eq_2048_l98_98830


namespace monotonically_increasing_intervals_sin_value_l98_98734

noncomputable def f (x : Real) : Real := 2 * Real.cos (x - Real.pi / 3) * Real.cos x + 1

theorem monotonically_increasing_intervals :
  ∀ (k : Int), ∃ (a b : Real), a = k * Real.pi - Real.pi / 3 ∧ b = k * Real.pi + Real.pi / 6 ∧
                 ∀ (x y : Real), a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y :=
sorry

theorem sin_value 
  (α : Real) (hα : 0 < α ∧ α < Real.pi / 2) 
  (h : f (α + Real.pi / 12) = 7 / 6) : 
  Real.sin (7 * Real.pi / 6 - 2 * α) = 2 * Real.sqrt 2 / 3 :=
sorry

end monotonically_increasing_intervals_sin_value_l98_98734


namespace no_identical_concat_groups_l98_98819

theorem no_identical_concat_groups (k : ℕ) : 
  ¬ ∃ (G1 G2 : list ℕ), (G1 ++ G2 = list.range (k + 1) \ {0}) ∧ (list.foldl (λ a b, a * 10 ^ (nat.log 10 b + 1) + b) 0 G1 = list.foldl (λ a b, a * 10 ^ (nat.log 10 b + 1) + b) 0 G2) :=
by
  sorry

end no_identical_concat_groups_l98_98819


namespace length_of_brick_proof_l98_98974

noncomputable def length_of_brick (courtyard_length courtyard_width : ℕ) (brick_width : ℕ) (total_bricks : ℕ) : ℕ :=
  let total_area_cm := courtyard_length * courtyard_width * 10000
  total_area_cm / (brick_width * total_bricks)

theorem length_of_brick_proof :
  length_of_brick 25 16 10 20000 = 20 :=
by
  unfold length_of_brick
  sorry

end length_of_brick_proof_l98_98974


namespace ab_not_2_l98_98369

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then a^(x-1) - b else -real.log x / real.log 2 + 1

theorem ab_not_2 {a b : ℝ} (h1 : a > 0) (h2 : a ≠ 1) 
(h_monotonic : ∀ x y : ℝ, x ≤ y → f a b x ≤ f a b y) : 
a * b ≠ 2 :=
sorry

end ab_not_2_l98_98369


namespace probability_odd_and_less_than_5000_l98_98075

def isValidOdd (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 7

def isValidLeading (d : ℕ) : Prop := d = 1 ∨ d = 3

def possibleDig : finset ℕ := {1, 3, 7, 8}

def countValidNumbers : ℕ :=
  let valid_lead_digits := {1, 3}
  let valid_unit_digits := {1, 3, 7}
  valid_lead_digits.sum (λ ld, valid_unit_digits.sum (λ ud, if ld = ud then 2 else 2 * 2))

def countTotalNumbers : ℕ := 12

def probabilityOfValidNumbers : ℚ :=
  (countValidNumbers : ℚ) / (countTotalNumbers : ℚ)

theorem probability_odd_and_less_than_5000 :
  probabilityOfValidNumbers = 2 / 3 :=
by
  have h1 : countValidNumbers = 8 := by sorry
  have h2 : countTotalNumbers = 12 := by sorry
  unfold probabilityOfValidNumbers
  rw [h1, h2]
  norm_num

end probability_odd_and_less_than_5000_l98_98075


namespace intersection_of_line_and_ellipse_is_empty_l98_98335

theorem intersection_of_line_and_ellipse_is_empty :
  let A := { "line" }
  let B := { "ellipse" }
  A ∩ B = ∅ :=
by
  let A : set String := { "line" }
  let B : set String := { "ellipse" }
  show A ∩ B = ∅
  sorry

end intersection_of_line_and_ellipse_is_empty_l98_98335


namespace lower_limit_of_prime_range_l98_98915

theorem lower_limit_of_prime_range :
  let upper_limit := 85 / 6
  let upper_whole := nat.ceil upper_limit
  ∃ x : ℕ, x < 11 ∧ upper_whole = 15 ∧ (∀ p : ℕ, ∃ q : ℕ, prime p ∧ prime q ∧ p > x ∧ q > x ∧ p < 15 ∧ q < 15 ∧ p ≠ q) → x = 10 :=
by {
  let upper_limit := 85 / 6,
  let upper_whole := nat.ceil upper_limit,
  sorry
}

print lower_limit_of_prime_range

end lower_limit_of_prime_range_l98_98915


namespace function_unique_l98_98219

open Function

-- Define the domain and codomain
def NatPos : Type := {n : ℕ // n > 0}

-- Define the function f from positive integers to positive integers
noncomputable def f : NatPos → NatPos := sorry

-- Provide the main theorem
theorem function_unique (f : NatPos → NatPos) :
  (∀ (m n : NatPos), (m.val ^ 2 + (f n).val) ∣ ((m.val * (f m).val) + n.val)) →
  (∀ n : NatPos, f n = n) :=
by
  sorry

end function_unique_l98_98219


namespace find_six_numbers_l98_98635

noncomputable def multiply_numbers (a b c d e f : ℕ) : ℕ := a * b * c * d * e * f

theorem find_six_numbers :
  ∃ a b c d e f : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1 ∧ e ≠ 1 ∧ f ≠ 1 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
  multiply_numbers a b c d e f = 135135 ∧
  {a, b, c, d, e, f} = {3, 5, 7, 9, 11, 13} :=
by {
  use [3, 5, 7, 9, 11, 13],
  repeat {split},
  all_goals {intros; try {norm_num}; done},
  sorry
}

end find_six_numbers_l98_98635


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98253

-- Problem 1: Fair coin, probability of even heads
def fair_coin_even_heads_prob (n : ℕ) : Prop :=
  0.5 = 0.5

-- Problem 2: Biased coin, probability of even heads
def biased_coin_even_heads_prob (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : Prop :=
  let q := 1 - p in
  (0.5 * (1 + (1 - 2 * p)^n) = (1 + (1 - 2*p)^n) / 2)

-- Mock proof to ensure Lean accepts the definitions
theorem fair_coin_even_heads (n : ℕ) : fair_coin_even_heads_prob n :=
begin
  -- Proof intentionally omitted
  sorry
end

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : biased_coin_even_heads_prob n p hp :=
begin
  -- Proof intentionally omitted
  sorry
end

end fair_coin_even_heads_biased_coin_even_heads_l98_98253


namespace number_of_integer_pairs_l98_98443

-- Let ω be a nonreal root of z^4 = 1
def ω : ℂ := Complex.I -- Complex.I represents the imaginary unit i in Lean

-- Define the condition that |a * ω + b| = 1
def magnitude_condition (a b : ℤ) : Prop := 
  abs (a * ω + b) = 1

-- Define the statement that we want to prove
theorem number_of_integer_pairs :
  {p : ℤ × ℤ // magnitude_condition p.1 p.2}.to_finset.card = 4 :=
by
  sorry

end number_of_integer_pairs_l98_98443


namespace tile_probability_l98_98537

-- Define the tile sets
def box_C := {1, 2, 3, ..., 25}
def box_D := {15, 16, 17, ..., 39}

-- Conditions for the tiles in box C and box D
def tile_C_condition (n : Int) : Prop := n < 20
def tile_D_condition (n : Int) : Prop := (n % 2 = 1) ∨ n > 35

-- Probabilities calculation
def probability_C := 19 / 25
def probability_D := 15 / 25

-- The result calculation
def overall_probability := 19 / 25 * 3 / 5

-- Proof goal
theorem tile_probability : overall_probability = (57 / 125) := 
  by
  -- Proof steps would go here, using sorry for now
  sorry

end tile_probability_l98_98537


namespace average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l98_98396

theorem average_children_per_grade (G3_girls G3_boys G3_club : ℕ) 
                                  (G4_girls G4_boys G4_club : ℕ) 
                                  (G5_girls G5_boys G5_club : ℕ) 
                                  (H1 : G3_girls = 28) 
                                  (H2 : G3_boys = 35) 
                                  (H3 : G3_club = 12) 
                                  (H4 : G4_girls = 45) 
                                  (H5 : G4_boys = 42) 
                                  (H6 : G4_club = 15) 
                                  (H7 : G5_girls = 38) 
                                  (H8 : G5_boys = 51) 
                                  (H9 : G5_club = 10) :
   (63 + 87 + 89) / 3 = 79.67 :=
by sorry

theorem average_girls_per_grade (G3_girls G4_girls G5_girls : ℕ) 
                                (H1 : G3_girls = 28) 
                                (H2 : G4_girls = 45) 
                                (H3 : G5_girls = 38) :
   (28 + 45 + 38) / 3 = 37 :=
by sorry

theorem average_boys_per_grade (G3_boys G4_boys G5_boys : ℕ)
                               (H1 : G3_boys = 35) 
                               (H2 : G4_boys = 42) 
                               (H3 : G5_boys = 51) :
   (35 + 42 + 51) / 3 = 42.67 :=
by sorry

theorem average_club_members_per_grade (G3_club G4_club G5_club : ℕ) 
                                       (H1 : G3_club = 12)
                                       (H2 : G4_club = 15)
                                       (H3 : G5_club = 10) :
   (12 + 15 + 10) / 3 = 12.33 :=
by sorry

end average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l98_98396


namespace rectangular_to_spherical_l98_98208

theorem rectangular_to_spherical :
  ∃ (ρ θ φ : ℝ), 
    ρ = 3 * Real.sqrt 7 ∧ 
    θ = 7 * Real.pi / 6 ∧ 
    φ = Real.arccos(-2 / Real.sqrt 7) ∧ 
    (ρ > 0) ∧ 
    (0 ≤ θ ∧ θ < 2 * Real.pi) ∧ 
    (0 ≤ φ ∧ φ ≤ Real.pi) := 
sorry

end rectangular_to_spherical_l98_98208


namespace perfect_square_factors_count_l98_98751

def perfectSquares := [4, 9, 16, 25, 36, 49, 64, 81]

def countNumbersWithPerfectSquareFactors : Nat :=
  List.length (List.filter (fun n => perfectSquares.any (fun p => n % p = 0)) [1..100])

theorem perfect_square_factors_count :
  countNumbersWithPerfectSquareFactors = 41 := sorry

end perfect_square_factors_count_l98_98751


namespace count_numbers_with_perfect_square_factors_l98_98774

open Set

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ m * m ∣ n

theorem count_numbers_with_perfect_square_factors (s : Finset ℕ) (hs : s = Finset.range 101) :
  (Finset.filter has_perfect_square_factor_other_than_one s).card = 41 :=
by {
  sorry
}

end count_numbers_with_perfect_square_factors_l98_98774


namespace BoatsRUs_canoes_total_l98_98628

def geometric_series_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * ((r^n - 1) / (r - 1))

theorem BoatsRUs_canoes_total : 
  let a := 9 in
  let r := 3 in
  let n := 7 in
  geometric_series_sum a r n = 9827 :=
by
  sorry

end BoatsRUs_canoes_total_l98_98628


namespace hyesu_has_longest_tape_l98_98466

-- Definitions for the lengths of tapes
def minji_tape : ℝ := 0.74
def seungyeon_tape : ℝ := 13 / 20
def hyesu_tape : ℝ := 4 / 5

-- Theorem: Hyesu has the longest colored tape
theorem hyesu_has_longest_tape :
  hyesu_tape > minji_tape ∧ hyesu_tape > seungyeon_tape :=
by {
  -- Proof would go here
  sorry
}

end hyesu_has_longest_tape_l98_98466


namespace sequence_proof_l98_98743

-- Define the sequence (u_n) along with given conditions and recurrence relation
noncomputable def u : ℕ → ℝ
| 0       := 1 -- Note: The indexing starts from 0 for technical reasons. u₁ starts from n = 1 in terms of problem.
| (n + 1) := 4 - 2 * (u n + 1) / (2^(u n))

-- Theorem stating the required proof
theorem sequence_proof :
  (∀ n : ℕ, 1 ≤ u n ∧ u n ≤ 3) ∧ (∃ L : ℝ, L = 3 ∧ tendsto (λ n, u n) at_top (𝓝 L)) :=
sorry

end sequence_proof_l98_98743


namespace total_amount_equivalent_l98_98949

theorem total_amount_equivalent :
  ∃ P q r s t : ℝ,
    q = (2/5) * P ∧ 
    r = (1/7) * P ∧ 
    P = q + r + 35 ∧ 
    s = 2 * P ∧ 
    t = (1/2) * (q + r) ∧ 
    P + q + r + s + t = 291.03 := 
begin
  sorry
end

end total_amount_equivalent_l98_98949


namespace katya_total_notebooks_l98_98177

-- Definitions based on the conditions provided
def cost_per_notebook : ℕ := 4
def total_rubles : ℕ := 150
def stickers_for_free_notebook : ℕ := 5
def initial_stickers : ℕ := total_rubles / cost_per_notebook

-- Hypothesis stating the total notebooks Katya can obtain
theorem katya_total_notebooks : initial_stickers + (initial_stickers / stickers_for_free_notebook) + 
    ((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) / stickers_for_free_notebook) +
    (((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) % stickers_for_free_notebook + 1) / stickers_for_free_notebook) = 46 :=
by
  sorry

end katya_total_notebooks_l98_98177


namespace total_students_of_school_l98_98962

theorem total_students_of_school (x : ℕ) (h1 : 128 = (1 : ℝ/100) * 0.5 * x) : x = 256 :=
by
  sorry

end total_students_of_school_l98_98962


namespace max_popsicles_with_10_dollars_l98_98859

def price (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 3 then 2
  else if n = 5 then 3
  else if n = 7 then 4
  else 0

theorem max_popsicles_with_10_dollars : ∀ (a b c d : ℕ),
  a * price 1 + b * price 3 + c * price 5 + d * price 7 = 10 →
  a + 3 * b + 5 * c + 7 * d ≤ 17 :=
sorry

end max_popsicles_with_10_dollars_l98_98859


namespace subtract_from_sum_base3_l98_98497

def base3_add (a b : Num) := Num.pos (a + b)
def base3_sub (a b : Num) := Num.pos (a - b)

theorem subtract_from_sum_base3 : 
  base3_sub 
    (base3_add (base3_add (Num.ofNat 122) (Num.ofNat 2002)) (Num.ofNat 10101)) 
    (Num.ofNat 1012) 
    = Num.ofNat 11022 := 
by 
  -- Definitions for the given problem
  let a := Num.ofNat 122
  let b := Num.ofNat 2002
  let c := Num.ofNat 10101
  let d := Num.ofNat 1012
  let s := base3_add (base3_add a b) c
  -- Prove that the result equals 11022 (in base 3)
  sorry

end subtract_from_sum_base3_l98_98497


namespace pyramid_levels_l98_98852

theorem pyramid_levels (n : ℕ) (h : (n * (n + 1) * (2 * n + 1)) / 6 = 225) : n = 6 :=
by
  sorry

end pyramid_levels_l98_98852


namespace interest_rate_first_part_l98_98620

theorem interest_rate_first_part (P1 P2 : ℝ) (r1 r2 : ℝ) (t1 t2 : ℝ) (total_sum : ℝ) (I1 I2 : ℝ) :
  total_sum = 2743 ∧ P2 = 1688 ∧ r2 = 0.05 ∧ t2 = 3 ∧ t1 = 8 ∧
  I2 = P2 * r2 * t2 ∧ I1 = P1 * r1 * t1 ∧
  P1 = total_sum - P2 ∧ I1 = I2 →
  r1 = 0.03 :=
by
  intros h,
  sorry

end interest_rate_first_part_l98_98620


namespace geometric_seq_ad_eq_2_l98_98354

open Real

def geometric_sequence (a b c d : ℝ) : Prop :=
∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r 

def is_max_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
f x = y ∧ ∀ z : ℝ, z ≠ x → f x ≥ f z

theorem geometric_seq_ad_eq_2 (a b c d : ℝ) :
  geometric_sequence a b c d →
  is_max_point (λ x => 3 * x - x ^ 3) b c →
  a * d = 2 :=
by
  sorry

end geometric_seq_ad_eq_2_l98_98354


namespace geometric_seq_a7_l98_98303

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98303


namespace prove_inequality_l98_98516

noncomputable def proof_problem (x y z : ℝ)
  (h1 : x + y + z = 0)
  (h2 : |x| + |y| + |z| ≤ 1) : Prop :=
  x + y/3 + z/5 ≤ 2/5

theorem prove_inequality (x y z : ℝ) 
  (h1 : x + y + z = 0) 
  (h2 : |x| + |y| + |z| ≤ 1) : proof_problem x y z h1 h2 :=
sorry

end prove_inequality_l98_98516


namespace volume_eq_circumradius_eq_l98_98584

variable (a b c : ℝ)

noncomputable def volume_of_tetrahedron (a b c : ℝ) : ℝ :=
  (sqrt ((a^2 + c^2 - b^2) * (a^2 + b^2 - c^2) * (b^2 + c^2 - a^2))) / (6 * sqrt 2)

noncomputable def circumradius_of_tetrahedron (a b c : ℝ) : ℝ :=
  sqrt ((a^2 + b^2 + c^2) / 8)

theorem volume_eq : volume_of_tetrahedron a b c =
  (sqrt ((a^2 + c^2 - b^2) * (a^2 + b^2 - c^2) * (b^2 + c^2 - a^2))) / (6 * sqrt 2) :=
by sorry

theorem circumradius_eq : circumradius_of_tetrahedron a b c =
  sqrt ((a^2 + b^2 + c^2) / 8) :=
by sorry

end volume_eq_circumradius_eq_l98_98584


namespace num_pairs_12_students_l98_98680

theorem num_pairs_12_students : ∀ (n : ℕ), n = 12 → (nat.choose n 2 = 66) :=
by {
  intros,
  sorry
}

end num_pairs_12_students_l98_98680


namespace sum_of_first_12_even_numbers_is_156_l98_98107

theorem sum_of_first_12_even_numbers_is_156 :
  let evens := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24] in
  (evens.sum = 156) := by
  sorry

end sum_of_first_12_even_numbers_is_156_l98_98107


namespace mass_percentage_O_in_CaO_l98_98673

theorem mass_percentage_O_in_CaO :
  (16.00 / (40.08 + 16.00)) * 100 = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l98_98673


namespace expected_value_gt_median_l98_98970

noncomputable def density_function (f : ℝ → ℝ) (a b : ℝ) (X : ℝ → ℝ) :=
  (∀ x, x < a → f x = 0) ∧ 
  (∀ x, x ≥ b → f x = 0) ∧ 
  (∀ x, a ≤ x ∧ x < b → f x > 0) ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y < b → f x ≥ f y) ∧ 
  (∀ x, continuous_at f x)

theorem expected_value_gt_median 
  (a b : ℝ) 
  (X : ℝ → ℝ) 
  (f : ℝ → ℝ) 
  (density : density_function f a b X) 
  (median : ℝ) 
  (h_median : median = 0) : 
  ∃ E > median, E = ∫ x in a..b, x * f x :=
by
  sorry

end expected_value_gt_median_l98_98970


namespace interchange_digits_l98_98781

theorem interchange_digits (a b m : ℕ) (hab : a * b ≠ 0) (h: 10 * a + b = m * (a * b)):
  ∃ y, 10 * b + a = y * (a * b) ∧ y = 11 - m :=
by
  use 11 - m
  split
  sorry
  rfl

end interchange_digits_l98_98781


namespace limit_sol_l98_98666

open Real

def binom (i j : ℕ) : ℝ := (nat.factorial i) / ((nat.factorial j) * (nat.factorial (i - j)))

noncomputable def limit_expr (n : ℕ) : ℝ :=
  (binom (3 * n) n / binom (2 * n) n)^(1 / n : ℝ)

theorem limit_sol : tendsto limit_expr atTop (𝓝 (27 / 16)) :=
sorry

end limit_sol_l98_98666


namespace problem_solution_l98_98744

open Set

-- Define the sets A and B based on given problem conditions
def A : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3 ∧ x ∈ ℕ}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 9)}

-- Define the intersection of A and the complement of B in ℝ
def result : Set ℝ := A ∩ (Bᶜ)

-- The theorem to prove the intersection condition
theorem problem_solution : result = {1, 2} :=
by
  sorry  -- Proof will be skipped

#eval result  -- To evaluate and verify the result

end problem_solution_l98_98744


namespace area_of_triangle_QTS_l98_98536

theorem area_of_triangle_QTS :
  ∀ (PQ RS areaPQRS : ℝ) (ratio1 ratio2 : ℕ),
    PQ = 15 -> 
    RS = 36 ->
    areaPQRS = 387 -> 
    ratio1 = 3 -> 
    ratio2 = 5 ->
    (let h := 2 * areaPQRS / (PQ + RS) in
     let h1 := h * ratio1 / (ratio1 + ratio2) in
     let areaQTS := 0.5 * RS * h1 in
     areaQTS = 34.2) :=
by intros PQ RS areaPQRS ratio1 ratio2
   intros hPQ hRS hareaPQRS hratio1 hratio2
   let h := 2 * areaPQRS / (PQ + RS)
   let h1 := h * ratio1 / (ratio1 + ratio2)
   let areaQTS := 0.5 * RS * h1
   sorry

end area_of_triangle_QTS_l98_98536


namespace total_profit_2400_l98_98998
open Real

noncomputable def total_profit (A_profit : ℝ) (A_fraction B_fraction C_fraction : ℝ) : ℝ :=
  A_profit / A_fraction

theorem total_profit_2400 :
  let A_fraction := 1/3
  let B_fraction := 1/4
  let C_fraction := 1/5
  let A_profit := 800 : ℝ
  total_profit A_profit A_fraction B_fraction C_fraction = 2400 :=
sorry

end total_profit_2400_l98_98998


namespace find_a7_l98_98299

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98299


namespace log_expression_value_l98_98197

theorem log_expression_value :
    log 6 2 + 2 * log 6 (sqrt 3) + 10 ^ (log 10 2) = 5 := by
  -- Apply properties of logarithms and exponents to simplify
  sorry

end log_expression_value_l98_98197


namespace fraction_multiplication_division_l98_98202

-- We will define the fractions and state the equivalence
def fraction_1 : ℚ := 145 / 273
def fraction_2 : ℚ := 2 * (173 / 245) -- equivalent to 2 173/245
def fraction_3 : ℚ := 21 * (13 / 15) -- equivalent to 21 13/15

theorem fraction_multiplication_division :
  (frac1 * frac2 / frac3) = 7395 / 112504 := 
by sorry

end fraction_multiplication_division_l98_98202


namespace range_sum_l98_98936

theorem range_sum (f : ℝ → ℝ) (hf : ∀ x, f x = 3 / (3 + 9 * x ^ 2)) 
    (h_range : ∀ y, y ∈ set.range f ↔ (0 : ℝ) < y ∧ y ≤ 1) : 
  let a := 0 in let b := 1 in a + b = 1 :=
by 
  sorry

end range_sum_l98_98936


namespace harmonic_mean_1_3_6_l98_98643

-- Define the harmonic mean of a list of positive real numbers
def harmonic_mean (l : List ℝ) : ℝ :=
  (l.length : ℝ) / (l.map (λ x => 1 / x)).sum

-- Statement of the proof problem
theorem harmonic_mean_1_3_6 : harmonic_mean [1, 3, 6] = 2 := by
  sorry

end harmonic_mean_1_3_6_l98_98643


namespace passing_grade_fraction_l98_98796

variables (students : ℕ) -- total number of students in Mrs. Susna's class

-- Conditions
def fraction_A : ℚ := 1/4
def fraction_B : ℚ := 1/2
def fraction_C : ℚ := 1/8
def fraction_D : ℚ := 1/12
def fraction_F : ℚ := 1/24

-- Prove the fraction of students getting a passing grade (C or higher) is 7/8
theorem passing_grade_fraction : 
  fraction_A + fraction_B + fraction_C = 7/8 :=
by
  sorry

end passing_grade_fraction_l98_98796


namespace stock_original_value_undetermined_l98_98161

-- Define the conditions
def yield := 0.08  -- 8%
def market_value := 75  -- $75

-- Define the theorem stating that the percentage of stock's original value cannot be determined
theorem stock_original_value_undetermined 
  (yield : ℝ) (market_value : ℝ) : 
  (yield = 0.08) → (market_value = 75) → ∀ original_value : ℝ, "cannot be determined without the original value or annual dividend amount" := 
by
  intros h1 h2 original_value
  sorry

end stock_original_value_undetermined_l98_98161


namespace triangle_tan_sin_l98_98013

/-- In triangle ABC, we are given AB = 25 and BC = 20. We aim to prove:
The largest possible value of tan A is 4/3 and the largest possible value of sin A is 4/5. -/
theorem triangle_tan_sin (ABC : Triangle)
  (hAB : ABC.length AB = 25)
  (hBC : ABC.length BC = 20) :
  (max_tan_angle ABC A = 4/3) ∧ (max_sin_angle ABC A = 4/5) := 
by
  sorry

end triangle_tan_sin_l98_98013


namespace part1_part2_part3_l98_98966

-- Functional relationship between y (daily sales volume) and x (selling price)
theorem part1 (x : ℝ) (h1 : 60 ≤ x ∧ x ≤ 80) : (let y := 20 + 2 * (80 - x) in y = 180 - 2 * x) :=
by sorry

-- Price for a profit of 432 yuan per day
theorem part2 (x : ℝ) (h2 : (x - 60) * (-2 * x + 180) = 432) : x = 72 ∨ x = 78 :=
by sorry

-- Price that maximizes daily sales profit
theorem part3 : (x : ℝ) → (let w := (x - 60) * (-2 * x + 180) in ∀ y, y = (if x = 75 then 450 else w) → x = 75) :=
by sorry

end part1_part2_part3_l98_98966


namespace maximize_total_profit_a_eq_1_div_3_range_of_a_l98_98424

def profit_A (x : ℝ) : ℝ := (1/5) * real.sqrt x
def profit_B (a : ℝ) (x : ℝ) : ℝ := (1/5) * a * x
def total_profit (a : ℝ) (x : ℝ) : ℝ := profit_A x + profit_B a (5 - x)

-- Part 1: Prove that the maximum total profit is achieved when x = 9/4 given a = 1/3
theorem maximize_total_profit_a_eq_1_div_3 : 
  ∀ (x : ℝ), 
    (1 ≤ x ∧ x ≤ 4) → 
    total_profit (1/3) x ≤ total_profit (1/3) (9/4) := 
sorry

-- Part 2: Prove the range of values for a such that the total profit equals (-4a + 3)/5
theorem range_of_a (y : ℝ) : 
  (y = (-4 * a + 3) / 5) → 
  (1 ≤ t ∧ t ≤ 2) → 
  (a = 1 / (t + 3)) → 
  (1/5 ≤ a ∧ a ≤ 1/4) := 
sorry

end maximize_total_profit_a_eq_1_div_3_range_of_a_l98_98424


namespace find_a7_l98_98285

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98285


namespace ball_hits_ground_at_10_over_7_l98_98504

def ball_hits_ground (t : ℚ) : Prop :=
  -4.9 * t^2 + 3.5 * t + 5 = 0

theorem ball_hits_ground_at_10_over_7 : ball_hits_ground (10 / 7) :=
by
  sorry

end ball_hits_ground_at_10_over_7_l98_98504


namespace hyuksu_total_meat_l98_98776

/-- 
Given that Hyuksu ate 2.6 kilograms (kg) of meat yesterday and 5.98 kilograms (kg) of meat today,
prove that the total kilograms (kg) of meat he ate in two days is 8.58 kg.
-/
theorem hyuksu_total_meat (yesterday today : ℝ) (hy1 : yesterday = 2.6) (hy2 : today = 5.98) :
  yesterday + today = 8.58 := 
by
  rw [hy1, hy2]
  norm_num

end hyuksu_total_meat_l98_98776


namespace hexagon_perimeter_is_12_l98_98814

-- Definitions based on conditions
structure Hexagon :=
  (sides : ℝ)
  (angles : Fin 6 → ℝ)
  (equilateral : ∀ i, sides = sides)
  (angle_properties : ∀ i, angles i ∈ {45, 135})

-- Given the conditions
def hexagon : Hexagon := {
  sides := 2, -- computation from solution
  angles := λ i, if i % 2 = 0 then 45 else 135,
  equilateral := λ i, rfl,
  angle_properties := λ i, by simp [Finset.mem_insert, Finset.mem_singleton, Nat.mod_eq_zero_or_pos]
}

-- The property to be proved
def perimeter_of_hexagon (h: Hexagon) : ℝ := 6 * h.sides

theorem hexagon_perimeter_is_12 : perimeter_of_hexagon(hexagon) = 12 :=
by sorry

end hexagon_perimeter_is_12_l98_98814


namespace isosceles_triangle_perimeter_l98_98634

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₁ : a = 12) (h₂ : b = 12) (h₃ : c = 17) : a + b + c = 41 :=
by
  rw [h₁, h₂, h₃]
  norm_num

end isosceles_triangle_perimeter_l98_98634


namespace total_alligators_seen_l98_98483

theorem total_alligators_seen (samara : ℕ) (friend_avg : ℕ) (friends : ℕ) 
  (h_samara : samara = 20) (h_friend_avg : friend_avg = 10) (h_friends : friends = 3) : 
  samara + friends * friend_avg = 50 := 
by
  -- Initial conditions
  rw [h_samara, h_friend_avg, h_friends]
  -- Evaluate the expression
  simp
  -- Prove the final result
  sorry

end total_alligators_seen_l98_98483


namespace proposition_4_correct_l98_98693

variables {m n : Line} {α β : Plane}
-- Assumptions 
def non_coincident_lines (m n : Line) : Prop := m ≠ n
def non_coincident_planes (α β : Plane) : Prop := α ≠ β
def line_in_plane (m : Line) (α : Plane) : Prop := ∀ p, p ∈ m → p ∈ α
def lines_parallel (m n : Line) : Prop := ∀ p₁ p₂, p₁ ∈ m → p₂ ∈ n → ∃ q, q ∈ LineThrough p₁ p₂
def lines_perpendicular (m n : Line) : Prop := m ⊥ n
def planes_parallel (α β : Plane) : Prop := α ∥ β
def plane_intersection (α β : Plane) (n : Line) : Prop := α ∩ β = n

-- Question 4 only, since it is asking to prove this proposition is correct:
theorem proposition_4_correct 
  (hmn : non_coincident_lines m n)
  (hαβ : non_coincident_planes α β)
  (h1 : lines_perpendicular m α)
  (h2 : lines_perpendicular m β) 
  : planes_parallel α β := 
sorry

end proposition_4_correct_l98_98693


namespace arrange_descending_l98_98450

noncomputable def a : ℝ := 0.9 ^ 2
noncomputable def b : ℝ := 2 ^ 0.9
noncomputable def c : ℝ := Real.log 0.9 / Real.log 2

theorem arrange_descending : c < a ∧ a < b := by
  sorry

end arrange_descending_l98_98450


namespace find_a7_l98_98291

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98291


namespace expression_equiv_target_l98_98793

noncomputable def expression_form : ℚ :=
  let sqrt6 := Real.sqrt 6
  let sqrt8 := Real.sqrt 8
  in sqrt6 + 1 / sqrt6 + sqrt8 + 1 / sqrt8

noncomputable def target_form (p q r : ℕ) : ℚ :=
  (p * Real.sqrt 6 + q * Real.sqrt 8) / r

theorem expression_equiv_target :
  ∃ (p q r : ℕ), target_form p q r = expression_form ∧ p + q + r = 19 :=
by
  let p := 7
  let q := 9
  let r := 3
  sorry

end expression_equiv_target_l98_98793


namespace orthocentric_common_perpendiculars_intersect_at_point_l98_98181

-- Define an orthocentric tetrahedron
structure OrthocentricTetrahedron (A B C D : Point) :=
(altitudesIntersect : ∃ H : Point, ∀ X ∈ {A, B, C, D}, is_altitude X H)

-- Define a function that checks if lines are perpendicular
def are_perpendicular (l1 l2 : Line) : Prop := ...

-- Define a function for common perpendiculars of skew lines
def common_perpendicular (l1 l2 : Line) : Line := ...

-- The main theorem stating the common perpendicular intersections
theorem orthocentric_common_perpendiculars_intersect_at_point
  {A B C D : Point} (T : OrthocentricTetrahedron A B C D) :
  ∃ H : Point, ∀ (x y : Point) (hx : x ∈ {A, B, C, D}) (hy : y ∈ {A, B, C, D}) (hxy : x ≠ y),
    let l1 := line_through x y in
    let l2 := ...
    common_perpendicular l1 l2 = l2 :=
sorry

end orthocentric_common_perpendiculars_intersect_at_point_l98_98181


namespace inequality_holds_for_m_n_l98_98735

-- Problem definition in Lean:
variable (m n : ℕ)
variable {x : ℝ}

-- Conditions
def f (x : ℝ) (a : ℝ) : ℝ := a * log x + 1/2 * x^2 - (1 + a) * x

-- Correct answer (inequality)
theorem inequality_holds_for_m_n : 
  0 < m → 0 < n → 
  (∑ k in Finset.range m.succ, (∑ j in Finset.range n, 1 / log (k.succ + j.succ))) > (1 / (m * (m + n))) :=
by
  sorry

end inequality_holds_for_m_n_l98_98735


namespace fishing_problem_l98_98582

theorem fishing_problem (a b c d : ℕ)
  (h1 : a + b + c + d = 11)
  (h2 : 1 ≤ a) 
  (h3 : 1 ≤ b) 
  (h4 : 1 ≤ c) 
  (h5 : 1 ≤ d) : 
  a < 3 ∨ b < 3 ∨ c < 3 ∨ d < 3 :=
by
  -- This is a placeholder for the proof
  sorry

end fishing_problem_l98_98582


namespace circumscribed_triangle_area_relation_l98_98598

theorem circumscribed_triangle_area_relation
    (a b c: ℝ) (h₀: a = 8) (h₁: b = 15) (h₂: c = 17)
    (triangle_area: ℝ) (circle_area: ℝ) (X Y Z: ℝ)
    (hZ: Z > X) (hXY: X < Y)
    (triangle_area_calc: triangle_area = 60)
    (circle_area_calc: circle_area = π * (c / 2)^2) :
    X + Y = Z := by
  sorry

end circumscribed_triangle_area_relation_l98_98598


namespace infinitely_many_odd_terms_l98_98455

def seq (n k : ℕ) : ℕ :=
  ⌊ (n^k : ℚ) / k ⌋

theorem infinitely_many_odd_terms (n : ℕ) (hn : n > 1) : ∃ᶠ k in at_top, Odd (seq n k) :=
sorry

end infinitely_many_odd_terms_l98_98455


namespace keaton_yearly_earnings_l98_98433

theorem keaton_yearly_earnings:
  (let 
    oranges_harvest_times_per_year := 12 / 2,
    oranges_earning_per_year := oranges_harvest_times_per_year * 50,
    apples_harvest_times_per_year := 12 / 3,
    apples_earning_per_year := apples_harvest_times_per_year * 30,
    total_earning_per_year := oranges_earning_per_year + apples_earning_per_year
  in total_earning_per_year = 420) :=
begin
  sorry
end

end keaton_yearly_earnings_l98_98433


namespace find_w_l98_98390

theorem find_w (w : ℕ) (h1 : w > 0) (h2 : ∑ d in (digits (10^w - 74)), d = 440)
  (h3 : (∀ i, i < (digits (10^w - 74)).length - 1 → prime ((digits (10^w - 74)).nth i + (digits (10^w - 74)).nth (i+1)))) : w = 50 :=
sorry

end find_w_l98_98390


namespace b_work_alone_days_l98_98140

theorem b_work_alone_days (a_can_complete : 12 > 0) (a_work_3_days : true) : ∃ x, (0 < x) ∧ (6 = x) :=
by
  -- Conditions
  have work_A_per_day : ℚ := 1 / 12
  have work_A_3_days : ℚ := 3 * work_A_per_day
  have remaining_work : ℚ := 1 - work_A_3_days
  have work_remaining : remaining_work = 3 / 4
  have work_A_more_3_days : ℚ := 3 * work_A_per_day

  -- Work together for 3 days
  intros
  use 6
  split
  . norm_num
  . norm_num
  have work_B_per_day := 1 / 6

  -- Proof placeholder
  sorry

end b_work_alone_days_l98_98140


namespace smallest_m_l98_98842

def a : ℕ → ℤ
| 0     := 3
| (n+1) := (n+1+1) * a n - n

theorem smallest_m (m : ℕ) (h : m ≥ 2005) : 
  (∃ k : ℕ, m = 2005 + k ∧ a (m+1) - 1 ∣ a m ^ 2 - 1) :=
by
  intros
  sorry

end smallest_m_l98_98842


namespace count_numbers_with_square_factors_l98_98770

theorem count_numbers_with_square_factors :
  let squares := [4, 9, 16, 25, 36, 49, 64]
  let multiples (n : ℕ) := ∀ k ∈ squares, n % k = 0
  let count_multiples (n : ℕ) := (1..100).count multiples
  count_multiples squares = 48 :=
  sorry

end count_numbers_with_square_factors_l98_98770


namespace james_twitch_income_l98_98024

theorem james_twitch_income :
  let tier1_base := 120
  let tier2_base := 50
  let tier3_base := 30
  let tier1_gifted := 10
  let tier2_gifted := 25
  let tier3_gifted := 15
  let tier1_new := tier1_base + tier1_gifted
  let tier2_new := tier2_base + tier2_gifted
  let tier3_new := tier3_base + tier3_gifted
  let tier1_income := tier1_new * 4.99
  let tier2_income := tier2_new * 9.99
  let tier3_income := tier3_new * 24.99
  let total_income := tier1_income + tier2_income + tier3_income
  total_income = 2522.50 :=
by
  sorry

end james_twitch_income_l98_98024


namespace solve_equation_l98_98659

theorem solve_equation : ∃ x : ℝ, (1 / (x + 3) = 3 / (x - 1)) ∧ x = -5 :=
by
  use -5
  split
  sorry
  refl

end solve_equation_l98_98659


namespace fair_coin_even_heads_unfair_coin_even_heads_l98_98247

-- Define the probability function for an even number of heads for a fair coin
theorem fair_coin_even_heads (n : ℕ) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * 0.5^k * 0.5^(n-k) else 0) = 0.5 :=
sorry

-- Define the probability function for an even number of heads for an unfair coin
theorem unfair_coin_even_heads (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * p^k * (1-p)^(n-k) else 0) = 
    (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_unfair_coin_even_heads_l98_98247


namespace inequality_condition_l98_98668

theorem inequality_condition (k : ℝ) (h : k ≥ 4) 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + (k * c) / (a + b) ≥ 2) :=
sorry

end inequality_condition_l98_98668


namespace sum_of_divisors_143_l98_98199

theorem sum_of_divisors_143 : (∑ d in (Finset.divisors 143), d) = 168 := 
by {
  have h1 : Nat.factors 143 = [11, 13],
  {
    -- Proof for the prime factorization of 143
    sorry
  },
  have h2 : ∀ p ∈ [11, 13], Nat.Prime p,
  {
    -- Proof that both 11 and 13 are prime numbers
    sorry
  },
  -- Use the sum of divisors formula for the product of prime factors
  sorry
}

end sum_of_divisors_143_l98_98199


namespace five_digit_square_numbers_condition_l98_98213

theorem five_digit_square_numbers_condition:
  ∃ (N : ℕ), N^2 = 11664 ∨ N^2 = 12996 ∨ N^2 = 34596 ∨ N^2 = 53361 ∧ 
  let A := N^2 / 10000,
      B := (N^2 / 1000) % 10,
      C := (N^2 / 100) % 10,
      D := (N^2 / 10) % 10,
      E := N^2 % 10 in
  A + B + C + E = 2 * D :=
by
  sorry

end five_digit_square_numbers_condition_l98_98213


namespace train_crosses_platform_in_25_002_seconds_l98_98995

noncomputable theory

def train_crossing_time (train_length : ℕ) (platform_length : ℕ) (train_speed_kmph : ℕ) : ℚ :=
  let train_length_m := train_length -- length in meters
  let platform_length_m := platform_length -- length in meters
  let total_distance := train_length_m + platform_length_m -- total distance in meters
  let train_speed_ms := train_speed_kmph * (10 : ℚ) / 36 -- converting kmph to m/s
  total_distance / train_speed_ms

theorem train_crosses_platform_in_25_002_seconds :
  train_crossing_time 120 380 72 = 25.002 :=
by
  sorry

end train_crosses_platform_in_25_002_seconds_l98_98995


namespace sin_angle_ratio_l98_98416

theorem sin_angle_ratio (A B C D : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D]
  (triangle_ABC: is_triangle A B C)
  (angle_B_eq_45 : ∠ B = 45)
  (angle_C_eq_60 : ∠ C = 60)
  (D_divides_BC : divides D B C 2 3) :
  (sin (∠ BAD) / sin (∠ CAD)) = (4 * sqrt 6 / 9) :=
by 
  sorry

end sin_angle_ratio_l98_98416


namespace geometric_sequence_residue_system_l98_98040

noncomputable def exists_geometric_sequence (p r k : ℕ) (a : ℕ → ℕ) : Prop :=
  (prime p) ∧ (odd_prime r) ∧ (p = 2 * r * k + 1) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ r → a j = a i * ratio r (j-i)) ∧
  (p = finset.sum (finset.range r) a) ∧
  ∃ b : ℕ → ℕ,
    (∀ i j, i < j ∧ i + j < 2 * k → b j = b i * ratio k (j-i)) ∧
    (∃ A B, 
      (A = finset.image a (finset.range r)) ∧ 
      (B = finset.image b (finset.range (2 * k))) ∧ 
      (is_complete_residue_system A B p))

theorem geometric_sequence_residue_system
  {p r k : ℕ} {a : ℕ → ℕ} 
  (hp : prime p) 
  (hr : odd_prime r) 
  (hpk : p = 2 * r * k + 1) 
  (ha : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ r → a j = a i * ratio r (j-i))
  (hpsum : p = finset.sum (finset.range r) a) :
  exists_geometric_sequence p r k a :=
sorry

end geometric_sequence_residue_system_l98_98040


namespace sum_projections_l98_98016

universe u

variables {α : Type u} [Nonempty α] [MetricSpace α]

structure Triangle (α : Type u) [MetricSpace α] :=
(A B C : α)
(AB AC BC : ℝ)
(AB_pos : 0 < AB)
(AC_pos : 0 < AC)
(BC_pos : 0 < BC)
(equal_ab : dist A B = AB)
(equal_ac : dist A C = AC)
(equal_bc : dist B C = BC)

def centroid (A B C : α) : α := sorry -- centroid calculation goes here

def projection (P Q R : α) (G : α) : ℝ := sorry -- projection calculation goes here

theorem sum_projections {A B C: α} (t : Triangle α)
  (G : α) (P : α := projection t.B t.C G)
  (Q : α := projection t.A t.C G)
  (R : α := projection t.A t.B G) :
  t.AB = 4 → t.AC = 6 → t.BC = 5 → (projection t.B t.C G) + (projection t.A t.C G) + (projection t.A t.B G) = (5 * real.sqrt 7) / 7 :=
by
  sorry

end sum_projections_l98_98016


namespace principal_amount_l98_98984

theorem principal_amount (R T SI P : ℝ) (hR : R = 17.5) (hT : T = 2.5) (hSI : SI = 3150) :
  SI = P * R * T / 100 → P = 7200 :=
by
  intros h
  rw [hR, hT, hSI] at h
  rw [←mul_assoc, ←mul_comm T, mul_assoc] at h
  norm_num at h
  exact h

end principal_amount_l98_98984


namespace units_digit_of_3_pow_1987_l98_98557

theorem units_digit_of_3_pow_1987 : 3 ^ 1987 % 10 = 7 := by
  sorry

end units_digit_of_3_pow_1987_l98_98557


namespace convex_pentagon_no_collinear_medians_non_convex_pentagon_collinear_medians_l98_98945

variable {A B C D E : Type} [ConvexPentagon A B C D E] [NonConvexPentagon A B C D E]
variables {T1 T2 T3 : Triangle} [MedianIntersect T1 T2 T3]

-- (a) Convex Pentagon
theorem convex_pentagon_no_collinear_medians :
  ¬Collinear
  (Centroid (median T1))
  (Centroid (median T2))
  (Centroid (median T3)) := 
sorry

-- (b) Non-Convex Pentagon
theorem non_convex_pentagon_collinear_medians :
  Collinear
  (Centroid (median T1))
  (Centroid (median T2))
  (Centroid (median T3)) := 
sorry

end convex_pentagon_no_collinear_medians_non_convex_pentagon_collinear_medians_l98_98945


namespace chord_length_l98_98742

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

def parabola_eqn (p : Point) : Prop :=
  (p.y)^2 = 4 * p.x

def on_directrix (p : Point) (d : ℝ) : Prop :=
  p.x + d = 0

def circle_eqn (center radius : Point) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius.x^2 + radius.y^2

theorem chord_length (M : Point) (A : Point) (l : ℝ):
  parabola_eqn M →
  on_directrix ⟨M.x + 1, M.y⟩ l →
  circle_eqn M ⟨M.x + 1, M.y⟩ A →
  A.x = 3 ∧ A.y = 0 →
  M.x = 2 →
  ∃ BC, BC = 2 * Real.sqrt 5 :=
by
  intro h_parab h_direct h_circle h_A h_M
  use 2 * Real.sqrt 5
  sorry

end chord_length_l98_98742


namespace possible_selection_l98_98989

-- Define the structure of a round-robin tournament
structure Tournament (α : Type) :=
(players : Finset α)
(defeats : α → α → Prop)
(circular_defeat : ∀ (p : α) (s : List α), s.nodup → set.toFinset s = players → ∃ q ∈ s.append [p], defeats q p ∧ defeats q (next s p))
  where next (s : List α) (p : α) : α := s.get? ((s.indexOf p + 1) % s.length) |>.get p

-- The main theorem we need to prove
theorem possible_selection {α : Type} (t : Tournament α) :
  ∃ S : Finset α, S.nonempty ∧ S ≠ t.players ∧ ∀ (p ∈ S) (q ∈ t.players \ S), t.defeats p q :=
sorry

end possible_selection_l98_98989


namespace gardening_project_total_cost_l98_98629

noncomputable def cost_gardening_project : ℕ := 
  let number_rose_bushes := 20
  let cost_per_rose_bush := 150
  let cost_fertilizer_per_bush := 25
  let gardener_work_hours := [6, 5, 4, 7]
  let gardener_hourly_rate := 30
  let soil_amount := 100
  let cost_per_cubic_foot := 5

  let cost_roses := number_rose_bushes * cost_per_rose_bush
  let cost_fertilizer := number_rose_bushes * cost_fertilizer_per_bush
  let total_work_hours := List.sum gardener_work_hours
  let cost_labor := total_work_hours * gardener_hourly_rate
  let cost_soil := soil_amount * cost_per_cubic_foot

  cost_roses + cost_fertilizer + cost_labor + cost_soil

theorem gardening_project_total_cost : cost_gardening_project = 4660 := by
  sorry

end gardening_project_total_cost_l98_98629


namespace limit_S_div_l_ln_l98_98688

noncomputable def C (a x : ℝ) : ℝ := (a / 2) * (Real.exp (x / a) + Real.exp (-x / a))

noncomputable def l (a t : ℝ) : ℝ :=
  let b := Real.acosh (t / a)
  2 * a * Real.sinh (b / a)

noncomputable def S (a t : ℝ) : ℝ :=
  let b := Real.acosh (t / a)
  2 * (t * b - a^2 * Real.sinh (b / a))

theorem limit_S_div_l_ln (a : ℝ) (h_pos : 0 < a) :
  tendsto (fun t => S a t / (l a t * Real.log t)) at_top (𝓝 a) :=
by sorry

end limit_S_div_l_ln_l98_98688


namespace find_a7_l98_98275

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98275


namespace minimizing_incenter_l98_98020

theorem minimizing_incenter (A B C I D E F : Point)
  (hI_in_triangle : I ∈ triangle A B C)
  (h_perpendicular_D : foot_of_perpendicular I B C = D)
  (h_perpendicular_E : foot_of_perpendicular I C A = E)
  (h_perpendicular_F : foot_of_perpendicular I A B = F) :
  (∀ J, J ∈ interior_triangle A B C → (frac BC (dist J D) + frac CA (dist J E) + frac AB (dist J F)) ≥ (frac BC (dist I D) + frac CA (dist I E) + frac AB (dist I F))) ↔ is_incenter I A B C :=
sorry

end minimizing_incenter_l98_98020


namespace total_snakes_among_pet_owners_l98_98525

theorem total_snakes_among_pet_owners :
  let owns_only_snakes := 15
  let owns_cats_and_snakes := 7
  let owns_dogs_and_snakes := 10
  let owns_birds_and_snakes := 2
  let owns_snakes_and_hamsters := 3
  let owns_cats_dogs_and_snakes := 4
  let owns_cats_snakes_and_hamsters := 2
  let owns_all_categories := 1
  owns_only_snakes + owns_cats_and_snakes + owns_dogs_and_snakes + owns_birds_and_snakes + owns_snakes_and_hamsters + owns_cats_dogs_and_snakes + owns_cats_snakes_and_hamsters + owns_all_categories = 44 :=
by
  sorry

end total_snakes_among_pet_owners_l98_98525


namespace kite_diagonal_length_l98_98807

theorem kite_diagonal_length 
  (AC BD : ℝ)
  (h1 : AC = 22)
  (h2 : AC = 2 * BD)
  (h3 : ∃ O, ∠ (A O B) = 90 ∧ ∠ (C O D) = 90)
  : BD = 11 := 
by {
  -- Proof goes here
  sorry
}

end kite_diagonal_length_l98_98807


namespace num_tuples_abc_l98_98583

theorem num_tuples_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 2019 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  ∃ n, n = 574 := sorry

end num_tuples_abc_l98_98583


namespace calc_expression_l98_98205

theorem calc_expression :
  (- (2 / 5) : ℝ)^0 - (0.064 : ℝ)^(1/3) + 3^(Real.log (2 / 5) / Real.log 3) + Real.log 2 / Real.log 10 - Real.log (1 / 5) / Real.log 10 = 2 := 
by
  sorry

end calc_expression_l98_98205


namespace derivative_at_pi_l98_98362

-- Define the function f
def f (x : ℝ) : ℝ := x * Real.sin x

-- Define the statement to be proved
theorem derivative_at_pi : deriv f π = -π :=
by
  -- Proof is not required per instructions
  sorry

end derivative_at_pi_l98_98362


namespace gnollish_valid_sentences_l98_98873

def valid_sentences_count : ℕ :=
  let words := ["splargh", "glumph", "amr", "krack"]
  let total_words := 4
  let total_sentences := total_words ^ 3
  let invalid_splargh_glumph := 2 * total_words
  let invalid_amr_krack := 2 * total_words
  let total_invalid := invalid_splargh_glumph + invalid_amr_krack
  total_sentences - total_invalid

theorem gnollish_valid_sentences : valid_sentences_count = 48 :=
by
  sorry

end gnollish_valid_sentences_l98_98873


namespace transport_cost_bound_min_transport_cost_speed_max_value_of_a_l98_98543

noncomputable def transport_cost (v a b : ℝ) : ℝ :=
  (b * v^2 + a) * (100 / v)

theorem transport_cost_bound
  (v : ℝ) (b : ℝ := 1/50) (a : ℝ := 300) :
  v ∈ Icc 60 120 → transport_cost v a b ≤ 500 := 
by
sorry

theorem min_transport_cost_speed 
  (a b : ℝ) :
  3600 * b ≤ a ∧ a ≤ 14400 * b → ∀ v ∈ Icc 60 120, transport_cost v a b = 100 * b * (v + (a / (b * v))) :=
by
sorry

theorem max_value_of_a
  (a b : ℝ) (h : 3600 * b ≤ a ∧ a ≤ 14400 * b) : 
  a ≤ 360 := 
by
sorry

end transport_cost_bound_min_transport_cost_speed_max_value_of_a_l98_98543


namespace jerry_total_logs_l98_98825

theorem jerry_total_logs :
  let pine_logs := 8 * 80
  let maple_logs := 3 * 60
  let walnut_logs := 4 * 100
  pine_logs + maple_logs + walnut_logs = 1220 :=
by
  let pine_logs := 8 * 80
  let maple_logs := 3 * 60
  let walnut_logs := 4 * 100
  have h_pine : pine_logs = 640 := by rfl
  have h_maple : maple_logs = 180 := by rfl
  have h_walnut : walnut_logs = 400 := by rfl
  calc
    640 + 180 + 400 = 1220 : by rfl

end jerry_total_logs_l98_98825


namespace solve_for_k_l98_98092

theorem solve_for_k (k x : ℝ) (h₁ : 4 * k - 3 * x = 2) (h₂ : x = -1) : 
  k = -1 / 4 := 
by sorry

end solve_for_k_l98_98092


namespace smallest_value_of_N_l98_98803

theorem smallest_value_of_N :
  ∃ N : ℕ, N ≥ 1 ∧
  (∃ (c1 c2 c3 c4 c5 c6 : ℕ),
    c1 = 6 * c2 - 5 ∧ 
    N + c2 = 6 * c1 - 5 ∧
    2 * N + c3 = 6 * c4 - 2 ∧
    3 * N + c4 = 6 * c5 + 1 ∧
    4 * N + c5 = 6 * c6 - 3 ∧
    5 * N + c6 = 6 * c3 - 5) ∧ 
  (∀ M, (M ≥ 1 ∧ M < N) → ¬ (∃ (d1 d2 d3 d4 d5 d6 : ℕ),
      d1 = 6 * d2 - 5 ∧ 
      M + d2 = 6 * d1 - 5 ∧
      2 * M + d3 = 6 * d4 - 2 ∧
      3 * M + d4 = 6 * d5 + 1 ∧
      4 * M + d5 = 6 * d6 - 3 ∧
      5 * M + d6 = 6 * d3 - 5)) :=
begin
  use 110,
  split,
  { norm_num },
  existsi [29, 4, 27, 14, 18, 32],
  simp,
  norm_num,
  -- rest of proof (if needed) will go here
  sorry
end

end smallest_value_of_N_l98_98803


namespace no_solution_values_l98_98787

theorem no_solution_values (m : ℝ) :
  (∀ x : ℝ, x ≠ 5 → x ≠ -5 → (1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25))) ↔
  m = -1 ∨ m = 5 ∨ m = -5 / 11 :=
by
  sorry

end no_solution_values_l98_98787


namespace Antonium_value_on_May_31_l98_98077

def C : ℕ → ℚ
| 1       := 1
| (n+1)   := n + 1

def A : ℕ → ℚ
| 1       := 1
| (n+1)   := (C n + A n) / (C n * A n)

theorem Antonium_value_on_May_31 :
  A 92 = 92 / 91 :=
  sorry

end Antonium_value_on_May_31_l98_98077


namespace fourth_line_divides_area_possible_angle_values_l98_98921

-- Define necessary elements of the problem
variables {A B C D O : Point}
variable (quadrilateral_inscribed_circle : Quadrilateral A B C D)
variable (inscribed_circle_center : Circle O)
variable (line_AO : LineSegment O A)
variable (line_BO : LineSegment O B)
variable (line_CO : LineSegment O C)
variable (line_DO : LineSegment O D)
variables {angle_A angle_B angle_C angle_D : Real}

-- Conditions
axiom line_divides_area_AO : divides_area_equally quadrilateral_inscribed_circle line_AO
axiom line_divides_area_BO : divides_area_equally quadrilateral_inscribed_circle line_BO
axiom line_divides_area_CO : divides_area_equally quadrilateral_inscribed_circle line_CO

-- Part (a) Statement
theorem fourth_line_divides_area :
  divides_area_equally quadrilateral_inscribed_circle line_DO := sorry

-- Additional condition for part (b)
axiom angle_A_is_72 : angle_A = 72

-- Part (b) Statement
theorem possible_angle_values :
  (angle_B = 108 ∧ angle_C = 72 ∧ angle_D = 108) ∨ (angle_B = 72 ∧ angle_C = 72 ∧ angle_D = 144) := sorry

end fourth_line_divides_area_possible_angle_values_l98_98921


namespace range_of_m_l98_98708

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l98_98708


namespace clock_strike_time_l98_98914

theorem clock_strike_time (t : ℕ) (n m : ℕ) (I : ℕ) : 
  t = 12 ∧ n = 3 ∧ m = 6 ∧ 2 * I = t → (m - 1) * I = 30 := by 
  sorry

end clock_strike_time_l98_98914


namespace problem_solution_l98_98321

theorem problem_solution (a b : ℝ) (h : a^2 + b^2 = 4) :
  real.cbrt (a * (b - 4)) + real.sqrt (a * b - 3 * a + 2 * b - 6) = 2 :=
sorry

end problem_solution_l98_98321


namespace T_n_bounds_l98_98722

noncomputable def S_n (n : ℕ) : ℝ := sorry

noncomputable def a_n (n : ℕ) : ℝ := if n = 1 then 2 else 4 * a_n (n - 1)

noncomputable def b_n (n : ℕ) : ℝ := real.log2 (a_n n)

noncomputable def c_n (n : ℕ) : ℝ :=
  4 / ((b_n n + 1) * (b_n (n + 1) + 3))

noncomputable def T_n (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, c_n (k + 1))

theorem T_n_bounds (n : ℕ) : 1/3 ≤ T_n (n + 1) ∧ T_n (n + 1) < 3/4 :=
by
  sorry

end T_n_bounds_l98_98722


namespace people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l98_98982

def f (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
  else if 9 ≤ n ∧ n ≤ 32 then 360 * 3 ^ ((n - 8) / 12) + 3000
  else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
  else 0 -- default case for unsupported values

def g (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 18 then 0
  else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
  else if 33 ≤ n ∧ n ≤ 45 then 8800
  else 0 -- default case for unsupported values

theorem people_entering_2pm_to_3pm :
  f 21 + f 22 + f 23 + f 24 = 17460 := sorry

theorem people_leaving_2pm_to_3pm :
  g 21 + g 22 + g 23 + g 24 = 9000 := sorry

theorem peak_visitors_time :
  ∀ n, 1 ≤ n ∧ n ≤ 45 → 
    (n = 28 ↔ ∀ m, 1 ≤ m ∧ m ≤ 45 → f m - g m ≤ f 28 - g 28) := sorry

end people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l98_98982


namespace largest_area_is_82225_l98_98924

noncomputable def largest_triangle_area (AB BC AC : ℝ) (hAB : AB = 12) (h_ratio : BC / AC = 39 / 40) : ℝ :=
  if 12 + 39 * (AC / 40) > AC / 40 ∧ 40 * (AC / 40) + 39 * (AC / 40) > 12 then
    let x := AC / 40 in
    let s := (12 + 79 * x) / 2 in
    let area_squared := s * (s - 12) * (s - 39 * x) * (s - 40 * x) in
    (Real.sqrt (area_squared) / 16)
  else 0

theorem largest_area_is_82225 (AB BC AC : ℝ) (hAB : AB = 12) (h_ratio : BC / AC = 39 / 40) : 
  largest_triangle_area AB BC AC hAB h_ratio = 822.25 :=
by sorry

end largest_area_is_82225_l98_98924


namespace part1_part2_l98_98355

noncomputable def Sn (a: ℕ → ℕ) (n: ℕ) : ℕ := 2 * a n - n

theorem part1 (a : ℕ → ℕ) (n : ℕ) (Sn : ℕ → ℕ) (hSn : ∀ n, Sn n = 2 * a n - n) :
  ∃ r, ∀ n, a n + 1 = r ^ n :=
sorry

noncomputable def bn (n : ℕ) (a: ℕ → ℕ) : ℝ := n / (2^(n-1) * (2^n - a n))

noncomputable def Tn (n : ℕ) (a: ℕ → ℕ) : ℝ := ∑ i in Finset.range (n+1), bn i a

theorem part2 (a : ℕ → ℕ) (h : ∀ i, a i = 2^i - 1) (n : ℕ) :
  Tn n a = 4 - (n + 2) / 2^(n-1) :=
sorry

end part1_part2_l98_98355


namespace smallest_AB_l98_98338

-- Assume any necessary data structures and properties from Mathlib
noncomputable def Point : Type := (ℤ × ℤ)

-- Define the distance function following the given problem conditions
def distance (P Q : Point) : ℤ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  (x1 - x2)^2 + (y1 - y2)^2

-- Define the condition that the points are noncollinear
def noncollinear (A B C : Point) : Prop :=
  ∃ k1 k2 k3 : ℤ, k1 * (fst A - fst B) = k2 * (fst B - fst C) ∧ k1 * (snd A - snd B) = k3 * (snd B - snd C) ∧ k2 ≠ k3

-- The main theorem to prove the smallest value of AB
theorem smallest_AB (A B C : Point) (h_noncollinear : noncollinear A B C) (h_integer_coords : ∀ (P : Point), ∃ x y : ℤ, P = (x, y)) (h_integer_distances : ∃ n m p : ℤ, distance A B = n^2 ∧ distance A C = m^2 ∧ distance B C = p^2) :
  distance A B = 3 :=
by
  sorry

end smallest_AB_l98_98338


namespace bridge_length_l98_98931

variable (train_length : ℕ) (speed_kmh : ℕ) (time_sec : ℕ)

def convert_speed (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

def length_of_bridge (train_length speed_kmh time_sec : ℕ) : ℕ :=
  let speed_m_s := convert_speed speed_kmh
  let total_distance := speed_m_s * time_sec
  total_distance - train_length

theorem bridge_length :
  (train_length = 140) →
  (speed_kmh = 45) →
  (time_sec = 30) →
  length_of_bridge 140 45 30 = 235 :=
by
  intros h1 h2 h3
  dsimp [convert_speed, length_of_bridge]
  rw [h1, h2, h3]
  simp
  sorry

end bridge_length_l98_98931


namespace angle_BXY_thirty_degrees_l98_98811

theorem angle_BXY_thirty_degrees 
  (AB_par_CD : ∀ (a b c d : ℝ), parallel AB CD)
  (AXE_CYX_relation : ∀ (AXE_angle CYX_angle : ℝ), AXE_angle = 4 * CYX_angle - 90) :
  ∀ (BXY_angle : ℝ), BXY_angle = 30 := by
  sorry

end angle_BXY_thirty_degrees_l98_98811


namespace maximum_value_fraction_sum_l98_98706

theorem maximum_value_fraction_sum (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : 0 < c) (hd : 0 < d) (h1 : a + c = 20) (h2 : (a : ℝ) / b + (c : ℝ) / d < 1) :
  (a : ℝ) / b + (c : ℝ) / d ≤ 1385 / 1386 :=
sorry

end maximum_value_fraction_sum_l98_98706


namespace rabbit_time_to_travel_2_miles_l98_98168

noncomputable def rabbit_travel_time (d v: ℕ) : ℕ :=
(time_in_minutes: ℕ) * 60 / v * d

theorem rabbit_time_to_travel_2_miles :
  rabbit_travel_time 2 5 = 24 := sorry

end rabbit_time_to_travel_2_miles_l98_98168


namespace find_tangent_line_constant_l98_98386

noncomputable theory
open Real

theorem find_tangent_line_constant (k t x₁ x₂ : ℝ) 
  (h₁ : k = exp x₁)
  (h₂ : k = exp (x₂ + 1))
  (h₃ : x₁ = x₂ + 1)
  (h₄ : k * x₁ + t = exp x₁ + 2)
  (h₅ : k * x₂ + t = exp (x₂ + 1)) :
  t = 4 - 2 * log 2 :=
by sorry

end find_tangent_line_constant_l98_98386


namespace range_of_a_l98_98834

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ [-2, 0] then 2 - (1/2)^x else 2 - 2^x

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∃! x ∈ Ioc(-2, 6), f x - log a (x + 2) = 0) ↔ a ∈ Ioc(√2/4, 1/2) :=
sorry

end range_of_a_l98_98834


namespace count_numbers_with_perfect_square_factors_l98_98772

open Set

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ m * m ∣ n

theorem count_numbers_with_perfect_square_factors (s : Finset ℕ) (hs : s = Finset.range 101) :
  (Finset.filter has_perfect_square_factor_other_than_one s).card = 41 :=
by {
  sorry
}

end count_numbers_with_perfect_square_factors_l98_98772


namespace fair_coin_even_heads_unfair_coin_even_heads_l98_98244

-- Define the probability function for an even number of heads for a fair coin
theorem fair_coin_even_heads (n : ℕ) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * 0.5^k * 0.5^(n-k) else 0) = 0.5 :=
sorry

-- Define the probability function for an even number of heads for an unfair coin
theorem unfair_coin_even_heads (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  (∑ k in finset.range (n+1), if even k then (nat.choose n k : ℝ) * p^k * (1-p)^(n-k) else 0) = 
    (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_unfair_coin_even_heads_l98_98244


namespace average_death_rate_l98_98804

variable (birth_rate : ℕ) (net_increase_day : ℕ)

noncomputable def death_rate_per_two_seconds (birth_rate net_increase_day : ℕ) : ℕ :=
  let seconds_per_day := 86400
  let net_increase_per_second := net_increase_day / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  let death_rate_per_second := birth_rate_per_second - net_increase_per_second
  2 * death_rate_per_second

theorem average_death_rate
  (birth_rate : ℕ := 4) 
  (net_increase_day : ℕ := 86400) :
  death_rate_per_two_seconds birth_rate net_increase_day = 2 :=
sorry

end average_death_rate_l98_98804


namespace num_integers_for_perfect_square_ratio_l98_98687

theorem num_integers_for_perfect_square_ratio :
  (∃! (n : ℤ), (n / (30 - 2 * n) : ℚ) = m ^ 2 ∧ m ∈ ℤ ∧ 0 < n ∧ n < 15) → 
  (∃ (n1 n2 : ℤ), n1 ≠ n2 ∧ 
    (n1 / (30 - 2 * n1) : ℚ) = m1 ^ 2 ∧ m1 ∈ ℤ ∧ 
    (n2 / (30 - 2 * n2) : ℚ) = m2 ^ 2 ∧ m2 ∈ ℤ ∧ 
    0 < n1 ∧ n1 < 15 ∧ 0 < n2 ∧ n2 < 15)
:=
begin
  sorry
end

end num_integers_for_perfect_square_ratio_l98_98687


namespace min_value_function_l98_98511

theorem min_value_function (x : ℝ) (h : 1 < x) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y ≥ 3) :=
sorry

end min_value_function_l98_98511


namespace Simson_line_bisects_PM_midpoint_l98_98480

-- Given definitions and conditions
variable {Triangle : Type} [Simplex Triangle]
variable {P M : Triangle.Point} -- Points P and M in the triangle
variable [Circumcircle : CircumcircleProperties Triangle] -- Circumcircle of the triangle
variable [Orthocenter : OrthocenterProperties M Triangle] -- Orthocenter of the triangle

-- Statement of the theorem: Proving the Simson line bisects the segment PM
theorem Simson_line_bisects_PM_midpoint (SimsonLine : Line) (onCircumcircle : P ∈ Circumcircle.points)
  (M_is_orthocenter : Orthocenter.is_orthocenter M Triangle)
  (SimsonLine_property : SimsonLine = SimsonLine_of P)
  : SimsonLine.bisects_segment P M :=
sorry

end Simson_line_bisects_PM_midpoint_l98_98480


namespace max_additional_plates_l98_98409

def largest_additional_plates (A B C : ℕ) (n m : ℕ) : ℕ :=
  max (A * (B + 2) * C - A * B * C) 
      (max (A * (B + 1) * (C + 1) - A * B * C) 
           (max ((A + 2) * B * C - A * B * C) 
                (A * B * (C + 2) - A * B * C)))

theorem max_additional_plates (A B C : ℕ) (hA : A = 5) (hB : B = 2) (hC : C = 4)
  : largest_additional_plates A B C 2 2 = 40 :=
by {
  rw [hA, hB, hC],
  -- the largest_additional_plates function returns the required 40 additional plates
  -- proofs will be filled in here
  sorry
}

end max_additional_plates_l98_98409


namespace gold_copper_ratio_l98_98947

theorem gold_copper_ratio (G C : ℕ) 
  (h1 : 19 * G + 9 * C = 18 * (G + C)) : 
  G = 9 * C :=
by
  sorry

end gold_copper_ratio_l98_98947


namespace total_puppies_l98_98626

theorem total_puppies (a_b_given: ℕ) (a_b_kept: ℕ) (b_b_given: ℕ) (b_b_kept: ℕ):
  a_b_given = 20 → a_b_kept = 8 →
  b_b_given = 10 → b_b_kept = 6 →
  (a_b_given + a_b_kept + b_b_given + b_b_kept) = 44 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end total_puppies_l98_98626


namespace none_of_the_above_holds_l98_98836

variable (t : ℝ)
def x := t^(2 / (t - 2))
def y := t^(t / (t - 2))

theorem none_of_the_above_holds (ht : t > 2) :
  ¬ (y^x = x^(1/y)) ∧ ¬ (y^(1/x) = x^y) ∧ ¬ (y^x = x^y) ∧ ¬ (x^x = y^y) :=
by
  sorry

end none_of_the_above_holds_l98_98836


namespace count_numbers_with_perfect_square_factors_l98_98771

open Set

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ m * m ∣ n

theorem count_numbers_with_perfect_square_factors (s : Finset ℕ) (hs : s = Finset.range 101) :
  (Finset.filter has_perfect_square_factor_other_than_one s).card = 41 :=
by {
  sorry
}

end count_numbers_with_perfect_square_factors_l98_98771


namespace magnitude_of_vector_AB_l98_98707

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1)

-- Define the distance formula
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Proof statement
theorem magnitude_of_vector_AB : distance A B = Real.sqrt 2 := by
  sorry

end magnitude_of_vector_AB_l98_98707


namespace sum_of_exterior_angles_of_convex_polygon_l98_98135

theorem sum_of_exterior_angles_of_convex_polygon (P : Type) [polygon P] [convex P] :
  sum (exterior_angles P) = 360 := 
sorry

end sum_of_exterior_angles_of_convex_polygon_l98_98135


namespace solve_for_x_l98_98145

theorem solve_for_x (x : ℝ) (h : 0.60 * 500 = 0.50 * x) : x = 600 :=
  sorry

end solve_for_x_l98_98145


namespace range_of_a_l98_98352

variable {R : Type*}
variables (f : R → R) (a : R)

axiom decreasing_on_neg_inf_to_3 : ∀ x y, x ≤ y → x ≤ 3 → y ≤ 3 → f y ≤ f x

axiom inequality_for_all_x :
  ∀ x ∈ ℝ, f (a^2 - real.sin x) ≤ f (a + 1 + real.cos x ^ 2)

theorem range_of_a :
  -real.sqrt 2 ≤ a ∧ a ≤ (1 - real.sqrt 10) / 2 :=
sorry

end range_of_a_l98_98352


namespace even_heads_probability_fair_even_heads_probability_biased_l98_98242

variable  {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1)

/-- The probability of getting an even number of heads -/

/-- Case 1: Fair coin (p = 1/2) -/
theorem even_heads_probability_fair (n : ℕ) : 
  let p : ℝ := 1/2 in 
  let q : ℝ := 1 - p in 
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 0.5 := sorry

/-- Case 2: Biased coin -/
theorem even_heads_probability_biased {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1) : 
  let q : ℝ := 1 - p in
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 
  (1 + (1 - 2 * p)^n) / 2 := sorry

end even_heads_probability_fair_even_heads_probability_biased_l98_98242


namespace exists_i_with_α_close_to_60_l98_98327

noncomputable def α : ℕ → ℝ := sorry  -- Placeholder for the function α

theorem exists_i_with_α_close_to_60 :
  ∃ i : ℕ, abs (α i - 60) < 1
:= sorry

end exists_i_with_α_close_to_60_l98_98327


namespace div_by_eight_l98_98861

theorem div_by_eight (n : ℕ) : (5^n + 2 * 3^(n-1) + 1) % 8 = 0 :=
by
  sorry

end div_by_eight_l98_98861


namespace divide_talers_l98_98118

theorem divide_talers (loaves1 loaves2 : ℕ) (coins : ℕ) (loavesShared : ℕ) :
  loaves1 = 3 → loaves2 = 5 → coins = 8 → loavesShared = (loaves1 + loaves2) →
  (3 - loavesShared / 3) * coins / loavesShared = 1 ∧ (5 - loavesShared / 3) * coins / loavesShared = 7 := 
by
  intros h1 h2 h3 h4
  sorry

end divide_talers_l98_98118


namespace jackson_saving_l98_98427

theorem jackson_saving (total_amount : ℝ) (months : ℕ) (paychecks_per_month : ℕ) (savings_per_paycheck : ℝ) :
  total_amount = 3000 → months = 15 → paychecks_per_month = 2 →
  savings_per_paycheck = total_amount / months / paychecks_per_month :=
by sorry

end jackson_saving_l98_98427


namespace probability_sum_l98_98975

noncomputable def P : ℕ → ℝ := sorry

theorem probability_sum (n : ℕ) (h : n ≥ 7) :
  P n = (1/6) * (P (n-1) + P (n-2) + P (n-3) + P (n-4) + P (n-5) + P (n-6)) :=
sorry

end probability_sum_l98_98975


namespace base_length_of_isosceles_triangle_l98_98884

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l98_98884


namespace part_one_probability_part_two_distribution_part_two_expectation_part_three_most_likely_count_l98_98071

-- Definitions used in conditions
def choose (n k : ℕ) : ℕ := n.choose k
def probability_not_space_launch (n : ℕ) : ℝ := (choose 9 2 : ℝ) / (choose 10 2 : ℝ)
def probability_at_least_one_group (n : ℕ) : ℝ := 1 - (probability_not_space_launch n) ^ 4

-- Part 1
theorem part_one_probability : probability_at_least_one_group 10 = 369 / 625 := sorry

-- Definitions for part 2
def probability_distribution_X (k : ℕ) : ℝ := (choose 4 k : ℝ) * ((1 / 5) ^ k) * ((4 / 5) ^ (4 - k))
def expectation_X : ℝ := 4 * (1 / 5)

-- Part 2
theorem part_two_distribution (k : ℕ) (h : k ∈ {0, 1, 2, 3, 4}) : 
    probability_distribution_X k ∈ ({256/625, 256/625, 96/625, 16/625, 1/625} : set ℝ) := sorry

theorem part_two_expectation : expectation_X = 4 / 5 := sorry

-- Definition for part 3
def most_likely_to_choose_space_launch (n : ℕ) (h : n % 5 = 0) : ℕ := n / 5

-- Part 3
theorem part_three_most_likely_count (n : ℕ) (h : n % 5 = 0) : 
    most_likely_to_choose_space_launch n h = n / 5 := sorry

end part_one_probability_part_two_distribution_part_two_expectation_part_three_most_likely_count_l98_98071


namespace complex_mul_real_imag_l98_98725

theorem complex_mul_real_imag:
  let Z := (1 + Complex.i) * (2 + Complex.i ^ 607) in
  let m := Z.re in
  let n := Z.im in 
  m * n = 3 :=
by
  sorry

end complex_mul_real_imag_l98_98725


namespace find_some_value_l98_98147

theorem find_some_value (m n : ℝ) (some_value : ℝ) (p : ℝ) 
  (h1 : m = n / 6 - 2 / 5)
  (h2 : m + p = (n + some_value) / 6 - 2 / 5)
  (h3 : p = 3)
  : some_value = -12 / 5 :=
by
  sorry

end find_some_value_l98_98147


namespace worker_loading_time_l98_98623

theorem worker_loading_time
  (T : ℝ) 
  (h1 : ∀ x ∈ {0 .. T}, x ≠ 0 → 1 / x + 1 / 4 = 1 / 2.4) :
  T = 6 := by
  sorry

end worker_loading_time_l98_98623


namespace poly_div_simplification_l98_98586

-- Assume a and b are real numbers.
variables (a b : ℝ)

-- Theorem to prove the equivalence
theorem poly_div_simplification (a b : ℝ) : (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b :=
by
  -- The proof will go here
  sorry

end poly_div_simplification_l98_98586


namespace rajs_house_bathrooms_l98_98058

theorem rajs_house_bathrooms :
  ∀ (house_area bedroom_area_per_room num_bedrooms bathroom_area kitchen_area : ℕ),
  house_area = 1110 →
  bedroom_area_per_room = 11 * 11 →
  num_bedrooms = 4 →
  bathroom_area = 6 * 8 →
  kitchen_area = 265 →
  house_area = (num_bedrooms * bedroom_area_per_room) + (kitchen_area * 2) + (n * bathroom_area) →
  n = 2 :=
begin
  -- Proof is omitted
  sorry
end

end rajs_house_bathrooms_l98_98058


namespace closest_vertex_proof_l98_98064

def is_closest_vertex_to_origin 
  (center : ℝ × ℝ) 
  (area : ℝ) 
  (horizontal : bool)
  (dilation_center : ℝ × ℝ) 
  (scale_factor : ℝ) 
  (rotation_degrees : ℝ) 
  (target_vertex : ℝ × ℝ) : Prop := sorry

theorem closest_vertex_proof : 
  is_closest_vertex_to_origin (6, -6) 16 true (0, 0) 3 90 (-12, -12) :=
sorry

end closest_vertex_proof_l98_98064


namespace range_of_a_l98_98685

theorem range_of_a (a : ℝ) (h : ∀ t : ℝ, 0 < t → t ≤ 2 → (t / (t^2 + 9) ≤ a ∧ a ≤ (t + 2) / t^2)) : 
  (2 / 13) ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l98_98685


namespace circumcircle_BCK_tangent_to_m_l98_98585

theorem circumcircle_BCK_tangent_to_m
  (ABC : Triangle)
  (A B C C1 A1 B1 K m : Point)
  (h0 : Incircle ABC touches AB at C1)
  (h1 : Incircle ABC touches BC at A1)
  (h2 : Incircle ABC touches CA at B1)
  (h3 : midline_triangle A1 B1 C1 m)
  (h4 : is_parallel m B1 C1)
  (h5 : angle_bisector_meet B1 A1 C1 m K) :
  Tangent (circumcircle B C K) m := 
sorry

end circumcircle_BCK_tangent_to_m_l98_98585


namespace isosceles_triangle_base_length_l98_98886

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l98_98886


namespace geometric_seq_a7_l98_98306

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98306


namespace sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l98_98870

theorem sum_of_roots_eq_zero (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 + x2 = 0 :=
by
  sorry

theorem product_of_roots_eq_neg_twentyfive (x : ℝ) (h : |x|^2 - 3 * |x| - 10 = 0) :
  ∃ x1 x2 : ℝ, (|x1| = 5) ∧ (|x2| = 5) ∧ x1 * x2 = -25 :=
by
  sorry

end sum_of_roots_eq_zero_product_of_roots_eq_neg_twentyfive_l98_98870


namespace problem_I_problem_II_exists_arith_seq_problem_III_l98_98012

noncomputable def a_seq (m : ℝ) : ℕ → ℝ
  | 1 => 0
  | (n+1) => (a_seq m n) ^ 2 + m

theorem problem_I (m : ℝ) (h : m = 1) :
  a_seq m 2 = 1 ∧ a_seq m 3 = 2 ∧ a_seq m 4 = 5 :=
  by
    sorry

theorem problem_II_exists_arith_seq (m : ℝ) :
  (m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2) → 
  (∃ d : ℝ, d ≠ 0 ∧ a_seq m 3 - a_seq m 2 = d ∧ a_seq m 4 - a_seq m 3 = d) :=
  by
    sorry

theorem problem_III (m : ℝ) (h : m > 1/4) :
  ∃ k : ℕ, 1 ≤ k ∧ a_seq m k > 2016 :=
  by
    let d := m - 1/4
    have d_pos : d > 0 := by apply sub_pos.mpr h
    sorry

end problem_I_problem_II_exists_arith_seq_problem_III_l98_98012


namespace probability_product_divisible_by_four_l98_98069

open Finset

theorem probability_product_divisible_by_four :
  (∃ (favorable_pairs total_pairs : ℕ), favorable_pairs = 70 ∧ total_pairs = 190 ∧ favorable_pairs / total_pairs = 7 / 19) := 
sorry

end probability_product_divisible_by_four_l98_98069


namespace geometric_series_mod_2000_l98_98559

theorem geometric_series_mod_2000 :
  (∑ i in Finset.range 1501, 9^i) % 2000 = (1 + 9 + 9^2 + ⋯ + 9^1500) % 2000 :=
by
  let S := ∑ i in Finset.range 1501, 9^i
  have h1 : S = (9^1501 - 1) / 8, from sorry
  exact h1 % 2000

end geometric_series_mod_2000_l98_98559


namespace total_bill_is_270_l98_98872

-- Conditions as Lean definitions
def totalBill (T : ℝ) : Prop :=
  let eachShare := T / 10
  9 * (eachShare + 3) = T

-- Theorem stating that the total bill T is 270
theorem total_bill_is_270 (T : ℝ) (h : totalBill T) : T = 270 :=
sorry

end total_bill_is_270_l98_98872


namespace count_triples_satisfying_conditions_l98_98675

open Nat

theorem count_triples_satisfying_conditions :
  let gcd15 (a b c : ℕ) := gcd (gcd a b) c = 15
  let lcm_comp (a b c : ℕ) := lcm (lcm a b) c = 3^15 * 5^18 
  Finset.card { (a, b, c) : ℕ × ℕ × ℕ | gcd15 a b c ∧ lcm_comp a b c } = 8568 := by
  sorry

end count_triples_satisfying_conditions_l98_98675


namespace find_common_ratio_l98_98340

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Conditions:
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a n * q
def condition_a2 := a 2 = 2
def condition_a5 := a 5 = 1 / 4

theorem find_common_ratio
  (geom_seq : geometric_sequence a q)
  (cond_a2 : condition_a2)
  (cond_a5 : condition_a5) :
  q = 1 / 2 :=
sorry

end find_common_ratio_l98_98340


namespace x_finishes_remaining_work_in_18_days_l98_98580

theorem x_finishes_remaining_work_in_18_days 
    (x y : Type) [HasDiv x ℝ] [HasDiv y ℝ]
    (work_x : ℝ := 1)
    (work_y : ℝ := 1) 
    (x_days_finish_work : ℝ := 36) 
    (y_days_finish_work : ℝ := 24) 
    (y_days_worked : ℝ := 12) 
    (y_work_rate : ℝ := work_y / y_days_finish_work)
    (y_work_done : ℝ := y_days_worked * y_work_rate) :
    let remaining_work := work_x - y_work_done,
        x_work_rate := work_x / x_days_finish_work,
        x_days_remaining := remaining_work / x_work_rate
    in x_days_remaining = 18 := 
sorry

end x_finishes_remaining_work_in_18_days_l98_98580


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98252

-- Problem 1: Fair coin, probability of even heads
def fair_coin_even_heads_prob (n : ℕ) : Prop :=
  0.5 = 0.5

-- Problem 2: Biased coin, probability of even heads
def biased_coin_even_heads_prob (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : Prop :=
  let q := 1 - p in
  (0.5 * (1 + (1 - 2 * p)^n) = (1 + (1 - 2*p)^n) / 2)

-- Mock proof to ensure Lean accepts the definitions
theorem fair_coin_even_heads (n : ℕ) : fair_coin_even_heads_prob n :=
begin
  -- Proof intentionally omitted
  sorry
end

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : biased_coin_even_heads_prob n p hp :=
begin
  -- Proof intentionally omitted
  sorry
end

end fair_coin_even_heads_biased_coin_even_heads_l98_98252


namespace tangent_lines_max_triangle_area_l98_98323

-- Define the circle C with its equation
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 10 * x - 14 * y + 70 = 0

-- Define the point P
def point_P : (ℝ × ℝ) := (-3, 1)

-- Define the line l
def line_l (x y : ℝ) : Prop := 12 * x + 5 * y + 12 = 0

-- Define the tangent lines problem
theorem tangent_lines (x y : ℝ) :
  (∃ k : ℝ, y - 1 = k * (x + 3)) ∨ (x = -3) ∨ (4 * x + 3 * y + 9 = 0) :=
sorry

-- Define the maximum area problem
theorem max_triangle_area : ∃ (A B Q : (ℝ × ℝ)), 
  triangle_area (A B Q) = 3 * real.sqrt 3 ∧ 
  is_moving_point (circle_C) A B Q :=
sorry

-- Define the area calculation function and the moving point condition accordingly
noncomputable def triangle_area (A B Q : ℝ × ℝ) : ℝ := sorry
def is_moving_point (C : ℝ ×ℝ → Prop) (A B Q : ℝ ×ℝ) : Prop := sorry

end tangent_lines_max_triangle_area_l98_98323


namespace single_woman_work_rate_l98_98621

variable (days_to_complete : ℕ)

def work_rate (days : ℝ) : ℝ := 1 / days

variables (M W B : ℝ)
variables (team_work_rate : ℝ)

axiom h1 : team_work_rate = work_rate 5
axiom h2 : M = work_rate 10
axiom h3 : B = work_rate 25
axiom h4 : 3 * W = work_rate 15

theorem single_woman_work_rate :
  (2 * M + 3 * W + 4 * B = team_work_rate) →
  W = work_rate 50 :=
by
  intros h
  sorry

end single_woman_work_rate_l98_98621


namespace maximum_positive_numbers_l98_98050

theorem maximum_positive_numbers (a : ℕ → ℝ) (n : ℕ) (h₀ : n = 100)
  (h₁ : ∀ i : ℕ, 0 < a i) 
  (h₂ : ∀ i : ℕ, a i > a ((i + 1) % n) * a ((i + 2) % n)) : 
  ∃ m : ℕ, m ≤ 50 ∧ (∀ k : ℕ, k < m → (a k) > 0) :=
by sorry

end maximum_positive_numbers_l98_98050


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98250

-- Problem 1: Fair coin, probability of even heads
def fair_coin_even_heads_prob (n : ℕ) : Prop :=
  0.5 = 0.5

-- Problem 2: Biased coin, probability of even heads
def biased_coin_even_heads_prob (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : Prop :=
  let q := 1 - p in
  (0.5 * (1 + (1 - 2 * p)^n) = (1 + (1 - 2*p)^n) / 2)

-- Mock proof to ensure Lean accepts the definitions
theorem fair_coin_even_heads (n : ℕ) : fair_coin_even_heads_prob n :=
begin
  -- Proof intentionally omitted
  sorry
end

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (hp : 0 < p ∧ p < 1) : biased_coin_even_heads_prob n p hp :=
begin
  -- Proof intentionally omitted
  sorry
end

end fair_coin_even_heads_biased_coin_even_heads_l98_98250


namespace power_function_evaluation_l98_98741

theorem power_function_evaluation (f : ℝ → ℝ) (a : ℝ)
  (h₁ : ∀ x, f x = x ^ a)
  (h₂ : f 2 = (Real.sqrt 2) / 2) :
  f 4 = 1 / 2 := by
  sorry

end power_function_evaluation_l98_98741


namespace fill_pool_total_water_l98_98533

variable TinaPail TommyPail TimmyPail : ℕ

def Tina_pail := 4
def Tommy_pail := Tina_pail + 2
def Timmy_pail := 2 * Tommy_pail

def water_per_trip := Tina_pail + Tommy_pail + Timmy_pail
def total_water := water_per_trip * 3

theorem fill_pool_total_water : total_water = 66 := by
  -- reduce the equation
  unfold Tina_pail Tommy_pail Timmy_pail water_per_trip total_water
  -- check that it equals 66
  rfl

end fill_pool_total_water_l98_98533


namespace p_prime_eq_nq_l98_98954

noncomputable def p (x : ℝ) : ℝ := sorry  -- Define the polynomial p(x)
noncomputable def q (x : ℝ) : ℝ := sorry  -- Define the polynomial q(x)
def n : ℕ := sorry  -- Define the degree n of p(x)
def m : ℕ := sorry  -- Define the degree m of q(x)
def c : ℝ := sorry  -- Define the leading coefficient c of both polynomials

axiom h1 : degree p = n  -- p(x) is a polynomial of degree n
axiom h2 : degree q = m  -- q(x) is a polynomial of degree m
axiom h3 : leading_coeff p = c  -- Leading coefficient of p(x)
axiom h4 : leading_coeff q = c  -- Leading coefficient of q(x)
axiom h5 : ∀ x, (p x)^2 = (x^2 - 1) * (q x)^2 + 1  -- Given equation

theorem p_prime_eq_nq : ∀ x, derivative p x = n * q x :=
by
  sorry

end p_prime_eq_nq_l98_98954


namespace probability_painted_faces_l98_98600

theorem probability_painted_faces (total_cubes : ℕ) (corner_cubes : ℕ) (no_painted_face_cubes : ℕ) (successful_outcomes : ℕ) (total_outcomes : ℕ) 
  (probability : ℚ) : 
  total_cubes = 125 ∧ corner_cubes = 8 ∧ no_painted_face_cubes = 27 ∧ successful_outcomes = 216 ∧ total_outcomes = 7750 ∧ 
  probability = 72 / 2583 :=
by
  sorry

end probability_painted_faces_l98_98600


namespace perfect_square_factors_count_l98_98753

def perfectSquares := [4, 9, 16, 25, 36, 49, 64, 81]

def countNumbersWithPerfectSquareFactors : Nat :=
  List.length (List.filter (fun n => perfectSquares.any (fun p => n % p = 0)) [1..100])

theorem perfect_square_factors_count :
  countNumbersWithPerfectSquareFactors = 41 := sorry

end perfect_square_factors_count_l98_98753


namespace angles_of_A2B2C2_l98_98471

theorem angles_of_A2B2C2 (α β γ : ℝ) (A B C A₁ B₁ C₁ A₂ B₂ C₂ : Type) [Equilateral_triangle A B C]
  (isosceles_A₁BC : is_isosceles_triangle A₁ B C α)
  (isosceles_AB₁C : is_isosceles_triangle A B₁ C β)
  (isosceles_ABC₁ : is_isosceles_triangle A B C₁ γ)
  (sum_angles : α + β + γ = 60) 
  (intersect_A₂ : line B C₁ ∩ line B₁ C = A₂)
  (intersect_B₂ : line A C₁ ∩ line A₁ C = B₂)
  (intersect_C₂ : line A B₁ ∩ line A₁ B = C₂) :
  ∠ B₂ A₂ C₂ = 3 * α ∧ ∠ A₂ B₂ C₂ = 3 * β ∧ ∠ A₂ C₂ B₂ = 3 * γ :=
sorry

end angles_of_A2B2C2_l98_98471


namespace product_of_distances_to_tangents_equals_product_of_distances_to_sides_l98_98599

-- Defining the geometric conditions
variable {n : ℕ} (A : fin n → Points) (O : Points) (r : ℝ) (M : Points)
variable {A_on_circle : ∀ i, dist O (A i) = r}
variable {A_eq_center : ¬ ∃ i, A i = O}
variable {M_on_circle : dist O M = r}

-- Statement of the theorem
theorem product_of_distances_to_tangents_equals_product_of_distances_to_sides : 
  (∏ i in finset.range n, dist M (tangent_to_circle_at (A i))) = 
  (∏ i in finset.range n, dist M (line_through (A i) (A ((i + 1) % n)))) := 
sorry

end product_of_distances_to_tangents_equals_product_of_distances_to_sides_l98_98599


namespace symmetric_axis_of_g_l98_98364

-- Define the original function f
def f (x : ℝ) : ℝ := 2 * sin (2 * x + (Real.pi / 6))

-- Define the transformed function g
def g (x : ℝ) : ℝ := 2 * sin (4 * x + (Real.pi / 6))

-- Definition of the symmetric axis equation condition
def symmetric_axis (x : ℝ) : Prop := ∃ k : ℤ, x = (Real.pi / 12) + k * (Real.pi / 4)

-- The theorem we want to prove is that x = π / 12 is one symmetric axis of g(x)
theorem symmetric_axis_of_g : symmetric_axis (Real.pi / 12) :=
sorry

end symmetric_axis_of_g_l98_98364


namespace listK_consecutive_integers_count_l98_98850

-- Given conditions
def listK := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] -- A list K consisting of consecutive integers
def leastInt : Int := -5 -- The least integer in list K
def rangePosInt : Nat := 5 -- The range of the positive integers in list K

-- The theorem to prove
theorem listK_consecutive_integers_count : listK.length = 11 := by
  -- skipping the proof
  sorry

end listK_consecutive_integers_count_l98_98850


namespace compute_square_l98_98641

theorem compute_square : (80 + 5) ^ 2 = 7225 := by
  let a := 80
  let b := 5
  have identity : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by sorry
  calc
    (80 + 5) ^ 2 = (a + b) ^ 2 : by rw [←identity]
    ... = a ^ 2 + 2 * a * b + b ^ 2 : by sorry -- expand using the identity
    ... = 80 ^ 2 + 2 * 80 * 5 + 5 ^ 2 : by rw [a, b]
    ... = 6400 + 800 + 25 : by sorry -- arithmetic simplifications
    ... = 7225 : by sorry

end compute_square_l98_98641


namespace water_added_is_two_l98_98591

noncomputable def amount_of_water_added (initial_volume : ℕ) (initial_percentage_alcohol : ℝ) 
    (final_percentage_alcohol : ℝ) (amount_alcohol : ℝ := initial_volume * initial_percentage_alcohol) 
    : ℕ := 
let new_volume := amount_alcohol / final_percentage_alcohol in 
nat.ceil (new_volume - initial_volume)

theorem water_added_is_two : amount_of_water_added 15 (20 / 100) (17.647058823529413 / 100) = 2 :=
by
  -- This is the main theorem statement following from the problem
  sorry

end water_added_is_two_l98_98591


namespace find_a7_l98_98300

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98300


namespace log4x_and_4x_symmetric_y_eq_x_l98_98078

open Function

-- Define the functions
def f (x : ℝ) : ℝ := log x / log 4
def g (x : ℝ) : ℝ := 4 ^ x

-- Define the problem statement
theorem log4x_and_4x_symmetric_y_eq_x :
  (∀ x, f (g x) = x) ∧ (∀ y, g (f y) = y) :=
sorry

end log4x_and_4x_symmetric_y_eq_x_l98_98078


namespace trailing_zeros_product_remainder_l98_98032

theorem trailing_zeros_product_remainder :
  let M := (finset.range 50).sum (λ i, finset.range (i+1)).sum (λ j, (j+1) factors 5) / 5 :=
  M % 100 = 12 :=
by
-- use the conditions directly
sorry

end trailing_zeros_product_remainder_l98_98032


namespace positive_integer_solutions_to_equation_l98_98085

theorem positive_integer_solutions_to_equation :
  ( ∃ n, (2 * x + 3 * y = 763) ∧ x > 0 ∧ y > 0 ) ↔ n = 127 :=
sorry

end positive_integer_solutions_to_equation_l98_98085


namespace range_of_k_l98_98788

noncomputable def f (k x : ℝ) := k * x - Real.exp x
noncomputable def g (x : ℝ) := Real.exp x / x

theorem range_of_k (k : ℝ) (h : ∃ x : ℝ, x ≠ 0 ∧ f k x = 0) :
  k < 0 ∨ k ≥ Real.exp 1 := sorry

end range_of_k_l98_98788


namespace tim_score_in_math_l98_98578

def even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def sum_even_numbers (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem tim_score_in_math : sum_even_numbers even_numbers = 56 := by
  -- Proof steps would be here
  sorry

end tim_score_in_math_l98_98578


namespace Jill_has_6_peaches_l98_98429

-- Define the conditions
def Jake_has_18_fewer_peaches_than_Steven (Jake Steven : ℕ) : Prop :=
  Jake = Steven - 18

def Steven_has_13_more_peaches_than_Jill (Steven Jill : ℕ) : Prop :=
  Steven = Jill + 13

def Steven_has_19_peaches : Prop :=
  Steven = 19

def Sam_has_twice_as_many_peaches_as_Jill (Sam Jill : ℕ) : Prop :=
  Sam = 2 * Jill

-- Define the main theorem that we need to prove based on the conditions.
theorem Jill_has_6_peaches (Steven Jill : ℕ)
  (h1 : Steven_has_19_peaches)
  (h2 : Steven_has_13_more_peaches_than_Jill Steven Jill) :
  Jill = 6 :=
by
  -- Proof will be here
  sorry

end Jill_has_6_peaches_l98_98429


namespace units_digit_eight_consecutive_numbers_l98_98215

theorem units_digit_eight_consecutive_numbers (n : ℕ) : 
  (∏ i in finset.range 8, (n + i)) % 10 = 0 :=
by
  sorry

end units_digit_eight_consecutive_numbers_l98_98215


namespace sum_of_numbers_l98_98538

variable {R : Type*} [LinearOrderedField R]

theorem sum_of_numbers (x y : R) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 :=
by
  sorry

end sum_of_numbers_l98_98538


namespace number_of_teachers_l98_98618

theorem number_of_teachers
  (T S : ℕ)
  (h1 : T + S = 2400)
  (h2 : 320 = 320) -- This condition is trivial and can be ignored
  (h3 : 280 = 280) -- This condition is trivial and can be ignored
  (h4 : S / 280 = T / 40) : T = 300 :=
by
  sorry

end number_of_teachers_l98_98618


namespace isosceles_triangle_base_length_l98_98885

theorem isosceles_triangle_base_length (b : ℕ) (h₁ : 6 + 6 + b = 20) : b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l98_98885


namespace geometric_sequence_306th_term_l98_98406

def geometric_sequence_nth_term (a r : ℤ) (n : ℕ) : ℤ :=
  a * r^(n - 1)

theorem geometric_sequence_306th_term :
  let a := 9
  let a2 := -18
  let r := a2 / a
  r = -2 →
  geometric_sequence_nth_term a r 306 = -9 * 2^305 :=
by
  intros a a2 r
  simp
  intro hr
  rw hr
  sorry

end geometric_sequence_306th_term_l98_98406


namespace maximum_value_of_f_on_interval_l98_98083

-- Define the function f
def f (x : ℝ) : ℝ := (1 / 2) * x - Real.sin x

-- Define the interval
def interval : Set ℝ := Set.Icc (-Real.pi / 2) (Real.pi / 2)

-- State the theorem
theorem maximum_value_of_f_on_interval : 
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = -Real.pi / 4 + 1 :=
sorry

end maximum_value_of_f_on_interval_l98_98083


namespace altitude_inequality_l98_98037

variables {a b c h_a h_b h_c : ℝ}

-- Define conditions as hypotheses
def are_altitudes_of_triangle (a b c h_a h_b h_c : ℝ) : Prop :=
  ∃ (S : ℝ), S = 1/2 * a * h_a ∧ S = 1/2 * b * h_b ∧ S = 1/2 * c * h_c

-- Conjecture expressed in Lean
theorem altitude_inequality (h : are_altitudes_of_triangle a b c h_a h_b h_c) :
  (h_b^2 + h_c^2) / a^2 + (h_c^2 + h_a^2) / b^2 + (h_a^2 + h_b^2) / c^2 ≤ 9 / 2 :=
begin
  sorry,
end

end altitude_inequality_l98_98037


namespace checker_moves_10_cells_l98_98162

theorem checker_moves_10_cells :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ a 2 = 2 ∧ (∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) ∧ a 10 = 89 :=
by
  -- mathematical proof goes here
  sorry

end checker_moves_10_cells_l98_98162


namespace domain_of_f_l98_98670

theorem domain_of_f (k : ℝ) (hk : k < -4 / 3) :
  ∀ x : ℝ, -3 * (x * x) + 4 * x + k ≠ 0 :=
by
  intro x
  have hΔ : 16 + 12 * k < 0 := by
    calc
      16 + 12 * k < 0 : by
        exact add_lt_zero_of_lt_of_le (show (16 : ℝ) < 0 by norm_num1) (by linarith [hk])
  sorry

end domain_of_f_l98_98670


namespace even_heads_probability_fair_even_heads_probability_biased_l98_98243

variable  {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1)

/-- The probability of getting an even number of heads -/

/-- Case 1: Fair coin (p = 1/2) -/
theorem even_heads_probability_fair (n : ℕ) : 
  let p : ℝ := 1/2 in 
  let q : ℝ := 1 - p in 
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 0.5 := sorry

/-- Case 2: Biased coin -/
theorem even_heads_probability_biased {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1) : 
  let q : ℝ := 1 - p in
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 
  (1 + (1 - 2 * p)^n) / 2 := sorry

end even_heads_probability_fair_even_heads_probability_biased_l98_98243


namespace union_of_M_N_is_real_set_l98_98847

-- Define the set M
def M : Set ℝ := { x | x^2 + 3 * x + 2 > 0 }

-- Define the set N
def N : Set ℝ := { x | (1 / 2 : ℝ) ^ x ≤ 4 }

-- The goal is to prove that the union of M and N is the set of all real numbers
theorem union_of_M_N_is_real_set : M ∪ N = Set.univ :=
by
  sorry

end union_of_M_N_is_real_set_l98_98847


namespace sides_of_ABC_as_midlines_of_A1B1C1_l98_98105

open EuclideanGeometry

variables {A B C A1 B1 C1 : Point}

theorem sides_of_ABC_as_midlines_of_A1B1C1 
  (h1 : Parallel (Line.mk A B1) (Line.mk B C))
  (h2 : Parallel (Line.mk C B1) (Line.mk A B))
  (h3 : Parallel (Line.mk A C1) (Line.mk B C))
  (h4 : Parallel (Line.mk B C1) (Line.mk A C))
  (h5 : Parallel (Line.mk B A1) (Line.mk A C))
  (h6 : Parallel (Line.mk C A1) (Line.mk B A)) :
  Midpoint A (Segment.mk B1 C1) ∧
  Midpoint B (Segment.mk C1 A1) ∧
  Midpoint C (Segment.mk A1 B1) :=
sorry

end sides_of_ABC_as_midlines_of_A1B1C1_l98_98105


namespace blue_square_percentage_l98_98952

-- Define variables and hypotheses
variables (s : ℝ) (A_flag : ℝ := s^2) (percent_cross : ℝ := 36)
variables (percent_blue : ℝ := 2)

-- Define the condition that the cross occupies 36% of the flag's area
def cross_area (s : ℝ) : ℝ := 0.36 * s^2

-- Define the condition that the blue square's area
def blue_square_area (s : ℝ) : ℝ := 0.04 * s^2

-- Define the theorem that the blue square occupies 2% of the flag's area
theorem blue_square_percentage (s : ℝ) (h : cross_area s / A_flag = 0.36) : 
  ∃ (percent_blue : ℝ), (blue_square_area s / A_flag) * 100 = percent_blue ∧ percent_blue = 2 :=
by
  unfold A_flag cross_area blue_square_area
  sorry

end blue_square_percentage_l98_98952


namespace ratio_of_u_to_v_l98_98542

theorem ratio_of_u_to_v (b u v : ℝ) (Hu : u = -b/12) (Hv : v = -b/8) : 
  u / v = 2 / 3 := 
sorry

end ratio_of_u_to_v_l98_98542


namespace prob_exactly_4_correct_prob_exactly_5_correct_prob_all_8_correct_l98_98049

def C (n k : ℕ) : ℕ := nat.choose n k

theorem prob_exactly_4_correct :
  let total_ways_choose_8_out_of_64 := C 64 8 in
  let ways_choose_4_out_of_8 := C 8 4 in
  let ways_choose_4_out_of_56 := C 56 4 in
  let prob_4_correct := (ways_choose_4_out_of_8 * ways_choose_4_out_of_56) / total_ways_choose_8_out_of_64 in
  prob_4_correct = (C 8 4 * C 56 4) / C 64 8 := sorry

theorem prob_exactly_5_correct :
  let total_ways_choose_8_out_of_64 := C 64 8 in
  let ways_choose_5_out_of_8 := C 8 5 in
  let ways_choose_3_out_of_56 := C 56 3 in
  let prob_5_correct := (ways_choose_5_out_of_8 * ways_choose_3_out_of_56) / total_ways_choose_8_out_of_64 in
  prob_5_correct = (C 8 5 * C 56 3) / C 64 8 := sorry

theorem prob_all_8_correct :
  let total_ways_choose_8_out_of_64 := C 64 8 in
  let prob_8_correct := 1 / total_ways_choose_8_out_of_64 in
  prob_8_correct = 1 / C 64 8 := sorry

end prob_exactly_4_correct_prob_exactly_5_correct_prob_all_8_correct_l98_98049


namespace find_rs_l98_98033

-- Points in a vector space
structure Point :=
  (x : ℝ)
  (y : ℝ)

notation "ℝ²" => Point

-- Given points C and D in ℝ²
variables (C D : ℝ²)

-- Given the ratio of division CQ : QD = 3 : 5
def point_on_segment (C D : ℝ²) (m n : ℝ) : ℝ² := 
  { x := (m * D.x + n * C.x) / (m + n),
    y := (m * D.y + n * C.y) / (m + n) }

-- Define the point Q where CQ : QD = 3 : 5
def Q : ℝ² := point_on_segment C D 3 5

-- Hypothesis stating that Q can be expressed as a linear combination of C and D
theorem find_rs :
  ∃ (r s : ℝ), Q = { x := r * C.x + s * D.x, y := r * C.y + s * D.y } ∧ r = 5/8 ∧ s = 3/8 :=
by
  sorry

end find_rs_l98_98033


namespace mixture_produced_l98_98941

noncomputable def total_mixture_produced (X : Real) : Real :=
  let value_candy1 := 3.50 * X
  let value_candy2 := 4.30 * 6.25
  let total_value_mixture := value_candy1 + value_candy2
  let pounds_mixture := X + 6.25
  total_value_mixture = 4.00 * pounds_mixture

theorem mixture_produced (X : Real) : total_mixture_produced X = 10 :=
by
  sorry

end mixture_produced_l98_98941


namespace continuous_function_existence_l98_98254

theorem continuous_function_existence (a : ℝ) (h : 0 <= a) :
  (∃ f : ℝ → ℝ, continuous f ∧ ∀ x : ℝ, f (f x) = (x - a) ^ 2) ↔ a = 0 :=
begin
  sorry  -- proof to be filled in
end

end continuous_function_existence_l98_98254


namespace sum_of_m_and_n_is_negative_one_l98_98705

theorem sum_of_m_and_n_is_negative_one (m n : ℂ) (h_distinct : m ≠ n) (h_nonzero : m * n ≠ 0) (h_set : {m^2, n^2} = {m, n}) :
  m + n = -1 :=
sorry

end sum_of_m_and_n_is_negative_one_l98_98705


namespace find_a7_l98_98296

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98296


namespace M_subset_N_l98_98745

open Set

def M : Set ℝ := {x | ∃ k : ℤ, x = (k * π / 4) + (π / 4)}

def N : Set ℝ := {x | ∃ k : ℤ, x = (k * π / 8) - (π / 4)}

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l98_98745


namespace decreasing_sequence_after_99_l98_98909

theorem decreasing_sequence_after_99 :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (100^(n+1) / (n+1)!) < (100^n / n!) := 
sorry

end decreasing_sequence_after_99_l98_98909


namespace evaluate_expression_l98_98937

theorem evaluate_expression :
  2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 :=
by
  sorry

end evaluate_expression_l98_98937


namespace unit_vector_is_correct_l98_98332

-- Definition of points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Definition of the vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Magnitude of vector AB
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

-- Unit vector in the same direction as vectorAB
def unitVector (v : ℝ × ℝ) : ℝ × ℝ := (v.1 / magnitude v, v.2 / magnitude v)

-- The desired unit vector
theorem unit_vector_is_correct : unitVector vectorAB = (3/5, -4/5) :=
by
  -- skipping the actual proof steps
  sorry

end unit_vector_is_correct_l98_98332


namespace range_and_area_of_triangle_l98_98731

theorem range_and_area_of_triangle
  (f : ℝ → ℝ)
  (h_f : ∀ x, x ∈ set.Icc (Real.pi / 3) (11 * Real.pi / 24) → 
               f x = 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3)
  (x_in_domain : ∀ x, x ∈ set.Icc (Real.pi / 3) (11 * Real.pi / 24)) :
  set.range f = set.Icc (Real.sqrt 3) 2 ∧
  let a := Real.sqrt 3,
      b := 2,
      r := 3 * Real.sqrt 2 / 4,
      sinA := a / (2 * r),
      sinB := b / (2 * r),
      cosA := Real.sqrt 3 / 3,
      cosB := 1 / 3,
      sinC := sinA * cosB + cosA * sinB,
      S := 1 / 2 * a * b * sinC in
  S = Real.sqrt 2 :=
sorry

end range_and_area_of_triangle_l98_98731


namespace precise_set_criteria_l98_98563

-- Definitions representing the membership criteria for each option.
def is_tall_student (s: Student) : Prop := -- precise definition needed, assumed to be vague here
  sorry

def is_tall_tree (t: Tree) : Prop := -- precise definition needed, assumed to be vague here
  sorry

def is_first_grade_student (s: Student) (school: School) (date: Date) : Prop :=
  s.grade = 1 ∧ s.school = school ∧ date = Date.mk 2013 1

def has_high_basketball_skills (s: Student) : Prop := -- precise definition needed, assumed to be vague here
  sorry

-- The Lean statement of the problem
theorem precise_set_criteria :
  ∀ (students: Set Student) (trees: Set Tree) (school: School) (date: Date),
  {x | is_first_grade_student x school date} = students ↔ (students ⊆ {x | is_first_grade_student x school date}) :=
by sorry

end precise_set_criteria_l98_98563


namespace ramu_profit_percent_l98_98148

theorem ramu_profit_percent (car_cost : ℕ) (repair_cost : ℕ) (selling_price : ℕ) :
  car_cost = 42000 → 
  repair_cost = 8000 → 
  selling_price = 64900 → 
  (selling_price - (car_cost + repair_cost)) * 100 / (car_cost + repair_cost) = 29.8 := 
by
  intros h1 h2 h3
  have h4 := h1.symm
  have h5 := h2.symm
  have h6 := h3.symm
  sorry

end ramu_profit_percent_l98_98148


namespace find_f_2004_l98_98717

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom odd_g : ∀ x : ℝ, g (-x) = -g x
axiom g_eq_f_shift : ∀ x : ℝ, g x = f (x - 1)
axiom g_one : g 1 = 2003

theorem find_f_2004 : f 2004 = 2003 :=
  sorry

end find_f_2004_l98_98717


namespace cube_volume_increase_l98_98601

theorem cube_volume_increase (s : ℝ) : 
  let s_new := 1.15 * s
  let V_original := s^3
  let V_new := (1.15 * s)^3
  let percentage_increase := ((V_new - V_original) / V_original) * 100
  1.15^3 ≈ 1.520875 → 
  percentage_increase ≈ 52.09 :=
by
  sorry

end cube_volume_increase_l98_98601


namespace most_precise_value_l98_98001

def D := 3.27645
def error := 0.00518
def D_upper := D + error
def D_lower := D - error
def rounded_D_upper := Float.round (D_upper * 10) / 10
def rounded_D_lower := Float.round (D_lower * 10) / 10

theorem most_precise_value :
  rounded_D_upper = 3.3 ∧ rounded_D_lower = 3.3 → rounded_D_upper = 3.3 :=
by sorry

end most_precise_value_l98_98001


namespace equilateral_triangle_side_length_l98_98473

theorem equilateral_triangle_side_length (P M N O : ℝ) (XY YZ ZX : ℝ) (h1 : P⊥XY) (h2 : P⊥YZ) (h3 : P⊥ZX) (PM PN PO : ℝ) (hPM : PM = 2) (hPN : PN = 4) (hPO : PO = 6) :
  XY = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_side_length_l98_98473


namespace find_a7_l98_98267

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98267


namespace unique_polynomial_satisfying_conditions_l98_98657

theorem unique_polynomial_satisfying_conditions (P : ℝ → ℝ) :
  (P 0 = 1) ∧ 
  (∀ x y : ℝ, |y^2 - P x| ≤ 2 * |x| ↔ |x^2 - P y| ≤ 2 * |y|) ↔ 
  (P = λ x, x^2 + 1) := by
  sorry

end unique_polynomial_satisfying_conditions_l98_98657


namespace bobby_payment_l98_98631

theorem bobby_payment :
  let mold_cost := 250
  let labor_cost_per_hour := 75
  let hours := 8
  let discount := 0.80
  let total_labor_cost := labor_cost_per_hour * hours
  let discounted_labor_cost := discount * total_labor_cost
  let total_payment := mold_cost + discounted_labor_cost
  total_payment = 730 :=
by
  let mold_cost := 250
  let labor_cost_per_hour := 75
  let hours := 8
  let discount := 0.80
  let total_labor_cost := labor_cost_per_hour * hours
  let discounted_labor_cost := discount * total_labor_cost
  let total_payment := mold_cost + discounted_labor_cost
  sorry

end bobby_payment_l98_98631


namespace pqr_value_l98_98441

noncomputable def pqr_sum (p q r : ℝ) : ℝ :=
  p + q + r

theorem pqr_value
  (p q r : ℝ)
  (h1 : ∃ w : ℂ, ∀ z : ℂ, z^3 + p * z^2 + q * z + r = 0 ↔ (z = w + 4 * complex.i ∨ z = w + 10 * complex.i ∨ z = 3 * w - 5)) :
  pqr_sum p q r = -150.5 :=
by sorry

end pqr_value_l98_98441


namespace y_value_when_x_is_20_l98_98800

theorem y_value_when_x_is_20 :
  ∀ (x : ℝ), (∀ m c : ℝ, m = 2.5 → c = 3 → (y = m * x + c) → x = 20 → y = 53) :=
by
  sorry

end y_value_when_x_is_20_l98_98800


namespace man_is_older_by_22_l98_98165

/-- 
Given the present age of the son is 20 years and in two years the man's age will be 
twice the age of his son, prove that the man is 22 years older than his son.
-/
theorem man_is_older_by_22 (S M : ℕ) (h1 : S = 20) (h2 : M + 2 = 2 * (S + 2)) : M - S = 22 :=
by
  sorry  -- Proof will be provided here

end man_is_older_by_22_l98_98165


namespace completing_square_l98_98127

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l98_98127


namespace geom_progression_min_floor_value_l98_98779

theorem geom_progression_min_floor_value
  (a b c k r : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hk : k > 0) (hr : r > 1)
  (h_geom : b = k * r) (h_geom2 : c = k * r ^ 2) :
  (Int.floor ((a + b) / c) + Int.floor ((b + c) / a) + Int.floor ((c + a) / b) = 5) :=
sorry

end geom_progression_min_floor_value_l98_98779


namespace find_a7_l98_98289

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98289


namespace find_abc_l98_98009

noncomputable def problem_conditions (α β γ δ : ℝ)  : Prop :=
  (α ≠ β) ∧ (α ≠ γ) ∧ (α ≠ δ) ∧ (β ≠ γ) ∧ (β ≠ δ) ∧ (γ ≠ δ) ∧
  ({5, 7, 8, 9, 10, 12} = {x | ∃ u v ∈ {α, β, γ, δ}, u ≠ v ∧ x = u + v}) ∧
  ({6, 10, 14, 15, 21, 35} = {x | ∃ u v ∈ {α, β, γ, δ}, u ≠ v ∧ x = u * v}) ∧
  (α + β + γ + δ = 17) ∧
  (α * β * γ * δ = 210)

theorem find_abc (α β γ δ : ℝ) (a b c : ℝ) :
  problem_conditions α β γ δ →
  (5 * α * α + (-a) * α + b = 0) →
  (5 * β * β + (-a) * β + b = 0) →
  (γ * γ + (-b) * γ + c = 0) →
  (δ * δ + (-b) * δ + c = 0) →
  a = 35 ∧ b = 50 ∧ c = 21 :=
by
  intros _ _ _ _ -- Skipping proof with sorry
  sorry

end find_abc_l98_98009


namespace y_order_of_quadratic_l98_98791

theorem y_order_of_quadratic (k : ℝ) (y1 y2 y3 : ℝ) :
  (y1 = (-4)^2 + 4 * (-4) + k) → 
  (y2 = (-1)^2 + 4 * (-1) + k) → 
  (y3 = (1)^2 + 4 * (1) + k) → 
  y2 < y1 ∧ y1 < y3 :=
by
  intro hy1 hy2 hy3
  sorry

end y_order_of_quadratic_l98_98791


namespace completing_square_l98_98131

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l98_98131


namespace harmony_numbers_with_first_digit_2_count_l98_98927

def is_harmony_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (1000 ≤ n ∧ n < 10000) ∧ (a + b + c + d = 6)

noncomputable def count_harmony_numbers_with_first_digit_2 : ℕ :=
  Nat.card { n : ℕ // is_harmony_number n ∧ n / 1000 = 2 }

theorem harmony_numbers_with_first_digit_2_count :
  count_harmony_numbers_with_first_digit_2 = 15 :=
sorry

end harmony_numbers_with_first_digit_2_count_l98_98927


namespace area_of_ABIHFGD_l98_98007

noncomputable def side_length_of_square := 5
def A := (0, 0)
def B := (5, 0)
def C := (5, 5)
def D := (0, 5)
def E := (5, 5)
def F := (10, 5)
def G := (10, 10)
def H := (7.5, 5)
def I := (2.5, 5)

theorem area_of_ABIHFGD :
  let area_square := 25
  let overlap_area := (1 / 2 * side_length_of_square * (side_length_of_square / 2)) +
                      (1 / 2 * (side_length_of_square + (side_length_of_square / 2)) * (side_length_of_square / 2))
  2 * area_square - overlap_area = 34.375 :=
by
  sorry

end area_of_ABIHFGD_l98_98007


namespace find_a7_l98_98287

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98287


namespace find_n_find_constant_term_l98_98010

-- Define the binomial coefficient
def binom (n k : ℕ) := Nat.choose n k

-- Condition: The binomial coefficient of the third term is 35 more than that of the second term
def coeff_condition (n : ℕ) : Prop :=
  binom n 2 - binom n 1 = 35

-- Prove that if the coefficient condition holds, then n = 10
theorem find_n (n : ℕ) (h : coeff_condition n) : n = 10 := sorry

-- General term in the expansion of (x^4 + 1/x)^n
def general_term (n r : ℕ) := binom n r * x ^ (4 * (n - r)) * x ^ (-r)

-- Prove that for the expansion of (x^4 + 1/x)^10, the constant term is 45
theorem find_constant_term : general_term 10 8 = 45 := sorry

end find_n_find_constant_term_l98_98010


namespace proof_min_vertical_segment_length_l98_98896

noncomputable def min_vertical_segment_length : ℝ :=
  min (Real.sqrt (Real.pow (5 / 2) 2 - 15 / 4))
      (-(5 / 4))

theorem proof_min_vertical_segment_length :
  min_vertical_segment_length = 1.25 := by
  sorry

end proof_min_vertical_segment_length_l98_98896


namespace largest_valid_subset_l98_98065

open Set Function

-- Definitions
def valid_subset (S : Set ℕ) := ∀ (m n : ℕ), m ∈ S → n ∈ S → m ≠ n → m + n ∉ S

def candidate_set := {x | 1 ≤ x ∧ x ≤ 1000}

-- Theorem statement
theorem largest_valid_subset :
  ∃ S ⊆ candidate_set, valid_subset S ∧ S.card = 501 :=
sorry

end largest_valid_subset_l98_98065


namespace field_area_is_243_l98_98986

noncomputable def field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : ℝ :=
  w * l

theorem field_area_is_243 (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : field_area w l h1 h2 = 243 :=
  sorry

end field_area_is_243_l98_98986


namespace inscribed_polygon_cosine_l98_98154

theorem inscribed_polygon_cosine :
  ∀ (A B C D E : ℂ) (circle : set ℂ) (r : ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A ∧
    (|A - B| = 4) ∧ (|B - C| = 4) ∧ (|C - D| = 4) ∧ (|D - E| = 4) ∧ (|A - E| = 2) ∧
    (∀ P ∈ {A, B, C, D, E}, P ∈ circle) ∧
    (∀ (P Q : ℂ), P ∈ circle ∧ Q ∈ circle → |P - r| = |Q - r|) →
  (1 - complex.cos (angle (B - A) (B - C))) * (1 - complex.cos (angle (A - C) (E - C))) = 1 / 16 := sorry

end inscribed_polygon_cosine_l98_98154


namespace coin_toss_sequences_l98_98003

theorem coin_toss_sequences :
  ∃ (seqs : list (char × char)), list.length seqs = 25 ∧
  (∃ (HH HT TH TT : ℕ), HH = 3 ∧ HT = 5 ∧ TH = 4 ∧ TT = 7 ∧ 
  (seqs.count (λ x, x = ('H','H')) = HH ∧ 
   seqs.count (λ x, x = ('H','T')) = HT ∧ 
   seqs.count (λ x, x = ('T','H')) = TH ∧ 
   seqs.count (λ x, x = ('T','T')) = TT)) ∧ 
  list.length (list.filter (λ x, x = ('H','T') ∨ x = ('T','H')) seqs) + 1 - 1 = 
  10 ∧ (nat.choose (5 + 3 - 1) 2) * (nat.choose (10 + 4 - 1) 3) = 1200 :=
sorry

end coin_toss_sequences_l98_98003


namespace bobby_shoes_cost_l98_98633

theorem bobby_shoes_cost :
  let mold_cost := 250
  let hourly_rate := 75
  let hours_worked := 8
  let discount_rate := 0.20
  let labor_cost := hourly_rate * hours_worked
  let discounted_labor_cost := labor_cost * (1 - discount_rate)
  let total_cost := mold_cost + discounted_labor_cost
  mold_cost = 250 ∧ hourly_rate = 75 ∧ hours_worked = 8 ∧ discount_rate = 0.20 →
  total_cost = 730 := 
by
  sorry

end bobby_shoes_cost_l98_98633


namespace interest_rate_is_20_percent_l98_98956

theorem interest_rate_is_20_percent (P A : ℝ) (t : ℝ) (r : ℝ) 
  (h1 : P = 500) (h2 : A = 1000) (h3 : t = 5) :
  A = P * (1 + r * t) → r = 0.20 :=
by
  intro h
  sorry

end interest_rate_is_20_percent_l98_98956


namespace equidistant_line_l98_98226

theorem equidistant_line (A B C1 C2 C3 : ℝ) (h_parallel : ∀ (x y : ℝ), A * x + B * y = C1 ∧ A * x + B * y = C2 ∧ A * x + B * y = C3) :
  ∃ (C_m : ℝ), C_m = (C1 + 2 * C2 + C3) / 4 ∧ ∀ (x y : ℝ), A * x + B * y = C_m := 
by 
  existsi (C1 + 2 * C2 + C3) / 4
  split
  · refl
  · sorry

end equidistant_line_l98_98226


namespace jackson_savings_per_paycheck_l98_98426

-- Jackson wants to save $3,000
def total_savings : ℝ := 3000

-- The vacation is 15 months away
def months_to_save : ℝ := 15

-- Jackson is paid twice a month
def pay_times_per_month : ℝ := 2

-- Jackson's required savings per paycheck to have $3,000 saved in 15 months
theorem jackson_savings_per_paycheck :
  (total_savings / (months_to_save * pay_times_per_month)) = 100 :=
by simp [total_savings, months_to_save, pay_times_per_month]; norm_num; sorry

end jackson_savings_per_paycheck_l98_98426


namespace Maria_coffee_order_l98_98459

variable (visits_per_day : ℕ) (cups_per_visit : ℕ)

theorem Maria_coffee_order (h1 : visits_per_day = 2) (h2 : cups_per_visit = 3) :
  (visits_per_day * cups_per_visit) = 6 := by
  rw [h1, h2]
  exact rfl

end Maria_coffee_order_l98_98459


namespace sector_area_proof_l98_98169

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_proof
  (r : ℝ) (l : ℝ) (perimeter : ℝ) (theta : ℝ) (h1 : perimeter = 2 * r + l)
  (h2 : l = r * theta) (h3 : perimeter = 16) (h4 : theta = 2) :
  sector_area r theta = 16 := by
  sorry

end sector_area_proof_l98_98169


namespace graph_passes_through_fixed_point_l98_98346

theorem graph_passes_through_fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, 8) ∧ ∀ x, f x = 7 + a^(x-1) :=
sorry

end graph_passes_through_fixed_point_l98_98346


namespace land_purchase_price_l98_98465

variable (total_acres : ℕ) (sale_price_per_acre profit : ℝ)

theorem land_purchase_price :
  total_acres = 200 →
  sale_price_per_acre = 200 →
  profit = 6000 →
  let half_acres := total_acres / 2 in
  let revenue := half_acres * sale_price_per_acre in
  let cost := revenue - profit in
  (cost / half_acres) = 140 :=
by
  intros h1 h2 h3
  let half_acres : ℝ := total_acres / 2
  let revenue := half_acres * sale_price_per_acre
  let cost := revenue - profit
  have : (cost / half_acres = 140), sorry
  exact this

end land_purchase_price_l98_98465


namespace train_B_time_to_destination_after_meeting_A_l98_98925

def train_times_and_speeds := sorry

-- Define the main proof problem
theorem train_B_time_to_destination_after_meeting_A
  (v_A : ℝ)  -- speed of train A
  (t_A : ℝ)  -- time taken by train A to its destination after meeting train B
  (v_B: ℝ)  -- speed of train B
  (t_B: ℝ)  -- time taken by train B to its destination after meeting train A
  (h1: v_A = 100)  -- given condition 1
  (h2: t_A = 9)    -- given condition 2
  (h3: v_B = 150)  -- given condition 3
  : t_B = 9 := sorry

end train_B_time_to_destination_after_meeting_A_l98_98925


namespace application_sum_l98_98856

theorem application_sum (a b : ℝ) (h : a * real.sqrt (7 / b) = real.sqrt (a + 7 / b)) : a + b = 55 :=
sorry

end application_sum_l98_98856


namespace find_a7_l98_98290

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98290


namespace average_and_differences_l98_98545

def varsityEnrollment : ℕ := 1300
def northwestEnrollment : ℕ := 1500
def centralEnrollment : ℕ := 1800
def greenbriarEnrollment : ℕ := 1600

def averageEnrollment (v n c g : ℕ) : ℕ := (v + n + c + g) / 4

def positiveDifference (enrollment average : ℕ) : ℕ :=
  if enrollment > average then enrollment - average else average - enrollment

theorem average_and_differences :
  let avg := averageEnrollment varsityEnrollment northwestEnrollment centralEnrollment greenbriarEnrollment in
  avg = 1550 ∧
  positiveDifference varsityEnrollment avg = 250 ∧
  positiveDifference northwestEnrollment avg = 50 ∧
  positiveDifference centralEnrollment avg = 250 ∧
  positiveDifference greenbriarEnrollment avg = 50 :=
by
  sorry

end average_and_differences_l98_98545


namespace sum_of_special_integers_l98_98832

theorem sum_of_special_integers :
  let a := 0
  let b := 1
  let c := -1
  a + b + c = 0 := by
  sorry

end sum_of_special_integers_l98_98832


namespace find_t_l98_98375

def vector := (ℝ × ℝ)

def a : vector := (1, -1)
def b : vector := (6, -4)

def perpendicular {v1 v2 : vector} : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def scalar_mult (t : ℝ) (v : vector) : vector :=
  (t * v.1, t * v.2)

def vector_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem find_t (t : ℝ) :
  perpendicular a (vector_add (scalar_mult t a) b) ↔ t = -5 :=
by sorry

end find_t_l98_98375


namespace ellipse_eqn_slope_ab_eq_l98_98702

noncomputable def ellipse_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def point_on_ellipse (a b x y : ℝ) : Prop :=
  ellipse_equation a b x y
  
noncomputable def point_P := (1, -3/2 : ℝ)
noncomputable def a := 2
noncomputable def b := real.sqrt 3

theorem ellipse_eqn : epid prop :=
  let a := 2
  let b := sqrt 3
  ellipse_equation 2 (real.sqrt 3) x y =
    (x^2 / 4) + (y^2 / 3) = ellipse_equation
      
noncomputable def slope_of_line_ab : ℝ := -1 / 2

syntax "theorem" ident " : " term " := by " " " " sorry" : command

theorem slope_ab_eq : slope_of_line_ab = -1/2 := by sorry

end ellipse_eqn_slope_ab_eq_l98_98702


namespace no_intersection_points_l98_98228

theorem no_intersection_points : ¬ ∃ x y : ℝ, y = x ∧ y = x - 2 := by
  sorry

end no_intersection_points_l98_98228


namespace circle_equation_l98_98347

/-- Given that point C is above the x-axis and
    the circle C with center C is tangent to the x-axis at point A(1,0) and
    intersects with circle O: x² + y² = 4 at points P and Q such that
    the length of PQ is sqrt(14)/2, the standard equation of circle C
    is (x - 1)² + (y - 1)² = 1. -/
theorem circle_equation {C : ℝ × ℝ} (hC : C.2 > 0) (tangent_at_A : C = (1, C.2))
  (intersect_with_O : ∃ P Q : ℝ × ℝ, (P ≠ Q) ∧ (P.1 ^ 2 + P.2 ^ 2 = 4) ∧ 
  (Q.1 ^ 2 + Q.2 ^ 2 = 4) ∧ ((P.1 - 1)^2 + (P.2 - C.2)^2 = C.2^2) ∧ 
  ((Q.1 - 1)^2 + (Q.2 - C.2)^2 = C.2^2) ∧ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4)) :
  (C.2 = 1) ∧ ((x - 1)^2 + (y - 1)^2 = 1) :=
by
  sorry

end circle_equation_l98_98347


namespace find_a7_l98_98288

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7_l98_98288


namespace convert_point_to_polar_l98_98647

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2),
      θ := if y ≠ 0 then real.atan (y / x) else if x > 0 then 0 else real.pi in
  (r, if θ < 0 then θ + 2 * real.pi else θ)

theorem convert_point_to_polar :
  rectangular_to_polar 3 (-3) = (3 * real.sqrt 2, 7 * real.pi / 4) :=
by sorry

end convert_point_to_polar_l98_98647


namespace completing_square_l98_98132

theorem completing_square (x : ℝ) : (x^2 - 2 * x = 2) → ((x - 1)^2 = 3) :=
by
  sorry

end completing_square_l98_98132


namespace trigonometric_identity_proof_l98_98489

theorem trigonometric_identity_proof 
  (α β γ : ℝ) (a b c : ℝ)
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : 0 < γ ∧ γ < π)
  (hc : 0 < c)
  (hb : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (ha : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 :=
by
  sorry

end trigonometric_identity_proof_l98_98489


namespace find_a7_l98_98270

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98270


namespace Carmichael_family_children_count_l98_98500

-- Define the problem setup
def family_average_age (father_age children_total_age number_of_children : ℕ) : Prop :=
  (45 + father_age + children_total_age) / (2 + number_of_children) = 25

def father_and_children_average_age (father_age children_total_age number_of_children : ℕ) : Prop :=
  (father_age + children_total_age) / (1 + number_of_children) = 20

theorem Carmichael_family_children_count :
  ∃ (father_age children_total_age number_of_children : ℕ),
    family_average_age father_age children_total_age number_of_children ∧
    father_and_children_average_age father_age children_total_age number_of_children ∧
    number_of_children = 3 :=
begin
  sorry -- proof is omitted
end

end Carmichael_family_children_count_l98_98500


namespace average_weight_girls_combined_l98_98002

variable (nA_boys nA_students nB_boys nB_students n_girls_in_total : ℕ)
variable (avg_weight_boysA avg_weight_studentsA avg_weight_boysB avg_weight_studentsB : ℝ)

-- Given conditions
def classA_boys_weight := nA_boys * avg_weight_boysA
def classA_total_weight := nA_students * avg_weight_studentsA
def classB_boys_weight := nB_boys * avg_weight_boysB
def classB_total_weight := nB_students * avg_weight_studentsB

def total_weight_girls_classA := classA_total_weight - classA_boys_weight
def total_girls_A := nA_students - nA_boys
def total_weight_girls_classB := classB_total_weight - classB_boys_weight
def total_girls_B := nB_students - nB_boys

def total_weight_girls_all := total_weight_girls_classA + total_weight_girls_classB

-- Prove that the average weight of all girls in both classes is 39.65 kgs
theorem average_weight_girls_combined 
  (h1 : nA_boys = 15) (h2 : nA_students = 25)
  (h3 : nB_boys = 12) (h4 : nB_students = 22)
  (h5 : n_girls_in_total = 20)
  (h6 : avg_weight_boysA = 48) (h7 : avg_weight_studentsA = 45)
  (h8 : avg_weight_boysB = 52) (h9 : avg_weight_studentsB = 46) :
  (total_weight_girls_all / n_girls_in_total) = 39.65 := sorry

end average_weight_girls_combined_l98_98002


namespace sin_alpha_of_terminal_side_l98_98387

theorem sin_alpha_of_terminal_side :
  (∃ α : ℝ, (sin (5 * π / 6) = sin α) ∧ (cos (5 * π / 6) = cos α)) →
  sin (5 * π / 6) = - √3 / 2 :=
by
  intro h,
  have h1 := Real.sin_eq_sin_of_cos_eq_cos h,
  sorry

end sin_alpha_of_terminal_side_l98_98387


namespace gain_percent_is_33_33_l98_98784
noncomputable def gain_percent_calculation (C S : ℝ) := ((S - C) / C) * 100

theorem gain_percent_is_33_33
  (C S : ℝ)
  (h : 75 * C = 56.25 * S) :
  gain_percent_calculation C S = 33.33 := by
  sorry

end gain_percent_is_33_33_l98_98784


namespace fred_balloons_l98_98690

variable (initial_balloons : ℕ := 709)
variable (balloons_given : ℕ := 221)
variable (remaining_balloons : ℕ := 488)

theorem fred_balloons :
  initial_balloons - balloons_given = remaining_balloons :=
  by
    sorry

end fred_balloons_l98_98690


namespace value_of_a_minus_2b_l98_98381

theorem value_of_a_minus_2b 
  (a b : ℚ) 
  (h : ∀ y : ℚ, y > 0 → y ≠ 2 → y ≠ -3 → (a / (y-2) + b / (y+3) = (2 * y + 5) / ((y-2)*(y+3)))) 
  : a - 2 * b = 7 / 5 :=
sorry

end value_of_a_minus_2b_l98_98381


namespace max_possible_scores_l98_98981

theorem max_possible_scores (num_questions : ℕ) (points_correct : ℤ) (points_incorrect : ℤ) (points_unanswered : ℤ) :
  num_questions = 10 →
  points_correct = 4 →
  points_incorrect = -1 →
  points_unanswered = 0 →
  ∃ n, n = 45 :=
by
  sorry

end max_possible_scores_l98_98981


namespace area_of_L_shape_l98_98192

theorem area_of_L_shape (a : ℝ) (h_pos : a > 0) (h_eq : 4 * ((a + 3)^2 - a^2) = 5 * a^2) : 
  (a + 3)^2 - a^2 = 45 :=
by
  sorry

end area_of_L_shape_l98_98192


namespace fibonacci_triangle_area_l98_98498

def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| n := fib (n - 1) + fib (n - 2)

-- Function to compute the area of the triangle given the sides
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  if a + b > c ∧ a + c > b ∧ b + c > a then
    let s := (a + b + c) / 2 in
    real.sqrt (s * (s - a) * (s - b) * (s - c))
  else
    0

theorem fibonacci_triangle_area (n : ℕ) : triangle_area (real.sqrt (fib (2 * n + 1))) (real.sqrt (fib (2 * n + 2))) (real.sqrt (fib (2 * n + 3))) = 1 / 2 :=
by
  sorry

end fibonacci_triangle_area_l98_98498


namespace volume_rectangular_prism_l98_98911

def volume_of_prism : ℚ := 
let width := (22 / 7) in
let height := 2 * width in
let length := 4 * width in
(height * width * length)

theorem volume_rectangular_prism (sum_of_edges : ℚ) (height_eq_two_width : ℚ) (length_eq_four_width : ℚ) :
  sum_of_edges = 88 ∧ height_eq_two_width = 2 ∧ length_eq_four_width = 4 →
  volume_of_prism = 85184 / 343 := by
  sorry

end volume_rectangular_prism_l98_98911


namespace coeff_matrix_correct_l98_98747

-- Define the system of linear equations as given conditions
def eq1 (x y : ℝ) : Prop := 2 * x + 3 * y = 1
def eq2 (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the coefficient matrix
def coeffMatrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 3],
  ![1, -2]
]

-- The theorem stating that the coefficient matrix of the system is as defined
theorem coeff_matrix_correct (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : 
  coeffMatrix = ![
    ![2, 3],
    ![1, -2]
  ] :=
sorry

end coeff_matrix_correct_l98_98747


namespace find_number_l98_98780

theorem find_number (x : ℤ) : 45 - (28 - (x - (15 - 16))) = 55 ↔ x = 37 :=
by
  sorry

end find_number_l98_98780


namespace area_of_enclosed_region_l98_98198

theorem area_of_enclosed_region :
  let region := { p : ℝ × ℝ | abs (2 * p.1 + p.2) + abs (p.1 - 2 * p.2) ≤ 6 }
  area region = 6 * real.sqrt 3 :=
by sorry

end area_of_enclosed_region_l98_98198


namespace base_7_sum_of_product_l98_98905

-- Definitions of the numbers in base-10 for base-7 numbers
def base_7_to_base_10 (d1 d0 : ℕ) : ℕ := d1 * 7 + d0

def sum_digits_base_7 (n : ℕ) : ℕ := 
  let d2 := n / 343
  let r2 := n % 343
  let d1 := r2 / 49
  let r1 := r2 % 49
  let d0 := r1 / 7 + r1 % 7
  d2 + d1 + d0

def convert_10_to_7 (n : ℕ) : ℕ := 
  let d1 := n / 7
  let r1 := n % 7
  d1 * 10 + r1

theorem base_7_sum_of_product : 
  let n36  := base_7_to_base_10 3 6
  let n52  := base_7_to_base_10 5 2
  let nadd := base_7_to_base_10 2 0
  let prod := n36 * n52
  let suma := prod + nadd
  convert_10_to_7 (sum_digits_base_7 suma) = 23 :=
by
  sorry

end base_7_sum_of_product_l98_98905


namespace friends_reach_destinations_l98_98889

structure Journey :=
  (distance_km : ℕ)
  (friends_maykop : ℕ)
  (friends_belorechensk : ℕ)
  (bicycle_initial : bool)
  (walking_speed_kmh : ℕ)
  (cycling_speed_kmh : ℕ)
  (bicycle_unattended : bool)
  (joint_riding_not_allowed : bool)

def can_reach_destinations (j : Journey) (time_limit_hours : ℚ) : Prop :=
  ∃ (time_spent_hours : ℚ),
    time_spent_hours ≤ time_limit_hours 
    -- Additional conditions relating the journey parameters and time.

theorem friends_reach_destinations :
  ∀ (j : Journey),
    j.distance_km = 24 →
    j.friends_maykop = 2 →
    j.friends_belorechensk = 1 →
    j.bicycle_initial →
    j.walking_speed_kmh = 6 →
    j.cycling_speed_kmh = 18 →
    ¬ j.bicycle_unattended →
    j.joint_riding_not_allowed →
    can_reach_destinations j (2 + 40 / 60) :=
by { sorry }

end friends_reach_destinations_l98_98889


namespace even_heads_probability_fair_even_heads_probability_biased_l98_98241

variable  {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1)

/-- The probability of getting an even number of heads -/

/-- Case 1: Fair coin (p = 1/2) -/
theorem even_heads_probability_fair (n : ℕ) : 
  let p : ℝ := 1/2 in 
  let q : ℝ := 1 - p in 
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 0.5 := sorry

/-- Case 2: Biased coin -/
theorem even_heads_probability_biased {n : ℕ} (p : ℝ) (H : 0 < p ∧ p < 1) : 
  let q : ℝ := 1 - p in
  (∑ k in (finset.range (n + 1)).filter (even), nat.choose n k * (p^k) * (q^(n-k))) = 
  (1 + (1 - 2 * p)^n) / 2 := sorry

end even_heads_probability_fair_even_heads_probability_biased_l98_98241


namespace math_problem_l98_98833

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log 2 (1 + a * 2^x + 4^x)

def part1 (a : ℝ) : Prop :=
  f a 2 = f a (-1) + 4 ∧ a = -3 / 4

def part2 (a : ℝ) : Prop :=
  (∀ x : ℝ, x ≥ 1 → f a x ≥ x - 1) → a ≥ -2

theorem math_problem : ∃ a : ℝ, part1 a ∧ part2 a :=
sorry

end math_problem_l98_98833


namespace faulty_meter_shortfall_l98_98992

def cost_price := 1 -- represents selling at cost price
def profit_percent := 4.166666666666666 -- profit percent mentioned

-- Assuming shopkeeper should provide 1000 grams (hypothetical weight)
def actual_weight := 1000

-- Define the faulty weight computation based on the given formula
def faulty_weight (actual_weight : Real) (profit_percent : Real) : Real :=
  actual_weight / (1 + (profit_percent / 100))

-- Calculate the faulty weight with given actual weight and profit percent
def faulty_weight_value := faulty_weight actual_weight profit_percent

-- Expected faulty weight
def expected_faulty_weight := actual_weight - 40

-- Proof statement that the faulty weight is indeed 40 grams short
theorem faulty_meter_shortfall :
  faulty_weight_value = expected_faulty_weight :=
by
  sorry

end faulty_meter_shortfall_l98_98992


namespace a37_b37_sum_l98_98748

-- Declare the sequences as functions from natural numbers to real numbers
variables {a b : ℕ → ℝ}

-- State the hypotheses based on the conditions
variables (h1 : ∀ n, a (n + 1) = a n + a 2 - a 1)
variables (h2 : ∀ n, b (n + 1) = b n + b 2 - b 1)
variables (h3 : a 1 = 25)
variables (h4 : b 1 = 75)
variables (h5 : a 2 + b 2 = 100)

-- State the theorem to be proved
theorem a37_b37_sum : a 37 + b 37 = 100 := 
by 
  sorry

end a37_b37_sum_l98_98748


namespace state_b_selection_percentage_l98_98395

theorem state_b_selection_percentage
  (candidates_A candidates_B selected_A: ℕ) 
  (h_candidates_A_eq: candidates_A = 8400)
  (h_candidates_B_eq: candidates_B = 8400)
  (h_selected_A_perc: selected_A = 6 * candidates_A / 100)
  (additional_selected_B: ℕ)
  (h_additional_selected_B: additional_selected_B = 84) :
  (selected_B_percentage: ℕ) 
  (h_selected_B_eq: selected_B_percentage * candidates_B / 100 = selected_A + additional_selected_B) :
  selected_B_percentage = 7 :=
by
  sorry

end state_b_selection_percentage_l98_98395


namespace find_a7_l98_98282

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98282


namespace find_b_squared_l98_98713

-- Definitions and Conditions
def F1 := (-c, 0)
def F2 := (c, 0)
def ellipse (x y : ℝ) (b : ℝ) : Prop := x ^ 2 + (y ^ 2) / (b ^ 2) = 1
def on_ellipse (x y : ℝ) (b : ℝ) := ellipse x y b

-- Given assumptions
variable (c b : ℝ)
variable (h_ellipse : 0 < b ∧ b < 1)
variable (A B : ℝ × ℝ)
variable (h_A : A = (c, b ^ 2))
variable (h_B : B = (- (5 * c / 3), -(b ^ 2 / 3)))
variable (h_A_el : on_ellipse A.1 A.2 b)
variable (h_B_el : on_ellipse B.1 B.2 b)
variable (h_relation : ∥(fst A - F1.1, snd A - F1.2)∥ = 3 * ∥(fst B - F1.1, snd B - F1.2)∥)
variable (h_foci : 1 = b ^ 2 + c ^ 2)

-- Proof statement
theorem find_b_squared : b ^ 2 = 2 / 3 :=
sorry

end find_b_squared_l98_98713


namespace geometric_seq_a7_l98_98310

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end geometric_seq_a7_l98_98310


namespace sum_projections_l98_98017

universe u

variables {α : Type u} [Nonempty α] [MetricSpace α]

structure Triangle (α : Type u) [MetricSpace α] :=
(A B C : α)
(AB AC BC : ℝ)
(AB_pos : 0 < AB)
(AC_pos : 0 < AC)
(BC_pos : 0 < BC)
(equal_ab : dist A B = AB)
(equal_ac : dist A C = AC)
(equal_bc : dist B C = BC)

def centroid (A B C : α) : α := sorry -- centroid calculation goes here

def projection (P Q R : α) (G : α) : ℝ := sorry -- projection calculation goes here

theorem sum_projections {A B C: α} (t : Triangle α)
  (G : α) (P : α := projection t.B t.C G)
  (Q : α := projection t.A t.C G)
  (R : α := projection t.A t.B G) :
  t.AB = 4 → t.AC = 6 → t.BC = 5 → (projection t.B t.C G) + (projection t.A t.C G) + (projection t.A t.B G) = (5 * real.sqrt 7) / 7 :=
by
  sorry

end sum_projections_l98_98017


namespace rohan_food_expense_percentage_l98_98866

variables (house_rent entertainment conveyance savings salary : ℕ)
  (h₁ : house_rent = 20)
  (h₂ : entertainment = 10)
  (h₃ : conveyance = 10)
  (h₄ : savings = 1000)
  (h₅ : salary = 5000)

theorem rohan_food_expense_percentage : 
  100 - (house_rent + entertainment + conveyance + (savings * 100 / salary)) = 40 :=
by
  rw [h₁, h₂, h₃, h₄, h₅]
  norm_num
  sorry

end rohan_food_expense_percentage_l98_98866


namespace sum_common_divisors_36_18_l98_98677

theorem sum_common_divisors_36_18 : 
  ∑ d in (finset.filter (λ x, 36 % x = 0 ∧ 18 % x = 0) (finset.range 37)), d = 39 := 
by
  -- The actual proof would go here
  sorry

end sum_common_divisors_36_18_l98_98677


namespace perimeter_PQRST_l98_98813

-- Define the points P, Q, R, S, T in 2D space
structure Point where
  x : ℝ
  y : ℝ 

-- Define the distances between the points as per the problem
noncomputable def PQ := 5 : ℝ
noncomputable def QR := 5 : ℝ
noncomputable def PT := 6 : ℝ
noncomputable def TS := 8 : ℝ
noncomputable def R0S := Real.sqrt 10

-- Define a function to represent distances between points
def distance (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

-- Define a function to represent the perimeter of polygon PQRST
noncomputable def perimeter (P Q R S T : Point) : ℝ :=
  distance P Q + distance Q R + distance R S + distance S T + distance T P

-- Define points with given conditions
def P : Point := ⟨0, 6⟩
def Q : Point := ⟨5, 6⟩
def R : Point := ⟨5, 1⟩
def S : Point := ⟨8, 0⟩
def T : Point := ⟨0, 0⟩

-- Lean theorem to prove the perimeter of PQRST
theorem perimeter_PQRST : 
  perimeter P Q R S T = 24 + Real.sqrt 10 := 
sorry

end perimeter_PQRST_l98_98813


namespace base_length_of_isosceles_triangle_l98_98883

-- Definitions based on given conditions
def is_isosceles (a b : ℕ) (c : ℕ) :=
a = b ∧ c = c

def side_length : ℕ := 6
def perimeter : ℕ := 20

-- Theorem to prove the base length
theorem base_length_of_isosceles_triangle (b : ℕ) (h1 : 2 * side_length + b = perimeter) :
  b = 8 :=
sorry

end base_length_of_isosceles_triangle_l98_98883


namespace spring_spending_l98_98895

open Real

theorem spring_spending : 
  (spent_by_end_may : ℝ) → (spent_by_beginning_march : ℝ) →
  spent_by_end_may = 3.6 →
  spent_by_beginning_march = 1.2 →
  spent_by_end_may - spent_by_beginning_march = 2.4 :=
by
  intros spent_by_end_may spent_by_beginning_march hem hbm 
  rw [hem, hbm]
  exact (sub_eq_iff_eq_add'.mpr rfl).symm

end spring_spending_l98_98895


namespace triangle_perimeter_l98_98397

theorem triangle_perimeter (x : ℕ) (hx1 : x % 2 = 1) (hx2 : 5 < x) (hx3 : x < 11) : 
  (3 + 8 + x = 18) ∨ (3 + 8 + x = 20) :=
sorry

end triangle_perimeter_l98_98397


namespace ball_cost_l98_98385

def cost_per_ball (total_cost: ℝ) (num_balls: ℕ) : ℝ :=
  total_cost / num_balls

theorem ball_cost (total_cost: ℝ) (num_balls: ℕ) (cost: ℝ)
  (h1 : total_cost = 4.62)
  (h2 : num_balls = 3)
  (h3 : cost = total_cost / num_balls) : cost = 1.54 :=
by
  sorry

end ball_cost_l98_98385


namespace find_a7_l98_98272

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98272


namespace bigger_part_is_45_l98_98156

variable (x y : ℕ)

theorem bigger_part_is_45
  (h1 : x + y = 60)
  (h2 : 10 * x + 22 * y = 780) :
  max x y = 45 := by
  sorry

end bigger_part_is_45_l98_98156


namespace log_a_b_integer_probability_l98_98990

noncomputable def setS := {x | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 18 ∧ x = 3^n}

def count_valid_pairs : ℕ :=
  (finset.range 18).sum (λ x, ((18 / (x+1)).nat_floor - 1))

def total_distinct_pairs : ℕ := (18 * 17) / 2

def probability_log_is_integer : ℚ :=
  count_valid_pairs / total_distinct_pairs

theorem log_a_b_integer_probability :
  probability_log_is_integer = 40 / 153 :=
sorry

end log_a_b_integer_probability_l98_98990


namespace least_clock_equivalent_l98_98469

theorem least_clock_equivalent (t : ℕ) (h : t > 5) : 
  (t^2 - t) % 24 = 0 → t = 9 :=
by
  sorry

end least_clock_equivalent_l98_98469


namespace math_problem_l98_98562

noncomputable def correct_option : Prop := (sqrt 4)^2 = 4
noncomputable def incorrect_option_A : Prop := sqrt 4 ≠ 2 ∧ sqrt 4 ≠ -2
noncomputable def incorrect_option_C : Prop := sqrt ((-4)^2) ≠ -4
noncomputable def incorrect_option_D : Prop := (- (sqrt 4))^2 ≠ -4

theorem math_problem : correct_option ∧ incorrect_option_A ∧ incorrect_option_C ∧ incorrect_option_D :=
by {
  sorry
}

end math_problem_l98_98562


namespace exists_natural_number_k_to_lie_in_octahedron_l98_98798

theorem exists_natural_number_k_to_lie_in_octahedron 
  (x0 y0 z0 : ℚ) 
  (h : ∀ n : ℤ, x0 + y0 + z0 ≠ n ∧ x0 + y0 - z0 ≠ n ∧ x0 - y0 + z0 ≠ n ∧ x0 - y0 - z0 ≠ n) :
  ∃ k : ℕ, 
    let kx := k * x0,
        ky := k * y0,
        kz := k * z0
    in
    ∃ octahedron : set (ℚ × ℚ × ℚ),
      (kx, ky, kz) ∈ octahedron :=
sorry

end exists_natural_number_k_to_lie_in_octahedron_l98_98798


namespace coeff_x4_expansion_eq_seven_l98_98783

-- Define the problem and conditions. Assume n is a natural number.
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial expansion term function
def binomTerm (n r : ℕ) (x : ℝ) : ℝ :=
  (binom n r) * (x^(n-r)) * ((1 / (2*x))^r)

-- The given condition that the first three coefficients form an arithmetic sequence
def arithmetic_condition (n : ℕ) : Prop :=
  binom n 0 + (1/4) * binom n 2 = 2 * (binom n 1) * (1/2)

-- Main theorem we want to prove in Lean
theorem coeff_x4_expansion_eq_seven (n : ℕ) (hn : arithmetic_condition n) :
  let term := binomTerm n 2 1 in
  term = 7 :=
by
  sorry

end coeff_x4_expansion_eq_seven_l98_98783


namespace rightmost_two_digits_l98_98550

theorem rightmost_two_digits (a b c : ℕ) (h1 : a % 100 = 84) (h2 : b % 100 = 25) (h3 : c % 100 = 43) :
    ((a + b + c) % 100 = 52) :=
by
  have h_sum : (a + b + c) % 100 = (84 + 25 + 43) % 100 := by
    rw [h1, h2, h3]
  norm_num at h_sum
  exact h_sum

end rightmost_two_digits_l98_98550


namespace sum_of_first_10_terms_l98_98448

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = d

theorem sum_of_first_10_terms (h_seq : arithmetic_sequence a d) (h_d_nonzero : d ≠ 0)
  (h_eq : (a 4) ^ 2 + (a 5) ^ 2 = (a 6) ^ 2 + (a 7) ^ 2) :
  (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) * a 4 = 0 :=
by
  sorry

end sum_of_first_10_terms_l98_98448


namespace inequality_1_inequality_2_l98_98322

variable (a b : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_of_cubes_eq_two : a^3 + b^3 = 2

-- Question 1
theorem inequality_1 : (a + b) * (a^5 + b^5) ≥ 4 :=
by
  sorry

-- Question 2
theorem inequality_2 : a + b ≤ 2 :=
by
  sorry

end inequality_1_inequality_2_l98_98322


namespace triangle_inequality_cos2_l98_98393

variable {a b c A B C : ℝ}
variable {triangle_inequality : a ≤ b ∧ b ≤ c}

theorem triangle_inequality_cos2
  (h_triangle : (triangle_inequality : a ≤ b) ∧ (triangle_inequality : b ≤ c))
  (h_a : a = (A ≠ 0))
  (h_b : b = (B ≠ 0))
  (h_c : c = (C ≠ 0)) :
  2 * (cos (C / 2))^2 ≤ (a / (b + c) + b / (c + a) + c / (a + b)) ∧ (a / (b + c) + b / (c + a) + c / (a + b)) ≤ 2 * (cos (A / 2))^2 :=
by sorry

end triangle_inequality_cos2_l98_98393


namespace sin_C_eq_triangle_area_eq_l98_98018

variables {A B C : ℝ} {a b c : ℝ}

/-- In triangle ABC, if b = c (2 * sin A + cos A), then sin C = √5 / 5 --/
theorem sin_C_eq : (h : b = c * (2 * Real.sin A + Real.cos A)) -> Real.sin C = Real.sqrt 5 / 5 :=
sorry

/-- Given a = √2, B = 3π/4, and sin C = √5 / 5, the area of triangle ABC is 1 --/
theorem triangle_area_eq (ha : a = Real.sqrt 2) (hB : B = 3 * Real.pi / 4) (hC : Real.sin C = Real.sqrt 5 / 5) :
  let area := 1/2 * a * c * Real.sin B in area = 1 :=
sorry

end sin_C_eq_triangle_area_eq_l98_98018


namespace find_a7_l98_98301

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98301


namespace problem_statement_l98_98703

-- Given that f(x) is an even function.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Definition of the main condition f(x) + f(2 - x) = 0.
def special_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 0

-- Theorem: Given the conditions, show that f(x) has a period of 4 and f(x-1) is odd.
theorem problem_statement {f : ℝ → ℝ} (h_even : is_even f) (h_cond : special_condition f) :
  (∀ x, f (4 + x) = f x) ∧ (∀ x, f (-x - 1) = -f (x - 1)) :=
by
  sorry

end problem_statement_l98_98703


namespace length_of_CB_l98_98392

noncomputable def length_CB (CD DA CF : ℕ) (DF_parallel_AB : Prop) := 9 * (CD + DA) / CD

theorem length_of_CB {CD DA CF : ℕ} (DF_parallel_AB : Prop):
  CD = 3 → DA = 12 → CF = 9 → CB = 9 * 5 := by
  sorry

end length_of_CB_l98_98392


namespace village_population_l98_98398

theorem village_population (initial_population: ℕ) (died_percent left_percent: ℕ) (remaining_population current_population: ℕ)
    (h1: initial_population = 6324)
    (h2: died_percent = 10)
    (h3: left_percent = 20)
    (h4: remaining_population = initial_population - (initial_population * died_percent / 100))
    (h5: current_population = remaining_population - (remaining_population * left_percent / 100)):
  current_population = 4554 :=
  by
    sorry

end village_population_l98_98398


namespace length_of_platform_l98_98592

noncomputable def train_length : ℝ := 450
noncomputable def signal_pole_time : ℝ := 18
noncomputable def platform_time : ℝ := 39

theorem length_of_platform : 
  ∃ (L : ℝ), 
    (train_length / signal_pole_time = (train_length + L) / platform_time) → 
    L = 525 := 
by
  sorry

end length_of_platform_l98_98592


namespace calculate_f_neg2_l98_98835

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + x + 1 else 0 -- placeholder for definition to avoid unbounded function

lemma f_odd (x : ℝ) : f (-x) = -f x :=
sorry -- given condition that f is odd

lemma f_definition (x : ℝ) (h : x > 0) : f x = x^2 + x + 1 :=
sorry -- given condition that f(x) = x^2 + x + 1 when x > 0

theorem calculate_f_neg2 : f (-2) = -7 :=
by
  have h1 : f 2 = 7, from f_definition 2 (by linarith),
  have h2 : f (-2) = -f 2, from f_odd 2,
  rw [h1, h2],
  norm_num,

end calculate_f_neg2_l98_98835


namespace triangle_probability_l98_98818

/-- In triangle PQR with side lengths PQ = 7, QR = 24, and PR = 25,
the probability that a point S randomly selected inside the triangle
is closer to the midpoint of QR than to either P or R is 7/96. -/
theorem triangle_probability (P Q R S: ℝ → ℝ → Prop) 
  (PQ : dist (P 0 0) (Q 24 0) = 7) 
  (QR : dist (Q 24 0) (R 24 (7:ℝ)) = 24)
  (PR : dist (P 0 0) (R 24 (7:ℝ)) = 25) :
  ∃ M: ℝ → ℝ, (M (24) (3.5)) ∧ 
  ∃ region: ℝ → Prop, (region S) ∧ 
  ((area region) / (area (λ P Q R, triangle P Q R)) = (7 / 96)) :=
sorry

end triangle_probability_l98_98818


namespace find_nm_l98_98345

theorem find_nm (h : 62^2 + 122^2 = 18728) : 
  ∃ (n m : ℕ), (n = 92 ∧ m = 30) ∨ (n = 30 ∧ m = 92) ∧ n^2 + m^2 = 9364 := 
by 
  sorry

end find_nm_l98_98345


namespace find_a7_l98_98271

section GeometricSequence

variables {a : ℕ → ℝ} (q : ℝ) (a1 : ℝ)

-- a_n is defined as a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
∀ n, a n = a1 * q^(n-1)

-- Given conditions
axiom h1 : geometric_sequence a a1 q
axiom h2 : a 2 * a 4 * a 5 = a 3 * a 6
axiom h3 : a 9 * a 10 = -8

--The proof goal
theorem find_a7 : a 7 = -2 :=
sorry

end GeometricSequence

end find_a7_l98_98271


namespace find_a7_l98_98280

noncomputable def seq (n : ℕ) : ℝ

axiom geometric_seq : ∀ n : ℕ, seq (n + 1) = seq n * q

axiom condition1 : seq 2 * seq 4 * seq 5 = seq 3 * seq 6
axiom condition2 : seq 9 * seq 10 = -8

theorem find_a7 : seq 7 = -2 := 
sorry

end find_a7_l98_98280


namespace convert_point_to_polar_l98_98648

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2),
      θ := if y ≠ 0 then real.atan (y / x) else if x > 0 then 0 else real.pi in
  (r, if θ < 0 then θ + 2 * real.pi else θ)

theorem convert_point_to_polar :
  rectangular_to_polar 3 (-3) = (3 * real.sqrt 2, 7 * real.pi / 4) :=
by sorry

end convert_point_to_polar_l98_98648


namespace problem_equivalent_proof_l98_98316

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

def seq_condition1 (a : ℕ → ℝ) (q : ℝ) :=
  a 2 * a 4 * a 5 = a 3 * a 6

def seq_condition2 (a : ℕ → ℝ) :=
  a 9 * a 10 = -8

-- The problem to prove
theorem problem_equivalent_proof :
  is_geometric_sequence a q →
  seq_condition1 a q →
  seq_condition2 a →
  a 7 = -2 :=
by
  sorry

end problem_equivalent_proof_l98_98316


namespace divisible_by_primes_l98_98053

theorem divisible_by_primes (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  (100100 * x + 10010 * y + 1001 * z) % 7 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 11 = 0 ∧ 
  (100100 * x + 10010 * y + 1001 * z) % 13 = 0 := 
by
  sorry

end divisible_by_primes_l98_98053


namespace probability_z_eq_4_probability_beautiful_equation_l98_98184

open ProbabilityTheory

def faces : Finset ℕ := {1, 2, 3, 4}

def throw : Type := (ℕ × ℕ)

def z (b c : ℕ) : ℕ := (b - 3)^2 + (c - 3)^2

def is_beautiful_equation (b c : ℕ) : Prop :=
  ∃ x ∈ {1, 2, 3, 4}, x^2 - b * x - c = 0

theorem probability_z_eq_4 : 
  (∑ (b ∈ faces) (c ∈ faces), if z b c = 4 then 1 else 0) / 
    (faces.card * faces.card) = 1 / 8 :=
sorry

theorem probability_beautiful_equation : 
  (∑ (b ∈ faces) (c ∈ faces), if is_beautiful_equation b c then 1 else 0) /
    (faces.card * faces.card) = 3 / 16 :=
sorry

end probability_z_eq_4_probability_beautiful_equation_l98_98184


namespace parallelogram_side_length_l98_98802

theorem parallelogram_side_length (x y : ℚ) (h1 : 3 * x + 2 = 12) (h2 : 5 * y - 3 = 9) : x + y = 86 / 15 :=
by 
  sorry

end parallelogram_side_length_l98_98802


namespace katya_notebooks_l98_98178

theorem katya_notebooks (rubles: ℕ) (cost_per_notebook: ℕ) (stickers_per_exchange: ℕ) 
  (initial_rubles: ℕ) (initial_notebooks: ℕ) :
  (initial_notebooks = initial_rubles / cost_per_notebook) →
  (rubles = initial_notebooks * cost_per_notebook) →
  (initial_notebooks = 37) →
  (initial_rubles = 150) →
  (cost_per_notebook = 4) →
  (stickers_per_exchange = 5) →
  (rubles = 148) →
  let rec total_notebooks (notebooks stickers : ℕ) : ℕ :=
      if stickers < stickers_per_exchange then notebooks
      else let new_notebooks := stickers / stickers_per_exchange in
           total_notebooks (notebooks + new_notebooks) 
                           (stickers % stickers_per_exchange + new_notebooks) in
  total_notebooks initial_notebooks initial_notebooks = 46 :=
begin
  sorry
end

end katya_notebooks_l98_98178


namespace shaded_to_white_ratio_l98_98121

-- Define the condition for the setup
def vertices_condition (S : Type) [metric_space S] (vertices : ℕ → set S) (n : ℕ) : Prop :=
  ∀ (k < n), ∃ (s ∈ vertices k), s ∈ middle (vertices (k+1))

-- State the problem
theorem shaded_to_white_ratio (S : Type) [metric_space S] (vertices : ℕ → set S) (n : ℕ) :
  vertices_condition S vertices n →
  let shaded_area := 5
  let white_area := 3 in
  shaded_area / white_area = 5 / 3 :=
by
  sorry

end shaded_to_white_ratio_l98_98121


namespace martha_to_doris_ratio_l98_98918

-- Define the amounts involved
def initial_amount : ℕ := 21
def doris_spent : ℕ := 6
def remaining_after_doris : ℕ := initial_amount - doris_spent
def final_amount : ℕ := 12
def martha_spent : ℕ := remaining_after_doris - final_amount

-- State the theorem about the ratio
theorem martha_to_doris_ratio : martha_spent * 2 = doris_spent :=
by
  -- Detailed proof is skipped
  sorry

end martha_to_doris_ratio_l98_98918


namespace chris_money_before_birthday_l98_98639

/-- Chris's total money now is $279 -/
def money_now : ℕ := 279

/-- Money received from Chris's grandmother is $25 -/
def money_grandmother : ℕ := 25

/-- Money received from Chris's aunt and uncle is $20 -/
def money_aunt_uncle : ℕ := 20

/-- Money received from Chris's parents is $75 -/
def money_parents : ℕ := 75

/-- Total money received for his birthday -/
def money_received : ℕ := money_grandmother + money_aunt_uncle + money_parents

/-- Money Chris had before his birthday -/
def money_before_birthday : ℕ := money_now - money_received

theorem chris_money_before_birthday : money_before_birthday = 159 := by
  sorry

end chris_money_before_birthday_l98_98639


namespace slope_angle_of_line_l98_98910

theorem slope_angle_of_line :
  let lineEq := (λ x y : ℝ, x + sqrt 3 * y - 1 = 0)
  let slope := - (sqrt 3 / 3)
  in ∃ α : ℝ, α ∈ set.Ioc 0 180 ∧ tan (α * real.pi / 180) = slope ∧ α = 150 :=
by sorry

end slope_angle_of_line_l98_98910


namespace problem_l98_98411

noncomputable def median (A B C : Point) (A1 : Point) : Prop := Collinear A A1 C ∧ Symmetric B A1

noncomputable def angleBisector (A B C : Point) (A2 : Point) : Prop :=
  let ratio := (dist A2 B) / (dist A2 C)
  ratio = (dist A B) / (dist A C)

noncomputable def parallel (A2 K AC Line : Line) : Prop := 
  let s1 := slope A2 K
  let s2 := slope AC
  s1 = s2

noncomputable def perp (AA2 KC : Line) : Prop := 
  let p1 := slope AA2
  let p2 := slope KC
  p1 * p2 = -1

theorem problem {A B C A1 A2 K : Point} (h1 : median A B C A1) (h2 : angleBisector A B C A2) 
  (h3 : OnLine K (Line A A1)) (h4 : parallel (Line K A2) (Line A C)) :
  perp (Line A A2) (Line K C) :=
sorry

end problem_l98_98411


namespace find_f_2015_l98_98655

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_periodic_2 : ∀ x : ℝ, f x * f (x + 2) = 13
axiom f_at_1 : f 1 = 2

theorem find_f_2015 : f 2015 = 13 / 2 :=
by
  sorry

end find_f_2015_l98_98655


namespace f_prime_at_1_l98_98043

noncomputable def f : ℝ → ℝ := sorry -- We assume f is given but not explicitly defined.

theorem f_prime_at_1 (h1 : ∀ x > 0, differentiable_at ℝ f x)
    (h2 : ∀ x > 0, f (log x) = x + log x) :
    deriv f 1 = 1 + 1 / Real.exp 1 := by
  sorry

end f_prime_at_1_l98_98043


namespace minimum_distance_l98_98358

open Real

-- Define the curve C1
def C1 (t : ℝ) : ℝ × ℝ := (-4 + cos t, 3 + sin t)

-- Define the curve C2
def C2 (θ : ℝ) : ℝ × ℝ := (8 * cos θ, 3 * sin θ)

-- Define the line C3 as a linear equation
def C3 (x y : ℝ) : Prop := x - 2 * y - 7 = 0

-- Define the point P on C1 when t = π/2
def P : ℝ × ℝ := C1 (π / 2)

-- Define the point Q on C2 given by parameter θ
def Q (θ : ℝ) : ℝ × ℝ := C2 θ

-- Define the midpoint M of P and Q
def M (θ : ℝ) : ℝ × ℝ := 
  let p := P
  let q := Q θ
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

-- Define the minimum distance from M to C3
theorem minimum_distance (θ : ℝ) : 
  let d := (sqrt 5 / 5) * (abs (4 * cos θ - 3 * sin θ - 13)) in
  d >= (sqrt 5 / 5) * 8 :=
sorry

end minimum_distance_l98_98358


namespace exponential_equality_l98_98384

theorem exponential_equality (n : ℕ) (h : 4 ^ n = 64 ^ 2) : n = 6 :=
  sorry

end exponential_equality_l98_98384


namespace find_sale_in_third_month_l98_98604

def sales_indices := List 𝕙
def average_sales_per_month := 6500
def num_months :=  # let define the required months

def monthly_sales (sales_indices) :=
  | [6735, 6927, 6855, 7230, 6562, 4691]

def sales_in_3rd_month := 
  monthly_sales(sales_indices).get!(num_months_1, 3)


theorem find_sale_in_third_month
  (sales1 sales2 sales3 sales4 sales5 sales6 total_sales : ℕ)
  (avg_sales_per_month : ℕ)
  (h_sales1 : sales1 = 6735)
  (h_sales2 : sales2 = 6927)
  (h_sales4 : sales4 = 7230)
  (h_sales5 : sales5 = 6562)
  (h_avg_sales : avg_sales_per_month = 6500)
  (h_sales6 : sales6 = 4691)
  (h_total_sales : total_sales = avg_sales_per_month * 6)
  (total_sales = average_sales_per_month * num_months)
  : rs_6855 :=
by
  rw [h_sales1, h_sales2, h_sales3, h_sales4, h_sales5, h_sales6, h_avg_sales, h_total_sales]

)


end find_sale_in_third_month_l98_98604


namespace shirt_and_tie_outfits_l98_98067

theorem shirt_and_tie_outfits (shirts ties : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 7) : shirts * ties = 56 := 
by 
  rw [h_shirts, h_ties]
  exact (mul_comm 8 7).mp (mul_self_eq 8 7) 
  -- Here, I've used mul_comm to reorder the multiplication just as an example.
  -- Sorry - as a placeholder signifying the proof would go here.

end shirt_and_tie_outfits_l98_98067


namespace toms_score_l98_98535

theorem toms_score (T J : ℝ) (h1 : T = J + 30) (h2 : (T + J) / 2 = 90) : T = 105 := by
  sorry

end toms_score_l98_98535


namespace min_tiles_needed_l98_98552

def is_L_shaped (s : set (ℕ × ℕ)) : Prop :=
  (∃ (x y : ℕ), s = {(x, y), (x + 1, y), (x, y + 1)}) ∨
  (∃ (x y : ℕ), s = {(x, y), (x, y + 1), (x + 1, y + 1)}) ∨
  (∃ (x y : ℕ), s = {(x, y), (x + 1, y), (x + 1, y + 1)}) ∨
  (∃ (x y : ℕ), s = {(x, y), (x + 1, y), (x + 1, y + 1)})

def grid_5x5 : finset (ℕ × ℕ) := 
  (finset.range 5).product (finset.range 5)

noncomputable def min_L_shaped_tiles (P : Π s : finset (finset (ℕ × ℕ)), Prop) (t : finset (finset (ℕ × ℕ))) : ℕ :=
  if h : ∃ s, s ∈ t ∧ is_L_shaped s ∧ ∀ s' ∈ t, s' ≠ s → is_L_shaped s' ∧ disjoint s s' ∧
    (s ∪ s' = grid_5x5) ∧ ∀ r (⊆ t), is_L_shaped r ∧ ∃ q ∈ t, q ≠ s ∧ q ≠ r
  then classical.some h else 0

theorem min_tiles_needed : min_L_shaped_tiles is_L_shaped grid_5x5 = 4 :=
sorry

end min_tiles_needed_l98_98552


namespace z_conjugate_product_l98_98726

-- Define the complex number z = 2i / (1 + i)
def z : ℂ := (2 * complex.I) / (1 + complex.I)

-- Prove the statement that z * conjugate(z) = 2
theorem z_conjugate_product : z * conj z = 2 := by
  sorry

end z_conjugate_product_l98_98726


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98238

-- Define the probability of getting heads as p for the biased coin
variable {p : ℝ} (hp : 0 < p ∧ p < 1)

-- Define the number of tosses
variable {n : ℕ}

-- Fair coin probability of getting an even number of heads
theorem fair_coin_even_heads : 
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * (1 / 2)^n else 0) = 0.5 :=
sorry

-- Biased coin probability of getting an even number of heads
theorem biased_coin_even_heads :
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) else 0) 
  = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98238


namespace square_area_and_diagonal_ratio_l98_98512

theorem square_area_and_diagonal_ratio
    (a b : ℕ)
    (h_perimeter : 4 * a = 16 * b) :
    (a = 4 * b) ∧ ((a^2) / (b^2) = 16) ∧ ((a * Real.sqrt 2) / (b * Real.sqrt 2) = 4) :=
  by
  sorry

end square_area_and_diagonal_ratio_l98_98512


namespace consecutive_seating_probability_l98_98109

noncomputable def probability_consecutive_seating : ℚ :=
  (factorial 4 * factorial 4 * factorial 3) / factorial 11

theorem consecutive_seating_probability :
  probability_consecutive_seating = 1 / 5775 :=
by
  sorry

end consecutive_seating_probability_l98_98109


namespace lim_Sn_Sn_max_m_Tn_l98_98822

noncomputable def a : ℕ → ℕ 
| 0 := 3 -- a_1 = 3
| 1 := 5 -- a_2 = 5
| 2 := 7 -- a_3 = 7
| 3 := 9 -- a_4 = 9
| 4 := 11 -- a_5 = 11
| n+5 := (a n)^2 - 2*n*(a n) + 2 -- recursive definition

def b (n : ℕ) : ℕ := 11 - a n
def Sn (n : ℕ) : ℕ := (Finset.range n).sum (λ i, b (i + 1))
def Sn' (n : ℕ) : ℕ := (Finset.range n).sum (λ i, |b (i + 1)|)
noncomputable def C (n : ℕ) : ℚ := 1 / (n * (1 + a n))
def Tn (n : ℕ) : ℚ := (Finset.range n).sum (λ i, C (i + 1))

theorem lim_Sn_Sn' : ∃ l : ℤ, l = -1 ∧ (tendsto (λ n, (Sn n : ℚ) / Sn' n) at_top (𝓝 l)) := sorry

theorem max_m_Tn : ∃ m : ℕ, ∀ n : ℕ, Tn n > m / 32 ∧ m = 7 := sorry

end lim_Sn_Sn_max_m_Tn_l98_98822


namespace f_one_minus_a_l98_98716

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = f x
axiom f_one_plus_a {a : ℝ} : f (1 + a) = 1

theorem f_one_minus_a (a : ℝ) : f (1 - a) = -1 :=
by
  sorry

end f_one_minus_a_l98_98716


namespace workers_planted_33_walnut_trees_l98_98101

def initial_walnut_trees : ℕ := 22
def total_walnut_trees_after_planting : ℕ := 55
def walnut_trees_planted (initial : ℕ) (total : ℕ) : ℕ := total - initial

theorem workers_planted_33_walnut_trees :
  walnut_trees_planted initial_walnut_trees total_walnut_trees_after_planting = 33 :=
by
  unfold walnut_trees_planted
  rfl

end workers_planted_33_walnut_trees_l98_98101


namespace question1_solution_question2_solution_l98_98368

-- Definitions of the problem conditions
def f (x a : ℝ) : ℝ := abs (x - a)

-- First proof problem (Question 1)
theorem question1_solution (x : ℝ) : (f x 2) ≥ (4 - abs (x - 4)) ↔ (x ≥ 5 ∨ x ≤ 1) :=
by sorry

-- Second proof problem (Question 2)
theorem question2_solution (x : ℝ) (a : ℝ) (h_sol : 1 ≤ x ∧ x ≤ 2) 
  (h_ineq : abs (f (2 * x + a) a - 2 * f x a) ≤ 2) : a = 3 :=
by sorry

end question1_solution_question2_solution_l98_98368


namespace part1_intersection_part1_union_complement_part2_range_of_m_l98_98031

open Set

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (2 * m + 1)}

theorem part1_intersection (x : ℝ) : x ∈ (A ∩ B 4) ↔ 3 ≤ x ∧ x ≤ 5 :=
by sorry

theorem part1_union_complement (x : ℝ) : x ∈ (compl A ∪ B 4) ↔ x < 2 ∨ x ≥ 3 :=
by sorry

theorem part2_range_of_m (m : ℝ) : (∀ x, (x ∈ A ↔ x ∈ B m)) ↔ (2 ≤ m ∧ m ≤ 3) :=
by sorry

end part1_intersection_part1_union_complement_part2_range_of_m_l98_98031


namespace find_c_plus_d_l98_98474

-- Conditions as definitions
variables {P A C : Point }
variables {O₁ O₂ : Point}
variables {AB AP CP : ℝ}
variables {c d : ℕ}

-- Given conditions
def Point_on_diagonal (P A C : Point) : Prop := true -- We need to code the detailed properties of being on the diagonal
def circumcenter_of_triangle (P Q R O : Point) : Prop := true -- We need to code the properties of being a circumcenter
def AP_greater_than_CP (AP CP : ℝ) : Prop := AP > CP
def angle_right (A B O : Point) : Prop := true -- Define the right angle property

-- Main statement to prove
theorem find_c_plus_d : 
  Point_on_diagonal P A C ∧
  circumcenter_of_triangle A B P O₁ ∧ 
  circumcenter_of_triangle C D P O₂ ∧ 
  AP_greater_than_CP AP CP ∧
  AB = 10 ∧
  angle_right O₁ P O₂ ∧
  (AP = Real.sqrt c + Real.sqrt d) →
  (c + d = 100) :=
by
  sorry

end find_c_plus_d_l98_98474


namespace train_speed_l98_98622

theorem train_speed 
  (length_train : ℝ) (length_bridge : ℝ) (time : ℝ) 
  (h_length_train : length_train = 110)
  (h_length_bridge : length_bridge = 138)
  (h_time : time = 12.399008079353651) : 
  (length_train + length_bridge) / time * 3.6 = 72 :=
by
  sorry

end train_speed_l98_98622


namespace ryan_learning_schedule_l98_98060

theorem ryan_learning_schedule
  (E1 E2 E3 S1 S2 S3 : ℕ)
  (hE1 : E1 = 7) (hE2 : E2 = 6) (hE3 : E3 = 8)
  (hS1 : S1 = 4) (hS2 : S2 = 5) (hS3 : S3 = 3):
  (E1 + E2 + E3) - (S1 + S2 + S3) = 9 :=
by
  sorry

end ryan_learning_schedule_l98_98060


namespace nancy_clay_pots_l98_98854

theorem nancy_clay_pots : 
  ∃ M : ℕ, (M + 2 * M + 14 = 50) ∧ M = 12 :=
sorry

end nancy_clay_pots_l98_98854


namespace find_b_for_translated_line_l98_98400

theorem find_b_for_translated_line (b : ℝ) :
  (∃ line : ℝ → ℝ, line = λ x, 2 * x + b ∧
    (∃ translated_line : ℝ → ℝ, translated_line = λ x, 2 * x + b - 2 ∧ translated_line 0 = 0)) →
  b = 2 :=
by
  intro h
  obtain ⟨line, hl, ⟨translated_line, htl, htlo⟩⟩ := h
  rw [htl, funext (λ x, rfl)] at htlo
  simp at htlo
  exact htlo.symm

end find_b_for_translated_line_l98_98400


namespace total_alligators_seen_l98_98485

theorem total_alligators_seen (a_samara : ℕ) (n_friends : ℕ) (avg_friends : ℕ) (h_samara : a_samara = 20) (h_friends : avg_friends = 10) (h_n_friends : n_friends = 3) : a_samara + n_friends * avg_friends = 50 := by
    rw [h_samara, h_friends, h_n_friends]
    norm_num
    exact eq.refl 50

end total_alligators_seen_l98_98485


namespace students_on_field_trip_l98_98653

-- Definitions based on the conditions
def van_capacity : ℕ := 4
def adults : ℕ := 6
def vans_needed : ℕ := 2

-- Theorem statement for the number of students
theorem students_on_field_trip : ∃ S : ℕ, (S = vans_needed * van_capacity - adults) :=
begin
  use 2,
  calc 2 = 2 * 4 - 6 : by norm_num,
end

end students_on_field_trip_l98_98653


namespace PQ_tangent_to_tau_iff_AB_eq_AC_l98_98838

theorem PQ_tangent_to_tau_iff_AB_eq_AC 
  (A B C D X P Q : Point) 
  (triangle_ABC : Triangle A B C)
  (D_interior : D ∈ segment B C) 
  (AD_intersects_circumcircle_ABC_at_X : intersects (line_through A D) (circumcircle triangle_ABC) = {X})
  (P_is_foot_from_X_to_AB : is_perpendicular_foot X A B P)
  (Q_is_foot_from_X_to_AC : is_perpendicular_foot X A C Q)
  (tau : Circle := circle_with_diameter X D) :
  (is_tangent PQ tau) ↔ (segment_length A B = segment_length A C) :=
sorry

end PQ_tangent_to_tau_iff_AB_eq_AC_l98_98838


namespace problem_statement_l98_98380

theorem problem_statement (x y z t : ℝ) (h : (x + y) / (y + z) = (z + t) / (t + x)) : x * (z + t + y) = z * (x + y + t) :=
sorry

end problem_statement_l98_98380


namespace relationship_between_a_b_l98_98837

theorem relationship_between_a_b (a b x : ℝ) (h1 : 2 * x = a + b) (h2 : 2 * x^2 = a^2 - b^2) : 
  a = -b ∨ a = 3 * b :=
  sorry

end relationship_between_a_b_l98_98837


namespace probability_top_4_hearts_l98_98993

-- Definitions of cards and suits
def standard_deck : finset (ℕ × string) := sorry -- Assumed details of the deck are implemented.

-- Number of ways to pick the top 4 hearts 
def num_top_4_hearts : ℕ := 13 * 12 * 11 * 10

-- Total number of ways to choose any 4 cards from a shuffled standard deck
def total_possible_hands : ℕ := 52 * 51 * 50 * 49

-- Question: Probability that the top 4 cards are hearts
theorem probability_top_4_hearts : (13 * 12 * 11 * 10 / (52 * 51 * 50 * 49)) = (286 / 108290) :=
by
  -- Expected proof steps; Marked as 'sorry' since proof is not required
  sorry

end probability_top_4_hearts_l98_98993


namespace length_LL1_l98_98399

theorem length_LL1 (XZ : ℝ) (XY : ℝ) (YZ : ℝ) (X1Y : ℝ) (X1Z : ℝ) (LM : ℝ) (LN : ℝ) (MN : ℝ) (L1N : ℝ) (LL1 : ℝ) : 
  XZ = 13 → XY = 5 → 
  YZ = Real.sqrt (XZ^2 - XY^2) → 
  X1Y = 60 / 17 → 
  X1Z = 84 / 17 → 
  LM = X1Z → LN = X1Y → 
  MN = Real.sqrt (LM^2 - LN^2) → 
  (∀ k, L1N = 5 * k ∧ (7 * k + 5 * k) = MN → LL1 = 5 * k) →
  LL1 = 20 / 17 :=
by sorry

end length_LL1_l98_98399


namespace right_handed_total_l98_98948

theorem right_handed_total
  (total_players : ℕ)
  (throwers : ℕ)
  (left_handed_non_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (non_throwers : ℕ)
  (right_handed_non_throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = non_throwers / 3 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  right_handed_throwers = throwers →
  right_handed_throwers + right_handed_non_throwers = 56 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end right_handed_total_l98_98948


namespace ratio_XQ_QY_l98_98502

/-
Conditions:
1. The dodecagon consists of 12 unit squares.
2. Below the line \(\overline{PQ}\) lies a combination of two unit squares and a triangle with a base measuring 6 units.
3. \(\overline{PQ}\) bisects the area of the dodecagon.
-/

noncomputable def dodecagon_area : ℝ := 12
noncomputable def below_PQ_area : ℝ := 2 + (1/2) * 6 * (4/3) -- 2 unit squares + triangle area
noncomputable def PQ_bisection_area : ℝ := 6

-- Hypothesis: \(\overline{PQ}\) divides the area such that widths sum up to the horizontal length 6
axiom XQ_QY_sum_eq_six (XQ QY : ℝ) : XQ + QY = 6
axiom XQ_eq_two_QY (XQ QY : ℝ) : XQ = 2 * QY

theorem ratio_XQ_QY (XQ QY : ℝ) (hXQ_QY_sum : XQ_QY_sum_eq_six XQ QY) (hXQ_twoQY : XQ_eq_two_QY XQ QY) : (XQ / QY) = 2 :=
by
  sorry

end ratio_XQ_QY_l98_98502


namespace train_crossing_time_l98_98902

theorem train_crossing_time :
  ∀ (length_train length_platform speed : ℕ),
    length_train = 450 →
    length_train = length_platform →
    speed = 54 →
    (let speed_mps := speed * 1000 / 3600 in
     let total_distance := length_train + length_platform in
     let time := total_distance / speed_mps in
     time = 60) :=
by
  intros
  assume h1 h2 h3
  let speed_mps := speed * 1000 / 3600
  let total_distance := length_train + length_platform
  let time := total_distance / speed_mps
  exact sorry

end train_crossing_time_l98_98902


namespace proof_problem_l98_98806

variables (Rehana_Phoebe_ratio_in_5_years : ℕ → ℕ)
variables (Rehana_current_age Phoebe_current_age Jacob_current_age Xander_current_age : ℕ)

-- Conditions
def condition1 := ∀ (Rehana_current_age Phoebe_current_age : ℕ), 
  (Rehana_Phoebe_ratio_in_5_years (Rehana_current_age + 5)) = (Rehana_Phoebe_ratio_in_5_years (3 * (Phoebe_current_age + 5)))

def condition2 := Rehana_current_age = 25

def condition3 := Jacob_current_age = (3 * Phoebe_current_age) / 5

def condition4 := Xander_current_age = (Rehana_current_age + Jacob_current_age) - 4

-- Proposition
def proposition1 := Rehana_current_age = 25
def proposition2 := Phoebe_current_age = 5
def proposition3 := Jacob_current_age = 3
def proposition4 := Xander_current_age = 24

-- Ratio calculation
def proposition5 := (Rehana_current_age + Phoebe_current_age + Jacob_current_age) / Xander_current_age = 11 / 8

-- Proof statement
theorem proof_problem : 
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 →
  proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 ∧ proposition5 :=
sorry

end proof_problem_l98_98806


namespace problem1_max_min_abs_values_l98_98959

theorem problem1_max_min_abs_values :
  max (|2-3| - |2-2| + |2-1|) (|3-3| - |3-2| + |3-1|) = 2 ∧
  min (|2-3| - |2-2| + |2-1|) (|3-3| - |3-2| + |3-1|) = 1 := sorry

end problem1_max_min_abs_values_l98_98959


namespace ratio_of_bought_to_given_is_correct_l98_98189

def cavity_rate : ℕ := 4
def canes_from_parents : ℕ := 2
def canes_per_teacher : ℕ := 3
def num_teachers : ℕ := 4
def num_cavities : ℕ := 16

def total_canes_given : ℕ := canes_from_parents + canes_per_teacher * num_teachers
def total_canes_eaten : ℕ := num_cavities * cavity_rate
def canes_bought_with_allowance : ℕ := total_canes_eaten - total_canes_given
def ratio_bought_to_given : ℕ × ℕ := Nat.gcd_eq rfl ▸ (canes_bought_with_allowance / Nat.gcd canes_bought_with_allowance total_canes_given, total_canes_given / Nat.gcd canes_bought_with_allowance total_canes_given)

theorem ratio_of_bought_to_given_is_correct :
  ratio_bought_to_given = (25, 7) :=
  sorry

end ratio_of_bought_to_given_is_correct_l98_98189


namespace ratio_YP_PE_l98_98420

-- Definitions for the points and lengths
variables (X Y Z D E P : Type) [metric_space X] [metric_space Y] [metric_space Z] 
[var_XD: metric_space D] [var_YE : metric_space E] (XY XZ YZ : ℝ)

-- Given conditions
def triangle_XYZ : Prop := 
  dist X Z = 8 ∧ dist X Y = 6 ∧ dist Y Z = 4

-- Angle bisectors definitions
def angle_bisectors_intersect_P : Prop := 
  is_bisector X D Y Z ∧ is_bisector Y E X Z ∧ intersect_bisectors_at P X D Y E

-- Theorem statement
theorem ratio_YP_PE 
  (XYZ : triangle_XYZ X Y Z)
  (bisectors_P : angle_bisectors_intersect_P X Y Z D E P) :
  dist Y P / dist P E = 1.25 := 
sorry

end ratio_YP_PE_l98_98420


namespace average_salary_of_employees_l98_98501

theorem average_salary_of_employees (A : ℝ) 
  (h1 : (20 : ℝ) * A + 3400 = 21 * (A + 100)) : 
  A = 1300 := 
by 
  -- proof goes here 
  sorry

end average_salary_of_employees_l98_98501


namespace dodecagon_diagonals_l98_98214

def D (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals : D 12 = 54 :=
by
  sorry

end dodecagon_diagonals_l98_98214


namespace find_other_number_l98_98900

noncomputable def lcm (a b : Nat) : Nat :=
Nat.lcm a b

theorem find_other_number (x y : Nat) 
  (h1 : lcm x y = 2640) 
  (h2 : Nat.gcd x y = 24) 
  (h3 : x = 240) : y = 264 :=
by
  sorry

end find_other_number_l98_98900


namespace a_2010_eq_4_l98_98408

def units_digit (n : ℕ) : ℕ :=
  n % 10

def a : ℕ → ℕ
| 0 := 2
| 1 := 3
| (n+2) := units_digit (a (n + 1) * a n)

theorem a_2010_eq_4 : a 2009 = 4 :=
sorry

end a_2010_eq_4_l98_98408


namespace max_fridays_in_year_l98_98376

theorem max_fridays_in_year (days_in_common_year days_in_leap_year : ℕ) 
  (h_common_year : days_in_common_year = 365)
  (h_leap_year : days_in_leap_year = 366) : 
  ∃ (max_fridays : ℕ), max_fridays = 53 := 
by
  existsi 53
  sorry

end max_fridays_in_year_l98_98376


namespace find_B_l98_98157

def hundreds_digit_place (n : Int) : Int := n / 100
def tens_digit_place (n : Int) : Int := (n % 100) / 10
def ones_digit_place (n : Int) : Int := n % 10

theorem find_B : ∀ A B : Int, 600 + A * 10 + 5 + 100 * B + 3 = 748 → B = 1 :=
by
  intro A B h
  have h_hundreds : hundreds_digit_place (600 + A * 10 + 5) = 6 := by
    rw [hundreds_digit_place, Int.mod_eq_of_lt]
    norm_num
    norm_num
  have h_hundreds_B : hundreds_digit_place (100 * B + 3) = B := by
    rw [hundreds_digit_place, mul_div_cancel B]
    norm_num
  have h_hundreds_sum : hundreds_digit_place (748) = 7 := by
    rw [hundreds_digit_place]
    norm_num
  have eq_1 : 6 + B = 7 := by
    rw [← h_hundreds, ← h_hundreds_B, ← h_hundreds_sum, h]

  apply Int.add_right_cancel_iff.mp eq_1
  sorry

end find_B_l98_98157


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98230

variables (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1)

-- Definition for the fair coin case
def fair_coin_even_heads_probability (n : ℕ) : ℝ :=
  0.5

-- Definition for the biased coin case
def biased_coin_even_heads_probability (n : ℕ) (p : ℝ) : ℝ :=
  (1 + (1 - 2 * p)^n) / 2

-- Theorems/statements to prove
theorem fair_coin_even_heads (n : ℕ) :
  fair_coin_even_heads_probability n = 0.5 :=
sorry

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  biased_coin_even_heads_probability n p = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98230


namespace distance_between_intersections_l98_98223

theorem distance_between_intersections :
  let u := 0
  let v := 0
  let w := 1
  ∃ (y₁ y₂ : ℝ), (y₁^3 + y₁ - 2 = 0) ∧ (y₂^3 + y₂ - 2 = 0) ∧
    -- Check for distinct solutions
    (if y₁ ≠ y₂ then (x₁ x₂ : ℝ), (x₁ = y₁^3) ∧ (x₂ = y₂^3) ∧
      (x₁ + y₁ = 2) ∧ (x₂ + y₂ = 2) ∧
      let distance := sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) in distance = sqrt (u + v * sqrt w)
    else y₁ = y₂ ∧
      let distance := 0 in distance = sqrt (u + v * sqrt w)) :=
begin
  sorry
end

end distance_between_intersections_l98_98223


namespace monotonically_increasing_and_minimum_value_l98_98692

noncomputable def f (a x : ℝ) : ℝ := abs (x^2 + a * x)

def is_monotonically_increasing (a : ℝ) : Prop :=
  (a ≤ -2 ∨ a ≥ 0) ∧ ∀ (x y : ℝ), 0 ≤ x → x ≤ y → y ≤ 1 → f a x ≤ f a y

def M (a : ℝ) : ℝ := max (f a 0) (f a 1)

def is_minimum (a : ℝ) (val : ℝ) : Prop :=
  ∀ b : ℝ, (M b ≥ val)

theorem monotonically_increasing_and_minimum_value :
  is_monotonically_increasing 2 * (1 - sqrt 2) ∧ is_minimum (2 * (1 - sqrt 2)) (3 - 2 * sqrt 2) := by
  sorry

end monotonically_increasing_and_minimum_value_l98_98692


namespace snow_accumulation_after_seven_days_l98_98195

noncomputable def compacted_snow (s: ℝ) : ℝ := s - s * 0.1
noncomputable def feet_to_inches (f: ℝ) : ℝ := f * 12
noncomputable def inches_to_feet (i: ℝ) : ℝ := i / 12

def initial_snow : ℝ := 0.5
def day2_snow := compacted_snow (inches_to_feet 8)
def day3_melt := inches_to_feet 1
def day4_melt := inches_to_feet 1
def day4_removal := inches_to_feet 6
def day5_snow (d1 d2: ℝ) := 1.5 * (d1 + d2)
def day6_melt := inches_to_feet 3
def day6_accumulate := inches_to_feet 4

theorem snow_accumulation_after_seven_days : 
  let d1 := initial_snow,
      d2 := d1 + day2_snow,
      d3 := d2 - day3_melt,
      d4 := d3 - day4_melt - day4_removal,
      d5 := d4 + day5_snow d1 day2_snow,
      d6 := d5 - day6_melt + day6_accumulate,
      d7 := d6
  in d7 = 2.1667
:= sorry

end snow_accumulation_after_seven_days_l98_98195


namespace integer_solution_unique_l98_98382

theorem integer_solution_unique (n : ℤ) : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 3) ↔ n = 5 :=
by
  sorry

end integer_solution_unique_l98_98382


namespace max_percentage_increase_year_l98_98513

-- Define the revenues per year
def revenues : List (ℕ × ℕ) :=
  [(2000, 2000), (2001, 4000), (2002, 2000), (2003, 6000), (2004, 8000), (2005, 4000)]

-- Define the function to calculate percentage change
def percentageChange (previous current : ℕ) : ℚ :=
  if previous = 0 then 0 else ((current - previous) / previous.toRat) * 100

-- Define a function to find the year with the maximum percentage increase
def yearWithMaxPercentageIncrease (rev : List (ℕ × ℕ)) : ℕ :=
  let changes := rev.zip (rev.tail).map (λ ((year1, rev1), (year2, rev2)) => (year2, percentageChange rev1 rev2))
  let max_change := changes.foldl (λ max_year_current (year, change) 
    => if change > max_year_current.2 then (year, change) else max_year_current) (0, -100)
  max_change.1

-- State the theorem
theorem max_percentage_increase_year : yearWithMaxPercentageIncrease revenues = 2003 := 
by
  sorry

end max_percentage_increase_year_l98_98513


namespace number_of_perfect_square_factors_l98_98378

theorem number_of_perfect_square_factors :
  let n := (2^14) * (3^9) * (5^20)
  ∃ (count : ℕ), 
  (∀ (a : ℕ) (h : a ∣ n), (∃ k, a = k^2) → true) →
  count = 440 :=
by
  sorry

end number_of_perfect_square_factors_l98_98378


namespace variance_of_dataset_l98_98098

def dataset : List ℝ := [-2, -1, 0, 1, 2]

def mean (ds : List ℝ) : ℝ :=
  (ds.foldl (fun acc x => acc + x) 0) / ds.length

def variance (ds : List ℝ) : ℝ :=
  let μ := mean ds
  let squared_diffs := ds.map (fun x => (x - μ) ^ 2)
  (squared_diffs.foldl (fun acc x => acc + x) 0) / ds.length

theorem variance_of_dataset : variance dataset = 2 :=
by
  sorry

end variance_of_dataset_l98_98098


namespace solve_fraction_eq_l98_98221

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 3) :
  (x = 0 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) ∨ 
  (x = 2 / 3 ∧ (x^3 - 3*x^2) / (x^2 - 4*x + 3) + 2*x = 0) :=
sorry

end solve_fraction_eq_l98_98221


namespace find_initial_number_l98_98681

theorem find_initial_number (n : ℕ) :
  (∀ m : ℕ, (m = 5 → (nat.iterate (λ k,
    if k % 3 = 0 then k / 3 else k + 1) 5 n) = 1)) → n = 81 :=
sorry

end find_initial_number_l98_98681


namespace perpendicular_bisector_value_b_l98_98506

theorem perpendicular_bisector_value_b :
  ∀ b : ℝ, (∀ x y : ℝ, (2 + 8) / 2 = x ∧ (5 + 11) / 2 = y → x + y = b) → b = 13 :=
by
  intros b h
  specialize h 5 8
  simp at h
  exact h

end perpendicular_bisector_value_b_l98_98506


namespace integral_eq_l98_98201

noncomputable def integral_value : ℝ :=
  ∫ x in -2..1, |x^2 - 2|

theorem integral_eq :
  integral_value = 1 / 3 + 8 * Real.sqrt 2 / 3 :=
sorry

end integral_eq_l98_98201


namespace sum_of_first_three_terms_coefficients_l98_98124

theorem sum_of_first_three_terms_coefficients (b : ℚ) (hb : b ≠ 0) : 
  (∑ k in {0, 1, 2}, (nat.choose 7 k * 2^k : ℚ) * b^(7-2*k)) = 211 :=
by sorry

end sum_of_first_three_terms_coefficients_l98_98124


namespace probability_of_blue_face_l98_98119

theorem probability_of_blue_face (total_faces blue_faces : ℕ) (h_total : total_faces = 8) (h_blue : blue_faces = 5) : 
  blue_faces / total_faces = 5 / 8 :=
by
  sorry

end probability_of_blue_face_l98_98119


namespace systematic_sampling_interval_l98_98534

-- Definition of the population size and sample size
def populationSize : Nat := 800
def sampleSize : Nat := 40

-- The main theorem stating that the interval k in systematic sampling is 20
theorem systematic_sampling_interval : populationSize / sampleSize = 20 := by
  sorry

end systematic_sampling_interval_l98_98534


namespace maximum_sum_of_grid_l98_98590

noncomputable def max_grid_sum (grid : ℕ → ℕ → ℕ) : ℕ :=
  if h_cond : ∀ i j, 
    (i < 10 ∧ j < 8 → grid i j + grid i (j+1) + grid i (j+2) = 9)  ∧ 
    (i < 8 ∧ j < 10 → grid i j + grid (i+1) j + grid (i+2) j = 9)
  then 360
  else 0

theorem maximum_sum_of_grid :
  ∃ grid : ℕ → ℕ → ℕ,
    (∀ i j, (i < 10 ∧ j < 8 → grid i j + grid i (j+1) + grid i (j+2) = 9) ∧ 
            (i < 8 ∧ j < 10 → grid i j + grid (i+1) j + grid (i+2) j = 9)) ∧
    (max_grid_sum grid = 360) :=
sorry

end maximum_sum_of_grid_l98_98590


namespace player_one_wins_optimal_play_l98_98799

-- Define the game scenario
structure Game :=
  (rows : ℕ)
  (columns : ℕ)
  (vertices : ℕ)
  (connect : ℕ × ℕ → ℕ × ℕ → Prop)
  (endpoint_unique : ∀ v : ℕ × ℕ, ∃! seg, connect seg v)
  (common_points_allowed : ∀ seg1 seg2 : ℕ × ℕ × ℕ × ℕ, 
                            (seg1.1 = seg2.1 ∨ seg1.2 = seg2.2) → 
                            connect seg1.1 seg1.2 ∧ connect seg2.1 seg2.2)

-- State the theorem for player one's winning strategy
theorem player_one_wins_optimal_play : ∀ (g : Game), 
  g.rows = 50 → g.columns = 70 → g.vertices = 50 * 70 → 
  ∃ directions : (ℕ × ℕ) → (ℝ × ℝ), 
    (Σ p1 p2 : ℕ × ℕ, g.connect p1 p2 → directions p1 = -directions p2) ∧ 
    (∑ p : ℕ × ℕ, directions p = (0, 0)) :=
by
  sorry

end player_one_wins_optimal_play_l98_98799


namespace f_of_f_of_3_eq_13_div_9_l98_98452

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2 / x

theorem f_of_f_of_3_eq_13_div_9 : f (f 3) = 13 / 9 := by
  sorry

end f_of_f_of_3_eq_13_div_9_l98_98452


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98236

-- Define the probability of getting heads as p for the biased coin
variable {p : ℝ} (hp : 0 < p ∧ p < 1)

-- Define the number of tosses
variable {n : ℕ}

-- Fair coin probability of getting an even number of heads
theorem fair_coin_even_heads : 
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * (1 / 2)^n else 0) = 0.5 :=
sorry

-- Biased coin probability of getting an even number of heads
theorem biased_coin_even_heads :
  (∑ k in finset.range (n + 1), if even k then (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) else 0) 
  = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98236


namespace find_a_b_l98_98738

theorem find_a_b (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b) 
  (h2 : ∀ x, f' x = 3 * x^2 + 2 * a * x) 
  (h3 : f' 1 = -3) 
  (h4 : f 1 = 0) : 
  a = -3 ∧ b = 2 := 
by
  sorry

end find_a_b_l98_98738


namespace initial_water_amount_l98_98965

variable (W : ℝ)
variable (evap_per_day : ℝ := 0.014)
variable (days : ℕ := 50)
variable (evap_percent : ℝ := 7.000000000000001)

theorem initial_water_amount :
  evap_per_day * (days : ℝ) = evap_percent / 100 * W → W = 10 :=
by
  sorry

end initial_water_amount_l98_98965


namespace sum_second_half_points_185_l98_98000

-- Define the sequences for Lakers and Tigers with corresponding constraints
def lakers_sequence (a d : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (a, a + d, a + 2 * d, a + 3 * d)

def tigers_sequence (b r : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  (b, b * r, b * r^2, b * r^3)

-- Define the conditions
def conditions (a d b : ℕ) (r : ℚ) : Prop :=
  2 * a + d = b * (1 + r) ∧
  4 * a + 6 * d = b * (1 + r + r^2 + r^3) + 2 ∧
  4 * a + 6 * d ≤ 120 ∧
  b * (1 + r + r^2 + r^3) ≤ 120

-- Define the total points in the second half
def second_half_points (a d b : ℕ) (r : ℚ) : ℚ :=
  let (q3_l, q4_l) := (a + 2 * d, a + 3 * d) in
  let (q3_t, q4_t) := (b * r^2, b * r^3) in
  q3_l + q4_l + q3_t + q4_t

-- The final theorem statement
theorem sum_second_half_points_185 (a d b : ℕ) (r : ℚ) :
  conditions a d b r → second_half_points a d b r = 185 :=
begin
  intros h,
  sorry
end

end sum_second_half_points_185_l98_98000


namespace skew_implies_no_common_points_l98_98153

-- Definitions of the conditions
def skew_lines (a b : Type) [linear_ordered_field a] [linear_ordered_field b] : Prop :=
  ∃ P Q : a × b, P ≠ Q ∧ ∀ k l : a × b, k ≠ l → k ≠ P ∧ l ≠ Q

def no_common_points (a b : Type) [linear_ordered_field a] [linear_ordered_field b] : Prop :=
  ∀ P : a × b, ∀ Q : a × b, ¬∃ R : a × b, R = P ∧ R = Q

-- Theorem to be proven
theorem skew_implies_no_common_points (a b : Type) [linear_ordered_field a] [linear_ordered_field b] :
  (skew_lines a b → no_common_points a b) ∧ (¬(no_common_points a b → skew_lines a b)) :=
by {
  -- proof steps go here
  sorry
}

end skew_implies_no_common_points_l98_98153


namespace right_triangle_bc_l98_98808

noncomputable def triangle_ABC (B C A : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1) * (B.1 - A.1) + (B.2 - A.2) * (B.2 - A.2)
  let AC := (C.1 - A.1) * (C.1 - A.1) + (C.2 - A.2) * (C.2 - A.2)
  let BC := (C.1 - B.1) * (C.1 - B.1) + (C.2 - B.2) * (C.2 - B.2)
  let tan_B := (A.2 - C.2) / (B.1 - C.1)
  \((AC = 1) ∧ (AB / AC = 7) ∧ (angle C A B = 90)⟩

theorem right_triangle_bc (B C A : ℝ × ℝ) :
  triangle_ABC B C A → BC = 5 * sqrt 2 :=
by
  sorry

end right_triangle_bc_l98_98808


namespace coffee_shop_visits_l98_98462

theorem coffee_shop_visits
  (visits_per_day : ℕ)
  (cups_per_visit : ℕ)
  (h1 : visits_per_day = 2)
  (h2 : cups_per_visit = 3) :
  visits_per_day * cups_per_visit = 6 :=
by
  rw [h1, h2]
  norm_num
  sorry

end coffee_shop_visits_l98_98462


namespace grayson_fraction_l98_98999

variable (A G O : ℕ) -- The number of boxes collected by Abigail, Grayson, and Olivia, respectively
variable (C_per_box : ℕ) -- The number of cookies per box
variable (TotalCookies : ℕ) -- The total number of cookies collected by Abigail, Grayson, and Olivia

-- Given conditions
def abigail_boxes : ℕ := 2
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48
def total_cookies : ℕ := 276

-- Prove the fraction of the box that Grayson collected
theorem grayson_fraction :
  G * C_per_box = TotalCookies - (abigail_boxes + olivia_boxes) * cookies_per_box → 
  G / C_per_box = 3 / 4 := 
by
  sorry

-- Assume the variables from conditions
variable (G : ℕ := 36 / 48)
variable (TotalCookies := 276)
variable (C_per_box := 48)
variable (A := 2)
variable (O := 3)


end grayson_fraction_l98_98999


namespace expression_evaluation_l98_98637

theorem expression_evaluation : |(-7: ℤ)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end expression_evaluation_l98_98637


namespace part1_part2_l98_98363

-- Definition of f(x)
def f (x : ℝ) (a : ℝ) (c : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + c

-- Problem part 1
theorem part1 (a : ℝ) : f 1 a 19 > 0 ↔ -2 < a ∧ a < 8 :=
by {
  sorry
}

-- Problem part 2
theorem part2 (a c : ℝ) : 
  (∀ x : ℝ, f x a c > 0 ↔ x ∈ Ioo (-1 : ℝ) 3) →
  (a = 3 + Real.sqrt 3 ∨ a = 3 - Real.sqrt 3) ∧ c = 9 :=
by {
  sorry
}

end part1_part2_l98_98363


namespace problem_solved_by_at_least_three_l98_98801

theorem problem_solved_by_at_least_three (girls boys : Finset ℕ) (problems : Finset ℕ)
  (solved : (ℕ × ℕ) → Finset ℕ) 
  (h1 : girls.card = 21) 
  (h2 : boys.card = 21) 
  (h3 : ∀ g ∈ girls, ∀ b ∈ boys, (solved (g, b)).nonempty) 
  (h4 : ∀ p ∈ problems, ∀ g ∈ girls, solved (g, b).card ≤ 6) :
  ∃ p ∈ problems, ∃ (G : Finset ℕ) (B : Finset ℕ), G.card ≥ 3 ∧ B.card ≥ 3 ∧ ∀ g ∈ G, ∀ b ∈ B, p ∈ solved (g, b) :=
sorry

end problem_solved_by_at_least_three_l98_98801


namespace value_of_R_l98_98729

theorem value_of_R (g S R : ℚ) (h1 : R = g * S - 6) (h2 : (S = 7) → (R = 18)) :
  (∀ (S : ℚ), S = 9 → R = 174/7) :=
begin
  sorry
end

end value_of_R_l98_98729


namespace required_run_rate_l98_98815

theorem required_run_rate (initial_rate : ℝ) (initial_overs : ℕ) (target_runs : ℝ) (remaining_overs : ℕ) :
  initial_rate = 3.2 →
  initial_overs = 10 →
  target_runs = 350 →
  remaining_overs = 20 →
  ((target_runs - initial_rate * initial_overs) / remaining_overs) = 15.9 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end required_run_rate_l98_98815


namespace system_of_equations_solution_l98_98222

theorem system_of_equations_solution (x y z : ℝ) 
  (h : ∀ (n : ℕ), x * (1 - 1 / 2^(n : ℝ)) + y * (1 - 1 / 2^(n+1 : ℝ)) + z * (1 - 1 / 2^(n+2 : ℝ)) = 0) : 
  y = -3 * x ∧ z = 2 * x :=
sorry

end system_of_equations_solution_l98_98222


namespace base_length_of_isosceles_triangle_l98_98877

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l98_98877


namespace point_in_second_quadrant_l98_98006

def isInSecondQuadrant (x y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : isInSecondQuadrant (-1) 1 :=
by
  sorry

end point_in_second_quadrant_l98_98006


namespace find_angles_of_triangle_l98_98422

noncomputable def triangle_angles (A B C : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] :=
  ∃ BM AN : ℝ,
  ∃ ∡CBM ∡CAN : ℝ,
  BM = (1 / 2) * AN ∧ ∡CBM = 3 * ∡CAN ∧ 
  ∡A = 72 ∧ ∡B = 108 ∧ ∡C = 72

theorem find_angles_of_triangle :
  ∀ (A B C : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C], triangle_angles A B C :=
by
  sorry

end find_angles_of_triangle_l98_98422


namespace sum_first_10_terms_l98_98447

variable (a : ℕ → ℚ)
variable (d : ℚ)

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def geometric_sequence (a1 a3 a6 : ℚ) : Prop :=
  a3 * a3 = a1 * a6

theorem sum_first_10_terms (h_seq : arithmetic_sequence a d)
                           (h_nonzero : d ≠ 0)
                           (h_a1 : a 1 = 2)
                           (h_geom : geometric_sequence (a 1) (a 3) (a 6)) :
                           ((∑ i in Finset.range 10, a i) = 85 / 2) :=
by
  sorry

end sum_first_10_terms_l98_98447


namespace evaluate_cos_squares_identity_l98_98663

theorem evaluate_cos_squares_identity : 
  (3 - cos^2 (Real.pi / 9)) * (3 - cos^2 (2 * Real.pi / 9)) * (3 - cos^2 (4 * Real.pi / 9)) = (39 / 8) ^ 2 := 
sorry

end evaluate_cos_squares_identity_l98_98663


namespace inequality_solution_l98_98495

theorem inequality_solution (x : ℝ) :
  (\frac{1}{x - 1} - \frac{5}{x - 2} + \frac{5}{x - 3} - \frac{1}{x - 5} < \frac{1}{24}) ↔
  (x ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 1 2 ∪ set.Ioo 3 4 ∪ set.Ioo 5 ∞) :=
sorry

end inequality_solution_l98_98495


namespace price_per_lb_of_second_candy_l98_98086

variable (x : ℝ) -- Price per pound of the second candy
variable (p1 p2 : ℝ) -- Prices per pound of the candies
variable (w1 w2 total_weight : ℕ) -- Weights of the candies and total weight
variable (price_per_lb : ℝ) -- Price per pound of the mixture

-- Defining the conditions given in the problem
variables (h1 : w1 = 20) (h2 : p1 = 2.95)
variables (h3 : w2 = 10) (h4 : total_weight = 30) 
variables (h5 : price_per_lb = 3)

-- Defining the expected result according to the solution
def expected_price := 3.10

-- The actual equality that needs to be proven
theorem price_per_lb_of_second_candy :
  (20 * 2.95 + 10 * x = 30 * 3) → x = expected_price := 
by
  sorry

end price_per_lb_of_second_candy_l98_98086


namespace rate_per_meter_fencing_l98_98901

-- Defining the conditions
variables (w l : ℕ) (P : ℕ) (total_cost : ℕ)

-- Given conditions
def conditions :=
  l = w + 10 ∧
  P = 2*(w + l) ∧
  P = 300 ∧
  total_cost = 1950

-- The statement we want to prove
theorem rate_per_meter_fencing (h : conditions w l P total_cost) :
  total_cost / P = 6.5 :=
by { sorry }

end rate_per_meter_fencing_l98_98901


namespace katya_total_notebooks_l98_98176

-- Definitions based on the conditions provided
def cost_per_notebook : ℕ := 4
def total_rubles : ℕ := 150
def stickers_for_free_notebook : ℕ := 5
def initial_stickers : ℕ := total_rubles / cost_per_notebook

-- Hypothesis stating the total notebooks Katya can obtain
theorem katya_total_notebooks : initial_stickers + (initial_stickers / stickers_for_free_notebook) + 
    ((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) / stickers_for_free_notebook) +
    (((initial_stickers % stickers_for_free_notebook + (initial_stickers / stickers_for_free_notebook)) % stickers_for_free_notebook + 1) / stickers_for_free_notebook) = 46 :=
by
  sorry

end katya_total_notebooks_l98_98176


namespace quadratic_equation_standard_form_quadratic_equation_coefficients_l98_98505

theorem quadratic_equation_standard_form : 
  ∀ (x : ℝ), (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by
  sorry

theorem quadratic_equation_coefficients : 
  ∃ (a b c : ℝ), (a = 2 ∧ b = -6 ∧ c = -1) :=
by
  sorry

end quadratic_equation_standard_form_quadratic_equation_coefficients_l98_98505


namespace correct_option_l98_98133

def condition_A (a : ℝ) : Prop := a^3 * a^4 = a^12
def condition_B (a b : ℝ) : Prop := (-3 * a * b^3)^2 = -6 * a * b^6
def condition_C (a : ℝ) : Prop := (a - 3)^2 = a^2 - 9
def condition_D (x y : ℝ) : Prop := (-x + y) * (x + y) = y^2 - x^2

theorem correct_option (x y : ℝ) : condition_D x y := by
  sorry

end correct_option_l98_98133


namespace k_parallel_l98_98606

noncomputable def k_value : ℚ :=
  let slope_parallel : ℚ := 3 / 2
  let slope_through_points (k : ℚ) : ℚ := (20 + 3) / (k - 5)
  let eq_slopes (k : ℚ) := slope_through_points k = slope_parallel
  Classical.choose (exists_unique_of_exists_of_unique ⟨(61 : ℚ) / 3, sorry⟩ sorry)

theorem k_parallel (k : ℚ) (h : k_value = k) : 
  let slope_parallel := 3 / 2
  let slope_line := (20 + 3) / (k - 5)
  (slope_line = slope_parallel) := 
by 
  sorry

end k_parallel_l98_98606


namespace midpoint_line_op_eq_line_intersects_circle_l98_98401

theorem midpoint_line_op_eq {M N P : ℝ × ℝ}
    (hM : M = (2, 0))
    (hN : N = (0, (2*Real.sqrt 3/3)))
    (hP : P = (1, (Real.sqrt 3/3))) :
  ∃ k : ℝ, (∀ x : ℝ, P.snd = k * P.fst → k = Real.sqrt 3 / 3) :=
sorry

theorem line_intersects_circle
    (M N : ℝ × ℝ)
    (hM : M = (2, 0))
    (hN : N = (0, (2*Real.sqrt 3/3)))
    (center : ℝ × ℝ)
    (hcenter : center = (2, Real.sqrt 3))
    (radius : ℝ)
    (hradius : radius = 2)
    (line_eq : ∀ x y : ℝ, x + Real.sqrt 3 * y - 2 = 0) :
    let d := Real.abs ((center.1 + 3*center.2 - 2)/Real.sqrt (1 + 3^2))
    in d < radius :=
sorry

end midpoint_line_op_eq_line_intersects_circle_l98_98401


namespace quadrilateral_inequality_l98_98478

variable {A B C D : Type}
variable [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]

-- Define vectors representing the sides of the quadrilateral
variable (a b c d e f : ℝ)

-- The statement of the theorem
theorem quadrilateral_inequality (AB BC CD DA AC BD : ℝ)
  (h1 : AB = a) (h2 : BC = b) (h3 : CD = c) (h4 : DA = d) (h5 : AC = e) (h6 : BD = f) :
  AB^2 + BC^2 + CD^2 + DA^2 ≥ AC^2 + BD^2 := sorry

end quadrilateral_inequality_l98_98478


namespace perimeter_PF1F2_l98_98076

-- Defining the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 25) = 1

-- Defining constants
def a : ℝ := 5
def b : ℝ := 3
def c : ℝ := Real.sqrt (a^2 - b^2)
def F1 : (ℝ × ℝ) := (c, 0)
def F2 : (ℝ × ℝ) := (-c, 0)

-- Perimeter of triangle PF1F2
theorem perimeter_PF1F2 (P : ℝ × ℝ) (hp : ellipse P.1 P.2) : 
  P.1 ≠ c ∧ P.1 ≠ -c → (dist P F1 + dist P F2 + dist F1 F2 = 18) :=
by
  sorry

end perimeter_PF1F2_l98_98076


namespace relationship_between_sin_cos_l98_98654

variable {α β : ℝ}
variable {f : ℝ → ℝ}

axiom f_condition : ∀ x : ℝ, f(x - 1) = 4 / f(x) ∧ f(x) ≠ 0
axiom f_symmetry : ∀ x : ℝ, f(-x) = f(x)
axiom f_period : ∀ x : ℝ, f(x + 2) = f(x)
axiom f_monotonic : ∀ x y : ℝ, -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 → x < y → f(x) > f(y)
axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom α_plus_beta_obtuse : α + β < π / 2

theorem relationship_between_sin_cos :
  f (Real.sin α) < f (Real.cos β) := sorry

end relationship_between_sin_cos_l98_98654


namespace find_fg_of_sqrt3_l98_98451

def f (x : ℝ) := -3 * x + 5
def g (x : ℝ) := x^2 + 2 * x + 1

theorem find_fg_of_sqrt3 : f (g (-real.sqrt 3)) = -7 + 6 * real.sqrt 3 := 
by
  sorry

end find_fg_of_sqrt3_l98_98451


namespace suitable_for_promotion_l98_98186

-- Define the conditions
variable {plot_conditions_same : Prop}
variable (avg_yield : ℝ) (variance_A variance_B : ℝ)

-- Assume given values
axiom avg_yield_val : avg_yield = 1200
axiom variance_A_val : variance_A = 186.9
axiom variance_B_val : variance_B = 325.3
axiom plots_conditions_same : plot_conditions_same

-- Define the problem: prove rice variety A is more suitable for promotion
theorem suitable_for_promotion (h1 : plot_conditions_same)
  (h2 : avg_yield = 1200)
  (h3 : variance_A = 186.9)
  (h4 : variance_B = 325.3) : 
  variance_A < variance_B → "A is more suitable for promotion" := by
  sorry

end suitable_for_promotion_l98_98186


namespace only_n_divides_2_pow_n_minus_1_l98_98220

theorem only_n_divides_2_pow_n_minus_1 : ∀ (n : ℕ), n > 0 ∧ n ∣ (2^n - 1) ↔ n = 1 := by
  sorry

end only_n_divides_2_pow_n_minus_1_l98_98220


namespace count_perfect_square_factors_l98_98761

open Nat

def has_larger_square_factor (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

theorem count_perfect_square_factors :
  (Finset.filter has_larger_square_factor (Finset.range 101)).card = 42 := by
sorry

end count_perfect_square_factors_l98_98761


namespace find_a7_l98_98295

variable {a : ℕ → ℝ} (q : ℝ)

-- Define that the sequence a_n is geometric with ratio q.
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The conditions given in the problem.
variables (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
          (h2 : a 9 * a 10 = -8)

theorem find_a7
  (hg : is_geometric_sequence a q) :
  a 7 = -2 :=
sorry

end find_a7_l98_98295


namespace average_next_10_matches_l98_98072

noncomputable def average_runs_first_20_matches : ℝ := 40
noncomputable def total_runs_first_20_matches : ℕ := 20
noncomputable def total_matches : ℕ := 30
noncomputable def overall_average_runs_all_matches : ℝ := 33.333333333333336

theorem average_next_10_matches :
  let total_runs := (overall_average_runs_all_matches * total_matches : ℝ) in
  let runs_first_20 := (average_runs_first_20_matches * total_runs_first_20_matches : ℝ) in
  let runs_next_10 := total_runs - runs_first_20 in
  let average_next_10 := runs_next_10 / 10 in
  average_next_10 = 20 :=
by
  sorry

end average_next_10_matches_l98_98072


namespace frame_equilibrium_angle_l98_98976

variable (l σ G : ℝ)

theorem frame_equilibrium_angle (l_pos : 0 < l) (σ_pos : 0 < σ) (G_pos : 0 < G) :
  ∃ α : ℝ, cos α = (G / (8 * l * σ)) + sqrt ((G / (8 * l * σ)) ^ 2 + 1 / 2) := 
by
  sorry

end frame_equilibrium_angle_l98_98976


namespace express_in_scientific_notation_l98_98955

theorem express_in_scientific_notation :
  (0.00003 : ℝ) = 3 * 10^(-5) :=
sorry

end express_in_scientific_notation_l98_98955


namespace sum_of_coordinates_center_of_square_l98_98063

theorem sum_of_coordinates_center_of_square :
  let P := (3, 0)
  let Q := (5, 0)
  let R := (7, 0)
  let S := (13, 0)
  let m := 3 -- slope derived from solution steps, which we assume here as it doesn't rely on intermediate steps
  let center_x := (3 + 5) / 2
  let center_y := (7 + 13) / 2
  let center_sum := center_x + center_y
  center_sum = 32 / 5 := 
begin
  sorry
end

end sum_of_coordinates_center_of_square_l98_98063


namespace OA_OB_squared_sum_l98_98011

-- Define the parametric equations of C1
def C1_parametric (θ : Real) : Real × Real :=
  (2 * Real.cos θ, Real.sin θ)

-- Define the polar equation of C2
def C2_polar (θ : Real) : Real :=
  2 * Real.sin θ

-- Given points in polar coordinates
def M1 := (1 : Real, Real.pi / 2)
def M2 := (2 : Real, 0)

-- Define the key theorem to prove
theorem OA_OB_squared_sum :
  let A := C1_parametric M1.2;
  let B := C1_parametric (M2.2 + Real.pi / 2);
  1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 5 / 4 :=
sorry

end OA_OB_squared_sum_l98_98011


namespace find_a7_l98_98263

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98263


namespace find_a7_l98_98269

-- Definition of a geometric sequence
def geom_seq (a : ℕ → ℝ) : Prop := ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

-- Given conditions
variable (a : ℕ → ℝ)
variable h_geom : geom_seq a
variable h_eq1 : a 2 * a 4 * a 5 = a 3 * a 6
variable h_eq2 : a 9 * a 10 = -8

-- Question to prove
theorem find_a7 : a 7 = -2 := sorry

end find_a7_l98_98269


namespace perfect_square_factors_count_l98_98754

def perfectSquares := [4, 9, 16, 25, 36, 49, 64, 81]

def countNumbersWithPerfectSquareFactors : Nat :=
  List.length (List.filter (fun n => perfectSquares.any (fun p => n % p = 0)) [1..100])

theorem perfect_square_factors_count :
  countNumbersWithPerfectSquareFactors = 41 := sorry

end perfect_square_factors_count_l98_98754


namespace ellipse_trajectory_and_ab_distance_l98_98331

theorem ellipse_trajectory_and_ab_distance :
  let C1 := (λ x y : ℝ, x^2 + y^2 + 2 * x = 0)
  let C2 := (λ x y : ℝ, x^2 + y^2 - 2 * x - 8 = 0)
  ∃ E : ℝ → ℝ → Prop,
    (∀ x y : ℝ, E x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
    (∀ k : ℝ, let x := 1 in
      (∃ A B : ℝ × ℝ, E A.1 A.2 ∧ E B.1 B.2 ∧ A ≠ B ∧
      ((k = 0 → (|A.2 - B.2| = 3)) ∨ (k ≠ 0 → (3 < |A.2 - B.2| ∧ |A.2 - B.2| ≤ 4)))) :=
by
  sorry

end ellipse_trajectory_and_ab_distance_l98_98331


namespace Earl_owes_Fred_l98_98662

-- Define initial amounts of money each person has
def Earl_initial : ℤ := 90
def Fred_initial : ℤ := 48
def Greg_initial : ℤ := 36

-- Define debts
def Fred_owes_Greg : ℤ := 32
def Greg_owes_Earl : ℤ := 40

-- Define the total money Greg and Earl have together after debts are settled
def Greg_Earl_total_after_debts : ℤ := 130

-- Define the final amounts after debts are settled
def Earl_final (E : ℤ) : ℤ := Earl_initial - E + Greg_owes_Earl
def Fred_final (E : ℤ) : ℤ := Fred_initial + E - Fred_owes_Greg
def Greg_final : ℤ := Greg_initial + Fred_owes_Greg - Greg_owes_Earl

-- Prove that the total money Greg and Earl have together after debts are settled is 130
theorem Earl_owes_Fred (E : ℤ) (H : Greg_final + Earl_final E = Greg_Earl_total_after_debts) : E = 28 := 
by sorry

end Earl_owes_Fred_l98_98662


namespace triangle_ratio_proof_l98_98419

variable (X Y Z P Q R S : Type)
variables [InnerProductSpace ℝ X] [InnerProductSpace ℝ Y] [InnerProductSpace ℝ Z] [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q] [InnerProductSpace ℝ R] [InnerProductSpace ℝ S]
variables {XP PY XQ QZ : ℝ}
variables {XR XS : ℝ}

-- Given conditions
hypothesis h1 : XP = 2
hypothesis h2 : PY = 6
hypothesis h3 : XQ = 3
hypothesis h4 : QZ = 9

-- To prove 
theorem triangle_ratio_proof (h : ∀ (XP = 2) (PY = 6) (XQ = 3) (QZ = 9), XR / XS = 5 / 18) : XR / XS = 5 / 18 :=
  by sorry

end triangle_ratio_proof_l98_98419


namespace series_inequality_l98_98844

theorem series_inequality (n : ℕ) :
  (∑ k in Finset.range n, 1 / ((k + 1) * real.sqrt k)) < 2 * (1 - 1 / real.sqrt (n + 1)) :=
begin
  sorry
end

end series_inequality_l98_98844


namespace sum_mod_1000_l98_98170

-- Define the sequence using given initial values and the recursive relation
def a : ℕ → ℕ
| 0     := 0
| 1     := 2
| 2     := 1
| 3     := 3
| (n+3) := 2 * (a (n+2) + a (n+1) + a n)

-- Given values for a_20, a_21, a_22
lemma given_values : a 20 = 797161 ∧ a 21 = 1812441 ∧ a 22 = 4126643 :=
begin
  split,
  { unfold a, sorry },
  split,
  { unfold a, sorry },
  { unfold a, sorry }
end

-- Prove that the remainder of the sum of the first 19 terms divided by 1000 is 14
theorem sum_mod_1000 : (∑ k in Finset.range 19, a (k + 1)) % 1000 = 14 :=
begin
  have h : given_values, from given_values,
  unfold a,
  sorry
end

end sum_mod_1000_l98_98170


namespace determine_y_coordinates_l98_98410

-- Define the conditions of points making a rectangle
axiom rectangle_points : set (ℝ × ℝ)
axiom rectangle_property (x y : ℝ) : (x, y) ∈ rectangle_points → x + y < 4 → true
axiom probability_condition : (0.4 : ℝ)

-- Define the y-coordinates of the points with x-coordinate equal to 0
def y_coordinates (r : set (ℝ × ℝ)) : set ℝ := {y | (0, y) ∈ r}

-- Claim that determining the y-coordinates requires additional information
theorem determine_y_coordinates :
  ∃ y1 y2 : ℝ, y1 ∈ y_coordinates rectangle_points ∧ y2 ∈ y_coordinates rectangle_points ∧ y1 ≠ y2 → false :=
by {
  sorry
}

end determine_y_coordinates_l98_98410


namespace problem_statement_l98_98636

theorem problem_statement (x y : ℝ) : 
  ((-3 * x * y^2)^3 * (-6 * x^2 * y) / (9 * x^4 * y^5) = 18 * x * y^2) :=
by sorry

end problem_statement_l98_98636


namespace carlos_daisy_difference_l98_98188

theorem carlos_daisy_difference :
  let a := λ n : ℕ, 2 * n
  let b := λ n : ℕ, 2 * n - 1
  let c := λ n : ℕ, (a n)^2 + (b n)^2
  let d := λ n : ℕ, 2 * (a n) * (b n)
  ∑ n in Finset.range 50, (c (n + 1) - d (n + 1)) = 50 :=
by
  sorry

end carlos_daisy_difference_l98_98188


namespace trigonometric_inequality_l98_98056

variables (a x b c : ℝ)

theorem trigonometric_inequality :
  - (sin ((b - c) / 2)) ^ 2 ≤ (cos (a * x + b)) * (cos (a * x + c)) ∧
  (cos (a * x + b)) * (cos (a * x + c)) ≤ (cos ((b - c) / 2)) ^ 2 :=
by
  sorry

end trigonometric_inequality_l98_98056


namespace like_terms_exponents_l98_98790

theorem like_terms_exponents {m n : ℕ} (h1 : 4 * a * b^n = 4 * (a^1) * (b^n)) (h2 : -2 * a^m * b^4 = -2 * (a^m) * (b^4)) :
  (m = 1 ∧ n = 4) :=
by sorry

end like_terms_exponents_l98_98790


namespace wrench_twice_hammer_l98_98750

-- Definitions and conditions from the problem
def hammer_weight := ℝ
def wrench_weight := ℝ
def two_hammers_and_two_wrenches_weight (H W : ℝ) : ℝ := 2 * H + 2 * W
def eight_hammers_and_five_wrenches_weight (H W : ℝ) : ℝ := 8 * H + 5 * W

axiom condition : ∀ (H W : ℝ), two_hammers_and_two_wrenches_weight H W = (1/3) * eight_hammers_and_five_wrenches_weight H W

-- The proof statement
theorem wrench_twice_hammer (H W : ℝ) (h : condition H W) : W = 2 * H :=
by sorry

end wrench_twice_hammer_l98_98750


namespace ratio_of_u_to_v_l98_98541

theorem ratio_of_u_to_v (b u v : ℝ) (Hu : u = -b/12) (Hv : v = -b/8) : 
  u / v = 2 / 3 := 
sorry

end ratio_of_u_to_v_l98_98541


namespace age_difference_remains_constant_l98_98942

variables (a n : ℕ)

theorem age_difference_remains_constant :
  let xiao_shen_age := a
  let xiao_wang_age := a - 8 in
  xiao_shen_age - xiao_wang_age = 8 → 
  xiao_shen_age + (n + 3) - (xiao_wang_age + (n + 3)) = 8 :=
by
  sorry

end age_difference_remains_constant_l98_98942


namespace rhombus_diagonal_sum_maximum_l98_98529

theorem rhombus_diagonal_sum_maximum 
    (x y : ℝ) 
    (h1 : x^2 + y^2 = 100) 
    (h2 : x ≥ 6) 
    (h3 : y ≤ 6) : 
    x + y = 14 :=
sorry

end rhombus_diagonal_sum_maximum_l98_98529


namespace inequality_proof_l98_98518

theorem inequality_proof (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : |x| + |y| + |z| ≤ 1) : 
  x + y / 3 + z / 5 ≤ 2 / 5 :=
sorry

end inequality_proof_l98_98518


namespace distance_from_blast_site_l98_98571

def speed_of_sound : ℝ := 330  -- in meters per second
def time_difference : ℝ := 15  -- in seconds

theorem distance_from_blast_site :
  let distance := speed_of_sound * time_difference in
  distance = 4950 :=
by
  -- We don't need to provide the proof here, just state the theorem.
  sorry

end distance_from_blast_site_l98_98571


namespace part1_part2_l98_98749

def a (x : ℝ) : ℝ × ℝ := (-Real.sin x, 2)
def b (x : ℝ) : ℝ × ℝ := (1, Real.cos x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem part1 : f (Real.pi / 6) = Real.sqrt 3 - 1 / 2 := sorry

theorem part2 (h : f x = 0) : g x = 2 / 9 :=
  let g (x : ℝ) : ℝ := (Real.sin (Real.pi + x) + 4 * Real.cos (2 * Real.pi - x))
                        / (Real.sin (Real.pi / 2 - x) - 4 * Real.sin (-x))
  by sorry

end part1_part2_l98_98749


namespace arithmetic_mean_after_removing_two_l98_98991

theorem arithmetic_mean_after_removing_two :
  ∀ (s : Finset ℕ), s.card = 60 ∧ (s.sum / s.card : ℝ) = 42 →
  (∀ x ∈ s, x = 50 ∨ x = 65 → s.sum - x = 2405) →
  (∀ x ∈ s, x = 50 ∨ x = 65 → (s \ {x}).card = 58) →
  ∀ (new_mean : ℝ),
  new_mean = (2405 / 58) →
  new_mean = 41.5 :=
by
  sorry

end arithmetic_mean_after_removing_two_l98_98991


namespace minimum_black_edges_5x5_l98_98158

noncomputable def minimum_black_edges_on_border (n : ℕ) : ℕ :=
if n = 5 then 5 else 0

theorem minimum_black_edges_5x5 : 
  minimum_black_edges_on_border 5 = 5 :=
by sorry

end minimum_black_edges_5x5_l98_98158


namespace find_b_l98_98679

theorem find_b : ∃ b : ℝ, (15 ^ 2 * 9 ^ 2) / 356 = b ∧ b = 51.19047619047619 :=
by { use (15 ^ 2 * 9 ^ 2) / 356, split, refl, norm_num, sorry }

end find_b_l98_98679


namespace log_sum_four_neg_100_pow_4_eq_40000_l98_98493

noncomputable def log_2 := Real.log 2
noncomputable def log_5 := Real.log 5

theorem log_sum : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1 := 
by {
  -- Using the log product rule and properties of logarithms
  have h1 : Real.log 10 = Real.log 2 + Real.log 5,
  from Eq.symm (Real.log_mul (by norm_num : 0 < 2) (by norm_num : 0 < 5)),
  rw [←Real.log_mul (by norm_num : 0 < 2) (by norm_num : 0 < 5)] at h1,
  norm_num at h1,
  sorry 
}

noncomputable def neg_100 := -100
def four := 4
def pow_four := (neg_100 : ℤ) ^ 4
def four_times_pow_four := four * pow_four

theorem four_neg_100_pow_4_eq_40000 : four_times_pow_four = 40000 := 
by {
  -- Using properties of exponents and multiplication
  have h2 : (-100 : ℤ) ^ 4 = 100000000,
  norm_num,
  have h3 : 4 * 100000000 = 40000,
  norm_num,
  exact Eq.trans (by norm_num) h3
}

end log_sum_four_neg_100_pow_4_eq_40000_l98_98493


namespace find_h_l98_98079

theorem find_h (h : ℝ) (j k : ℝ) 
  (y_eq1 : ∀ x : ℝ, (4 * (x - h)^2 + j) = 2030)
  (y_eq2 : ∀ x : ℝ, (5 * (x - h)^2 + k) = 2040)
  (int_xint1 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (4 * x1 * x2 = 2032) )
  (int_xint2 : ∀ x1 x2 : ℝ, x1 > 0 → x2 > 0 → x1 ≠ x2 → (5 * x1 * x2 = 2040) ) :
  h = 20.5 :=
by
  sorry

end find_h_l98_98079


namespace lowest_possible_price_l98_98167

theorem lowest_possible_price
  (MSRP : ℝ)
  (D1 : ℝ)
  (D2 : ℝ)
  (P_final : ℝ)
  (h1 : MSRP = 45.00)
  (h2 : 0.10 ≤ D1 ∧ D1 ≤ 0.30)
  (h3 : D2 = 0.20) :
  P_final = 25.20 :=
by
  sorry

end lowest_possible_price_l98_98167


namespace probability_inside_sphere_l98_98613

noncomputable def volume_cube : ℝ := 64

noncomputable def volume_sphere : ℝ := (4 * π * 8) / 3

theorem probability_inside_sphere : 
  volume_sphere / volume_cube = π / 6 :=
by 
  sorry

end probability_inside_sphere_l98_98613


namespace midpoint_path_is_straight_line_l98_98938

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

-- Definitions of points and vectors
variables (A B A1 B1 M M1 A2 B2 : V)
variable (k : ℝ)

-- Definition of midpoints
def midpoint (P Q : V) : V := (P + Q) / 2

-- Conditions stating pedestrians are walking uniformly on straight paths
axiom walking_uniformly_on_straight_paths : (∀ t : ℝ, (A1 - A = t * (A2 - A)) ∧
                                              (B1 - B = t * (B2 - B)))

-- Proportional movement
axiom proportional_movement : (A2 - A = k * (A1 - A)) ∧
                              (B2 - B = k * (B1 - B))

-- Prove midpoints are collinear
theorem midpoint_path_is_straight_line :
  let M := midpoint A B in
  let M1 := midpoint A1 B1 in
  let M2 := midpoint A2 B2 in
  collinear ({M, M1, M2} : set V) :=
by 
  let M := midpoint A B,
  let M1 := midpoint A1 B1,
  let M2 := midpoint A2 B2,
  sorry

end midpoint_path_is_straight_line_l98_98938


namespace count_perfect_square_factors_l98_98756

open Finset

noncomputable def count_divisible_by (n : ℕ) (s : Finset ℕ) : ℕ :=
s.filter (λ x, x % n = 0).card

theorem count_perfect_square_factors :
  let S := (range 100).map (λ n, n + 1)
  let perfect_squares := [4, 9, 16, 25, 36, 49, 64, 81, 100] 
  let total := S.card 
  let count_4 := count_divisible_by 4 S
  let count_9 := count_divisible_by 9 S
  let count_16 := count_divisible_by 16 S
  let count_25 := count_divisible_by 25 S
  let count_36 := count_divisible_by 36 S
  let count_49 := count_divisible_by 49 S
  let count_64 := count_divisible_by 64 S
  let count_81 := count_divisible_by 81 S
  let count_100 := count_divisible_by 100 S
  count_4 + (count_9 - count_divisible_by (Nat.lcm 4 9) S) +
    (count_16 - count_divisible_by 4 S) +
    (count_25 - count_divisible_by (Nat.lcm 4 25) S) +
    (count_36 - count_divisible_by 4 S) +
    count_49 + (count_64 - count_divisible_by 4 S) +
    (count_81 - count_divisible_by (Nat.lcm 9 81) S) +
    (count_100 - count_divisible_by 4 S)
    = 40 := 
by
  sorry

end count_perfect_square_factors_l98_98756


namespace ordered_pair_count_l98_98445

theorem ordered_pair_count :
  let ω := by { have h := z.pow(4) - 1; exact (h.roots.nonreal) } in
  (∃! (a b : ℤ), a^2 + b^2 = 1).card = 4 :=
sorry

end ordered_pair_count_l98_98445


namespace min_difference_xue_jie_ti_neng_li_l98_98809

theorem min_difference_xue_jie_ti_neng_li : 
  ∀ (shu hsue jie ti neng li zhan shi : ℕ), 
  shu = 8 ∧ hsue = 1 ∧ jie = 4 ∧ ti = 3 ∧ neng = 9 ∧ li = 5 ∧ zhan = 7 ∧ shi = 2 →
  (shu * 1000 + hsue * 100 + jie * 10 + ti) = 1842 →
  (neng * 10 + li) = 95 →
  1842 - 95 = 1747 := 
by
  intros shu hsue jie ti neng li zhan shi h_digits h_xue_jie_ti h_neng_li
  sorry

end min_difference_xue_jie_ti_neng_li_l98_98809


namespace nancy_percentage_of_insurance_l98_98046

variable (monthly_cost : ℕ) (nancy_payment_annually : ℕ) (annual_cost : ℕ) (percentage_paid : ℚ)

theorem nancy_percentage_of_insurance
  (h1 : monthly_cost = 80)
  (h2 : nancy_payment_annually = 384) :
  let annual_cost := monthly_cost * 12
  let percentage_paid := (nancy_payment_annually : ℚ) / annual_cost * 100
in percentage_paid = 40 :=
by
  sorry

end nancy_percentage_of_insurance_l98_98046


namespace possible_slopes_l98_98607

theorem possible_slopes (k : ℝ) (H_pos : k > 0) :
  (∃ x1 x2 : ℤ, (x1 + x2 : ℝ) = k ∧ (x1 * x2 : ℝ) = -2020) ↔ 
  k = 81 ∨ k = 192 ∨ k = 399 ∨ k = 501 ∨ k = 1008 ∨ k = 2019 := 
by
  sorry

end possible_slopes_l98_98607


namespace angle_p_r_is_60_degrees_l98_98034

variables {V : Type*} [inner_product_space ℝ V]

def unit_vector (v : V) : Prop := ∥v∥ = 1

variables (p q r : V)
variable (lin_indep : ¬ collinear ({p, q, r} : set V))
variable (h1 : unit_vector p)
variable (h2 : unit_vector q)
variable (h3 : unit_vector r)
variable (trip_prod_eq : p ×ₗ (q ×ₗ r) = (q + 2 • r) / 2)

noncomputable def angle_between_vectors (u v : V) : ℝ :=
  real.arccos ((inner_product_space.inner u v) / (∥u∥ * ∥v∥))

theorem angle_p_r_is_60_degrees : angle_between_vectors p r = real.pi / 3 :=
sorry

end angle_p_r_is_60_degrees_l98_98034


namespace find_all_triples_l98_98669

def satisfying_triples (a b c : ℝ) : Prop :=
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 
  (a^2 + a*b = c) ∧ 
  (b^2 + b*c = a) ∧ 
  (c^2 + c*a = b)

theorem find_all_triples (a b c : ℝ) : satisfying_triples a b c ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end find_all_triples_l98_98669


namespace find_sale_in_third_month_l98_98603

def sale_in_first_month := 5700
def sale_in_second_month := 8550
def sale_in_fourth_month := 3850
def sale_in_fifth_month := 14045
def average_sale := 7800
def num_months := 5
def total_sales := average_sale * num_months

theorem find_sale_in_third_month (X : ℕ) 
  (H : total_sales = sale_in_first_month + sale_in_second_month + X + sale_in_fourth_month + sale_in_fifth_month) :
  X = 9455 :=
by
  sorry

end find_sale_in_third_month_l98_98603


namespace power_function_zeros_l98_98719

theorem power_function_zeros :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = x ^ 3) ∧ (f 2 = 8) ∧ (∀ y : ℝ, (f y - y = 0) ↔ (y = 0 ∨ y = 1 ∨ y = -1)) := by
  sorry

end power_function_zeros_l98_98719


namespace product_of_abcd_l98_98568

theorem product_of_abcd :
  ∃ (a b c d : ℚ), 
    3 * a + 4 * b + 6 * c + 8 * d = 42 ∧ 
    4 * (d + c) = b ∧ 
    4 * b + 2 * c = a ∧ 
    c - 2 = d ∧ 
    a * b * c * d = (367 * 76 * 93 * -55) / (37^2 * 74^2) :=
sorry

end product_of_abcd_l98_98568


namespace janet_saving_l98_98823

def tile_cost_difference_saving : ℕ :=
  let turquoise_cost_per_tile := 13
  let purple_cost_per_tile := 11
  let area_wall1 := 5 * 8
  let area_wall2 := 7 * 8
  let total_area := area_wall1 + area_wall2
  let tiles_per_square_foot := 4
  let number_of_tiles := total_area * tiles_per_square_foot
  let cost_difference_per_tile := turquoise_cost_per_tile - purple_cost_per_tile
  number_of_tiles * cost_difference_per_tile

theorem janet_saving : tile_cost_difference_saving = 768 := by
  sorry

end janet_saving_l98_98823


namespace production_period_l98_98968

-- Define the conditions as constants
def daily_production : ℕ := 1500
def price_per_computer : ℕ := 150
def total_earnings : ℕ := 1575000

-- Define the computation to find the period and state what we need to prove
theorem production_period : (total_earnings / price_per_computer) / daily_production = 7 :=
by
  -- you can provide the steps, but it's optional since the proof is omitted
  sorry

end production_period_l98_98968


namespace permutation_median_sum_l98_98029

open Function

noncomputable def median {α : Type*} [LinearOrder α] (a b c : α) : α := (a::b::c::[]).sort (≤).nthLe 1 (by simp)

noncomputable def S (xs : List ℕ) : Finset ℕ :=
  (Finset.range 98).filterMap (fun i => xs.nth (i + 2)).attach.map (median xs.nth! (i + 2))

theorem permutation_median_sum (xs : List ℕ) (hperm : (xs.sort (≤)) = List.range 1 101) :
  Finset.sum (S xs) id ≥ 1122 :=
  sorry

end permutation_median_sum_l98_98029


namespace intersection_of_sets_l98_98336

def setA (x : ℝ) : Prop := abs (x - 2) ≤ 1
def setB (x : ℝ) : Prop := x^2 - 5*x + 4 ≤ 0

theorem intersection_of_sets : set_of setA ∩ set_of setB = set.Icc 1 3 :=
by
  sorry

end intersection_of_sets_l98_98336


namespace equation_of_line_AB_equation_of_circle_AM_l98_98727

-- Definitions of points and midpoint
structure Point where
  x : ℝ
  y : ℝ
  deriving Repr, Inhabited

def A : Point := ⟨-1, 5⟩
def B : Point := ⟨-2, -1⟩
def C : Point := ⟨4, 3⟩

def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

def M : Point := midpoint B C

-- Prove the Equations
theorem equation_of_line_AB : 
  ∃ (a b : ℝ), ∀ (p : Point), (p = A ∨ p = B) → p.y = a * p.x + b := by
  sorry

theorem equation_of_circle_AM :
  ∃ (h k r : ℝ), (h, k) = ((A.x + M.x) / 2, (A.y + M.y) / 2) ∧ r = real.sqrt ((A.x - M.x) ^ 2 + (A.y - M.y) ^ 2) / 2 ∧
  ∀ (p : Point), p = A ∨ p = M → (p.x - h) ^ 2 + (p.y - k) ^ 2 = r ^ 2 := by
  sorry

end equation_of_line_AB_equation_of_circle_AM_l98_98727


namespace solve_problem_l98_98437

-- Let S be the set of words W=w_1w_2…w_n of length n from {x,y,z}
-- Define two words U and V to be similar if we can insert a string "xyz", "yzx", or "zxy"
-- into U to obtain V or into V to obtain U.
-- A word W is trivial if there is a sequence W0=λ, W1, ..., Wm such that W_i and W_(i+1) are similar.
def is_trivial (w : List Char) : Prop :=
  ∃ (m : ℕ) (words : Fin (m + 1) → List Char), words 0 = [] ∧ words m = w ∧
    ∀ i : Fin m, (words (Fin.mk (i + 1) sorry)).isSimilarTo (words i) 

-- The function f(n) specifies the number of trivial words of length 3n.
def f (n : ℕ) : ℕ := if n = 0 then 1 else sorry

-- Define the series sum which equals to p / q for relatively prime p and q.
def series_sum := ∑' (n : ℕ), (f n) * (225 / 8192 : ℚ) ^ n 

noncomputable def p := 32
noncomputable def q := 29

theorem solve_problem : series_sum = p / q ∧ Nat.coprime p q ∧ (p + q) = 61 :=
by
  sorry

end solve_problem_l98_98437


namespace smallest_product_l98_98472

/-- Define the digits available for forming 2-digit numbers --/
def digits := {3, 4, 7, 8}

/-- Define all possible 2-digit combinations from the given digits --/
def combinations := [(37, 48), (38, 47), (47, 38), (48, 37)]

/-- Define the function to calculate the product of two numbers --/
def product (a b : ℕ) : ℕ := a * b

/-- Prove that the smallest product from the given combinations is 1776 --/
theorem smallest_product : 
  let products := combinations.map (λ c => product c.1 c.2) in
  products.minimum = some 1776 :=
by
  sorry

end smallest_product_l98_98472


namespace amc_inequality_l98_98039

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem amc_inequality : (a / (b + c) + b / (a + c) + c / (a + b)) ≥ 3 / 2 :=
sorry

end amc_inequality_l98_98039


namespace find_angle_x_l98_98812

-- Definitions based on conditions
variable (A B C D E : Point)
variable (AB : Line)
variable (CD : Line)
variable (CE : Line)
variable (x : ℝ)

-- Conditions
axiom AB_line : Line AB
axiom CD_perpendicular_AB : Perpendicular CD AB
axiom CE_angle_with_AB : Angle CE AB = 65

-- Theorem statement
theorem find_angle_x : x = 25 := by
  sorry

end find_angle_x_l98_98812


namespace fair_coin_even_heads_biased_coin_even_heads_l98_98232

variables (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1)

-- Definition for the fair coin case
def fair_coin_even_heads_probability (n : ℕ) : ℝ :=
  0.5

-- Definition for the biased coin case
def biased_coin_even_heads_probability (n : ℕ) (p : ℝ) : ℝ :=
  (1 + (1 - 2 * p)^n) / 2

-- Theorems/statements to prove
theorem fair_coin_even_heads (n : ℕ) :
  fair_coin_even_heads_probability n = 0.5 :=
sorry

theorem biased_coin_even_heads (n : ℕ) (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  biased_coin_even_heads_probability n p = (1 + (1 - 2 * p)^n) / 2 :=
sorry

end fair_coin_even_heads_biased_coin_even_heads_l98_98232


namespace selected_numbers_count_l98_98407

noncomputable def check_num_of_selected_numbers : ℕ := 
  let n := 2015
  let max_num := n * n
  let common_difference := 15
  let starting_number := 14
  let count := (max_num - starting_number) / common_difference + 1
  count

theorem selected_numbers_count : check_num_of_selected_numbers = 270681 := by
  -- Skipping the actual proof
  sorry

end selected_numbers_count_l98_98407


namespace ellipse_equation_l98_98349

   theorem ellipse_equation :
     ∀ (a b : ℝ),
       (2 * b = 8 * Real.sqrt 2) →
       (b = 4 * Real.sqrt 2) →
       (∀ c, (c^2 = a^2 - b^2) → (1/3 = c/a)) →
       ∃ (a : ℝ), a^2 = 36 →
       (∀ (x y : ℝ), (x^2 / 36) + (y^2 / 32) = 1) :=
   by
     intro a b h1 h2 h3
     use a
     split
     · sorry
     · intro x y
       split
       · sorry
   
end ellipse_equation_l98_98349


namespace second_discount_percentage_l98_98908

theorem second_discount_percentage
  (initial_price : ℝ) 
  (first_discount_percent : ℝ) 
  (final_price : ℝ) 
  (second_discount_percent : ℝ) : 
  initial_price = 400 →
  first_discount_percent = 20 →
  final_price = 272 →
  second_discount_percent = 15 :=
by
  intros h1 h2 h3
  have p1 : final_price = initial_price * (1 - first_discount_percent / 100) * (1 - second_discount_percent / 100), 
  sorry
  -- Add the necessary logical steps to complete the proof here using the given conditions

end second_discount_percentage_l98_98908


namespace smallest_multiple_l98_98556

theorem smallest_multiple (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 45 = 0 ∧ m % 60 = 0 ∧ m % 25 ≠ 0 ∧ m = n) → n = 180 :=
by
  sorry

end smallest_multiple_l98_98556


namespace sin_four_arcsin_eq_l98_98261

theorem sin_four_arcsin_eq (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) :=
by
  sorry

end sin_four_arcsin_eq_l98_98261
