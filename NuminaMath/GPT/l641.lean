import Mathlib

namespace rotation_period_nullify_weight_distance_centrifugal_equals_gravitational_l641_641002

/-- Proof of the Earth's rotation period T that nullifies the weight of objects at the equator -/
theorem rotation_period_nullify_weight (r g : ℝ) (heq : g ≠ 0) :
    T = 2 * π * sqrt (r / g) :=
sorry

/-- Proof of the distance x from Earth where centrifugal force equals gravitational force given current rotation period T -/
theorem distance_centrifugal_equals_gravitational (r g T : ℝ) (heq : g ≠ 0 ∧ T ≠ 0 ∧ r ≠ 0) :
    x = (g * r^2 * T^2 / (4 * π^2)) ^ (1 / 3) :=
sorry

end rotation_period_nullify_weight_distance_centrifugal_equals_gravitational_l641_641002


namespace garden_perimeter_l641_641313

/-- Define the dimensions of the rectangle and triangle in the garden -/
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 4
def triangle_leg1 : ℕ := 3
def triangle_leg2 : ℕ := 4
def triangle_hypotenuse : ℕ := 5 -- calculated using Pythagorean theorem

/-- Prove that the total perimeter of the combined shape is 28 units -/
theorem garden_perimeter :
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  perimeter = 28 :=
by
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  have h : perimeter = 28 := sorry
  exact h

end garden_perimeter_l641_641313


namespace problem_equiv_proof_l641_641466

theorem problem_equiv_proof {x y : ℝ} (h : |x + y - 2| + (2x - 3y + 5)^2 = 0) :
  (x = 1 ∧ y = 9) ∨ (x = 5 ∧ y = 5) :=
by
  sorry

end problem_equiv_proof_l641_641466


namespace area_of_triangle_ABC_l641_641683

noncomputable def triangle_area_proof (A B C : Point) (circle_radius : ℝ) 
(line_segment : ℝ) (BF_perpendicular : ℝ) (BF_length : ℝ) : Prop :=
circle_radius = 2 * Real.sqrt 5 ∧ 
line_segment = 4 * Real.sqrt 5 ∧ 
BF_perpendicular = True ∧ 
BF_length = 2 → 
triangle_area A B C = 5 * Real.sqrt 5 / 3

theorem area_of_triangle_ABC
    (A B C : Point)
    (circle_radius : ℝ)
    (line_segment : ℝ)
    (BF_perpendicular : ℝ)
    (BF_length : ℝ) :
    triangle_area_proof A B C circle_radius line_segment BF_perpendicular BF_length :=
sorry

end area_of_triangle_ABC_l641_641683


namespace find_a1_l641_641323

def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 1) = (n * (a n / n)) + 1

theorem find_a1 (a : ℕ → ℕ) (h : sequence a) (h30 : a 30 = 30) : a 1 = 274 :=
sorry

end find_a1_l641_641323


namespace min_magnitude_w3z3_l641_641784

open Complex

theorem min_magnitude_w3z3
  (w z : ℂ)
  (h1 : |w + z| = 2)
  (h2 : |w^2 + z^2| = 8) :
  |w^3 + z^3| = 20 :=
sorry

end min_magnitude_w3z3_l641_641784


namespace solve_inequality_l641_641139

noncomputable def discriminant (a : ℝ) : ℝ := 4 + 4 * a

theorem solve_inequality (a : ℝ) : 
  ((a = 0) ∧ ∀ x : ℝ, (2 * x - 1 > 0) ↔ (x > 1 / 2)) ∨
  ((a ≤ -1) ∧ (∀ x : ℝ, ¬(ax^2 + 2*x - 1 > 0))) ∨
  ((a > 0) ∧ ∀ x : ℝ, ((x > (-1 + sqrt (1 + a)) / a) ∨ (x < (-1 - sqrt (1 + a)) / a)) ↔ (ax^2 + 2*x - 1 > 0)) ∨
  ((-1 < a ∧ a < 0) ∧ ∀ x : ℝ, ((-1 + sqrt (1 + a)) / a < x ∧ x < (-1 - sqrt (1 + a)) / a) ↔ (ax^2 + 2*x - 1 > 0)) :=
sorry

end solve_inequality_l641_641139


namespace ellipse_problem_l641_641406

open Real

-- Define the conditions and problem statement
theorem ellipse_problem :
  let a := sqrt 2
  let b := 1
  let focus : Point := (1, 0)
  let point_on_ellipse : Point := (-1, sqrt 2 / 2)
  let ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
  let T : Point := (a^2 / sqrt (a^2 - b^2), 0)
  
  -- Condition: Point (-1, sqrt 2 / 2) lies on the ellipse
  (ellipse point_on_ellipse.1 point_on_ellipse.2) →
  
  -- Condition: F(1,0) is one focus of the ellipse
  let c := sqrt (a^2 - b^2)
  c = 1 ->
  
  -- Condition: a > b > 0
  a > b ∧ b > 0 ->
  
  -- Prove 1: The equation of the ellipse is (x^2 / 2) + y^2 = 1
  ellipse x y = (x^2 / 2 + y^2 = 1) ∧
  
  -- Prove 2: The maximum area of ΔPQT is √2 / 2
  let max_area := sqrt 2 / 2
  (max_area = sqrt 2 / 2) ∧
  
  -- Prove 3: PQ and QT are collinear
  ∃ P Q T : Point,
    P ≠ Q ∧ Q ≠ T ∧
    (let PQ := Q - P;
     QT := T - Q in
     collinear PQ QT) :=
begin
  sorry
end

end ellipse_problem_l641_641406


namespace complex_division_result_l641_641812

theorem complex_division_result :
  let z := (⟨0, 1⟩ - ⟨2, 0⟩) / (⟨1, 0⟩ + ⟨0, 1⟩ : ℂ)
  let a := z.re
  let b := z.im
  a + b = 1 :=
by
  sorry

end complex_division_result_l641_641812


namespace calvin_total_insects_l641_641100

def R : ℕ := 15
def S : ℕ := 2 * R - 8
def C : ℕ := 11 -- rounded from (1/2) * R + 3
def P : ℕ := 3 * S + 7
def B : ℕ := 4 * C - 2
def E : ℕ := 3 * (R + S + C + P + B)
def total_insects : ℕ := R + S + C + P + B + E

theorem calvin_total_insects : total_insects = 652 :=
by
  -- service the proof here.
  sorry

end calvin_total_insects_l641_641100


namespace option_B_minimum_value_option_D_symmetric_line_l641_641826

noncomputable def f : ℝ → ℝ := λ x, Real.sin x + 1 / Real.abs (Real.sin x)

theorem option_B_minimum_value (x : ℝ): 
  ∃ x, f x = 0 := 
sorry

theorem option_D_symmetric_line (x : ℝ) : 
  f (π - x) = f x := 
sorry

end option_B_minimum_value_option_D_symmetric_line_l641_641826


namespace rectangle_area_proof_l641_641318

variable (x y : ℕ) -- Declaring the variables to represent length and width of the rectangle.

-- Declaring the conditions as hypotheses.
def condition1 := (x + 3) * (y - 1) = x * y
def condition2 := (x - 3) * (y + 2) = x * y
def condition3 := (x + 4) * (y - 2) = x * y

-- The theorem to prove the area is 36 given the above conditions.
theorem rectangle_area_proof (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : x * y = 36 :=
by
  sorry

end rectangle_area_proof_l641_641318


namespace find_a_plus_b_l641_641827

def f (x : ℝ) : ℝ := 2 * x + 3
def g (a x : ℝ) : ℝ := a * x + Real.log x
def y (a x : ℝ) : ℝ := (1 - 1/2*a) * x - 1/2 * Real.log x + 3/2

theorem find_a_plus_b (a b : ℝ) (x1 x2 : ℝ)
  (h1 : b = f x1)
  (h2 : b = g a x2)
  (h3 : ∀ x > 0, y a x = 2 → x = 1 / (2 - a)) 
  (h4 : x2 > 0)
  (h5 : 1 / (2 - a) = x2) :
  a + b = 2 :=
sorry

end find_a_plus_b_l641_641827


namespace angle_ABC_45_l641_641400

theorem angle_ABC_45 (A B C D : Type) [triangle A B C]
  (h1 : ∠BAC = 30)
  (h2 : median B D)
  (h3 : ∠BDC = 45) :
  ∠ABC = 45 :=
sorry

end angle_ABC_45_l641_641400


namespace percent_employed_l641_641887

theorem percent_employed (E : ℝ) : 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30 -- 1 - percent_females
  (percent_males * E = employed_males) → E = 70 := 
by 
  let employed_males := 0.21
  let percent_females := 0.70
  let percent_males := 0.30
  intro h
  sorry

end percent_employed_l641_641887


namespace ao_bisects_ch_l641_641516

noncomputable def triangle {α : Type*} [ordered_comm_group α] (A B C : Point α) : Prop :=
(A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

variables {α : Type*} [ordered_comm_group α]

variables (A B C T H O : Point α)
variables (AB AC AT CH AO : Line α)
variables (circumcircle_ABC : Circle α)

-- Conditions
def condition_1 : triangle A B C := Sorry
def condition_2 {C : Point α} (T : Point α) : Tangent T circumcircle_ABC := Sorry 
def condition_3 (H : Point α) (A B C : Point α) : Orthocenter H A B C := Sorry
def condition_4 (O : Point α) (A B C : Point α) : Circumcenter O A B C := Sorry
def condition_5 (CH_mid : Point α) : PassThrough CH_mid (Midpoint A T) := Sorry

-- Question to Prove
theorem ao_bisects_ch :
  bisects AO CH := sorry

end ao_bisects_ch_l641_641516


namespace Morse_code_sequences_l641_641046

theorem Morse_code_sequences : 
  let symbols (n : ℕ) := 2^n in
  symbols 1 + symbols 2 + symbols 3 + symbols 4 + symbols 5 = 62 :=
by
  sorry

end Morse_code_sequences_l641_641046


namespace powers_of_two_diff_div_by_1987_l641_641563

theorem powers_of_two_diff_div_by_1987 :
  ∃ a b : ℕ, a > b ∧ 1987 ∣ (2^a - 2^b) :=
by sorry

end powers_of_two_diff_div_by_1987_l641_641563


namespace find_a_l641_641587

theorem find_a 
  (a : ℝ) 
  (h : binom (6) 1 * a ^ 5 * (sqrt (3) / 6) = - sqrt (3)) :
  a = -1 := sorry

end find_a_l641_641587


namespace sum_lcm_15_eq_45_l641_641261

open Nat

theorem sum_lcm_15_eq_45 :
  ∑ m in {m : ℕ | m > 0 ∧ Nat.lcm m 15 = 45}, m = 60 :=
by
  sorry

end sum_lcm_15_eq_45_l641_641261


namespace sqrt_sum_ge_two_l641_641904

theorem sqrt_sum_ge_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 := 
by
  sorry

end sqrt_sum_ge_two_l641_641904


namespace isosceles_triangle_side_length_l641_641712

theorem isosceles_triangle_side_length :
  let a := 1
  let b := Real.sqrt 3
  let right_triangle_area := (1 / 2) * a * b
  let isosceles_triangle_area := right_triangle_area / 3
  ∃ s, s = Real.sqrt 109 / 6 ∧ 
    (∀ (base height : ℝ), 
      (base = a / 3 ∨ base = b / 3) ∧
      height = (2 * isosceles_triangle_area) / base → 
      1 / 2 * base * height = isosceles_triangle_area) :=
by
  sorry

end isosceles_triangle_side_length_l641_641712


namespace first_player_wins_optimal_play_l641_641698

-- Definition of the game conditions
def chessboard : Type := ℕ × ℕ
def initial_position (c : chessboard) : Prop := true -- any valid chessboard position is acceptable

def valid_move (dist_prev dist_current : ℕ) : Prop := dist_current > dist_prev

-- Predicate to determine if a player who cannot move loses
def cannot_move (pos : chessboard) (moves : list (chessboard × chessboard)) : Prop :=
  ∀ (new_pos : chessboard), ¬(valid_move (dist euclidean_distance (fst (head moves) pos))

-- The main theorem to prove
theorem first_player_wins_optimal_play (initial_pos : chessboard) :
  ∃ (first_player_wins : Prop), first_player_wins :=
  sorry

end first_player_wins_optimal_play_l641_641698


namespace max_product_two_integers_sum_2000_l641_641204

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641204


namespace math_problem_proof_l641_641879

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0

def line_parametric (t α : ℝ) : ℝ × ℝ :=
(-1 + t * cos α, t * sin α)

def intersect_line_circle (t α : ℝ) : Prop :=
let (x, y) := line_parametric t α in circle_equation x y

def range_alpha (α : ℝ) : Prop :=
α ∈ Ico 0 (π / 6) ∪ Ioo (5 * π / 6) π

def PA (t α : ℝ) : ℝ := 
real.sqrt ((-1 + t * cos α + 1)^2 + (t * sin α)^2)

def PB (t α : ℝ) : ℝ := 
real.sqrt ((-1 + t * cos α + 1)^2 + (t * sin α)^2)

def reciprocal_sum (PA PB : ℝ) : ℝ := 
1 / PA + 1 / PB

def range_reciprocal_sum (α : ℝ) : Prop :=
∃ t1 t2 : ℝ, intersect_line_circle t1 α ∧ intersect_line_circle t2 α ∧ t1 ≠ t2 ∧
let sum := reciprocal_sum (PA t1 α) (PB t2 α) in 
(2 * real.sqrt 3 / 3 < sum ∧ sum ≤ 4 / 3)

theorem math_problem_proof (α : ℝ) : 
(range_alpha α ∧ range_reciprocal_sum α) ↔ 
(α ∈ Ico 0 (π / 6) ∪ Ioo (5 * π / 6) π ∧
∃ t1 t2 : ℝ, let sum := reciprocal_sum (PA t1 α) (PB t2 α) in 
(2 * real.sqrt 3 / 3 < sum ∧ sum ≤ 4 / 3)) :=
sorry

end math_problem_proof_l641_641879


namespace food_insufficiency_l641_641118

-- Given conditions
def number_of_dogs : ℕ := 5
def food_per_meal : ℚ := 3 / 4
def meals_per_day : ℕ := 3
def initial_food : ℚ := 45
def days_in_two_weeks : ℕ := 14

-- Definitions derived from conditions
def daily_food_per_dog : ℚ := food_per_meal * meals_per_day
def daily_food_for_all_dogs : ℚ := daily_food_per_dog * number_of_dogs
def total_food_in_two_weeks : ℚ := daily_food_for_all_dogs * days_in_two_weeks

-- Proof statement: proving the food consumed exceeds the initial amount
theorem food_insufficiency : total_food_in_two_weeks > initial_food :=
by {
  sorry
}

end food_insufficiency_l641_641118


namespace joan_first_payment_l641_641508

theorem joan_first_payment (P : ℝ) 
  (total_amount : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (h_total : total_amount = 109300)
  (h_r : r = 3)
  (h_n : n = 7)
  (h_sum : total_amount = P * (1 - r^n) / (1 - r)) : 
  P = 100 :=
by
  -- proof goes here
  sorry

end joan_first_payment_l641_641508


namespace tourist_journey_home_days_l641_641325

theorem tourist_journey_home_days (x v : ℝ)
  (h1 : (x / 2 + 1) * v = 246)
  (h2 : x * (v + 15) = 276) :
  x + (x / 2 + 1) = 4 :=
by
  sorry

end tourist_journey_home_days_l641_641325


namespace coplanar_lines_l641_641930

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
  (2 + s, 5 - k * s, 3 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 * t, 4 + 2 * t, 6 - 2 * t)

theorem coplanar_lines (k : ℝ) :
  (exists s t : ℝ, line1 s k = line2 t) ∨ line1 1 k = (1, -k, k) ∧ line2 1 = (2, 2, -2) → k = -1 :=
by sorry

end coplanar_lines_l641_641930


namespace mark_gave_books_to_alice_l641_641115

theorem mark_gave_books_to_alice (x : ℕ) : 
  let mark_initial := 105 in
  let alice_initial := 15 in
  (mark_initial - x) = 3 * (alice_initial + x) → 
  x = 15 :=
by
  sorry

end mark_gave_books_to_alice_l641_641115


namespace junior_score_calculation_l641_641055

variable {total_students : ℕ}
variable {junior_score senior_average : ℕ}
variable {junior_ratio senior_ratio : ℚ}
variable {class_average total_average : ℚ}

-- Hypotheses from the conditions
theorem junior_score_calculation (h1 : junior_ratio = 0.2)
                               (h2 : senior_ratio = 0.8)
                               (h3 : class_average = 82)
                               (h4 : senior_average = 80)
                               (h5 : total_students = 10)
                               (h6 : total_average * total_students = total_students * class_average)
                               (h7 : total_average = (junior_ratio * junior_score + senior_ratio * senior_average))
                               : junior_score = 90 :=
sorry

end junior_score_calculation_l641_641055


namespace max_product_two_integers_sum_2000_l641_641211

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641211


namespace value_of_y_l641_641841

theorem value_of_y (x y : ℝ) (h1 : 3 * (x - y) = 18) (h2 : x + y = 20) : y = 7 := by
  sorry

end value_of_y_l641_641841


namespace max_group_cardinality_l641_641779

theorem max_group_cardinality {s : Finset ℕ} (h₁ : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 200) (h₂ : ∀ x y ∈ s, x ≠ y → (x + y) % 5 = 0) : s.card ≤ 40 :=
sorry

end max_group_cardinality_l641_641779


namespace num_std_devs_l641_641583

/--
The arithmetic mean of the normal distribution is 16.5, and the standard deviation is 1.5.
If a value is 13.5, prove that it is -2 standard deviations less than the mean.
-/
theorem num_std_devs (mean stddev value : ℝ) (h_mean : mean = 16.5) (h_stddev : stddev = 1.5) (h_value : value = 13.5) : 
  (value - mean) / stddev = -2 :=
by
  rw [h_mean, h_stddev, h_value]
  norm_num
  sorry

end num_std_devs_l641_641583


namespace max_product_l641_641223

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641223


namespace sqrt_25_eq_pm_5_l641_641999

theorem sqrt_25_eq_pm_5 : {x : ℝ | x^2 = 25} = {5, -5} :=
by
  sorry

end sqrt_25_eq_pm_5_l641_641999


namespace sheep_color_unchangeable_l641_641549

def initial_counts : ℕ × ℕ × ℕ := (22, 18, 15)

def allowed_transformations (b r g : ℕ) : list (ℕ × ℕ × ℕ) :=
  [ (b - 1, r - 1, g + 2), (b - 1, r + 2, g - 1), (b + 2, r - 1, g - 1) ]

def invariant (r g : ℕ) : ℕ := (r - g) % 3

theorem sheep_color_unchangeable (b r g : ℕ) :
  invariant r g = 0 ∧ (b, r, g) = initial_counts → 
  ∃ n, n = b + r + g → ∀ (b' r' g' : ℕ), 
    (b', r', g') ∈ (iterate allowed_transformations (b, r, g) _) →
    b' = n ∧ r' = 0 ∧ g' = 0 :=
by
  sorry

end sheep_color_unchangeable_l641_641549


namespace number_of_people_in_room_l641_641175

-- Given conditions
variables (people chairs : ℕ)
variables (three_fifths_people_seated : ℕ) (four_fifths_chairs : ℕ)
variables (empty_chairs : ℕ := 5)

-- Main theorem to prove
theorem number_of_people_in_room
    (h1 : 5 * empty_chairs = chairs)
    (h2 : four_fifths_chairs = 4 * chairs / 5)
    (h3 : three_fifths_people_seated = 3 * people / 5)
    (h4 : four_fifths_chairs = three_fifths_people_seated)
    : people = 33 := 
by
  -- Begin the proof
  sorry

end number_of_people_in_room_l641_641175


namespace row_product_is_minus_one_l641_641738

variables {a b : ℕ → ℝ} -- a_i, b_j are real numbers parameterized by natural numbers

-- All elements ai and bi are distinct
axiom h_distinct_a : ∀ i j : ℕ, i ≠ j → a i ≠ a j
axiom h_distinct_b : ∀ i j : ℕ, i ≠ j → b i ≠ b j

-- The product of the numbers in any column is equal to 1
axiom h_column_product : ∀ j : ℕ, j < 100 → (∏ i in Finset.range 100, (a i + b j)) = 1 

theorem row_product_is_minus_one (i : ℕ) (h_i : i < 100) : 
  (∏ j in Finset.range 100, (a i + b j)) = -1 := 
sorry

end row_product_is_minus_one_l641_641738


namespace coplanar_lines_k_values_l641_641351

theorem coplanar_lines_k_values (k : ℝ) :
  (∃ t u : ℝ, 
    (1 + t = 2 + u) ∧ 
    (2 + 2 * t = 5 + k * u) ∧ 
    (3 - k * t = 6 + u)) ↔ 
  (k = -2 + Real.sqrt 6 ∨ k = -2 - Real.sqrt 6) :=
sorry

end coplanar_lines_k_values_l641_641351


namespace ratio_unit_price_l641_641335

theorem ratio_unit_price (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  let vX := 1.25 * v
  let pX := 0.85 * p
  (pX / vX) / (p / v) = 17 / 25 := by
{
  sorry
}

end ratio_unit_price_l641_641335


namespace part1_part2_l641_641612

def seq_a (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) = (2^(n + 1) * a n) / ((n + 1/2) * a n + 2^n)

def seq_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b n = 2^n / a n

def seq_c (a : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
∀ n : ℕ, c n = 1 / (n * (n + 1) * a (n + 1))

def seq_S (c : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = ∑ i in finset.range n, c i

theorem part1 (a b : ℕ → ℝ) (h: seq_a a ∧ seq_b a b) :
  ∀ n : ℕ, b n = (n^2 + 1) / 2 :=
sorry   

theorem part2 (a c S : ℕ → ℝ) (h_a: seq_a a) (h_c: seq_c a c) (h_S: seq_S c S):
  ∀ n : ℕ, S n = 1/2 * (1 - (1/2)^(n+1) * (n+2)/(n+1)) ∧ 5/16 ≤ S n ∧ S n < 1/2 :=
sorry

end part1_part2_l641_641612


namespace lori_earnings_l641_641536

theorem lori_earnings
    (red_cars : ℕ)
    (white_cars : ℕ)
    (cost_red_car : ℕ)
    (cost_white_car : ℕ)
    (rental_time_hours : ℕ)
    (rental_time_minutes : ℕ)
    (correct_earnings : ℕ) :
    red_cars = 3 →
    white_cars = 2 →
    cost_red_car = 3 →
    cost_white_car = 2 →
    rental_time_hours = 3 →
    rental_time_minutes = rental_time_hours * 60 →
    correct_earnings = 2340 →
    (red_cars * cost_red_car + white_cars * cost_white_car) * rental_time_minutes = correct_earnings :=
by
  intros
  sorry

end lori_earnings_l641_641536


namespace jimmy_candy_bars_l641_641507

theorem jimmy_candy_bars :
  let candy_bar_cost := 0.75
  let lollipop_cost := 0.25
  let num_lollipops := 4
  let fraction_spent_on_candy := 1 / 6
  let driveway_charge := 1.5
  let num_driveways := 10
  let money_earned_from_shoveling := driveway_charge * num_driveways
  let money_spent_on_candy := fraction_spent_on_candy * money_earned_from_shoveling
  let cost_of_lollipops := num_lollipops * lollipop_cost
  let money_spent_on_candy_bars := money_spent_on_candy - cost_of_lollipops
  let num_candy_bars := money_spent_on_candy_bars / candy_bar_cost
  in
  num_candy_bars = 2 := sorry

end jimmy_candy_bars_l641_641507


namespace initial_population_l641_641609

theorem initial_population (P : ℝ) 
    (h1 : 1.25 * P * 0.70 = 363650) : 
    P = 415600 :=
sorry

end initial_population_l641_641609


namespace total_students_l641_641927

theorem total_students (T : ℕ) 
  (hA : (1/5) * T) 
  (hB : (1/4) * T) 
  (hC : (1/2) * T) 
  (hD : 25 = 25) : 
  T = 500 :=
by 
  have h : T = (1/5) * T + (1/4) * T + (1/2) * T + 25, 
  sorry

end total_students_l641_641927


namespace train_stops_15_min_per_hour_l641_641267

/-
Without stoppages, a train travels a certain distance with an average speed of 80 km/h,
and with stoppages, it covers the same distance with an average speed of 60 km/h.
Prove that the train stops for 15 minutes per hour.
-/
theorem train_stops_15_min_per_hour (D : ℝ) (h1 : 0 < D) :
  let T_no_stop := D / 80
  let T_stop := D / 60
  let T_lost := T_stop - T_no_stop
  let mins_per_hour := T_lost * 60
  mins_per_hour = 15 := by
  sorry

end train_stops_15_min_per_hour_l641_641267


namespace diagonals_form_45_deg_angle_l641_641865

theorem diagonals_form_45_deg_angle :
  ∃ θ : ℝ, θ = 45 ∧ 
  let A := (0, 0)
  let B := (5, 0)
  let C := (3, 2)
  let D := (0, 1) in
  let m_AC := (C.2 - A.2) / (C.1 - A.1)
  let m_BD := (D.2 - B.2) / (D.1 - B.1) in
  (Real.arctan m_AC - Real.arctan m_BD).abs = θ * Real.pi / 180 :=
sorry

end diagonals_form_45_deg_angle_l641_641865


namespace solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l641_641576

theorem solve_quadratic_eq1 : ∀ x : ℝ, 2 * x^2 + 5 * x + 3 = 0 → (x = -3/2 ∨ x = -1) :=
by
  intro x
  intro h
  sorry

theorem solve_quadratic_eq2_complete_square : ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l641_641576


namespace ratio_of_points_to_away_home_game_l641_641924

-- Definitions
def first_away_game_points (A : ℕ) : ℕ := A
def second_away_game_points (A : ℕ) : ℕ := A + 18
def third_away_game_points (A : ℕ) : ℕ := A + 20
def last_home_game_points : ℕ := 62
def next_game_points : ℕ := 55
def total_points (A : ℕ) : ℕ := A + (A + 18) + (A + 20) + 62 + 55

-- Given that the total points should be four times the points of the last home game
def target_points : ℕ := 4 * 62

-- The main theorem to prove
theorem ratio_of_points_to_away_home_game : ∀ A : ℕ,
  total_points A = target_points → 62 = 2 * A :=
by
  sorry

end ratio_of_points_to_away_home_game_l641_641924


namespace minimum_value_of_func_l641_641475

-- Define the circle and the line constraints, and the question
namespace CircleLineProblem

def is_center_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 1 = 0

def line_divides_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, is_center_of_circle x y → a * x - b * y + 3 = 0

noncomputable def func_to_minimize (a b : ℝ) : ℝ :=
  (2 / a) + (1 / (b - 1))

theorem minimum_value_of_func :
  ∃ (a b : ℝ), a > 0 ∧ b > 1 ∧ line_divides_circle a b ∧ func_to_minimize a b = 8 :=
by
  sorry

end CircleLineProblem

end minimum_value_of_func_l641_641475


namespace min_value_of_a_l641_641992

theorem min_value_of_a (r s t : ℕ) (h1 : r > 0) (h2 : s > 0) (h3 : t > 0)
  (h4 : r * s * t = 2310) (h5 : r + s + t = a) : 
  a = 390 → True :=
by { 
  intros, 
  sorry 
}

end min_value_of_a_l641_641992


namespace yard_length_l641_641872

theorem yard_length :
  let num_trees := 11
  let distance_between_trees := 18
  (num_trees - 1) * distance_between_trees = 180 :=
by
  let num_trees := 11
  let distance_between_trees := 18
  sorry

end yard_length_l641_641872


namespace special_property_of_five_numbers_l641_641155

-- Given the five numbers and their prime factor decompositions
variables (a1 a2 a3 a4 a5 : ℕ)

-- Define the LCM function for a list of natural numbers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
l.foldr lcm 1

-- Define the primary proof problem statement
theorem special_property_of_five_numbers 
  (i j k : Fin 5) (h_ij : i ≠ j) (h_jk : j ≠ k) (h_ik : i ≠ k) :
  lcm_list [a1, a2, a3, a4, a5] = lcm_list [(&[a1, a2, a3, a4, a5]).get ⟨i, _⟩, 
                                             (&[a1, a2, a3, a4, a5]).get ⟨j, _⟩, 
                                             (&[a1, a2, a3, a4, a5]).get ⟨k, _⟩] →
  ∀ p n, -- for any prime number p and its power n,
  (prime p ∧ dvd (p ^ n) a1 ∨ dvd (p ^ n) a2 ∨ dvd (p ^ n) a3 ∨ dvd (p ^ n) a4 ∨ dvd (p ^ n) a5) →
  (dvd (p ^ n) ((&[a1, a2, a3, a4, a5]).get ⟨i, _⟩) ∨ 
   dvd (p ^ n) ((&[a1, a2, a3, a4, a5]).get ⟨j, _⟩) ∨ 
   dvd (p ^ n) ((&[a1, a2, a3, a4, a5]).get ⟨k, _⟩)) := sorry

end special_property_of_five_numbers_l641_641155


namespace shaded_regions_area_closest_to_61_l641_641071

namespace GeometricProof

-- Variables for the problem
variables (A B C P S T V Q U : Point)
variables (r : ℝ) -- Radius of quarter circle

-- Problem conditions
axiom quarter_circle : circle B r
axiom side_length_10 : ∀ s : Square, s.side_length = 10
axiom point_on_AB : P ∈ (segment A B) ∧ S ∈ (segment A B)
axiom point_on_BC : T ∈ (segment B C) ∧ V ∈ (segment B C)
axiom point_on_circle : Q ∈ quarter_circle ∧ U ∈ quarter_circle
axiom line_ac : line A C

theorem shaded_regions_area_closest_to_61 :
  let area (s : Set Point) : ℝ := sorry -- Here we would calculate the area of a set of points
  let shaded_regions := [triangle D Q E, triangle E R F, triangle F U G].toSet -- Shaded regions
  let total_area := (shaded_regions.map area).sum
  |int_closest total_area 61 :=
  sorry
-- Note: Detailed definitions for geometrical entities and operations like area should be defined or imported here appropriately

end GeometricProof

end shaded_regions_area_closest_to_61_l641_641071


namespace symmetric_point_origin_l641_641592

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

theorem symmetric_point_origin (x y : ℝ) (h : (x, y) = (-2, 3)) :
  symmetric_point (x, y) = (2, -3) :=
by
  rw [h]
  unfold symmetric_point
  simp
  sorry

end symmetric_point_origin_l641_641592


namespace ab_equals_4_l641_641462

theorem ab_equals_4 (a b : ℝ) (h_pos : a > 0 ∧ b > 0)
  (h_area : (1/2) * (12 / a) * (8 / b) = 12) : a * b = 4 :=
by
  sorry

end ab_equals_4_l641_641462


namespace prob_X_gt_125_l641_641483

noncomputable def is_normal (X : ℝ → ℝ) (μ : ℝ) (σ : ℝ) : Prop :=
  ∀ x : ℝ, X x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - μ)^2 / (2 * σ^2))

axiom prob_interval (X : ℝ → ℝ) (μ : ℝ) (σ : ℝ) (a b : ℝ) : Prop :=
  ∀ a b : ℝ, X ∼ N(μ, σ^2) → P(a ≤ X ∧ X ≤ b) = 0.72

theorem prob_X_gt_125 (X : ℝ → ℝ) (σ : ℝ) :
  is_normal X 110 σ → prob_interval X 110 σ 95 125 → P(X > 125) = 0.14 :=
by
  intros h_norm h_prob_int
  sorry

end prob_X_gt_125_l641_641483


namespace valve_opening_l641_641264

variable (V : Type)
variable (open : V → Prop)
variables (v1 v2 v3 v4 v5 : V)

-- Conditions
axiom cond1 : open v2 → open v3 ∧ ¬ open v1
axiom cond2 : open v1 ∨ open v3 → ¬ open v5
axiom cond3 : ¬ (¬ open v4 ∧ ¬ open v5)

-- Theorem statement
theorem valve_opening : open v2 → open v3 ∧ open v4 :=
by
  sorry

end valve_opening_l641_641264


namespace symmetric_point_origin_l641_641594

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end symmetric_point_origin_l641_641594


namespace sum_special_integers_l641_641262

theorem sum_special_integers : (∑ n in (Finset.filter (λ x, ¬ x % 4 = 0) (Finset.range' 11 24)), n) = 173 := by
  sorry

end sum_special_integers_l641_641262


namespace Misha_l641_641659

theorem Misha's_decision_justified :
  let A_pos := 7 in
  let A_neg := 4 in
  let B_pos := 4 in
  let B_neg := 1 in
  (B_pos / (B_pos + B_neg) > A_pos / (A_pos + A_neg)) := 
sorry

end Misha_l641_641659


namespace sum_of_roots_of_quadratic_l641_641013

theorem sum_of_roots_of_quadratic (x1 x2 : ℝ) (h : x1 * x2 + -(x1 + x2) * 6 + 5 = 0) : x1 + x2 = 6 :=
by
-- Vieta's formulas for the sum of the roots of a quadratic equation state that x1 + x2 = -b / a.
sorry

end sum_of_roots_of_quadratic_l641_641013


namespace ratio_approx_34_l641_641017

variable (a b : ℝ)
hypothesis (h1 : a > b > 0)
hypothesis (h2 : (a + b) / 2 = 3 * Real.sqrt (a * b))

theorem ratio_approx_34 : (a / b).approx 34 :=
by
  sorry

end ratio_approx_34_l641_641017


namespace number_of_petri_dishes_l641_641492

def germs_in_lab : ℕ := 3700
def germs_per_dish : ℕ := 25
def num_petri_dishes : ℕ := germs_in_lab / germs_per_dish

theorem number_of_petri_dishes : num_petri_dishes = 148 :=
by
  sorry

end number_of_petri_dishes_l641_641492


namespace no_two_heads_consecutively_probability_l641_641288

theorem no_two_heads_consecutively_probability :
  (∃ (total_sequences : ℕ) (valid_sequences : ℕ),
    total_sequences = 2^10 ∧ valid_sequences = 1 + 10 + 36 + 56 + 35 + 6 ∧
    (valid_sequences / total_sequences : ℚ) = 9 / 64) :=
begin
  sorry
end

end no_two_heads_consecutively_probability_l641_641288


namespace find_x_l641_641105

theorem find_x (n : ℕ) (a : Fin (2 * n) → ℤ) (h_distinct : Function.Injective a) :
  ∀ x : ℤ, (∏ i in Finset.finRange (2 * n), (x - a i)) = (-1)^n * Nat.factorial n * Nat.factorial n ↔ 
  x = ∑ i in Finset.finRange (2 * n), a i / (2 * n : ℤ) :=
by sorry

end find_x_l641_641105


namespace arithmetic_sequence_count_l641_641457

-- Define the initial conditions
def a1 : ℤ := -3
def d : ℤ := 3
def an : ℤ := 45

-- Proposition stating the number of terms n in the arithmetic sequence
theorem arithmetic_sequence_count :
  ∃ n : ℕ, an = a1 + (n - 1) * d ∧ n = 17 :=
by
  -- Skip the proof
  sorry

end arithmetic_sequence_count_l641_641457


namespace train_crossing_time_l641_641326

theorem train_crossing_time (train_length : ℝ) (man_speed : ℝ) (train_speed : ℝ) :
  train_length = 100 → man_speed = 5 → train_speed = 54.99520038396929 → 
  ∃ t, real.abs (t - 6.0) < 1e-10 :=
by
  intros h1 h2 h3
  sorry

end train_crossing_time_l641_641326


namespace ways_to_arrange_animals_l641_641145

theorem ways_to_arrange_animals (chickens dogs cats : ℕ) (animals : ℕ) : 
  chickens = 5 → dogs = 2 → cats = 4 → animals = 11 → 
  (∃ n, n = 3! * 5! * 2! * 4! ∧ n = 34560) :=
by
  -- problem setup
  intro h_chickens h_dogs h_cats h_animals
  -- correct answer check
  use 3! * 5! * 2! * 4!
  -- finalize answer is expected result
  constructor
  · rfl
  · norm_num
  sorry

end ways_to_arrange_animals_l641_641145


namespace probabilities_sum_of_dice_div_4_l641_641664

theorem probabilities_sum_of_dice_div_4 :
  let P0 := 1/4 in
  let P1 := 2/9 in
  let P2 := 1/4 in
  let P3 := 5/18 in
  2 * P3 - 3 * P2 + P1 - P0 = -2/9 :=
by
  sorry

end probabilities_sum_of_dice_div_4_l641_641664


namespace quadrilateral_PS_length_l641_641944

noncomputable def length_of_PS (PQ QR RS : ℝ) (sin_R cos_Q : ℝ) (R_obtuse : Prop) : ℝ :=
  PS
  where PS = real.sqrt (PQ^2 + QR^2 + RS^2 + 2 * PQ * QR * (-cos_Q))

theorem quadrilateral_PS_length :
  ∀ (P Q R S : Type) (PQ QR RS : ℝ) (sin_R cos_Q : ℝ)
  (hPQ : PQ = 6)
  (hQR : QR = 7)
  (hRS : RS = 25)
  (hsin_R : sin_R = 4/5)
  (hcos_Q : cos_Q = -4/5)
  (R_obtuse : sin_R > 0),
  length_of_PS PQ QR RS sin_R cos_Q R_obtuse = real.sqrt 794 :=
by
  intros
  simp [length_of_PS, hPQ, hQR, hRS, hsin_R, hcos_Q, R_obtuse]
  sorry

end quadrilateral_PS_length_l641_641944


namespace different_genre_pairs_l641_641840

theorem different_genre_pairs :
  (∃ (m f b : ℕ), m = 4 ∧ f = 4 ∧ b = 4 ∧ (m + f + b = 12) 
   ∧ ((m * f + m * b + f * b) = 48)) :=
by
  use 4, 4, 4
  repeat {split}; try {rfl}
  sorry

end different_genre_pairs_l641_641840


namespace room_area_ratio_l641_641630

theorem room_area_ratio (total_squares overlapping_squares : ℕ) 
  (h_total : total_squares = 16) 
  (h_overlap : overlapping_squares = 4) : 
  total_squares / overlapping_squares = 4 := 
by 
  sorry

end room_area_ratio_l641_641630


namespace circle_intersects_range_l641_641493

theorem circle_intersects_range {x y a : ℝ} (h1 : x^2 + y^2 = 4) (h2 : y = x - 3) (h3 : (a, a - 3)) :
  (∀ P : ℝ × ℝ, (P = (a, a - 3) → ∃ r : ℝ, r = 1 → (x - a)^2 + (y - (a - 3))^2 = 1)) →
  0 ≤ a ∧ a ≤ 3 :=
by
  -- In this placeholder, the proof would be constructed
  sorry

end circle_intersects_range_l641_641493


namespace max_product_l641_641222

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641222


namespace hall_marriage_theorem_l641_641694

variables {V : Type} [Finite V]
variables {A B : Set V} {E : Set (V × V)}

def is_bipartite (A B : Set V) (E : Set (V × V)) := 
  ∀ ⦃x y⦄, (x, y) ∈ E → (x ∈ A ∧ y ∈ B) ∨ (x ∈ B ∧ y ∈ A)

def neighbors (S : Set V) (E : Set (V × V)) : Set V :=
  {y | ∃ x ∈ S, (x, y) ∈ E}

def matching_saturates (A : Set V) (M : Set (V × V)) : Prop := 
  ∀ x ∈ A, ∃ y, (x, y) ∈ M ∨ (y, x) ∈ M

theorem hall_marriage_theorem :
  is_bipartite A B E →
  (∀ S ⊆ A, (neighbors S E).card ≥ S.card) ↔
  ∃ M, matching_saturates A M :=
sorry

end hall_marriage_theorem_l641_641694


namespace trajectory_eq_l641_641849

theorem trajectory_eq (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 6 → x^2 + y^2 + 2 * x + 2 * y - 3 = 0 → 
    ∃ p q : ℝ, p = a + 1 ∧ q = b + 1 ∧ (p * x + q * y = (a^2 + b^2 - 3)/2)) →
  a^2 + b^2 + 2 * a + 2 * b + 1 = 0 :=
by
  intros h
  sorry

end trajectory_eq_l641_641849


namespace part1_part2_l641_641501

-- Definitions and conditions
variables {a b c : ℝ} {A B C : ℝ}
variables {D : Prop} (conds1 : b^2 = a * c) (conds2 : BD * sin(β) = a * sin(γ))
          (conds3 : AD = 2 * DC)

-- Part 1: Prove BD = b
theorem part1 : BD = b :=
by sorry

-- Part 2: Prove cos(Angle B) = 7/12
theorem part2 (h1: BD = b): cos(β) = 7/12 :=
by sorry

end part1_part2_l641_641501


namespace sequence_expression_and_range_l641_641449

noncomputable def Sn (n : ℕ) (a : ℝ) : ℝ :=
  if n ≤ 4 then 2^n - 1 else -n^2 + (a-1) * n

noncomputable def an (n : ℕ) (a : ℝ) : ℝ := 
  if n = 1 then 1 
  else if n ≤ 4 then Sn n a - Sn (n-1) a 
  else if n = 5 then Sn 5 a - Sn 4 a 
  else Sn n a - Sn (n-1) a

theorem sequence_expression_and_range (a : ℝ) :
  (∀ n : ℕ, an n a = 
    if n = 1 then 1 
    else if n ≤ 4 then 2^(n-1) 
    else if n = 5 then 5 * a - 45 
    else -2 * n + a) 
  → (5 * a - 45 ≥ 8) 
  → (5 * a - 45 ≥ -12 + a) 
  → a ≥ 53 / 5 :=
begin
  intros h1 h2 h3,
  sorry
end

end sequence_expression_and_range_l641_641449


namespace rectangle_to_square_l641_641553

variable (k n : ℕ)

theorem rectangle_to_square (h1 : k > 5) (h2 : k * (k - 5) = n^2) : n = 6 := by 
  sorry

end rectangle_to_square_l641_641553


namespace values_at_2012_and_2013_l641_641424

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f(x + p) = f(x)

variables (f)
axiom odd_function : is_odd f
axiom f_neg1_eq_2 : f (-1) = 2
axiom period_4 : has_period f 4

theorem values_at_2012_and_2013 : f 2012 = 0 ∧ f 2013 = -2 :=
by
  sorry

end values_at_2012_and_2013_l641_641424


namespace trigonometric_identity_l641_641126

theorem trigonometric_identity (α : ℝ) :
  (1 + Real.cos (2 * α + 630 * Real.pi / 180) + Real.sin (2 * α + 810 * Real.pi / 180)) /
  (1 - Real.cos (2 * α - 630 * Real.pi / 180) + Real.sin (2 * α + 630 * Real.pi / 180))
  = Real.cot α := 
sorry

end trigonometric_identity_l641_641126


namespace monotonicity_intervals_m_range_l641_641820

open Real

-- Define the function f(x) given the parameter a ≠ 0
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * a * x - 1

-- Prove the monotonicity conditions
theorem monotonicity_intervals (a : ℝ) (h : a ≠ 0) :
  (∀ x, a < 0 → (3 * x^2 - 3 * a > 0)) ∧
  (∀ x, a > 0 → ((x < -sqrt a → 3 * x^2 - 3 * a > 0) ∧ (x > sqrt a → 3 * x^2 - 3 * a > 0) ∧ (-sqrt a < x ∧ x < sqrt a → 3 * x^2 - 3 * a < 0))) :=
sorry

-- Prove the range for m considering extremum at x = -1, a = 1
theorem m_range (m : ℝ) : (∀ x, 1 > 0 → (1 = 1 → (f 1 x)' = 0) → (x = -1 ∨ x = 1 → (m > -3 ∧ m < 1))) :=
sorry

end monotonicity_intervals_m_range_l641_641820


namespace three_digit_concat_div_by_37_l641_641936

theorem three_digit_concat_div_by_37 :
  ∀ (a : List ℕ), 
  (∀ n ∈ a, 111 ≤ n ∧ n ≤ 999) ∧ 
  (∀ i j, i ≠ j → ¬(nth a i = nth a j)) → 
    (37 ∣ list_sum (list_map_with_index (λ i v, v * 10^(3*i)) a)) := 
by
  intros a ha hpairwise
  sorry

end three_digit_concat_div_by_37_l641_641936


namespace sum_first_10_terms_b_n_l641_641613

-- Define the sequence a_n
def a_n (n : ℕ) : ℚ := n^2 + 3*n + 2

-- Define the sequence of interest
def b_n (n : ℕ) : ℚ := 1 / a_n n

-- Prove the sum of the first 10 terms of b_n
theorem sum_first_10_terms_b_n : (∑ i in finset.range 10, b_n (i + 1)) = 5 / 12 := by
  sorry

end sum_first_10_terms_b_n_l641_641613


namespace line_KL_contains_O_l641_641567

variables {A B C D E F K L : Type*} [ordered_comm_ring Type*]
variables (circle : set (Type*)) {O : Type*}

-- Definitions and conditions
def quadrilateral_inscribed (A B C D O : Type*) : Prop := 
sorry  -- Placeholder for the definition that quadrilateral ABCD is inscribed in circle with center O

def midpoint_arc (x y m : Type*) (circle : set (Type*)) : Prop := 
sorry  -- Placeholder for the definition that m is the midpoint of the arc from x to y not containing the other vertices

def parallel_diagonals (E F K L : Type*) (AC BD : Type*) : Prop := 
sorry  -- Placeholder for definition of lines through E and F parallel to AC and BD intersect at K and L respectively

-- Main theorem
theorem line_KL_contains_O 
  (h_inscribed : quadrilateral_inscribed A B C D O)
  (h_midpoints_AB : midpoint_arc A B E circle)
  (h_midpoints_CD : midpoint_arc C D F circle)
  (h_parallel : parallel_diagonals E F K L (A*C) (B*D)) : 
  ∃ O, ∃ KL, KL = line_through K L ∧ O ∈ KL :=
sorry  -- Proof placeholder

end line_KL_contains_O_l641_641567


namespace no_two_consecutive_heads_probability_l641_641308

theorem no_two_consecutive_heads_probability : 
  let total_outcomes := 2 ^ 10 in
  let favorable_outcomes := 
    (∑ n in range 6, nat.choose (10 - n - 1) n) in
  (favorable_outcomes / total_outcomes : ℚ) = 9 / 64 :=
by
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 1 + 10 + 36 + 56 + 35 + 6
  have h_total: total_outcomes = 1024 := by norm_num
  have h_favorable: favorable_outcomes = 144 := by norm_num
  have h_fraction: (favorable_outcomes : ℚ) / total_outcomes = 9 / 64 := by norm_num / [total_outcomes, favorable_outcomes]
  exact h_fraction

end no_two_consecutive_heads_probability_l641_641308


namespace hyperbola_eccentricity_l641_641961

noncomputable def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 + (b^2) / (a^2))

theorem hyperbola_eccentricity :
  (eccentricity (real.sqrt 2) (real.sqrt 6)) = 2 := by
sorry

end hyperbola_eccentricity_l641_641961


namespace factorial_fraction_l641_641341

theorem factorial_fraction : (11! / (8! * 3!)) = 165 := by
  sorry

end factorial_fraction_l641_641341


namespace simplify_expression_l641_641574

theorem simplify_expression : 
    (sqrt (sqrt (4 : ℝ) ((1 / (65536 : ℝ)) ^ 2)) = 1 / 16) :=
    sorry

end simplify_expression_l641_641574


namespace proof_triangle_tangent_l641_641562

open Real

def isCongruentAngles (ω : ℝ) := 
  let a := 15
  let b := 18
  let c := 21
  ∃ (x y z : ℝ), 
  (y^2 = x^2 + a^2 - 2 * a * x * cos ω) 
  ∧ (z^2 = y^2 + b^2 - 2 * b * y * cos ω)
  ∧ (x^2 = z^2 + c^2 - 2 * c * z * cos ω)

def isTriangleABCWithSides (AB BC CA : ℝ) (ω : ℝ) (tan_ω : ℝ) : Prop := 
  (AB = 15) ∧ (BC = 18) ∧ (CA = 21) ∧ isCongruentAngles ω 
  ∧ tan ω = tan_ω

theorem proof_triangle_tangent : isTriangleABCWithSides 15 18 21 ω (88/165) := 
by
  sorry

end proof_triangle_tangent_l641_641562


namespace count_multiples_5_or_7_but_not_8_l641_641459

def is_multiple (n m : ℕ) : Prop := n % m = 0

def num_satisfy_conditions : ℕ :=
  (Finset.filter (λ n => (is_multiple n 5 ∨ is_multiple n 7) ∧ ¬is_multiple n 8) (Finset.range 101)).card

theorem count_multiples_5_or_7_but_not_8 : num_satisfy_conditions = 29 :=
by
  -- This is a placeholder for the proof
  sorry

end count_multiples_5_or_7_but_not_8_l641_641459


namespace minimum_value_of_weighted_sum_l641_641521

theorem minimum_value_of_weighted_sum 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) :
  3 * a + 6 * b + 9 * c ≥ 54 :=
sorry

end minimum_value_of_weighted_sum_l641_641521


namespace sheep_color_unchangeable_l641_641548

def initial_counts : ℕ × ℕ × ℕ := (22, 18, 15)

def allowed_transformations (b r g : ℕ) : list (ℕ × ℕ × ℕ) :=
  [ (b - 1, r - 1, g + 2), (b - 1, r + 2, g - 1), (b + 2, r - 1, g - 1) ]

def invariant (r g : ℕ) : ℕ := (r - g) % 3

theorem sheep_color_unchangeable (b r g : ℕ) :
  invariant r g = 0 ∧ (b, r, g) = initial_counts → 
  ∃ n, n = b + r + g → ∀ (b' r' g' : ℕ), 
    (b', r', g') ∈ (iterate allowed_transformations (b, r, g) _) →
    b' = n ∧ r' = 0 ∧ g' = 0 :=
by
  sorry

end sheep_color_unchangeable_l641_641548


namespace lily_hops_distance_after_four_hops_l641_641114

theorem lily_hops_distance_after_four_hops :
  let hop_distance : ℕ → ℚ := λ k, (3 / 4) ^ (k - 1) * (1 / 4)
  let total_distance : ℚ := (list.range 4).sum hop_distance
  total_distance = 175 / 256 
:= by
  sorry

end lily_hops_distance_after_four_hops_l641_641114


namespace probability_prime_multiple_of_5_l641_641560

noncomputable def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def is_multiple_of_5 (n : Nat) : Prop :=
  n % 5 = 0

theorem probability_prime_multiple_of_5 :
  let favorable_outcomes := { n : Nat | 1 ≤ n ∧ n ≤ 100 ∧ is_prime n ∧ is_multiple_of_5 n }.card
  let possible_outcomes := 100
  (favorable_outcomes : ℚ) / possible_outcomes = 1 / 100 :=
by
  sorry

end probability_prime_multiple_of_5_l641_641560


namespace volume_of_solid_of_revolution_l641_641747

theorem volume_of_solid_of_revolution (a : ℝ) : 
  let m := (a * Real.sqrt 3) / 2,
      G := (2 / 3) * m,
      s := 2 * Real.pi * G,
      A := (1 / 2) * a * m in
  (A * s) = (Real.pi * a^3 / 2) := by 
  sorry

end volume_of_solid_of_revolution_l641_641747


namespace no_two_heads_consecutively_probability_l641_641289

theorem no_two_heads_consecutively_probability :
  (∃ (total_sequences : ℕ) (valid_sequences : ℕ),
    total_sequences = 2^10 ∧ valid_sequences = 1 + 10 + 36 + 56 + 35 + 6 ∧
    (valid_sequences / total_sequences : ℚ) = 9 / 64) :=
begin
  sorry
end

end no_two_heads_consecutively_probability_l641_641289


namespace binders_can_bind_books_l641_641847

theorem binders_can_bind_books :
  (∀ (binders books days : ℕ), binders * days * books = 18 * 10 * 900 → 
    11 * binders * 12 = 660) :=
sorry

end binders_can_bind_books_l641_641847


namespace ratio_of_areas_l641_641184

-- Definitions based on the conditions given
variables (A B M N P Q O : Type) 
variables (AB BM BP : ℝ)

-- Assumptions
axiom hAB : AB = 6
axiom hBM : BM = 9
axiom hBP : BP = 5

-- Theorem statement
theorem ratio_of_areas (hMN : M ≠ N) (hPQ : P ≠ Q) :
  (1 / 121 : ℝ) = sorry :=
by sorry

end ratio_of_areas_l641_641184


namespace range_of_m_l641_641476

-- Definitions
def line1 (x : ℝ) := (1/2) * x - 1
def line2 (k x : ℝ) := k * x + 3 * k + 1

-- Theorem statement
theorem range_of_m (m : ℝ) (k : ℝ) (n : ℝ) 
  (h_intersect : line1 m = line2 k m)
  (h_decreasing : k < 0) : -3 < m ∧ m < 4 := 
sorry

end range_of_m_l641_641476


namespace both_firms_participate_social_optimality_l641_641065

variables (α V IC : ℝ)

-- Conditions definitions
def expected_income_if_both_participate (α V : ℝ) : ℝ :=
  α * (1 - α) * V + 0.5 * (α^2) * V

def condition_for_both_participation (α V IC : ℝ) : Prop :=
  expected_income_if_both_participate α V - IC ≥ 0

-- Values for specific case
noncomputable def V_specific : ℝ := 24
noncomputable def α_specific : ℝ := 0.5
noncomputable def IC_specific : ℝ := 7

-- Proof problem statement
theorem both_firms_participate : condition_for_both_participation α_specific V_specific IC_specific := by
  sorry

-- Definitions for social welfare considerations
def total_profit_if_both_participate (α V IC : ℝ) : ℝ :=
  2 * (expected_income_if_both_participate α V - IC)

def expected_income_if_one_participates (α V IC : ℝ) : ℝ :=
  α * V - IC

def social_optimal (α V IC : ℝ) : Prop :=
  total_profit_if_both_participate α V IC < expected_income_if_one_participates α V IC

theorem social_optimality : social_optimal α_specific V_specific IC_specific := by
  sorry

end both_firms_participate_social_optimality_l641_641065


namespace dispatch_plans_count_l641_641317

theorem dispatch_plans_count :
  let officials := {a, b, c, d, e, f, g, h} -- assuming these are the 8 officials
  let males := {a, b, c, d, e}
  let females := {f, g, h}
  ∃ groups : officials → bool,
    (∀ g, (card (groups g = true)) ≥ 3 ∧ (card (groups g = false)) ≥ 3) ∧
    (∃ g, (males ∩ groups g ≠ ∅)) ∧ (∃ g, (males ∩ groups (¬g) ≠ ∅)) ∧
    (card {g | groups g} = 2) →
  (count dispatch_plans = 180)
:= sorry

end dispatch_plans_count_l641_641317


namespace g_at_1_is_0_sum_g_from_1_to_2023_is_0_l641_641793

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom g_odd : ∀ x, g (-x) = -g x
axiom f_plus_g_eq_two : ∀ x, f x + g x = 2
axiom f_plus_g_shifted_eq_two : ∀ x, f x + g (x - 2) = 2

theorem g_at_1_is_0 : g 1 = 0 :=
by
  sorry
  
theorem sum_g_from_1_to_2023_is_0 : (∑ i in finset.range 2023, g (i + 1)) = 0 :=
by
  sorry

end g_at_1_is_0_sum_g_from_1_to_2023_is_0_l641_641793


namespace polynomial_has_root_l641_641387

noncomputable def polynomial : Polynomial ℚ :=
  Polynomial.X^4 - 10 * Polynomial.X^2 + 1

theorem polynomial_has_root : polynomial.eval (Real.sqrt 2 + Real.sqrt 3) polynomial = 0 := sorry

end polynomial_has_root_l641_641387


namespace conjugate_z_l641_641019

open Complex

def z : ℂ := 2 / (1 - I)

theorem conjugate_z : conj(z) = 1 - I := by
  sorry

end conjugate_z_l641_641019


namespace polynomial_exists_derangement_l641_641902

/-- Given a polynomial Q(x) with integer coefficients, prove there exists a polynomial P(x) 
    with integer coefficients such that for every integer n >= deg Q, 
    ∑ i = 0 to n (!i P(i)) / (i!(n-i)!) = Q(n) -/
theorem polynomial_exists_derangement (Q : Polynomial ℤ) : 
  ∃ P : Polynomial ℤ, ∀ (n : ℕ), n ≥ Q.natDegree → 
  ∑ i in Finset.range (n+1), (nat.derangements i * P.eval i) / (Nat.factorial i * Nat.factorial (n - i)) = Q.eval n :=
sorry

end polynomial_exists_derangement_l641_641902


namespace compute_XY_l641_641900

theorem compute_XY (BC AC AB : ℝ) (hBC : BC = 30) (hAC : AC = 50) (hAB : AB = 60) :
  let XA := (BC * AB) / AC 
  let AY := (BC * AC) / AB
  let XY := XA + AY
  XY = 61 :=
by
  sorry

end compute_XY_l641_641900


namespace eight_people_knaves_all_nine_people_knaves_three_l641_641354

-- Definitions and conditions for the first problem (8 people)
def num_people_8 := 8

def table_8 (P : ℕ → Prop) :=
  ∀ i, P i ∧ P ((i + 1) % num_people_8)

-- Prove that all participants are liars (knaves)
theorem eight_people_knaves_all (P : ℕ → Prop) (knight knave : Prop) [DecidablePred knight] :
  (∀ i, P i ↔ (P ((i + 1) % num_people_8) =/= P ((i + 2) % num_people_8))) →
  (∀ i, P i = knave) :=
sorry

-- Definitions and conditions for the second problem (9 people)
def num_people_9 := 9

def table_9 (P : ℕ → Prop) :=
  ∀ i, P i ∧ P ((i + 1) % num_people_9)

-- Prove that exactly three participants are liars (knaves)
theorem nine_people_knaves_three (P : ℕ → Prop) (knight knave : Prop) [DecidablePred knight] :
  (∀ i, P i ↔ (P ((i + 1) % num_people_9) =/= P ((i + 2) % num_people_9))) →
  (∃ (knaves : Finset ℕ), knaves.card = 3 ∧ ∀ i, (i ∈ knaves ↔ P i = knave)) :=
sorry

end eight_people_knaves_all_nine_people_knaves_three_l641_641354


namespace find_constant_c_l641_641470

noncomputable def poly (c : ℝ) (x : ℝ) := c * x^3 + 23 * x^2 - 5 * c * x + 55

theorem find_constant_c (c : ℝ) :
  (∃ p : ℝ → ℝ, poly c = λ x, (x - 5) * p x) → c = -6.3 :=
by
  sorry

end find_constant_c_l641_641470


namespace solve_for_x_l641_641461

noncomputable theory

def log_base (b x : ℝ) := Real.log x / Real.log b

def problem_condition (x : ℝ) : Prop :=
  log_base 3 (x^3) + log_base (1/3) x - log_base 3 9 = 7

theorem solve_for_x (x : ℝ) (h : problem_condition x) : x = 3^4.5 :=
by
  sorry

end solve_for_x_l641_641461


namespace no_integer_solutions_l641_641746

theorem no_integer_solutions (m n : ℤ) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2011) :=
by sorry

end no_integer_solutions_l641_641746


namespace inequality_proof_l641_641903

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b + c = 1) :
  a * (1 + b - c) ^ (1 / 3) + b * (1 + c - a) ^ (1 / 3) + c * (1 + a - b) ^ (1 / 3) ≤ 1 := 
by
  sorry

end inequality_proof_l641_641903


namespace points_in_half_plane_l641_641395

-- Define the points and the circle
variables (P B D : Point) (r : ℝ)

-- Assumptions based on the problem statement
-- P is the center of circle C with radius r
-- B is a point on the circumference of C
-- D is the point diametrically opposite to B

-- Define the circle C
def circle (P : Point) (r : ℝ) := { Q : Point | dist P Q = r }

-- Define the condition that D is diametrically opposite to B
def diametrically_opposite (P B D : Point) : Prop := dist P B = r ∧ dist P D = r ∧ dist B D = 2 * r

-- The set of points A such that dist A B < dist A D
def desired_region (P B D : Point) : set Point :=
  { A : Point | dist A B < dist A D }

-- The theorem statement
theorem points_in_half_plane (P B D : Point) (r : ℝ) 
  (h_circle : circle P r)
  (h_diam_opposite : diametrically_opposite P B D) :
  desired_region P B D = { A : Point | in_half_plane (perp_bisector B D) B A } :=
sorry

end points_in_half_plane_l641_641395


namespace collinear_and_bisector_l641_641519

open Real EuclideanSpace

def vec_a : EuclideanSpace ℝ (Fin 3) := ![8, -3, -5]
def vec_c : EuclideanSpace ℝ (Fin 3) := ![-1, -2, 3]
def vec_b : EuclideanSpace ℝ (Fin 3) := ![329/20, 37/60, 251/45]

theorem collinear_and_bisector :
  ∃ t : ℝ, vec_b = vec_a + t • (vec_c - vec_a) ∧
            (vec_a ⬝ vec_b) / (∥ vec_a ∥ * ∥ vec_b ∥) = (vec_b ⬝ vec_c) / (∥ vec_b ∥ * ∥ vec_c ∥) :=
sorry

end collinear_and_bisector_l641_641519


namespace xy_sum_l641_641844

theorem xy_sum (x y : ℝ) (h1 : sqrt (y - 5) = 5) (h2 : 2 ^ x = 8) : x + y = 33 := by
  sorry

end xy_sum_l641_641844


namespace correct_answers_for_prize_l641_641868

theorem correct_answers_for_prize :
  ∀ (total_questions correct_points incorrect_points prize_points x : ℕ),
    total_questions = 30 →
    correct_points = 4 →
    incorrect_points = -2 →
    prize_points = 60 →
    (correct_points * x + incorrect_points * (total_questions - x) ≥ prize_points) ↔ (x ≥ 20) :=
by
  intros total_questions correct_points incorrect_points prize_points x
  intros h_total h_correct h_incorrect h_prize_points
  rw [h_total, h_correct, h_incorrect, h_prize_points]
  sorry

end correct_answers_for_prize_l641_641868


namespace justify_misha_decision_l641_641653

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l641_641653


namespace slant_asymptote_sum_eq_14_l641_641448

noncomputable def slant_asymptote_sum (y : ℝ → ℝ) := 
  ∃ m b : ℝ, (∀ x:ℝ, y x = m * x + b + (10 / (x - 2))) → (x → ∞ or x → -∞) means (10 / (x - 2)) → 0

theorem slant_asymptote_sum_eq_14 : 
  slant_asymptote_sum (λ x => (3 * x^2 + 5 * x - 12) / (x - 2)) := 
begin
  sorry
end

end slant_asymptote_sum_eq_14_l641_641448


namespace cos_theta_value_l641_641808

theorem cos_theta_value (θ : ℝ) (h : ∀ x : ℝ, f x ≤ f θ) : 
  cos θ = -√5 / 5 :=
by
  let f := λ x : ℝ, sin x - 2 * cos x
  sorry

end cos_theta_value_l641_641808


namespace find_length_of_FC_l641_641780

theorem find_length_of_FC (DC CB AD AB ED FC : ℝ) (h1 : DC = 9) (h2 : CB = 10) (h3 : AB = (1 / 3) * AD) (h4 : ED = (2 / 3) * AD) : 
  FC = 13 := by
  sorry

end find_length_of_FC_l641_641780


namespace triangle_area_given_conditions_l641_641848

theorem triangle_area_given_conditions
    (CD : ℝ)
    (angle_BAC : ℝ)
    (h_CD : CD = Real.sqrt 2)
    (h_angle_BAC : angle_BAC = Real.pi / 4) :
    let AC := 2
    let BC := 2
    let area := (1 / 2) * AC * BC in
    area = 2 := 
by
    sorry

end triangle_area_given_conditions_l641_641848


namespace pool_pumping_problem_l641_641324

noncomputable def time_to_pump := (300 : ℕ) / (20 : ℕ)
noncomputable def total_capacity (C : ℝ) := 0.70 * C + 300 == 0.80 * C

theorem pool_pumping_problem (C : ℝ) (hC_eq: 0.70 * C + 300 = 0.80 * C) : 
  time_to_pump = 15 ∧ C = 3000 := 
by {
  have time_correct : time_to_pump = 15 := by {
    -- calculation steps skipped, assert directly
    sorry
  },
  have capacity_correct : C = 3000 := by {
    -- Use the provided equation to solve for C
    have h : 300 = 0.10 * C := by {
      rw [← sub_eq_of_eq_add hC_eq, mul_sub, ← sub_eq_zero] at hC_eq,
      exact hC_eq,
    },
    have h₁ : 0.10 * C = 300 := eq.symm h,
    have h₂ : C = 3000 := by linarith,
    exact h₂,
  },
  exact ⟨time_correct, capacity_correct⟩,
}

end pool_pumping_problem_l641_641324


namespace customers_not_caught_percentage_l641_641482

def percentage_customers_caught : ℝ := 22 / 100
def total_percentage_customers_sampling : ℝ := 27.5 / 100

theorem customers_not_caught_percentage : 
  total_percentage_customers_sampling = percentage_customers_caught + 5.5 / 100 :=
by sorry

end customers_not_caught_percentage_l641_641482


namespace num_degree_5_polynomials_to_permutation_l641_641008

theorem num_degree_5_polynomials_to_permutation :
  ∃ f : (ℝ[X] → finset (fin 6)), 
    (∀ (p : ℝ[X]), p.degree = 5 → ((range 1 7).image (λ x, p.eval x)) = {1, 2, 3, 4, 5, 6} → (f p).card = 714) :=
sorry

end num_degree_5_polynomials_to_permutation_l641_641008


namespace scooter_gain_percent_l641_641136

theorem scooter_gain_percent 
  (purchase_price : ℕ) 
  (repair_costs : ℕ) 
  (selling_price : ℕ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by
  sorry

end scooter_gain_percent_l641_641136


namespace father_current_age_l641_641273

variable (M F : ℕ)

/-- The man's current age is (2 / 5) of the age of his father. -/
axiom man_age : M = (2 / 5) * F

/-- After 12 years, the man's age will be (1 / 2) of his father's age. -/
axiom age_relation_in_12_years : (M + 12) = (1 / 2) * (F + 12)

/-- Prove that the father's current age, F, is 60. -/
theorem father_current_age : F = 60 :=
by
  sorry

end father_current_age_l641_641273


namespace complement_of_union_l641_641534

def U := {1, 2, 3, 4, 5}
def A := {1, 3}
def B := {3, 5}
def C := {2, 4}

theorem complement_of_union (hU : U = {1, 2, 3, 4, 5})
                            (hA : A = {1, 3})
                            (hB : B = {3, 5}) : 
  {x | x ∉ A ∪ B} = C :=
by
  sorry

end complement_of_union_l641_641534


namespace find_c_non_zero_constant_max_f_sequence_value_l641_641875

-- Definitions based on given conditions
def a_sequence (n : ℕ) : ℤ := 2 * n - 1
def b_sequence (n : ℕ) : ℤ := 2 ^ (n - 1)

def C_sequence (n : ℕ) : ℤ := a_sequence n + int.log (sqrt 2) (b_sequence n)
def T_sequence (n : ℕ) : ℤ := (2 * n ^ 2) - n
noncomputable def d_sequence (n : ℕ) (c : ℤ) : ℤ := T_sequence n / (n + c)

-- Proving the non-zero constant c
theorem find_c_non_zero_constant (c : ℤ) : 
  (d_sequence 2 c) * 2 = ((d_sequence 1 c) + (d_sequence 3 c)) → c = -1/2 :=
sorry

-- Proving the maximum value of the term in f(n) sequence
def f_sequence (n : ℕ) (c : ℤ) : ℚ := 
  d_sequence n c / ((n + 36) * d_sequence (n + 1) c)

theorem max_f_sequence_value (c : ℤ) : 
  (∃ n : ℕ, f_sequence n c = 1 / 49) :=
sorry

end find_c_non_zero_constant_max_f_sequence_value_l641_641875


namespace Elza_winning_strategy_l641_641750

-- Define a hypothetical graph structure
noncomputable def cities := {i : ℕ // 1 ≤ i ∧ i ≤ 2013}
def connected (c1 c2 : cities) : Prop := sorry

theorem Elza_winning_strategy 
  (N : ℕ) 
  (roads : (cities × cities) → Prop) 
  (h1 : ∀ c1 c2, roads (c1, c2) → connected c1 c2)
  (h2 : N = 1006): 
  ∃ (strategy : cities → Prop), 
  (∃ c1 c2 : cities, (strategy c1 ∧ strategy c2)) ∧ connected c1 c2 :=
by 
  sorry

end Elza_winning_strategy_l641_641750


namespace max_product_of_sum_2000_l641_641244

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641244


namespace probability_either_A1_or_B1_not_both_is_half_l641_641170

-- Definitions of the students
inductive Student
| A : ℕ → Student
| B : ℕ → Student
| C : ℕ → Student

-- Excellent grades students
def math_students := [Student.A 1, Student.A 2, Student.A 3]
def physics_students := [Student.B 1, Student.B 2]
def chemistry_students := [Student.C 1, Student.C 2]

-- Total number of ways to select one student from each category
def total_ways : ℕ := 3 * 2 * 2

-- Number of ways either A_1 or B_1 is selected but not both
def special_ways : ℕ := 1 * 1 * 2 + 2 * 1 * 2

-- Probability calculation
def probability := (special_ways : ℚ) / total_ways

-- Theorem to be proven
theorem probability_either_A1_or_B1_not_both_is_half :
  probability = 1 / 2 := by
  sorry

end probability_either_A1_or_B1_not_both_is_half_l641_641170


namespace rafael_earnings_l641_641132

theorem rafael_earnings 
  (hours_monday : ℕ) 
  (hours_tuesday : ℕ) 
  (hours_left : ℕ) 
  (rate_per_hour : ℕ) 
  (h_monday : hours_monday = 10) 
  (h_tuesday : hours_tuesday = 8) 
  (h_left : hours_left = 20) 
  (h_rate : rate_per_hour = 20) : 
  (hours_monday + hours_tuesday + hours_left) * rate_per_hour = 760 := 
by
  sorry

end rafael_earnings_l641_641132


namespace relationship_between_P_and_Q_l641_641781

-- Define the sets P and Q
def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem relationship_between_P_and_Q : P ⊇ Q :=
by
  sorry

end relationship_between_P_and_Q_l641_641781


namespace total_dividends_correct_l641_641657

-- Conditions
def net_profit (total_income expenses loan_penalty_rate : ℝ) : ℝ :=
  let net1 := total_income - expenses
  let loan_penalty := net1 * loan_penalty_rate
  net1 - loan_penalty

def total_loan_payments (monthly_payment months additional_payment : ℝ) : ℝ :=
  (monthly_payment * months) - additional_payment

def dividend_per_share (net_profit total_loan_payments num_shares : ℝ) : ℝ :=
  (net_profit - total_loan_payments) / num_shares

noncomputable def total_dividends_director (dividend_per_share shares_owned : ℝ) : ℝ :=
  dividend_per_share * shares_owned

theorem total_dividends_correct :
  total_dividends_director (dividend_per_share (net_profit 1500000 674992 0.2) (total_loan_payments 23914 12 74992) 1000) 550 = 246400 :=
sorry

end total_dividends_correct_l641_641657


namespace all_sheep_can_be_one_color_l641_641550

theorem all_sheep_can_be_one_color (b r v : ℕ) (h_initial : b = 22 ∧ r = 18 ∧ v = 15) 
(∀ (b' r' v' : ℕ), (b' = b - 1 ∧ r' = r - 1 ∧ v' = v + 2) ∨ (b' = b - 1 ∧ r' = r + 2 ∧ v' = v - 1) ∨ (b' = b + 2 ∧ r' = r - 1 ∧ v' = v - 1)) :
∃ (b r v : ℕ), (r = 0 ∧ v = 0 ∧ b = 55) :=
sorry

end all_sheep_can_be_one_color_l641_641550


namespace isosceles_triangle_area_theorem_l641_641685

def isosceles_triangle_area (b h : ℝ) : Prop :=
  (b > 0 ∧ h > 0) → area_of_isosceles_triangle b h = b * h / 2

theorem isosceles_triangle_area_theorem (b h : ℝ) (hb : b > 0) (hh : h > 0) :
  isosceles_triangle_area b h :=
by
  unfold isosceles_triangle_area
  intros _ _
  linarith

end isosceles_triangle_area_theorem_l641_641685


namespace max_radius_of_circle_l641_641679

theorem max_radius_of_circle (c : ℝ × ℝ → Prop) (h1 : c (16, 0)) (h2 : c (-16, 0)) :
  ∃ r : ℝ, r = 16 :=
by
  sorry

end max_radius_of_circle_l641_641679


namespace area_of_square_is_80_l641_641148

noncomputable def circle_equation : Prop :=
  ∀ (x y : ℝ), 2 * x^2 = -2 * y^2 + 12 * x - 4 * y + 20

noncomputable def is_square_inscribed (x y : ℝ) : Prop :=
  ∃ (s : ℝ), s > 0 ∧ 2 * x^2 = -2 * y^2 + 12 * x - 4 * y + 20 ∧
  (s^(2 : ℕ)) = 80 ∧ 
  (circle_equation x y)

theorem area_of_square_is_80 :
  ∃ (s : ℝ), s > 0 ∧ s^2 = 80 :=
by
  sorry

end area_of_square_is_80_l641_641148


namespace part_I_part_II_case1_part_II_case2_part_II_case3_l641_641440

-- Define f and g
def f (x : ℝ) : ℝ := (1 / 2) * x^2 - (1 / 2)
def g (a x : ℝ) : ℝ := a * Real.log x

-- Proof of the first part
theorem part_I (a : ℝ) : (f 1 = 0 ∧ g a 1 = 0 ∧ deriv f 1 = deriv (g a) 1) → a = 1 :=
by
  -- Proof would follow verifying given conditions imply a = 1
  sorry

-- Define F
def F (x m a : ℝ) : ℝ := f x - m * g a x

-- Analyzing F for three cases of m
theorem part_II_case1 : ∀ (m : ℝ), m ≤ 1 → (∀ x ∈ Icc 1 Real.exp, F x m 1 ≥ 0) :=
by
  -- Proof would show F is minimized at 1 for m ≤ 1
  sorry

theorem part_II_case2 (m : ℝ) : 1 < m ∧ m < Real.exp^2 → (∀ x ∈ Icc 1 Real.exp, F x m 1 ≥ F (Real.sqrt m) m 1) :=
by
  -- Proof would show F is minimized at sqrt(m) for 1 < m < e^2
  sorry

theorem part_II_case3 (m : ℝ) : m ≥ Real.exp^2 → (∀ x ∈ Icc 1 Real.exp, F x m 1 ≥ F Real.exp m 1) :=
by
  -- Proof would show F is minimized at e for m ≥ e^2
  sorry

end part_I_part_II_case1_part_II_case2_part_II_case3_l641_641440


namespace min_value_expression_l641_641016

theorem min_value_expression (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) : (x * y + x^2) ≥ 4 :=
sorry

end min_value_expression_l641_641016


namespace determinant_trig_matrix_eq_one_l641_641751

theorem determinant_trig_matrix_eq_one (α θ : ℝ) :
  Matrix.det ![
  ![Real.cos α * Real.cos θ, Real.cos α * Real.sin θ, Real.sin α],
  ![Real.sin θ, -Real.cos θ, 0],
  ![Real.sin α * Real.cos θ, Real.sin α * Real.sin θ, -Real.cos α]
  ] = 1 :=
by
  sorry

end determinant_trig_matrix_eq_one_l641_641751


namespace lipschitz_condition_main_theorem_l641_641347

noncomputable def mean {n : ℕ} (a : Fin n → ℝ) : ℝ :=
  (∑ i, a i) / n

theorem lipschitz_condition (f : ℝ → ℝ) (L : ℝ) : Prop :=
  ∀ x y, |f x - f y| ≤ L * |x - y|

theorem main_theorem (n : ℕ) (a : Fin n → ℝ) (f : ℝ → ℝ) (x : ℝ) 
  (h1 : n ≥ 1) (h2 : ∀ i, a i ∈ Icc (-1 : ℝ) 1) (h3 : (∑ i, a i) = 0)
  (h4 : lipschitz_condition f 1) (hx : x ∈ Icc (-1 : ℝ) 1) :
  |f x - mean a f| ≤ 1 :=
sorry

end lipschitz_condition_main_theorem_l641_641347


namespace quadratic_inequality_solution_l641_641138

theorem quadratic_inequality_solution : ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l641_641138


namespace termite_ridden_fraction_l641_641926

theorem termite_ridden_fraction (T : ℝ)
  (h1 : (3 / 10) * T = 0.1) : T = 1 / 3 :=
by
  -- proof goes here
  sorry

end termite_ridden_fraction_l641_641926


namespace sin_cos_sum_l641_641843

theorem sin_cos_sum (α : ℝ) (h: sin α ^ 4 + cos α ^ 4 = 1) : sin α + cos α = 1 ∨ sin α + cos α = -1 := by
  sorry

end sin_cos_sum_l641_641843


namespace initial_temperature_is_20_l641_641505

-- Define the initial temperature, final temperature, rate of increase and time
def T_initial (T_final : ℕ) (rate_of_increase : ℕ) (time : ℕ) : ℕ :=
  T_final - rate_of_increase * time

-- Statement: The initial temperature is 20 degrees given the specified conditions.
theorem initial_temperature_is_20 :
  T_initial 100 5 16 = 20 :=
by
  sorry

end initial_temperature_is_20_l641_641505


namespace total_cookies_l641_641269

theorem total_cookies
  (num_bags : ℕ)
  (cookies_per_bag : ℕ)
  (h_num_bags : num_bags = 286)
  (h_cookies_per_bag : cookies_per_bag = 452) :
  num_bags * cookies_per_bag = 129272 :=
by
  sorry

end total_cookies_l641_641269


namespace hiker_speed_correct_l641_641707

noncomputable def hiker_speed_in_still_water : ℝ :=
  let v := 12.68
  v

theorem hiker_speed_correct (v c : ℝ) 
  (h1 : 14 * (v + c) = 250)
  (h2 : 16 * (v - c) = 120) : 
  v ≈ 12.68 :=
by sorry

end hiker_speed_correct_l641_641707


namespace initial_speed_is_11_point_2_l641_641726

-- Define a function to convert hours and minutes to fractional hours
def time_in_hours (hours : ℕ) (minutes : ℕ) : ℝ :=
  hours + minutes / 60.0

-- Define all the conditions
def departure_time : ℝ := time_in_hours 5 20
def initial_travel_time : ℝ := time_in_hours 2 15
def reduced_speed : ℝ := 60
def total_distance : ℝ := 350
def arrival_time : ℝ := time_in_hours 13 0.25

-- Calculate the remaining travel time
noncomputable def remaining_travel_time : ℝ := arrival_time - (departure_time + initial_travel_time)

-- Calculate the distance covered at reduced speed
noncomputable def distance_reduced_speed : ℝ := remaining_travel_time * reduced_speed

-- Calculate the distance covered at initial speed
noncomputable def distance_initial_speed : ℝ := total_distance - distance_reduced_speed

-- Calculate the initial speed
noncomputable def initial_speed : ℝ := distance_initial_speed / initial_travel_time

-- The proof problem statement
theorem initial_speed_is_11_point_2 : initial_speed = 11.2 := by
  sorry

end initial_speed_is_11_point_2_l641_641726


namespace vector_inequality_l641_641407

open Real

variables (a b c d : ℝ^n) -- Assuming vectors in ℝ^n for generality

-- Hypotheses
def vectors_non_parallel (a b c d : ℝ^n) : Prop := 
  ¬(a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d) ∧ 
  (∀ (u v : ℝ^n), u ≠ v → ¬u.is_scalar_multiple v)

theorem vector_inequality (h1 : a + b + c + d = 0) (h2 : vectors_non_parallel a b c d) : 
  |a| + |b| + |c| + |d| > |a + b| + |a + c| + |a + d| :=
by 
  sorry

end vector_inequality_l641_641407


namespace partition_communicating_pairs_l641_641619

-- Description of the conditions
variables (Representatives : Type) [Fintype Representatives] 
          (n : ℕ) 
          (h_rep_count : Fintype.card Representatives = 1000)
          (can_communicate : Representatives → Representatives → Representatives → Prop)
          (h_comm : ∀ (a b c : Representatives), can_communicate a b c)

-- Main theorem statement
theorem partition_communicating_pairs :
  ∃ (pairing : Representatives → Representatives), 
  (∀ r, ∃ s, pairing r = s ∧ ∀ (r₁ r₂ s : Representatives), 
  pairing r₁ = s → pairing r₂ = s → can_communicate r₁ r₂ s)
  ∧ (∀ (r s : Representatives), r ≠ s → pairing r ≠ pairing s) 
  ∧ (Fintype.card {p : Representatives // pairing p == p} = 500) :=
sorry

end partition_communicating_pairs_l641_641619


namespace no_two_heads_consecutively_probability_l641_641287

theorem no_two_heads_consecutively_probability :
  (∃ (total_sequences : ℕ) (valid_sequences : ℕ),
    total_sequences = 2^10 ∧ valid_sequences = 1 + 10 + 36 + 56 + 35 + 6 ∧
    (valid_sequences / total_sequences : ℚ) = 9 / 64) :=
begin
  sorry
end

end no_two_heads_consecutively_probability_l641_641287


namespace cos_squared_sum_l641_641344

theorem cos_squared_sum :
  ∑ k in Finset.range 18, Real.cos (10 * (k + 1) * Real.pi / 180)^2 = 8 := by
  sorry

end cos_squared_sum_l641_641344


namespace partition_bijection_l641_641102

-- Definitions from the problem conditions
def b (n : ℕ) := {parts : Multiset ℕ | ∀ p ∈ parts, ∃ k : ℕ, p = 2^k}.card
def c (n : ℕ) := 
  {parts : Multiset ℕ | 
    (∀ p ∈ parts, ∃ k : ℕ, p = 2^k) ∧ 
    (∀ m : ℕ, (m > 0 ∧ m ≤ n ∧ ∃ k, m = 2^k) → m ∈ parts)}.card

-- The goal to prove
theorem partition_bijection (n : ℕ) : b (n + 1) = 2 * c n :=
sorry

end partition_bijection_l641_641102


namespace sqrt_identity_l641_641280

variable {a b : ℝ}

theorem sqrt_identity (h : a^2 ≥ b) : 
  sqrt (a + sqrt b) = sqrt ((a + sqrt (a^2 - b)) / 2) + sqrt ((a - sqrt (a^2 - b)) / 2) ∧
  sqrt (a - sqrt b) = sqrt ((a + sqrt (a^2 - b)) / 2) - sqrt ((a - sqrt (a^2 - b)) / 2) := 
sorry

end sqrt_identity_l641_641280


namespace parabola_hyperbola_tangent_parallel_l641_641828

theorem parabola_hyperbola_tangent_parallel {p : ℝ} (hp : p > 0)
  (M : ℝ × ℝ)
  (hM : M.1 = - (p * sqrt 3 / 3) ∧ M.2 = -(p / 6))
  (focus1 : prod ℝ ℝ) (h_focus1 : focus1 = (0, -p / 2))
  (focus2 : prod ℝ ℝ) (h_focus2 : focus2 = (-2, 0))
  (line_eq : ∀ (x y : ℝ), y = -(p / 2) * x + p → y = -(sqrt 3 / 3) * x ∧ x = - p * sqrt 3 / 3)
  (tangent_parallel_asymptote : ∀ (x₀ : ℝ), (-x₀ / p = sqrt 3 / 3) → x₀ = -p * sqrt 3 / 3) :
  p = 4 * sqrt 3 / 3 :=
by
  sorry

end parabola_hyperbola_tangent_parallel_l641_641828


namespace no_consecutive_heads_probability_l641_641301

def prob_no_consecutive_heads (n : ℕ) : ℝ := 
  -- This is the probability function for no consecutive heads
  if n = 10 then (9 / 64) else 0

theorem no_consecutive_heads_probability :
  prob_no_consecutive_heads 10 = 9 / 64 :=
by
  -- Proof would go here
  sorry

end no_consecutive_heads_probability_l641_641301


namespace interval_f_increasing_inequality_g_holds_l641_641441

variable (x : ℝ) (m : ℝ) (ω : ℝ := 2)

def f (x : ℝ) := sin (4 * x + π / 3)

def increasing_intervals : Set ℝ := 
  {x | 0 ≤ x ∧ x ≤ π ∧ ((x ≥ 0 ∧ x ≤ π / 24) ∨ (x ≥ 7 * π / 24 ∧ x ≤ 13 * π / 24) ∨ (x ≥ 19 * π / 24 ∧ x ≤ π))}

def g (x : ℝ) := sin (2 * x - π / 6)

theorem interval_f_increasing : ∀ x, x ∈ [0, π] → x ∈ increasing_intervals := sorry

theorem inequality_g_holds : ∀ x, x ∈ [0, π / 2] → g x ^ 2 - 2 * m * g x + 2 * m + 1 > 0 → m > 1 - sqrt 2 := sorry

end interval_f_increasing_inequality_g_holds_l641_641441


namespace max_product_of_two_integers_sum_2000_l641_641256

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641256


namespace original_speed_l641_641524

noncomputable def circumference_feet := 10
noncomputable def feet_to_miles := 5280
noncomputable def seconds_to_hours := 3600
noncomputable def shortened_time := 1 / 18000
noncomputable def speed_increase := 6

theorem original_speed (r : ℝ) (t : ℝ) : 
  r * t = (circumference_feet / feet_to_miles) * seconds_to_hours ∧ 
  (r + speed_increase) * (t - shortened_time) = (circumference_feet / feet_to_miles) * seconds_to_hours
  → r = 6 := 
by
  sorry

end original_speed_l641_641524


namespace cara_skates_distance_l641_641631

theorem cara_skates_distance :
  let C := (0, 0)
  let D := (150, 0)
  let speed_cara := 6
  let speed_dan := 5
  let angle := 45
  ∃ t : ℝ, (5 * t)^2 = (6 * t)^2 + 150^2 - 2 * 6 * t * 150 * real.cos (real.pi / 4) →
  6 * t = 208.5 := 
by
  let t : ℝ := 34.75
  use t
  sorry

end cara_skates_distance_l641_641631


namespace volume_of_parallelepiped_l641_641614

theorem volume_of_parallelepiped 
  (l w h : ℝ)
  (h1 : l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5)
  (h2 : h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13)
  (h3 : h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) 
  : l * w * h = 750 :=
sorry

end volume_of_parallelepiped_l641_641614


namespace bread_cost_is_30_l641_641893

variable (cost_sandwich : ℝ)
variable (cost_ham : ℝ)
variable (cost_cheese : ℝ)

def cost_bread (cost_sandwich cost_ham cost_cheese : ℝ) : ℝ :=
  cost_sandwich - cost_ham - cost_cheese

theorem bread_cost_is_30 (H1 : cost_sandwich = 0.90)
  (H2 : cost_ham = 0.25)
  (H3 : cost_cheese = 0.35) :
  cost_bread cost_sandwich cost_ham cost_cheese = 0.30 :=
by
  rw [H1, H2, H3]
  simp [cost_bread]
  sorry

end bread_cost_is_30_l641_641893


namespace fewest_trips_l641_641453

theorem fewest_trips (total_objects : ℕ) (capacity : ℕ) (h_objects : total_objects = 17) (h_capacity : capacity = 3) : 
  (total_objects + capacity - 1) / capacity = 6 :=
by
  sorry

end fewest_trips_l641_641453


namespace inequality_proof_l641_641914

variable {a b c : ℝ}
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)

theorem inequality_proof :
  (a + 3 * c) / (a + 2 * b + c) + 
  (4 * b) / (a + b + 2 * c) - 
  (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 := 
sorry

end inequality_proof_l641_641914


namespace solve_equation1_solve_equation2_l641_641949

open Real

theorem solve_equation1 (x : ℝ) : (x^2 - 4 * x + 3 = 0) ↔ (x = 1 ∨ x = 3) := by
  sorry

theorem solve_equation2 (x : ℝ) : (x * (x - 2) = 2 * (2 - x)) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_equation1_solve_equation2_l641_641949


namespace product_probability_l641_641572

def S : Set ℤ := {4, 13, 22, 29, 37, 43, 57, 63, 71}

def pairs (s: Set ℤ) : Set (ℤ × ℤ) :=
  {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 < p.2}

def product_gt_150 (s : Set ℤ) : Set (ℤ × ℤ) :=
  {p ∈ pairs s | p.1 * p.2 > 150}

def probability_gt_150 : ℚ :=
  (product_gt_150 S).to_finset.card / (pairs S).to_finset.card

theorem product_probability : probability_gt_150 = 8/9 := by
  sorry

end product_probability_l641_641572


namespace min_distance_between_curves_l641_641531

noncomputable def minDistance : ℝ :=
  let f (a b : ℝ) := (a - real.exp b)^2 + (real.exp a - b)^2
  in real.sqrt (real.lift infi (λ (ab : ℝ × ℝ), f ab.1 ab.2))

theorem min_distance_between_curves : minDistance = real.sqrt 2 :=
sorry

end min_distance_between_curves_l641_641531


namespace parabola_focus_directrix_distance_l641_641957

theorem parabola_focus_directrix_distance :
  ∀ (y x : ℝ), 
    y^2 = 8 * x → 
    ∃ p : ℝ, 2 * p = 8 ∧ p = 4 := by
  sorry

end parabola_focus_directrix_distance_l641_641957


namespace domain_of_g_g_x_smallest_x_test_l641_641097

noncomputable def g (x : ℝ) : ℝ := real.sqrt (x - 5)

theorem domain_of_g_g_x :
  ∀ x, (x ≥ 30) ↔ (∃ y, (y = g x) ∧ (y ≥ 5)) :=
by
  intros x
  split
  {
    intro h
    use g x
    split
    {
      refl
    }
    {
      exact (real.sqrt_nonneg (x - 5)).trans (le_of_eq h.symm)
    }
  }
  {
    rintros ⟨y, hy1, hy2⟩
    rw [hy1, ← real.sqrt_inj_of_nonneg] at hy2
    {
      assume : x ≥ 5,
      exact (le_of_eq (hy2.trans (by linarith)))
    }
    all_goals
    {
      exact (real.sqrt_nonneg _)
    }
  }

lemma smallest_x_in_domain :
  ∃ x, (x ≥ 30) ∧ (∀ y, y < 30 → ¬(∃ z, (g z = y) ∧ (y ≥ 5))) := 
by 
  use 30
  split
  {
    linarith
  }
  {
    intros y hy,
    rintros ⟨z, hz1, hz2⟩,
    have := domain_of_g_g_x z
    rw [hz1] at this,
    exact hy.not_le this
  }

theorem smallest_x_test :
  ∃ x, (∃ y, (y = g x) ∧ (y ≥ 5)) ∧ ∀ z, z < 30 → ¬(∃ y, (y = g z) ∧ (y ≥ 5)) :=
by
  use 30
  exact smallest_x_in_domain

end domain_of_g_g_x_smallest_x_test_l641_641097


namespace fractions_sum_l641_641281

theorem fractions_sum (a : ℝ) (h : a ≠ 0) : (1 / a) + (2 / a) = 3 / a := 
by 
  sorry

end fractions_sum_l641_641281


namespace jordan_normal_form_exp_d_dt_spectrum_of_ad_A_l641_641360

-- Definitions of the operators
def quasi_polynomials (λ : ℝ) : Type := { p : ℝ[X] // degree p < 5 }
def operator_d_dt (λ : ℝ) (p : quasi_polynomials λ) : quasi_polynomials λ :=
  ⟨polynomial.derivative (p.val) + λ • p.val, sorry⟩ -- Proof that derivative + λ • p.val still has degree < 5

def exp_operator_d_dt (λ : ℝ) : Type :=
  sorry -- Define exp(λ id + f) and show its Jordan form is e^λ I_6 + e^λ J_6

def jordan_form_exp_operator_d_dt (λ : ℝ) : Prop :=
  sorry -- Conjecture that its Jordan form is e^λ I_6 + e^λ J_6

-- Definition of ad_A operator
def ad_A {n : Type} [fintype n] [decidable_eq n] (A : matrix n n ℝ) (B : matrix n n ℝ) : matrix n n ℝ :=
  A.mul B - B.mul A

def spectrum_ad_A {n : Type} [fintype n] [decidable_eq n] (A : matrix n n ℝ) : set ℝ :=
  {λ | ∃ (v : matrix n n ℝ), v ≠ 0 ∧ (ad_A A v) = λ • v}

-- Theorems to prove

theorem jordan_normal_form_exp_d_dt {λ : ℝ} :
  jordan_form_exp_operator_d_dt λ :=
sorry

theorem spectrum_of_ad_A {n : Type} [fintype n] [decidable_eq n] (A : matrix n n ℝ) (hA : A = diagonal_matrix) :
  spectrum_ad_A A = {λ_i - λ_j | i j : fin (fintype.card n)} :=
sorry

end jordan_normal_form_exp_d_dt_spectrum_of_ad_A_l641_641360


namespace quadrilateral_with_equal_diagonals_not_rectangle_l641_641671

structure Quadrilateral (α : Type _) :=
(diagonals_equal : Prop)
(diagonals_perpendicular : Prop)
(diagonals_bisect_each_other : Prop)

structure Rectangle (α : Type _) extends Quadrilateral α :=
(is_rectangle : Prop)

structure Rhombus (α : Type _) extends Quadrilateral α :=
(is_rhombus : Prop)

structure Parallelogram (α : Type _) extends Quadrilateral α :=
(diagonals_perpendicular_equal : Prop)
(is_square : Prop)

axiom quadrilateral_with_equal_diagonals_is_rectangle (α : Type _) (quad : Quadrilateral α) :
  quad.diagonals_equal → quad.is_rectangle

axiom quadrilateral_with_diagonals_perpendicular_and_bisect_each_other_is_rhombus (α : Type _) (quad : Quadrilateral α) :
  quad.diagonals_perpendicular → quad.diagonals_bisect_each_other → quad.is_rhombus

axiom parallelogram_with_diagonals_perpendicular_and_equal_is_square (α : Type _) (parallelogram : Parallelogram α) :
  parallelogram.diagonals_perpendicular_equal → parallelogram.is_square 

axiom quadrilateral_with_diagonals_bisect_each_other_is_parallelogram (α : Type _) (quad : Quadrilateral α) :
  quad.diagonals_bisect_each_other → quad.is_parallelogram

theorem quadrilateral_with_equal_diagonals_not_rectangle (α : Type _) (quad : Quadrilateral α) :
  quad.diagonals_equal → ¬(quad.is_rectangle) := by
  sorry

end quadrilateral_with_equal_diagonals_not_rectangle_l641_641671


namespace dixie_cup_ounces_l641_641266

def gallons_to_ounces (gallons : ℕ) : ℕ := gallons * 128

def initial_water_gallons (gallons : ℕ) : ℕ := gallons_to_ounces gallons

def total_chairs (rows chairs_per_row : ℕ) : ℕ := rows * chairs_per_row

theorem dixie_cup_ounces (initial_gallons rows chairs_per_row water_left : ℕ) 
  (h1 : initial_gallons = 3) 
  (h2 : rows = 5) 
  (h3 : chairs_per_row = 10) 
  (h4 : water_left = 84) 
  (h5 : 128 = 128) : 
  (initial_water_gallons initial_gallons - water_left) / total_chairs rows chairs_per_row = 6 :=
by 
  sorry

end dixie_cup_ounces_l641_641266


namespace max_product_l641_641224

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641224


namespace last_three_digits_7_pow_103_l641_641372

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l641_641372


namespace escalator_length_l641_641724

theorem escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (total_time : ℝ) : 
  escalator_speed = 12 → 
  person_speed = 2 → 
  total_time = 15 → 
  let effective_speed := escalator_speed + person_speed in 
  let L := effective_speed * total_time in 
  L = 210 :=
by
  intros h_escalator_speed h_person_speed h_total_time
  let effective_speed := escalator_speed + person_speed
  let L := effective_speed * total_time
  sorry

end escalator_length_l641_641724


namespace number_of_people_in_room_l641_641176

-- Given conditions
variables (people chairs : ℕ)
variables (three_fifths_people_seated : ℕ) (four_fifths_chairs : ℕ)
variables (empty_chairs : ℕ := 5)

-- Main theorem to prove
theorem number_of_people_in_room
    (h1 : 5 * empty_chairs = chairs)
    (h2 : four_fifths_chairs = 4 * chairs / 5)
    (h3 : three_fifths_people_seated = 3 * people / 5)
    (h4 : four_fifths_chairs = three_fifths_people_seated)
    : people = 33 := 
by
  -- Begin the proof
  sorry

end number_of_people_in_room_l641_641176


namespace group_B_equal_l641_641723

noncomputable def neg_two_pow_three := (-2)^3
noncomputable def minus_two_pow_three := -(2^3)

theorem group_B_equal : neg_two_pow_three = minus_two_pow_three :=
by sorry

end group_B_equal_l641_641723


namespace intersection_perpendicular_l641_641409

variables (α β γ : Plane) (a : Line)

theorem intersection_perpendicular
  (h1 : α ⊥ γ)
  (h2 : β ⊥ γ)
  (h3 : α ∩ β = a) :
  a ⊥ γ :=
sorry

end intersection_perpendicular_l641_641409


namespace part_a_part_b_l641_641906

/- Definition of the geometric entities and conditions as given -/
structure Triangle :=
(A B C I E F X : Point)
(incircle : Circle)
(circumcircle : Circle)
(M N P: Point)
(U V : Point)
(tangency_points : E tangency CA incircle ∧ F tangency AB incircle)
(midpoints : M midpoint BC ∧ N midpoint CA ∧ P midpoint AB)
(intersections : U intersection EF MN ∧ V intersection EF MP)
(arc_midpoint : X midpoint arc BAC circumcircle not_containing B not_containing C ∧ X equidistant B C)

/- Proofs needed -/
theorem part_a (T : Triangle) : I ∈ ray CV :=
by sorry

theorem part_b (T : Triangle) : XI bisects UV :=
by sorry

end part_a_part_b_l641_641906


namespace count_distinct_digit_odd_last_l641_641837

theorem count_distinct_digit_odd_last :
  ∃ n : ℕ, n = 2304 ∧ ∀ x : ℕ,
    (2000 ≤ x ∧ x ≤ 9999 ∧ (x % 10) % 2 = 1 ∧
    (∀ i j k l : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
      x = i * 1000 + j * 100 + k * 10 + l)) → 
    x mod 10000 = 2304 :=
sorry

end count_distinct_digit_odd_last_l641_641837


namespace order_of_abc_l641_641012

noncomputable def a := Real.log 1.2
noncomputable def b := (11 / 10) - (10 / 11)
noncomputable def c := 1 / (5 * Real.exp 0.1)

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end order_of_abc_l641_641012


namespace function_domain_l641_641150

theorem function_domain (x : ℝ) :
  (x + 5 ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≥ -5) ∧ (x ≠ -2) :=
by
  sorry

end function_domain_l641_641150


namespace sum_of_positive_integers_l641_641766

theorem sum_of_positive_integers (n : ℕ) (h : ∀ n, n > 0 → sqrt (7.2 * n - 13.2) < 3.6) : n = 6 :=
by
  -- Placeholder for proof
  sorry

end sum_of_positive_integers_l641_641766


namespace lagrange_interpolation_generalized_l641_641188

theorem lagrange_interpolation_generalized
  (n m: ℕ)
  (r r_1 r_2 ... r_m: ℕ)
  (x: ℝ)
  (x_1 x_2 ... x_n t_1 t_2 ... t_m: ℝ)
  (hn: n ≠ 0)
  (hm: m ≠ 0)
  (distinct_x: ∀ i j, i ≠ j → x_i ≠ x_j)
  (sum_rk: ∑ k in finRange m, r_k = r)
  (r_ge_m: r ≥ m):
  ∑ j in finRange n, (∏ k in finRange n \ {j}, (x - x_k) / (x_j - x_k)) *
    ∏ i in finRange n, (x_j - t_i) ^ r_i
  = (∏ i in finRange n, (x - t_i) ^ r_i) - ∏ i in finRange n, (x - x_i) ^ r_i := by
  sorry

end lagrange_interpolation_generalized_l641_641188


namespace smallest_of_x_y_z_l641_641624

variables {a b c d : ℕ}

/-- Given that x, y, and z are in the ratio a, b, c respectively, 
    and their sum x + y + z equals d, and 0 < a < b < c,
    prove that the smallest of x, y, and z is da / (a + b + c). -/
theorem smallest_of_x_y_z (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : 0 < d)
    (h_sum : ∀ k : ℚ, x = k * a → y = k * b → z = k * c → x + y + z = d) : 
    (∃ k : ℚ, x = k * a ∧ y = k * b ∧ z = k * c ∧ k = d / (a + b + c) ∧ x = da / (a + b + c)) :=
by 
  sorry

end smallest_of_x_y_z_l641_641624


namespace find_d_single_point_l641_641143

theorem find_d_single_point :
  ∃ (d : ℝ), ∀ (x y : ℝ),
  2 * x^2 + y^2 + 4 * x - 6 * y + d = 0 ↔ x = -1 ∧ y = 3 :=
by
  let d := 11
  use d
  intro x y
  split
  { intro h
    -- proof needed
    sorry },
  { rintro ⟨rfl, rfl⟩
    -- proof needed
    sorry }

end find_d_single_point_l641_641143


namespace no_two_consecutive_heads_l641_641294

/-- The probability that two heads never appear consecutively when a coin is tossed 10 times is 9/64. -/
theorem no_two_consecutive_heads (tosses : ℕ) (prob: ℚ):
  tosses = 10 → 
  (prob = 9 / 64) → 
  prob = probability_no_consecutive_heads tosses := sorry

end no_two_consecutive_heads_l641_641294


namespace find_n_l641_641869

theorem find_n (n : ℕ) (a_n D_n d_n : ℕ) (h1 : n > 5) (h2 : D_n - d_n = a_n) : n = 9 := 
by 
  sorry

end find_n_l641_641869


namespace range_of_positive_integers_is_8_l641_641680

-- Define the list k
def k := list.range' (-4) 14

-- Define the positive integers in list k
def positive_integers := k.filter (λ x, x > 0)

-- Find the range of the positive integers
def range_positive_integers := list.maximum' positive_integers - list.minimum' positive_integers

-- The theorem to prove
theorem range_of_positive_integers_is_8 : range_positive_integers = 8 := by
  sorry

end range_of_positive_integers_is_8_l641_641680


namespace equal_angles_at_midpoints_l641_641056

variable {A B C D M N P Q : Type}
variables [EuclideanPlane A B C D M N P Q]
variables {AB CD : ℝ}
variable {quadr : ConvexQuadrilateral A B C D}

-- Given conditions
variables (h1 : quadr.is_not_parallelogram)
variables (h2 : AB = CD)
variables (h3 : midpoint A C M)
variables (h4 : midpoint B D N)
variables (h5 : line_through_midpoint MN intersect AB at P)
variables (h6 : line_through_midpoint MN intersect CD at Q)

-- Goal
theorem equal_angles_at_midpoints 
  (h1 : quadr.is_not_parallelogram)
  (h2 : AB = CD)
  (h3 : midpoint A C M)
  (h4 : midpoint B D N)
  (h5 : line_through_midpoint MN intersect AB at P)
  (h6 : line_through_midpoint MN intersect CD at Q) : 
  ∠BPM = ∠CQN := by
  sorry

end equal_angles_at_midpoints_l641_641056


namespace sin_B_triangle_area_l641_641478

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem sin_B (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5) :
  Real.sin B = Real.sqrt 10 / 10 := by
  sorry

theorem triangle_area (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hDiff : c - a = 5 - Real.sqrt 10) (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  1 / 2 * a * c * Real.sin B = 5 / 2 := by
  sorry

end sin_B_triangle_area_l641_641478


namespace max_acceleration_uphill_l641_641696

variables (θ μ g : ℝ)
-- Condition: $\mu > \tan \theta$
variable (h_condition : μ > Real.tan θ)

def max_acceleration (g θ μ : ℝ) : ℝ :=
  g * (μ * Real.cos θ - Real.sin θ)

theorem max_acceleration_uphill (h : μ > Real.tan θ) :
    ∀ (g θ μ : ℝ), g > 0 → max_acceleration g θ μ = g * (μ * Real.cos θ - Real.sin θ) :=
by 
  assume g θ μ hg,
  exact rfl

end max_acceleration_uphill_l641_641696


namespace belt_gap_can_walk_under_l641_641192

theorem belt_gap_can_walk_under :
  let C := 40000 * 1000 -- Convert km to meters
  let C_new := C + 10
  let r := C / (2 * Real.pi)
  let r_new := C_new / (2 * Real.pi)
  let gap := r_new - r
  gap ≈ 1.59 :=
by
  sorry

end belt_gap_can_walk_under_l641_641192


namespace Misha_l641_641658

theorem Misha's_decision_justified :
  let A_pos := 7 in
  let A_neg := 4 in
  let B_pos := 4 in
  let B_neg := 1 in
  (B_pos / (B_pos + B_neg) > A_pos / (A_pos + A_neg)) := 
sorry

end Misha_l641_641658


namespace hyperbola_eccentricity_l641_641411

noncomputable def hyperbola := 
  { x : ℝ // x > 0 } → { y : ℝ // y > 0 } → Type

structure point (x y : ℝ)

def focus (c : ℝ) : point c 0 := ⟨c, 0⟩
def endpoint_conjugate_axis (b : ℝ) : point 0 b := ⟨0, b⟩
def on_hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2 - y^2 / b^2 = 1)

theorem hyperbola_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (F : point c 0) (B : point 0 b) (P : point x y)
  (H1 : on_hyperbola a b x y)
  (H2 : vector.eq_sub_mul (P, B) 2 (B, F)) :
  (eccentricity a c = (sqrt 10) / 2) :=
sorry

end hyperbola_eccentricity_l641_641411


namespace solid_surface_area_eq_l641_641329

-- Given definition: A solid formed by rotating an equilateral triangle of side length a around a line containing one of its sides
def solid_surface_area (a : ℝ) : ℝ :=
2 * Real.pi * (Real.sqrt 3 / 2 * a) * a

-- Prove that the surface area is equal to sqrt(3) * pi * a^2
theorem solid_surface_area_eq (a : ℝ) : solid_surface_area a = Real.sqrt 3 * Real.pi * a ^ 2 :=
by
  -- here a proof would go, but 'sorry' is used to skip it
  sorry

end solid_surface_area_eq_l641_641329


namespace range_of_k_intersecting_hyperbola_l641_641806

theorem range_of_k_intersecting_hyperbola :
  (∀ b : ℝ, ∃ x y : ℝ, y = k * x + b ∧ x^2 - 2 * y^2 = 1) →
  -Real.sqrt 2 / 2 < k ∧ k < Real.sqrt 2 / 2 :=
sorry

end range_of_k_intersecting_hyperbola_l641_641806


namespace no_two_heads_consecutively_probability_l641_641290

theorem no_two_heads_consecutively_probability :
  (∃ (total_sequences : ℕ) (valid_sequences : ℕ),
    total_sequences = 2^10 ∧ valid_sequences = 1 + 10 + 36 + 56 + 35 + 6 ∧
    (valid_sequences / total_sequences : ℚ) = 9 / 64) :=
begin
  sorry
end

end no_two_heads_consecutively_probability_l641_641290


namespace sum_of_squares_l641_641886

theorem sum_of_squares {S : ℕ → ℕ} (h : ∀ n, S n = 2^n - 1) :
  ∀ n, (∑ i in finset.range (n+1), (if i = 0 then S 1 else S (i+1) - S i)^2) = (4^(n + 1) - 1) / 3 :=
by sorry

end sum_of_squares_l641_641886


namespace train_speed_ratio_l641_641998

theorem train_speed_ratio 
  (distance_2nd_train : ℕ)
  (time_2nd_train : ℕ)
  (speed_1st_train : ℚ)
  (H1 : distance_2nd_train = 400)
  (H2 : time_2nd_train = 4)
  (H3 : speed_1st_train = 87.5) :
  distance_2nd_train / time_2nd_train = 100 ∧ 
  (speed_1st_train / (distance_2nd_train / time_2nd_train)) = 7 / 8 :=
by
  sorry

end train_speed_ratio_l641_641998


namespace find_a_l641_641123

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (x - a)^2 + y^2 = (x^2 + (y-1)^2)) ∧ (¬ ∃ x y : ℝ, y = x + 1) → a = 1 :=
by
  sorry

end find_a_l641_641123


namespace max_product_l641_641227

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641227


namespace bisect_AM_BC_l641_641736

noncomputable theory

-- Definitions of circles, tangency, and point relationships
-- These definitions are assumed to be provided by Mathlib or defined abstractly for this problem

variables {S1 S2 : Circle} 
variables {A B C M : Point}

-- Conditions
axiom tangent_S1_A : S1.TangentAt A AC -- Circle S1 tangents AC at A
axiom tangent_S1_C : S1.TangentAt C AB -- Circle S1 tangents AB at C
axiom tangent_S2_C : S2.TangentAt C AC -- Circle S2 tangents AC at C
axiom passes_S2_B : S2.PassesThrough B -- Circle S2 passes through B
axiom intersection_S1_S2_M : S1.IntersectsAt S2 M -- Circles S1 and S2 intersect at M

-- Question to prove
theorem bisect_AM_BC : Bisects (LineThrough A M) BC :=
by
  sorry

end bisect_AM_BC_l641_641736


namespace maximum_possible_median_l641_641557

theorem maximum_possible_median
  (total_cans : ℕ)
  (total_customers : ℕ)
  (min_cans_per_customer : ℕ)
  (alt_min_cans_per_customer : ℕ)
  (exact_min_cans_count : ℕ)
  (atleast_min_cans_count : ℕ)
  (min_cans_customers : ℕ)
  (alt_min_cans_customer: ℕ): 
  (total_cans = 300) → 
  (total_customers = 120) →
  (min_cans_per_customer = 2) →
  (alt_min_cans_per_customer = 4) →
  (min_cans_customers = 59) →
  (alt_min_cans_customer = 61) →
  (min_cans_per_customer * min_cans_customers + alt_min_cans_per_customer * (total_customers - min_cans_customers) = total_cans) →
  max (min_cans_per_customer + 1) (alt_min_cans_per_customer - 1) = 3 :=
sorry

end maximum_possible_median_l641_641557


namespace last_three_digits_of_7_pow_103_l641_641366

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l641_641366


namespace last_three_digits_7_pow_103_l641_641371

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l641_641371


namespace geometry_problem_l641_641607

noncomputable def Q : ℝ × ℝ := (Real.sqrt 2, 0)

def line_eq (x y : ℝ) : Prop := y - x * Real.sqrt 2 + 4 = 0

def parabola_eq (x y : ℝ) : Prop := y^2 = 2 * x + 4

def points_C_and_D (C D : ℝ × ℝ) : Prop :=
  (C ≠ D) ∧ (parabola_eq C.1 C.2) ∧ (parabola_eq D.1 D.2) ∧ (line_eq C.1 C.2) ∧ (line_eq D.1 D.2)

def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem geometry_problem (C D : ℝ × ℝ) (h : points_C_and_D C D) :
  |distance C Q - distance D Q| = Real.sqrt 3 :=
by
  sorry

end geometry_problem_l641_641607


namespace cos_eq_solutions_count_l641_641074

theorem cos_eq_solutions_count : 
  ∀ (a b : ℝ), (0 ≤ a) → (a ≤ π) → (0 ≤ b) → (b ≤ π) → 
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π) → cos (7 * x) = cos (5 * x)) → 
  (∀ S : Set ℝ, (∀ x : ℝ, x ∈ S ↔ (0 ≤ x ∧ x ≤ π ∧ cos (7 * x) = cos (5 * x))) → (S.card = 7)) := 
sorry

end cos_eq_solutions_count_l641_641074


namespace root_fraction_power_equality_l641_641965

theorem root_fraction_power_equality :
  (11^(1/5)) / (11^(1/2)) = 11^(-3/10) :=
by
  sorry

end root_fraction_power_equality_l641_641965


namespace number_of_siblings_l641_641543

-- Definitions based on the conditions
def total_spent : ℕ := 150
def spending_per_parent : ℕ := 30
def parents : ℕ := 2
def spending_per_sibling : ℕ := 30

-- Theorem statement: Given these conditions, prove the number of siblings Mia has
theorem number_of_siblings (total_spent = 150) (spending_per_parent = 30) (parents = 2) (spending_per_sibling = 30) : 
    (total_spent - parents * spending_per_parent) / spending_per_sibling = 3 :=
by
  sorry

end number_of_siblings_l641_641543


namespace binary_addition_subtraction_l641_641716

def bin_10101 : ℕ := 0b10101
def bin_1011 : ℕ := 0b1011
def bin_1110 : ℕ := 0b1110
def bin_110001 : ℕ := 0b110001
def bin_1101 : ℕ := 0b1101
def bin_101100 : ℕ := 0b101100

theorem binary_addition_subtraction :
  bin_10101 + bin_1011 + bin_1110 + bin_110001 - bin_1101 = bin_101100 := 
sorry

end binary_addition_subtraction_l641_641716


namespace imaginary_part_of_z_l641_641800

open Complex

theorem imaginary_part_of_z (z : ℂ) (hz : z + z * Complex.i = 2) : z.im = -1 := by
  sorry

end imaginary_part_of_z_l641_641800


namespace find_five_numbers_l641_641761

theorem find_five_numbers :
  ∃ a b c d e,
    (a + b = 0 ∨ a + b = 2 ∨ a + b = 4 ∨ a + b = 5 ∨ a + b = 7 ∨ a + b = 9 ∨ a + b = 10 ∨ a + b = 12 ∨ a + b = 14 ∨ a + b = 17) ∧
    (a + c = 0 ∨ a + c = 2 ∨ a + c = 4 ∨ a + c = 5 ∨ a + c = 7 ∨ a + c = 9 ∨ a + c = 10 ∨ a + c = 12 ∨ a + c = 14 ∨ a + c = 17) ∧
    (a + d = 0 ∨ a + d = 2 ∨ a + d = 4 ∨ a + d = 5 ∨ a + d = 7 ∨ a + d = 9 ∨ a + d = 10 ∨ a + d = 12 ∨ a + d = 14 ∨ a + d = 17) ∧
    (a + e = 0 ∨ a + e = 2 ∨ a + e = 4 ∨ a + e = 5 ∨ a + e = 7 ∨ a + e = 9 ∨ a + e = 10 ∨ a + e = 12 ∨ a + e = 14 ∨ a + e = 17) ∧
    (b + c = 0 ∨ b + c = 2 ∨ b + c = 4 ∨ b + c = 5 ∨ b + c = 7 ∨ b + c = 9 ∨ b + c = 10 ∨ b + c = 12 ∨ b + c = 14 ∨ b + c = 17) ∧
    (b + d = 0 ∨ b + d = 2 ∨ b + d = 4 ∨ b + d = 5 ∨ b + d = 7 ∨ b + d = 9 ∨ b + d = 10 ∨ b + d = 12 ∨ b + d = 14 ∨ b + d = 17) ∧
    (b + e = 0 ∨ b + e = 2 ∨ b + e = 4 ∨ b + e = 5 ∨ b + e = 7 ∨ b + e = 9 ∨ b + e = 10 ∨ b + e = 12 ∨ b + e = 14 ∨ b + e = 17) ∧
    (c + d = 0 ∨ c + d = 2 ∨ c + d = 4 ∨ c + d = 5 ∨ c + d = 7 ∨ c + d = 9 ∨ c + d = 10 ∨ c + d = 12 ∨ c + d = 14 ∨ c + d = 17) ∧
    (c + e = 0 ∨ c + e = 2 ∨ c + e = 4 ∨ c + e = 5 ∨ c + e = 7 ∨ c + e = 9 ∨ c + e = 10 ∨ c + e = 12 ∨ c + e = 14 ∨ c + e = 17) ∧
    (d + e = 0 ∨ d + e = 2 ∨ d + e = 4 ∨ d + e = 5 ∨ d + e = 7 ∨ d + e = 9 ∨ d + e = 10 ∨ d + e = 12 ∨ d + e = 14 ∨ d + e = 17) ∧
    a + b + c + d + e = 20 :=
  ⟨-1, 1, 3, 6, 11, by sorry⟩

end find_five_numbers_l641_641761


namespace Morse_code_sequences_l641_641049

theorem Morse_code_sequences : 
  let symbols (n : ℕ) := 2^n in
  symbols 1 + symbols 2 + symbols 3 + symbols 4 + symbols 5 = 62 :=
by
  sorry

end Morse_code_sequences_l641_641049


namespace greatest_four_digit_divisible_l641_641702

theorem greatest_four_digit_divisible (p : ℕ) :
  (1000 ≤ p ∧ p < 10000) ∧ 
  (63 ∣ p) ∧ 
  (63 ∣ reverse_digits p) ∧ 
  (11 ∣ p) → 
  p = 9779 :=
by
  sorry

-- Helper function to reverse the digits of a natural number
noncomputable def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10 in
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0


end greatest_four_digit_divisible_l641_641702


namespace distinguish_by_rolling_time_l641_641667

def solid_ball_moment_of_inertia (M R : ℝ) : ℝ :=
  (2 / 5) * M * R^2

def hollow_sphere_moment_of_inertia (M R : ℝ) : ℝ :=
  (2 / 3) * M * R^2

theorem distinguish_by_rolling_time (M R : ℝ) :
  solid_ball_moment_of_inertia M R ≠ hollow_sphere_moment_of_inertia M R :=
by
  sorry

end distinguish_by_rolling_time_l641_641667


namespace greatest_product_sum_2000_eq_1000000_l641_641237

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641237


namespace product_of_decimals_l641_641730

theorem product_of_decimals :
  0.5 * 0.8 = 0.40 :=
by
  -- Proof will go here; using sorry to skip for now
  sorry

end product_of_decimals_l641_641730


namespace smallest_positive_angle_l641_641378

theorem smallest_positive_angle (x : ℝ) (h : sin(4 * x) * sin(6 * x) = cos(4 * x) * cos(6 * x)) : x = 9 * real.pi / 180 :=
by
  sorry

end smallest_positive_angle_l641_641378


namespace find_angle_C_l641_641028

theorem find_angle_C (a b c : ℝ) (B C : ℝ)
    (h1 : b^2 + c^2 - b * c = a^2)
    (h2 : a/b = real.sqrt 3)
    (h3 : ∠A = 60) (h4 : ∠B = 30) :
    ∠C = 90 :=
  sorry

end find_angle_C_l641_641028


namespace coeff_x2_in_expansion_l641_641586

theorem coeff_x2_in_expansion : 
  let expr := x - (2 / x) in 
  let expansion := expr ^ 6 in 
  (∃ c : ℤ, expr = 60 * x^2) := sorry

end coeff_x2_in_expansion_l641_641586


namespace vinegar_start_l641_641933

-- Define the conditions as variables and constants
variables (jars cucumbers leftVinegar ouncesPerJarPickle: ℕ)
variables (picklesPerCucumber picklesPerJar: ℕ)

-- Assigning given conditions
def initCond := (jars = 4) ∧ (cucumbers = 10) ∧ (leftVinegar = 60) ∧ 
                 (picklesPerCucumber = 6) ∧ (picklesPerJar = 12) ∧ 
                 (ouncesPerJarPickle = 10)

-- Final statement to prove Phillip started with 100 ounces of vinegar
theorem vinegar_start (initCond : initCond): 
   (∨ initCond → (initialVinegar: ℕ) ∧ initialVinegar = 100) := 
begin
  obtain ⟨_, ⟩,
  use 100,
  sorry, -- Replaced with actual proof steps
end

end vinegar_start_l641_641933


namespace g_neg_x_l641_641103

def g (x : ℝ) : ℝ := (x^2 + 2*x + 3) / (x^2 - 2*x + 3)

theorem g_neg_x (x : ℝ) (h : x^2 ≠ 3) : g (-x) = 1 / g x :=
  sorry

end g_neg_x_l641_641103


namespace max_product_two_integers_sum_2000_l641_641203

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641203


namespace smallest_positive_period_max_value_in_interval_min_value_in_interval_l641_641433

def f (x : ℝ) : ℝ := 4 * Real.cos x - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T := 
  sorry

theorem max_value_in_interval :
  ∀ x ∈ Set.Ioo 0 (Real.pi / 2), f x ≤ 3 := 
  sorry

theorem min_value_in_interval :
  ∀ x ∈ Set.Ioo 0 (Real.pi / 2), -1 ≤ f x := 
  sorry

end smallest_positive_period_max_value_in_interval_min_value_in_interval_l641_641433


namespace brendas_age_l641_641718

theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B) 
  (h2 : J = B + 8) 
  (h3 : A = J) 
: B = 8 / 3 := 
by 
  sorry

end brendas_age_l641_641718


namespace part1_part2_part3_l641_641853

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : ∀ x, f(x) ≠ 0
axiom f_mult : ∀ x y, f(x) * f(y) = f(x + y)
axiom f_gt_one_when_neg : ∀ x, x < 0 → f(x) > 1
axiom f_at_4 : f(4) = 1 / 16
axiom f_bound : ∀ a x, a ∈ Icc (-1) 1 → f(x^2 - 2*a*x + 2) ≤ 1 / 4

theorem part1 : f(0) = 1 ∧ ∀ x, f(x) > 0 := 
by sorry

theorem part2 : ∀ x1 x2 : ℝ, x1 > x2 → f(x1) < f(x2) := 
by sorry

theorem part3 : (∀ a ∈ Icc (-1) 1, f(x^2 - 2*a*x + 2) ≤ 1/4) 
          → ∀ x, x ≤ -2 ∨ x = 0 ∨ x ≥ 2 := 
by sorry

end part1_part2_part3_l641_641853


namespace area_of_rectangular_garden_l641_641338

theorem area_of_rectangular_garden (length width : ℝ) (h_length : length = 2.5) (h_width : width = 0.48) :
  length * width = 1.2 :=
by
  sorry

end area_of_rectangular_garden_l641_641338


namespace trigonometric_identity_l641_641796

open Real

theorem trigonometric_identity
  (α β γ φ : ℝ)
  (h1 : sin α + 7 * sin β = 4 * (sin γ + 2 * sin φ))
  (h2 : cos α + 7 * cos β = 4 * (cos γ + 2 * cos φ)) :
  2 * cos (α - φ) = 7 * cos (β - γ) :=
by sorry

end trigonometric_identity_l641_641796


namespace calc_expression_l641_641732

theorem calc_expression : (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 :=
  sorry

end calc_expression_l641_641732


namespace partition_natural_numbers_l641_641939

theorem partition_natural_numbers :
  ∃ (f : ℕ → ℕ), (∀ n, 1 ≤ f n ∧ f n ≤ 100) ∧
  (∀ a b c, a + 99 * b = c → f a = f c ∨ f a = f b ∨ f b = f c) :=
sorry

end partition_natural_numbers_l641_641939


namespace hyperbola_eccentricity_eq_two_l641_641962

theorem hyperbola_eccentricity_eq_two :
  (∀ x y : ℝ, ((x^2 / 2) - (y^2 / 6) = 1) → 
    let a_squared := 2
    let b_squared := 6
    let a := Real.sqrt a_squared
    let b := Real.sqrt b_squared
    let e := Real.sqrt (1 + b_squared / a_squared)
    e = 2) := 
sorry

end hyperbola_eccentricity_eq_two_l641_641962


namespace exists_similar_1995digit_numbers_l641_641922

theorem exists_similar_1995digit_numbers :
  ∃ (A B C : ℕ), 
  (A = nat_digits_to_number (list.repeat [4, 5, 9] 665)) ∧
  (B = nat_digits_to_number (list.repeat [4, 9, 5] 665)) ∧
  (C = nat_digits_to_number (list.repeat [9, 5, 4] 665)) ∧
  (A + B = C) ∧
  (nat_digits_length A = 1995) ∧
  (nat_digits_length B = 1995) ∧
  (nat_digits_length C = 1995)  :=
by {
  -- Code for the proof will reside here
  sorry
}

-- Helper functions to convert digits to numbers and lengths
def nat_digits_to_number (ds : list ℕ) : ℕ :=
  ds.foldr (λ d acc, d + 10 * acc) 0

def nat_digits_length (n : ℕ) : ℕ :=
  n.digits.size

end exists_similar_1995digit_numbers_l641_641922


namespace inequality_proof_l641_641443

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log x - x + 1 / x

-- Define a, b, and c
def a : ℝ := f Real.exp 1  -- f(e)
def b : ℝ := f Real.pi     -- f(π)
def c : ℝ := f (Real.log 30 / Real.log 2)  -- f(log_2 30)

-- Define the theorem to prove the inequality
theorem inequality_proof : c < b ∧ b < a := 
by {
  sorry -- proof not required
}

end inequality_proof_l641_641443


namespace equation_of_hyperbola_length_of_AB_l641_641790

-- Given conditions
variables (a b x y : ℝ)
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def eccentricity (c a : ℝ) : ℝ :=
  c / a

def vertex (a : ℝ) : ℝ × ℝ :=
  (a, 0)

def focus (c : ℝ) : ℝ × ℝ :=
  (c, 0)

def line_through_focus (m c : ℝ) (x : ℝ) : ℝ :=
  m * (x - c)

-- Statements to prove
theorem equation_of_hyperbola :
  (∃ (a b c : ℝ), hyperbola a b x y ∧ eccentricity c a = sqrt 3 ∧ vertex (sqrt 3) ∧
    c = sqrt 3 * a ∧ b^2 = c^2 - a^2) →
  (∃ (x : ℝ), hyperbola (sqrt 3) (sqrt 6) x y) := sorry

theorem length_of_AB :
  (∃ (a b c : ℝ), hyperbola a b x y ∧ eccentricity c a = sqrt 3 ∧ vertex (sqrt 3) ∧
    c = sqrt 3 * a ∧ b^2 = c^2 - a^2 ∧
    ∃ (l : ℝ), line_through_focus (sqrt 3 / 3) c l) →
  (∃ (A B : ℝ × ℝ), abs (dist A B) = 16 * sqrt 3 / 5) := sorry

end equation_of_hyperbola_length_of_AB_l641_641790


namespace sum_rationalize_denominator_l641_641343

theorem sum_rationalize_denominator :
  ∑ n in Finset.range 4998 + 3, (1 / (n * Real.sqrt (n + 1) + (n + 1) * Real.sqrt n)) = 
  (1 / Real.sqrt 3) - (1 / Real.sqrt 5001) :=
by
  sorry

end sum_rationalize_denominator_l641_641343


namespace quadratic_radical_type_l641_641797

-- Problem statement: Given that sqrt(2a + 1) is a simplest quadratic radical and the same type as sqrt(48), prove that a = 1.

theorem quadratic_radical_type (a : ℝ) (h1 : ((2 * a) + 1) = 3) : a = 1 :=
by
  sorry

end quadratic_radical_type_l641_641797


namespace pie_shop_total_earnings_l641_641711

theorem pie_shop_total_earnings :
  let price_per_slice_custard := 3
  let price_per_slice_apple := 4
  let price_per_slice_blueberry := 5
  let slices_per_whole_custard := 10
  let slices_per_whole_apple := 8
  let slices_per_whole_blueberry := 12
  let num_whole_custard_pies := 6
  let num_whole_apple_pies := 4
  let num_whole_blueberry_pies := 5
  let total_earnings :=
    (num_whole_custard_pies * slices_per_whole_custard * price_per_slice_custard) +
    (num_whole_apple_pies * slices_per_whole_apple * price_per_slice_apple) +
    (num_whole_blueberry_pies * slices_per_whole_blueberry * price_per_slice_blueberry)
  total_earnings = 608 := by
  sorry

end pie_shop_total_earnings_l641_641711


namespace max_product_l641_641226

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641226


namespace average_price_of_returned_cans_l641_641894

theorem average_price_of_returned_cans (total_cans : ℕ) (returned_cans : ℕ) (remaining_cans : ℕ)
  (avg_price_total : ℚ) (avg_price_remaining : ℚ) :
  total_cans = 6 →
  returned_cans = 2 →
  remaining_cans = 4 →
  avg_price_total = 36.5 →
  avg_price_remaining = 30 →
  (avg_price_total * total_cans - avg_price_remaining * remaining_cans) / returned_cans = 49.5 :=
by
  intros h_total_cans h_returned_cans h_remaining_cans h_avg_price_total h_avg_price_remaining
  rw [h_total_cans, h_returned_cans, h_remaining_cans, h_avg_price_total, h_avg_price_remaining]
  sorry

end average_price_of_returned_cans_l641_641894


namespace correct_combined_average_l641_641863

noncomputable def average_marks : ℝ :=
  let num_students : ℕ := 100
  let avg_math_marks : ℝ := 85
  let avg_science_marks : ℝ := 89
  let incorrect_math_marks : List ℝ := [76, 80, 95, 70, 90]
  let correct_math_marks : List ℝ := [86, 70, 75, 90, 100]
  let incorrect_science_marks : List ℝ := [105, 60, 80, 92, 78]
  let correct_science_marks : List ℝ := [95, 70, 90, 82, 88]

  let total_incorrect_math := incorrect_math_marks.sum
  let total_correct_math := correct_math_marks.sum
  let diff_math := total_correct_math - total_incorrect_math

  let total_incorrect_science := incorrect_science_marks.sum
  let total_correct_science := correct_science_marks.sum
  let diff_science := total_correct_science - total_incorrect_science

  let incorrect_total_math := avg_math_marks * num_students
  let correct_total_math := incorrect_total_math + diff_math

  let incorrect_total_science := avg_science_marks * num_students
  let correct_total_science := incorrect_total_science + diff_science

  let combined_total := correct_total_math + correct_total_science
  combined_total / (num_students * 2)

theorem correct_combined_average :
  average_marks = 87.1 :=
by
  sorry

end correct_combined_average_l641_641863


namespace how_many_more_stickers_l641_641511

-- Define the conditions
def Karl_stickers : ℕ := 25
variable (x : ℕ) -- x is the number of stickers Ryan has more than Karl
def Ryan_stickers : ℕ := Karl_stickers + x
def Ben_stickers : ℕ := Ryan_stickers - 10

-- Prove the question given the conditions
theorem how_many_more_stickers (h : Karl_stickers + Ryan_stickers + Ben_stickers = 105) : (Ryan_stickers - Karl_stickers) = 20 :=
sorry

end how_many_more_stickers_l641_641511


namespace least_value_x_div_57_is_57_l641_641637

def least_x (n : ℕ) : ℕ :=
  if h : ∃ x : ℕ, 23 * x % n = 0 then nat.find h else 0

theorem least_value_x_div_57_is_57 : least_x 57 = 57 :=
by
  unfold least_x
  have h : ∃ x : ℕ, 23 * x % 57 = 0 := by
    use 57
    norm_num
  rw dif_pos h
  exact nat.find_spec h

end least_value_x_div_57_is_57_l641_641637


namespace unplanted_fraction_l641_641858

theorem unplanted_fraction (a b hypotenuse : ℕ) (side_length_P : ℚ) 
                          (h1 : a = 5) (h2 : b = 12) (h3 : hypotenuse = 13)
                          (h4 : side_length_P = 5 / 3) : 
                          (side_length_P * side_length_P) / ((a * b) / 2) = 5 / 54 :=
by
  sorry

end unplanted_fraction_l641_641858


namespace division_minutes_per_day_l641_641137

-- Define the conditions
def total_hours : ℕ := 5
def minutes_multiplication_per_day : ℕ := 10
def days_total : ℕ := 10

-- Convert hours to minutes
def total_minutes : ℕ := total_hours * 60

-- Total minutes spent on multiplication
def total_minutes_multiplication : ℕ := minutes_multiplication_per_day * days_total

-- Total minutes spent on division
def total_minutes_division : ℕ := total_minutes - total_minutes_multiplication

-- Minutes spent on division per day
def minutes_division_per_day : ℕ := total_minutes_division / days_total

-- The theorem to prove
theorem division_minutes_per_day : minutes_division_per_day = 20 := by
  sorry

end division_minutes_per_day_l641_641137


namespace max_product_of_two_integers_sum_2000_l641_641248

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641248


namespace greatest_product_sum_2000_l641_641212

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641212


namespace tan_2A_eq_area_triangle_eq_l641_641417

-- Define the conditions and variables
variables (S : ℝ) (AB AC : ℝ) (cosC : ℝ) (BC : ℝ)

-- Assume the provided conditions
def conditions := 
  (AB * AC = S) ∧
  (cosC = 3/5) ∧
  (BC = 2)

-- Prove the value of tan 2A
theorem tan_2A_eq : conditions S AB AC cosC BC → (tan (2 * atan (2 / 1))) = -4/3 :=
by
  sorry

-- Prove the area of the triangle ABC is 8/5
theorem area_triangle_eq : conditions S AB AC cosC BC → (S = 8/5) :=
by
  sorry

end tan_2A_eq_area_triangle_eq_l641_641417


namespace problem1_problem2_problem3_problem4_l641_641169

open scoped BigOperators

section problem

variables (M F : ℕ) -- Number of male and female athletes
variables (cM cF : ℕ) -- Designating the male and female captains

noncomputable def waysToSelectTeam_3M_2F (M F : ℕ) : ℕ :=
  Nat.choose M 3 * Nat.choose F 2

theorem problem1 (hM : M = 6) (hF : F = 4) : waysToSelectTeam_3M_2F M F = 120 :=
by
  rw [hM, hF]
  norm_num

noncomputable def waysToSelectTeam_with_at_least_1F (M F : ℕ) : ℕ :=
  Nat.choose 10 5 - Nat.choose M 5

theorem problem2 (H : M = 6) : waysToSelectTeam_with_at_least_1F M F = 246 :=
by
  rw [H]
  norm_num

noncomputable def waysToSelectTeam_with_at_least_1C (M F : ℕ) (cM cF : ℕ) : ℕ :=
  Nat.choose 8 4 + Nat.choose 8 4 + Nat.choose 8 3

theorem problem3 (hM : M = 6) (hF : F = 4) (hcM : cM = 1) (hcF : cF = 1) : waysToSelectTeam_with_at_least_1C M F cM cF = 196 :=
by
  rw [hM, hF, hcM, hcF]
  norm_num

noncomputable def waysToSelectTeam_with_1C_and_1F (M F : ℕ) (cM cF : ℕ) : ℕ :=
  Nat.choose 9 4 + (Nat.choose 8 4 - Nat.choose 5 4)

theorem problem4 (hM : M = 6) (hF : F = 4) (hcM : cM = 1) (hcF : cF = 1) : waysToSelectTeam_with_1C_and_1F M F cM cF = 191 :=
by
  rw [hM, hF, hcM, hcF]
  norm_num

end problem

end problem1_problem2_problem3_problem4_l641_641169


namespace option_b_correct_l641_641328

variable (Line Plane : Type)

-- Definitions for perpendicularity and parallelism
variable (perp parallel : Line → Plane → Prop) (parallel_line : Line → Line → Prop)

-- Assumptions reflecting the conditions in the problem
axiom perp_alpha_1 {a : Line} {alpha : Plane} : perp a alpha
axiom perp_alpha_2 {b : Line} {alpha : Plane} : perp b alpha

-- The statement to prove
theorem option_b_correct (a b : Line) (alpha : Plane) :
  perp a alpha → perp b alpha → parallel_line a b :=
by
  intro h1 h2
  -- proof omitted
  sorry

end option_b_correct_l641_641328


namespace problem1_problem2_problem3_l641_641437

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 1 - 4 / (2 * a^x + a)

-- Problem statements:
theorem problem1 (h1 : ∀ x, f(-x) 2 = -f(x) 2) : 2 = 2 := 
sorry

theorem problem2 (h1 : ∀ x, f(-x) 2 = -f(x) 2) : Set.Ioo (-1 : ℝ) 1 = Set.Ico (-1 : ℝ) 1 := 
sorry

theorem problem3 (h1 : ∀ x, f(-x) 2 = -f(x) 2) (t : ℝ) : 
(∀ x ∈ Set.Ioo 0 1, t * f(x) 2 ≥ 2^x - 2) ↔ (t ≥ 0) :=
sorry

end problem1_problem2_problem3_l641_641437


namespace problem_statement_l641_641514

-- Define the Möbius function µ(k)
def mu (k : ℕ) : ℤ :=
  if ∃ d : ℕ, d * d ∣ k then 0
  else (-1) ^ (Nat.factorization k).card

-- Define P_n(x) for a positive integer n
def P_n (n : ℕ) (x : ℤ) : ℤ :=
  ∑ k in (Finset.divisors n), mu k * x ^ (n / k)

-- Given a monic polynomial f(x) with integer coefficients of degree m and its roots z_i
variables {m : ℕ} (f : Polynomial ℤ) (hf : f.monic) (hdeg : f.degree = m)

-- Assume the roots of f(x) are z_1, z_2, ..., z_m
variable (roots : Fin (m + 1) → ℂ)
variable (hroots : ∀ i, f.eval (roots i) = 0)

-- Prove the given theorem for any positive integer n
theorem problem_statement (n : ℕ) (positive_n : 0 < n) :
  n ∣ ∑ i, P_n n (roots i) :=
  sorry

end problem_statement_l641_641514


namespace Misha_l641_641661

theorem Misha's_decision_justified :
  let A_pos := 7 in
  let A_neg := 4 in
  let B_pos := 4 in
  let B_neg := 1 in
  (B_pos / (B_pos + B_neg) > A_pos / (A_pos + A_neg)) := 
sorry

end Misha_l641_641661


namespace avg_rate_of_change_nonzero_l641_641495

theorem avg_rate_of_change_nonzero (Δx : ℝ) (h : Δx = 0 → false) : Δx ≠ 0 :=
by
  intro h1
  exact h h1
  sorry

end avg_rate_of_change_nonzero_l641_641495


namespace probability_not_greater_than_two_l641_641171

theorem probability_not_greater_than_two : 
  let cards := [1, 2, 3, 4]
  let favorable_cards := [1, 2]
  let total_scenarios := cards.length
  let favorable_scenarios := favorable_cards.length
  let prob := favorable_scenarios / total_scenarios
  prob = 1 / 2 :=
by
  sorry

end probability_not_greater_than_two_l641_641171


namespace probability_of_red_ball_l641_641488

theorem probability_of_red_ball (total_balls red_balls black_balls white_balls : ℕ)
  (h1 : total_balls = 7)
  (h2 : red_balls = 2)
  (h3 : black_balls = 4)
  (h4 : white_balls = 1) :
  (red_balls / total_balls : ℚ) = 2 / 7 :=
by {
  sorry
}

end probability_of_red_ball_l641_641488


namespace floor_a_n_squared_l641_641713

def seq_a : ℕ → ℝ
| 0       := 1
| (n + 1) := (seq_a n) / n + n / (seq_a n)

theorem floor_a_n_squared (n : ℕ) (h : 4 ≤ n) : 
  ∃ (a_n : ℝ), a_n = seq_a n ∧ ⌊(a_n)^2⌋ = n := 
sorry

end floor_a_n_squared_l641_641713


namespace no_two_heads_consecutive_probability_l641_641300

noncomputable def probability_no_two_heads_consecutive (total_flips : ℕ) : ℚ :=
  if total_flips = 10 then 144 / 1024 else 0

theorem no_two_heads_consecutive_probability :
  probability_no_two_heads_consecutive 10 = 9 / 64 :=
by
  unfold probability_no_two_heads_consecutive
  rw [if_pos rfl]
  norm_num
  sorry

end no_two_heads_consecutive_probability_l641_641300


namespace sequence_general_term_l641_641427

def S (n : ℕ) : ℤ := 3 * n^2 + 2 * n - 1

def a : ℕ → ℤ
| 1     := 4
| (n+1) := if n = 0 then 4 else 6 * (n + 1) - 1

theorem sequence_general_term (n : ℕ) : 
  a n = if n = 1 then 4 else 6 * n - 1 := 
sorry

end sequence_general_term_l641_641427


namespace is_factorization_l641_641668

-- given an equation A,
-- Prove A is factorization: 
-- i.e., x^3 - x = x * (x + 1) * (x - 1)

theorem is_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end is_factorization_l641_641668


namespace solve_for_a_and_b_l641_641948

noncomputable def z_eq (a b : ℝ): ℂ := (a + b * complex.I) * ((a + b * complex.I) + 2 * complex.I) * ((a + b * complex.I) - 3 * complex.I)

theorem solve_for_a_and_b (a b: ℝ) (h₁: 0 < a) (h₂: 0 < b) : z_eq a b = 8048 * complex.I → "More analysis required" :=
by sorry

end solve_for_a_and_b_l641_641948


namespace triangle_A1B1C1_acute_l641_641602

theorem triangle_A1B1C1_acute {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (hsum : α + β + γ = 180) :
  (α + β) / 2 < 90 ∧ (β + γ) / 2 < 90 ∧ (γ + α) / 2 < 90 :=
by
  -- Conditions: inscribed circle touches sides BC, CA, AB at A1, B1, C1 respectively,
  -- and angles of triangle ABC are α, β, γ.
  have h1 : α / 2 + β / 2 + γ / 2 = 90, from by linarith,
  -- Prove that each angle is less than 90 degrees
  split
  all_goals
  linarith
  sorry

end triangle_A1B1C1_acute_l641_641602


namespace max_product_two_integers_sum_2000_l641_641208

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641208


namespace sum_of_numbers_l641_641018

theorem sum_of_numbers (avg : ℝ) (num : ℕ) (h1 : avg = 5.2) (h2 : num = 8) : 
  (avg * num = 41.6) :=
by
  sorry

end sum_of_numbers_l641_641018


namespace prob2_l641_641089

-- Given conditions
def X : ℝ → ℝ := sorry -- Define the random variable X
def mu : ℝ := 500
def sigma : ℝ := 60
def dist_X : ProbabilityDistribution := normal_distribution mu sigma

-- Given probability condition
axiom prob1 : dist_X.cdf 440 = 0.16

-- Goal: Prove the required probability
theorem prob2 : dist_X.prob (set.Ici 560) = 0.16 := 
sorry

end prob2_l641_641089


namespace problem_solution_l641_641578

-- Definitions of conditions
variables (m n : ℕ)
variable (h_gcd : Nat.gcd m n = 6)
variable (h_lcm : Nat.lcm m n = 210)
variable (h_sum : m + n = 60)

-- Lean theorem stating the goal we want to prove
theorem problem_solution : (1/m:ℚ) + (1/n) = 1/21 :=
by 
  -- Using the provided conditions
  have h_prod : m * n = Nat.gcd m n * Nat.lcm m n, from Nat.mul_gcd_lcm m n,
  rw [h_gcd, h_lcm] at h_prod,
  have h_mn : m * n = 6 * 210 := h_prod,
  have h_mn_value : m * n = 1260 := by norm_num [h_mn],
  have h_rewrite : (1/m:ℚ) + (1/n) = (m + n) / (m * n) := by field_simp,
  rw [h_sum, h_mn_value, Nat.cast_add, Nat.cast_mul] at h_rewrite,
  norm_cast at h_rewrite,
  rw [h_rewrite],
  norm_num,
  sorry

end problem_solution_l641_641578


namespace rectangle_to_square_l641_641552

variable (k n : ℕ)

theorem rectangle_to_square (h1 : k > 5) (h2 : k * (k - 5) = n^2) : n = 6 := by 
  sorry

end rectangle_to_square_l641_641552


namespace fraction_meaningful_if_not_neg_two_l641_641643

theorem fraction_meaningful_if_not_neg_two {a : ℝ} : (a + 2 ≠ 0) ↔ (a ≠ -2) :=
by sorry

end fraction_meaningful_if_not_neg_two_l641_641643


namespace difference_is_20_l641_641665

def x : ℕ := 10

def a : ℕ := 3 * x

def b : ℕ := 20 - x

theorem difference_is_20 : a - b = 20 := 
by 
  sorry

end difference_is_20_l641_641665


namespace width_of_rectangular_plot_l641_641319

theorem width_of_rectangular_plot 
  (length : ℝ) 
  (poles : ℕ) 
  (distance_between_poles : ℝ) 
  (num_poles : ℕ) 
  (total_wire_length : ℝ) 
  (perimeter : ℝ) 
  (width : ℝ) :
  length = 90 ∧ 
  distance_between_poles = 5 ∧ 
  num_poles = 56 ∧ 
  total_wire_length = (num_poles - 1) * distance_between_poles ∧ 
  total_wire_length = 275 ∧ 
  perimeter = 2 * (length + width) 
  → width = 47.5 :=
by
  sorry

end width_of_rectangular_plot_l641_641319


namespace inequality_l641_641394

theorem inequality (a b c : ℝ) (h₀ : 0 < c) (h₁ : c < b) (h₂ : b < a) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 :=
by sorry

end inequality_l641_641394


namespace min_x_value_l641_641464

theorem min_x_value (x : Real) (hx : x > 0) :
  (log x ≥ log 3 + (1/3) * log x) → (x ≥ 3 * Real.sqrt 3) :=
by
  sorry

end min_x_value_l641_641464


namespace probability_non_red_l641_641144

def total_faces : ℕ := 10
def red_faces : ℕ := 5
def yellow_faces : ℕ := 3
def blue_face : ℕ := 1
def green_face : ℕ := 1

def non_red_faces : ℕ := yellow_faces + blue_face + green_face

theorem probability_non_red :
  non_red_faces.toRational / total_faces.toRational = (1 : ℚ) / 2 := 
  sorry

end probability_non_red_l641_641144


namespace total_palindromes_l641_641283

-- Define a palindrome for a 24-hour digital clock
def is_palindrome (time: String) : Prop :=
  time = time.reverse

-- Three-digit palindromes considering only valid hour and minute constraints
def valid_three_digit_palindromes : Nat :=
  3 * 6

-- Four-digit palindromes considering only valid hour and minute constraints
def valid_four_digit_palindromes : Nat :=
  10 * 6

-- Total number of palindromes
theorem total_palindromes : valid_three_digit_palindromes + valid_four_digit_palindromes = 78 :=
  by
    sorry

end total_palindromes_l641_641283


namespace group_B_equal_l641_641722

noncomputable def neg_two_pow_three := (-2)^3
noncomputable def minus_two_pow_three := -(2^3)

theorem group_B_equal : neg_two_pow_three = minus_two_pow_three :=
by sorry

end group_B_equal_l641_641722


namespace tan_inequality_general_conclusion_l641_641798

variable {θ : ℝ} (hθ : θ ∈ (0, π / 2))
variable {n : ℕ} (hn : 0 < n)
variable {m : ℝ}

theorem tan_inequality_general_conclusion
  (h1 : tan θ + 1 / tan θ ≥ 2)
  (h2 : tan θ + 4 / (tan θ)^2 ≥ 3)
  (h3 : tan θ + 27 / (tan θ)^3 ≥ 4)
  (h_gen : ∀ (n : ℕ), 0 < n → tan θ + m / (tan θ)^(n) ≥ n + 1) :
  m = n^n :=
sorry

end tan_inequality_general_conclusion_l641_641798


namespace equal_areas_of_subtriangles_l641_641912

theorem equal_areas_of_subtriangles
  (A B C P D E F : Point)
  (h_outside: P not in triangle_ABC)
  (h_intersections: AP ∩ BC = D ∧ BP ∩ CA = E ∧ CP ∩ AB = F)
  (h_equal_areas: area (triangle P B D) = area (triangle P C E) ∧ area (triangle P C E) = area (triangle P A F))
  : area (triangle A B C) = area (triangle P B D) := 
sorry

end equal_areas_of_subtriangles_l641_641912


namespace smallest_possible_value_of_a_l641_641984

theorem smallest_possible_value_of_a (a b : ℕ) :
  (∃ (r1 r2 r3 : ℕ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    (r1 * r2 * r3 = 2310) ∧
    (a = r1 + r2 + r3)) →
  a = 52 :=
begin
  sorry
end

end smallest_possible_value_of_a_l641_641984


namespace find_number_l641_641857

-- Given conditions
variables (z n s : ℝ)
hypothesis h1 : z ≠ 0
hypothesis h2 : z = Real.sqrt (n * z * s - 9 * s^2)
hypothesis h3 : z = 3

-- Proof statement
theorem find_number (h1 : z ≠ 0) (h2 : z = Real.sqrt (n * z * s - 9 * s^2)) (h3 : z = 3) : n = 3 + 3 * s := 
by
  sorry

end find_number_l641_641857


namespace Morse_code_number_of_distinct_symbols_l641_641039

def count_sequences (n : ℕ) : ℕ :=
  2 ^ n

theorem Morse_code_number_of_distinct_symbols :
  (count_sequences 1) + (count_sequences 2) + (count_sequences 3) + (count_sequences 4) + (count_sequences 5) = 62 :=
by
  simp [count_sequences]
  norm_num
  sorry

end Morse_code_number_of_distinct_symbols_l641_641039


namespace root_interval_range_l641_641472

theorem root_interval_range (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^3 - 3*x + m = 0) → (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end root_interval_range_l641_641472


namespace average_male_students_score_l641_641584

def average_male_score (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ) : ℕ :=
  let total_sum := (male_count + female_count) * total_avg
  let female_sum := female_count * female_avg
  let male_sum := total_sum - female_sum
  male_sum / male_count

theorem average_male_students_score
  (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ)
  (h1 : total_avg = 90) (h2 : female_avg = 92) (h3 : male_count = 8) (h4 : female_count = 20) :
  average_male_score total_avg female_avg male_count female_count = 85 :=
by {
  sorry
}

end average_male_students_score_l641_641584


namespace xy_value_l641_641015

theorem xy_value (x y : ℝ) (h : (x + complex.i) * (3 + y * complex.i) = 2 + 4 * complex.i) : x * y = 1 :=
sorry

end xy_value_l641_641015


namespace max_product_of_two_integers_sum_2000_l641_641254

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641254


namespace morse_code_symbols_l641_641040

def morse_code_symbols_count : ℕ :=
  let count n := 2^n
  (count 1) + (count 2) + (count 3) + (count 4) + (count 5)

theorem morse_code_symbols :
  morse_code_symbols_count = 62 :=
by
  unfold morse_code_symbols_count
  simp
  sorry

end morse_code_symbols_l641_641040


namespace equilateral_triangle_l641_641966

-- Define the context and parameters
variables {A B C A1 B1 C1 A0 B0 C0 : Type}

-- Definitions of medians intersecting circumcircles and equal areas of triangles
def medians_intersect_circumcircle (triangle : Type) (A0 B0 C0 : Type) : Prop :=
  -- Placeholder definition for medians intersecting the circumcircle
  -- Actual geometric definitions would be needed here.
  sorry

def areas_equal (triangle : Type) (A0 B0 C0 : Type) : Prop :=
  -- Placeholder definition for equal areas
  -- Proper geometric properties and area comparison would be defined here.
  sorry

-- Main theorem to prove
theorem equilateral_triangle {A B C A1 B1 C1 A0 B0 C0: Type}
  (h1: medians_intersect_circumcircle (triangle A B C) A0 B0 C0)
  (h2: areas_equal (triangle A B C) A0 B0 C0) : equilateral (triangle A B C) :=
sorry

end equilateral_triangle_l641_641966


namespace range_of_g_l641_641765

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x - Real.arctan x

theorem range_of_g :
  ∀ x ∈ set.Icc (-1:ℝ) 1, g x ∈ set.Icc (Real.pi / 4) (3 * Real.pi / 4) := 
sorry

end range_of_g_l641_641765


namespace smallest_positive_period_of_f_center_of_symmetry_of_f_intervals_of_monotonic_increase_of_f_l641_641442

noncomputable def f (x : ℝ) : ℝ := 
  sin (2*x + π/3) + sqrt 3 - 2*sqrt 3 * (cos x)^2 + 1

theorem smallest_positive_period_of_f : (∀ x : ℝ, f (x + π) = f x) ∧ ¬ (∃ T : ℝ, T > 0 ∧ T < π ∧ ∀ x : ℝ, f (x + T) = f x) :=
by sorry

theorem center_of_symmetry_of_f (k : ℤ) : 
  ∃ c : ℝ, (∀ x : ℝ, f (c + x) = f (c - x)) ∧ c = (k * π / 2 + π / 6) :=
by sorry

theorem intervals_of_monotonic_increase_of_f (k : ℤ) : 
  (∀ x : ℝ, (k * π - π / 12) ≤ x ∧ x ≤ (k * π + 5 * π / 12) → ∀ y : ℝ, (k * π - π / 12) ≤ y ∧ y ≤ x → f y ≤ f x) ∧ 
  (∀ x : ℝ, x < (k * π - π / 12) ∨ x > (k * π + 5 * π / 12) → ∃ y : ℝ, y > x ∧ y < (k * π + π / 2) ∧ f x > f y) :=
by sorry

end smallest_positive_period_of_f_center_of_symmetry_of_f_intervals_of_monotonic_increase_of_f_l641_641442


namespace BC_length_l641_641877

def right_triangle_cosine_and_side_length (A B C : Type*) [right_triangle A B C] (c : ℝ) : Prop :=
  ∃ (a b : ℝ),  ∠B = 90 ∧ cos A = 3/5 ∧ AB = 10 ∧ BC = c ∧ c = 50 / 3

theorem BC_length {A B C : Type*} [right_triangle A B C] : 
  (cos A = 3 / 5) ∧ (AB = 10) ∧ (∠B = 90) → (BC = 50 / 3) :=
by sorry

end BC_length_l641_641877


namespace max_dot_product_sum_l641_641446

variables (a b e : EuclideanSpace ℝ (Fin 2))
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 2) (hab : inner a b = 1) (he_unit : ∥e∥ = 1)

theorem max_dot_product_sum :
  (⨅ e : EuclideanSpace ℝ (Fin 2), ∥e∥ = 1) (inner a e).abs + (inner b e).abs = Real.sqrt 7 :=
by
  sorry

end max_dot_product_sum_l641_641446


namespace max_product_of_sum_2000_l641_641243

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641243


namespace eccentricity_range_l641_641350

open Real

variables {a b c : ℝ} (e : ℝ)

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def eccentricity : ℝ :=
  √(1 + (b^2 / a^2))

theorem eccentricity_range
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (h_line_slopes_both_branches : b / a > 1)
  (h_line_slope_three_only_right : b / a < 3)
  (h_sq_b_a : c = √(a^2 + b^2)) :
  sqrt 2 < eccentricity a b ∧ eccentricity a b < sqrt 10 :=
by
  sorry

end eccentricity_range_l641_641350


namespace harkamal_grapes_purchase_l641_641000

variable (G : ℕ) -- G represents the amount of kg of grapes purchased.
axiom rate_grapes : ℕ := 80
axiom rate_mangoes : ℕ := 55
axiom mangoes_quantity : ℕ := 9
axiom total_paid : ℕ := 1135

theorem harkamal_grapes_purchase :
  rate_grapes * G + rate_mangoes * mangoes_quantity = total_paid → G = 8 :=
by
  sorry

end harkamal_grapes_purchase_l641_641000


namespace polygon_with_36_degree_exterior_angles_is_decagon_l641_641870

-- Axioms and definitions based on the problem's conditions
axiom sum_exterior_angles : ∀ (n : ℕ), n ≥ 3 → (Polygon.n n).sum_exterior_angles = 360

-- Problem statement made into a Lean theorem:
theorem polygon_with_36_degree_exterior_angles_is_decagon :
  ∃ (n : ℕ), n = 10 ∧ ∀ (degree : ℕ), degree = 36 → 360 / degree = n :=
begin
  -- Formalization of conditions and proof would go here
  sorry
end

end polygon_with_36_degree_exterior_angles_is_decagon_l641_641870


namespace problem_statement_l641_641010

theorem problem_statement :
  let triangle := 1 in
  let O := -1 in
  let square := 0 in
  (square + triangle) * O = -1 :=
by
  sorry

end problem_statement_l641_641010


namespace find_value_of_a_l641_641832

theorem find_value_of_a (a : ℝ) :
  (∃ (l₁ l₂ : ℝ → ℝ → Prop), 
    l₁ (λ x y, a * x - 3 * y + 1 = 0) ∧ 
    l₂ (λ x y, x + (a + 1) * y + 1 = 0) ∧ 
    (∀ x y, l₁ x y -> l₂ x y -> x / sqrt (x * x + y * y) * (x + a) + y / sqrt (x * x + y * y) * (-3) = 0)) ->
  a = -3/2 := 
sorry

end find_value_of_a_l641_641832


namespace determine_digit_I_l641_641500

theorem determine_digit_I (F I V E T H R N : ℕ) (hF : F = 8) (hE_odd : E = 1 ∨ E = 3 ∨ E = 5 ∨ E = 7 ∨ E = 9)
  (h_diff : F ≠ I ∧ F ≠ V ∧ F ≠ E ∧ F ≠ T ∧ F ≠ H ∧ F ≠ R ∧ F ≠ N 
             ∧ I ≠ V ∧ I ≠ E ∧ I ≠ T ∧ I ≠ H ∧ I ≠ R ∧ I ≠ N 
             ∧ V ≠ E ∧ V ≠ T ∧ V ≠ H ∧ V ≠ R ∧ V ≠ N 
             ∧ E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ N 
             ∧ T ≠ H ∧ T ≠ R ∧ T ≠ N 
             ∧ H ≠ R ∧ H ≠ N 
             ∧ R ≠ N)
  (h_verify_sum : (10^3 * 8 + 10^2 * I + 10 * V + E) + (10^4 * T + 10^3 * H + 10^2 * R + 11 * E) = 10^3 * N + 10^2 * I + 10 * N + E) :
  I = 4 := 
sorry

end determine_digit_I_l641_641500


namespace symmetric_point_origin_l641_641593

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

theorem symmetric_point_origin (x y : ℝ) (h : (x, y) = (-2, 3)) :
  symmetric_point (x, y) = (2, -3) :=
by
  rw [h]
  unfold symmetric_point
  simp
  sorry

end symmetric_point_origin_l641_641593


namespace last_three_digits_of_7_pow_103_l641_641368

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l641_641368


namespace smallest_x_domain_of_g_g_l641_641095

def g (x : ℝ) : ℝ := real.sqrt (x - 5)

theorem smallest_x_domain_of_g_g :
  ∃ x : ℝ, x = 30 ∧ ∀ y, y ≥ 30 → real.sqrt (real.sqrt (y - 5) - 5) = g (g y) :=
by
  sorry

end smallest_x_domain_of_g_g_l641_641095


namespace arts_school_probability_l641_641861

theorem arts_school_probability :
  let cultural_courses := 3
  let arts_courses := 3
  let total_periods := 6
  let total_arrangements := Nat.factorial total_periods
  let no_adjacent_more_than_one_separator := (72 + 216 + 144)
  (no_adjacent_more_than_one_separator : ℝ) / (total_arrangements : ℝ) = (3 / 5 : ℝ) := 
by 
  sorry

end arts_school_probability_l641_641861


namespace max_product_two_integers_sum_2000_l641_641207

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641207


namespace max_product_l641_641225

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641225


namespace zoo_ticket_sales_l641_641690

-- Define the number of total people, number of adults, and ticket prices
def total_people : ℕ := 254
def num_adults : ℕ := 51
def adult_ticket_price : ℕ := 28
def kid_ticket_price : ℕ := 12

-- Define the number of kids as the difference between total people and number of adults
def num_kids : ℕ := total_people - num_adults

-- Define the revenue from adult tickets and kid tickets
def revenue_adult_tickets : ℕ := num_adults * adult_ticket_price
def revenue_kid_tickets : ℕ := num_kids * kid_ticket_price

-- Define the total revenue
def total_revenue : ℕ := revenue_adult_tickets + revenue_kid_tickets

-- Theorem to prove the total revenue equals 3864
theorem zoo_ticket_sales : total_revenue = 3864 :=
  by {
    -- sorry allows us to skip the proof
    sorry
  }

end zoo_ticket_sales_l641_641690


namespace alpha_plus_beta_eq_118_l641_641621

theorem alpha_plus_beta_eq_118 (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96 * x + 2209) / (x^2 + 63 * x - 3969)) : α + β = 118 :=
by
  sorry

end alpha_plus_beta_eq_118_l641_641621


namespace john_sales_percentage_l641_641084

noncomputable def percentage_buyers (houses_visited_per_day : ℕ) (work_days_per_week : ℕ) (weekly_sales : ℝ) (low_price : ℝ) (high_price : ℝ) : ℝ :=
  let total_houses_per_week := houses_visited_per_day * work_days_per_week
  let average_sale_per_customer := (low_price + high_price) / 2
  let total_customers := weekly_sales / average_sale_per_customer
  (total_customers / total_houses_per_week) * 100

theorem john_sales_percentage :
  percentage_buyers 50 5 5000 50 150 = 20 := 
by 
  sorry

end john_sales_percentage_l641_641084


namespace circles_coloring_l641_641187

theorem circles_coloring :
  ∃ (colors : Finset ℕ) (circles : list ℕ),
    colors.card = 4 ∧
    circles.length = 5 ∧
    (∀ (i j : ℕ), i ≠ j → (circles.nth i ≠ circles.nth j)) →  -- Page connection condition
    circles_colorings_count = 756 :=   -- Total colorings count
sorry

end circles_coloring_l641_641187


namespace min_b_minus_a_l641_641816

noncomputable def f (x : ℝ) : ℝ := ∑ k in (finset.range 2014), (-1) ^ k * (x ^ k / (k + 1))
noncomputable def g (x : ℝ) : ℝ := ∑ k in (finset.range 2014), (-1) ^ (k + 1) * (x ^ k / (k + 1))

def F (x : ℝ) : ℝ := f (x + 3) * g (x - 4)

theorem min_b_minus_a : ∃ (a b : ℤ), (∀ x : ℝ, F x = 0 → x ≥ ↑a ∧ x ≤ ↑b) ∧ b - a = 10 := sorry

end min_b_minus_a_l641_641816


namespace inequality_proof_l641_641943

variable (a b c d : ℝ)

theorem inequality_proof (ha : 0 ≤ a ∧ a ≤ 1)
                       (hb : 0 ≤ b ∧ b ≤ 1)
                       (hc : 0 ≤ c ∧ c ≤ 1)
                       (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1) ^ 2 ≥ 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :=
sorry

end inequality_proof_l641_641943


namespace median_impossibility_l641_641124

theorem median_impossibility
  (A B C M : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
  (BC : Line B C)
  (on_BC : M ∈ BC)
  (r_ACM : ℕ)
  (r_ABM : ℕ)
  (h1 : 2 * r_ACM = r_ABM) :
  ¬(AM.is_median_of_triangle_ABC) :=
sorry

end median_impossibility_l641_641124


namespace find_AC_length_l641_641026

variables {A B C D E : Type*}
variables [circle A B C] [right_angle A B C] [on_segment D A C] [on_segment E A B]
variables (AC AD DE EC BE : ℝ)

noncomputable def AC_length : Prop :=
  ∀ (x : ℝ), AC = x → AD = x / 3 → DE = x / 3 → EC = x / 3 → BE = 2 → x = 3

theorem find_AC_length : AC_length :=
by {
  intro x,
  assume AC_eq AD_eq DE_eq EC_eq BE_eq,
  have h1 : ∀ x, BE^2 + AC^2 = (2 * x / 3) ^ 2, from sorry,
  have h2 : ∀ x, AD_eq + . AC = 3, from sorry,
  simp at *,
  exact 3,
  sorry
}

end find_AC_length_l641_641026


namespace distinct_sequences_count_l641_641836

theorem distinct_sequences_count :
  (let available_letters := ['E', 'X', 'A', 'M', 'E'],
       sequences := {s : List Char | s.head = 'P' 
                     ∧ s.getLast! = 'L' 
                     ∧ ∀ x, x ∈ s → x ≠ 'P' 
                     ∧ ∀ x, x ∈ s → x ≠ 'L' 
                     ∧ s.toFinset.card = 5 
                     ∧ s.filter (fun c => c = 'E').length ≤ 2},
       num_sequences := sequences.toList.length)
  in num_sequences = 24 :=
by
  sorry

end distinct_sequences_count_l641_641836


namespace find_a20_l641_641588

variable {a : ℕ → ℤ}
variable {d : ℤ}
variable {a_1 : ℤ}

def isArithmeticSeq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def formsGeomSeq (a1 a3 a4 : ℤ) : Prop :=
  (a3 - a1)^2 = a1 * (a4 - a1)

theorem find_a20 (h1 : isArithmeticSeq a (-2))
                 (h2 : formsGeomSeq a_1 (a_1 + 2*(-2)) (a_1 + 3*(-2)))
                 (ha1 : a_1 = 8) :
  a 20 = -30 :=
by
  sorry

end find_a20_l641_641588


namespace collinear_U_P_V_l641_641520

variables {O1 O2 P Q U V : Type} [metric_space O1] [metric_space O2]
variables {C1 : Metric.Conemetric O1} {C2 : Metric.ConeMetric O2}

noncomputable def angles := sorry

theorem collinear_U_P_V (hCollinear: U ∈ (line P Q ∩ line Q V)) 
(hPU : P ∈ circle O1) (hPQ: Q ∈ circle O1)
(hQV : Q ∈ circle O2) (hVO : V ∈ circle O2) :
  ∠UQV = ∠O1QO2 :=
begin
  sorry
end

end collinear_U_P_V_l641_641520


namespace greatest_product_sum_2000_l641_641214

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641214


namespace no_consecutive_heads_probability_l641_641303

def prob_no_consecutive_heads (n : ℕ) : ℝ := 
  -- This is the probability function for no consecutive heads
  if n = 10 then (9 / 64) else 0

theorem no_consecutive_heads_probability :
  prob_no_consecutive_heads 10 = 9 / 64 :=
by
  -- Proof would go here
  sorry

end no_consecutive_heads_probability_l641_641303


namespace proj_onto_b_l641_641909

noncomputable theory

open_locale big_operators

/-- A type representing vectors in ℝ³. --/
@[derive add_comm_group]
def R3 : Type := fin 3 → ℝ

/-- Orthogonality condition --/
def orthogonal (a b : R3) : Prop := dot_product a b = 0

/-- Projection function --/
def proj (u v : R3) : R3 := ((u • u) / (dot_product u u)) • u

variable (a b : R3)

-- Given conditions
variables (orth_ab : orthogonal a b)
          (proj_onto_a : proj a ⟨4, -4, 1⟩ = ⟨-∞, -∞, ∞⟩) -- Note: Sould contain actual PM values.

-- The goal to prove
theorem proj_onto_b :
  proj b ⟨4, -4, 1⟩ = ⟨32/7, -20/7, 5/7⟩ :=
sorry

end proj_onto_b_l641_641909


namespace cookies_eq_23_l641_641539

def total_packs : Nat := 27
def cakes : Nat := 4
def cookies : Nat := total_packs - cakes

theorem cookies_eq_23 : cookies = 23 :=
by
  -- Proof goes here
  sorry

end cookies_eq_23_l641_641539


namespace extreme_values_of_f_inequality_for_a_geq_2_sum_x1_x2_geq_l641_641824

-- Problem 1: Prove the extreme values of the function
theorem extreme_values_of_f (x : ℝ) (h1 : x > 0) : 
  f x = ln x - x^2 + x → ∃! y, y = 1 ∧ f y = 0 ∧ ∀ z, f z ≤ f 1 :=
sorry

-- Problem 2: Prove the inequality
theorem inequality_for_a_geq_2 (a x : ℝ) (h1 : a ≥ 2) (h2 : x > 0) :
  f x = ln x - x^2 + x →
  f x < (a / 2 - 1) * x^2 + a * x - 1 :=
sorry

-- Problem 3: Prove the given condition
theorem sum_x1_x2_geq (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0)
  (h3 : f x1 + f x2 + 2 * (x1^2 + x2^2) + x1 * x2 = 0) :
  x1 + x2 ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end extreme_values_of_f_inequality_for_a_geq_2_sum_x1_x2_geq_l641_641824


namespace max_product_of_two_integers_sum_2000_l641_641194

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641194


namespace exists_root_in_interval_l641_641604

noncomputable def f (x : ℝ) : ℝ := log x - 1 / x

theorem exists_root_in_interval : ∃ x ∈ set.Ioo 2 3, f x = 0 := by
  sorry

end exists_root_in_interval_l641_641604


namespace area_of_triangle_ABC_l641_641444

noncomputable def f (x : ℝ) := Real.sin (π * x + π / 4)
noncomputable def g (x : ℝ) := Real.cos (π * x + π / 4)
def interval : Set ℝ := Set.Icc (-5 / 4) (7 / 4)

theorem area_of_triangle_ABC :
  let A := (-1 : ℝ, -Real.sqrt 2 / 2)
  let B := (0 : ℝ, Real.sqrt 2 / 2)
  let C := (1 : ℝ, -Real.sqrt 2 / 2)
  Set.mem A interval ∧ Set.mem B interval ∧ Set.mem C interval 
  ∧ (f A.1 = A.2 ∧ g A.1 = A.2)
  ∧ (f B.1 = B.2 ∧ g B.1 = B.2)
  ∧ (f C.1 = C.2 ∧ g C.1 = C.2)
  ∧ (∃ x, f x = g x ∧ x = A.1 ∨ x = B.1 ∨ x = C.1)
  → (1 / 2 : ℝ) * (Real.sqrt 2 / 2 - (- (Real.sqrt 2 / 2))) * (1 - (-1)) = Real.sqrt 2 := 
sorry

end area_of_triangle_ABC_l641_641444


namespace polynomial_division_result_l641_641098

theorem polynomial_division_result :
  let p : Polynomial ℚ := 3 * Polynomial.X^4 + 9 * Polynomial.X^3 - 6 * Polynomial.X^2 + 2 * Polynomial.X - 4 in
  let g : Polynomial ℚ := Polynomial.X^2 - 2 * Polynomial.X + 3 in
  let ⟨m, s⟩ := Polynomial.div_mod p g in
  s.degree < g.degree → m.eval 1 + s.eval (-2) = 7 := by
  intros p g H
  let ⟨m, s⟩ := Polynomial.div_mod p g
  have h1 : m.eval 1 = 6 := sorry
  have h2 : s.eval (-2) = 1 := sorry
  rw [h1, h2]
  norm_num

end polynomial_division_result_l641_641098


namespace degree_g_is_six_l641_641141

theorem degree_g_is_six 
  (f g : Polynomial ℂ) 
  (h : Polynomial ℂ) 
  (h_def : h = f.comp g + Polynomial.X * g) 
  (deg_h : h.degree = 7) 
  (deg_f : f.degree = 3) 
  : g.degree = 6 := 
sorry

end degree_g_is_six_l641_641141


namespace fred_basketball_games_total_l641_641389

theorem fred_basketball_games_total (games_this_year : ℕ) (games_last_year : ℕ): 
  games_this_year = 60 → 
  games_last_year = 25 → 
  games_this_year + games_last_year = 85 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end fred_basketball_games_total_l641_641389


namespace morse_code_symbols_l641_641044

def morse_code_symbols_count : ℕ :=
  let count n := 2^n
  (count 1) + (count 2) + (count 3) + (count 4) + (count 5)

theorem morse_code_symbols :
  morse_code_symbols_count = 62 :=
by
  unfold morse_code_symbols_count
  simp
  sorry

end morse_code_symbols_l641_641044


namespace max_distance_point_to_circle_l641_641430

noncomputable def circle_center : ℝ × ℝ := (3, 4)
noncomputable def circle_radius : ℝ := 5
noncomputable def point : ℝ × ℝ := (2, 3)

theorem max_distance_point_to_circle : 
  let d := real.sqrt ((point.1 - circle_center.1) ^ 2 + (point.2 - circle_center.2) ^ 2) in
  d + circle_radius = real.sqrt 2 + 5 :=
by sorry

end max_distance_point_to_circle_l641_641430


namespace lori_earnings_equation_l641_641537

theorem lori_earnings_equation : 
  ∀ (white_cars red_cars : ℕ) 
    (white_rent_cost red_rent_cost minutes_per_hour rental_hours : ℕ), 
  white_cars = 2 →
  red_cars = 3 →
  white_rent_cost = 2 →
  red_rent_cost = 3 →
  minutes_per_hour = 60 →
  rental_hours = 3 →
  (white_cars * white_rent_cost + red_cars * red_rent_cost) * rental_hours * minutes_per_hour = 2340 := 
by
  intros white_cars red_cars white_rent_cost red_rent_cost minutes_per_hour rental_hours 
         h_white_cars h_red_cars h_white_rent_cost h_red_rent_cost h_minutes_per_hour h_rental_hours
  rw [h_white_cars, h_red_cars, h_white_rent_cost, h_red_rent_cost, h_minutes_per_hour, h_rental_hours]
  calc
    (2 * 2 + 3 * 3) * 3 * 60 = (4 + 9) * 3 * 60 : by rw [mul_add, mul_comm 3]
    ... = 13 * 3 * 60 : by rw [add_mul]
    ... = 39 * 60 : by rw [mul_assoc]
    ... = 2340  : by norm_num
  done

end lori_earnings_equation_l641_641537


namespace small_truck_capacity_l641_641545

theorem small_truck_capacity :
  ∀ (total_fruits large_truck_capacity small_truck_capacity large_trucks_used : ℕ),
    total_fruits = 134 →
    large_truck_capacity = 15 →
    large_trucks_used = 8 →
    small_truck_capacity = total_fruits - large_truck_capacity * large_trucks_used →
    (total_fruits - large_truck_capacity * large_trucks_used) = 14 →
    small_truck_capacity = 14 :=
by
  intros total_fruits large_truck_capacity small_truck_capacity large_trucks_used
  intros h1 h2 h3 h4 h5
  rw [h4, h5]
  exact h5

end small_truck_capacity_l641_641545


namespace no_two_heads_consecutive_probability_l641_641297

noncomputable def probability_no_two_heads_consecutive (total_flips : ℕ) : ℚ :=
  if total_flips = 10 then 144 / 1024 else 0

theorem no_two_heads_consecutive_probability :
  probability_no_two_heads_consecutive 10 = 9 / 64 :=
by
  unfold probability_no_two_heads_consecutive
  rw [if_pos rfl]
  norm_num
  sorry

end no_two_heads_consecutive_probability_l641_641297


namespace total_cost_correct_l641_641866

def serviceCost_per_vehicle := 2.20
def fuelCost_per_liter := 0.70
def miniVan_tank_capacity := 65.0
def truck_multiplier := 1.2
def number_of_miniVans := 4
def number_of_trucks := 2

def truck_tank_capacity := miniVan_tank_capacity * (1 + truck_multiplier)
def cost_per_miniVan := miniVan_tank_capacity * fuelCost_per_liter
def cost_per_truck := truck_tank_capacity * fuelCost_per_liter
def total_service_cost := serviceCost_per_vehicle * (number_of_miniVans + number_of_trucks)
def total_fuel_cost_for_miniVans := cost_per_miniVan * number_of_miniVans
def total_fuel_cost_for_trucks := cost_per_truck * number_of_trucks
def total_cost := total_service_cost + total_fuel_cost_for_miniVans + total_fuel_cost_for_trucks

theorem total_cost_correct : total_cost = 395.40 :=
by sorry

end total_cost_correct_l641_641866


namespace cannot_fill_table_with_primes_l641_641502

theorem cannot_fill_table_with_primes : 
  ¬ ∃ (a : Fin 9 → Fin 2002 → ℕ), 
    (∀ j : Fin 9, Prime (Σ (i : Fin 2002), a j i)) ∧ 
    (∀ i : Fin 2002, Prime (Σ (j : Fin 9), a j i)) := by
  sorry

end cannot_fill_table_with_primes_l641_641502


namespace Robin_hair_initial_length_l641_641570

theorem Robin_hair_initial_length (x : ℝ) (h1 : x + 8 - 20 = 2) : x = 14 :=
by
  sorry

end Robin_hair_initial_length_l641_641570


namespace integer_between_sqrt3_add1_and_sqrt11_l641_641603

theorem integer_between_sqrt3_add1_and_sqrt11 :
  (∀ x, (1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) → (2 < Real.sqrt 3 + 1 ∧ Real.sqrt 3 + 1 < 3) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) ∧ x = 3) :=
by
  sorry

end integer_between_sqrt3_add1_and_sqrt11_l641_641603


namespace min_value_of_f_l641_641822

noncomputable def f (a x : ℝ) : ℝ := (a + x^2) / x

theorem min_value_of_f (a b : ℝ) (ha : 0 < a) (hb : b > sqrt a) :
  ∃ x ∈ Ioo 0 b, f a x = 2 * sqrt a :=
sorry

end min_value_of_f_l641_641822


namespace directors_dividends_correct_l641_641646

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l641_641646


namespace inequality_proof_l641_641916

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
by 
  sorry

end inequality_proof_l641_641916


namespace count_complex_numbers_l641_641845

theorem count_complex_numbers (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum_le_5 : a + b ≤ 5) :
  {z : ℂ | ∃ a b : ℕ, (a > 0) ∧ (b > 0) ∧ (a + b ≤ 5) ∧ (z = a + b * (complex.I : ℂ))}.to_finset.card = 10 :=
by {
  -- Proof omitted
  sorry
}

end count_complex_numbers_l641_641845


namespace greatest_product_sum_2000_eq_1000000_l641_641235

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641235


namespace inequality_proof_l641_641942

variable (a b c d : ℝ)

theorem inequality_proof (ha : 0 ≤ a ∧ a ≤ 1)
                       (hb : 0 ≤ b ∧ b ≤ 1)
                       (hc : 0 ≤ c ∧ c ≤ 1)
                       (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1) ^ 2 ≥ 4 * (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) :=
sorry

end inequality_proof_l641_641942


namespace village_Y_increase_rate_l641_641189

theorem village_Y_increase_rate :
  ∃ r : ℕ, 
    (68_000 - 1_200 * 13 = 42_000 + r * 13) ∧ 
    r = 800 :=
by
  existsi 800
  split
  · calc
      68_000 - 1_200 * 13
          = 68_000 - 15_600 : by simp
      ... = 52_400 : by simp
      ... = 42_000 + 800 * 13 : by { rw [nat.add_comm, nat.mul_comm], norm_num }

  · rfl


end village_Y_increase_rate_l641_641189


namespace max_product_of_sum_2000_l641_641247

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641247


namespace single_percentage_reduction_l641_641161

theorem single_percentage_reduction (P : ℝ) (h1 : 0 < P) :
  let first_reduction := 0.75 * P,
      second_reduction := 0.30 * P in
  1 - (second_reduction / P) = 0.7 :=
by
  have first_reduction := 0.75 * P,
  have second_reduction := 0.3 * P,
  calc
    1 - (second_reduction / P)
        = 1 - (0.3 * P / P) : by rw second_reduction
    ... = 1 - 0.3   : by rw ← mul_div_cancel 0.3 (ne_of_gt h1)
    ... = 0.7       : by norm_num

end single_percentage_reduction_l641_641161


namespace calc_result_l641_641467

theorem calc_result : (-2 * -3 + 2) = 8 := sorry

end calc_result_l641_641467


namespace triangle_inequality_l641_641852

theorem triangle_inequality (a : ℝ) (h1 : a + 3 > 5) (h2 : a + 5 > 3) (h3 : 3 + 5 > a) :
  2 < a ∧ a < 8 :=
by {
  sorry
}

end triangle_inequality_l641_641852


namespace tetrahedron_volume_and_height_l641_641684

open Real

def point := ℝ × ℝ × ℝ

def A1 : point := (-4, 2, 6)
def A2 : point := (2, -3, 0)
def A3 : point := (-10, 5, 8)
def A4 : point := (-5, 2, -4)

noncomputable def volume_tetrahedron (A1 A2 A3 A4: point) : ℝ :=
  let v1 := (A2.1 - A1.1, A2.2 - A1.2, A2.3 - A1.3)
  let v2 := (A3.1 - A1.1, A3.2 - A1.2, A3.3 - A1.3)
  let v3 := (A4.1 - A1.1, A4.2 - A1.2, A4.3 - A1.3)
  (1 / 6) * abs (
    v1.1 * (v2.2 * v3.3 - v2.3 * v3.2) -
    v1.2 * (v2.1 * v3.3 - v2.3 * v3.1) +
    v1.3 * (v2.1 * v3.2 - v2.2 * v3.1)
  )

noncomputable def height (A1 A2 A3 A4: point) : ℝ :=
  let v1 := (A2.1 - A1.1, A2.2 - A1.2, A2.3 - A1.3)
  let v2 := (A3.1 - A1.1, A3.2 - A1.2, A3.3 - A1.3)
  let cross_product := (
    v1.2 * v2.3 - v1.3 * v2.2,
    v1.3 * v2.1 - v1.1 * v2.3,
    v1.1 * v2.2 - v1.2 * v2.1
  )
  let area := (1 / 2) * sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)
  (3 * volume_tetrahedron A1 A2 A3 A4) / area

theorem tetrahedron_volume_and_height :
  volume_tetrahedron A1 A2 A3 A4 = 56 / 3 ∧ height A1 A2 A3 A4 = 4 :=
by {
  sorry
}

end tetrahedron_volume_and_height_l641_641684


namespace k_minus_m_value_l641_641022

-- Definitions of conditions
def line1 (k x : ℝ) : ℝ := k * x + 1
def circle (k m x y : ℝ) : ℝ := x^2 + y^2 + k * x + m * y - 4
def symmetry_line (x y : ℝ) : ℝ := x + y - 1

-- The main statement to be proved
theorem k_minus_m_value (k m : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), 
    line1 k x1 = y1 ∧ line1 k x2 = y2 ∧ 
    circle k m x1 y1 = 0 ∧ circle k m x2 y2 = 0 ∧ 
    (x1 + y1 - 1) = -(x2 + y2 - 1)) → 
  k - m = 4 :=
by
  sorry

end k_minus_m_value_l641_641022


namespace unique_products_count_l641_641006

def set_original : Set ℕ := {1, 2, 4, 7, 13}

def set_reduced : Set ℕ := {2, 4, 7, 13}

def valid_products : Set ℕ :=
{ x | ∃ a b c d : ℕ, 
  (a ∈ set_reduced ∧ b ∈ set_reduced ∧ c ∈ set_reduced ∧ d ∈ set_reduced) ∧
  (a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d) ∧
  ((x = a * b ∧ a ≠ b) ∨ 
   (x = a * b * c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) ∨ 
   (x = a * b * c * d ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ d)) }

theorem unique_products_count : ∀ S : Set ℕ, S = valid_products → S.card = 11 :=
by
  sorry

end unique_products_count_l641_641006


namespace cars_meet_cars_apart_l641_641807

section CarsProblem

variable (distance : ℕ) (speedA speedB : ℕ) (distanceToMeet distanceApart : ℕ)

def meetTime := distance / (speedA + speedB)
def apartTime1 := (distance - distanceApart) / (speedA + speedB)
def apartTime2 := (distance + distanceApart) / (speedA + speedB)

theorem cars_meet (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85):
  meetTime distance speedA speedB = 9 / 4 := by
  sorry

theorem cars_apart (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85) (h4: distanceApart = 50):
  apartTime1 distance speedA speedB distanceApart = 2 ∧ apartTime2 distance speedA speedB distanceApart = 5 / 2 := by
  sorry

end CarsProblem

end cars_meet_cars_apart_l641_641807


namespace sum_v1_to_v4_l641_641191

noncomputable def v0 : ℝ × ℝ := (2, 1)
noncomputable def w0 : ℝ × ℝ := (0, 5)

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_sq := v.1 * v.1 + v.2 * v.2
  let scalar := dot_product / norm_sq
  (scalar * v.1, scalar * v.2)

noncomputable def vn (n : ℕ) : ℝ × ℝ :=
  if n = 0 then v0 else proj (wn (n - 1)) v0
noncomputable def wn (n : ℕ) : ℝ × ℝ :=
  if n = 0 then w0 else proj (vn n) w0

theorem sum_v1_to_v4 : vn 1 + vn 2 + vn 3 + vn 4 = (8, 4) := by
  sorry

end sum_v1_to_v4_l641_641191


namespace area_of_CMKD_is_19_over_56_l641_641934

variables {A B C D M K : Point}
variables [parallelogram : Parallelogram ABCD]
variables (BM MC : ℝ) (area_ABCD : ℝ) (area_CMKD : ℝ)

-- Conditions -- 
def M_divides_BC_in_ratio (BM : ℝ) (MC : ℝ) : Prop :=
  BM / MC = 3

def AM_intersects_BD_at_K : Prop :=
  ∃ K, line_through A M ∧ line_through B D ∧ intersect (line_through A M) (line_through B D) = K

def area_parallelogram_is_1 (area_ABCD : ℝ) : Prop :=
  area parallelogram = 1

-- To Prove --
theorem area_of_CMKD_is_19_over_56 
  (hM : M_divides_BC_in_ratio BM MC)
  (hK : AM_intersects_BD_at_K)
  (harea : area_parallelogram_is_1 area_ABCD) : 
  area_CMKD = 19 / 56 :=
sorry

end area_of_CMKD_is_19_over_56_l641_641934


namespace employee_salary_l641_641629

theorem employee_salary (total_pay : ℕ) (ratio : ℚ) (y : ℚ) (x : ℚ) : 
  total_pay = 616 → ratio = 1.20 → total_pay = (y + x) → x = ratio * y → y = 280 := 
by {
  intros h_total h_ratio h_sum h_relation,
  sorry
}

end employee_salary_l641_641629


namespace solveFamousAmericansProblem_l641_641972

def famousAmericansProblem (total july: ℕ) : Prop :=
  (july / total.toReal) * 100 = 12.5

theorem solveFamousAmericansProblem :
  famousAmericansProblem 120 15 :=
by
  sorry

end solveFamousAmericansProblem_l641_641972


namespace add_water_to_achieve_target_concentration_l641_641007

-- Definitions based on conditions
def initial_volume : ℝ := 50 -- initial volume of solution in ounces
def initial_concentration : ℝ := 0.4 -- initial concentration of sodium chloride
def target_concentration : ℝ := 0.25 -- target concentration of sodium chloride
def sodium_chloride_mass : ℝ := initial_concentration * initial_volume -- mass of sodium chloride in ounces

-- The proof goal
theorem add_water_to_achieve_target_concentration (w : ℝ)
  (h : sodium_chloride_mass / (initial_volume + w) = target_concentration) :
  w = 30 :=
by 
  have initial_volume := 50.0
  have initial_concentration := 0.4
  have target_concentration := 0.25
  have sodium_chloride_mass := initial_concentration * initial_volume
  have equation := sodium_chloride_mass / (initial_volume + w) = target_concentration
  sorry -- the proof goes here

end add_water_to_achieve_target_concentration_l641_641007


namespace cut_chessboard_into_squares_l641_641898

def chessboard := fin 7 → fin 7 → Prop

theorem cut_chessboard_into_squares (
  cut_along_edges : ∀ (c : chessboard) (p1 p2 : Prop), p1 ∨ p2
) : (∃ (pieces : list (fin 7 → fin 7 → Prop)), pieces.length = 6 ∧ 
    (∃ (p6 p3 p2 : (fin 7 → fin 7 → Prop)),
      (p6 = (λ i j, i < 6 ∧ j < 6)) ∧ 
      (p3 = (λ i j, i < 3 ∧ j < 3)) ∧ 
      (p2 = (λ i j, i < 2 ∧ j < 2)) ∧ 
      ∃ (arrangement : list (fin 7 → fin 7 → Prop) → Prop), 
      (arrangement pieces))) :=
sorry

end cut_chessboard_into_squares_l641_641898


namespace Celia_weeks_is_4_l641_641342

noncomputable def Celia_budget (W : ℕ) : Prop :=
  let S := (100 * W) + 1500 + 30 + 50 in
  0.10 * S = 198

theorem Celia_weeks_is_4 : ∃ W : ℕ, Celia_budget W ∧ W = 4 :=
by
  existsi 4
  repeat { sorry }

end Celia_weeks_is_4_l641_641342


namespace joel_age_when_dad_twice_l641_641082

theorem joel_age_when_dad_twice (
  (joel_age : ℕ) (joels_dad_age : ℕ) : ℕ : 
  joel_age = 5 ∧ joels_dad_age = 32) : ∃ t : ℕ, joels_dad_age + t = 2 * (joel_age + t) ∧ joel_age + t = 27 := by {
  sorry -- Proof omitted
}

end joel_age_when_dad_twice_l641_641082


namespace almond_croissant_cost_l641_641735

theorem almond_croissant_cost (
  cost_white_bread : ℝ := 3.50,
  cost_baguette : ℝ := 1.50,
  cost_sourdough : ℝ := 4.50,
  loaves_white_bread_per_week : ℕ := 2,
  loaves_sourdough_per_week : ℕ := 2,
  weeks : ℕ := 4,
  total_expenditure : ℝ := 78
) : (total_expenditure - (loaves_white_bread_per_week * cost_white_bread + cost_baguette + loaves_sourdough_per_week * cost_sourdough) * weeks) = 8 :=
by
  sorry

end almond_croissant_cost_l641_641735


namespace range_of_f_l641_641970

noncomputable def f (x : ℝ) : ℝ := x^2 + 4 * x - 5

theorem range_of_f : Set.range (f ∘ (fun x => x ∈ Set.Icc (-3 : ℝ) 2)) = Set.Icc (-9 : ℝ) 7 := by
  sorry

end range_of_f_l641_641970


namespace range_of_f_l641_641611

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_f : (set.range (λ x, f x) ∩ set.Icc (-1 : ℝ) (4 : ℝ) = set.Icc 1 10) :=
by 
  -- sorry is used to skip the proof, as per instructions
  sorry

end range_of_f_l641_641611


namespace simplify_expression_l641_641529

variable {a b d : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0)

def x : ℝ := b / d + d / b
def y : ℝ := a / d + d / a
def z : ℝ := a / b + b / a

theorem simplify_expression :
  x^2 + y^2 + z^2 - 2 * x * y * z = -10 :=
by sorry

end simplify_expression_l641_641529


namespace sequence_c_arithmetic_sum_of_sequence_a_l641_641451

-- Part 1: Define c_n and prove it is an arithmetic sequence
theorem sequence_c_arithmetic 
  (a b : ℕ → ℝ) 
  (h_b_ne_zero : ∀ n, b n ≠ 0) 
  (initial_conditions : a 1 = 1 ∧ b 1 = 1)
  (recurrence_relation : ∀ n, b (n + 1) * (a n + 3 * b n) = a (n + 1) * b n) :
  (∀ n : ℕ, c n = a n / b n) ∧ ∀ n : ℕ, c n = 3 * (n + 1) - 2 := 
  sorry

-- Part 2: Given b_n is a geometric sequence and prove the sum S_n
theorem sum_of_sequence_a 
  (a b : ℕ → ℝ) 
  (h_b_ne_zero : ∀ n, b n ≠ 0) 
  (initial_conditions : a 1 = 1 ∧ b 1 = 1)
  (recurrence_relation : ∀ n, b (n + 1) * (a n + 3 * b n) = a (n + 1) * b n)
  (b_geometric : ∃ q : ℝ, (∀ n : ℕ, b (n + 1) = (1/2) * b n) ∧ (q > 0))
  (b_3_condition : (b 3)^2 = 4 * b 2 * b 6) :
  ∀ n, S_n = (finset.range n).sum (λ k, a (k + 1)) = 8 - (6 * n + 8) * ((1/2) ^ n) := 
  sorry

end sequence_c_arithmetic_sum_of_sequence_a_l641_641451


namespace factorize_expression_l641_641753

variable (a b : ℝ)

theorem factorize_expression : (a - b)^2 + 6 * (b - a) + 9 = (a - b - 3)^2 :=
by
  sorry

end factorize_expression_l641_641753


namespace john_needs_to_add_empty_cans_l641_641083

theorem john_needs_to_add_empty_cans :
  ∀ (num_full_cans : ℕ) (weight_per_full_can total_weight weight_per_empty_can required_weight : ℕ),
  num_full_cans = 6 →
  weight_per_full_can = 14 →
  total_weight = 88 →
  weight_per_empty_can = 2 →
  required_weight = total_weight - (num_full_cans * weight_per_full_can) →
  required_weight / weight_per_empty_can = 2 :=
by
  intros
  sorry

end john_needs_to_add_empty_cans_l641_641083


namespace exists_polygon_with_n_triangulations_l641_641935

noncomputable theory

-- Definitions that would be appropriate for a full formal proof
-- would need to be defined, for now, we assume "Polygon" exists, along with relevant properties.

-- Assuming appropriate definitions for no_three_collinear_vertices and triangulations as predicates or functions

def no_three_collinear_vertices (P : Polygon) : Prop := sorry
def triangulations (P : Polygon) : ℕ := sorry

theorem exists_polygon_with_n_triangulations :
  ∀ (n : ℕ), 0 < n → ∃ (P : Polygon), no_three_collinear_vertices P ∧ triangulations P = n := 
by
  intro n hn
  sorry

end exists_polygon_with_n_triangulations_l641_641935


namespace total_distance_traveled_l641_641709

/- Problem Statement:
   Given:
   - Vm: the speed of the man in still water is 6 kmph
   - Vr: the speed of the river is 2 kmph
   - The total time taken to row to a place and back is 1 hour
   Prove:
   - The total distance traveled by the man is 5.34 km
-/

def Vm : ℝ := 6
def Vr : ℝ := 2
def T : ℝ := 1

theorem total_distance_traveled (Vm Vr T : ℝ) (hVm : Vm = 6) (hVr : Vr = 2) (hT : T = 1) :
  let D := 8 / 3 in 2 * D = 5.34 :=
by
  sorry

end total_distance_traveled_l641_641709


namespace greatest_product_sum_2000_eq_1000000_l641_641233

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641233


namespace fibonacci_second_occurrence_last_digit_l641_641348

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

def units_digit (n : ℕ) : ℕ :=
n % 10

theorem fibonacci_second_occurrence_last_digit (n : ℕ) : 
  (∃ k : ℕ, k ≤ n ∧ units_digit (fibonacci k) = 2) →
  (∀ d : ℕ, d < 10 →
      ∃! m : ℕ, m ≤ n ∧ units_digit (fibonacci m) = d) →
  ∃ n1 n2 : ℕ, n1 < n2 ∧ units_digit (fibonacci n2) = 2 ∧ 
    (∀ t : ℕ, t < n2 → (units_digit (fibonacci t) = 2 → t = n1)) → 
  ∃ m : ℕ, m <= n ∧ units_digit (fibonacci m) = 2 :=
sorry

end fibonacci_second_occurrence_last_digit_l641_641348


namespace inequality_abc_l641_641941

theorem inequality_abc (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) :=
by
  -- Proof goes here
  sorry

end inequality_abc_l641_641941


namespace reading_time_difference_l641_641897

theorem reading_time_difference :
  let pages := 360 in
  let julian_rate := 120 in
  let alexa_rate := 80 in
  let julian_time := pages / julian_rate in
  let alexa_time := pages / alexa_rate in
  let time_difference := alexa_time - julian_time in
  (time_difference * 60) = 90 := by
  sorry

end reading_time_difference_l641_641897


namespace Morse_code_distinct_symbols_l641_641053

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l641_641053


namespace sum_of_excluded_x_l641_641106

noncomputable def A : ℝ := 3
noncomputable def B : ℝ := 10 / 9
noncomputable def C : ℝ := 40 / 9

theorem sum_of_excluded_x :
  let f : ℝ → ℝ := λ x, ((x + B) * (A * x + 40)) / ((x + C) * (x + 10))
  (∀ x, f x = 3) → (∃ x1 x2, (x1 = -(C) ∨ x1 = -10) ∧ (x2 = -(C) ∨ x2 = -10) ∧ x1 ≠ x2 ∧ x1 + x2 = -130 / 9) :=
by
  sorry

end sum_of_excluded_x_l641_641106


namespace max_product_l641_641228

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641228


namespace directors_dividends_correct_l641_641649

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l641_641649


namespace base8_product_l641_641257

theorem base8_product (n : ℕ) (h : n = 7254) :
  (let digits := [1, 6, 1, 2, 6] in digits.foldl (λ acc x => acc * x) 1 = 72) :=
by
  sorry

end base8_product_l641_641257


namespace uphill_speed_is_30_l641_641697

-- Definitions according to the conditions
def downhill_speed : ℝ := 60
def uphill_distance : ℝ := 100
def downhill_distance : ℝ := 50
def average_speed : ℝ := 36
def total_distance : ℝ := uphill_distance + downhill_distance

-- Define time taken for uphill and downhill
noncomputable def time_uphill (V_up : ℝ) : ℝ := uphill_distance / V_up
def time_downhill : ℝ := downhill_distance / downhill_speed

-- Define the total time and average speed formula
noncomputable def total_time (V_up : ℝ) : ℝ := time_uphill V_up + time_downhill
noncomputable def avg_speed_formula (V_up : ℝ) : ℝ := total_distance / total_time V_up

-- Statement to prove
theorem uphill_speed_is_30 :
  Exists (λ V_up : ℝ, avg_speed_formula V_up = average_speed ∧ V_up = 30) :=
by
  sorry

end uphill_speed_is_30_l641_641697


namespace find_number_l641_641977

theorem find_number (n : ℕ) : 10 ≤ n ∧ n < 100 ∧ 70 ≤ n ∧ n ≤ 80 ∧ n % 8 = 0 → n = 72 :=
by
  intro h,
  sorry

end find_number_l641_641977


namespace find_cherry_pie_with_at_most_two_bites_l641_641119

theorem find_cherry_pie_with_at_most_two_bites
  {A : Finset ℕ}
  (H : A.card = 7)
  (rice_pies cabbage_pies cherry_pie : ℕ)
  (H_rice : rice_pies = 3)
  (H_cabbage : cabbage_pies = 3)
  (H_cherry : cherry_pie = 1)
  (H_arranged : A = {1, 2, 3, 4, 5, 6, 7})
  (H_rotated : ∃ rotated_A : Finset ℕ, rotated_A = A) :
  ∃ (masha_strategy : ℕ → bool), ∀ a ∈ A, masha_strategy a = tt → a = cherry_pie :=
sorry

end find_cherry_pie_with_at_most_two_bites_l641_641119


namespace a5_value_l641_641414

theorem a5_value (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a2 - a1 = 2)
  (h2 : a3 - a2 = 4)
  (h3 : a4 - a3 = 8)
  (h4 : a5 - a4 = 16) :
  a5 = 31 := by
  sorry

end a5_value_l641_641414


namespace max_f_l641_641384

noncomputable def f (x : ℝ) : ℝ :=
  min (min (2 * x + 2) (1 / 2 * x + 1)) (-3 / 4 * x + 7)

theorem max_f : ∃ x : ℝ, f x = 17 / 5 :=
by
  sorry

end max_f_l641_641384


namespace Morse_code_number_of_distinct_symbols_l641_641035

def count_sequences (n : ℕ) : ℕ :=
  2 ^ n

theorem Morse_code_number_of_distinct_symbols :
  (count_sequences 1) + (count_sequences 2) + (count_sequences 3) + (count_sequences 4) + (count_sequences 5) = 62 :=
by
  simp [count_sequences]
  norm_num
  sorry

end Morse_code_number_of_distinct_symbols_l641_641035


namespace find_starting_number_l641_641363

theorem find_starting_number (n : ℤ) (h1 : ∀ k : ℤ, n ≤ k ∧ k ≤ 38 → k % 4 = 0) (h2 : (n + 38) / 2 = 22) : n = 8 :=
sorry

end find_starting_number_l641_641363


namespace probability_right_oar_works_0_l641_641695

variable (P : Type) [ProbabilitySpace P]

def probability_oar_works_left_and_right_is_0.6 : Prop :=
  let P_R : ℝ := Probability fun ω => right_oar_works ω in
  let P_L : ℝ := Probability fun ω => left_oar_works ω in
  P_R = 0.6

def probability_of_rowing_with_at_least_one_oar : Prop :=
  let P_R : ℝ := Probability fun ω => right_oar_works ω in
  let P_L : ℝ := Probability fun ω => left_oar_works ω in
  Probability fun ω => right_oar_works ω ∨ left_oar_works ω = 0.84

theorem probability_right_oar_works_0.6 
  (h : probability_of_rowing_with_at_least_one_oar P) : probability_oar_works_left_and_right_is_0.6 P := 
sorry

end probability_right_oar_works_0_l641_641695


namespace measure_of_angle_B_scalene_triangle_l641_641182

theorem measure_of_angle_B_scalene_triangle (A B C : ℝ) (hA_gt_0 : A > 0) (hB_gt_0 : B > 0) (hC_gt_0 : C > 0) 
(h_angles_sum : A + B + C = 180) (hB_eq_2A : B = 2 * A) (hC_eq_3A : C = 3 * A) : B = 60 :=
by
  sorry

end measure_of_angle_B_scalene_triangle_l641_641182


namespace least_distance_tetrahedron_l641_641486

noncomputable def P (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (2/3 * A.1 + 1/3 * B.1, 2/3 * A.2 + 1/3 * B.2, 2/3 * A.3 + 1/3 * B.3)
noncomputable def Q (C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (1/3 * C.1 + 2/3 * D.1, 1/3 * C.2 + 2/3 * D.2, 1/3 * C.3 + 2/3 * D.3)

theorem least_distance_tetrahedron 
  (A B C D : ℝ × ℝ × ℝ)
  (h_eq_len1 : dist A B = 1 ∧ dist A C = 1 ∧ dist A D = 1 ∧ dist B C = 1 ∧ dist B D = 1 ∧ dist C D = 1) :
  dist (P A B) (Q C D) = sqrt (10) / 3 :=
by
  sorry

end least_distance_tetrahedron_l641_641486


namespace basketball_holes_l641_641116

theorem basketball_holes (soccer_balls total_basketballs soccer_balls_with_hole balls_without_holes basketballs_without_holes: ℕ) 
  (h1: soccer_balls = 40) 
  (h2: total_basketballs = 15)
  (h3: soccer_balls_with_hole = 30) 
  (h4: balls_without_holes = 18) 
  (h5: basketballs_without_holes = 8) 
  : (total_basketballs - basketballs_without_holes = 7) := 
by
  sorry

end basketball_holes_l641_641116


namespace multiple_of_P_l641_641471

theorem multiple_of_P (P Q R : ℝ) (T : ℝ) (x : ℝ) (total_profit Rs900 : ℝ)
  (h1 : P = 6 * Q)
  (h2 : P = 10 * R)
  (h3 : R = T / 5.1)
  (h4 : total_profit = Rs900 + (T - R)) :
  x = 10 :=
by
  sorry

end multiple_of_P_l641_641471


namespace max_product_of_sum_2000_l641_641242

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641242


namespace max_non_overlapping_regions_l641_641699

theorem max_non_overlapping_regions (n : ℕ) (h : n > 0) :
  let radii := 2 * n in
  let sectors := radii in
  let additional_regions := n + 1 in
  sectors + additional_regions = 3 * n + 1 := by
    sorry

end max_non_overlapping_regions_l641_641699


namespace min_value_of_a_l641_641993

theorem min_value_of_a (r s t : ℕ) (h1 : r > 0) (h2 : s > 0) (h3 : t > 0)
  (h4 : r * s * t = 2310) (h5 : r + s + t = a) : 
  a = 390 → True :=
by { 
  intros, 
  sorry 
}

end min_value_of_a_l641_641993


namespace greatest_product_sum_2000_l641_641213

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641213


namespace junior_score_is_90_l641_641060

theorem junior_score_is_90 {n : ℕ} (hn : n > 0)
    (j : ℕ := n / 5) (s : ℕ := 4 * n / 5)
    (overall_avg : ℝ := 86)
    (senior_avg : ℝ := 85)
    (junior_score : ℝ)
    (h1 : 20 * j = n)
    (h2 : 80 * s = n * 4)
    (h3 : overall_avg * n = 86 * n)
    (h4 : senior_avg * s = 85 * s)
    (h5 : j * junior_score = overall_avg * n - senior_avg * s) :
    junior_score = 90 :=
by
  sorry

end junior_score_is_90_l641_641060


namespace abs_square_implication_l641_641268

theorem abs_square_implication (a b : ℝ) (h : abs a > abs b) : a^2 > b^2 :=
by sorry

end abs_square_implication_l641_641268


namespace number_of_rectangles_l641_641398

theorem number_of_rectangles (a b : ℝ) (ha_lt_b : a < b) :
  ∃! (x y : ℝ), (x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = (a * b) / 4) := 
sorry

end number_of_rectangles_l641_641398


namespace integral_exp_x_plus_2x_l641_641752

theorem integral_exp_x_plus_2x : ∫ x in 0..1, (exp x + 2 * x) = Real.exp 1 := by
  sorry

end integral_exp_x_plus_2x_l641_641752


namespace mason_correct_needed_to_pass_l641_641542

def num_total_questions : ℕ := 80
def num_arithmetic_questions : ℕ := 15
def num_algebra_questions : ℕ := 25
def num_geometry_questions : ℕ := 40

def perc_arithmetic_correct : ℝ := 0.60
def perc_algebra_correct : ℝ := 0.50
def perc_geometry_correct : ℝ := 0.80
def perc_passing_grade : ℝ := 0.70

theorem mason_correct_needed_to_pass :
  let total_correct := perc_arithmetic_correct * num_arithmetic_questions +
                        perc_algebra_correct * num_algebra_questions +
                        perc_geometry_correct * num_geometry_questions
  in total_correct + 2.5 = perc_passing_grade * num_total_questions :=
by
  sorry

end mason_correct_needed_to_pass_l641_641542


namespace change_for_fifty_cents_l641_641460

theorem change_for_fifty_cents :
  ∃ (ways : ℕ), ways = 17 ∧ ∀ (coins : list ℕ), 
    (∀ c ∈ coins, c ∈ [1, 5, 10, 50]) →
    (coins.sum = 50) →
    (count_occurrences coins 50 ≤ 2) →
    ways = 17 :=
by
  sorry

 -- Helper definition to compute the sum of elements in a list
namespace list

def sum (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

end list

-- Additional helper function to count occurrences of an element in a list, if needed
def count_occurrences (l : list ℕ) (x : ℕ) : ℕ :=
  l.foldr (λ y acc, if y = x then acc + 1 else acc) 0

end change_for_fifty_cents_l641_641460


namespace basketball_lineup_ways_l641_641932

theorem basketball_lineup_ways :
  let players := 16
  let twins := 2 -- Betty and Bobbi
  let seniors := 5
  let lineup_size := 7
  (∃ (ways : ℕ), 
    ways = (2 * (binomial seniors 2 * binomial (players - twins - seniors) 4 +
                 binomial seniors 3 * binomial (players - twins - seniors) 3)) ∧
    ways = 4200)
:=
begin
  sorry
end

end basketball_lineup_ways_l641_641932


namespace find_m_of_slope_is_12_l641_641976

theorem find_m_of_slope_is_12 (m : ℝ) :
  let A := (-m, 6)
  let B := (1, 3 * m)
  let slope := (3 * m - 6) / (1 + m)
  slope = 12 → m = -2 :=
by
  sorry

end find_m_of_slope_is_12_l641_641976


namespace sum_of_children_ages_l641_641620

theorem sum_of_children_ages :
  ∃ E: ℕ, E = 12 ∧ 
  (∃ a b c d e : ℕ, a = E ∧ b = E - 2 ∧ c = E - 4 ∧ d = E - 6 ∧ e = E - 8 ∧ 
   a + b + c + d + e = 40) :=
sorry

end sum_of_children_ages_l641_641620


namespace smallest_possible_value_of_a_l641_641982

theorem smallest_possible_value_of_a (a b : ℕ) :
  (∃ (r1 r2 r3 : ℕ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    (r1 * r2 * r3 = 2310) ∧
    (a = r1 + r2 + r3)) →
  a = 52 :=
begin
  sorry
end

end smallest_possible_value_of_a_l641_641982


namespace magnet_cost_is_three_l641_641333

noncomputable def stuffed_animal_cost : ℕ := 6
noncomputable def combined_stuffed_animals_cost : ℕ := 2 * stuffed_animal_cost
noncomputable def magnet_cost : ℕ := combined_stuffed_animals_cost / 4

theorem magnet_cost_is_three : magnet_cost = 3 :=
by
  sorry

end magnet_cost_is_three_l641_641333


namespace midline_intersects_incircle_segment_length_inside_incircle_l641_641955

open Real

def isosceles_triangle (A B C : Point) (BC AB AC : ℝ) : Prop :=
BC = 34 ∧ AB = 49 ∧ AC = 49 ∧ AB = AC

theorem midline_intersects_incircle 
(A B C M N O : Point) (BC AB AC : ℝ) 
(h_isosceles : isosceles_triangle A B C BC AB AC)
(h_midline : RelPointOnLine M B A ∧ RelPointOnLine N C A ∧ LineParallel MN BC ∧ LineMidSegment M A B ∧ LineMidSegment N A C)
(h_incircle : Incircle O A B C)
: LineIntersectsCircle M N O A :=
sorry

theorem segment_length_inside_incircle 
(M N O A B C : Point) (BC AB AC : ℝ) 
(h_isosceles : isosceles_triangle A B C BC AB AC)
(h_midline : RelPointOnLine M B A ∧ RelPointOnLine N C A ∧ LineParallel MN BC ∧ LineMidSegment M A B ∧ LineMidSegment N A C)
(h_incircle : Incircle O A B C)
: segmentLengthInsideCircle M N O = 8 :=
sorry

end midline_intersects_incircle_segment_length_inside_incircle_l641_641955


namespace problem1_problem2_l641_641532
noncomputable theory

-- Definitions for problem (1)
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)
def g (x : ℝ) : ℝ := 4 + abs (x - 3) - abs (x - 1)

-- Theorem for part (1)
theorem problem1 (x : ℝ) : f x 3 >= g x ↔ x ∈ (-∞, 0] ∪ [4, +∞) := 
sorry

-- Definitions for problem (2)
def inequality_solution (x a : ℝ) : Prop := f x a ≤ 1 + abs (x - 3)
def a_value (m n : ℝ) (hm : m > 0) (hn : n > 0) : ℝ := 1 / m + 1 / (2 * n)

-- Theorem for part (2)
theorem problem2 {m n : ℝ} (h : m > 0) (h' : n > 0) : (a_value m n h h' = 2) → (m + 2 * n ≥ 2) := 
sorry

end problem1_problem2_l641_641532


namespace divisibility_qk_prime_numbers_l641_641110

theorem divisibility_qk_prime_numbers
  (a b c k q : ℕ)
  (h1 : a ≥ 1)
  (h2 : b ≥ 1)
  (h3 : c ≥ 1)
  (h4 : k ≥ 1)
  (h5 : q ≥ 1)
  (h6 : ∃ p : ℕ, is_prime p ∧ p ∣ c ∧ p ≠ 1) :
  ∃ (primes : finset ℕ),
    ∀ (p : ℕ), (p ∈ primes → is_prime p ∧ p ∣ (a ^ (c ^ k) - b ^ (c ^ k)))
              → primes.card ≥ q * k :=
by sorry

end divisibility_qk_prime_numbers_l641_641110


namespace pascal_log2_fraction_l641_641093

def pascal_row_sum (n : ℕ) : ℕ := 2 ^ n

def g (n : ℕ) : ℕ := Int.log2 (pascal_row_sum n)

theorem pascal_log2_fraction (n : ℕ) :
  g(n) / Real.log2 3 = n / Real.log2 3 := 
by
  -- Proof omitted
  sorry

end pascal_log2_fraction_l641_641093


namespace zuminglish_word_remainder_l641_641859

/- Definitions corresponding to problem conditions -/
def is_vowel (ch : Char) : Prop := ch = 'O'
def is_consonant (ch : Char) : Prop := ch = 'M' ∨ ch = 'P'

def valid_word (w : List Char) : Prop :=
  ∀ i j, i < j ∧ j < w.length ∧ w.get? i = some 'O' ∧ w.get? j = some 'O' →
    ∃ k, i < k ∧ k < j ∧ is_consonant (w.get k) ∧
           ∃ l, k < l ∧ l < j ∧ is_consonant (w.get l)

def count_valid_words (n : Nat) : Nat := sorry

/- Theorem statement -/
theorem zuminglish_word_remainder (N : Nat) :
  count_valid_words 8 % 1000 = 696 :=
sorry

/- Initial values -/
def a_n : (Nat → Nat) := 
  λ n, if n = 2 then 4 else 2 * (a_n (n-1) + c_n (n-1))
def b_n : (Nat → Nat) := 
  λ n, if n = 2 then 2 else a_n (n-1)
def c_n : (Nat → Nat) := 
  λ n, if n = 2 then 2 else 2 * b_n (n-1)

#eval a_n 9    -- for calculating a_9
#eval b_n 9    -- for calculating b_9
#eval c_n 9    -- for calculating c_9

def a_n_t9 : Nat := a_n 9
def b_n_t9 : Nat := b_n 9
def c_n_t9 : Nat := c_n 9

def N := a_n_t9 + b_n_t9 + c_n_t9

example : N % 1000 = 696 := 
  by exact_mod_cast rfl

end zuminglish_word_remainder_l641_641859


namespace instantaneous_velocity_at_4_l641_641431

variable (s : ℝ → ℝ)
variable (t : ℝ)

noncomputable def velocity (t: ℝ) : ℝ := deriv s t

theorem instantaneous_velocity_at_4
  (s_def : s = λ t, 1 - t + t^2)
  (t_val : t = 4) :
  velocity s t = 7 := by
  sorry

end instantaneous_velocity_at_4_l641_641431


namespace cone_slant_height_l641_641615

theorem cone_slant_height
  (r : ℝ) (CSA : ℝ)
  (h_r : r = 12)
  (h_CSA : CSA = 527.7875658030853) :
  ∃ l : ℝ, abs (l - 14) < 1e-7 :=
by
  let pi := Real.pi
  let l := CSA / (pi * r)
  use l
  have : l = 527.7875658030853 / (pi * 12), by
    rw [h_CSA, h_r]
  have approx_l : abs (l - 14) < 1e-7, from sorry
  exact approx_l

end cone_slant_height_l641_641615


namespace min_lowest_score_and_judge_count_l641_641154

theorem min_lowest_score_and_judge_count
  (n : ℕ)
  (h1 : ∀ k, k < n → (0 ≤ y k ∧ y k ≤ 10))
  (h2 : S = 9.64 * n)
  (h3 : ∃ x, (S - x) / (n - 1) = 9.60)
  (h4 : ∃ y, (S - y) / (n - 1) = 9.68):
  n = 49 ∧ ∃ y, y <= 10 ∧ y >= 0 :=
by
  sorry

end min_lowest_score_and_judge_count_l641_641154


namespace integer_values_of_a_l641_641815

-- Definitions based on the given problem's conditions
def has_integer_roots (a : ℤ) : Prop :=
  ∃ x : ℤ, (a - 1) * x^2 + 2 * x - a - 1 = 0 

-- Main theorem statement to be proven
theorem integer_values_of_a :
  { a : ℤ | has_integer_roots a }.finite.to_finset.card = 5 :=
sorry

end integer_values_of_a_l641_641815


namespace ratio_longest_shortest_side_eq_one_l641_641528

theorem ratio_longest_shortest_side_eq_one
  (ABC : Type) [triangle ABC]
  (A B C : vertices ABC)
  (h_a : Altitude A)
  (beta_b : AngleBisector B)
  (m_c : Median C)
  (h_a_eq : h_a = beta_b)
  (beta_b_eq : beta_b = m_c) :
  ratio_longest_shortest_side ABC = 1 :=
sorry

end ratio_longest_shortest_side_eq_one_l641_641528


namespace volume_of_solid_eq_3_pi_squared_div_16_l641_641340

noncomputable def volume_of_solid : ℝ :=
  π * ∫ x in 0..(π / 2), (sin x ^ 2)^2

theorem volume_of_solid_eq_3_pi_squared_div_16 :
  volume_of_solid = (3 * π ^ 2) / 16 :=
by
  sorry

end volume_of_solid_eq_3_pi_squared_div_16_l641_641340


namespace geometric_seq_general_formula_sum_c_seq_terms_l641_641792

noncomputable def a_seq (n : ℕ) : ℕ := 2 * 3 ^ (n - 1)

noncomputable def S_seq (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (a_seq n - 2) / 2

theorem geometric_seq_general_formula (n : ℕ) (h : n > 0) : 
  a_seq n = 2 * 3 ^ (n - 1) := 
by {
  sorry
}

noncomputable def d_n (n : ℕ) : ℕ :=
  (a_seq (n + 1) - a_seq n) / (n + 1)

noncomputable def c_seq (n : ℕ) : ℕ :=
  d_n n / (n * a_seq n)

noncomputable def T_n (n : ℕ) : ℕ :=
  2 * (1 - 1 / (n + 1)) * n

theorem sum_c_seq_terms (n : ℕ) (h : n > 0) : 
  T_n n = 2 * n / (n + 1) :=
by {
  sorry
}

end geometric_seq_general_formula_sum_c_seq_terms_l641_641792


namespace min_a2_plus_b2_l641_641823

noncomputable def f (x : ℝ) : ℝ := Real.log ((Real.exp 1) * x / (Real.exp 1 - x))

theorem min_a2_plus_b2 : 
  (∑ i in Finset.range 2017 + 1, f ((i + 1 : ℝ) * Real.exp 1 / 2018)) = (2017 / 4) * (a + b) → 
  a + b = 4 → 
  a^2 + b^2 = 8 :=
by
  sorry

end min_a2_plus_b2_l641_641823


namespace quadrilateral_inequality_l641_641907

variable (A B C D : ℝ)
variable (α β γ δ : ℝ) [H1 : has_inscribed_circle (A B C D)]
  (hα: ∀ α, α ≥ 60 ∧ α ≤ 120) (hβ: ∀ β, β ≥ 60 ∧ β ≤ 120)
  (hγ: ∀ γ, γ ≥ 60 ∧ γ ≤ 120) (hδ: ∀ δ, δ ≥ 60 ∧ δ ≤ 120)

theorem quadrilateral_inequality :
  (1/3) * |A^3 - D^3| ≤ |B^3 - D^3| ∧ |B^3 - D^3| ≤ 3 * |A^3 - D^3| := by
  sorry

end quadrilateral_inequality_l641_641907


namespace centroid_midpoint_triangle_eq_centroid_original_triangle_l641_641937

/-
Prove that the centroid of the triangle formed by the midpoints of the sides of another triangle
is the same as the centroid of the original triangle.
-/
theorem centroid_midpoint_triangle_eq_centroid_original_triangle
  (A B C M N P : ℝ × ℝ)
  (hM : M = (A + B) / 2)
  (hN : N = (A + C) / 2)
  (hP : P = (B + C) / 2) :
  (M.1 + N.1 + P.1) / 3 = (A.1 + B.1 + C.1) / 3 ∧
  (M.2 + N.2 + P.2) / 3 = (A.2 + B.2 + C.2) / 3 :=
by
  sorry

end centroid_midpoint_triangle_eq_centroid_original_triangle_l641_641937


namespace problem_l641_641911

variable {f : ℝ → ℝ}

-- Condition: f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Condition: f is monotonically decreasing on (0, +∞)
def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f y < f x

theorem problem (h_even : even_function f) (h_mon_dec : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l641_641911


namespace no_two_heads_consecutive_probability_l641_641299

noncomputable def probability_no_two_heads_consecutive (total_flips : ℕ) : ℚ :=
  if total_flips = 10 then 144 / 1024 else 0

theorem no_two_heads_consecutive_probability :
  probability_no_two_heads_consecutive 10 = 9 / 64 :=
by
  unfold probability_no_two_heads_consecutive
  rw [if_pos rfl]
  norm_num
  sorry

end no_two_heads_consecutive_probability_l641_641299


namespace h_at_3_l641_641349

-- Definitions based on the given conditions:
def h (x : ℝ) : ℝ := ( ((x + 1) * (x^2 + 1) * (x^4 + 1) * ... * (x^(2^2008) + 1)) - 1 ) / (x^(2^2009 - 1) - 1)

-- Hypothesis based on the given conditions
axiom h_def : ∀ x : ℝ, (x^(2^2009 - 1) - 1) * h(x) = (x + 1) * (x^2 + 1) * (x^4 + 1) * ... * (x^(2^2008) + 1) - 1

-- Statement to be proven
theorem h_at_3 : h 3 = 3 := 
by 
sorry

end h_at_3_l641_641349


namespace Morse_code_distinct_symbols_l641_641052

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l641_641052


namespace range_of_m_l641_641162

theorem range_of_m (m : ℝ) :
  (¬ ∃ x > 1, x^2 + (m - 2) * x + 3 - m < 0) ↔ m ∈ set.Ici (-1) := 
sorry

end range_of_m_l641_641162


namespace tan_double_angle_point_l641_641024

variable (α : Real)
variable (P : Real × Real)
variable (hx : P = (1, -4))

theorem tan_double_angle_point :
  (P = (1, -4)) → ∃ α : Real, Float.tan (2 * α) = 8 / 15 :=
by
  intro h
  rw [hx] at h
  sorry

end tan_double_angle_point_l641_641024


namespace polar_curve_equation_l641_641762

theorem polar_curve_equation (ρ θ : ℝ) :
  (ρ * cos θ = 2 * sin (2 * θ)) ↔ 
  (θ ≠ π/2 ∧ θ ≠ 3 * π/2 → ρ = 4 * sin θ) ∧ 
  (θ = π/2 ∨ θ = 3 * π/2 → ρ = ρ) :=
by
  sorry

end polar_curve_equation_l641_641762


namespace age_proof_l641_641173

noncomputable def father_age_current := 33
noncomputable def xiaolin_age_current := 3

def father_age (X : ℕ) := 11 * X
def future_father_age (F : ℕ) := F + 7
def future_xiaolin_age (X : ℕ) := X + 7

theorem age_proof (F X : ℕ) (h1 : F = father_age X) 
  (h2 : future_father_age F = 4 * future_xiaolin_age X) : 
  F = father_age_current ∧ X = xiaolin_age_current :=
by 
  sorry

end age_proof_l641_641173


namespace time_to_shovel_snow_l641_641513

noncomputable def initial_rate : ℕ := 30
noncomputable def decay_rate : ℕ := 2
noncomputable def driveway_width : ℕ := 6
noncomputable def driveway_length : ℕ := 15
noncomputable def snow_depth : ℕ := 2

noncomputable def total_snow_volume : ℕ := driveway_width * driveway_length * snow_depth

def snow_shoveling_time (initial_rate decay_rate total_volume : ℕ) : ℕ :=
-- Function to compute the time needed, assuming definition provided
sorry

theorem time_to_shovel_snow 
  : snow_shoveling_time initial_rate decay_rate total_snow_volume = 8 :=
sorry

end time_to_shovel_snow_l641_641513


namespace brady_passing_yards_proof_l641_641625

def tom_brady_current_passing_yards 
  (record_yards : ℕ) (games_left : ℕ) (average_yards_needed : ℕ) 
  (total_yards_needed_to_break_record : ℕ :=
    record_yards + 1) : ℕ :=
  total_yards_needed_to_break_record - games_left * average_yards_needed

theorem brady_passing_yards_proof :
  tom_brady_current_passing_yards 5999 6 300 = 4200 :=
by 
  sorry

end brady_passing_yards_proof_l641_641625


namespace trig_identity_example_l641_641336

theorem trig_identity_example :
  (2 * (Real.sin (Real.pi / 6)) - Real.tan (Real.pi / 4)) = 0 :=
by
  -- Definitions from conditions
  have h1 : Real.sin (Real.pi / 6) = 1/2 := Real.sin_pi_div_six
  have h2 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  rw [h1, h2]
  sorry -- The proof is omitted as per instructions

end trig_identity_example_l641_641336


namespace fold_symmetry_coordinates_l641_641773

theorem fold_symmetry_coordinates (m n : ℝ) :
  ∃ l : ℝ → ℝ, 
    l 2 = 1 ∧ 
    ∀ p q : ℝ × ℝ,
      (p = (0, 2) ∨ p = (4, 0) ∨ p = (9, 5) ∨ p = (m, n)) → 
      (q = (4, 0) ∨ q = (0, 2) ∨ q = (m, n) ∨ q = (9, 5)) →
      (p ≠ q) → 
      (l (fst p) = snd p) → 
      (l (fst q) = snd q) →
        ((fst p - fst q)^2 + (snd p - snd q)^2 = (fst (pointwise_add (symm_axis (axis l)) p q) - 0)^2 + (snd (pointwise_add (symm_axis (axis l)) p q) - 0)^2) →
        m + n = 10 := by
sorry

end fold_symmetry_coordinates_l641_641773


namespace max_product_two_integers_sum_2000_l641_641205

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641205


namespace max_quadrilateral_area_l641_641491

namespace Geometry

noncomputable def circleC1 : Set (ℝ × ℝ) := { p | (p.1 - 1)^2 + p.2^2 = 9 }
noncomputable def circleC2 : Set (ℝ × ℝ) := { p | (p.1 + 1)^2 + p.2^2 = 1 }

def isTangent (C : Set (ℝ × ℝ)) (C1 : Set (ℝ × ℝ)) := ∃ r : ℝ, ∀ p ∈ C, ∃ q ∈ C1, ∥p - q∥ = r
def isExternallyTangent (C : Set (ℝ × ℝ)) (C2 : Set (ℝ × ℝ)) := ∃ r : ℝ, ∀ p ∈ C, ∃ q ∈ C2, ∥p - q∥ = r

def pointP : ℝ × ℝ := (-2, 0)
def pointQ : ℝ × ℝ := (2, 0)
def passingLine (m : ℝ) : Set (ℝ × ℝ) := { p | p.1 = m * p.2 + 1 }

theorem max_quadrilateral_area :
  let E := { p : ℝ × ℝ | (p.1)^2 / 4 + (p.2)^2 / 3 = 1 ∧ p ≠ (-2, 0) } in
  (∀ C : Set (ℝ × ℝ), isTangent C circleC1 ∧ isExternallyTangent C circleC2 → 
   E = { p : ℝ × ℝ | (p.1)^2 / 4 + (p.2)^2 / 3 = 1 ∧ p ≠ (-2, 0) }) ∧
  (∃ m : ℝ, 
    let l := passingLine m in
    let intersection := { p : ℝ × ℝ | p ∈ E ∧ p ∈ l } in
    ∃ A B : ℝ × ℝ, A ∈ intersection ∧ B ∈ intersection ∧
    let area_APBQ := λ A B : ℝ × ℝ, (2 * ∥A - B∥ * 1) / 2 in
    (A ≠ B ∧ area_APBQ A B ≤ 6 ∧ ∀ C D : ℝ × ℝ, C ∈ intersection ∧ D ∈ intersection ∧ C ≠ D → area_APBQ C D ≤ 6)
  ) :=
by simp [isTangent, isExternallyTangent, pointP, pointQ, passingLine]
   sorry

end Geometry

end max_quadrilateral_area_l641_641491


namespace number_of_chickens_and_ducks_l641_641673

/-- Xiao Wang's family raises 239 chickens and ducks in total, where
the number of ducks is 15 more than three times the number of chickens.
We need to prove the number of chickens and ducks. --/
theorem number_of_chickens_and_ducks :
  ∃ (c d : ℕ), c + d = 239 ∧ d = 3 * c + 15 ∧ c = 56 ∧ d = 183 :=
begin
  -- The proof goes here
  sorry
end

end number_of_chickens_and_ducks_l641_641673


namespace gretchen_fewest_trips_l641_641455

def fewestTrips (total_objects : ℕ) (max_carry : ℕ) : ℕ :=
  (total_objects + max_carry - 1) / max_carry

theorem gretchen_fewest_trips : fewestTrips 17 3 = 6 := 
  sorry

end gretchen_fewest_trips_l641_641455


namespace max_log_sum_value_l641_641801

open Real

noncomputable def max_log_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 2 * x + y = 20) : ℝ :=
  log x + log y

theorem max_log_sum_value : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 20 ∧ max_log_sum x y (by linarith) (by linarith) (by linarith) = 2 - log 2 :=
exists.intro 5 (exists.intro 10 (and.intro (by linarith) (and.intro (by linarith) (and.intro (by linarith) (by linarith)))))

end max_log_sum_value_l641_641801


namespace cos_neg_x_plus_3pi_over_2_l641_641412

theorem cos_neg_x_plus_3pi_over_2 (x : ℝ) (h₁ : tan x = -12 / 5) (h₂ : π / 2 < x ∧ x < π) :
  cos (-x + 3 * π / 2) = -12 / 13 := 
sorry

end cos_neg_x_plus_3pi_over_2_l641_641412


namespace lori_earnings_l641_641535

theorem lori_earnings
    (red_cars : ℕ)
    (white_cars : ℕ)
    (cost_red_car : ℕ)
    (cost_white_car : ℕ)
    (rental_time_hours : ℕ)
    (rental_time_minutes : ℕ)
    (correct_earnings : ℕ) :
    red_cars = 3 →
    white_cars = 2 →
    cost_red_car = 3 →
    cost_white_car = 2 →
    rental_time_hours = 3 →
    rental_time_minutes = rental_time_hours * 60 →
    correct_earnings = 2340 →
    (red_cars * cost_red_car + white_cars * cost_white_car) * rental_time_minutes = correct_earnings :=
by
  intros
  sorry

end lori_earnings_l641_641535


namespace incorrect_statement_in_biology_l641_641153

theorem incorrect_statement_in_biology :
  (forall (s : Statement), (s ∈ {Statement.A, Statement.B, Statement.C, Statement.D}) → (s ↔ exception_mayr_law_in_biology)) →
  Statement.incorrect_in_biology = Statement.A :=
sorry

end incorrect_statement_in_biology_l641_641153


namespace expected_winning_value_l641_641121

-- Definitions of prime and composite numbers
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop :=
  n = 4 ∨ n = 6 ∨ n = 8

def winnings (n : ℕ) : ℝ :=
  if is_prime n then n 
  else if n = 8 then -3
  else 0

noncomputable def expected_value : ℝ :=
  (1/8) * (winnings 2) + (1/8) * (winnings 3) + (1/8) * (winnings 4) +
  (1/8) * (winnings 5) + (1/8) * (winnings 6) + (1/8) * (winnings 7) +
  (1/8) * (winnings 8) + (1/8) * (winnings (nat.choose 4 [1, 2, 3, 4, 5, 6, 7, 8].head!))

theorem expected_winning_value : expected_value = 7 / 4 :=
  sorry

end expected_winning_value_l641_641121


namespace base15_mod_9_l641_641265

noncomputable def base15_to_decimal : ℕ :=
  2 * 15^3 + 6 * 15^2 + 4 * 15^1 + 3 * 15^0

theorem base15_mod_9 (n : ℕ) (h : n = base15_to_decimal) : n % 9 = 0 :=
sorry

end base15_mod_9_l641_641265


namespace probability_prime_multiple_of_5_l641_641561

noncomputable def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def is_multiple_of_5 (n : Nat) : Prop :=
  n % 5 = 0

theorem probability_prime_multiple_of_5 :
  let favorable_outcomes := { n : Nat | 1 ≤ n ∧ n ≤ 100 ∧ is_prime n ∧ is_multiple_of_5 n }.card
  let possible_outcomes := 100
  (favorable_outcomes : ℚ) / possible_outcomes = 1 / 100 :=
by
  sorry

end probability_prime_multiple_of_5_l641_641561


namespace lcm_of_12_and_15_l641_641146
-- Import the entire Mathlib library

-- Define the given conditions
def HCF (a b : ℕ) : ℕ := gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / (gcd a b)

-- Given the values
def a := 12
def b := 15
def hcf := 3

-- State the proof problem
theorem lcm_of_12_and_15 : LCM a b = 60 :=
by
  -- Proof goes here (skipped)
  sorry

end lcm_of_12_and_15_l641_641146


namespace proof_problem_l641_641913

variables {n : ℕ} (x : Fin n → ℝ)
noncomputable def sum_n (f : Fin n → ℝ) : ℝ := ∑ i, f i

def valid_inputs (n : ℕ) (x : Fin n → ℝ) : Prop :=
  1 < n ∧ (∀ i, 0 < x i) ∧ sum_n x = 1

theorem proof_problem {n : ℕ} (x : Fin n → ℝ) (h : valid_inputs n x) :
  ∑ i, 1 / (x i - (x i)^3) ≥ n^4 / (n^2 - 1) :=
sorry

end proof_problem_l641_641913


namespace basketball_free_throws_l641_641284

theorem basketball_free_throws (total_players : ℕ) (number_captains : ℕ) (players_not_including_one : ℕ) 
  (free_throws_per_captain : ℕ) (total_free_throws : ℕ) 
  (h1 : total_players = 15)
  (h2 : number_captains = 2)
  (h3 : players_not_including_one = total_players - 1)
  (h4 : free_throws_per_captain = players_not_including_one * number_captains)
  (h5 : total_free_throws = free_throws_per_captain)
  : total_free_throws = 28 :=
by
  -- Proof is not required, so we provide sorry to skip it.
  sorry

end basketball_free_throws_l641_641284


namespace area_ratio_of_inscribed_quadrilateral_l641_641945

theorem area_ratio_of_inscribed_quadrilateral (r : ℝ) (h : r > 0) :
  let A_PQRS := r^2 * Real.sqrt 3,
      A_circle := Real.pi * r^2,
      ratio := A_PQRS / A_circle in
  ratio = Real.sqrt 3 / Real.pi →
  let a := (0 : ℝ),
      b := (3 : ℝ),
      c := (1 : ℝ) in
  a + b + c = 4 :=
by
  sorry

end area_ratio_of_inscribed_quadrilateral_l641_641945


namespace positive_difference_of_solutions_l641_641638

theorem positive_difference_of_solutions :
  ∀ (x : ℝ), (|2 * x - 3| = 15) → 
  let x1 := (3 + 15) / 2,
      x2 := (3 - 15) / 2 in
  |x1 - x2| = 15 :=
begin
  sorry
end

end positive_difference_of_solutions_l641_641638


namespace smallest_value_of_a_l641_641987

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end smallest_value_of_a_l641_641987


namespace OQ_parallel_to_KI_l641_641888

noncomputable def Triangle (K I A O N H Q : Type) :=
  KI < KA ∧
  AngleBisector K - K.intersect IA = O ∧
  Midpoint N IA ∧
  PerpendicularFoot H I KO ∧
  H.intersect KN = Q →
  Parallel OQ KI

-- Definitions of AngleBisector, Midpoint, PerpendicularFoot, and Parallel should be defined accordingly
def AngleBisector (K : Type) := sorry
def Midpoint (N : Type) (IA : Type) := sorry
def PerpendicularFoot (H : Type) (I : Type) (KO : Type) := sorry
def Parallel (OQ : Type) (KI : Type) := sorry

theorem OQ_parallel_to_KI (K I A O N H Q : Type) [Triangle K I A O N H Q] : 
  Parallel OQ KI := 
sorry

end OQ_parallel_to_KI_l641_641888


namespace max_product_two_integers_sum_2000_l641_641206

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641206


namespace find_side_length_a_l641_641799

noncomputable def length_of_a (A B : ℝ) (b : ℝ) : ℝ :=
  b * Real.sin A / Real.sin B

theorem find_side_length_a :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = Real.pi / 3 → B = Real.pi / 4 → b = Real.sqrt 6 →
  a = length_of_a A B b →
  a = 3 :=
by
  intros a b c A B C hA hB hb ha
  rw [hA, hB, hb] at ha
  sorry

end find_side_length_a_l641_641799


namespace directors_dividends_correct_l641_641647

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l641_641647


namespace modulus_eq_two_l641_641112

noncomputable def modulus_of_z (z : ℂ) (h : z * (2 - 3 * complex.I) = 6 + 4 * complex.I) : ℝ :=
  complex.abs z

theorem modulus_eq_two (z : ℂ) (h : z * (2 - 3 * complex.I) = 6 + 4 * complex.I) : complex.abs z = 2 :=
sorry

end modulus_eq_two_l641_641112


namespace count_expressible_integers_l641_641375

theorem count_expressible_integers :
  ∃ (count : ℕ), count = 1138 ∧ (∀ n, (n ≤ 2000) → (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n)) :=
sorry

end count_expressible_integers_l641_641375


namespace find_triangle_height_l641_641582

-- Given conditions
def triangle_area : ℝ := 960
def base : ℝ := 48

-- The problem is to find the height such that 960 = (1/2) * 48 * height
theorem find_triangle_height (height : ℝ) 
  (h_area : triangle_area = (1/2) * base * height) : height = 40 := by
  sorry

end find_triangle_height_l641_641582


namespace manufacturing_cost_l641_641156

variable (M : ℝ)

def transportation_cost_per_shoe : ℝ := 500 / 100
def selling_price : ℝ := 270
def gain_factor : ℝ := 1.20

theorem manufacturing_cost :
  M + transportation_cost_per_shoe = (selling_price / gain_factor) → M = 220 :=
by
  sorry

end manufacturing_cost_l641_641156


namespace find_divisor_l641_641122

theorem find_divisor : ∃ D : ℕ, 14698 = (D * 89) + 14 ∧ D = 165 :=
by
  use 165
  sorry

end find_divisor_l641_641122


namespace game_infinite_for_min_k_game_finite_for_max_k_l641_641788

theorem game_infinite_for_min_k (n : ℕ) (hn : n ≥ 2) : 
  ∃ k, k = 3 * n^2 - 4 * n + 1 := 
begin
  sorry
end

theorem game_finite_for_max_k (n : ℕ) (hn : n ≥ 2) : 
  ∃ k, k = 2 * n^2 - 2 * n - 1 := 
begin
  sorry
end

end game_infinite_for_min_k_game_finite_for_max_k_l641_641788


namespace angle_bisector_between_median_and_altitude_l641_641565

variables {A B C H D M : Type}

-- Define points A, B, C forming a triangle
variables (triangle_ABC : Triangle A B C)

-- Define the foot of the altitude from B to AC
def altitude_foot (triangle_ABC : Triangle A B C) : Point := H

-- Define the foot of the angle bisector from B to AC
def angle_bisector_foot (triangle_ABC : Triangle A B C) : Point := D

-- Define the midpoint of AC
def midpoint (A C : Point) : Point := M

-- Prove that D lies between H and M
theorem angle_bisector_between_median_and_altitude 
(triangle_ABC : Triangle A B C)
(H_altitude : altitude_foot triangle_ABC = H)
(D_bisector : angle_bisector_foot triangle_ABC = D)
(M_mid : midpoint A C = M) :
D lies_between H and M :=
sorry

end angle_bisector_between_median_and_altitude_l641_641565


namespace greatest_product_sum_2000_l641_641216

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641216


namespace sum_and_subtract_decimal_l641_641731

theorem sum_and_subtract_decimal : 
  let a := 0.804 
  let b := 0.007 
  let c := 0.0008 
  let d := 0.00009 in
  (a + b + c - d = 0.81171) := 
by 
  let a := 0.804 
  let b := 0.007 
  let c := 0.0008 
  let d := 0.00009
  sorry

end sum_and_subtract_decimal_l641_641731


namespace toothpicks_needed_l641_641009

-- Defining the number of rows in the large equilateral triangle.
def rows : ℕ := 10

-- Formula to compute the total number of smaller equilateral triangles.
def total_small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

-- Number of small triangles in this specific case.
def num_small_triangles : ℕ := total_small_triangles rows

-- Total toothpicks without sharing sides.
def total_sides_no_sharing (n : ℕ) : ℕ := 3 * num_small_triangles

-- Adjust for shared toothpicks internally.
def shared_toothpicks (n : ℕ) : ℕ := (total_sides_no_sharing n - 3 * rows) / 2 + 3 * rows

-- Total boundary toothpicks.
def boundary_toothpicks (n : ℕ) : ℕ := 3 * rows

-- Final total number of toothpicks required.
def total_toothpicks (n : ℕ) : ℕ := shared_toothpicks n + boundary_toothpicks n

-- The theorem to be proved
theorem toothpicks_needed : total_toothpicks rows = 98 :=
by
  -- You can complete the proof.
  sorry

end toothpicks_needed_l641_641009


namespace jack_salt_amount_l641_641892

noncomputable def amount_of_salt (volume_salt_1 : ℝ) (volume_salt_2 : ℝ) : ℝ :=
  volume_salt_1 + volume_salt_2

noncomputable def total_salt_ml (total_salt_l : ℝ) : ℝ :=
  total_salt_l * 1000

theorem jack_salt_amount :
  let day1_water_l := 4.0
  let day2_water_l := 4.0
  let day1_salt_percentage := 0.18
  let day2_salt_percentage := 0.22
  let total_salt_before_evaporation := amount_of_salt (day1_water_l * day1_salt_percentage) (day2_water_l * day2_salt_percentage)
  let final_salt_ml := total_salt_ml total_salt_before_evaporation
  final_salt_ml = 1600 :=
by
  sorry

end jack_salt_amount_l641_641892


namespace retail_price_is_120_l641_641675

variable R : ℝ
variable wholesale_price : ℝ := 90
variable profit_percentage : ℝ := 0.20
variable discount_percentage : ℝ := 0.10

theorem retail_price_is_120 :
  R * (1 - discount_percentage) = wholesale_price * (1 + profit_percentage) → R = 120 :=
by
  intros h
  sorry

end retail_price_is_120_l641_641675


namespace area_of_ABCD_is_integer_l641_641073

-- Geometric conditions
variables {O A B C D : Point}
variables (BC_is_tangent : Tangent BC Circle(O))
variables (AB_perp_BC : Perpendicular A B C)
variables (BC_perp_CD : Perpendicular B C D)
variables (AD_is_diameter : Diameter A D Circle(O))

-- Length conditions
variables (AB_len : Length A B = 9)
variables (CD_len : Length C D = 4)

-- Formal statement in Lean
theorem area_of_ABCD_is_integer :
  ∃ (area : ℕ), (area = compute_area of_trapezoid ABCD) :=
by { sorry }

end area_of_ABCD_is_integer_l641_641073


namespace triangle_CE_10_l641_641076

theorem triangle_CE_10 
  (A B C D E : Type) 
  (angle : A → B → C → Real)
  (on_AC : A → C → D → Prop)
  (on_AC_between : A → C → E → Prop) 
  (BD_bisects_EBC : ∀ α β : Real, α = β → β = α)
  (AC_eq : Real)
  (BC_eq : Real)
  (BE_eq : Real)
  (angle_is_obtuse : angle B A C > 90)
  (angle_ABD : angle A B D = 90):
  AC_eq = 35 → BC_eq = 7 → BE_eq = 5 → ∃ (x : Real), x = 10 := 
by
  sorry

end triangle_CE_10_l641_641076


namespace justify_misha_decision_l641_641652

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l641_641652


namespace id_tags_divided_by_10_l641_641311

def uniqueIDTags (chars : List Char) (counts : Char → Nat) : Nat :=
  let permsWithoutRepetition := 
    Nat.factorial 7 / Nat.factorial (7 - 5)
  let repeatedCharTagCount := 10 * 10 * 6
  permsWithoutRepetition + repeatedCharTagCount

theorem id_tags_divided_by_10 :
  uniqueIDTags ['M', 'A', 'T', 'H', '2', '0', '3'] (fun c =>
    if c = 'M' then 1 else
    if c = 'A' then 1 else
    if c = 'T' then 1 else
    if c = 'H' then 1 else
    if c = '2' then 2 else
    if c = '0' then 1 else
    if c = '3' then 1 else 0) / 10 = 312 :=
by
  sorry

end id_tags_divided_by_10_l641_641311


namespace smallest_number_of_students_l641_641871

-- Define the total number of students in the assembly.
def total_students (x : ℕ) : ℕ := 5 * x + 2

-- Prove that the smallest number of students in the assembly is 52.
theorem smallest_number_of_students :
  ∃ x : ℕ, (total_students x > 50) ∧ (total_students x = 52) :=
by
  existsi 10
  simp [total_students]
  disclaimer_eq sorry

end smallest_number_of_students_l641_641871


namespace range_of_a_l641_641023

variable {x a : ℝ}

theorem range_of_a (h1 : 2 * x - a < 0)
                   (h2 : 1 - 2 * x ≥ 7)
                   (h3 : ∀ x, x ≤ -3) : ∀ a, a > -6 :=
by
  sorry

end range_of_a_l641_641023


namespace degrees_for_lemon_pie_in_pie_chart_l641_641484

theorem degrees_for_lemon_pie_in_pie_chart :
  ∀ (total_students chocolate_pref apple_pref blueberry_pref : ℕ)
  (remaining_students cherry_lemon_students : ℚ),
  total_students = 42 →
  chocolate_pref = 15 →
  apple_pref = 9 →
  blueberry_pref = 7 →
  remaining_students = total_students - (chocolate_pref + apple_pref + blueberry_pref) →
  cherry_lemon_students = remaining_students / 2 →
  let lemon_pie_fraction := (cherry_lemon_students : ℚ) / total_students in
  let lemon_pie_degrees := lemon_pie_fraction * 360 in
  lemon_pie_degrees ≈ 47 :=
by
  intros
  sorry

end degrees_for_lemon_pie_in_pie_chart_l641_641484


namespace percent_decrease_is_20_percent_l641_641512

-- Conditions
def last_week_soup_price := 7.50 / 3
def last_week_bread_price := 5 / 2
def this_week_soup_price := 8 / 4
def this_week_bread_price := 6 / 3

-- Prices
def average_last_week_price := (last_week_soup_price + last_week_bread_price) / 2
def average_this_week_price := (this_week_soup_price + this_week_bread_price) / 2

-- Question: Calculate the percent decrease in price per item on average for a bundle
def percent_decrease := ((average_last_week_price - average_this_week_price) / average_last_week_price) * 100

theorem percent_decrease_is_20_percent : percent_decrease = 20 := by
  sorry

end percent_decrease_is_20_percent_l641_641512


namespace john_can_drive_200_miles_on_25_dollars_l641_641509

def fuel_efficiency (miles_per_gallon : ℕ) := 40
def gas_cost_per_gallon (dollars_per_gallon : ℕ) := 5
def money_spent_on_gas (dollars : ℕ) := 25

theorem john_can_drive_200_miles_on_25_dollars :
  let gallons := (money_spent_on_gas 25) / (gas_cost_per_gallon 5) in
  let total_miles := gallons * (fuel_efficiency 40) in
  total_miles = 200 := by
  sorry

end john_can_drive_200_miles_on_25_dollars_l641_641509


namespace sum_roots_l641_641768

theorem sum_roots :
  (∀ (x : ℂ), (3 * x^3 - 2 * x^2 + 4 * x - 15 = 0) → 
              x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (∀ (x : ℂ), (4 * x^3 - 16 * x^2 - 28 * x + 35 = 0) → 
              x = y₁ ∨ x = y₂ ∨ x = y₃) →
  (x₁ + x₂ + x₃ + y₁ + y₂ + y₃ = 14 / 3) :=
by
  sorry

end sum_roots_l641_641768


namespace abs_neg_2022_l641_641579

-- Definition of absolute value
def abs_value (a : ℝ) : ℝ :=
  if a ≥ 0 then a else -a

-- Problem statement
theorem abs_neg_2022 : abs_value (-2022) = 2022 := by
  sorry

end abs_neg_2022_l641_641579


namespace trapezoid_area_l641_641687

-- Define rectangle PQRS and its area
variables (PQRS : Type) [rect : rectangle PQRS]
variable (has_area : area PQRS = 20)

-- Define trapezoid TURS
variable TURS : Type

-- The statement regarding the area of TURS
theorem trapezoid_area (h : TURS = trapezoid_rect PQRS) : area TURS = 8 :=
sorry

end trapezoid_area_l641_641687


namespace calculate_price_l641_641085

-- Define variables for prices
def sugar_price_in_terms_of_salt (T : ℝ) : ℝ := 2 * T
def rice_price_in_terms_of_salt (T : ℝ) : ℝ := 3 * T
def apple_price : ℝ := 1.50
def pepper_price : ℝ := 1.25

-- Define pricing conditions
def condition_1 (T : ℝ) : Prop :=
  5 * (sugar_price_in_terms_of_salt T) + 3 * T + 2 * (rice_price_in_terms_of_salt T) + 3 * apple_price + 4 * pepper_price = 35

def condition_2 (T : ℝ) : Prop :=
  4 * (sugar_price_in_terms_of_salt T) + 2 * T + 1 * (rice_price_in_terms_of_salt T) + 2 * apple_price + 3 * pepper_price = 24

-- Define final price calculation with discounts
def total_price (T : ℝ) : ℝ :=
  8 * (sugar_price_in_terms_of_salt T) * 0.9 +
  5 * T +
  (rice_price_in_terms_of_salt T + 3 * (rice_price_in_terms_of_salt T - 0.5)) +
  -- adding two free apples to the count
  5 * apple_price +
  6 * pepper_price

-- Main theorem to prove
theorem calculate_price (T : ℝ) (h1 : condition_1 T) (h2 : condition_2 T) :
  total_price T = 55.64 :=
sorry -- proof omitted

end calculate_price_l641_641085


namespace cos_theta_eq_neg_sqrt2_div_4_l641_641833

   variable {a b : EuclideanSpace ℝ (Fin 2)}
   variable (ha : ∥a∥ = Real.sqrt 2) (hb : ∥b∥ = 1) (hab : ∥a - b∥ = 2)

   theorem cos_theta_eq_neg_sqrt2_div_4 : 
     (inner a b) / (∥a∥ * ∥b∥) = -Real.sqrt 2 / 4 := by 
     sorry
   
end cos_theta_eq_neg_sqrt2_div_4_l641_641833


namespace max_product_of_two_integers_sum_2000_l641_641249

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641249


namespace diagonal_difference_zero_l641_641737

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 2, 3], ![4, 5, 6], ![7, 8, 9]]

def swapped_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  initial_matrix.swap [⟨0, by simp⟩, ⟨2, by simp⟩]

def main_diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2

def secondary_diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 2 + m 1 1 + m 2 0

theorem diagonal_difference_zero :
    abs (main_diagonal_sum swapped_matrix - secondary_diagonal_sum swapped_matrix) = 0 :=
by
  sorry

end diagonal_difference_zero_l641_641737


namespace find_n_value_l641_641423

-- Definitions
def general_term (n r : ℕ) : ℤ := binomial n r * x^(n - 4 * r)

noncomputable def no_constant_term_expansion (n : ℕ) : Prop :=
  ∀ r : ℕ, ¬ general_term n r = 0

-- The main theorem
theorem find_n_value (n : ℕ) (hn : 2 ≤ n ∧ n ≤ 8 ∧ n ∈ ℕ* ) :
  (no_constant_term_expansion n) ↔ n = 5 := by
  sorry

end find_n_value_l641_641423


namespace cartesian_equations_distance_range_l641_641490

-- Define the given conditions for line l
def parametric_line_l (t : ℝ) : ℝ × ℝ := (t + 1, t + 4)

-- Define the given conditions for curve C
def polar_curve_C (theta : ℝ) : ℝ :=
  sqrt 3 / sqrt (1 + 2*(cos theta)^2)

-- Verification of Cartesian equations 
theorem cartesian_equations :
  ∀ (t : ℝ) (theta : ℝ),
    (parametric_line_l t).fst - (parametric_line_l t).snd + 3 = 0 ∧
    (polar_curve_C theta)^2 + 2 * (polar_curve_C theta)^2 * (cos theta)^2 = 3 →
    ∀ (x y : ℝ),
      3*x^2 + y^2 = 3 ↔ x^2 + y^2 / 3 = 1 :=
by sorry

-- Verification of the range of the distance d
theorem distance_range :
  ∀ (alpha : ℝ),
    let x := cos alpha,
    y := sqrt 3 * sin alpha,
    line_eq := x - y + 3 = 0,
    distance := abs (2 * cos (alpha + π / 3) + 3) / sqrt 2
    →
    distance ∈ Icc (sqrt 2 / 2) (5 * sqrt 2 / 2) :=
by sorry

end cartesian_equations_distance_range_l641_641490


namespace fewest_trips_l641_641452

theorem fewest_trips (total_objects : ℕ) (capacity : ℕ) (h_objects : total_objects = 17) (h_capacity : capacity = 3) : 
  (total_objects + capacity - 1) / capacity = 6 :=
by
  sorry

end fewest_trips_l641_641452


namespace last_three_digits_of_7_pow_103_l641_641365

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l641_641365


namespace distinct_elements_in_powers_of_i_l641_641158

theorem distinct_elements_in_powers_of_i : 
  ∀ (i : ℂ) (nat_star : ℕ → Prop), (i^2 = -1) → (∀ n, nat_star n ↔ 0 < n)
  → set.finite ({x | ∃ n, nat_star n ∧ x = i^n} : set ℂ)
  → fintype.card ({x | ∃ n, nat_star n ∧ x = i^n} : set ℂ) = 4 :=
by
  intros i nat_star hi hnat_star hfinite
  sorry

end distinct_elements_in_powers_of_i_l641_641158


namespace inscribed_circle_diameter_l641_641636

noncomputable def diameter_inscribed_circle (PQ PR QR : ℝ) : ℝ :=
  let s := (PQ + PR + QR) / 2 in
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR)) in
  let radius := area / s in
  2 * radius

theorem inscribed_circle_diameter :
  (diameter_inscribed_circle 13 8 15) = (10 * Real.sqrt 3 / 3) :=
by
  sorry

end inscribed_circle_diameter_l641_641636


namespace find_a1_l641_641399

theorem find_a1 (a : ℕ → ℕ) (h1 : a 5 = 14) (h2 : ∀ n, a (n+1) - a n = n + 1) : a 1 = 0 :=
by
  sorry

end find_a1_l641_641399


namespace volume_of_intersection_rotated_cube_l641_641151

variables (a α : ℝ)

theorem volume_of_intersection_rotated_cube (h1 : a > 0) (h2 : 0 < α ∧ α < 2 * π) :
  let cot_half_alpha := Real.cot (α / 2) in
  let numerator := 3 * a^3 * (1 + cot_half_alpha^2) in
  let denominator := (1 + Real.sqrt 3 * cot_half_alpha)^2 in
  (a^3 - 6 * (1 / 6) * a * (x * (a - x)) = numerator / denominator) := sorry

end volume_of_intersection_rotated_cube_l641_641151


namespace total_apples_l641_641541

-- Definitions based on the problem conditions
def marin_apples : ℕ := 8
def david_apples : ℕ := (3 * marin_apples) / 4
def amanda_apples : ℕ := (3 * david_apples) / 2 + 2

-- The statement that we need to prove
theorem total_apples : marin_apples + david_apples + amanda_apples = 25 := by
  -- The proof steps will go here
  sorry

end total_apples_l641_641541


namespace age_of_B_l641_641479

variables (A B : ℕ)

-- Conditions
def condition1 := A + 10 = 2 * (B - 10)
def condition2 := A = B + 7

-- Theorem stating the present age of B
theorem age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 :=
by
  sorry

end age_of_B_l641_641479


namespace remainder_3_pow_17_mod_19_is_13_l641_641258

noncomputable def remainder_3_pow_17_mod_19 : ℕ :=
  let a := 3 ^ 17 in
  let b := a % 19 in
  b

theorem remainder_3_pow_17_mod_19_is_13 : remainder_3_pow_17_mod_19 = 13 :=
  sorry

end remainder_3_pow_17_mod_19_is_13_l641_641258


namespace jackson_fishes_per_day_l641_641481

def total_fishes : ℕ := 90
def jonah_per_day : ℕ := 4
def george_per_day : ℕ := 8
def competition_days : ℕ := 5

def jackson_per_day (J : ℕ) : Prop :=
  (total_fishes - (jonah_per_day * competition_days + george_per_day * competition_days)) / competition_days = J

theorem jackson_fishes_per_day : jackson_per_day 6 :=
  by
    sorry

end jackson_fishes_per_day_l641_641481


namespace escalator_time_l641_641353

theorem escalator_time
    (x : ℝ) -- Clea's walking speed in units per second
    (y : ℝ) -- distance of the escalator in units
    (k : ℝ) -- speed of the escalator
    (h1 : 80 * x = y)
    (h2 : 30 * (x + k) = y)
    (h3 : k = (5 / 3) * x) :
    let k_slow := 0.8 * k in
    let t := y / k_slow in
    t = 60 :=
by
  sorry

end escalator_time_l641_641353


namespace greatest_product_sum_2000_eq_1000000_l641_641234

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641234


namespace max_product_of_two_integers_sum_2000_l641_641202

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641202


namespace g_is_defined_correctly_l641_641526

-- Define the function g and the condition it satisfies
def g (x : ℝ) : ℝ := 2 * x + 3

-- The main theorem as a Lean statement
theorem g_is_defined_correctly (g : ℝ → ℝ) :
  (∀ x y : ℝ, (g x * g y - g (x * y)) / 4 = x + y + 1) → (∀ x : ℝ, g x = 2 * x + 3) :=
begin
  sorry
end

end g_is_defined_correctly_l641_641526


namespace max_product_of_two_integers_sum_2000_l641_641200

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641200


namespace kevin_hops_distance_l641_641086

theorem kevin_hops_distance :
  let initial_distance := 2
  let hop_fraction := 1 / 4
  let num_hops := 7
  let hopped_distance := Σ (i : ℕ) in Finset.range num_hops, initial_distance * (hop_fraction ^ (i + 1))
  hopped_distance = (14297 : ℚ) / 2048 := by
  sorry

end kevin_hops_distance_l641_641086


namespace circles_externally_tangent_l641_641610

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def center_dist (C1 C2 : Circle) : ℝ :=
  real.sqrt ((C1.center.1 - C2.center.1) ^ 2 + (C1.center.2 - C2.center.2) ^ 2)

def externally_tangent (C1 C2 : Circle) : Prop :=
  center_dist C1 C2 = C1.radius + C2.radius

noncomputable def C1 : Circle := ⟨(-2, 2), 1⟩
noncomputable def C2 : Circle := ⟨(2, 5), 4⟩

theorem circles_externally_tangent : externally_tangent C1 C2 :=
  sorry

end circles_externally_tangent_l641_641610


namespace solve_for_a_l641_641910

noncomputable def is_real (x : ℝ) : Prop := True

def complex_conj (z : ℂ) : ℂ := conj z

def problem_statement (a : ℝ) : Prop :=
  let z := (1 + Complex.I) / (1 + a * Complex.I)
  z = complex_conj z

theorem solve_for_a : ∀ (a : ℝ), problem_statement a → a = 1 :=
by
  intros a h
  sorry

end solve_for_a_l641_641910


namespace find_ratio_BD_DF_l641_641674

-- Definitions of points on the plane
variables {Point : Type} [AffineSpace Point]

-- Definitions of given points A, B, C, D, E, F, G
variables (A B C D E F G : Point)

-- Assumptions
variables (h1 : Collinear A B C)
variables (h2 : Collinear G B D)
variables (h3 : ∃ E', E = E' + ⟨3, 2, 0⟩)
variables (h4 : LinesMeet A D E G F)
variables (h5 : Angle CAD = Angle EGD)
variables (h6 : Distance E F = Distance F G)
variables (h7 : Ratio AB BC = 1 / 2)
variables (h8 : Ratio CD DE = 3 / 2)

-- Prove that BD : DF = √3 : 2
theorem find_ratio_BD_DF : Ratio BD DF = Real.sqrt 3 / 2 :=
by
  sorry

end find_ratio_BD_DF_l641_641674


namespace find_f6_eq_2_l641_641421

def f : ℝ → ℝ 
| x := if x < 0 then x^3 - 1
       else if -1 ≤ x ∧ x ≤ 1 then f(-x) = -f(x)
       else if x > 1/2 then f(x + 1/2) = f(x - 1/2)
       else sorry -- Placeholder to handle if cases exhaustively

theorem find_f6_eq_2 : f 6 = 2 :=
by
  -- Define conditions given in the problem
  let h1 : ∀ x < 0, f x = x^3 - 1 := sorry,
  let h2 : ∀ x ∈ Icc (-1:ℝ) 1, f (-x) = -f(x) := sorry,
  let h3 : ∀ x > 1/2, f (x + 1/2) = f (x - 1/2) := sorry,
  -- Prove f(6) = 2
  sorry

end find_f6_eq_2_l641_641421


namespace extreme_point_at_zero_l641_641152

noncomputable def f (x : ℝ) := (x^2 - 1)^3 + 2

theorem extreme_point_at_zero : 
  ∃ c : ℝ, c = 0 ∧ (∀ ε : ℝ, ε > 0 → f (0) ≤ f (0 + ε) ∧ f (0) ≤ f (0 - ε)) := 
by
  existsi (0 : ℝ)
  split
  rfl
  sorry

end extreme_point_at_zero_l641_641152


namespace parabola_vertex_and_intercept_l641_641159

theorem parabola_vertex_and_intercept 
  (a b c h : ℝ) 
  (h_ne_zero : h ≠ 0)
  (vertex : ∀ x, y = a * (x - h)^2 + h)
  (intercept : y = a * (0 - h)^2 + h = -2h) :
  b = 6 :=
sorry

end parabola_vertex_and_intercept_l641_641159


namespace no_consecutive_heads_probability_l641_641302

def prob_no_consecutive_heads (n : ℕ) : ℝ := 
  -- This is the probability function for no consecutive heads
  if n = 10 then (9 / 64) else 0

theorem no_consecutive_heads_probability :
  prob_no_consecutive_heads 10 = 9 / 64 :=
by
  -- Proof would go here
  sorry

end no_consecutive_heads_probability_l641_641302


namespace translate_sin_to_left_by_pi_over_4_l641_641626

def original_function (x : ℝ) : ℝ :=
  3 * sin (2 * x - (Real.pi / 6))

def translated_function (x : ℝ) : ℝ :=
  3 * sin (2 * (x + Real.pi / 4) - (Real.pi / 6))

def expected_result_function (x : ℝ) : ℝ :=
  3 * sin (2 * x + (Real.pi / 3))

theorem translate_sin_to_left_by_pi_over_4 :
  translated_function = expected_result_function :=
by
  sorry

end translate_sin_to_left_by_pi_over_4_l641_641626


namespace max_product_two_integers_sum_2000_l641_641210

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641210


namespace smallest_possible_value_of_a_l641_641985

theorem smallest_possible_value_of_a (a b : ℕ) :
  (∃ (r1 r2 r3 : ℕ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    (r1 * r2 * r3 = 2310) ∧
    (a = r1 + r2 + r3)) →
  a = 52 :=
begin
  sorry
end

end smallest_possible_value_of_a_l641_641985


namespace appropriate_import_range_l641_641568

def mung_bean_import_range (p0 : ℝ) (p_desired_min p_desired_max : ℝ) (x : ℝ) : Prop :=
  p0 - (x / 100) ≤ p_desired_max ∧ p0 - (x / 100) ≥ p_desired_min

theorem appropriate_import_range : 
  ∃ x : ℝ, 600 ≤ x ∧ x ≤ 800 ∧ mung_bean_import_range 16 8 10 x :=
sorry

end appropriate_import_range_l641_641568


namespace geometric_sequence_seventh_term_l641_641789

variable {G : Type*} [Field G]

def is_geometric (a : ℕ → G) (q : G) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_seventh_term
  (a : ℕ → G) (q : G)
  (h1 : a 0 + a 1 = 3)
  (h2 : a 1 + a 2 = 6)
  (hq : is_geometric a q) :
  a 6 = 64 := 
sorry

end geometric_sequence_seventh_term_l641_641789


namespace smallest_positive_period_minimum_value_value_f_beta_squared_l641_641825

def f : ℝ → ℝ := λ x, sin (x + 7/4 * Real.pi) + cos (x - 3/4 * Real.pi)

-- Theorem to prove the smallest positive period of f(x) is 2π.
theorem smallest_positive_period : Function.Periodic f (2 * Real.pi) :=
  sorry

-- Theorem to prove the minimum value of f(x) is -√2.
theorem minimum_value : ∃ x, f x = -Real.sqrt 2 :=
  sorry

-- Constants α and β with given constraints
variables (α β : ℝ)
  (hαβ1 : 0 < α)
  (hαβ2 : α < β)
  (hαβ3 : β ≤ Real.pi / 2)
  (hcos1 : Real.cos (β - α) = 4/5)
  (hcos2 : Real.cos (β + α) = -4/5)

-- Theorem to prove [f(β)]^2 = 2.
theorem value_f_beta_squared : (f β) ^ 2 = 2 :=
  sorry

end smallest_positive_period_minimum_value_value_f_beta_squared_l641_641825


namespace collinear_B_C_N_l641_641794

variables {A B M C D E F N : Point}
variables ( hMA_le_MB : dist A M ≤ dist M B)
variables (hSq1 : is_square A M C D)
variables (hSq2 : is_square M B E F)
variables (hSameSide : same_side A B C F)
variables (hCirc1 : is_circumcircle A M C D (circ P))
variables (hCirc2 : is_circumcircle M B E F (circ Q))
variables (hIntersects : ∃ (M N : Point), inter P Q M ∧ inter P Q N)

theorem collinear_B_C_N :
    collinear (Set.insert C (Set.singleton (Set.insert B (Set.singleton N))))
:= sorry

end collinear_B_C_N_l641_641794


namespace concurrency_ad_gf_he_l641_641029

open EuclideanGeometry

variables {A B C D E F G H : Point}

theorem concurrency_ad_gf_he 
  (h_iso : Triangle A B C)
  (h_ab_ac : AB = AC)
  (h_ad_perp_bc : Perp AD BC)
  (h_e : LineThrough D ⟶ E ∈ AB)
  (h_f : LineThrough D ⟶ F ∈ AC)
  (h_g : LineParallel BC ⟶ A ∩ LineThrough D G)
  (h_h : LineParallel BC ⟶ A ∩ LineThrough D H)
  (h_g_parallel_bc : GH ∥ BC) :
  Concurrent AD GF HE :=
begin
  sorry
end

end concurrency_ad_gf_he_l641_641029


namespace line_passes_through_fixed_point_l641_641975

-- Given a line equation kx - y + 1 - 3k = 0
def line_equation (k x y : ℝ) : Prop := k * x - y + 1 - 3 * k = 0

-- We need to prove that this line passes through the point (3,1)
theorem line_passes_through_fixed_point (k : ℝ) : line_equation k 3 1 :=
by
  sorry

end line_passes_through_fixed_point_l641_641975


namespace max_sum_integers_differ_by_60_l641_641186

theorem max_sum_integers_differ_by_60 (b : ℕ) (c : ℕ) (h_diff : 0 < b) (h_sqrt : (Nat.sqrt b : ℝ) + (Nat.sqrt (b + 60) : ℝ) = (Nat.sqrt c : ℝ)) (h_not_square : ¬ ∃ (k : ℕ), k * k = c) :
  ∃ (b : ℕ), b + (b + 60) = 156 := 
sorry

end max_sum_integers_differ_by_60_l641_641186


namespace triangle_area_l641_641682

theorem triangle_area (base height : ℝ) (h_base : base = 4) (h_height : height = 8) :
  (base * height) / 2 = 16 :=
by
  rw [h_base, h_height]
  norm_num
  sorry

end triangle_area_l641_641682


namespace point_D_not_on_graph_l641_641672

def f (x : ℝ) : ℝ := (x - 1) / (x + 2)

theorem point_D_not_on_graph : ¬ ∃ y : ℝ, y = f (-2) := by
  intro h
  cases h with y hy
  have h : -2 + 2 = 0 := rfl
  rw h at hy
  simp at hy
  contradiction

end point_D_not_on_graph_l641_641672


namespace find_k_l641_641346

theorem find_k :
  ∃ k : ℕ , (∃ (a : ℕ → ℕ) (strict_incr : ∀ i, a i < a (i + 1)), 
    (∀ i < k, a i < a (i + 1)) ∧
    (∑ i in finset.range k, 2 ^ (a i)) = ((2 ^ 325 + 1) / (2 ^ 25 + 1))) ∧ 
    (k = 151) :=
sorry

end find_k_l641_641346


namespace min_distance_midpoint_PQ_to_line_C3_l641_641885

-- Definitions from conditions in a)
def C1_curve (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, 3 + Real.sin t)
def C2_curve (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)
def line_C3 (x y : ℝ) : Prop := x - 2 * y = 7

-- Proving the minimum distance as described in proof problem
theorem min_distance_midpoint_PQ_to_line_C3 :
  let P := (-4, 4)
  let Q (θ : ℝ) := C2_curve θ
  let M (θ : ℝ) := ((-2 + 4 * Real.cos θ), (2 + (3 / 2) * Real.sin θ))
  let distance (M : ℝ × ℝ) : ℝ := 
    let x := M.1
    let y := M.2
    abs (x - 2 * y - 7) / Real.sqrt (1^2 + (-2)^2)
  ∀ θ : ℝ, distance (M θ) = (Real.sqrt 5 / 5) * abs (4 * Real.cos θ - 3 * Real.sin θ - 13) :=
sorry

end min_distance_midpoint_PQ_to_line_C3_l641_641885


namespace avg_age_family_now_l641_641276

namespace average_age_family

-- Define initial conditions
def avg_age_husband_wife_marriage := 23
def years_since_marriage := 5
def age_child := 1
def number_of_family_members := 3

-- Prove that the average age of the family now is 19 years
theorem avg_age_family_now :
  (2 * avg_age_husband_wife_marriage + 2 * years_since_marriage + age_child) / number_of_family_members = 19 := by
  sorry

end average_age_family

end avg_age_family_now_l641_641276


namespace total_students_exam_l641_641487

theorem total_students_exam (pass_rate_A fail_A pass_rate_B fail_B pass_rate_C fail_C pass_rate_D fail_D overall_pass_rate : ℚ) 
  (h_pass_rate_A : pass_rate_A = 32 / 100) (h_fail_A : fail_A = 180)
  (h_pass_rate_B : pass_rate_B = 45 / 100) (h_fail_B : fail_B = 140)
  (h_pass_rate_C : pass_rate_C = 28 / 100) (h_fail_C : fail_C = 215)
  (h_pass_rate_D : pass_rate_D = 39 / 100) (h_fail_D : fail_D = 250)
  (h_overall_pass_rate : overall_pass_rate = 38 / 100) :
  let total_students_A := fail_A / (1 - pass_rate_A),
      total_students_B := fail_B / (1 - pass_rate_B),
      total_students_C := fail_C / (1 - pass_rate_C),
      total_students_D := fail_D / (1 - pass_rate_D)
  in total_students_A + total_students_B + total_students_C + total_students_D = 1229 := 
by
  sorry

end total_students_exam_l641_641487


namespace ce_bisects_angle_bcd_l641_641061

open EuclideanGeometry

variables {A B C E K L D : Point}
variables (Γ : Circle) (triangleABC : Triangle)
variables [IsAcuteTriangle triangleABC]
variables (diamAE : Diameter (circumcircle triangleABC) A E)
variables (tangentThroughB : TangentLine Γ B)
variables (intersectionK : Intersection (line AC) (tangentThroughB) K)
variables (projectionL : Projection K (line AE) L)
variables (intersectionD : Intersection (line KL) (line AB) D)

theorem ce_bisects_angle_bcd
  (h1 : acute_triangle A B C)
  (h2 : diameter Γ A E)
  (h3 : ∃ (K : Point), K ∈ (line AC) ∩ (tangentThroughB))
  (h4 : orthogonal_projection K (line AE) L)
  (h5 : ∃ (D : Point), D ∈ (line KL) ∩ (line AB)) :
  is_angle_bisector C E B D :=
sorry

end ce_bisects_angle_bcd_l641_641061


namespace solution_a_l641_641676

noncomputable def problem_a (a b c y : ℕ) : Prop :=
  a + b + c = 30 ∧ b + c + y = 30 ∧ a = 2 ∧ y = 3

theorem solution_a (a b c y x : ℕ)
  (h : problem_a a b c y)
  : x = 25 :=
by sorry

end solution_a_l641_641676


namespace integral_sin4_cos3_l641_641756

theorem integral_sin4_cos3 (C : ℝ) :
  ∫ x in ℝ, (sin x)^4 * (cos x)^3 = (1 / 5) * (sin x)^5 - (1 / 7) * (sin x)^7 + C :=
by sorry

end integral_sin4_cos3_l641_641756


namespace find_circle_and_line_eqns_l641_641805

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 16

noncomputable def line1_eq (x : ℝ) : Prop := x = 1

noncomputable def line2_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 23 = 0

theorem find_circle_and_line_eqns :
  (circle_eq (-1) 1) ∧ (circle_eq 3 5) ∧ 
  (∃ x y, (2 * x - y - 5 = 0) ∧ circle_eq x y) ∧ 
  (∃ k, (line1_eq 1 ∧ |(4 * k^2 / (k^2 + 1) + 12)| = 16) ∨ (line2_eq 1 5 ∧ |(4 * k^2 / (16 * k^2 + 9) + 3)| = 16)) :=
sorry

end find_circle_and_line_eqns_l641_641805


namespace slope_of_tangent_line_at_neg1_l641_641813

-- Define the function f(x)
def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := x^2

-- State the theorem
theorem slope_of_tangent_line_at_neg1 : 
  f_prime (-1) = 1 := 
by 
  -- Proof is omitted
  sorry

end slope_of_tangent_line_at_neg1_l641_641813


namespace coefficient_b_nonzero_l641_641744

noncomputable def Q (x : ℝ) : ℝ := x * (x + 1) * (x - 1) * (x + 2) * (x - 2)

-- Statement to be proven: The coefficient 'b' in Q(x) = x^5 + ax^4 + bx^3 + cx^2 + dx + f cannot be zero.
theorem coefficient_b_nonzero : 
  let a := 0 in
  let b := -5 in
  let c := 0 in
  let d := 4 in
  let f := 0 in
  b ≠ 0 :=
by {
  -- Sorry is used to skip the actual proof
  sorry
}

end coefficient_b_nonzero_l641_641744


namespace range_g_l641_641376

open Real

def g (A : ℝ) : ℝ := (cos A * (2 * sin A ^ 2 + sin A ^4 + 2 * cos A ^ 2 + cos A ^ 2 * sin A ^ 2)) / 
                      (cot A * (csc A - cos A * cot A))

theorem range_g (A : ℝ) (hA : ∀ n : ℤ, A ≠ n * π / 2) : 
  Set.range (λ A, g A) = Set.Icc 2 3 :=
by
  sorry

end range_g_l641_641376


namespace equilateral_triangle_perimeter_l641_641494

theorem equilateral_triangle_perimeter (z : ℂ) (hz : z ≠ 0 ∧ z ≠ 1) 
    (h_equilateral : ∃ k : ℤ, (k = 1 ∨ k = 2) ∧ (z^3 - z = (complex.exp (complex.I * (2 * real.pi / 3) * k) * (z^2 - z)))) :
    abs (3 * (z^2 - z)) = 3 * real.sqrt 3 :=
by sorry

end equilateral_triangle_perimeter_l641_641494


namespace positive_roots_implies_nonnegative_m_l641_641447

variables {x1 x2 m : ℝ}

theorem positive_roots_implies_nonnegative_m (h1 : x1 > 0) (h2 : x2 > 0)
  (h3 : x1 * x2 = 1) (h4 : x1 + x2 = m + 2) : m ≥ 0 :=
by
  sorry

end positive_roots_implies_nonnegative_m_l641_641447


namespace simplified_expression_at_one_l641_641573

noncomputable def original_expression (a : ℚ) : ℚ :=
  (2 * a + 2) / a / (4 / (a ^ 2)) - a / (a + 1)

theorem simplified_expression_at_one : original_expression 1 = 1 / 2 := by
  sorry

end simplified_expression_at_one_l641_641573


namespace smallest_positive_sum_exists_l641_641748

theorem smallest_positive_sum_exists (a : Fin 100 → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
∃ S > 0, S = ∑ i in Finset.range 100, ∑ j in Finset.Ico i 100, a i * a j ∧ S = 22 :=
sorry

end smallest_positive_sum_exists_l641_641748


namespace hash_difference_l641_641465

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 8 5) - (hash 5 8) = -12 := by
  sorry

end hash_difference_l641_641465


namespace green_blue_tile_difference_is_15_l641_641080

def initial_blue_tiles : Nat := 13
def initial_green_tiles : Nat := 6
def second_blue_tiles : Nat := 2 * initial_blue_tiles
def second_green_tiles : Nat := 2 * initial_green_tiles
def border_green_tiles : Nat := 36
def total_blue_tiles : Nat := initial_blue_tiles + second_blue_tiles
def total_green_tiles : Nat := initial_green_tiles + second_green_tiles + border_green_tiles
def tile_difference : Nat := total_green_tiles - total_blue_tiles

theorem green_blue_tile_difference_is_15 : tile_difference = 15 := by
  sorry

end green_blue_tile_difference_is_15_l641_641080


namespace amicable_284_220_l641_641677

def proper_divisors (n : ℕ) := {d : ℕ | d ∣ n ∧ d < n}

def sum_proper_divisors (n : ℕ) : ℕ := (proper_divisors n).sum

theorem amicable_284_220 :
  sum_proper_divisors 284 = 220 ∧ sum_proper_divisors 220 = 284 :=
by
  sorry

end amicable_284_220_l641_641677


namespace max_product_of_two_integers_sum_2000_l641_641196

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641196


namespace total_paving_cost_correct_l641_641622

noncomputable def total_cost_of_paving : ℝ :=
  let area_rect := 5.5 * 3.75 in
  let cost_rect := area_rect * 1400 in
  let area_triangle := (4 * 3) / 2 in
  let cost_triangle := area_triangle * 1500 in
  let area_trapezoid := ((6 + 3.5) * 2.5) / 2 in
  let cost_trapezoid := area_trapezoid * 1600 in
  cost_rect + cost_triangle + cost_trapezoid

theorem total_paving_cost_correct : total_cost_of_paving = 56875 :=
by
  sorry

end total_paving_cost_correct_l641_641622


namespace max_product_of_two_integers_sum_2000_l641_641251

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641251


namespace length_of_AG_l641_641496

variables {A B C D E G : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space G]

def triangle_ABC (A B C : Type) : Prop :=
∃ (AB AC : ℕ), (∃ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
  (metric_space.dist A B = 3) ∧
  (metric_space.dist A C = 3 * real.sqrt 3) ∧
  (metric_space.dist B C = real.sqrt (3^2 + (3 * real.sqrt 3)^2)))

def right_angle_at_A (A B C : Type) : Prop :=
∃ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
  (∃ α : real.angle, α = π / 2) ∧
  (real.angle.sine α = 0) ∧
  (real.angle.tangent α = 0)

def altitude_AD (A B C D : Type) : Prop :=
∃ (A B C D : Type) [metric_space D],
  (metric_space.dist A D = 3 * real.sqrt 3 / 2)

def median_BE (A B E : Type) := 
∃ (A B E : Type) [metric_space E],
  (metric_space.dist B E = 3 / 2)

def intersects_be_at_G (B E G : Type) : Prop :=
∃ (B E G : Type) [metric_space G],
  (metric_space.dist B G = 3 * real.sqrt 3 / 14)

theorem length_of_AG (A B C D E G : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space G]
  (h1 : triangle_ABC A B C)
  (h2 : right_angle_at_A A B C)
  (h3 : altitude_AD A B C D)
  (h4 : median_BE A B E)
  (h5 : intersects_be_at_G B E G) :
  metric_space.dist A G = 3 * real.sqrt 3 / 14 :=
sorry

end length_of_AG_l641_641496


namespace good_numbers_gt_six_digit_l641_641634

/-- We define the function Z(A) where we write the digits of A in base 10 form in reverse. -/
def Z (A : ℕ) : ℕ :=
  A.digits.reverse.digits_to_nat 10

/-- A number A is called "good" if:
  1. The first and last digits of A are different.
  2. None of its digits are 0.
  3. Z(A^2) = (Z(A))^2.
-/
def is_good (A : ℕ) : Prop :=
  A.digits.head ≠ A % 10 ∧
  ∀ d ∈ A.digits, d ≠ 0 ∧
  Z (A * A) = (Z A) * (Z A)

/-- The set of good numbers greater than 10^6 is precisely {111112, 211111, 1111112, 2111111}. -/
theorem good_numbers_gt_six_digit :
  {A : ℕ | A > 10^6 ∧ is_good A} = {111112, 211111, 1111112, 2111111} :=
by
  sorry

end good_numbers_gt_six_digit_l641_641634


namespace boat_travel_time_downstream_is_27_minutes_l641_641997

-- Define the conditions
def speed_boat_still_water : ℝ := 20
def rate_current : ℝ := 5
def distance_downstream : ℝ := 11.25

-- Effective speed downstream
def effective_speed_downstream : ℝ :=
  speed_boat_still_water + rate_current
   
-- Time taken to travel downstream in hours
def time_downstream_hours (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

-- Time taken to travel downstream in minutes
def time_downstream_minutes (time_hours : ℝ) : ℝ :=
  time_hours * 60
  
-- The theorem to be proven
theorem boat_travel_time_downstream_is_27_minutes :
  time_downstream_minutes (time_downstream_hours distance_downstream effective_speed_downstream) = 27 :=
by
  sorry

end boat_travel_time_downstream_is_27_minutes_l641_641997


namespace probability_not_greater_than_4_l641_641663

theorem probability_not_greater_than_4 (faces: Finset ℕ) (labels: ∀ x, x ∈ faces → x ∈ {1, 2, 3, 4, 5, 6}): 
  (∃! x, x ∈ faces ∧ x ≤ 4) → (probability (λ x, x ≤ 4) = 2 / 3) :=
sorry

end probability_not_greater_than_4_l641_641663


namespace number_of_correct_propositions_l641_641831

-- Definitions of points, lines, and planes used in conditions

def Point : Type := ℝ × ℝ × ℝ
def Line : Type := Set Point
def Plane : Type := Set Point

-- Conditions for the propositions

variables (A B C : Point) (a b : Line) (α : Plane)

-- Definitions of relations between points, lines, and planes

def on_plane (P : Point) (p : Plane) : Prop := P ∈ p
def on_line (P : Point) (l : Line) : Prop := P ∈ l
def line_intersects_plane (l : Line) (p : Plane) : Prop := ∃ P : Point, P ∈ l ∧ P ∈ p
def line_parallel_plane (l : Line) (p : Plane) : Prop := ¬ line_intersects_plane l p
def perpendicular (l : Line) (p : Plane) : Prop := ∀ x : Point, x ∈ l → x ∈ p
def perpendicular_lines (l₁ l₂ : Line) : Prop := ∀ x : Point, x ∈ l₁ → x ∉ l₂

-- Propositions to be checked

def prop1 : Prop :=
  on_plane A α ∧ on_plane B α ∧ on_line C (λ P, P = (A, B)) →
  on_plane C α

def prop2 : Prop :=
  ¬ on_plane A α ∧ ¬ on_plane B α →
  line_parallel_plane (λ P, P = (A, B)) α

def prop3 : Prop :=
  perpendicular a α ∧ perpendicular_lines b a →
  line_parallel_plane b α

-- Proof statement: to prove the number of correct propositions is 1

theorem number_of_correct_propositions : (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0) = 1 :=
  by sorry

end number_of_correct_propositions_l641_641831


namespace box_volume_max_l641_641850

noncomputable def volume (a x : ℝ) : ℝ :=
  (a - 2 * x) ^ 2 * x

theorem box_volume_max (a : ℝ) (h : 0 < a) :
  ∃ x, 0 < x ∧ x < a / 2 ∧ volume a x = volume a (a / 6) ∧ volume a (a / 6) = (2 * a^3) / 27 :=
by
  sorry

end box_volume_max_l641_641850


namespace inequality_proof_l641_641917

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
by 
  sorry

end inequality_proof_l641_641917


namespace distance_from_origin_to_center_l641_641068

/-- Define the parametric equations of the circle -/
def x (θ : ℝ) : ℝ := 2 * Real.cos θ
def y (θ : ℝ) : ℝ := 2 + 2 * Real.sin θ

/-- Define the center of the circle based on the parametric equations -/
def center : ℝ × ℝ := (0, 2)

/-- Define the distance formula between two points in 2D Cartesian coordinates -/
def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - 0) ^ 2 + (p.2 - 0) ^ 2)

/-- Proof statement that the distance from the origin to the center of the circle is 2 -/
theorem distance_from_origin_to_center :
  distance_from_origin center = 2 :=
sorry

end distance_from_origin_to_center_l641_641068


namespace angle_QPH_ninety_degrees_l641_641580

noncomputable def midpoint (P Q : Point) : Point :=
sorry -- Definition of the midpoint

noncomputable def symmetric_point (P Q : Point) (line : Line) : Point :=
sorry -- Definition of symmetric point with respect to a line

noncomputable def orthocenter (A B C : Point) : Point :=
sorry -- Definition of orthocenter of a triangle

variables {A B C A1 C1 H Q P : Point}
variables (AA1 CC1 : Line)
variables (triangle_ABC : Triangle A B C)

-- Assumptions
variables (hA : is_altitude triangle_ABC AA1)
variables (hB : is_altitude triangle_ABC CC1)
variables (hH : H = orthocenter A B C)
variables (midK : Point)
variables (hK : midpoint AC = midK)
variables (hQ : Q = symmetric_point midK A AA1)
variables (hP : P = midpoint A1 C1)

theorem angle_QPH_ninety_degrees (h_conditions : 
  is_altitude triangle_ABC AA1 ∧ 
  is_altitude triangle_ABC CC1 ∧ 
  H = orthocenter A B C ∧ 
  Q = symmetric_point (midpoint A C) A AA1 ∧ 
  P = midpoint A1 C1) : 
  ∠ Q P H = 90 :=
sorry

end angle_QPH_ninety_degrees_l641_641580


namespace f_inequality_l641_641432

-- Function definition for f(n)
def f (n : ℕ) : ℚ := (List.range n).map (λ k, 1 / (k + 1 : ℚ)).sum

-- The statement to prove
theorem f_inequality (n : ℕ) (h : n > 0) : f (2 ^ n) ≥ (n + 2) / 2 := by
  sorry

end f_inequality_l641_641432


namespace smallest_a_l641_641978

def root_product (P : Polynomial ℚ) : ℚ :=
  P.coeff 0

def poly_sum_roots_min_a (r1 r2 r3 : ℤ) (a b c : ℚ) : Prop :=
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
  r1 * r2 * r3 = 2310 ∧
  root_product (Polynomial.monomial 3 1 - Polynomial.monomial 2 a + Polynomial.monomial 1 b - Polynomial.monomial 0 2310) = 2310 ∧
  r1 + r2 + r3 = a

theorem smallest_a : ∃ a b : ℚ, ∀ r1 r2 r3 : ℤ, poly_sum_roots_min_a r1 r2 r3 a b 2310 → a = 28
  by sorry

end smallest_a_l641_641978


namespace f_even_f_period_f_not_decreasing_f_min_value_l641_641434

def f (x : ℝ) : ℝ := cos x ^ 4 + sin x ^ 2

-- 1. f(x) is an even function.
theorem f_even : ∀ x : ℝ, f (-x) = f x := 
by 
  sorry

-- 2. π/2 is a period of the function f(x).
theorem f_period : ∀ x : ℝ, f (x + π / 2) = f x := 
by 
  sorry

-- 3. The function f(x) is not decreasing in the interval (0, π/2).
theorem f_not_decreasing : ¬ (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < π / 2 → f x > f y) := 
by 
  sorry

-- 4. The minimum value of the function f(x) is 3/4.
theorem f_min_value : ∃ x : ℝ, f x = 3 / 4 := 
by 
  sorry

end f_even_f_period_f_not_decreasing_f_min_value_l641_641434


namespace imag_part_of_z_l641_641787

theorem imag_part_of_z : 
  let z : ℂ := (i - 3) / (i + 1)
  in z.im = 2 := 
by
  let z : ℂ := (i - 3) / (i + 1)
  have : z = -1 + 2 * i := 
  sorry
  -- proving the imaginary part.
  rw this
  show ( -1 + 2 * i).im = 2
  simp

end imag_part_of_z_l641_641787


namespace max_cells_colored_without_gremista_l641_641577

def is_gremista (board : ℕ → ℕ → Bool) : Prop :=
  ∃ (rows : Finset ℕ) (cols : Finset ℕ),
    ((rows.card = 3 ∧ cols.card = 2) ∨ (rows.card = 2 ∧ cols.card = 3)) ∧
    (∀ r ∈ rows, ∀ c ∈ cols, board r c = tt)

def can_color_without_gremista (n : ℕ) : Prop :=
  ∃ board : ℕ → ℕ → Bool,
    (∀ r c, board r c = tt → r < 10 ∧ c < 10) ∧ -- Ensure cells are within 10x10 grid
    (Finset.card (Finset.univ.image (λ rc, (rc.1, rc.2)) \ (λ rc, board rc.1 rc.2 = tt)) = n) ∧
    ¬is_gremista board

theorem max_cells_colored_without_gremista : ∃ n, can_color_without_gremista n ∧ ∀ m, can_color_without_gremista m → m ≤ n :=
begin
  use 46,
  sorry
end

end max_cells_colored_without_gremista_l641_641577


namespace eight_primes_condition_l641_641757

theorem eight_primes_condition (x : Fin 8 → ℕ) (h_prime : ∀ i, Prime (x i)) :
  let S := (∑ i, x i ^ 2)
  let P := (∏ i, x i)
  4 * P - S = 992 → 
  ∀ i, x i = 2 := 
sorry

end eight_primes_condition_l641_641757


namespace no_two_consecutive_heads_l641_641295

/-- The probability that two heads never appear consecutively when a coin is tossed 10 times is 9/64. -/
theorem no_two_consecutive_heads (tosses : ℕ) (prob: ℚ):
  tosses = 10 → 
  (prob = 9 / 64) → 
  prob = probability_no_consecutive_heads tosses := sorry

end no_two_consecutive_heads_l641_641295


namespace find_expression_value_l641_641014

theorem find_expression_value (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := 
by 
  sorry

end find_expression_value_l641_641014


namespace contrapositive_example_l641_641590

theorem contrapositive_example (a b m : ℝ) :
  (a > b → a * (m^2 + 1) > b * (m^2 + 1)) ↔ (a * (m^2 + 1) ≤ b * (m^2 + 1) → a ≤ b) :=
by sorry

end contrapositive_example_l641_641590


namespace distance_between_planes_is_zero_l641_641364

def plane1 (x y z : ℝ) := 3 * x - 9 * y + 6 * z = 12
def plane2 (x y z : ℝ) := 6 * x - 18 * y + 12 * z = 24

theorem distance_between_planes_is_zero :
  ∀ x y z : ℝ, plane1 x y z → plane2 x y z → 0 = 0 :=
begin
  intros x y z h1 h2,
  sorry -- proof not required as per instructions
end

end distance_between_planes_is_zero_l641_641364


namespace quadratic_csq_l641_641952

theorem quadratic_csq (x q t : ℝ) (h : 9 * x^2 - 36 * x - 81 = 0) (hq : q = -2) (ht : t = 13) :
  q + t = 11 :=
by
  sorry

end quadratic_csq_l641_641952


namespace infinite_squares_in_arithmetic_progression_l641_641429

theorem infinite_squares_in_arithmetic_progression
  (a d : ℕ) (hposd : 0 < d) (hpos : 0 < a) (k n : ℕ)
  (hk : a + k * d = n^2) :
  ∃ (t : ℕ), ∃ (m : ℕ), (a + (k + t) * d = m^2) := by
  sorry

end infinite_squares_in_arithmetic_progression_l641_641429


namespace greatest_product_sum_2000_eq_1000000_l641_641230

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641230


namespace smallest_value_of_a_l641_641986

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end smallest_value_of_a_l641_641986


namespace no_two_heads_consecutively_probability_l641_641286

theorem no_two_heads_consecutively_probability :
  (∃ (total_sequences : ℕ) (valid_sequences : ℕ),
    total_sequences = 2^10 ∧ valid_sequences = 1 + 10 + 36 + 56 + 35 + 6 ∧
    (valid_sequences / total_sequences : ℚ) = 9 / 64) :=
begin
  sorry
end

end no_two_heads_consecutively_probability_l641_641286


namespace probability_not_both_ends_l641_641771

theorem probability_not_both_ends :
  let total_arrangements := 120
  let both_ends_arrangements := 12
  let favorable_arrangements := total_arrangements - both_ends_arrangements
  let probability := favorable_arrangements / total_arrangements
  total_arrangements = 120 ∧ both_ends_arrangements = 12 ∧ favorable_arrangements = 108 ∧ probability = 0.9 :=
by
  sorry

end probability_not_both_ends_l641_641771


namespace MorseCodeDistinctSymbols_l641_641033

theorem MorseCodeDistinctSymbols:
  (1.sequence (λ _, bool).length = {1, 2, 3, 4, 5}).card = 62 :=
by
  sorry

end MorseCodeDistinctSymbols_l641_641033


namespace total_pushups_l641_641270

theorem total_pushups (zachary_pushups : ℕ) (david_more_pushups : ℕ) 
  (h1 : zachary_pushups = 44) (h2 : david_more_pushups = 58) : 
  zachary_pushups + (zachary_pushups + david_more_pushups) = 146 :=
by
  sorry

end total_pushups_l641_641270


namespace geoff_election_l641_641274

theorem geoff_election (Votes: ℝ) (Percent: ℝ) (ExtraVotes: ℝ) (x: ℝ) 
  (h1 : Votes = 6000) 
  (h2 : Percent = 1) 
  (h3 : ExtraVotes = 3000) 
  (h4 : ReceivedVotes = (Percent / 100) * Votes) 
  (h5 : TotalVotesNeeded = ReceivedVotes + ExtraVotes) 
  (h6 : x = (TotalVotesNeeded / Votes) * 100) :
  x = 51 := 
  by 
    sorry

end geoff_election_l641_641274


namespace abs_inequality_solution_l641_641951

theorem abs_inequality_solution (x : ℝ) :
  |2 * x - 2| + |2 * x + 4| < 10 ↔ x ∈ Set.Ioo (-4 : ℝ) (2 : ℝ) := 
by sorry

end abs_inequality_solution_l641_641951


namespace locus_of_tangent_points_l641_641791

theorem locus_of_tangent_points (O : Point) (R : ℝ) :
    (∀ (X M: Point), (XM: Segment) X M: ((XO: Segment) X O) → 
    ((intersect_tangent semicircle (point_of_tangency M) = M) → XM = XO) ) → 
    (locus M =
        { P : Point | dist(P, line parallel to diameter AB at distance R) ∧ 
                     P ∉ {A, B, center of the semicircle} }) :=
sorry

end locus_of_tangent_points_l641_641791


namespace last_two_digits_10pow_93_div_10pow_31_plus_3_l641_641605

theorem last_two_digits_10pow_93_div_10pow_31_plus_3 :
  ∀ (m n : ℕ), m = 10^93 → n = 10^31 + 3 →
  (nat.floor ((m : ℚ) / n)).%100 = 08 :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end last_two_digits_10pow_93_div_10pow_31_plus_3_l641_641605


namespace vector_norm_squared_addition_l641_641108

open Matrix

noncomputable def proof_problem (a b : Matrix (Fin 2) (Fin 1) ℝ) : Prop :=
  let n := (a + b) / (2 : ℝ)
  (n = ![![4], ![-1]] ∧ (a ⬝ b) = 10) → (re (a ⬝ a) + re (b ⬝ b) = 48)

theorem vector_norm_squared_addition (a b : Matrix (Fin 2) (Fin 1) ℝ) : proof_problem a b :=
by
  sorry

end vector_norm_squared_addition_l641_641108


namespace average_ab_l641_641608

theorem average_ab {a b : ℝ} (h : (3 + 5 + 7 + a + b) / 5 = 15) : (a + b) / 2 = 30 :=
by
  sorry

end average_ab_l641_641608


namespace chord_length_triangle_area_l641_641445

-- Definitions and assumptions based on the given conditions
def parabola (C : ℝ → ℝ → Prop) : Prop := ∀ x y, C x y ↔ y^2 = 4 * x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
def line (l : ℝ → ℝ → Prop) : Prop := ∀ x y, l x y ↔ y = x - 2
def intersect (C l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  (C A.fst A.snd ∧ l A.fst A.snd) ∧ (C B.fst B.snd ∧ l B.fst B.snd) ∧ A ≠ B

-- Resulting statements to be proved
theorem chord_length {C l : ℝ → ℝ → Prop} {A B : ℝ × ℝ}
  (h_parabola : parabola C) (h_line : line l) (h_intersect : intersect C l A B) :
  dist A B = 4 * real.sqrt 6 :=
sorry

theorem triangle_area {C l : ℝ → ℝ → Prop} {A B F : ℝ × ℝ}
  (h_f : focus F) (h_parabola : parabola C) (h_line : line l) (h_intersect : intersect C l A B) :
  (1/2) * (dist A B) * (abs (F.snd - (A.snd + B.snd) / 2) / real.sqrt(1 + 1)) = 2 * real.sqrt 3 :=
sorry

end chord_length_triangle_area_l641_641445


namespace max_product_of_two_integers_sum_2000_l641_641197

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641197


namespace trevor_brother_twice_age_years_ago_l641_641181

theorem trevor_brother_twice_age_years_ago :
  ∃ (x : ℕ), (x = 20) ∧ (let trevor_age := 26 in 
                          let brother_age := 32 in
                          2 * (trevor_age - x) = (brother_age - x)) :=
sorry

end trevor_brother_twice_age_years_ago_l641_641181


namespace max_product_of_sum_2000_l641_641245

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641245


namespace WXYZ_is_parallelogram_l641_641087

variables {A B C D W X Y Z : Point}
variables (incenters_parallelogram : ∃ (I1 I2 I3 I4: Point), 
  ∃ (incenter_AWZ : Incenter I1 (Triangle A W Z)),
  ∃ (incenter_BXW : Incenter I2 (Triangle B X W)),
  ∃ (incenter_CYX : Incenter I3 (Triangle C Y X)),
  ∃ (incenter_DZY : Incenter I4 (Triangle D Z Y)),
  IsParallelogram (Quadrilateral I1 I2 I3 I4))

theorem WXYZ_is_parallelogram (h1 : IsParallelogram (Quadrilateral A B C D))
  (h2 : OnSide W A B) (h3 : OnSide X B C) 
  (h4 : OnSide Y C D) (h5 : OnSide Z D A) : 
  IsParallelogram (Quadrilateral W X Y Z) :=
by sorry

end WXYZ_is_parallelogram_l641_641087


namespace books_left_classics_l641_641504

theorem books_left_classics {authors : ℕ} {books_per_author : ℕ} {books_lent : ℕ} {books_misplaced : ℕ}
    (authors_eq : authors = 10)
    (books_per_author_eq : books_per_author = 45)
    (books_lent_eq : books_lent = 17)
    (books_misplaced_eq : books_misplaced = 8) :
    authors * books_per_author - (books_lent + books_misplaced) = 425 :=
by
  rw [authors_eq, books_per_author_eq, books_lent_eq, books_misplaced_eq]
  norm_num
  rfl

end books_left_classics_l641_641504


namespace complex_equal_real_imag_l641_641163

theorem complex_equal_real_imag (b : ℝ) : (let c := (1 + b * complex.I) / (1 - complex.I) in 
                                          c.re = c.im) ↔ b = 0 :=
by
  sorry

end complex_equal_real_imag_l641_641163


namespace max_product_of_two_integers_sum_2000_l641_641198

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641198


namespace area_ratio_trapezoid_STQR_to_triangle_PQR_l641_641183

theorem area_ratio_trapezoid_STQR_to_triangle_PQR
  (P Q R X U S T : Type)
  [plane_geometry P Q R X U S T] 
  [equilateral_triangle P Q R]
  (h1 : parallel X Y Q R)
  (h2 : parallel U V Q R)
  (h3 : parallel S T Q R)
  (h4 : PX = XU)
  (h5 : XU = US)
  (h6 : US = ST) :
  area STQR / area PQR = 9 / 25 :=
sorry

end area_ratio_trapezoid_STQR_to_triangle_PQR_l641_641183


namespace Morse_code_number_of_distinct_symbols_l641_641038

def count_sequences (n : ℕ) : ℕ :=
  2 ^ n

theorem Morse_code_number_of_distinct_symbols :
  (count_sequences 1) + (count_sequences 2) + (count_sequences 3) + (count_sequences 4) + (count_sequences 5) = 62 :=
by
  simp [count_sequences]
  norm_num
  sorry

end Morse_code_number_of_distinct_symbols_l641_641038


namespace simplify_expression_l641_641842

theorem simplify_expression (k : ℤ) (c d : ℤ) 
(h1 : (5 * k + 15) / 5 = c * k + d) 
(h2 : ∀ k, d + c * k = k + 3) : 
c / d = 1 / 3 := 
by 
  sorry

end simplify_expression_l641_641842


namespace no_domino_tiling_possible_l641_641891

-- Define a type for representing the board 
structure Chessboard :=
  (size : ℕ × ℕ) -- board dimensions
  (squares : list (ℕ × ℕ)) -- list of occupied squares

-- Define the proposition and conditions
theorem no_domino_tiling_possible : ¬ ∃ (tiling : list ((ℕ × ℕ) × (ℕ × ℕ))), 
  (forall dom ∈ tiling, -- dominos in tiling
     let (cell1, cell2) := dom in 
     cell1 ≠ cell2 ∧ abs (cell1.fst - cell2.fst) + abs (cell1.snd - cell2.snd) = 1) -- ensure each domino covers 2 adjacent squares
  ∧ (forall x y, (x, y) ∉ Chessboard.squares → ((x, y) = (0, 0) ∨ (x, y) = (Chessboard.size.fst - 1, Chessboard.size.snd - 1))) -- two opposite corners removed
  ∧ (sum dom ∈ tiling, dom.snd - dom.fst = 2 ∨ dom.snd - dom.fst = 0) := -- tiling covers modified board
sorry

end no_domino_tiling_possible_l641_641891


namespace smallest_n_satisfies_inequality_l641_641377

theorem smallest_n_satisfies_inequality :
  ∃ n : ℕ, n = 5 ∧ (27 + 18 + 12 + 8 + ∑ i in finset.range (n+1), 27*(2/3)^i > 72) :=
by
  sorry

end smallest_n_satisfies_inequality_l641_641377


namespace min_value_complex_expr_l641_641099

open Complex

theorem min_value_complex_expr (z : ℂ) (h : abs z = 2) : 
  ∃ w : ℂ, w = (z - 2) ^ 2 * (z + 2) ∧ abs w = 0 :=
by
  use (z - 2) ^ 2 * (z + 2)
  split
  · refl
  sorry

end min_value_complex_expr_l641_641099


namespace length_of_ZW_l641_641072

variable (YX XW YZ ZW : ℝ)
variable (θ : ℝ)

-- Conditions
def conditions : Prop :=
  YX = 3 ∧
  XW = 5 ∧
  YZ = 6 ∧
  ZW = 4 ∧
  ∠ XYZ = ∠ YXW

-- Problem statement
theorem length_of_ZW (h : conditions YX XW YZ ZW θ) : ZW = sqrt 51 := 
sorry

end length_of_ZW_l641_641072


namespace Morse_code_sequences_l641_641047

theorem Morse_code_sequences : 
  let symbols (n : ℕ) := 2^n in
  symbols 1 + symbols 2 + symbols 3 + symbols 4 + symbols 5 = 62 :=
by
  sorry

end Morse_code_sequences_l641_641047


namespace all_solutions_of_diophantine_eq_l641_641410

theorem all_solutions_of_diophantine_eq
  (a b c x0 y0 : ℤ) (h_gcd : Int.gcd a b = 1)
  (h_sol : a * x0 + b * y0 = c) :
  ∀ x y : ℤ, (a * x + b * y = c) →
  ∃ t : ℤ, x = x0 + b * t ∧ y = y0 - a * t :=
by
  sorry

end all_solutions_of_diophantine_eq_l641_641410


namespace find_sales_expression_maximize_profit_l641_641923

variables (x y : ℝ)
def cost_price := 40
def initial_price := 60
def initial_sales := 20

theorem find_sales_expression
  (hx : x = 60 → y = 20 - 2 * ((x - initial_price) / 5) * 10)
  : y = 140 - 2 * x :=
by sorry

theorem maximize_profit
  (cost : ℝ := cost_price)
  (sales_expression : y = 140 - 2 * x)
  : let profit := (x - cost) * y in
    (∀ x, profit ≤ -2 * (x - 55)^2 + 450) ∧
    (∃ (max_x : ℝ), max_x = 55 ∧ profit = 450)
:=
by sorry

end find_sales_expression_maximize_profit_l641_641923


namespace smallest_positive_integer_pair_five_l641_641742

def is_pair {a b n : ℕ} (h : a^2 + b^2 = n) : Prop :=
a ≥ 1 ∧ b ≥ 1

def number_of_pairs (n : ℕ) : ℕ :=
(nat.filter (λ p, is_pair (p.1 ^ 2 + p.2 ^ 2 = n))).length

theorem smallest_positive_integer_pair_five :
  ∃ n : ℕ, number_of_pairs n = 5 ∧ (∀ m : ℕ, m < n -> number_of_pairs m ≠ 5) :=
begin
  let n := 200,
  sorry
end

end smallest_positive_integer_pair_five_l641_641742


namespace cross_country_winning_scores_l641_641057

theorem cross_country_winning_scores :
  let total_sum := (Finset.range 12).sum (λ x, x + 1)
  let max_winning_score := total_sum / 2
  let min_winning_score := (Finset.range 6).sum (λ x, x + 1)
  max_winning_score = 39 → min_winning_score = 21 →
  18 = max_winning_score - min_winning_score + 1 :=
by
  let total_sum := (Finset.range 12).sum (λ x, x + 1)
  let max_winning_score := total_sum / 2
  let min_winning_score := (Finset.range 6).sum (λ x, x + 1)
  assume h1 : max_winning_score = 39
  assume h2 : min_winning_score = 21
  have h3 : 18 = 39 - 21 + 1, by linarith
  exact h3


end cross_country_winning_scores_l641_641057


namespace handshake_count_l641_641059

theorem handshake_count:
  ∃ (total_people friends subgroup total_handshakes : ℕ),
    total_people = 40 ∧
    friends = 25 ∧
    subgroup = 15 ∧
    total_handshakes = 345 →
    total_handshakes = 25 * 10  + (15.choose 2 - 5 * 1) :=
begin
  sorry
end

end handshake_count_l641_641059


namespace triangle_abc_inequality_l641_641889

theorem triangle_abc_inequality
  (a b c : ℝ)
  (A B C : ℝ)
  (hA : A + B + C = π)
  (hA_pos : 0 < A ∧ A < π)
  (hB_pos : 0 < B ∧ B < π)
  (hC_pos : 0 < C ∧ C < π)
  (h2A : a = 2 * R * sin A)
  (h2B : b = 2 * R * sin B)
  (h2C : c = 2 * R * sin C)
  (R r : ℝ)
  (hAe : R = a / (2 * sin A))
  (hBe : R = b / (2 * sin B))
  (hCe : R = c / (2 * sin C))
  (hAr : r = (a * b * c) / (4 * R))
  (hBe : r = (a + b + c) / 2)
: 
  (b^2 + c^2) / a + (c^2 + a^2) / b + (a^2 + b^2) / c ≥ 2 * (a + b + c) := 
sorry

end triangle_abc_inequality_l641_641889


namespace degree_polynomial_l641_641846

-- Define the variables and conditions as per the problem statement
variables (m n : ℕ)
variables (hm : m > 0) (hn : n > 0)

-- Define the polynomial
def polynomial (m n : ℕ) := (λ x y : ℕ, x^m + y^n + (4^(m+n)): ℕ)

-- State the theorem
theorem degree_polynomial (x y : ℕ) (hm : m > 0) (hn : n > 0) :
  degree (polynomial m n x y) = max m n :=
sorry

end degree_polynomial_l641_641846


namespace max_product_of_two_integers_sum_2000_l641_641255

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641255


namespace sum_of_angles_of_z5_eq_neg_half_minus_sqrt3i_div2_l641_641379

theorem sum_of_angles_of_z5_eq_neg_half_minus_sqrt3i_div2 :
  let θk := [48, 120, 192, 264, 336].map (λ x => x * real.pi/180)
  in (θk.sum = 960 * real.pi / 180) := sorry

end sum_of_angles_of_z5_eq_neg_half_minus_sqrt3i_div2_l641_641379


namespace directors_dividends_correct_l641_641648

theorem directors_dividends_correct :
  let net_profit : ℝ := (1500000 - 674992) - 0.2 * (1500000 - 674992)
  let total_loan_payments : ℝ := 23914 * 12 - 74992
  let profit_for_dividends : ℝ := net_profit - total_loan_payments
  let dividend_per_share : ℝ := profit_for_dividends / 1000
  let total_dividends_director : ℝ := dividend_per_share * 550
  total_dividends_director = 246400.0 :=
by
  sorry

end directors_dividends_correct_l641_641648


namespace _l641_641803

noncomputable def z1 (m : ℝ) : ℂ :=
  (m^2 + m + 1) + (m^2 + m - 4) * Complex.i

noncomputable def z2 : ℂ :=
  3 - 2 * Complex.i

lemma condition_for_z1_eq_z2 (m : ℝ) : (z1 m = z2) ↔ (m = 1 ∨ m = -2) :=
by 
  sorry

/-
We are asked to prove that m = 1 is a sufficient but not necessary condition for z1 = z2.
This can be restated in terms of a theorem that asserts the logical equivalence for different values
of m that makes z1 and z2 equal. The statement condition_for_z1_eq_z2 does this by capturing the 
equivalence needed.
-/

end _l641_641803


namespace find_y_values_l641_641530

theorem find_y_values (x : ℝ) (y : ℝ) 
  (h : x^2 + 4 * ((x / (x + 3))^2) = 64) : 
  y = (x + 3)^2 * (x - 2) / (2 * x + 3) → 
  y = 250 / 3 :=
sorry

end find_y_values_l641_641530


namespace max_a_l641_641345

-- Define the conditions
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 1 ≤ x ∧ x ≤ 50 → ¬ ∃ (y : ℤ), line_equation m x = y

def m_range (m a : ℚ) : Prop := (2 : ℚ) / 5 < m ∧ m < a

-- Define the problem statement
theorem max_a (a : ℚ) : (a = 22 / 51) ↔ (∃ m, no_lattice_points m ∧ m_range m a) :=
by 
  sorry

end max_a_l641_641345


namespace negation_statement_l641_641157

variable (x y : ℝ)

theorem negation_statement :
  ¬ (x > 1 ∧ y > 2) ↔ (x ≤ 1 ∨ y ≤ 2) :=
by
  sorry

end negation_statement_l641_641157


namespace third_character_has_2_lines_l641_641506

-- Define the number of lines characters have
variables (x y z : ℕ)

-- The third character has x lines
-- Condition: The second character has 6 more than three times the number of lines the third character has
def second_character_lines : ℕ := 3 * x + 6

-- Condition: The first character has 8 more lines than the second character
def first_character_lines : ℕ := second_character_lines x + 8

-- The first character has 20 lines
def first_character_has_20_lines : Prop := first_character_lines x = 20

-- Prove that the third character has 2 lines
theorem third_character_has_2_lines (h : first_character_has_20_lines x) : x = 2 :=
by
  -- Skipping the proof
  sorry

end third_character_has_2_lines_l641_641506


namespace altitude_product_difference_l641_641873

variable (α : Type*)
variable [LinearOrderedField α]

structure Triangle (α : Type*) :=
  (A B C : α)
  (acute : ∀ (angle : angle), angle < π / 2)  -- This is a simplification, we only care about acute-ness for formalization

structure AltitudeIntersection (α : Type*) :=
  (A B C D E : α)
  (AD BE : α)  -- lines AD is an altitude from A, BE from B
  (H : α)      -- Intersection of altitudes
  (HD : α)
  (HE : α)
  (HD_val : HD = 6)
  (HE_val : HE = 3)


-- Given the above structure, we want to prove:
theorem altitude_product_difference (T: Triangle α) (AI: AltitudeIntersection α) :
  (BD * DC - AE * EC) = 27 :=
begin
  sorry
end

end altitude_product_difference_l641_641873


namespace greatest_product_sum_2000_l641_641215

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641215


namespace geometric_sequence_term_l641_641420

theorem geometric_sequence_term :
  ∃ (a_n : ℕ → ℕ),
    -- common ratio condition
    (∀ n, a_n (n + 1) = 2 * a_n n) ∧
    -- sum of first 4 terms condition
    (a_n 1 + a_n 2 + a_n 3 + a_n 4 = 60) ∧
    -- conclusion: value of the third term
    (a_n 3 = 16) :=
by
  sorry

end geometric_sequence_term_l641_641420


namespace ladder_new_base_distance_after_slip_l641_641691

-- Define the initial conditions and parameters
def initial_ladder_length : ℝ := 30
def initial_base_distance : ℝ := 9
def slip_down_distance : ℝ := 5
def slide_out_distance : ℝ := 3

-- State the theorem.
theorem ladder_new_base_distance_after_slip :
  let initial_height_squared := initial_ladder_length^2 - initial_base_distance^2 in
  let initial_height := real.sqrt initial_height_squared in
  let new_height := initial_height - slip_down_distance in
  let new_base := initial_base_distance + slide_out_distance in
  new_height^2 + new_base^2 = initial_ladder_length^2 → new_base = 12 :=
sorry

end ladder_new_base_distance_after_slip_l641_641691


namespace omega_infinite_elements_l641_641518

theorem omega_infinite_elements {Ω : set (ℝ × ℝ)} (nonempty_Ω : Ω.nonempty)
  (midpoint_cond : ∀ P ∈ Ω, ∃ A B ∈ Ω, P = ((A.1 + B.1)/2, (A.2 + B.2)/2)) :
  Ω.infinite :=
by
  sorry

end omega_infinite_elements_l641_641518


namespace no_two_consecutive_heads_probability_l641_641307

theorem no_two_consecutive_heads_probability : 
  let total_outcomes := 2 ^ 10 in
  let favorable_outcomes := 
    (∑ n in range 6, nat.choose (10 - n - 1) n) in
  (favorable_outcomes / total_outcomes : ℚ) = 9 / 64 :=
by
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 1 + 10 + 36 + 56 + 35 + 6
  have h_total: total_outcomes = 1024 := by norm_num
  have h_favorable: favorable_outcomes = 144 := by norm_num
  have h_fraction: (favorable_outcomes : ℚ) / total_outcomes = 9 / 64 := by norm_num / [total_outcomes, favorable_outcomes]
  exact h_fraction

end no_two_consecutive_heads_probability_l641_641307


namespace fraction_simplification_l641_641381

theorem fraction_simplification (x y : ℚ) (hx : x = 4 / 6) (hy : y = 5 / 8) :
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  rw [hx, hy]
  sorry

end fraction_simplification_l641_641381


namespace cooper_fence_bricks_l641_641739

theorem cooper_fence_bricks :
  ∀ (length height depth walls : ℕ), 
    length = 20 → 
    height = 5 → 
    depth = 2 → 
    walls = 4 → 
    (length * height * depth * walls = 800) :=
by 
  intros length height depth walls h_length h_height h_depth h_walls
  rw [h_length, h_height, h_depth, h_walls]
  have h1 : 20 * 5 = 100 := by norm_num
  have h2 : 100 * 2 = 200 := by norm_num
  have h3 : 200 * 4 = 800 := by norm_num
  rw [h1, h2, h3]
  norm_num

end cooper_fence_bricks_l641_641739


namespace rectangular_reconfiguration_l641_641555

theorem rectangular_reconfiguration (k : ℕ) (n : ℕ) (h₁ : k - 5 > 0) (h₂ : k ≥ 6) (h₃ : k ≤ 9) :
  (k * (k - 5) = n^2) → (n = 6) :=
by {
  sorry  -- proof is omitted
}

end rectangular_reconfiguration_l641_641555


namespace max_product_of_sum_2000_l641_641241

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641241


namespace num_distinguishable_arrangements_l641_641569

/- Definitions for the problem conditions -/
def num_gold_coins : ℕ := 5
def num_silver_coins : ℕ := 3
def total_coins : ℕ := num_gold_coins + num_silver_coins

/- The main theorem statement -/
theorem num_distinguishable_arrangements : 
  (nat.choose total_coins num_gold_coins) * 30 = 1680 :=
by
  -- Placeholder for proof
  sorry

end num_distinguishable_arrangements_l641_641569


namespace sqrt_expression_l641_641641

noncomputable def a : ℝ := 5 - 3 * Real.sqrt 2
noncomputable def b : ℝ := 5 + 3 * Real.sqrt 2

theorem sqrt_expression : 
  Real.sqrt (a^2) + Real.sqrt (b^2) + 2 = 12 :=
by
  sorry

end sqrt_expression_l641_641641


namespace solve_system_of_equations_l641_641950

theorem solve_system_of_equations :
  ∃ (x y z : ℚ),
    4 * x - 6 * y + 2 * z = -14 ∧
    8 * x + 3 * y - z = -15 ∧
    3 * x + z = 7 ∧
    x = 100 / 33 ∧
    y = 146 / 33 ∧
    z = 29 / 11 :=
by
  use 100 / 33, 146 / 33, 29 / 11
  split; norm_num
  split; norm_num
  split; norm_num
  done

end solve_system_of_equations_l641_641950


namespace max_AB_value_l641_641419

section
variables {R : ℝ} (hR : 1 < R ∧ R < 2)

-- Define the ellipse G
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the circle M
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = R^2

-- Define the tangency condition of line l to circle M
def tangent_circle (k m : ℝ) : Prop := m^2 = R^2 * (k^2 + 1)

-- Define the tangency condition of line l to ellipse G
def tangent_ellipse (k m : ℝ) : Prop := m^2 = 4 * k^2 + 1

-- Prove the correct conditions and maximum distance
theorem max_AB_value : 
  ∃ (R : ℝ), 1 < R ∧ R < 2 ∧ R = sqrt 2 ∧ 
  (∀ A B : ℝ × ℝ, ellipse_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧ 
   tangent_circle A.1 A.2 ∧ tangent_ellipse B.1 B.2 → 
   |A.1 - B.1 + A.2 - B.2| ≤ 1) ∧
  (|A.1 - B.1 + A.2 - B.2| = 1) :=
sorry -- Proof not required as per the instructions

end

end max_AB_value_l641_641419


namespace number_in_2019th_field_l641_641973

theorem number_in_2019th_field (f : ℕ → ℕ) (h1 : ∀ n, 0 < f n) (h2 : ∀ n, f n * f (n+1) * f (n+2) = 2018) :
  f 2018 = 1009 := sorry

end number_in_2019th_field_l641_641973


namespace solve_sqrt_eq_l641_641380

theorem solve_sqrt_eq :
  ∃ x : ℝ, sqrt (4 - 2 * x) = 8 ∧ x = -30 :=
by
  exists (-30 : ℝ)
  split
  · -- sqrt (4 - 2 * -30) = 8
    sorry
  · -- x = -30
    rfl

end solve_sqrt_eq_l641_641380


namespace sqrt_inequality_implies_square_inequality_not_square_inequality_implies_sqrt_inequality_l641_641011

theorem sqrt_inequality_implies_square_inequality (a b : ℝ) (h : (sqrt a) - (sqrt b) > 0) : (a^2 - b^2 > 0) :=
sorry

theorem not_square_inequality_implies_sqrt_inequality (a b : ℝ) (h : (a^2 - b^2 > 0)) : ¬ ((sqrt a) - (sqrt b) > 0) :=
sorry

end sqrt_inequality_implies_square_inequality_not_square_inequality_implies_sqrt_inequality_l641_641011


namespace trains_clear_time_l641_641279

-- Definitions based on conditions
def length_train1 : ℕ := 160
def length_train2 : ℕ := 280
def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

-- Conversion factor from km/h to m/s
def kmph_to_mps (s : ℕ) : ℕ := s * 1000 / 3600

-- Computation of relative speed in m/s
def relative_speed_mps : ℕ := kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

-- Total distance to be covered for the trains to clear each other
def total_distance : ℕ := length_train1 + length_train2

-- Time taken for the trains to clear each other
def time_to_clear_each_other : ℕ := total_distance / relative_speed_mps

-- Theorem stating that time taken is 22 seconds
theorem trains_clear_time : time_to_clear_each_other = 22 := by
  sorry

end trains_clear_time_l641_641279


namespace Morse_code_sequences_l641_641048

theorem Morse_code_sequences : 
  let symbols (n : ℕ) := 2^n in
  symbols 1 + symbols 2 + symbols 3 + symbols 4 + symbols 5 = 62 :=
by
  sorry

end Morse_code_sequences_l641_641048


namespace logic_problem_l641_641855

variables (p q : Prop)

theorem logic_problem (hnp : ¬ p) (hpq : ¬ (p ∧ q)) : ¬ (p ∨ q) ∨ (p ∨ q) :=
by 
  sorry

end logic_problem_l641_641855


namespace trig_equation_solution_l641_641271

theorem trig_equation_solution :
  ∃ (k : ℤ) (x : ℝ),
  (5.32 * sin (2 * x) * sin (6 * x) * cos (4 * x) + 1 / 4 * cos (12 * x) = 0) ∧
  (5.331 + tan (2 * x) * tan (5 * x) - sqrt 2 * tan (2 * x) * cos (3 * x) / cos (5 * x) = 0) ∧
  (x = (Real.pi / 8) * (2 * k + 1) ∨ x = (Real.pi / 12) * (6 * k + 1) ∨ x = (Real.pi / 12) * (6 * k - 1)) := sorry

end trig_equation_solution_l641_641271


namespace angle_between_vectors_l641_641835

variables (a b : ℝ^3)

-- Given conditions
def condition1 : Prop := (2 • a - b) • (a + b) = 6
def condition2 : Prop := ‖a‖ = 2
def condition3 : Prop := ‖b‖ = 1

-- The proof statement
theorem angle_between_vectors (a b : ℝ^3) (h1 : condition1 a b) (h2 : condition2 a) (h3 : condition3 b) :
  real.angle a b = 2 * real.pi / 3 :=
sorry

end angle_between_vectors_l641_641835


namespace investment_period_l641_641185

theorem investment_period (P : ℝ) (r1 r2 : ℝ) (diff : ℝ) (t : ℝ) :
  P = 900 ∧ r1 = 0.04 ∧ r2 = 0.045 ∧ (P * r2 * t) - (P * r1 * t) = 31.50 → t = 7 :=
by
  sorry

end investment_period_l641_641185


namespace min_value_expression_l641_641413

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b - a - 2 * b = 0) :
  ∃ p : ℝ, p = (a^2/4 - 2/a + b^2 - 1/b) ∧ p = 7 :=
by sorry

end min_value_expression_l641_641413


namespace determine_phi_find_increasing_intervals_l641_641818

noncomputable def f (x φ : ℝ) : ℝ := sin x * cos x * sin φ + cos x ^ 2 * cos φ + 1 / 2 * cos (π + φ)

theorem determine_phi (h : 0 < φ ∧ φ < π)
                      (h_point : f (π / 3) φ = 1 / 4):
  φ = π / 3 :=
  sorry

noncomputable def g (x : ℝ) : ℝ := 1 / 2 * cos (2 * x)

theorem find_increasing_intervals (a b : ℝ)
                                  (h_interval : a = -π / 4 ∧ b = 2 * π / 3)
                                  (h : 0 < φ ∧ φ < π)
                                  (h_point: f (π / 3) φ = 1 / 4)
                                  (h_phi : φ = π / 3):
  increasing_on g (Set.Icc a 0) ∧ increasing_on g (Set.Icc (π / 2) b) :=
  sorry

end determine_phi_find_increasing_intervals_l641_641818


namespace hyperbola_asymptotes_l641_641020

variables (a b : ℝ)

theorem hyperbola_asymptotes (ha : a > 0) (hb : b > 0) (ecc : sqrt 10 = sqrt (a^2 + b^2) / a):
  ∃ k : ℝ, (k = 3) ∧ (for all x y : ℝ, (y = k * x) ∨ (y = - k * x)) := 
by {
  sorry
}

end hyperbola_asymptotes_l641_641020


namespace angle_PAC_equals_one_third_angle_BAC_l641_641027

-- The main theorem proving the desired angle relation in triangle ABC with given conditions
theorem angle_PAC_equals_one_third_angle_BAC 
    {A B C P : Point} 
    (hABC : ∠ B = 2 * ∠ C)
    (hP_interior : interior_point P)
    (hAP_AB : AP = AB)
    (hPB_PC : PB = PC) 
    : ∠ PAC = ∠ BAC / 3 := 
sorry

end angle_PAC_equals_one_third_angle_BAC_l641_641027


namespace at_least_one_not_less_than_two_l641_641919

theorem at_least_one_not_less_than_two
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x ∈ {a + 1 / b, b + 1 / c, c + 1 / a}, x ≥ 2 :=
by
  -- Proof is omitted
  sorry

end at_least_one_not_less_than_two_l641_641919


namespace parallelogram_area_l641_641362

open Real

def vector_cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

def area_of_parallelogram (u v : ℝ × ℝ × ℝ) : ℝ :=
  vector_magnitude (vector_cross_product u v)

theorem parallelogram_area :
  area_of_parallelogram (4, 2, -3) (2, -6, 5) = 2 * real.sqrt 381 :=
by
  sorry

end parallelogram_area_l641_641362


namespace bottom_left_square_side_eq_l641_641600

-- Definitions for the side lengths of squares
def smallest_square_side := 1
def largest_square_side := x
def next_largest_square_side := x - 1
def third_largest_square_side := x - 2
def bottom_left_square_side := y

-- Equation representing the sum of the sides of the top and bottom of the rectangle
theorem bottom_left_square_side_eq :
  ∀ x y, x + (x - 1) = (x - 2) + (x - 3) + y → y = 4 :=
by
  intros x y h
  sorry

end bottom_left_square_side_eq_l641_641600


namespace range_of_a_for_three_distinct_zeros_l641_641851

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a_for_three_distinct_zeros :
  (∀ (a : ℝ), (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) ↔ -2 < a ∧ a < 2): sorry

end range_of_a_for_three_distinct_zeros_l641_641851


namespace solve_equation_l641_641642

theorem solve_equation : ∃ x : ℝ, (3 * x - 6 = | -8 + 5 |) ∧ x = 3 :=
by
  sorry

end solve_equation_l641_641642


namespace michael_remaining_yards_l641_641725

theorem michael_remaining_yards (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (m y : ℕ)
  (h1 : miles_per_marathon = 50)
  (h2 : yards_per_marathon = 800)
  (h3 : yards_per_mile = 1760)
  (h4 : num_marathons = 5)
  (h5 : y = (yards_per_marathon * num_marathons) % yards_per_mile)
  (h6 : m = miles_per_marathon * num_marathons + (yards_per_marathon * num_marathons) / yards_per_mile) :
  y = 480 :=
sorry

end michael_remaining_yards_l641_641725


namespace constant_term_in_expansion_l641_641589

open Nat

theorem constant_term_in_expansion :
  let expr := (x + (1/x) - 2)^5
  (∀ x : ℝ, x ≠ 0) →
  is_constant_term(expr, -252) := 
sorry

end constant_term_in_expansion_l641_641589


namespace complex_problem_l641_641092

noncomputable theory

variables {a b c x y z : ℂ}

theorem complex_problem 
  (h1: a ≠ 0) (h2: b ≠ 0) (h3: c ≠ 0) (h4: x ≠ 0) (h5: y ≠ 0) (h6: z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : xy + xz + yz = 7)
  (h11 : x + y + z = 4) : 
  xyz = 10 :=
sorry

end complex_problem_l641_641092


namespace acute_triangle_l641_641165

theorem acute_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
                       (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a)
                       (h7 : a^3 + b^3 = c^3) :
                       c^2 < a^2 + b^2 :=
by {
  sorry
}

end acute_triangle_l641_641165


namespace lcm_factorial_15_power_l641_641763

theorem lcm_factorial_15_power :
  nat.lcm (2^11 * 3^6 * 5^3 * 7 * 11 * 13) (2^3 * 3^9 * 5^4 * 7) = 2^11 * 3^9 * 5^4 * 7 * 11 * 13 :=
by
  sorry

end lcm_factorial_15_power_l641_641763


namespace a_beats_b_by_16_meters_l641_641058

theorem a_beats_b_by_16_meters :
  let A_time := 615 -- Seconds A takes to complete the race
  let B_time := 625 -- B's race completion time (A_time + 10 seconds)
  let Distance := 1000 -- Total distance of the race in meters
  let Speed_A := Distance / A_time -- Speed of A
  let Speed_B := Distance / B_time -- Speed of B
  d = Speed_B * 10 -- Calculating the distance by which A beats B
  16 := d
:= sorry

end a_beats_b_by_16_meters_l641_641058


namespace base5_product_l641_641193

theorem base5_product (a b : Nat) (h₁ : a = 132) (h₂ : b = 22) : 
  let prod_base5 := 4004 in 
  base5_product a b = prod_base5 :=
begin
  sorry
end

end base5_product_l641_641193


namespace arrange_cards_correct_l641_641503

noncomputable def card_arrangement_possible (x y z : ℕ) : Prop :=
  x + y + z = 100 ∧ x ≤ 50 ∧ y ≤ 50 ∧ z ≤ 50 → 
  ∃ (sequence : List ℕ), 
    (∀ i, i ∈ sequence → i = 1 ∨ i = 2 ∨ i = 3) ∧
    (∀ i, i < sequence.length - 1 → ¬ (sequence.get i = sequence.get (i+1) 
      ∧ sequence.get i = sequence.get (i + 1))) ∧
    (∀ i, i < sequence.length - 2 → 
      ¬ ((sequence.get i = 1 ∧ sequence.get (i+1) = 2 ∧ sequence.get (i+2) = 3) ∨ 
         (sequence.get i = 3 ∧ sequence.get (i+1) = 2 ∧ sequence.get (i+2) = 1)))

theorem arrange_cards_correct (x y z : ℕ) (h : x + y + z = 100) (h₁ : x ≤ 50) (h₂ : y ≤ 50) (h₃ : z ≤ 50) : 
  card_arrangement_possible x y z :=
sorry

end arrange_cards_correct_l641_641503


namespace last_three_digits_of_7_pow_103_l641_641370

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l641_641370


namespace find_common_difference_l641_641069

noncomputable def common_difference (d : ℝ) : Prop :=
  let a : ℕ → ℝ := λ n, 4 + (n - 2) * d in
  (1 + a 3) * (4 + a 10) = (a 6) ^ 2

theorem find_common_difference :
  ∃ d : ℝ, 4 + (2 - 2) * d = 4 ∧
           common_difference d :=
by
  use 3
  sorry

end find_common_difference_l641_641069


namespace intersection_polar_coords_l641_641878

noncomputable def curve_M_parametric (t : ℝ) (ht : t > 0) : ℝ × ℝ :=
  (2 * Real.sqrt 3 / (Real.sqrt 3 - t), (2 * Real.sqrt 3 * t) / (Real.sqrt 3 - t))

def M_standard_eq (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 2) ∧ (x > 2 ∨ x < 0)

def C_polar_eq (θ : ℝ) : ℝ := 4 * Real.cos θ

def C_cartesian_eq (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 = 0

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem intersection_polar_coords :
  ∃ (ρ θ : ℝ), (ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) ∧
    (M_standard_eq (polar_to_cartesian ρ θ).1 (polar_to_cartesian ρ θ).2) ∧
    (C_cartesian_eq (polar_to_cartesian ρ θ).1 (polar_to_cartesian ρ θ).2) ∧
    ρ = 2 * Real.sqrt 3 ∧ θ = Real.pi / 6 :=
sorry

end intersection_polar_coords_l641_641878


namespace exists_abs_f_ge_two_l641_641088

open Real

theorem exists_abs_f_ge_two (a b : ℝ) : 
  ∃ x_0 ∈ Icc 1 9, abs (a * x_0 + b + 9 / x_0) ≥ 2 := 
by sorry

end exists_abs_f_ge_two_l641_641088


namespace subsets_count_l641_641995

variable (T : Finset ℕ) (w x y z v : ℕ)

theorem subsets_count {T : Finset ℕ} 
  (hT : T = {w, x, y, z, v}) :
  ( ∃ (A B : Finset ℕ), (A ∪ B = T) ∧ (A ∩ B).card = 3 ∧ A ≠ B ) ∧ 
  (count {A B : Finset ℕ | A ∪ B = T ∧ (A ∩ B).card = 3 ∧ A ≠ B} ) = 20 :=
by sorry

end subsets_count_l641_641995


namespace average_percentage_decrease_l641_641719

theorem average_percentage_decrease :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 100 * (1 - x)^2 = 81 ∧ x = 0.1 :=
by
  sorry

end average_percentage_decrease_l641_641719


namespace distinct_right_triangles_with_area_twice_perimeter_eq_one_l641_641004

theorem distinct_right_triangles_with_area_twice_perimeter_eq_one :
  (∀ (a b c : ℕ), (c = Int.sqrt (a^2 + b^2)) →
    (a * b / 2 = 2 * (a + b + c)) →
    ∃! (a b c : ℕ), (c = Int.sqrt (a^2 + b^2)) ∧ (a * b / 2 = 2 * (a + b + c))).length = 1 :=
sorry

end distinct_right_triangles_with_area_twice_perimeter_eq_one_l641_641004


namespace ways_to_choose_cooks_and_cleaners_l641_641382

theorem ways_to_choose_cooks_and_cleaners : 
  ∀ (friends : Finset ℕ) (h : friends.card = 5), 
    ∃! (cooks : Finset ℕ), cooks.card = 3 ∧ friends \ cooks = 2 := by
  sorry

end ways_to_choose_cooks_and_cleaners_l641_641382


namespace sum_of_readings_ammeters_l641_641770

variables (I1 I2 I3 I4 I5 : ℝ)

noncomputable def sum_of_ammeters (I1 I2 I3 I4 I5 : ℝ) : ℝ :=
  I1 + I2 + I3 + I4 + I5

theorem sum_of_readings_ammeters :
  I1 = 2 ∧ I2 = I1 ∧ I3 = 2 * I1 ∧ I5 = I3 + I1 ∧ I4 = (5 / 3) * I5 →
  sum_of_ammeters I1 I2 I3 I4 I5 = 24 :=
by
  sorry

end sum_of_readings_ammeters_l641_641770


namespace initial_overs_l641_641883

theorem initial_overs (x : ℝ) (r1 : ℝ) (r2 : ℝ) (target : ℝ) (overs_remaining : ℝ) :
  r1 = 3.2 ∧ overs_remaining = 22 ∧ r2 = 11.363636363636363 ∧ target = 282 ∧
  (r1 * x + r2 * overs_remaining = target) → x = 10 :=
by
  intro h
  obtain ⟨hr1, ho, hr2, ht, heq⟩ := h
  sorry

end initial_overs_l641_641883


namespace smallest_positive_l641_641743

-- Define the given expressions
def exprA := 14 - 4 * Real.sqrt 15
def exprB := 4 * Real.sqrt 15 - 14
def exprC := 22 - 6 * Real.sqrt 17
def exprD := 64 - 12 * Real.sqrt 34
def exprE := 12 * Real.sqrt 34 - 64

-- Define the proof problem statement
theorem smallest_positive :
  exprB = 4 * Real.sqrt 15 - 14 → 
  (∀ x ∈ {exprA, exprB, exprC, exprD, exprE}, x < 0 ∨ (x > 0 → exprB ≤ x)) :=
by
  intros
  sorry

end smallest_positive_l641_641743


namespace triangle_sides_external_tangent_l641_641628

theorem triangle_sides_external_tangent (R r : ℝ) (h : R > r) :
  ∃ (AB BC AC : ℝ),
    AB = 2 * Real.sqrt (R * r) ∧
    AC = 2 * r * Real.sqrt (R / (R + r)) ∧
    BC = 2 * R * Real.sqrt (r / (R + r)) :=
by
  sorry

end triangle_sides_external_tangent_l641_641628


namespace vector_parallel_l641_641834

theorem vector_parallel {x : ℝ} (h : (4 / x) = (-2 / 5)) : x = -10 :=
  by
  sorry

end vector_parallel_l641_641834


namespace problem1_problem2_l641_641734

variable (x y : ℝ)

theorem problem1 :
  x^4 * x^3 * x - (x^4)^2 + (-2 * x)^3 * x^5 = -8 * x^8 :=
by sorry

theorem problem2 :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end problem1_problem2_l641_641734


namespace smallest_possible_degree_of_polynomial_l641_641954

theorem smallest_possible_degree_of_polynomial :
  ∀ (p : Polynomial ℚ),
    (p ≠ 0) ∧
    (p.eval (3 - 2 * Real.sqrt 2) = 0) ∧
    (p.eval (3 + 2 * Real.sqrt 2) = 0) ∧
    (p.eval (-3 - 2 * Real.sqrt 2) = 0) ∧
    (p.eval (-3 + 2 * Real.sqrt 2) = 0) ∧
    (p.eval (1 + Real.sqrt 7) = 0) ∧
    (p.eval (1 - Real.sqrt 7) = 0) ∧
    (p.eval (4 + Real.sqrt 11) = 0) ∧
    (p.eval (4 - Real.sqrt 11) = 0) →
    ∃ n, degree p = n ∧ n = 8 :=
sorry

end smallest_possible_degree_of_polynomial_l641_641954


namespace profit_percentage_is_20_l641_641134

def Robi_contribution : ℝ := 4000
def Rudy_extra_ratio : ℝ := 1/4
def Rudy_contribution : ℝ := Robi_contribution * (1 + Rudy_extra_ratio)
def each_person_profit : ℝ := 900
def total_profit : ℝ := 2 * each_person_profit
def total_contribution : ℝ := Robi_contribution + Rudy_contribution
def profit_percentage : ℝ := (total_profit / total_contribution) * 100

theorem profit_percentage_is_20 :
  profit_percentage = 20 := by
  sorry

end profit_percentage_is_20_l641_641134


namespace earrings_ratio_l641_641334

theorem earrings_ratio :
  ∀ (total_pairs : ℕ) (given_pairs : ℕ) (total_earrings : ℕ) (given_earrings : ℕ),
    total_pairs = 12 →
    given_pairs = total_pairs / 2 →
    total_earrings = total_pairs * 2 →
    given_earrings = total_earrings / 2 →
    total_earrings = 36 →
    given_earrings = 12 →
    (total_earrings / given_earrings = 3) :=
by
  sorry

end earrings_ratio_l641_641334


namespace tangent_line_at_one_inequality_for_x_gt_zero_l641_641533

def f (x : ℝ) : ℝ := Real.exp x - x^2
def g (x : ℝ) : ℝ := Real.exp x + (2 - Real.exp 1) * x - 1

theorem tangent_line_at_one :
  ∃ m b, m = Real.exp 1 - 2 ∧ b = Real.exp 1 - 1 ∧ ∀ x, f x = m * x + b ↔ x = 1 := 
sorry

theorem inequality_for_x_gt_zero (x : ℝ) (hx : 0 < x) :
  (Real.exp x + (2 - Real.exp 1) * x - 1) / x ≥ Real.log x + 1 := 
sorry

end tangent_line_at_one_inequality_for_x_gt_zero_l641_641533


namespace arithmetic_seq_sum_specific_max_arithmetic_seq_sum_l641_641404

variable {α : Type*} [Field α]
variables (a_1 : α) (d : α) (n : α)

def arithmetic_seq_sum (a₁ d : α) (n : α) : α :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

theorem arithmetic_seq_sum_specific
  (h₁ : a₁ = 31) 
  (h₂ : arithmetic_seq_sum 31 d 10 = arithmetic_seq_sum 31 d 22)
  : arithmetic_seq_sum 31 2 n = n^2 + 30 * n :=
sorry

theorem max_arithmetic_seq_sum
  (h₁ : a₁ = 31) 
  (h₂ : arithmetic_seq_sum 31 d 10 = arithmetic_seq_sum 31 d 22)
  : ∃ n : ℕ, n = 16 ∧ arithmetic_seq_sum 31 2 n = 736 :=
sorry

end arithmetic_seq_sum_specific_max_arithmetic_seq_sum_l641_641404


namespace cm_co_eq_cn_cb_l641_641515

variable (AB CD AE CO DE BC CM CO CN CB : ℝ)
variable (O M N : Point)

-- Given conditions
axiom Circle (O : Point) (r : ℝ)
axiom Diameter (A B : Point) : (dist A B = 2 * r)
axiom Center (A B : Point) (O : Point) : (dist A O = dist B O)
axiom Perpendicular (x y : Line) : ∃ θ, θ = 90
axiom Intersects (E F : Line) (X : Point) : Point

-- Statement to prove
theorem cm_co_eq_cn_cb :
   (Intersects AE CO M) →
   (Intersects DE BC N) →
   (Perpendicular AB CD) →
   (Diameter A B) →
   (dist C M / dist C O = dist C N / dist C B) :=
by sorry

end cm_co_eq_cn_cb_l641_641515


namespace greatest_product_sum_2000_l641_641219

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641219


namespace minimum_trees_planted_l641_641627

theorem minimum_trees_planted :
  ∀ (trees_per_side : ℕ) (triangle_sides : ℕ), 
  triangle_sides = 3 → trees_per_side = 5 → 
  ∃ (minimum_trees : ℕ), minimum_trees = 12 :=
by
  intros trees_per_side triangle_sides h_triangle_sides h_trees_per_side
  use 12
  sorry

end minimum_trees_planted_l641_641627


namespace cooper_fence_bricks_l641_641740

theorem cooper_fence_bricks :
  ∀ (length height depth walls : ℕ), 
    length = 20 → 
    height = 5 → 
    depth = 2 → 
    walls = 4 → 
    (length * height * depth * walls = 800) :=
by 
  intros length height depth walls h_length h_height h_depth h_walls
  rw [h_length, h_height, h_depth, h_walls]
  have h1 : 20 * 5 = 100 := by norm_num
  have h2 : 100 * 2 = 200 := by norm_num
  have h3 : 200 * 4 = 800 := by norm_num
  rw [h1, h2, h3]
  norm_num

end cooper_fence_bricks_l641_641740


namespace polynomial_sum_divisible_l641_641101

theorem polynomial_sum_divisible {P : ℤ[X]} (n : ℤ) (hn : 0 < n): 
  n ∣ ∑ k in Finset.range (n^2), (P.eval k) := 
sorry

end polynomial_sum_divisible_l641_641101


namespace parallelogram_max_lambda_mu_l641_641489

theorem parallelogram_max_lambda_mu
  (A B C D P : Type)
  [inner_product_space ℝ A B C D P]
  (AB AD : A)
  (AP : A)
  (λ μ : ℝ)
  (h1 : ∠BAD = 60°)
  (h2 : ∥AB∥ = 1)
  (h3 : ∥AD∥ = √2)
  (h4 : ∥AP∥ = √2 / 2)
  (h5 : AP = λ • AB + μ • AD) :
  (λ + √2 * μ) ≤ (√6 / 3) := sorry

end parallelogram_max_lambda_mu_l641_641489


namespace number_of_terms_in_arithmetic_sequence_l641_641838

-- Definitions derived directly from the conditions
def first_term : ℕ := 2
def common_difference : ℕ := 4
def last_term : ℕ := 2010

-- Lean statement for the proof problem
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 503 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l641_641838


namespace seq_product_eq_l641_641729

theorem seq_product_eq :
  (∏ n in (finset.range 55), (n + 1) / (n + 5)) = (6 : ℚ) / 195006 := by 
  sorry

end seq_product_eq_l641_641729


namespace square_2007th_position_l641_641967

noncomputable def square_position (n : ℕ) : String :=
  if n % 2 = 1 then "DCBA" else "ABCD"

theorem square_2007th_position : square_position 2007 = "DCBA" :=
by
  simp [square_position]
  sorry

end square_2007th_position_l641_641967


namespace shaded_area_correct_l641_641497

-- Define the radii of the circles
def r_large := 8
def r_small := 4

-- Define the areas of the circles
def area_large := π * (r_large ^ 2)
def area_small := π * (r_small ^ 2)

-- Define the shaded area
def area_shaded := area_large - 2 * area_small

-- Theorem stating the result
theorem shaded_area_correct :
  area_shaded = 32 * π := by
  sorry

end shaded_area_correct_l641_641497


namespace xyz_inequality_l641_641786

theorem xyz_inequality : ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by
  intros
  sorry

end xyz_inequality_l641_641786


namespace sum_squares_induction_l641_641632

theorem sum_squares_induction (n : ℕ) (h : n ≥ 1) :
  ∑ i in (finset.range (n + 1)).filter (λ i, i > 0), i^2 = (n^4 + n^2) / 2 :=
by sorry

end sum_squares_induction_l641_641632


namespace hyperbola_eccentricity_l641_641856

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = 2 * a) :
  sqrt (1 + (b^2 / a^2)) = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l641_641856


namespace MorseCodeDistinctSymbols_l641_641031

theorem MorseCodeDistinctSymbols:
  (1.sequence (λ _, bool).length = {1, 2, 3, 4, 5}).card = 62 :=
by
  sorry

end MorseCodeDistinctSymbols_l641_641031


namespace triangle_angle_C_triangle_max_area_l641_641077

noncomputable def cos (θ : Real) : Real := sorry
noncomputable def sin (θ : Real) : Real := sorry

theorem triangle_angle_C (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) : C = (2 * Real.pi) / 3 :=
sorry

theorem triangle_max_area (a b c : Real) (A B C : Real) (h1: 0 < A ∧ A < Real.pi)
  (h2: 0 < B ∧ B < Real.pi) (h3: 0 < C ∧ C < Real.pi)
  (h4: (2 * a + b) * cos C + c * cos B = 0) (hc : c = 6)
  (hC : C = (2 * Real.pi) / 3) : 
  ∃ (S : Real), S = 3 * Real.sqrt 3 := 
sorry

end triangle_angle_C_triangle_max_area_l641_641077


namespace no_two_heads_consecutive_probability_l641_641296

noncomputable def probability_no_two_heads_consecutive (total_flips : ℕ) : ℚ :=
  if total_flips = 10 then 144 / 1024 else 0

theorem no_two_heads_consecutive_probability :
  probability_no_two_heads_consecutive 10 = 9 / 64 :=
by
  unfold probability_no_two_heads_consecutive
  rw [if_pos rfl]
  norm_num
  sorry

end no_two_heads_consecutive_probability_l641_641296


namespace max_product_of_two_integers_sum_2000_l641_641253

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641253


namespace exists_five_consecutive_divisible_by_2014_l641_641760

theorem exists_five_consecutive_divisible_by_2014 :
  ∃ (a b c d e : ℕ), 53 = a ∧ 54 = b ∧ 55 = c ∧ 56 = d ∧ 57 = e ∧ 100 > a ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ 2014 ∣ (a * b * c * d * e) :=
by 
  sorry

end exists_five_consecutive_divisible_by_2014_l641_641760


namespace ratio_hooper_bay_to_other_harbors_l641_641456

-- Definitions based on conditions
def other_harbors_lobster : ℕ := 80
def total_lobster : ℕ := 480
def combined_other_harbors_lobster := 2 * other_harbors_lobster
def hooper_bay_lobster := total_lobster - combined_other_harbors_lobster

-- The theorem to prove
theorem ratio_hooper_bay_to_other_harbors : hooper_bay_lobster / combined_other_harbors_lobster = 2 :=
by
  sorry

end ratio_hooper_bay_to_other_harbors_l641_641456


namespace Morse_code_distinct_symbols_l641_641054

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l641_641054


namespace greatest_four_digit_divisible_l641_641703

theorem greatest_four_digit_divisible (p : ℕ) :
  (1000 ≤ p ∧ p < 10000) ∧ 
  (63 ∣ p) ∧ 
  (63 ∣ reverse_digits p) ∧ 
  (11 ∣ p) → 
  p = 9779 :=
by
  sorry

-- Helper function to reverse the digits of a natural number
noncomputable def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10 in
  digits.reverse.foldl (λ acc d, acc * 10 + d) 0


end greatest_four_digit_divisible_l641_641703


namespace find_m_l641_641830

def U : Set Nat := {1, 2, 3}
def A (m : Nat) : Set Nat := {1, m}
def complement (s t : Set Nat) : Set Nat := {x | x ∈ s ∧ x ∉ t}

theorem find_m (m : Nat) (h1 : complement U (A m) = {2}) : m = 3 :=
by
  sorry

end find_m_l641_641830


namespace max_product_l641_641229

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641229


namespace quadratic_eq_has_two_distinct_real_roots_l641_641166

-- We start by defining the quadratic equation and its properties.
def quadratic_eq (x : ℝ) : Prop := (x - 1) * (x + 5) = 3 * x + 1

-- The discriminant for a general quadratic equation and its computation.
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- State the problem.
theorem quadratic_eq_has_two_distinct_real_roots : 
  (∃ a b c : ℝ, quadratic_eq x ∧ 
   (a = 1) ∧ (b = 1) ∧ (c = -6) ∧ discriminant a b c > 0) :=
begin
  sorry
end

end quadratic_eq_has_two_distinct_real_roots_l641_641166


namespace lori_earnings_equation_l641_641538

theorem lori_earnings_equation : 
  ∀ (white_cars red_cars : ℕ) 
    (white_rent_cost red_rent_cost minutes_per_hour rental_hours : ℕ), 
  white_cars = 2 →
  red_cars = 3 →
  white_rent_cost = 2 →
  red_rent_cost = 3 →
  minutes_per_hour = 60 →
  rental_hours = 3 →
  (white_cars * white_rent_cost + red_cars * red_rent_cost) * rental_hours * minutes_per_hour = 2340 := 
by
  intros white_cars red_cars white_rent_cost red_rent_cost minutes_per_hour rental_hours 
         h_white_cars h_red_cars h_white_rent_cost h_red_rent_cost h_minutes_per_hour h_rental_hours
  rw [h_white_cars, h_red_cars, h_white_rent_cost, h_red_rent_cost, h_minutes_per_hour, h_rental_hours]
  calc
    (2 * 2 + 3 * 3) * 3 * 60 = (4 + 9) * 3 * 60 : by rw [mul_add, mul_comm 3]
    ... = 13 * 3 * 60 : by rw [add_mul]
    ... = 39 * 60 : by rw [mul_assoc]
    ... = 2340  : by norm_num
  done

end lori_earnings_equation_l641_641538


namespace symmetric_point_origin_l641_641595

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end symmetric_point_origin_l641_641595


namespace arithmetic_sequence_terms_count_l641_641839

theorem arithmetic_sequence_terms_count (a d l : Int) (h1 : a = 20) (h2 : d = -3) (h3 : l = -5) :
  ∃ n : Int, l = a + (n - 1) * d ∧ n = 8 :=
by
  sorry

end arithmetic_sequence_terms_count_l641_641839


namespace min_value_of_a_l641_641991

theorem min_value_of_a (r s t : ℕ) (h1 : r > 0) (h2 : s > 0) (h3 : t > 0)
  (h4 : r * s * t = 2310) (h5 : r + s + t = a) : 
  a = 390 → True :=
by { 
  intros, 
  sorry 
}

end min_value_of_a_l641_641991


namespace unique_g_l641_641908

noncomputable def T := {x : ℝ // x ≠ 0}

def g (T → ℝ) := sorry

theorem unique_g:
  (∃! g : T → ℝ, (
    g 2 = 1 ∧ 
    (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0 → g (1 / (x + y)) = g (1 / x) + g (1 / y)) ∧ 
    (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x + y ≠ 0 → (x + y) * g (x + y) = 2 * x * y * g x * g y)
  )) :=
sorry

end unique_g_l641_641908


namespace no_two_consecutive_heads_l641_641292

/-- The probability that two heads never appear consecutively when a coin is tossed 10 times is 9/64. -/
theorem no_two_consecutive_heads (tosses : ℕ) (prob: ℚ):
  tosses = 10 → 
  (prob = 9 / 64) → 
  prob = probability_no_consecutive_heads tosses := sorry

end no_two_consecutive_heads_l641_641292


namespace smallest_value_of_a_l641_641988

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end smallest_value_of_a_l641_641988


namespace fraction_irrducible_a_fraction_irrducible_b_fraction_irrducible_c_l641_641129

open Nat

theorem fraction_irrducible_a (n : ℕ) : gcd (2 * n + 13) (n + 7) = 1 := 
  sorry

theorem fraction_irrducible_b (n : ℕ) : gcd (2 * n^2 - 1) (n + 1) = 1 := 
  sorry

theorem fraction_irrducible_c (n : ℕ) : gcd (n^2 - n + 1) (n^2 + 1) = 1 :=
  sorry

end fraction_irrducible_a_fraction_irrducible_b_fraction_irrducible_c_l641_641129


namespace minimum_value_fraction_l641_641804

open Real

theorem minimum_value_fraction (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  ∃ x, x = 2 * (sqrt 2 - 1) ∧ (∀ a b in set_of ((λ x, x < 0) : ℝ → Prop), 
  ∀ minValue, minValue == (λ a b : ℝ, if (a < 0 ∧ b < 0) then (a / (a + 2 * b) + b / (a + b)) else 0) 
    a b + (λ a b : ℝ, if (a < 0 ∧ b < 0) then -2 * (sqrt 2 - 1) else 0) a b :=
    2 * (sqrt 2 - 1) end)
:= sorry

end minimum_value_fraction_l641_641804


namespace complex_div_simplification_l641_641688

theorem complex_div_simplification : (3 : ℂ) / (1 - complex.i)^2 = (3 / 2) * complex.i :=
by
  sorry

end complex_div_simplification_l641_641688


namespace justify_misha_decision_l641_641650

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l641_641650


namespace animal_legs_l641_641556

theorem animal_legs (total_animals : ℕ) (cats_percentage dogs_percentage birds_percentage insects_percentage : ℝ)
    (cat_legs dog_legs bird_legs insect_legs : ℕ) 
    (h1 : total_animals = 300) (h2 : cats_percentage = 0.50) 
    (h3 : dogs_percentage = 0.30) (h4 : birds_percentage = 0.10) 
    (h5 : insects_percentage = 0.10) (h6 : cat_legs = 4) 
    (h7 : dog_legs = 4) (h8 : bird_legs = 2) 
    (h9 : insect_legs = 6) 
    : 
    let cats := cats_percentage * total_animals 
    let dogs := dogs_percentage * total_animals 
    let birds := birds_percentage * total_animals 
    let insects := insects_percentage * total_animals 
    in 
    cats * cat_legs = 600 
    ∧ dogs * dog_legs = 360 
    ∧ birds * bird_legs = 60 
    ∧ insects * insect_legs = 180 := 
by {
 sorry
}

end animal_legs_l641_641556


namespace people_in_room_eq_33_l641_641178

variable (people chairs : ℕ)

def chairs_empty := 5
def chairs_total := 5 * 5
def chairs_occupied := (4 * chairs_total) / 5
def people_seated := 3 * people / 5

theorem people_in_room_eq_33 : 
    (people_seated = chairs_occupied ∧ chairs_total - chairs_occupied = chairs_empty)
    → people = 33 :=
by
  sorry

end people_in_room_eq_33_l641_641178


namespace positive_root_less_than_inv_factorial_l641_641566

theorem positive_root_less_than_inv_factorial {n : ℕ} {x : ℝ} (h_eqn : x * (x + 1) * (x + 2) * ... * (x + n) = 1) (h_pos : x > 0) : 
  x < 1 / (nat.factorial n) := sorry

end positive_root_less_than_inv_factorial_l641_641566


namespace no_two_consecutive_heads_probability_l641_641310

theorem no_two_consecutive_heads_probability : 
  let total_outcomes := 2 ^ 10 in
  let favorable_outcomes := 
    (∑ n in range 6, nat.choose (10 - n - 1) n) in
  (favorable_outcomes / total_outcomes : ℚ) = 9 / 64 :=
by
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 1 + 10 + 36 + 56 + 35 + 6
  have h_total: total_outcomes = 1024 := by norm_num
  have h_favorable: favorable_outcomes = 144 := by norm_num
  have h_fraction: (favorable_outcomes : ℚ) / total_outcomes = 9 / 64 := by norm_num / [total_outcomes, favorable_outcomes]
  exact h_fraction

end no_two_consecutive_heads_probability_l641_641310


namespace series_converges_l641_641633

theorem series_converges (u : ℕ → ℝ) (h : ∀ n, u n = n / (3 : ℝ)^n) :
  ∃ l, 0 ≤ l ∧ l < 1 ∧ ∑' n, u n = l := by
  sorry

end series_converges_l641_641633


namespace prime_and_multiple_of_5_probability_l641_641558

def is_prime (n : ℕ) : Prop :=
  nat.prime n

def is_multiple_of_5 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 5 * k

def card_numbers : finset ℕ := finset.range 101   -- numbers 1 to 100

theorem prime_and_multiple_of_5_probability :
  let eligible_cards := card_numbers.filter (λ n, is_prime n ∧ is_multiple_of_5 n) in
  eligible_cards.card.to_real / card_numbers.card.to_real = 1 / 100 :=
by 
  sorry

end prime_and_multiple_of_5_probability_l641_641558


namespace inequality_abc_l641_641940

theorem inequality_abc (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) :=
by
  -- Proof goes here
  sorry

end inequality_abc_l641_641940


namespace smallest_c1_in_arithmetic_sequence_l641_641167

theorem smallest_c1_in_arithmetic_sequence (S3 S7 : ℕ) (S3_natural : S3 > 0) (S7_natural : S7 > 0)
    (c1_geq_one_third : ∀ d : ℚ, (c1 : ℚ) = (7*S3 - S7) / 14 → c1 ≥ 1/3) : 
    ∃ c1 : ℚ, c1 = 5/14 ∧ c1 ≥ 1/3 := 
by 
  sorry

end smallest_c1_in_arithmetic_sequence_l641_641167


namespace symmetric_point_origin_l641_641596

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end symmetric_point_origin_l641_641596


namespace hyperbola_eccentricity_eq_two_l641_641963

theorem hyperbola_eccentricity_eq_two :
  (∀ x y : ℝ, ((x^2 / 2) - (y^2 / 6) = 1) → 
    let a_squared := 2
    let b_squared := 6
    let a := Real.sqrt a_squared
    let b := Real.sqrt b_squared
    let e := Real.sqrt (1 + b_squared / a_squared)
    e = 2) := 
sorry

end hyperbola_eccentricity_eq_two_l641_641963


namespace chinese_money_plant_sales_l641_641316

/-- 
Consider a scenario where a plant supplier sells 20 pieces of orchids for $50 each 
and some pieces of potted Chinese money plant for $25 each. He paid his two workers $40 each 
and bought new pots worth $150. The plant supplier had $1145 left from his earnings. 
Prove that the number of pieces of potted Chinese money plants sold by the supplier is 15.
-/
theorem chinese_money_plant_sales (earnings_orchids earnings_per_orchid: ℤ)
  (num_orchids: ℤ)
  (earnings_plants earnings_per_plant: ℤ)
  (worker_wage num_workers: ℤ)
  (new_pots_cost remaining_money: ℤ)
  (earnings: ℤ)
  (P : earnings_orchids = num_orchids * earnings_per_orchid)
  (Q : earnings = earnings_orchids + earnings_plants)
  (R : earnings - (worker_wage * num_workers + new_pots_cost) = remaining_money)
  (conditions: earnings_per_orchid = 50 ∧ num_orchids = 20 ∧ earnings_per_plant = 25 ∧ worker_wage = 40 ∧ num_workers = 2 ∧ new_pots_cost = 150 ∧ remaining_money = 1145):
  earnings_plants / earnings_per_plant = 15 := 
by
  sorry

end chinese_money_plant_sales_l641_641316


namespace f_f_one_l641_641425

-- Define the function f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Given conditions
def f (x : ℝ) : ℝ :=
  if x > 0 then 2^(x - 1) - 3 else -(2^(-x - 1) - 3)

-- The main statement to prove
theorem f_f_one : 
  is_odd_function f →
  (∀ x, x > 0 → f x = 2^(x - 1) - 3) →
  f (f 1) = 1 := 
by
  intros h_odd h_condition
  -- Proof goes here
  sorry

end f_f_one_l641_641425


namespace unique_maximum_point_in_interval_l641_641776

noncomputable def f (x : ℝ) := Real.sin x + x - Real.exp x

theorem unique_maximum_point_in_interval :
  ∃ x₀ ∈ set.Icc (0 : ℝ) real.pi, 
    (∀ x ∈ set.Icc (0 : ℝ) x₀, f x ≤ f x₀) ∧ 
    (∀ x ∈ set.Icc x₀ real.pi, f x₀ ≥ f x) ∧ 
    ∃ x_min ∈ ({(0 : ℝ), real.pi} : set ℝ), f x_min ≤ f x forAll (x ∈ set.Icc (0 : ℝ) real.pi, x ≠ x_min) :=
by sorry

end unique_maximum_point_in_interval_l641_641776


namespace b_c_values_l641_641435

theorem b_c_values (b c : ℝ) :
  (log 3 (2 * x^2 + b * x + c) / (x^2 + 1)) ∈ (set.Icc 0 1) →
  (b = 2 ∧ c = 2) ∨ (b = -2 ∧ c = 2) :=
sorry

end b_c_values_l641_641435


namespace system_of_equations_solution_l641_641616

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + y = -1) 
  (h2 : x + z = 0) 
  (h3 : y + z = 1) : 
  x = -1 ∧ y = 0 ∧ z = 1 :=
by
  sorry

end system_of_equations_solution_l641_641616


namespace calculate_expression_l641_641259

theorem calculate_expression : 
  let a := 0.82
  let b := 0.1
  a^3 - b^3 / (a^2 + 0.082 + b^2) = 0.7201 := sorry

end calculate_expression_l641_641259


namespace goods_train_speed_correct_l641_641315

open Real

-- Define all constants based on the conditions
def man_train_speed_kmph : ℝ := 60
def goods_train_length_m : ℝ := 280
def passing_time_s : ℝ := 9

-- Define the relative speed of the goods train in kmph
def relative_speed_kmph := (goods_train_length_m / passing_time_s) * 3.6

-- Define the speed of the goods train
def goods_train_speed_kmph := relative_speed_kmph - man_train_speed_kmph

-- The theorem to be proven
theorem goods_train_speed_correct : goods_train_speed_kmph = 52 := by
  sorry

end goods_train_speed_correct_l641_641315


namespace last_person_in_circle_is_Dan_l641_641862

-- Define the participants and their positions in the circle
inductive Friend
| Arn
| Bob
| Cyd
| Dan
| Eve
| Fon
| Gin
| Hal
deriving Inhabited

-- Define the elimination rule
def leavesCircle (n : Nat) : Bool :=
  (n % 3 = 0) ∨ (n % 6 = 0) ∨ ('3' ∈ n.toString.toList ∧ '6' ∈ n.toString.toList)

noncomputable def lastPersonStanding (participants : List Friend) : Friend :=
  sorry

-- The list of participants in initial positions 1 through 8 (Arn to Hal)
def participants : List Friend :=
  [Friend.Arn, Friend.Bob, Friend.Cyd, Friend.Dan, Friend.Eve, Friend.Fon, Friend.Gin, Friend.Hal]

-- Statement of the proof problem
theorem last_person_in_circle_is_Dan : lastPersonStanding participants = Friend.Dan :=
sorry

end last_person_in_circle_is_Dan_l641_641862


namespace least_sum_of_exponents_l641_641468

theorem least_sum_of_exponents {n : ℕ} (h : n = 520) (h_exp : ∃ (a b : ℕ), 2^a + 2^b = n ∧ a ≠ b ∧ a = 9 ∧ b = 3) : 
    (∃ (s : ℕ), s = 9 + 3) :=
by
  sorry

end least_sum_of_exponents_l641_641468


namespace radius_of_circumscribed_sphere_eq_a_l641_641164

-- Assume a to be a real number representing the side length of the base and height of the hexagonal pyramid
variables (a : ℝ)

-- Representing the base as a regular hexagon and the pyramid as having equal side length and height
def regular_hexagonal_pyramid (a : ℝ) : Type := {b : ℝ // b = a}

-- The radius of the circumscribed sphere to a given regular hexagonal pyramid
def radius_of_circumscribed_sphere (a : ℝ) : ℝ := a

-- Theorem stating that the radius of the sphere circumscribed around a regular hexagonal pyramid 
-- with side length and height both equal to a is a
theorem radius_of_circumscribed_sphere_eq_a (a : ℝ) :
  radius_of_circumscribed_sphere a = a :=
by {
  sorry
}

end radius_of_circumscribed_sphere_eq_a_l641_641164


namespace chess_piece_bound_l641_641547

theorem chess_piece_bound (m : ℕ) (k : ℕ) (a : ℕ → ℕ) (h_sum : (∑ i in finset.range k, a i) = m) :
  (∑ i in finset.range k, (a i * m)) ≤ m^2 →
  ∑ i in finset.range k, (if (a i * m) ≥ (10 * m) then 1 else 0) ≤ (m / 10) := 
sorry

end chess_piece_bound_l641_641547


namespace orient_edges_of_connected_graph_l641_641128

universe u

-- Definition of a graph structure
-- Note: Only the necessary parts for this problem are included
structure Graph (V : Type u) :=
(Adj : V → V → Prop)
(connected : ∀ u v : V, ∃ p : List V, p.length > 1 ∧ p.head = some u ∧ p.last = some v ∧ ∀ (i : Fin (p.length - 1)), Adj (p.nth_le i (by linarith)) (p.nth_le (i + 1) (by linarith)))

open List

/-- Given a connected graph G, there exists an orientation of its edges such that there is a directed path 
    from a particular vertex A to any other vertex -/
theorem orient_edges_of_connected_graph (V : Type u) (G : Graph V) (A : V) :
  ∃ (orig : V → V → Prop), (∀ u v : V, G.Adj u v ↔ orig u v ∨ orig v u) 
    ∧ (∀ v : V, ∃ p : List V, p.head = some A ∧ p.last = some v ∧ ∀ i : Fin (p.length - 1), orig (p.nth_le i (by linarith)) (p.nth_le (i + 1) (by linarith))) :=
sorry

end orient_edges_of_connected_graph_l641_641128


namespace problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l641_641282

-- Problem 1: Prove the remainder of the Euclidean division of \(9^{100}\) by 8 is 1.
theorem problem1_remainder_of_9_power_100_mod_8 :
  (9 ^ 100) % 8 = 1 :=
by
sorry

-- Problem 2: Prove the last digit of \(2012^{2012}\) is 6.
theorem problem2_last_digit_of_2012_power_2012 :
  (2012 ^ 2012) % 10 = 6 :=
by
sorry

end problem1_remainder_of_9_power_100_mod_8_problem2_last_digit_of_2012_power_2012_l641_641282


namespace logs_added_per_hour_l641_641701

theorem logs_added_per_hour (x : ℕ) : 
  let initial_logs := 6
  let burn_per_hour := 3
  let logs_after_one_hour := initial_logs - burn_per_hour + x
  let logs_after_two_hours := logs_after_one_hour - burn_per_hour + x
  let logs_after_three_hours := logs_after_two_hours - burn_per_hour + x
  logs_after_three_hours = 3 → x = 2 := 
by
  intro h
  have h1 : logs_after_one_hour = initial_logs - burn_per_hour + x := rfl
  have h2 : logs_after_two_hours = logs_after_one_hour - burn_per_hour + x := rfl
  have h3 : logs_after_three_hours = logs_after_two_hours - burn_per_hour + x := rfl
  sorry

end logs_added_per_hour_l641_641701


namespace student_weekly_allowance_l641_641678

theorem student_weekly_allowance (A : ℝ) (h1 : (4 / 15) * A = 1) : A = 3.75 :=
by
  sorry

end student_weekly_allowance_l641_641678


namespace smallest_x_domain_of_g_g_l641_641094

def g (x : ℝ) : ℝ := real.sqrt (x - 5)

theorem smallest_x_domain_of_g_g :
  ∃ x : ℝ, x = 30 ∧ ∀ y, y ≥ 30 → real.sqrt (real.sqrt (y - 5) - 5) = g (g y) :=
by
  sorry

end smallest_x_domain_of_g_g_l641_641094


namespace chance_Z_winning_l641_641864

-- Given conditions as Lean definitions
def p_x : ℚ := 1 / (3 + 1)
def p_y : ℚ := 3 / (2 + 3)
def p_z : ℚ := 1 - (p_x + p_y)

-- Theorem statement: Prove the equivalence of the winning ratio for Z
theorem chance_Z_winning : 
  p_z = 3 / (3 + 17) :=
by
  -- Since we include no proof, we use sorry to indicate it
  sorry

end chance_Z_winning_l641_641864


namespace frustum_volume_l641_641396

noncomputable def slantHeight (r R : ℝ) (π : ℝ) (S_lateral S_bases : ℝ) := 
  (6 * π)⁻¹ * S_bases

noncomputable def height (l : ℝ) (R r : ℝ) := 
  sqrt (l^2 - (R - r)^2)

noncomputable def volume (h r R : ℝ) (π : ℝ) := 
  (1/3 * π) * h * (R^2 + r^2 + R * r)

theorem frustum_volume
  (r R : ℝ) (π : ℝ) (S_lateral S_bases : ℝ)
  (h := height (slantHeight r R π S_lateral S_bases) R r)
  (V := volume h r R π)
  (hr : r = 2)
  (hR : R = 4)
  (hπ : π = real.pi)
  (hS : S_lateral = 20 * real.pi := by sorry)
  (S_vol : V = (224 * real.pi) / 9) : true := by
  sorry

end frustum_volume_l641_641396


namespace sum_of_elements_T_l641_641090

-- Definitions for the conditions
def is_repeating_decimal_of_form_def (x : ℝ) : Prop :=
  ∃ (d e f : ℕ), 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧ 1 <= f ∧ f <= 9 ∧ d ≠ e ∧ e ≠ f ∧ d ≠ f ∧ x = d * 100 + e * 10 + f

def T : set ℝ := {x | ∃ (d e f : ℕ), is_repeating_decimal_of_form_def (0.0ddef)}

-- The actual proof statement
theorem sum_of_elements_T : 
  ∑ x in T, x = 28 :=
sorry

end sum_of_elements_T_l641_641090


namespace morse_code_symbols_l641_641041

def morse_code_symbols_count : ℕ :=
  let count n := 2^n
  (count 1) + (count 2) + (count 3) + (count 4) + (count 5)

theorem morse_code_symbols :
  morse_code_symbols_count = 62 :=
by
  unfold morse_code_symbols_count
  simp
  sorry

end morse_code_symbols_l641_641041


namespace MackenzieSpent_correct_l641_641728

noncomputable def N : ℝ :=
  let U := 9.99 in
  (127.92 - 2 * U) / 6

noncomputable def MackenzieSpent : ℝ :=
  let U := 9.99 in
  let N := (127.92 - 2 * U) / 6 in
  3 * N + 8 * U

theorem MackenzieSpent_correct : MackenzieSpent = 133.89 :=
by
  let U := 9.99
  let N := (127.92 - 2 * 9.99) / 6
  have hN: N = 17.99 := by 
    calc
      N = (127.92 - 2 * 9.99) / 6 : rfl
      ... = 107.94 / 6 : by norm_num
      ... = 17.99 : by norm_num
  rw [N, hN]
  calc
    MackenzieSpent = 3 * 17.99 + 8 * 9.99 : rfl
    ... = 53.97 + 79.92 : by norm_num
    ... = 133.89 : by norm_num

end MackenzieSpent_correct_l641_641728


namespace tim_more_points_than_joe_l641_641860

variable (J K T : ℕ)

theorem tim_more_points_than_joe (h1 : T = 30) (h2 : T = K / 2) (h3 : J + T + K = 100) : T - J = 20 :=
by
  sorry

end tim_more_points_than_joe_l641_641860


namespace tiffany_bags_found_day_after_next_day_l641_641179

noncomputable def tiffany_start : Nat := 10
noncomputable def tiffany_next_day : Nat := 3
noncomputable def tiffany_total : Nat := 20
noncomputable def tiffany_day_after_next_day : Nat := 20 - (tiffany_start + tiffany_next_day)

theorem tiffany_bags_found_day_after_next_day : tiffany_day_after_next_day = 7 := by
  sorry

end tiffany_bags_found_day_after_next_day_l641_641179


namespace graph_A_is_f_plus_2_l641_641971

def f (x : ℝ) : ℝ :=
if h : x ≥ -3 ∧ x ≤ 0 then -2 - x
else if h : x ≥ 0 ∧ x ≤ 2 then (4 - (x - 2)^2).sqrt - 2
else if h : x ≥ 2 ∧ x ≤ 3 then 2 * (x - 2)
else 0

def funca (x : ℝ) := f x + 2
def funcc (x : ℝ) := f x - 1

theorem graph_A_is_f_plus_2 : ∀ x, graph_A (x) = funca (x) :=
by
  sorry

end graph_A_is_f_plus_2_l641_641971


namespace rectangular_reconfiguration_l641_641554

theorem rectangular_reconfiguration (k : ℕ) (n : ℕ) (h₁ : k - 5 > 0) (h₂ : k ≥ 6) (h₃ : k ≤ 9) :
  (k * (k - 5) = n^2) → (n = 6) :=
by {
  sorry  -- proof is omitted
}

end rectangular_reconfiguration_l641_641554


namespace all_boys_are_brothers_l641_641867

open Set

variable {Boy : Type} [Fintype Boy] (boys : Finset Boy)
variable (B : Boy → Finset Boy) (n : Nat)

-- The size of the group of boys is seven
variable [hcp : boys.card = 7]

-- Condition: Each boy has at least three brothers among the others
variable (brother : ∀ b ∈ boys, (B b ∩ boys).card ≥ 3)

theorem all_boys_are_brothers : ∀ b1 b2 ∈ boys, b1 ≠ b2 → b1 ∈ B b2 :=
begin
  sorry
end

end all_boys_are_brothers_l641_641867


namespace topmost_circle_values_l641_641331

noncomputable theory

def is_valid_arrangement (A B C D E F : ℤ) : Prop :=
  A = |B - C| ∧ B = |D - E| ∧ C = |E - F| ∧
  D ≠ E ∧ E ≠ F ∧ D ≠ F ∧
  {D, E, F} ⊆ {1, 2, 3, 4, 5, 6}

theorem topmost_circle_values :
  ∃ (A B C D E F : ℤ), is_valid_arrangement A B C D E F ∧ A ∈ {1, 2, 3} :=
sorry

end topmost_circle_values_l641_641331


namespace Morse_code_distinct_symbols_l641_641051

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l641_641051


namespace asian_population_percentage_in_west_is_57_l641_641332

variable (NE MW South West : ℕ)

def total_asian_population (NE MW South West : ℕ) : ℕ :=
  NE + MW + South + West

def west_asian_population_percentage
  (NE MW South West : ℕ) (total_asian_population : ℕ) : ℚ :=
  (West : ℚ) / (total_asian_population : ℚ) * 100

theorem asian_population_percentage_in_west_is_57 :
  total_asian_population 2 3 4 12 = 21 →
  west_asian_population_percentage 2 3 4 12 21 = 57 :=
by
  intros
  sorry

end asian_population_percentage_in_west_is_57_l641_641332


namespace length_AB_is_sqrt_14_l641_641070

-- Definitions for parametric and polar equations
def parametric_eq1 (t : ℝ) : ℝ × ℝ := (1 - (sqrt 2 / 2) * t, 1 + (sqrt 2 / 2) * t)
def polar_eq2 (ρ θ : ℝ) : Bool := ρ ^ 2 - 2 * ρ * (cos θ) - 3 = 0

-- Definitions for cartesian equations
def cartesian_eq1 (x y : ℝ) : Prop := x + y = 2
def cartesian_eq2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 - 2 * x - 3 = 0

-- Definitions for common points A and B
def common_points (t1 t2 : ℝ) : Prop := 
  ∃ (x1 y1 x2 y2 : ℝ), 
    parametric_eq1 t1 = (x1, y1) ∧
    cartesian_eq2 x1 y1 ∧
    parametric_eq1 t2 = (x2, y2) ∧
    cartesian_eq2 x2 y2

-- Distance function between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

-- Prove the length of the line segment AB is √14
theorem length_AB_is_sqrt_14 : 
  ∀ (t1 t2 : ℝ), 
    common_points t1 t2 →
    distance (fst (parametric_eq1 t1)) (snd (parametric_eq1 t1)) (fst (parametric_eq1 t2)) (snd (parametric_eq1 t2)) = sqrt 14 :=
by
  sorry

end length_AB_is_sqrt_14_l641_641070


namespace min_value_of_a_l641_641990

theorem min_value_of_a (r s t : ℕ) (h1 : r > 0) (h2 : s > 0) (h3 : t > 0)
  (h4 : r * s * t = 2310) (h5 : r + s + t = a) : 
  a = 390 → True :=
by { 
  intros, 
  sorry 
}

end min_value_of_a_l641_641990


namespace minimize_at_incenter_l641_641402

-- Define the problem
variables {A B C P A' B' C' : Type*}

-- Assumptions about points and projections
variables [tri : Triangle A B C]
variables [projPA' : PerpendicularProjection P BC A']
variables [projPB' : PerpendicularProjection P CA B']
variables [projPC' : PerpendicularProjection P AB C']

-- Define the question as a function to minimize
def S (a BC PA' CA PB' AB PC' : Real) : Real :=
  (BC / PA') + (CA / PB') + (AB / PC')

-- The correct answer statement: achieving the minimum when P is the incenter
theorem minimize_at_incenter
  (hmin : ∀ P, S a BC PA' CA PB' AB PC' ≥ S a BC (incenter_triangle A B C BC PA' CA PB' AB PC') CA (incenter_triangle A B C BC PA' CA PB' AB PC') AB (incenter_triangle A B C BC PA' CA PB' AB PC')) :
  is_incenter P A B C :=
sorry

end minimize_at_incenter_l641_641402


namespace sin_right_triangle_l641_641356

theorem sin_right_triangle (FG GH : ℝ) (h1 : FG = 13) (h2 : GH = 12) (h3 : FG^2 = FH^2 + GH^2) : 
  sin_H = 5 / 13 :=
by sorry

end sin_right_triangle_l641_641356


namespace symmetric_points_x_axis_l641_641795

theorem symmetric_points_x_axis (a b : ℤ) 
  (h1 : a - 1 = 2) (h2 : 5 = -(b - 1)) : (a + b) ^ 2023 = -1 := 
by
  -- The proof steps will go here.
  sorry

end symmetric_points_x_axis_l641_641795


namespace train_speed_l641_641710

theorem train_speed (length_goods_train : ℝ) (speed_goods_train_kmph : ℝ) (passing_time_seconds : ℝ) (relative_speed_relative_speed : ℝ) : 
(speed_goods_train_kmph = 52) →
(length_goods_train = 280) →
(passing_time_seconds = 9) →
(length_goods_train / passing_time_seconds / (1 / 3.6) - speed_goods_train_kmph = 60.16) :=
begin
   intros h1 h2 h3,
   have speed_goods_train_mps : ℝ := speed_goods_train_kmph * (1 / 3.6),
   have distance_covered : ℝ := length_goods_train,
   have relative_speed : ℝ := distance_covered / passing_time_seconds,
   have speed_mans_train_kmph : ℝ := (relative_speed - speed_goods_train_mps) * 3.6,
   rw [h1, h2, h3, if_pos] at *,
   
   sorry
end

end train_speed_l641_641710


namespace derivative_at_0_l641_641956

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- State the theorem
theorem derivative_at_0 : f' 0 = 4 :=
by {
  -- Inserting sorry to skip the proof
  sorry
}

end derivative_at_0_l641_641956


namespace distance_from_A_to_B_l641_641285

theorem distance_from_A_to_B :
  ∀ (distance_covered : ℕ) (time_taken : ℕ) (midpoint_time : ℕ),
  distance_covered = 480 →
  time_taken = 6 →
  midpoint_time = 3 →
  (let speed : ℕ := distance_covered / time_taken in
   let total_time : ℕ := (time_taken + midpoint_time) * 2 in
   let distance_from_A_to_B : ℕ := speed * total_time in
   distance_from_A_to_B = 1440) := sorry

end distance_from_A_to_B_l641_641285


namespace greatest_product_sum_2000_eq_1000000_l641_641232

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641232


namespace smallest_positive_integer_solution_l641_641260

def is_unique_k (n : ℕ) : Prop :=
  ∃! (k : ℤ), (7 : ℚ) / 16 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 9 / 20

theorem smallest_positive_integer_solution : ∃ (n : ℕ), n = 63 ∧ is_unique_k n :=
by
  use 63
  split
  { refl }
  {
    sorry
  }

end smallest_positive_integer_solution_l641_641260


namespace smallest_a_l641_641981

def root_product (P : Polynomial ℚ) : ℚ :=
  P.coeff 0

def poly_sum_roots_min_a (r1 r2 r3 : ℤ) (a b c : ℚ) : Prop :=
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
  r1 * r2 * r3 = 2310 ∧
  root_product (Polynomial.monomial 3 1 - Polynomial.monomial 2 a + Polynomial.monomial 1 b - Polynomial.monomial 0 2310) = 2310 ∧
  r1 + r2 + r3 = a

theorem smallest_a : ∃ a b : ℚ, ∀ r1 r2 r3 : ℤ, poly_sum_roots_min_a r1 r2 r3 a b 2310 → a = 28
  by sorry

end smallest_a_l641_641981


namespace min_value_f_l641_641821

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / (Real.sin x * Real.cos x)

theorem min_value_f :
  ∃ x ∈ set.Ioo 0 (Real.pi / 2), f x = 2 * Real.sqrt 2 := 
sorry

end min_value_f_l641_641821


namespace range_of_y_l641_641415

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_of_y :
  (∀ x, f(x) ∈ Icc (3/8:ℝ) (4/9:ℝ)) →
  (∀ y, ∃ x, y = f(x) + real.sqrt (1 - 2 * f(x))) ↔ y ∈ Icc (7/9:ℝ) (7/8:ℝ) :=
begin
  sorry
end

end range_of_y_l641_641415


namespace chord_slope_range_l641_641814

theorem chord_slope_range (x1 y1 x2 y2 x0 y0 : ℝ) (h1 : x1^2 + (y1^2)/4 = 1) (h2 : x2^2 + (y2^2)/4 = 1)
  (h3 : x0 = (x1 + x2) / 2) (h4 : y0 = (y1 + y2) / 2)
  (h5 : x0 = 1/2) (h6 : 1/2 ≤ y0 ∧ y0 ≤ 1) :
  -4 ≤ (-2 / y0) ∧ -2 ≤ (-2 / y0) :=
by
  sorry

end chord_slope_range_l641_641814


namespace max_product_l641_641221

theorem max_product (x : ℤ) : x + (2000 - x) = 2000 → x * (2000 - x) ≤ 1000000 :=
by
  sorry

end max_product_l641_641221


namespace morse_code_symbols_l641_641043

def morse_code_symbols_count : ℕ :=
  let count n := 2^n
  (count 1) + (count 2) + (count 3) + (count 4) + (count 5)

theorem morse_code_symbols :
  morse_code_symbols_count = 62 :=
by
  unfold morse_code_symbols_count
  simp
  sorry

end morse_code_symbols_l641_641043


namespace max_a_for_decreasing_cos_minus_sin_l641_641474

theorem max_a_for_decreasing_cos_minus_sin :
  ∀ (f : ℝ → ℝ) (a : ℝ), 
  (∀ x : ℝ, f x = cos x - sin x) →
  (∀ x : ℝ, -a ≤ x ∧ x ≤ a → f (x + 1) ≤ f x) →
  a ≤ π / 4 :=
by
  intros f a h1 h2
  sorry

end max_a_for_decreasing_cos_minus_sin_l641_641474


namespace parabola_equation_l641_641428

/-- The vertex of the parabola is at the origin and the focus is at (0, 1) -/
def parabola_vertex_focus (C : Type) [AddCommGroup C] [VectorSpace ℝ C] {x y : C} (h_vertex : x = (0, 0)) (h_focus : y = (0, 1)) : Prop :=
  ∃ k : ℝ, ∀ (x : ℝ), (x^2 = 4 * k * y)

/-- Line l passes through point (0,1), intersects parabola C at points A, B and intersects line y = x - 2 at points M, N -/
def line_intersection (A B M N : Type) [AddCommGroup A] [VectorSpace ℝ A] (h_line1 : ∀ {x : ℝ}, y = (x - 2)) (h_point : A ≠ B) (h_interp : B ≠ M ∧ B ≠ N) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, |MN| = 2 * sqrt(2) * sqrt((5 / t + 3) ^ 2 + 16 / 25) ∧ t ≠ 0

theorem parabola_equation (C M N : Type) [AddCommGroup M] [VectorSpace ℝ M]
  (h_vertex : x = (0, 0))
  (h_focus : y = (0, 1)) 
  (h_intersection : ∀ {x : ℝ}, y = (x - 2))
  (h_points : A ≠ B)
  (h_noncollinear : B ≠ M ∧ B ≠ N) :
  parabola_vertex_focus C h_vertex h_focus ∧ line_intersection A B M N h_intersection h_points h_noncollinear :=
sorry

end parabola_equation_l641_641428


namespace max_angle_OMB_l641_641107

theorem max_angle_OMB (O A M B : Point) (h_circle : IsCircle O A M) (h_midpoint : Midpoint B O A) :
  ∀ (θ : ℝ), θ = ∠ O M B → θ ≤ π / 6 :=
by {
  sorry
}

end max_angle_OMB_l641_641107


namespace quadrilateral_tangential_quadrilateral_cyclic_l641_641160

variables {A B C D M : Point}
variables (equidistant1 : equidistant M (line A B) (line C D))
variables (equidistant2 : equidistant M (line B C) (line A D))
variables (area_eq : area (quadrilateral A B C D) = distance M A * distance M C + distance M B * distance M D)

theorem quadrilateral_tangential :
  tangential (quadrilateral A B C D) :=
by sorry

theorem quadrilateral_cyclic :
  cyclic (quadrilateral A B C D) :=
by sorry

end quadrilateral_tangential_quadrilateral_cyclic_l641_641160


namespace Morse_code_distinct_symbols_l641_641050

theorem Morse_code_distinct_symbols : 
  (2^1) + (2^2) + (2^3) + (2^4) + (2^5) = 62 :=
by sorry

end Morse_code_distinct_symbols_l641_641050


namespace exists_root_within_interval_l641_641938

theorem exists_root_within_interval
  (a b : ℕ → ℝ) :
  ∃ x ∈ set.Icc (0 : ℝ) (2 * Real.pi), (∑ k in finset.range (n + 1), a k * (Real.sin (k * x)) + b k * (Real.cos (k * x))) = 0 :=
  sorry

end exists_root_within_interval_l641_641938


namespace cylindrical_cone_shape_l641_641774

-- Definition of the shapes in cylindrical coordinates
def is_cylindrical_cone (θ r z : ℝ) (c : ℝ) : Prop :=
  θ = c ∧ r = |z|

-- The theorem to prove
theorem cylindrical_cone_shape (c : ℝ) (θ r z : ℝ) (h : is_cylindrical_cone θ r z c) :
  cone θ r z :=
sorry

end cylindrical_cone_shape_l641_641774


namespace sam_seashells_l641_641135

theorem sam_seashells (a b : ℕ) (h₁ : a = 18) (h₂ : b = 17) : a + b = 35 := by
  rw [h₁, h₂]
  norm_num

end sam_seashells_l641_641135


namespace morse_code_symbols_l641_641042

def morse_code_symbols_count : ℕ :=
  let count n := 2^n
  (count 1) + (count 2) + (count 3) + (count 4) + (count 5)

theorem morse_code_symbols :
  morse_code_symbols_count = 62 :=
by
  unfold morse_code_symbols_count
  simp
  sorry

end morse_code_symbols_l641_641042


namespace people_in_room_eq_33_l641_641177

variable (people chairs : ℕ)

def chairs_empty := 5
def chairs_total := 5 * 5
def chairs_occupied := (4 * chairs_total) / 5
def people_seated := 3 * people / 5

theorem people_in_room_eq_33 : 
    (people_seated = chairs_occupied ∧ chairs_total - chairs_occupied = chairs_empty)
    → people = 33 :=
by
  sorry

end people_in_room_eq_33_l641_641177


namespace finance_to_manufacturing_ratio_l641_641117

theorem finance_to_manufacturing_ratio : 
    let finance_angle := 72
    let manufacturing_angle := 108
    (finance_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 2 ∧ 
    (manufacturing_angle:ℕ) / (Nat.gcd finance_angle manufacturing_angle) = 3 := 
by 
    sorry

end finance_to_manufacturing_ratio_l641_641117


namespace greatest_product_sum_2000_eq_1000000_l641_641236

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641236


namespace prime_and_multiple_of_5_probability_l641_641559

def is_prime (n : ℕ) : Prop :=
  nat.prime n

def is_multiple_of_5 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 5 * k

def card_numbers : finset ℕ := finset.range 101   -- numbers 1 to 100

theorem prime_and_multiple_of_5_probability :
  let eligible_cards := card_numbers.filter (λ n, is_prime n ∧ is_multiple_of_5 n) in
  eligible_cards.card.to_real / card_numbers.card.to_real = 1 / 100 :=
by 
  sorry

end prime_and_multiple_of_5_probability_l641_641559


namespace y_increasing_in_interval_l641_641666

noncomputable def is_increasing_interval : Prop :=
  ∀ x y : ℝ, 
    (π < x ∧ x < 2 * π) → 
    (π < y ∧ y < 2 * π) → 
    (x < y) → 
    (x * Real.cos x - Real.sin x) < (y * Real.cos y - Real.sin y)
  
theorem y_increasing_in_interval : is_increasing_interval :=
  sorry

end y_increasing_in_interval_l641_641666


namespace eliana_fuel_purchase_l641_641749

-- Define the conditions
constants (Q : ℝ) (S : ℝ)

-- Given conditions as assumptions
axiom total_spent : 3 * Q + S * Q + 4 * Q = 90
axiom fuel_increase : S = (3 + 4) / 2

-- The theorem to be proven
theorem eliana_fuel_purchase : (3 * Q) + (4 * Q) = 60 :=
by
  sorry

end eliana_fuel_purchase_l641_641749


namespace complex_conjugate_difference_l641_641785

def z : ℂ := (1 - I) / (2 + 2 * I)

theorem complex_conjugate_difference : z - conj(z) = -I := by
  sorry

end complex_conjugate_difference_l641_641785


namespace find_speed_l641_641327

variables (x V : ℝ)

-- Conditions
def initial_distance := x
def speed_first_part := V
def distance_second_part := 2 * x
def speed_second_part := 20
def total_distance := 3 * x
def avg_speed := 24

-- Time calculations
def time_first_part := initial_distance / speed_first_part
def time_second_part := distance_second_part / speed_second_part
def total_time := total_distance / avg_speed

-- Lean 4 statement
theorem find_speed (h : total_time = time_first_part + time_second_part) : V = 40 :=
by sorry

end find_speed_l641_641327


namespace geometric_sequence_product_l641_641890

theorem geometric_sequence_product (a b : ℝ) (h : (-1 : ℝ), a, b, 2 form_geom_seq) : a * b = -2 :=
by
  sorry -- Proof goes here

end geometric_sequence_product_l641_641890


namespace intervals_of_monotonic_increase_max_area_of_triangle_l641_641522

-- Define the function f
def f (x : ℝ) : ℝ := sin x * cos x - cos ((x + π / 4))^2

-- Lean statement for problem (I)
theorem intervals_of_monotonic_increase: 
  ∀ x : ℝ, (f x = sin x * cos x - cos ((x + π / 4))^2) →
  (∀ k : ℤ, (- π / 4 + k * π ≤ x ∧ x ≤ π / 4 + k * π) → monotonic f x) := 
  sorry

-- Assumptions for problem (II)
variables (A B C : ℝ) (a b c : ℝ)
axiom acute_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2
axiom side_relationships : a = 1 ∧ sin A = (1/2)

-- Lean statement for problem (II)
theorem max_area_of_triangle:
  ∀ a b c A : ℝ,
  (a = 1 ∧ f (A / 2) = 0) →
  (1 + sqrt 3 * b * c ≥ 2 * b * c → (1/2) * b * c * sin (π / 6)) ≤ (2 + sqrt 3) / 4 := 
  sorry

end intervals_of_monotonic_increase_max_area_of_triangle_l641_641522


namespace Morse_code_number_of_distinct_symbols_l641_641036

def count_sequences (n : ℕ) : ℕ :=
  2 ^ n

theorem Morse_code_number_of_distinct_symbols :
  (count_sequences 1) + (count_sequences 2) + (count_sequences 3) + (count_sequences 4) + (count_sequences 5) = 62 :=
by
  simp [count_sequences]
  norm_num
  sorry

end Morse_code_number_of_distinct_symbols_l641_641036


namespace largest_k_inequality_l641_641755

theorem largest_k_inequality
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_pos : (a + b) * (b + c) * (c + a) > 0) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a ≥ 
  (1 / 2) * abs ((a^3 - b^3) / (a + b) + (b^3 - c^3) / (b + c) + (c^3 - a^3) / (c + a)) :=
by
  sorry

end largest_k_inequality_l641_641755


namespace switches_in_position_A_l641_641623

-- Definitions

def N : ℕ := 6860
def positions : List String := ["A", "B", "C", "D", "E"]
def initial_position : String := "A"

def label (x y z w : ℕ) : ℕ := (2^x) * (3^y) * (5^z) * (7^w)
def step_advance (label_ i : ℕ) : List ℕ :=
  (List.range N).filter (fun j => (label (j % 6) ((j // 6) % 6) ((j // 36) % 6) ((j // 216) % 6)) % label_ i = 0)

def is_in_position_A (x y z w : ℕ) : Prop := (6 - x) * (6 - y) * (6 - z) * (6 - w) % 5 = 0

-- Theorem to Prove

theorem switches_in_position_A : ∃ count : ℕ, count = 6455 :=
  sorry

end switches_in_position_A_l641_641623


namespace part1_part2_l641_641426

-- Define the conditions for the problem
def polar_to_rect (ρ θ : ℝ) := (x y : ℝ) (h₁ : x = ρ * cos θ) (h₂ : y = ρ * sin θ) : Prop := x = -1 + (3/5)*t ∧ y = -1 + (4/5)*t

-- Define the polar equation of the curve C
def curve_C (ρ θ : ℝ) := ρ = sqrt 2 * sin (θ + π/4)

-- The rectangular equation of the curve C
theorem part1 (x y : ℝ) (h₁ : x = ρ * cos θ) (h₂ : y = ρ * sin θ) : x^2 + y^2 - x - y = 0 := sorry

-- Calculate the distance between the two intersection points M and N
theorem part2 (f : ℝ → ℝ) (M N : ℝ × ℝ) (t₁ t₂ : ℝ) 
    (line_param : ℝ → ℝ × ℝ) 
    (eq1 : ∀ t, line_param t = (-1 + (3/5)*t, -1 + (4/5)*t))
    (curve_eq : ∀ θ, curve_C (ρ * cos θ, ρ * sin θ)) : 
    dist M N = sqrt (41) / 5 := sorry

end part1_part2_l641_641426


namespace number_of_four_digit_numbers_with_one_repeated_digit_l641_641005

theorem number_of_four_digit_numbers_with_one_repeated_digit :
  (∃ (x : Finset ℕ), {d : ℕ // d ∈ {1, 2, 3, 4}} ∧ -- Choose digit appearing twice
  ∃ (p : Finset (Finset ℕ)), {pos : ℕ // pos ∈ {1, 2, 3, 4}} ∧ (p.card = 2) ∧ -- Choose positions for the repeated digit
  ∀ (a b : ℕ), a ∈ {1, 2, 3, 4} → b ∈ {1, 2, 3, 4} ∧ a ≠ x ∧ b ≠ x ) -- Conditions for remaining digits
  → 144 = 4 * 6 * 6 :=
by
  sorry

end number_of_four_digit_numbers_with_one_repeated_digit_l641_641005


namespace m_range_l641_641403

variable (a1 b1 : ℝ)

def arithmetic_sequence (n : ℕ) : ℝ := a1 + 2 * (n - 1)
def geometric_sequence (n : ℕ) : ℝ := b1 * 2^(n - 1)

def a2_condition : Prop := arithmetic_sequence a1 2 + geometric_sequence b1 2 < -2
def a1_b1_condition : Prop := a1 + b1 > 0

theorem m_range : a1_b1_condition a1 b1 ∧ a2_condition a1 b1 → 
  let a4 := arithmetic_sequence a1 4 
  let b3 := geometric_sequence b1 3 
  let m := a4 + b3 
  m < 0 := 
by
  sorry

end m_range_l641_641403


namespace minimize_PA2_PB2_PC2_l641_641149

def point : Type := ℝ × ℝ

noncomputable def distance_sq (P Q : point) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

noncomputable def PA_sq (P : point) : ℝ := distance_sq P (5, 0)
noncomputable def PB_sq (P : point) : ℝ := distance_sq P (0, 5)
noncomputable def PC_sq (P : point) : ℝ := distance_sq P (-4, -3)

noncomputable def circumcircle (P : point) : Prop := 
  P.1^2 + P.2^2 = 25

noncomputable def objective_function (P : point) : ℝ := 
  PA_sq P + PB_sq P + PC_sq P

theorem minimize_PA2_PB2_PC2 : ∃ P : point, circumcircle P ∧ 
  (∀ Q : point, circumcircle Q → objective_function P ≤ objective_function Q) :=
sorry

end minimize_PA2_PB2_PC2_l641_641149


namespace roots_of_quadratic_eq_with_rational_coeffs_l641_641564

theorem roots_of_quadratic_eq_with_rational_coeffs (a b c d e : ℚ) (b_notin_Q : ∀ q : ℚ, q ≠ b) :
  (∃ root1 root2 : ℝ, 
  (a : ℝ) * root1^2 + b * root1 + c = 0 ∧ 
  (a : ℝ) * root2^2 + b * root2 + c = 0 ∧ 
  root1 = d + real.sqrt e ∧ 
  root2 = d - real.sqrt e) :=
sorry

end roots_of_quadratic_eq_with_rational_coeffs_l641_641564


namespace exists_monic_poly_degree_8_with_sqrt3_sqrt5_as_root_l641_641358

theorem exists_monic_poly_degree_8_with_sqrt3_sqrt5_as_root :
  ∃ p : Polynomial ℚ, Monic p ∧ degree p = 8 ∧ root_of_polynomial p (√3 + √5) := by
  sorry

end exists_monic_poly_degree_8_with_sqrt3_sqrt5_as_root_l641_641358


namespace gcd_greatest_possible_value_l641_641383

noncomputable def Sn (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_greatest_possible_value (n : ℕ) (hn : 0 < n) : 
  Nat.gcd (3 * Sn n) (n + 1) = 1 :=
sorry

end gcd_greatest_possible_value_l641_641383


namespace common_solution_l641_641644

theorem common_solution :
  (∀ m ∈ ({-5, -4, -3, -1, 0, 1, 3, 23, 124, 1000} : Set ℤ),
  (2 * m + 1) * (1 : ℤ) + (2 - 3 * m) * (-1 : ℤ) + 1 - 5 * m = 0) :=
by {
  intros m hm,
  -- verify that the common solution holds for all given m
  finCasesList : ∀ m ∈ {-5, -4, -3, -1, 0, 1, 3, 23, 124, 1000},
  calc
    (2 * m + 1) * 1 + (2 - 3 * m) * (-1) + 1 - 5 * m = 0 : sorry
}

end common_solution_l641_641644


namespace boat_speed_in_still_water_l641_641272

variable (B S : ℝ)

def downstream_speed := 10
def upstream_speed := 4

theorem boat_speed_in_still_water :
  B + S = downstream_speed → 
  B - S = upstream_speed → 
  B = 7 :=
by
  intros h₁ h₂
  -- We would insert the proof steps here
  sorry

end boat_speed_in_still_water_l641_641272


namespace blue_balls_unchanged_l641_641025

def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5
def added_yellow_balls : ℕ := 4

theorem blue_balls_unchanged :
  initial_blue_balls = 2 := by
  sorry

end blue_balls_unchanged_l641_641025


namespace tan_difference_identity_l641_641782

theorem tan_difference_identity (x : ℝ) (h1 : sin x = 4 / 5) (h2 : x ∈ Ioo (π / 2) π) :
  tan (x - (π / 4)) = 7 := 
sorry

end tan_difference_identity_l641_641782


namespace Morse_code_number_of_distinct_symbols_l641_641037

def count_sequences (n : ℕ) : ℕ :=
  2 ^ n

theorem Morse_code_number_of_distinct_symbols :
  (count_sequences 1) + (count_sequences 2) + (count_sequences 3) + (count_sequences 4) + (count_sequences 5) = 62 :=
by
  simp [count_sequences]
  norm_num
  sorry

end Morse_code_number_of_distinct_symbols_l641_641037


namespace MorseCodeDistinctSymbols_l641_641030

theorem MorseCodeDistinctSymbols:
  (1.sequence (λ _, bool).length = {1, 2, 3, 4, 5}).card = 62 :=
by
  sorry

end MorseCodeDistinctSymbols_l641_641030


namespace smallest_possible_value_of_a_l641_641983

theorem smallest_possible_value_of_a (a b : ℕ) :
  (∃ (r1 r2 r3 : ℕ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    (r1 * r2 * r3 = 2310) ∧
    (a = r1 + r2 + r3)) →
  a = 52 :=
begin
  sorry
end

end smallest_possible_value_of_a_l641_641983


namespace michael_money_ratio_l641_641925

theorem michael_money_ratio :
  ∀ (M B A A' : ℕ), -- M: Michael’s initial money, B: Brother’s initial money, A: Amount spent on candy, A': Amount brother has left
  M = 42 → B = 17 → A = 3 → A' = 35 →
  let given := A' + A - B in
  let ratio := (given, M) in
  ratio = (1, 2) :=
by
  intros M B A A' hM hB hA hA'
  let given := A' + A - B
  let ratio := (given, M)
  have h1 : given = 21 := by sorry
  have h2 : ratio = (21, 42) := by sorry
  have h3 : (21 / Nat.gcd 21 42, 42 / Nat.gcd 21 42) = (1, 2) := by sorry
  exact h3

end michael_money_ratio_l641_641925


namespace greatest_product_sum_2000_eq_1000000_l641_641231

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641231


namespace seokjin_rank_l641_641884

theorem seokjin_rank (jimin_rank seokjin_offset : ℕ) (h1 : jimin_rank = 24) (h2 : seokjin_offset = 19) : ∃ seokjin_rank : ℕ, seokjin_rank = jimin_rank - seokjin_offset ∧ seokjin_rank = 5 :=
by
  existsi (jimin_rank - seokjin_offset)
  split
  { rw [h1, h2]
    norm_num }
  { rw [h1, h2]
    norm_num }
  sorry

end seokjin_rank_l641_641884


namespace power_product_l641_641104

theorem power_product (m n : ℕ) (hm : 2 < m) (hn : 0 < n) : 
  (2^m - 1) * (2^n + 1) > 0 :=
by 
  sorry

end power_product_l641_641104


namespace brendas_age_l641_641717

theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B) 
  (h2 : J = B + 8) 
  (h3 : A = J) 
: B = 8 / 3 := 
by 
  sorry

end brendas_age_l641_641717


namespace no_two_consecutive_heads_l641_641291

/-- The probability that two heads never appear consecutively when a coin is tossed 10 times is 9/64. -/
theorem no_two_consecutive_heads (tosses : ℕ) (prob: ℚ):
  tosses = 10 → 
  (prob = 9 / 64) → 
  prob = probability_no_consecutive_heads tosses := sorry

end no_two_consecutive_heads_l641_641291


namespace polynomial_expression_l641_641416

theorem polynomial_expression
  (p : ℝ → ℝ)
  (h₁ : ∃ a b c d e f, p = λ x, a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f)
  (h₂ : p 0 = -1)
  (h₃ : p 1 = 1)
  (h₄ : ∀ q : ℝ → ℝ, p = q + 1 → (q = (λ x, (x - 0)^3 * k x) → k 0 = 0))
  (h₅ : ∀ r : ℝ → ℝ, p = r - 1 → (r = (λ x, (x - 1)^3 * m x) → m 1 = 0)) :
  p = λ x, 12 * x^5 - 30 * x^4 + 20 * x^3 - 1 :=
sorry

end polynomial_expression_l641_641416


namespace total_dividends_correct_l641_641654

-- Conditions
def net_profit (total_income expenses loan_penalty_rate : ℝ) : ℝ :=
  let net1 := total_income - expenses
  let loan_penalty := net1 * loan_penalty_rate
  net1 - loan_penalty

def total_loan_payments (monthly_payment months additional_payment : ℝ) : ℝ :=
  (monthly_payment * months) - additional_payment

def dividend_per_share (net_profit total_loan_payments num_shares : ℝ) : ℝ :=
  (net_profit - total_loan_payments) / num_shares

noncomputable def total_dividends_director (dividend_per_share shares_owned : ℝ) : ℝ :=
  dividend_per_share * shares_owned

theorem total_dividends_correct :
  total_dividends_director (dividend_per_share (net_profit 1500000 674992 0.2) (total_loan_payments 23914 12 74992) 1000) 550 = 246400 :=
sorry

end total_dividends_correct_l641_641654


namespace caratheodory_extension_theorem_l641_641075

-- Definitions based on given conditions
variable {Ω : Type*} {𝓐 : Set (Set Ω)} {P : MeasureTheory.Measure Ω}

-- A measure space (Ω, 𝓐, P)
variables (Ω 𝓐 P) [measure_space Ω]

-- Distance definition
def dist (A B : Set Ω) : ℝ := P (A \triangle B)

-- Definition of the outer measure P^*
noncomputable def P_star (A : Set Ω) : ℝ :=
  inf {P (⋃ (A_n : ℕ → Set Ω), (∀ n, A_n n ∈ 𝓐) ∧ (A ⊆ ⋃ n, A_n n)}

-- Definition of closure of an algebra under the distance
def closure_of_algebra : Set (Set Ω) :=
  {B | ∃ A ∈ 𝓐, dist A B < ε}

-- The problem statement to prove in Lean
theorem caratheodory_extension_theorem :
  let 𝓐_star := closure_of_algebra 𝓐 P in
  (∀ B : Set Ω, B ∈ 𝓐_star → Bᶜ ∈ 𝓐_star) ∧ 
  (∀ B_i : ℕ → (Set Ω), (∀ i, B_i i ∈ 𝓐_star) → (⋃ i, B_i i) ∈ 𝓐_star) ∧
  (P_star Ω = 1) ∧
  (∀ B : Set Ω, B ∈ 𝓐_star → P_star B = P B)
:= by sorry

end caratheodory_extension_theorem_l641_641075


namespace average_width_of_bookshelf_is_4_125_l641_641895

theorem average_width_of_bookshelf_is_4_125 : 
  let widths := [5, 3 / 4, 1.5, 3.25, 4, 3, 7 / 2, 12] in
  (widths.sum / widths.length = 4.125) :=
by
  -- Define the widths of the books
  let widths := [5, 3 / 4, 1.5, 3.25, 4, 3, 7 / 2, 12]
  -- Compute the total sum of widths
  have h_sum : widths.sum = 33 := by sorry
  -- Compute the number of books
  have h_length : widths.length = 8 := by sorry
  -- Compute the average width
  show widths.sum / widths.length = 4.125
  rw [h_sum, h_length]
  norm_num -- verifies that 33 / 8 = 4.125 as claimed
  sorry

end average_width_of_bookshelf_is_4_125_l641_641895


namespace gcd_factorial_l641_641775

open Int

theorem gcd_factorial (n : ℕ) (hn : 0 < n) : gcd (2 * factorial n) (factorial n) = factorial n := 
by
  sorry

end gcd_factorial_l641_641775


namespace num_students_second_class_is_50_l641_641147

def avg_marks_first_class := 40
def num_students_first_class := 24
def avg_marks_second_class := 60
def combined_avg_marks := 53.513513513513516

noncomputable def number_of_students_in_second_class : ℕ :=
(324.324324324324 / 6.486486486486484).toNat

theorem num_students_second_class_is_50 :
  number_of_students_in_second_class = 50 := by
  sorry

end num_students_second_class_is_50_l641_641147


namespace monotonicity_and_range_of_a_l641_641817

noncomputable def f (x a : ℝ) := Real.log x - a * x - 2

theorem monotonicity_and_range_of_a (a : ℝ) (h : a ≠ 0) :
  ((∀ x > 0, (Real.log x - a * x - 2) < (Real.log (x + 1) - a * (x + 1) - 2)) ↔ (a < 0)) ∧
  ((∃ M, M = Real.log (1/a) - a * (1/a) - 2 ∧ M > a - 4) → 0 < a ∧ a < 1) := sorry

end monotonicity_and_range_of_a_l641_641817


namespace no_such_nat_n_l641_641081

theorem no_such_nat_n :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → (10 * (10 * a + n) + b) % (10 * a + b) = 0 :=
by
  sorry

end no_such_nat_n_l641_641081


namespace binomial_expansion_coefficient_l641_641477

theorem binomial_expansion_coefficient {n : ℕ} (h : (2 : ℕ)^n = 32) :
  ∑ r in Finset.range (n + 1), Nat.choose n r * (1 : ℚ)^(n - r) * (1 : ℚ)^r = 32 →
  ∑ r in Finset.range (n + 1), (Nat.choose n r * ((√(1 : ℚ))^(n - r) * (1 / 1)^(r))) = 5 :=
by
  sorry

end binomial_expansion_coefficient_l641_641477


namespace calculation_of_nested_cuberoot_l641_641337

theorem calculation_of_nested_cuberoot (M : Real) (h : 1 < M) : (M^1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3)^(1 / 3 + 1 / 3) = M^(40 / 81) := 
by 
  sorry

end calculation_of_nested_cuberoot_l641_641337


namespace max_value_trig_expression_l641_641374

theorem max_value_trig_expression :
  ∃ (x : ℝ), (-π/2 ≤ x ∧ x ≤ -π/4) ∧ 
  (∀ y, y = sin (x + 3 * π / 4) + cos (x + π / 3) + cos (x + π / 4) → 
  y ≤ 2 * cos (-π / 24)) :=
sorry

end max_value_trig_expression_l641_641374


namespace max_expression_tends_to_infinity_l641_641525

noncomputable def maximize_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))

theorem max_expression_tends_to_infinity : 
  ∀ (x y z : ℝ), -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 → 
    ∃ M : ℝ, maximize_expression x y z > M :=
by
  intro x y z h
  sorry

end max_expression_tends_to_infinity_l641_641525


namespace greatest_product_sum_2000_l641_641220

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641220


namespace max_sum_arith_seq_l641_641874

theorem max_sum_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = 8 + (n - 1) * d) →
  d ≠ 0 →
  a 1 = 8 →
  a 5 ^ 2 = a 1 * a 7 →
  S n = n * a 1 + (n * (n - 1) * d) / 2 →
  ∃ n : ℕ, S n = 36 :=
by
  intros
  sorry

end max_sum_arith_seq_l641_641874


namespace sequence_exists_l641_641359

theorem sequence_exists :
  ∃ (seq : ℕ → ℕ), 
    (∀ i : ℕ, seq i = (i + 4)!) ∧ 
    (∀ i : ℕ, ∃ (divisors : fin (i+5) → ℕ), 
      (∀ r1 r2 : fin (i+5), r1 ≠ r2 → divisors r1 ≠ divisors r2) ∧
      (∀ r : fin (i+5), divisors r ∣ seq i ∧ finset.univ.sum (λ r, divisors r) = seq i)) :=
by sorry

end sequence_exists_l641_641359


namespace rise_in_water_level_is_correct_l641_641320

def volume_rectangular_prism (l w h : ℝ) : ℝ :=
  l * w * h

def radius_of_cylinder (diameter : ℝ) : ℝ :=
  diameter / 2

def rise_in_water_level (V : ℝ) (r : ℝ) : ℝ :=
  V / (Real.pi * r^2)

theorem rise_in_water_level_is_correct :
  let l := 15
  let w := 20
  let h := 21
  let diameter := 30
  let V_rect := volume_rectangular_prism l w h 
  let r := radius_of_cylinder diameter
  let h_rise := rise_in_water_level V_rect r
  abs (h_rise - 8.91) < 0.01 :=
by
  -- Provide the proof here
  sorry

end rise_in_water_level_is_correct_l641_641320


namespace industrial_plant_gizmos_l641_641974

noncomputable def labor_production (c d : ℕ) (workers gadgets gizmos hours : ℕ) : Prop :=
  workers * hours = gadgets * c + gizmos * d

theorem industrial_plant_gizmos :
  ∀ (c d : ℕ),
  (∃ c d,
    (labor_production c d 120 360 240 1) ∧
    (labor_production c d 40 240 360 3) ∧
    ∀ n, labor_production c d 60 240 n 4 → n = 0) :=
begin
  -- proof is skipped
  sorry
end

end industrial_plant_gizmos_l641_641974


namespace arrangement_is_114_l641_641772

namespace RoomArrangementProblem

variables (A B C D E : Type)

def arrangement_count : Nat :=
  -- Based on the conditions and solution derived
  114

theorem arrangement_is_114 :
  ∀ (A B C D E : Type), 
  ∃ (arrangements : Nat), arrangements = arrangement_count A B C D E :=
  by
  intros
  use arrangement_count A B C D E
  exact eq.refl (arrangement_count A B C D E)

end RoomArrangementProblem

end arrangement_is_114_l641_641772


namespace range_of_a_l641_641450

-- Define the sets A and C
def A : set ℝ := {x | x^2 - 6x + 5 < 0}
def C (a : ℝ) : set ℝ := {x | 3 * a - 2 < x ∧ x < 4 * a - 3}

-- Define the property that C is a subset of A
def subset_C_A (a : ℝ) : Prop := C(a) ⊆ A

-- Define the proof statement
theorem range_of_a : {a : ℝ | subset_C_A a} = {a : ℝ | a ≤ 2} :=
by {
    sorry
}

end range_of_a_l641_641450


namespace max_product_of_two_integers_sum_2000_l641_641199

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641199


namespace tangent_line_to_circle_l641_641598

noncomputable def equation_of_tangent_line (P : ℝ × ℝ) (circle_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), P.1 + 2 * P.2 + b = 0 ∧ ∀ x y, circle_eq x y = 0 → ‖⟨x, y⟩ - P‖ = 5

theorem tangent_line_to_circle :
  let P : ℝ × ℝ := (1, 2),
      circle_eq : ℝ → ℝ → ℝ := fun x y => x^2 + y^2 - 5 in
  equation_of_tangent_line P circle_eq :=
sorry

end tangent_line_to_circle_l641_641598


namespace locus_of_midpoint_l641_641401

open Real

noncomputable def circumcircle_eq (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := 1
  let b := 3
  let r2 := 5
  (a, b, r2)

theorem locus_of_midpoint (A B C N : ℝ × ℝ) :
  N = (6, 2) ∧ A = (0, 1) ∧ B = (2, 1) ∧ C = (3, 4) → 
  let P := (7 / 2, 5 / 2)
  let r2 := 5 / 4
  ∃ x y : ℝ, 
  (x, y) = P ∧ (x - 7 / 2)^2 + (y - 5 / 2)^2 = r2 :=
by sorry

end locus_of_midpoint_l641_641401


namespace both_firms_participate_l641_641062

-- Definitions based on the conditions
variable (V IC : ℝ) (α : ℝ)
-- Assumptions
variable (hα : 0 < α ∧ α < 1)
-- Part (a) condition transformation
def participation_condition := α * (1 - α) * V + 0.5 * α^2 * V ≥ IC

-- Given values for part (b)
def V_value : ℝ := 24
def α_value : ℝ := 0.5
def IC_value : ℝ := 7

-- New definitions for given values
def part_b_condition := (α_value * (1 - α_value) * V_value + 0.5 * α_value^2 * V_value) ≥ IC_value

-- Profits for part (c) comparison
def profit_when_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
def profit_when_one := α * V - IC

-- Proof problem statement in Lean 4
theorem both_firms_participate (hV : V = 24) (hα : α = 0.5) (hIC : IC = 7) :
    (α * (1 - α) * V + 0.5 * α^2 * V) ≥ IC ∧ profit_when_both V alpha IC > profit_when_one V α IC := by
  sorry

end both_firms_participate_l641_641062


namespace greatest_product_sum_2000_eq_1000000_l641_641238

noncomputable def maxProduct : ℕ :=
  let P : ℕ → ℕ := λ x, x * (2000 - x)
  1000000

theorem greatest_product_sum_2000_eq_1000000 :
  ∃ x y : ℕ, x + y = 2000 ∧ x * y = maxProduct :=
by
  use 1000, 1000
  simp [maxProduct]
  sorry

end greatest_product_sum_2000_eq_1000000_l641_641238


namespace max_product_two_integers_sum_2000_l641_641209

noncomputable def max_product_sum_2000 : ℤ :=
  let P := λ x : ℤ, x * (2000 - x)
  1000 * (2000 - 1000)

theorem max_product_two_integers_sum_2000 : ∃ x : ℤ, 0 ≤ x ∧ x ≤ 2000 ∧ (x * (2000 - x) = 1000000) :=
by {
  use 1000,
  split,
  { exact le_refl 1000, },
  split,
  { exact le_of_lt (show 1000 < 2000, by norm_num), },
  { norm_num, }
}

end max_product_two_integers_sum_2000_l641_641209


namespace ant_walks_distance_l641_641882

theorem ant_walks_distance :
  ∀ (A B C : ℝ × ℝ), 
  (A = (0, 0)) →
  (B = (0, 5)) →
  (C = (8, 5)) →
  let AB := 5
  let BC := 8
  let CA := Real.sqrt (AB^2 + BC^2)
  AB + BC + CA = 13 + Real.sqrt 89 :=
by
  intros A B C hA hB hC
  let AB := 5
  let BC := 8
  let CA := Real.sqrt (AB^2 + BC^2)
  have h1 : AB + BC + CA = 5 + 8 + Real.sqrt 89 := sorry
  rw [h1]
  rfl

end ant_walks_distance_l641_641882


namespace general_term_smallest_n_l641_641829

def a (n : ℕ) : ℕ
| 0     := 1
| 1     := 4
| (n+2) := 3 * a (n+1) - 2 * a n

def S (n : ℕ) : ℕ :=
  (List.range n).sum (λ k => a k)

theorem general_term (n : ℕ) : a n = 3 * 2^(n-1) - 2 := 
by
  sorry

theorem smallest_n (n : ℕ) (h : S n > 21 - 2 * n) : n = 4 :=
by
  have h1 : S 4 = 3 * (2^4 - 1) - 2 * 4 := sorry
  have h2 : S 3 = 3 * (2^3 - 1) - 2 * 3 := sorry
  have h3 : S 2 = 3 * (2^2 - 1) - 2 * 2 := sorry
  have h4 : S 1 = 3 * (2^1 - 1) - 2 * 1 := sorry
  have h5 : S 0 = 3 * (2^0 - 1) - 2 * 0 := sorry
  have shrink : 21 - 2 * 4 = 13 := sorry
  linarith

end general_term_smallest_n_l641_641829


namespace semicircle_radius_l641_641277

noncomputable def radius_of_semicircle (P : ℝ) (h : P = 144) : ℝ :=
  144 / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (h : P = 144) : radius_of_semicircle P h = 144 / (Real.pi + 2) :=
  by sorry

end semicircle_radius_l641_641277


namespace quadratic_has_real_roots_l641_641854

theorem quadratic_has_real_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (λ x, x^2 - x + k + 1) x₁ = 0 ∧ (λ x, x^2 - x + k + 1) x₂ = 0) ↔ k ≤ -3/4 := 
by
  sorry

end quadratic_has_real_roots_l641_641854


namespace MorseCodeDistinctSymbols_l641_641034

theorem MorseCodeDistinctSymbols:
  (1.sequence (λ _, bool).length = {1, 2, 3, 4, 5}).card = 62 :=
by
  sorry

end MorseCodeDistinctSymbols_l641_641034


namespace sum_of_reciprocals_le_30000_l641_641706

-- Define a condition that checks if a number contains the sequence "239"
def contains_sequence_239 (n : ℕ) : Prop :=
  n.digit_list.any (λ ds, (List.inits ds).any (λ sublist, sublist = [2, 3, 9]))

-- Define the problem statement
theorem sum_of_reciprocals_le_30000 (
  f : ℕ → ℕ,
  hf : ∀ n,  contains_sequence_239 (f n) = false
) : ∑' n, (1 : ℝ) / (f n) ≤ 30000 := 
sorry

end sum_of_reciprocals_le_30000_l641_641706


namespace most_stable_shooting_performance_l641_641388

theorem most_stable_shooting_performance {A B C D : ℕ} 
  (n_tests : ℕ) (same_avg : ∀ p, p ∈ {A, B, C, D} → True)
  (var_A var_B var_C var_D : ℝ) 
  (hA : var_A = 0.6)
  (hB : var_B = 1.1)
  (hC : var_C = 0.9)
  (hD : var_D = 1.2) :
  var_A < var_C ∧ var_A < var_B ∧ var_A < var_D :=
by {
  sorry
}

end most_stable_shooting_performance_l641_641388


namespace term_100_is_4_l641_641714

-- Definitions based on the conditions
def sequence : ℕ → ℕ
| 0       := 6
| (n + 1) :=
  if even (sequence n) then sequence n / 2
  else 3 * sequence n + 1

-- The statement of the theorem
theorem term_100_is_4 : sequence 99 = 4 :=
sorry

end term_100_is_4_l641_641714


namespace volume_of_substance_l641_641618

theorem volume_of_substance (k : ℝ) (V W : ℝ) (h1 : V = k * W) (h2 : 48 = k * 112) :
  (∃ V₁ : ℝ, V₁ = k * 56 ∧ V₁ = 24) :=
by
  have k_val: k = 3 / 7 := by linarith
  use (3 / 7) * 56
  split
  . exact h1
  . linarith

end volume_of_substance_l641_641618


namespace last_three_digits_7_pow_103_l641_641373

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l641_641373


namespace total_cost_is_correct_l641_641946

def cost_shirt (S : ℝ) : Prop := S = 12
def cost_shoes (Sh S : ℝ) : Prop := Sh = S + 5
def cost_dress (D : ℝ) : Prop := D = 25
def discount_shoes (Sh Sh' : ℝ) : Prop := Sh' = Sh - 0.10 * Sh
def discount_dress (D D' : ℝ) : Prop := D' = D - 0.05 * D
def cost_bag (B twoS Sh' D' : ℝ) : Prop := B = (twoS + Sh' + D') / 2
def total_cost_before_tax (T_before twoS Sh' D' B : ℝ) : Prop := T_before = twoS + Sh' + D' + B
def sales_tax (tax T_before : ℝ) : Prop := tax = 0.07 * T_before
def total_cost_including_tax (T_total T_before tax : ℝ) : Prop := T_total = T_before + tax
def convert_to_usd (T_usd T_total : ℝ) : Prop := T_usd = T_total * 1.18

theorem total_cost_is_correct (S Sh D Sh' D' twoS B T_before tax T_total T_usd : ℝ) :
  cost_shirt S →
  cost_shoes Sh S →
  cost_dress D →
  discount_shoes Sh Sh' →
  discount_dress D D' →
  twoS = 2 * S →
  cost_bag B twoS Sh' D' →
  total_cost_before_tax T_before twoS Sh' D' B →
  sales_tax tax T_before →
  total_cost_including_tax T_total T_before tax →
  convert_to_usd T_usd T_total →
  T_usd = 119.42 :=
by
  sorry

end total_cost_is_correct_l641_641946


namespace circle_equation_l641_641418

theorem circle_equation 
  (h k : ℝ) 
  (H_center : k = 2 * h)
  (H_tangent : ∃ (r : ℝ), (h - 1)^2 + (k - 0)^2 = r^2 ∧ r = k) :
  (x - 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_l641_641418


namespace train_travel_time_l641_641606

variable (x : ℝ)

theorem train_travel_time : 
  let distance := 236
  let speed_D := x - 40
  let time_difference := 1/4
  (distance / speed_D - distance / x = time_difference) :=
by
  let distance := 236
  let speed_D := x - 40
  let time_difference := 1/4
  show (distance / speed_D - distance / x = time_difference) from sorry

end train_travel_time_l641_641606


namespace simplify_and_evaluate_l641_641575

theorem simplify_and_evaluate
  (a b : ℝ)
  (h : |a - 1| + (b + 2)^2 = 0) :
  ((2 * a + b)^2 - (2 * a + b) * (2 * a - b)) / (-1 / 2 * b) = 0 := 
sorry

end simplify_and_evaluate_l641_641575


namespace max_product_of_sum_2000_l641_641246

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641246


namespace rhombus_constant_product_point_C_trajectory_l641_641881

theorem rhombus_constant_product (A B C D O : Point) (hRhombus : Rhombus A B C D) (hSideLen : ∀ a b, SideLen a b = 4) 
  (hOB : dist O B = 6) (hOD : dist O D = 6) :
  dist O A * dist O C = 20 :=
sorry

theorem point_C_trajectory (A C : Point) (x y : ℝ) 
  (hSemicircle : ∀ (A : Point), in_semicircle A ∧ 2 ≤ x ∧ x ≤ 4) :
  ∃ y, (A = (2 + 2 * cos y, 2 * sin y) → C = (5, 5 * tan (y / 2))) ∧ -5 ≤ y ∧ y ≤ 5 :=
sorry

end rhombus_constant_product_point_C_trajectory_l641_641881


namespace isosceles_triangle_length_l641_641125

theorem isosceles_triangle_length (a : ℝ) (h_graph_A : ∃ y, (a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2})
  (h_graph_B : ∃ y, (-a, y) ∈ {p : ℝ × ℝ | p.snd = -p.fst^2}) 
  (h_isosceles : ∃ O : ℝ × ℝ, O = (0, 0) ∧ 
    dist (a, -a^2) O = dist (-a, -a^2) O ∧ dist (a, -a^2) (-a, -a^2) = dist (-a, -a^2) O) :
  dist (a, -a^2) (0, 0) = 2 * Real.sqrt 3 := sorry

end isosceles_triangle_length_l641_641125


namespace solve_for_y_l641_641947

theorem solve_for_y (y : ℝ) : 3^y + 9 = 4 * 3^y - 44 → y = Real.logb 3 53 - 1 :=
by
  intro h
  sorry

end solve_for_y_l641_641947


namespace total_books_proof_l641_641571

noncomputable def economics_books (T : ℝ) := (1/4) * T + 10
noncomputable def rest_books (T : ℝ) := T - economics_books T
noncomputable def social_studies_books (T : ℝ) := (3/5) * rest_books T - 5
noncomputable def other_books := 13
noncomputable def science_books := 12
noncomputable def total_books_equation (T : ℝ) :=
  T = economics_books T + social_studies_books T + science_books + other_books

theorem total_books_proof : ∃ T : ℝ, total_books_equation T ∧ T = 80 := by
  sorry

end total_books_proof_l641_641571


namespace find_q_sum_b_q1_sum_b_q_neg_half_l641_641397

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
a 1 = 1 ∧ (∀ n ≥ 3, a n = (a (n-1) + a (n-2)) / 2) ∧ (∀ n, a n = a 1 * q^(n-1))

theorem find_q : 
  ∃ q, (∀ a, geometric_sequence a q → (q = 1 ∨ q = -1/2)) :=
sorry

noncomputable def sequence_b (a : ℕ → ℝ) (n : ℕ) : ℕ → ℝ := 
λ n, n * a n

noncomputable def sum_sequence (f : ℕ → ℝ) (n : ℕ) : ℝ := 
(n + 1) * n / 2

theorem sum_b_q1 (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) (q : ℝ) : 
  geometric_sequence a q → q = 1 → 
  sum_sequence b n = (n * (n + 1)) / 2 :=
sorry

noncomputable def sum_b_q_neg_half (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := 
∑ i in range n, i * (-1/2)^(i-1)

theorem sum_b_q_neg_half (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) (q : ℝ) : 
  geometric_sequence a q → q = -1/2 → 
  sum_sequence b n = (4 / 9) - ((4 / 9) + (2 * n / 3)) * (-1/2)^n :=
sorry

end find_q_sum_b_q1_sum_b_q_neg_half_l641_641397


namespace ellipse_equation_shared_foci_lines_through_fixed_point_minimum_triangle_area_l641_641405

-- Definitions and conditions
noncomputable def is_ellipse (a b c : ℝ) (e : ℝ) : Prop :=
  c / a = e ∧ a > b ∧ b > 0

noncomputable def ellipse_standard_eq (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2 / a^2 + x^2 / b^2 = 1)

noncomputable def is_hyperbola (x y : ℝ) : Prop := 
  y^2 - x^2 = 1

noncomputable def foci_shared (f1 f2 : (ℝ × ℝ)) : Prop :=
  f1 = (0, sqrt 2) ∧ f2 = (0, -sqrt 2)

-- Theorems to be proved
theorem ellipse_equation_shared_foci (e : ℝ) (C : ℝ × ℝ) (H : ℝ × ℝ ) :
  foci_shared C H ∧ is_ellipse (sqrt 3) 1 (sqrt 3) e → 
  ellipse_standard_eq (sqrt 3) 1 :=
sorry

theorem lines_through_fixed_point (A M N : ℝ × ℝ) :
  (A = (0, -sqrt 3)) ∧ (M ≠ A) ∧ (N ≠ A) ∧ 
  (∃ (k₁ k₂ : ℝ), k₁ * k₂ = -3) →
  ∃ (P : ℝ × ℝ), P = (0, 0) ∧ (M, N) passes through (0, 0) :=
sorry

theorem minimum_triangle_area (A M N P : ℝ × ℝ) :
  (A = (0, -sqrt 3)) ∧ (M ≠ A) ∧ (N ≠ A) ∧ 
  (P ≠ M) ∧ (P ≠ N) ∧ (|MP| = |NP|) →
  ∃ (area : ℝ), area = 3 / 2 :=
sorry

end ellipse_equation_shared_foci_lines_through_fixed_point_minimum_triangle_area_l641_641405


namespace find_a_l641_641809

theorem find_a (a : ℝ)
  (h1 : ∀ x y, y = \frac{a x + a^2 + 1}{x + a - 1} ↔ x = y)
  (h2 : ∃ x y, (x, y) = (3, -2) ⇒ (y, x) lies on C) : 
  a = 3 :=
sorry

end find_a_l641_641809


namespace problem1_problem2_l641_641733

-- Problem 1 proof statement
theorem problem1 : sqrt 4 - (1 / 2)⁻¹ + (2 - 1 / 7) ^ 0 = 1 := by
  sorry

-- Problem 2 proof statement
theorem problem2 (a : ℝ) (h : a ≠ 0) : (a - 1 / a) / ((a^2 + 2 * a + 1) / a) = (a - 1) / (a + 1) := by
  sorry

end problem1_problem2_l641_641733


namespace min_value_of_expression_l641_641408

theorem min_value_of_expression (a b c d : ℤ) (h : a * d - b * c = 1) :
  ∃ (s : Finset (ℤ × ℤ × ℤ × ℤ)),
    (∀ x ∈ s, (let (a, b, c, d) := x in 
                 a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 + a * b + c * d - a * c - b * d - b * c) = 2) ∧
    (∀ (a b c d : ℤ), 
       (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 + a * b + c * d - a * c - b * d - b * c ≥ 2)) :=
sorry

end min_value_of_expression_l641_641408


namespace difference_of_triangle_areas_l641_641499

variables {A B C D F : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace F]

-- Define lengths
variables (AB BC AF : ℝ)
-- Required assumptions
variable (angle_FAB_right : is_right_angle (∠ F A B))
variable (angle_ABC_right : is_right_angle (∠ A B C))
variable (AB_eq_5 : AB = 5)
variable (BC_eq_7 : BC = 7)
variable (AF_eq_10 : AF = 10)
variable (AC_BF_intersect_D : ∃ D, collinear_three_points A C D ∧ collinear_three_points B F D ∧ noncollinear_three_points A B D)

-- Areas of triangles
noncomputable def area_triangle_ADF : ℝ := sorry
noncomputable def area_triangle_BDC : ℝ := sorry
noncomputable def area_triangle_ABF : ℝ := 1/2 * AB * AF
noncomputable def area_triangle_BAC : ℝ := 1/2 * AB * BC

theorem difference_of_triangle_areas :
  area_triangle_ADF - area_triangle_BDC = 7.5 :=
sorry

end difference_of_triangle_areas_l641_641499


namespace sum_proper_divisors_81_l641_641640

theorem sum_proper_divisors_81 : ∑ d in {1, 3, 9, 27}, d = 40 := 
by sorry

end sum_proper_divisors_81_l641_641640


namespace greatest_product_sum_2000_l641_641217

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641217


namespace energy_increase_l641_641597

-- Defining the problem conditions and stating the theorem.

variables {E V : Type}
variables [linear_ordered_field E] [discrete_field E]
variables [metric_space V] [normed_group V] [normed_space E V]

-- Given conditions
def energy_inversely_proportional (E : V → V → E) : Prop :=
∀ (q1 q2 : V) (d1 d2 : E), (d1 = dist q1 q2) → (E q1 q2 = k / d1)

def energy_directly_proportional (E : V → V → E) : Prop :=
∀ (q1 q2 : V) (Q1 Q2 : E), (Q q1 = Q1) → (Q q2 = Q2) → (E q1 q2 = k * Q1 * Q2)

def square_configuration (charges : list V) (E_total : E) : Prop :=
charges.length = 4 ∧ 
∀ (i j : ℕ) (hi : i < 4) (hj : j < 4) (hij : i ≠ j), 
  let q1 := charges.nth_le i hi, 
  let q2 := charges.nth_le j hj in
  E q1 q2 = E_total / 6

-- Assertion to prove
theorem energy_increase (E : V → V → E) (charges : list V) (E_new : E) :
  energy_inversely_proportional E → 
  energy_directly_proportional E → 
  square_configuration charges 20 →
  let q_center := (charges.nth_le 0) in
  let center := (charges 0 + charges 1 + charges 2 + charges 3) / 4 in
  E q_center center = E_new →
  E_new - 20 = 20*(sqrt 2 - 1) := 
sorry

end energy_increase_l641_641597


namespace find_rate_of_current_l641_641996

-- Define the conditions
def speed_in_still_water (speed : ℝ) : Prop := speed = 15
def distance_downstream (distance : ℝ) : Prop := distance = 7.2
def time_in_hours (time : ℝ) : Prop := time = 0.4

-- Define the effective speed downstream
def effective_speed_downstream (boat_speed current_speed : ℝ) : ℝ := boat_speed + current_speed

-- Define rate of current
def rate_of_current (current_speed : ℝ) : Prop :=
  ∃ (c : ℝ), effective_speed_downstream 15 c * 0.4 = 7.2 ∧ c = current_speed

-- The theorem stating the proof problem
theorem find_rate_of_current : rate_of_current 3 :=
by
  sorry

end find_rate_of_current_l641_641996


namespace probability_sum_9_l641_641275

def set_r : Set ℕ := {2, 3, 4, 5}
def set_b : Set ℕ := {4, 5, 6, 7, 8}

def successful_pairs : Set (ℕ × ℕ) := 
  { (x, y) | x ∈ set_r ∧ y ∈ set_b ∧ x + y = 9 }

def total_pairs_count : ℕ := set_r.card * set_b.card
def successful_pairs_count : ℕ := successful_pairs.toFinset.card

theorem probability_sum_9 : 
  (successful_pairs_count : ℚ) / (total_pairs_count : ℚ) = 3 / 20 := 
sorry

end probability_sum_9_l641_641275


namespace find_digit_B_in_combination_l641_641485

theorem find_digit_B_in_combination :
  let combination := (nat.choose 60 12)
  (2250500500 ≤ combination) ∧ (combination < 2260000000) → 
  (combination / 10000000 % 10 = 5)

proof
  sorry

end find_digit_B_in_combination_l641_641485


namespace product_roots_l641_641802

noncomputable def root1 (x1 : ℝ) : Prop := x1 * Real.log x1 = 2006
noncomputable def root2 (x2 : ℝ) : Prop := x2 * Real.exp x2 = 2006

theorem product_roots (x1 x2 : ℝ) (h1 : root1 x1) (h2 : root2 x2) : x1 * x2 = 2006 := sorry

end product_roots_l641_641802


namespace find_b_squared_l641_641705

-- Definition of the complex number function
def f (a b : ℝ) (z : ℂ) : ℂ := (a + b * complex.I) * z

-- Conditions: a, b are positive numbers; |a + bi| = 10; a = 1
variables (a b : ℝ) (h1 : a = 1) (h2 : |a + b * complex.I| = 10)
-- Given this information, we need to prove b^2 = 100
theorem find_b_squared : b^2 = 100 := 
sorry

end find_b_squared_l641_641705


namespace both_firms_participate_l641_641063

-- Definitions based on the conditions
variable (V IC : ℝ) (α : ℝ)
-- Assumptions
variable (hα : 0 < α ∧ α < 1)
-- Part (a) condition transformation
def participation_condition := α * (1 - α) * V + 0.5 * α^2 * V ≥ IC

-- Given values for part (b)
def V_value : ℝ := 24
def α_value : ℝ := 0.5
def IC_value : ℝ := 7

-- New definitions for given values
def part_b_condition := (α_value * (1 - α_value) * V_value + 0.5 * α_value^2 * V_value) ≥ IC_value

-- Profits for part (c) comparison
def profit_when_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
def profit_when_one := α * V - IC

-- Proof problem statement in Lean 4
theorem both_firms_participate (hV : V = 24) (hα : α = 0.5) (hIC : IC = 7) :
    (α * (1 - α) * V + 0.5 * α^2 * V) ≥ IC ∧ profit_when_both V alpha IC > profit_when_one V α IC := by
  sorry

end both_firms_participate_l641_641063


namespace gretchen_fewest_trips_l641_641454

def fewestTrips (total_objects : ℕ) (max_carry : ℕ) : ℕ :=
  (total_objects + max_carry - 1) / max_carry

theorem gretchen_fewest_trips : fewestTrips 17 3 = 6 := 
  sorry

end gretchen_fewest_trips_l641_641454


namespace num_multiples_of_5_divisors_l641_641458

theorem num_multiples_of_5_divisors (a b c d : ℕ) (n : ℕ) 
  (h_a : 0 ≤ a ∧ a ≤ 1) 
  (h_b : 0 ≤ b ∧ b ≤ 2) 
  (h_c : c = 1) 
  (h_d : 0 ≤ d ∧ d ≤ 2) 
  (h_n : n = 2 ^ a * 3 ^ b * 5 ^ c * 7 ^ d) : 
  (finset.filter (λ x, 5 ∣ x) (finset.divisors 4410)).card = 18 := 
begin
  sorry
end

end num_multiples_of_5_divisors_l641_641458


namespace remainder_abc_div_n_l641_641523

theorem remainder_abc_div_n (n : ℕ) (a b c : ℤ) 
  (h1 : b % n ≠ 0) (h2 : c % n ≠ 0) (h3 : (a * c) % n = (b⁻¹) % n) : (a * b * c) % n = 1 % n :=
by sorry

end remainder_abc_div_n_l641_641523


namespace distribution_plans_equiv_210_l641_641727

noncomputable def number_of_distribution_plans : ℕ := sorry -- we will skip the proof

theorem distribution_plans_equiv_210 :
  number_of_distribution_plans = 210 := by
  sorry

end distribution_plans_equiv_210_l641_641727


namespace asymptote_equation_l641_641810

theorem asymptote_equation 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (x y : ℝ)
  (hyperbola : x^2 / a^2 - y^2 / b^2 = 1) 
  (parabola : y^2 = 8 * x) 
  (focus_shared : true) -- representing the common focus fact and having coordinates (2,0)
  (intersection : true) -- representing the intersection fact
  (distance_pf : |sqrt( (x - 2)^2  + y^2 )| = 5) : 
  x + y = 0 ∨ x - y = 0 :=
sorry 

end asymptote_equation_l641_641810


namespace domain_of_log_function_l641_641958

theorem domain_of_log_function (x : ℝ) :
  (-1 < x ∧ x < 1) ↔ (1 - x) / (1 + x) > 0 :=
by sorry

end domain_of_log_function_l641_641958


namespace find_tan_beta_l641_641393

variables (π α β : ℝ)

-- Given conditions
axiom tan_pi_minus_alpha : tan (π - α) = - (1 / 5)
axiom tan_alpha_minus_beta : tan (α - β) = 1 / 3

-- Prove the statement
theorem find_tan_beta : tan β = - (1 / 8) :=
by
  -- conditions assumed as axioms
  assume π α β
  exact sorry -- proof would be here

end find_tan_beta_l641_641393


namespace manuscript_pages_l641_641994

theorem manuscript_pages (P : ℕ) (rate_first : ℕ) (rate_revision : ℕ) 
  (revised_once_pages : ℕ) (revised_twice_pages : ℕ) (total_cost : ℕ) :
  rate_first = 6 →
  rate_revision = 4 →
  revised_once_pages = 35 →
  revised_twice_pages = 15 →
  total_cost = 860 →
  6 * (P - 35 - 15) + 10 * 35 + 14 * 15 = total_cost →
  P = 100 :=
by
  intros h_first h_revision h_once h_twice h_cost h_eq
  sorry

end manuscript_pages_l641_641994


namespace minimum_components_needed_l641_641700

-- Define the parameters of the problem
def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 7
def fixed_monthly_cost : ℝ := 16500
def selling_price_per_component : ℝ := 198.33

-- Define the total cost as a function of the number of components
def total_cost (x : ℝ) : ℝ :=
  fixed_monthly_cost + (production_cost_per_component + shipping_cost_per_component) * x

-- Define the revenue as a function of the number of components
def revenue (x : ℝ) : ℝ :=
  selling_price_per_component * x

-- Define the theorem to be proved
theorem minimum_components_needed (x : ℝ) : x = 149 ↔ total_cost x ≤ revenue x := sorry

end minimum_components_needed_l641_641700


namespace max_product_of_two_integers_sum_2000_l641_641201

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641201


namespace total_tiles_144_l641_641120

-- Define the dimensions of the dining room
def diningRoomLength : ℕ := 15
def diningRoomWidth : ℕ := 20

-- Define the border width using 1x1 tiles
def borderWidth : ℕ := 2

-- Area of each 3x3 tile
def tileArea : ℕ := 9

-- Calculate the dimensions of the inner area after the border
def innerAreaLength : ℕ := diningRoomLength - 2 * borderWidth
def innerAreaWidth : ℕ := diningRoomWidth - 2 * borderWidth

-- Calculate the area of the inner region
def innerArea : ℕ := innerAreaLength * innerAreaWidth

-- Calculate the number of 3x3 tiles
def numThreeByThreeTiles : ℕ := (innerArea + tileArea - 1) / tileArea -- rounded up division

-- Calculate the number of 1x1 tiles for the border
def numOneByOneTiles : ℕ :=
  2 * (innerAreaLength + innerAreaWidth + 4 * borderWidth)

-- Total number of tiles
def totalTiles : ℕ := numOneByOneTiles + numThreeByThreeTiles

-- Prove that the total number of tiles is 144
theorem total_tiles_144 : totalTiles = 144 := by
  sorry

end total_tiles_144_l641_641120


namespace roots_form_arithmetic_sequence_and_f_8_l641_641969

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^5 - 20 * x^4 + a * x^3 + b * x^2 + c * x + 24

theorem roots_form_arithmetic_sequence_and_f_8 (a b c : ℝ) :
  (∃ m n : ℝ, f(x) = (x - (m - 2 * n))(x - (m - n))(x - m)(x - (m + n))(x - (m + 2 * n)) ∧ 5 * m = 20 ∧ (16 - 4 * n^2) * (16 - n^2) = -6) →
  f 8 a b c = -24 :=
by
  sorry

end roots_form_arithmetic_sequence_and_f_8_l641_641969


namespace hexagon_angle_proof_l641_641721

noncomputable def hexagon_angle : ℕ := 150

theorem hexagon_angle_proof :
  ∀ {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F],
  (∀ (x y z w u v : ℚ), 
  (dist x y = dist y z ∧ dist z w = dist w u ∧ dist u v = dist v x) ∧ 
  (angle A B C = 90 ∧ angle B C D = 90 ∧ angle C D E = 90) ∧ 
  (convex_polygon [A, B, C, D, E, F])) → 
  angle E F D = hexagon_angle := sorry

end hexagon_angle_proof_l641_641721


namespace range_f_lt_0_l641_641021

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≥ f y
def f_at_2 (f : ℝ → ℝ) : Prop := f 2 = 0

-- The main statement to prove
theorem range_f_lt_0 :
  is_even_function f →
  is_decreasing_on f (Iio 0 ∪ {0}) →
  f_at_2 f →
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 :=
by
  intros h_even h_decreasing h_f_at_2
  sorry

end range_f_lt_0_l641_641021


namespace no_nuples_solution_l641_641764

open Matrix

theorem no_nuples_solution :
  ∀ (a b c d e f g h i : ℝ),
  (matrix.mul 
      ![![a, b, c], ![d, e, f], ![g, h, i]] 
      ![![1/a, 1/b, 1/c], ![1/d, 1/e, 1/f], ![1/g, 1/h, 1/i]] 
  = 1) → false :=
begin
  intros a b c d e f g h i hM,
  sorry
end

end no_nuples_solution_l641_641764


namespace approximate_value_0_991_pow_5_l641_641581

noncomputable def approximateBinomial (x : ℝ) (n : ℕ) (k : ℕ) : ℝ :=
(nat.choose n k) * x^k * (1 - x)^(n - k)

noncomputable def approximatePower (x : ℝ) (n : ℕ) : ℝ :=
∑ k in finset.range (n + 1), approximateBinomial x n k

noncomputable def round_to (x : ℝ) (precision : ℝ) : ℝ :=
Float.round (x / precision) * precision

theorem approximate_value_0_991_pow_5 : round_to (approximatePower (0.009) 5) 0.001 = 0.956 := 
by
  sorry

end approximate_value_0_991_pow_5_l641_641581


namespace solution_set_l641_641758

def floor (x : ℝ) : ℤ := Int.floor x

def satisfies_equation (x : ℝ) : Prop :=
  floor (floor (3 * x : ℤ) - 3 / 2) = floor (x + 3)

theorem solution_set :
  ∀ x : ℝ, satisfies_equation x ↔ (7 / 3 ≤ x ∧ x < 8 / 3) :=
by sorry

end solution_set_l641_641758


namespace election_winner_votes_l641_641278

theorem election_winner_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 360) :
  0.62 * V = 930 :=
by {
  sorry
}

end election_winner_votes_l641_641278


namespace domain_of_g_g_x_smallest_x_test_l641_641096

noncomputable def g (x : ℝ) : ℝ := real.sqrt (x - 5)

theorem domain_of_g_g_x :
  ∀ x, (x ≥ 30) ↔ (∃ y, (y = g x) ∧ (y ≥ 5)) :=
by
  intros x
  split
  {
    intro h
    use g x
    split
    {
      refl
    }
    {
      exact (real.sqrt_nonneg (x - 5)).trans (le_of_eq h.symm)
    }
  }
  {
    rintros ⟨y, hy1, hy2⟩
    rw [hy1, ← real.sqrt_inj_of_nonneg] at hy2
    {
      assume : x ≥ 5,
      exact (le_of_eq (hy2.trans (by linarith)))
    }
    all_goals
    {
      exact (real.sqrt_nonneg _)
    }
  }

lemma smallest_x_in_domain :
  ∃ x, (x ≥ 30) ∧ (∀ y, y < 30 → ¬(∃ z, (g z = y) ∧ (y ≥ 5))) := 
by 
  use 30
  split
  {
    linarith
  }
  {
    intros y hy,
    rintros ⟨z, hz1, hz2⟩,
    have := domain_of_g_g_x z
    rw [hz1] at this,
    exact hy.not_le this
  }

theorem smallest_x_test :
  ∃ x, (∃ y, (y = g x) ∧ (y ≥ 5)) ∧ ∀ z, z < 30 → ¬(∃ y, (y = g z) ∧ (y ≥ 5)) :=
by
  use 30
  exact smallest_x_in_domain

end domain_of_g_g_x_smallest_x_test_l641_641096


namespace find_a_l641_641920

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 0 then Real.log x else x + ∫ t in 0..a, 3 * t^2

theorem find_a (a : ℝ) : f (f 1 a) a = 1 → a = 1 :=
by
  intro h
  have h1 : f 1 a = 0 := by
    simp [f]
    rw [if_pos]
    exact Real.log_one
  rw [h1] at h
  simp [f] at h
  exact sorry

end find_a_l641_641920


namespace total_dividends_correct_l641_641655

-- Conditions
def net_profit (total_income expenses loan_penalty_rate : ℝ) : ℝ :=
  let net1 := total_income - expenses
  let loan_penalty := net1 * loan_penalty_rate
  net1 - loan_penalty

def total_loan_payments (monthly_payment months additional_payment : ℝ) : ℝ :=
  (monthly_payment * months) - additional_payment

def dividend_per_share (net_profit total_loan_payments num_shares : ℝ) : ℝ :=
  (net_profit - total_loan_payments) / num_shares

noncomputable def total_dividends_director (dividend_per_share shares_owned : ℝ) : ℝ :=
  dividend_per_share * shares_owned

theorem total_dividends_correct :
  total_dividends_director (dividend_per_share (net_profit 1500000 674992 0.2) (total_loan_payments 23914 12 74992) 1000) 550 = 246400 :=
sorry

end total_dividends_correct_l641_641655


namespace sum_of_tens_ones_digits_of_7_pow_15_l641_641263

theorem sum_of_tens_ones_digits_of_7_pow_15 :
  let n := 7^15 in (n / 10 % 10) + (n % 10) = 7 :=
by
  sorry

end sum_of_tens_ones_digits_of_7_pow_15_l641_641263


namespace parallelogram_vector_sum_l641_641876

theorem parallelogram_vector_sum (A B C D : ℝ × ℝ) (parallelogram : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D ∧ (C - A = D - B) ∧ (B - D = A - C)) :
  (B - A) + (C - B) = C - A :=
by
  sorry

end parallelogram_vector_sum_l641_641876


namespace rafael_weekly_earnings_l641_641131

def weekly_hours (m t r : ℕ) : ℕ := m + t + r
def total_earnings (hours wage : ℕ) : ℕ := hours * wage

theorem rafael_weekly_earnings : 
  ∀ (m t r wage : ℕ), m = 10 → t = 8 → r = 20 → wage = 20 → total_earnings (weekly_hours m t r) wage = 760 :=
by
  intros m t r wage hm ht hr hw
  rw [hm, ht, hr, hw]
  simp only [weekly_hours, total_earnings]
  sorry -- Proof needs to be completed

end rafael_weekly_earnings_l641_641131


namespace tangent_line_inverse_common_points_compare_values_l641_641436

noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- (Ⅰ) Prove that if the line y = kx + 1 is tangent to the graph of the inverse of f(x), then k = 1/e^2. -/
theorem tangent_line_inverse (k : ℝ) : (k = 1 / Real.exp 2) ↔ ∃ x₀ : ℝ, y₀ : ℝ, (x₀ > 0 ∧ y₀ = Real.log x₀ ∧ k = 1 / x₀ ∧ -1 + Real.log x₀ = 1) := 
sorry

/-- (Ⅱ) Given x > 0, discuss the number of common points between the curve y = e^x / x^2 and the line y = m (m > 0). -/
theorem common_points (x m : ℝ) (hx : x > 0) (hm : m > 0) :
  ((0 < m ∧ m < Real.exp 2 / 4) ↔ ∀ x > 0, f x / x^2 ≠ m) ∧
  (m = Real.exp 2 / 4 ↔ ∃ x > 0, f x / x^2 = m) ∧
  (m > Real.exp 2 / 4 ↔ ∃ x1 x2 > 0, (x1 ≠ x2) ∧ f x1 / x1^2 = m ∧ f x2 / x2^2 = m) :=
sorry

/-- (Ⅲ) Let a < b, compare the values of f((a + b) / 2) and (f b - f a) / (b - a). -/
theorem compare_values (a b : ℝ) (ha : a < b) :
  f ((a + b) / 2) < (f b - f a) / (b - a) :=
sorry

end tangent_line_inverse_common_points_compare_values_l641_641436


namespace divisors_count_l641_641918

def n : ℕ := 2^29 * 5^21
def n_squared : ℕ := n^2

def count_divisors (m : ℕ) : ℕ :=
  (m.factors.nodup.length.succ).prod

def count_factors_less_than (m k : ℕ) : ℕ :=
  ((m.factors.nodup.length.succ).prod - 1) / 2

def count_divisors_not_dividing (m k : ℕ) : ℕ :=
  count_factors_less_than (m^2) m - count_divisors m

theorem divisors_count : count_divisors_not_dividing n n = 608 := by
  sorry

end divisors_count_l641_641918


namespace numPermutationsOfDigits_l641_641180

def numUniqueDigitPermutations : Nat :=
  Nat.factorial 4

theorem numPermutationsOfDigits : numUniqueDigitPermutations = 24 := 
by
  -- proof goes here
  sorry

end numPermutationsOfDigits_l641_641180


namespace ratio_of_speed_l641_641720

variables (L : ℝ) (v_t v_c : ℝ)

def t1 := (4 / 9 * L) / v_t
def t2 := (4 / 9 * L) / v_t
def t3 := L / v_c
def t4 := (1 / 9 * L) / v_t

theorem ratio_of_speed (h1 : t1 = t1)
                       (h2 : t3 = 8 / 9 * L / v_t) :
  v_c = 9 * v_t :=
by {
  have h3: v_c = 9 * v_t, from sorry,
  exact h3,
}

end ratio_of_speed_l641_641720


namespace Misha_l641_641660

theorem Misha's_decision_justified :
  let A_pos := 7 in
  let A_neg := 4 in
  let B_pos := 4 in
  let B_neg := 1 in
  (B_pos / (B_pos + B_neg) > A_pos / (A_pos + A_neg)) := 
sorry

end Misha_l641_641660


namespace justify_misha_decision_l641_641651

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l641_641651


namespace one_third_of_1206_is_300_percent_of_134_l641_641681

-- Definitions based on conditions in part a)
def one_third (n : ℕ) : ℕ := n / 3
def percentage (part whole : ℕ) : ℕ := (part * 100) / whole

-- Main statement proofing the problem
theorem one_third_of_1206_is_300_percent_of_134 : percentage (one_third 1206) 134 = 300 := by
  -- Conditions based on the problem requirements
  have h1 : one_third 1206 = 402 :=
    by sorry
  have h2 : percentage 402 134 = 300 :=
    by sorry
  -- With the above conditions, proving the main statement directly
  rw [h1, h2]
  sorry

end one_third_of_1206_is_300_percent_of_134_l641_641681


namespace three_cyclists_distance_l641_641174

theorem three_cyclists_distance :
  ∀ (t : ℝ) (x y z : ℝ) (hx : 0 ≤ x ∧ x < 2 * π) (hy : 0 ≤ y ∧ y < 2 * π) (hz : 0 ≤ z ∧ z < 2 * π),
  (hx ≠ hy ∧ hy ≠ hz ∧ hz ≠ hx) →
  let dxy := abs (x - y) % (2 * π)
  let dyz := abs (y - z) % (2 * π)
  let dzx := abs (z - x) % (2 * π)
  ¬(dxy > π ∧ dyz > π ∧ dzx > π) := 
begin
  sorry, -- Proof goes here
end

end three_cyclists_distance_l641_641174


namespace num_lattice_points_on_triangle_perimeter_l641_641708

def is_lattice_point (p : ℤ × ℤ) : Prop :=
  ∃ (x y : ℤ), p = (x, y)

def on_segment (p1 p2 : ℤ × ℤ) (p : ℤ × ℤ) : Prop :=
  ∃ (t : ℚ), 0 ≤ t ∧ t ≤ 1 ∧ (p.1 : ℚ) = p1.1 + t * (p2.1 - p1.1) ∧ (p.2 : ℚ) = p1.2 + t * (p2.2 - p1.2)

def on_perimeter (vertices : list (ℤ × ℤ)) (p : ℤ × ℤ) : Prop :=
  ∃ (v1 v2 : ℤ × ℤ), 
    v1 ∈ vertices ∧ 
    v2 ∈ vertices ∧ 
    v1 ≠ v2 ∧ 
    on_segment v1 v2 p

def count_lattice_points_on_perimeter (vertices : list (ℤ × ℤ)) : ℕ :=
  (list.finset (λ p, is_lattice_point p ∧ on_perimeter vertices p)).card

theorem num_lattice_points_on_triangle_perimeter :
  count_lattice_points_on_perimeter [(2, 1), (12, 21), (17, 6)] = 20 :=
by sorry

end num_lattice_points_on_triangle_perimeter_l641_641708


namespace sum_of_digits_smallest_N_l641_641901

/-- Proof Statement: The sum of digits of the smallest positive integer N,
such that N + 2N + 3N + ... + 9N is a number all of whose digits are equal, is 37. -/
theorem sum_of_digits_smallest_N (N : ℕ) (h : ∃ (N : ℕ), (N > 0) ∧ 
  (∀ m < N, N + 2 * N + 3 * N + 4 * N + 5 * N + 6 * N + 7 * N + 8 * N + 9 * N = 
  ∑ i in (finset.range 10).filter(λ x, x ≠ 0), i * N) ∧ (N + 2 * N + 3 * N + 4 * N + 5 * N + 6 * N + 7 * N + 8 * N + 9 * N = 
  ((10^9 - 1) / 9) * some divisor_of_405)) : 
  (N = 12345679) → sum_of_digits N = 37 :=
  by
  sorry

end sum_of_digits_smallest_N_l641_641901


namespace max_product_of_sum_2000_l641_641240

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641240


namespace complex_solution_count_l641_641003

open Complex

theorem complex_solution_count :
  {z : ℂ | abs z < 20 ∧ exp z = (z - 2) / (z + 2)}.finite.card = 8 :=
by sorry

end complex_solution_count_l641_641003


namespace moles_of_amyl_alcohol_combined_l641_641361

-- Definition of the problem including conditions
def reaction_equation := "C5H11OH + HCl → C5H11Cl + H2O"
def mass_of_water : Real := 18 -- in grams
def molar_mass_of_water : Real := 18.015 -- in g/mol

-- Stating the theorem to prove
theorem moles_of_amyl_alcohol_combined (mass_of_water = 18) (molar_mass_of_water = 18.015) : 
  let moles_of_water := mass_of_water / molar_mass_of_water
  moles_of_water = 1 → 1 = 1 := 
by
  sorry

end moles_of_amyl_alcohol_combined_l641_641361


namespace even_function_value_of_a_l641_641819

theorem even_function_value_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x * (Real.exp x + a * Real.exp (-x))) (h_even : ∀ x : ℝ, f x = f (-x)) : a = -1 := 
by
  sorry

end even_function_value_of_a_l641_641819


namespace parrots_are_red_l641_641928

-- Definitions for fractions.
def total_parrots : ℕ := 160
def green_fraction : ℚ := 5 / 8
def blue_fraction : ℚ := 1 / 4

-- Definition for calculating the number of parrots.
def number_of_green_parrots : ℚ := green_fraction * total_parrots
def number_of_blue_parrots : ℚ := blue_fraction * total_parrots
def number_of_red_parrots : ℚ := total_parrots - number_of_green_parrots - number_of_blue_parrots

-- The theorem to prove.
theorem parrots_are_red : number_of_red_parrots = 20 := by
  -- Proof is omitted.
  sorry

end parrots_are_red_l641_641928


namespace polygon_sides_and_angles_polygon_sides_and_angles_valid_n_l641_641759

noncomputable def is_equal_sides_and_angles_polygon (n : ℕ) : Prop :=
  ∃ (P : fin n → ℝ^3),
    (∀ i j : fin n, P i ≠ P j → dist (P i) (P j) = dist (P 0) (P 1)) ∧
    (∀ i : fin n, angle (P i) (P (i + 1)%n) = angle (P 0) (P 1))

theorem polygon_sides_and_angles :
  ∀ n : ℕ, n = 5 ∨ n < 4 → ¬ is_equal_sides_and_angles_polygon n :=
by sorry

theorem polygon_sides_and_angles_valid_n : 
  ∀ n : ℕ, is_equal_sides_and_angles_polygon n → (n ≥ 4 ∧ n ≠ 5) :=
by sorry

end polygon_sides_and_angles_polygon_sides_and_angles_valid_n_l641_641759


namespace symmetric_line_eq_l641_641964

theorem symmetric_line_eq (x y : ℝ) :  
  (x - 2 * y + 3 = 0) → (x + 2 * y + 3 = 0) :=
sorry

end symmetric_line_eq_l641_641964


namespace notebook_cost_l641_641480

theorem notebook_cost (s n c : ℕ) (h1 : s ≥ 19) (h2 : n > 2) (h3 : c > n) (h4 : s * c * n = 3969) : c = 27 :=
sorry

end notebook_cost_l641_641480


namespace total_cookies_l641_641692

def total_chocolate_chip_batches := 5
def cookies_per_chocolate_chip_batch := 8
def total_oatmeal_batches := 3
def cookies_per_oatmeal_batch := 7
def total_sugar_batches := 1
def cookies_per_sugar_batch := 10
def total_double_chocolate_batches := 1
def cookies_per_double_chocolate_batch := 6

theorem total_cookies : 
  (total_chocolate_chip_batches * cookies_per_chocolate_chip_batch) +
  (total_oatmeal_batches * cookies_per_oatmeal_batch) +
  (total_sugar_batches * cookies_per_sugar_batch) +
  (total_double_chocolate_batches * cookies_per_double_chocolate_batch) = 77 :=
by sorry

end total_cookies_l641_641692


namespace find_x_l641_641498

-- Definitions for given conditions
variable {α : Type} [LinearOrderedField α]
variable (ACB AED x : α)

-- Given conditions
def condition1 := ACB + 40 + 60 = 180
def condition2 := (BC_parallel_DE : Prop)

-- Theorem statement
theorem find_x (h1 : condition1) (h2 : BC_parallel_DE) : x = 100 :=
by
  sorry

end find_x_l641_641498


namespace selling_price_ratio_l641_641314

-- Define the initial cost price
def CP := 100

-- Define the selling prices at different profit percentages
def SP1 := CP + (0.30 * CP)
def SP2 := CP + (2.60 * CP)

-- Prove the ratio of the new selling price to the original selling price is 36 / 13
theorem selling_price_ratio : (SP2 / SP1) = (36 / 13) := 
by 
  sorry

end selling_price_ratio_l641_641314


namespace total_number_of_animals_l641_641754

-- Definitions for the animal types
def heads_per_hen := 2
def legs_per_hen := 8
def heads_per_peacock := 3
def legs_per_peacock := 9
def heads_per_zombie_hen := 6
def legs_per_zombie_hen := 12

-- Given total heads and legs
def total_heads := 800
def total_legs := 2018

-- Proof that the total number of animals is 203
theorem total_number_of_animals : 
  ∀ (H P Z : ℕ), 
    heads_per_hen * H + heads_per_peacock * P + heads_per_zombie_hen * Z = total_heads
    ∧ legs_per_hen * H + legs_per_peacock * P + legs_per_zombie_hen * Z = total_legs 
    → H + P + Z = 203 :=
by
  sorry

end total_number_of_animals_l641_641754


namespace pond_sustain_capacity_l641_641704

-- Defining the initial number of frogs
def initial_frogs : ℕ := 5

-- Defining the number of tadpoles
def number_of_tadpoles (frogs: ℕ) : ℕ := 3 * frogs

-- Defining the number of matured tadpoles (those that survive to become frogs)
def matured_tadpoles (tadpoles: ℕ) : ℕ := (2 * tadpoles) / 3

-- Defining the total number of frogs after tadpoles mature
def total_frogs_after_mature (initial_frogs: ℕ) (matured_tadpoles: ℕ) : ℕ :=
  initial_frogs + matured_tadpoles

-- Defining the number of frogs that need to find a new pond
def frogs_to_leave : ℕ := 7

-- Defining the number of frogs the pond can sustain
def frogs_pond_can_sustain (total_frogs: ℕ) (frogs_to_leave: ℕ) : ℕ :=
  total_frogs - frogs_to_leave

-- The main theorem stating the number of frogs the pond can sustain given the conditions
theorem pond_sustain_capacity : frogs_pond_can_sustain
  (total_frogs_after_mature initial_frogs (matured_tadpoles (number_of_tadpoles initial_frogs)))
  frogs_to_leave = 8 := by
  -- proof goes here
  sorry

end pond_sustain_capacity_l641_641704


namespace total_sales_l641_641585

theorem total_sales (T : ℝ) (h1 : (2 / 5) * T = (2 / 5) * T) (h2 : (3 / 5) * T = 48) : T = 80 :=
by
  -- added sorry to skip proofs as per the requirement
  sorry

end total_sales_l641_641585


namespace problem_1_problem_2_l641_641113

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- 1. Prove that A ∩ B = {x | -2 < x ≤ 2}
theorem problem_1 : A ∩ B = {x | -2 < x ∧ x ≤ 2} :=
by
  sorry

-- 2. Prove that (complement U A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3}
theorem problem_2 : (U \ A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3} :=
by
  sorry

end problem_1_problem_2_l641_641113


namespace sum_greatest_odd_divisors_correct_l641_641767

def greatest_odd_divisor (n : Nat) : Nat :=
  let rec helper := λ m : Nat =>
    if m % 2 = 1 then m else helper (m / 2)
  helper n

def range := List.range' 2007 (4012 - 2007 + 1)

def sum_greatest_odd_divisors : Nat :=
  range.foldl (λ acc n => acc + greatest_odd_divisor n) 0

theorem sum_greatest_odd_divisors_correct :
  sum_greatest_odd_divisors = 4024036 :=
by
  sorry

end sum_greatest_odd_divisors_correct_l641_641767


namespace all_sheep_can_be_one_color_l641_641551

theorem all_sheep_can_be_one_color (b r v : ℕ) (h_initial : b = 22 ∧ r = 18 ∧ v = 15) 
(∀ (b' r' v' : ℕ), (b' = b - 1 ∧ r' = r - 1 ∧ v' = v + 2) ∨ (b' = b - 1 ∧ r' = r + 2 ∧ v' = v - 1) ∨ (b' = b + 2 ∧ r' = r - 1 ∧ v' = v - 1)) :
∃ (b r v : ℕ), (r = 0 ∧ v = 0 ∧ b = 55) :=
sorry

end all_sheep_can_be_one_color_l641_641551


namespace arithmetic_sequence_sum_l641_641811

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n+1) = a n + d)
  (h1 : a 2 + a 3 = 1)
  (h2 : a 10 + a 11 = 9) :
  a 5 + a 6 = 4 :=
sorry

end arithmetic_sequence_sum_l641_641811


namespace find_n_l641_641140

theorem find_n
  (b q : ℝ)
  (h : ∃ d : ℝ, ∀ x : ℝ, (x^2 - b*x + d) / ((b + 1)*x - q) = (2*b + 2 - 2) / (2*b + 2 + 2) ∧
    (roots (polynomial.C d + polynomial.X^2 -b * polynomial.X) = [r, -r])) :
  n = 2*b + 2 := by
  sorry

end find_n_l641_641140


namespace circle_diameter_given_area_l641_641635

theorem circle_diameter_given_area : 
  (∃ (r : ℝ), 81 * Real.pi = Real.pi * r^2 ∧ 2 * r = d) → d = 18 := by
  sorry

end circle_diameter_given_area_l641_641635


namespace max_prime_numbers_in_table_l641_641880

noncomputable def problem_75x75_table : Prop :=
  let N := 75 * 75
  let conditions (table : Fin N → ℕ) : Prop :=
    (∀ i j, i ≠ j → table i ≠ table j) ∧
    (∀ i, ∃ p ∈ Nat.primeFactors (table i), True) ∧
    (∀ i, (Nat.primeFactors (table i)).length ≤ 3) ∧
    (∀ i, ∃ k, k ≠ i ∧ (table i).gcd (table k) ≠ 1)
  ∃ (table : Fin N → ℕ), conditions table ∧
    (Nat.primeFactors (table i)).length ≤ 4218

theorem max_prime_numbers_in_table :
  problem_75x75_table :=
sorry

end max_prime_numbers_in_table_l641_641880


namespace add_base6_l641_641715

def base6_to_base10 (n : Nat) : Nat :=
  let rec aux (n : Nat) (exp : Nat) : Nat :=
    match n with
    | 0     => 0
    | n + 1 => aux n (exp + 1) + (n % 6) * (6 ^ exp)
  aux n 0

def base10_to_base6 (n : Nat) : Nat :=
  let rec aux (n : Nat) : Nat :=
    if n = 0 then 0
    else
      let q := n / 6
      let r := n % 6
      r + 10 * aux q
  aux n

theorem add_base6 (a b : Nat) (h1 : base6_to_base10 a = 5) (h2 : base6_to_base10 b = 13) : base10_to_base6 (base6_to_base10 a + base6_to_base10 b) = 30 :=
by
  sorry

end add_base6_l641_641715


namespace max_product_of_two_integers_sum_2000_l641_641250

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641250


namespace no_two_heads_consecutive_probability_l641_641298

noncomputable def probability_no_two_heads_consecutive (total_flips : ℕ) : ℚ :=
  if total_flips = 10 then 144 / 1024 else 0

theorem no_two_heads_consecutive_probability :
  probability_no_two_heads_consecutive 10 = 9 / 64 :=
by
  unfold probability_no_two_heads_consecutive
  rw [if_pos rfl]
  norm_num
  sorry

end no_two_heads_consecutive_probability_l641_641298


namespace number_of_apples_l641_641693

theorem number_of_apples (n : ℕ) (h : (1 * (n - 1) / (n * (n - 1) / 2)) = 0.25) : n = 8 :=
by sorry

end number_of_apples_l641_641693


namespace hyperbola_eccentricity_l641_641960

noncomputable def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 + (b^2) / (a^2))

theorem hyperbola_eccentricity :
  (eccentricity (real.sqrt 2) (real.sqrt 6)) = 2 := by
sorry

end hyperbola_eccentricity_l641_641960


namespace sin_cos_mul_zero_l641_641392

variable {θ : ℝ}

theorem sin_cos_mul_zero (h : sin θ + cos θ = -1) : sin θ * cos θ = 0 := by
  sorry

end sin_cos_mul_zero_l641_641392


namespace linear_equation_one_variable_l641_641669

-- Definitions for the given conditions
def optionA : Prop := 11 * x - 7 = 0
def optionB : Prop := 4 * a - 1 = 8
def optionC : Prop := 6 * x + y = 3
def optionD : Prop := x ^ 3 - x = 4 * x

-- Statement to prove
theorem linear_equation_one_variable (h : optionB) : true := sorry

end linear_equation_one_variable_l641_641669


namespace sequence_product_2023_l641_641783

theorem sequence_product_2023 :
  ∀ (a : ℕ → ℚ),
    a 1 = -1 →
    (∀ k, a (k + 1) = 1 / (1 - a k)) →
    (∏ i in finset.range 2023, a (i + 1)) = -1 :=
by
  intros a h1 h2
  sorry

end sequence_product_2023_l641_641783


namespace no_consecutive_heads_probability_l641_641304

def prob_no_consecutive_heads (n : ℕ) : ℝ := 
  -- This is the probability function for no consecutive heads
  if n = 10 then (9 / 64) else 0

theorem no_consecutive_heads_probability :
  prob_no_consecutive_heads 10 = 9 / 64 :=
by
  -- Proof would go here
  sorry

end no_consecutive_heads_probability_l641_641304


namespace at_least_one_divisible_by_10_l641_641355

-- Given: Five distinct integers
variable (a b c d e : ℤ)

-- Given condition: For any choice of three distinct integers, their product is divisible by 10
def property (a b c d e : ℤ) : Prop := 
  ∀ (x y z : ℤ), (x, y, z) ∈ Set.ofList [a, b, c, d, e] → x ≠ y → x ≠ z → y ≠ z → 10 ∣ (x * y * z)

-- to show: At least one of a, b, c, d, e is divisible by 10
theorem at_least_one_divisible_by_10 (a b c d e : ℤ) (h : property a b c d e) : 
  10 ∣ a ∨ 10 ∣ b ∨ 10 ∣ c ∨ 10 ∣ d ∨ 10 ∣ e :=
sorry

end at_least_one_divisible_by_10_l641_641355


namespace alternating_sum_g_l641_641527

def g (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem alternating_sum_g :
  (∑ k in finset.range 2020, (-1 : ℝ) ^ (k + 1) * g ((k + 1) / 2021)) = 0 := 
sorry

end alternating_sum_g_l641_641527


namespace question_solution_l641_641670

theorem question_solution 
  (hA : -(-1) = abs (-1))
  (hB : ¬ (∃ n : ℤ, ∀ m : ℤ, n < m ∧ m < 0))
  (hC : (-2)^3 = -2^3)
  (hD : ∃ q : ℚ, q = 0) :
  ¬ (∀ q : ℚ, q > 0 ∨ q < 0) := 
by {
  sorry
}

end question_solution_l641_641670


namespace toothpicks_in_100th_stage_l641_641968

theorem toothpicks_in_100th_stage :
  ∀ (a₁ d n : ℕ), a₁ = 4 → d = 4 → n = 100 → (a₁ + (n - 1) * d) = 400 :=
by
  intros a₁ d n ha₁ hd hn
  rw [ha₁, hd, hn]
  sorry

end toothpicks_in_100th_stage_l641_641968


namespace max_product_of_two_integers_sum_2000_l641_641252

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l641_641252


namespace MorseCodeDistinctSymbols_l641_641032

theorem MorseCodeDistinctSymbols:
  (1.sequence (λ _, bool).length = {1, 2, 3, 4, 5}).card = 62 :=
by
  sorry

end MorseCodeDistinctSymbols_l641_641032


namespace symmetric_point_origin_l641_641591

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

theorem symmetric_point_origin (x y : ℝ) (h : (x, y) = (-2, 3)) :
  symmetric_point (x, y) = (2, -3) :=
by
  rw [h]
  unfold symmetric_point
  simp
  sorry

end symmetric_point_origin_l641_641591


namespace total_dividends_correct_l641_641656

-- Conditions
def net_profit (total_income expenses loan_penalty_rate : ℝ) : ℝ :=
  let net1 := total_income - expenses
  let loan_penalty := net1 * loan_penalty_rate
  net1 - loan_penalty

def total_loan_payments (monthly_payment months additional_payment : ℝ) : ℝ :=
  (monthly_payment * months) - additional_payment

def dividend_per_share (net_profit total_loan_payments num_shares : ℝ) : ℝ :=
  (net_profit - total_loan_payments) / num_shares

noncomputable def total_dividends_director (dividend_per_share shares_owned : ℝ) : ℝ :=
  dividend_per_share * shares_owned

theorem total_dividends_correct :
  total_dividends_director (dividend_per_share (net_profit 1500000 674992 0.2) (total_loan_payments 23914 12 74992) 1000) 550 = 246400 :=
sorry

end total_dividends_correct_l641_641656


namespace at_least_one_negative_root_l641_641778

theorem at_least_one_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (x^2 - 6*a*x - 2 + 2*a + 9*a^2 = 0)) ↔ a < (-1 + Real.sqrt 19) / 9 := by
  sorry

end at_least_one_negative_root_l641_641778


namespace white_area_percentage_l641_641330

-- Definitions for conditions
def side_length_large_white_square : ℝ := 14
def side_length_small_white_square : ℝ := 6
def side_length_fabric : ℝ := 20

-- Define the total area, the area of large and small white squares, and the percentage of the white area
def total_area_fabric : ℝ := side_length_fabric ^ 2
def area_large_white_square : ℝ := side_length_large_white_square ^ 2
def area_small_white_square : ℝ := side_length_small_white_square ^ 2
def total_white_area : ℝ := area_large_white_square + area_small_white_square
def percentage_white_area : ℝ := (total_white_area / total_area_fabric) * 100

-- The proof statement
theorem white_area_percentage : percentage_white_area = 58 := by
  sorry

end white_area_percentage_l641_641330


namespace rafael_weekly_earnings_l641_641130

def weekly_hours (m t r : ℕ) : ℕ := m + t + r
def total_earnings (hours wage : ℕ) : ℕ := hours * wage

theorem rafael_weekly_earnings : 
  ∀ (m t r wage : ℕ), m = 10 → t = 8 → r = 20 → wage = 20 → total_earnings (weekly_hours m t r) wage = 760 :=
by
  intros m t r wage hm ht hr hw
  rw [hm, ht, hr, hw]
  simp only [weekly_hours, total_earnings]
  sorry -- Proof needs to be completed

end rafael_weekly_earnings_l641_641130


namespace calculate_expression_l641_641339

theorem calculate_expression : 
  1 - (1/3)^(-1/2) - 1/(2 - Real.sqrt 3) - (27/8)^(1/3) + (Real.sqrt 7 - Real.sqrt 103)^0 + (-2/3)^(-1) = -1 - 2 * Real.sqrt 3 := 
by 
  sorry

end calculate_expression_l641_641339


namespace smallest_a_l641_641979

def root_product (P : Polynomial ℚ) : ℚ :=
  P.coeff 0

def poly_sum_roots_min_a (r1 r2 r3 : ℤ) (a b c : ℚ) : Prop :=
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
  r1 * r2 * r3 = 2310 ∧
  root_product (Polynomial.monomial 3 1 - Polynomial.monomial 2 a + Polynomial.monomial 1 b - Polynomial.monomial 0 2310) = 2310 ∧
  r1 + r2 + r3 = a

theorem smallest_a : ∃ a b : ℚ, ∀ r1 r2 r3 : ℤ, poly_sum_roots_min_a r1 r2 r3 a b 2310 → a = 28
  by sorry

end smallest_a_l641_641979


namespace general_form_equation_l641_641601

theorem general_form_equation (x : ℝ) : 
  x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := 
by 
  sorry

end general_form_equation_l641_641601


namespace find_m_n_l641_641357

-- Definition of the quadratic equation and condition on the roots
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Definitions provided by conditions from the problem
def positive_diff_between_roots (a b c : ℝ) (m n : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, quadratic_eq a b c x1 ∧ quadratic_eq a b c x2 ∧
  (x1 - x2).abs = (x1 - x2).abs ∧ -- (x1 - x2).abs = 2 * Real.sqrt m / n (simplifying to ignore the ambiguous differentiation)
  m = 249 ∧ n = 5

-- Main theorem
theorem find_m_n :
  (∃ m n : ℕ, positive_diff_between_roots 5 (-7) (-10) (m : ℝ) (n : ℝ) ∧
    ¬ ∃ p : ℕ, nat.prime p ∧ p^2 ∣ m ∧ n ≥ 0 ∧ m + n = 254) :=
sorry

end find_m_n_l641_641357


namespace expression_simplification_l641_641599

-- Definitions for P and Q based on x and y
def P (x y : ℝ) := x + y
def Q (x y : ℝ) := x - y

-- The mathematical property to prove
theorem expression_simplification (x y : ℝ) (h : x ≠ 0) (k : y ≠ 0) : 
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (x^2 - y^2) / (x * y) := 
by
  -- Sorry is used to skip the proof here
  sorry

end expression_simplification_l641_641599


namespace area_of_triangle_eq_one_l641_641391

open Real

variables (a b : ℝ × ℝ)
def vector_a : ℝ × ℝ := (cos (2/3 * π), sin (2/3 * π))
def OA := (a : ℝ × ℝ) - b
def OB := (a : ℝ × ℝ) + b
def is_isosceles_right_triangle (O A B : ℝ × ℝ) : Prop := 
  (dist O A = dist O B) ∧ (∥OA∥ * ∥OB∥ = cos (π / 2) * ∥OA∥ * ∥OB∥)

theorem area_of_triangle_eq_one
  (ha : a = vector_a) 
  (hOAB : is_isosceles_right_triangle (0,0) OA OB) :
  (1 / 2) * ∥OA∥ * ∥OB∥ = 1 :=
by sorry

end area_of_triangle_eq_one_l641_641391


namespace sum_of_digits_next_exact_multiple_l641_641544

noncomputable def Michael_next_age_sum_of_digits (L M T n : ℕ) : ℕ :=
  let next_age := M + n
  ((next_age / 10) % 10) + (next_age % 10)

theorem sum_of_digits_next_exact_multiple :
  ∀ (L M T n : ℕ),
    T = 2 →
    M = L + 4 →
    (∀ k : ℕ, k < 8 → ∃ m : ℕ, L = m * T + k * T) →
    (∃ n, (M + n) % (T + n) = 0) →
    Michael_next_age_sum_of_digits L M T n = 9 :=
by
  intros
  sorry

end sum_of_digits_next_exact_multiple_l641_641544


namespace david_initial_money_l641_641741

theorem david_initial_money (S X : ℕ) (h1 : S - 800 = 500) (h2 : X = S + 500) : X = 1800 :=
by
  sorry

end david_initial_money_l641_641741


namespace smallest_a_l641_641980

def root_product (P : Polynomial ℚ) : ℚ :=
  P.coeff 0

def poly_sum_roots_min_a (r1 r2 r3 : ℤ) (a b c : ℚ) : Prop :=
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
  r1 * r2 * r3 = 2310 ∧
  root_product (Polynomial.monomial 3 1 - Polynomial.monomial 2 a + Polynomial.monomial 1 b - Polynomial.monomial 0 2310) = 2310 ∧
  r1 + r2 + r3 = a

theorem smallest_a : ∃ a b : ℚ, ∀ r1 r2 r3 : ℤ, poly_sum_roots_min_a r1 r2 r3 a b 2310 → a = 28
  by sorry

end smallest_a_l641_641980


namespace problem_inequality_l641_641546

theorem problem_inequality {a : ℝ} (h : ∀ x : ℝ, (x - a) * (1 - x - a) < 1) : 
  -1/2 < a ∧ a < 3/2 := by
  sorry

end problem_inequality_l641_641546


namespace cucumber_weight_evaporation_l641_641312

theorem cucumber_weight_evaporation :
  let w_99 := 50
  let p_99 := 0.99
  let evap_99 := 0.01
  let w_98 := 30
  let p_98 := 0.98
  let evap_98 := 0.02
  let w_97 := 20
  let p_97 := 0.97
  let evap_97 := 0.03

  let initial_water_99 := p_99 * w_99
  let dry_matter_99 := w_99 - initial_water_99
  let evaporated_water_99 := evap_99 * initial_water_99
  let new_weight_99 := (initial_water_99 - evaporated_water_99) + dry_matter_99

  let initial_water_98 := p_98 * w_98
  let dry_matter_98 := w_98 - initial_water_98
  let evaporated_water_98 := evap_98 * initial_water_98
  let new_weight_98 := (initial_water_98 - evaporated_water_98) + dry_matter_98

  let initial_water_97 := p_97 * w_97
  let dry_matter_97 := w_97 - initial_water_97
  let evaporated_water_97 := evap_97 * initial_water_97
  let new_weight_97 := (initial_water_97 - evaporated_water_97) + dry_matter_97

  let total_new_weight := new_weight_99 + new_weight_98 + new_weight_97
  total_new_weight = 98.335 :=
 by
  sorry

end cucumber_weight_evaporation_l641_641312


namespace greatest_product_sum_2000_l641_641218

theorem greatest_product_sum_2000 : 
  ∃ (x y : ℤ), x + y = 2000 ∧ ∀ z w : ℤ, z + w = 2000 → z * w ≤ x * y ∧ x * y = 1000000 := 
by
  sorry

end greatest_product_sum_2000_l641_641218


namespace inequality_proof_l641_641915

variable {a b c : ℝ}
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_pos_c : 0 < c)

theorem inequality_proof :
  (a + 3 * c) / (a + 2 * b + c) + 
  (4 * b) / (a + b + 2 * c) - 
  (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 := 
sorry

end inequality_proof_l641_641915


namespace necessary_but_not_sufficient_l641_641617

theorem necessary_but_not_sufficient (x : ℝ) :
  (|x - 1| < 2 → x(x - 3) < 0) ∧ ¬(|x - 1| < 2 ↔ x(x - 3) < 0) :=
by
  sorry

end necessary_but_not_sufficient_l641_641617


namespace shortest_chord_standard_equation_l641_641067

-- Define the parametric equations for the line and the circle
variables (t a α : ℝ)

def line_parametric_x (t : ℝ) : ℝ := 3 + t
def line_parametric_y (t a : ℝ) : ℝ := 1 + a * t

def circle_parametric_x (α : ℝ) : ℝ := 2 + 2 * Real.cos α
def circle_parametric_y (α : ℝ) : ℝ := 2 * Real.sin α

-- The statement to prove
theorem shortest_chord_standard_equation {a : ℝ} : 
  (∃ t α : ℝ, line_parametric_x t = circle_parametric_x α ∧ 
               line_parametric_y t a = circle_parametric_y α) →
  (∃ m : ℝ, ∃ c : ℝ, m * 3 + c - 1 = 0 ∧
                 m * 2 + c = 0 ∧
                 ∀ (x y : ℝ), y - 1 = a * (x - 3) → x + y - 4 = 0) :=
begin
  sorry
end

end shortest_chord_standard_equation_l641_641067


namespace S_1024_is_1024_l641_641321

/-- Define the sequence of sets S_n based on the given conditions --/
noncomputable def S : ℕ → set ℕ
| 1 := {1}
| 2 := {2}
| (n + 1) := {k | (k - 1 ∈ S n) ⊻ (k ∈ S (n - 1))}

/-- The theorem stating the solution to the problem --/
theorem S_1024_is_1024 : S 1024 = {1024} :=
sorry

end S_1024_is_1024_l641_641321


namespace exponent_equality_l641_641386

theorem exponent_equality (x : ℝ) : 8 = 2^3 ∧ 32 = 2^5 ∧ 2^x * 8^(x+1) = 32^3 ↔ x = 3 :=
by sorry

end exponent_equality_l641_641386


namespace solve_for_x_l641_641352

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : x^3 - 2 * x^2 = 0 ↔ x = 2 :=
by sorry

end solve_for_x_l641_641352


namespace linear_valid_arrangements_circular_valid_arrangements_l641_641066

def word := "EFFERVESCES"
def multiplicities := [("E", 4), ("F", 2), ("S", 2), ("R", 1), ("V", 1), ("C", 1)]

-- Number of valid linear arrangements
def linear_arrangements_no_adj_e : ℕ := 88200

-- Number of valid circular arrangements
def circular_arrangements_no_adj_e : ℕ := 6300

theorem linear_valid_arrangements : 
  ∃ n, n = linear_arrangements_no_adj_e := 
  by
    sorry 

theorem circular_valid_arrangements :
  ∃ n, n = circular_arrangements_no_adj_e :=
  by
    sorry

end linear_valid_arrangements_circular_valid_arrangements_l641_641066


namespace max_product_of_two_integers_sum_2000_l641_641195

theorem max_product_of_two_integers_sum_2000 : 
  ∃ (x : ℕ), (2000 * x - x^2 = 1000000 ∧ 0 ≤ x ∧ x ≤ 2000) := 
by
  sorry

end max_product_of_two_integers_sum_2000_l641_641195


namespace Morse_code_sequences_l641_641045

theorem Morse_code_sequences : 
  let symbols (n : ℕ) := 2^n in
  symbols 1 + symbols 2 + symbols 3 + symbols 4 + symbols 5 = 62 :=
by
  sorry

end Morse_code_sequences_l641_641045


namespace base9_base8_decimal_eq_seventy_one_l641_641645

theorem base9_base8_decimal_eq_seventy_one (C D : ℕ) (h1 : 9 * C + D = 8 * D + C) (h2 : 0 ≤ C ∧ C ≤ 8) (h3 : 0 ≤ D ∧ D ≤ 7) : 9 * C + D = 71 :=
by 
  have h4 : 8 * C = 7 * D, from (by linarith),
  have h5 : C = 7, from sorry,
  have h6 : D = 8, from sorry,
  rw [h5, h6]

end base9_base8_decimal_eq_seventy_one_l641_641645


namespace angle_AKT_eq_angle_CAM_l641_641929

-- Definitions based on the problem conditions
variable {ABC : Triangle}
variable {C : Point} {D : Point} {K : Point} {T : Point} {M : Point}

-- Conditions
variable (isosceles_ABC : is_isosceles ABC C)
variable (D_on_AC : D.on_line AC)
variable (K_on_arc_CD : K.on_smaller_arc_CD (circumcircle (triangle BCD)))
variable (T_on_intersection : T.on_intersection (ray CK) (line_parallel_to BC (through A)))
variable (M_is_midpoint : is_midpoint M D T)

-- Theorem to prove
theorem angle_AKT_eq_angle_CAM :
  ∠AKT = ∠CAM :=
sorry

end angle_AKT_eq_angle_CAM_l641_641929


namespace lengths_rel_AB_and_CA_l641_641079

noncomputable def Triangle (A B C : Type) := sorry

variables {A B C : Type}
variables [Triangle A B C]

-- Definitions for the constructed triangles
noncomputable def Triangle_ABE := sorry
noncomputable def Triangle_CAD := sorry

-- Condition stating triangles ABE and CAD are similar and constructed outwards
axiom similar_ABE_CAD : Triangle_ABE ≃ Triangle_CAD

-- Condition stating that BD = CE
axiom BD_eq_CE : len (Segment BD) = len (Segment CE)

theorem lengths_rel_AB_and_CA : BD_eq_CE → (¬ ∃ k, len (Segment AB) = k * len (Segment AC)) :=
by
  sorry

end lengths_rel_AB_and_CA_l641_641079


namespace no_two_consecutive_heads_l641_641293

/-- The probability that two heads never appear consecutively when a coin is tossed 10 times is 9/64. -/
theorem no_two_consecutive_heads (tosses : ℕ) (prob: ℚ):
  tosses = 10 → 
  (prob = 9 / 64) → 
  prob = probability_no_consecutive_heads tosses := sorry

end no_two_consecutive_heads_l641_641293


namespace no_integer_solution_wrt_omega_l641_641091

theorem no_integer_solution_wrt_omega (omega : ℂ) (a b : ℤ) (h_root : omega^4 = -1) (h_i : omega = complex.I) :
  |a * omega + b| ≠ real.sqrt 2 :=
by
  sorry

end no_integer_solution_wrt_omega_l641_641091


namespace no_such_integers_l641_641109

noncomputable def omega : ℂ := complex.exp (2 * real.pi * complex.I / 5)

theorem no_such_integers (a b c d k : ℤ) (h_k : k > 1) :
  ¬ ((a + b * omega + c * omega^2 + d * omega^3)^k = 1 + omega) :=
sorry

end no_such_integers_l641_641109


namespace points_lie_on_line_l641_641385

variable (t a : ℝ)

def x := (Real.cos t) ^ 2 - a
def y := (Real.sin t) ^ 2 + a

theorem points_lie_on_line : x t a + y t a = 1 := by
  sorry

end points_lie_on_line_l641_641385


namespace sequence_converges_to_1_l641_641921

noncomputable def a : ℕ → ℝ
| 1     := 1.0
| (n+2) := Nat.sqrt (n + 2 + Real.sqrt (n + 2 + ... (Real.sqrt n + 1)) -- n nested square roots

theorem sequence_converges_to_1 (a : ℕ → ℝ) :
  (∀ n, a 1 = 1 ∧ (∀ m > 1, a m = Real.sqrt (m + Real.sqrt (m + Real.sqrt ... (Real.sqrt m)))) →
  ∃ L : ℝ, L = 1 ∧ Filter.Tendsto (λ n, a n / Real.sqrt n) Filter.atTop (𝓝 L))
:=
begin
  sorry
end

end sequence_converges_to_1_l641_641921


namespace last_three_digits_of_7_pow_103_l641_641369

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 614 := by
  sorry

end last_three_digits_of_7_pow_103_l641_641369


namespace solve_system_of_equations_l641_641953

theorem solve_system_of_equations (x y : ℝ) :
    (5 * x * (1 + 1 / (x^2 + y^2)) = 12 ∧ 5 * y * (1 - 1 / (x^2 + y^2)) = 4) ↔
    (x = 2 ∧ y = 1) ∨ (x = 2 / 5 ∧ y = -(1 / 5)) :=
by
  sorry

end solve_system_of_equations_l641_641953


namespace symmetric_points_l641_641777

open Real

theorem symmetric_points (a : ℝ) : 
  (∀ k : ℤ, a = -π / 2 + 2 * k * π) ↔ 
  (cos (2 * a) = sin a ∧ -sin (2 * a) = cos a) :=
by
  sorry

end symmetric_points_l641_641777


namespace max_value_abc_l641_641469

theorem max_value_abc (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_constraint : 2 * a + 4 * b + 8 * c = 16) : 
  abc_max := 64 / 27 := 
by
  sorry

end max_value_abc_l641_641469


namespace problem_result_l641_641142

noncomputable def max_value (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) : ℝ :=
  2 * x^2 + x * y + y^2

theorem problem (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) :
  max_value x y hx = (75 + 60 * Real.sqrt 2) / 7 :=
sorry

theorem result : 75 + 60 + 2 + 7 = 144 :=
by norm_num

end problem_result_l641_641142


namespace additional_allergy_sufferers_l641_641931

theorem additional_allergy_sufferers (total_population : ℕ) (allergy_ratio : ℚ) (sample_size : ℕ) (diagnosed_allergies : ℕ) :
  (total_population = 4) → (allergy_ratio = 1 / total_population) → (sample_size = 300) → (diagnosed_allergies = 20) → 
  ((allergy_ratio * sample_size) - diagnosed_allergies = 55) :=
by
  intro h1 h2 h3 h4
  have h5 : allergy_ratio * sample_size = 75 := by sorry
  rw h5
  linarith

end additional_allergy_sufferers_l641_641931


namespace no_two_consecutive_heads_probability_l641_641306

theorem no_two_consecutive_heads_probability : 
  let total_outcomes := 2 ^ 10 in
  let favorable_outcomes := 
    (∑ n in range 6, nat.choose (10 - n - 1) n) in
  (favorable_outcomes / total_outcomes : ℚ) = 9 / 64 :=
by
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 1 + 10 + 36 + 56 + 35 + 6
  have h_total: total_outcomes = 1024 := by norm_num
  have h_favorable: favorable_outcomes = 144 := by norm_num
  have h_fraction: (favorable_outcomes : ℚ) / total_outcomes = 9 / 64 := by norm_num / [total_outcomes, favorable_outcomes]
  exact h_fraction

end no_two_consecutive_heads_probability_l641_641306


namespace both_firms_participate_social_optimality_l641_641064

variables (α V IC : ℝ)

-- Conditions definitions
def expected_income_if_both_participate (α V : ℝ) : ℝ :=
  α * (1 - α) * V + 0.5 * (α^2) * V

def condition_for_both_participation (α V IC : ℝ) : Prop :=
  expected_income_if_both_participate α V - IC ≥ 0

-- Values for specific case
noncomputable def V_specific : ℝ := 24
noncomputable def α_specific : ℝ := 0.5
noncomputable def IC_specific : ℝ := 7

-- Proof problem statement
theorem both_firms_participate : condition_for_both_participation α_specific V_specific IC_specific := by
  sorry

-- Definitions for social welfare considerations
def total_profit_if_both_participate (α V IC : ℝ) : ℝ :=
  2 * (expected_income_if_both_participate α V - IC)

def expected_income_if_one_participates (α V IC : ℝ) : ℝ :=
  α * V - IC

def social_optimal (α V IC : ℝ) : Prop :=
  total_profit_if_both_participate α V IC < expected_income_if_one_participates α V IC

theorem social_optimality : social_optimal α_specific V_specific IC_specific := by
  sorry

end both_firms_participate_social_optimality_l641_641064


namespace kibble_recommendation_difference_l641_641689

theorem kibble_recommendation_difference :
  (0.2 * 1000 : ℝ) < (0.3 * 1000) ∧ ((0.3 * 1000) - (0.2 * 1000)) = 100 :=
by
  sorry

end kibble_recommendation_difference_l641_641689


namespace max_product_of_sum_2000_l641_641239

theorem max_product_of_sum_2000 : 
  ∀ (x : ℝ), x + (2000 - x) = 2000 → (x * (2000 - x) ≤ 1000000) ∧ (∀ y : ℝ, y * (2000 - y) = 1000000 → y = 1000) :=
by
  sorry

end max_product_of_sum_2000_l641_641239


namespace ratio_girls_left_early_l641_641172

theorem ratio_girls_left_early
  (total_people : ℕ) (initial_girls : ℕ) (remaining_people : ℕ) 
  (fraction_boys_left : ℚ)
  (h_total : total_people = 600)
  (h_girls : initial_girls = 240)
  (h_remaining : remaining_people = 480)
  (h_fraction : fraction_boys_left = 1 / 4) :
  let boys_initial := total_people - initial_girls in
  let girls_left := remaining_people - (total_people - boys_initial * fraction_boys_left) in
  (girls_left / initial_girls : ℚ) = 1 / 8 :=
by
  sorry

end ratio_girls_left_early_l641_641172


namespace complex_modulus_squared_l641_641390

theorem complex_modulus_squared :
  let Z := (1 + Real.sqrt 3 * Complex.I) / 2 in
  Z * Complex.conj Z = 1 :=
by
  let Z := (1 + Real.sqrt 3 * Complex.I) / 2
  sorry

end complex_modulus_squared_l641_641390


namespace triangle_is_isosceles_l641_641127

variable {a b c : ℝ}
variable {A B C A1 B1 C1 : Type}

-- Definitions and conditions from the problem
def triangle : Type := ∀ {A B C : Type}, Prop

def is_angle_bisector (A B C A1 : Type) : Prop := sorry
def segment_length (x y : Type) (l : ℝ) : Prop := sorry
def perpendiculars_intersect_at_one_point (A B1 C1 : Type) : Prop := sorry

-- Angle bisectors and side lengths
def AA1_bisector (A B C A1 : Type) := is_angle_bisector A B C A1
def BB1_bisector (A B C B1 : Type) := is_angle_bisector A B C B1
def CC1_bisector (A B C C1 : Type) := is_angle_bisector A B C C1

def BC_side_length (BC : ℝ) := segment_length B C a
def AC_side_length (AC : ℝ) := segment_length A C b
def AB_side_length (AB : ℝ) := segment_length A B c

-- The proof statement about the isosceles triangle
theorem triangle_is_isosceles (h1 : AA1_bisector A B C A1)
                              (h2 : BB1_bisector A B C B1)
                              (h3 : CC1_bisector A B C C1)
                              (h4 : BC_side_length a)
                              (h5 : AC_side_length b)
                              (h6 : AB_side_length c)
                              (h7 : perpendiculars_intersect_at_one_point A B1 C1) :
                              a = b ∨ a = c ∨ b = c :=
sorry

end triangle_is_isosceles_l641_641127


namespace remaining_bottle_caps_l641_641745

-- Definitions based on conditions
def initial_bottle_caps : ℕ := 65
def eaten_bottle_caps : ℕ := 4

-- Theorem
theorem remaining_bottle_caps : initial_bottle_caps - eaten_bottle_caps = 61 :=
by
  sorry

end remaining_bottle_caps_l641_641745


namespace rafael_earnings_l641_641133

theorem rafael_earnings 
  (hours_monday : ℕ) 
  (hours_tuesday : ℕ) 
  (hours_left : ℕ) 
  (rate_per_hour : ℕ) 
  (h_monday : hours_monday = 10) 
  (h_tuesday : hours_tuesday = 8) 
  (h_left : hours_left = 20) 
  (h_rate : rate_per_hour = 20) : 
  (hours_monday + hours_tuesday + hours_left) * rate_per_hour = 760 := 
by
  sorry

end rafael_earnings_l641_641133


namespace remainder_division_l641_641662

theorem remainder_division (n r : ℕ) (k : ℤ) (h1 : n % 25 = r) (h2 : (n + 15) % 5 = r) (h3 : 0 ≤ r ∧ r < 25) : r = 5 :=
sorry

end remainder_division_l641_641662


namespace no_consecutive_heads_probability_l641_641305

def prob_no_consecutive_heads (n : ℕ) : ℝ := 
  -- This is the probability function for no consecutive heads
  if n = 10 then (9 / 64) else 0

theorem no_consecutive_heads_probability :
  prob_no_consecutive_heads 10 = 9 / 64 :=
by
  -- Proof would go here
  sorry

end no_consecutive_heads_probability_l641_641305


namespace equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l641_641190

-- Define a transformation predicate for words
inductive transform : List Char -> List Char -> Prop
| xy_to_yyx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 'y'] ++ l2) (l1 ++ ['y', 'y', 'x'] ++ l2)
| yyx_to_xy : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 'y', 'x'] ++ l2) (l1 ++ ['x', 'y'] ++ l2)
| xt_to_ttx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 't'] ++ l2) (l1 ++ ['t', 't', 'x'] ++ l2)
| ttx_to_xt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 't', 'x'] ++ l2) (l1 ++ ['x', 't'] ++ l2)
| yt_to_ty : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 't'] ++ l2) (l1 ++ ['t', 'y'] ++ l2)
| ty_to_yt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 'y'] ++ l2) (l1 ++ ['y', 't'] ++ l2)

-- Reflexive and transitive closure of transform
inductive transforms : List Char -> List Char -> Prop
| base : ∀ l, transforms l l
| step : ∀ l m n, transform l m → transforms m n → transforms l n

-- Definitions for the words and their information
def word1 := ['x', 'x', 'y', 'y']
def word2 := ['x', 'y', 'y', 'y', 'y', 'x']
def word3 := ['x', 'y', 't', 'x']
def word4 := ['t', 'x', 'y', 't']
def word5 := ['x', 'y']
def word6 := ['x', 't']

-- Proof statements
theorem equivalent_xy_xxyy : transforms word1 word2 :=
by sorry

theorem not_equivalent_xyty_txy : ¬ transforms word3 word4 :=
by sorry

theorem not_equivalent_xy_xt : ¬ transforms word5 word6 :=
by sorry

end equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l641_641190


namespace range_omega_l641_641439

open Real

noncomputable def f (ω x : ℝ) : ℝ := 4 * sin (ω * x) - sin (ω * x / 2 + π / 4) ^ 2 - 2 * (sin (ω * x)) ^ 2

theorem range_omega (ω : ℝ) : 
  (ω > 0) ∧ 
    (∀ x y : ℝ, -π / 4 ≤ x ∧ x ≤ 3 * π / 4 → -π / 4 ≤ y ∧ y ≤ 3 * π / 4 → x ≤ y → f ω x ≤ f ω y) ∧
    (∃ x : ℝ, 0 ≤ x ∧ ∀ y : ℝ, 0 ≤ y → y ≤ x → f ω y ≤ f ω x) ↔ 
  ω ∈ set.Icc (1 / 2) (2 / 3) :=
sorry

end range_omega_l641_641439


namespace sum_of_first_21_terms_l641_641322

def is_constant_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem sum_of_first_21_terms (a : ℕ → ℕ) (h1 : is_constant_sum_sequence a 5) (h2 : a 1 = 2) : (Finset.range 21).sum a = 52 :=
by
  sorry

end sum_of_first_21_terms_l641_641322


namespace solve_x2_plus_4y2_l641_641463

theorem solve_x2_plus_4y2 (x y : ℝ) (h₁ : x + 2 * y = 6) (h₂ : x * y = -6) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end solve_x2_plus_4y2_l641_641463


namespace find_m_and_n_l641_641438

theorem find_m_and_n
  (f : ℝ → ℝ)
  (m n : ℝ)
  (hf : ∀ x, f(x) = 2 * x^3 + 3 * m * x^2 + 3 * n * x - 6)
  (extreme1 : ∀ g, (∀ x, g(x) = derivative f x) → g(1) = 0)
  (extreme2 : ∀ g, (∀ x, g(x) = derivative f x) → g(2) = 0) :
  m = -3 ∧ n = 4 :=
  sorry

end find_m_and_n_l641_641438


namespace discount_savings_l641_641899

theorem discount_savings (initial_price discounted_price : ℝ)
  (h_initial : initial_price = 475)
  (h_discounted : discounted_price = 199) :
  initial_price - discounted_price = 276 :=
by
  rw [h_initial, h_discounted]
  sorry

end discount_savings_l641_641899


namespace smallest_integer_value_l641_641639

theorem smallest_integer_value (y : ℤ) (h : 7 - 3 * y < -8) : y ≥ 6 :=
sorry

end smallest_integer_value_l641_641639


namespace xunzi_wangzhi_l641_641686

theorem xunzi_wangzhi :
  (∀ (activities : Prop), activities → 
  ((C_1 : acting_according_to_objective_laws → expected_results) ∧ 
   (C_2 : subjective_initiative_must_be_based_on_objective_laws)) →
  ② ∧ ④ = true :=
by
  sorry

end xunzi_wangzhi_l641_641686


namespace determine_a_values_l641_641473

theorem determine_a_values (a : ℝ) : (∀ x y : ℝ, x ≤ y ∧ y ≤ 1 → f' x ≤ 0) → a ≤ -1 :=
by
  let f := fun x : ℝ => x^2 + 2*a*x + 2
  let f' := fun x : ℝ => 2*x + 2*a
  let condition := ∀ x y : ℝ, x ≤ y ∧ y ≤ 1 → f' x ≤ 0
  sorry

end determine_a_values_l641_641473


namespace candyStoreSpending_l641_641001

-- Definitions based on conditions provided
def weeklyAllowance : ℚ := 345 / 100   -- John's weekly allowance is $3.45
def arcadeFraction : ℚ := 3 / 5        -- John spent 3/5 of his allowance at the arcade
def toyStoreFraction : ℚ := 1 / 3      -- John spent 1/3 of his remaining allowance at the toy store

-- Main theorem to prove
theorem candyStoreSpending :
  let arcadeSpending := arcadeFraction * weeklyAllowance
  let remainingAfterArcade := weeklyAllowance - arcadeSpending
  let toyStoreSpending := toyStoreFraction * remainingAfterArcade
  let remainingAfterToyStore := remainingAfterArcade - toyStoreSpending
  remainingAfterToyStore = 92 / 100 := 
by
  sorry

end candyStoreSpending_l641_641001


namespace monday_has_greatest_temp_range_l641_641168

-- Define the temperatures
def high_temp (day : String) : Int :=
  if day = "Monday" then 6 else
  if day = "Tuesday" then 3 else
  if day = "Wednesday" then 4 else
  if day = "Thursday" then 4 else
  if day = "Friday" then 8 else 0

def low_temp (day : String) : Int :=
  if day = "Monday" then -4 else
  if day = "Tuesday" then -6 else
  if day = "Wednesday" then -2 else
  if day = "Thursday" then -5 else
  if day = "Friday" then 0 else 0

-- Define the temperature range for a given day
def temp_range (day : String) : Int :=
  high_temp day - low_temp day

-- Statement to prove: Monday has the greatest temperature range
theorem monday_has_greatest_temp_range : 
  temp_range "Monday" > temp_range "Tuesday" ∧
  temp_range "Monday" > temp_range "Wednesday" ∧
  temp_range "Monday" > temp_range "Thursday" ∧
  temp_range "Monday" > temp_range "Friday" := 
sorry

end monday_has_greatest_temp_range_l641_641168


namespace chocolate_bars_per_box_l641_641510

theorem chocolate_bars_per_box (total_chocolate_bars boxes : ℕ) (h1 : total_chocolate_bars = 710) (h2 : boxes = 142) : total_chocolate_bars / boxes = 5 := by
  sorry

end chocolate_bars_per_box_l641_641510


namespace jordan_machine_solution_l641_641896

theorem jordan_machine_solution (x : ℝ) (h : 2 * x + 3 - 5 = 27) : x = 14.5 :=
sorry

end jordan_machine_solution_l641_641896


namespace no_two_consecutive_heads_probability_l641_641309

theorem no_two_consecutive_heads_probability : 
  let total_outcomes := 2 ^ 10 in
  let favorable_outcomes := 
    (∑ n in range 6, nat.choose (10 - n - 1) n) in
  (favorable_outcomes / total_outcomes : ℚ) = 9 / 64 :=
by
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 1 + 10 + 36 + 56 + 35 + 6
  have h_total: total_outcomes = 1024 := by norm_num
  have h_favorable: favorable_outcomes = 144 := by norm_num
  have h_fraction: (favorable_outcomes : ℚ) / total_outcomes = 9 / 64 := by norm_num / [total_outcomes, favorable_outcomes]
  exact h_fraction

end no_two_consecutive_heads_probability_l641_641309


namespace units_digit_29_pow_8_pow_7_l641_641769

/-- The units digit of 29 raised to an arbitrary power follows a cyclical pattern. 
    For the purposes of this proof, we use that 29^k for even k ends in 1.
    Since 8^7 is even, we prove the units digit of 29^(8^7) is 1. -/
theorem units_digit_29_pow_8_pow_7 : (29^(8^7)) % 10 = 1 :=
by
  have even_power_cycle : ∀ k, k % 2 = 0 → (29^k) % 10 = 1 := sorry
  have eight_power_seven_even : (8^7) % 2 = 0 := by norm_num
  exact even_power_cycle (8^7) eight_power_seven_even

end units_digit_29_pow_8_pow_7_l641_641769


namespace area_of_triangle_l641_641078

noncomputable def f (x B : ℝ) : ℝ := sin(2 * x + B) + sqrt(3) * cos(2 * x + B)

theorem area_of_triangle
  (A B C : ℝ) -- angles of triangle ABC
  (a b c : ℝ) -- sides opposite to angles A, B, C respectively
  (h1 : A + B + C = π) -- angles in a triangle sum up to π
  (h2 : f (π / 12) B = 3)
  (h3 : sin(A) = sqrt(3) / 2) 
  (h4 : b = sqrt(3))
  : (if A = π / 3 then (1 / 2) * a * b = (3 * sqrt(3)) / 2 else true) ∧ -- condition for A = π / 3
    (if A = 2 * π / 3 then (1 / 2) * b * sin(C) = (3 * sqrt(3)) / 4 else true) -- condition for A = 2 * π / 3
:= sorry

end area_of_triangle_l641_641078


namespace domain_transformation_l641_641422

variable (f : ℝ → ℝ)
def is_domain (D : set ℝ) := ∀ x, f(x) ≠ 0 → x ∈ D

theorem domain_transformation {f : ℝ → ℝ} :
  is_domain f (-2, 0) → is_domain f (-3, 1) :=
by
  sorry

end domain_transformation_l641_641422


namespace sum_powers_mod_m_l641_641111

theorem sum_powers_mod_m (n r : ℕ) (a x : Fin (n + 1) → ℤ) (h1 : 2 ≤ r)
    (h2 : ∀ k : ℕ, (1 ≤ k ∧ k ≤ r) → ∑ j : Fin (n + 1), a j * x j ^ k = 0) :
    ∀ m : ℕ, (r+1 ≤ m ∧ m ≤ 2*r+1) → ∑ j : Fin (n + 1), a j * x j ^ m % m = 0 :=
by
  sorry

end sum_powers_mod_m_l641_641111


namespace smallest_value_of_a_l641_641989

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end smallest_value_of_a_l641_641989


namespace domain_of_log_expression_l641_641959

theorem domain_of_log_expression : 
  (set_of (λ x, x^2 - 4 * x - 21 > 0)) = (set_of (λ x, x < -3)) ∪ (set_of (λ x, x > 7)) :=
sorry

end domain_of_log_expression_l641_641959


namespace last_three_digits_of_7_pow_103_l641_641367

theorem last_three_digits_of_7_pow_103 : (7^103) % 1000 = 327 :=
by
  sorry

end last_three_digits_of_7_pow_103_l641_641367


namespace best_representation_of_marias_trip_l641_641540

-- Setting up the trip conditions as definitions
def mariaTrip (graph : Type) (G : graph) : Prop :=
  -- Northbound conditions
  driving_slowly_through_city_traffic G ∧
  slowing_down_at_checkpoint G ∧
  accelerating_past_checkpoint_to_lake G ∧
  -- Lake stop conditions
  stopping_for_picnic_and_walk G ∧
  -- Southbound conditions
  driving_rapidly_back_until_checkpoint G ∧
  slowing_down_at_checkpoint G ∧
  driving_slowly_back_through_city_traffic G

-- The graph satisfying these conditions
def graph_A : Type := sorry

-- Our goal statement 
theorem best_representation_of_marias_trip : mariaTrip graph graph_A :=
by
  sorry -- Proof steps omitted for brevity 

end best_representation_of_marias_trip_l641_641540


namespace exists_platinum_matrix_iff_l641_641905

/-- Definition of a platinum matrix satisfying the given conditions -/
def is_platinum_matrix (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  (∀ i j, 1 ≤ M i j ∧ M i j ≤ n) ∧
  (∀ k, ∃! i, ∃! j, M i j = k) ∧
  (∀ k, ∃! i, ∃! j, M k k = k) ∧
  (∃ S : Finset (Fin n × Fin n), S.card = n ∧
    (∀ (i j : Fin n), (i, j) ∈ S → 1 ≤ M i j ∧ M i j ≤ n ∧ i ≠ j ∧ i ≠ k ∧ j ≠ l))

/-- The equivalence theorem to prove the existence of platinum matrices for certain n -/
theorem exists_platinum_matrix_iff {n : ℕ} (h : n ≠ 2 ∧ n ≠ 6) :
  ∃ M : Matrix (Fin n) (Fin n) ℕ, is_platinum_matrix M :=
sorry

end exists_platinum_matrix_iff_l641_641905


namespace det_M_det_H_eq_det_A_l641_641517

variables {K : Type*} [Field K] {n : ℕ}

-- Definitions for the block matrices and M belonging to GL_{2n}(K)
def block_matrix (A B C D : Matrix (Fin n) (Fin n) K) : Matrix (Fin (2 * n)) (Fin (2 * n)) K :=
  Matrix.vstack (Matrix.hstack A B) (Matrix.hstack C D)

def is_invertible (M : Matrix (Fin (2 * n)) (Fin (2 * n)) K) : Prop :=
  ∃ (M_inv : Matrix (Fin (2 * n)) (Fin (2 * n)) K), M ⬝ M_inv = 1 ∧ M_inv ⬝ M = 1

-- Hypotheses
variables (A B C D E F G H : Matrix (Fin n) (Fin n) K)
variables (M : Matrix (Fin (2 * n)) (Fin (2 * n)) K)
variables (M_inv : Matrix (Fin (2 * n)) (Fin (2 * n)) K)

-- M is in GL_{2n}(K)
hypothesis (hM : M ∈ GL (2 * n))

-- Block matrix form of M and its inverse M_inv
hypothesis (hM_block : M = block_matrix A B C D)
hypothesis (hM_inv_block : M_inv = block_matrix E F G H)

-- Theorem statement
theorem det_M_det_H_eq_det_A : ∃ M_inv : Matrix (Fin (2 * n)) (Fin (2 * n)) K, M ⬝ M_inv = 1 ∧ M_inv ⬝ M = 1 → (det M * det H = det A) :=
sorry

end det_M_det_H_eq_det_A_l641_641517
