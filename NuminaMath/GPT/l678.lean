import Mathlib

namespace expected_seconds_to_1214_l678_678856

noncomputable def expected_seconds_until_next_minute (initial_seconds_elapsed : ℕ) : ℝ :=
  let remaining_seconds := 50 in
  let E_X := (0 + 49) / 2.0 in
  initial_seconds_elapsed + E_X

theorem expected_seconds_to_1214 :
  expected_seconds_until_next_minute 10 = 25 :=
by
  sorry

end expected_seconds_to_1214_l678_678856


namespace smallest_m_plus_n_l678_678283

-- Definitions for constraints and variables
variables (m n : ℕ) (h1 : 1 < n) (h2 : n < m) (x : ℕ → ℂ)

-- Hypotheses based on given conditions
def conditions : Prop :=
  (∀ k : ℕ, 1 ≤ k ∧ k < n → ∑ i in finset.range n, x i ^ k = 1) ∧
  (∑ i in finset.range n, x i ^ n = 2) ∧
  (∑ i in finset.range n, x i ^ m = 4)

-- Main theorem statement
theorem smallest_m_plus_n (h : conditions m n x) : m + n = 34 :=
sorry  -- Proof to be filled in later

end smallest_m_plus_n_l678_678283


namespace no_fib_right_triangle_l678_678862

def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem no_fib_right_triangle (n : ℕ) : 
  ¬ (fibonacci n)^2 + (fibonacci (n+1))^2 = (fibonacci (n+2))^2 := 
by 
  sorry

end no_fib_right_triangle_l678_678862


namespace kendra_total_earnings_l678_678230

-- Definitions of the conditions based on the problem statement
def kendra_earnings_2014 : ℕ := 30000 - 8000
def laurel_earnings_2014 : ℕ := 30000
def kendra_earnings_2015 : ℕ := laurel_earnings_2014 + (laurel_earnings_2014 / 5)

-- The statement to be proved
theorem kendra_total_earnings : kendra_earnings_2014 + kendra_earnings_2015 = 58000 :=
by
  -- Using Lean tactics for the proof
  sorry

end kendra_total_earnings_l678_678230


namespace perfect_squares_l678_678552

variable (m n a : ℕ)

theorem perfect_squares (h : a = m * n) :
    ∃ k : ℕ, (\left(\frac{m + n}{2}\right)^2 - a = k^2) ∧ 
    ∃ l : ℕ, (\left(\frac{m - n}{2}\right)^2 + a = l^2) :=
by sorry

end perfect_squares_l678_678552


namespace PA_plus_PB_l678_678211

noncomputable def C : Set (ℝ × ℝ) := { p | ∃ α, p.1 = 3 * Real.cos α ∧ p.2 = Real.sin α }

noncomputable def l : Set (ℝ × ℝ) := { p | ∃ θ ρ, ρ = Real.sqrt (p.1^2 + p.2^2) ∧ θ = Real.arctan2 p.2 p.1 ∧ ρ * Real.sin (θ - π / 4) = Real.sqrt 2 }

noncomputable def standard_eq_C (x y : ℝ) : Prop := (x^2 / 9 + y^2 = 1)
noncomputable def inclination_angle_l : ℝ := π / 4

theorem PA_plus_PB : ∀ P : ℝ × ℝ, P = (0, 2) → (∀ A B, A ∈ C → B ∈ C → A ∈ l → B ∈ l → |P.1 - A.1| + |P.2 - A.2| + (|P.1 - B.1| + |P.2 - B.2|) = 18 * Real.sqrt 2/5) :=
by
  sorry

end PA_plus_PB_l678_678211


namespace geometric_inequality_l678_678216

variable {T : Type} [MetricSpace T]
variable (O : T) (F : List T)

def perimeter (F : List T) [MetricSpace T] := 
  List.sum (List.zipWith dist F (List.tail F ++ [List.head F]))

def sum_of_distances_to_point (O : T) (F : List T) [MetricSpace T] := 
  List.sum (F.map (dist O))

-- Assumes F is a list of pairs where each pair is a tuple (point on line, normal vector of line)
def sum_of_distances_to_lines (O : T) (F : List (T × T)) [NormedAddCommGroup T] [NormedSpace ℝ T] :=
  List.sum (F.map (fun ⟨(A, n)⟩ => abs (n ⬝ (O - A)) / ∥n∥))

theorem geometric_inequality 
  (H_polygon : List (T × T)) :
  let D := sum_of_distances_to_point O F
  let P := perimeter F
  let H := sum_of_distances_to_lines O H_polygon
  D^2 - H^2 ≥ P^2 / 4 :=
sorry

end geometric_inequality_l678_678216


namespace odd_number_of_divisors_implies_perfect_square_l678_678650

theorem odd_number_of_divisors_implies_perfect_square (n : ℕ) (hn_pos : 0 < n) (hn_odd_divisors : odd (divisors n).card) :
  ∃ d : ℕ, d * d = n := 
sorry

end odd_number_of_divisors_implies_perfect_square_l678_678650


namespace f_1_value_f_even_f_x_range_l678_678296

noncomputable def f : ℝ → ℝ := sorry
def D := {x : ℝ | x ≠ 0}
def f_multiplicative (x₁ x₂ : ℝ) (hx₁ : x₁ ∈ D) (hx₂ : x₂ ∈ D) : Prop := f (x₁ * x₂) = f x₁ + f x₂
def f4 := f 4 = 3
def f_increasing (x : ℝ) (hx : 0 < x) : Prop := ∀ y : ℝ, 0 < y → x < y → f x < f y
def f_condition (x : ℝ) : Prop := f (x - 2) + f (x + 1) ≤ 3
def x_range (x : ℝ) : Prop := (-2 ≤ x ∧ x < -1) ∨ (-1 < x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)

-- Problem 1
theorem f_1_value (h : ∀ x₁ x₂ ∈ D, f_multiplicative x₁ x₂ (by assumption) (by assumption)) : f 1 = 0 :=
sorry

-- Problem 2
theorem f_even (h : ∀ x₁ x₂ ∈ D, f_multiplicative x₁ x₂ (by assumption) (by assumption)) : ∀ x ∈ D, f (-x) = f x :=
sorry

-- Problem 3
theorem f_x_range (h1 : ∀ x₁ x₂ ∈ D, f_multiplicative x₁ x₂ (by assumption) (by assumption))
                   (h2 : f4) (h3 : ∀ x, f_condition x) (h4 : ∀ x, f_increasing x (by assumption)) :
                   ∀ x, x_range x :=
sorry

end f_1_value_f_even_f_x_range_l678_678296


namespace crows_eat_quarter_l678_678367

def first_crow_rate (N : ℕ) : ℕ :=
  N / 30  -- first crow eats N/30 nuts per hour

def second_crow_rate (N : ℕ) : ℕ :=
  N / 32  -- second crow eats N/32 nuts per hour

-- Remaining nuts after first crow eats 2 hours
noncomputable def remaining_nuts (N : ℕ) : ℕ :=
  (14 * N) / 15

-- Combined rate of both crows
noncomputable def combined_rate (N : ℕ) : ℕ :=
  first_crow_rate N + second_crow_rate N

-- Time to eat a quarter of nuts together
noncomputable def time_to_eat_quarter (N : ℕ) : ℝ :=
  (N / 4 : ℝ) / combined_rate N 

-- Total time from the beginning
noncomputable def total_time (N : ℕ) : ℝ :=
  2 + (120 / 31 : ℝ)

-- Goal: Prove that the total time is approximately 5.87 hours
theorem crows_eat_quarter (N : ℕ) : 
  total_time N ≈ 5.87 :=
sorry

end crows_eat_quarter_l678_678367


namespace daily_sale_correct_l678_678644

-- Define the original and additional amounts in kilograms
def original_rice := 4 * 1000 -- 4 tons converted to kilograms
def additional_rice := 4000 -- kilograms
def total_rice := original_rice + additional_rice -- total amount of rice in kilograms
def days := 4 -- days to sell all the rice

-- Statement to prove: The amount to be sold each day
def daily_sale_amount := 2000 -- kilograms per day

theorem daily_sale_correct : total_rice / days = daily_sale_amount :=
by 
  -- This is a placeholder for the proof
  sorry

end daily_sale_correct_l678_678644


namespace area_of_A_inter_B_l678_678167

noncomputable def setA : set (ℝ × ℝ) := { p | let (x, y) := p in x^2 + (y - 2)^2 ≤ 1 }
noncomputable def setBm (m : ℝ) : set (ℝ × ℝ) := { p | let (x, y) := p in y = -(x - m)^2 + 2 * m }
noncomputable def setB : set (ℝ × ℝ) := ⋃ m, setBm m
noncomputable def intersectionArea : ℝ := arctan 2 - 2 / 5

theorem area_of_A_inter_B :
  ∃ area, area = intersectionArea :=
begin
  use intersectionArea,
  sorry
end

end area_of_A_inter_B_l678_678167


namespace probability_odd_sum_grid_l678_678306

def numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
def grid := Matrix (Fin 4) (Fin 4) ℕ

def is_odd_sum (lst : List ℕ) : Prop :=
  (List.sum lst) % 2 = 1

def valid_grid (m : grid) : Prop :=
  (∀ i, is_odd_sum (Matrix.row m i).toList) ∧
  (∀ j, is_odd_sum (Matrix.col m j).toList)

def total_arrangements := 16.factorial

noncomputable def num_valid_arrangements : ℕ := 
  2520 -- result from combinatorial calculation

theorem probability_odd_sum_grid :
  (num_valid_arrangements : ℚ) / total_arrangements = 1 / 118920 :=
by sorry

end probability_odd_sum_grid_l678_678306


namespace center_of_tangent_circle_l678_678744

theorem center_of_tangent_circle (x y : ℝ) 
    (h1 : 3 * x - 4 * y = 20) 
    (h2 : 3 * x - 4 * y = -40) 
    (h3 : x - 3 * y = 0) : 
    (x, y) = (-6, -2) := 
by
    sorry

end center_of_tangent_circle_l678_678744


namespace train_passes_man_in_approximately_18_seconds_l678_678043

noncomputable def length_of_train : ℝ := 330 -- meters
noncomputable def speed_of_train : ℝ := 60 -- kmph
noncomputable def speed_of_man : ℝ := 6 -- kmph

noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (5/18)

noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_of_train + speed_of_man)

noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_passes_man_in_approximately_18_seconds :
  abs (time_to_pass length_of_train relative_speed_mps - 18) < 1 :=
by
  sorry

end train_passes_man_in_approximately_18_seconds_l678_678043


namespace graduation_ceremony_sequences_l678_678461

theorem graduation_ceremony_sequences : 
  ∃ (A_pos : Fin 6 → Bool) (BC_pos : Fin 6 → Bool),
    (∑ i, A_pos i = 3) ∧ 
    (∑ i, BC_pos i = 1) ∧ 
    (∀ i, BC_pos i → ∃ j, BC_pos j ∧ Abs (i - j) = 1) ∧ 
    (∃ (f : Fin 6 → Fin 6), Injective f ∧ (∑ i, BC_pos (f i) = 1) ∧ (∑ i, A_pos (f i) = 3)) →
    ∃ s : Finset (Fin 6), s.card = 120 :=
sorry

end graduation_ceremony_sequences_l678_678461


namespace smallest_value_c_plus_d_l678_678620

noncomputable def problem1 (c d : ℝ) : Prop :=
c > 0 ∧ d > 0 ∧ (c^2 ≥ 12 * d) ∧ ((3 * d)^2 ≥ 4 * c)

theorem smallest_value_c_plus_d : ∃ c d : ℝ, problem1 c d ∧ c + d = 4 / Real.sqrt 3 + 4 / 9 :=
sorry

end smallest_value_c_plus_d_l678_678620


namespace mutated_frog_percentage_l678_678110

theorem mutated_frog_percentage 
  (extra_legs : ℕ) 
  (two_heads : ℕ) 
  (bright_red : ℕ) 
  (normal_frogs : ℕ) 
  (h_extra_legs : extra_legs = 5) 
  (h_two_heads : two_heads = 2) 
  (h_bright_red : bright_red = 2) 
  (h_normal_frogs : normal_frogs = 18) 
  : ((extra_legs + two_heads + bright_red) * 100 / (extra_legs + two_heads + bright_red + normal_frogs)).round = 33 := 
by
  sorry

end mutated_frog_percentage_l678_678110


namespace probability_BD_greater_than_10_l678_678704

theorem probability_BD_greater_than_10 :
  ∀ (A B C P D : Point),
    triangle_right_ABC A B C →
    ∠ ACB = 90 ∧ ∠ ABC = 45 ∧ AB = 10 * sqrt 2 →
    P ∈ triangle ABC →
    (extend_line_segment B P A C = D) →
    probability_BD_greater_than_10 A B C P D = 1 :=
begin
  sorry -- Proof will be provided here.
end

end probability_BD_greater_than_10_l678_678704


namespace area_inequality_l678_678577

variables {ABC K L M N R F : Type} -- Variables representing points

-- Definitions of areas and the main statement
def area_triangle (A B C : Type) : ℝ := sorry  -- Definition of the area of a triangle

-- Areas of sub-triangles
def E : ℝ := area_triangle ABC
def E1 : ℝ := area_triangle AMR
def E2 : ℝ := area_triangle CKR
def E3 : ℝ := area_triangle BKF
def E4 : ℝ := area_triangle ALF
def E5 : ℝ := area_triangle BNM
def E6 : ℝ := area_triangle CLN

-- Main statement of the problem
theorem area_inequality (ABC K L M N R F : Type) :
  E ≥ 8 * (real.sqrt (E1 * E2 * E3 * E4 * E5 * E6)) ^ (1/6) :=
begin
  sorry -- Proof goes here
end

end area_inequality_l678_678577


namespace triangle_with_medians_exists_l678_678235

-- Defining a triangle and its properties
structure Triangle :=
  (A B C : ℝ × ℝ)
  (area : ℝ)
  (median_a median_b median_c : ℝ)

-- The given triangle
def ABC : Triangle :=
  { A := (0, 0),
    B := (2, 0),
    C := (1, 1),
    area := 1,
    median_a := (1 / 2) * real.sqrt 10,
    median_b := (1 / 2) * real.sqrt 10,
    median_c := 1 }

-- The theorem statement proving the existence and area of the new triangle
theorem triangle_with_medians_exists (t : Triangle)
  (h1 : t.area = 1)
  (h2 : t.median_a = (1 / 2) * real.sqrt 10)
  (h3 : t.median_b = (1 / 2) * real.sqrt 10)
  (h4 : t.median_c = 1) :
  ∃ t' : Triangle, t'.median_a = t.median_a ∧ t'.median_b = t.median_b ∧ t'.median_c = t.median_c ∧ t'.area = 4 / 3 := 
by
  sorry

end triangle_with_medians_exists_l678_678235


namespace range_of_m_l678_678190

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x + 1

theorem range_of_m (m : ℝ) : (∀ x, x ≤ 1 → f m x ≥ f m 1) ↔ 0 ≤ m ∧ m ≤ 1 / 3 := by
  sorry

end range_of_m_l678_678190


namespace perpendicular_vectors_implies_value_of_m_l678_678168

def vector_a : ℝ × ℝ := (-2, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (3, m)

theorem perpendicular_vectors_implies_value_of_m (m : ℝ) (h : vector_a.1 * vector_b m.1 + vector_a.2 * vector_b m.2 = 0) : m = 2 :=
by 
  sorry

end perpendicular_vectors_implies_value_of_m_l678_678168


namespace line_through_N_and_M1_passes_through_incenter_l678_678883

def right_triangle (A B C : ℝ × ℝ) : Prop := 
  (∠ BAC = 90°)

def incenter (A B C : ℝ × ℝ) (I : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, ∀ P : ℝ × ℝ, (P ∈ {AB, BC, CA}) → dist P I = r

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + Q.1) / 2, (P.2 + Q.2) / 2

theorem line_through_N_and_M1_passes_through_incenter
  {A B C I N : ℝ × ℝ}
  (M1 := midpoint B C)
  (h_right_triangle : right_triangle A B C)
  (h_incenter : incenter A B C I)
  (h_AN_eq_r : ∃ r : ℝ, dist A N = r)
  (h_N_on_AC : N ∈ line_segment A C)
  (h_M1_mid_BC : M1 = midpoint B C) :
  ∃ l : line, N ∈ l ∧ M1 ∈ l ∧ I ∈ l :=
sorry

end line_through_N_and_M1_passes_through_incenter_l678_678883


namespace binary_to_base4_l678_678846

theorem binary_to_base4 :
  let binary := 1101001101
  -- Convert the binary to a list of base-4 digits
  let base4 := [3, 1, 1, 3, 1, 0]
  let binary_to_base4 (n : Nat) : List Nat :=
    -- Each group of 2 binary digits converted to base 4
    let bin_to_base4_digit :=
      λ b : Fin 4 => match b with
        | ⟨0, _⟩ => 0
        | ⟨1, _⟩ => 1
        | ⟨2, _⟩ => 2
        | ⟨3, _⟩ => 3
        | _ => 0
    -- Binary digits to base 4 conversion
    List.reverseBaseToListBase n [] bin_to_base4_digit 2 in
  binary_to_base4 binary = base4 → 1101001101₂ = "311310"


end binary_to_base4_l678_678846


namespace price_of_Microtron_stock_l678_678829

theorem price_of_Microtron_stock
  (n d : ℕ) (p_d p p_m : ℝ) 
  (h1 : n = 300) 
  (h2 : d = 150) 
  (h3 : p_d = 44) 
  (h4 : p = 40) 
  (h5 : p_m = 36) : 
  (d * p_d + (n - d) * p_m) / n = p := 
sorry

end price_of_Microtron_stock_l678_678829


namespace M4_d_M_eq_l678_678529

def M (n : ℕ) : Set ℕ := { S | ∃ l : List (Fin (2 * n)), Perm (range (2 * n + 1)) l ∧ S = (List.sum (l.pairwiseDifferences.map abs))}

def d_M (n : ℕ) : ℕ := M n |>.card

theorem M4 :
  M 4 = {4, 6, 8, 10, 12, 14, 16} :=
sorry

theorem d_M_eq (n : ℕ) :
  d_M n = (n^2 - n + 2) / 2 :=
sorry

end M4_d_M_eq_l678_678529


namespace find_circumcircle_diameter_l678_678569

open Real

noncomputable def circumcircle_diameter (a : ℝ) (B : ℝ) (S : ℝ) : ℝ :=
  let c := 4 * sqrt 2 in
  let b := sqrt (1 + 32 - 8) in
  b / sin B

theorem find_circumcircle_diameter :
  circumcircle_diameter 1 (π / 4) 2 = 5 * sqrt 2 :=
begin
  sorry
end

end find_circumcircle_diameter_l678_678569


namespace tank_overflow_time_l678_678647

noncomputable def rate_pipe_A := 1 / 32.0
noncomputable def rate_pipe_B := 3 * rate_pipe_A
noncomputable def rate_pipe_C := rate_pipe_B / 2

noncomputable def combined_rate := rate_pipe_A + rate_pipe_B + rate_pipe_C
noncomputable def time_to_fill_tank := 1 / combined_rate

theorem tank_overflow_time :
  abs (time_to_fill_tank - 5.82) < 0.01 :=
by
  sorry

end tank_overflow_time_l678_678647


namespace sum_of_factors_72_l678_678417

theorem sum_of_factors_72 : 
  ∑ d in (finset.filter (λ x, 72 % x = 0) (finset.range 73)), d = 195 := by
  sorry

end sum_of_factors_72_l678_678417


namespace student_comparison_l678_678010

def Student : Type := ℕ -- Representing each student by their height as a natural number.

-- Given arrangement: 10x20 grid representing students' heights.
variable (grid : Fin 10 → Fin 20 → Student)

-- Define the tallest student in each column (longitudinal row)
def tallest_in_column (j : Fin 20) : Student :=
  Finset.fold max 0 (Finset.univ.image (λ i, grid i j))

-- Define the shortest among the tallest students in each column
def shortest_of_tallest_in_columns : Student :=
  Finset.fold min (tallest_in_column ⟨0, sorry⟩) (Finset.univ.image tallest_in_column)

-- Define the shortest student in each row (transverse row)
def shortest_in_row (i : Fin 10) : Student :=
  Finset.fold min 0 (Finset.univ.image (λ j, grid i j))

-- Define the tallest among the shortest students in each row
def tallest_of_shortest_in_rows : Student :=
  Finset.fold max (shortest_in_row ⟨0, sorry⟩) (Finset.univ.image shortest_in_row)

-- Main theorem to prove: Student A (shortest of tallest in columns) is at least as tall as Student B (tallest of shortest in rows)
theorem student_comparison :
  shortest_of_tallest_in_columns grid ≥ tallest_of_shortest_in_rows grid := sorry

end student_comparison_l678_678010


namespace type_A_people_2014_l678_678573

def count_Type_A_people (n : Nat) : Nat :=
  if n % 2 = 0 then n / 2 else 0

theorem type_A_people_2014 : count_Type_A_people 2014 = 1007 := 
by
  unfold count_Type_A_people
  decide
  rfl

end type_A_people_2014_l678_678573


namespace vector_magnitude_sum_l678_678879

noncomputable def vecMag {α : Type*} [inner_product_space α ℝ] (v : α) : ℝ :=
  by
    sorry

variables (a b : ℝ)
variables (angle_ab : ℝ)

-- Conditions
hypothetical_1 : vecMag a = 1
hypothetical_2 : vecMag b = 2
hypothetical_3 : angle_ab = real.pi / 3 -- 60 degrees in radians

-- Proof
theorem vector_magnitude_sum (a b : ℝ) (angle_ab : ℝ) 
  (ha : vecMag a = 1) (hb : vecMag b = 2) 
  (hangle : angle_ab = real.pi / 3) : 
  vecMag (a + b) = real.sqrt 7 := 
by
  sorry

end vector_magnitude_sum_l678_678879


namespace dan_helmet_craters_l678_678428

variable (D S : ℕ)
variable (h1 : D = S + 10)
variable (h2 : D + S + 15 = 75)

theorem dan_helmet_craters : D = 35 := by
  sorry

end dan_helmet_craters_l678_678428


namespace total_tickets_sold_l678_678388

-- Define the parameters and conditions
def VIP_ticket_price : ℝ := 45.00
def general_ticket_price : ℝ := 20.00
def total_revenue : ℝ := 7500.00
def tickets_difference : ℕ := 276

-- Define the total number of tickets sold
def total_number_of_tickets (V G : ℕ) : ℕ := V + G

-- The theorem to be proved
theorem total_tickets_sold (V G : ℕ) 
  (h1 : VIP_ticket_price * V + general_ticket_price * G = total_revenue)
  (h2 : V = G - tickets_difference) : 
  total_number_of_tickets V G = 336 :=
by
  sorry

end total_tickets_sold_l678_678388


namespace sum_of_factors_72_l678_678416

theorem sum_of_factors_72 : ∑ d in (finset.filter (∣ 72) (finset.range (73))), d = 195 :=
by
  -- given condition: 72 = 2^3 * 3^2
  have h : factors 72 = [2, 2, 2, 3, 3],
  { sorry },
  -- steps to compute the sum of factors based on the prime factorization
  sorry

end sum_of_factors_72_l678_678416


namespace find_f_neg_2023_l678_678521

-- Given conditions
variables {a b c : ℝ}
def f (x : ℝ) : ℝ := a * x^3 + b * x - c / x + 2

-- Proof statement (question == answer given conditions)
theorem find_f_neg_2023 (h : f 2023 = 6) : f (-2023) = -2 :=
by
  sorry

end find_f_neg_2023_l678_678521


namespace num_two_digit_powers_of_3_l678_678963

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678963


namespace smallest_distance_in_equilateral_triangle_l678_678101

theorem smallest_distance_in_equilateral_triangle : 
  ∀ (A B C X : ℝ) (P Q : ℝ), 
    (A + B + C = 1) ∧ 
    (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ 
    (X > 0) ∧ (X < 1) ∧ 
    collinear X P Q ∧ 
    P ∈ sides A B C ∧ 
    Q ∈ sides A B C 
    → distance P Q ≥ (2 / 3) := 
sorry

end smallest_distance_in_equilateral_triangle_l678_678101


namespace kayak_trip_friends_l678_678763

theorem kayak_trip_friends :
  ∀ (G : simple_graph (fin 450)), (∀ v : (fin 450), degree v ≥ 100) →
  (∀ s : finset (fin 450), s.card = 200 → ∃ u v ∈ s, G.adj u v) →
  ∃ S : finset (fin 450), S.card = 302 ∧
  ∃ pairing : finset (fin 450 × fin 450), pairing.card = 151 ∧
  (∀ p ∈ pairing, ∃ u v : (fin 450), u ∈ S ∧ v ∈ S ∧ G.adj u v) :=
begin
  sorry
end

end kayak_trip_friends_l678_678763


namespace exists_increasing_a_l678_678223

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x)

theorem exists_increasing_a (a : ℝ) :
  (∀ x ∈ Set.Icc (2 : ℝ) (4 : ℝ), (a > 0) → (differentiable_at ℝ (λ x, Real.log (a * x^2 - x)) x) → 
  deriv (λ x, Real.log (a * x^2 - x)) x > 0) ↔ a > 1 :=
sorry

end exists_increasing_a_l678_678223


namespace volunteer_assignment_correct_l678_678869

def volunteerAssignment : ℕ := 5
def pavilions : ℕ := 4

def numberOfWays (volunteers pavilions : ℕ) : ℕ := 72 -- This is based on the provided correct answer.

theorem volunteer_assignment_correct : 
  numberOfWays volunteerAssignment pavilions = 72 := 
by
  sorry

end volunteer_assignment_correct_l678_678869


namespace triangle_isosceles_at_A_l678_678242

theorem triangle_isosceles_at_A 
  (Γ : Type) [circle Γ O]
  (A B C D : Γ)
  (hAB_gt_AO : AB > AO)
  (hC_on_bisector_OAB : bisector ( ∠ O A B ) ∩ Γ = {C})
  (D_on_circumcircle_OCB : second_intersection (circumcircle O C B) (AB) = D) :
  is_isosceles_at A (triangle A D O) :=
sorry

end triangle_isosceles_at_A_l678_678242


namespace initial_weight_of_beef_l678_678387

theorem initial_weight_of_beef (W : ℝ) 
  (stage1 : W' = 0.70 * W) 
  (stage2 : W'' = 0.80 * W') 
  (stage3 : W''' = 0.50 * W'') 
  (final_weight : W''' = 315) : 
  W = 1125 := by 
  sorry

end initial_weight_of_beef_l678_678387


namespace MB_length_l678_678706

noncomputable def length_MB (a b : ℝ) : ℝ := Real.sqrt (a * b)

theorem MB_length (a b MA MD : ℝ) (h1 : MA = a) (h2 : MD = b) : length_MB a b = sqrt (a * b) :=
by
  sorry

end MB_length_l678_678706


namespace propositions_correct_l678_678493

theorem propositions_correct :
  (∀ m : ℝ, let L := (3+m)*(-3) + 4*3 - 3 + 3*m in L = 0) ∧
  (∀ (x y : ℝ),
    (∃ (x1 y1 : ℝ),
    (x1, y1) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 } ∧ (x - (3 / 2))^2 + (y - 2)^2 = 1)) ∧
  (∀ b : ℝ,
    let M := { p : ℝ × ℝ | p.2 = √(1 - p.1^2) },
    N := { p : ℝ × ℝ | p.2 = p.1 + b } in
    M ∩ N ≠ ∅ → b ∈ [-√2, √2]) ∧
  (∀ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    let C := { p : ℝ × ℝ | (p.1 - b)^2 + (p.2 - c)^2 = a^2 } in
    (∃ p ∈ C, p.2 = 0) ∧ (∀ q ∈ C, q.1 ≠ 0) →
    (let inter := { x // ∃ y : ℝ, (a*x + b*y + c = 0) ∧ (x + y + 1 = 0) in x < 0 ∧ ∀ x, y > 0 } → y > 0) :=
begin
  sorry
end

end propositions_correct_l678_678493


namespace carlotta_performance_time_l678_678465

theorem carlotta_performance_time :
  ∀ (s p t : ℕ),  -- s for singing, p for practicing, t for tantrums
  (∀ (n : ℕ), p = 3 * n ∧ t = 5 * n) →
  s = 6 →
  (s + p + t) = 54 :=
by 
  intros s p t h1 h2
  rcases h1 1 with ⟨h3, h4⟩
  sorry

end carlotta_performance_time_l678_678465


namespace general_term_l678_678003

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, 1 ≤ n → a (n + 1) = (1/3) * a n + 1

theorem general_term (a : ℕ → ℝ) (h : sequence a) : 
    ∀ n : ℕ, 1 ≤ n → a n = (3/2) - (1/2) * (1/3)^(n-1) :=
sorry

end general_term_l678_678003


namespace concurrent_lines_l678_678320

theorem concurrent_lines 
  (a1 a2 b1 b2 c1 c2 : ℝ) :
  let A := (a1, a2) in
  let B := (b1, b2) in
  let C := (c1, c2) in
  let A1 := (-a1, a2) in
  let B1 := (-b1, b2) in
  let C1 := (-c1, c2) in
  let line1 := (c2 - b2, c1 - b1) * ⟨x, y⟩ = - (c2 - b2) * a1 + (c1 - b1) * a2 in
  let line2 := (a2 - c2, a1 - c1) * ⟨x, y⟩ = (a2 - c2) * b1 + (a1 - c1) * b2 in
  let line3 := (b2 - a2, b1 - a1) * ⟨x, y⟩ = (b2 - a2) * c1 + (b1 - a1) * c2 in
  ∃ P : ℝ × ℝ, P ∈ line1 ∧ P ∈ line2 ∧ P ∈ line3 :=
sorry

end concurrent_lines_l678_678320


namespace quadratic_inequality_solution_range_l678_678525

theorem quadratic_inequality_solution_range (a : ℝ) :
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + 1 / 4 ≤ 0) ↔ 0 < a ∧ a < 4 :=
by
  sorry

end quadratic_inequality_solution_range_l678_678525


namespace find_f_x_squared_l678_678125

-- Define the function f with the given condition
noncomputable def f (x : ℝ) : ℝ := (x + 1)^2

-- Theorem statement
theorem find_f_x_squared : f (x^2) = (x^2 + 1)^2 :=
by
  sorry

end find_f_x_squared_l678_678125


namespace sphere_radius_in_truncated_cone_l678_678393

noncomputable def truncated_cone_sphere_radius (r1 r2 h : ℝ) : ℝ :=
  let CH := Real.sqrt (r1 * r1 - (r1 - r2) * (r1 - r2))
  in CH - h

theorem sphere_radius_in_truncated_cone
  (radius_top radius_bottom height : ℝ)
  (h_radius_top : radius_top = 24)
  (h_radius_bottom : radius_bottom = 6)
  (h_height : height = 20)
  (h_tangent : True) -- This assumes the sphere is tangent to all required sides
  : truncated_cone_sphere_radius radius_top radius_bottom height = 4 :=
by
  sorry

end sphere_radius_in_truncated_cone_l678_678393


namespace min_units_type_A_l678_678369

-- Definitions based on given conditions
variables (x y k : ℝ)

-- Given conditions
def condition1 : Prop := x + y = 40
def condition2 : Prop := 90 / x = 150 / y
def condition3 : Prop := 15 * k + 25 * (100 - k) ≤ 2000

-- Main statement
theorem min_units_type_A 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  : x = 15 ∧ y = 25 ∧ k ≥ 50 :=
by
  sorry

end min_units_type_A_l678_678369


namespace neither_sufficient_nor_necessary_condition_l678_678500

theorem neither_sufficient_nor_necessary_condition (a b : ℝ) :
  ¬ ((a < 0 ∧ b < 0) → (a * b * (a - b) > 0)) ∧
  ¬ ((a * b * (a - b) > 0) → (a < 0 ∧ b < 0)) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l678_678500


namespace num_two_digit_powers_of_3_l678_678941

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678941


namespace prove_clothing_colors_l678_678813

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l678_678813


namespace count_two_digit_powers_of_three_l678_678981

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678981


namespace count_two_digit_powers_of_three_l678_678982

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678982


namespace find_f_prime_one_l678_678875

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x * f'(1)

theorem find_f_prime_one : (f'(1) = -2) :=
by 
  sorry

end find_f_prime_one_l678_678875


namespace infinite_subset_with_fixed_gcd_l678_678623

noncomputable theory
open Classical

def A (n : ℕ) : Prop := ∃ (p : ℕ → ℕ) (k : ℕ), (∀ i < k, Prime (p i)) ∧ (∀ i < k, p i ∣ n) ∧ (k ≤ 1990 ∧ n > 1)

theorem infinite_subset_with_fixed_gcd
  (A_infinite : ∀ (n : ℕ), A n → True)
  (A_condition : ∀ n, A n → n ∈ A) :
  ∃ B ⊆ A, Infinite B ∧ (∃ (d : ℕ), ∀ b1 b2 ∈ B, b1 ≠ b2 → gcd b1 b2 = d) :=
sorry

end infinite_subset_with_fixed_gcd_l678_678623


namespace harmony_numbers_count_l678_678709

def is_harmony_number (n : ℕ) : Prop :=
  (n >= 1000) ∧ (n < 10000) ∧ (n.digits.sum = 6)

def first_digit_is_two (n : ℕ) : Prop :=
  (n / 1000 = 2)

def harmony_numbers_with_first_digit_two : set ℕ :=
  {n : ℕ | is_harmony_number n ∧ first_digit_is_two n}

theorem harmony_numbers_count : 
  (harmony_numbers_with_first_digit_two.to_finset.card = 15) :=
by sorry

end harmony_numbers_count_l678_678709


namespace find_x_that_satisfies_fx_equals_2_l678_678514

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

theorem find_x_that_satisfies_fx_equals_2 :
  ∀ x : ℝ, f x = 2 → x = 0 := 
by
  intros x h
  sorry

end find_x_that_satisfies_fx_equals_2_l678_678514


namespace integer_solutions_to_equation_l678_678858

theorem integer_solutions_to_equation :
  { p : ℤ × ℤ | (p.1 ^ 2 * p.2 + 1 = p.1 ^ 2 + 2 * p.1 * p.2 + 2 * p.1 + p.2) } =
  { (-1, -1), (0, 1), (1, -1), (2, -7), (3, 7) } :=
by
  sorry

end integer_solutions_to_equation_l678_678858


namespace two_digit_powers_of_3_count_l678_678986

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678986


namespace symmetric_point_l678_678863

def P : (ℝ × ℝ × ℝ) := (4, -2, 6)  -- Define the point P with given coordinates
def symmetricP (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2, -p.3)  -- Define a function to find the symmetric point regarding xOy plane

theorem symmetric_point (P : ℝ × ℝ × ℝ) : symmetricP P = (P.1, P.2, -P.3) :=
by
  simp [symmetricP]
  sorry

end symmetric_point_l678_678863


namespace no_root_is_zero_l678_678090

theorem no_root_is_zero :
  ¬ (∃ x : ℝ, 3 * x^2 - 5 = 50 ∧ x = 0) ∧
  ¬ (∃ x : ℝ, (3 * x - 2)^2 = (x - 2)^2 ∧ x = 0) ∧
  ¬ (∃ x : ℝ, sqrt(x^2 - 15) = sqrt(x + 2) ∧ x = 0) :=
by
  split;
  { intro h,
    -- proof goes here
    sorry }

end no_root_is_zero_l678_678090


namespace max_linear_partitions_eq_l678_678237
open Set

def is_linear_partition {α : Type*} (S A B : Set α) : Prop :=
  A ∪ B = S ∧ A ∩ B = ∅ ∧ (∃ l : AffineLine ℝ 2, ∀ x ∈ S, x ∉ l)

noncomputable def max_linear_partitions (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

theorem max_linear_partitions_eq (S : Set (ℝ × ℝ)) (hS : finite S) (n : ℕ) 
  (h_card : S.card = n) : 
  ∃ A B : Set (ℝ × ℝ), is_linear_partition S A B ∧ n ≤ max_linear_partitions n :=
sorry

end max_linear_partitions_eq_l678_678237


namespace silvia_order_total_cost_l678_678665

theorem silvia_order_total_cost :
  let quiche_price : ℝ := 15
  let croissant_price : ℝ := 3
  let biscuit_price : ℝ := 2
  let quiche_count : ℝ := 2
  let croissant_count : ℝ := 6
  let biscuit_count : ℝ := 6
  let discount_rate : ℝ := 0.10
  let pre_discount_total : ℝ := (quiche_price * quiche_count) + (croissant_price * croissant_count) + (biscuit_price * biscuit_count)
  let discount_amount : ℝ := pre_discount_total * discount_rate
  let post_discount_total : ℝ := pre_discount_total - discount_amount
  pre_discount_total > 50 → post_discount_total = 54 :=
by
  sorry

end silvia_order_total_cost_l678_678665


namespace probability_y_gt_x_l678_678839

-- Define the uniform distribution and the problem setup
def uniform_distribution (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the variables
variables (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000)

-- Define the probability calculation function (assuming some proper definition for probability)
noncomputable def probability_event (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the event that Laurent's number is greater than Chloe's number
def event_y_gt_x : Set (ℝ × ℝ) := {p | p.2 > p.1}

-- State the theorem
theorem probability_y_gt_x (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000) :
  probability_event event_y_gt_x = 3/4 :=
sorry

end probability_y_gt_x_l678_678839


namespace base8_subtraction_l678_678062

theorem base8_subtraction : (53 - 26 : ℕ) = 25 :=
by sorry

end base8_subtraction_l678_678062


namespace line_passes_through_vertex_count_l678_678107

theorem line_passes_through_vertex_count :
  (∃ a : ℝ, ∀ (x : ℝ), x = 0 → (x + a = a^2)) ↔ (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_vertex_count_l678_678107


namespace find_m_max_min_FA_FB_product_l678_678523

noncomputable def ellipse_equation := ∀ (x y : ℝ), (x : ℝ) ^ 2 / 4 + (y : ℝ) ^ 2 / 3 = 1

def line_equation (m : ℝ) (α t : ℝ) : ℝ × ℝ :=
  (m + t * cos α, t * sin α)

def left_focus_ellipse : ℝ × ℝ :=
  (-1, 0)

theorem find_m (m α t : ℝ) :
  (line_equation m α t).fst = -1 :=
  sorry

noncomputable def FA_FB_product (α : ℝ) : ℝ :=
  9 / (3 + sin α ^ 2)

theorem max_min_FA_FB_product :
  (max_value : FA_FB_product 0 = 3) ∧ (min_value : FA_FB_product 1 = 9 / 4) :=
  sorry

end find_m_max_min_FA_FB_product_l678_678523


namespace part1_part2_l678_678654

theorem part1 (x : ℝ) : 2 * x^2 + 5 * x + 3 > x^2 + 3 * x + 1 :=
by
  sorry

theorem part2 {a b : ℝ} (h1 : a > b) (h2 : b > 0) : sqrt a > sqrt b :=
by
  sorry

end part1_part2_l678_678654


namespace average_score_for_girls_l678_678397

variable (A a B b : ℕ)
variable (h1 : 71 * A + 76 * a = 74 * (A + a))
variable (h2 : 81 * B + 90 * b = 84 * (B + b))
variable (h3 : 71 * A + 81 * B = 79 * (A + B))

theorem average_score_for_girls
  (h1 : 71 * A + 76 * a = 74 * (A + a))
  (h2 : 81 * B + 90 * b = 84 * (B + b))
  (h3 : 71 * A + 81 * B = 79 * (A + B))
  : (76 * a + 90 * b) / (a + b) = 84 := by
  sorry

end average_score_for_girls_l678_678397


namespace local_max_iff_range_of_a_l678_678901

theorem local_max_iff_range_of_a (f : ℝ → ℝ) (a : ℝ) (h_deriv : ∀ x, deriv f x = a * (x - a) * (x - 1)) :
  (∃ ε > 0, ∀ x, abs (x - 1) < ε → f(1) ≥ f(x)) ↔ (a < 0 ∨ a > 1) :=
sorry

end local_max_iff_range_of_a_l678_678901


namespace solution_correctness_l678_678082

-- Assuming p, q, r are primes
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the property of being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define the problem conditions
def condition (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_perfect_square (p^q + p^r)

-- Define the set of all solutions
def solutions : set (ℕ × ℕ × ℕ) :=
  { t | ∃ k : ℕ, is_prime k ∧ ¬ even k ∧ (t = (2, k, k)) } ∪
  { (3, 3, 2), (3, 2, 3) }

-- The theorem to prove
theorem solution_correctness :
  ∀ triplet : ℕ × ℕ × ℕ, condition triplet.1 triplet.2 triplet.3 ↔ triplet ∈ solutions :=
by sorry

end solution_correctness_l678_678082


namespace intersection_eq_l678_678195

def M : Set ℕ := {x | x < 6}
def N : Set ℤ := {x | (2 < x ∧ x < 9)}

theorem intersection_eq : (M ∩ N) = {3, 4, 5} := by
  sorry

end intersection_eq_l678_678195


namespace rectangle_to_semicircle_circumference_l678_678352

-- Define the geometric entities and properties.
def circumference_of_semicircle (diameter : ℝ) : ℝ :=
  (Real.pi * diameter) / 2 + diameter

theorem rectangle_to_semicircle_circumference :
  let length := 22
  let breadth := 16
  let perimeter_of_rect := 2 * (length + breadth)
  let side_of_square := perimeter_of_rect / 4
  let diameter_of_semicircle := side_of_square in
  side_of_square = diameter_of_semicircle ->
  circumference_of_semicircle diameter_of_semicircle ≈ 48.85 :=
  by
    intros
    sorry

end rectangle_to_semicircle_circumference_l678_678352


namespace median_number_of_moons_l678_678715

def celestial_bodies : List (String × Nat) := [
  ("Mercury", 0),
  ("Venus", 0),
  ("Earth", 1),
  ("Mars", 2),
  ("Jupiter", 20),
  ("Saturn", 25),
  ("Uranus", 18),
  ("Neptune", 2),
  ("Pluto", 5),
  ("Eris", 1),
  ("Submoon of Saturn's Moon", 1)
]

def median (l : List ℕ) : ℕ :=
  let sorted := l |>.sort (· ≤ ·)
  sorted.get (sorted.length / 2)

theorem median_number_of_moons : median (celestial_bodies.map (·.snd)) = 2 := by
  sorry

end median_number_of_moons_l678_678715


namespace rectangle_same_color_l678_678582

-- Definition of a strip in the Cartesian plane
def is_strip (n : ℤ) (p : ℝ × ℝ) : Prop :=
  n ≤ p.1 ∧ p.1 < n + 1

-- Definition of a color type
inductive Color
| red : Color
| blue : Color

-- Hypothesis that each strip is colored either red or blue
axiom strip_colored : ∀ (n : ℤ), Color

-- Condition: Given two distinct positive integers a and b
variables (a b : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_distinct : a ≠ b)

-- Statement to prove in Lean:
theorem rectangle_same_color : ∃ (p1 p2 p3 p4 : ℝ × ℝ),
  is_strip (p1.1.toInt) p1 ∧ is_strip (p2.1.toInt) p2 ∧
  is_strip (p3.1.toInt) p3 ∧ is_strip (p4.1.toInt) p4 ∧
  (strip_colored (p1.1.toInt) = strip_colored (p2.1.toInt)) ∧
  (strip_colored (p1.1.toInt) = strip_colored (p3.1.toInt)) ∧
  (strip_colored (p1.1.toInt) = strip_colored (p4.1.toInt)) ∧
  (p2.1 = p1.1 + a ∧ p2.2 = p1.2) ∧
  (p3.1 = p1.1 ∧ p3.2 = p1.2 + b) ∧
  (p4.1 = p1.1 + a ∧ p4.2 = p1.2 + b) :=
sorry

end rectangle_same_color_l678_678582


namespace f_odd_f_monotonically_increasing_f_max_min_l678_678912

def f (x : ℝ) : ℝ := x - 1 / x

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

theorem f_monotonically_increasing : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f x1 < f x2 := by
  sorry

theorem f_max_min : 
  f 1 = 0 ∧ f 4 = 15 / 4 :=
  by
  sorry

end f_odd_f_monotonically_increasing_f_max_min_l678_678912


namespace trigonometric_expression_l678_678138

theorem trigonometric_expression 
  (α : ℝ) 
  (h₁ : Real.tan α = -3 / 4)
  (h₂ : π / 2 < α ∧ α < π) : 
  sqrt 2 * Real.cos (α - π / 4) = -7 / 5 := sorry

end trigonometric_expression_l678_678138


namespace power_modulus_difference_l678_678069

theorem power_modulus_difference (m : ℤ) :
  (51 % 6 = 3) → (9 % 6 = 3) → ((51 : ℤ)^1723 - (9 : ℤ)^1723) % 6 = 0 :=
by 
  intros h1 h2
  sorry

end power_modulus_difference_l678_678069


namespace magnitude_of_a_is_2_l678_678534

variable {t : ℝ}

def vector_a (t : ℝ) : ℝ × ℝ := (1, t)
def vector_b (t : ℝ) : ℝ × ℝ := (-1, t)
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_a_is_2 (t : ℝ) (h : real.sqrt (1 + t^2) = 2) : 
magnitude (vector_a t) = 2 := by
  sorry

end magnitude_of_a_is_2_l678_678534


namespace subscription_ways_three_households_l678_678537

def num_subscription_ways (n_households : ℕ) (n_newspapers : ℕ) : ℕ :=
  if h : n_households = 3 ∧ n_newspapers = 5 then
    180
  else
    0

theorem subscription_ways_three_households :
  num_subscription_ways 3 5 = 180 :=
by
  unfold num_subscription_ways
  split_ifs
  . rfl
  . contradiction


end subscription_ways_three_households_l678_678537


namespace square_area_l678_678310

theorem square_area :
  let p1 := (1: ℝ, -2: ℝ)
  let p2 := (-3: ℝ, 5: ℝ)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 65 :=
by
  let p1 := (1: ℝ, -2: ℝ)
  let p2 := (-3: ℝ, 5: ℝ)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  calc side_length^2 
      = Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)^2 : by sorry
  ... = 65 : by sorry

end square_area_l678_678310


namespace entree_cost_14_l678_678173

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l678_678173


namespace B_can_win_with_initial_config_B_l678_678013

def initial_configuration_B := (6, 2, 1)

def A_starts_and_B_wins (config : (Nat × Nat × Nat)) : Prop := sorry

theorem B_can_win_with_initial_config_B : A_starts_and_B_wins initial_configuration_B :=
sorry

end B_can_win_with_initial_config_B_l678_678013


namespace additional_discount_during_sale_l678_678768

theorem additional_discount_during_sale:
  ∀ (list_price : ℝ) (max_typical_discount_pct : ℝ) (lowest_possible_sale_pct : ℝ),
  30 ≤ max_typical_discount_pct ∧ max_typical_discount_pct ≤ 50 ∧
  lowest_possible_sale_pct = 40 ∧ 
  list_price = 80 →
  ((max_typical_discount_pct * list_price / 100) - (lowest_possible_sale_pct * list_price / 100)) * 100 / 
    (max_typical_discount_pct * list_price / 100) = 20 :=
by
  sorry

end additional_discount_during_sale_l678_678768


namespace sqrt_12_bounds_l678_678092

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 :=
by
  sorry

end sqrt_12_bounds_l678_678092


namespace imaginary_part_of_z_l678_678254

noncomputable def imaginaryPart (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_z :
  ∀ (z : ℂ), (i * z = (Complex.abs (2 + i) + 2 * i)) →
  imaginaryPart z = -Real.sqrt 5 :=
by
  intros z h,
  sorry

end imaginary_part_of_z_l678_678254


namespace find_x_l678_678622

-- Let \( x \) be a real number such that 
-- \( x = 2 \left( \frac{1}{x} \cdot (-x) \right) - 5 \).
-- Prove \( x = -7 \).

theorem find_x (x : ℝ) (h : x = 2 * (1 / x * (-x)) - 5) : x = -7 :=
by
  sorry

end find_x_l678_678622


namespace average_calls_per_day_l678_678599

/-- Conditions: Jean's calls per day -/
def calls_mon : ℕ := 35
def calls_tue : ℕ := 46
def calls_wed : ℕ := 27
def calls_thu : ℕ := 61
def calls_fri : ℕ := 31

/-- Assertion: The average number of calls Jean answers per day -/
theorem average_calls_per_day :
  (calls_mon + calls_tue + calls_wed + calls_thu + calls_fri) / 5 = 40 :=
by sorry

end average_calls_per_day_l678_678599


namespace identify_clothes_l678_678796

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l678_678796


namespace sqrt_expr_value_l678_678073

theorem sqrt_expr_value : sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := 
by sorry

end sqrt_expr_value_l678_678073


namespace complete_the_square_result_l678_678722

-- Define the equation
def initial_eq (x : ℝ) : Prop := x^2 + 4 * x + 3 = 0

-- State the theorem based on the condition and required to prove the question equals the answer
theorem complete_the_square_result (x : ℝ) : initial_eq x → (x + 2) ^ 2 = 1 := 
by
  intro h
  -- Proof is to be skipped
  sorry

end complete_the_square_result_l678_678722


namespace identify_clothing_l678_678781

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l678_678781


namespace floor_tiling_l678_678340

-- Define that n can be expressed as 7k for some integer k.
theorem floor_tiling (n : ℕ) (h : ∃ x : ℕ, n^2 = 7 * x) : ∃ k : ℕ, n = 7 * k := by
  sorry

end floor_tiling_l678_678340


namespace right_triangle_median_length_l678_678710

theorem right_triangle_median_length (D E F : EuclideanSpace ℝ (Fin 2))
  (DE : D.dist E = 15) (EF : E.dist F = 20) (DF : D.dist F = 25) :
  let M : EuclideanSpace ℝ (Fin 2) := (E + F) / 2
  in D.dist M = 12.5 :=
sorry

end right_triangle_median_length_l678_678710


namespace telescoping_series_part_sum_l678_678592

theorem telescoping_series_part_sum (n : ℕ) (h : n > 1) :
  ∃ i j : ℤ, (i > 0) ∧ (j > i) ∧ (1 / (n : ℤ)) = ∑ k in (finset.range (j.to_nat - i.to_nat + 1)).map (λ k, i.to_nat + k), (1 / (k * (k + 1) : ℤ)) :=
  by
  sorry

end telescoping_series_part_sum_l678_678592


namespace time_to_cover_length_l678_678730

def escalator_rate : ℝ := 12 -- rate of the escalator in feet per second
def person_rate : ℝ := 8 -- rate of the person in feet per second
def escalator_length : ℝ := 160 -- length of the escalator in feet

theorem time_to_cover_length : escalator_length / (escalator_rate + person_rate) = 8 := by
  sorry

end time_to_cover_length_l678_678730


namespace harmonic_mean_pairs_l678_678106

theorem harmonic_mean_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x < y) 
    (hmean : (2 * x * y) / (x + y) = 2^30) :
    (∃! n, n = 29) :=
by
  sorry

end harmonic_mean_pairs_l678_678106


namespace area_of_triangle_formed_by_tangency_points_l678_678068

theorem area_of_triangle_formed_by_tangency_points :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  let O1O2 := r1 + r2
  let O2O3 := r2 + r3
  let O1O3 := r1 + r3
  let s := (O1O2 + O2O3 + O1O3) / 2
  let A := Real.sqrt (s * (s - O1O2) * (s - O2O3) * (s - O1O3))
  let r := A / s
  r^2 = 5 / 3 := 
by
  sorry

end area_of_triangle_formed_by_tangency_points_l678_678068


namespace smallest_number_to_add_quotient_of_resulting_number_l678_678457

theorem smallest_number_to_add (k : ℕ) : 456 ∣ (897326 + k) → k = 242 := 
sorry

theorem quotient_of_resulting_number : (897326 + 242) / 456 = 1968 := 
sorry

end smallest_number_to_add_quotient_of_resulting_number_l678_678457


namespace sum_of_arithmetic_sequence_l678_678880

noncomputable def geometric_sequence (a_n : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a_n (n + 1) = a_n n * r

noncomputable def arithmetic_sequence (b_n : ℕ → ℝ) :=
  ∃ d, ∀ n, b_n (n + 1) = b_n n + d

theorem sum_of_arithmetic_sequence 
  (a_n : ℕ → ℝ) (r : ℝ) (h1 : geometric_sequence a_n r) (h2 : r ≠ 1)
  (h3 : log 2 (a_n 1 * a_n 2 * a_n 3 * a_n 4 * a_n 5 * a_n 6 * a_n 7 * a_n 8 * a_n 9 * a_n 10 * a_n 11 * a_n 12 * a_n 13) = 13)
  (b_n : ℕ → ℝ) (h4 : arithmetic_sequence b_n)
  (h5 : b_n 7 = a_n 7) :
  b_n 1 + b_n 2 + b_n 3 + b_n 4 + b_n 5 + b_n 6 + b_n 7 + b_n 8 + b_n 9 + b_n 10 + b_n 11 + b_n 12 + b_n 13 = 26 
  := 
  sorry

end sum_of_arithmetic_sequence_l678_678880


namespace clothes_color_proof_l678_678784

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l678_678784


namespace three_x_plus_three_y_plus_three_z_l678_678191

theorem three_x_plus_three_y_plus_three_z (x y z : ℝ) 
  (h1 : y + z = 20 - 5 * x)
  (h2 : x + z = -18 - 5 * y)
  (h3 : x + y = 10 - 5 * z) :
  3 * x + 3 * y + 3 * z = 36 / 7 := by
  sorry

end three_x_plus_three_y_plus_three_z_l678_678191


namespace doubled_dimensions_new_volume_l678_678019

-- Define the original volume condition
def original_volume_condition (π r h : ℝ) : Prop := π * r^2 * h = 5

-- Define the new volume function after dimensions are doubled
def new_volume (π r h : ℝ) : ℝ := π * (2 * r)^2 * (2 * h)

-- The Lean statement for the proof problem 
theorem doubled_dimensions_new_volume (π r h : ℝ) (h_orig : original_volume_condition π r h) : 
  new_volume π r h = 40 :=
by 
  sorry

end doubled_dimensions_new_volume_l678_678019


namespace distance_EQ_l678_678703

theorem distance_EQ (EF FG GH HE : ℕ) (hEF : EF = 100) (hFG : FG = 40) (hGH : GH = 25) (hHE : HE = 80) (par_EF_GH : EF ∥ GH) :
  ∃ p q : ℕ, p = 10 ∧ q = 1 ∧ p + q = 11 :=
by
  use 10, 1
  constructor
  · sorry -- Proof that p = 10
  constructor
  · sorry -- Proof that q = 1
  · sorry -- Proof that p + q = 11

end distance_EQ_l678_678703


namespace alice_polygon_area_l678_678385

-- Definition: A regular octagon inscribed in a circle of radius 2
def regular_octagon_inscribed_circle_radius_2 {A B C D E F G H : Type} :=
is_regular_convex_polygon A B C D E F G H ∧
(inscribed_in_circle 2 A B C D E F G H)

-- Optimal play and game draw assumptions
axiom optimal_play_and_game_draw (Alice_points : set (point)) :
(Alice_points ⊆ {A B C D E F G H}) → 
(∀ A1 A2 A3 ∈ Alice_points, ¬ right_angle A1 A2 A3) → 
(∀ B1 B2 B3 ∈ Bob_points, ¬ right_angle B1 B2 B3) → 
(Alice_points ∪ Bob_points = {A B C D E F G H})

-- Possible resulting configurations for Alice's points and their areas
theorem alice_polygon_area (Alice_points : set point) :
  regular_octagon_inscribed_circle_radius_2 ∧
  optimal_play_and_game_draw Alice_points →
  (area (convex_hull Alice_points) = 2 * sqrt 2) ∨
  (area (convex_hull Alice_points) = 4 + 2 * sqrt 2) :=
sorry

end alice_polygon_area_l678_678385


namespace solve_problem_l678_678834

noncomputable def problem_statement : Prop :=
  let a := Real.arcsin (4/5)
  let b := Real.arccos (1/2)
  Real.sin (a + b) = (4 + 3 * Real.sqrt 3) / 10

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l678_678834


namespace arun_profit_calculation_l678_678405

theorem arun_profit_calculation :
  ∀ (w1 w2 : ℕ) (r1 r2 s : ℝ),
  w1 = 30 → w2 = 20 → r1 = 11.50 → r2 = 14.25 → s = 14.49 →
  let cost1 := w1 * r1 in
  let cost2 := w2 * r2 in
  let total_cost := cost1 + cost2 in
  let total_weight := w1 + w2 in
  let selling_price := total_weight * s in
  let profit := selling_price - total_cost in
  let profit_percentage := (profit / total_cost) * 100 in
  profit_percentage = 15 :=
by
  intros w1 w2 r1 r2 s hw1 hw2 hr1 hr2 hs
  simp [hw1, hw2, hr1, hr2, hs]
  let cost1 := 30 * 11.50
  let cost2 := 20 * 14.25
  let total_cost := cost1 + cost2
  let total_weight := 30 + 20
  let selling_price := total_weight * 14.49
  let profit := selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  have h1 : cost1 = 345 := by norm_num
  have h2 : cost2 = 285 := by norm_num
  have h3 : total_cost = 630 := by norm_num
  have h4 : total_weight = 50 := by norm_num
  have h5 : selling_price = 724.50 := by norm_num
  have h6 : profit = 724.50 - 630 := by norm_num
  have h7 : profit_percentage = (94.50 / 630) * 100 := by norm_num
  norm_num at *
  sorry

end arun_profit_calculation_l678_678405


namespace integer_count_satisfying_inequality_l678_678182

theorem integer_count_satisfying_inequality :
  { n : ℤ | -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 }.to_finset.card = 8 := 
sorry

end integer_count_satisfying_inequality_l678_678182


namespace problem_p3_l678_678000

theorem problem_p3 : 
  ∀ (x y z w : ℕ),
  23 * x + 47 * y - 3 * z = 434 →
  47 * x - 23 * y - 4 * w = 183 →
  19 * z + 17 * w = 91 →
  (13 * x - 14 * y)^3 - (15 * z + 16 * w)^3 = -456190 := by
  sorry

end problem_p3_l678_678000


namespace portion_of_pizza_eaten_l678_678749

-- Define the conditions
def total_slices : ℕ := 16
def slices_left : ℕ := 4
def slices_eaten : ℕ := total_slices - slices_left

-- Define the portion of pizza eaten
def portion_eaten := (slices_eaten : ℚ) / (total_slices : ℚ)

-- Statement to prove
theorem portion_of_pizza_eaten : portion_eaten = 3 / 4 :=
by sorry

end portion_of_pizza_eaten_l678_678749


namespace f_equals_alternate_binary_representation_l678_678870

noncomputable def f : ℚ → ℚ
| x := if 0 ≤ x ∧ x < 1/2 then f(2*x) / 4
       else if 1/2 ≤ x ∧ x < 1 then (3/4 : ℚ) + f(2*x - 1) / 4
       else 0

def binary_to_rational (b : List Bool) : ℚ :=
  b.foldr (λ (bit : Bool) (acc : ℚ), acc / 2 + if bit then 1 / 2 else 0) 0

def alternate_binary (b : List Bool) : List Bool :=
  b.foldr (λ (bit : Bool) (acc : List Bool), bit :: bit :: acc) []

def rationalize_alternate (b : List Bool) : ℚ :=
  binary_to_rational (alternate_binary b)

theorem f_equals_alternate_binary_representation :
  ∀ (b : List Bool),
  f (binary_to_rational b) = rationalize_alternate b :=
by
  sorry

end f_equals_alternate_binary_representation_l678_678870


namespace carlotta_performance_time_l678_678466

theorem carlotta_performance_time :
  ∀ (s p t : ℕ),  -- s for singing, p for practicing, t for tantrums
  (∀ (n : ℕ), p = 3 * n ∧ t = 5 * n) →
  s = 6 →
  (s + p + t) = 54 :=
by 
  intros s p t h1 h2
  rcases h1 1 with ⟨h3, h4⟩
  sorry

end carlotta_performance_time_l678_678466


namespace minimum_value_l678_678187

theorem minimum_value (a b : ℝ) (h1 : 4 ≤ a^2 + b^2) (h2 : a^2 + b^2 ≤ 9) : 2 ≤ a^2 - a * b + b^2 :=
sorry

example (a b : ℝ) (h1 : 4 ≤ a^2 + b^2) (h2 : a^2 + b^2 ≤ 9) : (∃ x, x = a^2 - a * b + b^2 ∧ x = 2) :=
begin
  use a^2 - a * b + b^2,
  split,
  { refl },
  { apply minimum_value,
    repeat { assumption } }
end

end minimum_value_l678_678187


namespace max_min_values_a2_monotonicity_range_l678_678515

def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - a * x - 1

noncomputable def f_a2 (x : ℝ) : ℝ := f x 2

theorem max_min_values_a2 :
  (∀ x ∈ set.Icc (-5) (5), f_a2 x ≤ 43 / 2) ∧ 
  (∃ x ∈ set.Icc (-5) (5), f_a2 x = 43 / 2) ∧
  (∀ x ∈ set.Icc (-5) (5), -3 ≤ f_a2 x) ∧ 
  (∃ x ∈ set.Icc (-5) (5), f_a2 x = -3) :=
sorry

theorem monotonicity_range (a : ℝ) :
  (∀ x1 x2 ∈ set.Icc (-5) (5), x1 < x2 → f x1 a ≤ f x2 a ∨ f x1 a ≥ f x2 a)
  → (a ≤ -5 ∨ a ≥ 5) :=
sorry

end max_min_values_a2_monotonicity_range_l678_678515


namespace sqrt_41_40_39_38_plus_1_l678_678072

theorem sqrt_41_40_39_38_plus_1 : Real.sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := by
  sorry

end sqrt_41_40_39_38_plus_1_l678_678072


namespace range_of_a_l678_678127

theorem range_of_a
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x, f x = x^3 - 3 * x)
  (h2 : ∀ y, y = f (f x - a)) :
  (∃ a > 0, y = f (f x - a) ∧ ∃ z, y z = 0 ∧ (2 - Real.sqrt 3 < a ∧ a < 2)) → 
  2 - Real.sqrt 3 < a ∧ a < 2 :=
  sorry

end range_of_a_l678_678127


namespace cyclic_quadrilateral_equal_sides_l678_678360

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop := sorry
noncomputable def same_area (ADM ABCM : Quadrilateral) : Prop := sorry
noncomputable def same_perimeter (ADM ABCM : Quadrilateral) : Prop := sorry

theorem cyclic_quadrilateral_equal_sides {A B C D : Point} {M : Point}
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : M ∈ line_segment C D)
  (h3 : same_area ⟨A D M⟩ ⟨A B C M⟩)
  (h4 : same_perimeter ⟨A D M⟩ ⟨A B C M⟩) :
  (dist A B = dist A D) ∨ (dist A D = dist B C) :=
begin
  sorry
end

end cyclic_quadrilateral_equal_sides_l678_678360


namespace six_digit_number_not_divisible_by_15_l678_678196

theorem six_digit_number_not_divisible_by_15
  (digits : Multiset ℕ)
  (h_digits : digits = {1, 2, 3, 4, 4, 6}) :
  ∀ n ∈ Multiset.permutations digits, ¬ (15 ∣ n) :=
by
  sorry

end six_digit_number_not_divisible_by_15_l678_678196


namespace price_of_AC_l678_678314

theorem price_of_AC (x : ℝ) (price_car price_ac : ℝ)
  (h1 : price_car = 3 * x) 
  (h2 : price_ac = 2 * x) 
  (h3 : price_car = price_ac + 500) : 
  price_ac = 1000 := sorry

end price_of_AC_l678_678314


namespace mutant_frog_percentage_proof_l678_678116

/-- Number of frogs with extra legs -/
def frogs_with_extra_legs := 5

/-- Number of frogs with 2 heads -/
def frogs_with_two_heads := 2

/-- Number of frogs that are bright red -/
def frogs_bright_red := 2

/-- Number of normal frogs -/
def normal_frogs := 18

/-- Total number of mutant frogs -/
def total_mutant_frogs := frogs_with_extra_legs + frogs_with_two_heads + frogs_bright_red

/-- Total number of frogs -/
def total_frogs := total_mutant_frogs + normal_frogs

/-- Calculate the percentage of mutant frogs rounded to the nearest integer -/
def mutant_frog_percentage : ℕ := (total_mutant_frogs * 100 / total_frogs).toNat

theorem mutant_frog_percentage_proof:
  mutant_frog_percentage = 33 := 
  by 
    -- Proof skipped
    sorry

end mutant_frog_percentage_proof_l678_678116


namespace income_of_second_member_l678_678321

theorem income_of_second_member (n : ℕ) (avg_income total_income income1 income2 income3 income4 : ℕ) 
  (h1 : n = 4) (h2 : avg_income = 10000) (h3 : total_income = avg_income * n)
  (h4 : income1 = 8000) (h5 : income2 = 6000) (h6 : income3 = 11000) 
  (h7 : total_income = income1 + income2 + income3 + income4) : income4 = 15000 :=
by
  rw [h2, h1] at h3
  rw [Nat.mul_comm] at h3
  rw [h4, h5, h6] at h3
  sorry

end income_of_second_member_l678_678321


namespace clothes_color_proof_l678_678783

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l678_678783


namespace functional_equation_solution_l678_678860

noncomputable def unique_function_f (f : ℝ → ℝ) :=
  (∀ x y : ℝ, f(x^4 + y) = x^3 * f(x) + f(f(y))) ∧
  ({x : ℝ | f(x) = 0}.finite)

theorem functional_equation_solution (f : ℝ → ℝ) (h : unique_function_f f) : 
  ∀ x : ℝ, f(x) = x :=
by
  sorry

end functional_equation_solution_l678_678860


namespace angle_DMR_eq_MDR_l678_678483

-- Definitions
variables (Γ : Type) [circle_Γ : circle Γ]
variables (A B C H D E P M N R : point Γ)
variables (acute_angled_ABC : is_acute_angled_triangle A B C)
variables (orthocenter_H : is_orthocenter H A B C)
variables (AD_perp_BC : perp AD BC D)
variables (BE_perp_AC : perp BE AC E)
variables (P_on_minor_arc_BC : on_minor_arc P B C Γ)
variables (M_proj_bc : projection P M BC)
variables (N_proj_ac : projection P N AC)
variables (R_intersection_PH_MN : intersection R PH MN)

-- Theorem statement
theorem angle_DMR_eq_MDR : angle D M R = angle M D R :=
sorry -- Proof is not required

end angle_DMR_eq_MDR_l678_678483


namespace open_spots_level4_correct_l678_678757

noncomputable def open_spots_level_4 (total_levels : ℕ) (spots_per_level : ℕ) (open_spots_level1 : ℕ) (open_spots_level2 : ℕ) (open_spots_level3 : ℕ) (full_spots_total : ℕ) : ℕ := 
  let total_spots := total_levels * spots_per_level
  let open_spots_total := total_spots - full_spots_total 
  let open_spots_first_three := open_spots_level1 + open_spots_level2 + open_spots_level3
  open_spots_total - open_spots_first_three

theorem open_spots_level4_correct :
  open_spots_level_4 4 100 58 (58 + 2) (58 + 2 + 5) 186 = 31 :=
by
  sorry

end open_spots_level4_correct_l678_678757


namespace zoo_sea_lions_l678_678407

variable (S P : ℕ)

theorem zoo_sea_lions (h1 : S / P = 4 / 11) (h2 : P = S + 84) : S = 48 := 
sorry

end zoo_sea_lions_l678_678407


namespace C_investment_is_12000_l678_678396

-- Define the investments and profits as given in the conditions
def A_investment : ℝ := 8000
def B_investment : ℝ := 10000
def B_profit : ℝ := 3000
def profit_difference_AC : ℝ := 1199.9999999999998

-- Define the constants for solving the problem
def profit_per_rupee := B_profit / B_investment
def A_profit := A_investment * profit_per_rupee

-- Statement of the problem in Lean
theorem C_investment_is_12000 : ∃ C_investment : ℝ, 
  (profit_difference_AC + A_profit) = (C_investment * profit_per_rupee) ∧ 
  C_investment = 12000 := 
by 
  existsi (12000 : ℝ)
  sorry

end C_investment_is_12000_l678_678396


namespace plane_equation_valid_l678_678758

variables 
  (x1 y1 z1 x2 y2 z2 m : ℝ)
  (plane_eq : ℝ → ℝ → ℝ)

-- Define the plane equation
def plane_eq (x y : ℝ) : ℝ := 
  let C := z1 - real.sqrt 3 * x1 in
  plane_eq x y = real.sqrt 3 * x - (z1 - real.sqrt 3 * x1)

-- Define the distance from P2 to the plane
def distance_to_plane : ℝ := 
  (real.abs (real.sqrt 3 * x2 - z2 + z1 - real.sqrt 3 * x1)) / 2

-- The main goal
theorem plane_equation_valid :
  plane_eq = λ x y, real.sqrt 3 * x - z1 + real.sqrt 3 * x1 = 2 * m ∨ 
             plane_eq = λ x y, real.sqrt 3 * x - z1 + real.sqrt 3 * x1 = -2 * m :=
by sorry

end plane_equation_valid_l678_678758


namespace solve_quadratic_eq1_solve_quadratic_eq2_l678_678670

theorem solve_quadratic_eq1 (x : ℝ) :
  x^2 - 4 * x + 3 = 0 ↔ (x = 3 ∨ x = 1) :=
sorry

theorem solve_quadratic_eq2 (x : ℝ) :
  x^2 - x - 3 = 0 ↔ (x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) :=
sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l678_678670


namespace lines_intersect_at_single_point_l678_678358

open Classical
noncomputable theory

-- Definitions of points and lines
structure Point (α : Type) := (x y z : α)
structure Line (α : Type) := (p1 p2 : Point α)

-- Assuming the given conditions as definitions
variables {α : Type} [Field α]

def P : Point α := sorry
def A1 : Point α := sorry
def A2 : Point α := sorry
def B1 : Point α := sorry
def B2 : Point α := sorry
def C1 : Point α := sorry
def C2 : Point α := sorry

-- Definition of centers of spheres
def O (i j k : Nat) : Point α :=
  -- O_ijk is the center of the sphere passing through {A_i, B_j, C_k, P}
  sorry

-- The lines formed by the diagonals
def L1 : Line α := ⟨O 1 1 1, O 2 2 2⟩
def L2 : Line α := ⟨O 1 1 2, O 2 2 1⟩
def L3 : Line α := ⟨O 1 2 1, O 2 1 2⟩
def L4 : Line α := ⟨O 2 1 1, O 1 2 2⟩

-- The statement to prove
theorem lines_intersect_at_single_point :
  ∃ Q : Point α, (true : Prop) ∧
  (L1.p1 = Q ∨ L1.p2 = Q) ∧
  (L2.p1 = Q ∨ L2.p2 = Q) ∧
  (L3.p1 = Q ∨ L3.p2 = Q) ∧
  (L4.p1 = Q ∨ L4.p2 = Q) :=
sorry

end lines_intersect_at_single_point_l678_678358


namespace seating_arrangement_constraint_l678_678580

theorem seating_arrangement_constraint : 
  let total_arrangements := fact 8
  let alice_bob_together := fact 7 * fact 2
  let cindy_dave_together := fact 7 * fact 2
  let both_pairs_together := fact 6 * fact 2 * fact 2
  total_arrangements - (2 * alice_bob_together + 2 * cindy_dave_together - both_pairs_together) = 23040 :=
by
  -- Define factorials to avoid repetition
  let fact_8 := 40320
  let fact_7 := 5040
  let fact_6 := 720
  have total_arrangements_eq : fact 8 = fact_8 := by rfl
  have alice_bob_together_eq : fact 7 * fact 2 = 2 * fact_7 := by rfl
  have cindy_dave_together_eq : fact 7 * fact 2 = 2 * fact_7 := by rfl
  have both_pairs_together_eq : fact 6 * fact 2 * fact 2 = 4 * fact_6 := by rfl
  calc
    40320 - (2 * 5040 * 2 + 2 * 5040 * 2 - 720 * 4) = 23040 := by sorry

end seating_arrangement_constraint_l678_678580


namespace two_digit_powers_of_3_count_l678_678990

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678990


namespace sqrt_inequality_l678_678380

theorem sqrt_inequality (x : ℝ) (hx : x > 0) : (sqrt x > 3 * x) ↔ (x < 1 / 9) :=
by {
  sorry
}

end sqrt_inequality_l678_678380


namespace even_function_a_eq_one_l678_678559

noncomputable def f (x a : ℝ) : ℝ := x * Real.log (x + Real.sqrt (a + x ^ 2))

theorem even_function_a_eq_one (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 1 :=
by
  sorry

end even_function_a_eq_one_l678_678559


namespace min_value_of_log_func_l678_678303

noncomputable def f (x : ℝ) : ℝ := log (x^2 - 2 * x + 3)

theorem min_value_of_log_func : (∀ x ∈ set.univ, f x ≥ log 2) :=
begin
  -- use the conditions directly
  sorry
end

end min_value_of_log_func_l678_678303


namespace cost_per_sq_foot_is_20_l678_678384

-- Defining the parameters
variable (l w h total_cost : ℕ)
#check nat -- nat is used for natural numbers
#check nat.div_nat -- for integer division

-- Surface area calculation function
def surface_area (l w h : ℕ) : ℕ :=
  2 * l * w + 2 * l * h + 2 * w * h

-- Definition of the problem's conditions 
def conditions : Prop := 
  l = 4 ∧ w = 5 ∧ h = 3 ∧ total_cost = 1880

-- The main goal: proving cost per square foot
def cost_per_sq_foot (l w h total_cost : ℕ) : ℕ :=
  total_cost / (surface_area l w h)

-- Statement of the theorem
theorem cost_per_sq_foot_is_20 (l w h total_cost : ℕ) (hc : conditions l w h total_cost) : 
  cost_per_sq_foot l w h total_cost = 20 :=
by
  -- We are only required to state the theorem, so insert sorry and skip the proof.
  sorry

end cost_per_sq_foot_is_20_l678_678384


namespace exists_winning_teams_l678_678091

theorem exists_winning_teams 
  (teams : Fin 8 → ℕ) -- Representing the teams as indexed by natural numbers for simplicity.
  (played : ∀ i j : Fin 8, i ≠ j → (teams i) ≠ (teams j)) -- Round-robin condition where each team plays against each other exactly once.
  (win : {i j : Fin 8 // i ≠ j} → Prop) -- Win relation between teams.
  :
  ∃ (A B C D : Fin 8), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    win ⟨A, B, sorry⟩ ∧ win ⟨A, C, sorry⟩ ∧ win ⟨A, D, sorry⟩ ∧
    win ⟨B, C, sorry⟩ ∧ win ⟨B, D, sorry⟩ ∧
    win ⟨C, D, sorry⟩ := 
sorry

end exists_winning_teams_l678_678091


namespace sqrt_12_bounds_l678_678093

theorem sqrt_12_bounds : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 :=
by
  sorry

end sqrt_12_bounds_l678_678093


namespace greater_than_89_times_315_sum_of_two_abundant_l678_678379

open Nat

-- Definition of abundant
def is_abundant (n : ℕ) : Prop :=
  (n > 0) ∧ ((∑ d in divisors n, d) - n > n)

-- The theorem to be proven
theorem greater_than_89_times_315_sum_of_two_abundant:
  ∀ n : ℕ, n > 89 * 315 → ∃ a b : ℕ, is_abundant a ∧ is_abundant b ∧ n = a + b :=
by
  assume n
  intro hn
  sorry

end greater_than_89_times_315_sum_of_two_abundant_l678_678379


namespace average_calls_per_day_l678_678601

theorem average_calls_per_day :
  let calls := [35, 46, 27, 61, 31] in
  (calls.sum / (calls.length : ℝ)) = 40 :=
by
  sorry

end average_calls_per_day_l678_678601


namespace num_two_digit_powers_of_3_l678_678958

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678958


namespace sum_floor_a_k_l678_678425

def a_k (k : ℕ) : ℝ := 2^100 / (2^50 + 2^k)

theorem sum_floor_a_k :
  ∑ k in Finset.range (101), (⌊a_k k⌋ : ℤ) = 101 * 2^49 - 50 :=
by
  sorry

end sum_floor_a_k_l678_678425


namespace clothes_color_proof_l678_678786

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l678_678786


namespace two_digit_powers_of_three_l678_678945

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678945


namespace production_value_approx_l678_678609

noncomputable def total_production_value (initial_value : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  (initial_value * (growth_rate ^ years)) + 
  (initial_value * (growth_rate ^ (years - 1))) +
  (initial_value * (growth_rate ^ (years - 2))) +
  (initial_value * (growth_rate ^ (years - 3))) +
  (initial_value * (growth_rate ^ (years - 4)))

theorem production_value_approx :
  let initial_value := 1
  let growth_rate := 1.1
  total_production_value initial_value growth_rate 5 ≈ 6.6 :=
by
  let initial_value := 1
  let growth_rate := 1.1
  have approx : (growth_rate ^ 5) ≈ 1.6 := sorry
  have h₁ : initial_value * (growth_rate ^ 5) ≈ initial_value * 1.6 := by apply approx
  have h₂ : initial_value * 1.1 = 1.1 := by simp
  have h₃ : initial_value * (1.1 ^ 2) = 1.21 := by norm_num [pow_two, mul_assoc]
  have h₄ : initial_value * (1.1 ^ 3) = 1.331 := by norm_num [pow_succ, pow_two, mul_assoc]
  have h₅ : initial_value * (1.1 ^ 4) = 1.4641 := by norm_num [pow_succ, pow_two, mul_assoc]
  have total : 1.1 + 1.21 + 1.331 + 1.4641 + 1.6 ≈ 6.6 := by sorry
  show total_production_value initial_value growth_rate 5 ≈ 6.6, from sorry

end production_value_approx_l678_678609


namespace sum_of_squares_l678_678892

variables (x y z w : ℝ)

def condition1 := (x^2 / (2^2 - 1^2)) + (y^2 / (2^2 - 3^2)) + (z^2 / (2^2 - 5^2)) + (w^2 / (2^2 - 7^2)) = 1
def condition2 := (x^2 / (4^2 - 1^2)) + (y^2 / (4^2 - 3^2)) + (z^2 / (4^2 - 5^2)) + (w^2 / (4^2 - 7^2)) = 1
def condition3 := (x^2 / (6^2 - 1^2)) + (y^2 / (6^2 - 3^2)) + (z^2 / (6^2 - 5^2)) + (w^2 / (6^2 - 7^2)) = 1
def condition4 := (x^2 / (8^2 - 1^2)) + (y^2 / (8^2 - 3^2)) + (z^2 / (8^2 - 5^2)) + (w^2 / (8^2 - 7^2)) = 1

theorem sum_of_squares : condition1 x y z w → condition2 x y z w → 
                          condition3 x y z w → condition4 x y z w →
                          (x^2 + y^2 + z^2 + w^2 = 36) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_of_squares_l678_678892


namespace kendra_total_earnings_l678_678231

-- Definitions of the conditions based on the problem statement
def kendra_earnings_2014 : ℕ := 30000 - 8000
def laurel_earnings_2014 : ℕ := 30000
def kendra_earnings_2015 : ℕ := laurel_earnings_2014 + (laurel_earnings_2014 / 5)

-- The statement to be proved
theorem kendra_total_earnings : kendra_earnings_2014 + kendra_earnings_2015 = 58000 :=
by
  -- Using Lean tactics for the proof
  sorry

end kendra_total_earnings_l678_678231


namespace chosen_number_l678_678038

theorem chosen_number (x : ℝ) (h : 2 * x - 138 = 102) : x = 120 := by
  sorry

end chosen_number_l678_678038


namespace entree_cost_14_l678_678170

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l678_678170


namespace problem_statement_l678_678478

theorem problem_statement (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 :=
by sorry

end problem_statement_l678_678478


namespace angle_in_third_quadrant_l678_678009

-- Define the concept of an angle being in a specific quadrant
def is_in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

-- Prove that -1200° is in the third quadrant
theorem angle_in_third_quadrant :
  is_in_third_quadrant (240) → is_in_third_quadrant (-1200 % 360 + 360 * (if -1200 % 360 ≤ 0 then 1 else 0)) :=
by
  sorry

end angle_in_third_quadrant_l678_678009


namespace max_min_distance_sum_l678_678688

variable (a b c x y z : ℝ)

theorem max_min_distance_sum (h1 : a ≤ b) (h2 : b ≤ c) (h3 : x^2 + y^2 + z^2 = 2) 
    (h4 : 0 ≤ x) (h5 : x ≤ 1) (h6 : 0 ≤ y) (h7 : y ≤ 1) (h8 : 0 ≤ z) (h9 : z ≤ 1) :
    ∃ max min, 
      max = c + sqrt(a^2 + b^2) ∧
      min = a + b ∧
      max = a * x + b * y + c * z ∧
      min = a * x + b * y + c * z :=
begin
  sorry
end

end max_min_distance_sum_l678_678688


namespace syrup_cost_l678_678029

theorem syrup_cost {brownie ice_cream nuts total cost_per_serving : ℝ}
  (h_brownie : brownie = 2.50)
  (h_ice_cream : ice_cream = 1.00)
  (h_nuts : nuts = 1.50)
  (h_total : total = 7.00)
  (h_scoops : 2)
  (h_double_syrup : 2)
  (h_cost_serving : h_scoops * ice_cream + brownie + nuts = 6.00) :
  cost_per_serving = 0.50 :=
by
  have h_known_costs : 2 * ice_cream + brownie + nuts = 6.00, from h_cost_serving,
  have h_double_syrup_cost : total - h_known_costs = 1.00, from (h_total - h_known_costs), -- Calculating double syrup cost
  show cost_per_serving = 0.50, from (h_double_syrup_cost / h_double_syrup).symm

end syrup_cost_l678_678029


namespace smallest_positive_period_and_intervals_of_monotonic_increase_find_value_of_a_l678_678533

noncomputable def vectorA (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)
noncomputable def vectorB (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, -1/2)
noncomputable def f (x : ℝ) : ℝ := 
  let a := vectorA x
  let b := vectorB x
  ((a.1 + b.1) * a.1 + (a.2 + b.2) * a.2) - 2

theorem smallest_positive_period_and_intervals_of_monotonic_increase :
  ∃ T : ℝ, T = Real.pi ∧ 
  ∀ k : ℤ, ∃ l u : ℝ, (l = k * Real.pi - Real.pi / 3) ∧ (u = k * Real.pi + Real.pi / 6) ∧ 
  ∀ x : ℝ, (l <= x ∧ x <= u) -> f x is strictly increasing := sorry

theorem find_value_of_a 
  (A : ℝ) (a b c : ℝ) 
  (h1 : (b, a, c) are_arithmetic_progression) 
  (h2 : f A = 1/2) 
  (h3 : b * c * Real.cos(A) = 9) : 
  a = 3 * Real.sqrt 2 := sorry

end smallest_positive_period_and_intervals_of_monotonic_increase_find_value_of_a_l678_678533


namespace largest_possible_value_l678_678629

theorem largest_possible_value (x y z : ℕ) (hx : x > 1) (hy : y > 1) (hz : z > 1) (h : ∑ k in Finset.range (x+1), Nat.factorial k = y^2) : 
  (x = 3 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 3 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 4) → x + y + z = 8 := 
by
  sorry

end largest_possible_value_l678_678629


namespace probability_problem_l678_678518

noncomputable def f (x : ℝ) := 3 * x - 4 / x

theorem probability_problem :
  let D := set.Ioo (-1 : ℝ) (11 : ℝ)
      interval := set.Ioo (1 : ℝ) (4 : ℝ)
  in ∀ x₀ ∈ interval, set.Ioo (2 : ℝ) (4 : ℝ)
  in ∀ (x₀ : ℝ), x₀ ∈ interval → (f x₀ / x₀ ≥ 2) →
      (∃ (P : ℚ), P = 2 / 3) ∧ (P = 2 /3) :=
begin
  intros x₀ hx₀,
  sorry
end

end probability_problem_l678_678518


namespace dan_helmet_craters_l678_678430

namespace HelmetCraters

variables {Dan Daniel Rin : ℕ}

/-- Condition 1: Dan's skateboarding helmet has ten more craters than Daniel's ski helmet. -/
def condition1 (C_d C_daniel : ℕ) : Prop := C_d = C_daniel + 10

/-- Condition 2: Rin's snorkel helmet has 15 more craters than Dan's and Daniel's helmets combined. -/
def condition2 (C_r C_d C_daniel : ℕ) : Prop := C_r = C_d + C_daniel + 15

/-- Condition 3: Rin's helmet has 75 craters. -/
def condition3 (C_r : ℕ) : Prop := C_r = 75

/-- The main theorem: Dan's skateboarding helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (C_d C_daniel C_r : ℕ) 
    (h1 : condition1 C_d C_daniel) 
    (h2 : condition2 C_r C_d C_daniel) 
    (h3 : condition3 C_r) : C_d = 35 :=
by {
    -- We state that the answer is 35 based on the conditions
    sorry
}

end HelmetCraters

end dan_helmet_craters_l678_678430


namespace two_digit_powers_of_three_l678_678949

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678949


namespace identify_clothes_l678_678797

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l678_678797


namespace cos_72_deg_l678_678070

theorem cos_72_deg :
  cos (72 * Real.pi / 180) = (-1 + Real.sqrt 5) / 4 :=
by
  sorry

end cos_72_deg_l678_678070


namespace two_digit_numbers_in_form_3_pow_n_l678_678964

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678964


namespace determine_disco_ball_price_l678_678067

variable (x y z : ℝ)

-- Given conditions
def budget_constraint : Prop := 4 * x + 10 * y + 20 * z = 600
def food_cost : Prop := y = 0.85 * x
def decoration_cost : Prop := z = x / 2 - 10

-- Goal
theorem determine_disco_ball_price (h1 : budget_constraint x y z) (h2 : food_cost x y) (h3 : decoration_cost x z) :
  x = 35.56 :=
sorry 

end determine_disco_ball_price_l678_678067


namespace tangent_line_at_zero_l678_678685

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_at_zero :
  ∃ (m b : ℝ), (∀ (x : ℝ), f x = Real.exp x) ∧ (∀ x, f' x = Real.exp x) ∧ m = 1 ∧ b = 1 ∧ ∀ x, (f x - b = m * (x - 0)) → (y = x + 1) :=
begin
  -- Conditions
  sorry
end

end tangent_line_at_zero_l678_678685


namespace john_cleans_entire_house_in_4_hours_l678_678726

variables (J N : ℝ)

def John_time : Prop :=
  (1/3.6) = (1/(2 * N / 3)) + (1/N)

def clean_half : Prop :=
  (J = 2 * N / 3)

theorem john_cleans_entire_house_in_4_hours 
  (h1 : clean_half J N)
  (h2 : John_time J N) :
  J = 4 :=
by
  sorry

end john_cleans_entire_house_in_4_hours_l678_678726


namespace even_integers_150_800_count_l678_678179

def is_even (n : ℕ) : Prop := n % 2 = 0

def digits_are_all_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.nodup

def contains_no_0 (n : ℕ) : Prop :=
  ¬ (0 ∈ n.digits 10)

def digits_in_set (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ∈ {1, 3, 4, 6, 7, 8}

def valid_even_integers_count : ℕ :=
  (List.range' 151 649).countp (λ n, is_even n ∧ digits_are_all_different n ∧ contains_no_0 n ∧ digits_in_set n)

theorem even_integers_150_800_count :
  valid_even_integers_count = 39 :=
sorry

end even_integers_150_800_count_l678_678179


namespace symmetric_points_result_l678_678891

def symmetric_x_axis (P Q : ℝ × ℝ) : Prop :=
P.1 = Q.1 ∧ P.2 = -Q.2

theorem symmetric_points_result (a b : ℝ) 
  (h : symmetric_x_axis (a, 3) (4, b)) : 
  (a + b) ^ 2021 = 1 :=
by
  have ha : a = 4 := h.1
  have hb : b = -3 := by 
    rw [←h.2, neg_neg]
  rw [ha, hb]
  norm_num

end symmetric_points_result_l678_678891


namespace percentage_mutant_frogs_is_33_l678_678112

def num_extra_legs_frogs := 5
def num_two_heads_frogs := 2
def num_bright_red_frogs := 2
def num_normal_frogs := 18

def total_mutant_frogs := num_extra_legs_frogs + num_two_heads_frogs + num_bright_red_frogs
def total_frogs := total_mutant_frogs + num_normal_frogs

theorem percentage_mutant_frogs_is_33 :
  Float.round (100 * total_mutant_frogs.toFloat / total_frogs.toFloat) = 33 :=
by 
  -- placeholder for the proof
  sorry

end percentage_mutant_frogs_is_33_l678_678112


namespace identify_clothing_l678_678777

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l678_678777


namespace finite_f_omega_l678_678882

noncomputable def f_series (f : ℚ → ℚ) : ℕ → Set ℚ  
| 0       := Set.univ  
| (n + 1) := f '' (f_series f n)  

noncomputable def f_omega (f : ℚ → ℚ) : Set ℚ := 
⋂ n, f_series f n

theorem finite_f_omega 
  (f : ℚ → ℚ) 
  (hf : ∀ x, polynomial.eval x (polynomial.map (coe : ℚ → ℝ) (polynomial.of_fn (list.of_fn (f x)))) = f x)
  (deg_polynomial : 2 ≤ (polynomial.of_fn (list.of_fn (f 0))).natDegree) :
  (f_omega f).finite :=
begin
  sorry,
end

end finite_f_omega_l678_678882


namespace find_a_for_area_l678_678482

theorem find_a_for_area (a : ℝ) (h1 : a > 0) (h2 : ∫ x in 0..a, sqrt x = a^2) : a = 4 / 9 :=
sorry

end find_a_for_area_l678_678482


namespace average_score_of_students_l678_678304

-- Define the conditions and the required average score calculation
theorem average_score_of_students {M F : ℝ} (h1 : M = 0.4 * (M + F)) 
  (h2 : ∀ x, x ∈ set.univ → x ≤ 75) (h3 : ∀ x, x ∈ set.univ → x ≤ 80) 
  : ((75 * M) + (80 * F)) / (M + F) = 78 :=
sorry

end average_score_of_students_l678_678304


namespace largest_square_side_l678_678261

theorem largest_square_side {m n : ℕ} (h1 : m = 72) (h2 : n = 90) : Nat.gcd m n = 18 :=
by
  sorry

end largest_square_side_l678_678261


namespace problem1_problem2_l678_678364

-- Part 1: Prove that the expression simplifies to 5
theorem problem1 : real.sqrt 2 + (1 / 2) ^ (-2 : ℤ) + (-1 : ℝ) ^ 0 - 2 * real.sin (real.pi / 4) = 5 := 
by
  sorry

-- Part 2: Prove that the only positive integer solutions for the system are x = 1 and y = 13
theorem problem2 (x y : ℕ) (h1 : 2 * x + y = 15) (h2 : y + 7 * x ≤ 22) : x = 1 ∧ y = 13 := 
by
  sorry

end problem1_problem2_l678_678364


namespace two_digit_numbers_in_form_3_pow_n_l678_678970

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678970


namespace sum_dk_squared_l678_678433

noncomputable def d (k : ℕ) : ℚ := k + 1 / (3 * k + d k)

theorem sum_dk_squared : ∑ k in Finset.range 10, (d (k + 1))^2 = 395 :=
by
  sorry

end sum_dk_squared_l678_678433


namespace solve_eq_l678_678669

theorem solve_eq : ∃ x : ℝ, (x ≠ 0) ∧ (x^2 + 3 * x + 2) / (x^2 + 1) = x - 2 :=
by
  use 4
  split
  {
    simp, -- Show that 4 ≠ 0
  },
  {
    sorry -- This part is left for the actual proof
  }

end solve_eq_l678_678669


namespace prove_clothing_colors_l678_678811

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l678_678811


namespace arithmetic_expression_eval_l678_678063

theorem arithmetic_expression_eval : 8 / 4 - 3 - 9 + 3 * 9 = 17 :=
by
  sorry

end arithmetic_expression_eval_l678_678063


namespace brad_net_profit_l678_678410

-- Definitions based on the conditions
def glasses_per_gallon_small := 16
def glasses_per_gallon_medium := 10
def glasses_per_gallon_large := 6

def cost_per_gallon_small := 2.00
def cost_per_gallon_medium := 3.50
def cost_per_gallon_large := 5.00

def price_per_glass_small := 1.00
def price_per_glass_medium := 1.75
def price_per_glass_large := 2.50

def gallons_made_each_size := 2

def small_glasses_total := 32
def medium_glasses_total := 20
def large_glasses_total := 12

def small_glasses_drank := 4
def medium_glasses_bought := 3
def medium_glasses_spilled := 1
def large_glasses_unsold := 2

def stand_setup_cost := 15.00
def advertisement_cost := 10.00

-- Definition of net profit calculation
def net_profit : ℝ :=
  let total_cost := 2 * cost_per_gallon_small + 
                    2 * cost_per_gallon_medium + 
                    2 * cost_per_gallon_large + 
                    stand_setup_cost + 
                    advertisement_cost in
  let total_revenue := (32 - 4) * price_per_glass_small +
                       (20 - 3 - 1) * price_per_glass_medium +
                       (12 - 2) * price_per_glass_large in
  total_revenue - total_cost

-- Proof statement
theorem brad_net_profit : net_profit = 35.00 := by
  sorry

end brad_net_profit_l678_678410


namespace find_a_value_l678_678900

theorem find_a_value (a x : ℝ) (h : (x + a / x) * (2 * x - 1)^5) :
  ∃ a, ∃ c, (∀ x ≠ 0, c = 30 → a = 3) :=
begin
  sorry
end

end find_a_value_l678_678900


namespace cone_volume_l678_678506

theorem cone_volume {r_arc l_arc : ℝ} (h_r_arc : r_arc = 2) (h_l_arc : l_arc = 2 * Real.pi) :
  let r_cone := l_arc / (2 * Real.pi),
      h_cone := Real.sqrt (r_arc^2 - r_cone^2),
      V := (1/3) * Real.pi * r_cone^2 * h_cone
  in V = (Real.sqrt 3 * Real.pi) / 3 := by
  sorry

end cone_volume_l678_678506


namespace explicit_form_of_f_l678_678248

noncomputable def f (x : ℝ) : ℝ := sorry

theorem explicit_form_of_f :
  (∀ x : ℝ, f x + f (x + 3) = 0) →
  (∀ x : ℝ, -1 < x ∧ x ≤ 1 → f x = 2 * x - 3) →
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → f x = -2 * x + 9) :=
by
  intros h1 h2
  sorry

end explicit_form_of_f_l678_678248


namespace monotonicity_of_f_compare_f_values_l678_678516

open Real

def f (x : ℝ) : ℝ := x + x^3

theorem monotonicity_of_f : ∀ x y : ℝ, x < y → f(x) < f(y) := 
by 
  sorry

theorem compare_f_values (a b : ℝ) (h : a + b > 0) : f(a) + f(b) > 0 := 
by
  sorry

end monotonicity_of_f_compare_f_values_l678_678516


namespace part1_S_2018_eq_241_part2_exists_alpha_beta_l678_678104

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def nearest_perfect_square_distance (n : ℕ) : ℕ :=
  min (n - (Nat.sqrt n) ^ 2) (((Nat.sqrt n) + 1) ^ 2 - n)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_good_number (n : ℕ) : Prop :=
  is_perfect_square n ∨ is_perfect_cube (nearest_perfect_square_distance n)

def S (N : ℕ) : ℕ :=
  (Finset.range (N + 1)).filter is_good_number |>.card

theorem part1_S_2018_eq_241 : S 2018 = 241 :=
  sorry

theorem part2_exists_alpha_beta :
  ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (∀ eps > 0, ∃ N0 : ℕ, ∀ N ≥ N0, abs (S N / N^α - β) < eps) :=
  let α := 2 / 3
  let β := 3 / 2
  show ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (∀ eps > 0, ∃ N0 : ℕ, ∀ N ≥ N0, abs (S N / N^α - β) < eps), from
  ⟨α, β, by norm_num, by norm_num, sorry⟩

end part1_S_2018_eq_241_part2_exists_alpha_beta_l678_678104


namespace minimum_f_of_a_l678_678913

variable (a : ℝ) (x : ℝ)

def domain_y (x : ℝ) : Prop := 1 < x ∧ x ≤ 2

def f (a x : ℝ) := a * 2^(x+2) + 3 * 4^x

theorem minimum_f_of_a (h_a : a < -3) (h_x : domain_y x) :
  f a x ≥ if a ≤ -6 then 48 + 16 * a else - (4 * a^2) / 3 :=
sorry

end minimum_f_of_a_l678_678913


namespace unique_g_function_l678_678241

open Set

noncomputable def S : Set ℝ := {x | x ≠ 0}
noncomputable def g (f : S → ℝ) : Prop :=
  f 1 = 2 ∧
  (∀ x y ∈ S, (x + y ∈ S) → f (x + y) = f x + f y) ∧
  (∀ x y ∈ S, (x + y ∈ S) → (x + y) * f (x + y) = x * y * f x * f y)

theorem unique_g_function : ∀ f : S → ℝ, g f ↔ f = (λ x, 2 / x) := sorry

end unique_g_function_l678_678241


namespace num_two_digit_powers_of_3_l678_678940

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678940


namespace probability_of_selection_l678_678567

open Nat

/-- The wardrobe contains 6 shirts, 7 pairs of pants, and 8 pairs of socks. -/
structure Wardrobe :=
  (shirts : ℕ)
  (pants : ℕ)
  (socks : ℕ)

def myWardrobe : Wardrobe := { shirts := 6, pants := 7, socks := 8 }

/-- Probability of selecting exactly two shirts, one pair of pants, and one pair of socks -/
theorem probability_of_selection (w : Wardrobe) (total_clothing : ℕ) (selected_clothing : ℕ) :
  total_clothing = w.shirts + w.pants + w.socks →
  selected_clothing = 4 →
  (6.choose 2) * (7.choose 1) * (8.choose 1) / (total_clothing.choose selected_clothing) = 40 / 285 :=
by
  intros h_total h_selected
  sorry

end probability_of_selection_l678_678567


namespace find_lambda_mu_nu_l678_678121

open_locale classical

variables {R : Type*} [Field R]
variables {V : Type*} [AddCommGroup V] [Module R V]

-- Define the vectors as given in the conditions
def m : V := sorry
def j : V := sorry
def k : V := sorry

def a1 : V := 2 • m - j + k
def a2 : V := m + 3 • j - 2 • k
def a3 : V := -2 • m + j - 3 • k
def a4 : V := 3 • m + 2 • j + 5 • k

-- Pairwise orthogonality and unit vectors assumed
axiom orthogonal_m_j (h : Orthogonal m j) : true
axiom orthogonal_m_k (h : Orthogonal m k) : true
axiom orthogonal_j_k (h : Orthogonal j k) : true

axiom unit_vector_m (h : ∥m∥ = 1) : true
axiom unit_vector_j (h : ∥j∥ = 1) : true
axiom unit_vector_k (h : ∥k∥ = 1) : true

theorem find_lambda_mu_nu (λ μ ν : R)
  (h : a4 = λ • a1 + μ • a2 + ν • a3) : λ = -2 ∧ μ = 1 ∧ ν = -3 := 
by sorry

end find_lambda_mu_nu_l678_678121


namespace determine_clothes_l678_678822

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l678_678822


namespace identify_clothing_l678_678780

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l678_678780


namespace percent_of_x_is_y_l678_678732

variable (x y : ℝ)

theorem percent_of_x_is_y (h : 0.20 * (x - y) = 0.15 * (x + y)) : (y / x) * 100 = 100 / 7 :=
by
  sorry

end percent_of_x_is_y_l678_678732


namespace number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l678_678588

-- Let's define the conditions.
def balls_in_first_pocket : Nat := 2
def balls_in_second_pocket : Nat := 4
def balls_in_third_pocket : Nat := 5

-- Proof for the first question
theorem number_of_ways_to_take_one_ball_from_pockets : 
  balls_in_first_pocket + balls_in_second_pocket + balls_in_third_pocket = 11 := 
by
  sorry

-- Proof for the second question
theorem number_of_ways_to_take_one_ball_each_from_pockets : 
  balls_in_first_pocket * balls_in_second_pocket * balls_in_third_pocket = 40 := 
by
  sorry

end number_of_ways_to_take_one_ball_from_pockets_number_of_ways_to_take_one_ball_each_from_pockets_l678_678588


namespace integer_count_satisfying_inequality_l678_678181

theorem integer_count_satisfying_inequality :
  { n : ℤ | -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 }.to_finset.card = 8 := 
sorry

end integer_count_satisfying_inequality_l678_678181


namespace terry_the_tiger_area_l678_678287

theorem terry_the_tiger_area (a b c p q : ℤ) (h1 : cube_edge_length = 2)
  (h2 : leash_length = 2)
  (h3 : tethered_at_center_of_face)
  (h4 : ∀ point, short_dist_along_cube_face point ≤ 2 → point ∈ roaming_area)
  (h5 : area_formula = (p * π) / q + a * (sqrt b) + c)
  (h6 : ∀ x, x^2 > 1 → ¬ ∃ (n : ℤ), n ≠ 0 ∧ x ∣ n^2)
  (h7 : gcd p q = 1)
  (h8 : q > 0) :
  p + q + a + b + c = 14 := sorry

end terry_the_tiger_area_l678_678287


namespace nth_term_correct_l678_678574

noncomputable def term_in_sequence (n : ℕ) : ℚ :=
  2^n / (2^n + 3)

theorem nth_term_correct (n : ℕ) : term_in_sequence n = 2^n / (2^n + 3) :=
by
  sorry

end nth_term_correct_l678_678574


namespace neg_p_exists_x_l678_678919

-- Let p be the proposition: For all x in ℝ, x^2 - 3x + 3 > 0
def p : Prop := ∀ x : ℝ, x^2 - 3 * x + 3 > 0

-- Prove that the negation of p implies that there exists some x in ℝ such that x^2 - 3x + 3 ≤ 0
theorem neg_p_exists_x : ¬p ↔ ∃ x : ℝ, x^2 - 3 * x + 3 ≤ 0 :=
by {
  sorry
}

end neg_p_exists_x_l678_678919


namespace exponentiation_rule_l678_678837

theorem exponentiation_rule (b : ℝ) : (-2 * b) ^ 3 = -8 * b ^ 3 :=
by sorry

end exponentiation_rule_l678_678837


namespace range_of_a_l678_678512

noncomputable def f (x : ℝ) (k a : ℝ) :=
  if -1 ≤ x ∧ x < k then
    log 2 (1 - x) + 1
  else if k ≤ x ∧ x ≤ a then
    x ^ 2 - 2 * x + 1
  else
    0
-- our function

theorem range_of_a (k a : ℝ) (hk : -1 ≤ k) (ha : k ≤ a) :
  (∃ (k : ℝ), range f x k a = set.Icc 0 2) ↔ a ∈ set.Ioc (1 / 2) (1 + real.sqrt 2) :=
by
  sorry

end range_of_a_l678_678512


namespace Kerry_age_l678_678606

theorem Kerry_age :
  (let cost_per_box := 2.5 in
   let total_cost := 5 in
   let number_of_boxes := total_cost / cost_per_box in
   let candles_per_box := 12 in
   let total_candles := number_of_boxes * candles_per_box in
   let number_of_cakes := 3 in
   let kerry_age := total_candles / number_of_cakes in
   kerry_age = 8) :=
by
  sorry

end Kerry_age_l678_678606


namespace part1_part2_l678_678536

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (cos x, (sqrt 3) * cos x)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (cos x, sin x)
noncomputable def f (x : ℝ) : ℝ := (cos x)*(cos x) + (sqrt 3)*(cos x)*(sin x)

theorem part1 (x : ℝ) (h : x ∈ set.Icc 0 (↑(Real.pi) / 2)) :
  (vec_a x).1 * (vec_b x).2 = (vec_a x).2 * (vec_b x).1 → x = (↑(Real.pi) / 2) ∨ x = (↑(Real.pi) / 3) :=
  sorry

theorem part2 :
  ∃ x ∈ set.Icc 0 (↑(Real.pi) / 2), f x = (3 / 2) :=
  sorry

end part1_part2_l678_678536


namespace probability_of_one_pair_l678_678645

-- Define the number of socks and the draw
def num_socks := 10
def num_colors := 5
def draw_socks := 5

-- Define the condition that there are two socks of each color
def socks_per_color := 2

-- Define the probability calculation
def total_combinations : ℕ := Nat.choose num_socks draw_socks

def favorable_combinations : ℕ :=
  (Nat.choose num_colors 1) * -- Choose 1 color for the pair
  (Nat.choose socks_per_color 2) * -- From chosen color pick the pair
  (Nat.choose (num_colors - 1) (draw_socks - 2)) * -- Choose 3 different colors from remaining 4
  (Nat.choose socks_per_color 1) ^ 3 -- From each of these 3 different colors, pick 1 sock

def probability : ℚ :=
  favorable_combinations.to_rat / total_combinations.to_rat

-- Proof statement
theorem probability_of_one_pair : probability = 20 / 31.5 :=
by
  sorry

end probability_of_one_pair_l678_678645


namespace f_log2_5_value_l678_678157

def f (x : ℝ) : ℝ :=
if x < 1 then 2^x else f (x - 1)

noncomputable def f_log2_5 : ℝ :=
f (Real.log 5 / Real.log 2)

theorem f_log2_5_value : f_log2_5 = 5 / 4 := by
  sorry

end f_log2_5_value_l678_678157


namespace problem_theorem_l678_678253

noncomputable def proof_problem (a b c d : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : 0 < b) (h₃ : b < 1) (h₄ : 0 < c) (h₅ : c < 1) (h₆ : 0 < d) (h₇ : d < 1) (h_sum : a + b + c + d = 2) : Prop :=
  sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ≤ (a * c + b * d) / 2

theorem problem_theorem (a b c d : ℝ) (h₀ : 0 < a) (h₁ : a < 1) (h₂ : 0 < b) (h₃ : b < 1) (h₄ : 0 < c) (h₅ : c < 1) (h₆ : 0 < d) (h₇ : d < 1) (h_sum : a + b + c + d = 2) :
  proof_problem a b c d h₀ h₁ h₂ h₃ h₄ h₅ h₆ h₇ h_sum :=
by sorry

end problem_theorem_l678_678253


namespace find_lines_passing_through_A_parallel_to_beta_and_angle_with_alpha_l678_678028

-- Define the given elements as variables
variable (A : Point) (alpha beta : Plane) (theta : ℝ)

-- Statement of the problem
theorem find_lines_passing_through_A_parallel_to_beta_and_angle_with_alpha :
  ∃ (B C : Point),
    line_through A B ∧ line_through A C ∧
    parallel_to (line_through A B) beta ∧ parallel_to (line_through A C) beta ∧
    angle_with_plane (line_through A B) alpha = θ ∧ angle_with_plane (line_through A C) alpha = θ :=
by
  sorry

end find_lines_passing_through_A_parallel_to_beta_and_angle_with_alpha_l678_678028


namespace cupcakes_per_package_l678_678826

theorem cupcakes_per_package (total_cupcakes : ℕ) (eaten_cupcakes : ℕ) (packages : ℕ)
  (h_total : total_cupcakes = 50)
  (h_eaten : eaten_cupcakes = 5)
  (h_packages : packages = 9) :
  (total_cupcakes - eaten_cupcakes) / packages = 5 :=
by
  rw [h_total, h_eaten, h_packages]
  simp
  sorry

end cupcakes_per_package_l678_678826


namespace hyperbola_eccentricity_range_l678_678916

def hyperbola (x y b : ℝ) := x^2 - (y^2 / b^2) - 1 = 0
def circle (x y : ℝ) := x^2 + (y - 2)^2 - 1 = 0
def asymptote_condition (b : ℝ) := (2 / (sqrt (b^2 + 1))) ≥ 1
def eccentricity (b : ℝ) := sqrt (1 + b^2)

theorem hyperbola_eccentricity_range
  (b : ℝ)
  (hb : b > 0)
  (asym_cond : asymptote_condition b) 
  : 1 < eccentricity b ∧ eccentricity b ≤ 2 :=
begin
  sorry
end

end hyperbola_eccentricity_range_l678_678916


namespace probability_diamond_or_ace_l678_678368

theorem probability_diamond_or_ace (total_cards : ℕ) (diamonds : ℕ) (aces : ℕ) (jokers : ℕ)
  (not_diamonds_nor_aces : ℕ) (p_not_diamond_nor_ace : ℚ) (p_both_not_diamond_nor_ace : ℚ) : 
  total_cards = 54 →
  diamonds = 13 →
  aces = 4 →
  jokers = 2 →
  not_diamonds_nor_aces = 38 →
  p_not_diamond_nor_ace = 19 / 27 →
  p_both_not_diamond_nor_ace = (19 / 27) ^ 2 →
  1 - p_both_not_diamond_nor_ace = 368 / 729 :=
by 
  intros
  sorry

end probability_diamond_or_ace_l678_678368


namespace percentage_of_alcohol_in_second_vessel_l678_678394

-- Define the problem conditions
def capacity1 : ℝ := 2
def percentage1 : ℝ := 0.35
def alcohol1 := capacity1 * percentage1

def capacity2 : ℝ := 6 
def percentage2 (x : ℝ) : ℝ := 0.01 * x
def alcohol2 (x : ℝ) := capacity2 * percentage2 x

def total_capacity : ℝ := 8
def final_percentage : ℝ := 0.37
def total_alcohol := total_capacity * final_percentage

theorem percentage_of_alcohol_in_second_vessel (x : ℝ) :
  alcohol1 + alcohol2 x = total_alcohol → x = 37.67 :=
by sorry

end percentage_of_alcohol_in_second_vessel_l678_678394


namespace number_of_such_functions_l678_678257

def M : Set ℤ := { -2, 0, 1 }
def N : Set ℤ := { 1, 2, 3, 4, 5 }

def is_odd (n : ℤ) : Prop := n % 2 = 1 ∨ n % 2 = -1

def satisfies_condition (f : ℤ → ℤ) (x : ℤ) : Prop :=
x + f x + x * f x % 2 ≠ 0

theorem number_of_such_functions : 
  ∃ f : {x // x ∈ M} → {y // y ∈ N}, 
    (∀ x : {x // x ∈ M}, satisfies_condition (λ x, (f x).val) x.val) ∧ 
       (set_finite (set_of (λ f, ∀ x : {x // x ∈ M}, satisfies_condition (λ x, (f x).val) x.val)) = 45) :=
sorry

end number_of_such_functions_l678_678257


namespace participants_are_multiple_of_7_l678_678408

theorem participants_are_multiple_of_7 (P : ℕ) (h1 : P % 2 = 0)
  (h2 : ∀ p, p = P / 2 → P + p / 7 = (4 * P) / 7)
  (h3 : (4 * P) / 7 * 7 = 4 * P) : ∃ k : ℕ, P = 7 * k := 
by
  sorry

end participants_are_multiple_of_7_l678_678408


namespace log_sum_equality_l678_678442

noncomputable def evaluate_log_sum : ℝ :=
  3 / (Real.log 1000^4 / Real.log 8) + 4 / (Real.log 1000^4 / Real.log 10)

theorem log_sum_equality :
  evaluate_log_sum = (9 * Real.log 2 / Real.log 10 + 4) / 12 :=
by
  sorry

end log_sum_equality_l678_678442


namespace disagree_parents_count_l678_678390

def total_parents : ℕ := 800  -- Total number of parents surveyed
def agree_percentage : ℝ := 20  -- Percentage of parents who agree to the tuition fee increase

def disagree_parents (P : ℕ) (A : ℝ) : ℝ := (1 - A / 100) * P

theorem disagree_parents_count :
  disagree_parents total_parents agree_percentage = 640 := by
  sorry

end disagree_parents_count_l678_678390


namespace true_discount_initial_time_l678_678017

noncomputable def TD (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := (P * R * T) / 100

theorem true_discount_initial_time :
  ∀ (P : ℝ) (TD_double : ℝ),
  P = 110 →
  TD_double = 18.333333333333332 →
  TD P (8.333333333333332 / P) (1 / 2) = 9.166666666666666 :=
by
  intros P TD_double hP hTD_double
  rw [hP, hTD_double]
  sorry  -- Proof is omitted as per the instructions

end true_discount_initial_time_l678_678017


namespace percentage_mutant_frogs_is_33_l678_678114

def num_extra_legs_frogs := 5
def num_two_heads_frogs := 2
def num_bright_red_frogs := 2
def num_normal_frogs := 18

def total_mutant_frogs := num_extra_legs_frogs + num_two_heads_frogs + num_bright_red_frogs
def total_frogs := total_mutant_frogs + num_normal_frogs

theorem percentage_mutant_frogs_is_33 :
  Float.round (100 * total_mutant_frogs.toFloat / total_frogs.toFloat) = 33 :=
by 
  -- placeholder for the proof
  sorry

end percentage_mutant_frogs_is_33_l678_678114


namespace Triangle_BG_GE_BF_is_right_triangle_l678_678212

theorem Triangle_BG_GE_BF_is_right_triangle
  {A B C D E F G H : Point}
  (h_acute : is_acute_triangle A B C)
  (h_angle_bisector : is_angle_bisector AD A B C)
  (h_de : perpendicular DE AC)
  (h_df : perpendicular DF AB)
  (h_e_on_bc : foot_of_perpendicular E D AC)
  (h_f_on_ab : foot_of_perpendicular F D AB)
  (H_intersection : intersect BE CF H)
  (circumcircle : circumcircle (triangle A F H))
  (G_on_circumcircle : G ∈ circumcircle)
  : is_right_triangle (triangle B G E) :=
sorry

end Triangle_BG_GE_BF_is_right_triangle_l678_678212


namespace sector_central_angle_l678_678146

theorem sector_central_angle (r l α : ℝ) (h1 : 2 * r + l = 6) (h2 : 1/2 * l * r = 2) :
  α = l / r → (α = 1 ∨ α = 4) :=
by
  sorry

end sector_central_angle_l678_678146


namespace number_of_knights_l678_678270

def isKnight : ℕ → Prop := sorry
def isLiar : ℕ → Prop := sorry

axiom KnightsAlwaysTellTruth (n : ℕ) (h : isKnight n) : 
  isKnight (n+1) ∨ isKnight (n-1)

axiom LiarsAlwaysLie (n : ℕ) (h : isLiar n) :
  isLiar (n+1) ∧ isLiar (n-1) = false

def carousel := list ℕ

def initialSetup : carousel := sorry -- 39 initial people

def additionalLiar : ℕ := sorry -- The additional Liar

axiom conformPattern : ∀ n, n < 40 → 
  if isKnight n then (isKnight (n+1) ∨ isKnight (n-1))
  else (isLiar n ∧ (isKnight (n+1) ∧ isKnight (n-1)))

theorem number_of_knights : ∃ N, N = 26 :=
  sorry

end number_of_knights_l678_678270


namespace distinct_perfect_square_sums_l678_678664

-- Define the problem statement
theorem distinct_perfect_square_sums {n : ℕ} (hn : 0 < n) :
  ∃ (A : array (n × n) ℕ),
    (∀ i : Fin n, is_square (A.sum_row i)) ∧
    (∀ j : Fin n, is_square (A.sum_column j)) ∧
    (∀ i1 i2 : Fin n, i1 ≠ i2 → A.sum_row i1 ≠ A.sum_row i2) ∧
    (∀ j1 j2 : Fin n, j1 ≠ j2 → A.sum_column j1 ≠ A.sum_column j2) :=
sorry

end distinct_perfect_square_sums_l678_678664


namespace find_x_parallel_vectors_l678_678198

theorem find_x_parallel_vectors (x y : ℝ) (h : (2, 1) ∥ (4, y)) : y = x + 1 → x = 1 := 
by 
  unfold ∥ at h 
  sorry

end find_x_parallel_vectors_l678_678198


namespace u_plus_i_eq_neg8_plus_i_l678_678676

/-- Given three complex numbers represented by real and imaginary parts, 
    and certain conditions on the imaginary parts and the real parts, 
    this theorem states the value of the imaginary part of the third complex number. -/
theorem u_plus_i_eq_neg8_plus_i {p r u : ℝ} (q : ℝ) (s : ℝ) :
  q = 5 →
  s = 2 * q →
  ∃ t : ℝ, t = -p - r →
  (p + q * complex.i) + (r + s * complex.i) + (t + u * complex.i) = 7 * complex.i →
  u + complex.i = -8 + complex.i :=
by
  intros hq hs ht hsum
  sorry

end u_plus_i_eq_neg8_plus_i_l678_678676


namespace centroid_dot_product_l678_678199

theorem centroid_dot_product
  (A B C G : EuclideanGeometry.Point)
  (h1 : EuclideanGeometry.dist A B = 1)
  (h2 : EuclideanGeometry.dist B C = Real.sqrt 2)
  (h3 : EuclideanGeometry.dist A C = Real.sqrt 3)
  (h4 : EuclideanGeometry.centroid A B C G) :
  (EuclideanGeometry.vec A G) • (EuclideanGeometry.vec A C) = 4 / 3 :=
sorry

end centroid_dot_product_l678_678199


namespace apples_in_first_group_l678_678282

variable (A O : ℝ) (X : ℕ)

-- Given conditions
axiom h1 : A = 0.21
axiom h2 : X * A + 3 * O = 1.77
axiom h3 : 2 * A + 5 * O = 1.27 

-- Goal: Prove that the number of apples in the first group is 6
theorem apples_in_first_group : X = 6 := 
by 
  sorry

end apples_in_first_group_l678_678282


namespace power_mod_l678_678727

theorem power_mod (h : 5 ^ 200 ≡ 1 [MOD 1000]) : 5 ^ 6000 ≡ 1 [MOD 1000] :=
by
  sorry

end power_mod_l678_678727


namespace calculate_paving_cost_l678_678351

theorem calculate_paving_cost
  (length : ℝ) (width : ℝ) (rate_per_sq_meter : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_rate : rate_per_sq_meter = 1200) :
  (length * width * rate_per_sq_meter = 24750) :=
by
  sorry

end calculate_paving_cost_l678_678351


namespace decagon_ratio_bisect_l678_678426

theorem decagon_ratio_bisect (area_decagon unit_square area_trapezoid : ℕ) 
  (h_area_decagon : area_decagon = 12) 
  (h_bisect : ∃ RS : ℕ, ∃ XR : ℕ, RS * 2 = area_decagon) 
  (below_RS : ∃ base1 base2 height : ℕ, base1 = 3 ∧ base2 = 3 ∧ base1 * height + 1 = 6) 
  : ∃ XR RS : ℕ, RS ≠ 0 ∧ XR / RS = 1 := 
sorry

end decagon_ratio_bisect_l678_678426


namespace determine_range_of_a_l678_678436

noncomputable def range_of_a : Set ℝ := {a | ∃ x : ℝ, (-x^2 + (2 + a) * x + 2 = 0) ∧ (x ∈ Ioo (-1) 3)}

theorem determine_range_of_a : range_of_a = Set.Icc (-1) (1 / 3) :=
by
  sorry

end determine_range_of_a_l678_678436


namespace no_such_n_l678_678871

def greatest_prime_factor (n : ℕ) : ℕ :=
sorry -- Definition of the greatest prime factor

theorem no_such_n (P : ℕ → ℕ) :
  (∀ n > 1, P n = greatest_prime_factor n) →
  ¬(∃ n > 1, P n = n ^ (1 / 2) ∧ P (n + 63) = (n + 63) ^ (1 / 2)) :=
by
  intros hP hn
  simp only [ge, gt_iff_lt, exists_prop] at hn
  rcases hn with ⟨n, hn1, hn2, hn3⟩
  sorry

end no_such_n_l678_678871


namespace log_product_eq_one_l678_678153

open Real

noncomputable def log_product (x y : ℝ) (hx : 1 < x) (hy : 1 < y) : ℝ :=
  log x y * log y x

theorem log_product_eq_one (x y : ℝ) (hx : 1 < x) (hy : 1 < y) :
  log_product x y hx hy = 1 :=
by
  sorry

end log_product_eq_one_l678_678153


namespace graph_intersections_count_l678_678756

noncomputable def parametric_x (t : ℝ) : ℝ := Real.cos t + t / 3
def parametric_y (t : ℝ) : ℝ := Real.sin t
def period : ℝ := 2 * Real.pi

theorem graph_intersections_count :
  (finset.card (finset.filter (λ t : ℝ, 0 ≤ parametric_x t ∧ parametric_x t ≤ 30 ∧ parametric_y t = parametric_y (t + period))
  (finset.range 10))) = 10 := 
sorry

end graph_intersections_count_l678_678756


namespace part1_part2_l678_678012

-- Definitions and conditions
def total_length : ℝ := 64
def ratio_larger_square_area : ℝ := 2.25
def total_area : ℝ := 160

-- Given problem parts
theorem part1 (x : ℝ) (h : (64 - 4 * x) / 4 * (64 - 4 * x) / 4 = 2.25 * x * x) : x = 6.4 :=
by
  -- Proof needs to be provided
  sorry

theorem part2 (y : ℝ) (h : (16 - y) * (16 - y) + y * y = 160) : y = 4 ∧ (64 - 4 * y) = 48 :=
by
  -- Proof needs to be provided
  sorry

end part1_part2_l678_678012


namespace smallest_positive_angle_l678_678842

theorem smallest_positive_angle (x : ℝ) (hx : 0 < x) :
  tan (3 * x + 9) = cot (x - 9) → x = 22.5 :=
by
  intro h
  sorry

end smallest_positive_angle_l678_678842


namespace prove_clothing_colors_l678_678814

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l678_678814


namespace find_Finley_age_l678_678660

variable (Roger Jill Finley : ℕ)
variable (Jill_age : Jill = 20)
variable (Roger_age : Roger = 2 * Jill + 5)
variable (Finley_condition : 15 + (Roger - Jill) = Finley - 30)

theorem find_Finley_age : Finley = 55 :=
by
  sorry

end find_Finley_age_l678_678660


namespace polygon_intersections_l678_678657

theorem polygon_intersections :
  let n_polygons := [6, 7, 8, 9] in
  let pairs := [(6,7), (6,8), (6,9), (7,8), (7,9), (8,9)] in
  let intersection_points (p q : Nat) : Nat := 2 * min p q in
  (pairs.map (λ pq, intersection_points pq.1 pq.2)).sum = 80 :=
by
  -- The problem statement and initial setup

  let n_polygons := [6, 7, 8, 9]
  let pairs := [(6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
  let intersection_points (p q : Nat) : Nat := 2 * min p q

  -- Proof setup would go here, sorry is used to placeholder for the proof.
  sorry

end polygon_intersections_l678_678657


namespace part1_a_part1_b_estimated_probability_high_quality_dolls_10000_l678_678040

def data : List (Nat × Nat) := [(10, 9), (100, 96), (1000, 951), (2000, 1900), (3000, 2856), (5000, 4750)]

theorem part1_a (h : data_look 1000 data = 951) : 951 / 1000 = 0.951 :=
by
  sorry

theorem part1_b (h : data_look 5000 data = 4750) : 4750 / 5000 = 0.95 :=
by
  sorry

theorem estimated_probability : 
  let frequencies := data.map (fun (n, m) => (m / n : ℝ));
  abs ((frequencies.sum / frequencies.length : ℝ) - 0.95) < 0.01 :=
by
  sorry

theorem high_quality_dolls_10000 : 
  abs (10000 * 0.95 - 9500) < 1 :=
by
  sorry

end part1_a_part1_b_estimated_probability_high_quality_dolls_10000_l678_678040


namespace carpet_needed_l678_678760

-- Define the dimensions of the room and the column
structure Room where
  length : ℝ  -- in feet
  width : ℝ   -- in feet

def room_area (r : Room) : ℝ := r.length * r.width

structure Column where
  side_length : ℝ  -- in feet

def column_area (c : Column) : ℝ := c.side_length * c.side_length

def net_area_to_be_carpeted (r : Room) (c : Column) : ℝ := room_area r - column_area c

def square_feet_to_square_yards (sq_ft : ℝ) : ℝ := sq_ft / 9

def rounded_square_yards (sq_yds : ℝ) : ℕ := Int.ceil sq_yds

-- Main theorem: Given the dimensions of the room and the column, determine the required carpet area in square yards
theorem carpet_needed (r : Room) (c : Column) 
  (room_length : r.length = 15) (room_width : r.width = 8) 
  (column_side_length : c.side_length = 2) : 
  rounded_square_yards (square_feet_to_square_yards (net_area_to_be_carpeted r c)) = 13 :=
by
  sorry

end carpet_needed_l678_678760


namespace sin_A_value_correct_l678_678543

noncomputable def sin_value (A : ℝ) (h : A ∈ Ioo (Real.pi / 4) Real.pi) (h2 : Real.sin (A + Real.pi / 4) = 7 * Real.sqrt 2 / 10) : Prop :=
  Real.sin A = 4 / 5

theorem sin_A_value_correct (A : ℝ) (h : A ∈ Ioo (Real.pi / 4) Real.pi) (h2 : Real.sin (A + Real.pi / 4) = 7 * Real.sqrt 2 / 10) : sin_value A h h2 :=
by {
  sorry
}

end sin_A_value_correct_l678_678543


namespace negation_of_proposition_l678_678336

theorem negation_of_proposition :
  (¬ (∃ x : ℝ, x < 0 ∧ x^2 - 2 * x > 0)) ↔ (∀ x : ℝ, x < 0 → x^2 - 2 * x ≤ 0) :=
by sorry

end negation_of_proposition_l678_678336


namespace sum_of_sequence_a_eq_1_sum_of_sequence_a_ne_1_l678_678419

theorem sum_of_sequence_a_eq_1 (n : ℕ) (h: n > 0) : 
  let S_n := (∑ i in Finset.range n, (1 / (1 : ℝ)^i) + (3 * (i+1) - 2)) 
  in S_n = (n * (3 * n + 1)) / 2 := sorry

theorem sum_of_sequence_a_ne_1 (n : ℕ) (a : ℝ) (h_n : n > 0) (h_a : a ≠ 0): 
  let S_n := (∑ i in Finset.range n, (1 / a^i) + (3 * (i+1) - 2)) 
  in S_n = ((a - a^(1 - n)) / (a - 1)) + (n * (3 * n - 1) / 2) := sorry

end sum_of_sequence_a_eq_1_sum_of_sequence_a_ne_1_l678_678419


namespace residue_of_11_pow_2048_mod_19_l678_678716

theorem residue_of_11_pow_2048_mod_19 :
  (11 ^ 2048) % 19 = 16 := 
by
  sorry

end residue_of_11_pow_2048_mod_19_l678_678716


namespace identify_clothes_l678_678802

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l678_678802


namespace area_of_square_l678_678311

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem area_of_square : 
  let p1 := (1 : ℝ, -2 : ℝ)
  let p2 := (-3 : ℝ, 5 : ℝ)
  let side_length := distance p1 p2
  side_length = Real.sqrt 65 → 
  (side_length)^2 = 65 :=
by
  intros p1 p2 side_length h1
  simp [distance, p1, p2] at h1
  calc (Real.sqrt ((-3 - 1)^2 + (5 - -2)^2))
       = Real.sqrt ((-4)^2 + 7^2) : by simp
  ... = Real.sqrt (16 + 49) : by simp
  ... = Real.sqrt 65 : by simp
  sorry

end area_of_square_l678_678311


namespace trigonometric_identity_l678_678499

theorem trigonometric_identity (x y r : ℝ) (hx : x = -4/5) (hy : y = 3/5) (hr : r = 1)
  (hcos : cos (atan2 y x) = x / r) (hsin : sin (atan2 y x) = y / r) :
  2 * sin (atan2 y x) + cos (atan2 y x) = 2 / 5 := 
by 
  have α := atan2 y x
  have hcos' : cos α = x / r := hcos
  have hsin' : sin α = y / r := hsin
  have h_cos_val : cos α = -4/5 := by 
    rw [hx, hr, div_one]; exact hcos'
  have h_sin_val : sin α = 3/5 := by
    rw [hy, hr, div_one]; exact hsin'
  rw [h_cos_val, h_sin_val]
  calc
    2 * 3/5 + -4/5 = 6/5 - 4/5 : by ring
              ...  = 2/5 : by ring

end trigonometric_identity_l678_678499


namespace tax_calculation_correct_l678_678048

def calculate_tax (s e b1 b2 r1 r2 r3 : ℝ) : ℝ :=
  let taxable_income := s - e
  if taxable_income ≤ b1 then
    taxable_income * r1
  else if taxable_income ≤ b2 then
    (b1 * r1) + (taxable_income - b1) * r2
  else
    (b1 * r1) + (b2 - b1) * r2 + (taxable_income - b2) * r3

theorem tax_calculation_correct :
  calculate_tax 20000 5000 3000 12000 0.03 0.10 0.20 = 1590 :=
by repeat { sorry }

end tax_calculation_correct_l678_678048


namespace maximum_value_of_a_l678_678246

theorem maximum_value_of_a (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) : a ≤ 2924 := 
sorry

end maximum_value_of_a_l678_678246


namespace find_f_2017_l678_678474

noncomputable def f : ℝ → ℝ := λ x,
  if x >= 0 then f (x - 5)
  else log 3 (-x)

theorem find_f_2017 : f 2017 = 1 := sorry

end find_f_2017_l678_678474


namespace probability_closer_to_1_1_than_4_1_l678_678034

open Real

def rectangle := { p : ℝ × ℝ // 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 }

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem probability_closer_to_1_1_than_4_1 :
  let closer_region := { p : rectangle // distance (p.val) (1,1) < distance (p.val) (4,1) } in
  (set.univ.val.sum (λ x, if x ∈ closer_region then 1 else 0)) / (set.univ.val.sum (λ _, 1)) = 5 / 6 :=
sorry

end probability_closer_to_1_1_than_4_1_l678_678034


namespace mutant_frog_percentage_proof_l678_678117

/-- Number of frogs with extra legs -/
def frogs_with_extra_legs := 5

/-- Number of frogs with 2 heads -/
def frogs_with_two_heads := 2

/-- Number of frogs that are bright red -/
def frogs_bright_red := 2

/-- Number of normal frogs -/
def normal_frogs := 18

/-- Total number of mutant frogs -/
def total_mutant_frogs := frogs_with_extra_legs + frogs_with_two_heads + frogs_bright_red

/-- Total number of frogs -/
def total_frogs := total_mutant_frogs + normal_frogs

/-- Calculate the percentage of mutant frogs rounded to the nearest integer -/
def mutant_frog_percentage : ℕ := (total_mutant_frogs * 100 / total_frogs).toNat

theorem mutant_frog_percentage_proof:
  mutant_frog_percentage = 33 := 
  by 
    -- Proof skipped
    sorry

end mutant_frog_percentage_proof_l678_678117


namespace kwik_e_tax_revenue_l678_678289

def price_federal : ℕ := 50
def price_state : ℕ := 30
def price_quarterly : ℕ := 80

def num_federal : ℕ := 60
def num_state : ℕ := 20
def num_quarterly : ℕ := 10

def revenue_federal := num_federal * price_federal
def revenue_state := num_state * price_state
def revenue_quarterly := num_quarterly * price_quarterly

def total_revenue := revenue_federal + revenue_state + revenue_quarterly

theorem kwik_e_tax_revenue : total_revenue = 4400 := by
  sorry

end kwik_e_tax_revenue_l678_678289


namespace number_of_subsets_intersecting_B_l678_678877

theorem number_of_subsets_intersecting_B (m n : ℕ) (h : m > n) :
  let A := finset.range(m + 1).filter (λ x, x > 0),
      B := finset.range(n + 1).filter (λ x, x > 0) in
  (2 : ℤ) ^ m - (2 : ℤ) ^ (m - n) = finset.card {C | ∃ x, C ∈ A ∧ C ∩ B ≠ ∅} :=
sorry

end number_of_subsets_intersecting_B_l678_678877


namespace xy_value_is_one_l678_678134

open Complex

theorem xy_value_is_one (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : x * y = 1 :=
by
  sorry

end xy_value_is_one_l678_678134


namespace regression_analysis_l678_678469

variable (n : Nat)
variable (x y : Fin n → ℝ)
variable (r : ℝ)
variable (R_squared : ℝ)

def is_C_incorrect (R_squared : ℝ) : Prop :=
  ∃ (x̄ ȳ : ℝ), x̄ = (∑ i in finRange n, x i) / n ∧ ȳ = (∑ i in finRange n, y i) / n ∧ 
  R_squared < 1 → R_squared ≥ 0

theorem regression_analysis (h_r : abs r = 0.9462) (h_R_squared : ¬is_C_incorrect R_squared):
  h_R_squared → R_squared ≥ 0 ∧ R_squared < 1 :=
by sorry

end regression_analysis_l678_678469


namespace kendra_total_earnings_l678_678232

theorem kendra_total_earnings (laurel2014 kendra2014 kendra2015 : ℕ) 
  (h1 : laurel2014 = 30000)
  (h2 : kendra2014 = laurel2014 - 8000)
  (h3 : kendra2015 = 1.20 * laurel2014) :
  kendra2014 + kendra2015 = 58000 :=
by
  sorry

end kendra_total_earnings_l678_678232


namespace num_two_digit_powers_of_3_l678_678937

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678937


namespace complete_square_result_l678_678332

theorem complete_square_result (x : ℝ) :
  (x^2 - 4 * x - 3 = 0) → ((x - 2) ^ 2 = 7) :=
by sorry

end complete_square_result_l678_678332


namespace side_length_of_square_l678_678441

theorem side_length_of_square (area : ℕ) (h : area = 49) : ∃ (s : ℕ), s * s = area ∧ s = 7 :=
by {
  use 7,
  split,
  { rw h,
    norm_num },
  { reflexivity }
}

end side_length_of_square_l678_678441


namespace geometric_sequence_a10_l678_678460

-- Lean statement describing the conditions and proof for a geometric sequence
theorem geometric_sequence_a10 (a : ℕ → ℤ) (r : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_sum : ∀ n, (finset.range (2*n)).sum a = 3 * (finset.range n).sum (λ i, a (2*i + 1)))
  (h_product : a 1 * a 2 * a 3 = 8) :
  a 10 = 512 :=
sorry

end geometric_sequence_a10_l678_678460


namespace line_equation_through_P_l678_678089

theorem line_equation_through_P 
  (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hP : P = (3, -1))
  (hA : A = (2, -3))
  (hB : B = (4, 5)) :
  (4 * (P.1 : ℝ) - (P.2 : ℝ) - 13 = 0 ∨ P.1 = 3) ↔ 
  (4 * (hP.1 : ℝ) - hP.2 - 13 = 0 ∨ hP.1 = 3) :=
  sorry

end line_equation_through_P_l678_678089


namespace percentage_loss_is_15_l678_678767

variable (SP1 SP2 : ℝ) -- SP1 is the selling price at 20% gain, SP2 is the selling price when loss occurs.
variable (CP : ℝ) -- CP is the cost price.
variable (percentage_loss : ℝ)

-- Given conditions
variable h1 : SP1 = 192
variable h2 : SP2 = 136
variable h3 : CP = SP1 / 1.2

-- To prove
theorem percentage_loss_is_15 :
  percentage_loss = ((CP - SP2) / CP) * 100 →
  percentage_loss = 15 := 
by
  sorry

end percentage_loss_is_15_l678_678767


namespace ratio_of_wealth_specified_l678_678847

noncomputable def world_population : ℝ := sorry
noncomputable def world_wealth : ℝ := sorry

def population_X : ℝ := 0.4 * world_population
def wealth_X : ℝ := 0.5 * world_wealth
def wealth_per_citizen_X : ℝ := wealth_X / population_X

def population_Y : ℝ := 0.2 * world_population
def wealth_Y : ℝ := 0.5 * world_wealth

-- Wealthiest 10% of Y
def population_Y_wealthiest_10 : ℝ := 0.1 * population_Y
def wealth_Y_wealthiest_10 : ℝ := 0.9 * wealth_Y

-- Wealth per citizen among wealthiest 10% of Y
def wealth_per_citizen_Y_wealthiest_10 : ℝ := wealth_Y_wealthiest_10 / population_Y_wealthiest_10

-- Remaining population and wealth in Y
def population_Y_remaining : ℝ := 0.9 * population_Y
def wealth_Y_remaining : ℝ := wealth_Y - wealth_Y_wealthiest_10

-- Wealth per citizen among remaining 90% of Y
def wealth_per_citizen_Y_remaining : ℝ := wealth_Y_remaining / population_Y_remaining

-- Average wealth per citizen in Y
def average_wealth_per_citizen_Y : ℝ := 
  (0.1 * wealth_per_citizen_Y_wealthiest_10 + 0.9 * wealth_per_citizen_Y_remaining)

-- Ratio of wealth per citizen X to Y
def ratio_wealth_per_citizen_X_to_Y : ℝ :=
  wealth_per_citizen_X / average_wealth_per_citizen_Y

theorem ratio_of_wealth_specified:
  ratio_wealth_per_citizen_X_to_Y = 10 / 19 :=
sorry

end ratio_of_wealth_specified_l678_678847


namespace find_solutions_l678_678859

def is_solution (m n p : ℕ) : Prop :=
  p.prime ∧ p^n + 144 = m^2

theorem find_solutions :
  (∃ m n p : ℕ, m > 0 ∧ n > 0 ∧ p.prime ∧ (m, n, p) = (20, 8, 2)) ∨
  (∃ m n p : ℕ, m > 0 ∧ n > 0 ∧ p.prime ∧ (m, n, p) = (7, 2, 5)) :=
by
  sorry

end find_solutions_l678_678859


namespace determinant_of_A_l678_678076

-- Define the 2x2 matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![7, -2], ![-3, 6]]

-- The statement to be proved
theorem determinant_of_A : Matrix.det A = 36 := 
  by sorry

end determinant_of_A_l678_678076


namespace monotonic_function_l678_678907

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then a * x^2 + 3 else (a + 2) * Real.exp (a * x)

theorem monotonic_function (a : ℝ) : (∀ x y : ℝ, x ≤ y → f x ≤ f y) ↔ (0 < a ∧ a ≤ 1) :=
by
  sorry

end monotonic_function_l678_678907


namespace triangulation_catalan_l678_678209

noncomputable def catalan (n : ℕ) : ℕ :=
  (Nat.choose (2 * n) n) / (n + 1)

noncomputable def triangulations (n : ℕ) : ℕ :=
  if n < 3 then 1 else 
  (∑ k in Finset.range (n - 2),
    triangulations (k + 2) * triangulations (n - k - 1))

theorem triangulation_catalan (n : ℕ) :
  triangulations n = catalan (n - 2) :=
sorry

end triangulation_catalan_l678_678209


namespace circle_occupies_62_8_percent_l678_678081

noncomputable def largestCirclePercentage (length : ℝ) (width : ℝ) : ℝ :=
  let radius := width / 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := length * width
  (circle_area / rectangle_area) * 100

theorem circle_occupies_62_8_percent : largestCirclePercentage 5 4 = 62.8 := 
by 
  /- Sorry, skipping the proof -/
  sorry

end circle_occupies_62_8_percent_l678_678081


namespace find_x1_l678_678893

noncomputable def sqrt_80_inv : ℝ :=
(15 / real.sqrt 80)

theorem find_x1 (x1 x2 x3 x4 : ℝ)
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 9 / 16) :
  x1 = 1 - sqrt_80_inv :=
sorry

end find_x1_l678_678893


namespace identify_clothes_l678_678799

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l678_678799


namespace standard_equation_of_ellipse_max_area_inscribed_circle_l678_678887

def ellipse_params (a b : ℝ) (h : a > b ∧ b > 0) (h_ecc : sqrt (a^2 - b^2) / a = 1 / 2) : Prop :=
  (∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)

theorem standard_equation_of_ellipse :
  ∀ (a b : ℝ), 
    a > b → 
    b > 0 → 
    sqrt (a^2 - b^2) / a = 1 / 2 → 
    (\exists c : ℝ, a^2 = b^2 + c^2) →
    a = 2 →  
    b = sqrt 3 →  
    c = 1 →  
    ∀ x y, x^2 / 4 + y^2 / 3 = 1 
    :=
by 
  intros a b h1 h2 h3 h4 ha hb hc x y
  sorry

theorem max_area_inscribed_circle (F₁ F₂ : ℝ × ℝ) (M N : ℝ × ℝ) 
  (hM : M ∈ λ ⟨x₁, y₁⟩, x₁^2 / 4 + y₁^2 / 3 = 1) 
  (hN : N ∈ λ ⟨x₂, y₂⟩, x₂^2 / 4 + y₂^2 / 3 = 1) 
  (hF₂ : F₂.1 = 1 ∧ F₂.2 = 0) : ∃ (R : ℝ), R = 3 / 4 ∧ ∃ eq : (ℝ × ℝ) → ℝ, eq F₂ = F₂ :=
by 
  rintros F₁ F₂ ⟨x₁, y₁⟩ ⟨x₂, y₂⟩ hM hN hF₂ 
  sorry

end standard_equation_of_ellipse_max_area_inscribed_circle_l678_678887


namespace max_servings_fruit_punch_l678_678381

/-- 
Given:
1. 3 oranges are required for 8 servings,
2. 2 liters of juice are required for 8 servings,
3. 1 liter of soda is required for 8 servings,
4. Kim has 10 oranges,
5. Kim has 12 liters of juice,
6. Kim has 5 liters of soda,

Show that the greatest number of servings of fruit punch that Kim can prepare is 26.
-/
theorem max_servings_fruit_punch :
  let servings_per_recipe := 8 in
  let oranges_per_recipe := 3 in
  let juice_per_recipe := 2 in
  let soda_per_recipe := 1 in
  let kim_oranges := 10 in
  let kim_juice := 12 in
  let kim_soda := 5 in
  max_servings servings_per_recipe oranges_per_recipe juice_per_recipe soda_per_recipe kim_oranges kim_juice kim_soda = 26 :=
by {
  sorry 
}

end max_servings_fruit_punch_l678_678381


namespace count_two_digit_powers_of_three_l678_678983

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678983


namespace sin_B_value_l678_678202

noncomputable def triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
a + c = 2 * b ∧ A - C = π / 3

theorem sin_B_value {a b c A B C : ℝ}
  (h : triangle a b c A B C) :
  Real.sin B = Real.sqrt 39 / 8 :=
by sorry

end sin_B_value_l678_678202


namespace least_possible_value_lcm_gcd_fraction_l678_678452

theorem least_possible_value_lcm_gcd_fraction :
  ∀ (a b c : ℕ), (0 < a ∧ 0 < b ∧ 0 < c) ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  let l := Nat.lcm in let g := Nat.gcd in 
  (∃ q : ℚ, q = (l a b + l b c + l c a) / (g a b + g b c + g c a) ∧ q = 5/2) :=
by
  sorry

end least_possible_value_lcm_gcd_fraction_l678_678452


namespace division_scaling_l678_678733

theorem division_scaling (h : 204 / 12.75 = 16) : 2.04 / 1.275 = 16 :=
sorry

end division_scaling_l678_678733


namespace symmetric_circle_equation_l678_678888

noncomputable def circle1 : set (ℝ × ℝ) := {p | (p.1 + 1)^2 + (p.2 - 1)^2 = 1}

noncomputable def line : set (ℝ × ℝ) := {p | p.1 - p.2 = 1}

noncomputable def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.2 + 1, p.1 - 1)

theorem symmetric_circle_equation :
  (∀ p ∈ {p : ℝ × ℝ | (symmetric_point p).1 + 1)^2 + (symmetric_point p).2 - 1)^2 = 1},
  (p.1 - 2)^2 + (p.2 + 2)^2 = 1 ∈ circle1 :=
sorry

end symmetric_circle_equation_l678_678888


namespace value_of_a_l678_678554

theorem value_of_a (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := sorry

end value_of_a_l678_678554


namespace colors_of_clothes_l678_678806

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l678_678806


namespace theta_in_second_quadrant_l678_678122

-- Our problem conditions and theorem
theorem theta_in_second_quadrant (θ : ℝ) 
  (h1 : sin(π / 2 + θ) < 0) 
  (h2 : tan(π - θ) > 0) : 
  (∃ k : ℤ, θ = π / 2 + k * π ∨ θ = -π / 2 + k * π) :=
by sorry

end theta_in_second_quadrant_l678_678122


namespace dan_helmet_craters_l678_678429

variable (D S : ℕ)
variable (h1 : D = S + 10)
variable (h2 : D + S + 15 = 75)

theorem dan_helmet_craters : D = 35 := by
  sorry

end dan_helmet_craters_l678_678429


namespace distance_from_point_to_intersection_l678_678576

theorem distance_from_point_to_intersection (d_a d_beta d_gamma : ℝ) (h_a : d_a = 3) (h_beta : d_beta = 4) (h_gamma : d_gamma = 12) :
  (√(d_a^2 + d_beta^2 + d_gamma^2) = 13) :=
by
  sorry

end distance_from_point_to_intersection_l678_678576


namespace two_digit_powers_of_3_count_l678_678988

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678988


namespace cone_lateral_surface_area_l678_678316

open Real

def lateral_surface_area_cone (r l : ℝ) : ℝ :=
  (1 / 2) * (2 * π * r) * l

theorem cone_lateral_surface_area :
  lateral_surface_area_cone 2 4 = 8 * π :=
by
  sorry

end cone_lateral_surface_area_l678_678316


namespace base_r_representation_26_eq_32_l678_678079

theorem base_r_representation_26_eq_32 (r : ℕ) : 
  26 = 3 * r + 6 → r = 8 :=
by
  sorry

end base_r_representation_26_eq_32_l678_678079


namespace disagree_parents_count_l678_678391

def total_parents : ℕ := 800  -- Total number of parents surveyed
def agree_percentage : ℝ := 20  -- Percentage of parents who agree to the tuition fee increase

def disagree_parents (P : ℕ) (A : ℝ) : ℝ := (1 - A / 100) * P

theorem disagree_parents_count :
  disagree_parents total_parents agree_percentage = 640 := by
  sorry

end disagree_parents_count_l678_678391


namespace tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l678_678120

open Real

theorem tan_alpha_minus_pi_over_4_eq_neg_3_over_4 (α β : ℝ) 
  (h1 : tan (α + β) = 1 / 2) 
  (h2 : tan β = 1 / 3) : 
  tan (α - π / 4) = -3 / 4 :=
sorry

end tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l678_678120


namespace product_of_functions_l678_678915

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x + 3)
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem product_of_functions (x : ℝ) (hx : x ≠ -3) : f x * g x = x - 3 := by
  -- proof goes here
  sorry

end product_of_functions_l678_678915


namespace no_such_strictly_monotonic_functions_l678_678850

open Function

theorem no_such_strictly_monotonic_functions :
  ¬ (∃ (f g : ℕ → ℕ), StrictMono f ∧ StrictMono g ∧ ∀ n : ℕ, f(g(g(n))) < g(f(n))) :=
by
  sorry

end no_such_strictly_monotonic_functions_l678_678850


namespace correct_outfits_l678_678789

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l678_678789


namespace quadrilateral_area_correct_l678_678731

-- Given definitions
def d : ℝ := 10
def h1 : ℝ := 7
def h2 : ℝ := 3

-- Definition of area according to the given formula
def quadrilateral_area (d h1 h2 : ℝ) : ℝ := (1/2) * d * (h1 + h2)

-- Statement to prove
theorem quadrilateral_area_correct : quadrilateral_area d h1 h2 = 50 := by
  sorry

end quadrilateral_area_correct_l678_678731


namespace min_value_of_xy_l678_678141

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 4 * x * y - x - 2 * y = 4) : 
  xy >= 2 :=
sorry

end min_value_of_xy_l678_678141


namespace initial_integer_value_l678_678402

theorem initial_integer_value (x : ℤ) (h : (x + 2) * (x + 2) = x * x - 2016) : x = -505 := 
sorry

end initial_integer_value_l678_678402


namespace cross_product_u_v_l678_678835

-- Define the vectors u and v
def u : ℝ × ℝ × ℝ := (3, -4, 7)
def v : ℝ × ℝ × ℝ := (2, 5, -3)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

-- State the theorem to be proved
theorem cross_product_u_v : cross_product u v = (-23, 23, 23) :=
  sorry

end cross_product_u_v_l678_678835


namespace hyperbola_h_k_a_b_sum_l678_678208

noncomputable def h : ℝ := 1
noncomputable def k : ℝ := -3
noncomputable def a : ℝ := 3
noncomputable def c : ℝ := 3 * Real.sqrt 5
noncomputable def b : ℝ := 6

theorem hyperbola_h_k_a_b_sum :
  h + k + a + b = 7 :=
by
  sorry

end hyperbola_h_k_a_b_sum_l678_678208


namespace count_negative_numbers_l678_678399

def number_set : Set ℚ := {-2, 1/2, 0, 3}

def is_negative (x : ℚ) : Prop := x < 0

theorem count_negative_numbers :
  (number_set.to_finset.filter is_negative).card = 1 := 
sorry

end count_negative_numbers_l678_678399


namespace part1_part2_l678_678581

-- Part (1)
def initial_capital := 500.0
def growth_rate := 1.4
def operating_cost_yearly := 100.0

def capital_at_end_of_3rd_year (a0 : ℝ) (r : ℝ) (oc : ℝ) : ℝ :=
  let a1 := a0 * r - oc
  let a2 := a1 * r - oc
  a2 * r - oc

theorem part1 : capital_at_end_of_3rd_year initial_capital growth_rate operating_cost_yearly = 936 :=
  by
  sorry

-- Part (2)
def target_capital := 3000.0
def geometric_sum (r : ℝ) (n : ℕ) : ℝ :=
  (1 - r^n) / (1 - r)

def maximum_operating_cost (a0 : ℝ) (r : ℝ) (target : ℝ) (years : ℕ) : ℝ :=
  (a0 * r^years - target) / geometric_sum r years

theorem part2 : maximum_operating_cost initial_capital growth_rate target_capital 6 ≤ 46.8 :=
  by
  sorry

end part1_part2_l678_678581


namespace handshake_bound_l678_678700

theorem handshake_bound (V : Type) [Fintype V] (G : SimpleGraph V) :
  Fintype.card V = 200 →
  (∀ u v : V, G.Adj u v ∨ G.degree' u + G.degree' v ≥ 200) →
  G.edgeFinset.card ≥ 10000 :=
by
  intros h_card h_condition
  sorry

end handshake_bound_l678_678700


namespace colors_of_clothes_l678_678809

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l678_678809


namespace three_powers_in_two_digit_range_l678_678998

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l678_678998


namespace rectangular_and_line_equations_length_and_product_l678_678590

variables (θ t : ℝ)

-- Define the polar curve equation
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ * sin θ ^ 2 = 4 * cos θ

-- Define the rectangular equation of the curve C
def rectangular_curve (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

-- Parametric equations of the line l
def line_parametric (t : ℝ) : (ℝ × ℝ) :=
  (2 + t, -1 - t)

-- General equation of the line l
def general_line (x y : ℝ) : Prop :=
  x + y = 1

-- Prove that the rectangular equation of the curve C and the general equation of line l hold
theorem rectangular_and_line_equations (x y : ℝ) :
  (∃ ρ θ, polar_curve ρ θ ∧ rectangular_curve x y) ∧ 
  (∃ t, line_parametric t = (x, y) ∧ general_line x y ∧ y ^ 2 = 4 * x) :=
sorry

-- Prove the length of the line segment and the product of distances
theorem length_and_product (x y : ℝ) :
  (∃ t1 t2, t1 + t2 = 2 * real.sqrt 2 ∧ t1 * t2 = -14 ∧
    real.sqrt ((t1 + t2) ^ 2 - 4 * t1 * t2) = 8 ∧ 
    abs (t1 * t2) = 14) :=
sorry

end rectangular_and_line_equations_length_and_product_l678_678590


namespace percentage_mutant_frogs_is_33_l678_678113

def num_extra_legs_frogs := 5
def num_two_heads_frogs := 2
def num_bright_red_frogs := 2
def num_normal_frogs := 18

def total_mutant_frogs := num_extra_legs_frogs + num_two_heads_frogs + num_bright_red_frogs
def total_frogs := total_mutant_frogs + num_normal_frogs

theorem percentage_mutant_frogs_is_33 :
  Float.round (100 * total_mutant_frogs.toFloat / total_frogs.toFloat) = 33 :=
by 
  -- placeholder for the proof
  sorry

end percentage_mutant_frogs_is_33_l678_678113


namespace boarding_probability_correct_l678_678503

def interval_train_arrival : ℕ := 10
def time_at_station : ℕ := 1
def probability_boarding_immediately := time_at_station.toRat / interval_train_arrival.toRat

theorem boarding_probability_correct :
  probability_boarding_immediately = (1 : ℚ) / 10 :=
by
  sorry

end boarding_probability_correct_l678_678503


namespace num_integer_pairs_ineq_l678_678867

/-- 
  Define the problem conditions and prove the number of pairs of integers (x, y) 
  that satisfy the inequalities equals to the result.
--/
theorem num_integer_pairs_ineq (x y : ℤ) : 
  (∃ x y : ℤ, y > 2^x + 3 * 2^65 ∧ y ≤ 70 + (2^64 - 1) * x) ↔ (61 * 2^69 + 2144) := 
by
  -- The problem conditions translated into Lean:
  have h₁ : y > 2^x + 3 * 2^65 := sorry,
  have h₂ : y ≤ 70 + (2^64 - 1) * x := sorry,
  -- The proof of the resulting count of pairs is required to fill in here
  sorry

end num_integer_pairs_ineq_l678_678867


namespace area_DEF_leq_EF2_over_4AD2_l678_678047

-- Definitions of given conditions
variables {A B C D E F P : Type*}

variable (area_triangle_ABC : ℝ)
variable (is_triangle_ABC : A B C)
variable (points_on_sides : D ∈ segment B C ∧ E ∈ segment C A ∧ F ∈ segment A B)
variable (cyclic_AFDE : cyclic {A, F, D, E})

-- Goal statement
theorem area_DEF_leq_EF2_over_4AD2
  (h1 : area_triangle_ABC = 1)
  (h2 : is_triangle_ABC)
  (h3 : points_on_sides)
  (h4 : cyclic_AFDE) :
  ∃ (area_DEF : ℝ), 
    area_DEF ≤ EF^2 / (4 * (AD : ℝ)^2) := 
sorry

end area_DEF_leq_EF2_over_4AD2_l678_678047


namespace sin_2A_div_sin_C_eq_one_l678_678201

theorem sin_2A_div_sin_C_eq_one
  (a b c : ℝ)
  (h1 : a = 4)
  (h2 : b = 5)
  (h3 : c = 6)
  : ∀ (A C : ℝ), (∀ (cos_C : ℝ), cos_C = (a^2 + b^2 - c^2) / (2 * a * b)) →
             (∀ (sin_C : ℝ), sin_C = sqrt(1 - cos_C ^ 2)) →
             (∀ (cos_A : ℝ), cos_A = (b^2 + c^2 - a^2) / (2 * b * c)) →
             (∀ (sin_A : ℝ), sin_A = sqrt(1 - cos_A ^ 2)) →
             ∃ (two_times_sin_A_cos_A : ℝ), two_times_sin_A_cos_A = 2 * sin_A * cos_A →
             (sin_2A_div_sin_C : ℝ), sin_2A_div_sin_C = two_times_sin_A_cos_A / sin_C →
             sin_2A_div_sin_C = 1 :=
by {
  sorry
}

end sin_2A_div_sin_C_eq_one_l678_678201


namespace eval_at_neg_five_l678_678517

def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem eval_at_neg_five : f (-5) = 12 :=
by
  sorry

end eval_at_neg_five_l678_678517


namespace function_defined_all_real_l678_678083

theorem function_defined_all_real (λ : ℝ) :
  λ ∈ Ioo (-2 * sqrt 3) (2 * sqrt 3) → 
  ∀ x : ℝ, x^2 - λ * x + 3 ≠ 0 :=
by 
  assume h
  sorry

end function_defined_all_real_l678_678083


namespace expression_evaluation_polynomial_deduction_l678_678740
-- Import all the necessary libraries

-- Define the expressions and the statements to prove

-- Problem 1: Expression Evaluation
theorem expression_evaluation : 
  (2 + 7 / 9) ^ (1 / 2) - (2 * Real.sqrt 3 - Real.pi) ^ 0 -
  (2 + 10 / 27) ^ (-2 / 3) + 0.25 ^ (-3 / 2) = 389 / 48 :=
  sorry

-- Problem 2: Given x + x^(-1) = 4 and 0 < x < 1, deducing the polynomial in terms of t
theorem polynomial_deduction (x : ℝ) (hx1 : x + x⁻¹ = 4) (hx2 : 0 < x ∧ x < 1) :
  let t := x ^ (1 / 2) in t ^ 6 - 14 * t ^ 2 + 1 = 0 :=
  sorry

end expression_evaluation_polynomial_deduction_l678_678740


namespace correct_outfits_l678_678794

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l678_678794


namespace determine_clothes_l678_678819

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l678_678819


namespace solve_system_eqns_l678_678672

theorem solve_system_eqns (x y z a : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = a^2)
  (h3 : x^3 + y^3 + z^3 = a^3) :
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = a ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = a) := 
by
  sorry

end solve_system_eqns_l678_678672


namespace answer1_answer2_answer3_l678_678041

-- Define the sample data
def sample_data := [(10, 9, 0.9), (100, 96, 0.96), (1000, 951, _), (2000, 1900, 0.95), (3000, 2856, 0.952), (5000, 4750, _)]

-- Predefined values for a and b
def a : ℚ := 951 / 1000
def b : ℚ := 4750 / 5000

-- The probability of selecting a high-quality doll
def estimated_probability : ℚ := 0.95

-- The number of high-quality dolls in a batch of 10000
def high_quality_dolls_in_10000 : ℕ := (10000 : ℚ) * estimated_probability

theorem answer1 : a = 0.951 ∧ b = 0.95 := by
  sorry

theorem answer2 : estimated_probability = 0.95 := by
  sorry

theorem answer3 : high_quality_dolls_in_10000 = 9500 := by
  sorry

end answer1_answer2_answer3_l678_678041


namespace problem1_problem2_l678_678633

namespace MathProof

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - (m-1)*x + 2*m

theorem problem1 (m : ℝ) :
  (∀ x, 0 < x → f x m > 0) → -2 * Real.sqrt 6 + 5 ≤ m ∧ m ≤ 2 * Real.sqrt 6 + 5 :=
sorry

theorem problem2 (m : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ f x m = 0) → -2 < m ∧ m < 0 :=
sorry

end MathProof

end problem1_problem2_l678_678633


namespace min_value_l678_678473

variable (a b c : ℝ)

theorem min_value (h1 : a > b) (h2 : b > c) (h3 : a - c = 5) : 
  (a - b) ^ 2 + (b - c) ^ 2 = 25 / 2 := 
sorry

end min_value_l678_678473


namespace complex_numbers_roots_l678_678249

noncomputable theory

-- Definitions of p, q, r as complex numbers with given conditions
variables (p q r : ℂ)

-- Statement of the main theorem
theorem complex_numbers_roots (h1 : p + q + r = 2) (h2 : p * q * r = 2) (h3 : p*q + p*r + q*r = 0) :
  (p = 1 ∧ q = -1 + complex.I ∧ r = -1 - complex.I) ∨
  (p = 1 ∧ q = -1 - complex.I ∧ r = -1 + complex.I) ∨
  (p = -1 + complex.I ∧ q = 1 ∧ r = -1 - complex.I) ∨
  (p = -1 + complex.I ∧ q = -1 - complex.I ∧ r = 1) ∨
  (p = -1 - complex.I ∧ q = 1 ∧ r = -1 + complex.I) ∨
  (p = -1 - complex.I ∧ q = -1 + complex.I ∧ r = 1) :=
sorry

end complex_numbers_roots_l678_678249


namespace range_of_a_l678_678162

noncomputable def f (x : ℝ) (hx : 0 < x) : ℝ := 2 * x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) (hx : 0 < x) : ℝ := -x^2 + a * x - 3

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → f x (by assumption) > g x a (by assumption)) → a < 4 :=
by {
  sorry
}

end range_of_a_l678_678162


namespace prob1_prob2_l678_678509

noncomputable def alpha : ℝ := sorry -- Angle α
noncomputable def P : ℝ × ℝ := (-3, real.sqrt 3)
noncomputable def sin_alpha : ℝ := real.sqrt 3 / 2
noncomputable def cos_alpha : ℝ := -1 / 2
noncomputable def tan_alpha : ℝ := -real.sqrt 3 / 3

-- Problem 1: Prove sin 2α - tan α = -√3 / 6
theorem prob1 (hα : real.angle.from_slope (-3) = alpha) :
  real.sin (2 * α) - real.tan α = -real.sqrt 3 / 6 :=
sorry

-- Problem 2: Prove range of the function y over the interval [0, 2π / 3] is [-2, 1]
noncomputable def f (x : ℝ) : ℝ := real.cos (x - α) * real.cos α - real.sin (x - α) * real.sin α
noncomputable def y (x : ℝ) : ℝ := real.sqrt 3 * f (real.pi / 2 - 2 * x) - 2 * (f x) ^ 2

theorem prob2 :
  set.range (λ x, y x) = set.Icc (-2 : ℝ) (1 : ℝ) :=
sorry

end prob1_prob2_l678_678509


namespace sin_cos_sixth_power_l678_678243

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1/2) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 :=
by
  sorry

end sin_cos_sixth_power_l678_678243


namespace sixty_pair_is_five_seven_l678_678927

/-- 
Define the sequence of integer pairs.
-/
def sequence_of_pairs : ℕ → (ℕ × ℕ)
| 0 := (1, 1)
| (n + 1) := 
  let k := (2 * n + 1 : ℕ).sqrt
  if k * k = (2 * n + 1) then 
    (k / 2 + 1, k / 2 + 1)
  else 
    match n + 1 - (k * (k - 1) / 2) with
    | 0 := (k / 2 + 1, k / 2 + 1)
    | m := (m, k - m)

/-- 
Prove that the 60th pair in the sequence of integer pairs is (5, 7). 
-/
theorem sixty_pair_is_five_seven : sequence_of_pairs 59 = (5, 7) := 
  sorry

end sixty_pair_is_five_seven_l678_678927


namespace sample_size_calculation_l678_678764

-- Definitions based on the conditions
def num_classes : ℕ := 40
def num_representatives_per_class : ℕ := 3

-- Theorem statement we aim to prove
theorem sample_size_calculation : num_classes * num_representatives_per_class = 120 :=
by
  sorry

end sample_size_calculation_l678_678764


namespace maximize_intersection_area_l678_678269

variable {Point : Type*}
variable {AB CD : Point → Point}
variable (A B C D K M : Point)

def parallelogram (a b c d : Point) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  ∃ p q r s, a = p ∧ b = q ∧ c = r ∧ d = s

noncomputable
def distance (p q : Point) : ℝ := (sorry : ℝ)

theorem maximize_intersection_area
  (h1 : parallelogram A B C D)
  (hK : K ∈ set.Icc A B)
  (hM : M ∈ set.Icc C D) :
  distance A K = distance D M :=
sorry

end maximize_intersection_area_l678_678269


namespace determine_clothes_l678_678821

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l678_678821


namespace value_of_expression_l678_678188

theorem value_of_expression (a b : ℤ) (h : a - 2 * b - 3 = 0) : 9 - 2 * a + 4 * b = 3 := 
by 
  sorry

end value_of_expression_l678_678188


namespace min_max_value_l678_678480

theorem min_max_value
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : 0 ≤ x₁) (h₂ : 0 ≤ x₂) (h₃ : 0 ≤ x₃) (h₄ : 0 ≤ x₄) (h₅ : 0 ≤ x₅)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 1) :
  (min (max (x₁ + x₂) (max (x₂ + x₃) (max (x₃ + x₄) (x₄ + x₅)))) = 1 / 3) :=
sorry

end min_max_value_l678_678480


namespace mutually_exclusive_event_3_l678_678470

def is_odd (n : ℕ) := n % 2 = 1
def is_even (n : ℕ) := n % 2 = 0

def event_1 (a b : ℕ) := 
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

def event_2 (a b : ℕ) := 
is_odd a ∧ is_odd b

def event_3 (a b : ℕ) := 
is_odd a ∧ is_even a ∧ is_odd b ∧ is_even b

def event_4 (a b : ℕ) :=
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

theorem mutually_exclusive_event_3 :
  ∀ a b : ℕ, event_3 a b → ¬ event_1 a b ∧ ¬ event_2 a b ∧ ¬ event_4 a b := by
sorry

end mutually_exclusive_event_3_l678_678470


namespace bahs_equivalent_to_yahs_l678_678896

noncomputable def bah_to_rah (b : ℕ) : ℕ := b * 2
noncomputable def rah_to_yah (r : ℕ) : ℕ := r * 2
noncomputable def yahs_to_rahs (y: ℕ) : ℕ := y / 2
noncomputable def rahs_to_bahs (r: ℕ) : ℕ := r / 2

theorem bahs_equivalent_to_yahs (y : ℕ) :
  10 * (yah_to_rah (rah_to_yah 12)) = 20 * 12 → 
  1200 / (rah_to_yah 12) / (10 / 20) = 300 :=
by
  intro h
  sorry

end bahs_equivalent_to_yahs_l678_678896


namespace identify_clothing_l678_678775

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l678_678775


namespace calculate_expression_l678_678420

theorem calculate_expression : |(-2)| + 2023^0 - sqrt 4 = 1 := by
  sorry

end calculate_expression_l678_678420


namespace ellipse_distance_cd_l678_678844

theorem ellipse_distance_cd :
  ∀ (x y : ℝ), (4 * (x - 3) ^ 2 + 16 * (y + 2) ^ 2 = 64) →
    distance (3 + 4, -2) (3, -2 + 2) = 2 * Real.sqrt 5 :=
by
  intros x y h
  have h₁ : 4 * (x - 3) ^ 2 + 16 * (y + 2) ^ 2 = 64 := h
  have h₂ : 3 + 4 = 7 := by norm_num
  have h₃ : -2 + 2 = 0 := by norm_num
  rw [h₂, h₃]
  calc
    distance (7, 0) (3, 0) = Real.sqrt ((7 - 3) ^ 2 + (0 - 0) ^ 2) : by rw distance_eq
                        ... = Real.sqrt ((4) ^ 2 + 0)           : by norm_num
                        ... = 4                                 : Real.sqrt_sq_4
  sorry

end ellipse_distance_cd_l678_678844


namespace num_two_digit_powers_of_3_l678_678936

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678936


namespace Finley_age_proof_l678_678662

variable (Jill_age : ℕ) (Roger_age : ℕ) (Finley_age : ℕ)

-- Condition 1: Jill is 20 years old now
def Jill_current_age : Jill_age = 20 := rfl

-- Condition 2: Roger's age is 5 more than twice Jill's age
def Roger_age_relation : Roger_age = 2 * Jill_age + 5 := sorry

-- Condition 3: In 15 years, their age difference will be 30 years less than Finley's age
def Finley_age_relation (Jill_future_age Roger_future_age Finley_future_age : ℕ) : 
  Jill_future_age = Jill_age + 15 ∧ 
  Roger_future_age = Roger_age + 15 ∧ 
  Finley_future_age = Finley_age + 15 ∧ 
  (Roger_future_age - Jill_future_age = Finley_future_age - 30) := 
  sorry

-- Theorem: Find Finley's current age
theorem Finley_age_proof : Finley_age = 40 :=
by
  -- Assume all conditions mentioned above
  let Jill_age := 20
  let Roger_age := 2 * Jill_age + 5
  let Jill_future_age := Jill_age + 15
  let Roger_future_age := Roger_age + 15
  let Finley_future_age := Finley_age + 15

  -- Using the relation for Finley’s age in the future
  have h1 : Roger_future_age - Jill_future_age = Finley_future_age - 30 := sorry

  -- Calculate Jill's, Roger's, and Finley's future ages and show the math relation
  have h2 : Jill_future_age = 35 := by simp [Jill_age]
  have h3 : Roger_future_age = 60 := by simp [Jill_age, Roger_age]
  have h4 : Roger_future_age - Jill_future_age = 25 := by simp [h2, h3]

  -- Find Finley's future age
  have h5 : Finley_future_age = 55 := by linarith [h1, h4]

  -- Determine Finley's current age
  have h6 : Finley_age = 40 := by simp [h5]

  exact h6

end Finley_age_proof_l678_678662


namespace minimum_draws_divisible_by_3_or_5_l678_678204

theorem minimum_draws_divisible_by_3_or_5 (n : ℕ) (h : n = 90) :
  ∃ k, k = 49 ∧ ∀ (draws : ℕ), draws < k → ¬ (∃ x, 1 ≤ x ∧ x ≤ n ∧ (x % 3 = 0 ∨ x % 5 = 0)) :=
by {
  sorry
}

end minimum_draws_divisible_by_3_or_5_l678_678204


namespace find_odd_function_l678_678825

def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f x + f (-x) = 0

def f1 : ℝ → ℝ := λ x, x^(1/3)
def f2 : ℝ → ℝ := λ x, sin x + 1
def f3 : ℝ → ℝ := λ x, cos x
def f4 : ℝ → ℝ := λ x, log (x^2 + 1) / log 2

theorem find_odd_function :
  (is_odd f1) ∧ ¬(is_odd f2) ∧ ¬(is_odd f3) ∧ ¬(is_odd f4) :=
by
  sorry

end find_odd_function_l678_678825


namespace javier_average_hits_l678_678226

-- Define the total number of games Javier plays and the first set number of games
def total_games := 30
def first_set_games := 20

-- Define the hit averages for the first set of games and the desired season average
def average_hits_first_set := 2
def desired_season_average := 3

-- Define the total hits Javier needs to achieve the desired average by the end of the season
def total_hits_needed : ℕ := total_games * desired_season_average

-- Define the hits Javier made in the first set of games
def hits_made_first_set : ℕ := first_set_games * average_hits_first_set

-- Define the remaining games and the hits Javier needs to achieve in these games to meet his target
def remaining_games := total_games - first_set_games
def hits_needed_remaining_games : ℕ := total_hits_needed - hits_made_first_set

-- Define the average hits Javier needs in the remaining games to meet his target
def average_needed_remaining_games (remaining_games hits_needed_remaining_games : ℕ) : ℕ :=
  hits_needed_remaining_games / remaining_games

theorem javier_average_hits : 
  average_needed_remaining_games remaining_games hits_needed_remaining_games = 5 := 
by
  -- The proof is omitted.
  sorry

end javier_average_hits_l678_678226


namespace converse_false_inverse_false_contrapositive_true_l678_678189

-- Define the notion of the original implication statement and proving the converse, inverse, and contrapositive
variable {a b c : ℝ}

-- Original statement assumed true.
axiom original_statement_true : (ac^2 > bc^2) → (a > b)

-- The statement converse: If a > b, then ac^2 > bc^2. We want to show this as false.
def converse_statement : Prop := (a > b) → (ac^2 > bc^2)

-- The statement inverse: If ac^2 ≤ bc^2, then a ≤ b. We want to show this as false.
def inverse_statement : Prop := (ac^2 ≤ bc^2) → (a ≤ b)

-- The statement contrapositive, we need to prove it is true: If a ≤ b, then ac^2 ≤ bc^2.
def contrapositive_statement : Prop := (a ≤ b) → (ac^2 ≤ bc^2)

-- Statement types justifying the truth or falsity of each
theorem converse_false (h : a > b) (hc : c = 0) : ¬converse_statement :=
by
  sorry -- proof step to show converse is false

theorem inverse_false (h : ac^2 ≤ bc^2) (hc : c = 0) : ¬inverse_statement :=
by
  sorry -- proof step to show inverse is false

theorem contrapositive_true : contrapositive_statement :=
by
  sorry -- proof step to show contrapositive is true

end converse_false_inverse_false_contrapositive_true_l678_678189


namespace entree_cost_14_l678_678172

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l678_678172


namespace non_receivers_after_2020_candies_l678_678007

noncomputable def count_non_receivers (k n : ℕ) : ℕ := 
sorry

theorem non_receivers_after_2020_candies :
  count_non_receivers 73 2020 = 36 :=
sorry

end non_receivers_after_2020_candies_l678_678007


namespace claire_gerbils_l678_678423

variables (G H : ℕ)

-- Claire's total pets
def total_pets : Prop := G + H = 92

-- One-quarter of the gerbils are male
def male_gerbils (G : ℕ) : ℕ := G / 4

-- One-third of the hamsters are male
def male_hamsters (H : ℕ) : ℕ := H / 3

-- Total males are 25
def total_males : Prop := male_gerbils G + male_hamsters H = 25

theorem claire_gerbils : total_pets G H → total_males G H → G = 68 :=
by
  intro h1 h2
  sorry

end claire_gerbils_l678_678423


namespace correct_outfits_l678_678791

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l678_678791


namespace proper_subsets_count_l678_678256

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }

theorem proper_subsets_count :
  Finset.card (Finset.powerset (Finset.filter (∈ B) (Finset.of_set A)) \ {Finset.empty}) = 3 :=
by
  sorry

end proper_subsets_count_l678_678256


namespace brownies_pieces_count_l678_678635

theorem brownies_pieces_count (pan_length pan_width piece_length piece_width : ℕ) (h1 : pan_length = 24) (h2 : pan_width = 30) (h3 : piece_length = 3) (h4 : piece_width = 4) : 
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := 
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end brownies_pieces_count_l678_678635


namespace max_difference_l678_678550

theorem max_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  x ∈ ℤ → y ∈ ℤ → (y - x ≤ 6) ∧ (y - x = 6) :=
by {
  sorry
}

end max_difference_l678_678550


namespace work_rate_D_time_A_B_D_time_D_l678_678341

def workRate (person : String) : ℚ :=
  if person = "A" then 1/12 else
  if person = "B" then 1/6 else
  if person = "A_D" then 1/4 else
  0

theorem work_rate_D : workRate "A_D" - workRate "A" = 1/6 := by
  sorry

theorem time_A_B_D : (1 / (workRate "A" + workRate "B" + (workRate "A_D" - workRate "A"))) = 2.4 := by
  sorry
  
theorem time_D : (1 / (workRate "A_D" - workRate "A")) = 6 := by
  sorry

end work_rate_D_time_A_B_D_time_D_l678_678341


namespace parametric_eq_line_l678_678438

theorem parametric_eq_line (x y t : ℝ)
  (A : x = 1 + t ∧ y = 3 + t)
  (B : x = 2 + t ∧ y = 5 - 2t)
  (C : x = 1 - t ∧ y = 3 - 2t)
  (D : x = 2 + (2 * (sqrt 5) / 5) * t ∧ y = 5 + (sqrt 5 / 5) * t) :
  (2 * x - y + 1 = 0 ↔ (x = 1 - t ∧ y = 3 - 2t)) :=
sorry

end parametric_eq_line_l678_678438


namespace expected_total_rainfall_10_days_l678_678103

theorem expected_total_rainfall_10_days :
  let P_sun := 0.5
  let P_rain3 := 0.3
  let P_rain6 := 0.2
  let daily_rain := (P_sun * 0) + (P_rain3 * 3) + (P_rain6 * 6)
  daily_rain * 10 = 21 :=
by
  sorry

end expected_total_rainfall_10_days_l678_678103


namespace vector_sum_to_zero_l678_678414

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V] {A B C : V}

theorem vector_sum_to_zero (AB BC CA : V) (hAB : AB = B - A) (hBC : BC = C - B) (hCA : CA = A - C) :
  AB + BC + CA = 0 := by
  sorry

end vector_sum_to_zero_l678_678414


namespace range_of_m_l678_678193

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, mx^2 - 4*x + 1 = 0 ∧ ∀ y : ℝ, mx^2 - 4*x + 1 = 0 → y = x) → m ≤ 4 :=
sorry

end range_of_m_l678_678193


namespace magical_stack_card_count_l678_678683

theorem magical_stack_card_count :
  ∃ n, n = 157 + 78 ∧ 2 * n = 470 :=
by
  let n := 235
  use n
  have h1: n = 157 + 78 := by sorry
  have h2: 2 * n = 470 := by sorry
  exact ⟨h1, h2⟩

end magical_stack_card_count_l678_678683


namespace smallest_positive_d_exists_l678_678437

theorem smallest_positive_d_exists :
  ∃ (d : ℝ), (d ≥ 0) ∧ (∀ x y : ℝ, (x ≥ 0) → (y ≥ 0) → sqrt (x * y) + d * (x - y) ^ 2 ≥ (x + y) / 2) 
  ∧ (∀ ε > 0, ∃ x y : ℝ, (x ≥ 0) → (y ≥ 0) → sqrt (x * y) + ε * (x - y) ^ 2 < (x + y) / 2) := 
begin
  -- Proof to be provided
  sorry
end

end smallest_positive_d_exists_l678_678437


namespace david_spent_difference_l678_678848

-- Define the initial amount, remaining amount, amount spent and the correct answer
def initial_amount : Real := 1800
def remaining_amount : Real := 500
def spent_amount : Real := initial_amount - remaining_amount
def correct_difference : Real := spent_amount - remaining_amount

-- Prove that the difference between the amount spent and the remaining amount is $800
theorem david_spent_difference : correct_difference = 800 := by
  sorry

end david_spent_difference_l678_678848


namespace hannah_age_correct_l678_678931

def sibling_ages : List ℤ :=
  [103, 124, 146, 81, 114, 195, 183]

def average_age (ages : List ℤ) : ℤ :=
  ages.sum / ages.length

noncomputable def hannah_age : ℤ :=
  3.2 * average_age sibling_ages

theorem hannah_age_correct :
  hannah_age = 36 * 12 := by
    sorry

end hannah_age_correct_l678_678931


namespace max_product_of_real_roots_quadratic_eq_l678_678920

theorem max_product_of_real_roots_quadratic_eq : ∀ (k : ℝ), (∃ x y : ℝ, 4 * x ^ 2 - 8 * x + k = 0 ∧ 4 * y ^ 2 - 8 * y + k = 0) 
    → k = 4 :=
sorry

end max_product_of_real_roots_quadratic_eq_l678_678920


namespace min_val_xy_l678_678496

theorem min_val_xy (x y : ℝ) 
  (h : 2 * (Real.cos (x + y - 1))^2 = ((x + 1)^2 + (y - 1)^2 - 2 * x * y) / (x - y + 1)) : 
  xy ≥ (1 / 4) :=
sorry

end min_val_xy_l678_678496


namespace Finley_age_proof_l678_678661

variable (Jill_age : ℕ) (Roger_age : ℕ) (Finley_age : ℕ)

-- Condition 1: Jill is 20 years old now
def Jill_current_age : Jill_age = 20 := rfl

-- Condition 2: Roger's age is 5 more than twice Jill's age
def Roger_age_relation : Roger_age = 2 * Jill_age + 5 := sorry

-- Condition 3: In 15 years, their age difference will be 30 years less than Finley's age
def Finley_age_relation (Jill_future_age Roger_future_age Finley_future_age : ℕ) : 
  Jill_future_age = Jill_age + 15 ∧ 
  Roger_future_age = Roger_age + 15 ∧ 
  Finley_future_age = Finley_age + 15 ∧ 
  (Roger_future_age - Jill_future_age = Finley_future_age - 30) := 
  sorry

-- Theorem: Find Finley's current age
theorem Finley_age_proof : Finley_age = 40 :=
by
  -- Assume all conditions mentioned above
  let Jill_age := 20
  let Roger_age := 2 * Jill_age + 5
  let Jill_future_age := Jill_age + 15
  let Roger_future_age := Roger_age + 15
  let Finley_future_age := Finley_age + 15

  -- Using the relation for Finley’s age in the future
  have h1 : Roger_future_age - Jill_future_age = Finley_future_age - 30 := sorry

  -- Calculate Jill's, Roger's, and Finley's future ages and show the math relation
  have h2 : Jill_future_age = 35 := by simp [Jill_age]
  have h3 : Roger_future_age = 60 := by simp [Jill_age, Roger_age]
  have h4 : Roger_future_age - Jill_future_age = 25 := by simp [h2, h3]

  -- Find Finley's future age
  have h5 : Finley_future_age = 55 := by linarith [h1, h4]

  -- Determine Finley's current age
  have h6 : Finley_age = 40 := by simp [h5]

  exact h6

end Finley_age_proof_l678_678661


namespace inscribed_circle_radius_A_B_D_l678_678696

theorem inscribed_circle_radius_A_B_D (AB CD: ℝ) (angleA acuteAngleD: Prop)
  (M N: Type) (MN: ℝ) (area_trapezoid: ℝ)
  (radius: ℝ) : 
  AB = 2 ∧ CD = 3 ∧ angleA ∧ acuteAngleD ∧ MN = 4 ∧ area_trapezoid = (26 * Real.sqrt 2) / 3 
  → radius = (16 * Real.sqrt 2) / (15 + Real.sqrt 129) :=
by
  intro h
  sorry

end inscribed_circle_radius_A_B_D_l678_678696


namespace colors_of_clothes_l678_678803

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l678_678803


namespace log_eq_two_b_sub_log_l678_678542

variable {m n b : ℝ}

theorem log_eq_two_b_sub_log (h : real.logb 2 m = 2 * b - real.logb 2 (n + 1)) : 
  m = 2^(2 * b) / (n + 1) := by
  sorry

end log_eq_two_b_sub_log_l678_678542


namespace area_of_ABCD_is_correct_l678_678382

def Rectangle := { A B C D : Type }

variables (A B C D : Point)
def is_rectangle (A B C D : Point) : Prop := 
  -- Definition that ensures points form a rectangle
  sorry

def segment_length (P Q : Point) (l : ℝ) : Prop := distance P Q = l

theorem area_of_ABCD_is_correct (A B C D : Point) (DB : Vector) (E F : Point) :
  is_rectangle A B C D ∧
  collinear [D, E, B] ∧ collinear [D, F, B] ∧
  segment_length (D, E) 2 ∧ segment_length (E, F) 1 ∧ segment_length (F, B) 1 ∧
  perpendicular (Line L) (Vector DB) ∧ perpendicular (Line L') (Vector DB) ∧
  passes_through (Line L) A ∧ passes_through (Line L') C →
  area_of_rectangle A B C D ≈ 6.9 :=
sorry

end area_of_ABCD_is_correct_l678_678382


namespace carpenter_wood_split_l678_678016

theorem carpenter_wood_split :
  let original_length : ℚ := 35 / 8
  let first_cut : ℚ := 5 / 3
  let second_cut : ℚ := 9 / 4
  let remaining_length := original_length - first_cut - second_cut
  let part_length := remaining_length / 3
  part_length = 11 / 72 :=
sorry

end carpenter_wood_split_l678_678016


namespace tangent_slope_at_point_l678_678151

theorem tangent_slope_at_point (x : ℝ) (y : ℝ) (h : y = 2 * x^3) (h_point : (x, y) = (1, 2)) :
  let f (x : ℝ) := 2 * x ^ 3 in
  let f' (x : ℝ) := 6 * x ^ 2 in
  f' 1 = 6 :=
by
  sorry

end tangent_slope_at_point_l678_678151


namespace find_point_P_circumcircle_fixed_points_min_length_AB_l678_678889

-- Circle M: x^2 + (y - 4)^2 = 4
def circleM (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4

-- Line l: x - 2y = 0
def lineL (x y : ℝ) : Prop := x - 2 * y = 0

-- P is a moving point on line l; Tangents PA and PB drawn to circle M
-- 1. When PA = 2*sqrt(3), find coordinates of P 
theorem find_point_P (x y : ℝ) (hL : lineL x y) (hPA : ∃ A : ℝ × ℝ, (circleM A.1 A.2) ∧ (sqrt ((x - A.1)^2 + (y - A.2)^2) = 2 * sqrt 3)) :
  (x = 0 ∧ y = 0) ∨ (x = 16/5 ∧ y = 8/5) :=
sorry

-- 2. If the circumcircle of triangle PAM is circle N, does circle N pass through fixed points as P moves?
theorem circumcircle_fixed_points (P : ℝ × ℝ) (A M : ℝ × ℝ)
    (hP : lineL P.1 P.2) (hA : circleM A.1 A.2) (hM : circleM 0 4) :
  (∃ N : ℝ × ℝ, ∀ {x y : ℝ}, (circleN x y)) → 
  ∃ N : ℝ × ℝ, N = (0, 4) ∨ N = (8/5, 4/5) :=
sorry

-- 3. Find the minimum length of segment AB
theorem min_length_AB (x y A B M : ℝ)
    (hL : lineL x y) (hA : circleM A B) (hB : circleM x y) (hM : circleM 0 4) :
  ∃ d : ℝ, d = sqrt 11 :=
sorry

end find_point_P_circumcircle_fixed_points_min_length_AB_l678_678889


namespace numbers_not_coprime_l678_678593

theorem numbers_not_coprime (b : ℕ) (h : b = 2013^2013 + 2) : Int.gcd ((b^3 + 1 : ℤ)) ((b^2 + 2 : ℤ)) ≠ 1 := 
sorry

end numbers_not_coprime_l678_678593


namespace polynomial_non_zero_coefficients_l678_678653

theorem polynomial_non_zero_coefficients (n : ℕ) (P : Polynomial ℝ) (hP : P ≠ 0) :
  (Polynomial.C 1 + Polynomial.X) ^ (n - 1) * P).coeffCount ≥ n :=
by sorry

end polynomial_non_zero_coefficients_l678_678653


namespace quadratic_roots_correct_l678_678087

noncomputable theory

-- Define the quadratic equation parameters
def a : ℝ := 6
def b : ℝ := 5
def q : ℝ := 14.5

-- Given the roots of the quadratic equation
def root1 : ℂ := (-5 + complex.I * real.sqrt 323) / 12
def root2 : ℂ := (-5 - complex.I * real.sqrt 323) / 12

-- Define the quadratic equation in question
def quadratic (x : ℂ) : ℂ := a * x^2 + b * x + q

-- The goal is to show that the quadratic equation has the given roots
theorem quadratic_roots_correct :
  (quadratic root1 = 0 ∧ quadratic root2 = 0) ↔ q = 14.5 :=
by
  sorry

end quadratic_roots_correct_l678_678087


namespace line_intersects_iff_sufficient_l678_678498

noncomputable def sufficient_condition (b : ℝ) : Prop :=
b > 1

noncomputable def condition (b : ℝ) : Prop :=
b > 0

noncomputable def line_intersects_hyperbola (b : ℝ) : Prop :=
b > 2 / 3

theorem line_intersects_iff_sufficient (b : ℝ) (h : condition b) : 
  (sufficient_condition b) → (line_intersects_hyperbola b) ∧ ¬(line_intersects_hyperbola b) → (sufficient_condition b) :=
by {
  sorry
}

end line_intersects_iff_sufficient_l678_678498


namespace remainder_of_concatenated_numbers_l678_678240

def concatenatedNumbers : ℕ :=
  let digits := List.range (50) -- [0, 1, 2, ..., 49]
  digits.foldl (fun acc d => acc * 10 ^ (Nat.digits 10 d).length + d) 0

theorem remainder_of_concatenated_numbers :
  concatenatedNumbers % 50 = 49 :=
by
  sorry

end remainder_of_concatenated_numbers_l678_678240


namespace lemonade_solution_l678_678277

variable (x : ℕ)
variable (cups_sold_total : ℕ)
variable (cups_sold_last_week : ℕ)
variable (cups_sold_this_week : ℕ)

def lemonade_problem := 
  cups_sold_this_week = 13 / 10 * cups_sold_last_week ∧
  cups_sold_total = cups_sold_last_week + cups_sold_this_week ∧
  cups_sold_total = 46

theorem lemonade_solution : cups_sold_last_week = 20 :=
by 
  unfold lemonade_problem
  sorry

end lemonade_solution_l678_678277


namespace fencing_costs_l678_678383

noncomputable def sides_ratio (a b : ℕ) : Prop := 3 * a = 4 * b

noncomputable def field_area (a b : ℕ) (A : ℕ) : Prop := a * b = A

theorem fencing_costs (a b : ℕ) (A : ℕ) 
  (ha : sides_ratio a b)
  (hA : field_area a b 8112)
  (wrought_iron_cost : ℕ := 45)
  (wooden_cost : ℕ := 35)
  (chain_link_cost : ℕ := 25) :
  let x := Int.sqrt (A / 12) in
  let shorter_side := 3 * x in
  let longer_side := 4 * x in
  let perimeter := 2 * (shorter_side + longer_side) in
  (perimeter * wrought_iron_cost = 16380) ∧
  (perimeter * wooden_cost = 12740) ∧
  (perimeter * chain_link_cost = 9100) :=
by sorry

end fencing_costs_l678_678383


namespace factor_x10_minus_1_l678_678096

theorem factor_x10_minus_1 : 
  ∀ x : ℝ, ∃ a b c d : polynomial ℝ, (∏ f in {a, b, c, d}, f) = (polynomial.C x - polynomial.C 1) ^ 10 - 1 :=
by sorry

end factor_x10_minus_1_l678_678096


namespace two_digit_powers_of_3_count_l678_678984

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678984


namespace clothes_color_proof_l678_678782

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l678_678782


namespace number_of_pairs_101_l678_678579

theorem number_of_pairs_101 :
  (∃ n : ℕ, (∀ a b : ℕ, (a > 0) → (b > 0) → (a + b = 101) → (b > a) → (n = 50))) :=
sorry

end number_of_pairs_101_l678_678579


namespace two_digit_powers_of_3_count_l678_678987

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678987


namespace trigonometric_identity_tan_cos_simplification_l678_678077

theorem trigonometric_identity_tan_cos_simplification :
  tan 70 * cos 10 * (1 - sqrt 3 * tan 20) = 1 :=
by sorry

end trigonometric_identity_tan_cos_simplification_l678_678077


namespace compare_abc_l678_678245

noncomputable def a : ℝ := Real.log 1/5 / Real.log 2
noncomputable def b : ℝ := Real.log 1/5 / Real.log 3
noncomputable def c : ℝ := 2^(-0.1)

theorem compare_abc : c > a ∧ a > b := by
  sorry

end compare_abc_l678_678245


namespace book_width_l678_678318

noncomputable def phi_conjugate : ℝ := (Real.sqrt 5 - 1) / 2

theorem book_width {w l : ℝ} (h_ratio : w / l = phi_conjugate) (h_length : l = 14) :
  w = 7 * Real.sqrt 5 - 7 :=
by
  sorry

end book_width_l678_678318


namespace count_two_digit_powers_of_three_l678_678977

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678977


namespace geometry_problem_l678_678220

-- Definition for part (1)
def part1 (a b c : ℝ) (A B C : ℝ) (h1 : a = c * Math.sin A / Math.sin C)
  (h2 : b = c * Math.sin B / Math.sin C) 
  (h3 : cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h4 : cos B = (a^2 + c^2 - b^2) / (2 * a * c)) 
  (h : (a * Math.cos B + b * Math.cos A) / c = (Real.sqrt 3 * (a^2 + b^2 - c^2)) / (2 * a * b * Math.sin C)) : 
  Prop :=
  C = π / 3

-- Definition for part (2)
def part2 (a b c OC : ℝ) (A B C : ℝ) 
  (h1 : A + B + C = π)
  (h2 : C = π / 3)
  (O := ∃ O, O = bisector_point A B C)
  (h3 : Math.dist O A = 3)
  (h4 : Math.dist O B = 5)
  (h5 : AB = 7)
  (h6 : Area(O, A, B) = (15 * Real.sqrt 3) / 4) : 
  Prop :=
  OC = (15 * Real.sqrt 3) / 7

-- Lean statement combining both parts
theorem geometry_problem (a b c : ℝ) (A B C OC : ℝ)
  (h1 : a = c * Math.sin A / Math.sin C)
  (h2 : b = c * Math.sin B / Math.sin C) 
  (h3 : cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h4 : cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (h5 : (a * Math.cos B + b * Math.cos A) / c = (Real.sqrt 3 * (a^2 + b^2 - c^2)) / (2 * a * b * Math.sin C))
  (h6 : A + B + C = π)
  (O := ∃ O, O = bisector_point A B C)
  (h7 : Math.dist O A = 3)
  (h8 : Math.dist O B = 5)
  (h9 : AB = 7)
  (h10 : Area(O, A, B) = (15 * Real.sqrt 3) / 4) :
  (C = π / 3 ∧ OC = (15 * Real.sqrt 3) / 7) :=
by sorry


end geometry_problem_l678_678220


namespace midpoint_locus_annulus_l678_678334

-- Definitions of the circles and their properties
variables {k1 k2 : Type} [metric_space k1] [metric_space k2]
variables {M1 : k1 → ℝ^3} {M2 : k2 → ℝ^3}
variables (O1 O2 : ℝ^3) (r1 r2 : ℝ)

-- Midpoint function definition
def midpoint (p1 p2 : ℝ^3) : ℝ^3 := (p1 + p2) / 2

-- Condition: M1 moves on circle k1 with center O1 and radius r1
-- Condition: M2 moves on circle k2 with center O2 and radius r2
axiom M1_on_k1 (M1 : k1 → ℝ^3) : ∀ (x : k1), dist (M1 x) O1 = r1
axiom M2_on_k2 (M2 : k2 → ℝ^3) : ∀ (y : k2), dist (M2 y) O2 = r2

-- Goal: The locus of the midpoint of M1M2 lies in an annular region centered at the midpoint of O1 and O2
theorem midpoint_locus_annulus :
  let O := (O1 + O2) / 2 in
  let outer_radius := (r1 + r2) / 2 in
  let inner_radius := (r1 - r2) / 2 in
  ∀ (x : k1) (y : k2), 
    let mid := midpoint (M1 x) (M2 y) in
    dist mid O ≤ outer_radius ∧ dist mid O ≥ inner_radius := 
by {
  sorry
}

end midpoint_locus_annulus_l678_678334


namespace num_two_digit_powers_of_3_l678_678962

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678962


namespace total_profit_theorem_l678_678729

variables (a b c : ℝ)
variables (total_investment b_profit total_profit : ℝ)

-- Conditions
def conditions := 
  a + b + c = 50000 ∧
  a = b + 4000 ∧
  b = c + 5000 ∧
  b_profit = 10200 ∧
  b / total_investment = 17000 / 50000

-- Main theorem
theorem total_profit_theorem (h : conditions a b c total_investment b_profit total_profit) : 
  total_profit = 30000 :=
sorry

end total_profit_theorem_l678_678729


namespace defective_rate_y_l678_678440

variables (dx dy : ℝ) (px py : ℝ)

noncomputable def calculate_defective_rate_y : ℝ :=
  (0.007 - 0.005 * (1/3)) / (2/3)

theorem defective_rate_y :
  (∀ x dx py, dx = 0.005 → px = 1 / 3 → py = 2 / 3 → 
   0.007 = (px * dx) + (py * dy)) → 
  dy = 0.008 :=
by
  intro h
  have h_defect_rate : dy = calculate_defective_rate_y
  sorry

end defective_rate_y_l678_678440


namespace two_digit_numbers_in_form_3_pow_n_l678_678972

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678972


namespace cross_product_correct_l678_678449

def vector1 : ℝ × ℝ × ℝ := (3, 1, 4)
def vector2 : ℝ × ℝ × ℝ := (6, -2, 8)
def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.2.1 * v2.2.2 - v1.2.2 * v2.2.1,
   v1.2.2 * v2.1 - v1.2.1 * v2.3,
   v1.1 * v2.2.1 - v1.2.1 * v2.1)

theorem cross_product_correct : cross_product vector1 vector2 = (16, 0, -12) := by
  sorry

end cross_product_correct_l678_678449


namespace cost_of_student_ticket_l678_678326

theorem cost_of_student_ticket
  (cost_adult : ℤ)
  (total_tickets : ℤ)
  (total_revenue : ℤ)
  (adult_tickets : ℤ)
  (student_tickets : ℤ)
  (H1 : cost_adult = 6)
  (H2 : total_tickets = 846)
  (H3 : total_revenue = 3846)
  (H4 : adult_tickets = 410)
  (H5 : student_tickets = 436)
  : (total_revenue = adult_tickets * cost_adult + student_tickets * (318 / 100)) :=
by
  -- mathematical proof steps would go here
  sorry

end cost_of_student_ticket_l678_678326


namespace luke_total_points_l678_678636

-- Definitions based on conditions
def points_per_round : ℕ := 3
def rounds_played : ℕ := 26

-- Theorem stating the question and correct answer
theorem luke_total_points : points_per_round * rounds_played = 78 := 
by 
  sorry

end luke_total_points_l678_678636


namespace average_calls_per_day_l678_678598

/-- Conditions: Jean's calls per day -/
def calls_mon : ℕ := 35
def calls_tue : ℕ := 46
def calls_wed : ℕ := 27
def calls_thu : ℕ := 61
def calls_fri : ℕ := 31

/-- Assertion: The average number of calls Jean answers per day -/
theorem average_calls_per_day :
  (calls_mon + calls_tue + calls_wed + calls_thu + calls_fri) / 5 = 40 :=
by sorry

end average_calls_per_day_l678_678598


namespace three_powers_in_two_digit_range_l678_678999

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l678_678999


namespace monotonic_function_inequalities_l678_678914

noncomputable theory
open Classical

variable {f : ℝ → ℝ}
variable (a : ℝ)

-- Conditions
def is_even_shift (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = f (-x + 2)

def derivative_condition (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 2 → (x - 2) * f' x > 0

def a_range (a : ℝ) : Prop :=
  2 < a ∧ a < 3

-- We need to prove this
theorem monotonic_function_inequalities 
  (even_shift : is_even_shift f) 
  (deriv_cond : ∀ x, x ≠ 2 → (x - 2) * deriv_cond f' x)
  (a_in_range : a_range a) :
  let log_a := Real.log a / Real.log 2 in
  1 < log_a ∧ log_a < 2 →
  let pow_a := Real.pow 2 a in
  4 < pow_a ∧ pow_a < 8 →
  f log_a < f 3 ∧ f 3 < f pow_a := 
sorry

end monotonic_function_inequalities_l678_678914


namespace intersection_points_count_l678_678865

-- Define the equations
def eq1 (x y : ℝ) : Prop := x - y + 2 = 0
def eq2 (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0
def eq3 (x y : ℝ) : Prop := 3 * x - 2 * y - 1 = 0
def eq4 (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- Define the combined system for intersection points
def system1 (x y : ℝ) : Prop := eq1 x y ∧ eq3 x y
def system2 (x y : ℝ) : Prop := eq1 x y ∧ eq4 x y

-- The property to be proved
theorem intersection_points_count : 
  (∃ x y : ℝ, system1 x y) ∧ (∃ x y : ℝ, system2 x y) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, system1 x1 y1 → system2 x2 y2 → (x1 ≠ x2 ∨ y1 ≠ y2)) → 
  2 := 
by
  sorry

end intersection_points_count_l678_678865


namespace exists_positive_integer_divisible_by_15_and_sqrt_in_range_l678_678857

theorem exists_positive_integer_divisible_by_15_and_sqrt_in_range :
  ∃ (n : ℕ), (n % 15 = 0) ∧ (28 < Real.sqrt n) ∧ (Real.sqrt n < 28.5) ∧ (n = 795) :=
by
  sorry

end exists_positive_integer_divisible_by_15_and_sqrt_in_range_l678_678857


namespace rhombus_diagonals_perpendicular_not_rectangle_diagonal_property_l678_678759

/-- A type representing a general parallelogram -/
structure Parallelogram where
  sides_parallel_and_congruent : Prop

/-- A rhombus is a special type of parallelogram where the diagonals bisect each other and are perpendicular -/
structure Rhombus extends Parallelogram where
  diagonals_bisect : Prop
  diagonals_perpendicular : Prop

/-- A rectangle is a special type of parallelogram where the diagonals bisect each other and are congruent -/
structure Rectangle extends Parallelogram where
  diagonals_bisect : Prop
  diagonals_congruent : Prop

/-- The problem is to prove that a property unique to a rhombus is that its diagonals are perpendicular -/
theorem rhombus_diagonals_perpendicular_not_rectangle_diagonal_property 
  (rh : Rhombus) 
  (rect : Rectangle) : 
  rh.diagonals_perpendicular ∧ ¬ rect.diagonals_perpendicular := 
sorry

end rhombus_diagonals_perpendicular_not_rectangle_diagonal_property_l678_678759


namespace math_problem_l678_678520

theorem math_problem (m n : ℝ) (hmn : 0 < m ∧ m < n) :
  (∀ x : ℝ, f x = Real.log x - x + 1) →
  (∀ x : ℝ, f' x = 1 - 1) →
  (f m < f n) →
  (0 < m ∧ m < n) →
  (∃! a : ℝ, a = 1) →
  ∃ a : ℝ, (a = 1) → (\forall 0 < m < n, \frac{1}{n} - 1 < \frac{(\ln n - n + 1) - (\ln m - m + 1)}{n - m} < \frac{1}{m} - 1) :=
by
  sorry

end math_problem_l678_678520


namespace problem_statement_l678_678155

def f (x : ℝ) : ℝ :=
  if x >= 0 then Real.log x + 3 / Real.log 2 else x^2

theorem problem_statement : f (f (-1)) = 2 :=
by
  -- Using given piecewise function definition in Lean
  let f (x : ℝ) : ℝ := if x >= 0 then Real.log (x + 3) / Real.log 2 else x^2
  have h1 : f(-1) = (-1)^2 := by
    simp [f]
    sorry
  have h2 : f(f(-1)) = f(1) := by
    rw [h1]
    sorry
  show f(f(-1)) = 2
  sorry

end problem_statement_l678_678155


namespace number_of_balls_to_remove_l678_678378

theorem number_of_balls_to_remove:
  ∀ (x : ℕ), 120 - x = (48 : ℕ) / (0.75 : ℝ) → x = 56 :=
by sorry

end number_of_balls_to_remove_l678_678378


namespace reflection_of_F_l678_678689

def initial_shape : char := 'F'
def vertical_reflection (c : char) : char :=
  if c = 'F' then 'E' else c -- simplifying the logic for this specific problem
def horizontal_reflection (c : char) : char :=
  if c = 'E' then 'H' else c -- simplifying the logic for this specific problem

theorem reflection_of_F :
  horizontal_reflection (vertical_reflection initial_shape) = 'H' :=
by
  simp [initial_shape, vertical_reflection, horizontal_reflection]
  sorry

end reflection_of_F_l678_678689


namespace prod_cos_prime_l678_678355

theorem prod_cos_prime (n : ℕ) (h_prime: Prime n) (h_gt3 : n > 3) : 
  (∏ k in Finset.range (n - 1), (1 + 2 * Real.cos (2 * k * Real.pi / n))) = 3 := 
sorry

end prod_cos_prime_l678_678355


namespace num_two_digit_powers_of_3_l678_678955

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678955


namespace total_airflow_in_one_week_l678_678024

-- Define the conditions
def airflow_rate : ℕ := 10 -- liters per second
def working_time_per_day : ℕ := 10 -- minutes per day
def days_per_week : ℕ := 7

-- Define the conversion factors
def minutes_to_seconds : ℕ := 60

-- Define the total working time in seconds
def total_working_time_per_week : ℕ := working_time_per_day * days_per_week * minutes_to_seconds

-- Define the expected total airflow in one week
def expected_total_airflow : ℕ := airflow_rate * total_working_time_per_week

-- Prove that the expected total airflow is 42000 liters
theorem total_airflow_in_one_week : expected_total_airflow = 42000 := 
by
  -- assertion is correct given the conditions above 
  -- skip the proof
  sorry

end total_airflow_in_one_week_l678_678024


namespace rectangular_floor_length_l678_678035

theorem rectangular_floor_length
    (cost_per_square : ℝ)
    (total_cost : ℝ)
    (carpet_length : ℝ)
    (carpet_width : ℝ)
    (floor_width : ℝ)
    (floor_area : ℝ) 
    (H1 : cost_per_square = 15)
    (H2 : total_cost = 225)
    (H3 : carpet_length = 2)
    (H4 : carpet_width = 2)
    (H5 : floor_width = 6)
    (H6 : floor_area = floor_width * carpet_length * carpet_width * 15): 
    floor_area / floor_width = 10 :=
by
  sorry

end rectangular_floor_length_l678_678035


namespace claudia_fills_4ounce_glasses_l678_678424

theorem claudia_fills_4ounce_glasses :
  ∀ (total_water : ℕ) (five_ounce_glasses : ℕ) (eight_ounce_glasses : ℕ) 
    (four_ounce_glass_volume : ℕ),
  total_water = 122 →
  five_ounce_glasses = 6 →
  eight_ounce_glasses = 4 →
  four_ounce_glass_volume = 4 →
  (total_water - (five_ounce_glasses * 5 + eight_ounce_glasses * 8)) / four_ounce_glass_volume = 15 :=
by
  intros _ _ _ _ _ _ _ _ 
  sorry

end claudia_fills_4ounce_glasses_l678_678424


namespace population_after_decade_l678_678313

theorem population_after_decade : 
  ∀ (P r : ℕ) (n : ℕ), 
  P = 175000 ∧ r = 7 ∧ n = 10 → 
  P * (1 + r / 100) ^ n = 344251 := 
by 
  intros P r n h,
  cases h with h1 hrn,
  cases hrn with h2 h3,
  rw [h1, h2, h3],
  norm_num, -- Simplifies the arithmetic using Lean's normalization
  sorry      -- Skips the proof for now (non-trivial computation)

end population_after_decade_l678_678313


namespace general_cosine_identity_l678_678389

theorem general_cosine_identity (α : ℝ) :
  real.cos (α - 120 * real.pi / 180) + real.cos α + real.cos (α + 120 * real.pi / 180) = 0 := by
  sorry

end general_cosine_identity_l678_678389


namespace simplify_radicals_l678_678666

theorem simplify_radicals :
  (sqrt 5 - sqrt 40 + sqrt 45) = (4 * sqrt 5 - 2 * sqrt 10) :=
by
  have h1 : sqrt 40 = sqrt (4 * 10) := by sorry,
  have h2 : sqrt 40 = sqrt 4 * sqrt 10 := by sorry,
  have h3 : sqrt 45 = sqrt (9 * 5) := by sorry,
  have h4 : sqrt 45 = sqrt 9 * sqrt 5 := by sorry,
  rw [h2, h4],
  calc
    sqrt 5 - sqrt 40 + sqrt 45
        = sqrt 5 - (sqrt 4 * sqrt 10) + (sqrt 9 * sqrt 5) : by rw [h2, h4]
    ... = sqrt 5 - (2 * sqrt 10) + (3 * sqrt 5) : by rw [sqrt_mul, sqrt_mul]
    ... = (1 + 3) * sqrt 5 - 2 * sqrt 10 : by rw [add_mul]
    ... = 4 * sqrt 5 - 2 * sqrt 10 : by ring

end simplify_radicals_l678_678666


namespace two_digit_powers_of_3_count_l678_678985

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678985


namespace a_2009_eq_neg_6_l678_678486

-- Define the sequence with the given conditions
def a : ℕ → ℤ
| 0     := 3
| 1     := 6
| (n+2) := a (n+1) - a n

-- State the theorem to prove
theorem a_2009_eq_neg_6 : a 2009 = -6 :=
sorry

end a_2009_eq_neg_6_l678_678486


namespace total_volume_of_water_l678_678370

theorem total_volume_of_water (num_containers : ℕ) (volume_per_container : ℕ)
  (h_num_containers : num_containers = 2744)
  (h_volume_per_container : volume_per_container = 4) :
  num_containers * volume_per_container = 10976 :=
by {
  rw [h_num_containers, h_volume_per_container],
  norm_num,
}

end total_volume_of_water_l678_678370


namespace count_integers_satisfying_inequality_l678_678184

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), S.card = 8 ∧ ∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 :=
by
  sorry

end count_integers_satisfying_inequality_l678_678184


namespace two_digit_powers_of_three_l678_678950

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678950


namespace infinity_gcd_binom_l678_678239

theorem infinity_gcd_binom {k l : ℕ} : ∃ᶠ m in at_top, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end infinity_gcd_binom_l678_678239


namespace rectangle_perimeter_l678_678037

theorem rectangle_perimeter (s : ℝ) (h1 : 4 * s = 180) :
    let length := s
    let width := s / 3
    2 * (length + width) = 120 := 
by
  sorry

end rectangle_perimeter_l678_678037


namespace two_digit_powers_of_three_l678_678951

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678951


namespace nell_final_cards_l678_678638

noncomputable def final_number_of_cards 
  (initial_cards : ℕ) 
  (cards_given_to_jeff : ℕ) 
  (cards_bought_from_collector : ℕ) 
  (percentage_given_to_brother : ℚ) 
  : ℕ :=
  let cards_after_giving_to_jeff := initial_cards - cards_given_to_jeff
  let cards_after_buying_more := cards_after_giving_to_jeff + cards_bought_from_collector
  let cards_given_to_brother := (cards_after_buying_more * percentage_given_to_brother).to_nat
  cards_after_buying_more - cards_given_to_brother

theorem nell_final_cards : 
  final_number_of_cards 15350 4876 3129 0.12 = 11971 :=
  sorry

end nell_final_cards_l678_678638


namespace find_a_l678_678468

theorem find_a (a n : ℤ) (h1 : 2^n = 128) (h2 : (a - 2)^7 = 1) : a = 3 := by
  sorry

end find_a_l678_678468


namespace barry_wand_trick_l678_678641

theorem barry_wand_trick (n : ℕ) (h : (n + 3 : ℝ) / 3 = 50) : n = 147 := by
  sorry

end barry_wand_trick_l678_678641


namespace quadratic_intersects_x_axis_at_two_points_intersection_points_when_symmetric_to_y_axis_l678_678924

-- Define the quadratic function
def f (x k : ℝ) : ℝ := x^2 - 2 * k * x - 1

-- Problem 1: Prove that for all real k, the quadratic function intersects the x-axis at two distinct points
theorem quadratic_intersects_x_axis_at_two_points (k : ℝ) : 
  let Δ := (2 * k)^2 - 4 * 1 * (-1) in
  Δ > 0 := by
  let Δ := (2 * k)^2 - 4 * 1 * (-1)
  have h1 : Δ = 4 * k^2 + 4 := by
    sorry -- Intermediate calculation
  have h2 : 4 * k^2 + 4 > 0 := by
    sorry -- Proof that the discriminant is always positive
  exact h2

-- Problem 2: If the graph has the y-axis as its axis of symmetry, find the intersection points with the x-axis
theorem intersection_points_when_symmetric_to_y_axis :
  ∃ x, f x 0 = 0 ∧ (x = 1 ∨ x = -1) := by
  let k := 0
  let f_symmetric := f k
  have h : f_symmetric 1 = 0 ∧ f_symmetric (-1) = 0 := by
    sorry -- Proof that f(1, 0) = 0 and f(-1, 0) = 0
  existsi 1
  existsi -1
  exact h

end quadratic_intersects_x_axis_at_two_points_intersection_points_when_symmetric_to_y_axis_l678_678924


namespace find_height_of_triangular_prism_l678_678045

-- Define the conditions
def volume (V : ℝ) : Prop := V = 120
def base_side1 (a : ℝ) : Prop := a = 3
def base_side2 (b : ℝ) : Prop := b = 4

-- The final proof problem
theorem find_height_of_triangular_prism (V : ℝ) (a : ℝ) (b : ℝ) (h : ℝ) 
  (h1 : volume V) (h2 : base_side1 a) (h3 : base_side2 b) : h = 20 :=
by
  -- The actual proof goes here
  sorry

end find_height_of_triangular_prism_l678_678045


namespace two_digit_powers_of_three_l678_678944

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678944


namespace evaluate_polynomial_at_3_l678_678330

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem evaluate_polynomial_at_3 : f 3 = 1 := by
  sorry

end evaluate_polynomial_at_3_l678_678330


namespace identify_clothing_l678_678776

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l678_678776


namespace complement_intersection_l678_678531

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection (hU : U = {2, 3, 6, 8}) (hA : A = {2, 3}) (hB : B = {2, 6, 8}) :
  ((U \ A) ∩ B) = {6, 8} := 
by
  sorry

end complement_intersection_l678_678531


namespace complex_modulus_identity_l678_678147

variable {ℂ : Type} [is_R_or_C ℂ]
variables (z₁ z₂ z₃ z₄ : ℂ)
variables (hz₁ : ∥z₁∥ = 1) (hz₂ : ∥z₂∥ = 1) (hz₃ : ∥z₃∥ = 1) (hz₄ : ∥z₄∥ = 1)

theorem complex_modulus_identity :
  ∥z₁ - z₂∥^2 * ∥z₃ - z₄∥^2 + ∥z₁ + z₄∥^2 * ∥z₃ - z₂∥^2 =
  ∥z₁ * (z₂ - z₃) + z₃ * (z₂ - z₁) + z₄ * (z₁ - z₃)∥^2 :=
sorry

end complex_modulus_identity_l678_678147


namespace find_m_l678_678864

theorem find_m : ∃ m : ℤ, 0 ≤ m ∧ m ≤ 15 ∧ m ≡ 12345 [MOD 16] ∧ m = 9 :=
by
  use 9
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end find_m_l678_678864


namespace four_circles_rectangle_l678_678004

open Set

theorem four_circles_rectangle 
(O S: EuclideanSpace ℝ n) 
(S1 S2 S3 S4 : Set (EuclideanSpace ℝ n))
(A1 A2 A3 A4 B1 B2 B3 B4 : EuclideanSpace ℝ n)
(hcircle: CircleProp O S)
(hcenters: ∀ (O1 O2 O3 O4 : EuclideanSpace ℝ n), O1 ∈ S1 → O2 ∈ S2 → O3 ∈ S3 → O4 ∈ S4 
→ SetOfCentersOnCircle O)
(hint1: A1 ∈ S1 ∧ A1 ∈ S2)
(hint2: A2 ∈ S2 ∧ A2 ∈ S3)
(hint3: A3 ∈ S3 ∧ A3 ∈ S4)
(hint4: A4 ∈ S4 ∧ A4 ∈ S1)
(hA_on_S: ∀ (A : EuclideanSpace ℝ n), A ∈ {A1, A2, A3, A4} → A ∈ S)
(hB_distinct: B1 ≠ B2 ∧ B2 ≠ B3 ∧ B3 ≠ B4 ∧ B4 ≠ B1 ∧ B1 ≠ B3 ∧ B2 ≠ B4)
(hB_inside: ∀ (B : EuclideanSpace ℝ n), B ∈ {B1, B2, B3, B4} → B ∈ interior (closure S)) :
Rectangle B1 B2 B3 B4 :=
sorry

end four_circles_rectangle_l678_678004


namespace conjugate_point_in_third_quadrant_l678_678905

noncomputable def z : ℂ := (-3 + 1*I) / (2 + 1*I)
noncomputable def z_conjugate : ℂ := conj z
noncomputable def point_in_third_quadrant : Prop := (z_conjugate.re < 0) ∧ (z_conjugate.im < 0)

theorem conjugate_point_in_third_quadrant : point_in_third_quadrant :=
by
  sorry

end conjugate_point_in_third_quadrant_l678_678905


namespace distinct_roots_quadratic_find_k_given_one_root_l678_678527

-- Part (1): Proving the quadratic equation has two distinct real roots
theorem distinct_roots_quadratic (k : ℝ) : 
  let a := 1
  let b := -(2 * k + 1)
  let c := k^2 + k
  let Δ := b^2 - 4 * a * c
  Δ = 1 → Δ > 0 := 
by 
  intros a b c Δ h
  sorry

-- Part (2): Finding the value of k when one root is 1
theorem find_k_given_one_root (k : ℝ) : 
  (1 : ℝ)^2 - (2 * k + 1) * 1 + k^2 + k = 0 → 
  k = 0 ∨ k = 1 :=
by 
  intro h
  have h_eq : k^2 - k = 0,
  {
    sorry
  }
  have h_factored : k * (k - 1) = 0,
  {
    sorry
  }
  exact h_factored
  sorry

end distinct_roots_quadratic_find_k_given_one_root_l678_678527


namespace quadratic_roots_exist_intersection_points_y_axis_symmetry_l678_678922

-- Part (1): Prove the quadratic function intersects x-axis at two points for any k.
theorem quadratic_roots_exist (k : ℝ) : 
  let Δ := (-2*k)^2 - 4 * 1 * (-1) in
  Δ > 0 := by
  sorry

-- Part (2): Given y-axis as axis of symmetry for the quadratic function, find intersection points with x-axis.
theorem intersection_points_y_axis_symmetry :
  let k := 0 in
  let y := λ x : ℝ, x^2 - 2*k*x - 1 in
  y 1 = 0 ∧ y (-1) = 0 := by
  sorry

end quadratic_roots_exist_intersection_points_y_axis_symmetry_l678_678922


namespace cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l678_678222

noncomputable def p1 (x : ℝ) : ℝ :=
  if x < -1 ∨ x > 1 then 0 else 0.5

noncomputable def p2 (y : ℝ) : ℝ :=
  if y < 0 ∨ y > 2 then 0 else 0.5

noncomputable def F1 (x : ℝ) : ℝ :=
  if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1

noncomputable def F2 (y : ℝ) : ℝ :=
  if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1

noncomputable def p (x : ℝ) (y : ℝ) : ℝ :=
  if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25

noncomputable def F (x : ℝ) (y : ℝ) : ℝ :=
  if x ≤ -1 ∨ y ≤ 0 then 0
  else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y 
  else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
  else if x > 1 ∧ y ≤ 2 then 0.5 * y
  else 1

theorem cumulative_distribution_F1 (x : ℝ) : 
  F1 x = if x ≤ -1 then 0 else if x ≤ 1 then 0.5 * (x + 1) else 1 := by sorry

theorem cumulative_distribution_F2 (y : ℝ) : 
  F2 y = if y ≤ 0 then 0 else if y ≤ 2 then 0.5 * y else 1 := by sorry

theorem joint_density (x : ℝ) (y : ℝ) : 
  p x y = if (x < -1 ∨ x > 1 ∨ y < 0 ∨ y > 2) then 0 else 0.25 := by sorry

theorem joint_cumulative_distribution (x : ℝ) (y : ℝ) : 
  F x y = if x ≤ -1 ∨ y ≤ 0 then 0
          else if x ≤ 1 ∧ y ≤ 2 then 0.25 * (x + 1) * y
          else if x ≤ 1 ∧ y > 2 then 0.5 * (x + 1)
          else if x > 1 ∧ y ≤ 2 then 0.5 * y
          else 1 := by sorry

end cumulative_distribution_F1_cumulative_distribution_F2_joint_density_joint_cumulative_distribution_l678_678222


namespace prove_clothing_colors_l678_678812

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l678_678812


namespace amount_charged_for_kids_l678_678229

theorem amount_charged_for_kids (K A: ℝ) (H1: A = 2 * K) (H2: 8 * K + 10 * A = 84) : K = 3 :=
by
  sorry

end amount_charged_for_kids_l678_678229


namespace sum_abs_le_1000_l678_678284

-- Definitions
def numbers_on_circle : List (Int) := sorry -- List of 2002 numbers which are either 1 or -1
def adjacent_product_neg (l : List Int) : Prop := 
  ((l.zip (l.tail ++ [l.head])).map (λ ⟨a, b⟩, a * b)).sum < 0
def sum_of_list (l : List Int) : Int := l.sum

-- Theorem statement
theorem sum_abs_le_1000 
  (l : List Int) 
  (h1 : l.length = 2002) 
  (h2 : ∀ x, x ∈ l → x = 1 ∨ x = -1) 
  (h3 : adjacent_product_neg l) : 
  abs (sum_of_list l) ≤ 1000 := sorry

end sum_abs_le_1000_l678_678284


namespace quadratic_intersects_x_axis_at_two_points_intersection_points_when_symmetric_to_y_axis_l678_678925

-- Define the quadratic function
def f (x k : ℝ) : ℝ := x^2 - 2 * k * x - 1

-- Problem 1: Prove that for all real k, the quadratic function intersects the x-axis at two distinct points
theorem quadratic_intersects_x_axis_at_two_points (k : ℝ) : 
  let Δ := (2 * k)^2 - 4 * 1 * (-1) in
  Δ > 0 := by
  let Δ := (2 * k)^2 - 4 * 1 * (-1)
  have h1 : Δ = 4 * k^2 + 4 := by
    sorry -- Intermediate calculation
  have h2 : 4 * k^2 + 4 > 0 := by
    sorry -- Proof that the discriminant is always positive
  exact h2

-- Problem 2: If the graph has the y-axis as its axis of symmetry, find the intersection points with the x-axis
theorem intersection_points_when_symmetric_to_y_axis :
  ∃ x, f x 0 = 0 ∧ (x = 1 ∨ x = -1) := by
  let k := 0
  let f_symmetric := f k
  have h : f_symmetric 1 = 0 ∧ f_symmetric (-1) = 0 := by
    sorry -- Proof that f(1, 0) = 0 and f(-1, 0) = 0
  existsi 1
  existsi -1
  exact h

end quadratic_intersects_x_axis_at_two_points_intersection_points_when_symmetric_to_y_axis_l678_678925


namespace exists_infinite_subset_with_gcd_l678_678251

-- Definitions
def infinite_set_with_limited_prime_factors (A : Set ℤ) : Prop :=
  infinite A ∧ ∀ a ∈ A, ∃ n ≤ 1987, (prime_factors a).card ≤ n

theorem exists_infinite_subset_with_gcd 
  (A : Set ℤ) 
  (hA : infinite_set_with_limited_prime_factors A) : 
  ∃ (B : Set ℤ) (b : ℕ), infinite B ∧ (∀ x y ∈ B, gcd x y = b) :=
sorry

end exists_infinite_subset_with_gcd_l678_678251


namespace expected_value_of_D_squared_l678_678271

theorem expected_value_of_D_squared (P : ℕ → (ℝ × ℝ))
    (h₀ : ∀ i, (1 ≤ i ∧ i ≤ 16) → (P i).fst^2 + (P i).snd^2 = 16)
    (h₁ : ∀ i, (1 ≤ i ∧ i ≤ 16) → dist (P i) (P (i+1) % 16) = 1)
    (h₂ : pairwise (≠) (λ i, P i))
    (flip_coin : ℕ → Prop) :
  let Q i := if flip_coin i then P i else (-fst (P i), -snd (P i)),
      D := ∑ i in finset.range 16, (λ i, Q i) in
  let E_D_squared := expected_value (λ n, norm (D n)) in
  E_D_squared = 64 :=
sorry

end expected_value_of_D_squared_l678_678271


namespace num_two_digit_powers_of_3_l678_678961

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678961


namespace cookies_eaten_l678_678032

variable (original_cookies : ℕ) (left_cookies : ℕ)

theorem cookies_eaten (h1 : original_cookies = 18) (h2 : left_cookies = 9) : original_cookies - left_cookies = 9 :=
by
  simp [h1, h2]
  exact Nat.sub_self _

#check cookies_eaten

end cookies_eaten_l678_678032


namespace Kerry_age_l678_678607

theorem Kerry_age :
  (let cost_per_box := 2.5 in
   let total_cost := 5 in
   let number_of_boxes := total_cost / cost_per_box in
   let candles_per_box := 12 in
   let total_candles := number_of_boxes * candles_per_box in
   let number_of_cakes := 3 in
   let kerry_age := total_candles / number_of_cakes in
   kerry_age = 8) :=
by
  sorry

end Kerry_age_l678_678607


namespace map_width_l678_678395

theorem map_width (area length : ℝ) (h1 : area = 10) (h2 : length = 5) : ∃ width : ℝ, width = 2 :=
by
  use 2
  sorry

end map_width_l678_678395


namespace shaded_perimeter_l678_678586

noncomputable def perimeter := 14 + (21 * Real.pi / 2)

theorem shaded_perimeter
  (O : Point)
  (radius : ℝ)
  (OP OQ : ℝ)
  (PQ_angle : ℝ)
  (h1 : OP = 7)
  (h2 : OQ = 7)
  (h3 : PQ_angle = (270 : ℝ)) :
  perimeter = 14 + (21 * Real.pi / 2) :=
by
  sorry

end shaded_perimeter_l678_678586


namespace percent_defective_units_shipped_for_sale_l678_678215

variable (total_units : ℕ)
variable (defective_units_percentage : ℝ := 0.08)
variable (shipped_defective_units_percentage : ℝ := 0.05)

theorem percent_defective_units_shipped_for_sale :
  defective_units_percentage * shipped_defective_units_percentage * 100 = 0.4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l678_678215


namespace symmetric_point_sum_eq_seven_l678_678504

variable (A : ℝ × ℝ) (m n : ℝ)

theorem symmetric_point_sum_eq_seven (h : A = (2, -5)) (h_symmetric : A = (m, -(n))) : m + n = 7 :=
by
  have h_A : A = (2, -5) := h
  have h_m : m = 2 := by
    rw h_A at h_symmetric
    cases h_symmetric
    exact rfl
  have h_n : n = 5 := by
    rw h_A at h_symmetric
    cases h_symmetric
    exact rfl
  rw [h_m, h_n]
  rfl

#eval symmetric_point_sum_eq_seven (2, -5) 2 5 (rfl) (by exact rfl)

end symmetric_point_sum_eq_seven_l678_678504


namespace pure_imaginary_solution_l678_678501

theorem pure_imaginary_solution (a : ℝ) (h : (a^2 : ℂ) + complex.I / (1 - complex.I) = pure_imaginary a) : a = 1 ∨ a = -1 :=
sorry

end pure_imaginary_solution_l678_678501


namespace triangle_inequality_right_triangle_l678_678591

theorem triangle_inequality_right_triangle
  (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a + b) / Real.sqrt 2 ≤ c :=
by sorry

end triangle_inequality_right_triangle_l678_678591


namespace xy_equals_18_l678_678545

theorem xy_equals_18 (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 :=
by
  sorry

end xy_equals_18_l678_678545


namespace investment_amount_l678_678354

theorem investment_amount (R T V : ℝ) (hT : T = 0.9 * R) (hV : V = 0.99 * R) (total_sum : R + T + V = 6936) : R = 2400 :=
by sorry

end investment_amount_l678_678354


namespace number_of_centroid_positions_l678_678285

/-- Define the vertices of the rectangle -/
def vertices : set (ℝ × ℝ) :=
  {(0, 0), (12, 0), (12, 8), (0, 8)}

/-- Define the set of 48 equally spaced points along the perimeter of the rectangle -/
noncomputable def perimeter_points : set (ℝ × ℝ) := sorry

/-- Define the centroid of a triangle given three points, not necessarily distinct -/
def centroid (P Q R : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := P;
      (x2, y2) := Q;
      (x3, y3) := R
  in ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

/-- Prove the number of distinct centroid positions -/
theorem number_of_centroid_positions (P Q R : ℝ × ℝ)
  (hP : P ∈ perimeter_points) (hQ : Q ∈ perimeter_points) (hR : R ∈ perimeter_points)
  (hnc : ¬ collinear ℝ {P, Q, R}) :
  (finset.image (λ (pqr : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)), centroid pqr.1 pqr.2.1 pqr.2.2)
    ((finset.product perimeter_points.to_finset
      (finset.product perimeter_points.to_finset perimeter_points.to_finset)))
    ).card = 805 := sorry

end number_of_centroid_positions_l678_678285


namespace two_digit_numbers_in_form_3_pow_n_l678_678967

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678967


namespace find_exponent_l678_678099

theorem find_exponent (w : ℤ) : 7^3 * 7^w = 49 → w = -1 :=
by sorry

end find_exponent_l678_678099


namespace num_two_digit_powers_of_3_l678_678939

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678939


namespace system1_solution_system2_solution_l678_678671

theorem system1_solution : 
  ∃ (x y : ℤ), 2 * x + 3 * y = -1 ∧ y = 4 * x - 5 ∧ x = 1 ∧ y = -1 := by 
    sorry

theorem system2_solution : 
  ∃ (x y : ℤ), 3 * x + 2 * y = 20 ∧ 4 * x - 5 * y = 19 ∧ x = 6 ∧ y = 1 := by 
    sorry

end system1_solution_system2_solution_l678_678671


namespace initial_integer_value_l678_678401

theorem initial_integer_value (x : ℤ) (h : (x + 2) * (x + 2) = x * x - 2016) : x = -505 := 
sorry

end initial_integer_value_l678_678401


namespace number_of_valid_pairs_l678_678105

theorem number_of_valid_pairs : 
  (card { (b, c) : ℕ × ℕ | b > 0 ∧ c > 0 ∧ b^2 ≥ 4 * c ∧ c^2 ≥ 4 * b }) = 5 :=
by sorry

end number_of_valid_pairs_l678_678105


namespace five_digit_palindromes_count_l678_678064

theorem five_digit_palindromes_count : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) → 
  900 = 9 * 10 * 10 := 
by
  intro h
  sorry

end five_digit_palindromes_count_l678_678064


namespace equal_areas_of_quadrilaterals_l678_678325

theorem equal_areas_of_quadrilaterals
  (A B C D M M1 M2 M3 M4 : ℝ × ℝ)
  (hM1 : midpoint A B M1) (hM2 : midpoint B C M2)
  (hM3 : midpoint C D M3) (hM4 : midpoint D A M4)
  (hM_inter : intersection_point_of_parallel_lines M1 M2 M3 M4 M) :
  let area := λ P Q R S : ℝ × ℝ, area_of_quadrilateral P Q R S in
  area M M1 A M4 = area M M1 B M2 ∧
  area M M1 B M2 = area M M2 C M3 ∧
  area M M2 C M3 = area M M3 D M4 :=
by sorry

end equal_areas_of_quadrilaterals_l678_678325


namespace right_triangle_third_side_l678_678884

theorem right_triangle_third_side (a b : ℝ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :
  c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2) :=
by 
  sorry

end right_triangle_third_side_l678_678884


namespace non_receivers_after_2020_candies_l678_678008

noncomputable def count_non_receivers (k n : ℕ) : ℕ := 
sorry

theorem non_receivers_after_2020_candies :
  count_non_receivers 73 2020 = 36 :=
sorry

end non_receivers_after_2020_candies_l678_678008


namespace gravel_pile_volume_l678_678027

/-- A function that calculates the volume of a cone given its diameter and the height as a
proportion of its diameter. -/
def cone_volume (d : ℝ) (h : ℝ) : ℝ := (1/3) * π * (d/2)^2 * h

theorem gravel_pile_volume :
  let d := 10 in
  let h := 0.6 * d in
  cone_volume d h = 50 * π :=
by
  sorry

end gravel_pile_volume_l678_678027


namespace intersect_complement_A_B_eq_l678_678530

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

noncomputable def complement_A : Set ℝ := U \ A
noncomputable def intersection_complement_A_B : Set ℝ := complement_A U A ∩ B

theorem intersect_complement_A_B_eq : 
  U = univ ∧ A = {x : ℝ | x + 1 < 0} ∧ B = {x : ℝ | x - 3 < 0} →
  intersection_complement_A_B U A B = Icc (-1 : ℝ) 3 :=
by
  intro h
  sorry

end intersect_complement_A_B_eq_l678_678530


namespace crossword_max_ratio_l678_678488

theorem crossword_max_ratio (n : ℕ) (hn : n ≥ 2) (x y : ℕ) 
  (hx : ∀ crossword, is_crossword crossword → x = num_words crossword)
  (hy : ∀ crossword, is_crossword crossword → y = min_cover_words crossword) :
  ∃ crossword, is_crossword crossword ∧
    max_ratio crossword n = 1 + n / 2 :=
by
  sorry

def is_crossword (crossword : Set (ℕ × ℕ)) : Prop :=
  -- Add appropriate conditions here
  sorry

def num_words (crossword : Set (ℕ × ℕ)) : ℕ :=
  -- Calculate the number of words in the crossword
  sorry

def min_cover_words (crossword : Set (ℕ × ℕ)) : ℕ :=
  -- Calculate the minimum number of words needed to cover the crossword
  sorry

def max_ratio (crossword : Set (ℕ × ℕ)) (n : ℕ) : ℚ :=
  let x := num_words crossword
  let y := min_cover_words crossword
  (x : ℚ) / (y : ℚ)

end crossword_max_ratio_l678_678488


namespace find_d_l678_678130

-- Define the arithmetic sequence sum function based on the provided conditions.
def arithmetic_seq_sum (a₁ : ℝ) (d : ℝ) (n : ℝ) : ℝ :=
  (n / 2) * (2 * a₁ + (n - 1) * d)

-- Define the given condition in the problem statement.
theorem find_d (a₁ : ℝ) (d : ℝ) 
  (h : (arithmetic_seq_sum a₁ d 2016) / 2016 - (arithmetic_seq_sum a₁ d 16) / 16 = 100) :
  d = 1 / 10 :=
by
  sorry

end find_d_l678_678130


namespace sum_of_factors_72_l678_678415

theorem sum_of_factors_72 : ∑ d in (finset.filter (∣ 72) (finset.range (73))), d = 195 :=
by
  -- given condition: 72 = 2^3 * 3^2
  have h : factors 72 = [2, 2, 2, 3, 3],
  { sorry },
  -- steps to compute the sum of factors based on the prime factorization
  sorry

end sum_of_factors_72_l678_678415


namespace math_problem_l678_678066

def calc_expr : Int := 
  54322 * 32123 - 54321 * 32123 + 54322 * 99000 - 54321 * 99001

theorem math_problem :
  calc_expr = 76802 := 
by
  sorry

end math_problem_l678_678066


namespace real_root_exists_l678_678293

noncomputable def polynomial := 2 * X^5 + X^4 - 20 * X^3 - 10 * X^2 + 2 * X + 1

theorem real_root_exists : Polynomial.eval (Real.sqrt 3 + Real.sqrt 2) polynomial = 0 := 
by
  sorry

end real_root_exists_l678_678293


namespace entree_cost_14_l678_678175

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l678_678175


namespace maximum_grade_economics_l678_678356

theorem maximum_grade_economics :
  (∃ (x y : ℝ), x + y = 4.6 ∧ 2.5 * x ≤ 5 ∧ 1.5 * y ≤ 5 ∧
                 let O_mic := 2.5 * x, O_mac := 1.5 * y in
                 let A := 0.25 * O_mic + 0.75 * O_mac,
                     B := 0.75 * O_mic + 0.25 * O_mac in
                 ⌈min A B⌉ = 4)
:= sorry

end maximum_grade_economics_l678_678356


namespace max_distinct_lines_l678_678771

theorem max_distinct_lines (n : ℕ) (hn : n = 2022) : 
  ∃ f : ℕ → ℕ, (∀ k : ℕ, f k = k + 3) ∧ f n = 2025 :=
by
  let f := λ k, k + 3
  have h1 : ∀ k, f k = k + 3 := λ k, rfl
  have h2 : f 2022 = 2025 := rfl
  use f
  exact ⟨h1, h2⟩

end max_distinct_lines_l678_678771


namespace lock_opens_with_four_digits_lock_opens_with_five_digits_l678_678693

-- Part (a)
theorem lock_opens_with_four_digits {a b c d : ℕ} (h₁ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h₂ : a + b + c = 14 ∨ a * b * c = 14)
  (h₃ : a + c + d = 14 ∨ a * c * d = 14) :
  set.insert d (set.insert c (set.insert b {a})) = {1, 2, 5, 7} :=
sorry

-- Part (b)
theorem lock_opens_with_five_digits {a b c d e : ℕ} (h₁ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h₂ : a + b + c = 14 ∨ a * b * c = 14)
  (h₃ : a + c + d = 14 ∨ a * c * d = 14) : 
  set.insert e (set.insert d (set.insert c (set.insert b {a}))) = {1, 2, 5, 7, 9} :=
sorry

end lock_opens_with_four_digits_lock_opens_with_five_digits_l678_678693


namespace determine_clothes_l678_678817

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l678_678817


namespace triangle_inequality_acute_l678_678578

theorem triangle_inequality_acute (
  A B C X Y : Type
) [has_lt A] [has_lt B] [has_lt C] [has_lt X] [has_lt Y] [has_div C] [has_ge B] [has_add A] : 
  let ab := AB in
  let bc := BC in
  let ac := AC in
  triangle_acute ABC →
  ab > bc →
  AX = BY →
  XY ≥ AC / 2 :=
sorry

end triangle_inequality_acute_l678_678578


namespace average_calls_per_day_l678_678600

theorem average_calls_per_day :
  let calls := [35, 46, 27, 61, 31] in
  (calls.sum / (calls.length : ℝ)) = 40 :=
by
  sorry

end average_calls_per_day_l678_678600


namespace possible_values_of_A_B_l678_678485

theorem possible_values_of_A_B
  (n : ℕ)
  (h_n : n = 103)
  (red_vertices : ℕ)
  (h_red_vertices : red_vertices = 79)
  (blue_vertices : ℕ)
  (h_blue_vertices : blue_vertices = n - red_vertices)
  (A : ℕ)
  (B : ℕ)
  (h_A : A = red_vertices - (n - red_vertices - B))
  (values_B : ℕ → Prop)
  (h_values_B : ∀ B, values_B B ↔ (0 ≤ B ∧ B ≤ n - red_vertices))
  (non_similar_colorings : ℕ)
  (h_non_similar_colorings : ∀ (B : ℕ), B = 14 → non_similar_colorings = (nat.choose 24 10 * nat.choose 78 9) / 24):
  (∀ (B : ℕ), 0 ≤ B ∧ B ≤ 23 → A = 55 + B) ∧ non_similar_colorings = (nat.choose 24 10 * nat.choose 78 9) / 24 :=
begin
  sorry
end

end possible_values_of_A_B_l678_678485


namespace disk_tangent_10_24_l678_678372

noncomputable def radius_clock := 30 -- radius of the clock face in cm
noncomputable def radius_disk := 6   -- radius of the disk in cm
noncomputable def circumference_clock := 2 * 30 * Real.pi
noncomputable def circumference_disk := 2 * 6 * Real.pi
noncomputable def movement_ratio := circumference_disk / circumference_clock
noncomputable def tangency_angle_movement := 360 * movement_ratio

theorem disk_tangent_10_24 :
    ((movement_ratio * 360) = 72) ∧ 
    (10.4 / 12 ≈ (1 / 5) : Real) /- 1/5th of the clock face circumference -/
    → ((3 * 30 * Real.pi) + 72 = 10 * 30 * Real.pi + 24) := sorry

end disk_tangent_10_24_l678_678372


namespace fruit_vendor_profit_l678_678751

theorem fruit_vendor_profit :
  ∀ (cost_price_per_orange selling_price_per_orange desired_profit : ℚ), 
  (cost_price_per_orange = 15 / 8) →
  (selling_price_per_orange = 18 / 6) →
  (desired_profit = 150) →
  let profit_per_orange := selling_price_per_orange - cost_price_per_orange in
  let number_of_oranges := desired_profit / profit_per_orange in
  number_of_oranges.ceil = 134 :=
by
  intros cost_price_per_orange selling_price_per_orange desired_profit
  assume h1 h2 h3
  let profit_per_orange := selling_price_per_orange - cost_price_per_orange
  have : profit_per_orange = 3 - (15 / 8) := by rw [h1, h2]
  let number_of_oranges := desired_profit / profit_per_orange
  have : number_of_oranges = 150 / (3 - 15 / 8) := by rw h3
  exact this
  sorry

end fruit_vendor_profit_l678_678751


namespace complex_in_fourth_quadrant_l678_678719

def quadOfComplexNumber (m : ℝ) (z : ℂ) : Prop :=
  z = 2 + (m - 1) * complex.I → m < 1 → z.imag < 0 ∧ z.re > 0

theorem complex_in_fourth_quadrant (m : ℝ) : quadOfComplexNumber m (2 + (m - 1) * complex.I) :=
by
  intros
  sorry

end complex_in_fourth_quadrant_l678_678719


namespace dihedral_angle_of_mirrors_l678_678708

theorem dihedral_angle_of_mirrors (α : ℝ) 
  (cond1 : α > 0 ∧ α < 90) 
  (cond2 : ∀ n : ℕ, n ≥ 1 → 
          (parallel_to_first_mirror α ∧
           reflect_off_second_mirror α ∧ 
           reflect_off_first_mirror α ∧
           reflect_off_second_mirror_again α ∧
           reflect_off_first_mirror_again α ∧
           reflect_off_second_mirror_third_time α))
          : α = 30 :=
sorry

end dihedral_angle_of_mirrors_l678_678708


namespace complex_number_coordinates_l678_678476

-- Define i as the imaginary unit
def i := Complex.I

-- State the theorem
theorem complex_number_coordinates : (i * (1 - i)).re = 1 ∧ (i * (1 - i)).im = 1 :=
by
  -- Proof would go here
  sorry

end complex_number_coordinates_l678_678476


namespace max_subset_card_l678_678234

theorem max_subset_card (n : ℕ) : 
  ∃ (B : Finset ℕ), B ⊆ Finset.range (n + 1) ∧ 
  (∀ (x y : ℕ), x ∈ B → y ∈ B → x ≠ y → ¬(x + y) % (x - y) = 0) ∧ 
  B.card = Int.ceil (n / 3.0) := sorry

end max_subset_card_l678_678234


namespace find_x_min_construction_cost_l678_678327

-- Define the conditions for Team A and Team B
def Team_A_Daily_Construction (x : ℕ) : ℕ := x + 300
def Team_A_Daily_Cost : ℕ := 3600
def Team_B_Daily_Construction (x : ℕ) : ℕ := x
def Team_B_Daily_Cost : ℕ := 2200

-- Condition: The number of days Team A needs to construct 1800m^2 is equal to the number of days Team B needs to construct 1200m^2
def construction_days (x : ℕ) : Prop := 
  1800 / (x + 300) = 1200 / x

-- Define the total days worked and the minimum construction area condition
def total_days : ℕ := 22
def min_construction_area : ℕ := 15000

-- Define the construction cost function given the number of days each team works
def construction_cost (m : ℕ) : ℕ := 
  3600 * m + 2200 * (total_days - m)

-- Main theorem: Prove that x = 600 satisfies the conditions
theorem find_x (x : ℕ) (h : x = 600) : construction_days x := by sorry

-- Second theorem: Prove that the minimum construction cost is 56800 yuan
theorem min_construction_cost (m : ℕ) (h : m ≥ 6) : construction_cost m = 56800 := by sorry

end find_x_min_construction_cost_l678_678327


namespace length_LM_in_triangle_l678_678221

theorem length_LM_in_triangle 
  (A B C K L M : Type*) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace K] [MetricSpace L] [MetricSpace M]
  (angle_A: Real) (angle_B: Real) (angle_C: Real)
  (AK: Real) (BL: Real) (MC: Real) (KL: Real) (KM: Real)
  (H1: angle_A = 90) (H2: angle_B = 30) (H3: angle_C = 60) 
  (H4: AK = 4) (H5: BL = 31) (H6: MC = 3) 
  (H7: KL = KM) : 
  (LM = 20) :=
sorry

end length_LM_in_triangle_l678_678221


namespace arrange_inequalities_l678_678123

theorem arrange_inequalities (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  (2 * a * b / (a + b)) < (sqrt (a * b)) ∧
  (sqrt (a * b)) < ((a + b) / 2) ∧
  ((a + b) / 2) < (sqrt ((a^2 + b^2) / 2)) :=
sorry

end arrange_inequalities_l678_678123


namespace ConvexPolygon_l678_678279

structure ConvexPolygon (n : ℕ) :=
  (vertices : Fin n → Point)
  (convex : ∀ (i j k : Fin n), ccw (vertices i) (vertices j) (vertices k))

def isRightAngle (x y z : Point) : Prop :=
  ∠ x y z = π / 2 -- assuming angle measure in radians

theorem ConvexPolygon.rectangle_of_four_right_angles {n : ℕ} (P : ConvexPolygon n)
  (h1 : n = 4) 
  (h2 : ∀(i : Fin 4), isRightAngle (P.vertices i) (P.vertices (i + 1) % 4) (P.vertices (i + 2) % 4)) :
  ∃ (a b c d : Point), P.vertices = ![a, b, c, d] ∧ isRectangle a b c d :=
sorry

end ConvexPolygon_l678_678279


namespace magnitude_of_complex_number_l678_678150

theorem magnitude_of_complex_number : (| (1 + I) / I |) = Real.sqrt 2 := 
  sorry

end magnitude_of_complex_number_l678_678150


namespace num_two_digit_powers_of_3_l678_678934

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678934


namespace sin_2_theta_third_quadrant_l678_678363

noncomputable def theta_in_third_quadrant (θ : ℝ) : Prop :=
  π < θ ∧ θ < 3 * π / 2

theorem sin_2_theta_third_quadrant (θ : ℝ) 
  (h1 : theta_in_third_quadrant θ)
  (h2 : (Real.sin θ) ^ 4 + (Real.cos θ) ^ 4 = 5 / 9) : 
  Real.sin (2 * θ) = 2 * Real.sqrt (2) / 3 :=
sorry

end sin_2_theta_third_quadrant_l678_678363


namespace polynomial_to_eliminate_fractions_l678_678723

theorem polynomial_to_eliminate_fractions (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  let eq := (2 / x = 1 / (x - 1))
  in ∀ y, y = x * (x - 1) → eq → 2 * (x - 1) = x := 
by
  intro x hx1 hx2 y hy eq
  sorry

end polynomial_to_eliminate_fractions_l678_678723


namespace general_term_and_arithmetic_minimize_Sn_l678_678926

-- Given the sequence $\{a_n\}$ with the formula for the sum of the first $n$ terms: $S_n = 2n^2 - 26n$
def Sn (n : ℕ) : ℕ := 2 * n^2 - 26 * n

-- Prove that the general formula for the sequence is $a_n = 4n - 28$
-- and that this sequence is an arithmetic sequence.
theorem general_term_and_arithmetic (n : ℕ) (h₂ : 2 ≤ n) :
  ∃ an, an = 4*n - 28 ∧ ∀ m >= 2, a_m - a_(m-1) = 4 :=
sorry

-- Prove that the value of n that minimizes Sn is 7
theorem minimize_Sn :
  ∀ n, Sn n = 2 * n^2 - 26 * n → ∀ m, Sn m ≥ Sn 7 :=
sorry

end general_term_and_arithmetic_minimize_Sn_l678_678926


namespace xy_value_l678_678547

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := 
by
  sorry

end xy_value_l678_678547


namespace factorial_fraction_eq_zero_l678_678413

theorem factorial_fraction_eq_zero :
  ((5 * (Nat.factorial 7) - 35 * (Nat.factorial 6)) / Nat.factorial 8 = 0) :=
by
  sorry

end factorial_fraction_eq_zero_l678_678413


namespace no_blue_2x2x2_cube_l678_678046

-- Definitions based on the problem's conditions
def small_cube := ℕ
def face := ℕ

-- Total number of small cubes
def total_small_cubes : small_cube := 27

-- Total number of faces for each small cube
def faces_per_cube (c: small_cube) : face := 6

-- Total number of painted faces
def painted_faces : face := 54

-- Define the condition of painted faces per small cube
def painted_faces_condition (c: small_cube) : Prop :=
  ∀ (i : small_cube), i < total_small_cubes → faces_per_cube i = 2

-- The main theorem stating the mathematically equivalent proof problem
theorem no_blue_2x2x2_cube (c : small_cube) :
  painted_faces_condition c → ¬ (∃ (b : small_cube), b = 8 ∧ all_faces_blue b) :=
by
  sorry

end no_blue_2x2x2_cube_l678_678046


namespace BMN_area_l678_678214

open EuclideanGeometry

theorem BMN_area (A B C D M N : Point)
  (hAB : dist A B = 2)
  (hBC : dist B C = 1)
  (h_rect : Rectangle A B C D)
  (hM_mid : midpoint A D M)
  (h_isosci : dist B N = dist M N) :
  area (Triangle.mk B M N) = (Real.sqrt 5) / 2 :=
sorry

end BMN_area_l678_678214


namespace determine_clothes_l678_678823

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l678_678823


namespace centroid_inside_triangle_l678_678207

open_locale classical

variables {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]

structure Triangle (A B C : Type*) :=
( vertices : fin 3 → A )

structure Median (T : Triangle (Type*)) :=
( start : A )
( end : A )
( property : ∃ (m : fin 3), start = T.vertices m ∧ end = midpoint ℝ (T.vertices m) (T.vertices ((m + 1) % 3)) )

/-- The centroid (G) of a triangle is the intersection of the medians. -/
noncomputable def centroid (T : Triangle (Type*)) : A :=
sorry -- define the computation of the centroid

theorem centroid_inside_triangle (T : Triangle (Type*)) : ∃ (G : A), (∀ (M : Median T), line_segment ℝ M.start G ∈ interior T) := 
sorry

end centroid_inside_triangle_l678_678207


namespace magnitude_sum_of_scaled_vector_l678_678899

variables (a b : EuclideanGeometry.Point2D) (angle_ab : Real)
variables (norm_b : Real)

def a_def : EuclideanGeometry.Point2D := (Real.sqrt 3, 1) -- \(\overrightarrow{a} = (\sqrt{3}, 1)\)
def b_norm : Real := 1 -- \(|\overrightarrow{b}| = 1\)
def alpha_60 : Real := 60 -- angle between \(\overrightarrow{a}\) and \(\overrightarrow{b}\) is \(60^\circ\)

theorem magnitude_sum_of_scaled_vector : 
  let a := a_def,
      b_mag := b_norm,
      alpha := angle_ab in
  (a = (Real.sqrt 3, 1)) -> 
  (|b| = 1) -> 
  (angle_ab = 60) ->
  |a + 2 \cdot b| = 2 \cdot Real.sqrt 3 :=
by
  intros a_eq b_mag_eq angle_ab_eq
  have ha := a_eq
  have hb_mag := b_mag_eq
  have halpha := angle_ab_eq
  sorry

end magnitude_sum_of_scaled_vector_l678_678899


namespace inequality_proof_l678_678481

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / a + (c + a) / b + (a + b) / c ≥ ((a ^ 2 + b ^ 2 + c ^ 2) * (a * b + b * c + c * a)) / (a * b * c * (a + b + c)) + 3 := 
by
  -- Adding 'sorry' to indicate the proof is omitted
  sorry

end inequality_proof_l678_678481


namespace sqrt_12_estimate_l678_678095

theorem sqrt_12_estimate : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_estimate_l678_678095


namespace fundraising_individual_contribution_l678_678343

theorem fundraising_individual_contribution :
  ∀ (total_amount participants individual_contribution : ℕ),
  total_amount = 1800 →
  participants = 10 →
  individual_contribution = total_amount / participants →
  individual_contribution = 180 :=
by
  intros total_amount participants individual_contribution
  intro h_total
  intro h_participants
  intro h_contribution
  rw [h_total, h_participants] at h_contribution
  exact h_contribution

#eval fundraising_individual_contribution 1800 10 (1800 / 10) (by rfl) (by rfl) (by rfl)

end fundraising_individual_contribution_l678_678343


namespace ella_code_combinations_l678_678853

-- Define the conditions of the problem
def digits_used_once := {1, 2, 3, 4, 5, 6}

def condition_1 (code : List ℕ) : Prop :=
  ∀ n, (n > 3 → ∃ m, m < 4 ∧ (List.indexOf n code + 1 = List.indexOf m code)) ∧
       (n < 4 → ∃ m, m > 3 ∧ (List.indexOf n code + 1 = List.indexOf m code))

def count_combinations : ℕ :=
  72 -- The correct answer

theorem ella_code_combinations : ∀ (code : List ℕ),
  (length code = 6 ∧
   code.to_set = digits_used_once ∧
   condition_1 code) →
   count_combinations = 72 :=
by
  sorry -- Proof to be filled in

end ella_code_combinations_l678_678853


namespace _l678_678319

noncomputable def largest_term_in_binomial_expansion :
  Term != ∀ (x : ℝ), ((\frac{1}{x} - 1)^5 : ℝ) → Term := by
-- Define what largest_term and the binomial theorem mean
sorry

end _l678_678319


namespace quadratic_inequality_l678_678897

variable (a b c A B C : ℝ)

theorem quadratic_inequality
  (h₁ : a ≠ 0)
  (h₂ : A ≠ 0)
  (h₃ : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) :
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end quadratic_inequality_l678_678897


namespace off_the_rack_suit_cost_l678_678595

theorem off_the_rack_suit_cost (x : ℝ)
  (h1 : ∀ y, y = 3 * x + 200)
  (h2 : ∀ y, x + y = 1400) :
  x = 300 :=
by
  sorry

end off_the_rack_suit_cost_l678_678595


namespace quadrilateral_antiparallelogram_l678_678560

-- Define a quadrilateral with vertices and diagonals
variables (A B C D P : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited P]

-- Define the angles' conditions and intersection point P
variable (angle_APD : Prop)
variable (angle_BPC : Prop)

-- Given conditions: quadrilateral ABCD with diagonals AC and BD intersecting at P
axiom intersection_condition : ∃ (A B C D P : Type), True

-- Given condition: the angles are equal
axiom equal_angles_condition : angle_APD = angle_BPC

-- Definition of an anti-parallelogram based on the given conditions
def is_antiparallelogram (A B C D P : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited P] : Prop :=
  intersection_condition ∧ equal_angles_condition

-- Main theorem: Proving that the quadrilateral is an antiparallelogram
theorem quadrilateral_antiparallelogram :
  is_antiparallelogram A B C D P :=
by
  sorry

end quadrilateral_antiparallelogram_l678_678560


namespace probability_multiple_of_90_l678_678565

def is_multiple_of_90 (n : ℕ) : Prop :=
  90 ∣ n

noncomputable def selected_set : Finset ℕ := {4, 6, 18, 28, 30, 36, 50}

theorem probability_multiple_of_90 :
  let pairs := (selected_set.image (λ x, selected_set.image (prod.mk x))) in
  let valid_pairs := pairs.filter (λ p, p.1 ≠ p.2 ∧ is_multiple_of_90 (p.1 * p.2)) in
  (valid_pairs.card : ℚ) / (pairs.card : ℚ) = 5 / 21 :=
by
  sorry

end probability_multiple_of_90_l678_678565


namespace money_sister_gave_l678_678637

theorem money_sister_gave (months_saved : ℕ) (savings_per_month : ℕ) (total_paid : ℕ) 
  (h1 : months_saved = 3) 
  (h2 : savings_per_month = 70) 
  (h3 : total_paid = 260) : 
  (total_paid - (months_saved * savings_per_month) = 50) :=
by {
  sorry
}

end money_sister_gave_l678_678637


namespace one_value_of_B_l678_678872

theorem one_value_of_B (B : ℕ) (h1 : B ∈ {1, 3, 5, 9}) (h2 : (17 + B) % 3 = 0) :
  ∃! B, B ∈ {1, 3, 5, 9} ∧ (17 + B) % 3 = 0 :=
by
  sorry

end one_value_of_B_l678_678872


namespace xn_difference_bound_l678_678252

open Nat

noncomputable def x (n : ℕ) : ℝ :=
  Real.sqrt (2 + (List.range n).foldr (fun i acc => Real.sqrt (i + 1 + acc)) 0)

theorem xn_difference_bound (n : ℕ) (h : 2 ≤ n) : x (n + 1) - x n < 1 / Nat.factorial n := by
  sorry

end xn_difference_bound_l678_678252


namespace hcf_of_two_numbers_proof_l678_678300

def hcf_lcm_problem (A B : ℕ) (H : ℕ) : Prop :=
  A = 345 ∧
  A B H ≠ 0 ∧
  H > 0 ∧
  (lcm A B) = H * 13 * 15

theorem hcf_of_two_numbers_proof :
  ∃ H, ∀ A B : ℕ, hcf_lcm_problem A B H → H = 15 := by
  sorry

end hcf_of_two_numbers_proof_l678_678300


namespace quadratic_real_roots_l678_678526
-- Import the necessary library

-- Define the given quadratic equation and the proof statement
theorem quadratic_real_roots (k : ℝ) : 
  let a := 1
  let b := 2 * (k - 1)
  let c := k^2 - 1
  let Δ := b^2 - 4 * a * c
  (Δ ≥ 0) ↔ (k ≤ 1) :=
by
  let a := 1
  let b := 2 * (k - 1)
  let c := k^2 - 1
  let Δ := b^2 - 4 * a * c
  have h : Δ = -8 * k + 8 := by sorry
  rw h
  exact ⟨λ h₀, sorry, λ h₀, sorry⟩

end quadratic_real_roots_l678_678526


namespace number_of_correct_statements_is_zero_l678_678053

theorem number_of_correct_statements_is_zero :
  (∃ s₁ s₂ s₃ s₄ : Prop, -- These represent the four statements
    (s₁ ↔ false) ∧      -- Statement (1) is false
    (s₂ ↔ false) ∧      -- Statement (2) is false
    (s₃ ↔ false) ∧      -- Statement (3) is false
    (s₄ ↔ false)) →     -- Statement (4) is false
  (count_correct : ℕ := 0) := sorry

end number_of_correct_statements_is_zero_l678_678053


namespace find_triples_l678_678448

def is_solution (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

theorem find_triples :
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ is_solution a b c :=
sorry

end find_triples_l678_678448


namespace colors_of_clothes_l678_678804

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l678_678804


namespace sum_of_first_n_terms_l678_678166

open BigOperators

def a (n : ℕ) : ℝ := n / 3^n

noncomputable def S (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), a k

theorem sum_of_first_n_terms (n : ℕ) :
  S n = (3 / 4) - (2 * n + 3) / (4 * 3^n) :=
by
  sorry

end sum_of_first_n_terms_l678_678166


namespace tan_sum_pi_four_l678_678142

theorem tan_sum_pi_four (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : cos (2 * α) = 3 / 5) : 
  tan (α + π / 4) = 3 := 
by 
  sorry

end tan_sum_pi_four_l678_678142


namespace tangent_line_intercept_l678_678422

open Real EuclideanGeometry

noncomputable theory

def circle1_center : ℝ × ℝ := (3, 5)
def circle1_radius : ℝ := 5
def circle2_center : ℝ × ℝ := (16, 12)
def circle2_radius : ℝ := 10

theorem tangent_line_intercept:
  ∃ m b : ℝ, m > 0 ∧
  ((∀ x ∈ ℝ, (x - circle1_center.1)^2 + (m*x + b - circle1_center.2)^2 = circle1_radius^2) ∧
   (∀ x ∈ ℝ, (x - circle2_center.1)^2 + (m*x + b - circle2_center.2)^2 = circle2_radius^2)) → 
  b = 123 / 48 :=
sorry

end tangent_line_intercept_l678_678422


namespace tan_alpha_beta_l678_678894

noncomputable def tan_alpha := -1 / 3
noncomputable def cos_beta := (Real.sqrt 5) / 5
noncomputable def beta := (1:ℝ) -- Dummy representation for being in first quadrant

theorem tan_alpha_beta (h1 : tan_alpha = -1 / 3) 
                       (h2 : cos_beta = (Real.sqrt 5) / 5) 
                       (h3 : 0 < beta ∧ beta < Real.pi / 2) : 
  Real.tan (α + β) = 1 := 
sorry

end tan_alpha_beta_l678_678894


namespace sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l678_678854

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

theorem sec_225_eq_neg_sqrt2 :
  sec (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

theorem csc_225_eq_neg_sqrt2 :
  csc (225 * Real.pi / 180) = -Real.sqrt 2 := sorry

end sec_225_eq_neg_sqrt2_csc_225_eq_neg_sqrt2_l678_678854


namespace simplify_expression_l678_678337

theorem simplify_expression : 
  (sqrt 8 * 2^(1/2) + (18 + 6 * 3) / 3 - 8^(3/2)) = 4 + 12 - 16 * sqrt 2 :=
by 
  sorry

end simplify_expression_l678_678337


namespace tan_triple_angle_l678_678551

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 1/3) : Real.tan (3 * θ) = 13/9 :=
by
  sorry

end tan_triple_angle_l678_678551


namespace domain_of_f_f_is_monotonically_increasing_l678_678557

open Real

noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 8) + 3

theorem domain_of_f :
  ∀ x, (x ≠ 5 * π / 16 + k * π / 2) := sorry

theorem f_is_monotonically_increasing :
  ∀ x, (π / 16 < x ∧ x < 3 * π / 16 → f x < f (x + ε)) := sorry

end domain_of_f_f_is_monotonically_increasing_l678_678557


namespace area_of_triangle_PF1F2_l678_678164

-- Definitions based on given conditions
def a : ℝ := 1
def b : ℝ := 2 * Real.sqrt 6
def c : ℝ := Real.sqrt (a^2 + b^2)
def F1F2 : ℝ := 2 * c
def PF1 : ℝ := (3 / 5) * F1F2
def PF2 : ℝ := PF1 + 2 * a

-- Main theorem statement
theorem area_of_triangle_PF1F2 : 
    let area := (1 / 2) * PF1 * PF2 in
    area = 24 := 
by
  sorry

end area_of_triangle_PF1F2_l678_678164


namespace graph_inequality_solution_l678_678169

noncomputable def solution_set : Set (Real × Real) := {
  p | let x := p.1
       let y := p.2
       (y^2 - (Real.arcsin (Real.sin x))^2) *
       (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
       (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0
}

theorem graph_inequality_solution
  (x y : ℝ) :
  (y^2 - (Real.arcsin (Real.sin x))^2) *
  (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
  (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0 ↔
  (x, y) ∈ solution_set :=
by
  sorry

end graph_inequality_solution_l678_678169


namespace geometric_sequence_ratios_l678_678881

theorem geometric_sequence_ratios {n : ℕ} {r : ℝ}
  (h1 : 85 = (1 - r^(2*n)) / (1 - r^2))
  (h2 : 170 = r * 85) :
  r = 2 ∧ 2*n = 8 :=
by
  sorry

end geometric_sequence_ratios_l678_678881


namespace clothes_color_proof_l678_678788

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l678_678788


namespace number_of_solutions_l678_678849

-- Define the complex function f(z)
def f (z : ℂ) : ℂ := -complex.I * complex.conj z

-- Statement of the problem
theorem number_of_solutions (z : ℂ) : 
  (|z| = 4 ∧ f z = z) → 
  (∃ z1 z2 : ℂ, z ≠ z1 ∧ z ≠ z2 ∧ 
  (|z1| = 4 ∧ f z1 = z1) ∧ 
  (|z2| = 4 ∧ f z2 = z2)) :=
sorry

end number_of_solutions_l678_678849


namespace prove_clothing_colors_l678_678815

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l678_678815


namespace optionD_is_deductive_l678_678339

-- Conditions related to the reasoning options
inductive ReasoningProcess where
  | optionA : ReasoningProcess
  | optionB : ReasoningProcess
  | optionC : ReasoningProcess
  | optionD : ReasoningProcess

-- Definitions matching the equivalent Lean problem
def isDeductiveReasoning (rp : ReasoningProcess) : Prop :=
  match rp with
  | ReasoningProcess.optionA => False
  | ReasoningProcess.optionB => False
  | ReasoningProcess.optionC => False
  | ReasoningProcess.optionD => True

-- The proposition we need to prove
theorem optionD_is_deductive :
  isDeductiveReasoning ReasoningProcess.optionD = True := by
  sorry

end optionD_is_deductive_l678_678339


namespace digit_1_2_9_in_n_or_3n_l678_678652

theorem digit_1_2_9_in_n_or_3n (n : ℕ) (hn : 0 < n) : 
  ∃ d ∈ [1, 2, 9], d ∈ list.map (fun (x : ℕ) => x.digit 10) (n.digits 10) ∨ 
  d ∈ list.map (fun (x : ℕ) => x.digit 10) ((3 * n).digits 10) :=
sorry

end digit_1_2_9_in_n_or_3n_l678_678652


namespace factorize_expression_l678_678097

theorem factorize_expression (a : ℝ) : a^3 - 4 * a^2 + 4 * a = a * (a - 2)^2 := 
by
  sorry

end factorize_expression_l678_678097


namespace members_in_two_activities_l678_678762

theorem members_in_two_activities (total_members : ℕ) (not_paint : ℕ) (not_sculpt : ℕ) (not_draw : ℕ) 
(members_conditions: total_members = 150 ∧ not_paint = 55 ∧ not_sculpt = 90 ∧ not_draw = 40) : 
let paint := total_members - not_paint,
    sculpt := total_members - not_sculpt,
    draw := total_members - not_draw,
    sum_activities := paint + sculpt + draw,
    exactly_two_activities := sum_activities - total_members
in exactly_two_activities = 115 := by {
  cases members_conditions with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  simp [paint, sculpt, draw, sum_activities, exactly_two_activities, h3, h5, h6, h1],
  sorry
}

end members_in_two_activities_l678_678762


namespace inverse_prop_function_through_point_l678_678505

theorem inverse_prop_function_through_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = k / x) → (f 1 = 2) → (f (-1) = -2) :=
by
  intros f h_inv_prop h_f1
  sorry

end inverse_prop_function_through_point_l678_678505


namespace part_a_part_b_part_c_l678_678511

section math_problems

variable {x y : ℝ}

-- Part (a): Proof that the only positive solution y = x implies x belongs to a specific set.
theorem part_a : 
  ∀ x : ℝ, 0 < x → (∀ y : ℝ, 0 < y → (y^x = x^y → y = x)) ↔ (0 < x ∧ x ≤ 1) ∨ (x = Real.exp 1) :=
sorry

-- Part (b): Number of non-trivial solutions y for given x.
theorem part_b : 
  ∀ x : ℝ, 0 < x ∧ x ≠ Real.exp 1 ∧ 1 < x → ∃! y : ℝ, 0 < y ∧ y ≠ x ∧ y^x = x^y :=
sorry

-- Part (c): Probability that the non-trivial solution y lies in (0, e) for a chosen x in (0, e).
theorem part_c :
  (∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → ∀ y : ℝ, 0 < y ∧ y^x = x^y → y ∉ (Set.Ioi 0 ∩ Set.Iio (Real.exp 1))) → 
  ∀ (x_dist : MeasureTheory.Measure (Set.Ioo 0 (Real.exp 1))), 
  MeasureTheory.Probability (λ x, ∃ y, 0 < y ∧ y ^ x = x ^ y ∧ y < Real.exp 1) (Set.Ioo 0 (Real.exp 1)) = 0 :=
sorry

end math_problems

end part_a_part_b_part_c_l678_678511


namespace perimeter_of_trapezoid_l678_678218

-- Define the conditions as given in the initial problem statement.
def EF := 4
def GH := 10
def height := 5

-- Translate the conditions into mathematical statements.
noncomputable def EF_2 := real.sqrt (height^2 + ((GH - EF) / 2)^2)

-- Defining the final proof statement.
theorem perimeter_of_trapezoid :
  let perimeter := EF + GH + 2 * EF_2 in
  perimeter = 14 + 2 * real.sqrt 34 :=
by
  sorry

end perimeter_of_trapezoid_l678_678218


namespace train_length_l678_678773

theorem train_length (L : ℝ) : (L + 200) / 15 = (L + 300) / 20 → L = 100 :=
by
  intro h
  -- Skipping the proof steps
  sorry

end train_length_l678_678773


namespace math_problem_l678_678065

noncomputable def problem_expression : ℝ :=
  |(-1)| - 2 * Real.sin (Float.pi / 6) + (Real.pi - 3.14)^0 + (1/2)^(-2)

theorem math_problem :
  problem_expression = 5 :=
by
  sorry

end math_problem_l678_678065


namespace no_real_roots_of_sqrt_eq_l678_678084

theorem no_real_roots_of_sqrt_eq :
  ¬ ∃ (x : ℝ), (sqrt (x + 9) + sqrt (x - 2) = 3) :=
by
  sorry

end no_real_roots_of_sqrt_eq_l678_678084


namespace product_real_parts_l678_678435

theorem product_real_parts (x : ℂ) (h : x ^ 4 + 2 * x ^ 2 + 1 = 0) :
  (x = Complex.i ∨ x = -Complex.i) → 
  (Complex.re x) * (Complex.re (-x)) = 0 :=
by 
  sorry

end product_real_parts_l678_678435


namespace smallest_lambda_exists_l678_678484

variable {n : ℕ} (hn : n ≥ 2)

theorem smallest_lambda_exists (a b : Fin n → ℝ) 
    (ha : ∀ i, 0 < a i) 
    (hb : ∀ i, 0 ≤ b i ∧ b i ≤ 1 / 2) 
    (sum_a : ∑ i, a i = 1) 
    (sum_b : ∑ i, b i = 1) :
  (∏ i, a i) ≤ (1 / 2 * (1 / (n - 1))^(n - 1)) * (∑ i, (a i) * (b i)) :=
sorry

end smallest_lambda_exists_l678_678484


namespace determine_clothes_l678_678818

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l678_678818


namespace colors_of_clothes_l678_678805

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l678_678805


namespace solution_set_of_f_neg_2x_l678_678154

def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_of_f_neg_2x (a b : ℝ) (hf_sol : ∀ x : ℝ, (a * x - 1) * (x + b) > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x : ℝ, f a b (-2 * x) < 0 ↔ (x < -3/2 ∨ x > 1/2) :=
by
  sorry

end solution_set_of_f_neg_2x_l678_678154


namespace constant_term_expansion_l678_678197

theorem constant_term_expansion (n : ℕ) (h : (2 : ℝ)^n = 32) :
  ∑ k : ℕ in finset.range (n + 1), ((n.choose k : ℕ) * x^(2 * (n - k) - 3 * k)) k = 10 :=
sorry

end constant_term_expansion_l678_678197


namespace relatively_prime_days_in_november_l678_678761

theorem relatively_prime_days_in_november : 
  ∀ {d : ℕ}, 1 ≤ d ∧ d ≤ 30 → (finset.card (finset.filter (λ x, Nat.gcd x 11 = 1) (finset.range 31)) = 28) :=
by
  sorry

end relatively_prime_days_in_november_l678_678761


namespace identify_clothing_l678_678779

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l678_678779


namespace surface_area_of_sphere_l678_678036

noncomputable def radius_of_sphere (a : ℝ) : ℝ :=
  let b := a * (Real.sqrt 3) / 2 in Real.sqrt ((a / 2)^2 + b^2)

theorem surface_area_of_sphere (a : ℝ) (h : a = 3) : 
  4 * Real.pi * (radius_of_sphere a) ^ 2 = 21 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l678_678036


namespace cannot_tile_with_pentagon_l678_678724

theorem cannot_tile_with_pentagon :
  ∀ (n : ℕ), n ∈ {3, 4, 5, 6} →
  let interior_angle (n : ℕ) := 180 - 360 / n in
  ∀ (angles : ∀ (n : ℕ), List ℕ),
  angles 3 = [60] ∧ angles 4 = [90] ∧ angles 5 = [108] ∧ angles 6 = [120] →
  (∃ (k : ℕ), 360 % angles n = 0) ↔ n ≠ 5 :=
by {
  -- Skipping the actual proof
  sorry
}

end cannot_tile_with_pentagon_l678_678724


namespace main_theorem_l678_678467

noncomputable def α := Real
noncomputable def β := Real
noncomputable def γ := Real
noncomputable def δ := Real

axiom positive_reals (α β γ δ : Real) (h1 : α > 0) (h2 : β > 0) (h3 : γ > 0) (h4 : δ > 0)
axiom positive_integer (n : ℕ) (h : n > 0)

axiom floor_multiplication_eq (α β γ δ : Real) (n : ℕ) :
  (⌊α * n⌋ * ⌊β * n⌋ = ⌊γ * n⌋ * ⌊δ * n⌋)

axiom distinct_sets (α β γ δ : Real) :
  ({α, β} ≠ {γ, δ})

theorem main_theorem 
  (α β γ δ : Real)
  (h1 : α > 0) (h2 : β > 0) (h3 : γ > 0) (h4 : δ > 0)
  (n : ℕ) (h5 : n > 0)
  (h6 : ⌊α * n⌋ * ⌊β * n⌋ = ⌊γ * n⌋ * ⌊δ * n⌋)
  (h7 : {α, β} ≠ {γ, δ}) :
  α * β = γ * δ ∧ (∃ (a b c d : ℕ), α = a ∧ β = b ∧ γ = c ∧ δ = d) := 
sorry

end main_theorem_l678_678467


namespace shape_of_phi_constant_l678_678459

/-
  In spherical coordinates, given a constant angle φ = c,
  prove that the shape described is a cone with an opening angle c,
  vertex at the origin, and axis along the z-axis.
-/

variable {ρ θ : ℝ}
variable {c : ℝ}

theorem shape_of_phi_constant
  (h : ∀ ρ θ, 0 ≤ c ∧ c ≤ π → (ρ, θ, c) represents a point in spherical coordinates) :
  ∃ R : ℝ, ∀ ρ, θ, (ρ = R * cos c ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 0 ≤ c ∧ c ≤ π) ↔
    -- Shape is a cone with vertex at (0,0,0) and axis along z-axis
    (ρ * sin c, θ, c) is a point on cone.
:= sorry

end shape_of_phi_constant_l678_678459


namespace correct_choices_l678_678631

-- Define the function f(x)
def f (a b c x : ℝ) := a^x + b^x - c^x

-- The main theorem statement incorporating all three propositions
theorem correct_choices (a b c x : ℝ) (h1 : c > a > 0) (h2 : c > b > 0) 
  (h3 : a + b > c) (h4 : (a,b,c) = (a, b, c)) :
  (∀ (x : ℝ), x < 1 → f a b c x > 0) ∧ (∃ (x : ℝ), ¬ (a^x + b^x > c^x ∧ a^x + c^x > b^x ∧ b^x + c^x > a^x)) ∧ 
  (is_obtuse a b c → ∃ (x : ℝ), 1 < x ∧ x < 2 ∧ f a b c x = 0) :=
sorry

-- Helper definition to check if the triangle with sides a, b, c is obtuse
def is_obtuse (a b c : ℝ) : Prop := a^2 + b^2 - c^2 < 0

end correct_choices_l678_678631


namespace option_b_option_c_option_d_l678_678244

-- Geometric sequence with conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q

-- Given conditions
axiom a1_a9_eq_16 : a 1 * a 9 = 16

-- Proof for Option B
theorem option_b (h : a 1 = 1) : q = real.sqrt 2 ∨ q = -real.sqrt 2 := sorry

-- Proof for Option C
theorem option_c : real.log 2 (abs (∏ i in finset.range 9, a (i + 1))) = 18 := sorry

-- Proof for Option D
theorem option_d : (a 3) ^ 2 + (a 7) ^ 2 ≥ 32 := sorry

end option_b_option_c_option_d_l678_678244


namespace range_of_positive_integers_in_consecutive_list_l678_678259

theorem range_of_positive_integers_in_consecutive_list (K : List ℤ) (h1 : K.length = 12) (h2 : K.head = -3) :
  ∃ R, (R = 7) ∧ (∃ P, (∀ x ∈ K, x > 0 ↔ x ∈ P) ∧ ∃ a b, P = List.range' a b ∧ b = R + 1) := 
sorry

end range_of_positive_integers_in_consecutive_list_l678_678259


namespace cost_price_is_correct_l678_678375

variable (C : ℝ) -- Cost price of one computer table

axiom condition1 : ∀ (n : ℕ), n = 3 -- The customer buys 3 computer tables
axiom condition2 : ∀ (P : ℝ), P = 1.80 * C -- The shop charges 80% more than the cost price
axiom condition3 : ∀ (d : ℝ), d = 0.90 -- The customer receives a 10% discount on the total purchase
axiom condition4 : ∀ (f : ℝ), f = 7650 -- The final price paid for the computer tables is Rs. 7,650

theorem cost_price_is_correct (h1 : condition1 3) (h2 : condition2 (1.80 * C)) 
                               (h3 : condition3 0.90) (h4 : condition4 7650) :
  C = 7650 / (3 * 1.80 * 0.90) :=
by
  sorry

end cost_price_is_correct_l678_678375


namespace xy_sum_equal_two_or_minus_two_l678_678549

/-- 
Given the conditions |x| = 3, |y| = 5, and xy < 0, prove that x + y = 2 or x + y = -2. 
-/
theorem xy_sum_equal_two_or_minus_two (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) : x + y = 2 ∨ x + y = -2 := 
  sorry

end xy_sum_equal_two_or_minus_two_l678_678549


namespace gummy_vitamins_cost_l678_678603

def bottle_discounted_price (P D_s : ℝ) : ℝ :=
  P * (1 - D_s)

def normal_purchase_discounted_price (discounted_price D_n : ℝ) : ℝ :=
  discounted_price * (1 - D_n)

def bulk_purchase_discounted_price (discounted_price D_b : ℝ) : ℝ :=
  discounted_price * (1 - D_b)

def total_cost (normal_bottles bulk_bottles normal_price bulk_price : ℝ) : ℝ :=
  (normal_bottles * normal_price) + (bulk_bottles * bulk_price)

def apply_coupons (total_cost N_c C : ℝ) : ℝ :=
  total_cost - (N_c * C)

theorem gummy_vitamins_cost 
  (P N_c C D_s D_n D_b : ℝ) 
  (normal_bottles bulk_bottles : ℕ) :
  bottle_discounted_price P D_s = 12.45 → 
  normal_purchase_discounted_price 12.45 D_n = 11.33 → 
  bulk_purchase_discounted_price 12.45 D_b = 11.83 → 
  total_cost 4 3 11.33 11.83 = 80.81 → 
  apply_coupons 80.81 N_c C = 70.81 :=
sorry

end gummy_vitamins_cost_l678_678603


namespace percentage_decrease_2008_l678_678119

variable (P : ℝ) (X : ℝ)
variable (h2007 : P * 1.20 = P * (1 + 0.20))
variable (h2008 : P * 1.20 * (1 - X) = P * 1.20 - P * 1.20 * X)
variable (h2009 : P * 1.20 * (1 - X) * 1.35 = P * 1.215)

theorem percentage_decrease_2008 : X = 0.25 :=
by
  have h1 : 1.20 * (1 - X) * 1.35 = 1.215 := by rwa [mul_sub, mul_comm (1.20 * P) (1 - X)] at h2009
  have h2 : 1 - X = 0.75 := ... -- continuation of proof
  have h3 : X = 0.25 := ...
  sorry -- This line skips the proof steps, focusing on the statement structure only.

end percentage_decrease_2008_l678_678119


namespace polynomial_positive_values_l678_678100

theorem polynomial_positive_values :
  ∀ x y : ℝ, ∃ t : ℝ, t > 0 ∧ (t = x^2 + (x * y + 1)^2) :=
begin
  intros x y,
  sorry
end

end polynomial_positive_values_l678_678100


namespace one_millionth_digit_3_div_41_l678_678455

theorem one_millionth_digit_3_div_41 : 
  let repeating_sequence := "073170731707317"
  1_000_000 % repeating_sequence.length = 10 ∧
  repeating_sequence.get_digit_at(10) = 7 :=
by 
  sorry -- Proof is omitted.

end one_millionth_digit_3_div_41_l678_678455


namespace correct_table_count_l678_678056

def stools_per_table : ℕ := 8
def chairs_per_table : ℕ := 2
def legs_per_stool : ℕ := 3
def legs_per_chair : ℕ := 4
def legs_per_table : ℕ := 4
def total_legs : ℕ := 656

theorem correct_table_count (t : ℕ) :
  stools_per_table * legs_per_stool * t +
  chairs_per_table * legs_per_chair * t +
  legs_per_table * t = total_legs → t = 18 :=
by
  intros h
  sorry

end correct_table_count_l678_678056


namespace fill_question_mark_l678_678741

def sudoku_grid : Type := 
  List (List (Option ℕ))

def initial_grid : sudoku_grid := 
  [ [some 3, none, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ]

def valid_sudoku (grid : sudoku_grid) : Prop :=
  -- Ensure the grid is a valid 4x4 Sudoku grid
  -- Adding necessary constraints for rows, columns and 2x2 subgrids.
  sorry

def solve_sudoku (grid : sudoku_grid) : sudoku_grid :=
  -- Function that solves the Sudoku (not implemented for this proof statement)
  sorry

theorem fill_question_mark : solve_sudoku initial_grid = 
  [ [some 3, some 2, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ] :=
  sorry

end fill_question_mark_l678_678741


namespace circle_p_q_s_sum_l678_678615

open Real

-- Define the circle equation and center/radius definitions
def circle_eq (x y : ℝ) : Prop := x^2 - 4 * y - 16 = -y^2 + 26 * x + 36

structure CircleCenterRadius :=
  (p q s : ℝ)
  (center : (p, q))
  (radius : s)

noncomputable def circle_definition : CircleCenterRadius :=
  { p := 13, q := 2, s := 15, center := (13, 2), radius := 15 }

-- Prove that the sum p + q + s is equal to 30
theorem circle_p_q_s_sum : 
  (circle_definition.p + circle_definition.q + circle_definition.s) = 30 :=
by
  sorry

end circle_p_q_s_sum_l678_678615


namespace triangle_number_arrangement_l678_678357

noncomputable def numbers := [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

theorem triangle_number_arrangement : 
  ∃ (f : Fin 9 → Fin 9), 
    (numbers[f 0] + numbers[f 1] + numbers[f 2] = 
     numbers[f 3] + numbers[f 4] + numbers[f 5] ∧ 
     numbers[f 3] + numbers[f 4] + numbers[f 5] = 
     numbers[f 6] + numbers[f 7] + numbers[f 8]) :=
sorry

end triangle_number_arrangement_l678_678357


namespace sum_of_distinct_elements_not_square_l678_678861

open Set

noncomputable def setS : Set ℕ := { n | ∃ k : ℕ, n = 2^(2*k+1) }

theorem sum_of_distinct_elements_not_square (s : Finset ℕ) (hs: ∀ x ∈ s, x ∈ setS) :
  ¬∃ k : ℕ, s.sum id = k^2 :=
sorry

end sum_of_distinct_elements_not_square_l678_678861


namespace units_digit_of_odd_pos_int_non_div_by_3_btw_20_and_200_l678_678718

theorem units_digit_of_odd_pos_int_non_div_by_3_btw_20_and_200 :
  let product := ∏ x in Finset.filter (λ x, x % 2 = 1 ∧ x % 3 ≠ 0) (Finset.range 181).map (λ n, n + 20)
  in (product % 10) = 5 :=
by
  sorry

end units_digit_of_odd_pos_int_non_div_by_3_btw_20_and_200_l678_678718


namespace fourth_term_arithmetic_sequence_l678_678561

theorem fourth_term_arithmetic_sequence (a d : ℝ) (h : 2 * a + 2 * d = 12) : a + d = 6 := 
by
  sorry

end fourth_term_arithmetic_sequence_l678_678561


namespace present_cost_after_two_years_l678_678291

-- Defining variables and constants
def initial_cost : ℝ := 75
def inflation_rate : ℝ := 0.05
def first_year_increase1 : ℝ := 0.20
def first_year_decrease1 : ℝ := 0.20
def second_year_increase2 : ℝ := 0.30
def second_year_decrease2 : ℝ := 0.25

theorem present_cost_after_two_years : presents_cost = 77.40 :=
by
  let adjusted_initial_cost := initial_cost + (initial_cost * inflation_rate)
  let increased_cost_year1 := adjusted_initial_cost + (adjusted_initial_cost * first_year_increase1)
  let decreased_cost_year1 := increased_cost_year1 - (increased_cost_year1 * first_year_decrease1)
  let adjusted_cost_year1 := decreased_cost_year1 + (decreased_cost_year1 * inflation_rate)
  let increased_cost_year2 := adjusted_cost_year1 + (adjusted_cost_year1 * second_year_increase2)
  let decreased_cost_year2 := increased_cost_year2 - (increased_cost_year2 * second_year_decrease2)
  let presents_cost := decreased_cost_year2
  have h := (presents_cost : ℝ)
  have h := presents_cost
  sorry

end present_cost_after_two_years_l678_678291


namespace probability_of_B_l678_678555

variables {Ω : Type} [ProbabilitySpace Ω]

-- Definitions and Conditions
variables (A B : Event Ω) (p : Probability Ω)
variable h₁ : p(A) = 3 / 4
variable h₂ : p(A ∩ B) = 3 / 8
variable h₃ : p(∅ᶜ ∩ (A ∪ B)ᶜ) = 1 / 8

-- Goal
theorem probability_of_B (h₁ : p(A) = 3 / 4) (h₂ : p(A ∩ B) = 3 / 8) (h₃ : p(∅ᶜ ∩ (A ∪ B)ᶜ) = 1 / 8) : 
  p(B) = 1 / 2 :=
sorry

end probability_of_B_l678_678555


namespace worker_savings_fraction_l678_678728

theorem worker_savings_fraction (P : ℝ) (F : ℝ) (h1 : P > 0) (h2 : 12 * F * P = 5 * (1 - F) * P) : F = 5 / 17 :=
by
  sorry

end worker_savings_fraction_l678_678728


namespace units_digit_pow_3_cycle_l678_678639

theorem units_digit_pow_3_cycle (n : ℕ) :
  (∀ (k : ℕ), (3 ^ k) % 10 = if k % 4 = 0 then 1 else if k % 4 = 1 then 3 else if k % 4 = 2 then 9 else 7) →
  (3 ^ 2012) % 10 = 1 :=
begin
  intro pattern,
  have step1 : 2012 % 4 = 0, by norm_num,
  rw [step1, pattern 2012],
  norm_num,
  -- lean commands equivalent to the steps would be added here.
  sorry
end

end units_digit_pow_3_cycle_l678_678639


namespace value_of_sheep_l678_678608

theorem value_of_sheep (months_worked total_monthly_gold pay_in_months remaining_gold monthly_gold : ℕ) (total_sheep : ℕ) : 
  (total_monthly_gold * total_sheep + remaining_gold) / pay_in_months <= 16 :=
by
  have h: months_worked = 7,
  have total_monthly_gold = 20,
  have total_sheep = 1,
  have pay_in_months = 5,
  have remaining_gold = 5,
  have monthly_gold = 3,
  sorry

end value_of_sheep_l678_678608


namespace lighthouse_distance_correct_l678_678766

-- The ship's velocity
def velocity : ℝ := 22 * Real.sqrt 6

-- Time traveled by the ship in hours
def time_traveled : ℝ := 1.5

-- Distance traveled by the ship from point A to point B
def distance_AB : ℝ := velocity * time_traveled

-- Angle at point A (in degrees)
def angle_A : ℝ := 45

-- Angle at point B (in degrees)
def angle_B : ℝ := 15

-- Angle at lighthouse S
def angle_S : ℝ := 180 - (angle_A + angle_B)

-- Function to calculate distance BS using Law of Sines
noncomputable def distance_BS : ℝ :=
  let sin_A := Real.sin (Real.pi * angle_A / 180)
  let sin_S := Real.sin (Real.pi * angle_S / 180)
  (distance_AB * sin_A) / sin_S

-- The lean statement to prove the distance between lighthouse S and point B is 66 km
theorem lighthouse_distance_correct :
  distance_BS = 66 := by
  sorry

end lighthouse_distance_correct_l678_678766


namespace mutated_frog_percentage_l678_678111

theorem mutated_frog_percentage 
  (extra_legs : ℕ) 
  (two_heads : ℕ) 
  (bright_red : ℕ) 
  (normal_frogs : ℕ) 
  (h_extra_legs : extra_legs = 5) 
  (h_two_heads : two_heads = 2) 
  (h_bright_red : bright_red = 2) 
  (h_normal_frogs : normal_frogs = 18) 
  : ((extra_legs + two_heads + bright_red) * 100 / (extra_legs + two_heads + bright_red + normal_frogs)).round = 33 := 
by
  sorry

end mutated_frog_percentage_l678_678111


namespace sum_of_reciprocals_of_roots_l678_678458

theorem sum_of_reciprocals_of_roots :
  (∀ (r1 r2 : ℝ), (r1 + r2 = 17) ∧ (r1 * r2 = 8) → (1 / r1 + 1 / r2 = 17 / 8)) :=
begin
  -- Given that r1 and r2 are the roots of the equation x^2 - 17x + 8 = 0
  -- and by Vieta's formulas, we have r1 + r2 = 17 and r1 * r2 = 8,
  -- we need to prove (1 / r1 + 1 / r2 = 17 / 8).
  sorry
end

end sum_of_reciprocals_of_roots_l678_678458


namespace three_Y_five_l678_678194

-- Define the operation Y
def Y (a b : ℕ) : ℕ := 3 * b + 8 * a - a^2

-- State the theorem to prove the value of 3 Y 5
theorem three_Y_five : Y 3 5 = 30 :=
by
  sorry

end three_Y_five_l678_678194


namespace sum_of_x_intercepts_l678_678273

theorem sum_of_x_intercepts (c e : ℕ) (h1 : 2 * c = e) : 
    (((-1 / 2) + (-1) + (-3 / 2) + (-2) + (-5 / 2)) = -9.5) :=
by
  -- Assume all the necessary conditions
  have h2 : 2 * 1 = 2 := by norm_num,
  have h3 : 2 * 2 = 4 := by norm_num,
  have h4 : 2 * 3 = 6 := by norm_num,
  have h5 : 2 * 4 = 8 := by norm_num,
  have h6 : 2 * 5 = 10 := by norm_num,
  -- List all the possible pairs
  have pairs : list (ℕ × ℕ) := [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)],
  -- Evaluate the x-values
  have x_values : list ℚ := [-1 / 2, -1, -3 / 2, -2, -5 / 2],
  -- Calculate the sum of x-values
  have sum_x_values : ℚ := list.sum x_values,
  -- Final assertion
  exact congr_arg coe (by norm_num : -9.5 = -9.5)

end sum_of_x_intercepts_l678_678273


namespace count_two_digit_powers_of_three_l678_678976

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678976


namespace david_bike_distance_l678_678432

noncomputable def david_time_hours : ℝ := 2 + 1 / 3
noncomputable def david_speed_mph : ℝ := 6.998571428571427
noncomputable def david_distance : ℝ := 16.33

theorem david_bike_distance :
  david_speed_mph * david_time_hours = david_distance :=
by
  sorry

end david_bike_distance_l678_678432


namespace original_employees_l678_678347

theorem original_employees (x : ℝ) (h : 0.86 * x = 195) : x ≈ 227 :=
by sorry

end original_employees_l678_678347


namespace number_of_correct_propositions_l678_678052

noncomputable def unit_vector : Type := sorry
noncomputable def parallel_vectors_with_equal_magnitudes_are_equal_vectors : Type := sorry
noncomputable def compare_vectors_in_same_direction (a b : Type) : Prop := sorry
noncomputable def vectors_starting_ending_points_coincide (a b : Type) : Prop := sorry
noncomputable def transitive_parallel_vectors (a b c : Type) : Prop := sorry

theorem number_of_correct_propositions : 
  (¬ unit_vector) ∧
  (¬ parallel_vectors_with_equal_magnitudes_are_equal_vectors) ∧
  (¬ compare_vectors_in_same_direction a b) ∧
  (¬ vectors_starting_ending_points_coincide a b) ∧
  (¬ transitive_parallel_vectors a b c) -> 
  (0 = number_of_true_propositions :=
  begin
    sorry
  end)

end number_of_correct_propositions_l678_678052


namespace inequality_proof_l678_678630

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h : 1/a + 1/b + 1/c = a + b + c) :
  1/(2*a + b + c)^2 + 1/(2*b + c + a)^2 + 1/(2*c + a + b)^2 ≤ 3/16 :=
by
  sorry

end inequality_proof_l678_678630


namespace range_of_f_range_of_k_l678_678874

section MathProblem

def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6) + 2

theorem range_of_f : 
  ∀ x, (0 ≤ x ∧ x ≤ π / 2) → (1 ≤ f x ∧ f x ≤ 4) :=
by
  sorry

theorem range_of_k :
  ∀ x α, (0 ≤ x ∧ x ≤ π / 2) ∧ (π / 12 ≤ α ∧ α ≤ π / 3) →
    ∀ k, k * sqrt (1 + sin (2 * α)) - sin (2 * α) ≤ f x + 1 → 
    k ≤ (sqrt 6 / 2 + (1 / (sqrt 6 / 2))) :=
by
  sorry

end MathProblem

end range_of_f_range_of_k_l678_678874


namespace least_possible_value_of_expression_l678_678362

theorem least_possible_value_of_expression : 
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    {a, b, c} = {2, 3, 5} ∧ 
    (a + b : ℚ) / c / 2 = 1 / 2 :=
by
  sorry

end least_possible_value_of_expression_l678_678362


namespace even_factors_count_l678_678180

def is_factor (n a b c d : ℕ) := (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) ∣ n

theorem even_factors_count : 
  let n := 2 ^ 3 * 3 ^ 2 * 5 ^ 1 * 7 ^ 3
  in ∑ a in finset.range 4, ∑ b in finset.range 3, ∑ c in finset.range 2, ∑ d in finset.range 4, 
    ite (a > 0) 1 0 = 72 := 
by
  let n := 2^3 * 3^2 * 5^1 * 7^3
  have h_a_choices : finset.card (finset.filter (λ a, 0 < a) (finset.range 4)) = 3 :=
    by sorry
  have h_b_choices : finset.card (finset.range 3) = 3 :=
    by sorry
  have h_c_choices : finset.card (finset.range 2) = 2 :=
    by sorry
  have h_d_choices : finset.card (finset.range 4) = 4 :=
    by sorry
  have total_factors : 
    (finset.card (finset.filter (λ a, 0 < a) (finset.range 4))) * 
    (finset.card (finset.range 3)) * 
    (finset.card (finset.range 2)) * 
    (finset.card (finset.range 4)) = 72 :=
  by simp [h_a_choices, h_b_choices, h_c_choices, h_d_choices]; exact dec_trivial
  exact total_factors

end even_factors_count_l678_678180


namespace perp_bisector_of_segment_l678_678299

theorem perp_bisector_of_segment:
  ∀ (c : ℝ), 
  (∀ (x y : ℝ), ((x, y) = (2, 4) ∨ (x, y) = (6, 8)) → (x + y = c)) → 
  (∃ c = 10, ∀ (m : ℝ × ℝ), m = (4, 6) → (x, y) = m) → (4 + 6 = c) → c := 
by
  sorry

end perp_bisector_of_segment_l678_678299


namespace jerry_can_throw_things_l678_678264

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25
def office_points_threshold : ℕ := 100
def interruptions : ℕ := 2
def insults : ℕ := 4

theorem jerry_can_throw_things : 
  (office_points_threshold - (points_for_interrupting * interruptions + points_for_insulting * insults)) / points_for_throwing = 2 :=
by 
  sorry

end jerry_can_throw_things_l678_678264


namespace solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l678_678161

-- Define the function f(x) and g(x)
def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Define the inequality problem when a = 2
theorem solution_set_for_f_when_a_2 : 
  { x : ℝ | f x 2 ≤ 6 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

-- Prove the range of values for a when f(x) + g(x) ≥ 3
theorem range_of_a_for_f_plus_g_ge_3 : 
  ∀ a : ℝ, (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a :=
by
  sorry

end solution_set_for_f_when_a_2_range_of_a_for_f_plus_g_ge_3_l678_678161


namespace mutant_frog_percentage_proof_l678_678115

/-- Number of frogs with extra legs -/
def frogs_with_extra_legs := 5

/-- Number of frogs with 2 heads -/
def frogs_with_two_heads := 2

/-- Number of frogs that are bright red -/
def frogs_bright_red := 2

/-- Number of normal frogs -/
def normal_frogs := 18

/-- Total number of mutant frogs -/
def total_mutant_frogs := frogs_with_extra_legs + frogs_with_two_heads + frogs_bright_red

/-- Total number of frogs -/
def total_frogs := total_mutant_frogs + normal_frogs

/-- Calculate the percentage of mutant frogs rounded to the nearest integer -/
def mutant_frog_percentage : ℕ := (total_mutant_frogs * 100 / total_frogs).toNat

theorem mutant_frog_percentage_proof:
  mutant_frog_percentage = 33 := 
  by 
    -- Proof skipped
    sorry

end mutant_frog_percentage_proof_l678_678115


namespace min_distance_to_P_l678_678575

-- Defining the regular tetrahedron with edge length 1
structure RegularTetrahedron (V : Type*) :=
(O A B C : V)
(edge_length : ℝ)
(edge_length_eq_one : edge_length = 1)

-- Defining the point P as a vector combination
def point_P {V : Type*} [add_comm_group V] [vector_space ℝ V] (tetra : RegularTetrahedron V) 
  (x y z : ℝ) (h : x + y + z = 1) : V :=
x • (tetra.A - tetra.O) + y • (tetra.B - tetra.O) + z • (tetra.C - tetra.O) + tetra.O

-- Stating the minimum distance condition
theorem min_distance_to_P (V : Type*) [inner_product_space ℝ V] 
  (tetra : RegularTetrahedron V) (x y z : ℝ) (h : x + y + z = 1) : 
  ∃ P : V, (point_P tetra x y z h = P) 
  ∧ (dist P tetra.O = (Real.sqrt 6) / 3) :=
sorry

end min_distance_to_P_l678_678575


namespace jerry_can_throw_things_l678_678265

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25
def office_points_threshold : ℕ := 100
def interruptions : ℕ := 2
def insults : ℕ := 4

theorem jerry_can_throw_things : 
  (office_points_threshold - (points_for_interrupting * interruptions + points_for_insulting * insults)) / points_for_throwing = 2 :=
by 
  sorry

end jerry_can_throw_things_l678_678265


namespace stratified_sampling_l678_678371

theorem stratified_sampling (total_male total_female sample_size : ℕ) (prob : ℚ)
    (h_total_male : total_male = 200)
    (h_total_female : total_female = 300)
    (h_sample_size : sample_size = 50)
    (h_prob : prob = sample_size / (total_male + total_female)) :
    (total_male * prob = 20) ∧ (total_female * prob = 30) :=
begin
    -- Proof skipped
    sorry
end

end stratified_sampling_l678_678371


namespace part1_part2_l678_678479

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^4 - 4 * x^3 + (3 + m) * x^2 - 12 * x + 12

theorem part1 (m : ℤ) : 
  (∀ x : ℝ, f x m - f (1 - x) m + 4 * x^3 = 0) ↔ (m = 8 ∨ m = 12) := 
sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x m ≥ 0) ↔ (4 ≤ m) := 
sorry

end part1_part2_l678_678479


namespace number_of_proper_subsets_of_S_l678_678690

-- Define the set based on the given conditions
def S := {x : ℕ | -1 ≤ log (1/x) 10 ∧ log (1/x) 10 < -1/2}

-- Define the conjecture that the number of proper subsets of S is 2^90 - 1
theorem number_of_proper_subsets_of_S : (∃ S : set ℕ, S = {x : ℕ | -1 ≤ log (1/x) 10 ∧ log (1/x) 10 < -1/2}) →
  ∃ S : set ℕ, card S = 90 ∧ 2^90 - 1 = 2^card S - 1 :=
by
  sorry

end number_of_proper_subsets_of_S_l678_678690


namespace proof_problem_l678_678928

variables (a b : ℝ^3) -- Define vector variables
-- Define the conditions
def unit_vector (v : ℝ^3) : Prop := (v.dot v) = 1
def given_condition : Prop := (‖2 • a - b‖ = real.sqrt 3) -- Condition as given

-- Declare the goals to be proven
theorem proof_problem 
  (ha : unit_vector a) 
  (hb : unit_vector b)
  (h : given_condition) :
  (a.dot b = 1 / 2) ∧ (b.proj a = (1 / 2) • a) :=
by
  sorry

end proof_problem_l678_678928


namespace probJackAndJillChosen_l678_678224

-- Define the probabilities of each worker being chosen
def probJack : ℝ := 0.20
def probJill : ℝ := 0.15

-- Define the probability that Jack and Jill are both chosen
def probJackAndJill : ℝ := probJack * probJill

-- Theorem stating the probability that Jack and Jill are both chosen
theorem probJackAndJillChosen : probJackAndJill = 0.03 := 
by
  -- Replace this sorry with the complete proof
  sorry

end probJackAndJillChosen_l678_678224


namespace suitable_survey_method_l678_678328

-- Definitions based on conditions
def large_population (n : ℕ) : Prop := n > 10000  -- Example threshold for large population
def impractical_comprehensive_survey : Prop := true  -- Given in condition

-- The statement of the problem
theorem suitable_survey_method (n : ℕ) (h1 : large_population n) (h2 : impractical_comprehensive_survey) : 
  ∃ method : String, method = "sampling survey" :=
sorry

end suitable_survey_method_l678_678328


namespace determine_n_l678_678540

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem determine_n (n : ℕ) (h1 : binom n 2 + binom n 1 = 6) : n = 3 := 
by
  sorry

end determine_n_l678_678540


namespace count_perfect_squares_l678_678454

theorem count_perfect_squares (N : ℕ) :
  (∃ n ≤ N, ∃ k : ℕ, 15 * n = k^2) → N = 1000 →  count (λ n, ∃ k : ℕ, 15 * n = k^2 ∧ n ≤ 1000) = 8 :=
by
  sorry

end count_perfect_squares_l678_678454


namespace imaginary_part_of_square_l678_678301

open Complex

theorem imaginary_part_of_square (a b : ℝ) : 
  imag ((Complex.ofReal a - Complex.i * Complex.ofReal b) * (Complex.ofReal a - Complex.i * Complex.ofReal b)) = -6 := 
by sorry

end imaginary_part_of_square_l678_678301


namespace smallest_possible_range_l678_678020

theorem smallest_possible_range (s : Fin 7 → ℝ) (h_mean : (s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6) / 7 = 8)
  (h_median : s 3 = 10) (h_diff : s.order.sndr - s.order.fst ≥ 4) : ∃ a b, a = s.order.fst ∧ b = s.order.sndr ∧ b - a = 4 :=
by sorry

end smallest_possible_range_l678_678020


namespace area_of_circle_l678_678681

theorem area_of_circle (C : ℝ) (hC : C = 36 * Real.pi) : 
  ∃ k : ℝ, (∃ r : ℝ, r = 18 ∧ k = r^2 ∧ (pi * r^2 = k * pi)) ∧ k = 324 :=
by
  sorry

end area_of_circle_l678_678681


namespace identify_clothing_l678_678778

-- Define the children
inductive Person
| Alyna
| Bohdan
| Vika
| Grysha

open Person

-- Define color type
inductive Color
| Red
| Blue

open Color

-- Define clothing pieces
structure Clothing :=
(tshirt : Color)
(shorts : Color)

-- Definitions of the given conditions
def condition1 (a b : Clothing) : Prop :=
a.tshirt = Red ∧ b.tshirt = Red ∧ a.shorts ≠ b.shorts

def condition2 (v g : Clothing) : Prop :=
v.shorts = Blue ∧ g.shorts = Blue ∧ v.tshirt ≠ g.tshirt

def condition3 (a v : Clothing) : Prop :=
a.tshirt ≠ v.tshirt ∧ a.shorts ≠ v.shorts

-- The proof problem statement
theorem identify_clothing (ca cb cv cg : Clothing)
  (h1 : condition1 ca cb) -- Alyna and Bohdan condition
  (h2 : condition2 cv cg) -- Vika and Grysha condition
  (h3 : condition3 ca cv) -- Alyna and Vika condition
  : ca = ⟨Red, Red⟩ ∧ cb = ⟨Red, Blue⟩ ∧ cv = ⟨Blue, Blue⟩ ∧ cg = ⟨Red, Blue⟩ :=
sorry

end identify_clothing_l678_678778


namespace solve_eq_simplify_expression_l678_678739

-- Part 1: Prove the solution to the given equation

theorem solve_eq (x : ℚ) : (1 / (x - 1) + 1 = 3 / (2 * x - 2)) → x = 3 / 2 :=
sorry

-- Part 2: Prove the simplified value of the given expression when x=1/2

theorem simplify_expression : (x = 1/2) →
  ((x^2 / (1 + x) - x) / ((x^2 - 1) / (x^2 + 2 * x + 1)) = 1) :=
sorry

end solve_eq_simplify_expression_l678_678739


namespace polynomial_expansion_sum_l678_678186

theorem polynomial_expansion_sum :
  let f : ℝ → ℝ := λ x, (1 - 2 * x) ^ 2010
  let a := f 0
  let a_sum := ∑ i in Finset.range 2011, f 1
  ∀ (x : ℝ), (a_sum + 2009 * a) = 2010 :=
by
  let f : ℝ → ℝ := λ x, (1 - 2 * x) ^ 2010
  let a := f 0
  let a_sum := ∑ i in Finset.range 2011, f 1
  have : a = 1 := by sorry
  have : a_sum + 2009 * a = 2010 := by sorry
  exact this

end polynomial_expansion_sum_l678_678186


namespace integral_solution_l678_678412

noncomputable def integral_problem : ℂ :=
  ∮ z in (path.circle (0 : ℂ) (5 / 2)), (exp(2 * z) / (z^4 + 5 * z^2 - 9))

theorem integral_solution : 
  ∮ z in (path.circle (0 : ℂ) (5 / 2)), (exp(2 * z) / (z^4 + 5 * z^2 - 9)) = π * I * (sinh(4) / 13) :=
by
  -- The proof goes here
  sorry

end integral_solution_l678_678412


namespace evaluate_expression_is_negative_one_l678_678855

noncomputable def problem_statement : ℝ :=
  let lg : ℝ → ℝ := log10 in
  lg (5 / 2) + 2 * lg 2 - 2

theorem evaluate_expression_is_negative_one :
  problem_statement = -1 :=
sorry

end evaluate_expression_is_negative_one_l678_678855


namespace part1_part2_part3_l678_678477

variable {m x : ℝ}

def p : Prop := (x^2 - 8x - 20) ≤ 0
def q : Prop := (x^2 + 2x + 1 - m^2) ≤ 0

-- (1) Prove the range of 'm' such that 'p' implies 'q'
theorem part1 (hpq : p → q) : -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

-- (2) Prove the range of 'm' such that '¬p' implies '¬q'
theorem part2 (hnpq : ¬p → ¬q) : 3 ≤ m ∨ m ≤ -3 :=
sorry

-- (3) Prove the range of 'm' such that '¬p' is necessary but not sufficient for '¬q'
theorem part3 (hnpsufnq : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 3 ≤ m ∨ m ≤ -3 :=
sorry

end part1_part2_part3_l678_678477


namespace initial_integer_l678_678404

theorem initial_integer (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_integer_l678_678404


namespace number_of_correct_propositions_is_one_l678_678906

def planes_parallel_to_same_line (plane1 plane2 line : Type) :=
  (plane1 ∥ line) → (plane2 ∥ line) → (plane1 ∥ plane2)

def lines_parallel_to_same_plane (line1 line2 plane : Type) :=
  (line1 ∥ plane) → (line2 ∥ plane) → (line1 ∥ line2)

def lines_perpendicular_to_same_line (line1 line2 line : Type) :=
  (line1 ⊥ line) → (line2 ⊥ line) → (line1 ∥ line2)

def lines_perpendicular_to_same_plane (line1 line2 plane : Type) :=
  (line1 ⊥ plane) → (line2 ⊥ plane) → (line1 ∥ line2)

theorem number_of_correct_propositions_is_one
  (plane1 plane2 : Type) (line1 line2 : Type) (line p : Type) :
  ¬ (planes_parallel_to_same_line plane1 plane2 line) ∧
  ¬ (lines_parallel_to_same_plane line1 line2 plane1) ∧
  ¬ (lines_perpendicular_to_same_line line1 line2 p) ∧
  (lines_perpendicular_to_same_plane line1 line2 plane2) :=
sorry

end number_of_correct_propositions_is_one_l678_678906


namespace minimum_moves_necessary_l678_678011

theorem minimum_moves_necessary (n : ℕ) (positions : Fin n → Fin n) 
(people : Fin n) (swap : Fin n → Fin n → Fin n) (target_distance : ℕ)
(h1 : n = 2021) 
(h2 : target_distance = 1000)
(h3 : ∀ p, swap p ((p + 1) % n) = (p + 1) % n)
(h4 : ∀ p q, swap p q = swap q p)
(h5 : ∃ M, ∀ t, (positions t + target_distance) % n = M)
: ∃ M : ℕ, M = target_distance * (n - target_distance + 1) := sorry

end minimum_moves_necessary_l678_678011


namespace speed_of_second_car_is_55_l678_678705

noncomputable def speed_of_second_car : ℝ :=
  let v := 55 in
  v

theorem speed_of_second_car_is_55
  (time : ℝ)
  (speed_first_car : ℝ)
  (distance_apart : ℝ)
  (h1 : time = 3)
  (h2 : speed_first_car = 40)
  (h3 : distance_apart = 45)
  (v : ℝ)
  (h4 : 3 * v - speed_first_car * 3 = distance_apart) :
  v = 55 :=
by
  sorry

end speed_of_second_car_is_55_l678_678705


namespace log_function_increasing_interval_l678_678895

theorem log_function_increasing_interval (a : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x < y → y ≤ 3 → 4 - ax > 0 ∧ (4 - ax < 4 - ay)) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end log_function_increasing_interval_l678_678895


namespace valid_arrangements_l678_678102

open Nat

/-- Defining the sets of students and jobs ---/
inductive Student
| A | B | C | D | E
deriving DecidableEq, Repr

inductive Job
| translation | tour_guide | etiquette | driver
deriving DecidableEq, Repr

/-- Constraints on who can do which job ---/
def can_do_job (s : Student) (j : Job) : Prop :=
  match s, j with
  | Student.A, Job.driver => False
  | Student.B, Job.driver => False
  | Student.C, Job.driver => False
  | _, _ => True

/-- The main statement to prove the total number of valid job assignments ---/
theorem valid_arrangements : 
  (∃ (f : Student → Job), 
    (∀ s, can_do_job s (f s)) ∧ 
    List.nodup (List.map f [Student.A, Student.B, Student.C, Student.D, Student.E]) ∧  
    ∀ j, ∃ s, f s = j) → 
  count_valid_arrangements = 78 :=
sorry

end valid_arrangements_l678_678102


namespace line_and_circle_separate_l678_678929

theorem line_and_circle_separate
  (α β : ℝ)
  (h_angle : (2 * 3 * real.cos 60) = 6 * (real.cos α * real.cos β + real.sin α * real.sin β))
  (line_eq : ∀ (x y : ℝ), x*real.cos α - y*real.sin α + 1/2 = 0)
  (circle_eq : ∀ (x y : ℝ), (x - real.cos β)^2 + (y + real.sin β)^2 = 1/2) :
  (abs (real.cos (α - β) + 1/2)) > (real.sqrt 2 / 2) :=
begin
  sorry
end

end line_and_circle_separate_l678_678929


namespace num_two_digit_powers_of_3_l678_678954

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678954


namespace password_probability_l678_678050

theorem password_probability :
  let p_vowel : ℚ := 5 / 26,
      p_even_digit : ℚ := 1 / 2,
      p_consonant : ℚ := 21 / 26 in
  p_vowel * p_even_digit * p_consonant = 105 / 1352 :=
by
  let p_vowel : ℚ := 5 / 26
  let p_even_digit : ℚ := 1 / 2
  let p_consonant : ℚ := 21 / 26
  show p_vowel * p_even_digit * p_consonant = 105 / 1352
  sorry

end password_probability_l678_678050


namespace reciprocal_in_third_quadrant_l678_678585

open Complex

-- Define the complex number G with given conditions
variable (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_mag : a^2 + b^2 = 5)

def G : ℂ := -a + b * I

-- Statement to prove the location of 1/G
theorem reciprocal_in_third_quadrant :
  let G_inv := (1 : ℂ) / G in
  G_inv.re < 0 ∧ G_inv.im < 0 ∧ Complex.abs G_inv < 1 :=
by
  -- The steps and proof should be provided here, but for now we add sorry
  sorry

end reciprocal_in_third_quadrant_l678_678585


namespace acute_lines_count_l678_678917

def is_distinct {α : Type*} [DecidableEq α] (x y z : α) : Prop :=
  x ≠ y ∧ x ≠ z ∧ y ≠ z

def is_acute_angle (a b : Int) : Prop :=
  -a / b > 0

theorem acute_lines_count :
  (∃a b c : Int, a ∈ { -3, -2, -1, 0, 1, 2, 3 } ∧
  b ∈ { -3, -2, -1, 0, 1, 2, 3 } ∧
  c ∈ { -3, -2, -1, 0, 1, 2, 3 } ∧
  is_distinct a b c ∧ is_acute_angle a b) →
  43 :=
sorry

end acute_lines_count_l678_678917


namespace rectangular_floor_length_l678_678687

noncomputable def breadth : ℝ := (361 / 3.00001 / 3).sqrt
noncomputable def length : ℝ := 3 * breadth

theorem rectangular_floor_length :
  (361 / 3.00001).sqrt * 3 * 3 ≈ 19.002 :=
by
  sorry

end rectangular_floor_length_l678_678687


namespace angle_between_vectors_l678_678472

variables (a b : ℝ^3)
variables (θ : ℝ)

-- Given conditions
def magnitude_a := 4
def magnitude_b := 3
def dot_product_eq_61 : Prop := ((2 • a - 3 • b) ∙ (2 • a + b)) = 61

-- The proof goal
theorem angle_between_vectors :
  (‖a‖ = magnitude_a) →
  (‖b‖ = magnitude_b) →
  dot_product_eq_61 →
  θ = 2 * real.pi / 3 :=
by
  sorry

end angle_between_vectors_l678_678472


namespace change_is_4_25_l678_678596

-- Define the conditions
def apple_cost : ℝ := 0.75
def amount_paid : ℝ := 5.00

-- State the theorem
theorem change_is_4_25 : amount_paid - apple_cost = 4.25 :=
by
  sorry

end change_is_4_25_l678_678596


namespace fourth_term_is_six_l678_678563

-- Definitions from the problem
variables (a d : ℕ)

-- Condition that the sum of the third and fifth terms is 12
def sum_third_fifth_eq_twelve : Prop := (a + 2 * d) + (a + 4 * d) = 12

-- The fourth term of the arithmetic sequence
def fourth_term : ℕ := a + 3 * d

-- The theorem we need to prove
theorem fourth_term_is_six (h : sum_third_fifth_eq_twelve a d) : fourth_term a d = 6 := by
  sorry

end fourth_term_is_six_l678_678563


namespace amelia_wins_probability_l678_678824

def amelia_prob_heads : ℚ := 1 / 4
def blaine_prob_heads : ℚ := 3 / 7

def probability_blaine_wins_first_turn : ℚ := blaine_prob_heads

def probability_amelia_wins_first_turn : ℚ :=
  (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_second_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_third_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * 
  (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins : ℚ :=
  probability_amelia_wins_first_turn + probability_amelia_wins_second_turn + probability_amelia_wins_third_turn

theorem amelia_wins_probability : probability_amelia_wins = 223 / 784 := by
  sorry

end amelia_wins_probability_l678_678824


namespace incorrect_mode_l678_678528

theorem incorrect_mode (data : List ℕ) (hdata : data = [1, 2, 4, 3, 5]) : ¬ (∃ mode, mode = 5 ∧ (data.count mode > 1)) :=
by
  sorry

end incorrect_mode_l678_678528


namespace sally_quarters_l678_678276

theorem sally_quarters (initial_quarters spent_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 760) 
  (h2 : spent_quarters = 418) 
  (calc_final : final_quarters = initial_quarters - spent_quarters) : 
  final_quarters = 342 := 
by 
  rw [h1, h2] at calc_final 
  exact calc_final

end sally_quarters_l678_678276


namespace clothes_color_proof_l678_678785

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l678_678785


namespace minimize_distance_between_perpendicular_bases_l678_678268

variables {A B C D E F : Point}
variables (AB BC CA : Line)
variables (triangle : Triangle ABC)
variables {h_A : ℝ} -- The altitude from vertex A to side BC

noncomputable def is_minimum_distance (EF : ℝ) : Prop :=
  ∃ (D : Point), (D ∈ BC ∧ D = foot_of_altitude_from_A_to_BC) ∧ (distance (foot (perpendicular D AB)) (foot (perpendicular D AC)) = h_A)

theorem minimize_distance_between_perpendicular_bases
  (h_A_nonneg : 0 ≤ h_A) :
  ∀ (D : Point), D ∈ BC → (∃ (E : Point) (F : Point), perpendicular D E AB ∧ perpendicular D F AC) →
  is_minimum_distance (distance E F) :=
sorry

end minimize_distance_between_perpendicular_bases_l678_678268


namespace smallest_cookie_packages_l678_678344

/-- The smallest number of cookie packages Zoey can buy in order to buy an equal number of cookie
and milk packages. -/
theorem smallest_cookie_packages (n : ℕ) (h1 : ∃ k : ℕ, 5 * k = 7 * n) : n = 7 :=
sorry

end smallest_cookie_packages_l678_678344


namespace dihedral_angle_proof_l678_678128

noncomputable def angle_between_planes 
  (α β : Real) : Real :=
  Real.arcsin (Real.sin α * Real.sin β)

theorem dihedral_angle_proof 
  (α β : Real) 
  (α_non_neg : 0 ≤ α) 
  (α_non_gtr : α ≤ Real.pi / 2) 
  (β_non_neg : 0 ≤ β) 
  (β_non_gtr : β ≤ Real.pi / 2) :
  angle_between_planes α β = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end dihedral_angle_proof_l678_678128


namespace maximum_value_triangle_l678_678132

theorem maximum_value_triangle (A B C : Type) (angle_A : ℝ) (length_BC : ℝ) :
  angle_A = 60 ∧ length_BC = real.sqrt 3 →
  ∃ t, t = 2 * real.sqrt 7 ∧ ∀ (AB AC : ℝ), AB + 2 * AC ≤ t :=
sorry

end maximum_value_triangle_l678_678132


namespace total_airflow_in_one_week_l678_678023

-- Define the conditions
def airflow_rate : ℕ := 10 -- liters per second
def working_time_per_day : ℕ := 10 -- minutes per day
def days_per_week : ℕ := 7

-- Define the conversion factors
def minutes_to_seconds : ℕ := 60

-- Define the total working time in seconds
def total_working_time_per_week : ℕ := working_time_per_day * days_per_week * minutes_to_seconds

-- Define the expected total airflow in one week
def expected_total_airflow : ℕ := airflow_rate * total_working_time_per_week

-- Prove that the expected total airflow is 42000 liters
theorem total_airflow_in_one_week : expected_total_airflow = 42000 := 
by
  -- assertion is correct given the conditions above 
  -- skip the proof
  sorry

end total_airflow_in_one_week_l678_678023


namespace find_parabola_constant_l678_678682

theorem find_parabola_constant (a b c : ℝ) (h_vertex : ∀ y, (4:ℝ) = -5 / 4 * y * y + 5 / 2 * y + c)
  (h_point : (-1:ℝ) = -5 / 4 * (3:ℝ) ^ 2 + 5 / 2 * (3:ℝ) + c ) :
  c = 11 / 4 :=
sorry

end find_parabola_constant_l678_678682


namespace Proposition1_Proposition4_l678_678502

-- Define basic objects for lines, planes, and perpendicularity and parallelism relations
universe u
variable {α β : Type u}

structure Line (α : Type u) := (l : α)
structure Plane (α : Type u) := (p : α)

def perpendicular (l : Line α) (a : Plane α) : Prop := sorry
def contained_in (l : Line α) (a : Plane α) : Prop := sorry
def parallel (a b : Plane α) : Prop := sorry
def intersect (l1 l2 : Line α) (a : Plane α) : Prop := sorry

-- Propositions to be proved
theorem Proposition1 (l : Line α) (a : Plane α) (l1 l2 : Line α) (h₁ : intersect l1 l2 a) (h₂ : perpendicular l l1) (h₃ : perpendicular l l2):
  perpendicular l a :=
  sorry

theorem Proposition4 (l : Line α) (a b : Plane α) (h₁ : contained_in l b) (h₂ : perpendicular l a):
  perpendicular a b :=
  sorry

end Proposition1_Proposition4_l678_678502


namespace part1_part2_part3_l678_678876

variables {a b : EuclideanSpace ℝ (Fin 3)}

noncomputable def vector_a_norm : Real := 4
noncomputable def vector_b_norm : Real := 3

axiom h1 : ∥a∥ = vector_a_norm
axiom h2 : ∥b∥ = vector_b_norm
axiom h3 : (2 • a - 3 • b) ⬝ (2 • a + b) = 61

theorem part1 : a ⬝ b = -6 :=
by
  sorry

theorem part2 : 
  ∃ θ (h : 0 ≤ θ ∧ θ ≤ Real.pi), 
    Real.cos θ = (a ⬝ b) / (∥a∥ * ∥b∥) ∧ θ = 2 * Real.pi / 3 :=
by
  sorry

theorem part3 : ∥a + b∥ = Real.sqrt 13 :=
by
  sorry

end part1_part2_part3_l678_678876


namespace num_two_digit_powers_of_3_l678_678960

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678960


namespace certain_number_divisibility_l678_678933

theorem certain_number_divisibility {n : ℕ} (h : ∃ count : ℕ, count = 50 ∧ (count = (300 / (2 * n)))) : n = 3 :=
by
  sorry

end certain_number_divisibility_l678_678933


namespace speed_in_still_water_l678_678753

-- Definitions of the conditions
def V_upstream : ℝ := 34
def V_downstream : ℝ := 48

-- Theorem stating the problem to be proved
theorem speed_in_still_water (V_upstream = 34) (V_downstream = 48) : 
  (V_upstream + V_downstream) / 2 = 41 := 
sorry

end speed_in_still_water_l678_678753


namespace part1_a_part1_b_estimated_probability_high_quality_dolls_10000_l678_678039

def data : List (Nat × Nat) := [(10, 9), (100, 96), (1000, 951), (2000, 1900), (3000, 2856), (5000, 4750)]

theorem part1_a (h : data_look 1000 data = 951) : 951 / 1000 = 0.951 :=
by
  sorry

theorem part1_b (h : data_look 5000 data = 4750) : 4750 / 5000 = 0.95 :=
by
  sorry

theorem estimated_probability : 
  let frequencies := data.map (fun (n, m) => (m / n : ℝ));
  abs ((frequencies.sum / frequencies.length : ℝ) - 0.95) < 0.01 :=
by
  sorry

theorem high_quality_dolls_10000 : 
  abs (10000 * 0.95 - 9500) < 1 :=
by
  sorry

end part1_a_part1_b_estimated_probability_high_quality_dolls_10000_l678_678039


namespace incorrect_option_C_l678_678902

theorem incorrect_option_C (a b d : ℝ) (h₁ : ∀ x : ℝ, x ≠ d → x^2 + a * x + b > 0) (h₂ : a > 0) :
  ¬∀ x₁ x₂ : ℝ, (x₁ * x₂ > 0) → ((x₁, x₂) ∈ {p : (ℝ × ℝ) | p.1^2 + a * p.1 - b < 0 ∧ p.2^2 + a * p.2 - b < 0}) :=
sorry

end incorrect_option_C_l678_678902


namespace price_tags_advantages_l678_678643

/-- Given that a seller has attached price tags to all products,
    this strategy provides several advantages:
    1. Simplifies the purchasing process for buyers,
    2. Reduces the requirement for seller and personnel,
    3. Acts as an additional advertising method,
    4. Increases trust and perceived value.
-/
theorem price_tags_advantages
    (attached_price_tags : ∀ (product : Product), Product.HasPriceTag product) :
    (∀ (buyer : Buyer), buyer.PurchaseProcessSimplified) ∧
    (ReducedSellerRequirement_and_StaffWorkload) ∧
    (AdditionalAdvertisingMethod) ∧
    (IncreasedTrust_and_PerceivedValue) :=
sorry

end price_tags_advantages_l678_678643


namespace xy_equals_18_l678_678546

theorem xy_equals_18 (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 :=
by
  sorry

end xy_equals_18_l678_678546


namespace ordered_triples_count_l678_678538

theorem ordered_triples_count :
  ∃ (count : ℕ), count = 4 ∧
  (∃ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    Nat.lcm a b = 90 ∧
    Nat.lcm a c = 980 ∧
    Nat.lcm b c = 630) :=
by
  sorry

end ordered_triples_count_l678_678538


namespace find_extreme_values_find_m_range_for_zeros_l678_678911

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x + 2

theorem find_extreme_values (m : ℝ) :
  (∀ x > 0, m ≤ 0 → (f x m ≠ 0 ∨ ∀ y > 0, f y m ≥ f x m ∨ f y m ≤ f x m)) ∧
  (∀ x > 0, m > 0 → ∃ x_max, x_max = 1 / m ∧ ∀ y > 0, f y m ≤ f x_max m) := 
sorry

theorem find_m_range_for_zeros (m : ℝ) :
  (∃ a b, a = 1 / Real.exp 2 ∧ b = Real.exp 1 ∧ (f a m = 0 ∧ f b m = 0)) ↔ 
  (m ≥ 3 / Real.exp 1 ∧ m < Real.exp 1) :=
sorry

end find_extreme_values_find_m_range_for_zeros_l678_678911


namespace minimum_value_of_expression_l678_678868

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x * y / z + z * x / y + y * z / x) * (x / (y * z) + y / (z * x) + z / (x * y))

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  expression x y z ≥ 9 :=
sorry

end minimum_value_of_expression_l678_678868


namespace total_animal_legs_l678_678323

theorem total_animal_legs (total_animals : ℕ) (sheep : ℕ) (chickens : ℕ) : 
  total_animals = 20 ∧ sheep = 10 ∧ chickens = 10 ∧ 
  2 * chickens + 4 * sheep = 60 :=
by 
  sorry

end total_animal_legs_l678_678323


namespace ABCD_is_parallelogram_l678_678213

-- Definitions based on the conditions in (a)
variables {A B C D O K L M N : Type*}
variable [convex_quad (A B C D) : convex_quad AB CD]
variable [O_on_diagonals : intersects AC BD O]
variable [K_on_AB : on_segment K AB]
variable [L_on_BC : on_segment L BC]
variable [M_on_CD : on_segment M CD]
variable [N_on_AD : on_segment N AD]
variable [O_on_KM : on_segment O KM]
variable [O_on_LN : on_segment O LN]
variable [O_bisects_KM : bisects O KM]
variable [O_bisects_LN : bisects O LN]

-- The theorem to be proven
theorem ABCD_is_parallelogram : is_parallelogram ABCD :=
sorry

end ABCD_is_parallelogram_l678_678213


namespace simplify_trig_expression_l678_678280

theorem simplify_trig_expression :
  (Real.cos (72 * Real.pi / 180) * Real.sin (78 * Real.pi / 180) +
   Real.sin (72 * Real.pi / 180) * Real.sin (12 * Real.pi / 180) = 1 / 2) :=
by sorry

end simplify_trig_expression_l678_678280


namespace possible_values_f_11_l678_678377

noncomputable def f : ℕ → ℕ := sorry

axiom coprime_property (a b : ℕ) (h : a.coprime b) : f (a * b) = f a * f b
axiom prime_property (m k : ℕ) (hm : Nat.Prime m) (hk : Nat.Prime k) : f (m + k - 3) = f m + f k - f 3

theorem possible_values_f_11 : f 11 = 1 ∨ f 11 = 11 := sorry

end possible_values_f_11_l678_678377


namespace correct_outfits_l678_678795

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l678_678795


namespace activity_preference_order_l678_678057

def fraction_fishing : ℚ := 13 / 36
def fraction_hiking : ℚ := 8 / 27
def fraction_painting : ℚ := 7 / 18

theorem activity_preference_order :
  let activities := [("Fishing", fraction_fishing), ("Hiking", fraction_hiking), ("Painting", fraction_painting)]
  let sorted_activities := activities.sort (λ x y => x.2 ≥ y.2)
  sorted_activities.map Prod.fst = ["Painting", "Fishing", "Hiking"] := by
  sorry

end activity_preference_order_l678_678057


namespace identify_clothes_l678_678800

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l678_678800


namespace max_ab_bc_ca_l678_678140

theorem max_ab_bc_ca (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 3) :
  ab + bc + ca ≤ 3 :=
sorry

end max_ab_bc_ca_l678_678140


namespace ball_hit_ground_in_time_l678_678059

theorem ball_hit_ground_in_time :
  ∃ t : ℝ, t ≥ 0 ∧ -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 :=
by sorry

end ball_hit_ground_in_time_l678_678059


namespace carlotta_total_time_l678_678463

-- Define the main function for calculating total time
def total_time (performance_time practicing_ratio tantrum_ratio : ℕ) : ℕ :=
  performance_time + (performance_time * practicing_ratio) + (performance_time * tantrum_ratio)

-- Define the conditions from the problem
def singing_time := 6
def practicing_per_minute := 3
def tantrums_per_minute := 5

-- The expected total time based on the conditions
def expected_total_time := 54

-- The theorem to prove the equivalence
theorem carlotta_total_time :
  total_time singing_time practicing_per_minute tantrums_per_minute = expected_total_time :=
by
  sorry

end carlotta_total_time_l678_678463


namespace two_digit_powers_of_three_l678_678948

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678948


namespace find_x_l678_678909

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 1)

theorem find_x (x : ℝ) (h : deriv f x = x) : x = 0 ∨ x = 1 :=
by
  sorry

end find_x_l678_678909


namespace least_n_such_that_4125_divides_factorial_l678_678713

theorem least_n_such_that_4125_divides_factorial : 
  ∃ (n : ℕ), 4125 ∣ (nat.factorial n) ∧ 
    ∀ m, m < n → ¬(4125 ∣ (nat.factorial m)) :=
begin
  use 15,
  split,
  { sorry }, -- Showing 4125 divides 15!
  { intros m hm,
    sorry -- Showing no m less than 15 makes 4125 divide m! 
  },
end

end least_n_such_that_4125_divides_factorial_l678_678713


namespace count_two_digit_powers_of_three_l678_678979

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678979


namespace find_tan_theta_l678_678618

def mat_mul {α : Type} [has_mul α] (A B : matrix (fin 2) (fin 2) α) : matrix (fin 2) (fin 2) α :=
  λ i j, fin.cases_on i
    (fin.cases_on j (A 0 0 * B 0 0 + A 0 1 * B 1 0) (A 0 0 * B 0 1 + A 0 1 * B 1 1))
    (fin.cases_on j (A 1 0 * B 0 0 + A 1 1 * B 1 0) (A 1 0 * B 0 1 + A 1 1 * B 1 1))

variables (k : ℝ)
variables (θ : ℝ)
hypothesis h_k_pos : k > 0
def D : matrix (fin 2) (fin 2) ℝ := ![![k, 0], ![0, k]]
def R : matrix (fin 2) (fin 2) ℝ := ![![cos θ, -sin θ], ![sin θ, cos θ]]
def target_matrix : matrix (fin 2) (fin 2) ℝ := ![![6, -3], ![3, 6]]

theorem find_tan_theta :
  tan θ = 1 / 2 :=
by sorry

end find_tan_theta_l678_678618


namespace purple_position_sorted_alphabetically_l678_678566

theorem purple_position_sorted_alphabetically :
  ∀ (arrangements : List String),
  (∀ s ∈ arrangements, s.perm.multiplicity 'P' = 2 ∧
                       s.perm.multiplicity 'U' = 1 ∧
                       s.perm.multiplicity 'R' = 1 ∧
                       s.perm.multiplicity 'L' = 1 ∧
                       s.perm.multiplicity 'E' = 1) →
  ∃! n, arrangements.nth (n - 1) = some "PURPLE" → n = 226 := by
  sorry

end purple_position_sorted_alphabetically_l678_678566


namespace circle_line_distance_l678_678292

theorem circle_line_distance (a : ℝ) :
  let center := (3 : ℝ, 1 : ℝ),
      circ_eq := (λ (x y : ℝ), x^2 + y^2 - 6 * x - 2 * y + 3 = 0),
      line_eq := (λ (x y : ℝ), x + a * y - 1 = 0) in
  ∃ a, ∀ (x y : ℝ), circ_eq x y → let dist := |3 + a - 1| / sqrt (1 + a^2) in dist = 1 → a = -3 / 4 :=
by
  sorry

end circle_line_distance_l678_678292


namespace count_two_digit_powers_of_three_l678_678974

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678974


namespace range_of_m_l678_678890

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.cos (x + Real.pi / 6)

theorem range_of_m (m : ℝ) :
  (∀ a b : ℝ, (a ∈ set.Icc (Real.pi - m) m) ∧ ( b ∈ set.Icc (Real.pi - m) m) ∧ (a > b) → f a - f b < g (2 * a) - g (2 * b)) →
  Real.pi / 2 < m ∧ m < 17 * Real.pi / 24 :=
by
  sorry

end range_of_m_l678_678890


namespace seq_2014_zero_l678_678129

noncomputable def seq (n : ℕ) : ℕ → ℕ
| 1 := 1
| (2*n) := seq (n)
| (4*n - 3) := 1
| (4*n - 1) := 0
| _ := 0

theorem seq_2014_zero : seq 2014 = 0 := 
by sorry

end seq_2014_zero_l678_678129


namespace arithmetic_seq_general_formula_max_value_of_n_l678_678619

-- Definitions for Arithmetic Sequence and Geometric Conditions
def isArithmeticSeq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def sumFirstNTerms (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) : Prop :=
  S n = (n * (a 1 + a n)) / 2

def isGeometricSeq (a : ℕ → ℕ) (n1 n2 n3 : ℕ) : Prop :=
  a n2 ^ 2 = a n1 * a n3

-- The proof problem for Part (1)
theorem arithmetic_seq_general_formula (a : ℕ → ℕ) (d : ℕ) (S : ℕ → ℕ)
  (ha : isArithmeticSeq a d)
  (hS3 : S 3 = 9)
  (hg : isGeometricSeq a 2 5 14) :
  a = λ n, 2 * n - 1 :=
sorry

-- The proof problem for Part (2)
theorem max_value_of_n (S : ℕ → ℕ) :
  (∀ n, S n = n^2) → 
  ∀ n : ℕ, 
  (2 ≤ n → (1 - 1 / S 2) * (1 - 1 / S 3) * ... * (1 - 1 / S n) ≥ 1013 / 2022) → 
  n ≤ 505 :=
sorry
 
end arithmetic_seq_general_formula_max_value_of_n_l678_678619


namespace area_triangle_ABC_given_CHFI_l678_678434

-- Define the geometric conditions and the area of quadrilateral CHFI
variables (A B C D E F G H I : Point)
variables (hF_mid : Midpoint C D F) (hH_mid : Midpoint E D H) (hI_mid : Midpoint G D I)
variables (h_area_CHFI : area (Quadrilateral C H F I) = 5)

-- Define the main theorem to prove the area of triangle ABC given the conditions
theorem area_triangle_ABC_given_CHFI :
  area (Triangle A B C) = 40 := sorry

end area_triangle_ABC_given_CHFI_l678_678434


namespace train_speed_is_approx_l678_678044

noncomputable def train_speed_kmh (train_length platform_length: ℕ) (time_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / time_seconds
  speed_m_s * 3.6

theorem train_speed_is_approx (train_length : ℕ) (platform_length : ℕ) (time_seconds : ℝ)
  (h_train_length : train_length = 100)
  (h_platform_length : platform_length = 150)
  (h_time_seconds : time_seconds = 14.998800095992321) :
  train_speed_kmh train_length platform_length time_seconds ≈ 60.00 :=
by
  sorry

end train_speed_is_approx_l678_678044


namespace complex_pair_count_l678_678866

open Complex

theorem complex_pair_count :
  {a b : ℂ // a^4 * b^6 = 1 ∧ a^8 * b^3 = 1}.to_finset.card = 24 :=
by
  sorry

end complex_pair_count_l678_678866


namespace two_digit_powers_of_three_l678_678953

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678953


namespace prob_red_or_black_prob_not_green_l678_678366

noncomputable def box := {red := 5, black := 4, white := 2, green := 1}

def totalBalls := box.red + box.black + box.white + box.green

def prob (num : Nat) := (num : ℚ) / totalBalls

def prob_red := prob box.red
def prob_black := prob box.black
def prob_green := prob box.green

theorem prob_red_or_black : prob_red + prob_black = 3 / 4 := 
by {
  unfold prob_red prob_black prob,
  simp [box, totalBalls],
  sorry
}

theorem prob_not_green : 1 - prob_green = 11 / 12 :=
by {
  unfold prob_green prob,
  simp [box, totalBalls],
  sorry
}

end prob_red_or_black_prob_not_green_l678_678366


namespace max_u_l678_678878

theorem max_u (x y : ℝ) (h1 : tan x = 3 * tan y) (h2 : 0 ≤ y ∧ y ≤ x ∧ x < π / 2) : x - y ≤ π / 6 := 
sorry

end max_u_l678_678878


namespace meet_at_starting_point_second_time_l678_678680

/-- Define the time taken for one round for Racing Magic and Charging Bull -/
def time_racing_magic := 150 / 60 -- time in minutes per round
def time_charging_bull := 60 / 40 -- time in minutes per round

/-- Define the LCM function -/
def lcm (a b : ℚ) : ℚ :=
  a * b / gcd a b

/-- Main theorem statement -/
theorem meet_at_starting_point_second_time : 
  lcm time_racing_magic time_charging_bull = 15 :=
by 
  -- skipping the proof
  sorry

end meet_at_starting_point_second_time_l678_678680


namespace arithmetic_sequence_general_formula_sum_of_bn_terms_l678_678886

theorem arithmetic_sequence_general_formula (a : ℕ → ℕ) (h1 : a 3 = 6) (h2 : a 5 + a 8 = 26) :
  ∃ (a_1 d : ℕ), (a = λ n, a_1 + (n-1) * d) ∧ a_1 = 2 ∧ d = 2 :=
by
  sorry

theorem sum_of_bn_terms (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h : ∀ n, b n = 4^n + n)
  (hS : S n = ∑ i in finset.range n, b i) :
  S n = (4^(n+1) - 4)/3 + (n + n^2)/2 :=
by
  sorry

end arithmetic_sequence_general_formula_sum_of_bn_terms_l678_678886


namespace outcomes_with_more_than_3_streaks_l678_678658

theorem outcomes_with_more_than_3_streaks (R S : Type) (rounds : ℕ) (total_outcomes : ℕ) 
  (outcomes_with_1_streak outcomes_with_2_streaks outcomes_with_3_streaks : ℕ) :
  rounds = 10 →
  total_outcomes = 1024 →
  outcomes_with_1_streak = 2 →
  outcomes_with_2_streaks = 18 →
  outcomes_with_3_streaks = 72 →
  (total_outcomes - (outcomes_with_1_streak + outcomes_with_2_streaks + outcomes_with_3_streaks) = 932) :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  dsimp,
  norm_num,
end

end outcomes_with_more_than_3_streaks_l678_678658


namespace min_distance_z_w_l678_678250
open Complex

theorem min_distance_z_w (z w : ℂ) 
  (hz : abs (z - (2 - 4 * I)) = 2)
  (hw : abs (w - (6 - 5 * I)) = 4) :
  ∃ w, abs(z - w) = Real.sqrt 17 - 6 := sorry

end min_distance_z_w_l678_678250


namespace area_of_square_l678_678312

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem area_of_square : 
  let p1 := (1 : ℝ, -2 : ℝ)
  let p2 := (-3 : ℝ, 5 : ℝ)
  let side_length := distance p1 p2
  side_length = Real.sqrt 65 → 
  (side_length)^2 = 65 :=
by
  intros p1 p2 side_length h1
  simp [distance, p1, p2] at h1
  calc (Real.sqrt ((-3 - 1)^2 + (5 - -2)^2))
       = Real.sqrt ((-4)^2 + 7^2) : by simp
  ... = Real.sqrt (16 + 49) : by simp
  ... = Real.sqrt 65 : by simp
  sorry

end area_of_square_l678_678312


namespace evaluate_g_l678_678845

noncomputable def g (x : ℝ) : ℝ := log ((2 + x) / (2 - x))

theorem evaluate_g (x : ℝ) (h : -2 < x ∧ x < 2) :
  g ( (4 * x + x^2) / (2 * (1 + x)) ) = 2 * g x :=
by
  sorry

end evaluate_g_l678_678845


namespace fan_airflow_in_one_week_l678_678021

-- Define the conditions
def fan_airflow_per_second : ℕ := 10
def fan_working_minutes_per_day : ℕ := 10
def seconds_per_minute : ℕ := 60
def days_per_week : ℕ := 7

-- Define the proof problem
theorem fan_airflow_in_one_week : (fan_airflow_per_second * fan_working_minutes_per_day * seconds_per_minute * days_per_week = 42000) := 
by sorry

end fan_airflow_in_one_week_l678_678021


namespace intersects_y_axis_at_0_1_l678_678656

theorem intersects_y_axis_at_0_1 : ∀ x y : ℝ, y = -2 * x + 1 → (x = 0 → y = 1) :=
by
  intros x y h_eq h_x
  rw [h_x, mul_zero, add_zero] at h_eq
  exact h_eq

end intersects_y_axis_at_0_1_l678_678656


namespace number_of_rectangles_on_clock_face_l678_678288
-- We will use the Mathlib library for Lean 4

-- Define the problem in Lean 4 statement
theorem number_of_rectangles_on_clock_face : 
  let num_div_points := 12 in
  let num_rectangles := 
    (num_div_points / 2) * ((num_div_points / 2) - 1) / 2 in
  num_rectangles = 15 :=
by
  sorry

end number_of_rectangles_on_clock_face_l678_678288


namespace part_a_part_b_l678_678333

structure Points3D (A B C D : Type) [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D] :=
  (not_coplanar : ¬(affinely_independent ![A, B, C, D]))
  (AB_perp_CD : ∀ A B C D, (B - A) ⬝ (D - C) = 0)
  (AB2_CD2_eq_AD2_BC2 : ∀ A B C D, ∥B - A∥^2 + ∥D - C∥^2 = ∥D - A∥^2 + ∥C - B∥^2)

theorem part_a {A B C D : Type} [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D]
  (h : Points3D A B C D) : ∀ A B C D, (C - A) ⬝ (D - B) = 0 :=
sorry

theorem part_b {A B C D : Type} [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace D]
  (h : Points3D A B C D) (h1 : ∀ A B C D, ∥D - C∥ < ∥B - C∥ ∧ ∥B - C∥ < ∥B - D∥) :
  angle (plane_span ![A, B, C]) (plane_span ![A, D, C]) > π / 3 :=
sorry

end part_a_part_b_l678_678333


namespace train_pass_man_time_l678_678392

-- Define the conditions
def length_of_platform : ℝ := 150.012
def time_to_pass_platform : ℝ := 30
def speed_of_train_kmph : ℝ := 54
def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

-- Derived definition for the length of the train
def length_of_train : ℝ := speed_of_train_mps * time_to_pass_platform - length_of_platform

-- Problem: Calculate the time to pass the man standing on the platform
def time_to_pass_man : ℝ := length_of_train / speed_of_train_mps

-- The theorem we need to prove
theorem train_pass_man_time : time_to_pass_man = 20 := 
by
  sorry

end train_pass_man_time_l678_678392


namespace river_and_building_geometry_l678_678674

open Real

theorem river_and_building_geometry (x y : ℝ) :
  (tan 60 * x = y) ∧ (tan 30 * (x + 30) = y) → x = 15 ∧ y = 15 * sqrt 3 :=
by
  sorry

end river_and_building_geometry_l678_678674


namespace point_below_line_l678_678541

theorem point_below_line (m n : ℝ) (h : 2^m + 2^n < 2 * real.sqrt 2) : m + n < 1 :=
sorry

end point_below_line_l678_678541


namespace wire_length_of_cube_l678_678185

theorem wire_length_of_cube (V : ℝ) (hV : V = 3375) :
  let s := real.cbrt V in
  let num_edges := 12 in
  let total_wire_length := num_edges * s in
  total_wire_length = 180 :=
by
  sorry

end wire_length_of_cube_l678_678185


namespace children_without_candies_l678_678006

/-- There are 73 children standing in a circle. An evil Santa Claus walks around 
    the circle in a clockwise direction and distributes candies. First, he gives one candy 
    to the first child, then skips 1 child, gives one candy to the next child, 
    skips 2 children, gives one candy to the next child, skips 3 children, and so on.
    
    After distributing 2020 candies, he leaves. 
    
    This theorem states that the number of children who did not receive any candies 
    is 36. -/
theorem children_without_candies : 
  let n := 73
  let a : ℕ → ℕ := λk, (k * (k + 1) / 2) % n
  ∃ m : ℕ, (distributed_positions m 2020 73 = 37) → (73 - 37) = 36
  sorry

end children_without_candies_l678_678006


namespace range_of_sum_l678_678086

theorem range_of_sum (a b c d : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) ∧
  (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
sorry

end range_of_sum_l678_678086


namespace colors_of_clothes_l678_678808

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l678_678808


namespace correct_outfits_l678_678790

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l678_678790


namespace find_k_range_t_min_g_l678_678632

-- Function definition: f(x) = a^x - (k-1)a^x where a > 0 and a ≠ 1.
def f (a k x : ℝ) : ℝ := a^x - (k - 1) * a^x

-- Statement (a), finding k
theorem find_k (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (f_odd : ∀ x : ℝ, f a 1 x = -f a 1 (-x)) : 
  ∃ k, f a k 0 = 0 ∧ k = 2 :=
sorry

-- Statement (b), finding the range of t for the inequality
theorem range_t (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 1 < 0) : 
  ∃ t, (-3 < t ∧ t < 5) :=
sorry

-- Statement (c), finding the minimum value of g on [1, +∞)
def g (a x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * (a^x - a^(-x))

theorem min_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 1 = 3/2) : 
  ∃ m, m = g a 1 ∧ m = 5/4 :=
sorry

end find_k_range_t_min_g_l678_678632


namespace largest_c_inequality_l678_678627

open Set

theorem largest_c_inequality
  (n : ℕ) (h_n : 2 < n)
  (ℓ : ℕ) (h_ℓ : 1 ≤ ℓ ∧ ℓ ≤ n) :
  ∃ c : ℝ, c = (n + ℓ^2 - 2*ℓ) / (n * (n - 1)) :=
by
  use (n + ℓ^2 - 2*ℓ) / (n * (n - 1))
  sorry

end largest_c_inequality_l678_678627


namespace ratio_EG_GF_l678_678735

-- Given a triangle ABC, M is the midpoint of BC
variables {A B C M E F G : Type}
variables [EuclideanGeometry A B C M E F G]

-- Given the lengths of the sides AB and AC
axiom AB_eq_12 : distance A B = 12
axiom AC_eq_16 : distance A C = 16

-- Given that M is the midpoint of BC
axiom M_midpoint_BC : midpoint M B C

-- E is on AC and F is on AB such that AE = 2 * AF
axiom E_on_AC : on_line AC E = true
axiom F_on_AB : on_line AB F = true
axiom AE_eq_2AF : distance A E = 2 * distance A F

-- G is the intersection point of EF and AM
axiom G_intersection : intersects_line EF (line A M)

-- Prove: the ratio EG / GF = 3 / 2
theorem ratio_EG_GF : ratio (segment E G) (segment G F) = 3 / 2 := sorry

end ratio_EG_GF_l678_678735


namespace determine_clothes_l678_678820

-- Define the types
inductive Color where
  | red
  | blue
  deriving DecidableEq

structure Clothes where
  tshirt : Color
  shorts : Color

-- Definitions according to the problem's conditions
def Alyna : Clothes := { tshirt := Color.red, shorts := Color.red }
def Bohdan : Clothes := { tshirt := Color.red, shorts := Color.blue }
def Vika : Clothes := { tshirt := Color.blue, shorts := Color.blue }
def Grysha : Clothes := { tshirt := Color.red, shorts := Color.blue }

-- Problem statement in Lean
theorem determine_clothes : 
  (Alyna.tshirt = Color.red ∧ Alyna.shorts = Color.red) ∧
  (Bohdan.tshirt = Color.red ∧ Bohdan.shorts = Color.blue) ∧
  (Vika.tshirt = Color.blue ∧ Vika.shorts = Color.blue) ∧
  (Grysha.tshirt = Color.red ∧ Grysha.shorts = Color.blue) :=
sorry

end determine_clothes_l678_678820


namespace sum_of_four_digit_multiples_of_5_l678_678717

theorem sum_of_four_digit_multiples_of_5 :
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  S = 9895500 :=
by
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end sum_of_four_digit_multiples_of_5_l678_678717


namespace num_ordered_triplets_l678_678315

noncomputable def ordered_triplets_count : ℕ := 
  let s := { (x, y, z) : ℝ × ℝ × ℝ | 
              x + 2 * y + 4 * z = 12 ∧
              x * y + 4 * y * z + 2 * x * z = 22 ∧
              x * y * z = 6} in
  s.to_finset.card

theorem num_ordered_triplets : ordered_triplets_count = 6 := 
  sorry

end num_ordered_triplets_l678_678315


namespace log_sum_equality_l678_678443

noncomputable def evaluate_log_sum : ℝ :=
  3 / (Real.log 1000^4 / Real.log 8) + 4 / (Real.log 1000^4 / Real.log 10)

theorem log_sum_equality :
  evaluate_log_sum = (9 * Real.log 2 / Real.log 10 + 4) / 12 :=
by
  sorry

end log_sum_equality_l678_678443


namespace min_sum_of_2x2_grid_l678_678843

theorem min_sum_of_2x2_grid (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
(h_sum : a * b + c * d + a * c + b * d = 2015) : a + b + c + d = 88 :=
sorry

end min_sum_of_2x2_grid_l678_678843


namespace dan_helmet_craters_l678_678431

namespace HelmetCraters

variables {Dan Daniel Rin : ℕ}

/-- Condition 1: Dan's skateboarding helmet has ten more craters than Daniel's ski helmet. -/
def condition1 (C_d C_daniel : ℕ) : Prop := C_d = C_daniel + 10

/-- Condition 2: Rin's snorkel helmet has 15 more craters than Dan's and Daniel's helmets combined. -/
def condition2 (C_r C_d C_daniel : ℕ) : Prop := C_r = C_d + C_daniel + 15

/-- Condition 3: Rin's helmet has 75 craters. -/
def condition3 (C_r : ℕ) : Prop := C_r = 75

/-- The main theorem: Dan's skateboarding helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (C_d C_daniel C_r : ℕ) 
    (h1 : condition1 C_d C_daniel) 
    (h2 : condition2 C_r C_d C_daniel) 
    (h3 : condition3 C_r) : C_d = 35 :=
by {
    -- We state that the answer is 35 based on the conditions
    sorry
}

end HelmetCraters

end dan_helmet_craters_l678_678431


namespace analytical_expression_monotonicity_decreasing_inequality_solution_l678_678910

-- Define the function and the given conditions
def f (x : ℝ) : ℝ := (a * x + b) / (x^2 + 2)

-- Condition 1: f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def odd_condition : is_odd_function f := sorry

-- Condition 2: f(-1/2) = 2/9
def specific_value : f (-1 / 2) = 2 / 9 := sorry

-- Theorem 1: Prove the analytical expression of f(x)
theorem analytical_expression (x : ℝ) (h1 : is_odd_function f) (h2 : specific_value) :
  f x = -x / (x^2 + 2) := sorry

-- Theorem 2: Prove monotonicity of the function f(x)
theorem monotonicity_decreasing (h1 : ∀ x, f x = -x / (x^2 + 2)) :
  ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 := sorry

-- Define new function with given analytical expression for inequality part
def f' (x : ℝ) : ℝ := -x / (x^2 + 2)

-- Theorem 3: Prove the inequality implies 0 < t < 1 / 2
theorem inequality_solution (t : ℝ) (h : f' (t + 1 / 2) + f' (t - 1 / 2) < 0) :
  0 < t ∧ t < 1 / 2 := sorry

end analytical_expression_monotonicity_decreasing_inequality_solution_l678_678910


namespace sum_of_factors_72_l678_678418

theorem sum_of_factors_72 : 
  ∑ d in (finset.filter (λ x, 72 % x = 0) (finset.range 73)), d = 195 := by
  sorry

end sum_of_factors_72_l678_678418


namespace two_digit_powers_of_three_l678_678952

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678952


namespace candy_piece_count_l678_678840

theorem candy_piece_count (x : ℕ) (h : 10 * x = 80) : x = 8 :=
begin
  sorry
end

end candy_piece_count_l678_678840


namespace num_two_digit_powers_of_3_l678_678935

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678935


namespace concert_ticket_to_motorcycle_ratio_l678_678594

theorem concert_ticket_to_motorcycle_ratio (initial_amount spend_motorcycle remaining_amount : ℕ)
  (h_initial : initial_amount = 5000)
  (h_spend_motorcycle : spend_motorcycle = 2800)
  (amount_left := initial_amount - spend_motorcycle)
  (h_remaining : remaining_amount = 825)
  (h_amount_left : ∃ C : ℕ, amount_left - C - (1/4 : ℚ) * (amount_left - C) = remaining_amount) :
  ∃ C : ℕ, (C / amount_left) = (1 / 2 : ℚ) := sorry

end concert_ticket_to_motorcycle_ratio_l678_678594


namespace seq_a_eq_n_sum_seq_b_l678_678903

def sequence_a (n : ℕ) := ∑ i in range n, i

def sequence_s (n : ℕ) := n * (n + 1) / 2

theorem seq_a_eq_n : ∀ n, sequence_a n = n := by
  sorry

def sequence_b (n : ℕ) := (-1)^n * (n * 2^n + 1 / (real.sqrt (n + 1) - real.sqrt n))

def sequence_t (n : ℕ) := -11 / 9 - (3 * n + 1) / 9 * (-2)^(n + 1) + (-1)^n * real.sqrt (n + 1)

theorem sum_seq_b : ∀ n, ∑ i in range n, sequence_b i = sequence_t n := by
  sorry

end seq_a_eq_n_sum_seq_b_l678_678903


namespace g_below_f_l678_678247

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1
def g (x : ℝ) : ℝ := -x^2 + 2*x - 2

theorem g_below_f (x : ℝ) : g(x) < f(x) :=
by
  sorry

end g_below_f_l678_678247


namespace length_of_BC_l678_678200

-- Definitions for the problem
def angle_A : ℝ := 60
def side_AC : ℝ := 3
def side_AB : ℝ := 2

-- Cosine of 60 degrees
def cos_60 : ℝ := 1 / 2

-- Using the Cosine Rule, we want to show BC^2 = 7, hence BC = sqrt(7)
theorem length_of_BC :
  let BC := sqrt (side_AC^2 + side_AB^2 - 2 * side_AC * side_AB * cos_60)
  BC = sqrt 7 :=
by
  sorry

end length_of_BC_l678_678200


namespace circle_area_increase_l678_678349

theorem circle_area_increase (r : ℝ) : 
  let R := 1.12 * r in
  let A_original := Real.pi * r^2 in
  let A_new := Real.pi * R^2 in
  (A_new = 1.2544 * A_original) := 
by
  let R := 1.12 * r
  let A_original := Real.pi * r^2
  let A_new := Real.pi * (1.12 * r)^2
  have h1 : A_new = 1.2544 * A_original,
  { sorry }, -- Proof omitted for this task
  exact h1

end circle_area_increase_l678_678349


namespace john_kept_percentage_l678_678227

open Float

noncomputable def original_cost : Float := 20000.0
noncomputable def discount_rate : Float := 0.20
noncomputable def prize_money : Float := 70000.0
noncomputable def kept_money : Float := 47000.0

theorem john_kept_percentage :
  let discount := original_cost * discount_rate
  let cost_after_discount := original_cost - discount
  (kept_money / prize_money) * 100 ≈ 67.14 := 
by
  sorry

end john_kept_percentage_l678_678227


namespace measure_of_angle_YHZ_l678_678219

-- Triangle XYZ with altitudes XP, YQ, ZR meeting at orthocenter H and given angles
variables (X Y Z P Q R H : Type)
variables [triangle : noncomputable ℝ] -- Assume we have a triangle defined in ℝ
variables [altitudes : XP YQ ZR intersect at H in XYZ]

def angle_XYZ_eq_37 : ℝ := 37
def angle_XZY_eq_53 : ℝ := 53

theorem measure_of_angle_YHZ
  (altitudes : XP ∧ YQ ∧ ZR are altitudes in XYZ)
  (orthocenter : XP ∧ YQ ∧ ZR intersect at H)
  (angle_XYZ : ∠XYZ = angle_XYZ_eq_37)
  (angle_XZY : ∠XZY = angle_XZY_eq_53) : (∠YHZ = 90) :=
begin
  sorry,
end

end measure_of_angle_YHZ_l678_678219


namespace find_number_of_students_l678_678572

theorem find_number_of_students
  (n : ℕ)
  (average_marks : ℕ → ℚ)
  (wrong_mark_corrected : ℕ → ℕ → ℚ)
  (correct_avg_marks_pred : ℕ → ℚ → Prop)
  (h1 : average_marks n = 60)
  (h2 : wrong_mark_corrected 90 15 = 75)
  (h3 : correct_avg_marks_pred n 57.5) :
  n = 30 :=
sorry

end find_number_of_students_l678_678572


namespace two_digit_numbers_in_form_3_pow_n_l678_678969

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678969


namespace solve_for_x_l678_678494

noncomputable def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x : ∃ x y, 2 * f x - 16 = f y ∧ x = 7 ∧ y = 7 :=
by
  use 7
  use 7
  split
  { sorry } -- skip proof that 2 * f 7 - 16 = f 7
  split
  { exact rfl }
  { exact rfl }

end solve_for_x_l678_678494


namespace table_tennis_count_l678_678742

noncomputable def table_tennis_outcomes (A B : ℕ) : Prop :=
  ∃ outcomes : Finset (List (Sum Unit Unit)), 
  outcomes.card = 20 ∧
  ∀ outcome ∈ outcomes, 
    (A ∈ outcome.toList.map (λ x => x.inl)) ∨
    (B ∈ outcome.toList.map (λ x => x.inr))

theorem table_tennis_count :
  table_tennis_outcomes 3 3 := 
sorry

end table_tennis_count_l678_678742


namespace trough_water_after_40_days_l678_678750

theorem trough_water_after_40_days :
  ∀ (initial_water : ℝ) (evaporation_rate : ℝ) (days_passed : ℕ), 
  initial_water = 300 → 
  evaporation_rate = 0.75 → 
  days_passed = 40 → 
  initial_water - evaporation_rate * days_passed = 270 :=
by
  intros initial_water evaporation_rate days_passed h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end trough_water_after_40_days_l678_678750


namespace count_integers_satisfying_inequality_l678_678183

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), S.card = 8 ∧ ∀ n ∈ S, -11 ≤ n ∧ n ≤ 11 ∧ (n - 2) * (n + 4) * (n + 8) < 0 :=
by
  sorry

end count_integers_satisfying_inequality_l678_678183


namespace fan_airflow_in_one_week_l678_678022

-- Define the conditions
def fan_airflow_per_second : ℕ := 10
def fan_working_minutes_per_day : ℕ := 10
def seconds_per_minute : ℕ := 60
def days_per_week : ℕ := 7

-- Define the proof problem
theorem fan_airflow_in_one_week : (fan_airflow_per_second * fan_working_minutes_per_day * seconds_per_minute * days_per_week = 42000) := 
by sorry

end fan_airflow_in_one_week_l678_678022


namespace drawing_specific_cards_from_two_decks_l678_678471

def prob_of_drawing_specific_cards (total_cards_deck1 total_cards_deck2 : ℕ) 
  (specific_card1 specific_card2 : ℕ) : ℚ :=
(specific_card1 / total_cards_deck1) * (specific_card2 / total_cards_deck2)

theorem drawing_specific_cards_from_two_decks :
  prob_of_drawing_specific_cards 52 52 1 1 = 1 / 2704 :=
by
  -- The proof can be filled in here
  sorry

end drawing_specific_cards_from_two_decks_l678_678471


namespace count_valid_functions_l678_678258

-- Define the sets X and Y
def X : Set ℤ := {-10, -1, 1}
def Y : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the condition that x + f(x) is odd
def is_odd_sum (f : ℤ → ℤ) (x : ℤ) : Prop := (x + f x) % 2 ≠ 0

-- Define the function type we're interested in
def valid_function (f : ℤ → ℤ) : Prop := ∀ x ∈ X, is_odd_sum f x

-- Prove that there are 18 such valid functions
theorem count_valid_functions : (set.univ : Set (ℤ → ℤ)).count (λ f, valid_function f) = 18 :=
sorry

end count_valid_functions_l678_678258


namespace angle_bisector_and_area_ratio_l678_678497

theorem angle_bisector_and_area_ratio (A B C P : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space P]
  (h_right_triangle : ∀ (a b : ℝ), ∃ (A B C : Point), is_right_triangle A B C a b ) 
  (h_legs : BC = 5 ∧ AC = 12)
  (h_bisector : is_angle_bisector C.angle ACB (C bisects AB))
  (h_intersect : intersects_bisector C AB P):
(prove_length_Bisector_AB :  (length_of_bisector C P) = 156 / 17) (prove_area_ratio :
  (triangle_area_ratio ABC PCB) = 85 / 13): 
= sorry

end angle_bisector_and_area_ratio_l678_678497


namespace right_triangle_correct_set_l678_678725

theorem right_triangle_correct_set :
  (∃ (a b c : ℕ), a = 7 ∧ b = 24 ∧ c = 25 ∧ (a * a + b * b = c * c)) :=
begin
  use [7, 24, 25],
  split,
  { refl },
  split,
  { refl },
  { exact eq.refl 25 }
end

end right_triangle_correct_set_l678_678725


namespace quadrilateral_incompatibility_pentagon_independence_l678_678675

-- Define the properties and conditions for the quadrilateral (parallelogram)
def is_parallelogram_with_all_right_angles (Q : Type) [quadrilateral Q] : Prop :=
  (∀ (a : Q.angle), a = 90)

def has_equal_diagonals (Q : Type) [quadrilateral Q] : Prop :=
  (Q.diagonal₁.length = Q.diagonal₂.length)

-- Define the properties and conditions for the pentagon
def has_one_right_angle (P : Type) [pentagon P] : Prop :=
  ∃ (a : P.angle), a = 90

def has_unequal_diagonals (P : Type) [pentagon P] : Prop :=
  (P.diagonal₁.length ≠ P.diagonal₂.length)

-- The theorem to be proved for the quadrilateral
theorem quadrilateral_incompatibility (Q : Type) [quadrilateral Q] :
  is_parallelogram_with_all_right_angles Q → ¬ has_unequal_diagonals Q :=
sorry

-- The theorem to be proved for the pentagon
theorem pentagon_independence (P : Type) [pentagon P] :
  has_one_right_angle P → has_unequal_diagonals P :=
sorry

end quadrilateral_incompatibility_pentagon_independence_l678_678675


namespace total_spent_correct_l678_678597

def discounted_price (original_price discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

def taxed_price (original_price tax_rate : ℝ) : ℝ :=
  original_price * (1 + tax_rate)

def total_spent_at_music_store : ℝ :=
  discounted_price 142.46 0.10 + 
  discounted_price 26.55 0.15 +
  taxed_price 8.89 0.05 + 
  taxed_price 7 0.06 + 
  taxed_price 35.99 0.08

theorem total_spent_correct:
  (Float.round (total_spent_at_music_store * 100) / 100) = 206.41 := by
  sorry

end total_spent_correct_l678_678597


namespace part1_part2_l678_678143

variable (x y z : ℝ)

-- Condition: x, y, z are positive numbers
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

-- Part 1: Prove the inequality
theorem part1 : 
  (x / (y * z)) + (y / (z * x)) + (z / (x * y)) ≥ (1 / x) + (1 / y) + (1 / z) := 
  sorry

-- Condition for Part 2: x + y + z ≥ x * y * z
variable (hxyz : x + y + z ≥ x * y * z)

-- Part 2: Find the minimum value of u given the condition
theorem part2 :
  ((x / (y * z)) + (y / (z * x)) + (z / (x * y))) ≥ sqrt 3 := 
  sorry

end part1_part2_l678_678143


namespace num_two_digit_powers_of_3_l678_678959

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678959


namespace answer1_answer2_answer3_l678_678042

-- Define the sample data
def sample_data := [(10, 9, 0.9), (100, 96, 0.96), (1000, 951, _), (2000, 1900, 0.95), (3000, 2856, 0.952), (5000, 4750, _)]

-- Predefined values for a and b
def a : ℚ := 951 / 1000
def b : ℚ := 4750 / 5000

-- The probability of selecting a high-quality doll
def estimated_probability : ℚ := 0.95

-- The number of high-quality dolls in a batch of 10000
def high_quality_dolls_in_10000 : ℕ := (10000 : ℚ) * estimated_probability

theorem answer1 : a = 0.951 ∧ b = 0.95 := by
  sorry

theorem answer2 : estimated_probability = 0.95 := by
  sorry

theorem answer3 : high_quality_dolls_in_10000 = 9500 := by
  sorry

end answer1_answer2_answer3_l678_678042


namespace fraction_spent_at_grocery_proof_l678_678831

noncomputable def fraction_spent_at_grocery (initial_money : ℝ) (hardware_fraction : ℝ) (dry_cleaner_cost : ℝ) (final_money : ℝ) : ℝ :=
  let after_hardware := initial_money - (hardware_fraction * initial_money)
  let after_dry_cleaner := after_hardware - dry_cleaner_cost
  (after_dry_cleaner - final_money) / after_dry_cleaner

theorem fraction_spent_at_grocery_proof :
  fraction_spent_at_grocery 52 (1 / 4) 9 15 = 1 / 2 :=
by
  apply eq.rfl

end fraction_spent_at_grocery_proof_l678_678831


namespace sqrt_five_gt_two_l678_678841

theorem sqrt_five_gt_two : Real.sqrt 5 > 2 :=
by
  -- Proof goes here
  sorry

end sqrt_five_gt_two_l678_678841


namespace cooking_oil_remaining_l678_678015

theorem cooking_oil_remaining (initial_weight : ℝ) (fraction_used : ℝ) (remaining_weight : ℝ) :
  initial_weight = 5 → fraction_used = 4 / 5 → remaining_weight = 21 / 5 → initial_weight * (1 - fraction_used) ≠ remaining_weight → initial_weight * (1 - fraction_used) = 1 :=
by 
  intros h_initial_weight h_fraction_used h_remaining_weight h_contradiction
  sorry

end cooking_oil_remaining_l678_678015


namespace cylinder_height_l678_678769

theorem cylinder_height (LSA EA : ℝ) (r h : ℝ) (h_LSA : LSA = 16 * Real.pi) (h_EA : EA = 8 * Real.pi) (h_radius : r = 2) :
  h = 4 :=
by
  sorry

end cylinder_height_l678_678769


namespace problem1_problem2_l678_678361

-- Problem 1
theorem problem1 : sqrt ((-2) ^ 2) + real.cbrt (-27) - sqrt 16 = -5 := 
by
  sorry

-- Problem 2
theorem problem2 : sqrt 81 + abs (sqrt 3 - 2) - 50 * (sqrt (1 / 10)) ^ 2 = 6 - sqrt 3 := 
by
  sorry

end problem1_problem2_l678_678361


namespace brain_activity_indicates_motion_l678_678275

-- Definitions and conditions based on the problem
def theta_waves_produce_neurons := "Recent neuroscience research has shown that 'theta waves' produced by learning can promote the generation of brain neurons"
def brain_used_more_agile := "the more the brain is used, the more agile it becomes"

-- The conjecture to prove
theorem brain_activity_indicates_motion (H1 : theta_waves_produce_neurons) (H2 : brain_used_more_agile) : 
  "Motion is the mode of existence and the inherent fundamental attribute of matter" :=
sorry

end brain_activity_indicates_motion_l678_678275


namespace xyz_inequality_l678_678691

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  (x - y) * (y - z) * (x - z) ≤ 1 / Real.sqrt 2 :=
by
  sorry

end xyz_inequality_l678_678691


namespace two_digit_powers_of_three_l678_678946

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678946


namespace sum_of_median_and_third_l678_678305

def largest := 10
def smallest := -3
def median := 5
def numbers := [-3, 0, 5, 8, 10]

def condition1 (perm : List Int) : Prop :=
  perm.getLast ≠ 10 ∧ perm.getLast (perm.length - 2) ≠ 10 

def condition2 (perm : List Int) : Prop :=
  perm.head ≠ -3 ∧ perm.head (1) ≠ -3

def condition3 (perm : List Int) : Prop :=
  perm.getLast (2) ≠ 5
  
def condition4 (perm : List Int) : Prop :=
  (perm.head (1) % 2 = 1) 

theorem sum_of_median_and_third : 
  ∀ (perm : List Int), 
  perm ~ numbers →
  condition1 perm → 
  condition2 perm → 
  condition3 perm →
  condition4 perm →
  (perm.getLast 2) + median = 13 :=
by sorry

end sum_of_median_and_third_l678_678305


namespace xy_value_l678_678548

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := 
by
  sorry

end xy_value_l678_678548


namespace max_tickets_jane_can_buy_l678_678851

def ticket_price : ℝ := 15.75
def processing_fee : ℝ := 1.25
def jane_money : ℝ := 150.00

theorem max_tickets_jane_can_buy : ⌊jane_money / (ticket_price + processing_fee)⌋ = 8 := 
by
  sorry

end max_tickets_jane_can_buy_l678_678851


namespace pentagon_area_l678_678439

theorem pentagon_area (P : Type) [Convex P] (vertices : list P)
  (area_division : ∀ (i j : ℕ), 1 ≤ i < j ≤ 5 → ∃ Q T : set P, 
    (Q ∪ T = {vertices.nth_le i (Nat.mod_lt i (by decide)), vertices.nth_le j (Nat.mod_lt j (by decide))}) 
    ∧ isQuadrilateral Q ∧ isTriangle T ∧ area T = 1) :
  area P = 5 :=
sorry

end pentagon_area_l678_678439


namespace count_two_digit_powers_of_three_l678_678978

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678978


namespace entree_cost_l678_678177

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l678_678177


namespace subtraction_result_l678_678712

theorem subtraction_result: (3.75 - 1.4 = 2.35) :=
by
  sorry

end subtraction_result_l678_678712


namespace Bill_trips_l678_678832

theorem Bill_trips (total_trips : ℕ) (Jean_trips : ℕ) (Bill_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : Jean_trips = 23) 
  (h3 : Bill_trips + Jean_trips = total_trips) : 
  Bill_trips = 17 := 
by
  sorry

end Bill_trips_l678_678832


namespace shoe_factory_should_pay_attention_to_mode_l678_678721

-- Define the condition about the survey
def relevantShoeSizeMetric := "Mode"

-- Predicate for what the factory should pay the most attention to
def shouldPayAttentionTo (metric : String) : Prop :=
  metric = "Mode"

-- The statement to prove
theorem shoe_factory_should_pay_attention_to_mode :
  shouldPayAttentionTo relevantShoeSizeMetric := 
begin
  sorry
end

end shoe_factory_should_pay_attention_to_mode_l678_678721


namespace euler_totient_problem_l678_678238

open Nat

-- Define the Euler's Totient function φ notation
noncomputable def phi (n : ℕ) : ℕ := nat.totient n

-- Define the problem statement
theorem euler_totient_problem (n : ℕ) (p : ℕ) (hp : prime p) :
  phi(n) = phi(n * p) ↔ p = 2 ∧ (∀ k : ℕ, n ≠ 2 * k) := sorry

end euler_totient_problem_l678_678238


namespace original_savings_l678_678350

theorem original_savings (h1: Linda_saved_half : \frac{1}{2} x = 300) : x = 600 :=
sorry

end original_savings_l678_678350


namespace prove_clothing_colors_l678_678816

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l678_678816


namespace polynomial_exists_large_n_l678_678736

theorem polynomial_exists_large_n (n : ℕ) (h : n ≥ 100) :
  ∃ p : ℝ[X], 
    |p.eval 0| > (1 / 10) * (finset.sum (finset.range n) (λ k, (nat.choose n k : ℝ) * |p.eval k|)) ∧
    p.degree ≤ n - ⌊ 1 / 10 * real.sqrt n ⌋₊ - 1 :=
sorry

end polynomial_exists_large_n_l678_678736


namespace min_sum_of_fractions_l678_678614

theorem min_sum_of_fractions (A B C D : ℕ) (hA : A ∈ {1, 3, 4, 5, 6, 8, 9}) (hB : B ∈ {1, 3, 4, 5, 6, 8, 9}) 
  (hC : C ∈ {1, 3, 4, 5, 6, 8, 9}) (hD : D ∈ {1, 3, 4, 5, 6, 8, 9}) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) : 
  \(\frac{A}{B} + \frac{C}{D}\) ≥ \(\frac{11}{24}\) := sorry

end min_sum_of_fractions_l678_678614


namespace greatest_pentagon_area_l678_678373

-- Definition of a convex equilateral pentagon with specific properties
structure ConvexEquilateralPentagon (α β γ δ ε : ℝ) :=
  (side_length : ℝ)
  (angles : ℝ)
  (convex : Prop)
  (equilateral : Prop)
  (right_angles : Finset ℝ)
  (area : ℝ)

-- The given problem's conditions
theorem greatest_pentagon_area :
  ∀ (P : ConvexEquilateralPentagon),
    P.side_length = 2 ∧
    P.convex ∧
    P.equilateral ∧
    (∃ (A B : ℝ), (A ∈ P.right_angles ∧ B ∈ P.right_angles ∧ A ≠ B))
    → ∃ (m n : ℕ), P.area = m + Real.sqrt n ∧ 100 * m + n = 407 :=
by
  -- problem statement and conditions, proof is omitted
  sorry

end greatest_pentagon_area_l678_678373


namespace problem_solution_l678_678613

theorem problem_solution (m : ℕ) (hm : m > 0) :
  ∃ (x : Fin (m^2 + 1) → ℝ),
  (∀ (i : Fin (m^2 + 1)),
    x i = 1 + 2 * m * x i ^ 2 / (∑ j, (x j)^2)) ∧
  (∃ (perm : List (Fin (m^2 + 1)) → List (Fin (m^2 + 1))),
    List.map x (perm (List.finRange (m^2 + 1))) =
      (List.repeat (1 / m + 1) m^2) ++ [m + 1]) :=
sorry

end problem_solution_l678_678613


namespace find_atomic_weight_Cl_l678_678453

-- Given conditions
def atomic_weight_H : ℕ := 1
def atomic_weight_O : ℕ := 16
def molecular_weight : ℕ := 68

-- To find: atomic weight of chlorine (Cl)
noncomputable def atomic_weight_Cl : ℕ :=
  molecular_weight - (atomic_weight_H + 2 * atomic_weight_O) + 1

-- Proof statement
theorem find_atomic_weight_Cl : atomic_weight_Cl = 35 :=
by
  calc
    atomic_weight_Cl
        = molecular_weight - (atomic_weight_H + 2 * atomic_weight_O) + 1 : sorry
    ... = 35 : by simp

end find_atomic_weight_Cl_l678_678453


namespace num_two_digit_powers_of_3_l678_678942

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678942


namespace two_digit_numbers_in_form_3_pow_n_l678_678968

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678968


namespace correct_outfits_l678_678792

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l678_678792


namespace cos_beta_l678_678149

open Real InnerProductSpace

variables (e₁ e₂ : ℝ^2) (α β : ℝ)
variable (h_unit_e₁ : ∥e₁∥ = 1)
variable (h_unit_e₂ : ∥e₂∥ = 1)
variable (h_cos_α : cos α = 1 / 3)
variables (a b : ℝ^2)

definition a_def : a = 3 • e₁ - 2 • e₂ := sorry
definition b_def : b = 3 • e₁ - e₂ := sorry

theorem cos_beta :
  cos β = (2 * sqrt 2) / 3 :=
sorry

end cos_beta_l678_678149


namespace count_two_digit_powers_of_three_l678_678975

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678975


namespace kerry_age_l678_678605

theorem kerry_age (cost_per_box : ℝ) (boxes_bought : ℕ) (candles_per_box : ℕ) (cakes : ℕ) 
  (total_cost : ℝ) (total_candles : ℕ) (candles_per_cake : ℕ) (age : ℕ) :
  cost_per_box = 2.5 →
  boxes_bought = 2 →
  candles_per_box = 12 →
  cakes = 3 →
  total_cost = 5 →
  total_cost = boxes_bought * cost_per_box →
  total_candles = boxes_bought * candles_per_box →
  candles_per_cake = total_candles / cakes →
  age = candles_per_cake →
  age = 8 :=
by
  intros
  sorry

end kerry_age_l678_678605


namespace Jerry_throw_count_l678_678263

theorem Jerry_throw_count : 
  let interrupt_points := 5
  let insult_points := 10
  let throw_points := 25
  let threshold := 100
  let interrupt_count := 2
  let insult_count := 4
  let current_points := (interrupt_count * interrupt_points) + (insult_count * insult_points)
  let additional_points := threshold - current_points
  let throw_count := additional_points / throw_points
  in throw_count = 2 :=
by {
  have h1 : current_points = (2 * 5) + (4 * 10) := rfl,
  have h2 : current_points = 10 + 40 := by { rw [Nat.mul_def, Nat.add_def], },
  have h3 : current_points = 50 := by { rw Nat.add_def },
  have h4 : additional_points = 100 - 50 := rfl,
  have h5 : additional_points = 50 := by { rw Nat.sub_def },
  have h6 : throw_count = 50 / 25 := rfl,
  show throw_count = 2,
  rw Nat.div_def,
  exact h6
} sorry

end Jerry_throw_count_l678_678263


namespace least_of_consecutive_odds_l678_678556

noncomputable def average_of_consecutive_odds (n : ℕ) (start : ℤ) : ℤ :=
start + (2 * (n - 1))

theorem least_of_consecutive_odds
    (n : ℕ)
    (mean : ℤ)
    (h : n = 30 ∧ mean = 526) : 
    average_of_consecutive_odds 1 (mean * 2 - (n - 1)) = 497 :=
by
  sorry

end least_of_consecutive_odds_l678_678556


namespace smallest_N_unit_digit_l678_678634

noncomputable def smallest_N_sequence := 39

theorem smallest_N_unit_digit (N : ℕ) (nine_numbers_sequence : List ℕ) :
  nine_numbers_sequence.length = 9 ∧ List.head nine_numbers_sequence = some N ∧ List.last nine_numbers_sequence = some 0 ∧ 
  (∀ n in nine_numbers_sequence.tail, ∃ k : ℕ, k^2 ≤ n ∧ n - k^2 ∈ nine_numbers_sequence) ∧ 
  N = smallest_N_sequence → N % 10 = 9 := by
sorry

end smallest_N_unit_digit_l678_678634


namespace sally_savings_l678_678663

theorem sally_savings :
  let cost_parking := 10 in
  let cost_entrance := 55 in
  let cost_meal_pass := 25 in
  let miles_to_sea_world := 165 in
  let miles_per_gallon := 30 in
  let gas_cost_per_gallon := 3 in
  let additional_savings_needed := 95 in
  let total_cost := cost_parking + cost_entrance + cost_meal_pass + 
                    (2 * miles_to_sea_world / miles_per_gallon) * gas_cost_per_gallon in
  let amount_already_saved := total_cost - additional_savings_needed in
  amount_already_saved = 28 :=
by
  sorry

end sally_savings_l678_678663


namespace skiing_finish_protocols_l678_678278

/--
  Seven skiers with numbers 1, 2, ..., 7 started one after another and completed the course, each with a constant speed.
  Each skier was involved in exactly two overtakes. (Each overtake involves exactly two skiers - the one who overtakes
  and the one who is overtaken.) Prove that there can be no more than two different protocols in a race with the described
  properties.
-/
theorem skiing_finish_protocols :
  ∃ (f : list ℕ), f = [3, 2, 1, 6, 7, 4, 5] ∨ f = [3, 4, 1, 2, 7, 6, 5] := sorry

end skiing_finish_protocols_l678_678278


namespace inequality_S_sum_l678_678625

open Finset BigOperators

noncomputable def T (n : ℕ) : ℝ := (n * (n + 1)) / 2

noncomputable def S (n : ℕ) : ℝ := ∑ k in range (n + 1), 1 / T k

theorem inequality_S_sum :
  (∑ k in range 1996, 1 / S k) > 1001 :=
sorry

end inequality_S_sum_l678_678625


namespace trig_eq_solution_l678_678345

theorem trig_eq_solution (x : ℝ) (k : ℤ) (hx1 : sin (4 * x) ≠ 0) (hx2 : cos(2 * x)^4 - sin(2 * x)^4 ≠ 0) : 
    (cos(2 * x)^4 + sin(2 * x)^4) / (cos(2 * x)^4 - sin(2 * x)^4) - (1 / 2) * cos(4 * x) = 
    (sqrt 3 / 2) * arcsin (4 * x) ↔ x = (π / 12) * (3 * k + 1) :=
sorry

end trig_eq_solution_l678_678345


namespace fourth_term_arithmetic_sequence_l678_678562

theorem fourth_term_arithmetic_sequence (a d : ℝ) (h : 2 * a + 2 * d = 12) : a + d = 6 := 
by
  sorry

end fourth_term_arithmetic_sequence_l678_678562


namespace simplified_expression_correct_l678_678281

noncomputable def simplified_expression : ℝ := 0.3 * 0.8 + 0.1 * 0.5

theorem simplified_expression_correct : simplified_expression = 0.29 := by 
  sorry

end simplified_expression_correct_l678_678281


namespace volume_ratio_three_shapes_l678_678487

theorem volume_ratio_three_shapes
  (R : ℝ) (H : ℝ) (h : H = 2 * R) :
  let V_sphere := (4 / 3) * Real.pi * R^3,
      V_cone := (1 / 3) * Real.pi * R^2 * H,
      V_cylinder := Real.pi * R^2 * H in
  V_cylinder / V_sphere = 3 ∧ V_sphere / V_cone = 2 := by
  sorry

end volume_ratio_three_shapes_l678_678487


namespace variance_of_X_l678_678152

noncomputable def X : List (ℚ × ℚ) := [(0, 1/4), (2, 1/2), (4, 1/4)]

def expected_value (X : List (ℚ × ℚ)) : ℚ :=
  X.foldr (λ (x : ℚ × ℚ) acc, acc + x.1 * x.2) 0

def variance (X : List (ℚ × ℚ)) : ℚ :=
  let mean := expected_value X
  X.foldr (λ (x : ℚ × ℚ) acc, acc + (x.1 - mean)^2 * x.2) 0

theorem variance_of_X :
  variance X = 2 :=
by
  sorry

end variance_of_X_l678_678152


namespace count_two_digit_powers_of_three_l678_678980

theorem count_two_digit_powers_of_three : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 :=
by
  sorry

end count_two_digit_powers_of_three_l678_678980


namespace repeating_decimal_to_fraction_l678_678445

theorem repeating_decimal_to_fraction :
  let x := 0.46464646 in x = 46 / 99 :=
by
  sorry

end repeating_decimal_to_fraction_l678_678445


namespace find_y_l678_678307

-- Define the conditions (inversely proportional and sum condition)
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k
def sum_condition (x y : ℝ) : Prop := x + y = 50 ∧ x = 3 * y

-- Given these conditions, prove the value of y when x = -12
theorem find_y (k x y : ℝ)
  (h1 : inversely_proportional x y k)
  (h2 : sum_condition 37.5 12.5)
  (hx : x = -12) :
  y = -39.0625 :=
sorry

end find_y_l678_678307


namespace two_digit_numbers_in_form_3_pow_n_l678_678973

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678973


namespace find_values_of_abc_and_min_dist_l678_678532

noncomputable def check_constants (a b c : ℝ) : Prop :=
  (a = 1) ∧ (b = 2) ∧ (c = -1)

noncomputable def minimum_distance (d : ℝ) : Prop :=
  d = 3 * Real.sqrt 10 / 40

theorem find_values_of_abc_and_min_dist :
  ∃ (a b c d : ℝ), 
  f (x : ℝ) = x^3 + a * x ∧ 
  g (x : ℝ) = x^2 + b * x + c ∧ 
  f 1 = 2 ∧ g 1 = 2 ∧ 
  f'.eval 1 = g'.eval 1 ∧ 
  check_constants a b c ∧ 
  minimum_distance d :=
by
  sorry

end find_values_of_abc_and_min_dist_l678_678532


namespace mode_is_37_median_is_36_l678_678584

namespace ProofProblem

def data_set : List ℕ := [34, 35, 36, 34, 36, 37, 37, 36, 37, 37]

def mode (l : List ℕ) : ℕ := sorry -- Implementing a mode function

def median (l : List ℕ) : ℕ := sorry -- Implementing a median function

theorem mode_is_37 : mode data_set = 37 := 
  by 
    sorry -- Proof of mode

theorem median_is_36 : median data_set = 36 := 
  by
    sorry -- Proof of median

end ProofProblem

end mode_is_37_median_is_36_l678_678584


namespace matrix_product_l678_678098

open Matrix

def sequence_mat (k : Nat) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![(1 : ℕ), 2 * k - 1;
     0, 1]

theorem matrix_product :
  (∏ k in Finset.range 50, sequence_mat (k + 1)) =
  !![(1 : ℕ), 2500;
      0, 1] :=
sorry

end matrix_product_l678_678098


namespace count_two_digit_numbers_l678_678539

theorem count_two_digit_numbers {t u : ℕ} (h₁ : 10 ≤ 10 * t + u) (h₂ : 10 * t + u ≤ 99) (h₃ : t - u ≥ 2) :
  {n : ℕ | 10 ≤ n ∧ n ≤ 99 ∧ 2 ≤ (n / 10) - (n % 10)}.card = 36 :=
sorry

end count_two_digit_numbers_l678_678539


namespace sqrt_12_estimate_l678_678094

theorem sqrt_12_estimate : 3 < Real.sqrt 12 ∧ Real.sqrt 12 < 4 := by
  sorry

end sqrt_12_estimate_l678_678094


namespace k_monotonic_range_k_positive_range_l678_678513

-- Define function f(x)
def f (x : ℝ) (k : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define derivative of f(x)
def f' (x : ℝ) (k : ℝ) : ℝ := 8 * x - k

-- Define function g(x)
def g (x : ℝ) : ℝ := 4 * x - 8 / x

-- Prove the range of k for f(x) to be monotonic on [5, 20]
theorem k_monotonic_range {k : ℝ} :
  (∀ x ∈ Icc (5 : ℝ) 20, f' x k ≥ 0) ∨ (∀ x ∈ Icc (5 : ℝ) 20, f' x k ≤ 0) →
  k ∈ Iic 40 ∪ Ici 160 :=
sorry

-- Prove the range of k for f(x) > 0 on [5, 20]
theorem k_positive_range {k : ℝ} :
  (∀ x ∈ Icc (5 : ℝ) 20, f x k > 0) →
  k < 92 / 5 :=
sorry

end k_monotonic_range_k_positive_range_l678_678513


namespace not_coprime_n_p_minus_one_l678_678612

theorem not_coprime_n_p_minus_one (a b c n p : ℕ) (hp_prime : p.prime) 
  (hn_ge_two : n ≥ 2)
  (hp_div_a2_ab_b2 : p ∣ (a^2 + a * b + b^2))
  (hp_div_an_bn_cn : p ∣ (a^n + b^n + c^n))
  (hp_not_div_add_abc : ¬ (p ∣ (a + b + c))) : 
  ¬ (Nat.gcd n (p - 1) = 1) :=
by
  -- The actual proof is omitted here.
  sorry

end not_coprime_n_p_minus_one_l678_678612


namespace evaluate_expression_eq_l678_678421

theorem evaluate_expression_eq
  (a b : ℝ)
  (h : (a = -real.sqrt 2 ∨ a = real.sqrt 3 ∨ a = real.sqrt 6) ∧
       (b = -real.sqrt 2 ∨ b = real.sqrt 3 ∨ b = real.sqrt 6) ∧
       (a ≠ b)) :
  ((a + b) ^ 2 / real.sqrt 2 = (5 * real.sqrt 2) / 2 - 2 * real.sqrt 3 ∨
   (a + b) ^ 2 / real.sqrt 2 = 4 * real.sqrt 2 - 2 * real.sqrt 6 ∨
   (a + b) ^ 2 / real.sqrt 2 = (9 * real.sqrt 2) / 2 + 6) :=
sorry

end evaluate_expression_eq_l678_678421


namespace find_blanket_rate_l678_678031

theorem find_blanket_rate :
  let cost1 := 3 * 100
  let cost2 := 150
  let num_blankets := 6
  let total_cost := num_blankets * 150
  ∃ x : ℕ, 2 * x = total_cost - cost1 - cost2 ∧ x = 225 := 
by
  let cost1 := 3 * 100
  let cost2 := 150
  let num_blankets := 6
  let total_cost := num_blankets * 150
  use 225
  have h1 : 2 * 225 = 450 := by norm_num
  have h2 : total_cost - cost1 - cost2 = 450 := by norm_num
  exact ⟨h1, by norm_num⟩

end find_blanket_rate_l678_678031


namespace max_red_socks_l678_678748

theorem max_red_socks (r b w : ℕ) (h_total : r + b + w ≤ 2025) 
    (h_probability : (r * (r - 1) + b * (b - 1) + w * (w - 1)) * 3 = (r + b + w) * (r + b + w - 1)) :
    r ≤ 15 :=
begin
  sorry
end

end max_red_socks_l678_678748


namespace sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l678_678697

theorem sqrt_sqrt_of_81_eq_pm3_and_cube_root_self (x : ℝ) : 
  (∃ y : ℝ, y^2 = 81 ∧ (x^2 = y → x = 3 ∨ x = -3)) ∧ (∀ z : ℝ, z^3 = z → (z = 1 ∨ z = -1 ∨ z = 0)) := by
  sorry

end sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l678_678697


namespace minimum_handshakes_l678_678049

def binom (n k : ℕ) : ℕ := n.choose k

theorem minimum_handshakes (n_A n_B k_A k_B : ℕ) (h1 : binom (n_A + n_B) 2 + n_A + n_B = 465)
  (h2 : n_A < n_B) (h3 : k_A = n_A) (h4 : k_B = n_B) : k_A = 15 :=
by sorry

end minimum_handshakes_l678_678049


namespace probability_ratio_l678_678446

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_ratio (p q : ℝ) :
  (choose 10 2 * choose 5 3 * choose 5 2 : ℕ : ℝ) / (choose 50 5 : ℕ : ℝ) = q →
  (10 : ℝ) / (choose 50 5 : ℕ : ℝ) = p →
  q / p = 450 :=
by
  sorry

end probability_ratio_l678_678446


namespace tailor_trimming_l678_678772

theorem tailor_trimming (x : ℝ) (A B : ℝ)
  (h1 : ∃ (L : ℝ), L = 22) -- Original length of a side of the cloth is 22 feet
  (h2 : 6 = 6) -- Feet trimmed from two opposite edges
  (h3 : ∃ (remaining_area : ℝ), remaining_area = 120) -- 120 square feet of cloth remain after trimming
  (h4 : A = 22 - 2 * 6) -- New length of the side after trimming 6 feet from opposite edges
  (h5 : B = 22 - x) -- New length of the side after trimming x feet from the other two edges
  (h6 : remaining_area = A * B) -- Relationship of the remaining area
: x = 10 :=
by
  sorry

end tailor_trimming_l678_678772


namespace circle_cartesian_eq_line_intersect_circle_l678_678524

noncomputable def circle_eq := ∀ (ρ θ : ℝ), ρ = 2 * Real.cos θ → (ρ * Real.cos θ - 1) ^ 2 + (ρ * Real.sin θ) ^ 2 = 1

noncomputable def param_eq (t : ℝ) := (x y : ℝ) → x = 1/2 + (Real.sqrt 3)/2 * t ∧ y = 1/2 + 1/2 * t

theorem circle_cartesian_eq : (x - 1) ^ 2 + y ^ 2 = 1 :=
by sorry

theorem line_intersect_circle (x y : ℝ) (t : ℝ) (ρ θ : ℝ) 
  (h1 : ∀ (ρ θ : ℝ), ρ = 2 * Real.cos θ → (ρ * Real.cos θ - 1) ^ 2 + (ρ * Real.sin θ) ^ 2 = 1)
  (h2 : (x y : ℝ) → x = 1/2 + (Real.sqrt 3)/2 * t ∧ y = 1/2 + 1/2 * t) :
  (x - 1) ^ 2 + y ^ 2 = 1 ∧ | (t1 t2 : ℝ) → t1 * t2 = -1/2 | = 1/2 :=
by sorry

end circle_cartesian_eq_line_intersect_circle_l678_678524


namespace tabitha_honey_days_l678_678678

noncomputable def days_of_honey (cups_per_day servings_per_cup total_servings : ℕ) : ℕ :=
  total_servings / (cups_per_day * servings_per_cup)

theorem tabitha_honey_days :
  let cups_per_day := 3
  let servings_per_cup := 1
  let ounces_container := 16
  let servings_per_ounce := 6
  let total_servings := ounces_container * servings_per_ounce
  days_of_honey cups_per_day servings_per_cup total_servings = 32 :=
by
  sorry

end tabitha_honey_days_l678_678678


namespace initial_birds_l678_678673

theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
sorry

end initial_birds_l678_678673


namespace minimal_un_l678_678628

variable (n : ℕ) (hn : n > 0)

def count_divisibles (d: ℕ) (odd_seq : List ℕ) : ℕ :=
odd_seq.countp (λ x => x % d = 0)

theorem minimal_un :
  ∃ (u_n : ℕ), u_n = 2 * n - 1 ∧
  (∀ (d : ℕ) (hd: d > 0), 
  (∀ (seq : List ℕ),
    (∀ (i : ℕ), i < seq.length → seq.nth i = some (2 * i + 1)) →
    count_divisibles d seq ≥ 
    count_divisibles d (List.range (2 * n)).filter (λ x => x % 2 = 1)
  )) := 
sorry

end minimal_un_l678_678628


namespace expected_value_eight_l678_678342

-- Define the 10-sided die roll outcomes
def outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the value function for a roll outcome
def value (x : ℕ) : ℕ :=
  if x % 2 = 0 then x  -- even value
  else 2 * x  -- odd value

-- Calculate the expected value
def expected_value : ℚ :=
  (1 / 10 : ℚ) * (2 + 2 + 6 + 4 + 10 + 6 + 14 + 8 + 18 + 10)

-- The theorem stating the expected value equals 8
theorem expected_value_eight :
  expected_value = 8 := by
  sorry

end expected_value_eight_l678_678342


namespace min_value_expression_l678_678085

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) (hxy : x + y = 6) : 
  ( (x - 1)^2 / (y - 2) + ( (y - 1)^2 / (x - 2) ) ) >= 8 :=
by 
  sorry

end min_value_expression_l678_678085


namespace three_powers_in_two_digit_range_l678_678996

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l678_678996


namespace find_ellipse_and_slope_l678_678490

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  real.sqrt (1 - (b^2) / (a^2))

noncomputable def ellipse_equation (a b x y : ℝ) : ℝ :=
  (x^2) / (a^2) + (y^2) / (b^2)

theorem find_ellipse_and_slope (a b x y k m : ℝ)
(h1 : 0 < b) (h2 : b < a) 
(h3 : y ≠ k * x + m)
(h4 : ellipse_eccentricity a b = (real.sqrt 3) / 2)
(h5 : ellipse_equation a b 2 1 = 1)
(h6 : ∀ (x1 x2 : ℝ), (4 * k ^ 2 + 2 * k * m + m - 1) = 0 
           → k = -1 / 2) :
ellipse_equation a b x y = 1 ∧ k = -1 / 2 :=
  by {
    sorry
  }

end find_ellipse_and_slope_l678_678490


namespace same_number_of_friends_l678_678274

theorem same_number_of_friends (n : ℕ) (friends : Fin n → Fin n) :
  (∃ i j : Fin n, i ≠ j ∧ friends i = friends j) :=
by
  -- The proof is omitted.
  sorry

end same_number_of_friends_l678_678274


namespace proof_problem1_proof_problem2_l678_678836

noncomputable def problem1 : Prop :=
  sin (- (14 / 3) * Real.pi) + cos ((20 / 3) * Real.pi) + tan (- (53 / 6) * Real.pi) = (-3 - Real.sqrt 3) / 6

noncomputable def problem2 : Prop :=
  tan (675 * Real.pi / 180) - sin (-330 * Real.pi / 180) - cos (960 * Real.pi / 180) = -2

theorem proof_problem1 : problem1 := 
  by
    sorry

theorem proof_problem2 : problem2 := 
  by
    sorry

end proof_problem1_proof_problem2_l678_678836


namespace find_c_and_d_for_continuity_l678_678610

noncomputable def g (x : ℝ) (c d : ℝ) :=
  if x > 1 then c * x + 2
  else if -3 ≤ x ∧ x ≤ 1 then 2 * x - 4
  else 3 * x - d

theorem find_c_and_d_for_continuity (c d : ℝ) :
  (∀ x ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs ((g y c d) - (g x c d)) < ε) ↔ c = -4 ∧ d = 1 :=
by sorry

end find_c_and_d_for_continuity_l678_678610


namespace two_digit_numbers_count_l678_678932

theorem two_digit_numbers_count : 
  let digits := {1, 2, 3, 4, 5} in
  let count := (∑ x in digits, ∑ y in (digits \ {x}), 1) in
  count = 20 :=
by simp [digits]; sorry

end two_digit_numbers_count_l678_678932


namespace inverse_function_condition_l678_678698

theorem inverse_function_condition {a : ℝ} :
  (∃ g : ℝ → ℝ, ∀ x ∈ set.Icc (1 : ℝ) 2, g (f x) = x ∧ f (g x) = f x)
  ↔ a ∈ set.Iic (1 : ℝ) ∪ set.Ici (2 : ℝ) where
  f (x : ℝ) : ℝ := x^2 - 2 * a * x - 3 :=
sorry

end inverse_function_condition_l678_678698


namespace greatest_integer_m_divisor_l678_678335

theorem greatest_integer_m_divisor :
  ∃ m : ℕ, (∀ k : ℕ, (k > m → ¬ (20^k ∣ 50!))) ∧ m = 12 :=
by
  sorry

end greatest_integer_m_divisor_l678_678335


namespace minimize_total_cost_l678_678386

-- Define the conditions
def k := 0.005
def fuel_cost (v : ℝ) : ℝ := k * v^3
def other_cost := 80.0
def distance := 100.0

-- Define the total cost function
def total_cost (v : ℝ) : ℝ := (distance / v) * (fuel_cost v + other_cost)

-- Define the minimization speed
def min_speed : ℝ := 20

-- State the proof problem
theorem minimize_total_cost :
  (∀ v > 0, (total_cost v) ≥ (total_cost min_speed)) ∧ fuel_cost v = k*v^3 ∧ ∀ (v: ℝ), v = 20 ↔ (total_cost v = 600) :=
by
  sorry

end minimize_total_cost_l678_678386


namespace min_total_penalty_l678_678752

noncomputable def min_penalty (B W R : ℕ) : ℕ :=
  min (B * W) (min (2 * W * R) (3 * R * B))

theorem min_total_penalty (B W R : ℕ) :
  min_penalty B W R = min (B * W) (min (2 * W * R) (3 * R * B)) := by
  sorry

end min_total_penalty_l678_678752


namespace increasing_interval_a_l678_678558

def f (x : ℝ) := (real.sqrt 3) * real.sin (2 * x) + 3 * real.sin x^2 + real.cos x^2

theorem increasing_interval_a (a : ℝ) : (∀ x y : ℝ, -a ≤ x ∧ x < y ∧ y ≤ a → f x < f y) ↔ (0 < a ∧ a ≤ real.pi / 6) :=
sorry

end increasing_interval_a_l678_678558


namespace part_a_solution_part_b_solution_l678_678348

noncomputable def partition_exists_degree_2 : Prop :=
  ∃ black white : Set ℝ,
    black ∪ white = Set.Icc 0 1 ∧
    black ∩ white = ∅ ∧
    (∀ p : Polynomial ℝ, p.degree ≤ 2 → 
      ((∑ segment in black, p.eval segment.end - p.eval segment.start)
        = (∑ segment in white, p.eval segment.end - p.eval segment.start)))

noncomputable def partition_exists_degree_1995 : Prop :=
  ∃ black white : Set ℝ,
    black ∪ white = Set.Icc 0 1 ∧
    black ∩ white = ∅ ∧
    (∀ p : Polynomial ℝ, p.degree ≤ 1995 → 
      ((∑ segment in black, p.eval segment.end - p.eval segment.start)
        = (∑ segment in white, p.eval segment.end - p.eval segment.start)))

-- Statements without proofs
theorem part_a_solution : partition_exists_degree_2 := sorry

theorem part_b_solution : partition_exists_degree_1995 := sorry

end part_a_solution_part_b_solution_l678_678348


namespace intersection_points_form_square_l678_678707

variables {A B C D M N K L O : Type}
-- Assume A, B, C, D are the vertices of the square
-- Assume M, N, K, L are points of intersection of the lines with the sides of the square
-- Assume O is the center of the square
-- Assume lines are perpendicular and pass through O
-- Assume appropriate geometric definitions and axioms for square and intersection points

-- Theorem statement to be proven
theorem intersection_points_form_square 
  (is_square : is_square A B C D)
  (center_O : is_center A B C D O)
  (lines_perpendicular_at_O : perpendicular_lines_through_center O)
  (intersects_sides : intersect_sides A B C D M N K L)
  : is_square M N K L := 
sorry

end intersection_points_form_square_l678_678707


namespace g_min_max_value_l678_678124

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

-- Define the domain constraints
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 9

-- Define g(x)
def g (x : ℝ) : ℝ := f x ^ 2 + f (x ^ 2)

-- Prove the minimum value of g(x) is 6 and the maximum value is 13
theorem g_min_max_value : 
  (∀ x : ℝ, domain x → g x ≥ 6) ∧ (∃ x : ℝ, domain x ∧ g x = 6) ∧
  (∀ x : ℝ, domain x → g x ≤ 13) ∧ (∃ x : ℝ, domain x ∧ g x = 13) :=
by
  sorry

end g_min_max_value_l678_678124


namespace count_yellow_highlighters_l678_678571

-- Definitions of the conditions
def pink_highlighters : ℕ := 9
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 22

-- Definition based on the question
def yellow_highlighters : ℕ := total_highlighters - (pink_highlighters + blue_highlighters)

-- The theorem to prove the number of yellow highlighters
theorem count_yellow_highlighters : yellow_highlighters = 8 :=
by
  -- Proof omitted as instructed
  sorry

end count_yellow_highlighters_l678_678571


namespace bob_overtime_pay_rate_l678_678833

theorem bob_overtime_pay_rate :
  let regular_pay_rate := 5
  let total_hours := (44, 48)
  let total_pay := 472
  let overtime_hours (hours : Nat) := max 0 (hours - 40)
  let regular_hours (hours : Nat) := min 40 hours
  let total_regular_hours := regular_hours 44 + regular_hours 48
  let total_regular_pay := total_regular_hours * regular_pay_rate
  let total_overtime_hours := overtime_hours 44 + overtime_hours 48
  let total_overtime_pay := total_pay - total_regular_pay
  let overtime_pay_rate := total_overtime_pay / total_overtime_hours
  overtime_pay_rate = 6 := by sorry

end bob_overtime_pay_rate_l678_678833


namespace percentage_of_original_price_l678_678308
-- Define the original price and current price in terms of real numbers
def original_price : ℝ := 25
def current_price : ℝ := 20

-- Lean statement to verify the correctness of the percentage calculation
theorem percentage_of_original_price :
  (current_price / original_price) * 100 = 80 := 
by
  sorry

end percentage_of_original_price_l678_678308


namespace find_Finley_age_l678_678659

variable (Roger Jill Finley : ℕ)
variable (Jill_age : Jill = 20)
variable (Roger_age : Roger = 2 * Jill + 5)
variable (Finley_condition : 15 + (Roger - Jill) = Finley - 30)

theorem find_Finley_age : Finley = 55 :=
by
  sorry

end find_Finley_age_l678_678659


namespace birdhouse_total_cost_l678_678411

def num_small_birdhouses : ℕ := 3
def num_large_birdhouses : ℕ := 2
def small_birdhouse_planks : ℕ := 7
def small_birdhouse_nails : ℕ := 20
def large_birdhouse_planks : ℕ := 10
def large_birdhouse_nails : ℕ := 36
def nail_cost : ℚ := 0.05
def small_plank_cost : ℚ := 3
def large_plank_cost : ℚ := 5
def bulk_discount_threshold : ℕ := 100
def discount_rate : ℚ := 0.10

theorem birdhouse_total_cost : 
  let total_small_planks := (num_small_birdhouses * small_birdhouse_planks),
      total_large_planks := (num_large_birdhouses * large_birdhouse_planks),
      total_small_nails := (num_small_birdhouses * small_birdhouse_nails),
      total_large_nails := (num_large_birdhouses * large_birdhouse_nails),
      total_nails := total_small_nails + total_large_nails,
      total_plank_cost := (total_small_planks * small_plank_cost) + (total_large_planks * large_plank_cost),
      nails_cost_before_discount := total_nails * nail_cost,
      nails_discount := if total_nails > bulk_discount_threshold then nails_cost_before_discount * discount_rate else 0,
      total_nail_cost := nails_cost_before_discount - nails_discount,
      total_cost := total_plank_cost + total_nail_cost
  in total_cost = 168.94 := sorry

end birdhouse_total_cost_l678_678411


namespace rita_hours_per_month_l678_678210

theorem rita_hours_per_month :
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  let h_remaining := t - h_completed
  let h := h_remaining / m
  h = 220
:= by 
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  have h_remaining := t - h_completed
  have h := h_remaining / m
  sorry

end rita_hours_per_month_l678_678210


namespace coin_order_l678_678267

-- Define the coins
inductive Coin
| A | B | C | D | E | F

open Coin

-- Define the conditions as a total order relation.
axiom condition_1 : D > B
axiom condition_2 : D > C
axiom condition_3 : A > F
axiom condition_4 : C > A
axiom condition_5 : C > E
axiom condition_6 : A > F
axiom condition_7 : F > E
axiom condition_8 : D > B
axiom condition_9 : D > C
axiom condition_10 : B > E

-- Theorem to prove the correct order
theorem coin_order : [D, B, C, A, F, E] = [D, B, C, A, F, E] := by
  sorry

end coin_order_l678_678267


namespace price_tags_advantages_l678_678642

/-- Given that a seller has attached price tags to all products,
    this strategy provides several advantages:
    1. Simplifies the purchasing process for buyers,
    2. Reduces the requirement for seller and personnel,
    3. Acts as an additional advertising method,
    4. Increases trust and perceived value.
-/
theorem price_tags_advantages
    (attached_price_tags : ∀ (product : Product), Product.HasPriceTag product) :
    (∀ (buyer : Buyer), buyer.PurchaseProcessSimplified) ∧
    (ReducedSellerRequirement_and_StaffWorkload) ∧
    (AdditionalAdvertisingMethod) ∧
    (IncreasedTrust_and_PerceivedValue) :=
sorry

end price_tags_advantages_l678_678642


namespace find_principal_l678_678002

-- Define the conditions
def interest_rate : ℝ := 0.05
def time_period : ℕ := 10
def interest_less_than_principal : ℝ := 3100

-- Define the principal
def principal : ℝ := 6200

-- The theorem statement
theorem find_principal :
  ∃ P : ℝ, P - interest_less_than_principal = P * interest_rate * time_period ∧ P = principal :=
by
  sorry

end find_principal_l678_678002


namespace merchant_profit_percentage_l678_678346

def cost_price : ℝ := 100
def markup_percentage : ℝ := 40 / 100
def discount_percentage : ℝ := 10 / 100

def marked_price : ℝ := cost_price * (1 + markup_percentage)
def discount_amount : ℝ := marked_price * discount_percentage
def selling_price : ℝ := marked_price - discount_amount
def profit : ℝ := selling_price - cost_price
def profit_percentage : ℝ := (profit / cost_price) * 100

theorem merchant_profit_percentage : profit_percentage = 26 := by
  -- The detailed proof steps would go here, but we skip it for now
  sorry

end merchant_profit_percentage_l678_678346


namespace fourth_term_is_six_l678_678564

-- Definitions from the problem
variables (a d : ℕ)

-- Condition that the sum of the third and fifth terms is 12
def sum_third_fifth_eq_twelve : Prop := (a + 2 * d) + (a + 4 * d) = 12

-- The fourth term of the arithmetic sequence
def fourth_term : ℕ := a + 3 * d

-- The theorem we need to prove
theorem fourth_term_is_six (h : sum_third_fifth_eq_twelve a d) : fourth_term a d = 6 := by
  sorry

end fourth_term_is_six_l678_678564


namespace triangle_count_from_10_points_l678_678679

/--
Given 10 distinct points on the circumference of a circle,
the number of different triangles such that no two triangles share the same side.
-/
theorem triangle_count_from_10_points : (Nat.choose 10 3) = 120 := 
sorry

end triangle_count_from_10_points_l678_678679


namespace _l678_678139

noncomputable theorem acute_triangle_inequality 
  (A B C : ℝ)
  (h_acute : A + B + C = π)  -- Sum of angles in a triangle
  (h_acute_A : A < π / 2)
  (h_acute_B : B < π / 2)
  (h_acute_C : C < π / 2) :
  (cos A / cos B) ^ 2 + (cos B / cos C) ^ 2 + (cos C / cos A) ^ 2 + 8 * cos A * cos B * cos C ≥ 4 := 
by
  sorry

end _l678_678139


namespace functional_equation_solution_l678_678626

theorem functional_equation_solution {f : ℝ → ℝ}
  (h : ∀ x y : ℝ, f(x) * f(y) - f(x * y) = x^2 + y^2) :
  (∃ n s : ℕ, n = 1 ∧ s = 5 ∧ n * s = 5) :=
by
  sorry

end functional_equation_solution_l678_678626


namespace car_tank_capacity_is_12_gallons_l678_678711

noncomputable def truck_tank_capacity : ℕ := 20
noncomputable def truck_tank_half_full : ℕ := truck_tank_capacity / 2
noncomputable def car_tank_third_full (car_tank_capacity : ℕ) : ℕ := car_tank_capacity / 3
noncomputable def total_gallons_added : ℕ := 18

theorem car_tank_capacity_is_12_gallons (car_tank_capacity : ℕ) 
    (h1 : truck_tank_half_full + (car_tank_third_full car_tank_capacity) + 18 = truck_tank_capacity + car_tank_capacity) 
    (h2 : total_gallons_added = 18) : car_tank_capacity = 12 := 
by
  sorry

end car_tank_capacity_is_12_gallons_l678_678711


namespace unique_intersection_l678_678583

theorem unique_intersection (a : ℝ) (h : 2 * a = -1) :
  ∃! x, 2 * a = abs (x - a) - 1 :=
by
  sorry

end unique_intersection_l678_678583


namespace identify_clothes_l678_678798

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l678_678798


namespace double_probability_correct_l678_678747

def is_double (a : ℕ × ℕ) : Prop := a.1 = a.2

def total_dominoes : ℕ := 13 * 13

def double_count : ℕ := 13

def double_probability := (double_count : ℚ) / total_dominoes

theorem double_probability_correct : double_probability = 13 / 169 := by
  sorry

end double_probability_correct_l678_678747


namespace problem_statement_l678_678694

theorem problem_statement (n : ℕ) (p : ℕ) (k : ℕ) (x : ℕ → ℕ)
  (h1 : n ≥ 3)
  (h2 : ∀ i, 1 ≤ i → i ≤ n → x i < 2 * x 1)
  (h3 : ∀ i j, 1 ≤ i → i < j → j ≤ n → x i < x j)
  (hp : p.Prime)
  (hk : 0 < k)
  (hP : ∀ (P : ℕ), P = ∏ i in Finset.range n.succ \ {0}, x i → P % p^k = 0) :
  (∏ i in Finset.range n.succ \ {0}, x i) / p^k ≥ Nat.factorial n :=
sorry

end problem_statement_l678_678694


namespace number_of_skew_edges_to_AA₁_l678_678589

-- Define the concept of a quadrangular pyramid
structure Edge (V : Type) := (start : V) (end : V)
structure Pyramid (V : Type) := 
  (vertices : List V)
  (edges : List (Edge V))
  (edge_AA₁ : Edge V)

-- Define skewness relationship
def is_skew {V : Type} (e1 e2 : Edge V) : Prop := sorry

-- Example vertices and edges
axiom A A₁ B B₁ C C₁ D D₁ : ℕ  -- just using ℕ as a placeholder for vertices type
noncomputable def quadrangularPyramid : Pyramid ℕ := {
  vertices := [A, B, C, D, A₁, B₁, C₁, D₁],
  edges := [
    Edge.mk A B, Edge.mk B C, Edge.mk C D, Edge.mk D A,
    Edge.mk A A₁, Edge.mk B B₁, Edge.mk C C₁, Edge.mk D D₁,
    Edge.mk A₁ B₁, Edge.mk B₁ C₁, Edge.mk C₁ D₁, Edge.mk D₁ A₁
  ],
  edge_AA₁ := Edge.mk A A₁
}

-- Define the theorem
theorem number_of_skew_edges_to_AA₁ : 
  ∃ (edges_skew_to_AA₁ : List (Edge ℕ)), 
    edges_skew_to_AA₁.length = 4 ∧
    ∀ (e ∈ edges_skew_to_AA₁), is_skew quadrangularPyramid.edge_AA₁ e :=
sorry

end number_of_skew_edges_to_AA₁_l678_678589


namespace three_powers_in_two_digit_range_l678_678997

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l678_678997


namespace remainder_of_6_pow_50_mod_215_l678_678456

theorem remainder_of_6_pow_50_mod_215 :
  (6 ^ 50) % 215 = 36 := 
sorry

end remainder_of_6_pow_50_mod_215_l678_678456


namespace oil_bill_january_l678_678353

-- Define the problem in Lean
theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 := 
sorry

end oil_bill_january_l678_678353


namespace alexis_total_sewing_time_l678_678774

-- Define the time to sew a skirt and a coat
def t_skirt : ℕ := 2
def t_coat : ℕ := 7

-- Define the numbers of skirts and coats
def n_skirts : ℕ := 6
def n_coats : ℕ := 4

-- Define the total time
def total_time : ℕ := t_skirt * n_skirts + t_coat * n_coats

-- State the theorem
theorem alexis_total_sewing_time : total_time = 40 :=
by
  -- the proof would go here; we're skipping the proof as per instructions
  sorry

end alexis_total_sewing_time_l678_678774


namespace mutated_frog_percentage_l678_678109

theorem mutated_frog_percentage 
  (extra_legs : ℕ) 
  (two_heads : ℕ) 
  (bright_red : ℕ) 
  (normal_frogs : ℕ) 
  (h_extra_legs : extra_legs = 5) 
  (h_two_heads : two_heads = 2) 
  (h_bright_red : bright_red = 2) 
  (h_normal_frogs : normal_frogs = 18) 
  : ((extra_legs + two_heads + bright_red) * 100 / (extra_legs + two_heads + bright_red + normal_frogs)).round = 33 := 
by
  sorry

end mutated_frog_percentage_l678_678109


namespace change_max_value_l678_678720

-- Define the quadratic polynomial
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Conditions given in the problem
def first_condition (a b c : ℝ) : Prop :=
  let f_max1 := λ c, -b^2 / (4 * (a + 2)) + c
  in f_max1 c = -b^2 / (4 * a) + c + 10

def second_condition (a b c : ℝ) : Prop :=
  let f_max2 := λ c, -b^2 / (4 * (a - 5)) + c
  in f_max2 c = -b^2 / (4 * a) + c - 15 / 2

-- Proof goal based on the conditions
theorem change_max_value (a b c : ℝ)
  (h1 : first_condition a b c)
  (h2 : second_condition a b c) :
  let f_max3 := λ c, -b^2 / (4 * (a + 3)) + c
      f_max_orig := λ c, -b^2 / (4 * a) + c
  in f_max3 c - f_max_orig c = 45 / 2 :=
by
  sorry

end change_max_value_l678_678720


namespace sequence_next_number_l678_678217

theorem sequence_next_number :
  ∃ n, (2, 16, 4, 14, 6, 12, n, 10) = (2, 16, 4, 14, 6, 12, 8, 10) :=
by
  use 8
  sorry

end sequence_next_number_l678_678217


namespace longest_side_in_triangle_l678_678568

noncomputable theory
open Real

theorem longest_side_in_triangle (A B C : ℝ) (a b c : ℝ)
  (hB : B = 135) (hC : C = 15) (ha : a = 5) (hAngleSum : A = 180 - (B + C)) 
  (hSineRule : b = (a * sin B) / (sin A)) : b = 5 * sqrt 2 := 
sorry

end longest_side_in_triangle_l678_678568


namespace two_digit_powers_of_three_l678_678947

theorem two_digit_powers_of_three : {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.finite ∧ ({n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}).to_finset.card = 2 := by
  sorry

end two_digit_powers_of_three_l678_678947


namespace general_formula_of_arithmetic_seq_sum_of_bn_terms_l678_678489

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

-- Given an arithmetic sequence with common difference 2 and certain terms forming a geometric sequence
-- Statement 1: General formula of the arithemetic sequence
theorem general_formula_of_arithmetic_seq (a : ℕ → ℤ) (h_arith : arithmetic_sequence a 2)
  (h_geo : geometric_sequence (a 2) (a 3) (a 6)) :
  ∀ n, a n = 2 * n - 3 :=
by sorry

-- Statement 2: Sum of the first n terms of sequence {bn}
theorem sum_of_bn_terms (a : ℕ → ℤ) (h_arith : arithmetic_sequence a 2)
  (h_geo : geometric_sequence (a 2) (a 3) (a 6)) :
  ∀ n, (∑ i in finset.range n, 1 / (a (i + 1) * a i) : ℝ) = -n / (2 * n - 1) :=
by sorry

end general_formula_of_arithmetic_seq_sum_of_bn_terms_l678_678489


namespace xiao_ming_valid_paths_final_valid_paths_l678_678406

-- Definitions from conditions
def paths_segments := ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
def initial_paths := 256
def invalid_paths := 64

-- Theorem statement
theorem xiao_ming_valid_paths : initial_paths - invalid_paths = 192 :=
by sorry

theorem final_valid_paths : 192 * 2 = 384 :=
by sorry

end xiao_ming_valid_paths_final_valid_paths_l678_678406


namespace minimum_value_of_x_plus_y_l678_678544

noncomputable def minValueSatisfies (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y + x * y = 2 → x + y ≥ 2 * Real.sqrt 3 - 2

theorem minimum_value_of_x_plus_y (x y : ℝ) : minValueSatisfies x y :=
by sorry

end minimum_value_of_x_plus_y_l678_678544


namespace shoe_cost_l678_678376

variable (S : ℝ)

-- Define the conditions
variable (bag_cost : ℝ) (sock_cost : ℝ) (num_socks : ℕ) (total_paid : ℝ)
variable (discount_rate : ℝ) (threshold : ℝ)

def total_cost_before_discount :=
  S + num_socks * sock_cost + bag_cost

def total_cost_after_discount :=
  total_cost_before_discount - discount_rate * (total_cost_before_discount - threshold)

theorem shoe_cost (h1: bag_cost = 42)
                 (h2: sock_cost = 2)
                 (h3: num_socks = 2)
                 (h4: total_paid = 118)
                 (h5: discount_rate = 0.10)
                 (h6: threshold = 100)
                 (h7: total_cost_after_discount = total_paid) : 
  S = 74 :=
by
  sorry

end shoe_cost_l678_678376


namespace sum_of_floor_values_l678_678075

noncomputable def arithmetic_sequence (n : ℕ) : ℝ :=
  if n % 2 = 1 then 2 + (3:ℝ) * (n / 2)
  else (3.2:ℝ) + (3:ℝ) * ((n - 1) / 2)

theorem sum_of_floor_values :
  ∑ n in Finset.range 68, Int.floor (arithmetic_sequence n) = 3536 :=
by
  sorry

end sum_of_floor_values_l678_678075


namespace time_to_hit_ground_l678_678294

def height (t : ℝ) : ℝ := -9 * t^2 - 15 * t + 63

theorem time_to_hit_ground : ∃ t : ℝ, height t = 0 ∧ t ≈ 2.33 :=
by
  sorry

end time_to_hit_ground_l678_678294


namespace initial_integer_l678_678403

theorem initial_integer (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_integer_l678_678403


namespace ellipse_eccentricity_l678_678491

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : ∀ P Q : ℝ × ℝ, (P = (-a, 0)) → (Q = (a, 0)) → 
    let line_eq := (fun x y : ℝ => x / a - y / b + 2 = 0) in
    let dist_to_line := (fun x y : ℝ => abs ((b * x - a * y + 2 * a * b) / sqrt(a^2 + b^2))) in 
    dist_to_line 0 0 = a) :
  let e := sqrt(1 - (b^2 / a^2)) in
  e = sqrt(2 / 3) :=
by
  sorry

end ellipse_eccentricity_l678_678491


namespace absent_children_count_l678_678640

theorem absent_children_count : ∀ (total_children present_children absent_children bananas : ℕ), 
  total_children = 260 → 
  bananas = 2 * total_children → 
  bananas = 4 * present_children → 
  present_children + absent_children = total_children →
  absent_children = 130 :=
by
  intros total_children present_children absent_children bananas h1 h2 h3 h4
  sorry

end absent_children_count_l678_678640


namespace size_of_M_eq_size_of_N_l678_678624

variables (n : ℕ)

def M : set ℕ := { x | ∃ (l : list ℕ), l.count 1 = n ∧ l.count 2 = n ∧ all (λ d, d = 1 ∨ d = 2) l ∧ x = digits 10 (l.to_nat) }

def N : set ℕ := { y | ∃ (l : list ℕ), (∀ d, d ∈ l → d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4) ∧ l.count 1 = l.count 2 ∧ length l = n }

theorem size_of_M_eq_size_of_N : |M n| = |N n| :=
by sorry

end size_of_M_eq_size_of_N_l678_678624


namespace expectation_of_X_l678_678118

-- Conditions:
-- Defect rate of the batch of products is 0.05
def defect_rate : ℚ := 0.05

-- 5 items are randomly selected for quality inspection
def n : ℕ := 5

-- The probability of obtaining a qualified product in each trial
def P : ℚ := 1 - defect_rate

-- Question:
-- The random variable X, representing the number of qualified products, follows a binomial distribution.
-- Expectation of X
def expectation_X : ℚ := n * P

-- Prove that the mathematical expectation E(X) is equal to 4.75
theorem expectation_of_X :
  expectation_X = 4.75 := 
sorry

end expectation_of_X_l678_678118


namespace f_correct_f_monotonic_decreasing_f_lambda_solutions_l678_678144

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 then 0
  else if 0 < x ∧ x < 1 then 2^x / (4^x + 1)
  else if -1 < x ∧ x < 0 then -2^x / (4^x + 1)
  else 0

theorem f_correct (x : ℝ) (h1 : x ∈ set.Ioo (-1) 1) :
  f(x) = if x = 0 then 0
         else if 0 < x ∧ x < 1 then 2^x / (4^x + 1)
         else if -1 < x ∧ x < 0 then -2^x / (4^x + 1)
         else 0 :=
begin
  sorry
end

theorem f_monotonic_decreasing (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < 1) :
  f(x1) > f(x2) :=
begin
  sorry
end

theorem f_lambda_solutions (λ : ℝ) :
  (λ ∈ (set.Ioo (2/5) (1/2)) ∨ λ ∈ (set.Ioo (-1 / 2) (-2 / 5)) ∨ λ = 0) ↔
  ∃ x : ℝ, x ∈ set.Ioo (-1) 1 ∧ f(x) = λ :=
begin
  sorry
end

end f_correct_f_monotonic_decreasing_f_lambda_solutions_l678_678144


namespace monotonic_decreasing_on_pi_third_l678_678158

noncomputable def f (x : ℝ) : ℝ := cos x ^ 2 - sin x ^ 2

theorem monotonic_decreasing_on_pi_third : MonotonicDecreasingOn (cos ∘ (λ x, 2 * x)) (Ioo 0 (π / 3)) :=
sorry

end monotonic_decreasing_on_pi_third_l678_678158


namespace four_digit_odd_numbers_count_l678_678668

open Nat

def set_six_numbers : Set ℕ := {0, 1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem four_digit_odd_numbers_count :
  ∃ (count : ℕ), count = 144 ∧
    (∀ l : List ℕ, l.length = 4 → l.toFinset ⊆ set_six_numbers →
      (List.nth l 3).isSome ∧ is_odd (List.nthLe l 3 sorry) →
      count = (3 * 4 * (Perms ({0, 1, 2, 3, 4} \ ({List.nthLe l 3 sorry})).length).length)) :=
sorry

end four_digit_odd_numbers_count_l678_678668


namespace two_digit_numbers_in_form_3_pow_n_l678_678966

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678966


namespace max_percentage_difference_is_in_April_l678_678290

def sales_data (month : String) : ℕ × ℕ :=
  match month with
  | "January" => (5, 4)
  | "February" => (6, 4)
  | "March" => (5, 5)
  | "April" => (7, 4)
  | "May" => (3, 5)
  | _ => (0, 0)

def percentage_difference (R S : ℕ) : ℚ :=
  if R < S then (S - R) / R else (R - S) / S

theorem max_percentage_difference_is_in_April :
  let january_diff  := percentage_difference (prod.fst (sales_data "January")) (prod.snd (sales_data "January"))
  let february_diff := percentage_difference (prod.fst (sales_data "February")) (prod.snd (sales_data "February"))
  let march_diff    := percentage_difference (prod.fst (sales_data "March")) (prod.snd (sales_data "March"))
  let april_diff    := percentage_difference (prod.fst (sales_data "April")) (prod.snd (sales_data "April"))
  let may_diff      := percentage_difference (prod.fst (sales_data "May")) (prod.snd (sales_data "May"))
  april_diff > january_diff ∧
  april_diff > february_diff ∧
  april_diff > march_diff ∧
  april_diff > may_diff := 
sorry

end max_percentage_difference_is_in_April_l678_678290


namespace find_theta_l678_678447

theorem find_theta (θ : Real) : 
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x ^ 3 * Real.sin θ + x ^ 2 * Real.cos θ - x * (1 - x) + (1 - x) ^ 2 * Real.sin θ > 0) → 
  Real.sin θ > 0 → 
  Real.cos θ + Real.sin θ > 0 → 
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intro θ_range all_x_condition sin_pos cos_sin_pos
  sorry

end find_theta_l678_678447


namespace entree_cost_14_l678_678171

theorem entree_cost_14 (D E : ℝ) (h1 : D + E = 23) (h2 : E = D + 5) : E = 14 :=
sorry

end entree_cost_14_l678_678171


namespace sin_difference_theorem_l678_678137

open Real

-- Define the given conditions
def conditions (α : ℝ) : Prop := 
  π / 2 < α ∧ α < π ∧ 3 * sin (2 * α) = 2 * cos α

-- Define the target expression to be proven
def target_expression (α : ℝ) : ℝ := 
  sin (α - 9 * π / 2)

-- State the theorem to be proven
theorem sin_difference_theorem (α : ℝ) (h : conditions α) : target_expression α = 2 * sqrt 2 / 3 :=
by
  sorry

end sin_difference_theorem_l678_678137


namespace find_n_l678_678148

theorem find_n (n : ℕ) (hn : (n - 2) * (n - 3) / 12 = 14 / 3) : n = 10 := by
  sorry

end find_n_l678_678148


namespace no_perfect_squares_l678_678852

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def N1 : ℕ := readNat "3" * 10^99 + readNat "3"   -- Representing 333333...3
noncomputable def N2 : ℕ := readNat "6" * 10^99 + readNat "6"   -- Representing 666666...6
noncomputable def N3 : ℕ := 1 + (5 * 10^98 + 1)       -- Representing 151515...15
noncomputable def N4 : ℕ := 2 + (1 + (2 * 10^98 + 1) * 10) -- Representing 212121...21
noncomputable def N5 : ℕ := 2 + (7 * 10^98 + 7)       -- Representing 272727...27

theorem no_perfect_squares :
  ¬ isPerfectSquare N1 ∧
  ¬ isPerfectSquare N2 ∧
  ¬ isPerfectSquare N3 ∧
  ¬ isPerfectSquare N4 ∧
  ¬ isPerfectSquare N5 :=
by
  sorry

end no_perfect_squares_l678_678852


namespace percent_decrease_l678_678203

theorem percent_decrease(call_cost_1980 call_cost_2010 : ℝ) (h₁ : call_cost_1980 = 50) (h₂ : call_cost_2010 = 5) :
  ((call_cost_1980 - call_cost_2010) / call_cost_1980 * 100) = 90 :=
by
  sorry

end percent_decrease_l678_678203


namespace num_black_cars_l678_678322

theorem num_black_cars (total_cars : ℕ) (one_third_blue : ℚ) (one_half_red : ℚ) 
  (h1 : total_cars = 516) (h2 : one_third_blue = 1/3) (h3 : one_half_red = 1/2) :
  total_cars - (total_cars * one_third_blue + total_cars * one_half_red) = 86 :=
by
  sorry

end num_black_cars_l678_678322


namespace cosine_translation_phase_shift_l678_678298

variable (k : ℤ)
#check (π : ℝ) /- Check to ensure that π exists -/

theorem cosine_translation_phase_shift :
  ∀ (φ : ℝ), 
  cos (2 * (x - π / 6) + φ) = cos (2 * x - π / 3 + φ) 
  ∧ (∀ x : ℝ, cos (2 * x - π / 3 + φ) = - cos (-(2 * x - π / 3 + φ))) ->
  (∃ k : ℤ, φ = k * π + 5 * π / 6) :=
  sorry

end cosine_translation_phase_shift_l678_678298


namespace tangent_lengths_equal_l678_678677

-- Define the geometric setup
structure RightTriangle (D E F : Type) :=
  (DE DF EF : ℝ)
  (right_angle_at_E : ∃ (G : E), ∠DEF = 90)
  (DF_sqrt_85 : DF = Real.sqrt 85)
  (DE_eq_7 : DE = 7)
  (circle_tangent_to_DF_EF : 
    ∃ (C : Type) (center_on_DE : C ∈ DE) (Q : Type) (tangency_points : Q ∈ DF ∧ Q ∈ EF))

-- Define the proof goal
theorem tangent_lengths_equal (DEF : RightTriangle) : 
  (∃ (C : Type) (Q : Type), 
   C ∈ DEF.D.1 ∧ Q ∈ DEF.EF ∧ Q ∈ DEF.DF ∧
   DEF.EF = Real.sqrt 36) →
  DEF.EF = 6 → 
  DEF.EF = DEF.DF :=
by
  sorry

end tangent_lengths_equal_l678_678677


namespace quadratic_roots_exist_intersection_points_y_axis_symmetry_l678_678923

-- Part (1): Prove the quadratic function intersects x-axis at two points for any k.
theorem quadratic_roots_exist (k : ℝ) : 
  let Δ := (-2*k)^2 - 4 * 1 * (-1) in
  Δ > 0 := by
  sorry

-- Part (2): Given y-axis as axis of symmetry for the quadratic function, find intersection points with x-axis.
theorem intersection_points_y_axis_symmetry :
  let k := 0 in
  let y := λ x : ℝ, x^2 - 2*k*x - 1 in
  y 1 = 0 ∧ y (-1) = 0 := by
  sorry

end quadratic_roots_exist_intersection_points_y_axis_symmetry_l678_678923


namespace sqrt_41_40_39_38_plus_1_l678_678071

theorem sqrt_41_40_39_38_plus_1 : Real.sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := by
  sorry

end sqrt_41_40_39_38_plus_1_l678_678071


namespace num_two_digit_powers_of_3_l678_678938

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678938


namespace minimal_inverse_presses_l678_678260

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem minimal_inverse_presses (x : ℚ) (h : x = 50) : 
  ∃ n, n = 2 ∧ (reciprocal^[n] x = x) :=
by
  sorry

end minimal_inverse_presses_l678_678260


namespace special_collection_books_end_l678_678033

def special_collection_books_at_end_of_month (TotalInitialBooks LoanedBooks : ℕ) (PercentReturned : ℝ) : ℕ :=
  let BooksReturned := LoanedBooks * PercentReturned
  let BooksNotReturned := LoanedBooks - BooksReturned
  let FinalBooks := TotalInitialBooks - BooksNotReturned
  FinalBooks.toNat

theorem special_collection_books_end {TotalInitialBooks LoanedBooks : ℕ} {PercentReturned : ℝ} :
  TotalInitialBooks = 300 → LoanedBooks = 160 → PercentReturned = 0.65 →
  special_collection_books_at_end_of_month TotalInitialBooks LoanedBooks PercentReturned = 244 :=
by
  intros hTI hL hP
  rw [hTI, hL, hP]
  unfold special_collection_books_at_end_of_month
  sorry

end special_collection_books_end_l678_678033


namespace axisymmetric_squares_l678_678055

theorem axisymmetric_squares :
  ∀ (initial_shaded : ℕ), initial_shaded = 3 →
  ∃ (additional_needed : ℕ), additional_needed = 6 ∧
    -- some condition expressing the rectangles required symmetry
    sorry :=
begin
  intros initial_shaded h_initial,
  use 6,
  split,
  { refl },
  { -- Here would go the condition expressing achieving axis symmetry.
    sorry }
end

end axisymmetric_squares_l678_678055


namespace polar_coordinates_of_point_l678_678080

theorem polar_coordinates_of_point :
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  r = 4 ∧ theta = Real.pi / 3 :=
by
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  have h_r : r = 4 := by {
    -- Calculation for r
    sorry
  }
  have h_theta : theta = Real.pi / 3 := by {
    -- Calculation for theta
    sorry
  }
  exact ⟨h_r, h_theta⟩

end polar_coordinates_of_point_l678_678080


namespace odd_number_of_divisors_implies_perfect_square_l678_678649

theorem odd_number_of_divisors_implies_perfect_square (n : ℕ) (hn_pos : 0 < n) (hn_odd_divisors : odd (divisors n).card) :
  ∃ d : ℕ, d * d = n := 
sorry

end odd_number_of_divisors_implies_perfect_square_l678_678649


namespace evaluate_expr_correct_l678_678444

def evaluate_expr : Prop :=
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25)

theorem evaluate_expr_correct : evaluate_expr :=
by
  sorry

end evaluate_expr_correct_l678_678444


namespace symmetric_circle_equation_l678_678450

theorem symmetric_circle_equation :
  ∀ (a b : ℝ), 
    (∀ (x y : ℝ), (x-2)^2 + (y+1)^2 = 4 → y = x + 1) → 
    (∃ x y : ℝ, (x + 2)^2 + (y - 3)^2 = 4) :=
  by
    sorry

end symmetric_circle_equation_l678_678450


namespace children_without_candies_l678_678005

/-- There are 73 children standing in a circle. An evil Santa Claus walks around 
    the circle in a clockwise direction and distributes candies. First, he gives one candy 
    to the first child, then skips 1 child, gives one candy to the next child, 
    skips 2 children, gives one candy to the next child, skips 3 children, and so on.
    
    After distributing 2020 candies, he leaves. 
    
    This theorem states that the number of children who did not receive any candies 
    is 36. -/
theorem children_without_candies : 
  let n := 73
  let a : ℕ → ℕ := λk, (k * (k + 1) / 2) % n
  ∃ m : ℕ, (distributed_positions m 2020 73 = 37) → (73 - 37) = 36
  sorry

end children_without_candies_l678_678005


namespace general_term_arithmetic_sum_terms_geometric_l678_678136

section ArithmeticSequence

variables {S : ℕ → ℝ} {a : ℕ → ℝ} {d : ℝ}

-- Conditions for Part 1
def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  S 5 - S 2 = 195 ∧ d = -2 ∧
  ∀ n, S n = n * (a 1 + (n - 1) * (d / 2))

-- Prove the general term formula for the sequence {a_n}
theorem general_term_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) 
    (h : sum_arithmetic_sequence S a d) : 
    ∀ n, a n = -2 * n + 73 :=
sorry

end ArithmeticSequence


section GeometricSequence

variables {b : ℕ → ℝ} {n : ℕ} {T : ℕ → ℝ} {a : ℕ → ℝ}

-- Conditions for Part 2
def sum_geometric_sequence (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 1 = 13 ∧ b 2 = 65 ∧ a 4 = 65

-- Prove the sum of the first n terms for the sequence {b_n}
theorem sum_terms_geometric (b : ℕ → ℝ) (T : ℕ → ℝ) (a : ℕ → ℝ)
    (h : sum_geometric_sequence b T a) : 
    ∀ n, T n = 13 * (5^n - 1) / 4 :=
sorry

end GeometricSequence

end general_term_arithmetic_sum_terms_geometric_l678_678136


namespace P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l678_678398

-- Define the problem conditions and questions
def P_1 (n : ℕ) : ℚ := sorry
def P_2 (n : ℕ) : ℚ := sorry

-- Part (a)
theorem P2_3_eq_2_3 : P_2 3 = 2 / 3 := sorry

-- Part (b)
theorem P1_n_eq_1_n (n : ℕ) (h : n ≥ 1): P_1 n = 1 / n := sorry

-- Part (c)
theorem P2_recurrence (n : ℕ) (h : n ≥ 2) : 
  P_2 n = (2 / n) * P_1 (n-1) + ((n-2) / n) * P_2 (n-1) := sorry

-- Part (d)
theorem P2_n_eq_2_n (n : ℕ) (h : n ≥ 1): P_2 n = 2 / n := sorry

end P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l678_678398


namespace total_chocolate_bars_in_colossal_box_l678_678018

theorem total_chocolate_bars_in_colossal_box :
  let colossal_boxes := 350
  let sizable_boxes := 49
  let small_boxes := 75
  colossal_boxes * sizable_boxes * small_boxes = 1287750 :=
by
  sorry

end total_chocolate_bars_in_colossal_box_l678_678018


namespace sine_of_angle_add_pi_over_3_l678_678904

theorem sine_of_angle_add_pi_over_3 (t : ℝ) (h : t < 0) 
    (hθ_cos : cos θ = -(√5)/5) (hθ_sin : sin θ = -(2*√5)/5) :
    sin (θ + π/3) = -(2*√5 + √15)/10 :=
by sorry

end sine_of_angle_add_pi_over_3_l678_678904


namespace billy_distance_l678_678409

-- Billy starts at point A, walks 7 miles east to point B, and then turns 45 degrees north and walks 8 miles to point D.
-- We need to prove that the distance from A to D is 9 miles.

theorem billy_distance (s t : ℝ) (h₁ : s = 7) (h₂ : t = 8) : 
  let D := sqrt ((s ^ 2) + (t * (sqrt 2 / 2) ^ 2))
  in D = 9 := 
by
  -- Proof skipped
  sorry

end billy_distance_l678_678409


namespace two_digit_powers_of_3_count_l678_678989

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678989


namespace two_digit_powers_of_3_count_l678_678991

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678991


namespace pump_fill_time_without_leak_l678_678054

def time_with_leak := 10
def leak_empty_time := 10

def combined_rate_with_leak := 1 / time_with_leak
def leak_rate := 1 / leak_empty_time

def T : ℝ := 5

theorem pump_fill_time_without_leak
  (time_with_leak : ℝ)
  (leak_empty_time : ℝ)
  (combined_rate_with_leak : ℝ)
  (leak_rate : ℝ)
  (T : ℝ)
  (h1 : combined_rate_with_leak = 1 / time_with_leak)
  (h2 : leak_rate = 1 / leak_empty_time)
  (h_combined : 1 / T - leak_rate = combined_rate_with_leak) :
  T = 5 :=
by {
  sorry
}

end pump_fill_time_without_leak_l678_678054


namespace correct_range_of_x_l678_678492

variable {x : ℝ}

noncomputable def isosceles_triangle (x y : ℝ) : Prop :=
  let perimeter := 2 * y + x
  let relationship := y = - (1/2) * x + 8
  perimeter = 16 ∧ relationship

theorem correct_range_of_x (x y : ℝ) (h : isosceles_triangle x y) : 0 < x ∧ x < 8 :=
by
  -- The proof of the theorem is omitted
  sorry

end correct_range_of_x_l678_678492


namespace two_digit_numbers_in_form_3_pow_n_l678_678965

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678965


namespace two_digit_powers_of_3_count_l678_678993

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678993


namespace three_powers_in_two_digit_range_l678_678995

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l678_678995


namespace hyperbola_slope_product_l678_678163

open Real

theorem hyperbola_slope_product
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h : ∀ {x y : ℝ}, x ≠ 0 → (x^2 / a^2 - y^2 / b^2 = 1) → 
    ∀ {k1 k2 : ℝ}, (x = 0 ∨ y = 0) → (k1 * k2 = ((b^2) / (a^2)))) :
  (b^2 / a^2 = 3) :=
by 
  sorry

end hyperbola_slope_product_l678_678163


namespace shortest_distance_comet_l678_678746

noncomputable def shortest_distance_comet_distance (d : ℝ) : ℝ :=
  let p1 := d / 2
  let p2 := 3 * d / 2
  min (p1 / 2) (p2 / 2)

theorem shortest_distance_comet (d : ℝ) (h_angle: real.angle_of_line_and_axis = 60) : shortest_distance_comet_distance d = min (d / 4) (3 * d / 4) :=
sorry

end shortest_distance_comet_l678_678746


namespace vertex_on_x_axis_segment_cut_on_x_axis_l678_678921

-- Define the quadratic function
def quadratic_func (k x : ℝ) : ℝ :=
  (k + 2) * x^2 - 2 * k * x + 3 * k

-- The conditions to prove
theorem vertex_on_x_axis (k : ℝ) :
  (4 * k^2 - 4 * 3 * k * (k + 2) = 0) ↔ (k = 0 ∨ k = -3) :=
sorry

theorem segment_cut_on_x_axis (k : ℝ) :
  ((2 * k / (k + 2))^2 - 12 * k / (k + 2) = 16) ↔ (k = -8/3 ∨ k = -1) :=
sorry

end vertex_on_x_axis_segment_cut_on_x_axis_l678_678921


namespace sqrt_of_S_l678_678331

def initial_time := 16 * 3600 + 11 * 60 + 22
def initial_date := 16
def total_seconds_in_a_day := 86400
def total_seconds_in_an_hour := 3600

theorem sqrt_of_S (S : ℕ) (hS : S = total_seconds_in_a_day + total_seconds_in_an_hour) : 
  Real.sqrt S = 300 := 
sorry

end sqrt_of_S_l678_678331


namespace crow_speed_l678_678374

theorem crow_speed (d_nest_ditch : ℕ) (round_trips : ℕ) (total_time_hrs : ℝ) :
  d_nest_ditch = 200 → round_trips = 15 → total_time_hrs = 1.5 → 
  let total_distance_km := (round_trips * 2 * d_nest_ditch) / 1000 in
  let speed_kmph := total_distance_km / total_time_hrs in
  speed_kmph = 4 :=
by {
  intros h1 h2 h3,
  have h4: (round_trips * 2 * d_nest_ditch) / 1000 = 6,
    calc (15 * 2 * 200) / 1000 
      = 6000 / 1000 : by norm_num
      ... = 6 : by norm_num,
  have h5: 6 / 1.5 = 4,
    calc 6 / 1.5 
      = 4 : by norm_num,
  exact h5,
}

end crow_speed_l678_678374


namespace probability_alternating_draws_l678_678014

theorem probability_alternating_draws :
  (∃ w b, w = 5 ∧ b = 3 ∧ ∀ s, s = [1, 0, 1, 0, 1, 0, 1, 1] → (prob s = 1 / 56)) :=
sorry

end probability_alternating_draws_l678_678014


namespace ratio_of_small_triangle_area_l678_678324

theorem ratio_of_small_triangle_area (m n : ℕ) : 
  ∃ r s p : ℕ, 
  let square_area := s^2 in
  let triangle_original_area : ℕ := n * s^2 in
  let triangle_ADF_area : ℕ := m * s^2 in
  let triangle_FDB_area : ℕ := (n - m - 1) * s^2 in
  let total_triangle_area := triangle_ADF_area + triangle_FDB_area + square_area in
  total_triangle_area = triangle_original_area :=
by
  sorry

end ratio_of_small_triangle_area_l678_678324


namespace point_B_represents_2_or_neg6_l678_678145

def A : ℤ := -2

def B (move : ℤ) : ℤ := A + move

theorem point_B_represents_2_or_neg6 (move : ℤ) (h : move = 4 ∨ move = -4) : 
  B move = 2 ∨ B move = -6 :=
by
  cases h with
  | inl h1 => 
    rw [h1]
    unfold B
    unfold A
    simp
  | inr h1 => 
    rw [h1]
    unfold B
    unfold A
    simp

end point_B_represents_2_or_neg6_l678_678145


namespace mono_intervals_and_zeros_l678_678159

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x - (1 / 2) * a * x^2)

-- Task: Prove the main conditions related to the function f(x)
theorem mono_intervals_and_zeros (a : ℝ) :
  -- Question I
  (deriv (fun x => f x a) 2 = 0 → a = 5 / 4) ∧
  -- Question II
  ((a ≤ 0 → ∀ x > 0, deriv (fun x => f x a) x > 0) ∧
  (a > 0 → (∀ x ∈ Ioo 0 (sqrt (1 / a)), deriv (fun x => f x a) x > 0) ∧
  (∀ x ∈ Ioo (sqrt (1 / a)) +∞, deriv (fun x => f x a) x < 0))) ∧
  -- Question III
  ((0 ≤ a ∧ a < 4 / Real.exp 4) ∨ a = 1 / Real.exp 1 →
    ∃ (x₀ : ℝ), x₀ ∈ Icc 1 (Real.exp 2) ∧ f x₀ a = 0) ∧
  (4 / Real.exp 4 ≤ a ∧ a < 1 / Real.exp 1 →
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Icc 1 (Real.exp 2) ∧ x₂ ∈ Icc 1 (Real.exp 2) ∧
      f x₁ a = 0 ∧ f x₂ a = 0) ∧
  (a < 0 ∨ a > 1 / Real.exp 1 →
    ∀ x ∈ Icc 1 (Real.exp 2), f x a ≠ 0) :=
begin
  sorry
end

end mono_intervals_and_zeros_l678_678159


namespace find_equation_parabola_line_ab_fixed_point_l678_678135

-- Define the parabola and circle conditions
variable (p : ℝ) (h_pos : p > 0)
variable (C : set (ℝ × ℝ)) (F : ℝ × ℝ)
variable (M : ℝ × ℝ) (N : ℝ × ℝ)
variable (A B : ℝ × ℝ)

-- Equation of parabola
def parabola_equation : Prop := ∀ x y, (y, x) ∈ C ↔ y^2 = 2 * p * x

-- Condition on the focus F
def focus_condition : Prop := F = (p / 2, 0)

-- Condition on the circle passing through origin and F
def circle_through_origin_and_F : Prop := (0, 0) ∈ (euclidean_distance M F) ∧ F ∈ (euclidean_distance M F)

-- Distance of center M from directrix
def center_distance_condition : Prop := (euclidean_distance M ((-p / 2), 0)) = 3 / 2

-- Point N on the parabola
def point_on_parabola : Prop := N = (4, 4) ∧ (N ∈ C)

-- Perpendicular chords condition
def perpendicular_chords : Prop := 
  let k_NA := (A.2 - N.2) / ((A.1^2) / 4 - N.1)
  let k_NB := (B.2 - N.2) / ((B.1^2) / 4 - N.1)
  in k_NA * k_NB = -1

-- Line AB passes through fixed point
def line_passes_through_fixed_point : Prop :=
  ∀ y₁ y₂ x y, 
  let k_AB := (y₂ - y₁) / ((y₂^2 / 4) - (y₁^2 / 4))
  in (A = (y₁^2 / 4, y₁) ∧ B = (y₂^2 / 4, y₂)) → 
     y = k_AB * (x - (y₁^2 / 4)) + y₁  → (8, -4) ∈ (A :: B :: list.nil)

-- Proposition for part I
theorem find_equation_parabola : parabola_equation p ↔ ∃ (p : ℝ), focus_condition p ∧ circle_through_origin_and_F p ∧ center_distance_condition p := sorry

-- Proposition for part II
theorem line_ab_fixed_point : 
point_on_parabola N ∧ perpendicular_chords A B N → line_passes_through_fixed_point A B → (8, -4) ∈ (A :: B :: list.nil) := sorry

end find_equation_parabola_line_ab_fixed_point_l678_678135


namespace simplest_quadratic_radical_l678_678338

theorem simplest_quadratic_radical :
  ∀ (A : ℚ) (B : ℚ) (C : ℚ) (D : ℚ),
    (A = real.sqrt 8) →
    (B = real.sqrt (1 / 2)) →
    (C = real.sqrt (3^3)) →
    (D = real.sqrt 5) →
    is_simplest(D) (D = real.sqrt 5)
  := sorry

end simplest_quadratic_radical_l678_678338


namespace martian_socks_l678_678025

theorem martian_socks (r w b : ℕ) (h_r : r ≥ 5) (h_w : w ≥ 5) (h_b : b ≥ 5) :
  ∃ n, n = 13 ∧ ∀ (socks : list ℕ), (∀ s, s ∈ socks → s = 1 ∨ s = 2 ∨ s = 3) →
  (n = 13 → ∃ c, socks.filter (λ x, x = c).length ≥ 5) :=
by
  sorry

end martian_socks_l678_678025


namespace sin_B_eq_l678_678570

variable (a : ℝ) (b : ℝ) (A : ℝ)
variable (B : ℝ)

-- Given conditions
axiom h1 : a = 3 * Real.sqrt 3
axiom h2 : b = 4
axiom h3 : A = Real.pi / 6 -- 30 degrees in radians

-- Proof statement
theorem sin_B_eq : sin B = 2 * Real.sqrt 3 / 9 :=
sorry

end sin_B_eq_l678_678570


namespace half_of_number_l678_678365

theorem half_of_number (x : ℝ) (h : (4 / 15 * 5 / 7 * x - 4 / 9 * 2 / 5 * x = 8)) : (1 / 2 * x = 315) :=
sorry

end half_of_number_l678_678365


namespace determine_digit_d_l678_678553

noncomputable def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem determine_digit_d (a b c d : ℕ) (h_distinct : distinct_digits a b c d) 
  (h_eq : -((10^4)*d + (10^3)*a + (10^2)*b + (10^1)*a + (10^0)*c) = 2014 * d) :
  d = 2 :=
sorry

end determine_digit_d_l678_678553


namespace kerry_age_l678_678604

theorem kerry_age (cost_per_box : ℝ) (boxes_bought : ℕ) (candles_per_box : ℕ) (cakes : ℕ) 
  (total_cost : ℝ) (total_candles : ℕ) (candles_per_cake : ℕ) (age : ℕ) :
  cost_per_box = 2.5 →
  boxes_bought = 2 →
  candles_per_box = 12 →
  cakes = 3 →
  total_cost = 5 →
  total_cost = boxes_bought * cost_per_box →
  total_candles = boxes_bought * candles_per_box →
  candles_per_cake = total_candles / cakes →
  age = candles_per_cake →
  age = 8 :=
by
  intros
  sorry

end kerry_age_l678_678604


namespace problem1_eval_problem2_eval_l678_678838

-- Problem 1 equivalent proof problem
theorem problem1_eval : |(-2 + 1/4)| - (-3/4) + 1 - |(1 - 1/2)| = 3 + 1/2 := 
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2_eval : -3^2 - (8 / (-2)^3 - 1) + 3 / 2 * (1 / 2) = -6 + 1/4 :=
by
  sorry

end problem1_eval_problem2_eval_l678_678838


namespace clothes_color_proof_l678_678787

variables (Alyna_shirt Alyna_shorts Bohdan_shirt Bohdan_shorts Vika_shirt Vika_shorts Grysha_shirt Grysha_shorts : Type)
variables [decidable_eq Alyna_shirt] [decidable_eq Alyna_shorts]
          [decidable_eq Bohdan_shirt] [decidable_eq Bohdan_shorts]
          [decidable_eq Vika_shirt] [decidable_eq Vika_shorts]
          [decidable_eq Grysha_shirt] [decidable_eq Grysha_shorts]

axiom red : Alyna_shirt
axiom blue : Alyna_shorts

theorem clothes_color_proof
  (h1 : Alyna_shirt = red ∧ Bohdan_shirt = red ∧ Alyna_shorts ≠ Bohdan_shorts)
  (h2 : Vika_shorts = blue ∧ Grysha_shorts = blue ∧ Vika_shirt ≠ Grysha_shirt)
  (h3 : Alyna_shirt ≠ Vika_shirt ∧ Alyna_shorts ≠ Vika_shorts) :
  (Alyna_shirt = red ∧ Alyna_shorts = red ∧ 
   Bohdan_shirt = red ∧ Bohdan_shorts = blue ∧ 
   Vika_shirt = blue ∧ Vika_shorts = blue ∧ 
   Grysha_shirt = red ∧ Grysha_shorts = blue) :=
by
  sorry

end clothes_color_proof_l678_678787


namespace slices_per_person_is_correct_l678_678702

-- Conditions
def slices_per_tomato : Nat := 8
def total_tomatoes : Nat := 20
def people_for_meal : Nat := 8

-- Calculate number of slices for a single person
def slices_needed_for_single_person (slices_per_tomato : Nat) (total_tomatoes : Nat) (people_for_meal : Nat) : Nat :=
  (slices_per_tomato * total_tomatoes) / people_for_meal

-- The statement to be proved
theorem slices_per_person_is_correct : slices_needed_for_single_person slices_per_tomato total_tomatoes people_for_meal = 20 :=
by
  sorry

end slices_per_person_is_correct_l678_678702


namespace suff_not_nec_l678_678738

theorem suff_not_nec (x : ℝ) : (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬(x ≤ 0)) :=
by
  sorry

end suff_not_nec_l678_678738


namespace gage_skating_time_l678_678873

theorem gage_skating_time :
  let gage_times_in_minutes1 := 1 * 60 + 15 -- 1 hour 15 minutes converted to minutes
  let gage_times_in_minutes2 := 2 * 60      -- 2 hours converted to minutes
  let total_skating_time_8_days := 5 * gage_times_in_minutes1 + 3 * gage_times_in_minutes2
  let required_total_time := 10 * 95       -- 10 days * 95 minutes per day
  required_total_time - total_skating_time_8_days = 215 :=
by
  sorry

end gage_skating_time_l678_678873


namespace entree_cost_l678_678178

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l678_678178


namespace goose_eggs_laid_at_pond_fraction_geese_survived_first_month_did_not_survive_first_year_l678_678266

variable (E : ℕ) (g_survived_first_year : ℕ := 120)

axiom hatched_fraction : ℚ := 1 / 4
axiom survived_first_month_fraction : ℚ := 4 / 5

theorem goose_eggs_laid_at_pond (hatch_fraction_nonzero: hatched_fraction ≠ 0) (survive_fraction_nonzero: survived_first_month_fraction ≠ 0) :
  (g_survived_first_year : ℚ) / (survived_first_month_fraction * hatched_fraction) = 2400 :=
by sorry

theorem fraction_geese_survived_first_month_did_not_survive_first_year 
  (total_geese_first_month: ℚ := survived_first_month_fraction * hatched_fraction * 2400) :
  (total_geese_first_month - g_survived_first_year : ℚ) / total_geese_first_month = 3 / 4 :=
by sorry

end goose_eggs_laid_at_pond_fraction_geese_survived_first_month_did_not_survive_first_year_l678_678266


namespace num_two_digit_powers_of_3_l678_678943

theorem num_two_digit_powers_of_3 : 
  {n : ℤ // 10 ≤ 3 ^ n ∧ 3 ^ n < 100 }.to_finset.card = 2 :=
sorry

end num_two_digit_powers_of_3_l678_678943


namespace sqrt_expr_value_l678_678074

theorem sqrt_expr_value : sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := 
by sorry

end sqrt_expr_value_l678_678074


namespace stratified_sampling_correct_l678_678026

theorem stratified_sampling_correct (total_students : ℕ) (junior_students : ℕ) (undergrad_students : ℕ) (sample_size : ℕ) :
  total_students = 5600 →
  junior_students = 1300 →
  undergrad_students = 3000 →
  sample_size = 280 →
  let P := sample_size / total_students in
  let junior_sample_size := junior_students * P in
  let undergrad_sample_size := undergrad_students * P in
  let grad_sample_size := sample_size - junior_sample_size - undergrad_sample_size in
  junior_sample_size = 65 ∧ undergrad_sample_size = 150 ∧ grad_sample_size = 65 :=
by
  intros h_total h_junior h_undergrad h_sample
  let P := sample_size / total_students
  let junior_sample_size := junior_students * P
  let undergrad_sample_size := undergrad_students * P
  let grad_sample_size := sample_size - junior_sample_size - undergrad_sample_size
  have h_P: P = 1 / 20 := by sorry
  have h_junior_sample: junior_sample_size = 1300 * (1 / 20) := by sorry
  have h_undergrad_sample: undergrad_sample_size = 3000 * (1 / 20) := by sorry
  have h_graduate_sample: grad_sample_size = 65 := by sorry
  show junior_sample_size = 65 ∧ undergrad_sample_size = 150 ∧ grad_sample_size = 65 from
    ⟨by rw [h_junior_sample], by rw [h_undergrad_sample], h_graduate_sample⟩

end stratified_sampling_correct_l678_678026


namespace trees_died_proof_l678_678930

def treesDied (original : Nat) (remaining : Nat) : Nat := original - remaining

theorem trees_died_proof : treesDied 20 4 = 16 := by
  -- Here we put the steps needed to prove the theorem, which is essentially 20 - 4 = 16.
  sorry

end trees_died_proof_l678_678930


namespace no_such_sequence_exists_l678_678088

open List

def is_square_free (n : ℕ) : Prop := ∀ m : ℕ, m * m ∣ n → m = 1

def sum_is_square_free_if_and_only_if_of_large_enough_k (seq : List ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k > n → (is_square_free k ↔ is_square_free (seq.take k).sum)

theorem no_such_sequence_exists : ¬∃ (seq : List ℕ), 
  (∀ i j, seq.nth i ≠ seq.nth j ∧ seq.nth i ≠ seq.nth (i + 1)) ∧  -- non-constant and distinct
  (∀ i, seq.nth i ≠ none) ∧  -- infinite (i.e., every index has a value, in essence)
  sum_is_square_free_if_and_only_if_of_large_enough_k seq n := 
sorry

end no_such_sequence_exists_l678_678088


namespace extreme_points_a_leq_neg2_range_of_a_always_positive_l678_678156

noncomputable def f (x a : ℝ) : ℝ :=
  x * abs (x + a) - (1 / 2) * log x

theorem extreme_points_a_leq_neg2 (a x : ℝ) (h₁ : a ≤ -2) :
  x = (-a - sqrt (a^2 - 4)) / 4 ∨ x = (-a + sqrt (a^2 - 4)) / 4 ∨ x = -a →
  ∃ b : ℝ, (b = (-a - sqrt (a^2 - 4)) / 4 ∨ b = (-a + sqrt (a^2 - 4)) / 4 ∨ b = -a) ∧
  (∀ x < b, f x a < f b a) ∧ (∀ x > b, f x a > f b a) ∧ f b a < 0 :=
sorry

theorem range_of_a_always_positive (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a > 0) ↔ a > -1 :=
sorry

end extreme_points_a_leq_neg2_range_of_a_always_positive_l678_678156


namespace probability_event_a_l678_678205

-- Given conditions
def colors : List String := ["red", "yellow", "green"]
def num_colors : Nat := colors.length
def draws : ℕ := 3 -- Number of draws with replacement

-- Define the concept of a draw event
def event_a := List.replicate draws "red" ∨ List.replicate draws "yellow" ∨ List.replicate draws "green"

-- Define the total number of possible outcomes
def total_outcomes := num_colors ^ draws

-- Correct answer to prove
theorem probability_event_a : (event_a.card / total_outcomes) = (1 / 9) := by
  sorry

end probability_event_a_l678_678205


namespace ratio_of_amounts_l678_678192

theorem ratio_of_amounts (B J P : ℝ) (hB : B = 60) (hP : P = (1 / 3) * B) (hJ : J = B - 20) : J / P = 2 :=
by
  have hP_val : P = 20 := by sorry
  have hJ_val : J = 40 := by sorry
  have ratio : J / P = 40 / 20 := by sorry
  show J / P = 2
  sorry

end ratio_of_amounts_l678_678192


namespace average_weight_of_remaining_boys_l678_678699

theorem average_weight_of_remaining_boys (avg_weight_16: ℝ) (avg_weight_total: ℝ) (weight_16: ℝ) (total_boys: ℝ) (avg_weight_8: ℝ) : 
  (avg_weight_16 = 50.25) → (avg_weight_total = 48.55) → (weight_16 = 16 * avg_weight_16) → (total_boys = 24) → 
  (total_weight = total_boys * avg_weight_total) → (weight_16 + 8 * avg_weight_8 = total_weight) → avg_weight_8 = 45.15 :=
by
  intros h_avg_weight_16 h_avg_weight_total h_weight_16 h_total_boys h_total_weight h_equation
  sorry

end average_weight_of_remaining_boys_l678_678699


namespace identify_clothes_l678_678801

open Function

-- Definitions
def Alina : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Red"
def Bogdan : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"
def Vika : Prop := ∃ (tshirt short : String), tshirt = "Blue" ∧ short = "Blue"
def Grisha : Prop := ∃ (tshirt short : String), tshirt = "Red" ∧ short = "Blue"

-- Problem statement
theorem identify_clothes :
  Alina ∧ Bogdan ∧ Vika ∧ Grisha :=
by
  sorry -- Proof will be developed here

end identify_clothes_l678_678801


namespace ice_cream_melt_l678_678770

theorem ice_cream_melt (r_sphere r_cylinder : ℝ) (h : ℝ)
  (V_sphere : ℝ := (4 / 3) * Real.pi * r_sphere^3)
  (V_cylinder : ℝ := Real.pi * r_cylinder^2 * h)
  (H_equal_volumes : V_sphere = V_cylinder) :
  h = 4 / 9 := by
  sorry

end ice_cream_melt_l678_678770


namespace find_NK_l678_678206

variable (R α : ℝ)
variable (A B M N K : Point)
variable (circle : Circle) (chord : Chord) (orthocenter : Orthocenter)

-- Conditions
hypothesis (h_chord_subtension : chord.arc_length ≤ π)
hypothesis (h_radius : circle.radius = R)
hypothesis (h_point_on_circle : M ∈ circle)
hypothesis (h_segment_MN : |MN| = R)
hypothesis (h_segment_MK : |MK| = M.dist_orthocenter (triangle A B M))

-- Question
theorem find_NK :
  |NK| = R * real.sqrt (1 + 8 * (cos α) ^ 2) :=
sorry

end find_NK_l678_678206


namespace num_two_digit_powers_of_3_l678_678956

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678956


namespace square_area_l678_678309

theorem square_area :
  let p1 := (1: ℝ, -2: ℝ)
  let p2 := (-3: ℝ, 5: ℝ)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 65 :=
by
  let p1 := (1: ℝ, -2: ℝ)
  let p2 := (-3: ℝ, 5: ℝ)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  calc side_length^2 
      = Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)^2 : by sorry
  ... = 65 : by sorry

end square_area_l678_678309


namespace chord_parabola_constant_t_l678_678078

theorem chord_parabola_constant_t (c : ℝ) (h : c = 2/3) :
  ∀ {A B : ℝ × ℝ}, 
    (A.snd = A.fst^2) ∧ (B.snd = B.fst^2) ∧ (A ≠ B) ∧ (A ≠ (0, c)) ∧ (B ≠ (0, c)) ∧
    let AC := (A.fst - 0)^2 + (A.snd - c)^2,
        BC := (B.fst - 0)^2 + (B.snd - c)^2
    in
      (1 / AC) + (1 / BC) = 3 := 
by 
  sorry

end chord_parabola_constant_t_l678_678078


namespace pi_digits_product_even_l678_678295

theorem pi_digits_product_even (a : Fin 24 → ℕ) (h_perm : Multiset.of_list (List.ofFn a) = {3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4}) :
  Even ((a 0 - a 1) * (a 2 - a 3) * (a 4 - a 5) * (a 6 - a 7) * (a 8 - a 9) * (a 10 - a 11) * (a 12 - a 13) * (a 14 - a 15) * (a 16 - a 17) * (a 18 - a 19) * (a 20 - a 21) * (a 22 - a 23)) :=
by
  -- proof placeholder
  sorry

end pi_digits_product_even_l678_678295


namespace imaginary_part_of_complex_num_l678_678451

-- Definitions from the problem statement
def complex_num := (1 + 2 * Complex.I) / (1 + Complex.I)
def imaginary_part (z : ℂ) : ℝ := z.im

-- Main theorem to be proved
theorem imaginary_part_of_complex_num :
  imaginary_part complex_num = 1 / 2 :=
sorry

end imaginary_part_of_complex_num_l678_678451


namespace range_of_f_l678_678317

open Set

noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

theorem range_of_f : range (λ x, f x) ∩ (Icc 0 1) = Icc 2 5 := by
  sorry

end range_of_f_l678_678317


namespace problem_p_s_difference_l678_678646

def P : ℤ := 12 - (3 * 4)
def S : ℤ := (12 - 3) * 4

theorem problem_p_s_difference : P - S = -36 := by
  sorry

end problem_p_s_difference_l678_678646


namespace rachel_found_boxes_l678_678655

theorem rachel_found_boxes (pieces_per_box total_pieces B : ℕ) 
  (h1 : pieces_per_box = 7) 
  (h2 : total_pieces = 49) 
  (h3 : B = total_pieces / pieces_per_box) : B = 7 := 
by 
  sorry

end rachel_found_boxes_l678_678655


namespace Okeydokey_should_receive_25_earthworms_l678_678734

def applesOkeydokey : ℕ := 5
def applesArtichokey : ℕ := 7
def totalEarthworms : ℕ := 60
def totalApples : ℕ := applesOkeydokey + applesArtichokey
def okeydokeyProportion : ℚ := applesOkeydokey / totalApples
def okeydokeyEarthworms : ℚ := okeydokeyProportion * totalEarthworms

theorem Okeydokey_should_receive_25_earthworms : okeydokeyEarthworms = 25 := by
  sorry

end Okeydokey_should_receive_25_earthworms_l678_678734


namespace solution_set_bx2_minus_2ax_plus_1_l678_678475

variable {a b : ℝ}
variable {f : ℝ → ℝ}

def f (x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Given conditions
axiom sol_set_f_leq_zero : ∀ x : ℝ, (f x ≤ 0 ↔ -1 ≤ x ∧ x ≤ 2)
axiom b_eq_a_squared : b = a^2
axiom exists_f_product_one : ∀ (x1 : ℝ), 2 ≤ x1 ∧ x1 ≤ 4 → ∃ (x2 : ℝ), 2 ≤ x2 ∧ x2 ≤ 4 ∧ f x1 * f x2 = 1

-- To be proven
theorem solution_set_bx2_minus_2ax_plus_1 :
  ∀ x, (b*x^2 - 2*a*x + 1 ≤ 0 ↔ x ≤ -1 ∨ x ≥ 1 / 2) ∧
  (a = 3 + Real.sqrt 2 ∨ a = 3 - Real.sqrt 2) :=
begin
  sorry
end

end solution_set_bx2_minus_2ax_plus_1_l678_678475


namespace integer_between_sqrt_2_and_sqrt_8_l678_678462

theorem integer_between_sqrt_2_and_sqrt_8 (a : ℤ) (h1 : real.sqrt 2 < a) (h2 : a < real.sqrt 8) : a = 2 :=
sorry

end integer_between_sqrt_2_and_sqrt_8_l678_678462


namespace anna_ham_slices_l678_678830

theorem anna_ham_slices (h1 : 31 + 119 = 150) (h2 : 50 * 3 = 150) : 
  ∀ (n_slices_needed : ℕ) (n_sandwiches : ℕ) (slices_per_sandwich : ℕ), 
  (n_slices_needed = 31 + 119) → 
  (n_sandwiches = 50) → 
  (slices_per_sandwich = n_slices_needed / n_sandwiches) → 
  slices_per_sandwich = 3 :=
by
  intro n_slices_needed n_sandwiches slices_per_sandwich
  intro h_n_slices_needed h_n_sandwiches h_slices_per_sandwich
  rw [h_n_slices_needed, h_n_sandwiches, h_slices_per_sandwich]
  sorry

end anna_ham_slices_l678_678830


namespace num_possible_C_locations_correct_l678_678648

open Real

noncomputable def num_possible_C_locations (A B : Point) (AB : ℝ) : ℕ :=
  if AB = 10 then 8 else sorry

theorem num_possible_C_locations_correct :
  ∀ (A B : Point), dist A B = 10 → num_possible_C_locations A B 10 = 8 :=
by
  intros A B h_AB
  rw [num_possible_C_locations]
  simp [h_AB]
  sorry

end num_possible_C_locations_correct_l678_678648


namespace two_digit_numbers_in_form_3_pow_n_l678_678971

theorem two_digit_numbers_in_form_3_pow_n : ∃ (c : ℕ), c = 2 ∧ ∀ (n : ℕ), (3^n).digits = 2 ↔ n = 3 ∨ n = 4 := by
  sorry

end two_digit_numbers_in_form_3_pow_n_l678_678971


namespace two_digit_powers_of_3_count_l678_678992

theorem two_digit_powers_of_3_count : 
  {n : ℕ | 10 ≤ 3^n ∧ 3^n ≤ 99}.to_finset.card = 2 :=
by sorry

end two_digit_powers_of_3_count_l678_678992


namespace polar_to_cartesian_perpendicular_points_l678_678508

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ^2 = 9 / ((cos θ)^2 + 9 * (sin θ)^2)) :
  ∃ x y : ℝ, x = ρ * cos θ ∧ y = ρ * sin θ ∧ (x^2 / 9 + y^2 = 1) :=
by 
  sorry

theorem perpendicular_points (ρ1 ρ2 α : ℝ) (h1 : ρ1^2 = 9 / ((cos α)^2 + 9 * (sin α)^2))
  (h2 : ρ2^2 = 9 / ((cos (α + π/2))^2 + 9 * (sin (α + π/2))^2)) :
  (1 / ρ1^2) + (1 / ρ2^2) = 10 / 9 :=
by 
  sorry

end polar_to_cartesian_perpendicular_points_l678_678508


namespace alpha_value_l678_678908

theorem alpha_value (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.logb 3 (x + 1)) (h2 : f α = 1) : α = 2 := by
  sorry

end alpha_value_l678_678908


namespace necessary_not_sufficient_condition_l678_678754

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the condition for the problem
def condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the proof problem: Prove that the interval is a necessary but not sufficient condition for f(x) < 0
theorem necessary_not_sufficient_condition : 
  ∀ x : ℝ, condition x → ¬ (∀ y : ℝ, condition y → f y < 0) :=
sorry

end necessary_not_sufficient_condition_l678_678754


namespace three_powers_in_two_digit_range_l678_678994

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l678_678994


namespace extremal_points_range_a_sum_of_f_greater_than_two_l678_678522

noncomputable def f (x a : ℝ) := Real.exp x - (1/2) * x^2 - a * x

theorem extremal_points_range_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a ∈ set.Ioi 1 := by
sorry

theorem sum_of_f_greater_than_two (a : ℝ) (x1 x2 : ℝ) (h_extrema : x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0)
    : f x1 a + f x2 a > 2 := by
sorry

end extremal_points_range_a_sum_of_f_greater_than_two_l678_678522


namespace series_sum_l678_678695

def y : ℕ → ℝ
| 1 := 50
| (n + 2) := y (n + 1) ^ 2 - y (n + 1)

theorem series_sum :
  (∑ n in (Finset.range (10000)), 1 / (y (n + 1) - 1)) = 1 / 50 :=
sorry

end series_sum_l678_678695


namespace arithmetic_sequence_sum_l678_678131

theorem arithmetic_sequence_sum :
  (∃ (a_n : ℕ → ℝ), a_n 5 = 5 ∧ (finset.sum (finset.range 5) (λ n, a_n (n + 1)) = 15) ∧ 
    (finset.sum (finset.range 100) (λ n, 1 / (a_n n * a_n (n + 1))) = 100 / 101)) :=
begin
  let a_n := λ n, n,
  use a_n,
  split,
  { -- prove a_5 = 5
    sorry },
  split,
  { -- prove S_5 = 15
    sorry },
  { -- prove the sum of the sequence equals 100 / 101
    sorry },
end

end arithmetic_sequence_sum_l678_678131


namespace upstream_speed_l678_678030

variable (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)

def speed_of_man_in_still_water := V_m = 35
def speed_of_man_downstream := V_downstream = 45
def speed_of_man_upstream := V_upstream = 25

theorem upstream_speed
  (h1: speed_of_man_in_still_water V_m)
  (h2: speed_of_man_downstream V_downstream)
  : speed_of_man_upstream V_upstream :=
by
  -- Placeholder for the proof
  sorry

end upstream_speed_l678_678030


namespace curve_eq_two_intersecting_lines_l678_678684

theorem curve_eq_two_intersecting_lines :
  ∀ (ρ θ : ℝ), (ρ * (Real.cos θ ^ 2 - Real.sin θ ^ 2) = 0) ↔ (ρ * (Real.cos 2 * θ) = 0) ↔ (ρ = 0 ∨ Real.cos 2 * θ = 0) ↔ (ρ = 0 ∨ θ = π / 4 ∨ θ = 3 * π / 4) ↔ ((ρ * Real.cos θ = 0) ∧ (ρ * Real.sin θ = 0) ∨ (ρ * Real.cos θ = ρ * Real.sin θ) ∨ (ρ * Real.cos θ = -ρ * Real.sin θ)) ↔ (y = x ∨ y = -x) :=
by sorry

end curve_eq_two_intersecting_lines_l678_678684


namespace equation_has_12_solutions_l678_678611

theorem equation_has_12_solutions :
  ∃ (s : set (ℤ × ℤ)), s.card ≥ 12 ∧ ∀ (p : ℤ × ℤ), p ∈ s ↔ (p.fst ^ 2 + p.snd ^ 2 = 26 * p.fst) :=
by
  sorry

end equation_has_12_solutions_l678_678611


namespace jaden_initial_cars_l678_678225

theorem jaden_initial_cars : 
  ∃ (x : ℕ), (x + 28 + 12 - 8 - 3 = 43) ∧ (x = 14) :=
by
  exists 14
  simp
  split
  sorry
  sorry

end jaden_initial_cars_l678_678225


namespace vector_orthogonality_l678_678535

variables (x : ℝ)

def vec_a := (x - 1, 2)
def vec_b := (1, x)

theorem vector_orthogonality :
  (vec_a x).fst * (vec_b x).fst + (vec_a x).snd * (vec_b x).snd = 0 ↔ x = 1 / 3 := by
  sorry

end vector_orthogonality_l678_678535


namespace correct_outfits_l678_678793

-- Define the children
inductive Child
| Alyna
| Bohdan
| Vika
| Grysha

open Child

-- Define colors
inductive Color
| Red
| Blue

open Color

-- Define clothes
structure Clothes :=
  (tshirt : Color)
  (shorts : Color)

-- Define initial conditions
def condition1 := Alyna = Clothes.mk Red _ ∧ Bohdan = Clothes.mk Red _ ∧ Alyna.shorts ≠ Bohdan.shorts
def condition2 := Vika.shorts = Blue ∧ Grysha.shorts = Blue ∧ Vika.tshirt ≠ Grysha.tshirt
def condition3 := Alyna.tshirt ≠ Vika.tshirt ∧ Alyna.shorts ≠ Vika.shorts

-- Define the solution (i.e., what needs to be proved)
def solution := 
  (Alyna = Clothes.mk Red Red) ∧
  (Bohdan = Clothes.mk Red Blue) ∧
  (Vika = Clothes.mk Blue Blue) ∧
  (Grysha = Clothes.mk Red Blue)

theorem correct_outfits : condition1 ∧ condition2 ∧ condition3 -> solution :=
by sorry

end correct_outfits_l678_678793


namespace percentage_increase_john_l678_678602

theorem percentage_increase_john (initial_salary new_salary : ℝ) (initial_salary = 60) (new_salary = 80) :
  ((new_salary - initial_salary) / initial_salary) * 100 = 33.33 := by
  sorry

end percentage_increase_john_l678_678602


namespace policeman_hats_difference_l678_678061

theorem policeman_hats_difference
  (hats_simpson : ℕ)
  (hats_obrien_now : ℕ)
  (hats_obrien_before : ℕ)
  (H : hats_simpson = 15)
  (H_hats_obrien_now : hats_obrien_now = 34)
  (H_hats_obrien_twice : hats_obrien_before = hats_obrien_now + 1) :
  hats_obrien_before - 2 * hats_simpson = 5 :=
by
  sorry

end policeman_hats_difference_l678_678061


namespace correct_decreasing_interval_l678_678297

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - π / 4)

def interval_a := Set.Icc (3 * π / 8) (7 * π / 8)
def interval_b := Set.Icc (-π / 8) (3 * π / 8)
def interval_c := Set.Icc (3 * π / 4) (5 * π / 4)
def interval_d := Set.Icc (-π / 4) (π / 4)

theorem correct_decreasing_interval : ∀ x ∈ interval_a, Real.deriv f x ≤ 0 :=
sorry

end correct_decreasing_interval_l678_678297


namespace num_two_digit_powers_of_3_l678_678957

theorem num_two_digit_powers_of_3 : 
  {n : ℕ // 10 ≤ 3^n ∧ 3^n < 100}.card = 2 :=
by
  sorry

end num_two_digit_powers_of_3_l678_678957


namespace max_students_seated_l678_678827

theorem max_students_seated (rows : ℕ) (seats_in_first_row : ℕ) (seat_increment : ℕ) : 
  rows = 15 → seats_in_first_row = 8 → seat_increment = 2 → 
  (∀ i, 1 ≤ i ∧ i ≤ rows → 6 + 2 * i = seats_in_first_row + seat_increment * (i - 1)) → 
  (∀ i, 1 ≤ i ∧ i ≤ rows → ⌈(6 + 2 * i) / 2⌉ = 3 + i) → 
  ∑ i in (finset.range rows).succ, (3 + i) = 165 := 
by
  sorry

end max_students_seated_l678_678827


namespace part1_part2_l678_678621

def f (x : ℝ) : ℝ := (x^2 + 2 * x + 1) / x

theorem part1 : f 3 + f (-3) = 4 :=
sorry

theorem part2 : ∀ x : ℝ, x ≥ 2 → ∀ a : ℝ, (∀ y : ℝ, y ≥ 2 → f y ≥ a) ↔ a ≤ 9 / 2 :=
sorry

end part1_part2_l678_678621


namespace sum_of_slopes_tangent_lines_passing_origin_l678_678510

noncomputable def circleC := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 2}

def is_tangent (l : ℝ → ℝ) (C : ℝ × ℝ → Prop) : Prop :=
∃ k : ℝ, ∀ x : ℝ, l x = k * x ∧ ∀ p ∈ C, p.2 = l p.1 → (p.1 - 2)^2 + (l p.1 + 1)^2 = 2

def sum_of_slopes_of_tangents (C : ℝ × ℝ → Prop) (O : ℝ × ℝ) : ℝ :=
∑ k in {k | ∃ l, is_tangent l C ∧ (O.2 = l O.1)}, k

theorem sum_of_slopes_tangent_lines_passing_origin :
  sum_of_slopes_of_tangents circleC (0, 0) = -2 :=
sorry

end sum_of_slopes_tangent_lines_passing_origin_l678_678510


namespace isosceles_right_triangle_side_length_l678_678001

theorem isosceles_right_triangle_side_length
  (a b : ℝ)
  (h_triangle : a = b ∨ b = a)
  (h_hypotenuse : xy > yz)
  (h_area : (1 / 2) * a * b = 9) :
  xy = 6 :=
by
  -- proof will go here
  sorry

end isosceles_right_triangle_side_length_l678_678001


namespace penguin_seafood_protein_l678_678272

theorem penguin_seafood_protein
  (digest : ℝ) -- representing 30% 
  (digested : ℝ) -- representing 9 grams 
  (h : digest = 0.30) 
  (h1 : digested = 9) :
  ∃ x : ℝ, digested = digest * x ∧ x = 30 :=
by
  sorry

end penguin_seafood_protein_l678_678272


namespace smallest_number_in_set_l678_678400

/-- The set of numbers we are considering -/
def num_set : set ℤ := {0, -1, 1, -5}

/-- The proposition stating that -5 is the smallest number in the set num_set -/
theorem smallest_number_in_set : ∃ m ∈ num_set, ∀ n ∈ num_set, m ≤ n ∧ m = -5 :=
by 
  sorry

end smallest_number_in_set_l678_678400


namespace entree_cost_l678_678176

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l678_678176


namespace cannot_partition_weights_l678_678765

open Nat
open Finset

theorem cannot_partition_weights :
  ¬ ∃ (A B : Finset ℕ), (A ∪ B = (range 24).erase 21) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by {
  sorry
}

end cannot_partition_weights_l678_678765


namespace inequality_proof_l678_678651

noncomputable def inequality_holds (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop :=
  (a ^ 2) / (b - 1) + (b ^ 2) / (a - 1) ≥ 8

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  inequality_holds a b ha hb :=
sorry

end inequality_proof_l678_678651


namespace auction_site_TVs_correct_l678_678060

-- Define the number of TVs Beatrice looked at in person
def in_person_TVs : Nat := 8

-- Define the number of TVs Beatrice looked at online
def online_TVs : Nat := 3 * in_person_TVs

-- Define the total number of TVs Beatrice looked at
def total_TVs : Nat := 42

-- Define the number of TVs Beatrice looked at on the auction site
def auction_site_TVs : Nat := total_TVs - (in_person_TVs + online_TVs)

-- Prove that the number of TVs Beatrice looked at on the auction site is 10
theorem auction_site_TVs_correct : auction_site_TVs = 10 :=
by
  sorry

end auction_site_TVs_correct_l678_678060


namespace min_moves_to_consecutive_Xs_l678_678737

theorem min_moves_to_consecutive_Xs (s : list char) : s.length = 40 ∧ 
  s.count 'X' = 20 ∧ s.count 'O' = 20 → 
  min_swaps_to_consecutive_Xs s = 200 :=
by sorry

end min_moves_to_consecutive_Xs_l678_678737


namespace arithmetic_sequence_and_sum_l678_678885

noncomputable def a_n (n : ℕ) : ℤ := 2 - n

def S (n : ℕ) : ℤ := if n = 3 then 0 else if n = 5 then -5 else 0

def b_n (n : ℕ) : ℚ := 1 / ((a_n (2 * n - 1)) * (a_n (2 * n + 1)))

theorem arithmetic_sequence_and_sum :
  (∀ n, S n = if n = 3 then 0 else if n = 5 then -5 else 0) ∧
  (∀ n, a_n n = 2 - n) ∧
  (∀ n, ∑ i in Finset.range n, b_n i = - (n : ℚ) / (2 * n - 1)) :=
by
  sorry

end arithmetic_sequence_and_sum_l678_678885


namespace carlotta_total_time_l678_678464

-- Define the main function for calculating total time
def total_time (performance_time practicing_ratio tantrum_ratio : ℕ) : ℕ :=
  performance_time + (performance_time * practicing_ratio) + (performance_time * tantrum_ratio)

-- Define the conditions from the problem
def singing_time := 6
def practicing_per_minute := 3
def tantrums_per_minute := 5

-- The expected total time based on the conditions
def expected_total_time := 54

-- The theorem to prove the equivalence
theorem carlotta_total_time :
  total_time singing_time practicing_per_minute tantrums_per_minute = expected_total_time :=
by
  sorry

end carlotta_total_time_l678_678464


namespace colors_of_clothes_l678_678807

-- Define the colors
inductive Color
| red : Color
| blue : Color

open Color

-- Variables and Definitions
variable (Alyna_tshirt Bohdan_tshirt Vika_tshirt Grysha_tshirt : Color)
variable (Alyna_shorts Bohdan_shorts Vika_shorts Grysha_shorts : Color)

-- Conditions
def condition1 := Alyna_tshirt = red ∧ Bohdan_tshirt = red ∧ Alyna_shorts ≠ Bohdan_shorts
def condition2 := (Vika_tshirt ≠ Grysha_tshirt) ∧ Vika_shorts = blue ∧ Grysha_shorts = blue
def condition3 := Vika_tshirt ≠ Alyna_tshirt ∧ Alyna_shorts ≠ Vika_shorts

-- Theorem statement
theorem colors_of_clothes :
  condition1 →
  condition2 →
  condition3 →
  (Alyna_tshirt = red ∧ Alyna_shorts = red) ∧
  (Bohdan_tshirt = red ∧ Bohdan_shorts = blue) ∧
  (Vika_tshirt = blue ∧ Vika_shorts = blue) ∧
  (Grysha_tshirt = red ∧ Grysha_shorts = blue) := by
  sorry

end colors_of_clothes_l678_678807


namespace count_balanced_integers_l678_678828

def is_balanced (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3) = d1 + (d2 + d3) ∧ (100 ≤ n) ∧ (n ≤ 999)

theorem count_balanced_integers : ∃ c, c = 330 ∧ ∀ n, 100 ≤ n ∧ n ≤ 999 → is_balanced n ↔ c = 330 :=
sorry

end count_balanced_integers_l678_678828


namespace entree_cost_14_l678_678174

-- Define the conditions as given in part a)
def total_cost (e d : ℕ) : Prop := e + d = 23
def entree_more (e d : ℕ) : Prop := e = d + 5

-- The theorem to be proved
theorem entree_cost_14 (e d : ℕ) (h1 : total_cost e d) (h2 : entree_more e d) : e = 14 := 
by 
  sorry

end entree_cost_14_l678_678174


namespace find_b_l678_678507

noncomputable def minimum_distance (A B C x y : ℝ) := 
  abs (A * x + B * y + C) / (real.sqrt (A^2 + B^2))

variable (A B C x y : ℝ)
variable (d : ℝ)
variable (b : ℝ)

-- Assume d is given
axiom given_distance : d = 3 * real.sqrt 2 / 2 - 1

-- Goal: Find the value of b
theorem find_b :
  b = 3 * real.sqrt 2 / 2 - 1 :=
sorry

end find_b_l678_678507


namespace sphere_surface_area_l678_678495

theorem sphere_surface_area (A B C D O : ℝ³) (h : O = (C + D) / 2) (max_vol : volume (tetrahedron A B C D) = 8 / 3) :
  surface_area (sphere O (distance O C)) = 16 * π :=
sorry

end sphere_surface_area_l678_678495


namespace statement1_statement4_l678_678898

-- Define the basic geometric relationships
def perpendicular_to (l : Type) (α : Type) : Prop := sorry
def lies_within (m : Type) (α : Type) : Prop := sorry

-- Given conditions
variables (l m β : Type)
variables (α : Type) [perpendicular_to l α] [lies_within m α]

-- Define the statements to be proven
theorem statement1 : (perpendicular_to l m) → (lies_within m β) := sorry

theorem statement4 : perpendicular_to l α → perpendicular_to m β := sorry

end statement1_statement4_l678_678898


namespace decreasing_function_in_interval_l678_678051

-- Define the functions
def f1 (x : ℝ) := x⁻¹
def f2 (x : ℝ) := x^2
def f3 (x : ℝ) := 2^x
def f4 (x : ℝ) := (x + 1)⁻¹

-- Define the interval (-1, 1)
def interval := set.Ioo (-1 : ℝ) (1 : ℝ)

-- Prove that f4 is the only function decreasing in the interval
theorem decreasing_function_in_interval :
  ∀ f (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4), 
  (∀ x (h : x ∈ interval), 
    f = f4 → 
    (0 < x → f' x < 0 ∧ 0 > x → f' x < 0)) := sorry

end decreasing_function_in_interval_l678_678051


namespace smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l678_678519

noncomputable def f (x : Real) : Real :=
  Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, ( ∀ x, f (x + T') = f x) → T ≤ T') := by
  sorry

theorem f_ge_negative_sqrt_3_in_interval :
  ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), f x ≥ -Real.sqrt 3 := by
  sorry

end smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l678_678519


namespace Jerry_throw_count_l678_678262

theorem Jerry_throw_count : 
  let interrupt_points := 5
  let insult_points := 10
  let throw_points := 25
  let threshold := 100
  let interrupt_count := 2
  let insult_count := 4
  let current_points := (interrupt_count * interrupt_points) + (insult_count * insult_points)
  let additional_points := threshold - current_points
  let throw_count := additional_points / throw_points
  in throw_count = 2 :=
by {
  have h1 : current_points = (2 * 5) + (4 * 10) := rfl,
  have h2 : current_points = 10 + 40 := by { rw [Nat.mul_def, Nat.add_def], },
  have h3 : current_points = 50 := by { rw Nat.add_def },
  have h4 : additional_points = 100 - 50 := rfl,
  have h5 : additional_points = 50 := by { rw Nat.sub_def },
  have h6 : throw_count = 50 / 25 := rfl,
  show throw_count = 2,
  rw Nat.div_def,
  exact h6
} sorry

end Jerry_throw_count_l678_678262


namespace find_f_2000_l678_678686

noncomputable def f : ℝ → ℝ := sorry

axiom f_property1 : ∀ (x y : ℝ), f (x + y) = f (x * y)
axiom f_property2 : f (-1/2) = -1/2

theorem find_f_2000 : f 2000 = -1/2 := 
sorry

end find_f_2000_l678_678686


namespace problem_2_l678_678918

def parametric_eqn_C1 (θ : ℝ) : ℝ × ℝ := (2 * cos θ, sin θ)

noncomputable def polar_eqn_C2 (θ : ℝ) : ℝ := 2 * sin θ

noncomputable def rect_coords (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, rho * sin θ)

-- given points
def M1 : ℝ × ℝ := (1, π / 2)
def M2 : ℝ × ℝ := (2, 0)

-- Polar coordinate equations
noncomputable def polar_eqn_C1 (ρ θ : ℝ) : Prop :=
  (ρ^2 * cos(θ)^2) / 4 + ρ^2 * sin(θ)^2 = 1

noncomputable def rect_eqn_C2 (x y : ℝ) : Prop :=
  (x^2 + (y - 1)^2 = 1)

-- OA and OB lengths based on parametric equation of C1
noncomputable def length_OA (ρ1 θ : ℝ) : ℝ := 
  ρ1

noncomputable def length_OB (ρ2 θ : ℝ) : ℝ := 
  ρ2

-- Main theorem to prove
theorem problem_2 : 
  ∀ (θ : ℝ) (ρ1 ρ2 : ℝ), 
  polar_eqn_C1 ρ1 θ ∧
  polar_eqn_C1 ρ2 (θ + π/2) →
  (1 / (length_OA ρ1 θ)^2) + (1 / (length_OB ρ2 (θ + π/2))^2) = 5/4 :=
sorry

end problem_2_l678_678918


namespace count_elements_with_leftmost_digit_7_l678_678617

def S : Set ℕ := {k | ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 5000 ∧ k = 7^n}

def leftmost_digit (a : ℕ) : ℕ := a / 10^(Nat.log10 a) -- function to find the leftmost digit

theorem count_elements_with_leftmost_digit_7 :
  let elements_with_7_as_leftmost_digit := {k ∈ S | leftmost_digit k = 7}
  ∃ (n : ℕ), n = 700 ∧ n = elements_with_7_as_leftmost_digit.card :=
by
  sorry

end count_elements_with_leftmost_digit_7_l678_678617


namespace circumcircle_area_of_triangle_l678_678745

noncomputable def triangle_sides (a b c : ℝ) : Prop :=
  a = 4 ∧ b = 3 ∧ c = 2

noncomputable def semi_perimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def herons_area (a b c : ℝ) (s : ℝ) : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def circumradius (a b c : ℝ) (K : ℝ) : ℝ :=
  a * b * c / (4 * K)

noncomputable def circumcircle_area (R : ℝ) : ℝ :=
  Real.pi * R^2

theorem circumcircle_area_of_triangle :
  ∃ a b c (K R A : ℝ), triangle_sides a b c →
  let s := semi_perimeter a b c in
  K = herons_area a b c s →
  R = circumradius a b c K →
  A = circumcircle_area R →
  A = 13.548 * Real.pi :=
by
  sorry

end circumcircle_area_of_triangle_l678_678745


namespace consumer_installment_credit_l678_678058

theorem consumer_installment_credit (C : ℝ) (A : ℝ) (h1 : A = 0.36 * C) 
    (h2 : 75 = A / 2) : C = 416.67 :=
by
  sorry

end consumer_installment_credit_l678_678058


namespace miles_run_on_tuesday_l678_678286

-- Defining the distances run on specific days
def distance_monday : ℝ := 4.2
def distance_wednesday : ℝ := 3.6
def distance_thursday : ℝ := 4.4

-- Average distance run on each of the days Terese runs
def average_distance : ℝ := 4
-- Number of days Terese runs
def running_days : ℕ := 4

-- Defining the total distance calculated using the average distance and number of days
def total_distance : ℝ := average_distance * running_days

-- Defining the total distance run on Monday, Wednesday, and Thursday
def total_other_days : ℝ := distance_monday + distance_wednesday + distance_thursday

-- The distance run on Tuesday can be defined as the difference between the total distance and the total distance on other days
theorem miles_run_on_tuesday : 
  total_distance - total_other_days = 3.8 :=
by
  sorry

end miles_run_on_tuesday_l678_678286


namespace topsoil_cost_l678_678329

theorem topsoil_cost (cost_per_cubic_foot : ℝ) (cubic_yards : ℝ) (conversion_factor : ℝ) : 
  cubic_yards = 8 →
  cost_per_cubic_foot = 7 →
  conversion_factor = 27 →
  ∃ total_cost : ℝ, total_cost = 1512 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l678_678329


namespace parabola_fixed_point_thm_l678_678165

-- Define the parabola condition
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

-- Define the focus condition
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Define the slope product condition
def slope_product (A B : ℝ × ℝ) : Prop :=
  (A.1 ≠ 0 ∧ B.1 ≠ 0) → ((A.2 / A.1) * (B.2 / B.1) = -1 / 3)

-- Define the fixed point condition
def fixed_point (A B : ℝ × ℝ) : Prop :=
  ∃ x : ℝ, A ≠ B ∧ (x = 12) ∧ ((A.2 - B.2) / (A.1 - B.1)) * 12 = A.2

-- Problem statement in Lean
theorem parabola_fixed_point_thm (A B : ℝ × ℝ) (p : ℝ) :
  (∃ O : ℝ × ℝ, O = (0, 0)) →
  (∃ C : ℝ → ℝ → ℝ → Prop, C = parabola) →
  (∃ F : ℝ × ℝ, focus F) →
  parabola A.2 A.1 p →
  parabola B.2 B.1 p →
  slope_product A B →
  fixed_point A B :=
by 
-- Sorry is used to skip the proof
sorry

end parabola_fixed_point_thm_l678_678165


namespace prove_clothing_colors_l678_678810

variable (color : Type)
variable [DecidableEq color]

variable (red blue : color)
variable (person : Type)
variable [DecidableEq person]

namespace ColorsProblem

noncomputable def colors : person → color × color
| "Alyna"  => (red, red)
| "Bohdan" => (red, blue)
| "Vika"   => (blue, blue)
| "Grysha" => (red, blue)
| _        => (red, red)  -- default case, should not be needed

def Alyna := "Alyna"
def Bohdan := "Bohdan"
def Vika := "Vika"
def Grysha := "Grysha"

def clothing_match (p : person) (shirt shorts : color) := colors p = (shirt, shorts)

theorem prove_clothing_colors :
  clothing_match Alyna red red ∧
  clothing_match Bohdan red blue ∧
  clothing_match Vika blue blue ∧
  clothing_match Grysha red blue
:=
by
  sorry

end ColorsProblem

end prove_clothing_colors_l678_678810


namespace quadratic_rewrite_l678_678427

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 25) (h2 : 2 * d * e = -40) (h3 : e^2 + f = -75) : d * e = -20 := 
by 
  sorry

end quadratic_rewrite_l678_678427


namespace area_of_triangle_N1N2N3_l678_678587

variables {ABC : Type*} [Triangle ABC]
variables {A B C D E F N1 N2 N3 : Point}
variable {K : ℝ}

-- Conditions
-- 1. Triangle ABC with area K
def area_ABC_eq_K : area ABC = K := sorry

-- 2. Points D, E, and F divide sides in a 3:1 ratio
def ratio_Deonsides_ABC : divides_ratio D A C = 3/1 ∧ divides_ratio E A B = 3/1 ∧ divides_ratio F B C = 3/1 := sorry

-- 3. Lines CD, AE, and BF are one-fourth of their respective sides
def lines_one_fourth : (length D C = 1/4 * length A C) ∧ (length E A = 1/4 * length A B) ∧ (length F B = 1/4 * length B C) := sorry

-- Proof statement
theorem area_of_triangle_N1N2N3 : area (Triangle.mk N1 N2 N3) = 5/8 * K :=
by
  exact area_of_triangle_N1N2N3

end area_of_triangle_N1N2N3_l678_678587


namespace symmedian_of_transformed_triangle_l678_678302

theorem symmedian_of_transformed_triangle 
  {A B C K A1 B1 C1 : Type}
  [IsSymmedianPoint K A B C]
  (h1 : IsIntersectionPointOf K A1 A ABC.Circumcircle)
  (h2 : IsIntersectionPointOf K B1 B ABC.Circumcircle)
  (h3 : IsIntersectionPointOf K C1 C ABC.Circumcircle) :
  IsSymmedianPoint K A1 B1 C1 :=
sorry

end symmedian_of_transformed_triangle_l678_678302


namespace ones_digit_542_mul_3_is_6_l678_678692

/--
Given that the ones (units) digit of 542 is 2, prove that the ones digit of 542 multiplied by 3 is 6.
-/
theorem ones_digit_542_mul_3_is_6 (h: ∃ n : ℕ, 542 = 10 * n + 2) : (542 * 3) % 10 = 6 := 
by
  sorry

end ones_digit_542_mul_3_is_6_l678_678692


namespace factorial_equation_solution_l678_678133

open Nat

theorem factorial_equation_solution (x y : ℕ) (z : ℤ) :
  (∃ x y z, x! + y! = 48 * z + 2017 ∧ ∃ k, x = 6 ∨ x = 7) ↔
  (x = 1 ∧ y = 6 ∧ z = -27) ∨ (x = 6 ∧ y = 1 ∧ z = -27) ∨
  (x = 1 ∧ y = 7 ∧ z = 63) ∨ (x = 7 ∧ y = 1 ∧ z = 63) :=
by
  sorry

end factorial_equation_solution_l678_678133


namespace square_side_length_l678_678236

-- Define the geometric setup and the necessary distances.
structure SquareAndPoint :=
  (A B C D E : Point)
  (side_length : ℝ)
  (square : square A B C D)
  (AE_CE_eq_9 : dist A E = 9 ∧ dist C E = 9)
  (BE_eq_8 : dist B E = 8)

-- Main theorem statement
theorem square_side_length (data : SquareAndPoint) :
  data.side_length = 7 - 4 * real.sqrt 2 := by
  sorry

end square_side_length_l678_678236


namespace traveling_time_l678_678743

theorem traveling_time (distance : ℝ) (car_speed : ℝ) (speed_difference : ℝ) (total_distance : ℝ) : 
  (distance / (car_speed + (car_speed + speed_difference))) = total_distance :=
by
  -- given conditions/definitions
  have h1: car_speed = 44 := rfl
  have h2: speed_difference = 8 := rfl
  have h3: distance = 384 := rfl
  have h4: total_distance = 4 := rfl
  -- skip proof for now, inserting sorry
  sorry

end traveling_time_l678_678743


namespace willie_cream_from_farm_l678_678108

variable (total_needed amount_to_buy amount_from_farm : ℕ)

theorem willie_cream_from_farm :
  total_needed = 300 → amount_to_buy = 151 → amount_from_farm = total_needed - amount_to_buy → amount_from_farm = 149 := by
  intros
  sorry

end willie_cream_from_farm_l678_678108


namespace ratio_of_areas_l678_678359

theorem ratio_of_areas (A B C F D : Type) (ABC : Triangle A B C) (CF AD : Line)
  (hCF : is_angle_bisector CF ∧ endpoint CF A ∧ endpoint CF F ∧ endpoint CF C)
  (hAD : is_angle_bisector AD ∧ endpoint AD A ∧ endpoint AD D ∧ endpoint AD B)
  (hRatio : length AB : length AC : length BC = 21 : 28 : 20) :
  let S_AFD := area (Triangle A F D)
  let S_ABC := area (Triangle A B C)
  in S_AFD / S_ABC = 1 / 4 := by
  sorry

end ratio_of_areas_l678_678359


namespace simplify_expression_equals_two_l678_678667

noncomputable def simplify_expression : ℝ :=
  2 - 2 / (2 + 2 * real.sqrt 2) + 2 / (2 - 2 * real.sqrt 2)

theorem simplify_expression_equals_two : simplify_expression = 2 :=
by 
  unfold simplify_expression
  sorry

end simplify_expression_equals_two_l678_678667


namespace area_ratio_APQ_ABC_l678_678616

variable {ABC : Type} [fintype ABC] [semiring ABC]
variable (A B C M N P Q : ABC)
variable (s : ℝ)

-- Define midpoints M and N
def is_midpoint (M : ABC) (A B : ABC) : Prop := (dist A M = dist M B) ∧ (dist B M = dist M A)
def is_midpoint (N : ABC) (A C : ABC) : Prop := (dist A N = dist N C) ∧ (dist C N = dist N A)

-- P is a point on AB between A and M
def on_interval (P A M : ABC) : Prop := dist A P + dist P M = dist A M

-- Line through P parallel to MN
def parallel (P Q M N : ABC) : Prop := 
∃ k : ℝ, Q = M + k • (N - M) ∧ P = M + k • (N - M)

theorem area_ratio_APQ_ABC 
(h_midpointM : is_midpoint M A B)
(h_midpointN : is_midpoint N A C)
(h_P_on_AM : on_interval P A M)
(h_parallel : parallel P Q M N)
: 0 ≤ s ∧ s ≤ 1/4 :=
sorry

end area_ratio_APQ_ABC_l678_678616


namespace remainder_div_modulo_l678_678755

theorem remainder_div_modulo (N : ℕ) (h1 : N % 19 = 7) : N % 20 = 6 :=
by
  sorry

end remainder_div_modulo_l678_678755


namespace johnny_jogging_speed_l678_678228

theorem johnny_jogging_speed :
    ∃ v : ℝ, (v > 0) ∧
    let d := 6.461538461538462 in
    let bus_speed := 21 in
    let total_time := 1 in
    (d / v) + (d / bus_speed) = total_time ∧
    abs (v - 9.33) < 0.01 :=
by
  sorry

end johnny_jogging_speed_l678_678228


namespace kendra_total_earnings_l678_678233

theorem kendra_total_earnings (laurel2014 kendra2014 kendra2015 : ℕ) 
  (h1 : laurel2014 = 30000)
  (h2 : kendra2014 = laurel2014 - 8000)
  (h3 : kendra2015 = 1.20 * laurel2014) :
  kendra2014 + kendra2015 = 58000 :=
by
  sorry

end kendra_total_earnings_l678_678233


namespace monotonicity_f_m_eq_0_f_gt_e_div_2_minus_1_l678_678160

noncomputable theory

-- Define the function f
def f (x m : ℝ) := Real.exp x - m * x^2 - 2 * x

-- Part (I): Monotonicity of f when m=0
theorem monotonicity_f_m_eq_0 :
  (∀ x : ℝ, f x 0 = Real.exp x - 2 * x) ∧
  (∀ x : ℝ, (x < Real.log 2 → deriv (λ x, f x 0) x < 0) ∧ (x > Real.log 2 → deriv (λ x, f x 0) x > 0)) :=
by 
  sorry

-- Part (II): Inequality for f(x) under given conditions
theorem f_gt_e_div_2_minus_1 (m : ℝ) (h : m < Real.exp 1 / 2 - 1) (x : ℝ) (hx : 0 ≤ x) :
  f x m > Real.exp 1 / 2 - 1 :=
by 
  sorry

end monotonicity_f_m_eq_0_f_gt_e_div_2_minus_1_l678_678160


namespace prob_c_not_adjacent_to_a_or_b_l678_678701

-- Definitions for the conditions
def num_students : ℕ := 7
def a_and_b_together : Prop := true
def c_on_edge : Prop := true

-- Main theorem: probability c not adjacent to a or b under given conditions
theorem prob_c_not_adjacent_to_a_or_b
  (h1 : a_and_b_together)
  (h2 : c_on_edge) :
  ∃ (p : ℚ), p = 0.8 := by
  sorry

end prob_c_not_adjacent_to_a_or_b_l678_678701


namespace binomial_to_incomplete_beta_l678_678255

noncomputable def binom_cdf (n : ℕ) (p : ℝ) (m : ℕ) : ℝ :=
∑ k in finset.range (m + 1), nat.choose n k * p^k * (1 - p)^(n - k)

noncomputable def beta_function (a b : ℝ) : ℝ :=
∫ x in 0..1, x^(a - 1) * (1 - x)^(b - 1)

noncomputable def incomplete_beta (m n : ℕ) (p : ℝ) : ℝ :=
∫ x in p..1, x^m * (1 - x)^(n - m - 1)

theorem binomial_to_incomplete_beta (n : ℕ) (p : ℝ) (m : ℕ) (h₀ : 0 ≤ p) (h₁ : p ≤ 1) : 
  binom_cdf n p m = (1 / beta_function (m + 1) (n - m)) * incomplete_beta m n p := 
sorry

end binomial_to_incomplete_beta_l678_678255


namespace max_n_for_sum_of_squares_l678_678714

theorem max_n_for_sum_of_squares (k : ℕ → ℕ) :
  (∀ i j, i ≠ j → k i ≠ k j) ∧ (∑ i in finset.range 19, (k i)^2 = 2500) → 
  ¬ (∃ n, n > 19 ∧ ∀ k' : ℕ → ℕ, (∀ i j, i ≠ j → k' i ≠ k' j) → ∑ i in finset.range n, (k' i)^2 = 2500) :=
sorry

end max_n_for_sum_of_squares_l678_678714


namespace f_2016_eq_cos_l678_678126

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := λ x, Real.cos x
| (n + 1) := λ x, deriv (f n) x

theorem f_2016_eq_cos (x : ℝ) : f 2016 x = Real.cos x :=
by sorry

end f_2016_eq_cos_l678_678126
