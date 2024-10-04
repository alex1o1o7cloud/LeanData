import Mathlib

namespace integral_absolute_value_l221_221655

noncomputable def integral_example : ℝ :=
  ∫ x in -2..2, |x^2 - 2x|

theorem integral_absolute_value :
  integral_example = 8 :=
by
  sorry

end integral_absolute_value_l221_221655


namespace triangle_ratio_l221_221766

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ)
  (hAB : ∠A = asin (b / c * sin ∠B))
  (hBC : ∠B + ∠C = π - ∠A) 
  (h_cond : b * cos ∠C + c * cos ∠B = sqrt 2 * b) : 
  a / b = sqrt 2 := 
sorry

end triangle_ratio_l221_221766


namespace greatest_x_value_l221_221669

theorem greatest_x_value :
  ∃ x, (∀ y, (y ≠ 6) → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧ (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧ x ≠ 6 ∧ x = -3 :=
sorry

end greatest_x_value_l221_221669


namespace circle_distance_to_line_l221_221368

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221368


namespace megacorp_fine_l221_221824

noncomputable def daily_mining_profit := 3000000
noncomputable def daily_oil_profit := 5000000
noncomputable def monthly_expenses := 30000000

def total_daily_profit := daily_mining_profit + daily_oil_profit
def annual_profit := 365 * total_daily_profit
def annual_expenses := 12 * monthly_expenses
def net_annual_profit := annual_profit - annual_expenses
def fine := (net_annual_profit * 1) / 100

theorem megacorp_fine : fine = 25600000 := by
  -- the proof steps will go here
  sorry

end megacorp_fine_l221_221824


namespace min_time_to_cover_distance_l221_221573

variable (distance : ℝ := 3)
variable (vasya_speed_run : ℝ := 4)
variable (vasya_speed_skate : ℝ := 8)
variable (petya_speed_run : ℝ := 5)
variable (petya_speed_skate : ℝ := 10)

theorem min_time_to_cover_distance :
  ∃ (t : ℝ), t = 0.5 ∧
    ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ distance ∧ 
    (distance - x) / vasya_speed_run + x / vasya_speed_skate = t ∧
    x / petya_speed_run + (distance - x) / petya_speed_skate = t :=
by
  sorry

end min_time_to_cover_distance_l221_221573


namespace randy_vacation_days_l221_221464

def hours_per_day := 5
def practice_days_per_week := 5
def total_hours_needed := 10000
def total_years := 8

def total_weeks_in_years (years : ℕ) : ℕ := years * 52
def practice_days_in_weeks (weeks : ℕ) : ℕ := weeks * practice_days_per_week
def total_practice_hours (days : ℕ) (hours_per_day : ℕ) : ℕ := days * hours_per_day
def days_to_hours (hours : ℕ) (hours_per_day : ℕ) : ℕ := hours / hours_per_day
def vacation_days_per_year (total_days : ℕ) (years : ℕ) : ℕ := total_days / years

theorem randy_vacation_days :
  let weeks := total_weeks_in_years total_years,
      practice_days := practice_days_in_weeks weeks,
      total_hours := total_practice_hours practice_days hours_per_day,
      excess_hours := total_hours - total_hours_needed,
      vacation_days := days_to_hours excess_hours hours_per_day
  in vacation_days_per_year vacation_days total_years = 10 := by
  sorry

end randy_vacation_days_l221_221464


namespace correct_calculation_l221_221559

theorem correct_calculation (x y : ℝ) : (x * y^2) ^ 2 = x^2 * y^4 :=
by
  sorry

end correct_calculation_l221_221559


namespace propositions_correct_l221_221870

theorem propositions_correct :
  (∀ (P1 P2 P3 P4: ℝ^3), (collinear P1 P2 P3 → coplanar P1 P2 P3 P4)) ∧ -- Proposition 1
  (trapezoid_remains_trapezoid_oblique_axonometric) -- Proposition 3
  :=
by
  split;
  { sorry }

-- Definitions
def collinear (P1 P2 P3: ℝ^3) : Prop := ∃ (a b: ℝ), a ≠ b ∧ P2 = a • P1 + P3 ∧ P2 = b • P3 + P1
def coplanar (P1 P2 P3 P4: ℝ^3) : Prop := ∃ α β γ δ: ℝ, α + β + γ + δ = 1 ∧ α • P1 + β • P2 + γ • P3 + δ • P4 = 0

def trapezoid_remains_trapezoid_oblique_axonometric : Prop := 
  ∀ (A B C D: ℝ^3), is_trapezoid A B C D → is_trapezoid (oblique_axometric_projection A) (oblique_axometric_projection B) (oblique_axometric_projection C) (oblique_axometric_projection D)

def is_trapezoid (A B C D: ℝ^3) : Prop := 
  let M := midpoint B C in
  collinear A D M ∧ collinear B C M

def oblique_axometric_projection (P: ℝ^3) : ℝ^3 := sorry -- Define oblique axonometric drawing projection here

end propositions_correct_l221_221870


namespace distance_between_points_l221_221543

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_between_points :
  distance (2, -2) (8, 8) = real.sqrt 136 :=
by
  sorry

end distance_between_points_l221_221543


namespace kolya_made_mistake_l221_221797

theorem kolya_made_mistake (ab cd effe : ℕ)
  (h_eq : ab * cd = effe)
  (h_eff_div_11 : effe % 11 = 0)
  (h_ab_cd_not_div_11 : ab % 11 ≠ 0 ∧ cd % 11 ≠ 0) :
  false :=
by
  -- Note: This is where the proof would go, but we are illustrating the statement only.
  sorry

end kolya_made_mistake_l221_221797


namespace find_n_l221_221254

theorem find_n
  (a a1 a2 ... an : ℕ)
  (n : ℕ)
  (h1 : (∑ k in range n, (1 + x)^k) = a + ∑ k in range n, ai * x^i)
  (h2 : ∑ k in range (n-1), ai = 29 - n) :
  n = 4 :=
sorry

end find_n_l221_221254


namespace distance_from_center_of_circle_to_line_l221_221329

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221329


namespace theater_ticket_sales_l221_221527

theorem theater_ticket_sales 
  (total_tickets : ℕ) (price_adult_ticket : ℕ) (price_senior_ticket : ℕ) (senior_tickets_sold : ℕ) 
  (Total_tickets_condition : total_tickets = 510)
  (Price_adult_ticket_condition : price_adult_ticket = 21)
  (Price_senior_ticket_condition : price_senior_ticket = 15)
  (Senior_tickets_sold_condition : senior_tickets_sold = 327) : 
  (183 * 21 + 327 * 15 = 8748) :=
by
  sorry

end theater_ticket_sales_l221_221527


namespace value_of_a_plus_b_l221_221298

theorem value_of_a_plus_b (a b : ℝ) (h1 : sqrt 44 = 2 * sqrt a) (h2 : sqrt 54 = 3 * sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l221_221298


namespace correct_inequality_l221_221752

variable (a b : ℝ)

theorem correct_inequality (h : a > b) : a - 3 > b - 3 :=
by
  sorry

end correct_inequality_l221_221752


namespace conditional_prob_l221_221932

-- Define the conditions
def set := {1, 2, 3, 4, 5}
def is_odd (n : ℕ) : Prop := n % 2 = 1
def event_A : Set ℕ := { x | x ∈ set ∧ is_odd x }
def event_B {x} : Set ℕ := { y | y ∈ set ∧ is_odd y ∧ y ≠ x }

-- Define the probabilities
def C {α : Type*} [Fintype α] [DecidableEq α] (s : Finset α) (k : ℕ) : ℕ :=
  Nat.choose s.card k

noncomputable def P (s t : Set ℕ) : ℚ :=
  ↑(s.inter t).card / ↑s.card

/-- To prove that the conditional probability P(B|A) equals to 1/2 -/
theorem conditional_prob :
  P (event_A) set = 3 / 5 ∧ P (event_A.inter (event_B 1)) set = 3 / 10 →
  P (event_B 1 | event_A) = 1 / 2 :=
by
  intros h
  sorry

end conditional_prob_l221_221932


namespace CodyDumplings_l221_221634

def CodyCookedFirstBatch := 14
def CodyAte50PercentFirstBatch := 0.5
def CodyCookedSecondBatch := 20
def CodyShared1_4RemainingFirstBatch := 1 / 4
def CodyShared2_5SecondBatch := 2 / 5
def FriendsAte15PercentTotal := 0.15

theorem CodyDumplings: 
  let A := CodyCookedFirstBatch 
  let P1 := CodyAte50PercentFirstBatch * A 
  let Q1 := CodyShared1_4RemainingFirstBatch * (A - P1).toNat 
  let B := CodyCookedSecondBatch
  let Q2 := CodyShared2_5SecondBatch * B
  let TotalRemaining := (A - P1 - Q1).toNat + (B - Q2).toNat 
  let P2 := FriendsAte15PercentTotal * TotalRemaining
  (TotalRemaining - P2.toNat) = 16 := 
by
  sorry

end CodyDumplings_l221_221634


namespace stability_of_triangles_in_structures_l221_221177

theorem stability_of_triangles_in_structures :
  ∀ (bridges cable_car_supports trusses : Type),
  (∃ (triangular_structures : Type), (triangular_structures → bridges) ∧ (triangular_structures → cable_car_supports) ∧ (triangular_structures → trusses)) →
  (∀ (triangle : Type), is_stable triangle) →
  ∀ (structure : Type), uses_triangle structure → is_stable structure :=
begin
  sorry
end

end stability_of_triangles_in_structures_l221_221177


namespace moving_point_satisfies_equation_l221_221241

noncomputable def distance_to_point (M : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
  real.sqrt ((M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2)

noncomputable def distance_to_line (M : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  real.abs (M.2 - y_line)

theorem moving_point_satisfies_equation (M : ℝ × ℝ) (h : distance_to_point M (0, 2) = distance_to_line M (-4) - 2) :
  M.1 ^ 2 = 8 * M.2 :=
sorry

end moving_point_satisfies_equation_l221_221241


namespace area_not_uniquely_determined_l221_221791

theorem area_not_uniquely_determined (A B C D E F : Point) 
  (h_area_ABC : area A B C = 16) (h_AD : dist A D = 3) (h_DB : dist D B = 5) 
  (h_DE_on_AB : D ∈ line_through A B) (h_E_on_BC : E ∈ line_through B C) 
  (h_F_on_CA : F ∈ line_through C A) (h_AreaEq : area A B E = area D B E F) : 
  ∃ area_ABE, ¬unique (area A B E) :=
begin
  sorry
end

end area_not_uniquely_determined_l221_221791


namespace max_value_y_eq_neg10_l221_221989

open Real

theorem max_value_y_eq_neg10 (x : ℝ) (hx : x > 0) : 
  ∃ y, y = 2 - 9 * x - 4 / x ∧ (∀ z, (∃ (x' : ℝ), x' > 0 ∧ z = 2 - 9 * x' - 4 / x') → z ≤ y) ∧ y = -10 :=
by
  sorry

end max_value_y_eq_neg10_l221_221989


namespace ratio_area_octagons_correct_l221_221107

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221107


namespace alice_bob_sum_proof_l221_221657

noncomputable def alice_bob_sum_is_22 : Prop :=
  ∃ A B : ℕ, (1 ≤ A ∧ A ≤ 50) ∧ (1 ≤ B ∧ B ≤ 50) ∧ (B % 3 = 0) ∧ (∃ k : ℕ, 2 * B + A = k^2) ∧ (A + B = 22)

theorem alice_bob_sum_proof : alice_bob_sum_is_22 :=
sorry

end alice_bob_sum_proof_l221_221657


namespace sticker_count_l221_221512

def stickers_per_page : ℕ := 25
def num_pages : ℕ := 35
def total_stickers : ℕ := 875

theorem sticker_count : num_pages * stickers_per_page = total_stickers :=
by {
  sorry
}

end sticker_count_l221_221512


namespace find_certain_fraction_l221_221215

theorem find_certain_fraction (x y : ℚ): 
  (3 / 5) / (6 / 7) = (7 / 15) / (x / y) → 
  x / y = 2 / 3 := 
by 
  intro h,
  sorry

end find_certain_fraction_l221_221215


namespace probability_real_complex_number_is_one_sixth_l221_221010

-- We define the conditions of the problem

def is_real (z : ℂ) : Prop := z.im = 0

def dice_outcomes : Finset (ℕ × ℕ) := 
  Finset.univ.product Finset.univ.filter (λ p : ℕ × ℕ, 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6)

def complex_number (m n : ℕ) : ℂ := (⟨m, 0⟩ : ℂ) * (⟨n, 0⟩ - ⟨0, m⟩)

def favorable_outcomes : Finset (ℕ × ℕ) :=
  dice_outcomes.filter (λ p, is_real (complex_number p.1 p.2))

def probability : ℚ := favorable_outcomes.card / dice_outcomes.card

-- The main theorem to prove
theorem probability_real_complex_number_is_one_sixth :
  probability = 1 / 6 :=
by
  sorry

end probability_real_complex_number_is_one_sixth_l221_221010


namespace cheesecake_percentage_eaten_l221_221197

-- Definitions based on the conditions
def calories_per_slice : ℕ := 350
def total_calories : ℕ := 2800
def slices_eaten_by_kiley : ℕ := 2

-- Prove the percentage of cheesecake Kiley ate is 25%
theorem cheesecake_percentage_eaten :
  ((slices_eaten_by_kiley * calories_per_slice) / total_calories.toRat) * 100 = 25 := 
by
  sorry

end cheesecake_percentage_eaten_l221_221197


namespace transform_graph_l221_221899

def f (x : ℝ) := sorry -- This represents the unknown function f(x) that we will prove to be a specific form

theorem transform_graph :
  (∀ (x : ℝ), f(x) = sin (x - π / 3)) ↔ 
  (∀ (x : ℝ), 
     let y := f(x) in  -- Start with y = f(x)
     let first_transform := λ x, f (2 * x) in  -- Applying transformations
     first_transform (x - π / 6)) = sin(2 * x) :=
by sorry

end transform_graph_l221_221899


namespace problem1_problem2_problem3_l221_221731

section

def f (a : ℝ) (x : ℝ) := a * x + Real.log x

-- Problem (I)
theorem problem1 (a : ℝ) (h : a = 2) : 
    tangent_line (f a) 1 = 3 * (x - 1) + 2 ↔ 3 * x - y - 1 = 0 := 
sorry

-- Problem (II)
theorem problem2 (a : ℝ) :
  (∀ x > 0, deriv (f a) x > 0) ↔ (0 ≤ a) ∨ ((a < 0) ∧ (∀ x > 0, 0 < x → x < -1/a)) :=
sorry

-- Problem (III)
theorem problem3 :
  (∀ x > 0, f (a, x) < 2) → a < -1/Real.exp 3 := 
sorry

end

end problem1_problem2_problem3_l221_221731


namespace sum_dk_squared_l221_221192

noncomputable def d_k (k : ℕ) : ℝ := 
  k + 1 / (3 * k + 1 / (3 * k + 1 / (3 * k + ...)))

theorem sum_dk_squared :
  (∑ k in Finset.range 10, (d_k (k + 1))^2) = 1165 := 
sorry

end sum_dk_squared_l221_221192


namespace altitude_of_triangle_l221_221981

theorem altitude_of_triangle
  (a b c : ℝ)
  (h₀ : a = 13)
  (h₁ : b = 14)
  (h₂ : c = 15)
  (s : ℝ := (a + b + c) / 2) :
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  in (2 * A) / c = 11.2 :=
by
  intros
  sorry

end altitude_of_triangle_l221_221981


namespace calculate_x_l221_221970

theorem calculate_x :
  let x := 289 + 2 * 17 * 8 + 64
  in x = 625 := by
  let x := 289 + 2 * 17 * 8 + 64
  show x = 625 from sorry

end calculate_x_l221_221970


namespace octagon_area_ratio_l221_221152

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221152


namespace equation_of_circleO_chord_length_l221_221694

-- Define circle C
def circleC (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 4

-- Define the equation of line L
def lineL (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Center of circle O at the origin
def centerO : ℝ × ℝ := (0, 0)

-- Step 1: Prove that equation of circle O is x^2 + y^2 = 9
theorem equation_of_circleO (x y : ℝ) :
  (∃ r : ℝ, r = 3 ∧ (O_center: x^2 + y^2 = r^2)) :=
  sorry

-- Step 2: Prove the length of the chord cut by the line on circle O is 12√5/5
theorem chord_length (r : ℝ) (h_eq : r = 3) :
  let d : ℝ := abs 3 / sqrt 5
  let l : ℝ := 2 * sqrt (r^2 - d^2)
  (l = 12 * sqrt 5 / 5) :=
  sorry

end equation_of_circleO_chord_length_l221_221694


namespace diameter_outer_boundary_correct_l221_221941

noncomputable def diameter_outer_boundary 
  (D_fountain : ℝ)
  (w_gardenRing : ℝ)
  (w_innerPath : ℝ)
  (w_outerPath : ℝ) : ℝ :=
  let R_fountain := D_fountain / 2
  let R_innerPath := R_fountain + w_gardenRing
  let R_outerPathInner := R_innerPath + w_innerPath
  let R_outerPathOuter := R_outerPathInner + w_outerPath
  2 * R_outerPathOuter

theorem diameter_outer_boundary_correct :
  diameter_outer_boundary 10 12 3 4 = 48 := by
  -- skipping proof
  sorry

end diameter_outer_boundary_correct_l221_221941


namespace determine_q_l221_221457

noncomputable def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem determine_q
  (p q r s t : ℝ)
  (h_roots : ∀ x, g p q r s t x = 0 ↔ x = -2 ∨ x = 0 ∨ x = 1 ∨ x = 3)
  (h_at_zero : g p q r s t 0 = 1) :
  q = -2 :=
begin
  sorry
end

end determine_q_l221_221457


namespace tennis_club_games_l221_221863

theorem tennis_club_games
  (members : Finset ℕ) (games : Finset (Finset ℕ))
  (h_members : members.card = 20)
  (h_games : games.card = 14)
  (h_game_size : ∀ g ∈ games, g.card = 2)
  (h_member_plays : ∀ m ∈ members, ∃ g ∈ games, m ∈ g) :
  ∃ sub_games ⊆ games, sub_games.card = 6 ∧
    (Finset.bUnion sub_games id).card = 12 :=
sorry

end tennis_club_games_l221_221863


namespace area_of_triangle_MEF_correct_l221_221400

noncomputable def area_of_triangle_MEF : ℝ :=
  let r := 10
  let chord_length := 12
  let parallel_segment_length := 15
  let angle_MOA := 30.0
  (1 / 2) * chord_length * (2 * Real.sqrt 21)

theorem area_of_triangle_MEF_correct :
  area_of_triangle_MEF = 12 * Real.sqrt 21 :=
by
  -- proof will go here
  sorry

end area_of_triangle_MEF_correct_l221_221400


namespace total_students_l221_221518

-- Definitions based on conditions
variable (T M Z : ℕ)  -- T for Tina's students, M for Maura's students, Z for Zack's students

-- Conditions as hypotheses
axiom h1 : T = M  -- Tina's classroom has the same amount of students as Maura's
axiom h2 : Z = (T + M) / 2  -- Zack's classroom has half the amount of total students between Tina and Maura's classrooms
axiom h3 : Z = 23  -- There are 23 students in Zack's class when present

-- Proof statement
theorem total_students : T + M + Z = 69 :=
  sorry

end total_students_l221_221518


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221121

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221121


namespace length_of_leg_smallest_triangle_l221_221995

theorem length_of_leg_smallest_triangle (hypotenuse_largest_triangle : ℝ) 
  (is_45_45_90 : ∀ (a b : ℝ), a = b * 1/√2) 
  (hypotenuse_connection : ∀ (a b c : ℝ), hypotenuse_largest_triangle = a 
     ∧ a = b * √2 ∧ b = c * √2 → c = 4) :
  hypotenuse_largest_triangle = 16 → 
  ∀ a b c, is_45_45_90 a b ∧ hypotenuse_connection a b c → c = 4 :=
begin
  sorry
end

end length_of_leg_smallest_triangle_l221_221995


namespace area_ratio_of_octagons_is_4_l221_221102

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221102


namespace area_ratio_of_octagons_is_4_l221_221092

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221092


namespace Anna_phone_chargers_l221_221625

-- Define the conditions and the goal in Lean
theorem Anna_phone_chargers (P L : ℕ) (h1 : L = 5 * P) (h2 : P + L = 24) : P = 4 :=
by
  sorry

end Anna_phone_chargers_l221_221625


namespace circle_distance_to_line_l221_221373

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221373


namespace octagon_area_ratio_l221_221148

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221148


namespace probability_sum_at_least_fifteen_l221_221470

theorem probability_sum_at_least_fifteen (s : Finset ℕ) (h_s : s = (Finset.range 14).map Nat.succ) :
  (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + c ≥ 15 ∧ a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a < b ∧ b < c) →
  ((369 : ℚ) ^ -1 * 76 = 19 / 91) :=
by
  sorry

end probability_sum_at_least_fifteen_l221_221470


namespace rotation_problem_l221_221183

variable (A B C : Point)

theorem rotation_problem (h₁ : rotate A B 480 = C) (h₂ : rotate A B x = C) (hx : x < 360) : x = 240 := 
by
  sorry

end rotation_problem_l221_221183


namespace lemon_drink_increase_l221_221587

-- Define the initial percentage of lemon juice
def initial_juice_perc : ℝ := 0.15

-- Define the new percentage of lemon juice
def new_juice_perc : ℝ := 0.10

-- Define the relationship between the initial volume and new volume
def volume_increase (x y : ℝ) : Prop :=
  y = 1.5 * x

-- The main statement that needs to be proven
theorem lemon_drink_increase (x y : ℝ) (h1 : initial_juice_perc * x = new_juice_perc * y) :
  volume_increase x y :=
begin
  sorry
end

end lemon_drink_increase_l221_221587


namespace distance_from_center_of_circle_to_line_l221_221324

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221324


namespace comparison_of_a_b_c_l221_221805

noncomputable def a := Real.log 2 / Real.log 0.3
noncomputable def b := Real.sqrt 0.3
noncomputable def c := 0.2 ^ (-0.3)

theorem comparison_of_a_b_c : a < b ∧ b < c := 
by {
  have h1 : a = log 2 / log 0.3 := rfl,
  have h2 : b = sqrt 0.3 := rfl,
  have h3 : c = 0.2 ^ (-0.3) := rfl,
  sorry
}

end comparison_of_a_b_c_l221_221805


namespace solve_log_inequality_l221_221432

noncomputable def f (a x : ℝ) : ℝ := a^(Real.logb 10 (x^2 - 2*x + 3))

theorem solve_log_inequality (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)
    (h₃ : ∃ x : ℝ, f a x = Real.sup (fun x => a^(Real.logb 10 (x^2 - 2*x + 3)))) :
    { x : ℝ | Real.logb a (x^2 - 5*x + 7) > 0 } = set.Ioo 2 3 :=
by
  sorry

end solve_log_inequality_l221_221432


namespace max_sin_A_plus_sin_C_l221_221260

variables {a b c S : ℝ}
variables {A B C : ℝ}

-- Assume the sides of the triangle
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Assume the angles of the triangle
variables (hA : A > 0) (hB : B > (Real.pi / 2)) (hC : C > 0)
variables (hSumAngles : A + B + C = Real.pi)

-- Assume the relationship between the area and the sides
variables (hArea : S = (1/2) * a * c * Real.sin B)

-- Assume the given equation holds
variables (hEquation : 4 * b * S = a * (b^2 + c^2 - a^2))

-- The statement to prove
theorem max_sin_A_plus_sin_C : (Real.sin A + Real.sin C) ≤ 9 / 8 :=
sorry

end max_sin_A_plus_sin_C_l221_221260


namespace midpoint_locus_parallel_l221_221217

-- Definitions
variables {l_1 l_2 : Line} -- Given parallel lines l_1 and l_2

-- Assumption of parallelism
axiom parallel (l1 l2 : Line) : Prop

noncomputable def midpoint (A B : Point) : Point := 
  let x := (A.x + B.x) / 2
  let y := (A.y + B.y) / 2
  {x := x, y := y}

theorem midpoint_locus_parallel (l_1 l_2 : Line) (A : Point) (h_A : A ∈ l_1) (B : Point) (h_B : B ∈ l_2) 
(h_parallel : parallel l_1 l_2) : 
∃ l, (∀ A B, A ∈ l_1 → B ∈ l_2 → midpoint A B ∈ l) ∧ parallel l l_1 :=
by
  sorry

end midpoint_locus_parallel_l221_221217


namespace minimum_perimeter_condition_l221_221483

def fractional_part (x : ℝ) := x - ⌊x⌋

noncomputable def smallest_perimeter_triangle : ℕ :=
  let l := 1500
  let m := 1000
  let n := 500
  l + m + n

theorem minimum_perimeter_condition (l m n : ℕ) :  
  l > m → m > n → 
  (fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧ 
   fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)) →
  (l + m + n) = 3000 :=
begin
  sorry
end

end minimum_perimeter_condition_l221_221483


namespace circle_tangent_distance_l221_221312

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221312


namespace triangle_area_l221_221900

variables (K L M N X Y Z : Type)
open Set

-- Condition definitions
def KX_XL_ratios := (K, X, L) ∈ [{kx // KX kx = 3 * XL xl}, {xl // XL xl = 2 * kx}]
def KY_YN_ratios := (K, Y, N) ∈ [{ky // KY ky = 4 * YN yn}, {yn // YN yn = 1 * ky}]
def NZ_ZM_ratios := (N, Z, M) ∈ [{nz // NZ nz = 2 * ZM zm}, {zm // ZM zm = 3 * nz}]
def area_of_KLMN := @intervalintegral.measure_of_set [[1]]

-- Problem statement
theorem triangle_area (square_area : area_of_KLMN K L M N = 1)
                      (KX_XL : KX_XL_ratios)
                      (KY_YN : KY_YN_ratios)
                      (NZ_ZM : NZ_ZM_ratios) :
                      area (triangle K X Y Z) = 3 / 10 := sorry

end triangle_area_l221_221900


namespace sum_distances_squared_l221_221715

theorem sum_distances_squared (a b x y : ℝ) :
  let A := (-a, -b)
  let B := (a, -b)
  let C := (a, b)
  let D := (-a, b)
  let P := (x, y)
  let dist2 (p1 p2 : ℝ × ℝ) : ℝ :=
    (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
in dist2 P A + dist2 P C = dist2 P B + dist2 P D :=
by
  sorry

end sum_distances_squared_l221_221715


namespace find_pairs_arithmetic_geometric_progression_l221_221644

theorem find_pairs_arithmetic_geometric_progression :
  let s : ℝ := 15
      a_eq : ∀ (a b : ℝ), a = (s + b) / 2
      geometric_cond : ∀ (a b : ℝ) (r : ℝ), ab = s * r^3
  in ∃! (a b : ℝ),  (s, a, b, a * b) is an arithmetic progression
    ∧ a * b is the fourth term of a geometric progression starting with 15 := sorry

end find_pairs_arithmetic_geometric_progression_l221_221644


namespace solve_system_of_equations_l221_221286

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (x / y + y / x) * (x + y) = 15 ∧ 
  (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 ∧
  ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l221_221286


namespace distance_between_incenter_and_circumcenter_of_right_triangle_l221_221601

theorem distance_between_incenter_and_circumcenter_of_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) (right_triangle : a^2 + b^2 = c^2) :
    ∃ (IO : ℝ), IO = Real.sqrt 5 :=
by
  rw [h1, h2, h3] at right_triangle
  have h_sum : 6^2 + 8^2 = 10^2 := by sorry
  exact ⟨Real.sqrt 5, by sorry⟩

end distance_between_incenter_and_circumcenter_of_right_triangle_l221_221601


namespace mitya_age_l221_221829

-- Definitions of the ages
variables (M S : ℕ)

-- Conditions based on the problem statements
axiom condition1 : M = S + 11
axiom condition2 : S = 2 * (S - (M - S))

-- The theorem stating that Mitya is 33 years old
theorem mitya_age : M = 33 :=
by
  -- Outline the proof
  sorry

end mitya_age_l221_221829


namespace circle_distance_to_line_l221_221374

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221374


namespace octagon_area_ratio_l221_221134

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221134


namespace rectangle_distances_eq_l221_221714

-- Define points and the distance squared function
structure Point where
  x : ℝ
  y : ℝ

def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the points A, B, C, D, and P
variables (a b x y : ℝ)

def A : Point := ⟨-a, -b⟩
def B : Point := ⟨ a, -b⟩
def C : Point := ⟨ a,  b⟩
def D : Point := ⟨-a,  b⟩
def P : Point := ⟨ x,  y⟩

theorem rectangle_distances_eq :
  dist_squared P A + dist_squared P C = dist_squared P B + dist_squared P D :=
by
  sorry

end rectangle_distances_eq_l221_221714


namespace recurring_fraction_eq_series_recurring_fraction_eq_fraction_l221_221665

noncomputable def recurring_fraction : ℝ := 0.5252525252

def infinite_geometric_series (a r : ℝ) : ℝ := a / (1 - r)

theorem recurring_fraction_eq_series :
  infinite_geometric_series (52 / 100) (1 / 100) = 52 / 99 := by
  sorry

theorem recurring_fraction_eq_fraction :
  recurring_fraction = 52 / 99 := by
  have recurring_fraction_series : recurring_fraction = infinite_geometric_series (52 / 100) (1 / 100) := by
    sorry
  rw [recurring_fraction_series, recurring_fraction_eq_series]
  sorry

end recurring_fraction_eq_series_recurring_fraction_eq_fraction_l221_221665


namespace linear_function_quadrants_l221_221761

theorem linear_function_quadrants (k b : ℝ) :
  (∀ x, (0 < x → 0 < k * x + b) ∧ (x < 0 → 0 < k * x + b) ∧ (x < 0 → k * x + b < 0)) →
  k > 0 ∧ b > 0 :=
by
  sorry

end linear_function_quadrants_l221_221761


namespace time_first_tap_to_fill_cistern_l221_221584

-- Defining the conditions
axiom second_tap_empty_time : ℝ
axiom combined_tap_fill_time : ℝ
axiom second_tap_rate : ℝ
axiom combined_tap_rate : ℝ

-- Specifying the given conditions
def problem_conditions :=
  second_tap_empty_time = 8 ∧
  combined_tap_fill_time = 8 ∧
  second_tap_rate = 1 / 8 ∧
  combined_tap_rate = 1 / 8

-- Defining the problem statement
theorem time_first_tap_to_fill_cistern :
  problem_conditions →
  (∃ T : ℝ, (1 / T - 1 / 8 = 1 / 8) ∧ T = 4) :=
by
  intro h
  sorry

end time_first_tap_to_fill_cistern_l221_221584


namespace prove_value_l221_221690

variable (m n : ℤ)

-- Conditions from the problem
def condition1 : Prop := m^2 + 2 * m * n = 384
def condition2 : Prop := 3 * m * n + 2 * n^2 = 560

-- Proposition to be proved
theorem prove_value (h1 : condition1 m n) (h2 : condition2 m n) : 2 * m^2 + 13 * m * n + 6 * n^2 - 444 = 2004 := by
  sorry

end prove_value_l221_221690


namespace ratio_of_octagon_areas_l221_221078

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221078


namespace find_num_biology_books_l221_221894

-- Given conditions
def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2548

-- Function to calculate combinations
def combination (n k : ℕ) := n.choose k

-- Statement to be proved
theorem find_num_biology_books (B : ℕ) (h1 : combination num_chemistry_books 2 = 28) 
  (h2 : combination B 2 * 28 = total_ways_to_pick) : B = 14 :=
by 
  -- Proof goes here
  sorry

end find_num_biology_books_l221_221894


namespace Hilltown_Volleyball_Club_Members_l221_221878

-- Definitions corresponding to the conditions
def knee_pad_cost : ℕ := 6
def uniform_cost : ℕ := 14
def total_expenditure : ℕ := 4000

-- Definition of total cost per member
def cost_per_member : ℕ := 2 * (knee_pad_cost + uniform_cost)

-- Proof statement
theorem Hilltown_Volleyball_Club_Members :
  total_expenditure % cost_per_member = 0 ∧ total_expenditure / cost_per_member = 100 := by
    sorry

end Hilltown_Volleyball_Club_Members_l221_221878


namespace wrapping_paper_fraction_used_l221_221843

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l221_221843


namespace pizza_slices_leftover_l221_221191

theorem pizza_slices_leftover :
  ∀ (H1 H2 C V : ℕ),
  H1 = 12 →
  H2 = 12 →
  C = 12 →
  V = 10 →
  let total_slices := H1 + H2 + C + V in
  let slices_eaten := 7 + 3 + 4 + 3 + 3 in
  total_slices - slices_eaten = 14 :=
by
  intros H1 H2 C V H1_def H2_def C_def V_def total_slices slices_eaten,
  rw [H1_def, H2_def, C_def, V_def] at total_slices,
  norm_num at total_slices,
  norm_num at slices_eaten,
  exact eq.refl 14


end pizza_slices_leftover_l221_221191


namespace exists_member_in_4_committees_l221_221511

-- Definitions of the conditions
def committee_structure (committees : Fin 11 → Finset ℕ) :=
  (∀ i, (committees i).card = 5) ∧
  (∀ i j, i ≠ j → ∃ m, m ∈ committees i ∧ m ∈ committees j)

-- Theorem statement
theorem exists_member_in_4_committees (committees : Fin 11 → Finset ℕ) (h : committee_structure committees) :
  ∃ m, (∑ i, if m ∈ committees i then 1 else 0) ≥ 4 := sorry

end exists_member_in_4_committees_l221_221511


namespace number_of_jump_sequences_l221_221036

def jump_sequences (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (a 3 = 3) ∧
  (∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 2))

theorem number_of_jump_sequences :
  ∃ a : ℕ → ℕ, jump_sequences a ∧ a 11 = 60 :=
by
  sorry

end number_of_jump_sequences_l221_221036


namespace range_PQ_PR_l221_221787

-- Definitions of the conditions
def C1 (x y : ℝ) := y^2 = 4 * x
def C2 (x y : ℝ) := (x - 4)^2 + y^2 = 8

-- The main theorem statement
theorem range_PQ_PR
(P : ℝ × ℝ)
(Q R : ℝ × ℝ)
(hP : C1 P.1 P.2)
(hQR : P ≠ Q ∧ P ≠ R ∧ Q ≠ R)
(h_intersect_QR : ∀ t : ℝ, ∃ x1 x2 : ℝ, 
  C1 t^2 (2 * t) ∧ (x1 - 4)^2 + (Q.2 + 2 * t - t^2)^2 = 8 
  ∧ (x2 - 4)^2 + (R.2 + 2 * t - t^2)^2 = 8 
  ∧ x1 ≠ x2)
: |(P.1 - Q.1) * (P.1 - R.1)| ∈ Set.union (Set.Ico 4 8) (Set.Ioo 8 200) :=
sorry -- proof will be provided here

end range_PQ_PR_l221_221787


namespace tangent_line_value_l221_221264

theorem tangent_line_value {f : ℝ → ℝ} (h_tangent : ∃ m b, ∀ x, f x = m * x + b → f 1 = (1 / 2 : ℝ) * 1 + 2 ∧ f' 1 = 1 / 2) :
  f 1 + f' 1 = 3 :=
by sorry

end tangent_line_value_l221_221264


namespace sequence_b_sequence_c_common_terms_product_l221_221266
  
noncomputable def S (n : ℕ) : ℕ := ∑ i in Finset.range (n+1), 3^i  -- Using hypothetical sum function

def b (n : ℕ) : ℕ := 3^n
def c (n : ℕ) : ℕ := 4 * n + 1
def a (n : ℕ) : ℕ := 9^n

theorem sequence_b (n : ℕ) :
  2 * S n = 3 * (b n - 1) :=
sorry

theorem sequence_c (c1 : ℕ) (c2 : ℕ) (c3 : ℕ) (h1 : c1 = 5) (hsum : c1 + c2 + c3 = 27) :
  c 1 = c1 ∧ c 2 = c2 ∧ c 3 = c3 :=
sorry

theorem common_terms_product :
  ∏ i in Finset.range 20, a (i + 1) = 9^210 :=
sorry

end sequence_b_sequence_c_common_terms_product_l221_221266


namespace find_value_l221_221749

theorem find_value (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a^2006 + (a + b)^2007 = 2 := 
by
  sorry

end find_value_l221_221749


namespace area_ratio_is_correct_l221_221137

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221137


namespace part1_real_number_part2_pure_imaginary_l221_221239

open Complex

/-- Part 1: Prove that if z is a real number, m is either 2 or -2. -/
theorem part1_real_number (m : ℝ) (z : ℂ) (h : z = ((m^2 - m - 2) : ℂ) + ((5 * m^2 - 20) : ℝ) * I) :
  (Im z = 0) → (m = 2 ∨ m = -2) :=
sorry

/-- Part 2: Prove that if z is a purely imaginary number, m is -1. -/
theorem part2_pure_imaginary (m : ℝ) (z : ℂ) (h : z = ((m^2 - m - 2) : ℂ) + ((5 * m^2 - 20) : ℝ) * I) :
  (Re z = 0 ∧ Im z ≠ 0) → (m = -1) :=
sorry

end part1_real_number_part2_pure_imaginary_l221_221239


namespace partition_three_sum_l221_221395

theorem partition_three_sum (n : ℕ) (h : 3 ≤ n) : 
  (∑ x in Finset.range (n - 2) + 1, (n - x - 1)) = (n - 1) * (n - 2) / 2 := 
  sorry

end partition_three_sum_l221_221395


namespace distance_to_line_is_constant_l221_221351

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221351


namespace ctg_double_alpha_l221_221206

-- Assume \(\sin (\alpha - 90^{\circ}) = -\frac{2}{3}\) and \(270^{\circ} < \alpha < 360^{\circ}\)
variables (α : ℝ)
axiom sin_alpha_minus_90 : Real.sin (α - Real.pi / 2) = -2 / 3
axiom alpha_range : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

-- Prove that ctg(2α) = √5 / 20
def ctg (x : ℝ) : ℝ := Real.cos x / Real.sin x

theorem ctg_double_alpha :
  ctg (2 * α) = (Real.sqrt 5) / 20 :=
sorry

end ctg_double_alpha_l221_221206


namespace area_ratio_is_correct_l221_221143

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221143


namespace difference_is_2395_l221_221491

def S : ℕ := 476
def L : ℕ := 6 * S + 15
def difference : ℕ := L - S

theorem difference_is_2395 : difference = 2395 :=
by
  sorry

end difference_is_2395_l221_221491


namespace sum_of_interior_angles_quadrilateral_l221_221911

theorem sum_of_interior_angles_quadrilateral : ∑ (angles : Fin 4 → ℝ), angles = 360 :=
by sorry

end sum_of_interior_angles_quadrilateral_l221_221911


namespace machine_A_sprockets_per_hour_l221_221925

theorem machine_A_sprockets_per_hour :
  ∃ (A : ℝ), 
    (∃ (G : ℝ), 
      (G = 1.10 * A) ∧ 
      (∃ (T : ℝ), 
        (660 = A * (T + 10)) ∧ 
        (660 = G * T) 
      )
    ) ∧ 
    (A = 6) :=
by
  -- Conditions and variables will be introduced here...
  -- Proof can be implemented here
  sorry

end machine_A_sprockets_per_hour_l221_221925


namespace find_natural_x_l221_221816

theorem find_natural_x 
  (x : ℕ) 
  (h1 : 2 * x > 70) 
  (h2 : x < 100) 
  (h3 : 3 * x > 25) 
  (h4 : x ≥ 10) 
  (h5 : x > 5) 
  (h_cond : (h1 ∨ ¬h1) ∧ (h2 ∨ ¬h2) ∧ (h3 ∨ ¬h3) ∧ (h4 ∨ ¬h4) ∧ (h5 ∨ ¬h5) ∧ (exactly 3 true) ∧ (exactly 2 false)): 
  x = 9 :=
by
  sorry

end find_natural_x_l221_221816


namespace length_of_CD_l221_221398

theorem length_of_CD (AB : ℝ) (angle_ACB : ℝ) (angle_BCD : ℝ) (angle_CDE : ℝ) (right_angle_ABC : ∠ABC = 90)
  (right_angle_BCD : ∠BCD = 90) (right_angle_CDE : ∠CDE = 90) (ABC_is_45_45_90 : ∠ACB = 45) 
  (BCD_is_45_45_90 : ∠BCD = 45) (CDE_is_45_45_90 : ∠CDE = 45) (AB_length : AB = 15) :
  ∃ CD : ℝ, CD = (15 * Real.sqrt 2) / 2 :=
by
  have BC := AB / Real.sqrt 2
  have CD := BC
  existsi (15 * Real.sqrt 2) / 2
  sorry

end length_of_CD_l221_221398


namespace exists_polynomial_cos_identity_l221_221226

theorem exists_polynomial_cos_identity (n : ℕ) (h : n > 0) : 
  ∃ p : Polynomial ℝ, ∀ x : ℝ, p (2 * Real.cos x) = 2 * Real.cos (n * x) :=
sorry

end exists_polynomial_cos_identity_l221_221226


namespace compartments_count_l221_221468

-- Definition of initial pennies per compartment
def initial_pennies_per_compartment : ℕ := 2

-- Definition of additional pennies added to each compartment
def additional_pennies_per_compartment : ℕ := 6

-- Definition of total pennies is 96
def total_pennies : ℕ := 96

-- Prove the number of compartments is 12
theorem compartments_count (c : ℕ) 
  (h1 : initial_pennies_per_compartment + additional_pennies_per_compartment = 8)
  (h2 : 8 * c = total_pennies) : 
  c = 12 :=
by
  sorry

end compartments_count_l221_221468


namespace ratio_of_areas_of_octagons_l221_221083

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221083


namespace ratio_of_octagon_areas_l221_221077

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221077


namespace area_ratio_of_octagons_is_4_l221_221101

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221101


namespace james_water_storage_l221_221419

theorem james_water_storage : 
  let b := 3 + 2 * 20 in
  4 * b = 172 :=
by
  sorry

end james_water_storage_l221_221419


namespace solve_for_n_l221_221855

theorem solve_for_n (n : ℤ) : 3^(2 * n + 1) = 1 / 27 → n = -2 := by
  sorry

end solve_for_n_l221_221855


namespace hyperbola_equation_l221_221591

-- Definitions of given conditions
def is_hyperbola_focus (a b : ℝ) (focus : ℝ × ℝ) : Prop :=
  let c := Real.sqrt (a^2 + b^2) in
  focus = (c, 0) ∨ focus = (-c, 0)

def is_parallel_to_asymptote (a b : ℝ) (slope : ℝ) : Prop :=
  slope = b / a

-- Given conditions
def given_conditions (a b : ℝ) : Prop :=
  is_hyperbola_focus a b (5, 0) ∧ is_parallel_to_asymptote a b (1 / 2)

-- The proof statement
theorem hyperbola_equation (a b : ℝ) (h : given_conditions a b) :
  (a^2 = 20) ∧ (b^2 = 5) → (∀ x y : ℝ, x^2 / 20 - y^2 / 5 = 1) :=
by
  sorry

end hyperbola_equation_l221_221591


namespace problem1_problem2_l221_221238

-- Definitions of conditions
def r (x m : ℝ) : Prop := sin x + cos x > m
def s (x m : ℝ) : Prop := x^2 + m * x + 1 > 0

-- Problem 1: Prove for all x in (1/2, 2], s(x) always holds implies m > -2.
theorem problem1 (m : ℝ) : (∀ x : ℝ, x ∈ Ioc (1/2 : ℝ) 2 → s x m) → m > -2 :=
sorry

-- Problem 2: Prove for all x in ℝ, r(x) and s(x) have exactly one true proposition implies m ∈ (-∞, -2] ∪ [-√2, 2)
theorem problem2 (m : ℝ) : 
  (∀ x : ℝ, (r x m ∧ ¬ s x m) ∨ (¬ r x m ∧ s x m)) → m ∈ (Set.Iic (-2) ∪ Set.Ico (-sqrt 2) 2) :=
sorry

end problem1_problem2_l221_221238


namespace triangle_AHI_area_l221_221701

theorem triangle_AHI_area (A B C G H I : Point) (area_ABC : ℝ) (h1 : area_ABC = 180)
  (h2 : Midpoint G B C) (h3 : Midpoint H A G) (h4 : Midpoint I A C) : 
  area (tri A H I) = 45 :=
sorry

end triangle_AHI_area_l221_221701


namespace find_M_l221_221957

theorem find_M (M : ℤ) (h1 : 22 < M) (h2 : M < 24) : M = 23 := by
  sorry

end find_M_l221_221957


namespace arithmetic_seq_sum_l221_221782

theorem arithmetic_seq_sum (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 :=
by 
  sorry

end arithmetic_seq_sum_l221_221782


namespace log_c_cos_y_eq_l221_221300

variables {c y p : ℝ}
-- conditions
axiom h1 : c > 2
axiom h2 : sin y > 0
axiom h3 : cos y > 0
axiom h4 : log c (sin y) = p

-- main statement
theorem log_c_cos_y_eq : log c (cos y) = (1 / 2) * log c (1 - c^(2 * p)) :=
sorry

end log_c_cos_y_eq_l221_221300


namespace parallelogram_angle_B_l221_221405

theorem parallelogram_angle_B (A C B D : ℝ) (h₁ : A + C = 110) (h₂ : A = C) : B = 125 :=
by sorry

end parallelogram_angle_B_l221_221405


namespace distance_C_to_origin_l221_221442

noncomputable def point {α : Type} [LinearOrderedField α] := (α × α)

def A : point ℝ := (1, 1)
def B : point ℝ := (4, 11 / 2)

def vector_sub (u v : point ℝ) : point ℝ := (u.1 - v.1, u.2 - v.2)
def scalar_mul (c : ℝ) (u : point ℝ) : point ℝ := (c * u.1, c * u.2)
def vector_add (u v : point ℝ) : point ℝ := (u.1 + v.1, u.2 + v.2)

def C : point ℝ :=
  let OA := A
  let OB := B
  let OC := (1 / 3) • (vector_add OA (scalar_mul 2 OB))
  OC

theorem distance_C_to_origin :
  let OC := C in
  real.sqrt (OC.1^2 + OC.2^2) = 5 :=
by
  sorry

end distance_C_to_origin_l221_221442


namespace Sn_2017_is_negative_1_over_2018_l221_221225

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def Sn (n : ℕ) : ℚ :=
  (factorial n : ℚ) * (∑ i in finset.range n, (i + 1) / factorial (i + 2) - 1)

-- The problem statement we want to prove in Lean 4.
theorem Sn_2017_is_negative_1_over_2018 : Sn 2017 = -1 / 2018 :=
sorry

end Sn_2017_is_negative_1_over_2018_l221_221225


namespace ratio_of_areas_of_octagons_l221_221060

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221060


namespace area_ratio_is_correct_l221_221142

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221142


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221114

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221114


namespace area_ratio_is_correct_l221_221144

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221144


namespace circle_tangent_distance_l221_221307

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221307


namespace volume_of_cone_l221_221725

theorem volume_of_cone (l : ℝ) (A : ℝ) (r : ℝ) (h : ℝ) : 
  l = 10 → A = 60 * Real.pi → (r = 6) → (h = Real.sqrt (10^2 - 6^2)) → 
  (1 / 3 * Real.pi * r^2 * h) = 96 * Real.pi :=
by
  intros
  -- here the proof would be written
  sorry

end volume_of_cone_l221_221725


namespace tangent_line_eq_l221_221664

theorem tangent_line_eq (x y : ℝ) : y = 2 - log x → (x = 1 ∧ y = 2) → y = -x + 3 :=
by
  intros h_curve h_point
  sorry

end tangent_line_eq_l221_221664


namespace problem_solution_l221_221189

def can_rearrange_to_square (pieces : List (List (ℕ × ℕ))) : Prop := sorry

def cut_and_rearrange_square : Prop :=
  ∃ pieces : List (List (ℕ × ℕ)),
    (length pieces = 5) ∧
    can_rearrange_to_square pieces

theorem problem_solution : cut_and_rearrange_square :=
  sorry

end problem_solution_l221_221189


namespace edward_initial_money_l221_221198

theorem edward_initial_money (initial_cost_books : ℝ) (discount_percent : ℝ) (num_pens : ℕ) 
  (cost_per_pen : ℝ) (money_left : ℝ) : 
  initial_cost_books = 40 → discount_percent = 0.25 → num_pens = 3 → cost_per_pen = 2 → money_left = 6 → 
  (initial_cost_books * (1 - discount_percent) + num_pens * cost_per_pen + money_left) = 42 :=
by
  sorry

end edward_initial_money_l221_221198


namespace circle_distance_condition_l221_221348

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221348


namespace same_side_of_line_l221_221380

open Real

theorem same_side_of_line (a : ℝ) :
  let O := (0, 0)
  let A := (1, 1)
  (O.1 + O.2 < a ↔ A.1 + A.2 < a) →
  a < 0 ∨ a > 2 := by
  sorry

end same_side_of_line_l221_221380


namespace real_numbers_correspond_to_number_line_l221_221562

-- Defining each condition in Lean

def frac_do_not_cover_all_points_on_number_line : Prop :=
  ∀ x : ℝ, ¬(∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

def rat_do_not_cover_all_points_on_number_line : Prop :=
  ∀ x : ℝ, ¬(∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

def irr_do_not_cover_all_points_on_number_line : Prop :=
  ∀ x : ℝ, ¬ rational x

def real_cover_all_points_on_number_line : Prop :=
  ∀ x : ℝ, true

-- The theorem we need to prove based on the conditions
theorem real_numbers_correspond_to_number_line :
  real_cover_all_points_on_number_line := 
by 
  sorry

end real_numbers_correspond_to_number_line_l221_221562


namespace area_ratio_of_octagons_is_4_l221_221097

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221097


namespace ratio_of_areas_of_octagons_l221_221090

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221090


namespace motorboat_travel_distance_ratio_l221_221040

theorem motorboat_travel_distance_ratio (S v u : ℝ) (h1 : u = 5 * (v) / 3 ∨ u = 7 * (v) / 3)
  (h2 : v > 0) (h3 : S > 0) :
  let t1 := S / (6 * v),
      S1 := (v + u) * t1,
      t2 := S1 / (6 * v),
      d := (5 * v - u) * t2 in
  (d = 20 * S / 81) → 
  let L1 := (5 * v - u) * t1 + d,
      L2 := (5 * v - u) * t1 + d in
  (L1 = 56 * S / 81 ∨ L2 = 65 * S / 81) :=
sorry

end motorboat_travel_distance_ratio_l221_221040


namespace cost_of_6_lollipops_l221_221966

noncomputable def lollipop_price : ℝ := 2.40 / 2

def total_price_without_discounts (n : ℕ) : ℝ :=
  n * lollipop_price

def discount (price : ℝ) : ℝ :=
  if price >= 4 * lollipop_price then price * 0.10 else 0

def apply_discount (price : ℝ) : ℝ :=
  price - discount(price)

def price_with_buy_5_get_1_free (n : ℕ) : ℝ :=
  let total_paid_lollipops := (5 * (n / 6)) + (n % 6) in
  total_paid_lollipops * lollipop_price

def final_price (n : ℕ) : ℝ :=
  price_with_buy_5_get_1_free(n) - discount(price_with_buy_5_get_1_free(n))

theorem cost_of_6_lollipops :
  final_price 6 = 6.48 :=
by
  sorry

end cost_of_6_lollipops_l221_221966


namespace handrail_length_correct_l221_221606

noncomputable def handrail_length (radius : ℝ) (rise : ℝ) (angle : ℝ) : ℝ :=
  let turns := angle / 360
  let circumference := 2 * Real.pi * radius
  let total_arc_length := turns * circumference
  let diagonal := Real.sqrt (rise ^ 2 + total_arc_length ^ 2)
  (Real.ceil (diagonal * 10) / 10) / 10

theorem handrail_length_correct : handrail_length 4 20 810 = 34.7 := 
  sorry

end handrail_length_correct_l221_221606


namespace max_product_xy_l221_221570

theorem max_product_xy (x y : ℝ) (h1 : sqrt (x + y - 1) + x^4 + y^4 - 1 / 8 ≤ 0) (h2 : x + y ≥ 1) :
  xy ≤ 1 / 4 := sorry

end max_product_xy_l221_221570


namespace trig_identity_l221_221256

variable {α : ℝ}

theorem trig_identity (h : Real.sin α = 2 * Real.cos α) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end trig_identity_l221_221256


namespace two_parabolas_intersect_points_l221_221637

-- Define the parameters for a parabola with specific conditions
structure Parabola where
  focus : (ℝ × ℝ)
  a : ℤ
  b : ℤ

axiom parabola_set : Finset Parabola
axiom parabola_focus : ∀ (p : Parabola), p.focus = (0, 0)
axiom parabola_a_range : ∀ (p : Parabola), p.a ∈ set_of (λ (a : ℤ), a ≥ -3 ∧ a ≤ 3)
axiom parabola_b_range : ∀ (p : Parabola), p.b ∈ set_of (λ (b : ℤ), b ≥ -2 ∧ b ≤ 2)
axiom parabola_unique_pairs : ∀ (p1 p2 p3 : Parabola), p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → ¬∃ (pt : ℝ × ℝ), parabola_intersect_at pt p1 p2 ∧ parabola_intersect_at pt p2 p3

noncomputable def parabola_intersect_at (pt : ℝ × ℝ) (p1 p2 : Parabola) : Prop := sorry 

theorem two_parabolas_intersect_points :
  ∑ (pt : ℝ × ℝ), ∃ p1 p2 : Parabola, p1 ≠ p2 ∧ parabola_intersect_at pt p1 p2 = 496 := sorry

end two_parabolas_intersect_points_l221_221637


namespace sum_of_central_squares_is_34_l221_221959

-- Defining the parameters and conditions
def is_adjacent (i j : ℕ) : Prop := 
  (i = j + 1 ∨ i = j - 1 ∨ i = j + 4 ∨ i = j - 4)

def valid_matrix (M : Fin 4 → Fin 4 → ℕ) : Prop := 
  ∀ (i j : Fin 4), 
  i < 3 ∧ j < 3 → is_adjacent (M i j) (M (i + 1) j) ∧ is_adjacent (M i j) (M i (j + 1))

def corners_sum_to_34 (M : Fin 4 → Fin 4 → ℕ) : Prop :=
  M 0 0 + M 0 3 + M 3 0 + M 3 3 = 34

-- Stating the proof problem
theorem sum_of_central_squares_is_34 :
  ∃ (M : Fin 4 → Fin 4 → ℕ), valid_matrix M ∧ corners_sum_to_34 M → 
  (M 1 1 + M 1 2 + M 2 1 + M 2 2 = 34) :=
by
  sorry

end sum_of_central_squares_is_34_l221_221959


namespace function_positivity_range_l221_221727

theorem function_positivity_range (m x : ℝ): 
  (∀ x, (2 * x^2 + (4 - m) * x + 4 - m > 0) ∨ (m * x > 0)) ↔ m < 4 :=
sorry

end function_positivity_range_l221_221727


namespace area_of_S_l221_221058

noncomputable def hexagon_to_set (z : ℂ) : Set ℂ :=
{t | ∃ (x y : ℝ), -x/√3 ≤ y ∧ y ≤ x/√3 ∧ z = x + y * complex.I}

noncomputable def R_subspace (S : Set ℂ) : Set ℂ :=
{t | ∃ (z : ℂ), z ∉ S ∧ t = 1 / z}

theorem area_of_S (h : {z : ℂ | abs.re z + abs.im z = 2}) (S : Set ℂ) :
  hexagon_to_set 2 = h →
  R_subspace h = S →
  area S = π / 2 :=
by
  sorry

end area_of_S_l221_221058


namespace correct_conclusions_about_f_l221_221728

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem correct_conclusions_about_f :
  (∀ x ∈ Ioo (-π/2) 0, f x < f (x + ε) ∧ ε > 0 ∧ x + ε ∈ Ioo (-π/2) 0) ∧
  (∀ x ∈ Ioo 0 (π/2), f x > f (x + ε) ∧ ε > 0 ∧ x + ε ∈ Ioo 0 (π/2)) ∧
  (¬(∃ x, f x = 1) ∧ (∃ x, (∀ y, f y ≥ f x))) ∧
  (∀ x ∈ Ioo 0 π, f x ≠ 0 ∧ ∀ y ∈ Ioo 0 π, ¬(f y < f x ∧ ∀ z ∈ Ioo 0 π, f z ≥ f y)) :=
begin
  sorry
end

end correct_conclusions_about_f_l221_221728


namespace arithmetic_sequence_a1_a5_product_l221_221887

theorem arithmetic_sequence_a1_a5_product 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a3 : a 3 = 3) 
  (h_cond : (1 / a 1) + (1 / a 5) = 6 / 5) : 
  a 1 * a 5 = 5 := 
by
  sorry

end arithmetic_sequence_a1_a5_product_l221_221887


namespace log_expression_value_l221_221912

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_value :
  log10 8 + 3 * log10 4 - 2 * log10 2 + 4 * log10 25 + log10 16 = 11 := by
  sorry

end log_expression_value_l221_221912


namespace calculate_minutes_worked_today_l221_221624

-- Define the conditions
def production_rate := 6 -- shirts per minute
def total_shirts_today := 72 

-- The statement to prove
theorem calculate_minutes_worked_today :
  total_shirts_today / production_rate = 12 := 
by
  sorry

end calculate_minutes_worked_today_l221_221624


namespace octagon_area_ratio_l221_221128

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221128


namespace polynomial_pairs_l221_221988

noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry -- Placeholder for degree function

theorem polynomial_pairs (f g : Polynomial ℝ)
  (H1 : ∀ x, x^2 * g.eval x = f.eval (g.eval x)) :
  (degree f = 3 ∧ degree g = 1) ∨ (degree f = 2 ∧ degree g = 2) :=
sorry

end polynomial_pairs_l221_221988


namespace sin_D_in_right_triangle_l221_221775

theorem sin_D_in_right_triangle (D E F : Type) [InnerProductSpace ℝ D] [InnerProductSpace ℝ E] [InnerProductSpace ℝ F]
  (angle_E_right : ∠ E = 90 * π / 180)
  (h : 5 * sin D = 12 * cos D) : sin D = 12 / 13 := 
sorry

end sin_D_in_right_triangle_l221_221775


namespace ratio_of_octagon_areas_l221_221075

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221075


namespace optimal_clothing_distribution_l221_221831

theorem optimal_clothing_distribution :
  ∃ (jackets t_shirts jeans : ℕ) (remaining : ℕ), 
    jackets = 4 ∧ t_shirts = 12 ∧ jeans = 3 ∧ remaining = 20 ∧
    (let total_cost := 100 * (jackets.div 3) + 75 * (t_shirts.div 4) + 60 * (jeans.div 2 + jeans % 2).div 2 in
     total_cost <= 400) :=
by {
  sorry
}

end optimal_clothing_distribution_l221_221831


namespace correct_calculation_l221_221556

theorem correct_calculation (x y : ℝ) : 
  ¬(2 * x^2 + 3 * x^2 = 6 * x^2) ∧ 
  ¬(x^4 * x^2 = x^8) ∧ 
  ¬(x^6 / x^2 = x^3) ∧ 
  ((x * y^2)^2 = x^2 * y^4) :=
by
  sorry

end correct_calculation_l221_221556


namespace find_f_2023_l221_221247

def is_strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, a < b → f a < f b

theorem find_f_2023 (f : ℕ → ℕ)
  (h_inc : is_strictly_increasing f)
  (h_relation : ∀ m n : ℕ, f (n + f m) = f n + m + 1) :
  f 2023 = 2024 :=
sorry

end find_f_2023_l221_221247


namespace find_YZ_value_l221_221408

-- Define the given conditions
variables (X Y Z : ℝ)
variables (YZ XY XZ : ℝ)

-- Conditions on the angles and side lengths
def angle_Y : X = 45 := sorry
def side_XY : XY = 100 := sorry
def side_XZ : XZ = 50 * sqrt 2 := sorry

-- Proof problem: Prove the value of side YZ
theorem find_YZ_value : YZ = 50 * sqrt 6 :=
  sorry

end find_YZ_value_l221_221408


namespace tan_period_increasing_l221_221960

theorem tan_period_increasing (x : ℝ) :
  (∃ T, 0 < T ∧ ∀ y, y ∈ (0, T) → y = x → tan (x - T) = tan x) ∧
  (∀ x, x ∈ set.Ioo (0 : ℝ) (π / 2) → 0 < tan (x - π / 4)) := 
sorry

end tan_period_increasing_l221_221960


namespace train_usual_time_l221_221920

theorem train_usual_time (T : ℝ) (h1 : T > 0) : 
  (4 / 5 : ℝ) * (T + 1/2) = T :=
by 
  sorry

end train_usual_time_l221_221920


namespace calories_per_cookie_l221_221652

theorem calories_per_cookie (C : ℝ) (h1 : ∀ cracker, cracker = 15)
    (h2 : ∀ cookie, cookie = C)
    (h3 : 7 * C + 10 * 15 = 500) :
    C = 50 :=
  by
    sorry

end calories_per_cookie_l221_221652


namespace correct_number_of_true_propositions_l221_221171

def regular_pyramid (p : Prop) : Prop := 
  -- Define the properties of a regular pyramid
  p

def right_prism (p : Prop) : Prop := 
  -- Define the properties of a right prism
  p

def cylinder_generatrix (p : Prop) : Prop := 
  -- Define the properties of the generatrix of a cylinder
  p

def cone_axial_section (p : Prop) : Prop := 
  -- Define the properties of the axial section of a cone
  p

def num_true_propositions (pyramid_prop prism_prop cylinder_prop cone_prop : Prop) : ℕ :=
  [pyramid_prop, prism_prop, cylinder_prop, cone_prop].count (λ p, p)

theorem correct_number_of_true_propositions :
  let prop1 := (regular_pyramid true) in
  let prop2 := (right_prism false) in
  let prop3 := (cylinder_generatrix true) in
  let prop4 := (cone_axial_section true) in
  num_true_propositions prop1 prop2 prop3 prop4 = 3 :=
by
  sorry

end correct_number_of_true_propositions_l221_221171


namespace distance_from_M0_to_plane_l221_221214

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def distance_point_to_plane (p : Point3D) (A B C D : ℝ) : ℝ :=
  abs (A * p.x + B * p.y + C * p.z + D) / real.sqrt (A ^ 2 + B ^ 2 + C ^ 2)

noncomputable def plane_equation_through_points (M1 M2 M3 : Point3D) : (ℝ × ℝ × ℝ × ℝ) :=
  let a1 := M2.x - M1.x
  let b1 := M2.y - M1.y
  let c1 := M2.z - M1.z
  let a2 := M3.x - M1.x
  let b2 := M3.y - M1.y
  let c2 := M3.z - M1.z
  let A := b1 * c2 - b2 * c1
  let B := c1 * a2 - c2 * a1
  let C := a1 * b2 - a2 * b1
  let D := -(A * M1.x + B * M1.y + C * M1.z)
  (A, B, C, D)

theorem distance_from_M0_to_plane :
  let M0 := Point3D.mk (-5) 3 7
  let M1 := Point3D.mk 2 (-1) 2
  let M2 := Point3D.mk 1 2 (-1)
  let M3 := Point3D.mk 3 2 1
  let (A, B, C, D) := plane_equation_through_points M1 M2 M3
  distance_point_to_plane M0 A B C D = 2 * real.sqrt 22 := by
  sorry

end distance_from_M0_to_plane_l221_221214


namespace shorter_leg_length_l221_221158

theorem shorter_leg_length (a b c : ℝ) (h1 : b = 10) (h2 : a^2 + b^2 = c^2) (h3 : c = 2 * a) : 
  a = 10 * Real.sqrt 3 / 3 :=
by
  sorry

end shorter_leg_length_l221_221158


namespace inequality_for_all_real_l221_221473

theorem inequality_for_all_real (a b c : ℝ) : 
  a^6 + b^6 + c^6 - 3 * a^2 * b^2 * c^2 ≥ 1/2 * (a - b)^2 * (b - c)^2 * (c - a)^2 :=
by 
  sorry

end inequality_for_all_real_l221_221473


namespace AK_perpendicular_BC_l221_221982

variables {A B C D E K : Type*} [T : triangle A B C] [T_circumcircle : circumcircle  A B C] [omega : circle A B C] [gamma : circle A (x : ℝ)]
noncomputable def acute_triangle (T : Type*) := ∃ A B C : Type*, is_acute_angle (A B C)
noncomputable def circumcircle (A B C : Type*) [T : triangle A B C] := ∃ ω: circle A B C , is_circumcircle A B C ω
noncomputable def intersection_points (gamma : circle) (omega : circle) := 
∃D E : Type*, is_intersection_point gamma omega A B D ∧ not_contains C D ∧ is_intersection_point gamma omega A C E ∧ not_contains B E

theorem AK_perpendicular_BC {A B C D E K : Type*} [acute_triangle A B C] [circumcircle A B C ω] [center_circle γ A] (T : Type*) :
  is_intersection_point BE CD K ∧ lies_on_circle K γ → 
  is_intersection_point A (perpendicular(B C)) K
:=
sorry

end AK_perpendicular_BC_l221_221982


namespace sum_of_a_b_l221_221293

-- Define the conditions in Lean
def a : ℝ := 1
def b : ℝ := 1

-- Define the proof statement
theorem sum_of_a_b : a + b = 2 := by
  sorry

end sum_of_a_b_l221_221293


namespace abs_diff_gt_two_l221_221813

def f (x : ℝ) : ℝ :=
  ∑ k in (Finset.range 1010), 1 / (x - (2 * k))

def g (x : ℝ) : ℝ :=
  ∑ k in (Finset.range 1009), 1 / (x - (2 * k + 1))

theorem abs_diff_gt_two
  (x : ℝ)
  (hx₀ : 0 < x)
  (hx₁ : x < 2018)
  (hx₂ : ¬(∃ k : ℤ, x = k)) : 
  |f(x) - g(x)| > 2 :=
by {
  sorry
}

end abs_diff_gt_two_l221_221813


namespace projection_of_a_on_b_l221_221763

open Real -- Use real numbers for vector operations

variables (a b : ℝ) -- Define a and b to be real numbers

-- Define the conditions as assumptions in Lean 4
def vector_magnitude_a (a : ℝ) : Prop := abs a = 1
def vector_magnitude_b (b : ℝ) : Prop := abs b = 1
def vector_dot_product (a b : ℝ) : Prop := (a + b) * b = 3 / 2

-- Define the goal to prove, using the assumptions
theorem projection_of_a_on_b (ha : vector_magnitude_a a) (hb : vector_magnitude_b b) (h_ab : vector_dot_product a b) : (abs a) * (a / b) = 1 / 2 :=
by
  sorry

end projection_of_a_on_b_l221_221763


namespace henry_present_age_l221_221888

theorem henry_present_age (H J : ℕ) (h1 : H + J = 41) (h2 : H - 7 = 2 * (J - 7)) : H = 25 :=
sorry

end henry_present_age_l221_221888


namespace ratio_area_octagons_correct_l221_221111

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221111


namespace airplane_rows_l221_221961

theorem airplane_rows (R : ℕ) 
  (h1 : ∀ n, n = 5) 
  (h2 : ∀ s, s = 7) 
  (h3 : ∀ f, f = 2) 
  (h4 : ∀ p, p = 1400):
  (2 * 5 * 7 * R = 1400) → R = 20 :=
by
  -- Assuming the given equation 2 * 5 * 7 * R = 1400
  sorry

end airplane_rows_l221_221961


namespace mark_profit_l221_221449

def initialPrice : ℝ := 100
def finalPrice : ℝ := 3 * initialPrice
def salesTax : ℝ := 0.05 * initialPrice
def totalInitialCost : ℝ := initialPrice + salesTax
def transactionFee : ℝ := 0.03 * finalPrice
def profitBeforeTax : ℝ := finalPrice - totalInitialCost
def capitalGainsTax : ℝ := 0.15 * profitBeforeTax
def totalProfit : ℝ := profitBeforeTax - transactionFee - capitalGainsTax

theorem mark_profit : totalProfit = 147.75 := sorry

end mark_profit_l221_221449


namespace base7_to_base10_of_43210_l221_221539

theorem base7_to_base10_of_43210 : 
  base7_to_base10 (list.num_from_digits [4, 3, 2, 1, 0]) 7 = 10738 :=
by
  def base7_to_base10 (digits : list ℕ) (base : ℕ) : ℕ :=
    digits.reverse.join_with base
  
  show base7_to_base10 [4, 3, 2, 1, 0] 7 = 10738
  sorry

end base7_to_base10_of_43210_l221_221539


namespace find_m_l221_221710

variables {R : Type} [RealOrderedRing R]

theorem find_m
  (O A B C : Point R)
  (θ : R)
  (h1 : is_circumcenter O A B C)
  (h2 : angle A O B = θ) : 
  ∃ m : R, ∀ (AB AC AO : R), 
  (cos (angle O B C) / sin (angle O B C)) * AB + (cos (angle O C B) / sin (angle O C B)) * AC = 2 * m * AO ↔ m = sin θ :=
sorry

end find_m_l221_221710


namespace distance_from_center_to_line_of_tangent_circle_l221_221316

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221316


namespace fencing_cost_l221_221505

def ratio (a b : ℕ) : Prop := ∃ k : ℕ, a = k * 3 ∧ b = k * 2
def area (a b : ℕ) : Prop := a * b = 2400
def cost_per_meter : ℝ := 0.5
def circumference_tree (r : ℝ) : ℝ := 2 * Real.pi * r
def total_cost (p t c : ℝ) : ℝ := p + t + c

theorem fencing_cost (length width : ℕ) (tree_cnt : ℕ) (pond_circumference : ℝ)
  (h_ratio : ratio length width) (h_area : area length width) 
  (h_trees : tree_cnt = 10) (h_pond : pond_circumference = 30) :
  total_cost (200 * cost_per_meter) (10 * circumference_tree 1 * cost_per_meter) 
  (30 * cost_per_meter) = 146.4 := sorry

end fencing_cost_l221_221505


namespace derivative_at_x0_l221_221818

variable {ℝ : Type*}

-- Given conditions
variables (f : ℝ → ℝ) (x0 : ℝ)
variable (h_diff : DifferentiableAt ℝ f x0)
variable (h_lim : Filter.Tendsto (fun Δx => (f (x0 - 3 * Δx) - f x0) / (2 * Δx)) (𝓝 0) (𝓝 1))

-- Lean statement to prove
theorem derivative_at_x0 :
  deriv f x0 = -2/3 := 
sorry

end derivative_at_x0_l221_221818


namespace polynomial_bound_l221_221529

noncomputable def polynomial : Type :=
  { f : ℝ[X] // ∃ n a, f = (X ^ n) + (∑ i in range n, (a i) * X ^ (i - 1)) }

theorem polynomial_bound (f : polynomial) (n : ℕ) :
  (∃ i ∈ range (n + 1), n ≠ 0 → (abs (eval i f.val)) ≥ n! / nat.choose n i) := sorry

end polynomial_bound_l221_221529


namespace problem1_problem2_l221_221734

-- Definition of the parabola C and circle O
def parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {xy | let (x, y) := xy in x^2 = 2 * p * y}

def circle : Set (ℝ × ℝ) :=
  {point | let (x, y) := point in x^2 + y^2 = 1}

-- Problem (1)
theorem problem1 (p : ℝ) (hp : p > 0) (F A : ℝ × ℝ)
  (hF : F ∈ circle)
  (hFocus : is_focus F (parabola p))
  (hA : A ∈ (parabola p ∩ circle)) :
  abs (dist A F) = sqrt 5 - 1 :=
sorry

-- Problem (2)
theorem problem2 (p : ℝ) (hp : p > 0) (M N : ℝ × ℝ) (line : (ℝ × ℝ) → Prop)
  (hlt : is_tangent line (parabola p) M)
  (hlc : is_tangent line cylinder N) :
  (∀ l, is_tangent l (parabola p) M → is_tangent l circle N → 
    (abs (dist M N) >= 2 * sqrt 2 ∧ p = sqrt 3)) :=
  sorry

end problem1_problem2_l221_221734


namespace wrapping_paper_area_correct_l221_221942

noncomputable def wrapping_paper_area (l w h : ℝ) (hlw : l ≥ w) : ℝ :=
  (l + 2*h)^2

theorem wrapping_paper_area_correct (l w h : ℝ) (hlw : l ≥ w) :
  wrapping_paper_area l w h hlw = (l + 2*h)^2 :=
by
  sorry

end wrapping_paper_area_correct_l221_221942


namespace max_fraction_seq_l221_221737

theorem max_fraction_seq (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 3)
  (h_recurr : ∀ n : ℕ, 2 ≤ n → 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)) :
  ∃ n : ℕ, n ∈ ℕ ∧ n ≠ 0 ∧ (∀ m : ℕ, m ∈ ℕ ∧ m ≠ 0 → (a m : ℚ) / m ≤ 3 / 2) ∧ (a n : ℚ) / n = 3 / 2 :=
by
  sorry

end max_fraction_seq_l221_221737


namespace vector_intersecting_line_parameter_l221_221642

theorem vector_intersecting_line_parameter :
  ∃ (a b s : ℝ), a = 3 * s + 5 ∧ b = 2 * s + 4 ∧
                   (∃ r, (a, b) = (3 * r, 2 * r)) ∧
                   (a, b) = (6, 14 / 3) :=
by
  sorry

end vector_intersecting_line_parameter_l221_221642


namespace greatest_decimal_is_7391_l221_221913

noncomputable def decimal_conversion (n d : ℕ) : ℝ :=
  n / d

noncomputable def forty_two_percent_of (r : ℝ) : ℝ :=
  0.42 * r

theorem greatest_decimal_is_7391 :
  let a := forty_two_percent_of (decimal_conversion 7 11)
  let b := decimal_conversion 17 23
  let c := 0.7391
  let d := decimal_conversion 29 47
  a < b ∧ a < c ∧ a < d ∧ b = c ∧ d < b :=
by
  have dec1 := forty_two_percent_of (decimal_conversion 7 11)
  have dec2 := decimal_conversion 17 23
  have dec3 := 0.7391
  have dec4 := decimal_conversion 29 47
  sorry

end greatest_decimal_is_7391_l221_221913


namespace circle_center_line_distance_l221_221340

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221340


namespace root_interval_l221_221876

def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem root_interval :
  ∃ a b : ℝ, a = 1/4 ∧ b = 1/2 ∧ (∀ x, a < x ∧ x < b → f(x) = 0) :=
by
  sorry

end root_interval_l221_221876


namespace find_amount_of_alcohol_l221_221945

theorem find_amount_of_alcohol (A W : ℝ) (h₁ : A / W = 4 / 3) (h₂ : A / (W + 7) = 4 / 5) : A = 14 := 
sorry

end find_amount_of_alcohol_l221_221945


namespace phi_value_l221_221276

theorem phi_value
  (ω : ℝ) (φ : ℝ)
  (hω : ω ≠ 0)
  (hφ : |φ| < (Real.pi / 2))
  (center_pts : (2 * Real.pi / 3, 0) = (7 * Real.pi / 6, 0))
  (monotonic_interval : Function.StrictMonoOn (fun x => Real.tan (ω * x + φ)) 
    ((2 * Real.pi / 3) .. (4 * Real.pi / 3))) :
  φ = - (Real.pi / 6) := 
sorry

end phi_value_l221_221276


namespace range_of_a_l221_221273

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ (1 ≤ a ∧ a < 5) := by
  sorry

end range_of_a_l221_221273


namespace people_with_fewer_than_seven_cards_l221_221755

theorem people_with_fewer_than_seven_cards (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ)
  (h1 : total_cards = 52) (h2 : num_people = 8) (h3 : total_cards = num_people * cards_per_person + extra_cards) (h4 : extra_cards < num_people) :
  ∃ fewer_than_seven : ℕ, num_people - extra_cards = fewer_than_seven :=
by
  have remainder := (52 % 8)
  have cards_per_person := (52 / 8)
  have number_fewer_than_seven := num_people - remainder
  existsi number_fewer_than_seven
  sorry

end people_with_fewer_than_seven_cards_l221_221755


namespace graduation_photo_arrangements_l221_221585

theorem graduation_photo_arrangements :
  let classroom_size := 6
  let total_arrangements := factorial (classroom_size + 1)
  let adjacent_ab_arrangements := 4 * factorial (classroom_size - 1)
  let non_adjacent_ab_arrangements := total_arrangements - adjacent_ab_arrangements
  non_adjacent_ab_arrangements = 528 :=
sorry

end graduation_photo_arrangements_l221_221585


namespace jan_25_on_thursday_l221_221650

/-- 
  Given that December 25 is on Monday,
  prove that January 25 in the following year falls on Thursday.
-/
theorem jan_25_on_thursday (day_of_week : Fin 7) (h : day_of_week = 0) : 
  ((day_of_week + 31) % 7 + 25) % 7 = 4 := 
sorry

end jan_25_on_thursday_l221_221650


namespace whittlesford_band_max_members_l221_221485

def max_band_members (k : ℕ) : ℕ := 45 * k

theorem whittlesford_band_max_members :
  ∃ k : ℤ, 45 * k % 37 = 28 ∧ 45 * k < 1500 ∧ k ≥ 0 ∧ max_band_members (k.to_nat) = 945 :=
by
  sorry

end whittlesford_band_max_members_l221_221485


namespace ratio_of_areas_of_octagons_l221_221066

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221066


namespace most_prolific_mathematician_is_euler_l221_221017

noncomputable def prolific_mathematician (collected_works_volume_count: ℕ) (publishing_organization: String) : String :=
  if collected_works_volume_count > 75 ∧ publishing_organization = "Swiss Society of Natural Sciences" then
    "Leonhard Euler"
  else
    "Unknown"

theorem most_prolific_mathematician_is_euler :
  prolific_mathematician 76 "Swiss Society of Natural Sciences" = "Leonhard Euler" :=
by
  sorry

end most_prolific_mathematician_is_euler_l221_221017


namespace speed_first_train_l221_221525

theorem speed_first_train
  (length_train1 length_train2 : ℕ)
  (speed_train2 : ℝ)
  (time_clear : ℝ)
  (distance_total : ℝ := length_train1 + length_train2)
  (distance_total_km : ℝ := distance_total / 1000)
  (time_hours : ℝ := time_clear / 3600)
  (relative_speed : ℝ := distance_total_km / time_hours) :

  length_train1 = 120 → 
  length_train2 = 320 → 
  speed_train2 = 30 → 
  time_clear = 21.998240140788738 →
  speed_train1 = relative_speed - speed_train2 
  → speed_train1 ≈ 41.96
:= by
  sorry

end speed_first_train_l221_221525


namespace negation_statement_l221_221301

theorem negation_statement (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) : ¬ (x = 0 ∧ y = 0) :=
sorry

end negation_statement_l221_221301


namespace grid_path_theorem_l221_221636

open Nat

variables (m n : ℕ)
variables (A B C : ℕ)

def conditions (m n : ℕ) : Prop := m ≥ 4 ∧ n ≥ 4

noncomputable def grid_path_problem (m n A B C : ℕ) : Prop :=
  conditions m n ∧
  ((m - 1) * (n - 1) = A + (B + C)) ∧
  A = B - C + m + n - 1

theorem grid_path_theorem (m n A B C : ℕ) (h : grid_path_problem m n A B C) : 
  A = B - C + m + n - 1 :=
  sorry

end grid_path_theorem_l221_221636


namespace West_oil_production_NonWest_oil_production_Russia_oil_production_oil_production_all_regions_l221_221392

-- Definitions for the conditions
def oil_per_person_West : ℝ := 55.084
def oil_per_person_NonWest : ℝ := 214.59
def oil_per_person_Russia : ℝ := 1038.33

-- Theorem statements asserting the oil production per person
theorem West_oil_production : 
  (55.084 : ℝ) = oil_per_person_West := sorry

theorem NonWest_oil_production : 
  (214.59 : ℝ) = oil_per_person_NonWest := sorry

theorem Russia_oil_production : 
  (1038.33 : ℝ) = oil_per_person_Russia := sorry

-- Conjunction of all the statements proving the original problem's solutions
theorem oil_production_all_regions : 
  (55.084 : ℝ) = oil_per_person_West ∧
  (214.59 : ℝ) = oil_per_person_NonWest ∧
  (1038.33 : ℝ) = oil_per_person_Russia := 
by 
    apply And.intro
    . exact West_oil_production
    . apply And.intro
        . exact NonWest_oil_production
        . exact Russia_oil_production

end West_oil_production_NonWest_oil_production_Russia_oil_production_oil_production_all_regions_l221_221392


namespace velocity_at_three_seconds_l221_221597

noncomputable def displacement (t : ℝ) : ℝ := t ^ (1/4 : ℝ)

theorem velocity_at_three_seconds :
  (deriv displacement 3) = 1 / (4 * (3 ^ (3 / 4 : ℝ))) :=
by
  apply congr_arg
  sorry -- Proof skipped

end velocity_at_three_seconds_l221_221597


namespace polynomial_coeff_sum_l221_221735

theorem polynomial_coeff_sum :
  let p := ((Polynomial.C 1 + Polynomial.X)^3 * (Polynomial.C 2 + Polynomial.X)^2)
  let a0 := p.coeff 0
  let a2 := p.coeff 2
  let a4 := p.coeff 4
  a4 + a2 + a0 = 36 := by 
  sorry

end polynomial_coeff_sum_l221_221735


namespace distance_from_center_to_line_of_tangent_circle_l221_221318

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221318


namespace original_number_l221_221545

theorem original_number (n : ℕ) (h : (n + 1) % 30 = 0) : n = 29 :=
by
  sorry

end original_number_l221_221545


namespace find_number_l221_221037

theorem find_number (x : ℝ) (h : 0.6667 * x + 0.75 = 1.6667) : x = 1.375 :=
sorry

end find_number_l221_221037


namespace tiger_meat_consumption_per_day_l221_221383

-- Conditions as definitions
def lion_consumption_per_day : ℕ := 25
def total_meat_available : ℕ := 90
def total_days : ℕ := 2

-- Question: How many kilograms of meat does the tiger consume per day?
-- Correct Answer: 20 kilograms

theorem tiger_meat_consumption_per_day : 
  let meat_for_lion := lion_consumption_per_day * total_days,
      remaining_meat := total_meat_available - meat_for_lion in
  remaining_meat / total_days = 20 := 
by
  -- Proof may be assumed to be carried out here
  sorry

end tiger_meat_consumption_per_day_l221_221383


namespace product_nonreal_roots_l221_221218

theorem product_nonreal_roots (x : ℂ) :
  (x - 1)^4 = 2006 → (x = 1 + complex.i * complex.sqrt (complex.root4 2006) ∨ x = 1 - complex.i * complex.sqrt (complex.root4 2006)) → 
  (1 + complex.i * complex.sqrt (complex.root4 2006)) * (1 - complex.i * complex.sqrt (complex.root4 2006)) = 1 + real.sqrt 2006 := 
sorry

end product_nonreal_roots_l221_221218


namespace apple_difference_l221_221199

def carla_apples : ℕ := 7
def tim_apples : ℕ := 1

theorem apple_difference : carla_apples - tim_apples = 6 := by
  sorry

end apple_difference_l221_221199


namespace ratio_of_areas_of_octagons_l221_221069

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221069


namespace product_first_20_a_n_l221_221268

noncomputable def b_n (n : ℕ) : ℕ := 3^n
noncomputable def c_n (n : ℕ) : ℕ := 4 * n + 1
noncomputable def a_n (n : ℕ) : ℕ := 9^n

theorem product_first_20_a_n : 
  ∏ i in Finset.range 20, a_n (i + 1) = 9 ^ 210 :=
by
  sorry

end product_first_20_a_n_l221_221268


namespace solve_for_n_l221_221857

theorem solve_for_n (n : ℤ) (h : 3^(2 * n + 1) = 1 / 27) : n = -2 := by
  sorry

end solve_for_n_l221_221857


namespace age_of_15th_student_l221_221489

theorem age_of_15th_student (avg_age_15_students avg_age_5_students avg_age_9_students : ℕ)
  (total_students total_age_15_students total_age_5_students total_age_9_students : ℕ)
  (h1 : total_students = 15)
  (h2 : avg_age_15_students = 15)
  (h3 : avg_age_5_students = 14)
  (h4 : avg_age_9_students = 16)
  (h5 : total_age_15_students = total_students * avg_age_15_students)
  (h6 : total_age_5_students = 5 * avg_age_5_students)
  (h7 : total_age_9_students = 9 * avg_age_9_students):
  total_age_15_students = total_age_5_students + total_age_9_students + 11 :=
by
  sorry

end age_of_15th_student_l221_221489


namespace choose_three_non_adjacent_l221_221852

-- We define the total weight of the cake
def total_weight : ℕ := 900

-- We represent the cake as a 3x3 grid of pieces
structure Grid :=
  (weights : Fin 3 × Fin 3 → ℕ)

-- We assume the grid has weight totaling 900 g
noncomputable def grid : Grid := 
{ weights := λ _, 100 } -- This assumes an equitable distribution for simplicity

-- Proving the main theorem that Petya can choose three non-adjacent pieces whose total weight is at least 300 g
theorem choose_three_non_adjacent (g : Grid) (H : ∑ x in (Finset.univ : Finset (Fin 3 × Fin 3)), g.weights x = total_weight) :
  ∃ (a b c : Fin 3 × Fin 3), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (a.1 ≠ b.1 ∨ a.2 ≠ b.2) ∧ 
  (b.1 ≠ c.1 ∨ b.2 ≠ c.2) ∧ 
  (a.1 ≠ c.1 ∨ a.2 ≠ c.2) ∧ 
  (g.weights a + g.weights b + g.weights c ≥ 300) :=
sorry

end choose_three_non_adjacent_l221_221852


namespace ratio_of_areas_of_octagons_l221_221087

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221087


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221122

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221122


namespace optionA_optionB_optionC_optionD_l221_221015

-- Statement for option A
theorem optionA : (∀ x : ℝ, x ≠ 3 → x^2 - 4 * x + 3 ≠ 0) ↔ (x^2 - 4 * x + 3 = 0 → x = 3) := sorry

-- Statement for option B
theorem optionB : (¬ (∀ x : ℝ, x^2 - x + 2 > 0) ↔ ∃ x0 : ℝ, x0^2 - x0 + 2 ≤ 0) := sorry

-- Statement for option C
theorem optionC (p q : Prop) : p ∧ q → p ∧ q := sorry

-- Statement for option D
theorem optionD (x : ℝ) : (x > -1 → x^2 + 4 * x + 3 > 0) ∧ ¬ (∀ x : ℝ, x^2 + 4 * x + 3 > 0 → x > -1) := sorry

end optionA_optionB_optionC_optionD_l221_221015


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221123

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221123


namespace solve_system_of_equations_l221_221287

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (x / y + y / x) * (x + y) = 15 ∧ 
  (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 ∧
  ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l221_221287


namespace books_total_l221_221795

theorem books_total (J T : ℕ) (hJ : J = 10) (hT : T = 38) : J + T = 48 :=
by {
  sorry
}

end books_total_l221_221795


namespace distance_from_center_of_circle_to_line_l221_221327

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221327


namespace soja_total_pages_l221_221477

-- Define the assumptions
constant x : ℕ -- Number of days Soja read.
constant P : ℕ -- Total number of pages in the book.
constant h1 : (2 / 3 : ℚ) * P = ((20 : ℕ) * (2 : ℕ) * x)
constant h2 : (P * (2 / 3 : ℚ)) - (P * (1 / 3 : ℚ)) = 100

-- The statement of the problem
theorem soja_total_pages : P = 300 :=
by
  have h : (1 / 3 : ℚ) * P = 100, from sorry,
  sorry

end soja_total_pages_l221_221477


namespace integral_of_rational_function_l221_221635

theorem integral_of_rational_function :
  ∫ (x : ℝ) in set.Ioc (-∞) ∞, 
    (x^3 + 6*x^2 + 11*x + 7) / ((x + 1) * (x + 2)^3) ⟹(ℝ,ℝ)∫ 
  (ln |x + 1| + 1 / (2 * (x + 2)^2)) + C :=
by 
  sorry

end integral_of_rational_function_l221_221635


namespace total_area_correct_l221_221964

noncomputable def total_area (PM : ℝ) (angle_PMC : ℝ) : ℝ :=
  if angle_PMC = π / 2 then
    let area_PMCD := PM^2
    let PQ := PM * √3
    let area_PQR := (√3 / 4) * PQ^2
    area_PMCD + area_PQR
  else
    0 -- This case should not happen as per the given angle condition.

theorem total_area_correct (PM := 12) (angle_PMC := π / 2) :
  total_area PM angle_PMC = 144 + 36 * √3 :=
by 
  unfold total_area
  rw if_pos rfl
  simp [PM, angle_PMC]
  /- The lean prover will symbolically verify the rest --/
  sorry

end total_area_correct_l221_221964


namespace ratio_of_areas_of_octagons_l221_221067

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221067


namespace box_volume_l221_221057

theorem box_volume (x : ℕ) (h_ratio : (x > 0)) (V : ℕ) (h_volume : V = 20 * x^3) : V = 160 :=
by
  sorry

end box_volume_l221_221057


namespace least_square_tiles_required_l221_221569

-- Define the dimensions of the room in meters
def length_meters : ℝ := 15.17
def breadth_meters : ℝ := 9.02

-- Convert the dimensions to centimeters
def length_cm : ℕ := (length_meters * 100).toNat
def breadth_cm : ℕ := (breadth_meters * 100).toNat

-- Define a function to compute the Euclidean algorithm (GCD)
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Define the GCD of the room dimensions in cm
def gcd_cm : ℕ := gcd length_cm breadth_cm

-- Define the area of the room in cm²
def area_room_cm2 : ℕ := length_cm * breadth_cm

-- Define the area of one tile in cm²
def tile_area_cm2 : ℕ := gcd_cm * gcd_cm

-- Define the number of tiles required
def num_tiles : ℕ := area_room_cm2 / tile_area_cm2

-- The theorem to prove the minimum number of tiles required is 814
theorem least_square_tiles_required : num_tiles = 814 :=
  sorry

end least_square_tiles_required_l221_221569


namespace original_speed_of_train_l221_221617

theorem original_speed_of_train :
  ∃ s : ℕ, (s > 0) ∧ (3 * s = 2 * (s + 30)) ∧ (s = 60) :=
by
  use 60
  split
  · -- s > 0
    exact Nat.zero_lt_succ 59
  split
  · -- 3 * s = 2 * (s + 30)
    calc 3 * 60
        = 180         : by ring
       ... = 2 * (60 + 30) : by ring
  · -- s = 60
    rfl

end original_speed_of_train_l221_221617


namespace circle_distance_condition_l221_221341

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221341


namespace range_of_f_l221_221219

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x / 2))^2 + 
  Real.pi * Real.arcsin (x / 2) - 
  (Real.arcsin (x / 2))^2 + 
  (Real.pi^2 / 6) * (x^2 + 2 * x + 1)

theorem range_of_f (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) :
  ∃ y : ℝ, (f y) = x ∧  (Real.pi^2 / 4) ≤ y ∧ y ≤ (39 * Real.pi^2 / 96) := 
sorry

end range_of_f_l221_221219


namespace value_of_a_plus_b_l221_221296

theorem value_of_a_plus_b (a b : ℕ) (h1 : Real.sqrt 44 = 2 * Real.sqrt a) (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l221_221296


namespace min_value_is_5_sqrt_2_l221_221869

def min_value_of_sqrt_sum (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + 4*x + 20) + Real.sqrt (x^2 + 2*x + 10)

theorem min_value_is_5_sqrt_2 : ∃ x : ℝ, min_value_of_sqrt_sum x = 5 * Real.sqrt 2 :=
sorry

end min_value_is_5_sqrt_2_l221_221869


namespace fraction_one_third_between_l221_221871

theorem fraction_one_third_between (a b : ℚ) (h1 : a = 1/6) (h2 : b = 1/4) : (1/3 * (b - a) + a = 7/36) :=
by
  -- Conditions
  have ha : a = 1/6 := h1
  have hb : b = 1/4 := h2
  -- Start proof
  sorry

end fraction_one_third_between_l221_221871


namespace total_charge_for_trip_l221_221794

def initial_fee := 2.05
def additional_charge_per_increment := 0.35
def distance_traveled := 3.6
def increment_distance := 2 / 5

theorem total_charge_for_trip :
  let increments := (distance_traveled * 5) / 2 in
  let additional_charge := increments * additional_charge_per_increment in
  (initial_fee + additional_charge) = 5.20 :=
by
  sorry

end total_charge_for_trip_l221_221794


namespace find_f_2023_l221_221246

def is_strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, a < b → f a < f b

theorem find_f_2023 (f : ℕ → ℕ)
  (h_inc : is_strictly_increasing f)
  (h_relation : ∀ m n : ℕ, f (n + f m) = f n + m + 1) :
  f 2023 = 2024 :=
sorry

end find_f_2023_l221_221246


namespace oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l221_221390

-- Definitions for oil consumption per person
def oilConsumptionWest : ℝ := 55.084
def oilConsumptionNonWest : ℝ := 214.59
def oilConsumptionRussia : ℝ := 1038.33

-- Lean statements
theorem oilProductionPerPerson_west : oilConsumptionWest = 55.084 := by
  sorry

theorem oilProductionPerPerson_nonwest : oilConsumptionNonWest = 214.59 := by
  sorry

theorem oilProductionPerPerson_russia : oilConsumptionRussia = 1038.33 := by
  sorry

end oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l221_221390


namespace div_sum_triple_12_l221_221977

-- Helper function to calculate the sum of divisors of n except n itself.
def sumOfDivisorsExceptSelf (n : Nat) : Nat :=
  (List.range (n + 1)).filter (λ d => d > 0 ∧ n % d == 0 ∧ d ≠ n).sum

-- Assertion to prove the required theorem.
theorem div_sum_triple_12 : sumOfDivisorsExceptSelf (sumOfDivisorsExceptSelf (sumOfDivisorsExceptSelf 12)) = 9 :=
  by sorry

end div_sum_triple_12_l221_221977


namespace only_n_4_l221_221997

theorem only_n_4 (n : ℕ) (h1 : n ≥ 3) : 
  (∃ (points : Fin n → ℝ × ℝ × ℝ), 
    (∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
      ¬Collinear (points i) (points j) (points k)) ∧
    (∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      ¬Concyclic (points i) (points j) (points k) (points l)) ∧
    (∀ (i j k l m: Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ m → 
      Circle (points i) (points j) (points k) ≅ Circle (points i) (points j) (points l))) 
  ↔ n = 4 :=
by 
  sorry

end only_n_4_l221_221997


namespace West_oil_production_NonWest_oil_production_Russia_oil_production_oil_production_all_regions_l221_221393

-- Definitions for the conditions
def oil_per_person_West : ℝ := 55.084
def oil_per_person_NonWest : ℝ := 214.59
def oil_per_person_Russia : ℝ := 1038.33

-- Theorem statements asserting the oil production per person
theorem West_oil_production : 
  (55.084 : ℝ) = oil_per_person_West := sorry

theorem NonWest_oil_production : 
  (214.59 : ℝ) = oil_per_person_NonWest := sorry

theorem Russia_oil_production : 
  (1038.33 : ℝ) = oil_per_person_Russia := sorry

-- Conjunction of all the statements proving the original problem's solutions
theorem oil_production_all_regions : 
  (55.084 : ℝ) = oil_per_person_West ∧
  (214.59 : ℝ) = oil_per_person_NonWest ∧
  (1038.33 : ℝ) = oil_per_person_Russia := 
by 
    apply And.intro
    . exact West_oil_production
    . apply And.intro
        . exact NonWest_oil_production
        . exact Russia_oil_production

end West_oil_production_NonWest_oil_production_Russia_oil_production_oil_production_all_regions_l221_221393


namespace find_a_from_hyperbola_conditions_l221_221278

theorem find_a_from_hyperbola_conditions (a b c : ℝ) (A B I1 I2 : ℝ × ℝ)
  (intersects : ∃ A B : ℝ × ℝ, (A.2 = 2 * (A.1 - c)) ∧ (B.2 = 2 * (B.1 - c)))
  (Hx1 : A ∈ { p : ℝ × ℝ | (p.1^2 / a^2 - p.2^2 / b^2 = 1) })
  (Hx2 : B ∈ { p : ℝ × ℝ | (p.1^2 / a^2 - p.2^2 / b^2 = 1) })
  (Hf1 : I1 = (incenter (A, '0, '0))) -- Placeholder for actual calculations
  (Hf2 : I2 = (incenter (B, '0, '0))) -- Placeholder for actual calculations
  (HI : dist I1 I2 = 2 * sqrt 5)
  (ecc : c = a * 2) -- eccentricity = 2
  (Hcond : a > 0 ∧ b > 0) :
  a = 2 :=
sorry

end find_a_from_hyperbola_conditions_l221_221278


namespace sqrt17_minus1_gt_3_l221_221972

theorem sqrt17_minus1_gt_3 :
  3 < (Real.sqrt 17) - 1 := 
by
  have h1 : Real.sqrt 16 < Real.sqrt 17 := 
    Real.sqrt_lt.mpr (by norm_num)
  have h2 : Real.sqrt 17 < Real.sqrt 25 := 
    Real.sqrt_lt.mpr (by norm_num)
  have h3 : 4 < Real.sqrt 17 := by linarith
  have h4 : Real.sqrt 17 < 5 := by linarith
  show 3 < (Real.sqrt 17) - 1
  linarith

end sqrt17_minus1_gt_3_l221_221972


namespace sequence_b_sequence_c_common_terms_product_l221_221267
  
noncomputable def S (n : ℕ) : ℕ := ∑ i in Finset.range (n+1), 3^i  -- Using hypothetical sum function

def b (n : ℕ) : ℕ := 3^n
def c (n : ℕ) : ℕ := 4 * n + 1
def a (n : ℕ) : ℕ := 9^n

theorem sequence_b (n : ℕ) :
  2 * S n = 3 * (b n - 1) :=
sorry

theorem sequence_c (c1 : ℕ) (c2 : ℕ) (c3 : ℕ) (h1 : c1 = 5) (hsum : c1 + c2 + c3 = 27) :
  c 1 = c1 ∧ c 2 = c2 ∧ c 3 = c3 :=
sorry

theorem common_terms_product :
  ∏ i in Finset.range 20, a (i + 1) = 9^210 :=
sorry

end sequence_b_sequence_c_common_terms_product_l221_221267


namespace complex_number_pow_two_l221_221974

theorem complex_number_pow_two (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by sorry

end complex_number_pow_two_l221_221974


namespace A_wins_3_1_probability_l221_221025

noncomputable def probability_A_wins_3_1 (p : ℚ) : ℚ :=
  let win_3_1 := binomial 4 3 * (p^3) * (1 - p)
  win_3_1

theorem A_wins_3_1_probability : probability_A_wins_3_1 (2/3) = 8/27 := by
  sorry

end A_wins_3_1_probability_l221_221025


namespace octagon_area_ratio_l221_221130

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221130


namespace find_BG_l221_221600

structure Point where
  x : ℝ
  y : ℝ

structure Hexagon where
  A B C D E F : Point
  side_length : ℝ
  is_regular : ∀ (X Y : Point), X ∈ {A, B, C, D, E, F} → Y ∈ {A, B, C, D, E, F} → (X ≠ Y → dist X Y = side_length)

noncomputable def midpoint (P Q : Point) : Point := 
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

noncomputable def hit_point (A B C : Point) : Point :=
  -- Assuming a function that calculates the hit point G by a laser from A to BC.
  sorry

noncomputable def is_midpoint (M P Q : Point) : Prop := 
  (M.x = (P.x + Q.x) / 2) ∧ (M.y = (P.y + Q.y) / 2)

noncomputable def distance (P Q : Point) : ℝ := 
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem find_BG :
  ∀ (hex : Hexagon), 
  let G := hit_point hex.A hex.B hex.C in
  let M := midpoint hex.D hex.E in
  is_midpoint M hex.D hex.E →
  hex.side_length = 2 →
  distance hex.B G = 2 / 5 :=
  sorry

end find_BG_l221_221600


namespace business_total_profit_l221_221579

noncomputable def total_profit (spending_ratio income_ratio total_income : ℕ) : ℕ :=
  let total_parts := spending_ratio + income_ratio
  let one_part_value := total_income / income_ratio
  let spending := spending_ratio * one_part_value
  total_income - spending

theorem business_total_profit :
  total_profit 5 9 108000 = 48000 :=
by
  -- We omit the proof steps, as instructed.
  sorry

end business_total_profit_l221_221579


namespace circle_center_line_distance_l221_221332

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221332


namespace Rajesh_completion_days_l221_221463

theorem Rajesh_completion_days :
  ∃ (R : ℝ), (∀ (work_rate_Rajesh : ℝ), work_rate_Rajesh = 1 / R → 
  (∀ (combined_work_rate : ℝ), combined_work_rate = (R + 3) / (3 * R) → 
  (∀ (total_amount Rahul_share : ℝ), total_amount = 150 ∧ Rahul_share = 60 →
  (∀ (share_Rajesh : ℝ), share_Rajesh = total_amount - Rahul_share → 
  (20 * R = share_Rajesh) ∧ (20 * R = 90)))) → R = 4.5)) :=
sorry

end Rajesh_completion_days_l221_221463


namespace circle_tangent_distance_l221_221313

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221313


namespace find_income_l221_221875

variable (x : ℝ)

def income : ℝ := 5 * x
def expenditure : ℝ := 4 * x
def savings : ℝ := income x - expenditure x

theorem find_income (h : savings x = 4000) : income x = 20000 :=
by
  rw [savings, income, expenditure] at h
  sorry

end find_income_l221_221875


namespace g_of_fraction_eq_g_of_twice_x_l221_221862

-- defining the domain condition for x
def domain (x : ℝ) : Prop := -1 < x ∧ x < 1

-- defining the function g
def g (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

-- statement of the theorem
theorem g_of_fraction_eq_g_of_twice_x (x : ℝ) (h : domain x) :
  g ((4 * x + x^2) / (1 + 4 * x + x^2)) = g (2 * x) :=
sorry

end g_of_fraction_eq_g_of_twice_x_l221_221862


namespace ratio_area_octagons_correct_l221_221106

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221106


namespace octagon_area_ratio_l221_221153

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221153


namespace union_of_A_and_B_l221_221253

open Set

variable (A B : Set ℤ)

def A_def : Set ℤ := {x | x ∈ (coe <$> (Set.univ : Set ℕ)) ∧ x ≤ 2}
def B_def : Set ℤ := {-1, 1}

theorem union_of_A_and_B :
  A_def ∪ B_def = ({-1, 0, 1, 2} : Set ℤ) := sorry

end union_of_A_and_B_l221_221253


namespace other_root_is_five_l221_221228

theorem other_root_is_five (m : ℝ) 
  (h : -1 is_root_m x^2 - 4 * x + m = 0) : 
  is_root x^2 - 4 * x + m = 0 5 := 
sorry

end other_root_is_five_l221_221228


namespace basketball_success_rate_l221_221035

theorem basketball_success_rate (p : ℝ) (h : 1 - p^2 = 16 / 25) : p = 3 / 5 :=
sorry

end basketball_success_rate_l221_221035


namespace determine_a_l221_221272

noncomputable theory

def f (a x : ℝ) : ℝ := log a (1 / (x + 1))

theorem determine_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)
  (h₃ : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f a x ∧ f a x ≤ 1) :
  a = 1 / 2 :=
begin
  sorry
end

end determine_a_l221_221272


namespace largest_altitude_l221_221424

-- Definitions
variable (A B C : Type*) (AB BC AC : ℝ)
variable (|_ : ℝ → ℝ) -- a "length" function interpreting |.| notation
axiom ab_eq : AB = |_ 8
axiom ac_eq : AC = |_ (2 * BC)

-- Target Definition and Theorem
def is_largest_altitude := true
theorem largest_altitude (A B C : Type*) (AB BC AC : ℝ) [decidable_eq A] [decidable_eq B] [decidable_eq C] :
  is_largest_altitude ∧ AB = 8 ∧ AC = 2 * BC -> 
  altitude := (16 / 3) := begin
  sorry,
end

end largest_altitude_l221_221424


namespace product_of_two_numbers_l221_221890

theorem product_of_two_numbers :
  ∃ x y : ℝ, (x + y = 72) ∧ (x - y = 12) ∧ (x / y = 3 / 2) ∧ (x * y = 1244.16) :=
by
  -- Introduce variables x and y
  obtain ⟨x, y, hx_sum, hy_diff, h_ratio⟩ := sorry
  
  -- Substitute the conditions
  have h_sum : x + y = 72 := hx_sum
  have h_diff : x - y = 12 := hy_diff
  have h_ratio : x / y = 3 / 2 := h_ratio
  
  -- Prove that the product is 1244.16
  obtain ⟨x, y, hx, hy⟩ := sorry
  
  -- Final verification
  use x, y
  split;
  try {assumption}
  exact sorry -- Proof of the product is 1244.16

end product_of_two_numbers_l221_221890


namespace joined_toucans_is_1_l221_221038

-- Define the number of toucans initially
def initial_toucans : ℕ := 2

-- Define the total number of toucans after some join
def total_toucans : ℕ := 3

-- Define the number of toucans that joined
def toucans_joined : ℕ := total_toucans - initial_toucans

-- State the theorem to prove that 1 toucan joined
theorem joined_toucans_is_1 : toucans_joined = 1 :=
by
  sorry

end joined_toucans_is_1_l221_221038


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221124

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221124


namespace expectation_defective_items_variance_of_defective_items_l221_221726
-- Importing the necessary library from Mathlib

-- Define the conditions
def total_products : ℕ := 100
def defective_products : ℕ := 10
def selected_products : ℕ := 3

-- Define the expected number of defective items
def expected_defective_items : ℝ := 0.3

-- Define the variance of defective items
def variance_defective_items : ℝ := 0.2645

-- Lean statements to verify the conditions and results
theorem expectation_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  p * (selected_products: ℝ) = expected_defective_items := by sorry

theorem variance_of_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  let n := (selected_products: ℝ)
  n * p * (1 - p) * (total_products - n) / (total_products - 1) = variance_defective_items := by sorry

end expectation_defective_items_variance_of_defective_items_l221_221726


namespace wrapping_paper_each_present_l221_221845

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l221_221845


namespace problem_statement_l221_221677

noncomputable def max_pairs_mod (A : Finset (Finset α)) : ℕ :=
  let P := A.powerset in
  let P2 := P.powerset in
  let count_pairs := (λ S, P.filter (λ T, S ∈ T ∧ S ⊆ T)).card in
  let total_pairs := (P.toFinset.sum count_pairs) in
  total_pairs % 1000

theorem problem_statement : ∀ (A : Finset (Finset ℕ)),
  A.card = 2015 →
  max_pairs_mod A = 907 :=
begin
  sorry
end

end problem_statement_l221_221677


namespace wrapping_paper_each_present_l221_221847

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l221_221847


namespace shoe_price_l221_221841

theorem shoe_price :
  ∀ (P : ℝ),
    (6 * P + 18 * 2 = 27 * 2) → P = 3 :=
by
  intro P H
  sorry

end shoe_price_l221_221841


namespace find_z_l221_221702

-- Given points A and B.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨1, -3, 3⟩

-- Define the distance function for 3D points.
def distance (P Q : Point3D) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Define M as equidistant from A and B, and on the z-axis.
def M (Z : ℝ) : Point3D := ⟨0, 0, Z⟩

-- Theorem: Prove that if M is equidistant from A and B, then Z = 7
theorem find_z (Z : ℝ) (heq : distance (M Z) A = distance (M Z) B) : Z = 7 :=
by
  sorry

end find_z_l221_221702


namespace value_of_m_l221_221750

theorem value_of_m (m : ℝ) (h : 0 ∈ ({m, m ^ 2 - 2 * m} : set ℝ)) : m = 2 :=
sorry

end value_of_m_l221_221750


namespace first_reduction_percentage_l221_221163

theorem first_reduction_percentage (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.90 = 0.819 * P → x = 9 :=
by
  intro h
  have hp : P ≠ 0 := sorry -- Assuming P is non-zero for price to make sense.
  calc
    (1 - x / 100) * 0.90 = 0.819 : by rw [← mul_assoc, mul_eq_mul_right_iff.mpr (or.inr h)]
    1 - x / 100 = 0.91 : by linarith
    x / 100 = 0.09 : by linarith
    x = 9 : by linarith

end first_reduction_percentage_l221_221163


namespace whale_fifth_hour_consumption_l221_221618

variables (x y : ℕ)

theorem whale_fifth_hour_consumption
  (h1 : ∀ n, 1 ≤ n → n ≤ 8 → let consumption := x + (n - 1) * y in (∑ i in finset.range 8, consumption) = 1200)
  (h2 : ∀ n, x + (n - 1) * y ≥ 0) :
  (x + 4 * y) = (x + 4 * y) :=
begin
  sorry
end

end whale_fifth_hour_consumption_l221_221618


namespace speed_A_correct_l221_221568

noncomputable def speed_A : ℝ :=
  200 / (19.99840012798976 * 60)

theorem speed_A_correct :
  speed_A = 0.16668 :=
sorry

end speed_A_correct_l221_221568


namespace total_students_in_classrooms_l221_221517

theorem total_students_in_classrooms (tina_students maura_students zack_students : ℕ) 
    (h1 : tina_students = maura_students)
    (h2 : zack_students = (tina_students + maura_students) / 2)
    (h3 : 22 + 1 = zack_students) : 
    tina_students + maura_students + zack_students = 69 := 
by 
  -- Proof steps would go here, but we include 'sorry' as per the instructions.
  sorry

end total_students_in_classrooms_l221_221517


namespace conveyance_spent_is_correct_l221_221466

-- Defining constants for Rohan's expenditures and savings
def food_percentage : ℝ := 0.4
def rent_percentage : ℝ := 0.2
def entertainment_percentage : ℝ := 0.1
def savings : ℝ := 1500
def salary : ℝ := 7500

-- The total percentage of salary spent on food, house rent, and entertainment
def accounted_percentage : ℝ := food_percentage + rent_percentage + entertainment_percentage

-- Total money spent on food, house rent, and entertainment
def accounted_money : ℝ := accounted_percentage * salary

-- The remaining salary after savings
def remaining_salary : ℝ := salary - savings

-- The money spent on conveyance
def conveyance_money : ℝ := remaining_salary - accounted_money

-- The percentage of salary spent on conveyance
def conveyance_percentage : ℝ := (conveyance_money / salary) * 100

-- The proof statement asserting the solved percentage
theorem conveyance_spent_is_correct : conveyance_percentage = 10 := sorry

end conveyance_spent_is_correct_l221_221466


namespace number_of_arrangements_six_people_not_head_l221_221574

theorem number_of_arrangements_six_people_not_head (n : ℕ) (h : n = 6) :
  (∃ P : Fin n → Prop, P 0 = False) → 
  (∑ P, P ≠ 0) * n! = 600 :=
by
  sorry

end number_of_arrangements_six_people_not_head_l221_221574


namespace multiples_of_3_or_4_probability_l221_221881

theorem multiples_of_3_or_4_probability :
  let total_cards := 36
  let multiples_of_3 := 12
  let multiples_of_4 := 9
  let multiples_of_both := 3
  let favorable_outcomes := multiples_of_3 + multiples_of_4 - multiples_of_both
  let probability := (favorable_outcomes : ℚ) / total_cards
  probability = 1 / 2 :=
by
  sorry

end multiples_of_3_or_4_probability_l221_221881


namespace sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l221_221769

def row_10_pascals_triangle : List ℕ := [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]

theorem sum_of_row_10_pascals_triangle :
  (List.sum row_10_pascals_triangle) = 1024 := by
  sorry

theorem sum_of_squares_of_row_10_pascals_triangle :
  (List.sum (List.map (fun x => x * x) row_10_pascals_triangle)) = 183756 := by
  sorry

end sum_of_row_10_pascals_triangle_sum_of_squares_of_row_10_pascals_triangle_l221_221769


namespace sequence_is_increasing_l221_221886

def S (n : ℕ) : ℤ :=
  n^2 + 2 * n - 2

def a : ℕ → ℤ
| 0       => 0
| 1       => 1
| n + 1   => S (n + 1) - S n

theorem sequence_is_increasing : ∀ n m : ℕ, n < m → a n < a m :=
  sorry

end sequence_is_increasing_l221_221886


namespace trigonometric_identity_l221_221608

theorem trigonometric_identity (α : ℝ) :
  cos(α)^2 + cos(α + 60 * (π / 180))^2 - cos(α) * cos(α + 60 * (π / 180)) = 3 / 4 := by
  sorry

end trigonometric_identity_l221_221608


namespace exponent_equality_l221_221303

theorem exponent_equality (y : ℕ) (z : ℕ) (h1 : 16 ^ y = 4 ^ z) (h2 : y = 8) : z = 16 := by
  sorry

end exponent_equality_l221_221303


namespace octagon_area_ratio_l221_221129

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221129


namespace greatest_x_value_l221_221668

theorem greatest_x_value :
  ∃ x, (∀ y, (y ≠ 6) → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧ (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧ x ≠ 6 ∧ x = -3 :=
sorry

end greatest_x_value_l221_221668


namespace tan_alpha_value_l221_221711

open Real

theorem tan_alpha_value (alpha : ℝ) (h1 : cos (π / 2 + alpha) = 2 * sqrt 2 / 3) (h2 : alpha ∈ Ioo (π / 2) (3 * π / 2)) : 
    tan alpha = 2 * sqrt 2 := 
    sorry

end tan_alpha_value_l221_221711


namespace solve_for_n_l221_221858

theorem solve_for_n (n : ℤ) (h : 3^(2 * n + 1) = 1 / 27) : n = -2 := by
  sorry

end solve_for_n_l221_221858


namespace trapezoid_area_l221_221210

variables (R₁ R₂ : ℝ)

theorem trapezoid_area (h_eq : h = 4 * R₁ * R₂ / (R₁ + R₂)) (mn_eq : mn = 2 * Real.sqrt (R₁ * R₂)) :
  S_ABCD = 8 * R₁ * R₂ * Real.sqrt (R₁ * R₂) / (R₁ + R₂) :=
sorry

end trapezoid_area_l221_221210


namespace practice_time_until_next_game_l221_221451

theorem practice_time_until_next_game
  (practice_hours_weekday : ℕ := 3)
  (practice_days_weekday : ℕ := 5)
  (practice_hours_saturday : ℕ := 5)
  (total_practice_hours_until_game : ℕ := 60) :
  ∃ (weeks : ℕ), weeks = 3 :=
by
let practice_hours_per_week := practice_hours_weekday * practice_days_weekday + practice_hours_saturday
have total_practice : practice_hours_per_week = 20 := by sorry
have weeks := total_practice_hours_until_game / practice_hours_per_week
have weeks_proof : weeks = 3 := by sorry
use weeks
exact weeks_proof

end practice_time_until_next_game_l221_221451


namespace cube_sum_from_square_l221_221294

noncomputable def a_plus_inv_a_squared_eq_5 (a : ℝ) : Prop :=
  (a + 1/a) ^ 2 = 5

theorem cube_sum_from_square (a : ℝ) (h : a_plus_inv_a_squared_eq_5 a) :
  a^3 + (1/a)^3 = 2 * Real.sqrt 5 ∨ a^3 + (1/a)^3 = -2 * Real.sqrt 5 :=
by
  sorry

end cube_sum_from_square_l221_221294


namespace distance_from_center_to_line_l221_221360

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221360


namespace overall_loss_percentage_l221_221605

theorem overall_loss_percentage 
  (CP_radio : ℕ) (Ad_radio : ℕ) (SP_radio : ℕ)
  (CP_tv : ℕ) (Ad_tv : ℕ) (SP_tv : ℕ)
  (CP_fridge : ℕ) (Ad_fridge : ℕ) (SP_fridge : ℕ)
  (H1 : CP_radio = 1500) (H2 : Ad_radio = 100) (H3 : SP_radio = 1335)
  (H4 : CP_tv = 5500) (H5 : Ad_tv = 200) (H6 : SP_tv = 5050)
  (H7 : CP_fridge = 12000) (H8 : Ad_fridge = 500) (H9 : SP_fridge = 11400) :
  let TCP_radio := CP_radio + Ad_radio
  let TCP_tv := CP_tv + Ad_tv
  let TCP_fridge := CP_fridge + Ad_fridge
  let total_TCP := TCP_radio + TCP_tv + TCP_fridge
  let total_SP := SP_radio + SP_tv + SP_fridge
  let total_loss := total_TCP - total_SP
  (total_loss.toFloat / total_TCP.toFloat) * 100 ≈ 10.18 :=
by
  sorry

end overall_loss_percentage_l221_221605


namespace log_ride_cost_l221_221649

theorem log_ride_cost (rides_FerrisWheel rides_RollerCoaster rides_LogRide : ℕ)
  (cost_FerrisWheel cost_RollerCoaster tickets_DollyHas tickets_DollyBuys : ℕ) :
  rides_FerrisWheel = 2 →
  rides_RollerCoaster = 3 →
  rides_LogRide = 7 →
  cost_FerrisWheel = 2 →
  cost_RollerCoaster = 5 →
  tickets_DollyHas = 20 →
  tickets_DollyBuys = 6 →
  let total_tickets_needed := tickets_DollyHas + tickets_DollyBuys
  let tickets_FerrisWheel := cost_FerrisWheel * rides_FerrisWheel
  let tickets_RollerCoaster := cost_RollerCoaster * rides_RollerCoaster
  let tickets_LogRide := total_tickets_needed - (tickets_FerrisWheel + tickets_RollerCoaster)
  tickets_LogRide / rides_LogRide = 1 := 
by
  intros
  simp only at *
  sorry

end log_ride_cost_l221_221649


namespace gift_laptops_unique_brands_l221_221195

-- Since we are translating a problem related to combinatorics, we need to
-- define the problem in Lean language using appropriate mathematical notation.
noncomputable def numWaysToGiftLaptops (totalBrands : Nat) (children : Nat) : Nat :=
  match children with
  | 0     => 1
  | _     => totalBrands * numWaysToGiftLaptops (totalBrands - 1) (children - 1)

theorem gift_laptops_unique_brands : numWaysToGiftLaptops 15 3 = 2730 := 
by
  sorry

end gift_laptops_unique_brands_l221_221195


namespace dice_probability_l221_221553

theorem dice_probability : 
  (∃ (die : Type) [finite die] [decidable_eq die] (roll : die), 
  (∀ x : die, x ∈ {1, 2, 3, 4, 5, 6})) → 
  ∃ p : ℚ, p = 1 / 3 :=
by
  sorry

end dice_probability_l221_221553


namespace octagon_area_ratio_l221_221125

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221125


namespace area_ratio_is_correct_l221_221146

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221146


namespace complement_intersection_eq_l221_221821

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_eq :
  U \ (A ∩ B) = {1, 4, 5} := by
  sorry

end complement_intersection_eq_l221_221821


namespace distance_from_center_to_line_l221_221359

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221359


namespace equation_of_line_l221_221868

theorem equation_of_line (x y : ℝ) :
  (∃ (x1 y1 : ℝ), (x1 = 0) ∧ (y1= 2) ∧ (y - y1 = 2 * (x - x1))) → (y = 2 * x + 2) :=
by
  sorry

end equation_of_line_l221_221868


namespace range_of_f_l221_221236

def f (x : ℝ) : ℝ := (x + (1 / x)) / (⌊x⌋ * ⌊1 / x⌋ + ⌊x⌋ + ⌊1 / x⌋ + 1)

theorem range_of_f (x : ℝ) (hx : x > 0) :
  (∃ y, y = f x ∧ (y = 1/2 ∨ (5/6 ≤ y ∧ y < 5/4))) :=
by
  sorry

end range_of_f_l221_221236


namespace distance_from_center_to_line_of_tangent_circle_l221_221322

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221322


namespace distance_to_line_is_constant_l221_221358

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221358


namespace octagon_area_ratio_l221_221154

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221154


namespace vet_appointments_cost_l221_221420

variable (x : ℝ)

def JohnVetAppointments (x : ℝ) : Prop := 
  (x + 0.20 * x + 0.20 * x + 100 = 660)

theorem vet_appointments_cost :
  (∃ x : ℝ, JohnVetAppointments x) → x = 400 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  simp [JohnVetAppointments] at hx
  sorry

end vet_appointments_cost_l221_221420


namespace distance_from_center_to_line_l221_221364

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221364


namespace distance_from_center_to_line_of_tangent_circle_l221_221317

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221317


namespace SUV_highway_mpg_l221_221623

theorem SUV_highway_mpg
  (city_mpg : ℝ) (max_distance : ℝ) (gallons : ℝ) 
  (h_city_mpg : city_mpg = 7.6)
  (h_max_distance : max_distance = 268.4)
  (h_gallons : gallons = 22) :
  (max_distance / gallons) = 12.2 :=
by
  rw [h_max_distance, h_gallons]
  norm_num
  sorry

end SUV_highway_mpg_l221_221623


namespace product_first_20_a_n_l221_221269

noncomputable def b_n (n : ℕ) : ℕ := 3^n
noncomputable def c_n (n : ℕ) : ℕ := 4 * n + 1
noncomputable def a_n (n : ℕ) : ℕ := 9^n

theorem product_first_20_a_n : 
  ∏ i in Finset.range 20, a_n (i + 1) = 9 ^ 210 :=
by
  sorry

end product_first_20_a_n_l221_221269


namespace ratio_of_areas_of_octagons_l221_221063

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221063


namespace cos_B_in_third_quadrant_l221_221377

theorem cos_B_in_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB: Real.sin B = 5 / 13) : Real.cos B = - 12 / 13 := by
  sorry

end cos_B_in_third_quadrant_l221_221377


namespace shortest_side_length_l221_221724

theorem shortest_side_length (perimeter : ℝ) (shortest : ℝ) (side1 side2 side3 : ℝ) 
  (h1 : side1 + side2 + side3 = perimeter)
  (h2 : side1 = 2 * shortest)
  (h3 : side2 = 2 * shortest) :
  shortest = 3 := by
  sorry

end shortest_side_length_l221_221724


namespace width_of_crate_l221_221939

-- Define crate dimensions
structure Crate where
  length : ℕ
  height : ℕ
  width  : ℕ

-- Define gas tank with radius
structure GasTank where
  radius : ℕ

-- Define the specific crate and gas tank
def myCrate : Crate := { length := 12, height := 18, width := _ }
def myTank : GasTank := { radius := 8 }

-- Specify the diameter of the tank
def tankDiameter (tank : GasTank) : ℕ := 2 * tank.radius

-- The width of the crate should be such that it equals the tank's diameter
theorem width_of_crate (c : Crate) (t : GasTank) (h : c.width = tankDiameter t) : 
  c.width = 16 :=
by 
  sorry

end width_of_crate_l221_221939


namespace octagon_area_ratio_l221_221157

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221157


namespace ratio_of_octagon_areas_l221_221071

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221071


namespace aquarium_visitors_not_ill_l221_221651

theorem aquarium_visitors_not_ill :
  let visitors_monday := 300
  let visitors_tuesday := 500
  let visitors_wednesday := 400
  let ill_monday := (15 / 100) * visitors_monday
  let ill_tuesday := (30 / 100) * visitors_tuesday
  let ill_wednesday := (20 / 100) * visitors_wednesday
  let not_ill_monday := visitors_monday - ill_monday
  let not_ill_tuesday := visitors_tuesday - ill_tuesday
  let not_ill_wednesday := visitors_wednesday - ill_wednesday
  let total_not_ill := not_ill_monday + not_ill_tuesday + not_ill_wednesday
  total_not_ill = 925 := 
by
  sorry

end aquarium_visitors_not_ill_l221_221651


namespace area_ratio_is_correct_l221_221145

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221145


namespace other_root_of_quadratic_l221_221229

theorem other_root_of_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + m = 0 → x = -1) → (∀ y : ℝ, y^2 - 4 * y + m = 0 → y = 5) :=
sorry

end other_root_of_quadratic_l221_221229


namespace translate_complex_l221_221167

def translation (z w : ℂ) := z + w

theorem translate_complex : ∀ (w : ℂ), translation (1 + 3 * Complex.I) w = 4 + 2 * Complex.I →
  translation (3 - 2 * Complex.I) w = 6 - 3 * Complex.I :=
by
  assume w h
  have h_w : w = 3 - Complex.I := by sorry -- solving for w from the given translation
  rw [h_w]
  rw [h_w] at h
  rw [translation]
  rw [translation]
  rw [← Complex.add_assoc]
  rw [← Complex.add_assoc]
  rw [Complex.add_comm (1 + 3 * Complex.I) w]
  rw [Complex.add_comm (3 - 2 * Complex.I) w]
  sorry


end translate_complex_l221_221167


namespace num_arithmetic_progressions_1000_l221_221396

theorem num_arithmetic_progressions_1000 : 
  (finset.sum (finset.range 334) (λ d, 1000 - 3 * d)) = 166167 :=
by {
  sorry
}

end num_arithmetic_progressions_1000_l221_221396


namespace octagon_area_ratio_l221_221151

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221151


namespace range_h_l221_221693

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6)

def g (x : ℝ) : ℝ := sin (2 * x - π / 6)

def h (x : ℝ) : ℝ := f x + g x + 2 * cos (x) ^ 2 - 1

theorem range_h :
  ∀ x, 0 ≤ x ∧ x ≤ π / 2 → -1 ≤ h x ∧ h x ≤ 2 := sorry

end range_h_l221_221693


namespace cannot_return_to_start_l221_221406

def transformation_N (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 2 * p.1)

def transformation_S (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 - 2 * p.1)

def transformation_E (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, p.2)

def transformation_W (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - 2 * p.2, p.2)

theorem cannot_return_to_start :
  ∀ (f : list (ℝ × ℝ → ℝ × ℝ)),
  f ≠ [] →
  f.head (1, real.sqrt 2) ≠ (1, real.sqrt 2) :=
by
  intros f h
  sorry

end cannot_return_to_start_l221_221406


namespace find_b_dot_c_l221_221430

variables (a b c : ℝ^3)

-- Assuming the given conditions
axiom norm_a : ‖a‖ = 1
axiom norm_b : ‖b‖ = 1
axiom norm_a_add_b : ‖a + b‖ = Real.sqrt 2
axiom vec_eqn : c - 2 • a - b = 2 • (a × b)

theorem find_b_dot_c :
  b ⬝ c = 1 :=
sorry

end find_b_dot_c_l221_221430


namespace angle_ACD_is_30_degrees_l221_221799

theorem angle_ACD_is_30_degrees
  (A B C D : Type)
  (α β γ : Type) -- Types representing angles
  (AC BC : ℝ) -- Lengths
  (isosceles_obtuse : α = β) -- Triangle ABC is isosceles and obtuse-angled
  (AD_eq_circumradius_BCD : Real) -- AD equals the circumradius of triangle BCD
  : αCD = 30 ^∘ := -- Need to show ∠ACD = 30 degrees

  sorry -- Proof to be completed

end angle_ACD_is_30_degrees_l221_221799


namespace dave_diner_total_cost_l221_221173

theorem dave_diner_total_cost (burger_count : ℕ) (fries_count : ℕ)
  (burger_cost : ℕ) (fries_cost : ℕ)
  (discount_threshold : ℕ) (discount_amount : ℕ)
  (h1 : burger_count >= discount_threshold) :
  burger_count = 6 → fries_count = 5 → burger_cost = 4 → fries_cost = 3 →
  discount_threshold = 4 → discount_amount = 2 →
  (burger_count * (burger_cost - discount_amount) + fries_count * fries_cost) = 27 :=
by
  intros hbc hfc hbcost hfcs dth da
  sorry

end dave_diner_total_cost_l221_221173


namespace range_of_m_specific_value_of_m_l221_221780

-- Definitions
def curve_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def line_equation (m t x y : ℝ) : Prop := x = m + 3 * t ∧ y = 4 * t
def intersection (m t : ℝ) : Prop := t^2 + (6/5) * (m - 1) * t + (m - 1)^2 - 1 = 0

-- Proving the range of m
theorem range_of_m (m : ℝ) : 
  (∀ (x y : ℝ), (∃ t : ℝ, curve_equation x y ∧ line_equation m t x y)) → (- 1 / 4 < m ∧ m < 9 / 4) :=
sorry

-- Proving the specific value of m
theorem specific_value_of_m (m : ℝ) : 
  (|MA| * |MB| = 1) → m = 1 :=
sorry

end range_of_m_specific_value_of_m_l221_221780


namespace range_of_a_for_monotonic_decrease_l221_221277

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 16 * Real.log x

def is_monotonically_decreasing_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ set.Icc (a-1) (a+2), ∀ y ∈ set.Icc (a-1) (a+2), x ≤ y → f x ≥ f y

theorem range_of_a_for_monotonic_decrease :
  (∀ a : ℝ, is_monotonically_decreasing_on_interval f a → 1 < a ∧ a ≤ 2) ∧ 
  (∀ a : ℝ, 1 < a ∧ a ≤ 2 → is_monotonically_decreasing_on_interval f a) :=
begin
  -- sorry, proof not required
  sorry,
end

end range_of_a_for_monotonic_decrease_l221_221277


namespace probability_divisibility_9_correct_l221_221819

-- Define the set S
def S : Set ℕ := { n | ∃ a b: ℕ, 0 ≤ a ∧ a < 40 ∧ 0 ≤ b ∧ b < 40 ∧ a ≠ b ∧ n = 2^a + 2^b }

-- Define the criteria for divisibility by 9
def divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

-- Define the total size of set S
def size_S : ℕ := 780  -- as calculated from combination

-- Count valid pairs (a, b) such that 2^a + 2^b is divisible by 9
def valid_pairs : ℕ := 133  -- as calculated from summation

-- Define the probability
def probability_divisible_by_9 : ℕ := valid_pairs / size_S

-- The proof statement
theorem probability_divisibility_9_correct:
  (valid_pairs : ℚ) / (size_S : ℚ) = 133 / 780 := sorry

end probability_divisibility_9_correct_l221_221819


namespace calculate_fraction_of_ids_l221_221586

theorem calculate_fraction_of_ids : 
  let chars := ["C", "A", "T", "2", "0", "1"]
  let ids_with_no_repeated_2 : ℕ := (finset.range 6).powerset.filter (λ s, card s = 5).card
  let ids_with_two_2s : ℕ := (finset.range 2).choose 5 2 * (finset.range 5).powerset.filter (λ s, card s = 3).card
  let total_ids := ids_with_no_repeated_2 + ids_with_two_2s
  in total_ids = 1320 → total_ids / 10 = 132 := 
by
  sorry

end calculate_fraction_of_ids_l221_221586


namespace oh_squared_l221_221426

theorem oh_squared {a b c R : ℝ} (O H : Point) (A B C : Triangle) 
  (hac : triangle.is_circumcenter O A B C) (hao : triangle.is_orthocenter H A B C)
  (sides : A.side_lengths = (a, b, c)) 
  (circumradius : R = 8)
  (side_lengths_sq_sum : a^2 + b^2 + c^2 = 50) : 
  oh_squared O H = 526 :=
sorry

end oh_squared_l221_221426


namespace median_of_consecutive_integers_l221_221006

theorem median_of_consecutive_integers (a b : ℤ) (h : a + b = 50) : 
  (a + b) / 2 = 25 := 
by 
  sorry

end median_of_consecutive_integers_l221_221006


namespace fill_in_the_blank_correct_option_l221_221902

-- Assume each option is defined
def options := ["the other", "some", "another", "other"]

-- Define a helper function to validate the correct option
def is_correct_option (opt: String) : Prop :=
  opt = "another"

-- The main problem statement
theorem fill_in_the_blank_correct_option :
  (∀ opt, opt ∈ options → is_correct_option opt → opt = "another") :=
by
  intro opt h_option h_correct
  simp [is_correct_option] at h_correct
  exact h_correct

-- Test case to check the correct option
example : is_correct_option "another" :=
by
  simp [is_correct_option]

end fill_in_the_blank_correct_option_l221_221902


namespace prob_win_3_1_correct_l221_221027

-- Defining the probability for winning a game
def prob_win_game : ℚ := 2 / 3

-- Defining the probability for losing a game
def prob_lose_game : ℚ := 1 - prob_win_game

-- A function to calculate the probability of winning the match with a 3:1 score
def prob_win_3_1 : ℚ :=
  let combinations := 3 -- Number of ways to lose exactly 1 game in the first 3 games (C_3^1)
  let win_prob := prob_win_game ^ 3 -- Probability for winning 3 games
  let lose_prob := prob_lose_game -- Probability for losing 1 game
  combinations * win_prob * lose_prob

-- The theorem that states the probability that player A wins with a score of 3:1
theorem prob_win_3_1_correct : prob_win_3_1 = 8 / 27 := by
  sorry

end prob_win_3_1_correct_l221_221027


namespace tan_x_eq_sqrt3_l221_221281

theorem tan_x_eq_sqrt3 (x : ℝ) (hx : sin (x + real.pi / 9) = cos (x + real.pi / 18) + cos (x - real.pi / 18)) :
  tan x = real.sqrt 3 :=
by
  sorry

end tan_x_eq_sqrt3_l221_221281


namespace wrapping_paper_per_present_l221_221850

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l221_221850


namespace difference_of_sides_of_rectangle_l221_221947

noncomputable def rectangle_area (x y : ℝ) : ℝ := x * y
noncomputable def rectangle_diagonal (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

theorem difference_of_sides_of_rectangle
  (x y A d : ℝ) (h1 : x > y) (h2 : x * y = A) (h3 : real.sqrt (x^2 + y^2) = d) :
  x - y = real.sqrt (d^2 - 4 * A) :=
sorry

end difference_of_sides_of_rectangle_l221_221947


namespace ratio_area_octagons_correct_l221_221109

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221109


namespace train_crossing_time_l221_221616

/-- Time for a train of length 1500 meters traveling at 108 km/h to cross an electric pole is 50 seconds -/
theorem train_crossing_time (length : ℕ) (speed_kmph : ℕ) 
    (h₁ : length = 1500) (h₂ : speed_kmph = 108) : 
    (length / ((speed_kmph * 1000) / 3600) = 50) :=
by
  sorry

end train_crossing_time_l221_221616


namespace Mrs_Brown_pays_108_for_shoes_l221_221416

def discounted_price (original price discount: Float): Float :=
  original - (original * (discount / 100))

def final_price (original_price: Float) (discount1 discount2: Float): Float :=
    let price_after_first_discount := discounted_price original_price discount1
    discounted_price price_after_first_discount discount2

theorem Mrs_Brown_pays_108_for_shoes :
  (number_of_children >= 3) → (original_price = 125) → (discount1 = 10) → (discount2 = 4) →
  final_price original_price discount1 discount2 = 108 := 
by
  intros h_children h_price h_discount1 h_discount2
  sorry

end Mrs_Brown_pays_108_for_shoes_l221_221416


namespace sine_wave_solution_l221_221968

theorem sine_wave_solution (a b c : ℝ) (h_pos_a : a > 0) 
  (h_amp : a = 3) 
  (h_period : (2 * Real.pi) / b = Real.pi) 
  (h_peak : (Real.pi / (2 * b)) - (c / b) = Real.pi / 6) : 
  a = 3 ∧ b = 2 ∧ c = Real.pi / 6 :=
by
  -- Lean code to construct the proof will appear here
  sorry

end sine_wave_solution_l221_221968


namespace area_of_trajectory_l221_221739

theorem area_of_trajectory (A B : ℝ × ℝ) (PA PB : ℝ) :
  A = (-2,0) → B = (1,0) → 
  ∀ P : ℝ × ℝ, PA = real.dist P A → PB = real.dist P B → PA = 2 * PB → 
  ∃ C : ℝ × ℝ, ∃ r : ℝ, P = C ∧ r = 2 ∧ 
  let circle_area := real.pi * r^2 
  in circle_area = 4 * real.pi :=
by sorry

end area_of_trajectory_l221_221739


namespace angle_bisector_slope_l221_221645

theorem angle_bisector_slope :
  let m₁ := 2
  let m₂ := 5
  let k := (7 - 2 * Real.sqrt 5) / 11
  True :=
by admit

end angle_bisector_slope_l221_221645


namespace butter_needed_for_4_dozen_l221_221528

theorem butter_needed_for_4_dozen (original_dozen : ℕ) (original_butter : ℝ) (desired_dozen : ℕ) (desired_butter : ℝ) 
  (h1 : original_dozen = 16) 
  (h2 : original_butter = 4) 
  (h3 : desired_dozen = 4) :
  desired_butter = 1 :=
by
  -- Conditions
  have scale_factor : ℝ := desired_dozen / original_dozen,
  have butter_needed : ℝ := original_butter * scale_factor,
  -- Calculation
  have sf_value : scale_factor = 1 / 4 := by sorry,
  have bn_value : butter_needed = 1 := by sorry,
  -- Conclusion
  exact bn_value

end butter_needed_for_4_dozen_l221_221528


namespace baseball_cap_factory_l221_221039

theorem baseball_cap_factory (caps_week1 caps_week2 caps_week3 : ℕ) 
  (h1 : caps_week1 = 320) (h2 : caps_week2 = 400) (h3 : caps_week3 = 300) : 
  let avg_caps_week : ℕ := (caps_week1 + caps_week2 + caps_week3) / 3 in
  let total_caps_4_weeks : ℕ := caps_week1 + caps_week2 + caps_week3 + avg_caps_week in
  total_caps_4_weeks = 1360 := 
by
  sorry

end baseball_cap_factory_l221_221039


namespace number_of_digits_of_p_is_11_l221_221923

def p : ℕ := (167 * 283 * 593 * 907 * 127 * 853 * 23) / (373 * 577 * 11)

theorem number_of_digits_of_p_is_11 : nat.floor (real.log10 p) + 1 = 11 := by
  sorry

end number_of_digits_of_p_is_11_l221_221923


namespace trains_meeting_distance_l221_221526

theorem trains_meeting_distance :
  ∃ D T : ℕ, (D = 20 * T) ∧ (D + 60 = 25 * T) ∧ (2 * D + 60 = 540) :=
by
  sorry

end trains_meeting_distance_l221_221526


namespace hypotenuse_length_l221_221180

theorem hypotenuse_length (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 60 := 
by 
  use 60
  sorry

end hypotenuse_length_l221_221180


namespace min_velocity_increase_l221_221770

theorem min_velocity_increase (V_A V_B V_C : ℕ) (d_AB d_AC : ℕ) (h_VB_lt_VA : V_B < V_A) :
  d_AB = 50 ∧ d_AC = 300 ∧ V_B = 50 ∧ V_C = 70 ∧ V_A = 68 →
  (let ΔV := (370 / 5) - V_A in ΔV = 6) :=
by
  intros h_conditions,
  cases h_conditions with h_dAB h_remainder,
  cases h_remainder with h_dAC h_remainder2,
  cases h_remainder2 with h_VB h_remainder3,
  cases h_remainder3 with h_VC h_VA,
  let quotient := 370 / 5,
  let ΔV := quotient - 68,
  focus
    { rw [h_VB, h_VC, h_VA] at ⊢,
      sorry }

end min_velocity_increase_l221_221770


namespace find_f_max_value_l221_221815

-- Lean definition of conditions for a "good" sequence of points
def is_good (n : ℕ) (P : fin n → ℝ × ℝ) : Prop :=
  n ≥ 3 ∧
  (∀ (i j k : fin n), i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    ¬ collinear (P i) (P j) (P k)) ∧
  non_self_intersecting (polygonal_path P) ∧
  (∀ i : fin (n - 2), counterclockwise (P i) (P (i + 1)) (P (i + 2)))

-- Lean definition for f(n)
def f (n : ℕ) : ℕ :=
  n^2 - 4*n + 6

-- Lean statement of the proof problem
theorem find_f_max_value (n : ℕ) (h : n ≥ 3) (P : fin n → ℝ × ℝ) (hP : is_good n P) : 
  ∃ (σ : equiv (fin n) (fin n)), is_good n (σ ∘ P) ∧
  f(n) = n^2 - 4*n + 6 :=
sorry

end find_f_max_value_l221_221815


namespace rare_card_cost_l221_221898

theorem rare_card_cost
  (num_rare : ℕ) (num_uncommon : ℕ) (num_common : ℕ)
  (cost_uncommon : ℝ) (cost_common : ℝ) (total_cost : ℝ)
  (h_rare : num_rare = 19) (h_uncommon : num_uncommon = 11) (h_common : num_common = 30)
  (h_cost_uncommon : cost_uncommon = 0.50) (h_cost_common : cost_common = 0.25)
  (h_total_cost : total_cost = 32) :
  ∃ (x : ℝ), 19 * x + 11 * 0.50 + 30 * 0.25 = 32 ∧ x = 1 :=
by
  use 1
  split
  sorry
  refl

end rare_card_cost_l221_221898


namespace identity_true_for_any_abc_l221_221679

theorem identity_true_for_any_abc : 
  ∀ (a b c : ℝ), (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
by
  sorry

end identity_true_for_any_abc_l221_221679


namespace number_of_outcomes_exactly_two_evening_l221_221682

theorem number_of_outcomes_exactly_two_evening (chickens : Finset ℕ) (h_chickens : chickens.card = 4) 
    (day_places evening_places : ℕ) (h_day_places : day_places = 2) (h_evening_places : evening_places = 3) :
    ∃ n, n = (chickens.card.choose 2) ∧ n = 6 :=
by
  sorry

end number_of_outcomes_exactly_two_evening_l221_221682


namespace ratio_of_areas_of_octagons_l221_221088

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221088


namespace teresa_age_when_michiko_born_l221_221484

def conditions (T M Michiko K Yuki : ℕ) : Prop := 
  T = 59 ∧ 
  M = 71 ∧ 
  M - Michiko = 38 ∧ 
  K = Michiko - 4 ∧ 
  Yuki = K - 3 ∧ 
  (Yuki + 3) - (26 - 25) = 25

theorem teresa_age_when_michiko_born :
  ∃ T M Michiko K Yuki, conditions T M Michiko K Yuki → T - Michiko = 26 :=
  by
  sorry

end teresa_age_when_michiko_born_l221_221484


namespace find_AE_l221_221810

noncomputable theory

open_locale classical

structure EquallySpacedCollinearPoints :=
(A B C D : Type*)
(equally_spaced : ∀ P ∈ {A, B, C, D}, ∃ d : ℝ, dist P Q = d)

def circle (diam : ℝ) := {p : ℝ × ℝ // dist p.1 p.2 = diam}

variables (A B C D E : Type*)
variables (AB_eq : dist A B = 2 * real.sqrt 3)
variables (BC_eq : dist B C = 2 * real.sqrt 3)
variables (CD_eq : dist C D = 2 * real.sqrt 3)
variables (AD_eq : dist A D = 6 * real.sqrt 3)
variables (BD_eq : dist B D = 4 * real.sqrt 3)

def ω := circle AD_eq
def ω' := circle BD_eq

-- Tangent line through A to ω' intersects ω at E
axiom tangent_intersect (tangent_to_omega'_at_A : A) (intersects_omega_at_E : E)

-- The theorem to prove
theorem find_AE : dist A E = 9 :=
sorry

end find_AE_l221_221810


namespace rectangle_area_equals_perimeter_right_triangle_area_equals_perimeter_l221_221290

-- Part (a)
def unique_rectangles_with_equal_area_and_perimeter : ℕ :=
  (2 : ℕ)

theorem rectangle_area_equals_perimeter :
  let unique_rectangles_count := unique_rectangles_with_equal_area_and_perimeter in
  ∃ ab_pairs : finset (ℕ × ℕ),
    (∀ p ∈ ab_pairs, (p.1 * p.2 = 2 * (p.1 + p.2))) ∧
    ab_pairs.card = unique_rectangles_count := sorry

-- Part (b)
def unique_right_triangles_with_equal_area_and_perimeter : ℕ :=
  (1 : ℕ)

theorem right_triangle_area_equals_perimeter :
  let unique_triangles_count := unique_right_triangles_with_equal_area_and_perimeter in
  ∃ abc_triples : finset (ℕ × ℕ × ℕ),
    (∀ t ∈ abc_triples, (t.1 ≤ t.2 ∧ t.2 < t.3) ∧
      (t.1 * t.2 = 2 * (t.1 + t.2 + t.3)) ∧
      (t.1^2 + t.2^2 = t.3^2)) ∧
    abc_triples.card = unique_triangles_count := sorry

end rectangle_area_equals_perimeter_right_triangle_area_equals_perimeter_l221_221290


namespace graph_translation_l221_221520

theorem graph_translation : 
  ∀ x : ℝ, 4 * cos (2 * x) = 4 * cos (2 * (x - (π / 8)) + π / 4) :=
by
  sorry

end graph_translation_l221_221520


namespace inequality_division_by_positive_l221_221754

theorem inequality_division_by_positive (x y : ℝ) (h : x > y) : (x / 5 > y / 5) :=
by
  sorry

end inequality_division_by_positive_l221_221754


namespace freezing_temperatures_l221_221496

theorem freezing_temperatures:
  let water_freezing := 0
  let alcohol_freezing := -117
  let mercury_freezing := -39
  (∀ t in [water_freezing, alcohol_freezing, mercury_freezing], t ≤ 0) ∧ 
  (∀ t in [water_freezing, alcohol_freezing, mercury_freezing], t ≥ -117) :=
  by {
    sorry
  }

end freezing_temperatures_l221_221496


namespace z_odd_perfect_square_l221_221292

theorem z_odd_perfect_square (x y z : ℕ) (hz : z > 0) (hx : x > 0) (hy : y > 0) 
  (h : z * (xz + 1)^2 = (5z + 2y) * (2z + y)) : ∃ (d : ℕ), z = d^2 ∧ d % 2 = 1 :=
by
  sorry

end z_odd_perfect_square_l221_221292


namespace redesigned_survey_respondents_l221_221610

theorem redesigned_survey_respondents 
  (total_customers_orig : ℕ) (responses_orig : ℕ) (responses_redesigned : ℕ) (delta_rate : ℝ)
  (h_total_customers_orig : total_customers_orig = 60)
  (h_responses_orig : responses_orig = 7)
  (h_responses_redesigned : responses_redesigned = 9)
  (h_delta_rate : delta_rate = 0.02) :
  ∃ (total_customers_redesigned : ℕ), 
    abs (total_customers_redesigned - 66) ≤ 1 ∧
    abs ((responses_redesigned / (total_customers_redesigned : ℝ)) - (responses_orig / (total_customers_orig : ℝ)) - delta_rate) < 0.02 :=
by
  sorry

end redesigned_survey_respondents_l221_221610


namespace greatest_x_solution_l221_221667

theorem greatest_x_solution (x : ℝ) (h₁ : (x^2 - x - 30) / (x - 6) = 2 / (x + 4)) : x ≤ -3 :=
sorry

end greatest_x_solution_l221_221667


namespace initial_average_jellybeans_per_bag_l221_221893

theorem initial_average_jellybeans_per_bag (A : ℝ) :
  let total_before := 34 * A
  let total_after := total_before + 362
  let new_avg := (A + 7)
  total_after / 35 = new_avg → 
  A = 117 :=
by
  intros total_before total_after new_avg h
  have h1 : total_after = 34 * A + 362 := by rw [← total_before]
  rw [h1, add_comm] at h
  sorry

end initial_average_jellybeans_per_bag_l221_221893


namespace rectangle_distances_eq_l221_221713

-- Define points and the distance squared function
structure Point where
  x : ℝ
  y : ℝ

def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the points A, B, C, D, and P
variables (a b x y : ℝ)

def A : Point := ⟨-a, -b⟩
def B : Point := ⟨ a, -b⟩
def C : Point := ⟨ a,  b⟩
def D : Point := ⟨-a,  b⟩
def P : Point := ⟨ x,  y⟩

theorem rectangle_distances_eq :
  dist_squared P A + dist_squared P C = dist_squared P B + dist_squared P D :=
by
  sorry

end rectangle_distances_eq_l221_221713


namespace find_a_from_f_neg1_exists_a_odd_function_range_of_a_for_zero_in_domain_l221_221688

/-- Definitions related to the function f(x) -/
def f (a x : ℝ) : ℝ := 1 - a / (2^x + 1)

/-- Problem (1): Given f(-1) == -1, find the value of a -/
theorem find_a_from_f_neg1 (a : ℝ) (h : f a (-1) = -1) : a = 3 := sorry

/-- Problem (2): Determine if there exists a real number a such that f(x) is an odd function -/
theorem exists_a_odd_function : ∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x := 
  ⟨2, by sorry⟩

/-- Problem (3): For the function f(x) to have a zero in its domain, find the range of values for a -/
theorem range_of_a_for_zero_in_domain (a : ℝ) : (∃ x : ℝ, f a x = 0) ↔ 1 < a := sorry

end find_a_from_f_neg1_exists_a_odd_function_range_of_a_for_zero_in_domain_l221_221688


namespace total_weight_of_nuts_l221_221583

theorem total_weight_of_nuts (weight_almonds weight_pecans : ℝ) (h1 : weight_almonds = 0.14) (h2 : weight_pecans = 0.38) : weight_almonds + weight_pecans = 0.52 :=
by
  sorry

end total_weight_of_nuts_l221_221583


namespace hyperbola_equation_l221_221720

-- Condition 1: Asymptotic lines of the hyperbola
def asymptotic_line1 (x y : ℝ) : Prop := 2 * x - 3 * y = 0
def asymptotic_line2 (x y : ℝ) : Prop := 2 * x + 3 * y = 0

-- Condition 2: Hyperbola passes through a specific point
def passes_through_point (C : ℝ → ℝ → Prop) : Prop := C (3 * real.sqrt 2) 2

-- Standard form of the hyperbola that we need to prove
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Proof statement
theorem hyperbola_equation :
  (∀ x y, asymptotic_line1 x y ∨ asymptotic_line2 x y) →
  passes_through_point hyperbola →
  hyperbola (3 * real.sqrt 2) 2 :=
by
  sorry

end hyperbola_equation_l221_221720


namespace correct_calculation_l221_221558

theorem correct_calculation (x y : ℝ) : (x * y^2) ^ 2 = x^2 * y^4 :=
by
  sorry

end correct_calculation_l221_221558


namespace ratio_of_areas_of_octagons_l221_221084

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221084


namespace convex_hexagon_within_convex_polygon_l221_221695

theorem convex_hexagon_within_convex_polygon (P : Type) [convex_polygon P] :
  ∃ H : convex_polygon P, area H ≥ (3 / 4) * area P :=
sorry

end convex_hexagon_within_convex_polygon_l221_221695


namespace prove_tan_570_eq_sqrt_3_over_3_l221_221223

noncomputable def tan_570_eq_sqrt_3_over_3 : Prop :=
  Real.tan (570 * Real.pi / 180) = Real.sqrt 3 / 3

theorem prove_tan_570_eq_sqrt_3_over_3 : tan_570_eq_sqrt_3_over_3 :=
by
  sorry

end prove_tan_570_eq_sqrt_3_over_3_l221_221223


namespace b_total_hire_charges_l221_221022

section car_hire

variables (base_cost : ℝ) (total_hours : ℝ)
variables (fuel_surcharge_rate : ℝ) (peak_hour_surcharge_rate : ℝ)
variables (b_hours : ℝ) (b_peak_hours : ℕ)
variables (base_rate : ℝ)

noncomputable def car_cost_per_hour (base_cost : ℝ) (total_hours : ℝ) : ℝ :=
  base_cost / total_hours

noncomputable def fuel_surcharge (fuel_surcharge_rate : ℝ) (hours : ℝ) : ℝ :=
  fuel_surcharge_rate * hours

noncomputable def peak_hour_surcharge (peak_hour_surcharge_rate : ℝ) (peak_hours : ℕ) : ℝ :=
  peak_hour_surcharge_rate * peak_hours

noncomputable def total_hire_charges (base_rate : ℝ) (hours : ℝ) (fuel_surcharge : ℝ) (peak_hour_surcharge : ℝ) : ℝ :=
  (base_rate * hours) + fuel_surcharge + peak_hour_surcharge

theorem b_total_hire_charges : 
    let total_hours := 32 in
    let base_cost := 720 in
    let fuel_surcharge_rate := 5 in
    let peak_hour_surcharge_rate := 10 in
    let b_hours := 10 in
    let b_peak_hours := 3 in
    let base_rate := car_cost_per_hour base_cost total_hours
    in total_hire_charges base_rate b_hours (fuel_surcharge fuel_surcharge_rate b_hours) (peak_hour_surcharge peak_hour_surcharge_rate b_peak_hours) = 305 :=
by {
  let total_hours := 32,
  let base_cost := 720,
  let fuel_surcharge_rate := 5,
  let peak_hour_surcharge_rate := 10,
  let b_hours := 10,
  let b_peak_hours := 3,
  let base_rate := car_cost_per_hour base_cost total_hours,
  have h1 : fuel_surcharge fuel_surcharge_rate b_hours = 50,
  {
    unfold fuel_surcharge,
    norm_num,
  },
  have h2 : peak_hour_surcharge peak_hour_surcharge_rate b_peak_hours = 30,
  {
    unfold peak_hour_surcharge,
    norm_num,
  },
  have h3 : total_hire_charges base_rate b_hours 50 30 = 305,
  {
    unfold total_hire_charges base_rate b_hours,
    norm_num,
  },
  exact h3
}

end car_hire

end b_total_hire_charges_l221_221022


namespace ring_arrangement_leftmost_digits_l221_221706

theorem ring_arrangement_leftmost_digits :
  let total_ways := (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3) in
  let leftmost_three_digits := (total_ways / 10^4) % 1000 in
  leftmost_three_digits = 126 := 
by
  sorry

end ring_arrangement_leftmost_digits_l221_221706


namespace minimum_shift_symmetric_graph_l221_221472

theorem minimum_shift_symmetric_graph 
  (φ : ℝ) (hφ_pos : φ > 0)
  (h_symmetric : ∀ x : ℝ, 2 * sin (x + π / 3 - φ) = - (2 * sin (x + π / 3 - φ))) :
  φ = π / 3 :=
by
  sorry -- Proof not required

end minimum_shift_symmetric_graph_l221_221472


namespace square_properties_l221_221004

-- Define the side length of the square in cm
def sideLength : ℝ := 30 * Real.sqrt 3

-- Calculate the diagonal and area of the square
def diagonal (s : ℝ) : ℝ := s * Real.sqrt 2
def area (s : ℝ) : ℝ := s ^ 2

-- Prove the correctness of the diagonal and area for the given side length
theorem square_properties (s : ℝ) (hS : s = sideLength) :
  diagonal s = 30 * Real.sqrt 6 ∧ area s = 2700 := 
  by
  sorry

end square_properties_l221_221004


namespace max_integer_solutions_p_eq_k_square_l221_221598

noncomputable def p : ℤ → ℤ -- Declaration of the polynomial p(x)

-- p(x) is a polynomial with integer coefficients
axiom p_int_coeffs (x : ℤ) : ∃ c : list ℤ, ∀ y, (x = y) → (p y = list.sum (list.map (λ i, c.nth i * (y ^ i)) (list.range (c.length))))

-- Condition: p(50) = 50
axiom p_50 : p 50 = 50 

-- Statement to prove: The maximum number of integer solutions k to p(k) = k^2 is at most 7
theorem max_integer_solutions_p_eq_k_square (h : ∀ k : ℤ, p k = k^2 → (k = 43 ∨ k = 57 ∨ k = 45 ∨ k = 55 ∨ k = 49 ∨ k = 51 ∨ k = 50)) : 
  set.finite {k : ℤ | p k = k^2} ∧ 7 ≤ (set.to_finset {k : ℤ | p k = k^2}).card := 
by
  sorry

end max_integer_solutions_p_eq_k_square_l221_221598


namespace binomial_expansion_zero_sum_of_terms_l221_221550

theorem binomial_expansion_zero_sum_of_terms (n b c : ℤ) (h₀ : n ≥ 2) (h₁ : c ≠ 0) :
  let a := b + c in
  let second_term := n * 2^(n-1) * b^(n-1) * c^(n+1) in
  let third_term := (n * (n-1) / 2) * 2^(n-2) * b^(n-2) * c^(n+2) in
  second_term + third_term = 0 ↔ n = (4 * b / c) + 1 := sorry

end binomial_expansion_zero_sum_of_terms_l221_221550


namespace ratio_of_areas_of_octagons_l221_221064

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221064


namespace solve_inequality_l221_221231

theorem solve_inequality (x: ℝ) : (25 - 5 * Real.sqrt 3) ≤ x ∧ x ≤ (25 + 5 * Real.sqrt 3) ↔ x ^ 2 - 50 * x + 575 ≤ 25 :=
by
  sorry

end solve_inequality_l221_221231


namespace base_seven_to_ten_l221_221531

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end base_seven_to_ten_l221_221531


namespace tenth_term_is_26_l221_221023

-- Definitions used from the conditions
def first_term : ℤ := 8
def common_difference : ℤ := 2
def term_number : ℕ := 10

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Proving that the 10th term is 26 given the conditions
theorem tenth_term_is_26 : nth_term first_term common_difference term_number = 26 := by
  sorry

end tenth_term_is_26_l221_221023


namespace chinese_remainder_example_l221_221546

theorem chinese_remainder_example :
  ∃ b : ℕ, b % 3 = 2 ∧ b % 4 = 3 ∧ b % 5 = 4 ∧ b % 7 = 6 ∧ b = 419 := 
by
  use 419
  split; 
  { norm_num }
  split; 
  { norm_num }
  split; 
  { norm_num }
  split; 
  { norm_num }
  sorry

end chinese_remainder_example_l221_221546


namespace distance_from_center_of_circle_to_line_l221_221328

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221328


namespace augustin_tower_height_l221_221627

/-!
  Given Augustin has six \(1 \times 2 \times \pi\) bricks that can be oriented to contribute a height of either 1 or 2 units each,
  prove that the number of distinct heights of the tower is 7.
-/

def num_distinct_heights (bricks : ℕ) (heights : List ℕ) : ℕ :=
  (List.range (bricks * heights.maximum)).filter (λ h, (0 : Fin bricks.succ).val = heights.sum).length

theorem augustin_tower_height :
  num_distinct_heights 6 [1, 2] = 7 :=
sorry

end augustin_tower_height_l221_221627


namespace order_of_four_l221_221689

theorem order_of_four {m n p q : ℝ} (hmn : m < n) (hpq : p < q) (h1 : (p - m) * (p - n) < 0) (h2 : (q - m) * (q - n) < 0) : m < p ∧ p < q ∧ q < n :=
by
  sorry

end order_of_four_l221_221689


namespace intersection_M_N_l221_221444

noncomputable def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℕ := {0, 1, 2}
def MN_intersection : Set ℕ := {0, 1}

theorem intersection_M_N : M ∩ N = MN_intersection :=
by
  sorry

end intersection_M_N_l221_221444


namespace correct_square_root_calculation_l221_221914

theorem correct_square_root_calculation:
  (∀ (x : ℝ), x² = 25 → x = 5 ∨ x = -5) ∧
  (¬(√(5^2) = ±5)) ∧
  (¬(√((-5)^2) = -5)) ∧
  (∀ (y : ℝ), y² = 25 → y = 5) → ∃! correct_statement, 
  correct_statement = "C"
:= 
begin
  sorry
end

end correct_square_root_calculation_l221_221914


namespace total_number_of_values_l221_221513

theorem total_number_of_values (S n : ℕ) (h1 : (S - 165 + 135) / n = 150) (h2 : S / n = 151) : n = 30 :=
by {
  sorry
}

end total_number_of_values_l221_221513


namespace stratified_sampling_junior_teachers_l221_221604

theorem stratified_sampling_junior_teachers 
    (total_teachers : ℕ) (senior_teachers : ℕ) 
    (intermediate_teachers : ℕ) (junior_teachers : ℕ) 
    (sample_size : ℕ) 
    (H1 : total_teachers = 200)
    (H2 : senior_teachers = 20)
    (H3 : intermediate_teachers = 100)
    (H4 : junior_teachers = 80) 
    (H5 : sample_size = 50)
    : (junior_teachers * sample_size / total_teachers = 20) := 
  by 
    sorry

end stratified_sampling_junior_teachers_l221_221604


namespace greg_age_is_16_l221_221186

-- Define the ages of Cindy, Jan, Marcia, and Greg based on the conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem statement: Prove that Greg's age is 16
theorem greg_age_is_16 : greg_age = 16 :=
by
  -- Proof would go here
  sorry

end greg_age_is_16_l221_221186


namespace ferry_speed_difference_l221_221232

theorem ferry_speed_difference :
  let V_p := 6
  let Time_P := 3
  let Distance_P := V_p * Time_P
  let Distance_Q := 2 * Distance_P
  let Time_Q := Time_P + 1
  let V_q := Distance_Q / Time_Q
  V_q - V_p = 3 := by
  sorry

end ferry_speed_difference_l221_221232


namespace sum_f_eq_2016_l221_221678

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12)

theorem sum_f_eq_2016 :
  (∑ k in finset.range 2016, f ((k + 1) / 2017)) = 2016 :=
by
  -- To be proven
  sorry

end sum_f_eq_2016_l221_221678


namespace symmetric_point_distance_l221_221781

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem symmetric_point_distance :
  let p1 : ℝ × ℝ := (2, -3)
  let p2 : ℝ × ℝ := (2, 3)
  distance p1.1 p1.2 p2.1 p2.2 = 6 :=
begin
  -- proof
  sorry
end

end symmetric_point_distance_l221_221781


namespace solve_quadratic_l221_221478

theorem solve_quadratic : ∃ x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 ∧ x = 5/3 := 
by
  sorry

end solve_quadratic_l221_221478


namespace distance_from_center_to_line_l221_221362

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221362


namespace sets_equal_l221_221423

noncomputable section

open Finset

variable {α : Type*} [Field α] [IsAlgorithmProposition α]

-- Defining two sets of complex numbers, A and B
def A (n : ℕ) := {a | a ∈ (range n).map (λ i, complex.abs i)}
def B (n : ℕ) := {b | b ∈ (range n).map (λ i, complex.exp i)}

-- Given condition that sums of symmetric polynomials over A and B are equal
def condition (A B : Set α) (k n : ℕ) : Prop := 
  ∑ (i in range n) ∑ (j in range i) (A.elem i + A.elem j)^k
  = ∑ (i in range n) ∑ (j in range i) (B.elem i + B.elem j)^k

-- Theorem stating that if the condition holds for all k, then A = B
theorem sets_equal {n : ℕ} (A B : Set α) :
  (∀ k, 1 ≤ k → k ≤ n → condition A B k n) → A = B := by
  sorry

end sets_equal_l221_221423


namespace area_ratio_is_correct_l221_221140

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221140


namespace find_f_2023_l221_221244

theorem find_f_2023 :
  ∃ f : ℕ → ℕ, (∀ m n : ℕ, f(n + f(m)) = f(n) + m + 1) ∧ (∀ k l : ℕ, k < l → f(k) < f(l)) ∧ f 2023 = 2024 :=
by
  sorry

end find_f_2023_l221_221244


namespace projections_locus_is_circle_l221_221243

-- Given a right circular cone with apex \( S \), base plane π and a point \( A \) outside the cone,
-- where the distance from \( A \) to the plane π equals the height of the cone.
-- Point \( M \) lies on the surface of the cone. 
-- A light ray emanating from \( A \) to \( M \) reflects off the surface of the cone and becomes parallel to plane π.
-- Prove that the locus of the projections of points \( M \) onto the plane of the base of the cone forms a circle.

noncomputable def cone_base_projections_locus (A S : ℝ^3) (height : ℝ) (π : set (ℝ^3)) 
  (hAπ : dist A π = height) (hcone : ∀ M : ℝ^3, M ∈ cone_surface A S → is_parallel (line_through_reflection A M) π) :
  set (ℝ^3) :=
{M_proj | let M := ℝ^3 in 
  M_proj = projection M π ∧ M ∈ cone_surface A S ∧ is_parallel (line_through_reflection A M) π}

theorem projections_locus_is_circle (A S : ℝ^3) (height : ℝ) (π : set (ℝ^3))
  (hAπ : dist A π = height) (hcone : ∀ M : ℝ^3, M ∈ cone_surface A S → is_parallel (line_through_reflection A M) π) :
  ∃ (circle : set (ℝ^3)), cone_base_projections_locus A S height π hAπ hcone = circle :=
sorry

end projections_locus_is_circle_l221_221243


namespace sum_of_first_n_terms_b_l221_221736

-- Definitions from the conditions
def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ := (S n + 1) / 2

def b : ℕ → ℕ
| 0     := 0  -- No b_0 specified, set to 0 to make definitions work
| (n+1) := if n = 0 then 3 else a n + b n

-- Statement to prove
theorem sum_of_first_n_terms_b (n : ℕ) : 
  (∑ k in Finset.range n, b (k + 1)) = 2 * n + 2^n - 1 :=
by sorry

end sum_of_first_n_terms_b_l221_221736


namespace area_TAB_l221_221789

-- Define the curves in rectangular and polar coordinate systems.
noncomputable def curve_C1_parametric (α : ℝ) : ℝ × ℝ :=
  (5 * Real.cos α, 5 + 5 * Real.sin α)

noncomputable def curve_C1_polar (θ ρ : ℝ) : Prop :=
  ρ = 10 * Real.sin θ

noncomputable def curve_C2_polar (θ ρ : ℝ) : Prop :=
  ρ = 10 * Real.cos θ

-- Define the fixed point and the intersection points.
def T : ℝ × ℝ := (4, 0)
def A : ℝ × ℝ := (5 * Real.sqrt 3, π / 3)
def B : ℝ × ℝ := (5, π / 3)

-- Define the function to calculate the area of a triangle in polar coordinates.
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (ρ1, θ1) := p1 in
  let (ρ2, θ2) := p2 in
  let (ρ3, θ3) := p3 in
  (1 / 2) * ρ1 * ρ2 * Real.sin (θ2 - θ1) + 
  (1 / 2) * ρ2 * ρ3 * Real.sin (θ3 - θ2) + 
  (1 / 2) * ρ3 * ρ1 * Real.sin (θ1 - θ3)

-- The area of ΔTAB.
theorem area_TAB :
  triangle_area T A B = 15 - 5 * Real.sqrt 3 :=
sorry

end area_TAB_l221_221789


namespace maximum_marks_l221_221955

-- Definitions based on the conditions
def passing_percentage : ℝ := 0.5
def student_marks : ℝ := 200
def marks_to_pass : ℝ := student_marks + 20

-- Lean 4 statement for the proof problem
theorem maximum_marks (M : ℝ) 
  (h1 : marks_to_pass = 220)
  (h2 : passing_percentage * M = marks_to_pass) :
  M = 440 :=
sorry

end maximum_marks_l221_221955


namespace find_modulus_of_z_find_m_range_l221_221240

section part_one

variable (b : ℝ)
variable (z : ℂ) (hz : z = b * complex.I) (hb : b ≠ 0)
variable (hreal : (z-2)/(1+complex.I) ∈ ℝ)

theorem find_modulus_of_z : |z| = 2 :=
sorry -- proof intentionally omitted

end part_one

section part_two

variable (m : ℝ)
variable (w : ℂ) (hz : w = (m + -2 * complex.I)^2)

theorem find_m_range (h_fourth_quad : w.re > 0 ∧ w.im < 0) : m > 2 :=
sorry -- proof intentionally omitted

end part_two

end find_modulus_of_z_find_m_range_l221_221240


namespace fill_tank_time_l221_221611

-- Definitions based on provided conditions
def pipeA_time := 60 -- Pipe A fills the tank in 60 minutes
def pipeB_time := 40 -- Pipe B fills the tank in 40 minutes

-- Theorem statement
theorem fill_tank_time (T : ℕ) : 
  (T / 2) / pipeB_time + (T / 2) * (1 / pipeA_time + 1 / pipeB_time) = 1 → 
  T = 48 :=
by
  intro h
  sorry

end fill_tank_time_l221_221611


namespace no_counterexample_l221_221984

theorem no_counterexample (n : ℕ) (h : n = 29 ∨ n = 37 ∨ n = 41 ∨ n = 58) :
  ¬(nat.prime n ∧ ¬nat.prime (n + 2)) :=
by 
  cases h;
  simp [nat.prime];
  sorry

end no_counterexample_l221_221984


namespace product_of_solutions_l221_221884

theorem product_of_solutions:
  (∀ x: ℝ, (x + 2) / (2*x + 2) = (4*x + 3) / (7*x + 3) → x = 0 ∨ x = 3):
  ∏ x in ({0, 3} : finset ℝ), x = 0 := 
by
  sorry

end product_of_solutions_l221_221884


namespace correct_propositions_l221_221188

def proposition_1 (line : Type) (plane : Type) [has_perpendicular line plane] : Prop :=
  ∀ (l : line) (p : plane), (∀ l₁ l₂ : line, l₁ ≠ l₂ → (l ⊥ l₁ ∧ l ⊥ l₂)) → (l ⊥ p)

def proposition_2 (line : Type) (plane : Type) [has_perpendicular line plane] : Prop :=
  ∀ (l : line) (p : plane), (∀ l' : line, (l' ≠ l → l' ∈ p → l ⊥ l') → (l ⊥ p))

def proposition_3 (line : Type) (plane : Type) [has_parallel line plane] : Prop :=
  ∀ (l m : line) (a : plane), (l ∥ a ∧ m ∥ a) → (l ∥ m)

def proposition_4 (line : Type) [has_parallel line line] [has_perpendicular line line] : Prop :=
  ∀ (a b l : line), (a ∥ b ∧ l ⊥ a) → (l ⊥ b)

theorem correct_propositions (line : Type) (plane : Type) [has_perpendicular line plane] [has_perpendicular line line] [has_parallel line plane] [has_parallel line line]
  : (proposition_2 line plane) ∧ (proposition_4 line) ∧ ¬(proposition_1 line plane) ∧ ¬(proposition_3 line plane) := by sorry

end correct_propositions_l221_221188


namespace parameter_tC_slope_angle_l221_221954

noncomputable theory

-- Definitions for the parametric equations of line l
def x (t : Real) (x0 : Real) (theta : Real) := x0 + t * (Real.cos theta)
def y (t : Real) (theta : Real) := t * (Real.sin theta)

-- Definition for the ellipse given
def ellipse (x y : Real) : Prop := (x^2 / 3) + y^2 = 1

-- Coordinates for left focus of ellipse
def F1 := (-Real.sqrt 2, 0 : Real)

-- Line l passes through F1 and intersects the positive y-axis at point C
def on_line (t x0 theta : Real) : Prop :=
  (x t x0 theta, y t theta) = (0, y t theta)

def point_C (x0 theta : Real) : Real :=
  (Real.sqrt 2) / (Real.cos theta)

-- The parametric equations of line l intersect the ellipse at points A and B
def intersections (x y : Real → Real → Real → Real) (F1x F1y theta : Real) :=
  ∃ tA tB : Real, ellipse (x tA F1x theta) (y tA theta) ∧ ellipse (x tB F1x theta) (y tB theta)

-- First proof: Proving the parameter t_C corresponding to point C
theorem parameter_tC (theta : Real) :
  (Real.sqrt 2) / (Real.cos theta) = point_C (-Real.sqrt 2) theta :=
sorry

-- Second proof: Proving the slope angle theta given the condition
theorem slope_angle (theta : Real) (tA tB : Real) :
  | (x tB (-Real.sqrt 2) theta - fst F1) | = | y tA theta | → theta = Real.pi / 6 :=
sorry

end parameter_tC_slope_angle_l221_221954


namespace hyperbola_with_foci_on_y_axis_l221_221551

variable (α : ℝ) {x y : ℝ}

theorem hyperbola_with_foci_on_y_axis (h : α ∈ (real.pi / 2, 3 * real.pi / 4)) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (x^2*sin α - y^2*cos α = 1) → (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (∀ (a b : ℝ), a > b) → 
  (y - axis contains foci) := 
sorry

end hyperbola_with_foci_on_y_axis_l221_221551


namespace number_being_divided_l221_221455

theorem number_being_divided (divisor quotient remainder number : ℕ) 
  (h_divisor : divisor = 3) 
  (h_quotient : quotient = 7) 
  (h_remainder : remainder = 1)
  (h_number : number = divisor * quotient + remainder) : 
  number = 22 :=
by
  rw [h_divisor, h_quotient, h_remainder] at h_number
  exact h_number

end number_being_divided_l221_221455


namespace mode_median_A_B_avg_comparison_l221_221045

variable (A_ages : List ℕ) (B_ages : List ℕ)
variable (mode_A : ℕ)
variable (median_B : ℕ)
variable (avg_A : ℕ)
variable (avg_B : ℕ)

-- Conditions: 
-- A_ages contains the ages of 10 employees from department A.
-- B_ages contains the ages of 10 employees from department B.

axiom mode_condition : mode_A = 32
axiom median_condition : median_B = 26
axiom avg_A_condition : avg_A = 27.1
axiom avg_B_condition : avg_B = 27.5

-- 1. Prove the mode and median:
theorem mode_median_A_B (hA : mode_A = 32) (hB : median_B = 26) : mode_A = 32 ∧ median_B = 26 :=
by
  exact ⟨hA, hB⟩

-- 2. Prove the average comparison between the two departments:
theorem avg_comparison (hA : avg_A = 27.1) (hB : avg_B = 27.5) : avg_A < avg_B :=
by
  rw [hA, hB]
  exact lt_trans (by norm_num) (by norm_num)

#check mode_median_A_B
#check avg_comparison


end mode_median_A_B_avg_comparison_l221_221045


namespace wrapping_paper_per_present_l221_221848

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l221_221848


namespace mitya_age_l221_221826

theorem mitya_age {M S: ℕ} (h1 : M = S + 11) (h2 : S = 2 * (S - (M - S))) : M = 33 :=
by
  -- proof steps skipped
  sorry

end mitya_age_l221_221826


namespace find_f_2023_l221_221245

theorem find_f_2023 :
  ∃ f : ℕ → ℕ, (∀ m n : ℕ, f(n + f(m)) = f(n) + m + 1) ∧ (∀ k l : ℕ, k < l → f(k) < f(l)) ∧ f 2023 = 2024 :=
by
  sorry

end find_f_2023_l221_221245


namespace other_root_is_five_l221_221227

theorem other_root_is_five (m : ℝ) 
  (h : -1 is_root_m x^2 - 4 * x + m = 0) : 
  is_root x^2 - 4 * x + m = 0 5 := 
sorry

end other_root_is_five_l221_221227


namespace other_root_of_quadratic_l221_221230

theorem other_root_of_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + m = 0 → x = -1) → (∀ y : ℝ, y^2 - 4 * y + m = 0 → y = 5) :=
sorry

end other_root_of_quadratic_l221_221230


namespace Hoelder_l221_221838

variable (A B p q : ℝ)

theorem Hoelder (hA : 0 < A) (hB : 0 < B) (hp : 0 < p) (hq : 0 < q) (h : 1 / p + 1 / q = 1) : 
  A^(1/p) * B^(1/q) ≤ A / p + B / q := 
sorry

end Hoelder_l221_221838


namespace circle_construction_l221_221740

open EuclideanGeometry

theorem circle_construction
  (a b : Line)
  (P : Point)
  (h_inter : intersecting a b)
  (h_P_on_b : on_line P b) :
  ∃ centers : set Point, (centers ⊆ (line b)) ∧
  ((size centers = 2 ∧ ∀ c ∈ centers, ∃ r : ℝ, Circle c r P ∈ intersect_set (line a)) ∨ 
   (size centers = 1 ∧ ∃ c ∈ centers, ∃ r : ℝ, Circle c r P ∈ intersect_set (line a)) ∨
   (size centers = ∞ ∧ ∀ c ∈ centers, ∃ r : ℝ, Circle c r P ∈ intersect_set (line a))) :=
sorry

end circle_construction_l221_221740


namespace g_50_eq_36_l221_221987

noncomputable def g : ℕ → ℕ
| x := if (∃ (n : ℕ), 3^n = x) then if (h : ∃ (n : ℕ), 3^n = x) then Nat.log 3 x (Classical.some_spec h) else 0
       else 2 + g (x + 2)

theorem g_50_eq_36 : g 50 = 36 :=
sorry

end g_50_eq_36_l221_221987


namespace circle_center_line_distance_l221_221338

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221338


namespace area_ratio_PQRSTU_to_ABCDEF_l221_221950

-- Definitions based on conditions
variables (A B C D E F P Q R S : Point) (s : ℝ)
variables (hABCDEF : RegularHexagon A B C D E F) 
variables (hPQ : onLine P A B) (hQR : onLine Q B C) 
variables (hRS : onLine R D E) (hSP : onLine S E F)
variables (hParallelAP : Parallel (Line A P) (Line C D))
variables (hParallelBQ : Parallel (Line B Q) (Line C D))
variables (hParallelCR : Parallel (Line C R) (Line F A))
variables (hParallelDS : Parallel (Line D S) (Line F A))
variables (hParallelER : Parallel (Line E R) (Line C D))
variables (hParallelFS : Parallel (Line F S) (Line C D))
variables (hDistance : Distance (PerpendicularLine C D F A) = (1/3) * s)

-- Theorem based on the question and the conditions, proving the area ratio
theorem area_ratio_PQRSTU_to_ABCDEF : 
  Area (Hexagon P Q R S T U) / Area (Hexagon A B C D E F) = 4 / 27 := 
sorry

end area_ratio_PQRSTU_to_ABCDEF_l221_221950


namespace range_of_m_l221_221808

noncomputable def abs_sum (x : ℝ) : ℝ := |x - 5| + |x - 3|

theorem range_of_m (m : ℝ) : (∃ x : ℝ, abs_sum x < m) ↔ m > 2 := 
by 
  sorry

end range_of_m_l221_221808


namespace smallest_positive_int_satisfies_congruence_l221_221549

theorem smallest_positive_int_satisfies_congruence :
  ∃ x : ℕ, 0 < x ∧ x < 35 ∧ (6 * x ≡ 17 [MOD 35]) := 
begin
  use 32,
  split,
  { -- 0 < 32
    exact nat.zero_lt_succ 31,
  },
  split,
  { -- 32 < 35
    exact nat.lt_succ_self 32,
  },
  { -- 6 * 32 ≡ 17 [MOD 35]
    sorry,
  }
end

end smallest_positive_int_satisfies_congruence_l221_221549


namespace circle_distance_to_line_l221_221376

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221376


namespace sqrt_of_imaginary_is_imaginary_l221_221461

open Complex

theorem sqrt_of_imaginary_is_imaginary (b : ℝ) (hb : b ≠ 0) : ∃ z : ℝ, sqrt (0 + b * I) = z * I := by
  sorry

end sqrt_of_imaginary_is_imaginary_l221_221461


namespace sum_mobile_phone_keypad_l221_221764

/-- The numbers on a standard mobile phone keypad are 0 through 9. -/
def mobile_phone_keypad : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- The sum of all the numbers on a standard mobile phone keypad is 45. -/
theorem sum_mobile_phone_keypad : mobile_phone_keypad.sum = 45 := by
  sorry

end sum_mobile_phone_keypad_l221_221764


namespace center_of_circumcircle_DEF_l221_221248

-- Assume we have an acute triangle ABC
variables (A B C : Type*) [EuclideanGeometry.triangle A B C]
-- Assume circles centered at A and C passing through B intersect again at F
variable (F : Type*) (circleA : EuclideanGeometry.circle A B) (circleC : EuclideanGeometry.circle C B) 
variable (intersection_circle : EuclideanGeometry.point_of_intersection A B C F)
-- Assume these circles intersect the circumcircle of triangle ABC at points D and E
variable (circumcircle_ABC : EuclideanGeometry.circumcircle A B C)
variables (D E : Type*) [EuclideanGeometry.point_on_circle A B C D E]
-- Assume segment BF intersects the circumcircle at point O
variable (O : Type*) [EuclideanGeometry.intersects_segment_circle B F A B C ]

theorem center_of_circumcircle_DEF :
  EuclideanGeometry.is_center_circumcircle O D E F :=
sorry

end center_of_circumcircle_DEF_l221_221248


namespace evaluate_expression_l221_221202

theorem evaluate_expression (x z : ℤ) (h1 : x = 2) (h2 : z = 1) : z * (z - 4 * x) = -7 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l221_221202


namespace transaction_volume_scientific_notation_l221_221916

-- Defining the given conditions
def transaction_volume_billion := 2684
def one_billion := 10^9

noncomputable def transaction_volume := transaction_volume_billion * one_billion

theorem transaction_volume_scientific_notation :
  transaction_volume = 2.684 * 10^11 :=
by
  -- proof to be completed
  sorry

end transaction_volume_scientific_notation_l221_221916


namespace circle_center_line_distance_l221_221339

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221339


namespace bathtub_capacity_l221_221418

-- Define the conditions
def tablespoons_per_pound := 1.5
def pounds_per_gallon := 8
def gallons_per_cubic_foot := 7.5
def cost_per_tablespoon := 0.50
def total_cost := 270.0

-- Define the question
def cubic_feet_of_water (total_cost : ℝ) 
(c tablespoons_per_pound : ℝ) 
(pounds_per_gallon : ℝ) 
(gallons_per_cubic_foot : ℝ) 
(cost_per_tablespoon : ℝ): ℝ :=
(total_cost / cost_per_tablespoon) / tablespoons_per_pound / pounds_per_gallon / gallons_per_cubic_foot

-- Prove that the bathtub can hold 6 cubic feet of water
theorem bathtub_capacity : cubic_feet_of_water total_cost tablespoons_per_pound pounds_per_gallon gallons_per_cubic_foot cost_per_tablespoon = 6 :=
  sorry

end bathtub_capacity_l221_221418


namespace depth_of_well_l221_221953

theorem depth_of_well
  (d : ℝ)
  (h1 : ∃ t1 t2 : ℝ, 18 * t1^2 = d ∧ t2 = d / 1150 ∧ t1 + t2 = 8) :
  d = 33.18 :=
sorry

end depth_of_well_l221_221953


namespace mowing_time_l221_221200

/-- Define the lawn dimensions -/
def lawn_length : ℝ := 100
def lawn_width : ℝ := 60

/-- Define the swath width and overlap in inches and convert to feet -/
def swath_width_inches : ℝ := 30
def overlap_inches : ℝ := 6
def effective_swath_width_feet : ℝ := (swath_width_inches - overlap_inches) / 12

/-- Define the walking speed in feet per hour -/
def walking_speed : ℝ := 4000

/-- Prove the time it takes to mow the lawn is 0.75 hours -/
theorem mowing_time : (lawn_length / effective_swath_width_feet * lawn_width) / walking_speed = 0.75 := by
  sorry

end mowing_time_l221_221200


namespace octagon_area_ratio_l221_221155

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221155


namespace largest_lucky_number_sum_of_n_given_difference_l221_221759

def is_lucky_number (M : ℕ) : Prop :=
  ∃ m : ℕ, M = m * (m + 3) ∧ M ≥ 100 ∧ M ≤ 999

theorem largest_lucky_number : ∃ M, is_lucky_number M ∧ ∀ M', is_lucky_number M' → M' ≤ 990 :=
sorry

theorem sum_of_n_given_difference (M N : ℕ) (hM : is_lucky_number M) (hN : is_lucky_number N) 
  (hDiff : M - N = 350) : ∑ N' in { N | is_lucky_number N ∧ ∃ M, is_lucky_number M ∧ M - N = 350 }, N' = 614 :=
sorry

end largest_lucky_number_sum_of_n_given_difference_l221_221759


namespace find_triples_l221_221660

theorem find_triples (x p n : ℕ) (hp : Nat.Prime p) :
  2 * x * (x + 5) = p^n + 3 * (x - 1) →
  (x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1) :=
by
  sorry

end find_triples_l221_221660


namespace tangent_length_external_tangent_length_internal_l221_221504

noncomputable def tangent_length_ext (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R + r) / R)

noncomputable def tangent_length_int (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R - r) / R)

theorem tangent_length_external (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_ext R r a h hAB :=
sorry

theorem tangent_length_internal (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_int R r a h hAB :=
sorry

end tangent_length_external_tangent_length_internal_l221_221504


namespace circle_distance_condition_l221_221342

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221342


namespace ratio_of_areas_of_octagons_l221_221082

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221082


namespace total_points_scored_l221_221956

theorem total_points_scored :
  ∀ (members : Fin 8 → Option ℕ), 
  let scoring_members := [members 0, members 1, members 2, members 3, members 4] in
  (members 0 = some 4) →
  (members 1 = some 6) →
  (members 2 = some 8) →
  (members 3 = some 8) →
  (members 4 = none) →
  (members 5 = none) →
  (members 6 = none) →
  (members 7 = none) →
  (scoring_members.filterMap id).sum = 26 := 
by
  intros members scoring_members h0 h1 h2 h3 h4 h5 h6 h7
  sorry

end total_points_scored_l221_221956


namespace line_through_M_chord_length_conds_l221_221495

theorem line_through_M_chord_length_conds (M : ℝ × ℝ) (hM : M = (0, 4))
    (C : ℝ × ℝ) (hC : C = (1, 0)) (r : ℝ) (hr : r = 2)
    (hChord : ∃ (p q : ℝ × ℝ), p = q ∧ dist p q = 2 * Real.sqrt 3
      ∧ ∀ x y, ((x - 1)^2 + y^2 = 4) ∧ (p, q ∈ (λ (M : ℝ × ℝ), ((x - 1)^2 + y^2 = 4))) ) :
    (∃ k : ℝ, k * 15 + 8 * k - 32 = 0) ∨ (∀ x y, x = 0) :=
sorry

end line_through_M_chord_length_conds_l221_221495


namespace problem_solution_l221_221802

structure Point : Type where
  x : ℝ
  y : ℝ

def rotate (p : Point) (center : Point) (angle : ℝ) : Point :=
  let (x, y) := (p.x - center.x, p.y - center.y)
  let rad := angle * (Real.pi / 180)
  let x' := x * Real.cos rad - y * Real.sin rad
  let y' := x * Real.sin rad + y * Real.cos rad
  ⟨x' + center.x, y' + center.y⟩

def reflect_y_eq_x (p : Point) : Point :=
  ⟨p.y, p.x⟩

def reflect_y_eq_neg_x (p : Point) : Point :=
  ⟨-p.y, -p.x⟩

def transform (p : Point) (center : Point) (trans : List (ℝ ⊕ (Point → Point))) : Point :=
  trans.foldl (λ acc t => match t with
    | Sum.inl angle => rotate acc center angle
    | Sum.inr reflect => reflect acc) p

def is_original_position (triangle : List Point) (center : Point) (trans : List (ℝ ⊕ (Point → Point))) : Prop :=
  let original_triangle := [⟨1, 1⟩, ⟨5, 1⟩, ⟨1, 4⟩]
  let transformed_triangle := triangle.map (λ p => transform p center trans)
  transformed_triangle = original_triangle

noncomputable def count_valid_sequences : ℕ :=
  let rotations := [60, 120, 240].map Sum.inl
  let reflections := [Sum.inr reflect_y_eq_x, Sum.inr reflect_y_eq_neg_x]
  let all_trans := rotations ++ reflections
  let sequences := List.product (List.product all_trans all_trans) all_trans
  sequences.filter (λ seq =>
    let (t1, (t2, t3)) := seq
    is_original_position [⟨1, 1⟩, ⟨5, 1⟩, ⟨1, 4⟩] ⟨1, 1⟩ [t1, t2, t3]).length

theorem problem_solution : count_valid_sequences = 12 := sorry

end problem_solution_l221_221802


namespace prob_win_3_1_correct_l221_221028

-- Defining the probability for winning a game
def prob_win_game : ℚ := 2 / 3

-- Defining the probability for losing a game
def prob_lose_game : ℚ := 1 - prob_win_game

-- A function to calculate the probability of winning the match with a 3:1 score
def prob_win_3_1 : ℚ :=
  let combinations := 3 -- Number of ways to lose exactly 1 game in the first 3 games (C_3^1)
  let win_prob := prob_win_game ^ 3 -- Probability for winning 3 games
  let lose_prob := prob_lose_game -- Probability for losing 1 game
  combinations * win_prob * lose_prob

-- The theorem that states the probability that player A wins with a score of 3:1
theorem prob_win_3_1_correct : prob_win_3_1 = 8 / 27 := by
  sorry

end prob_win_3_1_correct_l221_221028


namespace increasing_intervals_f_range_g_l221_221730

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * (sin x)^2
def g (x : ℝ) : ℝ := 2 * sin (2 * (x + π / 12)) - 1

theorem increasing_intervals_f (k : ℤ) :
  ∀ x ∈ set.Icc (-π / 6 + k * π) (π / 3 + k * π), 
     0 < derivative (λ x, f x) x :=
sorry

theorem range_g : 
  ∀ x ∈ set.Icc (-π / 6) (π / 3), 
     - sqrt 3 ≤ g x ∧ g x ≤ sqrt 3 :=
sorry

end increasing_intervals_f_range_g_l221_221730


namespace math_solution_l221_221205

noncomputable def math_problem : Prop :=
  let g (x : ℝ) : ℝ := (31:ℝ).sqrt + 56 / x in
  let B : ℝ := 
    let r1 := (31:ℝ).sqrt + (255:ℝ).sqrt / 2
    let r2 := (31:ℝ).sqrt - (255:ℝ).sqrt / 2 in
    r1 + r2 in
  B * B = 255

theorem math_solution : math_problem := by
  sorry

end math_solution_l221_221205


namespace surface_area_increase_by_44_percent_l221_221552

-- Define the variables and conditions
variables (s : ℝ)

-- The original surface area of the cube
def original_surface_area (s : ℝ) : ℝ := 6 * s^2

-- The new surface area of the cube after a 20% increase in edge length
def new_surface_area (s : ℝ) : ℝ := 6 * (1.2 * s)^2

-- The increase in surface area
def increase_in_surface_area (s : ℝ) : ℝ := new_surface_area s - original_surface_area s

-- The percentage increase in surface area
def percentage_increase (s : ℝ) : ℝ := (increase_in_surface_area s / original_surface_area s) * 100

-- The theorem to prove
theorem surface_area_increase_by_44_percent : percentage_increase s = 44 := by
  sorry

end surface_area_increase_by_44_percent_l221_221552


namespace points_concyclic_l221_221927

variables {A B C D M_a M_c N_a N_c P_a P_b P_c P_d : Point}

-- Assume \(ABCD\) is a cyclic quadrilateral
axiom cyclic_quadrilateral : cyclic {A, B, C, D}

-- Define the orthocenters for triangles within the quadrilateral
axiom orthocenter_ABC : orthocenter A B C = M_a
axiom orthocenter_BCD : orthocenter B C D = M_c
axiom orthocenter_ABD : orthocenter A B D = N_a
axiom orthocenter_ACD : orthocenter A C D = N_c

-- Define points \(P_a\), \(P_b\), \(P_c\), and \(P_d\) using orthocenters
-- The precise definition of these points depends on the orthocenters related to diagonals \(AC\) and \(BD\).
axiom points_defined : 
  from_orthocenters_and_diagonals (M_a, M_c, N_a, N_c) = (P_a, P_b, P_c, P_d)

-- Prove the points \(P_a\), \(P_b\), \(P_c\), and \(P_d\) are concyclic
theorem points_concyclic : 
  cyclic {P_a, P_b, P_c, P_d} :=
sorry

end points_concyclic_l221_221927


namespace total_supervisors_l221_221877

theorem total_supervisors (buses : ℕ) (supervisors_per_bus : ℕ) (h1 : buses = 7) (h2 : supervisors_per_bus = 3) :
  buses * supervisors_per_bus = 21 :=
by
  sorry

end total_supervisors_l221_221877


namespace ratio_of_areas_of_octagons_l221_221061

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221061


namespace arithmetic_sequence_floor_property_l221_221801

variable {a : ℕ → ℕ} {d : ℕ}
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a0 : 1 ≤ a 0 ∧ a 0 ≤ d)

noncomputable def c : ℝ := (1 + Real.sqrt (1 + 4 / d)) / 2

theorem arithmetic_sequence_floor_property :
    ∃ c, (c = (1 + Real.sqrt (1 + 4 / d)) / 2) ∧ ∀ n, b n = Int.floor (c * ↑(a n)) :=
by
  sorry 

end arithmetic_sequence_floor_property_l221_221801


namespace leftmost_three_nonzero_digits_of_ring_arrangements_l221_221708

noncomputable def num_ring_arrangements : ℕ := 
  Nat.binomial 10 6 * Nat.factorial 6 * Nat.binomial 9 3

def leftmost_three_nonzero_digits (n : ℕ) : ℕ := 
  (n / 10^(Nat.log10 n - 2))

theorem leftmost_three_nonzero_digits_of_ring_arrangements : 
  leftmost_three_nonzero_digits num_ring_arrangements = 126 :=
  by sorry

end leftmost_three_nonzero_digits_of_ring_arrangements_l221_221708


namespace max_area_l221_221409

-- Definitions
def triangle := Type
variables {A B C : triangle}
variable {a b : ℝ}
noncomputable def area (a b : ℝ) : ℝ := (a^2 + b^2 - 1) / 4

-- Given conditions
axiom c_eq_one : (1 : ℝ) = 1
axiom area_condition (a b : ℝ) : area a b = (a^2 + b^2 - 1) / 4

-- Theorem stating the maximum area
theorem max_area (a b : ℝ) : area a b ≤ (real.sqrt 2 + 1) / 4 :=
sorry

end max_area_l221_221409


namespace base7_to_base10_of_43210_l221_221538

theorem base7_to_base10_of_43210 : 
  base7_to_base10 (list.num_from_digits [4, 3, 2, 1, 0]) 7 = 10738 :=
by
  def base7_to_base10 (digits : list ℕ) (base : ℕ) : ℕ :=
    digits.reverse.join_with base
  
  show base7_to_base10 [4, 3, 2, 1, 0] 7 = 10738
  sorry

end base7_to_base10_of_43210_l221_221538


namespace functions_are_same_l221_221622

-- Define the functions
def f (x : ℝ) : ℝ := Real.sqrt (x^2)
def g (t : ℝ) : ℝ := abs t

-- Prove the functions are the same
theorem functions_are_same : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_same_l221_221622


namespace distance_from_center_to_line_l221_221366

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221366


namespace log_simplification_l221_221996

theorem log_simplification : log 5 (2 / sqrt 25) = log 5 2 - 1 := 
by
  sorry

end log_simplification_l221_221996


namespace octagon_area_ratio_l221_221132

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221132


namespace max_regions_divided_by_lines_max_regions_divided_by_circles_l221_221906

-- Statement 1: Maximum number of regions divided by n lines
theorem max_regions_divided_by_lines (n : ℕ) : 
  greatest_number_of_regions n = (n^2 + n + 2) / 2 := sorry

-- Statement 2: Maximum number of regions divided by n circles
theorem max_regions_divided_by_circles (n : ℕ) :
  greatest_number_of_regions n = n^2 - n + 2 := sorry

end max_regions_divided_by_lines_max_regions_divided_by_circles_l221_221906


namespace sum_of_cosines_l221_221631

-- Assuming degrees are converted to radians for trigonometric functions in Lean
def deg_to_rad (d : ℝ) : ℝ := d * Real.pi / 180

theorem sum_of_cosines :
  Real.cos (deg_to_rad 5) + Real.cos (deg_to_rad 77) + 
  Real.cos (deg_to_rad 149) + Real.cos (deg_to_rad 221) + 
  Real.cos (deg_to_rad 293) = 0 := 
by sorry

end sum_of_cosines_l221_221631


namespace problem_1_problem_2_l221_221443

section Problem1

variable (x a : ℝ)

-- Proposition p
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 1
theorem problem_1 : p 1 x ∧ q x → 2 < x ∧ x < 3 :=
by { sorry }

end Problem1

section Problem2

variable (a : ℝ)

-- Proposition p with a as a variable
def p_a (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0

-- Proposition q with x as a variable
def q_x (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Problem 2
theorem problem_2 : (∀ (x : ℝ), ¬p_a a x → ¬q_x x) → (1 < a ∧ a ≤ 2) :=
by { sorry }

end Problem2

end problem_1_problem_2_l221_221443


namespace stuck_at_A_or_B_l221_221903

def EulerianPath (G : Type) [Graph G] (A B : Vertex G) : Prop :=
  (d_A : odd_deg A) ∧ (d_B : odd_deg B) ∧ ∀ v ≠ A ≠ B, even_deg v

def PetrovGetsStuck (G : Type) [Graph G] (n : ℕ) (X : Vertex G) : Prop :=
  ∃ k ∈ {1, ..., n}, (∃ (e ∈ Edges G), ¬consumed_tickets e X k) 

theorem stuck_at_A_or_B (G : Type) [Graph G] (A B X : Vertex G) (n : ℕ) :
  is_connected G →
  EulerianPath G A B →
  PetrovGetsStuck G n X →
  X = A ∨ X = B :=
by
  sorry

end stuck_at_A_or_B_l221_221903


namespace mitya_age_l221_221827

theorem mitya_age {M S: ℕ} (h1 : M = S + 11) (h2 : S = 2 * (S - (M - S))) : M = 33 :=
by
  -- proof steps skipped
  sorry

end mitya_age_l221_221827


namespace octagon_area_ratio_l221_221150

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221150


namespace exists_three_points_l221_221696

theorem exists_three_points (n : ℕ) (h : 3 ≤ n) (points : Fin n → EuclideanSpace ℝ (Fin 2))
  (distinct : ∀ i j : Fin n, i ≠ j → points i ≠ points j) :
  ∃ (A B C : Fin n),
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    1 ≤ dist (points A) (points B) / dist (points A) (points C) ∧ 
    dist (points A) (points B) / dist (points A) (points C) < (n + 1) / (n - 1) := 
sorry

end exists_three_points_l221_221696


namespace find_omega_l221_221263

noncomputable def f (x : ℝ) (ω φ : ℝ) := Real.sin (ω * x + φ)

theorem find_omega (ω φ : ℝ) (hω : ω > 0) (hφ : 0 ≤ φ ∧ φ ≤ π)
  (h_even : ∀ x : ℝ, f x ω φ = f (-x) ω φ)
  (h_symm : ∀ x : ℝ, f (3 * π / 4 + x) ω φ = f (3 * π / 4 - x) ω φ)
  (h_mono : ∀ x1 x2 : ℝ, 0 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 2 → f x1 ω φ ≤ f x2 ω φ) :
  ω = 2 / 3 ∨ ω = 2 :=
sorry

end find_omega_l221_221263


namespace greg_age_is_16_l221_221184

-- Definitions based on given conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem stating that Greg's age is 16 years given the above conditions
theorem greg_age_is_16 : greg_age = 16 := by
  sorry

end greg_age_is_16_l221_221184


namespace train_length_l221_221166

theorem train_length (time : ℕ) (speed_kmh : ℕ) (conversion_factor : ℚ) (speed_ms : ℚ) (length : ℚ) :
  time = 50 ∧ speed_kmh = 36 ∧ conversion_factor = 5 / 18 ∧ speed_ms = speed_kmh * conversion_factor ∧ length = speed_ms * time →
  length = 500 :=
by
  sorry

end train_length_l221_221166


namespace base7_to_base10_of_43210_l221_221537

theorem base7_to_base10_of_43210 : 
  base7_to_base10 (list.num_from_digits [4, 3, 2, 1, 0]) 7 = 10738 :=
by
  def base7_to_base10 (digits : list ℕ) (base : ℕ) : ℕ :=
    digits.reverse.join_with base
  
  show base7_to_base10 [4, 3, 2, 1, 0] 7 = 10738
  sorry

end base7_to_base10_of_43210_l221_221537


namespace intersect_single_point_l221_221596

theorem intersect_single_point 
  (A B C D P Q M N : Type*) 
  [geometry A] [geometry B] [geometry C] [geometry D]
  [geometry P] [geometry Q] [geometry M] [geometry N]
  (trapezoid_non_isosceles : is_non_isosceles_trapezoid A B C D)
  (circle : circle_passes_through A B)
  (circle_intersects_legs : circle_intersects AD BC P Q)
  (circle_intersects_diagonals : circle_intersects AC BD M N)
  (parallels_AB_CD : parallel AB CD) :
  ∃ R, intersects PQ MN CD R :=
begin
  sorry
end

end intersect_single_point_l221_221596


namespace parabola_directrix_l221_221494

theorem parabola_directrix (x y : ℝ) :
    x^2 = - (1 / 4) * y → y = - (1 / 16) :=
by
  sorry

end parabola_directrix_l221_221494


namespace triangle_angles_l221_221043

theorem triangle_angles (A B C D : Point)
  (h_triangle : right_triangle A B C)
  (h_circle : diameter_circle B C)
  (h_intersect : intersects D A C)
  (h_ratio : segment_ratio A D D C 1 3) :
  angle A B C = 90 ∧ angle B A C = 30 ∧ angle B C A = 60 := 
sorry

end triangle_angles_l221_221043


namespace identify_geometric_shapes_l221_221628

-- Define the geometric shape based on the first problem description
def is_right_pentagonal_prism : Prop :=
  ∃ shape : Type, (∃ (faces : List shape) 
    (pentagon_face : shape) 
    (rectangle_face : shape), 
    faces.length = 7 ∧
    (∃ (f1 f2 : shape), f1 = pentagon_face ∧ f2 = pentagon_face ∧ Parallel f1 f2) ∧
    (∀ f ∈ faces.drop(2), f = rectangle_face ∧ Congruent f rectangle_face))

-- Define the geometric shape based on the second problem description
def is_cone : Prop :=
  ∃ (triangle : Type) 
    (height_line : Line) 
    (base_length : ℝ),
    (is_isosceles ∧
    rotated_shape_by_180 triangle height_line = cone)

-- Theorem stating that given the conditions, the shapes are as described
theorem identify_geometric_shapes :
  (is_right_pentagonal_prism → ∃ shape, shape = "Right Pentagonal Prism") ∧
  (is_cone → ∃ shape, shape = "Cone") :=
by
  sorry

end identify_geometric_shapes_l221_221628


namespace value_of_a_plus_b_l221_221297

theorem value_of_a_plus_b (a b : ℕ) (h1 : Real.sqrt 44 = 2 * Real.sqrt a) (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l221_221297


namespace ratio_of_areas_of_octagons_l221_221091

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221091


namespace relay_race_orders_l221_221196

open Finset

def athletes : finset (ℕ × ℕ × ℕ × ℕ) :=
  univ.filter (λ ⟨a, b, c, d⟩, a ≠ 1 ∧ b ≠ 2 ∧ c ≠ 3)

theorem relay_race_orders : athletes.card = 11 := by
  sorry

end relay_race_orders_l221_221196


namespace greg_age_is_16_l221_221187

-- Define the ages of Cindy, Jan, Marcia, and Greg based on the conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem statement: Prove that Greg's age is 16
theorem greg_age_is_16 : greg_age = 16 :=
by
  -- Proof would go here
  sorry

end greg_age_is_16_l221_221187


namespace sequence_general_formula_l221_221700

theorem sequence_general_formula (a : ℕ → ℝ) 
  (h1 : a 1 = 3) 
  (h3 : a 3 = 243) 
  (h_rec : ∀ n ≥ 2, log a (n + 1) + log a (n - 1) = 2 * log a n) : 
  ∀ n, a n = 3^(2 * n - 1) :=
sorry

end sequence_general_formula_l221_221700


namespace megacorp_fine_l221_221825

noncomputable def daily_mining_profit := 3000000
noncomputable def daily_oil_profit := 5000000
noncomputable def monthly_expenses := 30000000

def total_daily_profit := daily_mining_profit + daily_oil_profit
def annual_profit := 365 * total_daily_profit
def annual_expenses := 12 * monthly_expenses
def net_annual_profit := annual_profit - annual_expenses
def fine := (net_annual_profit * 1) / 100

theorem megacorp_fine : fine = 25600000 := by
  -- the proof steps will go here
  sorry

end megacorp_fine_l221_221825


namespace profit_percentage_is_4_l221_221019

-- Define the cost price and selling price
def cost_price : Nat := 600
def selling_price : Nat := 624

-- Calculate profit in dollars
def profit_dollars : Nat := selling_price - cost_price

-- Calculate profit percentage
def profit_percentage : Nat := (profit_dollars * 100) / cost_price

-- Prove that the profit percentage is 4%
theorem profit_percentage_is_4 : profit_percentage = 4 := by
  sorry

end profit_percentage_is_4_l221_221019


namespace number_of_good_numbers_l221_221440

def floor_div (x : ℝ) (n : ℕ) : ℤ := ⌊ x / n! ⌋

def f (x : ℝ) : ℤ :=
  ∑ k in Finset.range 2013, floor_div x (k + 1)

def is_good_number (n : ℤ) : Prop :=
  ∃ x : ℝ, f x = n

def good_numbers : Finset ℤ :=
  (Finset.range 1007).map ⟨λ k, 2 * k + 1, sorry⟩ -- Every odd number in the set.

theorem number_of_good_numbers : good_numbers.filter is_good_number = 587 := sorry

end number_of_good_numbers_l221_221440


namespace circle_distance_condition_l221_221343

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221343


namespace main_problem_l221_221721

-- Define the ellipse conditions
def isEllipse (C : Set (ℝ × ℝ)) :=
  (∀ x y, (x, y) ∈ C ↔ (x^2 / 4 + y^2 = 1))

-- Define the line conditions passing through the right focus
def lineThroughFocus (A B : ℝ × ℝ) :=
  (A.2 = A.1 - sqrt 3) ∧ (B.2 = B.1 - sqrt 3)

-- The center is at origin, foci on the x-axis, major axis length 4, and one point on the ellipse
def ellipseConditions (C : Set (ℝ × ℝ)) :=
  ∃ (a : ℝ) (b : ℝ), a = 2 ∧ 0 < b ∧
  (∀ x y, (x, y) ∈ C ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  (1, sqrt 3 / 2) ∈ C

-- The length of the segment AB when intersecting the ellipse and line
def segmentLength (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) :=
  ∀ (x1 x2 : ℝ × ℝ), l x1 ∧ l x2 →
  (by A = (sqrt 3, 0)) ∧ A.1 + B.1 = 8 * sqrt 3 / 5 ∧ A.1 * B.1 = 8 / 5 →
  ((x1.1 - x2.1)^2 + (x1.2 - x2.2)^2) = (8 / 5)^2

-- Main theorem to be proven
theorem main_problem :
  ∃ (C : Set (ℝ × ℝ)) (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop),
  ellipseConditions C ∧
  isEllipse C ∧
  lineThroughFocus A B ∧
  segmentLength A B l :=
sorry

end main_problem_l221_221721


namespace time_to_fill_tank_l221_221614

-- Definitions of conditions
def fill_by_pipe (time rate : ℚ) : ℚ := time * rate

def pipe_A_rate : ℚ := 1 / 60
def pipe_B_rate : ℚ := 1 / 40
def combined_rate : ℚ := pipe_A_rate + pipe_B_rate

-- Question to be proved
theorem time_to_fill_tank : 
    ∃ T : ℚ, (fill_by_pipe (T / 2) pipe_B_rate + fill_by_pipe (T / 2) combined_rate = 1) ∧ T = 30 :=
by 
    use 30
    sorry

end time_to_fill_tank_l221_221614


namespace candidate_marks_approx_45_l221_221938

theorem candidate_marks_approx_45 :
  ∃ (x : ℝ), x + 25 = 0.55 * 127.27 ∧ x ≈ 45 :=
by
  -- Stating the conditions
  have h1 : 0.55 * 127.27 ≈ 70 := sorry
  have h2 : (x : ℝ) ≈ 45 := sorry
  -- The actual problem
  use x,
  split,
  { sorry },
  { sorry }

end candidate_marks_approx_45_l221_221938


namespace arithmetic_fifth_term_l221_221381

theorem arithmetic_fifth_term (a d : ℝ) (h : (a + d) + (a + 3d) = 10) : a + 4d = 5 :=
by
  sorry

end arithmetic_fifth_term_l221_221381


namespace find_point_C_l221_221835

noncomputable def point_on_segment {A B C : ℝ × ℝ} (segment_AB : ℝ) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ C = (A.1 + k * (B.1 - A.1), A.2 + k * (B.2 - A.2))

theorem find_point_C :
  let A := (-3, -2)
  let B := (5, 10)
  let C := (3.4, 7.6)
  (2 : ℝ) * real.dist A C = real.dist A B ∧ point_on_segment (real.dist A B) :=
sorry

end find_point_C_l221_221835


namespace keiko_speed_calc_l221_221421

noncomputable def keiko_speed (r : ℝ) (time_diff : ℝ) : ℝ :=
  let circumference_diff := 2 * Real.pi * 8
  circumference_diff / time_diff

theorem keiko_speed_calc (r : ℝ) (time_diff : ℝ) :
  keiko_speed r 48 = Real.pi / 3 := by
  sorry

end keiko_speed_calc_l221_221421


namespace Q_not_invertible_l221_221429

-- Define the vector v
def v : ℝ × ℝ := (4, -5)

-- Define the unit vector u in the direction of v
def v_length : ℝ := Real.sqrt (4 ^ 2 + (-5) ^ 2)
def u : ℝ × ℝ := (4 / v_length, -5 / v_length)

-- Define the projection matrix Q
def Q : Matrix (Fin 2) (Fin 2) ℝ :=
  let uuT := Matrix.mulVecLin u (Matrix.vecLin u) in
  (1 / (41 : ℝ)) • uuT

-- Determine if Q is invertible
theorem Q_not_invertible : det Q = 0 :=
  sorry

end Q_not_invertible_l221_221429


namespace area_ratio_of_octagons_is_4_l221_221095

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221095


namespace find_missing_number_l221_221502

-- Define the given set of numbers with a missing element
def numbers : List ℕ := [6, 3, 1, 8, 5, 9, 2]
-- Define the median value
def median : ℕ := 8

-- The proposition statement
theorem find_missing_number (missing_number : ℕ) :
  List.median (missing_number :: numbers) = median → missing_number = 7 := 
by
  sorry

end find_missing_number_l221_221502


namespace exists_sphere_intersecting_tetrahedron_sphere_radius_5_intersects_tetrahedron_l221_221021

-- Definitions of geometrical objects and properties
structure Circle (r : ℝ) :=
  (radius : ℝ)
  (pos : 0 < r)

structure Sphere (r : ℝ) :=
  (radius : ℝ)
  (pos : 0 < r)

structure RegularTetrahedron :=
  (faces : ℕ)

-- Conditions for circles intersecting a tetrahedron
def intersects_faces (s : Sphere) (t : RegularTetrahedron) (radii : List ℝ) : Prop :=
  List.length radii = 4 ∧
  ∀ r, r ∈ radii → ∃ c : Circle r, r = c.radius

-- Theorem statements for both parts of the problem
theorem exists_sphere_intersecting_tetrahedron :
  ∃ s : Sphere, ∃ t : RegularTetrahedron, intersects_faces s t [1, 2, 3, 4] :=
sorry

theorem sphere_radius_5_intersects_tetrahedron :
  ∃ t : RegularTetrahedron, intersects_faces (Sphere.mk 5 (by norm_num)) t [1, 2, 3, 4] :=
sorry

end exists_sphere_intersecting_tetrahedron_sphere_radius_5_intersects_tetrahedron_l221_221021


namespace john_average_speed_l221_221412

noncomputable def time_uphill : ℝ := 45 / 60 -- 45 minutes converted to hours
noncomputable def distance_uphill : ℝ := 2   -- 2 km

noncomputable def time_downhill : ℝ := 15 / 60 -- 15 minutes converted to hours
noncomputable def distance_downhill : ℝ := 2   -- 2 km

noncomputable def total_distance : ℝ := distance_uphill + distance_downhill
noncomputable def total_time : ℝ := time_uphill + time_downhill

theorem john_average_speed : total_distance / total_time = 4 :=
by
  have h1 : total_distance = 4 := by sorry
  have h2 : total_time = 1 := by sorry
  rw [h1, h2]
  norm_num

end john_average_speed_l221_221412


namespace ratio_area_octagons_correct_l221_221110

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221110


namespace wrapping_paper_per_present_l221_221849

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end wrapping_paper_per_present_l221_221849


namespace points_at_distance_sqrt2_l221_221213

noncomputable def point_on_line {t : ℝ} : ℝ × ℝ := (3 - t, 4 + t)

def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem points_at_distance_sqrt2 (t : ℝ) :
  (distance (point_on_line t) (3, 4) = real.sqrt 2) ↔ 
  (point_on_line t = (4, 3) ∨ point_on_line t = (2, 5)) :=
sorry

end points_at_distance_sqrt2_l221_221213


namespace correct_option_C_l221_221641

variable (f : ℝ → ℝ) (h : ∀ x, deriv f x > 1 - f x)

theorem correct_option_C : exp 2 * f 2 + exp 2 > exp 1 * f 1 + exp 1 := 
sorry

end correct_option_C_l221_221641


namespace sequence_1_formula_sequence_2_formula_sequence_3_formula_l221_221018

theorem sequence_1_formula (n : ℕ) (hn : n > 0) : 
  (∃ a : ℕ → ℚ, (a 1 = 1/2) ∧ (a 2 = 1/6) ∧ (a 3 = 1/12) ∧ (a 4 = 1/20) ∧ (∀ n, a n = 1/(n*(n+1)))) :=
by
  sorry

theorem sequence_2_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℕ, (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (∀ n, a n = 2^(n-1))) :=
by
  sorry

theorem sequence_3_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℚ, (a 1 = 4/5) ∧ (a 2 = 1/2) ∧ (a 3 = 4/11) ∧ (a 4 = 2/7) ∧ (∀ n, a n = 4/(3*n + 2))) :=
by
  sorry

end sequence_1_formula_sequence_2_formula_sequence_3_formula_l221_221018


namespace distance_from_center_to_line_l221_221367

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221367


namespace common_area_of_triangles_is_25_l221_221523

-- Define basic properties and conditions of an isosceles right triangle with hypotenuse = 10 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = 10^2
def is_isosceles_right_triangle (a b : ℝ) : Prop := a = b ∧ hypotenuse a b

-- Definitions representing the triangls
noncomputable def triangle1 := ∃ a b : ℝ, is_isosceles_right_triangle a b
noncomputable def triangle2 := ∃ a b : ℝ, is_isosceles_right_triangle a b

-- The area common to both triangles is the focus
theorem common_area_of_triangles_is_25 : 
  triangle1 ∧ triangle2 → 
  ∃ area : ℝ, area = 25 
  := 
sorry

end common_area_of_triangles_is_25_l221_221523


namespace range_of_a_l221_221262

theorem range_of_a {A : Set ℝ} (h1: ∀ x ∈ A, 2 * x + a > 0) (h2: 1 ∉ A) (h3: 2 ∈ A) : -4 < a ∧ a ≤ -2 := 
sorry

end range_of_a_l221_221262


namespace distance_to_line_is_constant_l221_221350

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221350


namespace quadratic_function_expression_l221_221762

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_function_expression 
  (a b c : ℝ)
  (h1 : quadratic_function a b c (-2) = 0)
  (h2 : quadratic_function a b c 4 = 0)
  (h3 : ∃ x_max : ℝ, x_max = 1 ∧ quadratic_function a b c x_max = 9) :
  quadratic_function a b c = λ x, -x^2 + 2 * x + 8 :=
by {
  sorry
}

end quadratic_function_expression_l221_221762


namespace largest_three_digit_n_l221_221003

theorem largest_three_digit_n (n : ℤ) : 
  (100 ≤ n ∧ n < 1000) ∧ (40 * n ≡ 140 [MOD 320]) → n ≤ 995 := 
begin
  sorry
end

end largest_three_digit_n_l221_221003


namespace ratio_of_areas_of_octagons_l221_221089

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221089


namespace blue_number_multiple_of_3_l221_221832

theorem blue_number_multiple_of_3 :
  (∀ (circle : Type) (chord : circle → circle → Prop) (assign_value : circle → ℤ)
  (marked_segments : set (circle × circle × ℤ × ℤ)),
    (∃! (y : ℕ), (∀ p ∈ marked_segments, p.2.1 + p.2.2 = y) ∧ (∀ i, ∃ (p ∈ marked_segments), 
      (p.2.1 + p.2.2 = i)) ∧ (∀ (p : circle × circle × ℤ × ℤ), 
    (p ∈ marked_segments) → |p.2.1 - p.2.2| ∈ (set.range (λ x, 3 * x))) → false) :=
sorry

end blue_number_multiple_of_3_l221_221832


namespace number_of_boys_selected_l221_221891

theorem number_of_boys_selected {boys girls selections : ℕ} 
  (h_boys : boys = 11) (h_girls : girls = 10) (h_selections : selections = 6600) : 
  ∃ (k : ℕ), k = 2 :=
sorry

end number_of_boys_selected_l221_221891


namespace odd_function_value_at_1_l221_221807

-- Definition of the odd function f
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x ^ 2 - x - a else -f (-x) a

theorem odd_function_value_at_1 (a : ℝ) :
  (∀ x : ℝ, f x a = 2 * x ^ 2 - x - a ∨ f x a = -f (-x) a) →
  (f 0 a = 0) →
  (-a = 0) →
  f 1 a = -3 := by
sorry

end odd_function_value_at_1_l221_221807


namespace probability_of_no_obtuse_triangle_l221_221897

def angle_between_points (A B O : Point) : Angle := sorry -- Define the angle concept
def is_obtuse (A B O : Point) : Prop := sorry -- Define the obtuse condition based on angle

noncomputable def probability_no_obtuse_triangle: ℝ := sorry -- Calculate the probability (given in the problem)

theorem probability_of_no_obtuse_triangle (A B C O : Point) (hA : is_on_circle A O) (hB : is_on_circle B O) (hC : is_on_circle C O):
  probability_no_obtuse_triangle = 3 / 16 :=
sorry

end probability_of_no_obtuse_triangle_l221_221897


namespace minimally_intersecting_remainder_l221_221193

theorem minimally_intersecting_remainder :
  let S := {1, 2, 3, 4, 5, 6, 7}
  let P (A B C : Finset ℕ) :=
    A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ |A ∩ B| = 1 ∧ |B ∩ C| = 1 ∧ |C ∩ A| = 1 ∧ A ∩ B ∩ C = ∅
  let M := (Finset.powerset S).card
  M % 1000 = 760 := sorry

end minimally_intersecting_remainder_l221_221193


namespace correct_number_of_propositions_l221_221271

def is_certain_event : Prop := 
  ∀ (b1 b2 : ℕ), (b1 + b2 = 3) → (b1 > 1 ∨ b2 > 1)

def is_impossible_event : Prop :=
  ∀ (x : ℝ), x^2 ≥ 0

def is_random_event_Anshun : Prop :=
  true -- Placeholder since prediction of weather cannot be framed as a mathematical proof

def is_random_event_bulbs : Prop := 
  true -- Placeholder since probability by nature is not deterministic in Lean proofs

def count_correct_propositions : ℕ :=
  let p1 := is_certain_event in
  let p2 := is_impossible_event in
  let p3 := ¬is_random_event_Anshun in
  let p4 := is_random_event_bulbs in
  (if p1 then 1 else 0) +
  (if p2 then 1 else 0) +
  (if p3 then 1 else 0) +
  (if p4 then 1 else 0)

theorem correct_number_of_propositions : count_correct_propositions = 3 := by
  sorry

end correct_number_of_propositions_l221_221271


namespace sum_coordinates_l221_221834

variables (x y : ℝ)
def A_coord := (9, 3)
def M_coord := (3, 7)

def midpoint_condition_x : Prop := (x + 9) / 2 = 3
def midpoint_condition_y : Prop := (y + 3) / 2 = 7

theorem sum_coordinates (h1 : midpoint_condition_x x) (h2 : midpoint_condition_y y) : 
  x + y = 8 :=
by 
  sorry

end sum_coordinates_l221_221834


namespace distance_between_foci_of_rectangular_hyperbola_l221_221873

theorem distance_between_foci_of_rectangular_hyperbola (c : ℝ) (h : c = 4) : 
  let foci_distance := 2 * real.sqrt (2 * c) in
  foci_distance = 4 :=
by
  sorry

end distance_between_foci_of_rectangular_hyperbola_l221_221873


namespace correct_calculation_C_l221_221012

theorem correct_calculation_C (a b y x : ℝ) : 
  (7 * a + a ≠ 8 * a^2) ∧ 
  (5 * y - 3 * y ≠ 2) ∧ 
  (3 * x^2 * y - 2 * x^2 * y = x^2 * y) ∧ 
  (3 * a + 2 * b ≠ 5 * a * b) :=
by {
  split,
  { exact sorry, },
  split,
  { exact sorry, },
  split,
  { exact sorry, },
  { exact sorry, },
}

end correct_calculation_C_l221_221012


namespace partition_equation_solution_l221_221963

def partition (n : ℕ) : ℕ := sorry -- defining the partition function

theorem partition_equation_solution (n : ℕ) (h : partition n + partition (n + 4) = partition (n + 2) + partition (n + 3)) :
  n = 1 ∨ n = 3 ∨ n = 5 :=
sorry

end partition_equation_solution_l221_221963


namespace sightseeing_tour_arrangements_l221_221582

theorem sightseeing_tour_arrangements :
  let spots := ["Music Town", "Cangjie Town", "Rose Town", "Flower Expo Park", "Cangjie Park"]
  let adjacent (a b : String) (l : List String) := ∃ l₁ l₂ l₃, l = l₁ ++ [a, b] ++ l₂ ∨ l = l₁ ++ [b, a] ++ l₂
  ∀ (arrangement : List String),
    arrangement.perm spots ∧ -- the tour includes all the spots
    ¬ (arrangement.head = "Rose Town" ∨ arrangement.last = "Rose Town") ∧ -- Rose Town is not at beginning or end
    adjacent "Music Town" "Flower Expo Park" arrangement -- Music Town and Flower Expo Park are adjacent
  → arrangement.length = 5
  → ∃ (n : ℕ), n = 24 := 
sorry

end sightseeing_tour_arrangements_l221_221582


namespace find_common_ratio_and_difference_l221_221620

theorem find_common_ratio_and_difference (q d : ℤ) 
  (h1 : q^3 = 1 + 7 * d) 
  (h2 : 1 + q + q^2 + q^3 = 1 + 7 * d + 21) : 
  (q = 4 ∧ d = 9) ∨ (q = -5 ∧ d = -18) :=
by
  sorry

end find_common_ratio_and_difference_l221_221620


namespace number_of_multiples_of_15_l221_221746

theorem number_of_multiples_of_15 (a b multiple : ℕ) :
  a = 16 → b = 181 → multiple = 15 → (∃ (n : ℕ), 30 = multiple * n) → (∃ (m : ℕ), 180 = multiple * m) →
  ∃ (count : ℕ), count = 11 :=
by
  intros ha hb hmultiple hfirst hlast
  use 11
  sorry

end number_of_multiples_of_15_l221_221746


namespace distance_to_line_is_constant_l221_221353

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221353


namespace ratio_of_octagon_areas_l221_221074

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221074


namespace probability_sum_5_l221_221944

/-- A cube toy has 6 faces marked with the numbers 1, 2, 2, 3, 3, 3. The toy is thrown twice in succession. -/
def faces : List ℕ := [1, 2, 2, 3, 3, 3]

/-- There are 36 possible outcomes when the toy is thrown twice. Let A be the event that the sum of the numbers on the top faces is 5.
    We want to prove that the probability of A is 1/3. -/
theorem probability_sum_5 (h : ∀ x ∈ faces, ∀ y ∈ faces, x + y = 5 → (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2)) :
  (∃ outcomes : Finset (ℕ × ℕ), (∀ x ∈ faces, ∀ y ∈ faces, (x, y) ∈ outcomes ↔ x + y = 5) ∧ outcomes.card = 12) →
  (36 : ℝ).recip * 12 = (1 / 3 : ℝ) :=
by
  sorry

end probability_sum_5_l221_221944


namespace find_B_function_range_l221_221803

-- Define the triangle sides and angles
variables (a b c : ℝ) (A B C : ℝ)
-- Define the conditions
def triangle_conditions (h₁ : a + c = 1 + sqrt 3) 
                        (h₂ : b = 1)
                        (h₃ : sin C = sqrt 3 * sin A) : Prop := 
  true

-- Define the function
def f (x B : ℝ) : ℝ := 2 * sin (2 * x + B) + 4 * cos x ^ 2

-- Angle B calculation
theorem find_B (a c b A C : ℝ) (h₁ : a + c = 1 + sqrt 3) 
                           (h₂ : b = 1) 
                           (h₃ : sin C = sqrt 3 * sin A) : 
                           B = π / 6 := 
sorry

-- Range of the function
theorem function_range (B : ℝ) (h₄ : B = π / 6) : 
  set.range (λ x, f x B) = set.Icc (-4 : ℝ) (2 * sqrt 3 + 2) := 
sorry

end find_B_function_range_l221_221803


namespace largest_share_received_l221_221854

theorem largest_share_received (total_profit : ℝ) (ratios : List ℝ) (h_ratios : ratios = [1, 2, 2, 3, 4, 5]) 
  (h_profit : total_profit = 51000) : 
  let parts := ratios.sum 
  let part_value := total_profit / parts
  let largest_share := 5 * part_value 
  largest_share = 15000 := 
by 
  sorry

end largest_share_received_l221_221854


namespace distance_to_line_is_constant_l221_221357

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221357


namespace HerbertAgeNextYear_l221_221285

variable (Kris_age : ℕ) (Herbert_age : ℕ)

axiom KrisAgeIsTwentyFour : Kris_age = 24
axiom HerbertIsYounger : Herbert_age = Kris_age - 10

theorem HerbertAgeNextYear : Herbert_age + 1 = 15 := by
  rw [HerbertIsYounger, KrisAgeIsTwentyFour]
  reduce
  exact rfl


end HerbertAgeNextYear_l221_221285


namespace photograph_goal_reach_l221_221454

-- Define the initial number of photographs
def initial_photos : ℕ := 250

-- Define the percentage splits initially
def beth_pct_init : ℝ := 0.40
def my_pct_init : ℝ := 0.35
def julia_pct_init : ℝ := 0.25

-- Define the photographs taken initially by each person
def beth_photos_init : ℕ := 100
def my_photos_init : ℕ := 88
def julia_photos_init : ℕ := 63

-- Confirm initial photographs sum
example (h : beth_photos_init + my_photos_init + julia_photos_init = 251) : true := 
by trivial

-- Define today's decreased productivity percentages
def beth_decrease_pct : ℝ := 0.35
def my_decrease_pct : ℝ := 0.45
def julia_decrease_pct : ℝ := 0.25

-- Define the photographs taken today by each person after decreases
def beth_photos_today : ℕ := 65
def my_photos_today : ℕ := 48
def julia_photos_today : ℕ := 47

-- Sum of photographs taken today
def total_photos_today : ℕ := 160

-- Define the initial plus today's needed photographs to reach goal
def goal_photos : ℕ := 650

-- Define the additional number of photographs needed
def additional_photos_needed : ℕ := 399 - total_photos_today

-- Final proof statement
theorem photograph_goal_reach : 
  (beth_photos_init + my_photos_init + julia_photos_init) + (beth_photos_today + my_photos_today + julia_photos_today) + additional_photos_needed = goal_photos := 
by sorry

end photograph_goal_reach_l221_221454


namespace max_parts_three_planes_divide_space_l221_221291

-- Define the conditions given in the problem.
-- Condition 1: A plane divides the space into two parts.
def plane_divides_space (n : ℕ) : ℕ := 2

-- Condition 2: Two planes can divide the space into either three or four parts.
def two_planes_divide_space (n : ℕ) : ℕ := if n = 2 then 3 else 4

-- Condition 3: Three planes can divide the space into four, six, seven, or eight parts.
def three_planes_divide_space (n : ℕ) : ℕ := if n = 4 then 8 else sorry

-- The statement to be proved.
theorem max_parts_three_planes_divide_space : 
  ∃ n, three_planes_divide_space n = 8 := by
  use 4
  sorry

end max_parts_three_planes_divide_space_l221_221291


namespace circle_distance_to_line_l221_221371

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221371


namespace five_fourths_of_frac_and_mult_three_equals_nine_l221_221662

theorem five_fourths_of_frac_and_mult_three_equals_nine :
  let x := (5 / (4 : ℝ)) * (12 / (5 : ℝ)) in
  (x * 3) = 9 :=
by
  sorry

end five_fourths_of_frac_and_mult_three_equals_nine_l221_221662


namespace locus_of_centers_of_touching_circles_l221_221044

theorem locus_of_centers_of_touching_circles
  (radius : ℝ)
  (trihedron : {a : ℝ × ℝ × ℝ // a.1 > 0 ∧ a.2 > 0 ∧ a.3 > 0})
  (touches_faces : radius = 1)
  (center : ℝ × ℝ × ℝ) :
  (center.1^2 + center.2^2 + center.3^2 = 2) ∧
  (|center.1| ≤ 1) ∧
  (|center.2| ≤ 1) ∧
  (|center.3| ≤ 1) :=
sorry

end locus_of_centers_of_touching_circles_l221_221044


namespace total_students_l221_221519

-- Definitions based on conditions
variable (T M Z : ℕ)  -- T for Tina's students, M for Maura's students, Z for Zack's students

-- Conditions as hypotheses
axiom h1 : T = M  -- Tina's classroom has the same amount of students as Maura's
axiom h2 : Z = (T + M) / 2  -- Zack's classroom has half the amount of total students between Tina and Maura's classrooms
axiom h3 : Z = 23  -- There are 23 students in Zack's class when present

-- Proof statement
theorem total_students : T + M + Z = 69 :=
  sorry

end total_students_l221_221519


namespace divisibility_by_n_divisibility_by_n_plus_1_l221_221680

noncomputable def sum_of_squares (n : ℕ) : ℕ := ∑ k in Finset.range (n+1), ∑ i in Finset.range (k+1), i^2

theorem divisibility_by_n (n : ℕ) : ¬ (n ∣ sum_of_squares n) ↔ (n % 3 = 0 ∨ n % 4 = 0) :=
  sorry

theorem divisibility_by_n_plus_1 (n : ℕ) : ¬ ((n + 1) ∣ sum_of_squares n) ↔ ∃ k : ℕ, n = 4 * k + 1 :=
  sorry

end divisibility_by_n_divisibility_by_n_plus_1_l221_221680


namespace greatest_common_factor_90_126_180_l221_221544

noncomputable def gcd_three_numbers : ℕ :=
  gcd (gcd 90 126) 180

theorem greatest_common_factor_90_126_180 : gcd_three_numbers = 18 := by
  have h1 : 180 = 90 * 126 / (90 * 126 / 180) := sorry -- Given condition
  have h2 : ∀ (a b : ℕ), gcd a b = gcd b a := gcd.comm
  have h3 : ∀ (a b c : ℕ), gcd (gcd a b) c = gcd (gcd b a) c := by intro _ _ _ ; rw [← gcd.assoc]
  have h4 : ∀ (a b : ℕ), gcd (a * b) c = gcd a (gcd b c) := gcd.mul_right
  unfold gcd_three_numbers
  rw [h2, h3]
  sorry -- Further steps to complete the proof calculation

end greatest_common_factor_90_126_180_l221_221544


namespace product_of_possible_N_l221_221967

variable {A D N : ℤ}

theorem product_of_possible_N :
  let D := A + N
  let D_6 := D - 7
  let A_6 := A + 4
  let cond := abs (D_6 - A_6) = 1
  (cond → (N = 12 ∨ N = 10)) → 12 * 10 = 120 :=
by
  intro D_6 A_6 cond h
  have h₁ : N = 12 ∨ N = 10 := h cond
  cases h₁ with N12 N10
  · exact rfl
  · exact rfl

end product_of_possible_N_l221_221967


namespace circle_distance_condition_l221_221345

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221345


namespace trig_identity_proof_l221_221034

open Real -- Open the real namespace with trigonometric functions

theorem trig_identity_proof :
  (Real.cos (80 * Real.pi / 180) * Real.cos (20 * Real.pi / 180) + 
   Real.sin (100 * Real.pi / 180) * Real.sin (380 * Real.pi / 180)) = 1/2 :=
by sorry -- Skip the proof using placeholder

end trig_identity_proof_l221_221034


namespace alley_width_l221_221387

theorem alley_width (l m n w : ℝ) (h1 : cos (real.pi / 3) = m / l) (h2 : cos (7 * real.pi / 18) = n / l) :
  w = 1.732 * m :=
  sorry

end alley_width_l221_221387


namespace circle_center_line_distance_l221_221337

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221337


namespace functional_eqn_solution_l221_221658

theorem functional_eqn_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  (f = (λ x, 0) ∨ f = (λ x, 1)) :=
by
  sorry

end functional_eqn_solution_l221_221658


namespace circle_center_line_distance_l221_221336

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221336


namespace log_exponents_l221_221712

theorem log_exponents (a b : ℝ) (h₁ : 3^a = 225) (h₂ : 5^b = 225) : (1 / a) + (1 / b) = 1 / 2 := 
  sorry

end log_exponents_l221_221712


namespace largest_prime_exists_l221_221216

theorem largest_prime_exists :
  ∃ (p : ℕ) (a b : ℕ), p.prime ∧ p = 5 ∧ p = b / 2 * real.sqrt ((a - b) / (a + b)) ∧ a > 0 ∧ b > 0 := 
sorry

end largest_prime_exists_l221_221216


namespace angle_between_vectors_is_45_degrees_l221_221284

def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (3, 1)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
Real.sqrt (a.1 * a.1 + a.2 * a.2)

noncomputable def cosine (a b : ℝ × ℝ) : ℝ :=
dot_product a b / (magnitude a * magnitude b)

theorem angle_between_vectors_is_45_degrees :
  Real.acos (cosine vec_a vec_b) = Real.pi / 4 := sorry

end angle_between_vectors_is_45_degrees_l221_221284


namespace value_of_k_l221_221182

theorem value_of_k (k : ℝ) (h : ∫ x in 0..1, 3 * x^2 + k = 10) : k = 9 :=
sorry

end value_of_k_l221_221182


namespace masha_gets_chocolate_bar_l221_221169

noncomputable def masha_ensures_integer_sum : Prop :=
  ∀ (fractions : List (ℚ)), 
    (∀ f ∈ fractions, ∃ a b : ℕ, 0 < a ∧ a < b ∧ b ≤ 100 ∧ f = a / b ∧ (Nat.gcd a b = 1)) →
    ∃ (signs : List Int), 
      (∀ s : Int, s = 1 ∨ s = -1) →
      (∀ f ∈ fractions, signs.length = fractions.length) →
      ((signs.zip fractions).foldr (fun ⟨s, f⟩ acc => s * f + acc) 0).denom = 1

theorem masha_gets_chocolate_bar : masha_ensures_integer_sum := sorry

end masha_gets_chocolate_bar_l221_221169


namespace ellipse_equation_and_eccentricity_circle_equation_given_area_l221_221722

theorem ellipse_equation_and_eccentricity :
  (∀ (x y : ℝ), {O : ℝ × ℝ} (foci_on_x : ∀ (F₁ F₂ : ℝ × ℝ), F₁.2 = 0 ∧ F₂.2 = 0 ∧ dist F₁ F₂ = 2 ∧ O = (0,0)) (point_on_ellipse : (1, 3 / 2) ∈ {P : ℝ × ℝ | ∃ x y, P = (x, y) ∧ x^2 / 4 + y^2 / 3 = 1}),
    (∀ x y, x^2 / 4 + y^2 / 3 = 1) ∧
    (∀ a b c, 2 * c = 2 → a^2 = c^2 + b^2 → a = 2 ∧ b = sqrt 3 → (dist (0, -c) (dist (0, c)) = 2) → eccentricity = 1 / 2)) :=
by sorry

theorem circle_equation_given_area :
  (∀ (x y : ℝ), {A B F₁ F₂ l : ℝ × ℝ}
    (line_through_foci : ∃ t : ℝ, ∀ (P : ℝ × ℝ), l = (P.1, P.2) ∧ P.1 = t * P.2 - 1)
    (area_AF₂B : ∃ y₁ y₂ t,  area_AF₂B = 12 * sqrt 2 / 2 * |y₁ - y₂| / √2 = 12 * sqrt 2 / 7)
    (given_radius : ∃ r, r = 2 / sqrt t ^ 2 + 1),
    ∀ (x y : ℝ), (x-1)^2 + y^2 = 2)) :=
by sorry

end ellipse_equation_and_eccentricity_circle_equation_given_area_l221_221722


namespace distance_from_center_of_circle_to_line_l221_221330

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221330


namespace solution_l221_221555

noncomputable def x : ℕ := 13

theorem solution : (3 * x) - (36 - x) = 16 := by
  sorry

end solution_l221_221555


namespace pump_fill_time_without_leak_l221_221946

theorem pump_fill_time_without_leak
    (P : ℝ)
    (h1 : 2 + 1/7 = (15:ℝ)/7)
    (h2 : 1 / P - 1 / 30 = 7 / 15) :
  P = 2 := by
  sorry

end pump_fill_time_without_leak_l221_221946


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221117

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221117


namespace range_of_dot_product_between_EF_and_BA_l221_221259

-- Given conditions in Lean definitions
variables {A B C D E F : Point}
variables (AD DB EF : ℝ)
variables (AC BC : ℝ)
variables (DE DF : vector)
variables (u v : ℝ)

noncomputable def triangle_isosceles_and_conditions 
  (h_iso : AC = sqrt(5)) (h_iso2 : BC = sqrt(5))
  (AD_eq : AD = 1) (DB_eq : DB = 1) (EF_eq : EF = 1) 
  (dot_prod_cond : DE.dot DF ≤ 25/16) : 
  set ℝ :=
{ x | 4/3 ≤ x ∧ x ≤ 2 }

-- Rephrasing the math proof problem
theorem range_of_dot_product_between_EF_and_BA 
  (h_iso : AC = sqrt(5)) (h_iso2 : BC = sqrt(5))
  (AD_eq : AD = 1) (DB_eq : DB = 1) (EF_eq : EF = 1) 
  (dot_prod_cond : DE.dot DF ≤ 25/16) :
  ∃ u v, 4/3 ≤ u ∧  u ≤ 2 ∧
  (u * (E.x - F.x) + v * (E.y - F.y)) = (2 * (E.coords - F.coords)).x :=
sorry

end range_of_dot_product_between_EF_and_BA_l221_221259


namespace x_cubed_plus_y_cubed_l221_221691

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := 
by 
  sorry

end x_cubed_plus_y_cubed_l221_221691


namespace number_of_four_digit_numbers_l221_221742

theorem number_of_four_digit_numbers (digits : Finset ℕ) (h_digits : digits = {3, 0, 0, 3}) :
  number_of_valid_numbers digits = 6 := by
  sorry

end number_of_four_digit_numbers_l221_221742


namespace triangle_inequality_l221_221717

theorem triangle_inequality (a b c : ℝ) (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
by
  sorry

end triangle_inequality_l221_221717


namespace remainder_of_addition_and_division_l221_221548

theorem remainder_of_addition_and_division :
  (3452179 + 50) % 7 = 4 :=
by
  sorry

end remainder_of_addition_and_division_l221_221548


namespace circle_bisects_line_segment_l221_221861

-- Define the setup for the problem
structure RightTriangle (α : Type) :=
(A B C : α)
(right_angle_at_C : ∠ A C B = 90)

-- Define the notion of square on a leg
structure SquareOnLeg (α : Type) [EuclideanGeometry α] (T : RightTriangle α) :=
(A B C A₁ B₁ : α)
(square_on_AC : IsSquare A C A₁)
(square_on_BC : IsSquare B C B₁)
(circle_through_ABC : ∃ O r, Circle.center := O ∧ Circle.radius := r ∧ ∀ P ∈ {A₁, B₁, A, B, C}, P ∈ (Circle O r))

-- The proof problem
theorem circle_bisects_line_segment {α : Type} [EuclideanGeometry α] 
  (T : RightTriangle α) 
  (S : SquareOnLeg α T) : 
  let l := LineSegment.connect S.A₁ S.B₁ in
  let mid := Midpoint S.A₁ S.B₁ in
  let circle_through_ABC := S.circle_through_ABC in
  
  Circle.bisects_line_segment circle_through_ABC l :=
sorry

end circle_bisects_line_segment_l221_221861


namespace pascal_evens_and_multiples_of_4_l221_221744

def binomial (n k : Nat) : Nat := Nat.choose n k

def pascal_triangle (n : Nat) : List (List Nat) :=
  List.range n.map (λ i => List.range (i + 1).map (λ j => binomial i j))

def is_even (n : Nat) : Bool := n % 2 == 0
def is_multiple_of_4 (n : Nat) : Bool := n % 4 == 0

def count_condition (f : Nat → Bool) (lst : List Nat) : Nat :=
  List.length (lst.filter f)

def count_evens (triangle : List (List Nat)) : Nat :=
  List.sum (triangle.map (count_condition is_even))

def count_multiples_of_4 (triangle : List (List Nat)) : Nat :=
  List.sum (triangle.map (count_condition is_multiple_of_4))

theorem pascal_evens_and_multiples_of_4 :
  let triangle := pascal_triangle 12 in
  count_evens triangle = 30 ∧ count_multiples_of_4 triangle = 15 := by
  sorry

end pascal_evens_and_multiples_of_4_l221_221744


namespace ratio_of_octagon_areas_l221_221070

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221070


namespace wrapping_paper_each_present_l221_221846

theorem wrapping_paper_each_present (total_paper : ℚ) (num_presents : ℕ)
  (h1 : total_paper = 1 / 2) (h2 : num_presents = 5) :
  (total_paper / num_presents = 1 / 10) :=
by
  sorry

end wrapping_paper_each_present_l221_221846


namespace sequence_sum_l221_221935

theorem sequence_sum :
  (Σ i in Finset.range 2006, if i % 2 = 0 then -(i + 1) else i + 1) = -1003 :=
by
  sorry

end sequence_sum_l221_221935


namespace custom_op_example_l221_221382

def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

theorem custom_op_example : custom_op (-4) 6 = 68 :=
by
  sorry

end custom_op_example_l221_221382


namespace distance_from_center_to_line_of_tangent_circle_l221_221314

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221314


namespace jacket_price_correct_l221_221046

theorem jacket_price_correct (x : ℝ) : 
  let marked_price := 300
  let discount_rate := 0.3
  let profit := 20
  let selling_price := marked_price * (1 - discount_rate)
  in selling_price - x = profit → (300 * 0.7 - x = 20) := 
by
  intro h
  sorry

end jacket_price_correct_l221_221046


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221119

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221119


namespace problem_l221_221275

noncomputable def f (ω x : ℝ) : ℝ := (Real.sin (ω * x / 2))^2 + (1 / 2) * Real.sin (ω * x) - 1 / 2

theorem problem (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x ∈ Set.Ioo (Real.pi : ℝ) (2 * Real.pi) → f ω x ≠ 0) →
  ω ∈ Set.Icc 0 (1 / 8) ∪ Set.Icc (1 / 4) (5 / 8) :=
by
  sorry

end problem_l221_221275


namespace wrapping_paper_fraction_used_l221_221844

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l221_221844


namespace minimum_value_of_f_l221_221872

noncomputable def f (x : ℝ) : ℝ :=
  sin x ^ 2 + sqrt 3 * sin x * cos x

theorem minimum_value_of_f :
  ∀ x ∈ set.Icc (π/4) (π/2), f x ≥ 1 :=
begin
  sorry
end

end minimum_value_of_f_l221_221872


namespace Ryan_funding_goal_l221_221469

theorem Ryan_funding_goal 
  (avg_fund_per_person : ℕ := 10) 
  (people_recruited : ℕ := 80)
  (pre_existing_fund : ℕ := 200) :
  (avg_fund_per_person * people_recruited + pre_existing_fund = 1000) :=
by
  sorry

end Ryan_funding_goal_l221_221469


namespace smallest_m_for_partition_l221_221697

theorem smallest_m_for_partition (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), (∀ S : finset ℕ, S.card = m →
  (∀ A B : finset ℕ, A ∪ B = S ∧ A ∩ B = ∅ →
    ∃ x : finset ℕ, x.card = n ∧ x.sum (λ i, i) = x.sup id ∧ x ⊆ (A ∨ B))) ∧ m = n^2 - n - 1 :=
begin
  sorry
end

end smallest_m_for_partition_l221_221697


namespace circle_area_of_white_cube_l221_221450

/-- 
Marla has a large white cube with an edge length of 12 feet and enough green paint to cover 432 square feet.
Marla paints a white circle centered on each face of the cube, surrounded by a green border.
Prove the area of one of the white circles is 72 square feet.
 -/
theorem circle_area_of_white_cube
  (edge_length : ℝ) (paint_area : ℝ) (faces : ℕ)
  (h_edge_length : edge_length = 12)
  (h_paint_area : paint_area = 432)
  (h_faces : faces = 6) :
  ∃ (circle_area : ℝ), circle_area = 72 :=
by
  sorry

end circle_area_of_white_cube_l221_221450


namespace subset_primes_l221_221811

theorem subset_primes (P : Set ℕ) (hP : ∀ n, Prime n ↔ n ∈ P) (M : Set ℕ) (hM : M ⊆ P) (hM_card : M.card ≥ 3)
  (h_cond : ∀ (k : ℕ) (A : Finset ℕ), A.card = k ∧ A ⊂ M → (p_factors (∏ i in A, i - 1) ⊆ M)) :
  M = P :=
sorry

end subset_primes_l221_221811


namespace path_exceeds_14_units_l221_221992

variable (O A B C D P : Type)
variable [MetricSpace O]
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace D]
variable [MetricSpace P]

noncomputable def AB_segment_length : ℝ := 14
noncomputable def AC_distance : ℝ := 3
noncomputable def BD_distance : ℝ := 3

axiom AB_diameter (O : MetricSpace O) (A B : O) (AB_units : dist A B = AB_segment_length) : metric_space O
axiom C_point_on_AB (A C : O) (AC_units : dist A C = AC_distance) : C ∈ segment [A, B]
axiom D_point_on_AB (B D : O) (BD_units : dist B D = BD_distance) : D ∈ segment [A, B]
axiom P_on_circle (P : O) (AB : line_segment A B) (unit_dist : dist O P = O.radius) : P ∈ CircleCenterRadius O

theorem path_exceeds_14_units (O A B C D P : Type) [MetricSpace O]
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P]
  (AB_segment : AB_diameter O A B AB_segment_length)
  (C_on_AB : C_point_on_AB A C AC_distance)
  (D_on_AB : D_point_on_AB B D BD_distance)
  (P_circle : P_on_circle P (line_segment A B) dist O P O.radius) :
  ∃ P, path_length C P D > 14 := 
sorry

end path_exceeds_14_units_l221_221992


namespace projection_to_u_l221_221672

variable {α : Type*} [Field α]
variables (v : Fin 3 → α)
def u : Fin 3 → α := ![1, 3, -4]

def projection_matrix : Matrix (Fin 3) (Fin 3) α :=
  ![
    ![\frac{1}{26}, \frac{3}{26}, -\frac{2}{13}],
    ![\frac{3}{26}, \frac{9}{26}, -\frac{6}{13}],
    ![-\frac{2}{13}, -\frac{6}{13}, \frac{8}{13}]
  ]

theorem projection_to_u (v : Fin 3 → α) :
  projection_matrix.mulVec v = (u.inner_product v) • u / (u.inner_product u) :=
sorry

end projection_to_u_l221_221672


namespace value_of_a_plus_b_l221_221299

theorem value_of_a_plus_b (a b : ℝ) (h1 : sqrt 44 = 2 * sqrt a) (h2 : sqrt 54 = 3 * sqrt b) : a + b = 17 := 
sorry

end value_of_a_plus_b_l221_221299


namespace area_ratio_is_correct_l221_221139

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221139


namespace total_time_to_cover_distance_l221_221054

theorem total_time_to_cover_distance {dist_flat dist_uphill : ℝ}
  (walk_speed_flat run_speed_flat walk_speed_uphill run_speed_uphill : ℝ) 
  (total_dist_flat total_dist_uphill : ℝ)
  (percent_walk_flat percent_run_uphill : ℝ) 
  (h_walk_speed_flat : walk_speed_flat = 3)
  (h_run_speed_flat : run_speed_flat = 6)
  (h_walk_speed_uphill : walk_speed_uphill = 1.5)
  (h_run_speed_uphill : run_speed_uphill = 4)
  (h_total_dist_flat : total_dist_flat = 5)
  (h_total_dist_uphill : total_dist_uphill = 7)
  (h_percent_walk_flat : percent_walk_flat = 0.6)
  (h_percent_run_uphill : percent_run_uphill = 0.4) :
  let dist_walk_flat := percent_walk_flat * total_dist_flat,
      dist_run_flat := (1 - percent_walk_flat) * total_dist_flat,
      dist_run_uphill := percent_run_uphill * total_dist_uphill,
      dist_walk_uphill := (1 - percent_run_uphill) * total_dist_uphill,
      time_walk_flat := dist_walk_flat / walk_speed_flat,
      time_run_flat := dist_run_flat / run_speed_flat,
      time_run_uphill := dist_run_uphill / run_speed_uphill,
      time_walk_uphill := dist_walk_uphill / walk_speed_uphill,
      total_time := time_walk_flat + time_run_flat + time_run_uphill + time_walk_uphill
  in total_time = 4.833 := 
  by
    sorry

end total_time_to_cover_distance_l221_221054


namespace bella_age_is_five_l221_221629

-- Definitions from the problem:
def is_age_relation (bella_age brother_age : ℕ) : Prop :=
  brother_age = bella_age + 9 ∧ bella_age + brother_age = 19

-- The main proof statement:
theorem bella_age_is_five (bella_age brother_age : ℕ) (h : is_age_relation bella_age brother_age) :
  bella_age = 5 :=
by {
  -- Placeholder for proof steps
  sorry
}

end bella_age_is_five_l221_221629


namespace area_ratio_of_octagons_is_4_l221_221100

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221100


namespace find_set_B_l221_221820

open Set

variable (U : Finset ℕ) (A B : Finset ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (h1 : (U \ (A ∪ B)) = {1, 3})
variable (h2 : A ∩ (U \ B) = {2, 5})

theorem find_set_B : B = {4, 6, 7} := by
  sorry

end find_set_B_l221_221820


namespace log_inequality_solution_l221_221859

theorem log_inequality_solution (x : ℝ) :
  (log 3 (x^2 - 2) < log 3 ((3/2) * |x| - 1)) ↔ (-2 < x ∧ x < -real.sqrt 2) ∨ (real.sqrt 2 < x ∧ x < 2) ↔
  0 < x^2 - 2 ∧ 0 < (3/2) * |x| - 1 :=
begin
  sorry
end

end log_inequality_solution_l221_221859


namespace domain_of_composite_function_l221_221760

theorem domain_of_composite_function 
  (f : ℝ → ℝ)
  (h : ∀ x, f x ≠ f x → x ∈ Icc (-1 : ℝ) (2 : ℝ)) :
  ∀ x, f (2 * x - 1) ≠ f (2 * x - 1) → x ∈ Icc 0 (3/2 : ℝ) :=
by
  sorry

end domain_of_composite_function_l221_221760


namespace max_tied_teams_for_most_wins_in_round_robin_l221_221388

noncomputable def round_robin_tournament (n : ℕ) :=
  choose ( λ matches: ℕ, matches = n * (n - 1) / 2) -- total number of games played

theorem max_tied_teams_for_most_wins_in_round_robin (n : ℕ) (h₁ : n = 8) :
  let total_games := round_robin_tournament n in
  (total_games = 28) →
  (∀ wins, (∑ i in finset.range n, wins i) = total_games) →
  (∃ max_tied_teams, max_tied_teams = 7) :=
by
  sorry

end max_tied_teams_for_most_wins_in_round_robin_l221_221388


namespace count_multiples_of_14_between_100_and_400_l221_221747

theorem count_multiples_of_14_between_100_and_400 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (100 ≤ k ∧ k ≤ 400 ∧ 14 ∣ k) ↔ (∃ i : ℕ, k = 14 * i ∧ 8 ≤ i ∧ i ≤ 28)) :=
sorry

end count_multiples_of_14_between_100_and_400_l221_221747


namespace angle_sum_l221_221255

theorem angle_sum (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 3 / 4)
  (sin_β : Real.sin β = 3 / 5) :
  α + 3 * β = 5 * Real.pi / 4 := 
sorry

end angle_sum_l221_221255


namespace age_of_15th_student_l221_221488

theorem age_of_15th_student (avg_age_15_students avg_age_5_students avg_age_9_students : ℕ)
  (total_students total_age_15_students total_age_5_students total_age_9_students : ℕ)
  (h1 : total_students = 15)
  (h2 : avg_age_15_students = 15)
  (h3 : avg_age_5_students = 14)
  (h4 : avg_age_9_students = 16)
  (h5 : total_age_15_students = total_students * avg_age_15_students)
  (h6 : total_age_5_students = 5 * avg_age_5_students)
  (h7 : total_age_9_students = 9 * avg_age_9_students):
  total_age_15_students = total_age_5_students + total_age_9_students + 11 :=
by
  sorry

end age_of_15th_student_l221_221488


namespace quadratic_function_identity_l221_221014

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x ^ 2 + b * x + c

def f (x : ℝ) : ℝ := (1 / 2) * x
def g (x : ℝ) : ℝ := x ^ 2 + 1
def h (x : ℝ) : ℝ := 2 / (x - 3)
def j (x : ℝ) : ℝ := -3 / x

theorem quadratic_function_identity :
  is_quadratic g ∧ ¬ is_quadratic f ∧ ¬ is_quadratic h ∧ ¬ is_quadratic j :=
by
  sorry

end quadratic_function_identity_l221_221014


namespace no_root_l221_221479

theorem no_root :
  ∀ x : ℝ, x - (9 / (x - 4)) ≠ 4 - (9 / (x - 4)) :=
by
  intro x
  sorry

end no_root_l221_221479


namespace overtime_rate_per_hour_correct_l221_221892

-- Define the given constants and parameters
def working_days_per_week : Nat := 6
def working_hours_per_day : Nat := 10
def regular_earning_rate : Float := 2.10
def total_earnings_4_weeks : Float := 525
def total_hours_4_weeks : Nat := 245

-- Define the derived values
def regular_hours_per_week : Nat := working_days_per_week * working_hours_per_day
def regular_hours_4_weeks : Nat := regular_hours_per_week * 4
def overtime_hours_4_weeks : Nat := total_hours_4_weeks - regular_hours_4_weeks
def regular_earnings_4_weeks : Float := regular_hours_4_weeks * regular_earning_rate
def overtime_earnings_4_weeks : Float := total_earnings_4_weeks - regular_earnings_4_weeks

-- Define the statement to be proven
theorem overtime_rate_per_hour_correct :
  (overtime_earnings_4_weeks / (overtime_hours_4_weeks : Float)) = 4.20 :=
by
  -- Note: Proof steps are not required in this task
  sorry

end overtime_rate_per_hour_correct_l221_221892


namespace correct_calculation_C_l221_221013

theorem correct_calculation_C (a b y x : ℝ) : 
  (7 * a + a ≠ 8 * a^2) ∧ 
  (5 * y - 3 * y ≠ 2) ∧ 
  (3 * x^2 * y - 2 * x^2 * y = x^2 * y) ∧ 
  (3 * a + 2 * b ≠ 5 * a * b) :=
by {
  split,
  { exact sorry, },
  split,
  { exact sorry, },
  split,
  { exact sorry, },
  { exact sorry, },
}

end correct_calculation_C_l221_221013


namespace solve_log_inequality_l221_221482

theorem solve_log_inequality (x : ℝ) :
  8.57 * log 3 (log 4 ((4 * x - 1) / (x + 1))) - log (1 / 3) (log (1 / 4) ((x + 1) / (4 * x - 1))) < 0 
  → x > 2 / 3 :=
begin
  sorry
end

end solve_log_inequality_l221_221482


namespace focus_of_given_parabola_l221_221865

-- Define the given condition as a parameter
def parabola_eq (x y : ℝ) : Prop :=
  y = - (1/2) * x^2

-- Define the property for the focus of the parabola
def is_focus_of_parabola (focus : ℝ × ℝ) : Prop :=
  focus = (0, -1/2)

-- The theorem stating that the given parabola equation has the specific focus
theorem focus_of_given_parabola : 
  (∀ x y : ℝ, parabola_eq x y) → is_focus_of_parabola (0, -1/2) :=
by
  intro h
  unfold parabola_eq at h
  unfold is_focus_of_parabola
  sorry

end focus_of_given_parabola_l221_221865


namespace overall_average_is_correct_l221_221603

-- Define the number of students in each section
def students : List ℕ := [55, 35, 45, 42, 48, 50]

-- Define the mean marks obtained in the chemistry test for each section
def mean_marks : List ℕ := [50, 60, 55, 45, 53, 48]

-- Calculate the total marks for each section by multiplying students by mean marks
def total_marks_list : List ℕ := List.map (λ (p : ℕ × ℕ), p.1 * p.2) (List.zip students mean_marks)

-- Calculate the overall total marks by summing all total marks
def total_marks : ℕ := List.sum total_marks_list

-- Calculate the overall total number of students by summing all students
def total_students : ℕ := List.sum students

-- Calculate the overall average marks per student
def overall_average : ℚ := total_marks.toRat / total_students.toRat

theorem overall_average_is_correct :
  overall_average = 14159 / 275 := by
  sorry

end overall_average_is_correct_l221_221603


namespace angle_CDE_l221_221773

theorem angle_CDE (A B C E D : Type)
    [RightAngle : angle A = 90°]
    [RightAngleB : angle B = 90°]
    [RightAngleC : angle C = 90°]
    (angle_AEB : angle AEB = 60°)
    (angle_BED_BDE : ∀ {BDE : ℝ}, angle BED = 2 * angle BDE) :
    angle CDE = 60° := by
  sorry

end angle_CDE_l221_221773


namespace circle_distance_to_line_l221_221375

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221375


namespace find_k_value_l221_221719

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C P : V)
variables (k : ℝ)

-- Given conditions
def condition_1 : P = (1 / 2) • A + k • B - C := sorry
def condition_2 : ∃ (α β γ : ℝ), P = α • A + β • B + γ • C ∧ α + β + γ = 1 := sorry

-- Prove that given the conditions, k = 3/2
theorem find_k_value (h1 : condition_1) (h2 : condition_2) : k = 3 / 2 := sorry

end find_k_value_l221_221719


namespace min_top_block_number_l221_221978

-- Define the conditions of the problem
constant base_layer : list (fin 16 → ℕ)

noncomputable def average_rounded_down (a b c d : ℕ) : ℕ :=
  (a + b + c + d) / 4

noncomputable def pyramid : Type := 
  Σ (l1 : list ℕ),  -- Top layer (1 block)
  Σ (l2 : list ℕ),  -- 2nd layer (4 blocks)
  Σ (l3 : list ℕ),  -- 3rd layer (9 blocks)
  Σ (l4 : list ℕ),  -- Bottom layer (16 blocks)
  l4.length = 16 ∧ 
  l3.length = 9 ∧ 
  l2.length = 4 ∧ 
  l1.length = 1 ∧ 
  ∀ (i : ℕ) (h1 : i < 4), l2.nth_le i h1 = average_rounded_down 
                                     (l3.nth_le (3*i) (sorry)) 
                                     (l3.nth_le (3*i + 1) (sorry)) 
                                     (l3.nth_le (3*i + 2) (sorry)) 
                                     (l3.nth_le (3*i + 3) (sorry)) ∧
  ∀ (j : ℕ) (h2 : j < 9), l3.nth_le j h2 = average_rounded_down 
                                      (l4.nth_le (4*j) (sorry)) 
                                      (l4.nth_le (4*j + 1) (sorry)) 
                                      (l4.nth_le (4*j + 2) (sorry)) 
                                      (l4.nth_le (4*j + 3) (sorry))

-- State the mathematical proof problem
theorem min_top_block_number :
  ∃ p : pyramid, ∀ (t : ℕ) (ht : t ∈ p.1), t = 2 :=
sorry

end min_top_block_number_l221_221978


namespace arc_length_of_curve_l221_221029

noncomputable def arc_length_polar (ρ : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ (x : ℝ) in a..b, real.sqrt ((ρ x) ^ 2 + (ρ' x) ^ 2)

theorem arc_length_of_curve :
  arc_length_polar (λ φ : ℝ, 2 * real.sin φ) 0 (π/6) = π / 3 :=
by
  sorry

end arc_length_of_curve_l221_221029


namespace inequality_solution_l221_221860

theorem inequality_solution (x : ℝ) :
  (2*x + 3)/(x + 4) > (5*x + 6)/(3*x + 14) ↔ 
  x ∈ set.Ioo (-4 : ℝ) (-14/3 : ℝ) ∪ set.Ioi (6 + 3*real.sqrt 2) :=
by
  sorry

end inequality_solution_l221_221860


namespace tax_distribution_correct_l221_221993

inductive Tax
| propertyTaxOrganizations
| federalTax
| profitTaxOrganizations
| regionalTax
| transportFee

inductive BudgetLevel
| regionalBudget
| federalBudget
| regionalAndFederalBudget

open Tax BudgetLevel

def taxToBudgetLevel : Tax → BudgetLevel
| propertyTaxOrganizations => regionalBudget
| federalTax => federalBudget
| profitTaxOrganizations => regionalAndFederalBudget
| regionalTax => regionalBudget
| transportFee => regionalBudget

theorem tax_distribution_correct (t : Tax) :
  (t = propertyTaxOrganizations → taxToBudgetLevel t = regionalBudget) ∧
  (t = federalTax → taxToBudgetLevel t = federalBudget) ∧
  (t = profitTaxOrganizations → taxToBudgetLevel t = regionalAndFederalBudget) ∧
  (t = regionalTax → taxToBudgetLevel t = regionalBudget) ∧
  (t = transportFee → taxToBudgetLevel t = regionalBudget) :=
by
  intro t
  cases t
  all_goals {
    split; intro h; rw [h]; try { exact rfl }
  }

end tax_distribution_correct_l221_221993


namespace arithmetic_progression_sum_l221_221008

theorem arithmetic_progression_sum
  (a b c : ℤ) (d1 d2 d3 : ℤ) 
  (A B C : ℤ) (n : ℤ) :
  let a_k := (a + (n-1) * d1)
  let b_k := (b + (n-1) * d2)
  let c_k := (c + (n-1) * d3)
  let S_k := (A * a_k + B * b_k + C * c_k)
  in ∃ (c' d' : ℤ), S_k = c' + (n-1) * d' :=
by
  sorry

end arithmetic_progression_sum_l221_221008


namespace chicken_distribution_l221_221684

theorem chicken_distribution :
  (nat.choose 4 2) = 6 :=
by
  -- The proof is skipped
  sorry

end chicken_distribution_l221_221684


namespace ratio_of_areas_of_octagons_l221_221065

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221065


namespace octagon_area_ratio_l221_221126

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221126


namespace chips_marble_single_bag_unique_l221_221853

theorem chips_marble_single_bag_unique :
  ∀ (j1 j2 j3 j4 g1 g2 : ℕ),
    (j1, j2, j3, j4, g1, g2) ∈ {(16, 18, 22, 24, 26, 30, 36)},
    (j1 + j2 + j3 + j4) = 3 * (g1 + g2) →
    (16 + 18 + 22 + 24 + 26 + 30 + 36) - (j1 + j2 + j3 + j4 + g1 + g2) = 36 :=
by
  intros j1 j2 j3 j4 g1 g2 h_set 
  simp only [Finset.mem_insert, Finset.mem_singleton] at h_set 
  sorry -- Proof steps skipped

end chips_marble_single_bag_unique_l221_221853


namespace calculate_new_price_l221_221564

variables (original_price : ℝ) (discount_percentage : ℝ)

def discount_amount (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
  (discount_percentage / 100) * original_price

def new_price (original_price : ℝ) (discount_amount : ℝ) : ℝ :=
  original_price - discount_amount

theorem calculate_new_price :
  original_price = 150 ∧ discount_percentage = 10 →
  new_price original_price (discount_amount 150 10) = 135 := by
  sorry

end calculate_new_price_l221_221564


namespace tangent_length_sqrt_145_l221_221975

-- Define points
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (2, 5)
def R : ℝ × ℝ := (4, 1)

-- Statement to prove: The length of the segment tangent from O to the circle passing through P, Q, R is √145
theorem tangent_length_sqrt_145 (O P Q R: ℝ × ℝ) (hO : O = (0, 0)) (hP : P = (1, 2)) 
  (hQ : Q = (2, 5)) (hR : R = (4, 1)) :
  let circum_radius := dist (circumcenter P Q R) P in
  length_of_tangent_segment O (circumcircle P Q R circum_radius) = real.sqrt 145 :=
by
  sorry

end tangent_length_sqrt_145_l221_221975


namespace total_students_in_classrooms_l221_221516

theorem total_students_in_classrooms (tina_students maura_students zack_students : ℕ) 
    (h1 : tina_students = maura_students)
    (h2 : zack_students = (tina_students + maura_students) / 2)
    (h3 : 22 + 1 = zack_students) : 
    tina_students + maura_students + zack_students = 69 := 
by 
  -- Proof steps would go here, but we include 'sorry' as per the instructions.
  sorry

end total_students_in_classrooms_l221_221516


namespace basis_set_of_vectors_B_l221_221295

variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Declare V as a real vector space
variables {a b c : V} -- Declare the vectors

-- Assume that {a, b, c} forms a basis for the vector space V
axiom basis_a_b_c : Module.FiniteBasis ℝ V (Fin 3) (λ i : Fin 3, [a, b, c].nth i)

-- Define the set of vectors from option B
def set_of_vectors_B : Fin 3 → V
| 0 := a + b
| 1 := a - b
| 2 := c

-- The statement to prove: These vectors form a basis (are linearly independent)
theorem basis_set_of_vectors_B : Module.FiniteBasis ℝ V (Fin 3) set_of_vectors_B :=
sorry

end basis_set_of_vectors_B_l221_221295


namespace jogger_distance_ahead_of_train_l221_221590

theorem jogger_distance_ahead_of_train 
  (jogger_speed : ℝ := 9) -- jogger's speed in km/hr
  (train_speed : ℝ := 45) -- train's speed in km/hr
  (train_length : ℝ := 120) -- train's length in meters
  (pass_time : ℝ := 30) -- time taken to pass the jogger in seconds
  (km_to_m_s : ℝ := 5 / 18) -- conversion factor from km/hr to m/s
  : (D : ℝ) = 180 := 
begin
  let relative_speed := (train_speed - jogger_speed) * km_to_m_s, -- relative speed in m/s
  let distance_covered := relative_speed * pass_time, -- distance covered by the train in meters
  have h1 : distance_covered = train_length + D,
  { sorry }, -- This is where the detailed steps would go.
  exact h1
end

end jogger_distance_ahead_of_train_l221_221590


namespace octagon_area_ratio_l221_221147

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221147


namespace solve_for_n_l221_221856

theorem solve_for_n (n : ℤ) : 3^(2 * n + 1) = 1 / 27 → n = -2 := by
  sorry

end solve_for_n_l221_221856


namespace no_real_solution_l221_221990

theorem no_real_solution : ∀ x : ℝ, (2 * x - 10 * x + 24)^2 + 4 ≠ -2 * |x| := 
by {
  intro x,
  sorry
}

end no_real_solution_l221_221990


namespace calculate_x_l221_221748

theorem calculate_x :
  (∑ k in Finset.range 997, (2 * k + 1) * (998 - k)) = 997 * 498 * 331 :=
by
  sorry

end calculate_x_l221_221748


namespace hyperbola_sum_l221_221386

theorem hyperbola_sum (h k a b : ℝ) (c : ℝ) 
  (h_center : h = 1) (k_center : k = 0) 
  (a_vertex : a = 3) (b_sq : b^2 = (c^2 - a^2)) 
  (c_focus : c = sqrt 41) : 
  (h + k + a + b) = 4 + 4 * sqrt 2 := 
by
  have b_val : b = sqrt 32 := 
    by rw [b_sq, c_focus, sqr_sqrt, sqr_sub, sub_sub, sq_41, sq_3];
  rw [h_center, k_center, a_vertex, b_val]
  sorry

end hyperbola_sum_l221_221386


namespace base_seven_to_ten_l221_221532

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end base_seven_to_ten_l221_221532


namespace ring_arrangement_leftmost_digits_l221_221704

theorem ring_arrangement_leftmost_digits :
  let total_ways := (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3) in
  let leftmost_three_digits := (total_ways / 10^4) % 1000 in
  leftmost_three_digits = 126 := 
by
  sorry

end ring_arrangement_leftmost_digits_l221_221704


namespace total_savings_at_end_of_year_l221_221458

-- Defining constants for daily savings and the number of days in a year
def daily_savings : ℕ := 24
def days_in_year : ℕ := 365

-- Stating the theorem
theorem total_savings_at_end_of_year : daily_savings * days_in_year = 8760 :=
by
  sorry

end total_savings_at_end_of_year_l221_221458


namespace ordered_pair_A_B_l221_221874

open Polynomial

noncomputable def system_sum_of_roots : Prop :=
  let y := (x ^ 3 - 3 * x + 2)
  let eq1 := (2 * x + 3 * y = 3)
  ∃ x1 x2 x3 y1 y2 y3 : ℝ, (y1 = x1 ^ 3 - 3 * x1 + 2) ∧ (y2 = x2 ^ 3 - 3 * x2 + 2) ∧ (y3 = x3 ^ 3 - 3 * x3 + 2) ∧ 
                         (2 * x1 + 3 * y1 = 3) ∧ (2 * x2 + 3 * y2 = 3) ∧ (2 * x3 + 3 * y3 = 3) ∧ 
                         (x1 + x2 + x3 = 0) ∧ (y1 + y2 + y3 = 3)

theorem ordered_pair_A_B : 
  system_sum_of_roots :=
sorry

end ordered_pair_A_B_l221_221874


namespace distance_from_center_to_line_of_tangent_circle_l221_221321

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221321


namespace part1_part2_part3_l221_221581

section problem

-- Given conditions: frequencies of subject choices
def xiaoHong : List (String × Nat) := [("MM", 25), ("MP", 20), ("PM", 35), ("PP", 10), ("Rest", 10)]
def xiaoMing : List (String × Nat) := [("MM", 20), ("MP", 25), ("PM", 15), ("PP", 30), ("Rest", 10)]

-- Helper function to calculate probability
def probability (events: Nat) (total: Nat) : ℚ := events / total

-- Probabilities of Xiao Hong choosing subjects
def pr_XH_MM : ℚ := probability 25 100
def pr_XH_MP : ℚ := probability 20 100
def pr_XH_PM : ℚ := probability 35 100
def pr_XH_PP : ℚ := probability 10 100
def pr_XH_Rest : ℚ := probability 10 100

-- Probabilities of Xiao Ming choosing subjects
def pr_XM_MM : ℚ := probability 20 100
def pr_XM_MP : ℚ := probability 25 100
def pr_XM_PM : ℚ := probability 15 100
def pr_XM_PP : ℚ := probability 30 100
def pr_XM_Rest : ℚ := probability 10 100

theorem part1 : probability (3 * 5.choose 3) * (pr_XH_MM^3) * ((1 - pr_XH_MM)^2) = 45 / 512 := by
  sorry

-- Random variable X and its distribution
def X_distribution : List (Nat × ℚ) := [(0, 1/100), (1, 33/200), (2, 33/40)]

-- Expectation of X
def expectation_X : ℚ := (0 * 1/100) + (1 * 33/200) + (2 * 33/40)

theorem part2 : expectation_X = 363 / 200 := by
  sorry

-- Conditional probabilities for part 3
def pr_XH_pm_given_pp : ℚ := pr_XH_MP / (pr_XH_MP + pr_XH_PP)
def pr_XM_pm_given_pp : ℚ := pr_XM_MP / (pr_XM_MP + pr_XM_PP)

theorem part3 : pr_XH_pm_given_pp > pr_XM_pm_given_pp := by
  sorry

end problem

end part1_part2_part3_l221_221581


namespace distance_from_center_to_line_l221_221361

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221361


namespace point_not_on_stable_line_l221_221049

-- Definition of stable line
def is_stable_line (l : ℝ × ℝ → Prop) : Prop :=
  ∃ p1 p2 : ℚ × ℚ, p1 ≠ p2 ∧ l p1 ∧ l p2

-- The point we are considering
def my_point : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 3)

-- The theorem stating that my_point does not lie on any stable line
theorem point_not_on_stable_line (l : ℝ × ℝ → Prop) (h : is_stable_line l) : ¬ l my_point :=
sorry

end point_not_on_stable_line_l221_221049


namespace distance_from_D_to_midpoint_of_AB_l221_221774

theorem distance_from_D_to_midpoint_of_AB :
  ∀ (ABC : Type) [RightTriangle ABC] (A B C : ABC) 
  (D : Point) (E : Point) (F : Point) 
  (h1 : AB ⊥ BC)
  (h2 : InscribedCircleTouchPoints ABC D E F)
  (h3 : distance A B = 6)
  (h4 : distance B C = 8),
    distance D (midpoint A B) = 1 :=
by
  sorry

end distance_from_D_to_midpoint_of_AB_l221_221774


namespace trapezoid_area_l221_221776

-- Define the conditions of the problem
variables {b₁ b₂ : ℝ} (h : ℝ) (mid_segment : ℝ)

-- Given conditions
def is_trapezoid (b₁ b₂ h mid_segment : ℝ) : Prop :=
  b₁ = 63 ∧ h = 10 ∧ mid_segment = 5

-- The area of the trapezoid given the bases and height
def area_trapezoid (b₁ b₂ h : ℝ) : ℝ := (b₁ + b₂) * h / 2

-- The main theorem to prove the area is 580 under the given conditions
theorem trapezoid_area {b₁ b₂ : ℝ} (h : ℝ) (mid_segment : ℝ) :
  is_trapezoid b₁ b₂ h mid_segment → area_trapezoid b₁ b₂ h = 580 :=
by
  sorry

end trapezoid_area_l221_221776


namespace rowing_distance_upstream_l221_221594

noncomputable def boat_speed_in_still_water (d t s : ℕ) := (d - s * t) / t

theorem rowing_distance_upstream (d u t s : ℝ) (h₀ : d = 90) (h₁ : t = 3) (h₂ : s = 3) : 
  u = (boat_speed_in_still_water d t s - s) * t → 
  u = 72 := 
by sorry

end rowing_distance_upstream_l221_221594


namespace measure_minor_arc_LB_l221_221403

theorem measure_minor_arc_LB {Q : Circle} {L B S : Point}
  (angle_LBS : angle L B S = 60)
  (semi_circle_LBS : is_semicircle L B S) :
  measure_minor_arc L B = 60 :=
sorry

end measure_minor_arc_LB_l221_221403


namespace cups_filled_l221_221895

def total_tea : ℕ := 1050
def tea_per_cup : ℕ := 65

theorem cups_filled : Nat.floor (total_tea / (tea_per_cup : ℚ)) = 16 :=
by
  sorry

end cups_filled_l221_221895


namespace shorter_base_length_l221_221394

-- Define the conditions
variables (longer_base : ℝ) (segment_length : ℝ) (shorter_base : ℝ)

-- The length of the longer base
def longer_base_value : Prop := longer_base = 113

-- The length of the segment joining the midpoints of the diagonals is 5
def segment_length_value : Prop := segment_length = 5

-- The property relating the bases of the trapezoid
def trapezoid_property : Prop := segment_length = (longer_base - shorter_base) / 2

-- The theorem we need to prove
theorem shorter_base_length (h1 : longer_base_value) (h2 : segment_length_value) (h3 : trapezoid_property) : shorter_base = 103 :=
by
  -- The proof would go here
  sorry

end shorter_base_length_l221_221394


namespace lowest_place_l221_221929

-- Define the conditions of the problem
structure Conditions :=
  (points_first : ℕ := 1)
  (points_second : ℕ := 2)
  (points_third : ℕ := 3)
  (total_points : ℕ := points_first + points_second + points_third)

-- Lean statement to define and prove the equivalence
theorem lowest_place (c : Conditions) : 
  c.total_points = 6 → ∃ (N : ℕ), N + 1 = 7 :=
by
  intros h
  use 6
  exact Nat.succ_eq_add_one 6

end lowest_place_l221_221929


namespace circle_center_line_distance_l221_221333

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221333


namespace modulus_of_z_l221_221809

noncomputable def z : ℂ := (1 / (1 + complex.i)) + complex.i

theorem modulus_of_z : complex.abs z = (real.sqrt 2) / 2 :=
by 
  sorry 

end modulus_of_z_l221_221809


namespace find_number_l221_221883

theorem find_number (x : ℝ) (h : (x - 8 - 12) / 5 = 7) : x = 55 :=
sorry

end find_number_l221_221883


namespace remainder_m_plus_n_mod_1004_l221_221575

noncomputable def sum_infinite_series (a : ℕ) (b : ℕ) : ℚ :=
  (∑' r, (1 / ((2 * a)^r : ℚ))) * (∑' c, (1 / (b^c : ℚ)))

def structured_sum := sum_infinite_series 1004 1004

theorem remainder_m_plus_n_mod_1004 (m n : ℕ) (hmn_rel_prime : Nat.gcd m n = 1)
  (h_sum: structured_sum = m / n) : (m + n) % 1004 = 1 :=
sorry

end remainder_m_plus_n_mod_1004_l221_221575


namespace average_is_correct_l221_221530

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

def sum_of_numbers : ℕ := numbers.foldr (· + ·) 0

def number_of_values : ℕ := numbers.length

def average : ℚ := sum_of_numbers / number_of_values

theorem average_is_correct : average = 114391.82 := by
  sorry

end average_is_correct_l221_221530


namespace valid_placement_l221_221659

noncomputable def valid_perfect_squares : set ℕ := {4, 9, 16, 25}

def is_perfect_square_sum (x y : ℕ) : Prop :=
  x + y ∈ valid_perfect_squares

def valid_sequence (seq : list ℕ) : Prop :=
  (list.range 16).map (λn, n + 1) ~ seq ∧ -- each integer from 1 to 16 appears exactly once
  ∀ (i : ℕ), i < 15 → is_perfect_square_sum (seq.nth i).iget (seq.nth (i + 1)).iget -- neighboring integers sum to a perfect square

theorem valid_placement :
  valid_sequence [16, 9, 7, 2, 14, 11, 5, 4, 12, 13, 3, 6, 10, 15, 1, 8] ∧
  valid_sequence [8, 1, 15, 10, 6, 3, 13, 12, 4, 5, 11, 14, 2, 7, 9, 16] :=
by
  sorry

end valid_placement_l221_221659


namespace valid_volume_of_box_l221_221599

theorem valid_volume_of_box (V : ℕ) (h : V ∈ {80, 250, 500, 1000, 2000}) :
  ∃ (x : ℕ), (V = 10 * x^3) ↔ V = 80 :=
by {
  sorry
}

end valid_volume_of_box_l221_221599


namespace polynomial_identity_l221_221931

variable (x y : ℝ)

theorem polynomial_identity :
    (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 :=
sorry

end polynomial_identity_l221_221931


namespace distance_between_points_l221_221540

theorem distance_between_points :
  let x1 := 2
  let y1 := -2
  let x2 := 8
  let y2 := 8
  let dist := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  dist = Real.sqrt 136 :=
by
  -- Proof to be filled in here.
  sorry

end distance_between_points_l221_221540


namespace ratio_B_A_l221_221817

theorem ratio_B_A (A B : ℤ) (h : ∀ (x : ℝ), x ≠ -6 → x ≠ 0 → x ≠ 5 → 
  (A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x))) :
  (B : ℚ) / A = 2.2 := by
  sorry

end ratio_B_A_l221_221817


namespace find_solutions_l221_221685

theorem find_solutions (x y z w : ℝ) :
  (x + y + z + w = 10) ∧
  (x^2 + y^2 + z^2 + w^2 = 30) ∧
  (x^3 + y^3 + z^3 + w^3 = 100) ∧
  (x * y * z * w = 24) ↔
  ({x, y, z, w} = {1, 2, 3, 4}) :=
begin
  sorry
end

end find_solutions_l221_221685


namespace area_ratio_of_octagons_is_4_l221_221099

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221099


namespace integral_exp_eq_e_minus_1_l221_221179

noncomputable def integral_exponential : ℝ :=
  ∫ x in 0..1, Real.exp x

theorem integral_exp_eq_e_minus_1 : integral_exponential = Real.exp 1 - 1 :=
by
  sorry

end integral_exp_eq_e_minus_1_l221_221179


namespace area_of_shaded_figure_l221_221663

theorem area_of_shaded_figure (R : ℝ) (h₀ : 0 < R) :
  let α := 30 * (Real.pi / 180)
  in
    (½ * (2 * R)^2 * α) = (Real.pi * R^2) / 3 :=
by
let α := 30 * (Real.pi / 180)
calc
  ½ * (2 * R)^2 * α = ½ * 4 * R^2 * (Real.pi / 6) : by sorry
  ... = (4 * R^2 * Real.pi) / 12 : by sorry
  ... = (Real.pi * R^2) / 3 : by sorry

end area_of_shaded_figure_l221_221663


namespace problem_l221_221866

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

def slope_at_1: ℝ := (deriv f 1)

lemma tan_alpha (hx : 1 ≠ 0) : slope_at_1 = 3 := 
by 
  simp [f, deriv, differentiable_at_log, differentiable_at_inv, deriv_log, deriv_inv']; linarith

lemma cos_div_sin_minus_4cos {α : ℝ} (hα : Real.tan α = 3) : 
  (Real.cos α) / (Real.sin α - 4 * Real.cos α) = -1 := 
by 
  rw [hα, Real.tan_eq_sin_div_cos, div_div,
      mul_comm (Real.cos α), ← mul_assoc, mul_inv_cancel, mul_one]
  linarith [Real.sin_ne_zero_of_tan_ne_zero (by linarith)]

theorem problem (hx : 1 ≠ 0) : 
  (Real.cos (Real.arctan 3)) / (Real.sin (Real.arctan 3) - 4 * Real.cos (Real.arctan 3)) = -1 := 
by
  have hα : Real.tan (Real.arctan 3) = 3 := Real.tan_arctan (by linarith)
  exact cos_div_sin_minus_4cos hα

end problem_l221_221866


namespace units_digit_of_factorial_sum_l221_221991

theorem units_digit_of_factorial_sum :
  (1! + 2! + 3! + 4! + 5! + 6! + 7!) % 10 = 3 := by
  I have lemma units_digit (n : ℕ) : (if n < 5 then n! % 10 else 0) = 
  match n with
    | 0   := 1
    | 1   := 1
    | 2   := 2
    | 3   := 6
    | 4   := 4
    | _   := 0
    end
  sorry

end units_digit_of_factorial_sum_l221_221991


namespace tan_ratio_l221_221738

-- Define the vectors and their properties
variables (A B C : Type) [InnerProductSpace ℝ A]
variables (CA CB AB : A)
variables (tan_A tan_B : ℝ)
variables (cos_A cos_B sin_A sin_B : ℝ)
variables (angle_A angle_B : ℝ)

-- Define the given condition
def given_condition := 3 * (inner (CA + CB) AB) = 4 * (∥AB∥ ^ 2)

-- Define the result to be proved
def result := tan A / tan B = -7

theorem tan_ratio (h : given_condition) : result := 
by 
  sorry

end tan_ratio_l221_221738


namespace range_of_g_on_interval_l221_221258

-- Define the function g(x)
def g (x: ℝ) (m: ℝ) : ℝ := x ^ m

-- Prove the range of g(x) on the interval (0,1) is (0,1)
theorem range_of_g_on_interval (m : ℝ) (hm : m > 0) : 
  set.range (λ x, g x m) = set.Ioo 0 1 := by
  sorry

end range_of_g_on_interval_l221_221258


namespace time_to_drain_tank_l221_221055

theorem time_to_drain_tank (P L: ℝ) (hP : P = 1/3) (h_combined : P - L = 2/7) : 1 / L = 21 :=
by
  -- Proof omitted. Use the conditions given to show that 1 / L = 21.
  sorry

end time_to_drain_tank_l221_221055


namespace each_person_gets_1point25_l221_221032

-- Define the total amount of money.
def total_money : ℝ := 3.75

-- Define the number of people.
def number_of_people : ℕ := 3

-- Calculate the money each person would get assuming it's equally shared.
def money_per_person (total_money : ℝ) (number_of_people : ℕ) : ℝ :=
  total_money / number_of_people

-- Proof statement that each person will get $1.25.
theorem each_person_gets_1point25 : money_per_person total_money number_of_people = 1.25 :=
by
  sorry

end each_person_gets_1point25_l221_221032


namespace angle_bisector_divides_if_and_only_if_right_angle_l221_221407

-- Define the geometric objects
variables {A B C: Type} [triangle A B C]
variables {O: Type} [center_circumcircle O A B C]
variables {M: Type} [midpoint M A B]
variables {H: Type} [foot_altitude H C A B]
variables {D: Type} [midpoint_arc D A B C]

-- Define the angle C and conditions given
variables (angleC: ℝ) (angleC_is90: angleC = 90)
variables {bisector: Type} [angle_bisector bisector]
variables {median_and_altitude: Type} [divides_angle_median_altitude bisector]

-- Define the proof problem
theorem angle_bisector_divides_if_and_only_if_right_angle
  (triangle_conditions : AC ≠ BC ∧ bisector ∧ median_and_altitude) :
  ∃ (angleC: ℝ), angleC_is90 :=
sorry

end angle_bisector_divides_if_and_only_if_right_angle_l221_221407


namespace smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l221_221909

theorem smallest_prime_factor_of_5_pow_5_minus_5_pow_3 : Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p ∧ p ∣ (5^5 - 5^3) → p ≥ 2) := by
  sorry

end smallest_prime_factor_of_5_pow_5_minus_5_pow_3_l221_221909


namespace stock_comparison_l221_221621

-- Quantities of the first year depreciation or growth rates
def initial_investment : ℝ := 200.0
def dd_first_year_growth : ℝ := 1.10
def ee_first_year_decline : ℝ := 0.85
def ff_first_year_growth : ℝ := 1.05

-- Quantities of the second year depreciation or growth rates
def dd_second_year_growth : ℝ := 1.05
def ee_second_year_growth : ℝ := 1.15
def ff_second_year_decline : ℝ := 0.90

-- Mathematical expression to determine final values after first year
def dd_after_first_year := initial_investment * dd_first_year_growth
def ee_after_first_year := initial_investment * ee_first_year_decline
def ff_after_first_year := initial_investment * ff_first_year_growth

-- Mathematical expression to determine final values after second year
def dd_final := dd_after_first_year * dd_second_year_growth
def ee_final := ee_after_first_year * ee_second_year_growth
def ff_final := ff_after_first_year * ff_second_year_decline

-- Theorem representing the final comparison
theorem stock_comparison : ff_final < ee_final ∧ ee_final < dd_final :=
by {
  -- Here we would provide the proof, but as per instruction we'll place sorry
  sorry
}

end stock_comparison_l221_221621


namespace largest_prime_divisor_of_1202102_5_l221_221670

def base_5_to_decimal (n : String) : ℕ := 
  let digits := n.toList.map (λ c => c.toNat - '0'.toNat)
  digits.foldr (λ (digit acc : ℕ) => acc * 5 + digit) 0

def largest_prime_factor (n : ℕ) : ℕ := sorry -- Placeholder for the actual factorization logic.

theorem largest_prime_divisor_of_1202102_5 : 
  largest_prime_factor (base_5_to_decimal "1202102") = 307 := 
sorry

end largest_prime_divisor_of_1202102_5_l221_221670


namespace circle_radius_correct_l221_221940

noncomputable def radius_of_circle 
  (side_length : ℝ)
  (angle_tangents : ℝ)
  (sin_18 : ℝ) : ℝ := 
  sorry

theorem circle_radius_correct 
  (side_length : ℝ := 6 + 2 * Real.sqrt 5)
  (angle_tangents : ℝ := 36)
  (sin_18 : ℝ := (Real.sqrt 5 - 1) / 4) :
  radius_of_circle side_length angle_tangents sin_18 = 
  2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) :=
sorry

end circle_radius_correct_l221_221940


namespace solve_quadratic_l221_221481

theorem solve_quadratic (x : ℝ) : x^2 - 6 * x + 5 = 0 → x = 5 ∨ x = 1 :=
by
  assume h : x^2 - 6 * x + 5 = 0
  sorry

end solve_quadratic_l221_221481


namespace parallelogram_area_l221_221999

theorem parallelogram_area :
  let a := ⟨4, 2, -3⟩ : ℝ × ℝ × ℝ
  let b := ⟨2, -4, 5⟩ : ℝ × ℝ × ℝ
  real.sqrt (real.norm_sq (a.2 - b.2) + real.norm_sq (a.1 - b.1) + real.norm_sq (a.3 - b.3)) = 20 * real.sqrt 3 :=
by {
  sorry
}

end parallelogram_area_l221_221999


namespace smallest_number_of_rectangles_l221_221976

theorem smallest_number_of_rectangles (n : ℕ) (board_size : ℕ) (h_board_size : board_size = 2008) 
: ∃ M, M = 2009 ∧ (∀ rectangles : list (ℝ × ℝ), 
  (∀ c : (ℕ × ℕ), ∃ r ∈ rectangles, 
    (c.fst - 1) * board_size + c.snd - 1 < fst r ∧ (c.fst - 1) * board_size + c.snd - 1 < snd r)) 
  → list.length rectangles ≥ M) := 
begin
  use 2009,
  split,
  { exact rfl },
  { intros rectangles H,
    sorry
  }
end

end smallest_number_of_rectangles_l221_221976


namespace ratio_area_octagons_correct_l221_221113

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221113


namespace conjugate_in_third_quadrant_l221_221864

-- Define the given complex number and its conjugate
def z : ℂ := complex.cos (2 * real.pi / 3) + complex.sin (real.pi / 3) * complex.I
def conjugate_z : ℂ := complex.conj z

-- Define the coordinates of the conjugate as a pair of real numbers
def conjugate_coordinates : ℝ × ℝ := (conjugate_z.re, conjugate_z.im)

-- The target statement: Prove that the coordinates of the conjugate complex number place it in the third quadrant
theorem conjugate_in_third_quadrant : -real.sqrt 3 / 2 < 0 ∧ -1 / 2 < 0 := 
sorry

end conjugate_in_third_quadrant_l221_221864


namespace bouncy_balls_per_package_l221_221822

variable (x : ℝ)

def maggie_bought_packs : ℝ := 8.0 * x
def maggie_gave_away_packs : ℝ := 4.0 * x
def maggie_bought_again_packs : ℝ := 4.0 * x
def total_kept_bouncy_balls : ℝ := 80

theorem bouncy_balls_per_package :
  (maggie_bought_packs x = total_kept_bouncy_balls) → 
  x = 10 :=
by
  intro h
  sorry

end bouncy_balls_per_package_l221_221822


namespace satellite_work_lift_proof_l221_221647

noncomputable def work_lift_satellite (m : ℝ) (H : ℝ) (R₃ : ℝ) (g : ℝ) : ℝ :=
  let uH := R₃ + H
  let integral_rhs := ∫ u in R₃..uH, (m * g * R₃^2) / (u^2)
  -m * g * R₃^2 * (1 / R₃ - 1 / (R₃ + H))

theorem satellite_work_lift_proof :
  work_lift_satellite 5000 (400 * 10^3) (6380 * 10^3) 10 = 18820058997 :=
begin
  sorry
end

end satellite_work_lift_proof_l221_221647


namespace circle_distance_condition_l221_221346

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221346


namespace second_box_clay_capacity_l221_221948

def volume (d w l : ℝ) : ℝ := d * w * l

def scalingFactor (depth_factor width_factor length_factor : ℝ) : ℝ := depth_factor * width_factor * length_factor

theorem second_box_clay_capacity (depth1 width1 length1 clay1 : ℝ) (depth_factor width_factor : ℝ) :
  depth1 = 3 ∧ width1 = 4 ∧ length1 = 6 ∧ clay1 = 60 ∧ depth_factor = 3 ∧ width_factor = 4 →
  let depth2 := depth1 * depth_factor
  let width2 := width1 * width_factor
  let length2 := length1 in
  let V1 := volume depth1 width1 length1
  let V2 := volume depth2 width2 length2
  let volume_ratio := V2 / V1
  clay1 * volume_ratio = 720 :=
by
  intros h
  cases h
  -- proof omitted
  sorry

end second_box_clay_capacity_l221_221948


namespace factorization_l221_221656

theorem factorization (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4 * x * y = (x * y - 1 + x + y) * (x * y - 1 - x - y) :=
by sorry

end factorization_l221_221656


namespace last_digit_nat_number_l221_221499

theorem last_digit_nat_number (N : ℕ) (x : ℕ) (hN : N % 10 = x) 
  (hx : N = 2016 * x) : 
  N = 4032 ∨ N = 8064 ∨ N = 12096 ∨ N = 16128 := 
begin
  sorry
end

end last_digit_nat_number_l221_221499


namespace a7_arithmetic_sequence_l221_221783

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a1 : ℝ := 2
def a4 : ℝ := 5

theorem a7_arithmetic_sequence : ∃ d : ℝ, is_arithmetic_sequence a d ∧ a 1 = a1 ∧ a 4 = a4 → a 7 = 8 :=
by
  sorry

end a7_arithmetic_sequence_l221_221783


namespace divide_composite_products_l221_221994

theorem divide_composite_products :
  let first_three := [4, 6, 8]
  let next_three := [9, 10, 12]
  let prod_first_three := first_three.prod
  let prod_next_three := next_three.prod
  (prod_first_three : ℚ) / prod_next_three = 8 / 45 :=
by
  sorry

end divide_composite_products_l221_221994


namespace correct_equation_l221_221580

def initial_investment : ℝ := 2500
def expected_investment : ℝ := 6600
def growth_rate (x : ℝ) : ℝ := x

theorem correct_equation (x : ℝ) : 
  initial_investment * (1 + growth_rate x) + initial_investment * (1 + growth_rate x)^2 = expected_investment :=
by
  sorry

end correct_equation_l221_221580


namespace ratio_of_areas_of_octagons_l221_221081

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221081


namespace cone_vertex_angle_l221_221161

-- Let R be the radius of the sphere
-- Let h be the height of the cone
-- Let θ be the semi-vertical angle of the cone

noncomputable def radius : ℝ := sorry
noncomputable def volume_sphere (R : ℝ) : ℝ := (4 / 3) * Mathlib.pi * R^3
noncomputable def volume_cone (R : ℝ) (h : ℝ) : ℝ := (1 / 3) * Mathlib.pi * R^2 * h

theorem cone_vertex_angle (R h θ : ℝ) : 
  let V_S := volume_sphere R in
  let V_C := volume_cone R (7 / 2 * R) in
  V_S - V_C = (1 / 8) * V_S → 
  θ = Real.arctan (2 / 5) →
  2 * θ = Mathlib.pi / 3 := 
begin
  sorry
end

end cone_vertex_angle_l221_221161


namespace highlighter_difference_l221_221768

theorem highlighter_difference :
  ∀ (yellow pink blue : ℕ),
    yellow = 7 →
    pink = yellow + 7 →
    yellow + pink + blue = 40 →
    blue - pink = 5 :=
by
  intros yellow pink blue h_yellow h_pink h_total
  rw [h_yellow, h_pink] at h_total
  sorry

end highlighter_difference_l221_221768


namespace range_g_l221_221674

noncomputable def g (t : ℝ) : ℝ := (t^2 + 5/4 * t) / (t^2 + 1)

theorem range_g :
  set.range g = set.Icc (-5/16) (21/16) :=
sorry

end range_g_l221_221674


namespace number_of_lamps_on_third_level_l221_221033

/-- 
A seven-story pagoda has red lights doubling (ratio of 1/2) in number on each level from the top. 
Given a total of 381 lamps, how many lamps are on the third level from the top? 
-/
theorem number_of_lamps_on_third_level 
(geometric_sum : ∀ (a : ℕ), (\sum i in range 7, a / 2 ^ i) = 381) : 
(∃ (a : ℕ), a / 2^4 = 12) :=
sorry

end number_of_lamps_on_third_level_l221_221033


namespace sufficient_condition_for_parallel_l221_221609

def is_parallel_to_plane (a : line) (α : plane) : Prop :=
  ∃ β : plane, a ⊆ β ∧ β ∥ α

-- Given
variables (a : line) (α : plane)

-- The proof statement
theorem sufficient_condition_for_parallel :
  (∃ (β : plane), a ⊆ β ∧ β ∥ α) → is_parallel_to_plane a α :=
by {
  intro h,
  exact h,
  sorry -- The proof steps go here
}

end sufficient_condition_for_parallel_l221_221609


namespace minimum_positive_sum_l221_221806

-- Define the given conditions
def valid_b (b : Int) : Prop := b = -1 ∨ b = 0 ∨ b = 1 ∨ b = 2

def valid_sum (b : Fin 50 → Int) : Prop := (Finset.univ.sum b) % 2 = 0

def target_sum (b : Fin 50 → Int) : Int :=
Finset.sum (Finset.filter (λ (p : Fin 50 × Fin 50), p.1 < p.2) (Finset.univ.product Finset.univ)) 
  (λ (p : Fin 50 × Fin 50), b p.1 * b p.2)

-- State the main theorem to be proved
theorem minimum_positive_sum (b : Fin 50 → Int) 
  (h1 : ∀ i, valid_b (b i)) 
  (h2 : valid_sum b) : 

  ∃ (k : Int), k = target_sum b ∧ k = 11 := 
sorry

end minimum_positive_sum_l221_221806


namespace total_pennies_after_addition_l221_221467

def initial_pennies_per_compartment : ℕ := 10
def compartments : ℕ := 20
def added_pennies_per_compartment : ℕ := 15

theorem total_pennies_after_addition :
  (initial_pennies_per_compartment + added_pennies_per_compartment) * compartments = 500 :=
by 
  sorry

end total_pennies_after_addition_l221_221467


namespace ratio_of_octagon_areas_l221_221072

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221072


namespace range_of_function_l221_221220

theorem range_of_function :
  (∀ y : ℝ, (∃ x : ℝ, y = (x + 1) / (x ^ 2 + 1)) ↔ 0 ≤ y ∧ y ≤ 4/3) :=
by
  sorry

end range_of_function_l221_221220


namespace rectangle_area_220_l221_221926

noncomputable def side_square (area_square : ℝ) : ℝ := real.sqrt area_square

noncomputable def radius_circle (side_square : ℝ) : ℝ := side_square

noncomputable def length_rectangle (radius_circle : ℝ) : ℝ := (2 / 5) * radius_circle

noncomputable def area_rectangle (length_rectangle breadth_rectangle : ℝ) : ℝ :=
  length_rectangle * breadth_rectangle

theorem rectangle_area_220 :
  let area_square := 3025 in
  let breadth_rectangle := 10 in
  let side := side_square area_square in
  let radius := radius_circle side in
  let length := length_rectangle radius in
  area_rectangle length breadth_rectangle = 220 :=
by
  sorry

end rectangle_area_220_l221_221926


namespace congruent_figures_alignment_l221_221475

theorem congruent_figures_alignment (F1 F2 : PlaneFigure) (h1 : Congruent F1 F2) :
  ∃ (C : Point) (v : Vector),
  (aligned_by_translation F1 F2 v ∨ aligned_by_rotation F1 F2 C) ∧
  (¬aligned_by_translation F1 F2 v ↔ aligned_by_rotation F1 F2 C) :=
sorry

end congruent_figures_alignment_l221_221475


namespace fifth_pile_magazines_l221_221563

theorem fifth_pile_magazines :
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  fifth_pile = 13 :=
by
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  show fifth_pile = 13
  sorry

end fifth_pile_magazines_l221_221563


namespace stability_of_triangles_in_structures_l221_221178

theorem stability_of_triangles_in_structures :
  ∀ (bridges cable_car_supports trusses : Type),
  (∃ (triangular_structures : Type), (triangular_structures → bridges) ∧ (triangular_structures → cable_car_supports) ∧ (triangular_structures → trusses)) →
  (∀ (triangle : Type), is_stable triangle) →
  ∀ (structure : Type), uses_triangle structure → is_stable structure :=
begin
  sorry
end

end stability_of_triangles_in_structures_l221_221178


namespace circle_radius_ratio_iso_l221_221800

variables {α β : Type*}
variables (A B C D E T : α)
variables (r1 r2 : ℝ)
variables [metric α] [normed_group β]

-- Defining a rectangle ABCD
-- and point E on segment AD
def is_rectangle (A B C D : α) : Prop :=
  segment (A, B) ⊥ segment (B, C) ∧
  segment (B, C) ⊥ segment (C, D) ∧
  segment (C, D) ⊥ segment (D, A)

-- Defining inscribed circle tangency for quadrilateral BCDE
def inscribed_circle_tangent_Q (B C D E T : α) : Prop :=
  ∃ ω1 : set α,
  circle ω1(BCDE) ∧
  tangent_to_segment ω1 (B, E) T

-- Defining inscribed circle tangency for triangle ABE
def inscribed_circle_tangent_T (A B E T : α) : Prop :=
  ∃ ω2 : set α,
  circle ω2(ABE) ∧
  tangent_to_segment ω2 (B, E) T

theorem circle_radius_ratio_iso
  (h_rect : is_rectangle A B C D)
  (h_E_on_AD : E ∈ segment (A, D))
  (h_tangent_Q : inscribed_circle_tangent_Q B C D E T)
  (h_tangent_T : inscribed_circle_tangent_T A B E T) :
  r1 / r2 = (3 + real.sqrt 5) / 2 :=
sorry

end circle_radius_ratio_iso_l221_221800


namespace replace_asterisk_with_monomial_has_four_terms_l221_221402

theorem replace_asterisk_with_monomial_has_four_terms (x : ℝ) :
  let expr_1 := (x^4 - 3)^2
  let expr_2 := (x^3 + c * x^n)^2
  let combined_expr := expr_1 + expr_2 
  in (c = 3 ∧ n = 1) ∨ (c = sqrt 6 ∧ n = 2)
  → ((combined_expr.reduced).num_terms = 4) := 
sorry

end replace_asterisk_with_monomial_has_four_terms_l221_221402


namespace Q_value_Q_zeros_Q_digit_sum_l221_221427

def R (k : ℕ) := (10^k - 1) / 9

def Q := R 30 / R 6

theorem Q_value :
  Q = 1 + 10^6 + 10^{12} + 10^{18} + 10^{24} := sorry

theorem Q_zeros : 
  (Q.toString.filter (λ c => c = '0')).length = 20 := sorry

theorem Q_digit_sum : 
  (Q.toString.toList.map (λ c => c.toNat - '0'.toNat)).sum = 5 := sorry

end Q_value_Q_zeros_Q_digit_sum_l221_221427


namespace mass_percentage_Al_aluminum_carbonate_l221_221671

theorem mass_percentage_Al_aluminum_carbonate :
  let m_Al := 26.98  -- molar mass of Al in g/mol
  let m_C := 12.01  -- molar mass of C in g/mol
  let m_O := 16.00  -- molar mass of O in g/mol
  let molar_mass_CO3 := m_C + 3 * m_O  -- molar mass of CO3 in g/mol
  let molar_mass_Al2CO33 := 2 * m_Al + 3 * molar_mass_CO3  -- molar mass of Al2(CO3)3 in g/mol
  let mass_Al_in_Al2CO33 := 2 * m_Al  -- mass of Al in Al2(CO3)3 in g/mol
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  -- Proof goes here
  sorry

end mass_percentage_Al_aluminum_carbonate_l221_221671


namespace arrange_2015_integers_l221_221792

theorem arrange_2015_integers :
  ∃ (f : Fin 2015 → Fin 2015),
    (∀ i, (Nat.gcd ((f i).val + (f (i + 1)).val) 4 = 1 ∨ Nat.gcd ((f i).val + (f (i + 1)).val) 7 = 1)) ∧
    Function.Injective f ∧ 
    (∀ i, 1 ≤ (f i).val ∧ (f i).val ≤ 2015) :=
sorry

end arrange_2015_integers_l221_221792


namespace greg_age_is_16_l221_221185

-- Definitions based on given conditions
def cindy_age : ℕ := 5
def jan_age : ℕ := cindy_age + 2
def marcia_age : ℕ := 2 * jan_age
def greg_age : ℕ := marcia_age + 2

-- Theorem stating that Greg's age is 16 years given the above conditions
theorem greg_age_is_16 : greg_age = 16 := by
  sorry

end greg_age_is_16_l221_221185


namespace range_of_g_l221_221433

def f (x : ℝ) := 4 * x + 1
def g (x : ℝ) := f (f (f x))

theorem range_of_g :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → -43 ≤ g x ∧ g x ≤ 213 :=
by
  intros x hx
  have : g x = 64 * x + 21 := by
    unfold g
    unfold f
    ring
  rw this
  split
  case left =>
    linarith [hx.left]
  case right =>
    linarith [hx.right]

end range_of_g_l221_221433


namespace circle_distance_condition_l221_221347

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221347


namespace find_a_l221_221804

theorem find_a (a b c : ℂ) (h1 : a ∈ ℝ) (h2 : a + b + c = 5) (h3 : a * b + b * c + c * a = 5) (h4 : a * b * c = 5) : 
  a = 1 + real.cbrt 4 :=
by
  sorry

end find_a_l221_221804


namespace sum_distances_squared_l221_221716

theorem sum_distances_squared (a b x y : ℝ) :
  let A := (-a, -b)
  let B := (a, -b)
  let C := (a, b)
  let D := (-a, b)
  let P := (x, y)
  let dist2 (p1 p2 : ℝ × ℝ) : ℝ :=
    (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
in dist2 P A + dist2 P C = dist2 P B + dist2 P D :=
by
  sorry

end sum_distances_squared_l221_221716


namespace sales_fraction_of_year_l221_221924

theorem sales_fraction_of_year {A : ℝ} (h : 0 < A) :
  let total_sales_november := 11 * A,
      sales_december := 2 * A,
      total_sales_year := total_sales_november + sales_december in
  (sales_december / total_sales_year) = 2 / 13 := 
by
  let total_sales_november := 11 * A
  let sales_december := 2 * A
  let total_sales_year := total_sales_november + sales_december
  have : total_sales_year = 13 * A := sorry
  have : sales_december / total_sales_year = (2 * A) / (13 * A) := sorry
  have : (2 * A) / (13 * A) = 2 / 13 := sorry
  exact this

end sales_fraction_of_year_l221_221924


namespace distance_to_line_is_constant_l221_221355

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221355


namespace find_x_log_eq_l221_221207

theorem find_x_log_eq {x : ℝ} (h : log x 4 = log 27 3) : x = 64 := 
by
  sorry

end find_x_log_eq_l221_221207


namespace digit_sum_mod_sum_mod_prod_mod_skip_digits_0_or_9_permute_digits_l221_221002

-- Property 1: The sum of the digits property under modulo 9
theorem digit_sum_mod (n : ℕ) : 
  n % 9 = (n.digits 10).sum % 9 := sorry

-- Property 2: Sum of numbers under modulo 9
theorem sum_mod (ns : list ℕ) : 
  (ns.sum % 9) = (ns.map (λ n, n % 9)).sum % 9 := sorry

-- Property 3: Product of numbers under modulo 9
theorem prod_mod (ns : list ℕ) : 
  (ns.prod % 9) = (ns.map (λ n, n % 9)).prod % 9 := sorry

-- Property: Skipping digits 0 or 9 cannot be detected
theorem skip_digits_0_or_9 (n : ℕ) : 
  (n.digits 10).remove_all (to_digit 9)  % 9 = n % 9 :=
  sorry

-- Property: Permutations of digits cannot be detected
theorem permute_digits (n : ℕ) : 
  ((n.digits 10).permutations.map (λ perm, perm.sum)).head % 9 = n % 9 :=
  sorry

end digit_sum_mod_sum_mod_prod_mod_skip_digits_0_or_9_permute_digits_l221_221002


namespace find_S_l221_221933

variable {R S T c : ℝ}

theorem find_S
  (h1 : R = 2)
  (h2 : T = 1/2)
  (h3 : S = 4)
  (h4 : R = c * S / T)
  (h5 : R = 8)
  (h6 : T = 1/3) :
  S = 32 / 3 :=
by
  sorry

end find_S_l221_221933


namespace distance_from_center_to_line_l221_221363

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221363


namespace midpoint_iff_eq_s_two_CP_squared_l221_221441

noncomputable theory
open_locale classical

variable {α : Type*} [normed_field α]

-- Define an equilateral triangle ABC with side length a
structure EquilateralTriangle (α : Type*) :=
(a : α)
(A B C : Point α)
(h_eq : dist A B = a ∧ dist B C = a ∧ dist C A = a)

-- Define s and 2CP^2
def s (T : EquilateralTriangle α) (P : Point α) : α := (dist T.A P) ^ 2 + (dist T.B P) ^ 2
def two_CP_squared (T : EquilateralTriangle α) (P : Point α) : α := 2 * (dist T.C P) ^ 2

-- Define midpoint
def is_midpoint (P : Point α) (A B : Point α) : Prop :=
  dist A P = dist P B

-- Main statement to be proven
theorem midpoint_iff_eq_s_two_CP_squared 
  (T : EquilateralTriangle α) (P : Point α) :
  (is_midpoint P T.A T.B) ↔ (s T P = two_CP_squared T P) :=
sorry

end midpoint_iff_eq_s_two_CP_squared_l221_221441


namespace stadium_length_in_feet_l221_221880

theorem stadium_length_in_feet
  (length_in_yards : ℕ)
  (conversion_factor : ℕ)
  (H1 : length_in_yards = 61)
  (H2 : conversion_factor = 3) :
  length_in_yards * conversion_factor = 183 :=
by
  rw [H1, H2]
  exact (Nat.mul_comm 61 3).trans (congrArg (fun x => 61 * x) (Nat.mul_one 3)).symm
  sorry

end stadium_length_in_feet_l221_221880


namespace collinear_X_Y_Z_l221_221172

open EuclideanGeometry

theorem collinear_X_Y_Z
  (O C P Q M R X Y Z E F S T : Point)
  (AB : Line)
  (h1 : IsChord AB (Circle O))
  (h2 : IsMidpointOfArc M AB (Circle O))
  (h3 : IsExternalPoint C (Circle O))
  (h4 : IsTangentToCircle C S (Circle O))
  (h5 : IsTangentToCircle C T (Circle O))
  (h6 : IntersectionOfSegments (Segment M S) AB E)
  (h7 : IntersectionOfSegments (Segment M T) AB F)
  (h8 : PerpendicularToLineThroughPoint E AB X (Segment DS))
  (h9 : PerpendicularToLineThroughPoint F AB Y (Segment OT))
  (h10 : IsSecantOfCircle C P Q (Circle O))
  (h11 : IntersectionOfSegments (Segment M P) AB R)
  (h12 : IsCircumcenterOfTriangle Z (Triangle P Q R)) :
  Collinear X Y Z :=
  sorry

end collinear_X_Y_Z_l221_221172


namespace sum_of_digits_base7_777_l221_221910

theorem sum_of_digits_base7_777 : 
  let base7_repr := [2, 1, 6, 0] in 
  base7_repr.sum = 9 :=
by {
  sorry
}

end sum_of_digits_base7_777_l221_221910


namespace circle_tangent_distance_l221_221310

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221310


namespace black_grid_after_rotation_l221_221937
open ProbabilityTheory

noncomputable def probability_black_grid_after_rotation : ℚ := 6561 / 65536

theorem black_grid_after_rotation (p : ℚ) (h : p = 1 / 2) :
  probability_black_grid_after_rotation = (3 / 4) ^ 8 := 
sorry

end black_grid_after_rotation_l221_221937


namespace area_ratio_is_correct_l221_221138

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221138


namespace prime_factor_of_difference_l221_221980

theorem prime_factor_of_difference (A B C : ℕ) (hA : A ≠ C) : 
  let ABC := 100 * A + 10 * B + C;
      CBA := 100 * C + 10 * B + A
  in 3 ∣ (ABC - CBA) :=
by
  let ABC := 100 * A + 10 * B + C
  let CBA := 100 * C + 10 * B + A
  sorry

end prime_factor_of_difference_l221_221980


namespace ring_arrangement_leftmost_digits_l221_221705

theorem ring_arrangement_leftmost_digits :
  let total_ways := (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3) in
  let leftmost_three_digits := (total_ways / 10^4) % 1000 in
  leftmost_three_digits = 126 := 
by
  sorry

end ring_arrangement_leftmost_digits_l221_221705


namespace simplify_fraction_rationalize_denominator_l221_221476

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fraction := 5 / (sqrt 125 + 3 * sqrt 45 + 4 * sqrt 20 + sqrt 75)

theorem simplify_fraction_rationalize_denominator :
  fraction = sqrt 5 / 27 :=
by
  sorry

end simplify_fraction_rationalize_denominator_l221_221476


namespace polynomials_equal_if_equal_level_sets_l221_221283

open Complex

noncomputable def P (x : ℂ) : ℂ := sorry
noncomputable def Q (x : ℂ) : ℂ := sorry

def P_a (a : ℝ) : Set ℂ := {z | P(z) = a}
def Q_a (a : ℝ) : Set ℂ := {z | Q(z) = a}
def K : Set ℝ := {a | P_a a = Q_a a}

theorem polynomials_equal_if_equal_level_sets (h_non_constant_P : ¬is_constant P)
    (h_non_constant_Q : ¬is_constant Q)
    (h_two_elements_in_K : ∃ a b : ℝ, a ≠ b ∧ a ∈ K ∧ b ∈ K) :
  P = Q :=
sorry

end polynomials_equal_if_equal_level_sets_l221_221283


namespace circle_center_line_distance_l221_221335

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221335


namespace sum_log_tan_zero_l221_221654

theorem sum_log_tan_zero : ∑ i in finset.range 179 \ finset.singleton 90, real.log10 (real.tan (i + 1) * real.pi / 180) = 0 :=
sorry

end sum_log_tan_zero_l221_221654


namespace shadow_length_of_flagpole_l221_221588

theorem shadow_length_of_flagpole :
  ∀ (S : ℝ), (18 : ℝ) / S = (22 : ℝ) / 55 → S = 45 :=
by
  intro S h
  sorry

end shadow_length_of_flagpole_l221_221588


namespace circle_center_line_distance_l221_221334

noncomputable def distance_point_to_line (x₁ y₁ A B C : ℝ) : ℝ :=
|A * x₁ + B * y₁ + C| / Real.sqrt (A^2 + B^2)

theorem circle_center_line_distance (a : ℝ) (h : a^2 - 6 * a + 5 = 0) :
  distance_point_to_line a a 2 (-1) (-3) = (2 * Real.sqrt 5 / 5) :=
by
  sorry

end circle_center_line_distance_l221_221334


namespace game_not_fair_l221_221777

/-- Definition of the game conditions and outcomes -/
def balls : List ℕ := [1, 2, 3, 4]

def outcomes : List (ℕ × ℕ) := do
  a <- balls
  b <- balls.filter (≠ a)
  pure (a, b)

def playerA_wins (outcome : ℕ × ℕ) : Bool :=
  (outcome.fst + outcome.snd) % 2 = 1

def playerB_wins (outcome : ℕ × ℕ) : Bool :=
  ¬playerA_wins outcome

/-- Count the outcomes where each player wins -/
def count_playerA_wins : ℕ :=
  (outcomes.filter playerA_wins).length

def count_playerB_wins : ℕ :=
  (outcomes.filter playerB_wins).length

/-- The Lean theorem to prove the game is not fair -/
theorem game_not_fair : count_playerA_wins ≠ count_playerB_wins :=
by
  -- Steps to prove the theorem would go here.
  sorry

end game_not_fair_l221_221777


namespace expected_value_zero_l221_221056

theorem expected_value_zero (AB BC : ℝ) (hAB : AB = 6) (hBC : BC = 9) (P : ℝ × ℝ): 
  ∃ (h : ℝ), h = 5 ∧ ∃ (x y : ℝ), x ∈ set.Icc 0 6 ∧ y ∈ set.Icc 0 9 ∧ 
  (has_sum (λ P, (P.fst + 5) ^ 2 + (P.snd + 5) ^ 2 - P.fst ^ 2 - P.snd ^ 2) 0) :=
by
  sorry

end expected_value_zero_l221_221056


namespace area_enclosed_by_graph_l221_221209

theorem area_enclosed_by_graph (x y : ℝ) (h : x^2 + y^2 = |x| + |y|) : 
  enclosed_area x y h = π + 2 := 
sorry

end area_enclosed_by_graph_l221_221209


namespace sum_of_possible_values_of_g_l221_221439

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (x : ℝ) : ℝ := 3 * x - 4

theorem sum_of_possible_values_of_g :
  let x1 := (9 + 3 * Real.sqrt 5) / 2
  let x2 := (9 - 3 * Real.sqrt 5) / 2
  g x1 + g x2 = 19 :=
by
  sorry

end sum_of_possible_values_of_g_l221_221439


namespace find_f_29_l221_221397

theorem find_f_29 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 3) = (x - 3) * (x + 4)) : f 29 = 170 := 
by
  sorry

end find_f_29_l221_221397


namespace stuffed_animal_cost_l221_221448

variable (S : ℝ)  -- Cost of the stuffed animal
variable (total_cost_after_discount_gave_30_dollars : S * 0.10 = 3.6) 
-- Condition: cost of stuffed animal = $4.44
theorem stuffed_animal_cost :
  S = 4.44 :=
by
  sorry

end stuffed_animal_cost_l221_221448


namespace independence_of_Z_and_X_l221_221928

noncomputable theory
open ProbabilityTheory

variables {X Y Z : Type*}
variables [IsGaussianSystem X Y]
variables [RandomVector Y] [RandomVector X]

def covariance (Y X : Type*) [RandomVector Y] [RandomVector X] : Type* :=
  (E ((Y - E Y) (X - E X)ᵀ))

theorem independence_of_Z_and_X 
  {Y X : Type*} [RandomVector Y] [RandomVector X] [IsGaussianSystem X Y] 
  (h_cov : covariance Y X = E ((Y - E Y) (X - E X)ᵀ)) : 
  covariance (Y - covariance Y X * D X⁺) X = 0 :=
sorry

end independence_of_Z_and_X_l221_221928


namespace range_of_a_l221_221379

-- Define the function f
def f (x a : ℝ) : ℝ := 2^(x - 1) - a

-- The theorem states that if f(x, a) has a root, then a is in the range (0, +∞)
theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, f x a = 0) : a ∈ set.Ioi 0 :=
sorry

end range_of_a_l221_221379


namespace regular_octagon_exterior_angle_is_45_degrees_l221_221790

-- Definitions for the problem conditions
def is_regular_octagon (n : ℕ) := n = 8

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

def interior_angle (n : ℕ) : ℝ := sum_of_interior_angles n / n

-- The exterior angle theorem for regular polygons
def exterior_angle (n : ℕ) : ℝ := 180 - interior_angle n

-- Proof Statement
theorem regular_octagon_exterior_angle_is_45_degrees :
  ∀ (n : ℕ), is_regular_octagon n → exterior_angle n = 45 :=
by
  intros n hn,
  simp [is_regular_octagon, exterior_angle, interior_angle, sum_of_interior_angles],
  sorry

end regular_octagon_exterior_angle_is_45_degrees_l221_221790


namespace angle_between_line_and_plane_l221_221208

open Real

def plane1 (x y z : ℝ) : Prop := 2*x - y - 3*z + 5 = 0
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

def point_M : ℝ × ℝ × ℝ := (-2, 0, 3)
def point_N : ℝ × ℝ × ℝ := (0, 2, 2)
def point_K : ℝ × ℝ × ℝ := (3, -3, 1)

theorem angle_between_line_and_plane :
  ∃ α : ℝ, α = arcsin (22 / (3 * sqrt 102)) :=
by sorry

end angle_between_line_and_plane_l221_221208


namespace minimum_positive_period_l221_221500

noncomputable theory
open Real

-- Define the function 
def my_function (x : ℝ) : ℝ := tan ((π / 2) * x - π / 3)

-- Statement to prove that the minimum positive period of the function is 2
theorem minimum_positive_period :
  ∃ T > 0, (∀ x, my_function (x + T) = my_function x) ∧
  (∀ T' > 0, (∀ x, my_function (x + T') = my_function x) → T' ≥ 2) ∧
  T = 2 :=
sorry

end minimum_positive_period_l221_221500


namespace distance_to_line_is_constant_l221_221356

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221356


namespace roots_positive_and_fraction_simplification_l221_221431

theorem roots_positive_and_fraction_simplification (a b : ℝ) (h₁: a > b)
  (h₂ : a * b = 4) (h₃ : a + b = 6) :
  (a > 0 ∧ b > 0) ∧ (∃ k : ℝ, k = (sqrt a - sqrt b) / (sqrt a + sqrt b) ∧ k = sqrt 5 / 5) :=
by
  split
  -- Proof that a > 0 and b > 0
  sorry
  -- Proof that (sqrt a - sqrt b) / (sqrt a + sqrt b) = sqrt 5 / 5
  use (sqrt a - sqrt b) / (sqrt a + sqrt b)
  split
  sorry
  sorry

end roots_positive_and_fraction_simplification_l221_221431


namespace least_sum_of_variables_l221_221024

theorem least_sum_of_variables (x y z w : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)  
  (h : 2 * x^2 = 5 * y^3 ∧ 5 * y^3 = 8 * z^4 ∧ 8 * z^4 = 3 * w) : x + y + z + w = 54 := 
sorry

end least_sum_of_variables_l221_221024


namespace problem_l221_221729

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 / (x * x + 1) + sin x / (x * x + 1)

noncomputable def f' (x : ℝ) : ℝ := 
  ((2 * x + cos x) * (x * x + 1) - (2 * x + sin x) * (2 * x)) / (x * x + 1)^2

theorem problem :
  f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end problem_l221_221729


namespace coefficient_of_x4_in_expansion_l221_221971

theorem coefficient_of_x4_in_expansion : 
  (coeff ((1 + sqrt x) ^ 10) x ^ 4) = 45 := 
by 
  sorry

end coefficient_of_x4_in_expansion_l221_221971


namespace circle_distance_condition_l221_221344

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221344


namespace octagon_area_ratio_l221_221135

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221135


namespace ratio_of_distances_l221_221921

theorem ratio_of_distances (s_a t_a s_b t_b : ℝ)
  (h₀ : s_a = 80) (h₁ : t_a = 5) (h₂ : s_b = 100) (h₃ : t_b = 2) :
  (s_a * t_a) / (s_b * t_b) = 2 :=
by
  -- Define the distance covered by Car A.
  let d_a := s_a * t_a
  rw [h₀, h₁] at d_a
  have hd_a : d_a = 400 := by norm_num

  -- Define the distance covered by Car B.
  let d_b := s_b * t_b
  rw [h₂, h₃] at d_b
  have hd_b : d_b = 200 := by norm_num

  -- Calculate the ratio of distances.
  rw [hd_a, hd_b]
  norm_num
  sorry

end ratio_of_distances_l221_221921


namespace Mahdi_cycles_on_Sunday_l221_221823

constant Days : Type
variables (Monday Tuesday Wednesday Thursday Friday Saturday Sunday : Days)
variable (Mahdi_practices_one_sport_each_day_of_week : ∀ d : Days, ∃ s : Prop, s)

constant Sports : Type
variables (run basketball golf swim tennis cycle : Sports)
variable (Mahdi_basketball_on_Tuesday : basketball Tuesday)
variable (Mahdi_golf_on_Friday : golf Friday)
variable (Mahdi_runs_three_days_a_week : ∃ d1 d2 d3 : Days, (run d1 ∧ run d2 ∧ run d3) ∧ 
          (¬ consecutive d1 d2 ∧ ¬ consecutive d1 d3 ∧ ¬ consecutive d2 d3) ∨
          (consecutive d1 d2 ∧ ¬ consecutive d2 d3))
variable (Mahdi_never_cycles_day_after_tennis : ∀ d1 d2 : Days, tennis d1 → cycle d2 → ¬ consecutive d1 d2)
variable (Mahdi_never_cycles_day_before_swimming : ∀ d1 d2 : Days, swim d1 → cycle d2 → ¬ consecutive d2 d1)
variable (Mahdi_plays_sports : (swim ∨ tennis ∨ cycle))

theorem Mahdi_cycles_on_Sunday : cycle Sunday := 
sorry

end Mahdi_cycles_on_Sunday_l221_221823


namespace length_of_BC_l221_221786

theorem length_of_BC (AB AD DC BD : ℝ) (a : ℝ) (h1 : AB = 4) (h2 : AD = 4) (h3 : DC = 4) (h4 : BD = 2 * a)
  (h5 : cos θ = 1 / 8) : 
  ∃ (BC : ℝ), BC = 3 * Real.sqrt 2 :=
by
  sorry

end length_of_BC_l221_221786


namespace canoe_rental_cost_l221_221000

-- Definitions and conditions
variable (K C : ℝ) -- Using real numbers for kayak count and canoe cost
axiom kayak_cost : ℝ := 12
axiom total_revenue : ℝ := 432
axiom canoe_kayak_relation : K + 6 = 4 / 3 * K + 6
axiom additional_canoes : K + 6 - K = 6

-- Theorem statement
theorem canoe_rental_cost : 432 = 12 * K + (K + 6) * C → C = 9 :=
by
  intro h
  sorry -- Proof omitted

end canoe_rental_cost_l221_221000


namespace circle_tangent_distance_l221_221308

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221308


namespace range_of_g_l221_221274

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x / log 3 + m

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (f x m)^2 - f (x^2) m

theorem range_of_g :
  (∃ m : ℝ, f 1 m = 2) →
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → 2 ≤ g x 2 ∧ g x 2 ≤ 5) :=
by
  sorry

end range_of_g_l221_221274


namespace smallest_n_with_conditions_l221_221630

theorem smallest_n_with_conditions :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ n % 9 = 2 ∧ n % 7 = 5 ∧
             ∀ m, 100 ≤ m ∧ m < 1000 ∧ m % 9 = 2 ∧ m % 7 = 5 → n ≤ m :=
begin
  use 110,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm1 hm2 hm3 hm4,
    sorry }
end

end smallest_n_with_conditions_l221_221630


namespace cases_in_1995_l221_221767

theorem cases_in_1995 (initial_cases cases_2010 : ℕ) (years_total : ℕ) (years_passed : ℕ) (cases_1995 : ℕ)
  (h1 : initial_cases = 700000) 
  (h2 : cases_2010 = 1000) 
  (h3 : years_total = 40) 
  (h4 : years_passed = 25)
  (h5 : cases_1995 = initial_cases - (years_passed * (initial_cases - cases_2010) / years_total)) : 
  cases_1995 = 263125 := 
sorry

end cases_in_1995_l221_221767


namespace distance_from_center_to_line_of_tangent_circle_l221_221319

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221319


namespace distance_to_line_is_constant_l221_221354

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221354


namespace quotient_of_division_l221_221492

theorem quotient_of_division
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1370)
  (h2 : larger = 1626)
  (h3 : ∃ q r, larger = smaller * q + r ∧ r = 15) :
  ∃ q, larger = smaller * q + 15 ∧ q = 6 :=
by
  sorry

end quotient_of_division_l221_221492


namespace find_x_l221_221785
noncomputable theory

def point_on_side (T P R: Point) : Prop := on_segment T P R
def straight_line (Q R S: Point) : Prop := collinear Q R S

theorem find_x 
  (P Q R S T : Point) 
  (angle_RPQ : Angle)
  (angle_RQP : Angle)
  (angle_SRP : Angle)
  (x : ℝ) 
  (h1 : point_on_side T P R)
  (h2 : straight_line Q R S)
  (h3 : angle_RPQ = 30)
  (h4 : angle_RQP = 2 * x)
  (h5 : angle_SRP = angle_RPQ + angle_RQP)
  (h6 : angle_SRP + x = 180) : 
  x = 50 :=
  sorry

end find_x_l221_221785


namespace probability_sequence_consecutive_zeros_l221_221951

theorem probability_sequence_consecutive_zeros :
  let a_n : ℕ → ℕ := λ n, 3 ^ (n - 1),
      sequences_of_length_10 := 59049,
      valid_sequences := a_n 10,
      probability := valid_sequences / sequences_of_length_10,
      (m, n) := (1, 3) in
  m + n = 4 :=
begin
  sorry
end

end probability_sequence_consecutive_zeros_l221_221951


namespace sqrt_expr_simplify_l221_221632

theorem sqrt_expr_simplify :
  (sqrt 54 - sqrt 27) + sqrt 3 + 8 * sqrt (1 / 2) = 3 * sqrt 6 - 2 * sqrt 3 + 4 * sqrt 2 :=
by
  sorry

end sqrt_expr_simplify_l221_221632


namespace sum_of_coefficients_zero_l221_221798

open Real

theorem sum_of_coefficients_zero (a b c p1 p2 q1 q2 : ℝ)
  (h1 : ∃ p1 p2 : ℝ, p1 ≠ p2 ∧ a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0)
  (h2 : ∃ q1 q2 : ℝ, q1 ≠ q2 ∧ c * q1^2 + b * q1 + a = 0 ∧ c * q2^2 + b * q2 + a = 0)
  (h3 : q1 = p1 + (p2 - p1) / 2 ∧ p2 = p1 + (p2 - p1) ∧ q2 = p1 + 3 * (p2 - p1) / 2) :
  a + c = 0 := sorry

end sum_of_coefficients_zero_l221_221798


namespace chicken_distribution_l221_221683

theorem chicken_distribution :
  (nat.choose 4 2) = 6 :=
by
  -- The proof is skipped
  sorry

end chicken_distribution_l221_221683


namespace largest_piece_length_l221_221619

theorem largest_piece_length (v : ℝ) (hv : v + (3/2) * v + (9/4) * v = 95) : 
  (9/4) * v = 45 :=
by sorry

end largest_piece_length_l221_221619


namespace pentagon_coloring_count_l221_221676

-- Define the number of sides in the pentagon
def num_sides : ℕ := 5

-- Define the colors available for coloring
inductive Color
| red : Color
| yellow : Color
| blue : Color

-- Define the convex pentagon with unequal sides and a constraint that adjacent sides cannot be the same color
def is_valid_coloring (coloring : Fin num_sides → Color) : Prop :=
  ∀ i : Fin num_sides, coloring i ≠ coloring ((i + 1) % num_sides)

-- Define the problem in terms of proving the total number of distinct valid coloring methods
theorem pentagon_coloring_count : ∃ n : ℕ, n = 72 ∧ ∀ coloring, is_valid_coloring coloring → n = 72 :=
by
  use 72
  sorry

end pentagon_coloring_count_l221_221676


namespace midpoint_pentagon_area_l221_221917

-- Let S be the function that computes the area of a pentagon
noncomputable def area (p : Polygon) := sorry

structure Point : Type :=
(x : ℝ)
(y : ℝ)

structure Pentagon :=
(A B C D E : Point)

-- Define a function to calculate the midpoint of two points
def midpoint (p1 p2 : Point) : Point :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

-- Define a function to create a pentagon from midpoints of another pentagon’s sides
def midpoint_pentagon (p : Pentagon) : Pentagon :=
{ A := midpoint p.A p.B,
  B := midpoint p.B p.C,
  C := midpoint p.C p.D,
  D := midpoint p.D p.E,
  E := midpoint p.E p.A }

-- Our formalized problem statement
theorem midpoint_pentagon_area (p : Pentagon) :
  area (midpoint_pentagon p) ≥ (1 / 2) * area p :=
sorry

end midpoint_pentagon_area_l221_221917


namespace quadratic_root_exists_l221_221411

theorem quadratic_root_exists (a b c : ℝ) : 
  ∃ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ∨ (b * x^2 + 2 * c * x + a = 0) ∨ (c * x^2 + 2 * a * x + b = 0) :=
by sorry

end quadratic_root_exists_l221_221411


namespace two_pow_p_plus_three_pow_p_not_nth_power_l221_221756

theorem two_pow_p_plus_three_pow_p_not_nth_power (p n : ℕ) (prime_p : Nat.Prime p) (one_lt_n : 1 < n) :
  ¬ ∃ k : ℕ, 2 ^ p + 3 ^ p = k ^ n :=
sorry

end two_pow_p_plus_three_pow_p_not_nth_power_l221_221756


namespace true_proposition_among_given_l221_221252

variables {a : ℝ} {x : ℝ}

def line1 (a : ℝ) := ∀ x y : ℝ, x + a * y + 1 = 0
def line2 (a : ℝ) := ∀ x y : ℝ, (a - 2) * x + 3 * y + 1 = 0

def proposition_p : Prop := (a = -1 ∨ a = 3) ∧ a ≠ 3
def proposition_q : Prop := ∀ x ∈ set.Ioo 0 1, x^2 - x < 0

def correct_proposition : Prop := (¬ proposition_p) ∧ proposition_q

theorem true_proposition_among_given : correct_proposition :=
by {
  sorry
}

end true_proposition_among_given_l221_221252


namespace distance_from_center_of_circle_to_line_l221_221325

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221325


namespace domain_lg_function_l221_221867

theorem domain_lg_function (x : ℝ) : (∃ y, f x = log y ∧ y > 0) ↔ x > 1 := by
  sorry

end domain_lg_function_l221_221867


namespace hexagon_diagonals_sum_l221_221048

def inscribed_hexagon_diagonals (R : Real) (hexagon : Hexagon) : Prop :=
  (hexagon.side_length BC = 85) ∧ 
  (hexagon.side_length CD = 85) ∧ 
  (hexagon.side_length DE = 85) ∧ 
  (hexagon.side_length EF = 85) ∧ 
  (hexagon.side_length FA = 85) ∧ 
  (hexagon.side_length AB = 40) ∧ 
  (hexagon.diagonal_sum A = 374)

theorem hexagon_diagonals_sum :
  ∃ (hexagon : Hexagon), inscribed_hexagon_diagonals R hexagon :=
sorry

end hexagon_diagonals_sum_l221_221048


namespace distance_from_center_of_circle_to_line_l221_221331

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221331


namespace problem_statement_l221_221438

theorem problem_statement (M N : ℕ) 
  (hM : M = 2020 / 5) 
  (hN : N = 2020 / 20) : 10 * M / N = 40 := 
by
  sorry

end problem_statement_l221_221438


namespace arrangement_plans_count_l221_221686

/-- 
Given 5 volunteers and the requirement to select 3 people to serve at 
the Swiss Pavilion, the Spanish Pavilion, and the Italian Pavilion such that
1. Each pavilion is assigned 1 person.
2. Individuals A and B cannot go to the Swiss Pavilion.
Prove that the total number of different arrangement plans is 36.
-/

theorem arrangement_plans_count (volunteers : Finset ℕ) (A B : ℕ) (Swiss Pavilion : ℕ) :
  volunteers.card = 5 →
  A ≠ Swiss →
  B ≠ Swiss →
  ∃ arrangements, arrangements.card = 36 :=
begin
  sorry
end

end arrangement_plans_count_l221_221686


namespace elvis_writing_time_per_song_l221_221653

-- Define the conditions based on the problem statement
def total_studio_time_minutes := 300   -- 5 hours converted to minutes
def songs := 10
def recording_time_per_song := 12
def total_editing_time := 30

-- Define the total recording time
def total_recording_time := songs * recording_time_per_song

-- Define the total time available for writing songs
def total_writing_time := total_studio_time_minutes - total_recording_time - total_editing_time

-- Define the time to write each song
def time_per_song_writing := total_writing_time / songs

-- State the proof goal
theorem elvis_writing_time_per_song : time_per_song_writing = 15 := by
  sorry

end elvis_writing_time_per_song_l221_221653


namespace distance_from_point_to_line_polar_is_one_l221_221788

open Real

-- Define the point in polar coordinates
def point_polar : ℝ × ℝ := (2, π / 3)

-- Define the line in polar coordinates
def line_polar (ρ θ : ℝ) := ρ * cos θ + ρ * sqrt 3 * sin θ = 6

-- Define the point in rectangular coordinates
def point_rectangular : ℝ × ℝ := (1, sqrt 3)

-- Define the line in rectangular form
def line_rectangular (x y : ℝ) := x + sqrt 3 * y - 6 = 0

-- Define the distance formula from a point to a line in rectangular coordinates
def distance_point_to_line (x1 y1 A B C : ℝ) : ℝ :=
  abs (A * x1 + B * y1 + C) / sqrt (A * A + B * B)

-- Prove the distance from point (2, π / 3) in polar coordinates 
-- to line ρ(cos θ + sqrt 3 sin θ) = 6 is 1
theorem distance_from_point_to_line_polar_is_one :
  let x1, y1 := point_rectangular in
  let (A, B, C) := (1, sqrt 3, -6) in
  distance_point_to_line x1 y1 A B C = 1 :=
by
  sorry

end distance_from_point_to_line_polar_is_one_l221_221788


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221118

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221118


namespace age_of_15th_student_l221_221486

theorem age_of_15th_student 
  (average_age_15 : ℕ → ℕ → ℕ)
  (average_age_5 : ℕ → ℕ → ℕ)
  (average_age_9 : ℕ → ℕ → ℕ)
  (h1 : average_age_15 15 15 = 15)
  (h2 : average_age_5 5 14 = 14)
  (h3 : average_age_9 9 16 = 16) :
  let total_age_15 := 15 * 15 in
  let total_age_5 := 5 * 14 in
  let total_age_9 := 9 * 16 in
  let combined_total_age := total_age_5 + total_age_9 in
  let age_15th_student := total_age_15 - combined_total_age in
  age_15th_student = 11 := 
by
  simp [total_age_15, total_age_5, total_age_9, combined_total_age, age_15th_student]
  exact eq.refl 11

end age_of_15th_student_l221_221486


namespace number_of_noncongruent_triangles_l221_221836

-- Definition of the points in an isosceles triangle with AB = AC
variables (A B C M N O : Type) [IsoscelesTriangle A B C] (hAB_AC : distance A B = distance A C)

-- Definitions of the midpoints
def is_midpoint (X Y Z : Type) : Prop := (distance X Z = distance Z Y)

-- Conditions for the midpoints
variables (hM : is_midpoint A B M) (hN : is_midpoint B C N) (hO : is_midpoint C A O)

-- Statement of the theorem
theorem number_of_noncongruent_triangles : 
  ∃ count : ℕ, count = 5 :=
by
  use 5
  sorry

end number_of_noncongruent_triangles_l221_221836


namespace incorrect_sin_l221_221170

noncomputable def correct_answer := "C"

theorem incorrect_sin : 
  ¬(∀ (α k : ℝ), k ≠ 0 → (3 * k, 4 * k) = (cos α, sin α) → sin α = 4 / 5) :=
by
  sorry

end incorrect_sin_l221_221170


namespace base10_equivalent_of_43210_7_l221_221536

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end base10_equivalent_of_43210_7_l221_221536


namespace max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l221_221673

noncomputable def y (x : ℝ) : ℝ := 3 * x + 4 / x
def max_value (x : ℝ) := y x ≤ -4 * Real.sqrt 3

theorem max_y_value_of_3x_plus_4_div_x (h : x < 0) : max_value x :=
sorry

theorem corresponds_value_of_x (x : ℝ) (h : x = -2 * Real.sqrt 3 / 3) : y x = -4 * Real.sqrt 3 :=
sorry

end max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l221_221673


namespace base10_equivalent_of_43210_7_l221_221535

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end base10_equivalent_of_43210_7_l221_221535


namespace cut_isosceles_right_triangle_l221_221986

theorem cut_isosceles_right_triangle :
  ∃ (triangles : list (ℝ × ℝ × ℝ)), 
    (∀ t ∈ triangles, t.1 = t.2 ∧ t.3 = t.1 * real.sqrt 2) ∧
    (∀ (t1 t2 : ℝ × ℝ × ℝ), t1 ∈ triangles → t2 ∈ triangles → t1 ≠ t2 → t1.1 / t2.1 ≠ 1) :=
sorry

end cut_isosceles_right_triangle_l221_221986


namespace matrix_90_degree_rotation_l221_221646

theorem matrix_90_degree_rotation :
  ∃ (a b : ℚ), (a = 1 / 3 ∧ b = -5 / 4) ∧ 
    (
      let R := Matrix.vec_2_d (λ i j, if (i, j) = (0, 0) then a
                                 else if (i, j) = (0, 1) then b
                                 else if (i, j) = (1, 0) then 3 / 4
                                 else -1 / 4)
      in R.mul R = -1 • (1 : Matrix (Fin 2) (Fin 2) ℚ)
    ) := sorry

end matrix_90_degree_rotation_l221_221646


namespace distinct_sequences_count_l221_221743

theorem distinct_sequences_count :
  let letters := ['A', 'P', 'P', 'L', 'E']
  in (count the number of 5-letter sequences that start with 'A' and do not end with 'E' from the letters)
  = 15 :=
by
  -- Proof to be filled in later
  sorry

end distinct_sequences_count_l221_221743


namespace floor_sqrt_eq_l221_221474

theorem floor_sqrt_eq (n : ℤ) : 
  (⌊real.sqrt n + real.sqrt (n + 1)⌋ : ℤ) = (⌊real.sqrt (4 * n + 2)⌋ : ℤ) :=
sorry

end floor_sqrt_eq_l221_221474


namespace common_chord_length_l221_221522

-- Definitions for our problem conditions
def circle (radius : ℝ) (center : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | dist p center = radius}

def circles_overlap (c1 c2 : ℝ × ℝ) (r : ℝ) : Prop :=
  dist c1 c2 = r

-- Main Theorem
theorem common_chord_length (r : ℝ) (h : r = 15) (c1 c2 : ℝ × ℝ)
  (overlap : circles_overlap c1 c2 r) :
  ∃ l, l = 15 * real.sqrt 3 :=
by
  sorry

end common_chord_length_l221_221522


namespace ratio_of_areas_of_octagons_l221_221085

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221085


namespace base10_equivalent_of_43210_7_l221_221534

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end base10_equivalent_of_43210_7_l221_221534


namespace geometric_sequence_sum_log_k_l221_221447

theorem geometric_sequence_sum_log_k (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = 2 * a n)
  (h3 : S k = a 0 + a 1 + ... + a (k - 1)) 
  (h4 : log 4 (S k + 1) = 2) :
  k = 4 := 
sorry

end geometric_sequence_sum_log_k_l221_221447


namespace distance_from_center_of_circle_to_line_l221_221323

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221323


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221116

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221116


namespace statement_A_statement_B_statement_C_statement_D_l221_221692

variables {z1 z2 : ℂ}

-- 1. Prove that if \( z_1 = \overline{z_2} \), then \( \overline{z_1} = z_2 \).
theorem statement_A (h: z1 = conj(z2)) : conj(z1) = z2 :=
by { sorry }

-- 2. Disprove that if \( z_1 + z_2 \in \mathbb{R} \), then the imaginary parts of \( z_1 \) and \( z_2 \) are equal.
theorem statement_B (h: z1 + z2 ∈ ℝ) : (z1.im = z2.im) :=
by { have : (im(z1) + im(z2) = 0) := sorry, sorry }

-- 3. Prove that if \( z_1z_2 = 0 \), then \( z_1 = 0 \) or \( z_2 = 0 \).
theorem statement_C (h: z1 * z2 = 0) : z1 = 0 ∨ z2 = 0 :=
by { sorry }

-- 4. Disprove that if \( z_1^2 + z_2^2 = 0 \), then \( z_1 = z_2 = 0 \).
theorem statement_D (h: z1^2 + z2^2 = 0) : z1 = 0 ∧ z2 = 0 :=
by { sorry }

end statement_A_statement_B_statement_C_statement_D_l221_221692


namespace table_tennis_team_arrangements_l221_221389

theorem table_tennis_team_arrangements :
  let number_of_players := 10
  let main_players := 3
  let positions := 5
  (main_positions := [1, 3, 5] : List Nat) →
  let remaining_positions := 2
  let remaining_players := number_of_players - main_players
  Nat.choose remaining_players remaining_positions * Finset.permMultiset (Finset.range main_players) = 252 :=
begin
  sorry
end

end table_tennis_team_arrangements_l221_221389


namespace sin_theta_plus_pi_over_4_l221_221265

noncomputable def sin_sum_angle : Real :=
  let x := -3
  let y := 4
  let r := Real.sqrt (x*x + y*y)
  let sinθ := y / r
  let cosθ := x / r
  Real.sin (θ + π / 4)

theorem sin_theta_plus_pi_over_4 :
  sin_sum_angle (-3) (4) = Real.sqrt 2 / 10 :=
by
  sorry

end sin_theta_plus_pi_over_4_l221_221265


namespace magnitude_w_l221_221422

open Complex

def z : ℂ := ((5 : ℂ) - 3 * I)^2 * ((15 : ℂ) + 4 * I)^3 / (6 - 8 * I)
def w : ℂ := conj z / z

theorem magnitude_w : abs w = 1 :=
by
  sorry

end magnitude_w_l221_221422


namespace northwest_molded_break_even_price_l221_221830

theorem northwest_molded_break_even_price :
  ∀ (variable_cost_per_handle : ℝ) (fixed_cost_per_week : ℝ) (handles_needed_to_break_even : ℕ),
  variable_cost_per_handle = 0.60 →
  fixed_cost_per_week = 7640 →
  handles_needed_to_break_even = 1910 →
  let total_cost := (variable_cost_per_handle * (handles_needed_to_break_even : ℝ)) + fixed_cost_per_week in
  let selling_price_per_handle := total_cost / (handles_needed_to_break_even : ℝ) in
  selling_price_per_handle ≈ 4.60 :=
by
  intros variable_cost_per_handle fixed_cost_per_week handles_needed_to_break_even
  sorry

end northwest_molded_break_even_price_l221_221830


namespace distance_to_line_is_constant_l221_221352

noncomputable def center_of_circle (a : ℝ) : (ℝ × ℝ) := (a, a)

noncomputable def circle_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = a^2

noncomputable def distance_from_center_to_line (x1 y1 : ℝ) : ℝ :=
  abs (2 * x1 - y1 - 3) / sqrt (2^2 + (-1)^2)

theorem distance_to_line_is_constant (a : ℝ) (h : circle_equation a 2 1) (hx : a = 5 ∨ a = 1) :
  distance_from_center_to_line a a = 2 * sqrt 5 / 5 :=
by
  sorry

end distance_to_line_is_constant_l221_221352


namespace exists_real_A_distance_two_l221_221462

noncomputable theory

def t : ℝ := 5 + 2 * real.sqrt 6

def A : ℝ := t^2

theorem exists_real_A_distance_two (n : ℕ) : 
  ∃ (A : ℝ), (A = t^2) → (∃ (m : ℤ), 0 ≤ m ∧ m * m = int.ceil (A^n) - 2 ∧ dist (int.ceil (A^n)) (m * m) = 2) :=
by
  sorry

end exists_real_A_distance_two_l221_221462


namespace triangle_stability_l221_221175

/-!
# Triangle Stability Proof

Given that triangular structures are used for strength in bridges, cable car supports, and trusses, 
the mathematical principle that justifies this choice is the stability of triangles.
-/

theorem triangle_stability (bridges_car_trusses_use_triangles : ∀ {S : Type}, triangular_structure S) :
  (∀ {G : Type}, geometric_stability G → triangles_stability) := 
sorry

end triangle_stability_l221_221175


namespace find_area_of_oblique_triangle_l221_221257

noncomputable def area_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin C

theorem find_area_of_oblique_triangle
  (A B C a b c : ℝ)
  (h1 : c = Real.sqrt 21)
  (h2 : c * Real.sin A = Real.sqrt 3 * a * Real.cos C)
  (h3 : Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A))
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum_ABC : A + B + C = Real.pi)
  (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (tri_angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) :
  area_triangle a b c A B C = 5 * Real.sqrt 3 / 4 := 
sorry

end find_area_of_oblique_triangle_l221_221257


namespace ratio_area_octagons_correct_l221_221112

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221112


namespace circles_internally_tangent_l221_221882

/-- Define the first circle -/
def circle1_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

/-- Define the second circle -/
def circle2_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y - 24 = 0

/-- Define the center and radius of the first circle -/
def center1 : (ℝ × ℝ) := (0, 0)
def radius1 : ℝ := 2

/-- Define the center and radius of the second circle -/
def center2 : (ℝ × ℝ) := (3, -4)
def radius2 : ℝ := 7

noncomputable def distance_between_centers : ℝ :=
  real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

theorem circles_internally_tangent :
  distance_between_centers = radius2 - radius1 :=
by 
  sorry

end circles_internally_tangent_l221_221882


namespace other_liquid_cost_l221_221577

-- Definitions based on conditions
def total_fuel_gallons : ℕ := 12
def fuel_price_per_gallon : ℝ := 8
def oil_price_per_gallon : ℝ := 15
def fuel_cost : ℝ := total_fuel_gallons * fuel_price_per_gallon
def other_liquid_price_per_gallon (x : ℝ) : Prop :=
  (7 * x + 5 * oil_price_per_gallon = fuel_cost) ∨
  (7 * oil_price_per_gallon + 5 * x = fuel_cost)

-- Question: The cost of the other liquid per gallon
theorem other_liquid_cost :
  ∃ x, other_liquid_price_per_gallon x ∧ x = 3 :=
sorry

end other_liquid_cost_l221_221577


namespace clock_angles_3_20_l221_221007

theorem clock_angles_3_20 :
  let full_revolution := 360.0
  let initial_angle := full_revolution / 4
  let minute_hand_move := full_revolution / 60
  let hour_hand_move := full_revolution / (12 * 60)
  let rate_of_change := minute_hand_move - hour_hand_move
  let minutes_passed := 20
  let angle_change := rate_of_change * minutes_passed
  let new_angle := initial_angle + angle_change
  let smaller_angle := if new_angle < full_revolution - new_angle then new_angle else full_revolution - new_angle
  let larger_angle := if new_angle > full_revolution - new_angle then new_angle else full_revolution - new_angle
  in smaller_angle = 160.0 ∧ larger_angle = 200.0 :=
by
  sorry

end clock_angles_3_20_l221_221007


namespace jan_drives_more_miles_than_ian_l221_221565

-- Definitions of conditions
variables (s t d m: ℝ)

-- Ian's travel equation
def ian_distance := d = s * t

-- Han's travel equation
def han_distance := (d + 115) = (s + 8) * (t + 2)

-- Jan's travel equation
def jan_distance := m = (s + 12) * (t + 3)

-- The proof statement we want to prove
theorem jan_drives_more_miles_than_ian :
    (∀ (s t d m : ℝ),
    d = s * t →
    (d + 115) = (s + 8) * (t + 2) →
    m = (s + 12) * (t + 3) →
    (m - d) = 184.5) :=
    sorry

end jan_drives_more_miles_than_ian_l221_221565


namespace factor_property_l221_221302

noncomputable def P_and_Q : ℤ × ℤ :=
let b := -4 in
let c := 3 in
let P := 4 * (-4) + c + 3 in
let Q := 3 * c in
(P, Q)

theorem factor_property : 
  ∃ P Q : ℤ, (x^2 + 4*x + 3) ∣ (x^4 + P*x^2 + Q) ∧ P + Q = -1 := 
  begin
    use [(-10 : ℤ), (9 : ℤ)],
    split,
    { sorry }, -- Proof that (x^2 + 4*x + 3) is a factor of (x^4 + (-10)*x^2 + 9)
    { refl } -- Proof that P + Q = -1
  end

end factor_property_l221_221302


namespace circle_distance_to_line_l221_221370

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221370


namespace mrs_brown_shoes_price_l221_221413

def discount (price : ℝ) (percent : ℝ) : ℝ := price * percent / 100
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let price_after_first_discount := original_price - discount original_price discount1
  let total_price := price_after_first_discount - discount price_after_first_discount discount2
  total_price

theorem mrs_brown_shoes_price:
  (original_price : ℝ) (num_children : ℕ) (mother_discount : ℝ)
  (children_discount : ℝ)
  (H1 : original_price = 125)
  (H2 : num_children = 4)
  (H3 : mother_discount = 10)
  (H4 : children_discount = 4) :
  final_price original_price mother_discount children_discount = 108 :=
by 
  rw [H1, H3, H4]
  simp only [final_price, discount]
  norm_num
  sorry

end mrs_brown_shoes_price_l221_221413


namespace ratio_of_areas_of_octagons_l221_221062

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221062


namespace maximum_volume_pyramid_is_one_sixteenth_l221_221490

open Real  -- Opening Real namespace for real number operations

noncomputable def maximum_volume_pyramid : ℝ :=
  let a := 1 -- side length of the equilateral triangle base
  let base_area := (sqrt 3 / 4) * (a * a) -- area of the equilateral triangle with side length 1
  let median := sqrt 3 / 2 * a -- median length of the triangle
  let height := 1 / 2 * median -- height of the pyramid
  let volume := 1 / 3 * base_area * height -- volume formula for a pyramid
  volume

theorem maximum_volume_pyramid_is_one_sixteenth :
  maximum_volume_pyramid = 1 / 16 :=
by
  simp [maximum_volume_pyramid] -- Simplify the volume definition
  sorry -- Proof omitted

end maximum_volume_pyramid_is_one_sixteenth_l221_221490


namespace nantucket_meeting_fraction_l221_221042

def nantucket_fraction : ℝ :=
  let total_people := 300
  let females_meeting := 50
  let males_meeting := 2 * females_meeting
  let total_meeting := females_meeting + males_meeting
  total_meeting / total_people

theorem nantucket_meeting_fraction : nantucket_fraction = 1 / 2 :=
by
  let total_people := 300
  let females_meeting := 50
  let males_meeting := 2 * females_meeting
  let total_meeting := females_meeting + males_meeting
  have : total_meeting = 150, by norm_num
  have : total_meeting / total_people = 1 / 2, by norm_num
  exact this

end nantucket_meeting_fraction_l221_221042


namespace mia_weight_l221_221506

theorem mia_weight (a m : ℝ) (h1 : a + m = 220) (h2 : m - a = 2 * a) : m = 165 :=
sorry

end mia_weight_l221_221506


namespace octagon_area_ratio_l221_221127

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221127


namespace sum_of_circle_center_coordinates_eq_minus_one_l221_221194

theorem sum_of_circle_center_coordinates_eq_minus_one 
  (x y : ℝ) 
  (h : x^2 + y^2 = 6*x - 8*y + 24) : 
  (let (a, b) := (3, -4) in a + b) = -1 := 
by 
  -- Define the conditions here
  sorry

end sum_of_circle_center_coordinates_eq_minus_one_l221_221194


namespace chess_tournament_l221_221936

theorem chess_tournament :
  ∀ (n : ℕ), (∃ (players : ℕ) (total_games : ℕ),
  players = 8 ∧ total_games = 56 ∧ total_games = (players * (players - 1) * n) / 2) →
  n = 2 :=
by
  intros n h
  rcases h with ⟨players, total_games, h_players, h_total_games, h_eq⟩
  have := h_eq
  sorry

end chess_tournament_l221_221936


namespace solve_expression_l221_221203

theorem solve_expression (a x : ℝ) (h1 : a ≠ 0) (h2 : x ≠ a) : 
  (a / (2 * a + x) - x / (a - x)) / (x / (2 * a + x) + a / (a - x)) = -1 → 
  x = a / 2 :=
by
  sorry

end solve_expression_l221_221203


namespace find_values_of_ABC_l221_221889

-- Define the given conditions
def condition1 (A B C : ℕ) : Prop := A + B + C = 36
def condition2 (A B C : ℕ) : Prop := 
  (A + B) * 3 * 4 = (B + C) * 2 * 4 ∧ 
  (B + C) * 2 * 4 = (A + C) * 2 * 3

-- State the problem
theorem find_values_of_ABC (A B C : ℕ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B C) : 
  A = 12 ∧ B = 4 ∧ C = 20 :=
sorry

end find_values_of_ABC_l221_221889


namespace find_ellipse_params_l221_221962

theorem find_ellipse_params :
  ∃ (a b h k : ℝ), 
  (a > 0) ∧ (b > 0) ∧
  h = 1 ∧ k = 5 ∧
  a = (Real.sqrt 130 + Real.sqrt 170) / 2 ∧
  b = Real.sqrt (((Real.sqrt 130 + Real.sqrt 170) / 2)^2 - 4^2) ∧
  (∀ x y : ℝ, (x - 1)^2 / a^2 + (y - 5)^2 / b^2 = 1 → 
    (x = 12 ∧ y = 0)) :=
begin
  sorry
end

end find_ellipse_params_l221_221962


namespace chocolates_not_in_box_initially_l221_221453

theorem chocolates_not_in_box_initially 
  (total_chocolates : ℕ) 
  (chocolates_friend_brought : ℕ) 
  (initial_boxes : ℕ) 
  (additional_boxes : ℕ)
  (total_after_friend : ℕ)
  (chocolates_each_box : ℕ)
  (total_chocolates_initial : ℕ) :
  total_chocolates = 50 ∧ initial_boxes = 3 ∧ chocolates_friend_brought = 25 ∧ total_after_friend = 75 
  ∧ additional_boxes = 2 ∧ chocolates_each_box = 15 ∧ total_chocolates_initial = 50
  → (total_chocolates_initial - (initial_boxes * chocolates_each_box)) = 5 :=
by
  sorry

end chocolates_not_in_box_initially_l221_221453


namespace poly_coeff_sum_zero_l221_221280

theorem poly_coeff_sum_zero :
  let p := (2 * x^2 - 3 * x + 5) * (9 - 3 * x)
  ∃ a b c d, (p = a * x^3 + b * x^2 + c * x + d) ∧ (27 * a + 9 * b + 3 * c + d = 0) :=
by
  let p := (2 * x^2 - 3 * x + 5) * (9 - 3 * x)
  have h_eq : p = -6 * x^3 + 27 * x^2 - 42 * x + 45 := sorry
  use -6, 27, -42, 45
  finish

end poly_coeff_sum_zero_l221_221280


namespace find_all_values_z_l221_221661

theorem find_all_values_z :
  ∃ S : set ℝ, S = {-real.sqrt 3, -1, 1, real.sqrt 3} ∧ ∀ z : ℝ, z^4 - 4 * z^2 + 3 = 0 ↔ z ∈ S :=
by
  sorry

end find_all_values_z_l221_221661


namespace distance_from_center_to_line_of_tangent_circle_l221_221320

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221320


namespace no_factors_multiple_of_210_l221_221304

theorem no_factors_multiple_of_210 (n : ℕ) (h : n = 2^12 * 3^18 * 5^10) : ∀ d : ℕ, d ∣ n → ¬ (210 ∣ d) :=
by
  sorry

end no_factors_multiple_of_210_l221_221304


namespace distance_from_center_to_line_l221_221365

noncomputable def circle_center_is_a (a : ℝ) : Prop :=
  (2 - a)^2 + (1 - a)^2 = a^2

theorem distance_from_center_to_line (a : ℝ) (h : 0 < a) (hab : circle_center_is_a a) :
  (∀x0 y0 : ℝ, (x0, y0) = (a, a) → (|2 * x0 - y0 - 3| / Real.sqrt (2^2 + 1^2)) = (2 * Real.sqrt 5 / 5)) :=
by
  sorry

end distance_from_center_to_line_l221_221365


namespace intersecting_chords_theorem_l221_221249

theorem intersecting_chords_theorem
  (Γ₁ Γ₂ Γ₃ Γ₄ : Circle)
  (P A B C D : Point)
  (h1 : Γ₁.TangentAt P Γ₃)
  (h2 : Γ₂.TangentAt P Γ₄)
  (h₃ : Γ₁.IntersectAt A Γ₂)
  (h₄ : Γ₂.IntersectAt B Γ₃)
  (h₅ : Γ₃.IntersectAt C Γ₄)
  (h₆ : Γ₄.IntersectAt D Γ₁)
  (hPA : A ≠ P)
  (hPB : B ≠ P)
  (hPC : C ≠ P)
  (hPD : D ≠ P) :
  (dist A B * dist B C) / (dist A D * dist D C) = (dist P B)^2 / (dist P D)^2 := 
sorry

end intersecting_chords_theorem_l221_221249


namespace circle_distance_to_line_l221_221372

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221372


namespace Mrs_Brown_pays_108_for_shoes_l221_221415

def discounted_price (original price discount: Float): Float :=
  original - (original * (discount / 100))

def final_price (original_price: Float) (discount1 discount2: Float): Float :=
    let price_after_first_discount := discounted_price original_price discount1
    discounted_price price_after_first_discount discount2

theorem Mrs_Brown_pays_108_for_shoes :
  (number_of_children >= 3) → (original_price = 125) → (discount1 = 10) → (discount2 = 4) →
  final_price original_price discount1 discount2 = 108 := 
by
  intros h_children h_price h_discount1 h_discount2
  sorry

end Mrs_Brown_pays_108_for_shoes_l221_221415


namespace greatest_x_solution_l221_221666

theorem greatest_x_solution (x : ℝ) (h₁ : (x^2 - x - 30) / (x - 6) = 2 / (x + 4)) : x ≤ -3 :=
sorry

end greatest_x_solution_l221_221666


namespace number_of_outcomes_exactly_two_evening_l221_221681

theorem number_of_outcomes_exactly_two_evening (chickens : Finset ℕ) (h_chickens : chickens.card = 4) 
    (day_places evening_places : ℕ) (h_day_places : day_places = 2) (h_evening_places : evening_places = 3) :
    ∃ n, n = (chickens.card.choose 2) ∧ n = 6 :=
by
  sorry

end number_of_outcomes_exactly_two_evening_l221_221681


namespace volume_proof_l221_221224

variables (m n p d x V : Real)

namespace VolumeProof

-- Define the conditions
def diag_eq : Prop :=
  d^2 = (m * x)^2 + (n * x)^2 + (p * x)^2

def x_val : Prop :=
  x = d / sqrt(m^2 + n^2 + p^2)

-- Define the volume formula to be proven
def volume_formula : Prop :=
  V = (m * n * p * (d / sqrt(m^2 + n^2 + p^2))^3)

-- The main statement to be proven
theorem volume_proof 
  (h1 : diag_eq) 
  (h2 : x_val) 
  : volume_formula :=
  sorry

end VolumeProof

end volume_proof_l221_221224


namespace octagon_area_ratio_l221_221133

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221133


namespace jason_initial_speed_correct_l221_221793

noncomputable def jason_initial_speed (d : ℝ) (t1 t2 : ℝ) (v2 : ℝ) : ℝ :=
  let t_total := t1 + t2
  let d2 := v2 * t2
  let d1 := d - d2
  let v1 := d1 / t1
  v1

theorem jason_initial_speed_correct :
  jason_initial_speed 120 0.5 1 90 = 60 := 
by 
  sorry

end jason_initial_speed_correct_l221_221793


namespace solve_integral_problem_l221_221510

noncomputable def integral_problem : Prop :=
  ∫ x in 0 .. 1, (x^2 + Real.exp x - 1/3) = Real.exp 1 - 1

theorem solve_integral_problem : integral_problem := by
  sorry

end solve_integral_problem_l221_221510


namespace octagon_area_ratio_l221_221149

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221149


namespace smallest_m_last_three_digits_l221_221434

theorem smallest_m_last_three_digits : 
  ∃ m : ℕ, m > 0 ∧ (6 ∣ m) ∧ (8 ∣ m) ∧ 
  (∀ d ∈ (m.digits 10), d = 2 ∨ d = 7) ∧
  (2 ∈ m.digits 10) ∧ (7 ∈ m.digits 10) ∧
  (m.to_digits.nats.take_last_element 3 = [7, 2, 2]) := sorry

end smallest_m_last_three_digits_l221_221434


namespace coefficient_of_x5_in_expansion_l221_221401

-- Define the given polynomial expression
def polynomial := (λ (x : ℕ), x ^ 4 + 1 / (x ^ 2) + 2 * x)

-- Define the specific exponent expansion and coefficient we are looking for
def target_expansion := 5

-- State the theorem
theorem coefficient_of_x5_in_expansion : 
  coefficient_of_x_in_expansion (polynomial x) target_expansion = 252 :=
by
  sorry

end coefficient_of_x5_in_expansion_l221_221401


namespace area_ratio_of_octagons_is_4_l221_221096

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221096


namespace independence_iff_l221_221031

variable {Ω : Type*} {F : measurable_space Ω}
variable {ξ : ℕ → Ω → ℕ} -- discrete random variables
variable [probability_space Ω] -- probability space

def independent (ξ : ℕ → Ω → ℕ) : Prop :=
  ∀ (n : ℕ) (A : fin n → set Ω), prob (⋂ i, ξ i ⁻¹' A i) = ∏ i, prob (ξ i ⁻¹' A i)

def joint_probability {Ω} [probability_space Ω] (ξ : ℕ → Ω → ℕ) (x : ℕ → ℕ) : ℝ :=
  prob (⋂ i, {ω : Ω | ξ i ω = x i})

theorem independence_iff (ξ : ℕ → Ω → ℕ) :
  (∀ x : ℕ → ℕ, joint_probability ξ x = ∏ i, prob {ω | ξ i ω = x i}) ↔ independent ξ :=
by sorry

end independence_iff_l221_221031


namespace calculate_g3_l221_221758

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem calculate_g3 : g 3 = 3 / 17 :=
by {
    -- Here we add the proof steps if necessary, but for now we use sorry
    sorry
}

end calculate_g3_l221_221758


namespace area_ratio_is_correct_l221_221136

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221136


namespace paper_cups_pallets_l221_221162

theorem paper_cups_pallets (total_pallets : ℕ) (paper_towels_fraction tissues_fraction paper_plates_fraction : ℚ) :
  total_pallets = 20 → paper_towels_fraction = 1 / 2 → tissues_fraction = 1 / 4 → paper_plates_fraction = 1 / 5 →
  total_pallets - (total_pallets * paper_towels_fraction + total_pallets * tissues_fraction + total_pallets * paper_plates_fraction) = 1 :=
by sorry

end paper_cups_pallets_l221_221162


namespace correct_calculation_l221_221557

theorem correct_calculation (x y : ℝ) : 
  ¬(2 * x^2 + 3 * x^2 = 6 * x^2) ∧ 
  ¬(x^4 * x^2 = x^8) ∧ 
  ¬(x^6 / x^2 = x^3) ∧ 
  ((x * y^2)^2 = x^2 * y^4) :=
by
  sorry

end correct_calculation_l221_221557


namespace part1_part2_l221_221571

variables {x y z : ℝ}

-- x, y, z are positive real numbers
-- xy + yz + zx ≠ 1
-- (x^2 - 1)(y^2 - 1) / xy + (y^2 - 1)(z^2 - 1) / yz + (z^2 - 1)(x^2 - 1) / zx = 4

def xy_neq_1 (x y z : ℝ) : Prop := xy + yz + zx ≠ 1

def given_condition (x y z : ℝ) : Prop :=
  (x^2 - 1)*(y^2 - 1) / (xy) + 
  (y^2 - 1)*(z^2 - 1) / (yz) + 
  (z^2 - 1)*(x^2 - 1) / (zx) = 4

theorem part1 (h1 : xy_neq_1 x y z) (h2 : given_condition x y z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / xy + 1 / yz + 1 / zx) = 1 :=
sorry

theorem part2 (h1 : xy_neq_1 x y z) (h2 : given_condition x y z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  9*(x + y)*(y + z)*(z + x) ≥ 8*xyz*(xy + yz + zx) :=
sorry

end part1_part2_l221_221571


namespace digit_divisible_by_11_l221_221001

theorem digit_divisible_by_11 (B : ℕ) : 
  (∃ (B : ℕ), B = 5) ∧ (let odd_sum := B + 7
                         let even_sum := 12
                         abs (odd_sum - even_sum) % 11 = 0) :=
by
  existsi 5
  split
  sorry

end digit_divisible_by_11_l221_221001


namespace locus_of_constant_angle_arc_l221_221011

theorem locus_of_constant_angle_arc (A B : Point) (theta : ℝ) :
  exists C₁ C₂ : Circle, 
  C₁.chord = A + B ∧ C₂.chord = A + B ∧
  symmetric C₁ C₂ ∧ ∀ P, P ∈ C₁ ∨ P ∈ C₂ ↔ ∠APB = theta :=
sorry

end locus_of_constant_angle_arc_l221_221011


namespace prob_s25_move_to_s40_l221_221979

def bubble_pass_probability : ℚ :=
  let p := 1 in
  let q := 1640 in
  p / q

theorem prob_s25_move_to_s40
  (s : ℕ → ℝ) (h_distinct : ∀ i j, i ≠ j → s i ≠ s j) (h_random : random_order s)
  (h_initial_pos : ∀ i, 1 ≤ i ∧ i ≤ 50)
  (h_target_pos : s 25 = s 40)
  (P : ℚ := bubble_pass_probability)
  (h_coprime : nat.coprime 1 1640) :
  P = 1 / 1640 ∧ P.num + P.denom = 1641 :=
by sorry

end prob_s25_move_to_s40_l221_221979


namespace distance_between_points_l221_221541

theorem distance_between_points :
  let x1 := 2
  let y1 := -2
  let x2 := 8
  let y2 := 8
  let dist := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  dist = Real.sqrt 136 :=
by
  -- Proof to be filled in here.
  sorry

end distance_between_points_l221_221541


namespace reduced_price_per_dozen_is_approx_2_95_l221_221020

noncomputable def original_price : ℚ := 16 / 39
noncomputable def reduced_price := 0.6 * original_price
noncomputable def reduced_price_per_dozen := reduced_price * 12

theorem reduced_price_per_dozen_is_approx_2_95 :
  abs (reduced_price_per_dozen - 2.95) < 0.01 :=
by
  sorry

end reduced_price_per_dozen_is_approx_2_95_l221_221020


namespace not_every_constant_is_geometric_l221_221501

def is_constant_sequence (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, s n = s m

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem not_every_constant_is_geometric :
  (¬ ∀ s : ℕ → ℝ, is_constant_sequence s → is_geometric_sequence s) ↔
  ∃ s : ℕ → ℝ, is_constant_sequence s ∧ ¬ is_geometric_sequence s := 
by
  sorry

end not_every_constant_is_geometric_l221_221501


namespace circle_tangent_distance_l221_221311

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221311


namespace candle_burning_time_l221_221521

theorem candle_burning_time (ℓ : ℝ) (t : ℝ) 
  (h₁ : t = 300) 
  (h₂ : t = 180) 
  (burn_time_diff : 6 * 60 - tℓ = 3 * (6 * 60 - tℓ/180))
  : (6 - 3.5) ℝ = 3.5 ℝ :=
sorry

end candle_burning_time_l221_221521


namespace wrapping_paper_fraction_used_l221_221842

theorem wrapping_paper_fraction_used 
  (total_paper_used : ℚ)
  (num_presents : ℕ)
  (each_present_used : ℚ)
  (h1 : total_paper_used = 1 / 2)
  (h2 : num_presents = 5)
  (h3 : each_present_used = total_paper_used / num_presents) : 
  each_present_used = 1 / 10 := 
by
  sorry

end wrapping_paper_fraction_used_l221_221842


namespace mrs_brown_shoes_price_l221_221414

def discount (price : ℝ) (percent : ℝ) : ℝ := price * percent / 100
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let price_after_first_discount := original_price - discount original_price discount1
  let total_price := price_after_first_discount - discount price_after_first_discount discount2
  total_price

theorem mrs_brown_shoes_price:
  (original_price : ℝ) (num_children : ℕ) (mother_discount : ℝ)
  (children_discount : ℝ)
  (H1 : original_price = 125)
  (H2 : num_children = 4)
  (H3 : mother_discount = 10)
  (H4 : children_discount = 4) :
  final_price original_price mother_discount children_discount = 108 :=
by 
  rw [H1, H3, H4]
  simp only [final_price, discount]
  norm_num
  sorry

end mrs_brown_shoes_price_l221_221414


namespace triangle_existence_l221_221639

theorem triangle_existence (K c : ℝ) (γ : ℝ) :
  K > c ∧ γ < 180 ∧ K > 2 * c → ∃ (a b : ℝ), a + b = K - c ∧ γ = angle_opposite c a b :=
sorry

end triangle_existence_l221_221639


namespace day_after_73_days_from_monday_is_thursday_l221_221904

def days_of_week : Type := {d // d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6}

def Monday : days_of_week := ⟨0, by { left, refl }⟩ 
def Thursday : days_of_week := ⟨3, by { right, left, refl }⟩

def day_after_n_days (start: days_of_week) (n: ℕ) : days_of_week :=
  ⟨(start.val + n) % 7, by sorry⟩

theorem day_after_73_days_from_monday_is_thursday :
  day_after_n_days Monday 73 = Thursday :=
begin
  -- proof would go here
  sorry
end

end day_after_73_days_from_monday_is_thursday_l221_221904


namespace sum_first_40_terms_l221_221699

def sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n + (-1)^(n+1) * a (n+1) = 2 * n - 1

theorem sum_first_40_terms (a : ℕ → ℤ) (h : sequence a) :
  (∑ k in finset.range 40, a k) = 780 := by
  sorry

end sum_first_40_terms_l221_221699


namespace mitya_age_l221_221828

-- Definitions of the ages
variables (M S : ℕ)

-- Conditions based on the problem statements
axiom condition1 : M = S + 11
axiom condition2 : S = 2 * (S - (M - S))

-- The theorem stating that Mitya is 33 years old
theorem mitya_age : M = 33 :=
by
  -- Outline the proof
  sorry

end mitya_age_l221_221828


namespace age_difference_l221_221493

theorem age_difference (a b : ℕ) 
  (h1 : ∃ a b, a < 10 ∧ b < 10 ∧ (10 * a + b - (10 * b + a) = 45))
  (h2 : ∃ a b, (10 * a + b + 5 = 3 * (10 * b + a + 5))) : 
  10 * a + b - (10 * b + a) = 45 :=
by
  have hab1 : 10 * a + b - (10 * b + a) = 9 * (a - b) := by
     sorry
  have h : (10 * a + b + 5) = 3 * (10 * b + a + 5) := by
    sorry
  have hab2 : (10 * a + b + 5 = 3 * (10 * b + a + 5)) := by
    sorry
  assert h3 : 7 * a - 29 * b = 10 := by
    sorry
  have hb :  ∃ a b, a = 6 ∧ b = 1 := by
    sorry
  assumption sorry
  

end age_difference_l221_221493


namespace total_sand_arrived_l221_221515

theorem total_sand_arrived :
  let truck1_carry := 4.1
  let truck1_loss := 2.4
  let truck2_carry := 5.7
  let truck2_loss := 3.6
  let truck3_carry := 8.2
  let truck3_loss := 1.9
  (truck1_carry - truck1_loss) + 
  (truck2_carry - truck2_loss) + 
  (truck3_carry - truck3_loss) = 10.1 :=
by
  sorry

end total_sand_arrived_l221_221515


namespace pi_equality_l221_221428

-- Define π(n) as the product of the positive divisors of n
def pi (n : Nat) : Nat := n.divisors.to_list.prod

-- The theorem to be proven
theorem pi_equality (m n : Nat) (h : pi m = pi n) : m = n :=
sorry

end pi_equality_l221_221428


namespace cats_more_than_spinsters_l221_221885

def ratio (a b : ℕ) := ∃ k : ℕ, a = b * k

theorem cats_more_than_spinsters (S C : ℕ) (h1 : ratio 2 9) (h2 : S = 12) (h3 : 2 * C = 108) :
  C - S = 42 := by 
  sorry

end cats_more_than_spinsters_l221_221885


namespace A_wins_3_1_probability_l221_221026

noncomputable def probability_A_wins_3_1 (p : ℚ) : ℚ :=
  let win_3_1 := binomial 4 3 * (p^3) * (1 - p)
  win_3_1

theorem A_wins_3_1_probability : probability_A_wins_3_1 (2/3) = 8/27 := by
  sorry

end A_wins_3_1_probability_l221_221026


namespace calculate_water_usage_l221_221384

def water_fee (x : ℝ) : ℝ :=
  if h₁ : 0 ≤ x ∧ x ≤ 4/5 then 14.4 * x
  else if h₂ : 4/5 < x ∧ x ≤ 4/3 then 20.4 * x - 4.8
  else 24 * x - 9.6

theorem calculate_water_usage (x : ℝ) (y : ℝ) :
  (y = water_fee x ∧ y = 26.4) →
  (x = 1.5 ∧ 5*x = 7.5 ∧ 3*x = 4.5) ∧
  (4 * 1.8 + 3.5 * 3 = 17.7 ∧ 8.7 + 4/4 ∈ ℝ ∧ 5*x = 7.5 ∧ 3*x = 4.5) ∧
  water_fee x = 26.4 := by sorry

end calculate_water_usage_l221_221384


namespace triangle_existence_and_uniqueness_triangle_isosceles_no_triangle_exists_l221_221250

/- Definitions for the problem setup -/

noncomputable def Triangle := Type -- Type representing a triangle

structure TriangleData where
  a : ℝ  -- given side length BC
  f_a : ℝ  -- length of angle bisector AA'
  m_a : ℝ  -- length of altitude AA_0

def isTriangle (ABC : Triangle) : TriangleData → Prop := sorry

/-- Main proposition to prove: Existence of a unique triangle under given conditions --/
theorem triangle_existence_and_uniqueness (a f_a m_a : ℝ) (h1 : f_a > m_a) :
  ∃ (ABC : Triangle),
    (isTriangle ABC { a := a, f_a := f_a, m_a := m_a }) ∧
    (∀ ABC' : Triangle,
      (isTriangle ABC' { a := a, f_a := f_a, m_a := m_a }) → ABC = ABC') :=
  sorry

/-- Proposition for the isosceles condition when f_a = m_a --/
theorem triangle_isosceles (a f_a m_a : ℝ) (h1 : f_a = m_a) :
  ∃ (ABC : Triangle), isTriangle ABC { a := a, f_a := f_a, m_a := m_a } ∧ isIsosceles ABC :=
  sorry

/-- Proposition for the non-existence condition when f_a < m_a --/
theorem no_triangle_exists (a f_a m_a : ℝ) (h1 : f_a < m_a) :
  ¬ ∃ (ABC : Triangle), isTriangle ABC { a := a, f_a := f_a, m_a := m_a } :=
  sorry

end triangle_existence_and_uniqueness_triangle_isosceles_no_triangle_exists_l221_221250


namespace linear_function_does_not_pass_through_quadrant_3_l221_221498

theorem linear_function_does_not_pass_through_quadrant_3
  (f : ℝ → ℝ) (h : ∀ x, f x = -3 * x + 5) :
  ¬ (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ f x = y) :=
by
  sorry

end linear_function_does_not_pass_through_quadrant_3_l221_221498


namespace logarithm_values_count_l221_221233

theorem logarithm_values_count : 
  let S := {1, 2, 3, 4, 7, 9}
  ∃ n : ℕ, n = 17 ∧ 
  ∀ (a b: ℕ), 
    a ∈ S → b ∈ S → a ≠ b → 
    logarithm_distinct_count S n :=
by
  sorry

end logarithm_values_count_l221_221233


namespace ratio_area_octagons_correct_l221_221104

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221104


namespace part1_part2_part3_l221_221733

noncomputable def quadratic_has_real_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1^2 - 2*k*x1 + k^2 + k + 1 = 0 ∧ x2^2 - 2*k*x2 + k^2 + k + 1 = 0

theorem part1 (k : ℝ) :
  quadratic_has_real_roots k → k ≤ -1 :=
sorry

theorem part2 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ x1^2 + x2^2 = 10 → k = -2 :=
sorry

theorem part3 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ (|x1| + |x2| = 2) → k = -1 :=
sorry

end part1_part2_part3_l221_221733


namespace line_segments_form_x_shape_l221_221221

noncomputable def is_line_segment : set (ℝ × ℝ) :=
{p | ∃ (r : ℝ), r ≤ 2 ∧ (p = (r * cos (π / 4), r * sin (π / 4)) ∨ p = (r * cos (5 * π / 4), r * sin (5 * π / 4)))}

theorem line_segments_form_x_shape :
  is_line_segment =
  {p : ℝ × ℝ | 
     (∃ (r : ℝ), r ≤ 2 ∧ p = (r / sqrt 2, r / sqrt 2)) ∨ 
     (∃ (r : ℝ), r ≤ 2 ∧ p = (r / sqrt 2, -r / sqrt 2))} :=
begin
  sorry,
end

end line_segments_form_x_shape_l221_221221


namespace acute_obtuse_triangle_angles_l221_221242

theorem acute_obtuse_triangle_angles
  (A B C P D : Point) -- Define the points
  (hP_inside_triangle : P ∈ triangle A B C) -- Condition: Point P is inside the triangle
  (hD_outside_plane : ¬(D ∈ Plane ABC)) -- Condition: Point D is outside the plane of A, B, C
  (h_acute_exists : 
    acute (angle_APD: Angle D P A) ∨ 
    acute (angle_BPD: Angle D P B) ∨ 
    acute (angle_CPD: Angle D P C)
  ) -- Condition: One of the angles APD, BPD, CPD is acute
  : obtuse (angle_APD: Angle D P A) ∨
    obtuse (angle_BPD: Angle D P B) ∨
    obtuse (angle_CPD: Angle D P C) := 
sorry -- Proof is omitted.

end acute_obtuse_triangle_angles_l221_221242


namespace current_population_is_15336_l221_221503

noncomputable def current_population : ℝ :=
  let growth_rate := 1.28
  let future_population : ℝ := 25460.736
  let years := 2
  future_population / (growth_rate ^ years)

theorem current_population_is_15336 :
  current_population = 15536 := sorry

end current_population_is_15336_l221_221503


namespace fill_tank_time_l221_221612

-- Definitions based on provided conditions
def pipeA_time := 60 -- Pipe A fills the tank in 60 minutes
def pipeB_time := 40 -- Pipe B fills the tank in 40 minutes

-- Theorem statement
theorem fill_tank_time (T : ℕ) : 
  (T / 2) / pipeB_time + (T / 2) * (1 / pipeA_time + 1 / pipeB_time) = 1 → 
  T = 48 :=
by
  intro h
  sorry

end fill_tank_time_l221_221612


namespace age_of_15th_student_l221_221487

theorem age_of_15th_student 
  (average_age_15 : ℕ → ℕ → ℕ)
  (average_age_5 : ℕ → ℕ → ℕ)
  (average_age_9 : ℕ → ℕ → ℕ)
  (h1 : average_age_15 15 15 = 15)
  (h2 : average_age_5 5 14 = 14)
  (h3 : average_age_9 9 16 = 16) :
  let total_age_15 := 15 * 15 in
  let total_age_5 := 5 * 14 in
  let total_age_9 := 9 * 16 in
  let combined_total_age := total_age_5 + total_age_9 in
  let age_15th_student := total_age_15 - combined_total_age in
  age_15th_student = 11 := 
by
  simp [total_age_15, total_age_5, total_age_9, combined_total_age, age_15th_student]
  exact eq.refl 11

end age_of_15th_student_l221_221487


namespace ratio_of_areas_of_octagons_l221_221086

theorem ratio_of_areas_of_octagons (r : ℝ) :
  let side_length_inscribed := r
  let side_length_circumscribed := r * real.sec (real.pi / 8)
  let area_inscribed := 2 * (1 + real.sqrt 2) * side_length_inscribed ^ 2
  let area_circumscribed := 2 * (1 + real.sqrt 2) * side_length_circumscribed ^ 2
  area_circumscribed / area_inscribed = 4 - 2 * real.sqrt 2 :=
sorry

end ratio_of_areas_of_octagons_l221_221086


namespace triangle_stability_l221_221176

/-!
# Triangle Stability Proof

Given that triangular structures are used for strength in bridges, cable car supports, and trusses, 
the mathematical principle that justifies this choice is the stability of triangles.
-/

theorem triangle_stability (bridges_car_trusses_use_triangles : ∀ {S : Type}, triangular_structure S) :
  (∀ {G : Type}, geometric_stability G → triangles_stability) := 
sorry

end triangle_stability_l221_221176


namespace asymptote_equation_l221_221779

noncomputable def proof_problem (a b p : ℝ) (A B F O : ℝ × ℝ) (x y : ℝ) : Prop :=
  (∀ (a b p : ℝ), 0 < a ∧ 0 < b ∧ 0 < p →
    (∃ (A B F : ℝ × ℝ),
      x^2 / a^2 - y^2 / b^2 = 1 ∧
      x^2 = 2 * p * y ∧
      (|A.2 + p/2| + |B.2 + p/2| = 4 * |F.1|) →
        (y = (√2 / 2) * x ∨ y = -(√2 / 2) * x)))

theorem asymptote_equation:
  proof_problem :=
sorry

end asymptote_equation_l221_221779


namespace time_to_fill_tank_l221_221613

-- Definitions of conditions
def fill_by_pipe (time rate : ℚ) : ℚ := time * rate

def pipe_A_rate : ℚ := 1 / 60
def pipe_B_rate : ℚ := 1 / 40
def combined_rate : ℚ := pipe_A_rate + pipe_B_rate

-- Question to be proved
theorem time_to_fill_tank : 
    ∃ T : ℚ, (fill_by_pipe (T / 2) pipe_B_rate + fill_by_pipe (T / 2) combined_rate = 1) ∧ T = 30 :=
by 
    use 30
    sorry

end time_to_fill_tank_l221_221613


namespace min_positive_period_of_sin_cos_l221_221879

theorem min_positive_period_of_sin_cos (x : ℝ) : 
  ∃ T > 0, (∀ x, sin x * cos x = sin (x + T) * cos (x + T)) ∧ 
  (∀ T', T' > 0 → (∀ x, sin x * cos x = sin (x + T') * cos (x + T')) → T' ≥ T) :=
sorry

end min_positive_period_of_sin_cos_l221_221879


namespace profit_percent_is_correct_l221_221593

noncomputable def profit_percent : ℝ := 
  let marked_price_per_pen := 1 
  let pens_bought := 56 
  let effective_payment := 46 
  let discount := 0.01
  let cost_price_per_pen := effective_payment / pens_bought
  let selling_price_per_pen := marked_price_per_pen * (1 - discount)
  let total_selling_price := pens_bought * selling_price_per_pen
  let profit := total_selling_price - effective_payment
  (profit / effective_payment) * 100

theorem profit_percent_is_correct : abs (profit_percent - 20.52) < 0.01 :=
by
  sorry

end profit_percent_is_correct_l221_221593


namespace ratio_of_areas_of_octagons_l221_221068

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221068


namespace ratio_of_octagon_areas_l221_221076

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221076


namespace cost_of_each_orange_l221_221456

theorem cost_of_each_orange (calories_per_orange : ℝ) (total_money : ℝ) (calories_needed : ℝ) (money_left : ℝ) :
  calories_per_orange = 80 → 
  total_money = 10 → 
  calories_needed = 400 → 
  money_left = 4 → 
  (total_money - money_left) / (calories_needed / calories_per_orange) = 1.2 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cost_of_each_orange_l221_221456


namespace new_shipment_is_80_l221_221041

-- Definition of the problem parameters and their relationships
def original_cars := 40
def original_silver_cars := 0.20 * 40

def new_shipment (x : ℝ) := x
def new_silver_cars (x : ℝ) := 0.50 * x

-- Total cars after the shipment
def total_cars (x : ℝ) := original_cars + new_shipment x
def total_silver_cars (x : ℝ) := original_silver_cars + new_silver_cars x

-- Proving the percentage relation after the new shipment
theorem new_shipment_is_80 (x : ℝ) : total_silver_cars x / total_cars x = 0.40 → x = 80 :=
by
  unfold original_cars original_silver_cars new_shipment new_silver_cars total_cars total_silver_cars
  intros h

  -- We skip the actual proof as instructed
  sorry

end new_shipment_is_80_l221_221041


namespace sufficient_but_not_necessary_condition_l221_221753

variables {a b : ℝ}

theorem sufficient_but_not_necessary_condition (h₁ : b < -4) : |a| + |b| > 4 :=
by {
    sorry
}

end sufficient_but_not_necessary_condition_l221_221753


namespace circumcenter_barycentric_incenter_barycentric_orthocenter_barycentric_l221_221212

variables {α β γ : ℝ} {a b c : ℝ}

theorem circumcenter_barycentric (α β γ : ℝ) :
  barycentric_coords (circumcenter (triangle α β γ)) = (sin (2 * α), sin (2 * β), sin (2 * γ)) :=
sorry

theorem incenter_barycentric (a b c : ℝ) :
  barycentric_coords (incenter (triangle a b c)) = (a, b, c) :=
sorry

theorem orthocenter_barycentric (α β γ : ℝ) :
  barycentric_coords (orthocenter (triangle α β γ)) = (tan α, tan β, tan γ) :=
sorry

end circumcenter_barycentric_incenter_barycentric_orthocenter_barycentric_l221_221212


namespace ratio_of_octagon_areas_l221_221073

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221073


namespace find_missing_number_l221_221410

theorem find_missing_number (x : ℚ) (h : 11 * x + 4 = 7) : x = 9 / 11 :=
sorry

end find_missing_number_l221_221410


namespace distance_from_center_to_line_of_tangent_circle_l221_221315

theorem distance_from_center_to_line_of_tangent_circle 
  (a : ℝ) (ha : 0 < a) 
  (h_circle : (2 - a)^2 + (1 - a)^2 = a^2)
  (h_tangent : ∀ x y : ℝ, x = 0 ∨ y = 0): 
  (|2 * a - a - 3| / ((2:ℝ)^2 + (-1)^2).sqrt) = (2 * (5:ℝ).sqrt) / 5 :=
by
  sorry

end distance_from_center_to_line_of_tangent_circle_l221_221315


namespace problem_equiv_l221_221757

theorem problem_equiv {x : ℝ} (h : 3^x - 3^(x-1) = 18) : (3 * x)^x = 729 :=
by
  sorry

end problem_equiv_l221_221757


namespace sum_of_consecutive_integers_l221_221778

theorem sum_of_consecutive_integers : 
  (∃ n k : ℕ, n ≥ 2 ∧ k > 0 ∧ n * (2 * k + n - 1) = 1050) ∧ 
  (∀ n k, n ≥ 2 ∧ k > 0 ∧ n * (2 * k + n - 1) = 1050 → 
   n ∈ {2, 3, 5, 6, 10, 15, 21, 35, 50, 75}) → 
  (∃ m, m = 10) :=
by
  sorry

end sum_of_consecutive_integers_l221_221778


namespace exists_points_with_distance_le_sqrt5_l221_221460

-- Define the rectangle dimensions
def rectangleWidth : ℝ := 4
def rectangleHeight : ℝ := 3

-- Define the number of points
def numPoints : Nat := 6

-- Introduce the existence of points within the rectangle
def points (i : Fin numPoints) : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- The main statement
theorem exists_points_with_distance_le_sqrt5 :
  ∃ (i j : Fin numPoints), i ≠ j ∧ distance (points i) (points j) ≤ real.sqrt 5 :=
by
  sorry

end exists_points_with_distance_le_sqrt5_l221_221460


namespace max_n_value_l221_221471

def is_fancy (X : set (set ℝ)) : Prop :=
  X.card = 2022 ∧ 
  (∀ I ∈ X, ∃ a b : ℝ, 0 ≤ a ∧ b ≤ 1 ∧ I = set.Icc a b) ∧ 
  (∀ r ∈ set.Icc 0 1, (set.filter (λ i, r ∈ i) X).card ≤ 1011)

def n (A B : set (set ℝ)) : ℕ :=
  set.card {p : (set ℝ) × (set ℝ) | p.1 ∈ A ∧ p.2 ∈ B ∧ set.nonempty (p.1 ∩ p.2)}

theorem max_n_value (A B : set (set ℝ)) (hA : is_fancy A) (hB : is_fancy B) :
  n A B ≤ 3 * 1011^2 := 
sorry

end max_n_value_l221_221471


namespace circle_tangent_distance_l221_221309

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221309


namespace part1_part2_find_min_value_l221_221235

open Real

-- Proof of Part 1
theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a^2 / b + b^2 / a ≥ a + b :=
by sorry

-- Proof of Part 2
theorem part2 (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) ≥ 1 :=
by sorry

-- Corollary to find the minimum value
theorem find_min_value (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) = 1 ↔ x = 1 / 2 :=
by sorry

end part1_part2_find_min_value_l221_221235


namespace weekly_crab_meat_cost_l221_221796

-- Declare conditions as definitions
def dishes_per_day : ℕ := 40
def pounds_per_dish : ℝ := 1.5
def cost_per_pound : ℝ := 8
def closed_days_per_week : ℕ := 3
def days_per_week : ℕ := 7

-- Define the Lean statement to prove the weekly cost
theorem weekly_crab_meat_cost :
  let days_open_per_week := days_per_week - closed_days_per_week
  let pounds_per_day := dishes_per_day * pounds_per_dish
  let daily_cost := pounds_per_day * cost_per_pound
  let weekly_cost := daily_cost * (days_open_per_week : ℝ)
  weekly_cost = 1920 :=
by
  sorry

end weekly_crab_meat_cost_l221_221796


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221120

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221120


namespace second_player_can_ensure_divisibility_by_13_l221_221524

theorem second_player_can_ensure_divisibility_by_13 :
  ∀ (a : Fin 8 → ℤ), ∃ (b : Fin 8 → ℤ), 
  (∀ i, b i = a i + (if (i % 2 = 0) then -1 else 1)) →
  (∑ i in (Finset.range 8), b i * 8 ^ (i : ℕ)) % 13 = 0 := sorry

end second_player_can_ensure_divisibility_by_13_l221_221524


namespace find_floor_function_l221_221814

-- Define the conditions as Lean statements
def cond1 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x) + f(y) + 1 ≥ f(x + y) ∧ f(x + y) ≥ f(x) + f(y)

def cond2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f(0) ≥ f(x)

def cond3 (f : ℝ → ℝ) : Prop :=
  f(1) = 1 ∧ f(-1) = -1

-- Define the main theorem to be proven
theorem find_floor_function (f : ℝ → ℝ) :
  cond1 f → cond2 f → cond3 f → ∀ x : ℝ, f(x) = ⌊x⌋ :=
by
  intro h1 h2 h3
  sorry

end find_floor_function_l221_221814


namespace arithmetic_sequence_50th_term_l221_221643

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 7
  let n := 50
  (a_1 + (n - 1) * d) = 346 :=
by
  let a_1 := 3
  let d := 7
  let n := 50
  show (a_1 + (n - 1) * d) = 346
  sorry

end arithmetic_sequence_50th_term_l221_221643


namespace complex_number_solution_l221_221723

theorem complex_number_solution (z : ℂ) (h : (1 - complex.I) * z = 1 + complex.I) : z = complex.I :=
sorry

end complex_number_solution_l221_221723


namespace area_of_enclosed_region_l221_221905

def circle_area : ℝ :=
  let x2y2_term := (x y : ℝ) => x^2 + y^2
  let eq := fun (x y : ℝ) => x2y2_term x y + 6 * x + 16 * y + 18 = 0
  let radius := √55
by sorry

theorem area_of_enclosed_region :
  let eq := fun (x y : ℝ) => x^2 + y^2 + 6 * x + 16 * y + 18 = 0
  ∃ center radius, eq = (λ x y, (x + 3)^2 + (y + 8)^2 - 55) ∧ area_of_circle radius = 55 * Real.pi :=
sorry

end area_of_enclosed_region_l221_221905


namespace problem_solution_l221_221626

-- Define the conditions: stops and distance
def stops : Finset ℕ := Finset.range 15   -- Stops numbered 0 to 14 (instead of 1 to 15 for simplicity)
def distance (i j : ℕ) : ℕ := 100 * (abs (i - j))

-- Define the probability calculation
def probability_feet_le_500 : ℚ :=
  let valid_count := (2 * (5 + 6 + 7 + 8 + 9) + 5 * 10)
  let total_count := 15 * 14
  (valid_count : ℚ) / total_count

-- Define the final result m + n
def result_m_plus_n : ℕ :=
  let frac := probability_feet_le_500
  frac.num.natAbs + frac.denom.natAbs

-- State the final proof problem
theorem problem_solution : result_m_plus_n = 11 := by
  sorry

end problem_solution_l221_221626


namespace ratio_of_octagon_areas_l221_221080

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221080


namespace solution_is_unique_l221_221288

def conditions (x y : ℝ) : Prop :=
  (x/y + y/x) * (x + y) = 15 ∧
  (x^2/y^2 + y^2/x^2) * (x^2 + y^2) = 85

theorem solution_is_unique : ∀ x y : ℝ, conditions x y → (x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2) :=
by
  intro x y
  assume h : conditions x y
  sorry

end solution_is_unique_l221_221288


namespace rectangle_area_32_sqrt_5_l221_221282

theorem rectangle_area_32_sqrt_5 :
  let p1 := (⟨-3, 2⟩ : ℤ × ℤ)
  let p2 := (⟨1, -6⟩ : ℤ × ℤ)
  let side_length_1 := real.sqrt ( (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 )
  let side_length_2 := 8 -- This is derived from calculating the vertical distance as shown in solution steps.
  side_length_1 * side_length_2 = 32 * real.sqrt 5 :=
sorry

end rectangle_area_32_sqrt_5_l221_221282


namespace tan_alpha_parallel_sin_2alpha_perpendicular_angle_between_vectors_l221_221251

variables (α : ℝ) 
variables A B C O : ℝ → ℝ
def pointA := (3, 0)
def pointB := (0, 3)
def pointC := (cos α, sin α)
def pointO := (0, 0)

-- Question 1
theorem tan_alpha_parallel (α : ℝ) :
  ((cos α, sin α) = (3 * cos α, 3 * sin α)) → (tan α = -1) :=
sorry

-- Question 2
theorem sin_2alpha_perpendicular (α : ℝ) :
  let vectorAC := (cos α - 3, sin α)
      vectorBC := (cos α, sin α - 3) in
  (vectorAC.1 * vectorBC.1 + vectorAC.2 * vectorBC.2 = 0) →
  (sin 2α = -8/9) :=
sorry

-- Question 3
theorem angle_between_vectors (α : ℝ) :
  (0 < α) ∧ (α < π) ∧
  (sqrt ((3 + cos α)^2 + (sin α)^2) = sqrt 13) →
  (let angle_cosine := sin α in 
   angle_cosine = sqrt 3 / 2 →
   true) :=
sorry

end tan_alpha_parallel_sin_2alpha_perpendicular_angle_between_vectors_l221_221251


namespace shooter_probability_at_most_8_l221_221160

theorem shooter_probability_at_most_8 (P10 P9 P8 P_to_8: ℝ) 
  (hP10 : P10 = 0.24)
  (hP9 : P9 = 0.28)
  (hP8 : P8 = 0.19) :
  P_to_8 = 1 - P10 - P9 :=
begin
  sorry
end

end shooter_probability_at_most_8_l221_221160


namespace circle_tangent_distance_l221_221305

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221305


namespace Nick_has_7_Pennsylvania_state_quarters_l221_221452

theorem Nick_has_7_Pennsylvania_state_quarters :
  ∀ (total_quarters state_fraction pa_fraction : ℚ),
    total_quarters = 35 →
    state_fraction = 2 / 5 →
    pa_fraction = 1 / 2 →
    (pa_fraction * (state_fraction * total_quarters)) = 7 :=
by
  intros total_quarters state_fraction pa_fraction h_total h_state_fraction h_pa_fraction
  rw [h_total, h_state_fraction, h_pa_fraction]
  norm_num
  sorry

end Nick_has_7_Pennsylvania_state_quarters_l221_221452


namespace arithmetic_mean_of_primes_in_list_l221_221211

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def arithmetic_mean (s : Set ℕ) : ℚ :=
  let primes := {x ∈ s | is_prime x}
  let sum_of_primes := primes.to_finset.sum id
  let count_of_primes := primes.to_finset.card
  sum_of_primes / count_of_primes

theorem arithmetic_mean_of_primes_in_list :
  arithmetic_mean {37, 39, 41, 43, 45} = 121 / 3 :=
by
  sorry

end arithmetic_mean_of_primes_in_list_l221_221211


namespace area_ratio_of_octagons_is_4_l221_221094

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221094


namespace return_trip_time_in_minutes_l221_221595

-- Define the constants and inputs
def boat_speed : ℝ := 16  -- speed of the boat relative to the water in mph
def trip_upstream_time_minutes : ℝ := 20  -- time taken to travel upstream in minutes
def current_speed : ℝ := 2.28571428571  -- speed of the current in mph

-- Helper definitions
def trip_upstream_time_hours : ℝ := trip_upstream_time_minutes / 60  -- convert upstream time to hours
def effective_speed_upstream : ℝ := boat_speed - current_speed  -- speed of the boat against the current in mph
def distance_upstream : ℝ := effective_speed_upstream * trip_upstream_time_hours  -- distance traveled upstream in miles
def effective_speed_downstream : ℝ := boat_speed + current_speed  -- speed of the boat with the current in mph

-- Prove that the return trip downstream takes 15 minutes
theorem return_trip_time_in_minutes :
  (distance_upstream / effective_speed_downstream) * 60 = 15 :=
by
  sorry

end return_trip_time_in_minutes_l221_221595


namespace range_of_a_l221_221812

variables {α β : ℝ}

theorem range_of_a 
  (h₀ : 0 < α)
  (h₁ : α ≤ π)
  (h₂ : 0 < β)
  (h₃ : β ≤ π)
  (h₄ : α + β < π)
  (inequality : cos (sqrt α) + cos (sqrt β) > a + cos (sqrt (α * β))) :
  a ∈ Iio 1 :=
sorry

end range_of_a_l221_221812


namespace probability_two_heads_one_tail_in_three_tosses_l221_221554

theorem probability_two_heads_one_tail_in_three_tosses
(P : ℕ → Prop) (pr : ℤ) : 
  (∀ n, P n → pr = 1 / 2) -> 
  P 3 → pr = 3 / 8 :=
by
  sorry

end probability_two_heads_one_tail_in_three_tosses_l221_221554


namespace exchange_5_dollars_to_francs_l221_221615

-- Define the exchange rates
def dollar_to_lire (d : ℕ) : ℕ := d * 5000
def lire_to_francs (l : ℕ) : ℕ := (l / 1000) * 3

-- Define the main theorem
theorem exchange_5_dollars_to_francs : lire_to_francs (dollar_to_lire 5) = 75 :=
by
  sorry

end exchange_5_dollars_to_francs_l221_221615


namespace inequality_correct_l221_221009

theorem inequality_correct (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1) : (1 - a) ^ a > (1 - b) ^ b :=
sorry

end inequality_correct_l221_221009


namespace paths_count_12_l221_221741

def continuous_path_no_revisit (path : list (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), i ≠ j → path[i] ≠ path[j]

def paths_from_A_to_B : list (list (ℕ × ℕ)) :=
[
  [(1,5), (3,1)], -- A-C-B
  [(1,3), (3,5), (5,1)], -- A-D-C-B
  [(1,3), (3,7), (7,5), (5,1)], -- A-D-G-C-B
  [(1,3), (3,2), (2,4), (4,5), (5,1)], -- A-D-E-F-C-B
  [(1,5), (5,4), (4,1)], -- A-C-F-B
  [(1,5), (5,3), (3,4), (4,1)], -- A-C-D-F-B
  [(1,5), (5,3), (3,2), (2,4), (4,1)], -- A-C-D-E-F-B
  [(1,5), (5,3), (3,7), (7,4), (4,1)], -- A-C-D-G-F-B
  [(1,3), (3,5), (5,4), (4,1)], -- A-D-C-F-B
  [(1,3), (3,4), (4,1)], -- A-D-F-B
  [(1,3), (3,2), (2,4), (4,1)], -- A-D-E-F-B
  [(1,3), (3,7), (7,4), (4,1)] -- A-D-G-F-B
]

theorem paths_count_12 : count_paths (paths_from_A_to_B) = 12 := 
by
  apply continuous_path_no_revisit
  sorry

end paths_count_12_l221_221741


namespace arithmetic_geometric_mean_l221_221465

theorem arithmetic_geometric_mean (a b : ℝ) (h1 : a + b = 48) (h2 : a * b = 440) : a^2 + b^2 = 1424 := 
by 
  -- Proof goes here
  sorry

end arithmetic_geometric_mean_l221_221465


namespace area_ratio_of_octagons_is_4_l221_221093

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221093


namespace can_form_a_set_l221_221915

-- Definition of a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a triangle in a plane
structure Triangle where
  A B C : Point

-- Condition D: All points in a plane that are equidistant from the three vertices of a triangle
def equidistant_points_from_triangle (t : Triangle) : Set Point :=
  { p : Point | dist p t.A = dist p t.B ∧ dist p t.B = dist p t.C }

-- Prove that the set defined above can form a set according to the principles of definiteness and distinctness
theorem can_form_a_set (t : Triangle) : ∃ s : Set Point, s = equidistant_points_from_triangle t := 
by
  use equidistant_points_from_triangle t
  sorry


end can_form_a_set_l221_221915


namespace find_AC_l221_221399

theorem find_AC (AB DC AD : ℕ) (hAB : AB = 13) (hDC : DC = 20) (hAD : AD = 5) : 
  AC = 24.2 := 
sorry

end find_AC_l221_221399


namespace area_ratio_is_correct_l221_221141

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l221_221141


namespace find_shortest_tangent_length_l221_221436

noncomputable def E := (8 : ℝ, 0)
noncomputable def F := (-10 : ℝ, 0)
noncomputable def r1 := 5
noncomputable def r2 := 8

def dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def shortest_tangent_length : ℝ :=
  let EF := dist E F
  let internal_div := (r1 + r2)
  let G := ( ((r1 * F.1 + r2 * E.1) / internal_div), 0)
  let dist_RG := r1 * dist G F / internal_div
  let dist_SG := r2 * dist G E / internal_div
  dist_RG + dist_SG

theorem find_shortest_tangent_length :
  shortest_tangent_length = 1440 / 169 := 
sorry

end find_shortest_tangent_length_l221_221436


namespace productProbLessThan36_l221_221833

noncomputable def pacoSpinner : List ℤ := [1, 2, 3, 4, 5, 6]
noncomputable def manuSpinner : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def product_less_than_36 (p : ℤ) (m : ℤ) : Prop :=
  p * m < 36

def probability_product_less_than_36 : ℚ :=
  (∑ m in manuSpinner, ∑ p in pacoSpinner, if product_less_than_36 p m then 1 else 0) /
  (manuSpinner.length * pacoSpinner.length)

theorem productProbLessThan36 : probability_product_less_than_36 = 8 / 30 := sorry

end productProbLessThan36_l221_221833


namespace zachary_cans_second_day_l221_221566

-- Lean Statement
theorem zachary_cans_second_day :
  ∃ d : ℕ, d = 9 ∧ ∀ n : ℕ, n ≤ 7 →
    (n = 1 → (4 : ℕ)) ∧
    (n = 3 → 14) ∧
    (n = 7 → 34) ∧
    (∀ k : ℕ, k < 7 → (∃ m : ℕ, (14 - 4) / (7 - 3) = m)) :=
sorry

end zachary_cans_second_day_l221_221566


namespace base_seven_to_ten_l221_221533

theorem base_seven_to_ten :
  4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0 = 10738 :=
by
  sorry

end base_seven_to_ten_l221_221533


namespace sum_possible_students_l221_221159

theorem sum_possible_students :
  ∑ k in (Finset.filter (λ n, n % 8 = 2 ∧ 150 ≤ n ∧ n ≤ 250) (Finset.range 251)), k = 2626 := by 
sorry

end sum_possible_students_l221_221159


namespace sum_of_numbers_greater_than_0_l221_221514

def numbers := [0.8, 0.5, 0.9]
def condition (x : ℝ) : Prop := x > 0.7
def sum : ℝ := numbers.filter condition |>.sum

theorem sum_of_numbers_greater_than_0.7 : sum = 1.7 :=
by
  sorry

end sum_of_numbers_greater_than_0_l221_221514


namespace number_of_hens_l221_221047

theorem number_of_hens (H C G : ℕ) 
  (h1 : H + C + G = 120) 
  (h2 : 2 * H + 4 * C + 4 * G = 348) : 
  H = 66 := 
by 
  sorry

end number_of_hens_l221_221047


namespace area_of_quadrilateral_l221_221168

theorem area_of_quadrilateral 
  (area_ΔBDF : ℝ) (area_ΔBFE : ℝ) (area_ΔEFC : ℝ) (area_ΔCDF : ℝ) (h₁ : area_ΔBDF = 5)
  (h₂ : area_ΔBFE = 10) (h₃ : area_ΔEFC = 10) (h₄ : area_ΔCDF = 15) :
  (80 - (area_ΔBDF + area_ΔBFE + area_ΔEFC + area_ΔCDF)) = 40 := 
  by sorry

end area_of_quadrilateral_l221_221168


namespace factor_x8_minus_81_l221_221973

theorem factor_x8_minus_81 (x : ℝ) : x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := 
by 
  sorry

end factor_x8_minus_81_l221_221973


namespace product_identity_l221_221837

theorem product_identity (n : ℕ) (hn : 0 < n) :
  (∏ i in Finset.range n, n + (i + 1)) = 2^n * ∏ i in Finset.range n, (2 * i + 1) := by
sorry

end product_identity_l221_221837


namespace instantaneous_speed_at_4_l221_221053

def motion_equation (t : ℝ) : ℝ := t^2 - 2 * t + 5

theorem instantaneous_speed_at_4 :
  (deriv motion_equation 4) = 6 :=
by
  sorry

end instantaneous_speed_at_4_l221_221053


namespace perpendicular_slope_l221_221222

-- Define the given line equation and extract its slope
def given_line (x y : ℝ) : Prop := 2 * x + 3 * y = 6

-- The slope of the given line is -2/3
def slope_of_given_line := - (2 / 3 : ℝ)

-- The slope of the perpendicular line to the given line
def slope_of_perpendicular_line := - 1 / slope_of_given_line

-- Prove that the slope of the line perpendicular to the given line is 3/2
theorem perpendicular_slope : slope_of_perpendicular_line = (3 / 2 : ℝ) := 
by 
  -- This part will be filled in proof
  sorry

end perpendicular_slope_l221_221222


namespace spending_required_for_free_shipping_l221_221417

def shampoo_cost : ℕ := 10
def conditioner_cost : ℕ := 10
def lotion_cost : ℕ := 6
def shampoo_count : ℕ := 1
def conditioner_count : ℕ := 1
def lotion_count : ℕ := 3
def additional_spending_needed : ℕ := 12
def current_spending : ℕ := (shampoo_cost * shampoo_count) + (conditioner_cost * conditioner_count) + (lotion_cost * lotion_count)

theorem spending_required_for_free_shipping : current_spending + additional_spending_needed = 50 := by
  sorry

end spending_required_for_free_shipping_l221_221417


namespace speed_of_second_train_l221_221576

-- Define the given values
def length_train1 := 290.0 -- in meters
def speed_train1 := 120.0 -- in km/h
def length_train2 := 210.04 -- in meters
def crossing_time := 9.0 -- in seconds

-- Define the conversion factors and useful calculations
def meters_per_second_to_kmph (v : Float) : Float := v * 3.6
def total_distance := length_train1 + length_train2
def relative_speed_ms := total_distance / crossing_time
def relative_speed_kmph := meters_per_second_to_kmph relative_speed_ms

-- Define the proof statement
theorem speed_of_second_train : relative_speed_kmph - speed_train1 = 80.0 :=
by
  sorry

end speed_of_second_train_l221_221576


namespace ratio_of_areas_of_octagons_l221_221059

theorem ratio_of_areas_of_octagons
  (r : ℝ)
  (sin_45 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2)
  (sin_67_5 : Real.sin (Real.pi / 180 * 67.5) = Real.sqrt (2 + Real.sqrt 2) / 2)
  : let smaller_octagon_area := 2 * (1 + Real.sqrt 2) * r^2 in
    let larger_octagon_area := 
      2 * (1 + Real.sqrt 2) * (2 * r * (1 + Real.sqrt 2))^2 in
    larger_octagon_area / smaller_octagon_area = 12 + 8 * Real.sqrt 2 :=
by
  sorry

end ratio_of_areas_of_octagons_l221_221059


namespace locus_area_eq_five_a_squared_over_twelve_l221_221998

theorem locus_area_eq_five_a_squared_over_twelve 
  (a : ℝ) :
  let square_center := (0, 0)
  let vertices := [(a / 2, a / 2), (a / 2, -a / 2), (-a / 2, a / 2), (-a / 2, -a / 2)]
  let condition (P : ℝ × ℝ) := (P.1, P.2) = square_center ∨ ∃ x y, P = (x, y) ∧ 
    sqrt (x^2 + y^2) = min (a / 2 - abs x) (a / 2 - abs y)
  in let area := ∫∫ (x y : ℝ) in sorry, -- you would define the integration bounds given the condition and calculate the area correctly in the final proof
  area = (5 * a^2) / 12 := sorry

end locus_area_eq_five_a_squared_over_twelve_l221_221998


namespace minimum_d_exists_l221_221404

open Nat

theorem minimum_d_exists :
  ∃ (a b c d e f g h i k : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ k ∧
                                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ k ∧
                                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ k ∧
                                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ k ∧
                                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ k ∧
                                f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ k ∧
                                g ≠ h ∧ g ≠ i ∧ g ≠ k ∧
                                h ≠ i ∧ h ≠ k ∧
                                i ≠ k ∧
                                d = a + 3 * (e + h) + k ∧
                                d = 20 :=
by
  sorry

end minimum_d_exists_l221_221404


namespace toy_store_problem_l221_221165

variables (x y : ℕ)

theorem toy_store_problem (h1 : 8 * x + 26 * y + 33 * (31 - x - y) / 2 = 370)
                          (h2 : x + y + (31 - x - y) / 2 = 31) :
    x = 20 :=
sorry

end toy_store_problem_l221_221165


namespace area_R2_is_160_l221_221703

noncomputable def areaOfR2 : ℝ :=
  let R1_side1 := 4
  let R1_area := 32
  let R2_diagonal := 20
  -- The other side of R1, calculated from its area
  let R1_side2 := R1_area / R1_side1
  -- The ratio of side lengths in similar rectangles R1 and R2
  let ratio := R1_side2 / R1_side1
  -- Calculate side lengths for R2 using the diagonal
  let R2_side1 := real.sqrt (R2_diagonal ^ 2 / (1 + ratio ^ 2))
  let R2_side2 := ratio * R2_side1
  -- Calculate area of R2
  R2_side1 * R2_side2

theorem area_R2_is_160 :
  areaOfR2 = 160 := sorry

end area_R2_is_160_l221_221703


namespace cube_surface_area_l221_221949

-- Define the dimensions of the rectangular prism
def prism_length := 12
def prism_width := 4
def prism_height := 18

-- Define the volume of the rectangular prism
def prism_volume : Nat := prism_length * prism_width * prism_height

-- Define the edge length of the cube with the same volume as the rectangular prism
def cube_edge_length : Nat := Real.cbrt (prism_volume.toReal).toNat

-- Proof statement: The surface area of the cube with the same volume is 864 square inches
theorem cube_surface_area (h1 : cube_edge_length * cube_edge_length * cube_edge_length = prism_volume) :
  6 * (cube_edge_length * cube_edge_length) = 864 := by
  -- Volume of the rectangular prism is equal to volume of the cube
  have vol_prism_cube_eq : prism_volume = cube_edge_length * cube_edge_length * cube_edge_length := h1
  -- Surface area of the cube
  sorry

end cube_surface_area_l221_221949


namespace no_root_l221_221480

theorem no_root :
  ∀ x : ℝ, x - (9 / (x - 4)) ≠ 4 - (9 / (x - 4)) :=
by
  intro x
  sorry

end no_root_l221_221480


namespace sum_of_reciprocals_of_roots_l221_221435

theorem sum_of_reciprocals_of_roots {p q r s t : ℝ} 
  (h : ∀ (z : ℂ), z^5 + p * z^4 + q * z^3 + r * z^2 + s * z + t = 0 → |z| = 1) 
  (roots : ℕ → ℂ) 
  (hf : ∀ i, (roots i)^5 + p * (roots i)^4 + q * (roots i)^3 + r * (roots i)^2 + s * (roots i) + t = 0) 
  (h_distinct : function.injective roots) : ∑ i in finset.range 5, 1 / (roots i) = -p := by
  sorry

end sum_of_reciprocals_of_roots_l221_221435


namespace exp_log_equiv_l221_221234

theorem exp_log_equiv (a m n : ℝ) (ha : 0 < a) (hm : a ≠ 1) (h1 : log a 3 = m) (h2 : log a 5 = n) :
    a^(2 * m + n) = 75 :=
by
  sorry

end exp_log_equiv_l221_221234


namespace initial_ratio_of_milk_to_water_l221_221772

theorem initial_ratio_of_milk_to_water 
  (M W : ℕ) 
  (h1 : M + 10 + W = 30)
  (h2 : (M + 10) * 2 = W * 5)
  (h3 : M + W = 20) : 
  M = 11 ∧ W = 9 := 
by 
  sorry

end initial_ratio_of_milk_to_water_l221_221772


namespace angle_ratio_of_parallelogram_l221_221572

noncomputable def parallelogram_side_ratio := sqrt 3
noncomputable def parallelogram_diagonal_ratio := sqrt 7

theorem angle_ratio_of_parallelogram
  (x y : ℝ)
  (hx_side : parallelogram_side_ratio = sqrt 3)
  (hy_diag : parallelogram_diagonal_ratio = sqrt 7)
  (d1 d2 : ℝ)
  (hd1 : d1 = sqrt (1 + 3) * x)
  (hd2 : d2 = sqrt 7 * y)
  (h_eq : y = x) :
  let α : ℝ := 30 * (π / 180)
      β : ℝ := 150 * (π / 180) in
  β / α = 5 :=
by
  sorry

end angle_ratio_of_parallelogram_l221_221572


namespace ratio_area_octagons_correct_l221_221105

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221105


namespace equal_black_white_squares_in_5x5_grid_l221_221771

-- Define that a grid is a set of cells, each being black or white.
-- 5x5 grid definition with parameters if needed can be extended here, assuming some cells painted
-- and sides aligned can be parameter control-driven for color locations.

noncomputable def number_of_squares_with_equal_black_white_cells (grid : ℕ × ℕ → Prop) : ℕ :=
  -- The grid is a function from cell coordinates to a color property (e.g., black or white).
  -- χ grid needs to define resultant painting valid.

theorem equal_black_white_squares_in_5x5_grid (grid : ℕ × ℕ → Prop) :
  number_of_squares_with_equal_black_white_cells grid = 16 :=
sorry

end equal_black_white_squares_in_5x5_grid_l221_221771


namespace max_minus_min_all_three_languages_l221_221602

def student_population := 1500

def english_students (e : ℕ) : Prop := 1050 ≤ e ∧ e ≤ 1125
def spanish_students (s : ℕ) : Prop := 750 ≤ s ∧ s ≤ 900
def german_students (g : ℕ) : Prop := 300 ≤ g ∧ g ≤ 450

theorem max_minus_min_all_three_languages (e s g e_s e_g s_g e_s_g : ℕ) 
    (he : english_students e)
    (hs : spanish_students s)
    (hg : german_students g)
    (pie : e + s + g - e_s - e_g - s_g + e_s_g = student_population) 
    : (M - m = 450) :=
sorry

end max_minus_min_all_three_languages_l221_221602


namespace circle_distance_condition_l221_221349

theorem circle_distance_condition (a : ℝ) (h1 : (2 - a)^2 + (1 - a)^2 = a^2) (h2 : ∀ (x y : ℝ), ∃ c : ℝ, y = 2*x - c) :
    ∃ (center : ℝ × ℝ), ((center.1 = a) ∧ (center.2 = a) ∧ (center = (1, 1) ∨ center = (5, 5))) 
    → (∀ (line : ℝ × ℝ × ℝ), line = (2, -1, -3) → abs ((line.1 * a + line.2 * a + line.3) / real.sqrt (line.1 ^ 2 + line.2 ^ 2)) = (2 * real.sqrt 5 / 5)) :=
by
  intro a h1 h2 center h3 line h4
  sorry

end circle_distance_condition_l221_221349


namespace find_M_value_when_x_3_l221_221446

-- Definitions based on the given conditions
def polynomial (a b c d x : ℝ) : ℝ := a*x^5 + b*x^3 + c*x + d

-- Given conditions
variables (a b c d : ℝ)
axiom h₀ : polynomial a b c d 0 = -5
axiom h₁ : polynomial a b c d (-3) = 7

-- Desired statement: Prove that the value of polynomial at x = 3 is -17
theorem find_M_value_when_x_3 : polynomial a b c d 3 = -17 :=
by sorry

end find_M_value_when_x_3_l221_221446


namespace polar_coordinates_of_point_l221_221640

noncomputable theory

-- Define the point in rectangular coordinates
def point_rect : ℝ × ℝ := (2, -2)

-- Define the conversion to polar coordinates function
def convertToPolar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then
             if y >= 0 then Real.arctan (y / x)
             else 2 * Real.pi - Real.arctan (Real.abs y / x)
           else if x < 0 then
             if y >= 0 then Real.pi - Real.arctan (y / (Real.abs x))
             else Real.pi + Real.arctan (Real.abs y / (Real.abs x))
           else if y > 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ)

theorem polar_coordinates_of_point :
  convertToPolar (fst point_rect) (snd point_rect) = (2 * Real.sqrt 2, 7 * Real.pi / 4) :=
  by
    sorry

end polar_coordinates_of_point_l221_221640


namespace area_ratio_of_octagons_is_4_l221_221098

-- Define the given conditions
variable (r : ℝ) -- radius of the common circle

def side_length_circumscribed_octagon := √2 * r
def side_length_inscribed_octagon := r / √2

-- Areas of polygons scale with the square of the side length
def area_ratio_circumscribed_to_inscribed := (side_length_circumscribed_octagon r / side_length_inscribed_octagon r)^2

theorem area_ratio_of_octagons_is_4 :
  area_ratio_circumscribed_to_inscribed r = 4 := by
  sorry

end area_ratio_of_octagons_is_4_l221_221098


namespace adjacent_probability_Abby_Bridget_l221_221958

theorem adjacent_probability_Abby_Bridget :
  let kids := {k : Fin 6 // true},
  let positions := {p : Fin 2 × Fin 3 // true},
  let total_arrangements := Finset.perm.univ kids,
  let adjacent_arrangements := 
    (Finset.perm.univ kids).filter (λ perm, 
      ∃ i j, j < i < 6 ∧ (perm i = 0 ∧ perm j = 1 ∨ perm (i+1) = 1 ∧ perm i = 0)),
  P (total_arrangements.card : ℝ) = (adjacent_arrangements.card : ℝ) / (total_arrangements.card : ℝ) := 
  7 / 15 :=
by sorry

end adjacent_probability_Abby_Bridget_l221_221958


namespace exists_infinitely_many_N_l221_221839

-- Definition indicating that any odd square can be written as 8k + 1 for some integer k
def odd_square_form (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 8 * k + 1

-- Infinitely many primes of the form 8k + 3
def primes_of_form_8k3 : Prop :=
  ∃ p : ℕ, (prime p ∧ (∃ k : ℕ, p = 8 * k + 3))

-- Main theorem statement
theorem exists_infinitely_many_N :
  (∃ n : ℕ, ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ 
   (∃ k_p : ℕ, p = 8 * k_p + 3) ∧ (∃ k_q : ℕ, q = 8 * k_q + 3) ∧ n = 2 * p * q ∧ 
   ∀ (squares_used : ℕ), squares_used < 10 → ¬ (∃ (sq_sum : ℕ), odd_sum_of_squares sq_sum squares_used ∧ sq_sum = n)) :=
sorry

-- Supporting definition to express sum of odd squares
def odd_sum_of_squares (n : ℕ) (count : ℕ) : Prop :=
  ∃ sqs : list ℕ, (sqs.length = count) ∧ (∀ x ∈ sqs, odd_square_form x) ∧ (sqs.sum = n)

end exists_infinitely_many_N_l221_221839


namespace sequence_divisible_by_2020_l221_221279

/--
Given a sequence of positive integers \( a : ℕ → ℕ \) satisfying the recurrence relation:
\[ 2a_{n+2} = a_{n+1} + 4a_n \quad \text{for} \quad n=0, 1, \ldots, 3028 \]
there exists at least one integer in the sequence \( a_0, a_1, \ldots, a_{3030} \) that is divisible by \( 2^{2020} \).
-/
theorem sequence_divisible_by_2020 (a : ℕ → ℕ)
  (h_pos : ∀ n ≤ 3030, 0 < a n)
  (h_recurrence : ∀ n < 3029, 2 * a (n + 2) = a (n + 1) + 4 * a n) :
  ∃ i ≤ 3030, 2^2020 ∣ a i :=
begin
  sorry
end

end sequence_divisible_by_2020_l221_221279


namespace least_colors_hexagon_tiling_l221_221851

open SimpleGraph

-- Condition setup
def hexagon_tiling : SimpleGraph ℕ := {
  adj := λ a b, -- adjacency represents if two hexagons share a side
    -- (this needs an appropriate representation of the hexagon grid adjacency)
    sorry,
  sym := sorry, -- adjacency is symmetric
  loopless := sorry -- no loops in the graph, as a hexagon cannot share an edge with itself
}

-- Main statement
theorem least_colors_hexagon_tiling : chromatic_number hexagon_tiling = 3 := 
sorry

end least_colors_hexagon_tiling_l221_221851


namespace distance_from_center_of_circle_to_line_l221_221326

open Real

def circle_passes_through (a : ℝ) :=
  (2 - a) ^ 2 + (1 - a) ^ 2 = a ^ 2

def circle_tangent_to_axes (a : ℝ) :=
  a > 0

def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (2 * a - a - 3) / sqrt (2 ^ 2 + (-1) ^ 2)

theorem distance_from_center_of_circle_to_line
  (a : ℝ) (h1 : circle_passes_through a) (h2 : circle_tangent_to_axes a) :
  distance_from_center_to_line a = 2 * sqrt 5 / 5 :=
sorry

end distance_from_center_of_circle_to_line_l221_221326


namespace sharpened_length_l221_221930

theorem sharpened_length (original_length new_length : ℕ) (h1 : original_length = 31) (h2 : new_length = 14) :
  original_length - new_length = 17 :=
by {
  rw [h1, h2],
  norm_num,
  sorry,
}

end sharpened_length_l221_221930


namespace ellipse_equation_slope_of_line_AB_max_area_triangle_λ_l221_221718

-- Definitions for the given conditions
def point_P := (-1:ℝ, 3/2:ℝ)
def ellipse (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → True
def F1 := (-1:ℝ, 0:ℝ)
def F2 := (1:ℝ, 0:ℝ)
def origin := (0:ℝ, 0:ℝ)
def perpendicular_to_x_axis (p1 p2 : ℝ × ℝ) := p1.2 = p2.2

-- Conditions in Lean
axiom a_gt_b_gt_zero : ∀ a b : ℝ, a > b → b > 0
axiom P_on_ellipse : ∃ a b : ℝ, ellipse a b point_P.1 point_P.2
axiom F1_position : F1 = (-1:ℝ, 0:ℝ)
axiom F2_position : F2 = (1:ℝ, 0:ℝ)
axiom PF1_perpendicular : perpendicular_to_x_axis F1 point_P

-- Theorems to be proved
theorem ellipse_equation : ∃ a b : ℝ, a > b → b > 0 → a = 2 ∧ b^2 = 3 ∧ ellipse a b =
                          fun x y => x^2/4 + y^2/3 = 1 := 
begin
  sorry
end

theorem slope_of_line_AB : ∀ (A B : ℝ × ℝ) (λ : ℝ), 
  0 < λ ∧ λ < 4 ∧ λ ≠ 2 ∧ (vector (point_P) + vector (A) + vector (B) = λ * (vector point_P)) →
  slope (A, B) = 1/2 := 
begin
  sorry
end

theorem max_area_triangle_λ : ∀ (A B : ℝ × ℝ) (λ : ℝ), 
  0 < λ ∧ λ < 4 ∧ λ ≠ 2 ∧ (vector (point_P) + vector (A) + vector (B) = λ * (vector point_P)) →
  (max_area_of_triangle (point_P) (A) (B) = 9/2) → λ = 3 := 
begin
  sorry
end

end ellipse_equation_slope_of_line_AB_max_area_triangle_λ_l221_221718


namespace circle_distance_to_line_l221_221369

-- Define the conditions
def circle_pass_through (P : ℝ × ℝ) (a : ℝ) : Prop :=
  let (x, y) := P
  (x - a)^2 + (y - a)^2 = a^2

def is_tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def distance_to_line (center : ℝ × ℝ) (L : ℝ × ℝ → ℝ) : ℝ :=
  let (x₁, y₁) := center
  abs (L (x₁, y₁)) / sqrt ((L (1,0))^2 + (L (0,1))^2)

-- Define the line equation as a function
def line_eq (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  2*x - y - 3

-- Define the main proof goal
theorem circle_distance_to_line (a : ℝ) (h1 : circle_pass_through (2, 1) a) (h2 : is_tangent_to_axes a) :
  distance_to_line (a, a) line_eq = 2*sqrt 5 / 5 :=
sorry

end circle_distance_to_line_l221_221369


namespace max_rooks_on_chessboard_l221_221005

theorem max_rooks_on_chessboard (board_size : ℕ) (max_attacks : ℕ) 
  (h1 : board_size = 8) 
  (h2 : max_attacks = 1) : 
  ∃ k, k = 10 ∧ 
  ∀ (placements : list (ℕ × ℕ)), 
    (∀ (r1 r2 : (ℕ × ℕ)), r1 ∈ placements → r2 ∈ placements → 
     r1 ≠ r2 → 
     (r1.1 = r2.1 ∨ r1.2 = r2.2) → 
     abs (r1.1 - r2.1) + abs (r1.2 - r2.2) ≤ max_attacks) → 
    k = placements.length := by
  sorry

end max_rooks_on_chessboard_l221_221005


namespace edge_length_of_inscribed_cube_l221_221270

-- Define the conditions
def surface_area_sphere (r : ℝ) : ℝ := 4 * π * r^2
def radius_from_surface_area (area : ℝ) : ℝ := real.sqrt (area / (4 * π))
def inscribed_cube_edge (r : ℝ) : ℝ := (2 * r) / real.sqrt 3

-- Define the proof problem
theorem edge_length_of_inscribed_cube (r : ℝ) : 
  surface_area_sphere r = 4 * π →
  inscribed_cube_edge r = 2 * real.sqrt 3 / 3 :=
by
  sorry

end edge_length_of_inscribed_cube_l221_221270


namespace solution_is_unique_l221_221289

def conditions (x y : ℝ) : Prop :=
  (x/y + y/x) * (x + y) = 15 ∧
  (x^2/y^2 + y^2/x^2) * (x^2 + y^2) = 85

theorem solution_is_unique : ∀ x y : ℝ, conditions x y → (x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2) :=
by
  intro x y
  assume h : conditions x y
  sorry

end solution_is_unique_l221_221289


namespace ratio_area_octagons_correct_l221_221103

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221103


namespace leftmost_three_nonzero_digits_of_ring_arrangements_l221_221709

noncomputable def num_ring_arrangements : ℕ := 
  Nat.binomial 10 6 * Nat.factorial 6 * Nat.binomial 9 3

def leftmost_three_nonzero_digits (n : ℕ) : ℕ := 
  (n / 10^(Nat.log10 n - 2))

theorem leftmost_three_nonzero_digits_of_ring_arrangements : 
  leftmost_three_nonzero_digits num_ring_arrangements = 126 :=
  by sorry

end leftmost_three_nonzero_digits_of_ring_arrangements_l221_221709


namespace circle_tangent_distance_l221_221306

theorem circle_tangent_distance 
    (a : ℝ) (h_center : ∃ a : ℝ, a > 0 ∧ (circle_center = (a, a)) ∧ ((2 - a)^2 + (1 - a)^2 = a^2)) 
    (h_line : ∀ x y, distance_to_line (circle_center x y) (2 * x - y - 3) = 2 / sqrt (2^2 + 1^2)) :
    distance_to_line (circle_center 1 1) (2 * 1 - 1 - 3) = 2 * sqrt 5 / 5 ∧ distance_to_line (circle_center 5 5) (2 * 5 - 5 - 3) = 2 * sqrt 5 / 5 :=
by
    sorry

end circle_tangent_distance_l221_221306


namespace concentric_circles_area_difference_l221_221901

/-- Two concentric circles with radii 12 cm and 7 cm have an area difference of 95π cm² between them. -/
theorem concentric_circles_area_difference :
  let r1 := 12
  let r2 := 7
  let area_larger := Real.pi * r1^2
  let area_smaller := Real.pi * r2^2
  let area_difference := area_larger - area_smaller
  area_difference = 95 * Real.pi := by
sorry

end concentric_circles_area_difference_l221_221901


namespace ratio_of_octagon_areas_l221_221079

-- Define the relevant terms and conditions
noncomputable def radius_of_circle (r : ℝ) := r
noncomputable def side_length_inscribed_octagon (r : ℝ) := r * Real.sqrt 2
noncomputable def side_length_circumscribed_octagon (r : ℝ) := r * Real.sqrt 2

-- Area of a regular octagon in terms of its side length
noncomputable def area_regular_octagon (s : ℝ) := 2 * (1 + Real.sqrt 2) * s^2

-- Areas of inscribed and circumscribed octagons
noncomputable def area_inscribed_octagon (r : ℝ) := area_regular_octagon (side_length_inscribed_octagon r)
noncomputable def area_circumscribed_octagon (r : ℝ) := area_regular_octagon (side_length_circumscribed_octagon r)

-- The main statement to prove
theorem ratio_of_octagon_areas (r : ℝ) : 
  area_circumscribed_octagon r / area_inscribed_octagon r = 1 :=
by
  sorry

end ratio_of_octagon_areas_l221_221079


namespace E_150_eq_2975_l221_221437

def E (n : ℕ) : ℕ :=
  ∑ i in finset.range(n + 1), i * (n / (5 ^ i))

theorem E_150_eq_2975 : E 150 = 2975 := 
by 
  sorry

end E_150_eq_2975_l221_221437


namespace total_toothpicks_l221_221589

theorem total_toothpicks (long short : ℕ) (block_size : ℕ) (missing_block : ℕ) :
  long = 30 → 
  short = 20 → 
  block_size = 2 → 
  missing_block = 4 → 
  (long + 1) * short + (short + 1) * long - missing_block = 1242 := 
by {
  intros,
  sorry
}

end total_toothpicks_l221_221589


namespace sum_of_opposites_is_zero_l221_221378

theorem sum_of_opposites_is_zero (a b : ℚ) (h : a = -b) : a + b = 0 := 
by sorry

end sum_of_opposites_is_zero_l221_221378


namespace smallest_five_digit_divisible_by_2_3_8_9_l221_221908

-- Definitions for the conditions given in the problem
def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000
def divisible_by (n d : ℕ) : Prop := d ∣ n

-- The main theorem stating the problem
theorem smallest_five_digit_divisible_by_2_3_8_9 :
  ∃ n : ℕ, is_five_digit n ∧ divisible_by n 2 ∧ divisible_by n 3 ∧ divisible_by n 8 ∧ divisible_by n 9 ∧ n = 10008 :=
sorry

end smallest_five_digit_divisible_by_2_3_8_9_l221_221908


namespace popcorn_percentage_l221_221052

variable (x : ℕ)
noncomputable def ticket_cost : ℕ := 5
noncomputable def popcorn_cost : ℕ := (x * ticket_cost) / 100
noncomputable def soda_cost : ℕ := (50 * popcorn_cost) / 100

def total_cost (tickets popcorns sodas : ℕ) : ℕ :=
  tickets * ticket_cost + popcorns * popcorn_cost + sodas * soda_cost

theorem popcorn_percentage (h : total_cost 4 2 4 = 36) : x = 80 :=
by {
  sorry
  -- The steps of the proof would go here,
  -- but we are only required to state the theorem without proof
}

end popcorn_percentage_l221_221052


namespace factorization_1_factorization_2_l221_221204

variables {x y m n : ℝ}

theorem factorization_1 : x^3 + 2 * x^2 * y + x * y^2 = x * (x + y)^2 :=
sorry

theorem factorization_2 : 4 * m^2 - n^2 - 4 * m + 1 = (2 * m - 1 + n) * (2 * m - 1 - n) :=
sorry

end factorization_1_factorization_2_l221_221204


namespace zero_neither_pos_nor_neg_l221_221560

theorem zero_neither_pos_nor_neg :
  ¬ (0 > 0) ∧ ¬ (0 < 0) :=
by
  split
  -- Proof that zero is not positive
  exact Nat.not_lt_zero 0
  -- Proof that zero is not negative
  exact Nat.not_lt_zero 0

end zero_neither_pos_nor_neg_l221_221560


namespace distance_between_points_l221_221542

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem distance_between_points :
  distance (2, -2) (8, 8) = real.sqrt 136 :=
by
  sorry

end distance_between_points_l221_221542


namespace oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l221_221391

-- Definitions for oil consumption per person
def oilConsumptionWest : ℝ := 55.084
def oilConsumptionNonWest : ℝ := 214.59
def oilConsumptionRussia : ℝ := 1038.33

-- Lean statements
theorem oilProductionPerPerson_west : oilConsumptionWest = 55.084 := by
  sorry

theorem oilProductionPerPerson_nonwest : oilConsumptionNonWest = 214.59 := by
  sorry

theorem oilProductionPerPerson_russia : oilConsumptionRussia = 1038.33 := by
  sorry

end oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l221_221391


namespace incorrect_statement_C_l221_221561

theorem incorrect_statement_C :
  (∀ r : ℚ, ∃ p : ℝ, p = r) ∧  -- Condition A: All rational numbers can be represented by points on the number line.
  (∀ x : ℝ, x = 1 / x → x = 1 ∨ x = -1) ∧  -- Condition B: The reciprocal of a number equal to itself is ±1.
  (∀ f : ℚ, ∃ q : ℝ, q = f) →  -- Condition C (negation of C as presented): Fractions cannot be represented by points on the number line.
  (∀ x : ℝ, abs x ≥ 0) ∧ (∀ x : ℝ, abs x = 0 ↔ x = 0) →  -- Condition D: The number with the smallest absolute value is 0.
  false :=                      -- Prove that statement C is incorrect
by
  sorry

end incorrect_statement_C_l221_221561


namespace difficulty_degree_l221_221385

-- We define the scores given by the judges
def scores : List ℝ := [7.5, 8.8, 9.0, 6.0, 8.5]

-- Remove the highest and lowest scores and sum the remaining scores
def sum_remaining_scores (scores : List ℝ) : ℝ :=
  let sorted_scores := scores.sorted
  (sorted_scores.drop 1).init.sum

-- The given point value
def point_value : ℝ := 79.36

-- Sum of remaining scores
def remaining_scores_sum : ℝ := sum_remaining_scores scores

-- Given equation for point value and degree of difficulty
def degree_of_difficulty : ℝ :=
  point_value / remaining_scores_sum

-- The statement to prove
theorem difficulty_degree :
  degree_of_difficulty = 3.2 :=
by
  unfold degree_of_difficulty
  unfold remaining_scores_sum
  unfold sum_remaining_scores
  simp
  sorry

end difficulty_degree_l221_221385


namespace yura_score_l221_221648

theorem yura_score (total_competitions points_per_competition dima_score misha_score yura_score : ℕ) 
  (h1 : points_per_competition = 4)
  (h2 : total_competitions = 10)
  (h3 : dima_score = 22)
  (h4 : misha_score = 8)
  (total_points : ℕ := points_per_competition * total_competitions)
  (total_dima_misha : ℕ := dima_score + misha_score)
  (total_yura : ℕ := total_points - total_dima_misha) :
  yura_score = total_yura :=
by
  simp [total_yura]
  sorry

end yura_score_l221_221648


namespace brendan_fish_caught_afternoon_l221_221174

theorem brendan_fish_caught_afternoon (morning_fish : ℕ) (thrown_fish : ℕ) (dads_fish : ℕ) (total_fish : ℕ) :
  morning_fish = 8 → thrown_fish = 3 → dads_fish = 13 → total_fish = 23 → 
  (morning_fish - thrown_fish) + dads_fish + brendan_afternoon_catch = total_fish → 
  brendan_afternoon_catch = 5 :=
by
  intros morning_fish_eq thrown_fish_eq dads_fish_eq total_fish_eq fish_sum_eq
  sorry

end brendan_fish_caught_afternoon_l221_221174


namespace mul_equiv_l221_221922

theorem mul_equiv :
  (213 : ℝ) * 16 = 3408 →
  (16 : ℝ) * 21.3 = 340.8 :=
by
  sorry

end mul_equiv_l221_221922


namespace inverse_of_A_cubed_l221_221751

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3, 7],
    ![-2, -5]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3)⁻¹ = ![![13, -15],
                     ![-14, -29]] :=
by
  sorry

end inverse_of_A_cubed_l221_221751


namespace reciprocal_of_repeating_decimal_l221_221547

theorem reciprocal_of_repeating_decimal : 
  (1 : ℚ) / (34 / 99 : ℚ) = 99 / 34 :=
by sorry

end reciprocal_of_repeating_decimal_l221_221547


namespace billy_horses_l221_221969

theorem billy_horses (each_horse_oats_per_meal : ℕ) (meals_per_day : ℕ) (total_oats_needed : ℕ) (days : ℕ) 
    (h_each_horse_oats_per_meal : each_horse_oats_per_meal = 4)
    (h_meals_per_day : meals_per_day = 2)
    (h_total_oats_needed : total_oats_needed = 96)
    (h_days : days = 3) :
    (total_oats_needed / (days * (each_horse_oats_per_meal * meals_per_day)) = 4) :=
by
  sorry

end billy_horses_l221_221969


namespace part_a_possible_part_b_possible_part_c_impossible_l221_221459

-- Definitions
def color (ℝ → Prop) := sorry -- assume that color is a function from real numbers to a property (either blue or red)

def is_purple (x : ℝ) (colors : ℝ → Prop) : Prop :=
  ∀ ε > 0, ∃ y z ∈ Icc (x - ε) (x + ε), colors y ∧ ¬ colors z

-- Part (a)
theorem part_a_possible (colors : ℝ → Prop) :
  (∀ x : ℝ, is_purple x colors) ↔ True :=
sorry

-- Part (b)
theorem part_b_possible (colors : ℝ → Prop) :
  (∀ x : ℤ, is_purple x colors) ∧ (∀ y ∉ {z : ℤ | True} , ¬ is_purple y colors) ↔ True := 
sorry

-- Part (c)
theorem part_c_impossible (colors : ℝ → Prop) :
  (∀ x : ℚ, is_purple x colors) ∧ (∀ y ∉ {q : ℚ | True} , ¬ is_purple y colors) ↔ False :=
sorry

end part_a_possible_part_b_possible_part_c_impossible_l221_221459


namespace solution_count_l221_221237

theorem solution_count :
  { n : Nat //
     ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≤ y ∧ y ≤ z ∧ (1/. (x: ℚ) + 1/. (y: ℚ) + 1/. (z: ℚ) = 1/2) } = 10 :=
sorry

end solution_count_l221_221237


namespace probability_at_least_one_white_ball_l221_221578

theorem probability_at_least_one_white_ball :
  let n := 30
  let k := 2
  let white := 4
  let red := n - white
  let combinations (n k : ℕ) := nat.desc_factorial n k / nat.factorial k
  let probability := (combinations red 1 * combinations white 1 + combinations white 2) / combinations n 2
  probability = by sorry := 
begin
  let n := 30
  let k := 2
  let white := 4
  let red := n - white
  let combinations (n k : ℕ) := nat.desc_factorial n k / nat.factorial k
  let probability := (combinations red 1 * combinations white 1 + combinations white 2) / combinations n 2
  sorry
end

end probability_at_least_one_white_ball_l221_221578


namespace effective_speed_against_current_l221_221051

theorem effective_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (headwind_speed : ℝ)
  (obstacle_reduction_pct : ℝ)
  (h_speed_with_current : speed_with_current = 25)
  (h_speed_of_current : speed_of_current = 4)
  (h_headwind_speed : headwind_speed = 2)
  (h_obstacle_reduction_pct : obstacle_reduction_pct = 0.15) :
  let speed_in_still_water := speed_with_current - speed_of_current
  let speed_against_current_headwind := speed_in_still_water - speed_of_current - headwind_speed
  let reduction_due_to_obstacles := obstacle_reduction_pct * speed_against_current_headwind
  let effective_speed := speed_against_current_headwind - reduction_due_to_obstacles
  effective_speed = 12.75 := by
{
  sorry
}

end effective_speed_against_current_l221_221051


namespace total_profit_l221_221164

-- Define the relevant variables and conditions
variables (x y : ℝ) -- Cost prices of the two music players

-- Given conditions
axiom cost_price_first : x * 1.2 = 132
axiom cost_price_second : y * 1.1 = 132

theorem total_profit : 132 + 132 - y - x = 34 :=
by
  -- The proof body is not required
  sorry

end total_profit_l221_221164


namespace find_line_eq_l221_221050

-- Definition of points and conditions in the problem
structure Point :=
(x : ℝ)
(y : ℝ)

def P : Point := ⟨-1, 3⟩

def line1 (x y : ℝ) : Prop := x - 2*y + 3 = 0

def perp_line (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- The line passing through P and perpendicular to the given line
def line_passing_through_P_perpendicular : ∀ (x y : ℝ), Prop :=
  λ x y, ∃ (m : ℝ), line1 x y ∧ perp_line (-2) m ∧ y - 3 = m * (x + 1)

theorem find_line_eq : ∀ x y : ℝ, line_passing_through_P_perpendicular x y → 2*x + y - 1 = 0 :=
sorry

end find_line_eq_l221_221050


namespace triangle_perpendicular_l221_221687

noncomputable def triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :=
  ∃ (O_B O_C I : Type) [metric_space O_B] [metric_space O_C] [metric_space I], 
  ∃ (F : Type) [metric_space F], ∃ (E : Type) [metric_space E], 
  ∃ (P : Type) [metric_space P], 
  ∃ (touch_AB : O_B) (touch_AC : O_C) (intersection_OC_E_OB_F : P),
  P I ⊥ B C

-- Statement without proof
theorem triangle_perpendicular (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
(O_B O_C I : Type) [metric_space O_B] [metric_space O_C] [metric_space I] 
(F : Type) [metric_space F] (E : Type) [metric_space E] 
(P : Type) [metric_space P] 
(touch_AB : F = I)
(touch_AC : E = I)
(intersection_OC_E_OB_F : ∃ (E : Type) (O_C O_B P : Type), 
  P = intersection (line_through O_C E) (line_through O_B F)) :
P ⊥ (B C) :=
sorry

end triangle_perpendicular_l221_221687


namespace function_transformation_l221_221497

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x + (Real.pi / 4))

theorem function_transformation :
  ∀ x: ℝ, f x = Math.sin (2 * x + (Real.pi / 4)) :=
sorry

end function_transformation_l221_221497


namespace find_b_l221_221507

theorem find_b (a b c : ℚ) 
  (h1 : a + b + c = 120)
  (h2 : a + 8 = b - 3)
  (h3 : a + 8 = 3c) : b = 396 / 7 := 
sorry

end find_b_l221_221507


namespace find_chocolate_boxes_l221_221633

section
variable (x : Nat)
variable (candy_per_box : Nat := 8)
variable (caramel_boxes : Nat := 3)
variable (total_candy : Nat := 80)

theorem find_chocolate_boxes :
  8 * x + candy_per_box * caramel_boxes = total_candy -> x = 7 :=
by
  sorry
end

end find_chocolate_boxes_l221_221633


namespace arrangements_count_l221_221965

open Finset

-- Definitions for volunteer and elderly person types
inductive Volunteer : Type
| A | B | C | D | E | F

inductive Elderly : Type
| X | Y | Z

open Volunteer Elderly

-- A function that counts the valid arrangements given the constraints
def count_arrangements : Nat :=
  let volunteers := {A, B, C, D, E, F}
  let elderly := {X, Y, Z}
  let pairs := { (A, Y), (A, Z), (B, X), (B, Z), (C, X), (C, Y), (C, Z), (D, X), (D, Y), (D, Z), (E, X), (E, Y), (E, Z), (F, X), (F, Y), (F, Z) }

  -- Excluding invalid pairs based on constraints
  let valid_pairs := pairs.erase (A, X).erase (B, Y)
  
  -- Compute the number of valid ways to arrange the pairs
  let arrangements := valid_pairs.card
  arrangements

theorem arrangements_count : count_arrangements = 42 :=
by
  -- provide the actual proof here
  sorry

end arrangements_count_l221_221965


namespace ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221115

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
  let cos22_5 := (√(2 + √2) / 2)
  let r_prime := r / cos22_5
  let area_ratio := (r_prime / r)^2
  area_ratio

theorem ratio_of_area_of_inscribed_and_circumscribed_octagons (r : ℝ) :
  area_ratio_of_octagons r = (4 - 2 * √2) / 2 := by
  sorry

end ratio_of_area_of_inscribed_and_circumscribed_octagons_l221_221115


namespace work_completion_time_l221_221919

theorem work_completion_time (a b c : Type) (work : Type)
  (work_rate : b → ℝ) (WorkDone : work → b → ℝ)
  (a_rate : a → b → Prop) (c_rate : c → b → Prop)
  (b_work_done : work → 30) :
  (2 * work_rate b + work_rate b + 3 * work_rate b = 1/5) → (b_work_done work = 30) →
  (1/(2 * work_rate b + work_rate b + 3 * work_rate b) = 5) :=
by
  sorry

end work_completion_time_l221_221919


namespace complex_conjugate_solution_l221_221445

-- Definitions and conditions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 
def abs_eq_one (z : ℂ) : Prop := abs z = 1

-- The main theorem to prove
theorem complex_conjugate_solution (z : ℂ) (h1 : abs_eq_one z) (h2 : is_pure_imaginary ((3 + 4 * complex.I) * z)) :
  z.conj = (4/5 - 3/5 * complex.I) ∨ z.conj = (-4/5 + 3/5 * complex.I) :=
sorry

end complex_conjugate_solution_l221_221445


namespace house_cost_l221_221190

theorem house_cost
  (d : ℕ) (y : ℕ) (p : ℕ) (m : ℕ)
  (hd : d = 40)
  (hy : y = 10)
  (hp : p = 2)
  (hm : m = 12) :
  d + y * m * p = 280 :=
by 
  rw [hd, hy, hp, hm]
  norm_num

end house_cost_l221_221190


namespace constant_ratio_arithmetic_progressions_l221_221425

theorem constant_ratio_arithmetic_progressions
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d p a1 b1 : ℝ)
  (h_a : ∀ k : ℕ, a (k + 1) = a1 + k * d)
  (h_b : ∀ k : ℕ, b (k + 1) = b1 + k * p)
  (h_pos : ∀ k : ℕ, a (k + 1) > 0 ∧ b (k + 1) > 0)
  (h_int : ∀ k : ℕ, ∃ n : ℤ, (a (k + 1) / b (k + 1)) = n) :
  ∃ r : ℝ, ∀ k : ℕ, (a (k + 1) / b (k + 1)) = r :=
by
  sorry

end constant_ratio_arithmetic_progressions_l221_221425


namespace number_of_carbons_l221_221943

-- Definitions of given conditions
def molecular_weight (total_c total_h total_o c_weight h_weight o_weight : ℕ) := 
    total_c * c_weight + total_h * h_weight + total_o * o_weight

-- Given values
def num_hydrogen_atoms : ℕ := 8
def num_oxygen_atoms : ℕ := 2
def molecular_wt : ℕ := 88
def atomic_weight_c : ℕ := 12
def atomic_weight_h : ℕ := 1
def atomic_weight_o : ℕ := 16

-- The theorem to be proved
theorem number_of_carbons (num_carbons : ℕ) 
    (H_hydrogen : num_hydrogen_atoms = 8)
    (H_oxygen : num_oxygen_atoms = 2)
    (H_molecular_weight : molecular_wt = 88)
    (H_atomic_weight_c : atomic_weight_c = 12)
    (H_atomic_weight_h : atomic_weight_h = 1)
    (H_atomic_weight_o : atomic_weight_o = 16) :
    molecular_weight num_carbons num_hydrogen_atoms num_oxygen_atoms atomic_weight_c atomic_weight_h atomic_weight_o = molecular_wt → 
    num_carbons = 4 :=
by
  intros h
  sorry 

end number_of_carbons_l221_221943


namespace min_square_area_l221_221675

-- Define the line equation
def line_eq (x : ℝ) := 2 * x + 5

-- Define the parabola equation
def parabola_eq (x : ℝ) := x^2 - 4 * x

-- Define the square vertices distance formula based on parabola and line equations
def square_distance (x_1 x_2 : ℝ) :=
  let k := - (x_1 * x_2) in
  let d_sq := 5 * ((x_1 - x_2) ^ 2) - 180 - 20 * k in
  d_sq + (20 * (abs (k - 5) * (1 / real.sqrt 5))) ^ 2

-- Hypothesis: The minimum area of the square given the problem constraints.
theorem min_square_area : ∃ (area : ℝ), area ≈ 3.4 := by
  sorry

end min_square_area_l221_221675


namespace value_set_expression_locus_midpoint_AB_l221_221985

open_locale classical

variables (R r : ℝ) (hRr : R > r)
          (P : EuclideanSpace ℝ (fin 2))
          (hP : ∥P∥ = r)
          (B : EuclideanSpace ℝ (fin 2))
          (hB : ∥B∥ = R)

-- Definition of point C as the intersection of line BP with the larger circle
def C (B P : EuclideanSpace ℝ (fin 2)) : EuclideanSpace ℝ (fin 2) := 
  sorry -- Placeholder for the actual intersection point definition.

-- Definition of point A as the point on the smaller circle perpendicular to BP
def A (B P : EuclideanSpace ℝ (fin 2)) : EuclideanSpace ℝ (fin 2) :=
  sorry -- Placeholder for the actual perpendicular line intersection definition.

-- Prove that the set of values of the expression BC² + CA² + AB² is {6R² + 2r²}
theorem value_set_expression (B C A : EuclideanSpace ℝ (fin 2)) :
  ∥B - C∥^2 + ∥C - A∥^2 + ∥A - P∥^2 = 6 * R^2 + 2 * r^2 :=
sorry

-- Prove that the locus of the midpoint of the segment AB is a circle centered at the midpoint of OP with radius R / 2
theorem locus_midpoint_AB :
  sorry

end value_set_expression_locus_midpoint_AB_l221_221985


namespace complex_sum_eq_i_l221_221509

theorem complex_sum_eq_i : 
  (∑ k in Finset.range 2013, complex.i^(k+1)) = complex.i :=
by
  sorry

end complex_sum_eq_i_l221_221509


namespace problem_l221_221698

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 4
  else 4 * 2^(n - 2)

def b (n : ℕ) : ℕ :=
  if n = 1 then 2
  else n

theorem problem :
  (∑ i in finset.range 2017, 1 / ((b i.succ) * (b (i.succ + 1)))) = (3025 / 4036) :=
by
  sorry

end problem_l221_221698


namespace leftmost_three_nonzero_digits_of_ring_arrangements_l221_221707

noncomputable def num_ring_arrangements : ℕ := 
  Nat.binomial 10 6 * Nat.factorial 6 * Nat.binomial 9 3

def leftmost_three_nonzero_digits (n : ℕ) : ℕ := 
  (n / 10^(Nat.log10 n - 2))

theorem leftmost_three_nonzero_digits_of_ring_arrangements : 
  leftmost_three_nonzero_digits num_ring_arrangements = 126 :=
  by sorry

end leftmost_three_nonzero_digits_of_ring_arrangements_l221_221707


namespace farmer_orchard_gala_trees_l221_221918

noncomputable def farmer_orchard (T F G : ℕ) (h1 : F + 0.1 * T = 153) (h2 : F = 0.75 * T) : Prop :=
  G = T - F - 0.1 * T

theorem farmer_orchard_gala_trees (T F G : ℕ) (h1 : F + 0.1 * T = 153) (h2 : F = 0.75 * T) : farmer_orchard T F G h1 h2 :=
  by
    sorry

end farmer_orchard_gala_trees_l221_221918


namespace final_position_l221_221983

structure Position where
  base : ℝ × ℝ
  stem : ℝ × ℝ

def rotate180 (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectX (pos : Position) : Position :=
  { base := (pos.base.1, -pos.base.2),
    stem := (pos.stem.1, -pos.stem.2) }

def rotateHalfTurn (pos : Position) : Position :=
  { base := (-pos.base.1, -pos.base.2),
    stem := (-pos.stem.1, -pos.stem.2) }

def reflectY (pos : Position) : Position :=
  { base := (-pos.base.1, pos.base.2),
    stem := (-pos.stem.1, pos.stem.2) }

theorem final_position : 
  let initial_pos := Position.mk (1, 0) (0, 1)
  let pos1 := rotate180 initial_pos
  let pos2 := reflectX pos1
  let pos3 := rotateHalfTurn pos2
  let final_pos := reflectY pos3
  final_pos = { base := (-1, 0), stem := (0, -1) } :=
by
  sorry

end final_position_l221_221983


namespace tetrahedron_sphere_radius_l221_221952

variable (a b : ℝ)

theorem tetrahedron_sphere_radius (h1 : a > 0) (h2 : b > 0) (equal_edge : ∀ (x y : ℝ), x = y) : 
  ∃ R : ℝ, R = (√(2 * a * b)) / 2 :=
sorry

end tetrahedron_sphere_radius_l221_221952


namespace ratio_area_octagons_correct_l221_221108

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end ratio_area_octagons_correct_l221_221108


namespace triangle_ratio_l221_221765

theorem triangle_ratio {A B C G E F : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder G] [LinearOrder E] [LinearOrder F]
  (h1 : divides_ratios A C 3 1 F)
  (h2 : midpoint A B G)
  (h3 : intersection_line BC AG E):
  ratio BE EC = 3 / 4 := 
sorry

end triangle_ratio_l221_221765


namespace magnitude_of_z_l221_221784

def z : ℂ := (1 - 3 * Complex.i) / (1 - Complex.i)

theorem magnitude_of_z : Complex.abs z = Real.sqrt 5 :=
  sorry

end magnitude_of_z_l221_221784


namespace platform_length_l221_221567

theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) (L : ℝ)
  (h1 : train_length = 300) (h2 : pole_time = 18) (h3 : platform_time = 39) :
  (train_length / pole_time) = ((train_length + L) / platform_time) → L = 350 :=
by
  intro h
  rw [h1, h2, h3] at h
  calc
    -- Prove step by step if required
    sorry

end platform_length_l221_221567


namespace sum_inverse_b_2014_l221_221030

noncomputable def b (j : ℕ) (n : ℕ) : ℤ :=
  j ^ n * ∏ i in (Finset.filter (≠ j) (Finset.range (n + 1))), (i ^ n - j ^ n)

noncomputable def sum_inverse_b (n : ℕ) : ℚ :=
  ∑ j in (Finset.range (n + 1)), (1 : ℚ) / (b j n)

theorem sum_inverse_b_2014 : sum_inverse_b 2014 = 1 / (Nat.factorial 2014) ^ 2014 := by
  sorry

end sum_inverse_b_2014_l221_221030


namespace find_value_of_fraction_find_area_triangle_l221_221261

-- Definitions for the given conditions
variables (A B C : ℝ) (a b c : ℝ)
variable (d : ℝ) -- diameter of the circumcircle
variable (angleC : ℝ) -- angle C in radians

-- Given conditions
def circumcircle_diameter := d = 4 * real.sqrt 3 / 3
def side_opposite_angles := C = real.pi / 3
def side_lengths := C = real.pi / 3 → ∃ a b c, true

-- Prove the first part
theorem find_value_of_fraction (h1 : circumcircle_diameter) (h2 : side_opposite_angles) :
  ∃ a b c, (a / real.sin A) = (b / real.sin B) = (c / real.sin C) = 2 * (d / 2) → 
            ((a + b + c) / (real.sin A + real.sin B + real.sin C)) = 4 * real.sqrt 3 / 3 :=
sorry

-- Prove the second part
theorem find_area_triangle (h1 : circumcircle_diameter) (h2 : side_opposite_angles) (h3 : ∃ a b, a + b = a * b) :
  ∃ a b c, (a / real.sin A) = (b / real.sin B) = (c / real.sin C) = 2 * (d / 2) → 
            (area_triangle a b C = real.sqrt 3) :=
sorry

end find_value_of_fraction_find_area_triangle_l221_221261


namespace total_balloons_l221_221896

theorem total_balloons
  (g b y r : ℕ)  -- Number of green, blue, yellow, and red balloons respectively
  (equal_groups : g = b ∧ b = y ∧ y = r)
  (anya_took : y / 2 = 84) :
  g + b + y + r = 672 := by
sorry

end total_balloons_l221_221896


namespace derivative_at_one_l221_221732

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x

def f' (x : ℝ) : ℝ := 
  Real.exp x * Real.log x + Real.exp x / x

theorem derivative_at_one : f' 1 = Real.exp 1 := 
by 
  sorry

end derivative_at_one_l221_221732


namespace incorrect_statement_B_l221_221016

variable (x : ℝ)

def statementA : Prop := ∃ x : ℝ, Real.sin x ≥ 1

def statementB : Prop :=
  ¬ (¬ (∃ x : ℝ, x^2 + 1 ≤ 0) → (∀ x : ℝ, x^2 + 1 > 0))

def statementC : Prop := ∀ x : ℝ, x ≤ 0 → Real.exp x ≤ 1

def statementD : Prop :=
  (¬ (∃ x : ℝ, Real.sqrt (x^2) ≠ x)) ↔ (∀ x : ℝ, Real.sqrt (x^2) = x)

theorem incorrect_statement_B : statementB x := 
sorry

end incorrect_statement_B_l221_221016


namespace inverse_variation_l221_221508

theorem inverse_variation (k : ℝ) (y x : ℝ) (h1 : y * x^2 = k) (h2 : y = 6) (h3 : x = 3) : 
  ∃ x, y = 2 ∧ x = 3 * real.sqrt 3 :=
by
  sorry

end inverse_variation_l221_221508


namespace slope_of_perpendicular_line_l221_221181

theorem slope_of_perpendicular_line (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end slope_of_perpendicular_line_l221_221181


namespace equilateral_triangle_sum_constant_l221_221592

theorem equilateral_triangle_sum_constant
  (A B C O X Y Z : Point)
  (hABC : equilateral_triangle A B C)
  (hO : center O A B C)
  (hLine : line_through O X Y Z)
  (hIntersect : intersects_sides X Y Z A B C):
  (1 / (distance O X)^2 + 1 / (distance O Y)^2 + 1 / (distance O Z)^2) = 3 / 2 :=
sorry

end equilateral_triangle_sum_constant_l221_221592


namespace octagon_area_ratio_l221_221131

theorem octagon_area_ratio (r : ℝ) :
  let A_inscribed := 2 * (1 + Real.sqrt 2) * (r * Real.tan (Real.pi / 8))^2 in
  let R := r * (1 / Real.cos (Real.pi / 8)) in
  let A_circumscribed := 2 * (1 + Real.sqrt 2) * (2 * R * Real.sin (Real.pi / 8))^2 in
  A_circumscribed / A_inscribed = 4 * (3 + 2 * Real.sqrt 2) :=
by
  sorry

end octagon_area_ratio_l221_221131


namespace square_cut_into_rectangles_l221_221607

def congruent_rectangles_dimensions (side_length : ℕ) (half_cut : Prop) : Prop :=
  ∃ (length width : ℕ), half_cut → length = 4 ∧ width = 8

theorem square_cut_into_rectangles : 
  ∀ (side_length : ℕ), side_length = 8 →
  ∃ (length width : ℕ), (side_length / 2 = 4) → length = 4 ∧ width = 8 :=
by
  intros side_length h
  use 4
  use 8
  intro h_cut
  split
  . refl
  . exact (by rw h)

end square_cut_into_rectangles_l221_221607


namespace octagon_area_ratio_l221_221156

theorem octagon_area_ratio (r : ℝ) : 
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2 := 
by
  let inscribed_area := 2 * (1 + Real.sqrt 2) * r^2
  let circumscribed_area :=  (2 / Real.sqrt (2 + Real.sqrt 2))^2 * 2 * (1 + Real.sqrt 2) * r^2
  have h : circumscribed_area / inscribed_area = 4 - 2 * Real.sqrt 2
  exact h
  sorry

end octagon_area_ratio_l221_221156


namespace count_valid_integers_l221_221745

/-- 
  Define what it means for the sums to form an increasing arithmetic sequence:
  s₁ = a + b, s₂ = b + c, s₃ = c + d.
  They must satisfy s₂ - s₁ = s₃ - s₂
-/
def is_arithmetic_seq (a b c d : ℕ) : Prop :=
  let s₁ := a + b
  let s₂ := b + c
  let s₃ := c + d
  2 * s₂ = s₁ + s₃

/-- 
  Define the general conditions for the problem:
  1. a ≠ 0 (a must be a non-zero digit)
  2. The four-digit number formed by a, b, c, d must satisfy the arithmetic sequence condition
  3. The total sum of a, b, c, d must equal 20
-/
def valid_number (a b c d : ℕ) : Prop :=
  (a ≠ 0) ∧ is_arithmetic_seq a b c d ∧ (a + b + c + d = 20)

/-- 
  Create a list of all possible four-digit numbers and count how many valid ones there are.
-/
theorem count_valid_integers : 
  {n : ℕ // ∃ a b c d : ℕ,
    digit_1 a ∧ digit_2 b ∧ digit_3 c ∧ digit_4 d ∧ valid_number a b c d ∧ n = 1000 * a + 100 * b + 10 * c + d}.card = 9 :=
by
  sorry

end count_valid_integers_l221_221745


namespace evaluate_expression_l221_221638

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end evaluate_expression_l221_221638


namespace largest_digit_divisible_by_6_l221_221907

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 (N : ℕ) (hN : N ≤ 9) :
  (∃ m : ℕ, 56780 + N = m * 6) ∧ is_even N ∧ is_divisible_by_3 (26 + N) → N = 4 := by
  sorry

end largest_digit_divisible_by_6_l221_221907


namespace f_2_equals_0_l221_221934

def f (x : ℝ) (a b : ℝ) : ℝ := a * sin x + b * x^3 + 1

theorem f_2_equals_0 (a b : ℝ) (h : a * sin (-2 : ℝ) + b * (-2)^3 + 1 = 2) : 
  f 2 a b = 0 :=
by 
  unfold f
  have h_odd : f x a b = a * sin x + b * x^3 + 1,
  sorry 

end f_2_equals_0_l221_221934


namespace find_x_value_l221_221201

theorem find_x_value (x : ℤ)
    (h1 : (5 + 9) / 2 = 7)
    (h2 : (5 + x) / 2 = 10)
    (h3 : (x + 9) / 2 = 12) : 
    x = 15 := 
sorry

end find_x_value_l221_221201


namespace inequality_proof_l221_221840

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) : 
  (x*y^2/z + y*z^2/x + z*x^2/y) ≥ (x^2 + y^2 + z^2) := by
  sorry

end inequality_proof_l221_221840
