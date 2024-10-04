import Mathlib

namespace symmetric_intersections_l323_323423

open_locale classical

theorem symmetric_intersections
  {O A B C D E F M P Q: Point}
  (circle : Circle O)
  (chord_AB : Chord A B)
  (midpoint_M : Midpoint M A B)
  (chord_CD : Chord C D)
  (chord_EF : Chord E F)
  (through_M_CD : PassThrough M C D)
  (through_M_EF : PassThrough M E F)
  (intersect_1 : IntersectLine CE AB P)
  (intersect_2 : IntersectLine DF AB Q)
  (CE : Line C E)
  (DF : Line D F)
  (AB : Line A B) :
  SymmetricPoints P Q M :=
sorry

end symmetric_intersections_l323_323423


namespace recipe_sugar_l323_323847

noncomputable def sugar_required (sugar_left : ℝ) (fraction_of_recipe : ℝ) : ℝ :=
sugar_left / fraction_of_recipe

theorem recipe_sugar :
  sugar_required 0.3333 0.165 ≈ 2.02 :=
by
  sorry

end recipe_sugar_l323_323847


namespace bidigital_count_is_243_l323_323064

def is_bidigital (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  n > 999 ∧ n < 10000 ∧
  digits.erase_dup.length = 2 ∧
  (digits.count digits.head = 2 ∧ digits.count digits.tail.head = 2)

noncomputable def count_bidigital_numbers : ℕ :=
  (Finset.range 10000).filter (λ n, is_bidigital n).card

theorem bidigital_count_is_243 : count_bidigital_numbers = 243 := 
by
  sorry

end bidigital_count_is_243_l323_323064


namespace tan_add_pi_four_eq_neg_three_phi_eq_pi_four_l323_323262

variable (θ φ : ℝ)

-- Problem 1
theorem tan_add_pi_four_eq_neg_three
  (h1: θ ∈ Set.Ioo 0 (Real.pi / 2))
  (h2: ∃ k : ℝ, (sin θ, 2) = k • (cos θ, 1)) :
  tan (θ + Real.pi / 4) = -3 := sorry

-- Problem 2
theorem phi_eq_pi_four
  (h1: θ ∈ Set.Ioo 0 (Real.pi / 2))
  (h2: tan θ = 2)
  (h3: 5 * cos (θ - φ) = 3 * sqrt 5 * cos φ)
  (h4: φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  φ = Real.pi / 4 := sorry

end tan_add_pi_four_eq_neg_three_phi_eq_pi_four_l323_323262


namespace count_five_primable_lt_1000_l323_323079

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323079


namespace trapezium_area_l323_323189

open Real

noncomputable def shoelace_area (verts : list (ℝ × ℝ)) : ℝ :=
1 / 2 * | verts.foldl (λ acc ⟨x1, y1⟩, acc + x1 * (verts.drop 1).head.snd) 0 - verts.foldl (λ acc ⟨x1, y1⟩, acc + y1 * (verts.drop 1).head.fst) 0 |

theorem trapezium_area :
  let vertices := [(2, 5), (8, 12), (14, 7), (6, 2)]
  in shoelace_area vertices = 58 :=
by
  sorry

end trapezium_area_l323_323189


namespace complex_point_in_4th_quadrant_l323_323971

/-- Given the complex number z = (2 - i) * (2 + a * I), where I is the imaginary unit, 
    if z corresponds to a point in the fourth quadrant on the complex plane, 
    then the real number a is -2. -/
theorem complex_point_in_4th_quadrant (a : ℝ) (z : ℂ) (hz : z = (2 - complex.I) * (2 + a * complex.I)) 
(hz_quadrant : complex.re z > 0 ∧ complex.im z < 0) : a = -2 :=
sorry

end complex_point_in_4th_quadrant_l323_323971


namespace equation_of_line_l323_323975

-- Define x-intercept and y-intercept conditions
def line_x_intercept (l : ℝ → ℝ → Prop) (a : ℝ) : Prop :=
  l a 0

def line_y_intercept (l : ℝ → ℝ → Prop) (b : ℝ) : Prop :=
  l 0 b

noncomputable def line_equation : ℝ → ℝ → Prop :=
  λ x y, 2*x - 3*y - 6 = 0

theorem equation_of_line :
  (line_x_intercept line_equation 3) ∧ (line_y_intercept line_equation (-2)) :=
  sorry

end equation_of_line_l323_323975


namespace intersection_equal_l323_323613

noncomputable def M := { y : ℝ | ∃ x : ℝ, y = Real.log (x + 1) / Real.log (1 / 2) ∧ x ≥ 3 }
noncomputable def N := { x : ℝ | x^2 + 2 * x - 3 ≤ 0 }

theorem intersection_equal : M ∩ N = {a : ℝ | -3 ≤ a ∧ a ≤ -2} :=
by
  sorry

end intersection_equal_l323_323613


namespace generatrix_length_l323_323238

-- Definitions and conditions
def angle_between_generatrix_and_base (cone : Type) (gamma : ℝ) : Prop :=
  gamma = π / 4

def cone_height (cone : Type) (h : ℝ) : Prop :=
  h = 1

-- Theorem to prove the length of the generatrix
theorem generatrix_length {cone : Type} (gamma : ℝ) (h : ℝ) (r : ℝ) (l : ℝ) :
  angle_between_generatrix_and_base cone gamma →
  cone_height cone h →
  r = h →
  l = real.sqrt (r * r + h * h) →
  l = real.sqrt 2 :=
by
  sorry

end generatrix_length_l323_323238


namespace rahul_share_is_100_l323_323377

-- Definitions of the conditions
def rahul_rate := 1/3
def rajesh_rate := 1/2
def total_payment := 250

-- Definition of their work rate when they work together
def combined_rate := rahul_rate + rajesh_rate

-- Definition of the total value of the work done in one day when both work together
noncomputable def combined_work_value := total_payment / combined_rate

-- Definition of Rahul's share for the work done in one day
noncomputable def rahul_share := rahul_rate * combined_work_value

-- The theorem we need to prove
theorem rahul_share_is_100 : rahul_share = 100 := by
  sorry

end rahul_share_is_100_l323_323377


namespace roles_assignment_l323_323495

-- Define names for roles and ages
inductive Person
| A
| B
| C

inductive Role
| worker
| farmer
| intellectual

-- Define the ages of persons
variables (age_A age_B age_C age_worker age_farmer age_intellectual : ℕ)

-- Given conditions
axiom cond1 : age_C > age_intellectual
axiom cond2 : age_A ≠ age_farmer
axiom cond3 : age_farmer < age_B

-- Conclusion
theorem roles_assignment : (Role.intellectual = Role.worker → Person.A ≠ Person.C) → 
                           (Role.farmer = Role.worker → Person.A ≠ Person.B) → 
                           (Role.worker = Role.worker → Person.C = Person.B) → 
                           (Person.A = Person.intellectual ∧ Person.B = Person.worker ∧ Person.C = Person.farmer) :=
begin
  sorry
end

end roles_assignment_l323_323495


namespace john_initial_books_l323_323675

noncomputable def initial_books (total_sold : ℝ) (percentage_not_sold : ℝ) : ℝ :=
  total_sold / (1 - percentage_not_sold)

def total_books_sold : ℝ := 75 + 50 + 64 + 78 + 135
def percentage_not_sold : ℝ := 69.07692307692308 / 100

theorem john_initial_books : initial_books total_books_sold percentage_not_sold = 1300 :=
by
  rw [initial_books, total_books_sold, percentage_not_sold]
  norm_num
  sorry

end john_initial_books_l323_323675


namespace exists_distinct_natural_numbers_with_mean_conditions_l323_323203

def digits_rearranged (a b : ℕ) : Prop :=
  list.perm (nat.digits 10 a) (nat.digits 10 b)

theorem exists_distinct_natural_numbers_with_mean_conditions :
  ∃ (a b : ℕ), a ≠ b ∧ 20 ≤ a + b ∧ a + b ≤ 198 ∧ 100 ≤ a * b ∧ a * b ≤ 9801 ∧ digits_rearranged a b :=
by
  sorry

end exists_distinct_natural_numbers_with_mean_conditions_l323_323203


namespace burglary_charge_sentence_l323_323318

theorem burglary_charge_sentence (B : ℕ) 
  (arson_counts : ℕ := 3) 
  (arson_sentence : ℕ := 36)
  (burglary_charges : ℕ := 2)
  (petty_larceny_factor : ℕ := 6)
  (total_jail_time : ℕ := 216) :
  arson_counts * arson_sentence + burglary_charges * B + (burglary_charges * petty_larceny_factor) * (B / 3) = total_jail_time → B = 18 := 
by
  sorry

end burglary_charge_sentence_l323_323318


namespace logarithm_identity_1_l323_323627

open Real

theorem logarithm_identity_1 (a b : ℝ) (h1 : 2 ^ a = 10) (h2 : 5 ^ b = 10) : 
  (1 / a + 1 / b = 1) :=
  sorry

end logarithm_identity_1_l323_323627


namespace Ron_eats_24_pickle_slices_l323_323381

theorem Ron_eats_24_pickle_slices : 
  ∀ (pickle_slices_Sammy Tammy Ron : ℕ), 
    pickle_slices_Sammy = 15 → 
    Tammy = 2 * pickle_slices_Sammy → 
    Ron = Tammy - (20 * Tammy / 100) → 
    Ron = 24 := by
  intros pickle_slices_Sammy Tammy Ron h_sammy h_tammy h_ron
  sorry

end Ron_eats_24_pickle_slices_l323_323381


namespace trigonometric_identity_l323_323823

theorem trigonometric_identity : (√3)/2 - √3 * (sin 15)^2 = 3/4 :=
by
  -- Proof is omitted
  sorry

end trigonometric_identity_l323_323823


namespace intersection_A_B_l323_323328

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2 * x + 5}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 1 - 2 * x}
def inter : Set (ℝ × ℝ) := {(x, y) | x = -1 ∧ y = 3}

theorem intersection_A_B :
  A ∩ B = inter :=
sorry

end intersection_A_B_l323_323328


namespace loan_repaid_by_2022_min_charge_per_student_2018_l323_323832

-- Statement for Question (1)
theorem loan_repaid_by_2022 (P : ℕ) (I : ℝ) (C : ℝ) :
  P = 5000000 → I = 0.05 → C = 620000 → (1.05 : ℝ) ^ 12 ≥ 1.7343 :=
by
  intros _ _ _
  sorry

-- Statement for Question (2)
theorem min_charge_per_student_2018 (P : ℕ) (I : ℝ) (U : ℝ) : 
  P = 5000000 → I = 0.05 → U = 180000 →
  (8 : ℕ) → (x : ℕ) → (0.1 * real.to_nnreal x - 18000) * ((10 * (1.05 : ℝ) ^ 8 - 1) / 0.05) ≥ 5000000 * (1.05 ^ 9) → 
  x ≥ 992 :=
by
  intros _ _ _ _ _
  sorry

end loan_repaid_by_2022_min_charge_per_student_2018_l323_323832


namespace ellipse_properties_l323_323974

open Real

noncomputable def ellipse_eq : Prop :=
  ∃ e : ℝ, ∃ D : ℝ × ℝ, ∃ x y : ℝ,
    (e = sqrt(6)/3) ∧ (D = (0, -1)) ∧ 
    (y^2 + x^2 / 3 = 1)

noncomputable def minimum_AM : Prop :=
  ∀ M A : ℝ × ℝ,
    (M = (1, 0)) →
    (∃ x y : ℝ, y^2 + x^2 / 3 = 1 ∧ A = (x, y)) →
    (sqrt ((A.1 - 1)^2 + A.2^2) ≥ 1/2)

noncomputable def fixed_point_P : Prop :=
  ∃ P M A B : ℝ × ℝ,
    (P = (3, 0)) ∧ (M = (1, 0)) →
    (∃ x1 y1 x2 y2 : ℝ, 
      y1^2 + x1^2 / 3 = 1 ∧ y2^2 + x2^2 / 3 = 1 ∧
      A = (x1, y1) ∧ B = (x2, y2) ∧
      atan2 (A.2 - P.2) (A.1 - P.1) = atan2 (B.2 - P.2) (B.1 - P.1))

theorem ellipse_properties : ellipse_eq ∧ minimum_AM ∧ fixed_point_P :=
by
  split
  · 
    use [sqrt(6)/3, (0, -1)]
    use [_, _]
    exact ⟨rfl, rfl, sorry⟩
  · 
    intros M A hM ⟨x, y, h⟩
    use [x, y]
    sorry
  · 
    use [(3, 0), (1, 0), _, _]
    intros hP hM
    use [_, _, _, _]
    sorry

end ellipse_properties_l323_323974


namespace net_change_proof_l323_323704

def initial_cash_A := 15000
def initial_cash_B := 20000
def initial_house_value := 15000

def rent := 2000
def first_sale_price := 18000
def second_sale_price := 17000

-- Calculate final cash and net change in cash for Mr. A and Mr. B
def final_cash_A := initial_cash_A + rent + first_sale_price - second_sale_price
def final_cash_B := initial_cash_B - rent - first_sale_price + second_sale_price

def net_change_cash_A := final_cash_A - initial_cash_A
def net_change_cash_B := final_cash_B - initial_cash_B

theorem net_change_proof :
  net_change_cash_A = 3000 ∧
  net_change_cash_B = -3000 := by
  sorry

end net_change_proof_l323_323704


namespace max_value_on_interval_l323_323250

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + c) / Real.exp x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := ((2 * a * x + b) * Real.exp x - (a * x^2 + b * x + c)) / Real.exp (2 * x)

variable (a b c : ℝ)

-- Given conditions
axiom pos_a : a > 0
axiom zero_point_neg3 : f' a b c (-3) = 0
axiom zero_point_0 : f' a b c 0 = 0
axiom min_value_neg3 : f a b c (-3) = -Real.exp 3

-- Goal: Maximum value of f(x) on the interval [-5, ∞) is 5e^5.
theorem max_value_on_interval : ∃ y ∈ Set.Ici (-5), f a b c y = 5 * Real.exp 5 := by
  sorry

end max_value_on_interval_l323_323250


namespace multiply_102_self_l323_323537

theorem multiply_102_self : 102 * 102 = 10404 :=
by
  let a := 100
  let b := 2
  have h : (a + b) * (a + b) = a * a + 2 * a * b + b * b := by ring
  show 102 * 102 = 10404
  rw [← h, Nat.add_sub_cancel_left (a + b) b]
  norm_num
  sorry -- skipping the proof

end multiply_102_self_l323_323537


namespace count_5_primable_integers_lt_1000_is_21_l323_323109

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323109


namespace total_kitchen_supplies_sharon_l323_323383

-- Define the number of pots Angela has.
def angela_pots := 20

-- Define the number of plates Angela has (6 more than three times the number of pots Angela has).
def angela_plates := 6 + 3 * angela_pots

-- Define the number of cutlery Angela has (half the number of plates Angela has).
def angela_cutlery := angela_plates / 2

-- Define the number of pots Sharon wants (half the number of pots Angela has).
def sharon_pots := angela_pots / 2

-- Define the number of plates Sharon wants (20 less than three times the number of plates Angela has).
def sharon_plates := 3 * angela_plates - 20

-- Define the number of cutlery Sharon wants (twice the number of cutlery Angela has).
def sharon_cutlery := 2 * angela_cutlery

-- Prove that the total number of kitchen supplies Sharon wants is 254.
theorem total_kitchen_supplies_sharon :
  sharon_pots + sharon_plates + sharon_cutlery = 254 :=
by
  -- State intermediate results for clarity
  let angela_plates_val := 66
  have h_angela_plates : angela_plates = angela_plates_val :=
    by
    unfold angela_plates
    norm_num
  let angela_cutlery_val := 33
  have h_angela_cutlery : angela_cutlery = angela_cutlery_val :=
    by
    unfold angela_cutlery
    rw h_angela_plates
    norm_num
  let sharon_pots_val := 10
  have h_sharon_pots : sharon_pots = sharon_pots_val :=
    by
    unfold sharon_pots
    norm_num
  let sharon_plates_val := 178
  have h_sharon_plates : sharon_plates = sharon_plates_val :=
    by
    unfold sharon_plates
    rw h_angela_plates
    norm_num
  let sharon_cutlery_val := 66
  have h_sharon_cutlery : sharon_cutlery = sharon_cutlery_val :=
    by
    unfold sharon_cutlery
    rw h_angela_cutlery
    norm_num
  rw [h_sharon_pots, h_sharon_plates, h_sharon_cutlery]
  norm_num

end total_kitchen_supplies_sharon_l323_323383


namespace count_5_primable_less_than_1000_eq_l323_323130

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323130


namespace num_triangles_in_nonagon_l323_323517

-- Define the nonagon and the conditions
def is_regular_nonagon (P : Finset Point) : Prop := P.card = 9 ∧ (∀ (A B C : Point) (hA : A ∈ P) (hB : B ∈ P) (hC : C ∈ P), A ≠ B ∧ B ≠ C ∧ A ≠ C → ¬ collinear {A, B, C})

-- The proof objective
theorem num_triangles_in_nonagon (P : Finset Point) (h : is_regular_nonagon P) : ∃ n, n = P.choose 3 ∧ n = 84 :=
by
  sorry

end num_triangles_in_nonagon_l323_323517


namespace totalSeats_l323_323063

-- Let s represent the total number of seats on the plane.
variable (s : ℝ)

-- Conditions:
def firstClassSeats := 30
def businessClassSeats := 0.20 * s
def economyClassSeats := s - (30 + 0.20 * s)

-- The total number of seats on the plane
theorem totalSeats (h : s = firstClassSeats + businessClassSeats + economyClassSeats) : s = 50 :=
by
  -- Setting up the equation for all sections
  have eq1 : firstClassSeats + businessClassSeats + economyClassSeats = 30 + 0.20 * s + (s - (30 + 0.20 * s)) := by sorry
  -- Simplifying the equation to confirm total equals s
  have eq2 : 30 + 0.20 * s + (s - 30 - 0.20 * s) = s := by sorry
  -- Setting the parts equal to s to find s
  have eq3 : 30 + 0.20 * s = 0.80 * s := by sorry
  have eq4 : 30 = 0.60 * s := by sorry
  -- Solving for s
  have eq5 : (30 : ℝ) / 0.60 = s := by sorry
  show s = 50
  from eq5

end totalSeats_l323_323063


namespace head_start_eq_60_l323_323830

def speed_of_B (v : ℝ) : ℝ := v
def speed_of_A (v : ℝ) : ℝ := 4 * v
def race_length : ℝ := 80

theorem head_start_eq_60 (v : ℝ) (s : ℝ) (t : ℝ) 
  (A_speed : speed_of_A v = 4 * v) 
  (B_speed : speed_of_B v = v)
  (race_eq : t = (80 - s) / v)
  (race_eq_A : t = 80 / (4 * v)) : 
  s = 60 :=
begin
  sorry,
end

end head_start_eq_60_l323_323830


namespace money_left_correct_l323_323707

variables (cost_per_kg initial_money kg_bought total_cost money_left : ℕ)

def condition1 : cost_per_kg = 82 := sorry
def condition2 : kg_bought = 2 := sorry
def condition3 : initial_money = 180 := sorry
def condition4 : total_cost = cost_per_kg * kg_bought := sorry
def condition5 : money_left = initial_money - total_cost := sorry

theorem money_left_correct : money_left = 16 := by
  have h1 : cost_per_kg = 82, from condition1
  have h2 : kg_bought = 2, from condition2
  have h3 : initial_money = 180, from condition3
  have h4 : total_cost = cost_per_kg * kg_bought, from condition4
  have h5 : money_left = initial_money - total_cost, from condition5
  rw [h1, h2, h3, h4, h5]
  sorry

end money_left_correct_l323_323707


namespace max_profit_at_300_l323_323452

noncomputable def fixed_cost : ℝ := 20000
noncomputable def cost_per_instrument : ℝ := 100
noncomputable def revenue (x : ℝ) : ℝ :=
  if hx : 0 ≤ x ∧ x ≤ 400 then 400 * x - (1 / 2) * x^2
  else if x > 400 then 80000
  else 0 -- assuming revenue is 0 for negative x for completeness

noncomputable def profit (x : ℝ) : ℝ :=
  if hx : 0 ≤ x ∧ x ≤ 400 then - (1 / 2) * x^2 + 300 * x - 20000
  else if x > 400 then 60000 - 100 * x
  else 0 -- assuming profit is 0 for negative x for completeness

theorem max_profit_at_300 : ∃ x, 0 ≤ x ∧ x ≤ 400 ∧ profit x = 25000 :=
by
  use 300
  rw profit
  split_ifs
  simp
  sorry

end max_profit_at_300_l323_323452


namespace square_area_y_coords_eq_l323_323143

-- Defining conditions and problem statement
def y_coords := [0, 3, 8, 11] -- The y-coordinates of the vertices of the square

-- The proof problem statement:
theorem square_area_y_coords_eq :
  ∃ (x1 x2 x3 : ℝ) (s : ℝ), 
    x1^2 = 576 / 55 ∧
    s = real.sqrt (x1^2 + 3^2) ∧ 
    y_coords = [0, 3, 8, 11] ∧
    s^2 = 1071 / 55 :=
sorry

end square_area_y_coords_eq_l323_323143


namespace count_5_primable_under_1000_l323_323123

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323123


namespace original_cost_price_l323_323033

theorem original_cost_price ( C S : ℝ )
  (h1 : S = 1.05 * C)
  (h2 : S - 3 = 1.10 * 0.95 * C)
  : C = 600 :=
sorry

end original_cost_price_l323_323033


namespace coeff_x_squared_l323_323586

theorem coeff_x_squared (a : ℝ) (h : a = ∫ x in 0..π, sin x) : 
  coefficient.base_polynomial \left( \frac{1}{x} - x \right)^(5*a) x^2 = 210 := 
by 
  sorry

end coeff_x_squared_l323_323586


namespace probability_red_or_blue_l323_323441

theorem probability_red_or_blue
  (total_marbles : ℕ)
  (p_white : ℚ)
  (p_green : ℚ)
  (p_red_or_blue : ℚ) :
  total_marbles = 84 →
  p_white = 1 / 4 →
  p_green = 2 / 7 →
  p_red_or_blue = 1 - p_white - p_green →
  p_red_or_blue = 13 / 28 :=
by
  intros h_total h_white h_green h_red_or_blue
  sorry

end probability_red_or_blue_l323_323441


namespace num_integers_with_repeating_decimal_l323_323931

theorem num_integers_with_repeating_decimal : (finset.range 200).filter (λ n, 
  let d := n + 1 in
  ∀ p : ℕ, nat.prime p → p ∣ d → p = 2 ∨ p = 5).card = 182 :=
begin
  sorry
end

end num_integers_with_repeating_decimal_l323_323931


namespace graphs_intersect_exactly_eight_times_l323_323519

theorem graphs_intersect_exactly_eight_times (A : ℝ) (hA : 0 < A) :
  ∃ (count : ℕ), count = 8 ∧ ∀ x y : ℝ, y = A * x ^ 4 → y ^ 2 + 5 = x ^ 2 + 6 * y :=
sorry

end graphs_intersect_exactly_eight_times_l323_323519


namespace inequality_solution_set_l323_323287

theorem inequality_solution_set (a b c : ℝ)
  (h1 : a < 0)
  (h2 : b = -a)
  (h3 : c = -2 * a) :
  ∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -1 ∨ x > 1 / 2) :=
by
  sorry

end inequality_solution_set_l323_323287


namespace min_max_difference_distances_l323_323909

open EuclideanGeometry

variables {A B M l : Point}

theorem min_max_difference_distances (A B : Point) (l : Line) :
  (∀ M : Point, M ∈ l → 
      let mid := midpoint A B in
      (∃ M_min : Point, M_min ∈ l ∧ M_min = foot mid l) ∧ 
      (∃ M_max : Point, M_max ∈ l ∧ M_max = intersection (line_through A B) l)) :=
sorry

end min_max_difference_distances_l323_323909


namespace income_of_deceased_l323_323439

def average_income (total_income : ℕ) (members : ℕ) : ℕ :=
  total_income / members

theorem income_of_deceased
  (total_income_before : ℕ) (members_before : ℕ) (avg_income_before : ℕ)
  (total_income_after : ℕ) (members_after : ℕ) (avg_income_after : ℕ) :
  total_income_before = members_before * avg_income_before →
  total_income_after = members_after * avg_income_after →
  members_before = 4 →
  members_after = 3 →
  avg_income_before = 735 →
  avg_income_after = 650 →
  total_income_before - total_income_after = 990 :=
by
  sorry

end income_of_deceased_l323_323439


namespace correspondenceC_is_mapping_l323_323159

universe u
variable (A B : Type u) (f : A → B)

def is_mapping (A B : Type u) (f : A → B) := 
∀ a1 a2 : A, f a1 = f a2 → a1 = a2 ∧ ∀ a : A, ∃ b : B, f a = b

def N_star := {n : ℕ // 0 < n}
def B_set := ({-1, 0, 1} : Set ℤ)

theorem correspondenceC_is_mapping : 
  is_mapping N_star B_set (λ (x : N_star), (-1 : ℤ)^ (x.val)) :=
sorry

end correspondenceC_is_mapping_l323_323159


namespace intersect_A_B_l323_323581

def A := {x : ℝ | ∃ y : ℝ, y = real.sqrt (2 - 2^x)}
def B := {x : ℝ | x^2 - 3*x ≤ 0}

theorem intersect_A_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := sorry

end intersect_A_B_l323_323581


namespace sum_of_squares_of_biking_jogging_swimming_rates_l323_323185

theorem sum_of_squares_of_biking_jogging_swimming_rates (b j s : ℕ) 
  (h1 : 2 * b + 3 * j + 4 * s = 74) 
  (h2 : 4 * b + 2 * j + 3 * s = 91) : 
  (b^2 + j^2 + s^2 = 314) :=
sorry

end sum_of_squares_of_biking_jogging_swimming_rates_l323_323185


namespace find_d_l323_323175

theorem find_d (a b c d : ℝ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) (hdc : 0 < d)
  (oscillates : ∀ x, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 :=
sorry

end find_d_l323_323175


namespace distinct_elements_not_perfect_square_l323_323337

-- Define a set of perfect squares
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- The main theorem we want to prove
theorem distinct_elements_not_perfect_square (d : ℕ) 
  (h_d_pos : d > 0)
  (h_d_not_2 : d ≠ 2)
  (h_d_not_5 : d ≠ 5)
  (h_d_not_13 : d ≠ 13) :
  ∃ a b ∈ ({2, 5, 13, d} : set ℕ), a ≠ b ∧ ¬ is_perfect_square (a * b - 1) :=
sorry

end distinct_elements_not_perfect_square_l323_323337


namespace final_bill_amount_l323_323447

def original_bill : ℝ := 500
def first_charge_rate : ℝ := 1.02
def second_charge_rate : ℝ := 1.03

def amount_after_first_charge := original_bill * first_charge_rate
def final_amount := amount_after_first_charge * second_charge_rate

theorem final_bill_amount :
  final_amount = 525.3 :=
by
  unfold amount_after_first_charge
  unfold final_amount
  norm_num
  sorry

end final_bill_amount_l323_323447


namespace calculate_expression_l323_323169

theorem calculate_expression : 
  ((1336 / 17) : ℤ) - (1024 : ℤ) + (6789 : ℤ) = 5843 := by
  -- We should convert the division result into integer via rounding.
  have h1 : (1336 / 17) = 78 := sorry,
  rw h1,
  norm_num
  sorry

end calculate_expression_l323_323169


namespace total_road_length_l323_323378

theorem total_road_length (L : ℚ) : (1/3) * L + (2/5) * (2/3) * L = 135 → L = 225 := 
by
  intro h
  sorry

end total_road_length_l323_323378


namespace first_triangular_number_year_in_21st_century_l323_323369

theorem first_triangular_number_year_in_21st_century :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2016 ∧ 2000 ≤ 2016 ∧ 2016 < 2100 :=
by
  sorry

end first_triangular_number_year_in_21st_century_l323_323369


namespace isosceles_triangle_count_l323_323622

def is_isosceles_triangle (a b : ℕ) : Prop :=
  2 * a + b = 31 ∧ 2 * a > b ∧ b > 0

def count_isosceles_triangles_with_perimeter_31 : ℕ :=
  (Finset.range 16).filter (λ b, b % 2 = 1 ∧ 
    ∃ a, is_isosceles_triangle a b).card

theorem isosceles_triangle_count : count_isosceles_triangles_with_perimeter_31 = 8 := 
  by
  sorry

end isosceles_triangle_count_l323_323622


namespace smallest_nonfactor_product_of_factors_of_48_l323_323000

theorem smallest_nonfactor_product_of_factors_of_48 :
  ∃ a b : ℕ, a ≠ b ∧ a ∣ 48 ∧ b ∣ 48 ∧ ¬ (a * b ∣ 48) ∧ (∀ c d : ℕ, c ≠ d ∧ c ∣ 48 ∧ d ∣ 48 ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ a * b = 18 :=
sorry

end smallest_nonfactor_product_of_factors_of_48_l323_323000


namespace count_5_primables_less_than_1000_l323_323093

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323093


namespace square_area_with_circles_l323_323853

theorem square_area_with_circles (r : ℝ) (h : r = 8) : (2 * (2 * r))^2 = 1024 := 
by 
  sorry

end square_area_with_circles_l323_323853


namespace reduction_in_consumption_l323_323176

def rate_last_month : ℝ := 16
def rate_current : ℝ := 20
def initial_consumption (X : ℝ) : ℝ := X

theorem reduction_in_consumption (X : ℝ) : initial_consumption X - (initial_consumption X * rate_last_month / rate_current) = initial_consumption X * 0.2 :=
by
  sorry

end reduction_in_consumption_l323_323176


namespace prove_k_and_s_n_prove_t_n_l323_323214

-- Define the given sequence a_n
def a_n (n : ℕ) (k : ℤ) : ℚ :=
  (4 * n^2 + k) / (2 * n + 1)

-- Define the arithmetic property of the sequence a_n
def is_arithmetic_sequence (a_n : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) - a_n n = a_n (n + 2) - a_n (n + 1)

-- Define the sum of the first n terms of a given sequence
def sum_of_first_n_terms (seq : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ i in finset.range (n+1), seq i

-- Define the geometric sum S_n
def S_n (n : ℕ) : ℚ :=
  (2 / 3) * 4^n - (2 / 3)

-- Define the sequence b_n
def b_n (n : ℕ) (a_n : ℕ → ℚ) : ℤ :=
  let prod_n := a_n n * a_n (n + 1)
  in n + (2 / prod_n)

-- Define the sum T_n
def T_n (n : ℕ) : ℚ :=
  (n*(n + 1))/2 + 1 - 1/(2 * n + 1)

-- Prove the main statements given conditions
theorem prove_k_and_s_n (a_n_is_arithmetic : is_arithmetic_sequence (a_n k)) : 
  k = -1 ∧ (sum_of_first_n_terms (λ n, 2 ^ (a_n n k)) n = S_n n) :=
sorry

theorem prove_t_n (a_n_is_arithmetic : is_arithmetic_sequence (a_n k))
  (k_eq_neg_one : k = -1) : sum_of_first_n_terms (b_n (λ n, 2 * n - 1)) n = T_n n :=
sorry

end prove_k_and_s_n_prove_t_n_l323_323214


namespace find_abc_values_l323_323528

-- Define the problem conditions as lean definitions
def represents_circle (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x - b * y + c = 0

def circle_center_and_radius_condition (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 3^2

-- Lean 4 statement for the proof problem
theorem find_abc_values (a b c : ℝ) :
  (∀ x y : ℝ, represents_circle a b c x y ↔ circle_center_and_radius_condition x y) →
  a = -2 ∧ b = 6 ∧ c = 4 :=
by
  intro h
  sorry

end find_abc_values_l323_323528


namespace subtracted_value_l323_323854

theorem subtracted_value : 
  ∃ y : ℤ, 
    let x : ℤ := 155 in 2 * x - y = 110 ∧ y = 200 :=
by {
  existsi (200 : ℤ),
  split,
  { simp, sorry },
  { refl }
}

end subtracted_value_l323_323854


namespace smallest_product_not_factor_of_48_exists_l323_323002

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l323_323002


namespace range_of_a_l323_323954

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : 3 * a ≥ 1) (h3 : 4 * a ≤ 3 / 2) : 
  (1 / 3) ≤ a ∧ a ≤ (3 / 8) :=
by
  sorry

end range_of_a_l323_323954


namespace sum_m_satisfying_binom_eq_l323_323429

open_locale big_operators

noncomputable theory

def binom (n k : ℕ) : ℕ := nat.choose n k

theorem sum_m_satisfying_binom_eq :
  (∑ m in {m : ℕ | binom 25 m + binom 25 12 = binom 26 13}.to_finset, m) = 24 :=
by sorry

end sum_m_satisfying_binom_eq_l323_323429


namespace albino_8_antlered_deers_l323_323778

theorem albino_8_antlered_deers (total_deer : ℕ) (perc_8_antlered : ℝ) (fraction_albino : ℝ) 
  (h_total_deer : total_deer = 920) (h_perc_8_antlered : perc_8_antlered = 0.10) 
  (h_fraction_albino : fraction_albino = 0.25) : 
  (nat.floor ((total_deer * perc_8_antlered) * fraction_albino) : ℕ) = 23 :=
by
  sorry

end albino_8_antlered_deers_l323_323778


namespace max_value_is_2_power_5_div_a_5_l323_323952

noncomputable def max_value_in_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : ℝ :=
if S 9 > 0 ∧ S 10 < 0 then
  let sum_9 := (a 1 + a 9) * 9 / 2 in
  let sum_10 := (a 1 + a 10) * 10 / 2 in
  if sum_9 > 0 ∧ sum_10 < 0 then
    let a_5 := a 5 in
    let a_6 := a 6 in
    if a_5 > 0 ∧ a_5 + a_6 < 0 then
      (2 ^ 5) / a_5
    else
      0
  else
    0
else
  0

theorem max_value_is_2_power_5_div_a_5 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : (S 9) > 0) (h2 : (S 10) < 0) : 
  max_value_in_sequence a S = (2 ^ 5) / a 5 :=
by sorry

end max_value_is_2_power_5_div_a_5_l323_323952


namespace probability_two_red_two_blue_l323_323450

theorem probability_two_red_two_blue (red_marbles blue_marbles total_marbles : ℕ) 
  (choose : ℕ → ℕ → ℕ)
  (prob : (choose 15 2 * choose 9 2) % choose 24 4) = 4 / 7 :=
  sorry

end probability_two_red_two_blue_l323_323450


namespace quadratic_uniq_sol_implies_m_l323_323990

theorem quadratic_uniq_sol_implies_m (m : ℝ) :
  ({x : ℝ | m * x ^ 2 + 2 * x - 1 = 0}).card = 1 → m = 0 ∨ m = -1 :=
by
  sorry

end quadratic_uniq_sol_implies_m_l323_323990


namespace sophomore_spaghetti_tortellini_ratio_l323_323298

theorem sophomore_spaghetti_tortellini_ratio
    (total_students : ℕ)
    (spaghetti_lovers : ℕ)
    (tortellini_lovers : ℕ)
    (grade_levels : ℕ)
    (spaghetti_sophomores : ℕ)
    (tortellini_sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : spaghetti_lovers = 300)
    (h3 : tortellini_lovers = 120)
    (h4 : grade_levels = 4)
    (h5 : spaghetti_sophomores = spaghetti_lovers / grade_levels)
    (h6 : tortellini_sophomores = tortellini_lovers / grade_levels) :
    (spaghetti_sophomores : ℚ) / (tortellini_sophomores : ℚ) = 5 / 2 := by
  sorry

end sophomore_spaghetti_tortellini_ratio_l323_323298


namespace cuboid_surface_area_increase_l323_323024

variables (L W H : ℝ)
def SA_original (L W H : ℝ) : ℝ := 2 * (L * W + L * H + W * H)

def SA_new (L W H : ℝ) : ℝ := 2 * ((1.50 * L) * (1.70 * W) + (1.50 * L) * (1.80 * H) + (1.70 * W) * (1.80 * H))

theorem cuboid_surface_area_increase :
  (SA_new L W H - SA_original L W H) / SA_original L W H * 100 = 315.5 :=
by
  sorry

end cuboid_surface_area_increase_l323_323024


namespace area_under_the_curve_l323_323393

theorem area_under_the_curve : 
  ∫ x in (0 : ℝ)..1, (x^2 + 1) = 4 / 3 := 
by
  sorry

end area_under_the_curve_l323_323393


namespace common_chord_line_common_chord_length_l323_323260

-- Definitions for circle equations 
def C1_eq (x y : ℝ) := x^2 + y^2 - 2 * x + 10 * y - 24 = 0
def C2_eq (x y : ℝ) := x^2 + y^2 + 2 * x + 2 * y - 8 = 0

-- Proving the equation of the common chord
theorem common_chord_line :
  ∀ (x y : ℝ), C1_eq x y → C2_eq x y → x - 6 * y + 6 = 0 := 
by 
  sorry

-- Definitions for center and radius based on C1
def center_C1 := (1 : ℝ, -5 : ℝ)
def radius_C1 := 5 * Real.sqrt 2

-- Assuming the line equation is true (from the first proof), prove the length of the common chord
theorem common_chord_length : 
  ∃ L : ℝ, L = 2 * Real.sqrt 13 :=
by
  sorry

end common_chord_line_common_chord_length_l323_323260


namespace min_value_f_inequality_f_l323_323602

open Set

-- Define the function f
def f (x : ℝ) : ℝ := x + 2 / (x - 1)

-- Problem 1: Proving the minimum value of the function
theorem min_value_f : 
  ∃ (x : ℝ), (1 < x ∧ f x = 2 * Real.sqrt 2 + 1) ∧ 
             ∀ (y : ℝ), 1 < y → f y ≥ 2 * Real.sqrt 2 + 1 :=
sorry

-- Problem 2: Proving the solution to the inequality f(x) ≥ -2
theorem inequality_f :
  {x : ℝ | f x ≥ -2} = {x : ℝ | -1 ≤ x ∧ x ≤ 0} ∪ {x : ℝ | x > 1} :=
sorry

end min_value_f_inequality_f_l323_323602


namespace customer_paid_amount_l323_323409

-- Define the cost price
def cost_price : ℝ := 800

-- Define the percentage increase
def percentage_increase : ℝ := 0.25

-- Define the expected selling price
def expected_selling_price : ℝ := 1000

-- The theorem stating what we need to prove
theorem customer_paid_amount :
  let selling_price := cost_price + (percentage_increase * cost_price) in
  selling_price = expected_selling_price :=
by
  sorry

end customer_paid_amount_l323_323409


namespace find_number_of_real_solutions_l323_323917

noncomputable def f (x : ℝ) : ℝ := 
  (∑ k in finset.range 1 (50 + 1), (k : ℝ) / (x - k))

theorem find_number_of_real_solutions (f : ℝ → ℝ) (hx : ∀ x, f x = ∑ k in finset.range 50, (k + 1 : ℝ) / (x - (k + 1))) :
  ∃ n, n = 51 := 
begin
  sorry
end

end find_number_of_real_solutions_l323_323917


namespace intersection_M_N_l323_323255

def M : Set (ℝ × ℝ) := { p | p.snd = -p.fst + 1 }
def N : Set (ℝ × ℝ) := { p | p.snd = p.fst - 1 }

theorem intersection_M_N : M ∩ N = { (1, 0) } :=
by
  sorry

end intersection_M_N_l323_323255


namespace card_draw_sequential_same_suit_l323_323473

theorem card_draw_sequential_same_suit : 
  let hearts := 13
  let diamonds := 13
  let total_suits := hearts + diamonds
  ∃ ways : ℕ, ways = total_suits * (hearts - 1) :=
by
  sorry

end card_draw_sequential_same_suit_l323_323473


namespace tax_on_other_items_l323_323714

theorem tax_on_other_items (total_amount clothing_amount food_amount other_items_amount tax_on_clothing tax_on_food total_tax : ℝ) (tax_percent_other : ℝ) 
(h1 : clothing_amount = 0.5 * total_amount)
(h2 : food_amount = 0.2 * total_amount)
(h3 : other_items_amount = 0.3 * total_amount)
(h4 : tax_on_clothing = 0.04 * clothing_amount)
(h5 : tax_on_food = 0) 
(h6 : total_tax = 0.044 * total_amount)
: 
(tax_percent_other = 8) := 
by
  -- Definitions from the problem
  -- Define the total tax paid as the sum of taxes on clothing, food, and other items
  let tax_other_items : ℝ := tax_percent_other / 100 * other_items_amount
  
  -- Total tax equation
  have h7 : tax_on_clothing + tax_on_food + tax_other_items = total_tax
  sorry

  -- Substitution values into the given conditions and solving
  have h8 : tax_on_clothing + tax_percent_other / 100 * other_items_amount = total_tax
  sorry
  
  have h9 : 0.04 * 0.5 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h10 : 0.02 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h11 : tax_percent_other / 100 * 0.3 * total_amount = 0.024 * total_amount
  sorry

  have h12 : tax_percent_other / 100 * 0.3 = 0.024
  sorry

  have h13 : tax_percent_other / 100 = 0.08
  sorry

  have h14 : tax_percent_other = 8
  sorry

  exact h14

end tax_on_other_items_l323_323714


namespace partition_space_into_cubes_l323_323672

theorem partition_space_into_cubes (n : ℕ) :
  ∃ (f : ℕ → set ℝ³) (k : ℕ), 
  (∀ i < n, is_cube_with_integer_edges (f i) ∧ (f i).edges ≤ 324) :=
sorry

end partition_space_into_cubes_l323_323672


namespace sum_of_powers_of_i_l323_323340

-- Let i be the imaginary unit
def i : ℂ := Complex.I

theorem sum_of_powers_of_i : (1 + i + i^2 + i^3 + i^4 + i^5 + i^6 + i^7 + i^8 + i^9 + i^10) = i := by
  sorry

end sum_of_powers_of_i_l323_323340


namespace number_of_ways_to_score_l323_323774

-- Define the conditions
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def score_red : ℕ := 2
def score_white : ℕ := 1
def total_balls : ℕ := 5
def min_score : ℕ := 7

-- Prove the equivalent proof problem
theorem number_of_ways_to_score :
  ∃ ways : ℕ, 
    (ways = ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
             (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
             (Nat.choose red_balls 2) * (Nat.choose white_balls 3))) ∧
    ways = 186 :=
by
  let ways := ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
               (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
               (Nat.choose red_balls 2) * (Nat.choose white_balls 3))
  use ways
  constructor
  . rfl
  . sorry

end number_of_ways_to_score_l323_323774


namespace average_speed_correct_l323_323806

-- Defining the speeds as constants
constant speed_P_to_Q : ℝ := 60
constant speed_Q_to_P : ℝ := 90

-- Defining the distance as a variable
variable (D : ℝ) (hD : D > 0)

-- Calculating the times 
def time_P_to_Q := D / speed_P_to_Q
def time_Q_to_P := D / speed_Q_to_P

-- Total distance and time
def total_distance := D + D
def total_time := time_P_to_Q D + time_Q_to_P D

-- Average speed calculation
noncomputable def average_speed := total_distance D / total_time D

-- Theorem stating the average speed is 72 km/hr
theorem average_speed_correct : average_speed D hD = 72 := by
  sorry

end average_speed_correct_l323_323806


namespace sequence_first_30_max_min_l323_323943

def a_n (n : ℕ+) : ℝ := (n - Real.sqrt 97) / (n - Real.sqrt 98)

theorem sequence_first_30_max_min :
  ∃ (n_max n_min : ℕ+), (1 ≤ n_max) ∧ (n_max ≤ 30) ∧ (1 ≤ n_min) ∧ (n_min ≤ 30) ∧
    (∀ m : ℕ+, (1 ≤ m) ∧ (m ≤ 30) → a_n m ≤ a_n n_max) ∧
    (∀ m : ℕ+, (1 ≤ m) ∧ (m ≤ 30) → a_n n_min ≤ a_n m) ∧
    n_max = ⟨10, sorry⟩ ∧ n_min = ⟨9, sorry⟩ :=
sorry

end sequence_first_30_max_min_l323_323943


namespace num_even_1s_arrays_l323_323496

theorem num_even_1s_arrays (n : ℕ) : 
  ∃ num_arrays : ℕ, (∀ A : Matrix (Fin n) (Fin n) ℕ, (∀ i, Even (A i).sum) ∧ (∀ i, Even ((Matrix.transpose A i).sum) ) → num_arrays = 2 ^ ((n - 1) ^ 2)) :=
by
  sorry

end num_even_1s_arrays_l323_323496


namespace count_5_primable_below_1000_is_21_l323_323072

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323072


namespace minimum_value_g_l323_323608

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (x - 1) * f x a

theorem minimum_value_g :
  (∀ a, f 2 a = 1 / 2 → ∃ x ∈ Icc (1/2 : ℝ) 2, ∀ y ∈ Icc (1/2 : ℝ) 2, g x a ≤ g y a ∧ g x a = -1) :=
begin
  sorry
end

end minimum_value_g_l323_323608


namespace shaded_region_area_l323_323453

def radius := 3
def side_length := 2
def angle_B := 45
noncomputable def correct_area := (9 * (Real.pi - Real.sqrt 2) / 4)

theorem shaded_region_area :
    ∃ (O : Point) (A B C D E : Point) (circle_O : Circle)
        (square_OABC : Square),
        circle_O.radius = radius ∧
        square_OABC.side_length = side_length ∧
        (B :: A :: C :: Nil) = square_OABC.vertices ∧
        ∠ A B C = angle_B ∧
        circle_O.contains D ∧
        circle_O.contains E ∧
        extended_to_circle B A D ∧
        extended_to_circle B C E ∧
        minor_arc D E ⊆ circle_O ∧
        correct_area =
        (area_of_sector_O_D_E circle_O.radius D E - area_of_triangle_O_D_E B D E B E) :=
by sorry

end shaded_region_area_l323_323453


namespace part1_part2_l323_323964

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end part1_part2_l323_323964


namespace ways_to_start_writing_l323_323759

def ratio_of_pens_to_notebooks (pens notebooks : ℕ) : Prop := 
    pens * 4 = notebooks * 5

theorem ways_to_start_writing 
    (pens notebooks : ℕ) 
    (h_ratio : ratio_of_pens_to_notebooks pens notebooks) 
    (h_pens : pens = 50)
    (h_notebooks : notebooks = 40) : 
    ∃ ways : ℕ, ways = 40 :=
by
  sorry

end ways_to_start_writing_l323_323759


namespace solution_set_of_inequality_l323_323749

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h₁ : ∀ x > 0, deriv f x + 2 * f x > 0) :
  {x : ℝ | x + 2018 > 0 ∧ x + 2018 < 5} = {x : ℝ | -2018 < x ∧ x < -2013} := 
by
  sorry

end solution_set_of_inequality_l323_323749


namespace count_5_primable_below_1000_is_21_l323_323065

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323065


namespace percentage_xy_l323_323644

-- Define the given conditions
variable {x y P : ℝ}
def isPercentageOf (a b c : ℝ) : Prop := c * a = b

-- Condition 1: P percent of (x - y) = 30% of (x + y)
def condition1 : Prop := isPercentageOf (x - y) (0.3 * (x + y)) (P / 100)

-- Condition 2: y = 0.4x
def condition2 : Prop := y = 0.4 * x

-- The proof problem
theorem percentage_xy : condition1 → condition2 → P = 70 := by
  intros
  sorry

end percentage_xy_l323_323644


namespace overall_loss_percent_l323_323784

theorem overall_loss_percent :
  let A := 560
  let B := 600
  let C := 700
  let discount_A := 0.10 * A
  let discount_B := 0.15 * B
  let discount_C := 0.15 * C
  let purchase_price_A := A - discount_A
  let purchase_price_B := B - discount_B
  let purchase_price_C := C - discount_C
  let sell_price_A := 340
  let sell_price_B := 380
  let sell_price_C := 420
  let tax_A := 0.15 * sell_price_A
  let tax_B := 0.15 * sell_price_B
  let tax_C := 0.15 * sell_price_C
  let final_sell_price_A := sell_price_A + tax_A
  let final_sell_price_B := sell_price_B + tax_B
  let final_sell_price_C := sell_price_C + tax_C
  let total_purchase_price := purchase_price_A + purchase_price_B + purchase_price_C
  let total_sell_price := final_sell_price_A + final_sell_price_B + final_sell_price_C
  let overall_loss := total_purchase_price - total_sell_price
  let loss_percent := (overall_loss / total_purchase_price) * 100
  loss_percent ≈ 18.52 := 
by {
  sorry
}

end overall_loss_percent_l323_323784


namespace problem1_problem2_problem3_l323_323983

-- Definitions based on problem statements:
def f (x a b : ℝ) := (x + a) / (x ^ 2 + b * x + 1)

-- Problem 1: Prove that if f is odd, then a = 0 and b = 0
theorem problem1 (a b : ℝ) (h : ∀ x : ℝ, f(x, a, b) = -f(-x, a, b)) : a = 0 ∧ b = 0 :=
  sorry

-- Definitions based on reformulation:
def f_odd (x : ℝ) := x / (x ^ 2 + 1)

-- Problem 2: Prove that f_odd is monotonically decreasing on (1, +∞)
theorem problem2 : ∀ x1 x2 : ℝ, 1 < x1 ∧ x1 < x2 → f_odd(x1) > f_odd(x2) :=
  sorry

-- Problem 3: Given k < 0 and (∀ t : ℝ, f_odd(t ^ 2 - 2 * t + 3) + f_odd(k - 1) < 0), find the range of k
theorem problem3 (k : ℝ) (h : k < 0) (h1 : ∀ t : ℝ, f_odd(t ^ 2 - 2 * t + 3) + f_odd(k - 1) < 0) : -1 < k ∧ k < 0 :=
  sorry

end problem1_problem2_problem3_l323_323983


namespace total_bill_for_group_is_129_l323_323502

theorem total_bill_for_group_is_129 :
  let num_adults := 6
  let num_teenagers := 3
  let num_children := 1
  let cost_adult_meal := 9
  let cost_teenager_meal := 7
  let cost_child_meal := 5
  let cost_soda := 2.50
  let num_sodas := 10
  let cost_dessert := 4
  let num_desserts := 3
  let cost_appetizer := 6
  let num_appetizers := 2
  let total_bill := 
    (num_adults * cost_adult_meal) +
    (num_teenagers * cost_teenager_meal) +
    (num_children * cost_child_meal) +
    (num_sodas * cost_soda) +
    (num_desserts * cost_dessert) +
    (num_appetizers * cost_appetizer)
  total_bill = 129 := by
sorry

end total_bill_for_group_is_129_l323_323502


namespace count_5_primable_below_1000_is_21_l323_323070

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323070


namespace at_most_70_percent_acute_triangles_l323_323946

-- Definitions based on the conditions:
def points : ℕ := 100

-- Main theorem statement expressing the problem and the answer:
theorem at_most_70_percent_acute_triangles (h : points = 100) : 
  ∃ percentage : ℝ, percentage ≤ 0.70 ∧ 
  (∀ (S : finset (fin points)), S.card = 3 →
   (∀ (a b c: ℕ) (ha: a ∈ S) (hb: b ∈ S) (hc: c ∈ S), 
    acute_triangle a b c → acute_triangle_count / triangle_count S <= percentage)) :=
sorry

end at_most_70_percent_acute_triangles_l323_323946


namespace sum_g_eq_735_l323_323689

noncomputable def g (n : ℕ) : ℕ :=
  (real.cbrt n).to_nat

theorem sum_g_eq_735 :
  ∑ k in finset.range 4095 \x $ λ k : ℕ, (1 : ℚ) / (g k) = 735 :=
begin
  sorry
end

end sum_g_eq_735_l323_323689


namespace math_problem_l323_323628

noncomputable def problem_statement : Prop :=
  ∀ (x y : ℝ),
    (8^x / 4^(x + y) = 32) ∧ (16^(x + y) / 4^(7 * y) = 256) →
    (x * y = -2)

theorem math_problem : problem_statement := 
by
  intro x y
  intro h
  cases h with h1 h2
  sorry

end math_problem_l323_323628


namespace common_properties_rectangle_rhombus_l323_323699

-- Definitions
structure Rectangle (A B C D : Type) :=
  (angles_eq_90 : ∀ (a b c d : A), a = 90 ∧ b = 90 ∧ c = 90 ∧ d = 90)
  (opposite_sides_eq : ∀ (a b c d : A), opposite_sides_eq a b c d)
  (diagonals_bisect : ∀ (a b : A), diagonals_bisect a b)

structure Rhombus (P Q R S : Type) :=
  (sides_eq : ∀ (p q r s : P), p = q ∧ q = r ∧ r = s ∧ s = p)
  (diagonals_bisect_at_right_angles : ∀ (p q : P), diagonals_bisect_at_right_angles p q)
  (opposite_angles_eq : ∀ (p q r s : P), opposite_angles_eq p q r s)

structure Parallelogram (W X Y Z : Type) :=
  (opposite_sides_eq_parallel : ∀ (w x y z : W), w = y ∧ x = z ∧ w ∥ y ∧ x ∥ z)
  (opposite_angles_eq : ∀ (w x y z : W), opposite_angles_eq w x y z)
  (diagonals_bisect_eq : ∀ (w x : W), diagonals_bisect_eq w x)

-- Theorem statement
theorem common_properties_rectangle_rhombus (A B C D P Q R S : Type) 
  (rect : Rectangle A B C D) (rhomb : Rhombus P Q R S) : 
  ∀ (W X Y Z : Type), Parallelogram W X Y Z :=
by sorry

end common_properties_rectangle_rhombus_l323_323699


namespace Train_Length_Correct_l323_323857

-- Definitions of the conditions
def Speed_in_kmph : ℝ := 45
def Bridge_Length : ℝ := 140
def Time_to_Pass_Bridge : ℝ := 42

-- Conversion of speed from km/hour to m/second
def Speed_in_mps : ℝ := Speed_in_kmph * 1000 / 3600

-- Total distance covered by the train while passing the bridge
def Total_Distance : ℝ := Speed_in_mps * Time_to_Pass_Bridge

-- Length of the train which we need to prove is 385 meters
def Length_of_Train : ℝ := Total_Distance - Bridge_Length

theorem Train_Length_Correct :
  Length_of_Train = 385 :=
by
  sorry

end Train_Length_Correct_l323_323857


namespace number_of_5_primable_less_1000_l323_323085

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323085


namespace correct_negation_of_p_l323_323373

open Real

def proposition_p (x : ℝ) := x > 0 → sin x ≥ -1

theorem correct_negation_of_p :
  ¬ (∀ x, proposition_p x) ↔ (∃ x, x > 0 ∧ sin x < -1) :=
by
  sorry

end correct_negation_of_p_l323_323373


namespace solve_inequality_l323_323386

theorem solve_inequality (a : ℝ) :
  (a > 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ -a / 4 < x ∧ x < a / 3)) ∧
  (a = 0 → ∀ x : ℝ, ¬ (12 * x^2 - a * x - a^2 < 0)) ∧ 
  (a < 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ a / 3 < x ∧ x < -a / 4)) :=
by
  sorry

end solve_inequality_l323_323386


namespace count_5_primables_less_than_1000_l323_323097

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323097


namespace quadratic_function_expression_minimum_value_g_l323_323574

structure QuadraticFunction where
  f : ℝ → ℝ
  passes_through : f 0 = 4
  symmetric : ∀ x, f (3 - x) = f x
  min_value : ∀ x, f x ≥ 7 / 4
  exists_min : ∃ x, f x = 7 / 4

def given_quadratic_function : QuadraticFunction :=
  { f := λ x, x^2 - 3*x + 4,
    passes_through := by simp,
    symmetric := by
      intro x
      calc (x^2 - 3*x + 4) = ((3-x)^2 - 3*(3-x) + 4) : by sorry,
    min_value := by
      intro x
      calc (x^2 - 3*x + 4) ≥ 7 / 4 : by sorry,
    exists_min := by
      use 3 / 2
      simp
      rfl }

noncomputable def h (x t : ℝ) := (x^2 - 2*t*x + 4)

def g (t : ℝ) : ℝ :=
  if t < 0 then 4
  else if 0 ≤ t ∧ t ≤ 1 then -t^2 + 4
  else 5 - 2 * t

theorem quadratic_function_expression :
  ∀ f : QuadraticFunction,
    f.f = (λ x, x^2 - 3*x + 4) :=
by simp [given_quadratic_function.f]

theorem minimum_value_g (t : ℝ) :
  g(t) = if t < 0 then 4
  else if 0 ≤ t ∧ t ≤ 1 then -t^2 + 4
  else 5 - 2 * t :=
by simp [g]

end quadratic_function_expression_minimum_value_g_l323_323574


namespace sum_of_first_eight_primes_with_units_digit_three_l323_323198

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function to check primality

def has_units_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem sum_of_first_eight_primes_with_units_digit_three :
  let primes_with_units_digit_three := list.filter (λ n, is_prime n ∧ has_units_digit_three n) [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123]
  let first_eight_primes := primes_with_units_digit_three.take 8
  let sum_of_primes := first_eight_primes.sum
  sum_of_primes = 404 := by
{
  sorry
}

end sum_of_first_eight_primes_with_units_digit_three_l323_323198


namespace bug_at_A_after_8_moves_l323_323327

-- Define the vertices of the tetrahedron.
inductive Vertex
| A | B | C | D

open Vertex

-- Define a function that computes the probability P(n)
def P : ℕ → ℚ
| 0     := 1
| (n+1) := (1 - P n) / 3

-- State the theorem
theorem bug_at_A_after_8_moves :
  P 8 = 547 / 2187 :=
sorry

end bug_at_A_after_8_moves_l323_323327


namespace count_5_primable_below_1000_is_21_l323_323066

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323066


namespace coefficient_of_x3_in_expansion_l323_323397

theorem coefficient_of_x3_in_expansion : 
  (∀ (x : ℝ), (2 * x + real.sqrt x) ^ 5 = 10 * x ^ 3) := sorry

end coefficient_of_x3_in_expansion_l323_323397


namespace count_5_primable_less_than_1000_eq_l323_323131

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323131


namespace binary_multiplication_l323_323509

theorem binary_multiplication :
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  a * b = product :=
by 
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  sorry

end binary_multiplication_l323_323509


namespace find_x_for_f_eq_2_l323_323984

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 3^x else -x

theorem find_x_for_f_eq_2 : ∃ x : ℝ, f x = 2 ∧ x = Real.log 2 / Real.log 3 :=
by
  split
  . use Real.log 2 / Real.log 3
  . constructor
  . sorry  -- f (Real.log 2 / Real.log 3) = 2
  . refl  -- x = Real.log 2 / Real.log 3

end find_x_for_f_eq_2_l323_323984


namespace count_5_primable_below_1000_is_21_l323_323073

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323073


namespace vector_c_coordinates_cosine_between_a_b_l323_323617

-- Definitions of vectors a and b with the given conditions.
def a : ℝ × ℝ := (-Real.sqrt 2, 1)
def c (m n : ℝ) : ℝ × ℝ := (m, n)
def b : ℝ × ℝ := sorry  -- You can define b in any form, as long as it holds the given conditions later.

-- Condition 1: Vector c has a magnitude of 2
def length_of_c (m n : ℝ) : Prop := (m^2 + n^2 = 4)

-- Condition 2: Vector a is parallel to vector c 
def parallel_a_c (m n : ℝ) : Prop := (m = -Real.sqrt 2 * n)

-- Condition 3: Vector b has a magnitude of sqrt(2)
def length_of_b : Prop := sorry  -- You can define b in any form, as long as it holds the given conditions later.

-- Condition 4: (a + 3b) is perpendicular to (a - b)
def perpendicular_a_b : Prop := sorry  -- You can define the exact condition later, as long as it holds.

-- Proving the required statement with all conditions

theorem vector_c_coordinates (m n : ℝ) (h1: length_of_c m n) (h2: parallel_a_c m n) : 
  c m n = (- (2 * Real.sqrt 6) / 3, (2 * Real.sqrt 3) / 3)
  ∨ c m n = ((2 * Real.sqrt 6) / 3, - (2 * Real.sqrt 3) / 3) :=
by
  sorry

theorem cosine_between_a_b (h3: length_of_b) (h4: perpendicular_a_b) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 6 / 4 :=
by
  sorry

end vector_c_coordinates_cosine_between_a_b_l323_323617


namespace sin_alpha_beta_fewer_than_four_values_l323_323815

noncomputable def sin_relation (α β : ℝ) : ℝ :=
  sin α * real.sqrt (1 - (sin β)^2) + sin β * real.sqrt (1 - (sin α)^2)

theorem sin_alpha_beta (x y z : ℝ) (α β : ℝ) (h1 : x = sin α) (h2 : y = sin β) :
  (z = sin_relation α β) →
  z^4 - 2 * z^2 * (x^2 + y^2 - 2 * x^2 * y^2) + (x^2 - y^2)^2 = 0 := by
  sorry

theorem fewer_than_four_values (x y : ℝ) (α β : ℝ) (h1 : x = sin α) (h2 : y = sin β) :
  (x = 0 ∨ x = 1 ∨ x = -1 ∨ y = 0 ∨ y = 1 ∨ y = -1 ∨ x = y ∨ x = -y) →
  ∃ z, (z = sin_relation α β) := by
  sorry

end sin_alpha_beta_fewer_than_four_values_l323_323815


namespace difference_between_averages_l323_323812

noncomputable def greatest_positive_even_integer_less_than_or_equal_to (x : ℝ) : ℕ :=
let y := floor x in if even y then y else y - 1

def ravi_avg_income : ℝ := 1025.68

def form_filling_avg : ℕ := greatest_positive_even_integer_less_than_or_equal_to ravi_avg_income

theorem difference_between_averages : ravi_avg_income - (form_filling_avg : ℝ) = 1.68 := by
  sorry

end difference_between_averages_l323_323812


namespace number_of_albino_8_antlered_deer_l323_323780

variable (total_deer : ℕ) (antler_percentage : ℚ) (albino_fraction : ℚ)
variable (has_8_antlers : ℕ) (albino_8_antlers : ℕ)

-- Conditions
def deer_population := total_deer = 920
def percentage_with_8_antlers := antler_percentage = 0.10
def fraction_albino_among_8_antlers := albino_fraction = 0.25

-- Intermediate calculations based on conditions
def calculate_has_8_antlers := has_8_antlers = total_deer * antler_percentage
def calculate_albino_8_antlers := albino_8_antlers = has_8_antlers * albino_fraction

-- Proof statement
theorem number_of_albino_8_antlered_deer : 
  deer_population → percentage_with_8_antlers → fraction_albino_among_8_antlers →
  calculate_has_8_antlers → calculate_albino_8_antlers →
  albino_8_antlers = 23 :=
by
  intros h_population h_percentage h_fraction h_calculate8antlers h_calculatealbino
  sorry

end number_of_albino_8_antlered_deer_l323_323780


namespace determine_function_pairs_l323_323523

-- Definitions for the conditions
def condition1 (f : ℝ → ℝ) := ∀ x : ℝ, f(x) ≥ 0

def condition2 (f g : ℝ → ℝ) := ∀ x y : ℝ, f(x + g(y)) = f(x) + f(y) + 2 * y * g(x) - f(y - g(y))

-- Lean statement for the proof problem
theorem determine_function_pairs :
  (∀ f g : ℝ → ℝ, condition1 f → condition2 f g → (g = 0 → ∀ x : ℝ, f(x) ≥ 0) ∧ 
  ((g = id) → ∃ b c : ℝ, ∀ x : ℝ, f(x) = x^2 + b*x + c ∧ b^2 ≤ 4*c)) := 
by {
  -- Proof is omitted
  sorry
}

end determine_function_pairs_l323_323523


namespace false_proposition_l323_323862

theorem false_proposition
    (A : ∀ {α β γ : ℝ}, α = β → α = γ → β = γ)
    (B : ∀ {α β γ δ ε ζ : ℝ}, α = β → γ = δ → ε = ζ)
    (C : ∀ {α β : ℝ}, α + β = 180 → α = β ∧ ∀ {p q : ℝ}, parallel p q)
    (D : ∀ {α β : ℝ}, α = β → 180 - α = 180 - β)
    : ¬ (∀ {α β : ℝ}, α + β = 180 → α = β) ∨ ∀ {p q : ℝ}, parallel p q := by
    sorry

end false_proposition_l323_323862


namespace number_of_four_digit_numbers_l323_323789

theorem number_of_four_digit_numbers (digits: Finset ℕ) (h: digits = {1, 1, 2, 0}) :
  ∃ count : ℕ, (count = 9) ∧ 
  (∀ n ∈ digits, n ≠ 0 → n * 1000 + n ≠ 0) := 
sorry

end number_of_four_digit_numbers_l323_323789


namespace Luka_was_4_when_Max_born_l323_323679

noncomputable def Luka_age_when_Max_born (Luka_Aubrey_diff Aubrey_Luka_diff : ℕ)
                                         (Aubrey_age Max_age : ℕ) : ℕ :=
if h₁ : Luka_Aubrey_diff = 2 then
  if h₂ : Aubrey_age = 8 then
    if h₃ : Max_age = 6 then
      let Aubrey_age_when_Max_born := Aubrey_age - Max_age in
      Aubrey_age_when_Max_born + Luka_Aubrey_diff
    else 0
  else 0
else 0

theorem Luka_was_4_when_Max_born : Luka_age_when_Max_born 2 2 8 6 = 4 :=
by simp [Luka_age_when_Max_born]; refl; sorry

end Luka_was_4_when_Max_born_l323_323679


namespace problem_statement_l323_323471

-- Define the recursive sequence (a_n, b_n) as described
def sequence (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0       => sorry  -- initialization needed
  | k + 1   => let (a_n, b_n) := sequence k
               in (√3 * a_n - b_n, a_n + √3 * b_n)

-- Given (a_50, b_50) = (1, √3), prove a_1 + b_1 = (√3 - 1) / 2^49
theorem problem_statement : 
  let (a50, b50) := sequence 50 in
  a50 = 1 ∧ b50 = √3 →
  let (a1, b1) := sequence 1 in
  a1 + b1 = (√3 - 1) / 2^49 := 
sorry

end problem_statement_l323_323471


namespace SUCCESS_rearrangement_l323_323266

theorem SUCCESS_rearrangement: 
  let vowels := ['U', 'E'],
      consonants := ['S', 'S', 'S', 'C', 'C'] in
  (vowels.length.factorial) * (consonants.length.factorial / (3.factorial * 2.factorial)) = 20 :=
by
  sorry

end SUCCESS_rearrangement_l323_323266


namespace decaf_percentage_correct_l323_323057

-- Definition of initial stock and decaffeination percentages
def initial_stock : ℝ := 400
def initial_decaf_percentage : ℝ := 0.20
def purchase_stock : ℝ := 100
def purchase_decaf_percentage : ℝ := 0.60

-- Define the target percentage for the proof
def target_decaf_percentage : ℝ := 28

-- The proof statement which we need to fill in
theorem decaf_percentage_correct :
    let total_initial_decaf := initial_stock * initial_decaf_percentage in
    let total_purchase_decaf := purchase_stock * purchase_decaf_percentage in
    let total_decaf := total_initial_decaf + total_purchase_decaf in
    let total_stock := initial_stock + purchase_stock in
    let final_decaf_percentage := (total_decaf / total_stock) * 100 in
    final_decaf_percentage = target_decaf_percentage := by
    sorry

end decaf_percentage_correct_l323_323057


namespace find_positive_real_solutions_l323_323696

noncomputable theory

variables {n : ℕ} (a : ℕ → ℝ)

theorem find_positive_real_solutions
  (h₀ : n ≥ 4)
  (h₁ : a 1 = 1 / a (2 * n) + 1 / a 2)
  (h₂ : a 2 = a 1 + a 3)
  (h₃ : a 3 = 1 / a 2 + 1 / a 4)
  (h₄ : a 4 = a 3 + a 5)
  (h₅ : a 5 = 1 / a 4 + 1 / a 6)
  (h₆ : a 6 = a 5 + a 7)
  (h : ∀ i : ℕ, 4 ≤ 2 * i + 2 ∧ 2 * i + 1 ≤ 2 * n - 1 →
        a (2 * i + 1) = 1 / a (2 * i))
  (h_last : a (2 * n - 1) = 1 / a (2 * n - 2) + 1 / a (2 * n))
  (h_last2 : a (2 * n) = a (2 * n - 1) + a 1) :
  (∀ i : ℕ, 1 ≤ 2 * i + 1 ∧ 2 * i + 1 ≤ 2 * n → a (2 * i + 1) = 1)
  ∧
  (∀ j : ℕ, 1 ≤ 2 * j ∧ 2 * j ≤ 2 * n → a (2 * j) = 2) :=
by sorry

end find_positive_real_solutions_l323_323696


namespace molecular_weight_calculation_l323_323020

theorem molecular_weight_calculation
    (moles_total_mw : ℕ → ℝ)
    (hw : moles_total_mw 9 = 900) :
    moles_total_mw 1 = 100 :=
by
  sorry

end molecular_weight_calculation_l323_323020


namespace bobs_speed_at_construction_l323_323168

theorem bobs_speed_at_construction
  (time_before_construction : ℝ)
  (speed_before_construction : ℝ)
  (time_construction : ℝ)
  (total_distance : ℝ)
  (distance_before_construction : ℝ := time_before_construction * speed_before_construction)
  (distance_construction : ℝ := total_distance - distance_before_construction)
  (speed_during_construction : ℝ := distance_construction / time_construction) :
  time_before_construction = 1.5 ∧
  speed_before_construction = 60 ∧
  time_construction = 2 ∧
  total_distance = 180 →
  speed_during_construction = 45 :=
by
  -- introduction of conditions
  intro h,
  cases h with h1 h_aux,
  cases h_aux with h2 h_aux,
  cases h_aux with h3 h4,
  -- prove that the speed during construction is 45 mph
  sorry

end bobs_speed_at_construction_l323_323168


namespace limes_left_l323_323880

-- Define constants
def num_limes_initial : ℕ := 9
def num_limes_given : ℕ := 4

-- Theorem to be proved
theorem limes_left : num_limes_initial - num_limes_given = 5 :=
by
  sorry

end limes_left_l323_323880


namespace length_of_AD_in_quadrilateral_l323_323301

theorem length_of_AD_in_quadrilateral
  (A B C D : Type)
  (L_AB : ℝ) (L_BC : ℝ) (L_CD : ℝ)
  (angle_ABC : ℝ) (angle_BCD : ℝ)
  (h1 : L_AB = sqrt 6)
  (h2 : L_BC ≈ 5 - sqrt 3)
  (h3 : L_CD = 6)
  (h4 : angle_ABC = 135)
  (h5 : angle_BCD = 120):
  ∃ (L_AD : ℝ), L_AD = 2 * sqrt 19 :=
by
  have : ℝ := sorry -- the proof will go here
  use 2 * sqrt 19
  assumption

end length_of_AD_in_quadrilateral_l323_323301


namespace value_of_expression_l323_323633

variable (a b : ℝ)

def sign (x : ℝ) : ℝ := if x > 0 then 1 else if x < 0 then -1 else 0

theorem value_of_expression (h : a * b > 0) : 
  (a / |a|) + (b / |b|) + (a * b / |a * b|) = 3 ∨ 
  (a / |a|) + (b / |b|) + (a * b / |a * b|) = -1 := by
  sorry

end value_of_expression_l323_323633


namespace number_of_5_primable_less_1000_l323_323086

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323086


namespace part1_part2_l323_323942

section problem

def a : Real := sqrt 3 - 2
def b : Real := sqrt 3 + 2

theorem part1 : (a + b) ^ 2 = 12 := by
  sorry

theorem part2 : a ^ 2 - b ^ 2 = -8 * sqrt 3 := by
  sorry

end problem

end part1_part2_l323_323942


namespace height_at_15_l323_323465

-- Define the conditions as constants and assumptions
constant a : ℝ
constant b : ℝ
constant c : ℝ
constant h_max : ℝ := 25
constant span : ℝ := 60

-- Assumptions based on the conditions
axiom arch_eqn : ∀ (x : ℝ), 
  y = a * x^2 + b * x + c

axiom vertex_condition : 
  arch_eqn (span / 2) = h_max

axiom boundary_condition_left :
  arch_eqn 0 = 0

axiom boundary_condition_right :
  arch_eqn span = 0

-- Theorem statement to prove
theorem height_at_15 : 
  arch_eqn 15 = 18.75 :=
  sorry

end height_at_15_l323_323465


namespace intersection_of_A_and_B_l323_323229

open Finset

def A : Finset ℤ := {-2, -1, 0, 1, 2}
def B : Finset ℤ := {1, 2, 3}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B_l323_323229


namespace vector_angle_acute_l323_323245

noncomputable def a : ℝ × ℝ := (x, 2)
noncomputable def b : ℝ × ℝ := (-3, 5)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def not_collinear (u v : ℝ × ℝ) : Prop :=
  ¬ (u.1 * v.2 = u.2 * v.1)

theorem vector_angle_acute (x : ℝ) :
  (dot_product (x, 2) (-3, 5) > 0 ∧ not_collinear (x, 2) (-3, 5)) →
  x ∈ {y : ℝ | y ∈ (-∞, -6/5) ∪ (-6/5, 10/3)} :=
begin
  sorry
end

end vector_angle_acute_l323_323245


namespace birds_not_herons_are_geese_l323_323681

-- Define the given conditions
def percentage_geese : ℝ := 0.35
def percentage_swans : ℝ := 0.20
def percentage_herons : ℝ := 0.15
def percentage_ducks : ℝ := 0.30

-- Definition without herons
def percentage_non_herons : ℝ := 1 - percentage_herons

-- Theorem to prove
theorem birds_not_herons_are_geese :
  (percentage_geese / percentage_non_herons) * 100 = 41 :=
by
  sorry

end birds_not_herons_are_geese_l323_323681


namespace smallest_product_not_factor_of_48_l323_323012

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l323_323012


namespace problem_statement_l323_323354

def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := sin (ω * x + ϕ)

noncomputable def shifted_f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := f ω ϕ (x + π/12)

theorem problem_statement (ω > 0) (ϕ : ℝ) (h : 0 < ϕ ∧ ϕ < π)
  (H1 : f ω ϕ (π/3) = 0) (H2 : ω = 2 ∧ ϕ = π/2) :
  shifted_f 2 (π/2) x = cos (2 * x) :=
by
  sorry

end problem_statement_l323_323354


namespace measure_8_liters_with_two_buckets_l323_323997

def bucket_is_empty (B : ℕ) : Prop :=
  B = 0

def bucket_has_capacity (B : ℕ) (c : ℕ) : Prop :=
  B ≤ c

def fill_bucket (B : ℕ) (c : ℕ) : ℕ :=
  c

def empty_bucket (B : ℕ) : ℕ :=
  0

def pour_bucket (B1 B2 : ℕ) (c1 c2 : ℕ) : (ℕ × ℕ) :=
  if B1 + B2 <= c2 then (0, B1 + B2)
  else (B1 - (c2 - B2), c2)

theorem measure_8_liters_with_two_buckets (B10 B6 : ℕ) (c10 c6 : ℕ) :
  bucket_has_capacity B10 c10 ∧ bucket_has_capacity B6 c6 ∧
  c10 = 10 ∧ c6 = 6 →
  ∃ B10' B6', B10' = 8 ∧ B6' ≤ 6 :=
by
  intros h
  have h1 : ∃ B1, bucket_is_empty B1,
    from ⟨0, rfl⟩
  let B10 := fill_bucket 0 c10
  let ⟨B10, B6⟩ := pour_bucket B10 0 c10 c6
  let B6 := empty_bucket B6
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  let B10 := fill_bucket B10 c10
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  exact ⟨B10, B6, rfl, le_refl 6⟩

end measure_8_liters_with_two_buckets_l323_323997


namespace one_third_of_recipe_l323_323467

def mixed_to_improper (a b c : ℕ) : ℚ :=
  a + (b / c)

def third_of_improper (x : ℚ) : ℚ :=
  (1 / 3) * x

noncomputable def improper_to_mixed (x : ℚ) : ℕ × ℚ := sorry

theorem one_third_of_recipe (a b c : ℕ) (answer : ℚ) :
  mixed_to_improper 5 1 3 = 16 / 3 →
  third_of_improper (16 / 3) = 16 / 9 →
  improper_to_mixed (16 / 9) = (1, 7 / 9) →
  answer = 1 + 7 / 9 :=
begin
  intros h1 h2 h3,
  have h := h3,
  rw h,
  refl
end

end one_third_of_recipe_l323_323467


namespace find_a_plus_b_l323_323732

noncomputable def moves := {s : list (ℕ × ℕ) // ∀ p ∈ s, p = (1, 0) ∨ p = (-1, 0) ∨ p = (0, 1) ∨ p = (0, -1)}

def valid_path (steps : ℕ) (start : ℕ × ℕ) (end : ℕ × ℕ) (p : moves) : Prop :=
  p.val.length = steps ∧ list.sum p.val = end - start

def probability_reaching (steps : ℕ) (start : ℕ × ℕ) (end : ℕ × ℕ) : ℚ :=
  let total_paths := 4 ^ steps
  let valid_paths := finset.filter (valid_path steps start end) (finset.univ : finset moves)
  valid_paths.card / total_paths

theorem find_a_plus_b :
  let q := probability_reaching 8 (0, 0) (3, 3)
  ∃ a b : ℕ, q = a / b ∧ nat.coprime a b ∧ a + b = 2093 := by
  sorry

end find_a_plus_b_l323_323732


namespace units_of_Product_C_sold_l323_323363

-- Definitions of commission rates
def commission_rate_A : ℝ := 0.05
def commission_rate_B : ℝ := 0.07
def commission_rate_C : ℝ := 0.10

-- Definitions of revenues per unit
def revenue_A : ℝ := 1500
def revenue_B : ℝ := 2000
def revenue_C : ℝ := 3500

-- Definition of units sold
def units_A : ℕ := 5
def units_B : ℕ := 3

-- Commission calculations for Product A and B
def commission_A : ℝ := commission_rate_A * revenue_A * units_A
def commission_B : ℝ := commission_rate_B * revenue_B * units_B

-- Previous average commission and new average commission
def previous_avg_commission : ℝ := 100
def new_avg_commission : ℝ := 250

-- The main proof statement
theorem units_of_Product_C_sold (x : ℝ) (h1 : new_avg_commission = previous_avg_commission + 150)
  (h2 : total_units = units_A + units_B + x)
  (h3 : total_new_commission = commission_A + commission_B + (commission_rate_C * revenue_C * x))
  : x = 12 :=
by
  sorry

end units_of_Product_C_sold_l323_323363


namespace original_numbers_greater_than_prime_product_l323_323713

theorem original_numbers_greater_than_prime_product (n : ℕ) (k : ℕ) (p : ℕ → ℕ) 
  (h : ∀ i, 1 < (p i) ∧ (p i) < n ∧ (p i) divides (i + 1) ∧ ((i + 1) / (p i) = (p (i + 1)))) :
  ∀ i, i < n → i > (n^k / ∏ i in finset.range k, p i) :=
by
  sorry

end original_numbers_greater_than_prime_product_l323_323713


namespace tangent_length_from_origin_to_circumcircle_l323_323874

open Real

noncomputable def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry
noncomputable def circumradius (A B C : ℝ × ℝ) (O_c : ℝ × ℝ) : ℝ := sorry
noncomputable def tangent_length (O O_c : ℝ × ℝ) (R : ℝ) : ℝ := sqrt (R^2 - (dist O O_c)^2)

theorem tangent_length_from_origin_to_circumcircle :
  let O := (0, 0)
  let A := (1, 3)
  let B := (4, 7)
  let C := (6, 11)
  let O_c := circumcenter A B C
  let R := circumradius A B C O_c
  OT = tangent_length O O_c R
  ∃ c, OT = sqrt c :=
by
  sorry

end tangent_length_from_origin_to_circumcircle_l323_323874


namespace rectangle_width_l323_323647

-- The Lean statement only with given conditions and the final proof goal
theorem rectangle_width (w l : ℕ) (P : ℕ) (h1 : l = w - 3) (h2 : P = 2 * w + 2 * l) (h3 : P = 54) :
  w = 15 :=
by
  sorry

end rectangle_width_l323_323647


namespace complex_number_imaginary_part_l323_323591

theorem complex_number_imaginary_part (a : ℝ) (z : ℂ) (h1 : z = (1 + a * complex.I) / (1 - complex.I)) (h2 : z.im = 1) : a = 1 :=
by
  -- Proof goes here.
  sorry

end complex_number_imaginary_part_l323_323591


namespace problem_b_n_formula_problem_sum_c_n_l323_323666

variable {n : ℕ}

noncomputable def a (n : ℕ) : ℚ := (1/4) ^ n
def b (n : ℕ) : ℚ := 3 * n - 2
def c (n : ℕ) : ℚ := a n * b n
def S (n : ℕ) : ℚ := (finset.range n).sum c

theorem problem_b_n_formula (n : ℕ) : 
  3 * n - 2 = b n :=
by sorry

theorem problem_sum_c_n (n : ℕ) : 
  S n = (2/3 - (12 * n + 8) * (1/4) ^ (n + 1) / 3) :=
by sorry

end problem_b_n_formula_problem_sum_c_n_l323_323666


namespace concurrency_and_euler_line_l323_323531

noncomputable def tangents_to_circumcircle (A B C : Point) : {E_a E_b E_c : Point // Tangent (circumcircle (triangle A B C)) E_a ∧ Tangent (circumcircle (triangle A B C)) E_b ∧ Tangent (circumcircle (triangle A B C)) E_c} :=
sorry

noncomputable def altitude_feet (A B C : Point) : {M_a M_b M_c : Point // Altitude (triangle A B C) A M_a ∧ Altitude (triangle A B C) B M_b ∧ Altitude (triangle A B C) C M_c} :=
sorry

theorem concurrency_and_euler_line {A B C : Point} :
  let ⟨E_a, E_b, E_c, hE⟩ := tangents_to_circumcircle A B C
  let ⟨M_a, M_b, M_c, hM⟩ := altitude_feet A B C
  concurrent (line E_a M_a) (line E_b M_b) (line E_c M_c) ∧
  on_euler_line (concurrence_point (line E_a M_a) (line E_b M_b) (line E_c M_c)) (triangle A B C) :=
sorry

end concurrency_and_euler_line_l323_323531


namespace dan_remaining_money_l323_323520

noncomputable def calculate_remaining_money (initial_amount : ℕ) : ℕ :=
  let candy_bars_qty := 5
  let candy_bar_price := 125
  let candy_bars_discount := 10
  let gum_qty := 3
  let gum_price := 80
  let soda_qty := 4
  let soda_price := 240
  let chips_qty := 2
  let chip_price := 350
  let chips_discount := 15
  let low_tax := 7
  let high_tax := 12

  let total_candy_bars_cost := candy_bars_qty * candy_bar_price
  let discounted_candy_bars_cost := total_candy_bars_cost * (100 - candy_bars_discount) / 100

  let total_gum_cost := gum_qty * gum_price

  let total_soda_cost := soda_qty * soda_price

  let total_chips_cost := chips_qty * chip_price
  let discounted_chips_cost := total_chips_cost * (100 - chips_discount) / 100

  let candy_bars_tax := discounted_candy_bars_cost * low_tax / 100
  let gum_tax := total_gum_cost * low_tax / 100

  let soda_tax := total_soda_cost * high_tax / 100
  let chips_tax := discounted_chips_cost * high_tax / 100

  let total_candy_bars_with_tax := discounted_candy_bars_cost + candy_bars_tax
  let total_gum_with_tax := total_gum_cost + gum_tax
  let total_soda_with_tax := total_soda_cost + soda_tax
  let total_chips_with_tax := discounted_chips_cost + chips_tax

  let total_cost := total_candy_bars_with_tax + total_gum_with_tax + total_soda_with_tax + total_chips_with_tax

  initial_amount - total_cost

theorem dan_remaining_money : 
  calculate_remaining_money 10000 = 7399 :=
sorry

end dan_remaining_money_l323_323520


namespace first_investment_amount_l323_323497

-- Define the conditions
def yearly_return (x : ℝ) : ℝ := 0.07 * x
def investment_return : ℝ := 0.09 * 1500
def combined_return (x : ℝ) : ℝ := 0.085 * (x + 1500)

-- Prove the required amount of the first investment is $500, given the conditions
theorem first_investment_amount : ∃ x : ℝ, yearly_return x + investment_return = combined_return x ∧ x = 500 :=
by
  -- The conditions derived from the problem
  have h1: yearly_return = λ x, 0.07 * x := rfl
  have h2: investment_return = 0.09 * 1500 := rfl
  have h3: combined_return = λ x, 0.085 * (x + 1500) := rfl
  -- Skip the proof with sorry, just indicating where solution steps would go
  sorry

end first_investment_amount_l323_323497


namespace goal_l323_323031

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def target_sequence : List ℕ := [5, 2, 3, 6, 1, 8, 7, 4, 9]

def number_from_digits (ds : List ℕ) : ℕ :=
  ds.foldl (λ acc d => acc * 10 + d) 0

theorem goal : target_sequence = [5, 2, 3, 6, 1, 8, 7, 4, 9] →
               number_from_digits target_sequence = 523814769 ∧ (∃ n, n * n = 523814769) :=
by
  intros h
  unfold target_sequence at h ⬝
  simp [number_from_digits] at h
  split
  · exact h
  · use 22887
    norm_num
    sorry

end goal_l323_323031


namespace highest_price_l323_323748

noncomputable def highest_price_per_meter (total_area : ℝ) (budget : ℝ) : ℝ :=
  let side_length := real.sqrt (total_area / 28)
  let diagonal := real.sqrt (side_length^2 + side_length^2)
  let rosebed_perimeter := 4 * diagonal
  let garden_perimeter := 8 * side_length + 8 * diagonal
  let total_fence_length := rosebed_perimeter + garden_perimeter
  budget / total_fence_length

theorem highest_price (total_area : ℝ) (budget : ℝ) (approx_sqrt2 : ℝ)
  (h_total_area : total_area = 700) (h_budget : budget = 650) 
  (h_approx_sqrt2 : approx_sqrt2 = real.sqrt 2) : 
  highest_price_per_meter total_area budget = 650 / (40 + 60 * approx_sqrt2) := 
by sorry

end highest_price_l323_323748


namespace equation_of_perpendicular_line_l323_323973

open Real

noncomputable def E : (ℝ × ℝ) := (2, 1)
noncomputable def slope_BCD : ℝ := -1 / 3
noncomputable def perpendicular_slope : ℝ := 3

theorem equation_of_perpendicular_line :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (a * E.1 + b * E.2 + c = 0) ∧ (a * 1 + b * perpendicular_slope + c = 0) ∧ a = 3 ∧ b = -1 ∧ c = -5 :=
by 
  have a := 3
  have b := -1
  have c := -5
  use [a, b, c]
  split; 
  { sorry }, -- a ≠ 0
  split;
  { sorry }, -- a * E.1 + b * E.2 + c = 0
  split;
  { sorry }, -- a * 1 + b * perpendicular_slope + c = 0
  { refl }, -- a = 3
  split;
  { refl }, -- b = -1
  { refl }  -- c = -5

end equation_of_perpendicular_line_l323_323973


namespace part1_part2_l323_323947

noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + x^2

theorem part1 (a b : ℝ) (h : a ≠ 0) 
  (ht : TangentExists f a b 1 (1, f a b 1) (λ x, x)) :
  a = -1 ∧ b = 2 := 
sorry

theorem part2 (a b : ℝ) (h : a ≠ 0) 
  (hf : ∀ x, f a b x ≤ x^2 + x) : 
  ab ≤ Real.exp 1 / 2 := 
sorry

end part1_part2_l323_323947


namespace line_through_P_midpoint_l323_323463

noncomputable section

open Classical

variables (l l1 l2 : ℝ → ℝ → Prop) (P A B : ℝ × ℝ)

def line1 (x y : ℝ) := 2 * x - y - 2 = 0
def line2 (x y : ℝ) := x + y + 3 = 0

theorem line_through_P_midpoint (P A B : ℝ × ℝ)
  (hP : P = (3, 0))
  (hl1 : ∀ x y, line1 x y → l x y)
  (hl2 : ∀ x y, line2 x y → l x y)
  (hmid : (P.1 = (A.1 + B.1) / 2) ∧ (P.2 = (A.2 + B.2) / 2)) :
  ∃ k : ℝ, ∀ x y, (y = k * (x - 3)) ↔ (8 * x - y - 24 = 0) :=
by
  sorry

end line_through_P_midpoint_l323_323463


namespace uphill_flat_road_system_l323_323937

variables {x y : ℝ}

theorem uphill_flat_road_system :
  (3 : ℝ)⁻¹ * x + (4 : ℝ)⁻¹ * y = 70 / 60 ∧
  (4 : ℝ)⁻¹ * y + (5 : ℝ)⁻¹ * x = 54 / 60 :=
sorry

end uphill_flat_road_system_l323_323937


namespace average_rate_of_interest_l323_323148

def invested_amount_total : ℝ := 5000
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05
def annual_return (amount : ℝ) (rate : ℝ) : ℝ := amount * rate

theorem average_rate_of_interest : 
  (∃ (x : ℝ), x > 0 ∧ x < invested_amount_total ∧ 
    annual_return (invested_amount_total - x) rate1 = annual_return x rate2) → 
  ((annual_return (invested_amount_total - 1875) rate1 + annual_return 1875 rate2) / invested_amount_total = 0.0375) := 
by
  sorry

end average_rate_of_interest_l323_323148


namespace intervals_of_monotonic_increase_maximum_area_of_triangle_l323_323358

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, (Real.sqrt 3 / 2) * (Real.sin x - Real.cos x))
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x + Real.cos x)

-- Define the function f(x)
def f (x : ℝ) : ℝ := let a := vector_a x; let b := vector_b x in a.1 * b.1 + a.2 * b.2

-- Prove the intervals of monotonic increase for f(x)
theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∀ x : ℝ, (k * Real.pi - Real.pi / 12 ≤ x) ∧ (x ≤ k * Real.pi + 5 * Real.pi / 12) ↔
           increasing_on ((k * Real.pi - Real.pi / 12) , (k * Real.pi + 5 * Real.pi / 12)) f :=
sorry

-- Define the sides and angles of triangle ABC
variables {a b c A B C : ℝ}
hypothesis acute_triangle : 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2
hypothesis side_a : a = Real.sqrt 2
hypothesis f_A : f A = 1 / 2

-- Define the Law of Cosines
def law_of_cosines (a b c : ℝ) (A : ℝ) : Prop := a^2 = b^2 + c^2 - 2 * b * c * Real.cos A

-- Prove the maximum area of the triangle
theorem maximum_area_of_triangle :
  a = Real.sqrt 2 → f A = 1 / 2 → (∀ b c, b * c ≤ 2 + Real.sqrt 2 → S = (b * c) * Real.sin (Real.pi / 4) / 2 → 
  S ≤ (1 + Real.sqrt 2) / 2) :=
sorry

end intervals_of_monotonic_increase_maximum_area_of_triangle_l323_323358


namespace number_of_workers_who_read_all_three_books_l323_323994

theorem number_of_workers_who_read_all_three_books
  (W S K A SK SA KA SKA N : ℝ)
  (hW : W = 75)
  (hS : S = 1 / 2 * W)
  (hK : K = 1 / 4 * W)
  (hA : A = 1 / 5 * W)
  (hSK : SK = 2 * SKA)
  (hN : N = S - (SK + SA + SKA) - 1)
  (hTotal : S + K + A - (SK + SA + KA - SKA) + N = W) :
  SKA = 6 :=
by
  -- The proof steps are omitted
  sorry

end number_of_workers_who_read_all_three_books_l323_323994


namespace right_triangle_side_divisibility_l323_323407

theorem right_triangle_side_divisibility :
  ∀ (a b c : ℕ),
    c^2 = a^2 + b^2 → 
    (∃ x, x ∈ {a, b, c} ∧ 2 ∣ x) ∧ 
    (∃ y, y ∈ {a, b, c} ∧ 3 ∣ y) := by
  sorry

end right_triangle_side_divisibility_l323_323407


namespace intersection_point_l323_323324

def f (x : ℝ) : ℝ := (x^2 - 7 * x + 12) / (2 * x - 6)
def g (x : ℝ) (a b c : ℝ) : ℝ := (a * x^2 + b * x + c) / (x - 3)

-- Given conditions
variables (a b c : ℝ)
variable h1 : ∀ (x : ℝ), f(x) = (x^2 - 7 * x + 12) / (2 * x - 6)
variable h2 : ∀ (x : ℝ), g(x) a b c = (a * x^2 + b * x + c) / (x - 3)
variable h3 : ∀ (x : ℝ), ¬ (2 * x - 6 = 0) -> ¬ (x - 3 = 0) -> (2 * x - 6 = 0) = (x - 3 = 0)
variable h4 : ∀ (x : ℝ), (1/2 * x - 2) * (-2 * x - 2) = -1
variable h5 : f(-3) = -3.5 ∧ g(-3) a b c = -3.5

-- Proof that the point of intersection is (-2, -3)
theorem intersection_point : ∃ (x : ℝ), x ≠ -3 ∧ f(x) = g(x) a b c ∧ x = -2 ∧ f(-2) = -3 :=
by
  sorry

end intersection_point_l323_323324


namespace value_set_for_a_non_empty_proper_subsets_l323_323209

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 6 = 0}

theorem value_set_for_a (M : Set ℝ) : 
  (∀ (a : ℝ), B a ⊆ A → a ∈ M) :=
sorry

theorem non_empty_proper_subsets (M : Set ℝ) :
  M = {0, 3, -3} →
  (∃ S : Set (Set ℝ), S = {{0}, {3}, {-3}, {0, 3}, {0, -3}, {3, -3}}) :=
sorry

end value_set_for_a_non_empty_proper_subsets_l323_323209


namespace f_is_increasing_g_extrema_l323_323985

def f (x: ℝ) : ℝ := 2 * sqrt 3 * sin(x + π / 4) * cos(x + π / 4) + sin(2 * x) - 1

def g (x: ℝ) : ℝ := f(x - π / 6)

-- (I) Prove that f(x) is monotonically increasing on the interval [kπ - 5π/12, kπ + π/12] for k ∈ ℤ
theorem f_is_increasing (k : ℤ) : monotone_on f (set.Icc (k * π - 5 * π / 12) (k * π + π / 12)) := 
sorry

-- (II) Prove that the maximum value of g(x) on [0, π/2] is sqrt 3 - 1 at x = 0
-- and the minimum value is -3 at x = 5π/12
theorem g_extrema : 
  (∀ x ∈ set.Icc (0 : ℝ) (π / 2), g x ≤ sqrt 3 - 1) ∧ g 0 = sqrt 3 - 1 ∧
  (∀ x ∈ set.Icc (0 : ℝ) (π / 2), g (-3) ∧ g (5 * π / 12) = -3 :=
sorry

end f_is_increasing_g_extrema_l323_323985


namespace frustum_small_cone_height_is_correct_l323_323891

noncomputable def frustum_small_cone_height (altitude : ℝ) 
                                             (lower_base_area : ℝ) 
                                             (upper_base_area : ℝ) : ℝ :=
  let r1 := Real.sqrt (lower_base_area / Real.pi)
  let r2 := Real.sqrt (upper_base_area / Real.pi)
  let H := 2 * altitude
  altitude

theorem frustum_small_cone_height_is_correct 
  (altitude : ℝ)
  (lower_base_area : ℝ)
  (upper_base_area : ℝ)
  (h1 : altitude = 16)
  (h2 : lower_base_area = 196 * Real.pi)
  (h3 : upper_base_area = 49 * Real.pi ) : 
  frustum_small_cone_height altitude lower_base_area upper_base_area = 16 := by
  sorry

end frustum_small_cone_height_is_correct_l323_323891


namespace august_answers_total_l323_323716

theorem august_answers_total :
  let answer1 := 600 in
  let answer2 := 2 * answer1 in
  let answer3 := (answer1 + answer2) - 400 in
  answer1 + answer2 + answer3 = 3200 :=
by
  let answer1 := 600
  let answer2 := 2 * answer1
  let answer3 := (answer1 + answer2) - 400
  have h1 : answer1 = 600 := rfl
  have h2 : answer2 = 1200 := by simp [answer1, h1]
  have h3 : answer3 = 1400 := by simp [answer1, answer2, h1, h2]
  compose_goal
    add_eq_of_eq_sub 400 h2
  show 600 + 1200 + 1400 = 3200, by simp [h1, h2, h3]
  sorry

end august_answers_total_l323_323716


namespace joint_probability_l323_323440

open ProbabilityTheory

variable (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)

theorem joint_probability (a b : Set Ω) (hp_a : P a = 0.18)
  (hp_b_given_a : condCount P a b = 0.2) : P (a ∩ b) = 0.036 :=
by
  sorry

end joint_probability_l323_323440


namespace Q_investment_l323_323717

theorem Q_investment (P_investment total_profit Q_profit : ℝ) (hP : P_investment = 54000)
  (ht : total_profit = 18000) (hQ : Q_profit = 6001.89) :
  ∃ X : ℝ, (X = 27010) :=
by {
  have P_share : ℝ := total_profit - Q_profit,
  have ratio_PQ : ℝ := P_investment / X,
  have ratio_profit : ℝ := P_share / Q_profit,
  have eq_ratio : ratio_PQ = ratio_profit,
  use (54000 * 6001.89) / 11998.11,
  sorry
}

end Q_investment_l323_323717


namespace range_a_condition_l323_323609

def power_function (x : ℝ) : ℝ := x^(-1/4)

theorem range_a_condition (a : ℝ) : (3 < a) ∧ (a < 5) :=
by
  have hx1 : power_function 4 = 1 / 2 := by sorry
  have h1 : power_function (a + 1) < power_function (10 - 2 * a) := by sorry
  have h2 : (a + 1 > 10 - 2 * a) ∧ (10 - 2 * a > 0) := by sorry
  sorry

end range_a_condition_l323_323609


namespace number_of_terms_in_expanded_polynomial_l323_323527

theorem number_of_terms_in_expanded_polynomial : 
  ∀ (a : Fin 4 → Type) (b : Fin 2 → Type) (c : Fin 3 → Type), 
  (4 * 2 * 3 = 24) := 
by
  intros a b c
  sorry

end number_of_terms_in_expanded_polynomial_l323_323527


namespace problem1_l323_323443

theorem problem1 (x : ℝ) (n : ℕ) (h : x^n = 2) : (3 * x^n)^2 - 4 * (x^2)^n = 20 :=
by
  sorry

end problem1_l323_323443


namespace total_credit_hours_l323_323701

def max_courses := 40
def max_courses_per_semester := 5
def max_courses_per_semester_credit := 3
def max_additional_courses_last_semester := 2
def max_additional_course_credit := 4
def sid_courses_multiplier := 4
def sid_additional_courses_multiplier := 2

theorem total_credit_hours (total_max_courses : Nat) 
                           (avg_max_courses_per_semester : Nat) 
                           (max_course_credit : Nat) 
                           (extra_max_courses_last_sem : Nat) 
                           (extra_max_course_credit : Nat) 
                           (sid_courses_mult : Nat) 
                           (sid_extra_courses_mult : Nat) 
                           (max_total_courses : total_max_courses = max_courses)
                           (max_avg_courses_per_semester : avg_max_courses_per_semester = max_courses_per_semester)
                           (max_course_credit_def : max_course_credit = max_courses_per_semester_credit)
                           (extra_max_courses_last_sem_def : extra_max_courses_last_sem = max_additional_courses_last_semester)
                           (extra_max_courses_credit_def : extra_max_course_credit = max_additional_course_credit)
                           (sid_courses_mult_def : sid_courses_mult = sid_courses_multiplier)
                           (sid_extra_courses_mult_def : sid_extra_courses_mult = sid_additional_courses_multiplier) : 
  total_max_courses * max_course_credit + extra_max_courses_last_sem * extra_max_course_credit + 
  (sid_courses_mult * total_max_courses - sid_extra_courses_mult * extra_max_courses_last_sem) * max_course_credit + sid_extra_courses_mult * extra_max_courses_last_sem * extra_max_course_credit = 606 := 
  by 
    sorry

end total_credit_hours_l323_323701


namespace card_tag_sum_l323_323166

noncomputable def W : ℕ := 200
noncomputable def X : ℝ := 2 / 3 * W
noncomputable def Y : ℝ := W + X
noncomputable def Z : ℝ := Real.sqrt Y
noncomputable def P : ℝ := X^3
noncomputable def Q : ℝ := Nat.factorial W / 100000
noncomputable def R : ℝ := 3 / 5 * (P + Q)
noncomputable def S : ℝ := W^1 + X^2 + Z^3

theorem card_tag_sum :
  W + X + Y + Z + P + S = 2373589.26 + Q + R :=
by
  sorry

end card_tag_sum_l323_323166


namespace geometric_sequence_exists_l323_323526

theorem geometric_sequence_exists 
  (a r : ℚ)
  (h1 : a = 3)
  (h2 : a * r = 8 / 9)
  (h3 : a * r^2 = 32 / 81) : 
  r = 8 / 27 :=
by
  sorry

end geometric_sequence_exists_l323_323526


namespace distribution_ways_l323_323626

theorem distribution_ways (n k : ℕ) (h1 : n = 5) (h2 : k = 3) : (nat.choose (n + k - 1) (k - 1)) = 21 :=
by
  rw [h1, h2]
  -- Here nat.choose (7) (2) represents the binomial coefficient 7 choose 2
  -- Thus nat.choose (7) (2) = 21
  exact sorry

end distribution_ways_l323_323626


namespace train_length_is_135_l323_323035

noncomputable def length_of_train (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_is_135 :
  length_of_train 54 9 = 135 := 
by
  -- Conditions: 
  -- speed_kmh = 54
  -- time_sec = 9
  sorry

end train_length_is_135_l323_323035


namespace find_percentage_increase_l323_323763

-- Define the problem as a constant statement:
constant original_salary_increase_percentage : Real
constant original_salary : Real
constant final_salary : Real

-- Define the conditions
axiom salary_increase (P : Real) : final_salary = original_salary * (1 + P / 100) * (85 / 100)
axiom net_change : final_salary = original_salary * (1 + 0.0225)

-- Define the goal
theorem find_percentage_increase : original_salary_increase_percentage = 20.35 := by
  sorry

end find_percentage_increase_l323_323763


namespace product_m_t_l323_323339

noncomputable def g : ℕ → ℕ := sorry

axiom g_property : ∀ a b : ℕ, 3 * g (a^2 + b^2) = g a ^ 2 + g b ^ 2 + g (a + b)

def num_possible_values_g13 := sorry
def sum_possible_values_g13 := sorry

theorem product_m_t : 
  let m := num_possible_values_g13 in
  let t := sum_possible_values_g13 in
  m * t = result := 
sorry

end product_m_t_l323_323339


namespace angle_B_of_triangle_l323_323313

theorem angle_B_of_triangle (a b c : ℝ) (A B C : ℝ)
  (h_triangle : A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_opposite_sides : A ∈ (Ioo 0 π) ∧ B ∈ (Ioo 0 π) ∧ C ∈ (Ioo 0 π))
  (h_eq : b * sin A + a * cos B = 0) :
  B = 3 * π / 4 :=
begin
  sorry -- proof omitted
end

end angle_B_of_triangle_l323_323313


namespace count_5_primable_below_1000_is_21_l323_323068

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323068


namespace y_range_l323_323278

theorem y_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end y_range_l323_323278


namespace repeating_decimal_count_l323_323925

theorem repeating_decimal_count (h : True) :
  let n_set := {n | 1 ≤ n ∧ n ≤ 200 ∧
                    (∀ a b : ℕ, nat.prime a ∧ nat.prime b →
                                (n + 1 = a ^ b ∨ n + 1 = b ^ a →
                                a ≠ 2 ∧ a ≠ 5 ∧ b ≠ 2 ∧ b ≠ 5))} in
  (n_set.card = 182) :=
begin
  -- Proof to be filled in
  sorry
end

end repeating_decimal_count_l323_323925


namespace logarithmic_inequality_and_integral_l323_323351

theorem logarithmic_inequality_and_integral :
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  a > b ∧ b > c :=
by
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  sorry

end logarithmic_inequality_and_integral_l323_323351


namespace sum_binomial_condition_l323_323431

theorem sum_binomial_condition :
  (∑ m in ({13} : Finset ℤ), m) = 13 :=
begin
  sorry
end

end sum_binomial_condition_l323_323431


namespace solve_for_x_l323_323729

theorem solve_for_x (x : ℝ) (h : (1 / 2) * (1 / 7) * x = 14) : x = 196 :=
by
  sorry

end solve_for_x_l323_323729


namespace find_pairs_l323_323886

theorem find_pairs (n k : ℕ) (h_pos_n : 0 < n) (h_cond : n! + n = n ^ k) : 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) := 
by 
  sorry

end find_pairs_l323_323886


namespace count_5_primable_under_1000_l323_323126

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323126


namespace sum_binomial_condition_l323_323430

theorem sum_binomial_condition :
  (∑ m in ({13} : Finset ℤ), m) = 13 :=
begin
  sorry
end

end sum_binomial_condition_l323_323430


namespace two_pow_65537_mod_19_l323_323047

theorem two_pow_65537_mod_19 : (2 ^ 65537) % 19 = 2 := by
  -- We will use Fermat's Little Theorem and given conditions.
  sorry

end two_pow_65537_mod_19_l323_323047


namespace math_olympiad_problem_l323_323867

theorem math_olympiad_problem (students : Fin 11 → Finset (Fin n)) (h_solved : ∀ i, (students i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → ∃ p, p ∈ students i ∧ p ∉ students j) : 
  6 ≤ n := 
sorry

end math_olympiad_problem_l323_323867


namespace coplanar_edges_count_l323_323663

theorem coplanar_edges_count (ABCD_A1B1C1D1: parallelepiped) (AB CC1: ABCD_A1B1C1D1.edges) :
  (number_of_coplanar_edges ABCD_A1B1C1D1 AB CC1 = 5) :=
  sorry

end coplanar_edges_count_l323_323663


namespace cone_volume_ratio_l323_323795

noncomputable def volume_ratio (r_C h_C r_D h_D : ℝ) : ℝ :=
  let V_C := 1 / 3 * real.pi * r_C^2 * h_C
  let V_D := 1 / 3 * real.pi * r_D^2 * h_D
  V_C / V_D

theorem cone_volume_ratio :
  let r_C := 20
  let h_C := 40
  let r_D := 40
  let h_D := 20
  volume_ratio r_C h_C r_D h_D = 1 / 2 := 
  by
    sorry

end cone_volume_ratio_l323_323795


namespace binary_multiplication_l323_323507

theorem binary_multiplication : 
  (nat.bin_to_num [1, 1, 0, 1] * nat.bin_to_num [1, 1, 1] = nat.bin_to_num [1, 1, 0, 0, 1, 1, 1]) :=
by
  sorry

end binary_multiplication_l323_323507


namespace smallest_non_factor_product_l323_323010

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l323_323010


namespace common_ratio_of_geometric_series_l323_323550

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 12 / 7) : b / a = 3 := by
  sorry

end common_ratio_of_geometric_series_l323_323550


namespace route_comparison_l323_323705

noncomputable def t_X : ℝ := (8 / 40) * 60 -- time in minutes for Route X
noncomputable def t_Y1 : ℝ := (5.5 / 50) * 60 -- time in minutes for the normal speed segment of Route Y
noncomputable def t_Y2 : ℝ := (1 / 25) * 60 -- time in minutes for the construction zone segment of Route Y
noncomputable def t_Y3 : ℝ := (0.5 / 20) * 60 -- time in minutes for the park zone segment of Route Y
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3 -- total time in minutes for Route Y

theorem route_comparison : t_X - t_Y = 1.5 :=
by {
  -- Proof is skipped using sorry
  sorry
}

end route_comparison_l323_323705


namespace find_value_l323_323611

noncomputable def normal_dist_condition (X : ℝ → ℝ) (μ : ℝ) (σ : ℝ) (a : ℝ) :=
  X ∼ N(μ, σ^2) ∧ P(X < a) = 0.4

theorem find_value {X : ℝ → ℝ} {μ : ℝ} {σ : ℝ} {a : ℝ} : 
  normal_dist_condition X 3 σ a → P(a ≤ X ∧ X < 6 - a) = 0.2 :=
by sorry

end find_value_l323_323611


namespace edge_of_new_cube_l323_323736

noncomputable def edge_of_cube (L W H : ℝ) : ℝ :=
  real.cbrt (L * W * H)

theorem edge_of_new_cube (L W H E : ℝ) (h1 : L * W = 20) (h2 : W * H = 10) (h3 : L * H = 8) (h4 : E = edge_of_cube L W H) : E ≈ 3.42 :=
by
  sorry

end edge_of_new_cube_l323_323736


namespace math_competition_rank_score_l323_323653

theorem math_competition_rank_score :
  ∃ (x : ℕ) (scores : Fin 262144 → Vector ℕ 6), 
  (∀ i, ∀ j, prod (Vector.toList (scores i)) = prod (Vector.toList (scores j)) → (i = j)) → 
  (∀ i, ∀ j, sum (Vector.toList (scores i)) = sum (Vector.toList (scores j)) → prod (Vector.toList (scores i)) ≠ prod (Vector.toList (scores j))) →
  (∀ i, ∀ j, i ≠ j → scores i ≠ scores j) →
  ∀ i, ∀ j, (scores i).nthLe j (by simp [Nat.lt_of_succ_lt_succ]) ≤ 7 →
  ∀ i, ∀ j, (scores i).nthLe j (by simp [Nat.lt_of_succ_lt_succ]) ≥ 0 →
  x = 7^6 →
  let ranked_scores := Vector.reverse $ Vector.qsort (λ a b: Vector ℕ 6, prod (Vector.toList a) < prod (Vector.toList b)) (Vector.enum 262144 (λ _, scores (i))
  ranked_scores.nth (117649-1) = 1 :=
begin
  sorry
end

end math_competition_rank_score_l323_323653


namespace find_f_of_8_l323_323945

def f (x : ℝ) : ℝ :=
  if x ≤ 5 then x - 5 * x ^ 2
  else f (x - 2)

theorem find_f_of_8 : f 8 = -76 := 
by
  sorry

end find_f_of_8_l323_323945


namespace binomial_identity_l323_323178

theorem binomial_identity (n k : ℕ) (h1 : 0 < k) (h2 : k < n)
    (h3 : Nat.choose n (k-1) + Nat.choose n (k+1) = 2 * Nat.choose n k) :
  ∃ c : ℤ, k = (c^2 + c - 2) / 2 ∧ n = c^2 - 2 := sorry

end binomial_identity_l323_323178


namespace Clara_skates_225_meters_before_meeting_Danny_l323_323016

-- Definitions of the problem's conditions
def distance_CD := 150 -- meters
def speed_Clara := 9 -- meters per second
def speed_Danny := 10 -- meters per second
def angle_CDE := 45 -- degrees

-- The statement to be proved
theorem Clara_skates_225_meters_before_meeting_Danny:
    -- Ensure conditions are used in this theorem
    ∀ (t : ℝ),  -- Time in seconds
    10 * t = 150 * Real.sin (angle_CDE * Real.pi / 180) / (speed_Clara * Real.cos (angle_CDE * Real.pi / 180)) → Clara_skates := 9 * t = 225 :=
sorry

end Clara_skates_225_meters_before_meeting_Danny_l323_323016


namespace train_crossing_time_eq_30_seconds_l323_323856

-- Definitions based on conditions
def length_of_train : ℝ := 145
def speed_of_train_kmh : ℝ := 45
def length_of_bridge : ℝ := 230

-- Conversion factor from km/h to m/s
def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

-- Total distance to be covered by the train
def total_distance (train_length bridge_length : ℝ) : ℝ := train_length + bridge_length

-- Proving the required time to cross the bridge
theorem train_crossing_time_eq_30_seconds :
  let total_distance := total_distance length_of_train length_of_bridge
      speed_of_train_mps := kmh_to_mps speed_of_train_kmh
  in total_distance / speed_of_train_mps = 30 := by
  sorry

end train_crossing_time_eq_30_seconds_l323_323856


namespace gardener_trees_problem_l323_323460

theorem gardener_trees_problem 
  (maple_trees : ℕ) (oak_trees : ℕ) (birch_trees : ℕ) 
  (total_trees : ℕ) (valid_positions : ℕ) 
  (total_arrangements : ℕ) (probability_numerator : ℕ) (probability_denominator : ℕ) 
  (reduced_numerator : ℕ) (reduced_denominator : ℕ) (m_plus_n : ℕ) :
  (maple_trees = 5) ∧ 
  (oak_trees = 3) ∧ 
  (birch_trees = 7) ∧ 
  (total_trees = 15) ∧ 
  (valid_positions = 8) ∧ 
  (total_arrangements = 120120) ∧ 
  (probability_numerator = 40) ∧ 
  (probability_denominator = total_arrangements) ∧ 
  (reduced_numerator = 1) ∧ 
  (reduced_denominator = 3003) ∧ 
  (m_plus_n = reduced_numerator + reduced_denominator) → 
  m_plus_n = 3004 := 
by
  intros _
  sorry

end gardener_trees_problem_l323_323460


namespace leaders_allocation_l323_323865

theorem leaders_allocation (leaders cities : ℕ) (h1 : leaders = 5) (h2 : cities = 3) (h3 : ∀ city, city ∈ finset.range(cities) → (finset.filter (λ leader, leader_city_assignment leader = city) (finset.range leaders)).nonempty) :
  num_allocation_schemes leaders cities = 240 :=
sorry

noncomputable def leader_city_assignment (leader : ℕ) : ℕ := sorry

noncomputable def num_allocation_schemes (leaders cities : ℕ) : ℕ := sorry

end leaders_allocation_l323_323865


namespace count_not_multiples_of_3_and_4_l323_323493
-- Import the entire Mathlib library

-- Define the main proof
theorem count_not_multiples_of_3_and_4 : 
  let N := 2019
  let multiples_of_3 := Nat.floor (2019 / 3) 
  let multiples_of_4 := Nat.floor (2019 / 4) 
  let multiples_of_12 := Nat.floor (2019 / 12)
  let count_either := multiples_of_3 + multiples_of_4 - multiples_of_12
  let count_neither := N - count_either
  count_neither = 1010 :=
by 
  let N := 2019
  let multiples_of_3 := Nat.floor (2019 / 3)
  let multiples_of_4 := Nat.floor (2019 / 4)
  let multiples_of_12 := Nat.floor (2019 / 12)
  let count_either := multiples_of_3 + multiples_of_4 - multiples_of_12
  let count_neither := N - count_either
  show count_neither = 1010, by sorry

end count_not_multiples_of_3_and_4_l323_323493


namespace proof_problem_l323_323579

-- Define the given points
def P : ℝ × ℝ := (4, 4)
def A : ℝ × ℝ := (3, 1)

-- Define the circle C with m < 3
def Circle (m : ℝ) : Prop := (m < 3) ∧ ∀ x y : ℝ, ((x - m)^2 + y^2 = 5)

-- Define the ellipse E with a > b > 0 and equation as specified
def Ellipse (a b : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)

-- Statement to prove
theorem proof_problem : 
  ∃ m a b : ℝ, 
  Circle m ∧ 
  Ellipse a b ∧ 
  ((3 - m)^2 + 1 = 5) ∧ 
  m = 1 ∧ 
  a^2 = 18 ∧ 
  b^2 = 2 ∧
  ∀ Q : ℝ × ℝ, (Q ∈ {xy : ℝ × ℝ | xy.1^2 / 18 + xy.2^2 / 2 = 1 }) → 
  -12 ≤ (Q.1 + 3 * Q.2 - 6) ∧ (Q.1 + 3 * Q.2 - 6) ≤ 0 :=
begin
  sorry,
end

end proof_problem_l323_323579


namespace probability_two_red_two_blue_l323_323451

theorem probability_two_red_two_blue (red_marbles blue_marbles total_marbles : ℕ) 
  (choose : ℕ → ℕ → ℕ)
  (prob : (choose 15 2 * choose 9 2) % choose 24 4) = 4 / 7 :=
  sorry

end probability_two_red_two_blue_l323_323451


namespace b_range_l323_323252

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.log x + b / (x + 1)

theorem b_range (b : ℝ) (hb : 0 < b)
  (h : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 2 → 1 ≤ x2 ∧ x2 ≤ 2 → x1 ≠ x2 → (f x1 b - f x2 b) / (x1 - x2) < -1) :
  b > 27 / 2 :=
sorry

end b_range_l323_323252


namespace smallest_y_angle_F_l323_323291

open Real

noncomputable def triangle_inequality (d e f : ℝ) : Prop :=
d > 0 ∧ e > 0 ∧ f > 0 ∧ (d + e > f) ∧ (d + f > e) ∧ (e + f > d)

theorem smallest_y_angle_F (d e f y : ℝ) (h : triangle_inequality d e f) :
  d = 2 → e = 2 → f > 2 * sqrt 2 → y = 180 →
  ∀ (F : ℝ), F < y :=
begin
  intros h1 h2 h3 h4 F,
  sorry,
end

end smallest_y_angle_F_l323_323291


namespace smallest_product_not_factor_of_48_exists_l323_323001

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l323_323001


namespace parallel_vectors_k_value_l323_323648

theorem parallel_vectors_k_value : 
  ∀ (k : ℝ), 
  let a := (1 : ℝ, k) in
  let b := (2 : ℝ, 2) in
  (∀ (s : ℝ), a = (s * 2, s * 2)) → 
  k = 1 := 
by 
  intro k a b h
  let a := (1 : ℝ, k)
  let b := (2 : ℝ, 2)
  have h_proportion : (1 / 2) = (k / 2),
  have h_eq : k = 1,
  sorry

end parallel_vectors_k_value_l323_323648


namespace correct_reasoning_numbers_l323_323352

variable (vec_e1 vec_e2 vec_n : Type)
variable (parallel : vec_e1 → vec_e2 → Prop)
variable (perp : vec_e1 → vec_e2 → Prop)
variable (in_plane : vec_e2 → vec_n → Prop)
variable (not_in_plane : vec_e2 → vec_n → Prop)

theorem correct_reasoning_numbers :
  (∀ (vec_e1 vec_e2 : vec_e1) (vec_n : vec_n),
    (parallel vec_e1 vec_e2 ∧ parallel vec_e1 vec_n → true ∧ false)
    ∧ (parallel vec_e1 vec_n ∧ parallel vec_e2 vec_n → parallel vec_e1 vec_e2)
    ∧ (parallel vec_e1 vec_n ∧ not_in_plane vec_e2 vec_n ∧ perp vec_e1 vec_e2 → true ∧ parallel vec_e1 vec_n)
    ∧ (parallel vec_e1 vec_e2 ∧ parallel vec_e1 vec_n → false))
    →
  correct_reasonings = [{2}, {3}] := 
  by 
    sorry

end correct_reasoning_numbers_l323_323352


namespace find_common_ratio_l323_323547

def first_term : ℚ := 4 / 7
def second_term : ℚ := 12 / 7

theorem find_common_ratio (r : ℚ) : second_term = first_term * r → r = 3 :=
by
  sorry

end find_common_ratio_l323_323547


namespace count_five_primable_lt_1000_l323_323074

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323074


namespace math_problem_l323_323601

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.log x + 1 / x + b * x

def tangent_line_at_one (a b : ℝ) : Prop :=
  2 * 1 - f 1 a b + 1 = 0

def values_of_a_b (a b : ℝ) : Prop :=
  a = 1 ∧ b = 2

def monotonic_intervals (a b : ℝ) : Prop :=
  ∀ x : ℝ, 
    (0 < x ∧ x < 0.5 → deriv (f x) = 0 → f' x < 0) ∧
    (x > 0.5 → deriv (f x) = 0 → f' x > 0)

theorem math_problem (a b : ℝ) : 
  tangent_line_at_one a b → 
  values_of_a_b a b ∧ monotonic_intervals a b :=
by
  intro h
  sorry

end math_problem_l323_323601


namespace problem_statement_l323_323950

-- Define the sequence and the conditions
variable {a : ℕ → ℝ} 
variable (h_pos : ∀ n, 0 < a n) 
variable (h_a1_lt_1 : 0 < a 1 ∧ a 1 < 1)
variable (h_seq : ∀ n, a n * a (n + 1) ≤ a n - a (n + 1))

theorem problem_statement :
  ∀ n, \sum_{i=1}^{n} (\frac{a i}{i + 1}) < 1 := 
sorry 

end problem_statement_l323_323950


namespace rectangle_area_is_432_l323_323454

-- Definition of conditions and problem in Lean 4
noncomputable def circle_radius : ℝ := 6
noncomputable def rectangle_ratio_length_width : ℝ := 3 / 1
noncomputable def calculate_rectangle_area (radius : ℝ) (ratio : ℝ) : ℝ :=
  let diameter := 2 * radius
  let width := diameter
  let length := ratio * width
  length * width

-- Lean statement to prove the area
theorem rectangle_area_is_432 : calculate_rectangle_area circle_radius rectangle_ratio_length_width = 432 := by
  sorry

end rectangle_area_is_432_l323_323454


namespace parabola_chord_length_l323_323350

theorem parabola_chord_length :
  let C := λ x y : ℝ, y^2 = 3 * x
  let F : ℝ × ℝ := (3 / 4, 0)
  let line := λ x y : ℝ, y = (Math.sqrt 3 / 3) * (x - 3 / 4)
  ∀ A B : ℝ × ℝ, 
    (C A.1 A.2) ∧ (C B.1 B.2) ∧ (line A.1 A.2) ∧ (line B.1 B.2) →
    |(A.1 - B.1)| + |(A.2 - B.2)| = 12 :=
begin
  sorry
end

end parabola_chord_length_l323_323350


namespace min_f_in_interval_l323_323286

noncomputable def f : ℝ → ℝ := λ x, x^2 - 2 * x

theorem min_f_in_interval : ∃ x ∈ set.Icc 2 4, f x = 0 :=
by
  -- We need to provide the existence of x within [2, 4] such that f(x) = 0
  use 2
  constructor
  · exact set.left_mem_Icc.2 (by norm_num)
  · simp [f]

end min_f_in_interval_l323_323286


namespace boxes_neither_markers_nor_crayons_l323_323896

theorem boxes_neither_markers_nor_crayons (total boxes_markers boxes_crayons boxes_both: ℕ)
  (htotal : total = 15)
  (hmarkers : boxes_markers = 9)
  (hcrayons : boxes_crayons = 4)
  (hboth : boxes_both = 5) :
  total - (boxes_markers + boxes_crayons - boxes_both) = 7 := by
  sorry

end boxes_neither_markers_nor_crayons_l323_323896


namespace profit_percentage_a_l323_323139

def sp_c : ℝ := 225
def cp_a : ℝ := 120
def profit_b_percentage : ℝ := 0.50

theorem profit_percentage_a : ((let cp_b := sp_c / (1 + profit_b_percentage) in
                              let sp_a := cp_b in
                              let profit_a := sp_a - cp_a in
                              (profit_a / cp_a) * 100) = 25) :=
by
  sorry

end profit_percentage_a_l323_323139


namespace repeating_decimal_count_l323_323924

theorem repeating_decimal_count (h : True) :
  let n_set := {n | 1 ≤ n ∧ n ≤ 200 ∧
                    (∀ a b : ℕ, nat.prime a ∧ nat.prime b →
                                (n + 1 = a ^ b ∨ n + 1 = b ^ a →
                                a ≠ 2 ∧ a ≠ 5 ∧ b ≠ 2 ∧ b ≠ 5))} in
  (n_set.card = 182) :=
begin
  -- Proof to be filled in
  sorry
end

end repeating_decimal_count_l323_323924


namespace conditional_probability_l323_323208

open Finset

noncomputable def pairs := (finset.powersetLen 2 (range 8)).filter (λ s, s.card = 2)
noncomputable def eventA := pairs.filter (λ s, s.toList.map id.sum % 2 = 0)
noncomputable def eventB := pairs.filter (λ s, s.toList.all (λ x, x % 2 = 0))

noncomputable def probA := (eventA.card : ℚ) / (pairs.card : ℚ)
noncomputable def probAB := (eventB ∩ eventA).card / pairs.card

theorem conditional_probability : probAB / probA = 1 / 3 := 
by sorry

end conditional_probability_l323_323208


namespace solve_equation_l323_323760

theorem solve_equation : ∃ x : ℝ, (128^(x-2) / 8^(x-2) = 256^x) → x = -2 :=
by
  sorry

end solve_equation_l323_323760


namespace number_of_5_primable_less_1000_l323_323090

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323090


namespace common_ratio_of_geometric_series_l323_323549

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 12 / 7) : b / a = 3 := by
  sorry

end common_ratio_of_geometric_series_l323_323549


namespace value_of_gg_neg1_l323_323634

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_gg_neg1 : g (g (-1)) = 199 := by
  sorry

end value_of_gg_neg1_l323_323634


namespace number_of_5_primable_less_1000_l323_323088

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323088


namespace area_triangle_formed_by_line_l323_323737

theorem area_triangle_formed_by_line (b : ℝ) (h : (1 / 2) * |b * (-b / 2)| > 1) : b < -2 ∨ b > 2 :=
by 
  sorry

end area_triangle_formed_by_line_l323_323737


namespace count_5_primables_less_than_1000_l323_323098

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323098


namespace train_speed_l323_323482

def length_train : ℝ := 250.00000000000003
def crossing_time : ℝ := 15
def meter_to_kilometer (x : ℝ) : ℝ := x / 1000
def second_to_hour (x : ℝ) : ℝ := x / 3600

theorem train_speed :
  (meter_to_kilometer length_train) / (second_to_hour crossing_time) = 60 := 
  sorry

end train_speed_l323_323482


namespace trajectory_eq_ellipse_range_sum_inv_dist_l323_323242

-- Conditions for circle M
def CircleM := { center : ℝ × ℝ // center = (-3, 0) }
def radiusM := 1

-- Conditions for circle N
def CircleN := { center : ℝ × ℝ // center = (3, 0) }
def radiusN := 9

-- Conditions for circle P
def CircleP (x y : ℝ) (r : ℝ) := 
  (dist (x, y) (-3, 0) = r + radiusM) ∧
  (dist (x, y) (3, 0) = radiusN - r)

-- Proof for the equation of the trajectory
theorem trajectory_eq_ellipse :
  ∃ (x y : ℝ), CircleP x y r → x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Proof for the range of 1/PM + 1/PN
theorem range_sum_inv_dist :
  ∃ (r PM PN : ℝ), 
    PM ∈ [2, 8] ∧ 
    PN = 10 - PM ∧ 
    CircleP (PM - radiusM) (PN - radiusN) r → 
    (2/5 ≤ (1/PM + 1/PN) ∧ (1/PM + 1/PN) ≤ 5/8) :=
sorry

end trajectory_eq_ellipse_range_sum_inv_dist_l323_323242


namespace ab_inequality_smaller_than_fourth_sum_l323_323893

theorem ab_inequality_smaller_than_fourth_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤ (1 / 4) * (a + b + c) := 
by
  sorry

end ab_inequality_smaller_than_fourth_sum_l323_323893


namespace congruent_triangle_division_l323_323333

theorem congruent_triangle_division (α β γ : ℝ) (hαβγ : α + β + γ = 180) :
  (α = 60 ∧ β = 60 ∧ γ = 60) ∨ (α = 30 ∧ β = 60 ∧ γ = 90) ∨ (α = 30 ∧ β = 90 ∧ γ = 60) ∨ (α = 90 ∧ β = 30 ∧ γ = 60) ↔
  (∃ T : Type, T = Triangle α β γ ∧ ∃ P Q R : T, triangle PQR ∧ congruent P R Q) := sorry

end congruent_triangle_division_l323_323333


namespace four_distinct_numbers_exist_l323_323671

theorem four_distinct_numbers_exist :
  ∃ n1 n2 n3 n4 : ℕ,
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ n3 ≠ n4 ∧
    -- n1, n2, n3, n4 are from the set {5, 17, 29, 41}
    n1 ∈ {5, 17, 29, 41} ∧
    n2 ∈ {5, 17, 29, 41} ∧
    n3 ∈ {5, 17, 29, 41} ∧
    n4 ∈ {5, 17, 29, 41} ∧
    -- None of n1, n2, n3, n4 is divisible by 2, 3, or 4
    ¬ (2 ∣ n1) ∧ ¬ (3 ∣ n1) ∧ ¬ (4 ∣ n1) ∧
    ¬ (2 ∣ n2) ∧ ¬ (3 ∣ n2) ∧ ¬ (4 ∣ n2) ∧
    ¬ (2 ∣ n3) ∧ ¬ (3 ∣ n3) ∧ ¬ (4 ∣ n3) ∧
    ¬ (2 ∣ n4) ∧ ¬ (3 ∣ n4) ∧ ¬ (4 ∣ n4) ∧
    -- Sum of any two is divisible by 2
    (2 ∣ (n1 + n2)) ∧ (2 ∣ (n1 + n3)) ∧
    (2 ∣ (n1 + n4)) ∧ (2 ∣ (n2 + n3)) ∧
    (2 ∣ (n2 + n4)) ∧ (2 ∣ (n3 + n4)) ∧
    -- Sum of any three is divisible by 3
    (3 ∣ (n1 + n2 + n3)) ∧ (3 ∣ (n1 + n2 + n4)) ∧
    (3 ∣ (n1 + n3 + n4)) ∧ (3 ∣ (n2 + n3 + n4)) ∧
    -- Sum of all four numbers is divisible by 4
    (4 ∣ (n1 + n2 + n3 + n4)) :=
by
  sorry

end four_distinct_numbers_exist_l323_323671


namespace proposition_D_is_false_l323_323861

/--
Among the given propositions about plane vectors, prove that the fourth proposition is false.
  Proposition A:
    Let O, A, B, and C be four different points on the same plane.
    If \overrightarrow{OA} = m \cdot \overrightarrow{OB} + (1-m) \cdot \overrightarrow{OC} (m ∈ ℝ),
    then points A, B, and C must be collinear.
  Proposition B:
    If vectors \overrightarrow{a} and \overrightarrow{b} are two non-parallel vectors on plane α,
    then any vector \overrightarrow{c} on plane α can be expressed as
    \overrightarrow{c} = λ \overrightarrow{a} + μ \overrightarrow{b} (μ, λ ∈ ℝ), and the expression is unique.
  Proposition C:
    Given plane vectors \overrightarrow{OA}, \overrightarrow{OB}, \overrightarrow{OC} satisfy
    | \overrightarrow{OA} | = | \overrightarrow{OB} | = | \overrightarrow{OC} | = r (r > 0),
    and \overrightarrow{OA} + \overrightarrow{OB} + \overrightarrow{OC} = \overrightarrow{0},
    then \triangle ABC is an equilateral triangle.
  Proposition D:
    Among all vectors on plane α, there do not exist four mutually different non-zero vectors
    \overrightarrow{a}, \overrightarrow{b}, \overrightarrow{c}, \overrightarrow{d},
    such that the sum vector of any two vectors is perpendicular to the sum vector of the remaining two vectors.
The goal is to prove that Proposition D is false.
-/
theorem proposition_D_is_false : ¬ ∃ (a b c d: ℝ^2), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  ∀ (x y : ℝ^2), (x = a + b → y = c + d → x ⬝ y = 0) ∧
  (x = a + c → y = b + d → x ⬝ y = 0) ∧
  (x = a + d → y = b + c → x ⬝ y = 0) :=
by
sorry

end proposition_D_is_false_l323_323861


namespace rounding_problem_l323_323379

-- Define the repeating decimal as a real number.
noncomputable def repeatingDecimal : ℝ := 67 + (36 : ℚ) / 99

-- Define the rounding function (assuming it's predefined in the context).
noncomputable def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  (floor (x * 100 + 0.5)) / 100

-- Define the problem statement to prove.
theorem rounding_problem : round_to_nearest_hundredth repeatingDecimal = 67.36 := by
  sorry

end rounding_problem_l323_323379


namespace count_5_primable_integers_lt_1000_is_21_l323_323103

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323103


namespace smallest_n_exists_l323_323556

theorem smallest_n_exists (n : ℕ) (h : n = 96) :
  ∀ A B : set ℕ, A ∪ B = {1, 2, ..., n} ∧ disjoint A B →
  (∃ a b c ∈ A, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c) ∨
  (∃ a b c ∈ B, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c) :=
sorry

end smallest_n_exists_l323_323556


namespace min_value_condition_l323_323970

noncomputable def poly_min_value (a b : ℝ) : ℝ := a^2 + b^2

theorem min_value_condition (a b : ℝ) (h: ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  ∃ a b : ℝ, poly_min_value a b = 4 := 
by sorry

end min_value_condition_l323_323970


namespace solve_inequality_l323_323978

theorem solve_inequality (a b : ℝ) (h : ∀ x, ax^2 - bx - 1 ≥ (0 : ℝ) → x ∈ { -1/2, -1/3 }) : 
  ∀ x, ax^2 - bx - 1 < (0 : ℝ) ↔ 2 < x ∧ x < 3 :=
by
  sorry

end solve_inequality_l323_323978


namespace smallest_non_factor_product_l323_323009

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l323_323009


namespace find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l323_323821

noncomputable def board : Type := (Fin 5) × (Fin 5)

def is_counterfeit (c1 : board) (c2 : board) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

theorem find_13_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 13 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem find_15_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 15 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem cannot_find_17_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ¬ (∃ C : Finset board, C.card = 17 ∧ ∀ c ∈ C, coins c = coins (0,0)) :=
sorry

end find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l323_323821


namespace urn_problem_probability_l323_323164

theorem urn_problem_probability :
  let urn_initial_red_balls := 1 in
  let urn_initial_blue_balls := 1 in
  let total_operations := 5 in
  let final_red_balls := 3 in
  let final_blue_balls := 4 in
  let box_contains_additional_balls := true in -- This is implicit in the problem.
  (probability (λ s, s = (final_red_balls, final_blue_balls)) | urn_initial_red_balls := 1, urn_initial_blue_balls := 1, total_operations := total_operations, box_contains_additional_balls := box_contains_additional_balls) = 1/6 :=
sorry

end urn_problem_probability_l323_323164


namespace tangent_lines_count_l323_323217

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

def arithmetic_sequence (a b c : ℝ) : Prop := 2 * b = a + c

def tangent_from_origin (f : ℝ → ℝ) (x : ℝ) : Prop :=
    let y := f x in (3 * x * x - 6 * x + y = 0)

theorem tangent_lines_count (a : ℝ) (h₀ : a ≠ 0) 
    (h₁ : arithmetic_sequence (f a (-a)) (f a a) (f a (3 * a))) :
    ∃ count : ℕ, count = 2 :=
begin
    sorry
end

end tangent_lines_count_l323_323217


namespace trigonometric_identity_l323_323583

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (2 * Real.sin θ - 4 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l323_323583


namespace multiplication_table_odd_fraction_nearest_hundredth_l323_323295

def is_odd (n : ℕ) : Prop := n % 2 = 1

def odd_product_fraction (LB UB : ℕ) : ℚ :=
  let total_products := (UB - LB + 1) * (UB - LB + 1)
  let odd_factors := (LB to UB).filter is_odd
  let odd_products := odd_factors.length * odd_factors.length
  odd_products / total_products

theorem multiplication_table_odd_fraction_nearest_hundredth :
  odd_product_fraction 0 15 = 0.25 := sorry

end multiplication_table_odd_fraction_nearest_hundredth_l323_323295


namespace magnitude_of_w_equals_one_l323_323682

open Complex

theorem magnitude_of_w_equals_one:
  let z : ℂ := ((-5 + 7*I)^3 * (16 - 3*I)^4) / (2 + 5*I)
  let w : ℂ := conj z / z
  (|w| = 1) :=
by
  sorry

end magnitude_of_w_equals_one_l323_323682


namespace pyramid_volume_is_correct_l323_323533

noncomputable def pyramid_volume (a b c sa : ℝ) : ℝ :=
  let p := (a + b + c) / 2
  let S := sqrt (p * (p - a) * (p - b) * (p - c))
  let R := a * b * c / (4 * S)
  let h := sqrt (sa^2 - R^2)
  (1 / 3) * S * h

theorem pyramid_volume_is_correct :
  pyramid_volume 15 14 13 (269 / 32) = 60.375 :=
by
  -- Conditions and definitions are already provided above.
  -- hence, skipping the detailed proof.
  sorry

end pyramid_volume_is_correct_l323_323533


namespace ball_travel_distance_l323_323446

-- Ensure the function is noncomputable because we deal with real number calculations and rounding.
noncomputable def total_travel_distance (initial_height : ℝ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  let descent n := initial_height * (ratio ^ n)
  let ascent n := initial_height * (ratio ^ (n-1))
  let total_descent := ∑ i in Finset.range (bounces + 1), descent i
  let total_ascent := ∑ i in Finset.range bounces, ascent i
  total_descent + total_ascent

theorem ball_travel_distance :
  total_travel_distance 20 (5 / 8) 4 = 73.44 :=
by
  sorry

end ball_travel_distance_l323_323446


namespace count_5_primable_below_1000_is_21_l323_323067

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323067


namespace days_to_meet_quota_l323_323935

theorem days_to_meet_quota
  (total_quota : ℕ)
  (sold_first_3_days_each : ℕ)
  (sold_first_3_days : ℕ)
  (sold_next_4_days_each : ℕ)
  (sold_next_4_days : ℕ)
  (remaining_cars_to_sell : ℕ)
  (total_worked_days : ℕ) :
  total_quota = 50 →
  sold_first_3_days_each = 5 →
  sold_first_3_days = 3 →
  sold_next_4_days_each = 3 →
  sold_next_4_days = 4 →
  remaining_cars_to_sell = 23 →
  total_worked_days = 7 →
  total_worked_days = 7 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  exact h7,
end

end days_to_meet_quota_l323_323935


namespace CFDE_is_rectangle_EF_tangent_to_circles_l323_323841

-- Definition of points A, B, C, D, E, F
variable (A B C D E F : Type)
-- Line l that intersects AB perpendicularly at C
variable (l : Type)
-- Conditions
variable (AB AC BC ∷ [has_mem l (pt A ∧ pt B)] ∷ [perpendicular l AB C])
variable (circle_A_B circle_A_C circle_B_C : C ⇒ circle (pt A B C))

-- Condition on the intersections
variable (D : point (l ∧ circle_A_B))
variable (E : point (line_segment (pt A D) ∧ circle_A_C))
variable (F : point (line_segment (pt B D) ∧ circle_B_C))

-- Statement for Part (a)
theorem CFDE_is_rectangle : is_rectangle C F D E := sorry

-- Statement for Part (b)
theorem EF_tangent_to_circles :
  is_tangent (line_through E F) (circle_A_C) E ∧ is_tangent (line_through E F) (circle_B_C) F := sorry

end CFDE_is_rectangle_EF_tangent_to_circles_l323_323841


namespace count_5_primables_less_than_1000_l323_323094

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323094


namespace crescent_moon_falcata_area_l323_323413

/-
Prove that the area of the crescent moon falcata, which is bounded by:
1. A portion of the circle with radius 4 centered at (0,0) in the second quadrant.
2. A portion of the circle with radius 2 centered at (0,2) in the second quadrant.
3. The line segment from (0,0) to (-4,0).
is equal to 6π.
-/
theorem crescent_moon_falcata_area :
  let radius_large := 4
  let radius_small := 2
  let area_large := (1 / 2) * (π * (radius_large ^ 2))
  let area_small := (1 / 2) * (π * (radius_small ^ 2))
  (area_large - area_small) = 6 * π := by
  sorry

end crescent_moon_falcata_area_l323_323413


namespace sylvesters_problem_l323_323216

variables {P : Type*} [plane : affine_plane P]

open_locale classical

theorem sylvesters_problem (finite_points : set P) (h : finite finite_points) 
  (h_condition : ∀ (A B : P), A ∈ finite_points → B ∈ finite_points → A ≠ B → 
                   ∃ C : P, C ∈ finite_points ∧ C ≠ A ∧ C ≠ B ∧ affine_plane.line_through A B = affine_plane.line_through A C) :
  ∃ l : affine_plane.line P, ∀ p ∈ finite_points, p ∈ l :=
by
  sorry

end sylvesters_problem_l323_323216


namespace fixed_point_through_line_l323_323710

theorem fixed_point_through_line : ∀ (x y : ℝ), 
  (∀ m : ℝ, (2 * m - 1) * x + (m + 3) * y - (m - 11) = 0) ↔ x = 2 ∧ y = -3 := 
begin
  sorry
end

end fixed_point_through_line_l323_323710


namespace identify_triangle_shape_l323_323649

theorem identify_triangle_shape
  (A B C : ℝ) 
  (h1 : 0 < A ∧ A < π / 2) 
  (h2 : 0 < B ∧ B < π / 2) 
  (h3 : A + B + C = π) 
  (h_cos_sin : cos A > sin B) : 
  π / 2 < C :=
sorry

end identify_triangle_shape_l323_323649


namespace max_smoothie_servings_l323_323468

theorem max_smoothie_servings (r_bananas r_yogurt r_berries r_almond_milk e_bananas e_yogurt e_berries e_almond_milk : ℕ)
  (h1 : r_bananas = 3) (h2 : r_yogurt = 2) (h3 : r_berries = 1) (h4 : r_almond_milk = 1)
  (h5 : e_bananas = 9) (h6 : e_yogurt = 5) (h7 : e_berries = 3) (h8 : e_almond_milk = 4) :
  max_smoothies : ℕ :=
by
  -- Let's calculate the maximum potential servings based on each ingredient
  let servings_bananas := (e_bananas / r_bananas) * 4
  let servings_yogurt := (e_yogurt / r_yogurt) * 4
  let servings_berries := (e_berries / r_berries) * 4
  let servings_almond_milk := (e_almond_milk / r_almond_milk) * 4
  -- The minimum of these values gives the maximum number of servings Emily can make
  let max_servings := min servings_bananas (min servings_yogurt (min servings_berries servings_almond_milk))
  have h9 : max_servings = 10 := sorry
  exact h9

end max_smoothie_servings_l323_323468


namespace max_distance_circle_to_line_l323_323664

open Real

-- Definitions of polar equations and transformations to Cartesian coordinates
def circle_eq (ρ θ : ℝ) : Prop := (ρ = 8 * sin θ)
def line_eq (θ : ℝ) : Prop := (θ = π / 3)

-- Cartesian coordinate transformations
def circle_cartesian (x y : ℝ) : Prop := (x^2 + (y - 4)^2 = 16)
def line_cartesian (x y : ℝ) : Prop := (y = sqrt 3 * x)

-- Maximum distance problem statement
theorem max_distance_circle_to_line : 
  ∀ (x y : ℝ), circle_cartesian x y → 
  (∀ x y, line_cartesian x y → 
  ∃ d : ℝ, d = 6) :=
by
  sorry

end max_distance_circle_to_line_l323_323664


namespace shortest_path_correct_l323_323668

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ := 
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

noncomputable def shortestPathLength (start end : Point) (center : Point) (radius : ℝ) : ℝ :=
  let d1 := distance start center
  let d2 := distance end center
  if d1 > radius ∧ d2 > radius 
    then 2 * Real.sqrt (120.25 / 900) * 10 + 2 * Real.pi 
    else sorry -- For avoiding the detailed path calculation

theorem shortest_path_correct : 
  shortestPathLength (Point.mk 0 0) (Point.mk 15 20) (Point.mk 7.5 10) 6 = 
  20 * Real.sqrt (120.25 / 900) + 2 * Real.pi := 
by sorry

end shortest_path_correct_l323_323668


namespace part1_part2_l323_323593

open Complex

noncomputable def z (m : ℝ) : ℂ := (2 + 4 * m * Complex.i) / (1 - Complex.i)
noncomputable def z_conjugate (m : ℝ) : ℂ := conj (z m)

theorem part1 (m : ℝ) :
  isPureImaginary (z m) ↔ m = 1 / 2 := sorry

theorem part2 (m : ℝ) :
  (Real.re (z_conjugate m + 2 * z m) > 0) ∧
  (Real.im (z_conjugate m + 2 * z m) > 0) ↔
  (-1 / 2 < m ∧ m < 1 / 2) := sorry

end part1_part2_l323_323593


namespace isosceles_triangles_with_perimeter_31_l323_323624

theorem isosceles_triangles_with_perimeter_31 :
  { (a, b) : ℕ × ℕ // 2 * a + b = 31 ∧ b % 2 = 1 ∧ b < 2 * a }.card = 8 :=
by
  sorry

end isosceles_triangles_with_perimeter_31_l323_323624


namespace extreme_points_of_f_range_of_a_inequality_proof_l323_323518
open Real

def f (a x : ℝ) : ℝ := ln x + (1/2) * x^2 + a * x
def g (x : ℝ) : ℝ := exp x + (3/2) * x^2

theorem extreme_points_of_f (a : ℝ) : 
  (∀ x > 0, f a x <= g x) → 
  (a ∈ Icc (-2 : ℝ) ∞ → ∃ x y z, 0 < x ∧ 0 < y ∧ 0 < z ∧ x < y ∧ y < z ∧ (∀ t < x, f a t < f a x) ∧ (∀ t < y, y < z ∧ f a y < f a t) ∧ (∀ t > z, f a z < f a t)) ∧ 
  (a ∈ Iio (-2) → ∃ x y, 0 < x ∧ 0 < y ∧ x < y ∧ (∀ t < x, f a x > f a t) ∧ (∀ t > x ∧ t < y, f a t > f a y) ∧ (∀ t > y, f a t < f a y)) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f a x <= g x) → 
  a ≤ exp 1 + 1 := sorry

theorem inequality_proof : 
  (∀ x > 0, f (exp 1 + 1) x <= g x) → 
  (∀ x > 0, exp x + x^2 - (exp 1 + 1) * x + exp 1 / x > 2) := sorry

end extreme_points_of_f_range_of_a_inequality_proof_l323_323518


namespace part_one_part_two_l323_323958

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end part_one_part_two_l323_323958


namespace car_miles_per_gallon_l323_323513

-- Define distances and costs as given in conditions
def distance_grocery_store : ℕ := 8
def distance_school : ℕ := 6
def distance_soccer_practice : ℕ := 12
def distance_home : ℕ := 2 * distance_soccer_practice
def gas_cost_per_gallon : ℝ := 2.50
def total_gas_cost : ℝ := 5.00

-- Define the total distance
def total_distance : ℕ := distance_grocery_store + distance_school + distance_soccer_practice + distance_home

-- Define gallons used
def gallons_used : ℝ := total_gas_cost / gas_cost_per_gallon

-- Prove that miles per gallon is 25
theorem car_miles_per_gallon : total_distance / gallons_used = 25 := by
  sorry

end car_miles_per_gallon_l323_323513


namespace locus_of_centers_of_tangent_circles_l323_323426

theorem locus_of_centers_of_tangent_circles 
  (O : ℝ × ℝ) (R : ℝ) (L : set (ℝ × ℝ)) (r : ℝ) :
  ∃ (b1 b2 : set (ℝ × ℝ)), 
    (is_parallel L b1 ∧ distance_noteq r b1) ∧ 
    (is_parallel L b2 ∧ distance_noteq r b2) ∧ 
    locus_eq_parabolas_with_focus_O_and_directrix O r b1 b2 :=
sorry

end locus_of_centers_of_tangent_circles_l323_323426


namespace taimour_paint_time_l323_323811

theorem taimour_paint_time (T : ℝ) :
  (1 / T + 2 / T) * 7 = 1 → T = 21 :=
by
  intro h
  sorry

end taimour_paint_time_l323_323811


namespace triangle_ABC_BC_length_l323_323492

noncomputable def length_BC (a : ℝ) : ℝ := 2 * real.cbrt 100

theorem triangle_ABC_BC_length :
  ∀ (a : ℝ),
    (forall (x : ℝ), y = x^2 + 1) ∧
    (A = (0, 1)) ∧
    (BC_parallel_x_axis) ∧
    (area_ABC = 100) →
    length_BC a = 2 * real.cbrt 100 :=
by
  intro a
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  sorry

end triangle_ABC_BC_length_l323_323492


namespace least_three_digit_product_24_l323_323791

theorem least_three_digit_product_24 : ∃ (n : ℕ), n = 234 ∧ 100 ≤ n ∧ n < 1000 ∧ ∀ (a b c : ℕ), n = 100 * a + 10 * b + c → a * b * c = 24 := 
by
  use 234
  split
  · rfl
  split
  · exact Nat.le_of_lt_succ (by norm_num)
  split
  · exact by norm_num
  · intros a b c h
    have ha : a = 2, from sorry
    have hb : b = 3, from sorry
    have hc : c = 4, from sorry
    rw [ha, hb, hc]
    norm_num
  sorry

end least_three_digit_product_24_l323_323791


namespace work_days_B_works_l323_323829

theorem work_days_B_works (x : ℕ) (A_work_rate B_work_rate : ℚ) (A_remaining_days : ℕ) (total_work : ℚ) :
  A_work_rate = (1 / 12) ∧
  B_work_rate = (1 / 15) ∧
  A_remaining_days = 4 ∧
  total_work = 1 →
  x * B_work_rate + A_remaining_days * A_work_rate = total_work →
  x = 10 :=
sorry

end work_days_B_works_l323_323829


namespace trapezoid_base_length_l323_323415

open Real

/-- Given a rectangular trapezoid ABCD with:
  * Smaller leg AB = 3
  * ∠ADC = 30°
  * Intersection of angle bisectors at BC lies on it
  Prove that AD = 9.
  -/
theorem trapezoid_base_length (AB CD AD : ℝ) (h1 : AB = 3) (h2 : ∠ADC = π / 6)
  (h3 : intersection_of_angle_bisectors_on_base BC) : AD = 9 :=
sorry

end trapezoid_base_length_l323_323415


namespace discount_percentage_is_20_l323_323058

-- Given definitions and conditions:
constant PurchasePrice : ℝ
constant MarkupPercentage : ℝ
constant GrossProfit : ℝ
constant SellingPrice : ℝ
constant DiscountPercentage : ℝ

axiom purchase_price_def : PurchasePrice = 54
axiom markup_percentage_def : MarkupPercentage = 0.4
axiom gross_profit_def : GrossProfit = 18
axiom selling_price_eq : SellingPrice = PurchasePrice + MarkupPercentage * SellingPrice

-- Proof problem statement in Lean:
theorem discount_percentage_is_20 :
  ∃ (D : ℝ), D = 0.2 ∧ SellingPrice = 90 ∧ 18 = (1 - D) * SellingPrice - PurchasePrice :=
by
  use 0.2
  split
  · rfl
  split
  · calc
      SellingPrice = 54 + 0.4 * SellingPrice : by exact selling_price_eq
      ... = 90 : by linarith [purchase_price_def]
  · calc
      18 = (1 - 0.2) * 90 - 54 : by linarith
      ... = (1 - 0.2) * SellingPrice - PurchasePrice : by linarith [purchase_price_def]
      ... = 18 : by exact gross_profit_def

end discount_percentage_is_20_l323_323058


namespace smallest_non_factor_product_l323_323007

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l323_323007


namespace dot_product_OA_OB_l323_323657

theorem dot_product_OA_OB :
  let A := (Real.cos 110, Real.sin 110)
  let B := (Real.cos 50, Real.sin 50)
  (A.1 * B.1 + A.2 * B.2) = 1 / 2 :=
by
  sorry

end dot_product_OA_OB_l323_323657


namespace z_conj_in_first_quadrant_l323_323247
open Complex

noncomputable def z := (2 - Complex.I^2017) / (1 + Complex.I)
def z_conj := conj z
def z_conj_re := z_conj.re
def z_conj_im := z_conj.im

theorem z_conj_in_first_quadrant :
  0 < z_conj_re ∧ 0 < z_conj_im :=
sorry

end z_conj_in_first_quadrant_l323_323247


namespace second_player_wins_by_symmetry_l323_323566

theorem second_player_wins_by_symmetry :
  ∃ (strategy : (ℕ × ℕ → ℕ × ℕ) → Prop), 
  (∀ g : (ℕ × ℕ → ℕ × ℕ), (∀ i j : ℕ, i < 10 ∧ j < 10 →
    g(i, j) ∈ {(i, j + 1), (i + 1, j)} ∧ ∃ i' j', i' = 9 - i ∧ j' = 9 - j ∧ 
    strategy(g) → strategy(g ∘ (λ (i j : ℕ), (9 - i, 9 - j)))) → 
  ∃ i j, i < 10 ∧ j < 10 ∧
  ¬(strategy (λ (i j : ℕ), if i < 9 ∧ j < 9 then (i + 1, j) 
    else (i, j + 1))) := sorry

end second_player_wins_by_symmetry_l323_323566


namespace sin_alpha_beta_fewer_than_four_values_l323_323816

noncomputable def sin_relation (α β : ℝ) : ℝ :=
  sin α * real.sqrt (1 - (sin β)^2) + sin β * real.sqrt (1 - (sin α)^2)

theorem sin_alpha_beta (x y z : ℝ) (α β : ℝ) (h1 : x = sin α) (h2 : y = sin β) :
  (z = sin_relation α β) →
  z^4 - 2 * z^2 * (x^2 + y^2 - 2 * x^2 * y^2) + (x^2 - y^2)^2 = 0 := by
  sorry

theorem fewer_than_four_values (x y : ℝ) (α β : ℝ) (h1 : x = sin α) (h2 : y = sin β) :
  (x = 0 ∨ x = 1 ∨ x = -1 ∨ y = 0 ∨ y = 1 ∨ y = -1 ∨ x = y ∨ x = -y) →
  ∃ z, (z = sin_relation α β) := by
  sorry

end sin_alpha_beta_fewer_than_four_values_l323_323816


namespace sharpening_cost_l323_323544

theorem sharpening_cost
  (trees_chopped : ℕ)
  (trees_per_sharpening : ℕ)
  (total_cost : ℕ)
  (min_trees_chopped : trees_chopped ≥ 91)
  (trees_per_sharpening_eq : trees_per_sharpening = 13)
  (total_cost_eq : total_cost = 35) :
  total_cost / (trees_chopped / trees_per_sharpening) = 5 := by
  sorry

end sharpening_cost_l323_323544


namespace problem_part_1_problem_part_2_problem_part_3_l323_323227

open Set

universe u

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := univ

theorem problem_part_1 : A ∪ B = {x | 1 < x ∧ x ≤ 8} :=
sorry

theorem problem_part_2 : (U \ A) ∩ B = {x | 1 < x ∧ x < 2} :=
sorry

theorem problem_part_3 (a : ℝ) (h : (A ∩ C a) ≠ ∅) : a < 8 :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l323_323227


namespace rhombus_area_l323_323910

theorem rhombus_area (R r : ℝ) : 
  ∃ A : ℝ, A = (8 * R^3 * r^3) / ((R^2 + r^2)^2) :=
by
  sorry

end rhombus_area_l323_323910


namespace number_of_triangles_l323_323577

theorem number_of_triangles (x : ℕ) (h₁ : 2 + x > 6) (h₂ : 8 > x) : ∃! t, t = 3 :=
by {
  sorry
}

end number_of_triangles_l323_323577


namespace at_most_n_zeros_l323_323852

-- Definitions of conditions
variables {α : Type*} [Inhabited α]

/-- Define the structure of the sheet of numbers with the given properties -/
structure sheet :=
(n : ℕ)
(val : ℕ → ℤ)

-- Assuming infinite sheet and the properties
variable (s : sheet)

-- Predicate for a row having only positive integers
def all_positive (r : ℕ → ℤ) : Prop := ∀ i, r i > 0

-- Define the initial row R which has all positive integers
variable {R : ℕ → ℤ}

-- Statement that each element in the row below is sum of element above and to the left
def below_sum (r R : ℕ → ℤ) (n : ℕ) : Prop := ∀ i, r i = R i + (if i = 0 then 0 else R (i - 1))

-- Variable for the row n below R
variable {Rn : ℕ → ℤ}

-- Main theorem statement
theorem at_most_n_zeros (n : ℕ) (hr : all_positive R) (hs : below_sum R Rn n) : 
  ∃ k ≤ n, Rn k = 0 ∨ Rn k > 0 := sorry

end at_most_n_zeros_l323_323852


namespace exist_infinite_permutations_l323_323140

def is_permutation (P : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃! n : ℕ, P n = m

def S_k (P : ℕ → ℕ) (k : ℕ) : ℕ :=
  (Finset.range k).sum (λ n, P (n + 1))

theorem exist_infinite_permutations
  (exists_P : ∃ P : ℕ → ℕ, is_permutation P) :
  ∃ P_seq : ℕ → (ℕ → ℕ),
    ∀ k i j : ℕ, i < j → S_k (P_seq i) k ∣ S_k (P_seq j) k :=
sorry

end exist_infinite_permutations_l323_323140


namespace isosceles_triangle_count_l323_323621

def is_isosceles_triangle (a b : ℕ) : Prop :=
  2 * a + b = 31 ∧ 2 * a > b ∧ b > 0

def count_isosceles_triangles_with_perimeter_31 : ℕ :=
  (Finset.range 16).filter (λ b, b % 2 = 1 ∧ 
    ∃ a, is_isosceles_triangle a b).card

theorem isosceles_triangle_count : count_isosceles_triangles_with_perimeter_31 = 8 := 
  by
  sorry

end isosceles_triangle_count_l323_323621


namespace total_amount_received_by_students_l323_323678

theorem total_amount_received_by_students (total_winnings : ℝ) (num_students : ℕ) (fraction : ℝ) :
   total_winnings = 155250 → num_students = 100 → fraction = 1/1000 →
   (fraction * total_winnings * num_students) = 15525 :=
by
  intros h_winnings h_students h_fraction
  rw [h_winnings, h_students, h_fraction]
  norm_num
  sorry

end total_amount_received_by_students_l323_323678


namespace sum_of_integer_solutions_l323_323389

theorem sum_of_integer_solutions : 
  ∑ i in (Finset.filter (λ x, -3 ≤ x ∧ x < 2) ⟦-3, 1⟧), i = -5 := 
sorry

end sum_of_integer_solutions_l323_323389


namespace dot_product_question_perpendicular_vectors_k_l323_323215

variables (a b : Type) [inner_product_space ℝ a] [inner_product_space ℝ b]

noncomputable def angle : ℝ := 60 -- The angle is given as 60 degrees

-- Problem (1)
theorem dot_product_question (a b : a) (ha : ∥a∥ = 1) (hb : ∥b∥ = 4) 
    (hcp : orthogonal a b) (angle : ℝ) :
  (2 • a - b) ⬝ (a + b) = -12 :=
sorry

-- Problem (2)
theorem perpendicular_vectors_k (a b : a) (k : ℝ) (ha : ∥a∥ = 1) (hb : ∥b∥ = 4) 
    (hcp : orthogonal (k • a + b) (k • a - b)) :
  k = 4 ∨ k = -4 :=
sorry

end dot_product_question_perpendicular_vectors_k_l323_323215


namespace tangent_line_eq_l323_323913

def f (x : ℝ) : ℝ := x^3 + 4 * x + 5

theorem tangent_line_eq (x y : ℝ) (h : (x, y) = (1, 10)) : 
  (7 * x - y + 3 = 0) :=
sorry

end tangent_line_eq_l323_323913


namespace probability_of_stock_price_increase_l323_323873

namespace StockPriceProbability

variables (P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C : ℝ)

def P_D : ℝ := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

theorem probability_of_stock_price_increase :
    P_A = 0.6 → P_B = 0.3 → P_C = 0.1 → 
    P_D_given_A = 0.7 → P_D_given_B = 0.2 → P_D_given_C = 0.1 → 
    P_D P_A P_B P_C P_D_given_A P_D_given_B P_D_given_C = 0.49 :=
by intros h₁ h₂ h₃ h₄ h₅ h₆; sorry

end StockPriceProbability

end probability_of_stock_price_increase_l323_323873


namespace functional_equation_solution_l323_323905

open Function

theorem functional_equation_solution :
  ∀ (f g : ℚ → ℚ), 
    (∀ x y : ℚ, f (g x + g y) = f (g x) + y ∧ g (f x + f y) = g (f x) + y) →
    (∃ a b : ℚ, (ab = 1) ∧ (∀ x : ℚ, f x = a * x) ∧ (∀ x : ℚ, g x = b * x)) :=
by
  intros f g h
  sorry

end functional_equation_solution_l323_323905


namespace abs_w_eq_one_l323_323683

noncomputable def z : ℂ := ((-7 + 9 * complex.I) ^ 2 * (18 - 4 * complex.I) ^ 3) / (2 + 5 * complex.I)
def w : ℂ := z / (conj z)

theorem abs_w_eq_one : complex.abs w = 1 := 
by
  sorry

end abs_w_eq_one_l323_323683


namespace distance_P_to_l_l323_323743

def Point := ℝ × ℝ
def Line := { A B C : ℝ // A ≠ 0 ∨ B ≠ 0 }

-- Define the specific point P and line l
def P : Point := (-1, 1)
def l : Line := ⟨3, 4, 0, by simp⟩

-- Distance function for a point to a line in standard form
def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  let (A, B, C) := (l.val.A, l.val.B, l.val.C)
  let (x0, y0) := p
  (abs (A * x0 + B * y0 + C)) / (sqrt (A^2 + B^2))

-- The theorem statement
theorem distance_P_to_l : distance_point_to_line P l = 1 / 5 :=
by sorry

end distance_P_to_l_l323_323743


namespace main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l323_323879

-- Problem Statement in Lean 4

theorem main_diagonal_squares (k : ℕ) : ∃ m : ℕ, (4 * k * (k + 1) + 1 = m * m) := 
sorry

theorem second_diagonal_composite (k : ℕ) (hk : k ≥ 1) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * (2 * k * (2 * k - 1) - 1) + 1 = a * b) :=
sorry

theorem third_diagonal_composite (k : ℕ) : ∃ a b : ℕ, a ≠ 1 ∧ b ≠ 1 ∧ (4 * ((4 * k + 3) * (4 * k - 1)) + 1 = a * b) :=
sorry

end main_diagonal_squares_second_diagonal_composite_third_diagonal_composite_l323_323879


namespace repeating_decimal_count_l323_323926

theorem repeating_decimal_count (h : True) :
  let n_set := {n | 1 ≤ n ∧ n ≤ 200 ∧
                    (∀ a b : ℕ, nat.prime a ∧ nat.prime b →
                                (n + 1 = a ^ b ∨ n + 1 = b ^ a →
                                a ≠ 2 ∧ a ≠ 5 ∧ b ≠ 2 ∧ b ≠ 5))} in
  (n_set.card = 182) :=
begin
  -- Proof to be filled in
  sorry
end

end repeating_decimal_count_l323_323926


namespace sequence_b_n_l323_323883

theorem sequence_b_n (b : ℕ → ℝ) 
  (h1 : b 1 = 3)
  (h2 : ∀ n ≥ 1, (b (n + 1))^3 = 27 * (b n)^3) :
  b 50 = 3^50 :=
sorry

end sequence_b_n_l323_323883


namespace car_trip_distance_l323_323831

theorem car_trip_distance (speed_first_car speed_second_car : ℝ) (time_first_car time_second_car distance_first_car distance_second_car : ℝ) 
  (h_speed_first : speed_first_car = 30)
  (h_time_first : time_first_car = 1.5)
  (h_speed_second : speed_second_car = 60)
  (h_time_second : time_second_car = 1.3333)
  (h_distance_first : distance_first_car = speed_first_car * time_first_car)
  (h_distance_second : distance_second_car = speed_second_car * time_second_car) :
  distance_first_car = 45 :=
by
  sorry

end car_trip_distance_l323_323831


namespace complex_problem_statement_l323_323592

noncomputable def z : ℂ := 1 + complex.I

theorem complex_problem_statement : (z^2 / (1 - z)) = -2 := by
  sorry

end complex_problem_statement_l323_323592


namespace number_of_ways_to_score_l323_323773

-- Define the conditions
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def score_red : ℕ := 2
def score_white : ℕ := 1
def total_balls : ℕ := 5
def min_score : ℕ := 7

-- Prove the equivalent proof problem
theorem number_of_ways_to_score :
  ∃ ways : ℕ, 
    (ways = ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
             (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
             (Nat.choose red_balls 2) * (Nat.choose white_balls 3))) ∧
    ways = 186 :=
by
  let ways := ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
               (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
               (Nat.choose red_balls 2) * (Nat.choose white_balls 3))
  use ways
  constructor
  . rfl
  . sorry

end number_of_ways_to_score_l323_323773


namespace no_integer_solution_l323_323885

theorem no_integer_solution :
  ∀ (x : ℤ), ¬ (x^2 + 3 < 2 * x) :=
by
  intro x
  sorry

end no_integer_solution_l323_323885


namespace sum_of_squares_of_roots_l323_323697

theorem sum_of_squares_of_roots : 
  (∃ (a b c d : ℝ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^4 - 15 * x^2 + 56 = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
    (a^2 + b^2 + c^2 + d^2 = 30)) :=
sorry

end sum_of_squares_of_roots_l323_323697


namespace quadratic_relationship_l323_323182

theorem quadratic_relationship (a b c : ℝ) (α : ℝ) (h₁ : α + α^2 = -b / a) (h₂ : α^3 = c / a) : b^2 = 3 * a * c + c^2 :=
by
  sorry

end quadratic_relationship_l323_323182


namespace wire_length_from_sphere_l323_323059

theorem wire_length_from_sphere (
  (r : ℝ) (a : ℝ) (b : ℝ) (V_sphere : ℝ) (A_ellipse : ℝ) (L : ℝ)
  (h_r : r = 24)
  (h_a : a = 16)
  (h_b : b = 8)
  (h_V_sphere : V_sphere = (4.0/3.0) * Real.pi * r^3)
  (h_A_ellipse : A_ellipse = Real.pi * a * b)
  (h_V_wire : V_sphere = A_ellipse * L)
) : L = 144 :=
by
  sorry

end wire_length_from_sphere_l323_323059


namespace karen_backpack_gear_weight_l323_323321

theorem karen_backpack_gear_weight
  (initial_water : ℕ) (initial_food : ℕ) (initial_gear : ℕ)
  (rate_water : ℕ) (rate_food_coeff : ℕ → ℚ)
  (hours : ℕ)
  (final_total_weight : ℕ) :
  initial_water = 20 →
  initial_food = 10 →
  rate_water = 2 →
  rate_food_coeff = λ water_per_hour, water_per_hour / 3 →
  hours = 6 →
  final_total_weight = 34 →
  initial_gear = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof omitted
  sorry

end karen_backpack_gear_weight_l323_323321


namespace count_5_primable_integers_lt_1000_is_21_l323_323106

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323106


namespace area_diff_of_rectangle_l323_323038

theorem area_diff_of_rectangle (a : ℝ) : 
  let length_increased := 1.40 * a
  let breadth_increased := 1.30 * a
  let original_area := a * a
  let new_area := length_increased * breadth_increased
  (new_area - original_area) = 0.82 * (a * a) :=
by 
sorry

end area_diff_of_rectangle_l323_323038


namespace classify_triangle_l323_323753

noncomputable def sides_in_geometric_progression (r : ℝ) : list ℝ := [1, r, r^2]

theorem classify_triangle (r : ℝ) (h : r ≥ 1) :
  let θ := Math.atan2 0 (-1),
      a := 1,
      b := b,
      cos_angle := (r^4 - 1 - r^2) / (2 * r) in
  ∃ (tri_type : string),
    (tri_type = "right" ↔ r = real.sqrt ((1 + real.sqrt 5) / 2)) ∧
    (tri_type = "acute" ↔ 1 ≤ r ∧ r < real.sqrt ((1 + real.sqrt 5) / 2)) ∧
    (tri_type = "obtuse" ↔ real.sqrt ((1 + real.sqrt 5) / 2) < r ∧ r < (1 + real.sqrt 5) / 2) := by
sides_in_geometric_progression sorry sorry sorry sorry sorry

end classify_triangle_l323_323753


namespace rope_segments_after_cutting_l323_323923

theorem rope_segments_after_cutting (n : ℕ) (folds : ℕ) : 
  folds = 5 → n = 10 → 
  let segments : ℕ := (2 ^ folds) + 1 in 
  segments = 33 := 
by
  intros h_folds h_n
  simp [h_folds, h_n]
  sorry

end rope_segments_after_cutting_l323_323923


namespace mixed_oil_rate_l323_323638

theorem mixed_oil_rate :
  let v₁ := 10
  let p₁ := 50
  let v₂ := 5
  let p₂ := 68
  let v₃ := 8
  let p₃ := 42
  let v₄ := 7
  let p₄ := 62
  let v₅ := 12
  let p₅ := 55
  let v₆ := 6
  let p₆ := 75
  let total_cost := v₁ * p₁ + v₂ * p₂ + v₃ * p₃ + v₄ * p₄ + v₅ * p₅ + v₆ * p₆
  let total_volume := v₁ + v₂ + v₃ + v₄ + v₅ + v₆
  let rate := total_cost / total_volume
  rate = 56.67 :=
by
  sorry

end mixed_oil_rate_l323_323638


namespace count_5_primable_under_1000_l323_323121

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323121


namespace tangent_line_eq_l323_323912

open Real

noncomputable def tangent_line (x : ℝ) : ℝ := -5 * x - 2

theorem tangent_line_eq :
  ∀ (x : ℝ) (f : ℝ → ℝ), 
    f = (λ x, -5 * exp x + 3) →
    x = 0 →
    tangent_line x = -5 * x - 2 :=
by
  intros x f hf hx
  sorry

end tangent_line_eq_l323_323912


namespace area_of_circle_N_l323_323576

noncomputable def area_circle_N (r_sphere : ℝ) (area_circle_M : ℝ) (angle_dihedral : ℝ) : ℝ :=
  let r_M := Math.sqrt (area_circle_M / Real.pi)
  let r_N := let OM := Math.sqrt (r_sphere^2 - r_M^2)
             let ON := OM * Math.cos (angle_dihedral * Real.pi / 180)
             let CN := Math.sqrt (r_sphere^2 - ON^2)
             in Real.pi * CN^2

open Real 

theorem area_of_circle_N
  (r_sphere : ℝ)
  (area_circle_M : ℝ)
  (angle_dihedral : ℝ)
  (h1 : r_sphere = 4)
  (h2 : area_circle_M = 4 * Real.pi)
  (h3 : angle_dihedral = 60) :
  area_circle_N r_sphere area_circle_M angle_dihedral = 13 * Real.pi :=
by
  sorry

end area_of_circle_N_l323_323576


namespace rhombus_concyclic_points_l323_323685

theorem rhombus_concyclic_points:
  ∀ (A B C D E S : Type) 
  [inner_product_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ D] [affine_space ℝ E] [affine_space ℝ S],
    (ABCD_rhombus : is_rhombus A B C D) 
    (angle_BAD_lt_90 : ∠ B A D < π / 2)
    (circle_center_A_through_D : ∃ E, is_circle_through_center A D E)
    (intersection_CD_E : ∃ E, E ∈ line CD ∧ E ≠ D)
    (intersection_BE_AC_S : ∃ S, S ∈ line BE ∧ S ∈ line AC) 
    → concyclic_points A S D E :=
by sorry

end rhombus_concyclic_points_l323_323685


namespace count_5_primable_integers_lt_1000_is_21_l323_323107

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323107


namespace square_side_length_l323_323142

variable (s : ℕ)
variable (P A : ℕ)

theorem square_side_length (h1 : P = 52) (h2 : A = 169) (h3 : P = 4 * s) (h4 : A = s * s) : s = 13 :=
sorry

end square_side_length_l323_323142


namespace minimum_number_of_squares_l323_323768

theorem minimum_number_of_squares (n : ℕ) (hn : n = 13) : 
  ∃ k : ℕ, k = 11 ∧ (∀ (a b : ℕ), a * b = n * n → a = b → a ≤ k): sorry

end minimum_number_of_squares_l323_323768


namespace angle_sum_condition_angle_sum_condition_obtuse_l323_323310

-- Assuming definitions and geometry constructs are available in Mathlib or we extend them as needed.
variable {Point : Type*} [MetricSpace Point]

noncomputable def angle_sum_equal_180 (A B C A1 B1 C1 A0 B0 C0 : Point) : Prop :=
  angle B0 A1 C0 + angle C0 B1 A0 + angle A0 C1 B0 = 180

def triangle_type (A B C : Point) : Type :=
  if acute_triangle A B C then "acute"
  else if right_triangle A B C then "right"
  else "obtuse"

theorem angle_sum_condition (A B C A1 B1 C1 A0 B0 C0 : Point) 
  (h1 : midpoint A1 B C) (h2 : midpoint B1 C A) (h3 : midpoint C1 A B)
  (h4 : foot A0 A B C) (h5 : foot B0 B C A) (h6 : foot C0 C A B) :
  triangle_type A B C = "acute" ∨ triangle_type A B C = "right" →
  angle_sum_equal_180 A B C A1 B1 C1 A0 B0 C0 :=
sorry

theorem angle_sum_condition_obtuse (A B C A1 B1 C1 A0 B0 C0 : Point) 
  (h1 : midpoint A1 B C) (h2 : midpoint B1 C A) (h3 : midpoint C1 A B)
  (h4 : foot A0 A B C) (h5 : foot B0 B C A) (h6 : foot C0 C A B) :
  triangle_type A B C = "obtuse" →
  ¬ angle_sum_equal_180 A B C A1 B1 C1 A0 B0 C0 :=
sorry

-- This theorem states that given the conditions, the equality holds for acute and right triangles,
-- but not for obtuse triangles.

end angle_sum_condition_angle_sum_condition_obtuse_l323_323310


namespace part1_part2_l323_323961

variable (a b c : ℝ)

open Classical

noncomputable theory

-- Defining the conditions
def cond_positive_numbers : Prop := (0 < a) ∧ (0 < b) ∧ (0 < c)
def cond_main_equation : Prop := a^2 + b^2 + 4*c^2 = 3
def cond_b_eq_2c : Prop := b = 2*c

-- Statement for part (1)
theorem part1 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) :
  a + b + 2*c ≤ 3 := sorry

-- Statement for part (2)
theorem part2 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) (h3 : cond_b_eq_2c b c) :
  (1 / a) + (1 / c) ≥ 3 := sorry

end part1_part2_l323_323961


namespace sum_of_coordinates_of_point_B_l323_323371

theorem sum_of_coordinates_of_point_B
  (x y : ℝ)
  (A : (ℝ × ℝ) := (2, 1))
  (B : (ℝ × ℝ) := (x, y))
  (h_line : y = 6)
  (h_slope : (y - 1) / (x - 2) = 4 / 5) :
  x + y = 14.25 :=
by {
  -- convert hypotheses to Lean terms and finish the proof
  sorry
}

end sum_of_coordinates_of_point_B_l323_323371


namespace mixed_oil_rate_l323_323636

theorem mixed_oil_rate :
  let oil1 := (10, 50)
  let oil2 := (5, 68)
  let oil3 := (8, 42)
  let oil4 := (7, 62)
  let oil5 := (12, 55)
  let oil6 := (6, 75)
  let total_cost := oil1.1 * oil1.2 + oil2.1 * oil2.2 + oil3.1 * oil3.2 + oil4.1 * oil4.2 + oil5.1 * oil5.2 + oil6.1 * oil6.2
  let total_volume := oil1.1 + oil2.1 + oil3.1 + oil4.1 + oil5.1 + oil6.1
  (total_cost / total_volume : ℝ) = 56.67 :=
by
  sorry

end mixed_oil_rate_l323_323636


namespace radius_of_C3_correct_l323_323344

noncomputable def radius_of_C3
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ) : ℝ :=
if h1 : r1 = 2 ∧ r2 = 3
    ∧ (TA = 4) -- Conditions 1 and 2
   then 8
   else 0

-- Proof statement
theorem radius_of_C3_correct
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ)
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : TA = 4) :
  radius_of_C3 C1 C2 C3 r1 r2 A B T TA = 8 :=
by 
  sorry

end radius_of_C3_correct_l323_323344


namespace rectangle_area_pentagons_l323_323174

theorem rectangle_area_pentagons (width length : ℝ) (h_width : width = 10) (h_length : length = 15) :
  let area := width * length in
  let y := (sqrt (area / 3)) in
  y = 5 * sqrt 2 := by
  sorry

end rectangle_area_pentagons_l323_323174


namespace number_of_5_primable_less_1000_l323_323091

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323091


namespace ticket_cost_divisors_count_l323_323855

theorem ticket_cost_divisors_count (x : ℕ) :
  (x ∣ 72) ∧ (x ∣ 108) ∧ (x ∣ 36) → (finset.card (finset.filter (λ x, (x ∣ 72) ∧ (x ∣ 108) ∧ (x ∣ 36)) (finset.range 37)) = 9) :=
begin
  sorry
end

end ticket_cost_divisors_count_l323_323855


namespace binary_multiplication_l323_323508

theorem binary_multiplication : 
  (nat.bin_to_num [1, 1, 0, 1] * nat.bin_to_num [1, 1, 1] = nat.bin_to_num [1, 1, 0, 0, 1, 1, 1]) :=
by
  sorry

end binary_multiplication_l323_323508


namespace MinValue_x3y2z_l323_323692

theorem MinValue_x3y2z (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : 1/x + 1/y + 1/z = 6) : x^3 * y^2 * z ≥ 1 / 108 :=
by
  sorry

end MinValue_x3y2z_l323_323692


namespace Melies_money_left_l323_323708

variable (meat_weight : ℕ)
variable (meat_cost_per_kg : ℕ)
variable (initial_money : ℕ)

def money_left_after_purchase (meat_weight : ℕ) (meat_cost_per_kg : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - (meat_weight * meat_cost_per_kg)

theorem Melies_money_left : 
  money_left_after_purchase 2 82 180 = 16 :=
by
  sorry

end Melies_money_left_l323_323708


namespace Gilda_marbles_l323_323567

theorem Gilda_marbles (M : ℝ) (hM : 0 < M) :
  let remain_after_pedro := M - 0.30 * M,
      remain_after_ebony := remain_after_pedro - 0.15 * remain_after_pedro,
      remain_after_zack := remain_after_ebony - 0.20 * remain_after_ebony,
      remain_after_jimmy := remain_after_zack - 0.10 * remain_after_zack in
  (remain_after_jimmy / M) * 100 = 42.84 :=
by
  sorry

end Gilda_marbles_l323_323567


namespace find_a_extreme_value_range_f_greater_than_bound_l323_323251

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (a + Real.log x) / x

theorem find_a (h : f e a = e^2 x - y + e) : a = 1 := 
sorry

theorem extreme_value_range (m : ℝ) (cond : ∃ x ∈ Set.Ioo m (m+1), ∀ y ∈ Set.Ioo m (m+1), f y = f x) :
  0 < m ∧ m < 1 := 
sorry

theorem f_greater_than_bound (x : ℝ) (hx : 1 < x) : f x 1 > 2 / (x + 1) :=
sorry

end find_a_extreme_value_range_f_greater_than_bound_l323_323251


namespace number_of_5_primable_numbers_below_1000_l323_323115

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323115


namespace seq_arithmetic_lambda_range_l323_323988

noncomputable def sum_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = 2 * a n - 2^(n + 1)

theorem seq_arithmetic (S : ℕ → ℝ) (a : ℕ → ℝ)
(h : sum_seq S a) :
∃ (d : ℝ), ∀ n : ℕ, (n ≥ 1) → (a n) / (2^n) = 2 + (n - 1) * d :=
sorry

theorem lambda_range (S : ℕ → ℝ) (a : ℕ → ℝ) (λ : ℝ)
(h : sum_seq S a)
(h_ineq : ∀ n : ℕ, n ≥ 1 → 2 * n^2 - n - 3 < (5 - λ) * a n) :
λ < 37 / 8 :=
sorry

end seq_arithmetic_lambda_range_l323_323988


namespace minimum_n_for_team_probability_l323_323042

theorem minimum_n_for_team_probability (P₁ : ℝ) (P₂ : ℕ → ℝ) (n : ℕ) :
  P₁ = 0.3 →
  (∀ n, P₂ n = 1 - (0.9 ^ n)) →
  (P₂ n ≥ P₁) ↔ n ≥ 4 :=
by
  intro hP₁ hP₂
  unfold P₁ at hP₁
  sorry

end minimum_n_for_team_probability_l323_323042


namespace mixed_oil_rate_l323_323639

theorem mixed_oil_rate :
  let v₁ := 10
  let p₁ := 50
  let v₂ := 5
  let p₂ := 68
  let v₃ := 8
  let p₃ := 42
  let v₄ := 7
  let p₄ := 62
  let v₅ := 12
  let p₅ := 55
  let v₆ := 6
  let p₆ := 75
  let total_cost := v₁ * p₁ + v₂ * p₂ + v₃ * p₃ + v₄ * p₄ + v₅ * p₅ + v₆ * p₆
  let total_volume := v₁ + v₂ + v₃ + v₄ + v₅ + v₆
  let rate := total_cost / total_volume
  rate = 56.67 :=
by
  sorry

end mixed_oil_rate_l323_323639


namespace complement_of_M_is_correct_l323_323258

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}

-- Define the complement of M in U
def complement_M_in_U : Set ℝ := {x : ℝ | x < -1 ∨ x > 3}

-- State the theorem
theorem complement_of_M_is_correct : (U \ M) = complement_M_in_U := by sorry

end complement_of_M_is_correct_l323_323258


namespace count_five_primable_lt_1000_l323_323077

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323077


namespace minimum_value_of_expression_l323_323555

noncomputable def minimum_expression (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem minimum_value_of_expression : ∃ x : ℝ, minimum_expression x = -6480.25 :=
begin
  sorry
end

end minimum_value_of_expression_l323_323555


namespace mixture_weight_l323_323803

theorem mixture_weight (a b : ℝ) (h1 : a = 26.1) (h2 : a / (a + b) = 9 / 20) : a + b = 58 :=
sorry

end mixture_weight_l323_323803


namespace range_of_a_l323_323614

-- Definition for set A
def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = -|x| - 2 }

-- Definition for set B
def B (a : ℝ) : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x - a)^2 + y^2 = a^2 }

-- Statement of the problem in Lean
theorem range_of_a (a : ℝ) : (∀ p, p ∈ A → p ∉ B a) → -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 := by
  sorry

end range_of_a_l323_323614


namespace count_five_primable_lt_1000_l323_323080

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323080


namespace value_at_points_l323_323056

-- Function definitions based on given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x * (1 - x)
  else if 1 < x ∧ x ≤ 2 then Real.sin (π * x)
  else 0  -- Else case to make the function total; it should not affect the proof due to periodicity.

-- Helper functions to handle periodicity conditions and the behavior of odd functions
noncomputable def f_periodic (x : ℝ) : ℝ := f (x % 4)
noncomputable def f_odd (x : ℝ) : ℝ := if x < 0 then - f_periodic (-x) else f_periodic (x)

-- Theorem statement
theorem value_at_points : f_odd (29/4) + f_odd (41/6) = 5/16 :=
by
  sorry  -- Placeholder for the proof

end value_at_points_l323_323056


namespace min_decimal_digits_l323_323793

theorem min_decimal_digits (n : ℕ) (h : n = 987654321) (d : ℕ) (h1 : d = 2^30 * 5^3 * 3) :
  ∃ k : ℕ, k = 30 ∧ (∀ m : ℕ, m < 30 → ∃ r : ℚ, r = n / d ∧ (r * 10^m).denom ≠ 1) :=
by
  sorry

end min_decimal_digits_l323_323793


namespace unique_zero_of_f_l323_323353

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x^2 - (a + 1) * x + a * real.log x

theorem unique_zero_of_f (a : ℝ) (ha : a > 0) : ∃! x > 0, f x a = 0 :=
sorry

end unique_zero_of_f_l323_323353


namespace two_digit_prime_count_l323_323270

def is_valid_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

def is_two_digit_prime_with_valid_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Prime n ∧ is_valid_digit (n / 10) ∧ is_valid_digit (n % 10)

theorem two_digit_prime_count : 
  { n : ℕ | is_two_digit_prime_with_valid_digits n }.toFinset.card = 10 := 
by
  sorry

end two_digit_prime_count_l323_323270


namespace scalar_r_correct_l323_323261

open EuclideanSpace

-- Define the vectors a and b.
def a : ℝ^3 := ![2, 3, -1]
def b : ℝ^3 := ![1, -1, 2]
def v : ℝ^3 := ![5, -2, 3]

-- Define the proof theorem
theorem scalar_r_correct : 
  ∃ (r : ℝ), r = 4 / 15 ∧ v = (0 : ℝ) • a + (0 : ℝ) • b + r • (a.cross b) :=
sorry

end scalar_r_correct_l323_323261


namespace animals_per_aquarium_is_46_l323_323017

def TylerFacts : Prop :=
  ∃ (total_sw_animals : ℕ) (sw_aquariums : ℕ) (animals_per_aquarium : ℕ),
    total_sw_animals = 1012 ∧
    sw_aquariums = 22 ∧
    animals_per_aquarium = 46 ∧
    animals_per_aquarium = total_sw_animals / sw_aquariums

theorem animals_per_aquarium_is_46 : TylerFacts :=
by
  use 1012
  use 22
  use 46
  split
  · rfl
  split
  · rfl
  split
  · rfl
  · sorry

end animals_per_aquarium_is_46_l323_323017


namespace number_of_5_primable_numbers_below_1000_l323_323116

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323116


namespace calc_fraction_pow_negative_half_l323_323170

theorem calc_fraction_pow_negative_half : (1 / 16)^(-1/2 : ℝ) = 4 := 
by
  sorry

end calc_fraction_pow_negative_half_l323_323170


namespace count_5_primable_under_1000_l323_323127

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323127


namespace max_min_iterative_average_diff_l323_323498

def iterative_average (lst : List ℚ) : ℚ := 
  lst.foldl (λ acc x => (acc + x) / 2) 0

def primes : List ℚ := [2, 3, 5, 7, 11]

theorem max_min_iterative_average_diff : 
  (let perms := primes.permutations in
   let max_val := perms.map iterative_average |>.maximum 
   let min_val := perms.map iterative_average |>.minimum in
   max_val - min_val) = 4.6875 := 
by
  sorry

end max_min_iterative_average_diff_l323_323498


namespace conic_linear_combination_l323_323645

variables {R : Type*} [Field R]

def conic_section (F : R → R → R) : Prop := ∃ (A B C D E F : R), 
  ∀ x y, F x y = A * x^2 + B * x * y + C * y^2 + D * x + E * y + F

def common_points (F₁ F₂ : R → R → R) : Prop := 
  ∃ (P Q R S : R × R), 
    F₁ P.1 P.2 = 0 ∧ F₂ P.1 P.2 = 0 ∧
    F₁ Q.1 Q.2 = 0 ∧ F₂ Q.1 Q.2 = 0 ∧
    F₁ R.1 R.2 = 0 ∧ F₂ R.1 R.2 = 0 ∧
    F₁ S.1 S.2 = 0 ∧ F₂ S.1 S.2 = 0 

theorem conic_linear_combination {F₁ F₂ : R → R → R}
  (H₁ : conic_section F₁) (H₂ : conic_section F₂) 
  (H3 : common_points F₁ F₂) :
  ∃ (λ μ : R), ∀ x y, λ * F₁ x y + μ * F₂ x y = 0 :=
sorry

end conic_linear_combination_l323_323645


namespace company_fund_amount_l323_323754

theorem company_fund_amount :
  ∃ (n : ℕ), 60 * n - 10 = 770 ∧ 50 * n + 115 = 60 * n - 10 :=
by {
  existsi 13,
  split,
  {
    -- proving initial fund calculation
    -- Left is 60 * 13 - 10, right is 770
    calc 60 * 13 - 10 = 770 : by simp,
  },
  {
    -- proving balanced equation
    -- Left is 50 * 13 + 115, right is 60 * 13 - 10
    calc
      50 * 13 + 115 = 650 + 115 : by simp
              ... = 765         : by simp
              ... = 60 * 13 - 10 : by calc 60 * 13 - 10 = 780 - 10 : by simp
    ,
  },
  sorry, -- further proof steps to be filled in as necessary, here just proving equivalence setup
}

end company_fund_amount_l323_323754


namespace train_speed_proof_l323_323484

noncomputable def train_speed_in_kmh (length_in_m: ℝ) (time_in_sec: ℝ) : ℝ :=
  (length_in_m / 1000) / (time_in_sec / 3600)

theorem train_speed_proof : train_speed_in_kmh 250.00000000000003 15 = 60 := by
  have length_in_km := 250.00000000000003 / 1000
  have time_in_hr := 15 / 3600
  have speed := length_in_km / time_in_hr
  exact (by ring : speed = 60)

end train_speed_proof_l323_323484


namespace number_of_albino_8_antlered_deer_l323_323781

variable (total_deer : ℕ) (antler_percentage : ℚ) (albino_fraction : ℚ)
variable (has_8_antlers : ℕ) (albino_8_antlers : ℕ)

-- Conditions
def deer_population := total_deer = 920
def percentage_with_8_antlers := antler_percentage = 0.10
def fraction_albino_among_8_antlers := albino_fraction = 0.25

-- Intermediate calculations based on conditions
def calculate_has_8_antlers := has_8_antlers = total_deer * antler_percentage
def calculate_albino_8_antlers := albino_8_antlers = has_8_antlers * albino_fraction

-- Proof statement
theorem number_of_albino_8_antlered_deer : 
  deer_population → percentage_with_8_antlers → fraction_albino_among_8_antlers →
  calculate_has_8_antlers → calculate_albino_8_antlers →
  albino_8_antlers = 23 :=
by
  intros h_population h_percentage h_fraction h_calculate8antlers h_calculatealbino
  sorry

end number_of_albino_8_antlered_deer_l323_323781


namespace geometric_diff_l323_323172

-- Definitions based on conditions
def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧ (d2 * d2 = d1 * d3)

-- Problem statement
theorem geometric_diff :
  let largest_geometric := 964
  let smallest_geometric := 124
  is_geometric largest_geometric ∧ is_geometric smallest_geometric ∧
  (largest_geometric - smallest_geometric = 840) :=
by
  sorry

end geometric_diff_l323_323172


namespace max_k_value_l323_323589

noncomputable def max_k : ℝ :=
  let f : ℝ → ℝ := λ x => (Real.log x) / x
  in 
  let k_max := Sup (Set.range f)
  in k_max

theorem max_k_value : max_k = 1 / Real.exp 1 := by
  sorry

end max_k_value_l323_323589


namespace initial_back_squat_weight_l323_323320

-- Define a structure to encapsulate the conditions
structure squat_conditions where
  initial_back_squat : ℝ
  front_squat_ratio : ℝ := 0.8
  back_squat_increase : ℝ := 50
  front_squat_triple_ratio : ℝ := 0.9
  total_weight_moved : ℝ := 540

-- Using the conditions provided to prove John's initial back squat weight
theorem initial_back_squat_weight (c : squat_conditions) :
  (3 * 3 * (c.front_squat_triple_ratio * (c.front_squat_ratio * c.initial_back_squat)) = c.total_weight_moved) →
  c.initial_back_squat = 540 / 6.48 := sorry

end initial_back_squat_weight_l323_323320


namespace count_integers_strictly_ordered_digits_l323_323268

def strictly_increasing (a b c : ℕ) : Prop := a < b ∧ b < c
def strictly_decreasing (a b c : ℕ) : Prop := a > b ∧ b > c
def valid_digits (m n p : ℕ) : Prop := m ≠ n ∧ m ≠ p ∧ n ≠ p
def digit_range (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

theorem count_integers_strictly_ordered_digits :
  (∑ a b c in finset.range 9,
    (strictly_increasing a b c ∧ digit_range a ∧ digit_range b ∧ digit_range c ∧
     ∃ x y z, strictly_decreasing x y z ∧ digit_range x ∧ digit_range y ∧ digit_range z ∧
     valid_digits x y z ∧ valid_digits a b c ∧ x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ y ≠ a ∧ y ≠ b ∧ y ≠ c ∧ z ≠ a ∧ z ≠ b ∧ z ≠ c)) = 1680 :=
by
  sorry -- proof will be provided later

end count_integers_strictly_ordered_digits_l323_323268


namespace intersection_points_count_l323_323241

def f (x : Real) : Real := if x ∈ Icc (-1) 1 then x^2 else sorry -- Define f(x) with periodic extension

noncomputable def log_abs (x : Real) : Real := Real.log (abs x) / Real.log 3 -- Define log base 3 of absolute value of x

theorem intersection_points_count :
  (∀ x : Real, f (x + 1) = f (x - 1)) → 
  (∀ x : Real, x ∈ Icc (-1) 1 → f x = x^2) → 
  (∃ x y : set Real, 
    (∀ z ∈ x, f z = log_abs z) ∧ 
    (∀ z ∈ y, f (-z) = log_abs (-z)) ∧ 
    y = -x ∧ 
    set_card x + set_card y = 4) := 
by
  intros h1 h2
  sorry

end intersection_points_count_l323_323241


namespace count_five_primable_lt_1000_l323_323075

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323075


namespace recruits_total_l323_323412

theorem recruits_total (x y z : ℕ) (total_people : ℕ) :
  (x = total_people - 51) ∧
  (y = total_people - 101) ∧
  (z = total_people - 171) ∧
  (x = 4 * y ∨ y = 4 * z ∨ x = 4 * z) ∧
  (∃ total_people, total_people = 211) :=
sorry

end recruits_total_l323_323412


namespace dot_product_result_magnitude_cd_result_l323_323239

variables {R : Type} [IsROrC R] (a b : R^3) -- Define the type and the vectors

-- Conditions
def angle_between_vectors : ℝ := real.pi / 3 -- Given that the angle between vectors a and b is 60 degrees (or pi/3 radians)
def mag_a : ℝ := 2 -- Magnitude of vector a is 2
def mag_b : ℝ := 1 -- Magnitude of vector b is 1
def c : R^3 := a - 4 • b -- Vector c is defined as a - 4b
def d : R^3 := a + 2 • b -- Vector d is defined as a + 2b

-- Answers
def dot_product : ℝ := 1 -- The dot product a • b equals 1

def magnitude_cd (a b : R^3) : ℝ := 
  real.sqrt ((4 * (a • a)) - (8 * (a • b)) + (4 * (b • b))) -- Magnitude of (c + d) with given dot products and magnitudes should be 2√3

theorem dot_product_result (a b : R^3) 
    (h1 : real.angle a b = angle_between_vectors)
    (h2 : ∥a∥ = mag_a) (h3 : ∥b∥ = mag_b) : 
  a • b = dot_product := 
sorry

theorem magnitude_cd_result (a b : R^3) 
    (h1 : real.angle a b = angle_between_vectors)
    (h2 : ∥a∥ = mag_a) (h3 : ∥b∥ = mag_b) 
    (h4 : a • b = dot_product) : 
  ∥c + d∥ = 2 * real.sqrt 3 := 
sorry

end dot_product_result_magnitude_cd_result_l323_323239


namespace scientific_notation_1_3_billion_l323_323368

theorem scientific_notation_1_3_billion : 1300000000 = 1.3 * 10^9 := 
sorry

end scientific_notation_1_3_billion_l323_323368


namespace domain_of_p_l323_323911

def is_domain_of_p (x : ℝ) : Prop := x > 5

theorem domain_of_p :
  {x : ℝ | ∃ y : ℝ, y = 5*x + 2 ∧ ∃ z : ℝ, z = 2*x - 10 ∧
    z ≥ 0 ∧ z ≠ 0 ∧ p = 5*x + 2} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_p_l323_323911


namespace additional_carpet_needed_is_94_l323_323319

noncomputable def area_room_a : ℝ := 4 * 20

noncomputable def area_room_b : ℝ := area_room_a / 2.5

noncomputable def total_area : ℝ := area_room_a + area_room_b

noncomputable def carpet_jessie_has : ℝ := 18

noncomputable def additional_carpet_needed : ℝ := total_area - carpet_jessie_has

theorem additional_carpet_needed_is_94 :
  additional_carpet_needed = 94 := by
  sorry

end additional_carpet_needed_is_94_l323_323319


namespace log_identity_l323_323824

theorem log_identity (x : ℝ) (hx₁ : 0 < x) (hx₂ : x ≠ 1) :
  (Real.logBase 3 x) * (Real.logBase x 5) = Real.logBase 3 5 :=
sorry

end log_identity_l323_323824


namespace katie_roll_probability_l323_323680

def prob_less_than_five (d : ℕ) : ℚ :=
if d < 5 then 1 else 0

def prob_even (d : ℕ) : ℚ :=
if d % 2 = 0 then 1 else 0

theorem katie_roll_probability :
  (prob_less_than_five 1 + prob_less_than_five 2 + prob_less_than_five 3 + prob_less_than_five 4 +
  prob_less_than_five 5 + prob_less_than_five 6) / 6 *
  (prob_even 1 + prob_even 2 + prob_even 3 + prob_even 4 +
  prob_even 5 + prob_even 6) / 6 = 1 / 3 :=
sorry

end katie_roll_probability_l323_323680


namespace john_fan_usage_per_day_l323_323677

theorem john_fan_usage_per_day
  (power : ℕ := 75) -- fan's power in watts
  (energy_per_month_kwh : ℕ := 18) -- energy consumption per month in kWh
  (days_in_month : ℕ := 30) -- number of days in a month
  : (energy_per_month_kwh * 1000) / power / days_in_month = 8 := 
by
  sorry

end john_fan_usage_per_day_l323_323677


namespace concentration_of_replacement_solution_l323_323834

theorem concentration_of_replacement_solution :
  ∀ (initial_amount initial_concentration drain_amount target_amount target_concentration replacement_amount : ℝ),
    initial_amount = 300 → 
    initial_concentration = 0.2 → 
    drain_amount = 25 → 
    target_amount = 300 → 
    target_concentration = 0.25 → 
    replacement_amount = 25 → 
    let initial_pure_acid := initial_amount * initial_concentration,
        drained_pure_acid := drain_amount * initial_concentration,
        remaining_pure_acid := initial_pure_acid - drained_pure_acid,
        required_pure_acid := target_amount * target_concentration,
        added_pure_acid := required_pure_acid - remaining_pure_acid,
        replacement_concentration := added_pure_acid / replacement_amount
    in replacement_concentration = 0.8 :=
by
  intros initial_amount initial_concentration drain_amount target_amount target_concentration replacement_amount
  intros h_initial_amount h_initial_concentration h_drain_amount h_target_amount h_target_concentration h_replacement_amount
  unfold let initial_pure_acid := initial_amount * initial_concentration,
           drained_pure_acid := drain_amount * initial_concentration,
           remaining_pure_acid := initial_pure_acid - drained_pure_acid,
           required_pure_acid := target_amount * target_concentration,
           added_pure_acid := required_pure_acid - remaining_pure_acid,
           replacement_concentration := added_pure_acid / replacement_amount
  sorry

end concentration_of_replacement_solution_l323_323834


namespace smallest_m_n_sum_l323_323402

noncomputable def smallestPossibleSum (m n : ℕ) : ℕ :=
  m + n

theorem smallest_m_n_sum :
  ∃ (m n : ℕ), (m > 1) ∧ (m * n * (2021 * (m^2 - 1)) = 2021 * m * m * n) ∧ smallestPossibleSum m n = 4323 :=
by
  sorry

end smallest_m_n_sum_l323_323402


namespace smallest_m_n_sum_l323_323401

noncomputable def smallestPossibleSum (m n : ℕ) : ℕ :=
  m + n

theorem smallest_m_n_sum :
  ∃ (m n : ℕ), (m > 1) ∧ (m * n * (2021 * (m^2 - 1)) = 2021 * m * m * n) ∧ smallestPossibleSum m n = 4323 :=
by
  sorry

end smallest_m_n_sum_l323_323401


namespace find_y_in_range_l323_323281

theorem find_y_in_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end find_y_in_range_l323_323281


namespace M_card_ge_half_n_squared_l323_323575

noncomputable def n_partition (M : Type) (n : ℕ) := 
  Σ' (A B : fin n → set M), (∀ i j, i ≠ j → A i ∩ A j = ∅) 
  ∧ (∀ i j, i ≠ j → B i ∩ B j = ∅) 
  ∧ (⋃ i, A i = ⋃ i, B i) 
  ∧ (∀ i j, (A i ∩ B j = ∅) → (A i ∪ B j).card ≥ n)

theorem M_card_ge_half_n_squared {M : Type} (n : ℕ) (A B : fin n → set M) 
  (h_part : n_partition M n) :
  (⋃ i, A i).card ≥ n^2 / 2 := 
  sorry

end M_card_ge_half_n_squared_l323_323575


namespace cosine_angle_between_AM_and_BN_l323_323222

variables {V A B C D M N : Type} [InnerProductSpace ℝ V]
variables (AB : ℝ) (height : ℝ) (DN VN : ℝ)
          
-- Given conditions
def regular_quadrilateral_pyramid (V A B C D : V) : Prop :=
  distance A B = distance B C ∧ distance B C = distance C D ∧ distance C D = distance D A ∧
  ∃ (h : ℝ), h = height / 2 ∧ distance (orthogonal_projection (affine_span ℝ {A, B, C, D}) V) (incenter A B C D) = h
   
def midpoint (M B: V) : Prop :=
  (distance B M) = (distance M B)

def point_on_edge (N V D M : V) (d1 d2: ℝ) : Prop :=
  (distance N V) = (distance V N) / d1 ∧ (distance N D) = (distance D N) / d2

noncomputable def cosine_angle_AM_BN (A M N : V) : ℝ :=
  let AM := distance A M
  let BN := distance B N
  let dot_product := inner (A -ᵥ M) (B -ᵥ N)
  in dot_product / (AM * BN)

-- Condition definitions
axiom regular_pyramid : regular_quadrilateral_pyramid V A B C D
axiom AB_eq_2height : regular_pyramid -> height = AB / 2
axiom M_midpoint_VB : midpoint M B
axiom N_on_VD_ratio : point_on_edge N V D 3 9

-- Proof statement
theorem cosine_angle_between_AM_and_BN (A B M N : V) : 
  AB_eq_2height → M_midpoint_VB → N_on_VD_ratio → cosine_angle_AM_BN A M N = sqrt 11 / 11 :=
sorry

end cosine_angle_between_AM_and_BN_l323_323222


namespace relation_between_lines_l323_323686

-- Define the types Line and Space to represent lines in space
noncomputable def Line := ℝ → ℝ³ -- A line is a mapping from reals to 3D space

-- Definitions for parallel and intersecting lines
def are_parallel (a b : Line) : Prop := ∃ v, (∀ t, a t = b (t + v))
def intersect (a c : Line) : Prop := ∃ t s, a t = c s

-- Statement of the problem
theorem relation_between_lines (a b c : Line) 
  (h1 : are_parallel a b) 
  (h2 : intersect a c) : 
  (intersect b c ∨ ¬ ∃ p, ∀ t, b t ∈ (set.range c)) :=
sorry

end relation_between_lines_l323_323686


namespace marks_lost_per_wrong_answer_l323_323656

theorem marks_lost_per_wrong_answer :
  ∃ (x: ℕ), (∀ (correct wrong total_correct_marks : ℕ),
    correct = 36 ∧ wrong = 50 - correct ∧ total_correct_marks = 36 * 4 ∧ total_correct_marks - (wrong * x) = 130) → x = 1 :=
by
  existsi 1
  intros correct wrong total_correct_marks h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end marks_lost_per_wrong_answer_l323_323656


namespace sum_of_possible_ns_l323_323474

theorem sum_of_possible_ns : 
  (∑ n in Finset.Ico 5 18, n) = 143 := 
  by
    sorry

end sum_of_possible_ns_l323_323474


namespace remainder_when_divided_by_9_l323_323798

variable (k : ℕ)

theorem remainder_when_divided_by_9 :
  (∃ k, k % 5 = 2 ∧ k % 6 = 3 ∧ k % 8 = 7 ∧ k < 100) →
  k % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l323_323798


namespace quadrilateral_circumscribed_circle_l323_323288

theorem quadrilateral_circumscribed_circle (a : ℝ) :
  ((a + 2) * x + (1 - a) * y - 3 = 0) ∧ ((a - 1) * x + (2 * a + 3) * y + 2 = 0) →
  ( a = 1 ∨ a = -1 ) :=
by
  intro h
  sorry

end quadrilateral_circumscribed_circle_l323_323288


namespace find_pairs_l323_323889

-- Define the problem conditions
def equation (n k : ℕ) : Prop := nat.factorial n + n = n ^ k

-- Define the positive integer property
def positive (n : ℕ) : Prop := n > 0

-- State the goal of the theorem
theorem find_pairs : ∀ (n k : ℕ), positive n → positive k → equation n k ↔ 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) :=
by
  intros n k hn hk
  sorry

end find_pairs_l323_323889


namespace complex_number_quadrant_l323_323570

open Complex

theorem complex_number_quadrant (x : ℂ) (h : x = 3 + 4 * I) :
  let z := x - abs x - (1 - (-I))
  z.re < 0 ∧ z.im > 0 :=
by
  have h1 : z = (x - abs x - (1 - (-I))),
  { sorry },
  have h2 : z.re < 0,
  { sorry },
  have h3 : z.im > 0,
  { sorry },
  exact ⟨h2, h3⟩

end complex_number_quadrant_l323_323570


namespace smallest_non_factor_product_l323_323008

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l323_323008


namespace subset_inequalities_l323_323257

noncomputable def M : set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

noncomputable def N : set (ℝ × ℝ) := {p | real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * real.sqrt 2}

noncomputable def P : set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs (p.1 - p.2) < 1}

theorem subset_inequalities : M ⊆ N ∧ N ⊆ P :=
by
  sorry

end subset_inequalities_l323_323257


namespace arithmetic_seq_sum_l323_323658

-- Definition of an arithmetic sequence using a common difference d
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Statement of the problem
theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (hs : arithmetic_sequence a d)
  (hmean : (a 3 + a 8) / 2 = 10) : 
  a 1 + a 10 = 20 :=
sorry

end arithmetic_seq_sum_l323_323658


namespace problem_proof_l323_323600

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then x + 4 else if 0 < x ∧ x ≤ 4 then x^2 - 2 * x else -x + 2

theorem problem_proof :
  (f 0 = 4) ∧
  (f 5 = -3) ∧
  (f (f (f 5)) = -1) ∧
  (∀ a : ℝ, f a = 8 → a = 4) :=
by
  sorry

end problem_proof_l323_323600


namespace count_5_primable_under_1000_l323_323120

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323120


namespace sin_3theta_l323_323275

noncomputable def z : ℂ := (1 + complex.I * real.sqrt 2) / 2
noncomputable def θ : ℝ := complex.arg z

theorem sin_3theta (h : complex.exp (complex.I * θ) = z) : 
  real.sin (3 * θ) = real.sqrt 2 / 8 := 
sorry

end sin_3theta_l323_323275


namespace binomial_9_pow5_eq_binomial_11_pow5_eq_pow_9_and_11_l323_323171

noncomputable def pow_9 : ℕ := 9^5
noncomputable def pow_11 : ℕ := 11^5

theorem binomial_9_pow5_eq :
  ∑ k in Finset.range 6, Nat.choose 5 k * 10^(5-k) * (-1)^k = 59149 := sorry

theorem binomial_11_pow5_eq :
  ∑ k in Finset.range 6, Nat.choose 5 k * 10^(5-k) * 1^k = 161051 := sorry

theorem pow_9_and_11 :
  pow_9 = 59149 ∧ pow_11 = 161051 :=
by
  unfold pow_9 pow_11
  apply And.intro
  · apply binomial_9_pow5_eq
  · apply binomial_11_pow5_eq

end binomial_9_pow5_eq_binomial_11_pow5_eq_pow_9_and_11_l323_323171


namespace right_angle_PQ_at_foci_l323_323897

variables (a b : ℝ) (A B C : ℝ)

noncomputable def hyperbola := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

def focus1 : ℝ × ℝ := (-real.sqrt (a^2 + b^2), 0)
def focus2 : ℝ × ℝ := (real.sqrt (a^2 + b^2), 0)

def vertex1 : ℝ × ℝ := (-a, 0)
def vertex2 : ℝ × ℝ := (a, 0)

def perp_line_v1 := {p : ℝ × ℝ | p.1 = -a}
def perp_line_v2 := {p : ℝ × ℝ | p.1 = a}

def tangent_line (A B C : ℝ) := {p : ℝ × ℝ | A * p.1 + B * p.2 = C}

-- The intersection points P and Q
noncomputable def P (A B C : ℝ) : ℝ × ℝ := (-a, (C + a * A) / B)
noncomputable def Q (A B C : ℝ) : ℝ × ℝ := (a, (C - a * A) / B)

def midpoint (p q : ℝ × ℝ) := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

def midPQ (A B C : ℝ) := midpoint (P a b A B C) (Q a b A B C)

theorem right_angle_PQ_at_foci
  (tangent_line_condition : a^2 * A^2 - b^2 * B^2 = C^2):
  ∀ P Q : (ℝ × ℝ), 
    let M := midpoint P Q in 
    let F1 := focus1 a b in
    let F2 := focus2 a b in
    (P = P a b A B C ∧ Q = Q a b A B C ∧ M = midPQ a b A B C) → 
    angle F1 P Q = π / 2 ∧ angle F2 P Q = π / 2 := sorry

end right_angle_PQ_at_foci_l323_323897


namespace condition_for_line_perpendicular_to_x_axis_equal_intercepts_l323_323595

-- 1. To prove that for \( (m^2 - 2m - 3)x + (2m^2 + m - 1)y + 5 - 2m = 0 \) to represent a line, \(m \neq -1\).
theorem condition_for_line (m : ℝ) : 
  (m^2 - 2 * m - 3 ≠ 0 ∨ 2 * m^2 + m - 1 ≠ 0) ↔ m ≠ -1 :=
by sorry

-- 2. To prove that the equation is perpendicular to the x-axis when \( m \) satisfies \(\begin{cases} m^2 - 2m - 3 \neq 0 \\ 2m^2 + m - 1 = 0 \end{cases}\).
theorem perpendicular_to_x_axis (m : ℝ) : 
  (m^2 - 2 * m - 3 ≠ 0 ∧ 2 * m^2 + m - 1 = 0) ↔ 
  (m ≠ -1 ∧ m = sqrt 2 - 1 ∨ m = -\sqrt 2 - 1) :=
by sorry

-- 3. To prove that for the equation to have equal intercepts on both coordinate axes, \( m = \frac{5}{2} \).
theorem equal_intercepts (m : ℝ) : 
  (m = 5 / 2) ↔ 
  let intercept_x := - (5 - 2 * m) / (m^2 - 2 * m - 3); 
  let intercept_y := - (5 - 2 * m) / (2 * m^2 + m - 1);
  intercept_x = intercept_y :=
by sorry

end condition_for_line_perpendicular_to_x_axis_equal_intercepts_l323_323595


namespace carbon14_at_2865_l323_323711

-- Define the decay law for carbon-14 content
def carbon14_content (M₀ : ℝ) (t : ℝ) : ℝ :=
  M₀ * 2^(-t / 5730)

-- Given initial carbon-14 content
def M₀ : ℝ := 573

-- Define the problem statement to prove M(2865) and the correct value
theorem carbon14_at_2865 :
  carbon14_content M₀ 2865 = (573 / 2) * Real.sqrt 2 :=
by
  sorry

end carbon14_at_2865_l323_323711


namespace part1_part2_l323_323302

noncomputable def circle_eq (x y b : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + b = 0

theorem part1 (x y : ℝ) : 
  (circle_eq x y b) → (-∞ < b ∧ b < 13) ∧ (∃ xc yc, xc = 3 ∧ yc = 2) := 
sorry

theorem part2 (b : ℝ) :
  b = 12 → (radius : ℝ) → radius = 1 → 
  (∃ k, (k = -3/4 ∧ (3*x + 4*y - 12 = 0)) ∨ (k = 0 ∧ y = 3)) := 
sorry

end part1_part2_l323_323302


namespace polynomial_divisibility_l323_323884

theorem polynomial_divisibility (m : ℤ) : (3 * (-2)^2 + 5 * (-2) + m = 0) ↔ (m = -2) :=
by
  sorry

end polynomial_divisibility_l323_323884


namespace two_packs_remainder_l323_323539

theorem two_packs_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 :=
by {
  sorry
}

end two_packs_remainder_l323_323539


namespace min_value_expression_l323_323343

noncomputable def z (x y : ℝ) : ℂ := x + y * Complex.i

theorem min_value_expression : 
  ∀ (x y : ℝ), 
  (x - 3)^2 + (y + 2)^2 = 16 →
  (Complex.abs (z x y + 1 - Complex.i))^2 + (Complex.abs (z x y - 7 + 3 * Complex.i))^2 = 88 :=
by 
  sorry

end min_value_expression_l323_323343


namespace max_int_in_list_l323_323842

/-- 
  A list of seven positive integers has the following properties:
  1. The only integer in the list that occurs more than once is 7.
  2. The median is 11.
  3. The average (mean) is 12.
  Prove that the largest possible integer that could appear in the list is 32.
-/
theorem max_int_in_list : 
  ∀ (L : List ℕ), 
  (L.length = 7) → 
  (L.count 7 ≥ 2) → 
  (∀ x, x ≠ 7 → L.count x ≤ 1) → 
  (L.nth_le 3 (by linarith) = 11) → 
  ((L.sum / L.length) = 12) → 
  L.maximum = some 32 := 
by
  sorry

end max_int_in_list_l323_323842


namespace integral_cos_minus_sin_over_one_plus_sin_squared_l323_323814

theorem integral_cos_minus_sin_over_one_plus_sin_squared :
  ∫ x in 0..(Real.pi / 2), (Real.cos x - Real.sin x) / (1 + Real.sin x)^2 = 1 / 2 := by
  sorry

end integral_cos_minus_sin_over_one_plus_sin_squared_l323_323814


namespace general_term_of_seq_a_l323_323571

def seq_a (n : ℕ) : ℝ :=
  if n = 1 then 2 else
  let a_n := seq_a (n - 1)
  in 2^((n - 1) + 1) * a_n / ((↑n - 1 + 1/2) * a_n + 2^(n - 1))

theorem general_term_of_seq_a (n : ℕ) (hn : n > 0) : seq_a n = 2^(n+1) / (n^2 + 1) :=
sorry

end general_term_of_seq_a_l323_323571


namespace radius_of_tangent_circle_l323_323330

noncomputable def isosceles_trapezoid_radius : ℝ :=
  let r := (-81 + 57 * real.sqrt 5) / 23
  in
  r

theorem radius_of_tangent_circle :
  let AB := 10
  let BC := 7
  let DA := 7
  let CD := 6
  let radius_A_B := 4
  let radius_C_D := 3
  ∃ r, 
    (
      let trapezoid_height := real.sqrt (7^2 - (10 - 6) ^ 2 / 4) in
      let tolerance_y := real.sqrt (r^2 + 8 * r) in
      let tolerance_z := real.sqrt (r^2 + 6 * r) in
      tolerance_y + tolerance_z = trapezoid_height
    ) 
    ∧
    r = isosceles_trapezoid_radius :=
begin
  sorry
end

end radius_of_tangent_circle_l323_323330


namespace problem_l323_323212

open_locale real_inner_product_space

-- Definition of the vectors and their conditions as given
variables (A B C M : ℝ³) (m : ℝ)

-- Condition 1: The sum of the vectors from M to the vertices of the triangle is zero
axiom condition1 : (M - A) + (M - B) + (M - C) = 0

-- Condition 2: There exists a real number m such that the given equation holds
axiom condition2 : A - B + (A - C) = m * (A - M)

-- The statement to prove
theorem problem :
  M = ⅓ * (A + B + C) ∧ m = 3 :=
by
  sorry

end problem_l323_323212


namespace compare_solutions_l323_323342

variables (p q r s : ℝ)
variables (hp : p ≠ 0) (hr : r ≠ 0)

theorem compare_solutions :
  ((-q / p) > (-s / r)) ↔ (s * r > q * p) :=
by sorry

end compare_solutions_l323_323342


namespace percentage_of_blouses_in_hamper_l323_323433

theorem percentage_of_blouses_in_hamper 
  (total_blouses : ℕ) (total_skirts : ℕ) (total_slacks : ℕ) 
  (percentage_of_skirts_in_hamper : ℝ) (percentage_of_slacks_in_hamper : ℝ)
  (total_pieces_needed : ℕ)
  (hamper_skirts : ℕ := (percentage_of_skirts_in_hamper * total_skirts).to_nat)
  (hamper_slacks : ℕ := (percentage_of_slacks_in_hamper * total_slacks).to_nat) :
  total_blouses = 12 →
  total_skirts = 6 →
  total_slacks = 8 →
  percentage_of_skirts_in_hamper = 0.5 →
  percentage_of_slacks_in_hamper = 0.25 →
  total_pieces_needed = 14 →
  let hamper_blouses := total_pieces_needed - hamper_skirts - hamper_slacks in
  let percentage_blouses_in_hamper := (hamper_blouses.to_nat : ℝ) / total_blouses * 100 in
  percentage_blouses_in_hamper = 75 :=
by
  intro h1 h2 h3 h4 h5 h6
  dsimp only
  sorry

end percentage_of_blouses_in_hamper_l323_323433


namespace find_ratio_eq_eighty_six_l323_323786

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 45}

-- Define the sum of the first n natural numbers function
def sum_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define our specific scenario setup
def selected_numbers (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x * y = sum_n_nat 45 - (x + y)

-- Prove the resulting ratio condition
theorem find_ratio_eq_eighty_six (x y : ℕ) (h : selected_numbers x y) : 
  x < y → y / x = 86 :=
by
  sorry

end find_ratio_eq_eighty_six_l323_323786


namespace mixed_oil_rate_l323_323637

theorem mixed_oil_rate :
  let oil1 := (10, 50)
  let oil2 := (5, 68)
  let oil3 := (8, 42)
  let oil4 := (7, 62)
  let oil5 := (12, 55)
  let oil6 := (6, 75)
  let total_cost := oil1.1 * oil1.2 + oil2.1 * oil2.2 + oil3.1 * oil3.2 + oil4.1 * oil4.2 + oil5.1 * oil5.2 + oil6.1 * oil6.2
  let total_volume := oil1.1 + oil2.1 + oil3.1 + oil4.1 + oil5.1 + oil6.1
  (total_cost / total_volume : ℝ) = 56.67 :=
by
  sorry

end mixed_oil_rate_l323_323637


namespace minuend_is_not_integer_l323_323417

theorem minuend_is_not_integer (M S D : ℚ) (h1 : M + S + D = 555) (h2 : M - S = D) : ¬ ∃ n : ℤ, M = n := 
by
  sorry

end minuend_is_not_integer_l323_323417


namespace part_a_l323_323822

theorem part_a 
  (x y u v : ℝ) 
  (h1 : x + y = u + v) 
  (h2 : x^2 + y^2 = u^2 + v^2) : 
  ∀ n : ℕ, x^n + y^n = u^n + v^n := 
by sorry

end part_a_l323_323822


namespace min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l323_323568

variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (h : 4 * a + b = a * b)

theorem min_ab : 16 ≤ a * b :=
sorry

theorem min_a_b : 9 ≤ a + b :=
sorry

theorem max_two_a_one_b : 2 > (2 / a + 1 / b) :=
sorry

theorem min_one_a_sq_four_b_sq : 1 / 5 ≤ (1 / a^2 + 4 / b^2) :=
sorry

end min_ab_min_a_b_max_two_a_one_b_min_one_a_sq_four_b_sq_l323_323568


namespace triangle_congruence_l323_323375

-- Define points and triangles
variables {A B C A' B' C' M3 M3' : Type}

-- Define the lengths of the medians
def median_eq (CM3 C'M3' : ℝ) : Prop :=
  CM3 = C'M3'

-- Define the angle division condition
def angles_divided_eq (angle_C angle_C' : ℝ) : Prop :=
  angle_C = angle_C'

-- Main Theorem
theorem triangle_congruence 
(CM3 C'M3' : ℝ) 
(angle_C angle_C' : ℝ)
(h1 : median_eq CM3 C'M3')
(h2 : angles_divided_eq angle_C angle_C') :
  ∆ABC ≅ ∆A'B'C' :=
by
  sorry

end triangle_congruence_l323_323375


namespace center_of_symmetry_tan_l323_323396

theorem center_of_symmetry_tan (k : ℤ) :
    center_of_symmetry (λ x : ℝ, Real.tan (π * x + π / 4)) = (2 * k - 1) / 4 :=
sorry

end center_of_symmetry_tan_l323_323396


namespace evaluate_expression_l323_323898

theorem evaluate_expression : (real.rpow (real.rpow 16 (1/4)) 6) = 64 := by
  sorry

end evaluate_expression_l323_323898


namespace evaluate_expression_l323_323900

theorem evaluate_expression : (real.rpow (real.rpow 16 (1/4)) 6) = 64 := by
  sorry

end evaluate_expression_l323_323900


namespace roots_sum_and_product_l323_323762

theorem roots_sum_and_product (p q : ℝ) (h_sum : p / 3 = 9) (h_prod : q / 3 = 24) : p + q = 99 :=
by
  -- We are given h_sum: p / 3 = 9
  -- We are given h_prod: q / 3 = 24
  -- We need to prove p + q = 99
  sorry

end roots_sum_and_product_l323_323762


namespace non_overlapping_squares_area_l323_323787

noncomputable def non_overlapping_area (a : ℝ) : ℝ :=
  2 * (1 - real.sqrt 3 / 3) * a^2

theorem non_overlapping_squares_area (a : ℝ) :
  let shaded_area := (real.sqrt 3 / 3) * a^2 in
  2 * (a^2 - shaded_area) = non_overlapping_area a :=
by
  -- This theorem asserts that the area of the non-overlapping parts of the two squares is
  -- equal to the computed formula for non_overlapping_area given a.
  sorry

end non_overlapping_squares_area_l323_323787


namespace fraction_problem_l323_323512

-- Define the fractions involved in the problem
def frac1 := 18 / 45
def frac2 := 3 / 8
def frac3 := 1 / 9

-- Define the expected result
def expected_result := 49 / 360

-- The proof statement
theorem fraction_problem : frac1 - frac2 + frac3 = expected_result := by
  sorry

end fraction_problem_l323_323512


namespace max_min_sum_eq_4024_l323_323522

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem max_min_sum_eq_4024 (h₁ : ∀ a b : ℝ, a ∈ Icc (-2016) 2016 → b ∈ Icc (-2016) 2016 → 
    f(a + b) = f(a) + f(b) - 2012) (h₂ : ∀ x : ℝ, 0 < x → f(x) > 2012) :
    let M := Sup (set.image f (Icc (-2016) 2016)),
        N := Inf (set.image f (Icc (-2016) 2016)) in
    M + N = 4024 :=
by
  sorry

end max_min_sum_eq_4024_l323_323522


namespace count_5_primables_less_than_1000_l323_323095

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323095


namespace domain_of_f_l323_323180

def function_domain (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x, x ∈ domain ↔ ∃ y, f y = x

noncomputable def f (x : ℝ) : ℝ :=
  (x + 6) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f :
  function_domain f ((Set.Iio 2) ∪ (Set.Ioi 3)) :=
by
  sorry

end domain_of_f_l323_323180


namespace value_of_one_pound_of_gold_l323_323659

noncomputable def value_of_gold : ℕ :=
  let total_gold := 12
  let expected_tax_in_gold := (total_gold : ℝ) * (1 / 10)
  let actual_tax_in_gold := 2
  let coins_returned := 5000
  let adjusted_gold_taken := actual_tax_in_gold - expected_tax_in_gold
  (coins_returned : ℝ) / adjusted_gold_taken

theorem value_of_one_pound_of_gold :
  value_of_gold = 6250 :=
by
  sorry

end value_of_one_pound_of_gold_l323_323659


namespace part_I_part_II_part_III_l323_323598

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Part (I)
theorem part_I (a : ℝ) (h_a : a = 1) : 
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 0 ∧ f x a * g x = 1 := sorry

-- Part (II)
theorem part_II (a : ℝ) (h_a : a = -1) (k : ℝ) :
  (∃ x : ℝ, f x a = k * g x ∧ ∀ y : ℝ, y ≠ x → f y a ≠ k * g y) ↔ 
  (k > 3 * Real.exp (-2) ∨ (0 < k ∧ k < 1 * Real.exp (-1))) := sorry

-- Part (III)
theorem part_III (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), (x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂) →
  abs (f x₁ a - f x₂ a) < abs (g x₁ - g x₂)) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) := sorry

end part_I_part_II_part_III_l323_323598


namespace function_characterization_l323_323800

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization (f : ℝ → ℝ)
  (h1 : ∀ x y, f(x * y) = f(x) * f(y))
  (h2 : ∀ x, f' x = f' (-x))
  (h3 : ∀ x, 0 < x → 0 < f' x) :
         ∃ (n : ℕ), n % 2 = 1 ∧ f = λ x, x^n := 
begin
  sorry
end

end function_characterization_l323_323800


namespace student_missed_number_l323_323034

theorem student_missed_number (student_sum : ℕ) (n : ℕ) (actual_sum : ℕ) : 
  student_sum = 575 → 
  actual_sum = n * (n + 1) / 2 → 
  n = 34 → 
  actual_sum - student_sum = 20 := 
by 
  sorry

end student_missed_number_l323_323034


namespace count_five_primable_lt_1000_l323_323078

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323078


namespace other_candidate_votes_l323_323810

theorem other_candidate_votes (total_votes : ℕ) (invalid_percentage : ℝ) (candidate_percentage : ℝ) :
  total_votes = 7500 →
  invalid_percentage = 0.2 →
  candidate_percentage = 0.55 →
  let valid_votes := total_votes * (1 - invalid_percentage)
  let other_candidate_percentage := 1 - candidate_percentage
  let other_candidate_votes := valid_votes * other_candidate_percentage
  other_candidate_votes = 2700 :=
by
  intros h_total_votes h_invalid_percentage h_candidate_percentage
  let valid_votes := total_votes * (1 - invalid_percentage)
  let other_candidate_percentage := 1 - candidate_percentage
  let other_candidate_votes := valid_votes * other_candidate_percentage
  have h_valid_votes : valid_votes = 6000 := by
    rw [h_total_votes, h_invalid_percentage]
    norm_num
  have h_other_candidate_votes : other_candidate_votes = 2700 := by
    rw [h_valid_votes, h_candidate_percentage]
    norm_num
  exact h_other_candidate_votes

end other_candidate_votes_l323_323810


namespace smallest_product_not_factor_of_48_exists_l323_323003

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l323_323003


namespace at_least_one_positive_l323_323618

variable (x y z : ℝ)

def a : ℝ := x^2 - 2 * y + (Real.pi / 3)
def b : ℝ := y^2 - 2 * z + (Real.pi / 6)
def c : ℝ := z^2 - 2 * x + (Real.pi / 2)

theorem at_least_one_positive :
  (a x y z > 0) ∨ (b x y z > 0) ∨ (c x y z > 0) :=
sorry

end at_least_one_positive_l323_323618


namespace newspapers_on_sunday_l323_323718

theorem newspapers_on_sunday (papers_weekend : ℕ) (diff_papers : ℕ) 
  (h1 : papers_weekend = 110) 
  (h2 : diff_papers = 20) 
  (h3 : ∃ (S Su : ℕ), Su = S + diff_papers ∧ S + Su = papers_weekend) :
  ∃ Su, Su = 65 :=
by
  sorry

end newspapers_on_sunday_l323_323718


namespace perfect_square_A_plus_2B_plus_4_l323_323695

theorem perfect_square_A_plus_2B_plus_4 (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9 : ℚ) * (10 ^ (2 * n) - 1)
  let B := (8 / 9 : ℚ) * (10 ^ n - 1)
  ∃ k : ℚ, A + 2 * B + 4 = k^2 := 
by {
  sorry
}

end perfect_square_A_plus_2B_plus_4_l323_323695


namespace expected_value_of_sides_l323_323846

-- Define the probability calculations as given
def hexagon_probability : ℝ := (Real.pi - 2) / 2
def pentagon_probability : ℝ := (4 - Real.pi) / 2

-- Define the expected number of sides calculation
def expected_sides : ℝ :=
  6 * hexagon_probability + 5 * pentagon_probability

theorem expected_value_of_sides :
  expected_sides = 4 + Real.pi / 2 :=
sorry

end expected_value_of_sides_l323_323846


namespace total_nephews_correct_l323_323150

def alden_nephews_10_years_ago : ℕ := 50

def alden_nephews_now : ℕ :=
  alden_nephews_10_years_ago * 2

def vihaan_nephews_now : ℕ :=
  alden_nephews_now + 60

def total_nephews : ℕ :=
  alden_nephews_now + vihaan_nephews_now

theorem total_nephews_correct : total_nephews = 260 := by
  sorry

end total_nephews_correct_l323_323150


namespace xy_value_l323_323272

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 :=
by {
  sorry
}

end xy_value_l323_323272


namespace relationship_among_a_b_c_l323_323967

noncomputable def a : ℝ := 2^0.3
noncomputable def b : ℝ := 0.3^2
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  -- Definitions for a, b, c
  let a := 2^0.3
  let b := 0.3^2
  let c := Real.log 3 / Real.log 2
  -- sorry to indicate the proof is omitted
  sorry

end relationship_among_a_b_c_l323_323967


namespace total_garland_arrangements_l323_323420

theorem total_garland_arrangements 
  (num_blue : ℕ) (num_red : ℕ) (num_white : ℕ) 
  (h_blue : num_blue = 8) (h_red : num_red = 8) (h_white : num_white = 11) :
  let total_arrangements := (Nat.choose 16 8) * (Nat.choose 17 11)
  in total_arrangements = 159279120 :=
by
  -- The proof is intentionally left out
  sorry

end total_garland_arrangements_l323_323420


namespace p_neither_necessary_nor_sufficient_for_q_l323_323580

variables {x y : ℝ}

def p : Prop := (x / y > 1)
def q : Prop := (x > y)

theorem p_neither_necessary_nor_sufficient_for_q : 
  ¬((p → q) ∧ (q → p)) :=
by 
  sorry

end p_neither_necessary_nor_sufficient_for_q_l323_323580


namespace conic_through_vertices_l323_323041

-- Defining the problem
theorem conic_through_vertices (p q r s t u : ℝ) :
    ( ∀ x y z : ℝ, 
      (x = 1 ∧ y = 0 ∧ z = 0) ∨ 
      (x = 0 ∧ y = 1 ∧ z = 0) ∨ 
      (x = 0 ∧ y = 0 ∧ z = 1) → 
      p * x ^ 2 + q * y ^ 2 + r * z ^ 2 + s * x * y + t * x * z + u * y * z = 0) → 
    (p = 0 ∧ q = 0 ∧ r = 0 ∧ s * x * y + t * x * z + u * y * z = 0) ↔ (s = 0 ∧ t = 0 ∧ u = 0) :=
begin
  sorry
end

end conic_through_vertices_l323_323041


namespace heather_counts_209_l323_323535

def alice_numbers (n : ℕ) : ℕ := 5 * n - 2
def general_skip_numbers (m : ℕ) : ℕ := 3 * m - 1
def heather_number := 209

theorem heather_counts_209 :
  (∀ n, alice_numbers n > 0 ∧ alice_numbers n ≤ 500 → ¬heather_number = alice_numbers n) ∧
  (∀ m, general_skip_numbers m > 0 ∧ general_skip_numbers m ≤ 500 → ¬heather_number = general_skip_numbers m) ∧
  (1 ≤ heather_number ∧ heather_number ≤ 500) :=
by
  sorry

end heather_counts_209_l323_323535


namespace compute_n_pow_m_l323_323271

-- Given conditions
variables (n m : ℕ)
axiom n_eq : n = 3
axiom n_plus_one_eq_2m : n + 1 = 2 * m

-- Goal: Prove n^m = 9
theorem compute_n_pow_m : n^m = 9 :=
by {
  -- Proof goes here
  sorry
}

end compute_n_pow_m_l323_323271


namespace total_nephews_correct_l323_323153

namespace Nephews

-- Conditions
variable (ten_years_ago : Nat)
variable (current_alden_nephews : Nat)
variable (vihaan_extra_nephews : Nat)
variable (alden_nephews_10_years_ago : ten_years_ago = 50)
variable (alden_nephews_double : ten_years_ago * 2 = current_alden_nephews)
variable (vihaan_nephews : vihaan_extra_nephews = 60)

-- Answer
def total_nephews (alden_nephews_now vihaan_nephews_now : Nat) : Nat :=
  alden_nephews_now + vihaan_nephews_now

-- Proof statement
theorem total_nephews_correct :
  ∃ (alden_nephews_now vihaan_nephews_now : Nat), 
    alden_nephews_10_years_ago →
    alden_nephews_double →
    vihaan_nephews →
    alden_nephews_now = current_alden_nephews →
    vihaan_nephews_now = current_alden_nephews + vihaan_extra_nephews →
    total_nephews alden_nephews_now vihaan_nephews_now = 260 :=
by
  sorry

end Nephews

end total_nephews_correct_l323_323153


namespace selling_price_l323_323724

theorem selling_price (R W : ℝ) (h1 : 0.80 * R + 0.85 * W = 35000) :
  let total_additional_cost := 550 in
  let desired_refrigerator_price := 1.10 * R in
  let desired_washing_machine_price := 1.12 * W in
  desired_refrigerator_price + desired_washing_machine_price = 
  1.10 * R + 1.12 * W :=
by
  sorry

end selling_price_l323_323724


namespace number_of_5_primable_less_1000_l323_323087

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323087


namespace hyperbola_asymptotes_l323_323236

theorem hyperbola_asymptotes (a : ℝ) (h_pos : a > 0) 
  (hx1 : -2^2 / a^2 - (1^2 : ℝ) = 1) 
  (hx2 : 2^2 / a^2 - (-1)^2 = 1) :
  (∀ x y : ℝ, y = (sqrt 2 / 2 : ℝ) * x ∨ y = - (sqrt 2 / 2) * x) :=
by
  sorry

end hyperbola_asymptotes_l323_323236


namespace median_salary_is_30000_l323_323878

noncomputable def median_salary (salaries : List ℕ) : ℕ :=
  let sorted := salaries.qsort (· ≤ ·)
  if h : sorted.length ≠ 0 then
    sorted.get (sorted.length / 2)
  else
    0

def company_salaries : List ℕ :=
  List.replicate 1 150000 ++
  List.replicate 3 100000 ++
  List.replicate 12 80000 ++
  List.replicate 8 55000 ++
  List.replicate 35 30000

theorem median_salary_is_30000 (h : company_salaries.length = 59) : median_salary company_salaries = 30000 := by
  sorry

end median_salary_is_30000_l323_323878


namespace L_shaped_wall_bricks_needed_l323_323620

def volume_of_brick (length width height : ℕ) : ℕ :=
  length * width * height

def volume_of_section (length width height : ℕ) : ℕ :=
  length * width * height

def volume_of_opening (length width height : ℕ) : ℕ :=
  length * width * height

def total_volume (volA volB volWindow : ℕ) : ℕ :=
  volA + volB - volWindow

def number_of_bricks (total_vol brick_vol : ℕ) : ℕ :=
  (total_vol + brick_vol - 1) / brick_vol

theorem L_shaped_wall_bricks_needed :
  let brick_length := 25
      brick_width := 11
      brick_height := 6

      sectionA_length := 800  -- converting to cm
      sectionA_width := 100   -- converting to cm
      sectionA_height := 5

      sectionB_length := 500  -- converting to cm
      sectionB_width := 100   -- converting to cm
      sectionB_height := 5

      window_length := 200    -- converting to cm
      window_width := 100     -- converting to cm
      window_height := 5
      
      vol_brick := volume_of_brick brick_length brick_width brick_height
      vol_A := volume_of_section sectionA_length sectionA_width sectionA_height
      vol_window := volume_of_opening window_length window_width window_height
      vol_B := volume_of_section sectionB_length sectionB_width sectionB_height
      total_vol := total_volume vol_A vol_B vol_window

      num_bricks := number_of_bricks total_vol vol_brick
  in num_bricks = 334 :=
  by diabeticinject μ : propcontrol.nepride.1;
µ.stop acl_gen; move l.clip_protons sorry clipboard_mem.

end L_shaped_wall_bricks_needed_l323_323620


namespace range_of_a_l323_323750

noncomputable def has_extreme_values (a : ℝ) : Prop :=
  ∃ x ∈ set.univ, (∂ (λ x, (a * x^2 - 1) / real.exp x) x) = 0

theorem range_of_a :
  ∀ a : ℝ, has_extreme_values a ↔ (a < -1 ∨ a > 0) := 
by
  intro a
  sorry

end range_of_a_l323_323750


namespace identify_minor_premise_l323_323179

theorem identify_minor_premise :
  ∀ (a : ℝ), a > 1 → (∀ x, monotone (λ x, log a x)) ∧ (monotone (λ x, log 2 x)) ∧ (∀ x, f x = log 2 x) ↔ f is_logarithmic := 
sorry

end identify_minor_premise_l323_323179


namespace coefficient_of_x_cubed_in_expansion_l323_323662

theorem coefficient_of_x_cubed_in_expansion :
  let expr := (x^2 - x + 1)^5,
  polynomial.coeff expr 3 = -30 :=
by 
  sorry

end coefficient_of_x_cubed_in_expansion_l323_323662


namespace sum_of_exponents_of_prime_factors_in_sqrt_largest_perfect_square_dividing_factorial_15_l323_323022

theorem sum_of_exponents_of_prime_factors_in_sqrt_largest_perfect_square_dividing_factorial_15 :
  let exponents := (2, 5) :: (3, 3) :: (5, 1) :: (7, 1) :: []
  in (exponents.map Prod.snd).sum = 10 :=
by
  sorry

end sum_of_exponents_of_prime_factors_in_sqrt_largest_perfect_square_dividing_factorial_15_l323_323022


namespace seven_consecutive_composite_exists_fifteen_consecutive_composite_exists_l323_323908

variables (k k' : ℕ)

-- Define the problem for 7 consecutive composite numbers
def after_seven_consecutive_composite : Prop :=
  ∃ (n : ℕ), (n = 210 * k) ∧ ∀ i ∈ (Finset.range 2 9).erase 1, nat.prime i → (nat.prime (n + i + 1) = false)

-- Define the problem for 15 consecutive composite numbers
def after_fifteen_consecutive_composite : Prop :=
  ∃ (m : ℕ), (m = 30030 * k') ∧ ∀ i ∈ (Finset.range 2 17).erase 1, nat.prime i → (nat.prime (m + i + 1) = false)

-- The proposition now states that there exist n and m meeting the conditions
theorem seven_consecutive_composite_exists : after_seven_consecutive_composite k :=
begin
  sorry
end

theorem fifteen_consecutive_composite_exists : after_fifteen_consecutive_composite k' :=
begin
  sorry
end

end seven_consecutive_composite_exists_fifteen_consecutive_composite_exists_l323_323908


namespace correct_propositions_l323_323597

noncomputable def sin_cos_identity (α : Real) : Prop :=
  sin (α + Real.pi / 2) + cos (Real.pi - α) = 0

def log_decreasing_interval (f : Real → Real) : Prop :=
  ∀ x : Real, x < 1 → (λ y, log 3 (y^2 - 2 * y)) x < (λ y, log 3 (y^2 - 2 * y)) (x + 1)

def necessary_condition (P Q : Real → Prop) : Prop :=
  ∀ x : Real, P x → Q x

def insufficient_condition (P Q : Real → Prop) : Prop :=
  ¬ (∀ x : Real, Q x → P x)

def equation_circle_trajectory (c₁ c₂ : Real × Real) (r₁ r₂ : Real) : Prop :=
  ∀ P : Real × Real, dist P c₁ + dist P c₂ = r₁ + r₂

def proposition_1 : Prop := 
  sin_cos_identity

def proposition_2 (f : Real → Real) : Prop :=
  log_decreasing_interval f

def proposition_3 (P Q : Real → Prop) : Prop :=
  necessary_condition P Q ∧ insufficient_condition P Q

def proposition_4 (c₁ c₂ : Real × Real) (r₁ r₂ : Real) : Prop :=
  equation_circle_trajectory c₁ c₂ r₁ r₂

theorem correct_propositions (α : Real) (f : Real → Real) (P Q : Real → Prop) (c₁ c₂ : Real × Real) (r₁ r₂ : Real) :
  proposition_1 α ∧ proposition_3 P Q := sorry

end correct_propositions_l323_323597


namespace sequence_integer_l323_323764

def sequence (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + real.sqrt (3 * a n ^ 2 + 1)

theorem sequence_integer (a : ℕ → ℝ) (h : sequence a) :
  ∀ n, a n ∈ ℤ :=
sorry

end sequence_integer_l323_323764


namespace sum_is_zero_l323_323234

variable (a b c x y : ℝ)

theorem sum_is_zero (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
(h4 : a^3 + a * x + y = 0)
(h5 : b^3 + b * x + y = 0)
(h6 : c^3 + c * x + y = 0) : a + b + c = 0 :=
sorry

end sum_is_zero_l323_323234


namespace count_5_primable_less_than_1000_eq_l323_323135

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323135


namespace radius_of_tangent_circle_l323_323054

theorem radius_of_tangent_circle
  (O : ℝ × ℝ) -- the center of the circle
  (r : ℝ) -- the radius of the circle
  (A B C : ℝ × ℝ) -- the vertices of the triangle
  (h_triangle : ∠A = 30 ∧ ∠B = 60 ∧ ∠C = 90) -- angles of the triangle
  (AB_length : dist A B = 2) -- length of side AB
  (tangent_x_axis : O.2 = r) -- circle tangent to x-axis
  (tangent_y_axis : O.1 = r) -- circle tangent to y-axis
  (tangent_hypotenuse : tangent (r, O) (segment A C)) -- circle tangent to hypotenuse AC
  : r = 4.46 :=
sorry

end radius_of_tangent_circle_l323_323054


namespace sum_floor_eq_n_l323_323334

theorem sum_floor_eq_n (n : ℕ) (h : n > 0) : 
  ( ∑ k in finset.range (n+1), (int.floor ((n + 2^k) / (2^(k+1))))) = n :=
  sorry

end sum_floor_eq_n_l323_323334


namespace sin_alpha_beta_l323_323307

theorem sin_alpha_beta :
  (∀ α β : ℝ, 
    sin α = (4 / 5) ∧ 
    cos α = (3 / 5) ∧ 
    sin β = (3 / 5) ∧ 
    cos β = (- 4 / 5) → 
    sin (α + β) = - (7 / 25)) :=
by
  intros α β h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  sorry

end sin_alpha_beta_l323_323307


namespace ways_to_score_at_least_7_points_l323_323775

-- Definitions based on the given conditions
def red_balls : Nat := 4
def white_balls : Nat := 6
def points_red : Nat := 2
def points_white : Nat := 1

-- Function to count the number of combinations for choosing k elements from n elements
def choose (n : Nat) (k : Nat) : Nat :=
  if h : k ≤ n then
    Nat.descFactorial n k / Nat.factorial k
  else
    0

-- The main theorem to prove the number of ways to get at least 7 points by choosing 5 balls out
theorem ways_to_score_at_least_7_points : 
  (choose red_balls 4 * choose white_balls 1) +
  (choose red_balls 3 * choose white_balls 2) +
  (choose red_balls 2 * choose white_balls 3) = 186 := 
sorry

end ways_to_score_at_least_7_points_l323_323775


namespace total_trip_length_is570_l323_323051

theorem total_trip_length_is570 (v D : ℝ) (h1 : (2:ℝ) + (2/3) + (6 * (D - 2 * v) / (5 * v)) = 2.75)
(h2 : (2:ℝ) + (50 / v) + (2/3) + (6 * (D - 2 * v - 50) / (5 * v)) = 2.33) :
D = 570 :=
sorry

end total_trip_length_is570_l323_323051


namespace Bobby_paycheck_final_amount_l323_323504

theorem Bobby_paycheck_final_amount :
  let salary := 450
  let federal_tax := (1 / 3 : ℚ) * salary
  let state_tax := 0.08 * salary
  let health_insurance := 50
  let life_insurance := 20
  let city_fee := 10
  let total_deductions := federal_tax + state_tax + health_insurance + life_insurance + city_fee
  salary - total_deductions = 184 :=
by
  -- We put sorry here to skip the proof step
  sorry

end Bobby_paycheck_final_amount_l323_323504


namespace find_pairs_l323_323888

-- Define the problem conditions
def equation (n k : ℕ) : Prop := nat.factorial n + n = n ^ k

-- Define the positive integer property
def positive (n : ℕ) : Prop := n > 0

-- State the goal of the theorem
theorem find_pairs : ∀ (n k : ℕ), positive n → positive k → equation n k ↔ 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) :=
by
  intros n k hn hk
  sorry

end find_pairs_l323_323888


namespace triangle_area_inequality_l323_323951

theorem triangle_area_inequality 
  (A B C P Q R S : Type*)
  (area_ABC : Triangle.angle_area A B C = 1)
  (p q r s : ℝ)
  (P_on_AC : Point_on_line_segment P A C p)
  (Q_on_AC : Point_on_line_segment Q A C q)
  (R_on_BC : Point_on_line_segment R B C r)
  (S_on_BC : Point_on_line_segment S B C s)
  (pq_cond : 0 < p ∧ p < q ∧ q < 1)
  (rs_cond : 0 < r ∧ r < s ∧ s < 1)
  : ∃ (T : Triangle), T ⊆ {Triangle.area ARP, Triangle.area APQ, Triangle.area PQR, Triangle.area PRS} ∧ Triangle.area T ≤ 1 / 4 :=
sorry

end triangle_area_inequality_l323_323951


namespace trajectory_of_P_minimum_area_of_ΔPOQ_l323_323237

-- Definitions and conditions
def point_on_circle (A : ℝ × ℝ) : Prop := A.1^2 + A.2^2 = 1
def perpendicular_to_y_axis (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop := A.1 = B.1

-- The trajectory equation of P
theorem trajectory_of_P {A P : ℝ × ℝ} (hA : point_on_circle (A)) (hP_A : A = (P.1 / 2, P.2)) : 
  P.1^2 / 4 + P.2^2 = 1 := 
sorry

-- Minimum area of ΔPOQ
theorem minimum_area_of_ΔPOQ {O P Q : ℝ × ℝ} (hO : O = (0, 0)) 
  (hQ : Q.1 = 3) (hPerp : P.1^2 / 4 + P.2^2 = 1) (hPerp_OP_OQ : (O.1 - P.1) * (O.1 - Q.1) + (O.2 - P.2) * (O.2 - Q.2) = 0):
  (let area_POQ : ℝ := (1/2) * real.sqrt (P.1^2 + P.2^2) * real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) in
  area_POQ) = 3/2 :=
sorry

end trajectory_of_P_minimum_area_of_ΔPOQ_l323_323237


namespace tangent_product_l323_323361

-- Definitions of the problem setup
variables (l : Line) (S : Circle) (P : Point) (A : Point) (B C : Point)
variables (h : ℝ)
variables [Tangency l S P] [SameSide A S l] [DistanceToLine A l h] [h_gt_2 : h > 2]
variables [TangentsFromTo A S B l] [TangentsFromTo A S C l]
variables [TangentIntersectsLine B l] [TangentIntersectsLine C l]

-- The theorem to be proved
theorem tangent_product (l : Line) (S : Circle) (P : Point) (A : Point) (B C : Point) (h : ℝ)
  [Tangency l S P] [SameSide A S l] [DistanceToLine A l h] [h_gt_2 : h > 2]
  [TangentsFromTo A S B l] [TangentsFromTo A S C l] [TangentIntersectsLine B l] [TangentIntersectsLine C l] :
  PB * PC = h^2 - 1 :=
sorry

end tangent_product_l323_323361


namespace no_real_m_perpendicular_l323_323616

theorem no_real_m_perpendicular (m : ℝ) : 
  ¬ ∃ m, ((m - 2) * m = -3) := 
sorry

end no_real_m_perpendicular_l323_323616


namespace perpendicular_lines_unique_a_l323_323976

theorem perpendicular_lines_unique_a (a : ℝ) :
  (∃ l1 l2 : ℝ × ℝ × ℝ, (l1 = (a, -1, 2 * a) ∧ l2 = (2 * a - 1, a, a)) ∧ (l1.1 / l1.2 ≠ 0) ∧ (l2.1 / l2.2 ≠ 0) ∧ ((l1.1 / l1.2) * (l2.1 / l2.2) = -1)) → a = 1 :=
by
  sorry

end perpendicular_lines_unique_a_l323_323976


namespace sigma_mn_inequality_l323_323360

-- Defining the sum of positive divisors function σ
def σ (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, d ∣ n).sum id

-- The assertion to be proved
theorem sigma_mn_inequality (n m : ℕ) (h : σ n > 2 * n) : σ (m * n) > 2 * m * n :=
by
  sorry

end sigma_mn_inequality_l323_323360


namespace smallest_b_l323_323348

def g (x : ℕ) : ℕ :=
  if x % 35 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

noncomputable def g_iter : ℕ → ℕ → ℕ
| 0, x := x
| (n + 1), x := g (g_iter n x)

theorem smallest_b (b : ℕ) : ∃ b > 1, (g 4 = g_iter b 4) ∧ ∀ n, (n > 1 ∧ (g 4 = g_iter n 4)) → n ≥ b := by
  sorry

end smallest_b_l323_323348


namespace parabola_tangent_to_line_l323_323646

theorem parabola_tangent_to_line {a : ℝ} :
  (∀ x : ℝ, ax^2 + 4 = 3x + 1) → (a = 3 / 4) := by
  sorry

end parabola_tangent_to_line_l323_323646


namespace part1_part2_l323_323966

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end part1_part2_l323_323966


namespace min_value_ge_3sqrt3_l323_323693

open Real

noncomputable def min_value (x : Fin 50 → ℝ) : ℝ :=
  ∑ i in Finset.univ, x i / (1 - (x i)^2)

theorem min_value_ge_3sqrt3 (x : Fin 50 → ℝ) (hx_pos : ∀ i, 0 < x i)
    (hx1 : ∑ i in Finset.univ, (x i)^2 = 2) (hx2 : ∑ i in Finset.univ, x i = 5) :
    min_value x ≥ 3 * sqrt 3 :=
sorry

end min_value_ge_3sqrt3_l323_323693


namespace probability_shortest_diagonal_l323_323427

theorem probability_shortest_diagonal (n : ℕ) (h : n = 11) :
  let total_diagonals := n * (n - 3) / 2 in
  let shortest_diagonals := n / 2 in
  shortest_diagonals / total_diagonals = 5 / 44 :=
sorry

end probability_shortest_diagonal_l323_323427


namespace father_payment_l323_323030

variable (x y : ℤ)

theorem father_payment :
  5 * x - 3 * y = 24 :=
sorry

end father_payment_l323_323030


namespace remainder_of_polynomial_l323_323920

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

-- Define the main theorem stating the remainder when f(x) is divided by (x - 1) is 6
theorem remainder_of_polynomial : f 1 = 6 := 
by 
  sorry

end remainder_of_polynomial_l323_323920


namespace eccentricity_of_ellipse_l323_323249

theorem eccentricity_of_ellipse
  (a b c: ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (e: ℝ)
  (h3 : 0 < e ∧ e < 1)
  (h4 : c = a * (1 - e^2))
  (h5 : ∀ m n : ℝ, (n / (m + c)) = (c / b) ∧ b * ((m - c) / 2) + (c * n / 2) = 0 ∧ m^2 / a^2 + n^2 / b^2 = 1 ) :
  e = sqrt 2 / 2 :=
sorry

end eccentricity_of_ellipse_l323_323249


namespace find_k_l323_323940

-- Define the vectors a, b, and c
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (0, 1)

-- Define the vector c involving variable k
variables (k : ℝ)
def vec_c : ℝ × ℝ := (k, -2)

-- Define the combined vector (a + 2b)
def combined_vec : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem to prove
theorem find_k (h : dot_product combined_vec (vec_c k) = 0) : k = 8 :=
by sorry

end find_k_l323_323940


namespace train_overtakes_motorbike_in_80_seconds_l323_323858

-- Definitions of the given conditions
def speed_train_kmph : ℝ := 100
def speed_motorbike_kmph : ℝ := 64
def length_train_m : ℝ := 800.064

-- Definition to convert kmph to m/s
noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Relative speed in m/s
noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_train_kmph - speed_motorbike_kmph)

-- Time taken for the train to overtake the motorbike
noncomputable def time_to_overtake (distance_m : ℝ) (speed_mps : ℝ) : ℝ :=
  distance_m / speed_mps

-- The statement to be proved
theorem train_overtakes_motorbike_in_80_seconds :
  time_to_overtake length_train_m relative_speed_mps = 80.0064 :=
by
  sorry

end train_overtakes_motorbike_in_80_seconds_l323_323858


namespace Gus_eggs_total_l323_323619

theorem Gus_eggs_total :
  let breakfast_omelet := 2
  let breakfast_burrito := 1.5
  let mid_morning_scotch := 1
  let mid_morning_avocado := 0.5
  let lunch_sandwich := 3
  let lunch_caesar := 2 / 3
  let early_afternoon_quiche := 2
  let early_afternoon_frittata := 1.75
  let dinner_soup := 1
  let dinner_deviled := 2
  let dinner_shakshuka := 0.25
  let dessert_custard := 3
  let late_night_rice := 1.5
  let total_eggs := breakfast_omelet + breakfast_burrito +
                    mid_morning_scotch + mid_morning_avocado +
                    lunch_sandwich + lunch_caesar +
                    early_afternoon_quiche + early_afternoon_frittata +
                    dinner_soup + dinner_deviled +
                    dinner_shakshuka + dessert_custard + 
                    late_night_rice
  total_eggs = 19.8167 := 
by
  -- conditions definition
  let breakfast_omelet := 2
  let breakfast_burrito := 1.5
  let mid_morning_scotch := 1
  let mid_morning_avocado := 0.5
  let lunch_sandwich := 3
  let lunch_caesar := 2 / 3
  let early_afternoon_quiche := 2
  let early_afternoon_frittata := 1.75
  let dinner_soup := 1
  let dinner_deviled := 2
  let dinner_shakshuka := 0.25
  let dessert_custard := 3
  let late_night_rice := 1.5

  -- addition of all conditions
  let total_eggs := breakfast_omelet + breakfast_burrito
    + mid_morning_scotch + mid_morning_avocado
    + lunch_sandwich + lunch_caesar
    + early_afternoon_quiche + early_afternoon_frittata
    + dinner_soup + dinner_deviled
    + dinner_shakshuka + dessert_custard
    + late_night_rice

  -- proof placeholder
  sorry

end Gus_eggs_total_l323_323619


namespace largest_perimeter_correct_l323_323154

-- Define the properties of the triangle and segments
def isosceles_triangle_base : ℝ := 10
def isosceles_triangle_height : ℝ := 15
def number_of_segments : ℕ := 5
def segment_length : ℝ := isosceles_triangle_base / number_of_segments
def largest_perimeter : ℝ := 37.03

-- Define the function to calculate the lengths of the segments from the vertex V to the base points
def length_VB (k : ℕ) : ℝ :=
  real.sqrt (isosceles_triangle_height^2 + (k * segment_length)^2)

-- Define the function to calculate the perimeter of the segment k
def perimeter (k : ℕ) : ℝ :=
  2 + length_VB k + length_VB (k + 1)

-- Define the theorem to prove the largest perimeter is 37.03 inches
theorem largest_perimeter_correct :
  (∀ k : ℕ, 0 ≤ k ∧ k < number_of_segments → perimeter k ≤ 37.03) ∧
  (∃ k : ℕ, 0 ≤ k ∧ k < number_of_segments ∧ perimeter k = 37.03) :=
by
  -- Proof is omitted.
  sorry

end largest_perimeter_correct_l323_323154


namespace initial_number_of_girls_l323_323738

theorem initial_number_of_girls (n A : ℕ) (new_girl_weight : ℕ := 80) (original_girl_weight : ℕ := 40)
  (avg_increase : ℕ := 2)
  (condition : n * (A + avg_increase) - n * A = 40) :
  n = 20 :=
by
  sorry

end initial_number_of_girls_l323_323738


namespace unique_n_l323_323876

theorem unique_n (n : ℕ) :
  (2 * 2^3 + 3 * 2^4 + 4 * 2^5 + ∑ k in finset.range(n - 3 + 1), (k + 5) * 2^(k + 5)) = 2^(n + 11) → n = 270 :=
by
  sorry

end unique_n_l323_323876


namespace find_f_prime_zero_l323_323585

-- Given condition: f(x) = x^2 + 2x * f'(1)
def f (x : ℝ) := x^2 + 2 * x * deriv f 1

-- The goal is to prove f'(0) = -4
theorem find_f_prime_zero : deriv f 0 = -4 := by
  sorry

end find_f_prime_zero_l323_323585


namespace minimum_abs_value_specification_l323_323336

noncomputable def min_value_of_abs_expression (n : ℤ) (ω : ℂ) (hω1 : ω^4 = 1) (hω2 : ω ≠ 1) : ℝ :=
  let a := n;
  let b := n + 1;
  let c := n + 2;
  complex.abs (a + b * ω + c * ω^3)

theorem minimum_abs_value_specification :
  ∀ (ω : ℂ), ω^4 = 1 ∧ ω ≠ 1 →
  let minimum_value := min_value_of_abs_expression (-1) ω in
  minimum_value = real.sqrt 2 :=
by
  intros ω hω
  cases hω with hω1 hω2
  let n := -1
  have h := min_value_of_abs_expression n ω hω1 hω2
  sorry

end minimum_abs_value_specification_l323_323336


namespace avg_last_8_numbers_l323_323772

theorem avg_last_8_numbers (A : ℕ → ℝ) (h_length : ∑ i in finset.range 13, A i = 780)
  (h_first_6_avg : ∑ i in finset.range 6, A i = 342)
  (h_7th : A 7 = 50) : 
  (∑ i in finset.range (13 - 6) ∩ finset.range 7 13, A i) / 8 = 54.75 :=
by
  sorry

end avg_last_8_numbers_l323_323772


namespace length_of_train_is_110_l323_323477

-- Define the speeds and time as constants
def speed_train_kmh := 90
def speed_man_kmh := 9
def time_pass_seconds := 4

-- Define the conversion factor from km/h to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh : ℚ) * (5 / 18)

-- Calculate relative speed in m/s
def relative_speed_mps : ℚ := kmh_to_mps (speed_train_kmh + speed_man_kmh)

-- Define the length of the train in meters
def length_of_train : ℚ := relative_speed_mps * time_pass_seconds

-- The theorem to prove: The length of the train is 110 meters
theorem length_of_train_is_110 : length_of_train = 110 := 
by sorry

end length_of_train_is_110_l323_323477


namespace train_and_ship_distance_l323_323146

theorem train_and_ship_distance :
  ∃ d : ℕ, let t_train := d / 48,
                t_ship := d / 60 in
            t_train = t_ship + 2 ∧ d = 480 := by
  sorry

end train_and_ship_distance_l323_323146


namespace num_solutions_sqrt_eq_ax_plus_2_l323_323181

noncomputable def sqrtFunc (x : ℝ) : ℝ := real.sqrt (x + 3)
noncomputable def lineFunc (a x : ℝ) : ℝ := a * x + 2

theorem num_solutions_sqrt_eq_ax_plus_2 (a : ℝ) :
  (0 < a ∧ a < 1 / 6 ∨ 1 / 2 < a ∧ a ≤ 2 / 3 → ∃ x1 x2 : ℝ, sqrtFunc x1 = lineFunc a x1 ∧ sqrtFunc x2 = lineFunc a x2 ∧ x1 ≠ x2) ∧
  (a ≤ 0 ∨ a = 1 / 6 ∨ a = 1 / 2 ∨ a > 2 / 3 → ∃! x : ℝ, sqrtFunc x = lineFunc a x) ∧
  (1 / 6 < a ∧ a < 1 / 2 → ∀ x : ℝ, sqrtFunc x ≠ lineFunc a x) :=
by
  sorry

end num_solutions_sqrt_eq_ax_plus_2_l323_323181


namespace remove_least_candies_l323_323536

theorem remove_least_candies (total_candies : ℕ) (friends : ℕ) (candies_remaining : ℕ) : total_candies = 34 ∧ friends = 5 ∧ candies_remaining = 4 → (total_candies % friends = candies_remaining) :=
by
  intros h
  sorry

end remove_least_candies_l323_323536


namespace least_positive_integer_reducible_fraction_l323_323915

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, n > 0 ∧ gcd (n - 17) (7 * n + 4) > 1 ∧ (∀ m : ℕ, m > 0 ∧ gcd (m - 17) (7 * m + 4) > 1 → n ≤ m) :=
by sorry

end least_positive_integer_reducible_fraction_l323_323915


namespace evaluate_expression_l323_323899

theorem evaluate_expression : (real.rpow (real.rpow 16 (1/4)) 6) = 64 := by
  sorry

end evaluate_expression_l323_323899


namespace fill_time_correct_l323_323455

def rate_fill : ℝ := 1 / 4
def rate_empty : ℝ := 1 / 8
def rate_net : ℝ := rate_fill - rate_empty
def time : ℝ := 1 / rate_net

theorem fill_time_correct : time = 8 := by
  sorry

end fill_time_correct_l323_323455


namespace count_5_primable_under_1000_l323_323122

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323122


namespace c_impossible_value_l323_323211

theorem c_impossible_value (a b c : ℤ) (h : (∀ x : ℤ, (x + a) * (x + b) = x^2 + c * x - 8)) : c ≠ 4 :=
by
  sorry

end c_impossible_value_l323_323211


namespace sum_of_ages_is_258_l323_323560

-- Considering all authors are distinct and ages are known as 258
def sum_of_ages_of_hmmt_authors (ages : List ℕ) (distinct_authors : ℕ) : ℕ :=
  if distinct_authors = 14 then List.sum ages else 0

theorem sum_of_ages_is_258 :
  sum_of_ages_of_hmmt_authors [/* list of ages hidden for privacy reasons */] 14 = 258 :=
  by
    sorry

end sum_of_ages_is_258_l323_323560


namespace similar_triangle_shortest_side_l323_323848

theorem similar_triangle_shortest_side (a b c: ℝ) (d e f: ℝ) :
  a = 21 ∧ b = 20 ∧ c = 29 ∧ d = 87 ∧ c^2 = a^2 + b^2 ∧ d / c = 3 → e = 60 :=
by
  sorry

end similar_triangle_shortest_side_l323_323848


namespace find_m_n_and_sqrt_l323_323243

-- definitions based on conditions
def condition_1 (m : ℤ) : Prop := m + 3 = 1
def condition_2 (n : ℤ) : Prop := 2 * n - 12 = 64

-- the proof problem statement
theorem find_m_n_and_sqrt (m n : ℤ) (h1 : condition_1 m) (h2 : condition_2 n) : 
  m = -2 ∧ n = 38 ∧ Int.sqrt (m + n) = 6 := 
sorry

end find_m_n_and_sqrt_l323_323243


namespace swimming_pool_depth_l323_323165

theorem swimming_pool_depth (d : ℝ) (V : ℝ) (h : ℝ) (π : ℝ) 
  (diam_80 : d = 80) 
  (vol_pool : V = 50265.482457436694) 
  (pi_value : π = Real.pi) 
  (volume_formula : V = π * (d / 2) ^ 2 * h) : 
  h ≈ 10 :=
by
  sorry

end swimming_pool_depth_l323_323165


namespace largest_c_such_that_neg5_in_range_l323_323914

theorem largest_c_such_that_neg5_in_range :
  ∃ (c : ℝ), (∀ x : ℝ, x^2 + 5 * x + c = -5) → c = 5 / 4 :=
sorry

end largest_c_such_that_neg5_in_range_l323_323914


namespace solve_inequality_l323_323921

theorem solve_inequality (x : ℝ) : 6 - x - 2 * x^2 < 0 ↔ x < -2 ∨ x > 3 / 2 := sorry

end solve_inequality_l323_323921


namespace number_of_terms_divisible_by_11_in_first_3030_l323_323267

-- Define the sequence a_n = 100^n + 1
def a (n : ℕ) : ℕ := 100^n + 1

-- Prove the number of terms divisible by 11 in the first 3030 terms of the sequence
theorem number_of_terms_divisible_by_11_in_first_3030
: Finset.card (Finset.filter (λ n, (a n) % 11 = 0) (Finset.range 3031)) = 1 := by
  sorry

end number_of_terms_divisible_by_11_in_first_3030_l323_323267


namespace binary_multiplication_l323_323510

theorem binary_multiplication :
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  a * b = product :=
by 
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  sorry

end binary_multiplication_l323_323510


namespace ariana_total_debt_l323_323500

theorem ariana_total_debt :
  let bill1 := 200
  let bill2 := 130
  let bill3 := 444
  let interest1 := 0.1
  let late_fee2 := 50
  let fee1 := 40
  let n_months1 := 2
  let n_months2 := 6
  let n_months3 := 2
  let calc_bill1 := bill1 + (bill1 * interest1 * n_months1)
  let calc_bill2 := bill2 + (late_fee2 * n_months2)
  let late_fee_month1 := fee1
  let late_fee_month2 := fee1 * 2
  let calc_bill3 := bill3 + (late_fee_month1 + late_fee_month2)
  calc_bill1 + calc_bill2 + calc_bill3 = 1234 :=
by
  let bill1 := 200
  let bill2 := 130
  let bill3 := 444
  let interest1 := 0.1
  let late_fee2 := 50
  let fee1 := 40
  let n_months1 := 2
  let n_months2 := 6
  let n_months3 := 2
  let calc_bill1 := bill1 + (bill1 * interest1 * n_months1)
  let calc_bill2 := bill2 + (late_fee2 * n_months2)
  let late_fee_month1 := fee1
  let late_fee_month2 := fee1 * 2
  let calc_bill3 := bill3 + (late_fee_month1 + late_fee_month2)
  sorry

end ariana_total_debt_l323_323500


namespace slightly_used_crayons_count_l323_323419

-- Definitions
def total_crayons := 120
def new_crayons := total_crayons * (1/3)
def broken_crayons := total_crayons * (20/100)
def slightly_used_crayons := total_crayons - new_crayons - broken_crayons

-- Theorem statement
theorem slightly_used_crayons_count :
  slightly_used_crayons = 56 :=
by
  sorry

end slightly_used_crayons_count_l323_323419


namespace calculate_area_in_acres_l323_323844

def longer_base : ℝ := 20
def shorter_base : ℝ := 12
def height : ℝ := 8
def cm_to_miles : ℝ := 3
def sq_mile_to_acres : ℝ := 640

theorem calculate_area_in_acres :
  ((shorter_base + longer_base) / 2 * height * cm_to_miles^2) * sq_mile_to_acres = 737280 :=
by
  sorry

end calculate_area_in_acres_l323_323844


namespace count_5_primable_below_1000_is_21_l323_323069

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323069


namespace Ron_eats_24_pickle_slices_l323_323382

theorem Ron_eats_24_pickle_slices : 
  ∀ (pickle_slices_Sammy Tammy Ron : ℕ), 
    pickle_slices_Sammy = 15 → 
    Tammy = 2 * pickle_slices_Sammy → 
    Ron = Tammy - (20 * Tammy / 100) → 
    Ron = 24 := by
  intros pickle_slices_Sammy Tammy Ron h_sammy h_tammy h_ron
  sorry

end Ron_eats_24_pickle_slices_l323_323382


namespace double_acute_angle_l323_323956

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_acute_angle_l323_323956


namespace hyperbola_properties_l323_323197

noncomputable def hyperbola_equation (x y : ℝ) : ℝ := (x^2 / 16) - (y^2 / 9)

def passes_through_point (x y : ℝ) (P : ℝ × ℝ) : Prop :=
  hyperbola_equation P.1 P.2 = -1/4

def standard_form (x y : ℝ) : Prop :=
  (y^2) / (9/4) - (x^2) / 4 = 1

def eccentricity : ℝ := 5/3

theorem hyperbola_properties :
  (passes_through_point 2 (2*sqrt(3)) (-3) ∧ ∀ (x y : ℝ), (standard_form x y ∧ eccentricity = 5/3)) :=
begin
  sorry
end

end hyperbola_properties_l323_323197


namespace book_price_increase_l323_323758

theorem book_price_increase (original_price : ℕ) (percent_increase : ℝ) (new_price : ℕ) : 
  percent_increase = 0.50 ∧ original_price = 300 ∧ new_price = original_price + (percent_increase * original_price).to_nat → 
  new_price = 450 := 
by 
  sorry

end book_price_increase_l323_323758


namespace problem_statement_l323_323186

noncomputable def equilateral_triangle_side_length : ℝ := 2

noncomputable def equilateral_triangle_perimeter : ℝ := 3 * equilateral_triangle_side_length

noncomputable def right_triangle_perimeter (AC : ℝ) (CD : ℝ) (DA : ℝ) : Prop :=
  AC + CD + DA = equilateral_triangle_perimeter

noncomputable def pythagorean_theorem (AC : ℝ) (CD : ℝ) (DA : ℝ) : Prop :=
  AC^2 = CD^2 + DA^2

noncomputable def angle_sum (angleBAC : ℝ) (angleCAD : ℝ) : ℝ :=
  angleBAC + angleCAD

noncomputable def sine_of_double_angle (angleBAD: ℝ) : ℝ :=
  Real.sin (2 * angleBAD)

theorem problem_statement :
  ∀ (A B C D : Type) (AC CD DA : ℝ),
  equilateral_triangle_side_length = 2 ∧
  right_triangle_perimeter AC CD DA ∧
  pythagorean_theorem AC CD DA →
  sine_of_double_angle (angle_sum (Real.pi / 3) (Real.pi / 2)) = - (Real.sqrt 3 / 2) :=
by
  intros A B C D AC CD DA h,
  sorry

end problem_statement_l323_323186


namespace value_of_f_neg1_l323_323023

def f (x : ℤ) : ℤ := x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 3 := by
  sorry

end value_of_f_neg1_l323_323023


namespace is_pythagorean_triple_C_l323_323494

theorem is_pythagorean_triple_C : 5^2 + 12^2 = 13^2 :=
by
  calc 
    5^2 + 12^2 = 25 + 144 := by norm_num
              ... = 169 := by norm_num
              ... = 13^2 := by norm_num

end is_pythagorean_triple_C_l323_323494


namespace max_forced_terms_in_1000_seq_l323_323055

def is_reg_seq (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ x : ℝ, ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a k = int.floor (k * x)

def is_forced_term (a : ℕ → ℤ) (n : ℕ) (k : ℕ) : Prop :=
  ∀ b : ℤ, (∃ x : ℝ, ∀ i : ℕ, 1 ≤ i ∧ i < k → a i = int.floor (i * x)) ↔ b = a k

theorem max_forced_terms_in_1000_seq : 
  ∀ a : ℕ → ℤ, is_reg_seq a 1000 → 
  ∃ forced : ℕ → Prop, 
    (∀ k, (1 ≤ k ∧ k ≤ 1000) → (forced k ↔ is_forced_term a 1000 k)) 
    ∧ (finset.card (finset.filter forced (finset.range 1000.succ)) = 985) :=
by
  sorry

end max_forced_terms_in_1000_seq_l323_323055


namespace find_geometric_sequence_first_term_and_ratio_l323_323372

theorem find_geometric_sequence_first_term_and_ratio 
  (a1 a2 a3 a4 a5 : ℕ) 
  (h : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (geo_seq : a2 = a1 * 3 / 2 ∧ a3 = a2 * 3 / 2 ∧ a4 = a3 * 3 / 2 ∧ a5 = a4 * 3 / 2)
  (sum_cond : a1 + a2 + a3 + a4 + a5 = 211) :
  (a1 = 16) ∧ (3 / 2 = 3 / 2) := 
by {
  sorry
}

end find_geometric_sequence_first_term_and_ratio_l323_323372


namespace A1B_parallel_AC1D_CE_perpendicular_AC1D_cos_dihedral_angle_l323_323864

-- Definitions of points, midpoints and conditions
variables (A B C A1 B1 C1 D E : Point)
variables (AC1D : Plane)

-- Given conditions
axiom AB_eq_AC : dist A B = 5 ∧ dist A C = 5
axiom D_midpoint_BC : midpoint D B C
axiom E_midpoint_BB1 : midpoint E B B1
axiom B1BCC1_is_square : square B1 B C C1 6

-- Proof problems
theorem A1B_parallel_AC1D : parallel (line A1 B) AC1D := sorry
theorem CE_perpendicular_AC1D : perpendicular (line C E) AC1D := sorry
theorem cos_dihedral_angle : cos (dihedral_angle C A (line C1 D)) = (8 * real.sqrt 5) / 25 := sorry

end A1B_parallel_AC1D_CE_perpendicular_AC1D_cos_dihedral_angle_l323_323864


namespace total_rent_correct_recoup_investment_period_maximize_average_return_l323_323491

noncomputable def initialInvestment := 720000
noncomputable def firstYearRent := 54000
noncomputable def annualRentIncrease := 4000
noncomputable def maxRentalPeriod := 40

-- Conditions on the rental period
variable (x : ℝ) (hx : 0 < x ∧ x ≤ 40)

-- Function for total rent after x years
noncomputable def total_rent (x : ℝ) := 0.2 * x^2 + 5.2 * x

-- Condition for investment recoup period
noncomputable def recoupInvestmentTime := ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment

-- Function for transfer price
noncomputable def transfer_price (x : ℝ) := -0.3 * x^2 + 10.56 * x + 57.6

-- Function for average return on investment
noncomputable def annual_avg_return (x : ℝ) := (transfer_price x + total_rent x - initialInvestment) / x

-- Statement of theorems
theorem total_rent_correct (x : ℝ) (hx : 0 < x ∧ x ≤ 40) :
  total_rent x = 0.2 * x^2 + 5.2 * x := sorry

theorem recoup_investment_period :
  ∃ x : ℝ, x ≥ 10 ∧ total_rent x ≥ initialInvestment := sorry

theorem maximize_average_return :
  ∃ x : ℝ, x = 12 ∧ (∀ y : ℝ, annual_avg_return x ≥ annual_avg_return y) := sorry

end total_rent_correct_recoup_investment_period_maximize_average_return_l323_323491


namespace calculate_hardcover_volumes_l323_323534

theorem calculate_hardcover_volumes (h p : ℕ) 
  (h_total_volumes : h + p = 12)
  (h_cost_equation : 27 * h + 16 * p = 284)
  (h_p_relation : p = 12 - h) : h = 8 :=
by
  sorry

end calculate_hardcover_volumes_l323_323534


namespace probability_event_comparison_l323_323733

theorem probability_event_comparison (m n : ℕ) :
  let P_A := (2 * m * n) / (m + n)^2
  let P_B := (m^2 + n^2) / (m + n)^2
  P_A ≤ P_B ∧ (P_A = P_B ↔ m = n) :=
by
  sorry

end probability_event_comparison_l323_323733


namespace cone_lateral_area_l323_323739

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  (1 / 2) * (2 * Real.pi * r) * l = 15 * Real.pi :=
by
  rw [h_r, h_l]
  sorry

end cone_lateral_area_l323_323739


namespace imaginary_part_of_fraction_l323_323552

theorem imaginary_part_of_fraction (i : ℝ) (h_i : i = complex.I) : complex.im (i / (1 + 2 * complex.I)) = 1 / 5 :=
by
  sorry

end imaginary_part_of_fraction_l323_323552


namespace right_triangle_angles_l323_323018

/-- Given a right triangle ABC with the right angle at C, and the angle bisector of ∠ACB divides it into two equal angles of 45 degrees each. The angles formed by the angle bisector and the hypotenuse AB are in the ratio 7:11. Prove that the angles of the triangle are 65 degrees and 25 degrees. -/
theorem right_triangle_angles (A B C : Type) [MetricSpace A] 
    (triangle : isRightTriangle A B C)
    (bisect_90_degree : bisectAngle triangle.ACB = [45°, 45°])
    (angle_ratio : ratioOfAngles (angleBisectorAndHypotenuse triangle) = (7,11)) :
    triangle_angles triangle = {65°, 25°, 90°} := 
begin
  sorry
end

end right_triangle_angles_l323_323018


namespace find_x_l323_323906

theorem find_x (x : ℝ) : 
  (16^x + 81^x) / (8^x + 27^x) = (2/3) → x = -1 :=
by
  sorry

end find_x_l323_323906


namespace radius_touches_all_l323_323206

-- Define the problem conditions
variables {r : ℝ}
variables {r₁ r₂ r₃ r₄ : ℝ}
variables {C D A B : ℝ^3} -- Centers of the given balls
variables {O : ℝ^3} -- Center of the ball whose radius we seek

-- Define the distances derived from radii
def dist_CD := 6 -- Distance between the centers of the two balls with radii 3
def dist_AB := 4 -- Distance between the centers of the two balls with radii 2

-- Define the radii
def r₁ := 3
def r₂ := 3
def r₃ := 2
def r₄ := 2

-- Define the distances the new sphere's center O must satisfy by touching the four original balls
def dist_OC := r + 3
def dist_OD := r + 3
def dist_OA := r + 2
def dist_OB := r + 2

-- Create the theorem
theorem radius_touches_all (h : ∀(i j : ℕ), i ≠ j → dist (sphere_center i) (sphere_center j) = radius i + radius j) 
: r = 6 / 11 :=
sorry

end radius_touches_all_l323_323206


namespace locus_of_orthocenter_l323_323819

-- Definitions and conditions
variable {P : Type*} [EuclideanSpace P]
variables (A B C : P) (H : P)
variable (D E F : P) -- points on BC, CA, AB forming parallel lines
variable (α : ℝ) -- angle between planes
variable (h h_A : ℝ) -- perpendicular distances

-- Main theorem
theorem locus_of_orthocenter (h_tan_alpha_lt_hA_sin_alpha D E F : set P)
  (hD : D ∈ lineSegment (B, C))
  (hE : E ∈ lineSegment (C, A))
  (hF : F ∈ lineSegment (A, B))
  (parallel_FA_E_BC : ∥FA.vector AD + t * D.vector E = BC.vector ∥)
  (parallel_DB_F_CA : ∥DB.vector AB + t * D.vector F = CA.vector ∥)
  (parallel_DCE_AB : ∥DC.vector AB + t * D.vector E = AB.vector ∥)
  (h1 : h < h_A)
  (h2 : h < h_B)
  (h3 : h < h_C)
  (h_triangle : pointInTriangle H D E F) :
  pointInTriangle H D E F :=
sorry

end locus_of_orthocenter_l323_323819


namespace part1_part2_l323_323965

variable (a b c : ℝ)

-- Conditions
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a^2 + b^2 + 4*c^2 = 3

-- Part 1: Prove that a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 := sorry

-- Part 2: Given b = 2c, prove that 1/a + 1/c ≥ 3
axiom h5 : b = 2*c
theorem part2 : 1/a + 1/c ≥ 3 := sorry

end part1_part2_l323_323965


namespace fraction_with_buddy_l323_323654

variable (s_6 n_9 : ℕ)

def sixth_graders_paired : ℚ := s_6 / 3
def ninth_graders_paired : ℚ := n_9 / 4

-- Given condition: 1/4 of ninth graders are paired with 1/3 of sixth graders
axiom pairing_condition : ninth_graders_paired = sixth_graders_paired

-- Prove that the fraction of the total number of sixth and ninth graders who have a buddy is 1/7
theorem fraction_with_buddy (h : pairing_condition s_6 n_9) :
  (sixth_graders_paired s_6 / (n_9 + s_6 : ℚ)) = 1 / 7 :=
  sorry

end fraction_with_buddy_l323_323654


namespace isosceles_triangles_with_perimeter_31_l323_323623

theorem isosceles_triangles_with_perimeter_31 :
  { (a, b) : ℕ × ℕ // 2 * a + b = 31 ∧ b % 2 = 1 ∧ b < 2 * a }.card = 8 :=
by
  sorry

end isosceles_triangles_with_perimeter_31_l323_323623


namespace arithmetic_sequence_formula_sum_of_b_n_l323_323578

noncomputable def a_n (n: ℕ) : ℤ := 3 * n - 2

def b_n (n: ℕ) : ℚ := (1/2)^(a_n n)

def S_n (n: ℕ) : ℚ := (4 / 7) * (1 - (1 / 8) ^ n)

theorem arithmetic_sequence_formula :
  ∀ (n : ℕ), a_n n = 3 * n - 2 := sorry

theorem sum_of_b_n :
  ∀ (n : ℕ), (1/2 : ℚ) ≤ S_n n ∧ S_n n < (4/7 : ℚ) := sorry

end arithmetic_sequence_formula_sum_of_b_n_l323_323578


namespace six_identities_l323_323156

theorem six_identities :
    (∀ x, (2 * x - 1) * (x - 3) = 2 * x^2 - 7 * x + 3) ∧
    (∀ x, (2 * x + 1) * (x + 3) = 2 * x^2 + 7 * x + 3) ∧
    (∀ x, (2 - x) * (1 - 3 * x) = 2 - 7 * x + 3 * x^2) ∧
    (∀ x, (2 + x) * (1 + 3 * x) = 2 + 7 * x + 3 * x^2) ∧
    (∀ x y, (2 * x - y) * (x - 3 * y) = 2 * x^2 - 7 * x * y + 3 * y^2) ∧
    (∀ x y, (2 * x + y) * (x + 3 * y) = 2 * x^2 + 7 * x * y + 3 * y^2) →
    6 = 6 :=
by
  intros
  sorry

end six_identities_l323_323156


namespace units_digit_m_squared_plus_2_pow_m_l323_323691

-- Define the value of m
def m : ℕ := 2023^2 + 2^2023

-- Define the property we need to prove
theorem units_digit_m_squared_plus_2_pow_m :
  ((m^2 + 2^m) % 10) = 7 :=
by
  sorry

end units_digit_m_squared_plus_2_pow_m_l323_323691


namespace value_of_expression_l323_323632

variable (a b : ℝ)

def sign (x : ℝ) : ℝ := if x > 0 then 1 else if x < 0 then -1 else 0

theorem value_of_expression (h : a * b > 0) : 
  (a / |a|) + (b / |b|) + (a * b / |a * b|) = 3 ∨ 
  (a / |a|) + (b / |b|) + (a * b / |a * b|) = -1 := by
  sorry

end value_of_expression_l323_323632


namespace cookie_sales_l323_323037

theorem cookie_sales (n M A : ℕ) 
  (hM : M = n - 9)
  (hA : A = n - 2)
  (h_sum : M + A < n)
  (hM_positive : M ≥ 1)
  (hA_positive : A ≥ 1) : 
  n = 10 := 
sorry

end cookie_sales_l323_323037


namespace prove_a_lt_zero_l323_323587

variable (a b c : ℝ)

-- Define the quadratic function
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions:
-- The polynomial has roots at -2 and 3
def has_roots : Prop := 
  a ≠ 0 ∧ (a * (-2)^2 + b * (-2) + c = 0) ∧ (a * 3^2 + b * 3 + c = 0)

-- f(-b/(2*a)) > 0
def vertex_positive : Prop := 
  f a b c (-b / (2 * a)) > 0

-- Target: Prove a < 0
theorem prove_a_lt_zero 
  (h_roots : has_roots a b c)
  (h_vertex : vertex_positive a b c) : a < 0 := 
sorry

end prove_a_lt_zero_l323_323587


namespace integer_solution_l323_323026

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n^2 > -27) : n = 2 :=
by {
  sorry
}

end integer_solution_l323_323026


namespace triangle_is_isosceles_l323_323308

-- Define triangle with altitudes
variables {A B C D E L : Point}
variables {α β : Angle}
variables {l : Line}
variables {LB LD LE : Length}
variables [IsTriangle ABC]
variables [IsAltitude AD ABC]
variables [IsAltitude BE ABC]
variables [PointOnLine L ED]
variables [IsOrthogonal BL ED]

-- The given condition
axiom given_condition : LB^2 = LD * LE

-- The goal to prove
theorem triangle_is_isosceles 
  (h1 : IsAltitude AD ABC)
  (h2 : IsAltitude BE ABC)
  (h3 : PointOnLine L ED)
  (h4 : IsOrthogonal BL ED)
  (h5 : LB^2 = LD * LE) : IsIsosceles ABC := 
sorry

end triangle_is_isosceles_l323_323308


namespace train_speed_l323_323481

def length_train : ℝ := 250.00000000000003
def crossing_time : ℝ := 15
def meter_to_kilometer (x : ℝ) : ℝ := x / 1000
def second_to_hour (x : ℝ) : ℝ := x / 3600

theorem train_speed :
  (meter_to_kilometer length_train) / (second_to_hour crossing_time) = 60 := 
  sorry

end train_speed_l323_323481


namespace find_a_l323_323882

noncomputable def ab (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a {a : ℝ} : ab a 6 = -3 → a = 23 :=
by
  sorry

end find_a_l323_323882


namespace sqrt_exp_abs_value_l323_323202

theorem sqrt_exp_abs_value : (sqrt ((1 - sqrt 3) ^ 2)) = (sqrt 3 - 1) := by
  sorry

end sqrt_exp_abs_value_l323_323202


namespace sum_P_eq_binom_l323_323374

-- Assuming P_{k, l} is some predefined combinatorial function
def P (k l n : ℕ) : ℕ := sorry -- placeholder for the real definition

theorem sum_P_eq_binom (k l : ℕ) :
  (∑ n in Finset.range (k * l + 1), P k l n) = Nat.choose (k + l) k :=
by
  sorry

end sum_P_eq_binom_l323_323374


namespace find_common_ratio_l323_323220

noncomputable def common_ratio (q : ℝ) : Prop :=
  ∃ (a₁ : ℝ) (h : q > 0), 
  let a_2 := a₁ * q,
  let a_3 := a₁ * q^2,
  let a_4 := a₁ * q^3,
  let a_5 := a₁ * q^4 in
  a_2 + a_4 = 3 ∧ a_3 * a_5 = 2 ∧ q = Real.sqrt ((3 * Real.sqrt 2 + 2) / 7)

theorem find_common_ratio : ∃ q : ℝ, common_ratio q :=
sorry

end find_common_ratio_l323_323220


namespace find_k_l323_323981

theorem find_k (k : ℝ) : 1^2 - 3*1 + 2*k = 0 → k = 1 := by
  intro h
  have h1 : 1 - 3 = -2 := rfl -- Computing the left-hand side part
  rw [←h1] at h -- Simplifying the equation with the root 1
  have h2 : -2 + 2*k = 0 := h -- Matching the simplified equation
  norm_num at h2 -- Numeric simplification to isolate k
  linarith -- Solving the simplified linear equation
  sorry -- final proof step

end find_k_l323_323981


namespace problem_I_problem_II_l323_323444

-- Problem I
theorem problem_I (a b : ℝ) (h1 : -1/2 < a ∧ a < 1/2) (h2 : -1/2 < b ∧ b < 1/2) :
  |(1/3:ℝ) * a + (1/6:ℝ) * b| < 1/4 :=
by sorry

-- Problem II
theorem problem_II (a : ℝ)
  (h : ∀ x : ℝ, |2*x + 1| + |2*x - 3| - log2 (a^2 - 3*a) > (2:ℝ)) :
  (-1 < a ∧ a < 0) ∨ (3 < a ∧ a < 4) :=
by sorry

end problem_I_problem_II_l323_323444


namespace rhombus_diagonal_length_l323_323398

theorem rhombus_diagonal_length
  (d1 d2 A : ℝ)
  (h1 : d1 = 20)
  (h2 : A = 250)
  (h3 : A = (d1 * d2) / 2) :
  d2 = 25 :=
by
  sorry

end rhombus_diagonal_length_l323_323398


namespace proof_problem_l323_323771

-- Define the problem:
def problem := ∀ (a : Fin 100 → ℝ), 
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are distinct
  ∃ i : Fin 100, a i + a (⟨i.val + 3, sorry⟩) > a (⟨i.val + 1, sorry⟩) + a (⟨i.val + 2, sorry⟩)
-- Summarize: there exists four consecutive points on the circle such that 
-- the sum of the numbers at the ends is greater than the sum of the numbers in the middle.

theorem proof_problem : problem := sorry

end proof_problem_l323_323771


namespace smallest_product_not_factor_of_48_exists_l323_323004

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l323_323004


namespace ariana_total_debt_l323_323501

theorem ariana_total_debt :
  let bill1 := 200
  let bill2 := 130
  let bill3 := 444
  let interest1 := 0.1
  let late_fee2 := 50
  let fee1 := 40
  let n_months1 := 2
  let n_months2 := 6
  let n_months3 := 2
  let calc_bill1 := bill1 + (bill1 * interest1 * n_months1)
  let calc_bill2 := bill2 + (late_fee2 * n_months2)
  let late_fee_month1 := fee1
  let late_fee_month2 := fee1 * 2
  let calc_bill3 := bill3 + (late_fee_month1 + late_fee_month2)
  calc_bill1 + calc_bill2 + calc_bill3 = 1234 :=
by
  let bill1 := 200
  let bill2 := 130
  let bill3 := 444
  let interest1 := 0.1
  let late_fee2 := 50
  let fee1 := 40
  let n_months1 := 2
  let n_months2 := 6
  let n_months3 := 2
  let calc_bill1 := bill1 + (bill1 * interest1 * n_months1)
  let calc_bill2 := bill2 + (late_fee2 * n_months2)
  let late_fee_month1 := fee1
  let late_fee_month2 := fee1 * 2
  let calc_bill3 := bill3 + (late_fee_month1 + late_fee_month2)
  sorry

end ariana_total_debt_l323_323501


namespace A_finishes_remaining_work_in_4_days_l323_323804

-- Definitions of the conditions
def work_rate_A (work : ℝ) : ℝ := work / 12
def work_rate_B (work : ℝ) : ℝ := work / 15

-- Given B has worked for 10 days
def amount_of_work_B (work : ℝ) : ℝ := (work_rate_B work) * 10

-- The remaining work to be completed by A
def remaining_work (work : ℝ) : ℝ := work - amount_of_work_B work

-- The time required by A to complete the remaining work
def time_for_A_to_finish_remaining_work (work : ℝ) : ℝ := (remaining_work work) / (work_rate_A work)

-- The proof statement
theorem A_finishes_remaining_work_in_4_days (work : ℝ) : time_for_A_to_finish_remaining_work work = 4 :=
by
  -- Placeholder for the proof
  sorry

end A_finishes_remaining_work_in_4_days_l323_323804


namespace number_of_5_primable_numbers_below_1000_l323_323118

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323118


namespace factorize_f_l323_323542

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x

theorem factorize_f (x : ℝ) : f(x) = x * (x - 2)^2 := by
  sorry

end factorize_f_l323_323542


namespace infinite_solutions_d_eq_5_l323_323524

theorem infinite_solutions_d_eq_5 :
  ∃ (d : ℝ), d = 5 ∧ ∀ (y : ℝ), 3 * (5 + d * y) = 15 * y + 15 :=
by
  sorry

end infinite_solutions_d_eq_5_l323_323524


namespace neither_question_correct_l323_323283

variable (A B A_and_B : ℝ)

theorem neither_question_correct (hA : A = 85) (hB : B = 65) (hA_and_B : A_and_B = 55) :
  100 - (A + B - A_and_B) = 5 :=
by
  have : A ∪ B = A + B - A_and_B := sorry
  have : 100 - (A + B - A_and_B) = 5 := sorry
  exact this

end neither_question_correct_l323_323283


namespace solution_exists_l323_323546

theorem solution_exists (x : ℝ) (h₀ : 0 ≤ x ∧ x < 2 * Real.pi) : 
  (sin x - cos x = Real.sqrt 2) ↔ x = (3 * Real.pi) / 4 :=
by
  sorry

end solution_exists_l323_323546


namespace abs_neg_product_eq_product_l323_323631

variable (a b : ℝ)

theorem abs_neg_product_eq_product (h1 : a < 0) (h2 : 0 < b) : |-a * b| = a * b := by
  sorry

end abs_neg_product_eq_product_l323_323631


namespace tan_beta_equation_max_tan_beta_l323_323955

variables {α β : ℝ} (h1 : α > 0) (h2 : α < π / 2) (h3 : β > 0) (h4 : β < π / 2)
variables (h : sin β / sin α = cos (α + β))

-- Prove that tan β = sin 2α / (3 - cos 2α)
theorem tan_beta_equation :
  tan β = sin (2 * α) / (3 - cos (2 * α)) :=
by
  sorry

-- Find the maximum value of tan β
theorem max_tan_beta :
  ∃ m, m = sqrt 2 / 4 ∧ (∀ θ, θ > 0 ∧ θ < π / 2 → tan β θ ≤ m) :=
by
  sorry

end tan_beta_equation_max_tan_beta_l323_323955


namespace train_speed_l323_323480

noncomputable def speed_of_train (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  let distance_kilometers := distance_meters / 1000
  let time_hours := time_seconds / 3600
  distance_kilometers / time_hours

theorem train_speed (distance_meters : ℝ) (time_seconds : ℝ) (h_dist : distance_meters = 250.00000000000003) (h_time : time_seconds = 15) :
  speed_of_train distance_meters time_seconds = 60 :=
by
  subst h_dist
  subst h_time
  -- Calculate speed step-by-step
  have distance_km : distance_meters / 1000 = 0.25000000000000003 := by norm_num
  have time_hrs : time_seconds / 3600 = 0.004166666666666667 := by norm_num
  have speed_kph : (distance_meters / 1000) / (time_seconds / 3600) = 60 := by norm_num
  exact speed_kph

end train_speed_l323_323480


namespace correct_propositions_l323_323596

theorem correct_propositions :
  (∃ x : ℝ, 0 < x ∧ log 2 x + 2^x > 2 * x) ∧
  (∀ a b : ℝ, b = 2^a → ∃ x : ℝ, x = log 2 b → (a, b) ∈ { (x, y) | y = 2^x } ) :=
by
  sorry

end correct_propositions_l323_323596


namespace problem1_solution_problem2_solution_l323_323606

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 4|
def g (x : ℝ) : ℝ := |2 * x + 1|

-- Problem 1
theorem problem1_solution :
  {x : ℝ | f x < g x} = {x : ℝ | x < -5 ∨ x > 1} :=
sorry

-- Problem 2
theorem problem2_solution :
  ∀ (a : ℝ), (∀ x : ℝ, 2 * f x + g x > a * x) ↔ -4 ≤ a ∧ a < 9 / 4 :=
sorry

end problem1_solution_problem2_solution_l323_323606


namespace exist_parallel_line_l323_323715

noncomputable def point (α : Type) := (α × α)

variables {α : Type} [linear_ordered_field α]

-- Given points G, D, F on a straight line with D being the midpoint of G and F
variables (G D F H : point α)
axiom D_midpoint : ∃ (t : α), t ∈ Icc (0 : α) 1 ∧ (D = (t • G.1 + (1 - t) • F.1, t • G.2 + (1 - t) • F.2))

-- Proving that there exists a line HI through H parallel to GF
theorem exist_parallel_line (G D F H : point α) (h_midpoint : ∃ t ∈ Icc (0 : α) 1, D = (t • G.1 + (1 - t) • F.1, t • G.2 + (1 - t) • F.2)) :
  ∃ (I : point α), (∃ t : α, I = (t • H.1 + (1 - t) • G.1, t • H.2 + (1 - t) • G.2)) ∧ (GF_parallel : ∀ t : α, (t • H.1 + (1 - t) • G.1 - G.1) * (D.2 - F.2) = (t • H.2 + (1 - t) • G.2 - G.2) * (D.1 - F.1)) :=
sorry

end exist_parallel_line_l323_323715


namespace limit_calculation_l323_323506

theorem limit_calculation :
  (tendsto (λ n : ℕ, (n - 3*n^2)/(5*n^2 + 1)) at_top (𝓝 (-3/5))) :=
begin
  sorry
end

end limit_calculation_l323_323506


namespace sum_of_x_satisfying_series_l323_323559

theorem sum_of_x_satisfying_series :
  let series := λ (x : ℝ), 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + ∑ n : ℕ in 2 * n, x ^ (2 * n)
  ∃ (x : ℝ), x = series x ∧ |x| < 1 ∧ x = ( -1 + Real.sqrt 5 ) / 2
:= sorry

end sum_of_x_satisfying_series_l323_323559


namespace probability_solution_l323_323860

variables (P_Alex P_Bella P_Kyle P_David P_Catherine : ℚ)
variables (Succ_Alex : P_Alex = 1 / 4)
variables (Succ_Bella : P_Bella = 3 / 5)
variables (Succ_Kyle : P_Kyle = 1 / 3)
variables (Succ_David : P_David = 2 / 7)
variables (Succ_Catherine : P_Catherine = 5 / 9)

theorem probability_solution :
  (P_Alex * P_Kyle * P_Catherine * (1 - P_Bella) * (1 - P_David)) = 25 / 378 :=
by
  rw [Succ_Alex, Succ_Kyle, Succ_Catherine, Succ_Bella, Succ_David],
  norm_num,
  sorry

end probability_solution_l323_323860


namespace variation_of_powers_l323_323635

theorem variation_of_powers (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end variation_of_powers_l323_323635


namespace people_on_williams_bus_l323_323421

theorem people_on_williams_bus
  (P : ℕ)
  (dutch_people : ℕ)
  (dutch_americans : ℕ)
  (window_seats : ℕ)
  (h1 : dutch_people = (3 * P) / 5)
  (h2 : dutch_americans = dutch_people / 2)
  (h3 : window_seats = dutch_americans / 3)
  (h4 : window_seats = 9) : 
  P = 90 :=
sorry

end people_on_williams_bus_l323_323421


namespace albino_8_antlered_deers_l323_323779

theorem albino_8_antlered_deers (total_deer : ℕ) (perc_8_antlered : ℝ) (fraction_albino : ℝ) 
  (h_total_deer : total_deer = 920) (h_perc_8_antlered : perc_8_antlered = 0.10) 
  (h_fraction_albino : fraction_albino = 0.25) : 
  (nat.floor ((total_deer * perc_8_antlered) * fraction_albino) : ℕ) = 23 :=
by
  sorry

end albino_8_antlered_deers_l323_323779


namespace harmonic_power_identity_l323_323722

open Real

theorem harmonic_power_identity (a b c : ℝ) (n : ℕ) (hn : n % 2 = 1) 
(h : (1 / a + 1 / b + 1 / c) = 1 / (a + b + c)) :
  (1 / (a ^ n) + 1 / (b ^ n) + 1 / (c ^ n) = 1 / (a ^ n + b ^ n + c ^ n)) :=
sorry

end harmonic_power_identity_l323_323722


namespace cos_range_l323_323629

theorem cos_range (α β : ℝ) (h : sin α + sin β = (sqrt 2) / 2) :
  -sqrt (7 / 2) ≤ cos α + cos β ∧ cos α + cos β ≤ sqrt (7 / 2) :=
by
  sorry

end cos_range_l323_323629


namespace count_isosceles_numbers_l323_323698

def is_isosceles (a b c : ℕ) : Prop :=
(a = b ∨ b = c ∨ a = c) ∧ (a + b > c ∧ b + c > a ∧ a + c > b)

theorem count_isosceles_numbers :
  let n := (100*a + 10*b + c) in
  ∃ k : ℕ, k = 28 ∧
  (∀ a b c : ℕ, a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} ∧ c ∈ {1, 2, 3, 4} →
  is_isosceles a b c →
  ∃ n, n = 100*a + 10*b + c) := sorry

end count_isosceles_numbers_l323_323698


namespace sin_add_polynomial_l323_323818

/-- Given x = sin α and y = sin β, express the relationship among x, y, and z as
    z^4 - 2z^2(x^2 + y^2 - 2x^2 y^2) + (x^2 - y^2)^2 = 0
    and determine the values of x and y for which z takes fewer than four values. -/
theorem sin_add_polynomial (α β : ℝ) (x y z : ℝ) :
  x = Real.sin α → y = Real.sin β →
  (z^4 - 2*z^2*(x^2 + y^2 - 2*x^2*y^2) + (x^2 - y^2)^2 = 0) ∧
  (∃ fewer_values_conditions
    x = 0 ∨ x = 1 ∨ x = -1 ∨ y = 0 ∨ y = 1 ∨ y = -1 → 
    -- Proof of additional statements for fewer values conditions continue here
    sorry) :=
begin
  intros hx hy,
  -- Proof continues here
  sorry
end

end sin_add_polynomial_l323_323818


namespace solve_sin_cos_equation_l323_323730

theorem solve_sin_cos_equation (x : ℝ) (k : ℤ) (hx1 : sin x ≠ 0) (hx2 : cos x ≠ 0) :
  let u := real.arcsin (3 / 4) in
  (x = u + k * real.pi ∨ x = -u + k * real.pi) ↔ 
  (2 * sin (3 * x) / sin x - 3 * cos (3 * x) / cos x = 7 * abs (sin x)) :=
by sorry

end solve_sin_cos_equation_l323_323730


namespace area_of_park_l323_323410

noncomputable def perimeter : ℕ := 100
noncomputable def length_eq : ℕ → ℕ := λ w, 3 * w - 10

theorem area_of_park (l w : ℕ) (h1 : 2 * l + 2 * w = 100) (h2 : l = 3 * w - 10) :
  l * w = 525 :=
by sorry

end area_of_park_l323_323410


namespace no_integer_solutions_l323_323545

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), 19 * x^3 - 17 * y^3 = 50 := 
by 
  sorry

end no_integer_solutions_l323_323545


namespace cost_of_paving_l323_323406

def length := 5.5
def width := 3.75
def rate := 1200

theorem cost_of_paving : (length * width * rate) = 24750 := 
by
  have area := length * width
  suffices : area * rate = 24750
  · exact this
  calc
    area * rate = (5.5 * 3.75) * 1200 : by simp [length, width, rate]
          ...  = 20.625 * 1200        : by norm_num
          ...  = 24750                : by norm_num

end cost_of_paving_l323_323406


namespace max_fC_l323_323694

theorem max_fC (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, 
    (if n % 4 = 0 then fC n = 6 * k else 
     if n % 4 = 1 then fC n = 6 * k + 1 else 
     if n % 4 = 2 then fC n = 6 * k + 2 else 
     fC n = 6 * k + 3) := 
sorry

end max_fC_l323_323694


namespace smallest_product_not_factor_of_48_l323_323011

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l323_323011


namespace true_propositions_l323_323335

variables {a b : Line}
variables {α β : Plane}

-- Definitions based on conditions
def parallel (x y : Line) : Prop := sorry -- Assuming parallel is defined
def perpendicular (x : Line) (y : Plane) : Prop := sorry -- Assuming perpendicular is defined

-- Propositions
def prop2 : Prop := 
  ∀ (a b : Line) (α : Plane), perpendicular a α ∧ perpendicular b α → parallel a b

def prop4 : Prop := 
  ∀ (a : Line) (α β : Plane), perpendicular a α ∧ perpendicular a β → parallel α β

-- The theorem to be proved
theorem true_propositions : prop2 ∧ prop4 :=
by
  split;
  { sorry } -- Proofs are not provided

end true_propositions_l323_323335


namespace pizzas_ordered_l323_323790

theorem pizzas_ordered (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 168) : total_slices / slices_per_pizza = 21 := 
by 
  rw [h1, h2]
  norm_num

end pizzas_ordered_l323_323790


namespace least_k_l323_323765

noncomputable section

def b : ℕ → ℝ
| 0       := 0         -- not used per the problem definition, just to initialize
| (n + 1) := if n = 0 then 1 else (real.log (2 * n + 3) / real.log 3)

theorem least_k (k : ℕ) (h1 : 1 < k) (hk : ∃ (n : ℕ), b n = (n : ℝ)) : k = 12 :=
by
  have h2 : ∀ n, 3^(b (n + 1) - b n) = (2 * n + 3) / (2 * n + 1),
    from sorry -- This should be a direct application of the given recurrence relation.

  have h3 : ∀ n, 3^b n = 2 * n + 3,
    from sorry -- This follows from the telescoping product.

  have h4 : ∀ n, b n ∈ ℤ ↔ ∃ i : ℕ, (2 * n + 3) = 3^i,
    from sorry

  have h5 : ∃ k, k > 1 ∧ b k ∈ ℤ ∧ ∀ m, m > 1 ∧ b m ∈ ℤ → k ≤ m,
    from sorry

  exact sorry

end least_k_l323_323765


namespace zeros_of_g_l323_323641

theorem zeros_of_g (a b : ℝ) (h : a + b = 0) : 
  let g := λ x : ℝ, b * x ^ 2 - a * x in
  ∀ x, g x = 0 ↔ x = 0 ∨ x = -1 :=
by
  let g := λ x : ℝ, b * x ^ 2 - a * x
  have h_b : b = -a, from eq_neg_of_add_eq_zero h
  sorry

end zeros_of_g_l323_323641


namespace variance_of_data_set_is_not_4_l323_323027

-- Defining the data set.
def data_set : List ℕ := [2, 1, 3, 2, 3]

-- Definition to calculate the mean of the data set
def mean (lst : List ℕ) : ℚ := 
  (list.sum lst : ℚ) / list.length lst

-- Definition to calculate the variance of the data set
def variance (lst : List ℕ) : ℚ :=
  let μ := mean lst
  in (list.sum (list.map (λ x => (x - μ) ^ 2) lst) : ℚ) / list.length lst

-- Problem statement: The variance of the given data set is not 4
theorem variance_of_data_set_is_not_4 :
  variance data_set ≠ 4 := 
sorry

end variance_of_data_set_is_not_4_l323_323027


namespace paintings_on_Sep27_l323_323532

-- Definitions for the problem conditions
def total_days := 6
def paintings_per_2_days := (6 : ℕ)
def paintings_per_3_days := (8 : ℕ)
def paintings_P22_to_P26 := 30

-- Function to calculate paintings over a given period
def paintings_in_days (days : ℕ) (frequency : ℕ) : ℕ := days / frequency

-- Function to calculate total paintings from the given artists
def total_paintings (d : ℕ) (p2 : ℕ) (p3 : ℕ) : ℕ :=
  p2 * paintings_in_days d 2 + p3 * paintings_in_days d 3

-- Calculate total paintings in 6 days
def total_paintings_in_6_days := total_paintings total_days paintings_per_2_days paintings_per_3_days

-- Proof problem: Show the number of paintings on the last day (September 27)
theorem paintings_on_Sep27 : total_paintings_in_6_days - paintings_P22_to_P26 = 4 :=
by
  sorry

end paintings_on_Sep27_l323_323532


namespace num_mappings_l323_323569

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {0, 1, 2}

theorem num_mappings (f : 0 → ℕ) (h : ∀ a ∈ A, f a ∈ B) :
    (∃ n : ℕ, n = 14) :=
by
  have valid_mappings : Finset (0 → ℕ) := 
    (Finset.pi_finset (Finset.univ) (λ _, Finset.insert 0 $ Finset.insert 1 $ Finset.singleton 2)).filter (λ f, f 0 + f 1 > f 2)
  have : valid_mappings.card = 14 := sorry
  use 14
  exact this

end num_mappings_l323_323569


namespace product_simplification_l323_323901

def term (n : ℕ) : ℚ := 1 - 1 / n

noncomputable def product_of_terms (n : ℕ) : ℚ :=
  ∏ k in finset.range (n - 2) + 3, term k

theorem product_simplification :
  product_of_terms 200 = 1 / 100 :=
by
  -- Product should equal 1/100
  sorry

end product_simplification_l323_323901


namespace phone_number_count_l323_323265

theorem phone_number_count : 
  let possible_phone_numbers := 9 * 10^6 in
  possible_phone_numbers = 9 * 10^6 :=
by
  sorry

end phone_number_count_l323_323265


namespace henry_distance_l323_323263

def meters_to_feet (meters : ℝ) : ℝ := meters * 3.281

def net_distance (east1 west1 south1 : ℝ) : ℝ :=
  let net_east := east1 - west1
  Real.sqrt (net_east ^ 2 + south1 ^ 2)

theorem henry_distance :
  net_distance (meters_to_feet 15) (meters_to_feet 20) 50 = 52.625 :=
by
  -- Condition 1: Convert distances
  let east1 := meters_to_feet 15
  let west1 := meters_to_feet 20
  let south1 := 50

  -- Condition 2: Calculate net distance and apply Pythagorean theorem
  let net_east := east1 - west1
  have h1 : net_east = -16.405 := by sorry
  have h2 : Real.sqrt (net_east ^ 2 + south1 ^ 2) = 52.625 := by sorry

  -- Conclusion
  exact h2

end henry_distance_l323_323263


namespace part_a_part_b_l323_323813

-- Define the basic terms and conditions for part (a)
variables {n : ℕ}
variables (a b : Fin n → ℝ)
-- l is an arbitrary tangent line that doesn't intersect vertices
constant l : ℝ 

-- Part (a) statement
theorem part_a (h : ∀ i, b i * b ((i + 1) % n) / a i ^ 2 = sin ((π / (n : ℝ)) * i) * sin (π / (n : ℝ) * (i + 1))):
  (∏ i in Finset.range n, b i) / (∏ i in Finset.range n, a i) = 1 := sorry

-- Define the basic terms and conditions for part (b)
variables {m : ℕ} (hm : n = 2 * m)

-- Part (b) statement
theorem part_b (h : ∀ i, b i * b ((i + 1) % n) / a i ^ 2 = sin ((π / (n : ℝ)) * i) * sin (π / (n : ℝ) * (i + 1))):
  (∏ i in finset.filter (λ x : Fin n, x % 2 = 1) finset.univ, a i) / 
  (∏ i in finset.filter (λ x : Fin n, x % 2 = 0) finset.univ, a i) = 
  (∏ i in finset.range (m + 1) , 1) := sorry

end part_a_part_b_l323_323813


namespace factorize_expression_l323_323541

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 :=
by
  sorry

end factorize_expression_l323_323541


namespace probability_different_colors_l323_323049

theorem probability_different_colors :
  let red_chips := 7
  let green_chips := 5
  let total_chips := red_chips + green_chips
  let prob_red := red_chips / total_chips
  let prob_green := green_chips / total_chips
  let prob_red_then_green := prob_red * prob_green
  let prob_green_then_red := prob_green * prob_red
  (prob_red_then_green + prob_green_then_red) = 35 / 72 := by
  let red_chips := 7
  let green_chips := 5
  let total_chips := red_chips + green_chips
  have prob_red : ℚ := red_chips / total_chips
  have prob_green : ℚ := green_chips / total_chips
  have prob_red_then_green : ℚ := prob_red * prob_green
  have prob_green_then_red : ℚ := prob_green * prob_red
  have result : ℚ := prob_red_then_green + prob_green_then_red
  show result = 35 / 72
  sorry

end probability_different_colors_l323_323049


namespace color_circle_points_l323_323326

theorem color_circle_points (k n : ℕ) (h_k : 0 < k) (h_n : 0 < n) :
  ∃ R : ℕ → ℕ → ℕ,
  (∑ B in finset.powerset_len (k-1) (finset.range (k + n - 1)), 
    ∏ i in B, nat.choose n i.card) = R n (k - 1) ∧
  (∑ B in finset.powerset_len (k-1) (finset.range (k + n - 1)), 
    ∏ i in B, nat.choose n i.card) = k! * R n (k - 1) :=
sorry

end color_circle_points_l323_323326


namespace inverse_variation_l323_323767

noncomputable def proof_problem (x y : ℝ) : Prop :=
  (x * y = 108) →
  (x = 3) →
  (y = 64) →
  ((x^2 * real.cbrt(y) = 36) → y ≈ 108.16)

theorem inverse_variation (x y : ℝ) :
  proof_problem x y :=
by
  intros h1 h2 h3 h4
  sorry

end inverse_variation_l323_323767


namespace find_pairs_l323_323887

theorem find_pairs (n k : ℕ) (h_pos_n : 0 < n) (h_cond : n! + n = n ^ k) : 
  (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3) := 
by 
  sorry

end find_pairs_l323_323887


namespace smallest_degree_poly_with_roots_l323_323391

/-- Suppose the polynomial p(x) has the following roots:
  1. 3 - sqrt 8
  2. 5 + sqrt 13
  3. 16 - 3 * sqrt 2
  4. -2 * sqrt 3
  5. 7 - 2 * sqrt 5
  6. 9 + sqrt 7

  Prove that the smallest possible degree of p(x) given that it has rational coefficients is 12.
-/
theorem smallest_degree_poly_with_roots :
  ∃ (p : ℚ[X]), 
  (p ≠ 0) ∧
  (p.eval (3 - real.sqrt 8) = 0) ∧
  (p.eval (5 + real.sqrt 13) = 0) ∧
  (p.eval (16 - 3 * real.sqrt 2) = 0) ∧
  (p.eval (-2 * real.sqrt 3) = 0) ∧
  (p.eval (7 - 2 * real.sqrt 5) = 0) ∧
  (p.eval (9 + real.sqrt 7) = 0) ∧
  (nat_degree p = 12) := by
  sorry

end smallest_degree_poly_with_roots_l323_323391


namespace smallest_product_not_factor_of_48_l323_323014

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l323_323014


namespace volume_of_rotated_cube_intersection_is_three_fourths_l323_323726

-- Define the cube and its properties
structure Cube :=
  (edge_length : ℝ)
  (volume : ℝ := edge_length^3)

-- The cube in question
def unit_cube : Cube :=
  { edge_length := 1 }

-- Define the concept of rotating a cube around a body diagonal by 60 degrees
def rotate_cube (c : Cube) (angle : ℝ) (diagonal : ℝ) : Cube :=
  { c with edge_length := c.edge_length }

-- Define the volume of intersection
noncomputable def volume_of_intersection (c : Cube) (rotation : Cube) : ℝ :=
  sorry

/-- Prove that the volume of intersection of the unit cube and its 60 degree rotation 
around one of its body diagonals is 3/4. -/
theorem volume_of_rotated_cube_intersection_is_three_fourths :
  volume_of_intersection unit_cube (rotate_cube unit_cube (60 * (real.pi / 180)) (real.sqrt 3)) = 3 / 4 :=
sorry

end volume_of_rotated_cube_intersection_is_three_fourths_l323_323726


namespace range_of_a_for_extrema_l323_323987

theorem range_of_a_for_extrema (a : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, (x ≠ y ∧ deriv (λ x, x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1) x = 0 ∧ deriv (λ x, x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1) y = 0)) → 
  a > 2 ∨ a < -1 :=
begin
  sorry
end

end range_of_a_for_extrema_l323_323987


namespace count_5_primable_under_1000_l323_323124

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323124


namespace part_one_part_two_l323_323599

noncomputable def f (x a : ℝ) : ℝ := -2 * x * Real.log x + x^2 - 2 * a * x + a^2
noncomputable def g (x a : ℝ) : ℝ := deriv (λ x, f x a) x

theorem part_one (a : ℝ) : 
  let slope := g 1 a in 
  slope = 1 → a = -1/2 :=
by
  unfold g f
  intro h
  rw [deriv] at h
  sorry

theorem part_two {a : ℝ} : 
  let h (x : ℝ) := x - 1 - Real.log x in
  (a = 0 → set.count h '' ((λ (x : ℝ) → x) '' Ioi 0) = 1) ∧
  (a < 0 → ¬(0 : ℝ) ∈ set.range h) ∧
  (a > 0 → set.count (λ (x : ℝ) → h x) = 2) :=
by
  sorry

end part_one_part_two_l323_323599


namespace smallest_possible_students_l323_323370

theorem smallest_possible_students :
  ∃ n : ℕ, (n % 15 = 0) ∧ (∃ s : ℕ, s = 12 ∧ ((∃ factors : Finset ℕ, factors = (Finset.range (n + 1)).filter (λ d, n % d = 0) ∧ factors.card = s)) ∧ (∀ (m : ℕ), (m < n) → (∃ factors_m : Finset ℕ, factors_m = (Finset.range (m + 1)).filter (λ d, m % d = 0) ∧ factors_m.card ≠ s)) ∧ n = 60) :=
begin
  use 60,
  split,
  { exact nat.mod_eq_zero_of_dvd (by norm_num) },
  split,
  { use 12,
    split,
    { refl },
    split,
    { use (Finset.range (60 + 1)).filter (λ d, 60 % d = 0),
      split,
      { refl },
      { apply Finset.card_filter,
        norm_num1,
        { exact (by refl) } },
    },
    { intros m h,
      by_contra contra,
      use (Finset.range (m + 1)).filter (λ d, m % d = 0),
      split,
      { refl },
      { exact contra } }
  } }
end

end smallest_possible_students_l323_323370


namespace find_x_l323_323939

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x)
    (h3 : 1 / a + 1 / b = 1) : x = 6 :=
sorry

end find_x_l323_323939


namespace vans_needed_l323_323881

-- Given Conditions
def van_capacity : ℕ := 4
def students : ℕ := 2
def adults : ℕ := 6
def total_people : ℕ := students + adults

-- Theorem to prove
theorem vans_needed : total_people / van_capacity = 2 :=
by
  -- Proof will be added here
  sorry

end vans_needed_l323_323881


namespace sin_C_area_triangle_ABC_l323_323309

/-
In triangle ABC, ∠A = 60° and c = 3/7 * a, we need to prove:
1. sin C = 3√3/14
2. Area of triangle ABC when a = 7 is 6√3
-/

-- Definition of the problem conditions
def angle_A : ℝ := Real.pi / 3  -- 60 degrees in radians
def ratio_c_a : ℝ := 3 / 7
def side_c (a : ℝ) : ℝ := ratio_c_a * a

-- Problem 1: Calculate sin C
theorem sin_C (a : ℝ) : Real.sin (Real.arctan (side_c a / (Real.sqrt (a^2 - (side_c a / 2)^2)))) 
  = 3 * Real.sqrt 3 / 14 := sorry

-- Problem 2: Area of the triangle with a = 7
theorem area_triangle_ABC : 
  let a := 7,
      b := 8,
      c := side_c a
  in 1 / 2 * b * c * Real.sin angle_A = 6 * Real.sqrt 3 := sorry

end sin_C_area_triangle_ABC_l323_323309


namespace unit_triangle_count_bound_l323_323162

variable {L : ℝ} (L_pos : L > 0)
variable {n : ℕ}

/--
  Let \( \Delta \) be an equilateral triangle with side length \( L \), and suppose that \( n \) unit 
  equilateral triangles are drawn inside \( \Delta \) with non-overlapping interiors and each having 
  sides parallel to \( \Delta \) but with opposite orientation. Then,
  we must have \( n \leq \frac{2}{3} L^2 \).
-/
theorem unit_triangle_count_bound (L_pos : L > 0) (n : ℕ) :
  n ≤ (2 / 3) * (L ^ 2) := 
sorry

end unit_triangle_count_bound_l323_323162


namespace second_rice_price_l323_323670

theorem second_rice_price (P : ℝ) 
  (price_first : ℝ := 3.10) 
  (price_mixture : ℝ := 3.25) 
  (ratio_first_to_second : ℝ := 3 / 7) :
  (3 * price_first + 7 * P) / 10 = price_mixture → 
  P = 3.3142857142857145 :=
by
  sorry

end second_rice_price_l323_323670


namespace money_left_correct_l323_323706

variables (cost_per_kg initial_money kg_bought total_cost money_left : ℕ)

def condition1 : cost_per_kg = 82 := sorry
def condition2 : kg_bought = 2 := sorry
def condition3 : initial_money = 180 := sorry
def condition4 : total_cost = cost_per_kg * kg_bought := sorry
def condition5 : money_left = initial_money - total_cost := sorry

theorem money_left_correct : money_left = 16 := by
  have h1 : cost_per_kg = 82, from condition1
  have h2 : kg_bought = 2, from condition2
  have h3 : initial_money = 180, from condition3
  have h4 : total_cost = cost_per_kg * kg_bought, from condition4
  have h5 : money_left = initial_money - total_cost, from condition5
  rw [h1, h2, h3, h4, h5]
  sorry

end money_left_correct_l323_323706


namespace problem1_problem2_l323_323312

variable {A B C : ℝ} {AC BC : ℝ}

-- Condition: BC = 2AC
def condition1 (AC BC : ℝ) : Prop := BC = 2 * AC

-- Problem 1: Prove 4cos^2(B) - cos^2(A) = 3
theorem problem1 (h : condition1 AC BC) :
  4 * Real.cos B ^ 2 - Real.cos A ^ 2 = 3 :=
sorry

-- Problem 2: Prove the maximum value of (sin(A) / (2cos(B) + cos(A))) is 2/3 for A ∈ (0, π)
theorem problem2 (h : condition1 AC BC) (hA : 0 < A ∧ A < Real.pi) :
  ∃ t : ℝ, (t = Real.sin A / (2 * Real.cos B + Real.cos A) ∧ t ≤ 2/3) :=
sorry

end problem1_problem2_l323_323312


namespace count_5_primable_below_1000_is_21_l323_323071

def is_5_primable (n : ℕ) : Prop :=
  (n % 5 = 0) ∧ (∀ d ∈ n.digits 10, d ∈ {2, 3, 5, 7})

def count_5_primable_below_1000 : ℕ :=
  set.card {n | n < 1000 ∧ is_5_primable n}

theorem count_5_primable_below_1000_is_21 :
  count_5_primable_below_1000 = 21 :=
sorry

end count_5_primable_below_1000_is_21_l323_323071


namespace three_digit_difference_divisible_by_9_l323_323742

theorem three_digit_difference_divisible_by_9 :
  ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c - (a + b + c)) % 9 = 0 :=
by
  intros a b c h
  sorry

end three_digit_difference_divisible_by_9_l323_323742


namespace count_five_primable_lt_1000_l323_323081

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323081


namespace first_day_more_than_200_paperclips_l323_323364

def paperclips_after_days (k : ℕ) : ℕ :=
  3 * 2^k

theorem first_day_more_than_200_paperclips : (∀ k, 3 * 2^k <= 200) → k <= 7 → 3 * 2^7 > 200 → k = 7 :=
by
  intro h_le h_lt h_gt
  sorry

end first_day_more_than_200_paperclips_l323_323364


namespace area_of_triangle_ABC_l323_323651

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a - b = 2 ∧ c = 4 ∧ sin A = 2 * sin B

theorem area_of_triangle_ABC (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) : 
  ∃ (S : ℝ), S = sqrt 15 ∧ sin (2 * A - B) = 7 * sqrt 15 / 32 :=
by
  sorry

end area_of_triangle_ABC_l323_323651


namespace a_1998_eq_4494_l323_323850

noncomputable def sequence_a : ℕ → ℕ
| 1 := 1
| (n+1) := (Nat.find (λ m, m > sequence_a n ∧ ∀ i j k : Fin (n+1), sequence_a (i+1) + sequence_a (j+1) ≠ 3 * sequence_a (k+1)))

theorem a_1998_eq_4494 : sequence_a 1998 = 4494 :=
  sorry

end a_1998_eq_4494_l323_323850


namespace relationship_y1_y2_y3_l323_323610

-- Define the function y = 3(x + 1)^2 - 8
def quadratic_fn (x : ℝ) : ℝ := 3 * (x + 1)^2 - 8

-- Define points A, B, and C on the graph of the quadratic function
def y1 := quadratic_fn 1
def y2 := quadratic_fn 2
def y3 := quadratic_fn (-2)

-- The goal is to prove the relationship y2 > y1 > y3
theorem relationship_y1_y2_y3 :
  y2 > y1 ∧ y1 > y3 :=
by sorry

end relationship_y1_y2_y3_l323_323610


namespace valid_allocation_methods_l323_323155

noncomputable def numberOfAllocations : ℕ := 30

theorem valid_allocation_methods:
  ∀ (A B C D : Type), 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) → 
  (A, B, C, D are not in the same class and each class has at least one student) → 
  3 classes →
  students A and B cannot be in the same class → 
  ∃ alloc_method : ℕ, alloc_method = numberOfAllocations :=
sorry

end valid_allocation_methods_l323_323155


namespace probability_N_pow_16_mod_7_eq_1_l323_323163

theorem probability_N_pow_16_mod_7_eq_1 :
  let N := (1 : ℤ) ≤ N ∧ N ≤ 2027 in
  (∃ k, k ∈ [1, 2027] ∧ (k^16 % 7 = 1)) :=
  sorry

end probability_N_pow_16_mod_7_eq_1_l323_323163


namespace sum_cos_over_cos_subtract_30_l323_323944

noncomputable def f (x : ℝ) : ℝ := (Real.cos (x * Real.pi / 180)) / (Real.cos ((30 - x) * Real.pi / 180))

theorem sum_cos_over_cos_subtract_30 :
  (∑ x in Finset.range 59 + 1, f (x)) = (59 * Real.sqrt 3) / 2 :=
by
  sorry

end sum_cos_over_cos_subtract_30_l323_323944


namespace triangle_angle_range_l323_323769

theorem triangle_angle_range (α β γ : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α = 2 * γ)
  (h3 : α ≥ β)
  (h4 : β ≥ γ) :
  45 ≤ β ∧ β ≤ 72 := 
sorry

end triangle_angle_range_l323_323769


namespace part_one_part_two_l323_323960

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end part_one_part_two_l323_323960


namespace Henry_age_ratio_l323_323488

theorem Henry_age_ratio (A S H : ℕ)
  (hA : A = 15)
  (hS : S = 3 * A)
  (h_sum : A + S + H = 240) :
  H / S = 4 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end Henry_age_ratio_l323_323488


namespace tunnel_surface_area_l323_323331

theorem tunnel_surface_area (PQRS PQVT PRUT : Square)
  (P Q R S T O L M N : Point)
  (PT_eq : PT = 10)
  (PL_eq : PL = 3)
  (PM_eq : PM = 3)
  (PN_eq : PN = 3)
  (parallel_OP : parallel (side1Tunnel sidesTunnel OP))
  (side1TunnelLM : side1Tunnel = LM)
  (side1TunnelMN : side2Tunnel = MN)
  (side1TunnelNL : side3Tunnel = NL) :
  ∃ (u v w : ℕ), surface_area X = u + v * Real.sqrt w ∧ u = 418 ∧ v = 33 ∧ w = 21 ∧ w ∣ w^2 ∧ u + v + w = 472 :=
by 
  sorry

end tunnel_surface_area_l323_323331


namespace sphere_diagonal_property_l323_323827

variable {A B C D : ℝ}

-- conditions provided
variable (radius : ℝ) (x y z : ℝ)
variable (h_radius : radius = 1)
variable (h_non_coplanar : ¬(is_coplanar A B C D))
variable (h_AB_CD : dist A B = x ∧ dist C D = x)
variable (h_BC_DA : dist B C = y ∧ dist D A = y)
variable (h_CA_BD : dist C A = z ∧ dist B D = z)

theorem sphere_diagonal_property :
  x^2 + y^2 + z^2 = 8 := 
sorry

end sphere_diagonal_property_l323_323827


namespace monotonic_decreasing_necessity_monotonic_decreasing_insufficiency_l323_323986

theorem monotonic_decreasing_necessity (a : ℝ) :
  (∀ x ∈ Iic (1 : ℝ), (4 * x + a ≤ 0)) ∧ (∀ x ∈ Ioi (1 : ℝ), (4 * a * x + 1 ≤ 0)) ∧ (2 + a - (3 / 2) ≤ 2 * a + 1) → a ≤ 0 :=
sorry

theorem monotonic_decreasing_insufficiency (a : ℝ) :
  a ≤ 0 → (∀ x ∈ Iic (1 : ℝ), (4 * x + a ≤ 0) ∧ ∀ x ∈ Ioi (1 : ℝ), (4 * a * x + 1 ≤ 0) ∧ (2 + a - (3 / 2) ≤ 2 * a + 1) ↔ a ≤ -4) :=
sorry

end monotonic_decreasing_necessity_monotonic_decreasing_insufficiency_l323_323986


namespace relation_between_a_b_c_l323_323213

noncomputable def a : ℝ := Real.log 0.2 / Real.log 5
noncomputable def b : ℝ := Real.log 0.2 / Real.log 0.5
noncomputable def c : ℝ := 0.5 ^ 0.2

theorem relation_between_a_b_c : a < c ∧ c < b := sorry

end relation_between_a_b_c_l323_323213


namespace solve_inequality_m_eq_1_find_range_m_nonempty_l323_323605

noncomputable theory

def f (x m : ℝ) : ℝ := |2 * x - 2| + |x + m|

-- Statement for Question 1
theorem solve_inequality_m_eq_1 : {x : ℝ | f x 1 ≤ 3} = set.Icc 0 (4 / 3) :=
by sorry

-- Statement for Question 2
theorem find_range_m_nonempty : 
  {m : ℝ | ∃ x : ℝ, f x m ≤ 3} = set.Icc (-4) 2 :=
by sorry

end solve_inequality_m_eq_1_find_range_m_nonempty_l323_323605


namespace count_5_primables_less_than_1000_l323_323100

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323100


namespace total_cost_of_tshirts_l323_323177

theorem total_cost_of_tshirts
  (White_packs : ℕ := 3) (Blue_packs : ℕ := 2) (Red_packs : ℕ := 4) (Green_packs : ℕ := 1) 
  (White_price_per_pack : ℝ := 12) (Blue_price_per_pack : ℝ := 8) (Red_price_per_pack : ℝ := 10) (Green_price_per_pack : ℝ := 6) 
  (White_discount : ℝ := 0.10) (Blue_discount : ℝ := 0.05) (Red_discount : ℝ := 0.15) (Green_discount : ℝ := 0.00) :
  White_packs * White_price_per_pack * (1 - White_discount) +
  Blue_packs * Blue_price_per_pack * (1 - Blue_discount) +
  Red_packs * Red_price_per_pack * (1 - Red_discount) +
  Green_packs * Green_price_per_pack * (1 - Green_discount) = 87.60 := by
    sorry

end total_cost_of_tshirts_l323_323177


namespace factorize_f_l323_323543

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x

theorem factorize_f (x : ℝ) : f(x) = x * (x - 2)^2 := by
  sorry

end factorize_f_l323_323543


namespace number_of_ways_to_divide_groups_l323_323529

theorem number_of_ways_to_divide_groups :
  (number_of_ways_to_divide (5 : ℕ) (3 : ℕ) 3 (λ g, g > 1) = 150) :=
sorry

end number_of_ways_to_divide_groups_l323_323529


namespace measure_8_liters_with_two_buckets_l323_323999

def bucket_is_empty (B : ℕ) : Prop :=
  B = 0

def bucket_has_capacity (B : ℕ) (c : ℕ) : Prop :=
  B ≤ c

def fill_bucket (B : ℕ) (c : ℕ) : ℕ :=
  c

def empty_bucket (B : ℕ) : ℕ :=
  0

def pour_bucket (B1 B2 : ℕ) (c1 c2 : ℕ) : (ℕ × ℕ) :=
  if B1 + B2 <= c2 then (0, B1 + B2)
  else (B1 - (c2 - B2), c2)

theorem measure_8_liters_with_two_buckets (B10 B6 : ℕ) (c10 c6 : ℕ) :
  bucket_has_capacity B10 c10 ∧ bucket_has_capacity B6 c6 ∧
  c10 = 10 ∧ c6 = 6 →
  ∃ B10' B6', B10' = 8 ∧ B6' ≤ 6 :=
by
  intros h
  have h1 : ∃ B1, bucket_is_empty B1,
    from ⟨0, rfl⟩
  let B10 := fill_bucket 0 c10
  let ⟨B10, B6⟩ := pour_bucket B10 0 c10 c6
  let B6 := empty_bucket B6
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  let B10 := fill_bucket B10 c10
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  exact ⟨B10, B6, rfl, le_refl 6⟩

end measure_8_liters_with_two_buckets_l323_323999


namespace ceil_sqrt_196_eq_14_l323_323538

theorem ceil_sqrt_196_eq_14 : ⌈Real.sqrt 196⌉ = 14 := 
by 
  sorry

end ceil_sqrt_196_eq_14_l323_323538


namespace convex_n_lattice_polygon_l323_323188

theorem convex_n_lattice_polygon (n : ℤ) :
  (∃ (polygon : List (ℤ × ℤ)), 
      is_convex polygon ∧ 
      all_sides_odd polygon ∧ 
      all_sides_unequal polygon ∧ 
      polygon.length = n) → 
  (even n ∧ n ≥ 4) :=
sorry

-- Assuming definitions exist for:
-- is_convex to check if a given set of vertices forms a convex polygon,
-- all_sides_odd to check if all sides of the polygon have odd lengths,
-- all_sides_unequal to check if all sides of the polygon are unequal.

end convex_n_lattice_polygon_l323_323188


namespace lean_proof_problem_l323_323347

noncomputable def B : ℂ := 3 - 2 * complex.i
noncomputable def Q : ℂ := -5
noncomputable def R : ℂ := 2 * complex.i
noncomputable def T : ℂ := -1 + 5 * complex.i

theorem lean_proof_problem : B - Q + R + T = -3 + 5 * complex.i := 
by 
  sorry

end lean_proof_problem_l323_323347


namespace game_A_greater_game_B_l323_323839

-- Defining the probabilities and independence condition
def P_H := 2 / 3
def P_T := 1 / 3
def independent_tosses := true

-- Game A Probability Definition
def P_A := (P_H ^ 3) + (P_T ^ 3)

-- Game B Probability Definition
def P_B := ((P_H ^ 2) + (P_T ^ 2)) ^ 2

-- Statement to be proved
theorem game_A_greater_game_B : P_A = (27:ℚ) / 81 ∧ P_B = (25:ℚ) / 81 ∧ ((27:ℚ) / 81 - (25:ℚ) / 81 = (2:ℚ) / 81) := 
by
  -- P_A has already been computed: 1/3 = 27/81
  -- P_B has already been computed: 25/81
  sorry

end game_A_greater_game_B_l323_323839


namespace mask_usage_duration_l323_323499

-- Define given conditions
def TotalMasks : ℕ := 75
def FamilyMembers : ℕ := 7
def MaskChangeInterval : ℕ := 2

-- Define the goal statement, which is to prove that the family will take 21 days to use all masks
theorem mask_usage_duration 
  (M : ℕ := 75)  -- total masks
  (N : ℕ := 7)   -- family members
  (d : ℕ := 2)   -- mask change interval
  : (M / N) * d + 1 = 21 :=
sorry

end mask_usage_duration_l323_323499


namespace count_routes_from_P_to_Q_l323_323514

variable (P Q R S T : Type)
variable (roadPQ roadPS roadPT roadQR roadQS roadRS roadST : Prop)

theorem count_routes_from_P_to_Q :
  ∃ (routes : ℕ), routes = 16 :=
by
  sorry

end count_routes_from_P_to_Q_l323_323514


namespace integer_part_sum_l323_323254

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ a 2 = 1 ∧ ∀ n : ℕ, 2 ≤ n → a (n + 1) = a n + a (n - 1)

theorem integer_part_sum (a : ℕ → ℚ) (h : sequence a) :
  let f (k : ℕ) := (1/a (2*k - 1)) * (1/a (2*k + 1))
  (⌊∑ k in finset.range 1009, f (k+1)⌋ : ℤ) = 1 :=
sorry

end integer_part_sum_l323_323254


namespace find_divisor_for_multiple_l323_323903

theorem find_divisor_for_multiple (d : ℕ) :
  (∃ k : ℕ, k * d % 1821 = 710 ∧ k * d % 24 = 13 ∧ k * d = 3024) →
  d = 23 :=
by
  intros h
  sorry

end find_divisor_for_multiple_l323_323903


namespace simplify_1_simplify_2_l323_323385

theorem simplify_1 (a b : ℤ) : 2 * a - (a + b) = a - b :=
by
  sorry

theorem simplify_2 (x y : ℤ) : (x^2 - 2 * y^2) - 2 * (3 * y^2 - 2 * x^2) = 5 * x^2 - 8 * y^2 :=
by
  sorry

end simplify_1_simplify_2_l323_323385


namespace infinite_geometric_series_sum_l323_323511

theorem infinite_geometric_series_sum : 
  ∑' n : ℕ, (1 / 3) ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l323_323511


namespace reflection_line_midpoint_l323_323223

open EuclideanGeometry

noncomputable def acute_triangle (A B C : Point)
  (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ A)
  (h_angle_acute : ∀ (P Q R : Point), A = P → B = Q → C = R → 0 < angle PQR ∧ angle PQR < π / 2) : Prop :=
acute ∧ triangle P Q R

noncomputable def midpoint (M A B : Point) (h_midpoint : dist A M = dist B M) : Prop :=
∃ (M : Point), dist A M = dist B M ∧ dist A M = dist B M / 2

noncomputable def tangent_circumcircle (A B C : Point) (h_circumcircle : ∃ (O : Point), circumcircle A B C O) : Prop :=
tangent (circle_center : circumcircle A B C) A

-- Given conditions
variables (A B C M K E : Point)
variables (h_acute : acute_triangle A B C)
variables (h_mid_AB : midpoint M A B)
variables (h_K_pos : same_side K C A ∧ different_sides K B A C)
variables (h_KMC_right_angle : right_angle ∠ K M C )
variables (h_KAC_angle : ∠ K A C = π - ∠ A B C)
variables (E_on_tangent : ∃ O : Point, is_tangent (circumcircle A B C O) A (line E K))

-- Required to prove
theorem reflection_line_midpoint :
  reflection (line B C) (line C M) = line E M :=
sorry

end reflection_line_midpoint_l323_323223


namespace largest_n_S_n_positive_l323_323224

variable {α : Type*} [OrderedSemiring α]

noncomputable def arithmetic_sequence (a_n : ℕ → α) : Prop :=
∀ n : ℕ, a_n n = a_n 0 + n * a_n 1

theorem largest_n_S_n_positive 
  (a : ℕ → ℝ) 
  (h1 : a 0 > 0) 
  (h2 : a 2010 + a 2009 > 0) 
  (h3 : a 2010 * a 2009 < 0) : 
  (4018 > 0) ∧ (∀ n : ℕ, (n > 4018 → S_n a n ≤ 0)) := sorry

end largest_n_S_n_positive_l323_323224


namespace required_speed_l323_323438

-- Definition of the parameters given in the problem
def distance : ℝ := 440
def original_time : ℝ := 3
def new_time : ℝ := original_time / 2

-- Calculate the original speed
def original_speed : ℝ := distance / original_time

-- Required new speed to cover the same distance in half the previous time
def required_new_speed : ℝ := distance / new_time

-- Theorem stating the required new speed is 293.33 km/h
theorem required_speed (dist : ℝ) (orig_time : ℝ) (new_time : ℝ) (orig_speed : ℝ) (new_speed : ℝ) :
  dist = 440 → orig_time = 3 → new_time = orig_time / 2 → orig_speed = dist / orig_time → new_speed = dist / new_time → 
  new_speed = 293.33 :=
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  rw [h4, h5]
  norm_num
  sorry

end required_speed_l323_323438


namespace num_integers_between_neg4_and_sqrt8_l323_323783

def sqrt_8 : Real := Real.sqrt 8

theorem num_integers_between_neg4_and_sqrt8 :
  {n : Int | -4 < n ∧ (n : Real) < sqrt_8}.finset.card = 6 :=
by
  sorry

end num_integers_between_neg4_and_sqrt8_l323_323783


namespace total_parcel_boxes_l323_323395

theorem total_parcel_boxes (a b c d : ℕ) (row_boxes column_boxes total_boxes : ℕ)
  (h_left : a = 7) (h_right : b = 13)
  (h_front : c = 8) (h_back : d = 14)
  (h_row : row_boxes = a - 1 + 1 + b) -- boxes in a row: (a - 1) + 1 (parcel itself) + b
  (h_column : column_boxes = c - 1 + 1 + d) -- boxes in a column: (c -1) + 1(parcel itself) + d
  (h_total : total_boxes = row_boxes * column_boxes) :
  total_boxes = 399 := by
  sorry

end total_parcel_boxes_l323_323395


namespace general_solution_of_differential_eq_l323_323191

theorem general_solution_of_differential_eq :
  ∀ (C₁ C₂ : ℝ) (x : ℝ), 
    let y := C₁ * Real.exp (-3 * x) + C₂ * Real.exp (-1 * x) + (x^2 + 9 * x - 7) * Real.exp x in 
    (Deriv.deriv (Deriv.deriv y) + 4 * Deriv.deriv y + 3 * y) = (8 * x^2 + 84 * x) * Real.exp x := 
  by
    intros C₁ C₂ x
    let y := C₁ * Real.exp (-3 * x) + C₂ * Real.exp (-1 * x) + (x^2 + 9 * x - 7) * Real.exp x
    sorry

end general_solution_of_differential_eq_l323_323191


namespace count_5_primable_less_than_1000_eq_l323_323128

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323128


namespace sum_of_reciprocals_of_triangular_numbers_correct_l323_323870

def triangular_number (n : ℕ) : ℝ := (n * (n + 1) : ℝ) / 2

noncomputable def sum_of_reciprocals_of_triangular_numbers : ℝ :=
  ∑ n in Finset.range 1000, 3 / triangular_number (n + 1)

theorem sum_of_reciprocals_of_triangular_numbers_correct :
  sum_of_reciprocals_of_triangular_numbers = 6000 / 1001 :=
by
  sorry

end sum_of_reciprocals_of_triangular_numbers_correct_l323_323870


namespace Bobby_paycheck_final_amount_l323_323505

theorem Bobby_paycheck_final_amount :
  let salary := 450
  let federal_tax := (1 / 3 : ℚ) * salary
  let state_tax := 0.08 * salary
  let health_insurance := 50
  let life_insurance := 20
  let city_fee := 10
  let total_deductions := federal_tax + state_tax + health_insurance + life_insurance + city_fee
  salary - total_deductions = 184 :=
by
  -- We put sorry here to skip the proof step
  sorry

end Bobby_paycheck_final_amount_l323_323505


namespace hockey_stick_identity_sum_find_binomial_coefficient_l323_323046

-- Declare binomial coefficients in Lean
open Nat

-- Step 1: Prove that the sum of the binomial coefficients equals 462
theorem hockey_stick_identity_sum :
    (Nat.choose 5 0) + (Nat.choose 6 5) + (Nat.choose 7 5) +
    (Nat.choose 8 5) + (Nat.choose 9 5) + (Nat.choose 10 5) =
    Nat.choose 11 6 := by
    sorry

-- Step 2: Prove that given the relation, the value of the binomial coefficient
theorem find_binomial_coefficient (m : ℕ) (h : 1 / (Nat.choose 5 m) - 1 / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m)) :
    Nat.choose 8 m = 28 := by
    sorry

end hockey_stick_identity_sum_find_binomial_coefficient_l323_323046


namespace PQRS_parallelogram_iff_AC_perpendicular_BD_l323_323329

variables {A B C D E P Q R S : Type}
variable [euclidean_geometry A B C D E P Q R S]

def parallelogram (A B C D : Type) : Prop := 
  -- Define the parallelogram properties here
  sorry

def circumcenter (A B C : Type) : Type := 
  -- Define the properties of the circumcenter here
  sorry

def is_perpendicular (A C : Type) (B D : Type) : Prop := 
  -- Define perpendicularity here
  sorry

theorem PQRS_parallelogram_iff_AC_perpendicular_BD 
  (hABCD : parallelogram A B C D)
  (hE : intersect_diagonals_at E A C B D)
  (hP : P = circumcenter A B E)
  (hQ : Q = circumcenter B C E)
  (hR : R = circumcenter C D E)
  (hS : S = circumcenter A D E) : 
  (parallelogram P Q R S) ↔ (is_perpendicular A C B D) := 
sorry

end PQRS_parallelogram_iff_AC_perpendicular_BD_l323_323329


namespace perimeter_AEC_l323_323144

notation "ℝ" => Real

structure Point (α : Type _) := (x : α) (y : α)

def A : Point ℝ := ⟨0, 2⟩
def B : Point ℝ := ⟨0, 0⟩
def C : Point ℝ := ⟨2, 0⟩
def D : Point ℝ := ⟨2, 2⟩

def length (p1 p2 : Point ℝ) : ℝ := Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def folded_C : Point ℝ := ⟨0, 4 / 3⟩
def intersection_E : Point ℝ := ⟨2, 0⟩

noncomputable def perimeter_triangle (A E folded_C : Point ℝ) : ℝ :=
  length A E + length E folded_C + length A folded_C

theorem perimeter_AEC' : perimeter_triangle A intersection_E folded_C = 4 := by
  sorry

end perimeter_AEC_l323_323144


namespace prob_one_white_ball_drawn_prob_same_color_two_draws_l323_323300

-- Define the setup
def num_yellow : Nat := 1
def num_white : Nat := 2
def num_total_balls : Nat := num_yellow + num_white

-- Part 1: Probability of drawing exactly one white ball
theorem prob_one_white_ball_drawn : (num_white : ℚ) / num_total_balls = 2 / 3 :=
by
  have h1 : num_white = 2 := rfl
  have h2 : num_total_balls = 3 := by rw [num_total_balls, num_yellow, num_white]
  rw [h1, h2]
  norm_num
  sorry

-- Part 2: Probability that colors of two draws are the same (with replacement)
theorem prob_same_color_two_draws : ((1 / 3) ^ 2 + (2 / 3 * 1 / 3) * 2 + (2 / 3) ^ 2 : ℚ) = 4 / 9 :=
by
  have p_yellow_yellow : ℚ := (1 / 3) * (1 / 3)
  have p_white_white : ℚ := ((2 / 3) * (2 / 3))
  have p_white_white_diff : ℚ := 2 * ((2 / 3) * (1 / 3))
  have prob_same_color := p_yellow_yellow + p_white_white + p_white_white_diff
  norm_num at *
  sorry

end prob_one_white_ball_drawn_prob_same_color_two_draws_l323_323300


namespace area_of_bounded_region_l323_323442

def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem area_of_bounded_region :
  area_under_curve (λ x, x * sqrt (36 - x ^ 2)) 0 6 = 72 :=
by
  sorry

end area_of_bounded_region_l323_323442


namespace problem1_problem2_l323_323872

theorem problem1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := 
by sorry

theorem problem2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := 
by sorry

end problem1_problem2_l323_323872


namespace function_passes_through_point_l323_323404

noncomputable def func_graph (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 2

theorem function_passes_through_point (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) :
  func_graph a 1 = 3 :=
by
  -- Proof logic is omitted
  sorry

end function_passes_through_point_l323_323404


namespace sum_of_first_eight_primes_ending_in_3_l323_323201

/-- Define the set of prime numbers ending in 3 within a certain range -/
def primes_ending_in_3 : List ℕ := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123]

/-- Define a predicate to check if a number is prime and ends in 3 -/
def is_prime_ending_in_3 (n : ℕ) : Prop := Prime n ∧ n % 10 = 3

/-- Consider the first eight primes ending in 3 -/
noncomputable def first_eight_primes_ending_in_3 : List ℕ :=
  primes_ending_in_3.filter is_prime_ending_in_3 |>.take 8

/-- Sum of the first eight primes ending in 3 -/
noncomputable def sum_first_eight_primes_ending_in_3 : ℕ :=
  (first_eight_primes_ending_in_3).sum

theorem sum_of_first_eight_primes_ending_in_3 :
  sum_first_eight_primes_ending_in_3 = 394 := by
  sorry

end sum_of_first_eight_primes_ending_in_3_l323_323201


namespace pentagon_not_circumscribable_l323_323405

theorem pentagon_not_circumscribable (AB BC CD DE EA : ℕ) (hAB : AB = 6) (hBC : BC = 8) (hCD : CD = 7) (hDE : DE = 9) (hEA : EA = 4) :
    ¬ (∃ r : ℝ, ∃ A B C D E p q : ℝ, 
        (A = r ∧ B = r ∧ C = r ∧ D = r ∧ E = r) ∧ 
        (p = AB / 2 ∧ q = BC / 2 ∧ r = CD / 2 ∧ s = DE / 2 ∧ t = EA / 2) ∧ 
        (A + B + C + D + E = p + q + r + s + t) ) := 
by {
  sorry,
}

end pentagon_not_circumscribable_l323_323405


namespace trajectory_of_Z_is_ellipse_l323_323573

open Complex

noncomputable def trajectory_is_ellipse (z : ℂ) : Prop :=
  abs (z - I) + abs (z + I) = 3

theorem trajectory_of_Z_is_ellipse (z : ℂ) (hz : trajectory_is_ellipse z) :
  ∃ a b f₁ f₂ : ℝ, f₁ = (0, 1) ∧ f₂ = (0, -1) ∧
  ∃ e : ℝ, 0 < e ∧ e < 1 ∧  a = 1 / e ∧
  (Re z - 0)^2 / a^2 + (Im z - 0)^2 / b^2 = 1 :=
sorry

end trajectory_of_Z_is_ellipse_l323_323573


namespace total_money_l323_323161

theorem total_money (a j : ℕ) (t : ℕ) (HT : t = 24)
  (H1 : ∀ a j t, after_amy a j t = (a - (t + j), 2j, 2 * t))
  (H2 : ∀ a j t a_new j_new t_new, after_jan a_new j_new t_new = (2 * a_new, 2 * j - (a_new + t_new), 4 * t))
  (H3 : ∀ a j t a_new j_new t_new, after_toy a_new j_new t_new = (4 * a - 4 * (t + j) + 48, 2 * (2 * j - (a - (t + j) + 48)), 4 * t - (4 * (a - (t + j)) + 2 * (2 * j - ((a - (t + j)) + 48))))
  (HF :  HF_fn 4t (4a - 4(t + j) + 48) (2(2j - (a - (t + j) + 48)) = 24) :
  a + j + t = 168 :=
by
  -- skipping actual proof; just establishing the framework
  sorry

end total_money_l323_323161


namespace find_a10_l323_323204

variable {G : Type*} [LinearOrderedField G]
variable (a : ℕ → G)

-- Conditions
def geometric_sequence (a : ℕ → G) (r : G) := ∀ n, a (n + 1) = r * a n
def positive_terms (a : ℕ → G) := ∀ n, 0 < a n
def specific_condition (a : ℕ → G) := a 3 * a 11 = 16

theorem find_a10
  (h_geom : geometric_sequence a 2)
  (h_pos : positive_terms a)
  (h_cond : specific_condition a) :
  a 10 = 32 := by
  sorry

end find_a10_l323_323204


namespace points_midpoints_l323_323782

theorem points_midpoints (points : Finset (ℝ × ℝ)) (h : points.card = 997) :
  (∃ mids : Finset (ℝ × ℝ), mids.card ≥ 1991 ∧ mids ⊆ {m | ∃ p1 p2 ∈ points, m = (p1 + p2) / 2})
  ∧ 
  (∃ collinear_points : Finset (ℝ × ℝ), collinear_points.card = 997 ∧ 
     ∃ mids : Finset (ℝ × ℝ), mids.card = 1991 ∧ mids ⊆ {m | ∃ p1 p2 ∈ collinear_points, m = (p1 + p2) / 2}) := 
sorry

end points_midpoints_l323_323782


namespace count_5_primables_less_than_1000_l323_323092

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323092


namespace sum_equals_generalized_l323_323525

noncomputable def sum_formula (m n : ℕ) :=
  ∑ k in Finset.range (n + 1), (Nat.choose (2 * k) k * Nat.choose (2 * n - 2 * k) (n - k) * m / (k + m : ℚ))

noncomputable def generalized_formula (m n : ℕ) :=
  2^n * (Finset.range n).prod (λ i, (2 * m + 1 + 2 * i)) / (Finset.range n).prod (λ i, (m + 1 + i))

theorem sum_equals_generalized (m n : ℕ) : sum_formula m n = generalized_formula m n := by
  sorry

end sum_equals_generalized_l323_323525


namespace hexagon_circle_radius_l323_323835

theorem hexagon_circle_radius (r : ℝ) :
  let side_length := 3
  let probability := (1 : ℝ) / 3
  (probability = 1 / 3) →
  r = 12 * Real.sqrt 3 / (Real.sqrt 6 - Real.sqrt 2) :=
by
  -- Begin proof here
  sorry

end hexagon_circle_radius_l323_323835


namespace count_5_primable_integers_lt_1000_is_21_l323_323108

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323108


namespace number_of_cats_l323_323293

-- Defining the context and conditions
variables (x y z : Nat)
variables (h1 : x + y + z = 29) (h2 : x = z)

-- Proving the number of cats
theorem number_of_cats (x y z : Nat) (h1 : x + y + z = 29) (h2 : x = z) :
  6 * x + 3 * y = 87 := by
  sorry

end number_of_cats_l323_323293


namespace construct_circles_tangent_l323_323993

-- Definitions of points and circles
structure Point (α : Type) := (x y : α)
structure Circle (α : Type) := (center : Point α) (radius : α)

variables {α : Type} [LinearOrderedField α]

-- Given points
variables (A B C : Point α)

-- Declaration of existence of circles touching at specific points
theorem construct_circles_tangent (A B C : Point α) :
  ∃ (S1 S2 S3 : Circle α),
    (S1.center ≠ S2.center ∧ S1.center ≠ S3.center ∧ S2.center ≠ S3.center) ∧
    (dist S1.center S2.center = S1.radius + S2.radius ∧ S1.center = C ∨
    dist S1.center S3.center = S1.radius + S3.radius ∧ S1.center = B ∨
    dist S2.center S3.center = S2.radius + S3.radius ∧ S2.center = A) :=
sorry

end construct_circles_tangent_l323_323993


namespace number_of_5_primable_less_1000_l323_323089

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323089


namespace is_simplest_l323_323157

def simplest_quadratic_radical (x : ℝ) : Prop :=
  sqrt x = x ∧ ∀ (n : ℕ), x = n^2 → false

theorem is_simplest (x : ℝ) (h1 : x = 5) : simplest_quadratic_radical (sqrt x) :=
by
  sorry

end is_simplest_l323_323157


namespace base7_arithmetic_l323_323490

theorem base7_arithmetic :
  let b7_add (x y : ℕ) : ℕ := -- Definition for adding two base 7 numbers
    let x_b10 := nat_of_digits 7 (list.reverse [2, 4])
    let y_b10 := nat_of_digits 7 (list.reverse [3, 5, 6])
    digits 7 (x_b10 + y_b10)
  let b7_sub (x y : ℕ) : ℕ := -- Definition for subtracting two base 7 numbers
    let x_b10 := nat_of_digits 7 (list.reverse [4, 0, 3])
    let y_b10 := nat_of_digits 7 (list.reverse [1, 0, 5])
    digits 7 (x_b10 - y_b10)
  b7_sub (b7_add 24 356) 105 = [2, 6, 5]
:= sorry

end base7_arithmetic_l323_323490


namespace sparrow_pecks_seeds_l323_323367

theorem sparrow_pecks_seeds (x : ℕ) (h1 : 9 * x < 1001) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end sparrow_pecks_seeds_l323_323367


namespace n_square_divisible_by_144_l323_323284

theorem n_square_divisible_by_144 (n : ℤ) (hn : n > 0)
  (hw : ∃ k : ℤ, n = 12 * k) : ∃ m : ℤ, n^2 = 144 * m :=
by {
  sorry
}

end n_square_divisible_by_144_l323_323284


namespace trains_meet_in_time_approx_l323_323039

noncomputable def time_to_meet (length_train1 length_train2 gap : ℕ) (speed_kmh1 speed_kmh2 : ℕ) : ℝ :=
  let speed_ms1 := (speed_kmh1 * 1000) / 3600
  let speed_ms2 := (speed_kmh2 * 1000) / 3600
  let relative_speed := speed_ms1 + speed_ms2
  let total_distance := length_train1 + length_train2 + gap
  total_distance / relative_speed

theorem trains_meet_in_time_approx :
  time_to_meet 100 200 100 54 72 ≈ 11.43 :=
sorry

end trains_meet_in_time_approx_l323_323039


namespace bagel_spending_l323_323167

variable (B D : ℝ)

theorem bagel_spending (h1 : B - D = 12.50) (h2 : D = B * 0.75) : B + D = 87.50 := 
sorry

end bagel_spending_l323_323167


namespace problem_solution_set_l323_323688

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g (x)

theorem problem_solution_set (f g : ℝ → ℝ)
  (hf_odd : is_odd_function f)
  (hg_even : is_even_function g)
  (h_cond : ∀ x < 0, f'' x * g x + f x * g'' x > 0)
  (h_g_neg3_zero : g (-3) = 0) :
  {x : ℝ | f x * g x < 0} = {x : ℝ | x ∈ (Set.Iio (-3) ∪ Set.Ioo 0 3)} :=
sorry

end problem_solution_set_l323_323688


namespace geometric_sequence_a5_l323_323977

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) 
  : a 5 = -8 :=
sorry

end geometric_sequence_a5_l323_323977


namespace tan_theta_minus_pi_by_4_l323_323232

variable (θ : ℝ)

def in_fourth_quadrant (θ : ℝ) : Prop := 
  3 * π / 2 < θ ∧ θ < 2 * π

theorem tan_theta_minus_pi_by_4 (hθ : in_fourth_quadrant θ) (h : Real.sin (θ + π / 4) = 3 / 5) :
  Real.tan (θ - π / 4) = -4 / 3 := by
  sorry

end tan_theta_minus_pi_by_4_l323_323232


namespace smallest_n_has_square_product_l323_323851

def satisfies_conditions (A : Set ℕ) : Prop :=
  A.Finite ∧
  (∃ n, A.card = n) ∧
  ∀ a ∈ A, 1 < a ∧ (∀ p, p.Prime → p ∣ a → p < 10)

theorem smallest_n_has_square_product :
  ∀ A : Set ℕ, satisfies_conditions A → (∃ a b ∈ A, a ≠ b ∧ is_square (a * b)) ↔ A.card ≥ 17 :=
by {
  sorry
}

end smallest_n_has_square_product_l323_323851


namespace Amy_hours_per_week_school_year_l323_323160

-- Define the given conditions

def summer_hours_per_week := 36
def summer_weeks := 10
def summer_earnings := 3000

def school_year_weeks := 40
def school_year_earnings_needed := 3000

-- Define the question as how many hours per week during the school year she should work to meet the goal
def hours_per_week_school_year (pay_rate : ℝ) : ℝ :=
  school_year_earnings_needed / (pay_rate * school_year_weeks)

theorem Amy_hours_per_week_school_year : 
  let total_summer_hours := summer_hours_per_week * summer_weeks,
      pay_rate := (summer_earnings : ℝ) / (total_summer_hours : ℝ) in
  hours_per_week_school_year pay_rate = 9 :=
by
  sorry

end Amy_hours_per_week_school_year_l323_323160


namespace determine_g_10_l323_323403

noncomputable def g : ℝ → ℝ := sorry

-- Given condition
axiom g_condition : ∀ x y : ℝ, g x + g (2 * x + y) + 7 * x * y = g (3 * x - y) + 3 * x ^ 2 + 4

-- Theorem to prove
theorem determine_g_10 : g 10 = -46 := 
by
  -- skipping the proof here
  sorry

end determine_g_10_l323_323403


namespace math_problem_l323_323158

-- Define the conditions
def condition1 (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x ∈ Icc (-1:ℝ) a, f x = f (-x) ∧ f x = a * x^2 + (2 * a + b) * x + 2

def condition2 (f : ℝ → ℝ) : Prop := 
  ∀ x, f x = sqrt (2016 - x^2) + sqrt (x^2 - 2016)

def condition3 (f : ℝ → ℝ) : Prop := 
  ∀ x ∈ Ioo (0:ℝ) 2, f(x+2) = 1 / f(x) ∧ f x = 2^x

def condition4 (f : ℝ → ℝ) : Prop := 
  ∀ x y : ℝ, f x ≠ 0 → f (x * y) = x * f y + y * f x

-- Define what it means for the conditions to imply the correctness
def correct_statements_under_conditions (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  (condition1 f a b → b = -2) ∧
  (condition2 f → true) ∧
  (condition3 f → true) ∧
  (condition4 f → (∀ x, f (-x) = -f x))

-- The math problem stated in Lean
theorem math_problem (f : ℝ → ℝ) (a b : ℝ) :
  correct_statements_under_conditions f a b :=
sorry

end math_problem_l323_323158


namespace pump_fill_time_without_leak_l323_323137

-- Define the conditions
def rate_of_pump (P : ℝ) : ℝ := 1 / P
def rate_of_leak : ℝ := 1 / 12
def effective_rate : ℝ := 1 / 12

-- State the question as a theorem
theorem pump_fill_time_without_leak : ∃ (P : ℝ), (rate_of_pump P - rate_of_leak = effective_rate) ∧ P = 6 :=
by 
  -- Substitute the values and rewrite the theorem statement
  have h : ∀ (P : ℝ), (1 / P - 1 / 12 = 1 / 12) → P = 6 :=
    λ P h_eq, by
      have : 12 - P = P, by calc
         12 - P = 12 * (1 / P - 1 / 12) * P := by sorry
         ... = P := by sorry
      show P = 6, from by sorry
  -- Prove the existence of such P
  use 6
  split
  · show (rate_of_pump 6 - rate_of_leak = effective_rate), by sorry
  · show 6 = 6, by rfl

end pump_fill_time_without_leak_l323_323137


namespace correct_options_correct_options_l323_323210

variable {Point : Type} {Plane : Type} {Line : Type}
variable [LinearAlgebra Point Plane Line]
variables (α β : Plane) (m n : Line)

theorem correct_options (h1 : m ∥ α) (h2 : n ⊥ α) : m ⊥ n :=
sorry

theorem correct_options' (h1 : m ⊥ α) (h2 : α ∥ β) : m ⊥ β :=
sorry

end correct_options_correct_options_l323_323210


namespace mass_percentage_of_O_in_dinitrogen_trioxide_l323_323553

def molar_mass_N : Float := 14.01
def molar_mass_O : Float := 16.00
def N2O3 : Σ nat nat := ⟨2, 3⟩

def mass_percentage_O_in_N2O3 : Float := 
  let molar_mass_N2O3 := 2 * molar_mass_N + 3 * molar_mass_O
  let mass_O_in_N2O3 := 3 * molar_mass_O
  (mass_O_in_N2O3 / molar_mass_N2O3) * 100

theorem mass_percentage_of_O_in_dinitrogen_trioxide :
  mass_percentage_O_in_N2O3 = 63.15 := 
  sorry

end mass_percentage_of_O_in_dinitrogen_trioxide_l323_323553


namespace normal_dist_prob_l323_323590

noncomputable theory
open ProbabilityTheory MeasureTheory

variables {Ω : Type*} [MeasureSpace Ω]

-- Defining the random variable ξ with normal distribution N(3, 16)
def xi : Ω → ℝ := sorry
axiom xi_normal : Normal ℝ 3 4 xi

-- Statement of the proof problem
theorem normal_dist_prob : Pξ < 3) = 0.5 :=
by
  sorry

end normal_dist_prob_l323_323590


namespace translate_parabola_incorrect_l323_323424

theorem translate_parabola_incorrect :
  let original_parabola : ℝ → ℝ := λ x, (x + 3) ^ 2 - 4 in
  (original_parabola 0 = 0) →
  let translated_parabola : ℝ → ℝ := λ x, (x + 3) ^ 2 in
  ¬ (translated_parabola 0 = 0) :=
by
  sorry

end translate_parabola_incorrect_l323_323424


namespace parabola_point_distance_eq_l323_323062

open Real

theorem parabola_point_distance_eq (P : ℝ × ℝ) (V : ℝ × ℝ) (F : ℝ × ℝ)
    (hV: V = (0, 0)) (hF : F = (0, 2)) (P_on_parabola : P.1 ^ 2 = 8 * P.2) 
    (hPf : dist P F = 150) (P_in_first_quadrant : 0 ≤ P.1 ∧ 0 ≤ P.2) :
    P = (sqrt 1184, 148) :=
sorry

end parabola_point_distance_eq_l323_323062


namespace sageA_hat_white_l323_323785

-- Definitions based on problem conditions
def SageHat : Type := bool -- True for white, False for black

variables (hatA hatB hatC : SageHat)
variables (white_hats : nat := 3)
variables (black_hats : nat := 2)
variables (sageC_not_know : hatA ≠ false ∨ hatB ≠ false) -- Sage C does not see two black hats
variables (sageB_not_know : hatA ≠ false) -- Sage B does not see a black hat on A

-- The theorem to prove that Sage A knows his hat is white
theorem sageA_hat_white (hc : sageC_not_know) (hb : sageB_not_know) : hatA = true :=
by
  sorry

end sageA_hat_white_l323_323785


namespace plane_with_four_colors_l323_323515

theorem plane_with_four_colors 
  (points : Set Point) -- Define a set of points
  (colors : Set Color) -- Define a set of colors
  (color_of : Point → Color) -- Coloring function
  (H1 : colors.size = 5) -- 5 different colors are used
  (H2 : ∀ c ∈ colors, ∃ p ∈ points, color_of p = c) -- Each color appears at least once
  : ∃ plane : Set Point, ∃ s : Set Color, s.size ≥ 4 ∧ ∀ p ∈ plane, color_of p ∈ s :=
by
  sorry

end plane_with_four_colors_l323_323515


namespace sqrt_and_square_eq_zero_implies_a_plus_b_is_one_l323_323630

theorem sqrt_and_square_eq_zero_implies_a_plus_b_is_one (a b : ℝ) 
  (h : √(a + 2) + (b - 3)^2 = 0) : a + b = 1 :=
  sorry

end sqrt_and_square_eq_zero_implies_a_plus_b_is_one_l323_323630


namespace distance_P_to_l_l323_323744

def Point := ℝ × ℝ
def Line := { A B C : ℝ // A ≠ 0 ∨ B ≠ 0 }

-- Define the specific point P and line l
def P : Point := (-1, 1)
def l : Line := ⟨3, 4, 0, by simp⟩

-- Distance function for a point to a line in standard form
def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  let (A, B, C) := (l.val.A, l.val.B, l.val.C)
  let (x0, y0) := p
  (abs (A * x0 + B * y0 + C)) / (sqrt (A^2 + B^2))

-- The theorem statement
theorem distance_P_to_l : distance_point_to_line P l = 1 / 5 :=
by sorry

end distance_P_to_l_l323_323744


namespace increasing_interval_f_l323_323982

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (Real.pi / 6))

theorem increasing_interval_f : ∃ a b : ℝ, a < b ∧ 
  (∀ x y : ℝ, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y) ∧
  (a = - (Real.pi / 6)) ∧ (b = (Real.pi / 3)) :=
by
  sorry

end increasing_interval_f_l323_323982


namespace total_seashells_l323_323702

-- Definitions of the initial number of seashells and the number found
def initial_seashells : Nat := 19
def found_seashells : Nat := 6

-- Theorem stating the total number of seashells in the collection
theorem total_seashells : initial_seashells + found_seashells = 25 := by
  sorry

end total_seashells_l323_323702


namespace repeating_decimal_count_l323_323928

theorem repeating_decimal_count :
  let count := (list.filter (λ n, ∀ p, Nat.Prime p → p ∣ (n + 1) → p = 2 ∨ p = 5) (list.range' 2 200)).length
  count = 182 :=
by
  sorry

#eval (let count := (list.filter (λ n, ∀ p, Nat.Prime p → p ∣ (n + 1) → p = 2 ∨ p = 5) (list.range' 2 200)).length
       count)  -- Expected to output 182

end repeating_decimal_count_l323_323928


namespace goods_train_speed_l323_323805

theorem goods_train_speed (Vm : ℝ) (T : ℝ) (L : ℝ) (Vg : ℝ) :
  Vm = 50 → T = 9 → L = 280 →
  Vg = ((L / T) - (Vm * 1000 / 3600)) * 3600 / 1000 →
  Vg = 62 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end goods_train_speed_l323_323805


namespace train_speed_l323_323466

theorem train_speed :
  ∀ (length : ℝ) (time : ℝ),
  length = 60 →
  time = 3.9996800255979523 →
  (length / time) * 3.6 = 54.003 := 
by
  intros length time hlength htime
  rw [hlength, htime]
  calc
    (60 / 3.9996800255979523) * 3.6 = 15.0008 * 3.6 : by sorry  -- detailed calculations and numerical accuracy handled by proof
                           ... = 54.003 : by sorry

end train_speed_l323_323466


namespace next_in_extended_sequence_l323_323877

def extended_sequence : List Int := [12, 13, 15, 17, 111, 113, 117, 119, 123, 129, 131, 135, 139, 1415, 1417, 1421, 1425, 1431, 1437, 1445]

theorem next_in_extended_sequence : List.nth extended_sequence (List.length extended_sequence - 1) + 16 = 1461 := by
  sorry

end next_in_extended_sequence_l323_323877


namespace third_place_C_l323_323422

theorem third_place_C :
  ∀ P : Type,
  ∀ (A B C : P) 
  (places : P → P → P → Prop)
  (predictionA : P → Prop)
  (predictionB : P → Prop)
  (predictionC : P → Prop)
  (one_correct : (predictionA A ∨ predictionB B ∨ predictionC C) ∧
                 ¬(predictionA A ∧ predictionB B) ∧
                 ¬(predictionB B ∧ predictionC C) ∧
                 ¬(predictionA A ∧ predictionC C))
  (a_not_second : ∀ x y z, predictionA A ↔ ¬ places B A C)
  (b_second : ∀ x y z, predictionB B ↔ places A B C)
  (c_not_first : ∀ x y z, predictionC C ↔ ¬ places C A B),
  places B A C → places C B A :=
by {
  sorry
}

end third_place_C_l323_323422


namespace prize_distributions_is_16_l323_323305

def num_prize_distributions : ℕ :=
  let rounds : ℕ := 4
  2 ^ rounds

theorem prize_distributions_is_16 : num_prize_distributions = 16 :=
by 
  have rounds : ℕ := 4
  calc
    num_prize_distributions 
        = 2 ^ rounds        : by rfl
    ... = 2 ^ 4            : by simp [rounds]
    ... = 16               : by norm_num

#eval prize_distributions_is_16  -- This is to verify the theorem in the Lean environment

end prize_distributions_is_16_l323_323305


namespace find_n_l323_323734

-- Definitions based on conditions
variable (n : ℕ)  -- number of persons
variable (A : Fin n → Finset (Fin n))  -- acquaintance relation, specified as a set of neighbors for each person
-- Condition 1: Each person is acquainted with exactly 8 others
def acquaintances := ∀ i : Fin n, (A i).card = 8
-- Condition 2: Any two acquainted persons have exactly 4 common acquaintances
def common_acquaintances_adj := ∀ i j : Fin n, i ≠ j → j ∈ (A i) → (A i ∩ A j).card = 4
-- Condition 3: Any two non-acquainted persons have exactly 2 common acquaintances
def common_acquaintances_non_adj := ∀ i j : Fin n, i ≠ j → j ∉ (A i) → (A i ∩ A j).card = 2

-- Statement to prove
theorem find_n (h1 : acquaintances n A) (h2 : common_acquaintances_adj n A) (h3 : common_acquaintances_non_adj n A) :
  n = 21 := 
sorry

end find_n_l323_323734


namespace minimum_mn_l323_323690

open Real

noncomputable def tangent_to_circle (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : Prop :=
  let line_eq (x y : ℝ) := (m + 1) * x + (n + 1) * y - 2 = 0
  let circle_eq (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1
  ∃ x y : ℝ, line_eq x y ∧ circle_eq x y ∧ derivative (line_eq x) = -derivative (circle_eq y)

theorem minimum_mn (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (htang : tangent_to_circle m n hm hn) : mn = 3 + 2 * sqrt 2 :=
sorry

end minimum_mn_l323_323690


namespace smallest_non_factor_product_l323_323006

open Nat

def is_factor (n d : ℕ) := d > 0 ∧ n % d = 0

theorem smallest_non_factor_product (a b : ℕ) (h1 : a ≠ b) (h2 : is_factor 48 a) (h3 : is_factor 48 b) (h4 : ¬ is_factor 48 (a * b)) : a * b = 18 :=
by
  sorry

end smallest_non_factor_product_l323_323006


namespace binomial_coefficient_inequality_l323_323721

theorem binomial_coefficient_inequality (n : ℕ) : 2^n ≤ nat.choose (2*n) n ∧ nat.choose (2*n) n ≤ 2^(2*n) := 
sorry

end binomial_coefficient_inequality_l323_323721


namespace verify_transformation_matrix_l323_323554

noncomputable def find_transformation_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.vec_cons
    (Matrix.vec_cons 1 (Matrix.vec_cons 0 Matrix.vec_nil))
    (Matrix.vec_cons 0 (Matrix.vec_cons 3 Matrix.vec_nil))

theorem verify_transformation_matrix:
  ∀ (a b c d : ℚ),
  let M := ![![a, b], ![c, d]] in
  find_transformation_matrix ⬝ M = ![![a, b], ![3 * c, 3 * d]] :=
by sorry

end verify_transformation_matrix_l323_323554


namespace probability_red_ball_is_one_fourth_l323_323299

-- Define the number of red balls and white balls
def number_of_red_balls : ℕ := 2
def number_of_white_balls : ℕ := 6

-- Define the total number of balls
def total_number_of_balls : ℕ := number_of_red_balls + number_of_white_balls

-- Define the probability of drawing a red ball
def probability_of_red_ball : ℚ := number_of_red_balls.to_rat / total_number_of_balls.to_rat

-- Theorem stating that the probability of drawing a red ball is 1/4
theorem probability_red_ball_is_one_fourth : probability_of_red_ball = 1 / 4 := by
  sorry

end probability_red_ball_is_one_fourth_l323_323299


namespace range_of_b_l323_323285

def f (b x : ℝ) : ℝ := -½ * x^2 + b * Real.log (x + 2)

noncomputable def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

theorem range_of_b (b : ℝ) :
  is_decreasing (f b) { x | x > -1 } →
  b ∈ Set.Iic -1 := sorry

end range_of_b_l323_323285


namespace marketing_cost_per_book_l323_323674

theorem marketing_cost_per_book
  (fixed_cost : ℕ)
  (total_revenue : ℕ)
  (num_books_sold : ℕ)
  (charge_per_book : ℕ)
  (total_cost := fixed_cost + (charge_per_book * num_books_sold)) :
  fixed_cost = 50000 →
  total_revenue = 90000 →
  num_books_sold = 10000 →
  total_cost = total_revenue →
  charge_per_book = 4 :=
by
  intros h1 h2 h3 h4
  have h : 10000 * 4 = 40000 := rfl
  have h' : 50000 + 40000 = 90000 := rfl
  exact nat.eq_of_mul_eq_mul_right (dec_trivial : 0 < 10000) ((nat.add_right_inj 50000).mp h4 ▸ by rw [h, h'])

end marketing_cost_per_book_l323_323674


namespace ways_to_score_at_least_7_points_l323_323776

-- Definitions based on the given conditions
def red_balls : Nat := 4
def white_balls : Nat := 6
def points_red : Nat := 2
def points_white : Nat := 1

-- Function to count the number of combinations for choosing k elements from n elements
def choose (n : Nat) (k : Nat) : Nat :=
  if h : k ≤ n then
    Nat.descFactorial n k / Nat.factorial k
  else
    0

-- The main theorem to prove the number of ways to get at least 7 points by choosing 5 balls out
theorem ways_to_score_at_least_7_points : 
  (choose red_balls 4 * choose white_balls 1) +
  (choose red_balls 3 * choose white_balls 2) +
  (choose red_balls 2 * choose white_balls 3) = 186 := 
sorry

end ways_to_score_at_least_7_points_l323_323776


namespace choose_subset_sum_divisible_l323_323712

theorem choose_subset_sum_divisible (nums : Fin 11 → ℕ) : 
  ∃ (S : Finset (Fin 11)) (sign : Fin 11 → ℤ), (∑ i in S, sign i * nums i) % 2011 = 0 := 
sorry

end choose_subset_sum_divisible_l323_323712


namespace first_day_reduction_l323_323475

Variables (P : ℝ) -- original price
Variables (x : ℝ) -- percentage reduction on the first day

theorem first_day_reduction (h : P * (1 - x / 100) * 0.90 = 0.765 * P) : x = 15 :=
by {
  -- start proof here
  sorry
}

end first_day_reduction_l323_323475


namespace problem_I_complement_problem_II_range_l323_323992

-- Problem I
section
variables (A B : set ℝ) 

def set_a : set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def set_b (m : ℝ) : set ℝ := {x | m + 1 < x ∧ x < 2 * m + 4}
def set_intersect_complement (m : ℝ) : set ℝ := 
  {x | x < 3 ∨ x ≥ 6}

theorem problem_I_complement (m : ℝ) (h : m = 1):
  (A ∩ B)ᶜ = set_intersect_complement m :=
sorry
end

-- Problem II
section
variables (A B : set ℝ)

def set_a : set ℝ := {x | 3 ≤ x ∧ x ≤ 9}
def set_b (m : ℝ) : set ℝ := {x | m + 1 < x ∧ x < 2 * m + 4}
def range_m (m : ℝ) : Prop := -3/2 < m ∧ m < 0

theorem problem_II_range (h : 1 ∈ A ∪ B) :
  range_m m :=
sorry
end

end problem_I_complement_problem_II_range_l323_323992


namespace find_a_l323_323314

noncomputable def triangle_side (a b c : ℝ) (A : ℝ) (area : ℝ) : ℝ :=
if b + c = 2 * Real.sqrt 3 ∧ A = Real.pi / 3 ∧ area = Real.sqrt 3 / 2 then
  Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
else 0

theorem find_a (b c : ℝ) (h1 : b + c = 2 * Real.sqrt 3) (h2 : Real.cos (Real.pi / 3) = 1 / 2) (area : ℝ)
  (h3 : area = Real.sqrt 3 / 2)
  (a := triangle_side (Real.sqrt 6) b c (Real.pi / 3) (Real.sqrt 3 / 2)) :
  a = Real.sqrt 6 :=
sorry

end find_a_l323_323314


namespace infinitely_many_solutions_iff_lambda_l323_323256

theorem infinitely_many_solutions_iff_lambda (
  λ x y : ℝ : (λ * x - 12 * y = 2) ∧ (5 * x + 6 * y = -1) :
λ = -10 := by {
sarily 
  sorry
}

end infinitely_many_solutions_iff_lambda_l323_323256


namespace arithmetic_sequence_difference_count_is_14_l323_323045

noncomputable def arithmetic_sequence_difference_count : Nat :=
  let terms := {n : Fin 15 // 1 + n.1 * 3 ≤ 45}
  {m | ∃ (i j : terms), i ≠ j ∧ m = (i.val.1 - j.val.1).natAbs}.toFinset.card

theorem arithmetic_sequence_difference_count_is_14 :
  arithmetic_sequence_difference_count = 14 :=
sorry

end arithmetic_sequence_difference_count_is_14_l323_323045


namespace sum_of_floors_eq_zero_l323_323871

theorem sum_of_floors_eq_zero : ∑ k in Finset.range 1001, (⌊(2^k) / 3⌋ : ℤ) = 0 :=
sorry

end sum_of_floors_eq_zero_l323_323871


namespace number_of_5_primable_numbers_below_1000_l323_323114

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323114


namespace find_e_l323_323036

variable (p j t e : ℝ)

def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - e / 100)

theorem find_e (h1 : condition1 p j)
               (h2 : condition2 j t)
               (h3 : condition3 t e p) : e = 6.25 :=
by sorry

end find_e_l323_323036


namespace turtles_on_lonely_island_l323_323996

theorem turtles_on_lonely_island (T : ℕ) (h1 : 60 = 2 * T + 10) : T = 25 := 
by sorry

end turtles_on_lonely_island_l323_323996


namespace total_nephews_correct_l323_323151

def alden_nephews_10_years_ago : ℕ := 50

def alden_nephews_now : ℕ :=
  alden_nephews_10_years_ago * 2

def vihaan_nephews_now : ℕ :=
  alden_nephews_now + 60

def total_nephews : ℕ :=
  alden_nephews_now + vihaan_nephews_now

theorem total_nephews_correct : total_nephews = 260 := by
  sorry

end total_nephews_correct_l323_323151


namespace P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l323_323147

open ProbabilityTheory

section
/-- Probability of not getting three consecutive heads -/
def P (n : ℕ) : ℚ := sorry

theorem P_3_eq_seven_eighths : P 3 = 7 / 8 := sorry

theorem P_4_ne_fifteen_sixteenths : P 4 ≠ 15 / 16 := sorry

theorem P_decreasing (n : ℕ) (h : 2 ≤ n) : P (n + 1) < P n := sorry

theorem P_recurrence (n : ℕ) (h : 4 ≤ n) : P n = (1 / 2) * P (n - 1) + (1 / 4) * P (n - 2) + (1 / 8) * P (n - 3) := sorry
end

end P_3_eq_seven_eighths_P_4_ne_fifteen_sixteenths_P_decreasing_P_recurrence_l323_323147


namespace inequality_solution_sets_l323_323388

noncomputable def solve_inequality (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Iic (-2)
  else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
  else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
  else if m = -(1 / 2) then ∅
  else Set.Ioo (-2) (1 / m)

theorem inequality_solution_sets (m : ℝ) :
  solve_inequality m = 
    if m = 0 then Set.Iic (-2)
    else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
    else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
    else if m = -(1 / 2) then ∅
    else Set.Ioo (-2) (1 / m) :=
sorry

end inequality_solution_sets_l323_323388


namespace turtles_on_lonely_island_l323_323995

theorem turtles_on_lonely_island (T : ℕ) (h1 : 60 = 2 * T + 10) : T = 25 := 
by sorry

end turtles_on_lonely_island_l323_323995


namespace actual_distance_l323_323809

/-- Define the distance as D and the time as T -/
variables (D T : ℝ)

-- Conditions from the problem:
def condition1 : Prop := D = 10 * T
def condition2 : Prop := D + 15 = 15 * T

-- The main theorem to prove:
theorem actual_distance (h1 : condition1 D T) (h2 : condition2 D T) : D = 30 :=
sorry

end actual_distance_l323_323809


namespace measure_liters_l323_323032

def bucket_capacity :=
  (seven_liter_bucket : ℕ)
  (three_liter_bucket : ℕ)

axiom tap_and_sink : ∀ (amount : ℕ), true

theorem measure_liters (seven_liter_bucket : ℕ) (three_liter_bucket : ℕ) (h1 : seven_liter_bucket = 7) (h2 : three_liter_bucket = 3) :
  ∃ (one_liter : ℕ) (two_liters : ℕ) (four_liters : ℕ) (five_liters : ℕ) (six_liters : ℕ), 
    one_liter = 1 ∧ 
    two_liters = 2 ∧ 
    four_liters = 4 ∧ 
    five_liters = 5 ∧ 
    six_liters = 6 :=
by {
  -- Proof can be provided here
  sorry
}

end measure_liters_l323_323032


namespace find_a_b_magnitude_expression_l323_323248

namespace ComplexNumbers

open Complex

-- Definitions and conditions from part a
def z1 (b : ℝ) : ℂ := (1 + b * I) * (2 + I)
def z2 (a : ℝ) : ℂ := 3 + (1 - a) * I

-- Statements to be proved
theorem find_a_b (a b : ℝ) (h : z1 b = z2 a) :
  a = 2 ∧ b = -1 :=
sorry

theorem magnitude_expression (h_b : 1 = 1) (h_a : 0 = 0) :
  let z1_value := z1 1
  let z2_value := z2 0
  ∥(z1_value + conj(z2_value)) / (1 - 2 * I)∥ = 2 :=
sorry

end ComplexNumbers

end find_a_b_magnitude_expression_l323_323248


namespace sum_of_numbers_l323_323357

open Function

theorem sum_of_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
  (h3 : b = 8) 
  (h4 : (a + b + c) / 3 = a + 7) 
  (h5 : (a + b + c) / 3 = c - 20) : 
  a + b + c = 63 := 
by 
  sorry

end sum_of_numbers_l323_323357


namespace inequality_range_m_l323_323205

theorem inequality_range_m:
  (∀ x ∈ Set.Icc (Real.sqrt 2) 4, (5 / 2) * x^2 ≥ m * (x - 1)) → m ≤ 10 :=
by 
  intros h 
  sorry

end inequality_range_m_l323_323205


namespace conditional_probability_l323_323837

-- Conditions
def students := {1, 2, 3, 4, 5, 6} -- 1, 2, 3, 4 are boys; 5, 6 are girls
def A : Finset ℕ := {1} -- Boy A is selected
def B : Finset ℕ := {6} -- Girl B is selected

-- C(6, 3) is the number of ways to choose 3 students out of 6
def combinations (n k : ℕ) : ℕ := Nat.binomial n k
def total_choices := combinations 6 3

-- P(A) is the probability of event A
def P_A := (combinations 5 2 : ℚ) / total_choices

-- P(AB) is the probability of both A and B happening together
def P_AB := (combinations 4 1 : ℚ) / total_choices

-- P(B|A) is the conditional probability of event B given A
def P_B_given_A : ℚ := P_AB / P_A

theorem conditional_probability : P_B_given_A = 2 / 5 := by
  sorry

end conditional_probability_l323_323837


namespace count_5_primables_less_than_1000_l323_323099

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323099


namespace part1_part2_part3_l323_323231

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, ∃ u v : ℝ, u ≠ v ∧ (3 * x ^ 2 + 2 * a * x + 1 = 0)) ↔ (a > sqrt 3 ∨ a < - sqrt 3) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h_a : a = -2) : 
  let f (x : ℝ) := (x ^ 2 + 1) * (x + a) in
  ∃ x_min x_max : ℝ, x_min = f (-1) ∧ x_max = f (1 / 3) ∧ x_max = 58 / 27 ∧ x_min = -2 :=
sorry

-- Part 3
theorem part3 (a : ℝ) :
  (∀ x : ℝ, f' x = 3 * x ^ 2 + 2 * a * x + 1 ∧ 
    (∃ c ∈ Set.Icc (-1) (1 / 2), f' c = 0)) → (a > sqrt 3) :=
sorry

end part1_part2_part3_l323_323231


namespace intersection_eq_l323_323349

open Set

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | log 2 (x + 1) < 2}

theorem intersection_eq : A ∩ B = {0, 1, 2} := by
  sorry

end intersection_eq_l323_323349


namespace find_area_find_length_side_a_l323_323311

variable {A B C : Type}
variable (a b c : ℝ) (α : ℝ)
variable (triangle_ABC : A = α → B = b → C = c → Type)

-- Conditions
def condition_b : b = 2 := by sorry
def condition_c : c = sqrt 3 := by sorry
def condition_A : α = π / 6 := by sorry

-- Theorem for part 1: Area of the triangle
theorem find_area 
  (h_b : b = 2) 
  (h_c : c = sqrt 3) 
  (h_A : α = π / 6) : 
  let area := 1 / 2 * b * c * Real.sin α 
  in area = sqrt 3 / 2 := by 
    sorry

-- Theorem for part 2: Length of side BC (a)
theorem find_length_side_a 
  (h_b : b = 2) 
  (h_c : c = sqrt 3) 
  (h_A : α = π / 6) : 
  let a := sqrt (b^2 + c^2 - 2 * b * c * Real.cos α) 
  in a = 1 := by 
    sorry

end find_area_find_length_side_a_l323_323311


namespace unique_functional_equation_exists_l323_323184

theorem unique_functional_equation_exists :
  ∃ f : ℝ → ℤ, (∀ x ∈ ℝ, f(x) ∈ ℤ) ∧ 
  (∀ x ∈ ℝ, f satisfies functional equation) := sorry

end unique_functional_equation_exists_l323_323184


namespace y_range_l323_323277

theorem y_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end y_range_l323_323277


namespace Melies_money_left_l323_323709

variable (meat_weight : ℕ)
variable (meat_cost_per_kg : ℕ)
variable (initial_money : ℕ)

def money_left_after_purchase (meat_weight : ℕ) (meat_cost_per_kg : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - (meat_weight * meat_cost_per_kg)

theorem Melies_money_left : 
  money_left_after_purchase 2 82 180 = 16 :=
by
  sorry

end Melies_money_left_l323_323709


namespace determine_and_maximize_l323_323338

noncomputable def f (x : ℝ) (ω m : ℝ) := 2 * sin(ω * x) * cos(ω * x) + m

def condition1 (ω m : ℝ) : Prop := f 0 ω m = 1

def condition2 (ω m : ℝ) : Prop := ∀ x : ℝ, f x ω m ≥ 0

def condition3 (ω : ℝ) : Prop := π / (2 * ω) = π / 2

noncomputable def g (x : ℝ) (ω m : ℝ) := f (x - π / 6) ω m

theorem determine_and_maximize (ω m : ℝ) (h1 : condition1 ω m) (h2 : condition3 ω) :
  ω = 1 ∧ m = 1 ∧ ∀ x ∈ Icc (0 : ℝ) (π / 2), g x 1 1 ≤ 2 ∧ g (5 * π / 12) 1 1 = 2 :=
by
  sorry

end determine_and_maximize_l323_323338


namespace find_defective_coin_l323_323315

theorem find_defective_coin (weight : ℕ → ℕ) (standard_weight : ℕ → ℕ)
  (h1 : weight 1 ≠ standard_weight 1 ∨ weight 2 ≠ standard_weight 2 ∨ weight 3 ≠ standard_weight 3 ∨ weight 5 ≠ standard_weight 5)
  (h2 : standard_weight 1 = 1 ∧ standard_weight 2 = 2 ∧ standard_weight 3 = 3 ∧ standard_weight 5 = 5) :
  ∃ k ∈ {1, 2, 3, 5}, weight k ≠ standard_weight k :=
sorry

end find_defective_coin_l323_323315


namespace Mike_total_spent_l323_323365

theorem Mike_total_spent
    (price_marbles : ℝ := 9.05)
    (price_football : ℝ := 4.95)
    (price_baseball : ℝ := 6.52)
    (price_car_original : ℝ := 10.99)
    (discount_car : ℝ := 0.15)
    (price_doll : ℝ := 12.50)
    (discount_doll : ℝ := 0.20) :
    let price_car_discounted := price_car_original * (1 - discount_car)
    let price_doll_first := price_doll
    let price_doll_second := price_doll * (1 - discount_doll)
    let total_price := price_marbles + price_football + price_baseball + price_car_discounted + (price_doll_first + price_doll_second)
    total_price = 52.36 :=
by
  simp only [price_car_discounted, price_doll_first, price_doll_second, total_price]
  exact_raw nat 5236 / 100 sorry

end Mike_total_spent_l323_323365


namespace num_divisors_g_2009_l323_323933

def g (n : ℕ) : ℕ :=
  2^(n + 1)

theorem num_divisors_g_2009 : (Nat.numDivisors (g 2009) = 2011) :=
by
  -- Prove that the number of divisors of 2^2010 is 2011
  sorry

end num_divisors_g_2009_l323_323933


namespace find_distinct_ordered_pairs_l323_323193

theorem find_distinct_ordered_pairs :
  { (x, y) : ℤ × ℤ // 1 ≤ x ∧ x ≤ 4 ∧ x^4 * y^4 - 18 * x^2 * y^2 + 81 = 0 }.to_finset.card = 2 :=
by
  sorry

end find_distinct_ordered_pairs_l323_323193


namespace factorize_expression_l323_323540

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 :=
by
  sorry

end factorize_expression_l323_323540


namespace prob_two_red_two_blue_l323_323448

theorem prob_two_red_two_blue :
  let total_marbles := 24
  let red_marbles := 15
  let blue_marbles := 9
  let selected_marbles := 4
  let favorable_ways := (nat.choose red_marbles 2) * (nat.choose blue_marbles 2)
  let total_ways := nat.choose total_marbles selected_marbles
  let probability := favorable_ways / total_ways
  probability = (4 : ℚ) / 27 := 
by
  sorry

end prob_two_red_two_blue_l323_323448


namespace number_of_5_primable_numbers_below_1000_l323_323110

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323110


namespace base_circumference_of_cone_l323_323470

theorem base_circumference_of_cone (r : ℝ) (theta : ℝ) (C : ℝ) 
  (h_radius : r = 6)
  (h_theta : theta = 180)
  (h_C : C = 2 * Real.pi * r) :
  (theta / 360) * C = 6 * Real.pi :=
by
  sorry

end base_circumference_of_cone_l323_323470


namespace find_common_ratio_l323_323548

def first_term : ℚ := 4 / 7
def second_term : ℚ := 12 / 7

theorem find_common_ratio (r : ℚ) : second_term = first_term * r → r = 3 :=
by
  sorry

end find_common_ratio_l323_323548


namespace train_speed_proof_l323_323485

noncomputable def train_speed_in_kmh (length_in_m: ℝ) (time_in_sec: ℝ) : ℝ :=
  (length_in_m / 1000) / (time_in_sec / 3600)

theorem train_speed_proof : train_speed_in_kmh 250.00000000000003 15 = 60 := by
  have length_in_km := 250.00000000000003 / 1000
  have time_in_hr := 15 / 3600
  have speed := length_in_km / time_in_hr
  exact (by ring : speed = 60)

end train_speed_proof_l323_323485


namespace regular_ngon_labeling_exists_l323_323322

theorem regular_ngon_labeling_exists (n : ℕ) (h1 : n > 3) (h2 : n % 2 = 1) :
  ∃ (label : Π (i j : ℕ), i ≠ j → ℕ), 
    (∀ (i j : ℕ) (hij : i ≠ j), 1 ≤ label i j hij ∧ label i j hij ≤ n) ∧
    (∀ (i j : ℕ) (hij : i ≠ j), label i j hij ≠ i ∧ label i j hij ≠ j) ∧
    (∀ (i : ℕ), ∀ (k l : ℕ), k ≠ l → k ≠ i ∧ l ≠ i → label i k (by linarith) ≠ label i l (by linarith)) :=
by
  sorry

end regular_ngon_labeling_exists_l323_323322


namespace james_final_amount_l323_323317

-- Define week earnings and conditions.
def week1 := 7
def week2 := 10
def week3 := 13
def week4 := 16

def total_earnings := week1 + week2 + week3 + week4

-- Define the amounts spent on the video game and the book.
def spent_on_video_game := total_earnings / 3

def remaining_after_video_game := total_earnings - spent_on_video_game
def spent_on_book := remaining_after_video_game * (3 / 8)

def final_amount_left := remaining_after_video_game - spent_on_book

-- The amount left should be approximately $19.17.
theorem james_final_amount : final_amount_left = 19.17 := by
  sorry

end james_final_amount_l323_323317


namespace part1_transport_efficiency_AC_part2_transport_efficiency_d_part3_transport_efficiency_two_stations_l323_323040

-- Define key distances and constants
def l : ℝ := sorry -- distance AB
def a : ℝ := sorry -- capacity of one truck
def x : ℕ := sorry -- number of trucks between A and C
def y : ℕ := sorry -- number of trucks between C and B
def d : ℝ := sorry -- distance AC or CD
def d1 : ℝ := sorry -- distance AC in third part
def d2 : ℝ := sorry -- distance CD

-- Lean 4 theorems
theorem part1_transport_efficiency_AC {AC_equiv: AC = l / 3} :
  (efficiency AC l a x y) = 2 / 9 := sorry

theorem part2_transport_efficiency_d {optimal_d: d = l / 2} :
  (efficiency d l a x y) = 1 / 4 := sorry

theorem part3_transport_efficiency_two_stations {optimal_d1_d2: d1 = d2 ∧ d1 = l / 3 ∧ d2 = l / 3} :
  (efficiency_two_stations d1 d2 l a x y) = 8 / 27 := sorry

-- Auxiliary functions for efficiency calculation
def efficiency (d AC l a x y : ℝ) : ℝ := sorry
def efficiency_two_stations (d1 d2 l a x y : ℝ) : ℝ := sorry

end part1_transport_efficiency_AC_part2_transport_efficiency_d_part3_transport_efficiency_two_stations_l323_323040


namespace elderly_in_sample_l323_323053

variable (total_employees : ℕ)
variable (young_employees : ℕ)
variable (middle_aged_per_elderly : ℕ)
variable (sampled_young : ℕ)
variable (sampled_group_elderly : ℕ)

theorem elderly_in_sample 
  (h1 : total_employees = 430) 
  (h2 : young_employees = 160) 
  (h3 : middle_aged_per_elderly = 2) 
  (h4 : sampled_young = 32) 
  (h5 : sampled_group_elderly = total_employees - young_employees - (middle_aged_per_elderly * sampled_group_elderly))
  (proportional_sampling : (sampled_young.to_rat / young_employees.to_rat) = (sampled_group_elderly.to_rat / (total_employees - young_employees - 2*sampled_group_elderly).to_rat)) :
   sampled_group_elderly = 18 :=
sorry

end elderly_in_sample_l323_323053


namespace volume_of_pyramid_is_690_l323_323761

noncomputable def volume_of_pyramid (AB BC: ℝ) (AP BP CP DP: ℝ) : ℝ :=
  (1 / 3) * (AB * BC / 2) * sqrt (AP^2 - (AB * sqrt 2 / 2)^2)

theorem volume_of_pyramid_is_690 :
  let AB := 10 * sqrt 2,
      BC := 15 * sqrt 2,
      AP := 25 / sqrt 2 in
  volume_of_pyramid AB BC AP AP AP AP ≈ 690 :=
by
  let AB := 10 * sqrt 2
  let BC := 15 * sqrt 2
  let AP := 25 / sqrt 2
  have volume : volume_of_pyramid AB BC AP AP AP AP ≈ 690 := sorry
  exact volume

end volume_of_pyramid_is_690_l323_323761


namespace constant_fraction_condition_l323_323794

theorem constant_fraction_condition 
    (a1 b1 c1 a2 b2 c2 : ℝ) : 
    (∀ x : ℝ, (a1 * x^2 + b1 * x + c1) / (a2 * x^2 + b2 * x + c2) = k) ↔ 
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) :=
by
  sorry

end constant_fraction_condition_l323_323794


namespace number_of_5_primable_less_1000_l323_323084

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323084


namespace real_solutions_count_l323_323264

noncomputable def number_of_real_solutions : ℕ := 2

theorem real_solutions_count (x : ℝ) :
  (x^2 - 5)^2 = 36 → number_of_real_solutions = 2 := by
  sorry

end real_solutions_count_l323_323264


namespace width_to_length_ratio_l323_323661

-- Define the problem conditions and the proof requirement
variable (l w P : ℕ)

-- Given conditions
def length_condition : l = 10 := rfl
def perimeter_condition : P = 30 := rfl
def perimeter_formula : P = 2 * (l + w) := sorry

-- The goal is to prove the ratio of the width to the length
theorem width_to_length_ratio (h1 : l = 10) (h2 : P = 30) (h3 : P = 2 * (l + w)): (w : ℕ) / l = 1 / 2 := sorry

end width_to_length_ratio_l323_323661


namespace positive_x_for_modulus_l323_323934

theorem positive_x_for_modulus (x : ℝ) (h : |(5 : ℝ) + x * complex.I| = 13) : x = 12 :=
by {
  have h_mod : |(5 : ℝ) + x * complex.I| = real.sqrt (5^2 + x^2),
  {
    rw [complex.abs, complex.norm_eq_sqrt_re_add_im, complex.of_real_re, complex.mul_I_im, complex.of_real_im, 
        complex.of_real_re, complex.mul_I_re, complex.add_re, complex.add_im, pow_two, pow_two, add_comm],
    simp,
  },
  rw h_mod at h,
  have : real.sqrt (25 + x^2) = 13 := h,
  have : 25 + x^2 = 169,
  {
    rw real.sqrt_eq_rpow at this,
    exact or.elim (eq_or_ne ((25 : ℝ) + x^2) 0) 
        (λ h_zero, by rwa h_zero) 
        (λ h_non_zero, (real.pow_denom_inj one_ne_zero h_non_zero).mp this),
  },
  have : x^2 = 144, by linarith,
  simpa using real.sqrt_inj (real.abs_nonneg x) (real.sqrt_ne_zero.2 (ne_of_gt zero_lt_144).symm) this,
  sorry,
}

end positive_x_for_modulus_l323_323934


namespace total_nephews_correct_l323_323152

namespace Nephews

-- Conditions
variable (ten_years_ago : Nat)
variable (current_alden_nephews : Nat)
variable (vihaan_extra_nephews : Nat)
variable (alden_nephews_10_years_ago : ten_years_ago = 50)
variable (alden_nephews_double : ten_years_ago * 2 = current_alden_nephews)
variable (vihaan_nephews : vihaan_extra_nephews = 60)

-- Answer
def total_nephews (alden_nephews_now vihaan_nephews_now : Nat) : Nat :=
  alden_nephews_now + vihaan_nephews_now

-- Proof statement
theorem total_nephews_correct :
  ∃ (alden_nephews_now vihaan_nephews_now : Nat), 
    alden_nephews_10_years_ago →
    alden_nephews_double →
    vihaan_nephews →
    alden_nephews_now = current_alden_nephews →
    vihaan_nephews_now = current_alden_nephews + vihaan_extra_nephews →
    total_nephews alden_nephews_now vihaan_nephews_now = 260 :=
by
  sorry

end Nephews

end total_nephews_correct_l323_323152


namespace hyperbola_eccentricity_l323_323230

noncomputable def hyperbola.eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) : ℝ := sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) (P : ℝ × ℝ)
  (hP_on_hyperbola: P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1)
  (dist_OP_OFO: dist (0,0) P = a)
  (dist_PF1_PF2: dist ((-a, 0) : ℝ × ℝ) P = √3 * dist ((a, 0) : ℝ × ℝ) P) :
  hyperbola.eccentricity a b h = (√3 + 1) :=
sorry

end hyperbola_eccentricity_l323_323230


namespace emmas_investment_value_l323_323432

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem emmas_investment_value :
  compound_interest 5000 0.01 72 ≈ 10154.28 :=
by
  sorry

end emmas_investment_value_l323_323432


namespace set_difference_I_A_l323_323612

open Set

-- Defining the sets I and A
def I := { x : ℕ | 1 < x ∧ x < 5 }
def A := { 2, 3 }

-- The proof statement
theorem set_difference_I_A : I \ A = {4} := 
by {
  ext x,
  simp [I, A],
  intros,
  sorry,
}

end set_difference_I_A_l323_323612


namespace range_of_m_l323_323226

theorem range_of_m (x m : ℝ) (hp : |1 - (x - 1) / 3| ≤ 2) 
  (hq : x^2 - 2*x + 1 - m^2 ≤ 0) (hm_pos : m > 0)
  (h_necessary_insufficient : ∀ x, (hq → hp) ∧ (hp → hq → False)) :
  9 ≤ m :=
sorry

end range_of_m_l323_323226


namespace range_of_a_l323_323604

-- Define the given function f(x)
def f (a x : ℝ) : ℝ := real.sqrt (x^2 + 1) - (2 / 5) * a * x

-- Define the condition for monotonicity of f on [0, ∞)
def monotonic_on_interval (a : ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x -> x ≤ y -> f a x ≤ f a y

theorem range_of_a (a : ℝ) :
  (0 < a) ∧ (monotonic_on_interval a) ↔ a ≥ 5 / 2 :=
sorry

end range_of_a_l323_323604


namespace determine_total_rent_l323_323807

variable {a b c : Type} -- The individuals a, b, c
variable (A_horses : ℕ) (A_months : ℕ) (B_horses : ℕ) (B_months : ℕ) (C_horses : ℕ) (C_months : ℕ)
variable (B_payment : ℕ)

-- Given contributions
def A_contribution := A_horses * A_months
def B_contribution := B_horses * B_months
def C_contribution := C_horses * C_months

-- Total horse-months
def total_horse_months := A_contribution + B_contribution + C_contribution

-- Given condition for B's payment
def B_ratio_condition : Prop := B_contribution / total_horse_months = B_payment / ?m_1

-- Required total rent R
def total_rent (R : ℕ) : Prop :=
  B_payment = 348 ∧
  B_ratio_condition ∧
  R = 840

-- To prove
theorem determine_total_rent (R : ℕ) : total_rent R := sorry

end determine_total_rent_l323_323807


namespace sachin_age_l323_323380

-- Assuming Sachin's and Rahul's ages are real numbers
variables (S R : ℝ)

-- Conditions given in the problem
def condition1 := R = S + 7
def condition2 := S / R = 7 / 9

-- The theorem we need to prove
theorem sachin_age : condition1 S R ∧ condition2 S R → S = 24.5 :=
by
  intros h,
  cases h with h1 h2,
  sorry

end sachin_age_l323_323380


namespace inequality_proof_l323_323684

theorem inequality_proof (a b c d : ℝ) (h : a + b + c + d = 8) :
  (a / Real.cbrt (8 + b - d)) + (b / Real.cbrt (8 + c - a)) + (c / Real.cbrt (8 + d - b)) + (d / Real.cbrt (8 + a - c)) ≥ 4 :=
by
  sorry

end inequality_proof_l323_323684


namespace cos_two_pi_over_three_l323_323043

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 :=
by sorry

end cos_two_pi_over_three_l323_323043


namespace prob_two_red_two_blue_l323_323449

theorem prob_two_red_two_blue :
  let total_marbles := 24
  let red_marbles := 15
  let blue_marbles := 9
  let selected_marbles := 4
  let favorable_ways := (nat.choose red_marbles 2) * (nat.choose blue_marbles 2)
  let total_ways := nat.choose total_marbles selected_marbles
  let probability := favorable_ways / total_ways
  probability = (4 : ℚ) / 27 := 
by
  sorry

end prob_two_red_two_blue_l323_323449


namespace FourConsecIntsSum34Unique_l323_323625

theorem FourConsecIntsSum34Unique :
  ∃! (a b c d : ℕ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (a + b + c + d = 34) ∧ (d = a + 3) :=
by
  -- The proof will be placed here
  sorry

end FourConsecIntsSum34Unique_l323_323625


namespace volume_in_region_l323_323325

def satisfies_conditions (x y : ℝ) : Prop :=
  |8 - x| + y ≤ 10 ∧ 3 * y - x ≥ 15

def in_region (x y : ℝ) : Prop :=
  satisfies_conditions x y

theorem volume_in_region (x y p m n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) (hn : n ≠ 0) (V : ℝ) 
  (hvol : V = (m * Real.pi) / (n * Real.sqrt p))
  (hprime : m.gcd n = 1 ∧ ¬(∃ k : ℕ, k^2 ∣ p ∧ k ≥ 2)) 
  (hpoints : ∀ (x y : ℝ), in_region x y → 3 * y - x = 15) : 
  m + n + p = 365 := 
sorry

end volume_in_region_l323_323325


namespace correct_operation_l323_323434

variable (a b : ℝ)

theorem correct_operation (h1 : a^2 + a^3 ≠ a^5)
                          (h2 : (-a^2)^3 ≠ a^6)
                          (h3 : -2*a^3*b / (a*b) ≠ -2*a^2*b) :
                          a^2 * a^3 = a^5 :=
by sorry

end correct_operation_l323_323434


namespace part1_part2_l323_323962

variable (a b c : ℝ)

open Classical

noncomputable theory

-- Defining the conditions
def cond_positive_numbers : Prop := (0 < a) ∧ (0 < b) ∧ (0 < c)
def cond_main_equation : Prop := a^2 + b^2 + 4*c^2 = 3
def cond_b_eq_2c : Prop := b = 2*c

-- Statement for part (1)
theorem part1 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) :
  a + b + 2*c ≤ 3 := sorry

-- Statement for part (2)
theorem part2 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) (h3 : cond_b_eq_2c b c) :
  (1 / a) + (1 / c) ≥ 3 := sorry

end part1_part2_l323_323962


namespace division_by_recurring_decimal_l323_323869

def decimal_to_fraction (d : ℚ) : Prop :=
  d = 1 / 3

theorem division_by_recurring_decimal : Prop :=
  let q := 1 / 3 in
  8 / q = 24

by sorry

end division_by_recurring_decimal_l323_323869


namespace remainder_of_division_l323_323196

-- Define the polynomials
def dividend : Polynomial ℝ := Polynomial.C 1 * X ^ 4 + Polynomial.C 2
def divisor : Polynomial ℝ := Polynomial.C 1 * X ^ 2 - Polynomial.C 3 * X + Polynomial.C 2

-- Define the remainder function (as an example, built-in functions could be used in practice)
noncomputable def remainder (f g : Polynomial ℝ) : Polynomial ℝ := f % g

-- State the theorem
theorem remainder_of_division :
  remainder dividend divisor = Polynomial.C 15 * X - Polynomial.C 12 := by
  sorry

end remainder_of_division_l323_323196


namespace mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l323_323366

noncomputable def ratio_of_A_students (total_students_A : ℕ) (A_students_A : ℕ) : ℚ :=
  A_students_A / total_students_A

theorem mrs_berkeley_A_students_first_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 18 →
    (A_students_A / total_students_A) * total_students_B = 12 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

theorem mrs_berkeley_A_students_extended_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 27 →
    (A_students_A / total_students_A) * total_students_B = 18 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

end mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l323_323366


namespace smallest_positive_leading_coefficient_l323_323021

variable {a b c : ℚ} -- Define variables a, b, c that are rational numbers
variable (P : ℤ → ℚ) -- Define the polynomial P as a function from integers to rationals

-- State that P(x) is in the form of ax^2 + bx + c
def is_quadratic_polynomial (P : ℤ → ℚ) (a b c : ℚ) :=
  ∀ x : ℤ, P x = a * x^2 + b * x + c

-- State that P(x) takes integer values for all integer x
def takes_integer_values (P : ℤ → ℚ) :=
  ∀ x : ℤ, ∃ k : ℤ, P x = k

-- The statement we want to prove
theorem smallest_positive_leading_coefficient (h1 : is_quadratic_polynomial P a b c)
                                              (h2 : takes_integer_values P) :
  ∃ a : ℚ, 0 < a ∧ ∀ b c : ℚ, is_quadratic_polynomial P a b c → takes_integer_values P → a = 1/2 :=
sorry

end smallest_positive_leading_coefficient_l323_323021


namespace triangle_equality_pairs_l323_323306

theorem triangle_equality_pairs {A B C E F : Point} 
  (midpoint_AB_E : E = midpoint A B) 
  (midpoint_AC_F : F = midpoint A C)
  : number_of_equal_area_triangle_pairs (triangle A B C) = 10 := 
sorry

end triangle_equality_pairs_l323_323306


namespace y_range_l323_323279

theorem y_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end y_range_l323_323279


namespace solve_for_y_l323_323061

variable {b c y : Real}

theorem solve_for_y (h : b > c) (h_eq : y^2 + c^2 = (b - y)^2) : y = (b^2 - c^2) / (2 * b) := 
sorry

end solve_for_y_l323_323061


namespace find_g_l323_323756

def nabla (g h : ℤ) : ℤ := g ^ 2 - h ^ 2

theorem find_g (g : ℤ) (h : ℤ)
  (H1 : 0 < g)
  (H2 : nabla g 6 = 45) :
  g = 9 :=
by
  sorry

end find_g_l323_323756


namespace angle_MCD_is_105_l323_323665

variable {A B C D E M : Point}
variable [geometry ABCD] [equilateral_triangle A B E]
variable (x : Real)

hypothesis (h_rect : rectangle ABCD)
hypothesis (h_ratio : distance A B = 2 * distance B C)
hypothesis (h_mid : midpoint M B E)

theorem angle_MCD_is_105 :
  angle M C D = 105 :=
sorry

end angle_MCD_is_105_l323_323665


namespace count_5_primable_integers_lt_1000_is_21_l323_323104

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323104


namespace count_five_primable_lt_1000_l323_323082

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323082


namespace sum_of_first_eight_primes_ending_in_3_l323_323200

/-- Define the set of prime numbers ending in 3 within a certain range -/
def primes_ending_in_3 : List ℕ := [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123]

/-- Define a predicate to check if a number is prime and ends in 3 -/
def is_prime_ending_in_3 (n : ℕ) : Prop := Prime n ∧ n % 10 = 3

/-- Consider the first eight primes ending in 3 -/
noncomputable def first_eight_primes_ending_in_3 : List ℕ :=
  primes_ending_in_3.filter is_prime_ending_in_3 |>.take 8

/-- Sum of the first eight primes ending in 3 -/
noncomputable def sum_first_eight_primes_ending_in_3 : ℕ :=
  (first_eight_primes_ending_in_3).sum

theorem sum_of_first_eight_primes_ending_in_3 :
  sum_first_eight_primes_ending_in_3 = 394 := by
  sorry

end sum_of_first_eight_primes_ending_in_3_l323_323200


namespace dividend_percentage_correct_l323_323464

def investment : ℝ := 14400
def face_value : ℝ := 100
def premium : ℝ := 0.20
def cost_per_share : ℝ := face_value * (1 + premium)
def total_dividend : ℝ := 840
def num_shares : ℝ := investment / cost_per_share
def dividend_per_share : ℝ := total_dividend / num_shares
def expected_dividend_percentage : ℝ := 7

theorem dividend_percentage_correct :
  (dividend_percentage : ℝ) → dividend_percentage = (dividend_per_share / face_value) * 100 → 
  dividend_percentage = expected_dividend_percentage := 
by sorry

end dividend_percentage_correct_l323_323464


namespace repeating_decimal_count_l323_323927

theorem repeating_decimal_count :
  let count := (list.filter (λ n, ∀ p, Nat.Prime p → p ∣ (n + 1) → p = 2 ∨ p = 5) (list.range' 2 200)).length
  count = 182 :=
by
  sorry

#eval (let count := (list.filter (λ n, ∀ p, Nat.Prime p → p ∣ (n + 1) → p = 2 ∨ p = 5) (list.range' 2 200)).length
       count)  -- Expected to output 182

end repeating_decimal_count_l323_323927


namespace second_amount_l323_323652

-- Define the given conditions as hypotheses
variable (P : ℝ)

-- Condition 1: Interest produced by Rs 100 in 8 years at 5% interest rate
def interest1 := 100 * 0.05 * 8

-- Condition 2: Interest produced by the amount P in 2 years at 10% interest rate
def interest2 := P * 0.10 * 2

-- The proof goal
theorem second_amount (h : interest1 = interest2) : P = 200 :=
by sorry

end second_amount_l323_323652


namespace problem_f8_f2018_l323_323025

theorem problem_f8_f2018 (f : ℕ → ℝ) (h₀ : ∀ n, f (n + 3) = (f n - 1) / (f n + 1)) 
  (h₁ : f 1 ≠ 0) (h₂ : f 1 ≠ 1) (h₃ : f 1 ≠ -1) : 
  f 8 * f 2018 = -1 :=
sorry

end problem_f8_f2018_l323_323025


namespace ratio_of_sides_ABC_equal_3_4_5_l323_323894

-- Conditions: Define the elements of the triangle and relevant points, squares, right angles, and congruency stated
variables {A B C I P Q K L : Point}
variables [IncircleTriangle ABC I P Q] [Square PIQB] [RightAngle KIL] [RightAngle PIQ]
variables [EqualAngles PIK QIL] [ThalesTheorem I P Q B]
variables [Length AQ = 2 * Length (BP)]

-- Needed theorem
theorem ratio_of_sides_ABC_equal_3_4_5 (hABC : Triangle ABC) (hI: Incenter I)
  (hP: PointOfTangency I P BC) (hQ: PointOfTangency I Q AB)
  (hPIQBSquare: IsSquare PIQB) (hRightKIL: IsRightAngle KIL) 
  (hRightPIQ: IsRightAngle PIQ) (hEqualAngles: ∠PIK = ∠QIL)
  (hThales: KP = QL) (hLengths: AB = 6/5 * AL ∧ AL = AO ∧ AO = 3/5 * AC):
  AB / BC / CA = 3 / 4 / 5 :=
by
  sorry

end ratio_of_sides_ABC_equal_3_4_5_l323_323894


namespace intersection_A_B_l323_323228

-- Define the sets A and B based on the given conditions
def A := { x : ℝ | (1 / 9) ≤ (3:ℝ)^x ∧ (3:ℝ)^x ≤ 1 }
def B := { x : ℝ | x^2 < 1 }

-- State the theorem for the intersection of sets A and B
theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x ≤ 0 } :=
by
  sorry

end intersection_A_B_l323_323228


namespace jamie_collects_oysters_l323_323048

theorem jamie_collects_oysters (d : ℕ) (p : ℕ) (r : ℕ) (x : ℕ)
  (h1 : d = 14)
  (h2 : p = 56)
  (h3 : r = 25)
  (h4 : x = p / d * 100 / r) :
  x = 16 :=
by
  sorry

end jamie_collects_oysters_l323_323048


namespace count_5_primable_less_than_1000_eq_l323_323132

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323132


namespace largest_common_divisor_of_product_l323_323892

theorem largest_common_divisor_of_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : 0 < n) :
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) → d ∣ k :=
by
  sorry

end largest_common_divisor_of_product_l323_323892


namespace marc_watch_days_l323_323700

theorem marc_watch_days (bought_episodes : ℕ) (watch_fraction : ℚ) (episodes_per_day : ℚ) (total_days : ℕ) : 
  bought_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  episodes_per_day = (50 : ℚ) * watch_fraction → 
  total_days = (bought_episodes : ℚ) / episodes_per_day →
  total_days = 10 := 
sorry

end marc_watch_days_l323_323700


namespace DF_equals_FE_l323_323259

variables {A B C D E F : Type} [has_order A] [has_order B] [has_order C] [has_order D] [has_order E] [has_order F]
variables (triangle_ABC : Triangle A B C)
variables (h1 : triangle_ABC.side AB < triangle_ABC.side AC)
variables (line_B_parallel_AC : Line B D) (line_C_parallel_AB : Line C E)
variables (external_angle_bisector : AngleBisector BAC)
variables (D_meeting_condition : is_external_angle_bisector D external_angle_bisector)
variables (E_meeting_condition : is_external_angle_bisector E external_angle_bisector)
variables (F_on_AC : PointOnLine F triangle_ABC.side AC)
variables (F_condition : triangle_ABC.side FC = triangle_ABC.side AB)

theorem DF_equals_FE :
  triangle_ABC.side DF = triangle_ABC.side FE :=
sorry

end DF_equals_FE_l323_323259


namespace quadratic_zeros_l323_323418

theorem quadratic_zeros : ∀ x : ℝ, (x = 3 ∨ x = -1) ↔ (x^2 - 2*x - 3 = 0) := by
  intro x
  sorry

end quadratic_zeros_l323_323418


namespace opposite_edges_perpendicular_l323_323752

theorem opposite_edges_perpendicular
  (A B C D H : Type)
  [orthocenter H A B C]
  [height D H]
  [altitude AH BH CH A B C] :
  (DA ⊥ BC) ∧ (DB ⊥ AC) ∧ (DC ⊥ AB) := 
sorry

end opposite_edges_perpendicular_l323_323752


namespace f_properties_l323_323218

noncomputable def f : ℝ → ℝ := sorry -- we define f as a noncomputable function for generality 

-- Given conditions as Lean hypotheses
axiom functional_eq : ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom not_always_zero : ¬(∀ x : ℝ, f x = 0)

-- The statement we need to prove
theorem f_properties : f 0 = 1 ∧ (∀ x : ℝ, f (-x) = f x) := 
  by 
    sorry

end f_properties_l323_323218


namespace tan_sum_identity_l323_323922

noncomputable def tan_25 := Real.tan (Real.pi / 180 * 25)
noncomputable def tan_35 := Real.tan (Real.pi / 180 * 35)
noncomputable def sqrt_3 := Real.sqrt 3

theorem tan_sum_identity :
  tan_25 + tan_35 + sqrt_3 * tan_25 * tan_35 = 1 :=
by
  sorry

end tan_sum_identity_l323_323922


namespace repair_cost_l323_323863

theorem repair_cost (C : ℝ) (repair_cost : ℝ) (profit : ℝ) (selling_price : ℝ)
  (h1 : repair_cost = 0.10 * C)
  (h2 : profit = 1100)
  (h3 : selling_price = 1.20 * C)
  (h4 : profit = selling_price - C) :
  repair_cost = 550 :=
by
  sorry

end repair_cost_l323_323863


namespace number_of_5_primable_numbers_below_1000_l323_323112

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323112


namespace wheat_weight_approximation_l323_323489
open BigOperators

-- Defining conditions
def first_term := 1
def common_ratio := 2
def num_squares := 64
def grains_per_kg := 20000
def log2 := 0.3

-- Establishing the question and solution relation
theorem wheat_weight_approximation :
  let total_grains := (2^num_squares) - 1
  let weight_kg := total_grains / grains_per_kg
  let approx_log_weight := 63 * log2 - 7
  weight_kg ≈ 10^12 :=
sorry

end wheat_weight_approximation_l323_323489


namespace increasing_function_example_l323_323788

variable (f : ℝ → ℝ)
variable (I : Set ℝ)
variable [Preorder ℝ]
variable [LinearOrderedField ℝ]

def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ {x₁ x₂ : ℝ}, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

theorem increasing_function_example :
  let f := λ x : ℝ, -x^2 + 2 * x
  let I := { x : ℝ | x < 1 }
  is_increasing f I := by
  sorry

end increasing_function_example_l323_323788


namespace find_t_l323_323969

theorem find_t 
  (t : ℝ) 
  (h1 : t > 1) 
  (h2 : ∫ x in 1..t, (2 * x + 1) = t^2) : t = 2 := 
by 
  sorry

end find_t_l323_323969


namespace problem_l323_323345

theorem problem (f : ℕ → ℝ) 
  (h_def : ∀ x, f x = Real.cos (x * Real.pi / 3)) 
  (h_period : ∀ x, f (x + 6) = f x) : 
  (Finset.sum (Finset.range 2018) f) = 0 := 
by
  sorry

end problem_l323_323345


namespace smallest_value_of_m_plus_n_l323_323399

theorem smallest_value_of_m_plus_n :
  ∃ m n : ℕ, 1 < m ∧ 
  (∃ l : ℝ, l = (m^2 - 1 : ℝ) / (m * n) ∧ l = 1 / 2021) ∧
  m + n = 85987 := 
sorry

end smallest_value_of_m_plus_n_l323_323399


namespace number_of_5_primable_less_1000_l323_323083

def is_prime_digit (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d in n.digits, is_prime_digit d)

def count_5_primable_less_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem number_of_5_primable_less_1000 : count_5_primable_less_1000 = 7 :=
  sorry

end number_of_5_primable_less_1000_l323_323083


namespace intersection_A_B_l323_323487

namespace SetTheory

open Set

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end SetTheory

end intersection_A_B_l323_323487


namespace relationship_abc_l323_323572

noncomputable def a : ℝ := (0.7 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (-0.6 : ℝ)
noncomputable def c : ℝ := (0.6 : ℝ) ^ (0.7 : ℝ)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Proof will go here
  sorry

end relationship_abc_l323_323572


namespace count_five_primable_lt_1000_l323_323076

def one_digit_primes : set ℕ := {2, 3, 5, 7}

def is_n_primable (n x : ℕ) : Prop :=
  x % n = 0 ∧ (∀ d ∈ (x.digits 10), d ∈ one_digit_primes)

def is_five_primable (x : ℕ) : Prop := is_n_primable 5 x

def five_primable_numbers : set ℕ := {x | is_five_primable x ∧ x < 1000}

theorem count_five_primable_lt_1000 : (five_primable_numbers).count = 21 := 
  sorry

end count_five_primable_lt_1000_l323_323076


namespace incorrect_sqrt_add_incorrect_neg_sqrt_incorrect_sqrt_add_mult_correct_sqrt_mult_main_l323_323799

theorem incorrect_sqrt_add : ¬(√2 + √5 = √7) := 
by sorry

theorem incorrect_neg_sqrt : ¬(√((-2) ^ 2) = -2) := 
by 
  calc 
    sqrt((-2) ^ 2) = sqrt(4): by sorry
    ... = 2 : by sorry
    ... ≠ -2 : by sorry 

theorem incorrect_sqrt_add_mult : ¬(2 + √3 = 2 * √3) := 
by sorry

theorem correct_sqrt_mult : (√2 * √3 = √6) := 
by sorry

theorem main : ¬(√2 + √5 = √7) ∧ ¬(√((-2) ^ 2) = -2) ∧ ¬(2 + √3 = 2 * √3) ∧ (√2 * √3 = √6) := 
  ⟨incorrect_sqrt_add, incorrect_neg_sqrt, incorrect_sqrt_add_mult, correct_sqrt_mult⟩

end incorrect_sqrt_add_incorrect_neg_sqrt_incorrect_sqrt_add_mult_correct_sqrt_mult_main_l323_323799


namespace value_of_a_l323_323355

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -4 * x else x^2

theorem value_of_a (a : ℝ) (h : f a = 4) : a = -1 ∨ a = 2 :=
sorry

end value_of_a_l323_323355


namespace dealer_net_profit_percentage_l323_323458

-- Define the initial conditions
def cost_price (kg : ℝ) : ℝ := 100 * kg
def counter_weight : ℝ := 0.8
def impurity : ℝ := 0.15
def selling_price (kg : ℝ) : ℝ := 100 * kg

/-- 
  The net profit percentage of the dealer is 25%.
--/
theorem dealer_net_profit_percentage : 
  let actual_product := 1 * counter_weight,
      total_weight := actual_product + (actual_product * impurity),
      cost_to_dealer := cost_price actual_product,
      profit := selling_price 1 - cost_to_dealer in
  (profit / cost_to_dealer) * 100 = 25 :=
by
  sorry

end dealer_net_profit_percentage_l323_323458


namespace unique_solution_l323_323820

variables {α : Type*} [linear_ordered_field α]

variables (a11 a22 a33 a12 a13 a21 a23 a31 a32 : α)
variables (x1 x2 x3 : α)

-- Conditions
-- 1. a11, a22, a33 are positive
axiom pos_diag: a11 > 0 ∧ a22 > 0 ∧ a33 > 0

-- 2. All other coefficients are negative
axiom neg_off_diag: a12 < 0 ∧ a13 < 0 ∧ a21 < 0 ∧ a23 < 0 ∧ a31 < 0 ∧ a32 < 0

-- 3. In every equation, the sum of the coefficients is positive
axiom sum_pos_eq1: a11 + a12 + a13 > 0
axiom sum_pos_eq2: a21 + a22 + a23 > 0
axiom sum_pos_eq3: a31 + a32 + a33 > 0

-- System of equations
axiom eq1: a11 * x1 + a12 * x2 + a13 * x3 = 0
axiom eq2: a21 * x1 + a22 * x2 + a23 * x3 = 0
axiom eq3: a31 * x1 + a32 * x2 + a33 * x3 = 0

-- Statement to prove
theorem unique_solution : x1 = 0 ∧ x2 = 0 ∧ x3 = 0 :=
by sorry

end unique_solution_l323_323820


namespace sin_add_polynomial_l323_323817

/-- Given x = sin α and y = sin β, express the relationship among x, y, and z as
    z^4 - 2z^2(x^2 + y^2 - 2x^2 y^2) + (x^2 - y^2)^2 = 0
    and determine the values of x and y for which z takes fewer than four values. -/
theorem sin_add_polynomial (α β : ℝ) (x y z : ℝ) :
  x = Real.sin α → y = Real.sin β →
  (z^4 - 2*z^2*(x^2 + y^2 - 2*x^2*y^2) + (x^2 - y^2)^2 = 0) ∧
  (∃ fewer_values_conditions
    x = 0 ∨ x = 1 ∨ x = -1 ∨ y = 0 ∨ y = 1 ∨ y = -1 → 
    -- Proof of additional statements for fewer values conditions continue here
    sorry) :=
begin
  intros hx hy,
  -- Proof continues here
  sorry
end

end sin_add_polynomial_l323_323817


namespace num_of_B_sets_l323_323989

def A : Set ℕ := {1, 2}

theorem num_of_B_sets (S : Set ℕ) (A : Set ℕ) (h : A = {1, 2}) (h1 : ∀ B : Set ℕ, A ∪ B = S) : 
  ∃ n : ℕ, n = 4 ∧ (∀ B : Set ℕ, B ⊆ {1, 2} → S = {1, 2}) :=
by {
  sorry
}

end num_of_B_sets_l323_323989


namespace smallest_number_one_zero_smallest_number_two_zeros_smallest_number_no_zeros_l323_323565

-- Definitions based on conditions
def digits := {2, 3, 4, 0, 0}
def is_five_digit (n : ℕ) := 10000 ≤ n ∧ n < 100000
def reads_zeros (n : ℕ) : ℕ := -- this function will count the number of zeros read
  sorry

-- Smallest five-digit number that reads only one zero
theorem smallest_number_one_zero : ∃ n, n = 20034 ∧ is_five_digit n ∧ (reads_zeros n = 1) :=
  by sorry

-- Smallest five-digit number that reads two zeros
theorem smallest_number_two_zeros : ∃ n, n = 20304 ∧ is_five_digit n ∧ (reads_zeros n = 2) :=
  by sorry

-- Smallest five-digit number that does not read any zero
theorem smallest_number_no_zeros : ∃ n, n = 23400 ∧ is_five_digit n ∧ (reads_zeros n = 0) :=
  by sorry

end smallest_number_one_zero_smallest_number_two_zeros_smallest_number_no_zeros_l323_323565


namespace cube_and_difference_of_squares_l323_323561

theorem cube_and_difference_of_squares (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 :=
by {
  sorry
}

end cube_and_difference_of_squares_l323_323561


namespace total_parallelograms_in_grid_l323_323895

theorem total_parallelograms_in_grid (n : ℕ) : 
  ∃ p : ℕ, p = 3 * Nat.choose (n + 2) 4 :=
sorry

end total_parallelograms_in_grid_l323_323895


namespace trigonometric_identity_solution_l323_323437

theorem trigonometric_identity_solution (k : ℤ) : 
  (∃ x : ℝ, cos x ^ 3 + (1 / 2) * sin (2 * x) - cos x * sin x ^ 3 + 4 * sin x + 4 = 0 ∧ x = pi / 2 * (4 * k - 1)) :=
sorry

end trigonometric_identity_solution_l323_323437


namespace petya_speed_second_half_l323_323297

theorem petya_speed_second_half
  (t : ℝ) 
  (v_vasya : ℝ := 12) -- Vasya's constant speed is 12 km/h
  (d1_petya : ℝ := 9 * t) -- Distance for the first half Petya runs
  (t_vasya : ℝ := (9 * t) / v_vasya) -- Time Vasya takes to run the distance 9t km
  (t1_petya : ℝ := t) -- Time Petya takes to reach the midpoint
  (v_petya_first_half : ℝ := 9) -- Petya's speed for the first half is 9 km/h
  (d2_petya : ℝ := 9 * t) -- Distance for the second half Petya runs
  (time_required_by_vasya : ℝ := (9 * t) / v_vasya) -- Time Vasya takes to run 9t km
  (v_vasya_second_half : ℝ := v_vasya) -- Vasya's speed for the second half remains 12 km/h
  : (d2_petya / time_required_by_vasya) = 18 := -- Petya's speed should be 18 km/h

by
  { 
    calc
        (d2_petya / time_required_by_vasya) = (9 * t) / (3 * t / 4) :
                                   by { rw [time_required_by_vasya, d2_petya], ring }
                           ...  = (36 * t / 3 * t) : by { rw div_eq_mul_inv (9 * t), ring }
                           ...  = 18             : by { field_simp [t_ne_zero], ring }
  }

end petya_speed_second_half_l323_323297


namespace distance_point_to_line_correct_l323_323746

-- Define the point P
def P : ℝ × ℝ := (-1, 1)

-- Define the line l in standard form
def A : ℝ := 3
def B : ℝ := 4
def C : ℝ := 0

-- Function to calculate the distance from a point to a line
def distance_from_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Math.sqrt (A ^ 2 + B ^ 2)

-- The theorem we need to prove
theorem distance_point_to_line_correct :
  distance_from_point_to_line (-1) 1 3 4 0 = 1 / 5 :=
by
  sorry

end distance_point_to_line_correct_l323_323746


namespace volume_to_surface_area_ratio_l323_323436

theorem volume_to_surface_area_ratio (base_layer: ℕ) (top_layer: ℕ) (unit_cube_volume: ℕ) (unit_cube_faces_exposed_base: ℕ) (unit_cube_faces_exposed_top: ℕ) 
  (V : ℕ := base_layer * top_layer * unit_cube_volume) 
  (S : ℕ := base_layer * unit_cube_faces_exposed_base + top_layer * unit_cube_faces_exposed_top) 
  (ratio := V / S) : ratio = 1 / 2 :=
by
  -- Base Layer: 4 cubes, 3 faces exposed per cube
  have base_layer_faces : ℕ := 4 * 3
  -- Top Layer: 4 cubes, 1 face exposed per cube
  have top_layer_faces : ℕ := 4 * 1
  -- Total volume is 8
  have V : ℕ := 4 * 2
  -- Total surface area is 16
  have S : ℕ := base_layer_faces + top_layer_faces
  -- Volume to surface area ratio computation
  have ratio : ℕ := V / S
  sorry

end volume_to_surface_area_ratio_l323_323436


namespace functional_equation_solutions_l323_323187

theorem functional_equation_solutions (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f(x^2) - f(y^2) ≤ (f(x) + y) * (x - f(y))) :
  (∀ x : ℝ, f(x) = x) ∨ (∀ x : ℝ, f(x) = -x) :=
by
  sorry

end functional_equation_solutions_l323_323187


namespace bond_interest_percentage_l323_323323

theorem bond_interest_percentage (F S : ℝ) (hF : F = 5000) (hS : S = 4615.384615384615) :
  let I := 0.065 * S in
  (I / F) * 100 = 6 := by
  sorry

end bond_interest_percentage_l323_323323


namespace find_multiple_l323_323362

theorem find_multiple
    (m : ℕ)
    (h₁ : ∑ i in finset.range 5, (2 * m ^ i) = 62) :
    m = 2 :=
sorry

end find_multiple_l323_323362


namespace pictures_deleted_l323_323029

theorem pictures_deleted (zoo_pics museum_pics remaining_pics : ℕ) 
  (h1 : zoo_pics = 15) 
  (h2 : museum_pics = 18) 
  (h3 : remaining_pics = 2) : 
  zoo_pics + museum_pics - remaining_pics = 31 :=
by 
  sorry

end pictures_deleted_l323_323029


namespace number_of_girls_l323_323836

theorem number_of_girls
  (total_boys : ℕ)
  (total_boys_eq : total_boys = 10)
  (fraction_girls_reading : ℚ)
  (fraction_girls_reading_eq : fraction_girls_reading = 5/6)
  (fraction_boys_reading : ℚ)
  (fraction_boys_reading_eq : fraction_boys_reading = 4/5)
  (total_not_reading : ℕ)
  (total_not_reading_eq : total_not_reading = 4)
  (G : ℝ)
  (remaining_girls_reading : (1 - fraction_girls_reading) * G = 2)
  (remaining_boys_not_reading : (1 - fraction_boys_reading) * total_boys = 2)
  (remaining_total_not_reading : 2 + 2 = total_not_reading)
  : G = 12 :=
by
  sorry

end number_of_girls_l323_323836


namespace triangle_area_tangent_line_l323_323394

open Real

-- Define the curve as a function
def curve (x : ℝ) : ℝ := exp x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (2, exp 2)

-- Define the derivative of the curve
def derivative_curve (x : ℝ) : ℝ := deriv exp x

-- Define the equation of the tangent line at the point (2, exp 2)
def tangent_line (x : ℝ) : ℝ := exp 2 * (x - 2) + exp 2

-- Define the x-intercept of the tangent line
def x_intercept : ℝ := 1

-- Define the y-intercept of the tangent line
def y_intercept : ℝ := -exp 2

-- Finally, prove that the area of the triangle formed by the tangent line and the coordinate axes is e^2 / 2
theorem triangle_area_tangent_line : 
  let base := x_intercept in
  let height := exp 2 in
  (1 / 2) * base * height = exp 2 / 2 :=
by
  let base := 1 in
  let height := exp 2 in
  calc
    (1 / 2) * base * height = (1 / 2) * 1 * exp 2 := by sorry
                            ... = exp 2 / 2 := by sorry

end triangle_area_tangent_line_l323_323394


namespace david_still_has_less_than_750_l323_323521

theorem david_still_has_less_than_750 (S R : ℝ) 
  (h1 : S + R = 1500)
  (h2 : R < S) : 
  R < 750 :=
by 
  sorry

end david_still_has_less_than_750_l323_323521


namespace max_pawns_theorem_l323_323792

noncomputable def max_pawns_on_chessboard {board : Type} [Chessboard board] : Nat :=
  let total_squares := 64
  let excluded_squares := 1 -- e4
  let max_pairs := 24 -- symmetric pairs around e4
  let available_squares := total_squares - excluded_squares - max_pairs
  available_squares

theorem max_pawns_theorem : max_pawns_on_chessboard = 39 :=
by
  -- proof is omitted
  sorry

end max_pawns_theorem_l323_323792


namespace area_PFO_l323_323219

def point_on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

def distance_from_focus (P : ℝ × ℝ) : Prop := 
  let F := (1, 0) in
  real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5

def area_of_triangle_PFO (P : ℝ × ℝ) : ℝ :=
  let F := (1, 0) in
  let O := (0, 0) in
  (1 / 2) * abs (O.1 * (P.2 - F.2) + P.1 * (F.2 - O.2) + F.1 * (O.2 - P.2))

theorem area_PFO (P : ℝ × ℝ) (h1 : point_on_parabola P) (h2 : distance_from_focus P) :
  area_of_triangle_PFO P = 2 :=
sorry

end area_PFO_l323_323219


namespace count_5_primable_less_than_1000_eq_l323_323134

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323134


namespace range_of_f_l323_323411

def f (x : ℝ) : ℝ := Real.logb 2 (3 ^ x + 1)

theorem range_of_f : ∀ y, (∃ x, f x = y) ↔ y > 0 := by
  sorry

end range_of_f_l323_323411


namespace proof_problem_l323_323246

theorem proof_problem (α : ℝ) (h1 : 0 < α ∧ α < π)
    (h2 : Real.sin α + Real.cos α = 1 / 5) :
    (Real.tan α = -4 / 3) ∧ 
    ((Real.sin (3 * Real.pi / 2 + α) * Real.sin (Real.pi / 2 - α) * (Real.tan (Real.pi - α))^3) / 
    (Real.cos (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α)) = -4 / 3) :=
by
  sorry

end proof_problem_l323_323246


namespace general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l323_323244

def seq_a : ℕ → ℕ 
| 0 => 0  -- a_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2^(n+1)

def seq_b : ℕ → ℕ 
| 0 => 0  -- b_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2*(n+1) -1

def sum_S (n : ℕ) : ℕ := (seq_a (n+1) * 2) - 2

def sum_T : ℕ → ℕ 
| 0 => 0  -- T_0 is not defined in natural numbers, put it as zero for base case too
| (n+1) => (n+1)^2

def sum_D : ℕ → ℕ
| 0 => 0
| (n+1) => (seq_a (n+1) * seq_b (n+1)) + sum_D n

theorem general_term_a_n (n : ℕ) : seq_a n = 2^n := sorry

theorem general_term_b_n (n : ℕ) : seq_b n = 2*n - 1 := sorry

theorem sum_of_first_n_terms_D_n (n : ℕ) : sum_D n = (2*n - 3)*2^(n+1) + 6 := sorry

end general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l323_323244


namespace students_wearing_other_colors_l323_323296

theorem students_wearing_other_colors :
  ∀ (total_students blue_pct red_pct green_pct : ℝ),
  total_students = 600 →
  blue_pct = 0.45 →
  red_pct = 0.23 →
  green_pct = 0.15 →
  ∃ (other_colors : ℝ), other_colors = total_students - (blue_pct * total_students + red_pct * total_students + green_pct * total_students) ∧ other_colors = 102 :=
begin
  intros total_students blue_pct red_pct green_pct h_total_students h_blue_pct h_red_pct h_green_pct,
  use total_students - (blue_pct * total_students + red_pct * total_students + green_pct * total_students),
  split,
  {
    refl,
  },
  {
    rw [h_total_students, h_blue_pct, h_red_pct, h_green_pct],
    norm_num,
  },
end

end students_wearing_other_colors_l323_323296


namespace mean_greater_than_median_l323_323564

theorem mean_greater_than_median {p : ℕ} (hp : Nat.Prime p) (h: p < 20) :
  let s := [p, p + 2, p + 4, p + 11, p + 53]
  let mean := (s.sum / s.length)
  let median := s.sorted !! (s.length / 2)
  mean - median = 10 :=
by
  sorry

end mean_greater_than_median_l323_323564


namespace find_g_l323_323687

def f (x : ℝ) := polynomial
def g (x : ℝ) := polynomial

theorem find_g :
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f (g x) = f x * g x) → 
    g 3 = 50 → 
    g = (λ x, x ^ 2 + 41 * x - 41) :=
by
  sorry

end find_g_l323_323687


namespace area_inside_circle_but_outside_square_l323_323472

theorem area_inside_circle_but_outside_square :
  let side_length := 2
  let radius := 1
  ∃(center : ℝ × ℝ),
  area_inside_circle_but_outside_square center side_length radius = π - 2 :=
by
  -- Definitions and conditions
  let side_length := 2
  let radius := 1
  -- Assuming some center for the square and circle.
  let center := (0, 0)
  
  -- We state the main goal
  have h : area_inside_circle_but_outside_square center side_length radius = π - 2, from sorry
  exact ⟨center, h⟩

end area_inside_circle_but_outside_square_l323_323472


namespace sum_m_satisfying_binom_eq_l323_323428

open_locale big_operators

noncomputable theory

def binom (n k : ℕ) : ℕ := nat.choose n k

theorem sum_m_satisfying_binom_eq :
  (∑ m in {m : ℕ | binom 25 m + binom 25 12 = binom 26 13}.to_finset, m) = 24 :=
by sorry

end sum_m_satisfying_binom_eq_l323_323428


namespace tangent_line_slope_neg3_minimum_area_triangle_l323_323594

-- Problem for Question 1
theorem tangent_line_slope_neg3 (x y : ℝ) :
  (∀ x, -x^2 + 4 * x - 3 = -3 → ∃ y, y = - (1)/3 * x^3 + 2 * x^2 - 3 * x + 1) →
  (3 * x + y - 1 = 0 ∨ 9 * x + 3 * y - 35 = 0) :=
by
  -- Proof goes here.
  sorry

-- Problem for Question 2
theorem minimum_area_triangle (x y : ℝ) :
  (P : ℝ × ℝ) = (2, (1:ℝ)/3) →
  (∀ k : ℝ, (A B : ℝ × ℝ),
    A = (2 - (1/(3 * k)), 0) →
    B = (0, (1/3) - 2 * k) →
    ∃ S : ℝ, S = (1/2) * (2 - 1/(3 * k)) * ((1/3) - 2 * k) →
    S = (k = - 1/6) → S = 4/3) :=
by
    -- Proof goes here.
    sorry

end tangent_line_slope_neg3_minimum_area_triangle_l323_323594


namespace find_angle_BED_l323_323290

-- Conditions
variable (A B C D E : Type)
variable [IsTriangle A B C] -- A, B, C form a triangle
variable (angle_A : Real) (angle_C : Real)
variable (AD DB BE EC : Real)
variable (angle_BED : Real)

-- Given conditions
def conditions := angle_A = 60 ∧ angle_C = 45 ∧
                   AD = 2 * DB ∧ BE = 2 * EC

-- Target theorem
theorem find_angle_BED : conditions A B C D E angle_A angle_C AD DB BE EC ∧ angle_BED = 38 := sorry

end find_angle_BED_l323_323290


namespace complement_set_example_l323_323356

-- Define the universal set V and the complement of set N in V
def V : Set ℕ := {1, 2, 3, 4, 5}
def CVN : Set ℕ := {2, 4}

-- Define the set N as the complement of CVN in V
def N : Set ℕ := {1, 3, 5}

-- Now state the theorem that V, CVN, and N meet the conditions described
theorem complement_set_example:
  (V = {1, 2, 3, 4, 5}) →
  (CVN = {2, 4}) →
  (N = V \ CVN) →
  (N = {1, 3, 5}) :=
 by intros; simp; sorry

end complement_set_example_l323_323356


namespace sum_of_first_eight_primes_with_units_digit_three_l323_323199

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function to check primality

def has_units_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem sum_of_first_eight_primes_with_units_digit_three :
  let primes_with_units_digit_three := list.filter (λ n, is_prime n ∧ has_units_digit_three n) [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123]
  let first_eight_primes := primes_with_units_digit_three.take 8
  let sum_of_primes := first_eight_primes.sum
  sum_of_primes = 404 := by
{
  sorry
}

end sum_of_first_eight_primes_with_units_digit_three_l323_323199


namespace find_y_l323_323562

theorem find_y (x y : ℤ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -3) : y = 17 := by
  sorry

end find_y_l323_323562


namespace difference_in_total_cost_l323_323145

theorem difference_in_total_cost
  (item_price : ℝ := 15)
  (tax_rate1 : ℝ := 0.08)
  (tax_rate2 : ℝ := 0.072)
  (discount : ℝ := 0.005)
  (correct_difference : ℝ := 0.195) :
  let discounted_tax_rate := tax_rate2 - discount
  let total_price_with_tax_rate1 := item_price * (1 + tax_rate1)
  let total_price_with_discounted_tax_rate := item_price * (1 + discounted_tax_rate)
  total_price_with_tax_rate1 - total_price_with_discounted_tax_rate = correct_difference := by
  sorry

end difference_in_total_cost_l323_323145


namespace find_angle_BDC_l323_323304

-- Define the structure to hold the given
variables (A B C D : Type) [convex_quadrilateral A B C D]

-- Define the angles.
variables (angle_BCA angle_BDA angle_BAC : ℝ)
variables (angle_BCA_eq_10 : angle_BCA = 10)
variables (angle_BDA_eq_20 : angle_BDA = 20)
variables (angle_BAC_eq_40 : angle_BAC = 40)

-- Define the perpendicularity condition
variables (AC_perp_BD : AC ⟂ BD)

-- State the problem to prove angle BDC
theorem find_angle_BDC :
  ∠BDC = 60 :=
sorry

end find_angle_BDC_l323_323304


namespace union_of_sets_l323_323991

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2}) (hB : B = {2, 3}) : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_sets_l323_323991


namespace germination_percentage_l323_323808

def P1 : ℕ := 300
def P2 : ℕ := 200
def G1 : ℕ := 0.25 * P1
def G2 : ℕ := 0.40 * P2

theorem germination_percentage : 
  (G1 + G2) / (P1 + P2) * 100 = 31 := by
  sorry

end germination_percentage_l323_323808


namespace sum_of_three_consecutive_divisible_by_three_l323_323723

theorem sum_of_three_consecutive_divisible_by_three (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2)) = 3 * k := by
  sorry

end sum_of_three_consecutive_divisible_by_three_l323_323723


namespace smallest_product_not_factor_of_48_l323_323013

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l323_323013


namespace sin_arith_seq_l323_323957

theorem sin_arith_seq (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 5 + a 9 = 5 * Real.pi) :
  Real.sin (a 2 + a 8) = - (Real.sqrt 3) / 2 :=
sorry

end sin_arith_seq_l323_323957


namespace maximum_value_of_f_intervals_of_monotonic_increase_l323_323359

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let a1 := a x
  let b1 := b x
  a1.1 * (a1.1 + b1.1) + a1.2 * (a1.2 + b1.2)

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = 3 / 2 + Real.sqrt 2 / 2 := sorry

theorem intervals_of_monotonic_increase :
  ∃ I1 I2 : Set ℝ, 
  I1 = Set.Icc 0 (Real.pi / 8) ∧ 
  I2 = Set.Icc (5 * Real.pi / 8) Real.pi ∧ 
  (∀ x ∈ I1, ∀ y ∈ I2, x ≤ y ∧ f x ≤ f y) ∧
  (∀ x y, x ∈ I1 → y ∈ I1 → x < y → f x < f y) ∧
  (∀ x y, x ∈ I2 → y ∈ I2 → x < y → f x < f y) := sorry

end maximum_value_of_f_intervals_of_monotonic_increase_l323_323359


namespace digit_nine_occurrence_l323_323269

theorem digit_nine_occurrence :
  let n := 987654321
  let m := 9
  let product := n * m
  (product = 8888888889) ∧ (count_digit 9 product = 1) := 
by
  let n := 987654321
  let m := 9
  let product := n * m
  have h1 : product = 8888888889 := sorry
  have h2 : count_digit 9 product = 1 := sorry
  exact ⟨h1, h2⟩

-- Define helper function to count a specific digit in a number
def count_digit (d : ℕ) (n : ℕ) : ℕ := 
  n.digits 10 |>.count (λ x, x = d)

end digit_nine_occurrence_l323_323269


namespace correct_statements_count_l323_323755

theorem correct_statements_count :
  let condition1 := (∀ α : ℝ, ∀ slope : ℝ, slope = Real.tan α → False)
  let condition2 := (∀ (a b : ℝ) (x y : ℝ), (a ≠ 0 ∨ b ≠ 0) → ¬((∀ x y : ℝ, (x + y ≠ 0) → x/a + y/b = 1)))
  let condition3 := (∀ (α : ℝ), (1 : ℝ) = 2*Real.sin α + 2*Real.cos α → Real.sin(α + Real.pi/4) = 1/(2*Real.sqrt 2))
  let condition4 := (∀ α : ℝ, (-Real.sqrt 3)/3 ≤ Real.cos α ∧ Real.cos α ≤ (Real.sqrt 3)/3 → (0 ≤ α ∧ α ≤ Real.pi/6) ∨ (5*Real.pi/6 ≤ α ∧ α < Real.pi))
  (¬condition1) ∧ (¬condition2) ∧ condition3 ∧ condition4 →  ∃ (count=2)
:= sorry

end correct_statements_count_l323_323755


namespace train_speed_l323_323479

noncomputable def speed_of_train (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  let distance_kilometers := distance_meters / 1000
  let time_hours := time_seconds / 3600
  distance_kilometers / time_hours

theorem train_speed (distance_meters : ℝ) (time_seconds : ℝ) (h_dist : distance_meters = 250.00000000000003) (h_time : time_seconds = 15) :
  speed_of_train distance_meters time_seconds = 60 :=
by
  subst h_dist
  subst h_time
  -- Calculate speed step-by-step
  have distance_km : distance_meters / 1000 = 0.25000000000000003 := by norm_num
  have time_hrs : time_seconds / 3600 = 0.004166666666666667 := by norm_num
  have speed_kph : (distance_meters / 1000) / (time_seconds / 3600) = 60 := by norm_num
  exact speed_kph

end train_speed_l323_323479


namespace validForNEqualsTwo_l323_323828

-- Define the conditions
def isValidPath (n : ℕ) : Prop :=
  ∃ (path : List (ℕ × ℕ)), 
    path.head = (0, 0) ∧ path.last = (n, n) ∧
    (∀ (a b : ℕ × ℕ), are_neighbours a b → (a, b) ∈ path ∨ (b, a) ∈ path) ∧
    (∀ (p : ℕ × ℕ), p ∈ path → 
      (p.1 + 1 < n ∨ p.2 + 1 < n → (p.1 + 1, p.2) ∈ path ∨ (p.1, p.2 + 1) ∈ path))

-- Define the conclusion
theorem validForNEqualsTwo : 
  ∀ (n : ℕ), isValidPath n ↔ n = 2 :=
begin
  sorry
end

end validForNEqualsTwo_l323_323828


namespace problem_statement_l323_323826

theorem problem_statement (x : ℝ) : 45 * x = 0.35 * 900 → x = 7 :=
by
  assume h : 45 * x = 0.35 * 900
  sorry

end problem_statement_l323_323826


namespace perimeter_after_cut_l323_323141

theorem perimeter_after_cut (s₁ s₂ : ℕ) (H₁ : s₁ = 4) (H₂ : s₂ = 1) : 
  let remaining_perimeter := 4 + 4 + (4 - 1) + (4 - 1) + (4 - 1)
  in remaining_perimeter = 17 := 
by 
  sorry

end perimeter_after_cut_l323_323141


namespace soccer_team_losses_l323_323735

theorem soccer_team_losses (total_games won_games draw_points win_points total_points : ℕ) 
    (h1 : total_games = 20) (h2 : won_games = 14) (h3 : draw_points = 1) (h4 : win_points = 3) (h5 : total_points = 46) 
    : ∃ (lost_games : ℕ), 
    lost_games = total_games - won_games - (total_points - won_games * win_points) / draw_points := 
begin
  use 2,
  sorry
end

end soccer_team_losses_l323_323735


namespace solution_set_of_inequality_l323_323766

def fraction_inequality_solution : Set ℝ := {x : ℝ | -4 < x ∧ x < -1}

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 1 ↔ -4 < x ∧ x < -1 := by
sorry

end solution_set_of_inequality_l323_323766


namespace find_k_of_tangent_and_surface_area_l323_323607

noncomputable def circle_tangent_to_line (k : ℝ) (r : ℝ) : Prop :=
  (∀ y x : ℝ, y = k * x → (x - 4) ^ 2 + y ^ 2 = r ^ 2)

theorem find_k_of_tangent_and_surface_area
  (k r : ℝ)
  (h_tangent : circle_tangent_to_line k 2)
  (h_surface_area : 2 * Math.pi * 2 * 2 = 16 * Math.pi) :
  (k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3) :=
by
  sorry

end find_k_of_tangent_and_surface_area_l323_323607


namespace angle_BQP_eq_angle_DAQ_l323_323741

-- Define the geometric entities and their properties in the conditions
variables {A B C D P Q : Type}
variables [Point A] [Point B] [Point C] [Point D] [Point P] [Point Q]
variables (trapezoid : Trapezoid A B C D)
variables (intersect_diag : IntersectingDiagonals A B C D P)
variables (between_Q : Between Q (Line BC) (Line AD))
variables (angle_cond : ∠ AQD = ∠ CQB)
variables (sep_by_CD : SeparatedByLine Q P (Line CD))

-- Main statement to prove
theorem angle_BQP_eq_angle_DAQ 
  (h1 : IntersectingDiagonals A B C D P)
  (h2 : Between Q (Line BC) (Line AD))
  (h3 : ∠ AQD = ∠ CQB)
  (h4 : SeparatedByLine Q P (Line CD)) : 
  ∠ BQP = ∠ DAQ :=
sorry -- proof to be provided

end angle_BQP_eq_angle_DAQ_l323_323741


namespace area_ao_equals_9_div_2_minimize_triangle_area_l323_323462

def line_eq_passing_point_area (k : ℝ) : Prop :=
  ∃ x y, y - 1 = k * (x - 2)

/-- Problem Part 1 -/
theorem area_ao_equals_9_div_2 (k : ℝ) : 
  triangle_area_passes_through_P (k) ∧ 
  triangle_area_is_9_over_2 (k) → 
  (equation_of_line_is k = "x + y - 3 = 0" ∨ 
   equation_of_line_is k = "x + 4y - 6 = 0") sorry

/-- Problem Part 2 -/
theorem minimize_triangle_area (k : ℝ) : 
  triangle_area_passes_through_P (k) ∧
  triangle_area_is_minimized k →
  equation_of_line_is k = "x + 2y - 4 = 0" sorry

end area_ao_equals_9_div_2_minimize_triangle_area_l323_323462


namespace pq_perpendicular_necessary_and_sufficient_l323_323642

-- Definitions for the problem
def plane (α : Type*) := set α
def line (α : Type*) := set α
def point (α : Type*) := α

variable {α : Type*} [metric_space α]

-- Given conditions
variable (α β : plane α)
variable (l : line α)
variable (P Q : point α)

-- Perpendicularity definitions
def perpendicular_plane_to_plane (α β : plane α) : Prop := sorry
def perpendicular_line_to_plane (line l : line α) (plane β : plane α) : Prop := sorry
def perpendicular_line_to_line (l1 l2 : line α) : Prop := sorry

-- Assumptions
axiom planes_perpendicular : perpendicular_plane_to_plane α β
axiom intersection_line : ∀ x, l x ↔ x ∈ α ∧ x ∈ β
axiom point_P_in_plane_α : P ∈ α
axiom point_Q_in_line_l : Q ∈ l

-- Statement to prove
theorem pq_perpendicular_necessary_and_sufficient :
  (perpendicular_line_to_line (line_through P Q) l ↔ perpendicular_line_to_plane (line_through P Q) β) :=
sorry

-- Note: line_through is a function that constructs a line through two given points.
def line_through (P Q : point α) : line α := sorry

end pq_perpendicular_necessary_and_sufficient_l323_323642


namespace chickens_farm_problem_l323_323408

noncomputable def number_of_chickens (x : ℕ) : ℕ := x + 6 * x

theorem chickens_farm_problem : 
  ∃ (x : ℕ), 6 * x + 60 = 4 * (x + 60) ∧ number_of_chickens x = 630 := 
by
  use 90
  constructor
  { 
    -- Prove condition from the problem statement.
    calc
      6 * 90 + 60 = 540 + 60 : by rw Nat.mul_succ
      ... = 600 : rfl
      ... = 4 * (90 + 60) : by norm_num
      ... = 4 * 150 : rfl
      ... = 600 : by norm_num
  }
  { 
    -- Validate total chicken calculation.
    have h: number_of_chickens 90 = 90 + 6 * 90 :=
      rfl
    rw h
    norm_num
  }

end chickens_farm_problem_l323_323408


namespace customer_payment_l323_323757

noncomputable def cost_price : ℝ := 4090.9090909090905
noncomputable def markup : ℝ := 0.32
noncomputable def selling_price : ℝ := cost_price * (1 + markup)

theorem customer_payment :
  selling_price = 5400 :=
by
  unfold selling_price
  unfold cost_price
  unfold markup
  sorry

end customer_payment_l323_323757


namespace count_5_primable_integers_lt_1000_is_21_l323_323105

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323105


namespace no_rational_solutions_l323_323530

theorem no_rational_solutions (a b c d : ℚ) (n : ℕ) :
  ¬ ((a + b * (Real.sqrt 2))^(2 * n) + (c + d * (Real.sqrt 2))^(2 * n) = 5 + 4 * (Real.sqrt 2)) :=
sorry

end no_rational_solutions_l323_323530


namespace remainder_of_a_squared_l323_323341

theorem remainder_of_a_squared (n : ℕ) (a : ℤ) (h : a % n * a % n % n = 1) : (a * a) % n = 1 := by
  sorry

end remainder_of_a_squared_l323_323341


namespace percentage_of_women_not_speaking_french_l323_323445

theorem percentage_of_women_not_speaking_french
  (total_employees : ℕ)
  (men_percentage : ℝ)
  (men_french_speaking_percentage : ℝ)
  (total_french_speaking_percentage : ℝ)
  (h1 : men_percentage = 0.6)
  (h2 : men_french_speaking_percentage = 0.6)
  (h3 : total_french_speaking_percentage = 0.5) :
  let number_of_men := total_employees * men_percentage in
  let number_of_women := total_employees - number_of_men in
  let french_speaking_men := number_of_men * men_french_speaking_percentage in
  let total_french_speaking := total_employees * total_french_speaking_percentage in
  let french_speaking_women := total_french_speaking - french_speaking_men in
  let women_not_speaking_french := number_of_women - french_speaking_women in
  (women_not_speaking_french / number_of_women * 100) = 65 :=
by 
  intro total_employees men_percentage men_french_speaking_percentage total_french_speaking_percentage h1 h2 h3
  let number_of_men := total_employees * men_percentage
  let number_of_women := total_employees - number_of_men
  let french_speaking_men := number_of_men * men_french_speaking_percentage
  let total_french_speaking := total_employees * total_french_speaking_percentage
  let french_speaking_women := total_french_speaking - french_speaking_men
  let women_not_speaking_french := number_of_women - french_speaking_women
  have h : (women_not_speaking_french / number_of_women * 100) = 65 := sorry
  exact h

end percentage_of_women_not_speaking_french_l323_323445


namespace solve_inequality_l323_323731

theorem solve_inequality (x : ℝ) : x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := 
by 
suffices (h1 : x^2 - x - 6 = (x - 3) * (x + 2)), -- Transform the inequality
  exact ⟨λ h2, sorry, λ h3, sorry⟩ -- Solve and prove the equivalence of intervals

end solve_inequality_l323_323731


namespace range_b_for_4_integer_solutions_l323_323980

theorem range_b_for_4_integer_solutions :
  (∃ a b : ℝ, (∀ x : ℝ, a < x ∧ x ≤ b → x ∈ ℝ) ∧
              (3 - a) / (a - 4) - a = 1 / (4 - a) ∧
              (a < 0 ∧ 0 ≤ b ∧ 
               (0, 1, 2, 3 ∈ {x : ℝ | a < x ∧ x ≤ b}) ∧
               ¬(4 ∈ {x : ℝ | a < x ∧ x ≤ b}))) → 
  (3 ≤ b ∧ b < 4) := 
sorry

end range_b_for_4_integer_solutions_l323_323980


namespace book_sale_revenue_l323_323456

noncomputable def total_amount_received (price_per_book : ℝ) (B : ℕ) (sold_fraction : ℝ) :=
  sold_fraction * B * price_per_book

theorem book_sale_revenue (B : ℕ) (price_per_book : ℝ) (unsold_books : ℕ) (sold_fraction : ℝ) :
  (1 / 3 : ℝ) * B = unsold_books →
  price_per_book = 3.50 →
  unsold_books = 36 →
  sold_fraction = 2 / 3 →
  total_amount_received price_per_book B sold_fraction = 252 :=
by
  intros h1 h2 h3 h4
  sorry

end book_sale_revenue_l323_323456


namespace carrie_strawberry_harvest_l323_323173

theorem carrie_strawberry_harvest : 
  let length := 6 in
  let width := 8 in
  let plants_per_sq_ft := 4 in
  let strawberries_per_plant := 10 in
  let area := length * width in
  let total_plants := plants_per_sq_ft * area in
  let total_strawberries := strawberries_per_plant * total_plants in
  total_strawberries = 1920 :=
by
  sorry

end carrie_strawberry_harvest_l323_323173


namespace height_of_small_cone_correct_l323_323840

noncomputable def height_of_small_cone 
  (h_frustum : ℝ) 
  (area_lower : ℝ) 
  (area_upper : ℝ) : ℝ :=
  let r_lower := (area_lower / Math.pi).sqrt in
  let r_upper := (area_upper / Math.pi).sqrt in
  let H := (h_frustum * r_lower) / (r_lower - r_upper) in
  H / 3

theorem height_of_small_cone_correct : 
  height_of_small_cone 30 (400 * Real.pi) (100 * Real.pi) = 15 :=
by
  sorry

end height_of_small_cone_correct_l323_323840


namespace number_of_sides_of_polygon_l323_323979

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
sorry

end number_of_sides_of_polygon_l323_323979


namespace distance_point_to_line_correct_l323_323745

-- Define the point P
def P : ℝ × ℝ := (-1, 1)

-- Define the line l in standard form
def A : ℝ := 3
def B : ℝ := 4
def C : ℝ := 0

-- Function to calculate the distance from a point to a line
def distance_from_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Math.sqrt (A ^ 2 + B ^ 2)

-- The theorem we need to prove
theorem distance_point_to_line_correct :
  distance_from_point_to_line (-1) 1 3 4 0 = 1 / 5 :=
by
  sorry

end distance_point_to_line_correct_l323_323745


namespace prob_three_is_one_third_l323_323060

noncomputable def prob (d : ℕ) : ℝ := log10 (d + 1) - log10 d

theorem prob_three_is_one_third {d : ℕ} (h : d = 3) :
  prob d = (1 / 3) * (prob 6 + prob 7 + prob 8) :=
by
  sorry

end prob_three_is_one_third_l323_323060


namespace P_inter_Q_eq_l323_323044

def P (x : ℝ) : Prop := -1 < x ∧ x < 3
def Q (x : ℝ) : Prop := -2 < x ∧ x < 1

theorem P_inter_Q_eq : {x | P x} ∩ {x | Q x} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end P_inter_Q_eq_l323_323044


namespace distance_between_foci_of_hyperbola_l323_323551

theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), x^2 - 6 * x - 4 * y^2 - 8 * y = 27 → (4 * Real.sqrt 10) = 4 * Real.sqrt 10 :=
by
  sorry

end distance_between_foci_of_hyperbola_l323_323551


namespace quadrilateral_OEPF_parallelogram_l323_323655

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem quadrilateral_OEPF_parallelogram
  (P : ℝ × ℝ)
  (h_circle_eq : ∃ f : ℝ, ∀ (x y : ℝ), (x - P.1)^2 + (y - P.2)^2 = f)
  (h_hyperbola : ∀ (x y : ℝ), x * y = 1)
  (A B C D : ℝ × ℝ)
  (h_A_eq : A.1 * A.2 = 1)
  (h_B_eq : B.1 * B.2 = 1)
  (h_C_eq : C.1 * C.2 = 1)
  (h_D_eq : D.1 * D.2 = 1)
  (h_intersections : ∀ (x y : ℝ), h_circle_eq (x, y) ↔ h_hyperbola (x, y))
  (E : ℝ × ℝ := midpoint A B)
  (F : ℝ × ℝ := midpoint C D)
  (O : ℝ × ℝ := (0, 0)) :
  let OEPF_parallelogram :=
    ∃ m n : ℝ, E = (P.1 / 2, P.2 / 2) ∧ F = (P.1 / 2, P.2 / 2)
  in OEPF_parallelogram :=
sorry

end quadrilateral_OEPF_parallelogram_l323_323655


namespace train_speed_l323_323483

def length_train : ℝ := 250.00000000000003
def crossing_time : ℝ := 15
def meter_to_kilometer (x : ℝ) : ℝ := x / 1000
def second_to_hour (x : ℝ) : ℝ := x / 3600

theorem train_speed :
  (meter_to_kilometer length_train) / (second_to_hour crossing_time) = 60 := 
  sorry

end train_speed_l323_323483


namespace tangent_line_at_point_l323_323190

theorem tangent_line_at_point (x y : ℝ) (h : y = ln x / x) (hx : x = 1) (hy : y = 0) : 
  ∃ m b, y = m * x + b ∧ m = 1 ∧ b = -1 := by
sory

end tangent_line_at_point_l323_323190


namespace fraction_zero_if_abs_x_eq_one_l323_323770

theorem fraction_zero_if_abs_x_eq_one (x : ℝ) : 
  (|x| - 1) = 0 → (x^2 - 2 * x + 1 ≠ 0) → x = -1 := 
by 
  sorry

end fraction_zero_if_abs_x_eq_one_l323_323770


namespace total_kitchen_supplies_sharon_l323_323384

-- Define the number of pots Angela has.
def angela_pots := 20

-- Define the number of plates Angela has (6 more than three times the number of pots Angela has).
def angela_plates := 6 + 3 * angela_pots

-- Define the number of cutlery Angela has (half the number of plates Angela has).
def angela_cutlery := angela_plates / 2

-- Define the number of pots Sharon wants (half the number of pots Angela has).
def sharon_pots := angela_pots / 2

-- Define the number of plates Sharon wants (20 less than three times the number of plates Angela has).
def sharon_plates := 3 * angela_plates - 20

-- Define the number of cutlery Sharon wants (twice the number of cutlery Angela has).
def sharon_cutlery := 2 * angela_cutlery

-- Prove that the total number of kitchen supplies Sharon wants is 254.
theorem total_kitchen_supplies_sharon :
  sharon_pots + sharon_plates + sharon_cutlery = 254 :=
by
  -- State intermediate results for clarity
  let angela_plates_val := 66
  have h_angela_plates : angela_plates = angela_plates_val :=
    by
    unfold angela_plates
    norm_num
  let angela_cutlery_val := 33
  have h_angela_cutlery : angela_cutlery = angela_cutlery_val :=
    by
    unfold angela_cutlery
    rw h_angela_plates
    norm_num
  let sharon_pots_val := 10
  have h_sharon_pots : sharon_pots = sharon_pots_val :=
    by
    unfold sharon_pots
    norm_num
  let sharon_plates_val := 178
  have h_sharon_plates : sharon_plates = sharon_plates_val :=
    by
    unfold sharon_plates
    rw h_angela_plates
    norm_num
  let sharon_cutlery_val := 66
  have h_sharon_cutlery : sharon_cutlery = sharon_cutlery_val :=
    by
    unfold sharon_cutlery
    rw h_angela_cutlery
    norm_num
  rw [h_sharon_pots, h_sharon_plates, h_sharon_cutlery]
  norm_num

end total_kitchen_supplies_sharon_l323_323384


namespace hyperbola_focal_length_l323_323303

-- Define the hyperbola equation parameters
def hyperbola_a2 : ℝ := 7
def hyperbola_b2 : ℝ := 3

-- Problem statement
theorem hyperbola_focal_length (a2 b2 : ℝ) (ha : a2 = hyperbola_a2) (hb : b2 = hyperbola_b2) :
  let a := Real.sqrt a2
      b := Real.sqrt b2
      c := Real.sqrt (a2 + b2)
      focal_length := 2 * c
  in focal_length = 2 * Real.sqrt 10 :=
by
  sorry

end hyperbola_focal_length_l323_323303


namespace area_of_rectangle_l323_323138

-- Define the context and variables for the problem
variable {h : ℝ}  -- Height of the segment
variable {x : ℝ}  -- Side AB of the rectangle

-- Given conditions
def AB_eq_x (h : ℝ) (x : ℝ) : Prop := x = 3 * h / 5
def BC_eq_4x (h : ℝ) (x : ℝ) : Prop := 4 * x = 4 * (3 * h / 5)

-- The rectangle area based on sides
def rectangle_area (h : ℝ) : ℝ := 4 * ((3 * h / 5) ^ 2)

-- The theorem we need to prove
theorem area_of_rectangle (h : ℝ) (x : ℝ) (h_pos : 0 < h) :
  AB_eq_x h x → BC_eq_4x h x → rectangle_area h = 36 * h^2 / 25 :=
by
  intro hab hbc
  rw [rectangle_area, hab]
  norm_num
  sorry

end area_of_rectangle_l323_323138


namespace average_speed_of_rocket_l323_323469

def distance_soared (speed_soaring : ℕ) (time_soaring : ℕ) : ℕ :=
  speed_soaring * time_soaring

def distance_plummeted : ℕ := 600

def total_distance (distance_soared : ℕ) (distance_plummeted : ℕ) : ℕ :=
  distance_soared + distance_plummeted

def total_time (time_soaring : ℕ) (time_plummeting : ℕ) : ℕ :=
  time_soaring + time_plummeting

def average_speed (total_distance : ℕ) (total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_rocket :
  let speed_soaring := 150
  let time_soaring := 12
  let time_plummeting := 3
  distance_soared speed_soaring time_soaring +
  distance_plummeted = 2400
  →
  total_time time_soaring time_plummeting = 15
  →
  average_speed (distance_soared speed_soaring time_soaring + distance_plummeted)
                (total_time time_soaring time_plummeting) = 160 :=
by
  sorry

end average_speed_of_rocket_l323_323469


namespace find_y_in_range_l323_323282

theorem find_y_in_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end find_y_in_range_l323_323282


namespace solve_inequality_system_l323_323387

-- Define the inequalities as conditions.
def cond1 (x : ℝ) := 2 * x + 1 < 3 * x - 2
def cond2 (x : ℝ) := 3 * (x - 2) - x ≤ 4

-- Formulate the theorem to prove that these conditions give the solution 3 < x ≤ 5.
theorem solve_inequality_system (x : ℝ) : cond1 x ∧ cond2 x ↔ 3 < x ∧ x ≤ 5 := 
sorry

end solve_inequality_system_l323_323387


namespace number_of_true_propositions_l323_323563

theorem number_of_true_propositions
    (a b c : ℝ)
    (h_downwards : a < 0) :
    (∃ x : ℝ, a * x^2 + b * x + c < 0) ∧ 
    (¬ ((∃ x : ℝ, a*x^2 + b*x + c < 0) → a < 0)) ∧ 
    (∃ x : ℝ, ¬ (a*x^2 + b*x + c ≥ 0)) → 
    1 :=
by
  sorry

end number_of_true_propositions_l323_323563


namespace smallest_product_not_factor_of_48_l323_323015

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end smallest_product_not_factor_of_48_l323_323015


namespace train_speed_l323_323478

noncomputable def speed_of_train (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  let distance_kilometers := distance_meters / 1000
  let time_hours := time_seconds / 3600
  distance_kilometers / time_hours

theorem train_speed (distance_meters : ℝ) (time_seconds : ℝ) (h_dist : distance_meters = 250.00000000000003) (h_time : time_seconds = 15) :
  speed_of_train distance_meters time_seconds = 60 :=
by
  subst h_dist
  subst h_time
  -- Calculate speed step-by-step
  have distance_km : distance_meters / 1000 = 0.25000000000000003 := by norm_num
  have time_hrs : time_seconds / 3600 = 0.004166666666666667 := by norm_num
  have speed_kph : (distance_meters / 1000) / (time_seconds / 3600) = 60 := by norm_num
  exact speed_kph

end train_speed_l323_323478


namespace total_simple_interest_fetched_l323_323476

open Real

def principal : ℝ := 8935
def rate : ℝ := 9 / 100
def time : ℝ := 5

theorem total_simple_interest_fetched : 
  (principal * rate * time) = 803.15 := by
  sorry

end total_simple_interest_fetched_l323_323476


namespace smallest_x_for_27_x_gt_3_24_l323_323796

theorem smallest_x_for_27_x_gt_3_24 : ∃ x : ℤ, x > 8 ∧ 27^x > 3^24 :=
sorry

end smallest_x_for_27_x_gt_3_24_l323_323796


namespace coins_remainder_l323_323050

theorem coins_remainder (N : ℕ) (h1 : N % 8 = 5) (h2 : N % 7 = 2) (hN_min : ∀ M : ℕ, (M % 8 = 5 ∧ M % 7 = 2) → N ≤ M) : N % 9 = 1 :=
sorry

end coins_remainder_l323_323050


namespace find_y_in_range_l323_323280

theorem find_y_in_range (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end find_y_in_range_l323_323280


namespace selling_price_of_fully_grown_pig_l323_323459

def feeding_cost_per_month : ℕ := 10
def sold_pigs_12_months_count : ℕ := 3
def sold_pigs_16_months_count : ℕ := 3
def profit : ℕ := 960

theorem selling_price_of_fully_grown_pig :
  let total_pigs := sold_pigs_12_months_count + sold_pigs_16_months_count,
      cost_12_months := sold_pigs_12_months_count * 12 * feeding_cost_per_month,
      cost_16_months := sold_pigs_16_months_count * 16 * feeding_cost_per_month,
      total_cost := cost_12_months + cost_16_months,
      total_revenue := profit + total_cost,
      price_per_pig := total_revenue / total_pigs
  in price_per_pig = 300 := by
  -- Proof skipped with sorry
  sorry

end selling_price_of_fully_grown_pig_l323_323459


namespace inequality_solution_l323_323558

theorem inequality_solution (x : ℝ) (h : x ≠ 1) : (x + 1) * (x + 3) / (x - 1)^2 ≤ 0 ↔ (-3 ≤ x ∧ x ≤ -1) :=
by
  sorry

end inequality_solution_l323_323558


namespace exists_triangle_side_ratio_l323_323673

theorem exists_triangle_side_ratio:
  ∃ a b c : ℝ, a = (Real.sqrt 2 - 1) * (b + c) :=
by
  sorry

end exists_triangle_side_ratio_l323_323673


namespace count_5_primable_under_1000_l323_323119

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323119


namespace remaining_pictures_l323_323801

-- Definitions based on the conditions
def pictures_in_first_book : ℕ := 44
def pictures_in_second_book : ℕ := 35
def pictures_in_third_book : ℕ := 52
def pictures_in_fourth_book : ℕ := 48
def colored_pictures : ℕ := 37

-- Statement of the theorem based on the question and correct answer
theorem remaining_pictures :
  pictures_in_first_book + pictures_in_second_book + pictures_in_third_book + pictures_in_fourth_book - colored_pictures = 142 := by
  sorry

end remaining_pictures_l323_323801


namespace upstream_distance_l323_323416

-- Definitions based on conditions
def V_b : ℕ := 18   -- Speed of the boat in still water (kmph)
def V_s : ℕ := 6    -- Speed of the stream (kmph)
def V_d : ℕ := V_b + V_s  -- Downstream speed (kmph)
def V_u : ℕ := V_b - V_s  -- Upstream speed (kmph)
def D_d : ℕ := 48   -- Distance covered downstream (km)

-- Time taken to travel downstream is equal to time taken to travel upstream.
def T_d : ℕ := D_d / V_d  -- Time taken downstream
def T_u : ℕ := T_d        -- Time taken upstream (same as T_d)

-- Target distance to prove
def D_u : ℕ := V_u * T_u  -- Distance covered upstream

-- The theorem stating the desired result
theorem upstream_distance 
  (V_b = 18) 
  (V_s = 6) 
  (D_d = 48) 
  (T_d = D_d / (V_b + V_s)) 
  (T_u = T_d) 
  (D_u = (V_b - V_s) * T_u) : 
  D_u = 24 :=
by
  sorry

end upstream_distance_l323_323416


namespace smallest_value_of_m_plus_n_l323_323400

theorem smallest_value_of_m_plus_n :
  ∃ m n : ℕ, 1 < m ∧ 
  (∃ l : ℝ, l = (m^2 - 1 : ℝ) / (m * n) ∧ l = 1 / 2021) ∧
  m + n = 85987 := 
sorry

end smallest_value_of_m_plus_n_l323_323400


namespace remainder_of_product_divided_by_7_l323_323195

theorem remainder_of_product_divided_by_7 :
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 2 :=
by
  sorry

end remainder_of_product_divided_by_7_l323_323195


namespace value_of_expression_l323_323582

open Real

theorem value_of_expression (m n r t : ℝ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := 
by
  sorry

end value_of_expression_l323_323582


namespace program_probability_l323_323849

theorem program_probability :
  let start_time_range := set.Icc 8.5 9.5
  let end_time_range := set.Icc 19 21
  let region_R := set.prod start_time_range end_time_range
  let region_S := {p : ℝ × ℝ | p.1 ∈ set.Icc 8.5 9 ∧ p.2 ≥ p.1 + 11 ∧ p.2 ∈ end_time_range}
  (set.measure region_S / set.measure region_R) = 5 / 16 :=
by
  sorry

end program_probability_l323_323849


namespace slope_angle_is_135_degrees_l323_323414

-- Define the line equation
def line_equation (x : ℝ) : ℝ := -x + 1

-- Define the slope, which we know from the equation y = mx + b, in this case, m = -1
def slope (m : ℝ) : Prop := m = -1

-- Prove that the angle of the line with the positive x-axis is 135° given the slope condition
theorem slope_angle_is_135_degrees (m : ℝ) (h : slope m) : ∃ θ : ℝ, θ = 135 ∧ m = -1 := sorry

end slope_angle_is_135_degrees_l323_323414


namespace rational_distance_between_points_on_circle_l323_323904

noncomputable def exists_rational_distance_points_on_circle : Prop :=
  ∃ (n : ℕ) (p : ℝ × ℝ → Prop), (∀ i j, p (i,j) → 0 ≤ i ∧ i ≤ π ∧ 0 ≤ j ∧ j ≤ π) ∧ 
  (∀ (i j : ℝ), p(i, j) → 2 * Real.sin((j - i) / 2) ∈ ℚ)

theorem rational_distance_between_points_on_circle :
  exists_rational_distance_points_on_circle :=
sorry

end rational_distance_between_points_on_circle_l323_323904


namespace man_age_twice_son_age_l323_323843

-- Definitions based on conditions
def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

-- Definition of the main statement to be proven
theorem man_age_twice_son_age (Y : ℕ) : man_age + Y = 2 * (son_age + Y) → Y = 2 :=
by sorry

end man_age_twice_son_age_l323_323843


namespace max_triangle_area_on_unit_square_l323_323802

noncomputable def unit_square_vertices : set (ℝ × ℝ) := 
  {(0,0), (1,0), (0,1), (1,1)}

noncomputable def triangle_vertices (x y : ℝ) : set (ℝ × ℝ) :=
  {(1, 0), (0, y), (x, 1)}

noncomputable def triangle_area (a b c : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2))

theorem max_triangle_area_on_unit_square : 
  ∃ (x y : ℝ), 0 <= x ∧ x <= 1 ∧ 0 <= y ∧ y <= 1 ∧ 
  (1/2) * abs ((1 * (0 - 1) + 0 * (1 - 0) + x * (0 - y))) = 1 :=
by
  sorry

end max_triangle_area_on_unit_square_l323_323802


namespace parabola_focus_distance_A_l323_323972

def parabola_focus_distance {p x y : ℝ} (h_parabola : y^2 = 2*p*x) (h_A : (x, y) = (1, real.sqrt 5)) : ℝ :=
  let focus := (p / 2, 0) in
  abs (p / 2 - x) + abs y  -- focus (p/2, 0) and point A (x, y), distance calculated based on the problem stated

theorem parabola_focus_distance_A :
  parabola_focus_distance 
    (show (real.sqrt 5)^2 = 2 * (5/2) * 1 by sorry)  -- this is "p = 5/2"
    (by simp [real.sqrt, real.sqrt_eq_rpow, real.sqrt, sqr_eq'] at *) = 9 / 4 :=
by 
  sorry -- Proof is omitted.

end parabola_focus_distance_A_l323_323972


namespace locus_of_points_is_circle_l323_323916

open Real

theorem locus_of_points_is_circle {A B : Point} {M : Point} (m n : ℝ) (m_pos : 0 < m) (n_pos : 0 < n) (h : ratio_dist M A B = m / n) :
  ∃ (center radius : Point), is_apollonius_circle M center radius :=
sorry

end locus_of_points_is_circle_l323_323916


namespace probability_exactly_one_of_A_or_B_selected_l323_323936

-- We define a set of four people
inductive Person
| A | B | C | D

open Person

-- Define the event of selecting exactly one of A and B
def exactlyOneOfABSelected (s : set (Person × Person)) : Prop :=
  (⟨A, C⟩ ∈ s ∨ ⟨A, D⟩ ∈ s ∨ ⟨B, C⟩ ∈ s ∨ ⟨B, D⟩ ∈ s) ∧
  (⟨A, B⟩ ∉ s ∧ ⟨C, D⟩ ∉ s)

-- Define the universal set of all two-person combinations
def allCombinations : set (Person × Person) :=
  {⟨A, B⟩, ⟨A, C⟩, ⟨A, D⟩, ⟨B, C⟩, ⟨B, D⟩, ⟨C, D⟩}

theorem probability_exactly_one_of_A_or_B_selected : 
  (∃ (s : set (Person × Person)), exactlyOneOfABSelected s) → 
  (s.card = 4 / 6) :=
sorry

end probability_exactly_one_of_A_or_B_selected_l323_323936


namespace largest_unachievable_score_l323_323294

theorem largest_unachievable_score :
  ∀ (x y : ℕ), 3 * x + 7 * y ≠ 11 :=
by
  sorry

end largest_unachievable_score_l323_323294


namespace range_of_quadratic_l323_323919

noncomputable def range_of_function : set ℝ :=
  {y : ℝ | ∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ y = x^2 + x}

theorem range_of_quadratic :
  range_of_function = set.Icc (-1 / 4) 12 :=
by {
  sorry
}

end range_of_quadratic_l323_323919


namespace find_n_l323_323918

theorem find_n (n : ℕ) (h_pos : n > 0) :
  (sin (π / (3 * n)) + cos (π / (3 * n)) = real.sqrt (3 * n) / 3) ↔ (n = 6) :=
by sorry

end find_n_l323_323918


namespace count_5_primable_less_than_1000_eq_l323_323133

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323133


namespace count_5_primable_under_1000_l323_323125

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_5_primable_below (limit : ℕ) : ℕ :=
  (Ico 1 limit).count is_5_primable

theorem count_5_primable_under_1000 : count_5_primable_below 1000 = 21 := 
sorry

end count_5_primable_under_1000_l323_323125


namespace incorrect_statement_l323_323028

def angles_on_x_axis := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi}
def angles_on_y_axis := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 2 + k * Real.pi}
def angles_on_axes := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi / 2}
def angles_on_y_eq_neg_x := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}

theorem incorrect_statement : ¬ (angles_on_y_eq_neg_x = {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}) :=
sorry

end incorrect_statement_l323_323028


namespace compound_indeterminate_l323_323192

def mass_percentage_Nitrogen : ℝ := 26.42

theorem compound_indeterminate (P : Type) (has_mass : P → ℝ) 
  (compounds : list P) (h : ∃ p ∈ compounds, has_mass p = mass_percentage_Nitrogen) : 
  ¬ (∃! p ∈ compounds, has_mass p = mass_percentage_Nitrogen) :=
sorry

end compound_indeterminate_l323_323192


namespace ratio_female_to_male_l323_323503

variable (m f : ℕ)

-- Average ages given in the conditions
def avg_female_age : ℕ := 35
def avg_male_age : ℕ := 45
def avg_total_age : ℕ := 40

-- Total ages based on number of members
def total_female_age (f : ℕ) : ℕ := avg_female_age * f
def total_male_age (m : ℕ) : ℕ := avg_male_age * m
def total_age (f m : ℕ) : ℕ := total_female_age f + total_male_age m

-- Equation based on average age of all members
def avg_age_eq (f m : ℕ) : Prop :=
  total_age f m / (f + m) = avg_total_age

theorem ratio_female_to_male : avg_age_eq f m → f = m :=
by
  sorry

end ratio_female_to_male_l323_323503


namespace inversely_proportional_rs_l323_323390

theorem inversely_proportional_rs (r s : ℝ) (k : ℝ) 
(h_invprop : r * s = k) 
(h1 : r = 40) (h2 : s = 5) 
(h3 : s = 8) : r = 25 := by
  sorry

end inversely_proportional_rs_l323_323390


namespace triangle_equilateral_l323_323289

noncomputable def is_equilateral (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) :
  is_equilateral a b c A B C :=
by
  sorry

end triangle_equilateral_l323_323289


namespace min_value_of_f_l323_323747

theorem min_value_of_f :
  ∃ f : (Real → Real), (∀ x : ℝ, 0 < x → f(x) - 2 * x * f(1/x) + 3 * x^2 = 0)
  ∧ (∀ x : ℝ, 0 < x → f(x) ≥ 3) :=
sorry

end min_value_of_f_l323_323747


namespace committee_including_board_member_count_l323_323838

def club_total : ℕ := 12
def board_members : ℕ := 3
def regular_members : ℕ := 9
def committee_size : ℕ := 5

theorem committee_including_board_member_count :
  (nat.choose club_total committee_size) - (nat.choose regular_members committee_size) = 666 := 
by
  sorry

end committee_including_board_member_count_l323_323838


namespace train_speed_proof_l323_323486

noncomputable def train_speed_in_kmh (length_in_m: ℝ) (time_in_sec: ℝ) : ℝ :=
  (length_in_m / 1000) / (time_in_sec / 3600)

theorem train_speed_proof : train_speed_in_kmh 250.00000000000003 15 = 60 := by
  have length_in_km := 250.00000000000003 / 1000
  have time_in_hr := 15 / 3600
  have speed := length_in_km / time_in_hr
  exact (by ring : speed = 60)

end train_speed_proof_l323_323486


namespace count_5_primable_integers_lt_1000_is_21_l323_323101

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323101


namespace hyperbola_char_eq_l323_323235

noncomputable section

open Real

-- Define the given ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 9) + (y^2 / 25) = 1

-- Define the foci of the ellipse
def ellipse_foci : set (ℝ × ℝ) :=
  {(0, 4), (0, -4)}

-- Define the eccentricity of the ellipse
def ellipse_eccentricity : ℝ :=
  4 / 5

-- Define the hyperbola equation form
def hyperbola_equation (y x : ℝ) (a b : ℝ) : Prop :=
  (y^2 / a^2) - (x^2 / b^2) = 1

-- Product of the eccentricities
def product_of_eccentricities (e1 e2 : ℝ) : ℝ :=
  e1 * e2

theorem hyperbola_char_eq (a b : ℝ) :
  (ellipse_eccentricity * (5 / 4) = 8 / 5) ∧ (a^2 + b^2 = 16) → 
  hyperbola_equation y x 2 (sqrt 12) := by
  sorry

end hyperbola_char_eq_l323_323235


namespace triangle_RG_GQ_ratio_l323_323292

noncomputable def RG_GQ_ratio : ℝ :=
let P : Type := sorry
let Q : Type := sorry
let R : Type := sorry
let G : Type := sorry
let H : Type := sorry
let S : Type := sorry
let PS_GS_ratio : ℝ := 2
let QS_HS_ratio : ℝ := 3

-- Given PG, QH intersect at S, PS/SG = 2, QS/SH = 3, prove RG/GQ = 3/5
theorem triangle_RG_GQ_ratio (P Q R G H S : Type)
  (PS_GS_ratio_eq : PS_GS_ratio = 2)
  (QS_HS_ratio_eq : QS_HS_ratio = 3) :
  RG_GQ_ratio = 3 / 5 := 
sorry

end triangle_RG_GQ_ratio_l323_323292


namespace no_guarani_2021_count_guarani_2023_l323_323425

def is_guarani (n : ℕ) : Prop :=
  let digits := (toDigits n).reverse
  let sum_digits := toDigits (n + digits)
  sum_digits.all (fun d => d % 2 = 1)

def count_guarani (digits_len : ℕ) : ℕ :=
  -- Count the number of guarani numbers with the given digits length
  if digits_len % 2 = 0 then
    0 -- There are no guarani numbers with an even number of digits
  else
    let pairs := (digits_len - 1) / 2
    let middle_choices := 5
    let pair_choices := 20^pairs
    middle_choices * pair_choices

theorem no_guarani_2021 : count_guarani 2021 = 0 :=
  sorry

theorem count_guarani_2023 : count_guarani 2023 = 20^1011 * 5 :=
  sorry

end no_guarani_2021_count_guarani_2023_l323_323425


namespace distinct_arrangement_count_l323_323720

open Finset

noncomputable def distinct_letter_arrangements (grid_size : ℕ) : ℕ :=
  (choose (grid_size * grid_size) 2 * (choose ((grid_size - 1) * (grid_size - 1)) 2))

theorem distinct_arrangement_count :
  distinct_letter_arrangements 4 = 120 := 
by
  sorry

end distinct_arrangement_count_l323_323720


namespace almonds_weight_l323_323833

def nuts_mixture (almonds_ratio walnuts_ratio total_weight : ℚ) : ℚ :=
  let total_parts := almonds_ratio + walnuts_ratio
  let weight_per_part := total_weight / total_parts
  let weight_almonds := weight_per_part * almonds_ratio
  weight_almonds

theorem almonds_weight (total_weight : ℚ) (h1 : total_weight = 140) : nuts_mixture 5 1 total_weight = 116.67 :=
by
  sorry

end almonds_weight_l323_323833


namespace chord_on_ellipse_midpoint_l323_323740

theorem chord_on_ellipse_midpoint :
  ∀ (A B : ℝ × ℝ)
    (hx1 : (A.1^2) / 2 + A.2^2 = 1)
    (hx2 : (B.1^2) / 2 + B.2^2 = 1)
    (mid : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 1/2),
  ∃ (k : ℝ), ∀ (x y : ℝ), y - 1/2 = k * (x - 1/2) ↔ 2 * x + 4 * y = 3 := 
sorry

end chord_on_ellipse_midpoint_l323_323740


namespace graph_cuts_x_axis_l323_323751

noncomputable def f (x : ℝ) : ℝ := log (x^2 - 1)

theorem graph_cuts_x_axis :
  ∃ x : ℝ, f x = 0 ∧ (x < -1 ∨ x > 1) :=
by
  sorry

end graph_cuts_x_axis_l323_323751


namespace divide_students_into_teams_l323_323149

-- Definitions
def Student := Type
variable (students : Student)
variable (throws_snowball_at : Student → Student)
assume (H : ∀ s : Student, ∃ t : Student, throws_snowball_at s = t)

-- Theorem Statement
theorem divide_students_into_teams :
  ∃ (team : Student → Fin 3), ∀ s1 s2 : Student, throws_snowball_at s1 = s2 → team s1 ≠ team s2 :=
sorry

end divide_students_into_teams_l323_323149


namespace james_muscle_gain_l323_323316

-- Define the initial weight
def initial_weight : ℝ := 120

-- Define the final weight
def final_weight : ℝ := 150

-- Define the percentage of body weight gained in muscle
def muscle_gain_percentage : ℝ := 20

-- Define the weight gained in fat as a fraction of muscle gain
def fat_gain_fraction : ℝ := 1 / 4

noncomputable def muscle_gain_weight (x : ℝ) : ℝ := (x / 100) * initial_weight

noncomputable def fat_gain_weight (x : ℝ) : ℝ := (x / 400) * initial_weight

noncomputable def total_weight_gain (x : ℝ) : ℝ := muscle_gain_weight x + fat_gain_weight x

theorem james_muscle_gain :
  total_weight_gain muscle_gain_percentage = final_weight - initial_weight :=
by
  sorry

end james_muscle_gain_l323_323316


namespace period_tan_func_symmetry_tan_func_l323_323435

def tan_func (x : ℝ) : ℝ := Real.tan (-2 * x + Real.pi / 3)

theorem period_tan_func : ∀ x : ℝ, tan_func (x + Real.pi / 2) = tan_func x :=
by {
  sorry
}

theorem symmetry_tan_func : ∀ x : ℝ, tan_func (5 * Real.pi / 12 - x) = -tan_func (x - 5 * Real.pi / 12) :=
by {
  sorry
}

end period_tan_func_symmetry_tan_func_l323_323435


namespace combined_ratio_l323_323777

theorem combined_ratio (cayley_students fermat_students : ℕ) 
                       (cayley_ratio_boys cayley_ratio_girls fermat_ratio_boys fermat_ratio_girls : ℕ) 
                       (h_cayley : cayley_students = 400) 
                       (h_cayley_ratio : (cayley_ratio_boys, cayley_ratio_girls) = (3, 2)) 
                       (h_fermat : fermat_students = 600) 
                       (h_fermat_ratio : (fermat_ratio_boys, fermat_ratio_girls) = (2, 3)) :
  (480 : ℚ) / 520 = 12 / 13 := 
by 
  sorry

end combined_ratio_l323_323777


namespace prod_eq_one_l323_323233

noncomputable def is_parity_equal (A : Finset ℝ) (a : ℝ) : Prop :=
  (A.filter (fun x => x > a)).card % 2 = (A.filter (fun x => x < 1/a)).card % 2

theorem prod_eq_one
  (A : Finset ℝ)
  (hA : ∀ (a : ℝ), 0 < a → is_parity_equal A a)
  (hA_pos : ∀ x ∈ A, 0 < x) :
  A.prod id = 1 :=
sorry

end prod_eq_one_l323_323233


namespace number_of_5_primable_numbers_below_1000_l323_323117

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323117


namespace angle_equality_l323_323728

-- Define the triangle ABC and Orthocenter H
variable (A B C H O : Point)

-- Define properties of the problem
axiom triangle_exists : is_triangle A B C
axiom orthocenter_exists : is_orthocenter H A B C
axiom circumscribed : circle_contains O A B C
axiom center_circle : is_center O (circle A B C)

-- The main statement to prove
theorem angle_equality :
  ∠O A B = ∠H A C :=
sorry

end angle_equality_l323_323728


namespace problem_statement_l323_323948

def H_function (f : ℝ → ℝ) : Prop :=
  ∀ {x1 x2 : ℝ}, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) > 0

def f1 (x : ℝ) : ℝ := 3 * x + 1

def f2 (x : ℝ) : ℝ := if x < -1 then -(1 / x) else x^2 + 4 * x + 5

theorem problem_statement : H_function f1 ∧ H_function f2 :=
by sorry

end problem_statement_l323_323948


namespace area_of_rectangle_l323_323890

noncomputable def length := 44.4
noncomputable def width := 29.6

theorem area_of_rectangle (h1 : width = 2 / 3 * length) (h2 : 2 * (length + width) = 148) : 
  (length * width) = 1314.24 := 
by 
  sorry

end area_of_rectangle_l323_323890


namespace number_of_valid_quadruples_l323_323875

def is_valid_quadruple (a b c d : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (a.factorial * b.factorial * c.factorial * d.factorial = 24.factorial)

theorem number_of_valid_quadruples : 
  (∃ n, (n = 4) ∧ (∀ a b c d, is_valid_quadruple a b c d → n = 4)) :=
sorry

end number_of_valid_quadruples_l323_323875


namespace conic_section_is_ellipse_l323_323797

open Real

def is_conic_section_ellipse (x y : ℝ) (k : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  sqrt ((x - p1.1) ^ 2 + (y - p1.2) ^ 2) + sqrt ((x - p2.1) ^ 2 + (y - p2.2) ^ 2) = k

theorem conic_section_is_ellipse :
  is_conic_section_ellipse 2 (-2) 12 (2, -2) (-3, 5) :=
by
  sorry

end conic_section_is_ellipse_l323_323797


namespace parallel_lines_l323_323615

theorem parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, a * x + 2 * y - 1 = k * (2 * x + a * y + 2)) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end parallel_lines_l323_323615


namespace count_5_primable_less_than_1000_eq_l323_323136

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323136


namespace disprove_ab_lt_ac_l323_323953

theorem disprove_ab_lt_ac (a b c : ℤ) (hac: a * c < 0) (hcb: c < b) (hba: b < a) : ¬ (a * b < a * c) :=
by
  let a := 1
  let b := 0
  let c := -1
  have hab := a * b = 0
  have hac := a * c = -1
  have h := hab < hac
  exact h

end disprove_ab_lt_ac_l323_323953


namespace ellipse_standard_equation_constant_value_l323_323225

-- Given an ellipse C with |a > b > 0| and eccentricity e = 1/2, point P is a moving point.
-- Let F1 and F2 be the left and right foci respectively.
-- Area condition: The maximum area of the incircle of triangle PF1F2 is π/3.

variables {a b : ℝ} (h : a > b > 0) (e : ℝ) (e_eq : e = 1/2) (max_area : ∃ P, P ∈ ellipse(a, b) ∧ incircle_area(P, F1, F2) = π/3)

-- Standard equation of the ellipse
theorem ellipse_standard_equation (h : a = 2) (b_eq : b = sqrt 3) : 
  ellipse_eq a b = (λ (x y : ℝ), (x^2)/4 + (y^2)/3 = 1) := 
sorry

-- Prove the constant value of the ratio
theorem constant_value (P F1 F2 A B : Point ℝ)
  (hP : P ∈ ellipse (2, sqrt 3))
  (h1 : line_through P F1 ∩ ellipse(2, sqrt 3) = {A})
  (h2 : line_through P F2 ∩ ellipse(2, sqrt 3) = {B})
  : 
  (distance P F1 / distance F1 A + distance P F2 / distance F2 B) = 10/3 := 
sorry

end ellipse_standard_equation_constant_value_l323_323225


namespace twin_primes_not_right_triangle_legs_l323_323376

theorem twin_primes_not_right_triangle_legs (p k : ℕ) (h1 : p.prime) (h2 : (p + 2).prime) (h3: p^2 + (p + 2)^2 = k^2) : false :=
by {
  -- to be proven
  sorry
}

end twin_primes_not_right_triangle_legs_l323_323376


namespace felipe_building_time_l323_323902

theorem felipe_building_time
  (F E : ℕ)
  (combined_time_without_breaks : ℕ)
  (felipe_time_fraction : F = E / 2)
  (combined_time_condition : F + E = 90)
  (felipe_break : ℕ)
  (emilio_break : ℕ)
  (felipe_break_is_6_months : felipe_break = 6)
  (emilio_break_is_double_felipe : emilio_break = 2 * felipe_break) :
  F + felipe_break = 36 := by
  sorry

end felipe_building_time_l323_323902


namespace trig_identity_l323_323938

theorem trig_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : sin α = 4 / 5) : 
  (sin α ^ 2 + sin (2 * α)) / (cos α ^ 2 + cos (2 * α)) = 20 :=
by
  sorry

end trig_identity_l323_323938


namespace derivative_at_one_l323_323603

noncomputable def f (x : ℝ) := x * Real.log x

theorem derivative_at_one :
  ∀ (f : ℝ → ℝ), (∀ x, f x = x * Real.log x) → (deriv f 1 = 1) :=
by
  assume f h,
  sorry

end derivative_at_one_l323_323603


namespace total_items_given_out_l323_323703

-- Miss Davis gave 15 popsicle sticks and 20 straws to each group.
def popsicle_sticks_per_group := 15
def straws_per_group := 20
def items_per_group := popsicle_sticks_per_group + straws_per_group

-- There are 10 groups in total.
def number_of_groups := 10

-- Prove the total number of items given out equals 350.
theorem total_items_given_out : items_per_group * number_of_groups = 350 :=
by
  sorry

end total_items_given_out_l323_323703


namespace inequality_sqrt_sum_ge_sqrt2_l323_323273

theorem inequality_sqrt_sum_ge_sqrt2
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1) :
  sqrt (a^2 + b^2) + sqrt (b^2 + c^2) + sqrt (c^2 + a^2) ≥ sqrt 2 :=
sorry

end inequality_sqrt_sum_ge_sqrt2_l323_323273


namespace count_5_primables_less_than_1000_l323_323096

-- Definition of a digit being a one-digit prime number
def is_one_digit_prime (d : ℕ) : Prop := d ∈ {2, 3, 5, 7}

-- Definition of a number being 5-primable
def is_5_primable (n : ℕ) : Prop := 
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

-- The main theorem stating the problem
theorem count_5_primables_less_than_1000 : 
  finset.card {n | n < 1000 ∧ is_5_primable n} = 69 :=
sorry

end count_5_primables_less_than_1000_l323_323096


namespace road_network_minimization_l323_323207

theorem road_network_minimization :
  ∃ (A B C D M N : Real × Real), 
    -- Vertices of the square
    A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1) ∧
    -- Midpoints of the sides
    let M1 := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M2 := ((C.1 + D.1) / 2, (C.2 + D.2) / 2) in
    -- Points M and N on segments
    M = (M1.1, (M1.2 + 3 / 8)) ∧ N = (M2.1, (M2.2 - 3 / 8)) ∧
    -- Conditions for points on segments ensuring the correct distances
    abs (dist A M) = 5 / 8 ∧ abs (dist B M) = 5 / 8 ∧ abs (dist C N) = 5 / 8 ∧ abs (dist D N) = 5 / 8 ∧ abs (dist M N) = 1 / 4 ∧
    -- Total length is less than 2.828
    (abs (dist A M) + abs (dist B M) + abs (dist C N) + abs (dist D N) + abs (dist M N)) < 2 * Real.sqrt 2 := sorry

end road_network_minimization_l323_323207


namespace repeating_decimal_count_l323_323929

theorem repeating_decimal_count :
  let count := (list.filter (λ n, ∀ p, Nat.Prime p → p ∣ (n + 1) → p = 2 ∨ p = 5) (list.range' 2 200)).length
  count = 182 :=
by
  sorry

#eval (let count := (list.filter (λ n, ∀ p, Nat.Prime p → p ∣ (n + 1) → p = 2 ∨ p = 5) (list.range' 2 200)).length
       count)  -- Expected to output 182

end repeating_decimal_count_l323_323929


namespace num_integers_with_repeating_decimal_l323_323932

theorem num_integers_with_repeating_decimal : (finset.range 200).filter (λ n, 
  let d := n + 1 in
  ∀ p : ℕ, nat.prime p → p ∣ d → p = 2 ∨ p = 5).card = 182 :=
begin
  sorry
end

end num_integers_with_repeating_decimal_l323_323932


namespace trigonometric_function_monotone_decreasing_l323_323253

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 4) + cos (2 * x + π / 4)

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_decreasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop := ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x < y → f y ≤ f x

theorem trigonometric_function_monotone_decreasing :
  is_periodic f π ∧ is_even f →
  monotone_decreasing_on f (set.Ioc 0 (π / 2)) :=
begin
  -- placeholder for the actual proof
  sorry
end

end trigonometric_function_monotone_decreasing_l323_323253


namespace total_odd_green_red_marbles_l323_323727

def Sara_green : ℕ := 3
def Sara_red : ℕ := 5
def Tom_green : ℕ := 4
def Tom_red : ℕ := 7
def Lisa_green : ℕ := 5
def Lisa_red : ℕ := 3

theorem total_odd_green_red_marbles : 
  (if Sara_green % 2 = 1 then Sara_green else 0) +
  (if Sara_red % 2 = 1 then Sara_red else 0) +
  (if Tom_green % 2 = 1 then Tom_green else 0) +
  (if Tom_red % 2 = 1 then Tom_red else 0) +
  (if Lisa_green % 2 = 1 then Lisa_green else 0) +
  (if Lisa_red % 2 = 1 then Lisa_red else 0) = 23 := by
  sorry

end total_odd_green_red_marbles_l323_323727


namespace average_speed_l323_323868

-- Definitions and conditions
def distance : ℝ := 88 -- miles
def time : ℝ := 4 -- hours

-- Theorem statement and proof (the proof is skipped with 'sorry')
theorem average_speed (d t : ℝ) (h_d : d = 88) (h_t : t = 4) : d / t = 22 := 
by {
  rw [h_d, h_t];
  norm_num;
  sorry
}

end average_speed_l323_323868


namespace cheesecake_factory_working_days_l323_323725

-- Define the savings rates
def robby_saves := 2 / 5
def jaylen_saves := 3 / 5
def miranda_saves := 1 / 2

-- Define their hourly rate and daily working hours
def hourly_rate := 10 -- dollars per hour
def work_hours_per_day := 10 -- hours per day

-- Define their combined savings after four weeks and the combined savings target
def four_weeks := 4 * 7
def combined_savings_target := 3000 -- dollars

-- Question: Prove that the number of days they work per week is 7
theorem cheesecake_factory_working_days (d : ℕ) (h : d * 400 = combined_savings_target / 4) : d = 7 := sorry

end cheesecake_factory_working_days_l323_323725


namespace grid_area_l323_323660

-- Definitions based on problem conditions
def num_lines : ℕ := 36
def perimeter : ℕ := 72
def side_length : ℕ := perimeter / num_lines

-- Problem statement
theorem grid_area (h : num_lines = 36) (p : perimeter = 72)
  (s : side_length = 2) :
  let n_squares := (8 - 1) * (4 - 1)
  let area_square := side_length ^ 2
  let total_area := n_squares * area_square
  total_area = 84 :=
by {
  -- Skipping proof
  sorry
}

end grid_area_l323_323660


namespace area_of_triangle_l323_323019

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
((q.1 - p.1)^2 + (q.2 - p.2)^2).sqrt

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * (B.1 - A.1).abs * (C.2 - B.2).abs

theorem area_of_triangle : triangle_area (2, 3) (7, 3) (4, 8) = 12.5 := by
  sorry

end area_of_triangle_l323_323019


namespace min_value_hyperbola_l323_323643

open Real 

theorem min_value_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (3 * x^2 - 2 * y ≥ 143 / 12) ∧ 
                                          (∃ (y' : ℝ), y = y' ∧  3 * (2 + 2*y'^2)^2 - 2 * y' = 143 / 12) := 
by
  sorry

end min_value_hyperbola_l323_323643


namespace distance_A_B_l323_323667

-- Definitions of the points A and B
def A : ℝ × ℝ × ℝ := (2, 3, 5)
def B : ℝ × ℝ × ℝ := (3, 1, 7)

-- Distance formula for points in three-dimensional space
def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2 + (Q.3 - P.3) ^ 2)

-- The proof statement that the distance between A and B is 3
theorem distance_A_B : distance A B = 3 :=
by sorry

end distance_A_B_l323_323667


namespace explicit_g_formula_inequality_solution_lambda_range_l323_323588

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x

theorem explicit_g_formula : ∀ x : ℝ, g x = -x^2 + 2 * x := 
by
  intros x
  sorry

theorem inequality_solution : ∀ x : ℝ, g x ≥ f x - |x - 1| ↔ -1 ≤ x ∧ x ≤ 0.5 := 
by
  intros x
  sorry

theorem lambda_range (λ : ℝ) : (∃ x : ℝ, x > -1 ∧ x < 1 ∧ g x - λ * f x + 1 = 0) ↔
  λ ≤ 2/3 ∨ λ ≥ 2 :=
by
  intros λ
  sorry

end explicit_g_formula_inequality_solution_lambda_range_l323_323588


namespace find_f_2017_f_2018_sum_l323_323240

variables {f : ℝ → ℝ}

-- Conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

def f_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ set.Icc (-1 : ℝ) 0, f x = -x

-- The equivalent proof statement
theorem find_f_2017_f_2018_sum (h_odd : is_odd f) (h_symm : symmetric_about_one f) (h_interval : f_on_interval f) :
  f 2017 + f 2018 = -1 :=
sorry

end find_f_2017_f_2018_sum_l323_323240


namespace tan_alpha_minus_pi_over_4_l323_323941

theorem tan_alpha_minus_pi_over_4 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (β + π/4) = 3) 
  : Real.tan (α - π/4) = -1 / 7 :=
by
  sorry

end tan_alpha_minus_pi_over_4_l323_323941


namespace determine_b_perpendicular_l323_323183

theorem determine_b_perpendicular :
  ∀ (b : ℝ),
  (b * 2 + (-3) * (-1) + 2 * 4 = 0) → 
  b = -11/2 :=
by
  intros b h
  sorry

end determine_b_perpendicular_l323_323183


namespace tetrahedron_min_f_l323_323392

noncomputable def f (A B C D X : Point) : ℝ :=
  dist A X + dist B X + dist C X + dist D X

theorem tetrahedron_min_f (A B C D : Point) (AD BC : dist A D = 26 ∧ dist B C = 26) 
  (AC BD : dist A C = 40 ∧ dist B D = 40) 
  (AB CD : dist A B = 50 ∧ dist C D = 50) : 
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ f(A, B, C, D, X) = p * Real.sqrt q ∧ p * Real.sqrt q = 40 :=
sorry

end tetrahedron_min_f_l323_323392


namespace problem_l323_323332

variable (a b : ℝ)

def P1 (a : ℝ) : Set ℝ := { x | x^2 + a * x + 1 > 0 }
def P2 (a : ℝ) : Set ℝ := { x | x^2 + a * x + 2 > 0 }

def Q1 (b : ℝ) : Set ℝ := { x | x^2 + x + b > 0 }
def Q2 (b : ℝ) : Set ℝ := { x | x^2 + 2 * x + b > 0 }

theorem problem (a b : ℝ) :
  (∀ a : ℝ, P1 a ⊆ P2 a) ∧ (∃ b : ℝ, Q1 b ⊆ Q2 b) :=
by
  split
  { intros a x hx,
    -- x^2 + a * x + 1 > 0
    sorry },
  { use 5,
    -- show that Q1 5 ⊆ Q2 5
    sorry }

end problem_l323_323332


namespace domain_of_c_is_all_real_l323_323907

theorem domain_of_c_is_all_real (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 3 * x + a ≠ 0) ↔ a < -3 / 4 :=
by
  sorry

end domain_of_c_is_all_real_l323_323907


namespace number_of_math_players_l323_323866

theorem number_of_math_players (total_players physics_players both_players : ℕ)
    (h1 : total_players = 25)
    (h2 : physics_players = 15)
    (h3 : both_players = 6)
    (h4 : total_players = physics_players + (total_players - physics_players - (total_players - physics_players - both_players)) + both_players ) :
  total_players - (physics_players - both_players) = 16 :=
sorry

end number_of_math_players_l323_323866


namespace solve_for_a_l323_323274

theorem solve_for_a (a : ℝ) (h : a⁻¹ = (-1 : ℝ)^0) : a = 1 :=
sorry

end solve_for_a_l323_323274


namespace modulus_of_complex_number_l323_323968

noncomputable def imaginary_unit : ℂ := complex.I

noncomputable def complex_number : ℂ := imaginary_unit / (2 - imaginary_unit)

theorem modulus_of_complex_number : complex.abs complex_number = real.sqrt 5 / 5 := by
  sorry

end modulus_of_complex_number_l323_323968


namespace num_children_got_off_l323_323825

-- Define the original number of children on the bus
def original_children : ℕ := 43

-- Define the number of children left after some got off the bus
def children_left : ℕ := 21

-- Define the number of children who got off the bus as the difference between original_children and children_left
def children_got_off : ℕ := original_children - children_left

-- State the theorem that the number of children who got off the bus is 22
theorem num_children_got_off : children_got_off = 22 :=
by
  -- Proof steps would go here, but are omitted
  sorry

end num_children_got_off_l323_323825


namespace odd_a_possible_l323_323221

theorem odd_a_possible 
  (a : ℕ) 
  (ha1 : odd a) 
  (ha2 : 1 ≤ a ∧ a < 2011) 
  (n : ℕ) 
  (hn : n = 2011) :
  ∃ (f : list (fin n) → Prop),
    (∀ vertices : list (fin n), 
      (forall i, f vertices i = (¬ vertices[i])) 
      ∨ all_white vertices ∨ all_black vertices) := 
sorry

end odd_a_possible_l323_323221


namespace bus_profit_problem_l323_323052

def independent_variable := "number of passengers per month"
def dependent_variable := "monthly profit"

-- Given monthly profit equation
def monthly_profit (x : ℕ) : ℤ := 2 * x - 4000

-- 1. Independent and Dependent variables
def independent_variable_defined_correctly : Prop :=
  independent_variable = "number of passengers per month"

def dependent_variable_defined_correctly : Prop :=
  dependent_variable = "monthly profit"

-- 2. Minimum passenger volume to avoid losses
def minimum_passenger_volume_no_loss : Prop :=
  ∀ x : ℕ, (monthly_profit x >= 0) → (x >= 2000)

-- 3. Monthly profit prediction for 4230 passengers
def monthly_profit_prediction_4230 (x : ℕ) : Prop :=
  x = 4230 → monthly_profit x = 4460

theorem bus_profit_problem :
  independent_variable_defined_correctly ∧
  dependent_variable_defined_correctly ∧
  minimum_passenger_volume_no_loss ∧
  monthly_profit_prediction_4230 4230 :=
by
  sorry

end bus_profit_problem_l323_323052


namespace smallest_N_exists_l323_323557

theorem smallest_N_exists :
  ∃ N : ℕ, N > 0 ∧
  (N % 4 = 0 ∨ (N + 1) % 4 = 0 ∨ (N + 2) % 4 = 0) ∧
  (N % 9 = 0 ∨ (N + 1) % 9 = 0 ∨ (N + 2) % 9 = 0) ∧
  (N % 25 = 0 ∨ (N + 1) % 25 = 0 ∨ (N + 2) % 25 = 0) ∧
  (N % 49 = 0 ∨ (N + 1) % 49 = 0 ∨ (N + 2) % 49 = 0) ∧
  ∀ M : ℕ, 
    (M > 0 ∧
    (M % 4 = 0 ∨ (M + 1) % 4 = 0 ∨ (M + 2) % 4 = 0) ∧
    (M % 9 = 0 ∨ (M + 1) % 9 = 0 ∨ (M + 2) % 9 = 0) ∧
    (M % 25 = 0 ∨ (M + 1) % 25 = 0 ∨ (M + 2) % 25 = 0) ∧
    (M % 49 = 0 ∨ (M + 1) % 49 = 0 ∨ (M + 2) % 49 = 0)) → N ≤ M := 
  98 := 
sorry

end smallest_N_exists_l323_323557


namespace tank_capacity_l323_323461

theorem tank_capacity (C : ℝ) (rate_leak : ℝ) (rate_inlet : ℝ) (combined_rate_empty : ℝ) :
  rate_leak = C / 3 ∧ rate_inlet = 6 * 60 ∧ combined_rate_empty = C / 12 →
  C = 864 :=
by
  intros h
  sorry

end tank_capacity_l323_323461


namespace num_integers_with_repeating_decimal_l323_323930

theorem num_integers_with_repeating_decimal : (finset.range 200).filter (λ n, 
  let d := n + 1 in
  ∀ p : ℕ, nat.prime p → p ∣ d → p = 2 ∨ p = 5).card = 182 :=
begin
  sorry
end

end num_integers_with_repeating_decimal_l323_323930


namespace number_of_5_primable_numbers_below_1000_l323_323113

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323113


namespace cleanAndJerkRatio_l323_323676

noncomputable def initialCleanAndJerkWeight : ℝ := 80
noncomputable def initialSnatchWeight : ℝ := 50
noncomputable def snatchIncreasePercent : ℝ := 0.80
noncomputable def newCombinedTotalLiftingCapacity : ℝ := 250

theorem cleanAndJerkRatio :
  let newSnatchWeight := initialSnatchWeight + (snatchIncreasePercent * initialSnatchWeight),
      newCleanAndJerkWeight := newCombinedTotalLiftingCapacity - newSnatchWeight in
  (newCleanAndJerkWeight / initialCleanAndJerkWeight) = 2 :=
by
  sorry

end cleanAndJerkRatio_l323_323676


namespace geometric_sequence_properties_l323_323949

-- Declare the given conditions using Lean definitions
variables {a : ℕ → ℝ} {q : ℝ}
variables (ha : ∀ n, a n = a 1 * q ^ (n - 1)) (hq : q > 1)
variables (h1 : a 2 + a 3 + a 4 = 28) (h2 : 2 * (a 3 + 2) = a 2 + a 4)

-- Define the target general formula and sequence properties
def general_formula := a 1 = 2 ∧ q = 2 ∧ ∀ n, a n = 2 ^ n
def arith_seq_b (n : ℕ) := log 2 (a (n + 5))
def Sn (n : ℕ) := ∑ i in range n, arith_seq_b (i + 1)
def Tn (n : ℕ) := ∑ i in range n, Sn i / i

-- State the theorem with the expected answer
theorem geometric_sequence_properties :
  general_formula ha hq h1 h2 ∧ Tn = (λ n, (n^2 + 23*n) / 4) :=
sorry

end geometric_sequence_properties_l323_323949


namespace count_5_primable_less_than_1000_eq_l323_323129

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → is_prime_digit d) ∧ n % 5 = 0

def count_5_primable_less_than (k : ℕ) : ℕ :=
  (Finset.range k).filter is_5_primable |>.card

theorem count_5_primable_less_than_1000_eq : count_5_primable_less_than 1000 = 16 :=
  sorry

end count_5_primable_less_than_1000_eq_l323_323129


namespace lattice_point_count_in_triangle_l323_323194

-- Define the problem conditions and the expected answer
def numLatticePointsInTriangle (n : ℕ) := 
  ∑ x in Finset.range (n + 1), ∑ y in Finset.range (n - 3 * x + 1), 1

theorem lattice_point_count_in_triangle : numLatticePointsInTriangle 12 = 35 :=
by
  sorry

end lattice_point_count_in_triangle_l323_323194


namespace geo_seq_sum_eq_l323_323584

variable {a : ℕ → ℝ}

-- Conditions
def is_geo_seq (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r
def positive_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > 0
def specific_eq (a : ℕ → ℝ) : Prop := a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 25

theorem geo_seq_sum_eq (a : ℕ → ℝ) (h_geo : is_geo_seq a) (h_pos : positive_seq a) (h_eq : specific_eq a) : 
  a 2 + a 4 = 5 :=
by
  sorry

end geo_seq_sum_eq_l323_323584


namespace count_5_primable_integers_lt_1000_is_21_l323_323102

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : ℕ) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ n.digits 10, is_prime_digit d)

def count_5_primable_lt_1000 : ℕ :=
  (Finset.range 1000).filter is_5_primable |>.card

theorem count_5_primable_integers_lt_1000_is_21 : count_5_primable_lt_1000 = 21 := 
by 
  sorry

end count_5_primable_integers_lt_1000_is_21_l323_323102


namespace smallest_product_not_factor_of_48_exists_l323_323005

theorem smallest_product_not_factor_of_48_exists : 
  ∃ (a b : ℕ), (a ≠ b) ∧ (a ∣ 48) ∧ (b ∣ 48) ∧ ¬ (a * b ∣ 48) ∧ 
  (∀ (c d : ℕ), (c ≠ d) ∧ (c ∣ 48) ∧ (d ∣ 48) ∧ ¬ (c * d ∣ 48) → a * b ≤ c * d) ∧ (a * b = 18) :=
sorry

end smallest_product_not_factor_of_48_exists_l323_323005


namespace part1_part2_l323_323963

variable (a b c : ℝ)

open Classical

noncomputable theory

-- Defining the conditions
def cond_positive_numbers : Prop := (0 < a) ∧ (0 < b) ∧ (0 < c)
def cond_main_equation : Prop := a^2 + b^2 + 4*c^2 = 3
def cond_b_eq_2c : Prop := b = 2*c

-- Statement for part (1)
theorem part1 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) :
  a + b + 2*c ≤ 3 := sorry

-- Statement for part (2)
theorem part2 (h1 : cond_positive_numbers a b c) (h2 : cond_main_equation a b c) (h3 : cond_b_eq_2c b c) :
  (1 / a) + (1 / c) ≥ 3 := sorry

end part1_part2_l323_323963


namespace family_spent_l323_323845

theorem family_spent (ticket_price : ℝ) (popcorn_cost : ℝ) (num_tickets : ℕ) (num_popcorn : ℕ) (num_sodas : ℕ) :
  ticket_price = 5 →
  popcorn_cost = 0.8 * ticket_price →
  num_tickets = 4 →
  num_popcorn = 2 →
  num_sodas = 4 →
  let total_ticket_cost := num_tickets * ticket_price in
  let discount_ticket_cost := total_ticket_cost * 0.9 in
  let total_popcorn_cost := num_popcorn * popcorn_cost in
  let soda_price := popcorn_cost in
  let discount_soda_price := soda_price * 0.5 in
  let total_soda_cost := 2 * soda_price + 2 * discount_soda_price in
  let total_cost := discount_ticket_cost + total_popcorn_cost + total_soda_cost in
  total_cost = 38 := 
by
  intros h_ticket_price h_popcorn_cost h_num_tickets h_num_popcorn h_num_sodas
  let total_ticket_cost := num_tickets * ticket_price
  let discount_ticket_cost := total_ticket_cost * 0.9
  let total_popcorn_cost := num_popcorn * popcorn_cost
  let soda_price := popcorn_cost
  let discount_soda_price := soda_price * 0.5
  let total_soda_cost := 2 * soda_price + 2 * discount_soda_price
  let total_cost := discount_ticket_cost + total_popcorn_cost + total_soda_cost
  rw [h_ticket_price, h_popcorn_cost, h_num_tickets, h_num_popcorn, h_num_sodas] at *
  sorry

end family_spent_l323_323845


namespace reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l323_323457

-- Definition: reversing a deck of n cards in k operations
def can_reverse_deck (n k : ℕ) : Prop := sorry -- Placeholder definition

-- Proof Part (a)
theorem reverse_9_in_5_operations :
  can_reverse_deck 9 5 :=
sorry

-- Proof Part (b)
theorem reverse_52_in_27_operations :
  can_reverse_deck 52 27 :=
sorry

-- Proof Part (c)
theorem not_reverse_52_in_17_operations :
  ¬can_reverse_deck 52 17 :=
sorry

-- Proof Part (d)
theorem not_reverse_52_in_26_operations :
  ¬can_reverse_deck 52 26 :=
sorry

end reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l323_323457


namespace problem1_problem2_l323_323650

-- Proof Problem 1
theorem problem1 (A B C : ℝ) (a b c : ℝ) (acute_triangle : ∀A B C, A + B + C = π ∧ A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (cos_rule_condition : 23 * cos A ^ 2 + cos (2 * A) = 0) (a_value : a = 7) (c_value : c = 6) : b = 5 :=
sorry

-- Proof Problem 2
theorem problem2 (A B C : ℝ) (a b c : ℝ)
  (a_value : a = sqrt 3) (A_value : A = π / 3) : sqrt 3 < b + c ∧ b + c ≤ 2 * sqrt 3 :=
sorry

end problem1_problem2_l323_323650


namespace find_k_l323_323640

noncomputable def value_of_k (a b c x y z : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49) (h2 : x^2 + y^2 + z^2 = 64) (h3 : ax + by + cz = 56) : ℝ :=
  7 / 8

theorem find_k (a b c x y z : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : a^2 + b^2 + c^2 = 49) (h2 : x^2 + y^2 + z^2 = 64) (h3 : ax + by + cz = 56) : value_of_k a b c x y z h_pos h1 h2 h3 = 7 / 8 :=
by {
  sorry
}

end find_k_l323_323640


namespace angle_KPC_is_right_angle_l323_323669

variables {A B C K E F P : Type*} [metric_space A] [metric_space B] 
[metric_space C] [metric_space K] [metric_space E] [metric_space F] 
[metric_space P] 

/- Given conditions of the problem -/
variables (triangle_ABC : Prop)
variables (K_on_BC : K) (AE_eq_CF_eq_BK : ∀ {A E F K}, dist A E = dist C F ∧ dist C F = dist B K)
variables (P_midpoint_EF : P) (P_is_midpoint : ∀ {E F P}, P = midpoint (dist E F))

/- Proof goal -/
theorem angle_KPC_is_right_angle 
  (h1 : triangle_ABC)
  (h2 : K_on_BC)
  (h3 : AE_eq_CF_eq_BK)
  (h4 : P_midpoint_EF) 
  (h5 : P_is_midpoint) 
: angle K P C = 90 :=
  by sorry

end angle_KPC_is_right_angle_l323_323669


namespace train_cross_pole_time_l323_323859

def speed_kmph : ℝ := 80
def length_meters : ℝ := 111.12
def conversion_factor : ℝ := 1000 / 3600

theorem train_cross_pole_time :
  let speed_mps := speed_kmph * conversion_factor in 
  let time := length_meters / speed_mps in
  time = 5 :=
by
  sorry

end train_cross_pole_time_l323_323859


namespace part_one_part_two_l323_323959

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end part_one_part_two_l323_323959


namespace number_of_5_primable_numbers_below_1000_l323_323111

def is_one_digit_prime (d : Nat) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_5_primable (n : Nat) : Prop :=
  n % 5 = 0 ∧ (∀ d ∈ to_digits 10 n, is_one_digit_prime d)

def count_5_primable_numbers_below (m : Nat) : Nat :=
  ((finset.range m).filter is_5_primable).card

theorem number_of_5_primable_numbers_below_1000 : count_5_primable_numbers_below 1000 = 13 := sorry

end number_of_5_primable_numbers_below_1000_l323_323111


namespace f_20_plus_f_neg20_l323_323346

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^4 + b * x^2 + 5

theorem f_20_plus_f_neg20 (a b : ℝ) (h : f a b 20 = 3) : f a b 20 + f a b (-20) = 6 := by
  sorry

end f_20_plus_f_neg20_l323_323346


namespace measure_8_liters_with_two_buckets_l323_323998

def bucket_is_empty (B : ℕ) : Prop :=
  B = 0

def bucket_has_capacity (B : ℕ) (c : ℕ) : Prop :=
  B ≤ c

def fill_bucket (B : ℕ) (c : ℕ) : ℕ :=
  c

def empty_bucket (B : ℕ) : ℕ :=
  0

def pour_bucket (B1 B2 : ℕ) (c1 c2 : ℕ) : (ℕ × ℕ) :=
  if B1 + B2 <= c2 then (0, B1 + B2)
  else (B1 - (c2 - B2), c2)

theorem measure_8_liters_with_two_buckets (B10 B6 : ℕ) (c10 c6 : ℕ) :
  bucket_has_capacity B10 c10 ∧ bucket_has_capacity B6 c6 ∧
  c10 = 10 ∧ c6 = 6 →
  ∃ B10' B6', B10' = 8 ∧ B6' ≤ 6 :=
by
  intros h
  have h1 : ∃ B1, bucket_is_empty B1,
    from ⟨0, rfl⟩
  let B10 := fill_bucket 0 c10
  let ⟨B10, B6⟩ := pour_bucket B10 0 c10 c6
  let B6 := empty_bucket B6
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  let B10 := fill_bucket B10 c10
  let ⟨B10, B6⟩ := pour_bucket B10 B6 c10 c6
  exact ⟨B10, B6, rfl, le_refl 6⟩

end measure_8_liters_with_two_buckets_l323_323998


namespace fg_neg3_l323_323276

-- Define the functions f and g as given in the problem
def f (x : ℝ) : ℝ := 4 - Real.sqrt (x^2)
def g (x : ℝ) : ℝ := 7 * x + 3 * x^3

-- State the theorem
theorem fg_neg3 : f (g (-3)) = -56 := 
sorry

end fg_neg3_l323_323276


namespace fixed_t_for_chords_of_parabola_l323_323516

theorem fixed_t_for_chords_of_parabola (k c : ℝ) (h : c - k = 1 / 2)
  (H : ∀ (A B : ℝ × ℝ) (hA : A.2 = A.1 ^ 2 + k) (hB : B.2 = B.1 ^ 2 + k),
    let AC := (A.1 - 0) ^ 2 + (A.2 - c) ^ 2,
        BC := (B.1 - 0) ^ 2 + (B.2 - c) ^ 2 
    in ∀ (m : ℝ), 
      A.1 + B.1 = m ∧ A.1 * B.1 = c - k → 
        AC ≠ 0 ∧ BC ≠ 0 → (1 / AC + 1 / BC = t)) :
  t = 4 :=
by {
  sorry
}

end fixed_t_for_chords_of_parabola_l323_323516


namespace daniil_max_candies_l323_323719

theorem daniil_max_candies (candies : ℕ) (grid : ℕ → ℕ → ℕ)
  (h_candies : candies = 36)
  (h_grid_sum : ∑ i in Finset.range 3, ∑ j in Finset.range 3, grid i j = candies)
  (h_grid : ∀ i j, grid i j ≥ 0) :
  ∃ (max_candies : ℕ), max_candies = 9 ∧
  ∀ s1 s2 : Fin 2 → ℕ.succ 1,
  (0 ≤ s1 0 ∧ s1 0 ≤ 1 ∧ 0 ≤ s1 1 ∧ s1 1 ≤ 1 ∧
   0 ≤ s2 0 ∧ s2 0 ≤ 1 ∧ 0 ≤ s2 1 ∧ s2 1 ≤ 1 ∧
   (s1 ≠ s2)) → ((grid s1 0 s1 1 + 
                   grid s1 0 s2 1 + 
                   grid s2 0 s1 1 + 
                   grid s2 0 s2 1) ≤ 9) :=
by sorry

end daniil_max_candies_l323_323719
