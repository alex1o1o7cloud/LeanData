import Mathlib

namespace trig_problem_1_trig_problem_2_l1846_184689

noncomputable def trig_expr_1 : ℝ :=
  Real.cos (-11 * Real.pi / 6) + Real.sin (12 * Real.pi / 5) * Real.tan (6 * Real.pi)

noncomputable def trig_expr_2 : ℝ :=
  Real.sin (420 * Real.pi / 180) * Real.cos (750 * Real.pi / 180) +
  Real.sin (-330 * Real.pi / 180) * Real.cos (-660 * Real.pi / 180)

theorem trig_problem_1 : trig_expr_1 = Real.sqrt 3 / 2 :=
by
  sorry

theorem trig_problem_2 : trig_expr_2 = 1 :=
by
  sorry

end trig_problem_1_trig_problem_2_l1846_184689


namespace total_spending_is_correct_l1846_184607

def total_spending : ℝ :=
  let meal_expenses_10 := 10 * 18
  let meal_expenses_5 := 5 * 25
  let total_meal_expenses := meal_expenses_10 + meal_expenses_5
  let service_charge := 50
  let total_before_discount := total_meal_expenses + service_charge
  let discount := 0.05 * total_meal_expenses
  let total_after_discount := total_before_discount - discount
  let tip := 0.10 * total_before_discount
  total_after_discount + tip

theorem total_spending_is_correct : total_spending = 375.25 :=
by
  sorry

end total_spending_is_correct_l1846_184607


namespace horner_evaluation_at_2_l1846_184673

def f (x : ℤ) : ℤ := 3 * x^5 - 2 * x^4 + 2 * x^3 - 4 * x^2 - 7

theorem horner_evaluation_at_2 : f 2 = 16 :=
by {
  sorry
}

end horner_evaluation_at_2_l1846_184673


namespace box_dimensions_l1846_184618

theorem box_dimensions (x y z : ℝ) (h1 : x * y * z = 160) 
  (h2 : y * z = 80) (h3 : x * z = 40) (h4 : x * y = 32) : 
  x = 4 ∧ y = 8 ∧ z = 10 :=
by
  -- Placeholder for the actual proof steps
  sorry

end box_dimensions_l1846_184618


namespace count_restricted_arrangements_l1846_184686

theorem count_restricted_arrangements (n : ℕ) (hn : n = 5) : 
  (n.factorial - 2 * (n - 1).factorial) = 72 := 
by 
  sorry

end count_restricted_arrangements_l1846_184686


namespace kishore_miscellaneous_expenses_l1846_184602

theorem kishore_miscellaneous_expenses :
  ∀ (rent milk groceries education petrol savings total_salary total_specified_expenses : ℝ),
  rent = 5000 →
  milk = 1500 →
  groceries = 4500 →
  education = 2500 →
  petrol = 2000 →
  savings = 2300 →
  (savings / 0.10) = total_salary →
  (rent + milk + groceries + education + petrol) = total_specified_expenses →
  (total_salary - (total_specified_expenses + savings)) = 5200 :=
by
  intros rent milk groceries education petrol savings total_salary total_specified_expenses
  sorry

end kishore_miscellaneous_expenses_l1846_184602


namespace grooming_time_l1846_184685

theorem grooming_time (time_per_dog : ℕ) (num_dogs : ℕ) (days : ℕ) (minutes_per_hour : ℕ) :
  time_per_dog = 20 →
  num_dogs = 2 →
  days = 30 →
  minutes_per_hour = 60 →
  (time_per_dog * num_dogs * days) / minutes_per_hour = 20 := 
by
  intros
  exact sorry

end grooming_time_l1846_184685


namespace nuts_in_mason_car_l1846_184634

-- Define the constants for the rates of stockpiling
def busy_squirrel_rate := 30 -- nuts per day
def sleepy_squirrel_rate := 20 -- nuts per day
def days := 40 -- number of days
def num_busy_squirrels := 2 -- number of busy squirrels
def num_sleepy_squirrels := 1 -- number of sleepy squirrels

-- Define the total number of nuts
def total_nuts_in_mason_car : ℕ :=
  (num_busy_squirrels * busy_squirrel_rate * days) +
  (num_sleepy_squirrels * sleepy_squirrel_rate * days)

theorem nuts_in_mason_car :
  total_nuts_in_mason_car = 3200 :=
sorry

end nuts_in_mason_car_l1846_184634


namespace multiplicative_magic_square_h_sum_l1846_184622

theorem multiplicative_magic_square_h_sum :
  ∃ (h_vals : List ℕ), 
  (∀ h ∈ h_vals, ∃ (e : ℕ), e > 0 ∧ 25 * e = h ∧ 
    ∃ (b c d f g : ℕ), 
    75 * b * c = d * e * f ∧ 
    d * e * f = g * h * 3 ∧ 
    g * h * 3 = c * f * 3 ∧ 
    c * f * 3 = 75 * e * g
  ) ∧ h_vals.sum = 150 :=
by { sorry }

end multiplicative_magic_square_h_sum_l1846_184622


namespace rectangular_field_area_l1846_184665

theorem rectangular_field_area
  (x : ℝ) 
  (length := 3 * x) 
  (breadth := 4 * x) 
  (perimeter := 2 * (length + breadth))
  (cost_per_meter : ℝ := 0.25) 
  (total_cost : ℝ := 87.5) 
  (paise_per_rupee : ℝ := 100)
  (perimeter_eq_cost : 14 * x * cost_per_meter * paise_per_rupee = total_cost * paise_per_rupee) :
  (length * breadth = 7500) := 
by
  -- proof omitted
  sorry

end rectangular_field_area_l1846_184665


namespace ascorbic_acid_oxygen_mass_percentage_l1846_184624

noncomputable def mass_percentage_oxygen_in_ascorbic_acid : Float := 54.49

theorem ascorbic_acid_oxygen_mass_percentage :
  let C_mass := 12.01
  let H_mass := 1.01
  let O_mass := 16.00
  let ascorbic_acid_formula := (6, 8, 6) -- (number of C, number of H, number of O)
  let total_mass := 6 * C_mass + 8 * H_mass + 6 * O_mass
  let O_mass_total := 6 * O_mass
  mass_percentage_oxygen_in_ascorbic_acid = (O_mass_total / total_mass) * 100 := by
  sorry

end ascorbic_acid_oxygen_mass_percentage_l1846_184624


namespace triangle_angles_arithmetic_progression_l1846_184645

theorem triangle_angles_arithmetic_progression (α β γ : ℝ) (a c : ℝ) :
  (α < β) ∧ (β < γ) ∧ (α + β + γ = 180) ∧
  (∃ x : ℝ, β = α + x ∧ γ = β + x) ∧
  (a = c / 2) → 
  (α = 30) ∧ (β = 60) ∧ (γ = 90) :=
by
  intros h
  sorry

end triangle_angles_arithmetic_progression_l1846_184645


namespace marble_count_l1846_184620

theorem marble_count (r g b : ℕ) (h1 : g + b = 6) (h2 : r + b = 8) (h3 : r + g = 4) : r + g + b = 9 :=
sorry

end marble_count_l1846_184620


namespace bread_loaves_l1846_184659

theorem bread_loaves (loaf_cost : ℝ) (pb_cost : ℝ) (total_money : ℝ) (leftover_money : ℝ) : ℝ :=
  let spent_money := total_money - leftover_money
  let remaining_money := spent_money - pb_cost
  remaining_money / loaf_cost

example : bread_loaves 2.25 2 14 5.25 = 3 := by
  sorry

end bread_loaves_l1846_184659


namespace natasha_destination_distance_l1846_184687

theorem natasha_destination_distance
  (over_speed : ℕ)
  (time : ℕ)
  (speed_limit : ℕ)
  (actual_speed : ℕ)
  (distance : ℕ) :
  (over_speed = 10) →
  (time = 1) →
  (speed_limit = 50) →
  (actual_speed = speed_limit + over_speed) →
  (distance = actual_speed * time) →
  (distance = 60) :=
by
  sorry

end natasha_destination_distance_l1846_184687


namespace quadratic_inequality_range_a_l1846_184657

theorem quadratic_inequality_range_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end quadratic_inequality_range_a_l1846_184657


namespace geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l1846_184669

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ} (hq : 0 < q) (hq2 : q ≠ 1)

-- ① If $a_{1}=1$ and the common ratio is $\frac{1}{2}$, then $S_{n} < 2$;
theorem geom_seq_sum_lt_two (h₁ : a 1 = 1) (hq_half : q = 1 / 2) (n : ℕ) : S n < 2 := sorry

-- ② The sequence $\{a_{n}^{2}\}$ must be a geometric sequence
theorem geom_seq_squared (h_geom : ∀ n, a (n + 1) = q * a n) : ∃ r : ℝ, ∀ n, a n ^ 2 = r ^ n := sorry

-- ④ For any positive integer $n$, $a{}_{n}^{2}+a{}_{n+2}^{2}\geqslant 2a{}_{n+1}^{2}$
theorem geom_seq_square_inequality (h_geom : ∀ n, a (n + 1) = q * a n) (n : ℕ) (hn : 0 < n) : 
  a n ^ 2 + a (n + 2) ^ 2 ≥ 2 * a (n + 1) ^ 2 := sorry

end geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l1846_184669


namespace right_angled_triangles_count_l1846_184608

theorem right_angled_triangles_count :
    ∃ n : ℕ, n = 31 ∧ ∀ (a b : ℕ), (b < 2011) ∧ (a * a = (b + 1) * (b + 1) - b * b) → n = 31 :=
by
  sorry

end right_angled_triangles_count_l1846_184608


namespace intersection_of_P_and_Q_l1846_184690
-- Import the entire math library

-- Define the conditions for sets P and Q
def P := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | (x - 1)^2 ≤ 4}

-- Define the theorem to prove that P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3}
theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x ≤ 3} :=
by
  -- Placeholder for the proof
  sorry

end intersection_of_P_and_Q_l1846_184690


namespace smallest_solution_l1846_184653

theorem smallest_solution (x : ℝ) (h : x * |x| = 3 * x - 2) : 
  x = 1 ∨ x = 2 ∨ x = (-(3 + Real.sqrt 17)) / 2 :=
by
  sorry

end smallest_solution_l1846_184653


namespace find_p_l1846_184609

theorem find_p (p : ℚ) : (∀ x : ℚ, (3 * x + 4) = 0 → (4 * x ^ 3 + p * x ^ 2 + 17 * x + 24) = 0) → p = 13 / 4 :=
by
  sorry

end find_p_l1846_184609


namespace raja_monthly_income_l1846_184692

noncomputable def monthly_income (household_percentage clothes_percentage medicines_percentage savings : ℝ) : ℝ :=
  let spending_percentage := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage := 1 - spending_percentage
  savings / savings_percentage

theorem raja_monthly_income :
  monthly_income 0.35 0.20 0.05 15000 = 37500 :=
by
  sorry

end raja_monthly_income_l1846_184692


namespace value_of_expression_l1846_184658

theorem value_of_expression (a b : ℝ) (h : a + b = 4) : a^2 + 2 * a * b + b^2 = 16 := by
  sorry

end value_of_expression_l1846_184658


namespace my_problem_l1846_184616

theorem my_problem (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := 
sorry

end my_problem_l1846_184616


namespace problem_solution_l1846_184697

theorem problem_solution (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f ((x - y) ^ 2) = f x ^ 2 - 2 * x * f y + y ^ 2) :
    ∃ n s : ℕ, 
    (n = 2) ∧ 
    (s = 3) ∧
    (n * s = 6) :=
sorry

end problem_solution_l1846_184697


namespace find_roots_l1846_184644

theorem find_roots : ∀ z : ℂ, (z^2 + 2 * z = 3 - 4 * I) → (z = 1 - I ∨ z = -3 + I) :=
by
  intro z
  intro h
  sorry

end find_roots_l1846_184644


namespace middle_integer_is_zero_l1846_184610

-- Mathematical equivalent proof problem in Lean 4

theorem middle_integer_is_zero
  (n : ℤ)
  (h : (n - 2) + n + (n + 2) = (1 / 5) * ((n - 2) * n * (n + 2))) :
  n = 0 :=
by
  sorry

end middle_integer_is_zero_l1846_184610


namespace length_AP_eq_sqrt2_l1846_184667

/-- In square ABCD with side length 2, a circle ω with center at (1, 0)
    and radius 1 is inscribed. The circle intersects CD at point M,
    and line AM intersects ω at a point P different from M.
    Prove that the length of AP is √2. -/
theorem length_AP_eq_sqrt2 :
  let A := (0, 2)
  let M := (2, 0)
  let P : ℝ × ℝ := (1, 1)
  dist A P = Real.sqrt 2 :=
by
  sorry

end length_AP_eq_sqrt2_l1846_184667


namespace gcd_is_3_l1846_184613

noncomputable def a : ℕ := 130^2 + 240^2 + 350^2
noncomputable def b : ℕ := 131^2 + 241^2 + 351^2

theorem gcd_is_3 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_is_3_l1846_184613


namespace value_of_2a_plus_b_l1846_184637

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def is_tangent_perpendicular (a b : ℝ) : Prop :=
  let f' := (fun x => (1 : ℝ) / x - a)
  let slope_perpendicular_line := - (1/3 : ℝ)
  f' 1 * slope_perpendicular_line = -1 

def point_on_function (a b : ℝ) : Prop :=
  f a 1 = b

theorem value_of_2a_plus_b (a b : ℝ) 
  (h_tangent_perpendicular : is_tangent_perpendicular a b)
  (h_point_on_function : point_on_function a b) : 
  2 * a + b = -2 := sorry

end value_of_2a_plus_b_l1846_184637


namespace possible_values_for_a_l1846_184638

def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + 4 = 0}

theorem possible_values_for_a (a : ℝ) : (B a).Nonempty ∧ B a ⊆ A ↔ a = 4 :=
sorry

end possible_values_for_a_l1846_184638


namespace gain_percent_l1846_184654

def cycle_gain_percent (cp sp : ℕ) : ℚ :=
  (sp - cp) / cp * 100

theorem gain_percent {cp sp : ℕ} (h1 : cp = 1500) (h2 : sp = 1620) : cycle_gain_percent cp sp = 8 := by
  sorry

end gain_percent_l1846_184654


namespace sally_needs_8_napkins_l1846_184666

theorem sally_needs_8_napkins :
  let tablecloth_length := 102
  let tablecloth_width := 54
  let napkin_length := 6
  let napkin_width := 7
  let total_material_needed := 5844
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let material_needed_for_napkins := total_material_needed - tablecloth_area
  let number_of_napkins := material_needed_for_napkins / napkin_area
  number_of_napkins = 8 :=
by
  sorry

end sally_needs_8_napkins_l1846_184666


namespace decks_left_is_3_l1846_184698

-- Given conditions
def price_per_deck := 2
def total_decks_start := 5
def money_earned := 4

-- The number of decks sold
def decks_sold := money_earned / price_per_deck

-- The number of decks left
def decks_left := total_decks_start - decks_sold

-- The theorem to prove 
theorem decks_left_is_3 : decks_left = 3 :=
by
  -- Here we put the steps to prove
  sorry

end decks_left_is_3_l1846_184698


namespace min_both_attendees_l1846_184683

-- Defining the parameters and conditions
variable (n : ℕ) -- total number of attendees
variable (glasses name_tags both : ℕ) -- attendees wearing glasses, name tags, and both

-- Conditions provided in the problem
def wearing_glasses_condition (n : ℕ) (glasses : ℕ) : Prop := glasses = n / 3
def wearing_name_tags_condition (n : ℕ) (name_tags : ℕ) : Prop := name_tags = n / 2
def total_attendees_condition (n : ℕ) : Prop := n = 6

-- Theorem to prove the minimum attendees wearing both glasses and name tags is 1
theorem min_both_attendees (n glasses name_tags both : ℕ) (h1 : wearing_glasses_condition n glasses) 
  (h2 : wearing_name_tags_condition n name_tags) (h3 : total_attendees_condition n) : 
  both = 1 :=
sorry

end min_both_attendees_l1846_184683


namespace cube_faces_l1846_184631

theorem cube_faces : ∀ (c : {s : Type | ∃ (x y z : ℝ), s = ({ (x0, y0, z0) : ℝ × ℝ × ℝ | x0 ≤ x ∧ y0 ≤ y ∧ z0 ≤ z}) }), 
  ∃ (f : ℕ), f = 6 :=
by 
  -- proof would be written here
  sorry

end cube_faces_l1846_184631


namespace find_T_b_minus_T_neg_b_l1846_184641

noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

theorem find_T_b_minus_T_neg_b (b : ℝ) (h1 : -1 < b ∧ b < 1) (h2 : T b * T (-b) = 3240) (h3 : 1 - b^2 = 100 / 810) :
  T b - T (-b) = 324 * b :=
by
  sorry

end find_T_b_minus_T_neg_b_l1846_184641


namespace totalPieces_l1846_184674

   -- Definitions given by the conditions
   def packagesGum := 21
   def packagesCandy := 45
   def packagesMints := 30
   def piecesPerGumPackage := 9
   def piecesPerCandyPackage := 12
   def piecesPerMintPackage := 8

   -- Define the total pieces of gum, candy, and mints
   def totalPiecesGum := packagesGum * piecesPerGumPackage
   def totalPiecesCandy := packagesCandy * piecesPerCandyPackage
   def totalPiecesMints := packagesMints * piecesPerMintPackage

   -- The mathematical statement to prove
   theorem totalPieces :
     totalPiecesGum + totalPiecesCandy + totalPiecesMints = 969 :=
   by
     -- Proof is skipped
     sorry
   
end totalPieces_l1846_184674


namespace consecutive_product_neq_consecutive_even_product_l1846_184600

open Nat

theorem consecutive_product_neq_consecutive_even_product :
  ∀ m n : ℕ, m * (m + 1) ≠ 4 * n * (n + 1) :=
by
  intros m n
  -- Proof is omitted, as per instructions.
  sorry

end consecutive_product_neq_consecutive_even_product_l1846_184600


namespace angle_relationship_l1846_184630

variables {VU VW : ℝ} {x y z : ℝ} (h1 : VU = VW) 
          (angle_UXZ : ℝ) (angle_VYZ : ℝ) (angle_VZX : ℝ)
          (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z)

theorem angle_relationship (h1 : VU = VW) (h2 : angle_UXZ = x) (h3 : angle_VYZ = y) (h4 : angle_VZX = z) : 
    x = (y - z) / 2 := 
by 
    sorry

end angle_relationship_l1846_184630


namespace jessica_quarters_l1846_184640

theorem jessica_quarters (initial_quarters borrowed_quarters remaining_quarters : ℕ)
  (h1 : initial_quarters = 8)
  (h2 : borrowed_quarters = 3) :
  remaining_quarters = initial_quarters - borrowed_quarters → remaining_quarters = 5 :=
by
  intro h3
  rw [h1, h2] at h3
  exact h3

end jessica_quarters_l1846_184640


namespace max_value_f_max_value_f_at_13_l1846_184619

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_f : ∀ x : ℝ, f x ≤ 1 / 3 := by
  sorry

theorem max_value_f_at_13 : ∃ x : ℝ, f x = 1 / 3 := by
  sorry

end max_value_f_max_value_f_at_13_l1846_184619


namespace sum_of_six_smallest_multiples_of_12_l1846_184661

-- Define the six smallest distinct positive integer multiples of 12
def multiples_of_12 : List ℕ := [12, 24, 36, 48, 60, 72]

-- Define their sum
def sum_of_multiples : ℕ := multiples_of_12.sum

-- The proof statement
theorem sum_of_six_smallest_multiples_of_12 : sum_of_multiples = 252 := 
by
  sorry

end sum_of_six_smallest_multiples_of_12_l1846_184661


namespace surface_area_to_lateral_surface_ratio_cone_l1846_184625

noncomputable def cone_surface_lateral_area_ratio : Prop :=
  let radius : ℝ := 1
  let theta : ℝ := (2 * Real.pi) / 3
  let lateral_surface_area := Real.pi * radius^2 * (theta / (2 * Real.pi))
  let base_radius := (2 * Real.pi * radius * (theta / (2 * Real.pi))) / (2 * Real.pi)
  let base_area := Real.pi * base_radius^2
  let surface_area := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = (4 / 3)

theorem surface_area_to_lateral_surface_ratio_cone :
  cone_surface_lateral_area_ratio :=
  by
  sorry

end surface_area_to_lateral_surface_ratio_cone_l1846_184625


namespace c_share_l1846_184633

theorem c_share (A B C : ℝ) 
  (h1 : A = (1 / 2) * B)
  (h2 : B = (1 / 2) * C)
  (h3 : A + B + C = 392) : 
  C = 224 :=
by
  sorry

end c_share_l1846_184633


namespace product_of_last_two_digits_l1846_184650

theorem product_of_last_two_digits (A B : ℕ) (h₁ : A + B = 17) (h₂ : 4 ∣ (10 * A + B)) :
  A * B = 72 := sorry

end product_of_last_two_digits_l1846_184650


namespace arithmetic_sequence_term_l1846_184672

theorem arithmetic_sequence_term (a d n : ℕ) (h₀ : a = 1) (h₁ : d = 3) (h₂ : a + (n - 1) * d = 6019) :
  n = 2007 :=
sorry

end arithmetic_sequence_term_l1846_184672


namespace children_play_time_equal_l1846_184678

-- Definitions based on the conditions in the problem
def totalChildren := 7
def totalPlayingTime := 140
def playersAtATime := 2

-- The statement to be proved
theorem children_play_time_equal :
  (playersAtATime * totalPlayingTime) / totalChildren = 40 := by
sorry

end children_play_time_equal_l1846_184678


namespace min_value_fraction_l1846_184615

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) : 
  ∃x : ℝ, (x = (1/a + 2/b)) ∧ (∀y : ℝ, (y = (1/a + 2/b)) → y ≥ 8) :=
by
  sorry

end min_value_fraction_l1846_184615


namespace share_difference_l1846_184684

theorem share_difference (x : ℕ) (p q r : ℕ) 
  (h1 : 3 * x = p) 
  (h2 : 7 * x = q) 
  (h3 : 12 * x = r) 
  (h4 : q - p = 2800) : 
  r - q = 3500 := by {
  sorry
}

end share_difference_l1846_184684


namespace value_of_x_minus_2y_l1846_184643

theorem value_of_x_minus_2y (x y : ℝ) (h : 0.5 * x = y + 20) : x - 2 * y = 40 :=
sorry

end value_of_x_minus_2y_l1846_184643


namespace unique_rectangle_dimensions_l1846_184655

theorem unique_rectangle_dimensions (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < a ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = a * b / 4 :=
sorry

end unique_rectangle_dimensions_l1846_184655


namespace smallest_integer_value_of_x_l1846_184696

theorem smallest_integer_value_of_x (x : ℤ) (h : 7 + 3 * x < 26) : x = 6 :=
sorry

end smallest_integer_value_of_x_l1846_184696


namespace train_speed_l1846_184601

noncomputable def speed_of_train (length_of_train length_of_overbridge time: ℝ) : ℝ :=
  (length_of_train + length_of_overbridge) / time

theorem train_speed (length_of_train length_of_overbridge time speed: ℝ)
  (h1 : length_of_train = 600)
  (h2 : length_of_overbridge = 100)
  (h3 : time = 70)
  (h4 : speed = 10) :
  speed_of_train length_of_train length_of_overbridge time = speed :=
by
  simp [speed_of_train, h1, h2, h3, h4]
  sorry

end train_speed_l1846_184601


namespace minimum_sum_of_x_and_y_l1846_184623

variable (x y : ℝ)

-- Conditions
def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 4 * y = x * y

theorem minimum_sum_of_x_and_y (x y : ℝ) (h : conditions x y) : x + y ≥ 9 := by
  sorry

end minimum_sum_of_x_and_y_l1846_184623


namespace determine_exponent_l1846_184611

theorem determine_exponent (m : ℕ) (hm : m > 0) (h_symm : ∀ x : ℝ, x^m - 3 = (-(x))^m - 3)
  (h_decr : ∀ (x y : ℝ), 0 < x ∧ x < y → x^m - 3 > y^m - 3) : m = 1 := 
sorry

end determine_exponent_l1846_184611


namespace original_price_of_trouser_l1846_184676

-- Define conditions
def sale_price : ℝ := 20
def discount : ℝ := 0.80

-- Define what the proof aims to show
theorem original_price_of_trouser (P : ℝ) (h : sale_price = P * (1 - discount)) : P = 100 :=
sorry

end original_price_of_trouser_l1846_184676


namespace discounted_price_is_correct_l1846_184671

def original_price_of_cork (C : ℝ) : Prop :=
  C + (C + 2.00) = 2.10

def discounted_price_of_cork (C : ℝ) : ℝ :=
  C - (C * 0.12)

theorem discounted_price_is_correct :
  ∃ C : ℝ, original_price_of_cork C ∧ discounted_price_of_cork C = 0.044 :=
by
  sorry

end discounted_price_is_correct_l1846_184671


namespace divisor_inequality_l1846_184693

variable (k p : Nat)

-- Conditions
def is_prime (p : Nat) : Prop := Nat.Prime p
def is_divisor_of (k d : Nat) : Prop := ∃ m : Nat, d = k * m

-- Given conditions and the theorem to be proved
theorem divisor_inequality (h1 : k > 3) (h2 : is_prime p) (h3 : is_divisor_of k (2^p + 1)) : k ≥ 2 * p + 1 :=
  sorry

end divisor_inequality_l1846_184693


namespace find_n_l1846_184612

theorem find_n (n : ℕ) (h₀ : 0 ≤ n) (h₁ : n ≤ 11) (h₂ : 10389 % 12 = n) : n = 9 :=
by sorry

end find_n_l1846_184612


namespace circle_equation_l1846_184656

theorem circle_equation :
  ∃ (a : ℝ), (y - a)^2 + x^2 = 1 ∧ (1 - 0)^2 + (2 - a)^2 = 1 ∧
  ∀ a, (1 - 0)^2 + (2 - a)^2 = 1 → a = 2 →
  x^2 + (y - 2)^2 = 1 := by sorry

end circle_equation_l1846_184656


namespace closest_point_in_plane_l1846_184675

noncomputable def closest_point (x y z : ℚ) : Prop :=
  ∃ (t : ℚ), 
    x = 2 + 2 * t ∧ 
    y = 3 - 3 * t ∧ 
    z = 1 + 4 * t ∧ 
    2 * (2 + 2 * t) - 3 * (3 - 3 * t) + 4 * (1 + 4 * t) = 40

theorem closest_point_in_plane :
  closest_point (92 / 29) (16 / 29) (145 / 29) :=
by
  sorry

end closest_point_in_plane_l1846_184675


namespace incorrect_judgment_D_l1846_184681

theorem incorrect_judgment_D (p q : Prop) (hp : p = (2 + 3 = 5)) (hq : q = (5 < 4)) : 
  ¬((p ∧ q) ∧ (p ∨ q)) := by 
    sorry

end incorrect_judgment_D_l1846_184681


namespace find_money_of_Kent_l1846_184604

variable (Alison Brittany Brooke Kent : ℝ)

def money_relations (h1 : Alison = 4000)
    (h2 : Alison = Brittany / 2)
    (h3 : Brittany = 4 * Brooke)
    (h4 : Brooke = 2 * Kent) : Prop :=
  Kent = 1000

theorem find_money_of_Kent
  {Alison Brittany Brooke Kent : ℝ}
  (h1 : Alison = 4000)
  (h2 : Alison = Brittany / 2)
  (h3 : Brittany = 4 * Brooke)
  (h4 : Brooke = 2 * Kent) :
  money_relations Alison Brittany Brooke Kent h1 h2 h3 h4 :=
by 
  sorry

end find_money_of_Kent_l1846_184604


namespace auditorium_seats_l1846_184660

variable (S : ℕ)

theorem auditorium_seats (h1 : 2 * S / 5 + S / 10 + 250 = S) : S = 500 :=
by
  sorry

end auditorium_seats_l1846_184660


namespace line_length_l1846_184677

theorem line_length (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := 
by
  sorry

end line_length_l1846_184677


namespace compute_expression_l1846_184632

/-- Definitions of parts of the expression --/
def expr1 := 6 ^ 2
def expr2 := 4 * 5
def expr3 := 2 ^ 3
def expr4 := 4 ^ 2 / 2

/-- Main statement to prove --/
theorem compute_expression : expr1 + expr2 - expr3 + expr4 = 56 := 
by
  sorry

end compute_expression_l1846_184632


namespace pancakes_needed_l1846_184614

def short_stack_pancakes : ℕ := 3
def big_stack_pancakes : ℕ := 5
def short_stack_customers : ℕ := 9
def big_stack_customers : ℕ := 6

theorem pancakes_needed : (short_stack_customers * short_stack_pancakes + big_stack_customers * big_stack_pancakes) = 57 :=
by
  sorry

end pancakes_needed_l1846_184614


namespace find_k_l1846_184617

def distances (S x y k : ℝ) := (S - x * 0.75) * x / (x + y) + 0.75 * x = S * x / (x + y) - 18 ∧
                              S * x / (x + y) - (S - y / 3) * x / (x + y) = k

theorem find_k (S x y k : ℝ) (h₁ : x * y / (x + y) = 24) (h₂ : k = 24 / 3)
  : k = 8 :=
by 
  -- We need to fill in the proof steps here
  sorry

end find_k_l1846_184617


namespace triangle_angle_sum_l1846_184664

theorem triangle_angle_sum (a : ℝ) (x : ℝ) :
  0 < 2 * a + 20 ∧ 0 < 3 * a - 15 ∧ 0 < 175 - 5 * a ∧
  2 * a + 20 + 3 * a - 15 + x = 180 → 
  x = 175 - 5 * a ∧ max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
sorry

end triangle_angle_sum_l1846_184664


namespace harry_ron_difference_l1846_184663

-- Define the amounts each individual paid
def harry_paid : ℕ := 150
def ron_paid : ℕ := 180
def hermione_paid : ℕ := 210

-- Define the total amount
def total_paid : ℕ := harry_paid + ron_paid + hermione_paid

-- Define the amount each should have paid
def equal_share : ℕ := total_paid / 3

-- Define the amount Harry owes to Hermione
def harry_owes : ℕ := equal_share - harry_paid

-- Define the amount Ron owes to Hermione
def ron_owes : ℕ := equal_share - ron_paid

-- Define the difference between what Harry and Ron owe Hermione
def difference : ℕ := harry_owes - ron_owes

-- Prove that the difference is 30
theorem harry_ron_difference : difference = 30 := by
  sorry

end harry_ron_difference_l1846_184663


namespace positive_numbers_l1846_184679

theorem positive_numbers 
    (a b c : ℝ) 
    (h1 : a + b + c > 0) 
    (h2 : ab + bc + ca > 0) 
    (h3 : abc > 0) 
    : a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end positive_numbers_l1846_184679


namespace difference_of_squares_is_39_l1846_184629

theorem difference_of_squares_is_39 (L S : ℕ) (h1 : L = 8) (h2 : L - S = 3) : L^2 - S^2 = 39 :=
by
  sorry

end difference_of_squares_is_39_l1846_184629


namespace sequence_general_term_l1846_184603

-- Define the sequence
def a : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * a n + 1

-- State the theorem
theorem sequence_general_term (n : ℕ) : a n = 2^n - 1 :=
sorry

end sequence_general_term_l1846_184603


namespace number_exceeds_20_percent_by_40_eq_50_l1846_184668

theorem number_exceeds_20_percent_by_40_eq_50 (x : ℝ) (h : x = 0.20 * x + 40) : x = 50 := by
  sorry

end number_exceeds_20_percent_by_40_eq_50_l1846_184668


namespace arithmetic_mean_34_58_l1846_184646

theorem arithmetic_mean_34_58 :
  (3 / 4 : ℚ) + (5 / 8 : ℚ) / 2 = 11 / 16 := sorry

end arithmetic_mean_34_58_l1846_184646


namespace remaining_battery_life_l1846_184670

theorem remaining_battery_life :
  let capacity1 := 60
  let capacity2 := 80
  let capacity3 := 120
  let used1 := capacity1 * (3 / 4 : ℚ)
  let used2 := capacity2 * (1 / 2 : ℚ)
  let used3 := capacity3 * (2 / 3 : ℚ)
  let remaining1 := capacity1 - used1 - 2
  let remaining2 := capacity2 - used2 - 2
  let remaining3 := capacity3 - used3 - 2
  remaining1 + remaining2 + remaining3 = 89 := 
by
  sorry

end remaining_battery_life_l1846_184670


namespace positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l1846_184626

-- Definitions for the conditions
def eq1 (x y : ℝ) := x + 2 * y = 6
def eq2 (x y m : ℝ) := x - 2 * y + m * x + 5 = 0

-- Theorem for part (1)
theorem positive_integer_solutions :
  {x y : ℕ} → eq1 x y → (x = 4 ∧ y = 1) ∨ (x = 2 ∧ y = 2) :=
sorry

-- Theorem for part (2)
theorem value_of_m_when_sum_is_zero (x y : ℝ) (h : x + y = 0) :
  eq1 x y → ∃ m : ℝ, eq2 x y m → m = -13/6 :=
sorry

-- Theorem for part (3)
theorem fixed_solution (m : ℝ) : eq2 0 2.5 m :=
sorry

-- Theorem for part (4)
theorem integer_values_of_m (x : ℤ) :
  (∃ y : ℤ, eq1 x y ∧ ∃ m : ℤ, eq2 x y m) → m = -1 ∨ m = -3 :=
sorry

end positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l1846_184626


namespace range_of_m_for_real_roots_value_of_m_for_specific_roots_l1846_184662

open Real

variable {m x : ℝ}

def quadratic (m : ℝ) (x : ℝ) := x^2 + 2*(m-1)*x + m^2 + 2 = 0
  
theorem range_of_m_for_real_roots (h : ∃ x : ℝ, quadratic m x) : m ≤ -1/2 :=
sorry

theorem value_of_m_for_specific_roots
  (h : quadratic m x)
  (Hroots : ∃ x1 x2 : ℝ, quadratic m x1 ∧ quadratic m x2 ∧ (x1 - x2)^2 = 18 - x1 * x2) :
  m = -2 :=
sorry

end range_of_m_for_real_roots_value_of_m_for_specific_roots_l1846_184662


namespace john_avg_speed_last_30_minutes_l1846_184606

open Real

/-- John drove 160 miles in 120 minutes. His average speed during the first
30 minutes was 55 mph, during the second 30 minutes was 75 mph, and during
the third 30 minutes was 60 mph. Prove that his average speed during the
last 30 minutes was 130 mph. -/
theorem john_avg_speed_last_30_minutes (total_distance : ℝ) (total_time_minutes : ℝ)
  (speed_1 : ℝ) (speed_2 : ℝ) (speed_3 : ℝ) (speed_4 : ℝ) :
  total_distance = 160 →
  total_time_minutes = 120 →
  speed_1 = 55 →
  speed_2 = 75 →
  speed_3 = 60 →
  (speed_1 + speed_2 + speed_3 + speed_4) / 4 = total_distance / (total_time_minutes / 60) →
  speed_4 = 130 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end john_avg_speed_last_30_minutes_l1846_184606


namespace distance_traveled_is_6000_l1846_184639

-- Define the conditions and the question in Lean 4
def footprints_per_meter_Pogo := 4
def footprints_per_meter_Grimzi := 3 / 6
def combined_total_footprints := 27000

theorem distance_traveled_is_6000 (D : ℕ) :
  footprints_per_meter_Pogo * D + footprints_per_meter_Grimzi * D = combined_total_footprints →
  D = 6000 :=
by
  sorry

end distance_traveled_is_6000_l1846_184639


namespace inequality_proof_l1846_184642

variable {a b c : ℝ}

theorem inequality_proof (ha : a > 0) (hb : b > 0) (hc : c > 0) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
by
  sorry

end inequality_proof_l1846_184642


namespace percentage_saved_l1846_184647

theorem percentage_saved (amount_saved : ℝ) (amount_spent : ℝ) (h1 : amount_saved = 5) (h2 : amount_spent = 45) : 
  (amount_saved / (amount_spent + amount_saved)) * 100 = 10 :=
by 
  sorry

end percentage_saved_l1846_184647


namespace negation_forall_pos_l1846_184628

theorem negation_forall_pos (h : ∀ x : ℝ, 0 < x → x^2 + x + 1 > 0) :
  ∃ x : ℝ, 0 < x ∧ x^2 + x + 1 ≤ 0 :=
sorry

end negation_forall_pos_l1846_184628


namespace part1_part2_l1846_184680

-- Definitions for sets A and B
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x < -5 ∨ x > 1}

-- Prove (1): A ∪ B
theorem part1 : A ∪ B = {x : ℝ | x < -5 ∨ x > -3} :=
by
  sorry

-- Prove (2): A ∩ (ℝ \ B)
theorem part2 : A ∩ (Set.compl B) = {x : ℝ | -3 < x ∧ x ≤ 1} :=
by
  sorry

end part1_part2_l1846_184680


namespace margarets_mean_score_l1846_184648

noncomputable def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

open List

theorem margarets_mean_score :
  let scores := [86, 88, 91, 93, 95, 97, 99, 100]
  let cyprians_mean := 92
  let num_scores := 8
  let cyprians_scores := 4
  let margarets_scores := num_scores - cyprians_scores
  (scores.sum - cyprians_scores * cyprians_mean) / margarets_scores = 95.25 :=
by
  sorry

end margarets_mean_score_l1846_184648


namespace problem_equivalence_l1846_184695

theorem problem_equivalence : 4 * 299 + 3 * 299 + 2 * 299 + 298 = 2989 := by
  sorry

end problem_equivalence_l1846_184695


namespace quadratic_k_value_l1846_184652

theorem quadratic_k_value (a b c k : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : 4 * b * b - k * a * c = 0): 
  k = 16 / 3 :=
by
  sorry

end quadratic_k_value_l1846_184652


namespace wire_cut_square_octagon_area_l1846_184621

theorem wire_cut_square_octagon_area (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (equal_area : (a / 4)^2 = (2 * (b / 8)^2 * (1 + Real.sqrt 2))) : 
  a / b = Real.sqrt ((1 + Real.sqrt 2) / 2) := 
  sorry

end wire_cut_square_octagon_area_l1846_184621


namespace scientific_notation_of_32000000_l1846_184635

theorem scientific_notation_of_32000000 : 32000000 = 3.2 * 10^7 := 
  sorry

end scientific_notation_of_32000000_l1846_184635


namespace boat_distance_downstream_l1846_184699

theorem boat_distance_downstream (speed_boat_still: ℕ) (speed_stream: ℕ) (time: ℕ)
    (h1: speed_boat_still = 25)
    (h2: speed_stream = 5)
    (h3: time = 4) :
    (speed_boat_still + speed_stream) * time = 120 := 
sorry

end boat_distance_downstream_l1846_184699


namespace compare_logarithms_l1846_184682

theorem compare_logarithms (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3) 
                           (h2 : b = (Real.log 2 / Real.log 3)^2) 
                           (h3 : c = Real.log (2/3) / Real.log 4) : c < b ∧ b < a :=
by
  sorry

end compare_logarithms_l1846_184682


namespace isabel_reading_homework_pages_l1846_184694

-- Definitions for the given problem
def num_math_pages := 2
def problems_per_page := 5
def total_problems := 30

-- Calculation based on conditions
def math_problems := num_math_pages * problems_per_page
def reading_problems := total_problems - math_problems

-- The statement to be proven
theorem isabel_reading_homework_pages : (reading_problems / problems_per_page) = 4 :=
by
  -- The proof would go here.
  sorry

end isabel_reading_homework_pages_l1846_184694


namespace gcd_891_810_l1846_184627

theorem gcd_891_810 : Nat.gcd 891 810 = 81 := 
by
  sorry

end gcd_891_810_l1846_184627


namespace Bret_catches_12_frogs_l1846_184651

-- Conditions from the problem
def frogs_caught_by_Alster : Nat := 2
def frogs_caught_by_Quinn : Nat := 2 * frogs_caught_by_Alster
def frogs_caught_by_Bret : Nat := 3 * frogs_caught_by_Quinn

-- Statement of the theorem to be proved
theorem Bret_catches_12_frogs : frogs_caught_by_Bret = 12 :=
by
  sorry

end Bret_catches_12_frogs_l1846_184651


namespace shaded_area_proof_l1846_184636

noncomputable def total_shaded_area (side_length: ℝ) (large_square_ratio: ℝ) (small_square_ratio: ℝ): ℝ := 
  let S := side_length / large_square_ratio
  let T := S / small_square_ratio
  let large_square_area := S ^ 2
  let small_square_area := T ^ 2
  large_square_area + 12 * small_square_area

theorem shaded_area_proof
  (h1: ∀ side_length, side_length = 15)
  (h2: ∀ large_square_ratio, large_square_ratio = 5)
  (h3: ∀ small_square_ratio, small_square_ratio = 4)
  : total_shaded_area 15 5 4 = 15.75 :=
by
  sorry

end shaded_area_proof_l1846_184636


namespace quadratic_inequality_solution_l1846_184691

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end quadratic_inequality_solution_l1846_184691


namespace max_participants_l1846_184649

structure MeetingRoom where
  rows : ℕ
  cols : ℕ
  seating : ℕ → ℕ → Bool -- A function indicating if a seat (i, j) is occupied (true) or not (false)
  row_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating i (j+1) → seating i (j+2) → False
  col_condition : ∀ i : ℕ, ∀ j : ℕ, seating i j → seating (i+1) j → seating (i+2) j → False

theorem max_participants {room : MeetingRoom} (h : room.rows = 4 ∧ room.cols = 4) : 
  (∃ n : ℕ, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → n < 12) ∧
            (∀ m, (∀ i < room.rows, ∀ j < room.cols, room.seating i j → m < 12) → m ≤ 11)) :=
  sorry

end max_participants_l1846_184649


namespace length_of_each_side_is_25_nails_l1846_184688

-- Definitions based on the conditions
def nails_per_side := 25
def total_nails := 96

-- The theorem stating the equivalent mathematical problem
theorem length_of_each_side_is_25_nails
  (n : ℕ) (h1 : n = nails_per_side * 4 - 4)
  (h2 : total_nails = 96):
  n = nails_per_side :=
by
  sorry

end length_of_each_side_is_25_nails_l1846_184688


namespace factor_expression_l1846_184605

variable (x : ℝ)

theorem factor_expression :
  (18 * x ^ 6 + 50 * x ^ 4 - 8) - (2 * x ^ 6 - 6 * x ^ 4 - 8) = 8 * x ^ 4 * (2 * x ^ 2 + 7) :=
by
  sorry

end factor_expression_l1846_184605
