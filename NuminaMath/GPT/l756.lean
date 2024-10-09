import Mathlib

namespace batsman_average_increase_l756_75663

theorem batsman_average_increase 
  (A : ℕ)
  (h1 : ∀ n ≤ 11, (1 / (n : ℝ)) * (A * n + 60) = 38) 
  (h2 : 1 / 12 * (A * 11 + 60) = 38)
  (h3 : ∀ n ≤ 12, (A * n : ℝ) ≤ (A * (n + 1) : ℝ)) :
  38 - A = 2 := 
sorry

end batsman_average_increase_l756_75663


namespace rope_length_91_4_l756_75625

noncomputable def total_rope_length (n : ℕ) (d : ℕ) (pi_val : Real) : Real :=
  let linear_segments := 6 * d
  let arc_length := (d * pi_val / 3) * 6
  let total_length_per_tie := linear_segments + arc_length
  total_length_per_tie * 2

theorem rope_length_91_4 :
  total_rope_length 7 5 3.14 = 91.4 :=
by
  sorry

end rope_length_91_4_l756_75625


namespace vityas_miscalculation_l756_75697

/-- Vitya's miscalculated percentages problem -/
theorem vityas_miscalculation :
  ∀ (N : ℕ)
  (acute obtuse nonexistent right depends_geometry : ℕ)
  (H_acute : acute = 5)
  (H_obtuse : obtuse = 5)
  (H_nonexistent : nonexistent = 5)
  (H_right : right = 50)
  (H_total : acute + obtuse + nonexistent + right + depends_geometry = 100),
  depends_geometry = 110 :=
by
  intros
  sorry

end vityas_miscalculation_l756_75697


namespace runner_time_difference_l756_75630

theorem runner_time_difference 
  (v : ℝ)  -- runner's initial speed (miles per hour)
  (H1 : 0 < v)  -- speed is positive
  (d : ℝ)  -- total distance
  (H2 : d = 40)  -- total distance condition
  (t2 : ℝ)  -- time taken for the second half
  (H3 : t2 = 10)  -- second half time condition
  (H4 : v ≠ 0)  -- initial speed cannot be zero
  (H5: 20 = 10 * (v / 2))  -- equation derived from the second half conditions
  : (t2 - (20 / v)) = 5 := 
by
  sorry

end runner_time_difference_l756_75630


namespace size_ratio_l756_75635

variable {U : ℝ} (h1 : C = 1.5 * U) (h2 : R = 4 / 3 * C)

theorem size_ratio : R = 8 / 3 * U :=
by
  sorry

end size_ratio_l756_75635


namespace remaining_cubes_l756_75664

-- The configuration of the initial cube and the properties of a layer
def initial_cube : ℕ := 10
def total_cubes : ℕ := 1000
def layer_cubes : ℕ := (initial_cube * initial_cube)

-- The proof problem: Prove that the remaining number of cubes is 900 after removing one layer
theorem remaining_cubes : total_cubes - layer_cubes = 900 := 
by 
  sorry

end remaining_cubes_l756_75664


namespace odd_function_value_l756_75699

def f (a x : ℝ) : ℝ := -x^3 + (a-2)*x^2 + x

-- Test that f(x) is an odd function:
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_value (a : ℝ) (h : is_odd_function (f a)) : f a a = -6 :=
by
  sorry

end odd_function_value_l756_75699


namespace number_of_added_groups_l756_75604

-- Define the total number of students in the class
def total_students : ℕ := 47

-- Define the number of students per table and the number of tables
def students_per_table : ℕ := 3
def number_of_tables : ℕ := 6

-- Define the number of girls in the bathroom and the multiplier for students in the canteen
def girls_in_bathroom : ℕ := 3
def canteen_multiplier : ℕ := 3

-- Define the number of foreign exchange students from each country
def foreign_exchange_germany : ℕ := 3
def foreign_exchange_france : ℕ := 3
def foreign_exchange_norway : ℕ := 3

-- Define the number of students per recently added group
def students_per_group : ℕ := 4

-- Calculate the number of students currently in the classroom
def students_in_classroom := number_of_tables * students_per_table

-- Calculate the number of students temporarily absent
def students_in_canteen := girls_in_bathroom * canteen_multiplier
def temporarily_absent := girls_in_bathroom + students_in_canteen

-- Calculate the number of foreign exchange students missing
def foreign_exchange_missing := foreign_exchange_germany + foreign_exchange_france + foreign_exchange_norway

-- Calculate the total number of students accounted for
def student_accounted_for := students_in_classroom + temporarily_absent + foreign_exchange_missing

-- The proof statement (main goal)
theorem number_of_added_groups : (total_students - student_accounted_for) / students_per_group = 2 :=
by
  sorry

end number_of_added_groups_l756_75604


namespace chess_tournament_games_l756_75654

theorem chess_tournament_games (n : ℕ) (h : 2 * 404 = n * (n - 4)) : False :=
by
  sorry

end chess_tournament_games_l756_75654


namespace prove_smallest_geometric_third_term_value_l756_75665

noncomputable def smallest_value_geometric_third_term : ℝ :=
  let d_1 := -5 + 10 * Real.sqrt 2
  let d_2 := -5 - 10 * Real.sqrt 2
  let g3_1 := 39 + 2 * d_1
  let g3_2 := 39 + 2 * d_2
  min g3_1 g3_2

theorem prove_smallest_geometric_third_term_value :
  smallest_value_geometric_third_term = 29 - 20 * Real.sqrt 2 := by sorry

end prove_smallest_geometric_third_term_value_l756_75665


namespace find_subtracted_number_l756_75616

theorem find_subtracted_number (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
sorry

end find_subtracted_number_l756_75616


namespace additional_amount_needed_l756_75656

-- Define the amounts spent on shampoo, conditioner, and lotion
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost_per_bottle : ℝ := 6.00
def lotion_quantity : ℕ := 3

-- Define the amount required for free shipping
def free_shipping_threshold : ℝ := 50.00

-- Calculate the total amount spent
def total_spent : ℝ := shampoo_cost + conditioner_cost + (lotion_quantity * lotion_cost_per_bottle)

-- Define the additional amount needed for free shipping
def additional_needed_for_shipping : ℝ := free_shipping_threshold - total_spent

-- The final goal to prove
theorem additional_amount_needed : additional_needed_for_shipping = 12.00 :=
by
  sorry

end additional_amount_needed_l756_75656


namespace modulo_residue_addition_l756_75684

theorem modulo_residue_addition : 
  (368 + 3 * 78 + 8 * 242 + 6 * 22) % 11 = 8 := 
by
  have h1 : 368 % 11 = 5 := by sorry
  have h2 : 78 % 11 = 1 := by sorry
  have h3 : 242 % 11 = 0 := by sorry
  have h4 : 22 % 11 = 0 := by sorry
  sorry

end modulo_residue_addition_l756_75684


namespace binomial_product_l756_75611

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l756_75611


namespace alpha_in_third_quadrant_l756_75615

theorem alpha_in_third_quadrant (k : ℤ) (α : ℝ) :
  (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60 → 180 < α ∧ α < 240 :=
  sorry

end alpha_in_third_quadrant_l756_75615


namespace root_proof_l756_75651

noncomputable def p : ℝ := (-5 + Real.sqrt 21) / 2
noncomputable def q : ℝ := (-5 - Real.sqrt 21) / 2

theorem root_proof :
  (∃ (p q : ℝ), (∀ x : ℝ, x^3 + 6 * x^2 + 6 * x + 1 = 0 → (x = p ∨ x = q ∨ x = -1)) ∧ 
                 ((p = (-5 + Real.sqrt 21) / 2) ∧ (q = (-5 - Real.sqrt 21) / 2))) →
  (p / q + q / p = 23) :=
by
  sorry

end root_proof_l756_75651


namespace represent_sum_and_product_eq_231_l756_75650

theorem represent_sum_and_product_eq_231 :
  ∃ (x y z w : ℕ), x = 3 ∧ y = 7 ∧ z = 11 ∧ w = 210 ∧ (231 = x + y + z + w) ∧ (231 = x * y * z) :=
by
  -- The proof is omitted here.
  sorry

end represent_sum_and_product_eq_231_l756_75650


namespace remainder_of_4123_div_by_32_l756_75648

theorem remainder_of_4123_div_by_32 : 
  ∃ r, 0 ≤ r ∧ r < 32 ∧ 4123 = 32 * (4123 / 32) + r ∧ r = 27 := by
  sorry

end remainder_of_4123_div_by_32_l756_75648


namespace dinner_cost_l756_75645

theorem dinner_cost (tax_rate tip_rate total_cost : ℝ) (h_tax : tax_rate = 0.12) (h_tip : tip_rate = 0.20) (h_total : total_cost = 30.60) :
  let meal_cost := total_cost / (1 + tax_rate + tip_rate)
  meal_cost = 23.18 :=
by
  sorry

end dinner_cost_l756_75645


namespace cost_per_meter_of_fencing_l756_75606

/-- The sides of the rectangular field -/
def sides_ratio (length width : ℕ) : Prop := 3 * width = 4 * length

/-- The area of the rectangular field -/
def area (length width area : ℕ) : Prop := length * width = area

/-- The cost per meter of fencing -/
def cost_per_meter (total_cost perimeter : ℕ) : ℕ := total_cost * 100 / perimeter

/-- Prove that the cost per meter of fencing the field in paise is 25 given:
 1) The sides of a rectangular field are in the ratio 3:4.
 2) The area of the field is 8112 sq. m.
 3) The total cost of fencing the field is 91 rupees. -/
theorem cost_per_meter_of_fencing
  (length width perimeter : ℕ) 
  (h1 : sides_ratio length width)
  (h2 : area length width 8112)
  (h3 : perimeter = 2 * (length + width))
  (total_cost : ℕ)
  (h4 : total_cost = 91) :
  cost_per_meter total_cost perimeter = 25 :=
by
  sorry

end cost_per_meter_of_fencing_l756_75606


namespace sum_odds_200_600_l756_75693

-- Define the bounds 200 and 600 for our range
def lower_bound := 200
def upper_bound := 600

-- Define first and last odd integers in the range
def first_odd := 201
def last_odd := 599

-- Define the common difference in our arithmetic sequence
def common_diff := 2

-- Number of terms in the sequence
def n := ((last_odd - first_odd) / common_diff) + 1

-- Sum of the arithmetic sequence formula
def sum_arithmetic_seq (n : ℕ) (a l : ℕ) : ℕ :=
  n * (a + l) / 2

-- Specifically, the sum of odd integers between 200 and 600
def sum_odd_integers : ℕ := sum_arithmetic_seq n first_odd last_odd

-- Theorem stating the sum is equal to 80000
theorem sum_odds_200_600 : sum_odd_integers = 80000 :=
by sorry

end sum_odds_200_600_l756_75693


namespace percentage_increase_is_20_percent_l756_75601

noncomputable def SP : ℝ := 8600
noncomputable def CP : ℝ := 7166.67
noncomputable def percentageIncrease : ℝ := ((SP - CP) / CP) * 100

theorem percentage_increase_is_20_percent : percentageIncrease = 20 :=
by
  sorry

end percentage_increase_is_20_percent_l756_75601


namespace license_plate_count_l756_75620

-- Define the conditions
def num_digits : ℕ := 5
def num_letters : ℕ := 2
def digit_choices : ℕ := 10
def letter_choices : ℕ := 26

-- Define the statement to prove the total number of distinct licenses plates
theorem license_plate_count : 
  (digit_choices ^ num_digits) * (letter_choices ^ num_letters) * 2 = 2704000 :=
by
  sorry

end license_plate_count_l756_75620


namespace Theresa_video_games_l756_75600

variable (Tory Julia Theresa : ℕ)

def condition1 : Prop := Tory = 6
def condition2 : Prop := Julia = Tory / 3
def condition3 : Prop := Theresa = (Julia * 3) + 5

theorem Theresa_video_games : condition1 Tory → condition2 Tory Julia → condition3 Julia Theresa → Theresa = 11 := by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end Theresa_video_games_l756_75600


namespace area_of_T_prime_l756_75683

-- Given conditions
def AreaBeforeTransformation : ℝ := 9

def TransformationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4],![-2, 5]]

def AreaAfterTransformation (M : Matrix (Fin 2) (Fin 2) ℝ) (area_before : ℝ) : ℝ :=
  (M.det) * area_before

-- Problem statement
theorem area_of_T_prime : 
  AreaAfterTransformation TransformationMatrix AreaBeforeTransformation = 207 :=
by
  sorry

end area_of_T_prime_l756_75683


namespace problem_statement_l756_75628

-- Define the operation #
def op_hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The main theorem statement
theorem problem_statement (a b : ℕ) (h1 : op_hash a b = 100) : (a + b) + 6 = 11 := 
sorry

end problem_statement_l756_75628


namespace slightly_used_crayons_correct_l756_75638

def total_crayons : ℕ := 120
def new_crayons : ℕ := total_crayons / 3
def broken_crayons : ℕ := (total_crayons * 20) / 100
def slightly_used_crayons : ℕ := total_crayons - new_crayons - broken_crayons

theorem slightly_used_crayons_correct : slightly_used_crayons = 56 := sorry

end slightly_used_crayons_correct_l756_75638


namespace complement_union_l756_75607

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l756_75607


namespace maggi_ate_5_cupcakes_l756_75695

theorem maggi_ate_5_cupcakes
  (packages : ℕ)
  (cupcakes_per_package : ℕ)
  (left_cupcakes : ℕ)
  (total_cupcakes : ℕ := packages * cupcakes_per_package)
  (eaten_cupcakes : ℕ := total_cupcakes - left_cupcakes)
  (h1 : packages = 3)
  (h2 : cupcakes_per_package = 4)
  (h3 : left_cupcakes = 7) :
  eaten_cupcakes = 5 :=
by
  sorry

end maggi_ate_5_cupcakes_l756_75695


namespace total_weeds_correct_l756_75686

def tuesday : ℕ := 25
def wednesday : ℕ := 3 * tuesday
def thursday : ℕ := wednesday / 5
def friday : ℕ := thursday - 10
def total_weeds : ℕ := tuesday + wednesday + thursday + friday

theorem total_weeds_correct : total_weeds = 120 :=
by
  sorry

end total_weeds_correct_l756_75686


namespace negative_integer_is_minus_21_l756_75609

variable (n : ℤ) (hn : n < 0) (h : n * (-3) + 2 = 65)

theorem negative_integer_is_minus_21 : n = -21 :=
by
  sorry

end negative_integer_is_minus_21_l756_75609


namespace sqrt_5_is_quadratic_radical_l756_75680

variable (a : ℝ) -- a is a real number

-- Definition to check if a given expression is a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

theorem sqrt_5_is_quadratic_radical : is_quadratic_radical 5 :=
by
  -- Here, 'by' indicates the start of the proof block,
  -- but the actual content of the proof is replaced with 'sorry' as instructed.
  sorry

end sqrt_5_is_quadratic_radical_l756_75680


namespace calculate_square_difference_l756_75627

theorem calculate_square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 :=
by
  sorry

end calculate_square_difference_l756_75627


namespace standard_equation_of_circle_l756_75617

theorem standard_equation_of_circle
  (r : ℝ) (h_radius : r = 1)
  (h_center : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (x, y) = (a, b))
  (h_tangent_line : ∃ (a : ℝ), 1 = |4 * a - 3| / 5)
  (h_tangent_x_axis : ∃ (a : ℝ), a = 1) :
  (∃ (a b : ℝ), (x-2)^2 + (y-1)^2 = 1) :=
sorry

end standard_equation_of_circle_l756_75617


namespace find_consecutive_numbers_l756_75675

theorem find_consecutive_numbers :
  ∃ (a b c d : ℕ),
      a % 11 = 0 ∧
      b % 7 = 0 ∧
      c % 5 = 0 ∧
      d % 4 = 0 ∧
      b = a + 1 ∧
      c = a + 2 ∧
      d = a + 3 ∧
      (a % 10) = 3 ∧
      (b % 10) = 4 ∧
      (c % 10) = 5 ∧
      (d % 10) = 6 :=
sorry

end find_consecutive_numbers_l756_75675


namespace ninth_term_of_geometric_sequence_l756_75679

theorem ninth_term_of_geometric_sequence (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) : a * r^8 = 19683 := by
  sorry

end ninth_term_of_geometric_sequence_l756_75679


namespace function_periodicity_l756_75631

variable {R : Type*} [Ring R]

def periodic_function (f : R → R) (k : R) : Prop :=
  ∀ x : R, f (x + 4*k) = f x

theorem function_periodicity {f : ℝ → ℝ} {k : ℝ} (h : ∀ x, f (x + k) * (1 - f x) = 1 + f x) (hk : k ≠ 0) : 
  periodic_function f k :=
sorry

end function_periodicity_l756_75631


namespace smallest_positive_n_l756_75640

theorem smallest_positive_n (n : ℕ) (h : 1023 * n % 30 = 2147 * n % 30) : n = 15 :=
by
  sorry

end smallest_positive_n_l756_75640


namespace parabola_vertex_l756_75603

theorem parabola_vertex (y x : ℝ) : y^2 - 4*y + 3*x + 7 = 0 → (x = -1 ∧ y = 2) := 
sorry

end parabola_vertex_l756_75603


namespace range_of_expression_l756_75618

theorem range_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  -21 ≤ a^2 + b^2 - 6*a - 8*b ∧ a^2 + b^2 - 6*a - 8*b ≤ 39 :=
by
  sorry

end range_of_expression_l756_75618


namespace soda_costs_94_cents_l756_75602

theorem soda_costs_94_cents (b s: ℤ) (h1 : 4 * b + 3 * s = 500) (h2 : 3 * b + 4 * s = 540) : s = 94 := 
by
  sorry

end soda_costs_94_cents_l756_75602


namespace find_difference_square_l756_75632

theorem find_difference_square (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 6) :
  (x - y)^2 = 25 :=
by
  sorry

end find_difference_square_l756_75632


namespace total_cost_is_80_l756_75639

-- Conditions
def cost_flour := 3 * 3
def cost_eggs := 3 * 10
def cost_milk := 7 * 5
def cost_baking_soda := 2 * 3

-- Question and proof requirement
theorem total_cost_is_80 : cost_flour + cost_eggs + cost_milk + cost_baking_soda = 80 := by
  sorry

end total_cost_is_80_l756_75639


namespace solve_fractional_eq_l756_75624

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) : (4 / (x - 2) = 2 / x) → (x = -2) :=
by 
  sorry

end solve_fractional_eq_l756_75624


namespace problem_l756_75608

def binom (n k : ℕ) : ℕ := n.choose k

def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem problem : binom 10 3 * perm 8 2 = 6720 := by
  sorry

end problem_l756_75608


namespace hotel_made_correct_revenue_l756_75629

noncomputable def hotelRevenue : ℕ :=
  let totalRooms := 260
  let doubleRooms := 196
  let singleRoomCost := 35
  let doubleRoomCost := 60
  let singleRooms := totalRooms - doubleRooms
  let revenueSingleRooms := singleRooms * singleRoomCost
  let revenueDoubleRooms := doubleRooms * doubleRoomCost
  revenueSingleRooms + revenueDoubleRooms

theorem hotel_made_correct_revenue :
  hotelRevenue = 14000 := by
  sorry

end hotel_made_correct_revenue_l756_75629


namespace decimal_to_vulgar_fraction_l756_75621

theorem decimal_to_vulgar_fraction (d : ℚ) (h : d = 0.36) : d = 9 / 25 :=
by {
  sorry
}

end decimal_to_vulgar_fraction_l756_75621


namespace Nancy_money_in_dollars_l756_75690

-- Condition: Nancy has saved 1 dozen quarters
def dozen : ℕ := 12

-- Condition: Each quarter is worth 25 cents
def value_of_quarter : ℕ := 25

-- Condition: 100 cents is equal to 1 dollar
def cents_per_dollar : ℕ := 100

-- Proving that Nancy has 3 dollars
theorem Nancy_money_in_dollars :
  (dozen * value_of_quarter) / cents_per_dollar = 3 := by
  sorry

end Nancy_money_in_dollars_l756_75690


namespace length_of_ln_l756_75649

theorem length_of_ln (sin_N_eq : Real.sin angle_N = 3 / 5) (LM_eq : length_LM = 15) :
  length_LN = 25 :=
sorry

end length_of_ln_l756_75649


namespace group_discount_l756_75612

theorem group_discount (P : ℝ) (D : ℝ) :
  4 * (P - (D / 100) * P) = 3 * P → D = 25 :=
by
  intro h
  sorry

end group_discount_l756_75612


namespace fluctuations_B_greater_than_A_l756_75687

variable (A B : Type)
variable (mean_A mean_B : ℝ)
variable (var_A var_B : ℝ)

-- Given conditions
axiom avg_A : mean_A = 5
axiom avg_B : mean_B = 5
axiom variance_A : var_A = 0.1
axiom variance_B : var_B = 0.2

-- The proof problem statement
theorem fluctuations_B_greater_than_A : var_A < var_B :=
by sorry

end fluctuations_B_greater_than_A_l756_75687


namespace square_side_length_tangent_circle_l756_75659

theorem square_side_length_tangent_circle (r s : ℝ) :
  (∃ (O : ℝ × ℝ) (A : ℝ × ℝ) (AB : ℝ) (AD : ℝ),
    AB = AD ∧
    O = (r, r) ∧
    A = (0, 0) ∧
    dist O A = r * Real.sqrt 2 ∧
    s = dist (O.fst, 0) A ∧
    s = dist (0, O.snd) A ∧
    ∀ x y, (O = (x, y) → x = r ∧ y = r)) → s = 2 * r :=
by
  sorry

end square_side_length_tangent_circle_l756_75659


namespace find_y_minus_x_l756_75671

theorem find_y_minus_x (x y : ℕ) (hx : x + y = 540) (hxy : (x : ℚ) / (y : ℚ) = 7 / 8) : y - x = 36 :=
by
  sorry

end find_y_minus_x_l756_75671


namespace geometric_sequence_general_term_l756_75657

theorem geometric_sequence_general_term :
  ∀ (n : ℕ), (n > 0) →
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ (k : ℕ), k > 0 → a (k+1) = 2 * a k) ∧ a n = 2^(n-1)) :=
by
  sorry

end geometric_sequence_general_term_l756_75657


namespace number_of_tables_l756_75669

noncomputable def stools_per_table : ℕ := 7
noncomputable def legs_per_stool : ℕ := 4
noncomputable def legs_per_table : ℕ := 5
noncomputable def total_legs : ℕ := 658

theorem number_of_tables : 
  ∃ t : ℕ, 
  (∃ s : ℕ, s = stools_per_table * t ∧ legs_per_stool * s + legs_per_table * t = total_legs) ∧ t = 20 :=
by {
  sorry
}

end number_of_tables_l756_75669


namespace range_of_m_l756_75633

theorem range_of_m (m : ℝ) : 
  (m - 1 < 0 ∧ 4 * m - 3 > 0) → (3 / 4 < m ∧ m < 1) := 
by
  sorry

end range_of_m_l756_75633


namespace sum_tripled_numbers_l756_75672

theorem sum_tripled_numbers (x y S : ℝ) (h : x + y = S) : 
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 :=
by
  sorry

end sum_tripled_numbers_l756_75672


namespace eval_expr_l756_75694

theorem eval_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x^(2 * y) * y^(3 * x) / (y^(2 * y) * x^(3 * x))) = x^(2 * y - 3 * x) * y^(3 * x - 2 * y) :=
by
  sorry

end eval_expr_l756_75694


namespace relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l756_75668

variable (x y : ℝ)

-- Assume the initial fuel and consumption rate
def initial_fuel : ℝ := 48
def consumption_rate : ℝ := 0.6

-- Define the fuel consumption equation
def fuel_equation (distance : ℝ) : ℝ := -consumption_rate * distance + initial_fuel

-- Theorem proving the fuel equation satisfies the specific conditions
theorem relationship_between_y_and_x :
  ∀ (x : ℝ), y = fuel_equation x :=
by
  sorry

-- Theorem proving the fuel remaining after traveling 35 kilometers
theorem fuel_remaining_after_35_kilometers :
  fuel_equation 35 = 27 :=
by
  sorry

-- Theorem proving the maximum distance the car can travel without refueling
theorem max_distance_without_refueling :
  ∃ (x : ℝ), fuel_equation x = 0 ∧ x = 80 :=
by
  sorry

end relationship_between_y_and_x_fuel_remaining_after_35_kilometers_max_distance_without_refueling_l756_75668


namespace derivative_at_pi_over_4_l756_75677

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem derivative_at_pi_over_4 : (deriv f) (Real.pi / 4) = 0 := 
by
  sorry

end derivative_at_pi_over_4_l756_75677


namespace price_of_each_movie_in_first_box_l756_75691

theorem price_of_each_movie_in_first_box (P : ℝ) (total_movies_box1 : ℕ) (total_movies_box2 : ℕ) (price_per_movie_box2 : ℝ) (average_price : ℝ) (total_movies : ℕ) :
  total_movies_box1 = 10 →
  total_movies_box2 = 5 →
  price_per_movie_box2 = 5 →
  average_price = 3 →
  total_movies = 15 →
  10 * P + 5 * price_per_movie_box2 = average_price * total_movies →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end price_of_each_movie_in_first_box_l756_75691


namespace distance_left_to_drive_l756_75667

theorem distance_left_to_drive (total_distance : ℕ) (distance_driven : ℕ) 
  (h1 : total_distance = 78) (h2 : distance_driven = 32) : 
  total_distance - distance_driven = 46 := by
  sorry

end distance_left_to_drive_l756_75667


namespace smallest_m_n_sum_l756_75623

theorem smallest_m_n_sum (m n : ℕ) (hmn : m > n) (div_condition : 4900 ∣ (2023 ^ m - 2023 ^ n)) : m + n = 24 :=
by
  sorry

end smallest_m_n_sum_l756_75623


namespace select_team_of_5_l756_75646

def boys : ℕ := 7
def girls : ℕ := 9
def total_students : ℕ := boys + girls

theorem select_team_of_5 (n : ℕ := total_students) (k : ℕ := 5) :
  (Nat.choose n k) = 4368 :=
by
  sorry

end select_team_of_5_l756_75646


namespace evaluate_expression_l756_75660

noncomputable def a : ℝ := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def b : ℝ := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def c : ℝ := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def d : ℝ := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6

theorem evaluate_expression : (1/a + 1/b + 1/c + 1/d)^2 = 952576 / 70225 := by
  sorry

end evaluate_expression_l756_75660


namespace arithmetic_mean_of_arithmetic_progression_l756_75644

variable (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ)

/-- General term of an arithmetic progression -/
def arithmetic_progression (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem arithmetic_mean_of_arithmetic_progression (k p : ℕ) (hk : 1 < k) :
  a k = (a (k - p) + a (k + p)) / 2 := by
  sorry

end arithmetic_mean_of_arithmetic_progression_l756_75644


namespace binkie_gemstones_l756_75681

noncomputable def gemstones_solution : ℕ :=
sorry

theorem binkie_gemstones : ∀ (Binkie Frankie Spaatz Whiskers Snowball : ℕ),
  Spaatz = 1 ∧
  Whiskers = Spaatz + 3 ∧
  Snowball = 2 * Whiskers ∧ 
  Snowball % 2 = 0 ∧
  Whiskers % 2 = 0 ∧
  Spaatz = (1 / 2 * Frankie) - 2 ∧
  Binkie = 4 * Frankie ∧
  Binkie + Frankie + Spaatz + Whiskers + Snowball <= 50 →
  Binkie = 24 :=
sorry

end binkie_gemstones_l756_75681


namespace amy_remaining_money_l756_75652

-- Define initial amount and purchases
def initial_amount : ℝ := 15
def stuffed_toy_cost : ℝ := 2
def hot_dog_cost : ℝ := 3.5
def candy_apple_cost : ℝ := 1.5
def discount_rate : ℝ := 0.5

-- Define the discounted hot_dog_cost
def discounted_hot_dog_cost := hot_dog_cost * discount_rate

-- Define the total spent
def total_spent := stuffed_toy_cost + discounted_hot_dog_cost + candy_apple_cost

-- Define the remaining amount
def remaining_amount := initial_amount - total_spent

theorem amy_remaining_money : remaining_amount = 9.75 := by
  sorry

end amy_remaining_money_l756_75652


namespace cube_root_simplification_l756_75678

theorem cube_root_simplification (N : ℝ) (h : N > 1) : (N^3)^(1/3) * ((N^5)^(1/3) * ((N^3)^(1/3)))^(1/3) = N^(5/3) :=
by sorry

end cube_root_simplification_l756_75678


namespace degrees_to_radians_l756_75613

theorem degrees_to_radians (degrees : ℝ) (pi : ℝ) : 
  degrees * (pi / 180) = pi / 15 ↔ degrees = 12 :=
by 
  sorry

end degrees_to_radians_l756_75613


namespace time_to_paint_one_room_l756_75688

variables (rooms_total rooms_painted : ℕ) (hours_to_paint_remaining : ℕ)

-- The conditions
def painter_conditions : Prop :=
  rooms_total = 10 ∧ rooms_painted = 8 ∧ hours_to_paint_remaining = 16

-- The goal is to find out the hours to paint one room
theorem time_to_paint_one_room (h : painter_conditions rooms_total rooms_painted hours_to_paint_remaining) : 
  let rooms_remaining := rooms_total - rooms_painted
  let hours_per_room := hours_to_paint_remaining / rooms_remaining
  hours_per_room = 8 :=
by sorry

end time_to_paint_one_room_l756_75688


namespace intersection_M_N_l756_75653

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} :=
  by sorry

end intersection_M_N_l756_75653


namespace volume_in_cubic_yards_l756_75634

-- Define the conditions given in the problem
def volume_in_cubic_feet : ℕ := 216
def cubic_feet_per_cubic_yard : ℕ := 27

-- Define the theorem that needs to be proven
theorem volume_in_cubic_yards :
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 8 :=
by
  sorry

end volume_in_cubic_yards_l756_75634


namespace jessica_exam_time_l756_75605

theorem jessica_exam_time (total_questions : ℕ) (answered_questions : ℕ) (used_minutes : ℕ)
    (total_time : ℕ) (remaining_time : ℕ) (rate : ℚ) :
    total_questions = 80 ∧ answered_questions = 16 ∧ used_minutes = 12 ∧ total_time = 60 ∧ rate = (answered_questions : ℚ) / used_minutes →
    remaining_time = total_time - used_minutes →
    remaining_time = 48 :=
by
  -- Proof will be filled in here
  sorry

end jessica_exam_time_l756_75605


namespace find_theta_2phi_l756_75647

-- Given
variables {θ φ : ℝ}
variables (hθ_acute : 0 < θ ∧ θ < π / 2)
variables (hφ_acute : 0 < φ ∧ φ < π / 2)
variables (h_tanθ : Real.tan θ = 3 / 11)
variables (h_sinφ : Real.sin φ = 1 / 3)

-- To prove
theorem find_theta_2phi : 
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.tan x = (21 + 6 * Real.sqrt 2) / (77 - 6 * Real.sqrt 2) ∧ x = θ + 2 * φ := 
sorry

end find_theta_2phi_l756_75647


namespace ending_number_divisible_by_3_l756_75692

theorem ending_number_divisible_by_3 (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k < 13 → ∃ m, 10 ≤ m ∧ m ≤ n ∧ m % 3 = 0) →
  n = 48 :=
by
  intro h
  sorry

end ending_number_divisible_by_3_l756_75692


namespace proof_problem_l756_75655

theorem proof_problem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ∧ 
  (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ≤ Real.sqrt 2 :=
by
  sorry

end proof_problem_l756_75655


namespace dale_pasta_l756_75670

-- Define the conditions
def original_pasta : Nat := 2
def original_servings : Nat := 7
def final_servings : Nat := 35

-- Define the required calculation for the number of pounds of pasta needed
def required_pasta : Nat := 10

-- The theorem to prove
theorem dale_pasta : (final_servings / original_servings) * original_pasta = required_pasta := 
by
  sorry

end dale_pasta_l756_75670


namespace benches_required_l756_75662

theorem benches_required (students_base5 : ℕ := 312) (base_student_seating : ℕ := 5) (seats_per_bench : ℕ := 3) : ℕ :=
  let chairs := 3 * base_student_seating^2 + 1 * base_student_seating^1 + 2 * base_student_seating^0
  let benches := (chairs / seats_per_bench) + if (chairs % seats_per_bench > 0) then 1 else 0
  benches

example : benches_required = 28 :=
by sorry

end benches_required_l756_75662


namespace angle_C_of_quadrilateral_ABCD_l756_75676

theorem angle_C_of_quadrilateral_ABCD
  (AB CD BC AD : ℝ) (D : ℝ) (h_AB_CD : AB = CD) (h_BC_AD : BC = AD) (h_ang_D : D = 120) :
  ∃ C : ℝ, C = 60 :=
by
  sorry

end angle_C_of_quadrilateral_ABCD_l756_75676


namespace susan_strawberries_per_handful_l756_75689

-- Definitions of the given conditions
def total_picked := 75
def total_needed := 60
def strawberries_per_handful := 5

-- Derived conditions
def total_eaten := total_picked - total_needed
def number_of_handfuls := total_picked / strawberries_per_handful
def strawberries_eaten_per_handful := total_eaten / number_of_handfuls

-- The theorem we want to prove
theorem susan_strawberries_per_handful : strawberries_eaten_per_handful = 1 :=
by sorry

end susan_strawberries_per_handful_l756_75689


namespace pirate_treasure_probability_l756_75610

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_trap_no_treasure := 1 / 10
  let p_notreasure_notrap := 7 / 10
  let combinatorial_factor := Nat.choose 8 4
  let probability := (combinatorial_factor * (p_treasure ^ 4) * (p_notreasure_notrap ^ 4))
  probability = 33614 / 1250000 :=
by
  sorry

end pirate_treasure_probability_l756_75610


namespace sum_of_numbers_l756_75642

theorem sum_of_numbers {a b c : ℝ} (h1 : b = 7) (h2 : (a + b + c) / 3 = a + 8) (h3 : (a + b + c) / 3 = c - 20) : a + b + c = 57 :=
sorry

end sum_of_numbers_l756_75642


namespace ratio_of_milk_to_water_l756_75641

namespace MixtureProblem

def initial_milk (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (milk_ratio * total_volume) / (milk_ratio + water_ratio)

def initial_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) : ℕ :=
  (water_ratio * total_volume) / (milk_ratio + water_ratio)

theorem ratio_of_milk_to_water (total_volume : ℕ) (milk_ratio : ℕ) (water_ratio : ℕ) (added_water : ℕ) :
  milk_ratio = 4 → water_ratio = 1 → total_volume = 45 → added_water = 21 → 
  (initial_milk total_volume milk_ratio water_ratio) = 36 →
  (initial_water total_volume milk_ratio water_ratio + added_water) = 30 →
  (36 / 30 : ℚ) = 6 / 5 :=
by
  intros
  sorry

end MixtureProblem

end ratio_of_milk_to_water_l756_75641


namespace greatest_prime_factor_180_l756_75673

noncomputable def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

theorem greatest_prime_factor_180 : 
  ∃ p : ℕ, is_prime p ∧ p ∣ 180 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 180 → q ≤ p :=
  sorry

end greatest_prime_factor_180_l756_75673


namespace weng_total_earnings_l756_75636

noncomputable def weng_earnings_usd : ℝ :=
  let usd_per_hr_job1 : ℝ := 12
  let eur_per_hr_job2 : ℝ := 13
  let gbp_per_hr_job3 : ℝ := 9
  let hr_job1 : ℝ := 2 + 15 / 60
  let hr_job2 : ℝ := 1 + 40 / 60
  let hr_job3 : ℝ := 3 + 10 / 60
  let usd_to_eur : ℝ := 0.85
  let usd_to_gbp : ℝ := 0.76
  let eur_to_usd : ℝ := 1.18
  let gbp_to_usd : ℝ := 1.32
  let earnings_job1 : ℝ := usd_per_hr_job1 * hr_job1
  let earnings_job2_eur : ℝ := eur_per_hr_job2 * hr_job2
  let earnings_job2_usd : ℝ := earnings_job2_eur * eur_to_usd
  let earnings_job3_gbp : ℝ := gbp_per_hr_job3 * hr_job3
  let earnings_job3_usd : ℝ := earnings_job3_gbp * gbp_to_usd
  earnings_job1 + earnings_job2_usd + earnings_job3_usd

theorem weng_total_earnings : weng_earnings_usd = 90.19 :=
by
  sorry

end weng_total_earnings_l756_75636


namespace total_cost_is_100_l756_75619

def shirts : ℕ := 10
def pants : ℕ := shirts / 2
def cost_shirt : ℕ := 6
def cost_pant : ℕ := 8

theorem total_cost_is_100 :
  shirts * cost_shirt + pants * cost_pant = 100 := by
  sorry

end total_cost_is_100_l756_75619


namespace ab_proof_l756_75658

theorem ab_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 90 < a + b) (h4 : a + b < 99) 
  (h5 : 0.9 < (a : ℝ) / b) (h6 : (a : ℝ) / b < 0.91) : a * b = 2346 :=
sorry

end ab_proof_l756_75658


namespace max_value_exponent_l756_75685

theorem max_value_exponent {a b : ℝ} (h : 0 < b ∧ b < a ∧ a < 1) :
  max (max (a^b) (b^a)) (max (a^a) (b^b)) = a^b :=
sorry

end max_value_exponent_l756_75685


namespace range_of_m_l756_75637

open Real

def f (x m: ℝ) : ℝ := x^2 - 2 * x + m^2 + 3 * m - 3

def p (m: ℝ) : Prop := ∃ x, f x m < 0

def q (m: ℝ) : Prop := (5 * m - 1 > 0) ∧ (m - 2 > 0)

theorem range_of_m (m : ℝ) : ¬ (p m ∨ q m) ∧ ¬ (p m ∧ q m) → (m ≤ -4 ∨ m ≥ 2) :=
by
  sorry

end range_of_m_l756_75637


namespace total_hours_worked_l756_75696

theorem total_hours_worked :
  (∃ (hours_per_day : ℕ) (days : ℕ), hours_per_day = 3 ∧ days = 6) →
  (∃ (total_hours : ℕ), total_hours = 18) :=
by
  intros
  sorry

end total_hours_worked_l756_75696


namespace smallest_sum_l756_75643

theorem smallest_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_neq : x ≠ y)
  (h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 18) : x + y = 75 :=
by
  sorry

end smallest_sum_l756_75643


namespace range_of_f_l756_75698

theorem range_of_f (x : ℝ) (h : x ∈ Set.Icc (-3 : ℝ) 3) : 
  ∃ y, y ∈ Set.Icc (0 : ℝ) 25 ∧ ∀ z, z = (x^2 - 4*x + 4) → y = z :=
sorry

end range_of_f_l756_75698


namespace odd_function_condition_l756_75682

noncomputable def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ (a = 0 ∧ b = 0) :=
by
  sorry

end odd_function_condition_l756_75682


namespace polynomial_coefficient_a5_l756_75666

theorem polynomial_coefficient_a5 : 
  (∃ (a0 a1 a2 a3 a4 a5 a6 : ℝ), 
    (∀ (x : ℝ), ((2 * x - 1)^5 * (x + 2) = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6)) ∧ 
    a5 = 176) := sorry

end polynomial_coefficient_a5_l756_75666


namespace find_s_l756_75674

theorem find_s : ∃ s : ℚ, (∀ x : ℚ, (3 * x^2 - 8 * x + 9) * (5 * x^2 + s * x + 15) = 15 * x^4 - 71 * x^3 + 174 * x^2 - 215 * x + 135) ∧ s = -95 / 9 := sorry

end find_s_l756_75674


namespace area_of_three_layer_cover_l756_75614

-- Define the hall dimensions
def hall_width : ℕ := 10
def hall_height : ℕ := 10

-- Define the dimensions of the carpets
def carpet1_width : ℕ := 6
def carpet1_height : ℕ := 8
def carpet2_width : ℕ := 6
def carpet2_height : ℕ := 6
def carpet3_width : ℕ := 5
def carpet3_height : ℕ := 7

-- Theorem to prove area covered by the carpets in three layers
theorem area_of_three_layer_cover : 
  ∀ (w1 w2 w3 h1 h2 h3 : ℕ), w1 = carpet1_width → h1 = carpet1_height → w2 = carpet2_width → h2 = carpet2_height → w3 = carpet3_width → h3 = carpet3_height → 
  ∃ (area : ℕ), area = 6 :=
by
  intros w1 w2 w3 h1 h2 h3 hw1 hw2 hw3 hh1 hh2 hh3
  exact ⟨6, rfl⟩

#check area_of_three_layer_cover

end area_of_three_layer_cover_l756_75614


namespace Karsyn_payment_l756_75661

def percentage : ℝ := 20
def initial_price : ℝ := 600

theorem Karsyn_payment : (percentage / 100) * initial_price = 120 :=
by 
  sorry

end Karsyn_payment_l756_75661


namespace quadratic_min_value_l756_75622

theorem quadratic_min_value :
  ∃ x : ℝ, (∀ y : ℝ, y = x^2 - 4 * x + 7 → y ≥ 3) ∧ (x = 2 → (x^2 - 4 * x + 7 = 3)) :=
by
  sorry

end quadratic_min_value_l756_75622


namespace minimal_bananas_l756_75626

noncomputable def total_min_bananas : ℕ :=
  let b1 := 72
  let b2 := 72
  let b3 := 216
  let b4 := 72
  b1 + b2 + b3 + b4

theorem minimal_bananas (total_bananas : ℕ) (ratio1 ratio2 ratio3 ratio4 : ℕ) 
  (b1 b2 b3 b4 : ℕ) 
  (h_ratio : ratio1 = 4 ∧ ratio2 = 3 ∧ ratio3 = 2 ∧ ratio4 = 1) 
  (h_div_constraints : ∀ n m : ℕ, (n % m = 0 ∨ m % n = 0) ∧ n ≥ ratio1 * ratio2 * ratio3 * ratio4) 
  (h_bananas : b1 = 72 ∧ b2 = 72 ∧ b3 = 216 ∧ b4 = 72 ∧ 
              4 * (b1 / 2 + b2 / 6 + b3 / 9 + 7 * b4 / 72) = 3 * (b1 / 6 + b2 / 3 + b3 / 9 + 7 * b4 / 72) ∧ 
              2 * (b1 / 6 + b2 / 6 + b3 / 6 + 7 * b4 / 72) = (b1 / 6 + b2 / 6 + b3 / 9 + b4 / 8)) : 
  total_bananas = 432 := by
  sorry

end minimal_bananas_l756_75626
