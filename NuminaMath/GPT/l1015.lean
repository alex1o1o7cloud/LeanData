import Mathlib

namespace difference_of_numbers_l1015_101529

theorem difference_of_numbers (L S : ℕ) (h1 : L = 1620) (h2 : L = 6 * S + 15) : L - S = 1353 :=
by
  sorry

end difference_of_numbers_l1015_101529


namespace fraction_of_phones_l1015_101523

-- The total number of valid 8-digit phone numbers (b)
def valid_phone_numbers_total : ℕ := 5 * 10^7

-- The number of valid phone numbers that begin with 5 and end with 2 (a)
def valid_phone_numbers_special : ℕ := 10^6

-- The fraction of phone numbers that begin with 5 and end with 2
def fraction_phone_numbers_special : ℚ := valid_phone_numbers_special / valid_phone_numbers_total

-- Prove that the fraction of such phone numbers is 1/50
theorem fraction_of_phones : fraction_phone_numbers_special = 1 / 50 := by
  sorry

end fraction_of_phones_l1015_101523


namespace area_of_segment_solution_max_sector_angle_solution_l1015_101557
open Real

noncomputable def area_of_segment (α R : ℝ) : ℝ :=
  let l := (R * α)
  let sector := 0.5 * R * l
  let triangle := 0.5 * R^2 * sin α
  sector - triangle

theorem area_of_segment_solution : area_of_segment (π / 3) 10 = 50 * ((π / 3) - (sqrt 3 / 2)) :=
by sorry

noncomputable def max_sector_angle (c : ℝ) (hc : c > 0) : ℝ :=
  2

theorem max_sector_angle_solution (c : ℝ) (hc : c > 0) : max_sector_angle c hc = 2 :=
by sorry

end area_of_segment_solution_max_sector_angle_solution_l1015_101557


namespace smallest_positive_omega_l1015_101503

theorem smallest_positive_omega (f g : ℝ → ℝ) (ω : ℝ) 
  (hf : ∀ x, f x = Real.cos (ω * x)) 
  (hg : ∀ x, g x = Real.sin (ω * x - π / 4)) 
  (heq : ∀ x, f (x - π / 2) = g x) :
  ω = 3 / 2 :=
sorry

end smallest_positive_omega_l1015_101503


namespace M_in_fourth_quadrant_l1015_101518

-- Define the conditions
variables (a b : ℝ)

/-- Condition that point A(a, 3) and B(2, b) are symmetric with respect to the x-axis -/
def symmetric_points : Prop :=
  a = 2 ∧ 3 = -b

-- Define the point M and quadrant check
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- The theorem stating that if A(a, 3) and B(2, b) are symmetric wrt x-axis, M is in the fourth quadrant
theorem M_in_fourth_quadrant (a b : ℝ) (h : symmetric_points a b) : in_fourth_quadrant a b :=
by {
  sorry
}

end M_in_fourth_quadrant_l1015_101518


namespace cos_2015_eq_neg_m_l1015_101515

variable (m : ℝ)

-- Given condition
axiom sin_55_eq_m : Real.sin (55 * Real.pi / 180) = m

-- The proof problem
theorem cos_2015_eq_neg_m : Real.cos (2015 * Real.pi / 180) = -m :=
by
  sorry

end cos_2015_eq_neg_m_l1015_101515


namespace distance_from_M_to_x_axis_l1015_101521

-- Define the point M and its coordinates.
def point_M : ℤ × ℤ := (-9, 12)

-- Define the distance to the x-axis is simply the absolute value of the y-coordinate.
def distance_to_x_axis (p : ℤ × ℤ) : ℤ := Int.natAbs p.snd

-- Theorem stating the distance from point M to the x-axis is 12.
theorem distance_from_M_to_x_axis : distance_to_x_axis point_M = 12 := by
  sorry

end distance_from_M_to_x_axis_l1015_101521


namespace system_solutions_l1015_101533

theorem system_solutions (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = -1) : 
  b = -22 :=
by 
  sorry

end system_solutions_l1015_101533


namespace jimmy_yellow_marbles_correct_l1015_101569

def lorin_black_marbles : ℕ := 4
def alex_black_marbles : ℕ := 2 * lorin_black_marbles
def alex_total_marbles : ℕ := 19
def alex_yellow_marbles : ℕ := alex_total_marbles - alex_black_marbles
def jimmy_yellow_marbles : ℕ := 2 * alex_yellow_marbles

theorem jimmy_yellow_marbles_correct : jimmy_yellow_marbles = 22 := by
  sorry

end jimmy_yellow_marbles_correct_l1015_101569


namespace population_net_increase_period_l1015_101522

def period_in_hours (birth_rate : ℕ) (death_rate : ℕ) (net_increase : ℕ) : ℕ :=
  let net_rate_per_second := (birth_rate / 2) - (death_rate / 2)
  let period_in_seconds := net_increase / net_rate_per_second
  period_in_seconds / 3600

theorem population_net_increase_period :
  period_in_hours 10 2 345600 = 24 :=
by
  unfold period_in_hours
  sorry

end population_net_increase_period_l1015_101522


namespace speed_of_first_train_l1015_101549

noncomputable def length_of_first_train : ℝ := 280
noncomputable def speed_of_second_train_kmph : ℝ := 80
noncomputable def length_of_second_train : ℝ := 220.04
noncomputable def time_to_cross : ℝ := 9

noncomputable def relative_speed_mps := (length_of_first_train + length_of_second_train) / time_to_cross

noncomputable def relative_speed_kmph := relative_speed_mps * (3600 / 1000)

theorem speed_of_first_train :
  (relative_speed_kmph - speed_of_second_train_kmph) = 120.016 :=
by
  sorry

end speed_of_first_train_l1015_101549


namespace quadratic_expression_sum_l1015_101554

theorem quadratic_expression_sum :
  ∃ a h k : ℝ, (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
sorry

end quadratic_expression_sum_l1015_101554


namespace friends_prove_l1015_101559

theorem friends_prove (a b c d : ℕ) (h1 : 3^a * 7^b = 3^c * 7^d) (h2 : 3^a * 7^b = 21) :
  (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by {
  sorry
}

end friends_prove_l1015_101559


namespace product_as_difference_of_squares_l1015_101577

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ( (a + b) / 2 )^2 - ( (a - b) / 2 )^2 :=
by
  sorry

end product_as_difference_of_squares_l1015_101577


namespace bunch_of_bananas_cost_l1015_101548

def cost_of_bananas (A : ℝ) : ℝ := 5 - A

theorem bunch_of_bananas_cost (A B T : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = T) : B = cost_of_bananas A :=
by
  sorry

end bunch_of_bananas_cost_l1015_101548


namespace length_of_first_train_l1015_101583

theorem length_of_first_train
  (speed_first : ℕ)
  (speed_second : ℕ)
  (length_second : ℕ)
  (distance_between : ℕ)
  (time_to_cross : ℕ)
  (h1 : speed_first = 10)
  (h2 : speed_second = 15)
  (h3 : length_second = 150)
  (h4 : distance_between = 50)
  (h5 : time_to_cross = 60) :
  ∃ L : ℕ, L = 100 :=
by
  sorry

end length_of_first_train_l1015_101583


namespace Jenine_pencil_count_l1015_101556

theorem Jenine_pencil_count
  (sharpenings_per_pencil : ℕ)
  (hours_per_sharpening : ℝ)
  (total_hours_needed : ℝ)
  (cost_per_pencil : ℝ)
  (budget : ℝ)
  (already_has_pencils : ℕ) :
  sharpenings_per_pencil = 5 →
  hours_per_sharpening = 1.5 →
  total_hours_needed = 105 →
  cost_per_pencil = 2 →
  budget = 8 →
  already_has_pencils = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Jenine_pencil_count_l1015_101556


namespace quadratic_root_one_is_minus_one_l1015_101520

theorem quadratic_root_one_is_minus_one (m : ℝ) (h : ∃ x : ℝ, x = -1 ∧ m * x^2 + x - m^2 + 1 = 0) : m = 1 :=
by
  sorry

end quadratic_root_one_is_minus_one_l1015_101520


namespace rate_per_kg_for_grapes_l1015_101525

theorem rate_per_kg_for_grapes (G : ℝ) (h : 9 * G + 9 * 55 = 1125) : G = 70 :=
by
  -- sorry to skip the proof
  sorry

end rate_per_kg_for_grapes_l1015_101525


namespace visits_exactly_two_friends_l1015_101514

theorem visits_exactly_two_friends (a_visits b_visits c_visits vacation_period : ℕ) (full_period days : ℕ)
(h_a : a_visits = 4)
(h_b : b_visits = 5)
(h_c : c_visits = 6)
(h_vacation : vacation_period = 30)
(h_full_period : full_period = Nat.lcm (Nat.lcm a_visits b_visits) c_visits)
(h_days : days = 360)
(h_start_vacation : ∀ n, ∃ k, n = k * vacation_period + 30):
  ∃ n, n = 24 :=
by {
  sorry
}

end visits_exactly_two_friends_l1015_101514


namespace geometric_sequence_S_n_l1015_101546

-- Definitions related to the sequence
def a_n (n : ℕ) : ℕ := sorry  -- Placeholder for the actual sequence

-- Sum of the first n terms
def S_n (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms

-- Given conditions
axiom a1 : a_n 1 = 1
axiom Sn_eq_2an_plus1 : ∀ (n : ℕ), S_n n = 2 * a_n (n + 1)

-- Theorem to be proved
theorem geometric_sequence_S_n 
    (n : ℕ) (h : n > 1) 
    : S_n n = (3/2)^(n-1) := 
by 
  sorry

end geometric_sequence_S_n_l1015_101546


namespace find_a_l1015_101550

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^4 - 18 * x^3 + ((86 : ℝ)) * x^2 + 200 * x - 1984

-- Define the condition and statement
theorem find_a (α β γ δ : ℝ) (hαβγδ : α * β * γ * δ = -1984)
  (hαβ : α * β = -32) (hγδ : γ * δ = 62) :
  (∀ a : ℝ, a = 86) :=
  sorry

end find_a_l1015_101550


namespace total_sticks_used_l1015_101505

-- Definitions based on the conditions
def hexagons : Nat := 800
def sticks_for_first_hexagon : Nat := 6
def sticks_per_additional_hexagon : Nat := 5

-- The theorem to prove
theorem total_sticks_used :
  sticks_for_first_hexagon + (hexagons - 1) * sticks_per_additional_hexagon = 4001 := by
  sorry

end total_sticks_used_l1015_101505


namespace unique_positive_integer_k_for_rational_solutions_l1015_101524

theorem unique_positive_integer_k_for_rational_solutions :
  ∃ (k : ℕ), (k > 0) ∧ (∀ (x : ℤ), x * x = 256 - 4 * k * k → x = 8) ∧ (k = 7) :=
by
  sorry

end unique_positive_integer_k_for_rational_solutions_l1015_101524


namespace solution_exists_l1015_101599

-- Defining the variables x and y
variables (x y : ℝ)

-- Defining the conditions
def condition_1 : Prop :=
  3 * x ≥ 2 * y + 16

def condition_2 : Prop :=
  x^4 + 2 * (x^2) * (y^2) + y^4 + 25 - 26 * (x^2) - 26 * (y^2) = 72 * x * y

-- Stating the theorem that (6, 1) satisfies the conditions
theorem solution_exists : condition_1 6 1 ∧ condition_2 6 1 :=
by
  -- Convert conditions into expressions
  have h1 : condition_1 6 1 := by sorry
  have h2 : condition_2 6 1 := by sorry
  -- Conjunction of both conditions is satisfied
  exact ⟨h1, h2⟩

end solution_exists_l1015_101599


namespace simplify_expression_l1015_101510

theorem simplify_expression (x : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 9 = 45 * x + 27 :=
by
  sorry

end simplify_expression_l1015_101510


namespace count_prime_boring_lt_10000_l1015_101573

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_boring (n : ℕ) : Prop := 
  let digits := n.digits 10
  match digits with
  | [] => false
  | (d::ds) => ds.all (fun x => x = d)

theorem count_prime_boring_lt_10000 : 
  ∃! n, is_prime n ∧ is_boring n ∧ n < 10000 := 
by 
  sorry

end count_prime_boring_lt_10000_l1015_101573


namespace find_N_l1015_101516

theorem find_N (N : ℕ) (h₁ : ∃ (d₁ d₂ : ℕ), d₁ + d₂ = 3333 ∧ N = max d₁ d₂ ∧ (max d₁ d₂) / (min d₁ d₂) = 2) : 
  N = 2222 := sorry

end find_N_l1015_101516


namespace value_of_expression_l1015_101530

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  (a + b + c + d).sqrt + (a^2 - 2*a + 3 - b).sqrt - (b - c^2 + 4*c - 8).sqrt = 3

theorem value_of_expression (a b c d : ℝ) (h : proof_problem a b c d) : a - b + c - d = -7 :=
sorry

end value_of_expression_l1015_101530


namespace workshop_total_number_of_workers_l1015_101579

theorem workshop_total_number_of_workers
  (average_salary_all : ℝ)
  (average_salary_technicians : ℝ)
  (average_salary_non_technicians : ℝ)
  (num_technicians : ℕ)
  (total_salary_all : ℝ -> ℝ)
  (total_salary_technicians : ℕ -> ℝ)
  (total_salary_non_technicians : ℕ -> ℝ -> ℝ)
  (h1 : average_salary_all = 9000)
  (h2 : average_salary_technicians = 12000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : ∀ W, total_salary_all W = average_salary_all * W )
  (h6 : ∀ n, total_salary_technicians n = n * average_salary_technicians )
  (h7 : ∀ n W, total_salary_non_technicians n W = (W - n) * average_salary_non_technicians)
  (h8 : ∀ W, total_salary_all W = total_salary_technicians num_technicians + total_salary_non_technicians num_technicians W) :
  ∃ W, W = 14 :=
by
  sorry

end workshop_total_number_of_workers_l1015_101579


namespace total_cost_of_gas_l1015_101567

theorem total_cost_of_gas :
  ∃ x : ℚ, (4 * (x / 4) - 4 * (x / 7) = 40) ∧ x = 280 / 3 :=
by
  sorry

end total_cost_of_gas_l1015_101567


namespace find_d_l1015_101535

-- Define the proportional condition
def in_proportion (a b c d : ℕ) : Prop := a * d = b * c

-- Given values as parameters
variables {a b c d : ℕ}

-- Theorem to be proven
theorem find_d (h : in_proportion a b c d) (ha : a = 1) (hb : b = 2) (hc : c = 3) : d = 6 :=
sorry

end find_d_l1015_101535


namespace total_time_per_week_l1015_101586

noncomputable def meditating_time_per_day : ℝ := 1
noncomputable def reading_time_per_day : ℝ := 2 * meditating_time_per_day
noncomputable def exercising_time_per_day : ℝ := 0.5 * meditating_time_per_day
noncomputable def practicing_time_per_day : ℝ := (1/3) * reading_time_per_day

noncomputable def total_time_per_day : ℝ :=
  meditating_time_per_day + reading_time_per_day + exercising_time_per_day + practicing_time_per_day

theorem total_time_per_week :
  total_time_per_day * 7 = 29.17 := by
  sorry

end total_time_per_week_l1015_101586


namespace tensor_identity_l1015_101561

namespace tensor_problem

def otimes (x y : ℝ) : ℝ := x^2 + y

theorem tensor_identity (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a :=
by sorry

end tensor_problem

end tensor_identity_l1015_101561


namespace roger_shelves_l1015_101509

theorem roger_shelves (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : 
  total_books = 24 → 
  books_taken = 3 → 
  books_per_shelf = 4 → 
  Nat.ceil ((total_books - books_taken) / books_per_shelf) = 6 :=
by
  intros h_total h_taken h_per_shelf
  rw [h_total, h_taken, h_per_shelf]
  sorry

end roger_shelves_l1015_101509


namespace distribution_ways_l1015_101570

def number_of_ways_to_distribute_problems : ℕ :=
  let friends := 10
  let problems := 7
  let max_receivers := 3
  let ways_to_choose_friends := Nat.choose friends max_receivers
  let ways_to_distribute_problems := max_receivers ^ problems
  ways_to_choose_friends * ways_to_distribute_problems

theorem distribution_ways :
  number_of_ways_to_distribute_problems = 262440 :=
by
  -- Proof is omitted
  sorry

end distribution_ways_l1015_101570


namespace rectangle_perimeter_l1015_101519

theorem rectangle_perimeter 
  (w : ℝ) (l : ℝ) (hw : w = Real.sqrt 3) (hl : l = Real.sqrt 6) : 
  2 * (w + l) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := 
by 
  sorry

end rectangle_perimeter_l1015_101519


namespace janet_more_siblings_than_carlos_l1015_101517

-- Define the initial conditions
def masud_siblings := 60
def carlos_siblings := (3 / 4) * masud_siblings
def janet_siblings := 4 * masud_siblings - 60

-- The statement to be proved
theorem janet_more_siblings_than_carlos : janet_siblings - carlos_siblings = 135 :=
by
  sorry

end janet_more_siblings_than_carlos_l1015_101517


namespace work_together_l1015_101540

variable (W : ℝ) -- 'W' denotes the total work
variable (a_days b_days c_days : ℝ)

-- Conditions provided in the problem
axiom a_work : a_days = 18
axiom b_work : b_days = 6
axiom c_work : c_days = 12

-- The statement to be proved
theorem work_together :
  (W / a_days + W / b_days + W / c_days) * (36 / 11) = W := by
  sorry

end work_together_l1015_101540


namespace fill_time_with_leak_is_correct_l1015_101584

-- Define the conditions
def time_to_fill_without_leak := 8
def time_to_empty_with_leak := 24

-- Define the rates
def fill_rate := 1 / time_to_fill_without_leak
def leak_rate := 1 / time_to_empty_with_leak
def effective_fill_rate := fill_rate - leak_rate

-- Prove the time to fill with leak
def time_to_fill_with_leak := 1 / effective_fill_rate

-- The theorem to prove that the time is 12 hours
theorem fill_time_with_leak_is_correct :
  time_to_fill_with_leak = 12 := by
  simp [time_to_fill_without_leak, time_to_empty_with_leak, fill_rate, leak_rate, effective_fill_rate, time_to_fill_with_leak]
  sorry

end fill_time_with_leak_is_correct_l1015_101584


namespace Yuna_place_l1015_101563

theorem Yuna_place (Eunji_place : ℕ) (distance : ℕ) (Yuna_place : ℕ) 
  (h1 : Eunji_place = 100) 
  (h2 : distance = 11) 
  (h3 : Yuna_place = Eunji_place + distance) : 
  Yuna_place = 111 := 
sorry

end Yuna_place_l1015_101563


namespace train_ride_cost_difference_l1015_101596

-- Definitions based on the conditions
def bus_ride_cost : ℝ := 1.40
def total_cost : ℝ := 9.65

-- Lemma to prove the mathematical question
theorem train_ride_cost_difference :
  ∃ T : ℝ, T + bus_ride_cost = total_cost ∧ (T - bus_ride_cost) = 6.85 :=
by
  sorry

end train_ride_cost_difference_l1015_101596


namespace slope_range_l1015_101526

theorem slope_range (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ k : ℝ, k = (y + 2) / (x + 1) ∧ k ∈ Set.Ici (3 / 4) :=
sorry

end slope_range_l1015_101526


namespace num_solutions_eq_4_l1015_101527

theorem num_solutions_eq_4 (θ : ℝ) (h : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  ∃ n : ℕ, n = 4 ∧ (2 + 4 * Real.cos θ - 6 * Real.sin (2 * θ) + 3 * Real.tan θ = 0) :=
sorry

end num_solutions_eq_4_l1015_101527


namespace find_multiplier_l1015_101532

theorem find_multiplier (x y: ℤ) (h1: x = 127)
  (h2: x * y - 152 = 102): y = 2 :=
by
  sorry

end find_multiplier_l1015_101532


namespace factorize_x4_plus_81_l1015_101553

noncomputable def factorize_poly (x : ℝ) : (ℝ × ℝ) :=
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  (p, q)

theorem factorize_x4_plus_81 : ∀ x : ℝ, (x^4 + 81) = (factorize_poly x).fst * (factorize_poly x).snd := by
  intro x
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  have h : x^4 + 81 = p * q
  { sorry }
  exact h

end factorize_x4_plus_81_l1015_101553


namespace complement_of_M_l1015_101513

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M :
  ∀ x, x ∈ U \ M ↔ x < -2 ∨ x > 2 :=
by
  sorry

end complement_of_M_l1015_101513


namespace total_value_of_goods_l1015_101547

theorem total_value_of_goods (V : ℝ)
  (h1 : 0 < V)
  (h2 : ∃ t, V - 600 = t ∧ 0.12 * t = 134.4) :
  V = 1720 := 
sorry

end total_value_of_goods_l1015_101547


namespace find_numbers_l1015_101528

theorem find_numbers (x y z u n : ℤ)
  (h1 : x + y + z + u = 36)
  (h2 : x + n = y - n)
  (h3 : x + n = z * n)
  (h4 : x + n = u / n) :
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
sorry

end find_numbers_l1015_101528


namespace cow_calf_ratio_l1015_101538

theorem cow_calf_ratio (cost_cow cost_calf : ℕ) (h_cow : cost_cow = 880) (h_calf : cost_calf = 110) :
  cost_cow / cost_calf = 8 :=
by {
  sorry
}

end cow_calf_ratio_l1015_101538


namespace seating_arrangements_exactly_two_adjacent_empty_seats_l1015_101507

theorem seating_arrangements_exactly_two_adjacent_empty_seats : 
  (∃ (arrangements : ℕ), arrangements = 72) :=
by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_seats_l1015_101507


namespace pyramid_angles_sum_pi_over_four_l1015_101506

theorem pyramid_angles_sum_pi_over_four :
  ∃ (α β : ℝ), 
    α + β = Real.pi / 4 ∧ 
    α = Real.arctan ((Real.sqrt 17 - 3) / 4) ∧ 
    β = Real.pi / 4 - Real.arctan ((Real.sqrt 17 - 3) / 4) :=
by
  sorry

end pyramid_angles_sum_pi_over_four_l1015_101506


namespace speed_with_stream_l1015_101589

-- Define the given conditions
def V_m : ℝ := 7 -- Man's speed in still water (7 km/h)
def V_as : ℝ := 10 -- Man's speed against the stream (10 km/h)

-- Define the stream's speed as the difference
def V_s : ℝ := V_m - V_as

-- Define man's speed with the stream
def V_ws : ℝ := V_m + V_s

-- (Correct Answer): Prove the man's speed with the stream is 10 km/h
theorem speed_with_stream :
  V_ws = 10 := by
  -- Sorry for no proof required in this task
  sorry

end speed_with_stream_l1015_101589


namespace simplify_expression_l1015_101595

theorem simplify_expression (x y : ℝ) (P Q : ℝ) (hP : P = 2 * x + 3 * y) (hQ : Q = 3 * x + 2 * y) :
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (24 * x ^ 2 + 52 * x * y + 24 * y ^ 2) / (5 * x * y - 5 * y ^ 2) :=
by
  sorry

end simplify_expression_l1015_101595


namespace evaluate_g_l1015_101592

def g (a b c d : ℤ) : ℚ := (d * (c + 2 * a)) / (c + b)

theorem evaluate_g : g 4 (-1) (-8) 2 = 0 := 
by 
  sorry

end evaluate_g_l1015_101592


namespace polygon_sides_eq_eight_l1015_101504

theorem polygon_sides_eq_eight (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 :=
by
  intro h
  sorry

end polygon_sides_eq_eight_l1015_101504


namespace find_A_and_B_l1015_101582

theorem find_A_and_B (A : ℕ) (B : ℕ) (x y : ℕ) 
  (h1 : 1000 ≤ A ∧ A ≤ 9999) 
  (h2 : B = 10^5 * x + 10 * A + y) 
  (h3 : B = 21 * A)
  (h4 : x < 10) 
  (h5 : y < 10) : 
  A = 9091 ∧ B = 190911 :=
sorry

end find_A_and_B_l1015_101582


namespace initial_bees_l1015_101531

theorem initial_bees (B : ℕ) (h : B + 8 = 24) : B = 16 := 
by {
  sorry
}

end initial_bees_l1015_101531


namespace coin_change_count_ways_l1015_101512

theorem coin_change_count_ways :
  ∃ n : ℕ, (∀ q h : ℕ, (25 * q + 50 * h = 1500) ∧ q > 0 ∧ h > 0 → (1 ≤ h ∧ h < 30)) ∧ n = 29 :=
  sorry

end coin_change_count_ways_l1015_101512


namespace mowing_lawn_time_l1015_101544

def maryRate := 1 / 3
def tomRate := 1 / 4
def combinedRate := 7 / 12
def timeMaryAlone := 1
def lawnLeft := 1 - (timeMaryAlone * maryRate)

theorem mowing_lawn_time:
  (7 / 12) * (8 / 7) = (2 / 3) :=
by
  sorry

end mowing_lawn_time_l1015_101544


namespace area_increase_l1015_101511

theorem area_increase (r₁ r₂: ℝ) (A₁ A₂: ℝ) (side1 side2: ℝ) 
  (h1: side1 = 8) (h2: side2 = 12) (h3: r₁ = side2 / 2) (h4: r₂ = side1 / 2)
  (h5: A₁ = 2 * (1/2 * Real.pi * r₁ ^ 2) + 2 * (1/2 * Real.pi * r₂ ^ 2))
  (h6: A₂ = 4 * (Real.pi * r₂ ^ 2))
  (h7: A₁ = 52 * Real.pi) (h8: A₂ = 64 * Real.pi) :
  ((A₁ + A₂) - A₁) / A₁ * 100 = 123 :=
by
  sorry

end area_increase_l1015_101511


namespace find_integer_b_l1015_101565

theorem find_integer_b (z : ℝ) : ∃ b : ℝ, (z^2 - 6*z + 17 = (z - 3)^2 + b) ∧ b = 8 :=
by
  -- The proof would go here
  sorry

end find_integer_b_l1015_101565


namespace alex_loan_difference_l1015_101542

theorem alex_loan_difference :
  let P := (15000 : ℝ)
  let r1 := (0.08 : ℝ)
  let n := (2 : ℕ)
  let t := (12 : ℕ)
  let r2 := (0.09 : ℝ)
  
  -- Calculate the amount owed after 6 years with compound interest (first option)
  let A1_half := P * (1 + r1 / n)^(n * t / 2)
  let half_payment := A1_half / 2
  let remaining_balance := A1_half / 2
  let A1_final := remaining_balance * (1 + r1 / n)^(n * t / 2)
  
  -- Total payment for the first option
  let total1 := half_payment + A1_final
  
  -- Total payment for the second option (simple interest)
  let simple_interest := P * r2 * t
  let total2 := P + simple_interest
  
  -- Compute the positive difference
  let difference := abs (total1 - total2)
  
  difference = 24.59 :=
  by
  sorry

end alex_loan_difference_l1015_101542


namespace eval_expression_l1015_101552

theorem eval_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end eval_expression_l1015_101552


namespace binom_20_19_eq_20_l1015_101566

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end binom_20_19_eq_20_l1015_101566


namespace max_m_eq_half_l1015_101541

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 + m * x + m * Real.log x

theorem max_m_eq_half :
  ∃ m : ℝ, (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 ≤ 2) → (1 ≤ x2 ∧ x2 ≤ 2) → 
  x1 < x2 → |f x1 m - f x2 m| < x2^2 - x1^2)) ∧ m = 1/2 :=
sorry

end max_m_eq_half_l1015_101541


namespace vertex_at_fixed_point_l1015_101593

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 1

theorem vertex_at_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 2 :=
by
  sorry

end vertex_at_fixed_point_l1015_101593


namespace intersects_negative_half_axis_range_l1015_101537

noncomputable def f (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * m * x + 2 * m - 6

theorem intersects_negative_half_axis_range (m : ℝ) :
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) ↔ (∃ x : ℝ, f m x < 0) :=
sorry

end intersects_negative_half_axis_range_l1015_101537


namespace optimal_chalk_length_l1015_101575

theorem optimal_chalk_length (l : ℝ) (h₁: 10 ≤ l) (h₂: l ≤ 15) (h₃: l = 12) : l = 12 :=
by
  sorry

end optimal_chalk_length_l1015_101575


namespace original_price_l1015_101536

variable (P : ℝ)

theorem original_price (h : 560 = 1.05 * (0.72 * P)) : P = 740.46 := 
by
  sorry

end original_price_l1015_101536


namespace A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l1015_101578

def A : Set ℝ := { x | x^2 + x - 2 < 0 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

theorem A_union_B_when_m_neg_half : A ∪ B (-1/2) = { x | -2 < x ∧ x < 3/2 } :=
by
  sorry

theorem B_subset_A_implies_m_geq_zero (m : ℝ) : B m ⊆ A → 0 ≤ m :=
by
  sorry

end A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l1015_101578


namespace student_correct_ans_l1015_101590

theorem student_correct_ans (c w : ℕ) (h1 : c + w = 80) (h2 : 4 * c - w = 120) : c = 40 :=
by
  sorry

end student_correct_ans_l1015_101590


namespace min_value_l1015_101591

theorem min_value (a : ℝ) (h : a > 0) : a + 4 / a ≥ 4 :=
by sorry

end min_value_l1015_101591


namespace sin_value_l1015_101581

theorem sin_value (theta : ℝ) (h : Real.cos (3 * Real.pi / 14 - theta) = 1 / 3) : 
  Real.sin (2 * Real.pi / 7 + theta) = 1 / 3 :=
by
  -- Sorry replaces the actual proof which is not required for this task
  sorry

end sin_value_l1015_101581


namespace single_fraction_l1015_101555

theorem single_fraction (c : ℕ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 :=
by sorry

end single_fraction_l1015_101555


namespace problem1_problem2_l1015_101594

theorem problem1 : 24 - (-16) + (-25) - 15 = 0 :=
by
  sorry

theorem problem2 : (-81) + 2 * (1 / 4) * (4 / 9) / (-16) = -81 - (1 / 16) :=
by
  sorry

end problem1_problem2_l1015_101594


namespace hyperbola_foci_distance_l1015_101501

theorem hyperbola_foci_distance (c : ℝ) (h : c = Real.sqrt 2) : 
  let f1 := (c * Real.sqrt 2, c * Real.sqrt 2)
  let f2 := (-c * Real.sqrt 2, -c * Real.sqrt 2)
  Real.sqrt ((f2.1 - f1.1) ^ 2 + (f2.2 - f1.2) ^ 2) = 4 * Real.sqrt 2 := 
by
  sorry

end hyperbola_foci_distance_l1015_101501


namespace apple_distribution_ways_l1015_101539

-- Definitions based on conditions
def distribute_apples (a b c : ℕ) : Prop := a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3

-- Non-negative integer solutions to a' + b' + c' = 21
def num_solutions := Nat.choose 23 2

-- Theorem to prove
theorem apple_distribution_ways : distribute_apples 10 10 10 → num_solutions = 253 :=
by
  intros
  sorry

end apple_distribution_ways_l1015_101539


namespace cos_neg_13pi_over_4_l1015_101534

theorem cos_neg_13pi_over_4 : Real.cos (-13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_neg_13pi_over_4_l1015_101534


namespace find_n_l1015_101597

noncomputable def angles_periodic_mod_eq (n : ℤ) : Prop :=
  -100 < n ∧ n < 100 ∧ Real.tan (n * Real.pi / 180) = Real.tan (216 * Real.pi / 180)

theorem find_n (n : ℤ) (h : angles_periodic_mod_eq n) : n = 36 :=
  sorry

end find_n_l1015_101597


namespace length_of_shorter_leg_l1015_101560

variable (h x : ℝ)

theorem length_of_shorter_leg 
  (h_med : h / 2 = 5 * Real.sqrt 3) 
  (hypotenuse_relation : h = 2 * x) 
  (median_relation : h / 2 = h / 2) :
  x = 5 := by sorry

end length_of_shorter_leg_l1015_101560


namespace calc_expression1_calc_expression2_l1015_101508

-- Problem 1
theorem calc_expression1 (x y : ℝ) : (1/2 * x * y)^2 * 6 * x^2 * y = (3/2) * x^4 * y^3 := 
sorry

-- Problem 2
theorem calc_expression2 (a b : ℝ) : (2 * a + b)^2 = 4 * a^2 + 4 * a * b + b^2 := 
sorry

end calc_expression1_calc_expression2_l1015_101508


namespace arithmetic_sequence_common_difference_l1015_101551

variable {α : Type*} [AddGroup α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n, a (n + 1) = a n + (a 2 - a 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a3 : a 3 = -4) :
  a 3 - a 2 = -6 := 
sorry

end arithmetic_sequence_common_difference_l1015_101551


namespace right_angled_triangle_only_B_l1015_101574

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end right_angled_triangle_only_B_l1015_101574


namespace average_median_eq_l1015_101543

theorem average_median_eq (a b c : ℤ) (h1 : (a + b + c) / 3 = 4 * b)
  (h2 : a < b) (h3 : b < c) (h4 : a = 0) : c / b = 11 := 
by
  sorry

end average_median_eq_l1015_101543


namespace correct_system_of_equations_l1015_101580

theorem correct_system_of_equations (x y : ℝ) :
  (y - x = 4.5) ∧ (x - y / 2 = 1) ↔
  ((y - x = 4.5) ∧ (x - y / 2 = 1)) :=
by sorry

end correct_system_of_equations_l1015_101580


namespace player_matches_l1015_101571

theorem player_matches (n : ℕ) :
  (34 * n + 78 = 38 * (n + 1)) → n = 10 :=
by
  intro h
  have h1 : 34 * n + 78 = 38 * n + 38 := by sorry
  have h2 : 78 = 4 * n + 38 := by sorry
  have h3 : 40 = 4 * n := by sorry
  have h4 : n = 10 := by sorry
  exact h4

end player_matches_l1015_101571


namespace find_c_k_l1015_101588

noncomputable def a_n (n d : ℕ) := 1 + (n - 1) * d
noncomputable def b_n (n r : ℕ) := r ^ (n - 1)
noncomputable def c_n (n d r : ℕ) := a_n n d + b_n n r

theorem find_c_k (d r k : ℕ) (hd1 : c_n (k - 1) d r = 200) (hd2 : c_n (k + 1) d r = 2000) :
  c_n k d r = 423 :=
sorry

end find_c_k_l1015_101588


namespace negation_of_p_l1015_101564

-- Define the proposition p: ∀ x ∈ ℝ, sin x ≤ 1
def proposition_p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- The statement to prove the negation of proposition p
theorem negation_of_p : ¬proposition_p ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end negation_of_p_l1015_101564


namespace opposite_of_9_is_neg_9_l1015_101545

-- Definition of opposite number according to the given condition
def opposite (n : Int) : Int := -n

-- Proof statement that the opposite of 9 is -9
theorem opposite_of_9_is_neg_9 : opposite 9 = -9 :=
by
  sorry

end opposite_of_9_is_neg_9_l1015_101545


namespace fraction_simplified_form_l1015_101562

variables (a b c : ℝ)

noncomputable def fraction : ℝ := (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b)

theorem fraction_simplified_form (h : a^2 - c^2 + b^2 + 2 * a * b ≠ 0) :
  fraction a b c = (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b) :=
by sorry

end fraction_simplified_form_l1015_101562


namespace max_star_player_salary_l1015_101576

-- Define the constants given in the problem
def num_players : Nat := 12
def min_salary : Nat := 20000
def total_salary_cap : Nat := 1000000

-- Define the statement we want to prove
theorem max_star_player_salary :
  (∃ star_player_salary : Nat, 
    star_player_salary ≤ total_salary_cap - (num_players - 1) * min_salary ∧
    star_player_salary = 780000) :=
sorry

end max_star_player_salary_l1015_101576


namespace bake_sale_cookies_l1015_101587

theorem bake_sale_cookies (raisin_cookies : ℕ) (oatmeal_cookies : ℕ) 
  (h1 : raisin_cookies = 42) 
  (h2 : raisin_cookies / oatmeal_cookies = 6) :
  raisin_cookies + oatmeal_cookies = 49 :=
sorry

end bake_sale_cookies_l1015_101587


namespace triangles_side_product_relation_l1015_101598

-- Define the two triangles with their respective angles and side lengths
variables (A B C A1 B1 C1 : Type) 
          (angle_A angle_A1 angle_B angle_B1 : ℝ) 
          (a b c a1 b1 c1 : ℝ)

-- Given conditions
def angles_sum_to_180 (angle_A angle_A1 : ℝ) : Prop :=
  angle_A + angle_A1 = 180

def angles_equal (angle_B angle_B1 : ℝ) : Prop :=
  angle_B = angle_B1

-- The main theorem to be proven
theorem triangles_side_product_relation 
  (h1 : angles_sum_to_180 angle_A angle_A1)
  (h2 : angles_equal angle_B angle_B1) :
  a * a1 = b * b1 + c * c1 :=
sorry

end triangles_side_product_relation_l1015_101598


namespace extremum_of_cubic_function_l1015_101558

noncomputable def cubic_function (x : ℝ) : ℝ := 2 - x^2 - x^3

theorem extremum_of_cubic_function : 
  ∃ x_max x_min : ℝ, 
    cubic_function x_max = x_max_value ∧ 
    cubic_function x_min = x_min_value ∧ 
    ∀ x : ℝ, cubic_function x ≤ cubic_function x_max ∧ cubic_function x_min ≤ cubic_function x :=
sorry

end extremum_of_cubic_function_l1015_101558


namespace complement_union_l1015_101502

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end complement_union_l1015_101502


namespace grasshoppers_after_transformations_l1015_101572

-- Define initial conditions and transformation rules
def initial_crickets : ℕ := 30
def initial_grasshoppers : ℕ := 30

-- Define the transformations
def red_haired_transforms (g : ℕ) (c : ℕ) : ℕ × ℕ :=
  (g - 4, c + 1)

def green_haired_transforms (c : ℕ) (g : ℕ) : ℕ × ℕ :=
  (c - 5, g + 2)

-- Define the total number of transformations and the resulting condition
def total_transformations : ℕ := 18
def final_crickets : ℕ := 0

-- The proof goal
theorem grasshoppers_after_transformations : 
  initial_grasshoppers = 30 → 
  initial_crickets = 30 → 
  (∀ t, t = total_transformations → 
          ∀ g c, 
          (g, c) = (0, 6) → 
          (∃ m n, (m + n = t ∧ final_crickets = c))) →
  final_grasshoppers = 6 :=
by
  sorry

end grasshoppers_after_transformations_l1015_101572


namespace total_spending_march_to_july_l1015_101568

-- Define the conditions
def beginning_of_march_spending : ℝ := 1.2
def end_of_july_spending : ℝ := 4.8

-- State the theorem to prove
theorem total_spending_march_to_july : 
  end_of_july_spending - beginning_of_march_spending = 3.6 :=
sorry

end total_spending_march_to_july_l1015_101568


namespace ellipse_product_l1015_101585

/-- Given conditions:
1. OG = 8
2. The diameter of the inscribed circle of triangle ODG is 4
3. O is the center of an ellipse with major axis AB and minor axis CD
4. Point G is one focus of the ellipse
--/
theorem ellipse_product :
  ∀ (O G D : Point) (a b : ℝ),
    OG = 8 → 
    (a^2 - b^2 = 64) →
    (a - b = 4) →
    (AB = 2*a) →
    (CD = 2*b) →
    (AB * CD = 240) :=
by
  intros O G D a b hOG h1 h2 h3 h4
  sorry

end ellipse_product_l1015_101585


namespace population_growth_l1015_101500

theorem population_growth (scale_factor1 scale_factor2 : ℝ)
    (h1 : scale_factor1 = 1.2)
    (h2 : scale_factor2 = 1.26) :
    (scale_factor1 * scale_factor2) - 1 = 0.512 :=
by
  sorry

end population_growth_l1015_101500
