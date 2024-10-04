import Mathlib

namespace division_into_rectangles_l528_528027

theorem division_into_rectangles (figure : Type) (valid_division : figure → Prop) : (∃ ways, ways = 8) :=
by {
  -- assume given conditions related to valid_division using "figure"
  sorry
}

end division_into_rectangles_l528_528027


namespace tel_aviv_rain_probability_l528_528486

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l528_528486


namespace exists_unique_root_in_interval_l528_528519

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_unique_root_in_interval : 
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 :=
sorry

end exists_unique_root_in_interval_l528_528519


namespace Tricia_is_five_years_old_l528_528536

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l528_528536


namespace sum_of_twenty_numbers_l528_528219

theorem sum_of_twenty_numbers (a : ℕ → ℤ) (h₀ : ∀ n < 18, a n + a (n + 1) + a (n + 2) > 0) : 
  ¬ (∀ seq : Fin 20 → ℤ, (∀ n, seq n + seq ((n + 1) % 20) + seq ((n + 2) % 20) > 0) → (finset.univ.sum seq > 0)) :=
by
  sorry

end sum_of_twenty_numbers_l528_528219


namespace measure_angle_DFC_l528_528812

-- Define the conditions of the problem
variable {A B C D E F : Type*}
variable [geometry A B C D]
variable [midpoint E B C]
variable [foot_perpendicular F A D E]

-- Define the angle B measure
variable (angleB : ℝ) (h_angleB : angleB = 40)

-- Prove the measure of angle DFC
theorem measure_angle_DFC
: ∀ (ABCD : quadrilateral) (hc : rhombus ABCD) (hE : midpoint E (segment BC)) (hF : foot_perpendicular F A (line DE)),
  measure_angle D F C = 110 := by
  sorry

end measure_angle_DFC_l528_528812


namespace correct_statements_l528_528188

noncomputable def Andrew := sorry
noncomputable def Boris := sorry
noncomputable def Svetlana := sorry
noncomputable def Larisa := sorry

-- Conditions
axiom cond1 : Boris = max Boris Andrew Svetlana Larisa
axiom cond2a : Andrew < Svetlana
axiom cond2b : Larisa < Andrew
axiom marriage1 : Larisa's husband = Boris
axiom marriage2 : Andrew's wife = Svetlana

-- Proof goal
theorem correct_statements : 
  (Larisa < Andrew ∧ Andrew < Svetlana ∧ Svetlana < Boris ∧ Larisa's husband = Boris) 
  ∧ (¬(Boris's wife = Svetlana)) :=
    by {
      sorry
    }

end correct_statements_l528_528188


namespace sum_mnp_correct_l528_528523

noncomputable def sum_mnp : ℕ :=
  let m := 9
  let n := 17
  let p := 8
  m + n + p

theorem sum_mnp_correct (x : ℝ) (h_eq : x * (4 * x - 9) = -4) :
  ∃ (m n p : ℤ), (x = (m + real.sqrt n) / p ∨ x = (m - real.sqrt n) / p) ∧
  int.gcd m (int.gcd n p) = 1 ∧ sum_mnp = 34 := 
by {
  sorry
}

end sum_mnp_correct_l528_528523


namespace amount_spent_on_tracksuit_l528_528441

-- Definitions based on the conditions
def original_price (x : ℝ) := x
def discount_rate : ℝ := 0.20
def savings : ℝ := 30
def actual_spent (x : ℝ) := 0.8 * x

-- Theorem statement derived from the proof translation
theorem amount_spent_on_tracksuit (x : ℝ) (h : (original_price x) * discount_rate = savings) :
  actual_spent x = 120 :=
by
  sorry

end amount_spent_on_tracksuit_l528_528441


namespace arrangement_of_seating_l528_528979

-- Define the problem conditions
def democrats : ℕ := 6
def republicans : ℕ := 4
def total_people := democrats + republicans

-- Define the proof problem statement
theorem arrangement_of_seating : 
  let gaps := democrats  -- gaps between democrats
  let ways_to_arrange_democrats := fact (democrats - 1) -- 5!
  let ways_to_choose_gaps := nat.choose (democrats) republicans -- C(6, 4)
  let ways_to_arrange_republicans := fact republicans -- 4!
  ways_to_arrange_democrats * ways_to_choose_gaps * ways_to_arrange_republicans = 43200 :=
by
  sorry

end arrangement_of_seating_l528_528979


namespace number_of_ways_to_divide_l528_528025

-- Define the given shape
structure Shape :=
  (sides : Nat) -- Number of 3x1 stripes along the sides
  (centre : Nat) -- Size of the central square (3x3)

-- Define the specific problem shape
def problem_shape : Shape :=
  { sides := 4, centre := 9 } -- 3x1 stripes on all sides and a 3x3 centre

-- Theorem stating the number of ways to divide the shape into 1x3 rectangles
theorem number_of_ways_to_divide (s : Shape) (h1 : s.sides = 4) (h2 : s.centre = 9) : 
  ∃ ways, ways = 2 :=
by
  -- The proof is skipped
  sorry

end number_of_ways_to_divide_l528_528025


namespace trapezoid_AD_length_l528_528402

/-- Given a trapezoid ABCD where AD is the longer base, AC is perpendicular to CD and bisects ∠BAD, 
    ∠CDA = 60°, and the perimeter of the trapezoid is 2, then AD = 4/5 --/
theorem trapezoid_AD_length (A B C D : Type*) [trapezoid ABCD] 
  (h1 : is_longer_base AD) 
  (h2 : is_perpendicular AC CD) 
  (h3 : angle_bisector AC BAD) 
  (h4 : ∠CDA = 60) 
  (h5 : perimeter ABCD = 2) :
  length AD = 4 / 5 :=
sorry

end trapezoid_AD_length_l528_528402


namespace incorrect_statement_B_l528_528300

-- Defining the quadratic function
def quadratic_function (x : ℝ) : ℝ := -(x + 2)^2 - 3

-- Conditions derived from the problem
def statement_A : Prop := (∃ h k : ℝ, h < 0 ∧ k = 0)
def statement_B : Prop := (axis_of_symmetry (quadratic_function) = 2)
def statement_C : Prop := (¬ ∃ x : ℝ, quadratic_function x = 0)
def statement_D : Prop := (∀ x > -1, ∀ y > x, quadratic_function y < quadratic_function x)

-- The proof problem: show that statement B is incorrect
theorem incorrect_statement_B : statement_B = false :=
by sorry

end incorrect_statement_B_l528_528300


namespace largest_divisible_by_9_variant1_largest_divisible_by_9_variant2_largest_divisible_by_9_variant3_largest_divisible_by_9_variant4_largest_divisible_by_9_variant5_l528_528104

theorem largest_divisible_by_9_variant1 :
  ∃ (n : ℕ), n = 7654765464 ∧ (∃ (m : ℕ), m = 765476547654 ∧ ∀ (x : ℕ), x ∈ digits 10 m → ∑ d in digits 10 x, d = 66 ∧ x % 9 = 0) := sorry

theorem largest_divisible_by_9_variant2 :
  ∃ (n : ℕ), n = 7647645645 ∧ (∃ (m : ℕ), m = 764576457645 ∧ ∀ (x : ℕ), x ∈ digits 10 m → ∑ d in digits 10 x, d = 66 ∧ x % 9 = 0) := sorry

theorem largest_divisible_by_9_variant3 :
  ∃ (n : ℕ), n = 7457456745 ∧ (∃ (m : ℕ), m = 674567456745 ∧ ∀ (x : ℕ), x ∈ digits 10 m → ∑ d in digits 10 x, d = 66 ∧ x % 9 = 0) := sorry

theorem largest_divisible_by_9_variant4 :
  ∃ (n : ℕ), n = 4674567456 ∧ (∃ (m : ℕ), m = 456745674567 ∧ ∀ (x : ℕ), x ∈ digits 10 m → ∑ d in digits 10 x, d = 66 ∧ x % 9 = 0) := sorry

theorem largest_divisible_by_9_variant5 :
  ∃ (n : ℕ), n = 5475475467 ∧ (∃ (m : ℕ), m = 546754675467 ∧ ∀ (x : ℕ), x ∈ digits 10 m → ∑ d in digits 10 x, d = 66 ∧ x % 9 = 0) := sorry

end largest_divisible_by_9_variant1_largest_divisible_by_9_variant2_largest_divisible_by_9_variant3_largest_divisible_by_9_variant4_largest_divisible_by_9_variant5_l528_528104


namespace find_circle_center_l528_528202

noncomputable def midpoint_line (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def circle_center (x y : ℝ) : Prop :=
  6 * x - 5 * y = midpoint_line 40 (-20) ∧ 3 * x + 2 * y = 0

theorem find_circle_center : circle_center (20 / 27) (-10 / 9) :=
by
  -- Here would go the proof steps, but we skip it
  sorry

end find_circle_center_l528_528202


namespace P_minus_Q_equals_l528_528843

noncomputable def P := {x : ℝ | real.log x / real.log 2 < 1}
noncomputable def Q := {x : ℝ | |x - 2| < 1}
noncomputable def P_minus_Q := {x : ℝ | x ∈ P ∧ x ∉ Q}

theorem P_minus_Q_equals : P_minus_Q = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end P_minus_Q_equals_l528_528843


namespace geometric_series_sum_l528_528665

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (last_term : ℝ) 
  (h_a : a = 1) (h_r : r = -3) 
  (h_last_term : last_term = 6561) 
  (h_last_term_eq : a * r^n = last_term) : 
  a * (r^n - 1) / (r - 1) = 4921.25 :=
by
  -- Proof goes here
  sorry

end geometric_series_sum_l528_528665


namespace distance_after_3_minutes_l528_528641

/-- Let truck_speed be 65 km/h and car_speed be 85 km/h.
    Let time_in_minutes be 3 and converted to hours it is 0.05 hours.
    The goal is to prove that the distance between the truck and the car
    after 3 minutes is 1 kilometer. -/
def truck_speed : ℝ := 65 -- speed in km/h
def car_speed : ℝ := 85 -- speed in km/h
def time_in_minutes : ℝ := 3 -- time in minutes
def time_in_hours : ℝ := time_in_minutes / 60 -- converted time in hours
def distance_truck := truck_speed * time_in_hours
def distance_car := car_speed * time_in_hours
def distance_between : ℝ := distance_car - distance_truck

theorem distance_after_3_minutes : distance_between = 1 := by
  -- Proof steps would go here
  sorry

end distance_after_3_minutes_l528_528641


namespace find_m_l528_528783

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 5 * y + 20 = 0
def l2 (m x y : ℝ) : Prop := m * x + 2 * y - 10 = 0

-- Define the condition of perpendicularity
def lines_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Proving the value of m given the conditions
theorem find_m (m : ℝ) :
  (∃ x y : ℝ, l1 x y) → (∃ x y : ℝ, l2 m x y) → lines_perpendicular 2 (-5 : ℝ) m 2 → m = 5 :=
sorry

end find_m_l528_528783


namespace barrel_to_cask_ratio_l528_528825

theorem barrel_to_cask_ratio
  (k : ℕ) -- k is the multiple
  (B C : ℕ) -- B is the amount a barrel can store, C is the amount a cask can store
  (h1 : C = 20) -- C stores 20 gallons
  (h2 : B = k * C + 3) -- A barrel stores 3 gallons more than k times the amount a cask stores
  (h3 : 4 * B + C = 172) -- The total storage capacity is 172 gallons
  : B / C = 19 / 10 :=
sorry

end barrel_to_cask_ratio_l528_528825


namespace smallest_positive_period_of_f_interval_of_decrease_of_f_max_and_min_values_of_f_l528_528731

noncomputable def f (x : ℝ) : ℝ :=
  cos x * sin (x + π / 3) - sqrt 3 * cos x ^ 2 + sqrt 3 / 4

theorem smallest_positive_period_of_f :
  (∀ x : ℝ, f (x + π) = f x) ∧ (¬ ∃ T : ℝ, 0 < T ∧ T < π ∧ ∀ x : ℝ, f (x + T) = f x) :=
by sorry

theorem interval_of_decrease_of_f (k : ℤ) :
  (∀ x : ℝ, 
    (5 * π / 12 + k * π ≤ x ∧ x ≤ 11 * π / 12 + k * π) → 
    ∃ (y : ℝ) (hy : (5 * π / 12 + k * π ≤ y ∧ y ≤ 11 * π / 12 + k * π)), f' y < 0) :=
by sorry

theorem max_and_min_values_of_f :
  (∀ x : ℝ, - π / 4 ≤ x ∧ x ≤ π / 4 → f x ≤ 1 / 4 ∧ f x ≥ - 1 / 2) ∧ 
  (∃ x max (hx : - π / 4 ≤ x ∧ x ≤ π / 4), f x = 1 / 4) ∧ 
  (∃ x min (hx : - π / 4 ≤ x ∧ x ≤ π / 4), f x = - 1 / 2) :=
by sorry

end smallest_positive_period_of_f_interval_of_decrease_of_f_max_and_min_values_of_f_l528_528731


namespace distance_between_truck_and_car_l528_528628

def truck_speed : ℝ := 65 -- km/h
def car_speed : ℝ := 85 -- km/h
def time_minutes : ℝ := 3 -- minutes
def time_hours : ℝ := time_minutes / 60 -- converting minutes to hours

def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem distance_between_truck_and_car :
  let truck_distance := distance_traveled truck_speed time_hours
  let car_distance := distance_traveled car_speed time_hours
  truck_distance - car_distance = -1 := -- the distance is 1 km but negative when subtracting truck from car
by {
  sorry
}

end distance_between_truck_and_car_l528_528628


namespace time_to_cross_bridge_is_30_secs_l528_528619

def length_of_train : ℝ := 120
def speed_of_train_kmph : ℝ := 45
def length_of_bridge : ℝ := 255

def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600
def total_distance : ℝ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem time_to_cross_bridge_is_30_secs : time_to_cross_bridge = 30 :=
by
  -- Proof omitted
  sorry

end time_to_cross_bridge_is_30_secs_l528_528619


namespace fernanda_total_time_eq_90_days_l528_528701

-- Define the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Define the total time calculation
def total_time_to_finish_audiobooks (a h r : ℕ) : ℕ :=
  (h / r) * a

-- The assertion we need to prove
theorem fernanda_total_time_eq_90_days :
  total_time_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 :=
by sorry

end fernanda_total_time_eq_90_days_l528_528701


namespace coplanar_implies_k_eq_neg8_l528_528050

variables (V : Type*) [add_comm_group V] [vector_space ℝ V]
variables (O A B C D : V)
variables (k : ℝ)

-- The condition on the vectors.
def vector_condition : Prop :=
  4 • A - 3 • B + 7 • C + k • D = 0

-- The coplanarity condition (translated to k = -8)
theorem coplanar_implies_k_eq_neg8 (h : vector_condition A B C D k) : k = -8 :=
sorry

end coplanar_implies_k_eq_neg8_l528_528050


namespace dice_probability_third_six_l528_528250

theorem dice_probability_third_six 
  (p q : ℕ) 
  (h_rel_prime : Nat.coprime p q) 
  (h_prob_eq : p = 65 ∧ q = 102) :
  let fair_die_probability : ℚ := 1 / 6
  let biased_die_probability : ℚ := 2 / 3
  let other_sides_probability : ℚ := 1 / 15
  let choose_die_probability : ℚ := 1 / 2
  let two_sixes_probability_fair : ℚ := fair_die_probability ^ 2
  let two_sixes_probability_biased : ℚ := biased_die_probability ^ 2
  let total_two_sixes_probability : ℚ := choose_die_probability * two_sixes_probability_fair +
                                          choose_die_probability * two_sixes_probability_biased
  let conditional_prob_fair : ℚ := (two_sixes_probability_fair * choose_die_probability) /
                                    total_two_sixes_probability
  let conditional_prob_biased : ℚ := (two_sixes_probability_biased * choose_die_probability) /
                                       total_two_sixes_probability
  let third_roll_six_prob_fair : ℚ := fair_die_probability
  let third_roll_six_prob_biased : ℚ := biased_die_probability
  let final_probability : ℚ := third_roll_six_prob_fair * conditional_prob_fair +
                                 third_roll_six_prob_biased * conditional_prob_biased
  (p + q = 167)
  :=
sorry

end dice_probability_third_six_l528_528250


namespace distance_between_truck_and_car_l528_528627

def truck_speed : ℝ := 65 -- km/h
def car_speed : ℝ := 85 -- km/h
def time_minutes : ℝ := 3 -- minutes
def time_hours : ℝ := time_minutes / 60 -- converting minutes to hours

def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem distance_between_truck_and_car :
  let truck_distance := distance_traveled truck_speed time_hours
  let car_distance := distance_traveled car_speed time_hours
  truck_distance - car_distance = -1 := -- the distance is 1 km but negative when subtracting truck from car
by {
  sorry
}

end distance_between_truck_and_car_l528_528627


namespace intersection_empty_l528_528323

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l528_528323


namespace pairs_count_l528_528271

noncomputable def count_pairs : Nat :=
  let a := (12 + b) / 2
  let b := (a^2 / 12)
  if (12, a, b, ab).arith_seq ∧ (12, a, b).geo_seq then 1 else 0

theorem pairs_count : ∃! (a b : ℝ), count_pairs = 1 :=
by 
  -- We have given conditions to verify
  have h1: 12, a, b, ab form an arithmetic sequence := sorry
  have h2: 12, a, b form a geometric sequence := sorry
  use [12, 12] 
  simp [h1, h2, count_pairs]

end pairs_count_l528_528271


namespace orthocenter_construction_l528_528817

theorem orthocenter_construction {A B C D: Point} (h : IsTriangle A B C) 
  (hBD: IsAltitude B D C) (h45: Angle (A, B, D) = 45) 
  (hCircle: ∃ K, Circle K D = Distance D A ∧ OnLine (B, D, K)) : 
  IsOrthocenter K A B C :=
by 
  sorry

end orthocenter_construction_l528_528817


namespace solution_exists_l528_528957

namespace EquationSystem
-- Given the conditions of the equation system:
def eq1 (a b c d : ℝ) := a * b + a * c = 3 * b + 3 * c
def eq2 (a b c d : ℝ) := b * c + b * d = 5 * c + 5 * d
def eq3 (a b c d : ℝ) := a * c + c * d = 7 * a + 7 * d
def eq4 (a b c d : ℝ) := a * d + b * d = 9 * a + 9 * b

-- We need to prove that the solutions are as described:
theorem solution_exists (a b c d : ℝ) :
  eq1 a b c d → eq2 a b c d → eq3 a b c d → eq4 a b c d →
  (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨ ∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t :=
  by
    sorry
end EquationSystem

end solution_exists_l528_528957


namespace lottery_probability_l528_528009

theorem lottery_probability (p: ℝ) :
  (∀ n, 1 ≤ n ∧ n ≤ 15 → p = 2/3) →
  (true → p = 0.6666666666666666) →
  p = 2/3 :=
by
  intros h h'
  sorry

end lottery_probability_l528_528009


namespace sequence_term_2008_l528_528418

/--
Let \( u \) be a sequence where the first term is the smallest positive integer that is 1 more than a multiple of 3, 
the next two terms are the next two smallest positive integers that are each 2 more than a multiple of 3,
and the next three terms are the next three smallest positive integers that are each 3 more than a multiple of 3, and so on.
Prove that \( u_{2008} = 225 \).
-/
theorem sequence_term_2008 (u : ℕ → ℕ) (h1 : u 1 = 1)
  (h2 : ∀ n ∈ {2, 3}, u n = 1 + 3*(n-1)/2)
  (h3 : ∀ n ∈ {4, 5, 6}, u n = 1 + 3*(n-1))
  (h4 : ∀ n ∈ {7, 8, 9, 10}, u n = 1 + 3*(n-1)*2)
  -- Continue these conditions for all groups as per the problem, up to at least terms involved
  : u 2008 = 225 := sorry

end sequence_term_2008_l528_528418


namespace numerical_sequence_limit_exists_l528_528865

noncomputable def numerical_sequence_def (a_n : ℕ → ℝ) : Prop :=
  ∀ t ∈ Icc c d, ∃ l : ℂ, tendsto (λ n, complex.exp (complex.I * (t : ℂ) * a_n n)) at_top (𝓝 l)

theorem numerical_sequence_limit_exists (c d : ℝ) (a_n : ℕ → ℝ) 
  (hcd : c < d) (h : numerical_sequence_def c d a_n) : 
  ∃ a : ℝ, tendsto a_n at_top (𝓝 a) :=
begin
  sorry
end

end numerical_sequence_limit_exists_l528_528865


namespace cannot_transport_in_one_trip_l528_528564

/-- Condition definitions -/
def num_packages : ℕ := 50
def num_trucks : ℕ := 7
def truck_capacity : ℕ := 3000  -- kg
def package_weight (n : ℕ) : ℕ := 370 + 2 * (n - 1)

/-- Main theorem stating the transport impossibility -/
theorem cannot_transport_in_one_trip : 
  let total_weight := (num_packages * (370 + 470)) / 2
  let each_truck_capacity := truck_capacity * num_trucks
  total_weight = each_truck_capacity ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ num_packages → 
  (∑ i in range n, 370 + 2 * (i - 1)) ≤ truck_capacity := 
  if n = 8 then (∑ i in range n, 370 + 2 * (i - 1)) > truck_capacity 
  else if (n = num_packages) then total_weight > each_truck_capacity
  else sorry

end cannot_transport_in_one_trip_l528_528564


namespace peter_change_left_l528_528081

theorem peter_change_left
  (cost_small : ℕ := 3)
  (cost_large : ℕ := 5)
  (total_money : ℕ := 50)
  (num_small : ℕ := 8)
  (num_large : ℕ := 5) :
  total_money - (num_small * cost_small + num_large * cost_large) = 1 :=
by
  sorry

end peter_change_left_l528_528081


namespace bacteria_growth_relation_l528_528241

variable (w1: ℝ := 10.0) (w2: ℝ := 16.0) (w3: ℝ := 25.6)

theorem bacteria_growth_relation :
  (w2 / w1) = (w3 / w2) :=
by
  sorry

end bacteria_growth_relation_l528_528241


namespace M_inter_N_eq_l528_528064

def M : Set ℝ := {x | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_inter_N_eq : (M ∩ N) = Set.Ico 0 (1/2) := 
by
  sorry

end M_inter_N_eq_l528_528064


namespace problem_exist_formula_and_monotonic_intervals_l528_528348

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x + 1

theorem problem_exist_formula_and_monotonic_intervals :
  (∀ a b : ℝ, f(1) = a * (1:ℝ)^3 + b * (1:ℝ) + 1 → ∃ a b : ℝ, (a = 2 ∧ b = -6)) ∧
  (∀ x : ℝ, (∀ y : ℝ, f y = 2 * y^3 - 6 * y + 1) → deriv f x = 6 * x^2 - 6 →  (∀ x : ℝ, x < -1 ∨ x > 1 → deriv f x > 0)) :=
by
  sorry

end problem_exist_formula_and_monotonic_intervals_l528_528348


namespace sarahs_score_l528_528103

theorem sarahs_score (g s : ℕ) (h₁ : s = g + 30) (h₂ : (s + g) / 2 = 95) : s = 110 := by
  sorry

end sarahs_score_l528_528103


namespace even_number_of_items_selection_l528_528454

theorem even_number_of_items_selection (n : ℕ) : 
  ∃ (f : finset (finset ℕ)), f.card = 2 ^ (n - 1) :=
sorry

end even_number_of_items_selection_l528_528454


namespace sequence_range_increases_with_39_l528_528675

theorem sequence_range_increases_with_39 (s : Fin 11 → ℕ)
    (s_eq : s = ![41, 45, 52, 52, 53, 53, 60, 62, 67, 72, 78])
    (new_val : ℕ) 
    (new_val_eq : new_val = 39) :
    let new_seq := new_val :: s.toList
    let orig_range := 78 - 41
    let new_range := 78 - new_val
    new_range > orig_range := by
    sorry

end sequence_range_increases_with_39_l528_528675


namespace sqrt_inequality_l528_528881

theorem sqrt_inequality : sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 :=
by sorry

end sqrt_inequality_l528_528881


namespace find_A_max_min_l528_528234

theorem find_A_max_min :
  ∃ (A_max A_min : ℕ), 
    (A_max = 99999998 ∧ A_min = 17777779) ∧
    (∀ B A, 
      (B > 77777777) ∧
      (Nat.coprime B 36) ∧
      (A = (B % 10) * 10000000 + B / 10) →
      (A ≤ 99999998 ∧ A ≥ 17777779)) :=
by 
  existsi 99999998
  existsi 17777779
  split
  { 
    split 
    { 
      refl 
    }
    refl 
  }
  intros B A h
  sorry

end find_A_max_min_l528_528234


namespace bearing_of_fire_l528_528086

noncomputable def positions (A B C P: Type) : Prop :=
  (distance A B = 6) ∧
  (distance B C = 4) ∧
  (bearing B C = 330) ∧ -- 330° is equivalent to 30° west of north
  (signal_detected A P ∧ 
   signal_detected B P ∧ 
   signal_detected C P ∧ 
   time_difference A B = 4 ∧ 
   time_difference A C = 4 ∧ 
   signal_speed = 1)

theorem bearing_of_fire (A B C P : Type) (h : positions A B C P) : 
  bearing A P = 30 := 
sorry

end bearing_of_fire_l528_528086


namespace oscar_leap_longer_than_elmer_stride_l528_528276

theorem oscar_leap_longer_than_elmer_stride :
  ∀ (elmer_strides_per_gap oscar_leaps_per_gap gaps_between_poles : ℕ)
    (total_distance : ℝ),
  elmer_strides_per_gap = 60 →
  oscar_leaps_per_gap = 16 →
  gaps_between_poles = 60 →
  total_distance = 7920 →
  let elmer_stride_length := total_distance / (elmer_strides_per_gap * gaps_between_poles)
  let oscar_leap_length := total_distance / (oscar_leaps_per_gap * gaps_between_poles)
  oscar_leap_length - elmer_stride_length = 6.05 :=
by
  intros
  sorry

end oscar_leap_longer_than_elmer_stride_l528_528276


namespace inverse_function_value_l528_528354

theorem inverse_function_value :
  (∃ x, (x : ℝ) = 2⁻¹⁶ ∧ (λ g (x : ℝ), (x^6 - 1)/4) x = -1/8) :=
begin
  sorry
end

end inverse_function_value_l528_528354


namespace find_FA_dot_FB_l528_528393

-- Definitions
variables {F A B : Point}
def F := (1 : ℝ, 0 : ℝ)
def on_parabola (p : Point) := ∃ (x y : ℝ), p = (x, y) ∧ y^2 = 4 * x
def dot_product (u v : Point) := u.1 * v.1 + u.2 * v.2
def distance (u v : Point) := Real.sqrt ((u.1 - v.1)^2 + (u.2 - v.2)^2)

-- Conditions
axiom A_on_parabola : on_parabola A
axiom B_on_parabola : on_parabola B
axiom O_dot_product : dot_product (0, 0) A * dot_product (0, 0) B = -4
axiom distance_diff : abs (distance F A - distance F B) = 4 * Real.sqrt 3

-- Question
theorem find_FA_dot_FB : dot_product (F.1 - A.1, F.2 - A.2) (F.1 - B.1, F.2 - B.2) = -11 :=
sorry

end find_FA_dot_FB_l528_528393


namespace max_expr_value_l528_528929

variable {r s t u : ℕ}
variables (h₀ : {r, s, t, u} = {2, 3, 4, 5})

theorem max_expr_value : ∃ r s t u, {r, s, t, u} = {2, 3, 4, 5} ∧ r * s + u * r + t * r = 45 :=
by
  sorry

end max_expr_value_l528_528929


namespace part1_part2_l528_528983

theorem part1 (x y : ℕ) (h1 : 25 * x + 30 * y = 1500) (h2 : x = 2 * y - 4) : x = 36 ∧ y = 20 :=
by
  sorry

theorem part2 (x y : ℕ) (h1 : x + y = 60) (h2 : x ≥ 2 * y)
  (h_profit : ∃ p, p = 7 * x + 10 * y) : 
  ∃ x y profit, x = 40 ∧ y = 20 ∧ profit = 480 :=
by
  sorry

end part1_part2_l528_528983


namespace range_of_a_l528_528870

variable (f : ℝ → ℝ)
variable (a : ℝ)

theorem range_of_a (h1 : ∀ a : ℝ, (f (1 - 2 * a) / 2 ≥ f a))
                  (h2 : ∀ (x1 x2 : ℝ), x1 < x2 ∧ x1 + x2 ≠ 0 → f x1 > f x2) : a > (1 / 2) :=
by
  sorry

end range_of_a_l528_528870


namespace solution_set_inequality_l528_528923

theorem solution_set_inequality (x : ℝ) (hx : 0 < x) :
  (abs ((1 / log (1/2) x) + 2) > 3/2) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2^(2/7)) ∨ (4 < x) :=
by
  -- this is a placeholder for the proof
  sorry

end solution_set_inequality_l528_528923


namespace a_n_formula_T_n_formula_l528_528739

def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

axiom a_seq : ℕ → ℕ
axiom a_1 : a_seq 1 = 1
axiom seq_condition : S 1 (λ n, a_seq n) + (1 / 3) * S 5 (λ n, a_seq n) = (1 / 2) * S 3 (λ n, a_seq n)

theorem a_n_formula : ∀ n, a_seq n = n :=
sorry

def b_seq (n : ℕ) : ℕ := 2^(n - 1)
def T (n : ℕ) : ℕ := ∑ i in range n, (a_seq i + 1) * b_seq i + 1

theorem T_n_formula : ∀ n, T n = (n - 1) * 2^n + 1 :=
sorry

end a_n_formula_T_n_formula_l528_528739


namespace ratio_x_2y_l528_528715

theorem ratio_x_2y (x y : ℤ) (h : (7 * x + 8 * y) / (x - 2 * y) = 29) : x / (2 * y) = 3 / 2 :=
sorry

end ratio_x_2y_l528_528715


namespace exists_zero_in_interval_l528_528127

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem exists_zero_in_interval : ∃ c ∈ Set.Ioo Real.e 3, f c = 0 :=
by
  have h₁ : f 1 < 0 := by { simp [f, Real.log_one], norm_num }
  have h₂ : f Real.e > 0 := by { simp [f, Real.log_exp], norm_num }
  have h₃ : f 3 > 0 := by { simp [f, Real.log], linarith [Real.log_pos, show (3 : ℝ) > 0 by norm_num] }
  have h_sign_change : (f Real.e) * (f 3) < 0 := by { linarith }
  exact IntermediateValueTheorem f _ _ h_sign_change sorry

end exists_zero_in_interval_l528_528127


namespace chord_length_of_concentric_circles_l528_528900

open Real

theorem chord_length_of_concentric_circles 
  (a b : ℝ) 
  (h : a^2 - b^2 = 25) :
  ∃ c : ℝ, (c / 2)^2 + b^2 = a^2 ∧ c = 10 :=
by {
  existsi (10 : ℝ),
  sorry
}

end chord_length_of_concentric_circles_l528_528900


namespace count_divisors_24_528_l528_528773

theorem count_divisors_24_528 : 
  let divisors := {d : ℕ | d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 24528 % d = 0} in
  divisors.card = 8 :=
by
  sorry

end count_divisors_24_528_l528_528773


namespace winnie_keeps_lollipops_l528_528563

theorem winnie_keeps_lollipops :
  let cherry := 36
  let wintergreen := 125
  let grape := 8
  let shrimp_cocktail := 241
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  total_lollipops % friends = 7 :=
by
  sorry

end winnie_keeps_lollipops_l528_528563


namespace tables_needed_l528_528484

open Nat

theorem tables_needed (n_base_5 : ℕ) (seats_per_table : ℕ) (h_base5 : n_base_5 = 1 * (5^2) + 2 * (5^1) + 3 * (5^0)) :
  let n_base_10 := 25 + 10 + 3 in
  n_base_5 = n_base_10 → seats_per_table = 3 → ceil (n_base_10 / seats_per_table : ℝ) = 13 :=
by
  sorry

end tables_needed_l528_528484


namespace no_solution_exists_l528_528289

theorem no_solution_exists (n : ℕ) (hn : odd n) (hp : 0 < n) : ¬ (n^3 % 1000 = 668) :=
by {
    sorry
}

end no_solution_exists_l528_528289


namespace THH_before_HHH_l528_528958

-- Fair coin flip probabilities
def fair_coin := (1/2 : ℚ)

-- Definitions for sequences
def sequence_THH := [tt, ff, ff]
def sequence_HHH := [ff, ff, ff]

-- Event that checks for THH before HHH
noncomputable def Prob_THH_before_HHH : ℚ :=
  1 - (fair_coin ^ 3)

theorem THH_before_HHH : Prob_THH_before_HHH = 7 / 8 := by
  sorry

end THH_before_HHH_l528_528958


namespace smallest_possible_value_l528_528422

theorem smallest_possible_value (P : ℤ[X]) (a : ℕ) (h1 : 0 < a)
  (h2 : P.eval 1 = a) (h3 : P.eval 4 = a) (h4 : P.eval 7 = a) (h5 : P.eval 10 = a)
  (h6 : P.eval 3 = -a) (h7 : P.eval 6 = -a) (h8 : P.eval 9 = -a) (h9 : P.eval 12 = -a) :
  a = 4620 :=
by
  sorry

end smallest_possible_value_l528_528422


namespace incircle_homothety_tangent_to_circumcircle_l528_528043

-- Definitions and given conditions
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variable {triangle_ABC : A × B × C}
variable (is_right_angle_ABC : ∃ p : ℝ, angle C = p ∧ p = π / 2)
variable {homothety_center : C} {scale_factor : ℝ} (is_homothety : homothety homothety_center scale_factor)

-- Theorem statement
theorem incircle_homothety_tangent_to_circumcircle
  (h : angle C = π / 2) (homothety : homothety C 2) :
  ∀ (incircle : circle A) (circumcircle : circle A), 
  homothety (incircle) = incircle' → homothety (circumcircle) = circumcircle' → 
  is_tangent incircle' circumcircle' :=
sorry

end incircle_homothety_tangent_to_circumcircle_l528_528043


namespace prob_purchase_either_is_correct_prob_purchase_at_least_one_is_correct_prob_dist_correct_l528_528206

noncomputable def prob_purchase_either : ℝ := 0.5
noncomputable def prob_purchase_A : ℝ := 0.5
noncomputable def prob_purchase_B : ℝ := 0.6
noncomputable def prob_purchase_neither_A_B : ℝ := 0.2

theorem prob_purchase_either_is_correct :
  prob_purchase_either = prob_purchase_A * (1 - prob_purchase_B) + (1 - prob_purchase_A) * prob_purchase_B := 
sorry

theorem prob_purchase_at_least_one_is_correct :
  1 - (1 - prob_purchase_A) * (1 - prob_purchase_B) = 0.8 :=
sorry

noncomputable def binom_dist : ℕ → ℝ
| 0 := 0.2^3
| 1 := 3 * 0.8 * 0.2^2
| 2 := 3 * 0.8^2 * 0.2
| 3 := 0.8^3
| _ := 0

theorem prob_dist_correct :
  binom_dist 0 = 0.008 ∧ binom_dist 1 = 0.096 ∧ binom_dist 2 = 0.384 ∧ binom_dist 3 = 0.512 :=
sorry

end prob_purchase_either_is_correct_prob_purchase_at_least_one_is_correct_prob_dist_correct_l528_528206


namespace john_initial_investment_in_alpha_bank_is_correct_l528_528832

-- Definition of the problem conditions
def initial_investment : ℝ := 2000
def alpha_rate : ℝ := 0.04
def beta_rate : ℝ := 0.06
def final_amount : ℝ := 2398.32
def years : ℕ := 3

-- Alpha Bank growth factor after 3 years
def alpha_growth_factor : ℝ := (1 + alpha_rate) ^ years

-- Beta Bank growth factor after 3 years
def beta_growth_factor : ℝ := (1 + beta_rate) ^ years

-- The main theorem
theorem john_initial_investment_in_alpha_bank_is_correct (x : ℝ) 
  (hx : x * alpha_growth_factor + (initial_investment - x) * beta_growth_factor = final_amount) : 
  x = 246.22 :=
sorry

end john_initial_investment_in_alpha_bank_is_correct_l528_528832


namespace sum_is_correct_l528_528053

-- Define the variables and conditions
variables (a b c d : ℝ)
variable (x : ℝ)

-- Define the condition
def condition : Prop :=
  a + 1 = x ∧
  b + 2 = x ∧
  c + 3 = x ∧
  d + 4 = x ∧
  a + b + c + d + 5 = x

-- The theorem we need to prove
theorem sum_is_correct (h : condition a b c d x) : a + b + c + d = -10 / 3 :=
  sorry

end sum_is_correct_l528_528053


namespace g_root_is_correct_l528_528345

def f (x : Real) (a : Real) (g : Real → Real) : Real :=
  if x ≥ 0 then 1 - log a (x + 2)
  else g x

def f_is_odd (f : Real → Real) : Prop :=
  ∀ x : Real, f x = -f (-x)

def f_at_zero (f : Real → Real) : Prop :=
  f 0 = 0

noncomputable def g_root : Real :=
  let a : Real := 2 in
  let g : Real → Real := λ x, log a (2 - x) - 1 in
  let root := ∀ x : Real, g x = 2 in
  -6

theorem g_root_is_correct : g_root = -6 := 
  by sorry

end g_root_is_correct_l528_528345


namespace geometric_sequence_log_sum_l528_528421

open Real

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (h : ∀ n m, n = m -> a n = a m) (h_pos : ∀ n, 0 < a n) 
(h_geom : ∀ n, a (n + 1) = a n * a 1) (h_56 : a 5 * a 6 = 81) : 
  log 3 (a 1) + log 3 (a 2) + log 3 (a 3) + log 3 (a 4) + log 3 (a 5)
  + log 3 (a 6) + log 3 (a 7) + log 3 (a 8) + log 3 (a 9) + log 3 (a 10) = 20 :=
sorry

end geometric_sequence_log_sum_l528_528421


namespace _l528_528746

-- Definitions for Part 1
def pointA : ℝ × ℝ := (-1, 5)
def pointB : ℝ × ℝ := (5, 5)
def pointD : ℝ × ℝ := (5, -1)

noncomputable def circleC : (ℝ × ℝ) → ℝ → Prop := 
  λ center radius, ∀ point ∈ {pointA, pointB, pointD}, (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius

noncomputable theorem part1 : 
  (∃ center : ℝ × ℝ, ∃ radius : ℝ, circleC center radius) →
  (∃ center : ℝ × ℝ, ∃ radius : ℝ, center = (2, 2) ∧ radius = 18) := 
sorry

-- Definitions for Part 2
def circleM_eq (centerM : ℝ × ℝ) (radiusM : ℝ) : Prop :=
  ∀ point : ℝ × ℝ, (point.1 - centerM.1)^2 + (point.2 - centerM.2)^2 = radiusM → 
  centerM = (6, 6) ∧ radiusM = 2

noncomputable theorem part2 : 
  (∃ centerC : ℝ × ℝ, ∃ radiusC : ℝ, centerC = (2, 2) ∧ radiusC = 18) → 
  (∃ centerM : ℝ × ℝ, ∃ radiusM : ℝ, circleM_eq centerM radiusM) :=
sorry

end _l528_528746


namespace tangent_line_with_min_slope_inclination_angle_range_l528_528335

noncomputable def cubic_curve (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x + 1

theorem tangent_line_with_min_slope :
  ∃ x y : ℝ, cubic_curve 2 = 5/3 ∧ (deriv cubic_curve 2) = -1 ∧ (3 * x + 3 * y - 11 = 0) :=
by
  sorry

theorem inclination_angle_range :
  ∀ α : ℝ, (tan α ≥ -1) → (α ∈ set.Ico 0 (π / 2) ∪ set.Ico (3 * π / 4) π) :=
by
  sorry

end tangent_line_with_min_slope_inclination_angle_range_l528_528335


namespace probability_green_tile_l528_528199

-- Define the problem in Lean
def is_green (n : ℕ) : Prop := n % 7 = 3

def count_green_tiles (max_num : ℕ) : ℕ :=
  (Finset.range (max_num + 1)).filter is_green |>.card

def probability_of_green : ℚ :=
  count_green_tiles 100 / 100

theorem probability_green_tile :
  probability_of_green = 7 / 50 :=
by
  sorry

end probability_green_tile_l528_528199


namespace arithmetic_contains_geometric_iff_rational_l528_528302

theorem arithmetic_contains_geometric_iff_rational (a d : ℤ) (h_d : d ≠ 0) :
  (∃ (r : ℤ), ∃ (s : ℤ), ∃ (u : ℕ), a / d = s / r ∧ ∃ (n : ℕ → ℕ), ∀ m, a * (1 + r)^m = a + n m * d) ↔ a / d ∈ ℚ := 
sorry

end arithmetic_contains_geometric_iff_rational_l528_528302


namespace radio_highest_price_rank_l528_528243

theorem radio_highest_price_rank (P : Fin 16 → ℝ)
  (h_diff : ∀ i j, i ≠ j → P i ≠ P j)
  (h_radio_highest : ∃ i, P i = max (finset.univ : finset (Fin 16)).image P)
  (h_radio_13th_lowest : ∃ i, finset.sort (preorder.le P) (P '' set.univ).to_finset !nge = (finset.range 16).nth 12) :
  ∃ i, P i = max (finset.univ : finset (Fin 16)).image P → (finset.sort (preorder.le P) (P '' set.univ).to_finset).find_index P i = 0 :=
begin
  sorry
end

end radio_highest_price_rank_l528_528243


namespace concurrency_of_perpendiculars_l528_528801

theorem concurrency_of_perpendiculars
  (A B C A1 B1 C1 A2 B2 C2 D E F H: Type*)
  [triangle : IsTriangle ABC]
  [scalene : AreScalene ABC]
  [perpendicular : IsPerpendicular A BC A_1]
  [perpendicular2 : IsPerpendicular B CA B_1]
  [perpendicular3 : IsPerpendicular C AB C_1]
  [intersection1 : Intersection BC B1C1 A2]
  [intersection2 : Intersection CA C1A1 B2]
  [intersection3 : Intersection AB A1B1 C2]
  [midpoints : Midpoints ABC D E F]
  [orthocenter : IsOrthocenter ABC H]
  : Concurrent (PerpendicularFromMidpoint D A A2) 
                (PerpendicularFromMidpoint E B B2) 
                (PerpendicularFromMidpoint F C C2) :=
by
  sorry

end concurrency_of_perpendiculars_l528_528801


namespace annual_interest_rate_continuous_compounding_l528_528166

noncomputable def continuous_compounding_rate (A P : ℝ) (t : ℝ) : ℝ :=
  (Real.log (A / P)) / t

theorem annual_interest_rate_continuous_compounding :
  continuous_compounding_rate 8500 5000 10 = (Real.log (1.7)) / 10 :=
by
  sorry

end annual_interest_rate_continuous_compounding_l528_528166


namespace find_n_l528_528861

def number_of_divisors (n : ℕ) : ℕ :=
  if n = 0 then 0 else (factors n).length

theorem find_n (n : ℕ) (hn : 0 < n) (hd : number_of_divisors (n ^ n) = 861) : n = 20 :=
sorry

end find_n_l528_528861


namespace speed_of_stream_l528_528949

theorem speed_of_stream (v : ℝ) (canoe_speed : ℝ) 
  (upstream_speed_condition : canoe_speed - v = 3) 
  (downstream_speed_condition : canoe_speed + v = 12) :
  v = 4.5 := 
by 
  sorry

end speed_of_stream_l528_528949


namespace fraction_of_tank_used_l528_528654

theorem fraction_of_tank_used (speed : ℝ) (fuel_efficiency : ℝ) (initial_fuel : ℝ) (time_traveled : ℝ)
  (h_speed : speed = 40) (h_fuel_eff : fuel_efficiency = 1 / 40) (h_initial_fuel : initial_fuel = 12) 
  (h_time : time_traveled = 5) : 
  (speed * time_traveled * fuel_efficiency) / initial_fuel = 5 / 12 :=
by
  -- Here the proof would go, but we add sorry to indicate it's incomplete.
  sorry

end fraction_of_tank_used_l528_528654


namespace visible_faces_not_101_visible_faces_even_l528_528693

-- Define the structure and conditions of the problem
structure Block where
  faces : Fin 6 → Fin 7 -- faces are labeled 1 to 6

structure Cube where
  blocks : Fin 8 → Block

-- We define the condition that adjoining faces have the same number
def adjoining_face_condition (c : Cube) : Prop :=
  -- Write a property that enforces matching numbers on adjoining faces
  sorry

-- Define the total sum of the numbers on all small cubes
def total_sum (c : Cube) : ℕ :=
  ∑ i, ∑ j, (c.blocks i).faces j

-- Define the sum of the numbers on the visible (external) faces
def visible_sum (c : Cube) : ℕ :=
  -- Write a computation that extracts the visible face sum, separating internal and external faces
  sorry

-- The theorem statement
theorem visible_faces_not_101 (c : Cube) (h_adjoin : adjoining_face_condition c) :
  visible_sum c ≠ 101 :=
by
  sorry

-- Proving the higher-level theorem that the sum of visible faces is even
theorem visible_faces_even (c : Cube) (h_adjoin : adjoining_face_condition c) :
  visible_sum c % 2 = 0 :=
by
  sorry

end visible_faces_not_101_visible_faces_even_l528_528693


namespace days_until_see_grandma_l528_528833

def hours_in_a_day : ℕ := 24
def hours_until_see_grandma : ℕ := 48

theorem days_until_see_grandma : hours_until_see_grandma / hours_in_a_day = 2 := by
  sorry

end days_until_see_grandma_l528_528833


namespace find_point_p_coordinates_l528_528744

theorem find_point_p_coordinates : 
  ∃ P : ℝ × ℝ, 
  (P = (1/3, 0) ∨ P = (-5, 8)) ∧ 
  ∃ (A B : ℝ × ℝ), 
  A = (3, -4) ∧ 
  B = (-1, 2) ∧ 
  (P lies on the line through A and B) ∧ 
  (|A to P distance| = 2 * |P to B distance|) := 
sorry

end find_point_p_coordinates_l528_528744


namespace largest_four_digit_neg_int_congruent_to_2_mod_25_l528_528943

theorem largest_four_digit_neg_int_congruent_to_2_mod_25 :
  ∃ n : ℤ, (25 * n + 2 ∣ -10000 ≤ 25 * n + 2 ∧ 25 * n + 2 < 0 ∧ 25 * n + 2 ≡ 2 [MOD 25]) ∧
           (∀ m : ℤ, 25 * m + 2 ∣ -1000 ≤ 25 * m + 2 ∧ 25 * m + 2 < 0 ∧ 25 * m + 2 ≡ 2 [MOD 25] → 25 * m + 2 ≤ 25 * n + 2) :=
begin
  let answer := -1023,
  use (-41),
  have h1: 25 * (-41) + 2 = answer, by norm_num,
  split,
  {split,
    {exact dec_trivial},
    {exact dec_trivial}},
  {intros m hm,
   cases hm with hm1 hm2,
   cases hm2 with hm3 hm4,
   have : 25 * m + 2 < answer, sorry,
   exact this}
end

end largest_four_digit_neg_int_congruent_to_2_mod_25_l528_528943


namespace area_of_trapezoid_l528_528806

noncomputable def triangle_XYZ_is_isosceles : Prop := 
  ∃ (X Y Z : Type) (XY XZ : ℝ), XY = XZ

noncomputable def identical_smaller_triangles (area : ℝ) (num : ℕ) : Prop := 
  num = 9 ∧ area = 3

noncomputable def total_area_large_triangle (total_area : ℝ) : Prop := 
  total_area = 135

noncomputable def trapezoid_contains_smaller_triangles (contained : ℕ) : Prop :=
  contained = 4

theorem area_of_trapezoid (XYZ_area smaller_triangle_area : ℝ) 
    (num_smaller_triangles contained_smaller_triangles : ℕ) : 
    triangle_XYZ_is_isosceles → 
    identical_smaller_triangles smaller_triangle_area num_smaller_triangles →
    total_area_large_triangle XYZ_area →
    trapezoid_contains_smaller_triangles contained_smaller_triangles →
    (XYZ_area - contained_smaller_triangles * smaller_triangle_area) = 123 :=
by
  intros iso smaller_triangles total_area contained
  sorry

end area_of_trapezoid_l528_528806


namespace AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l528_528058

variables {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ)

-- Condition
def AB2A_eq_AB := A * B ^ 2 * A = A * B * A

-- Part (a): Prove that (AB)^2 = AB
theorem AB_squared_eq_AB (h : AB2A_eq_AB A B) : (A * B) ^ 2 = A * B :=
sorry

-- Part (b): Prove that (AB - BA)^3 = 0
theorem AB_minus_BA_cubed_eq_zero (h : AB2A_eq_AB A B) : (A * B - B * A) ^ 3 = 0 :=
sorry

end AB_squared_eq_AB_AB_minus_BA_cubed_eq_zero_l528_528058


namespace first_student_speed_l528_528543

theorem first_student_speed(
  (v : ℝ), 
  initial_distance : ℝ := 350,
  second_student_speed : ℝ := 1.9,
  time_meeting : ℝ := 100
) 
: v + second_student_speed = initial_distance / time_meeting → v = 1.6 := 
begin
  intros h,
  have h1 : v = initial_distance / time_meeting - second_student_speed,
  {
    rw ← h,
    ring,
  },
  have h2 : initial_distance / time_meeting = 3.5,
  {
    norm_num,
  },
  have eq_v : 3.5 - 1.9 = 1.6,
  {
    norm_num,
  },
  rw [← h2] at h1,
  rw h1,
  exact eq_v,
end

end first_student_speed_l528_528543


namespace total_food_consumed_l528_528133

theorem total_food_consumed (n1 n2 f1 f2 : ℕ) (h1 : n1 = 4000) (h2 : n2 = n1 - 500) (h3 : f1 = 10) (h4 : f2 = f1 - 2) : 
    n1 * f1 + n2 * f2 = 68000 := by 
  sorry

end total_food_consumed_l528_528133


namespace renovation_team_rates_renovation_plan_comparison_l528_528201

theorem renovation_team_rates (a b : ℕ) (h₀: a = b + 30)
    (h₁: 360 * b = 300 * a) : a = 180 ∧ b = 150 :=
begin
  -- skipped the proof
  sorry
end

theorem renovation_plan_comparison (a b S : ℕ) (h₀: a = 180)
    (h₁: b = 150) : 
    let plan1_time := (a + b) * S / (2 * a * b),
        plan2_time := 2 * S / (a + b)
    in plan2_time < plan1_time :=
begin
  -- skipped the proof
  sorry
end

end renovation_team_rates_renovation_plan_comparison_l528_528201


namespace gcd_225_135_gcd_72_168_bin_to_dec_11011_l528_528975

-- GCD of 225 and 135 using the Euclidean algorithm is 45
theorem gcd_225_135 : Int.gcd 225 135 = 45 := by
  sorry

-- GCD of 72 and 168 using the method of successive subtraction is 24
theorem gcd_72_168 : Int.gcd 72 168 = 24 := by
  sorry

-- Converting 11011 (binary) to decimal gives 27
theorem bin_to_dec_11011 : Nat.ofDigits 2 [1, 1, 0, 1, 1] = 27 := by
  sorry

end gcd_225_135_gcd_72_168_bin_to_dec_11011_l528_528975


namespace admission_price_for_children_l528_528529

theorem admission_price_for_children (people_at_play : ℕ) (admission_price_adult : ℕ) (total_receipts : ℕ) (adults_attended : ℕ) 
  (h1 : people_at_play = 610) (h2 : admission_price_adult = 2) (h3 : total_receipts = 960) (h4 : adults_attended = 350) : 
  ∃ (admission_price_child : ℕ), admission_price_child = 1 :=
by
  sorry

end admission_price_for_children_l528_528529


namespace rectangles_in_grid_l528_528604

theorem rectangles_in_grid (h_lines : ℕ) (v_lines : ℕ) : h_lines = 5 → v_lines = 6 → (∃ (n : ℕ), n = 150) :=
begin
  intros h_eq v_eq,
  use (Nat.choose h_lines 2 * Nat.choose v_lines 2),
  rw [h_eq, v_eq],
  have h_choose : Nat.choose 5 2 = 10 := by norm_num,
  have v_choose : Nat.choose 6 2 = 15 := by norm_num,
  rw [h_choose, v_choose],
  norm_num,
end

end rectangles_in_grid_l528_528604


namespace value_of_expression_l528_528966

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end value_of_expression_l528_528966


namespace area_of_shaded_region_l528_528159

-- Definitions for the conditions
def radius_1 : ℝ := 3
def radius_2 : ℝ := 5
def distance_between_centers (r1 r2 : ℝ) : ℝ := r1 + r2
def radius_circumscribing (r1 r2 d : ℝ) : ℝ := (r1 + r2 + real.sqrt ((r1 + r2)^2 + (r2 - r1)^2)) / 2

-- The conditions given in the problem
def c1 := radius_1 = 3
def c2 := radius_2 = 5
def c3 := distance_between_centers radius_1 radius_2 = 8
def radius_larger_circle := radius_circumscribing radius_1 radius_2 (distance_between_centers radius_1 radius_2)

-- The proof problem
theorem area_of_shaded_region : 
  (π * radius_larger_circle^2) - (π * radius_1^2) - (π * radius_2^2) = 14 * π + 32 * real.sqrt 2 * π :=
by 
  sorry

end area_of_shaded_region_l528_528159


namespace min_distance_tetrahedron_l528_528904

theorem min_distance_tetrahedron :
  ∀ (A B C D P Q : ℝ^3),
    (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C D = 1) ∧ (dist D A = 1) ∧ (dist A C = 1) ∧ (dist B D = 1) ∧
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • A + (1 - t) • B) ∧
    (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = s • C + (1 - s) • D) →
    dist P Q = (Real.sqrt 2) / 2 :=
by
  sorry

end min_distance_tetrahedron_l528_528904


namespace incorrect_axis_symmetry_l528_528298

noncomputable def quadratic_function (x : ℝ) : ℝ := - (x + 2)^2 - 3

theorem incorrect_axis_symmetry :
  (∀ x : ℝ, quadratic_function x < 0) ∧
  (∀ x : ℝ, x > -1 → (quadratic_function x < quadratic_function (-2))) ∧
  (¬∃ x : ℝ, quadratic_function x = 0) ∧
  (¬ ∀ x : ℝ, x = 2) →
  false :=
by
  sorry

end incorrect_axis_symmetry_l528_528298


namespace option_b_correct_l528_528954

theorem option_b_correct (a : ℝ) : (a ^ 3) * (a ^ 2) = a ^ 5 := 
by
  sorry

end option_b_correct_l528_528954


namespace triangle_area_equality_l528_528006

theorem triangle_area_equality
  (A B C D E F : Point)
  (ABF : Triangle A B F)
  (hD : D ∈ segment A B)
  (hC : C ∈ segment A F)
  (h_eq1 : dist A B = dist A C)
  (h_eq2 : dist B D = dist C F)
  (h_intersect : ∃ E, E ∈ line_segments_overlap (line B C) (line D F)) :
  area (triangle B D E) + area (triangle C F E) = area (triangle C D E) + area (triangle B F E) :=
by sorry

end triangle_area_equality_l528_528006


namespace train_speed_approximation_l528_528622

def speed_in_km_per_hr (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed_approximation :
  speed_in_km_per_hr 250 9 = 100 :=
by
  sorry

end train_speed_approximation_l528_528622


namespace units_digit_57_pow_57_l528_528169

theorem units_digit_57_pow_57 : 
  let units_digit_cycle : Fin 4 → ℕ := fun n =>
    match n with
    | 0 => 7
    | 1 => 9
    | 2 => 3
    | 3 => 1
  in
  (57 % 4 = 1) → (units_digit_cycle (57 % 4) = 7) :=
by
  -- conditions and definitions
  intros
  sorry

end units_digit_57_pow_57_l528_528169


namespace cos2α_plus_sin2α_neg_7_over_5_l528_528768

theorem cos2α_plus_sin2α_neg_7_over_5 (α : ℝ) :
  let a := (2, 1) in
  let b := (Real.sin α - Real.cos α, Real.sin α + Real.cos α) in 
  let parallel (u v : ℝ × ℝ) := ∃ k : ℝ, u = (k * v.1, k * v.2) in

  parallel a b → Real.cos (2 * α) + Real.sin (2 * α) = -7 / 5 :=
by
  intros
  sorry

end cos2α_plus_sin2α_neg_7_over_5_l528_528768


namespace danny_bottle_caps_after_collection_l528_528683

-- Definitions for the conditions
def initial_bottle_caps : ℕ := 69
def bottle_caps_thrown : ℕ := 60
def bottle_caps_found : ℕ := 58

-- Theorem stating the proof problem
theorem danny_bottle_caps_after_collection : 
  initial_bottle_caps - bottle_caps_thrown + bottle_caps_found = 67 :=
by {
  -- Placeholder for proof
  sorry
}

end danny_bottle_caps_after_collection_l528_528683


namespace compound_interest_calculation_l528_528707

theorem compound_interest_calculation:
  let A0 := 9828 in
  let r1 := 0.04 in
  let r2 := 0.05 in
  let A1 := A0 + A0 * r1 in
  let A2 := A1 + A1 * r2 in
  A2 = 10732.176 :=
by {
  sorry
}

end compound_interest_calculation_l528_528707


namespace percentage_increase_overtime_rate_l528_528200

-- Define the given conditions
def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def total_hours_worked : ℝ := 60
def total_compensation : ℝ := 1200

-- Define the theorem we want to prove
theorem percentage_increase_overtime_rate :
  let regular_earnings := regular_hours * regular_rate,
      overtime_hours := total_hours_worked - regular_hours,
      overtime_earnings := total_compensation - regular_earnings,
      overtime_rate_per_hour := overtime_earnings / overtime_hours,
      percentage_increase := ((overtime_rate_per_hour - regular_rate) / regular_rate) * 100
  in percentage_increase = 75 :=
by
  sorry

end percentage_increase_overtime_rate_l528_528200


namespace rectangles_single_row_7_rectangles_grid_7_4_l528_528592

def rectangles_in_single_row (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem rectangles_single_row_7 :
  rectangles_in_single_row 7 = 28 :=
by
  -- Add the proof here
  sorry

def rectangles_in_grid (rows cols : ℕ) : ℕ :=
  ((cols + 1) * cols / 2) * ((rows + 1) * rows / 2)

theorem rectangles_grid_7_4 :
  rectangles_in_grid 4 7 = 280 :=
by
  -- Add the proof here
  sorry

end rectangles_single_row_7_rectangles_grid_7_4_l528_528592


namespace centroid_circumcircle_const_l528_528842

section CircumcircleLemma

variable (A B C P : ℝ) (R : ℝ) (centroid : (ℝ × ℝ × ℝ) → ℝ) (circumcircle : (ℝ × ℝ × ℝ) → ℝ → Prop)
variable [LinearOrderedField ℝ] [MetricSpace ℝ]

theorem centroid_circumcircle_const {A B C P P₁ : ℝ×ℝ} (G : ℝ) (R : ℝ) :
  (P₁ ∈ circumcircle (A, B, C) R) →
  (\|P₁ - A\| ^ 2 + \|P₁ - B\| ^ 2 + \|P₁ - C\| ^ 2 - \|P₁ - (\frac{1}{3} * (A + B + C))\| ^ 2) = 6 * (R ^ 2) :=
by
  sorry

end CircumcircleLemma

end centroid_circumcircle_const_l528_528842


namespace standard_equation_ellipse_no_line_l_exists_l528_528342

variables {a b : ℝ} (x y : ℝ)
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

def F1 : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (-2, 0)

theorem standard_equation_ellipse :
  a = 2 ∧ b = sqrt 3 →
  (∀ x y, (x, y) ∈ ellipse x y ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by
  -- Proof is omitted
  sorry

theorem no_line_l_exists (x0 y0 : ℝ) :
  -2 < x0 ∧ x0 < 2 ∧ ((x0^2 / 4) + (y0^2 / 3) = 1) →
  ¬∃ l, (x0, y0) lies_on_circle l (F1, C) :=
by
  -- Proof is omitted
  sorry

end standard_equation_ellipse_no_line_l_exists_l528_528342


namespace seven_people_round_table_l528_528800

theorem seven_people_round_table : 
  let num_people := 7
  let num_linear_arrangements := nat.factorial num_people
  let num_circular_arrangements := num_linear_arrangements / num_people
  num_circular_arrangements = 720 :=
by
  sorry

end seven_people_round_table_l528_528800


namespace purchase_total_cost_l528_528070

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end purchase_total_cost_l528_528070


namespace visible_area_correct_l528_528436

-- Defining the problem conditions
def side_length : ℝ := 7
def visibility_distance : ℝ := 1.5

-- Define the function to compute the visible area
def visible_area (side_length visibility_distance : ℝ) : ℝ :=
  let full_square_area := side_length^2
  let invisible_inner_square_side := side_length - 2 * visibility_distance
  let invisible_inner_square_area := invisible_inner_square_side^2
  let visible_inner_area := full_square_area - invisible_inner_square_area

  let side_rectangles_area := 4 * (side_length * visibility_distance)
  let corner_circles_area := 4 * (real.pi * (visibility_distance^2) / 4)

  let visible_outer_area := side_rectangles_area + corner_circles_area

  visible_inner_area + visible_outer_area

-- Prove the total visible area is 82 km²
theorem visible_area_correct :
  real.floor (visible_area side_length visibility_distance) = 82 := by
  sorry

end visible_area_correct_l528_528436


namespace rationalize_denominator_eq_l528_528883

theorem rationalize_denominator_eq (P Q R : ℤ) (hP : P = 5) (hQ : Q = 343) (hR : R = 21)
    (hRQ_pos : (0 : ℤ) < R) (hQ_cond : ∀ (k : ℕ), ¬ (k ^ 4 ∣ Q)) :
    P + Q + R = 369 :=
by
  sorry

end rationalize_denominator_eq_l528_528883


namespace num_valid_schedules_l528_528225

def employees : Type := fin 7

def valid_schedule (schedule : employees → ℕ) : Prop :=
   -- Employee A is scheduled to work on either Monday (1) or Tuesday (2)
   (schedule 0 = 1 ∨ schedule 0 = 2) ∧
   -- Employee B is not scheduled to work on Tuesday (2)
   (schedule 1 ≠ 2) ∧
   -- Employee C is scheduled to work on Friday (5)
   (schedule 2 = 5) ∧
   -- Each employee takes one night shift without repetition
   function.injective schedule ∧
   -- There are 7 shifts (1 to 7), and they are all assigned
   (∀ t, 1 ≤ t ∧ t ≤ 7 → ∃ e, schedule e = t)

theorem num_valid_schedules :
  {schedule : employees → ℕ // valid_schedule schedule}.card = 216 :=
by sorry

end num_valid_schedules_l528_528225


namespace TetrahedronVolume_l528_528116

noncomputable def TetrahedronVolumeProof (AB AC BC BD AD : ℝ) (CD : ℝ) (volume : ℝ) :=
AB = 5 ∧ AC = 3 ∧ BC = 4 ∧ BD = 4 ∧ AD = 3 ∧ CD = (12/5 * Real.sqrt 2) ∧
volume = 24/5

theorem TetrahedronVolume : ∃ volume : ℝ, TetrahedronVolumeProof 5 3 4 4 3 ((12/5) * Real.sqrt 2) volume :=
begin
  sorry
end

end TetrahedronVolume_l528_528116


namespace Jolyn_older_than_Clarisse_l528_528293

structure AgeDifference where
  months : Int
  days : Int

def JolynToTherese : AgeDifference := { months := 2, days := 10 }
def ThereseToAivo : AgeDifference := { months := 5, days := 15 }
def LeonToAivo : AgeDifference := { months := 2, days := 25 }
def ClarisseToLeon : AgeDifference := { months := 3, days := 20 }
def JolynToClarisse : AgeDifference := { months := 1, days := 10 }

theorem Jolyn_older_than_Clarisse :
  let JolynToAivo := { months := JolynToTherese.months + ThereseToAivo.months, days := JolynToTherese.days + ThereseToAivo.days }
  let normalized_JolynToAivo := if JolynToAivo.days >= 30 then { months := JolynToAivo.months + JolynToAivo.days / 30, days := JolynToAivo.days % 30 } else JolynToAivo
  let ClarisseToAivo := { months := LeonToAivo.months + ClarisseToLeon.months, days := LeonToAivo.days + ClarisseToLeon.days }
  let normalized_ClarisseToAivo := if ClarisseToAivo.days >= 30 then { months := ClarisseToAivo.months + ClarisseToAivo.days / 30, days := ClarisseToAivo.days % 30 } else ClarisseToAivo
  let JolynToClarisse_calc := { months := normalized_JolynToAivo.months - normalized_ClarisseToAivo.months, days := normalized_JolynToAivo.days - normalized_ClarisseToAivo.days }
  JolynToClarisse_calc = JolynToClarisse := by 
  sorry

end Jolyn_older_than_Clarisse_l528_528293


namespace string_length_correct_l528_528203

noncomputable def length_of_string 
(circumference height : ℝ) (loops : ℕ) : ℝ :=
let vertical_travel := height / loops in
let loop_length := Real.sqrt (circumference ^ 2 + vertical_travel ^ 2) in
loops * loop_length

theorem string_length_correct 
(c : ℝ) (h : ℝ) (n : ℕ) (hc : c = 6) (hh : h = 15) (hn : n = 3) :
length_of_string c h n = 3 * Real.sqrt 61 :=
by
  rw [hc, hh, hn]
  dsimp [length_of_string]
  norm_num
  rw [Real.sqrt_eq_rpow]
  norm_num
  apply_real_sqrt
  sorry -- Replace this by actual computation proof step

end string_length_correct_l528_528203


namespace solve_equation_l528_528887

theorem solve_equation 
  (x : ℝ) 
  (h1 : 4 * x - 3 ≥ 0)
  (h2 : sqrt (4 * x - 3) ≠ 0) 
  (h3 : sqrt (4 * x - 3) + 16 / sqrt (4 * x - 3) = 8) 
  : x = 19 / 4 :=
sorry

end solve_equation_l528_528887


namespace sequence_of_numbers_exists_l528_528820

theorem sequence_of_numbers_exists :
  ∃ (a b : ℤ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) :=
sorry

end sequence_of_numbers_exists_l528_528820


namespace problem_solution_l528_528847

noncomputable def clubsuit (a b : ℝ) : ℝ := (3 * a / (2 * b)) * (b / (3 * a))

theorem problem_solution : (clubsuit 8 (clubsuit 4 7)) ⋆ 3 = 0.5 :=
by
  have h1 : ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → clubsuit a b = 0.5 := fun a b ha hb =>
    calc
      clubsuit a b = (3 * a / (2 * b)) * (b / (3 * a)) : rfl
      ... = (3 * a * b) / (2 * b * 3 * a) : by ring
      ... = 0.5 : by ring
  have h2 : (clubsuit 4 7 = 0.5) := by apply h1; norm_num
  have h3 : (clubsuit 8 (clubsuit 4 7) = 0.5) := by rw [h2, h1]; norm_num
  rw [h3, h1]; norm_num
  sorry

end problem_solution_l528_528847


namespace diff_squares_example_l528_528952

theorem diff_squares_example :
  (311^2 - 297^2) / 14 = 608 :=
by
  -- The theorem statement directly follows from the conditions and question.
  sorry

end diff_squares_example_l528_528952


namespace Tori_needed_more_correct_answers_l528_528389

theorem Tori_needed_more_correct_answers (
  total_problems : ℕ,
  arithmetic_problems : ℕ,
  algebra_problems : ℕ,
  geometry_problems : ℕ,
  correct_arithmetic_percentage : ℝ,
  correct_algebra_percentage : ℝ,
  correct_geometry_percentage : ℝ,
  passing_percentage : ℝ
) (h1 : total_problems = 80)
  (h2 : arithmetic_problems = 15)
  (h3 : algebra_problems = 30)
  (h4 : geometry_problems = 35)
  (h5 : correct_arithmetic_percentage = 0.8)
  (h6 : correct_algebra_percentage = 0.4)
  (h7 : correct_geometry_percentage = 0.6)
  (h8 : passing_percentage = 0.65) :
  let correct_arithmetic := correct_arithmetic_percentage * ↑arithmetic_problems,
      correct_algebra := correct_algebra_percentage * ↑algebra_problems,
      correct_geometry := correct_geometry_percentage * ↑geometry_problems,
      total_correct := correct_arithmetic + correct_algebra + correct_geometry,
      required_to_pass := passing_percentage * ↑total_problems,
      additional_needed := required_to_pass - total_correct in
  additional_needed = 7 := by
  sorry

end Tori_needed_more_correct_answers_l528_528389


namespace prime_numbers_r_s_sum_l528_528112

theorem prime_numbers_r_s_sum (p q r s : ℕ) (hp : Fact (Nat.Prime p)) (hq : Fact (Nat.Prime q)) 
  (hr : Fact (Nat.Prime r)) (hs : Fact (Nat.Prime s)) (h1 : p < q) (h2 : q < r) (h3 : r < s) 
  (eqn : p * q * r * s + 1 = 4^(p + q)) : r + s = 274 :=
by
  sorry

end prime_numbers_r_s_sum_l528_528112


namespace find_a_in_triangle_l528_528788

theorem find_a_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 3)
  (h2 : C = Real.pi / 3)
  (h3 : Real.sin B = 2 * Real.sin A)
  (h4 : a = 3) :
  a = Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l528_528788


namespace Enid_made_8_sweaters_l528_528694

def scarves : ℕ := 10
def sweaters_Aaron : ℕ := 5
def wool_per_scarf : ℕ := 3
def wool_per_sweater : ℕ := 4
def total_wool_used : ℕ := 82
def Enid_sweaters : ℕ := 8

theorem Enid_made_8_sweaters
  (scarves : ℕ)
  (sweaters_Aaron : ℕ)
  (wool_per_scarf : ℕ)
  (wool_per_sweater : ℕ)
  (total_wool_used : ℕ)
  (Enid_sweaters : ℕ)
  : Enid_sweaters = 8 :=
by
  sorry

end Enid_made_8_sweaters_l528_528694


namespace fernanda_total_time_to_finish_l528_528698

noncomputable def fernanda_days_to_finish_audiobooks
  (num_audiobooks : ℕ) (hours_per_audiobook : ℕ) (hours_listened_per_day : ℕ) : ℕ :=
num_audiobooks * hours_per_audiobook / hours_listened_per_day

-- Definitions based on the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Statement to prove
theorem fernanda_total_time_to_finish :
  fernanda_days_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 := 
sorry

end fernanda_total_time_to_finish_l528_528698


namespace find_extrema_l528_528941

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + (Real.sqrt 7 / 3) * Real.cos x * Real.sin x

theorem find_extrema : 
  ∃ x_max1 x_max2 x_min1 x_min2 : ℝ,
    0 ≤ x_max1 ∧ x_max1 ≤ 2*Real.pi ∧
    0 ≤ x_max2 ∧ x_max2 ≤ 2*Real.pi ∧
    0 ≤ x_min1 ∧ x_min1 ≤ 2*Real.pi ∧
    0 ≤ x_min2 ∧ x_min2 ≤ 2*Real.pi ∧
    f x_max1 = 7/6 ∧ f x_max2 = 7/6 ∧
    f x_min1 = -1/6 ∧ f x_min2 = -1/6 ∧
    (x_max1 = 69 * Real.pi / 180 + 18 * Real.pi / 180 / 60) ∧ 
    (x_max2 = 249 * Real.pi / 180 + 18 * Real.pi / 180 / 60) ∧
    (x_min1 = 159 * Real.pi / 180 + 18 * Real.pi / 180 / 60) ∧ 
    (x_min2 = 339 * Real.pi / 180 + 18 * Real.pi / 180 / 60) :=
begin
  sorry
end

end find_extrema_l528_528941


namespace value_of_constants_l528_528762

theorem value_of_constants (a b : ℝ) (h_a_pos : 0 < a) (h_a_not_one : a ≠ 1) (y : ℝ → ℝ)
  (h_y : ∀ x ∈ set.Icc (-3 / 2 : ℝ) 0, y x = b + a^(x^2 + 2 * x))
  (h_max : ∀ x ∈ set.Icc (-3 / 2 : ℝ) 0, y x ≤ 3)
  (h_min : ∀ x ∈ set.Icc (-3 / 2 : ℝ) 0, y x ≥ 5 / 2) :
  (a = 2 ∧ b = 2) ∨ (a = 2 / 3 ∧ b = 3 / 2) :=
sorry

end value_of_constants_l528_528762


namespace exists_n_stones_l528_528018

-- Define the grid, which is an 8x8 matrix
def grid : Matrix (Fin 8) (Fin 8) Bool := Matrix.fromFunction (λ _ _, false)

-- Define the condition that each row contains exactly 3 stones
def row_stones (r : Fin 8) : Finset (Fin 8) :=
  {c : Fin 8 | grid r c = true}

-- Define the condition that each column contains exactly 3 stones
def col_stones (c : Fin 8) : Finset (Fin 8) :=
  {r : Fin 8 | grid r c = true}
  
-- Assume the given conditions
variable (h1 : ∀ r : Fin 8, row_stones r).card = 3
variable (h2 : ∀ c : Fin 8, col_stones c).card = 3

theorem exists_n_stones :
  ∃ S : Finset (Fin 8 × Fin 8), S.card = 8 ∧
    (∀ (i j : Fin 8 × Fin 8), i ≠ j → i.1 ≠ j.1 ∧ i.2 ≠ j.2 ∧ (i ∈ S ∧ j ∈ S)) :=
by
  sorry

end exists_n_stones_l528_528018


namespace evaluate_expression_l528_528695

theorem evaluate_expression : (2^(2 + 1) - 4 * (2 - 1)^2)^2 = 16 :=
by
  sorry

end evaluate_expression_l528_528695


namespace total_arrangements_if_A_at_badminton_l528_528118

-- Define the background conditions
def games_duration := "September 23 to October 8, 2023"

-- Define the volunteers
inductive Volunteer
| A | B | C | D | E

-- Define the venues
inductive Venue
| Badminton | Swimming | Shooting | Gymnastics

-- Define the main condition where A goes to the badminton venue
def A_goes_to_badminton (a : Volunteer) (v : Venue) : Prop :=
a = Volunteer.A ∧ v = Venue.Badminton

-- The main theorem stating the total number of different arrangements
theorem total_arrangements_if_A_at_badminton :
  let volunteers := [Volunteer.A, Volunteer.B, Volunteer.C, Volunteer.D, Volunteer.E],
      venues := [Venue.Badminton, Venue.Swimming, Venue.Shooting, Venue.Gymnastics] in
  (∃ (arrangement : Volunteer → Venue), A_goes_to_badminton Volunteer.A Venue.Badminton ∧ 
    -- additional conditions to ensure each venue has at least one volunteer
    ∀ v, (∃ v' : Volunteer, arrangement v' = v) ∧ 
    -- each volunteer to only one venue
    ∀ v' v'', arrangement v' = arrangement v'' → v' = v'') →
  -- total number of arrangements is 60
  ∑ arr in (set_finite (volunteers_permutations volunteers venues)), 1 = 60 :=
sorry

end total_arrangements_if_A_at_badminton_l528_528118


namespace vanessa_phone_pictures_l528_528939

theorem vanessa_phone_pictures
  (C : ℕ) (P : ℕ) (hC : C = 7)
  (hAlbums : 5 * 6 = 30)
  (hTotal : 30 = P + C) :
  P = 23 := by
  sorry

end vanessa_phone_pictures_l528_528939


namespace triangle_area_l528_528017

theorem triangle_area (D E F L : Type) 
  [decidable_eq D] [decidable_eq E] [decidable_eq F] [decidable_eq L]
  (DE EF EL DL : ℝ)
  (DE_eq : DE = 15)
  (EL_eq : EL = 9)
  (EF_eq : EF = 17)
  (DL_eq : DL = (real.sqrt (DE^2 - EL^2)))
  (DL_is_altitude : DL = real.sqrt (15^2 - 9^2)) :
  (1/2) * EF * DL = 102 :=
by
  sorry

end triangle_area_l528_528017


namespace find_y_l528_528576

theorem find_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hrem : x % y = 5) (hdiv : (x : ℝ) / y = 96.2) : y = 25 := by
  sorry

end find_y_l528_528576


namespace value_spent_more_than_l528_528978

theorem value_spent_more_than (x : ℕ) (h : 8 * 12 + (x + 8) = 117) : x = 13 :=
by
  sorry

end value_spent_more_than_l528_528978


namespace puck_leaves_disk_l528_528991

noncomputable def time_to_leave_disk
  (R : ℝ) (n : ℝ)
  (h_R_pos : 0 < R) 
  (h_n_pos : 0 < n)
  : ℝ :=
  (Real.sqrt 15) / (2 * Real.pi * n)

theorem puck_leaves_disk
  (R n : ℝ)
  (h_R_pos : 0 < R)
  (h_n_pos : 0 < n) :
  let l := Real.sqrt (R ^ 2 - (R / 4) ^ 2)
  let V₀ := 2 * Real.pi * n * (R / 4)
  let T := l / V₀
  T = (Real.sqrt 15) / (2 * Real.pi * n) := 
begin
  -- Proof steps would go here
  sorry
end

end puck_leaves_disk_l528_528991


namespace tel_aviv_rain_probability_l528_528485

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l528_528485


namespace calculate_expression_l528_528663

theorem calculate_expression :
  500 * 996 * 0.0996 * 20 + 5000 = 997016 :=
by
  sorry

end calculate_expression_l528_528663


namespace number_of_mappings_that_satisfy_l528_528591

-- Definitions based on the conditions in the problem
def A := {-1, 0, 1}
def B := {-1, 0, 1}
def f : A → B := sorry  -- Placeholder for the mapping function

-- Theorem stating the required proof
theorem number_of_mappings_that_satisfy (h : ∀ f : A → B, (f ⟨-1, sorry⟩ ∈ A) → f (f ⟨-1, sorry⟩) < f ⟨1, sorry⟩) : ∃ n : ℕ, n = 9 := 
  sorry  -- Proof is omitted

end number_of_mappings_that_satisfy_l528_528591


namespace increasing_interval_f_l528_528910

noncomputable def f (x : ℝ) : ℝ := - (2 / 3) * x^3 + (3 / 2) * x^2 - x

theorem increasing_interval_f : ∀ x, (1 / 2) ≤ x ∧ x ≤ 1 → ∀ ε, ε > 0 → f' x x (ε, 0) :=
begin
  sorry
end

end increasing_interval_f_l528_528910


namespace cos_double_angle_l528_528303

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 5) = Real.sqrt 3 / 3) :
  Real.cos (2 * α + 2 * Real.pi / 5) = 1 / 3 :=
by
  sorry

end cos_double_angle_l528_528303


namespace burger_non_filler_percentage_l528_528597

theorem burger_non_filler_percentage (total_weight : ℕ) (filler_weight : ℕ) (h_total : total_weight = 180) (h_filler : filler_weight = 45) :
  ((total_weight - filler_weight) * 100 / total_weight) = 75 :=
by 
  rw [h_total, h_filler]
  simp
  sorry

end burger_non_filler_percentage_l528_528597


namespace value_of_c_l528_528782

theorem value_of_c (a b c d w x y z : ℕ) (primes : ∀ p ∈ [w, x, y, z], Prime p)
  (h1 : w < x) (h2 : x < y) (h3 : y < z) 
  (h4 : (w^a) * (x^b) * (y^c) * (z^d) = 660) 
  (h5 : (a + b) - (c + d) = 1) : c = 1 :=
by {
  sorry
}

end value_of_c_l528_528782


namespace sum_of_digits_9ab_l528_528423

noncomputable def a : ℤ := 3 * (10^1999 + 10^1998 + ... + 1)
noncomputable def b : ℤ := 7 * (10^1999 + 10^1998 + ... + 1)
noncomputable def ab : ℤ := 9 * a * b

-- Function to calculate the sum of digits of a number
def sum_of_digits (n : ℤ) : ℤ :=
  n.toString.data.map (λ c, c.to_nat - '0'.to_nat).foldl (· + ·) 0

theorem sum_of_digits_9ab : sum_of_digits ab = 18000 := by
  sorry

end sum_of_digits_9ab_l528_528423


namespace physics_letter_collections_l528_528401

theorem physics_letter_collections :
  let letters := ['P', 'H', 'Y', 'S', 'I', 'C', 'S']
  let vowels := ['I']
  let consonants := ['P', 'H', 'Y', 'C', 'S', 'S']
  let collections := 
    (finset.powerset (set.finset_of_vowels ∪ set.finset_of_consonants)).card
  collections = 11 :=
by
  sorry

end physics_letter_collections_l528_528401


namespace Andrew_spent_1395_dollars_l528_528719

-- Define the conditions
def cookies_per_day := 3
def cost_per_cookie := 15
def days_in_may := 31

-- Define the calculation
def total_spent := cookies_per_day * cost_per_cookie * days_in_may

-- State the theorem
theorem Andrew_spent_1395_dollars :
  total_spent = 1395 := 
by
  sorry

end Andrew_spent_1395_dollars_l528_528719


namespace truth_prob_l528_528004

-- Define the probabilities
def prob_A := 0.80
def prob_B := 0.60
def prob_C := 0.75

-- The problem statement
theorem truth_prob :
  prob_A * prob_B * prob_C = 0.27 :=
by
  -- Proof would go here
  sorry

end truth_prob_l528_528004


namespace no_tilable_6x6_no_uncut_lines_tilable_mn_no_uncut_lines_tilable_6x8_no_uncut_lines_l528_528179

-- Problem (a)
theorem no_tilable_6x6_no_uncut_lines :
  ¬ ∃ (tiling : Set (Set (ℤ × ℤ))),
    (∀ (d : ℤ × ℤ), d ∈ tiling → ∃ x1 x2 y, d = ((x1, y), (x2, y)) ∨ d = ((x1, y - 1), (x2, y - 1))) ∧
    ∀ line ∈ {l | ∃ x, {y | (x, y) ∈ tiling}.card > 1 ∨ ∃ y, {x | (x, y) ∈ tiling}.card > 1},
    False :=
sorry

-- Problem (b)
theorem tilable_mn_no_uncut_lines (m n : ℕ) (h1 : m > 6) (h2 : n > 6) (h3 : even (m * n)) :
  ∃ (tiling : Set (Set (ℤ × ℤ))),
    (∀ (d : ℤ × ℤ), d ∈ tiling → ∃ x1 x2 y, d = ((x1, y), (x2, y)) ∨ d = ((x1, y - 1), (x2, y - 1))) ∧
    ∀ line ∈ {l | ∃ x, {y | (x, y) ∈ tiling}.card > 1 ∨ ∃ y, {x | (x, y) ∈ tiling}.card > 1},
    False :=
sorry

-- Problem (c)
theorem tilable_6x8_no_uncut_lines :
  ∃ (tiling : Set (Set (ℤ × ℤ))),
    (∀ (d : ℤ × ℤ), d ∈ tiling → ∃ x1 x2 y, d = ((x1, y), (x2, y)) ∨ d = ((x1, y - 1), (x2, y - 1))) ∧
    ∀ line ∈ {l | ∃ x, {y | (x, y) ∈ tiling}.card > 1 ∨ ∃ y, {x | (x, y) ∈ tiling}.card > 1},
    False :=
sorry

end no_tilable_6x6_no_uncut_lines_tilable_mn_no_uncut_lines_tilable_6x8_no_uncut_lines_l528_528179


namespace factorize_x4_plus_16_l528_528283

theorem factorize_x4_plus_16: ∀ (x : ℝ), x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l528_528283


namespace average_score_is_correct_l528_528876

-- Define the given conditions
def numbers_of_students : List ℕ := [12, 28, 40, 35, 20, 10, 5]
def scores : List ℕ := [95, 85, 75, 65, 55, 45, 35]

-- Function to calculate the total score
def total_score (scores numbers : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ a b => a * b) scores numbers)

-- Calculate the average percent score
def average_percent_score (total number_of_students : ℕ) : ℕ :=
  total / number_of_students

-- Prove that the average percentage score is 70
theorem average_score_is_correct :
  average_percent_score (total_score scores numbers_of_students) 150 = 70 :=
by
  sorry

end average_score_is_correct_l528_528876


namespace derivative_log_base_3_l528_528757

noncomputable def f (x : ℝ) := log x / log 3

theorem derivative_log_base_3 (x : ℝ) (hx : x > 0) : deriv f x = 1 / (x * log 3) :=
by
  have : f x = log x / log 3 := rfl
  sorry

end derivative_log_base_3_l528_528757


namespace domain_of_function_l528_528689

theorem domain_of_function :
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ↔ (1 - x ≥ 0 ∧ x ≠ 0) :=
by
  sorry

end domain_of_function_l528_528689


namespace rain_probability_tel_aviv_l528_528495

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l528_528495


namespace smallest_integer_in_range_l528_528556

theorem smallest_integer_in_range :
  ∃ (n : ℕ), n > 1 ∧ n % 3 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ 131 ≤ n ∧ n ≤ 170 :=
by
  sorry

end smallest_integer_in_range_l528_528556


namespace find_tricias_age_l528_528532

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l528_528532


namespace main_proof_l528_528520

noncomputable theory

open_locale big_operators

-- Definitions given in the problem
def seq_a (a : ℕ → ℕ) := ∀ n, n ≥ 2 → a (n+1) = 2 * a n - a (n-1)
def init_a (a : ℕ → ℕ) := (a 4 = 4) ∧ (∑ i in finset.range 8, a (i+1) = 36)
def sum_a (a : ℕ → ℕ) (S : ℕ → ℕ) := ∀ n, S n = ∑ i in finset.range n, a (i+1)

-- Conditions for b_n and its property to be geometric
def seq_b (a b : ℕ → ℕ) :=
  ∀ n, (b 1 * a (2 * n - 1) + ∑ k in finset.range (n - 1) \{1}, (b k * a (2 * n + 1 - 2 * k)) + b n * a 1 = 3 * (2^n - 1) - 2 * a n)

def is_geometric (b : ℕ → ℕ) (r : ℕ) := ∀ n, b (n+1) = r * b n

def special_set (a b : ℕ → ℕ) (s : set (ℕ× ℕ)) :=
  ∀ m p, (m, p) ∈ s ↔ ((a m) / (b m) = 3 * (a p) / (b p)) ∧ m ∈ ℕ ∧ p ∈ ℕ

-- The theorem for the statements derived from the proof problem
theorem main_proof :
  ∃ a b S, 
    seq_a a ∧ 
    init_a a ∧ 
    sum_a a S ∧ 
    (∀ n, a n = n) ∧ 
    seq_b a b ∧ 
    is_geometric b 2 ∧ 
    special_set a b ({(6, 8)} : set (ℕ× ℕ)) :=
begin
  sorry,
end

end main_proof_l528_528520


namespace sphere_always_circular_cross_section_l528_528547

-- Define the geometric bodies
structure Cone :=
(radius : ℝ)
(height : ℝ)

structure Cylinder :=
(radius : ℝ)
(height : ℝ)

structure Sphere :=
(radius : ℝ)

structure FrustumOfCone :=
(smaller_radius : ℝ)
(larger_radius : ℝ)
(height : ℝ)

-- Define what it means for a shape to have a circular cross-section given any plane intersection
def always_circular_cross_section (G : Type) := 
  ∀ (P : Plane) (intersection : Set), intersection_with_plane G P = intersection → (is_circular intersection)

-- Define the Plane (assuming Plane can be a predefined structure within the library)
structure Plane :=
(normal_vector : ℝ → ℝ → ℝ) -- Just a placeholder definition for the plane

-- State the theorem to be proved
theorem sphere_always_circular_cross_section (S : Sphere) : 
  always_circular_cross_section Sphere :=
sorry

end sphere_always_circular_cross_section_l528_528547


namespace fixed_point_and_max_distance_eqn_l528_528764

-- Define line l1
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y - 8 = 0

-- Define line l2 parallel to l1 passing through origin
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y = 0

-- Define line y = x
def line_y_eq_x (x y : ℝ) : Prop :=
  y = x

-- Define line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop :=
  x + y = 0

theorem fixed_point_and_max_distance_eqn :
  (∀ m : ℝ, l1 m 2 2) ∧ (∀ m : ℝ, (l2 m 2 2 → false)) →
  (∃ x y : ℝ, l2 m x y ∧ line_x_plus_y_eq_0 x y) :=
by sorry

end fixed_point_and_max_distance_eqn_l528_528764


namespace angle_C_is_60_l528_528379

theorem angle_C_is_60 
  (A B C : ℝ) 
  (h1 : tan A + tan B + real.sqrt 3 = real.sqrt 3 * tan A * tan B)
  (h2 : A + B + C = 180) :
  C = 60 :=
sorry

end angle_C_is_60_l528_528379


namespace find_angle_between_vectors_l528_528767

noncomputable def theta_angle (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
  let CA := (A.1 - C.1, A.2 - C.2, A.3 - C.3)
  let dot_product := AB.1 * CA.1 + AB.2 * CA.2 + AB.3 * CA.3
  let AB_norm := real.sqrt (AB.1 ^ 2 + AB.2 ^ 2 + AB.3 ^ 2)
  let CA_norm := real.sqrt (CA.1 ^ 2 + CA.2 ^ 2 + CA.3 ^ 2)
  let cos_theta := dot_product / (AB_norm * CA_norm)
  real.acos cos_theta * (180 / real.pi)

theorem find_angle_between_vectors :
  theta_angle (1, 1, 1) (-1, 0, 4) (2, -2, 3) = 120 :=
by
  sorry

end find_angle_between_vectors_l528_528767


namespace system_of_equations_correct_l528_528586

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l528_528586


namespace Tricia_is_five_years_old_l528_528537

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l528_528537


namespace find_a0_a_sum_l528_528770

-- Given conditions
def e_2x_series (a : ℕ → ℝ) (x : ℝ) : ℝ := ∑ n in Finset.range 1000, a n * x^n

axiom differentiable_f_g {f g : ℝ → ℝ} (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) : 
  (∀ x, f x = g x) → (∀ x, deriv f x = deriv g x)

axiom taylor_exp_2x : ∀ (x : ℝ), e_2x_series (λ n, match n with | 0 => 1 | _ => n) x = Real.exp (2 * x)

-- Proof Problem
theorem find_a0_a_sum :
  (a0_eq : (λ (a : ℕ → ℝ) n, a n) 0 = 1) ∧
  (a_sum_eq : ∑ n in Finset.range 10, (λ (a : ℕ → ℝ) n, (a n / (n * a (n-1)))) = 20 / 11) :=
  sorry

end find_a0_a_sum_l528_528770


namespace find_x_plus_2y_sq_l528_528542

theorem find_x_plus_2y_sq (x y : ℝ) 
  (h : 8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1) : 
  x + 2 * y^2 = 1 / 2 :=
sorry

end find_x_plus_2y_sq_l528_528542


namespace sum_of_all_possible_values_of_g_25_l528_528850

def f (x : ℝ) : ℝ := 3 * x^2 - 2
def g (y : ℝ) : ℝ := if y = f 3 then 13 else if y = f (-3) then 7 else 0 

theorem sum_of_all_possible_values_of_g_25 : g 25 = 20 := 
by
  sorry

end sum_of_all_possible_values_of_g_25_l528_528850


namespace sum_of_coeffs_binomial_expansion_l528_528397

theorem sum_of_coeffs_binomial_expansion :
  let f := (2 * x - 3 * y)^9 in
  (f.eval (1, 1) = -1) :=
sorry

end sum_of_coeffs_binomial_expansion_l528_528397


namespace proof_problem_1_proof_problem_2_l528_528928

open Set

variable (U : Set ℝ := Set.univ) -- Universal set ℝ
variable (A : Set ℝ := { x | 3 ≤ x ∧ x < 8 }) -- Set A
variable (B : Set ℝ := { x | 2 < x ∧ x ≤ 6 }) -- Set B

theorem proof_problem_1 : 
  A ∩ B = { x | 3 ≤ x ∧ x ≤ 6 } ∧
  A ∪ B = { x | 2 < x ∧ x < 8 } ∧
  (U \ A) ∩ (U \ B) = { x | x ≤ 2 } ∪ { x | 8 ≤ x } := 
by 
  sorry

variable (C : ℝ → Set ℝ := λ a, { x | a < x }) -- Set C

theorem proof_problem_2 (a : ℝ) : 
  A ⊆ C a → a < 3 := 
by 
  sorry

end proof_problem_1_proof_problem_2_l528_528928


namespace jesse_stamps_total_l528_528407

theorem jesse_stamps_total (a e t : ℕ) (h1 : e = 333) (h2 : e = 3 * a) : t = e + a → t = 444 :=
by
  intros h3
  rw [h1] at h3
  rw [h2] at h3
  sorry

end jesse_stamps_total_l528_528407


namespace circle_through_A_and_B_tangent_to_S_l528_528677

open Classical

noncomputable def inversion_point (A B : Point) (r : ℝ) : Point := 
  sorry -- Placeholder for the actual inversion transformation definition

noncomputable def inversion_circle (A : Point) (S : Circle) (r : ℝ) : Circle := 
  sorry -- Placeholder for the actual circle inversion definition

theorem circle_through_A_and_B_tangent_to_S 
  (A B : Point) (S : Circle) : 
  (A ∈ S ∧ B ∈ S) → False ∨
  ∃! (C : Circle), 
    A ∈ C ∧ B ∈ C ∧ Tangent C S ∨
  (let B_inv := inversion_point A B 1 in
   let S_inv := inversion_circle A S 1 in
   B_inv ∈ S_inv → 
     ∃! (C : Circle), 
       A ∈ C ∧ B ∈ C ∧ Tangent C S) ∨
  (let B_inv := inversion_point A B 1 in
   let S_inv := inversion_circle A S 1 in
   B_inv ∉ S_inv → 
     ∃ (C1 C2 : Circle), 
       C1 ≠ C2 ∧
       A ∈ C1 ∧ B ∈ C1 ∧ Tangent C1 S ∧
       A ∈ C2 ∧ B ∈ C2 ∧ Tangent C2 S) ∨
  (let B_inv := inversion_point A B 1 in
   let S_inv := inversion_circle A S 1 in
   B_inv ∉ S_inv ∧ (∃ (C : Circle), Tangent C S → False)): sorry

end circle_through_A_and_B_tangent_to_S_l528_528677


namespace union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l528_528277

open Set

variables {α : Type*} (A B C : Set α)

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := sorry
theorem inter_comm : A ∩ B = B ∩ A := sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := sorry

-- Idempotence
theorem union_idem : A ∪ A = A := sorry
theorem inter_idem : A ∩ A = A := sorry

-- De Morgan's Laws
theorem de_morgan_union : compl (A ∪ B) = compl A ∩ compl B := sorry
theorem de_morgan_inter : compl (A ∩ B) = compl A ∪ compl B := sorry

end union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l528_528277


namespace max_S_min_S_l528_528281

def is_valid_sum (x1 x2 x3 x4 x5 : ℕ) : Prop :=
  x1 + x2 + x3 + x4 + x5 = 2006

def S (x1 x2 x3 x4 x5 : ℕ) : ℕ :=
  x1 * x2 + x1 * x3 + x1 * x4 + x1 * x5 + x2 * x3 + x2 * x4 + x2 * x5 +
  x3 * x4 + x3 * x5 + x4 * x5

def max_values (x1 x2 x3 x4 x5 : ℕ) : Prop :=
  x1 = 402 ∧ x2 = 401 ∧ x3 = 401 ∧ x4 = 401 ∧ x5 = 401

def min_values (x1 x2 x3 x4 x5 : ℕ) : Prop :=
  x1 = 402 ∧ x2 = 402 ∧ x3 = 402 ∧ x4 = 400 ∧ x5 = 400

def condition_abs_diff (x1 x2 x3 x4 x5 : ℕ) : Prop :=
  |x1 - x2| ≤ 2 ∧ |x1 - x3| ≤ 2 ∧ |x1 - x4| ≤ 2 ∧ |x1 - x5| ≤ 2 ∧
  |x2 - x3| ≤ 2 ∧ |x2 - x4| ≤ 2 ∧ |x2 - x5| ≤ 2 ∧
  |x3 - x4| ≤ 2 ∧ |x3 - x5| ≤ 2 ∧ |x4 - x5| ≤ 2

theorem max_S (x1 x2 x3 x4 x5 : ℕ) :
  is_valid_sum x1 x2 x3 x4 x5 → max_values x1 x2 x3 x4 x5 →
  S x1 x2 x3 x4 x5 = S 402 401 401 401 401 :=
by sorry

theorem min_S (x1 x2 x3 x4 x5 : ℕ) :
  is_valid_sum x1 x2 x3 x4 x5 → condition_abs_diff x1 x2 x3 x4 x5 →
  min_values x1 x2 x3 x4 x5 →
  S x1 x2 x3 x4 x5 = S 402 402 402 400 400 :=
by sorry

end max_S_min_S_l528_528281


namespace miles_built_first_day_l528_528999

def highway_extension (current_length new_length miles_first_day miles_needed : ℕ) : ℕ := do
  let x := miles_first_day
  let y := 3 * x
  let total_built := x + y
  let extension := new_length - current_length
  let built := extension - miles_needed
  total_built

theorem miles_built_first_day : 
  ∀ (current_length new_length miles_needed : ℕ),
  highway_extension current_length new_length 50 miles_needed = 200 → 
  50 = 50 :=  
by
  intros current_length new_length miles_needed h,
  rw [highway_extension] at h,
  sorry

end miles_built_first_day_l528_528999


namespace tricia_age_l528_528535

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l528_528535


namespace percent_defective_shipped_l528_528809

theorem percent_defective_shipped {P D S : ℕ} (hP : P = 100) (hD : D = 10) (hS : S = 0.5) :
  (S / D) * 100 = 5 := by
  sorry

end percent_defective_shipped_l528_528809


namespace quadratic_zero_count_in_interval_l528_528000

open Real

theorem quadratic_zero_count_in_interval (a : ℝ) (h : a > 3) :
  let f := λ x : ℝ, x^2 - a*x + 1 in
  ∃! x ∈ Ioo 0 2, f x = 0 := sorry

end quadratic_zero_count_in_interval_l528_528000


namespace Jake_read_pages_initially_l528_528824

-- Definitions based on conditions
variables (book_chapters book_total_pages : ℕ) (pages_read_later total_pages_read : ℕ)

-- Define the values based on the problem description
def book_chapters : ℕ := 8
def book_total_pages : ℕ := 95
def pages_read_later : ℕ := 25
def total_pages_read : ℕ := 62

-- The statement to be proved
theorem Jake_read_pages_initially : ∃ x : ℕ, x + 25 = 62 :=
by sorry

end Jake_read_pages_initially_l528_528824


namespace total_rainfall_november_l528_528797

theorem total_rainfall_november :
  (∃ (rain_per_day_first15 : ℕ) (double_rain_per_day_last15 : ℕ),
    (rain_per_day_first15 = 4) ∧
    (double_rain_per_day_last15 = 2 * rain_per_day_first15) ∧
    let total_first15 := 15 * rain_per_day_first15 in
    let total_last15 := 15 * double_rain_per_day_last15 in
    total_first15 + total_last15 = 180) :=
begin
  sorry
end

end total_rainfall_november_l528_528797


namespace vector_combination_solution_l528_528872

noncomputable def a : ℝ × ℝ × ℝ := (1, 1, 1)
noncomputable def b : ℝ × ℝ × ℝ := (3, -2, 4)
noncomputable def c : ℝ × ℝ × ℝ := (5, 0, -5)
noncomputable def result_vec : ℝ × ℝ × ℝ := (-3, 6, 9)

theorem vector_combination_solution :
  ∃ p q r : ℝ, result_vec = (p * a.1 + q * b.1 + r * c.1,
                              p * a.2 + q * b.2 + r * c.2,
                              p * a.3 + q * b.3 + r * c.3)
  ∧ (p, q, r) = (42 / 11, -6 / 11, -69 / 55) := by
  sorry

end vector_combination_solution_l528_528872


namespace center_square_side_length_is_40_l528_528142

-- Define the side length of the large square
def large_square_side_length : ℝ := 120

-- Define the total area of the large square
def large_square_area : ℝ := large_square_side_length * large_square_side_length

-- Define the area fraction occupied by each L-shaped region
def l_shaped_region_fraction : ℝ := 2 / 9

-- Define the number of L-shaped regions
def num_l_shaped_regions : ℝ := 4

-- Define the total area occupied by the L-shaped regions
def l_shaped_regions_total_area : ℝ := num_l_shaped_regions * l_shaped_region_fraction * large_square_area

-- Define the remaining area which is the area of the center square
def center_square_area : ℝ := large_square_area - l_shaped_regions_total_area

-- Define the side length of the center square
def center_square_side_length : ℝ := Real.sqrt center_square_area

-- The theorem to prove that the side length of the center square is 40 inches
theorem center_square_side_length_is_40 : center_square_side_length = 40 := by
  sorry

end center_square_side_length_is_40_l528_528142


namespace union_of_M_and_N_l528_528193

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l528_528193


namespace find_t_l528_528361

def vector (α : Type*) := (α × α)

variables (a b : vector ℝ)

def dot_product (v1 v2 : vector ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2

theorem find_t
  (a := (1 : ℝ, 2 : ℝ))
  (b := (4 : ℝ, 3 : ℝ))
  (h : dot_product a (t * (1, 2) + b) = 0) :
  t = -2 := 
sorry

end find_t_l528_528361


namespace max_edges_triangle_free_max_edges_no_k4_l528_528182

-- Define a graph with 30 vertices and no triangles
def triangle_free_graph (G : SimpleGraph (Fin 30)) : Prop :=
  ∀ (v w x : Fin 30), G.Adj v w → G.Adj w x → G.Adj v x → v = x ∨ v = w ∨ w = x

-- Define a graph with 30 vertices and no complete subgraph of 4 vertices
def no_k4_graph (G : SimpleGraph (Fin 30)) : Prop :=
  ∀ (v₁ v₂ v₃ v₄ : Fin 30),
    G.Adj v₁ v₂ → G.Adj v₂ v₃ → G.Adj v₃ v₄ → G.Adj v₁ v₃ → G.Adj v₁ v₄ → G.Adj v₂ v₄ → false

-- Part (a): Proof of maximum number of edges in a triangle-free graph with 30 vertices
theorem max_edges_triangle_free (G : SimpleGraph (Fin 30)) (h : triangle_free_graph G) :
  G.edge_finset.card ≤ 225 :=
sorry

-- Part (b): Proof of maximum number of edges in a 30-vertex graph without K_4
theorem max_edges_no_k4 (G : SimpleGraph (Fin 30)) (h : no_k4_graph G) :
  G.edge_finset.card ≤ 300 :=
sorry

end max_edges_triangle_free_max_edges_no_k4_l528_528182


namespace weights_system_l528_528583

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l528_528583


namespace part_a_part_b_part_c_min_val_part_d_circle_l528_528728

def z1 : Complex := ⟨2, 3⟩
def z2 (m : ℝ) : Complex := ⟨m, -1⟩

theorem part_a (m : ℝ) : (z1 / z2(m)).im = 0 → m = -2 / 3 :=
sorry

theorem part_b (m : ℝ) : (conj (z1 * z2(m)) = ⟨2 * m + 3, - (3 * m - 2)⟩) :=
sorry

theorem part_c_min_val (m : ℝ) : ∀ m, Complex.abs (z1 - z2(m)) ≥ 4 :=
sorry

theorem part_d_circle (z : Complex) : Complex.abs (z - z1) = 1 → z.re = 2 ∧ z.im = 3 :=
sorry

end part_a_part_b_part_c_min_val_part_d_circle_l528_528728


namespace inverse_is_correct_l528_528508

-- Definitions
def original_proposition (n : ℤ) : Prop := n < 0 → n ^ 2 > 0
def inverse_proposition (n : ℤ) : Prop := n ^ 2 > 0 → n < 0

-- Theorem stating the inverse
theorem inverse_is_correct : 
  (∀ n : ℤ, original_proposition n) → (∀ n : ℤ, inverse_proposition n) :=
by
  sorry

end inverse_is_correct_l528_528508


namespace solve_fractional_equation_l528_528469

noncomputable def fractional_equation (x : ℝ) : Prop :=
  (2 - x) / (x - 3) + 1 / (3 - x) = 1

theorem solve_fractional_equation :
  ∀ x : ℝ, x ≠ 3 → fractional_equation x ↔ x = 2 :=
by
  intros x h
  unfold fractional_equation
  sorry

end solve_fractional_equation_l528_528469


namespace inequality_a_b_l528_528417

theorem inequality_a_b (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
    a / (b + 1) + b / (a + 1) ≤ 1 :=
  sorry

end inequality_a_b_l528_528417


namespace tan_ratio_l528_528369

open Real

theorem tan_ratio (x y : ℝ) (h1 : sin x / cos y + sin y / cos x = 2) (h2 : cos x / sin y + cos y / sin x = 4) : 
  tan x / tan y + tan y / tan x = 2 :=
sorry

end tan_ratio_l528_528369


namespace largest_consecutive_sum_l528_528554

theorem largest_consecutive_sum (n : ℕ) (a : ℕ) 
  (h_sum : ∑ i in finset.range n, (a + i) = 30) 
  (h_pos : 0 < a) : n ≤ 5 :=
sorry

end largest_consecutive_sum_l528_528554


namespace distance_after_3_minutes_l528_528634

-- Define the given speeds and time interval
def speed_truck : ℝ := 65 -- in km/h
def speed_car : ℝ := 85 -- in km/h
def time_minutes : ℝ := 3 -- in minutes

-- The equivalent time in hours
def time_hours : ℝ := time_minutes / 60

-- Calculate the distances travelled by the truck and the car
def distance_truck : ℝ := speed_truck * time_hours
def distance_car : ℝ := speed_car * time_hours

-- Define the distance between the truck and the car
def distance_between : ℝ := distance_car - distance_truck

-- Theorem: The distance between the truck and car after 3 minutes is 1 km.
theorem distance_after_3_minutes : distance_between = 1 := by
  sorry

end distance_after_3_minutes_l528_528634


namespace maximum_expression_value_is_1989_l528_528729

noncomputable def maximum_expression_value (l : List ℕ) : ℕ :=
  l.headD 0 - l.drop 1.headD 0 - l.drop 2.headD 0 - l.drop 3.headD 0 - l.drop 4.headD 0 -->
  l.drop 5.headD 0 - l.drop 6.headD 0 - l.drop 7.headD 0 - l.drop 8.headD 0 - l.drop 9.headD 0 -->
  l.drop 10.headD 0 - l.drop 11.headD 0 - l.drop 12.headD 0 - l.drop 13.headD 0 - l.drop 14.headD 0 -->
  l.drop 15.headD 0 - l.drop 16.headD 0 - l.drop 17.headD 0 - l.drop 18.headD 0 - l.drop 19.headD 0 -->
  l.drop 20.headD 0 - l.drop 21.headD 0 - l.drop 22.headD 0 - l.drop 23.headD 0 - l.drop 24.headD 0 -->
  l.drop 25.headD 0 - l.drop 26.headD 0 - l.drop 27.headD 0 - l.drop 28.headD 0 - l.drop 29.headD 0 -->
  l.drop 30.headD 0 - l.drop 31.headD 0 - l.drop 32.headD 0 - l.drop 33.headD 0 - l.drop 34.headD 0 -->
  l.drop 35.headD 0 - l.drop 36.headD 0 - l.drop 37.headD 0 - l.drop 38.headD 0 - l.drop 39.headD 0 -->
  l.drop 40.headD 0 - l.drop 41.headD 0 - l.drop 42.headD 0 - l.drop 43.headD 0 - l.drop 44.headD 0 -->
  l.drop 45.headD 0 - l.drop 46.headD 0 - l.drop 47.headD 0 - l.drop 48.headD 0 - l.drop 49.headD 0 -->
  l.drop 50.headD 0 - l.drop 51.headD 0 - l.drop 52.headD 0 - l.drop 53.headD 0 - l.drop 54.headD 0

theorem maximum_expression_value_is_1989 (l : List ℕ) (h₁ : l.perm (List.range' 1 1990)) :
  maximum_expression_value l = 1989 :=
by
  sorry

end maximum_expression_value_is_1989_l528_528729


namespace area_of_circle_with_diameter_10_l528_528708

-- Definitions from the conditions
def diameter : ℝ := 10
def radius : ℝ := diameter / 2

-- Lean Statement (area of a circle with diameter 10 meters is approximately 78.54 square meters)
theorem area_of_circle_with_diameter_10 : 
  ∃ (π : ℝ), |π - real.pi| < 0.001 → |π * radius^2 - 78.54| < 0.1 :=
by
  let pi := 3.14159 -- approximation of π
  let area := pi * (radius ^ 2)
  have h : |area - 78.54| < 0.1
  sorry

end area_of_circle_with_diameter_10_l528_528708


namespace smallest_integer_remainder_l528_528947

theorem smallest_integer_remainder (a : ℤ) (h1 : a ≡ 6 [MOD 8]) (h2 : a ≡ 5 [MOD 9]) :
  a = 14 :=
begin
  sorry
end

end smallest_integer_remainder_l528_528947


namespace car_capacities_rental_plans_l528_528362

-- Define the capacities for part 1
def capacity_A : ℕ := 3
def capacity_B : ℕ := 4

theorem car_capacities (x y : ℕ) (h₁ : 2 * x + y = 10) (h₂ : x + 2 * y = 11) : 
  x = capacity_A ∧ y = capacity_B := by
  sorry

-- Define the valid rental plans for part 2
def valid_rental_plan (a b : ℕ) : Prop :=
  3 * a + 4 * b = 31

theorem rental_plans (a b : ℕ) (h : valid_rental_plan a b) : 
  (a = 1 ∧ b = 7) ∨ (a = 5 ∧ b = 4) ∨ (a = 9 ∧ b = 1) := by
  sorry

end car_capacities_rental_plans_l528_528362


namespace prob_D_l528_528090

variable (Ω : Type) [ProbabilitySpace Ω]

-- Event definitions
variable (A B C : Event Ω)
variable (P : Probability Ω)

-- Probability conditions
axiom prob_A : P A = 0.65
axiom prob_B : P B = 0.2
axiom prob_C : P C = 0.1

-- Event D definition
def D : Event Ω := B ∪ C

-- Theorem statement
theorem prob_D : P D = 0.3 :=
by
  rw [D, ProbabilityUnion]
  rw [prob_B, prob_C]
  exact (add_right_eq_self.mpr (by norm_num : 0.2 + 0.1 = 0.3))
  exact disjoint_of_subset_right sorry sorry  -- assuming that the events are mutually exclusive
  sorry  -- assuming required properties for union and probability measures

end prob_D_l528_528090


namespace expression_square_l528_528880

theorem expression_square (a b c d : ℝ) :
  (2*a + b + 2*c - d)^2 - (3*a + 2*b + 3*c - 2*d)^2 - (4*a + 3*b + 4*c - 3*d)^2 + (5*a + 4*b + 5*c - 4*d)^2 =
  (2*(a + b + c - d))^2 := 
sorry

end expression_square_l528_528880


namespace num_positive_four_digit_integers_of_form_xx75_l528_528774

theorem num_positive_four_digit_integers_of_form_xx75 : 
  ∃ n : ℕ, n = 90 ∧ ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → (∃ x: ℕ, x = 1000 * a + 100 * b + 75 ∧ 1000 ≤ x ∧ x < 10000) → n = 90 :=
sorry

end num_positive_four_digit_integers_of_form_xx75_l528_528774


namespace average_speed_correct_l528_528144

-- Define the conditions
def distance_first_hour := 90 -- in km
def distance_second_hour := 30 -- in km
def time_first_hour := 1 -- in hours
def time_second_hour := 1 -- in hours

-- Define the total distance and total time
def total_distance := distance_first_hour + distance_second_hour
def total_time := time_first_hour + time_second_hour

-- Define the average speed
def avg_speed := total_distance / total_time

-- State the theorem to prove the average speed is 60
theorem average_speed_correct :
  avg_speed = 60 := 
by 
  -- Placeholder for the actual proof
  sorry

end average_speed_correct_l528_528144


namespace find_fifth_result_l528_528971

theorem find_fifth_result (a : ℕ → ℤ) (h1 : (∑ i in finset.range 11, a i) / 11 = 42) 
    (h2 : (∑ i in finset.range 5, a i) / 5 = 49)
    (h3 : (∑ i in finset.range 7, a (i + 4)) / 7 = 52) : 
    a 4 = 147 := 
by sorry

end find_fifth_result_l528_528971


namespace polynomial_a5_coefficient_l528_528775

noncomputable theory

theorem polynomial_a5_coefficient :
  let p := (X^2 - 2*X + 2) ^ 5 in
  p.coeff 5 = -592 :=
by
  sorry

end polynomial_a5_coefficient_l528_528775


namespace log2_T_l528_528051

noncomputable def T : ℝ :=
  ∑ k in (Finset.range (1012 + 1)).filter (λ k, even k), (Nat.choose 1012 k : ℝ)

lemma T_value : T = 2^1011 := sorry

theorem log2_T : Real.logb 2 T = 1011 :=
by
  rw [T_value]
  rw [Real.logb_pow (by norm_num : 2 ≠ 1) (by norm_num : 2 > 0)]
  norm_num

end log2_T_l528_528051


namespace cos_double_angle_l528_528776

theorem cos_double_angle (x : ℝ) (h : 2 * sin x + cos (π / 2 - x) = 1) : cos (2 * x) = 7 / 9 :=
by
  sorry

end cos_double_angle_l528_528776


namespace expression_correct_l528_528960

variable (a b x y : ℝ)

def ax_by_eq (a x b y : ℝ) :=
  ax + by = 7

def ax2_by2_eq (a x b y : ℝ) :=
  ax^2 + by^2 = 49

def ax3_by3_eq (a x b y : ℝ) :=
  ax^3 + by^3 = 133

def ax4_by4_eq (a x b y : ℝ) :=
  ax^4 + by^4 = 406

theorem expression_correct :
  ax_by_eq a x b y →
  ax2_by2_eq a x b y →
  ax3_by3_eq a x b y →
  ax4_by4_eq a x b y →
  2014 * (x + y - x * y) - 100 * (a + b) = 6889.33 :=
by
  sorry

end expression_correct_l528_528960


namespace total_weight_under_total_weight_of_bags_l528_528524

theorem total_weight_under (deviations : list ℤ) (std_weight : ℤ) (n_bags : ℕ) (sum_deviation : ℤ) :
  deviations = [-6, -3, -1, -2, 7, 3, 4, -3, -2, 1] →
  std_weight = 150 →
  n_bags = 10 →
  sum_deviation = list.sum deviations →
  sum_deviation = -2 :=
by
  intros _ _ _ _
  sorry

theorem total_weight_of_bags (sum_deviation : ℤ) (std_weight : ℤ) (n_bags : ℕ) (total_weight : ℤ) :
  sum_deviation = -2 →
  std_weight = 150 →
  n_bags = 10 →
  total_weight = (n_bags * std_weight) + sum_deviation →
  total_weight = 1498 :=
by
  intros _ _ _ _
  sorry

end total_weight_under_total_weight_of_bags_l528_528524


namespace expand_polynomial_l528_528280

variable (x : ℝ)

theorem expand_polynomial :
  2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 :=
by
  sorry

end expand_polynomial_l528_528280


namespace count_of_n_up_to_1000_l528_528714

theorem count_of_n_up_to_1000 : 
  (finset.Icc 1 1000).filter (λ n, ∃ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊6*x⌋ = n).card = 456 :=
begin
  sorry
end

end count_of_n_up_to_1000_l528_528714


namespace fractional_eq_solution_l528_528477

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l528_528477


namespace sum_distances_correct_l528_528676

-- Define points A, B, C, D, and P on the coordinate plane
def A := (0, 0)
def B := (10, 0)
def C := (7, 5)
def D := (2, 3)
def P := (5, 3)

-- Define the distance formula between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distances from P to A, B, C, D
def AP := distance A P
def BP := distance B P
def CP := distance C P
def DP := distance D P

-- Define the sum of these distances, expressed in the form m + n√p
def sum_distances : ℝ := AP + BP + CP + DP

-- Formulate the problem statement as a proof goal
theorem sum_distances_correct :
  ∃ m n p : ℕ, sum_distances = (m : ℝ) + (n : ℝ) * real.sqrt (p : ℝ) ∧ (m + n = 5) :=
sorry

end sum_distances_correct_l528_528676


namespace total_high_sulfur_samples_l528_528611

-- Define the conditions as given in the problem
def total_samples : ℕ := 143
def heavy_oil_freq : ℚ := 2 / 11
def light_low_sulfur_freq : ℚ := 7 / 13
def no_low_sulfur_in_heavy_oil : Prop := ∀ (x : ℕ), (x / total_samples = heavy_oil_freq) → false

-- Define total high-sulfur samples
def num_heavy_oil := heavy_oil_freq * total_samples
def num_light_oil := total_samples - num_heavy_oil
def num_light_low_sulfur_oil := light_low_sulfur_freq * num_light_oil
def num_light_high_sulfur_oil := num_light_oil - num_light_low_sulfur_oil

-- Now state that we need to prove the total number of high-sulfur samples
theorem total_high_sulfur_samples : num_light_high_sulfur_oil + num_heavy_oil = 80 :=
by
  sorry

end total_high_sulfur_samples_l528_528611


namespace radius_calculation_l528_528567

noncomputable def radius_of_circle (n : ℕ) : ℝ :=
if 2 ≤ n ∧ n ≤ 11 then
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61
else
  0  -- Outside the specified range

theorem radius_calculation (n : ℕ) (hn : 2 ≤ n ∧ n ≤ 11) :
  radius_of_circle n =
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61 :=
sorry

end radius_calculation_l528_528567


namespace bisectors_of_plane_angles_coincide_l528_528738

open EuclideanGeometry

-- Define the triangular pyramid and points
variables {A B C D K L M P Q : Point}

-- Define the spheres and their intersections
-- S₁ is the sphere passing through points A, B, C that intersects AD, BD, CD at K, L, M respectively
def sphere_S1 (A B C K L M: Point) : Prop :=
  ∀ P, Sphere P = ⟨{A, B, C}⟩ → (P = K ∨ P = L ∨ P = M)

-- S₂ is the sphere passing through points A, B, D that intersects AC, BC, DC at P, Q, M respectively
def sphere_S2 (A B D P Q M: Point) : Prop :=
  ∀ P, Sphere P = ⟨{A, B, D}⟩ → (P = P ∨ P = Q ∨ P = M)

-- Given conditions:
variables {S1_intersects : sphere_S1 A B C K L M}
variables {S2_intersects : sphere_S2 A B D P Q M}
variables {parallel_KL_PQ : KL_parallel_PQ K L P Q}

-- The problem statement: Prove that the bisectors of the planar angles ∠KMQ and ∠LMP coincide.
theorem bisectors_of_plane_angles_coincide :
  bisectors_coincide ∠KMQ ∠LMP :=
sorry

end bisectors_of_plane_angles_coincide_l528_528738


namespace part_a_l528_528867

def system_of_equations (x y z a : ℝ) := 
  (x - a * y = y * z) ∧ (y - a * z = z * x) ∧ (z - a * x = x * y)

theorem part_a (x y z : ℝ) : 
  system_of_equations x y z 0 ↔ (x = 0 ∧ y = 0 ∧ z = 0) 
  ∨ (∃ x, y = x ∧ z = 1) 
  ∨ (∃ x, y = -x ∧ z = -1) := 
  sorry

end part_a_l528_528867


namespace cheese_volume_difference_l528_528937

theorem cheese_volume_difference 
  (a b c : ℝ) 
  (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) :
  let V2 := a * b * c,
      V1 := (3/2 * a) * (4/5 * b) * (7/10 * c) in
  V2 = (21/17) * V1 :=
by
  sorry

end cheese_volume_difference_l528_528937


namespace more_whistles_sean_than_charles_l528_528461

def whistles_sean : ℕ := 223
def whistles_charles : ℕ := 128

theorem more_whistles_sean_than_charles : (whistles_sean - whistles_charles) = 95 :=
by
  sorry

end more_whistles_sean_than_charles_l528_528461


namespace bankers_gain_is_270_l528_528902

-- Definitions of the conditions
def BD : ℝ := 1020
def r : ℝ := 12 / 100  -- converting percentage to decimal
def t : ℝ := 3

-- Definition of true discount (TD)
def TD : ℝ := BD / (1 + r * t)

-- Definition of banker's gain (BG)
def BG : ℝ := BD - TD

-- Theorem proving the banker's gain is Rs. 270
theorem bankers_gain_is_270 : BG = 270 := by
  sorry

end bankers_gain_is_270_l528_528902


namespace solve_fractional_equation_l528_528472

theorem solve_fractional_equation : 
  ∀ x : ℝ, x = 2 → (2 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by
  intro x hx
  rw hx
  simp
  sorry

end solve_fractional_equation_l528_528472


namespace three_digit_integer_count_l528_528926

theorem three_digit_integer_count :
  {x : ℕ // 100 ≤ x ∧ x < 1000 ∧ (x % 6 = 2) ∧ (x % 9 = 5) ∧ (x % 11 = 7)}.card = 5 :=
by sorry

end three_digit_integer_count_l528_528926


namespace wayne_needs_30_more_blocks_l528_528940

def initial_blocks : ℕ := 9
def additional_blocks : ℕ := 6
def total_blocks : ℕ := initial_blocks + additional_blocks
def triple_total : ℕ := 3 * total_blocks

theorem wayne_needs_30_more_blocks :
  triple_total - total_blocks = 30 := by
  sorry

end wayne_needs_30_more_blocks_l528_528940


namespace find_coefficients_l528_528724

theorem find_coefficients (a : ℕ → ℤ) (h : ∀ x : ℝ, (1 - 2 * x) ^ 2016 = ∑ i in finset.range 2017 , a i * (x - 2) ^ i) :
  (∑ i in finset.range 2017, (i * (-2)^i) * a i) = 4032 :=
by
  sorry

end find_coefficients_l528_528724


namespace distance_between_truck_and_car_l528_528632

noncomputable def speed_truck : ℝ := 65
noncomputable def speed_car : ℝ := 85
noncomputable def time : ℝ := 3 / 60

theorem distance_between_truck_and_car : 
  let Distance_truck := speed_truck * time,
      Distance_car := speed_car * time in
  Distance_car - Distance_truck = 1 :=
by {
  sorry
}

end distance_between_truck_and_car_l528_528632


namespace joan_initial_time_l528_528831

-- Define the given conditions
def time_on_piano : ℕ := 30
def time_writing_music : ℕ := 25
def time_reading_history : ℕ := 38
def time_left_for_exerciser : ℕ := 27

-- State the theorem
theorem joan_initial_time : 
  time_on_piano + time_writing_music + time_reading_history + time_left_for_exerciser = 120 :=
  by
    simp [time_on_piano, time_writing_music, time_reading_history, time_left_for_exerciser]
    rfl

end joan_initial_time_l528_528831


namespace line_equation_passing_point_l528_528209

theorem line_equation_passing_point
  (a T : ℝ) 
  (θ : ℝ) 
  (h : ℝ := (2 * T / a))
  (m : ℝ := Real.tan θ)
  (ha : a > 0) 
  (hθ : θ ≠ 0)
  (hp : (a, 0))
  (hy : h = 2 * T / a)
  (hm : m = Real.tan θ) :
  ∃ b c, b = m * a ∧ c = -a * m ∧ ∀ x y, y = m * x + c → m * x - y - a * m = 0 :=
by
  use [m, -a * m]
  split
  · exact hm ▸ rfl
  split
  · exact rfl
  intro x y hxy
  rw [hxy, hm, sub_eq_add_neg, add_assoc, add_neg_eq_zero]
  sorry

end line_equation_passing_point_l528_528209


namespace count_divisible_by_4_not_containing_4_l528_528645

/-
   Define a function that checks if a number contains the digit 4.
-/
def contains_digit_four (n : ℕ) : Bool :=
  n.digits 10 |> List.any (λ d => d = 4)

/-
   Define the main theorem statement to prove the count of numbers
   from 1 to 1000 that are divisible by 4 and do not contain the digit 4 is 162
-/
theorem count_divisible_by_4_not_containing_4 :
  let count := (List.range 1001).filter (λ n => n % 4 = 0 ∧ ¬contains_digit_four n)
  count.length = 162 :=
by
  sorry

end count_divisible_by_4_not_containing_4_l528_528645


namespace infinite_series_evaluation_l528_528255

noncomputable def infinite_series_term (n : ℕ) : ℝ :=
  (n^4 + 4*n^3 + 3*n^2 + 11*n + 15) / (2^n * (n^4 + 9))

theorem infinite_series_evaluation : 
  (∑' n in finset.Ico 3 ∞, infinite_series_term n) = 1 / 4 := 
by sorry

end infinite_series_evaluation_l528_528255


namespace num_valid_sequences_10_transformations_l528_528220

/-- Define the transformations: 
    L: 90° counterclockwise rotation,
    R: 90° clockwise rotation,
    H: reflection across the x-axis,
    V: reflection across the y-axis. -/
inductive Transformation
| L | R | H | V

/-- Define a function to get the number of valid sequences of transformations
    that bring the vertices E, F, G, H back to their original positions.-/
def countValidSequences : ℕ :=
  56

/-- The theorem to prove that the number of valid sequences
    of 10 transformations resulting in the identity transformation is 56. -/
theorem num_valid_sequences_10_transformations : 
  countValidSequences = 56 :=
sorry

end num_valid_sequences_10_transformations_l528_528220


namespace sum_gcd_lcm_is_159_l528_528951

-- Definitions for GCD and LCM for specific values
def gcd_45_75 := Int.gcd 45 75
def lcm_48_18 := Int.lcm 48 18

-- The proof problem statement
theorem sum_gcd_lcm_is_159 : gcd_45_75 + lcm_48_18 = 159 := by
  sorry

end sum_gcd_lcm_is_159_l528_528951


namespace plant_hormone_related_activities_l528_528909

-- Definitions of the conditions as variables
inductive PlantActivity
| fruit_ripening (description : String)
| leaves_turning_yellow (description : String)
| fruit_shedding (description : String)
| co2_fixation (description : String)
| topping_cotton_plants (description : String)
| absorption_mineral_elements (description : String)

open PlantActivity

-- Proof statement that {①, ③, ⑤} are related to plant hormones
theorem plant_hormone_related_activities :
  (fruit_ripening "Ethylene promotes fruit ripening") ∧
  (fruit_shedding "Abscisic acid promotes fruit shedding") ∧
  (topping_cotton_plants "Topping cotton plants reduces the concentration of auxin at the side buds, preventing flower and fruit shedding") :=
by
  sorry -- proof is skipped

end plant_hormone_related_activities_l528_528909


namespace find_angle_B_l528_528382

theorem find_angle_B (a b c : ℝ) (h : b^2 = a^2 + a * c + c^2) : 
  ∠B = 120 :=
by
  sorry

end find_angle_B_l528_528382


namespace find_a_l528_528805

-- Define the conditions given in the problem. 
def curve_polar_eq (ρ θ a : ℝ) : Prop := ρ * (Real.sin θ)^2 = 2 * a * (Real.cos θ)
def line_param_eq (t : ℝ) : (ℝ × ℝ) := (-2 + (Real.sqrt 2)/2 * t, -4 + (Real.sqrt 2)/2 * t)
def dist (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2)

-- Define the Cartesian coordinate equation of the curve.
def curve_cart_eq (x y a : ℝ) : Prop := y^2 = 2 * a * x

-- Define the standard equation of the line.
def line_std_eq (x y : ℝ) : Prop := x - y - 2 = 0

-- Main theorem to prove
theorem find_a (a : ℝ) (t1 t2 : ℝ) 
  (h_curve: ∀ ρ θ, curve_polar_eq ρ θ a)
  (h_line: ∀ t, (line_std_eq (line_param_eq t).fst (line_param_eq t).snd))
  (h_dist: dist (line_param_eq t1) (line_param_eq t2) = 2 * Real.sqrt 10) :
  a = 1 :=
by
  sorry

end find_a_l528_528805


namespace line_perpendicular_passing_P0_l528_528502

-- Definitions of the conditions
variables {A B C x y x0 y0 : ℝ}
def P0 := (x0, y0)
def line1 := λ x y : ℝ, A * x + B * y + C = 0

-- The statement we need to prove
theorem line_perpendicular_passing_P0 :
  (B * x - A * y - B * x0 + A * y0 = 0) :=
  sorry

end line_perpendicular_passing_P0_l528_528502


namespace fraction_length_EF_of_GH_l528_528450

-- Given the position relationships of points E and F on segment GH
theorem fraction_length_EF_of_GH 
  (E F G H : Type) 
  [LinearOrder E] [LinearOrder F] [LinearOrder G] [LinearOrder H] -- Assuming these types as linearly ordered
  (length : G -> ℝ) -- Function giving the length of segments
  (h_GE_EH : length G = 3 * length E)
  (h_GF_FH : length F = 8 * length H) :
  length E + length F - length G = 5/36 * length E + length F :=
sorry

end fraction_length_EF_of_GH_l528_528450


namespace constant_function_continuous_l528_528049

noncomputable theory

open Function

theorem constant_function_continuous {f : ℝ → ℝ} (h1 : Continuous f) (h2 : ∀ x y : ℝ, f (x + 2*y) = 2 * f x * f y) :
  ∃ c : ℝ, ∀ x, f x = c :=
by
  sorry

end constant_function_continuous_l528_528049


namespace find_xyz_squares_l528_528308

theorem find_xyz_squares (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end find_xyz_squares_l528_528308


namespace incorrect_statement_A_l528_528955

structure Quadrilateral where
  a b c d : ℝ
  perpendicular_diagonals : Prop
  bisecting_diagonals : Prop

def is_square (Q : Quadrilateral) : Prop :=
  (Q.a = Q.b) ∧ (Q.b = Q.c) ∧ (Q.c = Q.d) ∧ (Q.perpendicular_diagonals) ∧ (Q.bisecting_diagonals)
  
def is_rhombus (Q : Quadrilateral) : Prop :=
  (Q.a = Q.b) ∧ (Q.b = Q.c) ∧ (Q.c = Q.d) ∧ (Q.perpendicular_diagonals) ∧ (Q.bisecting_diagonals)
  
theorem incorrect_statement_A :
  ∀ (Q : Quadrilateral), (Q.perpendicular_diagonals ∧ Q.bisecting_diagonals) → ¬is_square Q :=
by
  intros Q h
  -- Placeholder for proof, assuming conditions but showing not a square.
  sorry

end incorrect_statement_A_l528_528955


namespace johnny_returned_cans_l528_528048

open Real

theorem johnny_returned_cans :
  ∀ (x : ℤ),
  let avg_cost_all : ℝ := 36.5
      avg_cost_remaining : ℝ := 30
      avg_cost_returned : ℝ := 49.5
      total_cans : ℤ := 6
      total_cost : ℝ := total_cans * avg_cost_all
      cost_remaining := (total_cans - x : ℝ) * avg_cost_remaining
      cost_returned := (x : ℝ) * avg_cost_returned
  in total_cost = cost_remaining + cost_returned → x = 2 :=
by
  intros x avg_cost_all avg_cost_remaining avg_cost_returned total_cans total_cost cost_remaining cost_returned h
  -- Proof steps would go here
  sorry

end johnny_returned_cans_l528_528048


namespace root_in_interval_l528_528194

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.log x - 5

theorem root_in_interval : ∃ c ∈ Ioo 1 2, f c = 0 := by
  have h1 : f 1 = -2 := by
    simp [f]
  have h2 : f 2 > 0 := by
    simp [f]
    linarith
  sorry

end root_in_interval_l528_528194


namespace non_adjacent_books_arrangement_l528_528149

noncomputable def countNonAdjacentArrangements : Nat :=
  5! - 2 * (4! * 2!) + (3! * 2! * 2!)

theorem non_adjacent_books_arrangement :
  countNonAdjacentArrangements = 48 :=
by
  unfold countNonAdjacentArrangements
  calc
    5! - 2 * (4! * 2!) + (3! * 2! * 2!)
        = 120 - 2 * 48 + 24
      : by norm_num
    ... = 48
      : by norm_num

end non_adjacent_books_arrangement_l528_528149


namespace polynomial_inequality_l528_528455

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
sorry

end polynomial_inequality_l528_528455


namespace find_A_max_min_l528_528238

def is_coprime_with_36 (n : ℕ) : Prop := Nat.gcd n 36 = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let rest := n / 10
  d * 10^7 + rest

theorem find_A_max_min (B : ℕ) 
  (h1 : B > 77777777) 
  (h2 : is_coprime_with_36 B) : 
  move_last_digit_to_first B = 99999998 ∨ 
  move_last_digit_to_first B = 17777779 := 
by
  sorry

end find_A_max_min_l528_528238


namespace female_democrats_count_l528_528150

theorem female_democrats_count (F M : ℕ) (h1 : F + M = 750) 
  (h2 : F / 2 ≠ 0) (h3 : M / 4 ≠ 0) 
  (h4 : F / 2 + M / 4 = 750 / 3) : F / 2 = 125 :=
by
  sorry

end female_democrats_count_l528_528150


namespace evaluate_expression_l528_528278

theorem evaluate_expression : 
  (125^(1/3 : ℝ)) * (64^(-1/6 : ℝ)) * (81^(1/4 : ℝ)) = 15 / 2 :=
by
  have h1 : 125 = 5^3 := rfl
  have h2 : 64 = 2^6 := rfl
  have h3 : 81 = 3^4 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end evaluate_expression_l528_528278


namespace volume_water_needed_l528_528893

noncomputable def radius_sphere : ℝ := 0.5
noncomputable def radius_cylinder : ℝ := 1
noncomputable def height_cylinder : ℝ := 2

theorem volume_water_needed :
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 :=
by
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3)
  let total_volume_spheres := 4 * volume_sphere
  let volume_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  have h : volume_cylinder - total_volume_spheres = (4 * Real.pi) / 3 := sorry
  exact h

end volume_water_needed_l528_528893


namespace parallel_planes_l528_528841

open EuclideanGeometry

variables {Point : Type} [EuclideanPlane Point] {A B C D M N K : Point}

-- Define the condition that A, B, C, D are four points not lying in the same plane
def not_coplanar (A B C D : Point) : Prop :=
  ¬ coplanar {A, B, C, D}

-- Define the midpoints
def midpoint (A D : Point) : Point := (A + D) / 2
def M := midpoint A D
def N := midpoint B D
def K := midpoint C D

-- Define the given condition: Points A, B, C, D are not coplanar and M, N, K are midpoints
axiom points_not_coplanar : not_coplanar A B C D

-- Theorem statement: The plane passing through points M, N, K is parallel to the plane passing through points A, B, C
theorem parallel_planes : plane_parallel (plane_of_points M N K) (plane_of_points A B C) :=
sorry

end parallel_planes_l528_528841


namespace factorial_identity_l528_528246

theorem factorial_identity : 8! - 7 * 7! - 7! = 0 :=
by
  sorry

end factorial_identity_l528_528246


namespace solve_fractional_equation_l528_528470

theorem solve_fractional_equation : 
  ∀ x : ℝ, x = 2 → (2 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by
  intro x hx
  rw hx
  simp
  sorry

end solve_fractional_equation_l528_528470


namespace tel_aviv_rain_probability_l528_528488

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l528_528488


namespace complex_point_in_first_quadrant_l528_528339

theorem complex_point_in_first_quadrant (z : ℂ) (hz : z = 2 + 1 * complex.I) : 
  z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_in_first_quadrant_l528_528339


namespace sum_of_D_coordinates_l528_528084

-- Define points as tuples for coordinates (x, y)
structure Point :=
  (x : ℤ)
  (y : ℤ)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def pointD : Point :=
  let C := Point.mk 11 5
  let N := Point.mk 5 9
  let x := 2 * N.x - C.x
  let y := 2 * N.y - C.y
  Point.mk x y

theorem sum_of_D_coordinates : 
  let D := pointD
  D.x + D.y = 12 := by
  sorry

end sum_of_D_coordinates_l528_528084


namespace age_of_hospital_l528_528022

theorem age_of_hospital (grant_current_age : ℕ) (future_ratio : ℚ)
                        (grant_future_age : grant_current_age + 5 = 30)
                        (hospital_age_ratio : future_ratio = 2 / 3) :
                        (grant_current_age = 25) → 
                        (grant_current_age + 5 = future_ratio * (grant_current_age + 5 + 5)) →
                        (grant_current_age + 5 + 5 - 5 = 40) :=
by
  sorry

end age_of_hospital_l528_528022


namespace find_k_l528_528360

-- Defining the vectors
def a (k : ℝ) : ℝ × ℝ := (k, -2)
def b : ℝ × ℝ := (2, 2)

-- Condition 1: a + b is not the zero vector
def non_zero_sum (k : ℝ) := (a k).1 + b.1 ≠ 0 ∨ (a k).2 + b.2 ≠ 0

-- Condition 2: a is perpendicular to a + b
def perpendicular (k : ℝ) := (a k).1 * ((a k).1 + b.1) + (a k).2 * ((a k).2 + b.2) = 0

-- The theorem to prove
theorem find_k (k : ℝ) (cond1 : non_zero_sum k) (cond2 : perpendicular k) : k = 0 := 
sorry

end find_k_l528_528360


namespace range_of_f_l528_528704

noncomputable def f (x : ℝ) : ℝ := arctan x + arctan ((2 - x) / (2 + x))

theorem range_of_f :
  (∀ y, (∃ x, f x = y) ↔ (y = arctan 2 ∨ y = π/4)) :=
by sorry

end range_of_f_l528_528704


namespace total_handshakes_l528_528079

def handshakes (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_handshakes (n : ℕ) (h : n = 11) : handshakes n + n = 66 :=
by {
  subst h,
  unfold handshakes,
  norm_num,
  sorry
}

end total_handshakes_l528_528079


namespace tangent_DNC_in_rhombus_l528_528400

theorem tangent_DNC_in_rhombus 
  (A B C D : Type) [metric_space A]
  [decidable_eq A] [inhabited A]
  (h_rhombus : ∀ (a b c d : A), rhombus a b c d)
  (h_angle_A : angle A B C = 60)
  (N : A)
  (h_divide : divides_side_in_ratio A B N 2 1) :
  (tan (angle D N C) = sqrt (243 / 121)) :=
sorry

end tangent_DNC_in_rhombus_l528_528400


namespace rain_probability_tel_aviv_l528_528493

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l528_528493


namespace one_hundred_fiftieth_digit_l528_528549

theorem one_hundred_fiftieth_digit (r : ℚ) (block: ℕ -> ℕ)
  (h₀ : r = 4 / 37)
  (h₁ : ∀ n, block n = (n % 3 = 0 → 1) ∧
                 (n % 3 = 1 → 0) ∧
                 (n % 3 = 2 → 8)) :
  block 149 = 0 :=
by sorry

end one_hundred_fiftieth_digit_l528_528549


namespace solve_fractional_equation_l528_528467

noncomputable def fractional_equation (x : ℝ) : Prop :=
  (2 - x) / (x - 3) + 1 / (3 - x) = 1

theorem solve_fractional_equation :
  ∀ x : ℝ, x ≠ 3 → fractional_equation x ↔ x = 2 :=
by
  intros x h
  unfold fractional_equation
  sorry

end solve_fractional_equation_l528_528467


namespace find_y_l528_528002

theorem find_y (x y : ℚ) (h1 : x = 103) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 1106600) : y = 1085 / 1030 :=
by
  have h3 : (x^2 - 4 * x + 4) * x * y = 1106600 := by
    rw [← h2]
    ring
  have h4 : (x - 2)^2 * x * y = 1106600 := by
    rw [h1] at h3
    norm_num at h3
    exact h3
  have h5 : 101^2 * 103 * y = 1106600 := by
    rw [← h1] at h4
    norm_num at h4
    exact h4
  have h6 : 10201 * 103 * y = 1106600 := by
    rw pow_two at h5
    exact h5
  have h7 : 103 * y = (1106600 / 10201) := by
    rw [mul_comm, ← div_eq_mul_one_div]
    exact h6
  have h8 : 103 * y = 108.5 := by
    norm_num at h7
    exact h7
  have h9 : y = 108.5 / 103 := by
    rw [← h8]
    norm_num at h8
  exact h9

end find_y_l528_528002


namespace cyclist_return_trip_average_speed_l528_528969

theorem cyclist_return_trip_average_speed :
  let first_leg_distance := 12
  let second_leg_distance := 24
  let first_leg_speed := 8
  let second_leg_speed := 12
  let round_trip_time := 7.5
  let distance_to_destination := first_leg_distance + second_leg_distance
  let time_to_destination := (first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)
  let return_trip_time := round_trip_time - time_to_destination
  let return_trip_distance := distance_to_destination
  (return_trip_distance / return_trip_time) = 9 := 
by
  sorry

end cyclist_return_trip_average_speed_l528_528969


namespace median_of_list_l528_528944

def list := (list.range (2525 + 1)).map (fun x => x) ++ (list.range (2525 + 1)).map (fun x => x * x)

theorem median_of_list : 
  let sorted_list := list.qsort (Nat.lt);
  sorted_list.length = 5050 → 
  sorted_list.nth (2524) * sorted_list.nth (2525) = 2525.5 := 
by
  sorry

end median_of_list_l528_528944


namespace distance_between_truck_and_car_l528_528629

noncomputable def speed_truck : ℝ := 65
noncomputable def speed_car : ℝ := 85
noncomputable def time : ℝ := 3 / 60

theorem distance_between_truck_and_car : 
  let Distance_truck := speed_truck * time,
      Distance_car := speed_car * time in
  Distance_car - Distance_truck = 1 :=
by {
  sorry
}

end distance_between_truck_and_car_l528_528629


namespace rain_probability_tel_aviv_l528_528491

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l528_528491


namespace seven_step_paths_l528_528120

-- Problem conditions and definitions
def is_black_square (row : ℕ) (col : ℕ) : Prop :=
  if row % 2 = 0 then col % 2 = 1 else col % 2 = 0

def is_white_square (row : ℕ) (col : ℕ) : Prop :=
  if row % 2 = 0 then col % 2 = 0 else col % 2 = 1

-- Function to check if a move is valid, given the current step number
def valid_move (step : ℕ) (from_row from_col to_row to_col : ℕ) : Prop :=
  (to_row = from_row + 1) ∧ (abs (to_col - from_col) ≤ 1) ∧ 
  (if step % 2 = 0 then is_white_square to_row to_col else true)

-- Definition to count the number of valid 7-step paths
def count_valid_paths (P Q : ℕ × ℕ) : ℕ :=
  sorry  -- Implementation of counting paths has been omitted

theorem seven_step_paths (P Q : ℕ × ℕ)
  (start_black : is_black_square P.1 P.2)
  (end_black : is_black_square Q.1 Q.2)
  (steps : ℕ = 7) :
  count_valid_paths P Q = 56 :=
sorry

end seven_step_paths_l528_528120


namespace total_animals_in_jacobs_flock_l528_528878

-- Define the conditions of the problem
def one_third_of_animals_are_goats (total goats : ℕ) : Prop := 
  3 * goats = total

def twelve_more_sheep_than_goats (goats sheep : ℕ) : Prop :=
  sheep = goats + 12

-- Define the main theorem to prove
theorem total_animals_in_jacobs_flock : 
  ∃ total goats sheep : ℕ, one_third_of_animals_are_goats total goats ∧ 
                           twelve_more_sheep_than_goats goats sheep ∧ 
                           total = 36 := 
by
  sorry

end total_animals_in_jacobs_flock_l528_528878


namespace radius_sphere_tangent_eq_l528_528143

-- Given conditions
variables (a b : ℝ) (h : a > 0) (k : b > 0)

-- Define function to calculate the radius.
def radius_of_sphere (a b : ℝ) : ℝ :=
  a / (2 * real.sqrt (3 * b^2 - a^2))

-- The theorem to be proven.
theorem radius_sphere_tangent_eq :
  radius_of_sphere a b = a / (2 * real.sqrt (3 * b^2 - a^2)) :=
sorry -- Proof to be completed.

end radius_sphere_tangent_eq_l528_528143


namespace mushrooms_collected_l528_528082

variable (P V : ℕ)

theorem mushrooms_collected (h1 : P = (V * 100) / (P + V)) (h2 : V % 2 = 1) :
  P + V = 25 ∨ P + V = 300 ∨ P + V = 525 ∨ P + V = 1900 ∨ P + V = 9900 := by
  sorry

end mushrooms_collected_l528_528082


namespace number_of_biscuits_l528_528938

theorem number_of_biscuits (dough_length dough_width biscuit_length biscuit_width : ℕ)
    (h_dough : dough_length = 12) (h_dough_width : dough_width = 12)
    (h_biscuit_length : biscuit_length = 3) (h_biscuit_width : biscuit_width = 3)
    (dough_area : ℕ := dough_length * dough_width)
    (biscuit_area : ℕ := biscuit_length * biscuit_width) :
    dough_area / biscuit_area = 16 :=
by
  -- assume dough_area and biscuit_area are calculated from the given conditions
  -- dough_area = 144 and biscuit_area = 9
  sorry

end number_of_biscuits_l528_528938


namespace factorial_multiple_of_3_l528_528296

theorem factorial_multiple_of_3 (n : ℤ) (h : n ≥ 9) : 3 ∣ (n+1) * (n+3) :=
sorry

end factorial_multiple_of_3_l528_528296


namespace solve_problem_l528_528242

noncomputable def problem_conditions (D P N : ℝ) : Prop :=
  D = P + N ∧ (D - 7) = (P + N - 7) ∧ (P + 4) ∧ abs((P + N - 7) - (P + 4)) = 4

theorem solve_problem : ∀ D P N : ℝ, problem_conditions D P N → (N = 15 ∨ N = 7) → (15 * 7 = 105) :=
by
  intros
  sorry

end solve_problem_l528_528242


namespace erica_blank_question_count_l528_528147

variable {C W B : ℕ}

theorem erica_blank_question_count
  (h1 : C + W + B = 20)
  (h2 : 7 * C - 4 * W = 100) :
  B = 1 :=
by
  sorry

end erica_blank_question_count_l528_528147


namespace coloring_paths_in_grid_l528_528121

theorem coloring_paths_in_grid :
  let n := 9
  let k := 3
  factorial n / (factorial k * factorial k * factorial k) = 1680 :=
by
  let n := 9
  let k := 3
  have h1 : factorial n = 362880 := by simp
  have h2 : factorial k = 6 := by simp
  have h3 : 6 * 6 * 6 = 216 := by norm_num
  have h4 : 362880 / 216 = 1680 := by norm_num
  rw [h1, h2] at h4
  exact h4
  sorry

end coloring_paths_in_grid_l528_528121


namespace range_of_m_l528_528319

variable (f : ℝ → ℝ) (m : ℝ)

-- Define the properties of the function f
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def function_f_property (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x → f x = x - sin x

-- The main theorem to prove
theorem range_of_m (h_odd : odd_function f) 
                   (h_prop : function_f_property f) 
                   (inequality : ∀ t : ℝ, f (-4 * t) > f (2 * m + m * t^2)) : 
                   m < -Real.sqrt 2 :=
by
  sorry

end range_of_m_l528_528319


namespace penny_dime_halfdollar_same_probability_l528_528896

def probability_same_penny_dime_halfdollar : ℚ :=
  let total_outcomes := 2 ^ 5
  let successful_outcomes := 2 * 2 * 2
  successful_outcomes / total_outcomes

theorem penny_dime_halfdollar_same_probability :
  probability_same_penny_dime_halfdollar = 1 / 4 :=
by 
  sorry

end penny_dime_halfdollar_same_probability_l528_528896


namespace sum_of_areas_eq_l528_528140

noncomputable def radius (n : ℕ) : ℝ :=
  2 * (1/3 : ℝ)^(n-1)

noncomputable def area (n : ℕ) : ℝ :=
  π * (radius n)^2

theorem sum_of_areas_eq : 
    (∑' n, area n) = (9 / 2) * π := 
by
  sorry

end sum_of_areas_eq_l528_528140


namespace inequality_pos_reals_l528_528107

theorem inequality_pos_reals (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) : 
  (x^2 + 2) * (y^2 + 2) * (z^2 + 2) ≥ 9 * (x * y + y * z + z * x) :=
by
  sorry

end inequality_pos_reals_l528_528107


namespace average_goals_increase_l528_528995

theorem average_goals_increase (A : ℚ) (h1 : 4 * A + 2 = 4) : (4 / 5 - A) = 0.3 := by
  sorry

end average_goals_increase_l528_528995


namespace sam_average_speed_proof_l528_528459

noncomputable def sam_average_speed_last_30_minutes 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_segment_time : ℝ) 
  (first_segment_speed : ℝ) 
  (second_segment_time : ℝ) 
  (second_segment_speed : ℝ) 
  (last_segment_time : ℝ) : ℝ :=
let first_segment_distance := first_segment_speed * first_segment_time,
    second_segment_distance := second_segment_speed * second_segment_time,
    travelled_distance := first_segment_distance + second_segment_distance,
    remaining_distance := total_distance - travelled_distance
in remaining_distance / last_segment_time

theorem sam_average_speed_proof : 
  sam_average_speed_last_30_minutes 120 2 (2/3) 50 (5/6) 60 (1/2) = 220/3 := by
  sorry

end sam_average_speed_proof_l528_528459


namespace sin_ineq_l528_528561

theorem sin_ineq (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (hy : 0 ≤ y ∧ y ≤ π) : 
  sin (x + y) ≤ sin x + sin y := 
sorry

end sin_ineq_l528_528561


namespace fifteenth_digit_sum_1_7_1_11_l528_528550

-- Define the repeating sequence for 1/7
def repeating_seq_1_7 : ℕ → ℕ
| 0 := 0
| 1 := 1
| 2 := 4
| 3 := 2
| 4 := 8
| 5 := 5
| _ := repeating_seq_1_7 (n % 6)

-- Define the repeating sequence for 1/11
def repeating_seq_1_11 : ℕ → ℕ
| 0 := 0
| 1 := 9
| _ := repeating_seq_1_11 (n % 2)

-- Define function to calculate the nth digit after the decimal point of the sum
def sum_repeating_digit (n : ℕ) : ℕ :=
  (repeating_seq_1_7 n + repeating_seq_1_11 n) % 10

-- The statement to be proven: The 15th digit after the decimal point is 2
theorem fifteenth_digit_sum_1_7_1_11 : sum_repeating_digit 15 = 2 :=
by {
  -- The detailed proof goes here
  sorry
}

end fifteenth_digit_sum_1_7_1_11_l528_528550


namespace ratio_jacob_edward_l528_528406

-- Definitions and conditions
def brian_shoes : ℕ := 22
def edward_shoes : ℕ := 3 * brian_shoes
def total_shoes : ℕ := 121
def jacob_shoes : ℕ := total_shoes - brian_shoes - edward_shoes

-- Statement of the problem
theorem ratio_jacob_edward (h_brian : brian_shoes = 22)
                          (h_edward : edward_shoes = 3 * brian_shoes)
                          (h_total : total_shoes = 121)
                          (h_jacob : jacob_shoes = total_shoes - brian_shoes - edward_shoes) :
                          jacob_shoes / edward_shoes = 1 / 2 :=
by sorry

end ratio_jacob_edward_l528_528406


namespace BP_eq_AC_l528_528432

noncomputable def points_and_circles := 
let A, B, C, D, P : Point in
let Γ, Δ : Circle in
( A ∈ Γ ∧ B ∈ Γ ∧ C ∈ Γ ) ∧ -- Points A, B, C lie on the circle Γ
( tangent Δ AC A ) ∧         -- Circle Δ is tangent to AC at A
( D ∈ Γ ∧ D ∈ Δ ) ∧          -- Circle Δ meets Γ again at D
( P ∈ AB ∧ P ∈ Δ ) ∧         -- Circle Δ meets line AB at P
( between A B P ) ∧          -- Point A lies between points B and P
( dist A D = dist D P )      -- AD = DP

theorem BP_eq_AC 
(points_and_circles : points_and_circles) : 
distance B P = distance A C := 
sorry

end BP_eq_AC_l528_528432


namespace train_speed_l528_528223

theorem train_speed (length_train length_bridge : ℝ) (time : ℝ) (h_length_train : length_train = 250)
  (h_length_bridge : length_bridge = 150) (h_time : time = 41.142857142857146) :
  (length_train + length_bridge) / time * 3.6 = 35 :=
by
  have h_total_distance : length_train + length_bridge = 400, from
    sorry, -- proved from h_length_train and h_length_bridge
  have h_speed_m_s : (length_train + length_bridge) / time = 9.722222222222223, from
    sorry, -- calculated from h_total_distance and h_time
  calc
    (length_train + length_bridge) / time * 3.6
        = 9.722222222222223 * 3.6 : by rw [h_speed_m_s]
    ... = 35 : by norm_num

end train_speed_l528_528223


namespace alternating_sum_is_100_l528_528279

theorem alternating_sum_is_100 :
  ∑ i in finset.range 200, (if even i then - (i : ℕ) else i) = 100 :=
by
  sorry

end alternating_sum_is_100_l528_528279


namespace inscribed_square_ratio_l528_528615

theorem inscribed_square_ratio 
  (a b : ℝ) 
  (h₁ : ∃ (T : Triangle), T.is_right_triangle ∧ T.has_sides 6 8 10 ∧ T.square_inscribed_at_right_angle a)
  (h₂ : ∃ (T' : Triangle), T'.is_isosceles_right ∧ T'.has_legs 6 6 ∧ T'.square_inscribed_on_hypotenuse b)
  : a / b = Real.sqrt(2) / 3 :=
sorry

end inscribed_square_ratio_l528_528615


namespace current_women_count_l528_528968

variable (x : ℕ) -- Let x be the common multiplier.
variable (initial_men : ℕ := 4 * x)
variable (initial_women : ℕ := 5 * x)

-- Conditions
variable (men_after_entry : ℕ := initial_men + 2)
variable (women_after_leave : ℕ := initial_women - 3)
variable (current_women : ℕ := 2 * women_after_leave)
variable (current_men : ℕ := 14)

-- Theorem statement
theorem current_women_count (h : men_after_entry = current_men) : current_women = 24 := by
  sorry

end current_women_count_l528_528968


namespace p_plus_q_eq_933_l528_528214

-- Define the points and conditions
variable (p q : ℝ)
variable (fold_condition_1 : (1,3).1 + (5,1).1 = 3 * 2 ∧ (1,3).2 + (5,1).2 = 2 * 2)
variable (fold_condition_2 : (8,4).1 + p = 2 * 2 * q ∧ (8,4).2 + q = 2 * (2 * p - 4))

-- Main theorem statement proving p + q = 9.\overline{3}
theorem p_plus_q_eq_933 : p + q = 9.3333333333 :=
by
    have midpoint_eq : (3, 2) = ((1 + 5) / 2, (3 + 1) / 2), from sorry
    have fold_line_eq : (3, 2).2 = 2 * (3, 2).1 - 4, from sorry
    have p_eq : p = 16 - 2 * q, from sorry
    have midpoint_fold_eq : ( (8 + p) / 2, (4 + q) / 2 ) = (3, 2), from sorry
    have q_eq : q = 4 + p, from sorry
    have sum_eq : p + q = 9.3333333333, from sorry
    exact sum_eq

end p_plus_q_eq_933_l528_528214


namespace system_of_equations_correct_l528_528585

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l528_528585


namespace find_func_find_n_l528_528803

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 2)
def point_B : ℝ × ℝ := (1, 4)

-- Define the function of the form y = kx + b
def line_func (k b x : ℝ) : ℝ := k * x + b

-- Conditions
axiom k_nonzero : ∀ k : ℝ, k ≠ 0

-- Functions passing through points A and B
axiom pass_A : ∃ (k b : ℝ), line_func k b (fst point_A) = snd point_A
axiom pass_B : ∃ (k b : ℝ), line_func k b (fst point_B) = snd point_B

-- Prove analytical expression of the function
theorem find_func : ∃ (k b : ℝ), k = 1 ∧ b = 3 :=
sorry

-- Define the second function y = (1/2)x + n
def another_func (n x : ℝ) : ℝ := (1/2) * x + n 

-- Prove the given condition for function when x > 2

theorem find_n : ∃ (n : ℝ), ∀ x > 2, 5 < another_func n x ∧ another_func n x < line_func 1 3 x :=
sorry

end find_func_find_n_l528_528803


namespace domain_of_f_l528_528167

-- Defining the function f(x)
def f (x : ℝ) : ℝ := real.root 4 (x - 5) + real.root 3 (x - 4)

-- Proving that the domain of f is [5, ∞)
theorem domain_of_f : set_of (λ x : ℝ, ∃ y : ℝ, f x = y) = set.Ici 5 :=
by
  sorry

end domain_of_f_l528_528167


namespace average_of_five_integers_l528_528482

-- Definitions for the integers and their properties
variables {k m r s t : ℕ}

axiom h1 : k < m
axiom h2 : m < r
axiom h3 : r < s
axiom h4 : s < t
axiom h5 : t = 42
axiom h6 : r = 17

theorem average_of_five_integers : (k + m + r + s + t) / 5 = 26.6 := 
by sorry

end average_of_five_integers_l528_528482


namespace tricia_age_l528_528534

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l528_528534


namespace train_crossing_time_l528_528595

def time_to_cross_man (train_length : ℕ) (train_speed_kmh : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let time_seconds := train_length / train_speed_ms
  time_seconds

theorem train_crossing_time (train_length : ℕ) (train_speed_kmh : ℕ) :
  train_length = 200 → train_speed_kmh = 80 → time_to_cross_man train_length train_speed_kmh = 9 :=
by
  intros h_length h_speed
  rw [h_length, h_speed]
  unfold time_to_cross_man
  have h_speed_ms : (80 * 1000) / 3600 = 22 := by norm_num
  rw [h_speed_ms]
  norm_num
  exact rfl

end train_crossing_time_l528_528595


namespace total_games_in_season_l528_528575

theorem total_games_in_season (n : ℕ) (k : ℕ) (n_eq : n = 50) (k_eq : k = 4) :
  ((n * (n - 1)) / 2) * k = 4900 :=
by
  rw [n_eq, k_eq]
  norm_num
  sorry

end total_games_in_season_l528_528575


namespace students_standing_arrangement_l528_528023

/-- In how many different ways can four students stand in a straight line if two students refuse to 
    stand next to each other and one student insists on standing at either the first or last position in the line? -/
theorem students_standing_arrangement :
  let total_arrangements : ℕ := 24,  -- Total arrangements without restrictions is 4!
      position_preference_arrangements : ℕ := 12,  -- Considering the position preference
      block_arrangements : ℕ := 12,  -- Arrangements where the two students are together
      invalid_position_preferred_blocks : ℕ := 4   -- Invalid configurations with position preference
  in position_preference_arrangements - invalid_position_preferred_blocks = 8 :=
by
  sorry

end students_standing_arrangement_l528_528023


namespace intersection_empty_l528_528322

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l528_528322


namespace function_properties_l528_528355
-- Import the necessary library

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 4) + cos (2 * x + π / 4)

-- Define the simplified function (using trigonometric identities)
noncomputable def simplified_f (x : ℝ) : ℝ := sqrt 2 * cos (2 * x)

-- State the theorem to prove the function is even and its symmetry
theorem function_properties :
  (∀ x : ℝ, f x = simplified_f x) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x : ℝ, simplified_f (π / 2 - x) = simplified_f (π / 2 + x)) :=
by
  sorry

end function_properties_l528_528355


namespace parabola_equation_l528_528606

theorem parabola_equation (A B : ℝ × ℝ) (x₁ x₂ y₁ y₂ p : ℝ) :
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  x₁ + x₂ = (p + 8) / 2 →
  x₁ * x₂ = 4 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 45 →
  (y₁ = 2 * x₁ - 4) →
  (y₂ = 2 * x₂ - 4) →
  ((y₁^2 = 2 * p * x₁) ∧ (y₂^2 = 2 * p * x₂)) →
  (y₁^2 = 4 * x₁ ∨ y₂^2 = -36 * x₂) := 
by {
  sorry
}

end parabola_equation_l528_528606


namespace find_angle_l528_528709

theorem find_angle :
  ∃ (x : ℝ), (90 - x = 0.4 * (180 - x)) → x = 30 :=
by
  sorry

end find_angle_l528_528709


namespace z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l528_528754

variables (m : ℝ)

def z_re (m : ℝ) : ℝ := 2 * m^2 - 3 * m - 2
def z_im (m : ℝ) : ℝ := m^2 - 3 * m + 2

-- Part (Ⅰ) Question 1
theorem z_real_iff_m_1_or_2 (m : ℝ) :
  z_im m = 0 ↔ (m = 1 ∨ m = 2) :=
sorry

-- Part (Ⅰ) Question 2
theorem z_complex_iff_not_m_1_and_2 (m : ℝ) :
  ¬ (m = 1 ∨ m = 2) ↔ (m ≠ 1 ∧ m ≠ 2) :=
sorry

-- Part (Ⅰ) Question 3
theorem z_pure_imaginary_iff_m_neg_half (m : ℝ) :
  z_re m = 0 ∧ z_im m ≠ 0 ↔ (m = -1/2) :=
sorry

-- Part (Ⅱ) Question
theorem z_in_second_quadrant (m : ℝ) :
  z_re m < 0 ∧ z_im m > 0 ↔ -1/2 < m ∧ m < 1 :=
sorry

end z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l528_528754


namespace writer_born_on_sunday_l528_528898

-- Define the given conditions
def isLeapYear (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

-- Calculate the number of leap years between two given years
def countLeapYears (startYear endYear : ℕ) : ℕ :=
  finset.card (finset.filter isLeapYear (finset.range (endYear - startYear)))

def daysInYear (year : ℕ) : ℕ :=
  if isLeapYear year then 366 else 365

-- Calculate the total number of days between two dates
def totalDaysBetween (startYear endYear : ℕ) : ℕ :=
  (finset.range (endYear - startYear)).sum daysInYear

-- Define the problem statement
theorem writer_born_on_sunday :
  let daysBack := totalDaysBetween 1780 2020
      daysModulo := daysBack % 7
      startDay := 6  -- Since 2020-02-07 is Friday (Day 6 of week if starting from Sunday as Day 0)
  in
    (startDay + 7 - daysModulo) % 7 = 0 := -- Day 0 is Sunday
sorry

end writer_born_on_sunday_l528_528898


namespace solution_set_l528_528747

-- Define the function f as even, defined on ℝ, monotonically increasing on [0, +∞), and f(1/2) = 0
variables {f : ℝ → ℝ}
variables (h_even : ∀ x, f x = f (-x))
variables (h_domain : ∀ x, x ∈ ℝ)
variables (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
variable (h_zero : f (1/2) = 0)

-- State the main theorem
theorem solution_set (x : ℝ) : (f (x - 2) > 0) ↔ (x > 5/2 ∨ x < 3/2) :=
sorry

end solution_set_l528_528747


namespace fraction_still_missing_l528_528593

theorem fraction_still_missing (x : ℕ) (hx : x > 0) :
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = (1/9 : ℚ) :=
by
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  have h_fraction_still_missing : (x - remaining) / x = (1/9 : ℚ) := sorry
  exact h_fraction_still_missing

end fraction_still_missing_l528_528593


namespace juniors_score_l528_528385

-- Define the context and conditions
variables (n : ℕ) (average_class : ℝ) (average_seniors : ℝ) (average_juniors : ℝ)
variables (percentage_juniors percentage_seniors : ℝ)
variables (total_students total_score_juniors total_score_seniors : ℝ)

-- Set the given values
def percentage_juniors := 0.2
def percentage_seniors := 0.8
def average_class := 85
def average_seniors := 82
noncomputable def total_students := n
noncomputable def total_score := average_class * total_students
noncomputable def total_score_seniors := average_seniors * (percentage_seniors * total_students)
noncomputable def total_score_juniors := total_score - total_score_seniors
noncomputable def average_juniors := total_score_juniors / (percentage_juniors * total_students)

-- Prove the assertion
theorem juniors_score (h : n > 0) : average_juniors = 97 :=
by
  sorry

end juniors_score_l528_528385


namespace correct_divisor_l528_528795

theorem correct_divisor :
  ∃ D : ℕ, (let X := 12 * 56 in X = D * 32) ∧ D = 21 :=
by {
  let X := 12 * 56,
  use 21,
  split,
  {
    rw mul_comm,
    exact eq.symm (mul_eq_mul_right_iff.mpr (or.inl (by norm_num))),
  },
  {
    refl,
  }
}

end correct_divisor_l528_528795


namespace infinite_impossible_values_of_d_l528_528513

theorem infinite_impossible_values_of_d 
  (pentagon_perimeter square_perimeter : ℕ) 
  (d : ℕ) 
  (h1 : pentagon_perimeter = 5 * (d + ((square_perimeter) / 4)) )
  (h2 : square_perimeter > 0)
  (h3 : pentagon_perimeter - square_perimeter = 2023) :
  ∀ n : ℕ, n > 404 → ¬∃ d : ℕ, d = n :=
by {
  sorry
}

end infinite_impossible_values_of_d_l528_528513


namespace five_letter_words_start_end_same_count_l528_528826

theorem five_letter_words_start_end_same_count : 
  let alphabet_size := 26 in
  let valid_word_count := alphabet_size ^ 4 in
  valid_word_count = 456976 :=
by
  let alphabet_size := 26
  let valid_word_count := alphabet_size ^ 4
  calc
    valid_word_count = 456976 : by rfl

end five_letter_words_start_end_same_count_l528_528826


namespace number_of_squares_in_100th_diagram_l528_528016

theorem number_of_squares_in_100th_diagram :
  (∃ f : ℕ → ℕ, (f 0 = 1) ∧ (f 1 = 7) ∧ (f 2 = 19) ∧ (f 3 = 37) ∧ (∀ n, f n = 3 * n^2 + 3 * n + 1)) →
  (∃ n, n = f 100 ∧ n = 30301) :=
begin
  intro h,
  obtain ⟨f, h_f0, h_f1, h_f2, h_f3, h_fn⟩ := h,
  use f 100,
  split,
  { exact h_fn 100 },
  { refl }
end

end number_of_squares_in_100th_diagram_l528_528016


namespace probability_complement_l528_528931

theorem probability_complement (p : ℝ) (h : p = 0.997) : 1 - p = 0.003 :=
by
  rw [h]
  norm_num

end probability_complement_l528_528931


namespace largest_determinable_1986_l528_528577

-- Define main problem with conditions
def largest_determinable_cards (total : ℕ) (select : ℕ) : ℕ :=
  total - 27

-- Statement we need to prove
theorem largest_determinable_1986 :
  largest_determinable_cards 2013 10 = 1986 :=
by
  sorry

end largest_determinable_1986_l528_528577


namespace find_tricias_age_l528_528530

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l528_528530


namespace division_into_rectangles_l528_528026

theorem division_into_rectangles (figure : Type) (valid_division : figure → Prop) : (∃ ways, ways = 8) :=
by {
  -- assume given conditions related to valid_division using "figure"
  sorry
}

end division_into_rectangles_l528_528026


namespace missed_games_l528_528244

variable (total_games : ℕ) (attended_games : ℕ)
variable (total_eq_39 : total_games = 39)
variable (attended_eq_14 : attended_games = 14)

theorem missed_games : total_games - attended_games = 25 :=
by
  rw [total_eq_39, attended_eq_14]
  exact Nat.sub_self sorry  -- Nat.sub_self handles subtraction for natural numbers but require checking natural subtraction.
  exact sorry  -- Completing the proof manually.

end missed_games_l528_528244


namespace problem_l528_528511

theorem problem (p q : ℕ) (hpq : Nat.gcd p q = 1) (hposp : 0 < p) (hposq : 0 < q)
  (hsum : ∑ x in { x : ℝ // ∃ (w : ℤ), w = floor x ∧ ({x} : ℝ) = x - w ∧ w * (x - w) = a * x^2 + 35 }, x = 715) :
  p + q = 1234 := 
sorry

end problem_l528_528511


namespace brick_weight_l528_528205

theorem brick_weight (concrete stone total : ℝ) (h1 : concrete = 0.17) (h2 : stone = 0.5) (h3 : total = 0.83) :
  ∃ bricks : ℝ, bricks = 0.16 ∧ bricks = total - (concrete + stone) :=
by
  use total - (concrete + stone)
  split
  . sorry
  . sorry

end brick_weight_l528_528205


namespace inverse_corresponding_angles_of_congruent_triangles_is_false_l528_528912

theorem inverse_corresponding_angles_of_congruent_triangles_is_false :
  ¬ (∀ {A B C D E F : Type} {triangle1 : Triangle A B C} {triangle2 : Triangle D E F},
     (corresponding_angles_equal triangle1 triangle2 → triangle_congruent triangle1 triangle2)) :=
sorry

end inverse_corresponding_angles_of_congruent_triangles_is_false_l528_528912


namespace length_of_AD_l528_528589

theorem length_of_AD {A D M C : Type} [metric_space A] [metric_space D] [metric_space M] [metric_space C]
  (B C: D) (M : D) (h1 : dist B C = dist C B) (h2 : dist C B = dist B A) (h3 : dist C M = 7) :
  dist A D = 56/3 :=
by
  sorry

end length_of_AD_l528_528589


namespace no_ordered_triples_l528_528216

theorem no_ordered_triples (x y z : ℕ)
  (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) :
  x * y * z + 2 * (x * y + y * z + z * x) ≠ 2 * (2 * (x * y + y * z + z * x)) + 12 :=
by {
  sorry
}

end no_ordered_triples_l528_528216


namespace tel_aviv_rain_probability_l528_528489

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l528_528489


namespace scientific_notation_150_billion_l528_528092

theorem scientific_notation_150_billion : 150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_150_billion_l528_528092


namespace license_plate_difference_l528_528786

theorem license_plate_difference :
  (26^4 * 10^3 - 26^5 * 10^2 = -731161600) :=
sorry

end license_plate_difference_l528_528786


namespace smallest_u_n_l528_528312

theorem smallest_u_n  (n : ℕ) (hn : 0 < n) : 
  ∃ u_n : ℕ, (∀ d : ℕ, odd d → 
                (∀ k : ℕ, k ≤ 2n - 1 → 
                  (∃ m : ℕ, 
                   m < k →  (m < m + d → 
                   m + d ≤ 2n - 1 → m ≤ u_n → (m + d) ∣ (2n-1))) → 
                   1 ≤ k → 1 ≤ u_n → k = m + d → d ∣ (u_n))) :=
begin
  use 2n - 1,
  sorry
end

end smallest_u_n_l528_528312


namespace minimum_point_translated_graph_l528_528505

-- Define the original equation of the graph
def original_graph (x : ℝ) : ℝ := 2 * abs x + 1

-- Define the translation functions
def translate_left (p : (ℝ × ℝ)) (units : ℝ) : (ℝ × ℝ) :=
  (p.1 - units, p.2)

def translate_down (p : (ℝ × ℝ)) (units : ℝ) : (ℝ × ℝ) :=
  (p.1, p.2 - units)

-- Prove the coordinates of the new minimum point
theorem minimum_point_translated_graph : 
  let original_min := (0, original_graph 0) in
  let left_translation := translate_left original_min 4 in
  let final_translation := translate_down left_translation 2 in
  final_translation = (-4, -1) :=
by 
  sorry

end minimum_point_translated_graph_l528_528505


namespace calc_perimeter_l528_528573

noncomputable def width (w: ℝ) (h: ℝ) : Prop :=
  h = w + 10

noncomputable def cost (P: ℝ) (rate: ℝ) (total_cost: ℝ) : Prop :=
  total_cost = P * rate

noncomputable def perimeter (w: ℝ) (P: ℝ) : Prop :=
  P = 2 * (w + (w + 10))

theorem calc_perimeter {w P : ℝ} (h_rate : ℝ) (h_total_cost : ℝ)
  (h1 : width w (w + 10))
  (h2 : cost (2 * (w + (w + 10))) h_rate h_total_cost) :
  P = 2 * (w + (w + 10)) →
  h_total_cost = 910 →
  h_rate = 6.5 →
  w = 30 →
  P = 140 :=
sorry

end calc_perimeter_l528_528573


namespace range_g_l528_528463

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin(2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * sin(2 * x - 2 * π / 3) + 1

theorem range_g :
  set.range (λ x : ℝ, g x) ∩ set.Icc (-π / 3) (π / 2) = set.Icc (-1) (sqrt 3 + 1) :=
by
  sorry

end range_g_l528_528463


namespace liam_marbles_exceeds_500_on_thursday_l528_528873

def day_of_week_on_sunday (n : ℕ) : String :=
  ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"].getD (n % 7) "Invalid"

theorem liam_marbles_exceeds_500_on_thursday :
  ∃ n : ℕ, 4^n > 500 ∧ day_of_week_on_sunday n = "Thursday" := by
  use 5
  split
  · calc
      4^5 = 1024 := by ring
      _ > 500 := by norm_num
  · rfl

end liam_marbles_exceeds_500_on_thursday_l528_528873


namespace find_A_max_min_l528_528236

def is_coprime_with_36 (n : ℕ) : Prop := Nat.gcd n 36 = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let rest := n / 10
  d * 10^7 + rest

theorem find_A_max_min (B : ℕ) 
  (h1 : B > 77777777) 
  (h2 : is_coprime_with_36 B) : 
  move_last_digit_to_first B = 99999998 ∨ 
  move_last_digit_to_first B = 17777779 := 
by
  sorry

end find_A_max_min_l528_528236


namespace find_value_of_p_l528_528733

noncomputable def parabola_directrix (p : ℝ) : ℝ := -p / 2

noncomputable def hyperbola_points (p : ℝ) : Set ℝ :=
  {y | y = real.sqrt (3 + 3 * p^2 / 4) ∨ y = - real.sqrt (3 + 3 * p^2 / 4)}

theorem find_value_of_p (p : ℝ) (h1 : 0 < p)
  (h2 : let M := parabola_directrix p in ∀ y ∈ hyperbola_points p, true)
  (h3 : ∀ (F : ℝ × ℝ) (M N : ℝ × ℝ), M = (parabola_directrix p, real.sqrt (3 + 3 * p^2 / 4)) ∧
     N = (parabola_directrix p, - real.sqrt (3 + 3 * p^2 / 4)) ∧ (F, M, N) form an isosceles right triangle
        (λ FMN, angle FMN = π / 4)) :
  p = 2 * real.sqrt 3 :=
sorry

end find_value_of_p_l528_528733


namespace tiling_impossible_l528_528181

theorem tiling_impossible (m n : ℕ) (odd_odd_marking : ℕ → ℕ → bool) (original_tiles_2x2_1x4 : (ℕ × ℕ) → bool)
  (removed_tile_2x2: ℕ) (added_tile_1x4 : ℕ) :
  ∀ grid : ℕ × ℕ → bool,
  (∀ i j, grid i j = false ∨ grid i j = true ∧
    (if grid i j = true then odd_odd_marking i j = true else odd_odd_marking i j = false)) → 
  ¬(∃ new_grid : ℕ × ℕ → bool, 
    (∀ i j, new_grid i j = false ∨ new_grid i j = true ∧
    (if new_grid i j = true then odd_odd_marking i j = true else odd_odd_marking i j = false)) ∧
    count_removed_tile_2x2_removed_correctly new_grid = removed_tile_2x2 ∧
    count_added_tile_1x4_added_correctly new_grid = added_tile_1x4) :=
sorry

end tiling_impossible_l528_528181


namespace even_number_of_items_selection_l528_528453

theorem even_number_of_items_selection (n : ℕ) : 
  ∃ (f : finset (finset ℕ)), f.card = 2 ^ (n - 1) :=
sorry

end even_number_of_items_selection_l528_528453


namespace notebooks_last_days_l528_528409

theorem notebooks_last_days (n p u : Nat) (total_pages days : Nat) 
  (h1 : n = 5)
  (h2 : p = 40)
  (h3 : u = 4)
  (h_total : total_pages = n * p)
  (h_days  : days = total_pages / u) :
  days = 50 := 
by
  sorry

end notebooks_last_days_l528_528409


namespace necessary_but_not_sufficient_condition_l528_528692

theorem necessary_but_not_sufficient_condition (x : Real) : 
  (ln(x + 1) < 0 → x < 0) ∧ (x < 0 → ¬(ln(x + 1) < 0)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l528_528692


namespace laura_charges_for_truck_l528_528835

theorem laura_charges_for_truck : 
  ∀ (car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars : ℕ),
  car_wash = 5 →
  suv_wash = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  num_cars = 7 →
  total_amount = 100 →
  car_wash * num_cars + suv_wash * num_suvs + truck_wash * num_trucks = total_amount →
  truck_wash = 6 :=
by
  intros car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars h1 h2 h3 h4 h5 h6 h7
  sorry

end laura_charges_for_truck_l528_528835


namespace part_I_part_II_l528_528045

noncomputable def f (x : ℝ) (m : ℝ) := m - |x - 2|

theorem part_I (m : ℝ) : (∀ x, f (x + 1) m >= 0 → 0 <= x ∧ x <= 2) ↔ m = 1 := by
  sorry

theorem part_II (a b c : ℝ) (m : ℝ) : (1 / a + 1 / (2 * b) + 1 / (3 * c) = m) → (m = 1) → (a + 2 * b + 3 * c >= 9) := by
  sorry

end part_I_part_II_l528_528045


namespace pie_piece_division_l528_528974

theorem pie_piece_division :
  ∃ (n : ℕ), (∀ k ∈ {10, 11}, ∃ (p : ℕ → ℕ), (∑ i in range k, p i = n) ∧ (∑ i in range k, p i * 1 / i) = 1) ∧ n = 20 := 
by 
  sorry

end pie_piece_division_l528_528974


namespace right_triangle_exists_l528_528681

theorem right_triangle_exists (a b c p q : ℝ) 
  (h1 : p = a + c) (h2 : q = b + c) (h3 : a^2 + b^2 = c^2) : 
  ∃ (c' : ℝ), c' = c ∧ p = a + c' ∧ q = b + c' :=
begin
  sorry
end

end right_triangle_exists_l528_528681


namespace isosceles_triangle_height_eq_four_times_base_l528_528509

theorem isosceles_triangle_height_eq_four_times_base (b h : ℝ) 
    (same_area : (b * 2 * b) = (1/2 * b * h)) : 
    h = 4 * b :=
by 
  -- sorry allows us to skip the proof steps
  sorry

end isosceles_triangle_height_eq_four_times_base_l528_528509


namespace sum_perpendiculars_equal_altitude_l528_528215

def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def is_inside (P A B C : Point) : Prop :=
  ∃ a b c : ℝ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧ 
    P = a • A + b • B + c • C

noncomputable def altitude (A B C : Point) : ℝ :=
  let h := (sqrt 3 / 2) * dist A B
  h

def sum_of_perpendiculars (P A B C : Point) : ℝ :=
  let PA' := dist_to_line P (line_of A B)
  let PB' := dist_to_line P (line_of B C)
  let PC' := dist_to_line P (line_of C A)
  PA' + PB' + PC'

theorem sum_perpendiculars_equal_altitude
  (A B C P : Point) (h₁ : equilateral_triangle A B C) (h₂ : is_inside P A B C) :
  sum_of_perpendiculars P A B C = altitude A B C :=
sorry

end sum_perpendiculars_equal_altitude_l528_528215


namespace problem_statement_l528_528094

variable (q p : ℚ)
#check λ (q p : ℚ), q / p

theorem problem_statement :
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  q / p = -(119 : ℚ) / 3 :=
by
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  sorry

end problem_statement_l528_528094


namespace matrix_transpose_product_l528_528254

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 1], ![4, -2]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![7, -3], ![2, 4]]

def product_transpose := (A ⬝ B)ᵀ
def expected_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![23, 24], ![-5, -20]]

theorem matrix_transpose_product :
  product_transpose = expected_matrix :=
by sorry

end matrix_transpose_product_l528_528254


namespace length_of_plot_correct_l528_528570

noncomputable def length_of_plot (b : ℕ) : ℕ := b + 30

theorem length_of_plot_correct (b : ℕ) (cost_per_meter total_cost : ℝ) 
    (h1 : length_of_plot b = b + 30)
    (h2 : cost_per_meter = 26.50)
    (h3 : total_cost = 5300)
    (h4 : 2 * (b + (b + 30)) * cost_per_meter = total_cost) :
    length_of_plot 35 = 65 :=
by
  sorry

end length_of_plot_correct_l528_528570


namespace faith_hourly_rate_l528_528284

-- Definitions / Conditions
def regular_hours_per_day := 8
def working_days_per_week := 5
def overtime_hours_per_day := 2
def weekly_earnings := 675

-- Calculate total hours worked in a week, both regular and overtime
def total_hours_week := (regular_hours_per_day * working_days_per_week) +
                        (overtime_hours_per_day * working_days_per_week)

-- Calculate the hourly rate
def hourly_rate := weekly_earnings / total_hours_week

theorem faith_hourly_rate :
  hourly_rate = 13.5 :=
by
  dsimp [hourly_rate, weekly_earnings, total_hours_week, regular_hours_per_day, working_days_per_week, overtime_hours_per_day]
  rw [Nat.mul_comm regular_hours_per_day working_days_per_week, Nat.mul_comm overtime_hours_per_day working_days_per_week]
  rw [← Nat.add_mul]  -- Simplify calculation for total hours
  norm_num  -- Automatically normalize the numerical expression
  sorry  -- Placeholder for the actual proof

end faith_hourly_rate_l528_528284


namespace sanjay_homework_fraction_l528_528183

theorem sanjay_homework_fraction :
  let original := 1
  let done_on_monday := 3 / 5
  let remaining_after_monday := original - done_on_monday
  let done_on_tuesday := 1 / 3 * remaining_after_monday
  let remaining_after_tuesday := remaining_after_monday - done_on_tuesday
  remaining_after_tuesday = 4 / 15 :=
by
  -- original := 1
  -- done_on_monday := 3 / 5
  -- remaining_after_monday := 1 - 3 / 5
  -- done_on_tuesday := 1 / 3 * (1 - 3 / 5)
  -- remaining_after_tuesday := (1 - 3 / 5) - (1 / 3 * (1 - 3 / 5))
  sorry

end sanjay_homework_fraction_l528_528183


namespace fernanda_total_time_to_finish_l528_528699

noncomputable def fernanda_days_to_finish_audiobooks
  (num_audiobooks : ℕ) (hours_per_audiobook : ℕ) (hours_listened_per_day : ℕ) : ℕ :=
num_audiobooks * hours_per_audiobook / hours_listened_per_day

-- Definitions based on the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Statement to prove
theorem fernanda_total_time_to_finish :
  fernanda_days_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 := 
sorry

end fernanda_total_time_to_finish_l528_528699


namespace hcf_of_two_numbers_l528_528507

theorem hcf_of_two_numbers (H : ℕ) 
(lcm_def : lcm a b = H * 13 * 14) 
(h : a = 280 ∨ b = 280) 
(is_factor_h : H ∣ 280) : 
H = 5 :=
sorry

end hcf_of_two_numbers_l528_528507


namespace isabella_total_haircut_length_l528_528405

theorem isabella_total_haircut_length :
  (18 - 14) + (14 - 9) = 9 := 
sorry

end isabella_total_haircut_length_l528_528405


namespace circle_construction_tangent_l528_528679

variables {A B : Point} {S : Circle}

theorem circle_construction_tangent (A B : Point) (S : Circle) :
  let B_star := inv A B in
  let S_star := inv_circle A S in
  if B_star ∈ S_star then
    ∃! C : Circle, C ∋ A ∧ C ∋ B ∧ tangent C S
  else if outside B_star S_star then
    ∃ C1 C2 : Circle, C1 ∋ A ∧ C1 ∋ B ∧ tangent C1 S ∧ C2 ∋ A ∧ C2 ∋ B ∧ tangent C2 S ∧ C1 ≠ C2
  else
    ¬ ∃ C : Circle, C ∋ A ∧ C ∋ B ∧ tangent C S :=
sorry

end circle_construction_tangent_l528_528679


namespace fractional_eq_solution_l528_528475

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l528_528475


namespace carlos_score_is_75_l528_528790

variables (num_items : ℕ) (lowella_score : ℕ) (pamela_add_score : ℕ) (pamela_score : ℕ) (mandy_score : ℕ) (carlos_score : ℕ)

def conditions : Prop :=
  num_items = 120 ∧
  lowella_score = 35 * num_items / 100 ∧
  pamela_add_score = 20 * lowella_score / 100 ∧
  pamela_score = lowella_score + pamela_add_score ∧
  mandy_score = 2 * pamela_score ∧
  carlos_score = (pamela_score + mandy_score) / 2

theorem carlos_score_is_75 (h : conditions) : carlos_score = 75 :=
by { sorry }

end carlos_score_is_75_l528_528790


namespace solve_fractional_equation_l528_528466

noncomputable def fractional_equation (x : ℝ) : Prop :=
  (2 - x) / (x - 3) + 1 / (3 - x) = 1

theorem solve_fractional_equation :
  ∀ x : ℝ, x ≠ 3 → fractional_equation x ↔ x = 2 :=
by
  intros x h
  unfold fractional_equation
  sorry

end solve_fractional_equation_l528_528466


namespace find_a_n_l528_528368

noncomputable def matrix_exp {α : Type*} [Field α] (A : Matrix (Fin 3) (Fin 3) α) (n : ℕ) : Matrix (Fin 3) (Fin 3) α := sorry

def A (a : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 3, a], 
    ![0, 1, 5], 
    ![0, 0, 1]]

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 27, 3000], 
    ![0, 1, 45], 
    ![0, 0, 1]]

theorem find_a_n (a n: ℕ) (H: matrix_exp (A a) n = B) : a + n = 278 := 
by
  sorry

end find_a_n_l528_528368


namespace smallest_perimeter_l528_528948

theorem smallest_perimeter :
  ∃ a b c : ℕ, (a + 1 = b) ∧ (b + 1 = c) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧ (a % 2 = 0) ∧ (a + b + c = 9) :=
by { use [2, 3, 4], sorry }

end smallest_perimeter_l528_528948


namespace cost_prices_sum_l528_528226

theorem cost_prices_sum
  (W B : ℝ)
  (h1 : 0.9 * W + 196 = 1.04 * W)
  (h2 : 1.08 * B - 150 = 1.02 * B) :
  W + B = 3900 := 
sorry

end cost_prices_sum_l528_528226


namespace watermelon_melon_weight_l528_528439

-- Setup definitions for weights of watermelon and melon
variable (W M : ℕ)

theorem watermelon_melon_weight :
  (2 * W > 3 * M → ∀ y, 3 * W ≤ 4 * M) ∧
  (3 * W > 4 * M → 12 * W ≤ 18 * M) :=
by
  sorry

end watermelon_melon_weight_l528_528439


namespace find_c_l528_528961

theorem find_c (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 310) : C = 10 :=
by
  sorry

end find_c_l528_528961


namespace johns_score_is_101_l528_528047

variable (c w s : ℕ)
variable (h1 : s = 40 + 5 * c - w)
variable (h2 : s > 100)
variable (h3 : c ≤ 40)
variable (h4 : ∀ s' > 100, s' < s → ∃ c' w', s' = 40 + 5 * c' - w')

theorem johns_score_is_101 : s = 101 := by
  sorry

end johns_score_is_101_l528_528047


namespace complex_number_properties_l528_528309

-- Define the complex number z
def z : ℂ := 3 + 4 * Complex.i

-- Define the proof problem
theorem complex_number_properties :
  z.re = 3 ∧ Complex.conj z = 3 - 4 * Complex.i ∧ z.im = 4 ∧ Complex.abs z = 5 := by
  sorry

end complex_number_properties_l528_528309


namespace cyclist_distance_proof_l528_528990

noncomputable def distance_first_part := 7.035

def speed_first_part := 10.0 -- km/hr
def speed_second_part := 7.0 -- km/hr
def distance_second_part := 10.0 -- km
def average_speed := 7.99 -- km/hr

def time_first_part := distance_first_part / speed_first_part
def time_second_part := distance_second_part / speed_second_part
def total_distance := distance_first_part + distance_second_part
def total_time := time_first_part + time_second_part

theorem cyclist_distance_proof :
  (total_distance / total_time) = average_speed :=
  sorry

end cyclist_distance_proof_l528_528990


namespace inequality_proofs_l528_528304

theorem inequality_proofs (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (ab_proof : a * b ≤ 1 / 8) ∧ 
  (sqrt_sum_proof : real.sqrt a + real.sqrt b ≤ real.sqrt 6 / 2) ∧
  ¬ (forall x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → (2 / x + 1 / y ≤ 8)) ∧
  ¬ (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 3 ^ (x + y) ≤ real.sqrt 3) :=
by
  sorry

end inequality_proofs_l528_528304


namespace relay_team_average_time_l528_528685

theorem relay_team_average_time :
  let d1 := 200
  let t1 := 38
  let d2 := 300
  let t2 := 56
  let d3 := 250
  let t3 := 47
  let d4 := 400
  let t4 := 80
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  let average_time_per_meter := total_time / total_distance
  average_time_per_meter = 0.1922 := by
  sorry

end relay_team_average_time_l528_528685


namespace remaining_stock_weighs_120_l528_528067

noncomputable def total_remaining_weight (green_beans_weight rice_weight sugar_weight : ℕ) :=
  let remaining_rice := rice_weight - (rice_weight / 3)
  let remaining_sugar := sugar_weight - (sugar_weight / 5)
  let remaining_stock := remaining_rice + remaining_sugar + green_beans_weight
  remaining_stock

theorem remaining_stock_weighs_120 : total_remaining_weight 60 30 50 = 120 :=
by
  have h1: 60 - 30 = 30 := by norm_num
  have h2: 60 - 10 = 50 := by norm_num
  have h3: 30 - (30 / 3) = 20 := by norm_num
  have h4: 50 - (50 / 5) = 40 := by norm_num
  have h5: 20 + 40 + 60 = 120 := by norm_num
  exact h5

end remaining_stock_weighs_120_l528_528067


namespace guadiana_permutations_count_l528_528840

-- Define the concept of a guadiana permutation 
-- and state the main theorem.
theorem guadiana_permutations_count (n : ℕ) (h : 0 < n) :
  ∃ f : Fin n → ℕ, (∀ k : Fin (n - 1), f k.succ ≠ f k) ∧ 
  (Bijective (λ a : {a : ℕ // a ∈ set.univ}, f ∘ !a)) :=
  let count := 2^(n-1) in
  sorry

end guadiana_permutations_count_l528_528840


namespace sqrt_eqn_seq_l528_528328

theorem sqrt_eqn_seq (a b : ℕ) (h1 : a = 7) (h2 : b = 48) 
(h3 : Real.sqrt (2 + 2 / 3) = 2 * Real.sqrt (2 / 3))
(h4 : Real.sqrt (3 + 3 / 8) = 3 * Real.sqrt (3 / 8))
(h5 : Real.sqrt (4 + 4 / 15) = 4 * Real.sqrt (4 / 15))
(h6 : Real.sqrt (7 + a / b) = 7 * Real.sqrt (a / b)) : a + b = 55 := 
by
  rw [h1, h2]
  -- Continue the proof as necessary
  sorry

end sqrt_eqn_seq_l528_528328


namespace no_two_obtuse_angles_in_triangle_l528_528392

theorem no_two_obtuse_angles_in_triangle (T : Type) [inhabited T] :
  (∀ (A B C : T), A + B + C = 180 ∧ A > 90 ∧ B > 90 → False) :=
by
  assume (A B C : T)
  assume h : A + B + C = 180 ∧ A > 90 ∧ B > 90
  sorry

end no_two_obtuse_angles_in_triangle_l528_528392


namespace points_on_same_side_of_line_l528_528690

theorem points_on_same_side_of_line (m : ℝ) :
  (2 * 0 + 0 + m > 0 ∧ 2 * -1 + 1 + m > 0) ∨ 
  (2 * 0 + 0 + m < 0 ∧ 2 * -1 + 1 + m < 0) ↔ 
  (m < 0 ∨ m > 1) :=
by
  sorry

end points_on_same_side_of_line_l528_528690


namespace ball_height_after_bounces_l528_528596

noncomputable def first_height_below_two (initial_height : ℝ) (bounce_ratio : ℝ) : ℕ :=
  Nat.ceil (Real.log (2 / initial_height) / Real.log bounce_ratio)

theorem ball_height_after_bounces (initial_height : ℝ) (bounce_ratio : ℝ) :
  initial_height = 800 ∧ bounce_ratio = 2/3 →
  first_height_below_two initial_height bounce_ratio = 10 :=
begin
  intros h,
  cases h with h_init h_ratio,
  simp [first_height_below_two, h_init, h_ratio],
  -- Here you would proceed with the steps similar to the solution, but we add sorry to skip:
  sorry
end

end ball_height_after_bounces_l528_528596


namespace cant_cover_chessboard_with_tiles_l528_528987

open Finset

def Chessboard (n : ℕ) : Type := Finset (Fin n × Fin n)

def is_white (n : ℕ) (cell : Fin n × Fin n) : Prop :=
  (cell.1.val + cell.2.val) % 2 = 0

def is_black (n : ℕ) (cell : Fin n × Fin n) : Prop :=
  ¬ is_white n cell

def remove_cells {α : Type} (s : Finset α) (cells : Finset α) : Finset α :=
  s \ cells

noncomputable def can_cover_with_tiles (board : Finset (Fin 8 × Fin 8)) (tile : Finset (Fin 8 × Fin 8)) :=
  let covered_squares := (λ x, {cell : Fin 8 × Fin 8 | x ∈ tile ∧ cell ∈ tile}) in
  ∀ t : Finset (Fin 8 × Fin 8), t ⊆ board → (∃ p : Finset (Fin 8 × Fin 8), p.card = t.card / 2 ∧ ∀ elem ∈ p, covered_squares elem ⊆ t)

def modified_chessboard : Finset (Fin 8 × Fin 8) :=
  remove_cells (chessboard 8) {((0 : Fin 8), (0 : Fin 8)), ((7 : Fin 8), (7 : Fin 8))}

theorem cant_cover_chessboard_with_tiles :
  ¬ can_cover_with_tiles modified_chessboard (chessboard 2) :=
sorry

end cant_cover_chessboard_with_tiles_l528_528987


namespace jellybean_probability_l528_528198

/-- A bowl contains 15 jellybeans: five red, three blue, five white, and two green. If you pick four 
    jellybeans from the bowl at random and without replacement, the probability that exactly three will 
    be red is 20/273. -/
theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let white_jellybeans := 5
  let green_jellybeans := 2
  let total_combinations := Nat.choose total_jellybeans 4
  let favorable_combinations := (Nat.choose red_jellybeans 3) * (Nat.choose (total_jellybeans - red_jellybeans) 1)
  let probability := favorable_combinations / total_combinations
  probability = 20 / 273 :=
by
  sorry

end jellybean_probability_l528_528198


namespace find_x_value_l528_528292

theorem find_x_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 :=
sorry

end find_x_value_l528_528292


namespace lea_total_cost_l528_528076

theorem lea_total_cost :
  let book_cost := 16 in
  let binders_count := 3 in
  let binder_cost := 2 in
  let notebooks_count := 6 in
  let notebook_cost := 1 in
  book_cost + (binders_count * binder_cost) + (notebooks_count * notebook_cost) = 28 :=
by
  sorry

end lea_total_cost_l528_528076


namespace possible_one_twelfth_not_possible_one_sixth_l528_528525

-- Define the initial quantities in the glasses
def glass_quantities : List ℚ := [1/2, 1/3, 1/4, 1/5, 1/8, 1/9, 1/10]

-- Part (a): Prove it is possible to have a glass filled to 1/12
theorem possible_one_twelfth :
  ∃ (transfers : list (ℚ × ℚ)), 
  (∑ (transfer : ℚ × ℚ) in transfers, transfer.2) = 1/12 := sorry

-- Part (b): Prove it is not possible to have a glass filled to 1/6
theorem not_possible_one_sixth :
  ¬∃ (transfers : list (ℚ × ℚ)), 
  (∑ (transfer : ℚ × ℚ) in transfers, transfer.2) = 1/6 := sorry

end possible_one_twelfth_not_possible_one_sixth_l528_528525


namespace count_valid_x_below_10000_l528_528366

theorem count_valid_x_below_10000 : (Finset.filter (λ x : ℕ, 2^x % 7 = x^2 % 7) (Finset.range 10000)).card = 2857 := 
sorry

end count_valid_x_below_10000_l528_528366


namespace golden_state_total_points_l528_528034

theorem golden_state_total_points :
  ∀ (Draymond Curry Kelly Durant Klay : ℕ),
  Draymond = 12 →
  Curry = 2 * Draymond →
  Kelly = 9 →
  Durant = 2 * Kelly →
  Klay = Draymond / 2 →
  Draymond + Curry + Kelly + Durant + Klay = 69 :=
by
  intros Draymond Curry Kelly Durant Klay
  intros hD hC hK hD2 hK2
  rw [hD, hC, hK, hD2, hK2]
  sorry

end golden_state_total_points_l528_528034


namespace hyperbola_properties_l528_528901

-- Define the hyperbola with its equation
def hyperbola : Type := {x : ℝ × ℝ // (x.1 ^ 2) / 16 - (x.2 ^ 2) / 4 = 1}

-- Parameters
def a : ℝ := 4
def b : ℝ := 2

-- Condition that P is on the hyperbola and |PF1| = 4
variable {P : hyperbola}
def PF1_distance := 4

-- Left and right foci F1 and F2
def F1 : ℝ × ℝ := (-a * Math.sqrt(2), 0)
def F2 : ℝ × ℝ := (a * Math.sqrt(2), 0)

-- Auxiliary term involving focal distances for |PF2|
def focal_distance (P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
  Math.sqrt((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)

-- Core statement: calculating the asymptotic lines and the distance |PF2|
theorem hyperbola_properties :
  (∀ P : hyperbola,
    let PF2_distance := focal_distance (P.val) F2 in
    (P.val.2 = (1/2) * P.val.1 ∨ P.val.2 = -(1/2) * P.val.1) ∧
    PF2_distance = PF1_distance + 2 * a) :=
by sorry

end hyperbola_properties_l528_528901


namespace parallel_not_coincident_lines_l528_528590

theorem parallel_not_coincident_lines (a : ℝ) :
  let L1 := λ x y : ℝ, a * x + 2 * y + 3 * a = 0,
      L2 := λ x y : ℝ, 3 * x + (a - 1) * y = a - 7,
      parallel := a * (a - 1) - 6 = 0,
      not_coincident := (3 * (a - 7) ≠ (3 * a) * -1) in
  parallel → not_coincident → a = 3 :=
by
{
  intros L1 L2 parallel not_coincident,
  rw [L1, L2] at *,
  sorry
}

end parallel_not_coincident_lines_l528_528590


namespace N_subseteq_M_l528_528431

/--
Let M = { x | ∃ n ∈ ℤ, x = n / 2 + 1 } and
N = { y | ∃ m ∈ ℤ, y = m + 0.5 }.
Prove that N is a subset of M.
-/
theorem N_subseteq_M : 
  let M := { x : ℝ | ∃ n : ℤ, x = n / 2 + 1 }
  let N := { y : ℝ | ∃ m : ℤ, y = m + 0.5 }
  N ⊆ M := sorry

end N_subseteq_M_l528_528431


namespace probability_first_die_l528_528544

theorem probability_first_die (n : ℕ) (n_pos : n = 4025) (m : ℕ) (m_pos : m = 2012) : 
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  (favorable_outcomes / total_outcomes : ℚ) = 1006 / 4025 :=
by
  have h_n : n = 4025 := n_pos
  have h_m : m = 2012 := m_pos
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  sorry

end probability_first_die_l528_528544


namespace domain_f₁_range_f₂_l528_528888

noncomputable def f₁ (x : ℝ) : ℝ := (x - 2)^0 / Real.sqrt (x + 1)
noncomputable def f₂ (x : ℝ) : ℝ := 2 * x - Real.sqrt (x - 1)

theorem domain_f₁ : ∀ x : ℝ, x > -1 ∧ x ≠ 2 → ∃ y : ℝ, y = f₁ x :=
by
  sorry

theorem range_f₂ : ∀ y : ℝ, y ≥ 15 / 8 → ∃ x : ℝ, y = f₂ x :=
by
  sorry

end domain_f₁_range_f₂_l528_528888


namespace correct_statement_l528_528172

theorem correct_statement : 
  ∃ (frustum : Type), 
    (∀ (Pyramid : Type) (parallel_plane_base_cut : Pyramid → frustum), 
      regular_frustum frustum → regular_pyramid Pyramid) :=
sorry

end correct_statement_l528_528172


namespace rain_probability_tel_aviv_l528_528492

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l528_528492


namespace triangles_congruent_by_two_sides_and_median_l528_528088

-- Define the points A, B, C and A1, B1, C1
variables {A B C A1 B1 C1 M M1 : Type}

-- Define lengths of segments
variables {AB A1B1 BC B1C1 AM A1M1 : ℝ}

-- Define predicates for midpoint property
def is_midpoint (M : Type) (B C : Type) := sorry
def is_midpoint (M1 : Type) (B1 C1 : Type) := sorry

-- Congruence relation
def congruent_triangles (A B C : Type) (A1 B1 C1 : Type) := sorry

-- Given conditions
variables (h1 : AB = A1B1)
variables (h2 : BC = B1C1)
variables (h3 : AM = A1M1)
variables (h4 : is_midpoint M B C)
variables (h5 : is_midpoint M1 B1 C1)

-- Proof statement
theorem triangles_congruent_by_two_sides_and_median :
  congruent_triangles A B C A1 B1 C1 :=
sorry

end triangles_congruent_by_two_sides_and_median_l528_528088


namespace rashmi_late_time_is_10_l528_528882

open Real

noncomputable def rashmi_late_time : ℝ :=
  let d : ℝ := 9.999999999999993
  let v1 : ℝ := 5 / 60 -- km per minute
  let v2 : ℝ := 6 / 60 -- km per minute
  let time1 := d / v1 -- time taken at 5 kmph
  let time2 := d / v2 -- time taken at 6 kmph
  let difference := time1 - time2
  let T := difference / 2 -- The time she was late or early
  T

theorem rashmi_late_time_is_10 : rashmi_late_time = 10 := by
  simp [rashmi_late_time]
  sorry

end rashmi_late_time_is_10_l528_528882


namespace initial_hens_proof_l528_528437

-- Define the problem conditions
def lay_rate (eggs : ℕ) (days : ℕ) := eggs / days
def initial_hens := 25
def additional_hens := 15
def total_hens (H : ℕ) := H + additional_hens
def total_eggs := 300
def total_days := 15

-- Prove that Martin initially had 25 hens
theorem initial_hens_proof : 
    (∃ (H : ℕ), 
    let rate_initial := lay_rate 80 10 in -- 8 eggs/day
    let rate_new := lay_rate total_eggs total_days in -- 20 eggs/day
    rate_initial = lay_rate rate_initial H ∧
    rate_new = lay_rate rate_new (total_hens H) ∧
    H = 25) := 
by 
  existsi initial_hens
  sorry -- Proof not provided

end initial_hens_proof_l528_528437


namespace math_problem_l528_528557

theorem math_problem :
  ((3^1 - 2 + 4^2 + 1)⁻¹ * 6) = (1 / 3) :=
by
  sorry

end math_problem_l528_528557


namespace boy_walking_speed_l528_528982

theorem boy_walking_speed 
  (travel_rate : ℝ) 
  (total_journey_time : ℝ) 
  (distance : ℝ) 
  (post_office_time : ℝ) 
  (walking_back_time : ℝ) 
  (walking_speed : ℝ): 
  travel_rate = 12.5 ∧ 
  total_journey_time = 5 + 48/60 ∧ 
  distance = 9.999999999999998 ∧ 
  post_office_time = distance / travel_rate ∧ 
  walking_back_time = total_journey_time - post_office_time ∧ 
  walking_speed = distance / walking_back_time 
  → walking_speed = 2 := 
by 
  intros h;
  sorry

end boy_walking_speed_l528_528982


namespace amount_of_money_C_l528_528227

theorem amount_of_money_C (a b c d : ℤ) 
  (h1 : a + b + c + d = 600)
  (h2 : a + c = 200)
  (h3 : b + c = 350)
  (h4 : a + d = 300)
  (h5 : a ≥ 2 * b) : c = 150 := 
by
  sorry

end amount_of_money_C_l528_528227


namespace divisible_by_9_l528_528973

-- Definition of the sum of digits function S
def sum_of_digits (n : ℕ) : ℕ := sorry  -- Assume we have a function that sums the digits of n

theorem divisible_by_9 (a : ℕ) (h₁ : sum_of_digits a = sum_of_digits (2 * a)) 
  (h₂ : a % 9 = sum_of_digits a % 9) (h₃ : (2 * a) % 9 = sum_of_digits (2 * a) % 9) : 
  a % 9 = 0 :=
by
  sorry

end divisible_by_9_l528_528973


namespace ratio_equation_solution_l528_528343

variable (x y z : ℝ)
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)
variables (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)

theorem ratio_equation_solution
  (h : y / (2 * x - z) = (x + y) / (2 * z) ∧ (x + y) / (2 * z) = x / y) :
  x / y = 3 :=
sorry

end ratio_equation_solution_l528_528343


namespace quad_equiv_proof_l528_528378

theorem quad_equiv_proof (a b : ℝ) (h : a ≠ 0) (hroot : a * 2019^2 + b * 2019 + 2 = 0) :
  ∃ x : ℝ, a * (x - 1)^2 + b * (x - 1) = -2 ∧ x = 2019 :=
sorry

end quad_equiv_proof_l528_528378


namespace find_common_ratio_l528_528717

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

-- Conditions
def is_geometric_sequence : Prop := ∀ n, a (n + 1) = a n * q 
def sum_of_terms : Prop := S n = ∑ i in finset.range n, a i
def initial_term : Prop := a 0 = 1
def arithmetic_sequence_condition : Prop := 2 * S 1 = a 0 + 5

theorem find_common_ratio (h1 : is_geometric_sequence a q) (h2 : sum_of_terms a S) (h3 : initial_term a) (h4 : arithmetic_sequence_condition a S) : 
  q = 2 := 
by 
  sorry

end find_common_ratio_l528_528717


namespace concentric_circles_false_statement_l528_528358

theorem concentric_circles_false_statement
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : b < c) :
  ¬ (b + a = c + b) :=
sorry

end concentric_circles_false_statement_l528_528358


namespace lea_total_cost_l528_528074

theorem lea_total_cost :
  let book_cost := 16 in
  let binders_count := 3 in
  let binder_cost := 2 in
  let notebooks_count := 6 in
  let notebook_cost := 1 in
  book_cost + (binders_count * binder_cost) + (notebooks_count * notebook_cost) = 28 :=
by
  sorry

end lea_total_cost_l528_528074


namespace solve_fractional_equation_l528_528468

noncomputable def fractional_equation (x : ℝ) : Prop :=
  (2 - x) / (x - 3) + 1 / (3 - x) = 1

theorem solve_fractional_equation :
  ∀ x : ℝ, x ≠ 3 → fractional_equation x ↔ x = 2 :=
by
  intros x h
  unfold fractional_equation
  sorry

end solve_fractional_equation_l528_528468


namespace point_c_in_second_quadrant_l528_528804

-- Definitions for the points
def PointA : ℝ × ℝ := (1, 2)
def PointB : ℝ × ℝ := (-1, -2)
def PointC : ℝ × ℝ := (-1, 2)
def PointD : ℝ × ℝ := (1, -2)

-- Definition of the second quadrant condition
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
p.1 < 0 ∧ p.2 > 0

theorem point_c_in_second_quadrant : in_second_quadrant PointC :=
sorry

end point_c_in_second_quadrant_l528_528804


namespace tricia_age_l528_528533

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l528_528533


namespace stanley_walk_distance_l528_528889

variable (run_distance walk_distance : ℝ)

theorem stanley_walk_distance : 
  run_distance = 0.4 ∧ run_distance = walk_distance + 0.2 → walk_distance = 0.2 :=
by
  sorry

end stanley_walk_distance_l528_528889


namespace no_polynomial_representation_pi_l528_528456

def is_prime (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def pi (n : ℕ) : ℕ :=
  (Finset.filter is_prime (Finset.range (n + 1))).card

theorem no_polynomial_representation_pi :
  ¬∃ (P Q : polynomial ℝ), ∀ x : ℕ, pi x = (P.eval ↑x) / (Q.eval ↑x) := 
sorry

end no_polynomial_representation_pi_l528_528456


namespace even_number_subsets_l528_528451

theorem even_number_subsets (n : ℕ) :
  ∃ k : ℕ, k = 2^(n-1) ∧ (∀ (s : finset (fin n)), s.card % 2 = 0 → (s.card = k)) :=
sorry

end even_number_subsets_l528_528451


namespace range_of_m_in_first_quadrant_l528_528394

theorem range_of_m_in_first_quadrant (m : ℝ) : ((m - 1 > 0) ∧ (m + 2 > 0)) ↔ m > 1 :=
by sorry

end range_of_m_in_first_quadrant_l528_528394


namespace distance_after_3_minutes_l528_528638

-- Define the given speeds and time interval
def speed_truck : ℝ := 65 -- in km/h
def speed_car : ℝ := 85 -- in km/h
def time_minutes : ℝ := 3 -- in minutes

-- The equivalent time in hours
def time_hours : ℝ := time_minutes / 60

-- Calculate the distances travelled by the truck and the car
def distance_truck : ℝ := speed_truck * time_hours
def distance_car : ℝ := speed_car * time_hours

-- Define the distance between the truck and the car
def distance_between : ℝ := distance_car - distance_truck

-- Theorem: The distance between the truck and car after 3 minutes is 1 km.
theorem distance_after_3_minutes : distance_between = 1 := by
  sorry

end distance_after_3_minutes_l528_528638


namespace triangle_AD_eq_2AC_l528_528404

theorem triangle_AD_eq_2AC
  (A B C F D : Type)
  [AC_lt_AB : AC < AB] 
  [median_AF : Angle_ratio_AF A F B C = 1 / 2]
  [perpendicular_B_AB : Perpendicular B AB AF D] :
  AD = 2 * AC :=
sorry

end triangle_AD_eq_2AC_l528_528404


namespace common_points_tangent_curve_range_of_a_l528_528760

noncomputable def f (x : ℝ) : ℝ := x * Real.ln x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 2

theorem common_points_tangent_curve (a : ℝ) :
  let slope := 1
      tangent_line (x : ℝ) := x - 1
      intersection_eq := (x^2 + (1 - a) * x + 1 = 0)
  in ((a < -1 ∨ a > 3) ↔ (nat_degree intersection_eq = 2)) ∧
     ((a = -1 ∨ a = 3) ↔ (nat_degree intersection_eq = 1)) ∧
     ((-1 < a ∧ a < 3) ↔ (nat_degree intersection_eq = 0)) :=
sorry

theorem range_of_a (a : ℝ) (e approx := 2.71) :
  (3 < a ∧ a ≤ e + 2 / e + 1) ↔
  ∃ (x y : ℝ), (x ∈ set.Icc (1 / e) e ∧ y = f x - g x a ∧ y = 0) :=
sorry

end common_points_tangent_curve_range_of_a_l528_528760


namespace sufficient_but_not_necessary_condition_l528_528191

-- Define the condition
variable (a : ℝ)

-- Theorem statement: $a > 0$ is a sufficient but not necessary condition for $a^2 > 0$
theorem sufficient_but_not_necessary_condition : 
  (a > 0 → a^2 > 0) ∧ (¬ (a > 0) → a^2 > 0) :=
  by
    sorry

end sufficient_but_not_necessary_condition_l528_528191


namespace simplify_to_quadratic_form_l528_528464

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((6 * p + 2) - 3 * p * 5) ^ 2 + (5 - 2 / 4) * (8 * p - 12)

theorem simplify_to_quadratic_form (p : ℝ) : simplify_expression p = 81 * p ^ 2 - 50 :=
sorry

end simplify_to_quadratic_form_l528_528464


namespace rectangle_area_y_l528_528259

theorem rectangle_area_y (y : ℚ) (h_pos: y > 0) 
  (h_area: ((6 : ℚ) - (-2)) * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_area_y_l528_528259


namespace gabby_grows_one_watermelon_l528_528723

theorem gabby_grows_one_watermelon:
  ∃ W : ℕ, let P := W + 12 in
  let Pl := 3 * P in
  W + P + Pl = 53 ∧ W = 1 :=
by
  sorry

end gabby_grows_one_watermelon_l528_528723


namespace hyperbola_projection_distance_constant_l528_528811

variables {a b x y : ℝ}

def is_on_hyperbola (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def projection_distance (x y a b : ℝ) (h : a > 0) (h' : b > 0) : ℝ :=
  (a^2 * b^2) / (a^2 + b^2)

theorem hyperbola_projection_distance_constant 
  (h : a > 0) (h' : b > 0) (x y : ℝ) 
  (hP : is_on_hyperbola x y a b) :
  let M, N := (some_projections x y a b h h') in
  |dist P M| * |dist P N| = projection_distance x y a b h h' :=
sorry

end hyperbola_projection_distance_constant_l528_528811


namespace yuna_initial_papers_l528_528560

theorem yuna_initial_papers :
  ∃ Y : ℕ, (∀ (Yoojung_papers: ℕ), 
    Yoojung_papers = 210 →
    (∀ (papers_given : ℕ),
    papers_given = 30 →
    (∀ (difference_after_transfer : ℕ),
    difference_after_transfer = 50 →
    (Yoojung_papers - papers_given = Y + papers_given + difference_after_transfer → Y = 100)))) :=
begin
  use 100, -- We are asserting that Y = 100
  intros Yoojung_papers hYoojung_papers papers_given hPapers_given difference_after_transfer hDifference_after_transfer,
  assume h,
  have h1 : Yoojung_papers - papers_given = 180 := by
  { rw [hYoojung_papers, hPapers_given],
    norm_num },
  have h2 : Y + papers_given + difference_after_transfer = 180 := by
  { rw [←h, h1] },
  linarith, -- solve for Y
end

end yuna_initial_papers_l528_528560


namespace ribbon_arrangement_count_correct_l528_528013

-- Definitions for the problem conditions
inductive Color
| red
| yellow
| blue

-- The color sequence from top to bottom
def color_sequence : List Color := [Color.red, Color.blue, Color.yellow, Color.yellow]

-- A function to count the valid arrangements
def count_valid_arrangements (sequence : List Color) : Nat :=
  -- Since we need to prove, we're bypassing the actual implementation with sorry
  sorry

-- The proof statement
theorem ribbon_arrangement_count_correct : count_valid_arrangements color_sequence = 12 :=
by
  sorry

end ribbon_arrangement_count_correct_l528_528013


namespace baseball_card_value_decrease_l528_528197

theorem baseball_card_value_decrease :
  (V : ℝ) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100)
  (hV1 : V * (1 - x / 100) * 0.9 = V * 0.36) :
  x = 60 := by
  sorry

end baseball_card_value_decrease_l528_528197


namespace f_f_of_4_max_value_of_f_l528_528349

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - real.sqrt x else real.exp (real.log 2 * x)

theorem f_f_of_4 : f (f 4) = 1 / 2 :=
by
  sorry

theorem max_value_of_f : ∃ x, f x = 1 ∧ ∀ y, f y ≤ 1 :=
by
  sorry

end f_f_of_4_max_value_of_f_l528_528349


namespace min_dot_product_l528_528207

noncomputable def direction_vector : (ℝ × ℝ) := (4, -4)

noncomputable def intersection_y_axis : (ℝ × ℝ) := (0, -4)

def on_line (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in y = -x - 4

def is_MN_distance_4 (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 4

def OM_dot_ON_min (M N O : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := M in
  let (x2, y2) := N in
  ((x1 * x2 + y1 * y2))

theorem min_dot_product 
  (M N : (ℝ × ℝ))
  (hM : on_line M)
  (hN : on_line N)
  (hMN : is_MN_distance_4 M N) :
  ∃ (O : ℝ × ℝ), OM_dot_ON_min M N O = 4 :=
sorry

end min_dot_product_l528_528207


namespace distance_between_truck_and_car_l528_528625

def truck_speed : ℝ := 65 -- km/h
def car_speed : ℝ := 85 -- km/h
def time_minutes : ℝ := 3 -- minutes
def time_hours : ℝ := time_minutes / 60 -- converting minutes to hours

def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem distance_between_truck_and_car :
  let truck_distance := distance_traveled truck_speed time_hours
  let car_distance := distance_traveled car_speed time_hours
  truck_distance - car_distance = -1 := -- the distance is 1 km but negative when subtracting truck from car
by {
  sorry
}

end distance_between_truck_and_car_l528_528625


namespace B_pow_2024_eq_identity_l528_528838

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![real.sqrt 2 / 2, 0, -real.sqrt 2 / 2],
    ![0, 1, 0],
    ![real.sqrt 2 / 2, 0, real.sqrt 2 / 2]]

theorem B_pow_2024_eq_identity : B ^ 2024 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := 
  sorry

end B_pow_2024_eq_identity_l528_528838


namespace incorrect_statement_B_l528_528299

-- Defining the quadratic function
def quadratic_function (x : ℝ) : ℝ := -(x + 2)^2 - 3

-- Conditions derived from the problem
def statement_A : Prop := (∃ h k : ℝ, h < 0 ∧ k = 0)
def statement_B : Prop := (axis_of_symmetry (quadratic_function) = 2)
def statement_C : Prop := (¬ ∃ x : ℝ, quadratic_function x = 0)
def statement_D : Prop := (∀ x > -1, ∀ y > x, quadratic_function y < quadratic_function x)

-- The proof problem: show that statement B is incorrect
theorem incorrect_statement_B : statement_B = false :=
by sorry

end incorrect_statement_B_l528_528299


namespace find_whole_number_l528_528702

theorem find_whole_number (N : ℕ) : 9.25 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 9.75 → N = 38 := by
  intros h
  have hN : 37 < (N : ℝ) ∧ (N : ℝ) < 39 := by
    -- This part follows directly from multiplying the inequality by 4.
    sorry

  -- Convert to integer comparison
  have h1 : 38 ≤ N := by
    -- Since 37 < N, N must be at least 38 as N is an integer.
    sorry
    
  have h2 : N < 39 := by
    sorry

  -- Conclude that N = 38 as it is the single whole number within the range.
  sorry

end find_whole_number_l528_528702


namespace count_pairs_satisfying_conditions_l528_528914

theorem count_pairs_satisfying_conditions :
  let valid_pairs := [(p, q) | p q : ℤ, p > 0, q > 0, p + q ≤ 100, p + 1 / q = 17 * (1 / p + q)]
  valid_pairs.to_finset.card = 5 :=
by
  sorry

end count_pairs_satisfying_conditions_l528_528914


namespace exist_common_point_in_regular_18gon_l528_528087

theorem exist_common_point_in_regular_18gon :
  ∃ (P : ℝ × ℝ), ∃ (A_1 A_2 A_3 A_4 A_5 A_6 A_7 A_8 A_9 A_{10} A_{11} A_{12} A_{13} A_{14} A_{15} A_{16} A_{17} A_{18} : ℝ × ℝ),
  regular_18gon {A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_{10}, A_{11}, A_{12}, A_{13}, A_{14}, A_{15}, A_{16}, A_{17}, A_{18}} ∧
  is_diagonal (A_2, A_{12}) ∧ is_diagonal (A_8, A_{18}) ∧ is_diagonal (A_5, A_{16}) ∧
  intersects_at (A_2, A_{12}) (A_8, A_{18}) P ∧ intersects_at (A_8, A_{18}) (A_5, A_{16}) P ∧ intersects_at (A_5, A_{16}) (A_2, A_{12}) P ∧
  ¬ (is_axis_of_symmetry (A_2, A_{12}) ∨ is_axis_of_symmetry (A_8, A_{18}) ∨ is_axis_of_symmetry (A_5, A_{16})) :=
sorry

end exist_common_point_in_regular_18gon_l528_528087


namespace saree_stripes_l528_528548

theorem saree_stripes
  (G : ℕ) (B : ℕ) (Br : ℕ) (total_stripes : ℕ) (total_patterns : ℕ)
  (h1 : G = 3 * Br)
  (h2 : B = 5 * G)
  (h3 : Br = 4)
  (h4 : B + G + Br = 100)
  (h5 : total_stripes = 100)
  (h6 : total_patterns = total_stripes / 3) :
  B = 84 ∧ total_patterns = 33 := 
  by {
    sorry
  }

end saree_stripes_l528_528548


namespace soccer_match_outcome_l528_528919

theorem soccer_match_outcome :
  ∃ n : ℕ, n = 4 ∧
  (∃ (num_wins num_draws num_losses : ℕ),
     num_wins * 3 + num_draws * 1 + num_losses * 0 = 19 ∧
     num_wins + num_draws + num_losses = 14) :=
sorry

end soccer_match_outcome_l528_528919


namespace sin_shifted_symmetric_l528_528885

def sin_shifted (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)

theorem sin_shifted_symmetric : ∃ f, (∀ x, f x = sin_shifted x) ∧ (∀ x, f (-x - Real.pi) = f (x - Real.pi)) :=
by
  use (fun x => Real.cos x)
  split
  · intro x
    show Real.sin (x + Real.pi / 2) = Real.cos x
    rw [Real.sin_add_pi_div_two]
  · intro x
    show Real.cos (-x - Real.pi) = Real.cos (x - Real.pi)
    rw [Real.cos_neg, Real.cos_add_pi]
    exact congr_arg Real.cos (add_comm _ _)

end sin_shifted_symmetric_l528_528885


namespace turner_total_tickets_l528_528158

-- Definition of conditions
def days := 3
def rollercoaster_rides_per_day := 3
def catapult_rides_per_day := 2
def ferris_wheel_rides_per_day := 1

def rollercoaster_ticket_cost := 4
def catapult_ticket_cost := 4
def ferris_wheel_ticket_cost := 1

-- Proof statement
theorem turner_total_tickets : 
  days * (rollercoaster_rides_per_day * rollercoaster_ticket_cost 
  + catapult_rides_per_day * catapult_ticket_cost 
  + ferris_wheel_rides_per_day * ferris_wheel_ticket_cost) 
  = 63 := 
by
  sorry

end turner_total_tickets_l528_528158


namespace min_chord_length_of_circle_l528_528749

/-- 
Given a line passing through the point (1, 1) which intersects a circle 
given by the equation x^2 + y^2 - 4*x - 6*y + 4 = 0 at points A and B, 
the minimum value of |AB| is 4.
-/
theorem min_chord_length_of_circle 
    (A B : Point) 
    (hA : on_circle A (2, 3) 3)
    (hB : on_circle B (2, 3) 3)
    (hl : line_through (1, 1) A B) : 
    distance A B >= 4 :=
sorry

end min_chord_length_of_circle_l528_528749


namespace probability_of_two_pairs_l528_528363

noncomputable def probability_exactly_two_pairs (sock_draw : Finset (Fin 10)) (conditions : Finset (Set (Fin 10))) : ℚ :=
  let counts := conditions.card / 10
  let ways := (Finset.choose 2 5) * ((Finset.choose 1 3) * 1)
  let total_ways := Finset.choose 5 10
  (ways : ℚ) / total_ways

theorem probability_of_two_pairs :
  let socks := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let conditions := Finset.powerset socks.toFinset
  probability_exactly_two_pairs socks.toFinset conditions = 5 / 42 :=
sorry

end probability_of_two_pairs_l528_528363


namespace distance_after_3_minutes_l528_528643

/-- Let truck_speed be 65 km/h and car_speed be 85 km/h.
    Let time_in_minutes be 3 and converted to hours it is 0.05 hours.
    The goal is to prove that the distance between the truck and the car
    after 3 minutes is 1 kilometer. -/
def truck_speed : ℝ := 65 -- speed in km/h
def car_speed : ℝ := 85 -- speed in km/h
def time_in_minutes : ℝ := 3 -- time in minutes
def time_in_hours : ℝ := time_in_minutes / 60 -- converted time in hours
def distance_truck := truck_speed * time_in_hours
def distance_car := car_speed * time_in_hours
def distance_between : ℝ := distance_car - distance_truck

theorem distance_after_3_minutes : distance_between = 1 := by
  -- Proof steps would go here
  sorry

end distance_after_3_minutes_l528_528643


namespace correct_elements_in_set_l528_528750

def solution_set (x : ℝ) : set ℝ := { x | x > 3 / 2 }

theorem correct_elements_in_set : (2 ∈ solution_set 2) ∧ (0 ∉ solution_set 0) := by
  sorry

end correct_elements_in_set_l528_528750


namespace correct_statements_l528_528185

-- Defining people and their ages
variable {Andrew Boris Svetlana Larisa : Prop}
variable [HasLess Andrew Svetlana] [HasLess Larisa Andrew] [HasLess Svetlana Boris]
variable oldest_is_larisa_husband : ∀ x, x = Boris → (∃ y, y = Larisa ∧ married_to y x)

variable married_to : Prop → Prop → Prop

theorem correct_statements :
  (∃ y, y = Larisa ∧ (∀ z, z = Boris → married_to y z)) ∧ (¬ married_to Boris Svetlana) :=
by
  -- Here we will provide the detailed proof steps as indicated in the solution section.
  sorry

end correct_statements_l528_528185


namespace checkerboard_segment_length_difference_bound_l528_528998

-- Declare the conditions of the grid and line setup.
variables (grid : Π (i j : ℕ), Prop) -- checkerboard pattern
variable (l : ℝ → ℝ) -- line not parallel to the sides of the cells

/-
  The main statement to prove:
  There exists a constant C (depending only on the line l) such that for any segment I parallel to l,
  the difference between the sums of the lengths of its red and blue parts does not exceed C.
-/
theorem checkerboard_segment_length_difference_bound (l : ℝ → ℝ) (h_l : ¬∃ k : ℚ, ∀ x : ℝ, l x = k * x) :
  ∃ C : ℝ, ∀ I : set ℝ, (∀ x y ∈ I, l x = l y) →
    ∃ red_length blue_length : ℝ, 
    (∀ (i j : ℕ), grid i j) → -- cells i, j are in grid pattern, red or blue
    ∑ length_in_red_parts I grid - ∑ length_in_blue_parts I grid ≤ C := 
sorry

end checkerboard_segment_length_difference_bound_l528_528998


namespace distance_is_half_volume_is_pi_squared_l528_528388

namespace SphereRotation

-- Define the radius of the sphere and the length of the chord
def radius : ℝ := 1
def chord_length : ℝ := real.sqrt 3

-- Define the distance from the center of the sphere to the line
def distance_center_to_line (r l : ℝ) : ℝ := real.sqrt (r^2 - (l / 2)^2)

-- Prove that the distance is 1/2 given the conditions
theorem distance_is_half : distance_center_to_line radius chord_length = 1/2 := by
  sorry

-- Define the volume of the torus formed by rotating the sphere about a line
def torus_volume (r R : ℝ) : ℝ := 2 * real.pi^2 * R * r^2

-- Prove that the volume is pi^2 given the conditions
theorem volume_is_pi_squared : torus_volume radius (1/2) = real.pi^2 := by
  sorry

end SphereRotation

end distance_is_half_volume_is_pi_squared_l528_528388


namespace induction_sum_term_l528_528545

theorem induction_sum_term (k : ℕ) (h : ∑ i in range (k + 1), 1 / (i + 1) > 13 / 14) : 
  1 / (2 * k + 1) - 1 / (2 * (k + 1)) = 1 / (2 * k + 1) - 1 / (2 * (k + 1)) := by
  sorry

end induction_sum_term_l528_528545


namespace remainder_of_122_div_20_l528_528446

theorem remainder_of_122_div_20 :
  (∃ (q r : ℕ), 122 = 20 * q + r ∧ r < 20 ∧ q = 6) →
  r = 2 :=
by
  sorry

end remainder_of_122_div_20_l528_528446


namespace total_fruits_112_l528_528151

-- Definition of conditions
def condition1 (A P x : ℕ) : Prop :=
  A = 5 * x + 4 ∧ P = 3 * x

def condition2 (A P x y: ℕ) : Prop :=
  A = 7 * y ∧ P = 3 * y + 12

-- Theorem statement
theorem total_fruits_112 (A P x y : ℕ) (h1 : condition1 A P x) (h2 : condition2 A P y) : 
  A + P = 112 :=
by
  sorry

end total_fruits_112_l528_528151


namespace evaluate_expression_to_zero_l528_528886

-- Assuming 'm' is an integer with specific constraints and providing a proof that the expression evaluates to 0 when m = -1
theorem evaluate_expression_to_zero (m : ℤ) (h1 : -2 ≤ m) (h2 : m ≤ 2) (h3 : m ≠ 0) (h4 : m ≠ 1) (h5 : m ≠ 2) (h6 : m ≠ -2) : 
  (m = -1) → ((m / (m - 2) - 4 / (m ^ 2 - 2 * m)) / (m + 2) / (m ^ 2 - m)) = 0 := 
by
  intro hm_eq_neg1
  sorry

end evaluate_expression_to_zero_l528_528886


namespace tan_20_plus_4sin_20_eq_sqrt3_l528_528290

theorem tan_20_plus_4sin_20_eq_sqrt3 :
  (Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180)) = Real.sqrt 3 := by
  sorry

end tan_20_plus_4sin_20_eq_sqrt3_l528_528290


namespace exists_point_intersecting_lines_through_two_circles_l528_528444

-- Define the setting: two circles on the plane
structure Circle := 
  (center : ℝ × ℝ)
  (radius : ℝ)
  (radius_pos : radius > 0)

-- Prove there exists a point such that any line passing through it intersects at least one of the circles
theorem exists_point_intersecting_lines_through_two_circles (c1 c2 : Circle) : 
  ∃ (p : ℝ × ℝ), (∀ (L : ℝ × ℝ → ℝ × ℝ), (L p = p ∨ ∃ (q : ℝ × ℝ), q ≠ p ∧ (L p = L q) ∧ (∃ r : ℝ × ℝ, (distance p r > c1.radius ∧ distance q r > c2.radius) ∧ ( ∃ r' : ℝ × ℝ, inside_circle c1 r' ∨ inside_circle c2 r' ) ) ) ) :=
sorry

-- Helper functions to check if a point is inside a circle
def inside_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := c.center in
  (p.1 - x)^2 + (p.2 - y)^2 < c.radius^2

-- Necessity of noncomputable settings for real numbers
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1 in
  let (x2, y2) := p2 in
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

end exists_point_intersecting_lines_through_two_circles_l528_528444


namespace rewrite_expression_and_compute_l528_528096

noncomputable def c : ℚ := 8
noncomputable def p : ℚ := -3 / 8
noncomputable def q : ℚ := 119 / 8

theorem rewrite_expression_and_compute :
  (∃ (c p q : ℚ), 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) →
  q / p = -119 / 3 :=
by
  sorry

end rewrite_expression_and_compute_l528_528096


namespace sally_balloons_l528_528830

/-- 
Given:
- Joan has 9 blue balloons.
- Jessica has 2 blue balloons.
- The total number of blue balloons among Joan, Sally, and Jessica is 16.
Prove:
- Sally has 5 blue balloons.
-/
theorem sally_balloons (joan_balloons jessica_balloons total_balloons : ℕ) (h1 : joan_balloons = 9) 
  (h2 : jessica_balloons = 2) (h3 : total_balloons = 16) :
  ∃ sally_balloons : ℕ, sally_balloons = 5 :=
begin
  sorry
end

end sally_balloons_l528_528830


namespace amy_bob_games_count_l528_528682

def crestwood_three_square (players : Finset ℕ) (Amy Bob : ℕ) : Prop :=
  ∃ (game : Finset (Finset ℕ)), 
    game.card = 3 ∧ 
    (∀ (p q r : ℕ), {p, q, r} ∈ game → p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧
    (∀ player ∈ players, ∃! g ∈ game, player ∈ g) ∧ 
    (Amy ∈ players ∧ Bob ∈ players)

theorem amy_bob_games_count
  (players : Finset ℕ) (Amy Bob : ℕ) (semester_games : Finset (Finset ℕ))
  (h9 : players.card = 9) 
  (h_game_split : ∀ (day_games : Finset (Finset ℕ)), day_games.card = 3 ∧ 
    (∀ (game : Finset ℕ), game ∈ day_games → game.card = 3) ∧ 
    (∀ player ∈ players, ∃! game ∈ day_games, player ∈ game))
  (h_once : ∀ (g : Finset ℕ), g.card = 3 → g ∈ semester_games) : 
  (∃! (game : Finset ℕ), Amy ∈ game ∧ Bob ∈ game) → 
  semester_games.filter (λ g, Amy ∈ g ∧ Bob ∈ g).card = 7 := by
sorry

end amy_bob_games_count_l528_528682


namespace solve_fractional_equation_l528_528473

theorem solve_fractional_equation : 
  ∀ x : ℝ, x = 2 → (2 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by
  intro x hx
  rw hx
  simp
  sorry

end solve_fractional_equation_l528_528473


namespace rectangle_perimeter_l528_528605

theorem rectangle_perimeter :
  ∃ (a b : ℤ), a ≠ b ∧ a * b = 2 * (2 * a + 2 * b) ∧ 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l528_528605


namespace radii_parallel_l528_528936

variables {Point Circle Line : Type}

-- Definitions of the conditions
def touches (C1 C2 : Circle) (E : Point) : Prop := 
  ∃τ : Line, tangent (C1, E) τ ∧ tangent (C2, E) τ

def passes_through (l : Line) (E : Point) : Prop := 
  on_line l E

def intersects (l : Line) (C : Circle) (P : Point) : Prop := 
  ∃ Q1 Q2 : Point, on_line l Q1 ∧ on_line l Q2 ∧ on_circle C Q1 ∧ on_circle C Q2 ∧ (Q1 = P ∨ Q2 = P) ∧ Q1 ≠ Q2

def the_centers (C1 C2 : Circle) (O1 O2 : Point) : Prop :=
  center C1 = O1 ∧ center C2 = O2

-- The main theorem
theorem radii_parallel 
  {C1 C2 : Circle} {E B C O1 O2 : Point} {l : Line} :
  touches C1 C2 E →
  passes_through l E →
  intersects l C1 B →
  intersects l C2 C →
  the_centers C1 C2 O1 O2 →
  parallel (radius O1 B) (radius O2 C) :=
by
  sorry

end radii_parallel_l528_528936


namespace find_d_minus_b_l528_528428

theorem find_d_minus_b (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^5 = b^4) (h2 : c^3 = d^2) (h3 : c - a = 19) : d - b = 757 := 
by sorry

end find_d_minus_b_l528_528428


namespace problem_1_problem_2_l528_528350

-- Problem 1: If f(x) is monotonically increasing on ℝ, find the range of values for a
theorem problem_1 (f : ℝ → ℝ) (a : ℝ) (h : ∀ x, (exp x - 2 * x - a) ≥ 0):
  a ≤ 2 - 2 * real.log 2 := sorry

-- Problem 2: If a = 1, prove that when x > 0, f(x) > 1 - (ln 2 / 2) - (ln 2 / 2)^2
theorem problem_2 (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x, f x = exp x - x^2 - x) (a_eq_1 : a = 1):
  ∀ x > 0, f x > 1 - (real.log 2 / 2) - (real.log 2 / 2)^2 := sorry

end problem_1_problem_2_l528_528350


namespace coefficient_of_a_neg_half_l528_528038

theorem coefficient_of_a_neg_half :
  let term_coeff := ∑ k in Finset.range 8, (Nat.choose 7 k) * a^(7 - k) * (-1 / a^(1/2))^k
  term_coeff = -21 :=
by
  sorry

end coefficient_of_a_neg_half_l528_528038


namespace count_valid_b1_l528_528848

theorem count_valid_b1 :
  let b : ℕ → ℕ := λ n, if n = 0 then 0 else
    (if b n-1 % 3 = 0 then b n-1 / 3 else 4 * b n-1 + 1) in
  (Σ n, n ≤ 1000 ∧ (∀i, 1 ≤ i ∧ i ≤ 3 → b (i+1) > b 1)).card = 665 :=
sorry

end count_valid_b1_l528_528848


namespace exists_subset_of_10_l528_528111

namespace GroupProof

-- Defining the class size and the groups with conditions
constant class_size : ℕ
constant students : Finset ℕ
constant group : Finset (Finset ℕ)

axiom class_size_is_46 : class_size = 46
axiom all_students : students.card = 46
axiom groups_are_triples : ∀ g ∈ group, g.card = 3
axiom groups_intersection_condition : ∀ g1 g2 ∈ group, g1 ≠ g2 → (g1 ∩ g2).card ≤ 1

-- Prove the existence of a subset of 10 students in which no group is properly contained
theorem exists_subset_of_10 : 
  ∃ (S : Finset ℕ), S ⊆ students ∧ S.card = 10 ∧ ∀ g ∈ group, ¬ (g ⊆ S) :=
sorry

end GroupProof

end exists_subset_of_10_l528_528111


namespace probability_of_extreme_value_l528_528992

def f (a b x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * a * x^2 + b * x

def has_extreme_value (a b : ℝ) : Prop :=
  let Δ := a^2 - 4 * b
  Δ > 0

def num_satisfying_pairs : ℕ :=
  -- count all (a, b) pairs for which has_extreme_value holds
  let pairs := [(a, b) | a <- [1, 2, 3, 4, 5, 6], b <- [1, 2, 3, 4, 5, 6], has_extreme_value a b]
  pairs.length

def total_pairs : ℕ := 6 * 6

noncomputable def probability_extreme_value : ℝ :=
  num_satisfying_pairs / total_pairs

theorem probability_of_extreme_value :
  probability_extreme_value = 17 / 36 :=
sorry

end probability_of_extreme_value_l528_528992


namespace sin_alpha_minus_beta_l528_528129

noncomputable def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 = 1
noncomputable def line_eqn (x y m : ℝ) : Prop := y = 2*x + m
noncomputable def len_AB (A B : ℝ × ℝ) : ℝ := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem sin_alpha_minus_beta 
  (m : ℝ) 
  (A B : ℝ × ℝ) 
  (h_circle_A : circle_eqn A.1 A.2)
  (h_circle_B : circle_eqn B.1 B.2)
  (h_line_A : line_eqn A.1 A.2 m)
  (h_line_B : line_eqn B.1 B.2 m)
  (h_len_AB : len_AB A B = real.sqrt 3) 
  (h_OA : len_AB (0, 0) A = 1) 
  (h_OB : len_AB (0, 0) B = 1) :
  ∃ α β : ℝ, sin (α - β) = real.sqrt 3 / 2 ∨ sin (α - β) = -real.sqrt 3 / 2 := 
sorry

end sin_alpha_minus_beta_l528_528129


namespace intersection_point_of_curve_and_line_l528_528908

theorem intersection_point_of_curve_and_line : 
  ∃ (e : ℝ), (0 < e) ∧ (e = Real.exp 1) ∧ ((e, e) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), x ^ y = y ^ x ∧ 0 ≤ x ∧ 0 ≤ y}) :=
by {
  sorry
}

end intersection_point_of_curve_and_line_l528_528908


namespace expected_allergy_sufferers_l528_528080

theorem expected_allergy_sufferers (proportion : ℚ) (sample_size : ℕ) (expected : ℕ) :
    proportion = 1 / 8 → sample_size = 400 → expected = proportion * sample_size :=
by {
  intro h1 h2,
  rw [h1, h2],
  norm_num,
  sorry
}

end expected_allergy_sufferers_l528_528080


namespace complement_A_complement_U_range_of_a_empty_intersection_l528_528065

open Set Real

noncomputable def complement_A_in_U := { x : ℝ | ¬ (x < -1 ∨ x > 3) }

theorem complement_A_complement_U
  {A : Set ℝ} (hA : A = {x | x^2 - 2 * x - 3 > 0}) :
  (complement_A_in_U = (Icc (-1) 3)) :=
by sorry

theorem range_of_a_empty_intersection
  {B : Set ℝ} {a : ℝ}
  (hB : B = {x | abs (x - a) > 3})
  (h_empty : (Icc (-1) 3) ∩ B = ∅) :
  (0 ≤ a ∧ a ≤ 2) :=
by sorry

end complement_A_complement_U_range_of_a_empty_intersection_l528_528065


namespace turbo_path_invariant_l528_528306

noncomputable def turbo_invariant (lines : finset (affine_subspace ℝ ℝ (fin 2))) (no_three_intersections : ∀ A B C ∈ lines, A ≠ B → B ≠ C → C ≠ A → A ∩ B ≠ A ∩ C ∨ B ∩ C ≠ A ∩ C) : Prop :=
  ∀ start (h_start : start ∈ ⋃ (l ∈ lines), (l : set (affine ℝ (fin 2)))),
    ∀ path, valid_turbo_path lines no_three_intersections start path → ∀ segment, travels_both_directions path segment = false

-- Definitions for valid_turbo_path and travels_both_directions would involve precise formalizations.
-- Here we assume hypothetical definitions for the sake of structure.
constant valid_turbo_path : finset (affine_subspace ℝ ℝ (fin 2)) → (∀ A B C, A ∈ lines → B ∈ lines → C ∈ lines → A ≠ B → B ≠ C → C ≠ A → A ∩ B ≠ A ∩ C ∨ B ∩ C ≠ A ∩ C) → 
  start : set (affine ℝ (fin 2) ) → list (set (affine ℝ (fin 2))) → Prop

constant travels_both_directions : list (set (affine ℝ (fin 2))) → set (affine ℝ (fin 2)) → Bool

theorem turbo_path_invariant (lines : finset (affine_subspace ℝ ℝ (fin 2))) 
  (no_three_intersections : ∀ A B C ∈ lines, A ≠ B → B ≠ C → C ≠ A → A ∩ B ≠ A ∩ C ∨ B ∩ C ≠ A ∩ C) 
  (path : list (set (affine ℝ (fin 2)))) 
  (start : set (affine ℝ (fin 2)) (h_start : start ∈ ⋃ (l ∈ lines), (l : set (affine ℝ (fin 2)))) : ∀ segment, travels_both_directions path segment = false := sorry

lemma turbo_cannot_traverse_same_segment : ∀ lines : finset (affine_subspace ℝ ℝ (fin 2)),
  (∀ A B C ∈ lines, A ≠ B → B ≠ C → C ≠ A → A ∩ B ≠ A ∩ C ∨ B ∩ C ≠ A ∩ C) →
  turbo_invariant lines (sorry) :=
  sorry

end turbo_path_invariant_l528_528306


namespace solve_fractional_equation_l528_528471

theorem solve_fractional_equation : 
  ∀ x : ℝ, x = 2 → (2 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by
  intro x hx
  rw hx
  simp
  sorry

end solve_fractional_equation_l528_528471


namespace calculate_interest_rate_l528_528115

variables (A : ℝ) (R : ℝ)

-- Conditions as definitions in Lean 4
def compound_interest_condition (A : ℝ) (R : ℝ) : Prop :=
  (A * (1 + R)^20 = 4 * A)

-- Theorem statement
theorem calculate_interest_rate (A : ℝ) (R : ℝ) (h : compound_interest_condition A R) : 
  R = (4)^(1/20) - 1 := 
sorry

end calculate_interest_rate_l528_528115


namespace solve_abs_eq_l528_528706

theorem solve_abs_eq (x : ℝ) : (|x + 2| = 3*x - 6) → x = 4 :=
by
  intro h
  sorry

end solve_abs_eq_l528_528706


namespace who_wrote_the_incorrect_equation_l528_528272

def Anton := "It was Boris or Vladimir"
def Boris := "Neither Dima nor I did it"
def Vladimir := "Anton and Boris are both lying"
def Gosha := "One of Anton and Boris is lying, and the other is telling the truth"
def Dima := "Gosha is lying"

def condition := "Three of them always tell the truth, and the other two always lie"

theorem who_wrote_the_incorrect_equation :
  (Anton → Boris ∨ Vladimir) ∧
  (Boris → ¬Dima ∧ ¬Boris) ∧
  (Vladimir → ¬Anton ∧ ¬Boris) ∧
  (Gosha → (¬Anton ∨ ¬Boris) ∧ (Anton ∨ Boris)) ∧
  (Dima → ¬Gosha) ∧
  (condition → ∃ liar, liar = Vladimir) :=
sorry

end who_wrote_the_incorrect_equation_l528_528272


namespace flags_count_l528_528263

-- Define the colors available
inductive Color
| purple | gold | silver

-- Define the number of stripes on the flag
def number_of_stripes : Nat := 3

-- Define a function to calculate the total number of combinations
def total_flags (colors : Nat) (stripes : Nat) : Nat :=
  colors ^ stripes

-- The main theorem we want to prove
theorem flags_count : total_flags 3 number_of_stripes = 27 :=
by
  -- This is the statement only, and the proof is omitted
  sorry

end flags_count_l528_528263


namespace beau_age_today_l528_528660

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end beau_age_today_l528_528660


namespace num_divisors_even_l528_528295

-- Define the function d(n) as the number of positive divisors of n
def num_divisors (n : ℕ) : ℕ :=
  (finset.range n).filter (λ d, n % d = 0).card

theorem num_divisors_even (k : ℕ) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ num_divisors a = k ∧ num_divisors b = k ∧ num_divisors (2 * a + 3 * b) = k)
  ↔ (k > 0 ∧ even k) :=
begin
  sorry
end

end num_divisors_even_l528_528295


namespace Tricia_is_five_years_old_l528_528538

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l528_528538


namespace train_crossing_time_is_30_seconds_l528_528620

def length_of_train : ℝ := 120
def speed_of_train_kmph : ℝ := 45
def length_of_bridge : ℝ := 255

def total_distance : ℝ := length_of_train + length_of_bridge
def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

theorem train_crossing_time_is_30_seconds :
  (total_distance / speed_of_train_mps) = 30 :=
by
  sorry

end train_crossing_time_is_30_seconds_l528_528620


namespace vector_DE_l528_528813

-- Define points and vectors in a vector space
variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C D E : V)

-- Given conditions in Lean
def condition1 := (0 : ℝ) = A - (2 • (B - E + E - B)) -- AE = 2EB
def condition2 := (0 : ℝ) = C - (2 • (B - D + D - B)) -- BC = 2BD

-- Target statement to prove
theorem vector_DE (h1 : condition1) (h2 : condition2) :
  E - D = - (1/3) • (A - B) - (1/2) • (B - C) :=
sorry

end vector_DE_l528_528813


namespace num_paths_APPLES_l528_528672

def diagram : list (list char) := [
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', 'A', 'P', 'A', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', 'A', 'P', 'P', 'P', 'A', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', 'A', 'P', 'P', 'L', 'P', 'P', 'A', ' ', ' ', ' ', ' '],
  [' ', ' ', 'A', 'P', 'P', 'L', 'E', 'L', 'P', 'P', 'A', ' ', ' ', ' '],
  [' ', 'A', 'P', 'P', 'L', 'E', 'S', 'E', 'L', 'P', 'P', 'A', ' ', ' '],
  ['A', 'P', 'P', 'L', 'E', 'S', 'H', 'S', 'E', 'L', 'P', 'P', 'A', ' ']]

/-- 
Problem: Prove that the number of valid paths which spell "APPLES" in the given diagram is equal to 32 
-/
theorem num_paths_APPLES : 
  (count_paths diagram "APPLES" 32) := 
sorry

end num_paths_APPLES_l528_528672


namespace rewrite_expression_and_compute_l528_528097

noncomputable def c : ℚ := 8
noncomputable def p : ℚ := -3 / 8
noncomputable def q : ℚ := 119 / 8

theorem rewrite_expression_and_compute :
  (∃ (c p q : ℚ), 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) →
  q / p = -119 / 3 :=
by
  sorry

end rewrite_expression_and_compute_l528_528097


namespace palindromic_A_plus_11_l528_528855

theorem palindromic_A_plus_11 (a b c : ℕ) (A : ℕ) 
  (h1 : A = 10001 * a + 90090) 
  (h2 : 1 ≤ a ∧ a ≤ 8) 
  (h3 : b = 9) 
  (h4 : c = 9) : 
  ∃ (n : ℕ), n = 8 :=
by
  use 8
  sorry

end palindromic_A_plus_11_l528_528855


namespace distance_traveled_by_car_l528_528617

theorem distance_traveled_by_car :
  let total_distance := 90
  let distance_by_foot := (1 / 5 : ℝ) * total_distance
  let distance_by_bus := (2 / 3 : ℝ) * total_distance
  let distance_by_car := total_distance - (distance_by_foot + distance_by_bus)
  distance_by_car = 12 :=
by
  sorry

end distance_traveled_by_car_l528_528617


namespace sampling_method_l528_528012

/-- Define the data given in the problem --/
def num_classes : ℕ := 16
def num_students_per_class : ℕ := 50
def selected_student : ℕ := 14

/-- Define the conclusion to be proven --/
def selection_method_is_systematic : Prop :=
  ∀ c : ℕ, c ≤ num_classes → (1 ≤ selected_student ∧ selected_student ≤ num_students_per_class)

theorem sampling_method :
  selection_method_is_systematic →
  "Systematic Sampling" :=
by
  sorry

end sampling_method_l528_528012


namespace smallest_angle_of_pentagon_l528_528501

theorem smallest_angle_of_pentagon
  (a d : ℝ)
  (h1 : (∀ i, i ∈ {0, 1, 2, 3, 4} → (i = 0 → a) ∧ (i = 1 → a + d) ∧ (i = 2 → a + 2 * d) ∧ (i = 3 → a + 3 * d) ∧ (i = 4 → a + 4 * d)))
  (h2 : ∑ i in {0, 1, 2, 3, 4}, (a + i * d) = 540)
  (h3 : a + 4 * d = 140) : a = 76 := 
by
sor露.


end smallest_angle_of_pentagon_l528_528501


namespace coffee_shop_lattes_l528_528500

theorem coffee_shop_lattes (T : ℕ) (hT : T = 6) : ∃ L : ℕ, L = 4 * T + 8 := 
by 
  use 32
  rw [hT]
  rfl

end coffee_shop_lattes_l528_528500


namespace find_m_over_n_l528_528905

noncomputable
def ellipse_intersection_midpoint (m n : ℝ) (P : ℝ × ℝ) : Prop :=
  let M := (P.1, 1 - P.1)
  let N := (1 - P.2, P.2)
  let midpoint_MN := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  P = midpoint_MN

noncomputable
def ellipse_condition (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

noncomputable
def line_condition (x y : ℝ) : Prop :=
  x + y = 1

noncomputable
def slope_OP_condition (P : ℝ × ℝ) : Prop :=
  P.2 / P.1 = (Real.sqrt 2 / 2)

theorem find_m_over_n
  (m n : ℝ)
  (P : ℝ × ℝ)
  (h1 : ellipse_condition m n P.1 P.2)
  (h2 : line_condition P.1 P.2)
  (h3 : slope_OP_condition P)
  (h4 : ellipse_intersection_midpoint m n P) :
  (m / n = 1) :=
sorry

end find_m_over_n_l528_528905


namespace find_function_l528_528703

-- Definition of function f from positive integers to positive integers
def f : ℕ+ → ℕ+

-- Condition given in the problem
axiom condition : ∀ x y : ℕ+, 2 * y * f (f (x^2) + x) = f (x + 1) * f (2 * x * y)

-- Theorem statement that proving f(x) = x for all x in positive integers
theorem find_function : ∀ x : ℕ+, f x = x :=
by 
  sorry

end find_function_l528_528703


namespace solution_set_log_inequality_l528_528522

theorem solution_set_log_inequality :
  {x : ℝ | log (1 / 2) (abs (x - π / 3)) ≥ log (1 / 2) (π / 2)} =
  {x : ℝ | -π / 6 ≤ x ∧ x ≤ 5 * π / 6 ∧ x ≠ π / 3} :=
by
  sorry

end solution_set_log_inequality_l528_528522


namespace seating_arrangement_l528_528390

-- We define the conditions under which we will prove our theorem.
def chairs : ℕ := 7
def people : ℕ := 5

/-- Prove that there are exactly 1800 ways to seat five people in seven chairs such that the first person cannot sit in the first or last chair. -/
theorem seating_arrangement : (5 * 6 * 5 * 4 * 3) = 1800 :=
by
  sorry

end seating_arrangement_l528_528390


namespace race_distance_l528_528932

theorem race_distance (a b c : ℝ) (s_A s_B s_C : ℝ) :
  s_A * a = 100 → 
  s_B * a = 95 → 
  s_C * a = 90 → 
  s_B = s_A - 5 → 
  s_C = s_A - 10 → 
  s_C * (s_B / s_A) = 100 → 
  (100 - s_C) = 5 * (5 / 19) :=
sorry

end race_distance_l528_528932


namespace remainder_mod_1000_l528_528852

noncomputable def p (x : ℕ) : ℕ := (List.range (2009)).sum (λ i, x ^ i)

theorem remainder_mod_1000 (x : ℕ) (hx : x = 2008) :
  let r : ℕ := p x % (x^4 + x^3 + 2*x^2 + x + 1)
  in |r| % 1000 = 64 := by
{ sorry }

end remainder_mod_1000_l528_528852


namespace train_crossing_time_is_30_seconds_l528_528621

def length_of_train : ℝ := 120
def speed_of_train_kmph : ℝ := 45
def length_of_bridge : ℝ := 255

def total_distance : ℝ := length_of_train + length_of_bridge
def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

theorem train_crossing_time_is_30_seconds :
  (total_distance / speed_of_train_mps) = 30 :=
by
  sorry

end train_crossing_time_is_30_seconds_l528_528621


namespace rain_probability_tel_aviv_l528_528497

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l528_528497


namespace smallest_n_l528_528856

open Finset

theorem smallest_n (S : Finset ℕ) (hS : S = (finset.range 99).erase 0) :
  ∃ n, ∀ T, T ⊆ S → T.card = n → 
  ∃ T', T' ⊆ T ∧ T'.card = 10 ∧ 
  ∃ T1 T2, T1 ∪ T2 = T' ∧ T1.card = 5 ∧ T2.card = 5 ∧
  (∃ t1 ∈ T1, ∀ t2 ∈ T1, t1 ≠ t2 → gcd t1 t2 = 1) ∧
  (∃ t2 ∈ T2, ∀ t3 ∈ T2, t2 ≠ t3 → ¬gcd t2 t3 = 1) :=
sorry

end smallest_n_l528_528856


namespace factorial_difference_l528_528667

theorem factorial_difference :
  9! - 8! = 322560 := 
by 
  sorry

end factorial_difference_l528_528667


namespace triangle_DEF_area_l528_528044

/--
In triangle DEF, point L lies on EF such that DL is an altitude.
Given DE = 15, EL = 9, and EF = 24, prove that the area of triangle DEF is 144.
-/
theorem triangle_DEF_area
  (D E F L: Point)
  (DE EL EF DL: ℝ)
  (h_DE: DE = 15)
  (h_EL: EL = 9)
  (h_EF: EF = 24)
  (h_altitude: is_altitude D L F)
  (h_pts: collinear E F L)
  (h_D_on_DEF: ∃ A, triangle DEF = A)
  : area (triangle DEF) = 144 := 
by
  sorry

end triangle_DEF_area_l528_528044


namespace determine_s_value_l528_528726

def f (x : ℚ) : ℚ := abs (x - 1) - abs x

def u : ℚ := f (5 / 16)
def v : ℚ := f u
def s : ℚ := f v

theorem determine_s_value : s = 1 / 2 :=
by
  -- Proof needed here
  sorry

end determine_s_value_l528_528726


namespace fibonacci_mod_7_105_l528_528899

def fibonacci : ℕ → ℕ
| 0      := 0
| 1      := 1
| (n+2)  := fibonacci (n+1) + fibonacci n

theorem fibonacci_mod_7_105 : fibonacci 105 % 7 = 2 := 
by
  sorry

end fibonacci_mod_7_105_l528_528899


namespace correct_statements_l528_528187

noncomputable def Andrew := sorry
noncomputable def Boris := sorry
noncomputable def Svetlana := sorry
noncomputable def Larisa := sorry

-- Conditions
axiom cond1 : Boris = max Boris Andrew Svetlana Larisa
axiom cond2a : Andrew < Svetlana
axiom cond2b : Larisa < Andrew
axiom marriage1 : Larisa's husband = Boris
axiom marriage2 : Andrew's wife = Svetlana

-- Proof goal
theorem correct_statements : 
  (Larisa < Andrew ∧ Andrew < Svetlana ∧ Svetlana < Boris ∧ Larisa's husband = Boris) 
  ∧ (¬(Boris's wife = Svetlana)) :=
    by {
      sorry
    }

end correct_statements_l528_528187


namespace prime_probability_is_5_12_l528_528274

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_probability : ℚ :=
  let spinnerA_choices := [1, 3, 4]
  let spinnerB_choices := [1, 2, 3, 5]
  let outcomes := [(a, b) | a ← spinnerA_choices, b ← spinnerB_choices]
  let primes := outcomes.count (fun (a, b) => is_prime (a + a * b))
  (primes : ℚ) / outcomes.length

theorem prime_probability_is_5_12 :
  prime_probability = 5 / 12 := 
sorry

end prime_probability_is_5_12_l528_528274


namespace divisibility_by_n5_plus_1_l528_528859

theorem divisibility_by_n5_plus_1 (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n^5 + 1 ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) :=
sorry

end divisibility_by_n5_plus_1_l528_528859


namespace probability_of_sqrt5_distance_l528_528040

def set_of_points := {(x, y) | x ∈ {-1, 0, 1} ∧ y ∈ {-1, 0, 1}}

def distance (p1 p2 : ℤ × ℤ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def has_sqrt5_pair (pts : set (ℤ × ℤ)) : Prop :=
  ∃ p1 p2, p1 ∈ pts ∧ p2 ∈ pts ∧ p1 ≠ p2 ∧ distance p1 p2 = real.sqrt 5

theorem probability_of_sqrt5_distance :
  let K := {(x, y) | x ∈ {-1, 0, 1} ∧ y ∈ {-1, 0, 1}} in
  let total_combinations := finset.card (finset.powersetLen 3 (K.to_finset)) in
  let favorable_combinations := finset.card (finset.filter (λ s, has_sqrt5_pair s) (finset.powersetLen 3 (K.to_finset))) in
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = 2 / 3 := 
begin
  sorry
end

end probability_of_sqrt5_distance_l528_528040


namespace increasing_intervals_min_value_max_value_l528_528906

noncomputable def f (x : ℝ) : ℝ := (2 : ℝ) ^ 0.5 * Real.cos (2 * x - (Real.pi / 4))

theorem increasing_intervals (k : ℤ) :
  ∀ x, (k : ℝ) * Real.pi - (3 * Real.pi / 8) ≤ x ∧ x ≤ (k : ℝ) * Real.pi + (Real.pi / 8) →
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ :=
sorry

theorem min_value :
  ∃ x, x ∈ Set.Icc (-Real.pi / 8) (Real.pi / 2) ∧ f x = -(2 ^ 0.5) ∧ x = Real.pi / 2 :=
sorry

theorem max_value :
  ∃ x, x ∈ Set.Icc (-Real.pi / 8) (Real.pi / 2) ∧ f x = 2 ^ 0.5 ∧ x = Real.pi / 8 :=
sorry

end increasing_intervals_min_value_max_value_l528_528906


namespace distance_between_foci_l528_528239

open Real

theorem distance_between_foci (h1 : tangent_at_point (ellipse 3 2 H1 x_axis) (3, 0))
(h2 : tangent_at_point (ellipse 3 2 H2 y_axis) (0, 2))
(h3 : tangent_to_line (ellipse 3 2 H3 (λ x, x + 1))) :
    distance_between_foci (ellipse 3 2 H4) = 2 * sqrt 5 := 
sorry

end distance_between_foci_l528_528239


namespace find_A_max_min_l528_528233

theorem find_A_max_min :
  ∃ (A_max A_min : ℕ), 
    (A_max = 99999998 ∧ A_min = 17777779) ∧
    (∀ B A, 
      (B > 77777777) ∧
      (Nat.coprime B 36) ∧
      (A = (B % 10) * 10000000 + B / 10) →
      (A ≤ 99999998 ∧ A ≥ 17777779)) :=
by 
  existsi 99999998
  existsi 17777779
  split
  { 
    split 
    { 
      refl 
    }
    refl 
  }
  intros B A h
  sorry

end find_A_max_min_l528_528233


namespace collinear_points_l528_528089

noncomputable def Quadrilateral := Type
variables {A B C D O M H N A_1 B_1 C_1 D_1 : Point}

-- Conditions of the problem
def is_inscribed (O : Point) (ABCD : Quadrilateral) : Prop := sorry
def is_altitude (A1 : Point) (triangle : Triangle) : Prop := sorry

open Triangle

-- Defining the triangles
def triangle_AOB (A B O : Point) := mk_triangle A B O
def triangle_COD (C D O : Point) := mk_triangle C D O

theorem collinear_points :
  is_inscribed O ABCD ∧
  is_altitude A_1 (triangle_AOB A B O) ∧
  is_altitude B_1 (triangle_AOB A B O) ∧
  is_altitude C_1 (triangle_COD C D O) ∧
  is_altitude D_1 (triangle_COD C D O) →
  collinear {A_1, B_1, C_1, D_1} :=
sorry

end collinear_points_l528_528089


namespace rain_probability_tel_aviv_l528_528494

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l528_528494


namespace price_decrease_l528_528651

theorem price_decrease (P : ℝ) (increase1 : ℝ := 1.20) (conversion1 : ℝ := 0.85) 
                        (increase2 : ℝ := 1.12) (tax : ℝ := 1.05) 
                        (conversion2 : ℝ := 1.18) : 
  P / (P * increase1 * conversion1 * increase2 * tax * conversion2) = 1 - 0.36471 :=
by
  have h : P * increase1 * conversion1 * increase2 * tax * conversion2 = 1.573848 * P,
  { sorry },
  calc
    P / (P * increase1 * conversion1 * increase2 * tax * conversion2)
      = 1 / 1.573848 : by rw [h]
    ... = 1 - 0.36471 : by norm_num

end price_decrease_l528_528651


namespace mohamed_donated_more_l528_528837

-- Definitions of the conditions
def toysLeilaDonated : ℕ := 2 * 25
def toysMohamedDonated : ℕ := 3 * 19

-- The theorem stating Mohamed donated 7 more toys than Leila
theorem mohamed_donated_more : toysMohamedDonated - toysLeilaDonated = 7 :=
by
  sorry

end mohamed_donated_more_l528_528837


namespace polynomial_degree_is_m_l528_528779
-- Bring in necessary Lean libraries

-- Define additinal namespaces (if needed)
--  open_locale polynomial

-- Define the predicate for degree of a polynomial
def polynomial_degree (P : ℕ → ℕ → polynomial ℕ) : ℕ := P.coeff.degree

-- Formulate the main theorem statement
theorem polynomial_degree_is_m (m n : ℕ) (hmn : m > n) :
  polynomial_degree (λ x y, polynomial.X^m + polynomial.X^n - (polynomial.C (2^(m+n)))) = m :=
sorry

end polynomial_degree_is_m_l528_528779


namespace find_angle_FNE_l528_528935

variable (DEF: Type) [IsTriangle DEF]
variable (DF EF: LineSegment DEF) (N: Point DEF)
variable (angleDFE angleDNF angleNFD angleFNE: ℝ)

axiom is_isosceles_triangle: DF = EF
axiom angle_DFE_108: angleDFE = 108
axiom angle_DNF_11: angleDNF = 11
axiom angle_NFD_19: angleNFD = 19

theorem find_angle_FNE : angleFNE = 42 :=
by 
  sorry

end find_angle_FNE_l528_528935


namespace exists_infinitely_many_terms_divisible_by_three_exists_infinitely_many_terms_divisible_by_gcd_20_equal_1_l528_528920

def sequence (a : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := sequence n + (sequence n % 10)

theorem exists_infinitely_many_terms_divisible_by_three (a : ℕ) 
  (strictly_increasing : ∀ n, sequence a (n+1) > sequence a n) : 
  ∃ infinitely_many_n, ∃ m, sequence a n = 3 * m :=
sorry

theorem exists_infinitely_many_terms_divisible_by_gcd_20_equal_1 (a : ℕ) (d : ℕ) 
  (gcd_d_20_eq_1 : Nat.gcd d 20 = 1)
  (strictly_increasing : ∀ n, sequence a (n+1) > sequence a n) : 
  ∃ infinitely_many_n, ∃ m, sequence a n = d * m :=
sorry

end exists_infinitely_many_terms_divisible_by_three_exists_infinitely_many_terms_divisible_by_gcd_20_equal_1_l528_528920


namespace perpendicular_and_intersection_l528_528344

theorem perpendicular_and_intersection :
  let line1 := λ (x y : ℚ), 4 * y - 3 * x = 16
  let line4 := λ (x y : ℚ), 3 * y + 4 * x = 15
  (∃ (x y : ℚ), line1 x y ∧ line4 x y ∧ (x = 12 / 25) ∧ (y = 109 / 25)) ∧
  ((3 / 4) * - (4 / 3) = -1) := by
    sorry

end perpendicular_and_intersection_l528_528344


namespace find_y_l528_528396

-- Define the polynomial f(x) = (x + y) * (x + 1)^4
noncomputable def f (x y : ℝ) : ℝ := (x + y) * (x + 1)^4

-- Define the statement to prove
theorem find_y (h : ∑ i in {1, 3, 5}, ((x + y) * (x + 1)^4).coeff i = 32) : y = 3 := 
by 
  sorry

end find_y_l528_528396


namespace movie_ticket_change_l528_528163

theorem movie_ticket_change (num_sisters : ℕ) (cost_per_ticket : ℕ) (money_brought : ℕ) 
    (h1 : num_sisters = 2) (h2 : cost_per_ticket = 8) (h3 : money_brought = 25) : 
    money_brought - num_sisters * cost_per_ticket = 9 :=
by
  rw [h1, h2, h3]
  sorry

end movie_ticket_change_l528_528163


namespace find_x_minus_y_l528_528777

variables (x y : ℚ)

theorem find_x_minus_y
  (h1 : 3 * x - 4 * y = 17)
  (h2 : x + 3 * y = 1) :
  x - y = 69 / 13 := 
sorry

end find_x_minus_y_l528_528777


namespace compute_expression_l528_528846

-- The definition and conditions
def is_nonreal_root_of_unity (ω : ℂ) : Prop := ω ^ 3 = 1 ∧ ω ≠ 1

-- The statement
theorem compute_expression (ω : ℂ) (hω : is_nonreal_root_of_unity ω) : 
  (1 - 2 * ω + 2 * ω ^ 2) ^ 6 + (1 + 2 * ω - 2 * ω ^ 2) ^ 6 = 0 :=
sorry

end compute_expression_l528_528846


namespace relationship_among_a_b_c_l528_528266

noncomputable def f (x : ℝ) : ℝ := sorry  -- The actual function definition is not necessary for this statement.

-- Lean statements for the given conditions
variables {f : ℝ → ℝ}

-- f is even
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- f(x+1) = -f(x)
def periodic_property (f : ℝ → ℝ) := ∀ x, f (x + 1) = - f x

-- f is monotonically increasing on [-1, 0]
def monotonically_increasing_on (f : ℝ → ℝ) := ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the relationship statement
theorem relationship_among_a_b_c (h1 : even_function f) (h2 : periodic_property f) 
  (h3 : monotonically_increasing_on f) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end relationship_among_a_b_c_l528_528266


namespace average_weight_of_students_l528_528794

theorem average_weight_of_students (b_avg_weight g_avg_weight : ℝ) (num_boys num_girls : ℕ)
  (hb : b_avg_weight = 155) (hg : g_avg_weight = 125) (hb_num : num_boys = 8) (hg_num : num_girls = 5) :
  (num_boys * b_avg_weight + num_girls * g_avg_weight) / (num_boys + num_girls) = 143 :=
by sorry

end average_weight_of_students_l528_528794


namespace proof_GHK_equilateral_l528_528310

variables {V : Type} [EuclideanSpace ℝ V]

def is_regular_hexagon (ABCDEF : six_cycle_points V) (r : ℝ) : Prop :=
  ∃ O : V,
    dist O ABCDEF.A = r ∧
    dist O ABCDEF.B = r ∧
    dist O ABCDEF.C = r ∧
    dist O ABCDEF.D = r ∧
    dist O ABCDEF.E = r ∧
    dist O ABCDEF.F = r ∧
    dist ABCDEF.A ABCDEF.B = dist ABCDEF.C ABCDEF.D ∧
    dist ABCDEF.A ABCDEF.B = dist ABCDEF.E ABCDEF.F

def midpoint (P Q : V) : V :=
  (P + Q) / 2

def equilateral_triangle (A B C : V) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist A B = dist C A

theorem proof_GHK_equilateral {ABCDEF : six_cycle_points V} (r : ℝ) (G H K : V)
  (h_hexagon : is_regular_hexagon ABCDEF r)
  (hG : G = midpoint ABCDEF.B ABCDEF.C)
  (hH : H = midpoint ABCDEF.D ABCDEF.E)
  (hK : K = midpoint ABCDEF.F ABCDEF.A) :
  equilateral_triangle G H K :=
sorry

end proof_GHK_equilateral_l528_528310


namespace tangent_line_eq_l528_528287

noncomputable def f : ℝ → ℝ := λ x, (1/3) * x^3 + 4/3

theorem tangent_line_eq :
  let df := λ x, x^2,
      slope := df 2,
      point := (2, 4 : ℝ),
      line := λ x, slope * (x - point.1) + point.2
  in  ∀ x y, y = line x ↔ 4 * x - y - 4 = 0 :=
by
  intro x y
  sorry

end tangent_line_eq_l528_528287


namespace find_extrema_A_l528_528232

def eight_digit_number(n : ℕ) : Prop := n ≥ 10^7 ∧ n < 10^8

def coprime_with_thirtysix(n : ℕ) : Prop := Nat.gcd n 36 = 1

def transform_last_to_first(n : ℕ) : ℕ := 
  let last := n % 10
  let rest := n / 10
  last * 10^7 + rest

theorem find_extrema_A :
  ∃ (A_max A_min : ℕ), 
    (∃ B_max B_min : ℕ, 
      eight_digit_number B_max ∧ 
      eight_digit_number B_min ∧ 
      coprime_with_thirtysix B_max ∧ 
      coprime_with_thirtysix B_min ∧ 
      B_max > 77777777 ∧ 
      B_min > 77777777 ∧ 
      transform_last_to_first B_max = A_max ∧ 
      transform_last_to_first B_min = A_min) ∧ 
    A_max = 99999998 ∧ 
    A_min = 17777779 := 
  sorry

end find_extrema_A_l528_528232


namespace measure_one_kg_grain_l528_528664

/-- Proving the possibility of measuring exactly 1 kg of grain
    using a balance scale, one 3 kg weight, and three weighings. -/
theorem measure_one_kg_grain :
  ∃ (weighings : ℕ) (balance_scale : ℕ → ℤ) (weight_3kg : ℤ → Prop),
  weighings = 3 ∧
  (∀ w, weight_3kg w ↔ w = 3) ∧
  ∀ n m, balance_scale n = 0 ∧ balance_scale m = 1 → true :=
sorry

end measure_one_kg_grain_l528_528664


namespace points_concyclic_l528_528443

variables {A B C D : Point}
variables (A1 B1 C1 D1 : Point)
variable h : Circle

-- Define the condition that A, B, C, and D lie on a circle.
def on_circle (P : Point) := P ∈ h

-- Define circles passing through neighboring points
def circle_AB : Circle := Circle.mk A B
def circle_BC : Circle := Circle.mk B C
def circle_CD : Circle := Circle.mk C D
def circle_DA : Circle := Circle.mk D A

-- Define second intersection points
def second_intersection (c1 c2 : Circle) : Point := sorry -- Assume this gives the second intersection

-- Assertion: the second intersections of the neighboring circles A1, B1, C1, and D1
def A1_def := second_intersection circle_AB circle_DA
def B1_def := second_intersection circle_AB circle_BC
def C1_def := second_intersection circle_BC circle_CD
def D1_def := second_intersection circle_CD circle_DA

-- To prove that A1, B1, C1, D1 are concyclic
theorem points_concyclic : on_circle A1 ∧ on_circle B1 ∧ on_circle C1 ∧ on_circle D1 → 
  ∃ k : Circle, A1 ∈ k ∧ B1 ∈ k ∧ C1 ∈ k ∧ D1 ∈ k :=
by
  sorry

end points_concyclic_l528_528443


namespace repeating_block_length_fraction_l528_528662

noncomputable def decimal_expansion_fraction : ℚ := 7 / 13

theorem repeating_block_length_fraction : ∀ (n : ℕ), (decimal_expansion_fraction.toReal.repr.drop 2).take n = "538461" → n = 6 := by
  sorry

end repeating_block_length_fraction_l528_528662


namespace total_paths_A_to_D_l528_528716

-- Given conditions
def paths_from_A_to_B := 2
def paths_from_B_to_C := 2
def paths_from_C_to_D := 2
def direct_path_A_to_C := 1
def direct_path_B_to_D := 1

-- Proof statement
theorem total_paths_A_to_D : 
  paths_from_A_to_B * paths_from_B_to_C * paths_from_C_to_D + 
  direct_path_A_to_C * paths_from_C_to_D + 
  paths_from_A_to_B * direct_path_B_to_D = 12 := 
  by
    sorry

end total_paths_A_to_D_l528_528716


namespace dasha_sergey_problem_l528_528684

variable {α : Type*}

theorem dasha_sergey_problem :
  ∃ (x y : ℕ) (numbers : Fin 158 → ℕ), 
  (∑ i, (numbers i) = 1580) ∧
  (numbers (Fin.ofNat 157) = x) ∧
  (numbers (Fin.ofNat 0) = y) ∧
  (∑ i, ((if i = Fin.ofNat 157 then 3 * x else if i = Fin.ofNat 0 then y - 20 else numbers i))  = 1580) ∧
  (∀ i, numbers i = 10) →
  (∃ n, n = 10 ∧ ∀ i, numbers i ≥ n) := sorry

end dasha_sergey_problem_l528_528684


namespace carson_class_average_score_l528_528440

noncomputable def average_score (scores : List (ℕ × ℕ)) : ℚ :=
  let total_score := scores.foldr (λ (p : ℕ × ℕ) acc, acc + p.1 * p.2) 0
  let total_students := scores.foldr (λ (p : ℕ × ℕ) acc, acc + p.2) 0
  total_score / total_students

theorem carson_class_average_score :
  average_score [(100, 10), (95, 20), (85, 30), (75, 40), (65, 15), (55, 5)] = 80.83 := by
  sorry

end carson_class_average_score_l528_528440


namespace probability_two_dice_sum_greater_than_9_eq_1_over_6_l528_528972

-- Define the sample space for a die roll
def die := {1, 2, 3, 4, 5, 6}

-- Define the event of rolling two dice
def event_space := die × die

-- Define the favorable event where the sum is greater than 9
def favorable_event (x : ℕ × ℕ) : Prop := (x.1 + x.2) > 9

-- Find the number of elements in a set that satisfy a predicate
def count {α : Type*} (s : Finset α) (p : α → Prop) [DecidablePred p] : ℕ :=
  (s.filter p).card

-- Define the probability that the sum on the top faces of both dice is greater than 9
def probability_sum_greater_than_9 : ℚ :=
  (count event_space favorable_event) / event_space.card

-- State the theorem
theorem probability_two_dice_sum_greater_than_9_eq_1_over_6 :
  probability_sum_greater_than_9 = 1 / 6 := by
  sorry

end probability_two_dice_sum_greater_than_9_eq_1_over_6_l528_528972


namespace square_side_length_l528_528614

theorem square_side_length {s : ℝ} (h1 : 4 * s = 60) : s = 15 := 
by
  linarith

end square_side_length_l528_528614


namespace angle_of_isosceles_trapezoid_in_monument_l528_528229

-- Define the larger interior angle x of an isosceles trapezoid in the monument
def larger_interior_angle_of_trapezoid (x : ℝ) : Prop :=
  ∃ n : ℕ, 
    n = 12 ∧
    ∃ α : ℝ, 
      α = 360 / (2 * n) ∧
      ∃ θ : ℝ, 
        θ = (180 - α) / 2 ∧
        x = 180 - θ

-- The theorem stating the larger interior angle x is 97.5 degrees
theorem angle_of_isosceles_trapezoid_in_monument : larger_interior_angle_of_trapezoid 97.5 :=
by 
  sorry

end angle_of_isosceles_trapezoid_in_monument_l528_528229


namespace never_exceeds_100_after_squared_presses_l528_528598

theorem never_exceeds_100_after_squared_presses (x : ℕ) (h : x = 1) : ∀ n : ℕ, (x ^ (2 ^ n)) ≤ 100 := 
by 
  intro n
  rw [h]
  simp
  sorry

end never_exceeds_100_after_squared_presses_l528_528598


namespace both_questions_correct_l528_528965

def total_students := 100
def first_question_correct := 75
def second_question_correct := 30
def neither_question_correct := 20

theorem both_questions_correct :
  (first_question_correct + second_question_correct - (total_students - neither_question_correct)) = 25 :=
by
  sorry

end both_questions_correct_l528_528965


namespace find_a_and_b_l528_528359

theorem find_a_and_b (a b : ℝ) (h1 : b ≠ 0) 
  (h2 : (ab = a + b ∨ ab = a - b ∨ ab = a / b) 
  ∧ (a + b = a - b ∨ a + b = a / b) 
  ∧ (a - b = a / b)) : 
  (a = 1 / 2 ∨ a = -1 / 2) ∧ b = -1 := by
  sorry

end find_a_and_b_l528_528359


namespace next_month_has_5_Wednesdays_l528_528014

-- The current month characteristics
def current_month_has_5_Saturdays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_5_Sundays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_4_Mondays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def current_month_has_4_Fridays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def month_ends_on_Sunday : Prop := ∃ day : ℕ, day = 30 ∧ day % 7 = 0

-- Prove next month has 5 Wednesdays
theorem next_month_has_5_Wednesdays 
  (h1 : current_month_has_5_Saturdays) 
  (h2 : current_month_has_5_Sundays)
  (h3 : current_month_has_4_Mondays)
  (h4 : current_month_has_4_Fridays)
  (h5 : month_ends_on_Sunday) :
  ∃ month : ℕ, month = 31 ∧ ∃ day : ℕ, day = 5 := 
sorry

end next_month_has_5_Wednesdays_l528_528014


namespace math_problem_l528_528054

-- Given the roots a, b, and c of the polynomial x^3 - 9x^2 + 11x - 1 = 0
variables {a b c : ℝ}
-- Conditions based on Vieta's formulas
variables (h1 : a + b + c = 9)
variables (h2 : a * b + a * c + b * c = 11)
variables (h3 : a * b * c = 1)

-- Define s based on problem statement
def s := (Real.sqrt a + Real.sqrt b + Real.sqrt c)

-- Prove that s^4 - 18s^2 - 8s = -37
theorem math_problem (h1 : a + b + c = 9) (h2 : a * b + a * c + b * c = 11) (h3 : a * b * c = 1) :
  s ^ 4 - 18 * s ^ 2 - 8 * s = -37 :=
begin
  sorry
end

end math_problem_l528_528054


namespace polynomial_divisibility_l528_528854

open Polynomial

noncomputable def A : Polynomial ℝ 
noncomputable def B : Polynomial ℝ 

theorem polynomial_divisibility (A B : ℝ[X][Y]) :
  (∃ (y : ℝ), ∀ᶠ y in at_top, ∃ Cx : ℝ[X], A = B * Cx) 
  ∧ (∃ (x : ℝ), ∀ᶠ x in at_top, ∃ Cy : ℝ[Y], A = B * Cy) 
  → ∃ C : Polynomial ℝ, A = B * C := 
by
  sorry

end polynomial_divisibility_l528_528854


namespace minimum_value_of_f_on_interval_l528_528130

def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem minimum_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (-3 : ℝ) (1 : ℝ), (∀ y ∈ set.Icc (-3 : ℝ) (1 : ℝ), f(x) ≤ f(y)) ∧ f(x) = -11 := 
by 
  sorry

end minimum_value_of_f_on_interval_l528_528130


namespace calculate_c_l528_528766

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6
def g (x : ℝ) : ℝ := -f x
def h (x : ℝ) : ℝ := f (-x)

theorem calculate_c : 
  let a := (λ a, ∃ x, f x = g x) 2
  let b := (λ b, ∃ x, f x = h x) 1
  let c := 10 * b + a :=
  c = 12 := 
by 
  sorry

end calculate_c_l528_528766


namespace infinite_series_sum_l528_528722

noncomputable def S : ℝ :=
∑' n, (if n % 3 == 0 then 1 / (3 ^ (n / 3)) else if n % 3 == 1 then -1 / (3 ^ (n / 3 + 1)) else -1 / (3 ^ (n / 3 + 2)))

theorem infinite_series_sum : S = 15 / 26 := by
  sorry

end infinite_series_sum_l528_528722


namespace misha_current_dollars_l528_528875

variable (x : ℕ)

def misha_needs_more : ℕ := 13
def total_amount : ℕ := 47

theorem misha_current_dollars : x = total_amount - misha_needs_more → x = 34 :=
by
  sorry

end misha_current_dollars_l528_528875


namespace incorrect_statement_l528_528956

theorem incorrect_statement : ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) :=
by
  -- Proof goes here
  sorry

end incorrect_statement_l528_528956


namespace ratio_father_to_daughter_l528_528603

def father's_age := 40
def daughter's_age := 10

theorem ratio_father_to_daughter : (father's_age / daughter's_age) = 4 :=
by
  have h : daughters_age ≠ 0 := by norm_num
  exact (div_eq_of_eq_mul (by norm_num)).mpr (rfl)

end ratio_father_to_daughter_l528_528603


namespace speed_in_still_water_l528_528210

theorem speed_in_still_water (upstream downstream : ℝ) (h_upstream : upstream = 37) (h_downstream : downstream = 53) : 
  (upstream + downstream) / 2 = 45 := 
by
  sorry

end speed_in_still_water_l528_528210


namespace geometric_a1_a3_a15_arithmetic_a1_a2_ak_l528_528314

noncomputable def seq (a : ℕ) (n : ℕ) : ℚ := n / (n + a)

theorem geometric_a1_a3_a15 (a : ℕ) (ha : a > 0) :
  (seq a 1) * (seq a 15) = (seq a 3) ^ 2 ↔ a = 9 :=
begin
  sorry
end

theorem arithmetic_a1_a2_ak (a k : ℕ) (ha : a > 0) (hk : k ≥ 3) :
  (seq a 1) + (seq a k) = 2 * (seq a 2) ↔ (a = 1 ∧ k = 5) ∨ (a = 2 ∧ k = 4) :=
begin
  sorry
end

end geometric_a1_a3_a15_arithmetic_a1_a2_ak_l528_528314


namespace not_q_is_false_l528_528321

variable (n : ℤ)

-- Definition of the propositions
def p (n : ℤ) : Prop := 2 * n - 1 % 2 = 1 -- 2n - 1 is odd
def q (n : ℤ) : Prop := (2 * n + 1) % 2 = 0 -- 2n + 1 is even

-- Proof statement: Not q is false, meaning q is false
theorem not_q_is_false (n : ℤ) : ¬ q n = False := sorry

end not_q_is_false_l528_528321


namespace TrigPowerEqualsOne_l528_528253

theorem TrigPowerEqualsOne : ((Real.cos (160 * Real.pi / 180) + Real.sin (160 * Real.pi / 180) * Complex.I)^36 = 1) :=
by
  sorry

end TrigPowerEqualsOne_l528_528253


namespace problem_statement_l528_528862

-- Definitions of the problem conditions
def problem_conditions (n : ℕ) (x : Fin n → ℝ) : Prop :=
  (n ≥ 1) ∧ (∀ i, 0 < x i) ∧ (Finset.univ.sum x = 1)

-- Statement of the main inequality in the problem
theorem problem_statement (n : ℕ) (x : Fin n → ℝ) (h : problem_conditions n x) :
  1 ≤ ∑ i : Fin n, 
      x i / (Real.sqrt (1 + ∑ j in Finset.univ.filter (·.val < i.val), x j) * 
              Real.sqrt (∑ j in Finset.univ.filter (·.val ≥ i.val), x j)) 
      ∧ 
      ∑ i : Fin n, 
      x i / (Real.sqrt (1 + ∑ j in Finset.univ.filter (·.val < i.val), x j) * 
              Real.sqrt (∑ j in Finset.univ.filter (·.val ≥ i.val), x j)) 
      < Real.pi / 2 :=
sorry

end problem_statement_l528_528862


namespace beau_age_today_l528_528659

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end beau_age_today_l528_528659


namespace rain_probability_tel_aviv_l528_528496

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l528_528496


namespace probability_red_red_red_l528_528565

-- Definition of probability for picking three red balls without replacement
def total_balls := 21
def red_balls := 7
def blue_balls := 9
def green_balls := 5

theorem probability_red_red_red : 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) * ((red_balls - 2) / (total_balls - 2)) = 1 / 38 := 
by sorry

end probability_red_red_red_l528_528565


namespace units_digit_of_N_l528_528844

def P (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (*) 1

def S (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (+) 0

theorem units_digit_of_N (a b c N : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9)
  (hN : N = 100 * a + 10 * b + c) (h : N = P N + S N) : c = 9 :=
by 
  sorry

end units_digit_of_N_l528_528844


namespace roads_with_five_possible_roads_with_four_not_possible_l528_528808

-- Problem (a)
theorem roads_with_five_possible :
  ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 5) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

-- Problem (b)
theorem roads_with_four_not_possible :
  ¬ ∃ (cities : Fin 16 → Finset (Fin 16)),
  (∀ c, cities c = {d | d ≠ c ∧ d ∈ cities c}) ∧
  (∀ c, (cities c).card ≤ 4) ∧
  (∀ c d, d ≠ c → ∃ e, e ≠ c ∧ e ≠ d ∧ d ∈ cities c ∪ {e}) := by
  sorry

end roads_with_five_possible_roads_with_four_not_possible_l528_528808


namespace beau_age_today_l528_528661

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end beau_age_today_l528_528661


namespace deductive_reasoning_l528_528697

structure Person :=
  (name : String)

def makesMistakes (p : Person) : Prop := True

axiom everyoneMakesMistakes : ∀ (p : Person), makesMistakes p
axiom oldWang : Person
axiom oldWangIsAPerson : True  -- Essentially redundant but included to match problem conditions

theorem deductive_reasoning :
  makesMistakes oldWang :=
by
  apply everyoneMakesMistakes
  sorry

end deductive_reasoning_l528_528697


namespace circle_geometry_problem_l528_528449

noncomputable def construct_circle_problem : Prop :=
  ∃ (A B C D E : ℝ×ℝ), -- there exist points on a plane
    (∃ (O : ℝ×ℝ) (r : ℝ), r > 0 ∧  -- O is the center, and r is the radius
    dist O A = r ∧
    dist O B = r ∧
    dist O C = r ∧
    dist O D = r) ∧ -- all points lie on the circle with center O and radius r
    (E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D) ∧ -- E is distinct from A, B, C, D
    let θ := 75 in
    let BE := dist B E in
    let DE := dist D E in
    let AE := dist A E in
    let EC := dist E C in
    let AEB := ∠ A E B in
    let ADE := ∠ A D E in
    (BE = 4 ∧ DE = 8) ∧  -- given segment lengths
    (ADE = θ ∧ ∠ C B E = θ) ∧  -- given angles
    let AB := dist A B in
    (AB^2 = 80 + 32*real.sqrt 3) ∧  -- verifying AB^2
    (80 + 32 + 3 = 115) -- calculate a + b + c

theorem circle_geometry_problem : construct_circle_problem :=
sorry -- Here is where the proof would go.

end circle_geometry_problem_l528_528449


namespace find_d_plus_e_l528_528810

-- Noncomputable Theory because we deal with nonconstructive proofs
noncomputable theory

-- Definitions based on the problem condition
def s : ℕ := 68
def a : ℕ := 20 -- As derived logically
def b : ℕ := 17 -- As derived logically
def c : ℕ := 21 -- As derived logically
def d : ℕ := 32 -- As derived logically
def e : ℕ := 16 -- As derived logically

-- Given conditions
def cond1 : 30 + e + 24 = s := by sorry
def cond2 : 15 + c + d = s := by sorry
def cond3 : a + 28 + b = s := by sorry
def cond4 : 30 + 15 + a = s := by sorry
def cond5 : e + c + 28 = s := by sorry
def cond6 : 24 + d + b = s := by sorry
def cond7 : 30 + c + b = s := by sorry
def cond8 : a + c + 24 = s := by sorry

-- Main theorem 
theorem find_d_plus_e : d + e = 48 := by
  exact sorry

end find_d_plus_e_l528_528810


namespace distance_P_to_plane_xOz_l528_528041

noncomputable def distance_to_plane_xOz (x y z : ℝ) : ℝ :=
  |y|

theorem distance_P_to_plane_xOz :
  let P := (-1 : ℝ, -2 : ℝ, -3 : ℝ)
  distance_to_plane_xOz P.1 P.2 P.3 = 2 :=
by
  simp [distance_to_plane_xOz]
  sorry

end distance_P_to_plane_xOz_l528_528041


namespace remainder_division_l528_528372

theorem remainder_division : 
  polynomial.eval (-1) (2 * X^3 - 3 * X^2 + X - 1) = -7 :=
by sorry

end remainder_division_l528_528372


namespace complex_in_third_quadrant_iff_l528_528688

-- Define complex numbers
def complex_number_z (a : ℝ) : ℂ := (3 - a * complex.I) / complex.I

-- Define condition for being in the third quadrant
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Prove that the complex number corresponds to a point in the third quadrant
theorem complex_in_third_quadrant_iff (a : ℝ) : in_third_quadrant (complex_number_z a) ↔ a > 0 :=
sorry

end complex_in_third_quadrant_iff_l528_528688


namespace golden_state_total_points_l528_528032

theorem golden_state_total_points :
  let draymond_points := 12
  let curry_points := 2 * draymond_points
  let kelly_points := 9
  let durant_points := 2 * kelly_points
  let klay_points := draymond_points / 2
  draymond_points + curry_points + kelly_points + durant_points + klay_points = 69 :=
by
  let draymond_points := 12
  let curry_points := 2 * draymond_points
  let kelly_points := 9
  let durant_points := 2 * kelly_points
  let klay_points := draymond_points / 2
  calc
    draymond_points + curry_points + kelly_points + durant_points + klay_points
    = 12 + (2 * 12) + 9 + (2 * 9) + (12 / 2) : by sorry
    = 69 : by sorry

end golden_state_total_points_l528_528032


namespace meals_left_to_distribute_l528_528251

theorem meals_left_to_distribute (meals_prepared : ℕ) (additional_meals : ℕ) (meals_given_away : ℕ) :
  meals_prepared = 113 → additional_meals = 50 → meals_given_away = 85 → (meals_prepared + additional_meals - meals_given_away) = 78 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end meals_left_to_distribute_l528_528251


namespace sum_arithmetic_sequence_l528_528925

theorem sum_arithmetic_sequence (m : ℕ) (S : ℕ → ℕ) 
  (h1 : S m = 30) 
  (h2 : S (3 * m) = 90) : 
  S (2 * m) = 60 := 
sorry

end sum_arithmetic_sequence_l528_528925


namespace two_xy_plus_y_sq_eq_two_l528_528192

noncomputable def sqrt11 := Real.sqrt 11

def x : ℝ := 3
def y : ℝ := sqrt11 - x

theorem two_xy_plus_y_sq_eq_two :
  2 * x * y + y^2 = 2 :=
by
  -- Insert the proof here
  sorry

end two_xy_plus_y_sq_eq_two_l528_528192


namespace min_gumballs_for_four_same_color_l528_528208

theorem min_gumballs_for_four_same_color (r w b g : ℕ) (h_r : r = 10) (h_w : w = 12) (h_b : b = 9) (h_g : g = 11) :
  ∃ n, n = 13 ∧ (∀ picked_gumballs : multiset ℕ, multiset.card picked_gumballs = n → 
                  (∃ color, multiset.count color picked_gumballs ≥ 4)) :=
by
  use 13
  split
  · rfl
  · intro picked_gumballs
    intro h_picked
    sorry

end min_gumballs_for_four_same_color_l528_528208


namespace quadratic_roots_sum_l528_528419

-- Definition of conditions
variables (α β : ℝ)
hypothesis (hαβ : α^2 + α - 2023 = 0 ∧ β^2 + β - 2023 = 0)
hypothesis (vieta : α + β = -1)

-- Proof statement
theorem quadratic_roots_sum (α β : ℝ) (hαβ : α^2 + α - 2023 = 0) (vieta : α + β = -1) : α^2 + 2*α + β = 2022 :=
by 
  sorry

end quadratic_roots_sum_l528_528419


namespace find_height_of_box_l528_528609

theorem find_height_of_box :
  ∃ (h : ℝ), (let w : ℝ := 10
                let l : ℝ := 20
                let A : ℝ := 40
                let center_distance (a b : ℝ) : ℝ := Real.sqrt (a^2 + (b/2)^2)
                let d₁ := center_distance w l
                let d₂ := center_distance w h
                let d₃ := center_distance l h
                -- Calculating the height h using the area of the triangle
                let base := d₁
                let height_altitude := (2 * A) / base
                let pythagorean_expr := height_altitude^2 + l^2 = (h/2)^2
                pythagorean_expr ∧ h = 24 * Real.sqrt 21 / 5) :=
begin
  sorry
end

end find_height_of_box_l528_528609


namespace average_price_per_pound_correct_l528_528829

theorem average_price_per_pound_correct :
  let joan_pounds := 3
      joan_price_per_pound := 2.80
      grant_pounds := 2
      grant_price_per_pound := 1.80
      total_cost := (joan_pounds * joan_price_per_pound) + (grant_pounds * grant_price_per_pound)
      total_pounds := joan_pounds + grant_pounds
  in (total_cost / total_pounds) = 2.40 :=
by
  sorry

end average_price_per_pound_correct_l528_528829


namespace grid_divisible_by_L_trinominoes_l528_528007

-- Define the structure of an L-shaped trinomino
structure L_trinomino :=
  (cells : Finset (ℕ × ℕ))
  (card_eq : cells.card = 3)
  (is_L_shape : -- your definition of an L-shape condition here
    sorry)

-- Define the main problem
theorem grid_divisible_by_L_trinominoes (n : ℕ) (h : n = 333) :
  ∀ grid : Finset (ℕ × ℕ), grid.card = (6*n + 1) * (6*n + 1) - 1 →
  ∃ sets : Finset (Finset (ℕ × ℕ)),
    (∀ t in sets, L_trinomino t) ∧
    Finset.bUnion sets id = grid ∧
    sets.pairwise_disjoint id :=
begin
  sorry -- proof not required as per instructions
end

end grid_divisible_by_L_trinominoes_l528_528007


namespace remainder_of_T_2027_l528_528510

theorem remainder_of_T_2027 :
  let T := ∑ k in Finset.range 97, Nat.choose 2024 k
  2027.Prime → T % 2027 = 375 :=
by
  intros h
  let T := ∑ k in Finset.range 97, Nat.choose 2024 k
  have h1 : 2027.Prime := h
  sorry

end remainder_of_T_2027_l528_528510


namespace find_m_from_parallel_l528_528338

theorem find_m_from_parallel (m : ℝ) : 
  (∃ (A B : ℝ×ℝ), A = (-2, m) ∧ B = (m, 4) ∧
  (∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -1 ∧
  (a * (B.1 - A.1) + b * (B.2 - A.2) = 0)) ) 
  → m = -8 :=
by
  sorry

end find_m_from_parallel_l528_528338


namespace grow_path_product_l528_528671

noncomputable def maxGrowingPathLength : ℕ := 12
noncomputable def numGrowingPaths : ℕ := 15

theorem grow_path_product (mr : ℕ) (m : ℕ) (r : ℕ)
  (h1 : m = maxGrowingPathLength)
  (h2 : r = numGrowingPaths) :
  mr = m * r :=
by
  rw [h1, h2]
  exact rfl

end grow_path_product_l528_528671


namespace ratio_of_money_with_Gopal_and_Krishan_l528_528517

theorem ratio_of_money_with_Gopal_and_Krishan 
  (R G K : ℕ) 
  (h1 : R = 735) 
  (h2 : K = 4335) 
  (h3 : R * 17 = G * 7) :
  G * 4335 = 1785 * K :=
by
  sorry

end ratio_of_money_with_Gopal_and_Krishan_l528_528517


namespace when_was_p_turned_off_l528_528541

noncomputable def pipe_p_rate := (1/12 : ℚ)  -- Pipe p rate
noncomputable def pipe_q_rate := (1/15 : ℚ)  -- Pipe q rate
noncomputable def combined_rate := (3/20 : ℚ) -- Combined rate of p and q when both are open
noncomputable def time_after_p_off := (1.5 : ℚ)  -- Time for q to fill alone after p is off
noncomputable def fill_cistern (t : ℚ) := combined_rate * t + pipe_q_rate * time_after_p_off

theorem when_was_p_turned_off (t : ℚ) : fill_cistern t = 1 ↔ t = 6 := sorry

end when_was_p_turned_off_l528_528541


namespace system_of_equations_solution_system_of_inequalities_no_solution_l528_528588

-- Problem 1: Solving system of linear equations
theorem system_of_equations_solution :
  ∃ x y : ℝ, x - 3*y = -5 ∧ 2*x + 2*y = 6 ∧ x = 1 ∧ y = 2 := by
  sorry

-- Problem 2: Solving the system of inequalities
theorem system_of_inequalities_no_solution :
  ¬ (∃ x : ℝ, 2*x < -4 ∧ (1/2)*x - 5 > 1 - (3/2)*x) := by
  sorry

end system_of_equations_solution_system_of_inequalities_no_solution_l528_528588


namespace deepak_present_age_l528_528970

def present_age_rahul (x : ℕ) : ℕ := 4 * x
def present_age_deepak (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age : ∀ (x : ℕ), 
  (present_age_rahul x + 22 = 26) →
  present_age_deepak x = 3 := 
by
  intros x h
  sorry

end deepak_present_age_l528_528970


namespace rhombus_diagonal_sum_l528_528612

theorem rhombus_diagonal_sum
  (d1 d2 : ℝ)
  (h1 : d1 ≤ 6)
  (h2 : 6 ≤ d2)
  (side_len : ℝ)
  (h_side : side_len = 5)
  (rhombus_relation : d1^2 + d2^2 = 4 * side_len^2) :
  d1 + d2 ≤ 14 :=
sorry

end rhombus_diagonal_sum_l528_528612


namespace spent_amount_l528_528559

def initial_amount : ℕ := 15
def final_amount : ℕ := 11

theorem spent_amount : initial_amount - final_amount = 4 :=
by
  sorry

end spent_amount_l528_528559


namespace total_cost_correct_l528_528073

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end total_cost_correct_l528_528073


namespace determine_phi_value_l528_528124

noncomputable def translate_sin_to_cos (phi : Real) : Prop :=
  (∀ x, sin (x + phi) = cos (x - π / 6))

theorem determine_phi_value (phi : Real) (h : 0 ≤ phi ∧ phi < 2 * π) :
  translate_sin_to_cos phi ↔ phi = π / 3 :=
sorry

end determine_phi_value_l528_528124


namespace ratio_avg_speeds_l528_528568

-- Define the variables according to the conditions
variables (distance_AB time_Eddy distance_AC time_Freddy : ℝ)
variables (h_AB : distance_AB = 540)
variables (h_AC : distance_AC = 300)
variables (h_TE : time_Eddy = 3)
variables (h_TF : time_Freddy = 4)

-- Define the average speeds
def avgSpeed_Eddy := distance_AB / time_Eddy
def avgSpeed_Freddy := distance_AC / time_Freddy

-- State the theorem
theorem ratio_avg_speeds : avgSpeed_Eddy / avgSpeed_Freddy = 2.4 :=
by
  rw [h_AB, h_TE, h_AC, h_TF]
  dsimp [avgSpeed_Eddy, avgSpeed_Freddy]
  norm_num
  sorry

end ratio_avg_speeds_l528_528568


namespace number_of_girls_and_boys_l528_528516

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) := g = 4 * (g + b) / 7 ∧ b = 3 * (g + b) / 7
def total_students (g b : ℕ) := g + b = 56

-- The main proof statement
theorem number_of_girls_and_boys (g b : ℕ) 
  (h_ratio : ratio_girls_to_boys g b)
  (h_total : total_students g b) : 
  g = 32 ∧ b = 24 :=
by {
  sorry
}

end number_of_girls_and_boys_l528_528516


namespace greatest_possible_value_l528_528479

theorem greatest_possible_value (x y : ℝ) (h1 : -4 ≤ x) (h2 : x ≤ -2) (h3 : 2 ≤ y) (h4 : y ≤ 4) : 
  ∃ z: ℝ, z = (x + y) / x ∧ (∀ z', z' = (x' + y') / x' ∧ -4 ≤ x' ∧ x' ≤ -2 ∧ 2 ≤ y' ∧ y' ≤ 4 → z' ≤ z) ∧ z = 0 :=
by
  sorry

end greatest_possible_value_l528_528479


namespace box_dimension_triples_l528_528217

theorem box_dimension_triples (N : ℕ) :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (1 / a + 1 / b + 1 / c = 1 / 8) → ∃ k, k = N := sorry 

end box_dimension_triples_l528_528217


namespace find_18th_permutation_l528_528917

theorem find_18th_permutation : 
  list.nth_le (list.permutations [1, 2, 5, 6].sort) 17 sorry = [5, 6, 2, 1] := sorry

end find_18th_permutation_l528_528917


namespace complete_square_cpjq_l528_528101

theorem complete_square_cpjq (j : ℝ) (c p q : ℝ) (h : 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) :
  c = 8 ∧ p = -3/8 ∧ q = 119/8 → q / p = -119/3 :=
by
  intros
  cases a with hc hpq
  cases hpq with hp hq
  rw [hc, hp, hq]
  have hp_ne_zero : (-3 / 8) ≠ 0 := by norm_num
  field_simp [hp_ne_zero]
  norm_num
  sorry

end complete_square_cpjq_l528_528101


namespace square_park_area_l528_528275

theorem square_park_area (side_length : ℝ) (h : side_length = 200) : side_length * side_length = 40000 := by
  sorry

end square_park_area_l528_528275


namespace cost_of_shoes_is_150_l528_528438

def cost_sunglasses : ℕ := 50
def pairs_sunglasses : ℕ := 2
def cost_jeans : ℕ := 100

def cost_basketball_cards : ℕ := 25
def decks_basketball_cards : ℕ := 2

-- Define the total amount spent by Mary and Rose
def total_mary : ℕ := cost_sunglasses * pairs_sunglasses + cost_jeans
def cost_shoes (total_rose : ℕ) (cost_cards : ℕ) : ℕ := total_rose - cost_cards

theorem cost_of_shoes_is_150 (total_spent : ℕ) :
  total_spent = total_mary →
  cost_shoes total_spent (cost_basketball_cards * decks_basketball_cards) = 150 :=
by
  intro h
  sorry

end cost_of_shoes_is_150_l528_528438


namespace unique_representation_l528_528864

theorem unique_representation {p x y : ℕ} 
  (hp : p > 2 ∧ Prime p) 
  (h : 2 * y = p * (x + y)) 
  (hx : x ≠ y) : 
  ∃ x y : ℕ, (1/x + 1/y = 2/p) ∧ x ≠ y := 
sorry

end unique_representation_l528_528864


namespace total_time_spent_l528_528190

-- Definitions based on the conditions
def number_of_chairs := 2
def number_of_tables := 2
def minutes_per_piece := 8
def total_pieces := number_of_chairs + number_of_tables

-- The statement we want to prove
theorem total_time_spent : total_pieces * minutes_per_piece = 32 :=
by
  sorry

end total_time_spent_l528_528190


namespace max_a_value_l528_528353

noncomputable def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5 * a * k + 3) * x + 7

theorem max_a_value : ∀ (k a : ℝ), (0 <= k) → (k <= 2) →
  (∀ (x1 : ℝ), (k <= x1) → (x1 <= k + a) →
  ∀ (x2 : ℝ), (k + 2 * a <= x2) → (x2 <= k + 4 * a) →
  f x1 k a >= f x2 k a) → 
  a <= (2 * Real.sqrt 6 - 4) / 5 := 
sorry

end max_a_value_l528_528353


namespace distance_between_truck_and_car_l528_528624

def truck_speed : ℝ := 65 -- km/h
def car_speed : ℝ := 85 -- km/h
def time_minutes : ℝ := 3 -- minutes
def time_hours : ℝ := time_minutes / 60 -- converting minutes to hours

def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem distance_between_truck_and_car :
  let truck_distance := distance_traveled truck_speed time_hours
  let car_distance := distance_traveled car_speed time_hours
  truck_distance - car_distance = -1 := -- the distance is 1 km but negative when subtracting truck from car
by {
  sorry
}

end distance_between_truck_and_car_l528_528624


namespace hyperbola_eccentricity_l528_528732

theorem hyperbola_eccentricity {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  let C := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 in
  let A := (a, 0) in
  let circle_A := ∀ x y : ℝ, (x - a)^2 + y^2 = b^2 in
  let asymptote := bx + ay = 0 in
  ∀ M N : ℝ × ℝ, 
    (M ≠ N) → 
    (circle_A M.1 M.2) →
    (circle_A N.1 N.2) →
    (asymptote M.1 M.2) →
    (asymptote N.1 N.2) →
    angle A M N = 120 :=
  ∃ e : ℝ, e = 2 := sorry

end hyperbola_eccentricity_l528_528732


namespace distance_after_3_minutes_l528_528636

-- Define the given speeds and time interval
def speed_truck : ℝ := 65 -- in km/h
def speed_car : ℝ := 85 -- in km/h
def time_minutes : ℝ := 3 -- in minutes

-- The equivalent time in hours
def time_hours : ℝ := time_minutes / 60

-- Calculate the distances travelled by the truck and the car
def distance_truck : ℝ := speed_truck * time_hours
def distance_car : ℝ := speed_car * time_hours

-- Define the distance between the truck and the car
def distance_between : ℝ := distance_car - distance_truck

-- Theorem: The distance between the truck and car after 3 minutes is 1 km.
theorem distance_after_3_minutes : distance_between = 1 := by
  sorry

end distance_after_3_minutes_l528_528636


namespace perpendicular_intersection_l528_528964

theorem perpendicular_intersection {ABC A1B1C1 : Triangle}
(H_perpendiculars_intersect : Prove_perpendiculars_intersect ABC A1B1C1) :
Prove_perpendiculars_intersect A1B1C1 ABC :=
sorry

end perpendicular_intersection_l528_528964


namespace prove_arithmetic_sequence_l528_528315

variable {a : ℕ → ℤ} -- Define the arithmetic sequence

-- Conditions given in the problem
def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of the first n terms
def sum_of_first_n_terms (a : ℕ → ℤ) : ℕ → ℤ
| 0     := 0
| (n+1) := a n + sum_of_first_n_terms n

-- Given conditions
def condition1 : Prop := sum_of_first_n_terms a 4 = 24
def condition2 : Prop := sum_of_first_n_terms a 7 = 63

-- General term of the arithmetic sequence
def general_term (n : ℕ) : ℤ := 2 * n + 1

-- Sum of the first 10 terms
def sum_of_first_10_terms : ℕ := 10

-- Proof to show the general term and sum of first 10 terms are correct
theorem prove_arithmetic_sequence (a : ℕ → ℤ) 
    (h1 : is_arithmetic_sequence a)
    (h2 : condition1)
    (h3 : condition2):
    (∀ n : ℕ, a n = general_term n) ∧ (sum_of_first_n_terms a 10 = 120) :=
by
  sorry

end prove_arithmetic_sequence_l528_528315


namespace tan_eq_sin_intersections_l528_528399

theorem tan_eq_sin_intersections :
  ∃ n : ℕ, n = 5 ∧ ∀ x ∈ range(-2 * Real.pi, 2 * Real.pi), tan x = sin x → x ∈ { -2 * Real.pi, -Real.pi, 0, Real.pi, 2 * Real.pi } :=
by
  sorry

end tan_eq_sin_intersections_l528_528399


namespace sum_solution_equation_l528_528784

theorem sum_solution_equation (n : ℚ) : (∃ x : ℚ, (n / x = 3 - n) ∧ (x = 1 / (n + (3 - n)))) → n = 3 / 4 := by
  intros h
  sorry

end sum_solution_equation_l528_528784


namespace triangle_DEF_area_l528_528157

-- Define the conditions of the problem
def right_triangle (D E F : ℝ) (DE : ℝ) (EF : ℝ) (angle_D : ℝ) : Prop :=
  angle_D = 90 ∧ DE = 8 ∧ EF = 10

-- Prove the area of triangle DEF
theorem triangle_DEF_area (D E F : ℝ) (angle_D : ℝ) (DE : ℝ) (EF : ℝ) (DF : ℝ) :
  right_triangle D E F DE EF angle_D →
  DF^2 = EF^2 - DE^2 →
  DF = 6 →
  (1/2 * DE * DF = 24) :=
begin
  intro h,
  intro hyp,
  intro df_val,
  sorry
end

end triangle_DEF_area_l528_528157


namespace rain_probability_tel_aviv_l528_528490

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l528_528490


namespace penny_dime_halfdollar_probability_l528_528895

-- Define the universe of outcomes for five coin flips
def coin_flip_outcomes : Finset (vector Bool 5) := Finset.univ

-- Define a predicate that checks if the penny, dime, and half-dollar come up the same
def same_penny_dime_halfdollar (v : vector Bool 5) : Prop :=
  v.head = v.nth 2 ∧ v.head = v.nth 4

-- The proof problem: prove that the probability of the penny, dime, and half-dollar being the same is 1/4
theorem penny_dime_halfdollar_probability :
  (Finset.filter same_penny_dime_halfdollar coin_flip_outcomes).card / coin_flip_outcomes.card = 1 / 4 :=
by
  sorry

end penny_dime_halfdollar_probability_l528_528895


namespace notebooks_last_days_l528_528410

theorem notebooks_last_days (n p u : Nat) (total_pages days : Nat) 
  (h1 : n = 5)
  (h2 : p = 40)
  (h3 : u = 4)
  (h_total : total_pages = n * p)
  (h_days  : days = total_pages / u) :
  days = 50 := 
by
  sorry

end notebooks_last_days_l528_528410


namespace find_z_l528_528267

-- Define the determinant for a 2x2 matrix
def determinant (a b c d : ℂ) : ℂ :=
  a * d - b * c

-- Given a condition on a complex matrix and value
theorem find_z (z : ℂ) (i : ℂ := complex.I) :
  determinant z i 1 i = 1 + i → z = 2 - i :=
by
  intro h
  sorry

end find_z_l528_528267


namespace magnitude_of_a_plus_b_l528_528062

open Real

noncomputable def magnitude (x y : ℝ) : ℝ :=
  sqrt (x^2 + y^2)

theorem magnitude_of_a_plus_b (m : ℝ) (a b : ℝ × ℝ)
  (h₁ : a = (m+2, 1))
  (h₂ : b = (1, -2*m))
  (h₃ : (a.1 * b.1 + a.2 * b.2 = 0)) :
  magnitude (a.1 + b.1) (a.2 + b.2) = sqrt 34 :=
by
  sorry

end magnitude_of_a_plus_b_l528_528062


namespace loss_percentage_cp_1500_sp_1260_l528_528903

theorem loss_percentage_cp_1500_sp_1260 :
  let cost_price := 1500
  let selling_price := 1260
  let loss := cost_price - selling_price
  let loss_percentage := (loss / cost_price.toFloat) * 100
  loss_percentage = 16 := by
  sorry

end loss_percentage_cp_1500_sp_1260_l528_528903


namespace time_to_cross_bridge_is_30_secs_l528_528618

def length_of_train : ℝ := 120
def speed_of_train_kmph : ℝ := 45
def length_of_bridge : ℝ := 255

def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600
def total_distance : ℝ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℝ := total_distance / speed_of_train_mps

theorem time_to_cross_bridge_is_30_secs : time_to_cross_bridge = 30 :=
by
  -- Proof omitted
  sorry

end time_to_cross_bridge_is_30_secs_l528_528618


namespace prob_union_A_B_l528_528334

variable (A B C : Set ℕ) -- A, B, C are events, represented as subsets of a sample space (e.g., ℕ).
variable [MeasurableSpace ℕ]
variable (μ : MeasureTheory.Measure ℕ) -- μ is a probability measure on the sample space ℕ.

-- Assumptions:
-- 1. A and B are mutually exclusive
-- 2. B and C are mutually exclusive
-- 3. P(A) = 0.3
-- 4. P(C) = 0.6

theorem prob_union_A_B :
  MeasureTheory.ProbabilityTheory.mutually_disjoint [({A}, {B}), ({B}, {C})] ∧
  μ A = 0.3 ∧
  μ C = 0.6 →
  μ (A ∪ B) = 0.7 :=
by
  sorry

end prob_union_A_B_l528_528334


namespace travel_agency_cost_l528_528037

theorem travel_agency_cost (x : ℕ) (h : x > 40) : 
  let cost_A := 400 + 20 * x,
      cost_B := 240 + 24 * x in
  cost_A < cost_B :=
by
  simp only [h]
  sorry

end travel_agency_cost_l528_528037


namespace pair_natural_numbers_with_perfect_square_sum_l528_528819

theorem pair_natural_numbers_with_perfect_square_sum :
  ∃ f : ℕ → ℕ, (∀ A, A ≠ f A ∧ ∃ m : ℕ, A + f A = m^2) ∧ bijective f :=
sorry


end pair_natural_numbers_with_perfect_square_sum_l528_528819


namespace hyperbola_eccentricity_range_l528_528357

noncomputable def parabola_focus (a : ℝ) : ℝ × ℝ := (a, 0)

def hyperbola_asymptote_distance (a b : ℝ) (focus : ℝ × ℝ) : ℝ :=
  let (x, y) := focus
  abs ((b * x + a * y) / (sqrt (a^2 + b^2)))

theorem hyperbola_eccentricity_range (a b : ℝ) (e : ℝ) :
  y^2 = 8*x →
  hyperbola_asymptote_distance a b (parabola_focus 2) ≤ sqrt 3 →
  e = sqrt (1 + b^2 / a^2) →
  1 < e ∧ e ≤ 2 :=
by
  sorry

end hyperbola_eccentricity_range_l528_528357


namespace equation_of_line_AB_equation_of_line_AC_l528_528122

-- Definition of points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 (-5)
def B := Point.mk (-3) 3
def C := Point.mk 2 0

-- Definition of a line in Ax + By + C = 0 form
structure Line :=
  (A : ℝ)
  (B : ℝ)
  (C : ℝ)

-- Statements to prove
theorem equation_of_line_AB : 
  ∃ (line : Line), line.A = 8 ∧ line.B = 3 ∧ line.C = 15 ∧ 
  ∀ (P : Point), ((P = A) ∨ (P = B) → (line.A * P.x + line.B * P.y + line.C = 0)) :=
begin
  sorry
end

theorem equation_of_line_AC : 
  ∃ (line : Line), line.A = 5 ∧ line.B = -2 ∧ line.C = -10 ∧ 
  ∀ (P : Point), ((P = A) ∨ (P = C) → (line.A * P.x + line.B * P.y + line.C = 0)) :=
begin
  sorry
end

end equation_of_line_AB_equation_of_line_AC_l528_528122


namespace total_cost_of_materials_l528_528601

theorem total_cost_of_materials :
  let cost_of_gravel := 5.91 * 30.50 in
  let cost_of_sand := 8.11 * 40.50 in
  cost_of_gravel + cost_of_sand = 508.71 :=
by sorry

end total_cost_of_materials_l528_528601


namespace triangle_propositions_correctness_l528_528816

theorem triangle_propositions_correctness
    (A B C a b c : Real)
    (hA_pos : 0 < A) (hB_pos : 0 < B) (hC_pos : 0 < C)
    (hA_sum : A + B + C = π) :
  (∃ h1 : sin (2*A) = sin (2*B), false) ∧
  (∃ h2 : sin B = cos A, false) ∧
  (∃ h3 : sin A ^ 2 + sin B ^ 2 > sin C ^ 2, false) ∧
  (∀ (h4 : a / cos A = b / cos B ∧ a / cos A = c / cos C),
    ∃ (h_sine : a / sin A = b / sin B ∧ a / sin A = c / sin C)
        (h_tan : tan A = tan B ∧ tan A = tan C)
        (h_eq_angles : A = B ∧ A = C),
      true) :=
by {
  sorry
}

end triangle_propositions_correctness_l528_528816


namespace beau_age_today_l528_528656

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end beau_age_today_l528_528656


namespace sum_remaining_four_l528_528375

-- Given conditions
def avg (nums : List ℝ) : ℝ := nums.sum / nums.length

-- The eight numbers whose average is given as 5.2
def eight_numbers : List ℝ := sorry

-- The sum of some four numbers from these eight numbers
def sum_four_of_eight : ℝ := 21

-- The statement to prove in Lean
theorem sum_remaining_four (h_avg : avg eight_numbers = 5.2) (h_sum4 : sum_four_of_eight = 21) :
  (eight_numbers.sum - sum_four_of_eight) = 20.6 :=
sorry

end sum_remaining_four_l528_528375


namespace assign_positions_l528_528165

-- Define the sets of candidates and positions
def candidates := {'A', 'B', 'C', 'D', 'E'}
def positions := {'class_president', 'vice_president', 'youth_league_secretary'}

-- Additional constraints
def previous_positions : Prop :=
  ∀ (a b c : candidates),
    a ∈ {'A'} ∧ b ∈ {'B'} ∧ c ∈ {'C'} → 
    ¬ (a = 'class_president' ∧ b = 'vice_president' ∧ c = 'youth_league_secretary')

-- The theorem to prove
theorem assign_positions (A B C D E : Prop) : 
  ∃ (ways : nat), ways = 32 := 
sorry

end assign_positions_l528_528165


namespace correct_statement_l528_528173

theorem correct_statement : 
  ∃ (frustum : Type), 
    (∀ (Pyramid : Type) (parallel_plane_base_cut : Pyramid → frustum), 
      regular_frustum frustum → regular_pyramid Pyramid) :=
sorry

end correct_statement_l528_528173


namespace find_formula_and_range_l528_528869

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 4^x + a * 2^x + b

theorem find_formula_and_range
  (a b : ℝ)
  (h₀ : f 0 a b = 1)
  (h₁ : f (-1) a b = -5 / 4) :
  f x (-3) 3 = 4^x - 3 * 2^x + 3 ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ 2 → 1 ≤ f x (-3) 3 ∧ f x (-3) 3 ≤ 25) :=
by
  sorry

end find_formula_and_range_l528_528869


namespace angle_EDB_eq_120_l528_528787

variables {A B C E D : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace E] [MetricSpace D]
variables (a b c e d : ℝ) -- Coordinates of the points
variables (angle_BCD : ℝ)

-- Given conditions
def E_midpoint_AB (AB : ℝ) : Prop :=
  (e = (a + b) / 2)

def AD_2DC (AD DC : ℝ) : Prop :=
  (AD = 2 * DC)

def angle_BCD_eq_60 : Prop :=
  (angle_BCD = 60)

-- Theorem to prove
theorem angle_EDB_eq_120 (AB AD DC : ℝ) 
  (hE : E_midpoint_AB AB)
  (hAD : AD_2DC AD DC)
  (hBCD : angle_BCD_eq_60) :
  ∃ angle_EDB : ℝ, angle_EDB = 120 := by
  sorry

end angle_EDB_eq_120_l528_528787


namespace daily_caffeine_limit_l528_528526

variable (caffeine_per_cup : Nat) (number_of_cups : Nat) (excess_caffeine : Nat) (daily_limit : Nat)

-- Conditions
def condition1 := caffeine_per_cup = 80
def condition2 := number_of_cups = 3
def condition3 := excess_caffeine = 40
def condition4 := daily_limit = number_of_cups * caffeine_per_cup - excess_caffeine

-- Proposition to prove: Lisa's daily caffeine limit is 200 mg
theorem daily_caffeine_limit : condition1 ∧ condition2 ∧ condition3 → daily_limit = 200 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  simp [condition1, condition2, condition3, condition4, h1, h2, h3]
  exact rfl

end daily_caffeine_limit_l528_528526


namespace find_speed_m2_l528_528993

constant u : ℝ -- Initial vertical speed (20 m/s)
constant t : ℝ -- Time after explosion (1 s)
constant g : ℝ -- Acceleration due to gravity (10 m/s^2)
constant m1_ratio : ℝ -- Mass ratio of m1 (1)
constant m2_ratio : ℝ -- Mass ratio of m2 (2)
constant speed_m1_horiz : ℝ -- Horizontal speed of smaller fragment (16 m/s)

axiom initial_condition1 : u = 20
axiom initial_condition2 : t = 1
axiom initial_condition3 : g = 10
axiom initial_condition4 : m1_ratio = 1
axiom initial_condition5 : m2_ratio = 2
axiom initial_condition6 : speed_m1_horiz = 16

noncomputable def speed_m2 : ℝ :=
  real.sqrt ((speed_m1_horiz / 2)^2 + (u - g * t)^2)

theorem find_speed_m2 : speed_m2 = 12.8 :=
sorry

end find_speed_m2_l528_528993


namespace fractional_eq_solution_l528_528474

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l528_528474


namespace notebooks_last_days_l528_528411

theorem notebooks_last_days (n p u : Nat) (total_pages days : Nat) 
  (h1 : n = 5)
  (h2 : p = 40)
  (h3 : u = 4)
  (h_total : total_pages = n * p)
  (h_days  : days = total_pages / u) :
  days = 50 := 
by
  sorry

end notebooks_last_days_l528_528411


namespace cistern_width_l528_528204

theorem cistern_width (w : ℝ) (h_length : 10 = 10) (h_depth : 1.5 = 1.5) (h_surface_area : 134 = 134) :
  10 * w + 2 * (1.5 * 10) + 2 * (1.5 * w) = 134 → w = 8 :=
by
  intro h
  have h_eq : 10 * w + 30 + 3 * w = 134 := by
    rw [← h_surface_area, ← h, mul_add, mul_comm, ← mul_assoc]
    exact h
  sorry

end cistern_width_l528_528204


namespace seat_arrangements_l528_528540

def pairs_same_house (seats : List (Option ℕ)) (pair : ℕ × ℕ) : Prop :=
  ∃ i, i < List.length seats ∧ seats.nth i = some (pair.1) ∧ seats.nth (i + 1) = some (pair.2)

def valid_arrangement (pairs : List (ℕ × ℕ)) (seats : List (Option ℕ)) : Prop :=
  List.length seats = 4 ∧
  ∀ pair ∈ pairs, pairs_same_house seats pair

theorem seat_arrangements : 
  let pairs := [(0, 1), (2, 3)] in
  ∃ arrangements : List (List (Option ℕ)),
  valid_arrangement pairs arrangements.head ∧
  List.length arrangements = 8 :=
by
  sorry

end seat_arrangements_l528_528540


namespace new_mixture_concentration_l528_528963

theorem new_mixture_concentration :
  let alcohol_2l := 0.30 * 2
  let alcohol_6l := 0.45 * 6
  let total_alcohol := alcohol_2l + alcohol_6l
  let total_volume := 10
  let concentration := total_alcohol / total_volume
  concentration = 0.33 := by
  let alcohol_2l := 0.30 * 2
  let alcohol_6l := 0.45 * 6
  let total_alcohol := alcohol_2l + alcohol_6l
  let total_volume := 10
  let concentration := total_alcohol / total_volume
  sorry

end new_mixture_concentration_l528_528963


namespace industrial_park_investment_l528_528649

noncomputable def investment_in_projects : Prop :=
  ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500

theorem industrial_park_investment :
  investment_in_projects :=
by
  have h : ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500 := 
    sorry
  exact h

end industrial_park_investment_l528_528649


namespace roots_quadratic_rational_expressions_l528_528430

theorem roots_quadratic_rational_expressions 
  (p q x1 x2 : ℚ) 
  (h1 : x1 + x2 = -p) 
  (h2 : x1 * x2 = q) :
  (x1^4 + x1^3 * x2 + x1^2 * x2^2 + x1 * x2^3 + x2^4) ∈ ℚ ∧ 
  (x1^4 * x2 + x1^3 * x2^2 + x1^2 * x2^3 + x1 * x2^4) ∈ ℚ := 
by
  have h3 : (x1^4 + x1^3 * x2 + x1^2 * x2^2 + x1 * x2^3 + x2^4) = (p^4 - 3 * q * p^2 + 7 * q^2), 
  sorry
  have h4 : (x1^4 * x2 + x1^3 * x2^2 + x1^2 * x2^3 + x1 * x2^4) = (p * q * (p^2 - q)), 
  sorry
  exact ⟨h3, h4⟩

end roots_quadratic_rational_expressions_l528_528430


namespace distance_between_truck_and_car_l528_528630

noncomputable def speed_truck : ℝ := 65
noncomputable def speed_car : ℝ := 85
noncomputable def time : ℝ := 3 / 60

theorem distance_between_truck_and_car : 
  let Distance_truck := speed_truck * time,
      Distance_car := speed_car * time in
  Distance_car - Distance_truck = 1 :=
by {
  sorry
}

end distance_between_truck_and_car_l528_528630


namespace race_completion_l528_528930

theorem race_completion (total_men : ℕ) (tripped_ratio : ℚ) (finished_tripped_ratio : ℚ)
                        (dehydrated_ratio : ℚ) (could_not_finish_dehydrated_ratio : ℚ)
                        (lost_ratio : ℚ) (found_way_back_ratio : ℚ)
                        (obstacle_ratio : ℚ) (finished_obstacle_ratio : ℚ) :
  total_men = 80 →
  tripped_ratio = 1/4 →
  finished_tripped_ratio = 1/3 →
  dehydrated_ratio = 2/3 →
  could_not_finish_dehydrated_ratio = 1/5 →
  lost_ratio = 12/100 →
  found_way_back_ratio = 1/2 →
  obstacle_ratio = 3/8 →
  finished_obstacle_ratio = 2/5 →
  let tripped_count := (tripped_ratio * total_men) in
  let finished_tripped_count := (finished_tripped_ratio * tripped_count).to_nat in
  let remaining_men := total_men - tripped_count in
  let dehydrated_count := (dehydrated_ratio * remaining_men).to_nat in
  let could_not_finish_dehydrated_count := (could_not_finish_dehydrated_ratio * dehydrated_count).to_nat in
  let finished_dehydrated_count := dehydrated_count - could_not_finish_dehydrated_count in
  let remaining_after_dehydration := remaining_men - dehydrated_count in
  let lost_count := (lost_ratio * remaining_after_dehydration).to_nat in
  let finished_lost_count := (found_way_back_ratio * lost_count).to_nat in
  let remaining_after_lost := remaining_after_dehydration - lost_count in
  let obstacle_count := (obstacle_ratio * remaining_after_lost).to_nat in
  let finished_obstacle_count := (finished_obstacle_ratio * obstacle_count).to_nat in

  finished_tripped_count + finished_dehydrated_count + finished_lost_count + finished_obstacle_count = 41 :=
by {
  intros,
  sorry
}

end race_completion_l528_528930


namespace positive_difference_for_6_points_l528_528196

def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def positiveDifferenceTrianglesAndQuadrilaterals (n : ℕ) : ℕ :=
  combinations n 3 - combinations n 4

theorem positive_difference_for_6_points : positiveDifferenceTrianglesAndQuadrilaterals 6 = 5 :=
by
  sorry

end positive_difference_for_6_points_l528_528196


namespace lowest_possible_sale_price_l528_528177

/-- 
Given: 
  * The list price of a jersey (P) is 80.
  * The maximum initial discount (D1) is 50%.
  * An additional summer sale discount (D2) is 20% off the original list price.
Prove: 
  * The lowest possible sale price is 30% of the list price.
--/
theorem lowest_possible_sale_price (P : ℝ) (D1 D2 : ℝ) (hP : P = 80) 
  (hD1 : D1 = 0.5) (hD2 : D2 = 0.2) : 
  let initial_discounted_price := P * (1 - D1) in
  let additional_discount := P * D2 in
  let final_sale_price := initial_discounted_price - additional_discount in
  final_sale_price = P * 0.3 :=
by
  sorry

end lowest_possible_sale_price_l528_528177


namespace fractional_eq_solution_l528_528476

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l528_528476


namespace number_of_ways_to_divide_l528_528024

-- Define the given shape
structure Shape :=
  (sides : Nat) -- Number of 3x1 stripes along the sides
  (centre : Nat) -- Size of the central square (3x3)

-- Define the specific problem shape
def problem_shape : Shape :=
  { sides := 4, centre := 9 } -- 3x1 stripes on all sides and a 3x3 centre

-- Theorem stating the number of ways to divide the shape into 1x3 rectangles
theorem number_of_ways_to_divide (s : Shape) (h1 : s.sides = 4) (h2 : s.centre = 9) : 
  ∃ ways, ways = 2 :=
by
  -- The proof is skipped
  sorry

end number_of_ways_to_divide_l528_528024


namespace area_of_triangular_region_l528_528623

-- Define the three lines
def line1 (x : ℝ) : ℝ := (1 / 2) * x + 3
def line2 (x : ℝ) : ℝ := -2 * x + 6
def line3 : ℝ := 1

-- Define the problem statement
theorem area_of_triangular_region : 
  let A := (-4 : ℝ, 1 : ℝ)
  let B := (5 / 2 : ℝ, 1 : ℝ)
  let C := (6 / 5 : ℝ, 18 / 5 : ℝ)
  1 / 2 * abs ((fst B - fst A) * (snd C - snd A) - (fst C - fst A) * (snd B - snd A)) = 8.45 :=
by
  sorry

end area_of_triangular_region_l528_528623


namespace particle_at_1_after_3_seconds_l528_528211

-- Definitions of the conditions
def Particle := (Int × Int)
def origin : Particle := (0, 0)
def move_left (p : Particle) : Particle := (p.1 - 1, p.0)
def move_right (p : Particle) : Particle := (p.1 + 1, p.0)
def move (p : Particle) (m : Bool) : Particle :=
  if m then move_right p else move_left p

def probability_at_point (final_pos : Particle) (steps : Nat) : ℚ :=
  let num_ways := Nat.choose steps (steps / 2 + final_pos.1)
  let total_possibilities := 2 ^ steps
  (num_ways : ℚ) / (total_possibilities : ℚ)

-- Proposition to prove
theorem particle_at_1_after_3_seconds :
  probability_at_point (1, 0) 3 = 3 / 8 :=
by
  sorry

end particle_at_1_after_3_seconds_l528_528211


namespace total_charge_correct_l528_528408

-- Definitions for the problem conditions
def initial_fee : ℝ := 2.25
def nighttime_rate : ℝ := 0.6 / (2/5)
def waiting_time_charge_per_minute : ℝ := 0.5

-- Definitions for the specific trip conditions
def trip_distance : ℝ := 3.6
def trip_start_time : ℂ := 7.5 -- Representing 7:30 PM as 7.5 (Time is not used in calculation directly)
def stop_durations : list ℝ := [3, 4]

-- Function to calculate the total trip charge
def total_trip_charge (distance : ℝ) (stops : list ℝ) : ℝ :=
  let distance_charge := (distance / (2/5)) * nighttime_rate
  let waiting_time_charge := (stops.sum) * waiting_time_charge_per_minute
  initial_fee + distance_charge + waiting_time_charge

-- The final statement to be proven
theorem total_charge_correct : total_trip_charge trip_distance stop_durations = 11.15 :=
by
  -- Proof is to be provided
  sorry

end total_charge_correct_l528_528408


namespace sum_of_n_digit_numbers_l528_528180

theorem sum_of_n_digit_numbers (n: ℕ) (h: n > 2) : 
  ∑ k in finset.Icc (10^(n-1)) (10^n - 1), k = 494 * ((10^(n-1)) - 1) * 10^(n-2) + 5 * 10^(2n - 3) := 
sorry

end sum_of_n_digit_numbers_l528_528180


namespace weights_system_l528_528584

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l528_528584


namespace jack_to_jill_routes_l528_528822

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def total_paths : ℕ := binomial 7 4

def paths_to_construction : ℕ := binomial 3 2

def paths_from_construction_to_jill : ℕ := binomial 4 2

def paths_through_construction : ℕ := paths_to_construction * paths_from_construction_to_jill

def valid_paths : ℕ := total_paths - paths_through_construction

theorem jack_to_jill_routes :
  valid_paths = 17 :=
by
  unfold total_paths paths_to_construction paths_from_construction_to_jill paths_through_construction valid_paths
  sorry

end jack_to_jill_routes_l528_528822


namespace jill_spent_on_clothing_l528_528445

variables (C T : ℝ)

-- Condition definitions
def spent_on_food := 0.10 * T
def spent_on_other_items := 0.40 * T
def tax_on_clothing := 0.04 * C * T
def tax_on_food := 0 * spent_on_food
def tax_on_other_items := 0.08 * spent_on_other_items
def total_tax_paid := 0.052 * T

-- Problem Statement
theorem jill_spent_on_clothing :
  0.04 * C * T + 0.08 * (0.40 * T) = 0.052 * T → C = 0.5 :=
by
  sorry

end jill_spent_on_clothing_l528_528445


namespace largest_angle_in_triangle_l528_528798

theorem largest_angle_in_triangle : 
  ∀ (a b c : ℝ), 
    (0 < a ∧ 0 < b ∧ 0 < c) →
    (10 * a = 24 * b) →
    (24 * b = 15 * c) →
    let θ := Real.arccos (7 / 8) in
    (θ = Real.pi / 2 ∨ θ > Real.pi / 2) ∧
    (∀ x, x ≠ θ → x < θ) :=
by
  intro a b c h_pos h_10a_24b h_24b_15c θ h_def_θ
  sorry

end largest_angle_in_triangle_l528_528798


namespace correct_statement_C_l528_528175

theorem correct_statement_C (A B D : Prop) (C : Prop) :
  (A ↔ ¬(∃ p1 p2 p3 : ℝ × ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ collinear p1 p2 p3)) →
  (B ↔ ¬(∃ (a : ∀ x : ℝ, ℝ × ℝ × ℝ) (α : ℝ × ℝ × ℝ → ℝ), line_outside_plane a α → intersects a α)) →
  (C ↔ (∃ (pyramid : Type) (P : pyramid → Prop), regular_pyramid pyramid ∧ plane_parallel_to_base P pyramid → regular_frustum (cut_pyramid P pyramid))) →
  (D ↔ ¬(∃ (prism : Type) (lateral_surface : prism → Prop), oblique_prism prism ∧ rectangle lateral_surface)) →
  C :=
by
  sorry

end correct_statement_C_l528_528175


namespace a_8_is_neg4_l528_528725

def seq (n : ℕ) : ℤ := 
  ∑ i in finset.range n, (-1 : ℤ) ^ i * (i + 1)

theorem a_8_is_neg4 : seq 8 = -4 := 
  sorry

end a_8_is_neg4_l528_528725


namespace form_square_with_tiles_l528_528221

-- Define the conditions of the problem
def right_triangle_area (base : ℝ) (height : ℝ) : ℝ := 
  (1 / 2) * base * height

def total_area (individual_area : ℝ) (number_of_tiles : ℕ) : ℝ := 
  number_of_tiles * individual_area

def square_side_length (area : ℝ) : ℝ := 
  Real.sqrt area

-- Prove that it is possible to form a square with the given conditions
theorem form_square_with_tiles : 
  ∀ (base height : ℝ) (n : ℕ), base = 1 → height = 2 → n = 20 → 
  let triangle_area := right_triangle_area base height in
  let total_triangle_area := total_area triangle_area n in
  square_side_length total_triangle_area = 2 * Real.sqrt 5 :=
by
  sorry

end form_square_with_tiles_l528_528221


namespace first_number_is_935_l528_528527

noncomputable def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

theorem first_number_is_935 : 
  let a := 935
  let b := 1383
  let r := 7
  let g := 32
  gcd (a - r) (b - r) = g ∧ (a - r) % g = 0 ∧ (b - r) % g = 0 ∧ a % g = r ∧ b % g = r → 
  a = 935 := sorry

end first_number_is_935_l528_528527


namespace distance_on_broken_sign_l528_528921

theorem distance_on_broken_sign :
  ∀ (Atown Cetown Betown SignpostA SignpostB : Type) 
    (dist : Atown → Cetown → ℝ) 
    (distATo : SignpostA → Atown → ℝ)
    (distAToC : SignpostA → Cetown → ℝ) 
    (distBTo : SignpostB → Atown → ℝ)
    (distBToC : SignpostB → Cetown → ℝ),
    (distATo SignpostA Atown = 7 ∧ distAToC SignpostA Cetown = 2) →
    (distBTo SignpostB Atown = 9 ∧ distBToC SignpostB Cetown = 4) →
    dist SignpostA SignpostB = 2 →
    dist SignpostA Betown = 5 - 4 →
    dist SignpostA Betown = 1 := by
  intros Atown Cetown Betown SignpostA SignpostB dist distATo distAToC distBTo distBToC
  intros h1 h2 hDist hTotal
  sorry

end distance_on_broken_sign_l528_528921


namespace jane_average_speed_l528_528125

/-
Question: Prove that the average speed over the period from 6 a.m. to 2 p.m. is 34.29 miles per hour given:
- The total distance traveled is 240 miles.
- The total travel time excluding the stop is 7 hours.
-/

def total_distance : ℝ := 240
def total_time : ℝ := 7
def average_speed := total_distance / total_time

theorem jane_average_speed : average_speed = 34.29 :=
by
  sorry

end jane_average_speed_l528_528125


namespace fixed_point_of_quadratic_l528_528735

theorem fixed_point_of_quadratic (m : ℝ) :
  ∃ A : ℝ × ℝ, A = (-1, 0) ∧ ∀ (m : ℝ), let y := -((A.1) ^ 2) + (m - 1) * (A.1) + m in y = A.2 :=
sorry

end fixed_point_of_quadratic_l528_528735


namespace find_integer_m_l528_528860

theorem find_integer_m 
  (m : ℤ) (h_pos : m > 0) 
  (h_intersect : ∃ (x y : ℤ), 17 * x + 7 * y = 1000 ∧ y = m * x + 2) : 
  m = 68 :=
by
  sorry

end find_integer_m_l528_528860


namespace new_ratio_of_partners_to_associates_l528_528994

theorem new_ratio_of_partners_to_associates : 
  ∀ (P A A' : ℕ) (h1 : 2 * A = 63 * P) (h2 : P = 20) (h3 : A' = A + 50), 
  let new_ratio := P / A' in new_ratio = 1 / 34 := 
by
  assume P A A' h1 h2 h3,
  let new_ratio := P / A',
  sorry

end new_ratio_of_partners_to_associates_l528_528994


namespace ellipse_intersection_points_l528_528373

theorem ellipse_intersection_points 
  (L1 L2 : affine) -- where affine is some affine space to represent our lines
  (hL1 : ∃ p1 p2 : Point, p1 ≠ p2 ∧ L1 ∩ ellipse = {p1, p2}) -- L1 intersects the ellipse at two points
  (hL2 : ∃ p3 p4 : Point, p3 ≠ p4 ∧ L2 ∩ ellipse = {p3, p4}) -- L2 intersects the ellipse at two points
  (heq : L1 ≠ L2) (htang : ∀ p, ¬(is_tangent L1 ellipse p ∨ is_tangent L2 ellipse p)) :
  ∃ (n : ℕ), n ∈ {2, 3, 4} ∧ count_points_of_intersection L1 L2 ellipse = n :=
sorry

end ellipse_intersection_points_l528_528373


namespace combined_work_time_l528_528184

def ajay_completion_time : ℕ := 8
def vijay_completion_time : ℕ := 24

theorem combined_work_time (T_A T_V : ℕ) (h1 : T_A = ajay_completion_time) (h2 : T_V = vijay_completion_time) :
  1 / (1 / (T_A : ℝ) + 1 / (T_V : ℝ)) = 6 :=
by
  rw [h1, h2]
  sorry

end combined_work_time_l528_528184


namespace bus_speed_l528_528918

noncomputable def radius : ℝ := 35 / 100  -- Radius in meters
noncomputable def rpm : ℝ := 500.4549590536851

noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def distance_in_one_minute : ℝ := circumference * rpm
noncomputable def distance_in_km_per_hour : ℝ := (distance_in_one_minute / 1000) * 60

theorem bus_speed :
  distance_in_km_per_hour = 66.037 :=
by
  -- The proof is skipped here as it is not required
  sorry

end bus_speed_l528_528918


namespace total_food_per_day_l528_528135

theorem total_food_per_day 
  (first_soldiers : ℕ)
  (second_soldiers : ℕ)
  (food_first_side_per_soldier : ℕ)
  (food_second_side_per_soldier : ℕ) :
  first_soldiers = 4000 →
  second_soldiers = first_soldiers - 500 →
  food_first_side_per_soldier = 10 →
  food_second_side_per_soldier = food_first_side_per_soldier - 2 →
  (first_soldiers * food_first_side_per_soldier + second_soldiers * food_second_side_per_soldier = 68000) :=
by
  intros h1 h2 h3 h4
  sorry

end total_food_per_day_l528_528135


namespace sequence_sum_nonnegative_l528_528218

def sequence : ℕ → ℤ
| 1       := 1
| (2*k)   := -sequence k
| (2*k-1) := (-1)^(k+1) * sequence k

theorem sequence_sum_nonnegative (n : ℕ) : (∑ i in finset.range n, sequence (i + 1)) ≥ 0 :=
sorry

end sequence_sum_nonnegative_l528_528218


namespace total_distance_l528_528796

-- Define the lengths of the segments and their midpoints
def length_CD : ℝ := 10
def length_C'D' : ℝ := 16
def midpoint_E : ℝ := length_CD / 2 -- Midpoint of CD
def midpoint_E' : ℝ := length_C'D' / 2 -- Midpoint of C'D'

-- Define the distance of Q from E towards D on CD
def distance_x : ℝ := 3

-- Given association property: distance from C to Q is half the distance from C' to Q'
def distance_CQ : ℝ := midpoint_E - distance_x
def distance_C'Q' : ℝ := 2 * distance_CQ
def distance_y : ℝ := midpoint_E' - distance_C'Q'

-- The final theorem to prove is that x + y = 7
theorem total_distance : distance_x + distance_y = 7 := by
  sorry

end total_distance_l528_528796


namespace math_inequality_l528_528056

noncomputable def inequality_holds (a : Fin n → ℝ) (r s : ℝ) : Prop :=
  (∑ i, a i ^ s) ^ (1 / s) ≤ (∑ i, a i ^ r) ^ (1 / r)

/-- Let a₁, a₂, ..., aₙ be non-negative real numbers.
If 0 < r < s, then (Σᵢ aᵢˢ)¹ᐟˢ ≤ (Σᵢ aᵢʳ)¹ᐟʳ, with equality holding if and only if at least n-1 of a₁, a₂, ..., aₙ are 0. -/
theorem math_inequality (a : Fin n → ℝ) (r s : ℝ) (h_a_nonneg : ∀ i, 0 ≤ a i)
  (h_r_pos : 0 < r) (h_s_gt_r : r < s) :
  inequality_holds a r s :=
sorry

#check @math_inequality

end math_inequality_l528_528056


namespace trigonometric_range_l528_528380

noncomputable def sin : ℝ → ℝ := Real.sin
noncomputable def cos : ℝ → ℝ := Real.cos

theorem trigonometric_range
  (a b c : ℝ)
  (A B C : ℝ)
  (h_cond : c * sin A = a * cos C)
  (h_angle_rel : A + B + C = π)
  (h_angle_bounds : 0 < A ∧ A < 3 * π / 4) :
  ∃ A_range : Set ℝ, A_range = Set.Ioc 1 2 ∧ (sqrt 3 * sin A - cos (B + π / 4)) ∈ A_range :=
by
  sorry

end trigonometric_range_l528_528380


namespace total_squares_l528_528771

theorem total_squares (num_groups : ℕ) (squares_per_group : ℕ) (total : ℕ) 
  (h1 : num_groups = 5) (h2 : squares_per_group = 5) (h3 : total = num_groups * squares_per_group) : 
  total = 25 :=
by
  rw [h1, h2] at h3
  exact h3

end total_squares_l528_528771


namespace total_cost_correct_l528_528072

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end total_cost_correct_l528_528072


namespace last_digit_to_appear_is_zero_l528_528686

def S : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := S (n+1) + S n

def S_mod_9 (n : ℕ) : ℕ := (S n) % 9

theorem last_digit_to_appear_is_zero :
  ∃ N, ∀ n ≤ N, ∃ k, S_mod_9 k = n :=
sorry

end last_digit_to_appear_is_zero_l528_528686


namespace length_of_min_graph_l528_528113

open Real

noncomputable def f (x : ℝ) := x^2 - 4 * x + 4
noncomputable def g (x : ℝ) := -x^2 + 6 * x - 8
noncomputable def h (x : ℝ) := 4
noncomputable def j (x : ℝ) := max (max (f x) (g x)) (h x)
noncomputable def k (x : ℝ) := min (min (f x) (g x)) (h x)

theorem length_of_min_graph :
  let  ℓ := (∫ x in 2..3, sqrt (1 + (2 * (x - 3))^2)) + (∫ x in 3..4, sqrt (1 + (2 * (x - 2))^2))
  in ℓ^2 = 8 :=
by
  sorry

end length_of_min_graph_l528_528113


namespace parabola_characteristics_unique_l528_528710

/-- Define the standard form of the parabola -/
def parabola_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The vertex form of the parabola given its vertex -/
def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h) ^ 2 + k

/-- The condition that the parabola has a vertex at (3, -2) -/
def vertex_condition (x : ℝ) : ℝ := vertex_form 4 3 (-2) x

/-- The condition that the parabola passes through the point (4, 2) -/
def point_condition : Prop := vertex_condition 4 = 2

/-- Proving the standard form equation for the given parabola characteristics -/
theorem parabola_characteristics_unique :
  ∃ a b c, point_condition ∧ ∀ x, vertex_condition x = parabola_equation a b c x :=
by
  use 4, -24, 34
  apply And.intro
  · exact (by linarith)
  · intro x
    have h1 : vertex_condition x = vertex_form 4 3 (-2) x := rfl
    unfold vertex_form at h1
    unfold parabola_equation
    rw [←h1]
    sorry

end parabola_characteristics_unique_l528_528710


namespace golden_state_total_points_l528_528035

theorem golden_state_total_points :
  ∀ (Draymond Curry Kelly Durant Klay : ℕ),
  Draymond = 12 →
  Curry = 2 * Draymond →
  Kelly = 9 →
  Durant = 2 * Kelly →
  Klay = Draymond / 2 →
  Draymond + Curry + Kelly + Durant + Klay = 69 :=
by
  intros Draymond Curry Kelly Durant Klay
  intros hD hC hK hD2 hK2
  rw [hD, hC, hK, hD2, hK2]
  sorry

end golden_state_total_points_l528_528035


namespace quadratic_function_fixed_point_l528_528736

theorem quadratic_function_fixed_point (m : ℝ) :
  ∃ A : ℝ × ℝ, A = (-1, 0) ∧ (∀ m : ℝ, ∃ x : ℝ, ∃ y : ℝ, y = -x^2 + (m-1)*x + m ∧ A = (x, y)) :=
by
  -- We will assert that A = (-1, 0)
  let A : ℝ × ℝ := (-1, 0) in
  use A,
  split,
  -- Proving that A is equal to (-1, 0)
  rfl,
  -- Now we show that for all m, point (-1, 0) lies on the graph of the function y = -x^2 + (m-1)x + m
  intro m,
  use (-1),
  use (0),
  split,
  -- Simplifying the function at x = -1
  calc
    0 = -(-1)^2 + (m - 1) * (-1) + m : by sorry
      ... = -1 + (m - 1) * (-1) + m : by sorry
      ... = -1 + - (m - 1) + m : by sorry
      ... = -1 - m + 1 + m : by sorry
      ... = 0 : by sorry,
  -- Verifying that the point (-1, 0) is (x, y)
  rfl

end quadratic_function_fixed_point_l528_528736


namespace triangle_constructible_range_t_l528_528721

def triangle_constructible (f : ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), f a + f b > f c ∧ f a + f c > f b ∧ f b + f c > f a

theorem triangle_constructible_range_t 
  (f : ℝ → ℝ) 
  (h_f : ∀ x : ℝ, f x = (2^x - t) / (2^x + 1)) 
  (h : triangle_constructible f) : 
  t ∈ set.Icc (-2 : ℝ) (-1 / 2) :=
sorry

end triangle_constructible_range_t_l528_528721


namespace proof_y0_value_l528_528802

theorem proof_y0_value 
  (α : ℝ)
  (x0 y0 : ℝ)
  (h1 : α ∈ (3 * π / 2, 2 * π))  -- α is in the fourth quadrant
  (h2 : cos(α - π / 3) = -sqrt 3 / 3)  -- given condition on cosine
  (h3 : (x0, y0) = (cos α, sin α))  -- P(x0, y0) is the intersection of the terminal side and the unit circle
  : y0 = (-sqrt 6 - 3) / 6 :=
sorry

end proof_y0_value_l528_528802


namespace complete_square_cpjq_l528_528099

theorem complete_square_cpjq (j : ℝ) (c p q : ℝ) (h : 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) :
  c = 8 ∧ p = -3/8 ∧ q = 119/8 → q / p = -119/3 :=
by
  intros
  cases a with hc hpq
  cases hpq with hp hq
  rw [hc, hp, hq]
  have hp_ne_zero : (-3 / 8) ≠ 0 := by norm_num
  field_simp [hp_ne_zero]
  norm_num
  sorry

end complete_square_cpjq_l528_528099


namespace greatest_z_for_7_divisor_of_50_factorial_l528_528569

theorem greatest_z_for_7_divisor_of_50_factorial :
  ∃ z : ℕ, (7^z ∣ factorial 50) ∧ ∀ w : ℕ, (7^w ∣ factorial 50) → w ≤ 8 :=
sorry

end greatest_z_for_7_divisor_of_50_factorial_l528_528569


namespace system_of_equations_correct_l528_528587

def weight_system (x y : ℝ) : Prop :=
  (5 * x + 6 * y = 1) ∧ (3 * x = y)

theorem system_of_equations_correct (x y : ℝ) :
  weight_system x y ↔ 
    (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 5 * x + 6 * y) :=
by sorry

end system_of_equations_correct_l528_528587


namespace find_s_t_u_sum_l528_528845

variables {a b c : ℝ^3} (s t u : ℝ)

-- Mutually orthogonal unit vectors
axiom orthogonal_unit_vectors : (a ⬝ b = 0) ∧ (b ⬝ c = 0) ∧ (c ⬝ a = 0) ∧ 
                                 (norm a = 1) ∧ (norm b = 1) ∧ (norm c = 1)

-- Given condition
axiom given_condition : b ⬝ (c × a) = 1

-- Given equation
axiom equation : b = s * (b × c) + t * (c × a) + u * (a × b)

theorem find_s_t_u_sum : s + t + u = 1 :=
by {
  -- Sorry, the proof is omitted.
  sorry
}

end find_s_t_u_sum_l528_528845


namespace red_tile_probability_l528_528981

def is_red_tile (n : ℕ) : Prop := n % 7 = 3

noncomputable def red_tiles_count : ℕ :=
  Nat.card {n : ℕ | n ≤ 70 ∧ is_red_tile n}

noncomputable def total_tiles_count : ℕ := 70

theorem red_tile_probability :
  (red_tiles_count : ℤ) / (total_tiles_count : ℤ) = (1 : ℤ) / 7 :=
sorry

end red_tile_probability_l528_528981


namespace analyze_quadratic_function_l528_528652

variable (x : ℝ)

def quadratic_function : ℝ → ℝ := λ x => x^2 - 4 * x + 6

theorem analyze_quadratic_function :
  (∃ y : ℝ, quadratic_function y = (x - 2)^2 + 2) ∧
  (∃ x0 : ℝ, quadratic_function x0 = (x0 - 2)^2 + 2 ∧ x0 = 2 ∧ (∀ y : ℝ, quadratic_function y ≥ 2)) :=
by
  sorry

end analyze_quadratic_function_l528_528652


namespace find_c_and_cosB_l528_528420

noncomputable def triangle_proof (a b : ℝ) (A B C : ℝ) (c : ℝ) : Prop :=
  a = 2 * real.sqrt 6 ∧
  b = 3 ∧
  real.sin (B + C)^2 + real.sqrt 2 * real.sin (2 * A) = 0 →
  c = 3 ∧
  real.cos B = real.sqrt 6 / 3

theorem find_c_and_cosB :
  ∃ (a b : ℝ) (A B C : ℝ) (c : ℝ), triangle_proof a b A B C c :=
begin
  use 2 * real.sqrt 6,
  use 3,
  existsi _, -- for A
  existsi _, -- for B
  existsi _, -- for C
  use 3,
  sorry
end

end find_c_and_cosB_l528_528420


namespace find_m_l528_528765

theorem find_m : 
  ∀ (m : ℝ), m > 0 ∧ (∃ P Q : ℝ × ℝ, (P.1 + P.2 = m) ∧ (Q.1 + Q.2 = m) ∧ (P.1^2 + P.2^2 = 1) ∧ (Q.1^2 + Q.2^2 = 1) ∧ 
  ∃ (O : ℝ × ℝ), O = (0, 0) ∧ ∃ angle : ℝ, angle = 120 ∧ 
  ∃ θ : ℝ, θ = angle_in_radians 120 ∧ 
  ∃ d : ℝ, d = dist_to_origin (P, Q, O, θ)) →
  m = (real.sqrt 2) / 2 :=
begin
  sorry
end

end find_m_l528_528765


namespace red_square_ones_even_l528_528610

def is_red (i j : ℕ) : Prop := (i + j) % 2 = 1

def is_white (i j : ℕ) : Prop := (i + j) % 2 = 0

def has_odd_ones_each_row (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i, (Finset.range 14).sum (λ j, grid i j) % 2 = 1

def has_odd_ones_each_column (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ j, (Finset.range 10).sum (λ i, grid i j) % 2 = 1

def total_ones_is_even (grid : ℕ → ℕ → ℕ) : Prop :=
  (Finset.range 10).sum (λ i, (Finset.range 14).sum (λ j, grid i j)) % 2 = 0

theorem red_square_ones_even (grid : ℕ → ℕ → ℕ) 
  (h_row : has_odd_ones_each_row grid)
  (h_column : has_odd_ones_each_column grid)
  (h_total_even : total_ones_is_even grid) :
  (Finset.range 10).sum (λ i, (Finset.range 14).sum (λ j, if is_red i j then grid i j else 0)) % 2 = 0 := 
sorry

end red_square_ones_even_l528_528610


namespace domain_of_g_l528_528552

theorem domain_of_g (x y : ℝ) : 
  (∃ g : ℝ, g = 1 / (x^2 + (x - y)^2 + y^2)) ↔ (x, y) ≠ (0, 0) :=
by sorry

end domain_of_g_l528_528552


namespace incorrect_axis_symmetry_l528_528297

noncomputable def quadratic_function (x : ℝ) : ℝ := - (x + 2)^2 - 3

theorem incorrect_axis_symmetry :
  (∀ x : ℝ, quadratic_function x < 0) ∧
  (∀ x : ℝ, x > -1 → (quadratic_function x < quadratic_function (-2))) ∧
  (¬∃ x : ℝ, quadratic_function x = 0) ∧
  (¬ ∀ x : ℝ, x = 2) →
  false :=
by
  sorry

end incorrect_axis_symmetry_l528_528297


namespace problem_statement_l528_528093

variable (q p : ℚ)
#check λ (q p : ℚ), q / p

theorem problem_statement :
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  q / p = -(119 : ℚ) / 3 :=
by
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  sorry

end problem_statement_l528_528093


namespace rain_probability_tel_aviv_l528_528499

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l528_528499


namespace braxton_total_earnings_l528_528890

-- Definitions of the given problem conditions
def students_ashwood : ℕ := 9
def days_ashwood : ℕ := 4
def students_braxton : ℕ := 6
def days_braxton : ℕ := 7
def students_cedar : ℕ := 8
def days_cedar : ℕ := 6

def total_payment : ℕ := 1080
def daily_wage_per_student : ℚ := total_payment / ((students_ashwood * days_ashwood) + 
                                                   (students_braxton * days_braxton) + 
                                                   (students_cedar * days_cedar))

-- The statement to be proven
theorem braxton_total_earnings :
  (students_braxton * days_braxton * daily_wage_per_student) = 360 := 
by
  sorry -- proof goes here

end braxton_total_earnings_l528_528890


namespace smallest_positive_period_of_f_perimeter_of_triangle_l528_528346

-- Definition for problem (Ⅰ)
def f (x : ℝ) : ℝ := 2 * cos x * cos (x - π / 3) - 1 / 2

-- Problem (Ⅰ)
theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ T = π :=
sorry

-- Definitions for problem (Ⅱ)
variable (A B C a b c : ℝ)
variable (area perimeter : ℝ)
-- Given conditions
def triangle_area : Prop := area = 2 * sqrt 3
def side_opposite_C : Prop := c = 2 * sqrt 3
def f_of_C : Prop := f C = 1 / 2

-- Problem (Ⅱ)
theorem perimeter_of_triangle :
  triangle_area ∧ side_opposite_C ∧ f_of_C → perimeter = 6 + 2 * sqrt 3 :=
sorry

end smallest_positive_period_of_f_perimeter_of_triangle_l528_528346


namespace constant_term_in_binomial_expansion_l528_528426

noncomputable def integral_result : ℕ :=
  ∫ x in 0..(Real.pi / 2), (6 * Real.sin x)

theorem constant_term_in_binomial_expansion :
  integral_result = 6 → 
  ∑ k in Finset.range (6 + 1), (Nat.choose 6 k * (-2)^k * (x^(6-3*k))) = 60 := 
sorry

end constant_term_in_binomial_expansion_l528_528426


namespace correct_statements_l528_528186

-- Defining people and their ages
variable {Andrew Boris Svetlana Larisa : Prop}
variable [HasLess Andrew Svetlana] [HasLess Larisa Andrew] [HasLess Svetlana Boris]
variable oldest_is_larisa_husband : ∀ x, x = Boris → (∃ y, y = Larisa ∧ married_to y x)

variable married_to : Prop → Prop → Prop

theorem correct_statements :
  (∃ y, y = Larisa ∧ (∀ z, z = Boris → married_to y z)) ∧ (¬ married_to Boris Svetlana) :=
by
  -- Here we will provide the detailed proof steps as indicated in the solution section.
  sorry

end correct_statements_l528_528186


namespace eq_team_with_at_least_one_boy_one_girl_l528_528119

variable (B G : ℕ)
variable (total_people B_count G_count team_size : ℕ)
variable (x y : ℕ)

-- Defining the conditions
def ChessClub := total_people = 18 ∧ B_count = 9 ∧ G_count = 9 ∧ team_size = 4

-- Probablity of selecting a 4 person team with at least one boy and one girl
noncomputable def probability (total_teams all_boys_teams all_girls_teams : ℕ) : ℚ :=
  (total_teams - (all_boys_teams + all_girls_teams)) / total_teams

theorem eq_team_with_at_least_one_boy_one_girl (h : ChessClub total_people B_count G_count team_size) : 
  probability (nat.choose total_people team_size) (nat.choose B_count team_size) (nat.choose G_count team_size) = 234 / 255 :=
by
  sorry

end eq_team_with_at_least_one_boy_one_girl_l528_528119


namespace fraction_is_integer_for_nonzero_real_l528_528301

noncomputable def is_integer (n : ℝ) : Prop := ∃ k : ℤ, n = k

theorem fraction_is_integer_for_nonzero_real (x : ℝ) (h : x ≠ 0) :
  is_integer (|x + |x|| / x) :=
sorry

end fraction_is_integer_for_nonzero_real_l528_528301


namespace largest_sum_of_digits_l528_528001

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) (h4 : 1 ≤ y ∧ y ≤ 10) (h5 : (1000 * (a * 100 + b * 10 + c)) = 1000) : 
  a + b + c = 8 :=
sorry

end largest_sum_of_digits_l528_528001


namespace arrangement_ways_l528_528977

-- Define the number of people and rows
def num_people := 6
def num_rows := 2
def people_per_row := num_people / num_rows

-- Define the restrictions
def front_row := {1, 2, 3}
def back_row := {4, 5, 6}
def A cannot be in the front row := ¬ A ∈ front_row
def B cannot be in the back row := ¬ B ∈ back_row

-- The statement to be proved
theorem arrangement_ways : 
    ∃ (count: ℕ), count = 216 ∧ 
    ∀ (A B: ℕ), 
      (A ∈ {1, 2, 3} → false) ∧   -- A cannot be in the front row
      (B ∈ {4, 5, 6} → false)     -- B cannot be in the back row
      → 
        count = 
        (num_people * (num_people - 1) / 2) * 
        (num_people * (num_people - 1)) := 
sorry

end arrangement_ways_l528_528977


namespace ae_plus_ap_eq_pd_l528_528015

-- Defining the given conditions
variables {A B C D E F P : Type}
variable [RightAngleTriangle A B C]  -- Right-angled triangle at ACB
variable [IncircleMeetPoints O A B C D E F]  -- Incircle meeting points D, E, F on BC, AC, AB respectively
variable [AD_Intersects_Incircle_At P AD]  -- AD cuts incircle at P
variable [AngleBPC_90 B P C]  -- Given angle BPC is 90 degrees

-- Proving the required equality
theorem ae_plus_ap_eq_pd (cond : RightAngleTriangle A B C) 
    (incircle_cond : IncircleMeetPoints O A B C D E F) 
    (ad_intersects : AD_Intersects_Incircle_At P AD)
    (angle_cond : AngleBPC_90 B P C) : 
    AE + AP = PD := by 
    sorry

end ae_plus_ap_eq_pd_l528_528015


namespace time_for_A_alone_l528_528962

variable {W : ℝ}
variable {x : ℝ}

theorem time_for_A_alone (h1 : (W / x) + (W / 24) = W / 12) : x = 24 := 
sorry

end time_for_A_alone_l528_528962


namespace parallel_condition_necessary_but_not_sufficient_l528_528580

-- Given definitions of lines and planes, and their parallelism
variable (Line Plane : Type)
variable (m: Line) (alpha: Plane)
variable (parallel : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_to_plane : Line → Plane → Prop)

-- Conditions
axiom countless_parallel_lines (m: Line) (alpha: Plane) : Prop := 
  ∀ l, line_in_plane l alpha → parallel m l

-- Goal: Prove that 'countless_parallel_lines' is necessary but not sufficient for 'parallel_to_plane'.
theorem parallel_condition_necessary_but_not_sufficient (m : Line) (alpha : Plane) :
  (countless_parallel_lines m alpha → ¬(∃ l, line_in_plane m alpha ∧ line_in_plane l alpha ∧ parallel m l)) 
  → ∀ l, (line_in_plane l alpha → parallel m l) ∧ ¬(parallel_to_plane m alpha ↔ (countless_parallel_lines m alpha ∧ line_in_plane m alpha)) := 
  sorry

end parallel_condition_necessary_but_not_sufficient_l528_528580


namespace square_area_increase_l528_528574

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let A_original := s^2 in
  let A_new := (1.40 * s)^2 in
  let percentage_increase := (A_new - A_original) / A_original * 100 in
  percentage_increase = 96 :=
by
  sorry

end square_area_increase_l528_528574


namespace books_jerry_added_l528_528827

def initial_action_figures : ℕ := 7
def initial_books : ℕ := 2

theorem books_jerry_added (B : ℕ) (h : initial_action_figures = initial_books + B + 1) : B = 4 :=
by
  sorry

end books_jerry_added_l528_528827


namespace rewrite_expression_and_compute_l528_528098

noncomputable def c : ℚ := 8
noncomputable def p : ℚ := -3 / 8
noncomputable def q : ℚ := 119 / 8

theorem rewrite_expression_and_compute :
  (∃ (c p q : ℚ), 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) →
  q / p = -119 / 3 :=
by
  sorry

end rewrite_expression_and_compute_l528_528098


namespace majority_property_all_k_l528_528429

-- Define the majority function (assuming it is predefined)
def majority (strings : list (fin n → bool)) : fin n → bool :=
  λ i, if (strings.map (λ s, s i)).count (λ b, b) > (strings.length / 2) then true else false

-- Define the set S and property P_k
variable {n : ℕ} (S : set (fin n → bool))

-- Define the property P_k
def P_k (k : ℕ) : Prop :=
  ∀ (l : list (fin n → bool)), l.length = 2 * k + 1 → (∀ x ∈ l, x ∈ S) → majority l ∈ S

-- Proof statement
theorem majority_property_all_k (h : ∃ k > 0, P_k S k) : ∀ k > 0, P_k S k :=
by
  sorry


end majority_property_all_k_l528_528429


namespace minimum_real_roots_is_one_l528_528425

noncomputable def minimum_real_roots (g : Polynomial ℝ) (h_deg : g.degree = 300) 
(h_roots : Multiset.card (Multiset.map (Complex.abs) (Polynomial.roots g)) = 150) : ℕ :=
  sorry

theorem minimum_real_roots_is_one (g : Polynomial ℝ) (h_deg : g.degree = 300) 
(h_roots : Multiset.card (Multiset.map (Complex.abs) (Polynomial.roots g)) = 150) : 
minimum_real_roots g h_deg h_roots = 1 :=
sorry

end minimum_real_roots_is_one_l528_528425


namespace find_complex_number_find_lambda_l528_528989

section ProofProblem

variable {z : ℂ} (λ : ℝ)
def in_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0
def is_purely_imaginary (w : ℂ) : Prop := w.re = 0

theorem find_complex_number (hz_mag : ‖z‖ = Real.sqrt 2)
  (hz_imag : is_purely_imaginary (z^2))
  (hz_quad : in_first_quadrant z) :
  z = 1 + Complex.i :=
sorry

theorem find_lambda (λ : ℝ)
  (a b c : ℝ × ℝ)
  (h_a : a = (1, 1))
  (h_b : b = (1, -1))
  (h_c : c = (0, 2))
  (h_orthogonal : (λ • ⟨a.1, a.2⟩ + ⟨b.1, b.2⟩) ∙ (λ • ⟨b.1, b.2⟩ + ⟨c.1, c.2⟩) = 0) :
  λ = 1 / 2 :=
sorry

end ProofProblem

end find_complex_number_find_lambda_l528_528989


namespace lea_total_cost_l528_528075

theorem lea_total_cost :
  let book_cost := 16 in
  let binders_count := 3 in
  let binder_cost := 2 in
  let notebooks_count := 6 in
  let notebook_cost := 1 in
  book_cost + (binders_count * binder_cost) + (notebooks_count * notebook_cost) = 28 :=
by
  sorry

end lea_total_cost_l528_528075


namespace even_number_subsets_l528_528452

theorem even_number_subsets (n : ℕ) :
  ∃ k : ℕ, k = 2^(n-1) ∧ (∀ (s : finset (fin n)), s.card % 2 = 0 → (s.card = k)) :=
sorry

end even_number_subsets_l528_528452


namespace distance_after_3_minutes_l528_528637

-- Define the given speeds and time interval
def speed_truck : ℝ := 65 -- in km/h
def speed_car : ℝ := 85 -- in km/h
def time_minutes : ℝ := 3 -- in minutes

-- The equivalent time in hours
def time_hours : ℝ := time_minutes / 60

-- Calculate the distances travelled by the truck and the car
def distance_truck : ℝ := speed_truck * time_hours
def distance_car : ℝ := speed_car * time_hours

-- Define the distance between the truck and the car
def distance_between : ℝ := distance_car - distance_truck

-- Theorem: The distance between the truck and car after 3 minutes is 1 km.
theorem distance_after_3_minutes : distance_between = 1 := by
  sorry

end distance_after_3_minutes_l528_528637


namespace power_equation_l528_528370

theorem power_equation (x a b : ℝ) (ha : 3^x = a) (hb : 5^x = b) : 45^x = a^2 * b :=
sorry

end power_equation_l528_528370


namespace quadratic_imaginary_solution_l528_528673

theorem quadratic_imaginary_solution :
  ∀ (p q : ℝ), (∀ x : ℂ, 5 * x^2 + 7 = 4 * x - 12 → x = p + q * complex.I ∨ x = p - q * complex.I) →
  p + q^2 = 101 / 25 :=
begin
  intros p q h,
  sorry
end

end quadratic_imaginary_solution_l528_528673


namespace factorial_identity_l528_528247

theorem factorial_identity : 8! - 7 * 7! - 7! = 0 :=
by
  sorry

end factorial_identity_l528_528247


namespace distance_between_truck_and_car_l528_528626

def truck_speed : ℝ := 65 -- km/h
def car_speed : ℝ := 85 -- km/h
def time_minutes : ℝ := 3 -- minutes
def time_hours : ℝ := time_minutes / 60 -- converting minutes to hours

def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem distance_between_truck_and_car :
  let truck_distance := distance_traveled truck_speed time_hours
  let car_distance := distance_traveled car_speed time_hours
  truck_distance - car_distance = -1 := -- the distance is 1 km but negative when subtracting truck from car
by {
  sorry
}

end distance_between_truck_and_car_l528_528626


namespace parallel_perpendicular_trans_l528_528727

variable (m n : Line)
variable (α : Plane)

-- Hypotheses
variable (h1 : m ≠ n)
variable (h2 : m ∥ n)
variable (h3 : m ⟂ α)

-- Conclusion
theorem parallel_perpendicular_trans (h1 : m ≠ n) (h2 : m ∥ n) (h3 : m ⟂ α) : n ⟂ α :=
sorry

end parallel_perpendicular_trans_l528_528727


namespace find_a_for_pure_imaginary_l528_528753

-- Definitions based on the conditions
def z (a : ℝ) : ℂ := (1 + a * complex.I) / (1 - complex.I)
def pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Main statement to prove
theorem find_a_for_pure_imaginary (a : ℝ) : pure_imaginary (z a) → a = 1 := 
sorry

end find_a_for_pure_imaginary_l528_528753


namespace binomial_expansion_constant_term_and_max_coefficient_l528_528752

theorem binomial_expansion_constant_term_and_max_coefficient :
  (∀ n : ℕ, ∃ k : ℕ, binomial_term (2*x + 1/(sqrt x)) n k → (n - 3/2 * 4 = 0 → n = 6)) ∧
  (binomial_coefficient 6 3 * 2^3 = 160) := 
begin
  sorry
end

end binomial_expansion_constant_term_and_max_coefficient_l528_528752


namespace find_x_range_l528_528114

variable {f : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

def monotically_decreasing (f : ℝ → ℝ) : Prop :=
∀ a b, a < b → f (a) > f (b)

def interval_is_solution (x : ℝ) : Prop :=
x ∈ Ico 0 (real.pi / 6) ∨ x ∈ Ioc (5 * real.pi / 6) real.pi

theorem find_x_range (h_odd: odd_function f) (h_monotone: monotically_decreasing f)
  (h_condition: ∀ x, x ∈ Icc 0 real.pi → f (real.sin x - 1) > - f (real.sin x)) :
  ∀ x, x ∈ Icc 0 real.pi → interval_is_solution x :=
sorry

end find_x_range_l528_528114


namespace determinant_eval_cos_gamma_l528_528696

noncomputable def determinant_matrix (α β γ : ℝ) : ℝ :=
  Matrix.det !![
    [Real.cos α * Real.cos β, Real.cos α * Real.sin β * Real.cos γ, -Real.sin α * Real.sin γ],
    [-Real.sin β * Real.cos γ, Real.cos β, 0],
    [Real.sin α * Real.cos β, Real.sin α * Real.sin β * Real.cos γ, Real.cos α * Real.cos γ]
  ]

theorem determinant_eval_cos_gamma (α β γ : ℝ) :
  determinant_matrix α β γ = Real.cos γ :=
by
  sorry

end determinant_eval_cos_gamma_l528_528696


namespace calculation_2015_l528_528942

theorem calculation_2015 :
  2015 ^ 2 - 2016 * 2014 = 1 :=
by
  sorry

end calculation_2015_l528_528942


namespace distance_after_3_minutes_l528_528639

/-- Let truck_speed be 65 km/h and car_speed be 85 km/h.
    Let time_in_minutes be 3 and converted to hours it is 0.05 hours.
    The goal is to prove that the distance between the truck and the car
    after 3 minutes is 1 kilometer. -/
def truck_speed : ℝ := 65 -- speed in km/h
def car_speed : ℝ := 85 -- speed in km/h
def time_in_minutes : ℝ := 3 -- time in minutes
def time_in_hours : ℝ := time_in_minutes / 60 -- converted time in hours
def distance_truck := truck_speed * time_in_hours
def distance_car := car_speed * time_in_hours
def distance_between : ℝ := distance_car - distance_truck

theorem distance_after_3_minutes : distance_between = 1 := by
  -- Proof steps would go here
  sorry

end distance_after_3_minutes_l528_528639


namespace minimize_d_and_distance_l528_528868

-- Define point and geometric shapes
structure Point :=
  (x : ℝ)
  (y : ℝ)

def Parabola (P : Point) : Prop := P.x^2 = 4 * P.y
def Circle (P1 : Point) : Prop := (P1.x - 2)^2 + (P1.y + 1)^2 = 1

-- Define the point P and point P1
variable (P : Point)
variable (P1 : Point)

-- Condition: P is on the parabola
axiom on_parabola : Parabola P

-- Condition: P1 is on the circle
axiom on_circle : Circle P1

-- Theorem: coordinates of P when the function d + distance(P, P1) is minimized
theorem minimize_d_and_distance :
  P = { x := 2 * Real.sqrt 2 - 2, y := 3 - 2 * Real.sqrt 2 } :=
sorry

end minimize_d_and_distance_l528_528868


namespace triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l528_528313

variable {a b c x : ℝ}

-- Part (1)
theorem triangle_ABC_is_isosceles (h : (a + b) * 1 ^ 2 - 2 * c * 1 + (a - b) = 0) : a = c :=
by 
  -- Proof omitted
  sorry

-- Part (2)
theorem roots_of_quadratic_for_equilateral (h_eq : a = b ∧ b = c ∧ c = a) : 
  (∀ x : ℝ, (a + a) * x ^ 2 - 2 * a * x + (a - a) = 0 → (x = 0 ∨ x = 1)) :=
by 
  -- Proof omitted
  sorry

end triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l528_528313


namespace notebooks_last_days_l528_528412

/-- John buys 5 notebooks, each with 40 pages. 
    He uses 4 pages per day. 
    Prove the notebooks last 50 days. -/
theorem notebooks_last_days : 
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  (notebooks * pages_per_notebook) / pages_per_day = 50 := 
by
  -- Definitions
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  calc
    (notebooks * pages_per_notebook) / pages_per_day
      = (5 * 40) / 4 : by rfl
      ... = 200 / 4 : by rfl
      ... = 50 : by rfl

end notebooks_last_days_l528_528412


namespace first_word_of_message_is_net_l528_528154

section encryption_problem

variables (x : ℕ → ℕ) (k1 k2 : ℕ)

-- Defining y_i based on given transformations
def y (i : ℕ) : ℕ :=
  if i % 2 = 1 then
    2 * x i + x (i + 2) + (-1) ^ (i + 1) / 2 - 1
  else
    x (i - 1) + x i + (-1) ^ (i / 2) * k2

-- The extended sequence after transformations
def transformed_sequence : ℕ → ℕ
  | i => (y i) % 32 

-- Given result sequence
def result_sequence : list ℕ := [23, 4, 21, 7, 24, 2, 26, 28, 28, 4, 2, 16, 24, 10]

-- Start the theorem state
theorem first_word_of_message_is_net (h_transformed : (∀ i, transformed_sequence i = result_sequence.nth (i).iget)) : 
  ∃ original_message : string, first_word original_message = "нет" :=
sorry

end encryption_problem

end first_word_of_message_is_net_l528_528154


namespace find_n_l528_528311

noncomputable def a (n : ℕ) : ℕ := 2 ^ n

noncomputable def b (n : ℕ) : ℤ := -n * 2 ^ n

noncomputable def S (n : ℕ) : ℤ := ∑ i in Finset.range (n + 1), b i

theorem find_n :
  ∃ n : ℕ, S n + n * 2 ^ (n + 1) > 50 :=
by {
  use 5,
  sorry
}

end find_n_l528_528311


namespace solution_set_a_eq_1_no_positive_a_for_all_x_l528_528763

-- Define the original inequality for a given a.
def inequality (a x : ℝ) : Prop := |a * x - 1| + |a * x - a| ≥ 2

-- Part 1: For a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | inequality 1 x } = {x : ℝ | x ≤ 0 ∨ x ≥ 2} :=
sorry

-- Part 2: There is no positive a such that the inequality holds for all x ∈ ℝ
theorem no_positive_a_for_all_x :
  ¬ ∃ a > 0, ∀ x : ℝ, inequality a x :=
sorry

end solution_set_a_eq_1_no_positive_a_for_all_x_l528_528763


namespace total_pencils_total_erasers_total_crayons_l528_528011

theorem total_pencils (n : ℕ) (n = 18) (p = 6) (p_add = 5) (extra_p : ℕ := 10 * p_add) : 18 * 6 + 10 * 5 = 158 := sorry

theorem total_erasers (n : ℕ) (n = 18) (e = 3) (e_add = 2) (extra_e : ℕ := 8 * e_add) : 18 * 3 + 8 * 2 = 70 := sorry

theorem total_crayons (n : ℕ) (n = 18) (c = 12) (c_add = 8) (extra_c : ℕ := 10 * c_add) : 18 * 12 + 10 * 8 = 296 := sorry

end total_pencils_total_erasers_total_crayons_l528_528011


namespace range_of_x_l528_528332
open Real

noncomputable def even_and_continuous (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x)

theorem range_of_x (f : ℝ → ℝ) (h1 : even_and_continuous f) 
  (h2 : continuous f)
  (h3 : ∀ x > 0, deriv f x < 0)
  (hx : f (log 10⁻¹) > f 1) :
  ∃ x, \(\frac{1}{10} < x < 10\) :=
sorry

end range_of_x_l528_528332


namespace inequality_proof_l528_528857

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a + 1 / b + 9 / c + 25 / d) ≥ (100 / (a + b + c + d)) :=
by
  sorry

end inequality_proof_l528_528857


namespace smallest_base_subset_proof_l528_528416

noncomputable def smallest_base_subset_size (n : ℕ) : ℕ :=
  let r := nat.find (λ r, r * (r + 1) / 2 ≥ n) in n + 1 + r

theorem smallest_base_subset_proof (n : ℕ) : 
  ∃ k, k = smallest_base_subset_size n ∧ ∀ (M : set ℤ) (P : set ℤ), 
    (M = {x | -n ≤ x ∧ x ≤ n}) → 
    (∀ x ∈ M, ∃ A ⊆ P, A.sum = x) → 
    k = n + 1 + (nat.find (λ r, r * (r + 1) / 2 ≥ n)) :=
by
  sorry

end smallest_base_subset_proof_l528_528416


namespace sum_of_smaller_angles_in_convex_pentagon_l528_528793

theorem sum_of_smaller_angles_in_convex_pentagon :
  ∃ (θ : ℕ → ℝ), θ 1 + θ 2 + θ 3 + θ 4 + θ 5 ∈ set.Icc 0 360 :=
sorry

end sum_of_smaller_angles_in_convex_pentagon_l528_528793


namespace distances_sum_l528_528294

theorem distances_sum (a b c d k : ℕ) (h_distances : (λ (distances : List ℕ), distances.sorted <| distances.length = 10 ∧ distances = [2, 5, 6, 8, 9, k, 15, 17, 20, 22])) :
  k = 14 :=
begin
  sorry
end

end distances_sum_l528_528294


namespace repeating_decimal_as_fraction_l528_528170

theorem repeating_decimal_as_fraction :
  (0.58207 : ℝ) = 523864865 / 999900 := sorry

end repeating_decimal_as_fraction_l528_528170


namespace solution_set_of_inequality_l528_528922

theorem solution_set_of_inequality:
  {x : ℝ | 3 ≤ |2 - x| ∧ |2 - x| < 9} = {x : ℝ | (-7 < x ∧ x ≤ -1) ∨ (5 ≤ x ∧ x < 11)} :=
by
  sorry

end solution_set_of_inequality_l528_528922


namespace max_fourth_term_l528_528924

open Nat

/-- Constants representing the properties of the arithmetic sequence -/
axiom a : ℕ
axiom d : ℕ
axiom pos1 : a > 0
axiom pos2 : a + d > 0
axiom pos3 : a + 2 * d > 0
axiom pos4 : a + 3 * d > 0
axiom pos5 : a + 4 * d > 0
axiom sum_condition : 5 * a + 10 * d = 75

/-- Theorem stating the maximum fourth term of the arithmetic sequence -/
theorem max_fourth_term : a + 3 * d = 22 := sorry

end max_fourth_term_l528_528924


namespace find_tricias_age_l528_528531

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l528_528531


namespace trapezoid_intersection_theorem_l528_528427

-- Lean definitions for a trapezoid and related geometric transformations
structure Trapezoid (α : Type) [Field α] :=
(A B C D E : α)
(AB : α)
(BC : α)
(AD : α)
(CD : α)
(E_int_AD_BC : E = A * AD / (A * AD + B * BC)) -- Definition of E as the intersection point of AD and BC

def B_n_plus_one (A_n A C B D E : α) : α := 
-- Definition for B_{n+1} - intersection of A_nC and BD
sorry

def A_n_plus_one (E B_n_plus_one A B : α) : α :=
-- Definition for A_{n+1} - intersection of EB_{n+1} and AB
sorry

theorem trapezoid_intersection_theorem
  (α : Type) [Field α] 
  (A B C D E : α) 
  (AB BC AD CD : α)
  (trp : Trapezoid α)
  (A_n : ℕ → α) (n : ℕ) :
  A_n B = AB / (n + 1) := 
sorry

end trapezoid_intersection_theorem_l528_528427


namespace problem_part1_problem_part2_problem_part3_l528_528105

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := { n | n > 3 }
def B : Set ℕ := { n | n % 3 = 0 }
def probability (E : Set ℕ) : ℚ := (Set.card E : ℚ) / (Set.card S : ℚ)

theorem problem_part1 : probability A = 4 / 7 := by
  sorry

theorem problem_part2 : probability B = 2 / 7 := by
  sorry

theorem problem_part3 : probability (A ∪ B) = 5 / 7 := by
  sorry

end problem_part1_problem_part2_problem_part3_l528_528105


namespace dylan_speed_constant_l528_528273

theorem dylan_speed_constant (d t s : ℝ) (h1 : d = 1250) (h2 : t = 25) (h3 : s = d / t) : s = 50 := 
by 
  -- Proof steps will go here
  sorry

end dylan_speed_constant_l528_528273


namespace num_special_fractions_eq_one_l528_528260

-- Definitions of relatively prime and positive
def are_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def is_positive (n : ℕ) : Prop := n > 0

-- Statement to prove the number of such fractions
theorem num_special_fractions_eq_one : 
  (∀ (x y : ℕ), is_positive x → is_positive y → are_rel_prime x y → 
    (x + 1) * 10 * y = (y + 1) * 11 * x →
    ((x = 5 ∧ y = 11) ∨ False)) := sorry

end num_special_fractions_eq_one_l528_528260


namespace total_employee_costs_in_February_l528_528791

def weekly_earnings (hours_per_week : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_week * rate_per_hour

def monthly_earnings 
  (hours_per_week : ℕ) 
  (rate_per_hour : ℕ) 
  (weeks_worked : ℕ) 
  (bonus_deduction : ℕ := 0) 
  : ℕ :=
  weeks_worked * weekly_earnings hours_per_week rate_per_hour + bonus_deduction

theorem total_employee_costs_in_February 
  (hours_Fiona : ℕ := 40) (rate_Fiona : ℕ := 20) (weeks_worked_Fiona : ℕ := 3)
  (hours_John : ℕ := 30) (rate_John : ℕ := 22) (overtime_hours_John : ℕ := 10)
  (hours_Jeremy : ℕ := 25) (rate_Jeremy : ℕ := 18) (bonus_Jeremy : ℕ := 200)
  (hours_Katie : ℕ := 35) (rate_Katie : ℕ := 21) (deduction_Katie : ℕ := 150)
  (hours_Matt : ℕ := 28) (rate_Matt : ℕ := 19) : 
  monthly_earnings hours_Fiona rate_Fiona weeks_worked_Fiona 
  + monthly_earnings hours_John rate_John 4 
    + overtime_hours_John * (rate_John * 3 / 2)
  + monthly_earnings hours_Jeremy rate_Jeremy 4 bonus_Jeremy
  + monthly_earnings hours_Katie rate_Katie 4 - deduction_Katie
  + monthly_earnings hours_Matt rate_Matt 4 = 13278 := 
by sorry

end total_employee_costs_in_February_l528_528791


namespace determine_ABCC_l528_528269

theorem determine_ABCC :
  ∃ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ 
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ 
    C ≠ D ∧ C ≠ E ∧ 
    D ≠ E ∧ 
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    1000 * A + 100 * B + 11 * C = (11 * D - E) * 100 + 11 * D * E ∧ 
    1000 * A + 100 * B + 11 * C = 1966 :=
sorry

end determine_ABCC_l528_528269


namespace simplify_expression_l528_528108

theorem simplify_expression (x y : ℝ) :
  5 * x - 3 * y + 9 * x ^ 2 + 8 - (4 - 5 * x + 3 * y - 9 * x ^ 2) = 18 * x ^ 2 + 10 * x - 6 * y + 4 :=
by
  sorry

end simplify_expression_l528_528108


namespace four_digit_numbers_with_2_and_3_adjacent_l528_528365

-- Definitions of the constraints
def four_digit_number (digits : List ℕ) : Prop :=
  digits.length = 4 ∧ (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) -- no repeating digits

def adjacent (digits : List ℕ) : Prop :=
  ∃ i, (digits.nth i = some 2 ∧ digits.nth (i+1) = some 3) ∨ (digits.nth i = some 3 ∧ digits.nth (i+1) = some 2)

def valid_digits (digits : List ℕ) : Prop :=
  digits.all (λ d, d ∈ [0, 1, 2, 3, 4, 5])

-- The theorem to prove
theorem four_digit_numbers_with_2_and_3_adjacent : 
  ∃ digits : List ℕ, four_digit_number digits ∧ adjacent digits ∧ valid_digits digits → list.length (list.filter (λ digits, four_digit_number digits ∧ adjacent digits ∧ valid_digits digits) (list.permutations [0, 1, 2, 3, 4, 5])) = 60 :=
by repeat sorry

end four_digit_numbers_with_2_and_3_adjacent_l528_528365


namespace final_amount_correct_l528_528212

def initial_amount : ℝ := 100

def wager (amount : ℝ) : ℝ := amount / 2

def win (amount : ℝ) : ℝ := amount + wager amount

def loss (amount : ℝ) : ℝ := amount - 0.6 * wager amount

def final_amount : ℝ :=
  let after_first_win := win initial_amount
  let after_second_win := win after_first_win
  let after_first_loss := loss after_second_win
  let final_amount := loss after_first_loss
  final_amount

theorem final_amount_correct : final_amount = 196 := by
  sorry

end final_amount_correct_l528_528212


namespace cos_identity_arithmetic_sequence_in_triangle_l528_528789

theorem cos_identity_arithmetic_sequence_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h1 : 2 * b = a + c)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : A + B + C = Real.pi)
  : 5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 4 := 
  sorry

end cos_identity_arithmetic_sequence_in_triangle_l528_528789


namespace bridge_length_l528_528566

theorem bridge_length (speed_in_kmph : ℝ) (time_in_minutes : ℝ) : 
  speed_in_kmph = 10 → 
  time_in_minutes = 18 → 
  (speed_in_kmph * (time_in_minutes / 60)) = 3 :=
by
  intros h_speed h_time
  rw [h_speed, h_time]
  norm_num
  sorry

end bridge_length_l528_528566


namespace cats_groomed_is_3_l528_528821

-- Define the conditions
def dog_grooming_time_hours : ℝ := 2.5
def cat_grooming_time_hours : ℝ := 0.5
def total_grooming_time_minutes : ℝ := 840
def number_of_dogs : ℕ := 5

-- Calculate the time in minutes for grooming a dog and a cat
def dog_grooming_time_minutes : ℝ := dog_grooming_time_hours * 60
def cat_grooming_time_minutes : ℝ := cat_grooming_time_hours * 60

-- Calculate total time spent grooming dogs
def total_dog_grooming_time : ℝ := number_of_dogs * dog_grooming_time_minutes

-- Calculate the time spent grooming cats
def total_cat_grooming_time : ℝ := total_grooming_time_minutes - total_dog_grooming_time

-- Calculate the number of cats groomed
def number_of_cats : ℝ := total_cat_grooming_time / cat_grooming_time_minutes

-- Proof statement
theorem cats_groomed_is_3 : number_of_cats = 3 := by
  -- The proof steps are omitted
  sorry

end cats_groomed_is_3_l528_528821


namespace line_through_BC_is_tangent_to_circumcircle_BDE_l528_528060

open EuclideanGeometry

variables {A B C D E : Point}
variable {k : Circle}

-- Conditions
axiom is_isosceles_triangle (h : Triangle A B C) (hiso : isIsoscelesTriangle A B C (dist EQ A C B C))
axiom lies_on_arc (D : Point) (h : lies_on D (arc (circumcircle A B C) B C) ∧ D ≠ B ∧ D ≠ C)
axiom intersection (E : Point) (h1 : lies_on E (line_through C D)) (h2 : lies_on E (line_through A B))

-- Question
theorem line_through_BC_is_tangent_to_circumcircle_BDE (h : is_isosceles_triangle A B C hiso)
  (h1 : lies_on_arc D h) (h2 : intersection E h1 h2) : isTangent (line_through B C) (circumcircle B D E) :=
sorry

end line_through_BC_is_tangent_to_circumcircle_BDE_l528_528060


namespace oranges_ratio_l528_528109

theorem oranges_ratio (initial_oranges_kgs : ℕ) (additional_oranges_kgs : ℕ) (total_oranges_three_weeks : ℕ) :
  initial_oranges_kgs = 10 →
  additional_oranges_kgs = 5 →
  total_oranges_three_weeks = 75 →
  (2 * (total_oranges_three_weeks - (initial_oranges_kgs + additional_oranges_kgs)) / 2) / (initial_oranges_kgs + additional_oranges_kgs) = 2 :=
by
  intros h_initial h_additional h_total
  sorry

end oranges_ratio_l528_528109


namespace square_area_l528_528128

theorem square_area (perimeter : ℝ) (h : perimeter = 32) : 
  ∃ (side area : ℝ), side = perimeter / 4 ∧ area = side * side ∧ area = 64 := 
by
  sorry

end square_area_l528_528128


namespace ellipse_properties_l528_528341

-- Define the properties of the ellipse
variables (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = 2) (h4 : (a^2 - b^2) / a^2 = 1 / 2)

-- Main theorem: finding the equation of the ellipse and maximum area of triangle OPQ
theorem ellipse_properties :
  (∃ C: ℝ × ℝ → Prop, (C = λ x y, (x^2 / 2) + y^2 = 1)) ∧
  (∃ M, M = (-2: ℝ, 0)) ∧
  (∃ max_area, max_area = sqrt 2 / 2) :=
by
  have b_val : b = 1 := by linarith [h3]
  have a_squared_val : a^2 = 2 := by linarith [h4, b_val]
  use λ (x y : ℝ), (x^2 / 2) + y^2 = 1
  use (-2: ℝ, 0)
  use sqrt 2 / 2
  sorry

end ellipse_properties_l528_528341


namespace annual_sales_profit_relationship_and_maximum_l528_528599

def cost_per_unit : ℝ := 6
def selling_price (x : ℝ) := x > 6
def sales_volume (u : ℝ) := u * 10000
def proportional_condition (x u : ℝ) := (585 / 8) - u = 2 * (x - 21 / 4) ^ 2
def sales_volume_condition : Prop := proportional_condition 10 28

theorem annual_sales_profit_relationship_and_maximum (x u y : ℝ) 
    (hx : selling_price x) 
    (hu : proportional_condition x u) 
    (hs : sales_volume_condition) :
    (y = (-2 * x^3 + 33 * x^2 - 108 * x - 108)) ∧ 
    (x = 9 → y = 135) := 
sorry

end annual_sales_profit_relationship_and_maximum_l528_528599


namespace race_distance_l528_528386

theorem race_distance (x : ℝ) (D : ℝ) (vA vB : ℝ) (head_start win_margin : ℝ):
  vA = 5 * x →
  vB = 4 * x →
  head_start = 100 →
  win_margin = 200 →
  (D - win_margin) / vB = (D - head_start) / vA →
  D = 600 :=
by 
  sorry

end race_distance_l528_528386


namespace length_of_base_l528_528268

def isosceles (A B C : ℝ) : Prop :=
A = B

def double_length_triangle (A B C : ℝ) : Prop :=
A = 2 * C ∨ C = 2 * A

theorem length_of_base (A B C : ℝ) (h_isosceles : isosceles A B C) (h_double : double_length_triangle B C A) (h_AB : A = 6) :
C = 3 :=
by
  have h1 : A = B := h_isosceles
  have h2 : A = 6 := h_AB
  cases h_double with
    | inl h =>
      have : A = 2 * C := h
      sorry
    | inr h =>
      have : C = 2 * A := h
      sorry

end length_of_base_l528_528268


namespace length_of_bridge_l528_528222

noncomputable def L_train : ℝ := 110
noncomputable def v_train : ℝ := 72 * (1000 / 3600)
noncomputable def t : ℝ := 12.099

theorem length_of_bridge : (v_train * t - L_train) = 131.98 :=
by
  -- The proof should come here
  sorry

end length_of_bridge_l528_528222


namespace ABCD_is_trapezoid_l528_528398

-- Definitions and conditions

-- In an inscribed quadrilateral ABCD
def inscribed_quadrilateral (A B C D : Point) : Prop := 
sorry  -- to be defined

-- Define O as the intersection point of AC and BD
def intersection_point (O A C B D : Point) : Prop := 
sorry  -- to be defined

-- The circumcircle of triangle COD passes through O1 (center of the circumcircle of quadrilateral ABCD)
def circumcircle_cod_through_O1 (C O D O1 : Point) : Prop := 
sorry  -- to be defined

-- Proof that quadrilateral ABCD is a trapezoid
-- i.e., sides BC and AD are parallel
theorem ABCD_is_trapezoid (A B C D O O1 : Point) 
  (h1 : inscribed_quadrilateral A B C D)
  (h2 : intersection_point O A C B D)
  (h3 : circumcircle_cod_through_O1 C O D O1) : 
  parallel (line B C) (line A D) :=
sorry

end ABCD_is_trapezoid_l528_528398


namespace factorial_cubic_divisors_l528_528687

theorem factorial_cubic_divisors :
  let P := (1! * 2! * 3! * 4! * 5! * 6!)
  let prime_factorization_P : ℕ → ℕ := fun n =>
    match n with
    | 2 => 12
    | 3 => 5
    | 5 => 2
    | _ => 0
  (∏ p in {2, 3, 5}, (prime_factorization_P p) // 3 + 1) = 10 :=
by
  sorry

end factorial_cubic_divisors_l528_528687


namespace min_cosine_largest_angle_l528_528317

theorem min_cosine_largest_angle (a b c : ℕ → ℝ) 
  (triangle_inequality: ∀ i, a i ≤ b i ∧ b i ≤ c i)
  (pythagorean_inequality: ∀ i, (a i)^2 + (b i)^2 ≥ (c i)^2)
  (A : ℝ := ∑' i, a i)
  (B : ℝ := ∑' i, b i)
  (C : ℝ := ∑' i, c i) :
  (A^2 + B^2 - C^2) / (2 * A * B) ≥ 1 - (Real.sqrt 2) :=
sorry

end min_cosine_largest_angle_l528_528317


namespace total_cost_correct_l528_528071

-- Define the cost of each category of items
def cost_of_book : ℕ := 16
def cost_of_binders : ℕ := 3 * 2
def cost_of_notebooks : ℕ := 6 * 1

-- Define the total cost calculation
def total_cost : ℕ := cost_of_book + cost_of_binders + cost_of_notebooks

-- Prove that the total cost of Léa's purchases is 28
theorem total_cost_correct : total_cost = 28 :=
by {
  -- This is where the proof would go, but it's omitted for now.
  sorry
}

end total_cost_correct_l528_528071


namespace max_consecutive_resistant_numbers_l528_528650

-- Definition of sum of divisors
def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range n.succ).filter (λ d, n % d = 0).sum id

-- Definition of a resistant number
def is_resistant (n : ℕ) : Prop :=
  n ≥ 2 ∧ Nat.gcd n (sum_of_divisors n) = 1

-- Lean statement proving the maximum number of consecutive resistant numbers is 5
theorem max_consecutive_resistant_numbers : ∃ (k : ℕ), 
(k ≤ 5) ∧
(∀ (a : ℕ), (∀ j : ℕ, j < k → is_resistant (a + j)) → k ≤ 5 ) :=
sorry

end max_consecutive_resistant_numbers_l528_528650


namespace rain_probability_tel_aviv_l528_528498

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l528_528498


namespace min_value_geometric_sequence_l528_528849

noncomputable def geometric_min_value (b1 b2 b3 : ℝ) (s : ℝ) : ℝ :=
  3 * b2 + 4 * b3

theorem min_value_geometric_sequence (s : ℝ) :
  ∃ s : ℝ, 2 = b1 ∧ b2 = 2 * s ∧ b3 = 2 * s^2 ∧ 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end min_value_geometric_sequence_l528_528849


namespace isosceles_triangle_sides_l528_528713

variable (n m : ℝ)
variable (h1 : 4 * m^2 > n^2)

theorem isosceles_triangle_sides (h1 : 4 * m^2 > n^2) :
  let BC := (2 * m^2) / Real.sqrt (4 * m^2 - n^2)
  let AC := (2 * m * n) / Real.sqrt (4 * m^2 - n^2)
  ∃ BC AC, BC = (2 * m^2) / Real.sqrt (4 * m^2 - n^2) ∧
            AC = (2 * m * n) / Real.sqrt (4 * m^2 - n^2) :=
by
  use (2 * m^2) / Real.sqrt (4 * m^2 - n^2)
  use (2 * m * n) / Real.sqrt (4 * m^2 - n^2)
  constructor
  · rfl
  · rfl

end isosceles_triangle_sides_l528_528713


namespace must_hold_true_l528_528340

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom derivative_f : ∀ (x : ℝ), f' x = derivative f x
axiom f_not_constant : ¬(∀ x y, f x = f y)
axiom inequality_condition : ∀ (x : ℝ), x ∈ Set.Ici 0 → (x + 1) * f x + x * f' x ≥ 0

theorem must_hold_true : f 1 < 2 * Real.exp 1 * f 2 :=
by
  have exp_one := Real.exp 1
  calc
    f 1 < 2 * exp_one * f 2 := sorry

end must_hold_true_l528_528340


namespace factorial_difference_l528_528670

theorem factorial_difference :
  9! - 8! = 322560 :=
by
  -- specify the definition of factorial
  have fact_8_eq : 8! = 40320 := by norm_num,
  have fact_9_eq : 9! = 9 * 8! := rfl,
  rw [fact_9_eq, fact_8_eq],
  norm_num

end factorial_difference_l528_528670


namespace xyz_cubed_over_xyz_eq_21_l528_528866

open Complex

theorem xyz_cubed_over_xyz_eq_21 {x y z : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 18)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 21 :=
sorry

end xyz_cubed_over_xyz_eq_21_l528_528866


namespace partitions_are_perfect_fifth_power_l528_528594

-- Define the type of partitions as a natural number operation
def F : ℕ → ℕ := sorry
-- Condition for how a broken domino is structured
def is_broken_domino_partition (n : ℕ) : Prop := sorry
-- Number of ways to partition a 1x5n rectangle
noncomputable def F_5n (n : ℕ) := F(n) ^ 5

theorem partitions_are_perfect_fifth_power (n : ℕ) (h : is_broken_domino_partition (5 * n)) : 
  F(5 * n) = F(n) ^ 5 :=
by
  sorry

end partitions_are_perfect_fifth_power_l528_528594


namespace circle_equation_l528_528743

open Real

-- Definitions
def A : ℝ × ℝ := (4, 9)
def B : ℝ × ℝ := (6, -3)

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def equation_of_circle (center : ℝ × ℝ) (radius : ℝ) : ℝ × ℝ → Prop :=
  λ p, (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Statement to Prove
theorem circle_equation : 
  equation_of_circle (midpoint A B) ((distance A B) / 2) = equation_of_circle (5, 3) (sqrt 37) :=
by
  -- Correct answer encoded directly into Lean statement
  sorry

end circle_equation_l528_528743


namespace find_extrema_A_l528_528230

def eight_digit_number(n : ℕ) : Prop := n ≥ 10^7 ∧ n < 10^8

def coprime_with_thirtysix(n : ℕ) : Prop := Nat.gcd n 36 = 1

def transform_last_to_first(n : ℕ) : ℕ := 
  let last := n % 10
  let rest := n / 10
  last * 10^7 + rest

theorem find_extrema_A :
  ∃ (A_max A_min : ℕ), 
    (∃ B_max B_min : ℕ, 
      eight_digit_number B_max ∧ 
      eight_digit_number B_min ∧ 
      coprime_with_thirtysix B_max ∧ 
      coprime_with_thirtysix B_min ∧ 
      B_max > 77777777 ∧ 
      B_min > 77777777 ∧ 
      transform_last_to_first B_max = A_max ∧ 
      transform_last_to_first B_min = A_min) ∧ 
    A_max = 99999998 ∧ 
    A_min = 17777779 := 
  sorry

end find_extrema_A_l528_528230


namespace part1_part2_l528_528984

theorem part1 (x y : ℕ) (h1 : 25 * x + 30 * y = 1500) (h2 : x = 2 * y - 4) : x = 36 ∧ y = 20 :=
by
  sorry

theorem part2 (x y : ℕ) (h1 : x + y = 60) (h2 : x ≥ 2 * y)
  (h_profit : ∃ p, p = 7 * x + 10 * y) : 
  ∃ x y profit, x = 40 ∧ y = 20 ∧ profit = 480 :=
by
  sorry

end part1_part2_l528_528984


namespace find_a_l528_528029

-- Define the right triangle and given conditions
axiom right_triangle_ABC (a b c : ℝ) (C : triangle) : Prop :=
C.angle C = 90 ∧ 
b = 6 ∧ 
c = 10 ∧
C.is_right_triangle

-- The goal is to find the value of a
theorem find_a (a b c : ℝ) (C : triangle) (h : right_triangle_ABC a b c C) : a = 8 :=
by sorry

end find_a_l528_528029


namespace power_function_through_point_l528_528506

theorem power_function_through_point (k α : ℝ) 
  (h1 : ∀ x, f x = k * x ^ α) 
  (h2 : f (1/2) = (2 : ℝ)⁻¹) 
  (h3 : (2 : ℝ) ^ (1/2) = real.sqrt 2) : 
  k + α = 3/2 := 
sorry

end power_function_through_point_l528_528506


namespace oatmeal_cookies_count_l528_528884

def total_cookies : ℕ := 36
def ratio_choco_oatmeal_peanut : Prod (Prod ℕ ℕ) ℕ := (2, 3, 4)

theorem oatmeal_cookies_count (h1 : total_cookies = 36)
                            (h2 : ratio_choco_oatmeal_peanut = (2, 3, 4)) :
  let parts := (2 + 3 + 4)
  let cookies_per_part := total_cookies / parts
  let oatmeal_cookies := 3 * cookies_per_part
  oatmeal_cookies = 12 := by
  sorry

end oatmeal_cookies_count_l528_528884


namespace area_GCD_l528_528478

-- Definitions from the conditions
def side_length := 12
def area_square := side_length * side_length
def E : ℝ := 1 / 3 * side_length
def F : ℝ := 1 / 2 * (side_length + E)
def G : ℝ := 1 / 2 * (side_length + E)
def area_BEGF := 50

-- Lean 4 statement of the problem
theorem area_GCD : 
  area_square = 144 ∧
  E = 4 ∧ 
  area_BEGF = 50 →
  let area_triangle_GCD := (side_length / 2) * (side_length / 2) / 2 
  in area_triangle_GCD = 4 := by sorry

end area_GCD_l528_528478


namespace triangle_height_l528_528036

theorem triangle_height (b h : ℕ) (A : ℕ) (hA : A = 50) (hb : b = 10) :
  A = (1 / 2 : ℝ) * b * h → h = 10 := 
by
  sorry

end triangle_height_l528_528036


namespace total_food_consumed_l528_528132

theorem total_food_consumed (n1 n2 f1 f2 : ℕ) (h1 : n1 = 4000) (h2 : n2 = n1 - 500) (h3 : f1 = 10) (h4 : f2 = f1 - 2) : 
    n1 * f1 + n2 * f2 = 68000 := by 
  sorry

end total_food_consumed_l528_528132


namespace problem_statement_l528_528059

variables (n h m b : ℕ)

theorem problem_statement (h1 : n ≥ h * (m + 1)) (h2 : h ≥ 1) :
  M (n, n * m, b) = m + 1 := sorry

end problem_statement_l528_528059


namespace find_x_l528_528003

theorem find_x
  (x : ℝ)
  (h : 5^29 * x^15 = 2 * 10^29) :
  x = 4 :=
by
  sorry

end find_x_l528_528003


namespace common_vertex_in_longest_paths_l528_528572

theorem common_vertex_in_longest_paths (G : Type) [graph : simple_graph G] (htree : is_tree G) :
  ∃ v : G, ∀ P : set G, is_longest_path G P → v ∈ P :=
sorry

end common_vertex_in_longest_paths_l528_528572


namespace cost_of_each_toy_car_l528_528102

theorem cost_of_each_toy_car (S M C A B : ℕ) (hS : S = 53) (hM : M = 7) (hA : A = 10) (hB : B = 14) 
(hTotalSpent : S - M = C + A + B) (hTotalCars : 2 * C / 2 = 11) : 
C / 2 = 11 :=
by
  rw [hS, hM, hA, hB] at hTotalSpent
  sorry

end cost_of_each_toy_car_l528_528102


namespace periodic_log_sum_l528_528337

noncomputable def f : ℝ → ℝ := 
λ x, log (x + 1) / log 2

theorem periodic_log_sum :
  (∀ x : ℝ, f (x + 2) = f x) → 
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = log (x + 1) / log 2) →
  f 2023 + f (-2024) = 1 :=
by
  intros h_periodicity h_definition
  sorry

end periodic_log_sum_l528_528337


namespace gcd_factorial_5040_l528_528126

theorem gcd_factorial_5040 (b : ℕ) (h1 : Nat.gcd (b + 1)! (b + 4)! = 5040) (h2 : b = 9) : (b + 1)! = 10! := 
by
  sorry

end gcd_factorial_5040_l528_528126


namespace purchase_total_cost_l528_528069

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end purchase_total_cost_l528_528069


namespace gcd_polynomial_l528_528331

theorem gcd_polynomial (b : ℤ) (h : ∃ k : ℤ, b = 2 * 997 * k) : 
  Int.gcd (3 * b^2 + 34 * b + 102) (b + 21) = 21 := 
by
  -- Proof would go here, but is omitted as instructed
  sorry

end gcd_polynomial_l528_528331


namespace length_of_ST_l528_528240

-- Definition of geometric entities and given conditions
variables {P Q R S T : ℝ}
variables (length_QR height_PQR area_trapezoid area_PQR : ℝ)
variables (isosceles_triangle_PQR : Prop)
variables (divided_by_ST : Prop)
variables (isosceles_trapezoid : Prop)
variables (isosceles_triangle_PST : Prop)

-- Given conditions
def triangle_PQR_area : Prop :=
  area_PQR = 180

def trapezoid_area : Prop :=
  area_trapezoid = 135

def altitude_PQR : Prop :=
  height_PQR = 30

def formula_area_triangle : Prop :=
  area_PQR = (1 / 2) * length_QR * height_PQR

def similar_triangles : Prop :=
  isosceles_triangle_PQR ∧ isosceles_triangle_PST ∧ divided_by_ST ∧ isosceles_trapezoid

def smaller_triangle_area : Prop :=
  area_PQR - area_trapezoid = 45

-- Question to prove
theorem length_of_ST : ∃ ST : ℝ, triangle_PQR_area ∧ trapezoid_area ∧ altitude_PQR ∧ formula_area_triangle ∧ similar_triangles ∧ smaller_triangle_area → ST = 6 :=
  sorry

end length_of_ST_l528_528240


namespace quadratic_conversion_l528_528262

def quadratic_to_vertex_form (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 1

theorem quadratic_conversion :
  (∀ x : ℝ, quadratic_to_vertex_form x = 2 * (x - 2)^2 - 9) :=
by
  sorry

end quadratic_conversion_l528_528262


namespace factorial_difference_l528_528669

theorem factorial_difference :
  9! - 8! = 322560 :=
by
  -- specify the definition of factorial
  have fact_8_eq : 8! = 40320 := by norm_num,
  have fact_9_eq : 9! = 9 * 8! := rfl,
  rw [fact_9_eq, fact_8_eq],
  norm_num

end factorial_difference_l528_528669


namespace sheila_hours_MWF_l528_528462

variable (hours_MWF : ℕ)
variable (weekly_earnings : ℕ)
variable (hourly_rate : ℕ)
variable (hours_Tuesday_Thursday_per_day : ℕ)
variable (days_Tuesday_Thursday : ℕ)

-- Conditions
def sheila_conditions :=
  weekly_earnings = 360 ∧
  hourly_rate = 10 ∧
  hours_Tuesday_Thursday_per_day = 6 ∧
  days_Tuesday_Thursday = 2

-- Proof statement
theorem sheila_hours_MWF (hconds : sheila_conditions) :
  hours_MWF = (weekly_earnings / hourly_rate) - (hours_Tuesday_Thursday_per_day * days_Tuesday_Thursday) := sorry

end sheila_hours_MWF_l528_528462


namespace find_factorial_number_l528_528285

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_factorial_number (n : ℕ) : Prop :=
  ∃ x y z : ℕ, (0 ≤ x ∧ x ≤ 5) ∧
               (0 ≤ y ∧ y ≤ 5) ∧
               (0 ≤ z ∧ z ≤ 5) ∧
               n = 100 * x + 10 * y + z ∧
               n = x.factorial + y.factorial + z.factorial

theorem find_factorial_number : ∃ n, is_three_digit_number n ∧ is_factorial_number n ∧ n = 145 :=
by {
  sorry
}

end find_factorial_number_l528_528285


namespace part1_l528_528986

def purchase_price (x y : ℕ) : Prop := 25 * x + 30 * y = 1500
def quantity_relation (x y : ℕ) : Prop := x = 2 * y - 4

theorem part1 (x y : ℕ) (h1 : purchase_price x y) (h2 : quantity_relation x y) : x = 36 ∧ y = 20 :=
sorry

end part1_l528_528986


namespace calculate_AC_squared_l528_528644

-- Define the given conditions
def AB : ℝ := 15
def BC : ℝ := 25
def theta : ℝ := π / 6  -- since θ = 30 degrees = π / 6 radians

-- Law of Cosines definition
def law_of_cosines (a b theta : ℝ) : ℝ := a^2 + b^2 - 2 * a * b * Real.cos theta

-- State the proof statement
theorem calculate_AC_squared :
  law_of_cosines AB BC theta = 850 - 375 * Real.sqrt 3 :=
by
  sorry

end calculate_AC_squared_l528_528644


namespace simplify_expression_l528_528465

theorem simplify_expression (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end simplify_expression_l528_528465


namespace sum_of_integers_with_product_differing_by_2_is_50_l528_528138

theorem sum_of_integers_with_product_differing_by_2_is_50 :
  ∃ x : ℤ, x * (x + 2) = 644 ∧ x + (x + 2) = 50 :=
begin
  sorry
end

end sum_of_integers_with_product_differing_by_2_is_50_l528_528138


namespace mean_of_combined_sets_l528_528913

theorem mean_of_combined_sets :
  (mean_first_set = 10) →
  (mean_second_set = 20) →
  (sum_largest_five_in_second_set = 120) →
  let total_sum_first_set := 4 * mean_first_set in
  let total_sum_second_set := 8 * mean_second_set in
  let sum_remaining_three_second_set := total_sum_second_set - sum_largest_five_in_second_set in
  let combined_sum := total_sum_first_set + total_sum_second_set in
  mean_combined_sets = (combined_sum / 12) :=
by
  assume mean_first_set mean_second_set sum_largest_five_in_second_set,
  let total_sum_first_set := 4 * mean_first_set,
  let total_sum_second_set := 8 * mean_second_set,
  let sum_remaining_three_second_set := total_sum_second_set - sum_largest_five_in_second_set,
  let combined_sum := total_sum_first_set + total_sum_second_set,
  show mean_combined_sets = (combined_sum / 12), from sorry

end mean_of_combined_sets_l528_528913


namespace fewest_four_dollar_frisbees_l528_528178

theorem fewest_four_dollar_frisbees (x y: ℕ): 
    x + y = 64 ∧ 3 * x + 4 * y = 200 → y = 8 := by sorry

end fewest_four_dollar_frisbees_l528_528178


namespace unique_tower_heights_l528_528546

def num_possible_heights (n : ℕ) : ℕ := (n + 1) * ((n + 1) - 1) / 2

def height_values : ℕ := 94

theorem unique_tower_heights : num_possible_heights 4 = 465 := 
by
  have h : num_possible_heights 4 = (4 + 1) * ((4 + 1) - 1) / 2 := rfl
  rw h
  norm_num

# Check the number of heights is as expected
# example (n : ℕ) : n = num_possible_heights 4 := sorry

end unique_tower_heights_l528_528546


namespace find_tangent_lines_l528_528711

noncomputable def tangent_lines (x y : ℝ) : Prop :=
  (x = 2 ∨ 3 * x - 4 * y + 10 = 0)

theorem find_tangent_lines :
  ∃ (x y : ℝ), tangent_lines x y ∧ (x^2 + y^2 = 4) ∧ ((x, y) ≠ (2, 4)) :=
by
  sorry

end find_tangent_lines_l528_528711


namespace minute_hand_rotation_set_back_10_minutes_l528_528785

-- Definition of the problem conditions
def fullCircleMinutes := 60
def fullCircleRadians := 2 * Real.pi

-- The angle rotated by the minute hand when the clock is set back by 10 minutes
def angleWhenSetBackByTenMinutes : Real := (10 / fullCircleMinutes) * fullCircleRadians

-- Statement to be proven
theorem minute_hand_rotation_set_back_10_minutes :
  angleWhenSetBackByTenMinutes = Real.pi / 3 :=
by
  sorry

end minute_hand_rotation_set_back_10_minutes_l528_528785


namespace find_integer_n_cos_l528_528712

theorem find_integer_n_cos : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ (Real.cos (n * Real.pi / 180) = Real.cos (1124 * Real.pi / 180)) ∧ n = 44 := by
  sorry

end find_integer_n_cos_l528_528712


namespace sum_of_D_coordinates_l528_528083

-- Define points as tuples for coordinates (x, y)
structure Point :=
  (x : ℤ)
  (y : ℤ)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def pointD : Point :=
  let C := Point.mk 11 5
  let N := Point.mk 5 9
  let x := 2 * N.x - C.x
  let y := 2 * N.y - C.y
  Point.mk x y

theorem sum_of_D_coordinates : 
  let D := pointD
  D.x + D.y = 12 := by
  sorry

end sum_of_D_coordinates_l528_528083


namespace circle_through_A_and_B_tangent_to_S_l528_528678

open Classical

noncomputable def inversion_point (A B : Point) (r : ℝ) : Point := 
  sorry -- Placeholder for the actual inversion transformation definition

noncomputable def inversion_circle (A : Point) (S : Circle) (r : ℝ) : Circle := 
  sorry -- Placeholder for the actual circle inversion definition

theorem circle_through_A_and_B_tangent_to_S 
  (A B : Point) (S : Circle) : 
  (A ∈ S ∧ B ∈ S) → False ∨
  ∃! (C : Circle), 
    A ∈ C ∧ B ∈ C ∧ Tangent C S ∨
  (let B_inv := inversion_point A B 1 in
   let S_inv := inversion_circle A S 1 in
   B_inv ∈ S_inv → 
     ∃! (C : Circle), 
       A ∈ C ∧ B ∈ C ∧ Tangent C S) ∨
  (let B_inv := inversion_point A B 1 in
   let S_inv := inversion_circle A S 1 in
   B_inv ∉ S_inv → 
     ∃ (C1 C2 : Circle), 
       C1 ≠ C2 ∧
       A ∈ C1 ∧ B ∈ C1 ∧ Tangent C1 S ∧
       A ∈ C2 ∧ B ∈ C2 ∧ Tangent C2 S) ∨
  (let B_inv := inversion_point A B 1 in
   let S_inv := inversion_circle A S 1 in
   B_inv ∉ S_inv ∧ (∃ (C : Circle), Tangent C S → False)): sorry

end circle_through_A_and_B_tangent_to_S_l528_528678


namespace proof_problem_l528_528189

noncomputable def a_n_seq (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2^(n-3) * (1 + 3)

-- Sum of the first n terms of a sequence {a_n}
def S_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range (n+1), a_n i

-- Sum of the first n terms of the sequence {S_n + 1}
def T_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range (n+1), (S_n a_n i + 1)

-- Condition: T_n = S_{n+1} - 1
def condition (a_n : ℕ → ℝ) (n : ℕ) : Prop :=
  T_n a_n n = S_n a_n (n + 1) - 1

-- Limit: lim_{n → ∞} \frac{a_2 + a_4 + \cdots + a_{2n}}{a_1 + a_3 + \cdots + a_{2n-1}} = 2
def sequence_limit (a_n : ℕ → ℝ) : ℝ :=
  real.Inf (Set.image (λ n, (∑ i in range (n+1), if i % 2 = 0 then a_n i else 0) / (∑ i in range n, if i % 2 = 1 then a_n i else 0)) (Set.Ioi 0))

theorem proof_problem : (∃! a_n : ℕ → ℝ, ∀ n, condition a_n n) → sequence_limit a_n_seq = 2 :=
sorry

end proof_problem_l528_528189


namespace equilateral_pentagon_regular_l528_528374

/-- Given an equilateral pentagon ABCDE with angle order A ≥ B ≥ C ≥ D ≥ E, 
    prove that it is a regular pentagon, i.e., all angles are equal. -/
theorem equilateral_pentagon_regular
  {A B C D E : Point}
  (h_eq_sides : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧ dist D E = dist E A)
  (h_angle_order: angle A B C ≥ angle B C D ∧ angle B C D ≥ angle C D E ∧ angle C D E ≥ angle D E A ∧ angle D E A ≥ angle E A B) :
  angle A B C = angle B C D ∧ angle B C D = angle C D E ∧ angle C D E = angle D E A ∧ angle D E A = angle E A B :=
sorry

end equilateral_pentagon_regular_l528_528374


namespace cylinder_volume_transformation_l528_528514

noncomputable def original_volume := 20
noncomputable def original_radius := r
noncomputable def original_height := h
noncomputable def new_radius := 3 * original_radius
noncomputable def new_height := 2 * original_height
noncomputable def volume (radius : ℝ) (height : ℝ) := π * radius ^ 2 * height
noncomputable def new_volume := volume new_radius new_height

theorem cylinder_volume_transformation :
  (original_volume = volume original_radius original_height) →
  new_volume = 360 := by
  sorry

end cylinder_volume_transformation_l528_528514


namespace cylinder_volume_transformation_l528_528515

noncomputable def original_volume := 20
noncomputable def original_radius := r
noncomputable def original_height := h
noncomputable def new_radius := 3 * original_radius
noncomputable def new_height := 2 * original_height
noncomputable def volume (radius : ℝ) (height : ℝ) := π * radius ^ 2 * height
noncomputable def new_volume := volume new_radius new_height

theorem cylinder_volume_transformation :
  (original_volume = volume original_radius original_height) →
  new_volume = 360 := by
  sorry

end cylinder_volume_transformation_l528_528515


namespace total_arrangements_if_A_at_badminton_l528_528117

-- Define the background conditions
def games_duration := "September 23 to October 8, 2023"

-- Define the volunteers
inductive Volunteer
| A | B | C | D | E

-- Define the venues
inductive Venue
| Badminton | Swimming | Shooting | Gymnastics

-- Define the main condition where A goes to the badminton venue
def A_goes_to_badminton (a : Volunteer) (v : Venue) : Prop :=
a = Volunteer.A ∧ v = Venue.Badminton

-- The main theorem stating the total number of different arrangements
theorem total_arrangements_if_A_at_badminton :
  let volunteers := [Volunteer.A, Volunteer.B, Volunteer.C, Volunteer.D, Volunteer.E],
      venues := [Venue.Badminton, Venue.Swimming, Venue.Shooting, Venue.Gymnastics] in
  (∃ (arrangement : Volunteer → Venue), A_goes_to_badminton Volunteer.A Venue.Badminton ∧ 
    -- additional conditions to ensure each venue has at least one volunteer
    ∀ v, (∃ v' : Volunteer, arrangement v' = v) ∧ 
    -- each volunteer to only one venue
    ∀ v' v'', arrangement v' = arrangement v'' → v' = v'') →
  -- total number of arrangements is 60
  ∑ arr in (set_finite (volunteers_permutations volunteers venues)), 1 = 60 :=
sorry

end total_arrangements_if_A_at_badminton_l528_528117


namespace distance_between_truck_and_car_l528_528633

noncomputable def speed_truck : ℝ := 65
noncomputable def speed_car : ℝ := 85
noncomputable def time : ℝ := 3 / 60

theorem distance_between_truck_and_car : 
  let Distance_truck := speed_truck * time,
      Distance_car := speed_car * time in
  Distance_car - Distance_truck = 1 :=
by {
  sorry
}

end distance_between_truck_and_car_l528_528633


namespace infinite_composites_in_sequence_l528_528839
open Nat

theorem infinite_composites_in_sequence
  (a : ℕ → ℕ)
  (k : ℕ)
  (h_mono : ∀ m n, m < n → a m < a n)
  (h_sum : ∀ n > k, ∃ i j < n, a n = a i + a j) : 
  ∃ᶠ n, ¬ prime (a n) :=
by 
  sorry

end infinite_composites_in_sequence_l528_528839


namespace number_of_odd_digit_5_digit_integers_divisible_by_5_l528_528772

theorem number_of_odd_digit_5_digit_integers_divisible_by_5 : 
  (card {n : ℕ | 10000 ≤ n ∧ n ≤ 99999 ∧ 
          (∀ i : ℕ, i ∈ [0,1,2,3,4] → (n.digits 10).get? i ∈ some [1,3,5,7,9]) ∧ 
          n % 5 = 0}) = 625 :=
by
  sorry

end number_of_odd_digit_5_digit_integers_divisible_by_5_l528_528772


namespace smaller_number_is_42_l528_528145

theorem smaller_number_is_42 (x y : ℕ) (h1 : x + y = 96) (h2 : y = x + 12) : x = 42 :=
by
  have h3 : 2 * x + 12 = 96, from sorry
  have h4 : 2 * x = 84, from sorry
  have h5 : x = 42, from sorry
  exact h5

end smaller_number_is_42_l528_528145


namespace molecular_weight_of_acetic_acid_l528_528945

-- Define the molecular weight of 7 moles of acetic acid
def molecular_weight_7_moles_acetic_acid := 420 

-- Define the number of moles of acetic acid
def moles_acetic_acid := 7

-- Define the molecular weight of 1 mole of acetic acid
def molecular_weight_1_mole_acetic_acid := molecular_weight_7_moles_acetic_acid / moles_acetic_acid

-- The theorem stating that given the molecular weight of 7 moles of acetic acid, we have the molecular weight of the acetic acid
theorem molecular_weight_of_acetic_acid : molecular_weight_1_mole_acetic_acid = 60 := by
  -- proof to be solved
  sorry

end molecular_weight_of_acetic_acid_l528_528945


namespace num_valid_sets_l528_528863

def is_prime (p : ℕ) : Prop := p > 1 ∧ (∀ m : ℕ, m ∣ p → m = 1 ∨ m = p)

def valid_set (A : Set ℤ) : Prop :=
  2023 ∈ A ∧ (∀ a b ∈ A, a ≠ b → is_prime (Int.natAbs (a - b)))

theorem num_valid_sets : ∀ {n : ℕ}, n ≥ 4 → (∃ A : Set ℤ, Set.card A = n ∧ valid_set A) → n = 4 :=
sorry

end num_valid_sets_l528_528863


namespace problem_l528_528758

def f (x a : ℝ) : ℝ := (1 / 2) * x ^ 2 - 2 * a * x - a * Real.log x

theorem problem (a : ℝ) :
  (∀ x1 x2 : ℝ, (1 < x1 ∧ x1 < 2) ∧ (1 < x2 ∧ x2 < 2) ∧ (x1 ≠ x2) →
     (f x2 a - f x1 a) / (x2 - x1) < 0) ↔ a ≥ 4 / 5 := by
  sorry

end problem_l528_528758


namespace sqrt_x_y_eq_sqrt3_l528_528305

theorem sqrt_x_y_eq_sqrt3 (x y : ℝ) (h : sqrt (x - 2) + (y + 1)^2 = 0) :
  sqrt (x - y) = sqrt 3 ∨ sqrt (x - y) = - sqrt 3 :=
by
  sorry

end sqrt_x_y_eq_sqrt3_l528_528305


namespace f_eq_n_l528_528730

def f (n : ℕ) : ℕ → ℕ
| 1 := 1
| x := if x % 2 = 0 then 2 else 3

axiom f_conditions : ∀ (n : ℕ) (x y : ℕ),
  (f 2 = 2) ∧
  (f (x * y) = f x * f y) ∧
  (f n > 0) ∧
  (x > y → f x > f y)

theorem f_eq_n (n : ℕ) (h_n : 0 < n) : f n = n :=
by
  sorry

end f_eq_n_l528_528730


namespace max_real_part_max_imag_part_l528_528741

noncomputable def z1 (θ : ℝ) : ℂ := θ.cos - ℂ.I
noncomputable def z2 (θ : ℝ) : ℂ := θ.sin + ℂ.I

theorem max_real_part (θ : ℝ) : ℜ(z1 θ * z2 θ) ≤ 3 / 2 := by
  sorry

theorem max_imag_part (θ : ℝ) : ℑ(z1 θ * z2 θ) ≤ Real.sqrt 2 := by
  sorry

end max_real_part_max_imag_part_l528_528741


namespace sum_S19_is_190_l528_528871

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∀ n m, a n + a m = a (n+1) + a (m-1)

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

-- Main theorem
theorem sum_S19_is_190 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_def : sum_of_first_n_terms a S)
  (h_condition : a 6 + a 14 = 20) :
  S 19 = 190 :=
sorry

end sum_S19_is_190_l528_528871


namespace set_membership_l528_528055

def a : ℝ := Real.sqrt 3

def M : Set ℝ := {x | x ≤ 3}

theorem set_membership : {a} ⊆ M := sorry

end set_membership_l528_528055


namespace intersection_empty_l528_528324

open Set

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := by
  sorry

end intersection_empty_l528_528324


namespace peter_total_pizza_eaten_l528_528448

def slices_total : Nat := 16
def peter_slices_eaten_alone : ℚ := 2 / 16
def shared_slice_total : ℚ := 1 / (3 * 16)

theorem peter_total_pizza_eaten : peter_slices_eaten_alone + shared_slice_total = 7 / 48 := by
  sorry

end peter_total_pizza_eaten_l528_528448


namespace geom_sequence_S4_l528_528751

noncomputable def geom_sum (a1 q : ℤ) (n : ℕ) : ℤ := a1 * (1 - q^n) / (1 - q)

theorem geom_sequence_S4 (a1 a2 a3 : ℤ) (q : ℤ) (S4 : ℤ) 
  (h1 : a1 - a2 = 2) (h2 : a2 - a3 = 6) : 
  S4 = geom_sum a1 q 4 :=
begin
  -- sorry gives a placeholder for the proof
  sorry,
end

end geom_sequence_S4_l528_528751


namespace find_angle_ACB_l528_528927

-- Define the problem setup
variables {A B C D : Type} [triangle : Triangle A B C]

-- Conditions
axiom isosceles : AB = BC
axiom angle_bisector : ∃ D, bisects_angle ∠CAB D ∧ D ∈ segment B C

-- Angles in degrees for convenience
noncomputable def angle_A : ℝ := measure_angle A B C
noncomputable def angle_B : ℝ := measure_angle B C A
noncomputable def angle_C : ℝ := measure_angle C A B
noncomputable def angle_D : ℝ := measure_angle A B D

-- Auxiliary angles
noncomputable def angle_BAD : ℝ := angle_A / 2
noncomputable def angle_DBC : ℝ := angle_B / 2
noncomputable def angle_BDA : ℝ := angle_B - angle_DBC

-- Difference constraint
axiom angle_diff : abs (angle_B - angle_BAD) = 40 ∨ abs (angle_BAD - angle_BDA) = 40

-- Theorem statement
theorem find_angle_ACB:
  angle_C = 20 ∨ angle_C = 40 ∨ angle_C = 68 ∨ angle_C = 4 :=
sorry

end find_angle_ACB_l528_528927


namespace diversity_friends_goal_l528_528395

theorem diversity_friends_goal (n : Nat) (h₁ : n = 10000)
    (condition : ∀ (x y z : Nat), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
                  (f x y = friend ∨ f x y = enemy) ∧ 
                  (f y z = friend ∨ f y z = enemy) ∧ 
                  (f z x = friend ∨ f z x = enemy)) :
    ∃ t : Nat, t = 5000 ∧
    (∀ i : Nat, i ≤ t → 
    (∀ (x y : Nat), x < n ∧ y < n → f x y = friend)) :=
by
    sorry

end diversity_friends_goal_l528_528395


namespace distance_after_3_minutes_l528_528635

-- Define the given speeds and time interval
def speed_truck : ℝ := 65 -- in km/h
def speed_car : ℝ := 85 -- in km/h
def time_minutes : ℝ := 3 -- in minutes

-- The equivalent time in hours
def time_hours : ℝ := time_minutes / 60

-- Calculate the distances travelled by the truck and the car
def distance_truck : ℝ := speed_truck * time_hours
def distance_car : ℝ := speed_car * time_hours

-- Define the distance between the truck and the car
def distance_between : ℝ := distance_car - distance_truck

-- Theorem: The distance between the truck and car after 3 minutes is 1 km.
theorem distance_after_3_minutes : distance_between = 1 := by
  sorry

end distance_after_3_minutes_l528_528635


namespace problem_solution_exists_l528_528705

theorem problem_solution_exists (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)]
  (h : a > 0 ∧ b > 0 ∧ n > 0 ∧ a ^ 2013 + b ^ 2013 = p ^ n) :
  ∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ n = 2013 * k + 1 ∧ p = 2 := by
  sorry

end problem_solution_exists_l528_528705


namespace quadrilateral_count_l528_528442

-- Define the number of points
def num_points := 9

-- Define the number of vertices in a quadrilateral
def vertices_in_quadrilateral := 4

-- Use a combination function to find the number of ways to choose 4 points out of 9
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem that asserts the number of quadrilaterals that can be formed
theorem quadrilateral_count : combination num_points vertices_in_quadrilateral = 126 :=
by
  -- The proof would go here
  sorry

end quadrilateral_count_l528_528442


namespace find_x1_l528_528424

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

noncomputable def x (x1 : ℝ) (n : ℕ) : ℝ :=
match n with
| 0     := 0
| 1     := 1
| 2     := (fib 1 + fib 0 * x1) / (fib 2 + fib 1 * x1)
| (n+2) := (fib (n+1) + fib n * x1) / (fib (n+2) + fib (n+1) * x1)

theorem find_x1 (x1 : ℝ) (hx : x 2004 x1 = 1 / x1 - 1) :
  x1 = (-1 + Real.sqrt 5) / 2 ∨ x1 = (-1 - Real.sqrt 5) / 2 :=
sorry

end find_x1_l528_528424


namespace smallest_positive_solution_l528_528168

theorem smallest_positive_solution :
  ∃ x : ℝ, x > 0 ∧ (x ^ 4 - 50 * x ^ 2 + 576 = 0) ∧ (∀ y : ℝ, y > 0 ∧ y ^ 4 - 50 * y ^ 2 + 576 = 0 → x ≤ y) ∧ x = 3 * Real.sqrt 2 :=
sorry

end smallest_positive_solution_l528_528168


namespace circle_radius_l528_528967

theorem circle_radius (d : ℝ) (h : d = 10) : d / 2 = 5 :=
by
  sorry

end circle_radius_l528_528967


namespace parabolic_arch_ensures_truck_passes_l528_528224

noncomputable def parabolic_arch_min_width (h t_height t_width : ℝ) : ℝ :=
  let p := ((h - sqrt(h^2 - 4 * t_width)) / 2)
  2 * abs(p)

theorem parabolic_arch_ensures_truck_passes : 
  ∀ (h t_height t_width : ℝ), h = -6.21 ∧ t_height = 3 ∧ t_width = 1.6 →
  parabolic_arch_min_width h t_height t_width ≈ 12.21 :=
by
  intros h t_height t_width H,
  have : parabolic_arch_min_width h t_height t_width = (12.21 : ℝ),
  by sorry
  exact this

end parabolic_arch_ensures_truck_passes_l528_528224


namespace corner_square_win_corner_square_win_odd_n_adjacent_square_win_l528_528213

theorem corner_square_win (n : ℕ) (even_n : n % 2 = 0) : 
  ∃ strategy : strategy_type, strategy_wins_for strategy "first" :=
sorry

theorem corner_square_win_odd_n (n : ℕ) (odd_n : n % 2 = 1) :
  ∃ strategy : strategy_type, strategy_wins_for strategy "second" :=
sorry

theorem adjacent_square_win (n : ℕ) (p : position) (adjacent_to_corner : is_adjacent_to_corner p) :
  ∃ strategy : strategy_type, strategy_wins_for strategy "first" :=
sorry

end corner_square_win_corner_square_win_odd_n_adjacent_square_win_l528_528213


namespace rectangle_of_conditions_l528_528028

variables {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables [add_comm_group A] [add_comm_group B] [add_comm_group C] [add_comm_group D]

def is_parallelogram (A B C D : Point) : Prop :=
  -- Definition of parallelogram based on vector equality and parallelism
  (∃ (v : Vector), v = (B - A) ∧ v = (C - D))

def equal_diagonals (A B C D : Point) : Prop :=
  -- Definition for equal diagonals |AC| = |BD|
  dist A C = dist B D

theorem rectangle_of_conditions (A B C D : Point)
  (h_parallelogram : is_parallelogram A B C D)
  (h_equal_diagonals : equal_diagonals A B C D) :
  is_rectangle A B C D :=
by
  -- Proof required to establish as a rectangle given conditions
  sorry

end rectangle_of_conditions_l528_528028


namespace find_x_l528_528769

variable (x : ℝ)
variable h₀ : x > 0

def vec_a := (8, (1 / 2) * x)
def vec_b := (x, 1)

-- Condition: (vec_a - 2 * vec_b) parallel (2 * vec_a + vec_b)
def parallel_condition (a1 a2 b1 b2 : ℝ) := a1 * b2 = a2 * b1

theorem find_x (h_par : parallel_condition (8 - 2 * x) ((1 / 2) * x - 2) (16 + x) (x + 1)) : x = 4 :=
sorry

end find_x_l528_528769


namespace diagonal_lengths_of_quadrilateral_l528_528139

theorem diagonal_lengths_of_quadrilateral (A B C D : Type) 
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]
  (α : ℝ) (β : ℝ) (γ : ℝ) 
  (AD BC : ℝ)
  (hα : α = 78) (hβ : β = 120) (hγ : γ = 102)
  (hAD : AD = 110) (hBC : BC = 69) :
  ∃ AC BD : ℝ, AC ≈ 79.54 ∧ BD ≈ 109.20 :=
by
  -- proof is to be provided
  sorry

end diagonal_lengths_of_quadrilateral_l528_528139


namespace derivative_and_tangent_line_l528_528761

noncomputable def f (x : ℝ) : ℝ := x^2 + x * Real.log x

theorem derivative_and_tangent_line :
  (deriv f 1 = 3) ∧ (tangent_line_at f 1 = λ x, 3 * x - 2) := 
by
  -- Definition of the function.
  have f_def : ∀ x, f x = x^2 + x * Real.log x := by intro x; rw [f]
  -- Differentiability proof.
  have f_deriv : ∀ x, deriv f x = 2 * x + Real.log x + 1 := 
  by 
    intro x
    calc
      deriv (λ x, x^2 + x * Real.log x) x
          = deriv (λ x, x^2) x + deriv (λ x, x * Real.log x) x := sorry
      ... = 2 * x + (Real.log x + 1) := sorry
      ... = 2 * x + Real.log x + 1 := sorry
  -- Tangent line equation proof at x = 1
  have tangent_eq : tangent_line_at f 1 = λ x, 3 * x - 2 := 
    by 
      calc
        tangent_line_at f 1
          = (deriv f 1) * (λ x, x - 1) + f 1 := sorry
        ... = 3 * (λ x, x - 1) + 1 := sorry
        ... = 3 * x - 2 := sorry

  -- Combine the results into one statement.
  exact ⟨(f_deriv 1).symm, tangent_eq.symm⟩

end derivative_and_tangent_line_l528_528761


namespace total_books_l528_528578

theorem total_books (shelves_mystery shelves_picture : ℕ) (books_per_shelf : ℕ) 
    (h_mystery : shelves_mystery = 5) (h_picture : shelves_picture = 4) (h_books_per_shelf : books_per_shelf = 6) : 
    shelves_mystery * books_per_shelf + shelves_picture * books_per_shelf = 54 := 
by 
  sorry

end total_books_l528_528578


namespace hexagon_area_l528_528647

theorem hexagon_area (s t : ℝ) (hs : 3 * s = 6 * t)
  (ht : (s^2 * real.sqrt 3) / 4 = 4) : 6 * (t^2 * real.sqrt 3) / 4 = 6 := by
sorry

end hexagon_area_l528_528647


namespace flu_infection_l528_528607

theorem flu_infection :
  ∃ x : ℕ, 
  (1 + x + x * (1 + x) = 81) ∧ 
  (1 + x * (1 + x) = 8) ∧ 
  (81 * x + 81 = 729) :=
by {
  -- We declare the two conditions
  let x := 8,
  
  -- Verifying the first condition: after two rounds of infection
  have h1 : 1 + x + x * (1 + x) = 81, by {
    calc 
      1 + x + x * (1 + x)
        = 1 + 8 + 8 * (1 + 8) : by rfl
    ... = 1 + 8 + 8 * 9 : by rfl
    ... = 1 + 8 + 72 : by rfl
    ... = 81 : by rfl,
  }, 
  
  -- Verifying the second condition: infection rate x=8 does not directly add but verify next continuing condition
  have h2 : 1 + x * (1 + x) = 8, by {
    calc
      x
      = 8 : by rfl,
  },

  -- Verifying the third condition: after three rounds of infection
  have h3 : 81 * x + 81 = 729, by {
    calc
      81 * x + 81
        = 81 * 8 + 81 : by rfl
    ... = 648 + 81 : by rfl
    ... = 729 : by rfl,
  },
  
  -- Combining all conditions,
  exact ⟨x, h1, h2, h3⟩,
}

end flu_infection_l528_528607


namespace factorial_division_example_l528_528666

theorem factorial_division_example : (11.factorial / (7.factorial * 4.factorial) = 330) := by
  sorry

end factorial_division_example_l528_528666


namespace count_nonneg_real_values_l528_528720

theorem count_nonneg_real_values (x : ℝ) (hx : 0 ≤ x)
  (hy : ∃ y : ℕ, (y : ℝ) = sqrt (196 - real.cbrt x)) :
  ∃ n : ℕ, n = 15 :=
by
  sorry

end count_nonneg_real_values_l528_528720


namespace quadratic_condition_l528_528377

theorem quadratic_condition 
  (m : ℝ) 
  (h_quad : ∀ x : ℝ, let y := x^2 + 2 * (Real.sqrt 5) * x + 3 * m - 1 
                       in (y > 0) ∨ (y = 0 ∧ x ≠ 0)) :
  (1 / 3) ≤ m ∧ m < 2 := 
sorry

end quadratic_condition_l528_528377


namespace third_vertex_x_coordinate_l528_528648

-- Define the vertices of the equilateral triangle
def A : ℝ × ℝ := (5, 0)
def B : ℝ × ℝ := (5, 8)

-- Define the conditions
def is_first_quadrant (x : ℝ) (y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equilateral_triangle (A B C : ℝ × ℝ) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

-- Define the solution
theorem third_vertex_x_coordinate (C : ℝ × ℝ) 
  (h1 : equilateral_triangle A B C) 
  (h2 : C.2 > 0) 
  (h3 : is_first_quadrant C.1 C.2) : 
  C.1 = 5 + 4 * Real.sqrt 3 :=
begin
  sorry
end

end third_vertex_x_coordinate_l528_528648


namespace moli_initial_payment_l528_528077

variable (R C S M : ℕ)

-- Conditions
def condition1 : Prop := 3 * R + 7 * C + 1 * S = M
def condition2 : Prop := 4 * R + 10 * C + 1 * S = 164
def condition3 : Prop := 1 * R + 1 * C + 1 * S = 32

theorem moli_initial_payment : condition1 R C S M ∧ condition2 R C S ∧ condition3 R C S → M = 120 := by
  sorry

end moli_initial_payment_l528_528077


namespace find_derivative_value_l528_528336

noncomputable def f (x : ℝ) := x^2 + 3 * x * f' 3

theorem find_derivative_value (f' : ℝ → ℝ) (hx : ∀ x, f x = x^2 + 3 * x * f' 3) :
  f' 3 = -3 :=
sorry

end find_derivative_value_l528_528336


namespace product_of_converted_numbers_l528_528282

def binary_to_nat (b : Nat) : Nat :=
  match b with
  | 0     => 0
  | 1     => 1
  | n     => 2 * binary_to_nat (n / 10) + (n % 10)

def ternary_to_nat (t : Nat) : Nat :=
  match t with
  | 0     => 0
  | n     => 3 * ternary_to_nat (n / 10) + (n % 10)

theorem product_of_converted_numbers :
  binary_to_nat 1110 * ternary_to_nat 102 = 154 :=
by
  sorry

end product_of_converted_numbers_l528_528282


namespace estimate_pi_l528_528091

theorem estimate_pi (pairs : Fin 120 → ℝ × ℝ)
  (h_pairs : ∀ i, (pairs i).fst ≥ 0 ∧ (pairs i).fst < 1 ∧ (pairs i).snd ≥ 0 ∧ (pairs i).snd < 1)
  (count_m : ℕ)
  (h_count_m : count_m = (Finset.univ.filter (λ i, (pairs i).fst ^ 2 + (pairs i).snd ^ 2 < 1 ∧
                                (pairs i).fst + (pairs i).snd > 1)).card)
  (h_count_value : count_m = 34) :
  Real.pi = 47 / 15 := 
by
  sorry

end estimate_pi_l528_528091


namespace harold_catches_up_at_12_miles_l528_528571

noncomputable def adrienne_speed : ℕ := 3
noncomputable def harold_speed : ℕ := 4
noncomputable def head_start_time : ℕ := 1 -- In hours

theorem harold_catches_up_at_12_miles :
  let T := 3 in 
  let adrienne_distance := adrienne_speed * (T + head_start_time) in
  let harold_distance := harold_speed * T in
  harold_distance = adrienne_distance → harold_distance = 12 :=
by
  intros
  let T := 3
  let adrienne_distance := adrienne_speed * (T + head_start_time)
  let harold_distance := harold_speed * T
  simp [adrienne_speed, harold_speed, head_start_time]
  apply eq.trans (by simp) (by simp)
  sorry

end harold_catches_up_at_12_miles_l528_528571


namespace restore_numbers_l528_528106

noncomputable def arithmetic_sequence (a d : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := arithmetic_sequence a d n + d

theorem restore_numbers :
  ∃ (T E L K A S : ℕ),
    T = 5 ∧
    (10 * E + L) = 12 ∧
    (10 * E + K) = 19 ∧
    (10 * L + A) = 26 ∧
    (11 * S) = 33 ∧
    ∀ (n : ℕ), arithmetic_sequence T 7 n = 5 + 7 * n :=
begin
  sorry
end

end restore_numbers_l528_528106


namespace multiplication_even_a_b_multiplication_even_a_a_l528_528818

def a : Int := 4
def b : Int := 3

theorem multiplication_even_a_b : a * b = 12 := by sorry
theorem multiplication_even_a_a : a * a = 16 := by sorry

end multiplication_even_a_b_multiplication_even_a_a_l528_528818


namespace desired_cost_per_pound_l528_528988

/-- 
Let $p_1 = 8$, $w_1 = 25$, $p_2 = 5$, and $w_2 = 50$ represent the prices and weights of two types of candies.
Calculate the desired cost per pound $p_m$ of the mixture.
-/
theorem desired_cost_per_pound 
  (p1 : ℝ) (w1 : ℝ) (p2 : ℝ) (w2 : ℝ) (p_m : ℝ) 
  (h1 : p1 = 8) (h2 : w1 = 25) (h3 : p2 = 5) (h4 : w2 = 50) :
  p_m = (p1 * w1 + p2 * w2) / (w1 + w2) → p_m = 6 :=
by 
  intros
  sorry

end desired_cost_per_pound_l528_528988


namespace ellipse_equation_lambda_plus_mu_constant_l528_528030

noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0): Set (ℝ × ℝ) :=
  {p | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1}

theorem ellipse_equation (a b c : ℝ) (ha : a = 2) (hb : b = 1) (hc : c = sqrt 3) :
  ellipse a b (by {split; linarith}) = {p | (p.1^2) / 4 + p.2^2 = 1} :=
sorry

theorem lambda_plus_mu_constant
  (a b c : ℝ) (ha : a = 2) (hb : b = 1) (hc : c = sqrt 3)
  (Q M N E : ℝ × ℝ) (l : Set (ℝ × ℝ))
  (hQ : Q = (-1,0)) (hE : E = (-4, λ x, ¬ x < 0)) -- This part needs clarification
  (hMQ_lam_QN : MQ = λ N, λ N, by {sorry}) -- Not sure what exactly is meant here
  (hME_mu_NE : ME = μ N, by {sorry}) -- Not sure what exactly is meant here
  : let λ := (MQ = λ Q) in let μ := (ME = μ N) in λ + μ = 0 :=
sorry

end ellipse_equation_lambda_plus_mu_constant_l528_528030


namespace increasing_function_inequality_l528_528745

variable {f : ℝ → ℝ}  -- f is a function from reals to reals

theorem increasing_function_inequality
  (h_inc : ∀ x y : ℝ, x < y → f x < f y)  -- condition 1: f is increasing
  (a b : ℝ)  -- condition 2: a and b are real numbers
  (hab : a + b > 0)  -- condition 3: a + b > 0
  : f(a) + f(b) > f(-a) + f(-b) := sorry  -- prove the desired inequality and skip the proof

end increasing_function_inequality_l528_528745


namespace apples_jackie_l528_528823

theorem apples_jackie (A : ℕ) (J : ℕ) (h1 : A = 8) (h2 : J = A + 2) : J = 10 := by
  -- Adam has 8 apples
  sorry

end apples_jackie_l528_528823


namespace alpha_value_l528_528891

-- Define the conditions in Lean
variables (α β γ k : ℝ)

-- Mathematically equivalent problem statements translated to Lean
theorem alpha_value :
  (∀ β γ, α = (k * γ) / β) → -- proportionality condition
  (α = 4) →
  (β = 27) →
  (γ = 3) →
  (∀ β γ, β = -81 → γ = 9 → α = -4) :=
by
  sorry

end alpha_value_l528_528891


namespace polar_to_rectangular_max_distance_MN_l528_528755

-- 1. Conversion to rectangular coordinates
theorem polar_to_rectangular (ρ θ : ℝ) : (ρ = 2 * cos θ) → (∀ x y : ℝ, x = ρ * cos θ ∧ y = ρ * sin θ → x^2 + y^2 - 2*x = 0) :=
by
  intros hpolar x y hxy
  sorry  -- use the polar coordinates transformations and their identities

-- 2. Maximum value of |MN|
theorem max_distance_MN :
  let M : ℝ × ℝ := (0, 2)
  let C : ℝ × ℝ := (1, 0)
  let radius : ℝ := 1
  let distance_MC := Real.sqrt (1^2 + 2^2)
  let max_MN : ℝ := distance_MC + radius
  (distance_MC = sqrt 5) →
  (max_MN = sqrt 5 + 1) :=
by
  intros hdist
  sorry

end polar_to_rectangular_max_distance_MN_l528_528755


namespace consecutive_integers_at_least_one_even_l528_528164

theorem consecutive_integers_at_least_one_even (a b c : ℤ) (h1 : b = a + 1) (h2 : c = b + 1) :
  (Even a ∨ Even b ∨ Even c) :=
by
  have h : ¬ (¬ Even a ∧ ¬ Even b ∧ ¬ Even c) := sorry
  by_contradiction h
  sorry

end consecutive_integers_at_least_one_even_l528_528164


namespace range_of_m_l528_528996

-- Definition of a decreasing function
def decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x > f y

-- Statement needs to incorporate all conditions and result
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (decreasing_f : decreasing_on f (set.Icc (-2) 2))
  (h : f (m - 1) < f (-m)) :
  (1 / 2 : ℝ) < m ∧ m ≤ 2 :=
sorry

end range_of_m_l528_528996


namespace matrix_equation_l528_528052

open Matrix

-- Define matrix B
def B : Matrix (Fin 2) (Fin 2) (ℤ) :=
  ![![1, -2], 
    ![-3, 5]]

-- The proof problem statement in Lean 4
theorem matrix_equation (r s : ℤ) (I : Matrix (Fin 2) (Fin 2) (ℤ))  [DecidableEq (ℤ)] [Fintype (Fin 2)] : 
  I = 1 ∧ B ^ 6 = r • B + s • I ↔ r = 2999 ∧ s = 2520 := by {
    sorry
}

end matrix_equation_l528_528052


namespace XiaoQian_wins_by_fourth_round_probability_l528_528581

-- Define the probability of each round's outcome
def prob_win : ℚ := 1 / 3
def prob_draw : ℚ := 1 / 3
def prob_lose : ℚ := 1 / 3

-- Define the probability calculation function
def prob_XiaoQian_wins_by_fourth_round : ℚ :=
  (3.choose 2) * (prob_win ^ 2) * (prob_lose) * (prob_win)

-- The theorem to prove
theorem XiaoQian_wins_by_fourth_round_probability :
  prob_XiaoQian_wins_by_fourth_round = 2 / 27 :=
by
  sorry

end XiaoQian_wins_by_fourth_round_probability_l528_528581


namespace fib_equation_l528_528257

noncomputable def Phi : ℝ := (1 + Real.sqrt 5) / 2

def fib : ℕ → ℝ
| 0       := 0
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

theorem fib_equation (n : ℕ) : 
  fib (n + 1) = Phi * fib n + (-1 / Phi) ^ n := 
sorry

end fib_equation_l528_528257


namespace calc_exponent_l528_528245

theorem calc_exponent (a b : ℕ) : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end calc_exponent_l528_528245


namespace problem_I_problem_II_l528_528148

-- Define the conditions
def students : Nat := 7

-- Problem (I) Setup
def arrangements_I : Nat :=
  3 * (6!)

-- Problem (II) Setup
def arrangements_II : Nat :=
  (6 * (6!)) - (5 * (5!))

-- Declare the statements
theorem problem_I (students_A_mid_or_sides : arrangements_I = 2160) : arrangements_I = 2160 := by
  sorry

theorem problem_II (students_A_not_left_B_not_right : arrangements_II = 3720) : arrangements_II = 3720 := by
  sorry

end problem_I_problem_II_l528_528148


namespace intersection_empty_l528_528325

open Set

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := by
  sorry

end intersection_empty_l528_528325


namespace num_teachers_at_King_High_School_l528_528834

-- Conditions
def num_students := 1500
def classes_per_student := 6
def classes_per_teacher := 3
def students_per_class := 35

-- Question and Proof
theorem num_teachers_at_King_High_School
  (H1 : num_students = 1500)
  (H2 : classes_per_student = 6)
  (H3 : classes_per_teacher = 3)
  (H4 : students_per_class = 35) :
  let total_classes := num_students * classes_per_student in
  let unique_classes := total_classes / students_per_class in
  let num_teachers := unique_classes / classes_per_teacher in
  num_teachers ≈ 86 := 
by
  sorry

end num_teachers_at_King_High_School_l528_528834


namespace circumference_of_circle_l528_528160

def speed_cyclist1 : ℝ := 7
def speed_cyclist2 : ℝ := 8
def meeting_time : ℝ := 42
def circumference : ℝ := 630

theorem circumference_of_circle :
  (speed_cyclist1 * meeting_time + speed_cyclist2 * meeting_time = circumference) :=
by
  sorry

end circumference_of_circle_l528_528160


namespace meeting_probability_l528_528980

def prob_meet (a b : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ |a - b| ≤ 1/3

theorem meeting_probability : 
  let P : ℝ := ∫ 0..1, ∫ 0..1, if prob_meet x y then 1 else 0 ∂y ∂x in
  P = 5/9 :=
sorry

end meeting_probability_l528_528980


namespace vector_geometry_problem_l528_528853

variable {V : Type*} [InnerProductSpace ℝ V]
variable (A B C P : V)

-- Define the conditions of the problem
def condition1 : Prop := dist A B = 1
def condition2 : Prop := (P - C) = (B - C) + 2 • (A - C)

-- Define the target value of the dot product
def target_value : ℝ := 3

-- The theorem we aim to prove
theorem vector_geometry_problem 
  (h1 : condition1 A B)
  (h2 : condition2 A B C P) :
  inner (P - A) (P - B) = target_value := 
sorry

end vector_geometry_problem_l528_528853


namespace house_painting_cost_l528_528616

def arithmetic_sequence_last_term (a d : ℕ) (n : ℕ) : ℕ := a + d * (n - 1)

def number_of_digits (n : ℕ) : ℕ := n.toString.length

def total_painting_cost (south_addresses north_addresses: List ℕ ): ℕ :=
  (south_addresses ++ north_addresses).sumBy number_of_digits

theorem house_painting_cost :
  let south_addresses := List.map (λ n, 5 + 7 * (n - 1)) (List.range 25).drop 1
  let north_addresses := List.map (λ n, 2 + 8 * (n - 1)) (List.range 25).drop 1
  total_painting_cost south_addresses north_addresses = 123 := by
  sorry

end house_painting_cost_l528_528616


namespace find_v1_v2_l528_528261

noncomputable def point_on_line_l (t : ℝ) : (ℝ × ℝ) :=
  (2 + 2 * t, 3 + t)

noncomputable def point_on_line_m (s : ℝ) : (ℝ × ℝ) :=
  (-3 + 2 * s, 7 + s)

theorem find_v1_v2 :
  ∃ (v1 v2 : ℝ), 
    (∀ t s : ℝ, let A := point_on_line_l t,
                     B := point_on_line_m s,
                     PA := (A.1 - B.1, A.2 - B.2) in 
    (v1, v2) = (1:ℝ) • (⟨0, 0⟩: ℝ × ℝ) → (v1 + v2 = 1) → (v1, v2) = (-1, 2)) :=
sorry

end find_v1_v2_l528_528261


namespace base_angle_of_isosceles_triangle_l528_528020

theorem base_angle_of_isosceles_triangle (vertex_angle : ℝ) (triangle_sum : ℝ) (isosceles : Bool) (h_vertex : vertex_angle = 70) (h_sum : triangle_sum = 180) (h_isosceles : isosceles = true) : 
  ∃ base_angle, base_angle = 55 :=
by
  -- Define the total angles sum.
  have h : 2 * 55 + 70 = 180 := by linarith [h_vertex, h_sum]
  use 55
  exact h
  sorry

end base_angle_of_isosceles_triangle_l528_528020


namespace arithmetic_seq_40th_term_l528_528558

theorem arithmetic_seq_40th_term (a₁ d : ℕ) (n : ℕ) (h1 : a₁ = 3) (h2 : d = 4) (h3 : n = 40) : 
  a₁ + (n - 1) * d = 159 :=
by
  sorry

end arithmetic_seq_40th_term_l528_528558


namespace largest_pies_without_ingredients_l528_528265

variable (total_pies : ℕ) (chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
variable (b : total_pies = 36)
variable (c : chocolate_pies = total_pies / 2)
variable (m : marshmallow_pies = 2 * total_pies / 3)
variable (k : cayenne_pies = 3 * total_pies / 4)
variable (s : soy_nut_pies = total_pies / 6)

theorem largest_pies_without_ingredients (total_pies chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
  (b : total_pies = 36)
  (c : chocolate_pies = total_pies / 2)
  (m : marshmallow_pies = 2 * total_pies / 3)
  (k : cayenne_pies = 3 * total_pies / 4)
  (s : soy_nut_pies = total_pies / 6) :
  9 = total_pies - chocolate_pies - marshmallow_pies - cayenne_pies - soy_nut_pies + 3 * cayenne_pies := 
by
  sorry

end largest_pies_without_ingredients_l528_528265


namespace calculation_is_one_l528_528249

noncomputable def calc_expression : ℝ :=
  (1/2)⁻¹ - (2021 + Real.pi)^0 + 4 * Real.sin (Real.pi / 3) - Real.sqrt 12

theorem calculation_is_one : calc_expression = 1 :=
by
  -- Each of the steps involved in calculating should match the problem's steps
  -- 1. (1/2)⁻¹ = 2
  -- 2. (2021 + π)^0 = 1
  -- 3. 4 * sin(π / 3) = 2√3 with sin(60°) = √3/2
  -- 4. sqrt(12) = 2√3
  -- Hence 2 - 1 + 2√3 - 2√3 = 1
  sorry

end calculation_is_one_l528_528249


namespace sum_of_m_for_common_root_l528_528950

theorem sum_of_m_for_common_root :
  let p1 := λ x : ℝ, x^2 - x - 6
  let p2 := λ x m : ℝ, x^2 - 9 * x + m
  (∃ x : ℝ, p1 x = 0 ∧ p2 x 18 = 0 ∨ p2 x (-22) = 0) ∧ 
  (18 + (-22) = -4) :=
sorry

end sum_of_m_for_common_root_l528_528950


namespace probability_diamond_then_ace_l528_528539

/-- Probability that the first card is a diamond and the second card is an ace from a standard deck of 52 cards -/
theorem probability_diamond_then_ace : 
  let prob := (1/52) * 
              (1/17) + 
              (3/13) * 
              (4/51) 
  in prob = 1/52 := by
  sorry

end probability_diamond_then_ace_l528_528539


namespace purchase_total_cost_l528_528068

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end purchase_total_cost_l528_528068


namespace limit_f_over_x_squared_l528_528858

theorem limit_f_over_x_squared (f : ℝ → ℝ) (a b c : ℝ)
  (hf : ∀ x y : ℝ, f (x - f y) ≥ f x + f (f y) - a * x * f y - b * f y - c)
  (h_range : ∀ x : ℝ, 0 < f x) :
  ∃ L : ℝ, L ≤ a / 2 ∧ (tendsto (λ x, f x / (x^2)) at_top (𝓝 L)) :=
by
  sorry

end limit_f_over_x_squared_l528_528858


namespace value_of_f1_l528_528057

variable (f : ℝ → ℝ)
open Function

theorem value_of_f1
  (h : ∀ x y : ℝ, f (f (x - y)) = f x * f y - f x + f y - 2 * x * y + 2 * x - 2 * y) :
  f 1 = -1 :=
sorry

end value_of_f1_l528_528057


namespace tangent_points_sum_constant_l528_528152

theorem tangent_points_sum_constant 
  (a : ℝ) (x1 y1 x2 y2 : ℝ)
  (hC1 : x1^2 = 4 * y1)
  (hC2 : x2^2 = 4 * y2)
  (hT1 : y1 - (-2) = (1/2)*x1*(x1 - a))
  (hT2 : y2 - (-2) = (1/2)*x2*(x2 - a)) :
  x1 * x2 + y1 * y2 = -4 :=
sorry

end tangent_points_sum_constant_l528_528152


namespace quadrilateral_with_perpendicular_diagonals_not_specific_l528_528376

-- Define a quadrilateral with the property that its diagonals are perpendicular.
structure Quadrilateral :=
(A B C D : Point)
(diagonal_perpendicular : ∃ (E : Point), right_angle (segment A C) (segment B D))

-- Define each specific type of quadrilateral
def Rhombus (quad : Quadrilateral) : Prop :=
  ∀ (A B C D : quad.Point), equal_sides quad.A quad.B quad.B quad.C quad.C quad.D quad.D quad.A

def Rectangle (quad : Quadrilateral) : Prop :=
  ∀ (A B C D : quad.Point), right_angle (segment quad.A quad.B) (segment quad.B quad.C)
  ∧ right_angle (segment quad.B quad.C) (segment quad.C quad.D)
  ∧ equal_length (segment quad.A quad.C) (segment quad.B quad.D)

def Square (quad : Quadrilateral) : Prop :=
  Rhombus quad ∧ Rectangle quad

def IsoscelesTrapezoid (quad : Quadrilateral) : Prop :=
  ∃ (A B C D : quad.Point), (parallel (segment quad.A quad.B) (segment quad.C quad.D)
  ∧ equal_length (segment quad.A quad.D) (segment quad.B quad.C))

-- The main proof statement: if a quadrilateral has perpendicular diagonals, it is not necessarily any of the specific types requested.
theorem quadrilateral_with_perpendicular_diagonals_not_specific (quad : Quadrilateral)
  (h: quad.diagonal_perpendicular) :
  ¬ (Rhombus quad ∨ Rectangle quad ∨ Square quad ∨ IsoscelesTrapezoid quad) := 
sorry

end quadrilateral_with_perpendicular_diagonals_not_specific_l528_528376


namespace systematic_sampling_10_people_in_interval_l528_528458

noncomputable def systematic_sampling_count 
  (total_people : ℕ) 
  (people_to_select : ℕ) 
  (first_number : ℕ) 
  (lower_bound : ℕ) 
  (upper_bound : ℕ) : ℕ :=
let group_size := total_people / people_to_select in
let common_difference := group_size in
let a_n := λ n : ℕ, first_number + common_difference * (n - 1) in
let count := Finset.card (Finset.filter
  (λ n, lower_bound ≤ a_n n ∧ a_n n ≤ upper_bound)
  (Finset.range (people_to_select + 1))) in
count

theorem systematic_sampling_10_people_in_interval :
  systematic_sampling_count 960 32 9 450 750 = 10 :=
by
  -- remaining proof to be completed
  sorry

end systematic_sampling_10_people_in_interval_l528_528458


namespace part1_l528_528985

def purchase_price (x y : ℕ) : Prop := 25 * x + 30 * y = 1500
def quantity_relation (x y : ℕ) : Prop := x = 2 * y - 4

theorem part1 (x y : ℕ) (h1 : purchase_price x y) (h2 : quantity_relation x y) : x = 36 ∧ y = 20 :=
sorry

end part1_l528_528985


namespace number_of_big_bonsai_sold_l528_528066

theorem number_of_big_bonsai_sold (cost_small cost_big small_bonsai_sold total_earnings : ℕ) : 
    cost_small = 30 ∧ 
    cost_big = 20 ∧ 
    small_bonsai_sold = 3 ∧ 
    total_earnings = 190 → 
    ∃ (x : ℕ), 3 * cost_small + x * cost_big = total_earnings ∧ x = 5 :=
by
  intro h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  have h5 : 3 * 30 + 5 * 20 = 190,
  { simp },
  existsi 5,
  split,
  { exact h5 },
  { refl }

end number_of_big_bonsai_sold_l528_528066


namespace ken_summit_time_l528_528460

variables (t : ℕ) (s : ℕ)

/--
Sari and Ken climb up a mountain. 
Ken climbs at a constant pace of 500 meters per hour,
and reaches the summit after \( t \) hours starting from 10:00.
Sari starts climbing 2 hours before Ken at 08:00 and is 50 meters behind Ken when he reaches the summit.
Sari is already 700 meters ahead of Ken when he starts climbing.
Prove that Ken reaches the summit at 15:00.
-/
theorem ken_summit_time (h1 : 500 * t = s * (t + 2) + 50)
  (h2 : s * 2 = 700) : t + 10 = 15 :=

sorry

end ken_summit_time_l528_528460


namespace min_students_blue_shirt_red_shoes_l528_528383

theorem min_students_blue_shirt_red_shoes
    (n : ℕ)
    (hn : n % 63 = 0)
    (h_blue_shirt : ∃ k : ℕ, (3 : ℤ) * n = 7 * k)
    (h_red_shoes : ∃ m : ℕ, (4 : ℤ) * n = 9 * m) :
    ∃ x : ℕ, x = 8 :=
by
  -- Place the proof here
  sorry

end min_students_blue_shirt_red_shoes_l528_528383


namespace correct_option_is_C_l528_528171

-- Define the polynomial expressions and their expected values as functions
def optionA (x : ℝ) : Prop := (x + 2) * (x - 5) = x^2 - 2 * x - 3
def optionB (x : ℝ) : Prop := (x + 3) * (x - 1 / 3) = x^2 + x - 1
def optionC (x : ℝ) : Prop := (x - 2 / 3) * (x + 1 / 2) = x^2 - 1 / 6 * x - 1 / 3
def optionD (x : ℝ) : Prop := (x - 2) * (-x - 2) = x^2 - 4

-- Problem Statement: Verify that the polynomial multiplication in Option C is correct
theorem correct_option_is_C (x : ℝ) : optionC x :=
by
  -- Statement indicating the proof goes here
  sorry

end correct_option_is_C_l528_528171


namespace range_of_m_l528_528258

noncomputable def setA : Set ℤ := {1, 2}

def quadratic_eq_solutions (m : ℝ) : Set ℝ :=
  {x | x ^ 2 - m * x + 2 = 0}

theorem range_of_m (m : ℝ) :
  B ⊆ setA →
  quadratic_eq_solutions m = ∅ ∨ quadratic_eq_solutions m = {1} ∨ quadratic_eq_solutions m = {2} ∨ quadratic_eq_solutions m = {1, 2} →
  (∃ x, x ∈ quadratic_eq_solutions m) ↔ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2) ∨ m = 3 :=
by
  sorry

end range_of_m_l528_528258


namespace total_wet_surface_area_of_cistern_l528_528600

def cistern_length := 7 -- in meters
def cistern_width := 4 -- in meters
def water_depth := 1 + 25 / 100 -- in meters (1 m 25 cm)

theorem total_wet_surface_area_of_cistern :
  let area_bottom := cistern_length * cistern_width,
      area_long_sides := 2 * (cistern_length * water_depth),
      area_short_sides := 2 * (cistern_width * water_depth),
      total_wet_surface_area := area_bottom + area_long_sides + area_short_sides
  in total_wet_surface_area = 83 :=
by
  let area_bottom := cistern_length * cistern_width
  let area_long_sides := 2 * (cistern_length * water_depth)
  let area_short_sides := 2 * (cistern_width * water_depth)
  let total_wet_surface_area := area_bottom + area_long_sides + area_short_sides
  show total_wet_surface_area = 83
  sorry

end total_wet_surface_area_of_cistern_l528_528600


namespace zachary_seventh_day_cans_l528_528959

-- Define the number of cans found by Zachary every day.
def cans_found_on (day : ℕ) : ℕ :=
  if day = 1 then 4
  else if day = 2 then 9
  else if day = 3 then 14
  else 5 * (day - 1) - 1

-- The theorem to prove the number of cans found on the seventh day.
theorem zachary_seventh_day_cans : cans_found_on 7 = 34 :=
by 
  sorry

end zachary_seventh_day_cans_l528_528959


namespace coeff_x2_expansion_l528_528333

noncomputable def n : ℕ := ∫ (x : ℝ) in 0..3, (2 * x - 1)

theorem coeff_x2_expansion :
  let n := ∫ (x : ℝ) in 0..3, (2 * x - 1)
  in (coeff_x2 (3 / (sqrt x) - x^(1/3))^n) = 1 :=
by
  sorry

end coeff_x2_expansion_l528_528333


namespace bailey_misses_percentage_l528_528655

theorem bailey_misses_percentage (total_shots scored_shots : ℕ) (h1 : total_shots = 8) (h2 : scored_shots = 6) : 
  (total_shots - scored_shots : ℚ) / total_shots * 100 = 25 :=
by
  simp [h1, h2]
  norm_num
  sorry

end bailey_misses_percentage_l528_528655


namespace radius_of_inscribed_circles_l528_528384

-- Given definitions
variable {R : ℝ} (inscribed : (ℝ → ℝ → Prop)) (tangent : (ℝ → ℝ → Prop))

-- Conditions
def large_circle_radius : Prop := R = 3
def six_smaller_circles (r : ℝ) : Prop := inscribed r 3 ∧ (∀ n : ℕ, (1 ≤ n ∧ n ≤ 6) → tangent r r ∧ tangent r 3)

-- Theorem to prove
theorem radius_of_inscribed_circles (r : ℝ) (H1 : large_circle_radius) (H2 : six_smaller_circles r) : r = 1 :=
  sorry

end radius_of_inscribed_circles_l528_528384


namespace polygon_sides_l528_528555

theorem polygon_sides (n : ℕ) (h1 : n ≥ 3) 
 (h2 : 0.3333333333333333 = (2 / (n - 3))) : n = 9 := 
begin
  sorry
end

end polygon_sides_l528_528555


namespace car_can_travel_l528_528078

-- Definitions for clarity
def cars : Type := ℕ  -- We'll use a natural number to denote cars for simplicity
def circular_road (n : ℕ) := fin n  -- Represent the circular road with finite cars

-- Problem Statement
theorem car_can_travel (n : ℕ) (h : ∀ c : circular_road n, (∀ g: cars, g = n) → 
  ∃ c' : circular_road n, (c' = c ∧ g = n)) : ∃ c : circular_road n, 
  ∀ (next_c : circular_road n) (tank : cars), tank ≥ 0 := 
sorry

end car_can_travel_l528_528078


namespace distance_between_girls_after_12_hours_l528_528161

def dist_1st_girl := (7 * 6) + (10 * 6)
def dist_2nd_girl := (3 * 8) + (2 * 4)
def total_distance := dist_1st_girl + dist_2nd_girl

theorem distance_between_girls_after_12_hours : total_distance = 134 :=
by
  unfold dist_1st_girl
  unfold dist_2nd_girl
  unfold total_distance
  sorry

end distance_between_girls_after_12_hours_l528_528161


namespace rabbits_total_distance_l528_528146

theorem rabbits_total_distance :
  let white_speed := 15
  let brown_speed := 12
  let grey_speed := 18
  let black_speed := 10
  let time := 7
  let white_distance := white_speed * time
  let brown_distance := brown_speed * time
  let grey_distance := grey_speed * time
  let black_distance := black_speed * time
  let total_distance := white_distance + brown_distance + grey_distance + black_distance
  total_distance = 385 :=
by
  sorry

end rabbits_total_distance_l528_528146


namespace line_through_A_parallel_line_through_B_perpendicular_l528_528195

-- 1. Prove the equation of the line passing through point A(2, 1) and parallel to the line 2x + y - 10 = 0 is 2x + y - 5 = 0.
theorem line_through_A_parallel :
  ∃ (l : ℝ → ℝ), (∀ x, 2 * x + l x - 5 = 0) ∧ (l 2 = 1) ∧ (∃ k, ∀ x, l x = -2 * (x - 2) + k) :=
sorry

-- 2. Prove the equation of the line passing through point B(3, 2) and perpendicular to the line 4x + 5y - 8 = 0 is 5x - 4y - 7 = 0.
theorem line_through_B_perpendicular :
  ∃ (m : ℝ) (l : ℝ → ℝ), (∀ x, 5 * x - 4 * l x - 7 = 0) ∧ (l 3 = 2) ∧ (m = -7) :=
sorry

end line_through_A_parallel_line_through_B_perpendicular_l528_528195


namespace cos_double_angle_l528_528778

theorem cos_double_angle (α : ℝ) (hα1 : tan α - 1 / tan α = 3 / 2) (hα2 : α ∈ Ioo (π / 4) (π / 2)) :
  cos (2 * α) = -3 / 5 := 
sorry

end cos_double_angle_l528_528778


namespace problem_statement_l528_528095

variable (q p : ℚ)
#check λ (q p : ℚ), q / p

theorem problem_statement :
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  q / p = -(119 : ℚ) / 3 :=
by
  let q := (119 : ℚ) / 8
  let p := -(3 : ℚ) / 8
  sorry

end problem_statement_l528_528095


namespace largest_common_term_in_range_l528_528646

def seq1 (n : ℕ) : ℕ := 5 + 9 * n
def seq2 (m : ℕ) : ℕ := 3 + 8 * m

theorem largest_common_term_in_range :
  ∃ (a : ℕ) (n m : ℕ), seq1 n = a ∧ seq2 m = a ∧ 1 ≤ a ∧ a ≤ 200 ∧ (∀ b, (∃ nf mf, seq1 nf = b ∧ seq2 mf = b ∧ 1 ≤ b ∧ b ≤ 200) → b ≤ a) :=
sorry

end largest_common_term_in_range_l528_528646


namespace milk_for_18_cookies_l528_528153

def milk_needed_to_bake_cookies (cookies : ℕ) (milk_per_24_cookies : ℚ) (quarts_to_pints : ℚ) : ℚ :=
  (milk_per_24_cookies * quarts_to_pints) * (cookies / 24)

theorem milk_for_18_cookies :
  milk_needed_to_bake_cookies 18 4.5 2 = 6.75 :=
by
  sorry

end milk_for_18_cookies_l528_528153


namespace minimum_value_l528_528718

-- Define geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * ((a 2 / a 1) ^ n)

-- Define the condition for positive geometric sequence
def positive_geometric_sequence (a : ℕ → ℝ) :=
  is_geometric_sequence a ∧ ∀ n : ℕ, a n > 0

-- Condition given in the problem
def condition (a : ℕ → ℝ) :=
  2 * a 4 + a 3 = 2 * a 2 + a 1 + 8

-- Define the problem statement to be proved
theorem minimum_value (a : ℕ → ℝ) (h1 : positive_geometric_sequence a) (h2 : condition a) :
  2 * a 6 + a 5 = 32 :=
sorry

end minimum_value_l528_528718


namespace sum_binom_eq_real_part_l528_528252

theorem sum_binom_eq_real_part :
  (1 / 2 ^ 1986) * ∑ n in Finset.range 994, (-3) ^ n * Nat.choose 1986 (2 * n) = -1 / 2 := 
sorry

end sum_binom_eq_real_part_l528_528252


namespace find_A_max_min_l528_528235

theorem find_A_max_min :
  ∃ (A_max A_min : ℕ), 
    (A_max = 99999998 ∧ A_min = 17777779) ∧
    (∀ B A, 
      (B > 77777777) ∧
      (Nat.coprime B 36) ∧
      (A = (B % 10) * 10000000 + B / 10) →
      (A ≤ 99999998 ∧ A ≥ 17777779)) :=
by 
  existsi 99999998
  existsi 17777779
  split
  { 
    split 
    { 
      refl 
    }
    refl 
  }
  intros B A h
  sorry

end find_A_max_min_l528_528235


namespace Jim_reads_pages_per_week_l528_528828

theorem Jim_reads_pages_per_week :
  ∀ (initial_rate hours_less initial_pages new_rate : ℕ),
    initial_rate = 40 →
    hours_less = 4 →
    initial_pages = 600 →
    new_rate = 3 / 2 * initial_rate →
    (initial_pages / initial_rate - hours_less) * new_rate = 660 :=
by 
  intros initial_rate hours_less initial_pages new_rate h1 h2 h3 h4
  have h5 : initial_pages / initial_rate = 15 := by sorry
  have h6 : initial_pages / initial_rate - hours_less = 11 := by sorry
  have h7 : new_rate = 60 := by sorry
  exact Eq.trans (mul_eq_mul_right _ _ _) h7
  sorry

end Jim_reads_pages_per_week_l528_528828


namespace fixed_point_of_quadratic_l528_528734

theorem fixed_point_of_quadratic (m : ℝ) :
  ∃ A : ℝ × ℝ, A = (-1, 0) ∧ ∀ (m : ℝ), let y := -((A.1) ^ 2) + (m - 1) * (A.1) + m in y = A.2 :=
sorry

end fixed_point_of_quadratic_l528_528734


namespace smallest_yummy_integer_l528_528288

theorem smallest_yummy_integer :
  ∃ B : ℤ, (∀ k : ℤ, B ≤ k → k ≤ B + 2001) → (∀ there_exists_intervals : List ℤ, 
  (there_exists_intervals.sum = 2002) → there_exists_intervals.head = B) → (B = -1001)
:= sorry

end smallest_yummy_integer_l528_528288


namespace inscribed_cube_volume_l528_528613

noncomputable def volume_of_inscribed_cube : ℝ :=
  let cube_edge_length := 12
  let cylinder_diameter := cube_edge_length
  let cylinder_radius := cylinder_diameter / 2
  let face_diagonal := cylinder_diameter
  let s := face_diagonal / Real.sqrt 2
  (s^3)

theorem inscribed_cube_volume :
  let cube_edge_length := 12
  let cylinder_diameter := cube_edge_length
  let cylinder_radius := cylinder_diameter / 2
  let face_diagonal := cylinder_diameter
  let s := face_diagonal / Real.sqrt 2 in
  (s^3) = 432 * Real.sqrt 2 := 
by
  -- proof goes here
  sorry

end inscribed_cube_volume_l528_528613


namespace max_divisors_with_remainder_10_l528_528933

theorem max_divisors_with_remainder_10 (m : ℕ) :
  (m > 0) → (∀ k, (2008 % k = 10) ↔ k < m) → m = 11 :=
by
  sorry

end max_divisors_with_remainder_10_l528_528933


namespace age_of_hospital_l528_528021

theorem age_of_hospital (grant_current_age : ℕ) (future_ratio : ℚ)
                        (grant_future_age : grant_current_age + 5 = 30)
                        (hospital_age_ratio : future_ratio = 2 / 3) :
                        (grant_current_age = 25) → 
                        (grant_current_age + 5 = future_ratio * (grant_current_age + 5 + 5)) →
                        (grant_current_age + 5 + 5 - 5 = 40) :=
by
  sorry

end age_of_hospital_l528_528021


namespace find_x_l528_528691

theorem find_x :
  ∃ x y z w : ℕ, 
    x = y + 10 ∧
    y = z + 15 ∧
    z = w + 25 ∧
    w = 95 ∧
    x = 145 :=
by
  use 145, 135, 120, 95
  repeat { split }
  any_goals { refl }
  sorry

end find_x_l528_528691


namespace num_four_digit_numbers_l528_528364

theorem num_four_digit_numbers (d1 d2 : Nat) (h1 : d1 = 3) (h2 : d2 = 8) : 
  (Finset.univ.filter (λ n, (n % 10 = d1) ∨ (n % 10 = d2)).card = 6) :=
sorry

end num_four_digit_numbers_l528_528364


namespace room_height_l528_528123

/--
Given:
1. The dimensions of a room are 25 feet * 15 feet * h feet.
2. There is one door of dimensions 6 feet * 3 feet.
3. There are three windows of dimensions 4 feet * 3 feet each.
4. The cost of white washing is Rs. 10 per square feet.
5. The total cost is Rs. 9060.

Prove that the height of the room h is 12 feet.
-/
theorem room_height (h : ℕ) (H1 : 25 * 15 * h) 
                    (H2 : 1 * (6 * 3))
                    (H3 : 3 * (4 * 3))
                    (H4 : 10) 
                    (H5 : 9060) : h = 12 := by
  sorry

end room_height_l528_528123


namespace tangent_line_at_pi_fx_lt_xcubed_max_k_l528_528759

noncomputable def f (x : ℝ) := Real.sin x - x * Real.cos x

theorem tangent_line_at_pi :
  let fp := (π, f π) in
  ∃ (m b : ℝ), ∀ x, (f x = m * x + b) ∧ (m = 0) ∧ (b = π) :=
by sorry

theorem fx_lt_xcubed (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  f(x) < (1 / 3) * x^3 :=
by sorry

theorem max_k (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  ∀ (k : ℝ), (f(x) > k * x - x * Real.cos x) → k ≤ 2 / π :=
by sorry

end tangent_line_at_pi_fx_lt_xcubed_max_k_l528_528759


namespace find_A_max_min_l528_528237

def is_coprime_with_36 (n : ℕ) : Prop := Nat.gcd n 36 = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let rest := n / 10
  d * 10^7 + rest

theorem find_A_max_min (B : ℕ) 
  (h1 : B > 77777777) 
  (h2 : is_coprime_with_36 B) : 
  move_last_digit_to_first B = 99999998 ∨ 
  move_last_digit_to_first B = 17777779 := 
by
  sorry

end find_A_max_min_l528_528237


namespace penny_dime_halfdollar_same_probability_l528_528897

def probability_same_penny_dime_halfdollar : ℚ :=
  let total_outcomes := 2 ^ 5
  let successful_outcomes := 2 * 2 * 2
  successful_outcomes / total_outcomes

theorem penny_dime_halfdollar_same_probability :
  probability_same_penny_dime_halfdollar = 1 / 4 :=
by 
  sorry

end penny_dime_halfdollar_same_probability_l528_528897


namespace inner_circumference_correct_l528_528911

-- Define the radius of the outer circle and the width of the race track
def radius_outer_circle : ℝ := 84.02817496043394
def width_race_track : ℝ := 14

-- Calculate the radius of the inner circle
def radius_inner_circle : ℝ := radius_outer_circle - width_race_track

-- Define the expected inner circumference
def inner_circumference : ℝ := 2 * Real.pi * radius_inner_circle

-- State the theorem that the inner circumference equals approximately 439.8229715025715 meters
theorem inner_circumference_correct : inner_circumference ≈ 439.8229715025715 :=
by
  -- This is where the proof would go
  sorry

end inner_circumference_correct_l528_528911


namespace polygon_is_scalene_l528_528916

namespace GeometryProof

def line1 (x : ℝ) : ℝ := 3 * x + 2
def line2 (x : ℝ) : ℝ := -4 * x + 2
def line3 (x : ℝ) : ℝ := -2

theorem polygon_is_scalene :
  let p1 := (0, line1 0)
  let p2 := (-4 / 3, line2 (-4 / 3))
  let p3 := (1, line3 1)
  let d1 := let ⟨x1, y1⟩ := p1; let ⟨x2, y2⟩ := p2 in Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let d2 := let ⟨x1, y1⟩ := p1; let ⟨x2, y2⟩ := p3 in Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let d3 := let ⟨x1, y1⟩ := p2; let ⟨x2, y2⟩ := p3 in Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  in d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 := 
by {
  let p1 := (0, (line1 0))
  let p2 := (-4 / 3, (line2 (-4 / 3)))
  let p3 := (1, (line3 1))
  let d1 := Real.sqrt ( ((-4 / 3) - 0)^2 + ((-2) - 2)^2 )
  let d2 := Real.sqrt ( (1 - 0)^2 + ((-2) - 2)^2 )
  let d3 := Real.sqrt ( (1 - (-4 / 3))^2 + ((-2) - (-2))^2 )
  have : d1 ≠ d2 := sorry
  have : d2 ≠ d3 := sorry
  have : d1 ≠ d3 := sorry
  exact and.intro ‹d1 ≠ d2› (and.intro ‹d2 ≠ d3› ‹d1 ≠ d3›)
}

end GeometryProof

end polygon_is_scalene_l528_528916


namespace probability_log2_interval_l528_528781

-- Define the interval [0, 9] as a real number range.
def interval : set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 9}

-- Define the predicate for the logarithmic inequality.
def log2_interval (x : ℝ) : Prop := 1 ≤ real.log x / real.log 2 ∧ real.log x / real.log 2 ≤ 2

-- Calculate the probability that a random real number from [0, 9] satisfies the inequality.
theorem probability_log2_interval :
  (measure_theory.measure.interval_oc 2 4).measure / (measure_theory.measure.interval_oc 0 9).measure = 2 / 9 :=
sorry

end probability_log2_interval_l528_528781


namespace perpendicular_distance_l528_528256

structure Vertex :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def S : Vertex := ⟨6, 0, 0⟩
def P : Vertex := ⟨0, 0, 0⟩
def Q : Vertex := ⟨0, 5, 0⟩
def R : Vertex := ⟨0, 0, 4⟩

noncomputable def distance_from_point_to_plane (S P Q R : Vertex) : ℝ := sorry

theorem perpendicular_distance (S P Q R : Vertex) (hS : S = ⟨6, 0, 0⟩) (hP : P = ⟨0, 0, 0⟩) (hQ : Q = ⟨0, 5, 0⟩) (hR : R = ⟨0, 0, 4⟩) :
  distance_from_point_to_plane S P Q R = 6 :=
  sorry

end perpendicular_distance_l528_528256


namespace sum_of_slope_and_y_intercept_equals_neg10_l528_528155

def A : (ℝ × ℝ) := (-2, 10)
def B : (ℝ × ℝ) := (3, 0)
def C : (ℝ × ℝ) := (10, 0)
def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
def slope (p q : ℝ × ℝ) : ℝ := (q.2 - p.2) / (q.1 - p.1)

theorem sum_of_slope_and_y_intercept_equals_neg10
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
  (M : ℝ × ℝ := midpoint A C)
  (m : ℝ := slope B M)
  (c : ℝ := M.2 - m * M.1) :
  m + c = -10 :=
sorry

end sum_of_slope_and_y_intercept_equals_neg10_l528_528155


namespace sum_of_x_coords_of_intersections_l528_528674

theorem sum_of_x_coords_of_intersections :
  let x_coords := [3, 9] in
  (∑ x in x_coords, x) = 12 :=
by
  let x_coords := [3, 9]
  have h₁ : list.sum x_coords = 12 := by rfl
  exact h₁

end sum_of_x_coords_of_intersections_l528_528674


namespace farmer_earns_from_runt_pig_l528_528602

def average_bacon_per_pig : ℕ := 20
def price_per_pound : ℕ := 6
def runt_pig_bacon : ℕ := average_bacon_per_pig / 2
def total_money_made (bacon_pounds : ℕ) (price_per_pound : ℕ) : ℕ := bacon_pounds * price_per_pound

theorem farmer_earns_from_runt_pig :
  total_money_made runt_pig_bacon price_per_pound = 60 :=
sorry

end farmer_earns_from_runt_pig_l528_528602


namespace golden_state_total_points_l528_528033

theorem golden_state_total_points :
  let draymond_points := 12
  let curry_points := 2 * draymond_points
  let kelly_points := 9
  let durant_points := 2 * kelly_points
  let klay_points := draymond_points / 2
  draymond_points + curry_points + kelly_points + durant_points + klay_points = 69 :=
by
  let draymond_points := 12
  let curry_points := 2 * draymond_points
  let kelly_points := 9
  let durant_points := 2 * kelly_points
  let klay_points := draymond_points / 2
  calc
    draymond_points + curry_points + kelly_points + durant_points + klay_points
    = 12 + (2 * 12) + 9 + (2 * 9) + (12 / 2) : by sorry
    = 69 : by sorry

end golden_state_total_points_l528_528033


namespace beau_age_today_l528_528658

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end beau_age_today_l528_528658


namespace swap_sum_2007_l528_528137

/-- Represents the initial and final conditions of the problem:
1. Numbers from 1 to 2006 are placed in a circle.
2. After swaps, each initially placed number moves to its diametrically opposite position.
We aim to prove that at least one swap involves numbers summing to 2007. -/

theorem swap_sum_2007 :
  ∃ (swaps : ℕ → (ℕ × ℕ)), -- an infinite sequence of swaps (pairs of swapped positions)
  (∀ i, i < 2006 → (swaps i).1 ≠ (swaps i).2 →  -- the positions are different
     (swaps i).1 + (swaps i).2 = 2007) →  -- positions sum to 2007
  ∃ i, i < 2006 ∧
  (∀ k, k ∈ {1, 2, ..., 2006}.symm →  -- all numbers are in initial range
     let x := (swaps i).1, y := (swaps i).2 in ((x, y) = (k, 2007 - k) ∨ (y, x) = (k, 2007 - k))) :=
sorry

end swap_sum_2007_l528_528137


namespace likes_spinach_not_music_lover_l528_528653

universe u

variable (Person : Type u)
variable (likes_spinach is_pearl_diver is_music_lover : Person → Prop)

theorem likes_spinach_not_music_lover :
  (∃ x, likes_spinach x ∧ ¬ is_pearl_diver x) →
  (∀ x, is_music_lover x → (is_pearl_diver x ∨ ¬ likes_spinach x)) →
  (∀ x, (¬ is_pearl_diver x → is_music_lover x) ∨ (is_pearl_diver x → ¬ is_music_lover x)) →
  (∀ x, likes_spinach x → ¬ is_music_lover x) :=
by
  sorry

end likes_spinach_not_music_lover_l528_528653


namespace length_may_remain_same_l528_528391

-- translation of given conditions to Lean definitions
def ObliqueProjection : Type := sorry -- placeholder
def LineSegment : Type := sorry -- placeholder

noncomputable def not_perpendicular_to_coordinate_axes (l : LineSegment) : Prop := sorry -- placeholder definition

-- statement of the proof problem
theorem length_may_remain_same (P : ObliqueProjection) (l : LineSegment) 
  (h : not_perpendicular_to_coordinate_axes l) : 
  (l.length_in_projection l P = l.length_in_original l) ∨ (l.length_in_projection l P ≠ l.length_in_original l) :=
sorry

end length_may_remain_same_l528_528391


namespace range_of_x_plus_2y_l528_528061

noncomputable def range_of_sum (x y : ℝ) : Set ℝ :=
  {z | ∃ θ : ℝ, x = sqrt 6 * Real.cos θ ∧ y = 2 * Real.sin θ ∧ z = x + 2 * y}

theorem range_of_x_plus_2y (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 12) :
  range_of_sum x y = Set.Icc (-sqrt 22) (sqrt 22) :=
by
  sorry

end range_of_x_plus_2y_l528_528061


namespace distance_after_3_minutes_l528_528640

/-- Let truck_speed be 65 km/h and car_speed be 85 km/h.
    Let time_in_minutes be 3 and converted to hours it is 0.05 hours.
    The goal is to prove that the distance between the truck and the car
    after 3 minutes is 1 kilometer. -/
def truck_speed : ℝ := 65 -- speed in km/h
def car_speed : ℝ := 85 -- speed in km/h
def time_in_minutes : ℝ := 3 -- time in minutes
def time_in_hours : ℝ := time_in_minutes / 60 -- converted time in hours
def distance_truck := truck_speed * time_in_hours
def distance_car := car_speed * time_in_hours
def distance_between : ℝ := distance_car - distance_truck

theorem distance_after_3_minutes : distance_between = 1 := by
  -- Proof steps would go here
  sorry

end distance_after_3_minutes_l528_528640


namespace sequence_inequality_l528_528521

noncomputable def a : ℕ → ℝ 
| 0 := (Real.sqrt 2) / 2
| (n+1) := (Real.sqrt 2) / 2 * Real.sqrt (1 - Real.sqrt (1 - (a n)^2))

def b : ℕ → ℝ :=
  λ n => Real.tan (Real.pi / 2^(n+2))

theorem sequence_inequality (n : ℕ) : 2^(n+2) * (Real.sin (Real.pi / 2^(n+2))) < Real.pi ∧ Real.pi < 2^(n+2) * (Real.tan (Real.pi / 2^(n+2))) :=
sorry

end sequence_inequality_l528_528521


namespace find_g_5_l528_528907

variable (g : ℝ → ℝ)

axiom func_eqn : ∀ x y : ℝ, x * g y = y * g x
axiom g_10 : g 10 = 15

theorem find_g_5 : g 5 = 7.5 :=
by
  sorry

end find_g_5_l528_528907


namespace part_I_part_II_l528_528347

section
variable {x : ℝ}

def f (x : ℝ) : ℝ := Real.log (x + 1) - x

-- PART I: Prove that f is monotonically decreasing for x > 0
theorem part_I : ∀ x > 0, deriv f x < 0 := by
  sorry

-- PART II: Prove the inequality for x > -1
theorem part_II (hx : x > -1) : 
  1 - (1 / (x + 1)) ≤ Real.log (x + 1) ∧ Real.log (x + 1) ≤ x := by
  sorry
end

end part_I_part_II_l528_528347


namespace tel_aviv_rain_probability_l528_528487

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l528_528487


namespace series_convergence_series_sum_series_limit_l528_528063

noncomputable def s (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (n + 1), (∏ i in Finset.range (k + 1), (1 - (i : ℝ) * x)) / (k.fact : ℝ)

theorem series_convergence (x : ℝ) (h : |x| < 1) : 
  ∃ L : ℝ, ∃ S : ℝ, S = (s n x) -> tendsto S at_top (nhds L) :=
sorry

theorem series_sum (n : ℕ) (x : ℝ) :
  s n x = (1 - (-x)^(n + 1)) / (1 + x) :=
sorry

theorem series_limit :
  tendsto (λ n, s n 0) at_top (nhds 1) :=
sorry

end series_convergence_series_sum_series_limit_l528_528063


namespace notebooks_last_days_l528_528414

/-- John buys 5 notebooks, each with 40 pages. 
    He uses 4 pages per day. 
    Prove the notebooks last 50 days. -/
theorem notebooks_last_days : 
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  (notebooks * pages_per_notebook) / pages_per_day = 50 := 
by
  -- Definitions
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  calc
    (notebooks * pages_per_notebook) / pages_per_day
      = (5 * 40) / 4 : by rfl
      ... = 200 / 4 : by rfl
      ... = 50 : by rfl

end notebooks_last_days_l528_528414


namespace polynomial_terms_and_degree_l528_528136

noncomputable def polynomial : ℕ := sorry

theorem polynomial_terms_and_degree :
  ∃ p : polynomial, 
  (number_of_terms p = 3) ∧ (degree p = 4) :=
sorry

end polynomial_terms_and_degree_l528_528136


namespace tan_alpha_plus_pi_over_4_sin_2alpha_cos_2_l528_528291

theorem tan_alpha_plus_pi_over_4 (alpha : ℝ) (h₁ : α ∈ (π / 2, π)) : 
  tan (alpha + π / 4) = -1 / 3 :=
sorry

theorem sin_2alpha_cos_2 (alpha : ℝ) (h₁ : α ∈ (π / 2, π)) :
  sin (2 * alpha) * cos (2) = -7 / 5 :=
sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_cos_2_l528_528291


namespace calculate_nested_expression_l528_528248

theorem calculate_nested_expression :
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2 = 1457 :=
by
  sorry

end calculate_nested_expression_l528_528248


namespace largest_multiple_of_7_less_than_neg_100_l528_528553

theorem largest_multiple_of_7_less_than_neg_100 : 
  ∃ (x : ℤ), (∃ n : ℤ, x = 7 * n) ∧ x < -100 ∧ ∀ y : ℤ, (∃ m : ℤ, y = 7 * m) ∧ y < -100 → y ≤ x :=
by
  sorry

end largest_multiple_of_7_less_than_neg_100_l528_528553


namespace minimum_of_f_range_of_a_l528_528351

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem minimum_of_f :
  let e_inv := (1 : ℝ) / Real.exp 1 in
  (∀ x > 0, f x ≥ -e_inv) ∧ (f e_inv = -e_inv) := 
by
  intro e_inv
  simp [f, e_inv]
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, f x ≥ a * x - 1) ↔ a ≤ 1 := 
by
  sorry

end minimum_of_f_range_of_a_l528_528351


namespace find_roses_in_october_l528_528503

variable (d : ℕ)
variable (R_Oct R_Nov R_Dec R_Jan R_Feb : ℕ)

-- Conditions
axiom h1 : R_Nov = 120
axiom h2 : R_Dec = 132
axiom h3 : R_Jan = 144
axiom h4 : R_Feb = 156
axiom h5 : R_Dec = R_Nov + d
axiom h6 : R_Jan = R_Dec + d
axiom h7 : R_Feb = R_Jan + d

noncomputable def roses_in_october : Prop := R_Oct = 108

theorem find_roses_in_october : roses_in_october := by
  have h_diff : d = 12 := by sorry
  have h_oct : R_Oct + d = R_Nov := by sorry
  show R_Oct = 108 from by sorry

end find_roses_in_october_l528_528503


namespace zeros_distance_l528_528748

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3*x^2 + a

theorem zeros_distance (a x1 x2 : ℝ) 
  (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) (h_order: x1 < x2) : 
  x2 - x1 = 3 := 
sorry

end zeros_distance_l528_528748


namespace count_possible_n_l528_528042

def triangle_sides (n : ℕ) := (AB AC BC : ℕ) 
  ( hAB : AB = 4 * n - 2)
  ( hAC : AC = 3 * n + 4)
  ( hBC : BC = 5 * n + 3) 
  (angle_cond : ∃ (A B C : ℝ), ∠B > ∠C ∧ ∠C > ∠A ∧ BC > AC ∧ AC > AB): 
  (h1 : AB + AC > BC) (h2 : AB + BC > AC) (h3 : AC + BC > AB)

theorem count_possible_n :
  ∃ (n_list : list ℕ), n_list.length = 5 ∧ 
  ∀ n, n ∈ n_list → 1 ≤ n ∧ n < 6 := 
begin
  sorry
end

end count_possible_n_l528_528042


namespace price_of_soda_l528_528141

theorem price_of_soda (regular_price_per_can : ℝ) (case_discount : ℝ) (bulk_discount : ℝ) (num_cases : ℕ) (num_cans : ℕ) :
  regular_price_per_can = 0.15 →
  case_discount = 0.12 →
  bulk_discount = 0.05 →
  num_cases = 3 →
  num_cans = 75 →
  (num_cans * ((regular_price_per_can * (1 - case_discount)) * (1 - bulk_discount))) = 9.405 :=
by
  intros h1 h2 h3 h4 h5
  -- normal price per can
  have hp1 : ℝ := regular_price_per_can
  -- price after case discount
  have hp2 : ℝ := hp1 * (1 - case_discount)
  -- price after bulk discount
  have hp3 : ℝ := hp2 * (1 - bulk_discount)
  -- total price
  have total_price : ℝ := num_cans * hp3
  -- goal
  sorry -- skip the proof, as only the statement is needed.

end price_of_soda_l528_528141


namespace part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l528_528326

open Set Real

def setA (a : ℝ) : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ a + 5}
def setB : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem part_one_a_two_complement_union (a : ℝ) (h : a = 2) :
  compl (setA a ∪ setB) = Iic 2 ∪ Ici 10 := sorry

theorem part_one_a_two_complement_intersection (a : ℝ) (h : a = 2) :
  compl (setA a) ∩ setB = Ioo 2 3 ∪ Ioo 7 10 := sorry

theorem part_two_subset (a : ℝ) (h : setA a ⊆ setB) :
  a < 5 := sorry

end part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l528_528326


namespace translation_is_elevator_l528_528228

-- Definitions representing the conditions
def P_A : Prop := true  -- The movement of elevators constitutes translation.
def P_B : Prop := false -- Swinging on a swing does not constitute translation.
def P_C : Prop := false -- Closing an open textbook does not constitute translation.
def P_D : Prop := false -- The swinging of a pendulum does not constitute translation.

-- The goal is to prove that Option A is the phenomenon that belongs to translation
theorem translation_is_elevator : P_A ∧ ¬P_B ∧ ¬P_C ∧ ¬P_D :=
by
  sorry -- proof not required

end translation_is_elevator_l528_528228


namespace fernanda_total_time_eq_90_days_l528_528700

-- Define the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Define the total time calculation
def total_time_to_finish_audiobooks (a h r : ℕ) : ℕ :=
  (h / r) * a

-- The assertion we need to prove
theorem fernanda_total_time_eq_90_days :
  total_time_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 :=
by sorry

end fernanda_total_time_eq_90_days_l528_528700


namespace smallest_multiple_not_multiple_of_11_l528_528946

theorem smallest_multiple_not_multiple_of_11 :
  ∃ n : ℕ, (n > 0) ∧ (45 ∣ n) ∧ (75 ∣ n) ∧ ¬ (11 ∣ n) ∧ ∀ m : ℕ, (m > 0) ∧ (45 ∣ m) ∧ (75 ∣ m) ∧ ¬ (11 ∣ m) → n ≤ m :=
begin
  sorry
end

end smallest_multiple_not_multiple_of_11_l528_528946


namespace find_side_length_l528_528814

theorem find_side_length (ABC : Triangle) 
  (A B C : ℝ) 
  (a b c : ℝ)
  (h1 : ∠C = 4 * ∠A)
  (h2 : a = 21)
  (h3 : c = 54) : 
  b = 21 * (16 * (sin A)^4 - 20 * (sin A)^2 + 5) := by
  sorry

end find_side_length_l528_528814


namespace problem1_problem2_l528_528434

-- Definitions for vector calculations
structure Vector2D where
  x : ℝ
  y : ℝ

-- Dot product of two vectors
def dot_product (a b : Vector2D) : ℝ :=
  a.x * b.x + a.y * b.y

-- Vector a
def a (x : ℝ) : Vector2D := ⟨Real.cos x, Real.sin x⟩

-- Vector b
def b (x : ℝ) : Vector2D := ⟨Real.cos x + 2 * Real.sqrt 3, Real.sin x⟩

-- Vector c
def c (α : ℝ) : Vector2D := ⟨Real.sin α, Real.cos α⟩

-- Function f(x)
def f (x α : ℝ) : ℝ :=
  dot_product (a x) (⟨(b x).x, (b x).y - 2 * (c α).y⟩)

-- Problem (1)
theorem problem1 (x α : ℝ) (h : dot_product (a x) (c α) = 0) : Real.cos (2 * x + 2 * α) = 1 := by
  sorry

-- Problem (2)
theorem problem2 (x : ℝ) (α := 0) : 
  ∃ k : ℤ, f x α ≤ 5 ∧ (f (2 * k * Real.pi - Real.pi / 6) α = 5) := by
  sorry

end problem1_problem2_l528_528434


namespace pizza_area_increase_l528_528780

theorem pizza_area_increase 
  (r : ℝ) 
  (A_medium A_large : ℝ) 
  (h_medium_area : A_medium = Real.pi * r^2)
  (h_large_area : A_large = Real.pi * (1.40 * r)^2) : 
  ((A_large - A_medium) / A_medium) * 100 = 96 := 
by 
  sorry

end pizza_area_increase_l528_528780


namespace cross_section_area_correct_l528_528879

noncomputable def area_cross_section (B C D E A P : Type) (S T k l : ℝ) (face_area_ABC : Set D) (total_surface_BCDE : Set E)
  (parallel_planes : Prop) (sphere_inscribed : Prop) (ratios_angle : Prop) : ℝ :=
sorry

theorem cross_section_area_correct (B C D E A P : Type) (S T k l : ℝ) (face_area_ABC : Set D) (total_surface_BCDE : Set E)
  (parallel_planes : Prop) (sphere_inscribed : Prop) (ratios_angle : Prop) :
  area_cross_section B C D E A P S T k l face_area_ABC total_surface_BCDE parallel_planes sphere_inscribed ratios_angle =
  (3 + a k (sqrt (2 - 2 * l))) / 1 :=
sorry

end cross_section_area_correct_l528_528879


namespace correct_statement_C_l528_528174

theorem correct_statement_C (A B D : Prop) (C : Prop) :
  (A ↔ ¬(∃ p1 p2 p3 : ℝ × ℝ × ℝ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ collinear p1 p2 p3)) →
  (B ↔ ¬(∃ (a : ∀ x : ℝ, ℝ × ℝ × ℝ) (α : ℝ × ℝ × ℝ → ℝ), line_outside_plane a α → intersects a α)) →
  (C ↔ (∃ (pyramid : Type) (P : pyramid → Prop), regular_pyramid pyramid ∧ plane_parallel_to_base P pyramid → regular_frustum (cut_pyramid P pyramid))) →
  (D ↔ ¬(∃ (prism : Type) (lateral_surface : prism → Prop), oblique_prism prism ∧ rectangle lateral_surface)) →
  C :=
by
  sorry

end correct_statement_C_l528_528174


namespace discriminant_of_quad_eq_l528_528551

def a : ℕ := 5
def b : ℕ := 8
def c : ℤ := -6

def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

theorem discriminant_of_quad_eq : discriminant 5 8 (-6) = 184 :=
by
  -- The proof is skipped
  sorry

end discriminant_of_quad_eq_l528_528551


namespace find_extrema_A_l528_528231

def eight_digit_number(n : ℕ) : Prop := n ≥ 10^7 ∧ n < 10^8

def coprime_with_thirtysix(n : ℕ) : Prop := Nat.gcd n 36 = 1

def transform_last_to_first(n : ℕ) : ℕ := 
  let last := n % 10
  let rest := n / 10
  last * 10^7 + rest

theorem find_extrema_A :
  ∃ (A_max A_min : ℕ), 
    (∃ B_max B_min : ℕ, 
      eight_digit_number B_max ∧ 
      eight_digit_number B_min ∧ 
      coprime_with_thirtysix B_max ∧ 
      coprime_with_thirtysix B_min ∧ 
      B_max > 77777777 ∧ 
      B_min > 77777777 ∧ 
      transform_last_to_first B_max = A_max ∧ 
      transform_last_to_first B_min = A_min) ∧ 
    A_max = 99999998 ∧ 
    A_min = 17777779 := 
  sorry

end find_extrema_A_l528_528231


namespace quadratic_coefficient_l528_528504

noncomputable def quadratic_at_vertex (a : ℝ) (x : ℝ) := a * (x + 2)^2 + 3

theorem quadratic_coefficient :
  (∃ a : ℝ, quadratic_at_vertex a 3 = -45) → ∃ a : ℝ, a = -48 / 25 :=
begin
  intro h,
  cases h with a ha,
  use a,
  -- The question requires completing the proof, but since the prompt asks only for the statement, we use 'sorry'
  sorry
end

end quadratic_coefficient_l528_528504


namespace tan_of_x_is_3_l528_528329

theorem tan_of_x_is_3 (x : ℝ) (h : Real.tan x = 3) (hx : Real.cos x ≠ 0) : 
  (Real.sin x + 3 * Real.cos x) / (2 * Real.sin x - 3 * Real.cos x) = 2 :=
by
  sorry

end tan_of_x_is_3_l528_528329


namespace triangle_YXZ_angle_l528_528815

noncomputable def angle_YXZ : Real :=
  Real.arctan (2 * Real.sqrt 5 / 5)

theorem triangle_YXZ_angle :
  ∃ (X Y Z W : Type) (XYZ : Triangle X Y Z),
    XYZ.right_angle_at X ∧
    XYZ.longest_side X Z ∧
    XYZ.point_on_side Y Z W ∧
    XYZ.side_ratio Y W W Z = 2 ∧
    XYZ.side_eq X Y Y W →
    angle_YXZ = 21.8 :=
sorry

end triangle_YXZ_angle_l528_528815


namespace weights_system_l528_528582

variables (x y : ℝ)

-- The conditions provided in the problem
def condition1 : Prop := 5 * x + 6 * y = 1
def condition2 : Prop := 4 * x + 7 * y = 5 * x + 6 * y

-- The statement to be proven
theorem weights_system (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) :
  (5 * x + 6 * y = 1) ∧ (4 * x + 7 * y = 4 * x + 7 * y) :=
sorry

end weights_system_l528_528582


namespace total_food_per_day_l528_528134

theorem total_food_per_day 
  (first_soldiers : ℕ)
  (second_soldiers : ℕ)
  (food_first_side_per_soldier : ℕ)
  (food_second_side_per_soldier : ℕ) :
  first_soldiers = 4000 →
  second_soldiers = first_soldiers - 500 →
  food_first_side_per_soldier = 10 →
  food_second_side_per_soldier = food_first_side_per_soldier - 2 →
  (first_soldiers * food_first_side_per_soldier + second_soldiers * food_second_side_per_soldier = 68000) :=
by
  intros h1 h2 h3 h4
  sorry

end total_food_per_day_l528_528134


namespace notebooks_last_days_l528_528413

/-- John buys 5 notebooks, each with 40 pages. 
    He uses 4 pages per day. 
    Prove the notebooks last 50 days. -/
theorem notebooks_last_days : 
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  (notebooks * pages_per_notebook) / pages_per_day = 50 := 
by
  -- Definitions
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  calc
    (notebooks * pages_per_notebook) / pages_per_day
      = (5 * 40) / 4 : by rfl
      ... = 200 / 4 : by rfl
      ... = 50 : by rfl

end notebooks_last_days_l528_528413


namespace distance_from_centroid_to_hypotenuse_l528_528387

-- Define the right triangle with the given conditions
structure RightTriangle :=
  (A B C O: Point)
  (angle_ACB : ∠ ACB = 90)
  (angle_CAB : ∠ CAB = α)
  (area : ℝ := S)
  (is_centroid : is_centroid O A B C)

-- The theorem to prove the distance from the centroid to the hypotenuse
theorem distance_from_centroid_to_hypotenuse
  (T : RightTriangle)
  (α : ℝ)
  (S : ℝ) :
  distance_to_hypotenuse T.O T.A T.B T.C = 1 / 3 * sqrt (S * sin (2 * α)) :=
sorry

end distance_from_centroid_to_hypotenuse_l528_528387


namespace mohamed_donated_more_l528_528836

-- Definitions of the conditions
def toysLeilaDonated : ℕ := 2 * 25
def toysMohamedDonated : ℕ := 3 * 19

-- The theorem stating Mohamed donated 7 more toys than Leila
theorem mohamed_donated_more : toysMohamedDonated - toysLeilaDonated = 7 :=
by
  sorry

end mohamed_donated_more_l528_528836


namespace analytic_function_l528_528286

noncomputable def real_part : ℂ → ℝ := λ z, 2 * Real.exp z.re * Real.cos z.im

theorem analytic_function (f : ℂ → ℂ)
  (h_real_part : ∀ z : ℂ, (f z).re = real_part z)
  (h_initial_condition : f 0 = 2) :
  ∀ z : ℂ, f z = 2 * Complex.exp z :=
by sorry

end analytic_function_l528_528286


namespace area_of_quadrilateral_is_16_l528_528031

-- Definition of vectors in the Cartesian coordinate plane
structure Vector := (x : ℝ) (y : ℝ)

def AB : Vector := ⟨6, 1⟩
def CD : Vector := ⟨-2, -3⟩

-- Variables for vectors BC and AD
variables (x y : ℝ)

def BC : Vector := ⟨x, y⟩
def AD : Vector := ⟨x + 4, y - 2⟩

-- Parallel condition: AD is parallel to BC
def parallel_condition : Prop := (x + 4) * y - (y - 2) * x = 0

-- Perpendicular condition: AC is perpendicular to BD
def AC : Vector := ⟨x + 6, y + 1⟩
def BD : Vector := ⟨x - 2, y - 3⟩
def perpendicular_condition : Prop := (x + 6) * (x - 2) + (y + 1) * (y - 3) = 0

-- Solving the system of equations for x and y to find the area of the quadrilateral
def quadratic_equation (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y - 15 = 0
def linear_equation (x y : ℝ) : Prop := x + 2 * y = 0

-- Define the area of the quadrilateral assuming the solutions for x and y satisfy the conditions
def area_quad (AC BD : Vector) : ℝ := 1 / 2 * (AC.x * BD.y - AC.y * BD.x)

-- Main theorem to be proved
theorem area_of_quadrilateral_is_16 :
  parallel_condition x y → perpendicular_condition x y →
  (quadratic_equation x y ∧ linear_equation x y) →
  area_quad (AC x y) (BD x y) = 16 :=
by sorry

end area_of_quadrilateral_is_16_l528_528031


namespace range_of_a_l528_528320

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc 0 1 → a ≥ Real.exp x) ∧ (∃ x : ℝ, x^2 + 4 * x + a = 0) →
  a ∈ set.Icc Real.exp 4 :=
by sorry

end range_of_a_l528_528320


namespace pyramid_edges_l528_528528

-- Define the conditions
def isPyramid (n : ℕ) : Prop :=
  (n + 1) + (n + 1) = 16

-- Statement to be proved
theorem pyramid_edges : ∃ (n : ℕ), isPyramid n ∧ 2 * n = 14 :=
by {
  sorry
}

end pyramid_edges_l528_528528


namespace mixed_alcohol_percentage_l528_528110

def ratio_volume (a b : ℕ) : ℚ := a / (a + b : ℚ)

def mix_ratio_volume (r1 r2 r3 : ℕ) (v1 v2 v3 : ℚ) : ℚ :=
  let total_volume := (r1 : ℚ) + (r2 : ℚ) + (r3 : ℚ)
  let total_alcohol := (v1 : ℚ) * (r1 : ℚ) + (v2 : ℚ) * (r2 : ℚ) + (v3 : ℚ) * (r3 : ℚ)
  (total_alcohol / total_volume) * 100

theorem mixed_alcohol_percentage :
  let A_ratio := (21, 4)
  let B_ratio := (2, 3)
  let C_ratio := (5, 7)
  let mix_ratio := (5, 6, 7)
  let A_alcohol := ratio_volume A_ratio.1 A_ratio.2
  let B_alcohol := ratio_volume B_ratio.1 B_ratio.2
  let C_alcohol := ratio_volume C_ratio.1 C_ratio.2
  let expected_percentage := mix_ratio_volume mix_ratio.1 mix_ratio.2 mix_ratio.3 A_alcohol B_alcohol C_alcohol
  expected_percentage = 52.87 :=
by
  sorry

end mixed_alcohol_percentage_l528_528110


namespace solution_set_of_inequality_l528_528316

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = 2^x - 4

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : satisfies_condition f) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
sorry

end solution_set_of_inequality_l528_528316


namespace company_speaks_french_percentage_l528_528792

variable (totalEmployees : ℕ)
variable (percentMen : ℝ) (percentMenSpeakFrench : ℝ)
variable (percentWomenDontSpeakFrench : ℝ)

def percentWomenSpeakFrench : ℝ := 1 - percentWomenDontSpeakFrench

theorem company_speaks_french_percentage
  (h_totalEmployees : totalEmployees = 100)
  (h_percentMen : percentMen = 0.7)
  (h_percentMenSpeakFrench : percentMenSpeakFrench = 0.5)
  (h_percentWomenDontSpeakFrench : percentWomenDontSpeakFrench = 0.83333333333333331) :
  let men := percentMen * totalEmployees
  let women := totalEmployees - men
  let menSpeakFrench := percentMenSpeakFrench * men
  let womenSpeakFrench := percentWomenSpeakFrench * women in
  (menSpeakFrench + womenSpeakFrench) / totalEmployees * 100 = 40 := by
  sorry

end company_speaks_french_percentage_l528_528792


namespace runners_order_count_l528_528435

theorem runners_order_count (n : ℕ) (h : n = 6) :
    (6 - 1)! = 120 :=
by
  sorry

end runners_order_count_l528_528435


namespace max_points_on_circle_l528_528085

noncomputable def circleMaxPoints (P C : ℝ × ℝ) (r1 r2 d : ℝ) : ℕ :=
  if d = r1 + r2 ∨ d = abs (r1 - r2) then 1 else 
  if d < r1 + r2 ∧ d > abs (r1 - r2) then 2 else 0

theorem max_points_on_circle (P : ℝ × ℝ) (C : ℝ × ℝ) :
  let rC := 5
  let distPC := 9
  let rP := 4
  circleMaxPoints P C rC rP distPC = 1 :=
by sorry

end max_points_on_circle_l528_528085


namespace quadratic_inequality_solution_l528_528005

variable {a b c : ℝ}

def f (x : ℝ) : ℝ := a*x^2 + b*x + c

theorem quadratic_inequality_solution (h1 : a < 0) (h2 : f (-2) = 0) (h3 : f 4 = 0) :
  f 2 > f (-1) ∧ f (-1) > f 5 :=
by
  sorry

end quadratic_inequality_solution_l528_528005


namespace Tom_sold_games_for_240_l528_528934

-- Define the value of games and perform operations as per given conditions
def original_value : ℕ := 200
def tripled_value : ℕ := 3 * original_value
def sold_percentage : ℕ := 40
def sold_value : ℕ := (sold_percentage * tripled_value) / 100

-- Assert the proof problem
theorem Tom_sold_games_for_240 : sold_value = 240 := 
by
  sorry

end Tom_sold_games_for_240_l528_528934


namespace machines_finish_job_in_24_over_11_hours_l528_528874

theorem machines_finish_job_in_24_over_11_hours :
    let work_rate_A := 1 / 4
    let work_rate_B := 1 / 12
    let work_rate_C := 1 / 8
    let combined_work_rate := work_rate_A + work_rate_B + work_rate_C
    (1 : ℝ) / combined_work_rate = 24 / 11 :=
by
  sorry

end machines_finish_job_in_24_over_11_hours_l528_528874


namespace polynomial_inequality_l528_528457

theorem polynomial_inequality {C : ℝ} :
  ∃ C, ∀ (p : ℝ[X]), p.degree = 1999 → |p.eval 0| ≤ C * ∫ x in -1..1, |p.eval x| := 
sorry

end polynomial_inequality_l528_528457


namespace inequality_reciprocal_l528_528330

theorem inequality_reciprocal (a b : ℝ) (hab : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end inequality_reciprocal_l528_528330


namespace bond_interest_percentage_l528_528415

noncomputable def interest_percentage_of_selling_price (face_value interest_rate : ℝ) (selling_price : ℝ) : ℝ :=
  (face_value * interest_rate) / selling_price * 100

theorem bond_interest_percentage :
  let face_value : ℝ := 5000
  let interest_rate : ℝ := 0.07
  let selling_price : ℝ := 5384.615384615386
  interest_percentage_of_selling_price face_value interest_rate selling_price = 6.5 :=
by
  sorry

end bond_interest_percentage_l528_528415


namespace area_calculation_l528_528008

-- Definitions for conditions
def small_square_side := 3 -- side of each small square in cm
def grid_side := small_square_side * 6 -- side of the entire grid in cm
def area_of_grid := grid_side^2 -- total area of the grid in square cm

def small_circle_radius := small_square_side / 2.0 -- radius of each smaller circle in cm
def area_of_small_circle := Real.pi * small_circle_radius^2
def num_small_circles := 5

def large_circle_radius := small_square_side * 2 -- radius of the larger circle in cm
def area_of_large_circle := Real.pi * large_circle_radius^2

def total_area_of_circles := (num_small_circles * area_of_small_circle) + area_of_large_circle

def visible_shaded_area := area_of_grid - total_area_of_circles

def A := 324.0
def B := 47.25

-- Statement to prove
theorem area_calculation : A - B * Real.pi = visible_shaded_area ∧ A + B = 371.25 :=
  by 
    sorry

end area_calculation_l528_528008


namespace average_age_students_l528_528010

theorem average_age_students 
  (total_students : ℕ)
  (group1 : ℕ)
  (group1_avg_age : ℕ)
  (group2 : ℕ)
  (group2_avg_age : ℕ)
  (student15_age : ℕ)
  (avg_age : ℕ) 
  (h1 : total_students = 15)
  (h2 : group1_avg_age = 14)
  (h3 : group2 = 8)
  (h4 : group2_avg_age = 16)
  (h5 : student15_age = 13)
  (h6 : avg_age = (84 + 128 + 13) / 15)
  (h7 : avg_age = 15) :
  group1 = 6 :=
by sorry

end average_age_students_l528_528010


namespace remainder_q_div_x_plus_2_l528_528953

-- Define the polynomial q(x)
def q (x : ℝ) := 2 * x^4 - 3 * x^2 - 13 * x + 6

-- The main theorem we want to prove
theorem remainder_q_div_x_plus_2 :
  q 2 = 6 → (q (-2) = 52) :=
by
  intros h
  have q_2 : q 2 = 6 := h
  have q_neg2 : q (-2) = 2 * (-2)^4 - 3 * (-2)^2 - 13 * (-2) + 6 := by rfl
  rw [q_neg2]
  linarith
  sorry -- The actual proof steps would go here

end remainder_q_div_x_plus_2_l528_528953


namespace election_votes_l528_528019

theorem election_votes:
  ∀ (total_votes valid_votes votes_for_X : ℕ),
  total_votes = 560000 →
  valid_votes = (85 * total_votes) / 100 →
  votes_for_X = (75 * valid_votes) / 100 →
  votes_for_X = 357000 :=
by
  intros total_votes valid_votes votes_for_X
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end election_votes_l528_528019


namespace tangent_line_at_a_equals_3_g_monotonicity_and_extrema_l528_528756

noncomputable def f (x a : ℝ) := (x+1) * Real.log x - a * (x-1)

noncomputable def g (x a : ℝ) := f x a / (x + 1)

theorem tangent_line_at_a_equals_3 : 
  (a = 3) → 
  let f_x := f x a in
  let y := f_x in
  let f'_x := Real.log x + 1/x - 2 in
  f'_x = -1 ∧ y = 0 → 
  ∀ x y, x +  y - 1 = 0 := 
sorry

theorem g_monotonicity_and_extrema (a : ℝ) (ha : a > 1) : 
  let g_x := g x a in
  let g'_x := (x^2 + 2*(1 - a)*x + 1) / (x * (x+1)^2) in
  if ha_2 : (1 < a ∧ a ≤ 2) 
  then ∀ x > 0, g'_x ≥ 0 ∧ g_x ⟶ monotone_on (0, +∞) ∧ no_extrema g_x (0, +∞) 
  else if ha_3 : (a > 2)
  then
    let F_x := x^2 + 2*(1 - a)*x + 1 in
    let Δ := 4*a*(a-2) in
    ∀ x, 
      if x < a-1 - sqrt(Δ) ∨ x > a-1 + sqrt(Δ) then g'_x > 0 
      else if a-1 - sqrt(Δ) < x ∧ x < a-1 + sqrt(Δ) then g'_x < 0 :=
     ∀ x, 
       ascending_intervals g_x (0, a-1-sqrt(Δ)) ∧ 
       ascending_intervals g_x (a-1+sqrt(Δ), +∞) ∧ 
       descending_intervals g_x (a-1 - sqrt(Δ), a-1 + sqrt(Δ)) ∧ 
       ∃ max, ∀ x, x = a-1 - sqrt(Δ) → is_max g_x x ∧ 
       ∃ min, ∀ x, x = a-1 + sqrt(Δ) → is_min g_x x := 
sorry

end tangent_line_at_a_equals_3_g_monotonicity_and_extrema_l528_528756


namespace quadratic_function_fixed_point_l528_528737

theorem quadratic_function_fixed_point (m : ℝ) :
  ∃ A : ℝ × ℝ, A = (-1, 0) ∧ (∀ m : ℝ, ∃ x : ℝ, ∃ y : ℝ, y = -x^2 + (m-1)*x + m ∧ A = (x, y)) :=
by
  -- We will assert that A = (-1, 0)
  let A : ℝ × ℝ := (-1, 0) in
  use A,
  split,
  -- Proving that A is equal to (-1, 0)
  rfl,
  -- Now we show that for all m, point (-1, 0) lies on the graph of the function y = -x^2 + (m-1)x + m
  intro m,
  use (-1),
  use (0),
  split,
  -- Simplifying the function at x = -1
  calc
    0 = -(-1)^2 + (m - 1) * (-1) + m : by sorry
      ... = -1 + (m - 1) * (-1) + m : by sorry
      ... = -1 + - (m - 1) + m : by sorry
      ... = -1 - m + 1 + m : by sorry
      ... = 0 : by sorry,
  -- Verifying that the point (-1, 0) is (x, y)
  rfl

end quadratic_function_fixed_point_l528_528737


namespace minimum_value_of_y_l528_528307

-- Define the conditions
variables (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1)

-- Define the objective function
def y (a b : ℝ) := (a + 1/(2015 * a)) * (b + 1/(2015 * b))

-- State the theorem
theorem minimum_value_of_y : 
  y a b = (a + 1/(2015 * a)) * (b + 1/(2015 * b)) ≥ (2 * real.sqrt 2016 - 2) / 2015 :=
by sorry

end minimum_value_of_y_l528_528307


namespace integer_solutions_count_l528_528270

theorem integer_solutions_count (x : ℤ) :
  (75 ^ 60 * x ^ 60 > x ^ 120 ∧ x ^ 120 > 3 ^ 240) → ∃ n : ℕ, n = 65 :=
by
  sorry

end integer_solutions_count_l528_528270


namespace distance_from_vertex_to_incenter_l528_528799

-- Define the equilateral triangle with side length 6
structure EquilateralTriangle :=
  (A B C : ℝ)
  (side_length : ℝ)
  (is_equilateral : (A = B ∧ B = C ∧ C = A) ∧ side_length = 6)

-- Define the incenter I of the triangle
def incenter (t : EquilateralTriangle) : ℝ :=
  t.side_length * sqrt 3 / 6

-- Define the distance from the incenter to vertex B
def BI_distance (t : EquilateralTriangle) : ℝ :=
  (2 / 3) * sqrt(t.side_length ^ 2 - (t.side_length / 2) ^ 2)

-- Assert the property that needs to be proven
theorem distance_from_vertex_to_incenter {t : EquilateralTriangle} (h : EquilateralTriangle.is_equilateral t) :
  BI_distance t = 2 * sqrt 3 :=
  sorry

end distance_from_vertex_to_incenter_l528_528799


namespace find_CD_l528_528264

-- Given Definitions
variables {A B C D E : Type*}
variables (dist : A → A → ℝ)

-- Given conditions as hypotheses
variables (h_cyclic : ∀ {P Q R S T : A} (c : A), 
  (dist P Q = dist R S ∧ dist P R ∧ dist R Q ∧ dist P T ∧ dist T Q ∧ dist T R) → 
  dist P Q = dist P T)
variable (h_angle_ABC : ∀ {P Q R : A}, 90 = 90)
variable (h_AB : dist A B = 15)
variable (h_BC : dist B C = 20)
variable (h_ABCDE : dist A B = dist D E ∧ dist E A)

-- Prove CD = 7
theorem find_CD : dist C D = 7 :=
sorry

end find_CD_l528_528264


namespace required_jump_height_to_dunk_l528_528518

theorem required_jump_height_to_dunk
  (rim_height_ft : ℕ) 
  (height_above_rim_inch : ℕ) 
  (player_height_ft : ℕ) 
  (reach_above_head_inch : ℕ) : 
  (rim_height_ft = 10) → 
  (height_above_rim_inch = 6) → 
  (player_height_ft = 6) → 
  (reach_above_head_inch = 22) → 
  (rim_height_ft * 12 + height_above_rim_inch - (player_height_ft * 12 + reach_above_head_inch) = 32) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end required_jump_height_to_dunk_l528_528518


namespace pet_store_initial_puppies_l528_528608

theorem pet_store_initial_puppies
  (sold: ℕ) (cages: ℕ) (puppies_per_cage: ℕ)
  (remaining_puppies: ℕ)
  (h1: sold = 30)
  (h2: cages = 6)
  (h3: puppies_per_cage = 8)
  (h4: remaining_puppies = cages * puppies_per_cage):
  (sold + remaining_puppies) = 78 :=
by
  sorry

end pet_store_initial_puppies_l528_528608


namespace parabola_intersections_l528_528162

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ :=
  2 * x^2 - 10 * x - 10

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ :=
  x^2 - 4 * x + 6

-- Define the theorem stating the points of intersection
theorem parabola_intersections :
  ∀ (p : ℝ × ℝ), (parabola1 p.1 = p.2) ∧ (parabola2 p.1 = p.2) ↔ (p = (-2, 18) ∨ p = (8, 38)) :=
by
  sorry

end parabola_intersections_l528_528162


namespace triangle_side_a_eq_4_l528_528381

noncomputable def a_side (b c : ℝ) : ℝ :=
  let bc := b * c
  let bc_cosA := bc * (1 / 2)
  let b2_c2 := (b + c) ^ 2 - 2 * bc
  let a2 := b2_c2 - 2 * bc_cosA
  real.sqrt a2

theorem triangle_side_a_eq_4 (A : ℝ) (b c : ℝ)
  (hA : A = real.pi / 3)
  (hbc_eq : ∀ x : ℝ, x^2 - 7*x + 11 = 0 → x = b ∨ x = c)
  (hb_plus_c : b + c = 7) (hbc : b * c = 11) :
  a_side b c = 4 :=
by
  sorry

end triangle_side_a_eq_4_l528_528381


namespace square_patch_side_length_l528_528367

theorem square_patch_side_length
  (L W : ℝ)
  (hL : L = 400)
  (hW : W = 300)
  (P : ℝ)
  (hP : P = 2 * L + 2 * W)
  (A_rect : ℝ)
  (hA_rect : A_rect = L * W)
  (A_square : ℝ)
  (hA_square : A_square = 7 * A_rect) :
  ∃ s : ℝ, s = Real.sqrt A_square ∧ s ≈ 916.515 := 
by
  sorry

end square_patch_side_length_l528_528367


namespace solution_set_xf_neg_l528_528851

-- Given conditions
variables {f : ℝ → ℝ}
hypothesis odd_f : ∀ x, f(-x) = -f(x)
hypothesis decreasing_f_neg : ∀ x y, x < y → x < 0 → y < 0 → f(x) > f(y)
hypothesis f_neg2_zero : f(-2) = 0

-- To prove: the solution set for x f(x) < 0 is (-∞, -2) ∪ (2, ∞)
theorem solution_set_xf_neg : 
  {x : ℝ | x * f(x) < 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_xf_neg_l528_528851


namespace necessary_but_not_sufficient_l528_528327

theorem necessary_but_not_sufficient (α : ℝ) : (α ≠ π / 3) → (sin α ≠ √3 / 2) → (∃ β, β ≠ π / 3 ∧ sin β = √3 / 2) :=
by
  sorry

end necessary_but_not_sufficient_l528_528327


namespace angle_A_of_triangle_l528_528403

-- Given conditions and required proof
theorem angle_A_of_triangle
  (a b c : ℝ)
  (hpos : a > 0)
  (hpos2 : b > 0)
  (hpos3 : c > 0)
  (h_area : (b^2 + c^2 - a^2) = 4 * (sqrt 3 * (1 / 2 * b * c * sin (1 / 2 * b * c * (sqrt 3 * cos (1 / 2 * b * sin a)))))) :
  ∠A = π / 3 :=
begin
  sorry
end

end angle_A_of_triangle_l528_528403


namespace lcm_sum_div_lcm_even_l528_528176

open Nat

theorem lcm_sum_div_lcm_even (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    2 ∣ (lcm x y + lcm y z) / lcm x z :=
sorry

end lcm_sum_div_lcm_even_l528_528176


namespace factorial_difference_l528_528668

theorem factorial_difference :
  9! - 8! = 322560 := 
by 
  sorry

end factorial_difference_l528_528668


namespace necessary_but_not_sufficient_l528_528433

open Set

variable {α : Type*}

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient : 
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ b, b ∈ M ∧ b ∉ N) := 
by 
  sorry

end necessary_but_not_sufficient_l528_528433


namespace convex_hull_perimeter_bounds_l528_528447

noncomputable def perimeter_hex_hept_convex_hull
  (unit_circle : Type) 
  (hexagon heptagon : unit_circle → Prop) 
  (r : ℝ) 
  (h_hex : ∀ x : unit_circle, hexagon x → distance x 0 = 1) 
  (h_hept : ∀ x : unit_circle, heptagon x → distance x 0 = 1) : Prop :=
  ∃ (k : ℝ → ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 84 → 
    k x = 2 * Real.sin (π / 7) 
          + 2 * ∑ i in Finset.range 6, 
              (Real.sin ((2 * i + 1 : ℝ) * π / 84 - x) 
              + Real.sin ((2 * i + 1 : ℝ) * π / 84 + x))) ∧ 
    (6.1610929 ≤ k 0 ∧ k (π / 84) ≤ 6.1647971)

theorem convex_hull_perimeter_bounds 
  (unit_circle : Type)
  (hexagon heptagon : unit_circle → Prop)
  (r : ℝ) 
  (h_hex : ∀ x : unit_circle, hexagon x → distance x 0 = 1) 
  (h_hept : ∀ x : unit_circle, heptagon x → distance x 0 = 1) : 
  (∃ (k : ℝ → ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 84 → 
    k x = 2 * Real.sin (π / 7) 
          + 2 * ∑ i in Finset.range 6, 
              (Real.sin ((2 * i + 1 : ℝ) * π / 84 - x) 
              + Real.sin ((2 * i + 1 : ℝ) * π / 84 + x))) ∧ 
    (6.1610929 ≤ k 0 ∧ k (π / 84) ≤ 6.1647971)) :=
sorry

end convex_hull_perimeter_bounds_l528_528447


namespace weight_of_replaced_person_l528_528483

-- Given conditions
variables (avg_increase : ℝ) (new_weight : ℝ)
-- Condition values
axiom avg_increase_value : avg_increase = 1.8
axiom new_weight_value : new_weight = 79.8

-- Statement to prove
theorem weight_of_replaced_person : let W := new_weight - 6 * avg_increase in W = 69 :=
by
  -- These definitions and the proof would be given here
  have h1 : new_weight = 79.8 := by exact new_weight_value
  have h2 : avg_increase = 1.8 := by exact avg_increase_value
  let W := new_weight - 6 * avg_increase
  have hW : W = 79.8 - 6 * 1.8 := by rw [h1, h2]
  have result : W = 69 := by norm_num at hW
  exact result

end weight_of_replaced_person_l528_528483


namespace complete_square_cpjq_l528_528100

theorem complete_square_cpjq (j : ℝ) (c p q : ℝ) (h : 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) :
  c = 8 ∧ p = -3/8 ∧ q = 119/8 → q / p = -119/3 :=
by
  intros
  cases a with hc hpq
  cases hpq with hp hq
  rw [hc, hp, hq]
  have hp_ne_zero : (-3 / 8) ≠ 0 := by norm_num
  field_simp [hp_ne_zero]
  norm_num
  sorry

end complete_square_cpjq_l528_528100


namespace final_coordinates_of_C_l528_528156

-- Define the initial coordinates of point C
def C := (-1, 4 : ℤ × ℤ)

-- Function to reflect a point over the y-axis
def reflect_y (p : ℤ × ℤ) : ℤ × ℤ :=
  (-(p.1), p.2)

-- Function to reflect a point over the x-axis
def reflect_x (p : ℤ × ℤ) : ℤ × ℤ :=
  (p.1, -(p.2))

-- Expected final coordinates after reflections
def C'' := (1, -4 : ℤ × ℤ)

-- Theorem statement to prove the final coordinates
theorem final_coordinates_of_C :
  reflect_x (reflect_y C) = C'' :=
by
  sorry

end final_coordinates_of_C_l528_528156


namespace expected_square_of_length_l528_528046

/-- The given problem conditions: 
  - Right triangle with sides 3, 4, 5.
  - Each cut is along the altitude to the hypotenuse.
  - Randomly discarding one of the resulting two pieces.
  - Process repeated infinitely.
-/
noncomputable def calc_expected_length : ℝ := (12 / 5) + (1 / 2) * (3 / 5 + 4 / 5) * calc_expected_length

theorem expected_square_of_length :
  let E := calc_expected_length in E * E = 64 :=
by
  let E := calc_expected_length
  have h : E = 8 := sorry
  sorry

end expected_square_of_length_l528_528046


namespace perpendicular_radical_axis_OM_PQ_l528_528807

variables {A B C D P Q M O : Point}
variable [MetricSpace Point]

-- Definitions of conditions
noncomputable def cyclic_quadrilateral (A B C D : Point) (O : Point) : Prop :=
  ∃ (circle : set Point), O ∈ circle ∧ A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle

noncomputable def extension_intersect (A B C D P Q : Point) : Prop :=
  is_collinear A B P ∧ is_collinear C D P ∧ is_collinear A D Q ∧ is_collinear B C Q

noncomputable def circumcircle_intersect_at_two_points (P B C Q D : Point) (M O : Point) : Prop :=
  ∃ (circle_PBC circle_QCD : set Point),
    M ∈ circle_PBC ∧ M ∈ circle_QCD ∧ C ∈ circle_PBC ∧ C ∈ circle_QCD ∧
    ∀ x : Point, x ∈ circle_PBC → dist x O = dist C O →
    ∀ y : Point, y ∈ circle_QCD → dist y O = dist C O

-- The goal to prove
theorem perpendicular_radical_axis_OM_PQ
  (ABCD_inscribed : cyclic_quadrilateral A B C D O)
  (extension_points : extension_intersect A B C D P Q)
  (circumcircles_intersect : circumcircle_intersect_at_two_points P B C Q D M O) :
  is_perpendicular (line_through O M) (line_through P Q) :=
sorry

end perpendicular_radical_axis_OM_PQ_l528_528807


namespace percentage_decrease_l528_528892

theorem percentage_decrease (x y k : ℝ) (h : x^3 * y = k) (hx : x' = 1.2 * x) :
  100 * (1 - y' / y) = 42.13 :=
by
  -- define the new x value as x' = 1.2 * x
  let x' := 1.2 * x
  -- calculate the new y value using the given condition and new x value
  let y' := k / (x'^3)
  calculate the percentage decrease
  sorry

end percentage_decrease_l528_528892


namespace ellipse_condition_suff_nec_l528_528579

theorem ellipse_condition_suff_nec (m n : ℝ) (h1 : n > m) (h2 : m > 0) (h3 : n > 0) :
  (mx^2 + ny^2 = 1) ↔ (n > m > 0) := 
sorry

end ellipse_condition_suff_nec_l528_528579


namespace quadractic_eqn_num_solutions_l528_528915

theorem quadractic_eqn_num_solutions :
  (x : ℝ) | x^2 - abs x - 6 = 0 → 2 :=
begin
  sorry
end

end quadractic_eqn_num_solutions_l528_528915


namespace volume_of_cuboctahedron_l528_528740

def points (i j : ℕ) (A : ℕ → ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x0, y0, z0) := A 0
  let (xi, yi, zi) := A i
  let (xj, yj, zj) := A j
  (xi - xj, yi - yj, zi - zj)

def is_cuboctahedron (points_set : Set (ℝ × ℝ × ℝ)) : Prop :=
  -- Insert specific conditions that define a cuboctahedron
  sorry

theorem volume_of_cuboctahedron : 
  let A := fun 
    | 0 => (0, 0, 0)
    | 1 => (1, 0, 0)
    | 2 => (0, 1, 0)
    | 3 => (0, 0, 1)
    | _ => (0, 0, 0)
  let P_ij := 
    {p | ∃ i j : ℕ, i ≠ j ∧ p = points i j A}
  ∃ v : ℝ, is_cuboctahedron P_ij ∧ v = 10 / 3 :=
sorry

end volume_of_cuboctahedron_l528_528740


namespace total_points_l528_528562

theorem total_points (a b c : ℕ) (ha : a = 2) (hb : b = 9) (hc : c = 4) : a + b + c = 15 :=
by
  rw [ha, hb, hc]
  norm_num
  sorry

end total_points_l528_528562


namespace beau_age_today_l528_528657

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end beau_age_today_l528_528657


namespace penny_dime_halfdollar_probability_l528_528894

-- Define the universe of outcomes for five coin flips
def coin_flip_outcomes : Finset (vector Bool 5) := Finset.univ

-- Define a predicate that checks if the penny, dime, and half-dollar come up the same
def same_penny_dime_halfdollar (v : vector Bool 5) : Prop :=
  v.head = v.nth 2 ∧ v.head = v.nth 4

-- The proof problem: prove that the probability of the penny, dime, and half-dollar being the same is 1/4
theorem penny_dime_halfdollar_probability :
  (Finset.filter same_penny_dime_halfdollar coin_flip_outcomes).card / coin_flip_outcomes.card = 1 / 4 :=
by
  sorry

end penny_dime_halfdollar_probability_l528_528894


namespace distance_after_3_minutes_l528_528642

/-- Let truck_speed be 65 km/h and car_speed be 85 km/h.
    Let time_in_minutes be 3 and converted to hours it is 0.05 hours.
    The goal is to prove that the distance between the truck and the car
    after 3 minutes is 1 kilometer. -/
def truck_speed : ℝ := 65 -- speed in km/h
def car_speed : ℝ := 85 -- speed in km/h
def time_in_minutes : ℝ := 3 -- time in minutes
def time_in_hours : ℝ := time_in_minutes / 60 -- converted time in hours
def distance_truck := truck_speed * time_in_hours
def distance_car := car_speed * time_in_hours
def distance_between : ℝ := distance_car - distance_truck

theorem distance_after_3_minutes : distance_between = 1 := by
  -- Proof steps would go here
  sorry

end distance_after_3_minutes_l528_528642


namespace find_a1_l528_528039

noncomputable def geometric_sequence := ℕ → ℝ

theorem find_a1 (a : geometric_sequence) (q : ℝ) (S5 : ℝ) 
  (h1 : q = -2) (h2 : S5 = 44)
  (h3 : ∀ n, a (n + 1) = a n * q) 
  (h4 : S5 = (1 - a 0 * q ^ 5) / (1 - q)) :
  a 0 = 4 :=
by
  sorry

end find_a1_l528_528039


namespace distance_between_truck_and_car_l528_528631

noncomputable def speed_truck : ℝ := 65
noncomputable def speed_car : ℝ := 85
noncomputable def time : ℝ := 3 / 60

theorem distance_between_truck_and_car : 
  let Distance_truck := speed_truck * time,
      Distance_car := speed_car * time in
  Distance_car - Distance_truck = 1 :=
by {
  sorry
}

end distance_between_truck_and_car_l528_528631


namespace part_one_solution_part_two_solution_l528_528356

variables {x m : ℝ}

theorem part_one_solution (h : m = 1) : 
  (2 - x) / 2 > x / 2 - 1 ↔ x < 2 :=
by sorry

theorem part_two_solution (h : m ≠ -1) : 
  let q := 2 - m * x 
  in q / 2 > x / 2 - 1 ↔
    if m > -1 then x < 2
    else x > 2 :=
by sorry

end part_one_solution_part_two_solution_l528_528356


namespace isosceles_triangle_chord_length_l528_528318

noncomputable def length_of_chord_intercepted_by_altitude (ABC : Triangle) (radius : ℝ) (isosceles : IsIsosceles ABC) (circumradius_one : CircumscribedRadius ABC = 1) : ℝ :=
  sorry

theorem isosceles_triangle_chord_length (ABC : Triangle) (isosceles : IsIsosceles ABC) (circumradius_one : CircumscribedRadius ABC = 1) :
  length_of_chord_intercepted_by_altitude ABC 1 isosceles circumradius_one = 3/2 :=
by sorry

end isosceles_triangle_chord_length_l528_528318


namespace box_office_growth_l528_528131

theorem box_office_growth (x : ℝ) :
  2.5 * (1 + x) ^ 2 = 3.6 ↔ x = sqrt(3.6 / 2.5) - 1 := 
sorry

end box_office_growth_l528_528131


namespace jersey_cost_difference_l528_528480

theorem jersey_cost_difference :
  let jersey_cost := 115
  let tshirt_cost := 25
  jersey_cost - tshirt_cost = 90 :=
by
  -- proof goes here
  sorry

end jersey_cost_difference_l528_528480


namespace opposite_neg_two_is_two_l528_528512

theorem opposite_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_neg_two_is_two_l528_528512


namespace speed_of_mother_minimum_running_time_l528_528877

namespace XiaotongTravel

def distance_to_binjiang : ℝ := 4320
def time_diff : ℝ := 12
def speed_rate : ℝ := 1.2

theorem speed_of_mother : 
  ∃ (x : ℝ), (distance_to_binjiang / x - distance_to_binjiang / (speed_rate * x) = time_diff) → (speed_rate * x = 72) :=
sorry

def distance_to_company : ℝ := 2940
def running_speed : ℝ := 150
def total_time : ℝ := 30

theorem minimum_running_time :
  ∃ (y : ℝ), ((distance_to_company - running_speed * y) / 72 + y ≤ total_time) → (y ≥ 10) :=
sorry

end XiaotongTravel

end speed_of_mother_minimum_running_time_l528_528877


namespace power_function_properties_l528_528742

theorem power_function_properties :
  (∀ n : ℝ, (n = -1 → ¬∀ x : ℝ, x > 0 → y = x^n → StrictMonoDecr_on y (Ioi 0)) ) ∧
  (∀ n : ℝ, n > 0 → y = x^n → (∃ x : ℝ, y x = 0) ∧ (y 1 = 1) → False ) ∧
  (∀ n : ℝ, (∃ x : ℝ, (x, x^n) ∈ Ioi 0 × Iio 0) → False) ∧
  (∀ n : ℝ, y = x^n → ∀ x : ℝ, x ∈ Ioi 1 → StrictMonoDecr_on y (Ioi 1) → n < 0 ) :=
by
  split
  · intro n h
    sorry -- Proof that (1) is incorrect
  · split
    · intro n h1 h2
      sorry -- Proof that (2) is incorrect
    · split
      · intro n h
        sorry -- Proof that (3) is correct
      · intro n h x h1 h2
        sorry -- Proof that (4) is correct

end power_function_properties_l528_528742


namespace circle_construction_tangent_l528_528680

variables {A B : Point} {S : Circle}

theorem circle_construction_tangent (A B : Point) (S : Circle) :
  let B_star := inv A B in
  let S_star := inv_circle A S in
  if B_star ∈ S_star then
    ∃! C : Circle, C ∋ A ∧ C ∋ B ∧ tangent C S
  else if outside B_star S_star then
    ∃ C1 C2 : Circle, C1 ∋ A ∧ C1 ∋ B ∧ tangent C1 S ∧ C2 ∋ A ∧ C2 ∋ B ∧ tangent C2 S ∧ C1 ≠ C2
  else
    ¬ ∃ C : Circle, C ∋ A ∧ C ∋ B ∧ tangent C S :=
sorry

end circle_construction_tangent_l528_528680


namespace part1_part2_1_part2_2_l528_528352

-- Definitions of the conditions
def f (x : ℝ) : ℝ := real.sqrt (3 - x) + real.log (x + 2) / real.log 4

def A : set ℝ := {x | -2 < x ∧ x ≤ 3}

def B : set ℝ := {x | x > 2 ∨ x < -3}

-- Statements to prove
theorem part1 : A = {x | -2 < x ∧ x ≤ 3} := sorry

theorem part2_1 : A ∩ B = {x | 2 < x ∧ x ≤ 3} := sorry

theorem part2_2 : (set.compl A) ∪ B = {x | x ≤ -2 ∨ x > 2} := sorry

end part1_part2_1_part2_2_l528_528352


namespace age_of_25th_student_l528_528481

-- Definitions derived from problem conditions
def averageAgeClass (totalAge : ℕ) (totalStudents : ℕ) : ℕ := totalAge / totalStudents
def totalAgeGivenAverage (numStudents : ℕ) (averageAge : ℕ) : ℕ := numStudents * averageAge

-- Given conditions
def totalAgeOfAllStudents := 25 * 24
def totalAgeOf8Students := totalAgeGivenAverage 8 22
def totalAgeOf10Students := totalAgeGivenAverage 10 20
def totalAgeOf6Students := totalAgeGivenAverage 6 28
def totalAgeOf24Students := totalAgeOf8Students + totalAgeOf10Students + totalAgeOf6Students

-- The proof that the age of the 25th student is 56 years
theorem age_of_25th_student : totalAgeOfAllStudents - totalAgeOf24Students = 56 := by
  sorry

end age_of_25th_student_l528_528481


namespace proof_2012_is_leap_year_proof_first_quarter_days_proof_years_since_1949_l528_528976

/- Prove that 2012 is a leap year -/
theorem proof_2012_is_leap_year : (2012 % 4 = 0 ∧ (2012 % 100 ≠ 0 ∨ 2012 % 400 = 0)) :=
by {
  have h : 2012 % 4 = 0 := by norm_num,
  have h2 : (2012 % 100 ≠ 0 ∨ 2012 % 400 = 0) := by norm_num,
  exact ⟨h, h2⟩
}

/- Given that 2012 is a leap year, prove that the first quarter of 2012 has 91 days -/
theorem proof_first_quarter_days (h : 2012 % 4 = 0 ∧ (2012 % 100 ≠ 0 ∨ 2012 % 400 = 0)) : 
  (leap_year_days := 31 * 2 + 29) = 91 := 
by {
  have jan_days : 31 := by norm_num,
  have feb_days : 29 := by norm_num,
  have mar_days : 31 := by norm_num,
  have quarter_days := jan_days + feb_days + mar_days,
  exact by norm_num 31 * 2 + 29
}

/- Prove that by October 1, 2012, the People's Republic of China has been established for 63 years -/
theorem proof_years_since_1949 : 2012 - 1949 = 63 := by norm_num

end proof_2012_is_leap_year_proof_first_quarter_days_proof_years_since_1949_l528_528976


namespace speed_of_goods_train_l528_528997

/-!
# Problem:
A goods train runs at a certain speed and crosses a 230 m long platform in 26 seconds.
The length of the goods train is 290.04 m.
What is the speed of the goods train in kmph?

# Solution:
The speed of the goods train is 72 km/h.
-/

def length_of_train := 290.04
def length_of_platform := 230
def time_to_cross := 26
def total_distance := length_of_train + length_of_platform
def speed_m_s := total_distance / time_to_cross
def speed_kmh := speed_m_s * 3.6
def expected_speed_kmh := 72

theorem speed_of_goods_train : speed_kmh = expected_speed_kmh :=
by
  sorry

end speed_of_goods_train_l528_528997


namespace combined_weight_l528_528371

noncomputable def Jake_weight : ℕ := 196
noncomputable def Kendra_weight : ℕ := 94

-- Condition: If Jake loses 8 pounds, he will weigh twice as much as Kendra
axiom lose_8_pounds (j k : ℕ) : (j - 8 = 2 * k) → j = Jake_weight → k = Kendra_weight

-- To Prove: The combined weight of Jake and Kendra is 290 pounds
theorem combined_weight (j k : ℕ) (h₁ : j = Jake_weight) (h₂ : k = Kendra_weight) : j + k = 290 := 
by  sorry

end combined_weight_l528_528371
