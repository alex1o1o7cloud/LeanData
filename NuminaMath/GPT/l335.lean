import Mathlib

namespace function_has_two_zeros_for_a_eq_2_l335_33548

noncomputable def f (a x : ℝ) : ℝ := a ^ x - x - 1

theorem function_has_two_zeros_for_a_eq_2 :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f 2 x1 = 0 ∧ f 2 x2 = 0) := sorry

end function_has_two_zeros_for_a_eq_2_l335_33548


namespace right_triangle_third_side_l335_33597

theorem right_triangle_third_side (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) : c = Real.sqrt (b^2 - a^2) :=
by
  rw [h1, h2]
  sorry

end right_triangle_third_side_l335_33597


namespace freshmen_sophomores_without_pets_l335_33550

theorem freshmen_sophomores_without_pets : 
  let total_students := 400
  let percentage_freshmen_sophomores := 0.50
  let percentage_with_pets := 1/5
  let freshmen_sophomores := percentage_freshmen_sophomores * total_students
  160 = (freshmen_sophomores - (percentage_with_pets * freshmen_sophomores)) :=
by
  sorry

end freshmen_sophomores_without_pets_l335_33550


namespace minimum_value_of_y_l335_33581

theorem minimum_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 36 * y) : y ≥ -7 :=
sorry

end minimum_value_of_y_l335_33581


namespace alex_seashells_l335_33522

theorem alex_seashells (mimi_seashells kyle_seashells leigh_seashells alex_seashells : ℕ) 
    (h1 : mimi_seashells = 2 * 12) 
    (h2 : kyle_seashells = 2 * mimi_seashells) 
    (h3 : leigh_seashells = kyle_seashells / 3) 
    (h4 : alex_seashells = 3 * leigh_seashells) : 
  alex_seashells = 48 := by
  sorry

end alex_seashells_l335_33522


namespace integer_solutions_eq_l335_33586

theorem integer_solutions_eq :
  { (x, y) : ℤ × ℤ | 2 * x ^ 4 - 4 * y ^ 4 - 7 * x ^ 2 * y ^ 2 - 27 * x ^ 2 + 63 * y ^ 2 + 85 = 0 }
  = { (3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3) } :=
by sorry

end integer_solutions_eq_l335_33586


namespace min_frac_sum_min_frac_sum_achieved_l335_33547

theorem min_frac_sum (a b : ℝ) (h₁ : 2 * a + 3 * b = 6) (h₂ : 0 < a) (h₃ : 0 < b) :
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

theorem min_frac_sum_achieved :
  (2 / (6 / 5) + 3 / (6 / 5)) = 25 / 6 :=
by sorry


end min_frac_sum_min_frac_sum_achieved_l335_33547


namespace sum_of_side_lengths_l335_33511

theorem sum_of_side_lengths (p q r : ℕ) (h : p = 8 ∧ q = 1 ∧ r = 5) 
    (area_ratio : 128 / 50 = 64 / 25) 
    (side_length_ratio : 8 / 5 = Real.sqrt (128 / 50)) :
    p + q + r = 14 := 
by 
  sorry

end sum_of_side_lengths_l335_33511


namespace range_of_a_l335_33504

theorem range_of_a (M N : Set ℝ) (a : ℝ) 
(hM : M = {x : ℝ | x < 2}) 
(hN : N = {x : ℝ | x < a}) 
(hSubset : M ⊆ N) : 
  2 ≤ a := 
sorry

end range_of_a_l335_33504


namespace angle_AFE_is_80_degrees_l335_33520

-- Defining the setup and given conditions
def point := ℝ × ℝ  -- defining a 2D point
noncomputable def A : point := (0, 0)
noncomputable def B : point := (1, 0)
noncomputable def C : point := (1, 1)
noncomputable def D : point := (0, 1)
noncomputable def E : point := (-1, 1.732)  -- Place E such that angle CDE ≈ 130 degrees

-- Conditions
def angle_CDE := 130
def DF_over_DE := 2  -- DF = 2 * DE
noncomputable def F : point := (0.5, 1)  -- This is an example position; real positioning depends on more details

-- Proving that the angle AFE is 80 degrees
theorem angle_AFE_is_80_degrees :
  ∃ (AFE : ℝ), AFE = 80 := sorry

end angle_AFE_is_80_degrees_l335_33520


namespace rectangle_area_perimeter_l335_33562

-- Defining the problem conditions
def positive_int (n : Int) : Prop := n > 0

-- The main statement of the problem
theorem rectangle_area_perimeter (a b : Int) (h1 : positive_int a) (h2 : positive_int b) : 
  ¬ (a + 2) * (b + 2) - 4 = 146 :=
by
  sorry

end rectangle_area_perimeter_l335_33562


namespace at_least_one_fraction_less_than_two_l335_33567

theorem at_least_one_fraction_less_than_two {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
by
  sorry

end at_least_one_fraction_less_than_two_l335_33567


namespace geometric_sequence_sum_l335_33523

-- Define the geometric sequence {a_n}
def a (n : ℕ) : ℕ := 2 * (1 ^ (n - 1))

-- Define the sum of the first n terms, s_n
def s (n : ℕ) : ℕ := (Finset.range n).sum (a)

-- The transformed sequence {a_n + 1} assumed also geometric
def b (n : ℕ) : ℕ := a n + 1

-- Lean theorem that s_n = 2n
theorem geometric_sequence_sum (n : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, (b (n + 1)) * (b (n + 1)) = (b n * b (n + 2))) : 
  s n = 2 * n :=
sorry

end geometric_sequence_sum_l335_33523


namespace arrangement_count_l335_33587

-- Given conditions
def num_basketballs : ℕ := 5
def num_volleyballs : ℕ := 3
def num_footballs : ℕ := 2
def total_balls : ℕ := num_basketballs + num_volleyballs + num_footballs

-- Way to calculate the permutations of multiset
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Proof statement
theorem arrangement_count : 
  factorial total_balls / (factorial num_basketballs * factorial num_volleyballs * factorial num_footballs) = 2520 :=
by
  sorry

end arrangement_count_l335_33587


namespace jenny_jellybeans_original_l335_33590

theorem jenny_jellybeans_original (x : ℝ) 
  (h : 0.75^3 * x = 45) : x = 107 := 
sorry

end jenny_jellybeans_original_l335_33590


namespace cost_difference_is_35_88_usd_l335_33578

/-
  Mr. Llesis bought 50 kilograms of rice at different prices per kilogram from various suppliers.
  He bought:
  - 15 kilograms at €1.2 per kilogram from Supplier A
  - 10 kilograms at €1.4 per kilogram from Supplier B
  - 12 kilograms at €1.6 per kilogram from Supplier C
  - 8 kilograms at €1.9 per kilogram from Supplier D
  - 5 kilograms at €2.3 per kilogram from Supplier E

  He kept 7/10 of the total rice in storage and gave the rest to Mr. Everest.
  The current conversion rate is €1 = $1.15.
  
  Prove that the difference in cost in US dollars between the rice kept and the rice given away is $35.88.
-/

def euros_to_usd (euros : ℚ) : ℚ :=
  euros * (115 / 100)

def total_cost : ℚ := 
  (15 * 1.2) + (10 * 1.4) + (12 * 1.6) + (8 * 1.9) + (5 * 2.3)

def cost_kept : ℚ := (7/10) * total_cost
def cost_given : ℚ := (3/10) * total_cost

theorem cost_difference_is_35_88_usd :
  euros_to_usd cost_kept - euros_to_usd cost_given = 35.88 := 
sorry

end cost_difference_is_35_88_usd_l335_33578


namespace value_of_x_l335_33527

-- Define the custom operation * for the problem
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Define the main problem statement
theorem value_of_x (x : ℝ) (h : star 3 (star 7 x) = 5) : x = 49 / 4 :=
by
  have h7x : star 7 x = 28 - 2 * x := by sorry  -- Derived from the definitions
  have h3star7x : star 3 (28 - 2 * x) = -44 + 4 * x := by sorry  -- Derived from substituting star 7 x
  sorry

end value_of_x_l335_33527


namespace fit_seven_rectangles_l335_33532

theorem fit_seven_rectangles (s : ℝ) (a : ℝ) : (s > 0) → (a > 0) → (14 * a ^ 2 ≤ s ^ 2 ∧ 2 * a ≤ s) → 
  (∃ (rectangles : Fin 7 → (ℝ × ℝ)), ∀ i, rectangles i = (a, 2 * a) ∧
   ∀ i j, i ≠ j → rectangles i ≠ rectangles j) :=
sorry

end fit_seven_rectangles_l335_33532


namespace hearing_aid_cost_l335_33556

theorem hearing_aid_cost
  (cost : ℝ)
  (insurance_coverage : ℝ)
  (personal_payment : ℝ)
  (total_aid_count : ℕ)
  (h : total_aid_count = 2)
  (h_insurance : insurance_coverage = 0.80)
  (h_personal_payment : personal_payment = 1000)
  (h_equation : personal_payment = (1 - insurance_coverage) * (total_aid_count * cost)) :
  cost = 2500 :=
by
  sorry

end hearing_aid_cost_l335_33556


namespace runners_never_meet_l335_33598

theorem runners_never_meet
    (x : ℕ)  -- Speed of first runner
    (a : ℕ)  -- 1/3 of the circumference of the track
    (C : ℕ)  -- Circumference of the track
    (hC : C = 3 * a)  -- Given that C = 3 * a
    (h_speeds : 1 * x = x ∧ 2 * x = 2 * x ∧ 4 * x = 4 * x)  -- Speed ratios: 1:2:4
    (t : ℕ)  -- Time variable
: ¬(∃ t, (x * t % C = 2 * x * t % C ∧ 2 * x * t % C = 4 * x * t % C)) :=
by sorry

end runners_never_meet_l335_33598


namespace cassie_nail_cutting_l335_33594

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l335_33594


namespace ratio_average_speed_l335_33543

-- Define the conditions based on given distances and times
def distanceAB : ℕ := 600
def timeAB : ℕ := 3

def distanceBD : ℕ := 540
def timeBD : ℕ := 2

def distanceAC : ℕ := 460
def timeAC : ℕ := 4

def distanceCE : ℕ := 380
def timeCE : ℕ := 3

-- Define the total distances and times for Eddy and Freddy
def distanceEddy : ℕ := distanceAB + distanceBD
def timeEddy : ℕ := timeAB + timeBD

def distanceFreddy : ℕ := distanceAC + distanceCE
def timeFreddy : ℕ := timeAC + timeCE

-- Define the average speeds for Eddy and Freddy
def averageSpeedEddy : ℚ := distanceEddy / timeEddy
def averageSpeedFreddy : ℚ := distanceFreddy / timeFreddy

-- Prove the ratio of their average speeds is 19:10
theorem ratio_average_speed (h1 : distanceAB = 600) (h2 : timeAB = 3) 
                           (h3 : distanceBD = 540) (h4 : timeBD = 2)
                           (h5 : distanceAC = 460) (h6 : timeAC = 4) 
                           (h7 : distanceCE = 380) (h8 : timeCE = 3):
  averageSpeedEddy / averageSpeedFreddy = 19 / 10 := by sorry

end ratio_average_speed_l335_33543


namespace team_CB_days_worked_together_l335_33542

def projectA := 1 -- Project A is 1 unit of work
def projectB := 5 / 4 -- Project B is 1.25 units of work
def work_rate_A := 1 / 20 -- Team A's work rate
def work_rate_B := 1 / 24 -- Team B's work rate
def work_rate_C := 1 / 30 -- Team C's work rate

noncomputable def combined_rate_without_C := work_rate_B + work_rate_C

noncomputable def combined_total_work := projectA + projectB

noncomputable def days_for_combined_work := combined_total_work / combined_rate_without_C

-- Statement to prove the number of days team C and team B worked together
theorem team_CB_days_worked_together : 
  days_for_combined_work = 15 := 
  sorry

end team_CB_days_worked_together_l335_33542


namespace compare_31_17_compare_33_63_compare_82_26_compare_29_80_l335_33509

-- Definition and proof obligation for each comparison

theorem compare_31_17 : 31^11 < 17^14 := sorry

theorem compare_33_63 : 33^75 > 63^60 := sorry

theorem compare_82_26 : 82^33 > 26^44 := sorry

theorem compare_29_80 : 29^31 > 80^23 := sorry

end compare_31_17_compare_33_63_compare_82_26_compare_29_80_l335_33509


namespace female_athletes_in_sample_l335_33505

theorem female_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ)
  (total_athletes_eq : total_athletes = 98)
  (male_athletes_eq : male_athletes = 56)
  (sample_size_eq : sample_size = 28)
  : (sample_size * (total_athletes - male_athletes) / total_athletes) = 12 :=
by
  sorry

end female_athletes_in_sample_l335_33505


namespace perimeter_of_square_B_l335_33515

theorem perimeter_of_square_B
  (perimeter_A : ℝ)
  (h_perimeter_A : perimeter_A = 36)
  (area_ratio : ℝ)
  (h_area_ratio : area_ratio = 1 / 3)
  : ∃ (perimeter_B : ℝ), perimeter_B = 12 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_B_l335_33515


namespace relationship_y1_y2_y3_l335_33510

theorem relationship_y1_y2_y3 :
  ∀ (y1 y2 y3 : ℝ), y1 = 6 ∧ y2 = 3 ∧ y3 = -2 → y1 > y2 ∧ y2 > y3 :=
by 
  intros y1 y2 y3 h
  sorry

end relationship_y1_y2_y3_l335_33510


namespace normal_complaints_calculation_l335_33557

-- Define the normal number of complaints
def normal_complaints (C : ℕ) : ℕ := C

-- Define the complaints when short-staffed
def short_staffed_complaints (C : ℕ) : ℕ := (4 * C) / 3

-- Define the complaints when both conditions are met
def both_conditions_complaints (C : ℕ) : ℕ := (4 * C) / 3 + (4 * C) / 15

-- Main statement to prove
theorem normal_complaints_calculation (C : ℕ) (h : 3 * (both_conditions_complaints C) = 576) : C = 120 :=
by sorry

end normal_complaints_calculation_l335_33557


namespace problem_solution_l335_33585

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = 7 := 
by
  sorry

end problem_solution_l335_33585


namespace intersection_of_M_and_N_l335_33516

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := sorry

end intersection_of_M_and_N_l335_33516


namespace red_balls_removed_to_certain_event_l335_33517

theorem red_balls_removed_to_certain_event (total_balls red_balls yellow_balls : ℕ) (m : ℕ)
  (total_balls_eq : total_balls = 8)
  (red_balls_eq : red_balls = 3)
  (yellow_balls_eq : yellow_balls = 5)
  (certain_event_A : ∀ remaining_red_balls remaining_yellow_balls,
    remaining_red_balls = red_balls - m → remaining_yellow_balls = yellow_balls →
    remaining_red_balls = 0) : m = 3 :=
by
  sorry

end red_balls_removed_to_certain_event_l335_33517


namespace min_selling_price_is_400_l335_33570

-- Definitions for the problem conditions
def total_products := 20
def average_price := 1200
def less_than_1000_count := 10
def price_of_most_expensive := 11000
def total_retail_price := total_products * average_price

-- The theorem to state the problem condition and the expected result
theorem min_selling_price_is_400 (x : ℕ) :
  -- Condition 1: Total retail price
  total_retail_price =
  -- 10 products sell for x dollars
  (10 * x) +
  -- 9 products sell for 1000 dollars
  (9 * 1000) +
  -- 1 product sells for the maximum price 11000
  price_of_most_expensive → 
  -- Conclusion: The minimum price x is 400
  x = 400 :=
by
  sorry

end min_selling_price_is_400_l335_33570


namespace pyramid_volume_correct_l335_33546

noncomputable def PyramidVolume (base_area : ℝ) (triangle_area_1 : ℝ) (triangle_area_2 : ℝ) : ℝ :=
  let side := Real.sqrt base_area
  let height_1 := (2 * triangle_area_1) / side
  let height_2 := (2 * triangle_area_2) / side
  let h_sq := height_1 ^ 2 - (Real.sqrt (height_1 ^ 2 + height_2 ^ 2 - 512)) ^ 2
  let height := Real.sqrt h_sq
  (1/3) * base_area * height

theorem pyramid_volume_correct :
  PyramidVolume 256 120 112 = 1163 := by
  sorry

end pyramid_volume_correct_l335_33546


namespace payback_time_l335_33535

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end payback_time_l335_33535


namespace locus_of_Y_right_angled_triangle_l335_33545

-- Conditions definitions
variables {A B C : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (b c m : ℝ) -- Coordinates and slopes related to the problem
variables (x : ℝ) -- Independent variable for the locus line

-- The problem statement
theorem locus_of_Y_right_angled_triangle 
  (A_right_angle : ∀ (α β : ℝ), α * β = 0) 
  (perpendicular_lines : b ≠ m * c) 
  (no_coincide : (b^2 * m - 2 * b * c - c^2 * m) ≠ 0) :
  ∃ (y : ℝ), y = (2 * b * c * (b * m - c) - x * (b^2 + 2 * b * c * m - c^2)) / (b^2 * m - 2 * b * c - c^2 * m) := 
sorry

end locus_of_Y_right_angled_triangle_l335_33545


namespace general_term_a_n_l335_33584

theorem general_term_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (2/3) * a n + 1/3) :
  ∀ n, a n = (-2)^(n-1) :=
sorry

end general_term_a_n_l335_33584


namespace mike_worked_four_hours_l335_33525

-- Define the time to perform each task in minutes
def time_wash_car : ℕ := 10
def time_change_oil : ℕ := 15
def time_change_tires : ℕ := 30

-- Define the number of tasks Mike performed
def num_wash_cars : ℕ := 9
def num_change_oil : ℕ := 6
def num_change_tires : ℕ := 2

-- Define the total minutes Mike worked
def total_minutes_worked : ℕ :=
  (num_wash_cars * time_wash_car) +
  (num_change_oil * time_change_oil) +
  (num_change_tires * time_change_tires)

-- Define the conversion from minutes to hours
def total_hours_worked : ℕ := total_minutes_worked / 60

-- Formalize the proof statement
theorem mike_worked_four_hours :
  total_hours_worked = 4 :=
by
  sorry

end mike_worked_four_hours_l335_33525


namespace washing_machine_capacity_l335_33551

def num_shirts : Nat := 19
def num_sweaters : Nat := 8
def num_loads : Nat := 3

theorem washing_machine_capacity :
  (num_shirts + num_sweaters) / num_loads = 9 := by
  sorry

end washing_machine_capacity_l335_33551


namespace modulus_of_complex_l335_33507

open Complex

theorem modulus_of_complex (z : ℂ) (h : z = 1 - (1 / Complex.I)) : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l335_33507


namespace tim_total_trip_time_l335_33541

theorem tim_total_trip_time (drive_time : ℕ) (traffic_multiplier : ℕ) (drive_time_eq : drive_time = 5) (traffic_multiplier_eq : traffic_multiplier = 2) :
  drive_time + drive_time * traffic_multiplier = 15 :=
by
  sorry

end tim_total_trip_time_l335_33541


namespace max_ab_l335_33574

theorem max_ab (a b : ℝ) (h : 4 * a + b = 1) (ha : a > 0) (hb : b > 0) : ab <= 1 / 16 :=
sorry

end max_ab_l335_33574


namespace monotonicity_f_inequality_proof_l335_33519

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem monotonicity_f :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (x + ε)) ∧ (∀ x : ℝ, 1 < x → f (x - ε) > f x) := 
sorry

theorem inequality_proof (x : ℝ) (hx : 1 < x) :
  1 < (x - 1) / Real.log x ∧ (x - 1) / Real.log x < x :=
sorry

end monotonicity_f_inequality_proof_l335_33519


namespace solve_for_x_l335_33561

theorem solve_for_x (x : ℝ) (h : 8 / x + 6 = 8) : x = 4 :=
sorry

end solve_for_x_l335_33561


namespace Carrie_pays_94_l335_33571

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l335_33571


namespace bryan_total_earnings_l335_33518

-- Declare the data given in the problem:
def num_emeralds : ℕ := 3
def num_rubies : ℕ := 2
def num_sapphires : ℕ := 3

def price_emerald : ℝ := 1785
def price_ruby : ℝ := 2650
def price_sapphire : ℝ := 2300

-- Calculate the total earnings from each type of stone:
def total_emeralds : ℝ := num_emeralds * price_emerald
def total_rubies : ℝ := num_rubies * price_ruby
def total_sapphires : ℝ := num_sapphires * price_sapphire

-- Calculate the overall total earnings:
def total_earnings : ℝ := total_emeralds + total_rubies + total_sapphires

-- Prove that Bryan got 17555 dollars in total:
theorem bryan_total_earnings : total_earnings = 17555 := by
  simp [total_earnings, total_emeralds, total_rubies, total_sapphires, num_emeralds, num_rubies, num_sapphires, price_emerald, price_ruby, price_sapphire]
  sorry

end bryan_total_earnings_l335_33518


namespace find_two_digit_number_l335_33599

theorem find_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100) 
                              (h2 : N % 2 = 0) (h3 : N % 11 = 0) 
                              (h4 : ∃ k : ℕ, (N / 10) * (N % 10) = k^3) :
  N = 88 :=
by {
  sorry
}

end find_two_digit_number_l335_33599


namespace xiaobo_probability_not_home_l335_33568

theorem xiaobo_probability_not_home :
  let r1 := 1 / 2
  let r2 := 1 / 4
  let area_circle := Real.pi
  let area_greater_r1 := area_circle * (1 - r1^2)
  let area_less_r2 := area_circle * r2^2
  let area_favorable := area_greater_r1 + area_less_r2
  let probability_not_home := area_favorable / area_circle
  probability_not_home = 13 / 16 := by
  sorry

end xiaobo_probability_not_home_l335_33568


namespace cream_ratio_l335_33500

theorem cream_ratio (j : ℝ) (jo : ℝ) (jc : ℝ) (joc : ℝ) (jdrank : ℝ) (jodrank : ℝ) :
  j = 15 ∧ jo = 15 ∧ jc = 3 ∧ joc = 2.5 ∧ jdrank = 0 ∧ jodrank = 0.5 →
  j + jc - jdrank = jc ∧ jo + jc - jodrank = joc →
  (jc / joc) = (6 / 5) :=
  by
  sorry

end cream_ratio_l335_33500


namespace nat_solutions_l335_33573

open Nat

theorem nat_solutions (a b c : ℕ) :
  (a ≤ b ∧ b ≤ c ∧ ab + bc + ca = 2 * (a + b + c)) ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 4) :=
by sorry

end nat_solutions_l335_33573


namespace flowers_died_l335_33553

theorem flowers_died : 
  let initial_flowers := 2 * 5
  let grown_flowers := initial_flowers + 20
  let harvested_flowers := 5 * 4
  grown_flowers - harvested_flowers = 10 :=
by
  sorry

end flowers_died_l335_33553


namespace sum_possible_values_q_l335_33555

/-- If natural numbers k, l, p, and q satisfy the given conditions,
the sum of all possible values of q is 4 --/
theorem sum_possible_values_q (k l p q : ℕ) 
    (h1 : ∀ a b : ℝ, a ≠ b → a * b = l → a + b = k → (∃ (c d : ℝ), c + d = (k * (l + 1)) / l ∧ c * d = (l + 2 + 1 / l))) 
    (h2 : a + 1 / b ≠ b + 1 / a)
    : q = 4 :=
sorry

end sum_possible_values_q_l335_33555


namespace possible_values_of_m_l335_33524

def f (x a m : ℝ) := abs (x - a) + m * abs (x + a)

theorem possible_values_of_m {a m : ℝ} (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) : m = 1 / 5 :=
by 
  sorry

end possible_values_of_m_l335_33524


namespace john_paid_more_than_jane_l335_33565

theorem john_paid_more_than_jane :
    let original_price : ℝ := 40.00
    let discount_percentage : ℝ := 0.10
    let tip_percentage : ℝ := 0.15
    let discounted_price : ℝ := original_price - (discount_percentage * original_price)
    let john_tip : ℝ := tip_percentage * original_price
    let john_total : ℝ := discounted_price + john_tip
    let jane_tip : ℝ := tip_percentage * discounted_price
    let jane_total : ℝ := discounted_price + jane_tip
    let difference : ℝ := john_total - jane_total
    difference = 0.60 :=
by
  sorry

end john_paid_more_than_jane_l335_33565


namespace quadratic_inequality_solution_l335_33540

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40 * z + 340 ≤ 4 ↔ 12 ≤ z ∧ z ≤ 28 := by 
  sorry

end quadratic_inequality_solution_l335_33540


namespace max_value_quadratic_l335_33526

theorem max_value_quadratic : ∃ x : ℝ, -9 * x^2 + 27 * x + 15 = 35.25 :=
sorry

end max_value_quadratic_l335_33526


namespace system_inconsistent_l335_33521

-- Define the coefficient matrix and the augmented matrices.
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -2, 3], ![2, 3, -1], ![3, 1, 2]]

def B1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, -2, 3], ![7, 3, -1], ![10, 1, 2]]

-- Calculate the determinants.
noncomputable def delta : ℤ := A.det
noncomputable def delta1 : ℤ := B1.det

-- The main theorem statement: the system is inconsistent if Δ = 0 and Δ1 ≠ 0.
theorem system_inconsistent (h₁ : delta = 0) (h₂ : delta1 ≠ 0) : False :=
sorry

end system_inconsistent_l335_33521


namespace solution_set_of_inequality_l335_33534

theorem solution_set_of_inequality (x : ℝ) : (x / (x - 1) < 0) ↔ (0 < x ∧ x < 1) := 
sorry

end solution_set_of_inequality_l335_33534


namespace total_cement_used_l335_33588

-- Define the amounts of cement used for Lexi's street and Tess's street
def cement_used_lexis_street : ℝ := 10
def cement_used_tess_street : ℝ := 5.1

-- Prove that the total amount of cement used is 15.1 tons
theorem total_cement_used : cement_used_lexis_street + cement_used_tess_street = 15.1 := sorry

end total_cement_used_l335_33588


namespace clock_resale_price_l335_33530

theorem clock_resale_price
    (C : ℝ)  -- original cost of the clock to the store
    (H1 : 0.40 * C = 100)  -- condition: difference between original cost and buy-back price is $100
    (H2 : ∀ (C : ℝ), resell_price = 1.80 * (0.60 * C))  -- store sold the clock again with a 80% profit on buy-back
    : resell_price = 270 := 
by
  sorry

end clock_resale_price_l335_33530


namespace greatest_four_digit_divisible_by_6_l335_33558

-- Define a variable to represent a four-digit number
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a variable to represent divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define a variable to represent divisibility by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- State the theorem to prove that 9996 is the greatest four-digit number divisible by 6
theorem greatest_four_digit_divisible_by_6 : 
  (∀ n : ℕ, is_four_digit_number n → divisible_by_6 n → n ≤ 9996) ∧ (is_four_digit_number 9996 ∧ divisible_by_6 9996) :=
by
  -- Insert the proof here
  sorry

end greatest_four_digit_divisible_by_6_l335_33558


namespace cheapest_pie_cost_is_18_l335_33559

noncomputable def crust_cost : ℝ := 2 + 1 + 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_containers_needed : ℕ := 3 * (16 / 8)
noncomputable def blueberry_filling_cost : ℝ := blueberry_containers_needed * blueberry_container_cost
noncomputable def cherry_filling_cost : ℝ := 14
noncomputable def cheapest_filling_cost : ℝ := min blueberry_filling_cost cherry_filling_cost
noncomputable def total_cheapest_pie_cost : ℝ := crust_cost + cheapest_filling_cost

theorem cheapest_pie_cost_is_18 : total_cheapest_pie_cost = 18 := by
  sorry

end cheapest_pie_cost_is_18_l335_33559


namespace multiply_add_distribute_l335_33564

theorem multiply_add_distribute :
  42 * 25 + 58 * 42 = 3486 := by
  sorry

end multiply_add_distribute_l335_33564


namespace total_pamphlets_correct_l335_33536

def mike_initial_speed := 600
def mike_initial_hours := 9
def mike_break_hours := 2
def leo_relative_hours := 1 / 3
def leo_relative_speed := 2

def total_pamphlets (mike_initial_speed mike_initial_hours mike_break_hours leo_relative_hours leo_relative_speed : ℕ) : ℕ :=
  let mike_pamphlets_before_break := mike_initial_speed * mike_initial_hours
  let mike_speed_after_break := mike_initial_speed / 3
  let mike_pamphlets_after_break := mike_speed_after_break * mike_break_hours
  let total_mike_pamphlets := mike_pamphlets_before_break + mike_pamphlets_after_break

  let leo_hours := mike_initial_hours * leo_relative_hours
  let leo_speed := mike_initial_speed * leo_relative_speed
  let leo_pamphlets := leo_hours * leo_speed

  total_mike_pamphlets + leo_pamphlets

theorem total_pamphlets_correct : total_pamphlets 600 9 2 (1 / 3 : ℕ) 2 = 9400 := 
by 
  sorry

end total_pamphlets_correct_l335_33536


namespace zoe_total_songs_l335_33544

-- Define the number of country albums Zoe bought
def country_albums : Nat := 3

-- Define the number of pop albums Zoe bought
def pop_albums : Nat := 5

-- Define the number of songs per album
def songs_per_album : Nat := 3

-- Define the total number of albums
def total_albums : Nat := country_albums + pop_albums

-- Define the total number of songs
def total_songs : Nat := total_albums * songs_per_album

-- Theorem statement asserting the total number of songs
theorem zoe_total_songs : total_songs = 24 := by
  -- Proof will be inserted here (currently skipped)
  sorry

end zoe_total_songs_l335_33544


namespace Toms_walking_speed_l335_33569

theorem Toms_walking_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (run_distance : ℝ)
  (run_speed : ℝ)
  (walk_distance : ℝ)
  (walk_time : ℝ)
  (walk_speed : ℝ)
  (h1 : total_distance = 1800)
  (h2 : total_time ≤ 20)
  (h3 : run_distance = 600)
  (h4 : run_speed = 210)
  (h5 : total_distance = run_distance + walk_distance)
  (h6 : total_time = walk_time + run_distance / run_speed)
  (h7 : walk_speed = walk_distance / walk_time) :
  walk_speed ≤ 70 := sorry

end Toms_walking_speed_l335_33569


namespace book_pages_total_l335_33514

theorem book_pages_total
  (pages_read_first_day : ℚ) (total_pages : ℚ) (pages_read_second_day : ℚ)
  (rem_read_ratio : ℚ) (read_ratio_mult : ℚ)
  (book_ratio: ℚ) (read_pages_ratio: ℚ)
  (read_second_day_ratio: ℚ):
  pages_read_first_day = 1 / 6 →
  pages_read_second_day = 42 →
  rem_read_ratio = 3 →
  read_ratio_mult = (2 / 6) →
  book_ratio = 3 / 5 →
  read_pages_ratio = 2 / 5 →
  read_second_day_ratio = (2 / 5 - 1 / 6) →
  total_pages = pages_read_second_day / read_second_day_ratio  →
  total_pages = 126 :=
by sorry

end book_pages_total_l335_33514


namespace evaluate_expression_l335_33592

theorem evaluate_expression :
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := 
by 
  sorry

end evaluate_expression_l335_33592


namespace lumber_price_increase_l335_33589

noncomputable def percentage_increase_in_lumber_cost : ℝ :=
  let original_cost_lumber := 450
  let cost_nails := 30
  let cost_fabric := 80
  let original_total_cost := original_cost_lumber + cost_nails + cost_fabric
  let increase_in_total_cost := 97
  let new_total_cost := original_total_cost + increase_in_total_cost
  let unchanged_cost := cost_nails + cost_fabric
  let new_cost_lumber := new_total_cost - unchanged_cost
  let increase_lumber_cost := new_cost_lumber - original_cost_lumber
  (increase_lumber_cost / original_cost_lumber) * 100

theorem lumber_price_increase :
  percentage_increase_in_lumber_cost = 21.56 := by
  sorry

end lumber_price_increase_l335_33589


namespace average_score_l335_33503

theorem average_score (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) :
  (20 * m + 23 * n) / (20 + 23) = 20 / 43 * m + 23 / 43 * n := sorry

end average_score_l335_33503


namespace transform_f_to_shift_left_l335_33512

theorem transform_f_to_shift_left (f : ℝ → ℝ) :
  ∀ x : ℝ, f (2 * x - 1) = f (2 * (x - 1) + 1) := by
  sorry

end transform_f_to_shift_left_l335_33512


namespace problem_I_problem_II_l335_33538

open Real -- To use real number definitions and sin function.
open Set -- To use set constructs like intervals.

noncomputable def f (x : ℝ) : ℝ := sin (4 * x - π / 6) + sqrt 3 * sin (4 * x + π / 3)

-- Proof statement for monotonically decreasing interval of f(x).
theorem problem_I (k : ℤ) : 
  ∃ k : ℤ, ∀ x : ℝ, x ∈ Icc ((π / 12) + (k * π / 2)) ((π / 3) + (k * π / 2)) → 
  (4 * x + π / 6) ∈ Icc ((π / 2) + 2 * k * π) ((3 * π / 2) + 2 * k * π) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 4)

-- Proof statement for the range of g(x) on the interval [-π, 0].
theorem problem_II : 
  ∀ x : ℝ, x ∈ Icc (-π) 0 → g x ∈ Icc (-2) (sqrt 2) := 
sorry

end problem_I_problem_II_l335_33538


namespace sam_pennies_total_l335_33595

def initial_pennies : ℕ := 980
def found_pennies : ℕ := 930
def exchanged_pennies : ℕ := 725
def gifted_pennies : ℕ := 250

theorem sam_pennies_total :
  initial_pennies + found_pennies - exchanged_pennies + gifted_pennies = 1435 := 
sorry

end sam_pennies_total_l335_33595


namespace coplanar_vectors_set_B_l335_33580

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables (a b c : V)

theorem coplanar_vectors_set_B
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ • (2 • a + b) + k₂ • (a + b + c) = 7 • a + 5 • b + 3 • c :=
by { sorry }

end coplanar_vectors_set_B_l335_33580


namespace mathd_inequality_l335_33513

theorem mathd_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : 
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x * y + y * z + z * x) :=
by
  sorry

end mathd_inequality_l335_33513


namespace integer_solutions_l335_33582

theorem integer_solutions (x y : ℤ) : 
  (x^2 + x = y^4 + y^3 + y^2 + y) ↔ 
  (x, y) = (0, -1) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (5, 2) :=
by
  sorry

end integer_solutions_l335_33582


namespace students_present_in_class_l335_33537

noncomputable def num_students : ℕ := 100
noncomputable def percent_boys : ℝ := 0.55
noncomputable def percent_girls : ℝ := 0.45
noncomputable def absent_boys_percent : ℝ := 0.16
noncomputable def absent_girls_percent : ℝ := 0.12

theorem students_present_in_class :
  let num_boys := percent_boys * num_students
  let num_girls := percent_girls * num_students
  let absent_boys := absent_boys_percent * num_boys
  let absent_girls := absent_girls_percent * num_girls
  let present_boys := num_boys - absent_boys
  let present_girls := num_girls - absent_girls
  present_boys + present_girls = 86 :=
by
  sorry

end students_present_in_class_l335_33537


namespace haley_small_gardens_l335_33583

theorem haley_small_gardens (total_seeds seeds_in_big_garden seeds_per_small_garden : ℕ) (h1 : total_seeds = 56) (h2 : seeds_in_big_garden = 35) (h3 : seeds_per_small_garden = 3) :
  (total_seeds - seeds_in_big_garden) / seeds_per_small_garden = 7 :=
by
  sorry

end haley_small_gardens_l335_33583


namespace sin_cos_quotient_l335_33508

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_prime (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem sin_cos_quotient 
  (x : ℝ)
  (h : f_prime x = 3 * f x) 
  : (Real.sin x ^ 2 - 3) / (Real.cos x ^ 2 + 1) = -14 / 9 := 
by 
  sorry

end sin_cos_quotient_l335_33508


namespace digit_in_tens_place_is_nine_l335_33554

/-
Given:
1. Two numbers represented as 6t5 and 5t6 (where t is a digit).
2. The result of subtracting these two numbers is 9?4, where '?' represents a single digit in the tens place.

Prove:
The digit represented by '?' in the tens place is 9.
-/

theorem digit_in_tens_place_is_nine (t : ℕ) (h1 : 0 ≤ t ∧ t ≤ 9) :
  let a := 600 + t * 10 + 5
  let b := 500 + t * 10 + 6
  let result := a - b
  (result % 100) / 10 = 9 :=
by {
  sorry
}

end digit_in_tens_place_is_nine_l335_33554


namespace square_area_l335_33577

theorem square_area :
  ∃ (s : ℝ), (8 * s - 2 = 30) ∧ (s ^ 2 = 16) :=
by
  sorry

end square_area_l335_33577


namespace find_daily_wage_of_c_l335_33596

noncomputable def daily_wage_c (a b c : ℕ) (days_a days_b days_c total_earning : ℕ) : ℕ :=
  if 3 * b = 4 * a ∧ 3 * c = 5 * a ∧ 
    total_earning = 6 * a + 9 * b + 4 * c then c else 0

theorem find_daily_wage_of_c (a b c : ℕ)
  (days_a days_b days_c total_earning : ℕ)
  (h1 : days_a = 6)
  (h2 : days_b = 9)
  (h3 : days_c = 4)
  (h4 : 3 * b = 4 * a)
  (h5 : 3 * c = 5 * a)
  (h6 : total_earning = 1554)
  (h7 : total_earning = 6 * a + 9 * b + 4 * c) : 
  daily_wage_c a b c days_a days_b days_c total_earning = 105 := 
by sorry

end find_daily_wage_of_c_l335_33596


namespace distinct_non_zero_real_numbers_l335_33579

theorem distinct_non_zero_real_numbers (
  a b c : ℝ
) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + 2 * b * x1 + c = 0 ∧ ax^2 + 2 * b * x2 + c = 0) 
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ bx^2 + 2 * c * x1 + a = 0 ∧ bx^2 + 2 * c * x2 + a = 0)
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ cx^2 + 2 * a * x1 + b = 0 ∧ cx^2 + 2 * a * x2 + b = 0) :=
sorry

end distinct_non_zero_real_numbers_l335_33579


namespace combined_total_circles_squares_l335_33572

-- Define the problem parameters based on conditions
def US_stars : ℕ := 50
def US_stripes : ℕ := 13
def circles (n : ℕ) : ℕ := (n / 2) - 3
def squares (n : ℕ) : ℕ := (n * 2) + 6

-- Prove that the combined number of circles and squares on Pete's flag is 54
theorem combined_total_circles_squares : 
    circles US_stars + squares US_stripes = 54 := by
  sorry

end combined_total_circles_squares_l335_33572


namespace largest_k_divides_3n_plus_1_l335_33575

theorem largest_k_divides_3n_plus_1 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, k = 2 ∧ n % 2 = 1 ∧ 2^k ∣ 3^n + 1 ∨ k = 1 ∧ n % 2 = 0 ∧ 2^k ∣ 3^n + 1 :=
sorry

end largest_k_divides_3n_plus_1_l335_33575


namespace part1_part2_l335_33506

noncomputable def f (a x : ℝ) : ℝ := a - 1/x - Real.log x

theorem part1 (a : ℝ) :
  a = 2 → ∃ m b : ℝ, (∀ x : ℝ, f a x = x * m + b) ∧ (∀ y : ℝ, f a 1 = y → b = y ∧ m = 0) :=
by
  sorry

theorem part2 (a : ℝ) :
  (∃! x : ℝ, f a x = 0) → a = 1 :=
by
  sorry

end part1_part2_l335_33506


namespace smallest_possible_perimeter_l335_33560

theorem smallest_possible_perimeter (a : ℕ) (h : a > 2) (h_triangle : a < a + (a + 1) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a) :
  3 * a + 3 = 12 :=
by
  sorry

end smallest_possible_perimeter_l335_33560


namespace intersection_A_B_l335_33566

open Set

def A : Set ℤ := {x : ℤ | ∃ y : ℝ, y = Real.sqrt (1 - (x : ℝ)^2)}
def B : Set ℤ := {y : ℤ | ∃ x : ℤ, x ∈ A ∧ y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := 
by {
  sorry
}

end intersection_A_B_l335_33566


namespace team_total_points_l335_33576

theorem team_total_points 
  (n : ℕ)
  (best_score actual : ℕ)
  (desired_avg : ℕ)
  (hypothetical_score : ℕ)
  (current_best_score : ℕ)
  (team_size : ℕ)
  (h1 : team_size = 8)
  (h2 : current_best_score = 85)
  (h3 : hypothetical_score = 92)
  (h4 : desired_avg = 84)
  (h5 : hypothetical_score - current_best_score = 7)
  (h6 : team_size * desired_avg = 672) :
  (actual = 665) :=
sorry

end team_total_points_l335_33576


namespace total_boxes_is_27_l335_33533

-- Defining the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Prove that the total number of boxes is as expected
theorem total_boxes_is_27 : stops * boxes_per_stop = 27 := by
  sorry

end total_boxes_is_27_l335_33533


namespace large_cube_painted_blue_l335_33563

theorem large_cube_painted_blue (n : ℕ) (hp : 1 ≤ n) 
  (hc : (6 * n^2) = (1 / 3) * 6 * n^3) : n = 3 := by
  have hh := hc
  sorry

end large_cube_painted_blue_l335_33563


namespace smallest_sum_ab_l335_33539

theorem smallest_sum_ab (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 2^10 * 3^6 = a^b) : a + b = 866 :=
sorry

end smallest_sum_ab_l335_33539


namespace length_of_AB_l335_33501

-- Definitions of the given entities
def is_on_parabola (A : ℝ × ℝ) : Prop := A.2^2 = 4 * A.1
def focus : ℝ × ℝ := (1, 0)
def line_through_focus (l : ℝ × ℝ → Prop) : Prop := l focus

-- The theorem we need to prove
theorem length_of_AB (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (h1 : is_on_parabola A)
  (h2 : is_on_parabola B)
  (h3 : line_through_focus l)
  (h4 : l A)
  (h5 : l B)
  (h6 : A.1 + B.1 = 10 / 3) :
  dist A B = 16 / 3 :=
sorry

end length_of_AB_l335_33501


namespace nat_ineq_qr_ps_l335_33549

theorem nat_ineq_qr_ps (a b p q r s : ℕ) (h₀ : q * r - p * s = 1) 
  (h₁ : (p : ℚ) / q < a / b) (h₂ : (a : ℚ) / b < r / s) 
  : b ≥ q + s := sorry

end nat_ineq_qr_ps_l335_33549


namespace jonathan_weekly_deficit_correct_l335_33593

def daily_intake_non_saturday : ℕ := 2500
def daily_intake_saturday : ℕ := 3500
def daily_burn : ℕ := 3000
def weekly_caloric_deficit : ℕ :=
  (7 * daily_burn) - ((6 * daily_intake_non_saturday) + daily_intake_saturday)

theorem jonathan_weekly_deficit_correct :
  weekly_caloric_deficit = 2500 :=
by
  unfold weekly_caloric_deficit daily_intake_non_saturday daily_intake_saturday daily_burn
  sorry

end jonathan_weekly_deficit_correct_l335_33593


namespace polynomial_p0_l335_33552

theorem polynomial_p0 :
  ∃ p : ℕ → ℚ, (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (3^n)) ∧ (p 0 = 1093) :=
by
  sorry

end polynomial_p0_l335_33552


namespace min_tiles_to_cover_region_l335_33529

noncomputable def num_tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area

theorem min_tiles_to_cover_region : num_tiles_needed 6 2 36 72 = 216 :=
by 
  -- This is the format needed to include the assumptions and reach the conclusion
  sorry

end min_tiles_to_cover_region_l335_33529


namespace problem_l335_33502

theorem problem (a b : ℕ) (ha : 2^a ∣ 180) (h2 : ∀ n, 2^n ∣ 180 → n ≤ a) (hb : 5^b ∣ 180) (h5 : ∀ n, 5^n ∣ 180 → n ≤ b) : (1 / 3) ^ (b - a) = 3 := by
  sorry

end problem_l335_33502


namespace David_total_swim_time_l335_33531

theorem David_total_swim_time :
  let t_freestyle := 48
  let t_backstroke := t_freestyle + 4
  let t_butterfly := t_backstroke + 3
  let t_breaststroke := t_butterfly + 2
  t_freestyle + t_backstroke + t_butterfly + t_breaststroke = 212 :=
by
  sorry

end David_total_swim_time_l335_33531


namespace lollipop_count_l335_33591

theorem lollipop_count (total_cost : ℝ) (cost_per_lollipop : ℝ) (h1 : total_cost = 90) (h2 : cost_per_lollipop = 0.75) : 
  total_cost / cost_per_lollipop = 120 :=
by 
  sorry

end lollipop_count_l335_33591


namespace sum_of_squares_of_roots_l335_33528

theorem sum_of_squares_of_roots 
  (r s t : ℝ) 
  (hr : y^3 - 8 * y^2 + 9 * y - 2 = 0) 
  (hs : y ≥ 0) 
  (ht : y ≥ 0):
  r^2 + s^2 + t^2 = 46 :=
sorry

end sum_of_squares_of_roots_l335_33528
