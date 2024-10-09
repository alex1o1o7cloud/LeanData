import Mathlib

namespace electricity_price_increase_percentage_l1340_134002

noncomputable def old_power_kW : ℝ := 0.8
noncomputable def additional_power_percent : ℝ := 50 / 100
noncomputable def old_price_per_kWh : ℝ := 0.12
noncomputable def cost_for_50_hours : ℝ := 9
noncomputable def total_hours : ℝ := 50
noncomputable def energy_consumed := old_power_kW * total_hours

theorem electricity_price_increase_percentage :
  ∃ P : ℝ, 
    (energy_consumed * P = cost_for_50_hours) ∧
    ((P - old_price_per_kWh) / old_price_per_kWh) * 100 = 87.5 :=
by
  sorry

end electricity_price_increase_percentage_l1340_134002


namespace discriminant_zero_geometric_progression_l1340_134091

variable (a b c : ℝ)

theorem discriminant_zero_geometric_progression
  (h : b^2 = 4 * a * c) : (b / (2 * a)) = (2 * c / b) :=
by
  sorry

end discriminant_zero_geometric_progression_l1340_134091


namespace copy_pages_15_dollars_l1340_134052

theorem copy_pages_15_dollars (cost_per_page : ℕ) (total_dollars : ℕ) (cents_per_dollar : ℕ) : 
  cost_per_page = 3 → total_dollars = 15 → cents_per_dollar = 100 → 
  (total_dollars * cents_per_dollar) / cost_per_page = 500 :=
by
  intros h1 h2 h3
  sorry

end copy_pages_15_dollars_l1340_134052


namespace more_triangles_with_perimeter_2003_than_2000_l1340_134078

theorem more_triangles_with_perimeter_2003_than_2000 :
  (∃ (count_2003 count_2000 : ℕ), 
   count_2003 > count_2000 ∧ 
   (∀ (a b c : ℕ), a + b + c = 2000 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
   (∀ (a b c : ℕ), a + b + c = 2003 → a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a))
  := 
sorry

end more_triangles_with_perimeter_2003_than_2000_l1340_134078


namespace initial_pepper_amount_l1340_134068
-- Import the necessary libraries.

-- Declare the problem as a theorem.
theorem initial_pepper_amount (used left : ℝ) (h₁ : used = 0.16) (h₂ : left = 0.09) :
  used + left = 0.25 :=
by
  -- The proof is not required here.
  sorry

end initial_pepper_amount_l1340_134068


namespace statement_l1340_134003

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Condition 2: f(x-2) = -f(x) for all x
def satisfies_periodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x - 2) = -f x

-- Condition 3: f is decreasing on [0, 2]
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- The proof statement
theorem statement (h1 : is_odd_function f) (h2 : satisfies_periodicity f) (h3 : is_decreasing_on f 0 2) :
  f 5 < f 4 ∧ f 4 < f 3 :=
sorry

end statement_l1340_134003


namespace cubes_with_odd_red_faces_l1340_134061

-- Define the dimensions and conditions of the block
def block_length : ℕ := 6
def block_width: ℕ := 6
def block_height : ℕ := 2

-- The block is painted initially red on all sides
-- Then the bottom face is painted blue
-- The block is cut into 1-inch cubes
-- 

noncomputable def num_cubes_with_odd_red_faces (length width height : ℕ) : ℕ :=
  -- Only edge cubes have odd number of red faces in this configuration
  let corner_count := 8  -- 4 on top + 4 on bottom (each has 4 red faces)
  let edge_count := 40   -- 20 on top + 20 on bottom (each has 3 red faces)
  let face_only_count := 32 -- 16 on top + 16 on bottom (each has 2 red faces)
  -- The resulting total number of cubes with odd red faces
  edge_count

-- The theorem we need to prove
theorem cubes_with_odd_red_faces : num_cubes_with_odd_red_faces block_length block_width block_height = 40 :=
  by 
    -- Proof goes here
    sorry

end cubes_with_odd_red_faces_l1340_134061


namespace negation_of_universal_l1340_134013

theorem negation_of_universal (P : ∀ x : ℝ, x^2 > 0) : ¬ ( ∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by 
  sorry

end negation_of_universal_l1340_134013


namespace max_parallelograms_in_hexagon_l1340_134056

theorem max_parallelograms_in_hexagon (side_hexagon side_parallelogram1 side_parallelogram2 : ℝ)
                                        (angle_parallelogram : ℝ) :
  side_hexagon = 3 ∧ side_parallelogram1 = 1 ∧ side_parallelogram2 = 2 ∧ angle_parallelogram = (π / 3) →
  ∃ n : ℕ, n = 12 :=
by 
  sorry

end max_parallelograms_in_hexagon_l1340_134056


namespace rotation_transforms_and_sums_l1340_134011

theorem rotation_transforms_and_sums 
    (D E F D' E' F' : (ℝ × ℝ))
    (hD : D = (0, 0)) (hE : E = (0, 20)) (hF : F = (30, 0)) 
    (hD' : D' = (-26, 23)) (hE' : E' = (-46, 23)) (hF' : F' = (-26, -7))
    (n : ℝ) (x y : ℝ)
    (rotation_condition : 0 < n ∧ n < 180)
    (angle_condition : n = 90) :
    n + x + y = 60.5 :=
by
  have hx : x = -49 := sorry
  have hy : y = 19.5 := sorry
  have hn : n = 90 := sorry
  sorry

end rotation_transforms_and_sums_l1340_134011


namespace eval_fraction_product_l1340_134084

theorem eval_fraction_product :
  ((1 + (1 / 3)) * (1 + (1 / 4)) = (5 / 3)) :=
by
  sorry

end eval_fraction_product_l1340_134084


namespace average_cost_across_all_products_sold_is_670_l1340_134096

-- Definitions based on conditions
def iphones_sold : ℕ := 100
def ipad_sold : ℕ := 20
def appletv_sold : ℕ := 80

def cost_iphone : ℕ := 1000
def cost_ipad : ℕ := 900
def cost_appletv : ℕ := 200

-- Calculations based on conditions
def revenue_iphone : ℕ := iphones_sold * cost_iphone
def revenue_ipad : ℕ := ipad_sold * cost_ipad
def revenue_appletv : ℕ := appletv_sold * cost_appletv

def total_revenue : ℕ := revenue_iphone + revenue_ipad + revenue_appletv
def total_products_sold : ℕ := iphones_sold + ipad_sold + appletv_sold

def average_cost := total_revenue / total_products_sold

-- Theorem to be proved
theorem average_cost_across_all_products_sold_is_670 :
  average_cost = 670 :=
by
  sorry

end average_cost_across_all_products_sold_is_670_l1340_134096


namespace difference_mean_median_is_neg_half_l1340_134035

-- Definitions based on given conditions
def scoreDistribution : List (ℕ × ℚ) :=
  [(65, 0.05), (75, 0.25), (85, 0.4), (95, 0.2), (105, 0.1)]

-- Defining the total number of students as 100 for easier percentage calculations
def totalStudents := 100

-- Definition to compute mean
def mean : ℚ :=
  scoreDistribution.foldl (λ acc (score, percentage) => acc + (↑score * percentage)) 0

-- Median score based on the distribution conditions
def median : ℚ := 85

-- Proving the proposition that the difference between the mean and the median is -0.5
theorem difference_mean_median_is_neg_half :
  median - mean = -0.5 :=
sorry

end difference_mean_median_is_neg_half_l1340_134035


namespace max_distance_traveled_l1340_134048

def distance_traveled (t : ℝ) : ℝ := 15 * t - 6 * t^2

theorem max_distance_traveled : ∃ t : ℝ, distance_traveled t = 75 / 8 :=
by
  sorry

end max_distance_traveled_l1340_134048


namespace traveling_zoo_l1340_134031

theorem traveling_zoo (x y : ℕ) (h1 : x + y = 36) (h2 : 4 * x + 6 * y = 100) : x = 14 ∧ y = 22 :=
by {
  sorry
}

end traveling_zoo_l1340_134031


namespace cars_in_garage_l1340_134018

theorem cars_in_garage (c : ℕ) 
  (bicycles : ℕ := 20) 
  (motorcycles : ℕ := 5) 
  (total_wheels : ℕ := 90) 
  (bicycle_wheels : ℕ := 2 * bicycles)
  (motorcycle_wheels : ℕ := 2 * motorcycles)
  (car_wheels : ℕ := 4 * c) 
  (eq : bicycle_wheels + car_wheels + motorcycle_wheels = total_wheels) : 
  c = 10 := 
by 
  sorry

end cars_in_garage_l1340_134018


namespace max_value_of_trig_function_l1340_134025

theorem max_value_of_trig_function : 
  ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 5 := sorry


end max_value_of_trig_function_l1340_134025


namespace assume_dead_heat_race_l1340_134057

variable {Va Vb L H : ℝ}

theorem assume_dead_heat_race (h1 : Va = (51 / 44) * Vb) :
  H = (7 / 51) * L :=
sorry

end assume_dead_heat_race_l1340_134057


namespace no_nat_triplet_square_l1340_134008

theorem no_nat_triplet_square (m n k : ℕ) : ¬ (∃ a b c : ℕ, m^2 + n + k = a^2 ∧ n^2 + k + m = b^2 ∧ k^2 + m + n = c^2) :=
by sorry

end no_nat_triplet_square_l1340_134008


namespace point_always_outside_circle_l1340_134053

theorem point_always_outside_circle (a : ℝ) : a^2 + (2 - a)^2 > 1 :=
by sorry

end point_always_outside_circle_l1340_134053


namespace cost_price_of_apple_l1340_134045

-- Define the given conditions SP = 20, and the relation between SP and CP.
variables (SP CP : ℝ)
axiom h1 : SP = 20
axiom h2 : SP = CP - (1/6) * CP

-- Statement to be proved.
theorem cost_price_of_apple : CP = 24 :=
by
  sorry

end cost_price_of_apple_l1340_134045


namespace average_length_l1340_134072

def length1 : ℕ := 2
def length2 : ℕ := 3
def length3 : ℕ := 7

theorem average_length : (length1 + length2 + length3) / 3 = 4 :=
by
  sorry

end average_length_l1340_134072


namespace original_average_marks_l1340_134099

theorem original_average_marks (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 30) 
  (h2 : new_avg = 90)
  (h3 : ∀ new_avg, new_avg = 2 * A → A = 90 / 2) : 
  A = 45 :=
by
  sorry

end original_average_marks_l1340_134099


namespace combination_problem_l1340_134029

theorem combination_problem (x : ℕ) (hx_pos : 0 < x) (h_comb : Nat.choose 9 x = Nat.choose 9 (2 * x + 3)) : x = 2 :=
by {
  sorry
}

end combination_problem_l1340_134029


namespace boys_in_class_l1340_134098

theorem boys_in_class (r : ℕ) (g b : ℕ) (h1 : g/b = 4/3) (h2 : g + b = 35) : b = 15 :=
  sorry

end boys_in_class_l1340_134098


namespace correct_statement_l1340_134064

/-- Given the following statements:
 1. Seeing a rainbow after rain is a random event.
 2. To check the various equipment before a plane takes off, a random sampling survey should be conducted.
 3. When flipping a coin 20 times, it will definitely land heads up 10 times.
 4. The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B.

 Prove that the correct statement is: Seeing a rainbow after rain is a random event.
-/
theorem correct_statement : 
  let statement_A := "Seeing a rainbow after rain is a random event"
  let statement_B := "To check the various equipment before a plane takes off, a random sampling survey should be conducted"
  let statement_C := "When flipping a coin 20 times, it will definitely land heads up 10 times"
  let statement_D := "The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B"
  statement_A = "Seeing a rainbow after rain is a random event" := by
sorry

end correct_statement_l1340_134064


namespace focus_of_parabola_l1340_134024

-- Define the equation of the given parabola
def given_parabola (x y : ℝ) : Prop := y = - (1 / 8) * x^2

-- Define the condition for the focus of the parabola
def is_focus (focus : ℝ × ℝ) : Prop := focus = (0, -2)

-- State the theorem
theorem focus_of_parabola : ∃ (focus : ℝ × ℝ), given_parabola x y → is_focus focus :=
by
  -- Placeholder proof
  sorry

end focus_of_parabola_l1340_134024


namespace average_time_to_win_permit_l1340_134033

theorem average_time_to_win_permit :
  let p n := (9/10)^(n-1) * (1/10)
  ∑' n, n * p n = 10 :=
sorry

end average_time_to_win_permit_l1340_134033


namespace Dean_handled_100_transactions_l1340_134067

-- Definitions for the given conditions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := (9 * Mabel_transactions) / 10 + Mabel_transactions
def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3
def Jade_transactions : ℕ := Cal_transactions + 14
def Dean_transactions : ℕ := (Jade_transactions * 25) / 100 + Jade_transactions

-- Define the theorem we need to prove
theorem Dean_handled_100_transactions : Dean_transactions = 100 :=
by
  -- Statement to skip the actual proof
  sorry

end Dean_handled_100_transactions_l1340_134067


namespace megan_folders_l1340_134009

theorem megan_folders (initial_files deleted_files files_per_folder : ℕ) (h1 : initial_files = 237)
    (h2 : deleted_files = 53) (h3 : files_per_folder = 12) :
    let remaining_files := initial_files - deleted_files
    let total_folders := (remaining_files / files_per_folder) + 1
    total_folders = 16 := 
by
  sorry

end megan_folders_l1340_134009


namespace find_n_l1340_134095

theorem find_n (m n : ℝ) (h1 : m + 2 * n = 1.2) (h2 : 0.1 + m + n + 0.1 = 1) : n = 0.4 :=
by
  sorry

end find_n_l1340_134095


namespace alley_width_theorem_l1340_134058

noncomputable def width_of_alley (a k h : ℝ) (h₁ : k = a / 2) (h₂ : h = a * (Real.sqrt 2) / 2) : ℝ :=
  Real.sqrt ((a * (Real.sqrt 2) / 2)^2 + (a / 2)^2)

theorem alley_width_theorem (a k h w : ℝ)
  (h₁ : k = a / 2)
  (h₂ : h = a * (Real.sqrt 2) / 2)
  (h₃ : w = width_of_alley a k h h₁ h₂) :
  w = (Real.sqrt 3) * a / 2 :=
by
  sorry

end alley_width_theorem_l1340_134058


namespace tree_growth_factor_l1340_134004

theorem tree_growth_factor 
  (initial_total : ℕ) 
  (initial_maples : ℕ) 
  (initial_lindens : ℕ) 
  (spring_total : ℕ) 
  (autumn_total : ℕ)
  (initial_maple_percentage : initial_maples = 3 * initial_total / 5)
  (spring_maple_percentage : initial_maples = spring_total / 5)
  (autumn_maple_percentage : initial_maples * 2 = autumn_total * 3 / 5) :
  autumn_total = 6 * initial_total :=
sorry

end tree_growth_factor_l1340_134004


namespace complement_union_eq_l1340_134026

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 3, 5}

theorem complement_union_eq:
  compl A ∪ B = {0, 2, 3, 5} :=
by
  sorry

end complement_union_eq_l1340_134026


namespace find_f_105_5_l1340_134085

noncomputable def f : ℝ → ℝ :=
sorry -- Definition of f

-- Hypotheses
axiom even_function (x : ℝ) : f x = f (-x)
axiom functional_equation (x : ℝ) : f (x + 2) = -f x
axiom function_values (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 3) : f x = x

-- Goal
theorem find_f_105_5 : f 105.5 = 2.5 :=
sorry

end find_f_105_5_l1340_134085


namespace scholarship_amount_l1340_134010

-- Definitions
def tuition_per_semester : ℕ := 22000
def parents_contribution : ℕ := tuition_per_semester / 2
def work_hours : ℕ := 200
def hourly_wage : ℕ := 10
def work_earnings : ℕ := work_hours * hourly_wage
def remaining_tuition : ℕ := tuition_per_semester - parents_contribution - work_earnings

-- Theorem to prove the scholarship amount
theorem scholarship_amount (S : ℕ) (h : 3 * S = remaining_tuition) : S = 3000 :=
by
  sorry

end scholarship_amount_l1340_134010


namespace stone_105_is_3_l1340_134079

def stone_numbered_at_105 (n : ℕ) := (15 + (n - 1) % 28)

theorem stone_105_is_3 :
  stone_numbered_at_105 105 = 3 := by
  sorry

end stone_105_is_3_l1340_134079


namespace find_quadruplets_l1340_134065

theorem find_quadruplets :
  ∃ (x y z w : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
  (xyz + 1) / (x + 1) = (yzw + 1) / (y + 1) ∧
  (yzw + 1) / (y + 1) = (zwx + 1) / (z + 1) ∧
  (zwx + 1) / (z + 1) = (wxy + 1) / (w + 1) ∧
  x + y + z + w = 48 ∧
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 :=
by
  sorry

end find_quadruplets_l1340_134065


namespace operation_positive_l1340_134037

theorem operation_positive (op : ℤ → ℤ → ℤ) (is_pos : op 1 (-2) > 0) : op = Int.sub :=
by
  sorry

end operation_positive_l1340_134037


namespace arithmetic_sequence_geometric_sum_l1340_134005

theorem arithmetic_sequence_geometric_sum (a₁ a₂ d : ℕ) (h₁ : d ≠ 0) 
    (h₂ : (2 * a₁ + d)^2 = a₁ * (4 * a₁ + 6 * d)) :
    a₂ = 3 * a₁ :=
by
  sorry

end arithmetic_sequence_geometric_sum_l1340_134005


namespace perfect_square_difference_l1340_134066

def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem perfect_square_difference :
  ∃ a b : ℕ, ∃ x y : ℕ,
    a = x^2 ∧ b = y^2 ∧
    lastDigit a = 6 ∧
    lastDigit b = 4 ∧
    lastDigit (a - b) = 2 ∧
    lastDigit a > lastDigit b :=
by
  sorry

end perfect_square_difference_l1340_134066


namespace part1_part2_l1340_134022

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : 
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := sorry

theorem part2 (a : ℝ) : 
  (∀ x, f x a > -a) ↔ (a > -3/2) := sorry

end part1_part2_l1340_134022


namespace max_value_quadratic_l1340_134028

theorem max_value_quadratic (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : x * (1 - x) ≤ 1 / 4 :=
by
  sorry

end max_value_quadratic_l1340_134028


namespace GroundBeefSalesTotalRevenue_l1340_134080

theorem GroundBeefSalesTotalRevenue :
  let price_regular := 3.50
  let price_lean := 4.25
  let price_extra_lean := 5.00

  let monday_revenue := 198.5 * price_regular +
                        276.2 * price_lean +
                        150.7 * price_extra_lean

  let tuesday_revenue := 210 * (price_regular * 0.90) +
                         420 * (price_lean * 0.90) +
                         150 * (price_extra_lean * 0.90)
  
  let wednesday_revenue := 230 * price_regular +
                           324.6 * 3.75 +
                           120.4 * price_extra_lean

  monday_revenue + tuesday_revenue + wednesday_revenue = 8189.35 :=
by
  sorry

end GroundBeefSalesTotalRevenue_l1340_134080


namespace number_of_hens_l1340_134006

theorem number_of_hens (H C : ℕ) (h1 : H + C = 44) (h2 : 2 * H + 4 * C = 128) : H = 24 :=
by
  sorry

end number_of_hens_l1340_134006


namespace combined_surface_area_of_cube_and_sphere_l1340_134088

theorem combined_surface_area_of_cube_and_sphere (V_cube : ℝ) :
  V_cube = 729 →
  ∃ (A_combined : ℝ), A_combined = 486 + 81 * Real.pi :=
by
  intro V_cube
  sorry

end combined_surface_area_of_cube_and_sphere_l1340_134088


namespace floodDamageInUSD_l1340_134069

def floodDamageAUD : ℝ := 45000000
def exchangeRateAUDtoUSD : ℝ := 1.2

theorem floodDamageInUSD : floodDamageAUD * (1 / exchangeRateAUDtoUSD) = 37500000 := 
by 
  sorry

end floodDamageInUSD_l1340_134069


namespace find_particular_number_l1340_134016

variable (x : ℝ)

theorem find_particular_number (h : 0.46 + x = 0.72) : x = 0.26 :=
sorry

end find_particular_number_l1340_134016


namespace smallest_number_in_systematic_sample_l1340_134030

theorem smallest_number_in_systematic_sample (n m x : ℕ) (products : Finset ℕ) :
  n = 80 ∧ m = 5 ∧ products = Finset.range n ∧ x = 42 ∧ x ∈ products ∧ (∃ k : ℕ, x = (n / m) * k + 10) → 10 ∈ products :=
by
  sorry

end smallest_number_in_systematic_sample_l1340_134030


namespace midpoint_C_is_either_l1340_134044

def A : ℝ := -7
def dist_AB : ℝ := 5

theorem midpoint_C_is_either (C : ℝ) (h : C = (A + (A + dist_AB / 2)) / 2 ∨ C = (A + (A - dist_AB / 2)) / 2) : 
  C = -9 / 2 ∨ C = -19 / 2 := 
sorry

end midpoint_C_is_either_l1340_134044


namespace lilies_per_centerpiece_l1340_134076

theorem lilies_per_centerpiece (centerpieces roses orchids cost total_budget price_per_flower number_of_lilies_per_centerpiece : ℕ) 
  (h0 : centerpieces = 6)
  (h1 : roses = 8)
  (h2 : orchids = 2 * roses)
  (h3 : cost = total_budget)
  (h4 : total_budget = 2700)
  (h5 : price_per_flower = 15)
  (h6 : cost = (centerpieces * roses * price_per_flower) + (centerpieces * orchids * price_per_flower) + (centerpieces * number_of_lilies_per_centerpiece * price_per_flower))
  : number_of_lilies_per_centerpiece = 6 := 
by 
  sorry

end lilies_per_centerpiece_l1340_134076


namespace method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l1340_134070

/-- Method 1: Membership card costs 200 yuan + 10 yuan per swim session. -/
def method1_cost (num_sessions : ℕ) : ℕ := 200 + 10 * num_sessions

/-- Method 2: Each swim session costs 30 yuan. -/
def method2_cost (num_sessions : ℕ) : ℕ := 30 * num_sessions

/-- Problem (1): Total cost for 3 swim sessions using Method 1 is 230 yuan. -/
theorem method1_three_sessions_cost : method1_cost 3 = 230 := by
  sorry

/-- Problem (2): Method 2 is more cost-effective than Method 1 for 9 swim sessions. -/
theorem method2_more_cost_effective_for_nine_sessions : method2_cost 9 < method1_cost 9 := by
  sorry

/-- Problem (3): Method 1 allows more sessions than Method 2 within a budget of 600 yuan. -/
theorem method1_allows_more_sessions : (600 - 200) / 10 > 600 / 30 := by
  sorry

end method1_three_sessions_cost_method2_more_cost_effective_for_nine_sessions_method1_allows_more_sessions_l1340_134070


namespace max_original_chess_pieces_l1340_134087

theorem max_original_chess_pieces (m n M N : ℕ) (h1 : m ≤ 19) (h2 : n ≤ 19) (h3 : M ≤ 19) (h4 : N ≤ 19) (h5 : M * N = m * n + 45) (h6 : M = m ∨ N = n) : m * n ≤ 285 :=
by
  sorry

end max_original_chess_pieces_l1340_134087


namespace biking_time_l1340_134055

noncomputable def east_bound_speed : ℝ := 22
noncomputable def west_bound_speed : ℝ := east_bound_speed + 4
noncomputable def total_distance : ℝ := 200

theorem biking_time :
  (east_bound_speed + west_bound_speed) * (t : ℝ) = total_distance → t = 25 / 6 :=
by
  -- The proof is omitted and replaced with sorry.
  sorry

end biking_time_l1340_134055


namespace minimum_moves_to_find_coin_l1340_134051

/--
Consider a circle of 100 thimbles with a coin hidden under one of them. 
You can check four thimbles per move. After each move, the coin moves to a neighboring thimble.
Prove that the minimum number of moves needed to guarantee finding the coin is 33.
-/
theorem minimum_moves_to_find_coin 
  (N : ℕ) (hN : N = 100) (M : ℕ) (hM : M = 4) :
  ∃! k : ℕ, k = 33 :=
by sorry

end minimum_moves_to_find_coin_l1340_134051


namespace mod_last_digit_l1340_134001

theorem mod_last_digit (N : ℕ) (a b : ℕ) (h : N = 10 * a + b) (hb : b < 10) : 
  N % 10 = b ∧ N % 2 = b % 2 ∧ N % 5 = b % 5 :=
by
  sorry

end mod_last_digit_l1340_134001


namespace negation_of_universal_prop_l1340_134023

theorem negation_of_universal_prop:
  (¬ (∀ x : ℝ, x ^ 3 - x ≥ 0)) ↔ (∃ x : ℝ, x ^ 3 - x < 0) := 
by 
sorry

end negation_of_universal_prop_l1340_134023


namespace parabola_equation_l1340_134046

theorem parabola_equation (P : ℝ × ℝ) :
  let d1 := dist P (-3, 0)
  let d2 := abs (P.1 - 2)
  (d1 = d2 + 1 ↔ P.2^2 = -12 * P.1) :=
by
  intro d1 d2
  sorry

end parabola_equation_l1340_134046


namespace solution_set_of_inequality_l1340_134074

theorem solution_set_of_inequality (x : ℝ) : 
  (|x+1| - |x-4| > 3) ↔ x > 3 :=
sorry

end solution_set_of_inequality_l1340_134074


namespace problem_b_l1340_134007

theorem problem_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) : a + b ≥ 2 :=
sorry

end problem_b_l1340_134007


namespace arithmetic_seq_8th_term_l1340_134049

theorem arithmetic_seq_8th_term (a d : ℤ) 
  (h4 : a + 3 * d = 22) 
  (h6 : a + 5 * d = 46) : 
  a + 7 * d = 70 :=
by 
  sorry

end arithmetic_seq_8th_term_l1340_134049


namespace sector_central_angle_l1340_134042

theorem sector_central_angle (r α: ℝ) (hC: 4 * r = 2 * r + α * r): α = 2 :=
by
  -- Proof is to be filled in
  sorry

end sector_central_angle_l1340_134042


namespace wrapping_paper_area_correct_l1340_134000

-- Define the length, width, and height of the box
variables (l w h : ℝ)

-- Define the function to calculate the area of the wrapping paper
def wrapping_paper_area (l w h : ℝ) : ℝ := 2 * (l + w + h) ^ 2

-- Statement problem that we need to prove
theorem wrapping_paper_area_correct :
  wrapping_paper_area l w h = 2 * (l + w + h) ^ 2 := 
sorry

end wrapping_paper_area_correct_l1340_134000


namespace simplify_expression_l1340_134093

theorem simplify_expression :
  (8 : ℝ)^(1/3) - (343 : ℝ)^(1/3) = -5 :=
by
  sorry

end simplify_expression_l1340_134093


namespace sum_g_values_l1340_134073

noncomputable def g (x : ℝ) : ℝ :=
if x > 3 then x^2 - 1 else
if x >= -3 then 3 * x + 2 else 4

theorem sum_g_values : g (-4) + g 0 + g 4 = 21 :=
by
  sorry

end sum_g_values_l1340_134073


namespace kerosene_cost_is_024_l1340_134040

-- Definitions from the conditions
def dozen_eggs_cost := 0.36 -- Cost of a dozen eggs is the same as 1 pound of rice which is $0.36
def pound_of_rice_cost := 0.36
def kerosene_cost := 8 * (0.36 / 12) -- Cost of kerosene is the cost of 8 eggs

-- Theorem to prove
theorem kerosene_cost_is_024 : kerosene_cost = 0.24 := by
  sorry

end kerosene_cost_is_024_l1340_134040


namespace group_size_l1340_134039

-- Define the conditions
variables (N : ℕ)
variable (h1 : (1 / 5 : ℝ) * N = (N : ℝ) * 0.20)
variable (h2 : 128 ≤ N)
variable (h3 : (1 / 5 : ℝ) * N - 128 = 0.04 * (N : ℝ))

-- Prove that the number of people in the group is 800
theorem group_size : N = 800 :=
by
  sorry

end group_size_l1340_134039


namespace b_product_l1340_134081

variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- All terms in the arithmetic sequence \{aₙ\} are non-zero.
axiom a_nonzero : ∀ n, a n ≠ 0

-- The sequence satisfies the given condition.
axiom a_cond : a 3 - (a 7)^2 / 2 + a 11 = 0

-- The sequence \{bₙ\} is a geometric sequence with ratio r.
axiom b_geometric : ∃ r, ∀ n, b (n + 1) = r * b n

-- And b₇ = a₇
axiom b_7 : b 7 = a 7

-- Prove that b₁ * b₁₃ = 16
theorem b_product : b 1 * b 13 = 16 :=
sorry

end b_product_l1340_134081


namespace number_of_trees_planted_l1340_134077

theorem number_of_trees_planted (initial_trees final_trees trees_planted : ℕ) 
  (h_initial : initial_trees = 22)
  (h_final : final_trees = 77)
  (h_planted : trees_planted = final_trees - initial_trees) : 
  trees_planted = 55 := by
  sorry

end number_of_trees_planted_l1340_134077


namespace sale_price_per_bearing_before_bulk_discount_l1340_134014

-- Define the given conditions
def machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := machines * ball_bearings_per_machine

def normal_cost_per_bearing : ℝ := 1
def total_normal_cost : ℝ := total_ball_bearings * normal_cost_per_bearing

def bulk_discount : ℝ := 0.20
def sale_savings : ℝ := 120

-- The theorem we need to prove
theorem sale_price_per_bearing_before_bulk_discount (P : ℝ) :
  total_normal_cost - (total_ball_bearings * P * (1 - bulk_discount)) = sale_savings → 
  P = 0.75 :=
by sorry

end sale_price_per_bearing_before_bulk_discount_l1340_134014


namespace boy_usual_time_to_school_l1340_134090

theorem boy_usual_time_to_school
  (S : ℝ) -- Usual speed
  (T : ℝ) -- Usual time
  (D : ℝ) -- Distance, D = S * T
  (hD : D = S * T)
  (h1 : 3/4 * D / (7/6 * S) + 1/4 * D / (5/6 * S) = T - 2) : 
  T = 35 :=
by
  sorry

end boy_usual_time_to_school_l1340_134090


namespace Alex_hula_hoop_duration_l1340_134094

-- Definitions based on conditions
def Nancy_duration := 10
def Casey_duration := Nancy_duration - 3
def Morgan_duration := Casey_duration * 3
def Alex_duration := Casey_duration + Morgan_duration - 2

-- The theorem we need to prove
theorem Alex_hula_hoop_duration : Alex_duration = 26 := by
  -- proof to be provided
  sorry

end Alex_hula_hoop_duration_l1340_134094


namespace at_least_2020_distinct_n_l1340_134060

theorem at_least_2020_distinct_n : 
  ∃ (N : Nat), N ≥ 2020 ∧ ∃ (a : Fin N → ℕ), 
  Function.Injective a ∧ ∀ i, ∃ k : ℚ, (a i : ℚ) + 0.25 = (k + 1/2)^2 := 
sorry

end at_least_2020_distinct_n_l1340_134060


namespace minimum_radius_part_a_minimum_radius_part_b_l1340_134063

-- Definitions for Part (a)
def a := 7
def b := 8
def c := 9
def R1 := 6

-- Statement for Part (a)
theorem minimum_radius_part_a : (c / 2) = R1 := by sorry

-- Definitions for Part (b)
def a' := 9
def b' := 15
def c' := 16
def R2 := 9

-- Statement for Part (b)
theorem minimum_radius_part_b : (c' / 2) = R2 := by sorry

end minimum_radius_part_a_minimum_radius_part_b_l1340_134063


namespace value_of_a_l1340_134089

def A := { x : ℝ | x^2 - 8*x + 15 = 0 }
def B (a : ℝ) := { x : ℝ | x * a - 1 = 0 }

theorem value_of_a (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end value_of_a_l1340_134089


namespace total_ribbon_length_l1340_134012

theorem total_ribbon_length (a b c d e f g h i : ℝ) 
  (H : a + b + c + d + e + f + g + h + i = 62) : 
  1.5 * (a + b + c + d + e + f + g + h + i) = 93 :=
by
  sorry

end total_ribbon_length_l1340_134012


namespace negation_proposition_l1340_134086

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ ∃ x0 : ℝ, x0^2 - 2*x0 + 4 > 0 :=
by
  sorry

end negation_proposition_l1340_134086


namespace bahs_equal_to_yahs_l1340_134041

theorem bahs_equal_to_yahs (bahs rahs yahs : ℝ) 
  (h1 : 18 * bahs = 30 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) : 
  1200 * yahs = 432 * bahs := 
by
  sorry

end bahs_equal_to_yahs_l1340_134041


namespace angle_B_is_30_degrees_l1340_134043

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Assuming the conditions given in the problem
variables (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) 
          (h2 : a > b)

-- The proof to establish the measure of angle B as 30 degrees
theorem angle_B_is_30_degrees (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) (h2 : a > b) : B = Real.pi / 6 :=
sorry

end angle_B_is_30_degrees_l1340_134043


namespace find_a_value_l1340_134038

noncomputable def collinear (points : List (ℚ × ℚ)) := 
  ∃ a b c, ∀ (x y : ℚ), (x, y) ∈ points → a * x + b * y + c = 0

theorem find_a_value (a : ℚ) :
  collinear [(3, -5), (-a + 2, 3), (2*a + 3, 2)] → a = -7 / 23 :=
by
  sorry

end find_a_value_l1340_134038


namespace average_prime_numbers_l1340_134082

-- Definitions of the visible numbers.
def visible1 : ℕ := 51
def visible2 : ℕ := 72
def visible3 : ℕ := 43

-- Definitions of the hidden numbers as prime numbers.
def hidden1 : ℕ := 2
def hidden2 : ℕ := 23
def hidden3 : ℕ := 31

-- Common sum of the numbers on each card.
def common_sum : ℕ := 74

-- Establishing the conditions given in the problem.
def condition1 : hidden1 + visible2 = common_sum := by sorry
def condition2 : hidden2 + visible1 = common_sum := by sorry
def condition3 : hidden3 + visible3 = common_sum := by sorry

-- Calculate the average of the hidden prime numbers.
def average_hidden_primes : ℚ := (hidden1 + hidden2 + hidden3) / 3

-- The proof statement that the average of the hidden prime numbers is 56/3.
theorem average_prime_numbers : average_hidden_primes = 56 / 3 := by
  sorry

end average_prime_numbers_l1340_134082


namespace father_l1340_134071

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l1340_134071


namespace total_camels_l1340_134092

theorem total_camels (x y : ℕ) (humps_eq : x + 2 * y = 23) (legs_eq : 4 * (x + y) = 60) : x + y = 15 :=
by
  sorry

end total_camels_l1340_134092


namespace weeks_jake_buys_papayas_l1340_134027

theorem weeks_jake_buys_papayas
  (jake_papayas : ℕ)
  (brother_papayas : ℕ)
  (father_papayas : ℕ)
  (total_papayas : ℕ)
  (h1 : jake_papayas = 3)
  (h2 : brother_papayas = 5)
  (h3 : father_papayas = 4)
  (h4 : total_papayas = 48) :
  (total_papayas / (jake_papayas + brother_papayas + father_papayas) = 4) :=
by
  sorry

end weeks_jake_buys_papayas_l1340_134027


namespace sum_max_min_ratio_l1340_134032

def ellipse_eq (x y : ℝ) : Prop :=
  5 * x^2 + x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0

theorem sum_max_min_ratio (p q : ℝ) (x y : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y → y / x = p ∨ y / x = q) → 
  p + q = 31 / 34 :=
by
  sorry

end sum_max_min_ratio_l1340_134032


namespace initial_money_is_10_l1340_134034

-- Definition for the initial amount of money
def initial_money (X : ℝ) : Prop :=
  let spent_on_cupcakes := (1 / 5) * X
  let remaining_after_cupcakes := X - spent_on_cupcakes
  let spent_on_milkshake := 5
  let remaining_after_milkshake := remaining_after_cupcakes - spent_on_milkshake
  remaining_after_milkshake = 3

-- The statement proving that Ivan initially had $10
theorem initial_money_is_10 (X : ℝ) (h : initial_money X) : X = 10 :=
by sorry

end initial_money_is_10_l1340_134034


namespace triangle_inequality_l1340_134054

variable {a b c : ℝ}

theorem triangle_inequality (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) (h₅ : a + b > c) (h₆ : a + c > b) (h₇ : b + c > a) : 
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
by
  sorry

end triangle_inequality_l1340_134054


namespace sheets_of_paper_per_week_l1340_134059

theorem sheets_of_paper_per_week
  (sheets_per_class_per_day : ℕ)
  (num_classes : ℕ)
  (school_days_per_week : ℕ)
  (total_sheets_per_week : ℕ) 
  (h1 : sheets_per_class_per_day = 200)
  (h2 : num_classes = 9)
  (h3 : school_days_per_week = 5)
  (h4 : total_sheets_per_week = sheets_per_class_per_day * num_classes * school_days_per_week) :
  total_sheets_per_week = 9000 :=
sorry

end sheets_of_paper_per_week_l1340_134059


namespace marble_distribution_l1340_134020

theorem marble_distribution (x : ℝ) (h : 49 = (3 * x + 2) + (x + 1) + (2 * x - 1) + x) :
  (3 * x + 2 = 22) ∧ (x + 1 = 8) ∧ (2 * x - 1 = 12) ∧ (x = 7) :=
by
  sorry

end marble_distribution_l1340_134020


namespace no_integer_solution_xyz_l1340_134019

theorem no_integer_solution_xyz : ¬ ∃ (x y z : ℤ),
  x^6 + x^3 + x^3 * y + y = 147^157 ∧
  x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := by
  sorry

end no_integer_solution_xyz_l1340_134019


namespace tap_B_fill_time_l1340_134021

theorem tap_B_fill_time :
  ∃ t : ℝ, 
    (3 * 10 + (12 / t) * 10 = 36) →
    t = 20 :=
by
  sorry

end tap_B_fill_time_l1340_134021


namespace inscribed_circle_ratio_l1340_134050

theorem inscribed_circle_ratio (a b h r : ℝ) (h_triangle : h = Real.sqrt (a^2 + b^2))
  (A : ℝ) (H1 : A = (1/2) * a * b) (s : ℝ) (H2 : s = (a + b + h) / 2) 
  (H3 : A = r * s) : (π * r / A) = (π * r) / (h + r) :=
sorry

end inscribed_circle_ratio_l1340_134050


namespace expected_value_of_problems_l1340_134047

-- Define the setup
def num_pairs : ℕ := 5
def num_shoes : ℕ := num_pairs * 2
def prob_same_color : ℚ := 1 / (num_shoes - 1)
def days : ℕ := 5

-- Define the expected value calculation using linearity of expectation
def expected_problems_per_day : ℚ := prob_same_color
def expected_total_problems : ℚ := days * expected_problems_per_day

-- Prove the expected number of practice problems Sandra gets to do over 5 days
theorem expected_value_of_problems : expected_total_problems = 5 / 9 := 
by 
  rw [expected_total_problems, expected_problems_per_day, prob_same_color]
  norm_num
  sorry

end expected_value_of_problems_l1340_134047


namespace smallest_value_of_reciprocal_sums_l1340_134036

theorem smallest_value_of_reciprocal_sums (r1 r2 s p : ℝ) 
  (h1 : r1 + r2 = s)
  (h2 : r1^2 + r2^2 = s)
  (h3 : r1^3 + r2^3 = s)
  (h4 : r1^4 + r2^4 = s)
  (h1004 : r1^1004 + r2^1004 = s)
  (h_r1_r2_roots : ∀ x, x^2 - s * x + p = 0) :
  (1 / r1^1005 + 1 / r2^1005) = 2 :=
by
  sorry

end smallest_value_of_reciprocal_sums_l1340_134036


namespace sofa_love_seat_cost_l1340_134015

theorem sofa_love_seat_cost (love_seat_cost : ℕ) (sofa_cost : ℕ) 
    (h₁ : love_seat_cost = 148) (h₂ : sofa_cost = 2 * love_seat_cost) :
    love_seat_cost + sofa_cost = 444 := 
by
  sorry

end sofa_love_seat_cost_l1340_134015


namespace sufficient_condition_l1340_134097

variable (a b c d : ℝ)

-- Condition p: a and b are the roots of the equation.
def condition_p : Prop := a * a + b * b + c * (a + b) + d = 0

-- Condition q: a + b + c = 0
def condition_q : Prop := a + b + c = 0

theorem sufficient_condition : condition_p a b c d → condition_q a b c := by
  sorry

end sufficient_condition_l1340_134097


namespace find_width_l1340_134017

-- Definition of the perimeter of a rectangle
def perimeter (L W : ℝ) : ℝ := 2 * (L + W)

-- The given conditions
def length := 13
def perimeter_value := 50

-- The goal to prove: if the perimeter is 50 and the length is 13, then the width must be 12
theorem find_width :
  ∃ (W : ℝ), perimeter length W = perimeter_value ∧ W = 12 :=
by
  sorry

end find_width_l1340_134017


namespace find_y_l1340_134083

theorem find_y :
  (∃ y : ℝ, (4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4) ∧ y = 1251) :=
by
  sorry

end find_y_l1340_134083


namespace seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l1340_134062

-- Problem 1
theorem seven_divides_n_iff_seven_divides_q_minus_2r (n q r : ℕ) (h : n = 10 * q + r) :
  (7 ∣ n) ↔ (7 ∣ (q - 2 * r)) := sorry

-- Problem 2
theorem seven_divides_2023 : 7 ∣ 2023 :=
  let q := 202
  let r := 3
  have h : 2023 = 10 * q + r := by norm_num
  have h1 : (7 ∣ 2023) ↔ (7 ∣ (q - 2 * r)) :=
    seven_divides_n_iff_seven_divides_q_minus_2r 2023 q r h
  sorry -- Here you would use h1 and prove the statement using it

-- Problem 3
theorem thirteen_divides_n_iff_thirteen_divides_q_plus_4r (n q r : ℕ) (h : n = 10 * q + r) :
  (13 ∣ n) ↔ (13 ∣ (q + 4 * r)) := sorry

end seven_divides_n_iff_seven_divides_q_minus_2r_seven_divides_2023_thirteen_divides_n_iff_thirteen_divides_q_plus_4r_l1340_134062


namespace train_time_to_pass_platform_l1340_134075

noncomputable def train_length : ℝ := 360
noncomputable def platform_length : ℝ := 140
noncomputable def train_speed_km_per_hr : ℝ := 45

noncomputable def train_speed_m_per_s : ℝ :=
  train_speed_km_per_hr * (1000 / 3600)

noncomputable def total_distance : ℝ :=
  train_length + platform_length

theorem train_time_to_pass_platform :
  (total_distance / train_speed_m_per_s) = 40 := by
  sorry

end train_time_to_pass_platform_l1340_134075
