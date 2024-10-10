import Mathlib

namespace routes_8x5_grid_l2887_288784

/-- The number of routes on a grid from (0,0) to (m,n) where only right and up movements are allowed -/
def numRoutes (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The theorem stating that the number of routes on an 8x5 grid is 12870 -/
theorem routes_8x5_grid : numRoutes 8 5 = 12870 := by sorry

end routes_8x5_grid_l2887_288784


namespace ray_remaining_nickels_l2887_288769

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Ray's initial amount in cents -/
def initial_amount : ℕ := 285

/-- Amount given to Peter in cents -/
def peter_amount : ℕ := 55

/-- Amount given to Paula in cents -/
def paula_amount : ℕ := 45

/-- Calculates the number of nickels from a given amount of cents -/
def cents_to_nickels (cents : ℕ) : ℕ := cents / nickel_value

theorem ray_remaining_nickels :
  let initial_nickels := cents_to_nickels initial_amount
  let peter_nickels := cents_to_nickels peter_amount
  let randi_nickels := cents_to_nickels (3 * peter_amount)
  let paula_nickels := cents_to_nickels paula_amount
  initial_nickels - (peter_nickels + randi_nickels + paula_nickels) = 4 := by
  sorry

end ray_remaining_nickels_l2887_288769


namespace parabola_through_three_points_l2887_288770

/-- A parabola with equation y = x^2 + bx + c passing through (-1, -11), (3, 17), and (2, 5) has b = 13/3 and c = -5 -/
theorem parabola_through_three_points :
  ∀ b c : ℚ,
  ((-1)^2 + b*(-1) + c = -11) →
  (3^2 + b*3 + c = 17) →
  (2^2 + b*2 + c = 5) →
  (b = 13/3 ∧ c = -5) :=
by sorry

end parabola_through_three_points_l2887_288770


namespace exactly_one_not_through_origin_l2887_288716

def f₁ (x : ℝ) : ℝ := x^4 + 1
def f₂ (x : ℝ) : ℝ := x^4 + x
def f₃ (x : ℝ) : ℝ := x^4 + x^2
def f₄ (x : ℝ) : ℝ := x^4 + x^3

def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

theorem exactly_one_not_through_origin :
  ∃! i : Fin 4, ¬passes_through_origin (match i with
    | 0 => f₁
    | 1 => f₂
    | 2 => f₃
    | 3 => f₄) :=
by sorry

end exactly_one_not_through_origin_l2887_288716


namespace plot_length_is_sixty_l2887_288717

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  lengthExcess : ℝ
  lengthIsTwentyMoreThanBreadth : length = breadth + lengthExcess
  fencingCostEquation : fencingCostPerMeter * (2 * (length + breadth)) = totalFencingCost

/-- Theorem stating that under given conditions, the length of the plot is 60 meters -/
theorem plot_length_is_sixty (plot : RectangularPlot)
  (h1 : plot.lengthExcess = 20)
  (h2 : plot.fencingCostPerMeter = 26.5)
  (h3 : plot.totalFencingCost = 5300) :
  plot.length = 60 := by
  sorry


end plot_length_is_sixty_l2887_288717


namespace albert_betty_age_ratio_l2887_288743

/-- Represents the ages of Albert, Mary, and Betty -/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.mary = ages.albert - 22 ∧
  ages.betty = 11

/-- The theorem to prove -/
theorem albert_betty_age_ratio (ages : Ages) :
  age_conditions ages → (ages.albert : ℚ) / ages.betty = 4 := by
  sorry

#check albert_betty_age_ratio

end albert_betty_age_ratio_l2887_288743


namespace combined_class_average_weight_l2887_288781

/-- Calculates the average weight of a combined class given two sections -/
def averageWeightCombinedClass (studentsA : ℕ) (studentsB : ℕ) (avgWeightA : ℚ) (avgWeightB : ℚ) : ℚ :=
  (studentsA * avgWeightA + studentsB * avgWeightB) / (studentsA + studentsB)

/-- Theorem stating the average weight of the combined class -/
theorem combined_class_average_weight :
  averageWeightCombinedClass 26 34 50 30 = 2320 / 60 := by
  sorry

#eval averageWeightCombinedClass 26 34 50 30

end combined_class_average_weight_l2887_288781


namespace inequality_solution_l2887_288759

theorem inequality_solution (x : ℕ) : 5 * x + 3 < 3 * (2 + x) ↔ x = 0 ∨ x = 1 := by
  sorry

end inequality_solution_l2887_288759


namespace f_monotonicity_and_range_f_non_positive_iff_m_eq_one_l2887_288715

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem f_monotonicity_and_range (m : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∨
  (∃ c, 0 < c ∧ 
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < c → f m x₁ < f m x₂) ∧
    (∀ x₁ x₂, c < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂)) :=
sorry

theorem f_non_positive_iff_m_eq_one (m : ℝ) :
  (∀ x, 0 < x → f m x ≤ 0) ↔ m = 1 :=
sorry

end f_monotonicity_and_range_f_non_positive_iff_m_eq_one_l2887_288715


namespace solution_set_implies_a_range_l2887_288739

theorem solution_set_implies_a_range :
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) → a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end solution_set_implies_a_range_l2887_288739


namespace jerry_shelf_theorem_l2887_288765

/-- The difference between action figures and books on Jerry's shelf -/
def shelf_difference (books : ℕ) (initial_figures : ℕ) (added_figures : ℕ) : ℕ :=
  (initial_figures + added_figures) - books

/-- Theorem stating the difference between action figures and books on Jerry's shelf -/
theorem jerry_shelf_theorem :
  shelf_difference 3 4 2 = 3 := by
  sorry

end jerry_shelf_theorem_l2887_288765


namespace line_equation_proof_l2887_288767

/-- Given two lines in the form y = mx + b, they are parallel if and only if they have the same slope m -/
def parallel_lines (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2

/-- A point (x, y) lies on a line y = mx + b if and only if y = mx + b -/
def point_on_line (x y m b : ℝ) : Prop := y = m * x + b

theorem line_equation_proof (x y : ℝ) : 
  parallel_lines (3/2) 3 (3/2) (-11/2) ∧ 
  point_on_line 3 (-1) (3/2) (-11/2) ∧
  3 * x - 2 * y - 11 = 0 ↔ y = (3/2) * x - 11/2 :=
sorry

end line_equation_proof_l2887_288767


namespace wheels_equation_l2887_288783

theorem wheels_equation (x y : ℕ) : 2 * x + 4 * y = 66 → y = (33 - x) / 2 :=
by sorry

end wheels_equation_l2887_288783


namespace music_festival_audience_count_l2887_288705

/-- Represents the distribution of audience for a band -/
structure BandDistribution where
  underThirtyMale : ℝ
  underThirtyFemale : ℝ
  thirtyToFiftyMale : ℝ
  thirtyToFiftyFemale : ℝ
  overFiftyMale : ℝ
  overFiftyFemale : ℝ

/-- The music festival with its audience distribution -/
def MusicFestival : List BandDistribution :=
  [
    { underThirtyMale := 0.04, underThirtyFemale := 0.0266667, thirtyToFiftyMale := 0.0375, thirtyToFiftyFemale := 0.0458333, overFiftyMale := 0.00833333, overFiftyFemale := 0.00833333 },
    { underThirtyMale := 0.03, underThirtyFemale := 0.07, thirtyToFiftyMale := 0.02, thirtyToFiftyFemale := 0.03, overFiftyMale := 0.00833333, overFiftyFemale := 0.00833333 },
    { underThirtyMale := 0.02, underThirtyFemale := 0.03, thirtyToFiftyMale := 0.0416667, thirtyToFiftyFemale := 0.0416667, overFiftyMale := 0.0133333, overFiftyFemale := 0.02 },
    { underThirtyMale := 0.0458333, underThirtyFemale := 0.0375, thirtyToFiftyMale := 0.03, thirtyToFiftyFemale := 0.0366667, overFiftyMale := 0.01, overFiftyFemale := 0.00666667 },
    { underThirtyMale := 0.015, underThirtyFemale := 0.0183333, thirtyToFiftyMale := 0.0333333, thirtyToFiftyFemale := 0.0333333, overFiftyMale := 0.03, overFiftyFemale := 0.0366667 },
    { underThirtyMale := 0.0583333, underThirtyFemale := 0.025, thirtyToFiftyMale := 0.03, thirtyToFiftyFemale := 0.0366667, overFiftyMale := 0.00916667, overFiftyFemale := 0.00750 }
  ]

theorem music_festival_audience_count : 
  let totalMaleUnder30 := (MusicFestival.map (λ b => b.underThirtyMale)).sum
  ∃ n : ℕ, n ≥ 431 ∧ n < 432 ∧ (90 : ℝ) / totalMaleUnder30 = n := by
  sorry

end music_festival_audience_count_l2887_288705


namespace dog_distance_l2887_288702

theorem dog_distance (s : ℝ) (ivan_speed dog_speed : ℝ) : 
  s > 0 → 
  ivan_speed > 0 → 
  dog_speed > 0 → 
  s = 3 → 
  dog_speed = 3 * ivan_speed → 
  (∃ t : ℝ, t > 0 ∧ ivan_speed * t = s / 4 ∧ dog_speed * t = 3 * s / 4) → 
  (dog_speed * (s / ivan_speed)) = 9 :=
by sorry

end dog_distance_l2887_288702


namespace cubic_roots_sum_of_squares_l2887_288707

theorem cubic_roots_sum_of_squares (α β γ : ℂ) : 
  (α^3 - 6*α^2 + 11*α - 6 = 0) → 
  (β^3 - 6*β^2 + 11*β - 6 = 0) → 
  (γ^3 - 6*γ^2 + 11*γ - 6 = 0) → 
  α^2 + β^2 + γ^2 = 14 := by
sorry

end cubic_roots_sum_of_squares_l2887_288707


namespace fraction_zero_implies_x_one_l2887_288763

theorem fraction_zero_implies_x_one (x : ℝ) : (x - 1) / (x - 5) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_one_l2887_288763


namespace binomial_expansion_103_l2887_288722

theorem binomial_expansion_103 : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end binomial_expansion_103_l2887_288722


namespace jonah_profit_l2887_288799

def pineapples : ℕ := 60
def base_price : ℚ := 2
def discount_rate : ℚ := 20 / 100
def rings_per_pineapple : ℕ := 12
def single_ring_price : ℚ := 4
def bundle_size : ℕ := 6
def bundle_price : ℚ := 20
def bundles_sold : ℕ := 35
def single_rings_sold : ℕ := 150

def discounted_price : ℚ := base_price * (1 - discount_rate)
def total_cost : ℚ := pineapples * discounted_price
def bundle_revenue : ℚ := bundles_sold * bundle_price
def single_ring_revenue : ℚ := single_rings_sold * single_ring_price
def total_revenue : ℚ := bundle_revenue + single_ring_revenue
def profit : ℚ := total_revenue - total_cost

theorem jonah_profit : profit = 1204 := by
  sorry

end jonah_profit_l2887_288799


namespace correct_train_process_l2887_288752

-- Define the actions as an inductive type
inductive TrainAction
  | BuyTicket
  | WaitForTrain
  | CheckTicket
  | BoardTrain

-- Define a type for a sequence of actions
def ActionSequence := List TrainAction

-- Define the correct sequence
def correctSequence : ActionSequence :=
  [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.CheckTicket, TrainAction.BoardTrain]

-- Define a predicate for a valid train-taking process
def isValidProcess (sequence : ActionSequence) : Prop :=
  sequence = correctSequence

-- Theorem statement
theorem correct_train_process :
  isValidProcess correctSequence :=
sorry

end correct_train_process_l2887_288752


namespace eight_digit_divisible_by_nine_l2887_288740

theorem eight_digit_divisible_by_nine (n : Nat) : 
  n ≤ 9 →
  (854 * 10^7 + n * 10^6 + 5 * 10^5 + 2 * 10^4 + 6 * 10^3 + 8 * 10^2 + 6 * 10 + 8) % 9 = 0 →
  n = 7 := by
sorry

end eight_digit_divisible_by_nine_l2887_288740


namespace multiple_of_nine_three_l2887_288764

theorem multiple_of_nine_three (S : ℤ) : 
  (∀ x : ℤ, 9 ∣ x → 3 ∣ x) →  -- All multiples of 9 are multiples of 3
  (Odd S) →                   -- S is an odd number
  (9 ∣ S) →                   -- S is a multiple of 9
  (3 ∣ S) :=                  -- S is a multiple of 3
by sorry

end multiple_of_nine_three_l2887_288764


namespace arithmetic_sequence_problem_l2887_288731

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_3 = 8 and a_6 = 5, a_9 = 2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a3 : a 3 = 8) 
  (h_a6 : a 6 = 5) : 
  a 9 = 2 := by
  sorry


end arithmetic_sequence_problem_l2887_288731


namespace circles_intersect_l2887_288757

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- Definition of intersection for two circles -/
def intersect (c : TwoCircles) : Prop :=
  let d := Real.sqrt ((c.center1.1 - c.center2.1)^2 + (c.center1.2 - c.center2.2)^2)
  d < c.radius1 + c.radius2 ∧ d > abs (c.radius1 - c.radius2)

/-- The main theorem: the given circles intersect -/
theorem circles_intersect : 
  let c := TwoCircles.mk (0, 0) 2 (-3, 4) 4
  intersect c := by sorry


end circles_intersect_l2887_288757


namespace concert_revenue_calculation_l2887_288711

def ticket_revenue (student_price : ℕ) (non_student_price : ℕ) (total_tickets : ℕ) (student_tickets : ℕ) : ℕ :=
  let non_student_tickets := total_tickets - student_tickets
  student_price * student_tickets + non_student_price * non_student_tickets

theorem concert_revenue_calculation :
  ticket_revenue 9 11 2000 520 = 20960 := by
  sorry

end concert_revenue_calculation_l2887_288711


namespace max_sum_divisible_into_two_parts_l2887_288795

theorem max_sum_divisible_into_two_parts (S : ℕ) : 
  (∃ (nums : List ℕ), 
    (∀ n ∈ nums, 0 < n ∧ n ≤ 10) ∧ 
    (nums.sum = S) ∧ 
    (∀ (partition : List ℕ × List ℕ), 
      partition.1 ∪ partition.2 = nums → 
      partition.1.sum ≤ 70 ∧ partition.2.sum ≤ 70)) →
  S ≤ 133 :=
by sorry

end max_sum_divisible_into_two_parts_l2887_288795


namespace gcd_lcm_product_l2887_288785

theorem gcd_lcm_product (a b : ℤ) : Nat.gcd a.natAbs b.natAbs * Nat.lcm a.natAbs b.natAbs = a.natAbs * b.natAbs := by
  sorry

end gcd_lcm_product_l2887_288785


namespace sum_of_possible_distances_l2887_288782

/-- Given two points A and B on a number line, where the distance between A and B is 2,
    and the distance between A and the origin O is 3,
    the sum of all possible distances between B and the origin O is 12. -/
theorem sum_of_possible_distances (A B : ℝ) : 
  (|A - B| = 2) → (|A| = 3) → (|B| + |-B| + |B - 2| + |-(B - 2)| = 12) := by
  sorry

end sum_of_possible_distances_l2887_288782


namespace joystick_payment_ratio_l2887_288708

/-- Proves that the ratio of Frank's payment for the joystick to the total cost of the joystick is 1:4 -/
theorem joystick_payment_ratio :
  ∀ (computer_table computer_chair joystick frank_joystick eman_joystick : ℕ),
    computer_table = 140 →
    computer_chair = 100 →
    joystick = 20 →
    frank_joystick + eman_joystick = joystick →
    computer_table + frank_joystick = computer_chair + eman_joystick + 30 →
    frank_joystick * 4 = joystick := by
  sorry

end joystick_payment_ratio_l2887_288708


namespace abs_T_equals_1024_l2887_288749

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^19 - (1 - i)^19

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by sorry

end abs_T_equals_1024_l2887_288749


namespace arithmetic_sequence_sum_6_l2887_288774

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ d : ℚ) : ℕ → ℚ
  | 0 => 0
  | n + 1 => a₁ + n * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- Theorem statement
theorem arithmetic_sequence_sum_6 (a₁ d : ℚ) :
  a₁ = 1/2 →
  sum_arithmetic_sequence a₁ d 4 = 20 →
  sum_arithmetic_sequence a₁ d 6 = 48 := by
  sorry

-- The proof is omitted as per the instructions

end arithmetic_sequence_sum_6_l2887_288774


namespace log_equation_solution_l2887_288744

-- Define the logarithm function
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equation_solution :
  ∃ x : ℝ, x = 17 ∧ log2 ((3*x + 9) / (5*x - 3)) + log2 ((5*x - 3) / (x - 2)) = 2 :=
by
  sorry

end log_equation_solution_l2887_288744


namespace bacteria_growth_time_l2887_288728

/-- The growth factor of bacteria per cycle -/
def growth_factor : ℕ := 4

/-- The duration of one growth cycle in hours -/
def cycle_duration : ℕ := 5

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 1000

/-- The final number of bacteria -/
def final_bacteria : ℕ := 256000

/-- The number of cycles needed to reach the final bacteria count -/
def num_cycles : ℕ := 4

theorem bacteria_growth_time :
  cycle_duration * num_cycles =
    (final_bacteria / initial_bacteria).log growth_factor * cycle_duration :=
by sorry

end bacteria_growth_time_l2887_288728


namespace power_of_half_equals_one_l2887_288758

theorem power_of_half_equals_one (a b : ℕ) : 
  (2^a : ℕ) ∣ 300 ∧ 
  (∀ k : ℕ, k > a → ¬((2^k : ℕ) ∣ 300)) ∧ 
  (3^b : ℕ) ∣ 300 ∧ 
  (∀ k : ℕ, k > b → ¬((3^k : ℕ) ∣ 300)) → 
  (1/2 : ℚ)^(b - a + 1) = 1 := by sorry

end power_of_half_equals_one_l2887_288758


namespace team_selection_count_l2887_288712

def boys := 10
def girls := 10
def team_size := 8
def min_boys := 3

def select_team (b g : ℕ) (t m : ℕ) : ℕ :=
  (Nat.choose b 3 * Nat.choose g 5) +
  (Nat.choose b 4 * Nat.choose g 4) +
  (Nat.choose b 5 * Nat.choose g 3) +
  (Nat.choose b 6 * Nat.choose g 2) +
  (Nat.choose b 7 * Nat.choose g 1) +
  (Nat.choose b 8 * Nat.choose g 0)

theorem team_selection_count :
  select_team boys girls team_size min_boys = 114275 :=
by sorry

end team_selection_count_l2887_288712


namespace carla_initial_marbles_l2887_288777

/-- The number of marbles Carla bought -/
def marbles_bought : ℝ := 489.0

/-- The total number of marbles Carla has now -/
def total_marbles : ℝ := 2778.0

/-- The number of marbles Carla started with -/
def initial_marbles : ℝ := total_marbles - marbles_bought

theorem carla_initial_marbles : initial_marbles = 2289.0 := by
  sorry

end carla_initial_marbles_l2887_288777


namespace yumis_farm_chickens_l2887_288792

/-- The number of chickens on Yumi's farm -/
def num_chickens : ℕ := 6

/-- The number of pigs on Yumi's farm -/
def num_pigs : ℕ := 9

/-- The number of legs each pig has -/
def pig_legs : ℕ := 4

/-- The number of legs each chicken has -/
def chicken_legs : ℕ := 2

/-- The total number of legs of all animals on Yumi's farm -/
def total_legs : ℕ := 48

theorem yumis_farm_chickens :
  num_chickens * chicken_legs + num_pigs * pig_legs = total_legs :=
by sorry

end yumis_farm_chickens_l2887_288792


namespace no_geometric_progression_with_11_12_13_l2887_288735

theorem no_geometric_progression_with_11_12_13 :
  ¬ ∃ (a q : ℝ) (k l n : ℕ), 
    (k < l ∧ l < n) ∧
    (a * q ^ k = 11) ∧
    (a * q ^ l = 12) ∧
    (a * q ^ n = 13) :=
sorry

end no_geometric_progression_with_11_12_13_l2887_288735


namespace no_solution_iff_n_eq_neg_half_l2887_288761

theorem no_solution_iff_n_eq_neg_half (n : ℝ) : 
  (∀ x y z : ℝ, ¬(2*n*x + y = 2 ∧ n*y + z = 2 ∧ x + 2*n*z = 2)) ↔ n = -1/2 :=
by sorry

end no_solution_iff_n_eq_neg_half_l2887_288761


namespace betty_daughter_age_difference_l2887_288745

/-- Proves that Betty's daughter is 40% younger than Betty given the specified conditions -/
theorem betty_daughter_age_difference (betty_age : ℕ) (granddaughter_age : ℕ) : 
  betty_age = 60 →
  granddaughter_age = 12 →
  granddaughter_age = (betty_age - (betty_age - granddaughter_age * 3)) / 3 →
  (betty_age - granddaughter_age * 3) / betty_age * 100 = 40 := by
  sorry

end betty_daughter_age_difference_l2887_288745


namespace f_difference_l2887_288732

/-- Sum of all positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℚ := (sigma n + n) / n

/-- Theorem stating the result of f(540) - f(180) -/
theorem f_difference : f 540 - f 180 = 7 / 90 := by sorry

end f_difference_l2887_288732


namespace min_even_integers_l2887_288737

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 26 → 
  a + b + c + d = 41 → 
  a + b + c + d + e + f = 57 → 
  ∃ (n : ℕ), n ≥ 1 ∧ 
  (∀ (m : ℕ), m < n → 
    ¬∃ (evens : Finset ℤ), evens.card = m ∧ 
    (∀ x ∈ evens, Even x) ∧ 
    evens ⊆ {a, b, c, d, e, f}) :=
sorry

end min_even_integers_l2887_288737


namespace remainder_seven_twelfth_mod_hundred_l2887_288709

theorem remainder_seven_twelfth_mod_hundred : 7^12 % 100 = 1 := by
  sorry

end remainder_seven_twelfth_mod_hundred_l2887_288709


namespace algebraic_expression_equality_l2887_288714

theorem algebraic_expression_equality (x y : ℝ) (h : x + 2*y = 2) : 1 - 2*x - 4*y = -3 := by
  sorry

end algebraic_expression_equality_l2887_288714


namespace quadratic_equations_roots_l2887_288776

theorem quadratic_equations_roots (a₁ a₂ a₃ : ℝ) 
  (h_positive : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0)
  (h_geometric : ∃ (r : ℝ), a₂ = a₁ * r ∧ a₃ = a₂ * r)
  (h_roots_1 : a₁^2 ≥ 4)
  (h_roots_2 : a₂^2 < 8) :
  a₃^2 < 16 := by
  sorry

#check quadratic_equations_roots

end quadratic_equations_roots_l2887_288776


namespace line_parallel_to_x_axis_l2887_288727

/-- A line parallel to the X-axis passing through the point (3, -2) has the equation y = -2 -/
theorem line_parallel_to_x_axis (line : Set (ℝ × ℝ)) : 
  ((3 : ℝ), -2) ∈ line →  -- The line passes through the point (3, -2)
  (∀ (x y₁ y₂ : ℝ), ((x, y₁) ∈ line ∧ (x, y₂) ∈ line) → y₁ = y₂) →  -- The line is parallel to the X-axis
  ∀ (x y : ℝ), (x, y) ∈ line ↔ y = -2 :=  -- The equation of the line is y = -2
by sorry

end line_parallel_to_x_axis_l2887_288727


namespace sequence_value_l2887_288798

/-- Given a sequence {aₙ} satisfying a₁ = 1 and aₙ - aₙ₋₁ = 2ⁿ⁻¹ for n ≥ 2, prove that a₈ = 255 -/
theorem sequence_value (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n-1) = 2^(n-1)) : 
  a 8 = 255 := by
sorry

end sequence_value_l2887_288798


namespace circle_diameter_twice_radius_l2887_288733

/-- A circle with a center, radius, and diameter. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  diameter : ℝ

/-- The diameter of a circle is twice its radius. -/
theorem circle_diameter_twice_radius (c : Circle) : c.diameter = 2 * c.radius := by
  sorry

end circle_diameter_twice_radius_l2887_288733


namespace final_card_count_l2887_288742

def baseball_card_problem (initial_cards : ℕ) (maria_takes : ℕ → ℕ) (peter_takes : ℕ) (paul_multiplies : ℕ → ℕ) : ℕ :=
  let after_maria := initial_cards - maria_takes initial_cards
  let after_peter := after_maria - peter_takes
  paul_multiplies after_peter

theorem final_card_count :
  baseball_card_problem 15 (fun n => (n + 1) / 2) 1 (fun n => 3 * n) = 18 := by
  sorry

end final_card_count_l2887_288742


namespace emilys_cards_l2887_288751

theorem emilys_cards (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 63)
  (h2 : final_cards = 70) :
  final_cards - initial_cards = 7 := by
  sorry

end emilys_cards_l2887_288751


namespace student_A_more_stable_l2887_288773

/-- Represents a student with their score variance -/
structure Student where
  name : String
  variance : ℝ

/-- Defines the concept of score stability based on variance -/
def moreStable (s1 s2 : Student) : Prop :=
  s1.variance < s2.variance

/-- Theorem stating that student A has more stable scores than student B -/
theorem student_A_more_stable :
  let studentA : Student := ⟨"A", 3.6⟩
  let studentB : Student := ⟨"B", 4.4⟩
  moreStable studentA studentB := by
  sorry

end student_A_more_stable_l2887_288773


namespace conor_weekly_vegetables_l2887_288780

-- Define the number of each vegetable Conor can chop in a day
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8

-- Define the number of days Conor works per week
def work_days_per_week : ℕ := 4

-- Theorem to prove
theorem conor_weekly_vegetables :
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := by
  sorry

end conor_weekly_vegetables_l2887_288780


namespace total_coins_l2887_288760

theorem total_coins (quarters_piles : Nat) (quarters_per_pile : Nat)
                    (dimes_piles : Nat) (dimes_per_pile : Nat)
                    (nickels_piles : Nat) (nickels_per_pile : Nat)
                    (pennies_piles : Nat) (pennies_per_pile : Nat)
                    (h1 : quarters_piles = 8) (h2 : quarters_per_pile = 5)
                    (h3 : dimes_piles = 6) (h4 : dimes_per_pile = 7)
                    (h5 : nickels_piles = 4) (h6 : nickels_per_pile = 4)
                    (h7 : pennies_piles = 3) (h8 : pennies_per_pile = 6) :
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 116 := by
  sorry

end total_coins_l2887_288760


namespace f_inv_composition_l2887_288747

-- Define the function f
def f : ℕ → ℕ
| 2 => 5
| 3 => 7
| 4 => 11
| 5 => 17
| 6 => 23
| 7 => 40  -- Extended definition
| _ => 0   -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 5 => 2
| 7 => 3
| 11 => 4
| 17 => 5
| 23 => 6
| 40 => 7  -- Extended definition
| _ => 0   -- Default case for other inputs

-- Theorem statement
theorem f_inv_composition : f_inv ((f_inv 23)^2 + (f_inv 5)^2) = 7 := by
  sorry

end f_inv_composition_l2887_288747


namespace yellow_ball_probability_l2887_288724

/-- Represents a ball with a color and label -/
structure Ball where
  color : String
  label : Char

/-- Represents the bag of balls -/
def bag : List Ball := [
  { color := "yellow", label := 'a' },
  { color := "yellow", label := 'b' },
  { color := "red", label := 'c' },
  { color := "red", label := 'd' }
]

/-- Calculates the probability of drawing a yellow ball on the first draw -/
def probYellowFirst (bag : List Ball) : ℚ :=
  (bag.filter (fun b => b.color = "yellow")).length / bag.length

/-- Calculates the probability of drawing a yellow ball on the second draw -/
def probYellowSecond (bag : List Ball) : ℚ :=
  let totalOutcomes := bag.length * (bag.length - 1)
  let favorableOutcomes := 2 * (bag.length - 2)
  favorableOutcomes / totalOutcomes

theorem yellow_ball_probability (bag : List Ball) :
  probYellowFirst bag = 1/2 ∧ probYellowSecond bag = 1/2 :=
sorry

end yellow_ball_probability_l2887_288724


namespace increase_by_fifty_percent_l2887_288741

theorem increase_by_fifty_percent (initial : ℝ) (increase : ℝ) (result : ℝ) : 
  initial = 350 → increase = 0.5 → result = initial * (1 + increase) → result = 525 := by
  sorry

end increase_by_fifty_percent_l2887_288741


namespace inequality_solution_l2887_288738

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (2 - x)) ∧
  (∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → (x₁ - x₂) / (f x₁ - f x₂) > 0)

/-- The solution set of the inequality -/
def solution_set (x : ℝ) : Prop :=
  x ≤ 0 ∨ x ≥ 4/3

/-- Theorem stating the solution of the inequality -/
theorem inequality_solution (f : ℝ → ℝ) (h : special_function f) :
  ∀ x, f (2*x - 1) - f (3 - x) ≥ 0 ↔ solution_set x :=
sorry

end inequality_solution_l2887_288738


namespace spherical_to_rectangular_conversion_l2887_288762

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 4
  let θ : ℝ := 5 * π / 3
  let φ : ℝ := π / 2
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (2, -2 * Real.sqrt 3, 0) := by sorry

end spherical_to_rectangular_conversion_l2887_288762


namespace leaf_collection_time_l2887_288768

/-- Represents the leaf collection problem --/
structure LeafCollection where
  totalLeaves : ℕ
  collectionRate : ℕ
  scatterRate : ℕ
  cycleTime : ℕ

/-- Calculates the time needed to collect all leaves --/
def collectionTime (lc : LeafCollection) : ℚ :=
  let netIncrease := lc.collectionRate - lc.scatterRate
  let cycles := (lc.totalLeaves - lc.scatterRate) / netIncrease
  let totalSeconds := (cycles + 1) * lc.cycleTime
  totalSeconds / 60

/-- Theorem stating that the collection time for the given problem is 21.5 minutes --/
theorem leaf_collection_time :
  let lc : LeafCollection := {
    totalLeaves := 45,
    collectionRate := 4,
    scatterRate := 3,
    cycleTime := 30
  }
  collectionTime lc = 21.5 := by sorry

end leaf_collection_time_l2887_288768


namespace det_A_eq_16_l2887_288754

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2, -4; 6, -1, 3; 2, -3, 5]

theorem det_A_eq_16 : Matrix.det A = 16 := by
  sorry

end det_A_eq_16_l2887_288754


namespace complement_of_union_l2887_288721

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {4, 5}
def B : Set Nat := {3, 4}

theorem complement_of_union :
  (A ∪ B)ᶜ = {1, 2, 6} := by sorry

end complement_of_union_l2887_288721


namespace vector_sum_magnitude_l2887_288720

-- Define the vector space
variable (V : Type) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors and their properties
variable (a b : V)
variable (h1 : a = (1 : ℝ) • (1, 0))
variable (h2 : ‖b‖ = 1)
variable (h3 : inner a b = -(1/2 : ℝ) * ‖a‖ * ‖b‖)

-- State the theorem
theorem vector_sum_magnitude :
  ‖a + 2 • b‖ = Real.sqrt 3 :=
sorry

end vector_sum_magnitude_l2887_288720


namespace parallelogram_bisector_slope_l2887_288755

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Checks if a line through the origin bisects a parallelogram into two congruent polygons -/
def bisects_parallelogram (m n : ℕ) (p : Parallelogram) : Prop :=
  -- This is a placeholder for the actual condition
  True

/-- The main theorem -/
theorem parallelogram_bisector_slope (p : Parallelogram) :
  p.v1 = ⟨20, 90⟩ ∧
  p.v2 = ⟨20, 228⟩ ∧
  p.v3 = ⟨56, 306⟩ ∧
  p.v4 = ⟨56, 168⟩ ∧
  bisects_parallelogram 369 76 p →
  369 / 76 = (p.v3.y - p.v1.y) / (p.v3.x - p.v1.x) :=
sorry

end parallelogram_bisector_slope_l2887_288755


namespace earliest_retirement_year_l2887_288700

/-- Rule of 70 retirement provision -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Employee's age in a given year -/
def age_in_year (hire_year : ℕ) (hire_age : ℕ) (current_year : ℕ) : ℕ :=
  (current_year - hire_year) + hire_age

/-- Employee's years of employment in a given year -/
def years_employed (hire_year : ℕ) (current_year : ℕ) : ℕ :=
  current_year - hire_year

theorem earliest_retirement_year 
  (hire_year : ℕ) 
  (hire_age : ℕ) 
  (retirement_year : ℕ) :
  hire_year = 1988 →
  hire_age = 32 →
  retirement_year = 2007 →
  rule_of_70 (age_in_year hire_year hire_age retirement_year) 
             (years_employed hire_year retirement_year) ∧
  ∀ y : ℕ, y < retirement_year →
    ¬(rule_of_70 (age_in_year hire_year hire_age y) 
                 (years_employed hire_year y)) :=
by sorry

end earliest_retirement_year_l2887_288700


namespace polynomial_product_identity_l2887_288771

theorem polynomial_product_identity (x z : ℝ) :
  (3 * x^4 - 4 * z^3) * (9 * x^8 + 12 * x^4 * z^3 + 16 * z^6) = 27 * x^12 - 64 * z^9 := by
  sorry

end polynomial_product_identity_l2887_288771


namespace largest_divisor_power_l2887_288766

-- Define the expression A
def A : ℕ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

-- State the theorem
theorem largest_divisor_power (k : ℕ) : (∀ m : ℕ, m > k → ¬(1991^m ∣ A)) ∧ (1991^k ∣ A) ↔ k = 1991 := by
  sorry

end largest_divisor_power_l2887_288766


namespace smallest_x_satisfying_equations_l2887_288706

theorem smallest_x_satisfying_equations : 
  ∃ x : ℝ, x = -12 ∧ 
    abs (x - 3) = 15 ∧ 
    abs (x + 2) = 10 ∧ 
    ∀ y : ℝ, (abs (y - 3) = 15 ∧ abs (y + 2) = 10) → y ≥ x :=
by sorry

end smallest_x_satisfying_equations_l2887_288706


namespace choose_computers_l2887_288726

theorem choose_computers (n : ℕ) : 
  (Nat.choose 3 2 * Nat.choose 3 1) + (Nat.choose 3 1 * Nat.choose 3 2) = 18 :=
by sorry

end choose_computers_l2887_288726


namespace tangents_and_line_of_tangency_l2887_288710

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

-- Define point P
def P : ℝ × ℝ := (-2, 3)

-- Define the tangent lines
def tangent1 (x y : ℝ) : Prop := (Real.sqrt 3 + 6) * x - 4 * y + 2 * Real.sqrt 3 - 3 = 0
def tangent2 (x y : ℝ) : Prop := (3 + Real.sqrt 3) * x + 4 * y - 6 + 2 * Real.sqrt 3 = 0

-- Define the line passing through points of tangency
def tangencyLine (x y : ℝ) : Prop := 3 * x - 2 * y - 3 = 0

theorem tangents_and_line_of_tangency :
  ∃ (M N : ℝ × ℝ),
    M ∈ C ∧ N ∈ C ∧
    (tangent1 M.1 M.2 ∨ tangent2 M.1 M.2) ∧
    (tangent1 N.1 N.2 ∨ tangent2 N.1 N.2) ∧
    tangencyLine M.1 M.2 ∧
    tangencyLine N.1 N.2 :=
by sorry

end tangents_and_line_of_tangency_l2887_288710


namespace equation_solution_l2887_288718

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 1 ∧ (2 * x) / (x - 1) - 1 = 4 / (1 - x) := by
  sorry

end equation_solution_l2887_288718


namespace decreasing_implies_positive_a_l2887_288719

/-- The function f(x) = a(x^3 - 3x) is decreasing on the interval (-1, 1) --/
def is_decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y

/-- The main theorem: if f(x) = a(x^3 - 3x) is decreasing on (-1, 1), then a > 0 --/
theorem decreasing_implies_positive_a (a : ℝ) :
  is_decreasing_on_interval (fun x => a * (x^3 - 3*x)) → a > 0 :=
by sorry

end decreasing_implies_positive_a_l2887_288719


namespace parallel_vectors_tan_alpha_l2887_288701

/-- Given two parallel vectors a and b, where a = (6, 8) and b = (sinα, cosα), prove that tanα = 3/4 -/
theorem parallel_vectors_tan_alpha (α : Real) : 
  let a : Fin 2 → Real := ![6, 8]
  let b : Fin 2 → Real := ![Real.sin α, Real.cos α]
  (∃ (k : Real), k ≠ 0 ∧ (∀ i, a i = k * b i)) → 
  Real.tan α = 3/4 := by
  sorry

end parallel_vectors_tan_alpha_l2887_288701


namespace condition_neither_sufficient_nor_necessary_l2887_288772

theorem condition_neither_sufficient_nor_necessary
  (m n : ℕ+) :
  ¬(∀ a b : ℝ, a > b → (a^(m:ℕ) - b^(m:ℕ)) * (a^(n:ℕ) - b^(n:ℕ)) > 0) ∧
  ¬(∀ a b : ℝ, (a^(m:ℕ) - b^(m:ℕ)) * (a^(n:ℕ) - b^(n:ℕ)) > 0 → a > b) :=
by sorry

end condition_neither_sufficient_nor_necessary_l2887_288772


namespace households_without_car_or_bike_l2887_288704

theorem households_without_car_or_bike
  (total : ℕ)
  (both : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (h_total : total = 90)
  (h_both : both = 14)
  (h_car : car = 44)
  (h_bike_only : bike_only = 35) :
  total - (car + bike_only + both) = 11 :=
by sorry

end households_without_car_or_bike_l2887_288704


namespace right_triangle_sides_l2887_288779

theorem right_triangle_sides (p Δ : ℝ) (hp : p > 0) (hΔ : Δ > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = p ∧
    a * b = 2 * Δ ∧
    c^2 = a^2 + b^2 ∧
    a = (p - (p^2 - 4*Δ)/(2*p) + ((p - (p^2 - 4*Δ)/(2*p))^2 - 8*Δ).sqrt) / 2 ∧
    b = (p - (p^2 - 4*Δ)/(2*p) - ((p - (p^2 - 4*Δ)/(2*p))^2 - 8*Δ).sqrt) / 2 :=
by sorry

end right_triangle_sides_l2887_288779


namespace quadratic_completion_of_square_l2887_288775

theorem quadratic_completion_of_square :
  ∀ x : ℝ, x^2 + 2*x + 3 = (x + 1)^2 + 2 := by
sorry

end quadratic_completion_of_square_l2887_288775


namespace cube_with_cylindrical_hole_l2887_288703

/-- The surface area of a cube with a cylindrical hole --/
def surface_area (cube_edge : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) (π : ℝ) : ℝ :=
  6 * cube_edge^2 - 2 * π * cylinder_radius^2 + 2 * π * cylinder_radius * cylinder_height

/-- The volume of a cube with a cylindrical hole --/
def volume (cube_edge : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) (π : ℝ) : ℝ :=
  cube_edge^3 - π * cylinder_radius^2 * cylinder_height

/-- Theorem stating the surface area and volume of the resulting geometric figure --/
theorem cube_with_cylindrical_hole :
  let cube_edge : ℝ := 10
  let cylinder_radius : ℝ := 2
  let cylinder_height : ℝ := 10
  let π : ℝ := 3
  surface_area cube_edge cylinder_radius cylinder_height π = 696 ∧
  volume cube_edge cylinder_radius cylinder_height π = 880 := by
  sorry

end cube_with_cylindrical_hole_l2887_288703


namespace survey_income_problem_l2887_288789

/-- Proves that given the conditions from the survey, the average income of the other 40 customers is $42,500 -/
theorem survey_income_problem (total_customers : ℕ) (wealthy_customers : ℕ) 
  (total_avg_income : ℝ) (wealthy_avg_income : ℝ) :
  total_customers = 50 →
  wealthy_customers = 10 →
  total_avg_income = 45000 →
  wealthy_avg_income = 55000 →
  let other_customers := total_customers - wealthy_customers
  let total_income := total_avg_income * total_customers
  let wealthy_income := wealthy_avg_income * wealthy_customers
  let other_income := total_income - wealthy_income
  other_income / other_customers = 42500 := by
sorry

end survey_income_problem_l2887_288789


namespace basketballs_with_holes_l2887_288791

/-- Given the number of soccer balls and basketballs, the number of soccer balls with holes,
    and the total number of balls without holes, calculate the number of basketballs with holes. -/
theorem basketballs_with_holes
  (total_soccer : ℕ)
  (total_basketball : ℕ)
  (soccer_with_holes : ℕ)
  (total_without_holes : ℕ)
  (h1 : total_soccer = 40)
  (h2 : total_basketball = 15)
  (h3 : soccer_with_holes = 30)
  (h4 : total_without_holes = 18) :
  total_basketball - (total_without_holes - (total_soccer - soccer_with_holes)) = 7 := by
  sorry


end basketballs_with_holes_l2887_288791


namespace tuna_sales_difference_l2887_288778

/-- Calculates the difference in daily revenue between peak and low seasons for tuna fish sales. -/
theorem tuna_sales_difference (peak_rate : ℕ) (low_rate : ℕ) (price : ℕ) (hours : ℕ) : 
  peak_rate = 6 → low_rate = 4 → price = 60 → hours = 15 →
  (peak_rate * price * hours) - (low_rate * price * hours) = 1800 := by
  sorry

end tuna_sales_difference_l2887_288778


namespace parabola_equation_for_given_focus_and_directrix_l2887_288786

/-- A parabola is defined by a focus point and a directrix line parallel to the x-axis. -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- The equation of a parabola given its focus and directrix. -/
def parabola_equation (p : Parabola) : ℝ → ℝ → Prop :=
  λ x y => x^2 = 4 * (p.focus.2 - p.directrix) * (y - (p.focus.2 + p.directrix) / 2)

theorem parabola_equation_for_given_focus_and_directrix :
  let p : Parabola := { focus := (0, 4), directrix := -4 }
  ∀ x y : ℝ, parabola_equation p x y ↔ x^2 = 16 * y := by
  sorry

end parabola_equation_for_given_focus_and_directrix_l2887_288786


namespace sin_4x_eq_sin_2x_solution_l2887_288756

open Set
open Real

def solution_set : Set ℝ := {π/6, π/2, π, 5*π/6, 7*π/6}

theorem sin_4x_eq_sin_2x_solution (x : ℝ) :
  x ∈ Ioo 0 (3*π/2) →
  (sin (4*x) = sin (2*x)) ↔ x ∈ solution_set :=
sorry

end sin_4x_eq_sin_2x_solution_l2887_288756


namespace milk_percentage_after_three_replacements_l2887_288723

/-- Represents the percentage of milk remaining after one replacement operation -/
def milk_after_one_replacement (initial_milk_percentage : Real) : Real :=
  initial_milk_percentage * 0.8

/-- Represents the percentage of milk remaining after three replacement operations -/
def milk_after_three_replacements (initial_milk_percentage : Real) : Real :=
  milk_after_one_replacement (milk_after_one_replacement (milk_after_one_replacement initial_milk_percentage))

theorem milk_percentage_after_three_replacements :
  milk_after_three_replacements 100 = 51.2 := by
  sorry

end milk_percentage_after_three_replacements_l2887_288723


namespace pencils_in_drawer_l2887_288796

/-- Given a drawer with initial pencils and some taken out, calculate the remaining pencils -/
def remaining_pencils (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem: If there were 9 pencils initially and 4 were taken out, 5 pencils remain -/
theorem pencils_in_drawer : remaining_pencils 9 4 = 5 := by
  sorry

end pencils_in_drawer_l2887_288796


namespace intersection_points_max_distance_values_l2887_288729

-- Define the line l
def line_l (a t : ℝ) : ℝ × ℝ := (a + 2*t, 1 - t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Part 1: Intersection points
theorem intersection_points :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := line_l (-2) t₁
    let (x₂, y₂) := line_l (-2) t₂
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    x₁ = -4*Real.sqrt 5/5 ∧ y₁ = 2*Real.sqrt 5/5 ∧
    x₂ = 4*Real.sqrt 5/5 ∧ y₂ = -2*Real.sqrt 5/5 :=
sorry

-- Part 2: Values of a
theorem max_distance_values :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), curve_C x y →
      (|x + 2*y - 2 - a| / Real.sqrt 5 ≤ 2 * Real.sqrt 5)) ∧
    (∃ (x y : ℝ), curve_C x y ∧
      |x + 2*y - 2 - a| / Real.sqrt 5 = 2 * Real.sqrt 5) →
    (a = 8 - 2*Real.sqrt 5 ∨ a = 2*Real.sqrt 5 - 12) :=
sorry

end intersection_points_max_distance_values_l2887_288729


namespace polynomial_roots_l2887_288736

theorem polynomial_roots : ∃ (x₁ x₂ x₃ x₄ : ℂ),
  (x₁ = (7 + Real.sqrt 37) / 6) ∧
  (x₂ = (7 - Real.sqrt 37) / 6) ∧
  (x₃ = (-3 + Real.sqrt 5) / 2) ∧
  (x₄ = (-3 - Real.sqrt 5) / 2) ∧
  (∀ x : ℂ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) :=
by sorry

end polynomial_roots_l2887_288736


namespace second_number_value_l2887_288748

theorem second_number_value (x : ℝ) (h : 8000 * x = 480 * (10^5)) : x = 6000 := by
  sorry

end second_number_value_l2887_288748


namespace polynomial_factorization_l2887_288797

theorem polynomial_factorization (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end polynomial_factorization_l2887_288797


namespace shoe_probability_theorem_l2887_288753

/-- Represents the number of pairs of shoes of a specific color -/
structure ColorPairs :=
  (count : ℕ)

/-- Represents Sue's shoe collection -/
structure ShoeCollection :=
  (total_pairs : ℕ)
  (black : ColorPairs)
  (brown : ColorPairs)
  (gray : ColorPairs)

/-- Calculates the probability of picking two shoes of the same color, one left and one right -/
def probability_same_color_different_feet (collection : ShoeCollection) : ℚ :=
  let total_shoes := 2 * collection.total_pairs
  let black_prob := (2 * collection.black.count : ℚ) / total_shoes * collection.black.count / (total_shoes - 1)
  let brown_prob := (2 * collection.brown.count : ℚ) / total_shoes * collection.brown.count / (total_shoes - 1)
  let gray_prob := (2 * collection.gray.count : ℚ) / total_shoes * collection.gray.count / (total_shoes - 1)
  black_prob + brown_prob + gray_prob

theorem shoe_probability_theorem (sue_collection : ShoeCollection) 
  (h1 : sue_collection.total_pairs = 12)
  (h2 : sue_collection.black.count = 7)
  (h3 : sue_collection.brown.count = 3)
  (h4 : sue_collection.gray.count = 2) :
  probability_same_color_different_feet sue_collection = 31 / 138 := by
  sorry

end shoe_probability_theorem_l2887_288753


namespace right_triangle_third_side_square_l2887_288746

theorem right_triangle_third_side_square (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ a^2 + b^2 = c^2) ∨ (a = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) →
  c^2 = 25 ∨ b^2 = 7 :=
by sorry

end right_triangle_third_side_square_l2887_288746


namespace ellipse_equation_and_sum_l2887_288730

theorem ellipse_equation_and_sum (t : ℝ) :
  let x := (3 * (Real.sin t - 2)) / (3 - Real.cos t)
  let y := (4 * (Real.cos t - 6)) / (3 - Real.cos t)
  ∃ (A B C D E F : ℤ),
    (144 : ℝ) * x^2 - 96 * x * y + 25 * y^2 + 192 * x - 400 * y + 400 = 0 ∧
    Int.gcd A (Int.gcd B (Int.gcd C (Int.gcd D (Int.gcd E F)))) = 1 ∧
    Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1257 :=
by
  sorry

end ellipse_equation_and_sum_l2887_288730


namespace intersection_equals_S_l2887_288713

def S : Set ℝ := {y | ∃ x : ℝ, y = 3 * x}
def T : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

theorem intersection_equals_S : S ∩ T = S := by sorry

end intersection_equals_S_l2887_288713


namespace arithmetic_geometric_sum_l2887_288725

/-- An arithmetic-geometric sequence -/
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r q : ℝ, ∀ n : ℕ, a (n + 1) = r * a n + q

/-- The statement to be proven -/
theorem arithmetic_geometric_sum (a : ℕ → ℝ) :
  arithmetic_geometric_sequence a →
  a 4 + a 6 = 5 →
  a 4 * a 6 = 6 →
  a 3 * a 5 + a 5 * a 7 = 13 := by
  sorry

end arithmetic_geometric_sum_l2887_288725


namespace logarithmic_equality_l2887_288787

noncomputable def log_expr1 (x : ℝ) : ℝ := Real.log ((7 * x / 2) - (17 / 4)) / Real.log ((x / 2 + 1)^2)

noncomputable def log_expr2 (x : ℝ) : ℝ := Real.log ((3 * x / 2) - 6)^2 / Real.log (((7 * x / 2) - (17 / 4))^(1/2))

noncomputable def log_expr3 (x : ℝ) : ℝ := Real.log (x / 2 + 1) / Real.log (((3 * x / 2) - 6)^(1/2))

theorem logarithmic_equality (x : ℝ) :
  (log_expr1 x = log_expr2 x ∧ log_expr1 x = log_expr3 x + 1) ∨
  (log_expr2 x = log_expr3 x ∧ log_expr2 x = log_expr1 x + 1) ∨
  (log_expr3 x = log_expr1 x ∧ log_expr3 x = log_expr2 x + 1) ↔
  x = 7 :=
sorry

end logarithmic_equality_l2887_288787


namespace sum_solution_equation_value_l2887_288788

/-- A sum solution equation is an equation of the form a/x = b where the solution for x is 1/(a+b) -/
def IsSumSolutionEquation (a b : ℚ) : Prop :=
  ∀ x, a / x = b ↔ x = 1 / (a + b)

/-- The main theorem: if n/x = 3-n is a sum solution equation, then n = 3/4 -/
theorem sum_solution_equation_value (n : ℚ) :
  IsSumSolutionEquation n (3 - n) → n = 3 / 4 := by
  sorry

end sum_solution_equation_value_l2887_288788


namespace symmetry_implies_sum_l2887_288734

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_wrt_origin (2*a + 1) 4 1 (3*b - 1) → 2*a + b = -3 := by
  sorry

end symmetry_implies_sum_l2887_288734


namespace sugar_recipe_calculation_l2887_288793

theorem sugar_recipe_calculation (initial_required : ℚ) (available : ℚ) : 
  initial_required = 1/3 → available = 1/6 → 
  (initial_required - available = 1/6) ∧ (2 * (initial_required - available) = 1/3) := by
  sorry

end sugar_recipe_calculation_l2887_288793


namespace negative_integer_square_plus_self_twelve_l2887_288790

theorem negative_integer_square_plus_self_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N % 3 = 0 → N = -3 := by sorry

end negative_integer_square_plus_self_twelve_l2887_288790


namespace unique_angle_satisfying_conditions_l2887_288794

theorem unique_angle_satisfying_conditions :
  ∃! x : ℝ, 0 ≤ x ∧ x < 2 * π ∧ 
    Real.sin x = -(1/2) ∧ Real.cos x = Real.sqrt 3 / 2 := by
  sorry

end unique_angle_satisfying_conditions_l2887_288794


namespace square_root_of_four_l2887_288750

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end square_root_of_four_l2887_288750
