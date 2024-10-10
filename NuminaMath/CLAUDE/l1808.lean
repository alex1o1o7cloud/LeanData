import Mathlib

namespace smallest_number_l1808_180859

-- Define the numbers
def A : ℝ := 5.67823
def B : ℝ := 5.678333333 -- Approximation of 5.678̅3
def C : ℝ := 5.678383838 -- Approximation of 5.67̅83
def D : ℝ := 5.678378378 -- Approximation of 5.6̅783
def E : ℝ := 5.678367836 -- Approximation of 5.̅6783

-- Theorem statement
theorem smallest_number : E < A ∧ E < B ∧ E < C ∧ E < D :=
by sorry

end smallest_number_l1808_180859


namespace lance_reading_plan_l1808_180823

/-- Represents the number of pages read on each day -/
structure ReadingPlan where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ

/-- Checks if a reading plan is valid according to the given conditions -/
def isValidPlan (plan : ReadingPlan) (totalPages : ℕ) : Prop :=
  plan.day2 = plan.day1 - 5 ∧
  plan.day3 = 35 ∧
  plan.day1 + plan.day2 + plan.day3 = totalPages

theorem lance_reading_plan (totalPages : ℕ) (h : totalPages = 100) :
  ∃ (plan : ReadingPlan), isValidPlan plan totalPages ∧ plan.day1 = 35 := by
  sorry

end lance_reading_plan_l1808_180823


namespace sum_is_composite_l1808_180856

theorem sum_is_composite (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end sum_is_composite_l1808_180856


namespace average_mark_calculation_l1808_180836

theorem average_mark_calculation (students_class1 students_class2 : ℕ) 
  (avg_class2 avg_total : ℚ) : 
  students_class1 = 20 →
  students_class2 = 50 →
  avg_class2 = 60 →
  avg_total = 54.285714285714285 →
  (students_class1 * (avg_total * (students_class1 + students_class2) - students_class2 * avg_class2)) / 
   (students_class1 * (students_class1 + students_class2)) = 40 := by
  sorry

end average_mark_calculation_l1808_180836


namespace number_division_problem_l1808_180835

theorem number_division_problem :
  ∃ x : ℝ, (x / 5 = 60 + x / 6) ∧ (x = 1800) := by
  sorry

end number_division_problem_l1808_180835


namespace right_triangle_squares_area_l1808_180875

theorem right_triangle_squares_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  c^2 = 1009 →
  4*a^2 + 4*b^2 + 4*c^2 = 8072 := by sorry

end right_triangle_squares_area_l1808_180875


namespace geometric_sequence_seventh_term_l1808_180855

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₁ = -16 and a₄ = 8, prove that a₇ = -4 -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_a1 : a 1 = -16)
  (h_a4 : a 4 = 8) :
  a 7 = -4 := by
  sorry

end geometric_sequence_seventh_term_l1808_180855


namespace min_square_value_l1808_180812

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ m : ℕ+, (15 * a + 16 * b : ℕ) = m^2)
  (h2 : ∃ n : ℕ+, (16 * a - 15 * b : ℕ) = n^2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 :=
sorry

end min_square_value_l1808_180812


namespace greatest_three_digit_multiple_of_17_l1808_180861

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l1808_180861


namespace fencing_cost_per_metre_l1808_180807

/-- Proof of fencing cost per metre for a rectangular field -/
theorem fencing_cost_per_metre
  (ratio_length_width : ℚ) -- Ratio of length to width
  (area : ℝ) -- Area of the field in square meters
  (total_cost : ℝ) -- Total cost of fencing
  (h_ratio : ratio_length_width = 3 / 4) -- The ratio of length to width is 3:4
  (h_area : area = 10092) -- The area is 10092 sq. m
  (h_cost : total_cost = 101.5) -- The total cost is 101.5
  : ∃ (length width : ℝ),
    length / width = ratio_length_width ∧
    length * width = area ∧
    (2 * (length + width)) * (total_cost / (2 * (length + width))) = total_cost ∧
    total_cost / (2 * (length + width)) = 0.25 :=
by sorry

end fencing_cost_per_metre_l1808_180807


namespace not_divisible_by_2310_l1808_180894

theorem not_divisible_by_2310 (n : ℕ) (h : n < 2310) : ¬(2310 ∣ n * (2310 - n)) := by
  sorry

end not_divisible_by_2310_l1808_180894


namespace charlottes_schedule_is_correct_l1808_180839

/-- Represents the number of hours it takes to walk each type of dog -/
structure WalkingTime where
  poodle : ℕ
  chihuahua : ℕ
  labrador : ℕ

/-- Represents the schedule for the week -/
structure Schedule where
  monday_poodles : ℕ
  monday_chihuahuas : ℕ
  tuesday_chihuahuas : ℕ
  wednesday_labradors : ℕ

/-- The total available hours for dog-walking in the week -/
def total_hours : ℕ := 32

/-- The walking times for each type of dog -/
def walking_times : WalkingTime := {
  poodle := 2,
  chihuahua := 1,
  labrador := 3
}

/-- Charlotte's schedule for the week -/
def charlottes_schedule : Schedule := {
  monday_poodles := 8,  -- This is what we want to prove
  monday_chihuahuas := 2,
  tuesday_chihuahuas := 2,
  wednesday_labradors := 4
}

/-- Calculate the total hours spent walking dogs based on the schedule and walking times -/
def calculate_total_hours (s : Schedule) (w : WalkingTime) : ℕ :=
  s.monday_poodles * w.poodle +
  s.monday_chihuahuas * w.chihuahua +
  s.tuesday_chihuahuas * w.chihuahua +
  s.wednesday_labradors * w.labrador

/-- Theorem stating that Charlotte's schedule is correct -/
theorem charlottes_schedule_is_correct :
  calculate_total_hours charlottes_schedule walking_times = total_hours :=
by sorry

end charlottes_schedule_is_correct_l1808_180839


namespace mama_bird_worms_l1808_180881

/-- The number of additional worms Mama bird needs to catch -/
def additional_worms_needed (num_babies : ℕ) (worms_per_baby_per_day : ℕ) (days : ℕ) 
  (papa_worms : ℕ) (mama_worms : ℕ) (stolen_worms : ℕ) : ℕ :=
  num_babies * worms_per_baby_per_day * days - (papa_worms + mama_worms - stolen_worms)

/-- Theorem stating that Mama bird needs to catch 34 more worms -/
theorem mama_bird_worms : 
  additional_worms_needed 6 3 3 9 13 2 = 34 := by sorry

end mama_bird_worms_l1808_180881


namespace mikes_age_l1808_180842

theorem mikes_age (claire_age jessica_age mike_age : ℕ) : 
  jessica_age = claire_age + 6 →
  claire_age + 2 = 20 →
  mike_age = 2 * (jessica_age - 3) →
  mike_age = 42 := by
  sorry

end mikes_age_l1808_180842


namespace class_fund_problem_l1808_180880

theorem class_fund_problem (total_amount : ℕ) (twenty_bill_count : ℕ) (other_bill_count : ℕ) 
  (h1 : total_amount = 120)
  (h2 : other_bill_count = 2 * twenty_bill_count)
  (h3 : twenty_bill_count = 3) :
  total_amount - (twenty_bill_count * 20) = 60 := by
  sorry

end class_fund_problem_l1808_180880


namespace tea_consumption_discrepancy_l1808_180849

theorem tea_consumption_discrepancy 
  (box_size : ℕ) 
  (cups_per_bag_min cups_per_bag_max : ℕ) 
  (darya_cups marya_cups : ℕ) :
  cups_per_bag_min = 3 →
  cups_per_bag_max = 4 →
  darya_cups = 74 →
  marya_cups = 105 →
  (∃ n : ℕ, n * cups_per_bag_min ≤ darya_cups ∧ darya_cups < (n + 1) * cups_per_bag_min ∧
            n * cups_per_bag_min ≤ marya_cups ∧ marya_cups < (n + 1) * cups_per_bag_min) →
  (∃ m : ℕ, m * cups_per_bag_max ≤ darya_cups ∧ darya_cups < (m + 1) * cups_per_bag_max ∧
            m * cups_per_bag_max ≤ marya_cups ∧ marya_cups < (m + 1) * cups_per_bag_max) →
  False :=
by sorry

end tea_consumption_discrepancy_l1808_180849


namespace dolls_made_l1808_180837

def accessories_per_doll : ℕ := 2 + 3 + 1 + 5

def time_per_doll_and_accessories : ℕ := 45 + accessories_per_doll * 10

def total_operation_time : ℕ := 1860000

theorem dolls_made : 
  total_operation_time / time_per_doll_and_accessories = 12000 := by sorry

end dolls_made_l1808_180837


namespace a_nonzero_sufficient_not_necessary_l1808_180803

/-- A cubic polynomial function -/
def cubic_polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The property that a cubic polynomial has a root -/
def has_root (a b c d : ℝ) : Prop := ∃ x : ℝ, cubic_polynomial a b c d x = 0

/-- The statement that "a≠0" is sufficient but not necessary for a cubic polynomial to have a root -/
theorem a_nonzero_sufficient_not_necessary :
  (∀ a b c d : ℝ, a ≠ 0 → has_root a b c d) ∧
  ¬(∀ a b c d : ℝ, has_root a b c d → a ≠ 0) :=
sorry

end a_nonzero_sufficient_not_necessary_l1808_180803


namespace bella_pizza_consumption_l1808_180899

theorem bella_pizza_consumption 
  (rachel_pizza : ℕ) 
  (total_pizza : ℕ) 
  (h1 : rachel_pizza = 598)
  (h2 : total_pizza = 952) :
  total_pizza - rachel_pizza = 354 := by
sorry

end bella_pizza_consumption_l1808_180899


namespace tetrahedron_edge_length_l1808_180841

/-- Configuration of five spheres with a tetrahedron -/
structure SpheresTetrahedron where
  /-- Radius of each sphere -/
  radius : ℝ
  /-- Distance between centers of adjacent spheres on the square -/
  square_side : ℝ
  /-- Height of the top sphere's center above the square -/
  height : ℝ
  /-- Edge length of the tetrahedron -/
  tetra_edge : ℝ
  /-- The radius is 2 -/
  radius_eq : radius = 2
  /-- The square side is twice the diameter -/
  square_side_eq : square_side = 4 * radius
  /-- The height is equal to the diameter -/
  height_eq : height = 2 * radius
  /-- The tetrahedron edge is the distance from a lower sphere to the top sphere -/
  tetra_edge_eq : tetra_edge ^ 2 = square_side ^ 2 + height ^ 2

/-- Theorem: The edge length of the tetrahedron is 4√2 -/
theorem tetrahedron_edge_length (config : SpheresTetrahedron) : 
  config.tetra_edge = 4 * Real.sqrt 2 := by
  sorry

end tetrahedron_edge_length_l1808_180841


namespace slide_wait_time_l1808_180893

theorem slide_wait_time (kids_swings : ℕ) (kids_slide : ℕ) (swing_wait_min : ℕ) (time_diff_sec : ℕ) :
  kids_swings = 3 →
  kids_slide = 2 * kids_swings →
  swing_wait_min = 2 →
  (kids_slide * swing_wait_min * 60 + time_diff_sec) - (kids_swings * swing_wait_min * 60) = 270 →
  kids_slide * swing_wait_min * 60 + time_diff_sec = 630 :=
by
  sorry

#check slide_wait_time

end slide_wait_time_l1808_180893


namespace prop_false_implies_a_lt_neg_13_div_2_l1808_180851

theorem prop_false_implies_a_lt_neg_13_div_2 (a : ℝ) :
  (¬ ∀ x ∈ Set.Icc 1 2, x^2 + a*x + 9 ≥ 0) → a < -13/2 := by
  sorry

end prop_false_implies_a_lt_neg_13_div_2_l1808_180851


namespace middle_group_frequency_l1808_180890

theorem middle_group_frequency 
  (sample_size : ℕ) 
  (num_rectangles : ℕ) 
  (middle_area_ratio : ℚ) : 
  sample_size = 300 →
  num_rectangles = 9 →
  middle_area_ratio = 1/5 →
  (middle_area_ratio * (1 - middle_area_ratio / (1 + middle_area_ratio))) * sample_size = 50 :=
by sorry

end middle_group_frequency_l1808_180890


namespace product_112_54_l1808_180864

theorem product_112_54 : 112 * 54 = 6048 := by
  sorry

end product_112_54_l1808_180864


namespace hyperbola_m_range_l1808_180827

-- Define the condition for a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m - 2) = 1

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ Set.Ioo (-2 : ℝ) 2 :=
by sorry

end hyperbola_m_range_l1808_180827


namespace train_crossing_time_l1808_180871

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 150 ∧ 
  train_speed_kmh = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 6 := by
  sorry

#check train_crossing_time

end train_crossing_time_l1808_180871


namespace max_rational_products_l1808_180888

/-- Represents a table with rational and irrational numbers as labels -/
structure LabeledTable where
  size : ℕ
  rowLabels : Fin size → ℝ
  colLabels : Fin size → ℝ
  distinctLabels : ∀ i j, (rowLabels i = colLabels j) → i = j
  rationalCount : ℕ
  irrationalCount : ℕ
  labelCounts : rationalCount + irrationalCount = size + size

/-- Counts the number of rational products in the table -/
def countRationalProducts (t : LabeledTable) : ℕ :=
  sorry

/-- Theorem stating the maximum number of rational products -/
theorem max_rational_products (t : LabeledTable) : 
  t.size = 50 ∧ t.rationalCount = 50 ∧ t.irrationalCount = 50 → 
  countRationalProducts t ≤ 1275 :=
sorry

end max_rational_products_l1808_180888


namespace crate_tower_probability_l1808_180850

def crate_dimensions := (3, 4, 6)
def num_crates := 11
def target_height := 50

def valid_arrangements (a b c : ℕ) : ℕ :=
  if a + b + c = num_crates ∧ 3 * a + 4 * b + 6 * c = target_height
  then Nat.factorial num_crates / (Nat.factorial a * Nat.factorial b * Nat.factorial c)
  else 0

def total_valid_arrangements : ℕ :=
  valid_arrangements 4 2 5 + valid_arrangements 2 5 4 + valid_arrangements 0 8 3

def total_possible_arrangements : ℕ := 3^num_crates

theorem crate_tower_probability : 
  (total_valid_arrangements : ℚ) / total_possible_arrangements = 72 / 115 := by
  sorry

end crate_tower_probability_l1808_180850


namespace fishing_problem_l1808_180826

theorem fishing_problem (total fish_jason fish_ryan fish_jeffery : ℕ) : 
  total = 100 ∧ 
  fish_ryan = 3 * fish_jason ∧ 
  fish_jeffery = 2 * fish_ryan ∧ 
  total = fish_jason + fish_ryan + fish_jeffery →
  fish_jeffery = 60 := by
sorry

end fishing_problem_l1808_180826


namespace geometric_sequence_property_l1808_180860

theorem geometric_sequence_property (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∃ r > 0, ∀ n, a (n + 1) = r * a n)
  (h_sum : a 1 + a 2 + a 3 = 18)
  (h_inv_sum : 1 / a 1 + 1 / a 2 + 1 / a 3 = 2) :
  a 2 = 3 := by
sorry

end geometric_sequence_property_l1808_180860


namespace abc_sum_l1808_180802

theorem abc_sum (a b c : ℕ+) 
  (h1 : a * b + c = 57)
  (h2 : b * c + a = 57)
  (h3 : a * c + b = 57) : 
  a + b + c = 9 := by
sorry

end abc_sum_l1808_180802


namespace apple_bags_theorem_l1808_180852

/-- Represents the possible number of apples in a bag -/
inductive BagSize
| small : BagSize  -- 6 apples
| large : BagSize  -- 12 apples

/-- Returns true if the given number is a valid total number of apples -/
def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ ∃ (small large : ℕ), n = 6 * small + 12 * large

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
sorry

end apple_bags_theorem_l1808_180852


namespace sum_of_roots_cubic_equation_l1808_180862

theorem sum_of_roots_cubic_equation : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 12*x) / (x + 3)
  ∃ (a b : ℝ), (∀ x ≠ -3, f x = 3 ↔ x = a ∨ x = b) ∧ a + b = 4 := by
  sorry

end sum_of_roots_cubic_equation_l1808_180862


namespace probability_theorem_l1808_180854

/-- A permutation of the first n natural numbers -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The property that a permutation satisfies iₖ ≥ k - 3 for all k -/
def SatisfiesInequality (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, (p k : ℕ) + 1 ≥ k.val - 2

/-- The number of permutations satisfying the inequality -/
def CountSatisfyingPermutations (n : ℕ) : ℕ :=
  (4 ^ (n - 3)) * 6

/-- The probability theorem -/
theorem probability_theorem (n : ℕ) (h : n > 3) :
  (CountSatisfyingPermutations n : ℚ) / (Nat.factorial n) =
  (↑(4 ^ (n - 3) * 6) : ℚ) / (Nat.factorial n) := by
  sorry


end probability_theorem_l1808_180854


namespace cricket_team_average_age_l1808_180848

theorem cricket_team_average_age (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) :
  team_size = 11 →
  captain_age = 26 →
  wicket_keeper_age_diff = 3 →
  let total_age := team_size * (captain_age + wicket_keeper_age_diff + 2) / 2
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + captain_age + wicket_keeper_age_diff)
  (remaining_age / remaining_players) + 1 = total_age / team_size →
  total_age / team_size = 32 := by
sorry

end cricket_team_average_age_l1808_180848


namespace shaded_region_value_l1808_180876

/-- Rectangle PQRS with PS = 2 and PQ = 4 -/
structure Rectangle where
  ps : ℝ
  pq : ℝ
  h_ps : ps = 2
  h_pq : pq = 4

/-- Points T, U, V, W positioned so that RT = RU = PW = PV = a -/
def points_position (rect : Rectangle) (a : ℝ) : Prop :=
  ∃ (t u v w : ℝ × ℝ), 
    (rect.pq - a = t.1) ∧ (rect.pq - a = u.1) ∧ (a = v.1) ∧ (a = w.1) ∧
    (rect.ps = t.2) ∧ (0 = u.2) ∧ (rect.ps = v.2) ∧ (0 = w.2)

/-- VU and WT pass through the center of the rectangle -/
def lines_through_center (rect : Rectangle) (a : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), center = (rect.pq / 2, rect.ps / 2)

/-- The shaded region is 1/8 the area of PQRS -/
def shaded_region_ratio (rect : Rectangle) (a : ℝ) : Prop :=
  3 * a = 1/8 * (rect.ps * rect.pq)

/-- Main theorem -/
theorem shaded_region_value (rect : Rectangle) :
  points_position rect (1/3) ∧ 
  lines_through_center rect (1/3) ∧ 
  shaded_region_ratio rect (1/3) := by
  sorry

end shaded_region_value_l1808_180876


namespace unique_solution_implies_equal_absolute_values_l1808_180857

theorem unique_solution_implies_equal_absolute_values (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x, a * (x - a)^2 + b * (x - b)^2 = 0) → |a| = |b| :=
by sorry

end unique_solution_implies_equal_absolute_values_l1808_180857


namespace fred_final_cards_l1808_180824

/-- The number of baseball cards Fred has after various transactions -/
def fred_cards (initial : ℕ) (given_away : ℕ) (new_cards : ℕ) : ℕ :=
  initial - given_away + new_cards

/-- Theorem stating that Fred ends up with 48 cards given the specific numbers in the problem -/
theorem fred_final_cards : fred_cards 26 18 40 = 48 := by
  sorry

end fred_final_cards_l1808_180824


namespace unique_root_in_unit_interval_l1808_180820

theorem unique_root_in_unit_interval :
  ∃! α : ℝ, |α| < 1 ∧ α^3 - 2*α + 2 = 0 := by
sorry

end unique_root_in_unit_interval_l1808_180820


namespace stock_price_fluctuation_l1808_180805

theorem stock_price_fluctuation (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.4
  let decrease_factor := 1 - 0.2857
  decrease_factor * increased_price = original_price := by
  sorry

end stock_price_fluctuation_l1808_180805


namespace ana_bonita_age_difference_ana_bonita_age_difference_proof_l1808_180809

theorem ana_bonita_age_difference : ℕ → Prop := fun n =>
  ∀ (A B : ℕ),
    A = B + n →                    -- Ana is n years older than Bonita
    A - 1 = 3 * (B - 1) →          -- Last year Ana was 3 times as old as Bonita
    A = B * B →                    -- This year Ana's age is the square of Bonita's age
    n = 2                          -- The age difference is 2 years

-- The proof goes here
theorem ana_bonita_age_difference_proof : ana_bonita_age_difference 2 := by
  sorry

#check ana_bonita_age_difference_proof

end ana_bonita_age_difference_ana_bonita_age_difference_proof_l1808_180809


namespace quadratic_equation_solution_l1808_180870

theorem quadratic_equation_solution (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let x₁ : ℝ := 4*a/(3*b)
  let x₂ : ℝ := -3*b/(4*a)
  (12*a*b*x₁^2 - (16*a^2 - 9*b^2)*x₁ - 12*a*b = 0) ∧
  (12*a*b*x₂^2 - (16*a^2 - 9*b^2)*x₂ - 12*a*b = 0) :=
by sorry

end quadratic_equation_solution_l1808_180870


namespace line_equation_l1808_180873

/-- A line passing through the point (2, 3) with opposite intercepts on the coordinate axes -/
structure LineWithOppositeIntercepts where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 3)
  point_condition : 3 = m * 2 + b
  -- The line has opposite intercepts on the axes
  opposite_intercepts : ∃ (k : ℝ), k ≠ 0 ∧ (b = k ∨ b = -k) ∧ (b / m = -k ∨ b / m = k)

/-- The equation of the line is either x - y + 1 = 0 or 3x - 2y = 0 -/
theorem line_equation (l : LineWithOppositeIntercepts) :
  (l.m = 1 ∧ l.b = -1) ∨ (l.m = 3/2 ∧ l.b = 0) :=
sorry

end line_equation_l1808_180873


namespace dog_grouping_theorem_l1808_180853

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Fluffy in the 4-dog group and Nipper in the 6-dog group -/
def dog_grouping_ways : ℕ := 2520

/-- The total number of dogs -/
def total_dogs : ℕ := 12

/-- The size of the first group (including Fluffy) -/
def group1_size : ℕ := 4

/-- The size of the second group (including Nipper) -/
def group2_size : ℕ := 6

/-- The size of the third group -/
def group3_size : ℕ := 2

theorem dog_grouping_theorem :
  dog_grouping_ways =
    Nat.choose (total_dogs - 2) (group1_size - 1) *
    Nat.choose (total_dogs - group1_size - 1) (group2_size - 1) :=
by sorry

end dog_grouping_theorem_l1808_180853


namespace maggie_total_spent_l1808_180840

def plant_books : ℕ := 20
def fish_books : ℕ := 7
def magazines : ℕ := 25
def book_cost : ℕ := 25
def magazine_cost : ℕ := 5

theorem maggie_total_spent : 
  (plant_books + fish_books) * book_cost + magazines * magazine_cost = 800 := by
sorry

end maggie_total_spent_l1808_180840


namespace gcd_324_135_l1808_180892

theorem gcd_324_135 : Nat.gcd 324 135 = 27 := by
  sorry

end gcd_324_135_l1808_180892


namespace arithmetic_sequence_common_difference_l1808_180869

/-- An arithmetic sequence {a_n} with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : a 1 + a 7 = 22)
  (h3 : a 4 + a 10 = 40) :
  d = 3 := by
sorry

end arithmetic_sequence_common_difference_l1808_180869


namespace mean_of_remaining_numbers_l1808_180895

theorem mean_of_remaining_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 90 →
  (a + b + c + d) / 4 = 86.25 := by
sorry

end mean_of_remaining_numbers_l1808_180895


namespace factorial_sum_equation_l1808_180818

theorem factorial_sum_equation : ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ k l : ℕ, n.factorial / 2 = k.factorial + l.factorial) ∧ S.sum id = 10 := by
  sorry

end factorial_sum_equation_l1808_180818


namespace closest_to_140_l1808_180887

def options : List ℝ := [120, 140, 160, 180, 200]

def expression : ℝ := 3.52 * 7.861 * (6.28 - 1.283)

theorem closest_to_140 : 
  ∀ x ∈ options, |expression - 140| ≤ |expression - x| := by
  sorry

end closest_to_140_l1808_180887


namespace scooter_price_proof_l1808_180816

theorem scooter_price_proof (initial_price : ℝ) : 
  (∃ (total_cost selling_price : ℝ),
    total_cost = initial_price + 300 ∧
    selling_price = 1260 ∧
    selling_price = total_cost * 1.05) →
  initial_price = 900 := by
sorry

end scooter_price_proof_l1808_180816


namespace nancy_carrots_l1808_180831

/-- The number of carrots Nancy threw out -/
def carrots_thrown_out : ℕ := 2

/-- The number of carrots Nancy initially picked -/
def initial_carrots : ℕ := 12

/-- The number of carrots Nancy picked the next day -/
def next_day_carrots : ℕ := 21

/-- The total number of carrots Nancy ended up with -/
def total_carrots : ℕ := 31

theorem nancy_carrots :
  initial_carrots - carrots_thrown_out + next_day_carrots = total_carrots :=
by sorry

end nancy_carrots_l1808_180831


namespace sum_of_solutions_abs_eq_l1808_180828

theorem sum_of_solutions_abs_eq (x : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |3 * x₁ - 12| = 6 ∧ |3 * x₂ - 12| = 6 ∧ x₁ + x₂ = 8) ∧ (∀ x : ℝ, |3 * x - 12| = 6 → x = 2 ∨ x = 6) :=
by sorry

end sum_of_solutions_abs_eq_l1808_180828


namespace exam_score_below_mean_l1808_180815

/-- Given an exam with mean score and a known score above the mean,
    calculate the score that is a certain number of standard deviations below the mean. -/
theorem exam_score_below_mean 
  (mean : ℝ) 
  (score_above : ℝ) 
  (sd_above : ℝ) 
  (sd_below : ℝ) 
  (h1 : mean = 88.8)
  (h2 : score_above = 90)
  (h3 : sd_above = 3)
  (h4 : sd_below = 7)
  (h5 : score_above = mean + sd_above * ((score_above - mean) / sd_above)) :
  mean - sd_below * ((score_above - mean) / sd_above) = 86 := by
sorry


end exam_score_below_mean_l1808_180815


namespace sum_y_coordinates_on_y_axis_l1808_180886

-- Define the circle
def circle_center : ℝ × ℝ := (-4, 3)
def circle_radius : ℝ := 5

-- Define a function to check if a point is on the circle
def on_circle (point : ℝ × ℝ) : Prop :=
  (point.1 - circle_center.1)^2 + (point.2 - circle_center.2)^2 = circle_radius^2

-- Define a function to check if a point is on the y-axis
def on_y_axis (point : ℝ × ℝ) : Prop :=
  point.1 = 0

-- Theorem statement
theorem sum_y_coordinates_on_y_axis :
  ∃ (p1 p2 : ℝ × ℝ),
    on_circle p1 ∧ on_circle p2 ∧
    on_y_axis p1 ∧ on_y_axis p2 ∧
    p1 ≠ p2 ∧
    p1.2 + p2.2 = 6 :=
  sorry

end sum_y_coordinates_on_y_axis_l1808_180886


namespace bigger_part_problem_l1808_180882

theorem bigger_part_problem (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 34 := by
  sorry

end bigger_part_problem_l1808_180882


namespace no_triangle_with_special_sides_l1808_180830

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define functions for altitude, angle bisector, and median
def altitude (t : Triangle) : ℝ := sorry
def angleBisector (t : Triangle) : ℝ := sorry
def median (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem no_triangle_with_special_sides :
  ¬ ∃ (t : Triangle),
    (t.a = altitude t ∧ t.b = angleBisector t ∧ t.c = median t) ∨
    (t.a = altitude t ∧ t.b = median t ∧ t.c = angleBisector t) ∨
    (t.a = angleBisector t ∧ t.b = altitude t ∧ t.c = median t) ∨
    (t.a = angleBisector t ∧ t.b = median t ∧ t.c = altitude t) ∨
    (t.a = median t ∧ t.b = altitude t ∧ t.c = angleBisector t) ∨
    (t.a = median t ∧ t.b = angleBisector t ∧ t.c = altitude t) := by
  sorry

end no_triangle_with_special_sides_l1808_180830


namespace plywood_width_l1808_180863

theorem plywood_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 24 →
  length = 4 →
  area = length * width →
  width = 6 := by
sorry

end plywood_width_l1808_180863


namespace system_solution_unique_l1808_180825

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x + y = 2 ∧ 2 * x - y = 8 :=
by
  -- The proof would go here
  sorry

end system_solution_unique_l1808_180825


namespace base_8_representation_of_512_l1808_180889

/-- Converts a natural number to its base-8 representation as a list of digits (least significant first) -/
def to_base_8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 8) :: aux (m / 8)
    aux n

theorem base_8_representation_of_512 :
  to_base_8 512 = [0, 0, 0, 1] := by
sorry

end base_8_representation_of_512_l1808_180889


namespace triathlon_problem_l1808_180897

/-- Triathlon problem -/
theorem triathlon_problem 
  (swim_distance : ℝ) 
  (cycle_distance : ℝ) 
  (run_distance : ℝ)
  (total_time : ℝ)
  (practice_swim_time : ℝ)
  (practice_cycle_time : ℝ)
  (practice_run_time : ℝ)
  (practice_total_distance : ℝ)
  (h_swim_distance : swim_distance = 1)
  (h_cycle_distance : cycle_distance = 25)
  (h_run_distance : run_distance = 4)
  (h_total_time : total_time = 5/4)
  (h_practice_swim_time : practice_swim_time = 1/16)
  (h_practice_cycle_time : practice_cycle_time = 1/49)
  (h_practice_run_time : practice_run_time = 1/49)
  (h_practice_total_distance : practice_total_distance = 5/4)
  (h_positive_speeds : ∀ v : ℝ, v > 0 → v + 1/v ≥ 2) :
  ∃ (cycle_time cycle_speed : ℝ),
    cycle_time = 5/7 ∧ 
    cycle_speed = 35 ∧
    cycle_distance / cycle_speed = cycle_time ∧
    swim_distance / (swim_distance / practice_swim_time) + 
    cycle_distance / cycle_speed + 
    run_distance / (run_distance / practice_run_time) = total_time ∧
    practice_swim_time * (swim_distance / practice_swim_time) + 
    practice_cycle_time * cycle_speed + 
    practice_run_time * (run_distance / practice_run_time) = practice_total_distance :=
by sorry


end triathlon_problem_l1808_180897


namespace distance_between_trees_l1808_180868

-- Define the yard length and number of trees
def yard_length : ℝ := 520
def num_trees : ℕ := 40

-- Theorem statement
theorem distance_between_trees :
  let num_spaces : ℕ := num_trees - 1
  let distance : ℝ := yard_length / num_spaces
  distance = 520 / 39 := by
  sorry

end distance_between_trees_l1808_180868


namespace chantel_bracelets_l1808_180804

/-- The number of bracelets Chantel gave away at soccer practice -/
def bracelets_given_at_soccer : ℕ := sorry

/-- The number of days Chantel makes 2 bracelets per day -/
def days_making_two : ℕ := 5

/-- The number of bracelets Chantel makes per day in the first period -/
def bracelets_per_day_first : ℕ := 2

/-- The number of bracelets Chantel gives away at school -/
def bracelets_given_at_school : ℕ := 3

/-- The number of days Chantel makes 3 bracelets per day -/
def days_making_three : ℕ := 4

/-- The number of bracelets Chantel makes per day in the second period -/
def bracelets_per_day_second : ℕ := 3

/-- The number of bracelets Chantel has at the end -/
def bracelets_at_end : ℕ := 13

theorem chantel_bracelets : 
  bracelets_given_at_soccer = 
    days_making_two * bracelets_per_day_first + 
    days_making_three * bracelets_per_day_second - 
    bracelets_given_at_school - 
    bracelets_at_end := by sorry

end chantel_bracelets_l1808_180804


namespace meat_spending_fraction_l1808_180865

/-- Represents John's spending at the supermarket -/
structure SupermarketSpending where
  total : ℝ
  fruitVeg : ℝ
  bakery : ℝ
  candy : ℝ
  meat : ℝ

/-- Theorem stating the fraction spent on meat products -/
theorem meat_spending_fraction (s : SupermarketSpending) 
  (h1 : s.total = 30)
  (h2 : s.fruitVeg = s.total / 5)
  (h3 : s.bakery = s.total / 10)
  (h4 : s.candy = 11)
  (h5 : s.total = s.fruitVeg + s.bakery + s.meat + s.candy) :
  s.meat / s.total = 8 / 15 := by
  sorry

end meat_spending_fraction_l1808_180865


namespace unique_solution_l1808_180846

theorem unique_solution : ∃! x : ℝ, ((x / 8) + 8 - 30) * 6 = 12 := by
  sorry

end unique_solution_l1808_180846


namespace pond_volume_calculation_l1808_180845

/-- The volume of a rectangular pond -/
def pond_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of a rectangular pond with dimensions 28 m × 10 m × 5 m is 1400 cubic meters -/
theorem pond_volume_calculation : pond_volume 28 10 5 = 1400 := by
  sorry

end pond_volume_calculation_l1808_180845


namespace total_students_is_150_l1808_180858

/-- In a school, when there are 60 boys, girls become 60% of the total number of students. -/
def school_condition (total_students : ℕ) : Prop :=
  (60 : ℝ) / total_students + 0.6 = 1

/-- The theorem states that under the given condition, the total number of students is 150. -/
theorem total_students_is_150 : ∃ (total_students : ℕ), 
  school_condition total_students ∧ total_students = 150 := by
  sorry

end total_students_is_150_l1808_180858


namespace student_line_arrangements_l1808_180872

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of students who refuse to stand next to each other
def num_refusing_adjacent : ℕ := 2

-- Define the number of students who must stand at an end
def num_at_end : ℕ := 1

-- Function to calculate the number of arrangements
def num_arrangements (n : ℕ) (r : ℕ) (e : ℕ) : ℕ :=
  2 * (n.factorial - (n - r + 1).factorial * r.factorial)

-- Theorem statement
theorem student_line_arrangements :
  num_arrangements num_students num_refusing_adjacent num_at_end = 144 :=
by sorry

end student_line_arrangements_l1808_180872


namespace solution_xyz_l1808_180832

theorem solution_xyz (x y z : ℝ) 
  (eq1 : 2*x + y = 4) 
  (eq2 : x + 2*y = 5) 
  (eq3 : 3*x - 1.5*y + z = 7) : 
  (x + y + z) / 3 = 10/3 := by
  sorry

end solution_xyz_l1808_180832


namespace perfect_square_power_of_two_l1808_180867

theorem perfect_square_power_of_two (n : ℕ+) : 
  (∃ m : ℕ, 2^8 + 2^11 + 2^(n : ℕ) = m^2) ↔ n = 12 := by
  sorry

end perfect_square_power_of_two_l1808_180867


namespace smallest_a_l1808_180884

/-- A parabola with vertex at (1/3, -25/27) described by y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a > 0 → b = -2*a/3
  vertex_y : a > 0 → c = a/9 - 25/27
  integer_sum : ∃ k : ℤ, 3*a + 2*b + 4*c = k

/-- The smallest possible value of a for the given parabola conditions -/
theorem smallest_a (p : Parabola) : 
  (∀ q : Parabola, q.a > 0 → p.a ≤ q.a) → p.a = 300/19 := by
  sorry

end smallest_a_l1808_180884


namespace subset_implies_a_values_l1808_180819

/-- The set M of solutions to the quadratic equation 2x^2 - 3x - 2 = 0 -/
def M : Set ℝ := {x | 2 * x^2 - 3 * x - 2 = 0}

/-- The set N of solutions to the linear equation ax = 1 -/
def N (a : ℝ) : Set ℝ := {x | a * x = 1}

/-- Theorem stating that if N is a subset of M, then a must be 0, -2, or 1/2 -/
theorem subset_implies_a_values (a : ℝ) (h : N a ⊆ M) : 
  a = 0 ∨ a = -2 ∨ a = 1/2 := by sorry

end subset_implies_a_values_l1808_180819


namespace correct_selection_methods_l1808_180806

def total_people : ℕ := 16
def people_per_class : ℕ := 4
def num_classes : ℕ := 4
def people_to_select : ℕ := 3

def selection_methods : ℕ := sorry

theorem correct_selection_methods :
  selection_methods = 472 := by sorry

end correct_selection_methods_l1808_180806


namespace m_range_l1808_180813

theorem m_range : ∃ m : ℝ, m = 3 * Real.sqrt 2 - 1 ∧ 3 < m ∧ m < 4 := by
  sorry

end m_range_l1808_180813


namespace codecracker_combinations_l1808_180817

/-- The number of available colors for the CodeCracker game -/
def num_colors : ℕ := 7

/-- The number of slots in the master code -/
def code_length : ℕ := 5

/-- The number of different master codes that can be formed in the CodeCracker game -/
def num_codes : ℕ := num_colors ^ code_length

theorem codecracker_combinations : num_codes = 16807 := by
  sorry

end codecracker_combinations_l1808_180817


namespace subset_implies_m_squared_l1808_180843

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B : Set ℝ := {3, 4}

-- State the theorem
theorem subset_implies_m_squared (m : ℝ) : B ⊆ A m → (m = 2 ∨ m = -2) := by
  sorry

end subset_implies_m_squared_l1808_180843


namespace participation_plans_specific_l1808_180814

/-- The number of ways to select three students from four, with one student always selected,
    for three different subjects. -/
def participation_plans (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  (n - 1).choose (k - 1) * m.factorial

theorem participation_plans_specific : participation_plans 4 3 3 = 18 := by
  sorry

#eval participation_plans 4 3 3

end participation_plans_specific_l1808_180814


namespace head_start_calculation_l1808_180821

/-- Proves that the head start given by A to B is 72 meters in a 96-meter race,
    given that A runs 4 times as fast as B and they finish at the same time. -/
theorem head_start_calculation (v_B : ℝ) (d : ℝ) 
  (h1 : v_B > 0)  -- B's speed is positive
  (h2 : 96 > d)   -- The head start is less than the total race distance
  (h3 : 96 / (4 * v_B) = (96 - d) / v_B)  -- A and B finish at the same time
  : d = 72 := by
  sorry

end head_start_calculation_l1808_180821


namespace buratino_bet_exists_pierrot_bet_impossible_papa_carlo_minimum_bet_karabas_barabas_impossible_l1808_180844

/-- Represents a bet on a horse race --/
structure HorseBet where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the odds for each horse --/
def odds : Fin 3 → ℚ
  | 0 => 4
  | 1 => 3
  | 2 => 1

/-- Calculates the total bet amount --/
def totalBet (bet : HorseBet) : ℕ :=
  bet.first + bet.second + bet.third

/-- Calculates the return for a given horse winning --/
def returnForHorse (bet : HorseBet) (horse : Fin 3) : ℚ :=
  match horse with
  | 0 => (bet.first : ℚ) * (odds 0 + 1)
  | 1 => (bet.second : ℚ) * (odds 1 + 1)
  | 2 => (bet.third : ℚ) * (odds 2 + 1)

/-- Checks if a bet guarantees a minimum return --/
def guaranteesReturn (bet : HorseBet) (minReturn : ℚ) : Prop :=
  ∀ horse : Fin 3, returnForHorse bet horse ≥ minReturn

theorem buratino_bet_exists :
  ∃ bet : HorseBet, totalBet bet = 50 ∧ guaranteesReturn bet 52 :=
sorry

theorem pierrot_bet_impossible :
  ¬∃ bet : HorseBet, totalBet bet = 25 ∧ guaranteesReturn bet 26 :=
sorry

theorem papa_carlo_minimum_bet :
  (∃ bet : HorseBet, guaranteesReturn bet ((totalBet bet : ℚ) + 5)) ∧
  (∀ s : ℕ, s < 95 → ¬∃ bet : HorseBet, totalBet bet = s ∧ guaranteesReturn bet ((s : ℚ) + 5)) :=
sorry

theorem karabas_barabas_impossible :
  ¬∃ bet : HorseBet, guaranteesReturn bet ((totalBet bet : ℚ) * (106 / 100)) :=
sorry

end buratino_bet_exists_pierrot_bet_impossible_papa_carlo_minimum_bet_karabas_barabas_impossible_l1808_180844


namespace total_weight_is_7000_l1808_180800

/-- The weight of the truck in pounds -/
def truck_weight : ℝ := 4800

/-- The weight of the trailer in pounds -/
def trailer_weight : ℝ := 0.5 * truck_weight - 200

/-- The total weight of the truck and trailer in pounds -/
def total_weight : ℝ := truck_weight + trailer_weight

/-- Theorem stating that the total weight of the truck and trailer is 7000 pounds -/
theorem total_weight_is_7000 : total_weight = 7000 := by
  sorry

end total_weight_is_7000_l1808_180800


namespace rectangle_dimensions_and_area_l1808_180801

theorem rectangle_dimensions_and_area (x : ℝ) : 
  (x - 3 > 0) →
  (3 * x + 4 > 0) →
  ((x - 3) * (3 * x + 4) = 12 * x - 7) →
  (x = (17 + Real.sqrt 349) / 6) :=
by sorry

end rectangle_dimensions_and_area_l1808_180801


namespace simplify_and_evaluate_l1808_180847

theorem simplify_and_evaluate (x : ℝ) (h : x = 2) :
  (1 + 1/x) / ((x^2 - 1) / x) = 1 := by
  sorry

end simplify_and_evaluate_l1808_180847


namespace possible_days_l1808_180833

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def Anya_lies (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Thursday

def Vanya_lies (d : Day) : Prop :=
  d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

def Anya_says_Friday (d : Day) : Prop :=
  (Anya_lies d ∧ d ≠ Day.Friday) ∨ (¬Anya_lies d ∧ d = Day.Friday)

def Vanya_says_Tuesday (d : Day) : Prop :=
  (Vanya_lies d ∧ d ≠ Day.Tuesday) ∨ (¬Vanya_lies d ∧ d = Day.Tuesday)

theorem possible_days :
  ∀ d : Day, (Anya_says_Friday d ∧ Vanya_says_Tuesday d) ↔ 
    (d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Friday) :=
by sorry

end possible_days_l1808_180833


namespace quotient_digits_of_203_div_single_digit_l1808_180879

theorem quotient_digits_of_203_div_single_digit :
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 →
  ∃ q : ℕ, 203 / d = q ∧ (100 ≤ q ∧ q ≤ 999 ∨ 10 ≤ q ∧ q ≤ 99) :=
by sorry

end quotient_digits_of_203_div_single_digit_l1808_180879


namespace winning_scenarios_is_60_l1808_180885

/-- The number of different winning scenarios for a lottery ticket distribution -/
def winning_scenarios : ℕ :=
  let total_tickets : ℕ := 8
  let num_people : ℕ := 4
  let tickets_per_person : ℕ := 2
  let first_prize : ℕ := 1
  let second_prize : ℕ := 1
  let third_prize : ℕ := 1
  let non_winning_tickets : ℕ := 5

  -- The actual computation of winning scenarios
  60

/-- Theorem stating that the number of winning scenarios is 60 -/
theorem winning_scenarios_is_60 : winning_scenarios = 60 := by
  sorry

end winning_scenarios_is_60_l1808_180885


namespace necessary_but_not_sufficient_l1808_180838

theorem necessary_but_not_sufficient (a b : ℝ) :
  (((1 / b) < (1 / a) ∧ (1 / a) < 0) → a < b) ∧
  (∃ a b, a < b ∧ ¬((1 / b) < (1 / a) ∧ (1 / a) < 0)) :=
by sorry

end necessary_but_not_sufficient_l1808_180838


namespace lake_crossing_cost_l1808_180834

/-- The cost of crossing a lake back and forth -/
theorem lake_crossing_cost (crossing_time : ℕ) (assistant_cost : ℕ) : 
  crossing_time = 4 → assistant_cost = 10 → crossing_time * 2 * assistant_cost = 80 := by
  sorry

#check lake_crossing_cost

end lake_crossing_cost_l1808_180834


namespace relay_team_permutations_l1808_180811

theorem relay_team_permutations (n : ℕ) (h : n = 4) : Nat.factorial (n - 1) = 6 := by
  sorry

end relay_team_permutations_l1808_180811


namespace mans_rate_l1808_180829

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 22)
  (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end mans_rate_l1808_180829


namespace bus_trip_difference_l1808_180866

def bus_trip (initial : ℕ) 
             (stop1_off stop1_on : ℕ) 
             (stop2_off stop2_on : ℕ) 
             (stop3_off stop3_on : ℕ) 
             (stop4_off stop4_on : ℕ) : ℕ :=
  let after_stop1 := initial - stop1_off + stop1_on
  let after_stop2 := after_stop1 - stop2_off + stop2_on
  let after_stop3 := after_stop2 - stop3_off + stop3_on
  let final := after_stop3 - stop4_off + stop4_on
  initial - final

theorem bus_trip_difference :
  bus_trip 41 12 5 7 10 14 3 9 6 = 18 := by
  sorry

end bus_trip_difference_l1808_180866


namespace inequality_solution_l1808_180810

theorem inequality_solution (x : Real) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
by sorry

end inequality_solution_l1808_180810


namespace parabola_distance_theorem_l1808_180808

/-- Parabola type -/
structure Parabola where
  /-- The equation of the parabola y^2 = 8x -/
  equation : ℝ → ℝ → Prop
  /-- The focus of the parabola -/
  focus : ℝ × ℝ
  /-- The directrix of the parabola -/
  directrix : ℝ → ℝ → Prop

/-- Point on the directrix -/
def PointOnDirectrix (p : Parabola) : Type := { point : ℝ × ℝ // p.directrix point.1 point.2 }

/-- Point on the parabola -/
def PointOnParabola (p : Parabola) : Type := { point : ℝ × ℝ // p.equation point.1 point.2 }

/-- Theorem: For a parabola y^2 = 8x, if FP = 4FQ, then |QF| = 3 -/
theorem parabola_distance_theorem (p : Parabola) 
  (hpeq : p.equation = fun x y ↦ y^2 = 8*x)
  (P : PointOnDirectrix p) 
  (Q : PointOnParabola p) 
  (hline : ∃ (t : ℝ), Q.val = p.focus + t • (P.val - p.focus))
  (hfp : ‖P.val - p.focus‖ = 4 * ‖Q.val - p.focus‖) :
  ‖Q.val - p.focus‖ = 3 := by sorry

end parabola_distance_theorem_l1808_180808


namespace trajectory_of_Q_l1808_180891

-- Define the points
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 6)
def C : ℝ × ℝ := (0, -2)
def D : ℝ × ℝ := (0, 2)

-- Define the moving point P
def P : ℝ × ℝ → Prop :=
  λ p => ‖p - A‖ / ‖p - B‖ = 1 / 2

-- Define the perpendicular bisector of PC
def perpBisector (p : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ q => ‖q - p‖ = ‖q - C‖

-- Define point Q
def Q (p : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ q => perpBisector p q ∧ ∃ t : ℝ, q = p + t • (D - p)

-- State the theorem
theorem trajectory_of_Q :
  ∀ p : ℝ × ℝ, P p →
    ∀ q : ℝ × ℝ, Q p q →
      q.2^2 - q.1^2 / 3 = 1 :=
sorry

end trajectory_of_Q_l1808_180891


namespace sum_of_two_numbers_l1808_180878

theorem sum_of_two_numbers (x y : ℤ) : x + y = 32 ∧ y = -36 → x = 68 := by
  sorry

end sum_of_two_numbers_l1808_180878


namespace function_inequalities_l1808_180896

/-- Given a function f(x) = x^2 - (a + 2)x + 4, where a is a real number -/
def f (a x : ℝ) : ℝ := x^2 - (a + 2)*x + 4

theorem function_inequalities (a : ℝ) :
  (∀ x, a < 2 → (f a x ≤ -2*a + 4 ↔ a ≤ x ∧ x ≤ 2)) ∧
  (∀ x, a = 2 → (f a x ≤ -2*a + 4 ↔ x = 2)) ∧
  (∀ x, a > 2 → (f a x ≤ -2*a + 4 ↔ 2 ≤ x ∧ x ≤ a)) ∧
  (∀ x, x ∈ Set.Icc 1 4 → f a x + a + 1 ≥ 0 ↔ a ∈ Set.Iic 4) :=
by sorry


end function_inequalities_l1808_180896


namespace max_area_convex_quadrilateral_l1808_180877

/-- A convex quadrilateral with diagonals d₁ and d₂ has an area S. -/
structure ConvexQuadrilateral where
  d₁ : ℝ
  d₂ : ℝ
  S : ℝ
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  S_pos : S > 0
  area_formula : ∃ α : ℝ, 0 ≤ α ∧ α ≤ π ∧ S = (1/2) * d₁ * d₂ * Real.sin α

/-- The maximum area of a convex quadrilateral is half the product of its diagonals. -/
theorem max_area_convex_quadrilateral (q : ConvexQuadrilateral) : 
  q.S ≤ (1/2) * q.d₁ * q.d₂ ∧ ∃ q' : ConvexQuadrilateral, q'.S = (1/2) * q'.d₁ * q'.d₂ := by
  sorry


end max_area_convex_quadrilateral_l1808_180877


namespace factorization_equality_l1808_180898

theorem factorization_equality (x : ℝ) :
  (x^2 + 5*x + 2) * (x^2 + 5*x + 3) - 12 = (x + 2) * (x + 3) * (x^2 + 5*x - 1) := by
  sorry

end factorization_equality_l1808_180898


namespace cube_sum_zero_l1808_180883

theorem cube_sum_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum_zero : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end cube_sum_zero_l1808_180883


namespace vector_collinearity_l1808_180822

theorem vector_collinearity (m n : ℝ) (h : n ≠ 0) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, 3]
  (∃ (k : ℝ), k ≠ 0 ∧ (fun i => m * a i - n * b i) = (fun i => k * (a i + 2 * b i))) →
  m / n = -1/2 := by
  sorry

end vector_collinearity_l1808_180822


namespace acquaintance_pigeonhole_l1808_180874

theorem acquaintance_pigeonhole (n : ℕ) (h : n ≥ 2) :
  ∃ (i j : Fin n), i ≠ j ∧ 
  ∃ (f : Fin n → Fin n), (∀ k, f k < n) ∧ f i = f j :=
by
  sorry

end acquaintance_pigeonhole_l1808_180874
