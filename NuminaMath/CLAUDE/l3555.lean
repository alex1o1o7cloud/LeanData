import Mathlib

namespace marcos_dad_strawberries_l3555_355506

theorem marcos_dad_strawberries (marco_weight : ℕ) (total_weight : ℕ) 
  (h1 : marco_weight = 15)
  (h2 : total_weight = 37) :
  total_weight - marco_weight = 22 := by
  sorry

end marcos_dad_strawberries_l3555_355506


namespace last_two_digits_of_sum_factorials_l3555_355570

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_factorials :
  sum_factorials 9 % 100 = 13 := by sorry

end last_two_digits_of_sum_factorials_l3555_355570


namespace simplify_and_rationalize_l3555_355546

theorem simplify_and_rationalize (x : ℝ) :
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l3555_355546


namespace lunks_needed_for_twenty_apples_l3555_355505

/-- Represents the number of lunks needed to purchase a given number of apples,
    given the exchange rates between lunks, kunks, and apples. -/
def lunks_for_apples (lunks_per_kunk : ℚ) (kunks_per_apple : ℚ) (num_apples : ℕ) : ℚ :=
  num_apples * kunks_per_apple * lunks_per_kunk

/-- Theorem stating that 21 lunks are needed to purchase 20 apples,
    given the specified exchange rates. -/
theorem lunks_needed_for_twenty_apples :
  lunks_for_apples (7/4) (3/5) 20 = 21 := by
  sorry

end lunks_needed_for_twenty_apples_l3555_355505


namespace village_assistants_selection_l3555_355576

-- Define the total number of college graduates
def total_graduates : ℕ := 10

-- Define the number of people to be selected
def selection_size : ℕ := 3

-- Define a function to calculate the number of ways to select k items from n items
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem village_assistants_selection :
  choose (total_graduates - 1) selection_size -
  choose (total_graduates - 3) selection_size = 49 := by
  sorry

end village_assistants_selection_l3555_355576


namespace ellipse_and_circle_problem_l3555_355508

theorem ellipse_and_circle_problem 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 2^2 = a^2 - b^2) -- condition for right focus at (2,0)
  : 
  (∀ x y : ℝ, x^2/6 + y^2/2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ 
  (∃ m : ℝ, ∃ c : Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ c ↔ (p.1^2 + (p.2 - 1/3)^2 = (1/3)^2)) ∧
    (∃ p1 p2 p3 p4 : ℝ × ℝ, 
      p1 ∈ c ∧ p2 ∈ c ∧ p3 ∈ c ∧ p4 ∈ c ∧
      p1.2 = p1.1^2 + m ∧ p2.2 = p2.1^2 + m ∧ p3.2 = p3.1^2 + m ∧ p4.2 = p4.1^2 + m ∧
      p1.1^2/6 + p1.2^2/2 = 1 ∧ p2.1^2/6 + p2.2^2/2 = 1 ∧ p3.1^2/6 + p3.2^2/2 = 1 ∧ p4.1^2/6 + p4.2^2/2 = 1)) := by
  sorry

end ellipse_and_circle_problem_l3555_355508


namespace triangle_angle_measure_l3555_355503

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.cos A = b / (2 * Real.cos B) ∧
  a / Real.cos A = c / (3 * Real.cos C) →
  A = π / 4 := by
sorry

end triangle_angle_measure_l3555_355503


namespace height_growth_l3555_355574

theorem height_growth (current_height : ℝ) (growth_rate : ℝ) (original_height : ℝ) : 
  current_height = 147 ∧ growth_rate = 0.05 → original_height = 140 :=
by
  sorry

end height_growth_l3555_355574


namespace freds_allowance_l3555_355587

def weekly_allowance (A x y : ℝ) : Prop :=
  -- Fred spent half of his allowance on movie tickets
  let movie_cost := A / 2
  -- Lunch cost y dollars less than the cost of the tickets
  let lunch_cost := x - y
  -- He earned 6 dollars from washing the car and 5 dollars from mowing the lawn
  let earned := 6 + 5
  -- At the end of the day, he had 20 dollars
  movie_cost + lunch_cost + earned + (A - movie_cost - lunch_cost) = 20

theorem freds_allowance :
  ∃ A x y : ℝ, weekly_allowance A x y ∧ A = 9 := by sorry

end freds_allowance_l3555_355587


namespace cooler_contents_l3555_355593

/-- The number of cherry sodas in the cooler -/
def cherry_sodas : ℕ := 8

/-- The number of orange pops in the cooler -/
def orange_pops : ℕ := 2 * cherry_sodas

/-- The total number of cans in the cooler -/
def total_cans : ℕ := 24

theorem cooler_contents : 
  cherry_sodas + orange_pops = total_cans ∧ cherry_sodas = 8 := by
  sorry

end cooler_contents_l3555_355593


namespace recreation_area_tents_l3555_355531

/-- Represents the number of tents in different parts of the campsite -/
structure CampsiteTents where
  north : ℕ
  east : ℕ
  center : ℕ
  south : ℕ

/-- Calculates the total number of tents in the campsite -/
def total_tents (c : CampsiteTents) : ℕ :=
  c.north + c.east + c.center + c.south

/-- Theorem stating the total number of tents in the recreation area -/
theorem recreation_area_tents :
  ∃ (c : CampsiteTents),
    c.north = 100 ∧
    c.east = 2 * c.north ∧
    c.center = 4 * c.north ∧
    c.south = 200 ∧
    total_tents c = 900 := by
  sorry

end recreation_area_tents_l3555_355531


namespace enclosure_probability_l3555_355589

def is_valid_configuration (c₁ c₂ c₃ d₁ d₂ d₃ : ℕ) : Prop :=
  d₁ ≥ 2 * c₁ ∧ d₁ > d₂ ∧ d₂ > d₃ ∧ c₁ > c₂ ∧ c₂ > c₃ ∧
  d₁ > c₁ ∧ d₂ > c₂ ∧ d₃ > c₃

def probability_of_valid_configuration : ℚ :=
  1 / 2

theorem enclosure_probability :
  ∀ (S : Finset ℕ) (c₁ c₂ c₃ d₁ d₂ d₃ : ℕ),
    S = Finset.range 100 →
    c₁ ∈ S ∧ c₂ ∈ S ∧ c₃ ∈ S →
    c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ →
    d₁ ∈ S.erase c₁ \ {c₂, c₃} ∧ d₂ ∈ S.erase c₁ \ {c₂, c₃, d₁} ∧ d₃ ∈ S.erase c₁ \ {c₂, c₃, d₁, d₂} →
    probability_of_valid_configuration = 1 / 2 :=
sorry

end enclosure_probability_l3555_355589


namespace inequality_proof_l3555_355590

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a > b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
sorry

end inequality_proof_l3555_355590


namespace manufacturing_sector_degrees_l3555_355569

/-- Represents the number of degrees in a full circle -/
def full_circle_degrees : ℝ := 360

/-- Represents the percentage of the circle occupied by the manufacturing department -/
def manufacturing_percentage : ℝ := 45

/-- Theorem: The manufacturing department sector in the circle graph occupies 162 degrees -/
theorem manufacturing_sector_degrees : 
  (manufacturing_percentage / 100) * full_circle_degrees = 162 := by
  sorry

end manufacturing_sector_degrees_l3555_355569


namespace rhombus_diagonal_l3555_355501

theorem rhombus_diagonal (area : ℝ) (ratio_long : ℝ) (ratio_short : ℝ) :
  area = 135 →
  ratio_long = 5 →
  ratio_short = 3 →
  (ratio_long * ratio_short * (longer_diagonal ^ 2)) / (2 * (ratio_long ^ 2)) = area →
  longer_diagonal = 15 := by
  sorry

end rhombus_diagonal_l3555_355501


namespace range_of_a_l3555_355524

open Set Real

theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
  let B : Set ℝ := Ioi 5
  (A ∩ B = ∅) → a ∈ Iic 2 ∪ Ici 3 := by
sorry


end range_of_a_l3555_355524


namespace information_spread_time_l3555_355591

theorem information_spread_time (population : ℕ) (h : population = 1000000) :
  ∃ n : ℕ, n ≥ 19 ∧ 2^(n+1) - 1 ≥ population :=
by sorry

end information_spread_time_l3555_355591


namespace only_non_algorithm_l3555_355568

/-- A process is a description of a task or method. -/
structure Process where
  description : String

/-- An algorithm is a process that has a sequence of defined steps. -/
structure Algorithm extends Process where
  has_defined_steps : Bool

/-- The property of having defined steps for a process. -/
def has_defined_steps (p : Process) : Prop :=
  ∃ (a : Algorithm), a.description = p.description

/-- The list of processes to be evaluated. -/
def processes : List Process :=
  [{ description := "The process of solving the equation 2x-6=0 involves moving terms and making the coefficient 1" },
   { description := "To get from Jinan to Vancouver, one must first take a train to Beijing, then transfer to a plane" },
   { description := "Solving the equation 2x^2+x-1=0" },
   { description := "Using the formula S=πr^2 to calculate the area of a circle with radius 3 involves computing π×3^2" }]

/-- The theorem stating that "Solving the equation 2x^2+x-1=0" is the only process without defined steps. -/
theorem only_non_algorithm :
  ∃! (p : Process), p ∈ processes ∧ ¬(has_defined_steps p) ∧
    p.description = "Solving the equation 2x^2+x-1=0" :=
  sorry

end only_non_algorithm_l3555_355568


namespace local_min_implies_a_eq_2_l3555_355571

/-- The function f(x) = x(x-a)^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

/-- f has a local minimum at x = 2 -/
def has_local_min_at_2 (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2

theorem local_min_implies_a_eq_2 :
  ∀ a : ℝ, has_local_min_at_2 a → a = 2 := by sorry

end local_min_implies_a_eq_2_l3555_355571


namespace thirty_six_hundredths_decimal_l3555_355529

theorem thirty_six_hundredths_decimal : (36 : ℚ) / 100 = 0.36 := by sorry

end thirty_six_hundredths_decimal_l3555_355529


namespace appropriate_sampling_methods_l3555_355534

/-- Represents a city with its number of sales outlets -/
structure City where
  name : String
  outlets : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SystematicSampling
  | SimpleRandomSampling

/-- Represents an investigation with its requirements -/
structure Investigation where
  id : ℕ
  totalOutlets : ℕ
  sampleSize : ℕ
  cities : List City

/-- Determines the most appropriate sampling method for an investigation -/
def mostAppropriateMethod (inv : Investigation) : SamplingMethod := sorry

/-- The main theorem stating the appropriate sampling methods for the given investigations -/
theorem appropriate_sampling_methods 
  (cityA : City)
  (cityB : City)
  (cityC : City)
  (cityD : City)
  (inv1 : Investigation)
  (inv2 : Investigation)
  (h1 : cityA.outlets = 150)
  (h2 : cityB.outlets = 120)
  (h3 : cityC.outlets = 190)
  (h4 : cityD.outlets = 140)
  (h5 : inv1.totalOutlets = 600)
  (h6 : inv1.sampleSize = 100)
  (h7 : inv1.cities = [cityA, cityB, cityC, cityD])
  (h8 : inv2.totalOutlets = 20)
  (h9 : inv2.sampleSize = 8)
  (h10 : inv2.cities = [cityC]) :
  (mostAppropriateMethod inv1 = SamplingMethod.StratifiedSampling) ∧ 
  (mostAppropriateMethod inv2 = SamplingMethod.SimpleRandomSampling) := by
  sorry

end appropriate_sampling_methods_l3555_355534


namespace kendra_cookie_theorem_l3555_355548

/-- Proves that each family member eats 38 chocolate chips given the conditions of Kendra's cookie baking scenario. -/
theorem kendra_cookie_theorem (
  family_members : ℕ)
  (choc_chip_batches : ℕ)
  (double_choc_chip_batches : ℕ)
  (cookies_per_choc_chip_batch : ℕ)
  (chips_per_choc_chip_cookie : ℕ)
  (cookies_per_double_choc_chip_batch : ℕ)
  (chips_per_double_choc_chip_cookie : ℕ)
  (h1 : family_members = 4)
  (h2 : choc_chip_batches = 3)
  (h3 : double_choc_chip_batches = 2)
  (h4 : cookies_per_choc_chip_batch = 12)
  (h5 : chips_per_choc_chip_cookie = 2)
  (h6 : cookies_per_double_choc_chip_batch = 10)
  (h7 : chips_per_double_choc_chip_cookie = 4)
  : (choc_chip_batches * cookies_per_choc_chip_batch * chips_per_choc_chip_cookie +
     double_choc_chip_batches * cookies_per_double_choc_chip_batch * chips_per_double_choc_chip_cookie) / family_members = 38 := by
  sorry

end kendra_cookie_theorem_l3555_355548


namespace midpoint_line_slope_zero_l3555_355512

/-- The slope of the line containing the midpoints of the segments [(1, 1), (3, 4)] and [(4, 1), (7, 4)] is 0. -/
theorem midpoint_line_slope_zero : 
  let midpoint1 := ((1 + 3) / 2, (1 + 4) / 2)
  let midpoint2 := ((4 + 7) / 2, (1 + 4) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = 0 := by
sorry

end midpoint_line_slope_zero_l3555_355512


namespace ackermann_3_2_l3555_355573

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 29 := by
  sorry

end ackermann_3_2_l3555_355573


namespace triangle_pentagon_side_ratio_l3555_355522

theorem triangle_pentagon_side_ratio :
  ∀ (t p : ℝ),
  t > 0 ∧ p > 0 →
  3 * t = 30 →
  5 * p = 30 →
  t / p = 5 / 3 := by
sorry

end triangle_pentagon_side_ratio_l3555_355522


namespace square_times_abs_fraction_equals_three_l3555_355532

theorem square_times_abs_fraction_equals_three :
  (-3)^2 * |-(1/3)| = 3 := by
  sorry

end square_times_abs_fraction_equals_three_l3555_355532


namespace dress_designs_count_l3555_355577

/-- The number of fabric colors available -/
def num_colors : ℕ := 3

/-- The number of different patterns available -/
def num_patterns : ℕ := 4

/-- The number of sleeve length options available -/
def num_sleeve_lengths : ℕ := 2

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns * num_sleeve_lengths

theorem dress_designs_count : total_designs = 24 := by
  sorry

end dress_designs_count_l3555_355577


namespace train_b_completion_time_l3555_355550

/-- Proves that Train B takes 2 hours to complete the route given the conditions -/
theorem train_b_completion_time 
  (route_length : ℝ) 
  (train_a_speed : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : route_length = 75) 
  (h2 : train_a_speed = 25) 
  (h3 : meeting_distance = 30) : 
  (route_length / ((route_length - meeting_distance) / (meeting_distance / train_a_speed))) = 2 := by
  sorry

end train_b_completion_time_l3555_355550


namespace quadratic_inequality_condition_l3555_355566

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2^(a+2) * x - 2^(a+3) + 12 > 0) ↔ a < 0 := by
  sorry

end quadratic_inequality_condition_l3555_355566


namespace probability_walk_300_or_less_l3555_355552

/-- Represents an airport with gates in a straight line. -/
structure Airport where
  num_gates : ℕ
  gate_distance : ℝ

/-- Calculates the number of gate pairs within a given distance. -/
def count_gate_pairs_within_distance (a : Airport) (max_distance : ℝ) : ℕ :=
  sorry

/-- Calculates the total number of possible gate pair assignments. -/
def total_gate_pairs (a : Airport) : ℕ :=
  sorry

/-- The main theorem stating the probability of walking 300 feet or less. -/
theorem probability_walk_300_or_less (a : Airport) :
  a.num_gates = 16 ∧ a.gate_distance = 75 →
  (count_gate_pairs_within_distance a 300 : ℚ) / (total_gate_pairs a : ℚ) = 9 / 20 :=
by sorry

end probability_walk_300_or_less_l3555_355552


namespace expression_equals_185_l3555_355509

theorem expression_equals_185 : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end expression_equals_185_l3555_355509


namespace expression_evaluation_l3555_355578

theorem expression_evaluation :
  (5^1003 + 6^1002)^2 - (5^1003 - 6^1002)^2 = 600 * 30^1002 := by
  sorry

end expression_evaluation_l3555_355578


namespace dubblefud_yellow_chips_l3555_355523

theorem dubblefud_yellow_chips :
  ∀ (yellow blue green red : ℕ),
  -- Yellow chips are worth 2 points
  -- Blue chips are worth 4 points
  -- Green chips are worth 5 points
  -- Red chips are worth 7 points
  -- The product of the point values of the chips is 560000
  2^yellow * 4^blue * 5^green * 7^red = 560000 →
  -- The number of blue chips equals twice the number of green chips
  blue = 2 * green →
  -- The number of red chips is half the number of blue chips
  red = blue / 2 →
  -- The number of yellow chips is 2
  yellow = 2 := by
sorry

end dubblefud_yellow_chips_l3555_355523


namespace road_repair_fractions_l3555_355533

theorem road_repair_fractions (road_length : ℝ) (first_week_fraction second_week_fraction : ℚ) :
  road_length = 1500 →
  first_week_fraction = 5 / 17 →
  second_week_fraction = 4 / 17 →
  (first_week_fraction + second_week_fraction = 9 / 17) ∧
  (1 - (first_week_fraction + second_week_fraction) = 8 / 17) := by
  sorry

end road_repair_fractions_l3555_355533


namespace equation_solution_l3555_355556

theorem equation_solution :
  ∃! x : ℝ, x ≠ -3 ∧ (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) :=
by
  use 9
  sorry

end equation_solution_l3555_355556


namespace hemisphere_surface_area_l3555_355575

theorem hemisphere_surface_area (base_area : ℝ) (Q : ℝ) : 
  base_area = 3 →
  Q = (2 * Real.pi * (Real.sqrt (3 / Real.pi))^2) + base_area →
  Q = 9 := by sorry

end hemisphere_surface_area_l3555_355575


namespace grape_juice_percentage_in_mixture_l3555_355572

/-- Represents a mixture with a certain volume and grape juice percentage -/
structure Mixture where
  volume : ℝ
  percentage : ℝ

/-- Calculates the total volume of grape juice in a mixture -/
def grapeJuiceVolume (m : Mixture) : ℝ := m.volume * m.percentage

/-- The problem statement -/
theorem grape_juice_percentage_in_mixture : 
  let mixtureA : Mixture := { volume := 15, percentage := 0.3 }
  let mixtureB : Mixture := { volume := 40, percentage := 0.2 }
  let mixtureC : Mixture := { volume := 25, percentage := 0.1 }
  let pureGrapeJuice : ℝ := 10

  let totalGrapeJuice := grapeJuiceVolume mixtureA + grapeJuiceVolume mixtureB + 
                         grapeJuiceVolume mixtureC + pureGrapeJuice
  let totalVolume := mixtureA.volume + mixtureB.volume + mixtureC.volume + pureGrapeJuice

  let resultPercentage := totalGrapeJuice / totalVolume

  abs (resultPercentage - 0.2778) < 0.0001 := by
  sorry


end grape_juice_percentage_in_mixture_l3555_355572


namespace lassie_bones_l3555_355562

theorem lassie_bones (initial_bones : ℕ) : 
  (initial_bones / 2 + 10 = 35) → initial_bones = 50 :=
by
  sorry

end lassie_bones_l3555_355562


namespace pennys_bakery_revenue_l3555_355510

/-- Represents the price and quantity of a type of cheesecake -/
structure Cheesecake where
  price_per_slice : ℕ
  pies_sold : ℕ

/-- Calculates the total revenue from a type of cheesecake -/
def revenue (c : Cheesecake) (slices_per_pie : ℕ) : ℕ :=
  c.price_per_slice * c.pies_sold * slices_per_pie

/-- The main theorem about Penny's bakery revenue -/
theorem pennys_bakery_revenue : 
  let slices_per_pie : ℕ := 6
  let blueberry : Cheesecake := { price_per_slice := 7, pies_sold := 7 }
  let strawberry : Cheesecake := { price_per_slice := 8, pies_sold := 5 }
  let chocolate : Cheesecake := { price_per_slice := 9, pies_sold := 3 }
  revenue blueberry slices_per_pie + revenue strawberry slices_per_pie + revenue chocolate slices_per_pie = 696 := by
  sorry


end pennys_bakery_revenue_l3555_355510


namespace smallest_square_division_l3555_355597

/-- A structure representing a division of a square into smaller squares. -/
structure SquareDivision (n : ℕ) :=
  (num_40 : ℕ)  -- number of 40x40 squares
  (num_49 : ℕ)  -- number of 49x49 squares
  (valid : 40 * num_40 + 49 * num_49 = n)
  (non_empty : num_40 > 0 ∧ num_49 > 0)

/-- The theorem stating that 2000 is the smallest n that satisfies the conditions. -/
theorem smallest_square_division :
  (∃ (d : SquareDivision 2000), True) ∧
  (∀ m : ℕ, m < 2000 → ¬∃ (d : SquareDivision m), True) :=
sorry

end smallest_square_division_l3555_355597


namespace peach_basket_problem_l3555_355540

theorem peach_basket_problem (n : ℕ) : 
  n % 4 = 2 →
  n % 6 = 4 →
  (n + 2) % 8 = 0 →
  120 ≤ n →
  n ≤ 150 →
  n = 142 :=
by sorry

end peach_basket_problem_l3555_355540


namespace equality_sum_l3555_355595

theorem equality_sum (M N : ℚ) : 
  (3 / 5 : ℚ) = M / 75 ∧ (3 / 5 : ℚ) = 90 / N → M + N = 195 := by
  sorry

end equality_sum_l3555_355595


namespace binary_addition_and_predecessor_l3555_355535

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  sorry

theorem binary_addition_and_predecessor :
  let M : List Bool := [false, true, true, true, false, true]
  let M_plus_5 : List Bool := [true, true, false, false, true, true]
  let M_plus_5_pred : List Bool := [false, true, false, false, true, true]
  (binary_to_decimal M) + 5 = binary_to_decimal M_plus_5 ∧
  binary_to_decimal M_plus_5 - 1 = binary_to_decimal M_plus_5_pred :=
by
  sorry

#check binary_addition_and_predecessor

end binary_addition_and_predecessor_l3555_355535


namespace inverse_inequality_l3555_355530

theorem inverse_inequality (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : 1 / a < 1 / b := by
  sorry

end inverse_inequality_l3555_355530


namespace training_trip_duration_l3555_355579

/-- The number of supervisors --/
def n : ℕ := 15

/-- The number of supervisors overseeing the pool each day --/
def k : ℕ := 3

/-- The number of ways to choose 2 supervisors from n supervisors --/
def total_pairs : ℕ := n.choose 2

/-- The number of pairs formed each day --/
def pairs_per_day : ℕ := k.choose 2

/-- The number of days required for the training trip --/
def days : ℕ := total_pairs / pairs_per_day

theorem training_trip_duration :
  n = 15 → k = 3 → days = 35 := by sorry

end training_trip_duration_l3555_355579


namespace special_trapezoid_area_l3555_355588

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- The measure of the adjacent angles in degrees -/
  adjacent_angle : ℝ
  /-- The measure of the angle between diagonals facing the base in degrees -/
  diag_angle : ℝ

/-- The area of a special trapezoid -/
noncomputable def area (t : SpecialTrapezoid) : ℝ := sorry

/-- Theorem stating the area of a specific trapezoid is 2 -/
theorem special_trapezoid_area :
  ∀ t : SpecialTrapezoid,
    t.shorter_base = 2 ∧
    t.adjacent_angle = 135 ∧
    t.diag_angle = 150 →
    area t = 2 :=
by sorry

end special_trapezoid_area_l3555_355588


namespace initial_roses_l3555_355511

theorem initial_roses (thrown_away : ℕ) (final_count : ℕ) :
  thrown_away = 33 →
  final_count = 17 →
  ∃ (initial : ℕ) (new_cut : ℕ),
    initial - thrown_away + new_cut = final_count ∧
    new_cut = thrown_away + 2 ∧
    initial = 15 := by
  sorry

end initial_roses_l3555_355511


namespace total_subjects_l3555_355560

theorem total_subjects (average_all : ℝ) (average_five : ℝ) (last_subject : ℝ) 
  (h1 : average_all = 76)
  (h2 : average_five = 74)
  (h3 : last_subject = 86) :
  ∃ n : ℕ, n = 6 ∧ 
    n * average_all = (n - 1) * average_five + last_subject :=
by
  sorry

end total_subjects_l3555_355560


namespace cookie_jar_spending_l3555_355599

theorem cookie_jar_spending (initial_amount : ℝ) (amount_left : ℝ) (doris_spent : ℝ) : 
  initial_amount = 21 →
  amount_left = 12 →
  initial_amount - (doris_spent + doris_spent / 2) = amount_left →
  doris_spent = 6 := by
sorry

end cookie_jar_spending_l3555_355599


namespace gcd_18_30_45_l3555_355557

theorem gcd_18_30_45 : Nat.gcd 18 (Nat.gcd 30 45) = 3 := by
  sorry

end gcd_18_30_45_l3555_355557


namespace distinct_terms_expansion_l3555_355527

/-- The number of distinct terms in the expansion of (a+b+c)(x+y+z+w+t) -/
def distinct_terms (a b c x y z w t : ℝ) : ℕ :=
  3 * 5

theorem distinct_terms_expansion (a b c x y z w t : ℝ) 
  (h_diff : a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ t ∧ 
            y ≠ z ∧ y ≠ w ∧ y ≠ t ∧ z ≠ w ∧ z ≠ t ∧ w ≠ t) : 
  distinct_terms a b c x y z w t = 15 := by
  sorry

end distinct_terms_expansion_l3555_355527


namespace hotdog_distribution_l3555_355526

theorem hotdog_distribution (E : ℚ) 
  (total_hotdogs : E + E + 2*E + 3*E = 14) : E = 2 := by
  sorry

end hotdog_distribution_l3555_355526


namespace power_function_through_point_l3555_355553

/-- A power function that passes through the point (2, √2) has exponent 1/2 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) →  -- f is a power function with exponent a
  f 2 = Real.sqrt 2 →  -- f passes through the point (2, √2)
  a = 1/2 := by
  sorry

end power_function_through_point_l3555_355553


namespace salary_change_percentage_l3555_355564

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 96 / 100 → x = 20 := by
  sorry

end salary_change_percentage_l3555_355564


namespace tory_sold_to_grandmother_l3555_355519

/-- Represents the cookie sales problem for Tory's school fundraiser -/
def cookie_sales (grandmother_packs : ℕ) : Prop :=
  let total_packs : ℕ := 50
  let uncle_packs : ℕ := 7
  let neighbor_packs : ℕ := 5
  let remaining_packs : ℕ := 26
  grandmother_packs + uncle_packs + neighbor_packs + remaining_packs = total_packs

/-- Proves that Tory sold 12 packs of cookies to his grandmother -/
theorem tory_sold_to_grandmother :
  ∃ (x : ℕ), cookie_sales x ∧ x = 12 := by
  sorry

end tory_sold_to_grandmother_l3555_355519


namespace equation_solution_l3555_355518

theorem equation_solution : 
  ∃! x : ℚ, (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 ∧ x = 9 / 28 := by
  sorry

end equation_solution_l3555_355518


namespace adams_house_number_range_l3555_355541

/-- Represents a range of house numbers -/
structure Range where
  lower : Nat
  upper : Nat
  valid : lower ≤ upper

/-- Checks if two ranges overlap -/
def overlaps (r1 r2 : Range) : Prop :=
  (r1.lower ≤ r2.upper ∧ r2.lower ≤ r1.upper) ∨
  (r2.lower ≤ r1.upper ∧ r1.lower ≤ r2.upper)

/-- The given ranges -/
def rangeA : Range := ⟨123, 213, by sorry⟩
def rangeB : Range := ⟨132, 231, by sorry⟩
def rangeC : Range := ⟨123, 312, by sorry⟩
def rangeD : Range := ⟨231, 312, by sorry⟩
def rangeE : Range := ⟨312, 321, by sorry⟩

/-- All ranges except E -/
def otherRanges : List Range := [rangeA, rangeB, rangeC, rangeD]

theorem adams_house_number_range :
  (∀ r ∈ otherRanges, ∃ r' ∈ otherRanges, r ≠ r' ∧ overlaps r r') ∧
  (∀ r ∈ otherRanges, ¬overlaps r rangeE) :=
by sorry

end adams_house_number_range_l3555_355541


namespace modulus_of_complex_fraction_l3555_355583

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + Complex.I) / (1 - Complex.I) → Complex.abs z = 1 := by
  sorry

end modulus_of_complex_fraction_l3555_355583


namespace min_value_squared_sum_l3555_355558

theorem min_value_squared_sum (x y : ℝ) (h : x * y = 1) :
  x^2 + 4*y^2 ≥ 4 ∧ ∃ (a b : ℝ), a * b = 1 ∧ a^2 + 4*b^2 = 4 :=
by sorry

end min_value_squared_sum_l3555_355558


namespace intersection_count_l3555_355542

-- Define the two curves
def curve1 (x y : ℝ) : Prop := (x + 2*y - 3) * (4*x - y + 1) = 0
def curve2 (x y : ℝ) : Prop := (2*x - y - 5) * (3*x + 4*y - 8) = 0

-- Define what it means for a point to be on both curves
def intersection_point (x y : ℝ) : Prop := curve1 x y ∧ curve2 x y

-- State the theorem
theorem intersection_count : 
  ∃ (points : Finset (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ points ↔ intersection_point p.1 p.2) ∧ 
    points.card = 4 := by
  sorry

end intersection_count_l3555_355542


namespace impossible_to_cover_modified_chessboard_l3555_355517

/-- Represents a chessboard with some squares removed -/
structure ModifiedChessboard where
  size : Nat
  removed : Finset (Nat × Nat)

/-- Represents a domino that covers two squares -/
structure Domino where
  square1 : Nat × Nat
  square2 : Nat × Nat

/-- Checks if a given set of dominos covers the modified chessboard -/
def covers (board : ModifiedChessboard) (dominos : Finset Domino) : Prop :=
  sorry

/-- The color of a square on a chessboard (assuming top-left is white) -/
def squareColor (pos : Nat × Nat) : Bool :=
  (pos.1 + pos.2) % 2 == 0

theorem impossible_to_cover_modified_chessboard :
  ∀ (dominos : Finset Domino),
    let board := ModifiedChessboard.mk 8 {(0, 0), (7, 7)}
    ¬ covers board dominos := by
  sorry

end impossible_to_cover_modified_chessboard_l3555_355517


namespace binomial_expansion_properties_l3555_355543

/-- The binomial expansion of (√x + 2/x²)¹⁰ -/
def binomial_expansion (x : ℝ) : ℕ → ℝ :=
  λ r => (Nat.choose 10 r) * (2^r) * (x^((10 - 5*r)/2))

/-- A term in the expansion is rational if its exponent is an integer -/
def is_rational_term (r : ℕ) : Prop :=
  (10 - 5*r) % 2 = 0

theorem binomial_expansion_properties :
  (∃ (S : Finset ℕ), S.card = 6 ∧ ∀ r, r ∈ S ↔ is_rational_term r) ∧
  (∃ r : ℕ, r = 7 ∧ ∀ k : ℕ, k ≠ r → |binomial_expansion 1 r| ≥ |binomial_expansion 1 k|) ∧
  binomial_expansion 1 7 = 15360 := by sorry

end binomial_expansion_properties_l3555_355543


namespace competition_earnings_difference_l3555_355502

/-- Represents the earnings of a seller for a single day -/
structure DayEarnings where
  regular_sales : ℝ
  discounted_sales : ℝ
  tax_rate : ℝ
  exchange_rate : ℝ

/-- Calculates the total earnings for a day after tax and currency conversion -/
def calculate_day_earnings (e : DayEarnings) : ℝ :=
  let total_sales := e.regular_sales + e.discounted_sales
  let after_tax := total_sales * (1 - e.tax_rate)
  after_tax * e.exchange_rate

/-- Represents the earnings of a seller for the two-day competition -/
structure CompetitionEarnings where
  day1 : DayEarnings
  day2 : DayEarnings

/-- Calculates the total earnings for the two-day competition -/
def calculate_total_earnings (e : CompetitionEarnings) : ℝ :=
  calculate_day_earnings e.day1 + calculate_day_earnings e.day2

/-- Theorem statement for the competition earnings -/
theorem competition_earnings_difference
  (bert_earnings tory_earnings : CompetitionEarnings)
  (h_bert_day1 : bert_earnings.day1 = {
    regular_sales := 9 * 18,
    discounted_sales := 3 * (18 * 0.85),
    tax_rate := 0.05,
    exchange_rate := 1
  })
  (h_bert_day2 : bert_earnings.day2 = {
    regular_sales := 10 * 15,
    discounted_sales := 0,
    tax_rate := 0.05,
    exchange_rate := 1.4
  })
  (h_tory_day1 : tory_earnings.day1 = {
    regular_sales := 10 * 20,
    discounted_sales := 5 * (20 * 0.9),
    tax_rate := 0.05,
    exchange_rate := 1
  })
  (h_tory_day2 : tory_earnings.day2 = {
    regular_sales := 8 * 18,
    discounted_sales := 0,
    tax_rate := 0.05,
    exchange_rate := 1.4
  }) :
  calculate_total_earnings tory_earnings - calculate_total_earnings bert_earnings = 71.82 := by
  sorry


end competition_earnings_difference_l3555_355502


namespace translation_theorem_l3555_355555

/-- Represents a quadratic function of the form y = a(x-h)^2 + k --/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a quadratic function horizontally --/
def translate (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h - d, k := f.k }

/-- The initial quadratic function y = 3(x-2)^2 + 1 --/
def initial_function : QuadraticFunction :=
  { a := 3, h := 2, k := 1 }

/-- Theorem stating that translating the initial function 2 units right then 2 units left
    results in y = 3x^2 + 3 --/
theorem translation_theorem :
  let f1 := translate initial_function (-2)
  let f2 := translate f1 2
  f2.a * (X - f2.h)^2 + f2.k = 3 * X^2 + 3 := by sorry

end translation_theorem_l3555_355555


namespace fish_value_in_rice_l3555_355500

/-- Represents the value of items in a barter system -/
structure BarterValue where
  fish : ℚ
  bread : ℚ
  rice : ℚ

/-- The barter system with given exchange rates -/
def barterSystem : BarterValue where
  fish := 1
  bread := 3/5
  rice := 1/10

theorem fish_value_in_rice (b : BarterValue) 
  (h1 : 5 * b.fish = 3 * b.bread) 
  (h2 : b.bread = 6 * b.rice) : 
  b.fish = 18/5 * b.rice := by
  sorry

#check fish_value_in_rice barterSystem

end fish_value_in_rice_l3555_355500


namespace kyle_gas_and_maintenance_l3555_355549

/-- Calculates the amount left for gas and maintenance given monthly income and expenses --/
def amount_left_for_gas_and_maintenance (monthly_income : ℕ) (rent utilities retirement_savings groceries insurance misc car_payment : ℕ) : ℕ :=
  monthly_income - (rent + utilities + retirement_savings + groceries + insurance + misc + car_payment)

/-- Theorem: Kyle's amount left for gas and maintenance is $350 --/
theorem kyle_gas_and_maintenance :
  amount_left_for_gas_and_maintenance 3200 1250 150 400 300 200 200 350 = 350 := by
  sorry

end kyle_gas_and_maintenance_l3555_355549


namespace quadratic_roots_imply_m_range_l3555_355514

theorem quadratic_roots_imply_m_range (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ∈ Set.Ioo 0 1 ∧ r₂ ∈ Set.Ioo 2 3 ∧ 
   r₁^2 - 2*m*r₁ + m^2 - 1 = 0 ∧ r₂^2 - 2*m*r₂ + m^2 - 1 = 0) →
  m ∈ Set.Ioo 1 2 := by
sorry

end quadratic_roots_imply_m_range_l3555_355514


namespace divisors_of_n_squared_l3555_355580

def has_exactly_four_divisors (n : ℕ) : Prop :=
  (∃ p : ℕ, Prime p ∧ n = p^3) ∨
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p * q)

theorem divisors_of_n_squared (n : ℕ) (h : has_exactly_four_divisors n) :
  (∃ d : ℕ, d = 7 ∧ (∀ x : ℕ, x ∣ n^2 ↔ x ∈ Finset.range (d + 1))) ∨
  (∃ d : ℕ, d = 9 ∧ (∀ x : ℕ, x ∣ n^2 ↔ x ∈ Finset.range (d + 1))) :=
by sorry

end divisors_of_n_squared_l3555_355580


namespace fraction_zero_implies_x_equals_two_l3555_355598

theorem fraction_zero_implies_x_equals_two (x : ℝ) :
  (x^2 - x - 2) / (x + 1) = 0 → x = 2 :=
by sorry

end fraction_zero_implies_x_equals_two_l3555_355598


namespace cubic_local_max_l3555_355592

/-- Given a cubic function with a local maximum, prove the product of two coefficients -/
theorem cubic_local_max (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^3 - a * x^2 - 2 * b * x + 2
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1) ∧ 
  (f 1 = -3) →
  a * b = 9 := by
sorry

end cubic_local_max_l3555_355592


namespace equation_solution_l3555_355582

theorem equation_solution : ∃ (y₁ y₂ y₃ : ℂ),
  y₁ = -Real.sqrt 3 ∧
  y₂ = -Real.sqrt 3 + Complex.I ∧
  y₃ = -Real.sqrt 3 - Complex.I ∧
  (∀ y : ℂ, (y^3 + 3*y^2*(Real.sqrt 3) + 9*y + 3*(Real.sqrt 3)) + (y + Real.sqrt 3) = 0 ↔ 
    y = y₁ ∨ y = y₂ ∨ y = y₃) := by
  sorry

end equation_solution_l3555_355582


namespace largest_angle_of_hexagon_l3555_355563

/-- Proves that in a convex hexagon with given interior angle measures, the largest angle is 4374/21 degrees -/
theorem largest_angle_of_hexagon (a : ℚ) : 
  (a + 2) + (2 * a - 3) + (3 * a + 1) + (4 * a) + (5 * a - 4) + (6 * a + 2) = 720 →
  max (a + 2) (max (2 * a - 3) (max (3 * a + 1) (max (4 * a) (max (5 * a - 4) (6 * a + 2))))) = 4374 / 21 := by
  sorry

end largest_angle_of_hexagon_l3555_355563


namespace more_solutions_without_plus_one_l3555_355565

/-- The upper bound for x, y, z, and t -/
def upperBound : ℕ := 10^6

/-- The number of integral solutions for x^2 - y^2 = z^3 - t^3 -/
def N : ℕ := sorry

/-- The number of integral solutions for x^2 - y^2 = z^3 - t^3 + 1 -/
def M : ℕ := sorry

/-- Theorem stating that N > M -/
theorem more_solutions_without_plus_one : N > M := by
  sorry

end more_solutions_without_plus_one_l3555_355565


namespace gcd_98_63_l3555_355538

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l3555_355538


namespace reading_time_difference_l3555_355539

/-- Proves that given Xanthia's and Molly's reading speeds and a book's page count,
    the difference in reading time is 180 minutes. -/
theorem reading_time_difference
  (xanthia_speed : ℕ) -- Xanthia's reading speed in pages per hour
  (molly_speed : ℕ) -- Molly's reading speed in pages per hour
  (book_pages : ℕ) -- Number of pages in the book
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 60)
  (h3 : book_pages = 360) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 180 :=
by
  sorry

#check reading_time_difference

end reading_time_difference_l3555_355539


namespace expand_polynomial_l3555_355544

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x + 7) = 4 * x^3 + 7 * x^2 - 8 * x + 21 := by
  sorry

end expand_polynomial_l3555_355544


namespace guest_bedroom_area_l3555_355586

/-- Proves that the area of each guest bedroom is 200 sq ft given the specified conditions --/
theorem guest_bedroom_area
  (total_rent : ℝ)
  (rent_rate : ℝ)
  (master_area : ℝ)
  (common_area : ℝ)
  (h1 : total_rent = 3000)
  (h2 : rent_rate = 2)
  (h3 : master_area = 500)
  (h4 : common_area = 600)
  : ∃ (guest_bedroom_area : ℝ),
    guest_bedroom_area = 200 ∧
    total_rent / rent_rate = master_area + common_area + 2 * guest_bedroom_area :=
by sorry

end guest_bedroom_area_l3555_355586


namespace frisbee_tournament_committees_l3555_355596

theorem frisbee_tournament_committees :
  let total_teams : ℕ := 4
  let members_per_team : ℕ := 8
  let host_committee_members : ℕ := 4
  let non_host_committee_members : ℕ := 2
  let total_committee_members : ℕ := 10

  (total_teams * (Nat.choose members_per_team host_committee_members) *
   (Nat.choose members_per_team non_host_committee_members) ^ (total_teams - 1)) = 6593280 :=
by sorry

end frisbee_tournament_committees_l3555_355596


namespace constant_sum_of_squares_l3555_355547

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of a point being on the major axis -/
def on_major_axis (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

/-- Definition of a line with slope 1/2 passing through (m, 0) -/
def line (m x y : ℝ) : Prop := y = (x - m) / 2

/-- Statement of the theorem -/
theorem constant_sum_of_squares (m x₁ y₁ x₂ y₂ : ℝ) : 
  ellipse 1 (Real.sqrt 3 / 2) →
  on_major_axis m →
  line m x₁ y₁ →
  line m x₂ y₂ →
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  (x₁ - m)^2 + y₁^2 + (x₂ - m)^2 + y₂^2 = 5 :=
sorry

end constant_sum_of_squares_l3555_355547


namespace geometry_book_pages_difference_l3555_355516

/-- Given that a new edition of a Geometry book has 450 pages and the old edition has 340 pages,
    prove that the new edition has 230 pages less than twice the number of pages in the old edition. -/
theorem geometry_book_pages_difference (new_edition : ℕ) (old_edition : ℕ)
  (h1 : new_edition = 450)
  (h2 : old_edition = 340) :
  2 * old_edition - new_edition = 230 := by
  sorry

end geometry_book_pages_difference_l3555_355516


namespace rectangular_field_length_l3555_355559

theorem rectangular_field_length (length width : ℝ) 
  (h1 : length * width = 144)
  (h2 : (length + 6) * width = 198) :
  length = 16 := by
  sorry

end rectangular_field_length_l3555_355559


namespace no_functions_satisfying_condition_l3555_355513

theorem no_functions_satisfying_condition : 
  ¬∃ (f g : ℝ → ℝ), ∀ x y : ℝ, x ≠ y → |f x - f y| + |g x - g y| > 1 := by
  sorry

end no_functions_satisfying_condition_l3555_355513


namespace polygon_quadrilateral_iff_exterior_eq_interior_l3555_355551

/-- A polygon is a quadrilateral if and only if the sum of its exterior angles
    equals the sum of its interior angles. -/
theorem polygon_quadrilateral_iff_exterior_eq_interior :
  ∀ n : ℕ, n ≥ 3 →
  (n = 4 ↔ (n - 2) * 180 = 360) :=
by sorry

end polygon_quadrilateral_iff_exterior_eq_interior_l3555_355551


namespace complex_number_problem_l3555_355594

/-- Given a complex number z = bi (b ∈ ℝ) such that (z-2)/(1+i) is real,
    prove that z = -2i and (m+z)^2 is in the first quadrant iff m < -2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) (h1 : z = Complex.I * b) 
    (h2 : ∃ (r : ℝ), (z - 2) / (1 + Complex.I) = r) :
  z = -2 * Complex.I ∧ 
  ∀ m : ℝ, (Complex.re ((m + z)^2) > 0 ∧ Complex.im ((m + z)^2) > 0) ↔ m < -2 := by
  sorry

end complex_number_problem_l3555_355594


namespace disconnected_circuit_scenarios_l3555_355520

/-- Represents a circuit with solder points -/
structure Circuit where
  total_points : ℕ
  is_disconnected : Bool

/-- Calculates the number of scenarios where solder points can fall off -/
def scenarios_with_fallen_points (c : Circuit) : ℕ :=
  2^c.total_points - 1

/-- Theorem: For a disconnected circuit with 6 solder points, there are 63 scenarios of fallen points -/
theorem disconnected_circuit_scenarios :
  ∀ (c : Circuit), c.total_points = 6 → c.is_disconnected = true →
  scenarios_with_fallen_points c = 63 := by
  sorry

#check disconnected_circuit_scenarios

end disconnected_circuit_scenarios_l3555_355520


namespace bus_stop_time_l3555_355504

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages -/
theorem bus_stop_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 60 → speed_with_stops = 50 → 
  (60 - (60 * speed_with_stops / speed_without_stops)) = 10 := by
  sorry

#check bus_stop_time

end bus_stop_time_l3555_355504


namespace subtraction_from_percentage_l3555_355525

theorem subtraction_from_percentage (n : ℝ) : n = 70 → (n * 0.5 - 10 = 25) := by
  sorry

end subtraction_from_percentage_l3555_355525


namespace min_value_expression_l3555_355536

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / (3 * a) + 3 / b ≥ 5 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ b / (3 * a) + 3 / b = 5 :=
by sorry

end min_value_expression_l3555_355536


namespace bus_driver_compensation_l3555_355585

-- Define the constants
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours_worked : ℕ := 44

-- Define the functions
def overtime_rate : ℝ := regular_rate * (1 + overtime_rate_increase)

def calculate_compensation (hours : ℕ) : ℝ :=
  if hours ≤ regular_hours then
    hours * regular_rate
  else
    regular_hours * regular_rate + (hours - regular_hours) * overtime_rate

-- Theorem to prove
theorem bus_driver_compensation :
  calculate_compensation total_hours_worked = 752 :=
by sorry

end bus_driver_compensation_l3555_355585


namespace fourth_power_nested_sqrt_l3555_355581

theorem fourth_power_nested_sqrt : 
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 2)))^4 = 7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 2 :=
by sorry

end fourth_power_nested_sqrt_l3555_355581


namespace greatest_common_divisor_of_90_and_m_l3555_355554

theorem greatest_common_divisor_of_90_and_m (m : ℕ) 
  (h1 : ∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
    (∀ (d : ℕ), d ∣ 90 ∧ d ∣ m ↔ d = d1 ∨ d = d2 ∨ d = d3)) :
  ∃ (d : ℕ), d ∣ 90 ∧ d ∣ m ∧ d = 9 ∧ 
    ∀ (x : ℕ), x ∣ 90 ∧ x ∣ m → x ≤ d :=
by sorry

end greatest_common_divisor_of_90_and_m_l3555_355554


namespace inequality_solution_1_inequality_solution_2_l3555_355528

-- Part 1
theorem inequality_solution_1 (x : ℝ) :
  (x + 1) / (x - 2) ≥ 3 ↔ 2 < x ∧ x ≤ 7/2 :=
sorry

-- Part 2
theorem inequality_solution_2 (x a : ℝ) :
  x^2 - a*x - 2*a^2 ≤ 0 ↔
    (a = 0 ∧ x = 0) ∨
    (a > 0 ∧ -a ≤ x ∧ x ≤ 2*a) ∨
    (a < 0 ∧ 2*a ≤ x ∧ x ≤ -a) :=
sorry

end inequality_solution_1_inequality_solution_2_l3555_355528


namespace arithmetic_sequence_common_difference_l3555_355567

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a 1 + (n - 1) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) (h : arithmetic_sequence a d) :
  (a 4 + 4)^2 = (a 2 + 2) * (a 6 + 6) → d = -1 := by
  sorry

#check arithmetic_sequence_common_difference

end arithmetic_sequence_common_difference_l3555_355567


namespace parabolas_intersect_l3555_355584

/-- Parabola function -/
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- Theorem: All parabolas with p + q = 2019 intersect at (1, 2020) -/
theorem parabolas_intersect (p q : ℝ) (h : p + q = 2019) : parabola p q 1 = 2020 := by
  sorry

end parabolas_intersect_l3555_355584


namespace certain_number_is_two_l3555_355515

theorem certain_number_is_two :
  ∃ x : ℚ, (287 * 287) + (269 * 269) - x * (287 * 269) = 324 ∧ x = 2 := by
  sorry

end certain_number_is_two_l3555_355515


namespace max_value_implies_a_l3555_355561

def f (a x : ℝ) : ℝ := a * x^3 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = 1/4 ∨ a = -1/11 := by
sorry

end max_value_implies_a_l3555_355561


namespace right_triangle_sets_l3555_355521

theorem right_triangle_sets : ∃! (a b c : ℝ), (a = 4 ∧ b = 6 ∧ c = 8) ∧
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ∧
  ((3^2 + 4^2 = 5^2) ∧ (5^2 + 12^2 = 13^2) ∧ (2^2 + 3^2 = (Real.sqrt 13)^2)) :=
by sorry

end right_triangle_sets_l3555_355521


namespace paper_fold_ratio_is_four_fifths_l3555_355507

/-- Represents the dimensions and folding of a rectangular piece of paper. -/
structure PaperFold where
  length : ℝ
  width : ℝ
  fold_ratio : ℝ
  division_parts : ℕ

/-- Calculates the ratio of the new visible area to the original area after folding. -/
def visible_area_ratio (paper : PaperFold) : ℝ :=
  -- Implementation details would go here
  sorry

/-- Theorem stating that for a specific paper folding scenario, the visible area ratio is 8/10. -/
theorem paper_fold_ratio_is_four_fifths :
  let paper : PaperFold := {
    length := 5,
    width := 2,
    fold_ratio := 1/2,
    division_parts := 3
  }
  visible_area_ratio paper = 8/10 := by
  sorry

end paper_fold_ratio_is_four_fifths_l3555_355507


namespace smallest_overlap_percentage_l3555_355545

theorem smallest_overlap_percentage (total : ℝ) (books_percent : ℝ) (movies_percent : ℝ)
  (h_total : total > 0)
  (h_books : books_percent = 95 / 100)
  (h_movies : movies_percent = 85 / 100) :
  (books_percent + movies_percent - 1 : ℝ) = 80 / 100 := by
  sorry

end smallest_overlap_percentage_l3555_355545


namespace simplify_fraction_l3555_355537

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a + 1 / b - (2 * a + b) / (2 * a * b) = 1 / (2 * a) := by
  sorry

end simplify_fraction_l3555_355537
