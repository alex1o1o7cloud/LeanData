import Mathlib

namespace final_time_sum_l178_17861

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the time after a given duration -/
def timeAfter (initial : Time) (duration : Time) : Time :=
  sorry

/-- Converts a Time to its representation on a 12-hour clock -/
def to12HourClock (t : Time) : Time :=
  sorry

theorem final_time_sum (initial : Time) (duration : Time) : 
  initial.hours = 15 ∧ initial.minutes = 0 ∧ initial.seconds = 0 →
  duration.hours = 158 ∧ duration.minutes = 55 ∧ duration.seconds = 32 →
  let finalTime := to12HourClock (timeAfter initial duration)
  finalTime.hours + finalTime.minutes + finalTime.seconds = 92 :=
sorry

end final_time_sum_l178_17861


namespace complex_equation_solution_l178_17881

theorem complex_equation_solution (z : ℂ) :
  z / (z - Complex.I) = Complex.I → z = (1 : ℂ) / 2 + Complex.I / 2 := by
  sorry

end complex_equation_solution_l178_17881


namespace stating_popsicle_count_l178_17884

/-- The number of popsicles in a box with specific melting rate properties -/
def num_popsicles : ℕ := 6

/-- The melting rate factor between consecutive popsicles -/
def melting_rate_factor : ℕ := 2

/-- The relative melting rate of the last popsicle compared to the first -/
def last_to_first_rate : ℕ := 32

/-- 
Theorem stating that the number of popsicles in the box is 6, given the melting rate properties
-/
theorem popsicle_count :
  (melting_rate_factor ^ (num_popsicles - 1) = last_to_first_rate) →
  num_popsicles = 6 := by
sorry

end stating_popsicle_count_l178_17884


namespace inverse_function_point_l178_17823

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log x / Real.log a

def has_inverse_point (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g x = y

theorem inverse_function_point (a : ℝ) :
  (a > 0 ∧ a ≠ 1) →
  has_inverse_point (f a) 2 4 →
  a = 4 := by sorry

end inverse_function_point_l178_17823


namespace rectangle_dimensions_l178_17872

theorem rectangle_dimensions (w : ℝ) (h : w > 0) :
  let l := 2 * w
  let area := w * l
  let perimeter := 2 * (w + l)
  area = 2 * perimeter → w = 6 ∧ l = 12 := by
  sorry

end rectangle_dimensions_l178_17872


namespace einstein_born_on_friday_l178_17813

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 400 == 0 || (year % 4 == 0 && year % 100 ≠ 0)

/-- Einstein's birth year -/
def einsteinBirthYear : Nat := 1865

/-- Einstein's 160th anniversary year -/
def anniversaryYear : Nat := 2025

/-- Day of the week of Einstein's 160th anniversary -/
def anniversaryDayOfWeek : DayOfWeek := DayOfWeek.Friday

/-- Calculates the day of the week Einstein was born -/
def einsteinBirthDayOfWeek : DayOfWeek := sorry

theorem einstein_born_on_friday :
  einsteinBirthDayOfWeek = DayOfWeek.Friday := by sorry

end einstein_born_on_friday_l178_17813


namespace quadratic_intersection_range_l178_17899

/-- For a quadratic function y = 2mx^2 + (8m+1)x + 8m that intersects the x-axis, 
    the range of m is [m ≥ -1/16 and m ≠ 0] -/
theorem quadratic_intersection_range (m : ℝ) : 
  (∃ x, 2*m*x^2 + (8*m + 1)*x + 8*m = 0) → 
  (m ≥ -1/16 ∧ m ≠ 0) :=
by sorry

end quadratic_intersection_range_l178_17899


namespace max_value_of_f_on_interval_l178_17883

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 + 3*x

-- State the theorem
theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ f c) ∧
  f c = 2 :=
sorry

end max_value_of_f_on_interval_l178_17883


namespace max_female_students_theorem_min_group_size_theorem_l178_17835

/-- Represents the composition of a study group --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

/-- The maximum number of female students when there are 4 teachers --/
def max_female_students_with_4_teachers : ℕ := 6

/-- The minimum number of people in a valid study group --/
def min_group_size : ℕ := 12

/-- Theorem: The maximum number of female students is 6 when there are 4 teachers --/
theorem max_female_students_theorem :
  ∀ g : StudyGroup, is_valid_group g → g.teachers = 4 → g.female_students ≤ max_female_students_with_4_teachers :=
sorry

/-- Theorem: The minimum number of people in a valid study group is 12 --/
theorem min_group_size_theorem :
  ∀ g : StudyGroup, is_valid_group g → g.male_students + g.female_students + g.teachers ≥ min_group_size :=
sorry

end max_female_students_theorem_min_group_size_theorem_l178_17835


namespace max_intersections_l178_17829

/-- Given 15 points on the positive x-axis and 10 points on the positive y-axis,
    with segments connecting each point on the x-axis to each point on the y-axis,
    the maximum number of intersection points in the interior of the first quadrant is 4725. -/
theorem max_intersections (x_points y_points : ℕ) (h1 : x_points = 15) (h2 : y_points = 10) :
  (x_points.choose 2) * (y_points.choose 2) = 4725 := by
  sorry

end max_intersections_l178_17829


namespace elizas_height_l178_17896

/-- Given the heights of Eliza's siblings and their total height, prove Eliza's height -/
theorem elizas_height (total_height : ℕ) (sibling1_height : ℕ) (sibling2_height : ℕ) (sibling3_height : ℕ) (sibling4_height : ℕ) :
  total_height = 330 ∧ 
  sibling1_height = 66 ∧ 
  sibling2_height = 66 ∧ 
  sibling3_height = 60 ∧ 
  sibling4_height = sibling1_height + 2 →
  ∃ (eliza_height : ℕ), eliza_height = 68 ∧ 
    total_height = sibling1_height + sibling2_height + sibling3_height + sibling4_height + eliza_height :=
by
  sorry

end elizas_height_l178_17896


namespace logical_equivalence_l178_17811

theorem logical_equivalence (P Q R S : Prop) :
  ((P ∨ ¬R) → (¬Q ∧ S)) ↔ ((Q ∨ ¬S) → (¬P ∧ R)) :=
by sorry

end logical_equivalence_l178_17811


namespace sum_of_three_numbers_l178_17831

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → b = 10 → (a + b + c) / 3 = a + 20 → (a + b + c) / 3 = c - 10 → 
  a + b + c = 0 := by
  sorry

end sum_of_three_numbers_l178_17831


namespace ratio_exists_l178_17830

theorem ratio_exists : ∃ (m n : ℤ), 
  m > 100 ∧ 
  n > 100 ∧ 
  m + n = 300 ∧ 
  3 * n = 2 * m := by
sorry

end ratio_exists_l178_17830


namespace range_of_f_l178_17898

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x < 5}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -4 ≤ y ∧ y < 5} :=
sorry

end range_of_f_l178_17898


namespace gcd_7524_16083_l178_17818

theorem gcd_7524_16083 : Nat.gcd 7524 16083 = 1 := by
  sorry

end gcd_7524_16083_l178_17818


namespace real_part_of_complex_expression_l178_17814

theorem real_part_of_complex_expression : Complex.re (1 + 2 / (Complex.I + 1)) = 2 := by
  sorry

end real_part_of_complex_expression_l178_17814


namespace abc_inequality_l178_17800

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end abc_inequality_l178_17800


namespace johns_money_to_father_l178_17827

def initial_amount : ℚ := 200
def fraction_to_mother : ℚ := 3/8
def amount_left : ℚ := 65

theorem johns_money_to_father :
  (initial_amount - fraction_to_mother * initial_amount - amount_left) / initial_amount = 3/10 := by
  sorry

end johns_money_to_father_l178_17827


namespace stratified_sampling_correct_l178_17863

/-- Represents the number of people in each age group -/
structure PopulationGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Represents the number of people to be sampled from each age group -/
structure SampleGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the total population -/
def totalPopulation (p : PopulationGroups) : ℕ :=
  p.elderly + p.middleAged + p.young

/-- Calculates the proportion of each group in the sample -/
def sampleProportion (p : PopulationGroups) (sampleSize : ℕ) : SampleGroups :=
  let total := totalPopulation p
  { elderly := (p.elderly * sampleSize + total - 1) / total,
    middleAged := (p.middleAged * sampleSize + total - 1) / total,
    young := (p.young * sampleSize + total - 1) / total }

/-- The main theorem to prove -/
theorem stratified_sampling_correct 
  (population : PopulationGroups)
  (sampleSize : ℕ) :
  population.elderly = 28 →
  population.middleAged = 54 →
  population.young = 81 →
  sampleSize = 36 →
  sampleProportion population sampleSize = { elderly := 6, middleAged := 12, young := 18 } := by
  sorry


end stratified_sampling_correct_l178_17863


namespace special_ellipse_equation_l178_17845

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- Semi-major axis length -/
  a : ℝ
  /-- Semi-minor axis length -/
  b : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- The axes of symmetry are the coordinate axes -/
  axes_are_coordinate_axes : True
  /-- One endpoint of minor axis and two foci form equilateral triangle -/
  equilateral_triangle : b / c = Real.sqrt 3
  /-- Foci are on the y-axis -/
  foci_on_y_axis : True
  /-- Relation between a and c -/
  a_minus_c : a - c = Real.sqrt 3
  /-- Pythagorean theorem for ellipse -/
  ellipse_relation : a^2 = b^2 + c^2

/-- The equation of the special ellipse -/
def ellipse_equation (e : SpecialEllipse) : Prop :=
  ∀ x y : ℝ, y^2 / 12 + x^2 / 9 = 1 ↔ y^2 / e.a^2 + x^2 / e.b^2 = 1

/-- The main theorem about the special ellipse -/
theorem special_ellipse_equation (e : SpecialEllipse) : ellipse_equation e := by
  sorry

end special_ellipse_equation_l178_17845


namespace opposites_equation_l178_17849

theorem opposites_equation (x : ℝ) : (2 * x - 1 = -(-x + 5)) → (2 * x - 1 = x - 5) := by
  sorry

end opposites_equation_l178_17849


namespace choir_arrangement_min_choir_members_l178_17805

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
by sorry

theorem min_choir_members : 
  ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
by sorry

end choir_arrangement_min_choir_members_l178_17805


namespace teacher_student_arrangements_l178_17857

/-- The number of arrangements for a teacher and students in a row --/
def arrangements (n : ℕ) : ℕ :=
  (n - 2) * n.factorial

/-- The problem statement --/
theorem teacher_student_arrangements :
  arrangements 6 = 480 := by
  sorry

end teacher_student_arrangements_l178_17857


namespace integer_pair_divisibility_l178_17869

theorem integer_pair_divisibility (a b : ℤ) (ha : a > 1) (hb : b > 1)
  (hab : a ∣ (b + 1)) (hba : b ∣ (a^3 - 1)) :
  (∃ s : ℤ, s ≥ 2 ∧ a = s ∧ b = s^3 - 1) ∨
  (∃ s : ℤ, s ≥ 3 ∧ a = s ∧ b = s - 1) :=
by sorry

end integer_pair_divisibility_l178_17869


namespace fruit_basket_strawberries_l178_17833

def fruit_basket (num_strawberries : ℕ) : Prop :=
  let banana_cost : ℕ := 1
  let apple_cost : ℕ := 2
  let avocado_cost : ℕ := 3
  let strawberry_dozen_cost : ℕ := 4
  let half_grape_bunch_cost : ℕ := 2
  let total_cost : ℕ := 28
  let num_bananas : ℕ := 4
  let num_apples : ℕ := 3
  let num_avocados : ℕ := 2
  banana_cost * num_bananas +
  apple_cost * num_apples +
  avocado_cost * num_avocados +
  strawberry_dozen_cost * (num_strawberries / 12) +
  half_grape_bunch_cost * 2 = total_cost

theorem fruit_basket_strawberries : 
  ∃ (n : ℕ), fruit_basket n ∧ n = 24 :=
sorry

end fruit_basket_strawberries_l178_17833


namespace johns_age_l178_17889

theorem johns_age (john dad : ℕ) 
  (h1 : john + 34 = dad) 
  (h2 : john + dad = 84) : 
  john = 25 := by
sorry

end johns_age_l178_17889


namespace sin_2alpha_plus_pi_3_l178_17801

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ) + 1

theorem sin_2alpha_plus_pi_3 (ω φ α : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ π/2 →
  (∀ x : ℝ, f ω φ (x + π/ω) = f ω φ x) →
  f ω φ (π/3) = 2 →
  f ω φ α = 8/5 →
  π/3 < α ∧ α < 5*π/6 →
  sin (2*α + π/3) = -24/25 := by
  sorry

end sin_2alpha_plus_pi_3_l178_17801


namespace intersection_A_B_l178_17897

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l178_17897


namespace complement_intersection_theorem_l178_17841

def U : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Finset ℕ := {0, 1, 3, 5, 8}
def B : Finset ℕ := {2, 4, 5, 6, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {7, 9} := by sorry

end complement_intersection_theorem_l178_17841


namespace striped_cube_loop_probability_l178_17819

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  /-- Each face has a stripe from midpoint to midpoint of opposite edges -/
  faces : Fin 6 → Bool
  /-- For any two opposing faces, one stripe must be perpendicular to the other -/
  opposing_perpendicular : ∀ i : Fin 3, faces i ≠ faces (i + 3)

/-- Predicate to check if a given striped cube forms a valid loop -/
def forms_loop (cube : StripedCube) : Prop :=
  ∃ i : Fin 3, cube.faces i = cube.faces (i + 3) ∧
    (cube.faces ((i + 1) % 3) ≠ cube.faces ((i + 4) % 3)) ∧
    (cube.faces ((i + 2) % 3) ≠ cube.faces ((i + 5) % 3))

/-- The total number of valid striped cube configurations -/
def total_configurations : ℕ := 64

/-- The number of striped cube configurations that form a loop -/
def loop_configurations : ℕ := 6

/-- Theorem stating the probability of a striped cube forming a loop -/
theorem striped_cube_loop_probability :
  (loop_configurations : ℚ) / total_configurations = 3 / 32 := by
  sorry

end striped_cube_loop_probability_l178_17819


namespace maria_furniture_assembly_l178_17809

/-- Given the number of chairs, tables, and total assembly time, 
    calculate the time spent on each piece of furniture. -/
def time_per_piece (chairs : ℕ) (tables : ℕ) (total_time : ℕ) : ℚ :=
  (total_time : ℚ) / (chairs + tables : ℚ)

/-- Theorem stating that for 2 chairs, 2 tables, and 32 minutes total time,
    the time per piece is 8 minutes. -/
theorem maria_furniture_assembly : 
  time_per_piece 2 2 32 = 8 := by
  sorry

end maria_furniture_assembly_l178_17809


namespace nora_paid_90_dimes_l178_17853

/-- The number of dimes in one dollar -/
def dimes_per_dollar : ℕ := 10

/-- The cost of the watch in dollars -/
def watch_cost : ℕ := 9

/-- The number of dimes Nora paid for the watch -/
def dimes_paid : ℕ := watch_cost * dimes_per_dollar

theorem nora_paid_90_dimes : dimes_paid = 90 := by
  sorry

end nora_paid_90_dimes_l178_17853


namespace yearly_savings_ratio_l178_17802

-- Define the fraction of salary spent each month
def fraction_spent : ℚ := 0.6666666666666667

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Theorem statement
theorem yearly_savings_ratio :
  (1 - fraction_spent) * months_in_year = 4 := by
  sorry

end yearly_savings_ratio_l178_17802


namespace investment_average_rate_l178_17834

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) : 
  total = 5500 ∧ 
  rate1 = 0.03 ∧ 
  rate2 = 0.07 ∧ 
  (∃ x : ℝ, x > 0 ∧ x < total ∧ rate1 * (total - x) = rate2 * x) →
  (rate1 * (total - (rate2 * total) / (rate1 + rate2)) + rate2 * ((rate2 * total) / (rate1 + rate2))) / total = 0.042 := by
  sorry

#check investment_average_rate

end investment_average_rate_l178_17834


namespace cake_distribution_l178_17825

theorem cake_distribution (n : ℕ) (initial_cakes : ℕ) : 
  n = 5 →
  initial_cakes = 2 * (n * (n - 1)) →
  initial_cakes = 40 :=
by
  sorry

end cake_distribution_l178_17825


namespace f_sum_theorem_l178_17842

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_sum_theorem : f π + (deriv f) (π / 2) = -3 / π := by
  sorry

end f_sum_theorem_l178_17842


namespace smallest_clock_equivalent_is_nine_l178_17828

/-- A number is clock equivalent to its square if it's congruent to its square modulo 12 -/
def IsClockEquivalent (n : ℕ) : Prop := n ≡ n^2 [MOD 12]

/-- The smallest number greater than 5 that is clock equivalent to its square -/
def SmallestClockEquivalent : ℕ := 9

theorem smallest_clock_equivalent_is_nine :
  IsClockEquivalent SmallestClockEquivalent ∧
  ∀ n : ℕ, 5 < n ∧ n < SmallestClockEquivalent → ¬IsClockEquivalent n :=
by sorry

end smallest_clock_equivalent_is_nine_l178_17828


namespace keyword_selection_theorem_l178_17895

theorem keyword_selection_theorem (n m k : ℕ) (h1 : n = 12) (h2 : m = 4) (h3 : k = 2) : 
  (Nat.choose n k * Nat.choose m 1 + Nat.choose m k) + 
  (Nat.choose n (k + 1) * Nat.choose m 1 + Nat.choose n k * Nat.choose m 2 + Nat.choose m (k + 1)) = 202 := by
  sorry

end keyword_selection_theorem_l178_17895


namespace cubic_inequality_l178_17806

theorem cubic_inequality (x : ℝ) : 
  x^3 - 12*x^2 + 36*x + 8 > 0 ↔ x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29 := by
  sorry

end cubic_inequality_l178_17806


namespace greatest_integer_solution_l178_17826

theorem greatest_integer_solution (x : ℤ) : (7 - 3 * x > 20) ↔ x ≤ -5 := by sorry

end greatest_integer_solution_l178_17826


namespace amanda_notebooks_l178_17807

theorem amanda_notebooks (initial : ℕ) : 
  initial + 6 - 2 = 14 → initial = 10 := by
  sorry

end amanda_notebooks_l178_17807


namespace stock_market_investment_l178_17824

theorem stock_market_investment (P : ℝ) (x : ℝ) (h : P > 0) :
  (P + x / 100 * P) * (1 - 30 / 100) = P * (1 + 4.999999999999982 / 100) →
  x = 50 := by
sorry

end stock_market_investment_l178_17824


namespace k_value_l178_17840

theorem k_value (a b c k : ℝ) 
  (h1 : 2 * a / (b + c) = k) 
  (h2 : 2 * b / (a + c) = k) 
  (h3 : 2 * c / (a + b) = k) : 
  k = 1 ∨ k = -2 := by
  sorry

end k_value_l178_17840


namespace monomial_evaluation_l178_17888

theorem monomial_evaluation : 0.007 * (-5)^7 * 2^9 = -280000 := by
  sorry

end monomial_evaluation_l178_17888


namespace negation_equivalence_l178_17846

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by sorry

end negation_equivalence_l178_17846


namespace cavalier_projection_triangle_area_l178_17838

/-- Given a right-angled triangle represented in an oblique cavalier projection
    with a hypotenuse of √2a, prove that its area is √2a² -/
theorem cavalier_projection_triangle_area (a : ℝ) (h : a > 0) :
  let leg1 := Real.sqrt 2 * a
  let leg2 := 2 * a
  (1 / 2) * leg1 * leg2 = Real.sqrt 2 * a^2 := by
  sorry

end cavalier_projection_triangle_area_l178_17838


namespace penny_stack_more_valuable_l178_17810

/-- Represents a stack of coins -/
structure CoinStack :=
  (onePence : ℕ)
  (twoPence : ℕ)
  (fivePence : ℕ)

/-- Calculates the height of a coin stack in millimeters -/
def stackHeight (stack : CoinStack) : ℚ :=
  1.6 * stack.onePence + 2.05 * stack.twoPence + 1.75 * stack.fivePence

/-- Calculates the value of a coin stack in pence -/
def stackValue (stack : CoinStack) : ℕ :=
  stack.onePence + 2 * stack.twoPence + 5 * stack.fivePence

/-- Checks if a stack is valid according to the problem constraints -/
def isValidStack (stack : CoinStack) : Prop :=
  stackHeight stack = stackValue stack ∧ 
  (stack.onePence > 0 ∨ stack.twoPence > 0 ∨ stack.fivePence > 0)

/-- Joe's optimal stack using only 1p and 5p coins -/
def joesStack : CoinStack :=
  ⟨65, 0, 12⟩

/-- Penny's optimal stack using only 2p and 5p coins -/
def pennysStack : CoinStack :=
  ⟨0, 65, 1⟩

theorem penny_stack_more_valuable :
  isValidStack joesStack ∧
  isValidStack pennysStack ∧
  stackValue pennysStack > stackValue joesStack :=
sorry

end penny_stack_more_valuable_l178_17810


namespace coin_touch_black_probability_l178_17852

/-- Represents the square layout with black regions -/
structure SquareLayout where
  side_length : ℝ
  corner_triangle_leg : ℝ
  center_circle_diameter : ℝ

/-- Represents a coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin touching any black region -/
def probability_touch_black (layout : SquareLayout) (coin : Coin) : ℝ :=
  sorry

/-- Theorem statement for the probability problem -/
theorem coin_touch_black_probability
  (layout : SquareLayout)
  (coin : Coin)
  (h1 : layout.side_length = 6)
  (h2 : layout.corner_triangle_leg = 1)
  (h3 : layout.center_circle_diameter = 2)
  (h4 : coin.diameter = 2) :
  probability_touch_black layout coin = (2 + Real.pi) / 16 := by
  sorry

end coin_touch_black_probability_l178_17852


namespace last_three_digits_of_7_to_103_l178_17878

theorem last_three_digits_of_7_to_103 :
  7^103 ≡ 343 [ZMOD 1000] := by
sorry

end last_three_digits_of_7_to_103_l178_17878


namespace solution_bounded_l178_17839

open Real

/-- A function satisfying the differential equation y'' + e^x y = 0 is bounded -/
theorem solution_bounded (f : ℝ → ℝ) (hf : ∀ x, (deriv^[2] f) x + exp x * f x = 0) :
  ∃ M, ∀ x, |f x| ≤ M :=
sorry

end solution_bounded_l178_17839


namespace sugar_calculation_l178_17892

theorem sugar_calculation (num_packs : ℕ) (pack_weight : ℕ) (leftover : ℕ) :
  num_packs = 30 →
  pack_weight = 350 →
  leftover = 50 →
  num_packs * pack_weight + leftover = 10550 := by
  sorry

end sugar_calculation_l178_17892


namespace smallest_block_volume_l178_17821

/-- A rectangular block made of 1-cm cubes -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The number of cubes in the block -/
def Block.volume (b : Block) : ℕ := b.length * b.width * b.height

/-- The number of cubes not visible when three faces are shown -/
def Block.hiddenCubes (b : Block) : ℕ := (b.length - 1) * (b.width - 1) * (b.height - 1)

/-- One dimension is at least 5 -/
def Block.hasLargeDimension (b : Block) : Prop :=
  b.length ≥ 5 ∨ b.width ≥ 5 ∨ b.height ≥ 5

theorem smallest_block_volume (b : Block) :
  b.hiddenCubes = 252 →
  b.hasLargeDimension →
  ∀ b' : Block, b'.hiddenCubes = 252 → b'.hasLargeDimension → b.volume ≤ b'.volume →
  b.volume = 280 := by
  sorry

end smallest_block_volume_l178_17821


namespace intersection_of_AB_and_CD_l178_17875

def A : ℝ × ℝ × ℝ := (2, -1, 2)
def B : ℝ × ℝ × ℝ := (12, -11, 7)
def C : ℝ × ℝ × ℝ := (1, 4, -7)
def D : ℝ × ℝ × ℝ := (4, -2, 13)

def line_intersection (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem intersection_of_AB_and_CD :
  line_intersection A B C D = (8/3, -7/3, 7/3) := by
  sorry

end intersection_of_AB_and_CD_l178_17875


namespace ferris_wheel_cost_l178_17890

/-- The cost of a Ferris wheel ride given the conditions of Zach's amusement park visit -/
theorem ferris_wheel_cost 
  (total_rides : Nat) 
  (roller_coaster_cost log_ride_cost : Nat) 
  (zach_initial_tickets zach_additional_tickets : Nat) :
  total_rides = 3 →
  roller_coaster_cost = 7 →
  log_ride_cost = 1 →
  zach_initial_tickets = 1 →
  zach_additional_tickets = 9 →
  roller_coaster_cost + log_ride_cost + 2 = zach_initial_tickets + zach_additional_tickets :=
by sorry

end ferris_wheel_cost_l178_17890


namespace trapezoid_side_length_l178_17886

/-- Represents a trapezoid EFGH -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  EG : ℝ
  area : ℝ

/-- Theorem stating the length of EG in the given trapezoid -/
theorem trapezoid_side_length (t : Trapezoid) 
  (h1 : t.EF = 10)
  (h2 : t.GH = 14)
  (h3 : t.area = 72)
  (h4 : t.EG = (((t.GH - t.EF) / 2) ^ 2 + (2 * t.area / (t.EF + t.GH)) ^ 2).sqrt) :
  t.EG = 2 * Real.sqrt 10 := by
  sorry

end trapezoid_side_length_l178_17886


namespace qizhi_median_is_65_l178_17822

/-- Represents the homework duration data for a group of students -/
structure HomeworkData where
  durations : List Nat
  counts : List Nat
  total_students : Nat

/-- Calculates the median of a dataset given its HomeworkData -/
def calculate_median (data : HomeworkData) : Rat :=
  sorry

/-- The specific homework data for the problem -/
def qizhi_data : HomeworkData :=
  { durations := [50, 60, 70, 80],
    counts := [14, 11, 10, 15],
    total_students := 50 }

/-- Theorem stating that the median of the given homework data is 65 minutes -/
theorem qizhi_median_is_65 : calculate_median qizhi_data = 65 := by
  sorry

end qizhi_median_is_65_l178_17822


namespace tan_theta_value_l178_17836

theorem tan_theta_value (θ : ℝ) (h : Real.tan (π / 4 + θ) = 1 / 2) : Real.tan θ = -1 / 3 := by
  sorry

end tan_theta_value_l178_17836


namespace lindas_cookies_l178_17837

theorem lindas_cookies (classmates : Nat) (cookies_per_student : Nat) 
  (cookies_per_batch : Nat) (oatmeal_batches : Nat) (additional_batches : Nat) :
  classmates = 24 →
  cookies_per_student = 10 →
  cookies_per_batch = 48 →
  oatmeal_batches = 1 →
  additional_batches = 2 →
  ∃ (chocolate_chip_batches : Nat),
    chocolate_chip_batches * cookies_per_batch + 
    oatmeal_batches * cookies_per_batch + 
    additional_batches * cookies_per_batch = 
    classmates * cookies_per_student ∧
    chocolate_chip_batches = 2 :=
by sorry

end lindas_cookies_l178_17837


namespace jake_weight_loss_l178_17885

theorem jake_weight_loss (jake_current sister_current total_current : ℕ) 
  (h1 : jake_current + sister_current = total_current)
  (h2 : jake_current = 156)
  (h3 : total_current = 224) :
  ∃ (weight_loss : ℕ), jake_current - weight_loss = 2 * (sister_current - weight_loss) ∧ weight_loss = 20 := by
sorry

end jake_weight_loss_l178_17885


namespace afternoon_pear_sales_l178_17832

/-- Given a salesman who sold pears in the morning and afternoon, this theorem proves
    that if he sold twice as much in the afternoon as in the morning, and the total
    amount sold was 480 kilograms, then he sold 320 kilograms in the afternoon. -/
theorem afternoon_pear_sales (morning_sales afternoon_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  morning_sales + afternoon_sales = 480 →
  afternoon_sales = 320 := by
  sorry

end afternoon_pear_sales_l178_17832


namespace determinant_of_specific_matrix_l178_17862

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 2, 6]
  Matrix.det A = 24 := by
  sorry

end determinant_of_specific_matrix_l178_17862


namespace spanish_only_count_l178_17808

/-- Represents the number of students in different language classes -/
structure LanguageClasses where
  total : ℕ
  french : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking Spanish only -/
def spanishOnly (lc : LanguageClasses) : ℕ :=
  lc.total - lc.french - lc.neither + lc.both

/-- Theorem stating the number of students taking Spanish only -/
theorem spanish_only_count (lc : LanguageClasses) 
  (h1 : lc.total = 28)
  (h2 : lc.french = 5)
  (h3 : lc.both = 4)
  (h4 : lc.neither = 13) :
  spanishOnly lc = 10 := by
  sorry

#check spanish_only_count

end spanish_only_count_l178_17808


namespace problem_statement_l178_17858

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem problem_statement (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end problem_statement_l178_17858


namespace acute_angles_sum_l178_17866

theorem acute_angles_sum (a b : Real) : 
  0 < a ∧ a < π/2 →
  0 < b ∧ b < π/2 →
  4 * Real.sin a ^ 2 + 3 * Real.sin b ^ 2 = 1 →
  4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0 →
  2 * a + 3 * b = π/2 := by
sorry

end acute_angles_sum_l178_17866


namespace total_time_is_541_l178_17815

-- Define the structure for a cupcake batch
structure CupcakeBatch where
  name : String
  bakeTime : ℕ
  iceTime : ℕ
  decorateTimePerCupcake : ℕ

-- Define the number of cupcakes per batch
def cupcakesPerBatch : ℕ := 6

-- Define the batches
def chocolateBatch : CupcakeBatch := ⟨"Chocolate", 18, 25, 10⟩
def vanillaBatch : CupcakeBatch := ⟨"Vanilla", 20, 30, 15⟩
def redVelvetBatch : CupcakeBatch := ⟨"Red Velvet", 22, 28, 12⟩
def lemonBatch : CupcakeBatch := ⟨"Lemon", 24, 32, 20⟩

-- Define the list of all batches
def allBatches : List CupcakeBatch := [chocolateBatch, vanillaBatch, redVelvetBatch, lemonBatch]

-- Calculate the total time for a single batch
def batchTotalTime (batch : CupcakeBatch) : ℕ :=
  batch.bakeTime + batch.iceTime + (batch.decorateTimePerCupcake * cupcakesPerBatch)

-- Theorem: The total time to make, ice, and decorate all cupcakes is 541 minutes
theorem total_time_is_541 : (allBatches.map batchTotalTime).sum = 541 := by
  sorry

end total_time_is_541_l178_17815


namespace inequality_equivalence_l178_17860

theorem inequality_equivalence (x : ℝ) : (x - 1) / 3 > 2 ↔ x > 7 := by
  sorry

end inequality_equivalence_l178_17860


namespace log_lt_x_div_one_minus_x_l178_17812

theorem log_lt_x_div_one_minus_x (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  Real.log (1 + x) < x / (1 - x) := by
  sorry

end log_lt_x_div_one_minus_x_l178_17812


namespace always_integer_l178_17882

theorem always_integer (m : ℕ) : ∃ k : ℤ, (m : ℚ) / 3 + (m : ℚ)^2 / 2 + (m : ℚ)^3 / 6 = k := by
  sorry

end always_integer_l178_17882


namespace intersection_complement_theorem_l178_17880

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

def complement_N (S : Set ℕ) : Set ℕ := {n : ℕ | n ∉ S}

theorem intersection_complement_theorem :
  A ∩ (complement_N B) = {1} := by sorry

end intersection_complement_theorem_l178_17880


namespace algebraic_expression_value_l178_17871

theorem algebraic_expression_value (k p : ℝ) :
  (∀ x : ℝ, (6 * x + 2) * (3 - x) = -6 * x^2 + k * x + p) →
  (k - p)^2 = 100 := by
sorry

end algebraic_expression_value_l178_17871


namespace vector_operations_l178_17894

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- Theorem stating the results of the vector operations -/
theorem vector_operations :
  (3 • a + b - 2 • c = ![0, 6]) ∧
  (a = (5/9 : ℝ) • b + (8/9 : ℝ) • c) := by
  sorry

end vector_operations_l178_17894


namespace arithmetic_mean_increase_l178_17891

theorem arithmetic_mean_increase (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  let original_mean := (b₁ + b₂ + b₃ + b₄ + b₅) / 5
  let new_mean := ((b₁ + 30) + (b₂ + 30) + (b₃ + 30) + (b₄ + 30) + (b₅ + 30)) / 5
  new_mean = original_mean + 30 := by
sorry

end arithmetic_mean_increase_l178_17891


namespace fourth_root_equation_solutions_l178_17893

theorem fourth_root_equation_solutions :
  {x : ℝ | (57 - 2*x)^(1/4) + (45 + 2*x)^(1/4) = 4} = {27, -17} := by
  sorry

end fourth_root_equation_solutions_l178_17893


namespace extremum_condition_l178_17870

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- A function has an extremum if it has either a local maximum or a local minimum -/
def has_extremum (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, (∀ x : ℝ, f x ≤ f x₀) ∨ (∀ x : ℝ, f x ≥ f x₀)

/-- The necessary and sufficient condition for f(x) = ax³ + x + 1 to have an extremum -/
theorem extremum_condition (a : ℝ) :
  has_extremum (f a) ↔ a < 0 := by sorry

end extremum_condition_l178_17870


namespace f_lower_bound_solution_set_range_l178_17817

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: f(x) ≥ 1 for all x
theorem f_lower_bound : ∀ x : ℝ, f x ≥ 1 := by sorry

-- Define the set of x values that satisfy the equation
def solution_set : Set ℝ := {x | ∃ a : ℝ, f x = (a^2 + 2) / Real.sqrt (a^2 + 1)}

-- Theorem 2: The solution set is equal to (-∞, 1/2] ∪ [5/2, +∞)
theorem solution_set_range : solution_set = Set.Iic (1/2) ∪ Set.Ici (5/2) := by sorry

end f_lower_bound_solution_set_range_l178_17817


namespace recurrence_relation_expected_value_after_50_centuries_l178_17877

/-- The expected value after n centuries in the 50 Cent game -/
def expected_value (n : ℕ) : ℚ :=
  0.5 + 0.25 * n

/-- The initial amount in dollars -/
def initial_amount : ℚ := 0.5

/-- The number of centuries -/
def num_centuries : ℕ := 50

/-- The recurrence relation for the expected value -/
theorem recurrence_relation (n : ℕ) :
  expected_value (n + 1) = (expected_value n + 0.5) / 2 :=
sorry

/-- The main theorem: The expected value after 50 centuries is $13.00 -/
theorem expected_value_after_50_centuries :
  expected_value num_centuries = 13 :=
sorry

end recurrence_relation_expected_value_after_50_centuries_l178_17877


namespace basic_operation_time_scientific_notation_l178_17848

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ coefficient ∧ coefficient < 10

/-- The time taken for one basic operation in seconds -/
def basicOperationTime : ℝ := 0.000000001

/-- The scientific notation representation of the basic operation time -/
def basicOperationTimeScientific : ScientificNotation :=
  { coefficient := 1
  , exponent := -9
  , norm := by sorry }

/-- Theorem stating that the basic operation time is correctly represented in scientific notation -/
theorem basic_operation_time_scientific_notation :
  basicOperationTime = basicOperationTimeScientific.coefficient * (10 : ℝ) ^ basicOperationTimeScientific.exponent :=
by sorry

end basic_operation_time_scientific_notation_l178_17848


namespace power_equality_solution_l178_17804

theorem power_equality_solution : ∃ x : ℝ, x^5 = 5^10 ∧ x = 25 := by
  sorry

end power_equality_solution_l178_17804


namespace expression_simplification_l178_17855

theorem expression_simplification : 
  let f (x : ℤ) := x^4 + 324
  ((f 12) * (f 26) * (f 38) * (f 50) * (f 62)) / 
  ((f 6) * (f 18) * (f 30) * (f 42) * (f 54)) = 3968 / 54 := by
  sorry

end expression_simplification_l178_17855


namespace partial_fraction_sum_zero_l178_17850

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_zero_l178_17850


namespace room_population_change_l178_17876

theorem room_population_change (initial_men initial_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  ∃ (current_women : ℕ),
    initial_men + 2 = 14 ∧
    current_women = 2 * (initial_women - 3) ∧
    current_women = 24 :=
by sorry

end room_population_change_l178_17876


namespace odd_function_implies_k_equals_two_inequality_range_minimum_value_of_g_l178_17820

noncomputable section

variable (a : ℝ) (k : ℝ)

def f (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem odd_function_implies_k_equals_two
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x, f a k x = -f a k (-x)) :
  k = 2 := by sorry

theorem inequality_range
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a 2 1 < 0) :
  (∀ t, (∀ x, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ -3 < t ∧ t < 5) := by sorry

theorem minimum_value_of_g
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a 2 1 = 3/2) :
  (∃ x_min : ℝ, x_min ∈ Set.Ici 1 ∧
    ∀ x, x ∈ Set.Ici 1 →
      a^(2*x) + a^(-2*x) - 2 * f a 2 x ≥ a^(2*x_min) + a^(-2*x_min) - 2 * f a 2 x_min) ∧
  (∃ x_0 : ℝ, x_0 ∈ Set.Ici 1 ∧ a^(2*x_0) + a^(-2*x_0) - 2 * f a 2 x_0 = 5/4) := by sorry

end odd_function_implies_k_equals_two_inequality_range_minimum_value_of_g_l178_17820


namespace peanut_butter_candy_count_l178_17867

/-- The number of candy pieces in the banana jar -/
def banana_candy : ℕ := 43

/-- The number of candy pieces in the grape jar -/
def grape_candy : ℕ := banana_candy + 5

/-- The number of candy pieces in the peanut butter jar -/
def peanut_butter_candy : ℕ := 4 * grape_candy

/-- Theorem: The peanut butter jar contains 192 pieces of candy -/
theorem peanut_butter_candy_count : peanut_butter_candy = 192 := by
  sorry

end peanut_butter_candy_count_l178_17867


namespace function_identity_l178_17887

theorem function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x ≤ x) 
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end function_identity_l178_17887


namespace perfect_square_trinomial_l178_17847

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*m*x + 16 = (x - a)^2) → (m = 4 ∨ m = -4) :=
by sorry

end perfect_square_trinomial_l178_17847


namespace cos_theta_plus_pi_fourth_l178_17859

theorem cos_theta_plus_pi_fourth (θ : ℝ) (h : Real.sin (θ - π/4) = 1/5) :
  Real.cos (θ + π/4) = -1/5 := by
  sorry

end cos_theta_plus_pi_fourth_l178_17859


namespace final_number_is_two_thirds_l178_17879

def board_numbers : List ℚ := List.map (λ k => k / 2016) (List.range 2016)

def transform (a b : ℚ) : ℚ := 3 * a * b - 2 * a - 2 * b + 2

theorem final_number_is_two_thirds :
  ∃ (moves : List (ℚ × ℚ)),
    moves.length = 2015 ∧
    (moves.foldl
      (λ board (a, b) => (transform a b) :: (board.filter (λ x => x ≠ a ∧ x ≠ b)))
      board_numbers).head? = some (2/3) :=
sorry

end final_number_is_two_thirds_l178_17879


namespace coefficient_x4_eq_21_l178_17851

/-- The coefficient of x^4 in the binomial expansion of (x+1/x-1)^6 -/
def coefficient_x4 : ℕ :=
  (Nat.choose 6 0) * (Nat.choose 6 1) + (Nat.choose 6 2) * (Nat.choose 4 0)

/-- Theorem stating that the coefficient of x^4 in the binomial expansion of (x+1/x-1)^6 is 21 -/
theorem coefficient_x4_eq_21 : coefficient_x4 = 21 := by
  sorry

end coefficient_x4_eq_21_l178_17851


namespace base7_5304_equals_1866_l178_17864

def base7_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base7_5304_equals_1866 :
  base7_to_decimal [5, 3, 0, 4] = 1866 := by
  sorry

end base7_5304_equals_1866_l178_17864


namespace bella_steps_l178_17868

/-- The distance between houses in miles -/
def distance : ℝ := 3

/-- The waiting time for Ella in minutes -/
def wait_time : ℝ := 10

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 4

/-- The length of Bella's step in feet -/
def step_length : ℝ := 3

/-- The number of feet in a mile -/
def feet_per_mile : ℝ := 5280

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1328

theorem bella_steps :
  ∃ (bella_speed : ℝ),
    bella_speed > 0 ∧
    (wait_time * bella_speed + 
     (distance * feet_per_mile - wait_time * bella_speed) / (bella_speed * (1 + speed_ratio))) * 
    bella_speed / step_length = steps_taken := by
  sorry

end bella_steps_l178_17868


namespace christina_account_balance_l178_17854

def initial_balance : ℕ := 27004
def transferred_amount : ℕ := 69
def remaining_balance : ℕ := 26935

theorem christina_account_balance :
  initial_balance - transferred_amount = remaining_balance :=
by sorry

end christina_account_balance_l178_17854


namespace other_communities_count_l178_17803

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ)
  (h_total : total = 650)
  (h_muslim : muslim_percent = 44/100)
  (h_hindu : hindu_percent = 28/100)
  (h_sikh : sikh_percent = 10/100) :
  ⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total⌋ = 117 := by
  sorry

end other_communities_count_l178_17803


namespace odd_function_value_l178_17874

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = x * (x - 1)) :
  f (-3) = -6 := by
  sorry

end odd_function_value_l178_17874


namespace square_sum_value_l178_17865

theorem square_sum_value (x y : ℝ) 
  (eq1 : y + 9 = 3 * (x - 1)^2)
  (eq2 : x + 9 = 3 * (y - 1)^2)
  (neq : x ≠ y) : 
  x^2 + y^2 = 71/9 := by
sorry

end square_sum_value_l178_17865


namespace hiker_distance_l178_17873

theorem hiker_distance (s t : ℝ) 
  (h1 : (s + 1) * (2/3 * t) = s * t) 
  (h2 : (s - 1) * (t + 3) = s * t) : 
  s * t = 6 := by
  sorry

end hiker_distance_l178_17873


namespace arrangement_count_correct_l178_17856

/-- The number of ways to arrange 4 passengers in 10 seats with exactly 5 consecutive empty seats -/
def arrangement_count : ℕ := 480

/-- The number of seats in the bus station -/
def total_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct : 
  arrangement_count = 
    (Nat.factorial num_passengers) * 
    (Nat.factorial 5 / (Nat.factorial 3)) := by
  sorry

end arrangement_count_correct_l178_17856


namespace function_f_property_l178_17843

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_f_property : 
  (∀ x, f x + 2 * f (27 - x) = x) → f 11 = 7 := by sorry

end function_f_property_l178_17843


namespace positive_square_harmonic_properties_l178_17844

/-- Definition of a positive square harmonic function -/
def PositiveSquareHarmonic (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ + x₂ ∈ Set.Icc 0 1 → f x₁ + f x₂ ≤ f (x₁ + x₂))

theorem positive_square_harmonic_properties :
  ∀ f : ℝ → ℝ, PositiveSquareHarmonic f →
    (∀ x ∈ Set.Icc 0 1, f x = x^2) ∧
    (f 0 = 0) ∧
    (∀ x ∈ Set.Icc 0 1, f x ≤ 2*x) :=
by sorry

end positive_square_harmonic_properties_l178_17844


namespace unique_base7_digit_l178_17816

/-- Converts a base 7 number of the form 52x4₇ to base 10 --/
def base7ToBase10 (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

/-- Checks if a number is divisible by 19 --/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

/-- The set of valid digits in base 7 --/
def base7Digits : Set ℕ := {0, 1, 2, 3, 4, 5, 6}

theorem unique_base7_digit : 
  ∃! x : ℕ, x ∈ base7Digits ∧ isDivisibleBy19 (base7ToBase10 x) := by sorry

end unique_base7_digit_l178_17816
