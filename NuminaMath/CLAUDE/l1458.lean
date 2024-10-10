import Mathlib

namespace brittany_age_after_vacation_l1458_145827

/-- Represents a person with an age --/
structure Person where
  age : ℕ

/-- Represents a vacation --/
structure Vacation where
  duration : ℕ
  birthdaysCelebrated : ℕ
  hasLeapYear : Bool

/-- Calculates the age of a person after a vacation --/
def ageAfterVacation (person : Person) (vacation : Vacation) : ℕ :=
  person.age + vacation.birthdaysCelebrated

theorem brittany_age_after_vacation (rebecca : Person) (brittany : Person) (vacation : Vacation) :
  rebecca.age = 25 →
  brittany.age = rebecca.age + 3 →
  vacation.duration = 4 →
  vacation.birthdaysCelebrated = 3 →
  vacation.hasLeapYear = true →
  ageAfterVacation brittany vacation = 31 := by
  sorry

#eval ageAfterVacation (Person.mk 28) (Vacation.mk 4 3 true)

end brittany_age_after_vacation_l1458_145827


namespace fayes_carrots_l1458_145881

/-- Proof of the number of carrots Faye picked -/
theorem fayes_carrots (good_carrots bad_carrots moms_carrots : ℕ) 
  (h1 : good_carrots = 12)
  (h2 : bad_carrots = 16)
  (h3 : moms_carrots = 5) :
  good_carrots + bad_carrots - moms_carrots = 23 := by
  sorry

end fayes_carrots_l1458_145881


namespace lagoonIslandMales_l1458_145815

/-- Represents the population of alligators on Lagoon Island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  females : ℕ
  juvenileFemales : ℕ
  adultFemales : ℕ

/-- The conditions of the alligator population on Lagoon Island -/
def lagoonIslandConditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.females ∧
  pop.females = pop.juvenileFemales + pop.adultFemales ∧
  pop.juvenileFemales = (2 * pop.females) / 5 ∧
  pop.adultFemales = 15

theorem lagoonIslandMales (pop : AlligatorPopulation) 
  (h : lagoonIslandConditions pop) : pop.males = 25 := by
  sorry

end lagoonIslandMales_l1458_145815


namespace jenny_recycling_payment_jenny_gets_three_cents_per_can_l1458_145862

/-- Calculates the amount Jenny gets paid per can given the recycling conditions -/
theorem jenny_recycling_payment (bottle_weight : ℕ) (can_weight : ℕ) (total_weight : ℕ) 
  (num_cans : ℕ) (bottle_payment : ℕ) (total_payment : ℕ) : ℕ :=
  let remaining_weight := total_weight - (num_cans * can_weight)
  let num_bottles := remaining_weight / bottle_weight
  let bottle_total_payment := num_bottles * bottle_payment
  let can_total_payment := total_payment - bottle_total_payment
  can_total_payment / num_cans

/-- Proves that Jenny gets paid 3 cents per can under the given conditions -/
theorem jenny_gets_three_cents_per_can : 
  jenny_recycling_payment 6 2 100 20 10 160 = 3 := by
  sorry

end jenny_recycling_payment_jenny_gets_three_cents_per_can_l1458_145862


namespace binary_110_equals_6_l1458_145875

def binary_to_decimal (b₂ b₁ b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_110_equals_6 :
  binary_to_decimal 1 1 0 = 6 := by sorry

end binary_110_equals_6_l1458_145875


namespace income_difference_after_raise_l1458_145869

-- Define the annual raise percentage
def annual_raise_percent : ℚ := 8 / 100

-- Define Don's raise amount
def don_raise : ℕ := 800

-- Define Don's wife's raise amount
def wife_raise : ℕ := 840

-- Define function to calculate original salary given the raise amount
def original_salary (raise : ℕ) : ℚ := (raise : ℚ) / annual_raise_percent

-- Define function to calculate new salary after raise
def new_salary (raise : ℕ) : ℚ := original_salary raise + raise

-- Theorem statement
theorem income_difference_after_raise :
  new_salary wife_raise - new_salary don_raise = 540 := by
  sorry

end income_difference_after_raise_l1458_145869


namespace triangle_area_l1458_145879

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) :
  (1/2) * a * b = 180 := by
sorry

end triangle_area_l1458_145879


namespace nth_equation_l1458_145871

theorem nth_equation (n : ℕ) : Real.sqrt ((n + 1) * (n + 3) + 1) = n + 2 := by
  sorry

end nth_equation_l1458_145871


namespace polynomial_equality_l1458_145822

theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, p x + (x^5 + 3*x^3 + 9*x) = 7*x^3 + 24*x^2 + 25*x + 1) →
  (∀ x, p x = -x^5 + 4*x^3 + 24*x^2 + 16*x + 1) :=
by
  sorry

end polynomial_equality_l1458_145822


namespace airport_distance_l1458_145899

/-- Proves that the distance to the airport is 315 miles given the problem conditions -/
theorem airport_distance : 
  ∀ (d : ℝ) (t : ℝ),
  (d = 45 * (t + 1.5)) →  -- If continued at initial speed, arriving on time
  (d - 45 = 60 * (t - 1)) →  -- Adjusted speed for remaining journey, arriving 1 hour early
  d = 315 := by sorry

end airport_distance_l1458_145899


namespace weekend_run_ratio_l1458_145801

/-- Represents the miles run by Bill and Julia over a weekend --/
structure WeekendRun where
  billSaturday : ℝ
  billSunday : ℝ
  juliaSunday : ℝ
  m : ℝ

/-- Conditions for a valid WeekendRun --/
def ValidWeekendRun (run : WeekendRun) : Prop :=
  run.billSunday = run.billSaturday + 4 ∧
  run.juliaSunday = run.m * run.billSunday ∧
  run.billSaturday + run.billSunday + run.juliaSunday = 32

theorem weekend_run_ratio (run : WeekendRun) 
  (h : ValidWeekendRun run) :
  run.juliaSunday / run.billSunday = run.m :=
by
  sorry

#check weekend_run_ratio

end weekend_run_ratio_l1458_145801


namespace max_area_rectangular_pen_max_area_60_feet_fencing_l1458_145898

/-- The maximum area of a rectangular pen given a fixed perimeter -/
theorem max_area_rectangular_pen (perimeter : ℝ) :
  perimeter > 0 →
  ∃ (area : ℝ), area = (perimeter / 4) ^ 2 ∧
  ∀ (width height : ℝ), width > 0 → height > 0 → width * 2 + height * 2 = perimeter →
  width * height ≤ area := by
  sorry

/-- The maximum area of a rectangular pen with 60 feet of fencing is 225 square feet -/
theorem max_area_60_feet_fencing :
  ∃ (area : ℝ), area = 225 ∧
  ∀ (width height : ℝ), width > 0 → height > 0 → width * 2 + height * 2 = 60 →
  width * height ≤ area := by
  sorry

end max_area_rectangular_pen_max_area_60_feet_fencing_l1458_145898


namespace power_of_81_l1458_145805

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by
  sorry

end power_of_81_l1458_145805


namespace original_students_per_section_l1458_145885

theorem original_students_per_section 
  (S : ℕ) -- Initial number of sections
  (x : ℕ) -- Initial number of students per section
  (h1 : S + 3 = 16) -- After admission, there are S + 3 sections, totaling 16
  (h2 : S * x + 24 = 16 * 21) -- Total students after admission equals 16 sections of 21 students each
  : x = 24 := by
  sorry

end original_students_per_section_l1458_145885


namespace green_toads_per_acre_l1458_145843

/-- Given information about toads in central Texas countryside -/
structure ToadPopulation where
  /-- The ratio of green toads to brown toads -/
  green_to_brown_ratio : ℚ
  /-- The percentage of brown toads that are spotted -/
  spotted_brown_percentage : ℚ
  /-- The number of spotted brown toads per acre -/
  spotted_brown_per_acre : ℕ

/-- Theorem stating the number of green toads per acre -/
theorem green_toads_per_acre (tp : ToadPopulation)
  (h1 : tp.green_to_brown_ratio = 1 / 25)
  (h2 : tp.spotted_brown_percentage = 1 / 4)
  (h3 : tp.spotted_brown_per_acre = 50) :
  (tp.spotted_brown_per_acre : ℚ) / (tp.spotted_brown_percentage * tp.green_to_brown_ratio) = 8 := by
  sorry

end green_toads_per_acre_l1458_145843


namespace max_e_value_l1458_145884

def is_valid_number (d e : ℕ) : Prop :=
  d ≤ 9 ∧ e ≤ 9 ∧ 
  (600000 + d * 10000 + 28000 + e) % 18 = 0

theorem max_e_value :
  ∃ (d : ℕ), is_valid_number d 8 ∧
  ∀ (d' e' : ℕ), is_valid_number d' e' → e' ≤ 8 :=
sorry

end max_e_value_l1458_145884


namespace camp_recoloring_l1458_145816

/-- A graph representing friendships in a summer camp -/
structure CampGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  degree_eleven : ∀ v ∈ vertices, (edges.filter (fun e => e.1 = v ∨ e.2 = v)).card = 11
  symmetric : ∀ ⦃a b⦄, (a, b) ∈ edges → (b, a) ∈ edges

/-- A valid coloring of the graph -/
def ValidColoring (G : CampGraph) (coloring : Nat → Fin 7) : Prop :=
  ∀ ⦃a b⦄, (a, b) ∈ G.edges → coloring a ≠ coloring b

theorem camp_recoloring (G : CampGraph) (initial_coloring : Nat → Fin 7)
    (h_valid : ValidColoring G initial_coloring)
    (fixed_vertices : Finset Nat)
    (h_fixed_size : fixed_vertices.card = 100)
    (h_fixed_subset : fixed_vertices ⊆ G.vertices) :
    ∃ (new_coloring : Nat → Fin 7),
      ValidColoring G new_coloring ∧
      (∃ v ∈ G.vertices \ fixed_vertices, new_coloring v ≠ initial_coloring v) ∧
      (∀ v ∈ fixed_vertices, new_coloring v = initial_coloring v) :=
  sorry

end camp_recoloring_l1458_145816


namespace product_equals_half_l1458_145813

theorem product_equals_half : 8 * 0.25 * 2 * 0.125 = (1 : ℝ) / 2 := by
  sorry

end product_equals_half_l1458_145813


namespace subtract_negative_five_l1458_145832

theorem subtract_negative_five : 2 - (-5) = 7 := by
  sorry

end subtract_negative_five_l1458_145832


namespace tangent_line_parallel_l1458_145859

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_parallel (a : ℝ) : 
  (f_derivative a 1 = 4) → a = 1 := by
  sorry

end tangent_line_parallel_l1458_145859


namespace hemisphere_surface_area_l1458_145814

theorem hemisphere_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    π * r^2 = 3 →
    let sphere_area := 4 * π * r^2
    let hemisphere_area := sphere_area / 2 + π * r^2
    hemisphere_area = 9 := by
  sorry

end hemisphere_surface_area_l1458_145814


namespace square_diagonal_l1458_145887

theorem square_diagonal (p : ℝ) (h : p = 200 * Real.sqrt 2) :
  let s := p / 4
  s * Real.sqrt 2 = 100 := by sorry

end square_diagonal_l1458_145887


namespace temperature_data_inconsistency_l1458_145818

theorem temperature_data_inconsistency (x_bar m S_squared : ℝ) 
  (h1 : x_bar = 0)
  (h2 : m = 4)
  (h3 : S_squared = 15.917)
  : |x_bar - m| > Real.sqrt S_squared := by
  sorry

end temperature_data_inconsistency_l1458_145818


namespace count_numbers_greater_than_threshold_l1458_145868

def numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
def threshold : ℚ := 1.1

theorem count_numbers_greater_than_threshold : 
  (numbers.filter (λ x => x > threshold)).length = 3 := by
  sorry

end count_numbers_greater_than_threshold_l1458_145868


namespace race_time_difference_l1458_145833

/-- Represents the race scenario with Malcolm and Joshua -/
structure RaceScenario where
  malcolm_speed : ℝ  -- Malcolm's speed in minutes per mile
  joshua_speed : ℝ   -- Joshua's speed in minutes per mile
  race_distance : ℝ  -- Race distance in miles

/-- Calculates the time difference between Malcolm and Joshua finishing the race -/
def time_difference (scenario : RaceScenario) : ℝ :=
  scenario.joshua_speed * scenario.race_distance - scenario.malcolm_speed * scenario.race_distance

/-- Theorem stating the time difference for the given race scenario -/
theorem race_time_difference (scenario : RaceScenario) 
  (h1 : scenario.malcolm_speed = 5)
  (h2 : scenario.joshua_speed = 7)
  (h3 : scenario.race_distance = 12) :
  time_difference scenario = 24 := by
  sorry

#eval time_difference { malcolm_speed := 5, joshua_speed := 7, race_distance := 12 }

end race_time_difference_l1458_145833


namespace sufficient_but_not_necessary_condition_l1458_145894

/-- A quadratic equation ax² + bx + c = 0 -/
structure QuadraticEq where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic equation -/
def discriminant (q : QuadraticEq) : ℝ := q.b^2 - 4*q.a*q.c

/-- A quadratic equation has real roots iff its discriminant is non-negative -/
def has_real_roots (q : QuadraticEq) : Prop := discriminant q ≥ 0

theorem sufficient_but_not_necessary_condition 
  (q1 q2 : QuadraticEq) 
  (h1 : has_real_roots q1)
  (h2 : has_real_roots q2)
  (h3 : q1.a ≠ q2.a) :
  (∀ w c, w * c > 0 → 
    has_real_roots ⟨q2.a, q1.b, q1.c⟩ ∨ has_real_roots ⟨q1.a, q2.b, q2.c⟩) ∧ 
  (∃ w c, w * c ≤ 0 ∧ 
    (has_real_roots ⟨q2.a, q1.b, q1.c⟩ ∨ has_real_roots ⟨q1.a, q2.b, q2.c⟩)) :=
sorry

end sufficient_but_not_necessary_condition_l1458_145894


namespace anna_coins_l1458_145841

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (twenty_cent : ℕ) : ℕ :=
  59 - 3 * five_cent

theorem anna_coins :
  ∀ (five_cent twenty_cent : ℕ),
    five_cent + twenty_cent = 15 →
    different_values five_cent twenty_cent = 24 →
    twenty_cent = 4 := by
  sorry

end anna_coins_l1458_145841


namespace gauss_1998_cycle_l1458_145831

def word_length : Nat := 5
def number_length : Nat := 4

theorem gauss_1998_cycle : Nat.lcm word_length number_length = 20 := by
  sorry

end gauss_1998_cycle_l1458_145831


namespace max_true_statements_l1458_145836

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^3 ∧ x^3 < 1),
    (x^3 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - x^3 ∧ x - x^3 < 1),
    (x^3 - x > 1)
  ]
  ∃ (true_statements : List Bool), 
    (∀ i, true_statements.get! i = true → statements.get! i) ∧
    true_statements.count true ≤ 3 ∧
    ∀ (other_true_statements : List Bool),
      (∀ i, other_true_statements.get! i = true → statements.get! i) →
      other_true_statements.count true ≤ true_statements.count true :=
by sorry


end max_true_statements_l1458_145836


namespace tree_height_ratio_l1458_145861

/-- Given three trees with specific height relationships, prove that the height of the smallest tree
    is 1/4 of the height of the middle-sized tree. -/
theorem tree_height_ratio :
  ∀ (h₁ h₂ h₃ : ℝ),
  h₁ = 108 →
  h₂ = h₁ / 2 - 6 →
  h₃ = 12 →
  h₃ / h₂ = 1 / 4 := by
sorry

end tree_height_ratio_l1458_145861


namespace necessary_but_not_sufficient_l1458_145889

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧
  ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by sorry

end necessary_but_not_sufficient_l1458_145889


namespace circle_radius_is_13_main_result_l1458_145844

/-- Represents a circle with tangents -/
structure CircleWithTangents where
  r : ℝ  -- radius of the circle
  ab : ℝ  -- length of tangent AB
  ac : ℝ  -- length of tangent AC
  de : ℝ  -- length of tangent DE perpendicular to BC

/-- Theorem: Given the conditions, the radius of the circle is 13 -/
theorem circle_radius_is_13 (c : CircleWithTangents) 
  (h1 : c.ab = 5) 
  (h2 : c.ac = 12) 
  (h3 : c.de = 13) : 
  c.r = 13 := by
  sorry

/-- The main result -/
theorem main_result : ∃ c : CircleWithTangents, 
  c.ab = 5 ∧ c.ac = 12 ∧ c.de = 13 ∧ c.r = 13 := by
  sorry

end circle_radius_is_13_main_result_l1458_145844


namespace specific_frustum_volume_l1458_145804

/-- A frustum with given base areas and lateral surface area -/
structure Frustum where
  upper_base_area : ℝ
  lower_base_area : ℝ
  lateral_surface_area : ℝ

/-- The volume of a frustum -/
def volume (f : Frustum) : ℝ := sorry

/-- Theorem stating the volume of the specific frustum -/
theorem specific_frustum_volume :
  ∃ (f : Frustum),
    f.upper_base_area = π ∧
    f.lower_base_area = 4 * π ∧
    f.lateral_surface_area = 6 * π ∧
    volume f = (7 * Real.sqrt 3 / 3) * π := by sorry

end specific_frustum_volume_l1458_145804


namespace largest_prime_factor_is_101_l1458_145877

/-- A sequence of four-digit integers with a cyclic digit property -/
def CyclicSequence := List Nat

/-- The sum of all terms in a cyclic sequence -/
def sequenceSum (seq : CyclicSequence) : Nat :=
  seq.sum

/-- Predicate to check if a sequence satisfies the cyclic digit property -/
def hasCyclicDigitProperty (seq : CyclicSequence) : Prop :=
  sorry -- Definition of the cyclic digit property

/-- The largest prime factor that always divides the sum of a cyclic sequence -/
def largestPrimeFactor (seq : CyclicSequence) : Nat :=
  sorry -- Definition to find the largest prime factor

theorem largest_prime_factor_is_101 (seq : CyclicSequence) 
    (h : hasCyclicDigitProperty seq) :
    largestPrimeFactor seq = 101 := by
  sorry

#check largest_prime_factor_is_101

end largest_prime_factor_is_101_l1458_145877


namespace basketball_game_price_l1458_145882

/-- The cost of Joan's video game purchase -/
def total_cost : ℝ := 9.43

/-- The cost of the racing game -/
def racing_game_cost : ℝ := 4.23

/-- The cost of the basketball game -/
def basketball_game_cost : ℝ := total_cost - racing_game_cost

theorem basketball_game_price : basketball_game_cost = 5.20 := by
  sorry

end basketball_game_price_l1458_145882


namespace device_working_prob_correct_l1458_145863

/-- A device with two components, each having a probability of failure --/
structure Device where
  /-- The probability of a single component being damaged --/
  component_failure_prob : ℝ
  /-- Assumption that the component failure probability is between 0 and 1 --/
  h_prob_range : 0 ≤ component_failure_prob ∧ component_failure_prob ≤ 1

/-- The probability of the device working --/
def device_working_prob (d : Device) : ℝ :=
  (1 - d.component_failure_prob) * (1 - d.component_failure_prob)

/-- Theorem stating that for a device with component failure probability of 0.1,
    the probability of the device working is 0.81 --/
theorem device_working_prob_correct (d : Device) 
    (h : d.component_failure_prob = 0.1) : 
    device_working_prob d = 0.81 := by
  sorry

end device_working_prob_correct_l1458_145863


namespace jamies_liquid_limit_l1458_145809

/-- Jamie's liquid consumption limit problem -/
theorem jamies_liquid_limit :
  let cup_oz : ℕ := 8  -- A cup is 8 ounces
  let pint_oz : ℕ := 16  -- A pint is 16 ounces
  let milk_consumed : ℕ := cup_oz  -- Jamie had a cup of milk
  let juice_consumed : ℕ := pint_oz  -- Jamie had a pint of grape juice
  let water_limit : ℕ := 8  -- Jamie can drink 8 more ounces before needing the bathroom
  milk_consumed + juice_consumed + water_limit = 32  -- Jamie's total liquid limit
  := by sorry

end jamies_liquid_limit_l1458_145809


namespace rational_function_identity_l1458_145880

theorem rational_function_identity (x : ℝ) (h1 : x ≠ 2) (h2 : x^2 + x + 1 ≠ 0) :
  (x + 3)^2 / ((x - 2) * (x^2 + x + 1)) = 
  25 / (7 * (x - 2)) + (-18 * x - 19) / (7 * (x^2 + x + 1)) := by
  sorry

#check rational_function_identity

end rational_function_identity_l1458_145880


namespace first_bus_students_l1458_145852

theorem first_bus_students (total_buses : ℕ) (initial_avg : ℕ) (remaining_avg : ℕ) : 
  total_buses = 6 → 
  initial_avg = 28 → 
  remaining_avg = 26 → 
  (total_buses * initial_avg - (total_buses - 1) * remaining_avg) = 38 := by
sorry

end first_bus_students_l1458_145852


namespace conic_focal_distance_l1458_145835

/-- The focal distance of a conic curve x^2 + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_focal_distance (m : ℝ) : 
  (m^2 = 2 * 8) →  -- m is the geometric mean between 2 and 8
  let focal_distance := 
    if m > 0 then 2 * Real.sqrt 3  -- Ellipse case
    else 2 * Real.sqrt 5           -- Hyperbola case
  (∃ (x y : ℝ), x^2 + y^2/m = 1) →  -- The conic curve exists
  focal_distance = 2 * Real.sqrt 3 ∨ focal_distance = 2 * Real.sqrt 5 :=
by sorry

end conic_focal_distance_l1458_145835


namespace squirrel_acorns_l1458_145823

/-- Given 5 squirrels collecting 575 acorns in total, and each squirrel needing 130 acorns for winter,
    the number of additional acorns each squirrel needs to collect is 15. -/
theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (acorns_needed : ℕ) :
  num_squirrels = 5 →
  total_acorns = 575 →
  acorns_needed = 130 →
  acorns_needed - (total_acorns / num_squirrels) = 15 :=
by
  sorry


end squirrel_acorns_l1458_145823


namespace smallest_x_value_l1458_145829

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (210 + x)) : 
  2 ≤ x.val :=
sorry

end smallest_x_value_l1458_145829


namespace z_value_proof_l1458_145845

theorem z_value_proof : 
  ∃ z : ℝ, (12 / 20 = (z / 20) ^ (1/3)) ∧ z = 4.32 :=
by
  sorry

end z_value_proof_l1458_145845


namespace symmetric_point_y_axis_l1458_145891

/-- Given a point M(-5, 2), its symmetric point with respect to the y-axis has coordinates (5, 2) -/
theorem symmetric_point_y_axis :
  let M : ℝ × ℝ := (-5, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
  symmetric_point M = (5, 2) := by
sorry

end symmetric_point_y_axis_l1458_145891


namespace correct_calculation_l1458_145895

/-- Represents the loan and investment scenario -/
structure LoanInvestment where
  loan_amount : ℝ
  interest_paid : ℝ
  business_profit_rate : ℝ
  total_profit : ℝ

/-- Calculates the interest rate and investment amount -/
def calculate_rate_and_investment (scenario : LoanInvestment) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correctness of the calculation -/
theorem correct_calculation (scenario : LoanInvestment) :
  scenario.loan_amount = 150000 ∧
  scenario.interest_paid = 42000 ∧
  scenario.business_profit_rate = 0.1 ∧
  scenario.total_profit = 25000 →
  let (rate, investment) := calculate_rate_and_investment scenario
  rate = 0.05 ∧ investment = 50000 :=
by sorry

end correct_calculation_l1458_145895


namespace permutations_count_l1458_145847

theorem permutations_count (n : ℕ) : Nat.factorial n = 6227020800 → n = 13 := by
  sorry

end permutations_count_l1458_145847


namespace right_triangle_abc_area_l1458_145892

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangleABC where
  -- Point A
  a : ℝ × ℝ
  -- Point B
  b : ℝ × ℝ
  -- Point C (right angle)
  c : ℝ × ℝ
  -- Hypotenuse length
  ab_length : ℝ
  -- Median through A equation
  median_a_slope : ℝ
  median_a_intercept : ℝ
  -- Median through B equation
  median_b_slope : ℝ
  median_b_intercept : ℝ
  -- Conditions
  right_angle_at_c : (a.1 - c.1) * (b.1 - c.1) + (a.2 - c.2) * (b.2 - c.2) = 0
  hypotenuse_length : (a.1 - b.1)^2 + (a.2 - b.2)^2 = ab_length^2
  median_a_equation : ∀ x y, y = median_a_slope * x + median_a_intercept → 
    2 * x = a.1 + c.1 ∧ 2 * y = a.2 + c.2
  median_b_equation : ∀ x y, y = median_b_slope * x + median_b_intercept → 
    2 * x = b.1 + c.1 ∧ 2 * y = b.2 + c.2

/-- The area of the right triangle ABC with given properties is 175 -/
theorem right_triangle_abc_area 
  (t : RightTriangleABC) 
  (h1 : t.ab_length = 50) 
  (h2 : t.median_a_slope = 1 ∧ t.median_a_intercept = 5)
  (h3 : t.median_b_slope = 2 ∧ t.median_b_intercept = 6) :
  abs ((t.a.1 * t.b.2 - t.b.1 * t.a.2) / 2) = 175 := by
  sorry

end right_triangle_abc_area_l1458_145892


namespace inequality_system_solution_l1458_145803

theorem inequality_system_solution (b : ℝ) : 
  (∀ (x y : ℝ), 2*b * Real.cos (2*(x-y)) + 8*b^2 * Real.cos (x-y) + 8*b^2*(b+1) + 5*b < 0 ∧
                 x^2 + y^2 + 1 > 2*b*x + 2*y + b - b^2) ↔ 
  (b < -1 - Real.sqrt 2 / 4 ∨ (-1/2 < b ∧ b < 0)) := by sorry

end inequality_system_solution_l1458_145803


namespace parallel_lines_sum_l1458_145840

/-- Two parallel lines with a given distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  h_m_pos : m > 0
  h_parallel : 1 / 2 = -2 / n
  h_distance : |m + 3| / Real.sqrt 5 = Real.sqrt 5

/-- The sum of m and n for the parallel lines is -2 -/
theorem parallel_lines_sum (l : ParallelLines) : l.m + l.n = -2 := by
  sorry

end parallel_lines_sum_l1458_145840


namespace distance_traveled_l1458_145874

/-- Given a person traveling at 6 km/h for 10 minutes, prove that the distance traveled is 1000 meters. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 6 → time = 1/6 → speed * time * 1000 = 1000 := by
  sorry

end distance_traveled_l1458_145874


namespace f_composition_equals_pi_plus_one_l1458_145838

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_plus_one :
  f (f (f (-1))) = Real.pi + 1 := by
  sorry

end f_composition_equals_pi_plus_one_l1458_145838


namespace arithmetic_sequence_sum_l1458_145839

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 1 = 3 →
  a 100 = 36 →
  a 3 + a 98 = 39 := by
sorry

end arithmetic_sequence_sum_l1458_145839


namespace flag_distribution_l1458_145806

theorem flag_distribution (total_flags : ℕ) (blue_percent red_percent : ℚ) :
  total_flags % 2 = 0 ∧
  blue_percent = 60 / 100 ∧
  red_percent = 45 / 100 ∧
  blue_percent + red_percent > 1 →
  blue_percent + red_percent - 1 = 5 / 100 :=
by sorry

end flag_distribution_l1458_145806


namespace fraction_equality_implies_power_equality_l1458_145872

theorem fraction_equality_implies_power_equality
  (a b c : ℝ) (k : ℕ) 
  (h_odd : Odd k)
  (h_eq : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  1/a^k + 1/b^k + 1/c^k = 1/(a^k + b^k + c^k) :=
by sorry

end fraction_equality_implies_power_equality_l1458_145872


namespace scaling_transformation_l1458_145866

-- Define the original circle equation
def original_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scale_x (x : ℝ) : ℝ := 5 * x
def scale_y (y : ℝ) : ℝ := 3 * y

-- State the theorem
theorem scaling_transformation :
  ∀ x' y' : ℝ, (∃ x y : ℝ, original_equation x y ∧ x' = scale_x x ∧ y' = scale_y y) →
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by sorry

end scaling_transformation_l1458_145866


namespace mans_upstream_rate_l1458_145834

/-- Prove that given a man's downstream rate, his rate in still water, and the current rate, his upstream rate can be calculated. -/
theorem mans_upstream_rate (downstream_rate still_water_rate current_rate : ℝ) 
  (h1 : downstream_rate = 32)
  (h2 : still_water_rate = 24.5)
  (h3 : current_rate = 7.5) :
  still_water_rate - current_rate = 17 := by
  sorry

end mans_upstream_rate_l1458_145834


namespace base5_500_l1458_145878

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 5 * acc + d) 0

theorem base5_500 : toBase5 500 = [4, 0, 0, 0] :=
sorry

end base5_500_l1458_145878


namespace book_area_calculation_l1458_145864

/-- Calculates the area of a book given its length expression, width, and conversion factor. -/
theorem book_area_calculation (x : ℝ) (inch_to_cm : ℝ) : 
  x = 5 → 
  inch_to_cm = 2.54 → 
  (3 * x - 4) * ((5 / 2) * inch_to_cm) = 69.85 := by
  sorry

end book_area_calculation_l1458_145864


namespace expand_expression_l1458_145888

theorem expand_expression (x y : ℝ) : 24 * (3 * x + 4 * y - 2) = 72 * x + 96 * y - 48 := by
  sorry

end expand_expression_l1458_145888


namespace series_sum_l1458_145857

/-- The sum of the series $\sum_{n=1}^{\infty} \frac{3^n}{9^n - 1}$ is equal to $\frac{1}{2}$ -/
theorem series_sum : ∑' n, (3^n : ℝ) / (9^n - 1) = 1/2 := by sorry

end series_sum_l1458_145857


namespace hash_3_7_l1458_145873

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem
theorem hash_3_7 : hash 3 7 = 63 := by
  sorry

end hash_3_7_l1458_145873


namespace one_contribution_before_john_l1458_145848

-- Define the problem parameters
def john_donation : ℝ := 100
def new_average : ℝ := 75
def increase_percentage : ℝ := 0.5

-- Define the theorem
theorem one_contribution_before_john
  (n : ℝ) -- Initial number of contributions
  (A : ℝ) -- Initial average contribution
  (h1 : A + increase_percentage * A = new_average) -- New average is 50% higher
  (h2 : n * A + john_donation = (n + 1) * new_average) -- Total amount equality
  : n = 1 := by
  sorry


end one_contribution_before_john_l1458_145848


namespace gain_percent_calculation_l1458_145855

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 20)
  (h2 : selling_price = 35) :
  (selling_price - cost_price) / cost_price * 100 = 75 := by
  sorry

end gain_percent_calculation_l1458_145855


namespace min_value_reciprocal_sum_l1458_145853

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ) = Real.sqrt (3^a * 3^b) → 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 2 :=
by sorry

end min_value_reciprocal_sum_l1458_145853


namespace area_KLMN_value_l1458_145808

/-- Triangle ABC with points K, L, N, and M -/
structure TriangleABC where
  -- Define the sides of the triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Define the positions of points K, L, and N
  AK : ℝ
  AL : ℝ
  BN : ℝ
  -- Ensure the triangle satisfies the given conditions
  h_AB : AB = 14
  h_BC : BC = 13
  h_AC : AC = 15
  h_AK : AK = 15/14
  h_AL : AL = 1
  h_BN : BN = 9

/-- The area of quadrilateral KLMN in the given triangle -/
def areaKLMN (t : TriangleABC) : ℝ := sorry

/-- Theorem stating that the area of KLMN is 36503/1183 -/
theorem area_KLMN_value (t : TriangleABC) : areaKLMN t = 36503/1183 := by sorry

end area_KLMN_value_l1458_145808


namespace cubic_roots_sum_cubes_l1458_145800

theorem cubic_roots_sum_cubes (p q r : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  p^3 - p^2 + p - 2 = 0 →
  q^3 - q^2 + q - 2 = 0 →
  r^3 - r^2 + r - 2 = 0 →
  p^3 + q^3 + r^3 = -6 := by
sorry

end cubic_roots_sum_cubes_l1458_145800


namespace sine_function_parameters_l1458_145886

theorem sine_function_parameters (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c < 0) :
  (∀ x, a * Real.sin (b * x) + c ≤ 3) ∧
  (∃ x, a * Real.sin (b * x) + c = 3) ∧
  (∀ x, a * Real.sin (b * x) + c ≥ -5) ∧
  (∃ x, a * Real.sin (b * x) + c = -5) →
  a = 4 ∧ c = -1 := by
sorry

end sine_function_parameters_l1458_145886


namespace unique_solution_l1458_145826

/-- Represents a three-digit number formed by digits U, H, and A -/
def three_digit_number (U H A : Nat) : Nat := 100 * U + 10 * H + A

/-- Represents a two-digit number formed by digits U and H -/
def two_digit_number (U H : Nat) : Nat := 10 * U + H

/-- Checks if a number is a valid digit (0-9) -/
def is_digit (n : Nat) : Prop := n ≤ 9

/-- Checks if three numbers are distinct -/
def are_distinct (a b c : Nat) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem stating the unique solution to the puzzle -/
theorem unique_solution :
  ∃! (U H A : Nat),
    is_digit U ∧ is_digit H ∧ is_digit A ∧
    are_distinct U H A ∧
    U ≠ 0 ∧
    three_digit_number U H A = Nat.lcm (two_digit_number U H) (Nat.lcm (two_digit_number U A) (two_digit_number H A)) ∧
    U = 1 ∧ H = 5 ∧ A = 0 := by
  sorry

end unique_solution_l1458_145826


namespace cubic_roots_classification_l1458_145849

/-- The discriminant of the cubic equation x³ + px + q = 0 -/
def discriminant (p q : ℝ) : ℝ := 4 * p^3 + 27 * q^2

/-- Theorem about the nature of roots for the cubic equation x³ + px + q = 0 -/
theorem cubic_roots_classification (p q : ℝ) :
  (discriminant p q > 0 → ∃ (x : ℂ), x^3 + p*x + q = 0 ∧ (∀ y : ℂ, y^3 + p*y + q = 0 → y = x ∨ y.im ≠ 0)) ∧
  (discriminant p q < 0 → ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^3 + p*x + q = 0 ∧ y^3 + p*y + q = 0 ∧ z^3 + p*z + q = 0) ∧
  (discriminant p q = 0 ∧ p = 0 ∧ q = 0 → ∃ (x : ℝ), ∀ y : ℝ, y^3 + p*y + q = 0 → y = x) ∧
  (discriminant p q = 0 ∧ (p ≠ 0 ∨ q ≠ 0) → ∃ (x y : ℝ), x ≠ y ∧ x^3 + p*x + q = 0 ∧ y^3 + p*y + q = 0 ∧ 
    ∀ z : ℝ, z^3 + p*z + q = 0 → z = x ∨ z = y) :=
by sorry

end cubic_roots_classification_l1458_145849


namespace range_of_m_l1458_145860

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - 1 < m^2 - 3*m) → m < 1 ∨ m > 2 := by
  sorry

end range_of_m_l1458_145860


namespace neighborhood_cable_cost_l1458_145837

/-- Calculate the total cost of cable for a neighborhood given the following conditions:
- 18 east-west streets, each 2 miles long
- 10 north-south streets, each 4 miles long
- 5 miles of cable needed to electrify 1 mile of street
- Cable costs $2000 per mile
-/
theorem neighborhood_cable_cost :
  let east_west_streets := 18
  let east_west_length := 2
  let north_south_streets := 10
  let north_south_length := 4
  let cable_per_mile := 5
  let cost_per_mile := 2000
  let total_street_length := east_west_streets * east_west_length + north_south_streets * north_south_length
  let total_cable_length := total_street_length * cable_per_mile
  let total_cost := total_cable_length * cost_per_mile
  total_cost = 760000 := by
  sorry

end neighborhood_cable_cost_l1458_145837


namespace power_function_m_value_l1458_145896

/-- A function f(x) is a power function if it can be written in the form f(x) = ax^n, where a and n are constants and a ≠ 0. -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x ^ n

/-- Given that y = (m^2 - 3)x^(2m) is a power function, m equals ±2. -/
theorem power_function_m_value (m : ℝ) :
  IsPowerFunction (fun x => (m^2 - 3) * x^(2*m)) → m = 2 ∨ m = -2 := by
  sorry

end power_function_m_value_l1458_145896


namespace total_lives_for_eight_friends_l1458_145851

/-- Calculates the total number of lives for a group of friends in a video game -/
def totalLives (numFriends : ℕ) (livesPerFriend : ℕ) : ℕ :=
  numFriends * livesPerFriend

/-- Proves that the total number of lives for 8 friends with 8 lives each is 64 -/
theorem total_lives_for_eight_friends : totalLives 8 8 = 64 := by
  sorry

end total_lives_for_eight_friends_l1458_145851


namespace vector_equality_implies_m_equals_two_l1458_145883

def a (m : ℝ) : ℝ × ℝ := (m, -2)
def b : ℝ × ℝ := (1, 1)

theorem vector_equality_implies_m_equals_two (m : ℝ) :
  ‖a m - b‖ = ‖a m + b‖ → m = 2 := by sorry

end vector_equality_implies_m_equals_two_l1458_145883


namespace nested_square_roots_equality_l1458_145876

theorem nested_square_roots_equality : Real.sqrt (36 * Real.sqrt (27 * Real.sqrt 9)) = 18 := by
  sorry

end nested_square_roots_equality_l1458_145876


namespace broccoli_sales_amount_l1458_145802

def farmers_market_sales (broccoli_sales : ℝ) : Prop :=
  let carrot_sales := 2 * broccoli_sales
  let spinach_sales := carrot_sales / 2 + 16
  let cauliflower_sales := 136
  broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales = 380

theorem broccoli_sales_amount : ∃ (x : ℝ), farmers_market_sales x ∧ x = 57 :=
  sorry

end broccoli_sales_amount_l1458_145802


namespace trampoline_jumps_l1458_145850

theorem trampoline_jumps (ronald_jumps rupert_jumps : ℕ) : 
  ronald_jumps = 157 →
  rupert_jumps > ronald_jumps →
  rupert_jumps + ronald_jumps = 400 →
  rupert_jumps - ronald_jumps = 86 := by
sorry

end trampoline_jumps_l1458_145850


namespace point_in_second_quadrant_l1458_145897

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -2
  let y : ℝ := 3
  second_quadrant x y :=
by sorry

end point_in_second_quadrant_l1458_145897


namespace regular_polygon_150_degree_angles_l1458_145890

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides -/
theorem regular_polygon_150_degree_angles (n : ℕ) : 
  (n ≥ 3) →                          -- A polygon has at least 3 sides
  (∀ i : ℕ, i < n → 150 = (n - 2) * 180 / n) →  -- Each interior angle is 150 degrees
  n = 12 := by
sorry

end regular_polygon_150_degree_angles_l1458_145890


namespace diamond_operation_result_l1458_145828

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the diamond operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.three
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four

theorem diamond_operation_result :
  diamond (diamond Element.three Element.four) (diamond Element.two Element.one) = Element.two := by
  sorry

end diamond_operation_result_l1458_145828


namespace rationalize_denominator_cube_root_l1458_145842

theorem rationalize_denominator_cube_root :
  ∃ (A B C : ℕ), 
    (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
    (∀ p : ℕ, Prime p → ¬(p^3 ∣ B)) ∧
    (5 / (3 * Real.rpow 7 (1/3)) = (A * Real.rpow B (1/3)) / C) ∧
    (A + B + C = 75) := by
  sorry

end rationalize_denominator_cube_root_l1458_145842


namespace decimal_to_fraction_l1458_145811

theorem decimal_to_fraction : (3.375 : ℚ) = 27 / 8 := by sorry

end decimal_to_fraction_l1458_145811


namespace jason_pokemon_cards_l1458_145870

theorem jason_pokemon_cards 
  (cards_given_away : ℕ) 
  (cards_remaining : ℕ) 
  (h1 : cards_given_away = 9) 
  (h2 : cards_remaining = 4) : 
  cards_given_away + cards_remaining = 13 := by
sorry

end jason_pokemon_cards_l1458_145870


namespace symmetric_point_example_l1458_145807

/-- Given a point (x, y) in the plane, the point symmetric to it with respect to the x-axis is (x, -y) -/
def symmetric_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The coordinates of the point symmetric to (3, 8) with respect to the x-axis are (3, -8) -/
theorem symmetric_point_example : symmetric_point_x_axis (3, 8) = (3, -8) := by
  sorry

end symmetric_point_example_l1458_145807


namespace inequality_proof_l1458_145846

theorem inequality_proof (x : ℝ) : (x - 4) / 2 - (x - 1) / 4 < 1 → x < 11 := by
  sorry

end inequality_proof_l1458_145846


namespace cyclic_iff_perpendicular_diagonals_l1458_145824

-- Define the basic geometric objects
variable (A B C D P Q R S : Point)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the incircle and its tangency points
def has_incircle_with_tangent_points (A B C D P Q R S : Point) : Prop := sorry

-- Define cyclic quadrilateral
def is_cyclic (A B C D : Point) : Prop := sorry

-- Define perpendicularity
def perpendicular (P Q R S : Point) : Prop := sorry

-- The main theorem
theorem cyclic_iff_perpendicular_diagonals 
  (h_quad : is_quadrilateral A B C D)
  (h_incircle : has_incircle_with_tangent_points A B C D P Q R S) :
  is_cyclic A B C D ↔ perpendicular P R Q S := by sorry

end cyclic_iff_perpendicular_diagonals_l1458_145824


namespace students_in_jungkooks_class_l1458_145825

theorem students_in_jungkooks_class :
  let glasses_wearers : Nat := 9
  let non_glasses_wearers : Nat := 16
  glasses_wearers + non_glasses_wearers = 25 :=
by sorry

end students_in_jungkooks_class_l1458_145825


namespace probability_perfect_square_sum_l1458_145819

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def dice_sum_outcomes : ℕ := 64

def perfect_square_sums : List ℕ := [4, 9, 16]

def ways_to_get_sum (sum : ℕ) : ℕ :=
  if sum = 4 then 3
  else if sum = 9 then 8
  else if sum = 16 then 1
  else 0

def total_favorable_outcomes : ℕ :=
  (perfect_square_sums.map ways_to_get_sum).sum

theorem probability_perfect_square_sum :
  (total_favorable_outcomes : ℚ) / dice_sum_outcomes = 3 / 16 := by
  sorry

end probability_perfect_square_sum_l1458_145819


namespace triangle_side_range_l1458_145817

theorem triangle_side_range (a : ℝ) : 
  let AB := (5 : ℝ)
  let BC := 2 * a + 1
  let AC := (12 : ℝ)
  (AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB) → (3 < a ∧ a < 8) :=
by sorry

end triangle_side_range_l1458_145817


namespace inequality_solution_l1458_145854

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end inequality_solution_l1458_145854


namespace container_weights_l1458_145856

theorem container_weights (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (w1 : x + y = 110) (w2 : y + z = 130) (w3 : z + x = 150) :
  x + y + z = 195 := by
  sorry

end container_weights_l1458_145856


namespace first_division_percentage_l1458_145812

theorem first_division_percentage (total_students : ℕ) 
  (second_division_percentage : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percentage = 54 / 100 →
  just_passed = 63 →
  (total_students : ℚ) * (25 / 100) = 
    total_students - (total_students : ℚ) * second_division_percentage - just_passed :=
by
  sorry

end first_division_percentage_l1458_145812


namespace simplify_fraction_x_squared_minus_y_squared_l1458_145867

-- Part 1
theorem simplify_fraction (a : ℝ) (h : a > 0) : 
  1 / (Real.sqrt a + 1) = (Real.sqrt a - 1) / 2 :=
sorry

-- Part 2
theorem x_squared_minus_y_squared (x y : ℝ) 
  (hx : x = 1 / (2 + Real.sqrt 3)) 
  (hy : y = 1 / (2 - Real.sqrt 3)) : 
  x^2 - y^2 = -8 * Real.sqrt 3 :=
sorry

end simplify_fraction_x_squared_minus_y_squared_l1458_145867


namespace largest_difference_l1458_145858

theorem largest_difference (U V W X Y Z : ℕ) 
  (hU : U = 2 * 1002^1003)
  (hV : V = 1002^1003)
  (hW : W = 1001 * 1002^1002)
  (hX : X = 2 * 1002^1002)
  (hY : Y = 1002^1002)
  (hZ : Z = 1002^1001) :
  (U - V > V - W) ∧ 
  (U - V > W - X) ∧ 
  (U - V > X - Y) ∧ 
  (U - V > Y - Z) :=
sorry

end largest_difference_l1458_145858


namespace highest_demand_week_sales_total_sales_check_l1458_145830

-- Define the sales for each week
def first_week_sales : ℕ := 20
def second_week_sales : ℕ := 3 * first_week_sales
def third_week_sales : ℕ := 2 * first_week_sales
def fourth_week_sales : ℕ := first_week_sales

-- Define the total sales for the month
def total_sales : ℕ := 300

-- Theorem to prove the highest demand week
theorem highest_demand_week_sales :
  max first_week_sales (max second_week_sales (max third_week_sales fourth_week_sales)) = 60 :=
by sorry

-- Verify that the sum of all weeks' sales equals the total monthly sales
theorem total_sales_check :
  first_week_sales + second_week_sales + third_week_sales + fourth_week_sales = total_sales :=
by sorry

end highest_demand_week_sales_total_sales_check_l1458_145830


namespace rational_inequality_l1458_145820

theorem rational_inequality (a b : ℚ) 
  (h1 : |a| < |b|) 
  (h2 : a > 0) 
  (h3 : b < 0) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end rational_inequality_l1458_145820


namespace fraction_simplification_l1458_145893

theorem fraction_simplification : (210 : ℚ) / 21 * 7 / 98 * 6 / 4 = 15 / 14 := by
  sorry

end fraction_simplification_l1458_145893


namespace cost_of_leftover_drinks_l1458_145810

theorem cost_of_leftover_drinks : 
  let soda_bought := 30
  let soda_price := 2
  let energy_bought := 20
  let energy_price := 3
  let smoothie_bought := 15
  let smoothie_price := 4
  let soda_consumed := 10
  let energy_consumed := 14
  let smoothie_consumed := 5
  
  let soda_leftover := soda_bought - soda_consumed
  let energy_leftover := energy_bought - energy_consumed
  let smoothie_leftover := smoothie_bought - smoothie_consumed
  
  let leftover_cost := soda_leftover * soda_price + 
                       energy_leftover * energy_price + 
                       smoothie_leftover * smoothie_price
  
  leftover_cost = 98 := by sorry

end cost_of_leftover_drinks_l1458_145810


namespace jillian_apartment_size_l1458_145821

/-- The cost per square foot of apartment rentals in Rivertown -/
def cost_per_sqft : ℚ := 1.20

/-- Jillian's maximum monthly budget for rent -/
def max_budget : ℚ := 720

/-- The largest apartment size Jillian should consider -/
def largest_apartment_size : ℚ := max_budget / cost_per_sqft

theorem jillian_apartment_size :
  largest_apartment_size = 600 :=
by sorry

end jillian_apartment_size_l1458_145821


namespace mothers_house_distance_l1458_145865

/-- The distance between your house and your mother's house -/
def total_distance : ℝ := 234.0

/-- The distance you have traveled so far -/
def traveled_distance : ℝ := 156.0

/-- Theorem stating that the total distance to your mother's house is 234.0 miles -/
theorem mothers_house_distance :
  (traveled_distance = (2/3) * total_distance) →
  total_distance = 234.0 :=
by
  sorry

#eval total_distance

end mothers_house_distance_l1458_145865
