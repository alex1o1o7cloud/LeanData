import Mathlib

namespace complex_magnitude_l2940_294084

theorem complex_magnitude (z : ℂ) (h : (1 - Complex.I) * z = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l2940_294084


namespace money_division_theorem_l2940_294096

/-- Represents the shares of P, Q, and R in the money division problem -/
structure Shares where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The problem of dividing money between P, Q, and R -/
def MoneyDivisionProblem (s : Shares) : Prop :=
  ∃ (x : ℝ),
    s.p = 5 * x ∧
    s.q = 11 * x ∧
    s.r = 19 * x ∧
    s.q - s.p = 12100

theorem money_division_theorem (s : Shares) 
  (h : MoneyDivisionProblem s) : s.r - s.q = 16133.36 := by
  sorry


end money_division_theorem_l2940_294096


namespace jenny_bottle_payment_l2940_294089

/-- Calculates the payment per bottle for Jenny's recycling --/
def payment_per_bottle (bottle_weight can_weight total_weight can_count can_payment total_payment : ℕ) : ℕ :=
  let remaining_weight := total_weight - can_count * can_weight
  let bottle_count := remaining_weight / bottle_weight
  let can_total_payment := can_count * can_payment
  let bottle_total_payment := total_payment - can_total_payment
  bottle_total_payment / bottle_count

/-- Theorem stating that Jenny's payment per bottle is 10 cents --/
theorem jenny_bottle_payment :
  payment_per_bottle 6 2 100 20 3 160 = 10 :=
by
  sorry

end jenny_bottle_payment_l2940_294089


namespace triangle_toothpick_count_l2940_294047

/-- The number of small equilateral triangles in the base row -/
def base_triangles : ℕ := 10

/-- The number of additional toothpicks in the isosceles row compared to the last equilateral row -/
def extra_isosceles_toothpicks : ℕ := 9

/-- The total number of small equilateral triangles in the main part of the large triangle -/
def total_equilateral_triangles : ℕ := (base_triangles * (base_triangles + 1)) / 2

/-- The number of toothpicks needed for the described triangle construction -/
def total_toothpicks : ℕ := 
  let equilateral_toothpicks := (3 * total_equilateral_triangles + 1) / 2
  let boundary_toothpicks := 2 * base_triangles + extra_isosceles_toothpicks
  equilateral_toothpicks + extra_isosceles_toothpicks + boundary_toothpicks - base_triangles

theorem triangle_toothpick_count : total_toothpicks = 110 := by sorry

end triangle_toothpick_count_l2940_294047


namespace largest_w_value_l2940_294052

theorem largest_w_value (w x y z : ℝ) 
  (sum_eq : w + x + y + z = 25)
  (prod_sum_eq : w*x + w*y + w*z + x*y + x*z + y*z = 2*y + 2*z + 193) :
  w ≤ 25/2 ∧ ∃ (w' : ℝ), w' = 25/2 ∧ 
    w' + x' + y' + z' = 25 ∧ 
    w'*x' + w'*y' + w'*z' + x'*y' + x'*z' + y'*z' = 2*y' + 2*z' + 193 :=
sorry

end largest_w_value_l2940_294052


namespace range_of_a_when_p_is_false_l2940_294062

theorem range_of_a_when_p_is_false :
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → a^2 - 5*a + 3 ≥ m + 2) →
  a ∈ Set.Iic 0 ∪ Set.Ici 5 :=
sorry

end range_of_a_when_p_is_false_l2940_294062


namespace probability_one_from_each_name_l2940_294072

/-- The probability of selecting one letter from each name when drawing two cards without replacement -/
theorem probability_one_from_each_name (total_cards : ℕ) (amelia_cards : ℕ) (lucas_cards : ℕ) :
  total_cards = amelia_cards + lucas_cards →
  total_cards = 10 →
  amelia_cards = 6 →
  lucas_cards = 4 →
  (amelia_cards * lucas_cards : ℚ) / ((total_cards * (total_cards - 1)) / 2) = 8 / 15 := by
  sorry

end probability_one_from_each_name_l2940_294072


namespace annas_money_l2940_294026

theorem annas_money (original spent remaining : ℚ) : 
  spent = (1 : ℚ) / 4 * original →
  remaining = (3 : ℚ) / 4 * original →
  remaining = 24 →
  original = 32 := by
sorry

end annas_money_l2940_294026


namespace intersection_of_A_and_B_l2940_294054

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | x^2 ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l2940_294054


namespace set_union_problem_l2940_294040

theorem set_union_problem (a b : ℝ) : 
  let M : Set ℝ := {3, 2*a}
  let N : Set ℝ := {a, b}
  M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
sorry

end set_union_problem_l2940_294040


namespace andrews_age_l2940_294034

theorem andrews_age :
  ∀ (a g : ℕ), 
    g = 15 * a →  -- Grandfather's age is fifteen times Andrew's age
    g - a = 70 →  -- Grandfather was 70 years old when Andrew was born
    a = 5         -- Andrew's age is 5
  := by sorry

end andrews_age_l2940_294034


namespace marys_bedrooms_l2940_294025

/-- Represents the number of rooms in a house -/
structure House where
  bedrooms : ℕ
  kitchen : Unit
  livingRoom : Unit

/-- Represents a vacuum cleaner -/
structure VacuumCleaner where
  batteryLife : ℕ  -- in minutes
  chargingTimes : ℕ

/-- Represents the time it takes to vacuum a room -/
def roomVacuumTime : ℕ := 4

theorem marys_bedrooms (h : House) (v : VacuumCleaner)
    (hv : v.batteryLife = 10 ∧ v.chargingTimes = 2) :
    h.bedrooms = 5 := by
  sorry

end marys_bedrooms_l2940_294025


namespace math_city_intersections_l2940_294016

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  if c.num_streets ≤ 1 then 0
  else (c.num_streets - 1) * (c.num_streets - 2) / 2

/-- Theorem stating that a city with 12 streets, no parallel streets, 
    and no triple intersections has 66 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 12 → c.no_parallel = true → 
  c.no_triple_intersections = true → num_intersections c = 66 :=
by
  sorry


end math_city_intersections_l2940_294016


namespace vacuum_pump_usage_l2940_294099

/-- The fraction of air remaining after each use of the pump -/
def remaining_fraction : ℝ := 0.4

/-- The target fraction of air to reach -/
def target_fraction : ℝ := 0.005

/-- The minimum number of pump uses required -/
def min_pump_uses : ℕ := 6

theorem vacuum_pump_usage (n : ℕ) :
  n ≥ min_pump_uses ↔ remaining_fraction ^ n < target_fraction :=
sorry

end vacuum_pump_usage_l2940_294099


namespace line_conditions_l2940_294065

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the lines from the problem
def line1 : Line := λ x y => x + y - 1
def line2 : Line := λ x y => x + y - 2
def line3 : Line := λ x y => x - 3*y + 3
def line4 : Line := λ x y => 3*x + y + 1

-- Define the points from the problem
def point1 : Point := (-1, 2)
def point2 : Point := (0, 1)

-- Define what it means for a line to pass through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l p.1 p.2 = 0

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y = k * l2 x y

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l1 x y = k * l2 y (-x)

-- State the theorem
theorem line_conditions : 
  (passes_through line1 point1 ∧ parallel line1 line2) ∧
  (passes_through line3 point2 ∧ perpendicular line3 line4) := by
  sorry

end line_conditions_l2940_294065


namespace sum_of_valid_a_l2940_294095

theorem sum_of_valid_a : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
    5 * x₁ ≥ 3 * (x₁ + 2) ∧ x₁ - (x₁ + 3) / 2 ≤ a / 16 ∧
    5 * x₂ ≥ 3 * (x₂ + 2) ∧ x₂ - (x₂ + 3) / 2 ≤ a / 16) ∧
   (∃ y : ℤ, y < 0 ∧ 5 + a * y = 2 * y - 7)) ∧
  (S.sum id = 22) := by
  sorry

end sum_of_valid_a_l2940_294095


namespace max_additional_plates_l2940_294050

/-- Represents the sets of letters for each position in the license plate --/
structure LicensePlateSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)
  (fourth : Finset Char)

/-- The initial license plate sets --/
def initialSets : LicensePlateSets :=
  { first := {'A', 'E', 'I', 'O', 'U'},
    second := {'B', 'C', 'D'},
    third := {'L', 'M', 'N', 'P'},
    fourth := {'S', 'T'} }

/-- The number of new letters that can be added --/
def newLettersCount : Nat := 3

/-- The maximum number of letters that can be added to a single set --/
def maxAddToSet : Nat := 2

/-- Calculates the number of possible license plates --/
def calculatePlates (sets : LicensePlateSets) : Nat :=
  sets.first.card * sets.second.card * sets.third.card * sets.fourth.card

/-- Theorem: The maximum number of additional license plates is 180 --/
theorem max_additional_plates :
  ∃ (newSets : LicensePlateSets),
    (calculatePlates newSets - calculatePlates initialSets = 180) ∧
    (∀ (otherSets : LicensePlateSets),
      (calculatePlates otherSets - calculatePlates initialSets) ≤ 180) :=
sorry


end max_additional_plates_l2940_294050


namespace tangent_fraction_equality_l2940_294059

theorem tangent_fraction_equality (α β : Real) 
  (h1 : Real.tan (α - β) = 2) 
  (h2 : Real.tan β = 4) : 
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 := by
  sorry

end tangent_fraction_equality_l2940_294059


namespace line_through_P_and_origin_equation_line_l_equation_l2940_294081

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Define the line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Define the line l passing through P and perpendicular to l₃
def line_l (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Theorem 1: The line passing through P and the origin has the equation x + y = 0
theorem line_through_P_and_origin_equation :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → line_through_P_and_origin x y :=
by sorry

-- Theorem 2: The line l passing through P and perpendicular to l₃ has the equation x - 2y + 6 = 0
theorem line_l_equation :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y ∧ l₃ x y → line_l x y :=
by sorry

end line_through_P_and_origin_equation_line_l_equation_l2940_294081


namespace intersection_of_M_and_N_l2940_294069

-- Define set M
def M : Set ℝ := {x | x / (x - 1) ≥ 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x | x > 1} := by sorry

end intersection_of_M_and_N_l2940_294069


namespace infinite_pairs_with_difference_one_l2940_294087

-- Define the property of being tuanis
def is_tuanis (a b : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (a + b).digits 10 → d = 0 ∨ d = 1

-- Define the sets A and B
def tuanis_set (A B : Set ℕ) : Prop :=
  (∀ a ∈ A, ∃ b ∈ B, is_tuanis a b) ∧
  (∀ b ∈ B, ∃ a ∈ A, is_tuanis a b)

-- The main theorem
theorem infinite_pairs_with_difference_one
  (A B : Set ℕ) (hA : Set.Infinite A) (hB : Set.Infinite B)
  (h_tuanis : tuanis_set A B) :
  (Set.Infinite {p : ℕ × ℕ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 = 1}) ∨
  (Set.Infinite {p : ℕ × ℕ | p.1 ∈ B ∧ p.2 ∈ B ∧ p.1 - p.2 = 1}) :=
sorry

end infinite_pairs_with_difference_one_l2940_294087


namespace sword_length_difference_is_23_l2940_294060

/-- The length difference between June's and Christopher's swords -/
def sword_length_difference : ℕ → ℕ → ℕ → ℕ
  | christopher_length, jameson_diff, june_diff =>
    let jameson_length := 2 * christopher_length + jameson_diff
    let june_length := jameson_length + june_diff
    june_length - christopher_length

theorem sword_length_difference_is_23 :
  sword_length_difference 15 3 5 = 23 := by
  sorry

end sword_length_difference_is_23_l2940_294060


namespace dispatch_riders_travel_time_l2940_294097

/-- Represents the travel scenario of two dispatch riders -/
structure DispatchRiders where
  a : ℝ  -- Speed increase of the first rider in km/h
  x : ℝ  -- Initial speed of the first rider in km/h
  y : ℝ  -- Speed of the second rider in km/h
  z : ℝ  -- Actual travel time of the first rider in hours

/-- The conditions of the dispatch riders' travel -/
def travel_conditions (d : DispatchRiders) : Prop :=
  d.a > 0 ∧ d.a < 30 ∧
  d.x > 0 ∧ d.y > 0 ∧ d.z > 0 ∧
  180 / d.x - 180 / d.y = 6 ∧
  d.z * (d.x + d.a) = 180 ∧
  (d.z - 3) * d.y = 180

/-- The theorem stating the travel times of both riders -/
theorem dispatch_riders_travel_time (d : DispatchRiders) 
  (h : travel_conditions d) : 
  d.z = (-3 * d.a + 3 * Real.sqrt (d.a^2 + 240 * d.a)) / (2 * d.a) ∧
  d.z - 3 = (-9 * d.a + 3 * Real.sqrt (d.a^2 + 240 * d.a)) / (2 * d.a) := by
  sorry

end dispatch_riders_travel_time_l2940_294097


namespace canal_meeting_participants_l2940_294080

theorem canal_meeting_participants (total : Nat) (greetings : Nat) : total = 12 ∧ greetings = 31 →
  ∃ (egyptians panamanians : Nat),
    egyptians + panamanians = total ∧
    egyptians > panamanians ∧
    egyptians * (egyptians - 1) / 2 + panamanians * (panamanians - 1) / 2 = greetings ∧
    egyptians = 7 ∧
    panamanians = 5 := by
  sorry

end canal_meeting_participants_l2940_294080


namespace geometric_sequence_sum_l2940_294042

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 2017 * 2016^n - 2018 * t) →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →
  t = 2017 / 2018 := by
  sorry

end geometric_sequence_sum_l2940_294042


namespace square_cross_section_cylinder_volume_l2940_294056

/-- A cylinder with a square cross-section and lateral surface area 4π has volume 2π -/
theorem square_cross_section_cylinder_volume (h : ℝ) (r : ℝ) : 
  h > 0 → r > 0 → h = 2*r → h * (4*r) = 4*π → π * r^2 * h = 2*π := by
  sorry

end square_cross_section_cylinder_volume_l2940_294056


namespace michael_pet_sitting_cost_l2940_294079

-- Define the number of cats and dogs
def num_cats : ℕ := 2
def num_dogs : ℕ := 3

-- Define the cost per animal per night
def cost_per_animal : ℕ := 13

-- Define the total number of animals
def total_animals : ℕ := num_cats + num_dogs

-- State the theorem
theorem michael_pet_sitting_cost :
  total_animals * cost_per_animal = 65 := by
  sorry

end michael_pet_sitting_cost_l2940_294079


namespace circle_area_l2940_294002

theorem circle_area (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 + 8 * x - 4 * y - 16 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ 
    π * radius^2 = 13 * π) :=
by sorry

end circle_area_l2940_294002


namespace dog_adult_weights_l2940_294012

-- Define the dog breeds
inductive DogBreed
| GoldenRetriever
| Labrador
| Poodle

-- Define the weight progression function
def weightProgression (breed : DogBreed) : ℕ → ℕ
| 0 => match breed with
  | DogBreed.GoldenRetriever => 6
  | DogBreed.Labrador => 8
  | DogBreed.Poodle => 4
| 1 => match breed with
  | DogBreed.GoldenRetriever => 12
  | DogBreed.Labrador => 24
  | DogBreed.Poodle => 16
| 2 => match breed with
  | DogBreed.GoldenRetriever => 24
  | DogBreed.Labrador => 36
  | DogBreed.Poodle => 32
| 3 => match breed with
  | DogBreed.GoldenRetriever => 48
  | DogBreed.Labrador => 72
  | DogBreed.Poodle => 32
| _ => 0

-- Define the final weight increase function
def finalWeightIncrease (breed : DogBreed) : ℕ :=
  match breed with
  | DogBreed.GoldenRetriever => 30
  | DogBreed.Labrador => 30
  | DogBreed.Poodle => 20

-- Define the adult weight function
def adultWeight (breed : DogBreed) : ℕ :=
  weightProgression breed 3 + finalWeightIncrease breed

-- Theorem statement
theorem dog_adult_weights :
  (adultWeight DogBreed.GoldenRetriever = 78) ∧
  (adultWeight DogBreed.Labrador = 102) ∧
  (adultWeight DogBreed.Poodle = 52) := by
  sorry

end dog_adult_weights_l2940_294012


namespace expression_evaluation_l2940_294005

theorem expression_evaluation (a b : ℚ) (h1 : a = -1) (h2 : b = 1/2) :
  5*a*b - 2*(3*a*b - (4*a*b^2 + 1/2*a*b)) - 5*a*b^2 = -3/4 := by
  sorry

end expression_evaluation_l2940_294005


namespace total_pencils_after_adding_l2940_294024

def initial_pencils : ℕ := 115
def added_pencils : ℕ := 100

theorem total_pencils_after_adding :
  initial_pencils + added_pencils = 215 := by
  sorry

end total_pencils_after_adding_l2940_294024


namespace sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2940_294010

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l2940_294010


namespace stratified_sampling_theorem_l2940_294063

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ

/-- Calculates the number of elderly employees to sample given the number of young employees sampled -/
def elderlyToSample (ec : EmployeeCount) (youngSampled : ℕ) : ℕ :=
  (youngSampled * ec.elderly) / ec.young

/-- Theorem stating that given the specific employee counts and 7 young employees sampled, 
    3 elderly employees should be sampled -/
theorem stratified_sampling_theorem (ec : EmployeeCount) 
  (h1 : ec.total = 750)
  (h2 : ec.young = 350)
  (h3 : ec.middleAged = 250)
  (h4 : ec.elderly = 150)
  (h5 : ec.total = ec.young + ec.middleAged + ec.elderly) :
  elderlyToSample ec 7 = 3 := by
  sorry

#eval elderlyToSample { total := 750, young := 350, middleAged := 250, elderly := 150 } 7

end stratified_sampling_theorem_l2940_294063


namespace x_negative_necessary_not_sufficient_for_quadratic_inequality_l2940_294035

theorem x_negative_necessary_not_sufficient_for_quadratic_inequality :
  (∀ x : ℝ, x^2 + x < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ x^2 + x ≥ 0) := by
  sorry

end x_negative_necessary_not_sufficient_for_quadratic_inequality_l2940_294035


namespace symbol_equation_solution_l2940_294090

theorem symbol_equation_solution (triangle circle : ℕ) 
  (h1 : triangle + circle + circle = 55)
  (h2 : triangle + circle = 40) :
  circle = 15 ∧ triangle = 25 := by
  sorry

end symbol_equation_solution_l2940_294090


namespace train_length_calculation_l2940_294055

/-- The length of each train in kilometers -/
def train_length : ℝ := 0.06

/-- The speed of the faster train in km/hr -/
def fast_train_speed : ℝ := 48

/-- The speed of the slower train in km/hr -/
def slow_train_speed : ℝ := 36

/-- The time taken for the faster train to pass the slower train in seconds -/
def passing_time : ℝ := 36

theorem train_length_calculation :
  let relative_speed := fast_train_speed - slow_train_speed
  let relative_speed_km_per_sec := relative_speed / 3600
  2 * train_length = relative_speed_km_per_sec * passing_time :=
by sorry

end train_length_calculation_l2940_294055


namespace unique_parallel_line_in_plane_l2940_294061

/-- A plane in 3D space -/
structure Plane3D where
  -- (Placeholder for plane definition)

/-- A line in 3D space -/
structure Line3D where
  -- (Placeholder for line definition)

/-- A point in 3D space -/
structure Point3D where
  -- (Placeholder for point definition)

/-- Predicate for a line being parallel to a plane -/
def parallel_line_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a point being on a plane -/
def point_on_plane (P : Point3D) (α : Plane3D) : Prop :=
  sorry

/-- Predicate for a line passing through a point -/
def line_through_point (l : Line3D) (P : Point3D) : Prop :=
  sorry

/-- Predicate for two lines being parallel -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for a line lying in a plane -/
def line_in_plane (l : Line3D) (α : Plane3D) : Prop :=
  sorry

theorem unique_parallel_line_in_plane 
  (l : Line3D) (α : Plane3D) (P : Point3D)
  (h1 : parallel_line_plane l α)
  (h2 : point_on_plane P α) :
  ∃! m : Line3D, line_through_point m P ∧ parallel_lines m l ∧ line_in_plane m α :=
sorry

end unique_parallel_line_in_plane_l2940_294061


namespace modulus_of_z_l2940_294029

theorem modulus_of_z (z : ℂ) : z = Complex.I * (2 - Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_z_l2940_294029


namespace x_value_proof_l2940_294082

theorem x_value_proof (x y : ℚ) : x / y = 12 / 5 → y = 20 → x = 48 := by
  sorry

end x_value_proof_l2940_294082


namespace circle_with_n_integer_points_l2940_294046

/-- A point on the coordinate plane with rational x-coordinate and irrational y-coordinate -/
structure SpecialPoint where
  x : ℚ
  y : ℝ
  y_irrational : Irrational y

/-- The number of integer points inside a circle -/
def IntegerPointsInside (center : ℝ × ℝ) (radius : ℝ) : ℕ :=
  sorry

/-- Theorem: For any non-negative integer n, there exists a circle on the coordinate plane
    that contains exactly n integer points in its interior -/
theorem circle_with_n_integer_points (n : ℕ) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), IntegerPointsInside center radius = n :=
sorry

end circle_with_n_integer_points_l2940_294046


namespace alpha_value_l2940_294008

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : ∀ x, (Real.sin α) ^ (x^2 - 2*x + 3) ≤ 1/4) : α = 5*π/6 := by
  sorry

end alpha_value_l2940_294008


namespace square_roots_sum_zero_l2940_294071

theorem square_roots_sum_zero (x a : ℝ) : 
  x > 0 → (a + 1) ^ 2 = x → (a - 3) ^ 2 = x → a = 1 := by
  sorry

end square_roots_sum_zero_l2940_294071


namespace largest_prime_factor_of_12321_l2940_294038

theorem largest_prime_factor_of_12321 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 12321 → q ≤ p :=
by
  -- The proof would go here
  sorry

end largest_prime_factor_of_12321_l2940_294038


namespace geometric_sequence_sum_property_l2940_294031

/-- Given a geometric sequence with first term a₁ and common ratio q,
    the sum of the 4th, 5th, and 6th terms squared equals the product of
    the sum of the 1st, 2nd, and 3rd terms and the sum of the 7th, 8th, and 9th terms. -/
theorem geometric_sequence_sum_property (a₁ q : ℝ) :
  (a₁ * q^3 + a₁ * q^4 + a₁ * q^5)^2 = (a₁ + a₁ * q + a₁ * q^2) * (a₁ * q^6 + a₁ * q^7 + a₁ * q^8) := by
  sorry

end geometric_sequence_sum_property_l2940_294031


namespace coefficient_x_squared_in_expansion_l2940_294000

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (fun k => Nat.choose 5 k * (2 * x) ^ k * 1 ^ (5 - k)) = 
  1 + 10 * x + 40 * x^2 + 80 * x^3 + 80 * x^4 + 32 * x^5 :=
by sorry

end coefficient_x_squared_in_expansion_l2940_294000


namespace rectangular_table_capacity_l2940_294083

/-- The number of rectangular tables in the library -/
def num_rectangular_tables : ℕ := 7

/-- The number of pupils a square table can seat -/
def pupils_per_square_table : ℕ := 4

/-- The number of square tables in the library -/
def num_square_tables : ℕ := 5

/-- The total number of pupils that can be seated -/
def total_pupils : ℕ := 90

/-- The number of pupils a rectangular table can seat -/
def pupils_per_rectangular_table : ℕ := 10

theorem rectangular_table_capacity :
  pupils_per_rectangular_table * num_rectangular_tables +
  pupils_per_square_table * num_square_tables = total_pupils :=
by sorry

end rectangular_table_capacity_l2940_294083


namespace inequality_proof_l2940_294076

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end inequality_proof_l2940_294076


namespace theorem_3_squeeze_theorem_l2940_294028

-- Theorem 3
theorem theorem_3 (v u : ℕ → ℝ) (n_0 : ℕ) 
  (h_v : ∀ ε > 0, ∃ N, ∀ n ≥ N, |v n| ≤ ε) 
  (h_u : ∀ n ≥ n_0, |u n| ≤ |v n|) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n| ≤ ε :=
sorry

-- Squeeze Theorem
theorem squeeze_theorem (u v w : ℕ → ℝ) (l : ℝ) (n_0 : ℕ)
  (h_u : ∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - l| ≤ ε)
  (h_w : ∀ ε > 0, ∃ N, ∀ n ≥ N, |w n - l| ≤ ε)
  (h_v : ∀ n ≥ n_0, u n ≤ v n ∧ v n ≤ w n) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |v n - l| ≤ ε :=
sorry

end theorem_3_squeeze_theorem_l2940_294028


namespace combination_equality_l2940_294007

theorem combination_equality (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 10 (3*x - 2)) → (x = 1 ∨ x = 3) := by
sorry

end combination_equality_l2940_294007


namespace product_equals_one_l2940_294019

theorem product_equals_one (n : ℕ) (x : ℕ → ℝ) (f : ℕ → ℝ) :
  n > 2 →
  (∀ i j, i % n = j % n → x i = x j) →
  (∀ i, f i = x i + x i * x (i + 1) + x i * x (i + 1) * x (i + 2) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) + 
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) +
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) * x (i + 6) +
    x i * x (i + 1) * x (i + 2) * x (i + 3) * x (i + 4) * x (i + 5) * x (i + 6) * x (i + 7)) →
  (∀ i j, f i = f j) →
  (∃ i j, x i ≠ x j) →
  (x 1 * x 2 * x 3 * x 4 * x 5 * x 6 * x 7 * x 8) = 1 := by
sorry

end product_equals_one_l2940_294019


namespace quotient_change_l2940_294039

theorem quotient_change (a b : ℝ) (h : b ≠ 0) :
  ((100 * a) / (b / 10)) = 1000 * (a / b) := by
sorry

end quotient_change_l2940_294039


namespace initial_number_of_kids_l2940_294085

theorem initial_number_of_kids (kids_left : ℕ) (kids_gone_home : ℕ) : 
  kids_left = 8 ∧ kids_gone_home = 14 → kids_left + kids_gone_home = 22 :=
by
  sorry

end initial_number_of_kids_l2940_294085


namespace factorial_quotient_equals_56_l2940_294045

theorem factorial_quotient_equals_56 :
  ∃! (n : ℕ), n > 0 ∧ n.factorial / (n - 2).factorial = 56 := by
  sorry

end factorial_quotient_equals_56_l2940_294045


namespace trajectory_of_P_l2940_294037

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the relationship between N, M, and P
def RelationNMP (xn yn xm ym xp yp : ℝ) : Prop :=
  C xn yn ∧ xm = 0 ∧ ym = yn ∧ xp = xn / 2 ∧ yp = yn

-- Theorem statement
theorem trajectory_of_P : 
  ∀ (x y : ℝ), (∃ (xn yn xm ym : ℝ), RelationNMP xn yn xm ym x y) → 
  x^2 / 2 + y^2 / 8 = 1 := by
sorry

end trajectory_of_P_l2940_294037


namespace average_age_decrease_l2940_294018

theorem average_age_decrease (initial_average : ℝ) : 
  let initial_total_age := 10 * initial_average
  let new_total_age := initial_total_age - 48 + 18
  let new_average := new_total_age / 10
  initial_average - new_average = 3 := by
sorry

end average_age_decrease_l2940_294018


namespace symmetry_proof_l2940_294092

/-- Given two lines in the 2D plane represented by their equations,
    this function returns true if they are symmetric with respect to the line y = -x. -/
def are_symmetric (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 (-y) (-x)

/-- The equation of the original line: 3x - 4y + 5 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the symmetric line: 4x - 3y + 5 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y + 5 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line x + y = 0 -/
theorem symmetry_proof : are_symmetric original_line symmetric_line :=
sorry

end symmetry_proof_l2940_294092


namespace tangent_point_and_perpendicular_line_l2940_294049

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P₀
structure Point where
  x : ℝ
  y : ℝ

def P₀ : Point := ⟨-1, -4⟩

-- Define the third quadrant
def in_third_quadrant (p : Point) : Prop := p.x < 0 ∧ p.y < 0

-- Define the tangent line slope
def tangent_slope : ℝ := 4

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + 4 * y + 17 = 0

theorem tangent_point_and_perpendicular_line :
  f P₀.x = P₀.y ∧
  f' P₀.x = tangent_slope ∧
  in_third_quadrant P₀ →
  perpendicular_line P₀.x P₀.y :=
by sorry

end tangent_point_and_perpendicular_line_l2940_294049


namespace problem_solution_l2940_294044

theorem problem_solution (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 := by
  sorry

end problem_solution_l2940_294044


namespace negation_of_monotonicity_like_property_l2940_294036

/-- The negation of a monotonicity-like property for a real-valued function -/
theorem negation_of_monotonicity_like_property (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := by sorry

end negation_of_monotonicity_like_property_l2940_294036


namespace inequality_solution_implies_m_range_l2940_294068

theorem inequality_solution_implies_m_range 
  (m : ℝ) 
  (h : ∀ x : ℝ, (m * x - 1) * (x - 2) > 0 ↔ (1 / m < x ∧ x < 2)) : 
  m < 0 := by
sorry

end inequality_solution_implies_m_range_l2940_294068


namespace set_A_proof_l2940_294051

def U : Set Nat := {0, 1, 2, 3, 4, 5}

theorem set_A_proof (A B : Set Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ A) ∩ B = {0, 4})
  (h4 : (U \ A) ∩ (U \ B) = {3, 5}) :
  A = {1, 2} := by
  sorry

end set_A_proof_l2940_294051


namespace diamond_equation_solution_l2940_294091

-- Define the binary operation
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Define the properties of the operation
axiom diamond_assoc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = (diamond a b) ^ c

axiom diamond_self (a : ℝ) (ha : a ≠ 0) : diamond a a = 1

-- State the theorem
theorem diamond_equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ diamond 2048 (diamond 4 x) = 16 := by sorry

end diamond_equation_solution_l2940_294091


namespace arithmetic_mean_problem_l2940_294027

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 84 → a = 32 := by sorry

end arithmetic_mean_problem_l2940_294027


namespace future_years_calculation_l2940_294086

/-- The number of years in the future when Shekhar will be 26 years old -/
def future_years : ℕ := 6

/-- Shekhar's current age -/
def shekhar_current_age : ℕ := 20

/-- Shobha's current age -/
def shobha_current_age : ℕ := 15

/-- The ratio of Shekhar's age to Shobha's age -/
def age_ratio : ℚ := 4 / 3

theorem future_years_calculation :
  (shekhar_current_age + future_years = 26) ∧
  (shekhar_current_age : ℚ) / shobha_current_age = age_ratio :=
by sorry

end future_years_calculation_l2940_294086


namespace mutuallyExclusiveNotContradictoryPairs_l2940_294032

-- Define the events
inductive Event : Type
| Miss : Event
| Hit : Event
| MoreThan4 : Event
| AtLeast5 : Event

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : Event) : Prop := sorry

-- Define contradictory events
def contradictory (e1 e2 : Event) : Prop := sorry

-- Define a function to count pairs of events that are mutually exclusive but not contradictory
def countMutuallyExclusiveNotContradictory (events : List Event) : Nat := sorry

-- Theorem to prove
theorem mutuallyExclusiveNotContradictoryPairs :
  let events := [Event.Miss, Event.Hit, Event.MoreThan4, Event.AtLeast5]
  countMutuallyExclusiveNotContradictory events = 2 := by sorry

end mutuallyExclusiveNotContradictoryPairs_l2940_294032


namespace bread_slices_problem_l2940_294006

theorem bread_slices_problem (initial_slices : ℕ) : 
  (initial_slices : ℚ) * (2/3) - 2 = 6 → initial_slices = 12 := by
  sorry

end bread_slices_problem_l2940_294006


namespace solve_seashells_problem_l2940_294098

def seashells_problem (monday_shells : ℕ) (total_money : ℚ) : Prop :=
  let tuesday_shells : ℕ := monday_shells / 2
  let total_shells : ℕ := monday_shells + tuesday_shells
  let money_per_shell : ℚ := total_money / total_shells
  monday_shells = 30 ∧ total_money = 54 → money_per_shell = 1.20

theorem solve_seashells_problem :
  seashells_problem 30 54 := by sorry

end solve_seashells_problem_l2940_294098


namespace specific_tetrahedron_volume_l2940_294064

/-- Represents a tetrahedron ABCD with given edge lengths -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BD : ℝ
  AD : ℝ
  CD : ℝ

/-- Calculate the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- The theorem stating that the volume of the specific tetrahedron is 24/5 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    AB := 5,
    AC := 3,
    BC := 4,
    BD := 4,
    AD := 3,
    CD := 12/5 * Real.sqrt 2
  }
  tetrahedronVolume t = 24/5 := by
  sorry

end specific_tetrahedron_volume_l2940_294064


namespace polynomial_division_remainder_l2940_294043

theorem polynomial_division_remainder : ∃ (Q : Polynomial ℤ) (R : Polynomial ℤ),
  (X : Polynomial ℤ)^50 = (X^2 - 5*X + 6) * Q + R ∧
  (Polynomial.degree R < 2) ∧
  R = (3^50 - 2^50) * X + (2^50 - 2 * 3^50) := by
  sorry

end polynomial_division_remainder_l2940_294043


namespace x_plus_four_value_l2940_294075

theorem x_plus_four_value (x t : ℝ) 
  (h1 : 6 * x + t = 4 * x - 9) 
  (h2 : t = 7) : 
  x + 4 = -4 := by
sorry

end x_plus_four_value_l2940_294075


namespace total_cost_of_suits_l2940_294053

def cost_of_first_suit : ℕ := 300

def cost_of_second_suit (first_suit_cost : ℕ) : ℕ :=
  3 * first_suit_cost + 200

def total_cost (first_suit_cost : ℕ) : ℕ :=
  first_suit_cost + cost_of_second_suit first_suit_cost

theorem total_cost_of_suits :
  total_cost cost_of_first_suit = 1400 := by
  sorry

end total_cost_of_suits_l2940_294053


namespace bakers_cakes_l2940_294030

/-- Baker's cake selling problem -/
theorem bakers_cakes (initial_cakes bought_cakes sold_difference : ℕ) 
  (h1 : initial_cakes = 8)
  (h2 : bought_cakes = 139)
  (h3 : sold_difference = 6) :
  bought_cakes + sold_difference = 145 := by
  sorry

#check bakers_cakes

end bakers_cakes_l2940_294030


namespace min_turtle_distance_l2940_294020

/-- Represents an observer watching the turtle --/
structure Observer where
  startTime : ℕ
  endTime : ℕ
  distanceObserved : ℕ

/-- Represents the turtle's movement --/
def TurtleMovement (observers : List Observer) : Prop :=
  -- The observation lasts for 6 minutes
  (∀ o ∈ observers, o.startTime ≥ 0 ∧ o.endTime ≤ 6 * 60) ∧
  -- Each observer watches for 1 minute continuously
  (∀ o ∈ observers, o.endTime - o.startTime = 60) ∧
  -- Each observer notes 1 meter of movement
  (∀ o ∈ observers, o.distanceObserved = 1) ∧
  -- The turtle is always being observed
  (∀ t : ℕ, t ≥ 0 ∧ t ≤ 6 * 60 → ∃ o ∈ observers, o.startTime ≤ t ∧ t < o.endTime)

/-- The theorem stating the minimum distance the turtle could have traveled --/
theorem min_turtle_distance (observers : List Observer) 
  (h : TurtleMovement observers) : 
  ∃ d : ℕ, d = 4 ∧ (∀ d' : ℕ, (∃ obs : List Observer, TurtleMovement obs ∧ d' = obs.length) → d ≤ d') :=
sorry

end min_turtle_distance_l2940_294020


namespace seventh_row_cans_l2940_294074

/-- Represents a triangular display of cans -/
structure CanDisplay where
  rows : Nat
  first_row_cans : Nat
  increment : Nat

/-- Calculate the number of cans in a specific row -/
def cans_in_row (d : CanDisplay) (row : Nat) : Nat :=
  d.first_row_cans + d.increment * (row - 1)

/-- Calculate the total number of cans in the display -/
def total_cans (d : CanDisplay) : Nat :=
  (d.rows * (2 * d.first_row_cans + (d.rows - 1) * d.increment)) / 2

/-- The main theorem -/
theorem seventh_row_cans (d : CanDisplay) :
  d.rows = 10 ∧ d.increment = 3 ∧ total_cans d < 150 →
  cans_in_row d 7 = 19 := by
  sorry

#eval cans_in_row { rows := 10, first_row_cans := 1, increment := 3 } 7

end seventh_row_cans_l2940_294074


namespace not_or_implies_both_false_l2940_294033

theorem not_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end not_or_implies_both_false_l2940_294033


namespace counting_sequence_53rd_term_l2940_294041

theorem counting_sequence_53rd_term : 
  let seq : ℕ → ℕ := λ n => n
  seq 53 = 10 := by
  sorry

end counting_sequence_53rd_term_l2940_294041


namespace problem_statement_l2940_294004

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := m > 1

-- State the theorem
theorem problem_statement (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (p m ∧ ¬(q m)) ∧ (1 < m ∧ m ≤ 2) :=
by sorry

end problem_statement_l2940_294004


namespace exactly_one_integer_satisfies_inequality_l2940_294070

theorem exactly_one_integer_satisfies_inequality :
  ∃! (n : ℕ), n > 0 ∧ 30 - 6 * n > 18 :=
by sorry

end exactly_one_integer_satisfies_inequality_l2940_294070


namespace distance_between_trees_l2940_294057

/-- Given a yard of length 325 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 13 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 325 → num_trees = 26 → (yard_length / (num_trees - 1 : ℝ)) = 13 := by
  sorry

end distance_between_trees_l2940_294057


namespace triangle_max_area_l2940_294022

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) 
  (h5 : 0 < C) (h6 : C < π) (h7 : A + B + C = π) 
  (h8 : Real.tan A * Real.tan B = 1) (h9 : Real.sqrt 3 = 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :
  (∃ (S : ℝ → ℝ), (∀ x, S x ≤ S (π/4)) ∧ S A = (3/4) * Real.sin (2*A)) :=
sorry

end triangle_max_area_l2940_294022


namespace carnival_activity_order_l2940_294014

/- Define the activities -/
inductive Activity
| Dodgeball
| MagicShow
| SingingContest

/- Define the popularity of each activity -/
def popularity : Activity → Rat
| Activity.Dodgeball => 9/24
| Activity.MagicShow => 4/12
| Activity.SingingContest => 1/3

/- Define the ordering of activities based on popularity -/
def more_popular (a b : Activity) : Prop :=
  popularity a > popularity b

/- Theorem statement -/
theorem carnival_activity_order :
  (more_popular Activity.Dodgeball Activity.MagicShow) ∧
  (more_popular Activity.MagicShow Activity.SingingContest) ∨
  (popularity Activity.MagicShow = popularity Activity.SingingContest) :=
sorry

end carnival_activity_order_l2940_294014


namespace polynomial_coefficient_theorem_l2940_294017

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, x^4 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4) →
  a₃ = -8 := by
sorry

end polynomial_coefficient_theorem_l2940_294017


namespace wage_multiple_l2940_294078

/-- Given Kem's hourly wage and Shem's daily wage for 8 hours, 
    calculate the multiple of Shem's hourly wage compared to Kem's. -/
theorem wage_multiple (kem_hourly_wage shem_daily_wage : ℚ) 
  (h1 : kem_hourly_wage = 4)
  (h2 : shem_daily_wage = 80)
  (h3 : shem_daily_wage = 8 * (shem_daily_wage / 8)) : 
  (shem_daily_wage / 8) / kem_hourly_wage = 5/2 := by
  sorry

end wage_multiple_l2940_294078


namespace apple_cost_per_kg_l2940_294048

theorem apple_cost_per_kg (p q : ℚ) : 
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 := by sorry

end apple_cost_per_kg_l2940_294048


namespace symmetric_function_minimum_value_l2940_294094

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1) * (x + 2) * (x^2 + a*x + b)

-- State the theorem
theorem symmetric_function_minimum_value (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- Symmetry condition
  (∃ x, ∀ y, f a b y ≥ f a b x ∧ f a b x = -9/4) :=
by sorry

end symmetric_function_minimum_value_l2940_294094


namespace unique_k_value_l2940_294058

/-- The equation has infinitely many solutions when the coefficients of x are equal on both sides -/
def has_infinitely_many_solutions (k : ℝ) : Prop :=
  3 * k = 15

/-- The value of k for which the equation has infinitely many solutions -/
def k_value : ℝ := 5

/-- Theorem stating that k_value is the unique solution -/
theorem unique_k_value :
  has_infinitely_many_solutions k_value ∧
  ∀ k : ℝ, has_infinitely_many_solutions k → k = k_value :=
by sorry

end unique_k_value_l2940_294058


namespace max_operations_l2940_294088

def operation_count (a b : ℕ) : ℕ := sorry

theorem max_operations (a b : ℕ) (ha : a = 2000) (hb : b < 2000) :
  operation_count a b ≤ 10 := by sorry

end max_operations_l2940_294088


namespace milton_books_total_l2940_294023

/-- The number of zoology books Milton has -/
def zoology_books : ℕ := 16

/-- The number of botany books Milton has -/
def botany_books : ℕ := 4 * zoology_books

/-- The total number of books Milton has -/
def total_books : ℕ := zoology_books + botany_books

theorem milton_books_total : total_books = 80 := by
  sorry

end milton_books_total_l2940_294023


namespace peculiar_animal_farm_l2940_294073

theorem peculiar_animal_farm (cats dogs : ℕ) : 
  dogs = cats + 180 →
  (cats + (dogs / 5 : ℚ)) / (cats + dogs : ℚ) = 32 / 100 →
  cats + dogs = 240 := by
sorry

end peculiar_animal_farm_l2940_294073


namespace abs_x_squared_minus_x_lt_two_l2940_294013

theorem abs_x_squared_minus_x_lt_two (x : ℝ) :
  |x^2 - x| < 2 ↔ -1 < x ∧ x < 2 := by sorry

end abs_x_squared_minus_x_lt_two_l2940_294013


namespace kim_total_sweaters_l2940_294077

/-- The number of sweaters Kim knit on each day of the week --/
structure SweaterCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Kim's sweater knitting for the week --/
def kim_sweater_conditions (sc : SweaterCount) : Prop :=
  sc.monday = 8 ∧
  sc.tuesday = sc.monday + 2 ∧
  sc.wednesday = sc.tuesday - 4 ∧
  sc.thursday = sc.wednesday ∧
  sc.friday = sc.monday / 2

/-- The theorem stating the total number of sweaters Kim knit that week --/
theorem kim_total_sweaters (sc : SweaterCount) 
  (h : kim_sweater_conditions sc) : 
  sc.monday + sc.tuesday + sc.wednesday + sc.thursday + sc.friday = 34 := by
  sorry


end kim_total_sweaters_l2940_294077


namespace club_M_members_eq_five_l2940_294009

/-- The number of people who joined club M in a company with the following conditions:
  - There are 60 people in total
  - There are 3 clubs: M, S, and Z
  - 18 people joined S
  - 11 people joined Z
  - Members of M did not join any other club
  - At most 26 people did not join any club
-/
def club_M_members : ℕ := by
  -- Define the total number of people
  let total_people : ℕ := 60
  -- Define the number of people in club S
  let club_S_members : ℕ := 18
  -- Define the number of people in club Z
  let club_Z_members : ℕ := 11
  -- Define the maximum number of people who didn't join any club
  let max_no_club : ℕ := 26
  
  -- The actual proof would go here
  sorry

theorem club_M_members_eq_five : club_M_members = 5 := by
  sorry

end club_M_members_eq_five_l2940_294009


namespace three_layer_runner_area_l2940_294093

/-- Given three table runners with a combined area of 204 square inches covering 80% of a table
    with an area of 175 square inches, and an area of 24 square inches covered by exactly two
    layers of runner, prove that the area covered by three layers of runner is 20 square inches. -/
theorem three_layer_runner_area
  (total_runner_area : ℝ)
  (table_area : ℝ)
  (coverage_percent : ℝ)
  (two_layer_area : ℝ)
  (h1 : total_runner_area = 204)
  (h2 : table_area = 175)
  (h3 : coverage_percent = 0.8)
  (h4 : two_layer_area = 24)
  : ∃ (three_layer_area : ℝ),
    three_layer_area = 20 ∧
    coverage_percent * table_area = (total_runner_area - two_layer_area - three_layer_area) + 2 * two_layer_area + 3 * three_layer_area :=
by sorry

end three_layer_runner_area_l2940_294093


namespace tee_price_calculation_l2940_294001

/-- The price of a single tee shirt in Linda's store -/
def tee_price : ℝ := 8

/-- The price of a single pair of jeans in Linda's store -/
def jeans_price : ℝ := 11

/-- The number of tee shirts sold in a day -/
def tees_sold : ℕ := 7

/-- The number of jeans sold in a day -/
def jeans_sold : ℕ := 4

/-- The total revenue for the day -/
def total_revenue : ℝ := 100

theorem tee_price_calculation :
  tee_price * tees_sold + jeans_price * jeans_sold = total_revenue :=
by sorry

end tee_price_calculation_l2940_294001


namespace total_wheels_is_119_l2940_294015

/-- The total number of wheels in Liam's three garages --/
def total_wheels : ℕ :=
  let first_garage := 
    (3 * 2) + 2 + (6 * 3) + (9 * 1) + (3 * 4)
  let second_garage := 
    (2 * 2) + (1 * 3) + (3 * 1) + (4 * 4) + (1 * 5) + 2
  let third_garage := 
    (3 * 2) + (4 * 3) + 1 + 1 + (2 * 4) + (1 * 5) + 7 - 1
  first_garage + second_garage + third_garage

/-- Theorem stating that the total number of wheels in Liam's three garages is 119 --/
theorem total_wheels_is_119 : total_wheels = 119 := by
  sorry

end total_wheels_is_119_l2940_294015


namespace cost_of_type_B_books_l2940_294021

/-- The cost of purchasing type B books given the total number of books and the number of type A books purchased -/
theorem cost_of_type_B_books (total_books : ℕ) (x : ℕ) (price_B : ℕ) 
  (h_total : total_books = 100)
  (h_price : price_B = 6)
  (h_x_le_total : x ≤ total_books) :
  price_B * (total_books - x) = 6 * (100 - x) :=
by sorry

end cost_of_type_B_books_l2940_294021


namespace circle_tangent_triangle_area_l2940_294067

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

/-- Checks if two circles are externally tangent -/
def externallyTangent (c1 c2 : Circle) : Prop := sorry

/-- Checks if a line segment is tangent to a circle -/
def isTangent (p1 p2 : Point) (c : Circle) : Prop := sorry

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

theorem circle_tangent_triangle_area 
  (ω₁ ω₂ ω₃ : Circle)
  (P₁ P₂ P₃ : Point)
  (h_radius : ω₁.radius = 24 ∧ ω₂.radius = 24 ∧ ω₃.radius = 24)
  (h_tangent : externallyTangent ω₁ ω₂ ∧ externallyTangent ω₂ ω₃ ∧ externallyTangent ω₃ ω₁)
  (h_on_circle : onCircle P₁ ω₁ ∧ onCircle P₂ ω₂ ∧ onCircle P₃ ω₃)
  (h_equidistant : distance P₁ P₂ = distance P₂ P₃ ∧ distance P₂ P₃ = distance P₃ P₁)
  (h_tangent_sides : isTangent P₁ P₂ ω₂ ∧ isTangent P₂ P₃ ω₃ ∧ isTangent P₃ P₁ ω₁)
  : ∃ (a b : ℕ), triangleArea P₁ P₂ P₃ = Real.sqrt a + Real.sqrt b ∧ a + b = 288 := by
  sorry

end circle_tangent_triangle_area_l2940_294067


namespace set_A_equals_zero_one_l2940_294003

def A : Set ℤ := {x | (2 * x - 3) / (x + 1) ≤ 0}

theorem set_A_equals_zero_one : A = {0, 1} := by
  sorry

end set_A_equals_zero_one_l2940_294003


namespace episodes_per_season_l2940_294011

theorem episodes_per_season
  (series1_seasons series2_seasons : ℕ)
  (episodes_lost_per_season : ℕ)
  (remaining_episodes : ℕ)
  (h1 : series1_seasons = 12)
  (h2 : series2_seasons = 14)
  (h3 : episodes_lost_per_season = 2)
  (h4 : remaining_episodes = 364) :
  (remaining_episodes + episodes_lost_per_season * (series1_seasons + series2_seasons)) / (series1_seasons + series2_seasons) = 16 := by
sorry

end episodes_per_season_l2940_294011


namespace sum_of_roots_absolute_value_equation_l2940_294066

theorem sum_of_roots_absolute_value_equation : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    (∀ x : ℝ, (|x + 3| - |x - 1| = x + 1) ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) ∧ 
    r₁ + r₂ + r₃ = -3 := by
  sorry

end sum_of_roots_absolute_value_equation_l2940_294066
