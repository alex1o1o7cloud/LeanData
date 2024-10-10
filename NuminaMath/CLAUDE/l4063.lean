import Mathlib

namespace initial_speed_is_50_l4063_406347

/-- Represents the journey with increasing speed -/
structure Journey where
  distance : ℝ  -- Total distance in km
  time : ℝ      -- Total time in hours
  speedIncrease : ℝ  -- Speed increase in km/h
  intervalTime : ℝ   -- Time interval for speed increase in hours

/-- Calculates the initial speed given a journey -/
def calculateInitialSpeed (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the initial speed is 50 km/h -/
theorem initial_speed_is_50 : 
  let j : Journey := {
    distance := 52,
    time := 48 / 60,  -- 48 minutes converted to hours
    speedIncrease := 10,
    intervalTime := 12 / 60  -- 12 minutes converted to hours
  }
  calculateInitialSpeed j = 50 := by
  sorry

end initial_speed_is_50_l4063_406347


namespace bicycle_route_length_l4063_406375

/-- The total length of a rectangular path given the length of one horizontal and one vertical side. -/
def rectangularPathLength (horizontal vertical : ℝ) : ℝ :=
  2 * (horizontal + vertical)

/-- Theorem: The total length of a rectangular path with horizontal sides of 13 km and vertical sides of 13 km is 52 km. -/
theorem bicycle_route_length : rectangularPathLength 13 13 = 52 := by
  sorry

end bicycle_route_length_l4063_406375


namespace vessel_base_length_l4063_406333

/-- Given a cube immersed in a rectangular vessel, this theorem proves the length of the vessel's base. -/
theorem vessel_base_length 
  (cube_edge : ℝ) 
  (vessel_width : ℝ) 
  (water_rise : ℝ) 
  (h1 : cube_edge = 12)
  (h2 : vessel_width = 15)
  (h3 : water_rise = 5.76)
  : ∃ (vessel_length : ℝ), vessel_length = 20 :=
by
  sorry


end vessel_base_length_l4063_406333


namespace world_expo_arrangements_l4063_406343

theorem world_expo_arrangements (n : ℕ) (k : ℕ) :
  n = 7 → k = 3 → (n.choose k) * ((n - k).choose k) = 140 := by
  sorry

end world_expo_arrangements_l4063_406343


namespace kevins_initial_cards_l4063_406379

theorem kevins_initial_cards (found_cards end_cards : ℕ) 
  (h1 : found_cards = 47) 
  (h2 : end_cards = 54) : 
  end_cards - found_cards = 7 := by
  sorry

end kevins_initial_cards_l4063_406379


namespace gcd_168_486_l4063_406330

def continuedProportionateReduction (a b : ℕ) : ℕ :=
  if a = 0 then b
  else if b = 0 then a
  else if a ≥ b then continuedProportionateReduction (a - b) b
  else continuedProportionateReduction a (b - a)

theorem gcd_168_486 :
  continuedProportionateReduction 168 486 = 6 ∧ 
  (∀ d : ℕ, d ∣ 168 ∧ d ∣ 486 → d ≤ 6) := by sorry

end gcd_168_486_l4063_406330


namespace kaleb_toy_purchase_l4063_406356

def number_of_toys (initial_money game_cost saving_amount toy_cost : ℕ) : ℕ :=
  ((initial_money - game_cost - saving_amount) / toy_cost)

theorem kaleb_toy_purchase :
  number_of_toys 12 8 2 2 = 1 := by
  sorry

end kaleb_toy_purchase_l4063_406356


namespace amc10_participation_increase_l4063_406386

def participation : Fin 6 → ℕ
  | 0 => 50  -- 2010
  | 1 => 56  -- 2011
  | 2 => 62  -- 2012
  | 3 => 68  -- 2013
  | 4 => 77  -- 2014
  | 5 => 81  -- 2015

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseBetween2013And2014 : Prop :=
  ∀ i : Fin 5, percentageIncrease (participation i) (participation (i + 1)) ≤
    percentageIncrease (participation 3) (participation 4)

theorem amc10_participation_increase : largestIncreaseBetween2013And2014 := by
  sorry

end amc10_participation_increase_l4063_406386


namespace valid_table_exists_l4063_406311

/-- Represents a geometric property of a shape -/
inductive Property
| HasAcuteAngle
| HasEqualSides
| Property3
| Property4

/-- Represents a geometric shape -/
inductive Shape
| Triangle1
| Triangle2
| Quadrilateral1
| Quadrilateral2

/-- A function that determines if a shape has a property -/
def hasProperty (s : Shape) (p : Property) : Bool :=
  match s, p with
  | Shape.Triangle1, Property.HasAcuteAngle => true
  | Shape.Triangle1, Property.HasEqualSides => false
  | Shape.Triangle2, Property.HasAcuteAngle => true
  | Shape.Triangle2, Property.HasEqualSides => true
  | Shape.Quadrilateral1, Property.HasAcuteAngle => false
  | Shape.Quadrilateral1, Property.HasEqualSides => false
  | Shape.Quadrilateral2, Property.HasAcuteAngle => true
  | Shape.Quadrilateral2, Property.HasEqualSides => false
  | _, _ => false  -- Default case for other combinations

/-- The main theorem stating the existence of a valid table -/
theorem valid_table_exists : ∃ (p3 p4 : Property),
  p3 ≠ Property.HasAcuteAngle ∧ p3 ≠ Property.HasEqualSides ∧
  p4 ≠ Property.HasAcuteAngle ∧ p4 ≠ Property.HasEqualSides ∧ p3 ≠ p4 ∧
  (∀ s : Shape, (hasProperty s Property.HasAcuteAngle).toNat +
                (hasProperty s Property.HasEqualSides).toNat +
                (hasProperty s p3).toNat +
                (hasProperty s p4).toNat = 3) ∧
  (∀ p : Property, (p = Property.HasAcuteAngle ∨ p = Property.HasEqualSides ∨ p = p3 ∨ p = p4) →
    (hasProperty Shape.Triangle1 p).toNat +
    (hasProperty Shape.Triangle2 p).toNat +
    (hasProperty Shape.Quadrilateral1 p).toNat +
    (hasProperty Shape.Quadrilateral2 p).toNat = 3) :=
by sorry

end valid_table_exists_l4063_406311


namespace min_swaps_upper_bound_min_swaps_lower_bound_min_swaps_exact_l4063_406340

/-- A swap operation on a matrix -/
def swap (M : Matrix (Fin n) (Fin n) ℕ) (i j k l : Fin n) : Matrix (Fin n) (Fin n) ℕ :=
  sorry

/-- Predicate to check if a matrix contains all numbers from 1 to n² -/
def valid_matrix (M : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  sorry

/-- The number of swaps needed to transform one matrix into another -/
def swaps_needed (A B : Matrix (Fin n) (Fin n) ℕ) : ℕ :=
  sorry

theorem min_swaps_upper_bound (n : ℕ) (h : n ≥ 2) :
  ∃ m : ℕ, ∀ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A → valid_matrix B →
    swaps_needed A B ≤ m ∧
    m = 2 * n * (n - 1) :=
  sorry

theorem min_swaps_lower_bound (n : ℕ) (h : n ≥ 2) :
  ∃ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A ∧ valid_matrix B ∧
    swaps_needed A B = 2 * n * (n - 1) :=
  sorry

theorem min_swaps_exact (n : ℕ) (h : n ≥ 2) :
  ∃! m : ℕ, (∀ (A B : Matrix (Fin n) (Fin n) ℕ),
    valid_matrix A → valid_matrix B →
    swaps_needed A B ≤ m) ∧
    (∃ (A B : Matrix (Fin n) (Fin n) ℕ),
      valid_matrix A ∧ valid_matrix B ∧
      swaps_needed A B = m) ∧
    m = 2 * n * (n - 1) :=
  sorry

end min_swaps_upper_bound_min_swaps_lower_bound_min_swaps_exact_l4063_406340


namespace cube_side_ratio_l4063_406334

/-- The ratio of side lengths of two cubes with given weights -/
theorem cube_side_ratio (w₁ w₂ : ℝ) (h₁ : w₁ > 0) (h₂ : w₂ > 0) :
  w₁ = 7 → w₂ = 56 → (w₂ / w₁)^(1/3 : ℝ) = 2 := by
  sorry

#check cube_side_ratio

end cube_side_ratio_l4063_406334


namespace cube_volume_increase_l4063_406385

theorem cube_volume_increase (s : ℝ) (h : s > 0) : 
  let original_volume := s^3
  let new_edge_length := 1.4 * s
  let new_volume := new_edge_length^3
  (new_volume - original_volume) / original_volume * 100 = 174.4 := by
sorry

end cube_volume_increase_l4063_406385


namespace not_in_range_iff_b_in_interval_l4063_406301

/-- The function f(x) = x^2 + bx + 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- Theorem stating that -3 is not in the range of f(x) if and only if b is in the open interval (-2√6, 2√6) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x, f b x ≠ -3) ↔ b ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) := by
  sorry

end not_in_range_iff_b_in_interval_l4063_406301


namespace walnut_trees_after_planting_l4063_406309

theorem walnut_trees_after_planting (initial_trees : ℕ) (new_trees : ℕ) :
  initial_trees = 4 → new_trees = 6 → initial_trees + new_trees = 10 := by
  sorry

end walnut_trees_after_planting_l4063_406309


namespace percentage_neither_is_twenty_percent_l4063_406367

/-- Represents the health survey data for teachers -/
structure HealthSurvey where
  total : ℕ
  high_bp : ℕ
  heart_trouble : ℕ
  both : ℕ

/-- Calculates the percentage of teachers with neither high blood pressure nor heart trouble -/
def percentage_neither (survey : HealthSurvey) : ℚ :=
  let neither := survey.total - (survey.high_bp + survey.heart_trouble - survey.both)
  (neither : ℚ) / survey.total * 100

/-- Theorem stating that the percentage of teachers with neither condition is 20% -/
theorem percentage_neither_is_twenty_percent (survey : HealthSurvey)
  (h_total : survey.total = 150)
  (h_high_bp : survey.high_bp = 90)
  (h_heart_trouble : survey.heart_trouble = 60)
  (h_both : survey.both = 30) :
  percentage_neither survey = 20 := by
  sorry

#eval percentage_neither { total := 150, high_bp := 90, heart_trouble := 60, both := 30 }

end percentage_neither_is_twenty_percent_l4063_406367


namespace olly_owns_three_dogs_l4063_406361

/-- The number of shoes needed for each animal -/
def shoes_per_animal : ℕ := 4

/-- The total number of shoes needed -/
def total_shoes : ℕ := 24

/-- The number of cats Olly owns -/
def num_cats : ℕ := 2

/-- The number of ferrets Olly owns -/
def num_ferrets : ℕ := 1

/-- Calculates the number of dogs Olly owns -/
def num_dogs : ℕ :=
  (total_shoes - (num_cats + num_ferrets) * shoes_per_animal) / shoes_per_animal

theorem olly_owns_three_dogs : num_dogs = 3 := by
  sorry

end olly_owns_three_dogs_l4063_406361


namespace distance_between_points_l4063_406307

/-- The distance between two points given specific travel conditions -/
theorem distance_between_points (speed_A speed_B : ℝ) (stop_time : ℝ) : 
  speed_A = 80 →
  speed_B = 70 →
  stop_time = 1/4 →
  ∃ (distance : ℝ), 
    distance / speed_A = distance / speed_B - stop_time ∧
    distance = 2240 := by
  sorry

end distance_between_points_l4063_406307


namespace isosceles_triangle_base_angles_l4063_406371

-- Define an isosceles triangle with one interior angle of 50°
structure IsoscelesTriangle :=
  (base_angle₁ : ℝ)
  (base_angle₂ : ℝ)
  (vertex_angle : ℝ)
  (is_isosceles : base_angle₁ = base_angle₂)
  (has_50_degree_angle : base_angle₁ = 50 ∨ base_angle₂ = 50 ∨ vertex_angle = 50)
  (angle_sum : base_angle₁ + base_angle₂ + vertex_angle = 180)

-- Theorem stating that the base angles are either 50° or 65°
theorem isosceles_triangle_base_angles (t : IsoscelesTriangle) :
  t.base_angle₁ = 50 ∨ t.base_angle₁ = 65 :=
sorry

end isosceles_triangle_base_angles_l4063_406371


namespace triangle_side_length_l4063_406321

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  b = 3 :=
by sorry

end triangle_side_length_l4063_406321


namespace chocolate_problem_l4063_406324

theorem chocolate_problem (total : ℕ) (eaten_with_nuts : ℚ) (left : ℕ) : 
  total = 80 →
  eaten_with_nuts = 4/5 →
  left = 28 →
  (total / 2 - (total / 2 * eaten_with_nuts) - (total - left)) / (total / 2) = 1/2 :=
by sorry

end chocolate_problem_l4063_406324


namespace employed_males_percentage_l4063_406328

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 60)
  (h2 : employed_females_percentage = 25)
  (h3 : total_population > 0) :
  let employed := (employed_percentage / 100) * total_population
  let employed_females := (employed_females_percentage / 100) * employed
  let employed_males := employed - employed_females
  (employed_males / total_population) * 100 = 45 := by
sorry

end employed_males_percentage_l4063_406328


namespace complete_graph_inequality_l4063_406372

/-- Given n points on a plane with no three collinear and some connected by line segments,
    N_k denotes the number of complete graphs of k points. -/
def N (n k : ℕ) : ℕ := sorry

theorem complete_graph_inequality (n : ℕ) (h_n : n > 1) :
  ∀ k ∈ Finset.range (n - 1) \ {0, 1},
  N n k ≠ 0 →
  (N n (k + 1) : ℝ) / (N n k) ≥ 
    (1 : ℝ) / ((k^2 : ℝ) - 1) * ((k^2 : ℝ) * (N n k) / (N n (k + 1)) - n) := by
  sorry

end complete_graph_inequality_l4063_406372


namespace rhombus_diagonal_roots_l4063_406373

theorem rhombus_diagonal_roots (m : ℝ) : 
  let side_length : ℝ := 5
  let quadratic (x : ℝ) := x^2 + (2*m - 1)*x + m^2 + 3
  ∃ (OA OB : ℝ), 
    OA^2 + OB^2 = side_length^2 ∧ 
    quadratic OA = 0 ∧ 
    quadratic OB = 0 →
    m = -3 :=
by sorry

end rhombus_diagonal_roots_l4063_406373


namespace fraction_equality_l4063_406313

theorem fraction_equality (a b : ℝ) (h : a / b = 1 / 2) : a / (a + b) = 1 / 3 := by
  sorry

end fraction_equality_l4063_406313


namespace robert_reading_capacity_l4063_406395

/-- The number of full books Robert can read in a given time -/
def books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours : ℕ) : ℕ :=
  (pages_per_hour * hours) / pages_per_book

/-- Theorem: Robert can read 2 full 360-page books in 8 hours at 120 pages per hour -/
theorem robert_reading_capacity : books_read 120 360 8 = 2 := by
  sorry

end robert_reading_capacity_l4063_406395


namespace pythagorean_triple_identification_l4063_406341

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_identification :
  ¬(is_pythagorean_triple 1 2 3) ∧
  ¬(is_pythagorean_triple 4 5 6) ∧
  ¬(is_pythagorean_triple 6 8 9) ∧
  is_pythagorean_triple 7 24 25 :=
by sorry

end pythagorean_triple_identification_l4063_406341


namespace john_video_release_l4063_406310

/-- Calculates the total minutes of video released per week by John --/
def total_video_minutes_per_week (short_video_length : ℕ) (long_video_multiplier : ℕ) (short_videos_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  let long_video_length := short_video_length * long_video_multiplier
  let total_minutes_per_day := short_video_length * short_videos_per_day + long_video_length
  total_minutes_per_day * days_per_week

/-- Theorem stating that John releases 112 minutes of video per week --/
theorem john_video_release : 
  total_video_minutes_per_week 2 6 2 7 = 112 := by
  sorry


end john_video_release_l4063_406310


namespace sam_pennies_l4063_406350

theorem sam_pennies (initial_pennies final_pennies : ℕ) 
  (h1 : initial_pennies = 98) 
  (h2 : final_pennies = 191) : 
  final_pennies - initial_pennies = 93 := by
  sorry

end sam_pennies_l4063_406350


namespace max_perimeter_of_divided_isosceles_triangle_l4063_406399

/-- The maximum perimeter of a piece when an isosceles triangle is divided into four equal areas -/
theorem max_perimeter_of_divided_isosceles_triangle :
  let base : ℝ := 12
  let height : ℝ := 15
  let segment_length : ℝ := base / 4
  let perimeter (k : ℝ) : ℝ := segment_length + Real.sqrt (height^2 + k^2) + Real.sqrt (height^2 + (k + 1)^2)
  let max_perimeter : ℝ := perimeter 2
  max_perimeter = 3 + Real.sqrt 229 + Real.sqrt 234 := by
  sorry

end max_perimeter_of_divided_isosceles_triangle_l4063_406399


namespace orthogonality_condition_l4063_406317

/-- Two circles are orthogonal if their tangents at intersection points are perpendicular -/
def orthogonal (R₁ R₂ d : ℝ) : Prop :=
  d^2 = R₁^2 + R₂^2

theorem orthogonality_condition (R₁ R₂ d : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : d > 0) :
  orthogonal R₁ R₂ d ↔ d^2 = R₁^2 + R₂^2 :=
sorry

end orthogonality_condition_l4063_406317


namespace subset_if_intersection_eq_elem_of_union_if_elem_of_intersection_fraction_inequality_necessary_not_sufficient_exists_non_positive_square_l4063_406383

-- Statement 1
theorem subset_if_intersection_eq (A B : Set α) : A ∩ B = A → A ⊆ B := by sorry

-- Statement 2
theorem elem_of_union_if_elem_of_intersection {A B : Set α} {x : α} :
  x ∈ A ∩ B → x ∈ A ∪ B := by sorry

-- Statement 3
theorem fraction_inequality_necessary_not_sufficient {a b : ℝ} :
  (a < b ∧ b < 0) → b / a < a / b := by sorry

-- Statement 4
theorem exists_non_positive_square : ∃ x : ℤ, x^2 ≤ 0 := by sorry

end subset_if_intersection_eq_elem_of_union_if_elem_of_intersection_fraction_inequality_necessary_not_sufficient_exists_non_positive_square_l4063_406383


namespace ten_power_plus_eight_div_nine_is_integer_l4063_406353

theorem ten_power_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, (10^n : ℤ) + 8 = 9 * k := by
  sorry

end ten_power_plus_eight_div_nine_is_integer_l4063_406353


namespace imaginary_part_of_one_plus_i_squared_l4063_406360

theorem imaginary_part_of_one_plus_i_squared (i : ℂ) : 
  i^2 = -1 → Complex.im ((1 + i)^2) = 2 := by
  sorry

end imaginary_part_of_one_plus_i_squared_l4063_406360


namespace mat_cost_per_square_meter_l4063_406348

/-- Calculates the cost per square meter of mat for a rectangular hall -/
theorem mat_cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (h_length : length = 20) 
  (h_width : width = 15) 
  (h_height : height = 5) 
  (h_expenditure : total_expenditure = 38000) : 
  total_expenditure / (length * width + 2 * (length * height + width * height)) = 58.46 := by
  sorry

end mat_cost_per_square_meter_l4063_406348


namespace grocery_store_salary_l4063_406396

/-- Calculates the total daily salary of employees in a grocery store. -/
def total_daily_salary (manager_salary : ℕ) (clerk_salary : ℕ) (num_managers : ℕ) (num_clerks : ℕ) : ℕ :=
  manager_salary * num_managers + clerk_salary * num_clerks

/-- Proves that the total daily salary of all employees in the grocery store is $16. -/
theorem grocery_store_salary : total_daily_salary 5 2 2 3 = 16 := by
  sorry

end grocery_store_salary_l4063_406396


namespace complement_intersection_theorem_l4063_406314

def U : Set ℕ := {x | x ≤ 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {4, 5, 6}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end complement_intersection_theorem_l4063_406314


namespace arithmetic_mean_of_S_l4063_406338

/-- The set of numbers from 9 to 999999999, where each number consists of all 9s -/
def S : Finset ℕ := Finset.image (λ i => (10^i - 1) / 9) (Finset.range 9)

/-- The arithmetic mean of the set S -/
def M : ℕ := (Finset.sum S id) / Finset.card S

theorem arithmetic_mean_of_S : M = 123456789 := by sorry

end arithmetic_mean_of_S_l4063_406338


namespace green_blue_difference_l4063_406377

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : Nat
  ratio : Fin 3 → Nat
  colorCount : DiskColor → Nat

/-- The theorem stating the difference between green and blue disks -/
theorem green_blue_difference (bag : DiskBag) 
  (h_total : bag.total = 54)
  (h_ratio : bag.ratio = ![3, 7, 8])
  (h_color_count : ∀ c, bag.colorCount c = (bag.total / (bag.ratio 0 + bag.ratio 1 + bag.ratio 2)) * match c with
    | DiskColor.Blue => bag.ratio 0
    | DiskColor.Yellow => bag.ratio 1
    | DiskColor.Green => bag.ratio 2) :
  bag.colorCount DiskColor.Green - bag.colorCount DiskColor.Blue = 15 := by
  sorry

end green_blue_difference_l4063_406377


namespace read_distance_guangzhou_shenyang_l4063_406357

/-- Represents a number in words -/
inductive NumberWord
  | million : ℕ → NumberWord
  | thousand : ℕ → NumberWord
  | hundred : ℕ → NumberWord
  | ten : ℕ → NumberWord
  | one : ℕ → NumberWord

/-- Represents the distance from Guangzhou to Shenyang in meters -/
def distance_guangzhou_shenyang : ℕ := 3036000

/-- Converts a natural number to its word representation -/
def number_to_words (n : ℕ) : List NumberWord :=
  sorry

/-- Theorem stating that the correct way to read 3,036,000 is "three million thirty-six thousand" -/
theorem read_distance_guangzhou_shenyang :
  number_to_words distance_guangzhou_shenyang = 
    [NumberWord.million 3, NumberWord.thousand 36] :=
  sorry

end read_distance_guangzhou_shenyang_l4063_406357


namespace steps_to_top_floor_l4063_406304

/-- The number of steps between each floor in the building -/
def steps_between_floors : ℕ := 13

/-- The total number of floors in the building -/
def total_floors : ℕ := 7

/-- The number of intervals between floors when going from ground to top floor -/
def floor_intervals : ℕ := total_floors - 1

/-- The total number of steps from ground floor to the top floor -/
def total_steps : ℕ := steps_between_floors * floor_intervals

theorem steps_to_top_floor :
  total_steps = 78 :=
sorry

end steps_to_top_floor_l4063_406304


namespace john_average_speed_l4063_406358

-- Define the start time, break time, end time, and total distance
def start_time : ℕ := 8 * 60 + 15  -- 8:15 AM in minutes
def break_start : ℕ := 12 * 60  -- 12:00 PM in minutes
def break_duration : ℕ := 30  -- 30 minutes
def end_time : ℕ := 14 * 60 + 45  -- 2:45 PM in minutes
def total_distance : ℕ := 240  -- miles

-- Calculate the total driving time in hours
def total_driving_time : ℚ :=
  (break_start - start_time + (end_time - (break_start + break_duration))) / 60

-- Define the average speed
def average_speed : ℚ := total_distance / total_driving_time

-- Theorem to prove
theorem john_average_speed :
  average_speed = 40 :=
sorry

end john_average_speed_l4063_406358


namespace complex_fraction_simplification_l4063_406351

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (5 + 7 * i) / (2 + 3 * i) = 31 / 13 - (1 / 13) * i :=
by sorry

end complex_fraction_simplification_l4063_406351


namespace one_row_with_ten_seats_l4063_406327

/-- Represents the seating arrangement in a theater --/
structure TheaterSeating where
  total_people : ℕ
  rows_with_ten : ℕ
  rows_with_nine : ℕ

/-- Checks if the seating arrangement is valid --/
def is_valid_seating (s : TheaterSeating) : Prop :=
  s.total_people = 55 ∧
  s.rows_with_ten * 10 + s.rows_with_nine * 9 = s.total_people

/-- Theorem stating that there is exactly one row seating 10 people --/
theorem one_row_with_ten_seats :
  ∃! s : TheaterSeating, is_valid_seating s ∧ s.rows_with_ten = 1 :=
sorry

end one_row_with_ten_seats_l4063_406327


namespace tan_alpha_value_l4063_406302

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 1/2) : Real.tan α = -1/3 := by
  sorry

end tan_alpha_value_l4063_406302


namespace binary_1010_is_10_l4063_406381

/-- Converts a binary digit (0 or 1) to its decimal value -/
def binaryToDecimal (digit : Nat) : Nat :=
  if digit = 0 ∨ digit = 1 then digit else 0

/-- Converts a list of binary digits to its decimal representation -/
def binaryListToDecimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + binaryToDecimal b * 2^i) 0

theorem binary_1010_is_10 :
  binaryListToDecimal [0, 1, 0, 1] = 10 := by
  sorry

end binary_1010_is_10_l4063_406381


namespace isosceles_triangle_exists_unique_l4063_406390

/-- An isosceles triangle with given heights -/
structure IsoscelesTriangle where
  /-- Height corresponding to the base -/
  m_a : ℝ
  /-- Height corresponding to one of the equal sides -/
  m_b : ℝ
  /-- Condition for existence -/
  h : 2 * m_a > m_b

/-- Theorem stating the existence and uniqueness of an isosceles triangle with given heights -/
theorem isosceles_triangle_exists_unique (m_a m_b : ℝ) :
  Nonempty (Unique (IsoscelesTriangle)) ↔ 2 * m_a > m_b :=
sorry

end isosceles_triangle_exists_unique_l4063_406390


namespace cookie_ratio_l4063_406369

theorem cookie_ratio (monday tuesday wednesday : ℕ) : 
  monday = 32 →
  tuesday = monday / 2 →
  monday + tuesday + (wednesday - 4) = 92 →
  wednesday / tuesday = 3 := by
  sorry

end cookie_ratio_l4063_406369


namespace circle_area_ratio_l4063_406397

theorem circle_area_ratio (R_A R_B : ℝ) (h : R_A > 0 ∧ R_B > 0) :
  (60 : ℝ) / 360 * (2 * Real.pi * R_A) = (40 : ℝ) / 360 * (2 * Real.pi * R_B) →
  (R_A^2 * Real.pi) / (R_B^2 * Real.pi) = 4 / 9 := by
  sorry

end circle_area_ratio_l4063_406397


namespace dinnerCostTheorem_l4063_406346

/-- Represents the cost breakdown of a dinner -/
structure DinnerCost where
  preTax : ℝ
  taxRate : ℝ
  tipRate : ℝ
  total : ℝ

/-- The combined pre-tax cost of two dinners -/
def combinedPreTaxCost (d1 d2 : DinnerCost) : ℝ :=
  d1.preTax + d2.preTax

/-- Calculates the total cost of a dinner including tax and tip -/
def calculateTotal (d : DinnerCost) : ℝ :=
  d.preTax * (1 + d.taxRate + d.tipRate)

theorem dinnerCostTheorem (johnDinner sarahDinner : DinnerCost) :
  johnDinner.taxRate = 0.12 →
  johnDinner.tipRate = 0.16 →
  sarahDinner.taxRate = 0.09 →
  sarahDinner.tipRate = 0.10 →
  johnDinner.total = 35.20 →
  sarahDinner.total = 22.00 →
  calculateTotal johnDinner = johnDinner.total →
  calculateTotal sarahDinner = sarahDinner.total →
  combinedPreTaxCost johnDinner sarahDinner = 46 := by
  sorry

#eval 46  -- This line is added to ensure the statement can be built successfully

end dinnerCostTheorem_l4063_406346


namespace marks_trees_l4063_406365

theorem marks_trees (current_trees : ℕ) 
  (h : current_trees + 12 = 25) : current_trees = 13 := by
  sorry

end marks_trees_l4063_406365


namespace line_x_intercept_l4063_406306

/-- The x-intercept of a straight line passing through points (2, -4) and (6, 8) is 10/3 -/
theorem line_x_intercept :
  let p1 : ℝ × ℝ := (2, -4)
  let p2 : ℝ × ℝ := (6, 8)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let x_intercept : ℝ := -b / m
  x_intercept = 10 / 3 := by
  sorry

end line_x_intercept_l4063_406306


namespace rectangular_hall_dimensions_l4063_406308

theorem rectangular_hall_dimensions (length width : ℝ) (area : ℝ) : 
  width = length / 2 →
  area = length * width →
  area = 200 →
  length - width = 10 := by
sorry

end rectangular_hall_dimensions_l4063_406308


namespace circle_C_properties_l4063_406388

/-- The circle C passing through A(4,1) and tangent to x-y-1=0 at B(2,1) -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 4}

/-- Point A -/
def point_A : ℝ × ℝ := (4, 1)

/-- Point B -/
def point_B : ℝ × ℝ := (2, 1)

/-- The line x-y-1=0 -/
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

theorem circle_C_properties :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  tangent_line point_B ∧
  ∀ (p : ℝ × ℝ), p ∈ circle_C ∧ tangent_line p → p = point_B :=
by sorry

end circle_C_properties_l4063_406388


namespace arcsin_neg_sqrt3_over_2_l4063_406382

theorem arcsin_neg_sqrt3_over_2 : Real.arcsin (-Real.sqrt 3 / 2) = -π / 3 := by
  sorry

end arcsin_neg_sqrt3_over_2_l4063_406382


namespace kayla_apples_l4063_406370

theorem kayla_apples (total : ℕ) (kayla kylie : ℕ) : 
  total = 200 →
  kayla + kylie = total →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end kayla_apples_l4063_406370


namespace geometric_series_relation_l4063_406337

/-- Given two infinite geometric series with the specified conditions, prove that n = 20/3 -/
theorem geometric_series_relation (a₁ b₁ a₂ b₂ n : ℝ) : 
  a₁ = 15 ∧ b₁ = 5 ∧ a₂ = 15 ∧ b₂ = 5 + n ∧ 
  (a₁ / (1 - b₁ / a₁)) * 3 = a₂ / (1 - b₂ / a₂) → 
  n = 20 / 3 := by
sorry

end geometric_series_relation_l4063_406337


namespace quadratic_equation_complete_square_l4063_406316

theorem quadratic_equation_complete_square :
  ∃ (r s : ℝ), 
    (∀ x, 15 * x^2 - 60 * x - 135 = 0 ↔ (x + r)^2 = s) ∧
    r + s = 7 := by
  sorry

end quadratic_equation_complete_square_l4063_406316


namespace sum_345_75_base6_l4063_406329

/-- Converts a natural number from base 10 to base 6 -/
def toBase6 (n : ℕ) : List ℕ :=
  sorry

/-- Adds two numbers in base 6 -/
def addBase6 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the sum of 345 and 75 in base 6 is 1540 -/
theorem sum_345_75_base6 :
  addBase6 (toBase6 345) (toBase6 75) = [1, 5, 4, 0] :=
sorry

end sum_345_75_base6_l4063_406329


namespace angle_d_measure_l4063_406315

/-- Given a triangle ABC with angles A = 85°, B = 34°, and C = 21°,
    if a smaller triangle is formed within ABC with one of its angles being D,
    then the measure of angle D is 140°. -/
theorem angle_d_measure (A B C D : Real) : 
  A = 85 → B = 34 → C = 21 → 
  A + B + C = 180 →
  ∃ (E F : Real), E ≥ 0 ∧ F ≥ 0 ∧ D + E + F = 180 ∧ A + B + C + E + F = 180 →
  D = 140 := by sorry

end angle_d_measure_l4063_406315


namespace segment_area_equilateral_triangle_l4063_406323

/-- The area of a circular segment cut off by one side of an equilateral triangle inscribed in a circle -/
theorem segment_area_equilateral_triangle (a : ℝ) (ha : a > 0) :
  let R := a / Real.sqrt 3
  let sector_area := π * R^2 / 3
  let triangle_area := a * R / 4
  sector_area - triangle_area = (a^2 * (4 * π - 3 * Real.sqrt 3)) / 36 := by
  sorry

end segment_area_equilateral_triangle_l4063_406323


namespace filter_price_calculation_l4063_406320

/-- Proves that the price of each of the remaining 2 filters is $22.55 -/
theorem filter_price_calculation (kit_price : ℝ) (filter1_price : ℝ) (filter2_price : ℝ) 
  (discount_percentage : ℝ) :
  kit_price = 72.50 →
  filter1_price = 12.45 →
  filter2_price = 11.50 →
  discount_percentage = 0.1103448275862069 →
  ∃ (x : ℝ), 
    x = 22.55 ∧
    kit_price = (1 - discount_percentage) * (2 * filter1_price + 2 * x + filter2_price) := by
  sorry

end filter_price_calculation_l4063_406320


namespace birth_death_rate_decisive_l4063_406349

/-- Represents the various characteristics of a population -/
inductive PopulationCharacteristic
  | Density
  | AgeComposition
  | SexRatio
  | BirthRate
  | DeathRate
  | ImmigrationRate
  | EmigrationRate

/-- Represents the impact of a characteristic on population size change -/
inductive Impact
  | Decisive
  | Indirect
  | Basic

/-- Function that maps a population characteristic to its impact on population size change -/
def characteristicImpact : PopulationCharacteristic → Impact
  | PopulationCharacteristic.Density => Impact.Basic
  | PopulationCharacteristic.AgeComposition => Impact.Indirect
  | PopulationCharacteristic.SexRatio => Impact.Indirect
  | PopulationCharacteristic.BirthRate => Impact.Decisive
  | PopulationCharacteristic.DeathRate => Impact.Decisive
  | PopulationCharacteristic.ImmigrationRate => Impact.Decisive
  | PopulationCharacteristic.EmigrationRate => Impact.Decisive

theorem birth_death_rate_decisive :
  ∀ c : PopulationCharacteristic,
    characteristicImpact c = Impact.Decisive →
    c = PopulationCharacteristic.BirthRate ∨ c = PopulationCharacteristic.DeathRate :=
by sorry

end birth_death_rate_decisive_l4063_406349


namespace quadratic_equation_root_values_l4063_406339

theorem quadratic_equation_root_values (a : ℝ) : 
  (∃ x : ℂ, x^2 - 2*a*x + a^2 - 4*a = 0 ∧ Complex.abs x = 3) →
  (a = 1 ∨ a = 9 ∨ a = 2 - Real.sqrt 13) :=
by sorry

end quadratic_equation_root_values_l4063_406339


namespace complex_division_real_l4063_406359

def complex (a b : ℝ) : ℂ := a + b * Complex.I

theorem complex_division_real (b : ℝ) :
  let z₁ : ℂ := complex 3 (-b)
  let z₂ : ℂ := complex 1 (-2)
  (∃ (r : ℝ), z₁ / z₂ = r) → b = 6 := by sorry

end complex_division_real_l4063_406359


namespace fgh_supermarket_difference_l4063_406322

theorem fgh_supermarket_difference (total : ℕ) (us : ℕ) (h1 : total = 84) (h2 : us = 47) (h3 : us > total - us) : us - (total - us) = 10 := by
  sorry

end fgh_supermarket_difference_l4063_406322


namespace product_of_numbers_l4063_406300

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 11) 
  (h2 : x^2 + y^2 = 185) : 
  x * y = 26 := by
sorry

end product_of_numbers_l4063_406300


namespace eight_spotlights_illuminate_space_l4063_406391

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a spotlight that can illuminate an octant -/
structure Spotlight where
  position : Point3D
  direction : Point3D  -- Normalized vector representing the direction

/-- Represents the space to be illuminated -/
def Space : Type := Unit

/-- Checks if a spotlight illuminates a given point in space -/
def illuminates (s : Spotlight) (p : Point3D) : Prop := sorry

/-- Checks if a set of spotlights illuminates the entire space -/
def illuminatesEntireSpace (spotlights : Finset Spotlight) : Prop := 
  ∀ p : Point3D, ∃ s ∈ spotlights, illuminates s p

/-- The main theorem stating that 8 spotlights can illuminate the entire space -/
theorem eight_spotlights_illuminate_space 
  (points : Finset Point3D) 
  (h : points.card = 8) : 
  ∃ spotlights : Finset Spotlight, 
    spotlights.card = 8 ∧ 
    (∀ s ∈ spotlights, ∃ p ∈ points, s.position = p) ∧
    illuminatesEntireSpace spotlights := by
  sorry

end eight_spotlights_illuminate_space_l4063_406391


namespace buy_three_items_ways_l4063_406366

/-- The number of headphones available for sale. -/
def headphones : ℕ := 9

/-- The number of computer mice available for sale. -/
def mice : ℕ := 13

/-- The number of keyboards available for sale. -/
def keyboards : ℕ := 5

/-- The number of "keyboard and mouse" sets available. -/
def keyboard_mouse_sets : ℕ := 4

/-- The number of "headphones and mouse" sets available. -/
def headphones_mouse_sets : ℕ := 5

/-- The total number of ways to buy three items: headphones, a keyboard, and a mouse. -/
def total_ways : ℕ := 646

/-- Theorem stating that the total number of ways to buy three items
    (headphones, keyboard, and mouse) is 646. -/
theorem buy_three_items_ways :
  headphones * keyboard_mouse_sets +
  keyboards * headphones_mouse_sets +
  headphones * mice * keyboards = total_ways := by
  sorry

end buy_three_items_ways_l4063_406366


namespace triangle_abc_properties_l4063_406384

theorem triangle_abc_properties (A B C : Real) (a b : Real) (S : Real) :
  A = 30 * Real.pi / 180 →
  B = 45 * Real.pi / 180 →
  a = Real.sqrt 2 →
  b = a * Real.sin B / Real.sin A →
  C = Real.pi - A - B →
  S = 1/2 * a * b * Real.sin C →
  b = 2 ∧ S = (Real.sqrt 3 + 1) / 2 := by
  sorry

end triangle_abc_properties_l4063_406384


namespace value_of_expression_l4063_406336

theorem value_of_expression (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by sorry

end value_of_expression_l4063_406336


namespace total_rainfall_is_correct_l4063_406398

-- Define conversion factors
def inch_to_cm : ℝ := 2.54
def mm_to_cm : ℝ := 0.1

-- Define daily rainfall measurements
def monday_rain : ℝ := 0.12962962962962962
def tuesday_rain : ℝ := 3.5185185185185186
def wednesday_rain : ℝ := 0.09259259259259259
def thursday_rain : ℝ := 0.10222222222222223
def friday_rain : ℝ := 12.222222222222221
def saturday_rain : ℝ := 0.2222222222222222
def sunday_rain : ℝ := 0.17444444444444446

-- Define the units for each day's measurement
inductive RainUnit
| Centimeter
| Millimeter
| Inch

def monday_unit : RainUnit := RainUnit.Centimeter
def tuesday_unit : RainUnit := RainUnit.Millimeter
def wednesday_unit : RainUnit := RainUnit.Centimeter
def thursday_unit : RainUnit := RainUnit.Inch
def friday_unit : RainUnit := RainUnit.Millimeter
def saturday_unit : RainUnit := RainUnit.Centimeter
def sunday_unit : RainUnit := RainUnit.Inch

-- Function to convert a measurement to centimeters based on its unit
def to_cm (measurement : ℝ) (unit : RainUnit) : ℝ :=
  match unit with
  | RainUnit.Centimeter => measurement
  | RainUnit.Millimeter => measurement * mm_to_cm
  | RainUnit.Inch => measurement * inch_to_cm

-- Theorem statement
theorem total_rainfall_is_correct : 
  to_cm monday_rain monday_unit +
  to_cm tuesday_rain tuesday_unit +
  to_cm wednesday_rain wednesday_unit +
  to_cm thursday_rain thursday_unit +
  to_cm friday_rain friday_unit +
  to_cm saturday_rain saturday_unit +
  to_cm sunday_rain sunday_unit = 2.721212629851652 := by
  sorry

end total_rainfall_is_correct_l4063_406398


namespace equation_solution_l4063_406319

theorem equation_solution (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y - 3 * z = 0)
  (eq2 : x + 5 * y - 12 * z = 0)
  (z_neq_0 : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 2*z^2) = -1053/1547 := by
  sorry

end equation_solution_l4063_406319


namespace expression_simplification_l4063_406305

theorem expression_simplification (k : ℚ) :
  (6 * k + 12) / 6 = k + 2 ∧
  ∃ (a b : ℤ), k + 2 = a * k + b ∧ a = 1 ∧ b = 2 ∧ a / b = 1 / 2 := by
  sorry

end expression_simplification_l4063_406305


namespace number_problem_l4063_406362

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → (40/100 : ℝ) * N = 180 := by
  sorry

end number_problem_l4063_406362


namespace hyperbola_asymptote_angle_implies_m_range_l4063_406335

/-- Given a hyperbola with equation x² + y²/m = 1, if the asymptote's inclination angle α 
    is in the interval (0, π/3), then m is in the interval (-3, 0). -/
theorem hyperbola_asymptote_angle_implies_m_range (m : ℝ) (α : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2/m = 1 → ∃ k : ℝ, y = k*x ∧ Real.arctan k = α) →
  0 < α ∧ α < π/3 →
  -3 < m ∧ m < 0 :=
sorry

end hyperbola_asymptote_angle_implies_m_range_l4063_406335


namespace y_completion_time_l4063_406344

/-- The time Y takes to complete the entire work alone, given:
  * X can do the entire work in 40 days
  * X works for 8 days
  * Y finishes the remaining work in 20 days
-/
theorem y_completion_time (x_total_days : ℕ) (x_worked_days : ℕ) (y_completion_days : ℕ) :
  x_total_days = 40 →
  x_worked_days = 8 →
  y_completion_days = 20 →
  (x_worked_days : ℚ) / x_total_days + (y_completion_days : ℚ) * (1 - (x_worked_days : ℚ) / x_total_days) = 1 →
  25 = (1 / (1 / y_completion_days * (1 - (x_worked_days : ℚ) / x_total_days))) := by
  sorry

end y_completion_time_l4063_406344


namespace similar_canister_capacity_l4063_406352

/-- Given that a small canister with volume 24 cm³ can hold 100 nails,
    prove that a similar canister with volume 72 cm³ can hold 300 nails,
    assuming the nails are packed in the same manner. -/
theorem similar_canister_capacity
  (small_volume : ℝ)
  (small_nails : ℕ)
  (large_volume : ℝ)
  (h1 : small_volume = 24)
  (h2 : small_nails = 100)
  (h3 : large_volume = 72)
  (h4 : small_volume > 0)
  (h5 : large_volume > 0) :
  (large_volume / small_volume) * small_nails = 300 := by
  sorry

#check similar_canister_capacity

end similar_canister_capacity_l4063_406352


namespace water_bottle_cost_l4063_406326

/-- Given Barbara's shopping information, prove the cost of each water bottle -/
theorem water_bottle_cost
  (tuna_packs : ℕ)
  (tuna_cost_per_pack : ℚ)
  (water_bottles : ℕ)
  (total_spent : ℚ)
  (different_goods_cost : ℚ)
  (h1 : tuna_packs = 5)
  (h2 : tuna_cost_per_pack = 2)
  (h3 : water_bottles = 4)
  (h4 : total_spent = 56)
  (h5 : different_goods_cost = 40) :
  (total_spent - different_goods_cost - tuna_packs * tuna_cost_per_pack) / water_bottles = 1.5 := by
  sorry

end water_bottle_cost_l4063_406326


namespace g_of_3_equals_6_l4063_406392

-- Define the function g
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Theorem statement
theorem g_of_3_equals_6 : g 3 = 6 := by
  sorry

end g_of_3_equals_6_l4063_406392


namespace sum_mod_five_l4063_406387

theorem sum_mod_five : (9375 + 9376 + 9377 + 9378) % 5 = 1 := by
  sorry

end sum_mod_five_l4063_406387


namespace spider_total_distance_l4063_406393

def spider_movement (start : ℤ) (first_move : ℤ) (second_move : ℤ) : ℕ :=
  (Int.natAbs (first_move - start)) + (Int.natAbs (second_move - first_move))

theorem spider_total_distance :
  spider_movement 3 (-4) 8 = 19 := by
  sorry

end spider_total_distance_l4063_406393


namespace pure_imaginary_condition_l4063_406342

theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) / (1 + a * Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = -1 := by
  sorry

end pure_imaginary_condition_l4063_406342


namespace max_value_theorem_l4063_406394

theorem max_value_theorem (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  ∀ a b c : ℝ, 9 * a^2 + 4 * b^2 + 25 * c^2 = 1 → c = 2 * b → 
  10 * x + 3 * y + 12 * z ≥ 10 * a + 3 * b + 12 * c ∧
  ∃ x₀ y₀ z₀ : ℝ, 9 * x₀^2 + 4 * y₀^2 + 25 * z₀^2 = 1 ∧ z₀ = 2 * y₀ ∧
  10 * x₀ + 3 * y₀ + 12 * z₀ = Real.sqrt 253 :=
by sorry

end max_value_theorem_l4063_406394


namespace sum_six_equals_twentyfour_l4063_406355

/-- An arithmetic sequence {a_n} with sum S_n -/
structure ArithmeticSequence where
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, S (n + 2) - S (n + 1) = S (n + 1) - S n

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.S n

theorem sum_six_equals_twentyfour (seq : ArithmeticSequence) 
  (h2 : sum_n seq 2 = 2) 
  (h4 : sum_n seq 4 = 10) : 
  sum_n seq 6 = 24 := by
  sorry

end sum_six_equals_twentyfour_l4063_406355


namespace club_officer_selection_l4063_406364

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members : ℕ) (experienced_members : ℕ) : ℕ :=
  experienced_members * (experienced_members - 1) * (total_members - 2) * (total_members - 3)

/-- Theorem stating the number of ways to choose officers in the given club scenario --/
theorem club_officer_selection :
  let total_members : ℕ := 12
  let experienced_members : ℕ := 4
  choose_officers total_members experienced_members = 1080 := by
  sorry

end club_officer_selection_l4063_406364


namespace sum_of_fractions_inequality_l4063_406312

theorem sum_of_fractions_inequality (x y z : ℝ) 
  (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + 
  (1 + y^2) / (1 + z + x^2) + 
  (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end sum_of_fractions_inequality_l4063_406312


namespace certain_number_problem_l4063_406363

theorem certain_number_problem (y : ℝ) : 0.5 * 10 = 0.05 * y - 20 → y = 500 := by
  sorry

end certain_number_problem_l4063_406363


namespace P_greater_than_Q_l4063_406378

theorem P_greater_than_Q (a : ℝ) (h : a > -38) :
  Real.sqrt (a + 40) - Real.sqrt (a + 41) > Real.sqrt (a + 38) - Real.sqrt (a + 39) := by
sorry

end P_greater_than_Q_l4063_406378


namespace circle_bisection_l4063_406368

/-- Given two circles in the plane:
    Circle 1: (x-a)^2 + (y-b)^2 = b^2 + 1
    Circle 2: (x+1)^2 + (y+1)^2 = 4
    If Circle 1 always bisects the circumference of Circle 2,
    then the relationship between a and b satisfies: a^2 + 2a + 2b + 5 = 0 -/
theorem circle_bisection (a b : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4 → 
    ∃ t : ℝ, x = -1 + t * (2 + 2*a) ∧ y = -1 + t * (2 + 2*b)) → 
  a^2 + 2*a + 2*b + 5 = 0 :=
sorry

end circle_bisection_l4063_406368


namespace valid_numbers_l4063_406303

def is_valid_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, 
    x ≤ 9 ∧ y ≤ 9 ∧
    n = 3000000 + x * 10000 + y * 100 + 3 ∧
    n % 13 = 0

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = 
  {3020303, 3050203, 3080103, 3090503, 3060603, 3030703, 3000803} := by
sorry

end valid_numbers_l4063_406303


namespace negation_of_universal_proposition_l4063_406354

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l4063_406354


namespace president_vice_president_count_l4063_406331

/-- The number of ways to select a president and vice president from 5 people -/
def president_vice_president_selections : ℕ := 20

/-- The number of people to choose from -/
def total_people : ℕ := 5

/-- The number of positions to fill -/
def positions_to_fill : ℕ := 2

theorem president_vice_president_count :
  president_vice_president_selections = total_people * (total_people - 1) :=
sorry

end president_vice_president_count_l4063_406331


namespace knicks_equivalence_l4063_406374

/-- Conversion rate between knicks and knacks -/
def knicks_to_knacks : ℚ := 3 / 8

/-- Conversion rate between knacks and knocks -/
def knacks_to_knocks : ℚ := 6 / 5

/-- The number of knicks equivalent to 30 knocks -/
def knicks_for_30_knocks : ℚ := 30 * (1 / knacks_to_knocks) * (1 / knicks_to_knacks)

theorem knicks_equivalence :
  knicks_for_30_knocks = 200 / 3 :=
sorry

end knicks_equivalence_l4063_406374


namespace candy_count_l4063_406389

/-- Given the total number of treats, chewing gums, and chocolate bars,
    prove that the number of candies of different flavors is 40. -/
theorem candy_count (total_treats chewing_gums chocolate_bars : ℕ) 
  (h1 : total_treats = 155)
  (h2 : chewing_gums = 60)
  (h3 : chocolate_bars = 55) :
  total_treats - (chewing_gums + chocolate_bars) = 40 := by
  sorry

end candy_count_l4063_406389


namespace cubic_root_sum_cubes_over_product_l4063_406325

theorem cubic_root_sum_cubes_over_product (p q a b c : ℝ) : 
  q ≠ 0 → 
  (∀ x : ℝ, x^3 + p*x + q = (x-a)*(x-b)*(x-c)) → 
  (a^3 + b^3 + c^3) / (a*b*c) = 3 := by
sorry

end cubic_root_sum_cubes_over_product_l4063_406325


namespace isabella_currency_exchange_l4063_406318

theorem isabella_currency_exchange :
  ∃ d : ℕ+, 
    (8 * d.val : ℚ) / 5 - 80 = d.val ∧ 
    (d.val / 100 + (d.val % 100) / 10 + d.val % 10 = 9) := by
  sorry

end isabella_currency_exchange_l4063_406318


namespace trig_identity_l4063_406332

theorem trig_identity (α β : ℝ) : 
  Real.sin α ^ 2 + Real.sin β ^ 2 - Real.sin α ^ 2 * Real.sin β ^ 2 + Real.cos α ^ 2 * Real.cos β ^ 2 = 1 := by
  sorry

end trig_identity_l4063_406332


namespace discount_calculation_l4063_406345

def calculate_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1) * (1 - discount2)

theorem discount_calculation (hat_price tie_price : ℝ) 
  (hat_discount1 hat_discount2 tie_discount1 tie_discount2 : ℝ) : 
  hat_price = 20 → tie_price = 15 → 
  hat_discount1 = 0.25 → hat_discount2 = 0.20 → 
  tie_discount1 = 0.10 → tie_discount2 = 0.30 → 
  calculate_final_price hat_price hat_discount1 hat_discount2 = 12 ∧ 
  calculate_final_price tie_price tie_discount1 tie_discount2 = 9.45 := by
  sorry

#check discount_calculation

end discount_calculation_l4063_406345


namespace watch_cost_price_l4063_406380

theorem watch_cost_price (C : ℝ) : 
  (C * 0.9 = C * (1 - 0.1)) →  -- Selling at 90% of C is a 10% loss
  (C * 1.03 = C * (1 + 0.03)) →  -- Selling at 103% of C is a 3% gain
  (C * 1.03 - C * 0.9 = 140) →  -- Difference between selling prices is 140
  C = 1076.92 := by
sorry

end watch_cost_price_l4063_406380


namespace melanie_turnips_count_l4063_406376

/-- The number of turnips Benny grew -/
def benny_turnips : ℕ := 113

/-- The total number of turnips Melanie and Benny grew together -/
def total_turnips : ℕ := 252

/-- The number of turnips Melanie grew -/
def melanie_turnips : ℕ := total_turnips - benny_turnips

theorem melanie_turnips_count : melanie_turnips = 139 := by
  sorry

end melanie_turnips_count_l4063_406376
