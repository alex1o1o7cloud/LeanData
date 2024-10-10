import Mathlib

namespace inscribed_cylinder_radius_l1932_193284

/-- 
A right circular cylinder is inscribed in a right circular cone.
The cylinder's diameter equals its height.
The cone has a diameter of 8 and an altitude of 10.
The axes of the cylinder and the cone coincide.
-/
theorem inscribed_cylinder_radius (r : ℝ) : r = 20 / 9 :=
  let cone_diameter := 8
  let cone_altitude := 10
  let cylinder_height := 2 * r
  -- The cylinder's diameter equals its height
  have h1 : cylinder_height = 2 * r := rfl
  -- The cone has a diameter of 8 and an altitude of 10
  have h2 : cone_diameter = 8 := rfl
  have h3 : cone_altitude = 10 := rfl
  -- The axes of the cylinder and the cone coincide (implicit in the problem setup)
  sorry


end inscribed_cylinder_radius_l1932_193284


namespace optimal_sampling_methods_l1932_193233

/-- Represents different sampling methods -/
inductive SamplingMethod
| Random
| Systematic
| Stratified

/-- Represents income levels -/
inductive IncomeLevel
| High
| Middle
| Low

/-- Represents a community with different income levels -/
structure Community where
  total_households : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population_size : ℕ
  sample_size : ℕ
  has_distinct_strata : Bool

/-- Determines the optimal sampling method for a given scenario -/
def optimal_sampling_method (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The community described in the problem -/
def problem_community : Community :=
  { total_households := 1000
  , high_income := 250
  , middle_income := 560
  , low_income := 190 }

/-- The household study sampling scenario -/
def household_study : SamplingScenario :=
  { population_size := 1000
  , sample_size := 200
  , has_distinct_strata := true }

/-- The discussion forum sampling scenario -/
def discussion_forum : SamplingScenario :=
  { population_size := 20
  , sample_size := 6
  , has_distinct_strata := false }

theorem optimal_sampling_methods :
  optimal_sampling_method household_study = SamplingMethod.Stratified ∧
  optimal_sampling_method discussion_forum = SamplingMethod.Random :=
sorry

end optimal_sampling_methods_l1932_193233


namespace greatest_two_digit_with_product_12_l1932_193217

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem greatest_two_digit_with_product_12 :
  ∃ (n : ℕ), is_two_digit n ∧ 
             digits_product n = 12 ∧
             (∀ (m : ℕ), is_two_digit m → digits_product m = 12 → m ≤ n) ∧
             n = 62 :=
by sorry

end greatest_two_digit_with_product_12_l1932_193217


namespace range_of_a_l1932_193283

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/2 : ℝ) 1 :=
sorry

end range_of_a_l1932_193283


namespace intersection_A_complement_B_l1932_193258

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end intersection_A_complement_B_l1932_193258


namespace marble_count_l1932_193245

theorem marble_count (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r := by sorry

end marble_count_l1932_193245


namespace runner_parade_time_l1932_193281

/-- Calculates the time taken for a runner to travel from the front to the end of a moving parade. -/
theorem runner_parade_time (parade_length : ℝ) (parade_speed : ℝ) (runner_speed : ℝ) 
  (h1 : parade_length = 2)
  (h2 : parade_speed = 3)
  (h3 : runner_speed = 6) :
  (parade_length / (runner_speed - parade_speed)) * 60 = 40 := by
  sorry

end runner_parade_time_l1932_193281


namespace philips_banana_groups_l1932_193225

theorem philips_banana_groups :
  let total_bananas : ℕ := 392
  let bananas_per_group : ℕ := 2
  let num_groups : ℕ := total_bananas / bananas_per_group
  num_groups = 196 := by
  sorry

end philips_banana_groups_l1932_193225


namespace knight_reachability_l1932_193298

/-- Represents a position on an infinite chessboard -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Represents a knight's move -/
def knight_move (p : Position) : Set Position :=
  {q : Position | (abs (q.x - p.x) = 2 ∧ abs (q.y - p.y) = 1) ∨
                  (abs (q.x - p.x) = 1 ∧ abs (q.y - p.y) = 2)}

/-- Represents the set of positions a knight can reach in exactly n moves -/
def reachable_in (start : Position) (n : ℕ) : Set Position :=
  match n with
  | 0 => {start}
  | n + 1 => ⋃ p ∈ reachable_in start n, knight_move p

/-- Represents whether a position is a black square -/
def is_black (p : Position) : Prop :=
  (p.x + p.y) % 2 = 0

/-- Represents the region described in the problem -/
def target_region (n : ℕ) (start : Position) : Set Position :=
  {p : Position | max (abs (p.x - start.x)) (abs (p.y - start.y)) ≤ 4*n + 1 ∧
                  abs (p.x - start.x) + abs (p.y - start.y) > 2*n}

/-- The main theorem to prove -/
theorem knight_reachability (n : ℕ) (start : Position) :
  ∀ p ∈ target_region n start, is_black p → p ∈ reachable_in start (2*n) :=
sorry

end knight_reachability_l1932_193298


namespace cattle_train_speed_l1932_193214

/-- The speed of the cattle train in mph -/
def cattle_speed : ℝ := 56

/-- The time difference in hours between the cattle train's departure and the diesel train's departure -/
def time_difference : ℝ := 6

/-- The duration in hours that the diesel train traveled -/
def diesel_travel_time : ℝ := 12

/-- The speed difference in mph between the cattle train and the diesel train -/
def speed_difference : ℝ := 33

/-- The total distance in miles between the two trains after the diesel train's travel -/
def total_distance : ℝ := 1284

theorem cattle_train_speed :
  cattle_speed * (time_difference + diesel_travel_time) +
  (cattle_speed - speed_difference) * diesel_travel_time = total_distance :=
sorry

end cattle_train_speed_l1932_193214


namespace middle_elementary_students_l1932_193289

theorem middle_elementary_students (total : ℕ) 
  (h_total : total = 12000)
  (h_elementary : (15 : ℚ) / 16 * total = upper_elementary + middle_elementary)
  (h_not_upper : (1 : ℚ) / 2 * total = junior_high + middle_elementary)
  (h_groups : total = junior_high + upper_elementary + middle_elementary) :
  middle_elementary = 4875 := by
  sorry

end middle_elementary_students_l1932_193289


namespace tenth_term_geometric_sequence_l1932_193249

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r ^ (n - 1)

theorem tenth_term_geometric_sequence :
  geometric_sequence 4 (5/3) 10 = 7812500/19683 := by
  sorry

end tenth_term_geometric_sequence_l1932_193249


namespace friday_to_thursday_ratio_is_two_to_one_l1932_193200

/-- Represents the daily sales of ground beef in kilograms -/
structure DailySales where
  thursday : ℝ
  friday : ℝ
  saturday : ℝ
  sunday : ℝ

/-- Theorem stating the ratio of Friday to Thursday sales is 2:1 -/
theorem friday_to_thursday_ratio_is_two_to_one (sales : DailySales) : 
  sales.thursday = 210 →
  sales.saturday = 130 →
  sales.sunday = sales.saturday / 2 →
  sales.thursday + sales.friday + sales.saturday + sales.sunday = 825 →
  sales.friday / sales.thursday = 2 := by
  sorry

#check friday_to_thursday_ratio_is_two_to_one

end friday_to_thursday_ratio_is_two_to_one_l1932_193200


namespace megan_deleted_files_l1932_193290

/-- Calculates the number of deleted files given the initial number of files,
    the number of folders after organizing, and the number of files per folder. -/
def deleted_files (initial_files : ℕ) (num_folders : ℕ) (files_per_folder : ℕ) : ℕ :=
  initial_files - (num_folders * files_per_folder)

/-- Proves that Megan deleted 21 files given the problem conditions. -/
theorem megan_deleted_files :
  deleted_files 93 9 8 = 21 := by
  sorry

end megan_deleted_files_l1932_193290


namespace fraction_multiplication_l1932_193236

theorem fraction_multiplication : (2 : ℚ) / 3 * 4 / 7 * 9 / 11 = 24 / 77 := by
  sorry

end fraction_multiplication_l1932_193236


namespace prime_divisor_property_l1932_193271

theorem prime_divisor_property (n k : ℕ) (h1 : n > 1) 
  (h2 : ∀ d : ℕ, d ∣ n → (d + k) ∣ n ∨ (d - k) ∣ n) : 
  Nat.Prime n :=
sorry

end prime_divisor_property_l1932_193271


namespace integer_roots_of_polynomial_l1932_193224

def polynomial (x : ℤ) : ℤ := x^3 - 5*x^2 - 8*x + 24

def is_root (x : ℤ) : Prop := polynomial x = 0

theorem integer_roots_of_polynomial :
  (∀ x : ℤ, is_root x ↔ (x = -3 ∨ x = 2 ∨ x = 4)) ∨
  (∀ x : ℤ, ¬is_root x) := by sorry

end integer_roots_of_polynomial_l1932_193224


namespace remainder_proof_l1932_193264

theorem remainder_proof : ∃ r : ℕ, r < 33 ∧ r < 8 ∧ 266 % 33 = r ∧ 266 % 8 = r :=
by
  -- The proof goes here
  sorry

end remainder_proof_l1932_193264


namespace unique_pair_sum_28_l1932_193252

theorem unique_pair_sum_28 :
  ∃! (a b : ℕ), a ≠ b ∧ a > 11 ∧ b > 11 ∧ a + b = 28 ∧
  (Even a ∨ Even b) ∧
  (∀ (c d : ℕ), c ≠ d ∧ c > 11 ∧ d > 11 ∧ c + d = 28 ∧ (Even c ∨ Even d) → (c = a ∧ d = b) ∨ (c = b ∧ d = a)) ∧
  a = 12 ∧ b = 16 :=
by sorry

end unique_pair_sum_28_l1932_193252


namespace parallel_planes_from_skew_parallel_lines_l1932_193250

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_from_skew_parallel_lines 
  (m n : Line) (α β : Plane) :
  skew m n →
  parallel_line_plane m α →
  parallel_line_plane n α →
  parallel_line_plane m β →
  parallel_line_plane n β →
  parallel_plane_plane α β :=
sorry

end parallel_planes_from_skew_parallel_lines_l1932_193250


namespace polynomial_simplification_l1932_193253

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 = 
  6*x^3 - x^2 + 23*x - 3 := by
sorry

end polynomial_simplification_l1932_193253


namespace spade_calculation_l1932_193213

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : (spade 8 5) + (spade 3 (spade 6 2)) = 4 := by
  sorry

end spade_calculation_l1932_193213


namespace spelling_bee_points_l1932_193204

theorem spelling_bee_points : 
  let max_points : ℝ := 7
  let dulce_points : ℝ := 5
  let val_points : ℝ := 4 * (max_points + dulce_points)
  let sarah_points : ℝ := 2 * dulce_points
  let steve_points : ℝ := 2.5 * (max_points + val_points)
  let team_points : ℝ := max_points + dulce_points + val_points + sarah_points + steve_points
  let opponents_points : ℝ := 200
  team_points - opponents_points = 7.5 := by sorry

end spelling_bee_points_l1932_193204


namespace sequence_non_positive_l1932_193269

theorem sequence_non_positive (N : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hN : a N = 0) 
  (h_rec : ∀ i ∈ Finset.range (N - 1), 
    a (i + 2) - 2 * a (i + 1) + a i = (a (i + 1))^2) :
  ∀ i ∈ Finset.range (N - 1), a (i + 1) ≤ 0 := by
  sorry

end sequence_non_positive_l1932_193269


namespace circle_coloring_exists_l1932_193246

/-- A point on a circle --/
structure CirclePoint where
  angle : Real

/-- A color (red or blue) --/
inductive Color
  | Red
  | Blue

/-- A coloring function for points on a circle --/
def ColoringFunction := CirclePoint → Color

/-- Predicate to check if three points form a right-angled triangle inscribed in the circle --/
def IsRightTriangle (p1 p2 p3 : CirclePoint) : Prop :=
  -- We assume this predicate exists and is correctly defined
  sorry

theorem circle_coloring_exists :
  ∃ (f : ColoringFunction),
    ∀ (p1 p2 p3 : CirclePoint),
      IsRightTriangle p1 p2 p3 →
        (f p1 ≠ f p2) ∨ (f p1 ≠ f p3) ∨ (f p2 ≠ f p3) :=
by
  sorry

end circle_coloring_exists_l1932_193246


namespace carrot_to_lettuce_ratio_l1932_193207

def lettuce_calories : ℕ := 50
def dressing_calories : ℕ := 210
def pizza_crust_calories : ℕ := 600
def pizza_cheese_calories : ℕ := 400
def total_consumed_calories : ℕ := 330

def pizza_total_calories : ℕ := pizza_crust_calories + (pizza_crust_calories / 3) + pizza_cheese_calories

def salad_calories (carrot_calories : ℕ) : ℕ := lettuce_calories + carrot_calories + dressing_calories

theorem carrot_to_lettuce_ratio :
  ∃ (carrot_calories : ℕ),
    (salad_calories carrot_calories / 4 + pizza_total_calories / 5 = total_consumed_calories) ∧
    (carrot_calories / lettuce_calories = 2) := by
  sorry

end carrot_to_lettuce_ratio_l1932_193207


namespace sixth_number_divisible_by_45_and_6_l1932_193291

/-- The least common multiple of 45 and 6 -/
def lcm_45_6 : ℕ := 90

/-- The first multiple of lcm_45_6 greater than 190 -/
def first_multiple : ℕ := 270

/-- The ending number we want to prove -/
def ending_number : ℕ := 720

/-- The theorem to prove -/
theorem sixth_number_divisible_by_45_and_6 : 
  ending_number = first_multiple + 5 * lcm_45_6 ∧ 
  ending_number % 45 = 0 ∧ 
  ending_number % 6 = 0 ∧
  ∀ n : ℕ, first_multiple ≤ n ∧ n < ending_number ∧ n % 45 = 0 ∧ n % 6 = 0 → 
    ∃ k : ℕ, k < 6 ∧ n = first_multiple + k * lcm_45_6 :=
by sorry

end sixth_number_divisible_by_45_and_6_l1932_193291


namespace original_mixture_volume_l1932_193260

/-- Proves that given a mixture with 20% alcohol, if adding 2 litres of water
    results in a new mixture with 17.647058823529413% alcohol,
    then the original mixture volume was 15 litres. -/
theorem original_mixture_volume
  (original_alcohol_percentage : Real)
  (added_water : Real)
  (new_alcohol_percentage : Real)
  (h1 : original_alcohol_percentage = 0.20)
  (h2 : added_water = 2)
  (h3 : new_alcohol_percentage = 0.17647058823529413)
  : ∃ (original_volume : Real),
    original_volume * original_alcohol_percentage /
    (original_volume + added_water) = new_alcohol_percentage ∧
    original_volume = 15 := by
  sorry

end original_mixture_volume_l1932_193260


namespace stratified_sampling_equal_probability_l1932_193206

/-- Represents a stratified sampling setup -/
structure StratifiedSampling where
  population : Type
  strata : Type
  num_layers : ℕ
  stratification : population → strata

/-- The probability of an individual being sampled in stratified sampling -/
def sample_probability (ss : StratifiedSampling) (individual : ss.population) : ℝ :=
  sorry

/-- Theorem stating that the sample probability is independent of the number of layers and stratification -/
theorem stratified_sampling_equal_probability (ss : StratifiedSampling) 
  (individual1 individual2 : ss.population) :
  sample_probability ss individual1 = sample_probability ss individual2 :=
sorry

end stratified_sampling_equal_probability_l1932_193206


namespace smallest_piece_length_l1932_193237

/-- Given a rod of length 120 cm cut into three pieces proportional to 3, 5, and 7,
    the length of the smallest piece is 24 cm. -/
theorem smallest_piece_length :
  let total_length : ℝ := 120
  let ratio_sum : ℝ := 3 + 5 + 7
  let smallest_ratio : ℝ := 3
  smallest_ratio * (total_length / ratio_sum) = 24 := by
  sorry

end smallest_piece_length_l1932_193237


namespace max_perfect_squares_pairwise_products_l1932_193228

theorem max_perfect_squares_pairwise_products (a b : ℕ) (h : a ≠ b) :
  let products := {a * (a + 2), b * (b + 2), a * b, a * (b + 2), (a + 2) * b, (a + 2) * (b + 2)}
  (∃ (s : Finset ℕ), s ⊆ products ∧ (∀ x ∈ s, ∃ y : ℕ, x = y ^ 2) ∧ s.card = 2) ∧
  (∀ (s : Finset ℕ), s ⊆ products → (∀ x ∈ s, ∃ y : ℕ, x = y ^ 2) → s.card ≤ 2) :=
sorry

end max_perfect_squares_pairwise_products_l1932_193228


namespace triangle_reconstruction_possible_l1932_193257

-- Define the basic types and structures
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the given points
variable (X Y Z : Point)

-- Define the properties of the given points
def is_circumcenter (X : Point) (t : Triangle) : Prop := sorry

def is_midpoint (Y : Point) (B C : Point) : Prop := sorry

def is_altitude_foot (Z : Point) (B A C : Point) : Prop := sorry

-- State the theorem
theorem triangle_reconstruction_possible 
  (h_circumcenter : ∃ t : Triangle, is_circumcenter X t)
  (h_midpoint : ∃ B C : Point, is_midpoint Y B C)
  (h_altitude_foot : ∃ A B C : Point, is_altitude_foot Z B A C) :
  ∃! t : Triangle, 
    is_circumcenter X t ∧ 
    is_midpoint Y t.B t.C ∧ 
    is_altitude_foot Z t.B t.A t.C :=
sorry

end triangle_reconstruction_possible_l1932_193257


namespace forgotten_digit_probability_l1932_193272

theorem forgotten_digit_probability : 
  let total_digits : ℕ := 10
  let max_attempts : ℕ := 2
  let favorable_outcomes : ℕ := (total_digits - 1) + (total_digits - 1)
  let total_outcomes : ℕ := total_digits * (total_digits - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 5 := by
  sorry

end forgotten_digit_probability_l1932_193272


namespace class_visual_conditions_most_comprehensive_l1932_193276

/-- Represents a survey option -/
inductive SurveyOption
| LightTubes
| ClassVisualConditions
| NationwideExerciseTime
| FoodPigmentContent

/-- Defines characteristics of a survey -/
structure SurveyCharacteristics where
  population_size : ℕ
  geographical_spread : Bool
  data_collection_feasibility : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive (s : SurveyCharacteristics) : Prop :=
  s.population_size ≤ 100 ∧ ¬s.geographical_spread ∧ s.data_collection_feasibility

/-- Assigns characteristics to each survey option -/
def survey_characteristics : SurveyOption → SurveyCharacteristics
| SurveyOption.LightTubes => ⟨50, false, false⟩
| SurveyOption.ClassVisualConditions => ⟨30, false, true⟩
| SurveyOption.NationwideExerciseTime => ⟨1000000, true, false⟩
| SurveyOption.FoodPigmentContent => ⟨500, true, false⟩

/-- Theorem stating that investigating visual conditions of a class is the most suitable for a comprehensive survey -/
theorem class_visual_conditions_most_comprehensive :
  ∀ (s : SurveyOption), s ≠ SurveyOption.ClassVisualConditions →
  is_comprehensive (survey_characteristics SurveyOption.ClassVisualConditions) ∧
  ¬(is_comprehensive (survey_characteristics s)) :=
by sorry

end class_visual_conditions_most_comprehensive_l1932_193276


namespace higher_interest_rate_theorem_l1932_193222

/-- Given a principal amount, two interest rates, and a time period,
    calculate the difference in interest earned between the two rates. -/
def interest_difference (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  principal * rate1 * time - principal * rate2 * time

theorem higher_interest_rate_theorem (R : ℝ) :
  interest_difference 5000 (R / 100) (12 / 100) 2 = 600 → R = 18 := by
  sorry

end higher_interest_rate_theorem_l1932_193222


namespace virginia_eggs_remaining_l1932_193286

theorem virginia_eggs_remaining (initial_eggs : ℕ) (eggs_taken : ℕ) : 
  initial_eggs = 200 → eggs_taken = 37 → initial_eggs - eggs_taken = 163 :=
by
  sorry

end virginia_eggs_remaining_l1932_193286


namespace amusement_park_initial_cost_l1932_193220

/-- The initial cost to open an amusement park, given the conditions described in the problem. -/
def initial_cost : ℝ → Prop := λ C =>
  let daily_running_cost := 0.01 * C
  let daily_revenue := 1500
  let days_to_breakeven := 200
  C = days_to_breakeven * (daily_revenue - daily_running_cost)

/-- Theorem stating that the initial cost to open the amusement park is $100,000. -/
theorem amusement_park_initial_cost :
  ∃ C : ℝ, initial_cost C ∧ C = 100000 := by
  sorry

end amusement_park_initial_cost_l1932_193220


namespace coin_flip_configurations_l1932_193227

theorem coin_flip_configurations (n : ℕ) (h : n = 10) : 
  (Finset.range n).card + (n.choose 2) = 46 := by
  sorry

end coin_flip_configurations_l1932_193227


namespace sin_alpha_value_l1932_193277

theorem sin_alpha_value (α β : ℝ) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin α = 3 * Real.sqrt 10 / 10 := by
  sorry

end sin_alpha_value_l1932_193277


namespace total_books_in_class_l1932_193231

theorem total_books_in_class (num_tables : ℕ) (books_per_table_ratio : ℚ) : 
  num_tables = 500 →
  books_per_table_ratio = 2 / 5 →
  (num_tables : ℚ) * books_per_table_ratio * num_tables = 100000 :=
by sorry

end total_books_in_class_l1932_193231


namespace quadrilateral_properties_l1932_193274

/-- A quadrilateral with coordinates of its four vertices -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The diagonals of a quadrilateral -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.A.1 - q.C.1, q.A.2 - q.C.2), (q.B.1 - q.D.1, q.B.2 - q.D.2))

/-- Check if the diagonals of a quadrilateral are equal -/
def equal_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  d1.1^2 + d1.2^2 = d2.1^2 + d2.2^2

/-- Check if the diagonals of a quadrilateral bisect each other -/
def diagonals_bisect (q : Quadrilateral) : Prop :=
  let mid1 := ((q.A.1 + q.C.1) / 2, (q.A.2 + q.C.2) / 2)
  let mid2 := ((q.B.1 + q.D.1) / 2, (q.B.2 + q.D.2) / 2)
  mid1 = mid2

/-- Check if the diagonals of a quadrilateral are perpendicular -/
def perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let (d1, d2) := diagonals q
  d1.1 * d2.1 + d1.2 * d2.2 = 0

/-- Check if all sides of a quadrilateral are equal -/
def equal_sides (q : Quadrilateral) : Prop :=
  let side1 := (q.A.1 - q.B.1)^2 + (q.A.2 - q.B.2)^2
  let side2 := (q.B.1 - q.C.1)^2 + (q.B.2 - q.C.2)^2
  let side3 := (q.C.1 - q.D.1)^2 + (q.C.2 - q.D.2)^2
  let side4 := (q.D.1 - q.A.1)^2 + (q.D.2 - q.A.2)^2
  side1 = side2 ∧ side2 = side3 ∧ side3 = side4

/-- A quadrilateral is a rectangle -/
def is_rectangle (q : Quadrilateral) : Prop :=
  equal_diagonals q ∧ diagonals_bisect q

/-- A quadrilateral is a rhombus -/
def is_rhombus (q : Quadrilateral) : Prop :=
  equal_sides q

/-- A quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop :=
  equal_diagonals q ∧ perpendicular_diagonals q

theorem quadrilateral_properties :
  (∀ q : Quadrilateral, equal_diagonals q ∧ diagonals_bisect q → is_rectangle q) ∧
  ¬(∀ q : Quadrilateral, perpendicular_diagonals q → is_rhombus q) ∧
  (∀ q : Quadrilateral, equal_diagonals q ∧ perpendicular_diagonals q → is_square q) ∧
  (∀ q : Quadrilateral, equal_sides q → is_rhombus q) :=
sorry

end quadrilateral_properties_l1932_193274


namespace ratio_of_terms_l1932_193219

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequences where
  a : ℕ → ℚ  -- First sequence
  b : ℕ → ℚ  -- Second sequence
  S : ℕ → ℚ  -- Sum of first n terms of a
  T : ℕ → ℚ  -- Sum of first n terms of b
  h_arithmetic_a : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  h_arithmetic_b : ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n
  h_sum_a : ∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2
  h_sum_b : ∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2
  h_ratio : ∀ n : ℕ, S n / T n = (2 * n : ℚ) / (3 * n + 1)

/-- Main theorem: If the ratio of sums is given, then a_5 / b_6 = 9 / 17 -/
theorem ratio_of_terms (seq : ArithmeticSequences) : seq.a 5 / seq.b 6 = 9 / 17 := by
  sorry

end ratio_of_terms_l1932_193219


namespace coronavirus_size_scientific_notation_l1932_193270

/-- The size of a novel coronavirus in meters -/
def coronavirus_size : ℝ := 0.000000125

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Convert a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation := sorry

theorem coronavirus_size_scientific_notation :
  to_scientific_notation coronavirus_size = ScientificNotation.mk 1.25 (-7) := by sorry

end coronavirus_size_scientific_notation_l1932_193270


namespace percentage_calculation_l1932_193235

theorem percentage_calculation (P : ℝ) : 
  (0.16 * (P / 100) * 93.75 = 6) → P = 40 := by
  sorry

end percentage_calculation_l1932_193235


namespace johnson_family_seating_l1932_193238

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem johnson_family_seating (num_boys num_girls : ℕ) 
  (h1 : num_boys = 5) 
  (h2 : num_girls = 4) 
  (h3 : num_boys + num_girls = 9) : 
  factorial (num_boys + num_girls) - factorial num_boys * factorial num_girls = 359760 :=
sorry

end johnson_family_seating_l1932_193238


namespace max_value_constraint_l1932_193265

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  x + 2*y + 2*z ≤ 15 := by
  sorry

end max_value_constraint_l1932_193265


namespace inequality_range_real_inequality_range_unit_interval_l1932_193230

-- Define the inequality function
def inequality (k x : ℝ) : Prop :=
  (k * x^2 + k * x + 4) / (x^2 + x + 1) > 1

-- Theorem for the first part of the problem
theorem inequality_range_real : 
  (∀ x : ℝ, inequality k x) ↔ k ∈ Set.Icc 1 13 := by sorry

-- Theorem for the second part of the problem
theorem inequality_range_unit_interval :
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → inequality k x) ↔ k ∈ Set.Ioi (-1/2) := by sorry

end inequality_range_real_inequality_range_unit_interval_l1932_193230


namespace tangent_line_equation_l1932_193210

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x - 2*y + 2*Real.log 2 - 1 = 0) :=
by sorry

end tangent_line_equation_l1932_193210


namespace largest_perfect_square_factor_3402_l1932_193223

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  sorry

theorem largest_perfect_square_factor_3402 :
  largest_perfect_square_factor 3402 = 81 := by sorry

end largest_perfect_square_factor_3402_l1932_193223


namespace sqrt_24_plus_3_bounds_l1932_193294

theorem sqrt_24_plus_3_bounds :
  (4 < Real.sqrt 24) ∧ (Real.sqrt 24 < 5) →
  (7 < Real.sqrt 24 + 3) ∧ (Real.sqrt 24 + 3 < 8) :=
by sorry

end sqrt_24_plus_3_bounds_l1932_193294


namespace functional_equation_solution_l1932_193215

-- Define the function type
def ContinuousFunction (α : Type*) := α → ℝ

-- State the theorem
theorem functional_equation_solution
  (f : ContinuousFunction ℝ)
  (h_cont : Continuous f)
  (h_domain : ∀ x : ℝ, x > 0 → f x ≠ 0)
  (h_eq : ∀ x y : ℝ, x > 0 → y > 0 →
    f (x + 1/x) + f (y + 1/y) = f (x + 1/y) + f (y + 1/x)) :
  ∃ c d : ℝ, ∀ x : ℝ, x > 0 → f x = c * x + d :=
sorry

end functional_equation_solution_l1932_193215


namespace grid_paths_l1932_193208

theorem grid_paths (total_steps : ℕ) (right_steps : ℕ) (up_steps : ℕ) 
  (h1 : total_steps = right_steps + up_steps)
  (h2 : total_steps = 10)
  (h3 : right_steps = 6)
  (h4 : up_steps = 4) :
  Nat.choose total_steps up_steps = 210 := by
  sorry

end grid_paths_l1932_193208


namespace cosine_amplitude_and_shift_l1932_193288

/-- Given a cosine function that oscillates between 5 and 1, prove its amplitude and vertical shift. -/
theorem cosine_amplitude_and_shift (a b c d : ℝ) : 
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  a = 2 ∧ d = 3 := by
  sorry

end cosine_amplitude_and_shift_l1932_193288


namespace point_equidistant_from_origin_and_A_l1932_193211

/-- Given a point P(x, y) that is 17 units away from both the origin O(0,0) and point A(16,0),
    prove that the coordinates of P must be either (8, 15) or (8, -15). -/
theorem point_equidistant_from_origin_and_A : ∀ x y : ℝ,
  (x^2 + y^2 = 17^2) →
  ((x - 16)^2 + y^2 = 17^2) →
  ((x = 8 ∧ y = 15) ∨ (x = 8 ∧ y = -15)) :=
by sorry

end point_equidistant_from_origin_and_A_l1932_193211


namespace base6_120_to_base2_l1932_193280

/-- Converts a number from base 6 to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 10 to base 2 --/
def base10ToBase2 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

theorem base6_120_to_base2 :
  base10ToBase2 (base6ToBase10 120) = [1, 1, 0, 0, 0, 0] := by
  sorry

end base6_120_to_base2_l1932_193280


namespace gcd_180_270_l1932_193254

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcd_180_270_l1932_193254


namespace geometric_sequence_ratio_l1932_193242

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : a 1 + a 2 + a 3 = 1)
  (h3 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 9) : 
  q = 2 := by
sorry

end geometric_sequence_ratio_l1932_193242


namespace min_value_theorem_l1932_193259

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x-1)^2 + (y-3)^2 = 4

-- Define the distance function
def dist_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₁ - x₂)^2 + (y₁ - y₂)^2

-- Define the condition |PC₁| = |PC₂|
def point_condition (a b : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧
  dist_squared a b x₁ y₁ = dist_squared a b x₂ y₂

-- Define the expression to be minimized
def expr_to_minimize (a b : ℝ) : ℝ := a^2 + b^2 - 6*a - 4*b + 13

-- State the theorem
theorem min_value_theorem :
  ∃ (min : ℝ), min = 8/5 ∧
  ∀ (a b : ℝ), point_condition a b → expr_to_minimize a b ≥ min :=
by sorry

end min_value_theorem_l1932_193259


namespace chucks_team_score_l1932_193287

theorem chucks_team_score (yellow_team_score lead : ℕ) 
  (h1 : yellow_team_score = 55)
  (h2 : lead = 17) :
  yellow_team_score + lead = 72 := by
  sorry

end chucks_team_score_l1932_193287


namespace limit_of_a_sequence_l1932_193232

def a (n : ℕ) : ℚ := n / (3 * n - 1)

theorem limit_of_a_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 1/3| < ε :=
sorry

end limit_of_a_sequence_l1932_193232


namespace food_supply_problem_l1932_193297

/-- Represents the food supply problem -/
theorem food_supply_problem (initial_men : ℕ) (additional_men : ℕ) (days_after_joining : ℕ) :
  initial_men = 760 →
  additional_men = 3040 →
  days_after_joining = 4 →
  ∃ (initial_days : ℕ),
    initial_days * initial_men = 
      (initial_days - 2) * initial_men + 
      days_after_joining * (initial_men + additional_men) ∧
    initial_days = 22 :=
by
  sorry

end food_supply_problem_l1932_193297


namespace three_boys_three_girls_arrangements_l1932_193267

/-- The number of possible arrangements for 3 boys and 3 girls in an alternating pattern -/
def alternating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  2 * (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of arrangements for 3 boys and 3 girls is 72 -/
theorem three_boys_three_girls_arrangements :
  alternating_arrangements 3 3 = 72 := by
  sorry

end three_boys_three_girls_arrangements_l1932_193267


namespace polygon_with_150_degree_angles_is_12_gon_l1932_193248

theorem polygon_with_150_degree_angles_is_12_gon (n : ℕ) 
  (h : n ≥ 3) 
  (interior_angle : ℝ) 
  (h_angle : interior_angle = 150) 
  (h_sum : (n - 2) * 180 = n * interior_angle) : n = 12 := by
sorry

end polygon_with_150_degree_angles_is_12_gon_l1932_193248


namespace sum_of_coefficients_l1932_193229

theorem sum_of_coefficients (g h i j k : ℤ) : 
  (∀ y : ℝ, 1000 * y^3 + 27 = (g * y + h) * (i * y^2 + j * y + k)) →
  g + h + i + j + k = 92 := by
sorry

end sum_of_coefficients_l1932_193229


namespace base_ten_and_twelve_satisfy_conditions_l1932_193295

/-- Represents a number in a given base -/
def NumberInBase (n : ℕ) (base : ℕ) : ℕ := n

/-- Checks if a number is even in a given base -/
def IsEvenInBase (n : ℕ) (base : ℕ) : Prop :=
  ∃ k : ℕ, NumberInBase n base = 2 * k

/-- Checks if three numbers are consecutive in a given base -/
def AreConsecutiveInBase (a b c : ℕ) (base : ℕ) : Prop :=
  NumberInBase b base = NumberInBase a base + 1 ∧
  NumberInBase c base = NumberInBase b base + 1

/-- The main theorem to prove -/
theorem base_ten_and_twelve_satisfy_conditions :
  (NumberInBase 24 10 = NumberInBase 4 10 * NumberInBase 6 10 ∧
   ∃ a b c : ℕ, AreConsecutiveInBase a b c 10 ∧
   (IsEvenInBase a 10 ∧ IsEvenInBase b 10 ∧ IsEvenInBase c 10 ∨
    ¬IsEvenInBase a 10 ∧ ¬IsEvenInBase b 10 ∧ ¬IsEvenInBase c 10)) ∧
  (NumberInBase 24 12 = NumberInBase 4 12 * NumberInBase 6 12 ∧
   ∃ a b c : ℕ, AreConsecutiveInBase a b c 12 ∧
   (IsEvenInBase a 12 ∧ IsEvenInBase b 12 ∧ IsEvenInBase c 12 ∨
    ¬IsEvenInBase a 12 ∧ ¬IsEvenInBase b 12 ∧ ¬IsEvenInBase c 12)) :=
by sorry


end base_ten_and_twelve_satisfy_conditions_l1932_193295


namespace parabola_transformation_l1932_193212

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ

/-- Shifts a parabola vertically -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { equation := λ x => p.equation x + shift }

/-- Shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { equation := λ x => p.equation (x - shift) }

theorem parabola_transformation (p : Parabola) (h : p.equation = λ x => 2 * x^2) :
  (horizontal_shift (vertical_shift p 3) 1).equation = λ x => 2 * (x - 1)^2 + 3 := by
  sorry

end parabola_transformation_l1932_193212


namespace volleyball_lineup_count_l1932_193251

def volleyball_team_size : ℕ := 16
def num_twins : ℕ := 2
def num_starters : ℕ := 8

theorem volleyball_lineup_count :
  (Nat.choose (volleyball_team_size - num_twins) num_starters) +
  (num_twins * Nat.choose (volleyball_team_size - num_twins) (num_starters - 1)) = 9867 := by
  sorry

end volleyball_lineup_count_l1932_193251


namespace card_trick_strategy_exists_l1932_193261

/-- Represents a card in the set of 29 cards -/
def Card := Fin 29

/-- Represents the strategy for selecting two cards to show -/
def Strategy := (Card × Card) → (Card × Card)

/-- Checks if two cards are adjacent in the circular arrangement -/
def adjacent (a b : Card) : Prop :=
  b.val = (a.val % 29 + 1) ∨ a.val = (b.val % 29 + 1)

/-- Determines if a strategy is valid for guessing hidden cards -/
def valid_strategy (s : Strategy) : Prop :=
  ∀ (hidden : Card × Card),
    let shown := s hidden
    ∃! (guessed : Card × Card),
      (guessed = hidden ∧ ¬adjacent guessed.1 guessed.2) ∨
      (guessed = hidden ∧ adjacent guessed.1 guessed.2)

/-- Theorem stating that there exists a valid strategy for the card trick -/
theorem card_trick_strategy_exists : ∃ (s : Strategy), valid_strategy s := by
  sorry

end card_trick_strategy_exists_l1932_193261


namespace roots_sum_power_l1932_193202

theorem roots_sum_power (c d : ℝ) : 
  c^2 - 5*c + 6 = 0 → d^2 - 5*d + 6 = 0 → c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end roots_sum_power_l1932_193202


namespace price_reduction_rate_l1932_193218

theorem price_reduction_rate (original_price final_price : ℝ) 
  (h1 : original_price = 200)
  (h2 : final_price = 128)
  (h3 : ∃ x : ℝ, final_price = original_price * (1 - x)^2) :
  ∃ x : ℝ, final_price = original_price * (1 - x)^2 ∧ x = 0.2 :=
by sorry

end price_reduction_rate_l1932_193218


namespace sum_of_absolute_roots_l1932_193201

def polynomial (x : ℝ) : ℝ := x^4 - 6*x^3 + 9*x^2 + 18*x - 24

theorem sum_of_absolute_roots : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x : ℝ, polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    |r₁| + |r₂| + |r₃| + |r₄| = 6 + 2 * Real.sqrt 6 :=
  sorry

end sum_of_absolute_roots_l1932_193201


namespace expression_evaluation_l1932_193221

theorem expression_evaluation (c : ℕ) (h : c = 4) : 
  (c^c - c * (c - 1)^(c - 1))^c = 148^4 := by
  sorry

end expression_evaluation_l1932_193221


namespace min_value_theorem_l1932_193240

/-- A geometric sequence with positive terms satisfying the given conditions -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r > 0, ∀ n, a (n + 1) = r * a n) ∧
  a 3 = a 2 + 2 * a 1

/-- The existence of terms satisfying the product condition -/
def ExistTerms (a : ℕ → ℝ) : Prop :=
  ∃ m n : ℕ, a m * a n = 64 * (a 1)^2

/-- The theorem statement -/
theorem min_value_theorem (a : ℕ → ℝ) 
  (h1 : GeometricSequence a) 
  (h2 : ExistTerms a) : 
  ∀ m n : ℕ, a m * a n = 64 * (a 1)^2 → 1 / m + 9 / n ≥ 2 :=
sorry

end min_value_theorem_l1932_193240


namespace bo_learning_words_l1932_193296

/-- Calculates the number of words to learn per day given the total number of flashcards,
    the percentage of known words, and the number of days to learn. -/
def words_per_day (total_cards : ℕ) (known_percentage : ℚ) (days_to_learn : ℕ) : ℚ :=
  (total_cards - (known_percentage * total_cards)) / days_to_learn

/-- Proves that given 800 flashcards, 20% known words, and 40 days to learn,
    the number of words to learn per day is 16. -/
theorem bo_learning_words :
  words_per_day 800 (1/5) 40 = 16 := by
  sorry

end bo_learning_words_l1932_193296


namespace prob_three_students_same_group_l1932_193239

/-- The probability that three specific students are assigned to the same group
    when 800 students are randomly assigned to 4 equal-sized groups -/
theorem prob_three_students_same_group :
  let total_students : ℕ := 800
  let num_groups : ℕ := 4
  let group_size : ℕ := total_students / num_groups
  -- Assuming each group has equal size
  (∀ g : Fin num_groups, (group_size : ℚ) = total_students / num_groups)
  →
  (probability_same_group : ℚ) = 1 / 16 := by
  sorry

end prob_three_students_same_group_l1932_193239


namespace multiplier_can_be_greater_than_one_l1932_193241

theorem multiplier_can_be_greater_than_one (a b : ℚ) (h : a * b ≤ b) : 
  ∃ (a : ℚ), a * b ≤ b ∧ a > 1 :=
sorry

end multiplier_can_be_greater_than_one_l1932_193241


namespace triangle_third_side_length_l1932_193247

theorem triangle_third_side_length : ∀ (x : ℝ),
  (x > 0 ∧ 5 + 9 > x ∧ x + 5 > 9 ∧ x + 9 > 5) → x = 8 ∨ (x < 8 ∨ x > 8) := by
  sorry

end triangle_third_side_length_l1932_193247


namespace jake_not_dropping_coffee_l1932_193285

theorem jake_not_dropping_coffee (trip_probability : ℝ) (drop_given_trip_probability : ℝ) :
  trip_probability = 0.4 →
  drop_given_trip_probability = 0.25 →
  1 - trip_probability * drop_given_trip_probability = 0.9 := by
sorry

end jake_not_dropping_coffee_l1932_193285


namespace circle_area_is_6pi_l1932_193279

/-- The equation of the circle C -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + 2 = 0

/-- The equation of the line -/
def line_equation (x y a : ℝ) : Prop :=
  y = a*x

/-- Definition of an equilateral triangle -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = d B C ∧ d B C = d C A

/-- The main theorem -/
theorem circle_area_is_6pi (a : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  (∃ x y, circle_equation x y a ∧ line_equation x y a) →
  is_equilateral_triangle A B C →
  (∃ r, π * r^2 = 6 * π) :=
sorry

end circle_area_is_6pi_l1932_193279


namespace term_degree_le_poly_degree_l1932_193234

/-- A polynomial of degree 6 -/
def Polynomial6 : Type := ℕ → ℚ

/-- The degree of a polynomial -/
def degree (p : Polynomial6) : ℕ := 6

/-- A term of a polynomial -/
def Term : Type := ℕ × ℚ

/-- The degree of a term -/
def termDegree (t : Term) : ℕ := t.1

theorem term_degree_le_poly_degree (p : Polynomial6) (t : Term) : 
  termDegree t ≤ degree p := by sorry

end term_degree_le_poly_degree_l1932_193234


namespace diana_bottle_caps_l1932_193216

theorem diana_bottle_caps (initial final eaten : ℕ) : 
  final = 61 → eaten = 4 → initial = final + eaten := by sorry

end diana_bottle_caps_l1932_193216


namespace wanda_eating_theorem_l1932_193262

/-- Pascal's triangle up to row n -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Check if a number is odd -/
def isOdd (n : ℕ) : Bool :=
  sorry

/-- Count odd numbers in Pascal's triangle up to row n -/
def countOddNumbers (triangle : List (List ℕ)) : ℕ :=
  sorry

/-- Check if a path in Pascal's triangle satisfies the no-sum condition -/
def validPath (path : List ℕ) : Bool :=
  sorry

/-- Main theorem -/
theorem wanda_eating_theorem :
  ∃ (path : List ℕ), 
    (path.length > 100000) ∧ 
    (∀ n ∈ path, n ∈ (PascalTriangle 2011).join) ∧
    (∀ n ∈ path, isOdd n) ∧
    validPath path :=
  sorry

end wanda_eating_theorem_l1932_193262


namespace allocation_methods_l1932_193293

/-- Represents the number of students --/
def num_students : ℕ := 5

/-- Represents the number of villages --/
def num_villages : ℕ := 3

/-- Represents the number of entities to be allocated (treating A and B as one entity) --/
def num_entities : ℕ := 4

/-- The number of ways to divide num_entities into num_villages non-empty groups --/
def ways_to_divide : ℕ := Nat.choose num_entities (num_villages - 1)

/-- The number of ways to arrange num_villages groups into num_villages villages --/
def ways_to_arrange : ℕ := Nat.factorial num_villages

/-- Theorem stating the total number of allocation methods --/
theorem allocation_methods :
  ways_to_divide * ways_to_arrange = 36 := by sorry

end allocation_methods_l1932_193293


namespace ducks_in_lake_l1932_193209

theorem ducks_in_lake (initial_ducks joining_ducks : ℕ) 
  (h1 : initial_ducks = 13) 
  (h2 : joining_ducks = 20) : 
  initial_ducks + joining_ducks = 33 := by
  sorry

end ducks_in_lake_l1932_193209


namespace greater_number_proof_l1932_193263

theorem greater_number_proof (a b : ℝ) (h_sum : a + b = 40) (h_diff : a - b = 2) (h_greater : a > b) : a = 21 := by
  sorry

end greater_number_proof_l1932_193263


namespace odd_function_zero_value_l1932_193226

/-- A function f is odd if f(-x) = -f(x) for all x in ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- For any odd function f defined on ℝ, f(0) = 0 -/
theorem odd_function_zero_value (f : ℝ → ℝ) (h : IsOdd f) : f 0 = 0 := by
  sorry

end odd_function_zero_value_l1932_193226


namespace power_sum_sequence_l1932_193268

theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11)
  (h_rec : ∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) :
  a^10 + b^10 = 123 := by
sorry

end power_sum_sequence_l1932_193268


namespace lizas_peanut_butter_cookies_l1932_193273

/-- Given the conditions of Liza's cookie-making scenario, prove that she used 2/5 of the remaining butter for peanut butter cookies. -/
theorem lizas_peanut_butter_cookies (total_butter : ℝ) (remaining_butter : ℝ) (peanut_butter_fraction : ℝ) :
  total_butter = 10 →
  remaining_butter = total_butter / 2 →
  2 = remaining_butter - peanut_butter_fraction * remaining_butter - (1 / 3) * (remaining_butter - peanut_butter_fraction * remaining_butter) →
  peanut_butter_fraction = 2 / 5 := by
  sorry

end lizas_peanut_butter_cookies_l1932_193273


namespace corresponding_sides_of_congruent_triangles_are_equal_l1932_193292

-- Define a triangle as a structure with three points
structure Triangle (α : Type*) :=
  (A B C : α)

-- Define congruence for triangles
def CongruentTriangles {α : Type*} (T1 T2 : Triangle α) : Prop :=
  sorry

-- Define the concept of corresponding sides
def CorrespondingSides {α : Type*} (T1 T2 : Triangle α) (s1 s2 : α × α) : Prop :=
  sorry

-- Define equality of sides
def EqualSides {α : Type*} (s1 s2 : α × α) : Prop :=
  sorry

-- Theorem: Corresponding sides of congruent triangles are equal
theorem corresponding_sides_of_congruent_triangles_are_equal
  {α : Type*} (T1 T2 : Triangle α) :
  CongruentTriangles T1 T2 →
  ∀ s1 s2, CorrespondingSides T1 T2 s1 s2 → EqualSides s1 s2 :=
by
  sorry

end corresponding_sides_of_congruent_triangles_are_equal_l1932_193292


namespace original_savings_calculation_l1932_193275

theorem original_savings_calculation (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) :
  furniture_fraction = 3 / 4 →
  tv_cost = 200 →
  (1 - furniture_fraction) * savings = tv_cost →
  savings = 800 := by
sorry

end original_savings_calculation_l1932_193275


namespace ellipse_x_axis_iff_l1932_193255

/-- Defines an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (k : ℝ) : Prop :=
  0 < k ∧ k < 2 ∧ ∀ x y : ℝ, x^2 / 2 + y^2 / k = 1 → 
    ∃ c : ℝ, c > 0 ∧ c < 1 ∧
      ∀ p : ℝ × ℝ, (p.1 - c)^2 + p.2^2 + (p.1 + c)^2 + p.2^2 = 2

/-- The condition 0 < k < 2 is necessary and sufficient for the equation
    x^2/2 + y^2/k = 1 to represent an ellipse with foci on the x-axis -/
theorem ellipse_x_axis_iff (k : ℝ) : is_ellipse_x_axis k ↔ (0 < k ∧ k < 2) :=
sorry

end ellipse_x_axis_iff_l1932_193255


namespace leftmost_digit_in_base9_is_5_l1932_193243

/-- Represents a number in base-3 as a list of digits -/
def Base3Number := List Nat

/-- Converts a base-3 number to its decimal (base-10) representation -/
def toDecimal (n : Base3Number) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its base-9 representation -/
def toBase9 (n : Nat) : List Nat :=
  sorry

/-- Gets the leftmost digit of a list of digits -/
def leftmostDigit (digits : List Nat) : Nat :=
  digits.head!

/-- The given base-3 number -/
def givenNumber : Base3Number :=
  [1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2]

theorem leftmost_digit_in_base9_is_5 :
  leftmostDigit (toBase9 (toDecimal givenNumber)) = 5 :=
sorry

end leftmost_digit_in_base9_is_5_l1932_193243


namespace balls_to_boxes_count_l1932_193256

/-- The number of ways to distribute n indistinguishable objects into k distinguishable groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls_to_boxes : ℕ := distribute 4 3

theorem balls_to_boxes_count :
  distribute_balls_to_boxes = 15 := by sorry

end balls_to_boxes_count_l1932_193256


namespace pure_imaginary_condition_l1932_193203

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a + I) * (1 - 2*I) = b*I) → a = -2 :=
by sorry

end pure_imaginary_condition_l1932_193203


namespace circle_and_tangent_lines_l1932_193299

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (m : ℝ), (x - m)^2 + (y - 3*m)^2 = (1 - m)^2 + (6 - 3*m)^2 ∧
             (x - m)^2 + (y - 3*m)^2 = (-2 - m)^2 + (3 - 3*m)^2

-- Define the line 3x-y=0
def center_line (x y : ℝ) : Prop := 3*x - y = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 1)

-- Theorem statement
theorem circle_and_tangent_lines :
  ∃ (x₀ y₀ r : ℝ),
    (∀ x y, circle_C x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    center_line x₀ y₀ ∧
    ((x₀ - 1)^2 + (y₀ - 3)^2 = 9) ∧
    (∀ x y, (5*x - 12*y - 8 = 0 ∨ x = 4) →
      ((x - x₀)^2 + (y - y₀)^2 = r^2 ∧
       ((x - 4)^2 + (y - 1)^2) * r^2 = ((x - x₀)*(4 - x₀) + (y - y₀)*(1 - y₀))^2)) :=
by sorry

end circle_and_tangent_lines_l1932_193299


namespace crane_flock_size_l1932_193278

theorem crane_flock_size (duck_flock_size : ℕ) (total_birds : ℕ) (h1 : duck_flock_size = 13) (h2 : total_birds = 221) (h3 : total_birds % duck_flock_size = 0) :
  ∃ (crane_flock_size : ℕ), crane_flock_size = total_birds ∧ total_birds % crane_flock_size = 0 := by
  sorry

end crane_flock_size_l1932_193278


namespace no_real_solutions_l1932_193282

theorem no_real_solutions :
  ¬ ∃ x : ℝ, x + Real.sqrt (x + 1) = 6 := by
  sorry

end no_real_solutions_l1932_193282


namespace function_properties_l1932_193244

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f (10 + x) = f (10 - x))
  (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
  is_odd f ∧ has_period f 40 := by sorry

end function_properties_l1932_193244


namespace max_market_women_eight_market_women_l1932_193266

def farthings_in_2s_2_1_4d : ℕ := 105

theorem max_market_women (n : ℕ) : n ∣ farthings_in_2s_2_1_4d → n ≤ 8 :=
sorry

theorem eight_market_women : ∃ (s : Finset ℕ), s.card = 8 ∧ ∀ n ∈ s, n ∣ farthings_in_2s_2_1_4d :=
sorry

end max_market_women_eight_market_women_l1932_193266


namespace f_monotone_decreasing_l1932_193205

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem stating that f is monotonically decreasing on (-∞, 1]
theorem f_monotone_decreasing :
  MonotoneOn f (Set.Iic 1) := by sorry

end f_monotone_decreasing_l1932_193205
