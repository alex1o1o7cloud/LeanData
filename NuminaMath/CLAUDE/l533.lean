import Mathlib

namespace not_always_parallel_to_intersection_l533_53342

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_to_intersection
  (α β : Plane) (m n : Line) :
  ¬(∀ (α β : Plane) (m n : Line),
    line_parallel_plane m α ∧ intersect α β n → parallel m n) :=
by sorry

end not_always_parallel_to_intersection_l533_53342


namespace abc_product_l533_53300

theorem abc_product (a b c : ℤ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a + b + c = 30 → 
  1 / a + 1 / b + 1 / c + 450 / (a * b * c) = 1 → 
  a * b * c = 1920 := by
sorry

end abc_product_l533_53300


namespace athena_spent_14_l533_53387

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℝ) (sandwich_count : ℕ) (drink_price : ℝ) (drink_count : ℕ) : ℝ :=
  sandwich_price * sandwich_count + drink_price * drink_count

/-- Theorem stating that Athena spent $14 on snacks -/
theorem athena_spent_14 :
  total_spent 3 3 2.5 2 = 14 := by
  sorry

end athena_spent_14_l533_53387


namespace inradius_formula_l533_53373

theorem inradius_formula (β γ R : Real) (hβ : 0 < β) (hγ : 0 < γ) (hβγ : β + γ < π) (hR : R > 0) :
  ∃ (r : Real), r = 4 * R * Real.sin (β / 2) * Real.sin (γ / 2) * Real.cos ((β + γ) / 2) :=
by sorry

end inradius_formula_l533_53373


namespace boat_speed_in_still_water_l533_53324

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (travel_time_minutes : ℝ) :
  current_speed = 8 →
  downstream_distance = 36.67 →
  travel_time_minutes = 44 →
  ∃ (boat_speed : ℝ), boat_speed = 42 ∧ 
    (boat_speed + current_speed) * (travel_time_minutes / 60) = downstream_distance :=
by sorry

end boat_speed_in_still_water_l533_53324


namespace translate_down_two_units_l533_53394

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateVertically (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - amount }

theorem translate_down_two_units :
  let original := Line.mk (-2) 0
  let translated := translateVertically original 2
  translated = Line.mk (-2) (-2) := by sorry

end translate_down_two_units_l533_53394


namespace distance_circle_center_to_point_l533_53351

/-- The distance between the center of the circle with equation x^2 + y^2 = 6x - 8y + 24
    and the point (-3, 4) is 10. -/
theorem distance_circle_center_to_point :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 = 6*x - 8*y + 24
  let center : ℝ × ℝ := (3, -4)
  let point : ℝ × ℝ := (-3, 4)
  (∃ (x y : ℝ), circle_eq x y) →
  Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = 10 :=
by sorry

end distance_circle_center_to_point_l533_53351


namespace people_joined_line_l533_53337

theorem people_joined_line (initial : ℕ) (left : ℕ) (current : ℕ) : 
  initial ≥ left → 
  current = (initial - left) + (current - (initial - left)) :=
by sorry

end people_joined_line_l533_53337


namespace cylinder_sphere_volume_ratio_l533_53346

/-- The ratio of the volume of a cylinder inscribed in a sphere to the volume of the sphere,
    where the cylinder's height is 4/3 of the sphere's radius. -/
theorem cylinder_sphere_volume_ratio (R : ℝ) (h : R > 0) :
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_height := (4 / 3) * R
  let cylinder_radius := Real.sqrt ((5 / 9) * R^2)
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  cylinder_volume / sphere_volume = 5 / 9 := by
  sorry

end cylinder_sphere_volume_ratio_l533_53346


namespace sqrt_25_l533_53309

theorem sqrt_25 : Real.sqrt 25 = 5 ∨ Real.sqrt 25 = -5 := by sorry

end sqrt_25_l533_53309


namespace trivia_team_members_l533_53308

theorem trivia_team_members (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  absent_members = 6 → points_per_member = 3 → total_points = 27 → 
  ∃ (total_members : ℕ), total_members = 15 ∧ 
  points_per_member * (total_members - absent_members) = total_points :=
by
  sorry

end trivia_team_members_l533_53308


namespace order_of_roots_l533_53395

theorem order_of_roots (a b c : ℝ) 
  (ha : a = 4^(2/3)) 
  (hb : b = 3^(2/3)) 
  (hc : c = 25^(1/3)) : 
  b < a ∧ a < c := by sorry

end order_of_roots_l533_53395


namespace circle_relationship_l533_53383

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Theorem about the relationship between two circles -/
theorem circle_relationship (R₁ R₂ d : ℝ) (c₁ c₂ : Circle) 
  (h₁ : c₁.radius = R₁)
  (h₂ : c₂.radius = R₂)
  (h₃ : R₁ ≠ R₂)
  (h₄ : ∃ x : ℝ, x^2 - 2*R₁*x + R₂^2 - d*(R₂ - R₁) = 0 ∧ 
        ∀ y : ℝ, y^2 - 2*R₁*y + R₂^2 - d*(R₂ - R₁) = 0 → y = x) :
  R₁ + R₂ = d ∧ (∀ p : ℝ × ℝ, ‖p‖ ≠ R₁ ∨ ‖p - (d, 0)‖ ≠ R₂) := by
sorry

end circle_relationship_l533_53383


namespace zeros_of_f_range_of_m_l533_53384

-- Define the function f
def f (a x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

-- Theorem for part (I)
theorem zeros_of_f'_depend_on_a (a : ℝ) :
  ∃ n : ℕ, n ∈ ({1, 2} : Set ℕ) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-1) 3 ∧ x₂ ∈ Set.Icc (-1) 3 ∧ 
   f' a x₁ = 0 ∧ f' a x₂ = 0 ∧ 
   ∀ x ∈ Set.Icc (-1) 3, f' a x = 0 → x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for part (II)
theorem range_of_m (a : ℝ) (h : a ∈ Set.Icc (-3) 0) :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 → 
    m - a * m^2 ≥ |f a x₁ - f a x₂|) → 
  m ∈ Set.Ici 5 :=
sorry

end zeros_of_f_range_of_m_l533_53384


namespace smallest_fraction_between_l533_53355

theorem smallest_fraction_between (p q : ℕ+) 
  (h1 : (3 : ℚ) / 5 < p / q)
  (h2 : p / q < (5 : ℚ) / 8)
  (h3 : ∀ (r s : ℕ+), (3 : ℚ) / 5 < r / s → r / s < (5 : ℚ) / 8 → s ≤ q) :
  q - p = 5 := by
  sorry

end smallest_fraction_between_l533_53355


namespace complex_real_condition_l533_53307

theorem complex_real_condition (a : ℝ) : 
  let Z : ℂ := (a - 5) / (a^2 + 4*a - 5) + (a^2 + 2*a - 15) * I
  Z.im = 0 → a = 3 := by
  sorry

end complex_real_condition_l533_53307


namespace tan_eq_sin_cos_unique_solution_l533_53313

open Real

theorem tan_eq_sin_cos_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ arccos 0.1 ∧ tan x = sin (cos x) := by
  sorry

end tan_eq_sin_cos_unique_solution_l533_53313


namespace labrador_starting_weight_l533_53381

/-- The starting weight of the labrador puppy -/
def L : ℝ := 40

/-- The starting weight of the dachshund puppy -/
def dachshund_weight : ℝ := 12

/-- The weight gain percentage for both dogs -/
def weight_gain_percentage : ℝ := 0.25

/-- The weight difference between the dogs at the end of the year -/
def weight_difference : ℝ := 35

/-- Theorem stating that the labrador puppy's starting weight satisfies the given conditions -/
theorem labrador_starting_weight :
  L * (1 + weight_gain_percentage) - dachshund_weight * (1 + weight_gain_percentage) = weight_difference := by
  sorry

end labrador_starting_weight_l533_53381


namespace crown_cost_l533_53359

/-- Given a total payment of $22,000 for a crown including a 10% tip,
    prove that the original cost of the crown was $20,000. -/
theorem crown_cost (total_payment : ℝ) (tip_percentage : ℝ) (h1 : total_payment = 22000)
    (h2 : tip_percentage = 0.1) : 
  ∃ (original_cost : ℝ), 
    original_cost * (1 + tip_percentage) = total_payment ∧ 
    original_cost = 20000 := by
  sorry

end crown_cost_l533_53359


namespace stratified_sampling_teachers_l533_53339

theorem stratified_sampling_teachers :
  let total_teachers : ℕ := 150
  let senior_teachers : ℕ := 45
  let intermediate_teachers : ℕ := 90
  let junior_teachers : ℕ := 15
  let sample_size : ℕ := 30
  let sample_senior : ℕ := 9
  let sample_intermediate : ℕ := 18
  let sample_junior : ℕ := 3
  
  (total_teachers = senior_teachers + intermediate_teachers + junior_teachers) →
  (sample_size = sample_senior + sample_intermediate + sample_junior) →
  (sample_senior : ℚ) / senior_teachers = (sample_intermediate : ℚ) / intermediate_teachers →
  (sample_senior : ℚ) / senior_teachers = (sample_junior : ℚ) / junior_teachers →
  (sample_size : ℚ) / total_teachers = (sample_senior : ℚ) / senior_teachers :=
by
  sorry


end stratified_sampling_teachers_l533_53339


namespace second_number_proof_l533_53331

theorem second_number_proof (a b c : ℚ) : 
  a + b + c = 98 ∧ 
  a / b = 2 / 3 ∧ 
  b / c = 5 / 8 → 
  b = 30 :=
by sorry

end second_number_proof_l533_53331


namespace time_from_velocity_and_displacement_l533_53321

/-- Given the equations for velocity and displacement, prove the formula for time. -/
theorem time_from_velocity_and_displacement
  (g V V₀ S S₀ a t : ℝ)
  (hV : V = g * (t - a) + V₀)
  (hS : S = (1/2) * g * (t - a)^2 + V₀ * (t - a) + S₀) :
  t = a + (V - V₀) / g :=
sorry

end time_from_velocity_and_displacement_l533_53321


namespace work_completion_time_l533_53386

-- Define the work completion times for Paul and Rose
def paul_time : ℝ := 80
def rose_time : ℝ := 120

-- Define the theorem
theorem work_completion_time : 
  let paul_rate := 1 / paul_time
  let rose_rate := 1 / rose_time
  let combined_rate := paul_rate + rose_rate
  (1 / combined_rate) = 48 := by sorry

end work_completion_time_l533_53386


namespace half_job_days_l533_53363

/-- 
Proves that the number of days to complete half a job is 6, 
given that it takes 6 more days to finish the entire job after completing half of it.
-/
theorem half_job_days : 
  ∀ (x : ℝ), (x + 6 = 2*x) → x = 6 := by
  sorry

end half_job_days_l533_53363


namespace binomial_expansion_coefficient_l533_53328

/-- The coefficient of the third term in the binomial expansion of (a + √x)^5 -/
def third_term_coefficient (a : ℝ) (x : ℝ) : ℝ := 10 * a^3 * x

/-- Theorem: If the coefficient of the third term in (a + √x)^5 is 80, then a = 2 -/
theorem binomial_expansion_coefficient (a : ℝ) (x : ℝ) :
  third_term_coefficient a x = 80 → a = 2 := by
  sorry

end binomial_expansion_coefficient_l533_53328


namespace square_difference_identity_l533_53371

theorem square_difference_identity : (50 + 12)^2 - (12^2 + 50^2) = 1200 := by
  sorry

end square_difference_identity_l533_53371


namespace origin_symmetry_coordinates_l533_53303

def point_symmetry (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem origin_symmetry_coordinates :
  let P : ℝ × ℝ := (-2, 3)
  let P1 : ℝ × ℝ := point_symmetry P.1 P.2
  P1 = (2, -3) := by sorry

end origin_symmetry_coordinates_l533_53303


namespace planes_parallel_if_common_perpendicular_l533_53390

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_common_perpendicular 
  (a b : Plane) (m : Line) : 
  a ≠ b → 
  perpendicular m a → 
  perpendicular m b → 
  parallel a b :=
sorry

end planes_parallel_if_common_perpendicular_l533_53390


namespace peanuts_added_l533_53325

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 10)
  (h2 : final_peanuts = 18) :
  final_peanuts - initial_peanuts = 8 := by
sorry

end peanuts_added_l533_53325


namespace smallest_y_with_given_remainders_l533_53326

theorem smallest_y_with_given_remainders : 
  ∃! y : ℕ, 
    y > 0 ∧
    y % 3 = 2 ∧ 
    y % 5 = 4 ∧ 
    y % 7 = 6 ∧
    ∀ z : ℕ, z > 0 ∧ z % 3 = 2 ∧ z % 5 = 4 ∧ z % 7 = 6 → y ≤ z :=
by
  -- Proof goes here
  sorry

end smallest_y_with_given_remainders_l533_53326


namespace missing_village_population_l533_53343

def village_populations : List ℕ := [803, 900, 1100, 945, 980, 1249]

theorem missing_village_population 
  (total_villages : ℕ) 
  (average_population : ℕ) 
  (known_populations : List ℕ) 
  (h1 : total_villages = 7)
  (h2 : average_population = 1000)
  (h3 : known_populations = village_populations)
  (h4 : known_populations.length = 6) :
  ∃ (missing_population : ℕ), 
    missing_population = total_villages * average_population - known_populations.sum ∧
    missing_population = 1023 :=
by sorry

end missing_village_population_l533_53343


namespace megan_zoo_pictures_l533_53318

/-- Represents the number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- Represents the total number of pictures Megan took -/
def total_pictures : ℕ := zoo_pictures + 18

/-- Represents the number of pictures remaining after deletion -/
def remaining_pictures : ℕ := total_pictures - 31

theorem megan_zoo_pictures : 
  zoo_pictures = 15 ∧ 
  total_pictures = zoo_pictures + 18 ∧ 
  remaining_pictures = 2 :=
sorry

end megan_zoo_pictures_l533_53318


namespace double_average_l533_53391

theorem double_average (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) :
  n = 11 →
  initial_avg = 36 →
  new_avg = 2 * initial_avg →
  new_avg = 72 :=
by sorry

end double_average_l533_53391


namespace quadratic_function_properties_l533_53378

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem quadratic_function_properties :
  (∀ x, f (x + 1) - f x = 2*x - 1) ∧
  f 0 = 2 ∧
  (∀ x ∈ Set.Icc (-2) 2, 1 ≤ f x ∧ f x ≤ 10) ∧
  (∀ t, 
    let min_value := 
      if t ≥ 1 then t^2 - 2*t + 2
      else if 0 < t ∧ t < 1 then 1
      else t^2 + 2*t + 1
    ∀ x ∈ Set.Icc t (t + 1), f x ≥ min_value) :=
by sorry

end quadratic_function_properties_l533_53378


namespace mathilda_debt_l533_53332

theorem mathilda_debt (initial_payment : ℝ) (remaining_percentage : ℝ) (original_debt : ℝ) : 
  initial_payment = 125 →
  remaining_percentage = 75 →
  initial_payment = (100 - remaining_percentage) / 100 * original_debt →
  original_debt = 500 := by
sorry

end mathilda_debt_l533_53332


namespace range_of_a_for_f_with_two_zeros_l533_53389

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- State the theorem
theorem range_of_a_for_f_with_two_zeros :
  (∃ a : ℝ, ∀ x : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0)) →
  (∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) → 1 ≤ a ∧ a ≤ 5) ∧
  (∀ a : ℝ, 1 ≤ a ∧ a ≤ 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
by sorry

end range_of_a_for_f_with_two_zeros_l533_53389


namespace movie_ticket_price_decrease_l533_53350

theorem movie_ticket_price_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100)
  (h2 : new_price = 80) : 
  (old_price - new_price) / old_price * 100 = 20 := by
  sorry

end movie_ticket_price_decrease_l533_53350


namespace camp_food_consumption_l533_53310

/-- Represents the amount of food eaten by dogs and puppies in a day -/
def total_food_eaten (num_puppies num_dogs : ℕ) 
                     (dog_meal_frequency puppy_meal_frequency : ℕ) 
                     (dog_meal_amount : ℚ) 
                     (dog_puppy_food_ratio : ℚ) : ℚ :=
  let dog_daily_food := dog_meal_amount * dog_meal_frequency
  let puppy_meal_amount := dog_meal_amount / dog_puppy_food_ratio
  let puppy_daily_food := puppy_meal_amount * puppy_meal_frequency
  (num_dogs : ℚ) * dog_daily_food + (num_puppies : ℚ) * puppy_daily_food

/-- Theorem stating the total food eaten by dogs and puppies in a day -/
theorem camp_food_consumption : 
  total_food_eaten 6 5 2 8 6 3 = 108 := by
  sorry

end camp_food_consumption_l533_53310


namespace triangle_side_value_l533_53319

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) : 
  (b^2 - c^2 + 2*a = 0) →
  (Real.tan C / Real.tan B = 3) →
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
  a = 4 := by
sorry

end triangle_side_value_l533_53319


namespace tournament_committee_count_l533_53366

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- The number of members chosen from the host team -/
def host_members : ℕ := 3

/-- The number of teams that select 2 members -/
def teams_select_two : ℕ := 3

/-- The number of teams that select 3 members (excluding the host) -/
def teams_select_three : ℕ := 1

/-- The total number of ways to form a tournament committee -/
def total_committees : ℕ := 229105500

theorem tournament_committee_count :
  (num_teams) *
  (Nat.choose team_size host_members) *
  (Nat.choose (num_teams - 1) teams_select_three) *
  (Nat.choose team_size host_members) *
  (Nat.choose team_size 2 ^ teams_select_two) = total_committees := by
  sorry

end tournament_committee_count_l533_53366


namespace complex_number_coordinate_l533_53336

theorem complex_number_coordinate (i : ℂ) (h : i^2 = -1) :
  (i^2015) / (i - 2) = -1/5 + 2/5 * i := by sorry

end complex_number_coordinate_l533_53336


namespace ball_returns_in_three_throws_l533_53385

/-- The number of boys in the circle -/
def n : ℕ := 15

/-- The number of positions skipped in each throw (including the thrower) -/
def skip : ℕ := 5

/-- The sequence of positions the ball reaches -/
def ball_sequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | i + 1 => (ball_sequence start i + skip) % n

/-- The theorem stating that it takes 3 throws for the ball to return to the start -/
theorem ball_returns_in_three_throws (start : ℕ) (h : start > 0 ∧ start ≤ n) : 
  ball_sequence start 3 = start :=
sorry

end ball_returns_in_three_throws_l533_53385


namespace principal_amount_l533_53334

/-- Given a principal amount P lent at simple interest rate r,
    prove that P = 710 given the conditions from the problem. -/
theorem principal_amount (P r : ℝ) : 
  (P + P * r * 3 = 920) →
  (P + P * r * 9 = 1340) →
  P = 710 := by
  sorry

end principal_amount_l533_53334


namespace equation_roots_l533_53345

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (21 / (x^2 - 9)) - (3 / (x - 3)) - 2
  ∀ x : ℝ, f x = 0 ↔ x = -3 ∨ x = 5 := by
sorry

end equation_roots_l533_53345


namespace gold_distribution_theorem_l533_53344

/-- The number of gold nuggets -/
def n : ℕ := 2020

/-- The sum of masses of all nuggets -/
def total_mass : ℕ := n * (n + 1) / 2

/-- The maximum difference in mass between the two chests -/
def max_diff : ℕ := n

/-- The guaranteed amount of gold in the heavier chest -/
def guaranteed_mass : ℕ := total_mass / 2 + max_diff / 2

theorem gold_distribution_theorem :
  ∃ (chest_mass : ℕ), chest_mass ≥ guaranteed_mass ∧ 
  chest_mass ≤ total_mass - (total_mass / 2 - max_diff / 2) :=
sorry

end gold_distribution_theorem_l533_53344


namespace fourth_transaction_is_37_l533_53320

/-- Represents the balance of class funds after a series of transactions -/
def class_funds (initial_balance : Int) (transactions : List Int) : Int :=
  initial_balance + transactions.sum

/-- Theorem: Given the initial balance and three transactions, 
    the fourth transaction must be 37 to reach the final balance of 82 -/
theorem fourth_transaction_is_37 
  (initial_balance : Int)
  (transaction1 transaction2 transaction3 : Int)
  (h1 : initial_balance = 0)
  (h2 : transaction1 = 230)
  (h3 : transaction2 = -75)
  (h4 : transaction3 = -110) :
  class_funds initial_balance [transaction1, transaction2, transaction3, 37] = 82 := by
  sorry

#eval class_funds 0 [230, -75, -110, 37]

end fourth_transaction_is_37_l533_53320


namespace intersection_of_M_and_N_l533_53301

def M : Set ℝ := {-1, 1, 2}
def N : Set ℝ := {x | x < 1}

theorem intersection_of_M_and_N : M ∩ N = {-1} := by
  sorry

end intersection_of_M_and_N_l533_53301


namespace van_transport_l533_53368

theorem van_transport (students_per_van : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : students_per_van = 28)
  (h2 : num_boys = 60)
  (h3 : num_girls = 80) :
  (num_boys + num_girls) / students_per_van = 5 := by
  sorry

#check van_transport

end van_transport_l533_53368


namespace G_equals_4F_l533_53367

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := Real.log ((1 + (4*x + x^4)/(1 + 4*x^3)) / (1 - (4*x + x^4)/(1 + 4*x^3)))

theorem G_equals_4F (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1 ∧ 1 + 4*x^3 ≠ 0) : G x = 4 * F x := by
  sorry

end G_equals_4F_l533_53367


namespace ceiling_floor_calculation_l533_53358

theorem ceiling_floor_calculation : 
  ⌈(18 : ℚ) / 5 * (-25 : ℚ) / 4⌉ - ⌊(18 : ℚ) / 5 * ⌊(-25 : ℚ) / 4⌋⌋ = 4 := by
  sorry

end ceiling_floor_calculation_l533_53358


namespace smallest_number_l533_53361

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def number_a : List Nat := [2, 0]
def number_b : List Nat := [3, 0]
def number_c : List Nat := [2, 3]
def number_d : List Nat := [3, 1]

theorem smallest_number :
  let a := base_to_decimal number_a 7
  let b := base_to_decimal number_b 5
  let c := base_to_decimal number_c 6
  let d := base_to_decimal number_d 4
  d < a ∧ d < b ∧ d < c := by sorry

end smallest_number_l533_53361


namespace nonnegative_integer_solutions_l533_53379

theorem nonnegative_integer_solutions : 
  {(x, y) : ℕ × ℕ | 3 * x^2 + 2 * 9^y = x * (4^(y + 1) - 1)} = {(3, 1), (2, 1)} := by
  sorry

end nonnegative_integer_solutions_l533_53379


namespace inequality_proof_l533_53376

theorem inequality_proof (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h : 1/x + 1/y + 1/z = 1) : 
  a^x + b^y + c^z ≥ (4*a*b*c*x*y*z) / ((x+y+z-3)^2) := by
  sorry

end inequality_proof_l533_53376


namespace closest_perfect_square_to_350_l533_53382

theorem closest_perfect_square_to_350 :
  ∀ n : ℕ, n ≠ 19 → (n ^ 2 : ℤ) ≠ 361 → |350 - (19 ^ 2 : ℤ)| ≤ |350 - (n ^ 2 : ℤ)| :=
by sorry

end closest_perfect_square_to_350_l533_53382


namespace gcd_228_1995_l533_53392

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l533_53392


namespace solution_set_is_open_interval_l533_53341

/-- A function f is decreasing on ℝ if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The set of all real numbers x satisfying f(1/|x|) < f(1) for a decreasing function f -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | f (1 / |x|) < f 1}

theorem solution_set_is_open_interval (f : ℝ → ℝ) (h : DecreasingOn f) :
  SolutionSet f = Set.Ioo (-1) 1 := by
  sorry

end solution_set_is_open_interval_l533_53341


namespace smallest_number_with_conditions_l533_53374

theorem smallest_number_with_conditions : ∃ n : ℕ, 
  (∀ m : ℕ, m ≥ n → (11 ∣ m ∧ ∀ k : ℕ, 3 ≤ k ∧ k ≤ 8 → m % k = 2) → m ≥ 3362) ∧
  (11 ∣ 3362) ∧
  (∀ k : ℕ, 3 ≤ k ∧ k ≤ 8 → 3362 % k = 2) :=
by sorry

end smallest_number_with_conditions_l533_53374


namespace intersection_point_on_line_and_plane_l533_53314

/-- The line passing through the point (1, -1, 1) in the direction (1, 0, -1) -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (1 + t, -1, 1 - t)

/-- The plane with equation 3x - 2y - 4z - 8 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  3 * x - 2 * y - 4 * z - 8 = 0

/-- The intersection point of the line and the plane -/
def intersection_point : ℝ × ℝ × ℝ := (-6, -1, 8)

theorem intersection_point_on_line_and_plane :
  (∃ t : ℝ, line t = intersection_point) ∧
  plane intersection_point ∧
  (∀ p : ℝ × ℝ × ℝ, (∃ t : ℝ, line t = p) → plane p → p = intersection_point) :=
by sorry

end intersection_point_on_line_and_plane_l533_53314


namespace cameron_typing_difference_l533_53348

theorem cameron_typing_difference (
  speed_before : ℕ) 
  (speed_after : ℕ) 
  (time : ℕ) 
  (h1 : speed_before = 10) 
  (h2 : speed_after = 8) 
  (h3 : time = 5) : 
  speed_before * time - speed_after * time = 10 :=
by sorry

end cameron_typing_difference_l533_53348


namespace carries_payment_is_94_l533_53396

/-- The amount Carrie pays for clothes at the mall -/
def carries_payment (num_shirts num_pants num_jackets shirt_cost pant_cost jacket_cost : ℕ) : ℕ :=
  let total_cost := num_shirts * shirt_cost + num_pants * pant_cost + num_jackets * jacket_cost
  total_cost / 2

/-- Theorem: Carrie pays $94 for the clothes -/
theorem carries_payment_is_94 : carries_payment 4 2 2 8 18 60 = 94 := by
  sorry

#eval carries_payment 4 2 2 8 18 60

end carries_payment_is_94_l533_53396


namespace integral_sqrt_rational_l533_53304

open Real MeasureTheory

/-- The definite integral of 5√(x+24) / ((x+24)^2 * √x) from x = 1 to x = 8 is equal to 1/8 -/
theorem integral_sqrt_rational : 
  ∫ x in (1 : ℝ)..8, (5 * Real.sqrt (x + 24)) / ((x + 24)^2 * Real.sqrt x) = 1/8 := by
  sorry

end integral_sqrt_rational_l533_53304


namespace complement_of_union_relative_to_U_l533_53333

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union_relative_to_U :
  (U \ (A ∪ B)) = {2, 4} := by sorry

end complement_of_union_relative_to_U_l533_53333


namespace no_solutions_to_absolute_value_equation_l533_53302

theorem no_solutions_to_absolute_value_equation :
  ¬∃ y : ℝ, |y - 2| = |y - 1| + |y - 4| := by
  sorry

end no_solutions_to_absolute_value_equation_l533_53302


namespace factorize_expression1_factorize_expression2_l533_53322

-- First expression
theorem factorize_expression1 (y : ℝ) :
  y + (y - 4) * (y - 1) = (y - 2)^2 := by sorry

-- Second expression
theorem factorize_expression2 (x y a b : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a - 2 * b) * (3 * a + 2 * b) := by sorry

end factorize_expression1_factorize_expression2_l533_53322


namespace inequality_system_solution_l533_53316

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 3 ≥ x)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  -2 < x ∧ x ≤ 3

-- Theorem statement
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ solution_set x := by
  sorry

end inequality_system_solution_l533_53316


namespace sound_distance_at_zero_celsius_sound_distance_in_five_seconds_l533_53388

/-- Represents the speed of sound in air at different temperatures -/
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- Default case for temperatures not in the table

/-- Calculates the distance traveled by sound in a given time at 0°C -/
def distance_traveled (time : Int) : Int :=
  (speed_of_sound 0) * time

theorem sound_distance_at_zero_celsius (time : Int) :
  distance_traveled time = speed_of_sound 0 * time :=
by sorry

theorem sound_distance_in_five_seconds :
  distance_traveled 5 = 1650 :=
by sorry

end sound_distance_at_zero_celsius_sound_distance_in_five_seconds_l533_53388


namespace percentage_with_no_conditions_is_10_percent_l533_53311

-- Define the total number of teachers
def total_teachers : ℕ := 150

-- Define the number of teachers with each condition
def high_blood_pressure : ℕ := 90
def heart_trouble : ℕ := 60
def diabetes : ℕ := 30

-- Define the number of teachers with combinations of conditions
def high_blood_pressure_and_heart_trouble : ℕ := 25
def heart_trouble_and_diabetes : ℕ := 10
def high_blood_pressure_and_diabetes : ℕ := 15
def all_three_conditions : ℕ := 5

-- Define the function to calculate the percentage
def percentage_with_no_conditions : ℚ :=
  let teachers_with_conditions := high_blood_pressure + heart_trouble + diabetes
    - high_blood_pressure_and_heart_trouble - heart_trouble_and_diabetes - high_blood_pressure_and_diabetes
    + all_three_conditions
  let teachers_with_no_conditions := total_teachers - teachers_with_conditions
  (teachers_with_no_conditions : ℚ) / (total_teachers : ℚ) * 100

-- Theorem statement
theorem percentage_with_no_conditions_is_10_percent :
  percentage_with_no_conditions = 10 := by
  sorry

end percentage_with_no_conditions_is_10_percent_l533_53311


namespace lines_parallel_iff_m_eq_neg_two_l533_53365

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line: 2x + my - 2m + 4 = 0 -/
def line1 (m : ℝ) : Line :=
  { a := 2, b := m, c := -2*m + 4 }

/-- The second line: mx + 2y - m + 2 = 0 -/
def line2 (m : ℝ) : Line :=
  { a := m, b := 2, c := -m + 2 }

/-- Theorem stating that the lines are parallel if and only if m = -2 -/
theorem lines_parallel_iff_m_eq_neg_two :
  ∀ m : ℝ, parallel (line1 m) (line2 m) ↔ m = -2 := by
  sorry

end lines_parallel_iff_m_eq_neg_two_l533_53365


namespace nonnegative_solutions_count_l533_53362

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 := by sorry

end nonnegative_solutions_count_l533_53362


namespace sum_of_angles_two_triangles_l533_53360

theorem sum_of_angles_two_triangles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ) :
  angle1 + angle3 + angle5 = 180 →
  angle2 + angle4 + angle6 = 180 →
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 := by
sorry

end sum_of_angles_two_triangles_l533_53360


namespace slope_range_for_intersecting_line_l533_53369

/-- The range of possible slopes for a line passing through a given point and intersecting a line segment -/
theorem slope_range_for_intersecting_line (M P Q : ℝ × ℝ) :
  M = (-1, 2) →
  P = (-4, -1) →
  Q = (3, 0) →
  let slope_range := {k : ℝ | k ≤ -1/2 ∨ k ≥ 1}
  ∀ k : ℝ,
    (∃ (x y : ℝ), (k * (x - M.1) = y - M.2 ∧
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
        x = P.1 + t * (Q.1 - P.1) ∧
        y = P.2 + t * (Q.2 - P.2))) ↔
    k ∈ slope_range :=
by sorry

end slope_range_for_intersecting_line_l533_53369


namespace bug_meeting_point_l533_53317

/-- Triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point on the perimeter of a triangle -/
structure PerimeterPoint where
  distanceFromP : ℝ

/-- Theorem: In a triangle PQR with side lengths PQ=7, QR=8, and PR=9, 
    if two bugs start simultaneously from P and crawl along the perimeter 
    in opposite directions at the same speed, meeting at point S, 
    then the length of QS is 5. -/
theorem bug_meeting_point (t : Triangle) (s : PerimeterPoint) : 
  t.a = 7 ∧ t.b = 8 ∧ t.c = 9 → s.distanceFromP = 12 → t.a + s.distanceFromP - 12 = 5 := by
  sorry

end bug_meeting_point_l533_53317


namespace tetromino_properties_l533_53305

-- Define a tetromino as a shape formed from 4 squares
structure Tetromino :=
  (squares : Finset (ℤ × ℤ))
  (size : squares.card = 4)

-- Define rotation equivalence
def rotationEquivalent (t1 t2 : Tetromino) : Prop := sorry

-- Define the set of distinct tetrominos
def distinctTetrominos : Finset Tetromino := sorry

-- Define a tiling of a rectangle
def tiling (w h : ℕ) (pieces : Finset Tetromino) : Prop := sorry

theorem tetromino_properties :
  -- There are exactly 7 distinct tetrominos
  distinctTetrominos.card = 7 ∧
  -- It is impossible to tile a 4 × 7 rectangle with one of each distinct tetromino
  ¬ tiling 4 7 distinctTetrominos := by sorry

end tetromino_properties_l533_53305


namespace complex_calculation_l533_53372

theorem complex_calculation (a b : ℂ) (ha : a = 3 + 2*Complex.I) (hb : b = 2 - 2*Complex.I) :
  3*a - 4*b = 1 + 14*Complex.I := by sorry

end complex_calculation_l533_53372


namespace unique_natural_number_satisfying_conditions_l533_53330

theorem unique_natural_number_satisfying_conditions :
  ∃! (x : ℕ), 
    (∃ (k : ℕ), 3 * x + 1 = k^2) ∧ 
    (∃ (t : ℕ), 6 * x - 2 = t^2) ∧ 
    Nat.Prime (6 * x^2 - 1) ∧
    x = 1 :=
by sorry

end unique_natural_number_satisfying_conditions_l533_53330


namespace f_of_5_eq_110_l533_53329

/-- The polynomial function f(x) = 3x^4 - 20x^3 + 38x^2 - 35x - 40 -/
def f (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 38 * x^2 - 35 * x - 40

/-- Theorem: f(5) = 110 -/
theorem f_of_5_eq_110 : f 5 = 110 := by
  sorry

end f_of_5_eq_110_l533_53329


namespace pepperoni_coverage_fraction_l533_53312

/-- Represents a circular pizza with pepperoni toppings -/
structure PizzaWithPepperoni where
  pizza_diameter : ℝ
  pepperoni_count_across : ℕ
  total_pepperoni_count : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def fraction_covered (p : PizzaWithPepperoni) : ℚ :=
  sorry

/-- Theorem stating that for a pizza with given specifications, 
    the fraction covered by pepperoni is 4/9 -/
theorem pepperoni_coverage_fraction 
  (p : PizzaWithPepperoni) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_count_across = 9)
  (h3 : p.total_pepperoni_count = 36) : 
  fraction_covered p = 4/9 := by
  sorry

end pepperoni_coverage_fraction_l533_53312


namespace particular_number_problem_l533_53327

theorem particular_number_problem : ∃! x : ℚ, ((x / 23) - 67) * 2 = 102 :=
  by sorry

end particular_number_problem_l533_53327


namespace negation_of_exactly_one_intersection_negation_of_if_3_or_4_then_equation_l533_53354

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the proposition for exactly one intersection point
def exactly_one_intersection (a b c : ℝ) : Prop :=
  ∃! x, f a b c x = 0

-- Define the proposition for no or at least two intersection points
def no_or_at_least_two_intersections (a b c : ℝ) : Prop :=
  (∀ x, f a b c x ≠ 0) ∨ (∃ x y, x ≠ y ∧ f a b c x = 0 ∧ f a b c y = 0)

-- Theorem for the negation of the first proposition
theorem negation_of_exactly_one_intersection (a b c : ℝ) :
  ¬(exactly_one_intersection a b c) ↔ no_or_at_least_two_intersections a b c :=
sorry

-- Define the proposition for the second statement
def if_3_or_4_then_equation : Prop :=
  (3^2 - 7*3 + 12 = 0) ∧ (4^2 - 7*4 + 12 = 0)

-- Theorem for the negation of the second proposition
theorem negation_of_if_3_or_4_then_equation :
  ¬if_3_or_4_then_equation ↔ (3^2 - 7*3 + 12 ≠ 0) ∨ (4^2 - 7*4 + 12 ≠ 0) :=
sorry

end negation_of_exactly_one_intersection_negation_of_if_3_or_4_then_equation_l533_53354


namespace relay_race_sarah_speed_l533_53380

/-- Relay race problem -/
theorem relay_race_sarah_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (sadie_speed : ℝ) 
  (sadie_time : ℝ) 
  (ariana_speed : ℝ) 
  (ariana_time : ℝ) 
  (h1 : total_distance = 17) 
  (h2 : total_time = 4.5) 
  (h3 : sadie_speed = 3) 
  (h4 : sadie_time = 2) 
  (h5 : ariana_speed = 6) 
  (h6 : ariana_time = 0.5) : 
  (total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time)) / 
  (total_time - sadie_time - ariana_time) = 4 := by
  sorry


end relay_race_sarah_speed_l533_53380


namespace max_distance_MN_l533_53352

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

def C₂ (x y : ℝ) : Prop := ∃ φ : ℝ, x = 2 * Real.cos φ ∧ y = Real.sin φ

def C₃ (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

-- Define the transformation
def transformation (x y : ℝ) : Prop := x^2 = 2*x ∧ y^2 = y

-- Define the tangent point condition
def is_tangent_point (M N : ℝ × ℝ) : Prop :=
  C₂ M.1 M.2 ∧ C₃ N.1 N.2 ∧
  ∃ t : ℝ, (N.1 - M.1)^2 + (N.2 - M.2)^2 = t^2 ∧
           ∀ P : ℝ × ℝ, C₃ P.1 P.2 → (P.1 - M.1)^2 + (P.2 - M.2)^2 ≥ t^2

-- Theorem statement
theorem max_distance_MN :
  ∀ M N : ℝ × ℝ, is_tangent_point M N →
  (N.1 - M.1)^2 + (N.2 - M.2)^2 ≤ 15 :=
sorry

end max_distance_MN_l533_53352


namespace quadratic_inequality_l533_53375

/-- Given a quadratic function f(x) = x^2 + 4x + c, prove that f(1) > c > f(-2) -/
theorem quadratic_inequality (c : ℝ) : let f : ℝ → ℝ := λ x ↦ x^2 + 4*x + c
  f 1 > c ∧ c > f (-2) := by
  sorry

end quadratic_inequality_l533_53375


namespace kelly_snacks_weight_l533_53397

theorem kelly_snacks_weight (peanuts raisins total : Real) : 
  peanuts = 0.1 → raisins = 0.4 → total = peanuts + raisins → total = 0.5 := by
  sorry

end kelly_snacks_weight_l533_53397


namespace and_implies_or_but_not_conversely_l533_53347

-- Define propositions p and q
variable (p q : Prop)

-- State the theorem
theorem and_implies_or_but_not_conversely :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) :=
by
  sorry


end and_implies_or_but_not_conversely_l533_53347


namespace largest_share_example_l533_53315

/-- Represents the profit distribution for partners in a business --/
structure ProfitDistribution where
  ratios : List Nat
  total_profit : ℚ

/-- Calculates the largest share of profit given a profit distribution --/
def largest_share (pd : ProfitDistribution) : ℚ :=
  let total_parts := pd.ratios.sum
  let value_per_part := pd.total_profit / total_parts
  value_per_part * pd.ratios.maximum.getD 0

/-- Theorem stating that for the given profit distribution, the largest share is $11,333.35 --/
theorem largest_share_example : 
  let pd : ProfitDistribution := { 
    ratios := [2, 3, 4, 1, 5],
    total_profit := 34000
  }
  largest_share pd = 11333.35 := by
  sorry

#eval largest_share { ratios := [2, 3, 4, 1, 5], total_profit := 34000 }

end largest_share_example_l533_53315


namespace lee_lawn_mowing_earnings_l533_53338

/-- Lee's lawn mowing earnings problem -/
theorem lee_lawn_mowing_earnings :
  ∀ (charge_per_lawn : ℕ) (lawns_mowed : ℕ) (tip_amount : ℕ) (num_tippers : ℕ),
    charge_per_lawn = 33 →
    lawns_mowed = 16 →
    tip_amount = 10 →
    num_tippers = 3 →
    charge_per_lawn * lawns_mowed + tip_amount * num_tippers = 558 :=
by
  sorry


end lee_lawn_mowing_earnings_l533_53338


namespace range_of_a_l533_53340

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by sorry

end range_of_a_l533_53340


namespace dogs_with_neither_l533_53364

/-- Given a kennel with dogs, prove the number of dogs wearing neither tags nor flea collars -/
theorem dogs_with_neither (total : ℕ) (with_tags : ℕ) (with_collars : ℕ) (with_both : ℕ) 
  (h1 : total = 80)
  (h2 : with_tags = 45)
  (h3 : with_collars = 40)
  (h4 : with_both = 6) :
  total - (with_tags + with_collars - with_both) = 1 := by
  sorry

#check dogs_with_neither

end dogs_with_neither_l533_53364


namespace sock_pair_count_l533_53398

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_sock_pairs (black white blue : ℕ) : ℕ :=
  black * white + black * blue + white * blue

/-- Theorem: There are 107 ways to choose a pair of socks of different colors
    from a drawer containing 5 black socks, 6 white socks, and 7 blue socks -/
theorem sock_pair_count :
  different_color_sock_pairs 5 6 7 = 107 := by
  sorry

end sock_pair_count_l533_53398


namespace supervisors_average_salary_l533_53353

/-- Given the following conditions in a factory:
  1. The average monthly salary of laborers and supervisors combined is 1250.
  2. There are 6 supervisors.
  3. There are 42 laborers.
  4. The average monthly salary of the laborers is 950.
  Prove that the average monthly salary of the supervisors is 3350. -/
theorem supervisors_average_salary
  (total_average : ℚ)
  (num_supervisors : ℕ)
  (num_laborers : ℕ)
  (laborers_average : ℚ)
  (h1 : total_average = 1250)
  (h2 : num_supervisors = 6)
  (h3 : num_laborers = 42)
  (h4 : laborers_average = 950) :
  (total_average * (num_supervisors + num_laborers) - laborers_average * num_laborers) / num_supervisors = 3350 := by
  sorry


end supervisors_average_salary_l533_53353


namespace brownie_division_l533_53349

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of brownies -/
def pan : Dimensions := ⟨15, 25⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 5⟩

/-- Theorem stating that the pan can be divided into exactly 25 pieces -/
theorem brownie_division :
  (area pan) / (area piece) = 25 := by sorry

end brownie_division_l533_53349


namespace cloth_cost_calculation_l533_53357

/-- The total cost of cloth given the length and price per metre -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem: The total cost of 9.25 m of cloth at $45 per metre is $416.25 -/
theorem cloth_cost_calculation :
  total_cost 9.25 45 = 416.25 := by
  sorry

end cloth_cost_calculation_l533_53357


namespace a_plus_b_equals_one_l533_53306

-- Define the universe U as the real numbers
def U : Type := ℝ

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | (x^2 + a*x + b)*(x - 1) = 0}

-- Define the theorem
theorem a_plus_b_equals_one (a b : ℝ) (B : Set ℝ) :
  (∃ (B : Set ℝ), (A a b ∩ B = {1, 2}) ∧ (A a b ∩ (Set.univ \ B) = {3})) →
  a + b = 1 := by
  sorry

end a_plus_b_equals_one_l533_53306


namespace tile_arrangement_count_l533_53335

/-- The number of distinguishable arrangements of tiles -/
def tileArrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 1 purple, 3 green, and 2 yellow tiles is 420 -/
theorem tile_arrangement_count :
  tileArrangements 1 1 3 2 = 420 := by
  sorry

end tile_arrangement_count_l533_53335


namespace arithmetic_expression_evaluation_l533_53399

theorem arithmetic_expression_evaluation : 2 - (3 - 4) - (5 - 6 - 7) = 11 := by
  sorry

end arithmetic_expression_evaluation_l533_53399


namespace function_properties_l533_53377

-- Define the function f from X to Y
variable {X Y : Type*}
variable (f : X → Y)

-- Theorem stating that none of the given statements are necessarily true for all functions
theorem function_properties :
  (∃ y : Y, ∀ x : X, f x ≠ y) ∧  -- Some elements in Y might not have a preimage in X
  (∃ x₁ x₂ : X, x₁ ≠ x₂ ∧ f x₁ = f x₂) ∧  -- Different elements in X can have the same image in Y
  (∃ y : Y, True)  -- Y is not empty
  :=
by sorry

end function_properties_l533_53377


namespace expansion_equality_l533_53370

-- Define a positive integer n
variable (n : ℕ+)

-- Define the condition that the coefficient of x^3 is the same in both expansions
def coefficient_equality (n : ℕ+) : Prop :=
  (Nat.choose (2 * n) 3) = 2 * (Nat.choose n 1)

-- Theorem statement
theorem expansion_equality (n : ℕ+) (h : coefficient_equality n) :
  n = 2 ∧ 
  ∀ k : ℕ, k ≤ n → 2 * (Nat.choose n k) ≤ 4 :=
sorry

end expansion_equality_l533_53370


namespace min_value_expression_l533_53393

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y^2 * z = 64) :
  x^2 + 8*x*y + 8*y^2 + 4*z^2 ≥ 1536 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y^2 * z = 64 ∧ x^2 + 8*x*y + 8*y^2 + 4*z^2 = 1536 :=
by sorry

end min_value_expression_l533_53393


namespace sum_of_reciprocal_divisors_eq_two_l533_53323

def sum_of_divisors (n : ℕ) : ℚ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => x)

def sum_of_reciprocal_divisors (n : ℕ) : ℚ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => 1 / x)

theorem sum_of_reciprocal_divisors_eq_two (n : ℕ) (h : sum_of_divisors n = 2 * n) :
  sum_of_reciprocal_divisors n = 2 :=
by sorry

end sum_of_reciprocal_divisors_eq_two_l533_53323


namespace ellipse_hyperbola_asymptotes_l533_53356

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 - y^2/25 = 1

-- Define the hyperbola
def hyperbola (K : ℝ) (x y : ℝ) : Prop := x^2/K + y^2/25 = 1

-- Define the asymptote condition
def same_asymptotes (K : ℝ) : Prop := ∀ (x y : ℝ), y = (5/4)*x ↔ y = (5/Real.sqrt K)*x

-- Theorem statement
theorem ellipse_hyperbola_asymptotes (K : ℝ) : 
  (∀ (x y : ℝ), ellipse x y ∧ hyperbola K x y) → same_asymptotes K → K = 16 := by
  sorry

end ellipse_hyperbola_asymptotes_l533_53356
