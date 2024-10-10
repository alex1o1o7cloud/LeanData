import Mathlib

namespace cubic_transformation_1993_l3222_322254

/-- Cubic transformation: sum of cubes of digits --/
def cubicTransform (n : ℕ) : ℕ := sorry

/-- Sequence of cubic transformations starting from n --/
def cubicSequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | i + 1 => cubicTransform (cubicSequence n i)

/-- Predicate for sequence alternating between two values --/
def alternatesBetween (seq : ℕ → ℕ) (a b : ℕ) : Prop :=
  ∃ k : ℕ, ∀ i ≥ k, seq i = a ∧ seq (i + 1) = b ∨ seq i = b ∧ seq (i + 1) = a

theorem cubic_transformation_1993 :
  alternatesBetween (cubicSequence 1993) 1459 919 := by sorry

end cubic_transformation_1993_l3222_322254


namespace x_value_l3222_322227

theorem x_value : ∃ x : ℝ, (0.65 * x = 0.20 * 617.50) ∧ (x = 190) := by
  sorry

end x_value_l3222_322227


namespace geography_book_count_l3222_322291

/-- Given a shelf of books with specific counts, calculate the number of geography books. -/
theorem geography_book_count (total : ℕ) (history : ℕ) (math : ℕ) 
  (h_total : total = 100)
  (h_history : history = 32)
  (h_math : math = 43) :
  total - history - math = 25 := by
  sorry

end geography_book_count_l3222_322291


namespace parabola_area_l3222_322209

-- Define the two parabolas
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 8 - x^2

-- Define the region
def R : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem parabola_area : 
  (∫ (x : ℝ) in R, g x - f x) = 64/3 := by sorry

end parabola_area_l3222_322209


namespace large_pizza_cost_l3222_322279

/-- Represents the cost and size of a pizza --/
structure Pizza where
  side_length : ℝ
  cost : ℝ

/-- Calculates the area of a square pizza --/
def pizza_area (p : Pizza) : ℝ := p.side_length ^ 2

theorem large_pizza_cost : ∃ (large_pizza : Pizza),
  let small_pizza := Pizza.mk 12 10
  let total_budget := 60
  let separate_purchase_area := 2 * (total_budget / small_pizza.cost * pizza_area small_pizza)
  large_pizza.side_length = 18 ∧
  large_pizza.cost = 21.6 ∧
  (total_budget / large_pizza.cost * pizza_area large_pizza) = separate_purchase_area + 36 := by
  sorry

end large_pizza_cost_l3222_322279


namespace patricia_hair_length_l3222_322257

/-- Given Patricia's hair growth scenario, prove the desired hair length after donation -/
theorem patricia_hair_length 
  (current_length : ℕ) 
  (donation_length : ℕ) 
  (growth_needed : ℕ) 
  (h1 : current_length = 14)
  (h2 : donation_length = 23)
  (h3 : growth_needed = 21) : 
  current_length + growth_needed - donation_length = 12 := by
  sorry

end patricia_hair_length_l3222_322257


namespace debt_payment_difference_l3222_322220

/-- Given a debt paid in 40 installments with specific conditions, 
    prove the difference between later and earlier payments. -/
theorem debt_payment_difference (first_payment : ℝ) (average_payment : ℝ) 
    (h1 : first_payment = 410)
    (h2 : average_payment = 442.5) : 
    ∃ (difference : ℝ), 
      20 * first_payment + 20 * (first_payment + difference) = 40 * average_payment ∧ 
      difference = 65 := by
  sorry

end debt_payment_difference_l3222_322220


namespace sqrt_meaningful_range_l3222_322270

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 + 3 * x) ↔ x ≥ -1/3 := by
  sorry

end sqrt_meaningful_range_l3222_322270


namespace sum_of_coefficients_l3222_322288

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 36 := by
sorry

end sum_of_coefficients_l3222_322288


namespace total_widgets_sold_is_360_l3222_322282

/-- The sum of an arithmetic sequence with first term 3, common difference 3, and 15 terms -/
def widget_sales_sum : ℕ :=
  let first_term := 3
  let common_difference := 3
  let num_days := 15
  (num_days * (2 * first_term + (num_days - 1) * common_difference)) / 2

/-- Theorem stating that the total number of widgets sold is 360 -/
theorem total_widgets_sold_is_360 : widget_sales_sum = 360 := by
  sorry

end total_widgets_sold_is_360_l3222_322282


namespace max_value_of_trig_function_l3222_322224

theorem max_value_of_trig_function :
  ∀ x : ℝ, (π / (1 + Real.tan x ^ 2)) ≤ π :=
by sorry

end max_value_of_trig_function_l3222_322224


namespace mike_work_hours_l3222_322202

def wash_time : ℕ := 10
def oil_change_time : ℕ := 15
def tire_change_time : ℕ := 30
def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def tire_sets_changed : ℕ := 2

theorem mike_work_hours : 
  (cars_washed * wash_time + cars_oil_changed * oil_change_time + tire_sets_changed * tire_change_time) / 60 = 4 := by
  sorry

end mike_work_hours_l3222_322202


namespace sand_remaining_l3222_322266

/-- Calculates the remaining amount of sand in a truck after transit -/
theorem sand_remaining (initial_sand lost_sand : ℝ) :
  initial_sand ≥ 0 →
  lost_sand ≥ 0 →
  lost_sand ≤ initial_sand →
  initial_sand - lost_sand = initial_sand - lost_sand :=
by
  sorry

#check sand_remaining 4.1 2.4

end sand_remaining_l3222_322266


namespace original_number_proof_l3222_322255

theorem original_number_proof : 
  ∀ x : ℝ, ((x / 8) * 16 + 20) / 4 = 34 → x = 58 := by
sorry

end original_number_proof_l3222_322255


namespace no_integer_satisfies_condition_l3222_322265

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_integer_satisfies_condition : 
  ¬ ∃ n : ℕ+, (n : ℕ) % sum_of_digits n = 0 → sum_of_digits (n * sum_of_digits n) = 3 := by
  sorry

end no_integer_satisfies_condition_l3222_322265


namespace lowest_price_pet_food_l3222_322248

/-- Calculates the final price of a pet food container after two consecutive discounts -/
def final_price (msrp : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  msrp * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the lowest possible price of a $35 pet food container
    after a maximum 30% discount and an additional 20% discount is $19.60 -/
theorem lowest_price_pet_food :
  final_price 35 0.3 0.2 = 19.60 := by
  sorry

end lowest_price_pet_food_l3222_322248


namespace andrew_flooring_planks_l3222_322298

/-- The number of wooden planks Andrew bought for his flooring project -/
def total_planks : ℕ := 91

/-- The number of planks used in Andrew's bedroom -/
def bedroom_planks : ℕ := 8

/-- The number of planks used in the living room -/
def living_room_planks : ℕ := 20

/-- The number of planks used in the kitchen -/
def kitchen_planks : ℕ := 11

/-- The number of planks used in the dining room -/
def dining_room_planks : ℕ := 13

/-- The number of planks used in the guest bedroom -/
def guest_bedroom_planks : ℕ := bedroom_planks - 2

/-- The number of planks used in each hallway -/
def hallway_planks : ℕ := 4

/-- The number of planks used in the study -/
def study_planks : ℕ := guest_bedroom_planks + 3

/-- The number of planks ruined in each bedroom -/
def bedroom_ruined_planks : ℕ := 3

/-- The number of planks ruined in the living room -/
def living_room_ruined_planks : ℕ := 2

/-- The number of planks ruined in the study -/
def study_ruined_planks : ℕ := 1

/-- The number of leftover planks -/
def leftover_planks : ℕ := 7

/-- The number of hallways -/
def number_of_hallways : ℕ := 2

theorem andrew_flooring_planks :
  total_planks = 
    bedroom_planks + bedroom_ruined_planks +
    living_room_planks + living_room_ruined_planks +
    kitchen_planks +
    dining_room_planks +
    guest_bedroom_planks + bedroom_ruined_planks +
    (hallway_planks * number_of_hallways) +
    study_planks + study_ruined_planks +
    leftover_planks :=
by sorry

end andrew_flooring_planks_l3222_322298


namespace division_problem_l3222_322294

theorem division_problem : 12 / (2 / (5 - 3)) = 12 := by
  sorry

end division_problem_l3222_322294


namespace max_remainder_division_l3222_322225

theorem max_remainder_division (n : ℕ) : 
  (n % 6 < 6) → (n / 6 = 18) → (n % 6 = 5) → n = 113 := by
  sorry

end max_remainder_division_l3222_322225


namespace factor_of_expression_l3222_322242

theorem factor_of_expression (x y z : ℝ) :
  ∃ (k : ℝ), x^2 - y^2 - z^2 + 2*y*z + 3*x + 2*y - 4*z = (x + y - z) * k := by
  sorry

end factor_of_expression_l3222_322242


namespace select_and_arrange_theorem_l3222_322275

/-- The number of ways to select k items from n items -/
def combination (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange k items -/
def permutation (k : ℕ) : ℕ := Nat.factorial k

/-- The total number of people -/
def total_people : ℕ := 9

/-- The number of people to be selected and arranged -/
def selected_people : ℕ := 3

/-- The number of ways to select and arrange people -/
def ways_to_select_and_arrange : ℕ := combination total_people selected_people * permutation selected_people

theorem select_and_arrange_theorem : ways_to_select_and_arrange = 504 := by
  sorry

end select_and_arrange_theorem_l3222_322275


namespace simplify_expression_l3222_322280

theorem simplify_expression (r s : ℝ) : 
  (2 * r^2 + 5 * r - 6 * s + 4) - (r^2 + 9 * r - 4 * s - 2) = r^2 - 4 * r - 2 * s + 6 := by
  sorry

end simplify_expression_l3222_322280


namespace range_of_a_l3222_322262

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≥ 0

-- Define the set A corresponding to ¬p
def A : Set ℝ := {x | x < -2 ∨ x > 10}

-- Define the set B corresponding to q
def B (a : ℝ) : Set ℝ := {x | x ≤ 1 - a ∨ x ≥ 1 + a}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧ A ⊆ B a ∧ A ≠ B a) → (0 < a ∧ a ≤ 3) :=
by sorry

end range_of_a_l3222_322262


namespace cos_pi_minus_alpha_l3222_322241

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 3) :
  Real.cos (π - α) = -1 / 3 := by
  sorry

end cos_pi_minus_alpha_l3222_322241


namespace claire_gerbils_l3222_322251

/-- Represents the number of gerbils Claire has -/
def num_gerbils : ℕ := 60

/-- Represents the number of hamsters Claire has -/
def num_hamsters : ℕ := 30

/-- The total number of pets Claire has -/
def total_pets : ℕ := 90

/-- The total number of male pets Claire has -/
def total_male_pets : ℕ := 25

theorem claire_gerbils :
  (num_gerbils + num_hamsters = total_pets) ∧
  (num_gerbils / 4 + num_hamsters / 3 = total_male_pets) →
  num_gerbils = 60 := by
  sorry

end claire_gerbils_l3222_322251


namespace existence_of_m_l3222_322284

theorem existence_of_m (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h_cd : c * d = 1) :
  ∃ m : ℕ, (0 < m) ∧ (a * b ≤ m^2) ∧ (m^2 ≤ (a + c) * (b + d)) := by
  sorry

end existence_of_m_l3222_322284


namespace tenth_triangular_number_l3222_322213

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tenth_triangular_number : triangular_number 10 = 55 := by
  sorry

end tenth_triangular_number_l3222_322213


namespace a_minus_b_values_l3222_322243

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  (a - b = 4) ∨ (a - b = 8) :=
sorry

end a_minus_b_values_l3222_322243


namespace boat_upstream_distance_l3222_322230

/-- Represents the speed and distance calculations for a boat in a stream -/
def boat_in_stream (boat_speed : ℝ) (downstream_distance : ℝ) : ℝ :=
  let stream_speed := downstream_distance - boat_speed
  boat_speed - stream_speed

theorem boat_upstream_distance 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : boat_speed = 11) 
  (h2 : downstream_distance = 16) : 
  boat_in_stream boat_speed downstream_distance = 6 := by
  sorry

end boat_upstream_distance_l3222_322230


namespace rectangle_dimension_change_l3222_322219

theorem rectangle_dimension_change (L B : ℝ) (p : ℝ) 
  (h1 : L > 0) (h2 : B > 0) (h3 : p > 0) :
  (L * (1 + p)) * (B * 0.75) = L * B * 1.05 → p = 0.4 := by
  sorry

end rectangle_dimension_change_l3222_322219


namespace conic_parabola_focus_coincidence_l3222_322264

/-- Given a conic section and a parabola, prove that the parameter m of the conic section is 9 when their foci coincide. -/
theorem conic_parabola_focus_coincidence (m : ℝ) : 
  m ≠ 0 → m ≠ 5 → 
  (∃ (x y : ℝ), x^2 / m + y^2 / 5 = 1) →
  (∃ (x y : ℝ), y^2 = 8*x) →
  (∃ (x₀ y₀ : ℝ), x₀^2 / m + y₀^2 / 5 = 1 ∧ y₀^2 = 8*x₀ ∧ x₀ = 2 ∧ y₀ = 0) →
  m = 9 :=
by sorry

end conic_parabola_focus_coincidence_l3222_322264


namespace ellipse_and_midpoint_trajectory_l3222_322289

/-- Definition of the ellipse -/
def Ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the midpoint trajectory -/
def MidpointTrajectory (x y : ℝ) : Prop := (x - 1/2)^2 + 4 * (y - 1/4)^2 = 1

/-- Theorem: The standard equation of the ellipse and the midpoint trajectory -/
theorem ellipse_and_midpoint_trajectory :
  (∀ x y, Ellipse x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∀ x₀ y₀ x y, Ellipse x₀ y₀ → x = (x₀ + 1) / 2 ∧ y = (y₀ + 1/2) / 2 → MidpointTrajectory x y) := by
  sorry

end ellipse_and_midpoint_trajectory_l3222_322289


namespace wendy_scholarship_amount_l3222_322217

theorem wendy_scholarship_amount 
  (wendy kelly nina : ℕ)  -- Scholarship amounts for each person
  (h1 : nina = kelly - 8000)  -- Nina's scholarship is $8000 less than Kelly's
  (h2 : kelly = 2 * wendy)    -- Kelly's scholarship is twice Wendy's
  (h3 : wendy + kelly + nina = 92000)  -- Total scholarship amount
  : wendy = 20000 := by
  sorry

end wendy_scholarship_amount_l3222_322217


namespace quadratic_equation_solution_l3222_322267

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by
sorry

end quadratic_equation_solution_l3222_322267


namespace max_value_theorem_l3222_322256

theorem max_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_constraint : x^2 - x*y + 2*y^2 = 8) :
  x^2 + x*y + 2*y^2 ≤ (72 + 32*Real.sqrt 2) / 7 := by
  sorry

end max_value_theorem_l3222_322256


namespace sock_counting_l3222_322208

theorem sock_counting (initial : ℕ) (thrown_away : ℕ) (new_bought : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + new_bought = initial + new_bought - thrown_away :=
by sorry

end sock_counting_l3222_322208


namespace sarah_candy_count_l3222_322268

/-- The number of candy pieces Sarah received for Halloween -/
def total_candy : ℕ := sorry

/-- The number of candy pieces Sarah ate -/
def eaten_candy : ℕ := 36

/-- The number of piles Sarah made with the remaining candy -/
def number_of_piles : ℕ := 8

/-- The number of candy pieces in each pile -/
def pieces_per_pile : ℕ := 9

/-- Theorem stating that the total number of candy pieces Sarah received is 108 -/
theorem sarah_candy_count : total_candy = 108 := by
  sorry

end sarah_candy_count_l3222_322268


namespace abs_negative_two_thirds_equals_two_thirds_l3222_322210

theorem abs_negative_two_thirds_equals_two_thirds : 
  |(-2 : ℚ) / 3| = 2 / 3 := by sorry

end abs_negative_two_thirds_equals_two_thirds_l3222_322210


namespace share_of_y_l3222_322278

/-- The share of y in a sum divided among x, y, and z, where for each rupee x gets,
    y gets 45 paisa and z gets 50 paisa, and the total amount is Rs. 78. -/
theorem share_of_y (x y z : ℝ) : 
  x + y + z = 78 →  -- Total amount condition
  y = 0.45 * x →    -- Relationship between y and x
  z = 0.5 * x →     -- Relationship between z and x
  y = 18 :=         -- Share of y
by sorry

end share_of_y_l3222_322278


namespace ab_power_2022_l3222_322276

theorem ab_power_2022 (a b : ℝ) (h : |3*a + 1| + (b - 3)^2 = 0) : (a*b)^2022 = 1 := by
  sorry

end ab_power_2022_l3222_322276


namespace sum_inequality_l3222_322253

theorem sum_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) : a + b ≤ 3 * c := by
  sorry

end sum_inequality_l3222_322253


namespace mean_equality_implies_y_value_l3222_322287

theorem mean_equality_implies_y_value : ∃ y : ℝ,
  (4 + 7 + 11 + 14) / 4 = (10 + y + 5) / 3 ∧ y = 12 := by
  sorry

end mean_equality_implies_y_value_l3222_322287


namespace water_height_in_cylinder_l3222_322203

/-- Given a cone with base radius 10 cm and height 15 cm, when its volume of water is poured into a cylinder with base radius 20 cm, the height of water in the cylinder is 1.25 cm. -/
theorem water_height_in_cylinder (π : ℝ) : 
  let cone_radius : ℝ := 10
  let cone_height : ℝ := 15
  let cylinder_radius : ℝ := 20
  let cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height
  let cylinder_height : ℝ := cone_volume / (π * cylinder_radius^2)
  cylinder_height = 1.25 := by
  sorry

end water_height_in_cylinder_l3222_322203


namespace distance_between_points_l3222_322212

theorem distance_between_points :
  let A : ℝ × ℝ := (8, -5)
  let B : ℝ × ℝ := (0, 10)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 17 := by
sorry

end distance_between_points_l3222_322212


namespace percent_commutation_l3222_322263

theorem percent_commutation (x : ℝ) (h : 0.3 * (0.4 * x) = 45) : 0.4 * (0.3 * x) = 45 := by
  sorry

end percent_commutation_l3222_322263


namespace geometric_sequence_product_l3222_322260

theorem geometric_sequence_product (x y z : ℝ) : 
  1 < x ∧ x < y ∧ y < z ∧ z < 4 →
  (∃ r : ℝ, r > 0 ∧ x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) →
  1 * x * y * z * 4 = 32 := by
sorry

end geometric_sequence_product_l3222_322260


namespace sin_equality_n_512_l3222_322240

theorem sin_equality_n_512 (n : ℤ) :
  -100 ≤ n ∧ n ≤ 100 ∧ Real.sin (n * π / 180) = Real.sin (512 * π / 180) → n = 28 :=
by sorry

end sin_equality_n_512_l3222_322240


namespace simple_interest_rate_for_doubling_l3222_322236

theorem simple_interest_rate_for_doubling (principal : ℝ) (h : principal > 0) : 
  ∃ (rate : ℝ), 
    (rate > 0) ∧ 
    (rate < 100) ∧
    (principal * (1 + rate / 100 * 25) = 2 * principal) ∧
    (rate = 4) := by
  sorry

end simple_interest_rate_for_doubling_l3222_322236


namespace limit_between_exponentials_l3222_322205

theorem limit_between_exponentials (a : ℝ) (ha : a > 0) :
  Real.exp a < (Real.exp (a + 1)) / (Real.exp 1 - 1) ∧
  (Real.exp (a + 1)) / (Real.exp 1 - 1) < Real.exp (a + 1) := by
  sorry

end limit_between_exponentials_l3222_322205


namespace acute_angles_inequality_l3222_322296

theorem acute_angles_inequality (α β : Real) 
  (h_α_acute : 0 < α ∧ α < π / 2) 
  (h_β_acute : 0 < β ∧ β < π / 2) : 
  Real.sin α ^ 3 * Real.cos β ^ 3 + Real.sin α ^ 3 * Real.sin β ^ 3 + Real.cos α ^ 3 ≥ Real.sqrt 3 / 3 := by
  sorry

end acute_angles_inequality_l3222_322296


namespace airport_distance_airport_distance_proof_l3222_322211

theorem airport_distance : ℝ → Prop :=
  fun d : ℝ =>
    let initial_speed : ℝ := 45
    let speed_increase : ℝ := 20
    let late_time : ℝ := 0.75  -- 45 minutes in hours
    let early_time : ℝ := 0.25  -- 15 minutes in hours
    let t : ℝ := (d / initial_speed) - late_time  -- Time if he continued at initial speed
    
    (d = initial_speed * (t + late_time)) ∧
    (d - initial_speed = (initial_speed + speed_increase) * (t - early_time)) →
    d = 61.875

-- The proof would go here
theorem airport_distance_proof : airport_distance 61.875 := by
  sorry

end airport_distance_airport_distance_proof_l3222_322211


namespace square_area_on_circle_and_tangent_l3222_322250

/-- Given a circle with radius 5 and a square with two vertices on the circle
    and two vertices on a tangent to the circle, the area of the square is 64. -/
theorem square_area_on_circle_and_tangent :
  ∀ (circle : ℝ → ℝ → Prop) (square : ℝ → ℝ → Prop) (r : ℝ),
  (r = 5) →  -- The radius of the circle is 5
  (∃ (A B C D : ℝ × ℝ),
    -- Two vertices of the square lie on the circle
    circle A.1 A.2 ∧ circle C.1 C.2 ∧
    -- The other two vertices lie on a tangent to the circle
    (∃ (t : ℝ → ℝ → Prop), t B.1 B.2 ∧ t D.1 D.2) ∧
    -- A, B, C, D form a square
    square A.1 A.2 ∧ square B.1 B.2 ∧ square C.1 C.2 ∧ square D.1 D.2) →
  (∃ (area : ℝ), area = 64) :=
by sorry

end square_area_on_circle_and_tangent_l3222_322250


namespace equality_holds_l3222_322235

-- Define the property P for the function f
def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f (x + y)| ≥ |f x + f y|

-- State the theorem
theorem equality_holds (f : ℝ → ℝ) (h : satisfies_inequality f) :
  ∀ x y : ℝ, |f (x + y)| = |f x + f y| := by
  sorry

end equality_holds_l3222_322235


namespace smallest_q_for_decimal_sequence_l3222_322223

theorem smallest_q_for_decimal_sequence (p q : ℕ+) : 
  (p : ℚ) / q = 0.123456789 → q ≥ 10989019 := by sorry

end smallest_q_for_decimal_sequence_l3222_322223


namespace compound_weight_l3222_322234

/-- Given a compound with a molecular weight of 1050, 
    the total weight of 6 moles of this compound is 6300 grams. -/
theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 1050 → moles = 6 → moles * molecular_weight = 6300 := by
  sorry

end compound_weight_l3222_322234


namespace acme_soup_words_count_l3222_322271

/-- Represents the number of times each vowel (A, E, I, O, U) appears -/
def vowel_count : ℕ := 5

/-- Represents the number of times Y appears -/
def y_count : ℕ := 3

/-- Represents the length of words to be formed -/
def word_length : ℕ := 5

/-- Represents the number of vowels (A, E, I, O, U) -/
def num_vowels : ℕ := 5

/-- Calculates the number of five-letter words that can be formed -/
def acme_soup_words : ℕ := 
  (num_vowels ^ word_length) + 
  (word_length * (num_vowels ^ (word_length - 1))) +
  (Nat.choose word_length 2 * (num_vowels ^ (word_length - 2))) +
  (Nat.choose word_length 3 * (num_vowels ^ (word_length - 3)))

theorem acme_soup_words_count : acme_soup_words = 7750 := by
  sorry

end acme_soup_words_count_l3222_322271


namespace quadratic_expression_value_l3222_322274

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 11) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 2041 / 25 :=
by sorry

end quadratic_expression_value_l3222_322274


namespace cross_in_square_l3222_322252

theorem cross_in_square (s : ℝ) (h : s > 0) : 
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by
  sorry

end cross_in_square_l3222_322252


namespace min_tan_angle_l3222_322206

/-- The set of complex numbers with nonnegative real and imaginary parts -/
def S : Set ℂ :=
  {z : ℂ | z.re ≥ 0 ∧ z.im ≥ 0}

/-- The condition |z^2 + 2| ≤ |z| -/
def satisfiesCondition (z : ℂ) : Prop :=
  Complex.abs (z^2 + 2) ≤ Complex.abs z

/-- The angle between a complex number and the real axis -/
noncomputable def angle (z : ℂ) : ℝ :=
  Real.arctan (z.im / z.re)

/-- The main theorem -/
theorem min_tan_angle :
  ∃ (min_tan : ℝ), min_tan = Real.sqrt 7 ∧
  ∀ z ∈ S, satisfiesCondition z →
  Real.tan (angle z) ≥ min_tan :=
sorry

end min_tan_angle_l3222_322206


namespace points_A_B_D_collinear_l3222_322249

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def vector_AB (a b : V) : V := a + 2 • b
def vector_BC (a b : V) : V := -5 • a + 6 • b
def vector_CD (a b : V) : V := 7 • a - 2 • b

theorem points_A_B_D_collinear (a b : V) :
  ∃ (k : ℝ), vector_AB a b = k • (vector_AB a b + vector_BC a b + vector_CD a b) :=
sorry

end points_A_B_D_collinear_l3222_322249


namespace rectangular_box_surface_area_l3222_322283

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 140) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 21) : 
  2 * (a*b + b*c + c*a) = 784 := by
sorry

end rectangular_box_surface_area_l3222_322283


namespace circle_equation_chord_length_implies_k_min_distance_l3222_322277

-- Define the circle
def circle_center : ℝ × ℝ := (3, -2)
def circle_radius : ℝ := 5

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 1)
def point_B : ℝ × ℝ := (-2, -2)

-- Define the line l that the circle center lies on
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the chord line
def chord_line (k x y : ℝ) : Prop := k * x - y + 5 = 0

-- Define the line for minimum distance
def line_min_dist (x y : ℝ) : Prop := x - y + 5 = 0

-- Theorem statements
theorem circle_equation : 
  ∀ x y : ℝ, (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ↔
  (x - 3)^2 + (y + 2)^2 = 25 := by sorry

theorem chord_length_implies_k :
  ∃ k : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ,
    chord_line k x₁ y₁ ∧ chord_line k x₂ y₂ ∧
    (x₁ - circle_center.1)^2 + (y₁ - circle_center.2)^2 = circle_radius^2 ∧
    (x₂ - circle_center.1)^2 + (y₂ - circle_center.2)^2 = circle_radius^2 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 64) →
  k = -20/21 := by sorry

theorem min_distance :
  ∀ P Q : ℝ × ℝ,
  ((P.1 - circle_center.1)^2 + (P.2 - circle_center.2)^2 = circle_radius^2) →
  line_min_dist Q.1 Q.2 →
  ∃ d : ℝ, d ≥ 5 * Real.sqrt 2 - 5 ∧
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ≥ d^2 := by sorry

end circle_equation_chord_length_implies_k_min_distance_l3222_322277


namespace tomato_price_equation_l3222_322207

/-- The original price per pound of tomatoes -/
def P : ℝ := sorry

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 0.968888888888889

/-- The proportion of tomatoes that were not ruined -/
def remaining_proportion : ℝ := 0.9

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.09

theorem tomato_price_equation : 
  (1 + profit_percentage) * P = selling_price * remaining_proportion := by sorry

end tomato_price_equation_l3222_322207


namespace toms_profit_l3222_322247

/-- Calculate Tom's total profit from lawn mowing and side jobs --/
theorem toms_profit (small_lawns : ℕ) (small_price : ℕ)
                    (medium_lawns : ℕ) (medium_price : ℕ)
                    (large_lawns : ℕ) (large_price : ℕ)
                    (gas_expense : ℕ) (maintenance_expense : ℕ)
                    (weed_jobs : ℕ) (weed_price : ℕ)
                    (hedge_jobs : ℕ) (hedge_price : ℕ)
                    (rake_jobs : ℕ) (rake_price : ℕ) :
  small_lawns = 4 →
  small_price = 12 →
  medium_lawns = 3 →
  medium_price = 15 →
  large_lawns = 1 →
  large_price = 20 →
  gas_expense = 17 →
  maintenance_expense = 5 →
  weed_jobs = 2 →
  weed_price = 10 →
  hedge_jobs = 3 →
  hedge_price = 8 →
  rake_jobs = 1 →
  rake_price = 12 →
  (small_lawns * small_price + medium_lawns * medium_price + large_lawns * large_price +
   weed_jobs * weed_price + hedge_jobs * hedge_price + rake_jobs * rake_price) -
  (gas_expense + maintenance_expense) = 147 :=
by sorry

end toms_profit_l3222_322247


namespace investment_rate_calculation_l3222_322204

theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (first_rate : ℝ) 
  (second_rate : ℝ) 
  (desired_income : ℝ) :
  total_investment = 15000 →
  first_investment = 5000 →
  second_investment = 6000 →
  first_rate = 0.03 →
  second_rate = 0.045 →
  desired_income = 800 →
  let remaining_investment := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let remaining_income := desired_income - income_from_first - income_from_second
  remaining_income / remaining_investment = 0.095 := by sorry

end investment_rate_calculation_l3222_322204


namespace f_properties_l3222_322295

/-- A function f that is constant for all x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1/a| + |x - a + 1|

theorem f_properties (a : ℝ) (h_a : a > 0) (h_const : ∀ x y, f a x = f a y) :
  (∀ x, f a x ≥ 1) ∧
  (f a 3 < 11/2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4) :=
by sorry

end f_properties_l3222_322295


namespace syrup_volume_proof_l3222_322239

/-- Calculates the final volume of syrup in cups -/
def final_syrup_volume (original_volume : ℚ) (reduction_factor : ℚ) (added_sugar : ℚ) (cups_per_quart : ℚ) : ℚ :=
  original_volume * cups_per_quart * reduction_factor + added_sugar

theorem syrup_volume_proof (original_volume : ℚ) (reduction_factor : ℚ) (added_sugar : ℚ) (cups_per_quart : ℚ)
  (h1 : original_volume = 6)
  (h2 : reduction_factor = 1 / 12)
  (h3 : added_sugar = 1)
  (h4 : cups_per_quart = 4) :
  final_syrup_volume original_volume reduction_factor added_sugar cups_per_quart = 3 := by
  sorry

#eval final_syrup_volume 6 (1/12) 1 4

end syrup_volume_proof_l3222_322239


namespace sqrt_equality_implies_y_value_l3222_322231

theorem sqrt_equality_implies_y_value (y : ℝ) :
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 8 → y = 40 / 3 := by
  sorry

end sqrt_equality_implies_y_value_l3222_322231


namespace rug_coverage_l3222_322297

theorem rug_coverage (rug_length : ℝ) (rug_width : ℝ) (floor_area : ℝ) 
  (h1 : rug_length = 2)
  (h2 : rug_width = 7)
  (h3 : floor_area = 64)
  (h4 : rug_length * rug_width ≤ floor_area) : 
  (floor_area - rug_length * rug_width) / floor_area = 25 / 32 := by
  sorry

end rug_coverage_l3222_322297


namespace perimeter_of_PQRSU_l3222_322233

-- Define the points as 2D vectors
def P : ℝ × ℝ := (0, 8)
def Q : ℝ × ℝ := (4, 8)
def R : ℝ × ℝ := (4, 4)
def S : ℝ × ℝ := (9, 0)
def U : ℝ × ℝ := (0, 0)

-- Define the conditions
def PQ_length : ℝ := 4
def PU_length : ℝ := 8
def US_length : ℝ := 9

-- Define the right angles
def angle_PUQ_is_right : (P.1 - U.1) * (Q.1 - U.1) + (P.2 - U.2) * (Q.2 - U.2) = 0 := by sorry
def angle_UPQ_is_right : (U.1 - P.1) * (Q.1 - P.1) + (U.2 - P.2) * (Q.2 - P.2) = 0 := by sorry
def angle_PQR_is_right : (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 := by sorry

-- Define the theorem
theorem perimeter_of_PQRSU : 
  let perimeter := PQ_length + 
                   Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) + 
                   Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) + 
                   US_length + 
                   PU_length
  perimeter = 25 + Real.sqrt 41 := by sorry

end perimeter_of_PQRSU_l3222_322233


namespace compound_oxygen_atoms_l3222_322293

/-- Represents the atomic weights of elements in g/mol -/
def atomic_weight : String → ℝ
  | "C" => 12.01
  | "H" => 1.008
  | "O" => 16.00
  | _ => 0

/-- Calculates the total mass of a given number of atoms of an element -/
def total_mass (element : String) (num_atoms : ℕ) : ℝ :=
  (atomic_weight element) * (num_atoms : ℝ)

/-- Represents the molecular composition of the compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ
  molecular_weight : ℝ

/-- Calculates the total mass of the compound based on its composition -/
def compound_mass (c : Compound) : ℝ :=
  total_mass "C" c.carbon + total_mass "H" c.hydrogen + total_mass "O" c.oxygen

/-- The theorem to be proved -/
theorem compound_oxygen_atoms (c : Compound) 
  (h1 : c.carbon = 3)
  (h2 : c.hydrogen = 6)
  (h3 : c.molecular_weight = 58) :
  c.oxygen = 1 := by
  sorry

end compound_oxygen_atoms_l3222_322293


namespace boat_capacity_l3222_322261

theorem boat_capacity (trips_per_day : ℕ) (total_people : ℕ) (total_days : ℕ) :
  trips_per_day = 4 →
  total_people = 96 →
  total_days = 2 →
  total_people / (trips_per_day * total_days) = 12 :=
by
  sorry

end boat_capacity_l3222_322261


namespace proposition_p_negation_and_range_l3222_322216

theorem proposition_p_negation_and_range (a : ℝ) :
  (¬∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ∧ 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0 → 0 < a ∧ a < 1) :=
by sorry

end proposition_p_negation_and_range_l3222_322216


namespace sqrt_sum_squares_integer_sqrt_sum_squares_not_integer_1_sqrt_sum_squares_not_integer_2_sqrt_sum_squares_not_integer_3_l3222_322228

theorem sqrt_sum_squares_integer (x y : ℤ) : x = 25530 ∧ y = 29464 →
  ∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_1 (x y : ℤ) : x = 37615 ∧ y = 26855 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_2 (x y : ℤ) : x = 15123 ∧ y = 32477 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

theorem sqrt_sum_squares_not_integer_3 (x y : ℤ) : x = 28326 ∧ y = 28614 →
  ¬∃ n : ℕ, n > 0 ∧ n^2 = x^2 + y^2 :=
by sorry

end sqrt_sum_squares_integer_sqrt_sum_squares_not_integer_1_sqrt_sum_squares_not_integer_2_sqrt_sum_squares_not_integer_3_l3222_322228


namespace not_valid_prism_diagonals_5_6_9_not_valid_prism_diagonals_7_8_11_l3222_322201

/-- A function to check if three positive real numbers can represent the lengths of external diagonals of a right regular prism -/
def is_valid_prism_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  b^2 + c^2 > a^2 ∧
  c^2 + a^2 > b^2

/-- Theorem stating that {5,6,9} cannot be the lengths of external diagonals of a right regular prism -/
theorem not_valid_prism_diagonals_5_6_9 :
  ¬ is_valid_prism_diagonals 5 6 9 :=
sorry

/-- Theorem stating that {7,8,11} cannot be the lengths of external diagonals of a right regular prism -/
theorem not_valid_prism_diagonals_7_8_11 :
  ¬ is_valid_prism_diagonals 7 8 11 :=
sorry

end not_valid_prism_diagonals_5_6_9_not_valid_prism_diagonals_7_8_11_l3222_322201


namespace pete_susan_speed_ratio_l3222_322272

/-- Given the walking and cartwheel speeds of Pete, Susan, and Tracy, prove that the ratio of Pete's backward walking speed to Susan's forward walking speed is 3. -/
theorem pete_susan_speed_ratio :
  ∀ (pete_backward pete_hands tracy_cartwheel susan_forward : ℝ),
  pete_hands > 0 →
  pete_backward > 0 →
  tracy_cartwheel > 0 →
  susan_forward > 0 →
  tracy_cartwheel = 2 * susan_forward →
  pete_hands = (1 / 4) * tracy_cartwheel →
  pete_hands = 2 →
  pete_backward = 12 →
  pete_backward / susan_forward = 3 := by
sorry

end pete_susan_speed_ratio_l3222_322272


namespace pauls_money_duration_l3222_322221

/-- 
Given Paul's earnings from mowing lawns and weed eating, and his weekly spending rate,
prove that the money will last for 2 weeks.
-/
theorem pauls_money_duration (lawn_earnings weed_earnings weekly_spending : ℕ) 
  (h1 : lawn_earnings = 3)
  (h2 : weed_earnings = 3)
  (h3 : weekly_spending = 3) :
  (lawn_earnings + weed_earnings) / weekly_spending = 2 := by
  sorry

end pauls_money_duration_l3222_322221


namespace distance_between_complex_points_l3222_322246

/-- The distance between two complex numbers 2+3i and -2+2i is √17 -/
theorem distance_between_complex_points : 
  Complex.abs ((2 : ℂ) + 3*I - ((-2 : ℂ) + 2*I)) = Real.sqrt 17 := by
  sorry

end distance_between_complex_points_l3222_322246


namespace evaluate_g_l3222_322245

/-- The function g(x) = 3x^2 - 6x + 5 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- Theorem: 3g(2) + 2g(-4) = 169 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-4) = 169 := by
  sorry

end evaluate_g_l3222_322245


namespace h_j_composition_l3222_322237

theorem h_j_composition (c d : ℝ) (h : ℝ → ℝ) (j : ℝ → ℝ)
  (h_def : ∀ x, h x = c * x + d)
  (j_def : ∀ x, j x = 3 * x - 4)
  (composition : ∀ x, j (h x) = 4 * x + 3) :
  c + d = 11 / 3 := by
sorry

end h_j_composition_l3222_322237


namespace weight_change_result_l3222_322229

/-- Calculates the final weight after a series of weight changes -/
def finalWeight (initialWeight : ℕ) (initialLoss : ℕ) : ℕ :=
  let weightAfterFirstLoss := initialWeight - initialLoss
  let weightAfterSecondGain := weightAfterFirstLoss + 2 * initialLoss
  let weightAfterThirdLoss := weightAfterSecondGain - 3 * initialLoss
  let finalWeightGain := 3  -- half of a dozen
  weightAfterThirdLoss + finalWeightGain

/-- Theorem stating that the final weight is 78 pounds -/
theorem weight_change_result : finalWeight 99 12 = 78 := by
  sorry

end weight_change_result_l3222_322229


namespace exercise_minutes_proof_l3222_322218

def javier_minutes : ℕ := 50
def javier_days : ℕ := 10

def sanda_minutes_1 : ℕ := 90
def sanda_days_1 : ℕ := 3
def sanda_minutes_2 : ℕ := 75
def sanda_days_2 : ℕ := 2
def sanda_minutes_3 : ℕ := 45
def sanda_days_3 : ℕ := 4

def total_exercise_minutes : ℕ := 1100

theorem exercise_minutes_proof :
  (javier_minutes * javier_days) +
  (sanda_minutes_1 * sanda_days_1) +
  (sanda_minutes_2 * sanda_days_2) +
  (sanda_minutes_3 * sanda_days_3) = total_exercise_minutes :=
by sorry

end exercise_minutes_proof_l3222_322218


namespace additional_investment_rate_barbata_investment_problem_l3222_322214

/-- Calculates the interest rate of an additional investment given initial investment parameters and desired total return rate. -/
theorem additional_investment_rate 
  (initial_investment : ℝ) 
  (initial_rate : ℝ) 
  (additional_investment : ℝ) 
  (total_rate : ℝ) : ℝ :=
  let total_investment := initial_investment + additional_investment
  let initial_income := initial_investment * initial_rate
  let total_desired_income := total_investment * total_rate
  let additional_income_needed := total_desired_income - initial_income
  additional_income_needed / additional_investment

/-- Proves that the additional investment rate is 0.08 (8%) given the specific problem parameters. -/
theorem barbata_investment_problem : 
  additional_investment_rate 1400 0.05 700 0.06 = 0.08 := by
  sorry

end additional_investment_rate_barbata_investment_problem_l3222_322214


namespace min_value_sum_reciprocal_squares_l3222_322226

/-- Two circles in the xy-plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0

/-- The property that the two circles have exactly three common tangents -/
def has_three_common_tangents (c : TwoCircles) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2 * c.a * x + c.a^2 - 4 = 0 ∧
                x^2 + y^2 - 4 * c.b * y - 1 + 4 * c.b^2 = 0

/-- The theorem stating the minimum value of 1/a^2 + 1/b^2 -/
theorem min_value_sum_reciprocal_squares (c : TwoCircles) 
  (h : has_three_common_tangents c) : 
  (∀ ε > 0, ∃ (c' : TwoCircles), has_three_common_tangents c' ∧ 
    1 / c'.a^2 + 1 / c'.b^2 < 1 + ε) ∧
  (∀ (c' : TwoCircles), has_three_common_tangents c' → 
    1 / c'.a^2 + 1 / c'.b^2 ≥ 1) :=
sorry

end min_value_sum_reciprocal_squares_l3222_322226


namespace factor_statements_l3222_322222

theorem factor_statements : 
  (∃ n : ℤ, 30 = 5 * n) ∧ (∃ m : ℤ, 180 = 9 * m) := by sorry

end factor_statements_l3222_322222


namespace chess_playoff_orders_l3222_322258

/-- Represents the structure of a chess playoff tournament --/
structure ChessPlayoff where
  numPlayers : Nat
  numMatches : Nat
  firstMatchPlayers : Fin 3 × Fin 3
  secondMatchPlayer : Fin 3

/-- Calculates the number of possible prize orders in a chess playoff tournament --/
def numPossibleOrders (tournament : ChessPlayoff) : Nat :=
  2^tournament.numMatches

/-- Theorem stating that the number of possible prize orders in the given tournament structure is 4 --/
theorem chess_playoff_orders (tournament : ChessPlayoff) 
  (h1 : tournament.numPlayers = 3)
  (h2 : tournament.numMatches = 2)
  (h3 : tournament.firstMatchPlayers = (⟨2, by norm_num⟩, ⟨1, by norm_num⟩))
  (h4 : tournament.secondMatchPlayer = ⟨0, by norm_num⟩) :
  numPossibleOrders tournament = 4 := by
  sorry


end chess_playoff_orders_l3222_322258


namespace females_dont_listen_l3222_322286

/-- A structure representing the survey results -/
structure SurveyResults where
  total_listen : Nat
  males_listen : Nat
  total_dont_listen : Nat
  total_respondents : Nat
  males_listen_le_total_listen : males_listen ≤ total_listen
  total_respondents_eq : total_respondents = total_listen + total_dont_listen

/-- The theorem stating the number of females who don't listen to the radio station -/
theorem females_dont_listen (survey : SurveyResults)
  (h_total_listen : survey.total_listen = 200)
  (h_males_listen : survey.males_listen = 75)
  (h_total_dont_listen : survey.total_dont_listen = 180)
  (h_total_respondents : survey.total_respondents = 380) :
  survey.total_dont_listen - (survey.total_respondents - survey.total_listen) = 180 := by
  sorry


end females_dont_listen_l3222_322286


namespace sequence_increasing_l3222_322299

theorem sequence_increasing (n : ℕ+) : 
  let a : ℕ+ → ℚ := fun k => k / (k + 2)
  a n < a (n + 1) := by
  sorry

end sequence_increasing_l3222_322299


namespace max_basketballs_l3222_322215

-- Define the cost of soccer balls and basketballs
def cost_3_soccer_2_basket : ℕ := 490
def cost_2_soccer_4_basket : ℕ := 660
def total_balls : ℕ := 62
def max_total_cost : ℕ := 6750

-- Define the function to calculate the total cost
def total_cost (soccer_balls : ℕ) (basketballs : ℕ) : ℕ :=
  let soccer_cost := (cost_3_soccer_2_basket * 2 - cost_2_soccer_4_basket * 3) / 2
  let basket_cost := (cost_2_soccer_4_basket * 3 - cost_3_soccer_2_basket * 2) / 2
  soccer_cost * soccer_balls + basket_cost * basketballs

-- Theorem to prove
theorem max_basketballs :
  ∃ (m : ℕ), m = 39 ∧
  (∀ (n : ℕ), n > m → total_cost (total_balls - n) n > max_total_cost) ∧
  total_cost (total_balls - m) m ≤ max_total_cost :=
sorry

end max_basketballs_l3222_322215


namespace young_inequality_l3222_322244

theorem young_inequality (A B p q : ℝ) (hA : A > 0) (hB : B > 0) (hp : p > 0) (hq : q > 0) (hpq : 1/p + 1/q = 1) :
  A^(1/p) * B^(1/q) ≤ A/p + B/q :=
by sorry

end young_inequality_l3222_322244


namespace british_flag_theorem_expected_value_zero_l3222_322232

/-- A rectangle in a 2D plane -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

/-- Theorem: For any rectangle ABCD and any point P, AP^2 + CP^2 - BP^2 - DP^2 = 0 -/
theorem british_flag_theorem (rect : Rectangle) (P : ℝ × ℝ) :
  distanceSquared rect.A P + distanceSquared rect.C P
  = distanceSquared rect.B P + distanceSquared rect.D P := by
  sorry

/-- Corollary: The expected value of AP^2 + CP^2 - BP^2 - DP^2 is always 0 -/
theorem expected_value_zero (rect : Rectangle) :
  ∃ E : ℝ, E = 0 ∧ ∀ P : ℝ × ℝ,
    E = distanceSquared rect.A P + distanceSquared rect.C P
      - distanceSquared rect.B P - distanceSquared rect.D P := by
  sorry

end british_flag_theorem_expected_value_zero_l3222_322232


namespace instantaneous_velocity_at_2s_l3222_322273

-- Define the distance function
def S (t : ℝ) : ℝ := 2 * (1 - t)^2

-- Define the instantaneous velocity function (derivative of S)
def v (t : ℝ) : ℝ := -4 * (1 - t)

-- Theorem statement
theorem instantaneous_velocity_at_2s :
  v 2 = 4 := by sorry

end instantaneous_velocity_at_2s_l3222_322273


namespace manager_average_salary_l3222_322269

/-- Represents the salary distribution in Plutarch Enterprises -/
structure SalaryDistribution where
  total_employees : ℝ
  marketer_ratio : ℝ
  engineer_ratio : ℝ
  marketer_salary : ℝ
  engineer_salary : ℝ
  average_salary : ℝ

/-- Theorem stating the average salary of managers in Plutarch Enterprises -/
theorem manager_average_salary (sd : SalaryDistribution) 
  (h1 : sd.marketer_ratio = 0.7)
  (h2 : sd.engineer_ratio = 0.1)
  (h3 : sd.marketer_salary = 50000)
  (h4 : sd.engineer_salary = 80000)
  (h5 : sd.average_salary = 80000) :
  (sd.average_salary * sd.total_employees - 
   (sd.marketer_ratio * sd.marketer_salary * sd.total_employees + 
    sd.engineer_ratio * sd.engineer_salary * sd.total_employees)) / 
   ((1 - sd.marketer_ratio - sd.engineer_ratio) * sd.total_employees) = 185000 := by
  sorry

end manager_average_salary_l3222_322269


namespace bus_stop_time_l3222_322281

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages. -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 50) 
  (h2 : speed_with_stops = 43) : ℝ :=
by
  -- The proof goes here
  sorry

#check bus_stop_time

end bus_stop_time_l3222_322281


namespace ram_price_calculation_ram_price_theorem_l3222_322200

theorem ram_price_calculation (initial_price : ℝ) 
  (increase_percentage : ℝ) (decrease_percentage : ℝ) : ℝ :=
  let increased_price := initial_price * (1 + increase_percentage)
  let final_price := increased_price * (1 - decrease_percentage)
  final_price

theorem ram_price_theorem : 
  ram_price_calculation 50 0.3 0.2 = 52 := by
  sorry

end ram_price_calculation_ram_price_theorem_l3222_322200


namespace mias_gift_spending_l3222_322238

theorem mias_gift_spending (total_spending : ℕ) (num_siblings : ℕ) (parent_gift : ℕ) (num_parents : ℕ) 
  (h1 : total_spending = 150)
  (h2 : num_siblings = 3)
  (h3 : parent_gift = 30)
  (h4 : num_parents = 2) :
  (total_spending - num_parents * parent_gift) / num_siblings = 30 := by
  sorry

end mias_gift_spending_l3222_322238


namespace julies_earnings_l3222_322290

/-- Calculates Julie's earnings for landscaping services --/
def calculate_earnings (
  lawn_rate : ℚ)
  (weed_rate : ℚ)
  (prune_rate : ℚ)
  (mulch_rate : ℚ)
  (lawn_hours_sept : ℚ)
  (weed_hours_sept : ℚ)
  (prune_hours_sept : ℚ)
  (mulch_hours_sept : ℚ) : ℚ :=
  let sept_earnings := 
    lawn_rate * lawn_hours_sept +
    weed_rate * weed_hours_sept +
    prune_rate * prune_hours_sept +
    mulch_rate * mulch_hours_sept
  let oct_earnings := 
    lawn_rate * (lawn_hours_sept * 1.5) +
    weed_rate * (weed_hours_sept * 1.5) +
    prune_rate * (prune_hours_sept * 1.5) +
    mulch_rate * (mulch_hours_sept * 1.5)
  sept_earnings + oct_earnings

/-- Theorem: Julie's total earnings for September and October --/
theorem julies_earnings : 
  calculate_earnings 4 8 10 12 25 3 10 5 = 710 := by
  sorry

end julies_earnings_l3222_322290


namespace smaller_solution_quadratic_equation_l3222_322292

theorem smaller_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => x^2 - 13*x + 36
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) ∧
  x₁ = 4 :=
by sorry

end smaller_solution_quadratic_equation_l3222_322292


namespace triangle_square_ratio_l3222_322259

/-- A triangle in a 2D plane --/
structure Triangle :=
  (a b c : ℝ)
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (hineq : a + b > c ∧ b + c > a ∧ c + a > b)

/-- The side length of the largest square inscribed in a triangle --/
noncomputable def maxInscribedSquareSide (t : Triangle) : ℝ := sorry

/-- The side length of the smallest square circumscribed around a triangle --/
noncomputable def minCircumscribedSquareSide (t : Triangle) : ℝ := sorry

/-- A triangle is right-angled if one of its angles is 90 degrees --/
def isRightTriangle (t : Triangle) : Prop := sorry

theorem triangle_square_ratio (t : Triangle) :
  minCircumscribedSquareSide t / maxInscribedSquareSide t ≥ 2 ∧
  (minCircumscribedSquareSide t / maxInscribedSquareSide t = 2 ↔ isRightTriangle t) :=
sorry

end triangle_square_ratio_l3222_322259


namespace inequality_proof_l3222_322285

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 * (y*z + z*x + x*y)^2 ≤ 3*(y^2 + y*z + z^2)*(z^2 + z*x + x^2)*(x^2 + x*y + y^2) := by
  sorry

end inequality_proof_l3222_322285
