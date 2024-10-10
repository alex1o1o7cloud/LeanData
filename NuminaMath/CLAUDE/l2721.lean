import Mathlib

namespace midpoint_coordinate_sum_l2721_272144

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (-3, 1/2) and (7, 9) is equal to 6.75 -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := -3
  let y₁ : ℝ := 1/2
  let x₂ : ℝ := 7
  let y₂ : ℝ := 9
  let mx : ℝ := (x₁ + x₂) / 2
  let my : ℝ := (y₁ + y₂) / 2
  mx + my = 6.75 := by sorry

end midpoint_coordinate_sum_l2721_272144


namespace max_parts_correct_max_parts_2004_l2721_272148

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating that max_parts gives the correct maximum number of parts -/
theorem max_parts_correct (n : ℕ) : 
  max_parts n = 1 + n * (n + 1) / 2 := by sorry

/-- The specific case for 2004 lines -/
theorem max_parts_2004 : max_parts 2004 = 2009011 := by sorry

end max_parts_correct_max_parts_2004_l2721_272148


namespace sufficient_not_necessary_l2721_272111

theorem sufficient_not_necessary (a b : ℝ) :
  (a > b ∧ b > 0) → (1 / a^2 < 1 / b^2) ∧
  ∃ (x y : ℝ), (1 / x^2 < 1 / y^2) ∧ ¬(x > y ∧ y > 0) :=
by sorry

end sufficient_not_necessary_l2721_272111


namespace solution_characterization_l2721_272162

def system_equations (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

theorem solution_characterization :
  ∀ y : ℝ,
  (∀ x₁ x₂ x₃ x₄ x₅ : ℝ, system_equations x₁ x₂ x₃ x₄ x₅ y →
    ((y ≠ 2 ∧ y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
     (y = 2 → ∃ u : ℝ, x₁ = u ∧ x₂ = u ∧ x₃ = u ∧ x₄ = u ∧ x₅ = u) ∧
     (y^2 + y - 1 = 0 →
       ∃ u v : ℝ, x₁ = u ∧ x₂ = v ∧ x₃ = -u + y*v ∧ x₄ = -y*(u + v) ∧ x₅ = y*u - v))) :=
by sorry


end solution_characterization_l2721_272162


namespace thirteen_factorial_divisible_by_eleven_l2721_272133

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: 13! is divisible by 11 -/
theorem thirteen_factorial_divisible_by_eleven :
  13 % 11 = 0 := by
  sorry

end thirteen_factorial_divisible_by_eleven_l2721_272133


namespace equation_solution_l2721_272183

theorem equation_solution :
  ∃ x : ℚ, (0.05 * x + 0.12 * (30 + x) = 15.6) ∧ (x = 1200 / 17) := by
  sorry

end equation_solution_l2721_272183


namespace x_squared_less_than_x_l2721_272102

theorem x_squared_less_than_x (x : ℝ) : x^2 < x ↔ 0 < x ∧ x < 1 := by
  sorry

end x_squared_less_than_x_l2721_272102


namespace solution_set_equals_interval_l2721_272176

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := {x : ℝ | 2 * x^2 + 8 * x ≤ -6}

-- State the theorem
theorem solution_set_equals_interval : 
  solution_set = Set.Icc (-3) (-1) := by sorry

end solution_set_equals_interval_l2721_272176


namespace no_positive_integer_solution_l2721_272156

theorem no_positive_integer_solution :
  ¬∃ (x y : ℕ+), x^2017 - 1 = (x - 1) * (y^2015 - 1) := by
sorry

end no_positive_integer_solution_l2721_272156


namespace polygon_with_150_degree_angles_polygon_with_14_diagonals_l2721_272155

-- Define a polygon
structure Polygon where
  sides : ℕ
  interiorAngle : ℝ
  diagonals : ℕ

-- Theorem 1: A polygon with interior angles of 150° has 12 sides
theorem polygon_with_150_degree_angles (p : Polygon) : 
  p.interiorAngle = 150 → p.sides = 12 := by sorry

-- Theorem 2: A polygon with 14 diagonals has interior angles that sum to 900°
theorem polygon_with_14_diagonals (p : Polygon) :
  p.diagonals = 14 → (p.sides - 2) * 180 = 900 := by sorry

end polygon_with_150_degree_angles_polygon_with_14_diagonals_l2721_272155


namespace no_real_solutions_l2721_272122

theorem no_real_solutions (n : ℝ) : 
  (∀ x : ℝ, (x + 6) * (x - 3) ≠ n + 4 * x) ↔ n < -73/4 :=
by sorry

end no_real_solutions_l2721_272122


namespace special_equation_solution_l2721_272127

theorem special_equation_solution :
  ∃ x : ℝ, 9 - 8 / 7 * x + 10 = 13.285714285714286 ∧ x = 5 := by
  sorry

end special_equation_solution_l2721_272127


namespace tournament_max_k_l2721_272185

def num_teams : ℕ := 20

-- Ice Hockey scoring system
def ice_hockey_max_k (n : ℕ) : ℕ := n - 2

-- Volleyball scoring system
def volleyball_max_k (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 5 else n - 4

theorem tournament_max_k :
  ice_hockey_max_k num_teams = 18 ∧
  volleyball_max_k num_teams = 15 := by
  sorry

#eval ice_hockey_max_k num_teams
#eval volleyball_max_k num_teams

end tournament_max_k_l2721_272185


namespace perfect_square_trinomial_k_l2721_272109

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + b)^2 -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, p x = (a * x + b)^2

/-- The given polynomial -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 8*x + k

theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial (f k) → k = 16 := by
  sorry

end perfect_square_trinomial_k_l2721_272109


namespace remainder_proof_l2721_272123

theorem remainder_proof (x y : ℤ) 
  (hx : x % 52 = 19) 
  (hy : (3 * y) % 7 = 5) : 
  ((x + 2*y)^2) % 7 = 1 := by
  sorry

end remainder_proof_l2721_272123


namespace inverse_true_implies_negation_true_l2721_272108

theorem inverse_true_implies_negation_true (P : Prop) :
  (¬P → ¬P) → (¬P) :=
by sorry

end inverse_true_implies_negation_true_l2721_272108


namespace complex_subtraction_simplification_l2721_272169

theorem complex_subtraction_simplification :
  (5 - 3 * Complex.I) - (-2 + 7 * Complex.I) = 7 - 10 * Complex.I :=
by sorry

end complex_subtraction_simplification_l2721_272169


namespace smallest_c_value_l2721_272101

theorem smallest_c_value (a b c : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression condition
  (c * c = a * b) →  -- geometric progression condition
  c ≥ 2 :=
by sorry

end smallest_c_value_l2721_272101


namespace sum_of_squares_bound_l2721_272130

theorem sum_of_squares_bound (x y z t : ℝ) 
  (h1 : |x + y + z - t| ≤ 1)
  (h2 : |y + z + t - x| ≤ 1)
  (h3 : |z + t + x - y| ≤ 1)
  (h4 : |t + x + y - z| ≤ 1) :
  x^2 + y^2 + z^2 + t^2 ≤ 1 := by
sorry

end sum_of_squares_bound_l2721_272130


namespace dad_steps_l2721_272124

theorem dad_steps (dad_masha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_ratio : ℕ → ℕ → Prop)
                  (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end dad_steps_l2721_272124


namespace triangle_properties_l2721_272161

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  b * Real.sin A = 2 * Real.sqrt 3 * a * (Real.cos (B / 2))^2 - Real.sqrt 3 * a ∧
  b = 4 * Real.sqrt 3 ∧
  Real.sin A * Real.cos B + Real.cos A * Real.sin B = 2 * Real.sin A →
  B = π / 3 ∧ 
  (1/2) * a * c * Real.sin B = 8 * Real.sqrt 3 :=
by sorry

end triangle_properties_l2721_272161


namespace expected_trait_count_is_forty_l2721_272118

/-- The probability of an individual having the genetic trait -/
def trait_probability : ℚ := 1 / 8

/-- The total number of people in the sample -/
def sample_size : ℕ := 320

/-- The expected number of people with the genetic trait in the sample -/
def expected_trait_count : ℚ := trait_probability * sample_size

theorem expected_trait_count_is_forty : expected_trait_count = 40 := by
  sorry

end expected_trait_count_is_forty_l2721_272118


namespace carol_invitation_packs_l2721_272143

theorem carol_invitation_packs (invitations_per_pack : ℕ) (total_invitations : ℕ) (h1 : invitations_per_pack = 9) (h2 : total_invitations = 45) :
  total_invitations / invitations_per_pack = 5 :=
by sorry

end carol_invitation_packs_l2721_272143


namespace geometric_sequence_solution_l2721_272116

theorem geometric_sequence_solution (x : ℝ) : 
  (1 : ℝ) < x ∧ x < 9 ∧ x^2 = 9 → x = 3 ∨ x = -3 := by
  sorry

end geometric_sequence_solution_l2721_272116


namespace point_p_final_position_l2721_272117

def final_position (initial : ℤ) (right_move : ℤ) (left_move : ℤ) : ℤ :=
  initial + right_move - left_move

theorem point_p_final_position :
  final_position (-2) 5 4 = -1 := by
  sorry

end point_p_final_position_l2721_272117


namespace president_secretary_selection_l2721_272187

/-- The number of ways to select 2 people from n people and assign them to 2 distinct roles -/
def permutation_two_roles (n : ℕ) : ℕ := n * (n - 1)

/-- There are 6 people to choose from -/
def number_of_people : ℕ := 6

theorem president_secretary_selection :
  permutation_two_roles number_of_people = 30 := by
  sorry

end president_secretary_selection_l2721_272187


namespace last_three_digits_of_7_to_99_l2721_272145

theorem last_three_digits_of_7_to_99 : 7^99 ≡ 573 [ZMOD 1000] := by
  sorry

end last_three_digits_of_7_to_99_l2721_272145


namespace first_valid_row_count_l2721_272129

def is_valid_arrangement (total_trees : ℕ) (num_rows : ℕ) : Prop :=
  num_rows > 0 ∧ total_trees % num_rows = 0

theorem first_valid_row_count : 
  let total_trees := 84
  ∀ (n : ℕ), n > 0 → is_valid_arrangement total_trees n →
    (is_valid_arrangement total_trees 6 ∧
     is_valid_arrangement total_trees 4) →
    2 ≤ n :=
by sorry

end first_valid_row_count_l2721_272129


namespace cathys_initial_amount_l2721_272167

/-- The amount of money Cathy had before her parents sent her money -/
def initial_amount : ℕ := sorry

/-- The amount of money Cathy's dad sent her -/
def dad_amount : ℕ := 25

/-- The amount of money Cathy's mom sent her -/
def mom_amount : ℕ := 2 * dad_amount

/-- The total amount Cathy has now -/
def total_amount : ℕ := 87

theorem cathys_initial_amount : 
  initial_amount = total_amount - (dad_amount + mom_amount) ∧ 
  initial_amount = 12 := by sorry

end cathys_initial_amount_l2721_272167


namespace birds_taken_out_l2721_272115

theorem birds_taken_out (initial_birds remaining_birds : ℕ) 
  (h1 : initial_birds = 19)
  (h2 : remaining_birds = 9) :
  initial_birds - remaining_birds = 10 := by
  sorry

end birds_taken_out_l2721_272115


namespace specific_quadrilateral_area_l2721_272128

/-- A point in the 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a quadrilateral given its four vertices. -/
def quadrilateralArea (p q r s : Point) : ℝ := sorry

/-- Theorem: The area of the quadrilateral with vertices at (7,6), (-5,1), (-2,-3), and (10,2) is 63 square units. -/
theorem specific_quadrilateral_area :
  let p : Point := ⟨7, 6⟩
  let q : Point := ⟨-5, 1⟩
  let r : Point := ⟨-2, -3⟩
  let s : Point := ⟨10, 2⟩
  quadrilateralArea p q r s = 63 := by sorry

end specific_quadrilateral_area_l2721_272128


namespace max_correct_answers_l2721_272157

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 5 →
  incorrect_score = -2 →
  total_score = 60 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 14 ∧
    ∀ c, c > 14 →
      ¬∃ (i u : ℕ), c + i + u = total_questions ∧
                    correct_score * c + incorrect_score * i = total_score :=
by sorry

end max_correct_answers_l2721_272157


namespace cracker_sales_percentage_increase_l2721_272138

theorem cracker_sales_percentage_increase
  (total_boxes : ℕ)
  (saturday_boxes : ℕ)
  (h1 : total_boxes = 150)
  (h2 : saturday_boxes = 60) :
  let sunday_boxes := total_boxes - saturday_boxes
  ((sunday_boxes - saturday_boxes) : ℚ) / saturday_boxes * 100 = 50 := by
  sorry

end cracker_sales_percentage_increase_l2721_272138


namespace unique_solution_l2721_272110

/-- A quadratic polynomial with exactly one root -/
structure UniqueRootQuadratic where
  a : ℝ
  b : ℝ
  has_unique_root : ∃! x : ℝ, x^2 + a * x + b = 0

/-- The composite polynomial with exactly one root -/
def composite_poly (g : UniqueRootQuadratic) (x : ℝ) : ℝ :=
  g.a * (x^5 + 2*x - 1) + g.b + g.a * (x^5 + 3*x + 1) + g.b

/-- Theorem stating the unique solution for a and b -/
theorem unique_solution (g : UniqueRootQuadratic) 
  (h : ∃! x : ℝ, composite_poly g x = 0) : 
  g.a = 74 ∧ g.b = 1369 := by
  sorry

end unique_solution_l2721_272110


namespace sqrt_equation_l2721_272142

theorem sqrt_equation (x y : ℝ) (h : Real.sqrt (x - 2) + (y - 3)^2 = 0) : 
  Real.sqrt (2*x + 3*y + 3) = 4 := by
  sorry

end sqrt_equation_l2721_272142


namespace starting_number_proof_l2721_272152

theorem starting_number_proof : ∃ x : ℝ, ((x - 2 + 4) / 1) / 2 * 8 = 77 ∧ x = 17.25 := by
  sorry

end starting_number_proof_l2721_272152


namespace train_length_l2721_272171

/-- Given a train that crosses a post in 15 seconds and a platform 100 m long in 25 seconds, its length is 150 meters. -/
theorem train_length (post_crossing_time platform_crossing_time platform_length : ℝ)
  (h1 : post_crossing_time = 15)
  (h2 : platform_crossing_time = 25)
  (h3 : platform_length = 100) :
  ∃ (train_length train_speed : ℝ),
    train_length = train_speed * post_crossing_time ∧
    train_length + platform_length = train_speed * platform_crossing_time ∧
    train_length = 150 :=
by
  sorry

end train_length_l2721_272171


namespace tan_double_angle_special_case_l2721_272140

theorem tan_double_angle_special_case (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.cos α + Real.sin α = -1/5) : Real.tan (2 * α) = -24/7 := by
  sorry

end tan_double_angle_special_case_l2721_272140


namespace bus_stop_walk_time_l2721_272160

/-- The time taken to walk to the bus stop at the usual speed, in minutes -/
def usual_time : ℝ := 30

/-- The time taken to walk to the bus stop at 4/5 of the usual speed, in minutes -/
def slower_time : ℝ := usual_time + 6

theorem bus_stop_walk_time : usual_time = 30 := by
  sorry

end bus_stop_walk_time_l2721_272160


namespace chemistry_mixture_volume_l2721_272153

theorem chemistry_mixture_volume (V : ℝ) :
  (0.6 * V + 100) / (V + 100) = 0.7 →
  V = 300 :=
by sorry

end chemistry_mixture_volume_l2721_272153


namespace endpoint_sum_endpoint_sum_proof_l2721_272151

/-- Given a line segment with one endpoint (6, 1) and midpoint (5, 7),
    the sum of the coordinates of the other endpoint is 17. -/
theorem endpoint_sum : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun x1 y1 mx my x2 y2 =>
    x1 = 6 ∧ y1 = 1 ∧ mx = 5 ∧ my = 7 ∧
    (x1 + x2) / 2 = mx ∧ (y1 + y2) / 2 = my →
    x2 + y2 = 17

theorem endpoint_sum_proof : endpoint_sum 6 1 5 7 4 13 := by
  sorry

end endpoint_sum_endpoint_sum_proof_l2721_272151


namespace holly_initial_amount_l2721_272195

/-- The amount of chocolate milk Holly drinks at breakfast, in ounces. -/
def breakfast_consumption : ℕ := 8

/-- The amount of chocolate milk Holly drinks at lunch, in ounces. -/
def lunch_consumption : ℕ := 8

/-- The amount of chocolate milk Holly drinks at dinner, in ounces. -/
def dinner_consumption : ℕ := 8

/-- The amount of chocolate milk Holly ends the day with, in ounces. -/
def end_of_day_amount : ℕ := 56

/-- The size of the new container Holly buys during lunch, in ounces. -/
def new_container_size : ℕ := 64

/-- Theorem stating that Holly began the day with 80 ounces of chocolate milk. -/
theorem holly_initial_amount :
  breakfast_consumption + lunch_consumption + dinner_consumption + end_of_day_amount = 80 :=
by sorry

end holly_initial_amount_l2721_272195


namespace uniform_prices_theorem_l2721_272174

/-- Represents a servant's employment terms and compensation --/
structure Servant where
  annual_salary : ℕ  -- Annual salary in Rupees
  service_months : ℕ  -- Months of service completed
  partial_payment : ℕ  -- Partial payment received in Rupees

/-- Calculates the price of a uniform given a servant's terms and compensation --/
def uniform_price (s : Servant) : ℕ :=
  (s.service_months * s.annual_salary - 12 * s.partial_payment) / (12 - s.service_months)

theorem uniform_prices_theorem (servant_a servant_b servant_c : Servant) 
  (h_a : servant_a = { annual_salary := 500, service_months := 9, partial_payment := 250 })
  (h_b : servant_b = { annual_salary := 800, service_months := 6, partial_payment := 300 })
  (h_c : servant_c = { annual_salary := 1200, service_months := 4, partial_payment := 200 }) :
  uniform_price servant_a = 500 ∧ 
  uniform_price servant_b = 200 ∧ 
  uniform_price servant_c = 300 := by
  sorry

#eval uniform_price { annual_salary := 500, service_months := 9, partial_payment := 250 }
#eval uniform_price { annual_salary := 800, service_months := 6, partial_payment := 300 }
#eval uniform_price { annual_salary := 1200, service_months := 4, partial_payment := 200 }

end uniform_prices_theorem_l2721_272174


namespace probability_one_of_each_l2721_272132

/-- The number of t-shirts in the wardrobe -/
def num_tshirts : ℕ := 3

/-- The number of pairs of jeans in the wardrobe -/
def num_jeans : ℕ := 7

/-- The number of hats in the wardrobe -/
def num_hats : ℕ := 4

/-- The total number of clothing items in the wardrobe -/
def total_items : ℕ := num_tshirts + num_jeans + num_hats

/-- The probability of selecting one t-shirt, one pair of jeans, and one hat -/
theorem probability_one_of_each : 
  (num_tshirts * num_jeans * num_hats : ℚ) / (total_items.choose 3) = 21 / 91 := by
  sorry

end probability_one_of_each_l2721_272132


namespace square_roots_problem_l2721_272173

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (2*x - 3)^2 = a) (h3 : (5 - x)^2 = a) : a = 49 := by
  sorry

end square_roots_problem_l2721_272173


namespace complement_of_union_l2721_272164

def U : Finset Nat := {1,2,3,4,5,6,7,8}
def M : Finset Nat := {1,3,5,7}
def N : Finset Nat := {5,6,7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end complement_of_union_l2721_272164


namespace find_z_value_l2721_272193

theorem find_z_value (z : ℝ) : (12^3 * z^3) / 432 = 864 → z = 6 := by
  sorry

end find_z_value_l2721_272193


namespace circle_op_inequality_solution_set_l2721_272170

-- Define the ⊙ operation
def circle_op (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem circle_op_inequality_solution_set :
  ∀ x : ℝ, circle_op x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
sorry

end circle_op_inequality_solution_set_l2721_272170


namespace parabola_shift_l2721_272180

/-- A parabola shifted 1 unit to the left -/
def shifted_parabola (x : ℝ) : ℝ := (x + 1)^2

/-- The original parabola -/
def original_parabola (x : ℝ) : ℝ := x^2

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  sorry

end parabola_shift_l2721_272180


namespace smores_cost_example_l2721_272113

/-- The cost of supplies for S'mores given the number of people, S'mores per person, and cost per set of S'mores. -/
def smoresCost (numPeople : ℕ) (smoresPerPerson : ℕ) (costPerSet : ℚ) (smoresPerSet : ℕ) : ℚ :=
  (numPeople * smoresPerPerson : ℚ) / smoresPerSet * costPerSet

/-- Theorem: The cost of S'mores supplies for 8 people eating 3 S'mores each, where 4 S'mores cost $3, is $18. -/
theorem smores_cost_example : smoresCost 8 3 3 4 = 18 := by
  sorry

#eval smoresCost 8 3 3 4

end smores_cost_example_l2721_272113


namespace evaluate_logarithmic_expression_l2721_272182

theorem evaluate_logarithmic_expression :
  Real.sqrt (Real.log 8 / Real.log 3 - Real.log 8 / Real.log 2 + Real.log 8 / Real.log 4) =
  Real.sqrt (3 * (2 * Real.log 2 - Real.log 3)) / Real.sqrt (2 * Real.log 3) := by
  sorry

end evaluate_logarithmic_expression_l2721_272182


namespace prize_order_count_is_32_l2721_272150

/-- Represents a bowling tournament with 6 players and a specific playoff system. -/
structure BowlingTournament where
  players : Fin 6
  /-- The number of matches in the tournament -/
  num_matches : Nat
  /-- Each match has two possible outcomes -/
  match_outcomes : Nat → Bool

/-- The number of different possible prize orders in the tournament -/
def prizeOrderCount (t : BowlingTournament) : Nat :=
  2^t.num_matches

/-- Theorem stating that the number of different prize orders is 32 -/
theorem prize_order_count_is_32 (t : BowlingTournament) :
  prizeOrderCount t = 32 :=
by sorry

end prize_order_count_is_32_l2721_272150


namespace seventh_term_largest_coefficient_l2721_272194

def binomial_expansion (x : ℝ) (n : ℕ) : ℕ → ℝ
  | r => (-1)^r * (Nat.choose n r) * x^(2*n - 3*r)

theorem seventh_term_largest_coefficient :
  ∃ (x : ℝ), ∀ (r : ℕ), r ≠ 6 →
    |binomial_expansion x 11 6| ≥ |binomial_expansion x 11 r| :=
sorry

end seventh_term_largest_coefficient_l2721_272194


namespace student_survey_l2721_272112

theorem student_survey (french_and_english : ℕ) (french_not_english : ℕ) 
  (percent_not_french : ℚ) :
  french_and_english = 25 →
  french_not_english = 65 →
  percent_not_french = 55/100 →
  french_and_english + french_not_english = (100 : ℚ) / (100 - percent_not_french) * 100 :=
by sorry

end student_survey_l2721_272112


namespace trigonometric_equation_l2721_272131

theorem trigonometric_equation (x : ℝ) (h : |Real.cos (2 * x)| ≠ 1) :
  8.451 * ((1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))) = (1/3) * Real.tan x ^ 4 ↔
  ∃ k : ℤ, x = π/3 * (3 * k + 1) ∨ x = π/3 * (3 * k - 1) :=
sorry

end trigonometric_equation_l2721_272131


namespace present_age_of_B_l2721_272146

/-- Given two natural numbers A and B representing ages, proves that B is 41 years old
    given the conditions:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is now 11 years older than B. -/
theorem present_age_of_B (A B : ℕ) 
    (h1 : A + 10 = 2 * (B - 10))
    (h2 : A = B + 11) : 
  B = 41 := by
  sorry


end present_age_of_B_l2721_272146


namespace ratio_345_iff_arithmetic_sequence_l2721_272172

/-- Represents a right-angled triangle with side lengths a, b, c where a < b < c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_lt_b : a < b
  b_lt_c : b < c
  right_angle : a^2 + b^2 = c^2

/-- The ratio of sides is 3:4:5 -/
def has_ratio_345 (t : RightTriangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k

/-- The sides form an arithmetic sequence -/
def is_arithmetic_sequence (t : RightTriangle) : Prop :=
  ∃ (d : ℝ), d > 0 ∧ t.b = t.a + d ∧ t.c = t.b + d

/-- The main theorem stating the equivalence of the two conditions -/
theorem ratio_345_iff_arithmetic_sequence (t : RightTriangle) :
  has_ratio_345 t ↔ is_arithmetic_sequence t :=
sorry

end ratio_345_iff_arithmetic_sequence_l2721_272172


namespace point_coordinates_l2721_272158

theorem point_coordinates (x y : ℝ) : 
  (x < 0 ∧ y > 0) →  -- Point P is in the second quadrant
  (|x| = 2) →        -- |x| = 2
  (y^2 = 1) →        -- y is the square root of 1
  (x = -2 ∧ y = 1)   -- Coordinates of P are (-2, 1)
  := by sorry

end point_coordinates_l2721_272158


namespace max_ant_path_theorem_l2721_272168

/-- Represents a cube with edge length 12 cm -/
structure Cube where
  edge_length : ℝ
  edge_length_eq : edge_length = 12

/-- Represents a path on the cube's edges -/
structure CubePath where
  length : ℝ
  no_repeat : Bool

/-- The maximum distance an ant can walk on the cube's edges without repetition -/
def max_ant_path (c : Cube) : ℝ := 108

/-- Theorem stating the maximum distance an ant can walk on the cube -/
theorem max_ant_path_theorem (c : Cube) :
  ∀ (path : CubePath), path.no_repeat → path.length ≤ max_ant_path c :=
by sorry

end max_ant_path_theorem_l2721_272168


namespace ratios_neither_necessary_nor_sufficient_l2721_272188

-- Define the coefficients for the two quadratic inequalities
variable (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ)

-- Define the solution sets for the two inequalities
def SolutionSet1 (x : ℝ) := a₁ * x^2 + b₁ * x + c₁ > 0
def SolutionSet2 (x : ℝ) := a₂ * x^2 + b₂ * x + c₂ > 0

-- Define the equality of ratios condition
def RatiosEqual := (a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)

-- Define the property of having the same solution set
def SameSolutionSet := ∀ x, SolutionSet1 a₁ b₁ c₁ x ↔ SolutionSet2 a₂ b₂ c₂ x

-- Theorem stating that the equality of ratios is neither necessary nor sufficient
theorem ratios_neither_necessary_nor_sufficient :
  ¬(RatiosEqual a₁ b₁ c₁ a₂ b₂ c₂ → SameSolutionSet a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(SameSolutionSet a₁ b₁ c₁ a₂ b₂ c₂ → RatiosEqual a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end ratios_neither_necessary_nor_sufficient_l2721_272188


namespace hot_dogs_remainder_l2721_272186

theorem hot_dogs_remainder : 35252983 % 6 = 1 := by
  sorry

end hot_dogs_remainder_l2721_272186


namespace least_three_digit_multiple_l2721_272163

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (3 ∣ n) ∧ (4 ∣ n) ∧ (7 ∣ n) ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((3 ∣ m) ∧ (4 ∣ m) ∧ (7 ∣ m))) ∧
  n = 168 := by
  sorry

end least_three_digit_multiple_l2721_272163


namespace original_number_proof_l2721_272121

theorem original_number_proof (n : ℕ) : 
  n - 7 = 62575 ∧ (62575 % 99 = 92) → n = 62582 := by
sorry

end original_number_proof_l2721_272121


namespace distinct_exponentiation_values_l2721_272103

-- Define a function to represent different parenthesizations of 3^3^3^3
def exponentiation_order (n : Nat) : Nat :=
  match n with
  | 0 => 3^(3^(3^3))  -- standard order
  | 1 => 3^((3^3)^3)
  | 2 => (3^3)^(3^3)
  | 3 => (3^(3^3))^3
  | _ => ((3^3)^3)^3

-- Theorem statement
theorem distinct_exponentiation_values :
  ∃ (S : Finset Nat), (Finset.card S = 5) ∧ 
  (∀ (i : Nat), i < 5 → exponentiation_order i ∈ S) ∧
  (∀ (x : Nat), x ∈ S → ∃ (i : Nat), i < 5 ∧ exponentiation_order i = x) :=
sorry

end distinct_exponentiation_values_l2721_272103


namespace percentage_problem_l2721_272135

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 700 → 
  0.3 * N = (P / 100) * 150 + 120 → 
  P = 60 := by
sorry

end percentage_problem_l2721_272135


namespace academy_skills_l2721_272119

theorem academy_skills (total : ℕ) (dancers : ℕ) (calligraphers : ℕ) (both : ℕ) : 
  total = 120 → 
  dancers = 88 → 
  calligraphers = 32 → 
  both = 18 → 
  total - (dancers + calligraphers - both) = 18 := by
sorry

end academy_skills_l2721_272119


namespace geometric_sequence_sum_l2721_272154

/-- Given a geometric sequence where the third term is 81 and the sixth term is 1,
    prove that the sum of the fourth and fifth terms is 36. -/
theorem geometric_sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  (∀ n, a (n + 1) = a n * r) →  -- Each term is r times the previous term
  a 3 = 81 →                    -- The third term is 81
  a 6 = 1 →                     -- The sixth term is 1
  a 4 + a 5 = 36 := by
  sorry

end geometric_sequence_sum_l2721_272154


namespace equation_solution_l2721_272175

theorem equation_solution :
  ∃ (a b p q : ℝ),
    (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
    ((a = (2^20 - 1)^(1/20) ∧ b = -(2^20 - 1)^(1/20)/2) ∨
     (a = -(2^20 - 1)^(1/20) ∧ b = (2^20 - 1)^(1/20)/2)) ∧
    p = -1 ∧ q = 1/4 := by
  sorry

end equation_solution_l2721_272175


namespace fraction_product_proof_l2721_272189

theorem fraction_product_proof : 
  (7 : ℚ) / 4 * 8 / 14 * 14 / 8 * 16 / 40 * 35 / 20 * 18 / 45 * 49 / 28 * 32 / 64 = 49 / 200 := by
  sorry

end fraction_product_proof_l2721_272189


namespace contrapositive_equivalence_l2721_272166

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔ (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end contrapositive_equivalence_l2721_272166


namespace equation_solution_l2721_272165

theorem equation_solution :
  ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end equation_solution_l2721_272165


namespace short_video_length_proof_l2721_272192

/-- Represents the length of short videos in minutes -/
def short_video_length : ℝ := 2

/-- Represents the number of videos released per day -/
def videos_per_day : ℕ := 3

/-- Represents the length multiplier for the long video -/
def long_video_multiplier : ℕ := 6

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the total video length per week in minutes -/
def total_weekly_length : ℝ := 112

theorem short_video_length_proof :
  short_video_length * (videos_per_day - 1 + long_video_multiplier) * days_per_week = total_weekly_length :=
by sorry

end short_video_length_proof_l2721_272192


namespace triangle_concurrency_l2721_272181

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b c : ℝ)

-- Define the triangle
def Triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define perpendicular
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define reflection
def Reflection (P Q : Point) (l : Line) : Prop := sorry

-- Define intersection
def Intersect (l1 l2 : Line) : Point := sorry

-- Define concurrency
def Concurrent (l1 l2 l3 : Line) : Prop := sorry

-- Theorem statement
theorem triangle_concurrency 
  (A B C D E F H E' F' X Y : Point) 
  (ABC : Triangle A B C)
  (not_right : ¬ Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0)) -- Assuming right angle is between x and y axes
  (D_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (E_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (F_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (H_ortho : H = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0))
  (E'_refl : Reflection E E' (Line.mk 1 0 0))
  (F'_refl : Reflection F F' (Line.mk 1 0 0))
  (X_def : X = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0))
  (Y_def : Y = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0)) :
  Concurrent (Line.mk 1 0 0) (Line.mk 0 1 0) (Line.mk 1 1 0) := by
  sorry

end triangle_concurrency_l2721_272181


namespace expected_stones_approx_l2721_272197

/-- The width of the river (scaled to 1) -/
def river_width : ℝ := 1

/-- The maximum jump distance (scaled to 0.01) -/
def jump_distance : ℝ := 0.01

/-- The probability that we cannot cross the river after n throws -/
noncomputable def P (n : ℕ) : ℝ :=
  ∑' i, (-1)^(i-1) * (n+1).choose i * (max (1 - i * jump_distance) 0)^n

/-- The expected number of stones needed to cross the river -/
noncomputable def expected_stones : ℝ :=
  ∑' n, P n

/-- Theorem stating the approximation of the expected number of stones -/
theorem expected_stones_approx :
  ∃ ε > 0, |expected_stones - 712.811| < ε :=
sorry

end expected_stones_approx_l2721_272197


namespace temperature_drop_l2721_272125

/-- Given an initial temperature and a temperature drop, calculate the final temperature. -/
def final_temperature (initial : ℤ) (drop : ℕ) : ℤ :=
  initial - drop

/-- Theorem: When the initial temperature is 3℃ and it drops by 5℃, the final temperature is -2℃. -/
theorem temperature_drop : final_temperature 3 5 = -2 := by
  sorry

end temperature_drop_l2721_272125


namespace min_hindi_speakers_l2721_272179

theorem min_hindi_speakers (total : ℕ) (english : ℕ) (both : ℕ) (hindi : ℕ) : 
  total = 40 → 
  english = 20 → 
  both ≥ 10 → 
  hindi = total + both - english →
  hindi ≥ 30 :=
by sorry

end min_hindi_speakers_l2721_272179


namespace max_children_in_candy_game_l2721_272107

/-- Represents the candy distribution game. -/
structure CandyGame where
  n : ℕ  -- number of children
  k : ℕ  -- number of complete circles each child passes candies
  a : ℕ  -- number of candies each child has when the game is interrupted

/-- Checks if the game satisfies the conditions. -/
def is_valid_game (game : CandyGame) : Prop :=
  ∃ (i j : ℕ), i < game.n ∧ j < game.n ∧ i ≠ j ∧
  (game.a + 2 * game.n * game.k - 2 * i) / (game.a + 2 * game.n * game.k - 2 * j) = 13

/-- The theorem stating the maximum number of children in the game. -/
theorem max_children_in_candy_game :
  ∃ (game : CandyGame), is_valid_game game ∧
    (∀ (other_game : CandyGame), is_valid_game other_game → other_game.n ≤ game.n) ∧
    game.n = 25 := by
  sorry

end max_children_in_candy_game_l2721_272107


namespace consecutive_sum_property_l2721_272190

theorem consecutive_sum_property : ∃ (a : Fin 10 → ℝ),
  (∀ i : Fin 6, (a i) + (a (i+1)) + (a (i+2)) + (a (i+3)) + (a (i+4)) > 0) ∧
  (∀ j : Fin 4, (a j) + (a (j+1)) + (a (j+2)) + (a (j+3)) + (a (j+4)) + (a (j+5)) + (a (j+6)) < 0) :=
by sorry

end consecutive_sum_property_l2721_272190


namespace one_and_half_of_number_l2721_272134

theorem one_and_half_of_number (x : ℚ) : (3 / 2) * x = 30 → x = 20 := by
  sorry

end one_and_half_of_number_l2721_272134


namespace complex_argument_one_minus_i_sqrt_three_l2721_272114

/-- The argument of the complex number 1 - i√3 is 5π/3 -/
theorem complex_argument_one_minus_i_sqrt_three (z : ℂ) : 
  z = 1 - Complex.I * Real.sqrt 3 → Complex.arg z = 5 * Real.pi / 3 := by
  sorry

end complex_argument_one_minus_i_sqrt_three_l2721_272114


namespace not_A_implies_not_all_mc_or_not_three_math_l2721_272178

-- Define the predicates
def got_all_mc_right (student : String) : Prop := sorry
def solved_at_least_three_math (student : String) : Prop := sorry
def received_A (student : String) : Prop := sorry

-- Ms. Carroll's rule
axiom ms_carroll_rule (student : String) :
  got_all_mc_right student ∧ solved_at_least_three_math student → received_A student

-- Theorem to prove
theorem not_A_implies_not_all_mc_or_not_three_math (student : String) :
  ¬(received_A student) → ¬(got_all_mc_right student) ∨ ¬(solved_at_least_three_math student) :=
by sorry

end not_A_implies_not_all_mc_or_not_three_math_l2721_272178


namespace whitewash_fence_l2721_272136

theorem whitewash_fence (k : ℕ) : 
  ∀ (x y : Fin (2^(k+1))), 
    (∃ (z : Fin (2^(k+1))), z ≠ x ∧ z ≠ y ∧ 
      (2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (y.val^2 + 3*y.val - 2)) ↔ 
       2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (z.val^2 + 3*z.val - 2)))) ∧
    (∀ (w : Fin (2^(k+1))), 
      2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (w.val^2 + 3*w.val - 2)) → 
      w = x ∨ w = y) :=
by sorry

#check whitewash_fence

end whitewash_fence_l2721_272136


namespace petes_walking_distance_l2721_272177

/-- Represents a pedometer with a maximum count --/
structure Pedometer where
  max_count : ℕ
  reset_count : ℕ
  final_reading : ℕ

/-- Calculates the total steps based on pedometer data --/
def total_steps (p : Pedometer) : ℕ :=
  p.reset_count * (p.max_count + 1) + p.final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem petes_walking_distance (p : Pedometer) (steps_per_mile : ℕ) :
  p.max_count = 99999 ∧
  p.reset_count = 38 ∧
  p.final_reading = 75000 ∧
  steps_per_mile = 1800 →
  steps_to_miles (total_steps p) steps_per_mile = 2150 := by
  sorry

#eval steps_to_miles (total_steps { max_count := 99999, reset_count := 38, final_reading := 75000 }) 1800

end petes_walking_distance_l2721_272177


namespace fruit_bowl_cherries_l2721_272198

theorem fruit_bowl_cherries :
  ∀ (b s r c : ℕ),
    b + s + r + c = 360 →
    s = 2 * b →
    r = 4 * s →
    c = 2 * r →
    c = 640 / 3 :=
by
  sorry

end fruit_bowl_cherries_l2721_272198


namespace permutation_combination_equality_l2721_272184

theorem permutation_combination_equality (n : ℕ) : (n.factorial / (n - 3).factorial = 6 * n.choose 4) → n = 7 := by
  sorry

end permutation_combination_equality_l2721_272184


namespace bird_nest_area_scientific_notation_l2721_272126

/-- The construction area of the National Stadium "Bird's Nest" in square meters -/
def bird_nest_area : ℝ := 258000

/-- The scientific notation representation of the bird_nest_area -/
def bird_nest_scientific : ℝ := 2.58 * (10 ^ 5)

theorem bird_nest_area_scientific_notation :
  bird_nest_area = bird_nest_scientific := by
  sorry

end bird_nest_area_scientific_notation_l2721_272126


namespace triangle_third_altitude_l2721_272106

theorem triangle_third_altitude (h₁ h₂ h₃ : ℝ) :
  h₁ = 8 → h₂ = 12 → h₃ > 0 →
  (1 / h₁ + 1 / h₂ > 1 / h₃) →
  h₃ > 4.8 := by
sorry

end triangle_third_altitude_l2721_272106


namespace regular_octagon_area_equals_diagonal_product_l2721_272137

/-- A regular octagon -/
structure RegularOctagon where
  -- We don't need to specify all properties of a regular octagon,
  -- just the existence of such a shape
  dummy : Unit

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- The length of the longest diagonal of a regular octagon -/
def longest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The length of the shortest diagonal of a regular octagon -/
def shortest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The area of a regular octagon is equal to the product of 
    the lengths of its longest and shortest diagonals -/
theorem regular_octagon_area_equals_diagonal_product (o : RegularOctagon) :
  area o = longest_diagonal o * shortest_diagonal o :=
sorry

end regular_octagon_area_equals_diagonal_product_l2721_272137


namespace marcy_water_amount_l2721_272191

/-- The amount of water Marcy keeps by her desk -/
def water_amount (sip_interval : ℕ) (sip_volume : ℕ) (total_time : ℕ) : ℚ :=
  (total_time / sip_interval * sip_volume : ℚ) / 1000

/-- Theorem stating that Marcy keeps 2 liters of water by her desk -/
theorem marcy_water_amount :
  water_amount 5 40 250 = 2 := by
  sorry

#eval water_amount 5 40 250

end marcy_water_amount_l2721_272191


namespace hyperbola_equation_l2721_272104

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m = b / a ∧ m = Real.sqrt 3) →
  (∃ (d : ℝ), d = 2 * Real.sqrt 3 ∧ d = b) →
  (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) :=
by sorry

end hyperbola_equation_l2721_272104


namespace rectangle_split_divisibility_l2721_272139

/-- The number of ways to split a 3 × n rectangle into 1 × 2 rectangles -/
def N (n : ℕ) : ℕ :=
  sorry

/-- The number of ways to split a 3 × n rectangle into 1 × 2 rectangles,
    where the last row has exactly two cells filled -/
def γ (n : ℕ) : ℕ :=
  sorry

theorem rectangle_split_divisibility (n : ℕ) (h : n = 200) :
  3 ∣ N n := by
  sorry

end rectangle_split_divisibility_l2721_272139


namespace polar_coordinates_of_point_l2721_272159

theorem polar_coordinates_of_point (x y : ℝ) (h : (x, y) = (-1, Real.sqrt 3)) :
  ∃ (ρ θ : ℝ), ρ = 2 ∧ θ = 2 * Real.pi / 3 ∧ 
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end polar_coordinates_of_point_l2721_272159


namespace min_value_theorem_l2721_272141

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - 2 * x - y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → a * b - 2 * a - b = 0 → x + y / 2 ≤ a + b / 2 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y - 2 * x - y = 0 ∧ x + y / 2 = 4 :=
by sorry

end min_value_theorem_l2721_272141


namespace solution_system_equations_l2721_272147

theorem solution_system_equations :
  let x : ℝ := -1
  let y : ℝ := 2
  ((x^2 + y) * Real.sqrt (y - 2*x) - 4 = 2*x^2 + 2*x + y) ∧
  (x^3 - x^2 - y + 6 = 4 * Real.sqrt (x + 1) + 2 * Real.sqrt (y - 1)) := by
  sorry

end solution_system_equations_l2721_272147


namespace wendis_chickens_l2721_272105

theorem wendis_chickens (initial : ℕ) 
  (h1 : 2 * initial - 1 + 6 = 13) : initial = 4 := by
  sorry

end wendis_chickens_l2721_272105


namespace pirate_treasure_l2721_272120

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by sorry

end pirate_treasure_l2721_272120


namespace rectangle_length_fraction_l2721_272100

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_area = 200)
  (h3 : rectangle_breadth = 10) :
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 4 / 7 := by
  sorry

end rectangle_length_fraction_l2721_272100


namespace find_b_l2721_272149

/-- Given two functions f and g, and a condition on their composition, prove the value of b. -/
theorem find_b (f g : ℝ → ℝ) (b : ℝ) 
  (hf : ∀ x, f x = (3 * x) / 7 + 4)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h_comp : f (g b) = 10) :
  b = -4.5 := by sorry

end find_b_l2721_272149


namespace craft_fair_ring_cost_l2721_272196

/-- Given the sales data from a craft fair, prove the cost of each ring --/
theorem craft_fair_ring_cost :
  let total_sales : ℚ := 320
  let num_necklaces : ℕ := 4
  let num_rings : ℕ := 8
  let num_earrings : ℕ := 5
  let num_bracelets : ℕ := 6
  let cost_necklace : ℚ := 20
  let cost_earrings : ℚ := 15
  let cost_ring : ℚ := 8.25
  let cost_bracelet : ℚ := 2 * cost_ring
  total_sales = num_necklaces * cost_necklace + num_rings * cost_ring +
                num_earrings * cost_earrings + num_bracelets * cost_bracelet
  → cost_ring = 8.25 := by
  sorry


end craft_fair_ring_cost_l2721_272196


namespace sqrt_equality_implies_unique_pair_l2721_272199

theorem sqrt_equality_implies_unique_pair :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (49 + Real.sqrt (153 + 24 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 49 := by
sorry

end sqrt_equality_implies_unique_pair_l2721_272199
