import Mathlib

namespace exponent_division_l1278_127872

theorem exponent_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end exponent_division_l1278_127872


namespace replaced_person_weight_l1278_127881

def group_size : ℕ := 8
def average_weight_increase : ℝ := 2.5
def new_person_weight : ℝ := 90

theorem replaced_person_weight :
  let total_weight_increase : ℝ := group_size * average_weight_increase
  let replaced_weight : ℝ := new_person_weight - total_weight_increase
  replaced_weight = 70 := by sorry

end replaced_person_weight_l1278_127881


namespace xy_z_squared_plus_one_representation_l1278_127834

theorem xy_z_squared_plus_one_representation (x y z : ℕ+) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end xy_z_squared_plus_one_representation_l1278_127834


namespace instantaneous_velocity_at_2_l1278_127837

/-- The displacement function of the object -/
def h (t : ℝ) : ℝ := 14 * t - t^2

/-- The velocity function of the object -/
def v (t : ℝ) : ℝ := 14 - 2 * t

/-- Theorem: The instantaneous velocity at t = 2 seconds is 10 meters/second -/
theorem instantaneous_velocity_at_2 : v 2 = 10 := by sorry

end instantaneous_velocity_at_2_l1278_127837


namespace tan_alpha_value_l1278_127894

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end tan_alpha_value_l1278_127894


namespace investment_decrease_l1278_127899

/-- Given an initial investment that increases by 50% in the first year
    and has a net increase of 4.999999999999982% after two years,
    prove that the percentage decrease in the second year is 30%. -/
theorem investment_decrease (initial : ℝ) (initial_pos : initial > 0) :
  let first_year := initial * 1.5
  let final := initial * 1.04999999999999982
  let second_year_factor := final / first_year
  second_year_factor = 0.7 := by sorry

end investment_decrease_l1278_127899


namespace factorization_equality_l1278_127840

theorem factorization_equality (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) := by
  sorry

end factorization_equality_l1278_127840


namespace hyperbola_equation_l1278_127863

/-- The equation of a hyperbola with specific properties -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a : ℝ), x^2/a^2 + y^2/4 = 1) →  -- Ellipse equation
  (∀ (t : ℝ), y = 2*x ∨ y = -2*x) →  -- Asymptotes
  (x^2 - y^2/4 = 1) →                -- Proposed hyperbola equation
  (x = 1 ∧ y = 0) →                  -- Right vertex of the ellipse
  True                               -- The equation represents the correct hyperbola
  := by sorry

end hyperbola_equation_l1278_127863


namespace square_area_increase_when_side_tripled_l1278_127812

theorem square_area_increase_when_side_tripled :
  ∀ (s : ℝ), s > 0 →
  (3 * s)^2 = 9 * s^2 := by
  sorry

end square_area_increase_when_side_tripled_l1278_127812


namespace pen_price_calculation_l1278_127817

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 630 ∧ num_pens = 30 ∧ num_pencils = 75 ∧ pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 16 :=
by
  sorry

end pen_price_calculation_l1278_127817


namespace nested_square_root_evaluation_l1278_127815

theorem nested_square_root_evaluation (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x + Real.sqrt (x + Real.sqrt x)) = Real.sqrt (x + Real.sqrt (x + x^(1/2))) := by
  sorry

end nested_square_root_evaluation_l1278_127815


namespace cube_side_ratio_l1278_127823

/-- Given two cubes of the same material, if one cube weighs 5 pounds and the other weighs 40 pounds,
    then the ratio of the side length of the heavier cube to the side length of the lighter cube is 2:1. -/
theorem cube_side_ratio (s S : ℝ) (h1 : s > 0) (h2 : S > 0) : 
  (5 : ℝ) / s^3 = (40 : ℝ) / S^3 → S / s = 2 := by
  sorry

end cube_side_ratio_l1278_127823


namespace counterexample_exists_l1278_127854

theorem counterexample_exists : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by sorry

end counterexample_exists_l1278_127854


namespace trigonometric_inequality_l1278_127819

theorem trigonometric_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  (Real.cos (A - B))^2 + (Real.cos (B - C))^2 + (Real.cos (C - A))^2 ≥ 
  24 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end trigonometric_inequality_l1278_127819


namespace peaches_picked_up_correct_l1278_127846

/-- Represents the fruit stand inventory --/
structure FruitStand where
  initialPeaches : ℕ
  initialOranges : ℕ
  peachesSold : ℕ
  orangesAdded : ℕ
  finalPeaches : ℕ
  finalOranges : ℕ

/-- Calculates the number of peaches picked up from the orchard --/
def peachesPickedUp (stand : FruitStand) : ℕ :=
  stand.finalPeaches - (stand.initialPeaches - stand.peachesSold)

/-- Theorem stating that the number of peaches picked up is correct --/
theorem peaches_picked_up_correct (stand : FruitStand) :
  peachesPickedUp stand = stand.finalPeaches - (stand.initialPeaches - stand.peachesSold) :=
by
  sorry

/-- Sally's fruit stand inventory --/
def sallysStand : FruitStand := {
  initialPeaches := 13
  initialOranges := 5
  peachesSold := 7
  orangesAdded := 22
  finalPeaches := 55
  finalOranges := 27
}

#eval peachesPickedUp sallysStand

end peaches_picked_up_correct_l1278_127846


namespace solutions_equation1_solutions_equation2_l1278_127821

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 + 4*x - 4 = 0
def equation2 (x : ℝ) : Prop := (x - 1)^2 = 2*(x - 1)

-- Theorem for the first equation
theorem solutions_equation1 :
  ∃ (x1 x2 : ℝ), 
    equation1 x1 ∧ equation1 x2 ∧ 
    x1 = -2 + 2 * Real.sqrt 2 ∧ 
    x2 = -2 - 2 * Real.sqrt 2 :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 :
  ∃ (x1 x2 : ℝ), 
    equation2 x1 ∧ equation2 x2 ∧ 
    x1 = 1 ∧ x2 = 3 :=
sorry

end solutions_equation1_solutions_equation2_l1278_127821


namespace bicycle_cost_price_l1278_127876

/-- The cost price of a bicycle given a series of sales with specified profit margins -/
theorem bicycle_cost_price
  (profit_A_to_B : Real)
  (profit_B_to_C : Real)
  (profit_C_to_D : Real)
  (final_price : Real)
  (h1 : profit_A_to_B = 0.50)
  (h2 : profit_B_to_C = 0.25)
  (h3 : profit_C_to_D = 0.15)
  (h4 : final_price = 320.75) :
  ∃ (cost_price : Real),
    cost_price = final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C) * (1 + profit_C_to_D)) :=
by
  sorry

end bicycle_cost_price_l1278_127876


namespace binomial_26_3_minus_10_l1278_127865

theorem binomial_26_3_minus_10 : Nat.choose 26 3 - 10 = 2590 := by sorry

end binomial_26_3_minus_10_l1278_127865


namespace not_always_greater_than_original_l1278_127807

theorem not_always_greater_than_original : ¬ (∀ x : ℝ, 1.25 * x > x) := by sorry

end not_always_greater_than_original_l1278_127807


namespace book_pages_count_l1278_127893

/-- Represents the number of pages Bill reads on a given day -/
def pagesReadOnDay (day : ℕ) : ℕ := 10 + 2 * (day - 1)

/-- Represents the total number of pages Bill has read up to a given day -/
def totalPagesRead (days : ℕ) : ℕ := (days * (pagesReadOnDay 1 + pagesReadOnDay days)) / 2

theorem book_pages_count :
  ∀ (total_days : ℕ) (reading_days : ℕ),
  total_days = 14 →
  reading_days = total_days - 2 →
  (totalPagesRead reading_days : ℚ) = (3/4) * (336 : ℚ) :=
by sorry

end book_pages_count_l1278_127893


namespace apartment_rent_calculation_l1278_127847

/-- Proves that the rent for a shared apartment is $1100 given specific conditions -/
theorem apartment_rent_calculation (utilities groceries roommate_payment : ℕ) 
  (h1 : utilities = 114)
  (h2 : groceries = 300)
  (h3 : roommate_payment = 757) :
  ∃ (rent : ℕ), rent = 1100 ∧ (rent + utilities + groceries) / 2 = roommate_payment :=
by sorry

end apartment_rent_calculation_l1278_127847


namespace total_handshakes_l1278_127871

def number_of_couples : ℕ := 15

-- Define the number of handshakes between men
def handshakes_between_men (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define the number of handshakes between women
def handshakes_between_women (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define the number of handshakes between men and women (excluding spouses)
def handshakes_men_women (n : ℕ) : ℕ := n * (n - 1)

theorem total_handshakes :
  handshakes_between_men number_of_couples +
  handshakes_between_women number_of_couples +
  handshakes_men_women number_of_couples = 420 := by
  sorry

end total_handshakes_l1278_127871


namespace intersection_of_A_and_B_l1278_127889

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {y | ∃ x, y = 2^x - 1}

theorem intersection_of_A_and_B : A ∩ B = {m | -1 < m ∧ m < 2} := by sorry

end intersection_of_A_and_B_l1278_127889


namespace not_p_or_not_q_is_true_l1278_127867

theorem not_p_or_not_q_is_true :
  ∀ (a b c : ℝ),
  let p := ∀ (a b c : ℝ), a > b → a + c > b + c
  let q := ∀ (a b c : ℝ), a > b ∧ b > 0 → a * c > b * c
  ¬p ∨ ¬q := by sorry

end not_p_or_not_q_is_true_l1278_127867


namespace students_not_visiting_any_exhibit_l1278_127833

def total_students : ℕ := 52
def botanical_visitors : ℕ := 12
def animal_visitors : ℕ := 26
def technology_visitors : ℕ := 23
def botanical_and_animal : ℕ := 5
def botanical_and_technology : ℕ := 2
def animal_and_technology : ℕ := 4
def all_three : ℕ := 1

theorem students_not_visiting_any_exhibit : 
  total_students - (botanical_visitors + animal_visitors + technology_visitors
                    - botanical_and_animal - botanical_and_technology - animal_and_technology
                    + all_three) = 1 := by sorry

end students_not_visiting_any_exhibit_l1278_127833


namespace F_properties_l1278_127848

-- Define the function f
def f (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) : ℝ → ℝ := sorry

-- Define the function F
def F (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) (x : ℝ) : ℝ :=
  (f a b h1 h2 x)^2 - (f a b h1 h2 (-x))^2

-- State the theorem
theorem F_properties (a b : ℝ) (h1 : 0 < b) (h2 : b < -a) :
  (∀ x, F a b h1 h2 x ≠ 0) →
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f a b h1 h2 x < f a b h1 h2 y) →
  (∀ x, F a b h1 h2 x = 0 ∨ -b ≤ x ∧ x ≤ b) ∧
  (∀ x, F a b h1 h2 (-x) = -(F a b h1 h2 x)) :=
by sorry

end F_properties_l1278_127848


namespace james_tennis_balls_l1278_127808

theorem james_tennis_balls (total_containers : Nat) (balls_per_container : Nat) : 
  total_containers = 5 → 
  balls_per_container = 10 → 
  2 * (total_containers * balls_per_container) = 100 := by
sorry

end james_tennis_balls_l1278_127808


namespace move_right_four_units_l1278_127835

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally in a Cartesian coordinate system -/
def moveRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem move_right_four_units :
  let p := Point.mk (-2) 3
  moveRight p 4 = Point.mk 2 3 := by
  sorry

end move_right_four_units_l1278_127835


namespace meeting_point_theorem_l1278_127804

/-- The distance between Maxwell's and Brad's homes in kilometers -/
def total_distance : ℝ := 40

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 5

/-- The distance traveled by Maxwell when they meet -/
def maxwell_distance : ℝ := 15

theorem meeting_point_theorem :
  maxwell_distance = total_distance * maxwell_speed / (maxwell_speed + brad_speed) :=
by sorry

end meeting_point_theorem_l1278_127804


namespace unique_solution_g100_l1278_127884

-- Define g₀(x)
def g₀ (x : ℝ) : ℝ := 2 * x + |x - 50| - |x + 50|

-- Define gₙ(x) recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem unique_solution_g100 :
  ∃! x, g 100 x = 0 :=
sorry

end unique_solution_g100_l1278_127884


namespace function_inequality_l1278_127842

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end function_inequality_l1278_127842


namespace cannot_obtain_123_l1278_127820

/-- Represents an arithmetic expression using numbers 1, 2, 3, 4, 5 and operations +, -, * -/
inductive Expr
| Num : Fin 5 → Expr
| Add : Expr → Expr → Expr
| Sub : Expr → Expr → Expr
| Mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → Int
| Expr.Num n => n.val.succ
| Expr.Add e1 e2 => eval e1 + eval e2
| Expr.Sub e1 e2 => eval e1 - eval e2
| Expr.Mul e1 e2 => eval e1 * eval e2

/-- Theorem stating that it's impossible to obtain 123 using the given constraints -/
theorem cannot_obtain_123 : ¬ ∃ e : Expr, eval e = 123 := by
  sorry

end cannot_obtain_123_l1278_127820


namespace unique_function_solution_l1278_127809

theorem unique_function_solution (f : ℕ → ℕ) :
  (∀ a b : ℕ, f (f a + f b) = a + b) ↔ (∀ n : ℕ, f n = n) := by
  sorry

end unique_function_solution_l1278_127809


namespace sara_green_marbles_l1278_127829

def total_green_marbles : ℕ := 7
def tom_green_marbles : ℕ := 4
def sara_red_marbles : ℕ := 5

theorem sara_green_marbles :
  ∃ (x : ℕ), x = total_green_marbles - tom_green_marbles :=
sorry

end sara_green_marbles_l1278_127829


namespace point_not_in_A_when_a_negative_l1278_127896

-- Define the set A
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 ≥ 0 ∧ a * p.1 + p.2 ≥ 2 ∧ p.1 - a * p.2 ≤ 2}

-- Theorem statement
theorem point_not_in_A_when_a_negative :
  ∀ a : ℝ, a < 0 → (1, 1) ∉ A a :=
by sorry

end point_not_in_A_when_a_negative_l1278_127896


namespace similar_triangles_leg_sum_l1278_127849

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 24 →
  (1/2) * c * d = 600 →
  a^2 + b^2 = 100 →
  (c / a)^2 = 25 →
  (d / b)^2 = 25 →
  c + d = 70 := by
sorry

end similar_triangles_leg_sum_l1278_127849


namespace infinite_series_sum_l1278_127806

/-- The sum of the infinite series ∑_{k=1}^∞ (k^3 / 3^k) is equal to 6 -/
theorem infinite_series_sum : 
  (∑' k : ℕ+, (k : ℝ)^3 / 3^(k : ℝ)) = 6 := by sorry

end infinite_series_sum_l1278_127806


namespace complement_of_M_l1278_127864

-- Define the set M
def M : Set ℝ := {x | x^2 - 2*x > 0}

-- State the theorem
theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = Set.Icc 0 2 := by sorry

end complement_of_M_l1278_127864


namespace bike_owners_without_scooters_l1278_127845

theorem bike_owners_without_scooters (total_population : ℕ) 
  (bike_owners : ℕ) (scooter_owners : ℕ) 
  (h1 : total_population = 420)
  (h2 : bike_owners = 380)
  (h3 : scooter_owners = 82)
  (h4 : ∀ p, p ∈ Set.range (Fin.val : Fin total_population → ℕ) → 
    (p ∈ Set.range (Fin.val : Fin bike_owners → ℕ) ∨ 
     p ∈ Set.range (Fin.val : Fin scooter_owners → ℕ))) :
  bike_owners - (bike_owners + scooter_owners - total_population) = 338 :=
sorry

end bike_owners_without_scooters_l1278_127845


namespace unique_solution_tan_equation_l1278_127862

theorem unique_solution_tan_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan (150 * π / 180 - x * π / 180) =
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end unique_solution_tan_equation_l1278_127862


namespace tetrahedron_volume_l1278_127830

/-- Theorem: Volume of a tetrahedron
  Given a tetrahedron with:
  - a, b: lengths of opposite edges
  - α: angle between these edges
  - c: distance between the lines containing these edges
  The volume V of the tetrahedron is given by V = (1/6) * a * b * c * sin(α)
-/
theorem tetrahedron_volume 
  (a b c : ℝ) 
  (α : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hα : 0 < α ∧ α < π) :
  ∃ V : ℝ, V = (1/6) * a * b * c * Real.sin α := by
  sorry


end tetrahedron_volume_l1278_127830


namespace ac_over_bd_equals_15_l1278_127858

theorem ac_over_bd_equals_15 
  (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 2 * c) 
  (h3 : c = 5 * d) 
  (h4 : d ≠ 0) : 
  (a * c) / (b * d) = 15 := by
sorry

end ac_over_bd_equals_15_l1278_127858


namespace min_sum_given_reciprocal_sum_l1278_127814

theorem min_sum_given_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 4/b = 2) : a + b ≥ 9/2 :=
by sorry

end min_sum_given_reciprocal_sum_l1278_127814


namespace smallest_number_proof_l1278_127883

/-- A function that checks if a natural number contains all digits from 0 to 9 exactly once -/
def has_all_digits_once (n : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number that is a multiple of 36 and contains all digits from 0 to 9 exactly once -/
def smallest_number_with_all_digits_divisible_by_36 : ℕ := sorry

theorem smallest_number_proof :
  smallest_number_with_all_digits_divisible_by_36 = 1023457896 ∧
  has_all_digits_once smallest_number_with_all_digits_divisible_by_36 ∧
  smallest_number_with_all_digits_divisible_by_36 % 36 = 0 ∧
  ∀ m : ℕ, m < smallest_number_with_all_digits_divisible_by_36 →
    ¬(has_all_digits_once m ∧ m % 36 = 0) :=
by sorry

end smallest_number_proof_l1278_127883


namespace square_sum_given_sum_and_product_l1278_127850

theorem square_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 7) (h2 : a * b = 10) : a^2 + b^2 = 29 := by
  sorry

end square_sum_given_sum_and_product_l1278_127850


namespace complex_addition_l1278_127805

theorem complex_addition (z₁ z₂ : ℂ) (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 3 + 4*I) : 
  z₁ + z₂ = 4 + 6*I := by
sorry

end complex_addition_l1278_127805


namespace intersection_of_A_and_B_l1278_127818

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 2*x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l1278_127818


namespace perpendicular_lines_from_quadratic_roots_l1278_127841

theorem perpendicular_lines_from_quadratic_roots : 
  ∀ (k₁ k₂ : ℝ), 
    k₁^2 - 3*k₁ - 1 = 0 → 
    k₂^2 - 3*k₂ - 1 = 0 → 
    k₁ ≠ k₂ →
    k₁ * k₂ = -1 :=
by sorry

end perpendicular_lines_from_quadratic_roots_l1278_127841


namespace positive_number_equality_l1278_127810

theorem positive_number_equality (x : ℝ) (h1 : x > 0) 
  (h2 : (2/3) * x = (64/216) * (1/x)) : x = 2/3 := by
  sorry

end positive_number_equality_l1278_127810


namespace butterfat_percentage_of_added_milk_l1278_127866

/-- Prove that the percentage of butterfat in the added milk is 10% -/
theorem butterfat_percentage_of_added_milk
  (initial_volume : ℝ)
  (initial_butterfat_percentage : ℝ)
  (added_volume : ℝ)
  (final_butterfat_percentage : ℝ)
  (h_initial_volume : initial_volume = 8)
  (h_initial_butterfat : initial_butterfat_percentage = 35)
  (h_added_volume : added_volume = 12)
  (h_final_butterfat : final_butterfat_percentage = 20)
  (h_total_volume : initial_volume + added_volume = 20) :
  let added_butterfat_percentage :=
    (final_butterfat_percentage * (initial_volume + added_volume) -
     initial_butterfat_percentage * initial_volume) / added_volume
  added_butterfat_percentage = 10 :=
by sorry

end butterfat_percentage_of_added_milk_l1278_127866


namespace first_player_wins_l1278_127892

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game --/
inductive Player | First | Second

/-- Represents a move in the game --/
structure Move :=
  (top_left : ℕ × ℕ)
  (size : ℕ)

/-- The game state --/
structure GameState :=
  (grid : Grid)
  (current_player : Player)
  (moves : List Move)

/-- Checks if a move is valid --/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Applies a move to the game state --/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over --/
def is_game_over (state : GameState) : Prop :=
  sorry

/-- Determines the winner of the game --/
def winner (state : GameState) : Option Player :=
  sorry

/-- Represents a strategy for playing the game --/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player --/
def is_winning_strategy (strategy : Strategy) (player : Player) : Prop :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player --/
theorem first_player_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy Player.First :=
sorry

end first_player_wins_l1278_127892


namespace sum_of_digits_theorem_l1278_127874

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: Given S(n) = 1365, S(n+1) = 1360 -/
theorem sum_of_digits_theorem (n : ℕ) (h : S n = 1365) : S (n + 1) = 1360 := by sorry

end sum_of_digits_theorem_l1278_127874


namespace cos_double_angle_special_case_l1278_127860

/-- Given that the terminal side of angle α intersects the unit circle 
    at point P(-4/5, 3/5), prove that cos 2α = 7/25 -/
theorem cos_double_angle_special_case (α : Real) 
  (h : ∃ (P : Real × Real), P.1 = -4/5 ∧ P.2 = 3/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
       P.1 = Real.cos α ∧ P.2 = Real.sin α) : 
  Real.cos (2 * α) = 7/25 := by
sorry

end cos_double_angle_special_case_l1278_127860


namespace fourth_grade_classrooms_difference_l1278_127891

theorem fourth_grade_classrooms_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 20 →
  guinea_pigs_per_class = 3 →
  num_classes = 5 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 85 :=
by
  sorry

end fourth_grade_classrooms_difference_l1278_127891


namespace distribute_and_simplify_l1278_127856

theorem distribute_and_simplify (a : ℝ) : -3 * a^2 * (4*a - 3) = -12*a^3 + 9*a^2 := by
  sorry

end distribute_and_simplify_l1278_127856


namespace solution_value_l1278_127800

theorem solution_value (x a : ℝ) : 
  2 * (x - 6) = -16 →
  a * (x + 3) = (1/2) * a + x →
  a^2 - (a/2) + 1 = 19 := by
sorry

end solution_value_l1278_127800


namespace round_trip_distance_l1278_127813

/-- Calculates the total distance of a round trip journey given speeds and times -/
theorem round_trip_distance 
  (speed_to : ℝ) 
  (speed_from : ℝ) 
  (time_to : ℝ) 
  (time_from : ℝ) 
  (h1 : speed_to = 4)
  (h2 : speed_from = 3)
  (h3 : time_to = 30 / 60)
  (h4 : time_from = 40 / 60) :
  speed_to * time_to + speed_from * time_from = 4 := by
  sorry

#check round_trip_distance

end round_trip_distance_l1278_127813


namespace x_equals_four_l1278_127803

/-- Custom operation € -/
def euro (x y : ℝ) : ℝ := 2 * x * y

/-- Theorem stating that x = 4 given the conditions -/
theorem x_equals_four :
  ∃ x : ℝ, euro 9 (euro x 5) = 720 ∧ x = 4 :=
by
  sorry

end x_equals_four_l1278_127803


namespace group_size_calculation_l1278_127827

theorem group_size_calculation (n : ℕ) : 
  (n : ℝ) * 14 + 34 = ((n : ℝ) + 1) * 16 → n = 9 := by
  sorry

end group_size_calculation_l1278_127827


namespace cyclic_inequality_l1278_127816

theorem cyclic_inequality (x y z p q : ℝ) 
  (h1 : y = x^2 + p*x + q)
  (h2 : z = y^2 + p*y + q)
  (h3 : x = z^2 + p*z + q) :
  x^2*y + y^2*z + z^2*x ≥ x^2*z + y^2*x + z^2*y := by
  sorry

end cyclic_inequality_l1278_127816


namespace recommended_apps_proof_l1278_127836

/-- The recommended number of apps for Roger's phone -/
def recommended_apps : ℕ := 35

/-- The maximum number of apps for optimal function -/
def max_optimal_apps : ℕ := 50

/-- The number of apps Roger currently has -/
def rogers_current_apps : ℕ := 2 * recommended_apps

/-- The number of apps Roger needs to delete -/
def apps_to_delete : ℕ := 20

theorem recommended_apps_proof :
  (rogers_current_apps = max_optimal_apps + apps_to_delete) ∧
  (rogers_current_apps = 2 * recommended_apps) ∧
  (max_optimal_apps = 50) ∧
  (apps_to_delete = 20) →
  recommended_apps = 35 := by sorry

end recommended_apps_proof_l1278_127836


namespace intersection_point_is_solution_l1278_127882

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (2, 3)

/-- First line equation -/
def line1 (x y : ℝ) : Prop := 10 * x - 5 * y = 5

/-- Second line equation -/
def line2 (x y : ℝ) : Prop := 8 * x + 2 * y = 22

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end intersection_point_is_solution_l1278_127882


namespace area_between_specific_lines_l1278_127822

/-- Line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area between two lines from x = 0 to x = 5 -/
def areaBetweenLines (l1 l2 : Line) : ℝ :=
  sorry

/-- The main theorem -/
theorem area_between_specific_lines :
  let line1 : Line := ⟨0, 3, 6, 0⟩
  let line2 : Line := ⟨0, 5, 10, 2⟩
  areaBetweenLines line1 line2 = 10 := by sorry

end area_between_specific_lines_l1278_127822


namespace birdwatching_sites_l1278_127844

theorem birdwatching_sites (x : ℕ) : 
  (7 * x + 5 * x + 80) / (2 * x + 10) = 7 → x + x = 10 := by
  sorry

end birdwatching_sites_l1278_127844


namespace quadratic_point_relation_l1278_127880

/-- The quadratic function f(x) = (x - 1)^2 -/
def f (x : ℝ) : ℝ := (x - 1)^2

theorem quadratic_point_relation (m : ℝ) :
  f m < f (m + 1) → m > 1/2 := by
  sorry

end quadratic_point_relation_l1278_127880


namespace smallest_n_value_l1278_127811

/-- Counts the number of factors of 5 in k! -/
def count_factors_of_5 (k : ℕ) : ℕ := sorry

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 3000 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ n' : ℕ, n' < n → ∃ m' : ℕ, a.factorial * b.factorial * c.factorial ≠ m' * (10 ^ n')) →
  n = 747 := by sorry

end smallest_n_value_l1278_127811


namespace max_students_on_playground_l1278_127885

def total_pencils : ℕ := 170
def total_notebooks : ℕ := 268
def total_erasers : ℕ := 120
def leftover_pencils : ℕ := 8
def shortage_notebooks : ℕ := 2
def leftover_erasers : ℕ := 12

theorem max_students_on_playground :
  let distributed_pencils := total_pencils - leftover_pencils
  let distributed_notebooks := total_notebooks + shortage_notebooks
  let distributed_erasers := total_erasers - leftover_erasers
  let max_students := Nat.gcd distributed_pencils (Nat.gcd distributed_notebooks distributed_erasers)
  max_students = 54 ∧
  (∃ (p n e : ℕ), 
    distributed_pencils = max_students * p ∧
    distributed_notebooks = max_students * n ∧
    distributed_erasers = max_students * e) ∧
  (∀ s : ℕ, s > max_students →
    ¬(∃ (p n e : ℕ),
      distributed_pencils = s * p ∧
      distributed_notebooks = s * n ∧
      distributed_erasers = s * e)) :=
by sorry

end max_students_on_playground_l1278_127885


namespace first_group_size_is_eight_l1278_127825

/-- The number of men in the first group that can complete a work in 18 days, working 8 hours a day -/
def first_group_size : ℕ := sorry

/-- The number of hours worked per day by both groups -/
def hours_per_day : ℕ := 8

/-- The number of days the first group takes to complete the work -/
def days_first_group : ℕ := 18

/-- The number of men in the second group -/
def second_group_size : ℕ := 12

/-- The number of days the second group takes to complete the work -/
def days_second_group : ℕ := 12

/-- The total amount of work done is constant and equal for both groups -/
axiom work_done_equal : first_group_size * hours_per_day * days_first_group = second_group_size * hours_per_day * days_second_group

theorem first_group_size_is_eight : first_group_size = 8 := by sorry

end first_group_size_is_eight_l1278_127825


namespace sin_870_degrees_l1278_127852

theorem sin_870_degrees : Real.sin (870 * π / 180) = 1 / 2 := by
  sorry

end sin_870_degrees_l1278_127852


namespace weakly_decreasing_exp_weakly_decreasing_ln_condition_weakly_decreasing_cos_condition_l1278_127873

-- Definition of weakly decreasing function
def WeaklyDecreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  (∀ x ∈ I, ∀ y ∈ I, x < y → f x ≥ f y) ∧
  (∀ x ∈ I, ∀ y ∈ I, x < y → x * f x ≤ y * f y)

-- Theorem 1
theorem weakly_decreasing_exp (x : ℝ) :
  WeaklyDecreasing (fun x => x / Real.exp x) (Set.Ioo 1 2) :=
sorry

-- Theorem 2
theorem weakly_decreasing_ln_condition (m : ℝ) :
  WeaklyDecreasing (fun x => Real.log x / x) (Set.Ioi m) → m ≥ Real.exp 1 :=
sorry

-- Theorem 3
theorem weakly_decreasing_cos_condition (k : ℝ) :
  WeaklyDecreasing (fun x => Real.cos x + k * x^2) (Set.Ioo 0 (Real.pi / 2)) →
  2 / (3 * Real.pi) ≤ k ∧ k ≤ 1 / Real.pi :=
sorry

end weakly_decreasing_exp_weakly_decreasing_ln_condition_weakly_decreasing_cos_condition_l1278_127873


namespace same_solution_implies_m_value_l1278_127832

theorem same_solution_implies_m_value :
  ∀ (m : ℝ) (x : ℝ),
    (-5 * x - 6 = 3 * x + 10) ∧
    (-2 * m - 3 * x = 10) →
    m = -2 :=
by
  sorry

end same_solution_implies_m_value_l1278_127832


namespace unique_solution_floor_equation_l1278_127897

theorem unique_solution_floor_equation :
  ∃! x : ℝ, x + ⌊x⌋ = 20.2 ∧ x = 10.2 := by
  sorry

end unique_solution_floor_equation_l1278_127897


namespace complement_of_A_in_U_l1278_127824

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}

theorem complement_of_A_in_U :
  (U \ A) = {2, 4} := by sorry

end complement_of_A_in_U_l1278_127824


namespace negative_two_minus_six_l1278_127877

theorem negative_two_minus_six : -2 - 6 = -8 := by
  sorry

end negative_two_minus_six_l1278_127877


namespace time_to_see_all_animals_after_import_l1278_127853

/-- Calculates the time required to see all animal types after importing new species -/
def time_to_see_all_animals (initial_types : ℕ) (time_per_type : ℕ) (new_species : ℕ) : ℕ :=
  (initial_types + new_species) * time_per_type

/-- Proves that the time required to see all animal types after importing new species is 54 minutes -/
theorem time_to_see_all_animals_after_import :
  time_to_see_all_animals 5 6 4 = 54 := by
  sorry

end time_to_see_all_animals_after_import_l1278_127853


namespace school_children_count_l1278_127888

theorem school_children_count (total_bananas : ℕ) : 
  (∃ (children : ℕ), 
    total_bananas = 2 * children ∧ 
    total_bananas = 4 * (children - 360)) →
  ∃ (children : ℕ), children = 720 := by
sorry

end school_children_count_l1278_127888


namespace right_trapezoid_perimeter_l1278_127855

/-- A right trapezoid with upper base a, lower base b, height h, and leg l. -/
structure RightTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  l : ℝ

/-- The perimeter of a right trapezoid. -/
def perimeter (t : RightTrapezoid) : ℝ := t.a + t.b + t.h + t.l

/-- The theorem stating the conditions and the result for the right trapezoid problem. -/
theorem right_trapezoid_perimeter (t : RightTrapezoid) :
  t.a < t.b →
  π * t.h^2 * t.a + (1/3) * π * t.h^2 * (t.b - t.a) = 80 * π →
  π * t.h^2 * t.b + (1/3) * π * t.h^2 * (t.b - t.a) = 112 * π →
  (1/3) * π * (t.a^2 + t.a * t.b + t.b^2) * t.h = 156 * π →
  perimeter t = 20 + 2 * Real.sqrt 13 := by
  sorry

end right_trapezoid_perimeter_l1278_127855


namespace mr_blue_flower_bed_yield_l1278_127826

/-- Represents the dimensions and yield of a flower bed -/
structure FlowerBed where
  length_paces : ℕ
  width_paces : ℕ
  pace_length : ℝ
  yield_per_sqft : ℝ

/-- Calculates the expected rose petal yield from a flower bed -/
def expected_yield (fb : FlowerBed) : ℝ :=
  (fb.length_paces : ℝ) * fb.pace_length *
  (fb.width_paces : ℝ) * fb.pace_length *
  fb.yield_per_sqft

/-- Theorem stating the expected yield for Mr. Blue's flower bed -/
theorem mr_blue_flower_bed_yield :
  let fb : FlowerBed := {
    length_paces := 18,
    width_paces := 24,
    pace_length := 1.5,
    yield_per_sqft := 0.4
  }
  expected_yield fb = 388.8 := by sorry

end mr_blue_flower_bed_yield_l1278_127826


namespace principal_interest_difference_l1278_127879

/-- Calculate the difference between principal and interest for a simple interest loan. -/
theorem principal_interest_difference
  (principal : ℕ)
  (rate : ℕ)
  (time : ℕ)
  (h1 : principal = 6200)
  (h2 : rate = 5)
  (h3 : time = 10) :
  principal - (principal * rate * time) / 100 = 3100 :=
by
  sorry

end principal_interest_difference_l1278_127879


namespace product_divisible_by_12_l1278_127859

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability that a single roll is not divisible by 3 -/
def prob_not_div_3 : ℚ := 5 / 8

/-- The probability that a single roll is divisible by 4 -/
def prob_div_4 : ℚ := 1 / 4

/-- The probability that the product of the rolls is divisible by 12 -/
def prob_div_12 : ℚ := 149 / 256

theorem product_divisible_by_12 :
  (1 - prob_not_div_3 ^ num_dice) *
  (1 - (1 - prob_div_4) ^ num_dice - num_dice * prob_div_4 * (1 - prob_div_4) ^ (num_dice - 1)) =
  prob_div_12 := by sorry

end product_divisible_by_12_l1278_127859


namespace largest_zero_correct_l1278_127831

/-- Sequence S defined recursively -/
def S : ℕ → ℤ
  | 0 => 0
  | (n + 1) => S n + (n + 1) * (if S n < n + 1 then 1 else -1)

/-- Predicate for S[k] = 0 -/
def is_zero (k : ℕ) : Prop := S k = 0

/-- The largest k ≤ 2010 such that S[k] = 0 -/
def largest_zero : ℕ := 1092

theorem largest_zero_correct :
  is_zero largest_zero ∧
  ∀ k, k ≤ 2010 → is_zero k → k ≤ largest_zero :=
by sorry

end largest_zero_correct_l1278_127831


namespace no_solution_functional_equation_l1278_127801

theorem no_solution_functional_equation :
  ¬∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x * y) = f x * f y + 2 * x * y :=
by sorry

end no_solution_functional_equation_l1278_127801


namespace simple_random_sampling_is_most_appropriate_l1278_127857

/-- Represents a box containing units of a product -/
structure Box where
  name : String
  units : ℕ

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Cluster

/-- Determines if a sampling method is appropriate for the given boxes and sample size -/
def is_appropriate_sampling_method (boxes : List Box) (sample_size : ℕ) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.SimpleRandom => true
  | _ => false

theorem simple_random_sampling_is_most_appropriate :
  let boxes : List Box := [
    { name := "large", units := 120 },
    { name := "medium", units := 60 },
    { name := "small", units := 20 }
  ]
  let sample_size : ℕ := 25
  ∀ method : SamplingMethod,
    is_appropriate_sampling_method boxes sample_size method →
    method = SamplingMethod.SimpleRandom :=
by
  sorry


end simple_random_sampling_is_most_appropriate_l1278_127857


namespace counterexample_exists_l1278_127843

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem counterexample_exists : ∃ n : ℕ, 
  ¬ is_prime n ∧ ¬ is_prime (n - 3) ∧ n = 15 := by
  sorry

end counterexample_exists_l1278_127843


namespace probability_two_red_one_blue_is_11_70_l1278_127870

def total_marbles : ℕ := 16
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 4

def probability_two_red_one_blue : ℚ :=
  (red_marbles * (red_marbles - 1) * blue_marbles) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem probability_two_red_one_blue_is_11_70 :
  probability_two_red_one_blue = 11 / 70 := by
  sorry

end probability_two_red_one_blue_is_11_70_l1278_127870


namespace exists_universal_shape_l1278_127839

/-- Represents a tetrimino --/
structure Tetrimino where
  cells : Finset (ℤ × ℤ)
  cell_count : cells.card = 4

/-- Represents the five types of tetriminoes --/
inductive TetriminoType
  | O
  | I
  | L
  | T
  | Z

/-- A shape is a set of cells in the plane --/
def Shape := Finset (ℤ × ℤ)

/-- Rotation of a tetrimino --/
def rotate (t : Tetrimino) : Tetrimino := sorry

/-- Check if a shape can be composed using only one type of tetrimino --/
def canComposeWithType (s : Shape) (type : TetriminoType) : Prop := sorry

/-- The main theorem --/
theorem exists_universal_shape :
  ∃ (s : Shape), ∀ (type : TetriminoType), canComposeWithType s type := by sorry

end exists_universal_shape_l1278_127839


namespace continued_proportionate_reduction_eq_euclidean_gcd_l1278_127875

/-- The Method of Continued Proportionate Reduction as used in ancient Chinese mathematics -/
def continued_proportionate_reduction (a b : ℕ) : ℕ :=
  sorry

/-- The Euclidean algorithm for finding the greatest common divisor -/
def euclidean_gcd (a b : ℕ) : ℕ :=
  sorry

/-- Theorem stating the equivalence of the two methods -/
theorem continued_proportionate_reduction_eq_euclidean_gcd :
  ∀ a b : ℕ, continued_proportionate_reduction a b = euclidean_gcd a b :=
sorry

end continued_proportionate_reduction_eq_euclidean_gcd_l1278_127875


namespace intersection_implies_m_equals_four_l1278_127838

def A : Set ℝ := {x | x ≥ 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - m*x ≤ 0}

theorem intersection_implies_m_equals_four (m : ℝ) : A ∩ B m = {x | 1 ≤ x ∧ x ≤ 4} → m = 4 := by
  sorry

end intersection_implies_m_equals_four_l1278_127838


namespace S_max_at_14_l1278_127828

/-- The sequence term for index n -/
def a (n : ℕ) : ℤ := 43 - 3 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := n * (40 + 43 - 3 * n) / 2

/-- The theorem stating that S reaches its maximum when n = 14 -/
theorem S_max_at_14 : ∀ k : ℕ, k > 0 → S 14 ≥ S k := by sorry

end S_max_at_14_l1278_127828


namespace inverse_proportion_k_value_l1278_127898

theorem inverse_proportion_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, x ≠ 0 → (k / x) = -1/2 ↔ x = 4) → k = -2 := by
  sorry

end inverse_proportion_k_value_l1278_127898


namespace arithmetic_sequence_property_l1278_127878

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (is_arithmetic_sequence a → a 1 + a 3 = 2 * a 2) ∧
  (∃ a : ℕ → ℝ, a 1 + a 3 = 2 * a 2 ∧ ¬is_arithmetic_sequence a) :=
sorry

end arithmetic_sequence_property_l1278_127878


namespace point_on_line_trig_identity_l1278_127868

theorem point_on_line_trig_identity (θ : Real) :
  2 * Real.cos θ + Real.sin θ = 0 →
  Real.cos (2 * θ) + (1/2) * Real.sin (2 * θ) = -1 := by
sorry

end point_on_line_trig_identity_l1278_127868


namespace subset_implies_a_equals_one_l1278_127886

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l1278_127886


namespace isosceles_triangle_perimeter_l1278_127869

/-- An isosceles triangle with sides of 3cm and 7cm has a perimeter of 13cm. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 3 ∧ b = 7 ∧ c = 3 →  -- Two sides are 3cm, one side is 7cm
  (a = b ∨ b = c ∨ a = c) →  -- The triangle is isosceles
  a + b + c = 13 :=  -- The perimeter is 13cm
by
  sorry

end isosceles_triangle_perimeter_l1278_127869


namespace absolute_value_inequality_l1278_127802

theorem absolute_value_inequality (x : ℝ) :
  |x + 2| + |x + 3| ≤ 2 ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

end absolute_value_inequality_l1278_127802


namespace inequality_proof_l1278_127851

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end inequality_proof_l1278_127851


namespace team_a_win_probability_l1278_127861

/-- The probability of Team A winning a single game -/
def p_win : ℚ := 3/5

/-- The probability of Team A losing a single game -/
def p_lose : ℚ := 2/5

/-- The number of ways to choose 2 wins out of 3 games -/
def combinations : ℕ := 3

theorem team_a_win_probability :
  combinations * p_win^3 * p_lose = 162/625 := by
  sorry

end team_a_win_probability_l1278_127861


namespace marsupial_protein_consumption_l1278_127887

theorem marsupial_protein_consumption (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_consumed : ℝ) : 
  absorption_rate = 0.40 →
  absorbed_amount = 16 →
  absorbed_amount = absorption_rate * total_consumed →
  total_consumed = 40 := by
  sorry

end marsupial_protein_consumption_l1278_127887


namespace joan_money_found_l1278_127895

def total_money (dimes_jacket : ℕ) (dimes_shorts : ℕ) (nickels_shorts : ℕ) 
  (quarters_jeans : ℕ) (pennies_jeans : ℕ) (nickels_backpack : ℕ) (pennies_backpack : ℕ) : ℚ :=
  (dimes_jacket + dimes_shorts) * (10 : ℚ) / 100 +
  (nickels_shorts + nickels_backpack) * (5 : ℚ) / 100 +
  quarters_jeans * (25 : ℚ) / 100 +
  (pennies_jeans + pennies_backpack) * (1 : ℚ) / 100

theorem joan_money_found :
  total_money 15 4 7 12 2 8 23 = (590 : ℚ) / 100 := by
  sorry

end joan_money_found_l1278_127895


namespace line_through_parabola_vertex_count_l1278_127890

/-- The number of values of a for which the line y = 2x + a passes through
    the vertex of the parabola y = x^2 + 2a^2 -/
theorem line_through_parabola_vertex_count : 
  ∃! (s : Finset ℝ), 
    (∀ a ∈ s, ∃ x y : ℝ, 
      y = 2 * x + a ∧ 
      y = x^2 + 2 * a^2 ∧ 
      ∀ x' : ℝ, x'^2 + 2 * a^2 ≤ x^2 + 2 * a^2) ∧ 
    s.card = 2 :=
sorry

end line_through_parabola_vertex_count_l1278_127890
