import Mathlib

namespace triangle_angle_y_l4115_411509

theorem triangle_angle_y (y : ℝ) : 
  (45 : ℝ) + 3 * y + y = 180 → y = 33.75 := by
  sorry

end triangle_angle_y_l4115_411509


namespace min_value_on_circle_l4115_411588

theorem min_value_on_circle (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ min := by
  sorry

end min_value_on_circle_l4115_411588


namespace point_transformation_l4115_411597

-- Define the transformations
def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (rotate_x_90 (reflect_xy (rotate_z_90 p)))

-- Theorem statement
theorem point_transformation :
  transform (2, 3, 4) = (3, 4, 2) := by
  sorry

end point_transformation_l4115_411597


namespace total_husk_bags_eaten_l4115_411569

-- Define the number of cows
def num_cows : ℕ := 26

-- Define the number of days
def num_days : ℕ := 26

-- Define the rate at which one cow eats husk
def cow_husk_rate : ℚ := 1 / 26

-- Theorem to prove
theorem total_husk_bags_eaten : 
  (num_cows : ℚ) * cow_husk_rate * (num_days : ℚ) = 26 := by
  sorry

end total_husk_bags_eaten_l4115_411569


namespace at_least_one_positive_l4115_411581

theorem at_least_one_positive (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (x : ℝ) (hx : x = a^2 - b*c)
  (y : ℝ) (hy : y = b^2 - c*a)
  (z : ℝ) (hz : z = c^2 - a*b) :
  x > 0 ∨ y > 0 ∨ z > 0 :=
by sorry

end at_least_one_positive_l4115_411581


namespace norris_game_spending_l4115_411557

/-- The amount of money Norris spent on the online game -/
def money_spent (september_savings october_savings november_savings money_left : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - money_left

/-- Theorem stating that Norris spent $75 on the online game -/
theorem norris_game_spending :
  money_spent 29 25 31 10 = 75 := by
  sorry

end norris_game_spending_l4115_411557


namespace orange_sum_l4115_411516

theorem orange_sum : 
  let tree1 : ℕ := 80
  let tree2 : ℕ := 60
  let tree3 : ℕ := 120
  let tree4 : ℕ := 45
  let tree5 : ℕ := 25
  let tree6 : ℕ := 97
  tree1 + tree2 + tree3 + tree4 + tree5 + tree6 = 427 := by
sorry

end orange_sum_l4115_411516


namespace logarithm_equation_solution_l4115_411508

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_equation_solution : 
  ∃ (x : ℝ), x > 0 ∧ 
  (log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + log_base (3 ^ (1/6 : ℝ)) x + 
   log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x + 
   log_base (Real.sqrt 3) x + log_base (Real.sqrt 3) x = 36) ∧
  x = Real.sqrt 3 :=
by sorry

end logarithm_equation_solution_l4115_411508


namespace parabola_translation_l4115_411599

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (1/2) 0 1
  let translated := translate original 1 (-3)
  y = 1/2 * x^2 + 1 → y = 1/2 * (x-1)^2 - 2 := by
  sorry


end parabola_translation_l4115_411599


namespace product_of_repeating_decimals_l4115_411591

/-- Represents a repeating decimal with a repeating part of length 2 -/
def RepeatingDecimal2 (a b : ℕ) : ℚ :=
  (a * 10 + b : ℚ) / 99

/-- Represents a repeating decimal with a repeating part of length 1 -/
def RepeatingDecimal1 (a : ℕ) : ℚ :=
  (a : ℚ) / 9

/-- The product of 0.overline{03} and 0.overline{3} is equal to 1/99 -/
theorem product_of_repeating_decimals :
  (RepeatingDecimal2 0 3) * (RepeatingDecimal1 3) = 1 / 99 := by
  sorry

end product_of_repeating_decimals_l4115_411591


namespace total_chips_amount_l4115_411547

def person1_chips : ℕ := 350
def person2_chips : ℕ := 268
def person3_chips : ℕ := 182

theorem total_chips_amount : person1_chips + person2_chips + person3_chips = 800 := by
  sorry

end total_chips_amount_l4115_411547


namespace tax_discount_commute_ana_equals_bob_miltonville_market_problem_l4115_411503

/-- Proves that the order of applying tax and discount doesn't affect the final price --/
theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) (discount_rate_pos : 0 < discount_rate) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) := by
  sorry

/-- Calculates Ana's total (tax then discount) --/
def ana_total (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 + tax_rate) * (1 - discount_rate)

/-- Calculates Bob's total (discount then tax) --/
def bob_total (price : ℝ) (tax_rate discount_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

/-- Proves that Ana's total equals Bob's total --/
theorem ana_equals_bob (price : ℝ) (tax_rate discount_rate : ℝ) 
  (tax_rate_pos : 0 < tax_rate) (discount_rate_pos : 0 < discount_rate) :
  ana_total price tax_rate discount_rate = bob_total price tax_rate discount_rate := by
  sorry

/-- Specific case for the problem --/
theorem miltonville_market_problem :
  ana_total 120 0.08 0.25 = bob_total 120 0.08 0.25 := by
  sorry

end tax_discount_commute_ana_equals_bob_miltonville_market_problem_l4115_411503


namespace diagonal_difference_l4115_411558

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The difference between the number of diagonals in an octagon and a heptagon -/
theorem diagonal_difference : num_diagonals 8 - num_diagonals 7 = 6 := by sorry

end diagonal_difference_l4115_411558


namespace cricket_target_run_l4115_411584

/-- Given a cricket game with specific run rates, calculate the target run. -/
theorem cricket_target_run (total_overs : ℕ) (first_overs : ℕ) (remaining_overs : ℕ)
  (first_rate : ℝ) (remaining_rate : ℝ) :
  total_overs = first_overs + remaining_overs →
  first_overs = 10 →
  remaining_overs = 22 →
  first_rate = 3.2 →
  remaining_rate = 11.363636363636363 →
  ↑⌊(first_overs : ℝ) * first_rate + (remaining_overs : ℝ) * remaining_rate⌋ = 282 := by
  sorry

end cricket_target_run_l4115_411584


namespace circle_radius_is_17_4_l4115_411507

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle is tangent to the y-axis at a given point -/
def isTangentToYAxis (c : Circle) (p : ℝ × ℝ) : Prop :=
  c.center.1 = c.radius ∧ p.1 = 0 ∧ p.2 = c.center.2

/-- Predicate to check if a given x-coordinate is an x-intercept of the circle -/
def isXIntercept (c : Circle) (x : ℝ) : Prop :=
  ∃ y, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ∧ y = 0

theorem circle_radius_is_17_4 (c : Circle) :
  isTangentToYAxis c (0, 2) →
  isXIntercept c 8 →
  c.radius = 17/4 := by
  sorry

end circle_radius_is_17_4_l4115_411507


namespace compound_vs_simple_interest_amount_l4115_411513

/-- The amount of money (in rupees) that results in a difference of 8.000000000000227
    between 8% compound interest and 4% simple interest over 2 years -/
theorem compound_vs_simple_interest_amount : ℝ := by
  -- Define the compound interest rate
  let compound_rate : ℝ := 0.08
  -- Define the simple interest rate
  let simple_rate : ℝ := 0.04
  -- Define the time period in years
  let time : ℝ := 2
  -- Define the difference between compound and simple interest amounts
  let difference : ℝ := 8.000000000000227

  -- Define the function for compound interest
  let compound_interest (p : ℝ) : ℝ := p * (1 + compound_rate) ^ time

  -- Define the function for simple interest
  let simple_interest (p : ℝ) : ℝ := p * (1 + simple_rate * time)

  -- The amount p that satisfies the condition
  let p : ℝ := difference / (compound_interest 1 - simple_interest 1)

  -- Assert that p is approximately equal to 92.59
  sorry


end compound_vs_simple_interest_amount_l4115_411513


namespace lemons_per_glass_l4115_411562

/-- Given that 9 glasses of lemonade can be made with 18 lemons,
    prove that the number of lemons needed per glass is 2. -/
theorem lemons_per_glass (total_glasses : ℕ) (total_lemons : ℕ) 
  (h1 : total_glasses = 9) (h2 : total_lemons = 18) :
  total_lemons / total_glasses = 2 := by
  sorry

end lemons_per_glass_l4115_411562


namespace boat_speed_in_still_water_l4115_411577

/-- Proves that a boat's speed in still water is 20 km/hr given specific conditions -/
theorem boat_speed_in_still_water :
  let current_speed : ℝ := 3
  let downstream_distance : ℝ := 9.2
  let downstream_time : ℝ := 24 / 60
  let downstream_speed : ℝ → ℝ := λ v => v + current_speed
  ∃ (v : ℝ), downstream_speed v * downstream_time = downstream_distance ∧ v = 20 :=
by sorry

end boat_speed_in_still_water_l4115_411577


namespace quadratic_properties_l4115_411531

/-- A quadratic function of the form y = -x^2 + bx + c -/
def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

theorem quadratic_properties :
  ∀ (b c : ℝ),
  (b = 4 ∧ c = 3 →
    (∃ (x y : ℝ), (x = 2 ∧ y = 7) ∧ 
      ∀ (t : ℝ), -1 ≤ t ∧ t ≤ 3 → 
        -2 ≤ quadratic_function b c t ∧ quadratic_function b c t ≤ 7)) ∧
  ((∀ (x : ℝ), x ≤ 0 → quadratic_function b c x ≤ 2) ∧
   (∀ (x : ℝ), x > 0 → quadratic_function b c x ≤ 3) ∧
   (∃ (x₁ x₂ : ℝ), x₁ ≤ 0 ∧ x₂ > 0 ∧ 
     quadratic_function b c x₁ = 2 ∧ quadratic_function b c x₂ = 3) →
    b = 2 ∧ c = 2) :=
by sorry

end quadratic_properties_l4115_411531


namespace negation_of_existence_negation_of_proposition_l4115_411506

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 1, p x) ↔ (∀ x > 1, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x > 1, x^2 - 2*x - 3 = 0) ↔ (∀ x > 1, x^2 - 2*x - 3 ≠ 0) := by sorry

end negation_of_existence_negation_of_proposition_l4115_411506


namespace correct_selection_count_l4115_411571

/-- The number of ways to select representatives satisfying given conditions -/
def select_representatives (num_boys num_girls : ℕ) : ℕ :=
  let total_students := num_boys + num_girls
  let num_subjects := 5
  3360

/-- The theorem stating the correct number of ways to select representatives -/
theorem correct_selection_count :
  select_representatives 5 3 = 3360 := by sorry

end correct_selection_count_l4115_411571


namespace problem_solution_l4115_411517

theorem problem_solution : 
  (99^2 = 9801) ∧ 
  ((-8)^2009 * (-1/8)^2008 = -8) := by
sorry

end problem_solution_l4115_411517


namespace problem_solution_l4115_411522

def g (x : ℝ) : ℝ := |x - 1| + |2*x + 4|
def f (a x : ℝ) : ℝ := |x - a| + 2 + a

theorem problem_solution :
  (∀ a : ℝ, ∀ x₁ : ℝ, ∃ x₂ : ℝ, g x₁ = f a x₂) →
  (∀ x : ℝ, x ∈ {x : ℝ | g x < 6} ↔ x ∈ Set.Ioo (-3) 1) ∧
  (∀ a : ℝ, (∀ x₁ : ℝ, ∃ x₂ : ℝ, g x₁ = f a x₂) → a ≤ 1) :=
by sorry

end problem_solution_l4115_411522


namespace relay_team_selection_l4115_411565

/-- The number of athletes in the track and field team -/
def total_athletes : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of twins -/
def num_twins : ℕ := 2

/-- The size of the relay team -/
def team_size : ℕ := 7

/-- The number of ways to choose the relay team -/
def num_ways : ℕ := 3762

theorem relay_team_selection :
  (num_triplets * (num_twins * (Nat.choose (total_athletes - num_triplets - 1 - 1) (team_size - 1 - 1)) +
  1 * (Nat.choose (total_athletes - num_triplets - 2) (team_size - 1 - 2)))) = num_ways :=
sorry

end relay_team_selection_l4115_411565


namespace triploid_oyster_principle_is_chromosome_variation_l4115_411589

/-- Represents the principle underlying oyster cultivation methods -/
inductive CultivationPrinciple
  | GeneticMutation
  | ChromosomeNumberVariation
  | GeneRecombination
  | ChromosomeStructureVariation

/-- Represents the ploidy level of an oyster -/
inductive Ploidy
  | Diploid
  | Triploid

/-- Represents the state of a cell during oyster reproduction -/
structure CellState where
  chromosomeSets : ℕ
  polarBodyReleased : Bool

/-- Represents the cultivation method for oysters -/
structure CultivationMethod where
  chemicalTreatment : Bool
  preventPolarBodyRelease : Bool
  solveFleshQualityDecline : Bool

/-- The principle of triploid oyster cultivation -/
def triploidOysterPrinciple (method : CultivationMethod) : CultivationPrinciple :=
  sorry

/-- Theorem stating that the principle of triploid oyster cultivation
    is chromosome number variation -/
theorem triploid_oyster_principle_is_chromosome_variation
  (method : CultivationMethod)
  (h1 : method.chemicalTreatment = true)
  (h2 : method.preventPolarBodyRelease = true)
  (h3 : method.solveFleshQualityDecline = true) :
  triploidOysterPrinciple method = CultivationPrinciple.ChromosomeNumberVariation :=
  sorry

end triploid_oyster_principle_is_chromosome_variation_l4115_411589


namespace mission_duration_l4115_411523

theorem mission_duration (planned_duration : ℝ) (overtime_percentage : ℝ) (second_mission_duration : ℝ) : 
  planned_duration = 5 ∧ 
  overtime_percentage = 0.6 ∧ 
  second_mission_duration = 3 → 
  planned_duration * (1 + overtime_percentage) + second_mission_duration = 11 :=
by sorry

end mission_duration_l4115_411523


namespace positive_integers_divisibility_l4115_411572

theorem positive_integers_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := by
  sorry

end positive_integers_divisibility_l4115_411572


namespace rabbit_measurement_probability_l4115_411576

theorem rabbit_measurement_probability :
  let total_rabbits : ℕ := 5
  let measured_rabbits : ℕ := 3
  let selected_rabbits : ℕ := 3
  let favorable_outcomes : ℕ := (measured_rabbits.choose 2) * ((total_rabbits - measured_rabbits).choose 1)
  let total_outcomes : ℕ := total_rabbits.choose selected_rabbits
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := by sorry

end rabbit_measurement_probability_l4115_411576


namespace stock_value_order_l4115_411549

/-- Represents the value of a stock over time -/
structure Stock :=
  (initial : ℝ)
  (first_year_change : ℝ)
  (second_year_change : ℝ)

/-- Calculates the final value of a stock after two years -/
def final_value (s : Stock) : ℝ :=
  s.initial * (1 + s.first_year_change) * (1 + s.second_year_change)

/-- The three stocks: Alabama Almonds (AA), Boston Beans (BB), and California Cauliflower (CC) -/
def AA : Stock := ⟨100, 0.2, -0.2⟩
def BB : Stock := ⟨100, -0.25, 0.25⟩
def CC : Stock := ⟨100, 0, 0⟩

theorem stock_value_order :
  final_value BB < final_value AA ∧ final_value AA < final_value CC :=
sorry

end stock_value_order_l4115_411549


namespace m_value_l4115_411548

theorem m_value (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : a + b = 2) : 
  m = 100 := by
sorry

end m_value_l4115_411548


namespace inequality_solution_l4115_411528

theorem inequality_solution (x : ℝ) : 3 * x^2 - x < 8 ↔ -4/3 < x ∧ x < 2 := by
  sorry

end inequality_solution_l4115_411528


namespace beths_school_students_l4115_411514

theorem beths_school_students (beth paul : ℕ) : 
  beth = 4 * paul →  -- Beth's school has 4 times as many students as Paul's
  beth + paul = 5000 →  -- Total students in both schools is 5000
  beth = 4000 :=  -- Prove that Beth's school has 4000 students
by
  sorry

end beths_school_students_l4115_411514


namespace diagonal_length_is_2_8_l4115_411587

/-- Represents a quadrilateral with given side lengths and a diagonal -/
structure Quadrilateral :=
  (side1 side2 side3 side4 diagonal : ℝ)

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Checks if the diagonal forms valid triangles with all possible combinations of sides -/
def diagonal_forms_valid_triangles (q : Quadrilateral) : Prop :=
  is_valid_triangle q.diagonal q.side1 q.side2 ∧
  is_valid_triangle q.diagonal q.side1 q.side3 ∧
  is_valid_triangle q.diagonal q.side1 q.side4 ∧
  is_valid_triangle q.diagonal q.side2 q.side3 ∧
  is_valid_triangle q.diagonal q.side2 q.side4 ∧
  is_valid_triangle q.diagonal q.side3 q.side4

theorem diagonal_length_is_2_8 (q : Quadrilateral) 
  (h1 : q.side1 = 1) (h2 : q.side2 = 2) (h3 : q.side3 = 5) (h4 : q.side4 = 7.5) (h5 : q.diagonal = 2.8) :
  diagonal_forms_valid_triangles q :=
by sorry

end diagonal_length_is_2_8_l4115_411587


namespace purely_imaginary_implies_a_equals_negative_one_l4115_411570

theorem purely_imaginary_implies_a_equals_negative_one :
  ∀ (a : ℝ), (Complex.I * (a - 1) : ℂ).im ≠ 0 →
  (a^2 - 1 + Complex.I * (a - 1) : ℂ).re = 0 →
  a = -1 :=
by sorry

end purely_imaginary_implies_a_equals_negative_one_l4115_411570


namespace tan_sin_product_l4115_411560

theorem tan_sin_product (A B : Real) (hA : A = 10 * Real.pi / 180) (hB : B = 35 * Real.pi / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
    1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) + 
    Real.tan A * (Real.sqrt 2 / 2) * (Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) := by
  sorry

end tan_sin_product_l4115_411560


namespace tanner_savings_l4115_411566

theorem tanner_savings (september_savings : ℤ) : 
  september_savings + 48 + 25 - 49 = 41 → september_savings = 17 := by
  sorry

end tanner_savings_l4115_411566


namespace inverse_variation_problem_l4115_411546

/-- The constant k in the inverse variation relationship -/
def k : ℝ := 192

/-- The relationship between z and x -/
def relation (z x : ℝ) : Prop := 3 * z = k / (x^3)

theorem inverse_variation_problem (z₁ z₂ x₁ x₂ : ℝ) 
  (h₁ : relation z₁ x₁)
  (h₂ : z₁ = 8)
  (h₃ : x₁ = 2)
  (h₄ : x₂ = 4) :
  z₂ = 1 ∧ relation z₂ x₂ := by
  sorry


end inverse_variation_problem_l4115_411546


namespace max_consecutive_integers_sum_max_consecutive_integers_sum_500_thirty_one_is_max_l4115_411520

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by sorry

theorem max_consecutive_integers_sum_500 : 
  ∀ k > 31, k * (k + 1) / 2 > 500 := by sorry

theorem thirty_one_is_max : 
  31 * 32 / 2 ≤ 500 ∧ ∀ n > 31, n * (n + 1) / 2 > 500 := by sorry

end max_consecutive_integers_sum_max_consecutive_integers_sum_500_thirty_one_is_max_l4115_411520


namespace blue_marble_difference_is_twenty_l4115_411559

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The difference in blue marbles between Jason and Tom -/
def blue_marble_difference : ℕ := jason_blue_marbles - tom_blue_marbles

theorem blue_marble_difference_is_twenty : blue_marble_difference = 20 := by
  sorry

end blue_marble_difference_is_twenty_l4115_411559


namespace range_of_a_l4115_411598

-- Define the conditions
def condition_p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4

def condition_q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, condition_q x → condition_p x a) →
  ∀ a : ℝ, -1 ≤ a ∧ a ≤ 6 :=
sorry

end range_of_a_l4115_411598


namespace range_m_for_always_negative_range_m_for_bounded_interval_l4115_411550

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Theorem 1
theorem range_m_for_always_negative (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Theorem 2
theorem range_m_for_bounded_interval (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 5) ↔ m < 6/7 :=
sorry

end range_m_for_always_negative_range_m_for_bounded_interval_l4115_411550


namespace correct_mark_proof_l4115_411564

/-- Proves that given a class of 20 pupils, if entering 73 instead of the correct mark
    increases the class average by 0.5, then the correct mark should have been 63. -/
theorem correct_mark_proof (n : ℕ) (wrong_mark correct_mark : ℝ) 
    (h1 : n = 20)
    (h2 : wrong_mark = 73)
    (h3 : (wrong_mark - correct_mark) / n = 0.5) :
  correct_mark = 63 := by
  sorry

end correct_mark_proof_l4115_411564


namespace haley_trees_died_l4115_411595

/-- The number of trees that died due to a typhoon -/
def trees_died (initial_trees : ℕ) (remaining_trees : ℕ) : ℕ :=
  initial_trees - remaining_trees

/-- Proof that 2 trees died in Haley's backyard after the typhoon -/
theorem haley_trees_died : trees_died 12 10 = 2 := by
  sorry

end haley_trees_died_l4115_411595


namespace obtuse_triangle_from_altitudes_l4115_411529

theorem obtuse_triangle_from_altitudes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a = 13) (h5 : b = 11) (h6 : c = 5) :
  (c^2 + b^2 - a^2) / (2 * b * c) < 0 :=
sorry

end obtuse_triangle_from_altitudes_l4115_411529


namespace initial_blue_balls_l4115_411510

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) (initial_blue : ℕ) : 
  total = 18 →
  removed = 3 →
  prob = 1 / 5 →
  (initial_blue - removed : ℚ) / (total - removed) = prob →
  initial_blue = 6 :=
by sorry

end initial_blue_balls_l4115_411510


namespace min_sum_of_system_l4115_411545

theorem min_sum_of_system (x y z : ℝ) 
  (eq1 : x + 3*y + 6*z = 1)
  (eq2 : x*y + 2*x*z + 6*y*z = -8)
  (eq3 : x*y*z = 2) :
  ∀ (a b c : ℝ), (a + 3*b + 6*c = 1 ∧ a*b + 2*a*c + 6*b*c = -8 ∧ a*b*c = 2) → 
  x + y + z ≤ a + b + c ∧ x + y + z = -8/3 :=
by sorry

end min_sum_of_system_l4115_411545


namespace open_box_volume_is_5120_l4115_411578

/-- The volume of an open box formed by cutting squares from a rectangular sheet. -/
def open_box_volume (sheet_length sheet_width cut_side : ℝ) : ℝ :=
  (sheet_length - 2 * cut_side) * (sheet_width - 2 * cut_side) * cut_side

/-- Theorem: The volume of the open box is 5120 m³ -/
theorem open_box_volume_is_5120 :
  open_box_volume 48 36 8 = 5120 := by
  sorry

end open_box_volume_is_5120_l4115_411578


namespace exam_questions_count_l4115_411525

/-- Calculates the total number of questions in an examination given specific conditions. -/
theorem exam_questions_count 
  (type_a_count : ℕ)
  (type_a_time : ℕ)
  (total_time : ℕ)
  (h1 : type_a_count = 50)
  (h2 : type_a_time = 72)
  (h3 : total_time = 180)
  (h4 : type_a_time * 2 ≤ total_time) :
  ∃ (type_b_count : ℕ),
    (type_a_count + type_b_count = 200) ∧
    (type_a_time + type_b_count * (type_a_time / type_a_count / 2) = total_time) :=
by sorry

end exam_questions_count_l4115_411525


namespace problem_solution_l4115_411500

theorem problem_solution : ∃! x : ℝ, x * 13.26 + x * 9.43 + x * 77.31 = 470 ∧ x = 4.7 := by
  sorry

end problem_solution_l4115_411500


namespace complement_of_union_l4115_411542

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4}

theorem complement_of_union (h1 : U = {1, 2, 3, 4, 5}) 
                            (h2 : A = {1, 2, 3}) 
                            (h3 : B = {3, 4}) : 
  U \ (A ∪ B) = {5} := by
  sorry

end complement_of_union_l4115_411542


namespace room_area_square_inches_l4115_411552

-- Define the conversion rate from feet to inches
def inches_per_foot : ℕ := 12

-- Define the side length of the room in feet
def room_side_feet : ℕ := 10

-- Theorem to prove the area of the room in square inches
theorem room_area_square_inches : 
  (room_side_feet * inches_per_foot) ^ 2 = 14400 := by
  sorry

end room_area_square_inches_l4115_411552


namespace clerts_in_120_degrees_proof_l4115_411534

/-- Represents the number of clerts in a full circle for Martian angle measurement -/
def clerts_in_full_circle : ℕ := 800

/-- Converts degrees to clerts -/
def degrees_to_clerts (degrees : ℚ) : ℚ :=
  (degrees / 360) * clerts_in_full_circle

/-- The number of clerts in a 120° angle -/
def clerts_in_120_degrees : ℕ := 267

theorem clerts_in_120_degrees_proof : 
  ⌊degrees_to_clerts 120⌋ = clerts_in_120_degrees :=
sorry

end clerts_in_120_degrees_proof_l4115_411534


namespace problem_solution_l4115_411573

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x ≤ -3 ∨ (23 ≤ x ∧ x < 27))
  (h2 : a < b) :
  a + 2*b + 3*c = 71 := by
sorry

end problem_solution_l4115_411573


namespace complex_fraction_equality_l4115_411568

theorem complex_fraction_equality : (3 - Complex.I) / (1 - Complex.I) = 2 + Complex.I := by
  sorry

end complex_fraction_equality_l4115_411568


namespace G_initial_conditions_G_recurrence_G_20_diamonds_l4115_411519

/-- The number of diamonds in the n-th figure of sequence G -/
def G (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 5
  else 2 * n * (n + 1)

/-- The sequence G satisfies the given initial conditions -/
theorem G_initial_conditions :
  G 1 = 1 ∧ G 2 = 5 ∧ G 3 = 17 := by sorry

/-- The recurrence relation for G_n, n ≥ 3 -/
theorem G_recurrence (n : ℕ) (h : n ≥ 3) :
  G n = G (n-1) + 4 * n := by sorry

/-- The main theorem: G_20 has 840 diamonds -/
theorem G_20_diamonds : G 20 = 840 := by sorry

end G_initial_conditions_G_recurrence_G_20_diamonds_l4115_411519


namespace unique_integer_pair_for_equal_W_values_l4115_411583

/-- The polynomial W(x) = x^4 - 3x^3 + 5x^2 - 9x -/
def W (x : ℤ) : ℤ := x^4 - 3*x^3 + 5*x^2 - 9*x

/-- Theorem: The only pair of different integers (a, b) satisfying W(a) = W(b) is (1, 2) -/
theorem unique_integer_pair_for_equal_W_values :
  ∀ a b : ℤ, a ≠ b ∧ W a = W b ↔ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by sorry

end unique_integer_pair_for_equal_W_values_l4115_411583


namespace floor_tiles_theorem_l4115_411592

/-- Represents a square floor divided into four congruent sections -/
structure SquareFloor :=
  (section_side : ℕ)

/-- The number of tiles on the main diagonal of the entire floor -/
def main_diagonal_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.section_side - 3

/-- The total number of tiles covering the entire floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  (4 * floor.section_side) ^ 2

/-- Theorem stating the relationship between the number of tiles on the main diagonal
    and the total number of tiles on the floor -/
theorem floor_tiles_theorem (floor : SquareFloor) 
  (h : main_diagonal_tiles floor = 75) : total_tiles floor = 25600 := by
  sorry

#check floor_tiles_theorem

end floor_tiles_theorem_l4115_411592


namespace postcard_height_l4115_411582

theorem postcard_height (perimeter width : ℝ) (h_perimeter : perimeter = 20) (h_width : width = 6) :
  let height := (perimeter - 2 * width) / 2
  height = 4 := by sorry

end postcard_height_l4115_411582


namespace arithmetic_sequence_constant_ratio_values_l4115_411521

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The ratio of a_n to a_{2n} is constant -/
def constant_ratio (a : ℕ → ℝ) : Prop :=
  ∃ c, ∀ n, a n / a (2 * n) = c

theorem arithmetic_sequence_constant_ratio_values
  (a : ℕ → ℝ) (h1 : arithmetic_sequence a) (h2 : constant_ratio a) :
  ∃ c, (c = 1 ∨ c = 1/2) ∧ ∀ n, a n / a (2 * n) = c :=
sorry

end arithmetic_sequence_constant_ratio_values_l4115_411521


namespace imaginary_part_of_z_l4115_411585

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 / (1 + i)) + i^3) = -3/2 :=
sorry

end imaginary_part_of_z_l4115_411585


namespace sixth_term_of_geometric_sequence_l4115_411544

def geometric_sequence (a : ℕ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem sixth_term_of_geometric_sequence 
  (a₁ : ℕ) (a₅ : ℕ) (h₁ : a₁ = 3) (h₅ : a₅ = 375) :
  ∃ r : ℝ, 
    geometric_sequence a₁ r 5 = a₅ ∧ 
    geometric_sequence a₁ r 6 = 9375 := by
sorry

end sixth_term_of_geometric_sequence_l4115_411544


namespace inverse_variation_problem_l4115_411518

/-- Given that x varies inversely as the square of y, prove that x = 1/9 when y = 6,
    given that y = 2 when x = 1. -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x = k / (y^2)) 
  (h2 : 1 = k / (2^2)) : 
  (y = 6) → (x = 1/9) := by
  sorry

end inverse_variation_problem_l4115_411518


namespace matrix_equation_solution_l4115_411540

theorem matrix_equation_solution :
  ∀ (a b c d : ℝ),
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![1, a; b, 1]
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![c, 2; 0, d]
  M * N = !![2, 4; -2, 0] →
  a = 1 ∧ b = -1 ∧ c = 2 ∧ d = 2 := by
sorry

end matrix_equation_solution_l4115_411540


namespace quadratic_inequality_l4115_411579

theorem quadratic_inequality (x : ℝ) : -9*x^2 + 6*x + 15 > 0 ↔ -1 < x ∧ x < 5/3 := by
  sorry

end quadratic_inequality_l4115_411579


namespace simple_interest_principal_l4115_411536

/-- Simple interest calculation --/
theorem simple_interest_principal (interest_rate : ℚ) (time_months : ℕ) (interest_earned : ℕ) (principal : ℕ) : 
  interest_rate = 50 / 3 → 
  time_months = 9 → 
  interest_earned = 8625 →
  principal = 69000 →
  interest_earned * 1200 = principal * interest_rate * time_months := by
  sorry

#check simple_interest_principal

end simple_interest_principal_l4115_411536


namespace sqrt_37_between_6_and_7_l4115_411527

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := by
  sorry

end sqrt_37_between_6_and_7_l4115_411527


namespace bus_passenger_count_l4115_411505

/-- Calculates the total number of passengers transported by a bus. -/
def totalPassengers (numTrips : ℕ) (initialPassengers : ℕ) (passengerDecrease : ℕ) : ℕ :=
  (numTrips * (2 * initialPassengers - (numTrips - 1) * passengerDecrease)) / 2

/-- Proves that the total number of passengers transported is 1854. -/
theorem bus_passenger_count : totalPassengers 18 120 2 = 1854 := by
  sorry

#eval totalPassengers 18 120 2

end bus_passenger_count_l4115_411505


namespace factors_of_48_l4115_411563

/-- The number of distinct positive factors of 48 -/
def num_factors_48 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 48 is 10 -/
theorem factors_of_48 : num_factors_48 = 10 := by sorry

end factors_of_48_l4115_411563


namespace grape_crates_count_l4115_411501

/-- Proves that the number of grape crates is 13 given the total number of crates and the number of mango and passion fruit crates. -/
theorem grape_crates_count (total_crates mango_crates passion_fruit_crates : ℕ) 
  (h1 : total_crates = 50)
  (h2 : mango_crates = 20)
  (h3 : passion_fruit_crates = 17) :
  total_crates - (mango_crates + passion_fruit_crates) = 13 := by
  sorry

end grape_crates_count_l4115_411501


namespace simplify_sqrt_expression_l4115_411530

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l4115_411530


namespace largest_product_sum_1976_l4115_411574

theorem largest_product_sum_1976 (n : ℕ) (factors : List ℕ) : 
  (factors.sum = 1976) →
  (factors.prod ≤ 2 * 3^658) :=
sorry

end largest_product_sum_1976_l4115_411574


namespace center_line_perpendicular_iff_arithmetic_progression_l4115_411541

/-- A triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The incenter of a triangle. -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle. -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The line passing through two points. -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The angle bisectors of a triangle. -/
def angle_bisectors (t : Triangle) : List (Set (ℝ × ℝ)) := sorry

/-- Two lines are perpendicular. -/
def perpendicular (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- The sides of a triangle form an arithmetic progression. -/
def arithmetic_progression (t : Triangle) : Prop :=
  t.a - t.b = t.b - t.c

theorem center_line_perpendicular_iff_arithmetic_progression (t : Triangle) :
  ∃ (bisector : Set (ℝ × ℝ)), bisector ∈ angle_bisectors t ∧
    perpendicular (line_through (incenter t) (circumcenter t)) bisector
  ↔ arithmetic_progression t := by sorry

end center_line_perpendicular_iff_arithmetic_progression_l4115_411541


namespace price_increase_percentage_l4115_411511

theorem price_increase_percentage (P : ℝ) (h : P > 0) : 
  let cheaper_price := 0.8 * P
  let price_increase := P - cheaper_price
  let percentage_increase := (price_increase / cheaper_price) * 100
  percentage_increase = 25 := by sorry

end price_increase_percentage_l4115_411511


namespace hexagon_diagonals_l4115_411512

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end hexagon_diagonals_l4115_411512


namespace boris_fudge_amount_l4115_411535

-- Define the conversion rate from pounds to ounces
def poundsToOunces (pounds : ℝ) : ℝ := pounds * 16

-- Define the amount of fudge eaten by each person
def tomasFudge : ℝ := 1.5
def katyaFudge : ℝ := 0.5

-- Define the total amount of fudge eaten by all three friends in ounces
def totalFudgeOunces : ℝ := 64

-- Theorem to prove
theorem boris_fudge_amount :
  let borisFudgeOunces := totalFudgeOunces - (poundsToOunces tomasFudge + poundsToOunces katyaFudge)
  borisFudgeOunces / 16 = 2 := by
  sorry

end boris_fudge_amount_l4115_411535


namespace theresa_final_week_hours_l4115_411596

/-- The number of weeks Theresa needs to work -/
def total_weeks : ℕ := 6

/-- The required average hours per week -/
def required_average : ℕ := 12

/-- The hours worked in the first five weeks -/
def first_five_weeks : List ℕ := [10, 13, 9, 14, 11]

/-- The sum of hours worked in the first five weeks -/
def sum_first_five : ℕ := first_five_weeks.sum

/-- The number of hours Theresa needs to work in the final week -/
def final_week_hours : ℕ := 15

theorem theresa_final_week_hours :
  (sum_first_five + final_week_hours) / total_weeks = required_average :=
sorry

end theresa_final_week_hours_l4115_411596


namespace louie_last_match_goals_l4115_411561

/-- The number of goals Louie scored in the last match -/
def last_match_goals : ℕ := sorry

/-- The number of seasons Louie's brother has played -/
def brothers_seasons : ℕ := 3

/-- The number of games in each season -/
def games_per_season : ℕ := 50

/-- The total number of goals scored by both brothers -/
def total_goals : ℕ := 1244

/-- The number of goals Louie scored in previous matches -/
def previous_goals : ℕ := 40

theorem louie_last_match_goals : 
  last_match_goals = 4 ∧
  brothers_seasons * games_per_season * (2 * last_match_goals) + 
  previous_goals + last_match_goals = total_goals :=
by sorry

end louie_last_match_goals_l4115_411561


namespace interval_intersection_l4115_411556

theorem interval_intersection (x : ℝ) : 
  (|4 - x| < 5 ∧ x^2 < 36) ↔ (-1 < x ∧ x < 6) := by
  sorry

end interval_intersection_l4115_411556


namespace problem_solution_l4115_411555

/-- Given M = 2x + y, N = 2x - y, P = xy, M = 4, and N = 2, prove that P = 1.5 -/
theorem problem_solution (x y M N P : ℝ) 
  (hM : M = 2*x + y)
  (hN : N = 2*x - y)
  (hP : P = x*y)
  (hM_val : M = 4)
  (hN_val : N = 2) :
  P = 1.5 := by
  sorry

end problem_solution_l4115_411555


namespace geometric_sequence_sum_l4115_411515

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) →
  (a 2 + a 6 = 2) := by
  sorry

end geometric_sequence_sum_l4115_411515


namespace xyz_sum_l4115_411593

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = y.val * z.val + x.val)
  (h2 : y.val * z.val + x.val = z.val * x.val + y.val)
  (h3 : z.val * x.val + y.val = x.val * y.val + z.val)
  (h4 : x.val * y.val + z.val = 56) : 
  x.val + y.val + z.val = 21 := by
sorry

end xyz_sum_l4115_411593


namespace cyclists_distance_l4115_411532

theorem cyclists_distance (a b : ℝ) : 
  (a = b^2) ∧ (a - 1 = 3 * (b - 1)) → (a - b = 0 ∨ a - b = 2) :=
by sorry

end cyclists_distance_l4115_411532


namespace max_value_of_expression_l4115_411590

theorem max_value_of_expression (t : ℝ) :
  (∃ (c : ℝ), ∀ (t : ℝ), (3^t - 4*t)*t / 9^t ≤ c) ∧
  (∃ (t : ℝ), (3^t - 4*t)*t / 9^t = 1/16) := by
  sorry

end max_value_of_expression_l4115_411590


namespace polynomial_symmetry_l4115_411539

-- Define the polynomial function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem polynomial_symmetry (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end polynomial_symmetry_l4115_411539


namespace polygon_sides_l4115_411553

/-- A polygon with side length 7 and perimeter 42 has 6 sides -/
theorem polygon_sides (side_length : ℕ) (perimeter : ℕ) (h1 : side_length = 7) (h2 : perimeter = 42) :
  perimeter / side_length = 6 := by
  sorry

end polygon_sides_l4115_411553


namespace ratio_of_sum_to_difference_l4115_411580

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  0 < b → b < a → a + b = 7 * (a - b) → a / b = 2 := by sorry

end ratio_of_sum_to_difference_l4115_411580


namespace smallest_sum_l4115_411538

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Represents the problem setup -/
def ProblemSetup (x y m : ℕ) : Prop :=
  TwoDigitInt x ∧
  TwoDigitInt y ∧
  y = reverseDigits x ∧
  x^2 + y^2 = m^2 ∧
  ∃ k, x + y = 9 * (2 * k + 1)

theorem smallest_sum (x y m : ℕ) (h : ProblemSetup x y m) :
  x + y + m ≥ 169 :=
sorry

end smallest_sum_l4115_411538


namespace saturday_duty_probability_is_one_sixth_l4115_411594

/-- A person's weekly night duty schedule -/
structure DutySchedule where
  total_duties : ℕ
  sunday_duty : Bool
  h_total : total_duties = 2
  h_sunday : sunday_duty = true

/-- The probability of being on duty on Saturday night given the duty schedule -/
def saturday_duty_probability (schedule : DutySchedule) : ℚ :=
  1 / 6

/-- Theorem stating that the probability of Saturday duty is 1/6 -/
theorem saturday_duty_probability_is_one_sixth (schedule : DutySchedule) :
  saturday_duty_probability schedule = 1 / 6 := by
  sorry

end saturday_duty_probability_is_one_sixth_l4115_411594


namespace mika_stickers_left_l4115_411533

/-- The number of stickers Mika has left after various changes -/
def stickers_left (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika has 2 stickers left -/
theorem mika_stickers_left :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end mika_stickers_left_l4115_411533


namespace no_such_hexagon_exists_l4115_411551

-- Define a hexagon as a collection of 6 points in 2D space
def Hexagon := Fin 6 → ℝ × ℝ

-- Define convexity for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define the condition that all sides are greater than 1
def all_sides_greater_than_one (h : Hexagon) : Prop :=
  ∀ i : Fin 6, dist (h i) (h ((i + 1) % 6)) > 1

-- Define the condition that the distance from M to any vertex is less than 1
def all_vertices_less_than_one_from_point (h : Hexagon) (m : ℝ × ℝ) : Prop :=
  ∀ i : Fin 6, dist (h i) m < 1

-- The main theorem
theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    all_sides_greater_than_one h ∧
    all_vertices_less_than_one_from_point h m :=
sorry

end no_such_hexagon_exists_l4115_411551


namespace average_of_next_ten_l4115_411567

def consecutive_integers_average (c d : ℤ) : Prop :=
  (7 * d = c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) ∧
  (c > 0)

theorem average_of_next_ten (c d : ℤ) 
  (h : consecutive_integers_average c d) : 
  (((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + 
    (d + 6) + (d + 7) + (d + 8) + (d + 9) + (d + 10)) / 10) = c + 9 :=
by sorry

end average_of_next_ten_l4115_411567


namespace parallel_lines_parallelograms_l4115_411526

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of parallelograms formed by intersecting parallel lines -/
def parallelograms_count (set1 : ℕ) (set2 : ℕ) : ℕ :=
  choose_two set1 * choose_two set2

theorem parallel_lines_parallelograms :
  parallelograms_count 3 5 = 30 := by
  sorry

end parallel_lines_parallelograms_l4115_411526


namespace greatest_fraction_with_same_digit_sum_l4115_411524

/-- A function that returns the sum of digits of a number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Predicate to check if a number is a four-digit number -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_fraction_with_same_digit_sum :
  ∀ m n : ℕ, isFourDigit m → isFourDigit n → sumOfDigits m = sumOfDigits n →
  (m : ℚ) / n ≤ 9900 / 1089 :=
sorry

end greatest_fraction_with_same_digit_sum_l4115_411524


namespace third_number_proof_l4115_411502

theorem third_number_proof (x : ℕ) : 
  let second := 3 * x - 7
  let third := 2 * x + 2
  x + second + third = 168 →
  third = 60 := by
sorry

end third_number_proof_l4115_411502


namespace unpainted_cubes_in_4x4x4_cube_l4115_411554

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  painted_size : ℕ
  total_cubes : ℕ
  painted_cubes : ℕ

/-- Theorem: In a 4x4x4 cube with 2x2 squares painted on each face, 56 unit cubes are unpainted -/
theorem unpainted_cubes_in_4x4x4_cube (c : PaintedCube) 
  (h_size : c.size = 4)
  (h_painted : c.painted_size = 2)
  (h_total : c.total_cubes = c.size ^ 3)
  (h_painted_count : c.painted_cubes = 8) :
  c.total_cubes - c.painted_cubes = 56 := by
  sorry

#check unpainted_cubes_in_4x4x4_cube

end unpainted_cubes_in_4x4x4_cube_l4115_411554


namespace expression_evaluation_l4115_411543

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 7
  (2*x + 3) * (2*x - 3) - (x + 2)^2 + 4*(x + 3) = 20 := by sorry

end expression_evaluation_l4115_411543


namespace fencing_cost_l4115_411504

-- Define the ratio of sides
def ratio_length : ℚ := 3
def ratio_width : ℚ := 4

-- Define the area of the field
def area : ℚ := 8112

-- Define the cost per meter in rupees
def cost_per_meter : ℚ := 25 / 100

-- Theorem statement
theorem fencing_cost :
  let x : ℚ := (area / (ratio_length * ratio_width)) ^ (1/2)
  let length : ℚ := ratio_length * x
  let width : ℚ := ratio_width * x
  let perimeter : ℚ := 2 * (length + width)
  let total_cost : ℚ := perimeter * cost_per_meter
  total_cost = 91 := by sorry

end fencing_cost_l4115_411504


namespace positive_expression_l4115_411575

theorem positive_expression (a b : ℝ) (ha : 0 < a ∧ a < 2) (hb : -2 < b ∧ b < 0) :
  0 < b + a^2 := by
  sorry

end positive_expression_l4115_411575


namespace games_for_23_teams_l4115_411586

/-- A single-elimination tournament where teams are eliminated after one loss and no ties are possible. -/
structure Tournament :=
  (num_teams : ℕ)

/-- The number of games needed to declare a champion in a single-elimination tournament. -/
def games_to_champion (t : Tournament) : ℕ := t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams, 22 games are needed to declare a champion. -/
theorem games_for_23_teams :
  ∀ t : Tournament, t.num_teams = 23 → games_to_champion t = 22 := by
  sorry


end games_for_23_teams_l4115_411586


namespace midpoint_locus_of_intersection_l4115_411537

/-- Given an arithmetic sequence A, B, C, this function represents the line Ax + By + C = 0 --/
def line (A B C : ℝ) (x y : ℝ) : Prop :=
  A * x + B * y + C = 0

/-- The parabola y = -2x^2 --/
def parabola (x y : ℝ) : Prop :=
  y = -2 * x^2

/-- The locus of the midpoint --/
def midpoint_locus (x y : ℝ) : Prop :=
  y + 1 = -(2 * x - 1)^2

/-- The main theorem --/
theorem midpoint_locus_of_intersection
  (A B C : ℝ) -- A, B, C are real numbers
  (h_arithmetic : A - 2*B + C = 0) -- A, B, C form an arithmetic sequence
  (x y : ℝ) -- x and y are real numbers
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line A B C x₁ y₁ ∧ parabola x₁ y₁ ∧
    line A B C x₂ y₂ ∧ parabola x₂ y₂ ∧
    x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2) -- (x, y) is the midpoint of the chord of intersection
  : midpoint_locus x y :=
sorry

end midpoint_locus_of_intersection_l4115_411537
