import Mathlib

namespace NUMINAMATH_CALUDE_dress_savings_l34_3474

/-- Given a dress with an original cost of $180, if someone buys it for 10 dollars less than half the price, they save $100. -/
theorem dress_savings (original_cost : ℕ) (purchase_price : ℕ) : 
  original_cost = 180 → 
  purchase_price = original_cost / 2 - 10 → 
  original_cost - purchase_price = 100 := by
sorry

end NUMINAMATH_CALUDE_dress_savings_l34_3474


namespace NUMINAMATH_CALUDE_rice_purchase_comparison_l34_3460

theorem rice_purchase_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (200 / (100 / a + 100 / b)) ≤ ((100 * a + 100 * b) / 200) := by
  sorry

#check rice_purchase_comparison

end NUMINAMATH_CALUDE_rice_purchase_comparison_l34_3460


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l34_3493

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l34_3493


namespace NUMINAMATH_CALUDE_bakery_flour_calculation_l34_3452

/-- Given a bakery that uses wheat flour and white flour, prove that the amount of white flour
    used is equal to the total amount of flour used minus the amount of wheat flour used. -/
theorem bakery_flour_calculation (total_flour white_flour wheat_flour : ℝ) 
    (h1 : total_flour = 0.3)
    (h2 : wheat_flour = 0.2) :
  white_flour = total_flour - wheat_flour := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_calculation_l34_3452


namespace NUMINAMATH_CALUDE_sum_inequality_l34_3446

theorem sum_inequality (a b c : ℝ) 
  (ha : 1/Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2)
  (hb : 1/Real.sqrt 2 ≤ b ∧ b ≤ Real.sqrt 2)
  (hc : 1/Real.sqrt 2 ≤ c ∧ c ≤ Real.sqrt 2) :
  (3/(a+2*b) + 3/(b+2*c) + 3/(c+2*a)) ≥ (2/(a+b) + 2/(b+c) + 2/(c+a)) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l34_3446


namespace NUMINAMATH_CALUDE_original_price_after_discounts_l34_3414

/-- 
Given an article sold at $126 after two successive discounts of 10% and 20%,
prove that its original price was $175.
-/
theorem original_price_after_discounts (final_price : ℝ) 
  (h1 : final_price = 126) 
  (discount1 : ℝ) (h2 : discount1 = 0.1)
  (discount2 : ℝ) (h3 : discount2 = 0.2) : 
  ∃ (original_price : ℝ), 
    original_price = 175 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) := by
  sorry

end NUMINAMATH_CALUDE_original_price_after_discounts_l34_3414


namespace NUMINAMATH_CALUDE_no_opposite_divisibility_l34_3495

theorem no_opposite_divisibility (k n a : ℕ) : 
  k ≥ 3 → n ≥ 3 → Odd k → Odd n → a ≥ 1 → 
  k ∣ (2^a + 1) → n ∣ (2^a - 1) → 
  ¬∃ b : ℕ, b ≥ 1 ∧ k ∣ (2^b - 1) ∧ n ∣ (2^b + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_opposite_divisibility_l34_3495


namespace NUMINAMATH_CALUDE_player_B_most_stable_l34_3442

/-- Represents a player in the shooting test -/
inductive Player : Type
  | A
  | B
  | C
  | D

/-- Returns the variance of a given player -/
def variance (p : Player) : ℝ :=
  match p with
  | Player.A => 0.66
  | Player.B => 0.52
  | Player.C => 0.58
  | Player.D => 0.62

/-- Defines what it means for a player to have the most stable performance -/
def has_most_stable_performance (p : Player) : Prop :=
  ∀ q : Player, variance p ≤ variance q

/-- Theorem stating that Player B has the most stable performance -/
theorem player_B_most_stable :
  has_most_stable_performance Player.B := by
  sorry

end NUMINAMATH_CALUDE_player_B_most_stable_l34_3442


namespace NUMINAMATH_CALUDE_expand_and_simplify_l34_3458

theorem expand_and_simplify (x : ℝ) : -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l34_3458


namespace NUMINAMATH_CALUDE_increasing_sequences_count_l34_3413

theorem increasing_sequences_count :
  let n := 2013
  let k := 12
  let count := Nat.choose (((n - 1) / 2) + k - 1) k
  (count = Nat.choose 1017 12) ∧
  (1017 % 1000 = 17) := by sorry

end NUMINAMATH_CALUDE_increasing_sequences_count_l34_3413


namespace NUMINAMATH_CALUDE_games_attended_l34_3454

def total_games : ℕ := 39
def missed_games : ℕ := 25

theorem games_attended : total_games - missed_games = 14 := by
  sorry

end NUMINAMATH_CALUDE_games_attended_l34_3454


namespace NUMINAMATH_CALUDE_larry_jogging_days_l34_3449

theorem larry_jogging_days (daily_jog_time : ℕ) (second_week_days : ℕ) (total_time : ℕ) : 
  daily_jog_time = 30 →
  second_week_days = 5 →
  total_time = 4 * 60 →
  (total_time - second_week_days * daily_jog_time) / daily_jog_time = 3 :=
by sorry

end NUMINAMATH_CALUDE_larry_jogging_days_l34_3449


namespace NUMINAMATH_CALUDE_cone_base_radius_l34_3477

theorem cone_base_radius 
  (unfolded_area : ℝ) 
  (generatrix : ℝ) 
  (h1 : unfolded_area = 15 * Real.pi) 
  (h2 : generatrix = 5) : 
  ∃ (base_radius : ℝ), base_radius = 3 ∧ unfolded_area = Real.pi * base_radius * generatrix :=
by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l34_3477


namespace NUMINAMATH_CALUDE_triangle_reconstruction_possible_l34_3471

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

end NUMINAMATH_CALUDE_triangle_reconstruction_possible_l34_3471


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l34_3434

theorem existence_of_special_numbers : ∃ (a b c : ℕ), 
  (a > 10^10 ∧ b > 10^10 ∧ c > 10^10) ∧
  (a * b * c) % (a + 2012) = 0 ∧
  (a * b * c) % (b + 2012) = 0 ∧
  (a * b * c) % (c + 2012) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l34_3434


namespace NUMINAMATH_CALUDE_pedros_plums_l34_3499

theorem pedros_plums (total_fruits : ℕ) (total_cost : ℕ) (plum_cost : ℕ) (peach_cost : ℕ) 
  (h1 : total_fruits = 32)
  (h2 : total_cost = 52)
  (h3 : plum_cost = 2)
  (h4 : peach_cost = 1) :
  ∃ (plums peaches : ℕ), 
    plums + peaches = total_fruits ∧ 
    plum_cost * plums + peach_cost * peaches = total_cost ∧
    plums = 20 :=
by sorry

end NUMINAMATH_CALUDE_pedros_plums_l34_3499


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l34_3486

-- Equation 1
theorem equation_one_solution :
  ∃! x : ℚ, (x / (2 * x - 1)) + (2 / (1 - 2 * x)) = 3 :=
by sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ x : ℚ, (4 / (x^2 - 4)) - (1 / (x - 2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l34_3486


namespace NUMINAMATH_CALUDE_problem_statement_l34_3430

open Real

theorem problem_statement :
  (¬ (∀ x : ℝ, sin x ≠ 1) ↔ (∃ x : ℝ, sin x = 1)) ∧
  ((∀ α : ℝ, α = π/6 → sin α = 1/2) ∧ ¬(∀ α : ℝ, sin α = 1/2 → α = π/6)) ∧
  ¬(∀ a : ℕ → ℝ, (∀ n : ℕ, a (n+1) = 3 * a n) ↔ (∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n)) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_statement_l34_3430


namespace NUMINAMATH_CALUDE_newspaper_delivery_start_l34_3412

def building_floors : ℕ := 20

def start_floor : ℕ → Prop
| f => ∃ (current : ℕ), 
    current = f + 5 - 2 + 7 ∧ 
    current = building_floors - 9

theorem newspaper_delivery_start : start_floor 1 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_delivery_start_l34_3412


namespace NUMINAMATH_CALUDE_max_inscribed_cylinder_volume_l34_3418

/-- 
Given a right circular cone with base radius R and height M, 
prove that the maximum volume of an inscribed right circular cylinder 
is 4πMR²/27, and this volume is 4/9 of the cone's volume.
-/
theorem max_inscribed_cylinder_volume (R M : ℝ) (hR : R > 0) (hM : M > 0) :
  let cone_volume := (1/3) * π * R^2 * M
  let max_cylinder_volume := (4/27) * π * M * R^2
  max_cylinder_volume = (4/9) * cone_volume := by
  sorry


end NUMINAMATH_CALUDE_max_inscribed_cylinder_volume_l34_3418


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l34_3466

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 2x^2 + (2 + 1/2)x + 1/2 has discriminant 2.25 -/
theorem quadratic_discriminant : discriminant 2 (2 + 1/2) (1/2) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l34_3466


namespace NUMINAMATH_CALUDE_cacl₂_production_l34_3480

/-- Represents the chemical reaction: CaCO₃ + 2HCl → CaCl₂ + CO₂ + H₂O -/
structure ChemicalReaction where
  cacO₃ : ℚ  -- moles of CaCO₃
  hcl : ℚ    -- moles of HCl
  cacl₂ : ℚ  -- moles of CaCl₂ produced

/-- The stoichiometric ratio of the reaction -/
def stoichiometricRatio : ℚ := 2

/-- Calculates the amount of CaCl₂ produced based on the limiting reactant -/
def calcCaCl₂Produced (reaction : ChemicalReaction) : ℚ :=
  min reaction.cacO₃ (reaction.hcl / stoichiometricRatio)

/-- Theorem stating that 2 moles of CaCl₂ are produced when 4 moles of HCl react with 2 moles of CaCO₃ -/
theorem cacl₂_production (reaction : ChemicalReaction) 
  (h1 : reaction.cacO₃ = 2)
  (h2 : reaction.hcl = 4) :
  calcCaCl₂Produced reaction = 2 := by
  sorry

end NUMINAMATH_CALUDE_cacl₂_production_l34_3480


namespace NUMINAMATH_CALUDE_initial_type_x_plants_l34_3416

def initial_total : ℕ := 50
def final_total : ℕ := 1042
def days : ℕ := 12
def x_growth_factor : ℕ := 2^4  -- Type X doubles 4 times in 12 days
def y_growth_factor : ℕ := 3^3  -- Type Y triples 3 times in 12 days

theorem initial_type_x_plants : 
  ∃ (x y : ℕ), 
    x + y = initial_total ∧ 
    x_growth_factor * x + y_growth_factor * y = final_total ∧ 
    x = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_type_x_plants_l34_3416


namespace NUMINAMATH_CALUDE_number_multiplied_by_7000_l34_3443

theorem number_multiplied_by_7000 : ∃ x : ℝ, x * 7000 = (28000 : ℝ) * (100 : ℝ)^1 ∧ x = 400 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_7000_l34_3443


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l34_3482

theorem simplify_nested_expression (x : ℝ) : 2 - (3 - (2 - (5 - (3 - x)))) = -1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l34_3482


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l34_3491

def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def B : Set ℝ := {-2, -1, 0, 1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l34_3491


namespace NUMINAMATH_CALUDE_speed_adjustment_l34_3445

/-- Given a constant distance traveled at 10 km/h in 6 minutes,
    the speed required to travel the same distance in 8 minutes is 7.5 km/h. -/
theorem speed_adjustment (initial_speed initial_time new_time : ℝ) :
  initial_speed = 10 →
  initial_time = 6 / 60 →
  new_time = 8 / 60 →
  let distance := initial_speed * initial_time
  let new_speed := distance / new_time
  new_speed = 7.5 := by
sorry

end NUMINAMATH_CALUDE_speed_adjustment_l34_3445


namespace NUMINAMATH_CALUDE_condition_sufficiency_for_increasing_f_l34_3468

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 3

theorem condition_sufficiency_for_increasing_f :
  (∀ x ≥ 2, Monotone (f 1)) ∧
  ¬(∀ a : ℝ, (∀ x ≥ 2, Monotone (f a)) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficiency_for_increasing_f_l34_3468


namespace NUMINAMATH_CALUDE_min_value_expression_l34_3404

theorem min_value_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * ((a + b)⁻¹ + (b + c)⁻¹ + (c + d)⁻¹ + (a + d)⁻¹) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l34_3404


namespace NUMINAMATH_CALUDE_craft_supplies_ratio_l34_3406

/-- Represents the craft supplies bought by a person -/
structure CraftSupplies :=
  (glueSticks : ℕ)
  (constructionPaper : ℕ)

/-- The ratio of two natural numbers -/
def ratio (a b : ℕ) : ℚ := a / b

theorem craft_supplies_ratio :
  ∀ (allison marie : CraftSupplies),
    allison.glueSticks = marie.glueSticks + 8 →
    marie.glueSticks = 15 →
    marie.constructionPaper = 30 →
    allison.glueSticks + allison.constructionPaper = 28 →
    ratio marie.constructionPaper allison.constructionPaper = 6 := by
  sorry

end NUMINAMATH_CALUDE_craft_supplies_ratio_l34_3406


namespace NUMINAMATH_CALUDE_city_population_ratio_l34_3488

theorem city_population_ratio (x y z : ℕ) 
  (h1 : y = 2 * z)
  (h2 : x = 12 * z) :
  x / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l34_3488


namespace NUMINAMATH_CALUDE_estimate_theorem_l34_3417

/-- Represents a company with employees and their distance from workplace -/
structure Company where
  total_employees : ℕ
  sample_size : ℕ
  within_1000m : ℕ
  within_2000m : ℕ

/-- Calculates the estimated number of employees living between 1000 and 2000 meters -/
def estimate_between_1000_2000 (c : Company) : ℕ :=
  let sample_between := c.within_2000m - c.within_1000m
  (sample_between * c.total_employees) / c.sample_size

/-- Theorem stating the estimated number of employees living between 1000 and 2000 meters -/
theorem estimate_theorem (c : Company) 
  (h1 : c.total_employees = 2000)
  (h2 : c.sample_size = 200)
  (h3 : c.within_1000m = 10)
  (h4 : c.within_2000m = 30) :
  estimate_between_1000_2000 c = 200 := by
  sorry

#eval estimate_between_1000_2000 { total_employees := 2000, sample_size := 200, within_1000m := 10, within_2000m := 30 }

end NUMINAMATH_CALUDE_estimate_theorem_l34_3417


namespace NUMINAMATH_CALUDE_passing_marks_l34_3421

/-- The passing marks problem -/
theorem passing_marks (T P : ℝ) 
  (h1 : 0.40 * T = P - 40)
  (h2 : 0.60 * T = P + 20)
  (h3 : 0.45 * T = P - 10) : 
  P = 160 := by sorry

end NUMINAMATH_CALUDE_passing_marks_l34_3421


namespace NUMINAMATH_CALUDE_parabola_shift_down_2_l34_3487

/-- The equation of a parabola after vertical shift -/
def shifted_parabola (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + b

/-- Theorem: Shifting y = x^2 down by 2 units results in y = x^2 - 2 -/
theorem parabola_shift_down_2 :
  shifted_parabola 1 (-2) = λ x => x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_down_2_l34_3487


namespace NUMINAMATH_CALUDE_goats_count_l34_3489

/-- Represents the number of animals on a farm --/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ
  chickens : ℕ
  ducks : ℕ

/-- Represents the conditions given in the problem --/
def farm_conditions (f : Farm) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows = f.goats + 4 ∧
  f.chickens = 3 * f.pigs ∧
  f.ducks = (f.cows + f.goats) / 2 ∧
  f.goats + f.cows + f.pigs + f.chickens + f.ducks = 172

/-- The theorem to be proved --/
theorem goats_count (f : Farm) (h : farm_conditions f) : f.goats = 12 := by
  sorry


end NUMINAMATH_CALUDE_goats_count_l34_3489


namespace NUMINAMATH_CALUDE_reverse_geometric_difference_l34_3465

/-- A 3-digit number is reverse geometric if it has 3 distinct digits which,
    when read from right to left, form a geometric sequence. -/
def is_reverse_geometric (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ) (r : ℚ),
    n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    0 < r ∧
    (b : ℚ) = c * r ∧
    (a : ℚ) = b * r

def largest_reverse_geometric : ℕ := sorry

def smallest_reverse_geometric : ℕ := sorry

theorem reverse_geometric_difference :
  largest_reverse_geometric - smallest_reverse_geometric = 789 :=
sorry

end NUMINAMATH_CALUDE_reverse_geometric_difference_l34_3465


namespace NUMINAMATH_CALUDE_flour_to_add_correct_l34_3407

/-- Represents the recipe and baking constraints -/
structure BakingProblem where
  total_flour : ℝ  -- Total flour required by the recipe
  total_sugar : ℝ  -- Total sugar required by the recipe
  flour_sugar_diff : ℝ  -- Difference between remaining flour and sugar to be added

/-- Calculates the amount of flour that needs to be added -/
def flour_to_add (problem : BakingProblem) : ℝ :=
  problem.total_flour

/-- Theorem stating that the amount of flour to add is correct -/
theorem flour_to_add_correct (problem : BakingProblem) 
  (h1 : problem.total_flour = 6)
  (h2 : problem.total_sugar = 13)
  (h3 : problem.flour_sugar_diff = 8) :
  flour_to_add problem = 6 ∧ 
  flour_to_add problem = problem.total_sugar - problem.flour_sugar_diff + problem.flour_sugar_diff := by
  sorry

#eval flour_to_add { total_flour := 6, total_sugar := 13, flour_sugar_diff := 8 }

end NUMINAMATH_CALUDE_flour_to_add_correct_l34_3407


namespace NUMINAMATH_CALUDE_quadratic_function_m_range_l34_3410

theorem quadratic_function_m_range (a c m : ℝ) :
  let f := fun x : ℝ => a * x^2 - 2 * a * x + c
  (∀ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, x < y → f x > f y) →
  f m ≤ f 0 →
  m ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_range_l34_3410


namespace NUMINAMATH_CALUDE_f_of_i_eq_zero_l34_3461

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the function f
def f (x : ℂ) : ℂ := x^3 - x^2 + x - 1

-- Theorem statement
theorem f_of_i_eq_zero : f i = 0 := by sorry

end NUMINAMATH_CALUDE_f_of_i_eq_zero_l34_3461


namespace NUMINAMATH_CALUDE_unique_stamp_solution_l34_3485

/-- Given a positive integer n, returns true if 120 cents is the greatest
    postage that cannot be formed using stamps of 9, n, and n+2 cents -/
def is_valid_stamp_set (n : ℕ+) : Prop :=
  (∀ k : ℕ, k ≤ 120 → ¬∃ a b c : ℕ, 9*a + n*b + (n+2)*c = k) ∧
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, 9*a + n*b + (n+2)*c = k)

/-- The only positive integer n that satisfies the stamp condition is 17 -/
theorem unique_stamp_solution :
  ∃! n : ℕ+, is_valid_stamp_set n ∧ n = 17 :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_solution_l34_3485


namespace NUMINAMATH_CALUDE_total_cats_is_twenty_l34_3432

-- Define the number of cats for each person
def jamie_persian : Nat := 4
def jamie_maine_coon : Nat := 2
def gordon_persian : Nat := jamie_persian / 2
def gordon_maine_coon : Nat := jamie_maine_coon + 1
def hawkeye_persian : Nat := 0
def hawkeye_maine_coon : Nat := gordon_maine_coon - 1
def natasha_persian : Nat := 3
def natasha_maine_coon : Nat := 4

-- Define the total number of cats
def total_cats : Nat :=
  jamie_persian + jamie_maine_coon +
  gordon_persian + gordon_maine_coon +
  hawkeye_persian + hawkeye_maine_coon +
  natasha_persian + natasha_maine_coon

-- Theorem to prove
theorem total_cats_is_twenty : total_cats = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_is_twenty_l34_3432


namespace NUMINAMATH_CALUDE_cake_angle_theorem_l34_3451

theorem cake_angle_theorem (n : ℕ) (initial_angle : ℝ) (final_angle : ℝ) : 
  n = 10 →
  initial_angle = 360 / n →
  final_angle = 360 / (n - 1) →
  final_angle - initial_angle = 4 := by
  sorry

end NUMINAMATH_CALUDE_cake_angle_theorem_l34_3451


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l34_3419

theorem solve_system_of_equations (x y : ℤ) 
  (h1 : x + y = 250) 
  (h2 : x - y = 200) : 
  y = 25 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l34_3419


namespace NUMINAMATH_CALUDE_special_function_properties_l34_3456

/-- A function satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  property : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y
  zero_map : f 0 = 0
  pi_half_map : f (Real.pi / 2) = 1

/-- The function is odd -/
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function is periodic with period 2π -/
def is_periodic_2pi (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x

/-- Main theorem: The special function is odd and periodic with period 2π -/
theorem special_function_properties (sf : SpecialFunction) :
    is_odd sf.f ∧ is_periodic_2pi sf.f := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l34_3456


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_3_l34_3447

/-- A function that returns true if a four-digit number of the form 258n is divisible by 3 -/
def isDivisibleBy3 (n : Nat) : Prop :=
  n ≥ 0 ∧ n ≤ 9 ∧ (2580 + n) % 3 = 0

/-- Theorem stating that a four-digit number 258n is divisible by 3 iff n is 0, 3, 6, or 9 -/
theorem four_digit_divisible_by_3 :
  ∀ n : Nat, isDivisibleBy3 n ↔ n = 0 ∨ n = 3 ∨ n = 6 ∨ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_3_l34_3447


namespace NUMINAMATH_CALUDE_same_terminal_side_diff_multiple_360_l34_3467

/-- Two angles have the same terminal side -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

/-- Theorem: If two angles have the same terminal side, their difference is a multiple of 360° -/
theorem same_terminal_side_diff_multiple_360 (α β : ℝ) :
  same_terminal_side α β → ∃ k : ℤ, α - β = k * 360 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_diff_multiple_360_l34_3467


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l34_3453

theorem inscribed_circle_radius_right_triangle 
  (a b c r : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_inscribed : r > 0 ∧ r * (a + b + c) = a * b) : 
  r = (a + b - c) / 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l34_3453


namespace NUMINAMATH_CALUDE_polynomial_simplification_l34_3444

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l34_3444


namespace NUMINAMATH_CALUDE_smoothie_combinations_l34_3483

theorem smoothie_combinations : 
  let num_flavors : ℕ := 5
  let num_toppings : ℕ := 8
  let topping_choices : ℕ := 3
  num_flavors * (Nat.choose num_toppings topping_choices) = 280 :=
by sorry

end NUMINAMATH_CALUDE_smoothie_combinations_l34_3483


namespace NUMINAMATH_CALUDE_preceding_binary_l34_3455

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (binary : List Bool) : ℕ :=
  binary.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

theorem preceding_binary (N : ℕ) (h : binaryToNat [true, true, false, false, false] = N) :
  natToBinary (N - 1) = [true, false, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_preceding_binary_l34_3455


namespace NUMINAMATH_CALUDE_investment_triple_period_l34_3490

/-- The annual interest rate as a real number -/
def r : ℝ := 0.341

/-- The condition for the investment to more than triple -/
def triple_condition (t : ℝ) : Prop := (1 + r) ^ t > 3

/-- The smallest investment period in years -/
def smallest_period : ℕ := 4

theorem investment_triple_period :
  (∀ t : ℝ, t < smallest_period → ¬(triple_condition t)) ∧
  (triple_condition (smallest_period : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_investment_triple_period_l34_3490


namespace NUMINAMATH_CALUDE_harrison_croissant_expenditure_l34_3400

/-- Calculates the total annual expenditure on croissants for Harrison --/
def annual_croissant_expenditure (
  regular_price : ℚ)
  (almond_price : ℚ)
  (chocolate_price : ℚ)
  (ham_cheese_price : ℚ)
  (weeks_in_year : ℕ)
  (ham_cheese_frequency : ℕ) : ℚ :=
  regular_price * weeks_in_year +
  almond_price * weeks_in_year +
  chocolate_price * weeks_in_year +
  ham_cheese_price * (weeks_in_year / ham_cheese_frequency)

theorem harrison_croissant_expenditure :
  annual_croissant_expenditure 3.5 5.5 4.5 6 52 2 = 858 :=
by sorry

end NUMINAMATH_CALUDE_harrison_croissant_expenditure_l34_3400


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_point_l34_3428

def danny_time : ℝ := 29

theorem time_difference_to_halfway_point :
  let steve_time := 2 * danny_time
  let danny_halfway_time := danny_time / 2
  let steve_halfway_time := steve_time / 2
  steve_halfway_time - danny_halfway_time = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_point_l34_3428


namespace NUMINAMATH_CALUDE_cos_inequality_range_l34_3438

theorem cos_inequality_range (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (Real.cos x ≤ 1/2 ↔ x ∈ Set.Icc (Real.pi/3) (5*Real.pi/3)) := by
sorry

end NUMINAMATH_CALUDE_cos_inequality_range_l34_3438


namespace NUMINAMATH_CALUDE_chips_on_line_after_moves_l34_3476

/-- Represents a configuration of chips on a plane -/
structure ChipConfiguration where
  num_chips : ℕ
  num_lines : ℕ

/-- Represents a move that can be applied to a chip configuration -/
def apply_move (config : ChipConfiguration) : ChipConfiguration :=
  { num_chips := config.num_chips,
    num_lines := min config.num_lines (2 ^ (config.num_lines - 1)) }

/-- Represents the initial configuration of chips on a convex 2000-gon -/
def initial_config : ChipConfiguration :=
  { num_chips := 2000,
    num_lines := 2000 }

/-- Applies n moves to the initial configuration -/
def apply_n_moves (n : ℕ) : ChipConfiguration :=
  (List.range n).foldl (λ config _ => apply_move config) initial_config

theorem chips_on_line_after_moves :
  (∀ n : ℕ, n ≤ 9 → (apply_n_moves n).num_lines > 1) ∧
  ∃ m : ℕ, m = 10 ∧ (apply_n_moves m).num_lines = 1 :=
sorry

end NUMINAMATH_CALUDE_chips_on_line_after_moves_l34_3476


namespace NUMINAMATH_CALUDE_temperature_conversion_fraction_l34_3464

theorem temperature_conversion_fraction : 
  ∀ (t k : ℝ) (fraction : ℝ),
    t = fraction * (k - 32) →
    (t = 20 ∧ k = 68) →
    fraction = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_fraction_l34_3464


namespace NUMINAMATH_CALUDE_inequality_equivalence_l34_3492

theorem inequality_equivalence (x : ℝ) : (x - 1) / 2 + 1 < (4 * x - 5) / 3 ↔ x > 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l34_3492


namespace NUMINAMATH_CALUDE_base8_253_to_base10_l34_3498

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds * 8^2 + tens * 8^1 + units * 8^0

-- Theorem statement
theorem base8_253_to_base10 : base8ToBase10 253 = 171 := by
  sorry

end NUMINAMATH_CALUDE_base8_253_to_base10_l34_3498


namespace NUMINAMATH_CALUDE_cattle_purchase_cost_l34_3415

theorem cattle_purchase_cost 
  (num_cattle : ℕ) 
  (feeding_cost_ratio : ℝ) 
  (weight_per_cattle : ℝ) 
  (selling_price_per_pound : ℝ) 
  (profit : ℝ) 
  (h1 : num_cattle = 100)
  (h2 : feeding_cost_ratio = 1.2)
  (h3 : weight_per_cattle = 1000)
  (h4 : selling_price_per_pound = 2)
  (h5 : profit = 112000) : 
  ∃ (purchase_cost : ℝ), purchase_cost = 40000 ∧ 
    num_cattle * weight_per_cattle * selling_price_per_pound - 
    (purchase_cost * (1 + (feeding_cost_ratio - 1))) = profit :=
by sorry

end NUMINAMATH_CALUDE_cattle_purchase_cost_l34_3415


namespace NUMINAMATH_CALUDE_perfect_square_partition_l34_3403

/-- A partition of a set of natural numbers -/
def Partition (n : ℕ) := (Fin n → Bool)

/-- Predicate to check if a sum is a perfect square -/
def IsPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

/-- Predicate to check if a partition contains two distinct numbers in one subset that sum to a perfect square -/
def HasPerfectSquarePair (n : ℕ) (p : Partition n) : Prop :=
  ∃ (i j : Fin n), i ≠ j ∧ p i = p j ∧ IsPerfectSquare (i.val + j.val + 2)

/-- The main theorem: for all n ≥ 15, any partition of {1, ..., n} contains a perfect square pair -/
theorem perfect_square_partition (n : ℕ) (hn : n ≥ 15) :
  ∀ p : Partition n, HasPerfectSquarePair n p :=
sorry

end NUMINAMATH_CALUDE_perfect_square_partition_l34_3403


namespace NUMINAMATH_CALUDE_smallest_possible_d_l34_3496

theorem smallest_possible_d : ∃ d : ℝ,
  (∀ d' : ℝ, d' ≥ 0 → (4 * Real.sqrt 3) ^ 2 + (d' - 2) ^ 2 = (4 * d') ^ 2 → d ≤ d') ∧
  (4 * Real.sqrt 3) ^ 2 + (d - 2) ^ 2 = (4 * d) ^ 2 ∧
  d = 26 / 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l34_3496


namespace NUMINAMATH_CALUDE_height_of_A_l34_3439

/-- The heights of four people A, B, C, and D satisfying certain conditions -/
structure Heights where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  sum_equality : A + B = C + D ∨ A + C = B + D ∨ A + D = B + C
  average_difference : (A + B) / 2 = (A + C) / 2 + 4
  D_taller : D = A + 10
  B_C_sum : B + C = 288

/-- The height of A is 139 cm -/
theorem height_of_A (h : Heights) : h.A = 139 := by
  sorry

end NUMINAMATH_CALUDE_height_of_A_l34_3439


namespace NUMINAMATH_CALUDE_abc_inequality_l34_3457

theorem abc_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1000)
  (h_sum : b * c * (1 - a) + a * (b + c) = 110) (h_a_lt_1 : a < 1) :
  10 < c ∧ c < 100 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l34_3457


namespace NUMINAMATH_CALUDE_line_intersects_circle_l34_3426

/-- The line 4x - 3y = 0 intersects the circle x^2 + y^2 = 36 -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), 4 * x - 3 * y = 0 ∧ x^2 + y^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l34_3426


namespace NUMINAMATH_CALUDE_intersection_at_one_point_l34_3435

theorem intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 2*x + 2 = -2*x - 2) ↔ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_at_one_point_l34_3435


namespace NUMINAMATH_CALUDE_roses_cut_theorem_l34_3475

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem: The number of roses Jessica cut is equal to the difference between
    the final number of roses and the initial number of roses in the vase -/
theorem roses_cut_theorem (initial_roses final_roses : ℕ) 
  (h : final_roses ≥ initial_roses) :
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

/-- Given the initial and final number of roses in the vase,
    calculate the number of roses Jessica cut -/
def calculate_roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  roses_cut initial_roses final_roses

#eval calculate_roses_cut 2 23  -- Should output 21

end NUMINAMATH_CALUDE_roses_cut_theorem_l34_3475


namespace NUMINAMATH_CALUDE_quadratic_intersection_points_l34_3427

/-- A quadratic function with at least one y-intercept -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  has_y_intercept : ∃ x, a * x^2 + b * x + c = 0

/-- The number of intersection points between f(x) and -f(-x) -/
def intersection_points_f_u (f : QuadraticFunction) : ℕ := 1

/-- The number of intersection points between f(x) and f(x+1) -/
def intersection_points_f_v (f : QuadraticFunction) : ℕ := 0

/-- The main theorem -/
theorem quadratic_intersection_points (f : QuadraticFunction) :
  7 * (intersection_points_f_u f) + 3 * (intersection_points_f_v f) = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_points_l34_3427


namespace NUMINAMATH_CALUDE_gcd_problem_l34_3422

theorem gcd_problem (n : ℕ) : 
  75 ≤ n ∧ n ≤ 90 ∧ Nat.gcd n 15 = 5 → n = 80 ∨ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l34_3422


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_max_sum_of_cubes_attained_l34_3401

theorem max_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 9) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 27 :=
by sorry

theorem max_sum_of_cubes_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 9 ∧ 
  a^3 + b^3 + c^3 + d^3 + e^3 > 27 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_max_sum_of_cubes_attained_l34_3401


namespace NUMINAMATH_CALUDE_petyas_friends_count_l34_3425

/-- The number of friends Petya has -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

theorem petyas_friends_count :
  (num_friends * 5 + 8 = total_stickers) ∧
  (num_friends * 6 = total_stickers + 11) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_count_l34_3425


namespace NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_l34_3463

theorem prime_arithmetic_progression_difference (a : ℕ → ℕ) (d : ℕ) :
  (∀ k, k ∈ Finset.range 15 → Nat.Prime (a k)) →
  (∀ k, k ∈ Finset.range 14 → a (k + 1) = a k + d) →
  (∀ k l, k < l → k ∈ Finset.range 15 → l ∈ Finset.range 15 → a k < a l) →
  d > 30000 := by
  sorry

end NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_l34_3463


namespace NUMINAMATH_CALUDE_digit_multiplication_problem_l34_3420

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if all elements in a list are different -/
def all_different (list : List Digit) : Prop :=
  ∀ i j, i ≠ j → list.get i ≠ list.get j

/-- Converts a three-digit number represented by digits to a natural number -/
def to_nat_3digit (a b c : Digit) : ℕ :=
  100 * a.val + 10 * b.val + c.val

/-- Converts a two-digit number represented by digits to a natural number -/
def to_nat_2digit (d e : Digit) : ℕ :=
  10 * d.val + e.val

/-- Converts a four-digit number represented by digits to a natural number -/
def to_nat_4digit (d1 d2 e1 e2 : Digit) : ℕ :=
  1000 * d1.val + 100 * d2.val + 10 * e1.val + e2.val

theorem digit_multiplication_problem (A B C D E : Digit) :
  all_different [A, B, C, D, E] →
  to_nat_3digit A B C * to_nat_2digit D E = to_nat_4digit D D E E →
  A.val + B.val = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_problem_l34_3420


namespace NUMINAMATH_CALUDE_tens_digit_of_6_to_2050_l34_3436

theorem tens_digit_of_6_to_2050 : 6^2050 % 100 = 56 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_to_2050_l34_3436


namespace NUMINAMATH_CALUDE_pen_cost_l34_3469

theorem pen_cost (notebook pen case : ℝ) 
  (total_cost : notebook + pen + case = 3.50)
  (pen_triple : pen = 3 * notebook)
  (case_more : case = notebook + 0.50) :
  pen = 1.80 := by
  sorry

end NUMINAMATH_CALUDE_pen_cost_l34_3469


namespace NUMINAMATH_CALUDE_sum_to_term_ratio_l34_3408

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  h1 : a 5 - a 3 = 12
  h2 : a 6 - a 4 = 24

/-- The sum of the first n terms of a geometric sequence -/
def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the ratio of sum to nth term -/
theorem sum_to_term_ratio (seq : GeometricSequence) (n : ℕ) :
  sum_n seq n / seq.a n = 2 - 2^(1 - n) :=
sorry

end NUMINAMATH_CALUDE_sum_to_term_ratio_l34_3408


namespace NUMINAMATH_CALUDE_julio_lost_fish_l34_3423

/-- Proves that Julio lost 15 fish given the fishing conditions -/
theorem julio_lost_fish (fish_per_hour : ℕ) (fishing_hours : ℕ) (final_fish_count : ℕ) : 
  fish_per_hour = 7 →
  fishing_hours = 9 →
  final_fish_count = 48 →
  fish_per_hour * fishing_hours - final_fish_count = 15 := by
sorry

end NUMINAMATH_CALUDE_julio_lost_fish_l34_3423


namespace NUMINAMATH_CALUDE_circumscribed_trapezoid_inequality_l34_3433

/-- A trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- Radius of the inscribed circle -/
  R : ℝ
  /-- Length of one base of the trapezoid -/
  a : ℝ
  /-- Length of the other base of the trapezoid -/
  b : ℝ
  /-- The trapezoid is circumscribed around the circle -/
  circumscribed : True

/-- For a trapezoid circumscribed around a circle with radius R and bases a and b, ab ≥ 4R^2 -/
theorem circumscribed_trapezoid_inequality (t : CircumscribedTrapezoid) : t.a * t.b ≥ 4 * t.R^2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_trapezoid_inequality_l34_3433


namespace NUMINAMATH_CALUDE_cube_root_eight_plus_negative_two_power_zero_l34_3494

theorem cube_root_eight_plus_negative_two_power_zero : 
  (8 : ℝ) ^ (1/3) + (-2 : ℝ) ^ 0 = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_plus_negative_two_power_zero_l34_3494


namespace NUMINAMATH_CALUDE_sine_inequality_l34_3441

theorem sine_inequality (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < 1) :
  0 < Real.sin x ∧ Real.sin x < Real.sin y := by sorry

end NUMINAMATH_CALUDE_sine_inequality_l34_3441


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l34_3479

theorem trigonometric_expression_evaluation :
  (Real.tan (150 * π / 180)) * (Real.cos (-210 * π / 180)) * (Real.sin (-420 * π / 180)) /
  ((Real.sin (1050 * π / 180)) * (Real.cos (-600 * π / 180))) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l34_3479


namespace NUMINAMATH_CALUDE_candy_distribution_l34_3459

/-- Proves that given the candy distribution conditions, the total number of children is 40 -/
theorem candy_distribution (total_candies : ℕ) (boys girls : ℕ) : 
  total_candies = 90 →
  total_candies / 3 = boys * 3 →
  2 * total_candies / 3 = girls * 2 →
  boys + girls = 40 := by
  sorry

#check candy_distribution

end NUMINAMATH_CALUDE_candy_distribution_l34_3459


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_lowest_terms_l34_3411

/-- The repeating decimal 0.4̄37 as a real number -/
def repeating_decimal : ℚ := 433 / 990

theorem repeating_decimal_equiv : repeating_decimal = 0.4 + (37 / 990) := by sorry

theorem fraction_lowest_terms : ∀ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 → (433 * a = 990 * b) → (a = 990 ∧ b = 433) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_lowest_terms_l34_3411


namespace NUMINAMATH_CALUDE_water_saving_calculation_l34_3473

/-- The amount of water Hyunwoo's family uses daily in liters -/
def daily_water_usage : ℝ := 215

/-- The fraction of water saved when adjusting the water pressure valve weakly -/
def water_saving_fraction : ℝ := 0.32

/-- The amount of water saved when adjusting the water pressure valve weakly -/
def water_saved : ℝ := daily_water_usage * water_saving_fraction

theorem water_saving_calculation :
  water_saved = 68.8 := by sorry

end NUMINAMATH_CALUDE_water_saving_calculation_l34_3473


namespace NUMINAMATH_CALUDE_cafeteria_pies_l34_3478

/-- Given a cafeteria with initial apples, apples handed out, and apples required per pie,
    calculates the number of pies that can be made with the remaining apples. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Proves that given 86 initial apples, after handing out 30 apples,
    and using 8 apples per pie, the number of pies that can be made is 7. -/
theorem cafeteria_pies :
  calculate_pies 86 30 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l34_3478


namespace NUMINAMATH_CALUDE_dani_pants_per_pair_l34_3405

/-- Calculates the number of pants in each pair given the initial number of pants,
    the number of pants after a certain number of years, the number of pairs received each year,
    and the number of years. -/
def pants_per_pair (initial_pants : ℕ) (final_pants : ℕ) (pairs_per_year : ℕ) (years : ℕ) : ℕ :=
  let total_pairs := pairs_per_year * years
  let total_new_pants := final_pants - initial_pants
  total_new_pants / total_pairs

theorem dani_pants_per_pair :
  pants_per_pair 50 90 4 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_per_pair_l34_3405


namespace NUMINAMATH_CALUDE_units_digit_of_n_l34_3431

def units_digit (a : ℕ) : ℕ := a % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : units_digit m = 6) :
  units_digit n = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l34_3431


namespace NUMINAMATH_CALUDE_max_additional_plates_l34_3497

/-- Represents the number of letters in each set for license plates -/
structure LicensePlateSets :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Calculates the total number of unique license plates -/
def totalPlates (sets : LicensePlateSets) : ℕ :=
  sets.first * sets.second * sets.third

/-- The initial configuration of letter sets -/
def initialSets : LicensePlateSets :=
  ⟨5, 3, 4⟩

/-- The number of new letters to be added -/
def newLetters : ℕ := 2

/-- Theorem: The maximum number of additional unique license plates is 30 -/
theorem max_additional_plates :
  ∃ (newSets : LicensePlateSets),
    (newSets.first + newSets.second + newSets.third = initialSets.first + initialSets.second + initialSets.third + newLetters) ∧
    (∀ (otherSets : LicensePlateSets),
      (otherSets.first + otherSets.second + otherSets.third = initialSets.first + initialSets.second + initialSets.third + newLetters) →
      (totalPlates newSets - totalPlates initialSets ≥ totalPlates otherSets - totalPlates initialSets)) ∧
    (totalPlates newSets - totalPlates initialSets = 30) :=
  sorry

end NUMINAMATH_CALUDE_max_additional_plates_l34_3497


namespace NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l34_3462

/-- The maximum number of parts that three planes can divide space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts that three planes can divide space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l34_3462


namespace NUMINAMATH_CALUDE_solve_equations_l34_3409

theorem solve_equations :
  (∃ x : ℝ, 5 * x - 2.9 = 12) ∧
  (∃ x : ℝ, 10.5 * x + 0.6 * x = 44) ∧
  (∃ x : ℝ, 8 * x / 2 = 1.5) :=
by
  constructor
  · use 1.82
    sorry
  constructor
  · use 3
    sorry
  · use 0.375
    sorry

end NUMINAMATH_CALUDE_solve_equations_l34_3409


namespace NUMINAMATH_CALUDE_history_score_is_84_percent_l34_3440

/-- Given a student's scores in math and a third subject, along with a desired overall average,
    this function calculates the required score in history. -/
def calculate_history_score (math_score : ℚ) (third_subject_score : ℚ) (desired_average : ℚ) : ℚ :=
  3 * desired_average - math_score - third_subject_score

/-- Theorem stating that given the specific scores and desired average,
    the calculated history score is 84%. -/
theorem history_score_is_84_percent :
  calculate_history_score 72 69 75 = 84 := by
  sorry

#eval calculate_history_score 72 69 75

end NUMINAMATH_CALUDE_history_score_is_84_percent_l34_3440


namespace NUMINAMATH_CALUDE_updated_mean_after_correction_l34_3450

theorem updated_mean_after_correction (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n = 100 →
  original_mean = 350 →
  decrement = 63 →
  (n : ℝ) * original_mean + n * decrement = n * 413 := by
sorry

end NUMINAMATH_CALUDE_updated_mean_after_correction_l34_3450


namespace NUMINAMATH_CALUDE_balls_to_boxes_count_l34_3470

/-- The number of ways to distribute n indistinguishable objects into k distinguishable groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute 7 indistinguishable balls into 3 distinguishable boxes,
    with each box containing at least one ball -/
def distribute_balls_to_boxes : ℕ := distribute 4 3

theorem balls_to_boxes_count :
  distribute_balls_to_boxes = 15 := by sorry

end NUMINAMATH_CALUDE_balls_to_boxes_count_l34_3470


namespace NUMINAMATH_CALUDE_g_of_3_l34_3448

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem g_of_3 : g 3 = 79 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l34_3448


namespace NUMINAMATH_CALUDE_floor_sqrt_24_squared_l34_3402

theorem floor_sqrt_24_squared : ⌊Real.sqrt 24⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_24_squared_l34_3402


namespace NUMINAMATH_CALUDE_puppies_per_cage_l34_3429

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 13)
  (h2 : sold_puppies = 7)
  (h3 : num_cages = 3)
  (h4 : num_cages > 0)
  (h5 : initial_puppies > sold_puppies) :
  (initial_puppies - sold_puppies) / num_cages = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_cage_l34_3429


namespace NUMINAMATH_CALUDE_farmer_children_count_l34_3481

theorem farmer_children_count : ∃ n : ℕ,
  (n ≠ 0) ∧
  (15 * n - 8 - 7 = 60) ∧
  (n = 5) := by
  sorry

end NUMINAMATH_CALUDE_farmer_children_count_l34_3481


namespace NUMINAMATH_CALUDE_problem_statement_l34_3437

theorem problem_statement (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 3) :
  x * Real.sqrt (y / x) + y * Real.sqrt (x / y) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l34_3437


namespace NUMINAMATH_CALUDE_rod_weight_10m_l34_3472

/-- Represents the weight of a rod given its length -/
def rod_weight (length : ℝ) : ℝ := sorry

/-- The constant of proportionality between rod length and weight -/
def weight_per_meter : ℝ := sorry

theorem rod_weight_10m (h1 : rod_weight 6 = 14.04) 
  (h2 : ∀ l : ℝ, rod_weight l = weight_per_meter * l) : 
  rod_weight 10 = 23.4 := by sorry

end NUMINAMATH_CALUDE_rod_weight_10m_l34_3472


namespace NUMINAMATH_CALUDE_car_speed_problem_l34_3484

theorem car_speed_problem (D : ℝ) (h : D > 0) :
  let t1 := D / 3 / 80
  let t2 := D / 3 / 30
  let t3 := D / 3 / 48
  45 = D / (t1 + t2 + t3) :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l34_3484


namespace NUMINAMATH_CALUDE_power_sum_problem_l34_3424

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 26)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^6 + b * y^6 = -220 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l34_3424
