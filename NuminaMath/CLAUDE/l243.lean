import Mathlib

namespace correct_seating_arrangements_l243_24339

def number_of_people : ℕ := 8

-- Define a function to calculate the number of seating arrangements
def seating_arrangements (n : ℕ) (restricted_pair : ℕ) : ℕ :=
  Nat.factorial n - Nat.factorial (n - 1) * restricted_pair

-- Theorem statement
theorem correct_seating_arrangements :
  seating_arrangements number_of_people 2 = 30240 := by
  sorry

end correct_seating_arrangements_l243_24339


namespace uranus_appearance_time_l243_24341

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def add_minutes (t : Time) (m : ℕ) : Time :=
  sorry

/-- Calculates the difference in minutes between two times -/
def minutes_difference (t1 t2 : Time) : ℕ :=
  sorry

theorem uranus_appearance_time 
  (mars_disappearance : Time)
  (jupiter_delay : ℕ)
  (uranus_delay : ℕ)
  (h_mars : mars_disappearance = ⟨0, 10, sorry, sorry⟩)  -- 12:10 AM
  (h_jupiter : jupiter_delay = 2 * 60 + 41)  -- 2 hours and 41 minutes
  (h_uranus : uranus_delay = 3 * 60 + 16)  -- 3 hours and 16 minutes
  : 
  let jupiter_appearance := add_minutes mars_disappearance jupiter_delay
  let uranus_appearance := add_minutes jupiter_appearance uranus_delay
  minutes_difference ⟨6, 0, sorry, sorry⟩ uranus_appearance = 7 :=
sorry

end uranus_appearance_time_l243_24341


namespace estimate_negative_sqrt_17_l243_24394

theorem estimate_negative_sqrt_17 : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end estimate_negative_sqrt_17_l243_24394


namespace batsman_average_increase_l243_24314

/-- Represents a batsman's performance -/
structure Batsman where
  total_runs_before_16th : ℕ
  runs_in_16th : ℕ
  average_after_16th : ℚ

/-- Calculates the increase in average for a batsman -/
def average_increase (b : Batsman) : ℚ :=
  b.average_after_16th - (b.total_runs_before_16th : ℚ) / 15

/-- Theorem: The increase in average is 3 for a batsman who scores 64 runs in the 16th inning
    and has an average of 19 after the 16th inning -/
theorem batsman_average_increase
  (b : Batsman)
  (h1 : b.runs_in_16th = 64)
  (h2 : b.average_after_16th = 19)
  (h3 : b.total_runs_before_16th + b.runs_in_16th = 16 * b.average_after_16th) :
  average_increase b = 3 := by
  sorry


end batsman_average_increase_l243_24314


namespace collinearity_condition_for_linear_combination_l243_24333

/-- Given points O, A, B are not collinear, and vector OP = m * vector OA + n * vector OB,
    points A, P, B are collinear if and only if m + n = 1 -/
theorem collinearity_condition_for_linear_combination
  (O A B P : EuclideanSpace ℝ (Fin 3))
  (m n : ℝ)
  (h_not_collinear : ¬ Collinear ℝ {O, A, B})
  (h_linear_combination : P - O = m • (A - O) + n • (B - O)) :
  Collinear ℝ {A, P, B} ↔ m + n = 1 := by sorry

end collinearity_condition_for_linear_combination_l243_24333


namespace equation_solutions_l243_24384

theorem equation_solutions :
  (∃ x : ℝ, x = 1/3 ∧ 3/(1-6*x) = 2/(6*x+1) - (8+9*x)/(36*x^2-1)) ∧
  (∃ z : ℝ, z = -3/7 ∧ 3/(1-z^2) = 2/((1+z)^2) - 5/((1-z)^2)) :=
by sorry

end equation_solutions_l243_24384


namespace binomial_expansion_properties_l243_24308

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of the first three terms of the binomial expansion -/
def first_three_sum (n : ℕ) : ℕ := binomial n 0 + binomial n 1 + binomial n 2

/-- The constant term in the expansion -/
def constant_term (n : ℕ) : ℤ := binomial n 4 * (-2)^4

/-- The coefficient with the largest absolute value in the expansion -/
def largest_coeff (n : ℕ) : ℤ := binomial n 8 * 2^8

theorem binomial_expansion_properties :
  ∃ n : ℕ, 
    first_three_sum n = 79 ∧ 
    constant_term n = 7920 ∧ 
    largest_coeff n = 126720 := by sorry

end binomial_expansion_properties_l243_24308


namespace rectangle_triangle_configuration_l243_24353

theorem rectangle_triangle_configuration (AB AD : ℝ) (h1 : AB = 8) (h2 : AD = 10) : ∃ (DE : ℝ),
  let ABCD_area := AB * AD
  let DCE_area := ABCD_area / 2
  let DC := AD
  let CE := 2 * DCE_area / DC
  DE^2 = DC^2 + CE^2 ∧ DE = 2 * Real.sqrt 41 := by sorry

end rectangle_triangle_configuration_l243_24353


namespace joans_initial_balloons_count_l243_24347

/-- The number of blue balloons Joan had initially -/
def joans_initial_balloons : ℕ := 9

/-- The number of balloons Sally popped -/
def popped_balloons : ℕ := 5

/-- The number of blue balloons Jessica has -/
def jessicas_balloons : ℕ := 2

/-- The total number of blue balloons they have now -/
def total_balloons_now : ℕ := 6

theorem joans_initial_balloons_count : 
  joans_initial_balloons = popped_balloons + (total_balloons_now - jessicas_balloons) :=
by sorry

end joans_initial_balloons_count_l243_24347


namespace correct_set_representations_l243_24334

-- Define the sets
def RealNumbers : Type := Real
def NaturalNumbers : Type := Nat
def Integers : Type := Int
def RationalNumbers : Type := Rat

-- State the theorem
theorem correct_set_representations :
  (RealNumbers = ℝ) ∧
  (NaturalNumbers = ℕ) ∧
  (Integers = ℤ) ∧
  (RationalNumbers = ℚ) := by
  sorry

end correct_set_representations_l243_24334


namespace dress_price_l243_24311

theorem dress_price (total_revenue : ℕ) (num_dresses : ℕ) (num_shirts : ℕ) (shirt_price : ℕ) (dress_price : ℕ) :
  total_revenue = 69 →
  num_dresses = 7 →
  num_shirts = 4 →
  shirt_price = 5 →
  num_dresses * dress_price + num_shirts * shirt_price = total_revenue →
  dress_price = 7 := by
sorry

end dress_price_l243_24311


namespace max_teams_tied_for_most_wins_l243_24315

/-- Represents a round-robin tournament --/
structure Tournament where
  num_teams : Nat
  games : Fin num_teams → Fin num_teams → Bool
  
/-- Tournament conditions --/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 7 ∧
  (∀ i j, i ≠ j → (t.games i j ↔ ¬t.games j i)) ∧
  (∀ i, ¬t.games i i)

/-- Number of wins for a team --/
def wins (t : Tournament) (team : Fin t.num_teams) : Nat :=
  (Finset.univ.filter (λ j => t.games team j)).card

/-- Maximum number of wins in the tournament --/
def max_wins (t : Tournament) : Nat :=
  Finset.univ.sup (λ team => wins t team)

/-- Number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : Nat :=
  (Finset.univ.filter (λ team => wins t team = max_wins t)).card

/-- The main theorem --/
theorem max_teams_tied_for_most_wins (t : Tournament) 
  (h : valid_tournament t) : 
  num_teams_with_max_wins t ≤ 6 ∧ 
  ∃ t' : Tournament, valid_tournament t' ∧ num_teams_with_max_wins t' = 6 := by
  sorry


end max_teams_tied_for_most_wins_l243_24315


namespace secant_slope_positive_l243_24371

open Real

noncomputable def f (x : ℝ) : ℝ := 2^x + x^3

theorem secant_slope_positive (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f x₁ - f x₂) / (x₁ - x₂) > 0 :=
sorry

end secant_slope_positive_l243_24371


namespace f_properties_l243_24324

-- Define the function f
def f (x : ℝ) := -x^2 - 4*x + 1

-- Theorem statement
theorem f_properties :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≤ m ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = 5) ∧
  (∀ (x y : ℝ), x < y ∧ y < -2 → f x < f y) :=
sorry

end f_properties_l243_24324


namespace not_divisible_by_twelve_l243_24366

theorem not_divisible_by_twelve (m : ℕ) (h1 : m > 0) 
  (h2 : ∃ (j : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 9 + (1 : ℚ) / m = j) : 
  ¬(12 ∣ m) := by
sorry

end not_divisible_by_twelve_l243_24366


namespace square_of_binomial_l243_24367

theorem square_of_binomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 14*x + k = (x - a)^2) ↔ k = 49 := by
  sorry

end square_of_binomial_l243_24367


namespace greatest_root_of_g_l243_24304

def g (x : ℝ) := 10 * x^4 - 16 * x^2 + 3

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/5) ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end greatest_root_of_g_l243_24304


namespace rubber_bands_total_l243_24343

theorem rubber_bands_total (harper_bands : ℕ) (brother_difference : ℕ) : 
  harper_bands = 15 → 
  brother_difference = 6 → 
  harper_bands + (harper_bands - brother_difference) = 24 := by
sorry

end rubber_bands_total_l243_24343


namespace inequality_proof_l243_24374

/-- For any non-integer real number x > 1, the following inequality holds -/
theorem inequality_proof (x : ℝ) (h1 : x > 1) (h2 : ¬ ∃ n : ℤ, x = n) :
  let fx := x - ⌊x⌋
  ((x + fx) / ⌊x⌋ - ⌊x⌋ / (x + fx)) + ((x + ⌊x⌋) / fx - fx / (x + ⌊x⌋)) > 9/2 := by
sorry

end inequality_proof_l243_24374


namespace water_jars_theorem_l243_24323

theorem water_jars_theorem (S L : ℚ) (h1 : S > 0) (h2 : L > 0) (h3 : S ≠ L) : 
  (1/3 : ℚ) * S = (1/2 : ℚ) * L → (1/2 : ℚ) * L + (1/3 : ℚ) * S = L := by
  sorry

end water_jars_theorem_l243_24323


namespace glass_bowl_selling_price_l243_24309

theorem glass_bowl_selling_price
  (total_bowls : ℕ)
  (cost_per_bowl : ℚ)
  (sold_bowls : ℕ)
  (percentage_gain : ℚ)
  (h1 : total_bowls = 115)
  (h2 : cost_per_bowl = 18)
  (h3 : sold_bowls = 104)
  (h4 : percentage_gain = 0.004830917874396135)
  : ∃ (selling_price : ℚ), selling_price = 20 ∧ 
    selling_price * sold_bowls = cost_per_bowl * total_bowls * (1 + percentage_gain) :=
by sorry

end glass_bowl_selling_price_l243_24309


namespace quadratic_roots_condition_l243_24383

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x + 2 = 0 ∧ k * y^2 - 2 * y + 2 = 0) ↔ 
  (k < 1/2 ∧ k ≠ 0) := by
sorry

end quadratic_roots_condition_l243_24383


namespace range_of_x_l243_24356

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) :
  14 - 2 * Real.sqrt 13 ≤ x ∧ x ≤ 14 + 2 * Real.sqrt 13 := by
  sorry

end range_of_x_l243_24356


namespace orange_mango_difference_l243_24301

/-- Represents the total produce in kilograms for each fruit type -/
structure FruitProduce where
  mangoes : ℕ
  apples : ℕ
  oranges : ℕ

/-- Calculates the total revenue given the price per kg and total produce -/
def totalRevenue (price : ℕ) (produce : FruitProduce) : ℕ :=
  price * (produce.mangoes + produce.apples + produce.oranges)

/-- Theorem stating the difference between orange and mango produce -/
theorem orange_mango_difference (produce : FruitProduce) : 
  produce.mangoes = 400 →
  produce.apples = 2 * produce.mangoes →
  produce.oranges > produce.mangoes →
  totalRevenue 50 produce = 90000 →
  produce.oranges - produce.mangoes = 200 := by
sorry

end orange_mango_difference_l243_24301


namespace infinite_divisibility_equivalence_l243_24302

theorem infinite_divisibility_equivalence :
  ∀ (a b c : ℕ+),
  (∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ (n : ℕ+), n ∈ S → (a + n) ∣ (b + c * n!)) ↔
  (∃ (k : ℕ) (t : ℤ), a = 2 * k + 1 ∧ b = t.natAbs ∧ c = (t.natAbs * (2 * k).factorial)) :=
by sorry

end infinite_divisibility_equivalence_l243_24302


namespace quarters_remaining_l243_24300

/-- Calculates the number of quarters remaining after paying for a dress -/
theorem quarters_remaining (initial_quarters : ℕ) (dress_cost : ℚ) (quarter_value : ℚ) : 
  initial_quarters = 160 → 
  dress_cost = 35 → 
  quarter_value = 1/4 → 
  initial_quarters - (dress_cost / quarter_value).floor = 20 := by
sorry

end quarters_remaining_l243_24300


namespace monochromatic_triangle_exists_l243_24364

/-- A type representing the colors of segments -/
inductive Color
| Red
| Blue

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A function type representing the coloring of segments -/
def Coloring := (Point × Point) → Color

/-- Theorem: Given 6 points in a plane with all segments colored either red or blue,
    there exists a triangle whose sides are all the same color -/
theorem monochromatic_triangle_exists (points : Fin 6 → Point) (coloring : Coloring) :
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (coloring (points i, points j) = coloring (points j, points k) ∧
     coloring (points j, points k) = coloring (points k, points i)) :=
sorry

end monochromatic_triangle_exists_l243_24364


namespace prob_adjacent_vertices_decagon_l243_24306

/-- A decagon is a polygon with 10 sides and 10 vertices. -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon. -/
def num_vertices : ℕ := 10

/-- The number of vertices adjacent to any given vertex in a decagon. -/
def num_adjacent_vertices : ℕ := 2

/-- The probability of selecting two adjacent vertices when choosing 2 distinct vertices at random from a decagon. -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  num_adjacent_vertices / (num_vertices - 1)

theorem prob_adjacent_vertices_decagon :
  ∀ d : Decagon, prob_adjacent_vertices d = 2 / 9 := by
  sorry

end prob_adjacent_vertices_decagon_l243_24306


namespace binomial_coefficient_third_term_expansion_l243_24393

theorem binomial_coefficient_third_term_expansion (x : ℤ) :
  Nat.choose 5 2 = 10 := by
  sorry

end binomial_coefficient_third_term_expansion_l243_24393


namespace inequality_proof_l243_24387

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / c) + (a * c / b) + (b * c / a) ≥ a + b + c := by
  sorry

end inequality_proof_l243_24387


namespace village_male_population_l243_24317

/-- Represents the population of a village -/
structure Village where
  total_population : ℕ
  num_parts : ℕ
  male_parts : ℕ

/-- Calculates the number of males in the village -/
def num_males (v : Village) : ℕ :=
  v.total_population * v.male_parts / v.num_parts

theorem village_male_population (v : Village) 
  (h1 : v.total_population = 600)
  (h2 : v.num_parts = 4)
  (h3 : v.male_parts = 2) : 
  num_males v = 300 := by
  sorry

#check village_male_population

end village_male_population_l243_24317


namespace papaya_production_l243_24326

/-- The number of papaya trees -/
def papaya_trees : ℕ := 2

/-- The number of mango trees -/
def mango_trees : ℕ := 3

/-- The number of mangos each mango tree produces -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits -/
def total_fruits : ℕ := 80

/-- The number of papayas each papaya tree produces -/
def papayas_per_tree : ℕ := 10

theorem papaya_production :
  papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = total_fruits :=
by sorry

end papaya_production_l243_24326


namespace quadratic_sum_of_b_and_c_l243_24354

/-- Given a quadratic expression x^2 - 24x + 50, prove that when written in the form (x+b)^2 + c, b + c = -106 -/
theorem quadratic_sum_of_b_and_c : ∃ b c : ℝ, 
  (∀ x : ℝ, x^2 - 24*x + 50 = (x + b)^2 + c) ∧ 
  (b + c = -106) := by
sorry

end quadratic_sum_of_b_and_c_l243_24354


namespace arrangements_of_opening_rooms_l243_24331

theorem arrangements_of_opening_rooms (n : ℕ) (hn : n = 6) :
  (Finset.sum (Finset.range 5) (fun k => Nat.choose n (k + 2))) = (2^n - (n + 1)) :=
sorry

end arrangements_of_opening_rooms_l243_24331


namespace pyramid_height_l243_24388

theorem pyramid_height (perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : perimeter = 32) (h_apex : apex_to_vertex = 12) :
  let side := perimeter / 4
  let center_to_corner := side * Real.sqrt 2 / 2
  Real.sqrt (apex_to_vertex ^ 2 - center_to_corner ^ 2) = 4 * Real.sqrt 7 := by
sorry

end pyramid_height_l243_24388


namespace bottles_sold_wed_to_sun_is_250_l243_24392

/-- Represents the inventory and sales of hand sanitizer bottles at Danivan Drugstore --/
structure DrugstoreInventory where
  initial_inventory : ℕ
  monday_sales : ℕ
  tuesday_sales : ℕ
  saturday_delivery : ℕ
  final_inventory : ℕ

/-- Calculates the number of bottles sold from Wednesday to Sunday --/
def bottles_sold_wed_to_sun (d : DrugstoreInventory) : ℕ :=
  d.initial_inventory - d.monday_sales - d.tuesday_sales + d.saturday_delivery - d.final_inventory

/-- Theorem stating that the number of bottles sold from Wednesday to Sunday is 250 --/
theorem bottles_sold_wed_to_sun_is_250 (d : DrugstoreInventory) 
    (h1 : d.initial_inventory = 4500)
    (h2 : d.monday_sales = 2445)
    (h3 : d.tuesday_sales = 900)
    (h4 : d.saturday_delivery = 650)
    (h5 : d.final_inventory = 1555) :
    bottles_sold_wed_to_sun d = 250 := by
  sorry

end bottles_sold_wed_to_sun_is_250_l243_24392


namespace problem_solution_l243_24352

-- Define the function f
def f (x m : ℝ) : ℝ := |x - 1| + |x + 3| - m

-- Define the theorem
theorem problem_solution :
  (∃ m : ℝ, ∀ x : ℝ, f x m < 5 ↔ -4 < x ∧ x < 2) →
  (∀ a b c : ℝ, a^2 + b^2/4 + c^2/9 = 1 → a + b + c ≤ Real.sqrt 14) :=
by
  sorry

end problem_solution_l243_24352


namespace inverse_sum_of_cube_function_l243_24360

def g (x : ℝ) : ℝ := x^3

theorem inverse_sum_of_cube_function :
  g⁻¹ 8 + g⁻¹ (-64) = -2 :=
by sorry

end inverse_sum_of_cube_function_l243_24360


namespace sqrt_necessary_not_sufficient_for_ln_l243_24378

theorem sqrt_necessary_not_sufficient_for_ln :
  (∀ x y, x > 0 ∧ y > 0 → (Real.log x > Real.log y → Real.sqrt x > Real.sqrt y)) ∧
  (∃ x y, Real.sqrt x > Real.sqrt y ∧ ¬(Real.log x > Real.log y)) := by
  sorry

end sqrt_necessary_not_sufficient_for_ln_l243_24378


namespace sequence_appearance_equivalence_l243_24307

/-- For positive real numbers a and b satisfying 2ab = a - b, 
    any positive integer n appears in the sequence (⌊ak + 1/2⌋)_{k≥1} 
    if and only if it appears at least three times in the sequence (⌊bk + 1/2⌋)_{k≥1} -/
theorem sequence_appearance_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a * b = a - b) :
  ∀ n : ℕ, n > 0 → 
    (∃ k : ℕ, k > 0 ∧ |a * k - n| < 1/2) ↔ 
    (∃ m₁ m₂ m₃ : ℕ, m₁ > 0 ∧ m₂ > 0 ∧ m₃ > 0 ∧ m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃ ∧ 
      |b * m₁ - n| < 1/2 ∧ |b * m₂ - n| < 1/2 ∧ |b * m₃ - n| < 1/2) := by
  sorry


end sequence_appearance_equivalence_l243_24307


namespace power_inequality_l243_24361

/-- Proof of inequality involving powers -/
theorem power_inequality (a b c : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n ≥ 5) :
  a^n + b^n + c^n ≥ a^(n-5) * b^3 * c^2 + b^(n-5) * c^3 * a^2 + c^(n-5) * a^3 * b^2 := by
  sorry

#check power_inequality

end power_inequality_l243_24361


namespace cost_of_tax_free_items_l243_24386

/-- Calculates the cost of tax-free items given total amount spent, sales tax paid, and tax rate -/
theorem cost_of_tax_free_items
  (total_spent : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h_total : total_spent = 25)
  (h_tax : sales_tax = 0.30)
  (h_rate : tax_rate = 0.06)
  : ∃ (cost_tax_free : ℝ), cost_tax_free = 20 :=
by sorry

end cost_of_tax_free_items_l243_24386


namespace annual_growth_rate_is_30_percent_l243_24342

-- Define the initial number of users and the number after 2 years
def initial_users : ℝ := 1000000
def users_after_2_years : ℝ := 1690000

-- Define the time period
def years : ℝ := 2

-- Define the growth rate as a function
def growth_rate (x : ℝ) : Prop :=
  initial_users * (1 + x)^years = users_after_2_years

-- Theorem statement
theorem annual_growth_rate_is_30_percent :
  ∃ (x : ℝ), x > 0 ∧ growth_rate x ∧ x = 0.3 := by
  sorry

end annual_growth_rate_is_30_percent_l243_24342


namespace fraction_zero_solution_l243_24329

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4) / (x + 4) = 0 ∧ x ≠ -4 → x = 2 ∨ x = -2 := by
  sorry

end fraction_zero_solution_l243_24329


namespace lea_purchases_cost_l243_24332

/-- The cost of a single book -/
def book_cost : ℕ := 16

/-- The cost of a single binder -/
def binder_cost : ℕ := 2

/-- The number of binders bought -/
def num_binders : ℕ := 3

/-- The cost of a single notebook -/
def notebook_cost : ℕ := 1

/-- The number of notebooks bought -/
def num_notebooks : ℕ := 6

/-- The total cost of Léa's purchases -/
def total_cost : ℕ := book_cost + (binder_cost * num_binders) + (notebook_cost * num_notebooks)

theorem lea_purchases_cost : total_cost = 28 := by
  sorry

end lea_purchases_cost_l243_24332


namespace quadratic_root_range_l243_24336

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x < 1 ∧ y > 1 ∧ x^2 - 4*a*x + 3*a^2 = 0 ∧ y^2 - 4*a*y + 3*a^2 = 0) →
  (1/3 < a ∧ a < 1) :=
by sorry

end quadratic_root_range_l243_24336


namespace profit_percentage_previous_year_l243_24318

/-- Profit as a percentage of revenue in the previous year, given:
  1. In 1999, revenues fell by 30% compared to the previous year.
  2. In 1999, profits were 14% of revenues.
  3. Profits in 1999 were 98% of the profits in the previous year. -/
theorem profit_percentage_previous_year (R : ℝ) (P : ℝ) 
  (h1 : 0.7 * R = R - 0.3 * R)  -- Revenue fell by 30%
  (h2 : 0.14 * (0.7 * R) = 0.098 * R)  -- Profits were 14% of revenues in 1999
  (h3 : 0.98 * P = 0.098 * R)  -- Profits in 1999 were 98% of previous year
  : P / R = 0.1 := by
  sorry

#check profit_percentage_previous_year

end profit_percentage_previous_year_l243_24318


namespace four_divided_by_p_l243_24397

theorem four_divided_by_p (q p : ℝ) 
  (h1 : 4 / q = 18) 
  (h2 : p - q = 0.2777777777777778) : 
  4 / p = 8 := by sorry

end four_divided_by_p_l243_24397


namespace envelope_weight_proof_l243_24344

/-- The weight of the envelope in Jessica's letter mailing scenario -/
def envelope_weight : ℚ := 2/5

/-- The number of pieces of paper used -/
def paper_count : ℕ := 8

/-- The weight of each piece of paper in ounces -/
def paper_weight : ℚ := 1/5

/-- The number of stamps needed -/
def stamps_needed : ℕ := 2

/-- The maximum weight in ounces that can be mailed with the given number of stamps -/
def max_weight (stamps : ℕ) : ℚ := stamps

theorem envelope_weight_proof :
  (paper_count : ℚ) * paper_weight + envelope_weight > (stamps_needed - 1 : ℚ) ∧
  (paper_count : ℚ) * paper_weight + envelope_weight ≤ stamps_needed ∧
  envelope_weight > 0 :=
sorry

end envelope_weight_proof_l243_24344


namespace functional_equation_solution_l243_24391

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (g 1 = 2) ∧ 
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y)

/-- The main theorem stating that the function g satisfying the functional equation
    is equal to 2(4^x - 3^x) for all real x -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end functional_equation_solution_l243_24391


namespace consecutive_even_integers_cube_sum_l243_24338

theorem consecutive_even_integers_cube_sum : 
  ∀ a : ℕ, 
    a > 0 → 
    (2*a - 2) * (2*a) * (2*a + 2) = 12 * (6*a) → 
    (2*a - 2)^3 + (2*a)^3 + (2*a + 2)^3 = 8568 :=
by
  sorry

end consecutive_even_integers_cube_sum_l243_24338


namespace total_score_is_38_l243_24351

/-- Represents the scores of three friends in a table football game. -/
structure Scores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the game and the total score calculation. -/
def game_result (s : Scores) : Prop :=
  s.marius = s.darius + 3 ∧
  s.matt = s.darius + 5 ∧
  s.darius = 10 ∧
  s.darius + s.matt + s.marius = 38

/-- Theorem stating that under the given conditions, the total score is 38. -/
theorem total_score_is_38 : ∃ s : Scores, game_result s :=
  sorry

end total_score_is_38_l243_24351


namespace smallest_three_digit_number_l243_24303

def Digits : Set Nat := {0, 3, 5, 6}

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n / 100 ∈ Digits) ∧
  ((n / 10) % 10 ∈ Digits) ∧
  (n % 10 ∈ Digits) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem smallest_three_digit_number :
  ∀ n : Nat, is_valid_number n → n ≥ 305 :=
sorry

end smallest_three_digit_number_l243_24303


namespace largest_binomial_coefficient_equality_holds_largest_n_is_six_l243_24321

theorem largest_binomial_coefficient (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem equality_holds : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_six : 
  ∃ (n : ℕ), n = 6 ∧ 
    Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧
    ∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n :=
by sorry

end largest_binomial_coefficient_equality_holds_largest_n_is_six_l243_24321


namespace extra_fruit_calculation_l243_24312

theorem extra_fruit_calculation (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 42)
  (h2 : green_apples = 7)
  (h3 : students = 9) : 
  red_apples + green_apples - students = 40 := by
  sorry

end extra_fruit_calculation_l243_24312


namespace solution_set_inequalities_l243_24377

theorem solution_set_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by sorry

end solution_set_inequalities_l243_24377


namespace pennsylvania_quarters_l243_24396

theorem pennsylvania_quarters (total : ℕ) (state_fraction : ℚ) (penn_fraction : ℚ) : 
  total = 35 → 
  state_fraction = 2 / 5 → 
  penn_fraction = 1 / 2 → 
  (total : ℚ) * state_fraction * penn_fraction = 7 := by
sorry

end pennsylvania_quarters_l243_24396


namespace min_value_theorem_l243_24372

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 + x*y = 315) :
  ∃ (m : ℝ), m = 105 ∧ ∀ z, x^2 + y^2 - x*y ≥ z → z ≤ m :=
sorry

end min_value_theorem_l243_24372


namespace distinct_collections_count_l243_24381

/-- Represents the count of each letter in ALGEBRAICS --/
structure LetterCount where
  a : Nat
  b : Nat
  c : Nat
  e : Nat
  g : Nat
  i : Nat
  l : Nat
  r : Nat
  s : Nat

/-- The initial count of letters in ALGEBRAICS --/
def initialCount : LetterCount :=
  { a := 2, b := 1, c := 1, e := 1, g := 1, i := 1, l := 1, r := 1, s := 1 }

/-- Counts the number of distinct collections of two vowels and two consonants --/
def countDistinctCollections (count : LetterCount) : Nat :=
  sorry

theorem distinct_collections_count :
  countDistinctCollections initialCount = 68 :=
sorry

end distinct_collections_count_l243_24381


namespace element_selection_theorem_l243_24390

variable {α : Type*} [DecidableEq α]

def SubsetProperty (S : Finset α) (n k : ℕ) (S_i : ℕ → Finset α) : Prop :=
  (S.card = n) ∧ 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ k * n → S_i i ⊆ S ∧ (S_i i).card = 2) ∧
  (∀ e ∈ S, (Finset.filter (fun i => e ∈ S_i i) (Finset.range (k * n))).card = 2 * k)

theorem element_selection_theorem (S : Finset α) (n k : ℕ) (S_i : ℕ → Finset α) 
  (h : SubsetProperty S n k S_i) :
  ∃ f : ℕ → α, 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ k * n → f i ∈ S_i i) ∧ 
    (∀ e ∈ S, (Finset.filter (fun i => f i = e) (Finset.range (k * n))).card = k) :=
sorry

end element_selection_theorem_l243_24390


namespace actual_car_body_mass_l243_24362

/-- Represents the scale factor between the model and the actual car body. -/
def scaleFactor : ℝ := 10

/-- Represents the mass of the model car body in kilograms. -/
def modelMass : ℝ := 1.5

/-- Calculates the mass of the actual car body given the scale factor and model mass. -/
def actualMass (s : ℝ) (m : ℝ) : ℝ := s^3 * m

/-- Theorem stating that the mass of the actual car body is 1500 kg. -/
theorem actual_car_body_mass :
  actualMass scaleFactor modelMass = 1500 := by
  sorry

end actual_car_body_mass_l243_24362


namespace frank_has_two_ten_dollar_bills_l243_24349

-- Define the problem parameters
def one_dollar_bills : ℕ := 7
def five_dollar_bills : ℕ := 4
def twenty_dollar_bill : ℕ := 1
def peanut_price_per_pound : ℕ := 3
def change : ℕ := 4
def daily_peanut_consumption : ℕ := 3
def days_in_week : ℕ := 7

-- Define the function to calculate the number of ten-dollar bills
def calculate_ten_dollar_bills : ℕ := 
  let total_without_tens : ℕ := one_dollar_bills + 5 * five_dollar_bills + 20 * twenty_dollar_bill
  let total_peanuts_bought : ℕ := daily_peanut_consumption * days_in_week
  let total_spent : ℕ := peanut_price_per_pound * total_peanuts_bought
  let amount_from_tens : ℕ := total_spent - total_without_tens + change
  amount_from_tens / 10

-- Theorem stating that Frank has exactly 2 ten-dollar bills
theorem frank_has_two_ten_dollar_bills : calculate_ten_dollar_bills = 2 := by
  sorry

end frank_has_two_ten_dollar_bills_l243_24349


namespace max_sum_cubes_l243_24335

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (M : ℝ), M = 5 * Real.sqrt 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 ≤ M ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = M :=
by sorry

end max_sum_cubes_l243_24335


namespace distance_PQ_is_2_25_l243_24316

/-- The distance between two points on a ruler -/
def distance_on_ruler (p q : ℚ) : ℚ := q - p

/-- The position of point P on the ruler -/
def P : ℚ := 1/2

/-- The position of point Q on the ruler -/
def Q : ℚ := 2 + 3/4

theorem distance_PQ_is_2_25 : distance_on_ruler P Q = 2.25 := by sorry

end distance_PQ_is_2_25_l243_24316


namespace transposition_changes_cycles_even_permutation_iff_even_diff_l243_24330

/-- A permutation of numbers 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Number of cycles in a permutation -/
def numCycles (σ : Permutation n) : ℕ := sorry

/-- Perform a transposition on a permutation -/
def transpose (σ : Permutation n) (i j : Fin n) : Permutation n := sorry

/-- A permutation is even -/
def isEven (σ : Permutation n) : Prop := sorry

theorem transposition_changes_cycles (n : ℕ) (σ : Permutation n) (i j : Fin n) :
  ∃ k : ℤ, k = 1 ∨ k = -1 ∧ numCycles (transpose σ i j) = numCycles σ + k :=
sorry

theorem even_permutation_iff_even_diff (n : ℕ) (σ : Permutation n) :
  isEven σ ↔ Even (n - numCycles σ) :=
sorry

end transposition_changes_cycles_even_permutation_iff_even_diff_l243_24330


namespace ajay_walking_distance_l243_24370

/-- Ajay's walking problem -/
theorem ajay_walking_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 3) 
  (h2 : time = 16.666666666666668) : 
  speed * time = 50 := by
  sorry

end ajay_walking_distance_l243_24370


namespace number_in_set_l243_24327

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a 3-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a 3-digit number -/
def reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- The theorem to be proved -/
theorem number_in_set (numbers : List ThreeDigitNumber) (reversed : ThreeDigitNumber) 
  (h_reversed : reversed ∈ numbers)
  (h_diff : reversed.units - reversed.hundreds = 2)
  (h_average_increase : (reversed_value reversed - value reversed : ℚ) / numbers.length = 198/10) :
  numbers.length = 10 := by
  sorry

end number_in_set_l243_24327


namespace quadratic_factorization_conditions_l243_24355

theorem quadratic_factorization_conditions (b : ℤ) : 
  ¬ ∀ (m n p q : ℤ), 
    (15 : ℤ) * x^2 + b * x + 75 = (m * x + n) * (p * x + q) → 
    ∃ (r s : ℤ), (15 : ℤ) * x^2 + b * x + 75 = (m * x + n) * (p * x + q) * (r * x + s) :=
by sorry

end quadratic_factorization_conditions_l243_24355


namespace calculation_proof_l243_24345

theorem calculation_proof : (30 / (8 + 2 - 5)) * 4 = 24 := by
  sorry

end calculation_proof_l243_24345


namespace smallest_relatively_prime_to_180_l243_24380

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_relatively_prime_to_180 :
  ∃ (x : ℕ), x > 1 ∧ is_relatively_prime x 180 ∧
  ∀ (y : ℕ), y > 1 ∧ y < x → ¬(is_relatively_prime y 180) :=
by
  use 7
  sorry

end smallest_relatively_prime_to_180_l243_24380


namespace fib_150_mod_9_l243_24368

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Fibonacci sequence modulo 9 repeats every 24 terms -/
axiom fib_mod_9_period : ∀ n, fib n % 9 = fib (n % 24) % 9

/-- The 6th Fibonacci number modulo 9 is 8 -/
axiom fib_6_mod_9 : fib 6 % 9 = 8

/-- The 150th Fibonacci number modulo 9 -/
theorem fib_150_mod_9 : fib 150 % 9 = 8 := by sorry

end fib_150_mod_9_l243_24368


namespace john_biking_distance_l243_24346

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 7^3 + d₂ * 7^2 + d₁ * 7^1 + d₀ * 7^0

/-- The problem statement --/
theorem john_biking_distance :
  base7ToBase10 3 9 5 6 = 1511 := by
  sorry

end john_biking_distance_l243_24346


namespace fencing_length_l243_24340

theorem fencing_length (area : ℝ) (uncovered_side : ℝ) : 
  area = 680 → uncovered_side = 8 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    uncovered_side + 2 * width = 178 := by
  sorry

end fencing_length_l243_24340


namespace quadratic_one_solution_l243_24395

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 36 = 0) → m = 12 * Real.sqrt 3 :=
by sorry

end quadratic_one_solution_l243_24395


namespace missing_fraction_sum_l243_24379

theorem missing_fraction_sum (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / 2 : ℚ) + (1 / 5 : ℚ) + (1 / 4 : ℚ) + (-9 / 20 : ℚ) + (-5 / 6 : ℚ) + x = 0.8333333333333334 
  → x = 0.8333333333333334 := by
sorry

end missing_fraction_sum_l243_24379


namespace sin_theta_equals_sqrt2_over_2_l243_24382

theorem sin_theta_equals_sqrt2_over_2 (θ : Real) (a : Real) (h1 : a ≠ 0) 
  (h2 : ∃ (x y : Real), x = a ∧ y = a ∧ Real.cos θ * Real.cos θ + Real.sin θ * Real.sin θ = 1 ∧ 
    Real.cos θ * x - Real.sin θ * y = 0 ∧ Real.sin θ * x + Real.cos θ * y = 0) : 
  |Real.sin θ| = Real.sqrt 2 / 2 := by
sorry

end sin_theta_equals_sqrt2_over_2_l243_24382


namespace star_two_neg_three_l243_24313

/-- The ⋆ operation defined on real numbers -/
def star (a b : ℝ) : ℝ := a^2 * b^2 + a - 1

/-- Theorem stating that 2 ⋆ (-3) = 37 -/
theorem star_two_neg_three : star 2 (-3) = 37 := by
  sorry

end star_two_neg_three_l243_24313


namespace equal_trout_division_l243_24350

theorem equal_trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 18 → num_people = 2 → total_trout / num_people = trout_per_person → trout_per_person = 9 := by
  sorry

end equal_trout_division_l243_24350


namespace smallest_triangle_side_l243_24363

theorem smallest_triangle_side : ∃ (t : ℕ), 
  (∀ (s : ℕ), s < t → ¬(7 < s + 13 ∧ 13 < 7 + s ∧ s < 7 + 13)) ∧ 
  (7 < t + 13 ∧ 13 < 7 + t ∧ t < 7 + 13) :=
by sorry

end smallest_triangle_side_l243_24363


namespace state_A_selection_percentage_l243_24337

theorem state_A_selection_percentage : 
  ∀ (total_candidates : ℕ) (state_B_percentage : ℚ) (extra_selected : ℕ),
    total_candidates = 8000 →
    state_B_percentage = 7 / 100 →
    extra_selected = 80 →
    ∃ (state_A_percentage : ℚ),
      state_A_percentage * total_candidates + extra_selected = state_B_percentage * total_candidates ∧
      state_A_percentage = 6 / 100 := by
  sorry

end state_A_selection_percentage_l243_24337


namespace area_of_region_l243_24389

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 77 ∧ 
   A = Real.pi * (((x + 8)^2 + (y - 3)^2) / 4) ∧
   x^2 + y^2 - 8 = 6*y - 16*x + 4) :=
by sorry

end area_of_region_l243_24389


namespace range_of_a_l243_24369

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := StrictMono (fun x => Real.log x / Real.log a)

-- Define the theorem
theorem range_of_a :
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  {a : ℝ | (-2 < a ∧ a ≤ 1) ∨ (a ≥ 2)} = {a : ℝ | a ∈ Set.Ioc (-2) 1 ∪ Set.Ici 2} :=
sorry

end range_of_a_l243_24369


namespace polynomial_powered_function_is_polynomial_l243_24398

/-- A function f: ℝ → ℝ such that (f(x))^n is a polynomial for every integer n ≥ 2 -/
def PolynomialPoweredFunction (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ p : Polynomial ℝ, ∀ x : ℝ, (f x)^n = p.eval x

/-- If f: ℝ → ℝ is a function such that (f(x))^n is a polynomial for every integer n ≥ 2,
    then f is a polynomial -/
theorem polynomial_powered_function_is_polynomial (f : ℝ → ℝ) 
  (h : PolynomialPoweredFunction f) : ∃ p : Polynomial ℝ, ∀ x : ℝ, f x = p.eval x :=
by
  sorry

end polynomial_powered_function_is_polynomial_l243_24398


namespace mowing_earnings_l243_24399

/-- Calculates the earnings for a single hour of mowing based on the hour number within a cycle -/
def hourly_pay (hour : ℕ) : ℕ :=
  5 * (hour % 6 + 1)

/-- Calculates the total earnings for a given number of hours of mowing -/
def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_pay |>.sum

/-- Theorem stating that mowing for 24 hours results in earnings of 420 dollars -/
theorem mowing_earnings : total_earnings 24 = 420 := by
  sorry

end mowing_earnings_l243_24399


namespace complex_number_in_second_quadrant_l243_24320

/-- The complex number z = i / (1 - i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_second_quadrant_l243_24320


namespace min_distance_between_curves_l243_24328

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (2 * x)

theorem min_distance_between_curves : 
  ∃ (d : ℝ), d = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt ((x - y)^2 + (f x - g y)^2) ≥ d :=
sorry

end min_distance_between_curves_l243_24328


namespace lock_probability_l243_24319

/-- Given a set of keys and a subset that can open a lock, 
    calculate the probability of randomly selecting a key that opens the lock -/
def probability_open_lock (total_keys : ℕ) (opening_keys : ℕ) : ℚ :=
  opening_keys / total_keys

/-- Theorem: The probability of opening a lock with 2 out of 5 keys is 2/5 -/
theorem lock_probability : 
  probability_open_lock 5 2 = 2 / 5 := by
  sorry

end lock_probability_l243_24319


namespace round_robin_tournament_l243_24358

theorem round_robin_tournament (x : ℕ) : x > 0 → (x * (x - 1)) / 2 = 15 → x = 6 := by
  sorry

end round_robin_tournament_l243_24358


namespace max_value_of_expression_l243_24373

theorem max_value_of_expression (x : ℝ) (h : x > 0) :
  1 - x - 16 / x ≤ -7 ∧ ∃ y > 0, 1 - y - 16 / y = -7 :=
by sorry

end max_value_of_expression_l243_24373


namespace june_election_win_l243_24357

/-- The minimum percentage of boys required for June to win the election -/
def min_boys_percentage : ℝ :=
  -- We'll define this later in the proof
  sorry

theorem june_election_win (total_students : ℕ) (boys_vote_percentage : ℝ) (girls_vote_percentage : ℝ) 
  (h_total : total_students = 200)
  (h_boys_vote : boys_vote_percentage = 67.5)
  (h_girls_vote : girls_vote_percentage = 25)
  (h_win_threshold : ∀ x : ℝ, x > 50 → x ≥ (total_students : ℝ) / 2 + 0.5) :
  ∃ ε > 0, abs (min_boys_percentage - 60) < ε ∧ 
  ∀ boys_percentage : ℝ, boys_percentage ≥ min_boys_percentage →
    (boys_percentage * boys_vote_percentage + (100 - boys_percentage) * girls_vote_percentage) / 100 > 50 :=
by sorry

end june_election_win_l243_24357


namespace balloon_ratio_l243_24376

theorem balloon_ratio : 
  let dan_balloons : ℝ := 29.0
  let tim_balloons : ℝ := 4.142857143
  dan_balloons / tim_balloons = 7 := by
sorry

end balloon_ratio_l243_24376


namespace sign_of_a_l243_24385

theorem sign_of_a (a b c : ℝ) (n : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ((-2)^8 * a^3 * b^3 * c^(n-1)) * ((-3)^3 * a^2 * b^5 * c^(n+1)) > 0) : 
  a < 0 := by
sorry

end sign_of_a_l243_24385


namespace jimmy_cookies_count_l243_24359

/-- Given that:
  - Crackers contain 15 calories each
  - Cookies contain 50 calories each
  - Jimmy eats 10 crackers
  - Jimmy consumes a total of 500 calories
  Prove that Jimmy eats 7 cookies -/
theorem jimmy_cookies_count :
  let cracker_calories : ℕ := 15
  let cookie_calories : ℕ := 50
  let crackers_eaten : ℕ := 10
  let total_calories : ℕ := 500
  let cookies_eaten : ℕ := (total_calories - cracker_calories * crackers_eaten) / cookie_calories
  cookies_eaten = 7 :=
by sorry

end jimmy_cookies_count_l243_24359


namespace min_value_theorem_l243_24322

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 4 * a + b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 1 → 1 / (2 * x) + 2 / y ≥ 1 / (2 * a) + 2 / b) ∧
  1 / (2 * a) + 2 / b = 8 := by
sorry

end min_value_theorem_l243_24322


namespace no_common_points_l243_24310

/-- Theorem: If a point (x, y) is inside the parabola y^2 = 4x, 
    then the line yy = 2(x + x) and the parabola have no common points. -/
theorem no_common_points (x y : ℝ) (h : y^2 < 4*x) : 
  ∀ (x' y' : ℝ), y'^2 = 4*x' → y'*y = 2*(x + x') → False :=
by sorry

end no_common_points_l243_24310


namespace max_distance_point_to_line_l243_24348

noncomputable def point_to_line_distance (θ : ℝ) : ℝ :=
  |3 * Real.cos θ + 4 * Real.sin θ - 4| / 5

theorem max_distance_point_to_line :
  ∃ (θ : ℝ), ∀ (φ : ℝ), point_to_line_distance θ ≥ point_to_line_distance φ ∧
  point_to_line_distance θ = 9/5 := by
  sorry

end max_distance_point_to_line_l243_24348


namespace square_area_ratio_l243_24305

theorem square_area_ratio : 
  let side_C : ℝ := 24
  let side_D : ℝ := 30
  let area_C := side_C ^ 2
  let area_D := side_D ^ 2
  area_C / area_D = 16 / 25 := by
  sorry

end square_area_ratio_l243_24305


namespace integer_ratio_problem_l243_24375

theorem integer_ratio_problem (a b c : ℕ) : 
  a < b → b < c → 
  a = 0 → b ≠ a + 1 → 
  (a + b + c : ℚ) / 3 = 4 * b → 
  c / b = 11 := by
  sorry

end integer_ratio_problem_l243_24375


namespace raj_house_bedrooms_l243_24325

/-- Represents the floor plan of Raj's house -/
structure RajHouse where
  total_area : ℕ
  bedroom_side : ℕ
  bathroom_length : ℕ
  bathroom_width : ℕ
  num_bathrooms : ℕ
  kitchen_area : ℕ

/-- Calculates the number of bedrooms in Raj's house -/
def num_bedrooms (house : RajHouse) : ℕ :=
  let bathroom_area := house.bathroom_length * house.bathroom_width * house.num_bathrooms
  let kitchen_living_area := 2 * house.kitchen_area
  let non_bedroom_area := bathroom_area + kitchen_living_area
  let bedroom_area := house.total_area - non_bedroom_area
  bedroom_area / (house.bedroom_side * house.bedroom_side)

/-- Theorem stating that Raj's house has 4 bedrooms -/
theorem raj_house_bedrooms :
  let house : RajHouse := {
    total_area := 1110,
    bedroom_side := 11,
    bathroom_length := 6,
    bathroom_width := 8,
    num_bathrooms := 2,
    kitchen_area := 265
  }
  num_bedrooms house = 4 := by
  sorry


end raj_house_bedrooms_l243_24325


namespace quadratic_real_roots_l243_24365

/-- For a quadratic equation x^2 + 2(k-1)x + k^2 - 1 = 0, 
    the equation has real roots if and only if k ≤ 1 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(k-1)*x + k^2 - 1 = 0) ↔ k ≤ 1 := by
  sorry

end quadratic_real_roots_l243_24365
