import Mathlib

namespace roots_equation_sum_l2207_220760

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 4*α - 1 = 0 → β^2 - 4*β - 1 = 0 → 3*α^3 + 4*β^2 = 80 + 35*α := by
  sorry

end roots_equation_sum_l2207_220760


namespace triangle_solutions_l2207_220746

/-- Function to determine the number of triangle solutions given two sides and an angle --/
def triangleSolutionsCount (a b : ℝ) (angleA : Real) : Nat :=
  sorry

theorem triangle_solutions :
  (triangleSolutionsCount 5 4 (120 * π / 180) = 1) ∧
  (triangleSolutionsCount 7 14 (150 * π / 180) = 0) ∧
  (triangleSolutionsCount 9 10 (60 * π / 180) = 2) := by
  sorry


end triangle_solutions_l2207_220746


namespace mike_changed_tires_on_ten_cars_l2207_220767

/-- The number of cars Mike changed tires on -/
def num_cars (total_tires num_motorcycles tires_per_motorcycle tires_per_car : ℕ) : ℕ :=
  (total_tires - num_motorcycles * tires_per_motorcycle) / tires_per_car

theorem mike_changed_tires_on_ten_cars :
  num_cars 64 12 2 4 = 10 := by sorry

end mike_changed_tires_on_ten_cars_l2207_220767


namespace arithmetic_series_sum_l2207_220701

theorem arithmetic_series_sum : ∀ (a₁ aₙ : ℕ), 
  a₁ = 5 → aₙ = 105 → 
  ∃ (n : ℕ), n > 1 ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ n → ∃ d, a₁ + (k - 1) * d = aₙ) →
  (n * (a₁ + aₙ)) / 2 = 5555 := by
sorry

end arithmetic_series_sum_l2207_220701


namespace inequality_proof_l2207_220796

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (c/a)*(8*b+c) + (d/b)*(8*c+d) + (a/c)*(8*d+a) + (b/d)*(8*a+b) ≥ 9*(a+b+c+d) := by
  sorry

end inequality_proof_l2207_220796


namespace min_value_expression_l2207_220793

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt 3 * 3^(a + b) * (1/a + 1/b) ≥ 12 ∧
  (Real.sqrt 3 * 3^(a + b) * (1/a + 1/b) = 12 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_value_expression_l2207_220793


namespace game_c_higher_probability_l2207_220732

-- Define the probability of getting heads
def p_heads : ℚ := 2/3

-- Define the probability of getting tails
def p_tails : ℚ := 1/3

-- Define the probability of winning Game C
def p_game_c : ℚ :=
  let p_first_three := p_heads^3 + p_tails^3
  let p_last_three := p_heads^3 + p_tails^3
  let p_overlap := p_heads^5 + p_tails^5
  p_first_three + p_last_three - p_overlap

-- Define the probability of winning Game D
def p_game_d : ℚ :=
  let p_first_last_two := (p_heads^2 + p_tails^2)^2
  let p_middle_three := p_heads^3 + p_tails^3
  let p_overlap := 2 * (p_heads^4 + p_tails^4)
  p_first_last_two + p_middle_three - p_overlap

-- Theorem statement
theorem game_c_higher_probability :
  p_game_c - p_game_d = 29/81 :=
sorry

end game_c_higher_probability_l2207_220732


namespace nancy_target_amount_l2207_220769

def hourly_rate (total_earnings : ℚ) (hours_worked : ℚ) : ℚ :=
  total_earnings / hours_worked

def target_amount (rate : ℚ) (target_hours : ℚ) : ℚ :=
  rate * target_hours

theorem nancy_target_amount 
  (initial_earnings : ℚ) 
  (initial_hours : ℚ) 
  (target_hours : ℚ) 
  (h1 : initial_earnings = 28)
  (h2 : initial_hours = 4)
  (h3 : target_hours = 10) :
  target_amount (hourly_rate initial_earnings initial_hours) target_hours = 70 :=
by
  sorry

end nancy_target_amount_l2207_220769


namespace hexagonal_tiling_chromatic_number_l2207_220721

/-- A type representing colors -/
inductive Color
| Red
| Green
| Blue

/-- A type representing a hexagonal tile in the plane -/
structure HexTile :=
  (id : ℕ)

/-- A function type that assigns colors to hexagonal tiles -/
def Coloring := HexTile → Color

/-- Predicate to check if two hexagonal tiles are adjacent (share a side) -/
def adjacent : HexTile → HexTile → Prop := sorry

/-- Predicate to check if a coloring is valid (no adjacent tiles have the same color) -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ h1 h2, adjacent h1 h2 → c h1 ≠ c h2

/-- The main theorem: The minimum number of colors needed is 3 -/
theorem hexagonal_tiling_chromatic_number :
  (∃ c : Coloring, valid_coloring c) ∧
  (∀ c : Coloring, valid_coloring c → (Set.range c).ncard ≥ 3) :=
sorry

end hexagonal_tiling_chromatic_number_l2207_220721


namespace geometric_sequence_problem_l2207_220771

/-- A geometric sequence is a sequence where each term after the first is found by 
    multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a with a₁ = 2 and a₃ = 4, prove that a₇ = 16 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 4) : 
  a 7 = 16 := by
sorry


end geometric_sequence_problem_l2207_220771


namespace total_surface_area_circumscribed_prism_l2207_220737

/-- A prism circumscribed about a sphere -/
structure CircumscribedPrism where
  -- The area of the base of the prism
  base_area : ℝ
  -- The semi-perimeter of the base of the prism
  semi_perimeter : ℝ
  -- The radius of the sphere
  sphere_radius : ℝ
  -- The base area is equal to the product of semi-perimeter and sphere radius
  base_area_eq : base_area = semi_perimeter * sphere_radius

/-- The total surface area of a circumscribed prism is 6 times its base area -/
theorem total_surface_area_circumscribed_prism (p : CircumscribedPrism) :
  ∃ (total_surface_area : ℝ), total_surface_area = 6 * p.base_area :=
by
  sorry

end total_surface_area_circumscribed_prism_l2207_220737


namespace solve_equation_l2207_220733

theorem solve_equation : ∃ x : ℚ, 5 * (x - 6) = 3 * (3 - 3 * x) + 9 ∧ x = 24 / 7 := by
  sorry

end solve_equation_l2207_220733


namespace quadratic_minimum_l2207_220731

def f (x : ℝ) : ℝ := x^2 - 6*x + 13

theorem quadratic_minimum :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 4 := by
  sorry

end quadratic_minimum_l2207_220731


namespace hexagon_dimension_theorem_l2207_220739

/-- Represents a hexagon with dimension y -/
structure Hexagon :=
  (y : ℝ)

/-- Represents a rectangle with length and width -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)

/-- Represents a square with side length -/
structure Square :=
  (side : ℝ)

/-- The theorem stating that for an 8x18 rectangle cut into two congruent hexagons 
    that can be repositioned to form a square, the dimension y of the hexagon is 6 -/
theorem hexagon_dimension_theorem (rect : Rectangle) (hex1 hex2 : Hexagon) (sq : Square) :
  rect.length = 18 ∧ 
  rect.width = 8 ∧
  hex1 = hex2 ∧
  rect.length * rect.width = sq.side * sq.side →
  hex1.y = 6 := by
  sorry

end hexagon_dimension_theorem_l2207_220739


namespace xiao_ming_error_l2207_220740

theorem xiao_ming_error (x : ℝ) : 
  (x + 1) / 2 - 1 = (x - 2) / 3 → 
  3 * (x + 1) - 1 ≠ 2 * (x - 2) :=
by
  sorry

end xiao_ming_error_l2207_220740


namespace workshop_average_salary_l2207_220749

/-- Given a workshop with workers, prove that the average salary is 8000 --/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (avg_salary_technicians : ℕ)
  (avg_salary_rest : ℕ)
  (h1 : total_workers = 30)
  (h2 : technicians = 10)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  (technicians * avg_salary_technicians + (total_workers - technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end workshop_average_salary_l2207_220749


namespace debate_team_count_l2207_220786

/-- The number of girls in the debate club -/
def num_girls : ℕ := 4

/-- The number of boys in the debate club -/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for each team -/
def girls_per_team : ℕ := 3

/-- The number of boys to be chosen for each team -/
def boys_per_team : ℕ := 3

/-- Theorem stating the total number of possible debate teams -/
theorem debate_team_count : 
  (Nat.choose num_girls girls_per_team) * (Nat.choose num_boys boys_per_team) = 80 := by
  sorry

end debate_team_count_l2207_220786


namespace total_cents_l2207_220727

-- Define the values in cents
def lance_cents : ℕ := 70
def margaret_cents : ℕ := 75 -- three-fourths of a dollar
def guy_cents : ℕ := 50 + 10 -- two quarters and a dime
def bill_cents : ℕ := 6 * 10 -- six dimes

-- Theorem to prove
theorem total_cents : 
  lance_cents + margaret_cents + guy_cents + bill_cents = 265 := by
  sorry

end total_cents_l2207_220727


namespace sufficient_not_necessary_condition_l2207_220744

/-- The quadratic function f(x) with parameter t -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 + t*x - t

/-- f has a root for a given t -/
def has_root (t : ℝ) : Prop := ∃ x, f t x = 0

theorem sufficient_not_necessary_condition :
  (∀ t ≥ 0, has_root t) ∧ (∃ t < 0, has_root t) := by sorry

end sufficient_not_necessary_condition_l2207_220744


namespace solve_exponential_equation_l2207_220702

theorem solve_exponential_equation (x : ℝ) (h : x ≠ 0) :
  x^(-(2/3) : ℝ) = 4 ↔ x = 1/8 ∨ x = -1/8 := by
  sorry

end solve_exponential_equation_l2207_220702


namespace unique_integer_solution_l2207_220735

theorem unique_integer_solution :
  ∃! (a : ℤ), ∃ (d e : ℤ), ∀ (x : ℤ), (x - a) * (x - 8) - 3 = (x + d) * (x + e) ∧ a = 6 :=
by sorry

end unique_integer_solution_l2207_220735


namespace total_pencils_l2207_220713

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of Chloe's friends who bought the same color box -/
def friends : ℕ := 5

/-- The total number of people who bought color boxes (Chloe and her friends) -/
def total_people : ℕ := friends + 1

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

theorem total_pencils :
  pencils_per_box * total_people = 42 :=
by sorry

end total_pencils_l2207_220713


namespace parabola_point_comparison_l2207_220772

theorem parabola_point_comparison 
  (m : ℝ) (t x₁ x₂ y₁ y₂ : ℝ) 
  (h_m : m > 0)
  (h_x₁ : t < x₁ ∧ x₁ < t + 1)
  (h_x₂ : t + 2 < x₂ ∧ x₂ < t + 3)
  (h_y₁ : y₁ = m * x₁^2 - 2 * m * x₁ + 1)
  (h_y₂ : y₂ = m * x₂^2 - 2 * m * x₂ + 1)
  (h_t : t ≥ 1) :
  y₁ < y₂ := by
sorry

end parabola_point_comparison_l2207_220772


namespace symmetric_circle_equation_l2207_220750

/-- Given a circle with equation x^2 + y^2 = 1 and a line of symmetry x - y - 2 = 0,
    the equation of the symmetric circle is (x-2)^2 + (y+2)^2 = 1 -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 = 1) →
  (x - y - 2 = 0) →
  (∃ (x' y' : ℝ), (x' - 2)^2 + (y' + 2)^2 = 1 ∧
    (∀ (p q : ℝ), (p - q - 2 = 0) → 
      ((x - p)^2 + (y - q)^2 = (x' - p)^2 + (y' - q)^2))) :=
by sorry

end symmetric_circle_equation_l2207_220750


namespace sum_of_cubes_counterexample_l2207_220790

theorem sum_of_cubes_counterexample : ¬∀ a : ℝ, (a + 1) * (a^2 - a + 1) = a^3 + 1 := by sorry

end sum_of_cubes_counterexample_l2207_220790


namespace min_value_reciprocal_sum_l2207_220788

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_dot_product : m * 1 + 1 * (n - 1) = 0) :
  ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_dot : x * 1 + 1 * (y - 1) = 0), 
    (1 / m + 1 / n ≥ 1 / x + 1 / y) ∧ (1 / x + 1 / y = 4) := by
  sorry

end min_value_reciprocal_sum_l2207_220788


namespace community_age_theorem_l2207_220777

/-- Represents the average age of a community given the ratio of women to men and their respective average ages -/
def community_average_age (women_ratio : ℚ) (men_ratio : ℚ) (women_avg_age : ℚ) (men_avg_age : ℚ) : ℚ :=
  (women_ratio * women_avg_age + men_ratio * men_avg_age) / (women_ratio + men_ratio)

/-- Theorem stating that for a community with a 3:2 ratio of women to men, where women's average age is 30 and men's is 35, the community's average age is 32 -/
theorem community_age_theorem :
  community_average_age (3/5) (2/5) 30 35 = 32 := by sorry

end community_age_theorem_l2207_220777


namespace largest_number_is_4968_l2207_220705

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : ℕ
  first_number : ℕ
  second_number : ℕ
  hTotal : total_students = 5000
  hRange : first_number ≥ 1 ∧ second_number ≤ total_students
  hOrder : first_number < second_number

/-- The largest number in the systematic sample -/
def largest_number (s : SystematicSample) : ℕ :=
  s.first_number + (s.second_number - s.first_number) * ((s.total_students - s.first_number) / (s.second_number - s.first_number))

/-- Theorem stating the largest number in the systematic sample -/
theorem largest_number_is_4968 (s : SystematicSample) 
  (h1 : s.first_number = 18) 
  (h2 : s.second_number = 68) : 
  largest_number s = 4968 := by
  sorry

end largest_number_is_4968_l2207_220705


namespace quadratic_equation_solution_l2207_220745

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 6*x + 5
  ∃ x₁ x₂ : ℝ, x₁ = 5 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end quadratic_equation_solution_l2207_220745


namespace polo_shirt_cost_l2207_220785

/-- Calculates the total cost of two discounted polo shirts with sales tax -/
theorem polo_shirt_cost : 
  let regular_price : ℝ := 50
  let discount1 : ℝ := 0.4
  let discount2 : ℝ := 0.3
  let sales_tax : ℝ := 0.08
  let discounted_price1 := regular_price * (1 - discount1)
  let discounted_price2 := regular_price * (1 - discount2)
  let total_before_tax := discounted_price1 + discounted_price2
  let total_with_tax := total_before_tax * (1 + sales_tax)
  total_with_tax = 70.20 := by sorry

end polo_shirt_cost_l2207_220785


namespace intersection_complement_equality_l2207_220719

def A : Set ℝ := {-2, -1, 0, 1, 2}

def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

theorem intersection_complement_equality : A ∩ (Set.univ \ B) = {-2} := by sorry

end intersection_complement_equality_l2207_220719


namespace largest_n_for_inequalities_l2207_220720

theorem largest_n_for_inequalities : ∃ (n : ℕ), n = 4 ∧ 
  (∃ (x : ℝ), ∀ (k : ℕ), k ≤ n → (k : ℝ) < x^k ∧ x^k < (k + 1 : ℝ)) ∧
  (∀ (m : ℕ), m > n → ¬∃ (x : ℝ), ∀ (k : ℕ), k ≤ m → (k : ℝ) < x^k ∧ x^k < (k + 1 : ℝ)) :=
by sorry

end largest_n_for_inequalities_l2207_220720


namespace infinitely_many_increasing_prime_divisors_l2207_220798

-- Define w(n) as the number of different prime divisors of n
def w (n : Nat) : Nat :=
  (Nat.factors n).toFinset.card

-- Theorem statement
theorem infinitely_many_increasing_prime_divisors :
  ∃ (S : Set Nat), Set.Infinite S ∧ ∀ n ∈ S, w n < w (n + 1) ∧ w (n + 1) < w (n + 2) := by
  sorry

end infinitely_many_increasing_prime_divisors_l2207_220798


namespace no_integer_solutions_l2207_220775

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^4 + y^2 = 6*y - 3 := by
  sorry

end no_integer_solutions_l2207_220775


namespace inverse_tan_product_range_l2207_220753

/-- Given an acute-angled triangle ABC where b^2 - a^2 = ac, 
    prove that 1 / (tan A * tan B) is in the open interval (0, 1) -/
theorem inverse_tan_product_range (A B C : ℝ) (a b c : ℝ) 
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sides : b^2 - a^2 = a*c) :
  0 < (1 : ℝ) / (Real.tan A * Real.tan B) ∧ (1 : ℝ) / (Real.tan A * Real.tan B) < 1 :=
sorry

end inverse_tan_product_range_l2207_220753


namespace cylinder_volume_change_l2207_220710

theorem cylinder_volume_change (r h : ℝ) (h_positive : 0 < h) (r_positive : 0 < r) : 
  π * r^2 * h = 15 → π * (3*r)^2 * (h/2) = 67.5 := by sorry

end cylinder_volume_change_l2207_220710


namespace max_third_term_arithmetic_sequence_l2207_220764

theorem max_third_term_arithmetic_sequence (a d : ℕ) : 
  a > 0 → d > 0 → a + (a + d) + (a + 2*d) + (a + 3*d) = 50 → 
  ∀ (b e : ℕ), b > 0 → e > 0 → b + (b + e) + (b + 2*e) + (b + 3*e) = 50 → 
  (a + 2*d) ≤ 16 := by
sorry

end max_third_term_arithmetic_sequence_l2207_220764


namespace profit_starts_third_year_average_profit_plan_more_effective_l2207_220741

/-- Represents a fishing company's financial situation -/
structure FishingCompany where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrement : ℕ
  annualIncome : ℕ

/-- Calculates the cumulative expenses after n years -/
def cumulativeExpenses (company : FishingCompany) (n : ℕ) : ℕ :=
  company.initialCost + n * company.firstYearExpenses + (n * (n - 1) / 2) * company.annualExpenseIncrement

/-- Calculates the cumulative income after n years -/
def cumulativeIncome (company : FishingCompany) (n : ℕ) : ℕ :=
  n * company.annualIncome

/-- Determines if the company is profitable after n years -/
def isProfitable (company : FishingCompany) (n : ℕ) : Prop :=
  cumulativeIncome company n > cumulativeExpenses company n

/-- Represents the two selling plans -/
inductive SellingPlan
  | AverageProfit
  | TotalNetIncome

/-- Theorem: The company begins to profit in the third year -/
theorem profit_starts_third_year (company : FishingCompany) 
  (h1 : company.initialCost = 490000)
  (h2 : company.firstYearExpenses = 60000)
  (h3 : company.annualExpenseIncrement = 20000)
  (h4 : company.annualIncome = 250000) :
  isProfitable company 3 ∧ ¬isProfitable company 2 :=
sorry

/-- Theorem: The average annual profit plan is more cost-effective -/
theorem average_profit_plan_more_effective (company : FishingCompany) 
  (h1 : company.initialCost = 490000)
  (h2 : company.firstYearExpenses = 60000)
  (h3 : company.annualExpenseIncrement = 20000)
  (h4 : company.annualIncome = 250000) :
  ∃ (n m : ℕ), 
    (∀ k, cumulativeIncome company k - cumulativeExpenses company k + 180000 ≤ n) ∧
    (∀ k, cumulativeIncome company k - cumulativeExpenses company k + 90000 ≤ m) ∧
    n > m :=
sorry

end profit_starts_third_year_average_profit_plan_more_effective_l2207_220741


namespace expression_value_l2207_220784

theorem expression_value : 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 := by
  sorry

end expression_value_l2207_220784


namespace derivative_f_at_zero_l2207_220756

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    (Real.rpow (1 - 2 * x^3 * Real.sin (5 / x)) (1/3)) - 1 + x
  else 
    0

theorem derivative_f_at_zero : 
  deriv f 0 = 1 := by sorry

end derivative_f_at_zero_l2207_220756


namespace inequality_equivalence_l2207_220708

theorem inequality_equivalence (x y : ℝ) : y - x < Real.sqrt (x^2) ↔ y < 0 ∨ y < 2*x := by
  sorry

end inequality_equivalence_l2207_220708


namespace perfect_square_divisibility_l2207_220703

theorem perfect_square_divisibility (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ (n : ℕ), x = n^2 := by
  sorry

end perfect_square_divisibility_l2207_220703


namespace expression_evaluation_l2207_220728

theorem expression_evaluation : 
  (-Real.sqrt 27 + Real.cos (30 * π / 180) - (π - Real.sqrt 2) ^ 0 + (-1/2)⁻¹) = 
  -(5 * Real.sqrt 3 + 6) / 2 := by sorry

end expression_evaluation_l2207_220728


namespace challenging_polynomial_theorem_l2207_220736

/-- Defines a quadratic polynomial q(x) = x^2 + bx + c -/
def q (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- Defines the composition q(q(x)) -/
def q_comp (b c : ℝ) (x : ℝ) : ℝ := q b c (q b c x)

/-- States that q(q(x)) = 1 has exactly four distinct real solutions -/
def has_four_solutions (b c : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    q_comp b c x₁ = 1 ∧ q_comp b c x₂ = 1 ∧ q_comp b c x₃ = 1 ∧ q_comp b c x₄ = 1 ∧
    ∀ (y : ℝ), q_comp b c y = 1 → y = x₁ ∨ y = x₂ ∨ y = x₃ ∨ y = x₄

/-- The product of roots for a quadratic polynomial -/
def root_product (b c : ℝ) : ℝ := c

theorem challenging_polynomial_theorem :
  has_four_solutions (3/4) 1 ∧
  (∀ b c : ℝ, has_four_solutions b c → root_product b c ≤ root_product (3/4) 1) ∧
  q (3/4) 1 (-3) = 31/4 := by sorry

end challenging_polynomial_theorem_l2207_220736


namespace ticket_cost_correct_l2207_220792

/-- The cost of one ticket for Sebastian's art exhibit -/
def ticket_cost : ℝ := 44

/-- The number of tickets Sebastian bought -/
def num_tickets : ℕ := 3

/-- The service fee for the online transaction -/
def service_fee : ℝ := 18

/-- The total amount Sebastian paid -/
def total_paid : ℝ := 150

/-- Theorem stating that the ticket cost is correct given the conditions -/
theorem ticket_cost_correct : 
  ticket_cost * num_tickets + service_fee = total_paid :=
by sorry

end ticket_cost_correct_l2207_220792


namespace propositions_truth_l2207_220715

theorem propositions_truth (a b c : ℝ) (k : ℕ+) :
  (a > b → a^(k : ℝ) > b^(k : ℝ)) ∧
  (c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) :=
sorry

end propositions_truth_l2207_220715


namespace right_triangle_sin_cos_relation_l2207_220759

theorem right_triangle_sin_cos_relation (A B C : ℝ) (h1 : 0 < A) (h2 : A < π / 2) :
  Real.cos B = 0 → 3 * Real.sin A = 4 * Real.cos A → Real.sin A = 4 / 5 := by
  sorry

end right_triangle_sin_cos_relation_l2207_220759


namespace james_arthur_muffin_ratio_l2207_220766

theorem james_arthur_muffin_ratio :
  let arthur_muffins : ℕ := 115
  let james_muffins : ℕ := 1380
  (james_muffins : ℚ) / (arthur_muffins : ℚ) = 12 := by sorry

end james_arthur_muffin_ratio_l2207_220766


namespace point_on_x_axis_distance_to_origin_l2207_220763

/-- If a point P with coordinates (m-2, m+1) is on the x-axis, then the distance from P to the origin is 3. -/
theorem point_on_x_axis_distance_to_origin :
  ∀ m : ℝ, (m + 1 = 0) → Real.sqrt ((m - 2)^2 + 0^2) = 3 := by
  sorry

end point_on_x_axis_distance_to_origin_l2207_220763


namespace function_inequality_l2207_220751

open Real

theorem function_inequality (f : ℝ → ℝ) (a : ℝ) (h_cont : Continuous f) (h_pos : a > 0) 
  (h_fa : f a = 1) (h_ineq : ∀ x y, x > 0 → y > 0 → f x * f y + f (a / x) * f (a / y) ≤ 2 * f (x * y)) :
  ∀ x y, x > 0 → y > 0 → f x * f y ≤ f (x * y) := by
  sorry

end function_inequality_l2207_220751


namespace quadratic_sum_inequality_l2207_220747

theorem quadratic_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≤ 3*b ∧ b ≤ 3*a)
  (hac : a ≤ 3*c ∧ c ≤ 3*a)
  (had : a ≤ 3*d ∧ d ≤ 3*a)
  (hbc : b ≤ 3*c ∧ c ≤ 3*b)
  (hbd : b ≤ 3*d ∧ d ≤ 3*b)
  (hcd : c ≤ 3*d ∧ d ≤ 3*c) :
  a^2 + b^2 + c^2 + d^2 < 2*(a*b + a*c + a*d + b*c + b*d + c*d) := by
  sorry

end quadratic_sum_inequality_l2207_220747


namespace probability_to_reach_target_is_79_1024_l2207_220770

/-- Represents a step direction --/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- Represents a position on the coordinate plane --/
structure Position :=
  (x : Int) (y : Int)

/-- The probability of a single step in any direction --/
def stepProbability : ℚ := 1/4

/-- The starting position --/
def start : Position := ⟨0, 0⟩

/-- The target position --/
def target : Position := ⟨3, 1⟩

/-- The maximum number of steps allowed --/
def maxSteps : ℕ := 6

/-- Calculates the probability of reaching the target position in at most maxSteps steps --/
noncomputable def probabilityToReachTarget (start : Position) (target : Position) (maxSteps : ℕ) : ℚ :=
  sorry

/-- The main theorem to prove --/
theorem probability_to_reach_target_is_79_1024 :
  probabilityToReachTarget start target maxSteps = 79/1024 :=
by sorry

end probability_to_reach_target_is_79_1024_l2207_220770


namespace oneBlack_twoWhite_mutually_exclusive_not_contradictory_l2207_220789

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing 2 black balls and 2 white balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.Black} + 2 • {BallColor.White}

/-- The event of drawing exactly one black ball -/
def oneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two white balls -/
def twoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- The theorem stating that oneBlack and twoWhite are mutually exclusive but not contradictory -/
theorem oneBlack_twoWhite_mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(oneBlack outcome ∧ twoWhite outcome)) ∧
  (∃ outcome : DrawOutcome, ¬oneBlack outcome ∧ ¬twoWhite outcome) :=
sorry

end oneBlack_twoWhite_mutually_exclusive_not_contradictory_l2207_220789


namespace mono_increasing_range_l2207_220726

/-- A function f is monotonically increasing on ℝ -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem mono_increasing_range (f : ℝ → ℝ) (h : MonoIncreasing f) :
  ∀ m : ℝ, f (2 * m - 3) > f (-m) → m > 1 := by
  sorry

end mono_increasing_range_l2207_220726


namespace waiter_initial_customers_l2207_220761

/-- Calculates the initial number of customers in a waiter's section --/
def initial_customers (tables : ℕ) (people_per_table : ℕ) (left_customers : ℕ) : ℕ :=
  tables * people_per_table + left_customers

/-- Theorem: The initial number of customers in the waiter's section was 62 --/
theorem waiter_initial_customers :
  initial_customers 5 9 17 = 62 := by
  sorry

end waiter_initial_customers_l2207_220761


namespace next_but_one_perfect_square_l2207_220799

theorem next_but_one_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, m > x ∧ m < n ∧ ∃ k : ℕ, m = k^2) ∧ n = x + 4 * Int.sqrt x + 4 :=
sorry

end next_but_one_perfect_square_l2207_220799


namespace gauss_candy_remaining_l2207_220780

/-- The number of lollipops that remain after packaging -/
def remaining_lollipops (total : ℕ) (per_package : ℕ) : ℕ :=
  total % per_package

/-- Theorem stating the number of remaining lollipops for the Gauss Candy Company problem -/
theorem gauss_candy_remaining : remaining_lollipops 8362 12 = 10 := by
  sorry

end gauss_candy_remaining_l2207_220780


namespace f_max_value_l2207_220794

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

/-- The maximum value of f(x) is 22 -/
theorem f_max_value : ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l2207_220794


namespace pool_count_l2207_220716

/-- The number of pools in two stores -/
def total_pools (store_a : ℕ) (store_b : ℕ) : ℕ :=
  store_a + store_b

/-- Theorem stating the total number of pools given the conditions -/
theorem pool_count : 
  let store_a := 200
  let store_b := 3 * store_a
  total_pools store_a store_b = 800 := by
sorry

end pool_count_l2207_220716


namespace find_X_l2207_220734

theorem find_X : ∃ X : ℚ, (X + 43 / 151) * 151 = 2912 ∧ X = 19 := by
  sorry

end find_X_l2207_220734


namespace dvd_rental_cost_l2207_220707

theorem dvd_rental_cost (total_dvds : ℕ) (total_cost : ℝ) (known_dvds : ℕ) (known_cost : ℝ) : 
  total_dvds = 7 → 
  total_cost = 12.6 → 
  known_dvds = 3 → 
  known_cost = 1.5 → 
  total_cost - (known_dvds * known_cost) = 8.1 :=
by sorry

end dvd_rental_cost_l2207_220707


namespace no_rational_solution_sqrt2_equation_l2207_220730

theorem no_rational_solution_sqrt2_equation :
  ∀ (x y z t : ℚ), (x + y * Real.sqrt 2)^2 + (z + t * Real.sqrt 2)^2 ≠ 5 + 4 * Real.sqrt 2 := by
  sorry

end no_rational_solution_sqrt2_equation_l2207_220730


namespace monotonic_f_range_a_l2207_220765

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + a*x + a/4 else a^x

/-- Theorem stating the range of a for monotonically increasing f(x) -/
theorem monotonic_f_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 4 := by
  sorry

end monotonic_f_range_a_l2207_220765


namespace inequality_solution_set_l2207_220774

theorem inequality_solution_set (x : ℝ) :
  -x^2 + 4*x + 5 < 0 ↔ x > 5 ∨ x < -1 := by sorry

end inequality_solution_set_l2207_220774


namespace division_problem_l2207_220738

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 141 →
  quotient = 8 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 17 := by
sorry

end division_problem_l2207_220738


namespace gcd_polynomial_and_b_l2207_220768

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 350 * k) :
  Int.gcd (2 * b^3 + 3 * b^2 + 5 * b + 70) b = 70 := by
  sorry

end gcd_polynomial_and_b_l2207_220768


namespace farm_chickens_count_l2207_220724

theorem farm_chickens_count (chicken_A duck_A chicken_B duck_B : ℕ) : 
  chicken_A + duck_A = 625 →
  chicken_B + duck_B = 748 →
  chicken_B = (chicken_A * 124) / 100 →
  duck_A = (duck_B * 85) / 100 →
  chicken_B = 248 := by
  sorry

end farm_chickens_count_l2207_220724


namespace ellipse_line_intersection_l2207_220729

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

-- Define the line l
def line_l (x y : ℝ) (k₁ : ℝ) : Prop := y = k₁ * (x + 2)

-- Define the point M
def point_M : ℝ × ℝ := (-2, 0)

-- Define the origin O
def point_O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem ellipse_line_intersection 
  (P₁ P₂ P : ℝ × ℝ) 
  (k₁ k₂ : ℝ) 
  (h₁ : k₁ ≠ 0)
  (h₂ : ellipse P₁.1 P₁.2)
  (h₃ : ellipse P₂.1 P₂.2)
  (h₄ : line_l P₁.1 P₁.2 k₁)
  (h₅ : line_l P₂.1 P₂.2 k₁)
  (h₆ : P = ((P₁.1 + P₂.1) / 2, (P₁.2 + P₂.2) / 2))
  (h₇ : k₂ = P.2 / P.1) :
  k₁ * k₂ = -1/2 := by sorry

end ellipse_line_intersection_l2207_220729


namespace consecutive_integers_sqrt_17_l2207_220714

theorem consecutive_integers_sqrt_17 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 17) → (Real.sqrt 17 < b) → (a + b = 9) := by
  sorry

end consecutive_integers_sqrt_17_l2207_220714


namespace tv_discount_theorem_l2207_220748

/-- Represents the price of a TV with successive discounts -/
def discounted_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price of a TV after successive discounts is 63% of the original price -/
theorem tv_discount_theorem :
  let original_price : ℝ := 450
  let discount1 : ℝ := 0.30
  let discount2 : ℝ := 0.10
  let final_price := discounted_price original_price discount1 discount2
  final_price / original_price = 0.63 := by
  sorry


end tv_discount_theorem_l2207_220748


namespace square_inequality_l2207_220787

theorem square_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end square_inequality_l2207_220787


namespace equal_area_rectangles_l2207_220709

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def areaOfRectangle (r : Rectangle) : ℝ :=
  r.length * r.width

theorem equal_area_rectangles (carol_rect jordan_rect : Rectangle) :
  carol_rect.length = 5 →
  carol_rect.width = 24 →
  jordan_rect.length = 12 →
  areaOfRectangle carol_rect = areaOfRectangle jordan_rect →
  jordan_rect.width = 10 :=
by
  sorry


end equal_area_rectangles_l2207_220709


namespace sum_37_29_base5_l2207_220781

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 5 representation --/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem sum_37_29_base5 :
  addBase5 (toBase5 37) (toBase5 29) = [2, 3, 1] :=
by sorry

end sum_37_29_base5_l2207_220781


namespace minyoung_line_size_l2207_220742

/-- Represents a line of people ordered by height -/
structure HeightLine where
  people : ℕ
  tallestToShortest : Fin people → Fin people

/-- A person's position from the tallest in the line -/
def positionFromTallest (line : HeightLine) (person : Fin line.people) : ℕ :=
  line.tallestToShortest person + 1

/-- A person's position from the shortest in the line -/
def positionFromShortest (line : HeightLine) (person : Fin line.people) : ℕ :=
  line.people - line.tallestToShortest person

theorem minyoung_line_size
  (line : HeightLine)
  (minyoung : Fin line.people)
  (h1 : positionFromTallest line minyoung = 2)
  (h2 : positionFromShortest line minyoung = 4) :
  line.people = 5 := by
  sorry

end minyoung_line_size_l2207_220742


namespace sum_of_rationals_is_rational_l2207_220723

theorem sum_of_rationals_is_rational (r₁ r₂ : ℚ) : ∃ (q : ℚ), r₁ + r₂ = q := by
  sorry

end sum_of_rationals_is_rational_l2207_220723


namespace total_oranges_l2207_220718

def oranges_per_box : ℝ := 10.0
def boxes_packed : ℝ := 2650.0

theorem total_oranges : oranges_per_box * boxes_packed = 26500.0 := by
  sorry

end total_oranges_l2207_220718


namespace seed_distribution_l2207_220755

theorem seed_distribution (n : ℕ) : 
  (n * (n + 1) / 2 : ℚ) + 100 = n * (3 * n + 1) / 2 → n = 10 := by
  sorry

end seed_distribution_l2207_220755


namespace women_stockbrokers_increase_l2207_220754

/-- Calculates the final number of women stockbrokers after a percentage increase -/
def final_number (initial : ℕ) (percent_increase : ℕ) : ℕ :=
  initial + (initial * percent_increase) / 100

/-- Theorem: Given 10,000 initial women stockbrokers and a 100% increase, 
    the final number is 20,000 -/
theorem women_stockbrokers_increase : 
  final_number 10000 100 = 20000 := by sorry

end women_stockbrokers_increase_l2207_220754


namespace marbles_selection_count_l2207_220717

def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5

theorem marbles_selection_count :
  (special_marbles * (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - 1))) = 1320 := by
  sorry

end marbles_selection_count_l2207_220717


namespace square_room_perimeter_l2207_220773

theorem square_room_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 500 → perimeter = 40 * Real.sqrt 5 := by sorry

end square_room_perimeter_l2207_220773


namespace charge_difference_l2207_220743

/-- Represents the pricing scheme of a psychologist -/
structure PricingScheme where
  firstHourCharge : ℝ
  additionalHourCharge : ℝ
  fiveHourTotal : ℝ
  twoHourTotal : ℝ

/-- Theorem stating the difference in charges for a specific pricing scheme -/
theorem charge_difference (p : PricingScheme) 
  (h1 : p.firstHourCharge > p.additionalHourCharge)
  (h2 : p.firstHourCharge + 4 * p.additionalHourCharge = p.fiveHourTotal)
  (h3 : p.firstHourCharge + p.additionalHourCharge = p.twoHourTotal)
  (h4 : p.fiveHourTotal = 350)
  (h5 : p.twoHourTotal = 161) : 
  p.firstHourCharge - p.additionalHourCharge = 35 := by
  sorry

end charge_difference_l2207_220743


namespace total_components_total_components_proof_l2207_220704

/-- The total number of components of types A, B, and C is 900. -/
theorem total_components : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun (total_B : ℕ) (total_C : ℕ) (sample_size : ℕ) (sample_A : ℕ) (sample_C : ℕ) (total : ℕ) =>
    total_B = 300 →
    total_C = 200 →
    sample_size = 45 →
    sample_A = 20 →
    sample_C = 10 →
    total = 900

/-- Proof of the theorem -/
theorem total_components_proof :
  total_components 300 200 45 20 10 900 := by
  sorry

end total_components_total_components_proof_l2207_220704


namespace lisa_walks_2100_meters_l2207_220797

/-- Calculates the total distance Lisa walks over two days given her usual pace and terrain conditions -/
def lisa_total_distance (usual_pace : ℝ) : ℝ :=
  let day1_morning := usual_pace * 60
  let day1_evening := (usual_pace * 0.7) * 60
  let day2_morning := (usual_pace * 1.2) * 60
  let day2_evening := (usual_pace * 0.6) * 60
  day1_morning + day1_evening + day2_morning + day2_evening

/-- Theorem stating that Lisa's total distance over two days is 2100 meters -/
theorem lisa_walks_2100_meters :
  lisa_total_distance 10 = 2100 := by
  sorry

#eval lisa_total_distance 10

end lisa_walks_2100_meters_l2207_220797


namespace integral_x_over_sqrt_x_squared_plus_one_l2207_220791

theorem integral_x_over_sqrt_x_squared_plus_one (x : ℝ) :
  deriv (λ x => Real.sqrt (x^2 + 1)) x = x / Real.sqrt (x^2 + 1) := by
  sorry

end integral_x_over_sqrt_x_squared_plus_one_l2207_220791


namespace shaded_area_calculation_l2207_220795

theorem shaded_area_calculation (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) : 
  square_side = 40 →
  triangle_base = 30 →
  triangle_height = 30 →
  square_side * square_side - 2 * (1/2 * triangle_base * triangle_height) = 700 := by
sorry

end shaded_area_calculation_l2207_220795


namespace count_pairs_with_harmonic_mean_5_20_l2207_220700

/-- The number of ordered pairs of positive integers with harmonic mean 5^20 -/
def count_pairs : ℕ := 20

/-- Harmonic mean of two numbers -/
def harmonic_mean (x y : ℕ) : ℚ := 2 * x * y / (x + y)

/-- Predicate for valid pairs -/
def is_valid_pair (x y : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ x < y ∧ harmonic_mean x y = 5^20

/-- The main theorem -/
theorem count_pairs_with_harmonic_mean_5_20 :
  (∃ (S : Finset (ℕ × ℕ)), S.card = count_pairs ∧
    ∀ (p : ℕ × ℕ), p ∈ S ↔ is_valid_pair p.1 p.2) :=
sorry

end count_pairs_with_harmonic_mean_5_20_l2207_220700


namespace translated_minimum_point_l2207_220779

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 4

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - 3) + 4

-- Theorem statement
theorem translated_minimum_point :
  ∃ (x_min : ℝ), (∀ (x : ℝ), g x_min ≤ g x) ∧ g x_min = 0 ∧ x_min = 2 := by
  sorry

end translated_minimum_point_l2207_220779


namespace sum_of_y_values_l2207_220762

theorem sum_of_y_values (x y z : ℝ) : 
  x + y = 7 → 
  x * z = -180 → 
  (x + y + z)^2 = 4 → 
  ∃ y₁ y₂ : ℝ, 
    (x + y₁ = 7 ∧ x * z = -180 ∧ (x + y₁ + z)^2 = 4) ∧
    (x + y₂ = 7 ∧ x * z = -180 ∧ (x + y₂ + z)^2 = 4) ∧
    y₁ ≠ y₂ ∧
    -(y₁ + y₂) = 42 :=
by sorry

end sum_of_y_values_l2207_220762


namespace prize_money_problem_l2207_220782

/-- The prize money problem -/
theorem prize_money_problem (total_students : Nat) (team_members : Nat) (member_prize : Nat) (extra_prize : Nat) :
  total_students = 10 →
  team_members = 9 →
  member_prize = 200 →
  extra_prize = 90 →
  ∃ (captain_prize : Nat),
    captain_prize = extra_prize + (captain_prize + team_members * member_prize) / total_students ∧
    captain_prize = 300 := by
  sorry

end prize_money_problem_l2207_220782


namespace inequality_system_solution_l2207_220711

theorem inequality_system_solution (x : ℝ) : 
  2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4 := by
  sorry

end inequality_system_solution_l2207_220711


namespace allowance_proof_l2207_220757

/-- The student's bi-weekly allowance -/
def allowance : ℝ := 233.89

/-- The amount left after all spending -/
def remaining : ℝ := 2.10

theorem allowance_proof :
  allowance * (4/9) * (1/3) * (4/11) * (1/6) = remaining := by sorry

end allowance_proof_l2207_220757


namespace a_3_value_l2207_220725

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = -3

theorem a_3_value (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 7 → a 3 = 1 := by
  sorry

end a_3_value_l2207_220725


namespace smallest_n_for_sqrt_inequality_l2207_220778

theorem smallest_n_for_sqrt_inequality :
  ∀ n : ℕ, n > 0 → (Real.sqrt (5 * n) - Real.sqrt (5 * n - 4) < 0.01) ↔ n ≥ 8001 :=
by sorry

end smallest_n_for_sqrt_inequality_l2207_220778


namespace unique_solution_is_one_l2207_220722

/-- A function satisfying f(x)f(y) = f(x-y) for all x and y, and is nonzero at some point -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y, f x * f y = f (x - y)) ∧ (∃ x, f x ≠ 0)

/-- The constant function 1 is the unique solution to the functional equation -/
theorem unique_solution_is_one :
  ∀ f : ℝ → ℝ, FunctionalEquation f → (∀ x, f x = 1) :=
by sorry

end unique_solution_is_one_l2207_220722


namespace largest_t_value_l2207_220706

theorem largest_t_value : ∃ t_max : ℚ, 
  (∀ t : ℚ, (16 * t^2 - 40 * t + 15) / (4 * t - 3) + 7 * t = 5 * t + 2 → t ≤ t_max) ∧ 
  (16 * t_max^2 - 40 * t_max + 15) / (4 * t_max - 3) + 7 * t_max = 5 * t_max + 2 ∧
  t_max = 7/4 := by
  sorry

end largest_t_value_l2207_220706


namespace shirts_produced_theorem_l2207_220752

/-- An industrial machine that produces shirts. -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  minutes_worked_today : ℕ

/-- Calculates the total number of shirts produced by the machine today. -/
def shirts_produced_today (machine : ShirtMachine) : ℕ :=
  machine.shirts_per_minute * machine.minutes_worked_today

/-- Theorem stating that a machine producing 6 shirts per minute working for 12 minutes produces 72 shirts. -/
theorem shirts_produced_theorem (machine : ShirtMachine)
    (h1 : machine.shirts_per_minute = 6)
    (h2 : machine.minutes_worked_today = 12) :
    shirts_produced_today machine = 72 := by
  sorry

end shirts_produced_theorem_l2207_220752


namespace seashells_sum_l2207_220758

/-- The number of seashells found by Joan, Jessica, and Jeremy -/
def joan_seashells : ℕ := 6
def jessica_seashells : ℕ := 8
def jeremy_seashells : ℕ := 12

/-- The total number of seashells found by Joan, Jessica, and Jeremy -/
def total_seashells : ℕ := joan_seashells + jessica_seashells + jeremy_seashells

theorem seashells_sum : total_seashells = 26 := by
  sorry

end seashells_sum_l2207_220758


namespace school_students_count_l2207_220712

theorem school_students_count :
  let blue_percent : ℝ := 0.45
  let red_percent : ℝ := 0.23
  let green_percent : ℝ := 0.15
  let other_count : ℕ := 102
  let total_count : ℕ := 600
  blue_percent + red_percent + green_percent + (other_count : ℝ) / total_count = 1 ∧
  (other_count : ℝ) / total_count = 1 - (blue_percent + red_percent + green_percent) :=
by sorry

end school_students_count_l2207_220712


namespace ceiling_sqrt_900_l2207_220783

theorem ceiling_sqrt_900 : ⌈Real.sqrt 900⌉ = 30 := by
  sorry

end ceiling_sqrt_900_l2207_220783


namespace solution_pairs_l2207_220776

theorem solution_pairs : 
  ∀ x y : ℝ, x - y = 10 ∧ x^2 + y^2 = 100 → (x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0) :=
by sorry

end solution_pairs_l2207_220776
