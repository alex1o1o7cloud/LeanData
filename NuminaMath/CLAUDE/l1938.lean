import Mathlib

namespace sin_neg_pi_third_l1938_193846

theorem sin_neg_pi_third : Real.sin (-π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end sin_neg_pi_third_l1938_193846


namespace sum_of_fractions_l1938_193883

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end sum_of_fractions_l1938_193883


namespace quadratic_equation_solution_l1938_193810

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (1 + Real.sqrt 17) / 4
  let x₂ : ℝ := (1 - Real.sqrt 17) / 4
  ∀ x : ℝ, 2 * x^2 - x = 2 ↔ (x = x₁ ∨ x = x₂) := by
sorry

end quadratic_equation_solution_l1938_193810


namespace total_tax_deduction_in_cents_l1938_193853

-- Define the hourly wage in dollars
def hourly_wage : ℝ := 25

-- Define the local tax rate as a percentage
def local_tax_rate : ℝ := 2

-- Define the state tax rate as a percentage
def state_tax_rate : ℝ := 0.5

-- Define the conversion rate from dollars to cents
def dollars_to_cents : ℝ := 100

-- Theorem statement
theorem total_tax_deduction_in_cents :
  (hourly_wage * dollars_to_cents) * (local_tax_rate / 100 + state_tax_rate / 100) = 62.5 := by
  sorry

end total_tax_deduction_in_cents_l1938_193853


namespace circle_ratio_after_diameter_increase_l1938_193842

/-- Theorem: For any circle with an initial diameter of 2r units, 
if the diameter is increased by 4 units, 
the ratio of the new circumference to the new diameter is equal to π. -/
theorem circle_ratio_after_diameter_increase (r : ℝ) (r_pos : r > 0) : 
  let initial_diameter : ℝ := 2 * r
  let new_diameter : ℝ := initial_diameter + 4
  let new_circumference : ℝ := 2 * π * (r + 2)
  new_circumference / new_diameter = π :=
by sorry

end circle_ratio_after_diameter_increase_l1938_193842


namespace boys_ages_l1938_193852

theorem boys_ages (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 29 →
  age1 = age2 →
  age3 = 11 →
  age1 = 9 := by
sorry

end boys_ages_l1938_193852


namespace binomial_10_5_l1938_193832

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end binomial_10_5_l1938_193832


namespace shot_put_surface_area_l1938_193849

/-- The surface area of a sphere with diameter 5 inches is 25π square inches. -/
theorem shot_put_surface_area :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 25 * Real.pi := by
  sorry

end shot_put_surface_area_l1938_193849


namespace lcm_gcf_ratio_l1938_193843

theorem lcm_gcf_ratio (a b : ℕ) (ha : a = 210) (hb : b = 462) : 
  Nat.lcm a b / Nat.gcd a b = 55 := by
sorry

end lcm_gcf_ratio_l1938_193843


namespace A_union_B_eq_A_l1938_193825

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (x + 1) * (x - 4) < 0}
def B : Set ℝ := {x : ℝ | Real.log x < 1}

-- State the theorem
theorem A_union_B_eq_A : A ∪ B = A := by sorry

end A_union_B_eq_A_l1938_193825


namespace aqua_park_earnings_l1938_193899

def admission_cost : ℚ := 12
def tour_cost : ℚ := 6
def meal_cost : ℚ := 10
def souvenir_cost : ℚ := 8

def group1_size : ℕ := 10
def group2_size : ℕ := 15
def group3_size : ℕ := 8

def group1_discount_rate : ℚ := 0.10
def group2_meal_discount_rate : ℚ := 0.05

def group1_total (admission_cost tour_cost meal_cost souvenir_cost : ℚ) (group_size : ℕ) (discount_rate : ℚ) : ℚ :=
  (1 - discount_rate) * (admission_cost + tour_cost + meal_cost + souvenir_cost) * group_size

def group2_total (admission_cost meal_cost : ℚ) (group_size : ℕ) (meal_discount_rate : ℚ) : ℚ :=
  admission_cost * group_size + (1 - meal_discount_rate) * meal_cost * group_size

def group3_total (admission_cost tour_cost souvenir_cost : ℚ) (group_size : ℕ) : ℚ :=
  (admission_cost + tour_cost + souvenir_cost) * group_size

theorem aqua_park_earnings : 
  group1_total admission_cost tour_cost meal_cost souvenir_cost group1_size group1_discount_rate +
  group2_total admission_cost meal_cost group2_size group2_meal_discount_rate +
  group3_total admission_cost tour_cost souvenir_cost group3_size = 854.5 := by
  sorry

end aqua_park_earnings_l1938_193899


namespace forty_percent_of_number_l1938_193801

theorem forty_percent_of_number (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 15 → 
  (40/100 : ℝ) * N = 180 := by
  sorry

end forty_percent_of_number_l1938_193801


namespace book_has_2000_pages_l1938_193811

/-- The number of pages Juan reads per hour -/
def pages_per_hour : ℕ := 250

/-- The time it takes Juan to grab lunch (in hours) -/
def lunch_time : ℕ := 4

/-- The time it takes Juan to read the book (in hours) -/
def reading_time : ℕ := 2 * lunch_time

/-- The total number of pages in the book -/
def book_pages : ℕ := pages_per_hour * reading_time

theorem book_has_2000_pages : book_pages = 2000 := by
  sorry

end book_has_2000_pages_l1938_193811


namespace total_balls_in_box_l1938_193813

theorem total_balls_in_box (yellow_balls : ℕ) (prob_yellow : ℚ) (total_balls : ℕ) : 
  yellow_balls = 6 → 
  prob_yellow = 1 / 9 → 
  prob_yellow = yellow_balls / total_balls → 
  total_balls = 54 := by sorry

end total_balls_in_box_l1938_193813


namespace notebook_cost_l1938_193851

theorem notebook_cost (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 380)
  (eq2 : 3 * x + 6 * y = 354) :
  x = 48 := by
sorry

end notebook_cost_l1938_193851


namespace omega_value_l1938_193890

/-- Given a function f(x) = 3sin(ωx) - √3cos(ωx) where ω > 0 and x ∈ ℝ,
    if f(x) is monotonically increasing in (-ω, 2ω) and
    symmetric about x = -ω, then ω = √(3π)/3 -/
theorem omega_value (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)
  (∀ x ∈ Set.Ioo (-ω) (2*ω), StrictMonoOn f (Set.Ioo (-ω) (2*ω))) →
  (∀ x, f (x - ω) = -f (-x - ω)) →
  ω = Real.sqrt (3 * Real.pi) / 3 := by
sorry

end omega_value_l1938_193890


namespace zoo_animals_count_l1938_193830

/-- The number of tiger enclosures in the zoo -/
def tiger_enclosures : ℕ := 4

/-- The number of zebra enclosures behind each tiger enclosure -/
def zebra_enclosures_per_tiger : ℕ := 2

/-- The ratio of giraffe enclosures to zebra enclosures -/
def giraffe_to_zebra_ratio : ℕ := 3

/-- The number of tigers in each tiger enclosure -/
def tigers_per_enclosure : ℕ := 4

/-- The number of zebras in each zebra enclosure -/
def zebras_per_enclosure : ℕ := 10

/-- The number of giraffes in each giraffe enclosure -/
def giraffes_per_enclosure : ℕ := 2

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 144

theorem zoo_animals_count :
  tiger_enclosures * tigers_per_enclosure +
  (tiger_enclosures * zebra_enclosures_per_tiger) * zebras_per_enclosure +
  (tiger_enclosures * zebra_enclosures_per_tiger * giraffe_to_zebra_ratio) * giraffes_per_enclosure =
  total_animals := by sorry

end zoo_animals_count_l1938_193830


namespace pyramid_z_value_l1938_193845

/-- Represents a three-level pyramid structure -/
structure Pyramid where
  z : ℕ
  x : ℕ
  y : ℕ
  bottom_left : ℕ
  bottom_middle : ℕ
  bottom_right : ℕ

/-- Checks if the pyramid satisfies the given conditions -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.bottom_left = p.z * p.x ∧
  p.bottom_middle = p.x * p.y ∧
  p.bottom_right = p.y * p.z

theorem pyramid_z_value :
  ∀ p : Pyramid,
    is_valid_pyramid p →
    p.bottom_left = 8 →
    p.bottom_middle = 40 →
    p.bottom_right = 10 →
    p.z = 4 :=
by
  sorry


end pyramid_z_value_l1938_193845


namespace max_value_of_function_l1938_193837

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 3/2) : 
  x * (3 - 2*x) ≤ 9/8 ∧ ∃ x₀, 0 < x₀ ∧ x₀ < 3/2 ∧ x₀ * (3 - 2*x₀) = 9/8 := by
  sorry

end max_value_of_function_l1938_193837


namespace min_sin_cos_expression_l1938_193867

theorem min_sin_cos_expression (A : Real) : 
  let f := λ x : Real => Real.sin (x / 2) - Real.sqrt 3 * Real.cos (x / 2)
  ∃ m : Real, (∀ x, f x ≥ m) ∧ f (-π/3) = m :=
sorry

end min_sin_cos_expression_l1938_193867


namespace tournament_rankings_l1938_193877

/-- Represents a team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match :=
(team1 : Team)
(team2 : Team)

/-- Represents the tournament structure -/
structure Tournament :=
(saturday_matches : List Match)
(sunday_winners_round_robin : List Team)
(sunday_losers_round_robin : List Team)

/-- Represents a ranking of teams -/
def Ranking := List Team

/-- Function to calculate the number of possible rankings -/
def number_of_rankings (t : Tournament) : Nat :=
  6 * 6

/-- The main theorem stating the number of possible ranking sequences -/
theorem tournament_rankings (t : Tournament) :
  number_of_rankings t = 36 :=
sorry

end tournament_rankings_l1938_193877


namespace difference_of_squares_23_15_l1938_193893

theorem difference_of_squares_23_15 : (23 + 15)^2 - (23 - 15)^2 = 304 := by
  sorry

end difference_of_squares_23_15_l1938_193893


namespace sum_of_coefficients_l1938_193895

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) →
  a₀ + a₁ + a₂ + a₃ = 27 := by
  sorry

end sum_of_coefficients_l1938_193895


namespace modular_inverse_32_mod_33_l1938_193805

theorem modular_inverse_32_mod_33 : ∃ x : ℕ, x ≤ 32 ∧ (32 * x) % 33 = 1 :=
by
  -- The proof goes here
  sorry

end modular_inverse_32_mod_33_l1938_193805


namespace complex_magnitude_problem_l1938_193888

theorem complex_magnitude_problem (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : Complex.abs (a + b + c) = 1)
  (h5 : Complex.abs (a - b) = Complex.abs (a - c))
  (h6 : b ≠ c) :
  Complex.abs (a + b) * Complex.abs (a + c) = 2 := by
  sorry

end complex_magnitude_problem_l1938_193888


namespace trigonometric_identity_l1938_193804

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) 
  (eq : (Real.sin θ ^ 6 / a ^ 2) + (Real.cos θ ^ 6 / b ^ 2) = 1 / (a + b)) :
  (Real.sin θ ^ 12 / a ^ 5) + (Real.cos θ ^ 12 / b ^ 5) = 1 / (a + b) ^ 5 := by
sorry

end trigonometric_identity_l1938_193804


namespace spending_fraction_is_three_fourths_l1938_193875

/-- Represents a person's monthly savings and spending habits -/
structure SavingsHabit where
  monthly_salary : ℝ
  savings_fraction : ℝ
  spending_fraction : ℝ
  savings_fraction_nonneg : 0 ≤ savings_fraction
  spending_fraction_nonneg : 0 ≤ spending_fraction
  fractions_sum_to_one : savings_fraction + spending_fraction = 1

/-- The theorem stating that if yearly savings are 4 times monthly spending, 
    then the spending fraction is 3/4 -/
theorem spending_fraction_is_three_fourths 
  (habit : SavingsHabit) 
  (yearly_savings_eq_four_times_monthly_spending : 
    12 * habit.savings_fraction * habit.monthly_salary = 
    4 * habit.spending_fraction * habit.monthly_salary) :
  habit.spending_fraction = 3/4 := by
  sorry

end spending_fraction_is_three_fourths_l1938_193875


namespace inequality_system_solution_l1938_193898

theorem inequality_system_solution : 
  {x : ℝ | x + 1 > 0 ∧ -2 * x ≤ 6} = {x : ℝ | x > -1} := by
  sorry

end inequality_system_solution_l1938_193898


namespace count_multiples_theorem_l1938_193872

/-- The count of positive integers not exceeding 500 that are multiples of 2 or 5 but not 6 -/
def count_multiples : ℕ := sorry

/-- The upper bound of the range -/
def upper_bound : ℕ := 500

/-- Predicate for a number being a multiple of 2 or 5 but not 6 -/
def is_valid_multiple (n : ℕ) : Prop :=
  n ≤ upper_bound ∧ (n % 2 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0

theorem count_multiples_theorem : count_multiples = 217 := by sorry

end count_multiples_theorem_l1938_193872


namespace lee_cookies_l1938_193808

/-- Given that Lee can make 24 cookies with 4 cups of flour, 
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (flour : ℚ) : ℚ :=
  (24 / 4) * flour

/-- Theorem stating that Lee can make 36 cookies with 6 cups of flour. -/
theorem lee_cookies : cookies_from_flour 6 = 36 := by
  sorry

end lee_cookies_l1938_193808


namespace consecutive_integers_sum_l1938_193848

theorem consecutive_integers_sum (x y z : ℤ) : 
  (x = y + 1) → 
  (y = z + 1) → 
  (x > y) → 
  (y > z) → 
  (2 * x + 3 * y + 3 * z = 5 * y + 11) → 
  z = 3 := by
sorry

end consecutive_integers_sum_l1938_193848


namespace right_triangle_perimeter_l1938_193881

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 200 →
  b = 20 →
  a^2 + b^2 = c^2 →
  a + b + c = 40 + 20 * Real.sqrt 2 := by
sorry

end right_triangle_perimeter_l1938_193881


namespace odd_function_half_value_l1938_193836

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

theorem odd_function_half_value (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → f a (1/2) = 1 := by
  sorry

end odd_function_half_value_l1938_193836


namespace fixed_cost_calculation_publishing_company_fixed_cost_l1938_193838

theorem fixed_cost_calculation (marketing_cost : ℕ) (selling_price : ℕ) (break_even_quantity : ℕ) : ℕ :=
  let net_revenue_per_book := selling_price - marketing_cost
  let fixed_cost := net_revenue_per_book * break_even_quantity
  fixed_cost

theorem publishing_company_fixed_cost :
  fixed_cost_calculation 4 9 10000 = 50000 := by
  sorry

end fixed_cost_calculation_publishing_company_fixed_cost_l1938_193838


namespace weight_of_ten_moles_example_l1938_193874

/-- Calculates the weight of a given number of moles of a compound with a known molecular weight. -/
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Proves that the weight of 10 moles of a compound with a molecular weight of 1080 grams/mole is 10800 grams. -/
theorem weight_of_ten_moles_example : weight_of_compound 10 1080 = 10800 := by
  sorry

end weight_of_ten_moles_example_l1938_193874


namespace power_fraction_simplification_l1938_193866

theorem power_fraction_simplification :
  (16 ^ 10 * 8 ^ 6) / (4 ^ 22) = 2 ^ 14 := by sorry

end power_fraction_simplification_l1938_193866


namespace sine_absolute_value_integral_l1938_193894

theorem sine_absolute_value_integral : ∫ x in (0)..(2 * Real.pi), |Real.sin x| = 4 := by
  sorry

end sine_absolute_value_integral_l1938_193894


namespace shed_blocks_count_l1938_193806

/-- Represents the dimensions of a rectangular structure -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular structure -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Represents the specifications of the shed -/
structure ShedSpecs where
  outer : Dimensions
  wallThickness : ℝ

/-- Calculates the inner dimensions of the shed -/
def innerDimensions (s : ShedSpecs) : Dimensions :=
  { length := s.outer.length - 2 * s.wallThickness,
    width := s.outer.width - 2 * s.wallThickness,
    height := s.outer.height - 2 * s.wallThickness }

/-- Calculates the number of blocks used in the shed construction -/
def blocksUsed (s : ShedSpecs) : ℝ :=
  volume s.outer - volume (innerDimensions s)

/-- The main theorem stating the number of blocks used in the shed construction -/
theorem shed_blocks_count :
  let shedSpecs : ShedSpecs := {
    outer := { length := 15, width := 12, height := 7 },
    wallThickness := 1.5
  }
  blocksUsed shedSpecs = 828 := by sorry

end shed_blocks_count_l1938_193806


namespace not_perfect_square_exists_l1938_193892

theorem not_perfect_square_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  ∃ n : ℕ+, ¬ ∃ k : ℕ, (a^n.val - 1) * (b^n.val - 1) = k^2 := by
  sorry

end not_perfect_square_exists_l1938_193892


namespace pq_ratio_implies_pg_ps_ratio_l1938_193812

/-- Triangle PQR with angle bisector PS intersecting MN at G -/
structure Triangle (P Q R S M N G : ℝ × ℝ) :=
  (M_on_PQ : ∃ t : ℝ, M = (1 - t) • P + t • Q ∧ 0 ≤ t ∧ t ≤ 1)
  (N_on_PR : ∃ t : ℝ, N = (1 - t) • P + t • R ∧ 0 ≤ t ∧ t ≤ 1)
  (S_angle_bisector : ∃ t : ℝ, S = (1 - t) • P + t • ((Q + R) / 2) ∧ 0 < t)
  (G_on_MN : ∃ t : ℝ, G = (1 - t) • M + t • N ∧ 0 ≤ t ∧ t ≤ 1)
  (G_on_PS : ∃ t : ℝ, G = (1 - t) • P + t • S ∧ 0 ≤ t ∧ t ≤ 1)

/-- The main theorem -/
theorem pq_ratio_implies_pg_ps_ratio 
  (P Q R S M N G : ℝ × ℝ) 
  (h : Triangle P Q R S M N G) 
  (hPM_MQ : ∃ (t : ℝ), M = (1 - t) • P + t • Q ∧ t = 1/4) 
  (hPN_NR : ∃ (t : ℝ), N = (1 - t) • P + t • R ∧ t = 1/4) :
  ∃ (t : ℝ), G = (1 - t) • P + t • S ∧ t = 5/18 :=
sorry

end pq_ratio_implies_pg_ps_ratio_l1938_193812


namespace function_equivalence_l1938_193880

theorem function_equivalence (x : ℝ) (h : x ≠ 0) :
  (2 * x + 3) / x = 2 + 3 / x := by sorry

end function_equivalence_l1938_193880


namespace triangle_tangent_segment_length_l1938_193800

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point on a line segment -/
structure PointOnSegment where
  segment : ℝ
  position : ℝ

/-- Checks if a point is on the incircle of a triangle -/
def isOnIncircle (t : Triangle) (p : PointOnSegment) : Prop := sorry

/-- Checks if a line segment is tangent to the incircle of a triangle -/
def isTangentToIncircle (t : Triangle) (p1 p2 : PointOnSegment) : Prop := sorry

/-- Main theorem -/
theorem triangle_tangent_segment_length 
  (t : Triangle) 
  (x y : PointOnSegment) :
  t.a = 19 ∧ t.b = 20 ∧ t.c = 21 →
  x.segment = t.a ∧ y.segment = t.c →
  x.position + y.position = t.a →
  isTangentToIncircle t x y →
  x.position = 67 / 10 := by
  sorry

end triangle_tangent_segment_length_l1938_193800


namespace geometric_progression_values_l1938_193829

theorem geometric_progression_values (p : ℝ) : 
  (4*p + 5 ≠ 0 ∧ 2*p ≠ 0 ∧ |p - 3| ≠ 0) ∧
  (2*p)^2 = (4*p + 5) * |p - 3| ↔ 
  p = -1 ∨ p = 15/8 := by sorry

end geometric_progression_values_l1938_193829


namespace import_tax_percentage_l1938_193854

/-- The import tax percentage calculation problem -/
theorem import_tax_percentage 
  (total_value : ℝ)
  (tax_threshold : ℝ)
  (tax_paid : ℝ)
  (h1 : total_value = 2570)
  (h2 : tax_threshold = 1000)
  (h3 : tax_paid = 109.90) :
  (tax_paid / (total_value - tax_threshold)) * 100 = 7 := by
sorry

end import_tax_percentage_l1938_193854


namespace one_student_owns_all_pets_l1938_193897

/-- Represents the pet ownership distribution in Sara's class -/
structure PetOwnership where
  total : ℕ
  dog_owners : ℕ
  cat_owners : ℕ
  bird_owners : ℕ
  no_pets : ℕ
  just_dogs : ℕ
  just_cats : ℕ
  just_birds : ℕ
  dogs_and_cats : ℕ
  dogs_and_birds : ℕ
  cats_and_birds : ℕ
  all_three : ℕ

/-- The theorem stating that exactly one student owns all three types of pets -/
theorem one_student_owns_all_pets (p : PetOwnership) : 
  p.total = 48 ∧ 
  p.dog_owners = p.total / 2 ∧ 
  p.cat_owners = p.total * 5 / 16 ∧ 
  p.bird_owners = 8 ∧ 
  p.no_pets = 7 ∧
  p.just_dogs = 12 ∧
  p.just_cats = 2 ∧
  p.just_birds = 4 ∧
  p.dog_owners = p.just_dogs + p.dogs_and_cats + p.dogs_and_birds + p.all_three ∧
  p.cat_owners = p.just_cats + p.dogs_and_cats + p.cats_and_birds + p.all_three ∧
  p.bird_owners = p.just_birds + p.dogs_and_birds + p.cats_and_birds + p.all_three ∧
  p.total = p.just_dogs + p.just_cats + p.just_birds + p.dogs_and_cats + p.dogs_and_birds + p.cats_and_birds + p.all_three + p.no_pets
  →
  p.all_three = 1 := by
  sorry

end one_student_owns_all_pets_l1938_193897


namespace gcd_1458_1479_l1938_193887

theorem gcd_1458_1479 : Nat.gcd 1458 1479 = 21 := by
  sorry

end gcd_1458_1479_l1938_193887


namespace water_weight_l1938_193834

/-- Proves that a gallon of water weighs 8 pounds given the conditions of the water tank problem -/
theorem water_weight (tank_capacity : ℝ) (empty_tank_weight : ℝ) (fill_percentage : ℝ) (current_weight : ℝ)
  (h1 : tank_capacity = 200)
  (h2 : empty_tank_weight = 80)
  (h3 : fill_percentage = 0.8)
  (h4 : current_weight = 1360) :
  (current_weight - empty_tank_weight) / (fill_percentage * tank_capacity) = 8 := by
  sorry

end water_weight_l1938_193834


namespace centroid_count_l1938_193882

/-- Represents a point on the perimeter of the square -/
structure PerimeterPoint where
  x : ℚ
  y : ℚ
  on_perimeter : (x = 0 ∨ x = 15 ∨ y = 0 ∨ y = 15) ∧ 
                 (0 ≤ x ∧ x ≤ 15) ∧ 
                 (0 ≤ y ∧ y ≤ 15)

/-- The set of 64 equally spaced points on the square's perimeter -/
def perimeter_points : Finset PerimeterPoint := sorry

/-- Checks if three points are collinear -/
def collinear (p q r : PerimeterPoint) : Prop := sorry

/-- Represents the centroid of a triangle -/
structure Centroid where
  x : ℚ
  y : ℚ
  inside_square : (0 < x ∧ x < 15) ∧ (0 < y ∧ y < 15)

/-- Calculates the centroid of a triangle given three points -/
def triangle_centroid (p q r : PerimeterPoint) : Centroid := sorry

/-- The set of all possible centroids -/
def all_centroids : Finset Centroid := sorry

/-- The main theorem to prove -/
theorem centroid_count : 
  (Finset.card perimeter_points = 64) →
  (∀ p ∈ perimeter_points, ∃ m n : ℕ, p.x = m / 15 ∧ p.y = n / 15) →
  (Finset.card all_centroids = 1849) := sorry

end centroid_count_l1938_193882


namespace milk_fraction_in_cup1_l1938_193873

theorem milk_fraction_in_cup1 (initial_tea : ℝ) (initial_milk : ℝ) (cup_size : ℝ) : 
  initial_tea = 6 →
  initial_milk = 8 →
  cup_size = 12 →
  let tea_transferred_to_cup2 := initial_tea / 3
  let tea_in_cup1_after_first_transfer := initial_tea - tea_transferred_to_cup2
  let total_in_cup2_after_first_transfer := initial_milk + tea_transferred_to_cup2
  let amount_transferred_back := total_in_cup2_after_first_transfer / 4
  let milk_ratio_in_cup2 := initial_milk / total_in_cup2_after_first_transfer
  let milk_transferred_back := amount_transferred_back * milk_ratio_in_cup2
  let final_tea_in_cup1 := tea_in_cup1_after_first_transfer + (amount_transferred_back - milk_transferred_back)
  let final_milk_in_cup1 := milk_transferred_back
  let total_liquid_in_cup1 := final_tea_in_cup1 + final_milk_in_cup1
  final_milk_in_cup1 / total_liquid_in_cup1 = 2 / 6.5 :=
by sorry

end milk_fraction_in_cup1_l1938_193873


namespace three_digit_numbers_divisible_by_17_l1938_193868

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 :=
by sorry

end three_digit_numbers_divisible_by_17_l1938_193868


namespace first_group_size_is_eight_l1938_193884

/-- The number of men in the first group -/
def first_group_size : ℕ := 8

/-- The number of hours worked per day -/
def hours_per_day : ℕ := 8

/-- The number of days the first group works -/
def days_first_group : ℕ := 24

/-- The number of men in the second group -/
def second_group_size : ℕ := 12

/-- The number of days the second group works -/
def days_second_group : ℕ := 16

theorem first_group_size_is_eight :
  first_group_size * hours_per_day * days_first_group =
  second_group_size * hours_per_day * days_second_group :=
by sorry

end first_group_size_is_eight_l1938_193884


namespace binary_quadratic_equation_value_l1938_193817

/-- Represents a binary quadratic equation in x and y with a constant m -/
def binary_quadratic_equation (x y m : ℝ) : Prop :=
  x^2 + 2*x*y + 8*y^2 + 14*y + m = 0

/-- Represents that an equation is equivalent to two lines -/
def represents_two_lines (f : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), ∀ x y m,
    f x y m ↔ (a₁*x + b₁*y + c₁ = 0 ∧ a₂*x + b₂*y + c₂ = 0)

theorem binary_quadratic_equation_value :
  represents_two_lines binary_quadratic_equation → ∃ m, ∀ x y, binary_quadratic_equation x y m :=
by
  sorry

end binary_quadratic_equation_value_l1938_193817


namespace stratified_sampling_theorem_l1938_193864

/-- Represents the number of students in each grade level -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Represents the sample size for each grade level -/
structure GradeSample where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- The total population of the school -/
def totalPopulation (gp : GradePopulation) : ℕ :=
  gp.freshmen + gp.sophomores + gp.juniors

/-- The total sample size -/
def totalSample (gs : GradeSample) : ℕ :=
  gs.freshmen + gs.sophomores + gs.juniors

/-- Checks if the sample is proportional to the population for each grade -/
def isProportionalSample (gp : GradePopulation) (gs : GradeSample) : Prop :=
  gs.freshmen * totalPopulation gp = gp.freshmen * totalSample gs ∧
  gs.sophomores * totalPopulation gp = gp.sophomores * totalSample gs ∧
  gs.juniors * totalPopulation gp = gp.juniors * totalSample gs

theorem stratified_sampling_theorem (gp : GradePopulation) (gs : GradeSample) :
  gp.freshmen = 300 →
  gp.sophomores = 200 →
  gp.juniors = 400 →
  totalPopulation gp = 900 →
  totalSample gs = 45 →
  isProportionalSample gp gs →
  gs.freshmen = 15 ∧ gs.sophomores = 10 ∧ gs.juniors = 20 := by
  sorry


end stratified_sampling_theorem_l1938_193864


namespace circle_properties_l1938_193818

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 3)^2 = 4

-- Define a point being inside a circle
def is_inside_circle (x y : ℝ) : Prop := x^2 + (y + 3)^2 < 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Theorem statement
theorem circle_properties :
  (is_inside_circle 1 (-2)) ∧
  (∀ x y : ℝ, line_y_eq_x x y → ¬ circle_C x y) := by
  sorry

end circle_properties_l1938_193818


namespace rearrangements_count_l1938_193885

def word : String := "Alejandro"
def subwords : List String := ["ned", "den"]

theorem rearrangements_count : 
  (List.length word.data - 2) * (Nat.factorial (List.length word.data - 2) / 2) * (List.length subwords) = 40320 := by
  sorry

end rearrangements_count_l1938_193885


namespace divisible_by_24_l1938_193878

theorem divisible_by_24 (a : ℤ) : ∃ k : ℤ, (a^2 + 3*a + 1)^2 - 1 = 24*k := by
  sorry

end divisible_by_24_l1938_193878


namespace some_number_value_l1938_193876

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * n * 49) : n = 5 := by
  sorry

end some_number_value_l1938_193876


namespace siblings_ratio_l1938_193839

/-- Given the number of siblings for Masud, Janet, and Carlos, prove the ratio of Carlos's to Masud's siblings -/
theorem siblings_ratio (masud_siblings : ℕ) (janet_siblings : ℕ) (carlos_siblings : ℕ) : 
  masud_siblings = 60 →
  janet_siblings = 4 * masud_siblings - 60 →
  janet_siblings = carlos_siblings + 135 →
  carlos_siblings * 4 = masud_siblings * 3 := by
  sorry

#check siblings_ratio

end siblings_ratio_l1938_193839


namespace athlete_arrangements_correct_l1938_193835

/-- The number of ways to arrange 5 athletes on 5 tracks with exactly two matches -/
def athlete_arrangements : ℕ := 20

/-- Proof that the number of arrangements is correct -/
theorem athlete_arrangements_correct : athlete_arrangements = 20 := by
  sorry

end athlete_arrangements_correct_l1938_193835


namespace carpet_coverage_percentage_l1938_193807

theorem carpet_coverage_percentage (carpet_length : ℝ) (carpet_width : ℝ) (room_area : ℝ) :
  carpet_length = 4 →
  carpet_width = 9 →
  room_area = 60 →
  (carpet_length * carpet_width) / room_area * 100 = 60 := by
  sorry

end carpet_coverage_percentage_l1938_193807


namespace current_calculation_l1938_193824

theorem current_calculation (Q R t I : ℝ) 
  (heat_eq : Q = I^2 * R * t)
  (resistance : R = 8)
  (heat_generated : Q = 72)
  (time : t = 2) :
  I = 3 * Real.sqrt 2 / 2 := by
  sorry

end current_calculation_l1938_193824


namespace product_of_tans_equals_two_l1938_193820

theorem product_of_tans_equals_two : (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2 := by
  sorry

end product_of_tans_equals_two_l1938_193820


namespace papaya_height_after_five_years_l1938_193821

/-- The height of a papaya tree after n years -/
def papayaHeight (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | 1 => 2
  | 2 => papayaHeight 1 + 1.5 * papayaHeight 1
  | 3 => papayaHeight 2 + 1.5 * papayaHeight 2
  | 4 => papayaHeight 3 + 2 * papayaHeight 3
  | 5 => papayaHeight 4 + 0.5 * papayaHeight 4
  | _ => 0  -- undefined for years beyond 5

theorem papaya_height_after_five_years :
  papayaHeight 5 = 23 := by
  sorry

end papaya_height_after_five_years_l1938_193821


namespace correct_mass_units_l1938_193815

-- Define the mass units
inductive MassUnit
| Kilogram
| Gram

-- Define a structure to represent a mass measurement
structure Mass where
  value : ℝ
  unit : MassUnit

-- Define Xiaogang's weight
def xiaogang_weight : Mass := { value := 25, unit := MassUnit.Kilogram }

-- Define chalk's weight
def chalk_weight : Mass := { value := 15, unit := MassUnit.Gram }

-- Theorem to prove the correct units for Xiaogang and chalk
theorem correct_mass_units :
  xiaogang_weight.unit = MassUnit.Kilogram ∧
  chalk_weight.unit = MassUnit.Gram :=
by sorry

end correct_mass_units_l1938_193815


namespace pool_water_removal_l1938_193891

/-- Calculates the number of gallons of water removed from a rectangular pool when lowering the water level. -/
def gallonsRemoved (length width depth : ℝ) (conversionFactor : ℝ) : ℝ :=
  length * width * depth * conversionFactor

/-- Proves that lowering the water level in a 60 ft by 10 ft rectangular pool by 6 inches removes 2250 gallons of water. -/
theorem pool_water_removal :
  let length : ℝ := 60
  let width : ℝ := 10
  let depth : ℝ := 0.5  -- 6 inches in feet
  let conversionFactor : ℝ := 7.5  -- 1 cubic foot = 7.5 gallons
  gallonsRemoved length width depth conversionFactor = 2250 := by
  sorry

#eval gallonsRemoved 60 10 0.5 7.5

end pool_water_removal_l1938_193891


namespace max_value_is_320_l1938_193886

def operation := ℝ → ℝ → ℝ

def add : operation := λ x y => x + y
def sub : operation := λ x y => x - y
def mul : operation := λ x y => x * y

def evaluate (op1 op2 op3 op4 : operation) : ℝ :=
  op4 (op3 (op2 (op1 25 1.2) 15) 18.8) 2.3

def is_valid_operation (op : operation) : Prop :=
  op = add ∨ op = sub ∨ op = mul

theorem max_value_is_320 :
  ∀ op1 op2 op3 op4 : operation,
    is_valid_operation op1 →
    is_valid_operation op2 →
    is_valid_operation op3 →
    is_valid_operation op4 →
    evaluate op1 op2 op3 op4 ≤ 320 :=
sorry

end max_value_is_320_l1938_193886


namespace velvet_area_for_box_l1938_193870

/-- The total area of velvet needed to line the inside of a box with given dimensions -/
theorem velvet_area_for_box (long_side_length long_side_width short_side_length short_side_width top_bottom_area : ℕ) 
  (h1 : long_side_length = 8)
  (h2 : long_side_width = 6)
  (h3 : short_side_length = 5)
  (h4 : short_side_width = 6)
  (h5 : top_bottom_area = 40) :
  2 * (long_side_length * long_side_width) + 
  2 * (short_side_length * short_side_width) + 
  2 * top_bottom_area = 236 := by
  sorry

#eval 2 * (8 * 6) + 2 * (5 * 6) + 2 * 40

end velvet_area_for_box_l1938_193870


namespace cube_face_planes_divide_space_l1938_193861

-- Define a cube in 3D space
def Cube := Set (ℝ × ℝ × ℝ)

-- Define the planes that each face of the cube lies on
def FacePlanes (c : Cube) := Set (Set (ℝ × ℝ × ℝ))

-- Define a function that counts the number of regions created by the face planes
def countRegions (c : Cube) : ℕ := sorry

-- Theorem stating that the face planes of a cube divide space into 27 regions
theorem cube_face_planes_divide_space (c : Cube) : 
  countRegions c = 27 := by sorry

end cube_face_planes_divide_space_l1938_193861


namespace abs_difference_range_l1938_193844

theorem abs_difference_range (t : ℝ) : let f := λ x : ℝ => Real.sin x + Real.cos x
                                        let g := λ x : ℝ => 2 * Real.cos x
                                        0 ≤ |f t - g t| ∧ |f t - g t| ≤ Real.sqrt 2 := by
  sorry

end abs_difference_range_l1938_193844


namespace gasoline_cost_calculation_l1938_193856

theorem gasoline_cost_calculation
  (cost_per_litre : ℝ)
  (distance_per_litre : ℝ)
  (distance_to_travel : ℝ)
  (cost_per_litre_positive : 0 < cost_per_litre)
  (distance_per_litre_positive : 0 < distance_per_litre) :
  cost_per_litre * distance_to_travel / distance_per_litre =
  cost_per_litre * (distance_to_travel / distance_per_litre) :=
by sorry

#check gasoline_cost_calculation

end gasoline_cost_calculation_l1938_193856


namespace inspector_group_b_count_l1938_193871

/-- Represents the problem of determining the number of inspectors in Group B -/
theorem inspector_group_b_count : 
  ∀ (a b : ℕ) (group_b_count : ℕ),
  a > 0 → b > 0 →
  (2 * (a + 2 * b)) / 2 = (2 * (a + 5 * b)) / 3 →  -- Equation from Group A's work
  (5 * (a + 5 * b)) / (group_b_count * 5) = (2 * (a + 2 * b)) / (8 * 2) →  -- Equation comparing Group A and B's work
  group_b_count = 12 := by
    sorry


end inspector_group_b_count_l1938_193871


namespace negation_of_universal_proposition_l1938_193865

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by
  sorry

end negation_of_universal_proposition_l1938_193865


namespace bike_sharing_growth_model_l1938_193819

/-- Represents the bike-sharing company's growth model -/
theorem bike_sharing_growth_model (x : ℝ) :
  let initial_bikes : ℕ := 1000
  let additional_bikes : ℕ := 440
  let growth_factor : ℝ := (1 + x)
  let months : ℕ := 2
  (initial_bikes : ℝ) * growth_factor ^ months = (initial_bikes : ℝ) + additional_bikes :=
by
  sorry

end bike_sharing_growth_model_l1938_193819


namespace p_greater_than_q_greater_than_r_l1938_193827

def P : ℚ := -1 / (201603 * 201604)
def Q : ℚ := -1 / (201602 * 201604)
def R : ℚ := -1 / (201602 * 201603)

theorem p_greater_than_q_greater_than_r : P > Q ∧ Q > R := by sorry

end p_greater_than_q_greater_than_r_l1938_193827


namespace equation_solution_l1938_193802

theorem equation_solution :
  let f (x : ℂ) := (x^2 + 4*x + 20) / (x^2 - 7*x + 12)
  let g (x : ℂ) := (x - 3) / (x - 1)
  ∀ x : ℂ, f x = g x ↔ x = (17 + Complex.I * Real.sqrt 543) / 26 ∨ x = (17 - Complex.I * Real.sqrt 543) / 26 :=
by sorry

end equation_solution_l1938_193802


namespace principal_calculation_l1938_193869

/-- Given an interest rate, time period, and a relationship between
    the principal and interest, prove that the principal is 9200. -/
theorem principal_calculation (r t : ℝ) (P : ℝ) :
  r = 0.12 →
  t = 3 →
  P * r * t = P - 5888 →
  P = 9200 := by
  sorry

end principal_calculation_l1938_193869


namespace fraction_equality_l1938_193828

theorem fraction_equality (P Q : ℤ) (x : ℝ) 
  (h : x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5) :
  (P / (x + 3 : ℝ)) + (Q / ((x^2 : ℝ) - 5*x)) = 
    ((x^2 : ℝ) - 3*x + 12) / (x^3 + x^2 - 15*x) →
  (Q : ℚ) / P = 20 / 9 := by
  sorry

end fraction_equality_l1938_193828


namespace fraction_simplification_l1938_193879

theorem fraction_simplification :
  (1^2 + 1) * (2^2 + 1) * (3^2 + 1) / ((2^2 - 1) * (3^2 - 1) * (4^2 - 1)) = 5 / 18 :=
by sorry

end fraction_simplification_l1938_193879


namespace product_increase_fifteen_times_l1938_193833

theorem product_increase_fifteen_times :
  ∃ (a₁ a₂ a₃ a₄ a₅ : ℕ),
    ((a₁ - 3) * (a₂ - 3) * (a₃ - 3) * (a₄ - 3) * (a₅ - 3) : ℤ) = 
    15 * (a₁ * a₂ * a₃ * a₄ * a₅) := by
  sorry

end product_increase_fifteen_times_l1938_193833


namespace right_triangle_altitude_ratio_l1938_193816

/-- 
Given a right triangle ABC with legs a and b (a ≤ b) and hypotenuse c,
where the triangle formed by its altitudes is also a right triangle,
prove that the ratio of the shorter leg to the longer leg is √((√5 - 1) / 2).
-/
theorem right_triangle_altitude_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≤ b) :
  a^2 + b^2 = c^2 →
  a^2 + (a^2 * b^2) / (a^2 + b^2) = b^2 →
  a / b = Real.sqrt ((Real.sqrt 5 - 1) / 2) := by
  sorry

end right_triangle_altitude_ratio_l1938_193816


namespace stock_price_uniqueness_l1938_193850

theorem stock_price_uniqueness : ¬∃ (k m : ℕ), (117/100)^k * (83/100)^m = 1 := by
  sorry

end stock_price_uniqueness_l1938_193850


namespace regular_polygon_perimeter_l1938_193814

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 :=
by sorry

end regular_polygon_perimeter_l1938_193814


namespace simplify_radical_product_l1938_193862

theorem simplify_radical_product (x : ℝ) (hx : x > 0) :
  Real.sqrt (45 * x) * Real.sqrt (32 * x) * Real.sqrt (18 * x) * (27 * x) ^ (1/3) = 72 * x ^ (1/3) * Real.sqrt (5 * x) := by
  sorry

end simplify_radical_product_l1938_193862


namespace triangle_cos_inequality_l1938_193863

/-- For any real numbers A, B, C that are angles of a triangle, 
    the inequality 8 cos A · cos B · cos C ≤ 1 holds. -/
theorem triangle_cos_inequality (A B C : Real) (h : A + B + C = π) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := by
  sorry

end triangle_cos_inequality_l1938_193863


namespace weight_loss_challenge_l1938_193826

theorem weight_loss_challenge (initial_weight : ℝ) (clothes_weight_percentage : ℝ) 
  (h1 : clothes_weight_percentage > 0)
  (h2 : initial_weight > 0) : 
  (0.90 * initial_weight + clothes_weight_percentage * 0.90 * initial_weight) / initial_weight = 0.918 → 
  clothes_weight_percentage = 0.02 := by
sorry

end weight_loss_challenge_l1938_193826


namespace mary_candy_ratio_l1938_193859

/-- The number of times Mary initially has more candy than Megan -/
def candy_ratio (megan_candy : ℕ) (mary_final_candy : ℕ) (mary_added_candy : ℕ) : ℚ :=
  (mary_final_candy - mary_added_candy : ℚ) / megan_candy

theorem mary_candy_ratio :
  candy_ratio 5 25 10 = 3 := by sorry

end mary_candy_ratio_l1938_193859


namespace solve_equation_l1938_193860

theorem solve_equation (y : ℚ) (h : (1 : ℚ) / 3 + 1 / y = 7 / 12) : y = 4 := by
  sorry

end solve_equation_l1938_193860


namespace sum_of_digits_equals_16_l1938_193858

/-- The sum of the digits of (10^38) - 85 when written as a base 10 integer -/
def sumOfDigits : ℕ :=
  -- Define the sum of digits here
  sorry

/-- Theorem stating that the sum of the digits of (10^38) - 85 is 16 -/
theorem sum_of_digits_equals_16 : sumOfDigits = 16 := by
  sorry

end sum_of_digits_equals_16_l1938_193858


namespace inverse_variation_problem_l1938_193803

/-- Represents the relationship between y, x, and z -/
def relation (k : ℝ) (x y z : ℝ) : Prop :=
  7 * y = (k * z) / (2 * x)^2

theorem inverse_variation_problem (k : ℝ) :
  relation k 1 20 5 →
  relation k 8 0.625 10 :=
by
  sorry


end inverse_variation_problem_l1938_193803


namespace fraction_addition_l1938_193889

theorem fraction_addition : (1 : ℚ) / 210 + 17 / 35 = 103 / 210 := by
  sorry

end fraction_addition_l1938_193889


namespace bisection_method_next_interval_l1938_193823

def f (x : ℝ) := x^3 - 2*x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x₀ := (a + b) / 2
  f a < 0 ∧ f b > 0 ∧ f x₀ > 0 →
  ∃ x ∈ Set.Ioo a x₀, f x = 0 :=
by
  sorry

end bisection_method_next_interval_l1938_193823


namespace negation_of_exists_equals_sin_l1938_193855

theorem negation_of_exists_equals_sin (x : ℝ) : 
  (¬ ∃ x : ℝ, x = Real.sin x) ↔ (∀ x : ℝ, x ≠ Real.sin x) := by
  sorry

end negation_of_exists_equals_sin_l1938_193855


namespace arm_wrestling_tournament_l1938_193809

/-- The number of participants with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k

theorem arm_wrestling_tournament (n : ℕ) (h1 : n > 7) (h2 : f n 7 5 = 42) : n = 8 := by
  sorry

end arm_wrestling_tournament_l1938_193809


namespace right_triangle_side_length_l1938_193822

/-- Given a right triangle ABC where:
    - The altitude from C to AB is 12 km
    - The sum of all sides (AB + BC + AC) is 60 km
    Prove that the length of AB is 22.5 km -/
theorem right_triangle_side_length 
  (A B C : ℝ × ℝ) 
  (is_right_triangle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (altitude : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2 - 
    (((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2))^2 / 
    ((B.1 - A.1)^2 + (B.2 - A.2)^2))) = 12)
  (perimeter : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) + 
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) + 
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 60) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 22.5 := by
  sorry


end right_triangle_side_length_l1938_193822


namespace lcm_of_3_4_6_15_l1938_193847

def numbers : List ℕ := [3, 4, 6, 15]

theorem lcm_of_3_4_6_15 : Nat.lcm (Nat.lcm (Nat.lcm 3 4) 6) 15 = 60 := by
  sorry

end lcm_of_3_4_6_15_l1938_193847


namespace parallel_transitivity_l1938_193841

-- Define a type for lines in a plane
def Line : Type := ℝ × ℝ → Prop

-- Define parallel relationship between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem parallel_transitivity (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end parallel_transitivity_l1938_193841


namespace square_a_times_a_plus_four_l1938_193896

theorem square_a_times_a_plus_four (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end square_a_times_a_plus_four_l1938_193896


namespace units_digit_of_six_to_sixth_l1938_193840

theorem units_digit_of_six_to_sixth (n : ℕ) : n = 6^6 → n % 10 = 6 := by
  sorry

end units_digit_of_six_to_sixth_l1938_193840


namespace sin_sum_equals_half_l1938_193831

theorem sin_sum_equals_half : 
  Real.sin (163 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1/2 := by
  sorry

end sin_sum_equals_half_l1938_193831


namespace birthday_75_days_later_l1938_193857

theorem birthday_75_days_later (birthday : ℕ) : 
  (birthday % 7 = 0) → ((birthday + 75) % 7 = 5) := by
  sorry

#check birthday_75_days_later

end birthday_75_days_later_l1938_193857
