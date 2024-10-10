import Mathlib

namespace inscribed_sphere_ratio_l2998_299836

/-- A regular tetrahedron with height H and an inscribed sphere of radius R -/
structure RegularTetrahedron where
  H : ℝ
  R : ℝ
  H_pos : H > 0
  R_pos : R > 0

/-- The ratio of the inscribed sphere radius to the tetrahedron height is 1:4 -/
theorem inscribed_sphere_ratio (t : RegularTetrahedron) : t.R / t.H = 1 / 4 := by
  sorry

end inscribed_sphere_ratio_l2998_299836


namespace solution_comparison_l2998_299880

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (-q / p > -q' / p') ↔ (q / p < q' / p') :=
by sorry

end solution_comparison_l2998_299880


namespace person_is_knight_l2998_299837

-- Define the type of person
inductive Person : Type
  | Knight : Person
  | Liar : Person

-- Define the statements
def lovesLinda (p : Person) : Prop := 
  match p with
  | Person.Knight => true
  | Person.Liar => false

def lovesKatie (p : Person) : Prop :=
  match p with
  | Person.Knight => true
  | Person.Liar => false

-- Define the theorem
theorem person_is_knight : 
  ∀ (p : Person), 
    (lovesLinda p = true ∨ lovesLinda p = false) → 
    (lovesLinda p → lovesKatie p) → 
    p = Person.Knight :=
by
  sorry


end person_is_knight_l2998_299837


namespace f_nested_application_l2998_299896

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 1

theorem f_nested_application : f (f (f (f (f 1)))) = 4 := by
  sorry

end f_nested_application_l2998_299896


namespace least_clock_equivalent_after_10_l2998_299814

def clock_equivalent (h : ℕ) : Prop :=
  (h ^ 2 - h) % 12 = 0

theorem least_clock_equivalent_after_10 :
  ∀ h : ℕ, h > 10 → clock_equivalent h → h ≥ 12 :=
by
  sorry

end least_clock_equivalent_after_10_l2998_299814


namespace fraction_evaluation_l2998_299819

theorem fraction_evaluation : 
  (20-19+18-17+16-15+14-13+12-11+10-9+8-7+6-5+4-3+2-1) / 
  (2-3+4-5+6-7+8-9+10-11+12-13+14-15+16-17+18-19+20) = 10/11 := by
  sorry

end fraction_evaluation_l2998_299819


namespace sin_period_l2998_299894

theorem sin_period (x : ℝ) : 
  let f : ℝ → ℝ := fun x => Real.sin ((1/2) * x + 3)
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = 4 * Real.pi :=
by sorry

end sin_period_l2998_299894


namespace stratified_sampling_third_group_size_l2998_299845

/-- Proves that in a stratified sampling scenario, given specific conditions, 
    the size of the third group is 1040. -/
theorem stratified_sampling_third_group_size 
  (total_sample : ℕ) 
  (grade11_sample : ℕ) 
  (grade10_pop : ℕ) 
  (grade11_pop : ℕ) 
  (h1 : total_sample = 81)
  (h2 : grade11_sample = 30)
  (h3 : grade10_pop = 1000)
  (h4 : grade11_pop = 1200) :
  ∃ n : ℕ, 
    (grade11_sample : ℚ) / total_sample = 
    grade11_pop / (grade10_pop + grade11_pop + n) ∧ 
    n = 1040 := by
  sorry

end stratified_sampling_third_group_size_l2998_299845


namespace sugar_added_indeterminate_l2998_299899

-- Define the recipe requirements
def total_flour : ℕ := 9
def total_sugar : ℕ := 5

-- Define Mary's current actions
def flour_added : ℕ := 3
def flour_to_add : ℕ := 6

-- Define a variable for the unknown amount of sugar added
variable (sugar_added : ℕ)

-- Theorem stating that sugar_added cannot be uniquely determined
theorem sugar_added_indeterminate : 
  ∀ (x y : ℕ), x ≠ y → 
  (x ≤ total_sugar ∧ y ≤ total_sugar) → 
  (∃ (state₁ state₂ : ℕ × ℕ), 
    state₁.1 = flour_added ∧ 
    state₁.2 = x ∧ 
    state₂.1 = flour_added ∧ 
    state₂.2 = y) :=
by sorry

end sugar_added_indeterminate_l2998_299899


namespace cake_radius_increase_l2998_299864

theorem cake_radius_increase (c₁ c₂ : ℝ) (h₁ : c₁ = 30) (h₂ : c₂ = 37.5) :
  (c₂ / (2 * Real.pi)) - (c₁ / (2 * Real.pi)) = 7.5 / (2 * Real.pi) := by
  sorry

end cake_radius_increase_l2998_299864


namespace function_properties_l2998_299828

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- Define the specific function h
def h (m : ℝ) (x : ℝ) : ℝ := f 2 9 x - m + 1

-- Theorem statement
theorem function_properties :
  (∃ (a b : ℝ), f a b (-1) = 0 ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≥ f a b (-1)) ∧
   (∀ x : ℝ, f a b x = x^3 + 6*x^2 + 9*x + 4)) ∧
  (∀ m : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    h m x₁ = 0 ∧ h m x₂ = 0 ∧ h m x₃ = 0) ↔ 1 < m ∧ m < 5) :=
by sorry

end function_properties_l2998_299828


namespace octagon_diagonals_l2998_299872

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l2998_299872


namespace no_perfect_squares_l2998_299874

theorem no_perfect_squares (a b : ℕ) : ¬(∃k m : ℕ, (a^2 + 2*b^2 = k^2) ∧ (b^2 + 2*a = m^2)) := by
  sorry

end no_perfect_squares_l2998_299874


namespace five_roots_sum_l2998_299807

noncomputable def f (x : ℝ) : ℝ :=
  if x = 2 then 1 else Real.log (abs (x - 2))

theorem five_roots_sum (b c : ℝ) 
  (h : ∃ x₁ x₂ x₃ x₄ x₅ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ 
                           x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                           x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅ ∧
                           (f x₁)^2 + b * (f x₁) + c = 0 ∧
                           (f x₂)^2 + b * (f x₂) + c = 0 ∧
                           (f x₃)^2 + b * (f x₃) + c = 0 ∧
                           (f x₄)^2 + b * (f x₄) + c = 0 ∧
                           (f x₅)^2 + b * (f x₅) + c = 0) :
  ∃ x₁ x₂ x₃ x₄ x₅ : ℝ, f (x₁ + x₂ + x₃ + x₄ + x₅) = 3 * Real.log 2 := by
  sorry

end five_roots_sum_l2998_299807


namespace variance_transform_l2998_299870

/-- The variance of a dataset -/
def variance (data : List ℝ) : ℝ := sorry

/-- Transform a dataset by multiplying each element by a and adding b -/
def transform (data : List ℝ) (a b : ℝ) : List ℝ := sorry

theorem variance_transform (data : List ℝ) (a b : ℝ) :
  variance data = 3 →
  variance (transform data a b) = 12 →
  |a| = 2 := by sorry

end variance_transform_l2998_299870


namespace f_derivative_f_extrema_log_inequality_l2998_299803

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

-- State the theorems
theorem f_derivative (a : ℝ) (x : ℝ) (h : x ≠ 0) :
  deriv (f a) x = (a * x - 1) / (a * x^2) :=
sorry

theorem f_extrema (e : ℝ) (h_e : e > 0) :
  let f_1 := f 1
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ Set.Icc (1/e) e, f_1 x ≤ max_val) ∧
    (∃ x ∈ Set.Icc (1/e) e, f_1 x = max_val) ∧
    (∀ x ∈ Set.Icc (1/e) e, f_1 x ≥ min_val) ∧
    (∃ x ∈ Set.Icc (1/e) e, f_1 x = min_val) ∧
    max_val = e - 2 ∧ min_val = 0 :=
sorry

theorem log_inequality (n : ℕ) (h : n > 1) :
  Real.log (n / (n - 1)) > 1 / n :=
sorry

end f_derivative_f_extrema_log_inequality_l2998_299803


namespace max_distance_circle_to_line_l2998_299825

/-- The maximum distance from any point on the circle (x-1)^2 + y^2 = 3 to the line x - y - 1 = 0 is √3 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 3}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}
  ∀ p ∈ circle, ∃ q ∈ line,
    ∀ r ∈ circle, ∀ s ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt 3 ∧
      ∃ p' ∈ circle, ∃ q' ∈ line,
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = Real.sqrt 3 :=
by sorry

end max_distance_circle_to_line_l2998_299825


namespace middle_group_frequency_l2998_299881

/-- Represents a frequency distribution histogram -/
structure FrequencyHistogram where
  rectangles : Fin 5 → ℝ
  total_sample_size : ℝ
  middle_rectangle_condition : rectangles 2 = (1/3) * (rectangles 0 + rectangles 1 + rectangles 3 + rectangles 4)
  total_area_condition : rectangles 0 + rectangles 1 + rectangles 2 + rectangles 3 + rectangles 4 = total_sample_size

/-- The theorem stating that the frequency of the middle group is 25 -/
theorem middle_group_frequency (h : FrequencyHistogram) (h_sample_size : h.total_sample_size = 100) :
  h.rectangles 2 = 25 := by
  sorry

end middle_group_frequency_l2998_299881


namespace M_is_power_of_three_l2998_299835

/-- Arithmetic sequence with a_n = n -/
def arithmetic_seq (n : ℕ) : ℕ := n

/-- Sequence of t_n values -/
def t_seq : ℕ → ℕ
  | 0 => 0
  | n+1 => (3^(n+1) - 1) / 2

/-- M_n is the sum of terms from (t_{n-1}+1)th to t_n th term -/
def M (n : ℕ) : ℕ :=
  let a := t_seq (n-1)
  let b := t_seq n
  (b * (b + 1) - a * (a + 1)) / 2

/-- Main theorem: M_n = 3^(2n-2) for all n ∈ ℕ -/
theorem M_is_power_of_three (n : ℕ) : M n = 3^(2*n - 2) := by
  sorry


end M_is_power_of_three_l2998_299835


namespace compound_composition_l2998_299826

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Sulphur in g/mol -/
def atomic_weight_S : ℝ := 32.06

/-- The number of Sulphur atoms in the compound -/
def num_S_atoms : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 150

/-- The number of Aluminium atoms in the compound -/
def num_Al_atoms : ℕ := 2

theorem compound_composition :
  num_Al_atoms * atomic_weight_Al + num_S_atoms * atomic_weight_S = molecular_weight := by
  sorry

end compound_composition_l2998_299826


namespace tim_kittens_l2998_299888

theorem tim_kittens (initial : ℕ) (given_away : ℕ) (received : ℕ) :
  initial = 6 →
  given_away = 3 →
  received = 9 →
  initial - given_away + received = 12 := by
sorry

end tim_kittens_l2998_299888


namespace probability_of_two_white_balls_l2998_299879

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def black_balls : ℕ := 2

def probability_two_white : ℚ := 3 / 10

theorem probability_of_two_white_balls :
  (Nat.choose white_balls 2) / (Nat.choose total_balls 2) = probability_two_white :=
sorry

end probability_of_two_white_balls_l2998_299879


namespace max_profit_at_three_l2998_299848

/-- Represents the annual operating cost for a given year -/
def annual_cost (n : ℕ) : ℚ := 2 * n

/-- Represents the total operating cost for n years -/
def total_cost (n : ℕ) : ℚ := n^2 + n

/-- Represents the annual operating income -/
def annual_income : ℚ := 11

/-- Represents the initial cost of the car -/
def initial_cost : ℚ := 9

/-- Represents the annual average profit for n years -/
def annual_average_profit (n : ℕ+) : ℚ := 
  annual_income - (total_cost n + initial_cost) / n

/-- Theorem stating that the annual average profit is maximized when n = 3 -/
theorem max_profit_at_three : 
  ∀ (m : ℕ+), annual_average_profit 3 ≥ annual_average_profit m :=
sorry

end max_profit_at_three_l2998_299848


namespace expression_bounds_l2998_299852

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) : 
  2 * Real.sqrt 2 + 2 ≤ 
    Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt ((b+1)^2 + (2 - c)^2) + 
    Real.sqrt ((c-1)^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2) ∧
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt ((b+1)^2 + (2 - c)^2) + 
  Real.sqrt ((c-1)^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2) ≤ 8 := by
  sorry

end expression_bounds_l2998_299852


namespace equal_intercepts_condition_l2998_299863

/-- The line equation ax + y - 2 - a = 0 has equal intercepts on x and y axes iff a = -2 or a = 1 -/
theorem equal_intercepts_condition (a : ℝ) : 
  (∃ (x y : ℝ), (a * x + y - 2 - a = 0 ∧ 
                ((x = 0 ∨ y = 0) ∧ 
                 (∀ x' y', a * x' + y' - 2 - a = 0 ∧ x' = 0 → y' = y) ∧
                 (∀ x' y', a * x' + y' - 2 - a = 0 ∧ y' = 0 → x' = x))))
  ↔ (a = -2 ∨ a = 1) :=
sorry

end equal_intercepts_condition_l2998_299863


namespace person_on_throne_l2998_299851

-- Define the possible characteristics of the person
inductive PersonType
| Liar
| Monkey
| Knight

-- Define the statement made by the person
def statement (p : PersonType) : Prop :=
  p = PersonType.Liar ∨ p = PersonType.Monkey

-- Theorem to prove
theorem person_on_throne (p : PersonType) (h : statement p) : 
  p = PersonType.Monkey ∧ p ≠ PersonType.Liar :=
sorry

end person_on_throne_l2998_299851


namespace car_trip_average_speed_l2998_299856

/-- Calculates the average speed of a car trip with multiple segments and delays -/
theorem car_trip_average_speed 
  (local_distance : ℝ) (local_speed : ℝ)
  (gravel_distance : ℝ) (gravel_speed : ℝ)
  (highway_distance : ℝ) (highway_speed : ℝ)
  (traffic_delay : ℝ) (obstruction_delay : ℝ)
  (h_local : local_distance = 60 ∧ local_speed = 30)
  (h_gravel : gravel_distance = 10 ∧ gravel_speed = 20)
  (h_highway : highway_distance = 105 ∧ highway_speed = 60)
  (h_traffic : traffic_delay = 0.25)
  (h_obstruction : obstruction_delay = 0.1667)
  : ∃ (average_speed : ℝ), 
    abs (average_speed - 37.5) < 0.1 ∧
    average_speed = (local_distance + gravel_distance + highway_distance) / 
      (local_distance / local_speed + gravel_distance / gravel_speed + 
       highway_distance / highway_speed + traffic_delay + obstruction_delay) :=
by sorry

end car_trip_average_speed_l2998_299856


namespace complex_equation_solution_l2998_299890

theorem complex_equation_solution (z : ℂ) : (z - 1) * I = 1 + I → z = 2 - I := by
  sorry

end complex_equation_solution_l2998_299890


namespace einstein_soda_sales_l2998_299861

def goal : ℝ := 500
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2
def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def remaining : ℝ := 258

theorem einstein_soda_sales :
  ∃ (soda_sold : ℕ),
    goal = pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold + remaining ∧
    soda_sold = 25 := by
  sorry

end einstein_soda_sales_l2998_299861


namespace some_number_value_l2998_299843

theorem some_number_value (some_number : ℝ) : 
  (3 * 10^2) * (4 * some_number) = 12 → some_number = 0.01 := by
  sorry

end some_number_value_l2998_299843


namespace sequence_inequality_l2998_299862

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) :
  n ≥ 2 →
  (∀ k, a k > 0) →
  (∀ k ∈ Finset.range (n - 1), (a (k - 1) + a k) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) →
  a n < 1 / (n - 1) := by
sorry

end sequence_inequality_l2998_299862


namespace log_square_equals_twenty_l2998_299839

theorem log_square_equals_twenty (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 8 / Real.log y)
  (h_product : x * y = 128) : 
  (Real.log (x / y) / Real.log 2)^2 = 20 := by
  sorry

end log_square_equals_twenty_l2998_299839


namespace arithmetic_geometric_mean_ratio_l2998_299891

theorem arithmetic_geometric_mean_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h_mean : (a + b) / 2 = 3 * Real.sqrt (a * b)) : 
  ∃ (n : ℤ), n = 34 ∧ ∀ (m : ℤ), |a / b - n| ≤ |a / b - m| :=
sorry

end arithmetic_geometric_mean_ratio_l2998_299891


namespace division_remainder_problem_l2998_299878

theorem division_remainder_problem (j : ℕ+) (h : ∃ b : ℕ, 120 = b * j^2 + 12) :
  ∃ k : ℕ, 180 = k * j + 0 := by
sorry

end division_remainder_problem_l2998_299878


namespace zero_unique_additive_multiplicative_property_l2998_299876

theorem zero_unique_additive_multiplicative_property :
  ∀ x : ℤ, (∀ z : ℤ, z + x = z) ∧ (∀ z : ℤ, z * x = 0) → x = 0 := by
  sorry

end zero_unique_additive_multiplicative_property_l2998_299876


namespace circle_properties_l2998_299868

-- Define the set of points (x, y) satisfying the equation
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.2 + 1 = 0}

-- Theorem statement
theorem circle_properties (p : ℝ × ℝ) (h : p ∈ S) :
  (∃ (z : ℝ), ∀ (q : ℝ × ℝ), q ∈ S → q.1 + q.2 ≤ z ∧ z = 2 + Real.sqrt 6) ∧
  (∀ (q : ℝ × ℝ), q ∈ S → q.1 ≠ 0 → -Real.sqrt 2 ≤ (q.2 + 1) / q.1 ∧ (q.2 + 1) / q.1 ≤ Real.sqrt 2) ∧
  (∀ (q : ℝ × ℝ), q ∈ S → 8 - 2*Real.sqrt 15 ≤ q.1^2 - 2*q.1 + q.2^2 + 1 ∧ 
                         q.1^2 - 2*q.1 + q.2^2 + 1 ≤ 8 + 2*Real.sqrt 15) :=
by
  sorry


end circle_properties_l2998_299868


namespace factory_uses_systematic_sampling_l2998_299887

/-- Represents a sampling method used in quality control --/
inductive SamplingMethod
| Systematic
| Random
| Stratified
| Cluster

/-- Represents a factory with a conveyor belt and quality inspection process --/
structure Factory where
  /-- The interval between product inspections in minutes --/
  inspection_interval : ℕ
  /-- Whether the inspection position on the conveyor belt is fixed --/
  fixed_position : Bool

/-- Determines if a given factory uses systematic sampling --/
def uses_systematic_sampling (f : Factory) : Prop :=
  f.inspection_interval > 0 ∧ f.fixed_position

/-- The factory described in the problem --/
def problem_factory : Factory :=
  { inspection_interval := 10
  , fixed_position := true }

/-- Theorem stating that the factory in the problem uses systematic sampling --/
theorem factory_uses_systematic_sampling :
  uses_systematic_sampling problem_factory :=
sorry

end factory_uses_systematic_sampling_l2998_299887


namespace no_positive_integer_solutions_l2998_299847

theorem no_positive_integer_solutions :
  ¬ ∃ (x : ℕ), 15 < 3 - 2 * (x : ℤ) := by
  sorry

end no_positive_integer_solutions_l2998_299847


namespace culture_and_messengers_l2998_299812

-- Define the types
structure Performance :=
  (troupe : String)
  (location : String)
  (impression : String)

-- Define the conditions
def legend_show : Performance :=
  { troupe := "Chinese Acrobatic Troupe",
    location := "United States",
    impression := "favorable" }

-- Define the properties we want to prove
def is_national_and_global (p : Performance) : Prop :=
  p.troupe ≠ p.location ∧ p.impression = "favorable"

def are_cultural_messengers (p : Performance) : Prop :=
  p.troupe = "Chinese Acrobatic Troupe" ∧ p.impression = "favorable"

-- The theorem to prove
theorem culture_and_messengers :
  is_national_and_global legend_show ∧ are_cultural_messengers legend_show :=
by sorry

end culture_and_messengers_l2998_299812


namespace sum_of_digits_next_l2998_299811

/-- S(n) is the sum of the digits of a positive integer n -/
def S (n : ℕ+) : ℕ :=
  sorry

theorem sum_of_digits_next (n : ℕ+) (h : S n = 1274) : S (n + 1) = 1239 :=
  sorry

end sum_of_digits_next_l2998_299811


namespace prob_even_sum_spinners_l2998_299810

/-- Represents a spinner with three sections -/
structure Spinner :=
  (sections : Fin 3 → ℕ)

/-- Calculates the probability of getting an even number on a spinner -/
def probEven (s : Spinner) : ℚ :=
  (Finset.filter (λ i => s.sections i % 2 = 0) Finset.univ).card / 3

/-- Calculates the probability of getting an odd number on a spinner -/
def probOdd (s : Spinner) : ℚ :=
  1 - probEven s

/-- The first spinner with sections 2, 3, and 7 -/
def spinner1 : Spinner :=
  ⟨λ i => [2, 3, 7].get i⟩

/-- The second spinner with sections 5, 3, and 6 -/
def spinner2 : Spinner :=
  ⟨λ i => [5, 3, 6].get i⟩

/-- The probability of getting an even sum when spinning both spinners -/
def probEvenSum (s1 s2 : Spinner) : ℚ :=
  probEven s1 * probEven s2 + probOdd s1 * probOdd s2

theorem prob_even_sum_spinners :
  probEvenSum spinner1 spinner2 = 5 / 9 := by
  sorry

end prob_even_sum_spinners_l2998_299810


namespace min_integer_solution_inequality_l2998_299869

theorem min_integer_solution_inequality :
  ∀ x : ℤ, (4 * (x + 1) + 2 > x - 1) ↔ (x ≥ -2) :=
by sorry

end min_integer_solution_inequality_l2998_299869


namespace faye_apps_left_l2998_299802

/-- The number of apps left after deletion -/
def apps_left (initial : ℕ) (deleted : ℕ) : ℕ :=
  initial - deleted

/-- Theorem stating that Faye has 4 apps left -/
theorem faye_apps_left : apps_left 12 8 = 4 := by
  sorry

end faye_apps_left_l2998_299802


namespace graph_properties_of_y_squared_equals_sin_x_squared_l2998_299882

theorem graph_properties_of_y_squared_equals_sin_x_squared :
  ∃ f : ℝ → Set ℝ, 
    (∀ x y, y ∈ f x ↔ y^2 = Real.sin (x^2)) ∧ 
    (0 ∈ f 0) ∧ 
    (∀ x y, y ∈ f x → -y ∈ f x) ∧
    (∀ x, (∃ y, y ∈ f x) → Real.sin (x^2) ≥ 0) :=
by sorry

end graph_properties_of_y_squared_equals_sin_x_squared_l2998_299882


namespace five_pq_odd_l2998_299820

theorem five_pq_odd (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (5 * p * q) := by
  sorry

end five_pq_odd_l2998_299820


namespace annual_earnings_difference_l2998_299892

/-- Calculates the difference in annual earnings between a new job and an old job -/
theorem annual_earnings_difference
  (new_wage : ℝ)
  (new_hours : ℝ)
  (old_wage : ℝ)
  (old_hours : ℝ)
  (weeks_per_year : ℝ)
  (h1 : new_wage = 20)
  (h2 : new_hours = 40)
  (h3 : old_wage = 16)
  (h4 : old_hours = 25)
  (h5 : weeks_per_year = 52) :
  new_wage * new_hours * weeks_per_year - old_wage * old_hours * weeks_per_year = 20800 := by
  sorry

#check annual_earnings_difference

end annual_earnings_difference_l2998_299892


namespace t_shaped_area_l2998_299883

/-- The area of a T-shaped region formed by subtracting two squares and a rectangle from a larger square --/
theorem t_shaped_area (side_large : ℝ) (side_small : ℝ) (rect_length rect_width : ℝ) : 
  side_large = side_small + rect_length →
  side_large = 6 →
  side_small = 2 →
  rect_length = 4 →
  rect_width = 2 →
  side_large^2 - (2 * side_small^2 + rect_length * rect_width) = 20 := by
  sorry

end t_shaped_area_l2998_299883


namespace sweets_expenditure_l2998_299860

theorem sweets_expenditure (initial_amount : ℚ) (friends_count : ℕ) (amount_per_friend : ℚ) (final_amount : ℚ) 
  (h1 : initial_amount = 71/10)
  (h2 : friends_count = 2)
  (h3 : amount_per_friend = 1)
  (h4 : final_amount = 405/100) :
  initial_amount - friends_count * amount_per_friend - final_amount = 21/20 := by
  sorry

end sweets_expenditure_l2998_299860


namespace rationalize_and_simplify_l2998_299823

theorem rationalize_and_simplify :
  (Real.sqrt 18) / (Real.sqrt 9 - Real.sqrt 3) = (3 * Real.sqrt 2 + Real.sqrt 6) / 2 := by
  sorry

end rationalize_and_simplify_l2998_299823


namespace prob_class1_drew_two_mc_correct_expected_rounds_correct_l2998_299832

-- Define the boxes
structure Box where
  multiple_choice : ℕ
  fill_in_blank : ℕ

-- Define the game
structure Game where
  box_a : Box
  box_b : Box
  class_6_first_win_prob : ℚ
  next_win_prob : ℚ

-- Define the problem
def chinese_culture_competition : Game :=
  { box_a := { multiple_choice := 5, fill_in_blank := 3 }
  , box_b := { multiple_choice := 4, fill_in_blank := 3 }
  , class_6_first_win_prob := 3/5
  , next_win_prob := 2/5
  }

-- Part 1: Probability calculation
def prob_class1_drew_two_mc (g : Game) : ℚ :=
  20/49

-- Part 2: Expected value calculation
def expected_rounds (g : Game) : ℚ :=
  537/125

-- Theorem statements
theorem prob_class1_drew_two_mc_correct (g : Game) :
  g = chinese_culture_competition →
  prob_class1_drew_two_mc g = 20/49 := by sorry

theorem expected_rounds_correct (g : Game) :
  g = chinese_culture_competition →
  expected_rounds g = 537/125 := by sorry

end prob_class1_drew_two_mc_correct_expected_rounds_correct_l2998_299832


namespace intersection_A_B_l2998_299831

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 + 3*x - 4 < 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end intersection_A_B_l2998_299831


namespace marly_bills_denomination_l2998_299898

theorem marly_bills_denomination (x : ℕ) : 
  (10 * 20 + 8 * x + 4 * 5 = 3 * 100) → x = 10 := by
  sorry

end marly_bills_denomination_l2998_299898


namespace mathematics_competition_is_good_l2998_299821

theorem mathematics_competition_is_good :
  ∃ (x₁ y₁ x₂ y₂ : ℕ),
    1000 * x₁ + y₁ = 2 * x₁ * y₁ ∧
    1000 * x₂ + y₂ = 2 * x₂ * y₂ ∧
    1000 * x₁ + y₁ = 13520 ∧
    1000 * x₂ + y₂ = 63504 :=
by sorry

end mathematics_competition_is_good_l2998_299821


namespace at_least_one_not_less_than_neg_two_l2998_299850

theorem at_least_one_not_less_than_neg_two
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 1/b ≥ -2) ∨ (b + 1/c ≥ -2) ∨ (c + 1/a ≥ -2) :=
by sorry

end at_least_one_not_less_than_neg_two_l2998_299850


namespace problem_solution_l2998_299801

theorem problem_solution :
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  (∀ p q : Prop, (¬(p ∨ q) → (¬p ∧ ¬q))) ∧
  ((∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
   (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≤ 2)) :=
by sorry

end problem_solution_l2998_299801


namespace f_abs_g_is_odd_l2998_299893

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem f_abs_g_is_odd 
  (hf : is_odd f) 
  (hg : is_even g) : 
  is_odd (λ x ↦ f x * |g x|) := by
  sorry

end f_abs_g_is_odd_l2998_299893


namespace second_rectangle_weight_l2998_299846

-- Define the properties of the rectangles
def length1 : ℝ := 4
def width1 : ℝ := 3
def weight1 : ℝ := 18
def length2 : ℝ := 6
def width2 : ℝ := 4

-- Theorem to prove
theorem second_rectangle_weight :
  ∀ (density : ℝ),
  density > 0 →
  let area1 := length1 * width1
  let area2 := length2 * width2
  let weight2 := (area2 / area1) * weight1
  weight2 = 36 := by
sorry

end second_rectangle_weight_l2998_299846


namespace fourth_root_256_times_cube_root_64_times_sqrt_16_l2998_299827

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 :
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 64 := by
  sorry

end fourth_root_256_times_cube_root_64_times_sqrt_16_l2998_299827


namespace product_of_differences_divisible_by_twelve_l2998_299815

theorem product_of_differences_divisible_by_twelve 
  (a b c d : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  12 ∣ ((a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d)) :=
by sorry

end product_of_differences_divisible_by_twelve_l2998_299815


namespace initial_pigs_l2998_299822

theorem initial_pigs (initial : ℕ) : initial + 22 = 86 → initial = 64 := by
  sorry

end initial_pigs_l2998_299822


namespace xyz_value_l2998_299817

-- Define the complex numbers x, y, and z
variable (x y z : ℂ)

-- Define the conditions
def condition1 : Prop := x * y + 5 * y = -20
def condition2 : Prop := y * z + 5 * z = -20
def condition3 : Prop := z * x + 5 * x = -20
def condition4 : Prop := x + y + z = 3

-- Theorem statement
theorem xyz_value (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) (h4 : condition4 x y z) :
  x * y * z = 105 := by
  sorry

end xyz_value_l2998_299817


namespace integer_roots_count_l2998_299849

/-- Represents a fourth-degree polynomial with integer coefficients -/
structure IntPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The number of integer roots of an IntPolynomial, counting multiplicity -/
def num_integer_roots (p : IntPolynomial) : ℕ := sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem integer_roots_count (p : IntPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 4 :=
sorry

end integer_roots_count_l2998_299849


namespace ten_mile_taxi_cost_l2998_299897

/-- Calculates the cost of a taxi ride -/
def taxiCost (baseFare mileCharge flatCharge thresholdMiles miles : ℚ) : ℚ :=
  baseFare + mileCharge * miles + if miles > thresholdMiles then flatCharge else 0

/-- Theorem: The cost of a 10-mile taxi ride is $5.50 -/
theorem ten_mile_taxi_cost :
  taxiCost 2 0.3 0.5 8 10 = 5.5 := by
  sorry

end ten_mile_taxi_cost_l2998_299897


namespace complex_coordinate_l2998_299844

theorem complex_coordinate (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) : 
  z = 4 - 2 * Complex.I := by
sorry

end complex_coordinate_l2998_299844


namespace fifty_third_odd_positive_integer_l2998_299816

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem: The 53rd odd positive integer is 105 -/
theorem fifty_third_odd_positive_integer : nthOddPositiveInteger 53 = 105 := by
  sorry

end fifty_third_odd_positive_integer_l2998_299816


namespace triangle_abc_isosceles_l2998_299866

/-- Given a triangle ABC where 2sin(A) * cos(B) = sin(C), prove that the triangle is isosceles -/
theorem triangle_abc_isosceles (A B C : ℝ) (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_condition : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B ∨ B = C ∨ A = C :=
sorry

end triangle_abc_isosceles_l2998_299866


namespace smallest_number_with_properties_l2998_299829

theorem smallest_number_with_properties : 
  ∃ (n : ℕ), n = 153846 ∧ 
  (∀ m : ℕ, m < n → 
    (m % 10 = 6 ∧ 
     ∃ k : ℕ, 6 * 10^k + (m - 6) / 10 = 4 * m) → False) ∧
  n % 10 = 6 ∧
  ∃ k : ℕ, 6 * 10^k + (n - 6) / 10 = 4 * n :=
sorry

end smallest_number_with_properties_l2998_299829


namespace project_hours_ratio_l2998_299886

/-- Proves that given the conditions of the project hours, the ratio of Pat's time to Kate's time is 4:3 -/
theorem project_hours_ratio :
  ∀ (pat kate mark : ℕ),
  pat + kate + mark = 189 →
  ∃ (r : ℚ), pat = r * kate →
  pat = (1 : ℚ) / 3 * mark →
  mark = kate + 105 →
  r = 4 / 3 := by
sorry

end project_hours_ratio_l2998_299886


namespace line_equation_from_triangle_l2998_299830

/-- Given a line passing through (-a, 0), (b, 0), and (0, h), where the area of the triangle
    formed in the second quadrant is T, prove that the equation of this line is
    2Tx - (b+a)^2y + 2T(b+a) = 0 -/
theorem line_equation_from_triangle (a b h T : ℝ) :
  (∃ (line : ℝ → ℝ → Prop),
    line (-a) 0 ∧
    line b 0 ∧
    line 0 h ∧
    (1/2 : ℝ) * (b + a) * h = T) →
  (∃ (line : ℝ → ℝ → Prop),
    ∀ x y, line x y ↔ 2 * T * x - (b + a)^2 * y + 2 * T * (b + a) = 0) :=
by sorry

end line_equation_from_triangle_l2998_299830


namespace james_teaching_years_l2998_299854

theorem james_teaching_years (james partner : ℕ) 
  (h1 : james = partner + 10)
  (h2 : james + partner = 70) : 
  james = 40 := by
sorry

end james_teaching_years_l2998_299854


namespace divisible_by_twelve_l2998_299859

theorem divisible_by_twelve (n : Nat) : n ≤ 9 → 5148 = 514 * 10 + n ↔ (514 * 10 + n) % 12 = 0 := by
  sorry

end divisible_by_twelve_l2998_299859


namespace existsNonIsoscelesWithFourEqualAreas_l2998_299875

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point
def Point := ℝ × ℝ

-- Function to check if a point is inside a triangle
def isInside (P : Point) (t : Triangle) : Prop := sorry

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop := sorry

-- Function to create smaller triangles by connecting P to vertices and drawing perpendiculars
def createSmallerTriangles (P : Point) (t : Triangle) : List Triangle := sorry

-- Function to check if 4 out of 6 triangles have equal areas
def fourEqualAreas (triangles : List Triangle) : Prop := sorry

-- The main theorem
theorem existsNonIsoscelesWithFourEqualAreas : 
  ∃ (t : Triangle) (P : Point), 
    isInside P t ∧ 
    ¬isIsosceles t ∧ 
    fourEqualAreas (createSmallerTriangles P t) := sorry

end existsNonIsoscelesWithFourEqualAreas_l2998_299875


namespace basketball_tournament_l2998_299853

theorem basketball_tournament (n : ℕ) (h_pos : n > 0) : 
  let total_players := 5 * n
  let total_matches := (total_players * (total_players - 1)) / 2
  let women_wins := 3 * total_matches / 7
  let men_wins := 4 * total_matches / 7
  (women_wins + men_wins = total_matches) → 
  (n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 5) :=
by sorry

end basketball_tournament_l2998_299853


namespace product_of_sum_and_cube_sum_l2998_299857

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (sum_eq : a + b = 5) 
  (cube_sum_eq : a^3 + b^3 = 125) : 
  a * b = 0 := by
sorry

end product_of_sum_and_cube_sum_l2998_299857


namespace parabola_directrix_l2998_299877

/-- Given a parabola defined by x = -1/8 * y^2, its directrix is x = 1/2 -/
theorem parabola_directrix (y : ℝ) : 
  let x := -1/8 * y^2
  let a := -1/8
  let focus_x := 1 / (4 * a)
  let directrix_x := -focus_x
  directrix_x = 1/2 := by sorry

end parabola_directrix_l2998_299877


namespace height_on_longest_side_l2998_299813

/-- Given a triangle with side lengths 6, 8, and 10, prove that the height on the longest side is 4.8 -/
theorem height_on_longest_side (a b c h : ℝ) : 
  a = 6 → b = 8 → c = 10 → 
  a^2 + b^2 = c^2 → 
  (1/2) * c * h = (1/2) * a * b → 
  h = 4.8 := by
  sorry

end height_on_longest_side_l2998_299813


namespace vector_properties_l2998_299895

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), ∀ i, v i = c * w i

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0 * w 0 + v 1 * w 1) = 0

theorem vector_properties :
  (∃ k : ℝ, parallel (fun i => k * (a i) + b i) (fun i => a i - 2 * (b i))) ∧
  perpendicular (fun i => (25/3) * (a i) + b i) (fun i => a i - 2 * (b i)) :=
by sorry

end vector_properties_l2998_299895


namespace max_triangle_perimeter_l2998_299885

theorem max_triangle_perimeter (x : ℕ) : 
  x > 0 ∧ x < 17 ∧ 8 + x > 9 ∧ 9 + x > 8 → 
  ∀ y : ℕ, y > 0 ∧ y < 17 ∧ 8 + y > 9 ∧ 9 + y > 8 → 
  8 + 9 + x ≥ 8 + 9 + y ∧ 
  8 + 9 + x ≤ 33 :=
by sorry

end max_triangle_perimeter_l2998_299885


namespace banana_boxes_theorem_l2998_299824

/-- The number of bananas Marilyn has -/
def total_bananas : ℕ := 40

/-- The number of bananas each box must contain -/
def bananas_per_box : ℕ := 5

/-- The number of boxes needed to store all bananas -/
def num_boxes : ℕ := total_bananas / bananas_per_box

theorem banana_boxes_theorem : num_boxes = 8 := by
  sorry

end banana_boxes_theorem_l2998_299824


namespace star_sum_24_five_pointed_star_24_l2998_299867

/-- Represents the vertices of a five-pointed star -/
inductive StarVertex
| A | B | C | D | E | F | G | H | J | K

/-- Assignment of numbers to the vertices of the star -/
def star_assignment : StarVertex → ℤ
| StarVertex.A => 1
| StarVertex.B => 2
| StarVertex.C => 3
| StarVertex.D => 4
| StarVertex.E => 5
| StarVertex.F => 10
| StarVertex.G => 12
| StarVertex.H => 9
| StarVertex.J => 6
| StarVertex.K => 8

/-- The set of all straight lines in the star -/
def star_lines : List (List StarVertex) := [
  [StarVertex.E, StarVertex.F, StarVertex.H, StarVertex.J],
  [StarVertex.F, StarVertex.G, StarVertex.K, StarVertex.J],
  [StarVertex.H, StarVertex.J, StarVertex.K, StarVertex.B],
  [StarVertex.J, StarVertex.E, StarVertex.K, StarVertex.C],
  [StarVertex.A, StarVertex.J, StarVertex.G, StarVertex.B]
]

/-- Theorem stating that the sum of numbers on each straight line equals 24 -/
theorem star_sum_24 : ∀ line ∈ star_lines, 
  (line.map star_assignment).sum = 24 := by sorry

/-- Main theorem proving the existence of a valid assignment -/
theorem five_pointed_star_24 : 
  ∃ (f : StarVertex → ℤ), ∀ line ∈ star_lines, (line.map f).sum = 24 := by
  use star_assignment
  exact star_sum_24

end star_sum_24_five_pointed_star_24_l2998_299867


namespace magician_card_decks_l2998_299833

theorem magician_card_decks (price : ℕ) (decks_left : ℕ) (earnings : ℕ) : 
  price = 7 → decks_left = 8 → earnings = 56 → 
  ∃ (initial_decks : ℕ), initial_decks = decks_left + earnings / price :=
by sorry

end magician_card_decks_l2998_299833


namespace inverse_function_theorem_l2998_299842

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_theorem (x : ℝ) (h : x > 0) :
  f (f_inv x) = x ∧ f_inv (f x) = x :=
by sorry

end inverse_function_theorem_l2998_299842


namespace mode_most_relevant_for_market_share_l2998_299834

/-- Represents a clothing model with its sales data -/
structure ClothingModel where
  id : ℕ
  sales : ℕ

/-- Represents a collection of clothing models -/
def ClothingModelData := List ClothingModel

/-- Calculates the mode of a list of natural numbers -/
def mode (l : List ℕ) : Option ℕ :=
  sorry

/-- Calculates the mean of a list of natural numbers -/
def mean (l : List ℕ) : ℚ :=
  sorry

/-- Calculates the median of a list of natural numbers -/
def median (l : List ℕ) : ℚ :=
  sorry

/-- Determines the most relevant statistical measure for market share survey -/
def mostRelevantMeasure (data : ClothingModelData) : String :=
  sorry

theorem mode_most_relevant_for_market_share (data : ClothingModelData) :
  mostRelevantMeasure data = "mode" :=
sorry

end mode_most_relevant_for_market_share_l2998_299834


namespace insect_crawl_properties_l2998_299808

def crawl_distances : List ℤ := [5, -3, 10, -8, -6, 12, -10]

theorem insect_crawl_properties :
  let cumulative_distances := crawl_distances.scanl (· + ·) 0
  (crawl_distances.sum = 0) ∧
  (cumulative_distances.map (Int.natAbs)).maximum? = some 14 ∧
  ((crawl_distances.map Int.natAbs).sum = 54) := by
  sorry

end insect_crawl_properties_l2998_299808


namespace min_entries_to_four_coins_l2998_299804

/-- Represents the state of coins and last entry -/
structure CoinState :=
  (coins : ℕ)
  (lastEntry : ℕ)

/-- Defines the coin machine rules -/
def coinMachine (entry : ℕ) : ℕ :=
  match entry with
  | 7 => 3
  | 8 => 11
  | 9 => 4
  | _ => 0

/-- Checks if an entry is valid -/
def isValidEntry (state : CoinState) (entry : ℕ) : Bool :=
  state.coins ≥ entry ∧ entry ≠ state.lastEntry ∧ (entry = 7 ∨ entry = 8 ∨ entry = 9)

/-- Makes an entry and returns the new state -/
def makeEntry (state : CoinState) (entry : ℕ) : CoinState :=
  { coins := state.coins - entry + coinMachine entry,
    lastEntry := entry }

/-- Defines the minimum number of entries to reach the target -/
def minEntries (start : ℕ) (target : ℕ) : ℕ := sorry

/-- Theorem stating the minimum number of entries to reach 4 coins from 15 coins is 4 -/
theorem min_entries_to_four_coins :
  minEntries 15 4 = 4 := by sorry

end min_entries_to_four_coins_l2998_299804


namespace exists_triangle_no_isosceles_triangle_l2998_299884

/-- The set of stick lengths -/
def stick_lengths : List ℝ := [1, 1.9, 1.9^2, 1.9^3, 1.9^4, 1.9^5, 1.9^6, 1.9^7, 1.9^8, 1.9^9]

/-- Function to check if three lengths can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three lengths can form an isosceles triangle -/
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  is_triangle a b c ∧ (a = b ∨ b = c ∨ c = a)

/-- Theorem stating that a triangle can be formed from the given stick lengths -/
theorem exists_triangle : ∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_triangle a b c :=
sorry

/-- Theorem stating that an isosceles triangle cannot be formed from the given stick lengths -/
theorem no_isosceles_triangle : ¬∃ (a b c : ℝ), a ∈ stick_lengths ∧ b ∈ stick_lengths ∧ c ∈ stick_lengths ∧ is_isosceles_triangle a b c :=
sorry

end exists_triangle_no_isosceles_triangle_l2998_299884


namespace vector_sum_coords_l2998_299809

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum_coords : 
  (2 : ℝ) • a + b = (-3, 4) := by sorry

end vector_sum_coords_l2998_299809


namespace geometric_concept_word_counts_l2998_299805

/-- A type representing geometric concepts -/
def GeometricConcept : Type := String

/-- A function that counts the number of words in a string -/
def wordCount (s : String) : Nat :=
  s.split (· == ' ') |>.length

/-- Theorem stating that there exist geometric concepts expressible in 1, 2, 3, and 4 words -/
theorem geometric_concept_word_counts :
  ∃ (a b c d : GeometricConcept),
    wordCount a = 1 ∧
    wordCount b = 2 ∧
    wordCount c = 3 ∧
    wordCount d = 4 :=
by sorry


end geometric_concept_word_counts_l2998_299805


namespace kaleb_final_score_l2998_299855

/-- Calculates Kaleb's final adjusted score in a trivia game -/
theorem kaleb_final_score : 
  let first_half_score : ℝ := 43
  let first_half_bonus1 : ℝ := 0.20
  let first_half_bonus2 : ℝ := 0.05
  let second_half_score : ℝ := 23
  let second_half_penalty1 : ℝ := 0.10
  let second_half_penalty2 : ℝ := 0.08
  
  let first_half_adjusted := first_half_score * (1 + first_half_bonus1 + first_half_bonus2)
  let second_half_adjusted := second_half_score * (1 - second_half_penalty1 - second_half_penalty2)
  
  first_half_adjusted + second_half_adjusted = 72.61
  := by sorry

end kaleb_final_score_l2998_299855


namespace double_angle_sine_fifteen_degrees_l2998_299838

theorem double_angle_sine_fifteen_degrees :
  2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end double_angle_sine_fifteen_degrees_l2998_299838


namespace pyramid_section_ratio_l2998_299889

/-- Represents a pyramid with a side edge and two points on it -/
structure Pyramid where
  -- Side edge length
  ab : ℝ
  -- Position of point K from A
  ak : ℝ
  -- Position of point M from A
  am : ℝ
  -- Conditions
  ab_pos : 0 < ab
  k_on_ab : 0 ≤ ak ∧ ak ≤ ab
  m_on_ab : 0 ≤ am ∧ am ≤ ab
  ak_eq_bm : ak = ab - am
  sections_area : (ak / ab)^2 + (am / ab)^2 = 2/3

/-- The main theorem -/
theorem pyramid_section_ratio (p : Pyramid) : (p.am - p.ak) / p.ab = 1 / Real.sqrt 3 := by
  sorry


end pyramid_section_ratio_l2998_299889


namespace trip_duration_l2998_299858

-- Define the type for time
structure Time where
  hours : ℕ
  minutes : ℕ

-- Define the function to calculate the angle between clock hands
def angleBetweenHands (t : Time) : ℝ := sorry

-- Define the function to find the time when hands are at a specific angle
def timeAtAngle (startHour startMinute : ℕ) (angle : ℝ) : Time := sorry

-- Define the function to calculate time difference
def timeDifference (t1 t2 : Time) : Time := sorry

-- The main theorem
theorem trip_duration : 
  let startTime := timeAtAngle 7 0 90
  let endTime := timeAtAngle 15 0 270
  let duration := timeDifference startTime endTime
  duration = Time.mk 8 29 := by sorry

end trip_duration_l2998_299858


namespace no_function_pair_exists_l2998_299865

theorem no_function_pair_exists : ¬∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3 := by
  sorry

end no_function_pair_exists_l2998_299865


namespace min_nSn_l2998_299840

/-- An arithmetic sequence with sum properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  sum_10 : S 10 = 0
  sum_15 : S 15 = 25

/-- The product of n and S_n for an arithmetic sequence -/
def nSn (seq : ArithmeticSequence) (n : ℕ) : ℝ := n * seq.S n

/-- The minimum value of nS_n for the given arithmetic sequence -/
theorem min_nSn (seq : ArithmeticSequence) :
  ∃ (min : ℝ), min = -49 ∧ ∀ (n : ℕ), n ≠ 0 → min ≤ nSn seq n :=
sorry

end min_nSn_l2998_299840


namespace otimes_equation_roots_l2998_299871

-- Define the new operation
def otimes (a b : ℝ) : ℝ := a * b^2 - b

-- Theorem statement
theorem otimes_equation_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ otimes 1 x = k ∧ otimes 1 y = k) ↔ k > -1/4 :=
sorry

end otimes_equation_roots_l2998_299871


namespace system_solution_unique_l2998_299873

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x + 2 * y = 5 ∧ x - 2 * y = 11 ∧ x = 4 ∧ y = -7/2 := by
  sorry

end system_solution_unique_l2998_299873


namespace factorial_plus_one_eq_power_l2998_299800

theorem factorial_plus_one_eq_power (n p : ℕ) : 
  (Nat.factorial (p - 1) + 1 = p ^ n) ↔ 
  ((n = 1 ∧ p = 2) ∨ (n = 1 ∧ p = 3) ∨ (n = 2 ∧ p = 5)) :=
sorry

end factorial_plus_one_eq_power_l2998_299800


namespace existence_of_sum_greater_than_one_l2998_299818

theorem existence_of_sum_greater_than_one : 
  ¬(∀ (x y : ℝ), x + y ≤ 1) := by sorry

end existence_of_sum_greater_than_one_l2998_299818


namespace center_shade_ratio_l2998_299806

/-- Represents a square grid -/
structure SquareGrid (n : ℕ) where
  size : ℕ
  total_area : ℝ
  cell_area : ℝ
  h_size : size = n
  h_cell_area : cell_area = total_area / (n^2 : ℝ)

/-- Represents a shaded region in the center of the grid -/
structure CenterShade (grid : SquareGrid 5) where
  area : ℝ
  h_area : area = 4 * (grid.cell_area / 2)

/-- The theorem stating the ratio of the shaded area to the total area -/
theorem center_shade_ratio (grid : SquareGrid 5) (shade : CenterShade grid) :
  shade.area / grid.total_area = 2 / 25 := by
  sorry

end center_shade_ratio_l2998_299806


namespace bill_difference_l2998_299841

theorem bill_difference : 
  ∀ (alice_bill bob_bill : ℝ),
  alice_bill * 0.25 = 5 →
  bob_bill * 0.10 = 4 →
  bob_bill - alice_bill = 20 :=
by
  sorry

end bill_difference_l2998_299841
