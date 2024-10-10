import Mathlib

namespace framed_painting_ratio_l2005_200501

-- Define the painting dimensions
def painting_width : ℝ := 20
def painting_height : ℝ := 30

-- Define the frame width variable
variable (x : ℝ)

-- Define the framed painting dimensions
def framed_width (x : ℝ) : ℝ := painting_width + 2 * x
def framed_height (x : ℝ) : ℝ := painting_height + 4 * x

-- State the theorem
theorem framed_painting_ratio :
  (∃ x > 0, framed_width x * framed_height x = 2 * painting_width * painting_height) →
  (min (framed_width x) (framed_height x)) / (max (framed_width x) (framed_height x)) = 3 / 5 := by
  sorry


end framed_painting_ratio_l2005_200501


namespace trigonometric_identities_l2005_200582

theorem trigonometric_identities (θ : ℝ) :
  (2 * Real.cos ((3 / 2) * Real.pi + θ) + Real.cos (Real.pi + θ)) /
  (3 * Real.sin (Real.pi - θ) + 2 * Real.sin ((5 / 2) * Real.pi + θ)) = 1 / 5 →
  (Real.tan θ = 3 / 13 ∧
   Real.sin θ ^ 2 + 3 * Real.sin θ * Real.cos θ = 20160 / 28561) :=
by sorry

end trigonometric_identities_l2005_200582


namespace arithmetic_geometric_mean_problem_l2005_200519

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20)
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
sorry

end arithmetic_geometric_mean_problem_l2005_200519


namespace not_prime_29n_plus_11_l2005_200514

theorem not_prime_29n_plus_11 (n : ℕ+) 
  (h1 : ∃ x : ℕ, 3 * n + 1 = x^2) 
  (h2 : ∃ y : ℕ, 10 * n + 1 = y^2) : 
  ¬ Nat.Prime (29 * n + 11) := by
sorry

end not_prime_29n_plus_11_l2005_200514


namespace geometric_sequence_common_ratio_l2005_200578

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of its first n terms,
    if 3S_3 = a_4 - 2 and 3S_2 = a_3 - 2, then q = 4 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : 3 * S 3 = a 4 - 2)
  (h4 : 3 * S 2 = a 3 - 2) :
  q = 4 := by
sorry

end geometric_sequence_common_ratio_l2005_200578


namespace system_solvability_l2005_200556

-- Define the system of equations
def system_of_equations (x y z a b c : ℝ) : Prop :=
  (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c

-- Define the solvability condition
def solvability_condition (a b c : ℝ) : Prop :=
  a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0

-- Theorem statement
theorem system_solvability (a b c : ℝ) :
  (∃ x y z : ℝ, system_of_equations x y z a b c) ↔ solvability_condition a b c :=
sorry

end system_solvability_l2005_200556


namespace certain_number_calculation_l2005_200591

theorem certain_number_calculation : 
  ∃ x : ℝ, abs (3889 + 12.952 - 47.95000000000027 - x) < 0.0005 ∧ 
           abs (x - 3854.002) < 0.0005 := by
  sorry

end certain_number_calculation_l2005_200591


namespace extremum_condition_l2005_200507

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_condition (a b : ℝ) : 
  (f a b 1 = 10) ∧ (f_derivative a b 1 = 0) → (a = 4 ∧ b = -11) :=
sorry

end extremum_condition_l2005_200507


namespace largest_expression_l2005_200588

theorem largest_expression : 
  (100 - 0 > 0 / 100) ∧ (100 - 0 > 0 * 100) := by sorry

end largest_expression_l2005_200588


namespace system_solution_l2005_200593

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k - 1 → 
  2*x + y = 5*k + 4 → 
  x + y = 5 → 
  k = 2 := by
sorry

end system_solution_l2005_200593


namespace store_discount_income_increase_l2005_200569

theorem store_discount_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (quantity_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1)
  (h2 : quantity_increase_rate = 0.12)
  : (1 + quantity_increase_rate) * (1 - discount_rate) - 1 = 0.008 := by
  sorry

end store_discount_income_increase_l2005_200569


namespace pizza_slice_volume_l2005_200564

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume :
  let thickness : ℝ := 1/2
  let diameter : ℝ := 12
  let num_slices : ℕ := 8
  let radius : ℝ := diameter / 2
  let pizza_volume : ℝ := π * radius^2 * thickness
  let slice_volume : ℝ := pizza_volume / num_slices
  slice_volume = 9*π/4 := by sorry

end pizza_slice_volume_l2005_200564


namespace total_toys_l2005_200559

theorem total_toys (bill_toys : ℕ) (hash_toys : ℕ) : 
  bill_toys = 60 → 
  hash_toys = (bill_toys / 2) + 9 → 
  bill_toys + hash_toys = 99 :=
by sorry

end total_toys_l2005_200559


namespace locus_is_S_l2005_200595

/-- A point moving along a line with constant velocity -/
structure MovingPoint where
  line : Set ℝ × ℝ  -- Represents a line in 2D space
  velocity : ℝ

/-- The locus of lines XX' -/
def locus (X X' : MovingPoint) : Set (Set ℝ × ℝ) := sorry

/-- The specific set S that represents the correct locus -/
def S : Set (Set ℝ × ℝ) := sorry

/-- Theorem stating that the locus of lines XX' is the set S -/
theorem locus_is_S (X X' : MovingPoint) (h : X.velocity ≠ X'.velocity) :
  locus X X' = S := by sorry

end locus_is_S_l2005_200595


namespace farm_legs_count_l2005_200510

/-- The number of legs for a given animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "chicken" => 2
  | "sheep" => 4
  | _ => 0

/-- The total number of animals in the farm -/
def total_animals : ℕ := 20

/-- The number of sheep in the farm -/
def num_sheep : ℕ := 10

/-- The number of chickens in the farm -/
def num_chickens : ℕ := total_animals - num_sheep

theorem farm_legs_count : 
  (num_sheep * legs_per_animal "sheep") + (num_chickens * legs_per_animal "chicken") = 60 := by
  sorry

end farm_legs_count_l2005_200510


namespace sum_of_common_terms_l2005_200516

/-- The sequence formed by common terms of {2n-1} and {3n-2} in ascending order -/
def a : ℕ → ℕ := sorry

/-- The sum of the first n terms of sequence a -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the first n terms of sequence a is 3n^2 - 2n -/
theorem sum_of_common_terms (n : ℕ) : S n = 3 * n^2 - 2 * n := by sorry

end sum_of_common_terms_l2005_200516


namespace fruit_seller_apples_l2005_200547

theorem fruit_seller_apples (initial_stock : ℕ) (remaining_stock : ℕ) 
  (sell_percentage : ℚ) (h1 : sell_percentage = 40 / 100) 
  (h2 : remaining_stock = 420) 
  (h3 : remaining_stock = initial_stock - (sell_percentage * initial_stock).floor) : 
  initial_stock = 700 := by
sorry

end fruit_seller_apples_l2005_200547


namespace gcf_of_60_and_90_l2005_200529

theorem gcf_of_60_and_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_of_60_and_90_l2005_200529


namespace diophantine_equation_min_max_sum_l2005_200531

theorem diophantine_equation_min_max_sum : 
  ∃ (p q : ℕ), 
    (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ 6 * x + 7 * y = 2012 → x + y ≥ p) ∧
    (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ 6 * x + 7 * y = 2012 → x + y ≤ q) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℕ), 
      x₁ > 0 ∧ y₁ > 0 ∧ 6 * x₁ + 7 * y₁ = 2012 ∧ x₁ + y₁ = p ∧
      x₂ > 0 ∧ y₂ > 0 ∧ 6 * x₂ + 7 * y₂ = 2012 ∧ x₂ + y₂ = q) ∧
    p + q = 623 := by
  sorry

end diophantine_equation_min_max_sum_l2005_200531


namespace problem_solution_l2005_200565

theorem problem_solution (x y : ℝ) (h1 : 0.2 * x = 200) (h2 : 0.3 * y = 150) :
  (0.8 * x - 0.5 * y) + 0.4 * (x + y) = 1150 := by
  sorry

end problem_solution_l2005_200565


namespace sine_arithmetic_sequence_l2005_200526

open Real

theorem sine_arithmetic_sequence (a : ℝ) : 
  0 < a ∧ a < 2 * π →
  (∃ r : ℝ, sin a + r = sin (2 * a) ∧ sin (2 * a) + r = sin (3 * a)) ↔ 
  a = π / 2 ∨ a = 3 * π / 2 := by
sorry

end sine_arithmetic_sequence_l2005_200526


namespace polynomial_division_l2005_200504

theorem polynomial_division (x : ℝ) :
  x^5 - 17*x^3 + 8*x^2 - 9*x + 12 = (x - 3) * (x^4 + 3*x^3 - 8*x^2 - 16*x - 57) + (-159) := by
  sorry

end polynomial_division_l2005_200504


namespace equation_solution_l2005_200576

theorem equation_solution (x y z : ℕ) :
  (x : ℚ) + 1 / ((y : ℚ) + 1 / (z : ℚ)) = 10 / 7 →
  x = 1 ∧ y = 2 ∧ z = 3 := by
sorry

end equation_solution_l2005_200576


namespace store_profit_percentage_l2005_200539

/-- Calculates the profit percentage on items sold in February given the markups and discount -/
theorem store_profit_percentage (initial_markup : ℝ) (new_year_markup : ℝ) (february_discount : ℝ) :
  initial_markup = 0.20 →
  new_year_markup = 0.25 →
  february_discount = 0.20 →
  (1 + initial_markup + new_year_markup * (1 + initial_markup)) * (1 - february_discount) - 1 = 0.20 := by
  sorry

#check store_profit_percentage

end store_profit_percentage_l2005_200539


namespace bridget_erasers_l2005_200528

theorem bridget_erasers (initial final given : ℕ) : 
  initial = 8 → final = 11 → given = final - initial :=
by
  sorry

end bridget_erasers_l2005_200528


namespace carys_savings_l2005_200583

/-- Problem: Cary's Lawn Mowing Savings --/
theorem carys_savings (shoe_cost : ℕ) (saved : ℕ) (earnings_per_lawn : ℕ) (lawns_per_weekend : ℕ)
  (h1 : shoe_cost = 120)
  (h2 : saved = 30)
  (h3 : earnings_per_lawn = 5)
  (h4 : lawns_per_weekend = 3) :
  (shoe_cost - saved) / (lawns_per_weekend * earnings_per_lawn) = 6 := by
  sorry

end carys_savings_l2005_200583


namespace program_cost_calculation_l2005_200532

/-- Calculates the total cost for running a computer program -/
theorem program_cost_calculation (program_time_seconds : ℝ) : 
  let milliseconds_per_second : ℝ := 1000
  let overhead_cost : ℝ := 1.07
  let cost_per_millisecond : ℝ := 0.023
  let tape_mounting_cost : ℝ := 5.35
  let program_time_milliseconds : ℝ := program_time_seconds * milliseconds_per_second
  let computer_time_cost : ℝ := program_time_milliseconds * cost_per_millisecond
  let total_cost : ℝ := overhead_cost + computer_time_cost + tape_mounting_cost
  program_time_seconds = 1.5 → total_cost = 40.92 := by
  sorry

end program_cost_calculation_l2005_200532


namespace program_output_l2005_200548

def program (a b : ℤ) : ℤ :=
  if a > b then a else b

theorem program_output : program 2 3 = 3 := by sorry

end program_output_l2005_200548


namespace bryan_pushups_l2005_200533

/-- The number of push-ups Bryan did in total -/
def total_pushups (sets : ℕ) (pushups_per_set : ℕ) (reduction : ℕ) : ℕ :=
  (sets - 1) * pushups_per_set + (pushups_per_set - reduction)

/-- Proof that Bryan did 40 push-ups in total -/
theorem bryan_pushups :
  total_pushups 3 15 5 = 40 := by
  sorry

end bryan_pushups_l2005_200533


namespace sum_positive_from_inequality_l2005_200513

theorem sum_positive_from_inequality (x y : ℝ) 
  (h : (3:ℝ)^x + (5:ℝ)^y > (3:ℝ)^(-y) + (5:ℝ)^(-x)) : 
  x + y > 0 := by
  sorry

end sum_positive_from_inequality_l2005_200513


namespace comparison_theorem_l2005_200549

theorem comparison_theorem :
  (-3/4 : ℚ) > -4/5 ∧ (3 : ℝ) > Real.rpow 9 (1/3) := by sorry

end comparison_theorem_l2005_200549


namespace prob_sum_le_10_is_25_72_l2005_200571

/-- The number of possible outcomes when rolling three fair six-sided dice -/
def total_outcomes : ℕ := 6^3

/-- The number of favorable outcomes (sum ≤ 10) when rolling three fair six-sided dice -/
def favorable_outcomes : ℕ := 75

/-- The probability of rolling three fair six-sided dice and obtaining a sum less than or equal to 10 -/
def prob_sum_le_10 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_le_10_is_25_72 : prob_sum_le_10 = 25 / 72 := by
  sorry

end prob_sum_le_10_is_25_72_l2005_200571


namespace root_transformation_l2005_200545

/-- Given that s₁, s₂, and s₃ are the roots of x³ - 4x² + 9 = 0,
    prove that 3s₁, 3s₂, and 3s₃ are the roots of x³ - 12x² + 243 = 0 -/
theorem root_transformation (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 9 = 0) ∧ 
  (s₂^3 - 4*s₂^2 + 9 = 0) ∧ 
  (s₃^3 - 4*s₃^2 + 9 = 0) → 
  ((3*s₁)^3 - 12*(3*s₁)^2 + 243 = 0) ∧ 
  ((3*s₂)^3 - 12*(3*s₂)^2 + 243 = 0) ∧ 
  ((3*s₃)^3 - 12*(3*s₃)^2 + 243 = 0) := by
sorry

end root_transformation_l2005_200545


namespace men_working_count_l2005_200508

/-- Represents the amount of work done by one person in one hour -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkGroup where
  size : ℕ
  work_rate : WorkRate
  days : ℕ
  hours_per_day : ℕ

/-- The total work done by a group is the product of their size, work rate, days, and hours per day -/
def total_work (group : WorkGroup) : ℝ :=
  group.size * group.work_rate.rate * group.days * group.hours_per_day

/-- Given the conditions of the problem, prove that the number of men working is 15 -/
theorem men_working_count (men_group women_group : WorkGroup) :
  men_group.days = 21 →
  men_group.hours_per_day = 8 →
  women_group.size = 21 →
  women_group.days = 60 →
  women_group.hours_per_day = 3 →
  3 * women_group.work_rate.rate = 2 * men_group.work_rate.rate →
  total_work men_group = total_work women_group →
  men_group.size = 15 := by
  sorry

end men_working_count_l2005_200508


namespace trigonometric_expression_l2005_200575

theorem trigonometric_expression (α : Real) 
  (h : Real.sin (π/4 + α) = 1/2) : 
  (Real.sin (5*π/4 + α) / Real.cos (9*π/4 + α)) * Real.cos (7*π/4 - α) = -1/2 := by
  sorry

end trigonometric_expression_l2005_200575


namespace joe_is_94_point_5_inches_tall_l2005_200585

-- Define the heights of Sara, Joe, and Alex
variable (S J A : ℝ)

-- Define the conditions from the problem
def combined_height : ℝ → ℝ → ℝ → Prop :=
  λ s j a => s + j + a = 180

def joe_height : ℝ → ℝ → Prop :=
  λ s j => j = 2 * s + 6

def alex_height : ℝ → ℝ → Prop :=
  λ s a => a = s - 3

-- Theorem statement
theorem joe_is_94_point_5_inches_tall
  (h1 : combined_height S J A)
  (h2 : joe_height S J)
  (h3 : alex_height S A) :
  J = 94.5 :=
sorry

end joe_is_94_point_5_inches_tall_l2005_200585


namespace opposite_point_exists_l2005_200570

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define diametrically opposite points
def DiametricallyOpposite (c : Circle) (p q : ℝ × ℝ) : Prop :=
  PointOnCircle c p ∧ PointOnCircle c q ∧
  (p.1 - c.center.1) = -(q.1 - c.center.1) ∧
  (p.2 - c.center.2) = -(q.2 - c.center.2)

-- Theorem statement
theorem opposite_point_exists (c : Circle) (A₁ : ℝ × ℝ) 
  (h : PointOnCircle c A₁) : 
  ∃ B₂ : ℝ × ℝ, DiametricallyOpposite c A₁ B₂ := by
  sorry

end opposite_point_exists_l2005_200570


namespace real_solutions_quadratic_l2005_200561

theorem real_solutions_quadratic (x : ℝ) : 
  (∃ y : ℝ, 4 * y^2 + 4 * x * y + x + 6 = 0) ↔ x ≤ -2 ∨ x ≥ 3 := by
  sorry

end real_solutions_quadratic_l2005_200561


namespace no_equal_product_l2005_200589

theorem no_equal_product (x y : ℕ) : x * (x + 1) ≠ 4 * y * (y + 1) := by
  sorry

end no_equal_product_l2005_200589


namespace carpet_price_falls_below_8_at_945_l2005_200568

def initial_price : ℝ := 10.00
def reduction_rate : ℝ := 0.9
def target_price : ℝ := 8.00

def price_after_n_reductions (n : ℕ) : ℝ :=
  initial_price * (reduction_rate ^ n)

theorem carpet_price_falls_below_8_at_945 :
  price_after_n_reductions 3 < target_price ∧
  price_after_n_reductions 2 ≥ target_price :=
by sorry

end carpet_price_falls_below_8_at_945_l2005_200568


namespace triangle_sqrt_inequality_l2005_200592

theorem triangle_sqrt_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hac : a + c > b) :
  (Real.sqrt a + Real.sqrt b > Real.sqrt c) ∧ 
  (Real.sqrt b + Real.sqrt c > Real.sqrt a) ∧ 
  (Real.sqrt a + Real.sqrt c > Real.sqrt b) := by
sorry

end triangle_sqrt_inequality_l2005_200592


namespace rectangular_prism_volume_l2005_200536

/-- Given a rectangular prism with side areas 15, 10, and 6 (in square inches),
    where the dimension associated with the smallest area is the hypotenuse of a right triangle
    formed by the other two dimensions, prove that the volume of the prism is 30 cubic inches. -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := by
  sorry

end rectangular_prism_volume_l2005_200536


namespace father_seven_times_son_age_l2005_200581

/-- 
Given a father who is currently 38 years old and a son who is currently 14 years old,
this theorem proves that 10 years ago, the father was seven times as old as his son.
-/
theorem father_seven_times_son_age (father_age : Nat) (son_age : Nat) (years_ago : Nat) : 
  father_age = 38 → son_age = 14 → 
  (father_age - years_ago) = 7 * (son_age - years_ago) → 
  years_ago = 10 := by
sorry

end father_seven_times_son_age_l2005_200581


namespace product_of_roots_l2005_200523

theorem product_of_roots (x : ℝ) : 
  (24 * x^2 + 36 * x - 648 = 0) → 
  (∃ r₁ r₂ : ℝ, (24 * r₁^2 + 36 * r₁ - 648 = 0) ∧ 
                (24 * r₂^2 + 36 * r₂ - 648 = 0) ∧ 
                (r₁ * r₂ = -27)) := by
  sorry

end product_of_roots_l2005_200523


namespace expand_and_simplify_l2005_200540

theorem expand_and_simplify (x y : ℝ) :
  x * (x - 3 * y) + (2 * x - y)^2 = 5 * x^2 - 7 * x * y + y^2 := by
  sorry

end expand_and_simplify_l2005_200540


namespace minnow_count_l2005_200538

theorem minnow_count (total : ℕ) (red green white : ℕ) : 
  (red : ℚ) / total = 2/5 →
  (green : ℚ) / total = 3/10 →
  white + red + green = total →
  red = 20 →
  white = 15 := by
sorry

end minnow_count_l2005_200538


namespace juan_marbles_count_l2005_200500

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of additional marbles Juan has compared to Connie -/
def juan_additional_marbles : ℕ := 25

/-- The total number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + juan_additional_marbles

theorem juan_marbles_count : juan_marbles = 64 := by sorry

end juan_marbles_count_l2005_200500


namespace arithmetic_mean_problem_l2005_200567

theorem arithmetic_mean_problem (x : ℝ) : (x + 1 = (5 + 7) / 2) → x = 5 := by
  sorry

end arithmetic_mean_problem_l2005_200567


namespace perpendicular_lines_not_both_perpendicular_to_plane_l2005_200566

-- Define the plane α
variable (α : Set (ℝ × ℝ × ℝ))

-- Define lines a and b
variable (a b : Set (ℝ × ℝ × ℝ))

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for a line to be perpendicular to a plane
def perpendicular_to_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- The theorem
theorem perpendicular_lines_not_both_perpendicular_to_plane :
  a ≠ b →
  perpendicular a b →
  ¬(perpendicular_to_plane a α ∧ perpendicular_to_plane b α) := by
  sorry

end perpendicular_lines_not_both_perpendicular_to_plane_l2005_200566


namespace no_rain_probability_l2005_200572

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end no_rain_probability_l2005_200572


namespace nested_sqrt_value_l2005_200534

theorem nested_sqrt_value :
  ∃ y : ℝ, y = Real.sqrt (3 + y) → y = (1 + Real.sqrt 13) / 2 := by
  sorry

end nested_sqrt_value_l2005_200534


namespace always_negative_monotone_decreasing_l2005_200580

/-- The function f(x) = kx^2 - 2x + 4k -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 4 * k

/-- Theorem 1: f(x) is always less than zero on ℝ iff k < -1/2 -/
theorem always_negative (k : ℝ) : (∀ x : ℝ, f k x < 0) ↔ k < -1/2 := by sorry

/-- Theorem 2: f(x) is monotonically decreasing on [2, 4] iff k ≤ 1/4 -/
theorem monotone_decreasing (k : ℝ) : 
  (∀ x y : ℝ, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f k x > f k y) ↔ k ≤ 1/4 := by sorry

end always_negative_monotone_decreasing_l2005_200580


namespace sock_pairs_calculation_l2005_200535

def calculate_sock_pairs (initial_socks thrown_away_socks new_socks : ℕ) : ℕ :=
  ((initial_socks - thrown_away_socks) + new_socks) / 2

theorem sock_pairs_calculation (initial_socks thrown_away_socks new_socks : ℕ) 
  (h1 : initial_socks ≥ thrown_away_socks) :
  calculate_sock_pairs initial_socks thrown_away_socks new_socks = 
  ((initial_socks - thrown_away_socks) + new_socks) / 2 := by
  sorry

#eval calculate_sock_pairs 28 4 36

end sock_pairs_calculation_l2005_200535


namespace find_a_find_m_range_l2005_200573

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + a

-- Statement 1
theorem find_a :
  (∀ x : ℝ, f 2 x < 5 ↔ -3/2 < x ∧ x < 1) →
  (∃! a : ℝ, ∀ x : ℝ, f a x < 5 ↔ -3/2 < x ∧ x < 1) :=
sorry

-- Statement 2
theorem find_m_range (m : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 5 → 2 * x^2 + x + 2 > m * x) →
  m < 5 :=
sorry

end find_a_find_m_range_l2005_200573


namespace solution_part_i_solution_part_ii_l2005_200599

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for part (I)
theorem solution_part_i :
  {x : ℝ | f x ≤ 4} = {x : ℝ | -5/3 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part (II)
theorem solution_part_ii :
  {a : ℝ | ∀ x ∈ {x : ℝ | f x ≤ 4}, |x + 3| + |x + a| < x + 6} =
  {a : ℝ | -4/3 < a ∧ a < 2} := by sorry

end solution_part_i_solution_part_ii_l2005_200599


namespace P_sufficient_not_necessary_for_Q_l2005_200512

-- Define propositions P and Q
def P (a b : ℝ) : Prop := a > b ∧ b > 0
def Q (a b : ℝ) : Prop := a^2 > b^2

-- Theorem stating that P is sufficient but not necessary for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ a b : ℝ, P a b → Q a b) ∧
  ¬(∀ a b : ℝ, Q a b → P a b) :=
sorry

end P_sufficient_not_necessary_for_Q_l2005_200512


namespace tamika_always_wins_l2005_200558

def tamika_set : Finset Nat := {6, 7, 8}
def carlos_set : Finset Nat := {2, 3, 5}

theorem tamika_always_wins :
  ∀ (t1 t2 : Nat) (c1 c2 : Nat),
    t1 ∈ tamika_set → t2 ∈ tamika_set → t1 ≠ t2 →
    c1 ∈ carlos_set → c2 ∈ carlos_set → c1 ≠ c2 →
    t1 * t2 > c1 * c2 := by
  sorry

#check tamika_always_wins

end tamika_always_wins_l2005_200558


namespace diophantine_equation_solutions_l2005_200522

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ) := {(3995, 3993), (1, -1), (1999, 3996005), (3996005, 1997), (1997, -3996005), (-3996005, 1995)}
  ∀ x y : ℤ, (1996 * x + 1998 * y + 1 = x * y) ↔ (x, y) ∈ S :=
by sorry

end diophantine_equation_solutions_l2005_200522


namespace triangle_area_passes_through_1_2_passes_through_neg1_6_x_intercept_correct_y_intercept_correct_l2005_200584

/-- A linear function passing through (1, 2) and (-1, 6) -/
def linear_function (x : ℝ) : ℝ := -2 * x + 4

/-- The x-intercept of the linear function -/
def x_intercept : ℝ := 2

/-- The y-intercept of the linear function -/
def y_intercept : ℝ := 4

/-- Theorem: The area of the triangle formed by the x-intercept, y-intercept, and origin is 4 -/
theorem triangle_area : (1/2 : ℝ) * x_intercept * y_intercept = 4 := by
  sorry

/-- The linear function passes through (1, 2) -/
theorem passes_through_1_2 : linear_function 1 = 2 := by
  sorry

/-- The linear function passes through (-1, 6) -/
theorem passes_through_neg1_6 : linear_function (-1) = 6 := by
  sorry

/-- The x-intercept is correct -/
theorem x_intercept_correct : linear_function x_intercept = 0 := by
  sorry

/-- The y-intercept is correct -/
theorem y_intercept_correct : linear_function 0 = y_intercept := by
  sorry

end triangle_area_passes_through_1_2_passes_through_neg1_6_x_intercept_correct_y_intercept_correct_l2005_200584


namespace concurrency_condition_l2005_200596

/-- Triangle ABC with sides a, b, and c, where AD is an altitude, BE is an angle bisector, and CF is a median -/
structure Triangle :=
  (a b c : ℝ)
  (ad_is_altitude : Bool)
  (be_is_angle_bisector : Bool)
  (cf_is_median : Bool)

/-- The lines AD, BE, and CF are concurrent -/
def are_concurrent (t : Triangle) : Prop := sorry

/-- Theorem stating the condition for concurrency of AD, BE, and CF -/
theorem concurrency_condition (t : Triangle) : 
  are_concurrent t ↔ t.a^2 * (t.a - t.c) = (t.b^2 - t.c^2) * (t.a + t.c) :=
sorry

end concurrency_condition_l2005_200596


namespace computer_table_price_l2005_200506

theorem computer_table_price (cost_price : ℝ) (markup_percentage : ℝ) 
  (h1 : cost_price = 4090.9090909090905)
  (h2 : markup_percentage = 32) :
  cost_price * (1 + markup_percentage / 100) = 5400 := by
  sorry

end computer_table_price_l2005_200506


namespace clearview_soccer_league_members_l2005_200594

/-- Represents the Clearview Soccer League --/
structure SoccerLeague where
  sockPrice : ℕ
  tshirtPriceIncrease : ℕ
  hatPrice : ℕ
  totalExpenditure : ℕ

/-- Calculates the number of members in the league --/
def calculateMembers (league : SoccerLeague) : ℕ :=
  let tshirtPrice := league.sockPrice + league.tshirtPriceIncrease
  let memberCost := 2 * (league.sockPrice + tshirtPrice + league.hatPrice)
  league.totalExpenditure / memberCost

/-- Theorem stating the number of members in the Clearview Soccer League --/
theorem clearview_soccer_league_members :
  let league := SoccerLeague.mk 3 7 2 3516
  calculateMembers league = 117 := by
  sorry

end clearview_soccer_league_members_l2005_200594


namespace oil_leak_total_l2005_200520

theorem oil_leak_total (before_repairs : ℕ) (during_repairs : ℕ) 
  (h1 : before_repairs = 6522) 
  (h2 : during_repairs = 5165) : 
  before_repairs + during_repairs = 11687 :=
by sorry

end oil_leak_total_l2005_200520


namespace number_ratio_l2005_200554

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  y = 2 * x →
  z = k * y →
  (x + y + z) / 3 = 165 →
  y = 90 →
  z / y = 4 := by
sorry

end number_ratio_l2005_200554


namespace ryan_chinese_learning_hours_l2005_200517

/-- Given Ryan's daily Chinese learning hours and number of learning days, 
    calculate the total hours spent learning Chinese -/
def total_chinese_hours (daily_hours : ℕ) (days : ℕ) : ℕ :=
  daily_hours * days

/-- Theorem stating that Ryan spends 24 hours learning Chinese in 6 days -/
theorem ryan_chinese_learning_hours :
  total_chinese_hours 4 6 = 24 := by
  sorry

end ryan_chinese_learning_hours_l2005_200517


namespace diesel_rates_indeterminable_l2005_200541

/-- Represents the diesel purchase data for a company over 4 years -/
structure DieselPurchaseData where
  /-- The diesel rates for each of the 4 years (in dollars per gallon) -/
  rates : Fin 4 → ℝ
  /-- The amount spent on diesel each year (in dollars) -/
  annual_spend : ℝ
  /-- The mean cost of diesel over the 4-year period (in dollars per gallon) -/
  mean_cost : ℝ

/-- Theorem stating that given the conditions, the individual yearly rates cannot be uniquely determined -/
theorem diesel_rates_indeterminable (data : DieselPurchaseData) : 
  data.mean_cost = 1.52 → 
  (∀ (i j : Fin 4), i ≠ j → data.rates i ≠ data.rates j) →
  (∀ (i : Fin 4), data.annual_spend / data.rates i = data.annual_spend / data.rates 0) →
  ¬∃! (rates : Fin 4 → ℝ), rates = data.rates :=
sorry


end diesel_rates_indeterminable_l2005_200541


namespace zero_not_in_range_of_g_l2005_200598

-- Define the function g
noncomputable def g : ℝ → ℤ
| x => if x > -3 then Int.ceil (2 / (x + 3))
       else if x < -3 then Int.floor (2 / (x + 3))
       else 0  -- arbitrary value for x = -3, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 := by sorry

end zero_not_in_range_of_g_l2005_200598


namespace distribution_schemes_7_5_2_l2005_200537

/-- The number of ways to distribute n identical items among k recipients,
    where two recipients must receive at least m items each. -/
def distribution_schemes (n k m : ℕ) : ℕ :=
  sorry

/-- The specific case for 7 items, 5 recipients, and 2 items minimum for two recipients -/
theorem distribution_schemes_7_5_2 :
  distribution_schemes 7 5 2 = 35 :=
sorry

end distribution_schemes_7_5_2_l2005_200537


namespace adam_bought_more_cat_food_l2005_200502

/-- Represents the number of packages of cat food Adam bought -/
def cat_packages : ℕ := 9

/-- Represents the number of packages of dog food Adam bought -/
def dog_packages : ℕ := 7

/-- Represents the number of cans in each package of cat food -/
def cans_per_cat_package : ℕ := 10

/-- Represents the number of cans in each package of dog food -/
def cans_per_dog_package : ℕ := 5

/-- Calculates the difference between the total number of cans of cat food and dog food -/
def cans_difference : ℕ := 
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package

theorem adam_bought_more_cat_food : cans_difference = 55 := by
  sorry

end adam_bought_more_cat_food_l2005_200502


namespace min_jellybeans_jellybeans_solution_l2005_200542

theorem min_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n ≥ 164 :=
by
  sorry

theorem jellybeans_solution : ∃ (n : ℕ), n = 164 ∧ n ≥ 150 ∧ n % 15 = 14 :=
by
  sorry

end min_jellybeans_jellybeans_solution_l2005_200542


namespace christy_tanya_spending_ratio_l2005_200574

/-- Represents the spending of Christy and Tanya at Target -/
structure TargetShopping where
  christy_spent : ℕ
  tanya_face_moisturizer_price : ℕ
  tanya_face_moisturizer_quantity : ℕ
  tanya_body_lotion_price : ℕ
  tanya_body_lotion_quantity : ℕ
  total_spent : ℕ

/-- Calculates Tanya's total spending -/
def tanya_total_spent (shopping : TargetShopping) : ℕ :=
  shopping.tanya_face_moisturizer_price * shopping.tanya_face_moisturizer_quantity +
  shopping.tanya_body_lotion_price * shopping.tanya_body_lotion_quantity

/-- Theorem stating the ratio of Christy's spending to Tanya's spending -/
theorem christy_tanya_spending_ratio (shopping : TargetShopping)
  (h1 : shopping.tanya_face_moisturizer_price = 50)
  (h2 : shopping.tanya_face_moisturizer_quantity = 2)
  (h3 : shopping.tanya_body_lotion_price = 60)
  (h4 : shopping.tanya_body_lotion_quantity = 4)
  (h5 : shopping.total_spent = 1020)
  (h6 : shopping.christy_spent + tanya_total_spent shopping = shopping.total_spent) :
  2 * tanya_total_spent shopping = shopping.christy_spent := by
  sorry

#check christy_tanya_spending_ratio

end christy_tanya_spending_ratio_l2005_200574


namespace unique_half_rectangle_l2005_200544

/-- Given a rectangle R with dimensions a and b (a < b), prove that there exists exactly one rectangle
    with dimensions x and y such that x < b, y < b, its perimeter is half of R's, and its area is half of R's. -/
theorem unique_half_rectangle (a b : ℝ) (hab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ 2 * (x + y) = a + b ∧ x * y = a * b / 2 := by
  sorry

end unique_half_rectangle_l2005_200544


namespace gcd_3Sn_nplus1_le_1_l2005_200546

def square_sum (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_3Sn_nplus1_le_1 (n : ℕ+) :
  Nat.gcd (3 * square_sum n) (n + 1) ≤ 1 :=
sorry

end gcd_3Sn_nplus1_le_1_l2005_200546


namespace quadratic_equation_solution_l2005_200518

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : ∀ x : ℝ, x^2 + c*x + d = 0 ↔ x = 2*c ∨ x = 3*d) : 
  c = 1/6 ∧ d = -1/6 := by
  sorry

end quadratic_equation_solution_l2005_200518


namespace christinas_speed_l2005_200521

/-- Prove Christina's speed given the problem conditions -/
theorem christinas_speed (initial_distance : ℝ) (jacks_speed : ℝ) (lindys_speed : ℝ) 
  (lindys_total_distance : ℝ) (h1 : initial_distance = 360) 
  (h2 : jacks_speed = 5) (h3 : lindys_speed = 12) (h4 : lindys_total_distance = 360) :
  ∃ (christinas_speed : ℝ), christinas_speed = 7 := by
  sorry


end christinas_speed_l2005_200521


namespace smallest_five_digit_multiple_l2005_200590

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_multiple : 
  (∀ k : ℕ, is_five_digit k ∧ 
            is_divisible_by k 2 ∧ 
            is_divisible_by k 3 ∧ 
            is_divisible_by k 5 ∧ 
            is_divisible_by k 7 ∧ 
            is_divisible_by k 11 
            → k ≥ 11550) ∧ 
  is_five_digit 11550 ∧ 
  is_divisible_by 11550 2 ∧ 
  is_divisible_by 11550 3 ∧ 
  is_divisible_by 11550 5 ∧ 
  is_divisible_by 11550 7 ∧ 
  is_divisible_by 11550 11 :=
by sorry

end smallest_five_digit_multiple_l2005_200590


namespace cupcakes_baked_and_iced_l2005_200562

/-- Represents the number of cups of sugar in a bag -/
def sugar_per_bag : ℕ := 6

/-- Represents the number of bags of sugar bought -/
def bags_bought : ℕ := 2

/-- Represents the number of cups of sugar Lillian has at home -/
def sugar_at_home : ℕ := 3

/-- Represents the number of cups of sugar needed for batter per dozen cupcakes -/
def sugar_for_batter : ℕ := 1

/-- Represents the number of cups of sugar needed for frosting per dozen cupcakes -/
def sugar_for_frosting : ℕ := 2

/-- Theorem stating that Lillian can bake and ice 5 dozen cupcakes -/
theorem cupcakes_baked_and_iced : ℕ := by
  sorry

end cupcakes_baked_and_iced_l2005_200562


namespace distinct_sums_count_l2005_200524

def S : Finset ℕ := {2, 5, 8, 11, 14, 17, 20}

def fourDistinctSum (s : Finset ℕ) : Finset ℕ :=
  (s.powerset.filter (λ t => t.card = 4)).image (λ t => t.sum id)

theorem distinct_sums_count : (fourDistinctSum S).card = 12 := by
  sorry

end distinct_sums_count_l2005_200524


namespace complex_magnitude_equals_three_sqrt_ten_l2005_200550

theorem complex_magnitude_equals_three_sqrt_ten (x : ℝ) :
  x > 0 → Complex.abs (-3 + x * Complex.I) = 3 * Real.sqrt 10 → x = 9 := by
  sorry

end complex_magnitude_equals_three_sqrt_ten_l2005_200550


namespace exponential_range_condition_l2005_200587

theorem exponential_range_condition (a : ℝ) :
  (∀ x > 0, a^x > 1) ↔ a > 1 := by sorry

end exponential_range_condition_l2005_200587


namespace pascal_interior_sum_l2005_200552

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum :
  (sumInteriorNumbers 5 = 14) →
  (sumInteriorNumbers 6 = 30) →
  (sumInteriorNumbers 8 = 126) :=
by
  sorry

end pascal_interior_sum_l2005_200552


namespace complex_number_equality_l2005_200511

theorem complex_number_equality : ∀ z : ℂ, z = 1 - 2*I → z = -I := by
  sorry

end complex_number_equality_l2005_200511


namespace stamps_for_heavier_envelopes_l2005_200503

/-- Represents the number of stamps required for each weight category --/
def stamps_required (weight : ℕ) : ℕ :=
  if weight < 5 then 2
  else if weight ≤ 10 then 5
  else 7

/-- The total number of stamps purchased --/
def total_stamps : ℕ := 126

/-- The number of envelopes weighing less than 5 pounds --/
def light_envelopes : ℕ := 6

/-- Theorem stating that the total number of stamps used for envelopes weighing 
    5-10 lbs and >10 lbs is 114 --/
theorem stamps_for_heavier_envelopes :
  ∃ (medium heavy : ℕ),
    total_stamps = 
      light_envelopes * stamps_required 4 + 
      medium * stamps_required 5 + 
      heavy * stamps_required 11 ∧
    medium * stamps_required 5 + heavy * stamps_required 11 = 114 :=
sorry

end stamps_for_heavier_envelopes_l2005_200503


namespace factorization_equality_l2005_200527

theorem factorization_equality (x : ℝ) : -4 * x^2 + 16 = 4 * (2 + x) * (2 - x) := by
  sorry

end factorization_equality_l2005_200527


namespace quadratic_points_relationship_l2005_200553

theorem quadratic_points_relationship :
  let f (x : ℝ) := (x - 2)^2 - 1
  let y₁ := f 4
  let y₂ := f (Real.sqrt 2)
  let y₃ := f (-2)
  y₃ > y₁ ∧ y₁ > y₂ := by
sorry

end quadratic_points_relationship_l2005_200553


namespace centroid_eq_circumcenter_implies_equilateral_l2005_200530

/-- A triangle in a 2D Euclidean space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- 
If the centroid of a triangle coincides with its circumcenter, 
then the triangle is equilateral
-/
theorem centroid_eq_circumcenter_implies_equilateral (t : Triangle) :
  centroid t = circumcenter t → is_equilateral t := by sorry

end centroid_eq_circumcenter_implies_equilateral_l2005_200530


namespace set_intersection_equality_l2005_200577

def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

theorem set_intersection_equality : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by
  sorry

end set_intersection_equality_l2005_200577


namespace sector_central_angle_l2005_200543

theorem sector_central_angle (r : ℝ) (A : ℝ) (θ : ℝ) : 
  r = 2 → A = 4 → A = (1/2) * r^2 * θ → θ = 2 := by
  sorry

end sector_central_angle_l2005_200543


namespace charles_pictures_l2005_200557

theorem charles_pictures (total_papers : ℕ) (today_pictures : ℕ) (yesterday_before_work : ℕ) (papers_left : ℕ) 
  (h1 : total_papers = 20)
  (h2 : today_pictures = 6)
  (h3 : yesterday_before_work = 6)
  (h4 : papers_left = 2) :
  total_papers - (today_pictures + yesterday_before_work) - papers_left = 6 := by
  sorry

end charles_pictures_l2005_200557


namespace fraction_exponent_product_l2005_200579

theorem fraction_exponent_product : (5 / 6 : ℚ)^2 * (2 / 3 : ℚ)^3 = 50 / 243 := by sorry

end fraction_exponent_product_l2005_200579


namespace larger_number_is_eight_l2005_200515

theorem larger_number_is_eight (x y : ℕ) (h1 : x = 2 * y) (h2 : x * y = 40) (h3 : x + y = 14) : x = 8 := by
  sorry

#check larger_number_is_eight

end larger_number_is_eight_l2005_200515


namespace pebble_count_l2005_200586

theorem pebble_count (white_pebbles : ℕ) (red_pebbles : ℕ) : 
  white_pebbles = 20 → 
  red_pebbles = white_pebbles / 2 → 
  white_pebbles + red_pebbles = 30 := by
sorry

end pebble_count_l2005_200586


namespace largest_four_digit_divisible_by_7_with_different_digits_l2005_200555

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem largest_four_digit_divisible_by_7_with_different_digits :
  ∃ (n : ℕ), is_four_digit n ∧ n % 7 = 0 ∧ has_different_digits n ∧
  ∀ (m : ℕ), is_four_digit m → m % 7 = 0 → has_different_digits m → m ≤ n :=
sorry

end largest_four_digit_divisible_by_7_with_different_digits_l2005_200555


namespace f_derivative_at_pi_half_l2005_200525

noncomputable def f (x : ℝ) := Real.exp x * Real.cos x

theorem f_derivative_at_pi_half : 
  deriv f (Real.pi / 2) = -Real.exp (Real.pi / 2) := by sorry

end f_derivative_at_pi_half_l2005_200525


namespace gcd_factorial_seven_eight_l2005_200509

theorem gcd_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_factorial_seven_eight_l2005_200509


namespace different_group_choices_l2005_200563

theorem different_group_choices (n : Nat) (h : n = 3) : 
  n^2 - n = 6 := by
  sorry

#check different_group_choices

end different_group_choices_l2005_200563


namespace a_10_value_l2005_200597

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 12 = 16)
  (h_7 : a 7 = 1) :
  a 10 = 15 := by
  sorry

end a_10_value_l2005_200597


namespace total_amount_distributed_l2005_200551

/-- Represents the share distribution among w, x, y, and z -/
structure ShareDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- The given share distribution -/
def given_distribution : ShareDistribution :=
  { w := 2
    x := 1.5
    y := 2.5
    z := 1.7 }

/-- Y's share in rupees -/
def y_share : ℝ := 48.50

/-- Theorem stating the total amount distributed -/
theorem total_amount_distributed : 
  let d := given_distribution
  let unit_value := y_share / d.y
  let total_units := d.w + d.x + d.y + d.z
  total_units * unit_value = 188.08 := by
  sorry

#check total_amount_distributed

end total_amount_distributed_l2005_200551


namespace rachel_leah_age_difference_l2005_200505

/-- Given that Rachel is 19 years old and the sum of Rachel and Leah's ages is 34,
    prove that Rachel is 4 years older than Leah. -/
theorem rachel_leah_age_difference :
  ∀ (rachel_age leah_age : ℕ),
  rachel_age = 19 →
  rachel_age + leah_age = 34 →
  rachel_age - leah_age = 4 :=
by
  sorry

end rachel_leah_age_difference_l2005_200505


namespace gcd_count_for_product_360_l2005_200560

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃! (s : Finset ℕ), ∀ x, x ∈ s ↔ ∃ (a' b' : ℕ+), 
    Nat.gcd a' b' = x ∧ Nat.gcd a' b' * Nat.lcm a' b' = 360 ∧ s.card = 12) := by
  sorry

end gcd_count_for_product_360_l2005_200560
