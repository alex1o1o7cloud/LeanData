import Mathlib

namespace farm_pigs_count_l410_41045

/-- The number of pigs remaining in a barn after changes -/
def pigs_remaining (initial : ℕ) (joined : ℕ) (moved : ℕ) : ℕ :=
  initial + joined - moved

/-- Theorem stating that given the initial conditions, the number of pigs remaining is 431 -/
theorem farm_pigs_count : pigs_remaining 364 145 78 = 431 := by
  sorry

end farm_pigs_count_l410_41045


namespace train_length_l410_41070

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → 
  time_s = 2.49980001599872 → 
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 :=
by
  sorry

end train_length_l410_41070


namespace square_area_ratio_l410_41017

theorem square_area_ratio (s : ℝ) (h : s > 0) :
  let original_area := s^2
  let new_side := s^3 * Real.pi^(1/3)
  let new_area := new_side^2
  original_area / new_area = s^4 * Real.pi^(2/3) :=
by sorry

end square_area_ratio_l410_41017


namespace common_point_properties_l410_41089

open Real

noncomputable section

variables (a b : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := 1/2 * x^2 + 2*a*x

def g (a b : ℝ) (x : ℝ) : ℝ := 3*a^2 * log x + b

def common_point (a b : ℝ) : Prop :=
  ∃ x > 0, f a x = g a b x ∧ (deriv (f a)) x = (deriv (g a b)) x

theorem common_point_properties (h : a > 0) (h_common : common_point a b) :
  (a = 1 → b = 5/2) ∧
  (b = 5/2 * a^2 - 3*a^2 * log a) ∧
  (b ≤ 3/2 * exp (2/3)) :=
sorry

end

end common_point_properties_l410_41089


namespace correlation_coefficient_relationship_l410_41052

/-- Represents the starting age of smoking -/
def X : Type := ℕ

/-- Represents the relative risk of lung cancer for different starting ages -/
def Y : Type := ℝ

/-- Represents the number of cigarettes smoked per day -/
def U : Type := ℕ

/-- Represents the relative risk of lung cancer for different numbers of cigarettes -/
def V : Type := ℝ

/-- The linear correlation coefficient between X and Y -/
def r1 : ℝ := sorry

/-- The linear correlation coefficient between U and V -/
def r2 : ℝ := sorry

/-- Theorem stating the relationship between r1 and r2 -/
theorem correlation_coefficient_relationship : r1 < 0 ∧ 0 < r2 := by sorry

end correlation_coefficient_relationship_l410_41052


namespace remaining_payment_example_l410_41071

/-- Given a deposit percentage and deposit amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  let total_cost := deposit_amount / deposit_percentage
  total_cost - deposit_amount

/-- Theorem stating that the remaining payment is $1350 given a 10% deposit of $150 -/
theorem remaining_payment_example : remaining_payment (10 / 100) 150 = 1350 := by
  sorry

end remaining_payment_example_l410_41071


namespace simplify_expression_l410_41019

theorem simplify_expression (x : ℝ) :
  3*x^3 + 4*x + 5*x^2 + 2 - (7 - 3*x^3 - 4*x - 5*x^2) = 6*x^3 + 10*x^2 + 8*x - 5 :=
by sorry

end simplify_expression_l410_41019


namespace parallel_lines_m_value_l410_41063

/-- Two lines are parallel if their coefficients are proportional -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line l₁: (m+3)x + 4y + 3m - 5 = 0 -/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y + 3 * m - 5 = 0

/-- The second line l₂: 2x + (m+5)y - 8 = 0 -/
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y - 8 = 0

/-- Theorem: If l₁ and l₂ are parallel, then m = -7 -/
theorem parallel_lines_m_value :
  ∀ m : ℝ, parallel (m + 3) 4 2 (m + 5) → m = -7 := by
  sorry

end parallel_lines_m_value_l410_41063


namespace profit_percentage_calculation_l410_41029

theorem profit_percentage_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 800)
  (h2 : selling_price = 1080) :
  (selling_price - cost_price) / cost_price * 100 = 35 := by
  sorry

end profit_percentage_calculation_l410_41029


namespace chocolates_in_box_l410_41039

/-- Represents the dimensions of a cuboid -/
structure Dimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.width * d.length * d.height

/-- The dimensions of the box -/
def box_dimensions : Dimensions :=
  { width := 30, length := 20, height := 5 }

/-- The dimensions of a single chocolate -/
def chocolate_dimensions : Dimensions :=
  { width := 6, length := 4, height := 1 }

/-- Theorem stating that the number of chocolates in the box is 125 -/
theorem chocolates_in_box :
  (volume box_dimensions) / (volume chocolate_dimensions) = 125 := by
  sorry

end chocolates_in_box_l410_41039


namespace smallest_n_with_partial_divisibility_l410_41080

theorem smallest_n_with_partial_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - 2*n + 1) % k = 0) ∧ 
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - 2*n + 1) % k ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - 2*m + 1) % k = 0) ∨ 
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - 2*m + 1) % k ≠ 0)) ∧
  n = 3 := by
sorry

end smallest_n_with_partial_divisibility_l410_41080


namespace zacks_marbles_l410_41062

theorem zacks_marbles (friend1 friend2 friend3 friend4 friend5 friend6 remaining : ℕ) 
  (h1 : friend1 = 20)
  (h2 : friend2 = 30)
  (h3 : friend3 = 35)
  (h4 : friend4 = 25)
  (h5 : friend5 = 28)
  (h6 : friend6 = 40)
  (h7 : remaining = 7) :
  friend1 + friend2 + friend3 + friend4 + friend5 + friend6 + remaining = 185 := by
  sorry

end zacks_marbles_l410_41062


namespace M_intersect_N_equals_nonnegative_reals_l410_41067

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def N : Set ℝ := {y | ∃ x, y = x^2 - 2}

-- State the theorem
theorem M_intersect_N_equals_nonnegative_reals :
  M ∩ N = Set.Ici (0 : ℝ) := by sorry

end M_intersect_N_equals_nonnegative_reals_l410_41067


namespace min_triangles_in_square_l410_41072

/-- Represents a square with points and its triangulation -/
structure SquareWithPoints where
  k : ℕ
  points : Finset (ℝ × ℝ)
  triangles : Finset (Finset (ℝ × ℝ))

/-- Predicate to check if a triangulation is valid -/
def ValidTriangulation (s : SquareWithPoints) : Prop :=
  (s.k > 2) ∧
  (s.points.card = s.k) ∧
  (∀ t ∈ s.triangles, (t ∩ s.points).card ≤ 1)

/-- The minimum number of triangles needed -/
def MinTriangles (s : SquareWithPoints) : ℕ := s.k + 1

/-- Theorem stating the minimum number of triangles needed -/
theorem min_triangles_in_square (s : SquareWithPoints) 
  (h : ValidTriangulation s) : 
  s.triangles.card ≥ MinTriangles s :=
sorry

end min_triangles_in_square_l410_41072


namespace sum_of_opposite_sign_l410_41094

/-- Two real numbers are opposite in sign if their product is less than or equal to zero -/
def opposite_sign (a b : ℝ) : Prop := a * b ≤ 0

/-- If two real numbers are opposite in sign, then their sum is zero -/
theorem sum_of_opposite_sign (a b : ℝ) : opposite_sign a b → a + b = 0 := by
  sorry

end sum_of_opposite_sign_l410_41094


namespace mango_loss_percentage_l410_41038

/-- Calculates the percentage of loss for a fruit seller selling mangoes. -/
theorem mango_loss_percentage 
  (loss_price : ℝ) 
  (profit_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : loss_price = 8)
  (h2 : profit_price = 10.5)
  (h3 : profit_percentage = 5) : 
  (loss_price - profit_price / (1 + profit_percentage / 100)) / (profit_price / (1 + profit_percentage / 100)) * 100 = 20 := by
sorry

end mango_loss_percentage_l410_41038


namespace money_division_l410_41011

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 3600 →
  r - q = 4500 := by
sorry

end money_division_l410_41011


namespace red_notebook_cost_l410_41096

/-- Proves that the cost of each red notebook is 4 dollars --/
theorem red_notebook_cost (total_spent : ℕ) (total_notebooks : ℕ) (red_notebooks : ℕ) 
  (green_notebooks : ℕ) (green_cost : ℕ) (blue_cost : ℕ) :
  total_spent = 37 →
  total_notebooks = 12 →
  red_notebooks = 3 →
  green_notebooks = 2 →
  green_cost = 2 →
  blue_cost = 3 →
  (total_spent - (green_notebooks * green_cost + (total_notebooks - red_notebooks - green_notebooks) * blue_cost)) / red_notebooks = 4 := by
sorry

end red_notebook_cost_l410_41096


namespace garage_sale_pricing_l410_41005

theorem garage_sale_pricing (total_items : ℕ) (radio_highest_rank : ℕ) (h1 : total_items = 34) (h2 : radio_highest_rank = 14) :
  total_items - radio_highest_rank + 1 = 22 := by
  sorry

end garage_sale_pricing_l410_41005


namespace vending_machine_probability_l410_41043

/-- Represents the vending machine scenario --/
structure VendingMachine where
  numToys : Nat
  priceStep : Rat
  minPrice : Rat
  maxPrice : Rat
  numFavoriteToys : Nat
  favoriteToyPrice : Rat
  initialQuarters : Nat

/-- Calculates the probability of needing to exchange the $20 bill --/
def probabilityNeedExchange (vm : VendingMachine) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem vending_machine_probability (vm : VendingMachine) :
  vm.numToys = 10 ∧
  vm.priceStep = 1/2 ∧
  vm.minPrice = 1/2 ∧
  vm.maxPrice = 5 ∧
  vm.numFavoriteToys = 2 ∧
  vm.favoriteToyPrice = 9/2 ∧
  vm.initialQuarters = 12
  →
  probabilityNeedExchange vm = 15/25 :=
by sorry

end vending_machine_probability_l410_41043


namespace complex_distance_l410_41081

theorem complex_distance (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 4)
  (h3 : Complex.abs (z₁ + z₂) = 5) :
  Complex.abs (z₁ - z₂) = 5 := by
  sorry

end complex_distance_l410_41081


namespace min_perimeter_noncongruent_isosceles_triangles_l410_41058

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ  -- Length of equal sides
  base : ℕ  -- Length of the base
  is_isosceles : side > base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt ((t.side : ℝ)^2 - ((t.base : ℝ) / 2)^2) / 2

/-- Theorem: Minimum perimeter of two noncongruent integer-sided isosceles triangles -/
theorem min_perimeter_noncongruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    9 * t2.base = 8 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      9 * s2.base = 8 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 842 :=
by sorry

end min_perimeter_noncongruent_isosceles_triangles_l410_41058


namespace pen_purchase_shortfall_l410_41073

/-- The amount of money needed to purchase a pen given the cost, initial amount, and borrowed amount -/
theorem pen_purchase_shortfall (pen_cost : ℕ) (initial_amount : ℕ) (borrowed_amount : ℕ) :
  pen_cost = 600 →
  initial_amount = 500 →
  borrowed_amount = 68 →
  pen_cost - (initial_amount + borrowed_amount) = 32 := by
  sorry

end pen_purchase_shortfall_l410_41073


namespace unique_solution_factorial_equation_l410_41040

theorem unique_solution_factorial_equation :
  ∃! (n : ℕ), n * n.factorial + 2 * n.factorial = 5040 :=
by
  -- Proof goes here
  sorry

end unique_solution_factorial_equation_l410_41040


namespace yeast_growth_proof_l410_41075

/-- Calculates the yeast population after a given time -/
def yeast_population (initial_population : ℕ) (growth_factor : ℕ) (interval_duration : ℕ) (total_time : ℕ) : ℕ :=
  initial_population * growth_factor ^ (total_time / interval_duration)

/-- Proves that the yeast population grows to 1350 after 18 minutes -/
theorem yeast_growth_proof :
  yeast_population 50 3 5 18 = 1350 := by
  sorry

end yeast_growth_proof_l410_41075


namespace eight_person_arrangement_l410_41085

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangements (n : ℕ) (a b c : ℕ) : ℕ :=
  factorial n - (factorial (n-1) * 2) - (factorial (n-2) * 6 - factorial (n-1) * 2)

theorem eight_person_arrangement : arrangements 8 1 1 1 = 36000 := by
  sorry

end eight_person_arrangement_l410_41085


namespace quadratic_equations_solutions_l410_41020

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 2*x = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) ∧
  (∃ x : ℝ, x*(x-3) = 7*(3-x) ↔ x = 3 ∨ x = -7) :=
by sorry

end quadratic_equations_solutions_l410_41020


namespace polynomial_expansion_l410_41069

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 - 2 * x + 4) * (4 * x^2 - 3 * x + 5) =
  12 * x^5 - 9 * x^4 + 7 * x^3 + 10 * x^2 - 2 * x + 20 := by
  sorry

end polynomial_expansion_l410_41069


namespace unique_sum_with_identical_digits_l410_41065

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number is a three-digit number with identical digits -/
def is_three_identical_digits (m : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ Finset.range 10 ∧ m = 111 * d

theorem unique_sum_with_identical_digits :
  ∃! (n : ℕ), is_three_identical_digits (sum_of_first_n n) :=
sorry

end unique_sum_with_identical_digits_l410_41065


namespace complement_intersection_theorem_l410_41083

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end complement_intersection_theorem_l410_41083


namespace water_consumption_theorem_l410_41095

/-- The amount of water drunk by the traveler and his camel in gallons -/
def total_water_gallons (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℚ :=
  (traveler_ounces + traveler_ounces * camel_multiplier) / ounces_per_gallon

/-- Theorem stating that the total water drunk is 2 gallons -/
theorem water_consumption_theorem :
  total_water_gallons 32 7 128 = 2 := by
  sorry

end water_consumption_theorem_l410_41095


namespace quadrilateral_on_exponential_curve_l410_41064

theorem quadrilateral_on_exponential_curve (e : ℝ) (h_e : e > 0) :
  ∃ m : ℕ+, 
    (1/2 * (e^(m : ℝ) - e^((m : ℝ) + 3)) = (e^2 - 1) / e) ∧ 
    (∀ k : ℕ+, k < m → 1/2 * (e^(k : ℝ) - e^((k : ℝ) + 3)) ≠ (e^2 - 1) / e) := by
  sorry

end quadrilateral_on_exponential_curve_l410_41064


namespace expression_evaluation_l410_41034

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := -1
  5 * x^2 - 2 * (3 * y^2 + 6 * x) + (2 * y^2 - 5 * x^2) = 8 := by
sorry

end expression_evaluation_l410_41034


namespace ship_always_illuminated_l410_41025

/-- A lighthouse with a rotating beam -/
structure Lighthouse where
  /-- The position of the lighthouse -/
  position : ℝ × ℝ
  /-- The distance the beam reaches -/
  beam_distance : ℝ
  /-- The velocity of the beam's extremity -/
  beam_velocity : ℝ

/-- A ship moving towards the lighthouse -/
structure Ship where
  /-- The initial position of the ship -/
  initial_position : ℝ × ℝ
  /-- The maximum speed of the ship -/
  max_speed : ℝ

/-- The theorem stating that a ship moving at most v/8 cannot reach the lighthouse without being illuminated -/
theorem ship_always_illuminated (L : Lighthouse) (S : Ship) 
    (h1 : S.max_speed ≤ L.beam_velocity / 8)
    (h2 : dist S.initial_position L.position ≤ L.beam_distance) :
    ∃ (t : ℝ), t ∈ Set.Icc 0 (Real.pi * L.beam_distance / L.beam_velocity) ∧ 
    dist (S.initial_position + t • (L.position - S.initial_position)) L.position ≤ L.beam_distance :=
  sorry


end ship_always_illuminated_l410_41025


namespace no_perfect_square_in_sequence_l410_41082

theorem no_perfect_square_in_sequence : ¬∃ (k n : ℕ), 3 * k - 1 = n ^ 2 := by
  sorry

end no_perfect_square_in_sequence_l410_41082


namespace quadratic_specific_value_l410_41041

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_specific_value (a b c : ℝ) :
  (∃ (f : ℝ → ℝ), f = quadratic a b c) →
  (∀ x, quadratic a b c x ≥ -4) →
  (quadratic a b c (-5) = -4) →
  (quadratic a b c 0 = 6) →
  (quadratic a b c (-3) = -2.4) :=
sorry

end quadratic_specific_value_l410_41041


namespace tangent_line_cubic_l410_41004

/-- The equation of the tangent line to y = x^3 - 1 at x = 1 is y = 3x - 3 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3 - 1) → 
  (∃ m b : ℝ, ∀ x' y' : ℝ, y' = m * x' + b ∧ 
    (y' = (x')^3 - 1 → x' = 1 → y' = m * x' + b) ∧
    (1 = 1^3 - 1 → 1 = m * 1 + b) ∧
    m = 3 ∧ b = -3) :=
by sorry

end tangent_line_cubic_l410_41004


namespace f_value_at_neg_five_halves_l410_41079

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_neg_five_halves :
  (∀ x, f x = f (-x)) →                     -- f is even
  (∀ x, f (x + 2) = f x) →                  -- f has period 2
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2*x*(1 - x)) → -- f definition for 0 ≤ x ≤ 1
  f (-5/2) = 1/2 := by sorry

end f_value_at_neg_five_halves_l410_41079


namespace quadratic_strictly_increasing_iff_l410_41003

/-- A function f: ℝ → ℝ is strictly increasing if for all x < y, f(x) < f(y) -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The quadratic function f(x) = ax^2 + 2x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 3

theorem quadratic_strictly_increasing_iff (a : ℝ) :
  StrictlyIncreasing (f a) ↔ -1/4 ≤ a ∧ a ≤ 0 :=
by sorry

end quadratic_strictly_increasing_iff_l410_41003


namespace total_people_in_line_l410_41056

/-- Given a line of people at an amusement park ride, this theorem proves
    the total number of people in line based on Eunji's position. -/
theorem total_people_in_line (eunji_position : ℕ) (people_behind_eunji : ℕ) :
  eunji_position = 6 →
  people_behind_eunji = 7 →
  eunji_position + people_behind_eunji = 13 := by
  sorry

#check total_people_in_line

end total_people_in_line_l410_41056


namespace fraction_modification_l410_41092

theorem fraction_modification (d : ℚ) : 
  (3 : ℚ) / d ≠ (1 : ℚ) / (3 : ℚ) →
  ((3 : ℚ) + 3) / (d + 3) = (1 : ℚ) / (3 : ℚ) →
  d = 15 := by
sorry

end fraction_modification_l410_41092


namespace number_ordering_l410_41086

def A : ℕ := 9^(9^9)
def B : ℕ := 99^9
def C : ℕ := (9^9)^9
def D : ℕ := (Nat.factorial 9)^(Nat.factorial 9)

theorem number_ordering : B < C ∧ C < A ∧ A < D := by sorry

end number_ordering_l410_41086


namespace gis_may_lead_to_overfishing_l410_41050

/-- Represents the use of GIS technology in fishery production -/
structure GISTechnology where
  locateSchools : Bool
  widelyIntroduced : Bool

/-- Represents the state of fishery resources -/
structure FisheryResources where
  overfishing : Bool
  exhausted : Bool

/-- The impact of GIS technology on fishery resources -/
def gisImpact (tech : GISTechnology) : FisheryResources :=
  { overfishing := tech.locateSchools ∧ tech.widelyIntroduced,
    exhausted := tech.locateSchools ∧ tech.widelyIntroduced }

theorem gis_may_lead_to_overfishing (tech : GISTechnology) 
  (h1 : tech.locateSchools = true) 
  (h2 : tech.widelyIntroduced = true) : 
  (gisImpact tech).overfishing = true ∧ (gisImpact tech).exhausted = true :=
by sorry

end gis_may_lead_to_overfishing_l410_41050


namespace g_of_x_plus_3_l410_41000

/-- Given a function g(x) = (x^2 + 3x) / 2, prove that g(x+3) = (x^2 + 9x + 18) / 2 for all real x -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ (x^2 + 3*x) / 2
  g (x + 3) = (x^2 + 9*x + 18) / 2 := by
sorry

end g_of_x_plus_3_l410_41000


namespace parallelogram_perimeter_l410_41013

theorem parallelogram_perimeter (n : ℕ) (h : n = 92) :
  ∃ (a b : ℕ), a * b = n ∧ (2 * a + 2 * b = 94 ∨ 2 * a + 2 * b = 50) :=
by
  sorry

end parallelogram_perimeter_l410_41013


namespace max_red_socks_is_990_l410_41018

/-- Represents the number of socks in a drawer -/
structure SockDrawer where
  red : ℕ
  blue : ℕ
  total_le_2000 : red + blue ≤ 2000
  blue_lt_red : blue < red
  total_odd : ¬ Even (red + blue)
  prob_same_color : (red * (red - 1) + blue * (blue - 1)) = (red + blue) * (red + blue - 1) / 2

/-- The maximum number of red socks possible in the drawer -/
def max_red_socks : ℕ := 990

/-- Theorem stating that the maximum number of red socks is 990 -/
theorem max_red_socks_is_990 (drawer : SockDrawer) : drawer.red ≤ max_red_socks := by
  sorry

end max_red_socks_is_990_l410_41018


namespace min_sum_reciprocals_l410_41036

theorem min_sum_reciprocals (n : ℕ+) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : 
  (1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) ≥ 1 ∧ 
  ((1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_sum_reciprocals_l410_41036


namespace minimizing_n_is_six_l410_41010

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  sum : ℕ → ℤ  -- The sum function
  first_fifth_sum : a 1 + a 5 = -14
  ninth_sum : sum 9 = -27

/-- The value of n that minimizes the sum of the first n terms -/
def minimizing_n (seq : ArithmeticSequence) : ℕ :=
  6

/-- Theorem stating that 6 is the value of n that minimizes S_n -/
theorem minimizing_n_is_six (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.sum n ≥ seq.sum (minimizing_n seq) :=
sorry

end minimizing_n_is_six_l410_41010


namespace parallel_lines_k_value_l410_41049

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The first line has equation y = 3x + 5 -/
def line1 : ℝ → ℝ := λ x => 3 * x + 5

/-- The second line has equation y = (6k)x + 1 -/
def line2 (k : ℝ) : ℝ → ℝ := λ x => 6 * k * x + 1

theorem parallel_lines_k_value :
  ∀ k : ℝ, parallel (line1 0 - line1 1) (line2 k 0 - line2 k 1) → k = 1/2 := by
sorry

end parallel_lines_k_value_l410_41049


namespace andy_inappropriate_joke_demerits_l410_41093

/-- Represents the number of demerits Andy got for making an inappropriate joke -/
def inappropriate_joke_demerits : ℕ := sorry

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of demerits Andy gets per instance of being late -/
def late_demerits_per_instance : ℕ := 2

/-- The number of times Andy was late -/
def late_instances : ℕ := 6

/-- The number of additional demerits Andy can get this month before getting fired -/
def remaining_demerits : ℕ := 23

theorem andy_inappropriate_joke_demerits :
  inappropriate_joke_demerits = 
    max_demerits - remaining_demerits - (late_demerits_per_instance * late_instances) :=
by sorry

end andy_inappropriate_joke_demerits_l410_41093


namespace circle_inequality_l410_41044

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) :=
by sorry

end circle_inequality_l410_41044


namespace complex_power_patterns_l410_41084

theorem complex_power_patterns (i : ℂ) (h : i^2 = -1) :
  ∀ n : ℕ,
    i^(4*n + 1) = i ∧
    i^(4*n + 2) = -1 ∧
    i^(4*n + 3) = -i :=
by sorry

end complex_power_patterns_l410_41084


namespace slope_of_CD_is_one_l410_41047

/-- Given a line y = kx (k > 0) passing through the origin and intersecting the curve y = e^(x-1)
    at two distinct points A(x₁, y₁) and B(x₂, y₂) where x₁ > 0 and x₂ > 0, and points C(x₁, ln x₁)
    and D(x₂, ln x₂) on the curve y = ln x, prove that the slope of line CD is 1. -/
theorem slope_of_CD_is_one (k x₁ x₂ : ℝ) (hk : k > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hy₁ : k * x₁ = Real.exp (x₁ - 1)) (hy₂ : k * x₂ = Real.exp (x₂ - 1)) :
  (Real.log x₂ - Real.log x₁) / (x₂ - x₁) = 1 :=
by sorry

end slope_of_CD_is_one_l410_41047


namespace max_b_no_lattice_points_b_max_is_maximum_l410_41002

/-- Represents a lattice point with integer coordinates -/
structure LatticePoint where
  x : Int
  y : Int

/-- Checks if a given point lies on the line y = mx + 3 -/
def lies_on_line (m : ℚ) (p : LatticePoint) : Prop :=
  p.y = m * p.x + 3

/-- The maximum value of b we want to prove -/
def b_max : ℚ := 76 / 151

theorem max_b_no_lattice_points :
  ∀ m : ℚ, 1/2 < m → m < b_max →
    ∀ x : ℤ, 0 < x → x ≤ 150 →
      ¬∃ p : LatticePoint, p.x = x ∧ lies_on_line m p :=
sorry

theorem b_max_is_maximum :
  ∀ b : ℚ, b > b_max →
    ∃ m : ℚ, 1/2 < m ∧ m < b ∧
      ∃ x : ℤ, 0 < x ∧ x ≤ 150 ∧
        ∃ p : LatticePoint, p.x = x ∧ lies_on_line m p :=
sorry

end max_b_no_lattice_points_b_max_is_maximum_l410_41002


namespace tom_ate_three_fruits_l410_41037

/-- The number of fruits Tom ate -/
def fruits_eaten (initial_oranges initial_lemons remaining_fruits : ℕ) : ℕ :=
  initial_oranges + initial_lemons - remaining_fruits

/-- Proof that Tom ate 3 fruits -/
theorem tom_ate_three_fruits :
  fruits_eaten 3 6 6 = 3 := by
  sorry

end tom_ate_three_fruits_l410_41037


namespace presidency_meeting_combinations_l410_41059

/-- The number of schools participating in the conference -/
def num_schools : ℕ := 4

/-- The number of members in each school -/
def members_per_school : ℕ := 5

/-- The number of representatives sent by the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of ways to choose representatives for the presidency meeting -/
def total_ways : ℕ := num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school ^ (num_schools - 1))

theorem presidency_meeting_combinations : total_ways = 5000 := by
  sorry

end presidency_meeting_combinations_l410_41059


namespace inscribed_triangle_circumscribed_square_l410_41027

theorem inscribed_triangle_circumscribed_square (r : ℝ) : 
  r > 0 → 
  let triangle_side := r * Real.sqrt 3
  let triangle_perimeter := 3 * triangle_side
  let square_side := r * Real.sqrt 2
  let square_area := square_side ^ 2
  triangle_perimeter = square_area →
  r = 3 * Real.sqrt 3 / 4 := by
sorry

end inscribed_triangle_circumscribed_square_l410_41027


namespace right_triangle_hypotenuse_l410_41014

theorem right_triangle_hypotenuse (m1 m2 : ℝ) (h_m1 : m1 = 6) (h_m2 : m2 = Real.sqrt 50) :
  ∃ a b h : ℝ,
    a > 0 ∧ b > 0 ∧
    m1^2 = a^2 + (b/2)^2 ∧
    m2^2 = b^2 + (a/2)^2 ∧
    h^2 = (2*a)^2 + (2*b)^2 ∧
    h = Real.sqrt 275.2 :=
by sorry

end right_triangle_hypotenuse_l410_41014


namespace binomial_probability_l410_41077

/-- A binomially distributed random variable with given mean and variance -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean_eq : n * p = 5 / 3
  var_eq : n * p * (1 - p) = 10 / 9

/-- The probability mass function for a binomial distribution -/
def binomialPMF (rv : BinomialRV) (k : ℕ) : ℝ :=
  (Nat.choose rv.n k) * (rv.p ^ k) * ((1 - rv.p) ^ (rv.n - k))

theorem binomial_probability (rv : BinomialRV) : 
  binomialPMF rv 4 = 10 / 243 := by
  sorry

end binomial_probability_l410_41077


namespace married_men_fraction_l410_41024

structure Gathering where
  total_women : ℕ
  single_women : ℕ
  married_women : ℕ
  married_men : ℕ

def Gathering.total_people (g : Gathering) : ℕ :=
  g.total_women + g.married_men

def Gathering.prob_single_woman (g : Gathering) : ℚ :=
  g.single_women / g.total_women

def Gathering.fraction_married_men (g : Gathering) : ℚ :=
  g.married_men / g.total_people

theorem married_men_fraction (g : Gathering) 
  (h1 : g.married_women = g.married_men)
  (h2 : g.total_women = g.single_women + g.married_women)
  (h3 : g.prob_single_woman = 1/4) :
  g.fraction_married_men = 3/7 := by
sorry

end married_men_fraction_l410_41024


namespace one_and_one_third_problem_l410_41022

theorem one_and_one_third_problem : ∃ x : ℚ, (4/3) * x = 36 ∧ x = 27 := by
  sorry

end one_and_one_third_problem_l410_41022


namespace fraction_equals_373_l410_41007

-- Define the factorization of x^4 + 324
def factor (x : ℤ) : ℤ × ℤ :=
  ((x * (x - 6) + 18), (x * (x + 6) + 18))

-- Define the numerator and denominator sequences
def num_seq : List ℤ := [10, 22, 34, 46, 58]
def den_seq : List ℤ := [4, 16, 28, 40, 52]

-- Define the fraction
def fraction : ℚ :=
  (num_seq.map (λ x => (factor x).1 * (factor x).2)).prod /
  (den_seq.map (λ x => (factor x).1 * (factor x).2)).prod

-- Theorem statement
theorem fraction_equals_373 : fraction = 373 := by
  sorry

end fraction_equals_373_l410_41007


namespace ratio_calculation_l410_41098

theorem ratio_calculation : 
  let numerator := (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484)
  let denominator := (8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)
  numerator / denominator = -423 := by
sorry

end ratio_calculation_l410_41098


namespace constant_sequence_is_ap_and_gp_l410_41048

def constant_sequence : ℕ → ℝ := λ n => 7

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem constant_sequence_is_ap_and_gp :
  is_arithmetic_progression constant_sequence ∧
  is_geometric_progression constant_sequence := by
  sorry

#check constant_sequence_is_ap_and_gp

end constant_sequence_is_ap_and_gp_l410_41048


namespace inequality_proof_l410_41033

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end inequality_proof_l410_41033


namespace evaluate_expression_l410_41051

theorem evaluate_expression : (-3)^7 / 3^5 + 2^6 - 4^2 = 39 := by
  sorry

end evaluate_expression_l410_41051


namespace u_diff_divisible_by_factorial_l410_41009

/-- The sequence u_k defined recursively -/
def u (a : ℕ+) : ℕ → ℕ
  | 0 => 1
  | k + 1 => a ^ (u a k)

/-- Theorem stating that n! divides u_{n+1} - u_n for all n ≥ 1 -/
theorem u_diff_divisible_by_factorial (a : ℕ+) (n : ℕ) (h : n ≥ 1) :
  (n.factorial : ℤ) ∣ (u a (n + 1) : ℤ) - (u a n : ℤ) := by
  sorry

end u_diff_divisible_by_factorial_l410_41009


namespace quadratic_equation_solution_l410_41074

theorem quadratic_equation_solution (a b : ℕ+) : 
  (∃ x : ℝ, x > 0 ∧ x^2 + 12*x = 73 ∧ x = Real.sqrt a - b) → a + b = 115 := by
  sorry

end quadratic_equation_solution_l410_41074


namespace product_equation_solution_l410_41055

theorem product_equation_solution : ∃! (B : ℕ), 
  B < 10 ∧ (10 * B + 2) * (90 + B) = 8016 := by sorry

end product_equation_solution_l410_41055


namespace market_value_calculation_l410_41088

/-- Calculates the market value of a share given its nominal value, dividend rate, and desired interest rate. -/
def marketValue (nominalValue : ℚ) (dividendRate : ℚ) (desiredInterestRate : ℚ) : ℚ :=
  (nominalValue * dividendRate) / desiredInterestRate

/-- Theorem stating that for a share with nominal value of 48, 9% dividend rate, and 12% desired interest rate, the market value is 36. -/
theorem market_value_calculation :
  marketValue 48 (9/100) (12/100) = 36 := by
  sorry

end market_value_calculation_l410_41088


namespace expression_value_l410_41031

theorem expression_value : 
  (10^2005 + 10^2007) / (10^2006 + 10^2006) = 101 / 20 := by
  sorry

end expression_value_l410_41031


namespace perpendicular_line_x_intercept_l410_41008

/-- Given a line L1: 4x - 3y = 12, prove that a line L2 perpendicular to L1 
    with y-intercept 3 has x-intercept 4 -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x - 3 * y = 12
  let m1 : ℝ := 4 / 3  -- slope of L1
  let m2 : ℝ := -3 / 4  -- slope of L2 (perpendicular to L1)
  let L2 : ℝ → ℝ → Prop := λ x y => y = m2 * x + 3  -- L2 with y-intercept 3
  ∃ x : ℝ, x = 4 ∧ L2 x 0 :=
by
  sorry

#check perpendicular_line_x_intercept

end perpendicular_line_x_intercept_l410_41008


namespace product_expansion_l410_41035

theorem product_expansion (x : ℝ) : 
  (3*x - 4) * (2*x^2 + 3*x - 1) = 6*x^3 + x^2 - 15*x + 4 := by
  sorry

end product_expansion_l410_41035


namespace min_distance_to_origin_l410_41046

theorem min_distance_to_origin (x y : ℝ) (h : 5 * x + 12 * y - 60 = 0) :
  ∃ (min : ℝ), min = 60 / 13 ∧ ∀ (a b : ℝ), 5 * a + 12 * b - 60 = 0 → min ≤ Real.sqrt (a^2 + b^2) := by
  sorry

end min_distance_to_origin_l410_41046


namespace sum_of_fifth_powers_l410_41066

theorem sum_of_fifth_powers (x y z : ℝ) 
  (eq1 : x + y + z = 3)
  (eq2 : x^3 + y^3 + z^3 = 15)
  (eq3 : x^4 + y^4 + z^4 = 35)
  (ineq : x^2 + y^2 + z^2 < 10) :
  x^5 + y^5 + z^5 = 83 := by
sorry

end sum_of_fifth_powers_l410_41066


namespace add_fractions_with_same_denominator_l410_41032

theorem add_fractions_with_same_denominator (a : ℝ) (h : a ≠ 0) :
  3 / a + 2 / a = 5 / a := by sorry

end add_fractions_with_same_denominator_l410_41032


namespace coffee_cream_ratio_l410_41099

/-- The ratio of cream in Joe's coffee to JoAnn's coffee -/
theorem coffee_cream_ratio :
  let initial_coffee : ℝ := 20
  let joe_drank : ℝ := 3
  let cream_added : ℝ := 4
  let joann_drank : ℝ := 3
  let joe_cream : ℝ := cream_added
  let joann_total : ℝ := initial_coffee + cream_added
  let joann_cream_ratio : ℝ := cream_added / joann_total
  let joann_cream : ℝ := cream_added - (joann_drank * joann_cream_ratio)
  (joe_cream / joann_cream) = 8 / 7 := by
sorry

end coffee_cream_ratio_l410_41099


namespace percent_k_equal_to_125_percent_j_l410_41001

theorem percent_k_equal_to_125_percent_j (j k l m : ℝ) 
  (h1 : 1.25 * j = (12.5 / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 3.5 * (2 * j)) :
  (12.5 / 100) * k = 1.25 * j := by
  sorry

end percent_k_equal_to_125_percent_j_l410_41001


namespace equation_value_l410_41026

theorem equation_value (x y : ℝ) (h : 2*x - y = -1) : 3 + 4*x - 2*y = 1 := by
  sorry

end equation_value_l410_41026


namespace sphere_radius_is_4_l410_41087

/-- Represents a cylindrical container with spheres -/
structure Container where
  initialHeight : ℝ
  sphereRadius : ℝ
  numSpheres : ℕ

/-- Calculates the final height of water in the container after adding spheres -/
def finalHeight (c : Container) : ℝ :=
  c.initialHeight + c.sphereRadius * 2

/-- The problem statement -/
theorem sphere_radius_is_4 (c : Container) :
  c.initialHeight = 8 ∧
  c.numSpheres = 3 ∧
  finalHeight c = c.initialHeight + c.sphereRadius * 2 →
  c.sphereRadius = 4 := by
  sorry


end sphere_radius_is_4_l410_41087


namespace remaining_problems_to_grade_l410_41016

-- Define the given conditions
def total_worksheets : ℕ := 17
def graded_worksheets : ℕ := 8
def problems_per_worksheet : ℕ := 7

-- State the theorem
theorem remaining_problems_to_grade :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end remaining_problems_to_grade_l410_41016


namespace smallest_batch_size_l410_41042

theorem smallest_batch_size (N : ℕ) (h1 : N > 70) (h2 : (21 * N) % 70 = 0) :
  N ≥ 80 ∧ ∀ m : ℕ, m > 70 ∧ (21 * m) % 70 = 0 → m ≥ N := by
  sorry

end smallest_batch_size_l410_41042


namespace e_recursive_relation_l410_41090

def e (n : ℕ) : ℕ := n^5

theorem e_recursive_relation (n : ℕ) :
  e (n + 6) = 6 * e (n + 5) - 15 * e (n + 4) + 20 * e (n + 3) - 15 * e (n + 2) + 6 * e (n + 1) - e n :=
by sorry

end e_recursive_relation_l410_41090


namespace number_ratio_l410_41030

theorem number_ratio (f s t : ℝ) : 
  s = 4 * f →
  (f + s + t) / 3 = 77 →
  f = 33 →
  f ≤ s ∧ f ≤ t →
  t / f = 2 := by
sorry

end number_ratio_l410_41030


namespace product_third_fourth_term_l410_41028

/-- An arithmetic sequence with common difference 2 and eighth term 20 -/
def ArithmeticSequence (a : ℕ) : ℕ → ℕ :=
  fun n => a + (n - 1) * 2

theorem product_third_fourth_term (a : ℕ) :
  ArithmeticSequence a 8 = 20 →
  ArithmeticSequence a 3 * ArithmeticSequence a 4 = 120 := by
  sorry

end product_third_fourth_term_l410_41028


namespace necessary_not_sufficient_condition_l410_41061

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-1/2 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define the set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | |x - m| ≥ 1}

-- Define the range of m
def m_range : Set ℝ := Set.Iic (-9/16) ∪ Set.Ici 3

-- Theorem statement
theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ t, t ∈ A → t ∈ B m) ∧ (∃ t, t ∈ B m ∧ t ∉ A) ↔ m ∈ m_range :=
sorry

end necessary_not_sufficient_condition_l410_41061


namespace max_value_expression_l410_41012

theorem max_value_expression (x y : ℝ) : 
  ∃ (M : ℝ), M = 24 - 2 * Real.sqrt 7 ∧ 
  ∀ (a b : ℝ), a ≤ M ∧ 
  (∃ (x y : ℝ), a = (Real.sqrt (9 - Real.sqrt 7) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) * 
                   (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y))) :=
by sorry

end max_value_expression_l410_41012


namespace planted_area_fraction_l410_41021

/-- A right triangle with legs of length 3 and 4 units -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right_triangle : leg1 = 3 ∧ leg2 = 4

/-- A square placed in the right angle corner of the triangle -/
structure CornerSquare (t : RightTriangle) where
  side_length : ℝ
  in_corner : side_length > 0
  distance_to_hypotenuse : ℝ
  is_correct_distance : distance_to_hypotenuse = 2

theorem planted_area_fraction (t : RightTriangle) (s : CornerSquare t) :
  (t.leg1 * t.leg2 / 2 - s.side_length ^ 2) / (t.leg1 * t.leg2 / 2) = 145 / 147 := by
  sorry

end planted_area_fraction_l410_41021


namespace lunch_special_cost_l410_41091

theorem lunch_special_cost (total_bill : ℚ) (num_people : ℕ) (h1 : total_bill = 24) (h2 : num_people = 3) :
  total_bill / num_people = 8 := by
  sorry

end lunch_special_cost_l410_41091


namespace oil_change_price_is_20_l410_41078

def oil_change_price (repair_price car_wash_price : ℕ) 
                     (oil_changes repairs car_washes : ℕ) 
                     (total_earnings : ℕ) : Prop :=
  ∃ (x : ℕ), 
    repair_price = 30 ∧
    car_wash_price = 5 ∧
    oil_changes = 5 ∧
    repairs = 10 ∧
    car_washes = 15 ∧
    total_earnings = 475 ∧
    x * oil_changes + repair_price * repairs + car_wash_price * car_washes = total_earnings ∧
    x = 20

theorem oil_change_price_is_20 : 
  ∀ (repair_price car_wash_price oil_changes repairs car_washes total_earnings : ℕ),
    oil_change_price repair_price car_wash_price oil_changes repairs car_washes total_earnings :=
by
  sorry

end oil_change_price_is_20_l410_41078


namespace parabola_tangent_to_line_l410_41076

-- Define the parabola and line
def parabola (b x : ℝ) : ℝ := b * x^2 + 4
def line (x : ℝ) : ℝ := 2 * x + 2

-- Define the tangency condition
def is_tangent (b : ℝ) : Prop :=
  ∃! x, parabola b x = line x

-- Theorem statement
theorem parabola_tangent_to_line :
  ∀ b : ℝ, is_tangent b → b = 1/2 := by
  sorry

end parabola_tangent_to_line_l410_41076


namespace sector_area_l410_41015

theorem sector_area (r : ℝ) (θ : ℝ) (chord_length : ℝ) : 
  θ = 2 ∧ 
  chord_length = 2 * Real.sin 1 ∧ 
  chord_length = 2 * r * Real.sin (θ / 2) →
  (1 / 2) * r^2 * θ = 1 := by
sorry

end sector_area_l410_41015


namespace power_function_value_l410_41068

-- Define the power function type
def PowerFunction := ℝ → ℝ

-- Define the property of passing through the point (3, √3/3)
def PassesThroughPoint (f : PowerFunction) : Prop :=
  f 3 = Real.sqrt 3 / 3

-- State the theorem
theorem power_function_value (f : PowerFunction) 
  (h : PassesThroughPoint f) : f (1/4) = 2 := by
  sorry

end power_function_value_l410_41068


namespace smallest_value_of_floor_sum_l410_41053

theorem smallest_value_of_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ⌊(a + b) / c⌋ + ⌊(b + c) / a⌋ + ⌊(c + a) / b⌋ ≥ 4 :=
by sorry

end smallest_value_of_floor_sum_l410_41053


namespace yvonne_success_probability_l410_41023

theorem yvonne_success_probability 
  (p_xavier : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_yvonne_not_zelda : ℝ) 
  (h1 : p_xavier = 1/5)
  (h2 : p_zelda = 5/8)
  (h3 : p_xavier_yvonne_not_zelda = 0.0375) :
  ∃ p_yvonne : ℝ, 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_yvonne_not_zelda ∧ 
    p_yvonne = 1/2 :=
by sorry

end yvonne_success_probability_l410_41023


namespace max_red_socks_l410_41006

def is_valid_sock_distribution (r b g : ℕ) : Prop :=
  let t := r + b + g
  t ≤ 2500 ∧
  (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 * t * (t - 1)) / 3

theorem max_red_socks :
  ∃ (r b g : ℕ),
    is_valid_sock_distribution r b g ∧
    r = 1625 ∧
    ∀ (r' b' g' : ℕ), is_valid_sock_distribution r' b' g' → r' ≤ r :=
by sorry

end max_red_socks_l410_41006


namespace quadratic_always_positive_l410_41054

theorem quadratic_always_positive (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 :=
sorry

end quadratic_always_positive_l410_41054


namespace arithmetic_sequence_ratio_l410_41097

/-- Two arithmetic sequences and their sum sequences -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n) ∧ T n = (n / 2) * (b 1 + b n)

/-- The ratio of sums condition -/
def sum_ratio_condition (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n) / (n + 3)

theorem arithmetic_sequence_ratio
  (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h1 : arithmetic_sequences a b S T)
  (h2 : sum_ratio_condition S T) :
  a 5 / b 5 = 21 / 4 := by
sorry

end arithmetic_sequence_ratio_l410_41097


namespace prob_both_paper_is_one_ninth_l410_41060

/-- Represents the possible choices in rock-paper-scissors -/
inductive Choice
| Rock
| Paper
| Scissors

/-- Represents the outcome of a rock-paper-scissors game -/
structure GameOutcome :=
  (player1 : Choice)
  (player2 : Choice)

/-- The set of all possible game outcomes -/
def allOutcomes : Finset GameOutcome :=
  sorry

/-- The set of outcomes where both players choose paper -/
def bothPaperOutcomes : Finset GameOutcome :=
  sorry

/-- The probability of both players choosing paper -/
def probBothPaper : ℚ :=
  (bothPaperOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_both_paper_is_one_ninth :
  probBothPaper = 1 / 9 := by
  sorry

end prob_both_paper_is_one_ninth_l410_41060


namespace outfit_combinations_l410_41057

theorem outfit_combinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : 
  shirts = 8 → ties = 6 → belts = 4 → shirts * ties * belts = 192 := by
  sorry

end outfit_combinations_l410_41057
