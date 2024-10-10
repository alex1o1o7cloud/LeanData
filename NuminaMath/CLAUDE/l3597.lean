import Mathlib

namespace max_circle_area_in_square_l3597_359701

/-- The area of the maximum size circle inscribed in a square -/
theorem max_circle_area_in_square (square_side : ℝ) (h : square_side = 10) :
  π * (square_side / 2)^2 = 25 * π := by
  sorry

end max_circle_area_in_square_l3597_359701


namespace mixture_problem_l3597_359700

/-- Given a mixture of milk and water with an initial ratio of 3:2, 
    if adding 10 liters of water changes the ratio to 2:3, 
    then the initial total quantity was 20 liters. -/
theorem mixture_problem (milk water : ℝ) : 
  milk / water = 3 / 2 →
  milk / (water + 10) = 2 / 3 →
  milk + water = 20 := by
sorry

end mixture_problem_l3597_359700


namespace f_sum_equals_six_l3597_359753

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 9
  else 4^(-x) + 3/2

-- Theorem statement
theorem f_sum_equals_six :
  f 27 + f (-Real.log 3 / Real.log 4) = 6 := by
  sorry

end f_sum_equals_six_l3597_359753


namespace cube_volume_from_painting_cost_l3597_359797

/-- Given a cube with a surface area that costs 343.98 rupees to paint at a rate of 13 paise per square centimeter, prove that the volume of the cube is 9261 cubic centimeters. -/
theorem cube_volume_from_painting_cost (cost : ℚ) (rate : ℚ) (volume : ℕ) : 
  cost = 343.98 ∧ rate = 13 / 100 → volume = 9261 := by
  sorry

end cube_volume_from_painting_cost_l3597_359797


namespace count_divisible_sum_l3597_359759

theorem count_divisible_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0) ∧
  (∀ n : ℕ, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧
  Finset.card S = 5 := by
  sorry

end count_divisible_sum_l3597_359759


namespace fabric_usage_period_l3597_359742

/-- The number of shirts Jenson makes per day -/
def shirts_per_day : ℕ := 3

/-- The number of pants Kingsley makes per day -/
def pants_per_day : ℕ := 5

/-- The amount of fabric used for one shirt (in yards) -/
def fabric_per_shirt : ℕ := 2

/-- The amount of fabric used for one pair of pants (in yards) -/
def fabric_per_pants : ℕ := 5

/-- The total amount of fabric needed (in yards) -/
def total_fabric_needed : ℕ := 93

/-- Theorem: The number of days needed to use the total fabric is 3 -/
theorem fabric_usage_period : 
  (shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pants) * 3 = total_fabric_needed :=
by sorry

end fabric_usage_period_l3597_359742


namespace unknown_number_proof_l3597_359781

theorem unknown_number_proof (x : ℝ) : 
  (x + 30 + 50) / 3 = (20 + 40 + 6) / 3 + 8 → x = 10 :=
by
  sorry

end unknown_number_proof_l3597_359781


namespace product_of_polynomials_l3597_359754

/-- Given two constants m and k, and the equation
    (9d^2 - 5d + m) * (4d^2 + kd - 6) = 36d^4 + 11d^3 - 59d^2 + 10d + 12
    prove that m + k = -7 -/
theorem product_of_polynomials (m k : ℝ) : 
  (∀ d : ℝ, (9*d^2 - 5*d + m) * (4*d^2 + k*d - 6) = 36*d^4 + 11*d^3 - 59*d^2 + 10*d + 12) →
  m + k = -7 := by
  sorry

end product_of_polynomials_l3597_359754


namespace semicircle_rotation_area_l3597_359766

theorem semicircle_rotation_area (R : ℝ) (h : R > 0) :
  let α : ℝ := 20 * π / 180
  let semicircle_area : ℝ := π * R^2 / 2
  let rotated_area : ℝ := α * (2*R)^2 / 2
  rotated_area = 2 * π * R^2 / 9 := by
  sorry

end semicircle_rotation_area_l3597_359766


namespace same_window_probability_l3597_359728

theorem same_window_probability (n : ℕ) (h : n = 3) :
  (n : ℝ) / (n * n : ℝ) = 1 / 3 := by
  sorry

#check same_window_probability

end same_window_probability_l3597_359728


namespace probability_problems_l3597_359748

def bag : Finset ℕ := {1, 2, 3, 4, 5, 6}

def isPrime (n : ℕ) : Prop := Nat.Prime n

def sumIs6 (a b : ℕ) : Prop := a + b = 6

theorem probability_problems :
  (∃ (S : Finset ℕ), S ⊆ bag ∧ (∀ n ∈ S, isPrime n) ∧ S.card / bag.card = 1 / 2) ∧
  (∃ (T : Finset (ℕ × ℕ)), T ⊆ bag.product bag ∧ 
    (∀ p ∈ T, sumIs6 p.1 p.2) ∧ 
    T.card / (bag.card * bag.card) = 5 / 36) := by
  sorry

end probability_problems_l3597_359748


namespace danas_class_size_l3597_359738

/-- Proves that the total number of students in Dana's senior high school class is 200. -/
theorem danas_class_size :
  ∀ (total_students : ℕ),
  (total_students : ℝ) * 0.6 * 0.5 * 0.5 = 30 →
  total_students = 200 := by
  sorry

end danas_class_size_l3597_359738


namespace frustum_cone_volume_l3597_359752

theorem frustum_cone_volume (frustum_volume : ℝ) (base_area_ratio : ℝ) (cone_volume : ℝ) :
  frustum_volume = 78 →
  base_area_ratio = 9 →
  (cone_volume - frustum_volume) / cone_volume = (1 / base_area_ratio.sqrt)^3 →
  cone_volume = 81 := by
sorry

end frustum_cone_volume_l3597_359752


namespace isosceles_triangle_properties_l3597_359722

-- Define the isosceles triangle ∆ABC
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (0, -1)

-- Define the equation of the line containing the altitude
def altitude_line (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the equation of the line containing side BC
def side_BC_line (x y : ℝ) : Prop := 3*x - y - 1 = 0

-- Define the equation of the circumscribed circle
def circumscribed_circle (x y : ℝ) : Prop := (x - 5/2)^2 + (y + 7/2)^2 = 50/4

theorem isosceles_triangle_properties :
  ∃ C : ℝ × ℝ,
    (∀ x y : ℝ, altitude_line x y → side_BC_line x y) ∧
    (∀ x y : ℝ, circumscribed_circle x y) :=
sorry

end isosceles_triangle_properties_l3597_359722


namespace P_subset_Q_l3597_359776

def P : Set ℝ := {x : ℝ | |x| < 2}
def Q : Set ℝ := {x : ℝ | x < 2}

theorem P_subset_Q : P ⊆ Q := by sorry

end P_subset_Q_l3597_359776


namespace multiply_25_26_8_l3597_359723

theorem multiply_25_26_8 : 25 * 26 * 8 = 5200 := by
  sorry

end multiply_25_26_8_l3597_359723


namespace josie_animal_count_l3597_359719

/-- The number of antelopes Josie counted -/
def num_antelopes : ℕ := 80

/-- The number of rabbits Josie counted -/
def num_rabbits : ℕ := num_antelopes + 34

/-- The number of hyenas Josie counted -/
def num_hyenas : ℕ := num_antelopes + num_rabbits - 42

/-- The number of wild dogs Josie counted -/
def num_wild_dogs : ℕ := num_hyenas + 50

/-- The number of leopards Josie counted -/
def num_leopards : ℕ := num_rabbits / 2

/-- The total number of animals Josie counted -/
def total_animals : ℕ := num_antelopes + num_rabbits + num_hyenas + num_wild_dogs + num_leopards

theorem josie_animal_count : total_animals = 605 := by
  sorry

end josie_animal_count_l3597_359719


namespace cuboid_edge_length_l3597_359734

/-- Definition of a cuboid with given dimensions and surface area -/
structure Cuboid where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  surface_area : ℝ

/-- The surface area formula for a cuboid -/
def surface_area_formula (c : Cuboid) : ℝ :=
  2 * (c.edge1 * c.edge2 + c.edge1 * c.edge3 + c.edge2 * c.edge3)

/-- Theorem: For a cuboid with edges 4 cm, x cm, and 6 cm, and a surface area of 148 cm², 
    the length of the second edge (x) is 5 cm -/
theorem cuboid_edge_length (c : Cuboid) 
    (h1 : c.edge1 = 4)
    (h2 : c.edge3 = 6)
    (h3 : c.surface_area = 148)
    (h4 : surface_area_formula c = c.surface_area) :
    c.edge2 = 5 := by
  sorry


end cuboid_edge_length_l3597_359734


namespace b_more_stable_than_a_l3597_359735

/-- Represents a shooter in the competition -/
structure Shooter where
  variance : ℝ

/-- Defines the stability of a shooter based on their variance -/
def is_more_stable (a b : Shooter) : Prop :=
  a.variance < b.variance

/-- Theorem stating that shooter B is more stable than shooter A -/
theorem b_more_stable_than_a :
  let shooterA : Shooter := ⟨1.8⟩
  let shooterB : Shooter := ⟨0.7⟩
  is_more_stable shooterB shooterA :=
by
  sorry

end b_more_stable_than_a_l3597_359735


namespace line_y_intercept_l3597_359795

/-- A straight line in the xy-plane passing through (100, 1000) with slope 9.9 has y-intercept 10 -/
theorem line_y_intercept :
  ∀ (f : ℝ → ℝ),
  (∀ x y, f x = 9.9 * x + y) →  -- Line equation with slope 9.9 and y-intercept y
  f 100 = 1000 →               -- Line passes through (100, 1000)
  f 0 = 10 :=                  -- y-intercept is 10
by
  sorry

end line_y_intercept_l3597_359795


namespace pizza_geometric_sum_l3597_359793

/-- The sum of a geometric series with first term 1/4, common ratio 1/2, and 6 terms -/
def geometricSum : ℚ :=
  let a : ℚ := 1/4
  let r : ℚ := 1/2
  let n : ℕ := 6
  a * (1 - r^n) / (1 - r)

/-- The fraction of pizza eaten after 6 trips -/
def pizzaEaten : ℚ := 63/128

theorem pizza_geometric_sum :
  geometricSum = pizzaEaten := by sorry

end pizza_geometric_sum_l3597_359793


namespace time_to_mow_one_line_l3597_359747

def total_lines : ℕ := 40
def total_flowers : ℕ := 56
def time_per_flower : ℚ := 1/2
def total_gardening_time : ℕ := 108

theorem time_to_mow_one_line :
  (total_gardening_time - total_flowers * time_per_flower) / total_lines = 2 := by
  sorry

end time_to_mow_one_line_l3597_359747


namespace line_through_point_with_slope_l3597_359751

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Creates a Line from a point and a slope -/
def lineFromPointSlope (x y m : ℝ) : Line :=
  { slope := m, yIntercept := y - m * x }

/-- The equation of a line in the form y = mx + b -/
def lineEquation (l : Line) (x : ℝ) : ℝ := l.slope * x + l.yIntercept

theorem line_through_point_with_slope (x₀ y₀ m : ℝ) :
  let l := lineFromPointSlope x₀ y₀ m
  ∀ x, lineEquation l x = m * (x - x₀) + y₀ := by sorry

end line_through_point_with_slope_l3597_359751


namespace odd_function_property_l3597_359764

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : OddFunction f) 
  (h_even : EvenFunction (fun x ↦ f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 9 = 1 := by
  sorry

end odd_function_property_l3597_359764


namespace c_plus_d_equals_negative_two_l3597_359717

theorem c_plus_d_equals_negative_two 
  (c d : ℝ) 
  (h1 : 2 = c + d / 3) 
  (h2 : 6 = c + d / (-3)) : 
  c + d = -2 := by
sorry

end c_plus_d_equals_negative_two_l3597_359717


namespace solve_equations_l3597_359782

theorem solve_equations :
  (∀ x : ℝ, (x - 2)^2 - 1 = 0 ↔ x = 3 ∨ x = 1) ∧
  (∀ x : ℝ, 3*(x - 2)^2 = x*(x - 2) ↔ x = 2 ∨ x = 3) ∧
  (∀ x : ℝ, 2*x^2 + 4*x - 5 = 0 ↔ x = -1 + Real.sqrt 14 / 2 ∨ x = -1 - Real.sqrt 14 / 2) :=
by sorry

end solve_equations_l3597_359782


namespace third_basket_apples_l3597_359788

/-- The number of apples originally in the third basket -/
def apples_in_third_basket : ℕ := 655

theorem third_basket_apples :
  ∀ (x y : ℕ),
  -- Total number of apples in all baskets
  (x + 2*y) + (x + 49) + (x + y) = 2014 →
  -- Number of apples left in first basket is twice the number left in third
  2*y = 2*(x + y - apples_in_third_basket) →
  -- The original number of apples in the third basket
  apples_in_third_basket = x + y :=
by
  sorry

#check third_basket_apples

end third_basket_apples_l3597_359788


namespace composition_of_even_and_odd_is_even_l3597_359787

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being an odd function
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Theorem statement
theorem composition_of_even_and_odd_is_even
  (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsEven (f ∘ g) := by sorry

end composition_of_even_and_odd_is_even_l3597_359787


namespace smallest_valid_number_correct_l3597_359736

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit number
  (n % 2 = 0) ∧  -- Even
  (n % 3 = 0) ∧  -- Divisible by 3
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits.toFinset = {1, 2, 3, 4, 9}  -- Uses each digit exactly once

def smallest_valid_number : ℕ := 14932

theorem smallest_valid_number_correct :
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≥ smallest_valid_number) ∧
  ((smallest_valid_number / 10) % 10 = 3) :=
by sorry

#eval smallest_valid_number
#eval (smallest_valid_number / 10) % 10

end smallest_valid_number_correct_l3597_359736


namespace baseball_card_packs_l3597_359706

/-- The number of people buying baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- The total number of packs of baseball cards -/
def total_packs : ℕ := (num_people * cards_per_person) / cards_per_pack

theorem baseball_card_packs :
  total_packs = 108 := by sorry

end baseball_card_packs_l3597_359706


namespace fraction_difference_l3597_359772

theorem fraction_difference : 
  let a := 2^2 + 4^2 + 6^2
  let b := 1^2 + 3^2 + 5^2
  (a / b) - (b / a) = 1911 / 1960 := by
  sorry

end fraction_difference_l3597_359772


namespace divisibility_by_five_l3597_359799

theorem divisibility_by_five (x y : ℕ+) (h1 : 2 * x^2 - 1 = y^15) (h2 : x > 1) : 
  5 ∣ x.val := by
sorry

end divisibility_by_five_l3597_359799


namespace union_dues_deduction_l3597_359749

def weekly_hours : ℕ := 42
def hourly_rate : ℚ := 10
def tax_rate : ℚ := 1/5
def insurance_rate : ℚ := 1/20
def take_home_pay : ℚ := 310

theorem union_dues_deduction :
  let gross_earnings := weekly_hours * hourly_rate
  let tax_deduction := tax_rate * gross_earnings
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := tax_deduction + insurance_deduction
  let net_before_union := gross_earnings - total_deductions
  net_before_union - take_home_pay = 5 := by sorry

end union_dues_deduction_l3597_359749


namespace min_value_a_sqrt_inequality_l3597_359785

theorem min_value_a_sqrt_inequality : 
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧ 
  (∀ (a : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ Real.sqrt 2) :=
by sorry

end min_value_a_sqrt_inequality_l3597_359785


namespace matrix_power_4_l3597_359724

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 0]]

theorem matrix_power_4 : A^4 = ![![5, 3], ![3, 2]] := by sorry

end matrix_power_4_l3597_359724


namespace simplify_fraction_l3597_359745

-- Define the expression
def f : ℚ := 1 / (1 / (1/2)^1 + 1 / (1/2)^3 + 1 / (1/2)^4)

-- Theorem statement
theorem simplify_fraction : f = 1 / 26 := by
  sorry

end simplify_fraction_l3597_359745


namespace opposite_sides_iff_a_in_range_l3597_359740

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation 3x - 2y + a = 0 -/
def line_equation (p : Point) (a : ℝ) : ℝ := 3 * p.x - 2 * p.y + a

/-- Two points are on opposite sides of the line if their line equation values have opposite signs -/
def opposite_sides (p1 p2 : Point) (a : ℝ) : Prop :=
  line_equation p1 a * line_equation p2 a < 0

/-- The main theorem -/
theorem opposite_sides_iff_a_in_range :
  ∀ (a : ℝ),
  opposite_sides (Point.mk 3 1) (Point.mk (-4) 6) a ↔ a > -7 ∧ a < 24 := by
  sorry

end opposite_sides_iff_a_in_range_l3597_359740


namespace quadratic_inequality_range_l3597_359779

theorem quadratic_inequality_range (x : ℝ) : 
  x^2 - 5*x + 6 ≤ 0 → 
  28 ≤ x^2 + 7*x + 10 ∧ x^2 + 7*x + 10 ≤ 40 := by
sorry

end quadratic_inequality_range_l3597_359779


namespace lcm_gcf_ratio_l3597_359708

theorem lcm_gcf_ratio : (Nat.lcm 240 540) / (Nat.gcd 240 540) = 36 := by
  sorry

end lcm_gcf_ratio_l3597_359708


namespace workers_efficiency_ratio_l3597_359713

/-- Given two workers A and B, where B can finish a job in 15 days and together they
    finish the job in 10 days, prove that the ratio of A's work efficiency to B's is 1/2. -/
theorem workers_efficiency_ratio
  (finish_time_B : ℝ)
  (finish_time_together : ℝ)
  (hB : finish_time_B = 15)
  (hTogether : finish_time_together = 10)
  (efficiency_ratio : ℝ)
  (h_efficiency : efficiency_ratio * (1 / finish_time_B) + (1 / finish_time_B) = 1 / finish_time_together) :
  efficiency_ratio = 1 / 2 :=
sorry

end workers_efficiency_ratio_l3597_359713


namespace hypotenuse_length_l3597_359774

theorem hypotenuse_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (right_angle : a^2 + b^2 = c^2)
  (sum_squares : a^2 + b^2 + c^2 = 1800) : c = 30 := by
  sorry

end hypotenuse_length_l3597_359774


namespace unique_quadratic_function_l3597_359750

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem unique_quadratic_function (f : ℝ → ℝ) 
  (hf : QuadraticFunction f)
  (h0 : f 0 = -5)
  (h1 : f (-1) = -4)
  (h2 : f 2 = -5) :
  ∀ x, f x = (1/3) * x^2 - (2/3) * x - 5 := by
  sorry

end unique_quadratic_function_l3597_359750


namespace append_digits_to_perfect_square_l3597_359796

/-- The number formed by 99 nines in a row -/
def X : ℕ := 10^99 - 1

/-- Theorem stating that there exists a natural number n such that 
    X * 10^100 ≤ n^2 < X * 10^100 + 10^100 -/
theorem append_digits_to_perfect_square :
  ∃ n : ℕ, X * 10^100 ≤ n^2 ∧ n^2 < X * 10^100 + 10^100 := by
  sorry

end append_digits_to_perfect_square_l3597_359796


namespace similar_triangles_height_l3597_359709

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 15 :=
by sorry

end similar_triangles_height_l3597_359709


namespace vector_operation_l3597_359707

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, -1)

theorem vector_operation :
  (1/3 : ℝ) • a - (4/3 : ℝ) • b = (-1, 2) := by
  sorry

end vector_operation_l3597_359707


namespace parallel_vectors_t_value_l3597_359758

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a and b, if they are parallel, then t = 2 or t = -2 -/
theorem parallel_vectors_t_value (t : ℝ) :
  let a : ℝ × ℝ := (1, t)
  let b : ℝ × ℝ := (t, 4)
  parallel a b → t = 2 ∨ t = -2 := by
  sorry


end parallel_vectors_t_value_l3597_359758


namespace bucket_capacity_ratio_l3597_359720

/-- Given two buckets P and Q, prove that their capacity ratio is 3:1 based on their filling times. -/
theorem bucket_capacity_ratio (P Q : ℝ) : 
  (P > 0) →  -- Bucket P has positive capacity
  (Q > 0) →  -- Bucket Q has positive capacity
  (60 * P = 45 * (P + Q)) →  -- Filling condition
  (P / Q = 3) :=  -- Ratio of capacities
by sorry

end bucket_capacity_ratio_l3597_359720


namespace mountain_lake_depth_l3597_359777

/-- Represents a cone-shaped mountain partially submerged in water -/
structure Mountain where
  height : ℝ
  above_water_ratio : ℝ

/-- Calculates the depth of the lake at the base of a partially submerged mountain -/
def lake_depth (m : Mountain) : ℝ :=
  m.height * (1 - (1 - m.above_water_ratio)^(1/3))

/-- Theorem stating that for a mountain of height 12000 feet with 1/6 of its volume above water,
    the depth of the lake at its base is 780 feet -/
theorem mountain_lake_depth :
  let m : Mountain := { height := 12000, above_water_ratio := 1/6 }
  lake_depth m = 780 := by sorry

end mountain_lake_depth_l3597_359777


namespace diamond_equal_is_three_lines_l3597_359773

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ◇ y = y ◇ x -/
def diamond_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and y = -x -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.2 = -p.1}

theorem diamond_equal_is_three_lines :
  diamond_equal_set = three_lines :=
sorry

end diamond_equal_is_three_lines_l3597_359773


namespace joohyeon_snacks_l3597_359784

/-- Represents the number of snacks bought by Joohyeon -/
def num_snacks : ℕ := 3

/-- Represents the number of candies bought by Joohyeon -/
def num_candies : ℕ := 5

/-- Cost of each candy in won -/
def candy_cost : ℕ := 300

/-- Cost of each snack in won -/
def snack_cost : ℕ := 500

/-- Total amount spent in won -/
def total_spent : ℕ := 3000

/-- Total number of items bought -/
def total_items : ℕ := 8

theorem joohyeon_snacks :
  (num_candies * candy_cost + num_snacks * snack_cost = total_spent) ∧
  (num_candies + num_snacks = total_items) :=
by sorry

end joohyeon_snacks_l3597_359784


namespace perpendicular_lines_k_values_l3597_359755

-- Define the coefficients of the lines
def a (k : ℝ) := k - 3
def b (k : ℝ) := 5 - k
def m (k : ℝ) := 2 * (k - 3)
def n : ℝ := -2

-- Define the perpendicularity condition
def perpendicular (k : ℝ) : Prop := a k * m k + b k * n = 0

-- Theorem statement
theorem perpendicular_lines_k_values :
  ∀ k : ℝ, perpendicular k → k = 1 ∨ k = 4 := by
  sorry

end perpendicular_lines_k_values_l3597_359755


namespace trajectory_of_moving_circle_l3597_359783

/-- The equation of the trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop := x^2 = -4*(y - 1)

/-- The semicircle in which the moving circle is inscribed -/
def semicircle (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ 0 ≤ y ∧ y ≤ 2

/-- The moving circle is tangent to the x-axis -/
def tangent_to_x_axis (x y : ℝ) : Prop := ∃ (r : ℝ), r > 0 ∧ y = r

theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), 
    0 < y → y ≤ 1 →
    tangent_to_x_axis x y →
    (∃ (x' y' : ℝ), semicircle x' y' ∧ 
      (x - x')^2 + (y - y')^2 = (2 - y)^2) →
    trajectory_equation x y :=
sorry

end trajectory_of_moving_circle_l3597_359783


namespace rectangle_diagonal_in_hexagon_l3597_359726

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- A rectangle inside the hexagon -/
structure Rectangle (h : RegularHexagon) :=
  (length : ℝ)
  (width : ℝ)
  (inside_hexagon : length + width ≤ h.side_length)

/-- Two congruent rectangles inside the hexagon -/
structure CongruentRectangles (h : RegularHexagon) :=
  (rect1 : Rectangle h)
  (rect2 : Rectangle h)
  (congruent : rect1.length = rect2.length ∧ rect1.width = rect2.width)

/-- The theorem to be proved -/
theorem rectangle_diagonal_in_hexagon 
  (h : RegularHexagon) 
  (r : CongruentRectangles h) : 
  Real.sqrt (r.rect1.length ^ 2 + r.rect1.width ^ 2) = 2 :=
sorry

end rectangle_diagonal_in_hexagon_l3597_359726


namespace garden_ratio_theorem_l3597_359718

/-- Represents the dimensions of a square garden surrounded by rectangular flower beds -/
structure GardenDimensions where
  s : ℝ  -- side length of the square garden
  x : ℝ  -- longer side of each rectangular bed
  y : ℝ  -- shorter side of each rectangular bed

/-- The theorem stating the ratio of the longer side to the shorter side of each rectangular bed -/
theorem garden_ratio_theorem (d : GardenDimensions) 
  (h1 : d.s > 0)  -- the garden has positive side length
  (h2 : d.s + 2 * d.y = Real.sqrt 3 * d.s)  -- outer square side length relation
  (h3 : d.x + d.y = Real.sqrt 3 * d.s)  -- outer square diagonal relation
  : d.x / d.y = 2 + Real.sqrt 3 := by
  sorry

end garden_ratio_theorem_l3597_359718


namespace four_numbers_proof_l3597_359780

theorem four_numbers_proof :
  ∀ (a b c d : ℝ),
    (a / b = 1/5 / (1/3)) →
    (a / c = 1/5 / (1/20)) →
    (b / c = 1/3 / (1/20)) →
    (d = 0.15 * b) →
    (b = a + c + d + 8) →
    (a = 48 ∧ b = 80 ∧ c = 12 ∧ d = 12) := by
  sorry

end four_numbers_proof_l3597_359780


namespace target_hit_probability_l3597_359786

theorem target_hit_probability (p_A p_B : ℝ) (h_A : p_A = 0.8) (h_B : p_B = 0.7) :
  1 - (1 - p_A) * (1 - p_B) = 0.94 := by
  sorry

end target_hit_probability_l3597_359786


namespace eight_sided_die_expected_value_l3597_359727

/-- The number of sides on the die -/
def num_sides : ℕ := 8

/-- The set of possible outcomes when rolling the die -/
def outcomes : Finset ℕ := Finset.range num_sides

/-- The expected value of rolling the die -/
def expected_value : ℚ := (Finset.sum outcomes (λ i => i + 1)) / num_sides

/-- Theorem: The expected value of rolling an eight-sided die with faces numbered from 1 to 8 is 4.5 -/
theorem eight_sided_die_expected_value :
  expected_value = 9/2 := by
  sorry

end eight_sided_die_expected_value_l3597_359727


namespace largest_divisor_five_consecutive_integers_l3597_359757

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) := by
sorry

end largest_divisor_five_consecutive_integers_l3597_359757


namespace oranges_taken_l3597_359705

/-- Given a basket of oranges, prove that the number of oranges taken is 5 -/
theorem oranges_taken (original : ℕ) (remaining : ℕ) (taken : ℕ) : 
  original = 8 → remaining = 3 → taken = original - remaining → taken = 5 := by
  sorry

end oranges_taken_l3597_359705


namespace congruence_problem_l3597_359768

theorem congruence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end congruence_problem_l3597_359768


namespace f_symmetry_l3597_359789

/-- Given a function f(x) = x³ + 2x, prove that f(a) + f(-a) = 0 for any real number a -/
theorem f_symmetry (a : ℝ) : let f (x : ℝ) := x^3 + 2*x; f a + f (-a) = 0 := by
  sorry

end f_symmetry_l3597_359789


namespace polygon_sides_when_angles_equal_l3597_359763

theorem polygon_sides_when_angles_equal (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 360 ↔ n = 4 := by
  sorry

end polygon_sides_when_angles_equal_l3597_359763


namespace kiras_breakfast_time_l3597_359771

/-- The time it takes Kira to make breakfast given the number of sausages, eggs, and cooking times -/
def breakfast_time (num_sausages : ℕ) (num_eggs : ℕ) (sausage_time : ℕ) (egg_time : ℕ) : ℕ :=
  num_sausages * sausage_time + num_eggs * egg_time

/-- Theorem stating that Kira's breakfast time is 39 minutes -/
theorem kiras_breakfast_time :
  breakfast_time 3 6 5 4 = 39 := by
  sorry

end kiras_breakfast_time_l3597_359771


namespace num_paths_to_bottom_right_l3597_359741

/-- Represents a vertex in the triangle grid --/
structure Vertex :=
  (x : Nat) (y : Nat)

/-- The number of paths to a vertex in the triangle grid --/
def numPaths : Vertex → Nat
| ⟨0, 0⟩ => 1  -- Top vertex
| ⟨0, y⟩ => 1  -- Left edge
| ⟨x, y⟩ => sorry  -- Other vertices

/-- The bottom right vertex of the triangle --/
def bottomRightVertex : Vertex :=
  ⟨3, 3⟩

/-- Theorem stating the number of paths to the bottom right vertex --/
theorem num_paths_to_bottom_right :
  numPaths bottomRightVertex = 22 := by sorry

end num_paths_to_bottom_right_l3597_359741


namespace tunnel_length_is_900_l3597_359765

/-- Calculates the length of a tunnel given train parameters -/
def tunnel_length (train_length : ℝ) (total_time : ℝ) (inside_time : ℝ) : ℝ :=
  -- Define the tunnel length calculation here
  0 -- Placeholder, replace with actual calculation

/-- Theorem stating that given the specified conditions, the tunnel length is 900 meters -/
theorem tunnel_length_is_900 :
  tunnel_length 300 60 30 = 900 := by
  sorry

end tunnel_length_is_900_l3597_359765


namespace least_distance_eight_girls_circle_l3597_359739

/-- The least total distance traveled by 8 girls on a circle -/
theorem least_distance_eight_girls_circle (r : ℝ) (h : r = 50) :
  let n : ℕ := 8  -- number of girls
  let angle := 2 * Real.pi / n  -- angle between adjacent girls
  let non_adjacent_angle := 3 * angle  -- angle to third girl (non-adjacent)
  let single_path := r * Real.sqrt (2 + Real.sqrt 2)  -- distance to one non-adjacent girl and back
  let total_distance := n * 4 * single_path  -- total distance for all girls
  total_distance = 1600 * Real.sqrt (2 + Real.sqrt 2) :=
by sorry


end least_distance_eight_girls_circle_l3597_359739


namespace inequality_solution_implies_a_value_l3597_359756

theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 4 < 0 ↔ 1 < x ∧ x < 4) → a = 5 := by
  sorry

end inequality_solution_implies_a_value_l3597_359756


namespace no_integer_solution_l3597_359798

theorem no_integer_solution : ¬∃ (a b : ℤ), a^2 + 1998 = b^2 := by
  sorry

end no_integer_solution_l3597_359798


namespace simplify_and_evaluate_l3597_359712

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 1) (h2 : b = -2) :
  (2*a + b)^2 - 3*a*(2*a - b) = -12 := by
  sorry

end simplify_and_evaluate_l3597_359712


namespace smallest_positive_integer_1729m_78945n_l3597_359725

theorem smallest_positive_integer_1729m_78945n :
  ∃ (m n : ℤ), 1729 * m + 78945 * n = (1 : ℤ) ∧
  ∀ (k : ℤ), k > 0 → (∃ (x y : ℤ), 1729 * x + 78945 * y = k) → k ≥ 1 :=
sorry

end smallest_positive_integer_1729m_78945n_l3597_359725


namespace joyce_typing_speed_l3597_359770

def team_size : ℕ := 5
def team_average : ℕ := 80
def rudy_speed : ℕ := 64
def gladys_speed : ℕ := 91
def lisa_speed : ℕ := 80
def mike_speed : ℕ := 89

theorem joyce_typing_speed :
  ∃ (joyce_speed : ℕ),
    joyce_speed = team_size * team_average - (rudy_speed + gladys_speed + lisa_speed + mike_speed) ∧
    joyce_speed = 76 := by
  sorry

end joyce_typing_speed_l3597_359770


namespace root_sum_product_l3597_359746

theorem root_sum_product (a b : ℝ) : 
  (a^2 + a - 3 = 0) → (b^2 + b - 3 = 0) → (ab - 2023*a - 2023*b = 2020) := by
  sorry

end root_sum_product_l3597_359746


namespace min_value_theorem_l3597_359769

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 2) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x^2 * y * z) ≤ (a + b) / (a^2 * b * c)) →
  (x + y) / (x^2 * y * z) = 13.5 :=
sorry

end min_value_theorem_l3597_359769


namespace parabola_directrix_parameter_l3597_359743

/-- Given a parabola with equation x^2 = ay and directrix y = -2, prove that a = 8 -/
theorem parabola_directrix_parameter (x y a : ℝ) : 
  (∀ x y, x^2 = a * y) →  -- Parabola equation
  (∃ p, p = -2 ∧ ∀ x, x^2 = a * p) →  -- Directrix equation
  a = 8 := by
sorry

end parabola_directrix_parameter_l3597_359743


namespace third_player_games_l3597_359704

/-- Represents a table tennis game with three players. -/
structure TableTennisGame where
  totalGames : ℕ
  player1Games : ℕ
  player2Games : ℕ
  player3Games : ℕ

/-- The rules of the game ensure that the total number of games is equal to the maximum number of games played by any player. -/
axiom total_games_rule (game : TableTennisGame) : game.totalGames = max game.player1Games (max game.player2Games game.player3Games)

/-- The sum of games played by all players is twice the total number of games. -/
axiom sum_of_games_rule (game : TableTennisGame) : game.player1Games + game.player2Games + game.player3Games = 2 * game.totalGames

/-- Theorem: In a three-player table tennis game where the loser gives up their spot, 
    if the first player plays 10 games and the second player plays 21 games, 
    then the third player must play 11 games. -/
theorem third_player_games (game : TableTennisGame) 
    (h1 : game.player1Games = 10) 
    (h2 : game.player2Games = 21) : 
  game.player3Games = 11 := by
  sorry

end third_player_games_l3597_359704


namespace linear_function_condition_l3597_359791

/-- A linear function f(x) = ax + b satisfies f(x)f(y) + f(x+y-xy) ≤ 0 for all x, y ∈ [0, 1]
    if and only if -1 ≤ b ≤ 0 and -(b + 1) ≤ a ≤ -b -/
theorem linear_function_condition (a b : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 →
    (a * x + b) * (a * y + b) + (a * (x + y - x * y) + b) ≤ 0) ↔
  (b ∈ Set.Icc (-1) 0 ∧ a ∈ Set.Icc (-(b + 1)) (-b)) :=
sorry

end linear_function_condition_l3597_359791


namespace equal_gumball_share_l3597_359775

/-- Calculates the number of gumballs each person gets after combining and equally sharing their initial amounts and purchases. -/
def gumballs_per_person (joanna_initial : ℕ) (jacques_initial : ℕ) (purchase_multiplier : ℕ) : ℕ :=
  let joanna_total := joanna_initial + joanna_initial * purchase_multiplier
  let jacques_total := jacques_initial + jacques_initial * purchase_multiplier
  let combined_total := joanna_total + jacques_total
  combined_total / 2

/-- Theorem stating that given the initial conditions and purchases, when Joanna and Jacques combine and equally share their gumballs, each person gets 250 gumballs. -/
theorem equal_gumball_share :
  gumballs_per_person 40 60 4 = 250 := by
  sorry

end equal_gumball_share_l3597_359775


namespace m_range_l3597_359711

-- Define the set M
def M : Set ℝ := {x | 3 * x^2 - 5 * x - 2 ≤ 0}

-- Define the set N
def N (m : ℝ) : Set ℝ := {m, m + 1}

-- Theorem statement
theorem m_range (m : ℝ) :
  M ∪ N m = M → m ∈ Set.Icc (-1/3 : ℝ) 1 :=
by sorry

end m_range_l3597_359711


namespace two_negative_roots_iff_q_gt_3sqrt2_div_4_l3597_359716

/-- A polynomial of degree 4 with parameter q -/
def f (q : ℝ) (x : ℝ) : ℝ := x^4 + 2*q*x^3 + 2*x^2 + 2*q*x + 4

/-- The condition for the polynomial to have at least two distinct negative real roots -/
def has_two_negative_roots (q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ f q x₁ = 0 ∧ f q x₂ = 0

/-- The theorem stating the condition on q for the polynomial to have at least two distinct negative real roots -/
theorem two_negative_roots_iff_q_gt_3sqrt2_div_4 :
  ∀ q : ℝ, has_two_negative_roots q ↔ q > 3 * Real.sqrt 2 / 4 :=
sorry

end two_negative_roots_iff_q_gt_3sqrt2_div_4_l3597_359716


namespace scalene_triangle_perimeter_scalene_triangle_perimeter_proof_l3597_359721

/-- A scalene triangle with sides of lengths 15, 10, and 7 has a perimeter of 32. -/
theorem scalene_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p =>
    a = 15 ∧ b = 10 ∧ c = 7 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    p = a + b + c →
    p = 32

/-- Proof of the theorem -/
theorem scalene_triangle_perimeter_proof : scalene_triangle_perimeter 15 10 7 32 := by
  sorry

end scalene_triangle_perimeter_scalene_triangle_perimeter_proof_l3597_359721


namespace sin_a_n_bound_l3597_359702

theorem sin_a_n_bound (a : ℕ → ℝ) :
  (a 1 = π / 3) →
  (∀ n, 0 < a n ∧ a n < π / 3) →
  (∀ n ≥ 2, Real.sin (a (n + 1)) ≤ (1 / 3) * Real.sin (3 * a n)) →
  ∀ n, Real.sin (a n) < 1 / Real.sqrt n :=
by sorry

end sin_a_n_bound_l3597_359702


namespace greatest_three_digit_multiple_of_17_l3597_359729

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l3597_359729


namespace total_horse_food_needed_l3597_359715

/- Given a farm with sheep and horses -/
structure Farm where
  sheep : ℕ
  horses : ℕ
  sheep_to_horse_ratio : ℚ
  food_per_horse : ℕ

/- Define the specific farm from the problem -/
def stewart_farm : Farm where
  sheep := 24
  horses := 56
  sheep_to_horse_ratio := 3 / 7
  food_per_horse := 230

/- Theorem statement -/
theorem total_horse_food_needed (f : Farm) (h1 : f.sheep = 24) 
    (h2 : f.sheep_to_horse_ratio = 3 / 7) (h3 : f.food_per_horse = 230) : 
    f.horses * f.food_per_horse = 12880 := by
  sorry

#check total_horse_food_needed

end total_horse_food_needed_l3597_359715


namespace total_coins_l3597_359737

def coin_distribution (x : ℕ) : Prop :=
  ∃ (pete paul : ℕ),
    paul = x ∧
    pete = 3 * x ∧
    pete = x * (x + 1) ∧
    x > 0

theorem total_coins (x : ℕ) (h : coin_distribution x) : x + 3 * x = 8 := by
  sorry

end total_coins_l3597_359737


namespace correct_calculation_l3597_359710

theorem correct_calculation (x : ℝ) (h : x / 6 = 52) : x + 40 = 352 := by
  sorry

end correct_calculation_l3597_359710


namespace mutually_exclusive_events_l3597_359761

def S : Set ℕ := {1, 2, 3, 4, 5}

def event1 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ (a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0)

def event2 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ ((a % 2 = 1 ∨ b % 2 = 1) ∧ (a % 2 = 1 ∧ b % 2 = 1))

def event3 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ ((a % 2 = 1 ∨ b % 2 = 1) ∧ (a % 2 = 0 ∧ b % 2 = 0))

def event4 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ (a % 2 = 1 ∨ b % 2 = 1) ∧ (a % 2 = 0 ∨ b % 2 = 0)

theorem mutually_exclusive_events :
  ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b →
  (¬(event1 a b ∧ event1 a b)) ∧
  (¬(event2 a b ∧ event2 a b)) ∧
  (¬(event3 a b ∧ event3 a b)) ∧
  (¬(event4 a b ∧ event4 a b)) ∧
  (∃ x y : ℕ, event1 x y ∧ event2 x y) ∧
  (∃ x y : ℕ, event1 x y ∧ event4 x y) ∧
  (∃ x y : ℕ, event2 x y ∧ event4 x y) ∧
  (∀ x y : ℕ, ¬(event3 x y ∧ event1 x y)) ∧
  (∀ x y : ℕ, ¬(event3 x y ∧ event2 x y)) ∧
  (∀ x y : ℕ, ¬(event3 x y ∧ event4 x y)) :=
by
  sorry

end mutually_exclusive_events_l3597_359761


namespace katies_ds_games_l3597_359714

/-- Theorem: Katie's DS Games
Given:
- Katie's new friends have 88 games
- Katie's old friends have 53 games
- All friends (including Katie) have 141 games in total
Prove that Katie has 0 DS games
-/
theorem katies_ds_games 
  (new_friends_games : ℕ) 
  (old_friends_games : ℕ)
  (total_games : ℕ)
  (h1 : new_friends_games = 88)
  (h2 : old_friends_games = 53)
  (h3 : total_games = 141)
  (h4 : total_games = new_friends_games + old_friends_games + katie_games)
  : katie_games = 0 := by
  sorry

#check katies_ds_games

end katies_ds_games_l3597_359714


namespace arithmetic_expression_equality_l3597_359767

theorem arithmetic_expression_equality : 8 / 4 - 3 * 2 + 9 - 3^2 = -4 := by
  sorry

end arithmetic_expression_equality_l3597_359767


namespace ratio_problem_l3597_359778

theorem ratio_problem (x y : ℝ) (h : (3*x - 2*y) / (2*x + y) = 4/5) : x / y = 2 := by
  sorry

end ratio_problem_l3597_359778


namespace lakota_spending_l3597_359744

/-- The price of a new compact disk -/
def new_cd_price : ℚ := 17.99

/-- The price of a used compact disk -/
def used_cd_price : ℚ := 9.99

/-- The number of new CDs Lakota bought -/
def lakota_new_cds : ℕ := 6

/-- The number of used CDs Lakota bought -/
def lakota_used_cds : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new_cds : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used_cds : ℕ := 8

/-- The total amount Mackenzie spent -/
def mackenzie_total : ℚ := 133.89

theorem lakota_spending :
  (lakota_new_cds : ℚ) * new_cd_price + (lakota_used_cds : ℚ) * used_cd_price = 127.92 :=
by sorry

end lakota_spending_l3597_359744


namespace rectangular_field_dimensions_l3597_359731

theorem rectangular_field_dimensions (m : ℝ) : 
  (2*m + 7) * (m - 2) = 51 → m = 5 := by
  sorry

end rectangular_field_dimensions_l3597_359731


namespace distance_on_line_l3597_359733

/-- The distance between two points on a line y = kx + b -/
theorem distance_on_line (k b x₁ x₂ : ℝ) :
  let P : ℝ × ℝ := (x₁, k * x₁ + b)
  let Q : ℝ × ℝ := (x₂, k * x₂ + b)
  ‖P - Q‖ = |x₁ - x₂| * Real.sqrt (1 + k^2) :=
by sorry

end distance_on_line_l3597_359733


namespace limit_implies_a_equals_one_l3597_359703

theorem limit_implies_a_equals_one (a : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((a * n - 2) / (n + 1)) - 1| < ε) →
  a = 1 := by
sorry

end limit_implies_a_equals_one_l3597_359703


namespace inequality_and_equality_conditions_l3597_359730

theorem inequality_and_equality_conditions 
  (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) :
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ∧
  (8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) = 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) ↔ 
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end inequality_and_equality_conditions_l3597_359730


namespace lucy_bought_six_fifty_cent_items_l3597_359794

/-- Represents the number of items Lucy bought at each price point -/
structure PurchasedItems where
  fifty_cent : ℕ
  one_fifty : ℕ
  three_dollar : ℕ

/-- The total number of items Lucy bought -/
def total_items : ℕ := 30

/-- The total purchase price in cents -/
def total_price : ℕ := 4500

/-- Theorem stating that Lucy bought 6 items at 50 cents -/
theorem lucy_bought_six_fifty_cent_items :
  ∃ (items : PurchasedItems),
    items.fifty_cent + items.one_fifty + items.three_dollar = total_items ∧
    50 * items.fifty_cent + 150 * items.one_fifty + 300 * items.three_dollar = total_price ∧
    items.fifty_cent = 6 := by
  sorry

end lucy_bought_six_fifty_cent_items_l3597_359794


namespace blood_sample_count_l3597_359762

theorem blood_sample_count (total_cells : ℕ) (first_sample_cells : ℕ) : 
  total_cells = 7341 → first_sample_cells = 4221 → total_cells - first_sample_cells = 3120 := by
  sorry

end blood_sample_count_l3597_359762


namespace parabola_circle_tangency_l3597_359790

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the line l
def line_l (x : ℝ) : Prop := x = 1

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

-- Define a line tangent to circle M
def tangent_to_circle_M (A B : ℝ × ℝ) : Prop :=
  ∃ (k m : ℝ), ∀ (x y : ℝ), y = k * x + m → 
    ((x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)) →
    (2 * k + 2 * m - 4)^2 / (k^2 + 1) = 1

theorem parabola_circle_tangency 
  (A₁ A₂ A₃ : ℝ × ℝ) 
  (h₁ : point_on_parabola A₁) 
  (h₂ : point_on_parabola A₂) 
  (h₃ : point_on_parabola A₃) 
  (h₄ : tangent_to_circle_M A₁ A₂) 
  (h₅ : tangent_to_circle_M A₁ A₃) :
  tangent_to_circle_M A₂ A₃ :=
sorry

end parabola_circle_tangency_l3597_359790


namespace jellybean_ratio_l3597_359732

/-- Given the number of jellybeans for Tino, Lee, and Arnold, prove the ratio of Arnold's to Lee's jellybeans --/
theorem jellybean_ratio 
  (tino lee arnold : ℕ) 
  (h1 : tino = lee + 24) 
  (h2 : tino = 34) 
  (h3 : arnold = 5) : 
  arnold / lee = 1 / 2 := by
  sorry

end jellybean_ratio_l3597_359732


namespace corridor_width_l3597_359760

theorem corridor_width 
  (w : ℝ) -- width of corridor
  (a : ℝ) -- length of ladder
  (h k : ℝ) -- heights on walls
  (h_pos : h > 0)
  (k_pos : k > 0)
  (a_pos : a > 0)
  (w_pos : w > 0)
  (angle_h : Real.cos (70 * π / 180) = h / a)
  (angle_k : Real.cos (60 * π / 180) = k / a)
  : w = h * Real.tan ((π / 2) - (70 * π / 180)) + k * Real.tan ((π / 2) - (60 * π / 180)) :=
by
  sorry


end corridor_width_l3597_359760


namespace polynomial_division_remainder_l3597_359792

theorem polynomial_division_remainder : ∃ q r : Polynomial ℝ, 
  (X^3 - 2 : Polynomial ℝ) = (X^2 - 2) * q + r ∧ 
  r.degree < (X^2 - 2).degree ∧ 
  r = 2*X - 2 := by
  sorry

end polynomial_division_remainder_l3597_359792
