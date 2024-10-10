import Mathlib

namespace largest_number_in_ratio_l3088_308876

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  b / a = 4 / 3 →
  c / a = 2 →
  a * b * c = 1944 →
  max a (max b c) = 18 := by
sorry

end largest_number_in_ratio_l3088_308876


namespace red_stamp_price_l3088_308822

theorem red_stamp_price 
  (red_count blue_count yellow_count : ℕ)
  (blue_price yellow_price : ℚ)
  (total_earnings : ℚ) :
  red_count = 20 →
  blue_count = 80 →
  yellow_count = 7 →
  blue_price = 4/5 →
  yellow_price = 2 →
  total_earnings = 100 →
  (red_count : ℚ) * (total_earnings - blue_count * blue_price - yellow_count * yellow_price) / red_count = 11/10 :=
by sorry

end red_stamp_price_l3088_308822


namespace min_distance_on_parabola_l3088_308802

/-- The minimum distance between two points on y = 2x² where the line
    connecting them is perpendicular to the tangent at one point -/
theorem min_distance_on_parabola :
  let f (x : ℝ) := 2 * x^2
  let tangent_slope (a : ℝ) := 4 * a
  let perpendicular_slope (a : ℝ) := -1 / (tangent_slope a)
  let distance (a : ℝ) := 
    let t := 4 * a^2
    Real.sqrt ((1 / (64 * t^2)) + (1 / (2 * t)) + t + 9/4)
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 3 / 4 ∧
    ∀ (a : ℝ), distance a ≥ min_dist :=
by sorry

end min_distance_on_parabola_l3088_308802


namespace geometric_mean_minimum_l3088_308826

theorem geometric_mean_minimum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_gm : Real.sqrt (a * b) = 2) :
  5 ≤ (b + 1/a) + (a + 1/b) ∧ 
  (∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ Real.sqrt (a₀ * b₀) = 2 ∧ (b₀ + 1/a₀) + (a₀ + 1/b₀) = 5) :=
sorry

end geometric_mean_minimum_l3088_308826


namespace triangle_area_in_circle_l3088_308807

theorem triangle_area_in_circle (r : ℝ) : 
  r > 0 → 
  let a := 5 * (10 / 13)
  let b := 12 * (10 / 13)
  let c := 13 * (10 / 13)
  a^2 + b^2 = c^2 → -- Pythagorean theorem
  c = 2 * r → -- diameter of the circle
  (1/2) * a * b = 3000/169 := by
sorry

end triangle_area_in_circle_l3088_308807


namespace intersection_complement_equality_l3088_308851

open Set

noncomputable def A : Set ℝ := {x | x ≥ -1}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

theorem intersection_complement_equality : A ∩ (univ \ B) = Icc (-1) 2 := by
  sorry

end intersection_complement_equality_l3088_308851


namespace intersection_of_sets_l3088_308896

/-- Given sets A and B, prove their intersection -/
theorem intersection_of_sets :
  let A : Set ℝ := {x | x < 1}
  let B : Set ℝ := {x | x^2 - x - 6 < 0}
  A ∩ B = {x | -2 < x ∧ x < 1} :=
by sorry

end intersection_of_sets_l3088_308896


namespace twelve_digit_numbers_with_consecutive_ones_l3088_308845

def fibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def valid_numbers (n : ℕ) : ℕ := 2^n

theorem twelve_digit_numbers_with_consecutive_ones : 
  (valid_numbers 12) - (fibonacci 11) = 3719 := by sorry

end twelve_digit_numbers_with_consecutive_ones_l3088_308845


namespace divisibility_property_l3088_308887

theorem divisibility_property (p : ℕ) (h1 : Even p) (h2 : p > 2) :
  ∃ k : ℤ, (p + 1) ^ (p / 2) - 1 = k * p := by
  sorry

end divisibility_property_l3088_308887


namespace scientific_notation_of_120_million_l3088_308803

theorem scientific_notation_of_120_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 120000000 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 7 :=
sorry

end scientific_notation_of_120_million_l3088_308803


namespace fish_in_tank_l3088_308893

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total →   -- One third of the fish are blue
  2 * spotted = blue → -- Half of the blue fish have spots
  spotted = 10 →       -- There are 10 blue, spotted fish
  total = 60 :=        -- Prove that the total number of fish is 60
by sorry

end fish_in_tank_l3088_308893


namespace reciprocal_of_2023_l3088_308890

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 := by
  sorry

end reciprocal_of_2023_l3088_308890


namespace mileage_difference_l3088_308839

/-- Calculates the difference between advertised and actual mileage -/
theorem mileage_difference (advertised_mpg : ℝ) (tank_capacity : ℝ) (total_miles : ℝ) :
  advertised_mpg = 35 →
  tank_capacity = 12 →
  total_miles = 372 →
  advertised_mpg - (total_miles / tank_capacity) = 4 := by
  sorry


end mileage_difference_l3088_308839


namespace students_walking_home_fraction_l3088_308838

theorem students_walking_home_fraction (total : ℚ) 
  (bus_fraction : ℚ) (auto_fraction : ℚ) (bike_fraction : ℚ) (scooter_fraction : ℚ) :
  bus_fraction = 1/3 →
  auto_fraction = 1/5 →
  bike_fraction = 1/8 →
  scooter_fraction = 1/15 →
  total = 1 →
  total - (bus_fraction + auto_fraction + bike_fraction + scooter_fraction) = 33/120 := by
  sorry

end students_walking_home_fraction_l3088_308838


namespace fly_distance_l3088_308865

/-- Prove that the distance traveled by a fly between two approaching cyclists is 50 km -/
theorem fly_distance (initial_distance : ℝ) (cyclist1_speed cyclist2_speed fly_speed : ℝ) :
  initial_distance = 50 →
  cyclist1_speed = 40 →
  cyclist2_speed = 60 →
  fly_speed = 100 →
  let relative_speed := cyclist1_speed + cyclist2_speed
  let time := initial_distance / relative_speed
  fly_speed * time = 50 := by sorry

end fly_distance_l3088_308865


namespace bobbys_shoe_cost_l3088_308815

/-- The total cost for Bobby's handmade shoes -/
def total_cost (mold_cost hourly_rate hours_worked discount_percentage : ℚ) : ℚ :=
  mold_cost + (hourly_rate * hours_worked) * (1 - discount_percentage)

/-- Theorem stating that Bobby's total cost for handmade shoes is $730 -/
theorem bobbys_shoe_cost :
  total_cost 250 75 8 0.2 = 730 := by
  sorry

end bobbys_shoe_cost_l3088_308815


namespace sin_2alpha_value_l3088_308857

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = 1/5) : 
  Real.sin (2 * α) = -24/25 := by
  sorry

end sin_2alpha_value_l3088_308857


namespace exists_valid_coloring_l3088_308880

/-- Represents a coloring of an infinite grid -/
def GridColoring := ℤ → ℤ → Bool

/-- Represents a move of the (m, n)-condylure -/
structure CondylureMove (m n : ℕ+) where
  horizontal : ℤ
  vertical : ℤ
  move_valid : (horizontal.natAbs = m ∧ vertical = 0) ∨ (horizontal = 0 ∧ vertical.natAbs = n)

/-- Theorem stating that for any positive m and n, there exists a grid coloring
    such that the (m, n)-condylure always lands on a different colored cell -/
theorem exists_valid_coloring (m n : ℕ+) :
  ∃ (coloring : GridColoring),
    ∀ (x y : ℤ) (move : CondylureMove m n),
      coloring (x + move.horizontal) (y + move.vertical) ≠ coloring x y :=
sorry

end exists_valid_coloring_l3088_308880


namespace smallest_n_for_252_terms_l3088_308864

def count_terms (n : ℕ) : ℕ := Nat.choose n 5

theorem smallest_n_for_252_terms : 
  (∀ k < 10, count_terms k ≠ 252) ∧ count_terms 10 = 252 := by sorry

end smallest_n_for_252_terms_l3088_308864


namespace earth_capacity_theorem_l3088_308855

/-- Represents the Earth's resource capacity --/
structure EarthCapacity where
  peopleA : ℕ  -- Number of people in scenario A
  yearsA : ℕ   -- Number of years in scenario A
  peopleB : ℕ  -- Number of people in scenario B
  yearsB : ℕ   -- Number of years in scenario B

/-- Calculates the maximum sustainable population given Earth's resource capacity --/
def maxSustainablePopulation (capacity : EarthCapacity) : ℕ :=
  ((capacity.peopleB * capacity.yearsB - capacity.peopleA * capacity.yearsA) / (capacity.yearsB - capacity.yearsA))

/-- Theorem stating the maximum sustainable population for given conditions --/
theorem earth_capacity_theorem (capacity : EarthCapacity) 
  (h1 : capacity.peopleA = 11)
  (h2 : capacity.yearsA = 90)
  (h3 : capacity.peopleB = 9)
  (h4 : capacity.yearsB = 210) :
  maxSustainablePopulation capacity = 75 := by
  sorry

end earth_capacity_theorem_l3088_308855


namespace tangent_line_through_origin_l3088_308818

/-- The function f(x) = x³ + x - 16 -/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_through_origin (x₀ : ℝ) :
  (f' x₀ = 13 ∧ f x₀ = -f' x₀ * x₀) →
  (x₀ = -2 ∧ f x₀ = -26 ∧ ∀ x, f' x₀ * x = f' x₀ * x₀ + f x₀) :=
sorry

end tangent_line_through_origin_l3088_308818


namespace apple_slices_equality_l3088_308849

/-- Represents the number of slices in an apple -/
structure Apple :=
  (slices : ℕ)

/-- Represents the amount of apple eaten -/
def eaten (a : Apple) (s : ℕ) : ℚ :=
  s / a.slices

theorem apple_slices_equality (yeongchan minhyuk : Apple) 
  (h1 : yeongchan.slices = 3)
  (h2 : minhyuk.slices = 12) :
  eaten yeongchan 1 = eaten minhyuk 4 :=
by sorry

end apple_slices_equality_l3088_308849


namespace sams_dimes_l3088_308824

/-- Sam's dimes problem -/
theorem sams_dimes (initial_dimes given_away_dimes : ℕ) 
  (h1 : initial_dimes = 9)
  (h2 : given_away_dimes = 7) :
  initial_dimes - given_away_dimes = 2 := by
  sorry

end sams_dimes_l3088_308824


namespace triangle_ratio_l3088_308808

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C →
  Real.cos A = -1/4 →
  b / c = 6 := by
  sorry

end triangle_ratio_l3088_308808


namespace slightly_used_crayons_l3088_308828

/-- Proves that the number of slightly used crayons is 56 -/
theorem slightly_used_crayons (total : ℕ) (new : ℕ) (broken : ℕ) (slightly_used : ℕ) : 
  total = 120 →
  new = total / 3 →
  broken = total / 5 →
  slightly_used = total - new - broken →
  slightly_used = 56 := by
  sorry

end slightly_used_crayons_l3088_308828


namespace translation_right_2_units_l3088_308805

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the right by a given distance -/
def translateRight (p : Point) (d : ℝ) : Point :=
  { x := p.x + d, y := p.y }

theorem translation_right_2_units :
  let A : Point := { x := 1, y := 2 }
  let A' : Point := translateRight A 2
  A'.x = 3 ∧ A'.y = 2 := by
  sorry

end translation_right_2_units_l3088_308805


namespace triangle_inequality_l3088_308862

theorem triangle_inequality (a b c R r : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R = (a * b * c) / (4 * area))
  (h_inradius : r = (2 * area) / (a + b + c))
  (h_area : area = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c) / 16)) :
  (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) :=
sorry

end triangle_inequality_l3088_308862


namespace odd_prime_square_root_l3088_308891

theorem odd_prime_square_root (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) 
  (h_pos : k > 0) (h_sqrt : ∃ n : ℕ, n > 0 ∧ n * n = k * k - p * k) :
  k = (p + 1)^2 / 4 := by
  sorry

end odd_prime_square_root_l3088_308891


namespace expression_simplification_l3088_308879

theorem expression_simplification :
  (Real.sqrt 2 * 2^(1/2 : ℝ) * 2) + (18 / 3 * 2) - (8^(1/2 : ℝ) * 4) = 16 - 8 * Real.sqrt 2 := by
  sorry

end expression_simplification_l3088_308879


namespace nonnegative_rational_function_l3088_308894

theorem nonnegative_rational_function (x : ℝ) :
  (x^2 - 6*x + 9) / (9 - x^3) ≥ 0 ↔ x ≤ 3 := by sorry

end nonnegative_rational_function_l3088_308894


namespace sqrt_inequality_and_fraction_bound_l3088_308810

theorem sqrt_inequality_and_fraction_bound : 
  (Real.sqrt 5 + Real.sqrt 7 > 1 + Real.sqrt 13) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y > 1 → 
    min ((1 + x) / y) ((1 + y) / x) < 3) := by
  sorry

end sqrt_inequality_and_fraction_bound_l3088_308810


namespace no_positive_integer_sequence_exists_positive_irrational_sequence_l3088_308885

-- Part 1: No sequence of positive integers satisfying the condition
theorem no_positive_integer_sequence :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2)) := by sorry

-- Part 2: Existence of a sequence of positive irrational numbers satisfying the condition
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, Irrational (a n) ∧ a n > 0) ∧
    (∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) := by sorry

end no_positive_integer_sequence_exists_positive_irrational_sequence_l3088_308885


namespace monic_polynomial_property_l3088_308854

def is_monic_polynomial_with_properties (p : ℝ → ℝ) : Prop :=
  (∀ x, ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆, p x = x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) ∧
  (∀ i : Fin 8, p i = i)

theorem monic_polynomial_property (p : ℝ → ℝ) 
  (h : is_monic_polynomial_with_properties p) : p 8 = 40328 := by
  sorry

end monic_polynomial_property_l3088_308854


namespace max_area_rectangular_pen_l3088_308846

/-- Given 60 feet of fencing for a rectangular pen where the length is exactly twice the width,
    the maximum possible area is 200 square feet. -/
theorem max_area_rectangular_pen (perimeter : ℝ) (width : ℝ) (length : ℝ) (area : ℝ) :
  perimeter = 60 →
  length = 2 * width →
  perimeter = 2 * length + 2 * width →
  area = length * width →
  area ≤ 200 ∧ ∃ w l, width = w ∧ length = l ∧ area = 200 :=
by sorry

end max_area_rectangular_pen_l3088_308846


namespace factorization_equality_l3088_308875

theorem factorization_equality (x y : ℝ) :
  x * (x - y) + y * (y - x) = (x - y)^2 := by
  sorry

end factorization_equality_l3088_308875


namespace pencil_rows_l3088_308801

theorem pencil_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 154) (h2 : pencils_per_row = 11) :
  total_pencils / pencils_per_row = 14 := by
sorry

end pencil_rows_l3088_308801


namespace birth_date_satisfies_conditions_l3088_308866

/-- Represents a date with year, month, and day components -/
structure Date where
  year : ℕ
  month : ℕ
  day : ℕ

/-- Calculates the age of a person at a given year, given their birth date -/
def age (birthDate : Date) (currentYear : ℕ) : ℕ :=
  currentYear - birthDate.year

/-- Represents the problem conditions -/
def satisfiesConditions (birthDate : Date) : Prop :=
  let ageIn1937 := age birthDate 1937
  ageIn1937 * ageIn1937 = 1937 - birthDate.year ∧ 
  ageIn1937 + birthDate.month = birthDate.day * birthDate.day

/-- The main theorem to prove -/
theorem birth_date_satisfies_conditions : 
  satisfiesConditions (Date.mk 1892 5 7) :=
sorry

end birth_date_satisfies_conditions_l3088_308866


namespace at_least_one_geq_quarter_l3088_308806

theorem at_least_one_geq_quarter (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (h_eq : x * y * z = (1 - x) * (1 - y) * (1 - z)) : 
  (1 - x) * y ≥ 1/4 ∨ (1 - y) * z ≥ 1/4 ∨ (1 - z) * x ≥ 1/4 :=
by sorry

end at_least_one_geq_quarter_l3088_308806


namespace position_of_2015_l3088_308872

/-- Represents a digit in the base-6 number system -/
inductive Digit : Type
| zero : Digit
| one : Digit
| two : Digit
| three : Digit
| four : Digit
| five : Digit

/-- Converts a base-6 number to its decimal equivalent -/
def toDecimal (n : List Digit) : Nat :=
  sorry

/-- Checks if a number is representable in base-6 using digits 0-5 -/
def isValidBase6 (n : Nat) : Prop :=
  sorry

/-- The sequence of numbers formed by digits 0-5 in ascending order -/
def base6Sequence : List Nat :=
  sorry

/-- The position of a number in the base6Sequence -/
def positionInSequence (n : Nat) : Nat :=
  sorry

/-- Theorem: The position of 2015 in the base-6 sequence is 443 -/
theorem position_of_2015 : positionInSequence 2015 = 443 :=
  sorry

end position_of_2015_l3088_308872


namespace min_a_value_l3088_308888

theorem min_a_value (x y : ℝ) (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 4 5) :
  ∃ (a : ℝ), (∀ (x' y' : ℝ), x' ∈ Set.Icc 1 2 → y' ∈ Set.Icc 4 5 → x' * y' ≤ a * x' ^ 2 + 2 * y' ^ 2) ∧ 
  (∀ (b : ℝ), (∀ (x' y' : ℝ), x' ∈ Set.Icc 1 2 → y' ∈ Set.Icc 4 5 → x' * y' ≤ b * x' ^ 2 + 2 * y' ^ 2) → b ≥ -6) :=
by
  sorry

#check min_a_value

end min_a_value_l3088_308888


namespace first_quarter_spending_river_town_l3088_308835

/-- The spending during the first quarter of a year, given the initial and end-of-quarter spending -/
def first_quarter_spending (initial_spending end_of_quarter_spending : ℝ) : ℝ :=
  end_of_quarter_spending - initial_spending

/-- Theorem: The spending during the first quarter is 3.1 million dollars -/
theorem first_quarter_spending_river_town : 
  first_quarter_spending 0 3.1 = 3.1 := by
  sorry

end first_quarter_spending_river_town_l3088_308835


namespace sphere_surface_area_of_circumscribed_cube_l3088_308867

theorem sphere_surface_area_of_circumscribed_cube (edge_length : ℝ) 
  (h : edge_length = 2 * Real.sqrt 3) :
  let diagonal := Real.sqrt 3 * edge_length
  let radius := diagonal / 2
  4 * Real.pi * radius ^ 2 = 36 * Real.pi := by
  sorry

end sphere_surface_area_of_circumscribed_cube_l3088_308867


namespace parabola_vector_max_value_l3088_308840

/-- The parabola C: x^2 = 4y -/
def parabola (p : ℝ × ℝ) : Prop := p.1^2 = 4 * p.2

/-- The line l intersecting the parabola at points A and B -/
def line_intersects (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ l ∧ B ∈ l ∧ parabola A ∧ parabola B

/-- Vector from origin to a point -/
def vec_from_origin (p : ℝ × ℝ) : ℝ × ℝ := p

/-- Vector between two points -/
def vec_between (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Scalar multiplication of a vector -/
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

/-- Vector equality -/
def vec_eq (v w : ℝ × ℝ) : Prop := v = w

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem parabola_vector_max_value 
  (l : Set (ℝ × ℝ)) (A B G : ℝ × ℝ) :
  line_intersects l A B →
  vec_eq (vec_between A B) (scalar_mul 2 (vec_between A G)) →
  (∃ (max : ℝ), 
    max = 16 ∧ 
    ∀ (X Y : ℝ × ℝ), parabola X → parabola Y → 
      (dot_product (vec_from_origin X) (vec_from_origin X) +
       dot_product (vec_from_origin Y) (vec_from_origin Y) -
       2 * dot_product (vec_from_origin X) (vec_from_origin Y) -
       4 * dot_product (vec_from_origin G) (vec_from_origin G)) ≤ max) :=
sorry

end parabola_vector_max_value_l3088_308840


namespace unique_number_equality_l3088_308844

theorem unique_number_equality : ∃! x : ℝ, 4 * x - 3 = 9 * (x - 7) := by
  sorry

end unique_number_equality_l3088_308844


namespace sequence_integer_count_l3088_308809

def sequence_term (n : ℕ) : ℚ :=
  9720 / 3^n

def is_integer (q : ℚ) : Prop :=
  ∃ z : ℤ, q = z

theorem sequence_integer_count :
  (∃ k : ℕ, ∀ n : ℕ, is_integer (sequence_term n) ↔ n ≤ k) ∧
  (∀ k : ℕ, (∀ n : ℕ, is_integer (sequence_term n) ↔ n ≤ k) → k = 5) :=
sorry

end sequence_integer_count_l3088_308809


namespace book_pages_calculation_l3088_308819

theorem book_pages_calculation (pages_per_day : ℕ) (days_read : ℕ) (fraction_read : ℚ) : 
  pages_per_day = 12 →
  days_read = 15 →
  fraction_read = 3/4 →
  (pages_per_day * days_read : ℚ) / fraction_read = 240 := by
sorry

end book_pages_calculation_l3088_308819


namespace marbles_from_henry_l3088_308831

theorem marbles_from_henry (initial_marbles end_marbles marbles_from_henry : ℕ) 
  (h1 : initial_marbles = 95)
  (h2 : end_marbles = 104)
  (h3 : end_marbles = initial_marbles + marbles_from_henry) :
  marbles_from_henry = 9 := by
  sorry

end marbles_from_henry_l3088_308831


namespace rectangleB_is_leftmost_l3088_308874

-- Define a structure for rectangles
structure Rectangle where
  name : Char
  w : Int
  x : Int
  y : Int
  z : Int

-- Define the five rectangles
def rectangleA : Rectangle := ⟨'A', 5, 2, 8, 10⟩
def rectangleB : Rectangle := ⟨'B', 2, 1, 6, 9⟩
def rectangleC : Rectangle := ⟨'C', 4, 7, 3, 0⟩
def rectangleD : Rectangle := ⟨'D', 9, 6, 5, 11⟩
def rectangleE : Rectangle := ⟨'E', 10, 4, 7, 2⟩

-- Define a list of all rectangles
def allRectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Define a function to check if a rectangle is leftmost
def isLeftmost (r : Rectangle) (rectangles : List Rectangle) : Prop :=
  ∀ other ∈ rectangles, r.w ≤ other.w

-- Theorem statement
theorem rectangleB_is_leftmost :
  isLeftmost rectangleB allRectangles :=
sorry

end rectangleB_is_leftmost_l3088_308874


namespace bcm_hens_count_l3088_308811

/-- Given a farm with chickens, calculate the number of Black Copper Marans (BCM) hens -/
theorem bcm_hens_count (total_chickens : ℕ) (bcm_percentage : ℚ) (bcm_hen_percentage : ℚ) : 
  total_chickens = 100 →
  bcm_percentage = 1/5 →
  bcm_hen_percentage = 4/5 →
  ↑(total_chickens : ℚ) * bcm_percentage * bcm_hen_percentage = 16 := by
  sorry

end bcm_hens_count_l3088_308811


namespace mike_lawn_mowing_earnings_l3088_308860

def mower_blade_cost : ℕ := 24
def game_cost : ℕ := 5
def num_games : ℕ := 9

theorem mike_lawn_mowing_earnings :
  ∃ (total_earnings : ℕ),
    total_earnings = mower_blade_cost + (game_cost * num_games) :=
by
  sorry

end mike_lawn_mowing_earnings_l3088_308860


namespace translation_result_l3088_308834

/-- Represents a point in the 2D Cartesian coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point downward by a given number of units -/
def translateDown (p : Point) (units : ℝ) : Point :=
  { x := p.x, y := p.y - units }

/-- Translates a point to the right by a given number of units -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The main theorem stating the result of the translation -/
theorem translation_result : 
  let A : Point := { x := -2, y := 2 }
  let B : Point := translateRight (translateDown A 4) 3
  B.x = 1 ∧ B.y = -2 := by sorry

end translation_result_l3088_308834


namespace max_value_of_f_l3088_308897

theorem max_value_of_f (x : ℝ) : 
  x / (x^2 + 9) + 1 / (x^2 - 6*x + 21) + Real.cos (2 * Real.pi * x) ≤ 1.25 := by
  sorry

end max_value_of_f_l3088_308897


namespace league_games_count_l3088_308853

/-- The number of unique games played in a league season --/
def uniqueGamesInSeason (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 30 teams, where each team plays 15 games against every other team,
    the total number of unique games played in the season is 6,525. --/
theorem league_games_count :
  uniqueGamesInSeason 30 15 = 6525 := by
  sorry

#eval uniqueGamesInSeason 30 15

end league_games_count_l3088_308853


namespace line_circle_intersection_l3088_308837

/-- The line equation y = √3 * x + m -/
def line_equation (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + m

/-- The circle equation x^2 + (y - 3)^2 = 6 -/
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 6

/-- Two points A and B on both the line and the circle -/
def intersection_points (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  line_equation A.1 A.2 m ∧ circle_equation A.1 A.2 ∧
  line_equation B.1 B.2 m ∧ circle_equation B.1 B.2

/-- The distance between points A and B is 2√2 -/
def distance_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8

theorem line_circle_intersection (m : ℝ) :
  (∃ A B : ℝ × ℝ, intersection_points A B m ∧ distance_condition A B) →
  m = -1 ∨ m = 7 := by sorry

end line_circle_intersection_l3088_308837


namespace inequality_proof_l3088_308884

theorem inequality_proof (x y : ℝ) : x^4 + y^4 + 8 ≥ 8*x*y := by
  sorry

end inequality_proof_l3088_308884


namespace intersection_is_empty_l3088_308882

open Set

def A : Set ℝ := Ioc (-1) 3
def B : Set ℝ := {2, 4}

theorem intersection_is_empty : A ∩ B = ∅ := by sorry

end intersection_is_empty_l3088_308882


namespace lisa_hourly_wage_l3088_308886

/-- Calculates the hourly wage of Lisa given Greta's work hours, hourly rate, and Lisa's equivalent work hours -/
theorem lisa_hourly_wage (greta_hours : ℕ) (greta_rate : ℚ) (lisa_hours : ℕ) : 
  greta_hours = 40 → 
  greta_rate = 12 → 
  lisa_hours = 32 → 
  (greta_hours * greta_rate) / lisa_hours = 15 := by
sorry

end lisa_hourly_wage_l3088_308886


namespace courtyard_paving_cost_l3088_308871

/-- Calculates the cost of paving a rectangular courtyard -/
theorem courtyard_paving_cost 
  (ratio_long : ℝ) 
  (ratio_short : ℝ) 
  (diagonal : ℝ) 
  (cost_per_sqm : ℝ) 
  (h_ratio : ratio_long / ratio_short = 4 / 3) 
  (h_diagonal : diagonal = 45) 
  (h_cost : cost_per_sqm = 0.5) : 
  ⌊(ratio_long * ratio_short * (diagonal^2 / (ratio_long^2 + ratio_short^2)) * cost_per_sqm * 100) / 100⌋ = 486 := by
sorry

end courtyard_paving_cost_l3088_308871


namespace expression_evaluation_l3088_308821

theorem expression_evaluation : 8 - 5 * (9 - (4 - 2)^2) * 2 = -42 := by
  sorry

end expression_evaluation_l3088_308821


namespace no_super_squarish_numbers_l3088_308841

-- Define a super-squarish number
def is_super_squarish (n : ℕ) : Prop :=
  -- Seven-digit number
  1000000 ≤ n ∧ n < 10000000 ∧
  -- No digit is zero
  ∀ d, (n / 10^d) % 10 ≠ 0 ∧
  -- Perfect square
  ∃ y, n = y^2 ∧
  -- First two digits are a perfect square
  ∃ a, (n / 100000)^2 = a ∧
  -- Next three digits are a perfect square
  ∃ b, ((n / 1000) % 1000)^2 = b ∧
  -- Last two digits are a perfect square
  ∃ c, (n % 100)^2 = c

-- Theorem statement
theorem no_super_squarish_numbers : ¬∃ n : ℕ, is_super_squarish n := by
  sorry

end no_super_squarish_numbers_l3088_308841


namespace shaded_fraction_of_square_l3088_308848

theorem shaded_fraction_of_square (total_squares : ℕ) (split_squares : ℕ) (triangle_area_fraction : ℚ) :
  total_squares = 16 →
  split_squares = 4 →
  triangle_area_fraction = 1/2 →
  (split_squares : ℚ) * triangle_area_fraction / total_squares = 1/8 :=
by sorry

end shaded_fraction_of_square_l3088_308848


namespace unit_vectors_equal_squared_magnitude_l3088_308836

/-- Two unit vectors have equal squared magnitudes -/
theorem unit_vectors_equal_squared_magnitude
  {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]
  (a b : n) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  ‖a‖^2 = ‖b‖^2 := by sorry

end unit_vectors_equal_squared_magnitude_l3088_308836


namespace tan_585_degrees_l3088_308823

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end tan_585_degrees_l3088_308823


namespace tangent_ratio_l3088_308829

/-- A cube with an inscribed sphere -/
structure CubeWithSphere where
  edge_length : ℝ
  sphere_radius : ℝ
  /- The sphere radius is half the edge length -/
  sphere_radius_eq : sphere_radius = edge_length / 2

/-- A point on the edge of a cube -/
structure EdgePoint (c : CubeWithSphere) where
  x : ℝ
  y : ℝ
  z : ℝ
  /- The point is on an edge -/
  on_edge : (x = 0 ∧ y = 0) ∨ (x = 0 ∧ z = 0) ∨ (y = 0 ∧ z = 0)

/-- A point on the inscribed sphere -/
structure SpherePoint (c : CubeWithSphere) where
  x : ℝ
  y : ℝ
  z : ℝ
  /- The point is on the sphere -/
  on_sphere : x^2 + y^2 + z^2 = c.sphere_radius^2

/-- Theorem: The ratio KE:EF is 4:5 -/
theorem tangent_ratio 
  (c : CubeWithSphere) 
  (K : EdgePoint c) 
  (E : SpherePoint c) 
  (F : EdgePoint c) 
  (h_K_midpoint : K.x = c.edge_length / 2 ∨ K.y = c.edge_length / 2 ∨ K.z = c.edge_length / 2)
  (h_tangent : ∃ t : ℝ, K.x + t * (E.x - K.x) = F.x ∧ 
                        K.y + t * (E.y - K.y) = F.y ∧ 
                        K.z + t * (E.z - K.z) = F.z)
  (h_skew : (F.x ≠ K.x ∨ F.y ≠ K.y) ∧ (F.y ≠ K.y ∨ F.z ≠ K.z) ∧ (F.x ≠ K.x ∨ F.z ≠ K.z)) :
  ∃ (ke ef : ℝ), ke / ef = 4 / 5 ∧ 
    ke^2 = (E.x - K.x)^2 + (E.y - K.y)^2 + (E.z - K.z)^2 ∧
    ef^2 = (F.x - E.x)^2 + (F.y - E.y)^2 + (F.z - E.z)^2 :=
by sorry

end tangent_ratio_l3088_308829


namespace unique_four_digit_number_l3088_308895

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ),
    n = 1000 * a + 100 * b + 10 * c + d ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

def reverse_number (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  1000 * d + 100 * c + 10 * b + a

theorem unique_four_digit_number :
  ∀ n : ℕ, is_valid_number n → (reverse_number n = n + 7182) → n = 1909 :=
sorry

end unique_four_digit_number_l3088_308895


namespace investment_average_rate_l3088_308856

/-- Proves that given a total investment split between two rates with equal returns, the average rate is as expected -/
theorem investment_average_rate 
  (total_investment : ℝ) 
  (rate1 rate2 : ℝ) 
  (h_total : total_investment = 5000)
  (h_rates : rate1 = 0.05 ∧ rate2 = 0.03)
  (h_equal_returns : ∃ (x : ℝ), x * rate1 = (total_investment - x) * rate2)
  : (((rate1 * (total_investment * rate1 / (rate1 + rate2))) + 
     (rate2 * (total_investment * rate2 / (rate1 + rate2)))) / total_investment) = 0.0375 :=
sorry

end investment_average_rate_l3088_308856


namespace arithmetic_mean_sqrt2_l3088_308827

theorem arithmetic_mean_sqrt2 (a b : ℝ) : 
  a = 1 / (Real.sqrt 2 + 1) → 
  b = 1 / (Real.sqrt 2 - 1) → 
  (a + b) / 2 = Real.sqrt 2 := by sorry

end arithmetic_mean_sqrt2_l3088_308827


namespace sum_of_repeating_decimals_l3088_308813

/-- Represents a repeating decimal with a single repeating digit -/
def repeating_decimal_single (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two repeating digits -/
def repeating_decimal_double (n : ℕ) : ℚ := n / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_single 6 + repeating_decimal_double 45 = 37 / 33 := by sorry

end sum_of_repeating_decimals_l3088_308813


namespace triangle_inequality_l3088_308858

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) : 
  3 * (a * b + a * c + b * c) ≤ (a + b + c)^2 ∧ (a + b + c)^2 < 4 * (a * b + a * c + b * c) := by
  sorry

end triangle_inequality_l3088_308858


namespace fence_cost_square_plot_l3088_308847

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) :
  area = 289 → price_per_foot = 57 →
  4 * Real.sqrt area * price_per_foot = 3876 := by
  sorry

end fence_cost_square_plot_l3088_308847


namespace painting_price_theorem_l3088_308850

theorem painting_price_theorem (total_cost : ℕ) (price : ℕ) (quantity : ℕ) :
  total_cost = 104 →
  price > 0 →
  quantity * price = total_cost →
  10 < quantity →
  quantity < 60 →
  (price = 2 ∨ price = 4 ∨ price = 8) :=
by sorry

end painting_price_theorem_l3088_308850


namespace tony_water_consumption_l3088_308859

/-- Calculates the daily water consumption given the bottle capacity, number of refills per week, and days in a week. -/
def daily_water_consumption (bottle_capacity : ℕ) (refills_per_week : ℕ) (days_in_week : ℕ) : ℚ :=
  (bottle_capacity * refills_per_week : ℚ) / days_in_week

/-- Proves that given a water bottle capacity of 84 ounces, filled 6 times per week, 
    and 7 days in a week, the daily water consumption is 72 ounces. -/
theorem tony_water_consumption :
  daily_water_consumption 84 6 7 = 72 := by
  sorry

end tony_water_consumption_l3088_308859


namespace total_ants_is_twenty_l3088_308873

/-- The number of ants found by Abe -/
def abe_ants : ℕ := 4

/-- The number of ants found by Beth -/
def beth_ants : ℕ := abe_ants + abe_ants / 2

/-- The number of ants found by CeCe -/
def cece_ants : ℕ := 2 * abe_ants

/-- The number of ants found by Duke -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_is_twenty : total_ants = 20 := by
  sorry

end total_ants_is_twenty_l3088_308873


namespace quadratic_point_value_l3088_308817

/-- Given a quadratic function y = -ax^2 + 2ax + 3 where a > 0,
    if the point P(m, 3) lies on the graph and m ≠ 0, then m = 2. -/
theorem quadratic_point_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 →
  3 = -a * m^2 + 2 * a * m + 3 →
  m = 2 := by
sorry

end quadratic_point_value_l3088_308817


namespace fred_marbles_l3088_308804

theorem fred_marbles (total : ℕ) (dark_blue : ℕ) (green : ℕ) (red : ℕ) :
  total = 63 →
  dark_blue ≥ total / 3 →
  green = 4 →
  total = dark_blue + green + red →
  red = 38 := by
sorry

end fred_marbles_l3088_308804


namespace intersection_of_A_and_B_l3088_308868

-- Define the sets A and B
def A : Set ℝ := {y | y > 1}
def B : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x > 1} := by sorry

end intersection_of_A_and_B_l3088_308868


namespace rectangular_field_area_l3088_308869

/-- The area of a rectangular field with one side of 4 meters and a diagonal of 5 meters is 12 square meters. -/
theorem rectangular_field_area : ∀ (w l : ℝ), 
  w = 4 → 
  w^2 + l^2 = 5^2 → 
  w * l = 12 := by
  sorry

end rectangular_field_area_l3088_308869


namespace solution_equivalence_l3088_308830

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (Real.sqrt 22, Real.sqrt 22), (-Real.sqrt 22, -Real.sqrt 22),
   (Real.sqrt 20, -Real.sqrt 20), (-Real.sqrt 20, Real.sqrt 20),
   (((-3 + Real.sqrt 5) / 2) * (2 * Real.sqrt (3 + Real.sqrt 5)), 2 * Real.sqrt (3 + Real.sqrt 5)),
   (((-3 + Real.sqrt 5) / 2) * (-2 * Real.sqrt (3 + Real.sqrt 5)), -2 * Real.sqrt (3 + Real.sqrt 5)),
   (((-3 - Real.sqrt 5) / 2) * (2 * Real.sqrt (3 - Real.sqrt 5)), 2 * Real.sqrt (3 - Real.sqrt 5)),
   (((-3 - Real.sqrt 5) / 2) * (-2 * Real.sqrt (3 - Real.sqrt 5)), -2 * Real.sqrt (3 - Real.sqrt 5))}

theorem solution_equivalence :
  {(x, y) : ℝ × ℝ | x^5 = 21*x^3 + y^3 ∧ y^5 = x^3 + 21*y^3} = solution_set :=
by sorry

end solution_equivalence_l3088_308830


namespace imaginary_part_of_complex_expression_l3088_308877

theorem imaginary_part_of_complex_expression :
  Complex.im ((2 * Complex.I) / (1 - Complex.I) + 2) = 1 := by sorry

end imaginary_part_of_complex_expression_l3088_308877


namespace orange_roses_count_l3088_308889

theorem orange_roses_count (red_roses : ℕ) (pink_roses : ℕ) (yellow_roses : ℕ) 
  (total_picked : ℕ) (h1 : red_roses = 12) (h2 : pink_roses = 18) 
  (h3 : yellow_roses = 20) (h4 : total_picked = 22) :
  ∃ (orange_roses : ℕ), 
    orange_roses = 8 ∧ 
    total_picked = red_roses / 2 + pink_roses / 2 + yellow_roses / 4 + orange_roses / 4 :=
by sorry

end orange_roses_count_l3088_308889


namespace cube_root_sum_equals_two_l3088_308800

theorem cube_root_sum_equals_two :
  (Real.rpow (7 + 3 * Real.sqrt 21) (1/3 : ℝ)) + (Real.rpow (7 - 3 * Real.sqrt 21) (1/3 : ℝ)) = 2 := by
  sorry

end cube_root_sum_equals_two_l3088_308800


namespace power_function_through_point_l3088_308842

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 = 2) : 
  f 2 = Real.sqrt 2 := by
  sorry

end power_function_through_point_l3088_308842


namespace sum_of_squares_l3088_308892

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 2) (h2 : x^3 + y^3 = 3) : x^2 + y^2 = 7/3 := by
  sorry

end sum_of_squares_l3088_308892


namespace susan_age_in_five_years_l3088_308816

/-- Represents the current year -/
def current_year : ℕ := 2023

/-- James' age in a given year -/
def james_age (year : ℕ) : ℕ := sorry

/-- Janet's age in a given year -/
def janet_age (year : ℕ) : ℕ := sorry

/-- Susan's age in a given year -/
def susan_age (year : ℕ) : ℕ := sorry

theorem susan_age_in_five_years :
  (∀ year : ℕ, james_age (year - 8) = 2 * janet_age (year - 8)) →
  james_age (current_year + 15) = 37 →
  (∀ year : ℕ, susan_age year = janet_age year - 3) →
  susan_age (current_year + 5) = 17 := by sorry

end susan_age_in_five_years_l3088_308816


namespace pen_purchase_problem_l3088_308833

theorem pen_purchase_problem :
  ∀ (x y : ℕ),
    1.7 * (x : ℝ) + 1.2 * (y : ℝ) = 15 →
    x = 6 ∧ y = 4 :=
by sorry

end pen_purchase_problem_l3088_308833


namespace fuel_refills_l3088_308898

theorem fuel_refills (total_spent : ℕ) (cost_per_refill : ℕ) (h1 : total_spent = 40) (h2 : cost_per_refill = 10) :
  total_spent / cost_per_refill = 4 := by
  sorry

end fuel_refills_l3088_308898


namespace shower_tiles_count_l3088_308852

/-- Represents a shower with three walls -/
structure Shower :=
  (width : Nat)  -- Number of tiles in width
  (height : Nat) -- Number of tiles in height

/-- Calculates the total number of tiles in a shower -/
def totalTiles (s : Shower) : Nat :=
  3 * s.width * s.height

/-- Theorem stating that a shower with 8 tiles in width and 20 in height has 480 tiles in total -/
theorem shower_tiles_count : 
  ∀ s : Shower, s.width = 8 → s.height = 20 → totalTiles s = 480 := by
  sorry

end shower_tiles_count_l3088_308852


namespace gcd_24_36_54_l3088_308814

theorem gcd_24_36_54 : Nat.gcd 24 (Nat.gcd 36 54) = 6 := by sorry

end gcd_24_36_54_l3088_308814


namespace exactly_three_primes_39p_plus_1_perfect_square_l3088_308899

theorem exactly_three_primes_39p_plus_1_perfect_square :
  ∃! (s : Finset Nat), 
    (∀ p ∈ s, Nat.Prime p ∧ ∃ n : Nat, 39 * p + 1 = n^2) ∧ 
    Finset.card s = 3 := by
  sorry

end exactly_three_primes_39p_plus_1_perfect_square_l3088_308899


namespace sum_and_reverse_contradiction_l3088_308883

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem sum_and_reverse_contradiction :
  let sum := 137 + 276
  sum = 413 ∧ reverse_digits sum ≠ 534 := by
  sorry

end sum_and_reverse_contradiction_l3088_308883


namespace inequality_solution_l3088_308825

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2 ↔ 
  (0 < x ∧ x ≤ 5/6) ∨ (2 < x) := by sorry

end inequality_solution_l3088_308825


namespace simplify_expression_l3088_308820

theorem simplify_expression (x y : ℝ) : (5 - 4*x) - (2 + 7*x - y) = 3 - 11*x + y := by
  sorry

end simplify_expression_l3088_308820


namespace two_x_plus_y_equals_five_l3088_308812

theorem two_x_plus_y_equals_five (x y : ℝ) 
  (eq1 : 7 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  2 * x + y = 5 := by
sorry

end two_x_plus_y_equals_five_l3088_308812


namespace cube_volume_decomposition_l3088_308861

theorem cube_volume_decomposition (x : ℝ) (hx : x > 0) :
  ∃ (y z : ℝ),
    y = (3/2 + Real.sqrt (3/2)) * x ∧
    z = (3/2 - Real.sqrt (3/2)) * x ∧
    y^3 + z^3 = x^3 := by
  sorry

end cube_volume_decomposition_l3088_308861


namespace tangent_lines_range_l3088_308878

/-- The range of k values for which two tangent lines exist from (1, 2) to the circle -/
theorem tangent_lines_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0) ∧ 
  ((1:ℝ)^2 + 2^2 + k*1 + 2*2 + k^2 - 15 > 0) ↔ 
  (k > 2 ∧ k < 8/3 * Real.sqrt 3) ∨ (k > -8/3 * Real.sqrt 3 ∧ k < -3) :=
sorry

end tangent_lines_range_l3088_308878


namespace negation_equivalence_l3088_308863

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5*x + 6 > 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) := by
  sorry

end negation_equivalence_l3088_308863


namespace total_feet_count_l3088_308881

theorem total_feet_count (total_heads : ℕ) (hen_count : ℕ) (hen_feet cow_feet : ℕ) : 
  total_heads = 48 → 
  hen_count = 26 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  (hen_count * hen_feet) + ((total_heads - hen_count) * cow_feet) = 140 := by
sorry

end total_feet_count_l3088_308881


namespace bicentric_shapes_l3088_308843

-- Define the property of being bicentric
def IsBicentric (shape : Type) : Prop :=
  ∃ (circumscribed inscribed : Type), 
    (∀ (s : shape), ∃ (c : circumscribed), True) ∧ 
    (∀ (s : shape), ∃ (i : inscribed), True)

-- Define the shapes
def Square : Type := Unit
def Rectangle : Type := Unit
def RegularPentagon : Type := Unit
def Hexagon : Type := Unit

-- State the theorem
theorem bicentric_shapes :
  IsBicentric Square ∧
  IsBicentric RegularPentagon ∧
  ¬(∀ (r : Rectangle), IsBicentric Rectangle) ∧
  ¬(∀ (h : Hexagon), IsBicentric Hexagon) :=
sorry

end bicentric_shapes_l3088_308843


namespace circle_radius_zero_l3088_308870

theorem circle_radius_zero (x y : ℝ) :
  25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end circle_radius_zero_l3088_308870


namespace algebraic_expression_evaluation_l3088_308832

theorem algebraic_expression_evaluation : 
  let x : ℚ := -1
  let y : ℚ := 1/2
  2 * (x^2 - 5*x*y) - 3 * (x^2 - 6*x*y) = 3 := by sorry

end algebraic_expression_evaluation_l3088_308832
