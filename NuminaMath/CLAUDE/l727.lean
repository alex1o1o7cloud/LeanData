import Mathlib

namespace NUMINAMATH_CALUDE_unique_positive_solution_l727_72731

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 2023*x^5 - 2021*x^4

-- Theorem statement
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l727_72731


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l727_72703

/-- Given a line L1 with equation x - 2y + 3 = 0 and a point P(1, -3),
    prove that the line L2 with equation 2x + y + 1 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => x - 2 * y + 3 = 0
  let L2 : ℝ → ℝ → Prop := λ x y => 2 * x + y + 1 = 0
  let P : ℝ × ℝ := (1, -3)
  (L2 P.1 P.2) ∧                           -- L2 passes through P
  (∀ (x1 y1 x2 y2 : ℝ),
    L1 x1 y1 → L1 x2 y2 → x1 ≠ x2 →        -- L1 and L2 are perpendicular
    L2 x1 y1 → L2 x2 y2 → 
    (x2 - x1) * ((x2 - x1) / (y2 - y1)) = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l727_72703


namespace NUMINAMATH_CALUDE_mary_remaining_stickers_l727_72749

/-- Calculates the number of remaining stickers for Mary --/
def remaining_stickers (initial : ℕ) (front_page : ℕ) (other_pages : ℕ) (stickers_per_page : ℕ) : ℕ :=
  initial - (front_page + other_pages * stickers_per_page)

/-- Proves that Mary has 44 stickers remaining --/
theorem mary_remaining_stickers :
  remaining_stickers 89 3 6 7 = 44 := by
  sorry

#eval remaining_stickers 89 3 6 7

end NUMINAMATH_CALUDE_mary_remaining_stickers_l727_72749


namespace NUMINAMATH_CALUDE_balloon_distribution_l727_72734

/-- Given a total number of balloons and the ratios between different colors,
    calculate the number of balloons for each color. -/
theorem balloon_distribution (total : ℕ) (red_ratio blue_ratio black_ratio : ℕ) 
    (h_total : total = 180)
    (h_red : red_ratio = 3)
    (h_black : black_ratio = 2)
    (h_blue : blue_ratio = 1) :
    ∃ (red blue black : ℕ),
      red = 90 ∧ blue = 30 ∧ black = 60 ∧
      red = red_ratio * blue ∧
      black = black_ratio * blue ∧
      red + blue + black = total :=
by
  sorry

#check balloon_distribution

end NUMINAMATH_CALUDE_balloon_distribution_l727_72734


namespace NUMINAMATH_CALUDE_min_y_value_l727_72727

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 36*y) :
  ∃ (y_min : ℝ), y_min = 18 - Real.sqrt 388 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 16*x' + 36*y' → y' ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_y_value_l727_72727


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l727_72716

theorem incorrect_inequality_transformation (a b : ℝ) (h : a < b) :
  ¬(3 - a < 3 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l727_72716


namespace NUMINAMATH_CALUDE_worker_a_completion_time_l727_72784

/-- The time it takes for Worker A and Worker B to complete a job together and independently -/
def combined_time : ℝ := 2.857142857142857

/-- The time it takes for Worker B to complete the job alone -/
def worker_b_time : ℝ := 10

/-- The time it takes for Worker A to complete the job alone -/
def worker_a_time : ℝ := 4

/-- Theorem stating that Worker A takes 4 hours to complete the job alone -/
theorem worker_a_completion_time :
  (1 / worker_a_time + 1 / worker_b_time) * combined_time = 1 := by
  sorry

end NUMINAMATH_CALUDE_worker_a_completion_time_l727_72784


namespace NUMINAMATH_CALUDE_increase_when_multiplied_l727_72779

theorem increase_when_multiplied (n : ℕ) (m : ℕ) (increase : ℕ) : n = 25 → m = 16 → increase = m * n - n → increase = 375 := by
  sorry

end NUMINAMATH_CALUDE_increase_when_multiplied_l727_72779


namespace NUMINAMATH_CALUDE_bicycle_helmet_cost_increase_l727_72758

/-- The percent increase in the combined cost of a bicycle and helmet --/
theorem bicycle_helmet_cost_increase 
  (bicycle_cost : ℝ) 
  (helmet_cost : ℝ) 
  (bicycle_increase_percent : ℝ) 
  (helmet_increase_percent : ℝ) 
  (h1 : bicycle_cost = 150)
  (h2 : helmet_cost = 50)
  (h3 : bicycle_increase_percent = 10)
  (h4 : helmet_increase_percent = 20) : 
  ((bicycle_cost * (1 + bicycle_increase_percent / 100) + 
    helmet_cost * (1 + helmet_increase_percent / 100)) - 
   (bicycle_cost + helmet_cost)) / (bicycle_cost + helmet_cost) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_helmet_cost_increase_l727_72758


namespace NUMINAMATH_CALUDE_expression_equality_l727_72785

theorem expression_equality : 
  (-2^3 = (-2)^3) ∧ 
  (2^3 ≠ 2*3) ∧ 
  (-((-2)^2) ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l727_72785


namespace NUMINAMATH_CALUDE_production_exceeds_target_in_2022_l727_72766

def initial_production : ℕ := 20000
def annual_increase_rate : ℝ := 0.2
def target_production : ℕ := 60000
def start_year : ℕ := 2015

theorem production_exceeds_target_in_2022 :
  let production_after_n_years (n : ℕ) := initial_production * (1 + annual_increase_rate) ^ n
  ∀ y : ℕ, y < 2022 - start_year → production_after_n_years y ≤ target_production ∧
  production_after_n_years (2022 - start_year) > target_production :=
by sorry

end NUMINAMATH_CALUDE_production_exceeds_target_in_2022_l727_72766


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_l727_72793

def sum_first_n_even (n : ℕ) : ℕ := n * (n + 1)

def sum_five_consecutive_even (k : ℕ) : ℕ := 5 * (2 * k + 2)

theorem smallest_of_five_consecutive_even : 
  ∃ k : ℕ, sum_five_consecutive_even k = sum_first_n_even 25 ∧ 
  2 * k + 2 = 126 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_l727_72793


namespace NUMINAMATH_CALUDE_h_of_3_eq_72_minus_18_sqrt_15_l727_72763

/-- Given functions f, g, and h defined as:
  f(x) = 3x + 6
  g(x) = (√(f(x)) - 3)²
  h(x) = f(g(x))
  Prove that h(3) = 72 - 18√15 -/
theorem h_of_3_eq_72_minus_18_sqrt_15 :
  let f : ℝ → ℝ := λ x ↦ 3 * x + 6
  let g : ℝ → ℝ := λ x ↦ (Real.sqrt (f x) - 3)^2
  let h : ℝ → ℝ := λ x ↦ f (g x)
  h 3 = 72 - 18 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_h_of_3_eq_72_minus_18_sqrt_15_l727_72763


namespace NUMINAMATH_CALUDE_smallest_n_for_perfect_square_l727_72776

theorem smallest_n_for_perfect_square : ∃ (n : ℕ), 
  (n = 12) ∧ 
  (∃ (k : ℕ), (2^n + 2^8 + 2^11 : ℕ) = k^2) ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), (2^m + 2^8 + 2^11 : ℕ) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_perfect_square_l727_72776


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l727_72718

theorem smallest_multiple_of_6_and_15 :
  ∃ b : ℕ, b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ c : ℕ, c > 0 → 6 ∣ c → 15 ∣ c → b ≤ c :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l727_72718


namespace NUMINAMATH_CALUDE_remainder_of_482157_div_6_l727_72730

theorem remainder_of_482157_div_6 : 482157 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_482157_div_6_l727_72730


namespace NUMINAMATH_CALUDE_exists_48_good_perfect_square_l727_72754

/-- A number is k-good if it can be split into two parts y and z where y = k * z -/
def is_k_good (k : ℕ) (n : ℕ) : Prop :=
  ∃ y z : ℕ, y * (10^(Nat.log 10 z + 1)) + z = n ∧ y = k * z

/-- The main theorem: there exists a 48-good perfect square -/
theorem exists_48_good_perfect_square : ∃ n : ℕ, is_k_good 48 n ∧ ∃ m : ℕ, n = m^2 :=
sorry

end NUMINAMATH_CALUDE_exists_48_good_perfect_square_l727_72754


namespace NUMINAMATH_CALUDE_macaroon_solution_l727_72706

/-- Represents the problem of calculating the remaining weight of macaroons --/
def macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) : Prop :=
  let total_weight := total_macaroons * weight_per_macaroon
  let macaroons_per_bag := total_macaroons / num_bags
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon
  let remaining_bags := num_bags - 1
  let remaining_weight := remaining_bags * weight_per_bag
  remaining_weight = 45

/-- Theorem stating the solution to the macaroon problem --/
theorem macaroon_solution : macaroon_problem 12 5 4 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_solution_l727_72706


namespace NUMINAMATH_CALUDE_charlie_bobby_age_difference_l727_72704

theorem charlie_bobby_age_difference :
  ∀ (jenny charlie bobby : ℕ),
  jenny = charlie + 5 →
  ∃ (x : ℕ), charlie + x = 11 ∧ jenny + x = 2 * (bobby + x) →
  charlie = bobby + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_charlie_bobby_age_difference_l727_72704


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l727_72775

theorem pure_imaginary_product (b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (y : ℝ), (1 + b * Complex.I) * (2 + Complex.I) = y * Complex.I) →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l727_72775


namespace NUMINAMATH_CALUDE_triathlon_speed_l727_72795

/-- Triathlon completion problem -/
theorem triathlon_speed (swim_distance : Real) (bike_distance : Real) (run_distance : Real)
  (total_time : Real) (swim_speed : Real) (run_speed : Real) :
  swim_distance = 0.5 ∧ 
  bike_distance = 20 ∧ 
  run_distance = 4 ∧ 
  total_time = 1.75 ∧ 
  swim_speed = 1 ∧ 
  run_speed = 4 →
  (bike_distance / (total_time - (swim_distance / swim_speed) - (run_distance / run_speed))) = 80 := by
  sorry

#check triathlon_speed

end NUMINAMATH_CALUDE_triathlon_speed_l727_72795


namespace NUMINAMATH_CALUDE_a4b4_value_l727_72723

theorem a4b4_value (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_a4b4_value_l727_72723


namespace NUMINAMATH_CALUDE_decimal_representation_symmetry_l727_72788

/-- The main period of the decimal representation of 1/p -/
def decimal_period (p : ℕ) : List ℕ :=
  sorry

/-- Count occurrences of a digit in a list -/
def count_occurrences (digit : ℕ) (l : List ℕ) : ℕ :=
  sorry

theorem decimal_representation_symmetry (p n : ℕ) (h1 : Nat.Prime p) (h2 : p ∣ 10^n + 1) :
  ∀ i ∈ Finset.range 10,
    count_occurrences i (decimal_period p) = count_occurrences (9 - i) (decimal_period p) :=
by sorry

end NUMINAMATH_CALUDE_decimal_representation_symmetry_l727_72788


namespace NUMINAMATH_CALUDE_frequency_distribution_required_l727_72738

/-- Represents a sample of data -/
structure Sample (α : Type*) where
  data : List α

/-- Represents a frequency distribution of a sample -/
def FrequencyDistribution (α : Type*) := α → ℕ

/-- Represents a range of values -/
structure Range (α : Type*) where
  lower : α
  upper : α

/-- Function to determine if a value is within a range -/
def inRange {α : Type*} [PartialOrder α] (x : α) (r : Range α) : Prop :=
  r.lower ≤ x ∧ x ≤ r.upper

/-- Theorem stating that a frequency distribution is required to understand 
    the proportion of a sample within a certain range -/
theorem frequency_distribution_required 
  {α : Type*} [PartialOrder α] (s : Sample α) (r : Range α) :
  ∃ (fd : FrequencyDistribution α), 
    (∀ x, x ∈ s.data → inRange x r → fd x > 0) ∧
    (∀ x, x ∉ s.data ∨ ¬inRange x r → fd x = 0) :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_required_l727_72738


namespace NUMINAMATH_CALUDE_children_count_l727_72751

def number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) : ℕ :=
  total_crayons / crayons_per_child

theorem children_count : number_of_children 6 72 = 12 := by
  sorry

end NUMINAMATH_CALUDE_children_count_l727_72751


namespace NUMINAMATH_CALUDE_six_paths_from_M_to_N_l727_72742

/- Define a directed graph with vertices and edges -/
def Graph : Type := List (Char × Char)

/- Define the graph structure for our problem -/
def problemGraph : Graph := [
  ('M', 'A'), ('M', 'B'),
  ('A', 'C'), ('A', 'D'),
  ('B', 'C'), ('B', 'N'),
  ('C', 'N'), ('D', 'N')
]

/- Function to count paths between two vertices -/
def countPaths (g : Graph) (start finish : Char) : Nat :=
  sorry

/- Theorem stating that there are 6 paths from M to N -/
theorem six_paths_from_M_to_N :
  countPaths problemGraph 'M' 'N' = 6 :=
sorry

end NUMINAMATH_CALUDE_six_paths_from_M_to_N_l727_72742


namespace NUMINAMATH_CALUDE_bruce_bags_l727_72773

/-- The number of bags Bruce can buy with his change after purchasing crayons, books, and calculators. -/
def bags_bruce_can_buy (crayons_packs : ℕ) (crayons_price : ℕ) 
                       (books : ℕ) (books_price : ℕ)
                       (calculators : ℕ) (calculators_price : ℕ)
                       (initial_money : ℕ) (bag_price : ℕ) : ℕ :=
  let total_spent := crayons_packs * crayons_price + books * books_price + calculators * calculators_price
  let change := initial_money - total_spent
  change / bag_price

theorem bruce_bags : 
  bags_bruce_can_buy 5 5 10 5 3 5 200 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bruce_bags_l727_72773


namespace NUMINAMATH_CALUDE_right_triangle_height_l727_72769

theorem right_triangle_height (a b c h : ℝ) : 
  (a = 12 ∧ b = 5 ∧ a^2 + b^2 = c^2) → 
  (h = 60/13 ∨ h = 5) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_height_l727_72769


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l727_72780

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (1 + Real.tan α) * (1 + Real.tan β) = 2 →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l727_72780


namespace NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l727_72726

theorem equation_solvable_for_small_primes :
  ∀ p : ℕ, p.Prime → p ≤ 100 →
  ∃ x y : ℕ, (y^37 : ℤ) ≡ (x^3 + 11 : ℤ) [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_equation_solvable_for_small_primes_l727_72726


namespace NUMINAMATH_CALUDE_continuity_at_two_and_delta_l727_72762

def f (x : ℝ) := -3 * x^2 - 5

theorem continuity_at_two_and_delta (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε ∧
  ∃ δ₀ > 0, δ₀ = ε/3 ∧ ∀ x, |x - 2| < δ₀ → |f x - f 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_two_and_delta_l727_72762


namespace NUMINAMATH_CALUDE_elevator_max_additional_weight_l727_72724

/-- The maximum weight a person can have to enter an elevator without overloading it,
    given the current occupants and the elevator's weight limit. -/
def max_additional_weight (adult_count : ℕ) (adult_avg_weight : ℝ)
                          (child_count : ℕ) (child_avg_weight : ℝ)
                          (max_elevator_weight : ℝ) : ℝ :=
  max_elevator_weight - (adult_count * adult_avg_weight + child_count * child_avg_weight)

/-- Theorem stating the maximum weight of the next person to enter the elevator
    without overloading it, given the specific conditions. -/
theorem elevator_max_additional_weight :
  max_additional_weight 3 140 2 64 600 = 52 := by
  sorry

end NUMINAMATH_CALUDE_elevator_max_additional_weight_l727_72724


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l727_72756

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x - 2| < 3) ↔ (∃ x : ℝ, |x - 2| ≥ 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l727_72756


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_bounded_l727_72711

theorem quadratic_always_nonnegative_implies_a_bounded (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) →
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_bounded_l727_72711


namespace NUMINAMATH_CALUDE_percentage_of_number_l727_72740

theorem percentage_of_number (x : ℝ) (y : ℝ) (z : ℝ) (h : x = (y / 100) * z) : 
  x = 120 ∧ y = 150 ∧ z = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l727_72740


namespace NUMINAMATH_CALUDE_intersection_points_l727_72725

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

def g (x : ℝ) : ℝ := -f x

def h (x : ℝ) : ℝ := f (-x)

theorem intersection_points :
  (∃ a b : ℝ, a ≠ b ∧ f a = g a ∧ f b = g b) ∧
  (∃! c : ℝ, f c = h c) ∧
  (∀ x y : ℝ, x ≠ y → (f x = g x ∧ f y = g y) → (f x = h x ∧ f y = h y) → False) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l727_72725


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l727_72761

/-- A triangle represented by three points on a 2D plane. -/
structure Triangle :=
  (a b c : ℕ × ℕ)

/-- Checks if a triangle is isosceles. -/
def isIsosceles (t : Triangle) : Bool :=
  let d1 := ((t.a.1 - t.b.1)^2 + (t.a.2 - t.b.2)^2 : ℕ)
  let d2 := ((t.b.1 - t.c.1)^2 + (t.b.2 - t.c.2)^2 : ℕ)
  let d3 := ((t.c.1 - t.a.1)^2 + (t.c.2 - t.a.2)^2 : ℕ)
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

def triangles : List Triangle := [
  ⟨(0, 6), (2, 6), (1, 4)⟩,
  ⟨(3, 4), (3, 6), (5, 4)⟩,
  ⟨(0, 1), (3, 2), (6, 1)⟩,
  ⟨(7, 4), (6, 6), (9, 4)⟩,
  ⟨(8, 1), (9, 3), (10, 0)⟩
]

theorem four_isosceles_triangles :
  (triangles.filter isIsosceles).length = 4 := by
  sorry


end NUMINAMATH_CALUDE_four_isosceles_triangles_l727_72761


namespace NUMINAMATH_CALUDE_vectors_form_basis_l727_72745

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

def is_basis (v w : V) : Prop :=
  LinearIndependent ℝ ![v, w] ∧ Submodule.span ℝ {v, w} = ⊤

theorem vectors_form_basis (ha : a ≠ 0) (hb : b ≠ 0) (hnc : ¬ ∃ (k : ℝ), a = k • b) :
  is_basis V (a + b) (a - b) := by
  sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l727_72745


namespace NUMINAMATH_CALUDE_linear_equation_implies_a_squared_plus_a_minus_one_equals_one_l727_72722

theorem linear_equation_implies_a_squared_plus_a_minus_one_equals_one (a : ℝ) :
  (∀ x, ∃ k, (a + 4) * x^(|a + 3|) + 8 = k * x + 8) →
  a^2 + a - 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_implies_a_squared_plus_a_minus_one_equals_one_l727_72722


namespace NUMINAMATH_CALUDE_polynomial_equality_l727_72765

theorem polynomial_equality (k : ℝ) : 
  (∀ x : ℝ, (x + 6) * (x - 5) = x^2 + k*x - 30) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l727_72765


namespace NUMINAMATH_CALUDE_race_length_race_length_is_165_l727_72752

theorem race_length : ℝ → Prop :=
  fun x =>
    ∀ (speed_a speed_b speed_c : ℝ),
      speed_a > 0 ∧ speed_b > 0 ∧ speed_c > 0 →
      x > 35 →
      speed_b * x = speed_a * (x - 15) →
      speed_c * x = speed_a * (x - 35) →
      speed_c * (x - 15) = speed_b * (x - 22) →
      x = 165

theorem race_length_is_165 : race_length 165 := by
  sorry

end NUMINAMATH_CALUDE_race_length_race_length_is_165_l727_72752


namespace NUMINAMATH_CALUDE_parabola_line_intersection_length_l727_72717

/-- Given a parabola x² = 2py (p > 0) and a line y = 2x + p/2 that intersects
    the parabola at points A and B, prove that the length of AB is 10p. -/
theorem parabola_line_intersection_length (p : ℝ) (h : p > 0) : 
  let parabola := fun x y => x^2 = 2*p*y
  let line := fun x y => y = 2*x + p/2
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A ≠ B ∧
    ‖A - B‖ = 10*p :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_length_l727_72717


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l727_72787

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l727_72787


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_rate_l727_72796

/-- Mrs. Hilt's daily reading rate -/
def daily_reading_rate (total_books : ℕ) (total_days : ℕ) : ℚ :=
  total_books / total_days

/-- Theorem: Mrs. Hilt reads 5 books per day -/
theorem mrs_hilt_reading_rate :
  daily_reading_rate 15 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_rate_l727_72796


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l727_72771

theorem medicine_supply_duration (pills : ℕ) (consumption_rate : ℚ) (consumption_days : ℕ) (days_per_month : ℕ) : 
  pills = 90 → 
  consumption_rate = 1/3 → 
  consumption_days = 3 → 
  days_per_month = 30 → 
  (pills * consumption_days / consumption_rate) / days_per_month = 27 := by
  sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l727_72771


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l727_72753

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l727_72753


namespace NUMINAMATH_CALUDE_distance_circle_to_line_l727_72774

/-- The distance from the center of the circle ρ=2cos θ to the line 2ρsin(θ + π/3)=1 is (√3 - 1) / 2 -/
theorem distance_circle_to_line :
  let circle : ℝ → ℝ → Prop := λ ρ θ ↦ ρ = 2 * Real.cos θ
  let line : ℝ → ℝ → Prop := λ ρ θ ↦ 2 * ρ * Real.sin (θ + π/3) = 1
  let circle_center : ℝ × ℝ := (1, 0)
  let distance := (Real.sqrt 3 - 1) / 2
  ∃ (d : ℝ), d = distance ∧ 
    d = (|Real.sqrt 3 * circle_center.1 + circle_center.2 - 1|) / Real.sqrt (3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_to_line_l727_72774


namespace NUMINAMATH_CALUDE_saturday_newspaper_delivery_l727_72710

/-- Given that Peter delivers newspapers on weekends, prove that he delivers 45 papers on Saturday. -/
theorem saturday_newspaper_delivery :
  ∀ (saturday_delivery sunday_delivery : ℕ),
  saturday_delivery + sunday_delivery = 110 →
  sunday_delivery = saturday_delivery + 20 →
  saturday_delivery = 45 := by
sorry

end NUMINAMATH_CALUDE_saturday_newspaper_delivery_l727_72710


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l727_72778

-- Define the original line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the rotated line l₁
def line_l₁ (x y : ℝ) : Prop := x + y - 5 = 0

-- State the theorem
theorem rotation_90_degrees :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), line_l x₀ y₀ ∧ 
    (x - 2) = -(y₀ - 3) ∧ 
    (y - 3) = (x₀ - 2)) →
  line_l₁ x y := by sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l727_72778


namespace NUMINAMATH_CALUDE_total_peanuts_l727_72728

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35

theorem total_peanuts : jose_peanuts + kenya_peanuts + malachi_peanuts = 386 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l727_72728


namespace NUMINAMATH_CALUDE_discarded_number_proof_l727_72705

theorem discarded_number_proof (numbers : Finset ℕ) (sum : ℕ) (x : ℕ) :
  Finset.card numbers = 50 →
  sum = Finset.sum numbers id →
  sum / 50 = 50 →
  55 ∈ numbers →
  x ∈ numbers →
  x ≠ 55 →
  (sum - 55 - x) / 48 = 50 →
  x = 45 :=
by sorry

end NUMINAMATH_CALUDE_discarded_number_proof_l727_72705


namespace NUMINAMATH_CALUDE_equilateral_triangle_symmetry_l727_72786

-- Define the shape types
inductive Shape
  | Rectangle
  | Rhombus
  | EquilateralTriangle
  | Circle

-- Define symmetry properties
def hasAxisSymmetry (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle          => true
  | Shape.Rhombus            => true
  | Shape.EquilateralTriangle => true
  | Shape.Circle             => true

def hasCenterSymmetry (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle          => true
  | Shape.Rhombus            => true
  | Shape.EquilateralTriangle => false
  | Shape.Circle             => true

-- Theorem statement
theorem equilateral_triangle_symmetry :
  ∃ (s : Shape), hasAxisSymmetry s ∧ ¬hasCenterSymmetry s ∧
  (∀ (t : Shape), t ≠ s → (hasAxisSymmetry t → hasCenterSymmetry t)) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_symmetry_l727_72786


namespace NUMINAMATH_CALUDE_initial_cards_l727_72768

theorem initial_cards (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 4 → total = 13 → initial + added = total → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_l727_72768


namespace NUMINAMATH_CALUDE_cost_of_pizza_slice_l727_72782

/-- The cost of a slice of pizza given the conditions of Zoe's purchase -/
theorem cost_of_pizza_slice (num_people : ℕ) (soda_cost : ℚ) (total_spent : ℚ) :
  num_people = 6 →
  soda_cost = 1/2 →
  total_spent = 9 →
  (total_spent - num_people * soda_cost) / num_people = 1 := by
  sorry

#check cost_of_pizza_slice

end NUMINAMATH_CALUDE_cost_of_pizza_slice_l727_72782


namespace NUMINAMATH_CALUDE_reduce_to_less_than_100_l727_72764

/-- Represents a digit from 4 to 9 -/
inductive ValidDigit
  | four
  | five
  | six
  | seven
  | eight
  | nine

/-- Represents a natural number composed of ValidDigits -/
def ValidNumber := List ValidDigit

/-- Represents an operation that can be performed on a ValidNumber -/
inductive Operation
  | deletePair (d : ValidDigit) : Operation
  | deleteDoublePair (d1 d2 : ValidDigit) : Operation
  | insertPair (d : ValidDigit) : Operation
  | insertDoublePair (d1 d2 : ValidDigit) : Operation

/-- Applies an operation to a ValidNumber -/
def applyOperation (n : ValidNumber) (op : Operation) : ValidNumber :=
  sorry

/-- Checks if a ValidNumber is less than 100 -/
def isLessThan100 (n : ValidNumber) : Prop :=
  sorry

theorem reduce_to_less_than_100 (n : ValidNumber) (h : n.length = 2019) :
  ∃ (ops : List Operation), isLessThan100 (ops.foldl applyOperation n) :=
  sorry

end NUMINAMATH_CALUDE_reduce_to_less_than_100_l727_72764


namespace NUMINAMATH_CALUDE_eggs_per_crate_l727_72707

theorem eggs_per_crate (initial_crates : ℕ) (given_away : ℕ) (additional_crates : ℕ) (final_count : ℕ) :
  initial_crates = 6 →
  given_away = 2 →
  additional_crates = 5 →
  final_count = 270 →
  ∃ (eggs_per_crate : ℕ), eggs_per_crate = 30 ∧
    final_count = (initial_crates - given_away + additional_crates) * eggs_per_crate :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_crate_l727_72707


namespace NUMINAMATH_CALUDE_total_suitcase_weight_is_434_l727_72790

/-- The total weight of all suitcases for a family vacation --/
def total_suitcase_weight : ℕ :=
  let siblings_suitcases := List.range 6 |>.sum
  let siblings_weight := siblings_suitcases * 10
  let parents_suitcases := 2 * 3
  let parents_weight := parents_suitcases * 12
  let grandparents_suitcases := 2 * 2
  let grandparents_weight := grandparents_suitcases * 8
  let relatives_suitcases := 8
  let relatives_weight := relatives_suitcases * 15
  siblings_weight + parents_weight + grandparents_weight + relatives_weight

theorem total_suitcase_weight_is_434 : total_suitcase_weight = 434 := by
  sorry

end NUMINAMATH_CALUDE_total_suitcase_weight_is_434_l727_72790


namespace NUMINAMATH_CALUDE_green_chips_count_l727_72750

theorem green_chips_count (total : ℕ) (red : ℕ) (h1 : total = 60) (h2 : red = 34) :
  total - (total / 6) - red = 16 := by
  sorry

end NUMINAMATH_CALUDE_green_chips_count_l727_72750


namespace NUMINAMATH_CALUDE_shopping_trip_expenses_l727_72746

theorem shopping_trip_expenses (total : ℝ) (food_percent : ℝ) :
  (total > 0) →
  (food_percent ≥ 0) →
  (food_percent ≤ 100) →
  (0.5 * total + food_percent / 100 * total + 0.4 * total = total) →
  (0.04 * 0.5 * total + 0.08 * 0.4 * total = 0.052 * total) →
  food_percent = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_expenses_l727_72746


namespace NUMINAMATH_CALUDE_distance_on_number_line_distance_negative_five_negative_one_l727_72744

theorem distance_on_number_line : ∀ (a b : ℝ), abs (a - b) = abs (b - a) :=
by sorry

theorem distance_negative_five_negative_one : abs (-5 - (-1)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distance_on_number_line_distance_negative_five_negative_one_l727_72744


namespace NUMINAMATH_CALUDE_table_tennis_equipment_theorem_l727_72702

/-- Represents the price of table tennis equipment and store discounts -/
structure TableTennisEquipment where
  racket_price : ℝ
  ball_price : ℝ
  store_a_discount : ℝ
  store_b_free_balls : ℕ

/-- Proves that given the conditions, the prices are correct and Store A is more cost-effective -/
theorem table_tennis_equipment_theorem (e : TableTennisEquipment)
  (h1 : 2 * e.racket_price + 3 * e.ball_price = 75)
  (h2 : 3 * e.racket_price + 2 * e.ball_price = 100)
  (h3 : e.store_a_discount = 0.1)
  (h4 : e.store_b_free_balls = 10) :
  e.racket_price = 30 ∧
  e.ball_price = 5 ∧
  (1 - e.store_a_discount) * (20 * e.racket_price + 30 * e.ball_price) <
  20 * e.racket_price + (30 - e.store_b_free_balls) * e.ball_price := by
  sorry

end NUMINAMATH_CALUDE_table_tennis_equipment_theorem_l727_72702


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l727_72794

/-- The volume of a right triangular prism with base edge length 2 and height 3 is √3. -/
theorem right_triangular_prism_volume : 
  let base_edge : ℝ := 2
  let height : ℝ := 3
  let base_area : ℝ := Real.sqrt 3 / 4 * base_edge ^ 2
  let volume : ℝ := 1 / 3 * base_area * height
  volume = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l727_72794


namespace NUMINAMATH_CALUDE_trajectory_max_value_l727_72757

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (x + 1)^2 + (4/3) * y^2 = 4

/-- The distance ratio condition -/
def distance_ratio (x y : ℝ) : Prop :=
  (x^2 + y^2) / ((x - 3)^2 + y^2) = 1/4

theorem trajectory_max_value :
  ∀ x y : ℝ, 
    distance_ratio x y → 
    trajectory x y → 
    2 * x^2 + y^2 ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_trajectory_max_value_l727_72757


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l727_72798

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : 2 * Real.cos α - Real.cos β = (3 : ℝ) / 2)
  (h2 : 2 * Real.sin α - Real.sin β = 2) :
  Real.cos (α - β) = -(5 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l727_72798


namespace NUMINAMATH_CALUDE_friendship_theorem_l727_72714

-- Define a type for people
def Person : Type := Nat

-- Define the friendship relation
def IsFriend (p q : Person) : Prop := sorry

-- State the theorem
theorem friendship_theorem :
  ∀ (group : Finset Person),
  (Finset.card group = 12) →
  ∃ (A B : Person),
    A ∈ group ∧ B ∈ group ∧ A ≠ B ∧
    ∃ (C D E F G : Person),
      C ∈ group ∧ D ∈ group ∧ E ∈ group ∧ F ∈ group ∧ G ∈ group ∧
      C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B ∧ E ≠ A ∧ E ≠ B ∧ F ≠ A ∧ F ≠ B ∧ G ≠ A ∧ G ≠ B ∧
      C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ E ≠ F ∧ E ≠ G ∧ F ≠ G ∧
      ((IsFriend C A ∧ IsFriend C B) ∨ (¬IsFriend C A ∧ ¬IsFriend C B)) ∧
      ((IsFriend D A ∧ IsFriend D B) ∨ (¬IsFriend D A ∧ ¬IsFriend D B)) ∧
      ((IsFriend E A ∧ IsFriend E B) ∨ (¬IsFriend E A ∧ ¬IsFriend E B)) ∧
      ((IsFriend F A ∧ IsFriend F B) ∨ (¬IsFriend F A ∧ ¬IsFriend F B)) ∧
      ((IsFriend G A ∧ IsFriend G B) ∨ (¬IsFriend G A ∧ ¬IsFriend G B)) :=
by
  sorry


end NUMINAMATH_CALUDE_friendship_theorem_l727_72714


namespace NUMINAMATH_CALUDE_smallest_integer_l727_72732

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 50) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 50 → b ≤ c := by
  sorry

#check smallest_integer

end NUMINAMATH_CALUDE_smallest_integer_l727_72732


namespace NUMINAMATH_CALUDE_roots_of_cubic_polynomials_l727_72729

theorem roots_of_cubic_polynomials (a b : ℝ) (r s : ℝ) :
  (∃ t, r + s + t = 0 ∧ r * s + r * t + s * t = a) →
  (∃ t', r + 4 + s - 3 + t' = 0 ∧ (r + 4) * (s - 3) + (r + 4) * t' + (s - 3) * t' = a) →
  b = -330 ∨ b = 90 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_cubic_polynomials_l727_72729


namespace NUMINAMATH_CALUDE_power_of_half_squared_times_32_l727_72739

theorem power_of_half_squared_times_32 : ∃ x : ℝ, x * (1/2)^2 = 2^3 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_of_half_squared_times_32_l727_72739


namespace NUMINAMATH_CALUDE_jeffrey_steps_to_mailbox_l727_72712

/-- Represents Jeffrey's walking pattern -/
structure WalkingPattern where
  forward : ℕ
  backward : ℕ

/-- Calculates the total steps taken given a walking pattern and distance -/
def totalSteps (pattern : WalkingPattern) (distance : ℕ) : ℕ :=
  let effectiveStep := pattern.forward - pattern.backward
  let cycles := distance / effectiveStep
  cycles * (pattern.forward + pattern.backward)

/-- Theorem: Jeffrey's total steps to the mailbox -/
theorem jeffrey_steps_to_mailbox :
  let pattern : WalkingPattern := { forward := 3, backward := 2 }
  let distance : ℕ := 66
  totalSteps pattern distance = 330 := by
  sorry


end NUMINAMATH_CALUDE_jeffrey_steps_to_mailbox_l727_72712


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l727_72772

theorem simplify_and_evaluate : 
  let a : ℝ := -2
  let b : ℝ := 1
  3 * a + 2 * (a - 1/2 * b^2) - (a - 2 * b^2) = -7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l727_72772


namespace NUMINAMATH_CALUDE_distance_traveled_l727_72719

/-- 
Given a skater's speed and time spent skating, calculate the total distance traveled.
-/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 10) (h2 : time = 8) :
  speed * time = 80 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l727_72719


namespace NUMINAMATH_CALUDE_nth_root_sum_theorem_l727_72713

theorem nth_root_sum_theorem (a : ℝ) (n : ℕ) (hn : n > 1) :
  let f : ℝ → ℝ := λ x => (x^n - a^n)^(1/n) + (2*a^n - x^n)^(1/n)
  ∀ x, f x = a ↔ 
    (a ≠ 0 ∧ 
      ((n % 2 = 1 ∧ (x = a * (2^(1/n)) ∨ x = a)) ∨ 
       (n % 2 = 0 ∧ a > 0 ∧ (x = a * (2^(1/n)) ∨ x = -a * (2^(1/n)) ∨ x = a ∨ x = -a)))) ∨
    (a = 0 ∧ 
      ((n % 2 = 1 ∧ true) ∨ 
       (n % 2 = 0 ∧ x = 0))) :=
by sorry


end NUMINAMATH_CALUDE_nth_root_sum_theorem_l727_72713


namespace NUMINAMATH_CALUDE_hitAtMostOnce_mutually_exclusive_hitBothTimes_l727_72747

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two shots -/
def TwoShotOutcome := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at most once -/
def hitAtMostOnce (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Miss, ShotOutcome.Miss) => True
  | (ShotOutcome.Hit, ShotOutcome.Miss) => True
  | (ShotOutcome.Miss, ShotOutcome.Hit) => True
  | (ShotOutcome.Hit, ShotOutcome.Hit) => False

/-- The event of hitting the target both times -/
def hitBothTimes (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Hit, ShotOutcome.Hit) => True
  | _ => False

theorem hitAtMostOnce_mutually_exclusive_hitBothTimes :
  ∀ (outcome : TwoShotOutcome), ¬(hitAtMostOnce outcome ∧ hitBothTimes outcome) :=
by sorry

end NUMINAMATH_CALUDE_hitAtMostOnce_mutually_exclusive_hitBothTimes_l727_72747


namespace NUMINAMATH_CALUDE_class_size_calculation_l727_72792

/-- The number of students supposed to be in Miss Smith's second period English class -/
def total_students : ℕ :=
  let tables := 6
  let students_per_table := 3
  let present_students := tables * students_per_table
  let bathroom_students := 3
  let canteen_students := 3 * bathroom_students
  let new_group_size := 4
  let new_groups := 2
  let new_students := new_groups * new_group_size
  let foreign_students := 3 + 3 + 3  -- Germany, France, Norway

  present_students + bathroom_students + canteen_students + new_students + foreign_students

theorem class_size_calculation :
  total_students = 47 := by
  sorry

end NUMINAMATH_CALUDE_class_size_calculation_l727_72792


namespace NUMINAMATH_CALUDE_ellipse_sum_theorem_l727_72791

/-- Represents an ellipse with center (h, k) and semi-axes a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse with center (-3, 5), semi-major axis 7, and semi-minor axis 4,
    the sum h + k + a + b equals 13 -/
theorem ellipse_sum_theorem (e : Ellipse)
    (center_h : e.h = -3)
    (center_k : e.k = 5)
    (semi_major : e.a = 7)
    (semi_minor : e.b = 4) :
    ellipse_sum e = 13 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_theorem_l727_72791


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l727_72736

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l727_72736


namespace NUMINAMATH_CALUDE_inequalities_theorem_l727_72797

theorem inequalities_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) ≤ (a + b) / 2 ∧
  Real.sqrt (a * b) ≤ (a + b) / 2 ∧
  (a + b) / 2 ≤ Real.sqrt ((a^2 + b^2) / 2) ∧
  b^2 / a + a^2 / b ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l727_72797


namespace NUMINAMATH_CALUDE_domain_shift_l727_72708

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x)
def domain_f : Set ℝ := Set.Icc (-1) 1

-- Define the domain of f(x + 1)
def domain_f_shifted : Set ℝ := Set.Icc (-2) 0

-- Theorem statement
theorem domain_shift :
  (∀ x ∈ domain_f, f x ≠ 0) →
  (∀ y ∈ domain_f_shifted, f (y + 1) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l727_72708


namespace NUMINAMATH_CALUDE_vector_operation_l727_72721

/-- Given vectors a = (2,4) and b = (-1,1), prove that 2a - b = (5,7) -/
theorem vector_operation (a b : ℝ × ℝ) : 
  a = (2, 4) → b = (-1, 1) → 2 • a - b = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l727_72721


namespace NUMINAMATH_CALUDE_square_difference_l727_72737

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l727_72737


namespace NUMINAMATH_CALUDE_largest_angle_cosine_l727_72789

theorem largest_angle_cosine (A B C : ℝ) (h1 : A = π/6) 
  (h2 : 2 * (B * C * Real.cos A) = 3 * (B^2 + C^2 - 2*B*C*Real.cos A)) :
  Real.cos (max A (max B C)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_cosine_l727_72789


namespace NUMINAMATH_CALUDE_equation_solution_l727_72759

theorem equation_solution :
  ∃ x : ℚ, (x + 3*x = 300 - (4*x + 5*x)) ∧ (x = 300/13) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l727_72759


namespace NUMINAMATH_CALUDE_parabola_axis_l727_72720

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := x^2 = -8*y

/-- The equation of the axis of a parabola -/
def axis_equation (y : ℝ) : Prop := y = 2

/-- Theorem: The axis of the parabola x^2 = -8y is y = 2 -/
theorem parabola_axis :
  (∀ x y : ℝ, parabola_equation x y) →
  (∀ y : ℝ, axis_equation y) :=
sorry

end NUMINAMATH_CALUDE_parabola_axis_l727_72720


namespace NUMINAMATH_CALUDE_problem_solution_l727_72743

theorem problem_solution (P Q : ℚ) : 
  (4 / 7 : ℚ) = P / 63 ∧ (4 / 7 : ℚ) = 98 / (Q - 14) → P + Q = 221.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l727_72743


namespace NUMINAMATH_CALUDE_building_heights_sum_l727_72799

/-- The combined height of three buildings with their antennas -/
def combined_height (esb_height esb_antenna wt_height wt_antenna owt_height owt_antenna : ℕ) : ℕ :=
  (esb_height + esb_antenna) + (wt_height + wt_antenna) + (owt_height + owt_antenna)

/-- Theorem stating the combined height of the three buildings -/
theorem building_heights_sum :
  combined_height 1250 204 1450 280 1368 408 = 4960 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_sum_l727_72799


namespace NUMINAMATH_CALUDE_thirteenth_digit_of_sum_l727_72748

def decimal_sum (a b : ℚ) : ℚ := a + b

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem thirteenth_digit_of_sum :
  let sum := decimal_sum (1/8) (1/11)
  nth_digit_after_decimal sum 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_digit_of_sum_l727_72748


namespace NUMINAMATH_CALUDE_li_age_l727_72741

/-- Given the ages of Zhang, Jung, and Li, prove that Li is 12 years old. -/
theorem li_age (zhang_age li_age jung_age : ℕ) 
  (h1 : zhang_age = 2 * li_age)
  (h2 : jung_age = zhang_age + 2)
  (h3 : jung_age = 26) :
  li_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_li_age_l727_72741


namespace NUMINAMATH_CALUDE_polynomial_equality_l727_72715

theorem polynomial_equality (a b A : ℝ) (h : A / (2 * a * b) = 1 - 4 * a^2) : 
  A = 2 * a * b - 8 * a^3 * b :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l727_72715


namespace NUMINAMATH_CALUDE_unchanged_100th_is_100_l727_72735

def is_valid_sequence (s : List ℕ) : Prop :=
  s.length = 1982 ∧ s.toFinset = Finset.range 1983 \ {0}

def swap_adjacent (s : List ℕ) : List ℕ :=
  s.zipWith (λ a b => if a > b then b else a) (s.tail.append [0])

def left_to_right_pass (s : List ℕ) : List ℕ :=
  (s.length - 1).fold (λ _ s' => swap_adjacent s') s

def right_to_left_pass (s : List ℕ) : List ℕ :=
  (left_to_right_pass s.reverse).reverse

def double_pass (s : List ℕ) : List ℕ :=
  right_to_left_pass (left_to_right_pass s)

theorem unchanged_100th_is_100 (s : List ℕ) :
  is_valid_sequence s →
  (double_pass s).nthLe 99 (by sorry) = s.nthLe 99 (by sorry) →
  s.nthLe 99 (by sorry) = 100 := by sorry

end NUMINAMATH_CALUDE_unchanged_100th_is_100_l727_72735


namespace NUMINAMATH_CALUDE_first_term_is_four_l727_72709

/-- Geometric sequence with common ratio -2 and sum of first 5 terms equal to 44 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * (-2)) ∧ 
  (a 1 + a 2 + a 3 + a 4 + a 5 = 44)

/-- The first term of the geometric sequence is 4 -/
theorem first_term_is_four (a : ℕ → ℝ) (h : geometric_sequence a) : a 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_four_l727_72709


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l727_72733

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 240

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = p * pig_value + g * goat_value

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 80

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l727_72733


namespace NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l727_72767

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → y < 3 → f a x > f a y) → 
  (a ≥ 0 ∧ a ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_decreasing_f_l727_72767


namespace NUMINAMATH_CALUDE_minimized_surface_area_sum_l727_72701

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- One face has sides of length 3, 4, and 5 -/
  base_sides : Fin 3 → ℝ
  base_sides_values : base_sides = ![3, 4, 5]
  /-- The volume of the tetrahedron is 24 -/
  volume : ℝ
  volume_value : volume = 24

/-- Represents the surface area of the tetrahedron in the form a√b + c -/
structure SurfaceArea where
  a : ℕ
  b : ℕ
  c : ℕ
  /-- b is not divisible by the square of any prime -/
  b_squarefree : ∀ p : ℕ, Prime p → ¬(p^2 ∣ b)

/-- The main theorem stating the sum of a, b, and c for the minimized surface area -/
theorem minimized_surface_area_sum (t : Tetrahedron) :
  ∃ (sa : SurfaceArea), (∀ other_sa : SurfaceArea, 
    sa.a * Real.sqrt sa.b + sa.c ≤ other_sa.a * Real.sqrt other_sa.b + other_sa.c) → 
    sa.a + sa.b + sa.c = 157 := by
  sorry

end NUMINAMATH_CALUDE_minimized_surface_area_sum_l727_72701


namespace NUMINAMATH_CALUDE_dilation_rotation_theorem_l727_72760

/-- The matrix representing a dilation by scale factor 4 centered at the origin -/
def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![4, 0; 0, 4]

/-- The matrix representing a 90-degree counterclockwise rotation -/
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The combined transformation matrix -/
def combined_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -4; 4, 0]

theorem dilation_rotation_theorem :
  rotation_matrix * dilation_matrix = combined_matrix :=
sorry

end NUMINAMATH_CALUDE_dilation_rotation_theorem_l727_72760


namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_l727_72781

theorem triangle_inequality_and_equality (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) 
    ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) ∧ 
  ((Real.sqrt (a + b - c) + Real.sqrt (b + c - a) + Real.sqrt (c + a - b) 
    = Real.sqrt a + Real.sqrt b + Real.sqrt c) ↔ (a = b ∧ b = c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_l727_72781


namespace NUMINAMATH_CALUDE_simplify_expression_l727_72700

theorem simplify_expression (a b c d x y : ℝ) (h : cx + dy ≠ 0) :
  (c*x*(b^2*x^2 + 3*b^2*y^2 + a^2*y^2) + d*y*(b^2*x^2 + 3*a^2*x^2 + a^2*y^2)) / (c*x + d*y) =
  (b^2 + 3*a^2)*x^2 + (a^2 + 3*b^2)*y^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l727_72700


namespace NUMINAMATH_CALUDE_constant_value_proof_l727_72777

theorem constant_value_proof (b c : ℝ) :
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_value_proof_l727_72777


namespace NUMINAMATH_CALUDE_original_number_is_27_l727_72770

theorem original_number_is_27 (x : ℕ) :
  (Odd (3 * x)) →
  (∃ k : ℕ, 3 * x = 9 * k) →
  (∃ y : ℕ, x * y = 108) →
  x = 27 := by
sorry

end NUMINAMATH_CALUDE_original_number_is_27_l727_72770


namespace NUMINAMATH_CALUDE_repeated_roots_coincide_l727_72783

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a point x -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A quadratic polynomial has a repeated root -/
def has_repeated_root (p : QuadraticPolynomial) : Prop :=
  ∃ r : ℝ, p.eval r = 0 ∧ (∀ x : ℝ, p.eval x = p.a * (x - r)^2)

/-- The sum of two quadratic polynomials -/
def add_poly (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a + q.a, p.b + q.b, p.c + q.c⟩

/-- Theorem: If P and Q are quadratic polynomials with repeated roots, 
    and P + Q also has a repeated root, then all these roots are equal -/
theorem repeated_roots_coincide (P Q : QuadraticPolynomial) 
  (hP : has_repeated_root P) 
  (hQ : has_repeated_root Q) 
  (hPQ : has_repeated_root (add_poly P Q)) : 
  ∃ r : ℝ, (∀ x : ℝ, P.eval x = P.a * (x - r)^2) ∧ 
            (∀ x : ℝ, Q.eval x = Q.a * (x - r)^2) ∧ 
            (∀ x : ℝ, (add_poly P Q).eval x = (P.a + Q.a) * (x - r)^2) := by
  sorry


end NUMINAMATH_CALUDE_repeated_roots_coincide_l727_72783


namespace NUMINAMATH_CALUDE_multiply_80641_and_9999_l727_72755

theorem multiply_80641_and_9999 : 80641 * 9999 = 805589359 := by
  sorry

end NUMINAMATH_CALUDE_multiply_80641_and_9999_l727_72755
