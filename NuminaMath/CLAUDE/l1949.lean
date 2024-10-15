import Mathlib

namespace NUMINAMATH_CALUDE_pet_store_parrots_l1949_194960

/-- The number of bird cages in the pet store -/
def num_cages : ℝ := 6.0

/-- The number of parakeets in the pet store -/
def num_parakeets : ℝ := 2.0

/-- The average number of birds that can occupy 1 cage -/
def birds_per_cage : ℝ := 1.333333333

/-- The number of parrots in the pet store -/
def num_parrots : ℝ := 6.0

theorem pet_store_parrots :
  num_parrots = num_cages * birds_per_cage - num_parakeets :=
by sorry

end NUMINAMATH_CALUDE_pet_store_parrots_l1949_194960


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l1949_194979

theorem final_sum_after_transformation (x y T : ℝ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l1949_194979


namespace NUMINAMATH_CALUDE_not_quadratic_radical_l1949_194977

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop := x ≥ 0

-- State the theorem
theorem not_quadratic_radical : ¬ is_quadratic_radical (-4) := by
  sorry

end NUMINAMATH_CALUDE_not_quadratic_radical_l1949_194977


namespace NUMINAMATH_CALUDE_min_value_theorem_l1949_194933

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 4/y ≥ 1/a + 4/b) →
  1/a + 4/b = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1949_194933


namespace NUMINAMATH_CALUDE_tetrahedral_toys_probability_l1949_194911

-- Define the face values of the tetrahedral toys
def face_values : Finset ℕ := {1, 2, 3, 5}

-- Define the sample space of all possible outcomes
def sample_space : Finset (ℕ × ℕ) := face_values.product face_values

-- Define m as the sum of the two face values
def m (outcome : ℕ × ℕ) : ℕ := outcome.1 + outcome.2

-- Define the event where m is not less than 6
def event_m_ge_6 : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x ≥ 6)

-- Define the event where m is odd
def event_m_odd : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x % 2 = 1)

-- Define the event where m is even
def event_m_even : Finset (ℕ × ℕ) := sample_space.filter (λ x => m x % 2 = 0)

theorem tetrahedral_toys_probability :
  (event_m_ge_6.card : ℚ) / sample_space.card = 1/2 ∧
  (event_m_odd.card : ℚ) / sample_space.card = 3/8 ∧
  (event_m_even.card : ℚ) / sample_space.card = 5/8 :=
sorry

end NUMINAMATH_CALUDE_tetrahedral_toys_probability_l1949_194911


namespace NUMINAMATH_CALUDE_compound_molar_mass_l1949_194993

/-- Given that 8 moles of a compound weigh 1600 grams, prove that its molar mass is 200 grams/mole -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 1600) (h2 : moles = 8) :
  mass / moles = 200 := by
  sorry

end NUMINAMATH_CALUDE_compound_molar_mass_l1949_194993


namespace NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l1949_194918

theorem pentagon_triangle_side_ratio :
  ∀ (t p : ℝ),
    t > 0 ∧ p > 0 →
    3 * t = 15 →
    5 * p = 15 →
    t / p = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_triangle_side_ratio_l1949_194918


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1949_194925

theorem trigonometric_equation_solutions (x : ℝ) :
  (5.14 * Real.sin (3 * x) + Real.sin (5 * x) = 2 * ((Real.cos (2 * x))^2 - (Real.sin (3 * x))^2)) ↔
  (∃ k : ℤ, x = π / 2 * (2 * k + 1) ∨ x = π / 18 * (4 * k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l1949_194925


namespace NUMINAMATH_CALUDE_octahedron_construction_count_l1949_194956

/-- The number of faces in a regular octahedron -/
def octahedron_faces : ℕ := 8

/-- The number of distinct colored triangles available -/
def available_colors : ℕ := 9

/-- The number of rotational symmetries around a fixed face of an octahedron -/
def rotational_symmetries : ℕ := 3

/-- The number of distinguishable ways to construct a regular octahedron -/
def distinguishable_constructions : ℕ := 13440

theorem octahedron_construction_count :
  (Nat.choose available_colors (octahedron_faces - 1)) * 
  (Nat.factorial (octahedron_faces - 1)) / 
  rotational_symmetries = distinguishable_constructions := by
  sorry

end NUMINAMATH_CALUDE_octahedron_construction_count_l1949_194956


namespace NUMINAMATH_CALUDE_circle_center_l1949_194973

theorem circle_center (c : ℝ × ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    (∀ p : ℝ × ℝ, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 → 
      (3 * p.1 + 4 * p.2 = 24 ∨ 3 * p.1 + 4 * p.2 = -6))) ∧ 
  c.1 - 3 * c.2 = 0 → 
  c = (27/13, 9/13) := by
sorry


end NUMINAMATH_CALUDE_circle_center_l1949_194973


namespace NUMINAMATH_CALUDE_reachability_l1949_194965

/-- Number of positive integer divisors of n -/
def τ (n : ℕ) : ℕ := sorry

/-- Sum of positive integer divisors of n -/
def σ (n : ℕ) : ℕ := sorry

/-- Number of positive integers less than or equal to n that are relatively prime to n -/
def φ (n : ℕ) : ℕ := sorry

/-- Represents the operation of applying τ, σ, or φ -/
inductive Operation
| tau : Operation
| sigma : Operation
| phi : Operation

/-- Applies an operation to a natural number -/
def applyOperation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.tau => τ n
  | Operation.sigma => σ n
  | Operation.phi => φ n

/-- Theorem: For any two integers a and b greater than 1, 
    there exists a finite sequence of operations that transforms a into b -/
theorem reachability (a b : ℕ) (ha : a > 1) (hb : b > 1) : 
  ∃ (ops : List Operation), 
    (ops.foldl (fun n op => applyOperation op n) a) = b :=
sorry

end NUMINAMATH_CALUDE_reachability_l1949_194965


namespace NUMINAMATH_CALUDE_bernie_postcards_final_count_l1949_194941

/-- Calculates the number of postcards Bernie has after his transactions -/
def postcards_after_transactions (initial_postcards : ℕ) (sell_price : ℕ) (buy_price : ℕ) : ℕ :=
  let sold_postcards := initial_postcards / 2
  let remaining_postcards := initial_postcards - sold_postcards
  let money_earned := sold_postcards * sell_price
  let new_postcards := money_earned / buy_price
  remaining_postcards + new_postcards

/-- Theorem stating that Bernie will have 36 postcards after his transactions -/
theorem bernie_postcards_final_count :
  postcards_after_transactions 18 15 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bernie_postcards_final_count_l1949_194941


namespace NUMINAMATH_CALUDE_cylinder_radius_in_cone_l1949_194914

/-- 
Given a right circular cone with diameter 14 and altitude 16, and an inscribed right circular 
cylinder whose diameter equals its height, prove that the radius of the cylinder is 56/15.
-/
theorem cylinder_radius_in_cone (r : ℚ) : 
  (16 : ℚ) - 2 * r = (16 : ℚ) / 7 * r → r = 56 / 15 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_radius_in_cone_l1949_194914


namespace NUMINAMATH_CALUDE_magician_trick_l1949_194923

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem magician_trick (numbers : Finset ℕ) (a d : ℕ) :
  numbers = Finset.range 16 →
  a ∈ numbers →
  d ∈ numbers →
  is_even a →
  is_even d →
  ∃ (b c : ℕ), b ∈ numbers ∧ c ∈ numbers ∧
    (b < c ∨ (b > c ∧ a < d)) ∧
    (c < d ∨ (c > d ∧ b < a)) →
  a * d = 120 := by
  sorry

end NUMINAMATH_CALUDE_magician_trick_l1949_194923


namespace NUMINAMATH_CALUDE_three_digit_squares_ending_in_self_l1949_194976

theorem three_digit_squares_ending_in_self (A : ℕ) : 
  (100 ≤ A ∧ A < 1000) ∧ (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_squares_ending_in_self_l1949_194976


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l1949_194983

theorem cube_roots_of_unity :
  let z₁ : ℂ := 1
  let z₂ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  (z₁^3 = 1) ∧ (z₂^3 = 1) ∧ (z₃^3 = 1) ∧
  ∀ z : ℂ, z^3 = 1 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l1949_194983


namespace NUMINAMATH_CALUDE_remainder_twelve_pow_2012_mod_5_l1949_194971

theorem remainder_twelve_pow_2012_mod_5 : 12^2012 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_twelve_pow_2012_mod_5_l1949_194971


namespace NUMINAMATH_CALUDE_sports_equipment_pricing_and_purchasing_l1949_194985

theorem sports_equipment_pricing_and_purchasing (x y a b : ℤ) : 
  (2 * x + y = 330) →
  (5 * x + 2 * y = 780) →
  (120 * a + 90 * b = 810) →
  (x = 120 ∧ y = 90) ∧ (a = 3 ∧ b = 5) :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_pricing_and_purchasing_l1949_194985


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l1949_194994

/-- Given points O, A, B, C in a Cartesian coordinate system where O is the origin,
    prove that the ratio of the magnitudes of BC to AC is 3,
    given that OC is a weighted sum of OA and OB. -/
theorem vector_ratio_theorem (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  C - O = 3/4 • (A - O) + 1/4 • (B - O) →
  ‖C - B‖ / ‖C - A‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l1949_194994


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1949_194955

-- Problem 1
theorem problem_1 : (1 * (1/6 - 5/7 + 2/3)) * (-42) = -5 := by sorry

-- Problem 2
theorem problem_2 : -(2^2) + (-3)^2 * (-2/3) - 4^2 / |(-4)| = -14 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1949_194955


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1949_194987

/-- The number of ways to arrange math and history books on a shelf -/
def arrange_books (num_math_books num_history_books : ℕ) : ℕ :=
  let end_arrangements := num_math_books * (num_math_books - 1)
  let remaining_math_arrangements := 2  -- factorial of (num_math_books - 2)
  let history_distributions := (Nat.choose num_history_books 2) * 
                               (Nat.choose (num_history_books - 2) 2) *
                               2  -- Last 2 is automatic
  let history_permutations := (2 * 2 * 2)  -- 2! for each of the 3 slots
  end_arrangements * remaining_math_arrangements * history_distributions * history_permutations

/-- Theorem stating the number of ways to arrange the books -/
theorem book_arrangement_count :
  arrange_books 4 6 = 17280 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1949_194987


namespace NUMINAMATH_CALUDE_same_color_probability_is_71_288_l1949_194969

/-- Represents a 24-sided die with colored sides -/
structure ColoredDie :=
  (purple : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (sparkly : ℕ)
  (total : ℕ)
  (sum_sides : purple + green + blue + yellow + sparkly = total)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.purple^2 + d.green^2 + d.blue^2 + d.yellow^2 + d.sparkly^2) / d.total^2

/-- Our specific 24-sided die -/
def our_die : ColoredDie :=
  { purple := 5
  , green := 6
  , blue := 8
  , yellow := 4
  , sparkly := 1
  , total := 24
  , sum_sides := by rfl }

theorem same_color_probability_is_71_288 :
  same_color_probability our_die = 71 / 288 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_71_288_l1949_194969


namespace NUMINAMATH_CALUDE_soil_bags_needed_l1949_194992

/-- Calculates the number of soil bags needed for raised beds -/
theorem soil_bags_needed
  (num_beds : ℕ)
  (length width height : ℝ)
  (soil_per_bag : ℝ)
  (h_num_beds : num_beds = 2)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_soil_per_bag : soil_per_bag = 4) :
  ⌈(num_beds * length * width * height) / soil_per_bag⌉ = 16 := by
  sorry

end NUMINAMATH_CALUDE_soil_bags_needed_l1949_194992


namespace NUMINAMATH_CALUDE_inequality_theorem_l1949_194984

-- Define the inequality and its solution set
def inequality (m : ℝ) (x : ℝ) : Prop := m - |x - 2| ≥ 1
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | inequality m x}

-- Define the theorem
theorem inequality_theorem (m : ℝ) 
  (h1 : solution_set m = Set.Icc 0 4) 
  (a b : ℝ) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a + b = m) : 
  m = 3 ∧ ∃ (min : ℝ), min = 9/2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = m → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1949_194984


namespace NUMINAMATH_CALUDE_least_divisible_by_first_ten_l1949_194934

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_divisible_by_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ n) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ m) → n ≤ m) ∧ n = 2520 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_first_ten_l1949_194934


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l1949_194919

def rent_percent : ℝ := 20
def dishwasher_percent : ℝ := 15
def bills_percent : ℝ := 10
def car_percent : ℝ := 8
def grocery_percent : ℝ := 12
def tax_percent : ℝ := 5
def savings_percent : ℝ := 40

theorem dhoni_leftover_earnings : 
  let total_expenses := rent_percent + dishwasher_percent + bills_percent + car_percent + grocery_percent + tax_percent
  let remaining_after_expenses := 100 - total_expenses
  let savings := (savings_percent / 100) * remaining_after_expenses
  let leftover := remaining_after_expenses - savings
  leftover = 18 := by sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l1949_194919


namespace NUMINAMATH_CALUDE_basic_computer_price_l1949_194986

theorem basic_computer_price 
  (total_price : ℝ) 
  (price_difference : ℝ) 
  (printer_ratio : ℝ) :
  total_price = 2500 →
  price_difference = 500 →
  printer_ratio = 1/6 →
  ∃ (basic_price printer_price : ℝ),
    basic_price + printer_price = total_price ∧
    printer_price = printer_ratio * (basic_price + price_difference + printer_price) ∧
    basic_price = 2000 :=
by sorry

end NUMINAMATH_CALUDE_basic_computer_price_l1949_194986


namespace NUMINAMATH_CALUDE_dilation_circle_to_ellipse_l1949_194927

/-- Given a circle A and a dilation transformation, prove the equation of the resulting curve C -/
theorem dilation_circle_to_ellipse :
  let circle_A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let dilation (p : ℝ × ℝ) : ℝ × ℝ := (2 * p.1, 3 * p.2)
  let curve_C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 / 9 = 1}
  (∀ p ∈ circle_A, dilation p ∈ curve_C) ∧
  (∀ q ∈ curve_C, ∃ p ∈ circle_A, dilation p = q) := by
sorry

end NUMINAMATH_CALUDE_dilation_circle_to_ellipse_l1949_194927


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1949_194975

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1949_194975


namespace NUMINAMATH_CALUDE_equal_prob_without_mult_higher_prob_even_with_mult_l1949_194939

/-- Represents a calculator with basic operations -/
structure Calculator where
  /-- The current display value -/
  display : ℕ
  /-- Whether multiplication is available -/
  mult_available : Bool

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Get the parity of a natural number -/
def getParity (n : ℕ) : Parity :=
  if n % 2 = 0 then Parity.Even else Parity.Odd

/-- The probability of getting an odd result after a sequence of operations -/
def probOddResult (c : Calculator) : ℝ :=
  sorry

theorem equal_prob_without_mult (c : Calculator) (h : c.mult_available = false) :
  probOddResult c = 1 / 2 :=
sorry

theorem higher_prob_even_with_mult (c : Calculator) (h : c.mult_available = true) :
  probOddResult c < 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_equal_prob_without_mult_higher_prob_even_with_mult_l1949_194939


namespace NUMINAMATH_CALUDE_total_dozens_shipped_l1949_194981

-- Define the number of boxes shipped last week
def boxes_last_week : ℕ := 10

-- Define the total number of pomelos shipped last week
def total_pomelos_last_week : ℕ := 240

-- Define the number of boxes shipped this week
def boxes_this_week : ℕ := 20

-- Theorem to prove
theorem total_dozens_shipped : ℕ := by
  -- The proof goes here
  sorry

-- Goal: prove that total_dozens_shipped = 60
example : total_dozens_shipped = 60 := by sorry

end NUMINAMATH_CALUDE_total_dozens_shipped_l1949_194981


namespace NUMINAMATH_CALUDE_tomato_seeds_problem_l1949_194940

/-- Represents the number of tomato seeds planted by Mike in the morning -/
def mike_morning : ℕ := sorry

/-- Represents the number of tomato seeds planted by Ted in the morning -/
def ted_morning : ℕ := sorry

/-- Represents the number of tomato seeds planted by Mike in the afternoon -/
def mike_afternoon : ℕ := 60

/-- Represents the number of tomato seeds planted by Ted in the afternoon -/
def ted_afternoon : ℕ := sorry

theorem tomato_seeds_problem :
  ted_morning = 2 * mike_morning ∧
  ted_afternoon = mike_afternoon - 20 ∧
  mike_morning + ted_morning + mike_afternoon + ted_afternoon = 250 →
  mike_morning = 50 := by sorry

end NUMINAMATH_CALUDE_tomato_seeds_problem_l1949_194940


namespace NUMINAMATH_CALUDE_river_distance_l1949_194943

theorem river_distance (d : ℝ) : 
  (¬(d ≤ 12)) → (¬(d ≥ 15)) → (¬(d ≥ 10)) → (12 < d ∧ d < 15) := by
  sorry

end NUMINAMATH_CALUDE_river_distance_l1949_194943


namespace NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l1949_194909

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for equilateral triangles
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = t2.side_length

-- Theorem statement
theorem not_all_equilateral_triangles_congruent :
  ∃ (t1 t2 : EquilateralTriangle), ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_not_all_equilateral_triangles_congruent_l1949_194909


namespace NUMINAMATH_CALUDE_tuesday_total_counts_l1949_194926

/-- Represents the number of times Carla counted tiles on Tuesday -/
def tile_counts : Nat := 2

/-- Represents the number of times Carla counted books on Tuesday -/
def book_counts : Nat := 3

/-- Theorem stating that the total number of counts on Tuesday is 5 -/
theorem tuesday_total_counts : tile_counts + book_counts = 5 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_total_counts_l1949_194926


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1949_194962

theorem stratified_sampling_theorem (total_sample : ℕ) 
  (school_A : ℕ) (school_B : ℕ) (school_C : ℕ) : 
  total_sample = 60 → 
  school_A = 180 → 
  school_B = 270 → 
  school_C = 90 → 
  (school_C * total_sample) / (school_A + school_B + school_C) = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1949_194962


namespace NUMINAMATH_CALUDE_problem_solution_l1949_194903

-- Define the equation
def equation (m x : ℝ) : ℝ := x^2 + m*x + 2*m + 5

-- Define the set A
def set_A : Set ℝ := {m : ℝ | ∀ x : ℝ, equation m x ≠ 0 ∨ ∃ y : ℝ, y ≠ x ∧ equation m x = 0 ∧ equation m y = 0}

-- Define the set B
def set_B (a : ℝ) : Set ℝ := {x : ℝ | 1 - 2*a ≤ x ∧ x ≤ a - 1}

theorem problem_solution :
  (∀ m : ℝ, m ∈ set_A ↔ -2 ≤ m ∧ m ≤ 10) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ set_A → x ∈ set_B a) ∧ (∃ x : ℝ, x ∈ set_B a ∧ x ∉ set_A) ↔ 11 ≤ a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1949_194903


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_five_l1949_194905

theorem remainder_sum_powers_mod_five :
  (Nat.pow 9 5 + Nat.pow 8 7 + Nat.pow 7 6) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_five_l1949_194905


namespace NUMINAMATH_CALUDE_perfect_squares_difference_l1949_194997

theorem perfect_squares_difference (n : ℕ) : 
  (∃ a : ℕ, n - 52 = a^2) ∧ (∃ b : ℕ, n + 37 = b^2) → n = 1988 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_l1949_194997


namespace NUMINAMATH_CALUDE_length_of_AE_l1949_194937

/-- Given four points A, B, C, D on a 2D plane, and E as the intersection of segments AB and CD,
    prove that the length of AE is 5√5/3. -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 3) →
  B = (6, 0) →
  C = (4, 2) →
  D = (2, 0) →
  E.1 = 10/3 →
  E.2 = 4/3 →
  (E.2 - A.2) / (E.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) →
  (E.2 - C.2) / (E.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 5 * Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AE_l1949_194937


namespace NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l1949_194964

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 10 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 12 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_reciprocal_l1949_194964


namespace NUMINAMATH_CALUDE_tv_cost_l1949_194906

theorem tv_cost (original_savings : ℝ) (furniture_fraction : ℚ) (tv_cost : ℝ) : 
  original_savings = 3000.0000000000005 →
  furniture_fraction = 5/6 →
  tv_cost = original_savings * (1 - furniture_fraction) →
  tv_cost = 500.0000000000001 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l1949_194906


namespace NUMINAMATH_CALUDE_min_y_is_e_l1949_194996

-- Define the function representing the given equation
def f (x y : ℝ) : Prop := Real.exp x = y * Real.log x + y * Real.log y

-- Theorem stating the minimum value of y
theorem min_y_is_e :
  ∃ (y_min : ℝ), y_min = Real.exp 1 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → f x y → y ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_is_e_l1949_194996


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l1949_194947

/-- Proposition p: For any x ∈ ℝ, x² - 2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: There exists x ∈ ℝ such that x² + 2ax + 2 - a = 0 -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := { a : ℝ | (a > -2 ∧ a < -1) ∨ a ≥ 1 }

theorem range_of_a_theorem (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_of_a := by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l1949_194947


namespace NUMINAMATH_CALUDE_same_speed_problem_l1949_194995

theorem same_speed_problem (x : ℝ) : 
  let jack_speed := x^2 - 11*x - 22
  let jill_distance := x^2 - 5*x - 60
  let jill_time := x + 6
  let jill_speed := jill_distance / jill_time
  jack_speed = jill_speed → jack_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_same_speed_problem_l1949_194995


namespace NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l1949_194970

theorem sequence_with_positive_triples_negative_sum : 
  ∃ (seq : Fin 20 → ℝ), 
    (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
    (Finset.sum Finset.univ seq < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_positive_triples_negative_sum_l1949_194970


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1949_194952

theorem max_value_trig_expression :
  ∀ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 ∧
  ∃ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1949_194952


namespace NUMINAMATH_CALUDE_even_function_shift_l1949_194972

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is monotonically increasing on an interval (a,b) if
    for all x, y in (a,b), x < y implies f(x) < f(y) -/
def MonoIncOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- A function has a symmetry axis at x = k if f(k + x) = f(k - x) for all x -/
def HasSymmetryAxis (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x, f (k + x) = f (k - x)

theorem even_function_shift (f : ℝ → ℝ) :
    IsEven f →
    MonoIncOn f 3 5 →
    HasSymmetryAxis (fun x ↦ f (x - 1)) 1 ∧
    MonoIncOn (fun x ↦ f (x - 1)) 4 6 := by
  sorry

end NUMINAMATH_CALUDE_even_function_shift_l1949_194972


namespace NUMINAMATH_CALUDE_profit_percentage_is_36_percent_l1949_194963

def selling_price : ℝ := 850
def profit : ℝ := 225

theorem profit_percentage_is_36_percent :
  (profit / (selling_price - profit)) * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_36_percent_l1949_194963


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l1949_194949

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- The sum of AB and CD is 300
  sum_sides : ab + cd = 300
  -- The ratio of areas is 5:4
  ratio_condition : area_ratio = 5 / 4

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC 
to the area of triangle ADC is 5:4, and AB + CD = 300 cm, 
then AB = 500/3 cm.
-/
theorem trapezoid_side_length (t : Trapezoid) : t.ab = 500 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l1949_194949


namespace NUMINAMATH_CALUDE_f_derivative_sum_l1949_194931

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- State the theorem
theorem f_derivative_sum (h1 : ∀ x, HasDerivAt f (f' x) x)
                         (h2 : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_sum_l1949_194931


namespace NUMINAMATH_CALUDE_remainder_of_second_division_l1949_194900

def p (x : ℝ) : ℝ := x^6 - 4*x^5 + 6*x^4 - 4*x^3 + x^2

def s1 (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2

def t1 : ℝ := p 1

def t2 : ℝ := s1 1

theorem remainder_of_second_division (x : ℝ) : t2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_second_division_l1949_194900


namespace NUMINAMATH_CALUDE_max_value_of_f_l1949_194920

def f (x : ℝ) : ℝ := -x^2 + 2*x + 8

theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1949_194920


namespace NUMINAMATH_CALUDE_strictly_decreasing_function_l1949_194945

/-- A function satisfying the given condition -/
noncomputable def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, ∃ (S : Finset ℝ), ∀ y ∈ S, y > 0 ∧ (x + f y) * (y + f x) ≤ 4

/-- The main theorem -/
theorem strictly_decreasing_function 
  (f : ℝ → ℝ) (h : SatisfiesCondition f) :
  ∀ x y, 0 < x ∧ x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_strictly_decreasing_function_l1949_194945


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1949_194936

theorem diophantine_equation_solution (x y z : ℤ) : x^2 + y^2 = 3*z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1949_194936


namespace NUMINAMATH_CALUDE_carpet_width_calculation_l1949_194990

theorem carpet_width_calculation (room_length room_width carpet_cost_per_sqm total_cost : ℝ) 
  (h1 : room_length = 13)
  (h2 : room_width = 9)
  (h3 : carpet_cost_per_sqm = 12)
  (h4 : total_cost = 1872) : 
  (total_cost / carpet_cost_per_sqm / room_length) * 100 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_calculation_l1949_194990


namespace NUMINAMATH_CALUDE_preimage_of_20_l1949_194902

def f (n : ℕ) : ℕ := 2^n + n

theorem preimage_of_20 : ∃! n : ℕ, f n = 20 ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_preimage_of_20_l1949_194902


namespace NUMINAMATH_CALUDE_expression_equality_l1949_194953

theorem expression_equality : 
  (5 + 8) * (5^2 + 8^2) * (5^4 + 8^4) * (5^8 + 8^8) * (5^16 + 8^16) * (5^32 + 8^32) = 8^32 - 5^32 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1949_194953


namespace NUMINAMATH_CALUDE_thanksgiving_to_christmas_l1949_194907

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by a given number of days
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDays d m)

theorem thanksgiving_to_christmas (thanksgiving : DayOfWeek) :
  thanksgiving = DayOfWeek.Thursday →
  advanceDays thanksgiving 29 = DayOfWeek.Friday :=
by sorry

#check thanksgiving_to_christmas

end NUMINAMATH_CALUDE_thanksgiving_to_christmas_l1949_194907


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1949_194908

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x * Real.exp x > 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ * Real.exp x₀ ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1949_194908


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l1949_194988

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 7 + a 13 = 4 * Real.pi) :
  Real.sin (a 2 + a 12) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_sum_l1949_194988


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1949_194924

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 999 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1949_194924


namespace NUMINAMATH_CALUDE_solution_difference_l1949_194948

theorem solution_difference (x : ℝ) : 
  (∃ y : ℝ, (7 - y^2 / 4)^(1/3) = -3 ∧ y ≠ x ∧ (7 - x^2 / 4)^(1/3) = -3) → 
  |x - y| = 2 * Real.sqrt 136 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l1949_194948


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1949_194916

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 10 → x ≠ -5 →
  (8 * x - 3) / (x^2 - 5*x - 50) = (77/15) / (x - 10) + (43/15) / (x + 5) :=
by
  sorry

#check partial_fraction_decomposition

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1949_194916


namespace NUMINAMATH_CALUDE_min_games_for_condition_l1949_194980

/-- The number of teams in the tournament -/
def num_teams : ℕ := 16

/-- The total number of possible games in a round-robin tournament -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- The maximum number of non-played games such that no three teams are mutually non-played -/
def max_non_played_games : ℕ := (num_teams / 2) ^ 2

/-- The minimum number of games that must be played to satisfy the condition -/
def min_games_played : ℕ := total_games - max_non_played_games

theorem min_games_for_condition : min_games_played = 56 := by sorry

end NUMINAMATH_CALUDE_min_games_for_condition_l1949_194980


namespace NUMINAMATH_CALUDE_tower_combinations_l1949_194928

/-- The number of different towers of height 7 that can be built using 3 red cubes, 4 blue cubes, and 2 yellow cubes -/
def num_towers : ℕ := 5040

/-- The height of the tower -/
def tower_height : ℕ := 7

/-- The number of red cubes -/
def red_cubes : ℕ := 3

/-- The number of blue cubes -/
def blue_cubes : ℕ := 4

/-- The number of yellow cubes -/
def yellow_cubes : ℕ := 2

/-- The total number of cubes -/
def total_cubes : ℕ := red_cubes + blue_cubes + yellow_cubes

theorem tower_combinations : num_towers = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tower_combinations_l1949_194928


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l1949_194957

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(p ∣ n)

theorem smallest_non_prime_non_square_with_large_factors : 
  (∀ m : ℕ, m < 4087 → 
    is_prime m ∨ 
    is_square m ∨ 
    ¬(has_no_prime_factor_less_than m 60)) ∧ 
  ¬(is_prime 4087) ∧ 
  ¬(is_square 4087) ∧ 
  has_no_prime_factor_less_than 4087 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l1949_194957


namespace NUMINAMATH_CALUDE_linear_independence_exp_trig_l1949_194944

theorem linear_independence_exp_trig (α β : ℝ) (h : β ≠ 0) :
  ∀ (α₁ α₂ : ℝ), (∀ x : ℝ, α₁ * Real.exp (α * x) * Real.sin (β * x) + 
                           α₂ * Real.exp (α * x) * Real.cos (β * x) = 0) →
                 α₁ = 0 ∧ α₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_independence_exp_trig_l1949_194944


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1949_194910

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem intersection_point_sum :
  ∃ (a b : ℝ), f a = f (a - 4) ∧ a + b = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1949_194910


namespace NUMINAMATH_CALUDE_expression_equality_l1949_194959

theorem expression_equality : |Real.sqrt 2 - 1| - (π + 1)^0 + Real.sqrt ((-3)^2) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1949_194959


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_self_l1949_194989

theorem product_of_fractions_equals_self (n : ℝ) (h : n > 0) : 
  n = (4/5 * n) * (5/6 * n) → n = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_self_l1949_194989


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1949_194961

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1949_194961


namespace NUMINAMATH_CALUDE_max_cables_for_given_network_l1949_194967

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  brand_a : ℕ
  brand_b : ℕ

/-- Represents the number of cables in the network. -/
def cables (n : ComputerNetwork) : ℕ := sorry

/-- Checks if all computers in the network can communicate. -/
def all_communicate (n : ComputerNetwork) : Prop := sorry

/-- The maximum number of cables needed for full communication. -/
def max_cables (n : ComputerNetwork) : ℕ := sorry

/-- Theorem stating the maximum number of cables for the given network. -/
theorem max_cables_for_given_network :
  ∀ (n : ComputerNetwork),
    n.brand_a = 20 ∧ n.brand_b = 20 →
    max_cables n = 20 ∧ all_communicate n := by sorry

end NUMINAMATH_CALUDE_max_cables_for_given_network_l1949_194967


namespace NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l1949_194966

/-- The number of permits Andrew stamps in a day given his schedule and stamping rate -/
def permits_stamped (appointments : ℕ) (appointment_duration : ℕ) (workday_hours : ℕ) (stamping_rate : ℕ) : ℕ :=
  let total_appointment_hours := appointments * appointment_duration
  let stamping_hours := workday_hours - total_appointment_hours
  stamping_hours * stamping_rate

/-- Theorem stating that Andrew stamps 100 permits given his specific schedule and stamping rate -/
theorem andrew_stamps_hundred_permits :
  permits_stamped 2 3 8 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_andrew_stamps_hundred_permits_l1949_194966


namespace NUMINAMATH_CALUDE_a_minus_b_value_l1949_194942

theorem a_minus_b_value (a b : ℚ) 
  (ha : |a| = 5) 
  (hb : |b| = 2) 
  (hab : |a + b| = a + b) : 
  a - b = 3 ∨ a - b = 7 :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l1949_194942


namespace NUMINAMATH_CALUDE_mary_hourly_wage_l1949_194930

def mary_long_day_hours : ℕ := 9
def mary_short_day_hours : ℕ := 5
def mary_long_days_per_week : ℕ := 3
def mary_short_days_per_week : ℕ := 2
def mary_weekly_earnings : ℕ := 407

def mary_total_weekly_hours : ℕ :=
  mary_long_day_hours * mary_long_days_per_week +
  mary_short_day_hours * mary_short_days_per_week

theorem mary_hourly_wage :
  mary_weekly_earnings / mary_total_weekly_hours = 11 := by
  sorry

end NUMINAMATH_CALUDE_mary_hourly_wage_l1949_194930


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1949_194929

theorem profit_percent_calculation (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.8 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l1949_194929


namespace NUMINAMATH_CALUDE_inversion_property_l1949_194954

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point in 2D plane -/
def Point := ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Inversion of a point with respect to a circle -/
def inversion (c : Circle) (p : Point) : Point := sorry

/-- Theorem: Inversion property -/
theorem inversion_property (c : Circle) (p p' : Point) : 
  p' = inversion c p → 
  distance c.center p * distance c.center p' = c.radius ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inversion_property_l1949_194954


namespace NUMINAMATH_CALUDE_consecutive_cube_divisible_l1949_194904

theorem consecutive_cube_divisible (k : ℕ+) :
  ∃ n : ℤ, ∀ j : ℕ, j ∈ Finset.range k →
    ∃ m : ℕ, m > 1 ∧ (n + j : ℤ) % (m^3 : ℤ) = 0 :=
sorry

end NUMINAMATH_CALUDE_consecutive_cube_divisible_l1949_194904


namespace NUMINAMATH_CALUDE_car_distance_l1949_194913

theorem car_distance (time : ℝ) (cyclist_distance : ℝ) (speed_difference : ℝ) :
  time = 8 →
  cyclist_distance = 88 →
  speed_difference = 5 →
  let cyclist_speed := cyclist_distance / time
  let car_speed := cyclist_speed + speed_difference
  car_speed * time = 128 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l1949_194913


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1949_194978

/-- An isosceles triangle with given height and median -/
structure IsoscelesTriangle where
  -- Height from the base to the vertex
  height : ℝ
  -- Median from a leg to the midpoint of the base
  median : ℝ

/-- The area of an isosceles triangle given its height and median -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem: The area of an isosceles triangle with height 18 and median 15 is 144 -/
theorem isosceles_triangle_area :
  let t : IsoscelesTriangle := { height := 18, median := 15 }
  area t = 144 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1949_194978


namespace NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l1949_194922

/-- Represents a clock with minute hands moving in opposite directions -/
structure Clock where
  coincidences : ℕ  -- Number of coincidences in an hour
  handsForward : ℕ  -- Number of hands moving forward
  handsBackward : ℕ -- Number of hands moving backward

/-- The total number of hands on the clock -/
def Clock.totalHands (c : Clock) : ℕ := c.handsForward + c.handsBackward

/-- Predicate to check if the clock configuration is valid -/
def Clock.isValid (c : Clock) : Prop :=
  c.handsForward * c.handsBackward * 2 = c.coincidences

/-- Theorem stating the maximum number of hands for a clock with 54 coincidences -/
theorem max_hands_for_54_coincidences :
  ∃ (c : Clock), c.coincidences = 54 ∧ c.isValid ∧
  ∀ (d : Clock), d.coincidences = 54 → d.isValid → d.totalHands ≤ c.totalHands :=
by
  sorry

end NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l1949_194922


namespace NUMINAMATH_CALUDE_stripe_length_on_cylinder_l1949_194950

/-- Proves that the length of a diagonal line on a rectangle with sides 30 inches and 16 inches is 34 inches. -/
theorem stripe_length_on_cylinder (circumference height : ℝ) (h1 : circumference = 30) (h2 : height = 16) :
  Real.sqrt (circumference^2 + height^2) = 34 :=
by sorry

end NUMINAMATH_CALUDE_stripe_length_on_cylinder_l1949_194950


namespace NUMINAMATH_CALUDE_calzone_time_is_124_l1949_194912

/-- The total time spent on making calzones -/
def total_calzone_time (onion_time garlic_pepper_time knead_time rest_time assemble_time : ℕ) : ℕ :=
  onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time

/-- Theorem stating the total time spent on making calzones is 124 minutes -/
theorem calzone_time_is_124 : 
  ∀ (onion_time garlic_pepper_time knead_time rest_time assemble_time : ℕ),
    onion_time = 20 →
    garlic_pepper_time = onion_time / 4 →
    knead_time = 30 →
    rest_time = 2 * knead_time →
    assemble_time = (knead_time + rest_time) / 10 →
    total_calzone_time onion_time garlic_pepper_time knead_time rest_time assemble_time = 124 :=
by
  sorry


end NUMINAMATH_CALUDE_calzone_time_is_124_l1949_194912


namespace NUMINAMATH_CALUDE_square_division_theorem_l1949_194932

theorem square_division_theorem (x : ℝ) (h1 : x > 0) :
  (∃ l : ℝ, l > 0 ∧ 2 * l = x^2 / 5) →
  (∃ a : ℝ, a > 0 ∧ x * a = x^2 / 5) →
  x = 8 ∧ x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_division_theorem_l1949_194932


namespace NUMINAMATH_CALUDE_sum_of_children_ages_l1949_194968

/-- Represents the ages of Cynthia's children -/
structure ChildrenAges where
  freddy : ℕ
  matthew : ℕ
  rebecca : ℕ

/-- Theorem stating the sum of Cynthia's children's ages -/
theorem sum_of_children_ages (ages : ChildrenAges) : 
  ages.freddy = 15 → 
  ages.matthew = ages.freddy - 4 → 
  ages.rebecca = ages.matthew - 2 → 
  ages.freddy + ages.matthew + ages.rebecca = 35 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_children_ages_l1949_194968


namespace NUMINAMATH_CALUDE_probability_square_or_circle_l1949_194998

/- Define the total number of figures -/
def total_figures : ℕ := 10

/- Define the number of squares -/
def num_squares : ℕ := 4

/- Define the number of circles -/
def num_circles : ℕ := 3

/- Theorem statement -/
theorem probability_square_or_circle :
  (num_squares + num_circles : ℚ) / total_figures = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_circle_l1949_194998


namespace NUMINAMATH_CALUDE_first_rope_length_l1949_194915

/-- Represents the lengths of ropes Tony found -/
structure Ropes where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Calculates the total length of ropes before tying -/
def total_length (r : Ropes) : ℝ :=
  r.first + r.second + r.third + r.fourth + r.fifth

/-- Calculates the length lost due to knots -/
def knot_loss (num_ropes : ℕ) (loss_per_knot : ℝ) : ℝ :=
  (num_ropes - 1 : ℝ) * loss_per_knot

/-- Theorem stating that given the conditions, the first rope Tony found is 20 feet long -/
theorem first_rope_length
  (r : Ropes)
  (h1 : r.second = 2)
  (h2 : r.third = 2)
  (h3 : r.fourth = 2)
  (h4 : r.fifth = 7)
  (h5 : total_length r - knot_loss 5 1.2 = 35) :
  r.first = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_rope_length_l1949_194915


namespace NUMINAMATH_CALUDE_overlapping_triangles_angle_sum_l1949_194901

/-- Given two overlapping triangles ABC and DEF where B and E are the same point,
    prove that the sum of angles A, B, C, D, and F is 290 degrees. -/
theorem overlapping_triangles_angle_sum
  (A B C D F : Real)
  (h1 : A = 40)
  (h2 : C = 70)
  (h3 : D = 50)
  (h4 : F = 60)
  (h5 : A + B + C = 180)  -- Sum of angles in triangle ABC
  (h6 : D + B + F = 180)  -- Sum of angles in triangle DEF (B is used instead of E)
  : A + B + C + D + F = 290 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_triangles_angle_sum_l1949_194901


namespace NUMINAMATH_CALUDE_stadium_fee_difference_l1949_194951

def stadium_capacity : ℕ := 2000
def entry_fee : ℕ := 20

theorem stadium_fee_difference :
  let full_capacity := stadium_capacity
  let partial_capacity := (3 * stadium_capacity) / 4
  let full_fees := full_capacity * entry_fee
  let partial_fees := partial_capacity * entry_fee
  full_fees - partial_fees = 10000 := by
sorry

end NUMINAMATH_CALUDE_stadium_fee_difference_l1949_194951


namespace NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l1949_194935

/-- Represents the duration of a workday in minutes -/
def workday_duration : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_duration : ℕ := 2 * first_meeting_duration

/-- Represents the total duration of both meetings in minutes -/
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

/-- Represents the percentage of the workday spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_duration : ℚ) / (workday_duration : ℚ) * 100

theorem meeting_percentage_is_37_5 : meeting_percentage = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l1949_194935


namespace NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l1949_194917

theorem sweets_neither_red_nor_green 
  (total : ℕ) 
  (red : ℕ) 
  (green : ℕ) 
  (h_total : total = 285) 
  (h_red : red = 49) 
  (h_green : green = 59) : 
  total - (red + green) = 177 := by
sorry

end NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l1949_194917


namespace NUMINAMATH_CALUDE_negative_sqrt_17_bound_l1949_194991

theorem negative_sqrt_17_bound : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_17_bound_l1949_194991


namespace NUMINAMATH_CALUDE_jasons_pepper_spray_dilemma_l1949_194946

theorem jasons_pepper_spray_dilemma :
  ¬ ∃ (raccoons squirrels opossums : ℕ),
    squirrels = 6 * raccoons ∧
    opossums = 2 * raccoons ∧
    raccoons + squirrels + opossums = 168 :=
by sorry

end NUMINAMATH_CALUDE_jasons_pepper_spray_dilemma_l1949_194946


namespace NUMINAMATH_CALUDE_sqrt_of_squared_negative_l1949_194938

theorem sqrt_of_squared_negative : Real.sqrt ((-5)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_squared_negative_l1949_194938


namespace NUMINAMATH_CALUDE_line_parameterization_l1949_194974

/-- Given a line y = -3x + 2 parameterized as [x; y] = [5; r] + t[k; 8], prove r = -13 and k = -4 -/
theorem line_parameterization (r k : ℝ) : 
  (∀ x y t : ℝ, y = -3 * x + 2 ↔ ∃ t, (x, y) = (5 + t * k, r + t * 8)) →
  r = -13 ∧ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l1949_194974


namespace NUMINAMATH_CALUDE_function_equality_l1949_194958

theorem function_equality (x : ℝ) (h : x > 0) : 
  (Real.sqrt x)^2 / x = x / (Real.sqrt x)^2 ∧ 
  (Real.sqrt x)^2 / x = 1 ∧ 
  x / (Real.sqrt x)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_equality_l1949_194958


namespace NUMINAMATH_CALUDE_sin_35pi_over_6_l1949_194999

theorem sin_35pi_over_6 : Real.sin (35 * π / 6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_35pi_over_6_l1949_194999


namespace NUMINAMATH_CALUDE_shaded_area_is_16_l1949_194921

/-- Represents the shaded area of a 6x6 grid with triangles and trapezoids -/
def shadedArea (gridSize : Nat) (triangleCount : Nat) (trapezoidCount : Nat) 
  (triangleSquares : Nat) (trapezoidSquares : Nat) : Nat :=
  triangleCount * triangleSquares + trapezoidCount * trapezoidSquares

/-- Theorem stating that the shaded area of the described grid is 16 square units -/
theorem shaded_area_is_16 : 
  shadedArea 6 2 4 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_16_l1949_194921


namespace NUMINAMATH_CALUDE_quadratic_roots_l1949_194982

theorem quadratic_roots (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 6*x + c = 0 ↔ x = (-3 + Real.sqrt c) ∨ x = (-3 - Real.sqrt c)) → 
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1949_194982
