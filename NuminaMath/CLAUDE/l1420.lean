import Mathlib

namespace quadratic_inequality_necessary_condition_quadratic_inequality_not_sufficient_l1420_142028

theorem quadratic_inequality_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → a ≥ 0 :=
by sorry

theorem quadratic_inequality_not_sufficient (a : ℝ) :
  a ≥ 0 → ¬(∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by sorry

end quadratic_inequality_necessary_condition_quadratic_inequality_not_sufficient_l1420_142028


namespace candidate_votes_l1420_142062

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 70 / 100 →
  ⌊(1 - invalid_percent) * candidate_percent * total_votes⌋ = 333200 :=
by sorry

end candidate_votes_l1420_142062


namespace least_n_triple_f_not_one_digit_l1420_142064

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Function f as defined in the problem -/
def f (n : ℕ) : ℕ := sumOfDigits n

/-- Theorem stating that 19999999999999999999999 is the least natural number n 
    such that f(f(f(n))) is not a one-digit number -/
theorem least_n_triple_f_not_one_digit :
  ∀ k : ℕ, k < 19999999999999999999999 → f (f (f k)) < 10 ∧ 
  f (f (f 19999999999999999999999)) ≥ 10 :=
sorry

end least_n_triple_f_not_one_digit_l1420_142064


namespace pillar_height_D_equals_19_l1420_142034

/-- Regular hexagon with pillars -/
structure HexagonWithPillars where
  sideLength : ℝ
  pillarHeightA : ℝ
  pillarHeightB : ℝ
  pillarHeightC : ℝ

/-- Calculate the height of pillar D given a hexagon with pillars -/
def pillarHeightD (h : HexagonWithPillars) : ℝ :=
  sorry

/-- Theorem stating the height of pillar D for the given hexagon -/
theorem pillar_height_D_equals_19 (h : HexagonWithPillars) 
  (h_side : h.sideLength = 10)
  (h_A : h.pillarHeightA = 15)
  (h_B : h.pillarHeightB = 12)
  (h_C : h.pillarHeightC = 11) :
  pillarHeightD h = 19 :=
sorry

end pillar_height_D_equals_19_l1420_142034


namespace max_a_for_same_range_l1420_142008

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1 / Real.exp 1) * Real.exp x + (a / 2) * x^2 - (a + 1) * x + a

theorem max_a_for_same_range : 
  ∃ (a_max : ℝ), a_max > 0 ∧ 
  (∀ (a : ℝ), a > 0 → 
    (Set.range (f a) = Set.range (fun x => f a (f a x))) → 
    a ≤ a_max) ∧
  (Set.range (f a_max) = Set.range (fun x => f a_max (f a_max x))) ∧
  a_max = 2 := by
sorry

end max_a_for_same_range_l1420_142008


namespace expression_value_l1420_142051

theorem expression_value : (5^2 - 5) * (6^2 - 6) - (7^2 - 7) = 558 := by
  sorry

end expression_value_l1420_142051


namespace sum_of_angles_octagon_pentagon_l1420_142052

theorem sum_of_angles_octagon_pentagon : 
  ∀ (octagon_angle pentagon_angle : ℝ),
  (octagon_angle = 180 * (8 - 2) / 8) →
  (pentagon_angle = 180 * (5 - 2) / 5) →
  octagon_angle + pentagon_angle = 243 :=
by sorry

end sum_of_angles_octagon_pentagon_l1420_142052


namespace midpoint_complex_l1420_142046

theorem midpoint_complex (z₁ z₂ : ℂ) (h₁ : z₁ = 2 + I) (h₂ : z₂ = 4 - 3*I) :
  (z₁ + z₂) / 2 = 3 - I := by
  sorry

end midpoint_complex_l1420_142046


namespace chemical_B_calculation_l1420_142003

/-- The amount of chemical B needed to create 1 liter of solution -/
def chemical_B_needed : ℚ :=
  2/3

/-- The amount of chemical B in the original mixture -/
def original_chemical_B : ℚ :=
  0.08

/-- The amount of water in the original mixture -/
def original_water : ℚ :=
  0.04

/-- The total amount of solution in the original mixture -/
def original_total : ℚ :=
  0.12

/-- The target amount of solution to be created -/
def target_amount : ℚ :=
  1

theorem chemical_B_calculation :
  original_chemical_B / original_total * target_amount = chemical_B_needed :=
by sorry

end chemical_B_calculation_l1420_142003


namespace rebecca_earrings_gemstones_l1420_142063

/-- Calculates the number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let magnets_per_earring : ℕ := 2
  let buttons_per_earring : ℕ := magnets_per_earring / 2
  let gemstones_per_earring : ℕ := buttons_per_earring * 3
  let earrings_per_set : ℕ := 2
  num_sets * earrings_per_set * gemstones_per_earring

theorem rebecca_earrings_gemstones :
  gemstones_needed 4 = 24 := by
  sorry

end rebecca_earrings_gemstones_l1420_142063


namespace unique_divisible_digit_l1420_142094

def number (A : Nat) : Nat := 353809 * 10 + A

theorem unique_divisible_digit :
  ∃! (A : Nat), A < 10 ∧ (number A).mod 5 = 0 ∧ (number A).mod 11 = 0 :=
by sorry

end unique_divisible_digit_l1420_142094


namespace beri_always_wins_l1420_142068

/-- Represents a strategy for choosing b given a. -/
def Strategy : Type := ℕ → ℕ

/-- Checks if a trinomial x^2 - px + q has integer solutions. -/
def has_integer_solutions (p q : ℕ) : Prop :=
  ∃ k : ℕ, p * p - 4 * q = k * k

/-- The main theorem stating that Beri can always win. -/
theorem beri_always_wins :
  ∃ (strategy : Strategy),
    ∀ a : ℕ,
      1 ≤ a → a ≤ 2020 →
        let b := strategy a
        1 ≤ b → b ≤ 2020 → b ≠ a →
          (has_integer_solutions a b ∨ has_integer_solutions b a) :=
sorry

end beri_always_wins_l1420_142068


namespace tourist_group_size_is_five_l1420_142097

/-- Represents the number of people in a tourist group satisfying given rooming conditions. -/
def tourist_group_size : ℕ :=
  let large_room_capacity : ℕ := 3
  let small_room_capacity : ℕ := 2
  let small_rooms_rented : ℕ := 1
  let total_people : ℕ := 5

  total_people

theorem tourist_group_size_is_five :
  let large_room_capacity : ℕ := 3
  let small_room_capacity : ℕ := 2
  let small_rooms_rented : ℕ := 1
  tourist_group_size = 5 ∧
  tourist_group_size % large_room_capacity = small_room_capacity ∧
  tourist_group_size ≥ small_room_capacity * small_rooms_rented :=
by
  sorry

#eval tourist_group_size

end tourist_group_size_is_five_l1420_142097


namespace inequality_proof_l1420_142089

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l1420_142089


namespace cone_lateral_surface_area_l1420_142065

/-- The lateral surface area of a cone with base area 25π and height 12 is 65π. -/
theorem cone_lateral_surface_area :
  ∀ (base_area height radius slant_height lateral_area : ℝ),
  base_area = 25 * Real.pi →
  height = 12 →
  base_area = Real.pi * radius^2 →
  slant_height^2 = radius^2 + height^2 →
  lateral_area = Real.pi * radius * slant_height →
  lateral_area = 65 * Real.pi := by
  sorry


end cone_lateral_surface_area_l1420_142065


namespace fourth_vertex_exists_l1420_142026

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a quadrilateral is inscribed in a circle -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is circumscribed around a circle -/
def isCircumscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem fourth_vertex_exists (A B C : Point) 
  (h_inscribed : isInscribed ⟨A, B, C, sorry⟩)
  (h_circumscribed : isCircumscribed ⟨A, B, C, sorry⟩)
  (h_AB_ge_BC : distance A B ≥ distance B C) :
  ∃ (D : Point), 
    isInscribed ⟨A, B, C, D⟩ ∧ 
    isCircumscribed ⟨A, B, C, D⟩ :=
by
  sorry

#check fourth_vertex_exists

end fourth_vertex_exists_l1420_142026


namespace base3_to_decimal_10101_l1420_142096

/-- Converts a base 3 number represented as a list of digits to its decimal (base 10) equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The base 3 representation of the number we want to convert -/
def base3Number : List Nat := [1, 0, 1, 0, 1]

/-- Theorem stating that the base 3 number 10101 is equal to 91 in base 10 -/
theorem base3_to_decimal_10101 :
  base3ToDecimal base3Number = 91 := by
  sorry

#eval base3ToDecimal base3Number -- This should output 91

end base3_to_decimal_10101_l1420_142096


namespace smallest_number_divisibility_l1420_142057

theorem smallest_number_divisibility (x : ℕ) : x = 6297 ↔ 
  (∀ y : ℕ, (y + 3) % 18 = 0 ∧ (y + 3) % 70 = 0 ∧ (y + 3) % 100 = 0 ∧ (y + 3) % 84 = 0 → y ≥ x) ∧
  (x + 3) % 18 = 0 ∧ (x + 3) % 70 = 0 ∧ (x + 3) % 100 = 0 ∧ (x + 3) % 84 = 0 :=
by sorry

end smallest_number_divisibility_l1420_142057


namespace student_allowance_l1420_142056

/-- Proves that the student's weekly allowance is $3.00 given the spending pattern described. -/
theorem student_allowance (allowance : ℝ) : 
  (2/5 : ℝ) * allowance + 
  (1/3 : ℝ) * ((3/5 : ℝ) * allowance) + 
  1.20 = allowance → 
  allowance = 3 :=
by sorry

end student_allowance_l1420_142056


namespace polynomial_expansion_l1420_142067

theorem polynomial_expansion (x : ℝ) :
  (1 + x^3) * (1 - x^4) * (1 + x^5) =
  1 + x^3 - x^4 + x^5 - x^7 + x^8 - x^9 - x^12 := by
  sorry

end polynomial_expansion_l1420_142067


namespace mixed_nuts_price_per_pound_l1420_142070

/-- Calculates the price per pound of mixed nuts given the following conditions:
  * Total weight of mixed nuts is 100 pounds
  * Price of peanuts is $3.50 per pound
  * Price of cashews is $4.00 per pound
  * Amount of cashews used is 60 pounds
-/
theorem mixed_nuts_price_per_pound 
  (total_weight : ℝ) 
  (peanut_price : ℝ) 
  (cashew_price : ℝ) 
  (cashew_weight : ℝ) 
  (h1 : total_weight = 100) 
  (h2 : peanut_price = 3.5) 
  (h3 : cashew_price = 4) 
  (h4 : cashew_weight = 60) : 
  (cashew_price * cashew_weight + peanut_price * (total_weight - cashew_weight)) / total_weight = 3.8 := by
  sorry

end mixed_nuts_price_per_pound_l1420_142070


namespace carpet_innermost_length_l1420_142042

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the carpet with three nested rectangles --/
structure Carpet where
  inner : Rectangle
  middle : Rectangle
  outer : Rectangle

/-- Checks if three numbers form an arithmetic progression --/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem carpet_innermost_length :
  ∀ (c : Carpet),
    c.inner.width = 2 →
    c.middle.length = c.inner.length + 4 →
    c.middle.width = c.inner.width + 4 →
    c.outer.length = c.middle.length + 4 →
    c.outer.width = c.middle.width + 4 →
    isArithmeticProgression (area c.inner) (area c.middle) (area c.outer) →
    c.inner.length = 4 := by
  sorry

#check carpet_innermost_length

end carpet_innermost_length_l1420_142042


namespace triangle_area_l1420_142035

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  1/2 * b * Real.cos A = Real.sin B →
  a = 2 * Real.sqrt 3 →
  b + c = 6 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 :=
by sorry

end triangle_area_l1420_142035


namespace largest_common_divisor_408_340_l1420_142045

theorem largest_common_divisor_408_340 : ∃ (n : ℕ), n = Nat.gcd 408 340 ∧ n = 68 := by
  sorry

end largest_common_divisor_408_340_l1420_142045


namespace BC_length_l1420_142012

-- Define the points and segments
variable (A B C D E : ℝ × ℝ)

-- Define the lengths of segments
def length (P Q : ℝ × ℝ) : ℝ := sorry

-- Define the conditions
def on_segment (P Q R : ℝ × ℝ) : Prop := sorry

axiom D_on_AE : on_segment A D E
axiom B_on_AD : on_segment A B D
axiom C_on_DE : on_segment D C E

axiom AB_length : length A B = 3 + 3 * length B D
axiom CE_length : length C E = 2 + 2 * length C D
axiom AE_length : length A E = 20

-- Theorem to prove
theorem BC_length : length B C = 4 := by sorry

end BC_length_l1420_142012


namespace difference_of_squares_153_147_l1420_142020

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end difference_of_squares_153_147_l1420_142020


namespace perfect_square_quadratic_l1420_142017

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x, ∃ y, 9 - m * x + x^2 = y^2) → (m = 6 ∨ m = -6) := by
  sorry

end perfect_square_quadratic_l1420_142017


namespace three_number_product_l1420_142032

theorem three_number_product (a b c m : ℚ) : 
  a + b + c = 210 →
  8 * a = m →
  b - 12 = m →
  c + 12 = m →
  a * b * c = 58116480 / 4913 := by
sorry

end three_number_product_l1420_142032


namespace log_stack_sum_l1420_142086

theorem log_stack_sum : ∀ (a₁ aₙ n : ℕ),
  a₁ = 12 →
  aₙ = 3 →
  n = 10 →
  (n : ℝ) / 2 * (a₁ + aₙ) = 75 := by
  sorry

end log_stack_sum_l1420_142086


namespace mixed_number_calculation_l1420_142040

theorem mixed_number_calculation : (3 + 3 / 4) * 1.3 + 3 / (2 + 2 / 3) = 6 := by
  sorry

end mixed_number_calculation_l1420_142040


namespace complex_magnitude_problem_l1420_142027

theorem complex_magnitude_problem (z : ℂ) : z = (5 * Complex.I) / (2 + Complex.I) - 3 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l1420_142027


namespace complex_equality_l1420_142005

theorem complex_equality (z₁ z₂ : ℂ) (a : ℝ) 
  (h1 : z₁ = 1 + Complex.I) 
  (h2 : z₂ = 3 + a * Complex.I) 
  (h3 : 3 * z₁ = z₂) : 
  a = 3 := by
sorry

end complex_equality_l1420_142005


namespace kim_shirts_proof_l1420_142098

def dozen : ℕ := 12

def initial_shirts : ℕ := 4 * dozen

def shirts_given_away : ℕ := initial_shirts / 3

def remaining_shirts : ℕ := initial_shirts - shirts_given_away

theorem kim_shirts_proof : remaining_shirts = 32 := by
  sorry

end kim_shirts_proof_l1420_142098


namespace sector_arc_length_l1420_142053

theorem sector_arc_length (r : ℝ) (θ : ℝ) (l : ℝ) : 
  r = 3 → θ = 2 * π / 3 → l = r * θ → l = 2 * π := by
  sorry

end sector_arc_length_l1420_142053


namespace derivative_cos_cubed_l1420_142085

theorem derivative_cos_cubed (x : ℝ) :
  let y : ℝ → ℝ := λ x => (Real.cos (2 * x + 3)) ^ 3
  deriv y x = -6 * (Real.cos (2 * x + 3))^2 * Real.sin (2 * x + 3) :=
by sorry

end derivative_cos_cubed_l1420_142085


namespace bills_earnings_l1420_142077

/-- Represents the earnings from dairy products -/
def dairy_earnings (total_milk : ℚ) (butter_ratio : ℚ) (sour_cream_ratio : ℚ) 
  (milk_to_butter : ℚ) (milk_to_sour_cream : ℚ) 
  (butter_price : ℚ) (sour_cream_price : ℚ) (milk_price : ℚ) : ℚ :=
  let butter_milk := total_milk * butter_ratio
  let sour_cream_milk := total_milk * sour_cream_ratio
  let whole_milk := total_milk - butter_milk - sour_cream_milk
  let butter_gallons := butter_milk / milk_to_butter
  let sour_cream_gallons := sour_cream_milk / milk_to_sour_cream
  butter_gallons * butter_price + sour_cream_gallons * sour_cream_price + whole_milk * milk_price

/-- Bill's earnings from his dairy products -/
theorem bills_earnings : 
  dairy_earnings 16 (1/4) (1/4) 4 2 5 6 3 = 41 := by
  sorry

end bills_earnings_l1420_142077


namespace connected_with_short_paths_l1420_142072

/-- A directed graph with n vertices -/
structure DirectedGraph (n : ℕ) where
  edges : Fin n → Fin n → Prop

/-- A path of length at most 2 between two vertices -/
def hasPathOfLengthAtMostTwo (G : DirectedGraph n) (u v : Fin n) : Prop :=
  G.edges u v ∨ ∃ w : Fin n, G.edges u w ∧ G.edges w v

/-- The main theorem statement -/
theorem connected_with_short_paths (n : ℕ) (h : n ≥ 5) :
  ∃ G : DirectedGraph n, ∀ u v : Fin n, hasPathOfLengthAtMostTwo G u v :=
sorry

end connected_with_short_paths_l1420_142072


namespace probability_both_selected_l1420_142092

theorem probability_both_selected (X Y : ℝ) (hX : X = 1/7) (hY : Y = 2/9) :
  X * Y = 2/63 := by
  sorry

end probability_both_selected_l1420_142092


namespace max_sum_for_2029_product_l1420_142066

theorem max_sum_for_2029_product (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → 
  A * B * C = 2029 → 
  A + B + C ≤ 297 := by
sorry

end max_sum_for_2029_product_l1420_142066


namespace fraction_zero_implies_x_negative_two_l1420_142036

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  (abs x - 2) / (x^2 - 4*x + 4) = 0 → x = -2 :=
by sorry

end fraction_zero_implies_x_negative_two_l1420_142036


namespace remainder_theorem_l1420_142090

theorem remainder_theorem : 7 * 10^20 + 1^20 ≡ 8 [ZMOD 11] := by
  sorry

end remainder_theorem_l1420_142090


namespace num_possible_heights_l1420_142078

/-- The dimensions of each block -/
def block_dimensions : Finset ℕ := {2, 3, 6}

/-- The number of blocks in the tower -/
def num_blocks : ℕ := 4

/-- A function to calculate all possible heights of the tower -/
def possible_heights : Finset ℕ := sorry

/-- The theorem stating that the number of possible heights is 14 -/
theorem num_possible_heights : Finset.card possible_heights = 14 := by sorry

end num_possible_heights_l1420_142078


namespace subset_complement_relation_l1420_142083

universe u

theorem subset_complement_relation {U : Type u} (M N : Set U) 
  (hM : M.Nonempty) (hN : N.Nonempty) (h : N ⊆ Mᶜ) : M ⊆ Nᶜ := by
  sorry

end subset_complement_relation_l1420_142083


namespace train_passing_time_l1420_142021

/-- The time it takes for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 110 →
  train_speed = 80 * 1000 / 3600 →
  man_speed = 8 * 1000 / 3600 →
  (train_length / (train_speed + man_speed)) = 4.5 := by
  sorry

#check train_passing_time

end train_passing_time_l1420_142021


namespace collinear_points_theorem_l1420_142049

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear -/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(m, 0), B(0, 1), and C(3, -1) are collinear, then m = 3/2 -/
theorem collinear_points_theorem (m : ℝ) :
  are_collinear (m, 0) (0, 1) (3, -1) → m = 3/2 :=
by
  sorry

end collinear_points_theorem_l1420_142049


namespace letters_difference_l1420_142018

/-- The number of letters Jack received in the morning -/
def morning_letters : ℕ := 8

/-- The number of letters Jack received in the afternoon -/
def afternoon_letters : ℕ := 7

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 6

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 2

theorem letters_difference : morning_letters - afternoon_letters = 1 := by
  sorry

end letters_difference_l1420_142018


namespace star_1993_1932_l1420_142038

-- Define the * operation
def star (x y : ℤ) : ℤ := x - y

-- State the theorem
theorem star_1993_1932 : star 1993 1932 = 61 :=
  by
  -- Define the properties of the star operation
  have h1 : ∀ x : ℤ, star x x = 0 := by sorry
  have h2 : ∀ x y z : ℤ, star x (star y z) = star x y + z := by sorry
  
  -- Prove the theorem
  sorry

end star_1993_1932_l1420_142038


namespace archery_score_proof_l1420_142093

/-- Represents the score of hitting a region in the archery target -/
structure RegionScore where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Represents the score of an archer -/
def archerScore (rs : RegionScore) (r1 r2 : Fin 3) : ℕ :=
  match r1, r2 with
  | 0, 0 => rs.A + rs.A
  | 0, 1 | 1, 0 => rs.A + rs.B
  | 0, 2 | 2, 0 => rs.A + rs.C
  | 1, 1 => rs.B + rs.B
  | 1, 2 | 2, 1 => rs.B + rs.C
  | 2, 2 => rs.C + rs.C

theorem archery_score_proof (rs : RegionScore) 
  (h1 : archerScore rs 2 0 = 15)  -- First archer: C and A
  (h2 : archerScore rs 2 1 = 18)  -- Second archer: C and B
  (h3 : archerScore rs 1 0 = 13)  -- Third archer: B and A
  : archerScore rs 1 1 = 16 :=    -- Fourth archer: B and B
by sorry

end archery_score_proof_l1420_142093


namespace square_minus_reciprocal_square_l1420_142088

theorem square_minus_reciprocal_square (x : ℝ) (h1 : x > 1) (h2 : x + 1/x = Real.sqrt 22) :
  x^2 - 1/x^2 = 3 * Real.sqrt 2 := by
sorry

end square_minus_reciprocal_square_l1420_142088


namespace abs_diff_and_opposite_l1420_142014

theorem abs_diff_and_opposite (a b : ℝ) (h : a < b) : 
  |((a - b) - (b - a))| = 2*b - 2*a := by sorry

end abs_diff_and_opposite_l1420_142014


namespace marble_leftover_l1420_142024

theorem marble_leftover (r p g : ℤ) 
  (hr : r % 6 = 5)
  (hp : p % 6 = 2)
  (hg : g % 6 = 3) :
  (r + p + g) % 6 = 4 := by
sorry

end marble_leftover_l1420_142024


namespace constant_values_l1420_142041

theorem constant_values (C A : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 → (C * x - 10) / ((x - 4) * (x - 5)) = A / (x - 4) + 2 / (x - 5)) →
  C = 12/5 ∧ A = 2/5 := by sorry

end constant_values_l1420_142041


namespace solve_for_q_l1420_142044

theorem solve_for_q (n m q : ℚ) 
  (h1 : (7 : ℚ) / 9 = n / 81)
  (h2 : (7 : ℚ) / 9 = (m + n) / 99)
  (h3 : (7 : ℚ) / 9 = (q - m) / 135) : 
  q = 119 := by sorry

end solve_for_q_l1420_142044


namespace arg_cube_quotient_complex_l1420_142071

theorem arg_cube_quotient_complex (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 5)
  (h3 : Complex.abs (z₁ + z₂) = 7) :
  Complex.arg ((z₂ / z₁) ^ 3) = π :=
sorry

end arg_cube_quotient_complex_l1420_142071


namespace sixty_six_green_squares_l1420_142095

/-- Represents a grid with colored squares -/
structure ColoredGrid :=
  (rows : Nat)
  (columns : Nat)
  (redRows : Nat)
  (redColumns : Nat)
  (blueRows : Nat)

/-- Calculates the number of green squares in the grid -/
def greenSquares (grid : ColoredGrid) : Nat :=
  grid.rows * grid.columns - (grid.redRows * grid.redColumns) - (grid.blueRows * grid.columns)

/-- Theorem stating that in the given grid configuration, there are 66 green squares -/
theorem sixty_six_green_squares :
  let grid : ColoredGrid := {
    rows := 10,
    columns := 15,
    redRows := 4,
    redColumns := 6,
    blueRows := 4
  }
  greenSquares grid = 66 := by sorry

end sixty_six_green_squares_l1420_142095


namespace square_inscribed_circle_tangent_l1420_142001

/-- Given a square with side length a, where two adjacent sides are divided into 6 and 10 equal parts respectively,
    prove that the line segment connecting the first division point (a/6) on one side to the fourth division point (4a/10)
    on the adjacent side is tangent to the inscribed circle of the square. -/
theorem square_inscribed_circle_tangent (a : ℝ) (h : a > 0) : 
  let square := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ a}
  let P := (a / 6, 0)
  let Q := (0, 2 * a / 5)
  let circle_center := (a / 2, a / 2)
  let circle_radius := a / 2
  let circle := {(x, y) : ℝ × ℝ | (x - a / 2)^2 + (y - a / 2)^2 = (a / 2)^2}
  let line_PQ := {(x, y) : ℝ × ℝ | y = -12/5 * x + 2*a/5}
  (∀ point ∈ line_PQ, point ∉ circle) ∧ 
  (∃ point ∈ line_PQ, (point.1 - circle_center.1)^2 + (point.2 - circle_center.2)^2 = circle_radius^2)
  := by sorry

end square_inscribed_circle_tangent_l1420_142001


namespace car_speed_problem_l1420_142073

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 80 →
  average_speed = 85 →
  ∃ (speed_first_hour : ℝ),
    speed_first_hour = 90 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by sorry

end car_speed_problem_l1420_142073


namespace average_price_per_movie_l1420_142011

theorem average_price_per_movie :
  let dvd_count : ℕ := 8
  let dvd_price : ℚ := 12
  let bluray_count : ℕ := 4
  let bluray_price : ℚ := 18
  let total_spent : ℚ := dvd_count * dvd_price + bluray_count * bluray_price
  let total_movies : ℕ := dvd_count + bluray_count
  (total_spent / total_movies : ℚ) = 14 := by
sorry

end average_price_per_movie_l1420_142011


namespace tangent_slope_angle_at_1_is_45_degrees_l1420_142055

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_angle_at_1_is_45_degrees :
  let slope := f' 1
  let angle := Real.arctan slope
  angle = π/4 := by sorry

end tangent_slope_angle_at_1_is_45_degrees_l1420_142055


namespace collinear_points_sum_l1420_142058

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, p2 = p1 + t1 • (p3 - p1) ∧ p3 = p1 + t2 • (p3 - p1)

/-- Given three collinear points (2,x,y), (x,3,y), and (x,y,4), prove that x + y = 6. -/
theorem collinear_points_sum (x y : ℝ) :
  collinear (2, x, y) (x, 3, y) (x, y, 4) → x + y = 6 := by
  sorry

end collinear_points_sum_l1420_142058


namespace center_student_coins_l1420_142060

/-- Represents the number of coins each student has in the network -/
structure CoinDistribution :=
  (center : ℕ)
  (first_ring : ℕ)
  (second_ring : ℕ)
  (outer_ring : ℕ)

/-- Defines the conditions of the coin distribution problem -/
def valid_distribution (d : CoinDistribution) : Prop :=
  -- Total number of coins is 3360
  d.center + 5 * d.first_ring + 5 * d.second_ring + 5 * d.outer_ring = 3360 ∧
  -- Center student exchanges with first ring
  d.center = d.first_ring ∧
  -- First ring exchanges with center, second ring, and other first ring students
  d.first_ring = d.center / 5 + d.second_ring / 2 ∧
  -- Second ring exchanges with first ring and outer ring
  d.second_ring = 2 * d.first_ring / 3 + d.outer_ring / 2 ∧
  -- Outer ring exchanges with second ring
  d.outer_ring = d.second_ring

/-- The main theorem stating that the center student must have 280 coins -/
theorem center_student_coins :
  ∀ d : CoinDistribution, valid_distribution d → d.center = 280 :=
by sorry

end center_student_coins_l1420_142060


namespace buyers_both_mixes_l1420_142033

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers purchasing cake mix
def cake_mix_buyers : ℕ := 50

-- Define the number of buyers purchasing muffin mix
def muffin_mix_buyers : ℕ := 40

-- Define the probability of selecting a buyer who purchases neither cake mix nor muffin mix
def prob_neither : ℚ := 27 / 100

-- Theorem to prove
theorem buyers_both_mixes (both_mixes : ℕ) : 
  (cake_mix_buyers + muffin_mix_buyers - both_mixes = total_buyers * (1 - prob_neither)) →
  both_mixes = 17 := by
  sorry

end buyers_both_mixes_l1420_142033


namespace range_of_sine_function_l1420_142000

theorem range_of_sine_function (ω : ℝ) (φ : ℝ) :
  ω > 0 →
  (∀ x, 3 * Real.sin (ω * x - π / 6) = 3 * Real.cos (2 * x + φ)) →
  (∀ x ∈ Set.Icc 0 (π / 2), 
    -3/2 ≤ 3 * Real.sin (ω * x - π / 6) ∧ 
    3 * Real.sin (ω * x - π / 6) ≤ 3) :=
by sorry

end range_of_sine_function_l1420_142000


namespace parallel_vectors_dot_product_l1420_142006

/-- Given vectors a⃗(x,2), b⃗=(2,1), c⃗=(3,x), if a⃗ ∥ b⃗, then a⃗ ⋅ c⃗ = 20 -/
theorem parallel_vectors_dot_product (x : ℝ) :
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, 1)
  let c : ℝ × ℝ := (3, x)
  (∃ (k : ℝ), a = k • b) →
  a.1 * c.1 + a.2 * c.2 = 20 := by
sorry

end parallel_vectors_dot_product_l1420_142006


namespace cos_2pi_3_plus_2alpha_l1420_142043

theorem cos_2pi_3_plus_2alpha (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 8 := by
  sorry

end cos_2pi_3_plus_2alpha_l1420_142043


namespace chickens_egg_laying_rate_l1420_142069

/-- Proves that given the initial conditions, each chicken lays 6 eggs per day. -/
theorem chickens_egg_laying_rate 
  (initial_chickens : ℕ) 
  (growth_factor : ℕ) 
  (weekly_eggs : ℕ) 
  (h1 : initial_chickens = 4)
  (h2 : growth_factor = 8)
  (h3 : weekly_eggs = 1344) : 
  (weekly_eggs / 7) / (initial_chickens * growth_factor) = 6 := by
  sorry

#eval (1344 / 7) / (4 * 8)  -- Should output 6

end chickens_egg_laying_rate_l1420_142069


namespace wrapping_paper_area_for_specific_box_l1420_142004

/-- Represents the dimensions of a rectangular box. -/
structure BoxDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the area of the square wrapping paper needed for a given box. -/
def wrappingPaperArea (box : BoxDimensions) : ℝ :=
  4 * box.width ^ 2

/-- Theorem stating that for a box with dimensions a × 2a × a, 
    the area of the square wrapping paper is 4a². -/
theorem wrapping_paper_area_for_specific_box (a : ℝ) (h : a > 0) :
  let box : BoxDimensions := ⟨a, 2*a, a⟩
  wrappingPaperArea box = 4 * a ^ 2 := by
  sorry

#check wrapping_paper_area_for_specific_box

end wrapping_paper_area_for_specific_box_l1420_142004


namespace pregnant_cows_l1420_142087

theorem pregnant_cows (total_cows : ℕ) (female_ratio : ℚ) (pregnant_ratio : ℚ) :
  total_cows = 44 →
  female_ratio = 1/2 →
  pregnant_ratio = 1/2 →
  ⌊(total_cows : ℚ) * female_ratio * pregnant_ratio⌋ = 11 := by
  sorry

end pregnant_cows_l1420_142087


namespace class_size_l1420_142025

theorem class_size (n : ℕ) (dima_shorter dima_taller lenya_shorter lenya_taller : ℕ) :
  n ≤ 30 ∧
  n = dima_shorter + dima_taller + 1 ∧
  n = lenya_shorter + lenya_taller + 1 ∧
  dima_taller = 4 * dima_shorter ∧
  lenya_shorter = 3 * lenya_taller →
  n = 21 :=
by sorry

end class_size_l1420_142025


namespace power_of_seven_mod_six_l1420_142023

theorem power_of_seven_mod_six : 7^51 % 6 = 1 := by
  sorry

end power_of_seven_mod_six_l1420_142023


namespace complex_fraction_equality_l1420_142002

/-- Given that i is the imaginary unit, prove that (2*i)/(1+i) = 1+i -/
theorem complex_fraction_equality : (2 * Complex.I) / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end complex_fraction_equality_l1420_142002


namespace back_parking_spaces_l1420_142076

/-- Proves that the number of parking spaces in the back of the school is 38 -/
theorem back_parking_spaces : 
  ∀ (front_spaces back_spaces : ℕ),
    front_spaces = 52 →
    ∃ (parked_cars available_spaces : ℕ),
      parked_cars = 39 ∧
      available_spaces = 32 ∧
      parked_cars + available_spaces = front_spaces + back_spaces ∧
      parked_cars - front_spaces = back_spaces / 2 →
        back_spaces = 38 := by
sorry

end back_parking_spaces_l1420_142076


namespace amusement_park_total_cost_l1420_142047

/-- The total cost of an amusement park trip for a group of children -/
def amusement_park_cost (num_children : ℕ) (ferris_wheel_riders : ℕ) (ferris_wheel_cost : ℕ) 
  (merry_go_round_cost : ℕ) (ice_cream_cones_per_child : ℕ) (ice_cream_cost : ℕ) : ℕ :=
  ferris_wheel_riders * ferris_wheel_cost + 
  num_children * merry_go_round_cost + 
  num_children * ice_cream_cones_per_child * ice_cream_cost

/-- Theorem stating the total cost for the given scenario -/
theorem amusement_park_total_cost : 
  amusement_park_cost 5 3 5 3 2 8 = 110 := by
  sorry

end amusement_park_total_cost_l1420_142047


namespace remaining_distance_l1420_142015

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 215) :
  total_distance - driven_distance = 985 := by
sorry

end remaining_distance_l1420_142015


namespace hash_2_4_5_l1420_142081

/-- The # operation for real numbers -/
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem stating that hash(2, 4, 5) equals -24 -/
theorem hash_2_4_5 : hash 2 4 5 = -24 := by sorry

end hash_2_4_5_l1420_142081


namespace abs_sum_lt_abs_diff_when_product_negative_l1420_142010

theorem abs_sum_lt_abs_diff_when_product_negative (a b : ℝ) (h : a * b < 0) :
  |a + b| < |a - b| := by sorry

end abs_sum_lt_abs_diff_when_product_negative_l1420_142010


namespace ladder_top_velocity_l1420_142091

/-- Given a ladder sliding down a wall, this theorem calculates the velocity of the top end of the ladder. -/
theorem ladder_top_velocity (a l τ : ℝ) (h_positive : a > 0 ∧ l > 0 ∧ τ > 0) :
  let x := a * τ^2 / 2
  let v₁ := a * τ
  let α := Real.arcsin (a * τ^2 / (2 * l))
  let v₂ := a^2 * τ^3 / Real.sqrt (4 * l^2 - a^2 * τ^4)
  (x = a * τ^2 / 2) →
  (v₁ = a * τ) →
  (Real.sin α = a * τ^2 / (2 * l)) →
  (v₁ * Real.sin α = v₂ * Real.cos α) →
  v₂ = a^2 * τ^3 / Real.sqrt (4 * l^2 - a^2 * τ^4) :=
by sorry

end ladder_top_velocity_l1420_142091


namespace sum_product_inequality_l1420_142029

theorem sum_product_inequality (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_one : a + b + c + d = 1) :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ 1 / 27 + 176 * a * b * c * d / 27 := by
  sorry

end sum_product_inequality_l1420_142029


namespace parking_garage_spots_per_level_l1420_142079

theorem parking_garage_spots_per_level :
  -- Define the number of levels in the parking garage
  let num_levels : ℕ := 4

  -- Define the number of open spots on each level
  let open_spots_level1 : ℕ := 58
  let open_spots_level2 : ℕ := open_spots_level1 + 2
  let open_spots_level3 : ℕ := open_spots_level2 + 5
  let open_spots_level4 : ℕ := 31

  -- Define the total number of full spots
  let full_spots : ℕ := 186

  -- Calculate the total number of open spots
  let total_open_spots : ℕ := open_spots_level1 + open_spots_level2 + open_spots_level3 + open_spots_level4

  -- Calculate the total number of spots
  let total_spots : ℕ := total_open_spots + full_spots

  -- The number of spots per level
  (total_spots / num_levels : ℕ) = 100 := by
  sorry

end parking_garage_spots_per_level_l1420_142079


namespace right_triangle_sin_cos_relation_l1420_142013

theorem right_triangle_sin_cos_relation (A B C : ℝ) :
  A = Real.pi / 2 →  -- ∠A = 90°
  Real.cos B = 3 / 5 →
  Real.sin C = 3 / 5 := by
  sorry

end right_triangle_sin_cos_relation_l1420_142013


namespace chris_cookies_l1420_142039

theorem chris_cookies (total_cookies : ℕ) (chris_fraction : ℚ) (eaten_fraction : ℚ) : 
  total_cookies = 84 →
  chris_fraction = 1/3 →
  eaten_fraction = 3/4 →
  (↑total_cookies * chris_fraction * eaten_fraction : ℚ) = 21 := by
  sorry

end chris_cookies_l1420_142039


namespace series_sum_proof_l1420_142016

theorem series_sum_proof : ∑' k, (k : ℝ) / (4 ^ k) = 4 / 9 := by sorry

end series_sum_proof_l1420_142016


namespace median_is_three_l1420_142019

def sibling_list : List Nat := [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6]

def median_index (n : Nat) : Nat :=
  (n + 1) / 2

theorem median_is_three :
  sibling_list.get? (median_index sibling_list.length - 1) = some 3 := by
  sorry

end median_is_three_l1420_142019


namespace product_103_97_l1420_142009

theorem product_103_97 : 103 * 97 = 9991 := by
  sorry

end product_103_97_l1420_142009


namespace geometric_sequence_and_max_value_l1420_142007

/-- Given real numbers a, b, c, and d forming a geometric sequence, and a function
    y = ln x - x attaining its maximum value c when x = b, prove that ad = -1 -/
theorem geometric_sequence_and_max_value (a b c d : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, x > 0 → Real.log x - x ≤ c) →        -- maximum value condition
  (Real.log b - b = c) →                         -- attains maximum at x = b
  a * d = -1 := by
sorry

end geometric_sequence_and_max_value_l1420_142007


namespace cost_price_percentage_l1420_142074

/-- Given a selling price and a profit percentage, calculates the cost price as a percentage of the selling price. -/
theorem cost_price_percentage (selling_price : ℝ) (profit_percentage : ℝ) :
  profit_percentage = 4.166666666666666 →
  (selling_price - (selling_price * profit_percentage / 100)) / selling_price * 100 = 95.83333333333334 :=
by
  sorry

end cost_price_percentage_l1420_142074


namespace imaginary_number_properties_l1420_142059

open Complex

theorem imaginary_number_properties (z : ℂ) (ω : ℝ) :
  z.im ≠ 0 →  -- z is an imaginary number
  ω = z.re + z.im * I + (z.re - z.im * I) / (z.re^2 + z.im^2) →  -- ω = z + 1/z
  -1 < ω ∧ ω < 2 →  -- -1 < ω < 2
  abs z = 1 ∧  -- |z| = 1
  -1/2 < z.re ∧ z.re < 1 ∧  -- real part of z is in (-1/2, 1)
  1 < abs (z - 2) ∧ abs (z - 2) < Real.sqrt 7  -- |z-2| is in (1, √7)
  := by sorry

end imaginary_number_properties_l1420_142059


namespace smaller_number_proof_l1420_142084

theorem smaller_number_proof (x y : ℕ) 
  (h1 : y - x = 2395)
  (h2 : y = 6 * x + 15) :
  x = 476 := by
  sorry

end smaller_number_proof_l1420_142084


namespace shop_length_is_18_l1420_142048

/-- Calculates the length of a shop given its monthly rent, width, and annual rent per square foot. -/
def shop_length (monthly_rent : ℕ) (width : ℕ) (annual_rent_per_sqft : ℕ) : ℕ :=
  (monthly_rent * 12) / (width * annual_rent_per_sqft)

/-- Proves that the length of the shop is 18 feet given the specified conditions. -/
theorem shop_length_is_18 :
  shop_length 1440 20 48 = 18 := by
  sorry

end shop_length_is_18_l1420_142048


namespace inclined_line_and_volume_l1420_142030

/-- A line passing through a point with a given inclination angle cosine -/
structure InclinedLine where
  point : ℝ × ℝ
  cos_angle : ℝ

/-- The general form equation coefficients of a line -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The volume of a geometric body -/
def GeometricBodyVolume : Type := ℝ

/-- Calculate the general form equation of the line -/
def calculateLineEquation (l : InclinedLine) : LineEquation :=
  sorry

/-- Calculate the volume of the geometric body -/
def calculateGeometricBodyVolume (eq : LineEquation) : GeometricBodyVolume :=
  sorry

theorem inclined_line_and_volume 
  (l : InclinedLine) 
  (h1 : l.point = (-1, 2)) 
  (h2 : l.cos_angle = Real.sqrt 2 / 2) : 
  let eq := calculateLineEquation l
  calculateGeometricBodyVolume eq = 9 * Real.pi ∧ 
  eq.a = 1 ∧ eq.b = -1 ∧ eq.c = -3 :=
sorry

end inclined_line_and_volume_l1420_142030


namespace water_mixture_percentage_l1420_142050

theorem water_mixture_percentage (initial_volume : ℝ) (initial_water_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 125 ∧
  initial_water_percentage = 0.2 ∧
  added_water = 8.333333333333334 →
  let initial_water := initial_volume * initial_water_percentage
  let new_water := initial_water + added_water
  let new_volume := initial_volume + added_water
  new_water / new_volume = 0.25 := by
  sorry

end water_mixture_percentage_l1420_142050


namespace matrix_N_satisfies_conditions_l1420_142061

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
  (N.mulVec (![1, 0, 0] : Fin 3 → ℝ) = ![(-1), 4, 0]) ∧
  (N.mulVec (![0, 1, 0] : Fin 3 → ℝ) = ![2, (-3), 5]) ∧
  (N.mulVec (![0, 0, 1] : Fin 3 → ℝ) = ![5, 2, (-1)]) ∧
  (N.mulVec (![1, 1, 1] : Fin 3 → ℝ) = ![6, 3, 4]) :=
by
  sorry

#check matrix_N_satisfies_conditions

end matrix_N_satisfies_conditions_l1420_142061


namespace solve_for_a_l1420_142075

def U (a : ℝ) : Set ℝ := {1, 2, a^2 + 2*a - 3}

def A (a : ℝ) : Set ℝ := {|a - 2|, 2}

theorem solve_for_a (a : ℝ) :
  U a = {1, 2, 0} ∧
  A a ∪ {0} = U a →
  a = 1 :=
sorry

end solve_for_a_l1420_142075


namespace g_comp_g_three_roots_l1420_142022

/-- The function g defined as g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_comp_g (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 3 distinct real roots if and only if d = 0 -/
theorem g_comp_g_three_roots :
  ∀ d : ℝ, (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧
    g_comp_g d r1 = 0 ∧ g_comp_g d r2 = 0 ∧ g_comp_g d r3 = 0 ∧
    ∀ x : ℝ, g_comp_g d x = 0 → x = r1 ∨ x = r2 ∨ x = r3) ↔ d = 0 :=
sorry

end g_comp_g_three_roots_l1420_142022


namespace quadratic_function_property_l1420_142080

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 1 = f 3 ∧ f 1 > f 4) → (a < 0 ∧ 4 * a + b = 0) := by
  sorry

end quadratic_function_property_l1420_142080


namespace average_and_difference_l1420_142037

theorem average_and_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 30)
  (h2 : (b + c) / 2 = 60)
  (h3 : c - a = 60) :
  c - a = 60 := by
  sorry

end average_and_difference_l1420_142037


namespace base_conversion_product_l1420_142031

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The problem statement --/
theorem base_conversion_product : 
  let numerator1 := to_base_10 [2, 6, 2] 8
  let denominator1 := to_base_10 [1, 3] 4
  let numerator2 := to_base_10 [1, 4, 4] 7
  let denominator2 := to_base_10 [2, 4] 5
  (numerator1 * numerator2) / (denominator1 * denominator2) = 147 := by
  sorry

end base_conversion_product_l1420_142031


namespace arithmetic_sequence_property_l1420_142099

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem to be proved -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 8 = 15 - a 5 → a 5 = 5 := by
  sorry

end arithmetic_sequence_property_l1420_142099


namespace snow_white_marbles_l1420_142054

theorem snow_white_marbles (x : ℕ) (y : ℕ) : 
  (x > 0) →
  (y > 0) →
  (y ≤ 6) →
  (7 * x - (1 + 2 + 3 + 4 + 5 + 6) - y = 46) →
  (x = 10 ∧ y = 3) := by
sorry

end snow_white_marbles_l1420_142054


namespace expression_simplification_l1420_142082

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (2 * x / (x^2 - 4) - 1 / (x + 2)) / ((x - 1) / (x - 2)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1420_142082
