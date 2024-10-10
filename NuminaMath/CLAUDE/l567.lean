import Mathlib

namespace concert_revenue_is_955000_l567_56742

/-- Calculates the total revenue of a concert given the following parameters:
  * total_seats: Total number of seats in the arena
  * main_seat_cost: Cost of a main seat ticket
  * back_seat_cost: Cost of a back seat ticket
  * back_seats_sold: Number of back seat tickets sold
-/
def concert_revenue (total_seats : ℕ) (main_seat_cost back_seat_cost : ℕ) (back_seats_sold : ℕ) : ℕ :=
  let main_seats_sold := total_seats - back_seats_sold
  let main_seat_revenue := main_seats_sold * main_seat_cost
  let back_seat_revenue := back_seats_sold * back_seat_cost
  main_seat_revenue + back_seat_revenue

/-- Theorem stating that the concert revenue is $955,000 given the specific conditions -/
theorem concert_revenue_is_955000 :
  concert_revenue 20000 55 45 14500 = 955000 := by
  sorry

#eval concert_revenue 20000 55 45 14500

end concert_revenue_is_955000_l567_56742


namespace document_word_count_l567_56769

/-- Calculates the number of words in a document based on typing speed and time --/
def document_words (original_speed : ℕ) (speed_reduction : ℕ) (typing_time : ℕ) : ℕ :=
  (original_speed - speed_reduction) * typing_time

/-- Proves that the number of words in the document is 810 --/
theorem document_word_count : document_words 65 20 18 = 810 := by
  sorry

end document_word_count_l567_56769


namespace gambler_winning_percentage_l567_56716

/-- Calculates the final winning percentage of a gambler --/
theorem gambler_winning_percentage
  (initial_games : ℕ)
  (initial_win_rate : ℚ)
  (additional_games : ℕ)
  (new_win_rate : ℚ)
  (h1 : initial_games = 30)
  (h2 : initial_win_rate = 2/5)
  (h3 : additional_games = 30)
  (h4 : new_win_rate = 4/5) :
  let total_games := initial_games + additional_games
  let total_wins := initial_games * initial_win_rate + additional_games * new_win_rate
  total_wins / total_games = 3/5 := by
sorry

#eval (2/5 : ℚ)  -- To verify that 2/5 is indeed 0.4
#eval (4/5 : ℚ)  -- To verify that 4/5 is indeed 0.8
#eval (3/5 : ℚ)  -- To verify that 3/5 is indeed 0.6

end gambler_winning_percentage_l567_56716


namespace triangle_similarity_l567_56718

-- Define the points as complex numbers
variable (z₁ z₂ z₃ t₁ t₂ t₃ z₁' z₂' z₃' : ℂ)

-- Define the similarity relation
def similar (a b c d e f : ℂ) : Prop :=
  (e - d) / (f - d) = (b - a) / (c - a)

-- State the theorem
theorem triangle_similarity :
  similar z₁ z₂ z₃ t₁ t₂ t₃ →  -- DBC similar to ABC
  similar z₂ z₃ z₁ t₂ t₃ t₁ →  -- ECA similar to ABC
  similar z₃ z₁ z₂ t₃ t₁ t₂ →  -- FAB similar to ABC
  similar t₂ t₃ t₁ z₁' t₃ t₂ →  -- A'FE similar to DBC
  similar t₃ t₁ t₂ z₂' t₁ t₃ →  -- B'DF similar to ECA
  similar t₁ t₂ t₃ z₃' t₂ t₁ →  -- C'ED similar to FAB
  similar z₁ z₂ z₃ z₁' z₂' z₃'  -- A'B'C' similar to ABC
:= by sorry

end triangle_similarity_l567_56718


namespace remainder_problem_l567_56793

theorem remainder_problem (f y : ℤ) : 
  y % 5 = 4 → (f + y) % 5 = 2 → f % 5 = 3 :=
by sorry

end remainder_problem_l567_56793


namespace parabola_focus_directrix_distance_l567_56739

/-- Given a parabola defined by y = 4x^2, the distance from its focus to its directrix is 1/8. -/
theorem parabola_focus_directrix_distance (x y : ℝ) : 
  y = 4 * x^2 → (distance_focus_to_directrix : ℝ) = 1/8 :=
by sorry

end parabola_focus_directrix_distance_l567_56739


namespace remainder_of_190_div_18_l567_56770

theorem remainder_of_190_div_18 :
  let g := Nat.gcd 60 190
  g = 18 → 190 % 18 = 10 := by
sorry

end remainder_of_190_div_18_l567_56770


namespace work_completion_time_l567_56791

/-- The number of days A takes to complete the work -/
def a_days : ℝ := 12

/-- B's efficiency compared to A -/
def b_efficiency : ℝ := 1.2

/-- The number of days B takes to complete the work -/
def b_days : ℝ := 10

theorem work_completion_time :
  a_days * b_efficiency = b_days := by sorry

end work_completion_time_l567_56791


namespace orchestra_members_count_l567_56760

theorem orchestra_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 4 ∧ 
  n % 9 = 6 ∧
  n = 212 := by sorry

end orchestra_members_count_l567_56760


namespace total_license_plates_l567_56776

/-- The number of vowels in the license plate system -/
def num_vowels : ℕ := 8

/-- The number of consonants in the license plate system -/
def num_consonants : ℕ := 26 - num_vowels

/-- The number of even digits (0, 2, 4, 6, 8) -/
def num_even_digits : ℕ := 5

/-- The structure of a license plate: consonant, vowel, consonant, even digit, even digit -/
def license_plate_structure := 
  num_consonants * num_vowels * num_consonants * num_even_digits * num_even_digits

/-- The total number of possible license plates -/
theorem total_license_plates : license_plate_structure = 25920 := by
  sorry

end total_license_plates_l567_56776


namespace greatest_odd_integer_below_sqrt_50_l567_56779

theorem greatest_odd_integer_below_sqrt_50 :
  ∀ x : ℕ, x % 2 = 1 → x^2 < 50 → x ≤ 7 :=
by sorry

end greatest_odd_integer_below_sqrt_50_l567_56779


namespace min_value_complex_expression_l567_56792

theorem min_value_complex_expression (w : ℂ) (h : Complex.abs (w - (3 - I)) = 3) :
  Complex.abs (w + (1 - I))^2 + Complex.abs (w - (7 - 2*I))^2 = 38 := by
  sorry

end min_value_complex_expression_l567_56792


namespace box_volume_increase_l567_56782

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5000)
  (surface_area : 2 * (l * w + l * h + w * h) = 1950)
  (edge_sum : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7198 := by
  sorry

end box_volume_increase_l567_56782


namespace linear_function_quadrants_l567_56706

theorem linear_function_quadrants (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x + b ∧ x > 0 ∧ y > 0) ∧  -- First quadrant
  (∃ x y : ℝ, y = a * x + b ∧ x < 0 ∧ y > 0) ∧  -- Second quadrant
  (∃ x y : ℝ, y = a * x + b ∧ x > 0 ∧ y < 0) →  -- Fourth quadrant
  ¬(∃ x y : ℝ, y = b * x - a ∧ x > 0 ∧ y < 0)   -- Not in fourth quadrant
:= by sorry

end linear_function_quadrants_l567_56706


namespace reema_loan_interest_l567_56704

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal rate time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem reema_loan_interest :
  let principal : ℕ := 1200
  let rate : ℕ := 6
  let time : ℕ := rate
  simple_interest principal rate time = 432 := by
  sorry

end reema_loan_interest_l567_56704


namespace smallest_root_of_equation_l567_56732

theorem smallest_root_of_equation : 
  let eq := fun x : ℝ => 2 * (x - 3 * Real.sqrt 5) * (x - 5 * Real.sqrt 3)
  ∃ (r : ℝ), eq r = 0 ∧ r = 3 * Real.sqrt 5 ∧ ∀ (s : ℝ), eq s = 0 → r ≤ s :=
by sorry

end smallest_root_of_equation_l567_56732


namespace fruit_box_arrangement_l567_56777

-- Define the fruits
inductive Fruit
  | Apple
  | Pear
  | Orange
  | Banana

-- Define a type for box numbers
inductive BoxNumber
  | One
  | Two
  | Three
  | Four

-- Define a function type for box labels
def BoxLabel := BoxNumber → Fruit

-- Define a function type for the actual content of boxes
def BoxContent := BoxNumber → Fruit

-- Define the property that all labels are incorrect
def AllLabelsIncorrect (label : BoxLabel) (content : BoxContent) : Prop :=
  ∀ b : BoxNumber, label b ≠ content b

-- Define the specific labels for each box
def SpecificLabels (label : BoxLabel) : Prop :=
  label BoxNumber.One = Fruit.Orange ∧
  label BoxNumber.Two = Fruit.Pear ∧
  (label BoxNumber.Three = Fruit.Apple ∨ label BoxNumber.Three = Fruit.Pear) ∧
  label BoxNumber.Four = Fruit.Apple

-- Define the conditional statement for Box 3
def Box3Condition (content : BoxContent) : Prop :=
  content BoxNumber.One = Fruit.Banana →
  (content BoxNumber.Three = Fruit.Apple ∨ content BoxNumber.Three = Fruit.Pear)

-- The main theorem
theorem fruit_box_arrangement :
  ∀ (label : BoxLabel) (content : BoxContent),
    AllLabelsIncorrect label content →
    SpecificLabels label →
    ¬Box3Condition content →
    content BoxNumber.One = Fruit.Banana ∧
    content BoxNumber.Two = Fruit.Apple ∧
    content BoxNumber.Three = Fruit.Orange ∧
    content BoxNumber.Four = Fruit.Pear :=
by
  sorry

end fruit_box_arrangement_l567_56777


namespace square_configuration_l567_56796

theorem square_configuration (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  let x : ℝ := (a - Real.sqrt 2) / b
  2 * Real.sqrt 2 * x + x = 1 →
  a + b = 11 := by
sorry

end square_configuration_l567_56796


namespace sin_shift_l567_56759

theorem sin_shift (x : ℝ) : 
  Real.sin (4 * x - π / 3) = Real.sin (4 * (x - π / 12)) := by
  sorry

end sin_shift_l567_56759


namespace perfect_square_divisibility_l567_56768

theorem perfect_square_divisibility (a b : ℕ) 
  (h : (a^2 + b^2 + a) % (a * b) = 0) : 
  ∃ k : ℕ, a = k^2 := by
sorry

end perfect_square_divisibility_l567_56768


namespace binary_1010101_to_decimal_l567_56787

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number 1010101₂ -/
def binary_1010101 : List Nat := [1, 0, 1, 0, 1, 0, 1]

/-- Theorem: The decimal equivalent of 1010101₂ is 85 -/
theorem binary_1010101_to_decimal :
  binary_to_decimal binary_1010101.reverse = 85 := by
  sorry

end binary_1010101_to_decimal_l567_56787


namespace range_of_a_l567_56713

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 2 3, x^2 + 5 > a*x) = False → a ∈ Set.Ici (2 * Real.sqrt 5) := by
  sorry

end range_of_a_l567_56713


namespace triangle_conjugates_l567_56736

/-- Triangle ABC with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Barycentric coordinates -/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ
  h_pos : α > 0 ∧ β > 0 ∧ γ > 0

/-- Isotomically conjugate points -/
def isIsotomicallyConjugate (tri : Triangle) (p q : BarycentricCoord) : Prop :=
  p.α * q.α = p.β * q.β ∧ p.β * q.β = p.γ * q.γ ∧ p.γ * q.γ = p.α * q.α

/-- Isogonally conjugate points -/
def isIsogonallyConjugate (tri : Triangle) (p q : BarycentricCoord) : Prop :=
  p.α * q.α = tri.a^2 ∧ p.β * q.β = tri.b^2 ∧ p.γ * q.γ = tri.c^2

/-- Main theorem -/
theorem triangle_conjugates (tri : Triangle) (p : BarycentricCoord) :
  let q₁ : BarycentricCoord := ⟨p.α⁻¹, p.β⁻¹, p.γ⁻¹, sorry⟩
  let q₂ : BarycentricCoord := ⟨tri.a^2 / p.α, tri.b^2 / p.β, tri.c^2 / p.γ, sorry⟩
  isIsotomicallyConjugate tri p q₁ ∧ isIsogonallyConjugate tri p q₂ := by
  sorry

end triangle_conjugates_l567_56736


namespace computer_price_increase_l567_56788

theorem computer_price_increase (y : ℝ) (h1 : 1.30 * y = 351) (h2 : 2 * y = 540) :
  2 * y = 540 := by
  sorry

end computer_price_increase_l567_56788


namespace class_average_problem_l567_56750

theorem class_average_problem (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) :
  total_students = 20 →
  excluded_students = 2 →
  excluded_avg = 45 →
  remaining_avg = 95 →
  (total_students * (total_students - excluded_students) * remaining_avg + 
   excluded_students * excluded_avg) / total_students = 90 := by
  sorry

end class_average_problem_l567_56750


namespace stepped_design_reduces_blind_spots_l567_56727

/-- Represents a hall design --/
structure HallDesign where
  shape : String
  is_stepped : Bool

/-- Represents the visibility in a hall --/
structure Visibility where
  blind_spots : ℕ

/-- A function that calculates visibility based on hall design --/
def calculate_visibility (design : HallDesign) : Visibility :=
  sorry

/-- The theorem stating that a stepped design reduces blind spots --/
theorem stepped_design_reduces_blind_spots 
  (flat_design stepped_design : HallDesign)
  (h1 : flat_design.shape = "flat")
  (h2 : flat_design.is_stepped = false)
  (h3 : stepped_design.shape = "stepped")
  (h4 : stepped_design.is_stepped = true) :
  (calculate_visibility stepped_design).blind_spots < (calculate_visibility flat_design).blind_spots :=
sorry

end stepped_design_reduces_blind_spots_l567_56727


namespace worker_original_wage_l567_56754

/-- Calculates the worker's original daily wage given the conditions of the problem -/
def calculate_original_wage (increase_percentage : ℚ) (final_take_home : ℚ) 
  (tax_rate : ℚ) (fixed_deduction : ℚ) : ℚ :=
  let increased_wage := (1 + increase_percentage) * (final_take_home + fixed_deduction) / (1 - tax_rate)
  increased_wage / (1 + increase_percentage)

/-- Theorem stating that the worker's original daily wage is $37.50 -/
theorem worker_original_wage :
  calculate_original_wage (1/2) 42 (1/5) 3 = 75/2 :=
sorry

end worker_original_wage_l567_56754


namespace candy_division_l567_56719

theorem candy_division (total_candies : ℕ) (num_groups : ℕ) (candies_per_group : ℕ) 
  (h1 : total_candies = 30)
  (h2 : num_groups = 10)
  (h3 : candies_per_group = total_candies / num_groups) :
  candies_per_group = 3 := by
  sorry

end candy_division_l567_56719


namespace last_k_digits_power_l567_56705

theorem last_k_digits_power (k n : ℕ) (A B : ℤ) 
  (h : A ≡ B [ZMOD 10^k]) : 
  A^n ≡ B^n [ZMOD 10^k] := by sorry

end last_k_digits_power_l567_56705


namespace min_value_problem_equality_condition_l567_56773

theorem min_value_problem (x : ℝ) (h : x > 0) : 3 * x + 4 / x ≥ 4 * Real.sqrt 3 := by
  sorry

theorem equality_condition : ∃ x : ℝ, x > 0 ∧ 3 * x + 4 / x = 4 * Real.sqrt 3 := by
  sorry

end min_value_problem_equality_condition_l567_56773


namespace units_digit_of_7_pow_5_l567_56753

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_7_pow_5 : unitsDigit (7^5) = 7 := by
  sorry

end units_digit_of_7_pow_5_l567_56753


namespace acme_profit_l567_56751

/-- Calculates the profit for a horseshoe manufacturing company. -/
def calculate_profit (initial_outlay : ℝ) (cost_per_set : ℝ) (selling_price : ℝ) (num_sets : ℕ) : ℝ :=
  let revenue := selling_price * num_sets
  let total_cost := initial_outlay + cost_per_set * num_sets
  revenue - total_cost

/-- Proves that the profit for Acme's horseshoe manufacturing is $15,337.50 -/
theorem acme_profit :
  calculate_profit 12450 20.75 50 950 = 15337.50 := by
  sorry

end acme_profit_l567_56751


namespace stratified_sampling_male_athletes_l567_56738

/-- Represents the total number of athletes -/
def total_athletes : ℕ := 30

/-- Represents the number of male athletes -/
def male_athletes : ℕ := 20

/-- Represents the number of female athletes -/
def female_athletes : ℕ := 10

/-- Represents the sample size -/
def sample_size : ℕ := 6

/-- Represents the number of male athletes to be sampled -/
def male_sample_size : ℚ := (male_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ)

theorem stratified_sampling_male_athletes :
  male_sample_size = 4 := by sorry

end stratified_sampling_male_athletes_l567_56738


namespace fermat_coprime_and_infinite_primes_l567_56726

def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_coprime_and_infinite_primes :
  (∀ n m : ℕ, n ≠ m → Nat.gcd (fermat n) (fermat m) = 1) ∧
  (¬ ∃ N : ℕ, ∀ p : ℕ, Prime p → p ≤ N) :=
sorry

end fermat_coprime_and_infinite_primes_l567_56726


namespace jace_road_trip_distance_l567_56741

/-- Represents a driving segment with speed in miles per hour and duration in hours -/
structure DrivingSegment where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance covered given a list of driving segments -/
def totalDistance (segments : List DrivingSegment) : ℝ :=
  segments.foldl (fun acc segment => acc + segment.speed * segment.duration) 0

/-- Jace's road trip theorem -/
theorem jace_road_trip_distance :
  let segments : List DrivingSegment := [
    { speed := 50, duration := 3 },
    { speed := 65, duration := 4.5 },
    { speed := 60, duration := 2.75 },
    { speed := 75, duration := 1.8333 },
    { speed := 55, duration := 2.6667 }
  ]
  ∃ ε > 0, |totalDistance segments - 891.67| < ε :=
by sorry

end jace_road_trip_distance_l567_56741


namespace tetrahedron_subdivision_existence_l567_56775

theorem tetrahedron_subdivision_existence : ∃ k : ℕ, (1 / 2 : ℝ) ^ k < (1 / 100 : ℝ) := by
  sorry

end tetrahedron_subdivision_existence_l567_56775


namespace intersection_of_A_and_B_l567_56781

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l567_56781


namespace least_number_to_add_l567_56786

theorem least_number_to_add (n : ℕ) : 
  (∀ m : ℕ, m < 7 → ¬((1789 + m) % 6 = 0 ∧ (1789 + m) % 4 = 0 ∧ (1789 + m) % 3 = 0)) ∧
  ((1789 + 7) % 6 = 0 ∧ (1789 + 7) % 4 = 0 ∧ (1789 + 7) % 3 = 0) :=
by sorry

end least_number_to_add_l567_56786


namespace inequality_proof_l567_56729

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b - c) * (b + 1/c - a) + (b + 1/c - a) * (c + 1/a - b) + (c + 1/a - b) * (a + 1/b - c) ≥ 3 := by
  sorry

end inequality_proof_l567_56729


namespace permutations_of_red_l567_56722

-- Define the number of letters in 'red'
def n : ℕ := 3

-- Theorem to prove
theorem permutations_of_red : Nat.factorial n = 6 := by
  sorry

end permutations_of_red_l567_56722


namespace segment_sum_bound_l567_56757

/-- A convex polygon in a 2D plane. -/
structure ConvexPolygon where
  -- We don't need to define the structure completely, just declare it
  area : ℝ

/-- A set of parallel lines in a 2D plane. -/
structure ParallelLines where
  -- Again, we don't need to define this completely
  count : ℕ
  spacing : ℝ

/-- The sum of lengths of segments cut by a polygon on parallel lines. -/
def sumOfSegments (polygon : ConvexPolygon) (lines : ParallelLines) : ℝ :=
  sorry -- Definition not provided, just declared

/-- Theorem statement -/
theorem segment_sum_bound
  (polygon : ConvexPolygon)
  (lines : ParallelLines)
  (h_area : polygon.area = 9)
  (h_lines : lines.count = 9)
  (h_spacing : lines.spacing = 1) :
  sumOfSegments polygon lines ≤ 10 :=
sorry

end segment_sum_bound_l567_56757


namespace point_on_line_l567_56799

/-- A point (x, y) lies on a line passing through two points (x₁, y₁) and (x₂, y₂) if it satisfies the equation of the line. -/
def lies_on_line (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y - y₁ = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁)

/-- The point (0,3) lies on the line passing through (-2,1) and (2,5). -/
theorem point_on_line : lies_on_line 0 3 (-2) 1 2 5 := by
  sorry

end point_on_line_l567_56799


namespace no_quadratic_function_exists_l567_56721

theorem no_quadratic_function_exists : 
  ¬ ∃ (b c : ℝ), 
    ((-4)^2 + b*(-4) + c = 1) ∧ 
    (∀ x : ℝ, 6*x ≤ 3*x^2 + 3 ∧ 3*x^2 + 3 ≤ x^2 + b*x + c) :=
by sorry

end no_quadratic_function_exists_l567_56721


namespace coffee_blend_price_l567_56780

/-- Given two coffee blends, this theorem proves the price of the second blend
    given the conditions of the problem. -/
theorem coffee_blend_price
  (price_blend1 : ℝ)
  (total_weight : ℝ)
  (total_price_per_pound : ℝ)
  (weight_blend2 : ℝ)
  (h1 : price_blend1 = 9)
  (h2 : total_weight = 20)
  (h3 : total_price_per_pound = 8.4)
  (h4 : weight_blend2 = 12)
  : ∃ (price_blend2 : ℝ),
    price_blend2 * weight_blend2 + price_blend1 * (total_weight - weight_blend2) =
    total_price_per_pound * total_weight ∧
    price_blend2 = 8 :=
by sorry

end coffee_blend_price_l567_56780


namespace cosine_sine_sum_l567_56778

theorem cosine_sine_sum (α : ℝ) : 
  (Real.cos (2 * α)) / (Real.sin (α - π/4)) = -Real.sqrt 2 / 2 → 
  Real.cos α + Real.sin α = 1/2 := by
sorry

end cosine_sine_sum_l567_56778


namespace prime_power_sum_l567_56767

theorem prime_power_sum (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → 2^p + 3^p = a^n → n = 1 := by
  sorry

end prime_power_sum_l567_56767


namespace f_properties_l567_56702

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

-- Main theorem
theorem f_properties (a : ℝ) (h : a > 1) :
  -- 1. Explicit formula for f
  (∀ x, f a x = (a / (a^2 - 1)) * (a^x - a^(-x))) ∧
  -- 2. f is odd and increasing
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x y, x < y → f a x < f a y) ∧
  -- 3. Range of m
  (∀ m, (∀ x ∈ Set.Ioo (-1) 1, f a (1 - m) + f a (1 - m^2) < 0) →
    1 < m ∧ m < Real.sqrt 2) :=
by sorry

end f_properties_l567_56702


namespace complex_equation_solution_l567_56720

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ x : ℂ, (3 - 2 * i * x = 6 + i * x) ∧ (x = i) := by
  sorry

end complex_equation_solution_l567_56720


namespace smallest_union_size_l567_56795

theorem smallest_union_size (A B : Finset ℕ) : 
  Finset.card A = 30 → 
  Finset.card B = 20 → 
  Finset.card (A ∩ B) ≥ 10 → 
  Finset.card (A ∪ B) ≥ 40 ∧ 
  ∃ (C D : Finset ℕ), Finset.card C = 30 ∧ 
                      Finset.card D = 20 ∧ 
                      Finset.card (C ∩ D) ≥ 10 ∧ 
                      Finset.card (C ∪ D) = 40 :=
by sorry

end smallest_union_size_l567_56795


namespace conference_handshakes_l567_56714

/-- The number of handshakes in a conference with multiple companies --/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_participants := num_companies * reps_per_company
  let handshakes_per_person := total_participants - reps_per_company
  (total_participants * handshakes_per_person) / 2

/-- Theorem: In a conference with 3 companies, each having 5 representatives,
    where every person shakes hands once with every person except those from their own company,
    the total number of handshakes is 75. --/
theorem conference_handshakes :
  number_of_handshakes 3 5 = 75 := by
  sorry

end conference_handshakes_l567_56714


namespace division_mistake_remainder_l567_56728

theorem division_mistake_remainder (d q r : ℕ) (h1 : d > 0) (h2 : 472 = d * q + r) (h3 : 427 = d * (q - 5) + r) : r = 4 := by
  sorry

end division_mistake_remainder_l567_56728


namespace radio_sale_profit_percentage_l567_56784

/-- Represents the problem of calculating profit percentage for a radio sale --/
theorem radio_sale_profit_percentage 
  (original_cost_usd : ℝ) 
  (exchange_rate : ℝ) 
  (discount_rate : ℝ) 
  (tax_rate : ℝ) 
  (final_price : ℝ) 
  (h1 : original_cost_usd = 110)
  (h2 : exchange_rate = 30)
  (h3 : discount_rate = 0.15)
  (h4 : tax_rate = 0.12)
  (h5 : final_price = 4830) :
  let original_cost_inr : ℝ := original_cost_usd * exchange_rate
  let selling_price_before_tax : ℝ := final_price / (1 + tax_rate)
  let profit : ℝ := selling_price_before_tax - original_cost_inr
  let profit_percentage : ℝ := (profit / original_cost_inr) * 100
  ∃ (ε : ℝ), abs (profit_percentage - 30.68) < ε ∧ ε > 0 := by
  sorry


end radio_sale_profit_percentage_l567_56784


namespace a_value_in_set_l567_56766

theorem a_value_in_set (A : Set ℝ) (a : ℝ) (h1 : A = {0, a, a^2})
  (h2 : 1 ∈ A) : a = -1 := by
  sorry

end a_value_in_set_l567_56766


namespace arithmetic_sequence_sum_base_6_l567_56707

def base_6_to_10 (n : ℕ) : ℕ := n

def base_10_to_6 (n : ℕ) : ℕ := n

def arithmetic_sum (a₁ aₙ n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_base_6 (a₁ aₙ d : ℕ) 
  (h₁ : a₁ = base_6_to_10 5)
  (h₂ : aₙ = base_6_to_10 31)
  (h₃ : d = 2)
  (h₄ : aₙ = a₁ + (n - 1) * d) :
  base_10_to_6 (arithmetic_sum a₁ aₙ ((aₙ - a₁) / d + 1)) = 240 := by
  sorry

end arithmetic_sequence_sum_base_6_l567_56707


namespace concentric_circles_ratio_l567_56710

theorem concentric_circles_ratio (a b : ℝ) (h : a > 0) (k : b > 0) :
  (π * b^2 - π * a^2 = 4 * π * a^2) → (a / b = 1 / Real.sqrt 5) := by
  sorry

end concentric_circles_ratio_l567_56710


namespace equal_cost_at_four_students_agency_a_cheaper_for_ten_students_l567_56709

/-- Represents a travel agency with a pricing strategy -/
structure Agency where
  teacherDiscount : ℝ  -- Discount for the teacher (0 means full price)
  studentDiscount : ℝ  -- Discount for students

/-- Calculates the total cost for a given number of students -/
def totalCost (a : Agency) (numStudents : ℕ) (fullPrice : ℝ) : ℝ :=
  fullPrice * (1 - a.teacherDiscount) + numStudents * fullPrice * (1 - a.studentDiscount)

/-- The full price of a ticket -/
def fullPrice : ℝ := 240

/-- Agency A's pricing strategy -/
def agencyA : Agency := ⟨0, 0.5⟩

/-- Agency B's pricing strategy -/
def agencyB : Agency := ⟨0.4, 0.4⟩

theorem equal_cost_at_four_students :
  ∃ n : ℕ, n = 4 ∧ totalCost agencyA n fullPrice = totalCost agencyB n fullPrice :=
sorry

theorem agency_a_cheaper_for_ten_students :
  totalCost agencyA 10 fullPrice < totalCost agencyB 10 fullPrice :=
sorry

end equal_cost_at_four_students_agency_a_cheaper_for_ten_students_l567_56709


namespace product_sum_fractions_l567_56755

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end product_sum_fractions_l567_56755


namespace minimum_age_vasily_l567_56748

theorem minimum_age_vasily (n : ℕ) (h_n : n = 64) :
  ∃ (V F : ℕ),
    V = F + 2 ∧
    F ≥ 5 ∧
    (∀ k : ℕ, k ≥ F → Nat.choose n k > Nat.choose n (k + 2)) ∧
    (∀ V' F' : ℕ, V' = F' + 2 → F' ≥ 5 → 
      (∀ k : ℕ, k ≥ F' → Nat.choose n k > Nat.choose n (k + 2)) → V' ≥ V) ∧
    V = 34 := by
  sorry

end minimum_age_vasily_l567_56748


namespace arctan_equation_solution_l567_56745

theorem arctan_equation_solution (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^3) = π / 4 → x = (1 + Real.sqrt 5) / 2 := by
sorry

end arctan_equation_solution_l567_56745


namespace no_solution_exponential_equation_l567_56724

theorem no_solution_exponential_equation :
  ¬ ∃ z : ℝ, (16 : ℝ) ^ (3 * z) = (64 : ℝ) ^ (2 * z + 5) := by
  sorry

end no_solution_exponential_equation_l567_56724


namespace zane_bought_two_shirts_l567_56712

/-- Calculates the number of polo shirts bought given the discount percentage, regular price, and total amount paid. -/
def polo_shirts_bought (discount_percent : ℚ) (regular_price : ℚ) (total_paid : ℚ) : ℚ :=
  let discounted_price := regular_price * (1 - discount_percent)
  total_paid / discounted_price

/-- Proves that Zane bought 2 polo shirts given the specified conditions. -/
theorem zane_bought_two_shirts : 
  polo_shirts_bought (40/100) 50 60 = 2 := by
  sorry

#eval polo_shirts_bought (40/100) 50 60

end zane_bought_two_shirts_l567_56712


namespace min_cuboids_for_cube_l567_56783

def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

theorem min_cuboids_for_cube : 
  let cube_side := Nat.lcm (Nat.lcm cuboid_length cuboid_width) cuboid_height
  let cube_volume := cube_side ^ 3
  let cuboid_volume := cuboid_length * cuboid_width * cuboid_height
  cube_volume / cuboid_volume = 3600 := by
  sorry

end min_cuboids_for_cube_l567_56783


namespace population_difference_is_167_l567_56798

/-- Represents a tribe with male and female populations -/
structure Tribe where
  males : Nat
  females : Nat

/-- The Gaga tribe -/
def gaga : Tribe := ⟨204, 468⟩

/-- The Nana tribe -/
def nana : Tribe := ⟨334, 516⟩

/-- The Dada tribe -/
def dada : Tribe := ⟨427, 458⟩

/-- The Lala tribe -/
def lala : Tribe := ⟨549, 239⟩

/-- The list of all tribes on the couple continent -/
def tribes : List Tribe := [gaga, nana, dada, lala]

/-- The total number of males on the couple continent -/
def totalMales : Nat := (tribes.map (·.males)).sum

/-- The total number of females on the couple continent -/
def totalFemales : Nat := (tribes.map (·.females)).sum

/-- The difference between females and males on the couple continent -/
def populationDifference : Nat := totalFemales - totalMales

theorem population_difference_is_167 : populationDifference = 167 := by
  sorry

end population_difference_is_167_l567_56798


namespace negation_equivalence_l567_56762

theorem negation_equivalence (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end negation_equivalence_l567_56762


namespace cone_volume_l567_56708

def slant_height : ℝ := 5
def base_radius : ℝ := 3

theorem cone_volume : 
  let height := Real.sqrt (slant_height^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * height = 12 * Real.pi := by
  sorry

end cone_volume_l567_56708


namespace line_slope_slope_value_l567_56764

theorem line_slope (x y : ℝ) :
  x + Real.sqrt 3 * y + 1 = 0 → (y = -(Real.sqrt 3 / 3) * x - (1 / Real.sqrt 3)) := by
  sorry

theorem slope_value :
  let m := -(Real.sqrt 3 / 3)
  ∀ x y : ℝ, x + Real.sqrt 3 * y + 1 = 0 → y = m * x - (1 / Real.sqrt 3) := by
  sorry

end line_slope_slope_value_l567_56764


namespace local_minimum_condition_l567_56731

/-- The function f(x) defined as x^3 - 3bx + b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + b

/-- Theorem stating the condition for f(x) to have a local minimum in (0,1) -/
theorem local_minimum_condition (b : ℝ) :
  (∃ x, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (f b) x) ↔ b ∈ Set.Ioo 0 1 := by
  sorry

#check local_minimum_condition

end local_minimum_condition_l567_56731


namespace ellipse_eccentricity_l567_56763

/-- The eccentricity of an ellipse with equation x²/a² + y² = 1, where a > 1 and the major axis length is 4 -/
theorem ellipse_eccentricity (a : ℝ) (h1 : a > 1) (h2 : 2 * a = 4) :
  let c := Real.sqrt (a^2 - 1)
  (c / a) = Real.sqrt 3 / 2 := by
  sorry

end ellipse_eccentricity_l567_56763


namespace tangent_slopes_reciprocal_implies_a_between_one_and_two_l567_56734

open Real

theorem tangent_slopes_reciprocal_implies_a_between_one_and_two 
  (f : ℝ → ℝ) (a : ℝ) (l₁ l₂ : ℝ → ℝ) :
  a ≠ 0 →
  (∀ x, f x = log x - a * (x - 1)) →
  (∃ x₁ y₁, l₁ 0 = 0 ∧ l₁ x₁ = y₁ ∧ y₁ = f x₁) →
  (∃ x₂ y₂, l₂ 0 = 0 ∧ l₂ x₂ = y₂ ∧ y₂ = exp x₂) →
  (∃ k₁ k₂, (∀ x, l₁ x = k₁ * x) ∧ (∀ x, l₂ x = k₂ * x) ∧ k₁ * k₂ = 1) →
  1 < a ∧ a < 2 := by
sorry

end tangent_slopes_reciprocal_implies_a_between_one_and_two_l567_56734


namespace distance_after_two_hours_l567_56730

/-- Alice's walking speed in miles per minute -/
def alice_speed : ℚ := 1 / 20

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Duration of travel in minutes -/
def duration : ℕ := 120

/-- The distance between Alice and Bob after the given duration -/
def distance_between (alice_speed bob_speed : ℚ) (duration : ℕ) : ℚ :=
  (alice_speed * duration) + (bob_speed * duration)

theorem distance_after_two_hours :
  distance_between alice_speed bob_speed duration = 15 := by
  sorry

end distance_after_two_hours_l567_56730


namespace smallest_integer_in_set_l567_56715

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 < 3 * ((7 * n + 21) / 7)) → 
  (∀ k : ℤ, k < n → k + 6 < 3 * ((7 * k + 21) / 7)) →
  n = -1 :=
by sorry

end smallest_integer_in_set_l567_56715


namespace current_monthly_production_l567_56700

/-- Represents the car manufacturing company's production data -/
structure CarProduction where
  targetAnnual : ℕ
  monthlyIncrease : ℕ
  currentMonthly : ℕ

/-- Theorem stating that the current monthly production is 100 cars -/
theorem current_monthly_production (cp : CarProduction) 
  (h1 : cp.targetAnnual = 1800)
  (h2 : cp.monthlyIncrease = 50)
  (h3 : cp.currentMonthly * 12 + cp.monthlyIncrease * 12 = cp.targetAnnual) :
  cp.currentMonthly = 100 := by
  sorry

#check current_monthly_production

end current_monthly_production_l567_56700


namespace function_value_problem_l567_56747

theorem function_value_problem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (2 * x - 1) = 3 * x + a) →
  f 3 = 2 →
  a = -4 :=
by sorry

end function_value_problem_l567_56747


namespace age_problem_l567_56746

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 17 →
  b = 6 :=
by
  sorry

end age_problem_l567_56746


namespace two_std_dev_below_mean_l567_56744

/-- For a normal distribution with mean μ and standard deviation σ,
    the value 2σ below the mean is μ - 2σ. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (hμ : μ = 15.5) (hσ : σ = 1.5) :
  μ - 2 * σ = 12.5 := by
  sorry

#check two_std_dev_below_mean

end two_std_dev_below_mean_l567_56744


namespace probability_reach_bottom_is_one_fifth_l567_56790

/-- Represents a dodecahedron -/
structure Dodecahedron where
  top_vertex : Vertex
  bottom_vertex : Vertex
  middle_vertices : Finset Vertex
  adjacent : Vertex → Finset Vertex

/-- The probability of an ant reaching the bottom vertex in two steps -/
def probability_reach_bottom (d : Dodecahedron) : ℚ :=
  1 / 5

/-- Theorem stating the probability of reaching the bottom vertex in two steps -/
theorem probability_reach_bottom_is_one_fifth (d : Dodecahedron) :
  probability_reach_bottom d = 1 / 5 :=
by sorry

end probability_reach_bottom_is_one_fifth_l567_56790


namespace equation_solution_l567_56785

theorem equation_solution : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ∧ x = 2 := by
  sorry

end equation_solution_l567_56785


namespace total_pencils_l567_56758

/-- Given 11 children, with each child having 2 pencils, the total number of pencils is 22. -/
theorem total_pencils (num_children : Nat) (pencils_per_child : Nat) (total_pencils : Nat) : 
  num_children = 11 → pencils_per_child = 2 → total_pencils = num_children * pencils_per_child →
  total_pencils = 22 := by
  sorry

end total_pencils_l567_56758


namespace new_person_weight_l567_56740

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight new_average : ℝ) :
  n = 10 ∧ 
  replaced_weight = 45 ∧
  new_average = initial_weight + 3 →
  (n * new_average - (n * initial_weight - replaced_weight)) = 75 :=
by sorry

end new_person_weight_l567_56740


namespace equal_volumes_of_modified_cylinders_l567_56765

/-- Theorem: Equal volumes of modified cylinders -/
theorem equal_volumes_of_modified_cylinders (r h : ℝ) (y : ℝ) 
  (hr : r = 5) (hh : h = 4) (hy : y ≠ 0) :
  π * (r + 2)^2 * h = π * r^2 * (h + y) → y = 96 / 25 := by
  sorry

end equal_volumes_of_modified_cylinders_l567_56765


namespace bookstore_problem_l567_56772

theorem bookstore_problem (total_books : ℕ) (unsold_books : ℕ) (customers : ℕ) :
  total_books = 40 →
  unsold_books = 4 →
  customers = 4 →
  (total_books - unsold_books) % customers = 0 →
  (total_books - unsold_books) / customers = 9 :=
by sorry

end bookstore_problem_l567_56772


namespace picture_area_l567_56703

/-- Given a sheet of paper with specified dimensions and margin, calculate the area of the picture --/
theorem picture_area (paper_width paper_length margin : ℝ) 
  (hw : paper_width = 8.5)
  (hl : paper_length = 10)
  (hm : margin = 1.5) : 
  (paper_width - 2 * margin) * (paper_length - 2 * margin) = 38.5 := by
  sorry

#check picture_area

end picture_area_l567_56703


namespace sqrt_12_times_sqrt_75_l567_56723

theorem sqrt_12_times_sqrt_75 : Real.sqrt 12 * Real.sqrt 75 = 30 := by
  sorry

end sqrt_12_times_sqrt_75_l567_56723


namespace box_volume_increase_l567_56711

theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 4320)
  (surface_area : 2 * (l * w + w * h + l * h) = 1704)
  (edge_sum : 4 * (l + w + h) = 208) :
  (l + 4) * (w + 4) * (h + 4) = 8624 := by
  sorry

end box_volume_increase_l567_56711


namespace quadratic_roots_sum_l567_56774

theorem quadratic_roots_sum (a b : ℝ) : 
  (∃ x : ℝ, x^2 + x - 2023 = 0) → 
  (a^2 + a - 2023 = 0) → 
  (b^2 + b - 2023 = 0) → 
  (a ≠ b) →
  a^2 + 2*a + b = 2022 := by
sorry

end quadratic_roots_sum_l567_56774


namespace isosceles_right_triangle_exists_l567_56789

/-- A coloring of an infinite grid using three colors -/
def GridColoring := ℤ × ℤ → Fin 3

/-- An isosceles right triangle on the grid -/
structure IsoscelesRightTriangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  is_right : (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0
  is_isosceles : (b.1 - a.1)^2 + (b.2 - a.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

/-- The main theorem: In any three-coloring of an infinite grid, 
    there exists an isosceles right triangle with vertices of the same color -/
theorem isosceles_right_triangle_exists (coloring : GridColoring) : 
  ∃ t : IsoscelesRightTriangle, 
    coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c :=
sorry

end isosceles_right_triangle_exists_l567_56789


namespace average_increase_l567_56771

theorem average_increase (numbers : Finset ℕ) (sum : ℕ) (added_value : ℕ) :
  numbers.card = 15 →
  sum = numbers.sum id →
  sum / numbers.card = 40 →
  added_value = 10 →
  (sum + numbers.card * added_value) / numbers.card = 50 := by
sorry

end average_increase_l567_56771


namespace thief_reasoning_flaw_l567_56749

/-- Represents the components of the thief's argument --/
inductive ArgumentComponent
  | MajorPremise
  | MinorPremise
  | Conclusion

/-- Represents the thief's ability to open a video recorder --/
def can_open (x : Prop) : Prop := x

/-- Represents the ownership of the video recorder --/
def is_mine (x : Prop) : Prop := x

/-- The thief's argument structure --/
def thief_argument (recorder : Prop) : Prop :=
  (is_mine recorder → can_open recorder) ∧
  (can_open recorder) ∧
  (is_mine recorder)

/-- The flaw in the thief's reasoning --/
def flaw_in_reasoning (component : ArgumentComponent) : Prop :=
  component = ArgumentComponent.MajorPremise

/-- Theorem stating that the flaw in the thief's reasoning is in the major premise --/
theorem thief_reasoning_flaw (recorder : Prop) :
  thief_argument recorder → flaw_in_reasoning ArgumentComponent.MajorPremise :=
by sorry

end thief_reasoning_flaw_l567_56749


namespace number_problem_l567_56717

theorem number_problem (x : ℝ) : 35 - 3 * x = 8 → x = 9 := by
  sorry

end number_problem_l567_56717


namespace ball_distribution_l567_56761

/-- The number of ways to distribute n indistinguishable balls into k boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of ways to distribute balls into three boxes with minimum requirements -/
def distributeWithMinimum (total : ℕ) (min1 min2 min3 : ℕ) : ℕ :=
  distribute (total - min1 - min2 - min3) 3

theorem ball_distribution :
  distributeWithMinimum 20 1 2 3 = 120 := by sorry

end ball_distribution_l567_56761


namespace train_speed_l567_56735

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 360 →
  bridge_length = 140 →
  time = 50 →
  (train_length + bridge_length) / time * 3.6 = 36 := by
  sorry

end train_speed_l567_56735


namespace roots_sum_of_reciprocal_squares_l567_56701

theorem roots_sum_of_reciprocal_squares (r s : ℂ) : 
  (3 * r^2 - 2 * r + 4 = 0) → 
  (3 * s^2 - 2 * s + 4 = 0) → 
  (1 / r^2 + 1 / s^2 = -5 / 4) := by
sorry

end roots_sum_of_reciprocal_squares_l567_56701


namespace largest_share_in_startup_l567_56743

def profit_split (total_profit : ℚ) (ratios : List ℚ) : List ℚ :=
  let sum_ratios := ratios.sum
  ratios.map (λ r => (r / sum_ratios) * total_profit)

theorem largest_share_in_startup (total_profit : ℚ) :
  let ratios : List ℚ := [3, 4, 4, 6, 7]
  let shares := profit_split total_profit ratios
  total_profit = 48000 →
  shares.maximum = some 14000 := by
sorry

end largest_share_in_startup_l567_56743


namespace age_ratio_proof_l567_56797

/-- Given three people a, b, and c with ages satisfying certain conditions,
    prove that the ratio of b's age to c's age is 2:1. -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →                  -- a is two years older than b
  b = 18 →                     -- b is 18 years old
  a + b + c = 47 →             -- total of ages is 47
  ∃ (k : ℕ), b = k * c →       -- b is some times as old as c
  b = 2 * c                    -- ratio of b's age to c's age is 2:1
  := by sorry

end age_ratio_proof_l567_56797


namespace prob_of_specific_sums_is_five_eighteenths_l567_56752

/-- Represents the faces of a die -/
def Die := List Nat

/-- The first die with faces 1, 3, 3, 5, 5, 7 -/
def die1 : Die := [1, 3, 3, 5, 5, 7]

/-- The second die with faces 2, 4, 4, 6, 6, 8 -/
def die2 : Die := [2, 4, 4, 6, 6, 8]

/-- Calculates the probability of a specific sum occurring when rolling two dice -/
def probOfSum (d1 d2 : Die) (sum : Nat) : Rat :=
  sorry

/-- Calculates the probability of the sum being 8, 10, or 12 when rolling the two specified dice -/
def probOfSpecificSums (d1 d2 : Die) : Rat :=
  (probOfSum d1 d2 8) + (probOfSum d1 d2 10) + (probOfSum d1 d2 12)

/-- Theorem stating that the probability of getting a sum of 8, 10, or 12 with the given dice is 5/18 -/
theorem prob_of_specific_sums_is_five_eighteenths :
  probOfSpecificSums die1 die2 = 5 / 18 := by
  sorry

end prob_of_specific_sums_is_five_eighteenths_l567_56752


namespace fencing_cost_l567_56737

/-- The total cost of fencing a rectangular field with a square pond -/
theorem fencing_cost (field_area : ℝ) (outer_fence_cost : ℝ) (pond_fence_cost : ℝ) : 
  field_area = 10800 ∧ 
  outer_fence_cost = 1.5 ∧ 
  pond_fence_cost = 1 → 
  ∃ (short_side long_side pond_side : ℝ),
    short_side * long_side = field_area ∧
    long_side = (4/3) * short_side ∧
    pond_side = (1/6) * short_side ∧
    2 * (short_side + long_side) * outer_fence_cost + 4 * pond_side * pond_fence_cost = 690 :=
by sorry

end fencing_cost_l567_56737


namespace first_positive_term_is_seventh_l567_56725

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem first_positive_term_is_seventh :
  let a₁ := -1
  let d := 1/5
  (∀ k < 7, arithmetic_sequence a₁ d k ≤ 0) ∧
  (arithmetic_sequence a₁ d 7 > 0) :=
by sorry

end first_positive_term_is_seventh_l567_56725


namespace sum_extrema_l567_56794

theorem sum_extrema (x y z w : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0) 
  (h_eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17/2) : 
  (∀ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 → 
    a + b + c + d ≤ 3) ∧ 
  (x + y + z + w ≥ -2 + 5/2 * Real.sqrt 2) ∧
  (∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 ∧ 
    a + b + c + d = 3) ∧
  (∃ a b c d : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ 
    a^2 + b^2 + c^2 + d^2 + a + 2*b + 3*c + 4*d = 17/2 ∧ 
    a + b + c + d = -2 + 5/2 * Real.sqrt 2) :=
by sorry

end sum_extrema_l567_56794


namespace negation_of_universal_statement_l567_56733

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - 3*x > 0) ↔ (∃ x : ℝ, x^3 - 3*x ≤ 0) := by
  sorry

end negation_of_universal_statement_l567_56733


namespace rectangular_solid_depth_l567_56756

theorem rectangular_solid_depth
  (length width surface_area : ℝ)
  (h_length : length = 9)
  (h_width : width = 8)
  (h_surface_area : surface_area = 314)
  (h_formula : surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth) :
  depth = 5 :=
by
  sorry

end rectangular_solid_depth_l567_56756
